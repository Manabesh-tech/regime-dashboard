import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz
from sqlalchemy import create_engine, text
from contextlib import contextmanager
import warnings

warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Crypto Exchange Analysis",
    page_icon="📊",
    layout="wide"
)

# Database setup
@st.cache_resource
def init_database():
    """Initialize database connection"""
    connection_string = "postgresql://public_replication:866^FKC4hllk@aws-jp-tk-surf-pg-public.cluster-csteuf9lw8dv.ap-northeast-1.rds.amazonaws.com:5432/replication_report"
    engine = create_engine(connection_string, pool_pre_ping=True)
    return engine

engine = init_database()

# Token mapping
SURF_TO_ROLLBIT = {
    "PUMP/USDT": "1000PUMP/USDT"
}

def map_token(surf_token):
    return SURF_TO_ROLLBIT.get(surf_token, surf_token)

# Main analyzer class
class CryptoAnalyzer:
    def __init__(self):
        self.metrics = ['choppiness', 'tick_atr_pct', 'trend_strength']
        self.point_counts = [500, 1500, 2500, 5000]
        self.data = {}
        self.health_data = {}
    
    def fetch_data(self, pairs, hours=1):
        """Fetch price data from database"""
        singapore_tz = pytz.timezone('Asia/Singapore')
        now = datetime.now(singapore_tz)
        
        # Get table name
        table_name = f"oracle_price_log_partition_{now.strftime('%Y%m%d')}"
        
        # Time range
        end_time = now
        start_time = now - timedelta(hours=hours)
        
        results = {}
        for pair in pairs:
            results[pair] = {}
            
            # Fetch Rollbit data
            rollbit_pair = map_token(pair)
            rollbit_query = f"""
                SELECT 
                    created_at + INTERVAL '8 hour' AS timestamp,
                    final_price AS price
                FROM {table_name}
                WHERE source_type = 1
                    AND pair_name = '{rollbit_pair}'
                    AND created_at >= '{start_time}'::timestamp - INTERVAL '8 hour'
                    AND created_at <= '{end_time}'::timestamp - INTERVAL '8 hour'
                ORDER BY timestamp DESC
            """
            
            # Fetch Surf data
            surf_query = f"""
                SELECT 
                    created_at + INTERVAL '8 hour' AS timestamp,
                    final_price AS price
                FROM {table_name}
                WHERE source_type = 0
                    AND pair_name = '{pair}'
                    AND created_at >= '{start_time}'::timestamp - INTERVAL '8 hour'
                    AND created_at <= '{end_time}'::timestamp - INTERVAL '8 hour'
                ORDER BY timestamp DESC
            """
            
            try:
                with engine.connect() as conn:
                    rollbit_df = pd.read_sql_query(text(rollbit_query), conn)
                    surf_df = pd.read_sql_query(text(surf_query), conn)
                    
                    if not rollbit_df.empty:
                        results[pair]['rollbit'] = rollbit_df
                    if not surf_df.empty:
                        results[pair]['surf'] = surf_df
                        
            except Exception as e:
                st.error(f"Error fetching {pair}: {e}")
        
        return results
    
    def calculate_choppiness(self, prices, window=20):
        """Calculate choppiness index"""
        if len(prices) < window:
            return None
        
        # Rolling calculations
        high_roll = prices.rolling(window).max()
        low_roll = prices.rolling(window).min()
        true_range = high_roll - low_roll
        
        # Sum of absolute changes
        abs_changes = prices.diff().abs()
        sum_changes = abs_changes.rolling(window).sum()
        
        # Choppiness calculation
        choppiness = 100 * (sum_changes / true_range)
        choppiness = choppiness.replace([np.inf, -np.inf], np.nan)
        
        return choppiness.mean()
    
    def calculate_metrics(self, prices, points):
        """Calculate trading metrics for given number of points"""
        if len(prices) < points:
            return None
        
        # Use most recent N points
        sample = prices.iloc[-points:]
        
        # Choppiness
        chop = self.calculate_choppiness(sample)
        
        # ATR percentage
        atr = sample.diff().abs().mean()
        atr_pct = (atr / sample.mean()) * 100
        
        # Trend strength
        window = min(20, points // 10)
        net_change = abs(sample.iloc[-1] - sample.iloc[-window])
        sum_changes = sample.diff().abs().rolling(window).sum().iloc[-1]
        trend = net_change / sum_changes if sum_changes > 0 else 0
        
        return {
            'choppiness': chop,
            'tick_atr_pct': atr_pct,
            'trend_strength': trend
        }
    
    def calculate_health_metrics(self, df):
        """Calculate 1-minute candle wick metrics for last 10 minutes"""
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['price'] = pd.to_numeric(df['price'])
        df = df.set_index('timestamp').sort_index()
        
        # Get last 10 minutes
        cutoff = df.index.max() - pd.Timedelta(minutes=10)
        recent = df[df.index >= cutoff]
        
        if len(recent) < 10:
            return None
        
        # Create 1-minute OHLC
        ohlc = recent['price'].resample('1T').ohlc()
        ohlc = ohlc.dropna()
        
        if len(ohlc) == 0:
            return None
        
        # Calculate metrics
        metrics = {
            'doji_pct': 0,
            'avg_wick_pct': 0,
            'avg_wick_size': 0,
            'high_wick_count': 0,
            'max_wick': 0,
            'wick_body_ratio': 0
        }
        
        doji_count = 0
        wick_pcts = []
        wick_sizes = []
        high_wick_count = 0
        wick_body_ratios = []
        
        for _, row in ohlc.iterrows():
            body = abs(row['close'] - row['open'])
            total_range = row['high'] - row['low']
            
            if total_range > 0:
                upper_wick = row['high'] - max(row['open'], row['close'])
                lower_wick = min(row['open'], row['close']) - row['low']
                total_wick = upper_wick + lower_wick
                
                body_pct = (body / total_range) * 100
                wick_pct = (total_wick / total_range) * 100
                
                if body_pct < 30:
                    doji_count += 1
                
                if wick_pct > 50:
                    high_wick_count += 1
                
                wick_pcts.append(wick_pct)
                wick_sizes.append(total_wick)
                
                if body > 0:
                    wick_body_ratios.append(total_wick / body)
                else:
                    wick_body_ratios.append(10.0)
        
        if len(ohlc) > 0:
            metrics['doji_pct'] = (doji_count / len(ohlc)) * 100
            metrics['avg_wick_pct'] = np.mean(wick_pcts) if wick_pcts else 0
            metrics['avg_wick_size'] = np.mean(wick_sizes) if wick_sizes else 0
            metrics['high_wick_count'] = high_wick_count
            metrics['max_wick'] = max(wick_sizes) if wick_sizes else 0
            metrics['wick_body_ratio'] = np.mean(wick_body_ratios) if wick_body_ratios else 0
        
        return metrics
    
    def analyze(self, data):
        """Analyze all data and store results"""
        for pair, exchanges in data.items():
            coin_key = pair.replace('/', '_')
            
            # Standard metrics
            if coin_key not in self.data:
                self.data[coin_key] = {}
            
            for exchange, df in exchanges.items():
                prices = pd.Series(df['price'].values, dtype=float)
                
                if exchange not in self.data[coin_key]:
                    self.data[coin_key][exchange] = {}
                
                # Calculate for each point count
                for pc in self.point_counts:
                    metrics = self.calculate_metrics(prices, pc)
                    if metrics:
                        self.data[coin_key][exchange][pc] = metrics
                
                # Health metrics
                health = self.calculate_health_metrics(df)
                if health:
                    if coin_key not in self.health_data:
                        self.health_data[coin_key] = {}
                    self.health_data[coin_key][exchange] = health
    
    def create_comparison_df(self, point_count):
        """Create comparison dataframe for specific point count"""
        rows = []
        
        for coin_key, exchanges in self.data.items():
            coin_name = coin_key.replace('_', '/')
            row = {'Coin': coin_name}
            
            for metric in self.metrics:
                rollbit_val = None
                surf_val = None
                
                if 'rollbit' in exchanges and point_count in exchanges['rollbit']:
                    rollbit_val = exchanges['rollbit'][point_count].get(metric)
                
                if 'surf' in exchanges and point_count in exchanges['surf']:
                    surf_val = exchanges['surf'][point_count].get(metric)
                
                metric_short = metric.replace('tick_atr_pct', 'ATR%').replace('choppiness', 'Chop').replace('trend_strength', 'Trend')
                
                row[f'{metric_short} ROLLBIT'] = rollbit_val
                row[f'{metric_short} SURF'] = surf_val
                
                if rollbit_val is not None and surf_val is not None:
                    row[f'{metric_short} Diff'] = surf_val - rollbit_val
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def create_health_df(self):
        """Create health metrics dataframe with two rows per coin"""
        rows = []
        
        for coin_key in sorted(self.health_data.keys()):
            coin_name = coin_key.replace('_', '/')
            
            # Rollbit row
            row_r = {
                'Coin': coin_name,
                'Exchange': 'Rollbit',
                'Doji %': None,
                'Avg Wick %': None,
                'Avg Wick Size': None,
                'High Wick Count': None,
                'Max Wick': None,
                'Wick/Body Ratio': None
            }
            
            if 'rollbit' in self.health_data[coin_key]:
                d = self.health_data[coin_key]['rollbit']
                row_r['Doji %'] = round(d['doji_pct'], 2)
                row_r['Avg Wick %'] = round(d['avg_wick_pct'], 2)
                row_r['Avg Wick Size'] = round(d['avg_wick_size'], 6)
                row_r['High Wick Count'] = int(d['high_wick_count'])
                row_r['Max Wick'] = round(d['max_wick'], 6)
                row_r['Wick/Body Ratio'] = round(d['wick_body_ratio'], 2)
            
            # Surf row
            row_s = {
                'Coin': coin_name,
                'Exchange': 'Surf',
                'Doji %': None,
                'Avg Wick %': None,
                'Avg Wick Size': None,
                'High Wick Count': None,
                'Max Wick': None,
                'Wick/Body Ratio': None
            }
            
            if 'surf' in self.health_data[coin_key]:
                d = self.health_data[coin_key]['surf']
                row_s['Doji %'] = round(d['doji_pct'], 2)
                row_s['Avg Wick %'] = round(d['avg_wick_pct'], 2)
                row_s['Avg Wick Size'] = round(d['avg_wick_size'], 6)
                row_s['High Wick Count'] = int(d['high_wick_count'])
                row_s['Max Wick'] = round(d['max_wick'], 6)
                row_s['Wick/Body Ratio'] = round(d['wick_body_ratio'], 2)
            
            rows.append(row_r)
            rows.append(row_s)
        
        return pd.DataFrame(rows) if rows else None

# UI
st.title("Crypto Exchange Analysis Dashboard")

# Time display
sg_tz = pytz.timezone('Asia/Singapore')
st.write(f"Current Time: {datetime.now(sg_tz).strftime('%Y-%m-%d %H:%M:%S')} SGT")

# Tabs
tab1, tab2, tab3 = st.tabs(["Parameter Comparison", "Rankings & Analysis", "Coin Health (1-min Wicks)"])

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    # Get pairs from database
    @st.cache_data
    def get_pairs():
        try:
            query = "SELECT DISTINCT pair_name FROM trade_pool_pairs WHERE status IN (1,2) ORDER BY pair_name"
            with engine.connect() as conn:
                result = conn.execute(text(query))
                return [row[0] for row in result]
        except:
            return ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    
    all_pairs = get_pairs()
    
    # Quick select buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Major Coins"):
            st.session_state.pairs = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT"]
    with col2:
        if st.button("Clear"):
            st.session_state.pairs = []
    
    # Pair selection
    if 'pairs' not in st.session_state:
        st.session_state.pairs = ["BTC/USDT", "ETH/USDT"]
    
    selected = st.multiselect(
        "Select Pairs",
        all_pairs,
        default=st.session_state.pairs
    )
    
    st.info("Analysis uses last 1 hour of data")
    
    analyze = st.button("🔍 Analyze", type="primary", use_container_width=True)

# Analysis
if analyze and selected:
    analyzer = CryptoAnalyzer()
    
    with st.spinner("Fetching and analyzing..."):
        # Fetch data
        data = analyzer.fetch_data(selected, hours=1)
        
        # Analyze
        analyzer.analyze(data)
        
        # Tab 1: Parameters
        with tab1:
            st.header("Parameter Comparison")
            
            for pc in analyzer.point_counts:
                with st.expander(f"{pc} Points", expanded=(pc==500)):
                    df = analyzer.create_comparison_df(pc)
                    if not df.empty:
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.info("No data available")
        
        # Tab 2: Rankings
        with tab2:
            st.header("Rankings & Analysis")
            
            tabs = st.tabs([f"{pc} Points" for pc in analyzer.point_counts])
            
            for i, pc in enumerate(analyzer.point_counts):
                with tabs[i]:
                    df = analyzer.create_comparison_df(pc)
                    if not df.empty:
                        st.dataframe(df, use_container_width=True, height=600)
        
        # Tab 3: Health
        with tab3:
            st.header("Coin Health Analysis")
            st.write("Based on 1-minute candles over last 10 minutes")
            
            df = analyzer.create_health_df()
            
            if df is not None:
                # Style function
                def style_row(row):
                    if row['Exchange'] == 'Rollbit':
                        return ['background-color: #e3f2fd'] * len(row)
                    else:
                        return ['background-color: #fff3e0'] * len(row)
                
                styled = df.style.apply(style_row, axis=1)
                
                st.dataframe(
                    styled,
                    use_container_width=True,
                    height=800,
                    column_config={
                        "High Wick Count": st.column_config.NumberColumn(format="%d"),
                        "Doji %": st.column_config.NumberColumn(format="%.2f"),
                        "Avg Wick %": st.column_config.NumberColumn(format="%.2f"),
                        "Avg Wick Size": st.column_config.NumberColumn(format="%.6f"),
                        "Max Wick": st.column_config.NumberColumn(format="%.6f"),
                        "Wick/Body Ratio": st.column_config.NumberColumn(format="%.2f")
                    }
                )
                
                # Download
                csv = df.to_csv(index=False)
                st.download_button("Download CSV", csv, "health_metrics.csv")
            else:
                st.info("No health data available")

elif analyze:
    st.warning("Please select at least one pair")