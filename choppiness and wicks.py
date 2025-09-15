import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from sqlalchemy import create_engine, text
import warnings

warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Crypto Exchange Analysis",
    page_icon="📊",
    layout="wide"
)

# Clear cache
st.cache_data.clear()

# Database connection
@st.cache_resource
def get_engine():
    return create_engine(
        "postgresql://public_replication:866^FKC4hllk@aws-jp-tk-surf-pg-public.cluster-csteuf9lw8dv.ap-northeast-1.rds.amazonaws.com:5432/replication_report",
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10
    )

engine = get_engine()

# Token mapping
def map_token(surf_token):
    mapping = {"PUMP/USDT": "1000PUMP/USDT"}
    return mapping.get(surf_token, surf_token)

class Analyzer:
    def __init__(self):
        # Only 500 and 1500 points
        self.point_counts = [500, 1500]
        self.data = {}
        self.health = {}
    
    def fetch(self, pairs):
        sg_tz = pytz.timezone('Asia/Singapore')
        now = datetime.now(sg_tz)
        table = f"oracle_price_log_partition_{now.strftime('%Y%m%d')}"
        
        # Fixed 1 hour lookback
        start = (now - timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')
        end = now.strftime('%Y-%m-%d %H:%M:%S')
        
        results = {}
        for pair in pairs:
            results[pair] = {}
            
            # Rollbit query
            rollbit_query = f"""
                SELECT created_at + INTERVAL '8 hour' AS timestamp, final_price AS price
                FROM {table}
                WHERE source_type = 1 AND pair_name = '{map_token(pair)}'
                AND created_at >= '{start}'::timestamp - INTERVAL '8 hour'
                AND created_at <= '{end}'::timestamp - INTERVAL '8 hour'
                ORDER BY timestamp DESC
            """
            
            # Surf query
            surf_query = f"""
                SELECT created_at + INTERVAL '8 hour' AS timestamp, final_price AS price
                FROM {table}
                WHERE source_type = 0 AND pair_name = '{pair}'
                AND created_at >= '{start}'::timestamp - INTERVAL '8 hour'
                AND created_at <= '{end}'::timestamp - INTERVAL '8 hour'
                ORDER BY timestamp DESC
            """
            
            try:
                with engine.connect() as conn:
                    rollbit_df = pd.read_sql_query(text(rollbit_query), conn)
                    surf_df = pd.read_sql_query(text(surf_query), conn)
                    
                    if not rollbit_df.empty:
                        results[pair]['rollbit'] = rollbit_df
                        st.success(f"✓ Rollbit {pair}: {len(rollbit_df)} points")
                    if not surf_df.empty:
                        results[pair]['surf'] = surf_df
                        st.success(f"✓ Surf {pair}: {len(surf_df)} points")
            except Exception as e:
                st.error(f"Error {pair}: {e}")
        
        return results
    
    def calculate_choppiness(self, prices, window=20):
        """Original choppiness calculation that works correctly"""
        diff = prices.diff().abs()
        sum_abs_changes = diff.rolling(window, min_periods=1).sum()
        price_range = prices.rolling(window, min_periods=1).max() - prices.rolling(window, min_periods=1).min()
        
        # Handle zero price range
        if (price_range == 0).any():
            price_range = price_range.replace(0, 1e-10)
        
        epsilon = 1e-10
        choppiness = 100 * sum_abs_changes / (price_range + epsilon)
        
        # Cap extreme values and handle NaN
        choppiness = np.minimum(choppiness, 1000)
        choppiness = choppiness.fillna(200)
        
        return choppiness.mean()
    
    def calc_metrics(self, prices, points):
        """Calculate metrics for specified number of points"""
        if len(prices) < points:
            return None
        
        # Use most recent N points
        sample = prices.iloc[-points:]
        
        # Choppiness (using original calculation)
        window = min(20, points // 10)
        chop = self.calculate_choppiness(sample, window)
        
        # ATR%
        atr = sample.diff().abs().mean()
        atr_pct = (atr / sample.mean()) * 100
        
        # Trend strength
        net_change = abs(sample.iloc[-1] - sample.iloc[-window])
        sum_changes = sample.diff().abs().rolling(window).sum().iloc[-1]
        trend = net_change / sum_changes if sum_changes > 0 else 0
        
        return {
            'choppiness': chop,
            'tick_atr_pct': atr_pct,
            'trend_strength': trend
        }
    
    def calc_health(self, df):
        """Calculate 1-minute candle health metrics"""
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
        ohlc = recent['price'].resample('1T').ohlc().dropna()
        
        if len(ohlc) == 0:
            return None
        
        # Initialize counters
        doji_count = 0
        high_wick_count = 0
        wick_percentages = []
        wick_sizes = []
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
                
                # Count dojis (body < 30% of range)
                if body_pct < 30:
                    doji_count += 1
                
                # Count high wick candles (wick > 50% of range)
                if wick_pct > 50:
                    high_wick_count += 1
                
                wick_percentages.append(wick_pct)
                wick_sizes.append(total_wick)
                
                if body > 0:
                    wick_body_ratios.append(total_wick / body)
                else:
                    wick_body_ratios.append(10.0)
        
        # Calculate metrics
        n = len(ohlc)
        metrics = {
            'doji_count': doji_count,  # Actual count, not percentage
            'doji_pct': (doji_count / n) * 100 if n > 0 else 0,
            'avg_wick_pct': np.mean(wick_percentages) if wick_percentages else 0,
            'avg_wick_size': np.mean(wick_sizes) if wick_sizes else 0,
            'high_wick_count': high_wick_count,  # Integer count
            'max_wick': max(wick_sizes) if wick_sizes else 0,
            'wick_body_ratio': np.mean(wick_body_ratios) if wick_body_ratios else 0
        }
        
        return metrics
    
    def analyze(self, data):
        """Analyze fetched data"""
        for pair, exchanges in data.items():
            key = pair.replace('/', '_')
            
            if key not in self.data:
                self.data[key] = {}
            
            for exchange, df in exchanges.items():
                prices = pd.Series(df['price'].values, dtype=float)
                
                if exchange not in self.data[key]:
                    self.data[key][exchange] = {}
                
                # Calculate metrics for 500 and 1500 points only
                for pc in self.point_counts:
                    metrics = self.calc_metrics(prices, pc)
                    if metrics:
                        self.data[key][exchange][pc] = metrics
                
                # Calculate health metrics
                health = self.calc_health(df)
                if health:
                    if key not in self.health:
                        self.health[key] = {}
                    self.health[key][exchange] = health
    
    def get_comparison_df(self, point_count):
        """Create comparison dataframe for parameter comparison"""
        rows = []
        
        for key, exchanges in self.data.items():
            coin = key.replace('_', '/')
            row = {'Coin': coin}
            
            # Add metrics
            for metric in ['choppiness', 'tick_atr_pct', 'trend_strength']:
                metric_short = metric.replace('tick_atr_pct', 'ATR%').replace('choppiness', 'Chop').replace('trend_strength', 'Trend')
                
                rollbit_val = None
                surf_val = None
                
                if 'rollbit' in exchanges and point_count in exchanges['rollbit']:
                    rollbit_val = exchanges['rollbit'][point_count].get(metric)
                if 'surf' in exchanges and point_count in exchanges['surf']:
                    surf_val = exchanges['surf'][point_count].get(metric)
                
                row[f'{metric_short} ROLLBIT'] = rollbit_val
                row[f'{metric_short} SURF'] = surf_val
                
                if rollbit_val is not None and surf_val is not None:
                    row[f'{metric_short} Diff'] = surf_val - rollbit_val
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def get_health_df(self):
        """Create health metrics dataframe with difference columns"""
        rows = []
        
        for key in sorted(self.health.keys()):
            coin = key.replace('_', '/')
            
            # Get data for both exchanges
            rollbit_data = self.health[key].get('rollbit', {})
            surf_data = self.health[key].get('surf', {})
            
            # Rollbit row
            rollbit_row = {
                'Coin': coin,
                'Exchange': 'Rollbit',
                'Doji Count': int(rollbit_data.get('doji_count', 0)),
                'Doji %': round(rollbit_data.get('doji_pct', 0), 2),
                'Avg Wick %': round(rollbit_data.get('avg_wick_pct', 0), 2),
                'Avg Wick Size': round(rollbit_data.get('avg_wick_size', 0), 6),
                'High Wick Count': int(rollbit_data.get('high_wick_count', 0)),
                'Max Wick': round(rollbit_data.get('max_wick', 0), 6),
                'Wick/Body Ratio': round(rollbit_data.get('wick_body_ratio', 0), 2),
                'Wick Count Diff': None,  # Will be filled in difference row
                'Doji Count Diff': None   # Will be filled in difference row
            }
            
            # Surf row
            surf_row = {
                'Coin': coin,
                'Exchange': 'Surf',
                'Doji Count': int(surf_data.get('doji_count', 0)),
                'Doji %': round(surf_data.get('doji_pct', 0), 2),
                'Avg Wick %': round(surf_data.get('avg_wick_pct', 0), 2),
                'Avg Wick Size': round(surf_data.get('avg_wick_size', 0), 6),
                'High Wick Count': int(surf_data.get('high_wick_count', 0)),
                'Max Wick': round(surf_data.get('max_wick', 0), 6),
                'Wick/Body Ratio': round(surf_data.get('wick_body_ratio', 0), 2),
                'Wick Count Diff': None,
                'Doji Count Diff': None
            }
            
            # Calculate differences (Surf - Rollbit)
            if rollbit_data and surf_data:
                wick_diff = int(surf_data.get('high_wick_count', 0)) - int(rollbit_data.get('high_wick_count', 0))
                doji_diff = int(surf_data.get('doji_count', 0)) - int(rollbit_data.get('doji_count', 0))
                
                # Add differences to both rows for visibility
                rollbit_row['Wick Count Diff'] = wick_diff
                rollbit_row['Doji Count Diff'] = doji_diff
                surf_row['Wick Count Diff'] = wick_diff
                surf_row['Doji Count Diff'] = doji_diff
            
            rows.append(rollbit_row)
            rows.append(surf_row)
        
        return pd.DataFrame(rows) if rows else None

# Main UI
st.title("Crypto Exchange Analysis Dashboard")
st.write(f"Time: {datetime.now(pytz.timezone('Asia/Singapore')).strftime('%Y-%m-%d %H:%M:%S')} SGT")

# Two tabs only
tab1, tab2 = st.tabs(["Parameter Comparison", "Coin Health (1-min Wicks)"])

# Sidebar
with st.sidebar:
    st.header("Settings")
    
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
    
    # Three buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Major"):
            st.session_state.selected = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT"]
    with col2:
        if st.button("All"):
            st.session_state.selected = all_pairs
    with col3:
        if st.button("Clear"):
            st.session_state.selected = []
    
    if 'selected' not in st.session_state:
        st.session_state.selected = ["BTC/USDT", "ETH/USDT"]
    
    selected = st.multiselect("Select Pairs", all_pairs, default=st.session_state.selected)
    
    st.info("Fixed 1 hour data lookback")
    st.info("Point counts: 500 and 1500 only")
    
    analyze = st.button("🔍 Analyze", type="primary", use_container_width=True)

# Analysis
if analyze and selected:
    analyzer = Analyzer()
    
    with st.spinner("Analyzing..."):
        data = analyzer.fetch(selected)
        analyzer.analyze(data)
        
        # Tab 1: Parameter Comparison
        with tab1:
            st.header("Parameter Comparison")
            st.write("Showing Choppiness, ATR%, and Trend Strength for 500 and 1500 points")
            
            for pc in analyzer.point_counts:
                with st.expander(f"{pc} Points", expanded=(pc==500)):
                    df = analyzer.get_comparison_df(pc)
                    if not df.empty:
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.info("No data available")
        
        # Tab 2: Coin Health
        with tab2:
            st.header("Coin Health Analysis")
            st.write("1-minute candles over last 10 minutes")
            st.info("Wick Count Diff and Doji Count Diff show (Surf - Rollbit)")
            
            df = analyzer.get_health_df()
            
            if df is not None:
                # Style function
                def style_row(row):
                    if row['Exchange'] == 'Rollbit':
                        return ['background-color: #e3f2fd'] * len(row)
                    return ['background-color: #fff3e0'] * len(row)
                
                styled = df.style.apply(style_row, axis=1)
                
                st.dataframe(
                    styled,
                    use_container_width=True,
                    height=800,
                    column_config={
                        "Doji Count": st.column_config.NumberColumn(format="%d"),
                        "High Wick Count": st.column_config.NumberColumn(format="%d"),
                        "Wick Count Diff": st.column_config.NumberColumn(format="%d"),
                        "Doji Count Diff": st.column_config.NumberColumn(format="%d"),
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