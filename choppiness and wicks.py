import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from sqlalchemy import create_engine, text
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Crypto Exchange Analysis",
    page_icon="📊",
    layout="wide"
)

# Database
@st.cache_resource
def get_db():
    return create_engine(
        "postgresql://public_replication:866^FKC4hllk@aws-jp-tk-surf-pg-public.cluster-csteuf9lw8dv.ap-northeast-1.rds.amazonaws.com:5432/replication_report",
        pool_pre_ping=True
    )

engine = get_db()

# Token mapping
TOKEN_MAP = {"PUMP/USDT": "1000PUMP/USDT"}

class CryptoAnalyzer:
    def __init__(self):
        self.metrics_data = {}
        self.health_data = {}
        
    def fetch_data(self, pairs):
        """Fetch only enough data for 1500 points analysis"""
        tz = pytz.timezone('Asia/Singapore')
        now = datetime.now(tz)
        table = f"oracle_price_log_partition_{now.strftime('%Y%m%d')}"
        
        # Only fetch ~20 minutes of data (enough for 1500 points + health metrics)
        start = now - timedelta(minutes=20)
        
        results = {}
        for pair in pairs:
            results[pair] = {}
            
            # Get Rollbit token name
            rollbit_pair = TOKEN_MAP.get(pair, pair)
            
            # Build queries with LIMIT to cap data
            rollbit_q = f"""
                SELECT 
                    created_at + INTERVAL '8 hour' AS timestamp,
                    final_price AS price
                FROM {table}
                WHERE source_type = 1 
                    AND pair_name = '{rollbit_pair}'
                    AND created_at >= '{start}'::timestamp - INTERVAL '8 hour'
                ORDER BY timestamp DESC
                LIMIT 1600
            """
            
            surf_q = f"""
                SELECT 
                    created_at + INTERVAL '8 hour' AS timestamp,
                    final_price AS price
                FROM {table}
                WHERE source_type = 0 
                    AND pair_name = '{pair}'
                    AND created_at >= '{start}'::timestamp - INTERVAL '8 hour'
                ORDER BY timestamp DESC
                LIMIT 1600
            """
            
            try:
                with engine.connect() as conn:
                    # Fetch Rollbit
                    rollbit_df = pd.read_sql(text(rollbit_q), conn)
                    if not rollbit_df.empty:
                        results[pair]['rollbit'] = rollbit_df
                        st.success(f"Rollbit {pair}: {len(rollbit_df)} points")
                    
                    # Fetch Surf
                    surf_df = pd.read_sql(text(surf_q), conn)
                    if not surf_df.empty:
                        results[pair]['surf'] = surf_df
                        st.success(f"Surf {pair}: {len(surf_df)} points")
                        
            except Exception as e:
                st.error(f"Error fetching {pair}: {e}")
                
        return results
    
    def calculate_choppiness(self, prices, window=20):
        """Original working choppiness calculation"""
        diff = prices.diff().abs()
        sum_abs = diff.rolling(window, min_periods=1).sum()
        range_roll = prices.rolling(window, min_periods=1).max() - prices.rolling(window, min_periods=1).min()
        
        # Handle zero range
        range_roll = range_roll.replace(0, 1e-10)
        
        chop = 100 * sum_abs / (range_roll + 1e-10)
        chop = np.minimum(chop, 1000)  # Cap at 1000
        chop = chop.fillna(200)  # Default 200
        
        return chop.mean()
    
    def analyze_metrics(self, data):
        """Analyze standard metrics for 500 and 1500 points"""
        for pair, exchanges in data.items():
            coin_key = pair.replace('/', '_')
            self.metrics_data[coin_key] = {}
            
            for exchange, df in exchanges.items():
                prices = pd.Series(df['price'].values, dtype=float)
                self.metrics_data[coin_key][exchange] = {}
                
                # Only 500 and 1500 points
                for points in [500, 1500]:
                    if len(prices) >= points:
                        sample = prices.iloc[-points:]
                        
                        # Choppiness
                        window = min(20, points // 10)
                        chop = self.calculate_choppiness(sample, window)
                        
                        # ATR%
                        atr = sample.diff().abs().mean()
                        atr_pct = (atr / sample.mean()) * 100
                        
                        # Trend
                        net = abs(sample.iloc[-1] - sample.iloc[-window])
                        sum_chg = sample.diff().abs().rolling(window).sum().iloc[-1]
                        trend = net / sum_chg if sum_chg > 0 else 0
                        
                        self.metrics_data[coin_key][exchange][points] = {
                            'choppiness': chop,
                            'atr_pct': atr_pct,
                            'trend': trend
                        }
    
    def analyze_health(self, data):
        """Analyze 1-minute candle health metrics"""
        for pair, exchanges in data.items():
            coin_key = pair.replace('/', '_')
            self.health_data[coin_key] = {}
            
            for exchange, df in exchanges.items():
                df = df.copy()
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['price'] = pd.to_numeric(df['price'])
                df = df.set_index('timestamp').sort_index()
                
                # Last 10 minutes only
                cutoff = df.index.max() - pd.Timedelta(minutes=10)
                recent = df[df.index >= cutoff]
                
                if len(recent) < 10:
                    continue
                
                # Create 1-minute candles
                ohlc = recent['price'].resample('1T').ohlc().dropna()
                
                if len(ohlc) == 0:
                    continue
                
                # Count metrics
                doji_count = 0
                high_wick_count = 0
                wick_pcts = []
                wick_sizes = []
                wick_body_ratios = []
                
                for _, candle in ohlc.iterrows():
                    body = abs(candle['close'] - candle['open'])
                    total_range = candle['high'] - candle['low']
                    
                    if total_range > 0:
                        upper_wick = candle['high'] - max(candle['open'], candle['close'])
                        lower_wick = min(candle['open'], candle['close']) - candle['low']
                        total_wick = upper_wick + lower_wick
                        
                        body_pct = (body / total_range) * 100
                        wick_pct = (total_wick / total_range) * 100
                        
                        # Doji: body < 30%
                        if body_pct < 30:
                            doji_count += 1
                        
                        # High wick: wick > 50%
                        if wick_pct > 50:
                            high_wick_count += 1
                        
                        wick_pcts.append(wick_pct)
                        wick_sizes.append(total_wick)
                        
                        if body > 0:
                            wick_body_ratios.append(total_wick / body)
                        else:
                            wick_body_ratios.append(10.0)
                
                self.health_data[coin_key][exchange] = {
                    'doji_count': doji_count,
                    'high_wick_count': high_wick_count,
                    'avg_wick_pct': np.mean(wick_pcts) if wick_pcts else 0,
                    'avg_wick_size': np.mean(wick_sizes) if wick_sizes else 0,
                    'max_wick': max(wick_sizes) if wick_sizes else 0,
                    'wick_body_ratio': np.mean(wick_body_ratios) if wick_body_ratios else 0
                }
    
    def get_metrics_df(self, points):
        """Create metrics comparison dataframe"""
        rows = []
        for coin_key, exchanges in self.metrics_data.items():
            coin = coin_key.replace('_', '/')
            row = {'Coin': coin}
            
            # Get values
            r_data = exchanges.get('rollbit', {}).get(points, {})
            s_data = exchanges.get('surf', {}).get(points, {})
            
            # Choppiness
            r_chop = r_data.get('choppiness')
            s_chop = s_data.get('choppiness')
            row['Chop ROLLBIT'] = r_chop
            row['Chop SURF'] = s_chop
            if r_chop and s_chop:
                row['Chop Diff'] = s_chop - r_chop
            
            # ATR%
            r_atr = r_data.get('atr_pct')
            s_atr = s_data.get('atr_pct')
            row['ATR% ROLLBIT'] = r_atr
            row['ATR% SURF'] = s_atr
            if r_atr and s_atr:
                row['ATR% Diff'] = s_atr - r_atr
            
            # Trend
            r_trend = r_data.get('trend')
            s_trend = s_data.get('trend')
            row['Trend ROLLBIT'] = r_trend
            row['Trend SURF'] = s_trend
            if r_trend and s_trend:
                row['Trend Diff'] = s_trend - r_trend
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def get_health_df(self):
        """Create health dataframe with difference columns"""
        rows = []
        
        for coin_key in sorted(self.health_data.keys()):
            coin = coin_key.replace('_', '/')
            
            r_data = self.health_data[coin_key].get('rollbit', {})
            s_data = self.health_data[coin_key].get('surf', {})
            
            # Calculate differences
            wick_diff = None
            doji_diff = None
            if r_data and s_data:
                wick_diff = s_data['high_wick_count'] - r_data['high_wick_count']
                doji_diff = s_data['doji_count'] - r_data['doji_count']
            
            # Rollbit row
            r_row = {
                'Coin': coin,
                'Exchange': 'Rollbit',
                'Doji Count': r_data.get('doji_count', 0),
                'High Wick Count': r_data.get('high_wick_count', 0),
                'Avg Wick %': round(r_data.get('avg_wick_pct', 0), 2),
                'Avg Wick Size': round(r_data.get('avg_wick_size', 0), 6),
                'Max Wick': round(r_data.get('max_wick', 0), 6),
                'Wick/Body Ratio': round(r_data.get('wick_body_ratio', 0), 2),
                'Wick Diff': wick_diff,
                'Doji Diff': doji_diff
            }
            
            # Surf row
            s_row = {
                'Coin': coin,
                'Exchange': 'Surf',
                'Doji Count': s_data.get('doji_count', 0),
                'High Wick Count': s_data.get('high_wick_count', 0),
                'Avg Wick %': round(s_data.get('avg_wick_pct', 0), 2),
                'Avg Wick Size': round(s_data.get('avg_wick_size', 0), 6),
                'Max Wick': round(s_data.get('max_wick', 0), 6),
                'Wick/Body Ratio': round(s_data.get('wick_body_ratio', 0), 2),
                'Wick Diff': wick_diff,
                'Doji Diff': doji_diff
            }
            
            rows.append(r_row)
            rows.append(s_row)
        
        return pd.DataFrame(rows) if rows else None

# UI
st.title("Crypto Exchange Analysis")
st.write(f"{datetime.now(pytz.timezone('Asia/Singapore')).strftime('%Y-%m-%d %H:%M:%S')} SGT")

# Two tabs
tab1, tab2 = st.tabs(["Parameter Comparison", "Coin Health"])

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    # Get pairs
    @st.cache_data
    def get_pairs():
        try:
            with engine.connect() as c:
                r = c.execute(text("SELECT DISTINCT pair_name FROM trade_pool_pairs WHERE status IN (1,2) ORDER BY pair_name"))
                return [row[0] for row in r]
        except:
            return ["BTC/USDT", "ETH/USDT"]
    
    pairs = get_pairs()
    
    # Buttons
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Major"):
            st.session_state.sel = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT"]
    with c2:
        if st.button("All"):
            st.session_state.sel = pairs
    with c3:
        if st.button("Clear"):
            st.session_state.sel = []
    
    if 'sel' not in st.session_state:
        st.session_state.sel = ["BTC/USDT", "ETH/USDT"]
    
    selected = st.multiselect("Select", pairs, st.session_state.sel)
    
    st.info("Max 1500 points analyzed")
    
    go = st.button("Analyze", type="primary")

# Run analysis
if go and selected:
    analyzer = CryptoAnalyzer()
    
    with st.spinner("Processing..."):
        # Fetch
        data = analyzer.fetch_data(selected)
        
        # Analyze
        analyzer.analyze_metrics(data)
        analyzer.analyze_health(data)
        
        # Tab 1
        with tab1:
            st.header("Parameters (500 & 1500 points)")
            
            # 500 points
            st.subheader("500 Points")
            df500 = analyzer.get_metrics_df(500)
            if not df500.empty:
                st.dataframe(df500, use_container_width=True)
            
            # 1500 points
            st.subheader("1500 Points")
            df1500 = analyzer.get_metrics_df(1500)
            if not df1500.empty:
                st.dataframe(df1500, use_container_width=True)
        
        # Tab 2
        with tab2:
            st.header("Health (10 x 1-min candles)")
            st.info("Doji: body<30%, High Wick: wick>50%. Diff = Surf - Rollbit")
            
            health_df = analyzer.get_health_df()
            if health_df is not None:
                # Style
                def style(r):
                    if r['Exchange'] == 'Rollbit':
                        return ['background-color: #e3f2fd'] * len(r)
                    return ['background-color: #fff3e0'] * len(r)
                
                st.dataframe(
                    health_df.style.apply(style, axis=1),
                    use_container_width=True,
                    height=800
                )