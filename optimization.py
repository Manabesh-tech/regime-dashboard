import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from sqlalchemy import create_engine, text
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

warnings.filterwarnings('ignore')

# Clear any problematic session state on startup
if 'clear_state' not in st.session_state:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.clear_state = True

st.set_page_config(
    page_title="Order Book N Optimizer",
    page_icon="📊",
    layout="wide"
)

# Database configuration
db_params = {
    'host': 'aws-jp-tk-surf-pg-public.cluster-csteuf9lw8dv.ap-northeast-1.rds.amazonaws.com',
    'port': 5432,
    'database': 'report_dev',
    'user': 'public_rw',
    'password': 'aTJ92^kl04hllk'
}

@st.cache_resource
def get_db():
    return create_engine(
        f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['database']}",
        pool_pre_ping=True,        isolation_level="AUTOCOMMIT",  # 设置自动提交模式

    )

engine = get_db()

def get_all_pairs():
    """Fetch all available pairs from database - no caching to avoid issues"""
    query = """
        SELECT DISTINCT pair_name 
        FROM trade_pool_pairs 
        WHERE status IN (1,2) 
        ORDER BY pair_name
    """
    try:
        with engine.connect() as conn:
            result = pd.read_sql(text(query), conn)
            pairs = result['pair_name'].tolist()
            if pairs:
                return pairs
    except:
        pass
    
    # Fallback to common pairs if query fails
    return ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT", 
            "BNB/USDT", "ADA/USDT", "AVAX/USDT", "WIF/USDT", "DOT/USDT"]

class OrderBookOptimizer:
    def __init__(self):
        self.results = {}
        
    def fetch_order_book_data(self, pair):
        """Fetch real order book data - last 1500 points for choppiness"""
        tz = pytz.timezone('Asia/Singapore')
        now = datetime.now(tz)
        table = f"oracle_order_book_partition_{now.strftime('%Y%m%d')}"
        
        # Fetch last 1500 points (roughly 12.5 minutes at 500ms intervals)
        query = f"""
            SELECT 
                created_at + INTERVAL '8 hour' AS timestamp,
                bids::text,
                asks::text,
                jsonb_array_length(bids::jsonb) as bid_levels,
                jsonb_array_length(asks::jsonb) as ask_levels,
                jsonb_array_length(bids::jsonb) + jsonb_array_length(asks::jsonb) as total_levels
            FROM {table}
            WHERE pair_name = '{pair}'
            ORDER BY created_at DESC
            LIMIT 1500
        """
        
        try:
            with engine.connect() as conn:
                df = pd.read_sql(text(query), conn)
                if not df.empty:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    # Sort by timestamp ascending for proper time series
                    df = df.sort_values('timestamp')
                    return df
                return pd.DataFrame()
        except Exception as e:
            st.error(f"Database error: {e}")
            return pd.DataFrame()
    
    def calculate_weighted_price(self, bids_str, asks_str, n_levels):
        """Calculate weighted mid price using n levels (union of bid/ask prices)"""
        try:
            bids = json.loads(bids_str)
            asks = json.loads(asks_str)
            
            if not bids or not asks:
                return None
            
            # Get best bid and ask for reference
            best_bid = float(bids[0]['p'])
            best_ask = float(asks[0]['p'])
            
            # For n=1, just use top of book
            if n_levels == 1:
                return (best_bid + best_ask) / 2
            
            # Create a union of price levels with proper sorting
            all_levels = []
            
            # Add all bid levels
            for i, bid in enumerate(bids):
                all_levels.append({
                    'price': float(bid['p']),
                    'volume': float(bid['v']),
                    'side': 'bid',
                    'distance': best_ask - float(bid['p'])  # Distance from opposite best
                })
            
            # Add all ask levels  
            for i, ask in enumerate(asks):
                all_levels.append({
                    'price': float(ask['p']),
                    'volume': float(ask['v']),
                    'side': 'ask',
                    'distance': float(ask['p']) - best_bid  # Distance from opposite best
                })
            
            # Sort by distance (closest spreads first)
            all_levels.sort(key=lambda x: x['distance'])
            
            # Take first n levels from the union
            selected_levels = all_levels[:n_levels]
            
            # Calculate weighted average
            total_value = sum(level['price'] * level['volume'] for level in selected_levels)
            total_volume = sum(level['volume'] for level in selected_levels)
            
            if total_volume > 0:
                return total_value / total_volume
            
            return None
            
        except Exception as e:
            return None
    
    def calculate_metrics(self, prices):
        """Calculate choppiness (1500 points) and doji metrics (last 10 min)"""
        if len(prices) < 20:
            return None
        
        # Choppiness on all data (up to 1500 points)
        window = 20
        diff = prices.diff().abs()
        sum_abs = diff.rolling(window, min_periods=1).sum()
        range_roll = prices.rolling(window, min_periods=1).max() - prices.rolling(window, min_periods=1).min()
        range_roll = range_roll.replace(0, 1e-10)
        chop = 100 * sum_abs / range_roll
        chop = np.minimum(chop, 1000)
        choppiness = chop.mean()
        
        # Doji count - only last 10 minutes of data
        doji_count = 0
        
        # Get last 10 minutes (assuming ~500ms intervals = 120 points per minute = 1200 points)
        last_10_min = prices.iloc[-1200:] if len(prices) > 1200 else prices
        
        if len(last_10_min) > 120:
            ohlc = last_10_min.resample('1T').ohlc().dropna()
            
            for _, candle in ohlc.iterrows():
                body = abs(candle['close'] - candle['open'])
                total_range = candle['high'] - candle['low']
                
                if total_range > 0:
                    body_pct = (body / total_range) * 100
                    if body_pct < 30:
                        doji_count += 1
        
        return {
            'choppiness': choppiness,
            'doji_count': doji_count,
            'data_points': len(prices),
            'volatility': (prices.std() / prices.mean()) * 100 if prices.mean() > 0 else 0
        }
    
    def generate_test_levels(self, max_depth):
        """Generate smart test levels based on maximum depth"""
        if max_depth <= 10:
            return list(range(1, max_depth + 1))
        elif max_depth <= 20:
            return [1, 2, 3, 5, 7, 10, 12, 15] + list(range(17, min(max_depth + 1, 21), 2))
        elif max_depth <= 50:
            return [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50][:max_depth]
        else:
            # For deep books, use logarithmic spacing
            levels = [1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50]
            if max_depth > 60:
                levels.extend([60, 70, 80, 90, 100])
            if max_depth > 100:
                levels.extend([120, 150, 180, 200])
            return [l for l in levels if l <= max_depth]
    
    def optimize_n(self, pair, custom_levels=None):
        """Find optimal N value for a specific pair"""
        df = self.fetch_order_book_data(pair)
        
        if df.empty:
            st.error(f"No order book data available for {pair}")
            return pd.DataFrame()
        
        # Get depth statistics
        max_depth = int(df['total_levels'].max())
        min_depth = int(df['total_levels'].min())
        avg_depth = df['total_levels'].mean()
        
        st.info(f"📊 {pair}: {len(df)} snapshots | Max union depth: {max_depth} | Min: {min_depth} | Avg: {avg_depth:.0f}")
        st.caption("Union depth = total unique price levels across bids and asks")
        
        # Determine test levels
        if custom_levels:
            level_range = [n for n in custom_levels if n <= max_depth]
        else:
            level_range = self.generate_test_levels(max_depth)
        
        st.write(f"Testing levels: {level_range}")
        
        results = []
        progress = st.progress(0)
        
        for i, n in enumerate(level_range):
            # Count valid snapshots (with enough total levels from union)
            valid_snapshots = df[df['total_levels'] >= n]
            
            if len(valid_snapshots) < 100:
                continue
            
            # Calculate prices for this N
            prices_list = []
            timestamps = []
            
            for _, row in valid_snapshots.iterrows():
                price = self.calculate_weighted_price(row['bids'], row['asks'], n)
                if price:
                    prices_list.append(price)
                    timestamps.append(row['timestamp'])
            
            if len(prices_list) > 100:
                prices = pd.Series(prices_list, index=pd.DatetimeIndex(timestamps))
                metrics = self.calculate_metrics(prices)
                
                if metrics:
                    chop = metrics['choppiness']
                    
                    # Score calculation
                    if 250 <= chop <= 350:
                        chop_score = 100
                    elif 200 <= chop <= 400:
                        chop_score = 50 - abs(chop - 300) / 2
                    else:
                        chop_score = max(0, 25 - abs(chop - 300) / 10)
                    
                    doji_penalty = metrics['doji_count'] * 10  # 10 points per doji
                    
                    total_score = chop_score - doji_penalty
                    
                    results.append({
                        'N': n,
                        'Choppiness': round(chop, 2),
                        'Doji Count': metrics['doji_count'],
                        'Score': round(total_score, 2),
                        'Data Points': metrics['data_points'],
                        'Volatility %': round(metrics['volatility'], 4),
                        'Status': '✅' if 250 <= chop <= 350 else '⚠️' if 200 <= chop <= 400 else '❌'
                    })
            
            progress.progress((i + 1) / len(level_range))
        
        progress.empty()
        return pd.DataFrame(results)

# Main UI
st.title("🎯 Order Book N-Level Optimizer")
st.markdown("Finding optimal order book depth (N) for choppiness 250-350 with minimum dojis")
st.info("📊 Choppiness: Last 1500 points (500ms intervals) | 🕯️ Doji: Last 10 minutes")

# Get pairs first
all_pairs = get_all_pairs()

# Create two columns for layout
col_left, col_right = st.columns([1, 2])

with col_left:
    st.header("Settings")
    
    # Single pair analysis
    selected_pair = st.selectbox(
        "Select Trading Pair",
        options=all_pairs,
        index=0
    )
    
    test_mode = st.radio(
        "Test Mode",
        ["Auto", "Custom"],
        help="Auto mode selects levels based on available depth"
    )
    
    custom_levels = None
    if test_mode == "Custom":
        custom_input = st.text_input(
            "Custom levels (comma-separated)", 
            "1,2,3,5,10,20,30,50"
        )
        try:
            custom_levels = [int(x.strip()) for x in custom_input.split(",")]
        except:
            st.error("Invalid format")
    
    if st.button("🚀 Run Single Pair Analysis", type="primary", use_container_width=True):
        st.session_state.run_single = True
    
    st.divider()
    
    # Bulk analysis
    st.subheader("Bulk Analysis")
    
    major_pairs = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT", "BNB/USDT"]
    available_majors = [p for p in major_pairs if p in all_pairs]
    
    # Simple multiselect without complex state management
    bulk_pairs = st.multiselect(
        "Select pairs for bulk analysis",
        all_pairs,
        default=available_majors[:3] if available_majors else all_pairs[:3]
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Add Majors"):
            st.session_state.bulk_add_majors = True
    with col2:
        if st.button("Add All"):
            st.session_state.bulk_add_all = True
    
    if st.button("🚀 Run Bulk Analysis", type="primary", use_container_width=True):
        st.session_state.run_bulk = True
        st.session_state.bulk_pairs_to_run = bulk_pairs

# Main results area
with col_right:
    # Single pair analysis
    if hasattr(st.session_state, 'run_single') and st.session_state.run_single:
        st.session_state.run_single = False
        
        optimizer = OrderBookOptimizer()
        
        with st.spinner(f"Analyzing {selected_pair}..."):
            results_df = optimizer.optimize_n(selected_pair, custom_levels)
            
            if not results_df.empty:
                # Find optimal
                optimal_idx = results_df['Score'].idxmax()
                optimal = results_df.loc[optimal_idx]
                
                st.success("✅ Optimization Complete!")
                
                # Metrics
                mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                with mcol1:
                    st.metric("Optimal N", f"{int(optimal['N'])} levels")
                with mcol2:
                    delta = optimal['Choppiness'] - 300
                    st.metric("Choppiness", f"{optimal['Choppiness']}", f"{delta:+.1f}")
                with mcol3:
                    st.metric("Doji Count", int(optimal['Doji Count']))
                with mcol4:
                    st.metric("Score", optimal['Score'])
                
                # Table
                st.subheader("Results")
                st.dataframe(results_df, use_container_width=True)
                
                # Charts
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("Choppiness vs N", "Doji Count vs N")
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=results_df['N'],
                        y=results_df['Choppiness'],
                        mode='lines+markers',
                        line=dict(color='blue', width=2)
                    ),
                    row=1, col=1
                )
                
                fig.add_hrect(
                    y0=250, y1=350,
                    fillcolor="green", opacity=0.2,
                    line_width=0, row=1, col=1
                )
                
                fig.add_trace(
                    go.Bar(
                        x=results_df['N'],
                        y=results_df['Doji Count'],
                        marker_color='red'
                    ),
                    row=1, col=2
                )
                
                fig.update_layout(
                    height=400,
                    showlegend=False,
                    title_text=f"{selected_pair} Optimization"
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Bulk analysis
    elif hasattr(st.session_state, 'run_bulk') and st.session_state.run_bulk:
        st.session_state.run_bulk = False
        pairs_to_analyze = st.session_state.get('bulk_pairs_to_run', [])
        
        if pairs_to_analyze:
            st.header("Bulk Analysis Results")
            optimizer = OrderBookOptimizer()
            
            summary_results = []
            
            for pair in pairs_to_analyze:
                with st.spinner(f"Analyzing {pair}..."):
                    results_df = optimizer.optimize_n(pair)
                    
                    if not results_df.empty:
                        optimal_idx = results_df['Score'].idxmax()
                        optimal = results_df.loc[optimal_idx]
                        
                        summary_results.append({
                            'Pair': pair,
                            'Optimal N': int(optimal['N']),
                            'Choppiness': optimal['Choppiness'],
                            'Doji Count': int(optimal['Doji Count']),
                            'Score': optimal['Score'],
                            'Status': optimal['Status']
                        })
            
            if summary_results:
                summary_df = pd.DataFrame(summary_results)
                st.dataframe(summary_df, use_container_width=True)
    else:
        st.info("👈 Select a pair and click 'Run' to start analysis")