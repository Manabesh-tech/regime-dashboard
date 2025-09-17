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

st.set_page_config(
    page_title="Order Book N Optimizer",
    page_icon="📊",
    layout="wide"
)

# Database configuration - use your actual credentials
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
        pool_pre_ping=True
    )

engine = get_db()

class OrderBookOptimizer:
    def __init__(self):
        self.results = {}

    def fetch_order_book_data(self, pair, minutes=30):
        """Fetch real order book data"""
        tz = pytz.timezone('Asia/Singapore')
        now = datetime.now(tz)
        table = f"oracle_order_book_partition_{now.strftime('%Y%m%d')}"

        query = f"""
            SELECT 
                created_at + INTERVAL '8 hour' AS timestamp,
                bids::text,
                asks::text,
                jsonb_array_length(bids::jsonb) as bid_levels,
                jsonb_array_length(asks::jsonb) as ask_levels,
                LEAST(jsonb_array_length(bids::jsonb), jsonb_array_length(asks::jsonb)) as max_levels
            FROM {table}
            WHERE pair_name = '{pair}'
                AND created_at >= NOW() - INTERVAL '{minutes} minutes'
            ORDER BY created_at ASC
        """

        try:
            with engine.connect() as conn:
                df = pd.read_sql(text(query), conn)
                if not df.empty:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    return df
                return pd.DataFrame()
        except Exception as e:
            st.error(f"Database error: {e}")
            return pd.DataFrame()

    def calculate_weighted_price(self, bids_str, asks_str, n_levels):
        """Calculate weighted mid price using n levels"""
        try:
            bids = json.loads(bids_str)
            asks = json.loads(asks_str)

            # Use only n levels
            bids = bids[:n_levels]
            asks = asks[:n_levels]

            if not bids or not asks:
                return None

            # For n=1, simple mid
            if n_levels == 1:
                return (float(bids[0]['p']) + float(asks[0]['p'])) / 2

            # Volume-weighted calculation
            bid_sum = sum(float(b['p']) * float(b['v']) for b in bids)
            bid_vol = sum(float(b['v']) for b in bids)
            ask_sum = sum(float(a['p']) * float(a['v']) for a in asks)
            ask_vol = sum(float(a['v']) for a in asks)

            if bid_vol > 0 and ask_vol > 0:
                bid_vwap = bid_sum / bid_vol
                ask_vwap = ask_sum / ask_vol
                return (bid_vwap + ask_vwap) / 2

            return None
        except:
            return None

    def calculate_metrics(self, prices):
        """Calculate choppiness and doji metrics"""
        if len(prices) < 20:
            return None

        # Choppiness (your formula)
        window = 20
        diff = prices.diff().abs()
        sum_abs = diff.rolling(window, min_periods=1).sum()
        range_roll = prices.rolling(window, min_periods=1).max() - prices.rolling(window, min_periods=1).min()
        range_roll = range_roll.replace(0, 1e-10)
        chop = 100 * sum_abs / range_roll
        chop = np.minimum(chop, 1000)
        choppiness = chop.mean()

        # Doji count (1-minute candles)
        doji_count = 0
        high_wick_count = 0

        if len(prices) > 120:  # Need at least 1 minute of data
            # Resample to 1-minute OHLC
            ohlc = prices.resample('1T').ohlc().dropna()

            for _, candle in ohlc.iterrows():
                body = abs(candle['close'] - candle['open'])
                total_range = candle['high'] - candle['low']

                if total_range > 0:
                    body_pct = (body / total_range) * 100
                    if body_pct < 30:
                        doji_count += 1

                    upper_wick = candle['high'] - max(candle['open'], candle['close'])
                    lower_wick = min(candle['open'], candle['close']) - candle['low']
                    wick_pct = ((upper_wick + lower_wick) / total_range) * 100
                    if wick_pct > 50:
                        high_wick_count += 1

        return {
            'choppiness': choppiness,
            'doji_count': doji_count,
            'high_wick_count': high_wick_count,
            'data_points': len(prices),
            'volatility': (prices.std() / prices.mean()) * 100 if prices.mean() > 0 else 0
        }

    def optimize_n(self, pair, level_range=[1,2,3,5,7,10,15,20,30,50]):
        """Find optimal N value"""
        df = self.fetch_order_book_data(pair)

        if df.empty:
            st.error(f"No order book data available for {pair}")
            return pd.DataFrame()

        max_available = df['max_levels'].min()
        st.info(f"📊 Found {len(df)} snapshots, max consistent depth: {max_available} levels")

        # Filter to available levels
        level_range = [n for n in level_range if n <= max_available]

        results = []
        progress = st.progress(0)

        for i, n in enumerate(level_range):
            # Calculate prices for this N
            prices_list = []
            timestamps = []

            for _, row in df.iterrows():
                if row['max_levels'] >= n:
                    price = self.calculate_weighted_price(row['bids'], row['asks'], n)
                    if price:
                        prices_list.append(price)
                        timestamps.append(row['timestamp'])

            if len(prices_list) > 100:
                prices = pd.Series(prices_list, index=pd.DatetimeIndex(timestamps))
                metrics = self.calculate_metrics(prices)

                if metrics:
                    # Calculate optimization score
                    chop = metrics['choppiness']

                    # Score calculation: prioritize 250-350 range, minimize dojis
                    if 250 <= chop <= 350:
                        chop_score = 100  # Perfect
                    elif 200 <= chop <= 400:
                        chop_score = 50 - abs(chop - 300) / 2  # Acceptable
                    else:
                        chop_score = max(0, 25 - abs(chop - 300) / 10)  # Poor

                    # Doji penalty (lower is better)
                    doji_penalty = metrics['doji_count'] * 5
                    wick_penalty = metrics['high_wick_count'] * 2

                    total_score = chop_score - doji_penalty - wick_penalty

                    results.append({
                        'N': n,
                        'Choppiness': round(chop, 2),
                        'Doji Count': metrics['doji_count'],
                        'High Wicks': metrics['high_wick_count'],
                        'Score': round(total_score, 2),
                        'Data Points': metrics['data_points'],
                        'Volatility %': round(metrics['volatility'], 4),
                        'Status': '✅' if 250 <= chop <= 350 else '⚠️' if 200 <= chop <= 400 else '❌'
                    })

            progress.progress((i + 1) / len(level_range))

        progress.empty()
        return pd.DataFrame(results)

# UI
st.title("🎯 Order Book N-Level Optimizer")
st.markdown("Finding optimal order book depth (N) for choppiness 250-350 with minimum dojis")

# Sidebar
with st.sidebar:
    st.header("Settings")

    pairs = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT", "BNB/USDT"]
    selected_pair = st.selectbox("Trading Pair", pairs)

    # Level presets
    presets = {
        "Quick (1-10)": [1, 2, 3, 5, 7, 10],
        "Standard (1-20)": [1, 2, 3, 5, 7, 10, 15, 20],
        "Extended (1-50)": [1, 2, 3, 5, 10, 15, 20, 30, 40, 50],
        "Full (1-100)": [1, 3, 5, 10, 20, 30, 50, 70, 100]
    }

    preset = st.selectbox("Level Range", list(presets.keys()))
    levels = presets[preset]

    st.info(f"Testing N values: {levels}")

    run_btn = st.button("🚀 Run Optimization", type="primary", use_container_width=True)

# Main area
if run_btn:
    optimizer = OrderBookOptimizer()

    with st.spinner(f"Analyzing {selected_pair}..."):
        results_df = optimizer.optimize_n(selected_pair, levels)

        if not results_df.empty:
            # Find optimal
            optimal_idx = results_df['Score'].idxmax()
            optimal = results_df.loc[optimal_idx]

            # Display optimal result
            st.success("✅ Optimization Complete!")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Optimal N", f"{int(optimal['N'])} levels")
            with col2:
                delta = optimal['Choppiness'] - 300
                st.metric("Choppiness", f"{optimal['Choppiness']}", f"{delta:+.1f}")
            with col3:
                st.metric("Doji Count", int(optimal['Doji Count']))
            with col4:
                st.metric("Score", optimal['Score'])

            # Results table
            st.subheader("All Results")

            # Highlight optimal row
            def highlight_row(row):
                if row.name == optimal_idx:
                    return ['background-color: #90EE90'] * len(row)
                elif row['Status'] == '✅':
                    return ['background-color: #FFFACD'] * len(row)
                return [''] * len(row)

            st.dataframe(
                results_df.style.apply(highlight_row, axis=1),
                use_container_width=True
            )

            # Visualization
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Choppiness vs N", "Doji Count vs N")
            )

            # Choppiness chart
            fig.add_trace(
                go.Scatter(
                    x=results_df['N'],
                    y=results_df['Choppiness'],
                    mode='lines+markers',
                    name='Choppiness',
                    line=dict(color='blue', width=2),
                    marker=dict(size=8)
                ),
                row=1, col=1
            )

            # Add target zone
            fig.add_hrect(
                y0=250, y1=350,
                fillcolor="green", opacity=0.2,
                line_width=0, row=1, col=1
            )

            # Doji chart
            fig.add_trace(
                go.Bar(
                    x=results_df['N'],
                    y=results_df['Doji Count'],
                    name='Doji Count',
                    marker_color='red'
                ),
                row=1, col=2
            )

            # Mark optimal
            fig.add_vline(
                x=optimal['N'],
                line_dash="dash",
                line_color="purple",
                opacity=0.5,
                row=1, col="all"
            )

            fig.update_xaxes(title="Order Book Levels (N)", type="log")
            fig.update_yaxes(title="Choppiness", row=1, col=1)
            fig.update_yaxes(title="Doji Count", row=1, col=2)

            fig.update_layout(
                height=400,
                title_text=f"{selected_pair} Optimization Results",
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

            # Recommendation
            st.info(f"""
            **Recommendation for {selected_pair}:**
            
            Use **{int(optimal['N'])} order book levels** for price calculation:
            - Achieves choppiness of {optimal['Choppiness']} (target: 250-350)
            - Produces {int(optimal['Doji Count'])} doji candles
            - High wick count: {int(optimal['High Wicks'])}
            
            This configuration provides the best balance between price responsiveness and stability.
            """)
        else:
            st.error("No valid results. Check data availability.")
else:
    st.info("👆 Click 'Run Optimization' to start analysis")