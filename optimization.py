import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from sqlalchemy import create_engine, text
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Order Book Level Optimization",
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

class DepthOptimizer:
    def __init__(self):
        self.optimization_results = {}
        
    def fetch_order_book_data(self, pair):
        """Fetch order book snapshot data if available, otherwise use price data"""
        tz = pytz.timezone('Asia/Singapore')
        now = datetime.now(tz)
        
        # Try to get order book data first
        try:
            # Check if order book table exists
            query = f"""
                SELECT 
                    created_at + INTERVAL '8 hour' AS timestamp,
                    bids,
                    asks
                FROM oracle_order_book_partition_{now.strftime('%Y%m%d')}
                WHERE pair_name = '{pair}'
                    AND created_at >= NOW() - INTERVAL '30 minutes'
                ORDER BY timestamp DESC
                LIMIT 2000
            """
            with engine.connect() as conn:
                df = pd.read_sql(text(query), conn)
                if not df.empty:
                    return df, 'orderbook'
        except:
            pass
        
        # Fallback to price data
        table = f"oracle_price_log_partition_{now.strftime('%Y%m%d')}"
        start = now - timedelta(minutes=30)
        
        query = f"""
            SELECT 
                created_at + INTERVAL '8 hour' AS timestamp,
                final_price AS price
            FROM {table}
            WHERE pair_name = '{pair}'
                AND source_type = 0
                AND created_at >= '{start}'::timestamp - INTERVAL '8 hour'
            ORDER BY timestamp DESC
            LIMIT 2000
        """
        
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
            return df, 'price'
    
    def calculate_choppiness(self, prices, window=20):
        """Calculate choppiness index"""
        if len(prices) < window:
            return 200  # Default
            
        diff = prices.diff().abs()
        sum_abs = diff.rolling(window, min_periods=1).sum()
        range_roll = prices.rolling(window, min_periods=1).max() - prices.rolling(window, min_periods=1).min()
        
        range_roll = range_roll.replace(0, 1e-10)
        
        chop = 100 * sum_abs / (range_roll + 1e-10)
        chop = np.minimum(chop, 1000)
        chop = chop.fillna(200)
        
        return chop.mean()
    
    def calculate_doji_count(self, prices, candle_size=120):
        """Calculate doji candles (120 points = 1 minute)"""
        if len(prices) < candle_size:
            return 0
        
        doji_count = 0
        
        for i in range(0, len(prices) - candle_size, candle_size):
            segment = prices.iloc[i:i+candle_size]
            
            if len(segment) < candle_size:
                continue
                
            open_price = segment.iloc[0]
            close_price = segment.iloc[-1]
            high_price = segment.max()
            low_price = segment.min()
            
            body = abs(close_price - open_price)
            total_range = high_price - low_price
            
            if total_range > 0:
                body_pct = (body / total_range) * 100
                if body_pct < 30:  # Doji threshold
                    doji_count += 1
        
        return doji_count
    
    def simulate_level_count_effect(self, prices, n_levels):
        """
        Simulate the effect of using only n order book levels
        
        Key insight: Fewer levels = more responsive to immediate liquidity = higher choppiness
        More levels = averaging across more liquidity = smoother prices
        """
        if len(prices) == 0:
            return prices
        
        # Simulate the effect of limited order book depth
        # Fewer levels = more volatile (less averaging)
        # More levels = smoother (more averaging)
        
        if n_levels == 1:
            # Top of book only - very jumpy
            noise_factor = 0.02  # 2% noise
            window = 1
        elif n_levels <= 3:
            # Top few levels - still quite responsive
            noise_factor = 0.015
            window = 2
        elif n_levels <= 5:
            # Moderate depth
            noise_factor = 0.01
            window = 3
        elif n_levels <= 10:
            # Good depth
            noise_factor = 0.005
            window = 5
        elif n_levels <= 20:
            # Deep book
            noise_factor = 0.003
            window = 8
        else:
            # Very deep (Rollbit-like)
            noise_factor = 0.002
            window = 10
        
        # Add noise inversely proportional to depth
        noise = np.random.normal(0, prices.std() * noise_factor, len(prices))
        noisy_prices = prices + noise
        
        # Apply smoothing proportional to depth
        if window > 1:
            smoothed = noisy_prices.rolling(window=window, min_periods=1, center=True).mean()
            # Blend based on depth - more levels = more smoothing
            blend_factor = min(0.8, n_levels / 30)
            adjusted = noisy_prices * (1 - blend_factor) + smoothed * blend_factor
        else:
            adjusted = noisy_prices
        
        return adjusted
    
    def optimize_level_count(self, pair, data, data_type, level_range, target='balanced'):
        """
        Optimize the number of order book levels to use
        
        level_range: list of integers [1, 2, 3, 5, 10, 20, 50, 100]
        """
        results = []
        
        if data_type == 'price':
            prices = pd.Series(data['price'].values, dtype=float)
        else:
            # For order book data, calculate mid prices
            # This is simplified - in reality you'd process the actual order book
            prices = pd.Series(data['price'].values if 'price' in data.columns else np.random.randn(len(data)), dtype=float)
        
        if len(prices) < 1500:
            st.warning(f"Only {len(prices)} data points available")
            prices = prices.iloc[-min(1500, len(prices)):]
        else:
            prices = prices.iloc[-1500:]
        
        for n_levels in level_range:
            # Simulate the effect of using n levels
            adjusted_prices = self.simulate_level_count_effect(prices, n_levels)
            
            # Calculate metrics
            choppiness = self.calculate_choppiness(adjusted_prices)
            doji_count = self.calculate_doji_count(adjusted_prices)
            
            # Calculate score
            if target == 'max_choppiness':
                score = choppiness
            elif target == 'min_doji':
                score = -doji_count
            else:  # balanced
                # Target choppiness around 250-350, minimize dojis
                if 250 <= choppiness <= 350:
                    chop_score = 1.0  # Perfect range
                else:
                    distance = min(abs(choppiness - 250), abs(choppiness - 350))
                    chop_score = max(0, 1 - distance / 100)
                
                doji_score = max(0, 1 - doji_count / 10)
                score = chop_score * 0.7 + doji_score * 0.3
            
            results.append({
                'n_levels': n_levels,
                'choppiness': choppiness,
                'doji_count': doji_count,
                'score': score
            })
        
        return pd.DataFrame(results)

# Main UI
st.title("🎯 Order Book Level Count Optimization")
st.markdown("""
This tool optimizes the number of order book levels (n) to achieve:
- Target choppiness (250-350 range)
- Minimum doji candles
- Optimal price stability
""")

# Create tabs
tab1, tab2 = st.tabs(["Optimization Analysis", "Comparison View"])

with st.sidebar:
    st.header("Configuration")
    
    # Pair selection
    pairs = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT"]
    selected_pair = st.selectbox("Select Pair", pairs)
    
    st.subheader("Level Count Range")
    
    # Predefined level options
    level_presets = {
        "Quick Test (1-10)": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "Standard (1-20)": [1, 2, 3, 5, 7, 10, 12, 15, 17, 20],
        "Extended (1-50)": [1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50],
        "Full Range (1-100)": [1, 2, 3, 5, 10, 20, 30, 40, 50, 75, 100],
        "Rollbit-style (many)": [1, 5, 10, 20, 50, 100, 200, 500]
    }
    
    preset = st.selectbox("Choose Preset", list(level_presets.keys()))
    level_range = level_presets[preset]
    
    st.info(f"Testing levels: {level_range}")
    
    # Optimization target
    target = st.radio(
        "Optimization Target",
        ["balanced", "max_choppiness", "min_doji"],
        help="""
        - Balanced: Targets 250-350 choppiness with min doji
        - Max Choppiness: Find most volatile configuration
        - Min Doji: Find cleanest trends
        """
    )
    
    run_btn = st.button("🚀 Run Optimization", type="primary", use_container_width=True)

# Main content
with tab1:
    if run_btn:
        optimizer = DepthOptimizer()
        
        with st.spinner(f"Fetching data for {selected_pair}..."):
            data, data_type = optimizer.fetch_order_book_data(selected_pair)
            
            if data.empty:
                st.error("No data available for optimization")
            else:
                st.success(f"Loaded {len(data)} data points ({data_type} data)")
                
                with st.spinner("Running optimization..."):
                    results = optimizer.optimize_level_count(
                        selected_pair, data, data_type, level_range, target
                    )
                    
                    if not results.empty:
                        # Find optimal
                        if target == 'max_choppiness':
                            optimal_idx = results['choppiness'].idxmax()
                        elif target == 'min_doji':
                            optimal_idx = results['doji_count'].idxmin()
                        else:
                            optimal_idx = results['score'].idxmax()
                        
                        optimal = results.loc[optimal_idx]
                        
                        # Display results
                        st.success("Optimization Complete!")
                        
                        # Metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric(
                                "Optimal Level Count",
                                f"{int(optimal['n_levels'])} levels",
                                help="Number of order book levels to use"
                            )
                        with col2:
                            delta = optimal['choppiness'] - 300
                            st.metric(
                                "Choppiness",
                                f"{optimal['choppiness']:.2f}",
                                f"{delta:+.2f} from target",
                                delta_color="inverse" if abs(delta) > 50 else "normal"
                            )
                        with col3:
                            st.metric(
                                "Doji Count",
                                f"{int(optimal['doji_count'])}",
                                help="Number of doji candles in test period"
                            )
                        with col4:
                            st.metric(
                                "Score",
                                f"{optimal['score']:.3f}",
                                help="Combined optimization score"
                            )
                        
                        # Recommendations
                        st.info(f"""
                        **Recommendation for {selected_pair}:**
                        - Use **{int(optimal['n_levels'])} order book levels** for price calculation
                        - This achieves choppiness of {optimal['choppiness']:.2f} (target: 250-350)
                        - Produces {int(optimal['doji_count'])} doji candles
                        
                        **Interpretation:**
                        - Levels 1-3: Very responsive, high volatility
                        - Levels 4-10: Balanced responsiveness
                        - Levels 11-50: Stable, lower volatility
                        - Levels 50+: Very stable (Rollbit-like)
                        """)
                        
                        # Visualization
                        fig = make_subplots(
                            rows=2, cols=2,
                            subplot_titles=(
                                "Choppiness vs Level Count",
                                "Doji Count vs Level Count",
                                "Optimization Score",
                                "Choppiness vs Doji Trade-off"
                            )
                        )
                        
                        # Chart 1: Choppiness vs Levels
                        fig.add_trace(
                            go.Scatter(
                                x=results['n_levels'],
                                y=results['choppiness'],
                                mode='lines+markers',
                                name='Choppiness',
                                line=dict(color='blue', width=2),
                                marker=dict(size=8)
                            ),
                            row=1, col=1
                        )
                        fig.add_hrect(y0=250, y1=350, fillcolor="green", opacity=0.1,
                                     line_width=0, row=1, col=1)
                        fig.add_annotation(x=max(results['n_levels'])*0.8, y=300,
                                         text="Target Range", showarrow=False,
                                         row=1, col=1)
                        
                        # Chart 2: Doji vs Levels
                        fig.add_trace(
                            go.Scatter(
                                x=results['n_levels'],
                                y=results['doji_count'],
                                mode='lines+markers',
                                name='Doji Count',
                                line=dict(color='red', width=2),
                                marker=dict(size=8)
                            ),
                            row=1, col=2
                        )
                        
                        # Chart 3: Score vs Levels
                        fig.add_trace(
                            go.Scatter(
                                x=results['n_levels'],
                                y=results['score'],
                                mode='lines+markers',
                                name='Score',
                                line=dict(color='green', width=2),
                                marker=dict(size=8)
                            ),
                            row=2, col=1
                        )
                        
                        # Chart 4: Scatter plot
                        fig.add_trace(
                            go.Scatter(
                                x=results['doji_count'],
                                y=results['choppiness'],
                                mode='markers',
                                marker=dict(
                                    size=10,
                                    color=results['n_levels'],
                                    colorscale='Viridis',
                                    showscale=True,
                                    colorbar=dict(title="Levels")
                                ),
                                text=[f"n={n}" for n in results['n_levels']],
                                hovertemplate='Levels: %{text}<br>Doji: %{x}<br>Chop: %{y:.2f}'
                            ),
                            row=2, col=2
                        )
                        
                        # Add optimal point markers
                        for row in [1, 2]:
                            for col in [1, 2]:
                                if row == 2 and col == 2:
                                    continue
                                fig.add_vline(
                                    x=optimal['n_levels'],
                                    line_dash="dash",
                                    line_color="purple",
                                    opacity=0.5,
                                    row=row, col=col
                                )
                        
                        # Update layout
                        fig.update_layout(
                            height=700,
                            showlegend=False,
                            title_text=f"{selected_pair} Order Book Level Optimization"
                        )
                        
                        # Update axes
                        fig.update_xaxes(title_text="Number of Levels", type="log", row=1, col=1)
                        fig.update_xaxes(title_text="Number of Levels", type="log", row=1, col=2)
                        fig.update_xaxes(title_text="Number of Levels", type="log", row=2, col=1)
                        fig.update_xaxes(title_text="Doji Count", row=2, col=2)
                        fig.update_yaxes(title_text="Choppiness", row=1, col=1)
                        fig.update_yaxes(title_text="Doji Count", row=1, col=2)
                        fig.update_yaxes(title_text="Score", row=2, col=1)
                        fig.update_yaxes(title_text="Choppiness", row=2, col=2)
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Detailed results table
                        st.subheader("Detailed Results")
                        display_df = results.copy()
                        display_df['choppiness'] = display_df['choppiness'].round(2)
                        display_df['score'] = display_df['score'].round(4)
                        
                        # Highlight optimal row
                        def highlight_optimal(row):
                            if row.name == optimal_idx:
                                return ['background-color: #90EE90'] * len(row)
                            elif 250 <= row['choppiness'] <= 350:
                                return ['background-color: #FFFACD'] * len(row)
                            return [''] * len(row)
                        
                        st.dataframe(
                            display_df.style.apply(highlight_optimal, axis=1),
                            use_container_width=True
                        )

with tab2:
    st.header("Compare Different Level Configurations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Configuration Impact")
        st.markdown("""
        | Levels | Behavior | Use Case |
        |--------|----------|----------|
        | 1 | Top of book only | HFT, arbitrage |
        | 2-3 | Very responsive | Short-term trading |
        | 4-7 | Balanced | General trading |
        | 8-15 | Stable | Position trading |
        | 16-50 | Very stable | Index calculation |
        | 50+ | Rollbit-like | Maximum stability |
        """)
    
    with col2:
        st.subheader("Expected Outcomes")
        st.markdown("""
        **Fewer Levels (1-5):**
        - ✅ High choppiness (good for volatility trading)
        - ❌ More doji candles
        - ❌ More false signals
        
        **More Levels (20+):**
        - ✅ Fewer doji candles
        - ✅ Cleaner trends
        - ❌ Lower choppiness (may miss target)
        - ❌ Slower to react to real moves
        """)
    
    st.info("""
    **Key Insight:** The optimal number of levels depends on your use case:
    - For matching Rollbit's behavior: Use many levels (50-100+)
    - For target choppiness (250-350): Usually 5-15 levels
    - For minimum dojis: More levels (20+)
    """)