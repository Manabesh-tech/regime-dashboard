# Save this as pages/06_5min_Volatility_Plot.py in your Streamlit app folder

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import psycopg2
import pytz
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="5min Volatility Plot",
    page_icon="📈",
    layout="wide"
)

# --- DB CONFIG ---
try:
    # You can use st.secrets in production, or hardcode for testing
    try:
        db_config = st.secrets["database"]
        db_params = {
            'host': db_config['host'],
            'port': db_config['port'],
            'database': db_config['database'],
            'user': db_config['user'],
            'password': db_config['password']
        }
    except:
        # Fallback to hardcoded credentials if secrets aren't available
        db_params = {
            'host': 'aws-jp-tk-surf-pg-public.cluster-csteuf9lw8dv.ap-northeast-1.rds.amazonaws.com',
            'port': 5432,
            'database': 'replication_report',
            'user': 'public_replication',
            'password': '866^FKC4hllk'
        }
    
    conn = psycopg2.connect(
        host=db_params['host'],
        port=db_params['port'],
        database=db_params['database'],
        user=db_params['user'],
        password=db_params['password']
    )
except Exception as e:
    st.error(f"Error connecting to the database: {e}")
    st.stop()

# --- UI Setup ---
st.title("5-Minute Volatility Plot")
st.subheader("Single Token - Last 24 Hours (Singapore Time)")

# Define parameters for the 5-minute timeframe
timeframe = "5min"
lookback_hours = 24  # Changed to 24 hours
rolling_window = 10  # Reduced window size for 5min data
expected_points = 288  # Expected data points per pair over 24 hours (24 hours * 12 5-min periods per hour)
singapore_timezone = pytz.timezone('Asia/Singapore')

# Get current time in Singapore timezone
now_utc = datetime.now(pytz.utc)
now_sg = now_utc.astimezone(singapore_timezone)
st.write(f"Current Singapore Time: {now_sg.strftime('%Y-%m-%d %H:%M:%S')}")

# Set extreme volatility threshold
extreme_vol_threshold = 1.0  # 100% annualized volatility

# Function to get partition tables based on date range - OPTIMIZED to query fewer tables
def get_partition_tables(conn, start_date, end_date):
    """
    Get list of partition tables that need to be queried based on date range.
    Returns a list of table names (oracle_price_log_partition_YYYYMMDD)
    """
    # Convert to datetime objects if they're strings
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str) and end_date:
        end_date = pd.to_datetime(end_date)
    elif end_date is None:
        end_date = datetime.now()
        
    # Ensure timezone is removed
    start_date = start_date.replace(tzinfo=None)
    end_date = end_date.replace(tzinfo=None)
        
    # Generate list of dates between start and end
    current_date = start_date
    dates = []
    
    while current_date <= end_date:
        dates.append(current_date.strftime("%Y%m%d"))
        current_date += timedelta(days=1)
    
    # Create table names from dates
    table_names = [f"oracle_price_log_partition_{date}" for date in dates]
    
    # Verify which tables actually exist in the database
    cursor = conn.cursor()
    existing_tables = []
    
    # Query to check all tables at once
    if table_names:
        table_list_str = "', '".join(table_names)
        cursor.execute(f"""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('{table_list_str}')
        """)
        
        existing_tables = [row[0] for row in cursor.fetchall()]
    
    cursor.close()
    
    if not existing_tables:
        st.warning(f"No partition tables found for the date range {start_date.date()} to {end_date.date()}")
    
    return existing_tables

# Function to build query across partition tables
def build_query_for_partition_tables(tables, pair_name, start_time, end_time):
    """
    Build a complete UNION query for multiple partition tables.
    This creates a complete, valid SQL query with correct WHERE clauses.
    """
    if not tables:
        return ""
        
    union_parts = []
    
    for table in tables:
        # Query for Surf data (source_type = 0)
        # Use a simpler, more direct query to avoid timezone complications
        query = f"""
        SELECT 
            pair_name,
            created_at + INTERVAL '8 hour' AS timestamp,
            final_price
        FROM 
            public.{table}
        WHERE 
            created_at >= '{start_time}'::timestamp - INTERVAL '8 hour'
            AND created_at <= '{end_time}'::timestamp - INTERVAL '8 hour'
            AND source_type = 0
            AND pair_name = '{pair_name}'
        """
        
        union_parts.append(query)
    
    # Join with UNION and add ORDER BY at the end
    complete_query = " UNION ".join(union_parts) + " ORDER BY timestamp"
    return complete_query

# Function to fetch trading pairs from database
@st.cache_data(ttl=600)
def fetch_trading_pairs():
    query = """
    SELECT pair_name 
    FROM trade_pool_pairs 
    WHERE status = 1
    ORDER BY pair_name
    """
    
    try:
        df = pd.read_sql_query(query, conn)
        return df['pair_name'].tolist()
    except Exception as e:
        st.error(f"Error fetching trading pairs: {e}")
        return ["BTC/USDT", "ETH/USDT"]  # Default pairs if database query fails

# Get all available tokens from DB by fetching active trading pairs
all_tokens = fetch_trading_pairs()

# Function to calculate volatility metrics - OPTIMIZED calculation
def calculate_volatility_metrics(price_series):
    if price_series is None or len(price_series) < 2:
        return {
            'realized_vol': np.nan
        }
    
    try:
        # Calculate log returns
        log_returns = np.diff(np.log(price_series))
        
        # Realized volatility - for 5min data, we need to adjust the annualization factor
        # 5min = 12 periods per hour * 24 hours * 365 days = 105120 periods per year
        realized_vol = np.std(log_returns) * np.sqrt(105120)  
        
        return {
            'realized_vol': realized_vol
        }
    except Exception as e:
        print(f"Error in volatility calculation: {e}")
        return {
            'realized_vol': np.nan
        }

# Volatility classification function
def classify_volatility(vol):
    if pd.isna(vol):
        return ("UNKNOWN", 0, "Insufficient data")
    elif vol < 0.30:  # 30% annualized volatility threshold for low volatility
        return ("LOW", 1, "Low volatility")
    elif vol < 0.60:  # 60% annualized volatility threshold for medium volatility
        return ("MEDIUM", 2, "Medium volatility")
    elif vol < 1.00:  # 100% annualized volatility threshold for high volatility
        return ("HIGH", 3, "High volatility")
    else:
        return ("EXTREME", 4, "Extreme volatility")

# Function to generate aligned 5-minute time blocks for the past 24 hours
def generate_aligned_time_blocks(current_time, hours_back=24):
    """
    Generate fixed 5-minute time blocks for past X hours,
    aligned with standard 5-minute intervals
    """
    # Round down to the nearest 5-minute mark
    minute = current_time.minute
    rounded_minute = (minute // 5) * 5
    latest_complete_block_end = current_time.replace(minute=rounded_minute, second=0, microsecond=0)
    
    # Generate block labels for display
    blocks = []
    for i in range(hours_back * 12):  # 24 hours of 5-minute blocks = 288 blocks
        block_end = latest_complete_block_end - timedelta(minutes=i*5)
        block_start = block_end - timedelta(minutes=5)
        block_label = f"{block_start.strftime('%H:%M')}"
        blocks.append((block_start, block_end, block_label))
    
    return blocks

# UI Controls - OPTIMIZED layout for better user experience
col1, col2 = st.columns([3, 1])

with col1:
    # Let user select a single token for analysis
    default_token = "BTC/USDT" if "BTC/USDT" in all_tokens else all_tokens[0]
    selected_token = st.selectbox(
        "Select Token", 
        all_tokens,
        index=all_tokens.index(default_token) if default_token in all_tokens else 0
    )

with col2:
    # Add a refresh button
    if st.button("Refresh Data", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.experimental_rerun()

# Fetch and calculate volatility for a token with 5min timeframe
@st.cache_data(ttl=300, show_spinner="Calculating volatility metrics...")
def fetch_and_calculate_volatility(token, lookback_hours=24):
    # Get current time in Singapore timezone
    now_utc = datetime.now(pytz.utc)
    now_sg = now_utc.astimezone(singapore_timezone)
    
    # Try to get data for a longer period than we need (36 hours instead of 24)
    # This gives us a better chance to capture the full 24 hours we want
    extended_lookback = lookback_hours + 12
    start_time_sg = now_sg - timedelta(hours=extended_lookback)
    
    # Convert for database query
    start_time = start_time_sg.strftime("%Y-%m-%d %H:%M:%S")
    end_time = now_sg.strftime("%Y-%m-%d %H:%M:%S")

    # Get relevant partition tables for this time range plus the previous day
    # This ensures we get all the data even if near day boundaries
    extra_day_start = start_time_sg - timedelta(days=1)
    partition_tables = get_partition_tables(conn, extra_day_start, now_sg)
    
    if not partition_tables:
        print(f"[{token}] No partition tables found for the specified date range")
        return None
    
    # Build query using partition tables
    query = build_query_for_partition_tables(
        partition_tables,
        pair_name=token,
        start_time=start_time,
        end_time=end_time
    )
    
    try:
        print(f"[{token}] Executing query across {len(partition_tables)} partition tables")
        df = pd.read_sql_query(query, conn)
        print(f"[{token}] Query executed. DataFrame shape: {df.shape}")

        if df.empty:
            print(f"[{token}] No data found.")
            return None

        # Process timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Work directly with the data
        df = df.set_index('timestamp').sort_index()
        price_data = df['final_price'].dropna()
        
        if price_data.empty:
            print(f"[{token}] No data after cleaning.")
            return None
        
        # Resample to exactly 5min intervals
        five_min_ohlc = price_data.resample('5min').ohlc().dropna()
        
        if five_min_ohlc.empty:
            print(f"[{token}] No 5-min data after resampling.")
            return None
        
        # Calculate rolling volatility on 5-minute close prices
        five_min_ohlc['realized_vol'] = five_min_ohlc['close'].rolling(window=rolling_window).apply(
            lambda x: calculate_volatility_metrics(x)['realized_vol']
        )

        # Calculate returns for price movement
        five_min_ohlc['returns'] = five_min_ohlc['close'].pct_change()
        
        # Get exactly the requested number of hours worth of data
        # 24 hours = 288 five-minute intervals (24 * 12)
        blocks_needed = lookback_hours * 12  # Number of 5-minute blocks in lookback period
        
        # Take only the most recent blocks_needed points
        recent_data = five_min_ohlc.tail(blocks_needed)
        
        # Check if we have enough data
        if len(recent_data) < blocks_needed * 0.5:  # If we have less than 50% of expected points
            print(f"[{token}] Warning: Only found {len(recent_data)} data points out of {blocks_needed} expected")
        
        # Classify volatility
        recent_data['vol_info'] = recent_data['realized_vol'].apply(classify_volatility)
        recent_data['vol_regime'] = recent_data['vol_info'].apply(lambda x: x[0])
        recent_data['vol_desc'] = recent_data['vol_info'].apply(lambda x: x[2])
        
        # Flag extreme volatility events
        recent_data['is_extreme'] = recent_data['realized_vol'] >= extreme_vol_threshold
        
        print(f"[{token}] Successful Volatility Calculation")
        return recent_data
    except Exception as e:
        st.error(f"Error processing {selected_token}: {e}")
        print(f"[{token}] Error processing: {e}")
        return None

# Process the selected token
with st.spinner(f"Calculating volatility for {selected_token}..."):
    vol_data = fetch_and_calculate_volatility(selected_token, lookback_hours)

# Create the plots
if vol_data is not None and not vol_data.empty:
    # Create a two-part figure with price and volatility
    fig = make_subplots(
        rows=2, 
        cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(f"{selected_token} Price (USDT)", f"{selected_token} Annualized Volatility (5min, %)"),
        row_heights=[0.6, 0.4]
    )
    
    # Add price candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=vol_data.index,
            open=vol_data['open'],
            high=vol_data['high'],
            low=vol_data['low'],
            close=vol_data['close'],
            name="Price"
        ),
        row=1, 
        col=1
    )
    
    # Convert to percentage for easier reading
    vol_data_pct = vol_data.copy()
    vol_data_pct['realized_vol'] = vol_data_pct['realized_vol'] * 100
    
    # Create color mapping for volatility levels
    colors = []
    for val in vol_data_pct['realized_vol']:
        if pd.isna(val):
            colors.append('rgba(100, 100, 100, 0.8)')  # Gray for missing
        elif val < 30:  # Low volatility
            colors.append('rgba(0, 255, 0, 0.8)')
        elif val < 60:  # Medium volatility
            colors.append('rgba(255, 255, 0, 0.8)')
        elif val < 100:  # High volatility
            colors.append('rgba(255, 165, 0, 0.8)')
        else:  # Extreme volatility
            colors.append('rgba(255, 0, 0, 0.8)')
    
    # Add volatility bar chart
    fig.add_trace(
        go.Bar(
            x=vol_data_pct.index,
            y=vol_data_pct['realized_vol'],
            marker_color=colors,
            name="Volatility",
            hovertemplate="%{x}<br>Vol: %{y:.1f}%<extra></extra>"
        ),
        row=2, 
        col=1
    )
    
    # Add volatility threshold lines
    fig.add_shape(
        type="line",
        x0=vol_data_pct.index.min(),
        x1=vol_data_pct.index.max(),
        y0=30,
        y1=30,
        line=dict(color="rgba(0, 255, 0, 0.5)", width=1, dash="dash"),
        row=2, 
        col=1
    )
    
    fig.add_shape(
        type="line",
        x0=vol_data_pct.index.min(),
        x1=vol_data_pct.index.max(),
        y0=60,
        y1=60,
        line=dict(color="rgba(255, 255, 0, 0.5)", width=1, dash="dash"),
        row=2, 
        col=1
    )
    
    fig.add_shape(
        type="line",
        x0=vol_data_pct.index.min(),
        x1=vol_data_pct.index.max(),
        y0=100,
        y1=100,
        line=dict(color="rgba(255, 0, 0, 0.5)", width=1, dash="dash"),
        row=2, 
        col=1
    )
    
    # Add annotations for the threshold lines
    fig.add_annotation(
        x=vol_data_pct.index.max(),
        y=30,
        text="30% - Low",
        showarrow=False,
        font=dict(size=10, color="green"),
        xanchor="left",
        row=2, 
        col=1
    )
    
    fig.add_annotation(
        x=vol_data_pct.index.max(),
        y=60,
        text="60% - Medium",
        showarrow=False,
        font=dict(size=10, color="yellow"),
        xanchor="left",
        row=2, 
        col=1
    )
    
    fig.add_annotation(
        x=vol_data_pct.index.max(),
        y=100,
        text="100% - Extreme",
        showarrow=False,
        font=dict(size=10, color="red"),
        xanchor="left",
        row=2, 
        col=1
    )
    
    # Update layout
    fig.update_layout(
        title=f"{selected_token} - 24 Hour Price and Volatility (5min bars, Singapore Time)",
        xaxis_rangeslider_visible=False,
        height=800,
        width=1000,
        showlegend=False,
        hovermode="x unified",
        xaxis=dict(
            type="date",
            tickformat="%H:%M\n%m/%d",
            tickangle=-45,
        ),
        yaxis2=dict(
            title="Annualized Volatility (%)",
            side="right",
            showgrid=True,
            zeroline=True,
            zerolinecolor='rgba(0,0,0,0.2)',
        )
    )
    
    # Display the figure
    st.plotly_chart(fig, use_container_width=True)
    
    # Display statistics
    st.subheader(f"Volatility Statistics for {selected_token} (Last 24 Hours)")
    
    # Calculate key metrics
    avg_vol = vol_data['realized_vol'].mean() * 100
    max_vol = vol_data['realized_vol'].max() * 100
    min_vol = vol_data['realized_vol'].min() * 100
    current_vol = vol_data['realized_vol'].iloc[-1] * 100
    extreme_count = vol_data['is_extreme'].sum()
    
    # Display metrics in columns
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Function to determine color
    def get_vol_color(vol):
        if pd.isna(vol):
            return 'gray'
        elif vol < 30:
            return 'green'
        elif vol < 60:
            return 'yellow'
        elif vol < 100:
            return 'orange'
        else:
            return 'red'
            
    col1.metric(
        "Current Vol", 
        f"{current_vol:.1f}%", 
        delta=f"{current_vol - avg_vol:.1f}%" if not pd.isna(current_vol) and not pd.isna(avg_vol) else None,
        delta_color="inverse"
    )
    col2.metric("Average Vol", f"{avg_vol:.1f}%")
    col3.metric("Maximum Vol", f"{max_vol:.1f}%")
    col4.metric("Minimum Vol", f"{min_vol:.1f}%")
    col5.metric("Extreme Events", f"{extreme_count}")
    
    # Display volatility regime distribution
    st.subheader("Volatility Regime Distribution")
    
    # Count occurrence of each regime
    regime_counts = vol_data['vol_regime'].value_counts()
    
    # Calculate percentages
    total_points = len(vol_data)
    regime_pct = {regime: count/total_points*100 for regime, count in regime_counts.items()}
    
    # Create ordered dict for consistent display
    ordered_regimes = ['LOW', 'MEDIUM', 'HIGH', 'EXTREME', 'UNKNOWN']
    regime_colors = ['rgba(0,255,0,0.8)', 'rgba(255,255,0,0.8)', 'rgba(255,165,0,0.8)', 'rgba(255,0,0,0.8)', 'rgba(100,100,100,0.8)']
    
    # Prepare data for pie chart
    pie_labels = []
    pie_values = []
    pie_colors = []
    
    for regime, color in zip(ordered_regimes, regime_colors):
        if regime in regime_counts:
            pie_labels.append(f"{regime} ({regime_pct.get(regime, 0):.1f}%)")
            pie_values.append(regime_counts.get(regime, 0))
            pie_colors.append(color)
    
    # Create pie chart
    fig_pie = go.Figure(
        data=[go.Pie(
            labels=pie_labels, 
            values=pie_values, 
            marker=dict(colors=pie_colors),
            textinfo='label+percent',
            hole=.3
        )]
    )
    
    fig_pie.update_layout(
        height=400,
        width=600
    )
    
    # Display side-by-side with some additional stats
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Display additional metrics
        st.subheader("Daily Volatility Analysis")
        
        # Calculate daily equivalent volatility
        daily_vol = avg_vol / np.sqrt(365)
        
        # Expected daily range based on volatility (assuming normal distribution)
        # 68% of the time, price should move within ±1 standard deviation
        one_day_move_68pct = daily_vol
        # 95% of the time, price should move within ±2 standard deviations
        one_day_move_95pct = daily_vol * 2
        # 99.7% of the time, price should move within ±3 standard deviations
        one_day_move_99pct = daily_vol * 3
        
        st.markdown(f"**Daily Volatility (24h Average):** {daily_vol:.2f}%")
        st.markdown("**Expected Daily Price Movement:**")
        st.markdown(f"- 68% of days: ±{one_day_move_68pct:.2f}%")
        st.markdown(f"- 95% of days: ±{one_day_move_95pct:.2f}%")
        st.markdown(f"- 99.7% of days: ±{one_day_move_99pct:.2f}%")
        
        # Calculate time spent in each volatility regime
        time_in_regime = {
            regime: count/total_points*24 for regime, count in regime_counts.items()
        }
        
        st.markdown("**Hours per Day in Each Volatility Regime:**")
        for regime in ordered_regimes:
            if regime in time_in_regime:
                color = get_vol_color(100 if regime == "EXTREME" else 
                                     50 if regime == "MEDIUM" else 
                                     80 if regime == "HIGH" else 
                                     20 if regime == "LOW" else 0)
                st.markdown(f"- <span style='color:{color}'>{regime}</span>: {time_in_regime.get(regime, 0):.1f} hours", unsafe_allow_html=True)

    # Show extreme volatility events if any
    if extreme_count > 0:
        st.subheader("Extreme Volatility Events")
        extreme_events = vol_data[vol_data['is_extreme']].copy()
        extreme_events['realized_vol_pct'] = extreme_events['realized_vol'] * 100
        extreme_events['time'] = extreme_events.index.strftime('%Y-%m-%d %H:%M')
        
        # Sort by volatility (highest first)
        extreme_events = extreme_events.sort_values(by='realized_vol', ascending=False)
        
        # Create dataframe for display
        display_df = pd.DataFrame({
            'Time (SG)': extreme_events['time'],
            'Volatility (%)': extreme_events['realized_vol_pct'].round(1),
            'Price': extreme_events['close'].round(2)
        })
        
        # Reset index
        display_df = display_df.reset_index(drop=True)
        
        # Display the table
        st.dataframe(display_df, use_container_width=True)
    
    with st.expander("Understanding Volatility Metrics", expanded=False):
        st.markdown("""
        ### Volatility Measurement Explanation
        
        **Annualized Volatility**: This is a measure of how much the price fluctuates, expressed as an annualized percentage. For example, a volatility of 50% means that the token's price is expected to move up or down by 50% over the course of a year (with 68% probability).
        
        **Volatility Regimes**:
        - **Low Volatility** (<30%): Price movements are relatively small and predictable
        - **Medium Volatility** (30-60%): Normal market conditions with moderate price movements
        - **High Volatility** (60-100%): Significant price swings, potentially due to important market events
        - **Extreme Volatility** (>100%): Very large price movements, often indicating unusual market stress
        
        **Calculation Method**:
        1. We calculate volatility using the standard deviation of logarithmic returns of 5-minute price data
        2. The result is annualized by multiplying by the square root of the number of 5-minute periods in a year
        3. For the main chart, we use a rolling window of 10 periods (50 minutes of data) to capture short-term volatility
        
        **Interpretation**:
        - Higher volatility generally indicates more risk and uncertainty
        - Volatility tends to cluster (periods of high volatility are often followed by more high volatility)
        - Extreme volatility events may indicate market disruptions or significant news
        """)
else:
    st.error(f"No data available for {selected_token} in the selected time period. Try another token or time range.")