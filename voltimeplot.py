# Save this as pages/06_Better_Volatility_Plot.py in your Streamlit app folder

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import psycopg2
import pytz
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="5min Volatility Plot - Improved",
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
st.title("5-Minute Volatility Time Plot (Improved)")
st.subheader("Single Token - Last 24 Hours (Singapore Time)")

# Define parameters
timeframe = "5min"
lookback_hours = 24
singapore_timezone = pytz.timezone('Asia/Singapore')

# Get current time in Singapore timezone and display prominently
now_utc = datetime.now(pytz.utc)
now_sg = now_utc.astimezone(singapore_timezone)
st.markdown(f"### Current Singapore Time: {now_sg.strftime('%Y-%m-%d %H:%M:%S')}")

# Function to get partition tables based on date range
def get_partition_tables(conn, start_date, end_date):
    """
    Get list of partition tables that need to be queried based on date range.
    Returns a list of table names (oracle_price_log_partition_YYYYMMDD)
    """
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
    
    The query keeps ALL 1-second data points for accurate volatility calculation.
    """
    if not tables:
        return ""
        
    union_parts = []
    
    for table in tables:
        # Query for Surf data (source_type = 0)
        # IMPORTANT: Explicitly convert to Singapore time by adding 8 hours
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

# Get all available tokens from DB
all_tokens = fetch_trading_pairs()

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

# UI Controls
col1, col2, col3 = st.columns([2, 1, 1])

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

with col3:
    # Debug option to see time details
    show_time_debug = st.checkbox("Show Time Debug Info", value=False)

# Fetch and calculate volatility for a token with 5min timeframe
@st.cache_data(ttl=60, show_spinner="Calculating volatility metrics...")  # Reduced cache time to 1 minute
def fetch_and_calculate_volatility(token, lookback_hours=24):
    # Get current time in Singapore timezone
    now_utc = datetime.now(pytz.utc)
    now_sg = now_utc.astimezone(singapore_timezone)
    
    # For checking current time in debug mode
    query_time_sg = now_sg.strftime("%Y-%m-%d %H:%M:%S")
    
    # Try to get data for a longer period than we need (36 hours instead of 24)
    extended_lookback = lookback_hours + 12
    start_time_sg = now_sg - timedelta(hours=extended_lookback)
    
    # Convert for database query
    start_time = start_time_sg.strftime("%Y-%m-%d %H:%M:%S")
    end_time = now_sg.strftime("%Y-%m-%d %H:%M:%S")

    # Get relevant partition tables for today and yesterday
    # Note: Singapore is UTC+8, so we need to cover both days
    extra_day_start = start_time_sg - timedelta(days=1)
    partition_tables = get_partition_tables(conn, extra_day_start, now_sg)
    
    debug_info = {
        "query_time_sg": query_time_sg,
        "start_time_sg": start_time_sg.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time_sg": end_time,
        "partition_tables": partition_tables
    }
    
    if not partition_tables:
        print(f"[{token}] No partition tables found for the specified date range")
        return None, debug_info
    
    # Build query using partition tables
    query = build_query_for_partition_tables(
        partition_tables,
        pair_name=token,
        start_time=start_time,
        end_time=end_time
    )
    
    try:
        print(f"[{token}] Executing query across {len(partition_tables)} partition tables")
        st.info(f"Fetching 1-second level data for {token} from {len(partition_tables)} partition tables...")
        df = pd.read_sql_query(query, conn)
        print(f"[{token}] Query executed. DataFrame shape: {df.shape}")
        
        debug_info["raw_data_shape"] = df.shape

        if df.empty:
            print(f"[{token}] No data found.")
            return None, debug_info

        # Process timestamps - IMPORTANT: timestamps should already be in Singapore time
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Display timestamp range for debugging
        debug_info["min_timestamp"] = df['timestamp'].min().strftime("%Y-%m-%d %H:%M:%S")
        debug_info["max_timestamp"] = df['timestamp'].max().strftime("%Y-%m-%d %H:%M:%S")
        
        # Work directly with the data
        df = df.set_index('timestamp').sort_index()
        raw_price_data = df['final_price'].dropna()
        
        if raw_price_data.empty:
            print(f"[{token}] No data after cleaning.")
            return None, debug_info
        
        # Create a DatetimeIndex with 5-minute frequency
        # Important: floor to exact 5-minute boundaries
        start_date = raw_price_data.index.min().floor('5min')
        end_date = raw_price_data.index.max().ceil('5min')
        five_min_periods = pd.date_range(start=start_date, end=end_date, freq='5min')
        
        debug_info["five_min_start"] = start_date.strftime("%Y-%m-%d %H:%M:%S")
        debug_info["five_min_end"] = end_date.strftime("%Y-%m-%d %H:%M:%S")
        debug_info["five_min_periods_count"] = len(five_min_periods)
        
        # Progress bar for volatility calculation
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("Calculating 5-minute volatility for each time window...")
        
        # Function to calculate volatility for a 5-minute window using 1-second data
        def calculate_intraperiod_volatility(prices):
            """
            Calculate volatility within a period using all data points.
            Returns annualized volatility.
            """
            if len(prices) < 2:
                return np.nan
            
            try:
                # Calculate log returns from all 1-second price points in this window
                log_returns = np.diff(np.log(prices))
                
                # Seconds per year = 60 * 60 * 24 * 365 = 31,536,000
                # Number of seconds in 5 minutes = 300
                annualization_factor = np.sqrt(31536000 / 300)
                
                # Calculate standard deviation and annualize
                vol = np.std(log_returns) * annualization_factor
                return vol
            except Exception as e:
                print(f"Error calculating intraperiod volatility: {e}")
                return np.nan
        
        # Pre-calculate the period boundaries for efficiency
        period_boundaries = [(period, period + pd.Timedelta(minutes=5)) 
                            for period in five_min_periods[:-1]]
        
        # Process each 5-minute window
        volatility_data = []
        ohlc_data = []
        
        for i, (start_time, end_time) in enumerate(period_boundaries):
            # Update progress
            progress = (i + 1) / len(period_boundaries)
            progress_bar.progress(progress)
            
            # Get data for this 5-minute window
            window_data = raw_price_data[(raw_price_data.index >= start_time) & 
                                        (raw_price_data.index < end_time)]
            
            if not window_data.empty and len(window_data) >= 2:
                # Calculate OHLC
                ohlc = {
                    'open': window_data.iloc[0],
                    'high': window_data.max(),
                    'low': window_data.min(),
                    'close': window_data.iloc[-1]
                }
                
                # Calculate volatility using all points in this window
                vol = calculate_intraperiod_volatility(window_data.values)
                
                volatility_data.append((start_time, vol))
                ohlc_data.append((start_time, ohlc))
                
                # Show detailed progress
                if i % 10 == 0:
                    status_text.text(f"Processing window {i+1}/{len(period_boundaries)}: {start_time.strftime('%Y-%m-%d %H:%M')} - {end_time.strftime('%H:%M')}")
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Create DataFrames from collected data
        vol_df = pd.DataFrame(volatility_data, columns=['timestamp', 'realized_vol']).set_index('timestamp')
        
        # Create OHLC DataFrame
        ohlc_df = pd.DataFrame(
            [(t, d['open'], d['high'], d['low'], d['close']) for t, d in ohlc_data],
            columns=['timestamp', 'open', 'high', 'low', 'close']
        ).set_index('timestamp')
        
        # Merge the DataFrames
        result_df = pd.concat([ohlc_df, vol_df], axis=1)
        
        # Calculate returns for price movement
        result_df['returns'] = result_df['close'].pct_change()
        
        # Debug - capture result range
        debug_info["result_min_timestamp"] = result_df.index.min().strftime("%Y-%m-%d %H:%M:%S")
        debug_info["result_max_timestamp"] = result_df.index.max().strftime("%Y-%m-%d %H:%M:%S")
        debug_info["result_count"] = len(result_df)
        
        # Get exactly the requested number of hours worth of data
        # 24 hours = 288 five-minute intervals (24 * 12)
        blocks_needed = lookback_hours * 12
        
        # Take only the most recent blocks_needed points
        recent_data = result_df.tail(blocks_needed)
        
        # Debug - capture final data range
        debug_info["final_min_timestamp"] = recent_data.index.min().strftime("%Y-%m-%d %H:%M:%S")
        debug_info["final_max_timestamp"] = recent_data.index.max().strftime("%Y-%m-%d %H:%M:%S")
        debug_info["final_count"] = len(recent_data)
        
        # Check if we have enough data
        if len(recent_data) < blocks_needed * 0.5:
            print(f"[{token}] Warning: Only found {len(recent_data)} data points out of {blocks_needed} expected")
            debug_info["warning"] = f"Only found {len(recent_data)} data points out of {blocks_needed} expected"
        
        # Classify volatility
        recent_data['vol_info'] = recent_data['realized_vol'].apply(classify_volatility)
        recent_data['vol_regime'] = recent_data['vol_info'].apply(lambda x: x[0])
        recent_data['vol_desc'] = recent_data['vol_info'].apply(lambda x: x[2])
        
        # Flag extreme volatility events
        recent_data['is_extreme'] = recent_data['realized_vol'] >= 1.0
        
        print(f"[{token}] Successful Volatility Calculation")
        return recent_data, debug_info
    except Exception as e:
        st.error(f"Error processing {token}: {e}")
        print(f"[{token}] Error processing: {e}")
        debug_info["error"] = str(e)
        return None, debug_info

# Process the selected token
with st.spinner(f"Calculating volatility for {selected_token}..."):
    vol_data, debug_info = fetch_and_calculate_volatility(selected_token, lookback_hours)

# Display time debug information if requested
if show_time_debug and debug_info:
    st.markdown("### Time and Data Debug Information")
    
    # Create a more readable format
    debug_df = pd.DataFrame([
        {"Parameter": "Query Time (SG)", "Value": debug_info.get("query_time_sg", "N/A")},
        {"Parameter": "Data Start Time (SG)", "Value": debug_info.get("start_time_sg", "N/A")},
        {"Parameter": "Data End Time (SG)", "Value": debug_info.get("end_time_sg", "N/A")},
        {"Parameter": "Raw Data Min Timestamp", "Value": debug_info.get("min_timestamp", "N/A")},
        {"Parameter": "Raw Data Max Timestamp", "Value": debug_info.get("max_timestamp", "N/A")},
        {"Parameter": "5min Periods Start", "Value": debug_info.get("five_min_start", "N/A")},
        {"Parameter": "5min Periods End", "Value": debug_info.get("five_min_end", "N/A")},
        {"Parameter": "Result Min Timestamp", "Value": debug_info.get("result_min_timestamp", "N/A")},
        {"Parameter": "Result Max Timestamp", "Value": debug_info.get("result_max_timestamp", "N/A")},
        {"Parameter": "Final Min Timestamp", "Value": debug_info.get("final_min_timestamp", "N/A")},
        {"Parameter": "Final Max Timestamp", "Value": debug_info.get("final_max_timestamp", "N/A")},
        {"Parameter": "Raw Data Size", "Value": str(debug_info.get("raw_data_shape", "N/A"))},
        {"Parameter": "5min Periods Count", "Value": debug_info.get("five_min_periods_count", "N/A")},
        {"Parameter": "Result Count", "Value": debug_info.get("result_count", "N/A")},
        {"Parameter": "Final Count", "Value": debug_info.get("final_count", "N/A")},
    ])
    
    st.table(debug_df)
    
    # Show partition tables
    st.markdown("#### Partition Tables Used")
    st.write(debug_info.get("partition_tables", []))
    
    # Show any warnings or errors
    if "warning" in debug_info:
        st.warning(debug_info["warning"])
    
    if "error" in debug_info:
        st.error(debug_info["error"])
    
    # If data is available, show the raw timestamps
    if vol_data is not None and not vol_data.empty:
        st.markdown("#### Sample of Time Periods in Data")
        # Show a few timestamps from the data
        time_sample = pd.DataFrame({
            "Index": range(1, min(11, len(vol_data) + 1)),
            "Timestamp": vol_data.index[:10].strftime("%Y-%m-%d %H:%M:%S")
        })
        st.table(time_sample)
        
        # Also show the last few timestamps
        st.markdown("#### Last Few Time Periods in Data")
        time_sample_end = pd.DataFrame({
            "Index": range(len(vol_data) - min(10, len(vol_data)) + 1, len(vol_data) + 1),
            "Timestamp": vol_data.index[-10:].strftime("%Y-%m-%d %H:%M:%S")
        })
        st.table(time_sample_end)

# Create a dedicated volatility plot
if vol_data is not None and not vol_data.empty:
    # Convert to percentage for easier reading
    vol_data_pct = vol_data.copy()
    vol_data_pct['realized_vol'] = vol_data_pct['realized_vol'] * 100
    
    # Get volatility metrics for chart labels
    avg_vol = vol_data_pct['realized_vol'].mean()
    max_vol = vol_data_pct['realized_vol'].max()
    current_vol = vol_data_pct['realized_vol'].iloc[-1]
    
    # Create color mapping for volatility levels
    vol_colors = []
    for val in vol_data_pct['realized_vol']:
        if pd.isna(val):
            vol_colors.append('rgba(100, 100, 100, 0.8)')  # Gray for missing
        elif val < 30:  # Low volatility
            vol_colors.append('rgba(0, 200, 0, 0.8)')
        elif val < 60:  # Medium volatility
            vol_colors.append('rgba(255, 200, 0, 0.8)')
        elif val < 100:  # High volatility
            vol_colors.append('rgba(255, 100, 0, 0.8)')
        else:  # Extreme volatility
            vol_colors.append('rgba(255, 0, 0, 0.8)')
    
    # Calculate minimum y-axis range to ensure visibility
    # If max vol is very low, still show up to at least 20% for visibility
    y_max = max(100, max_vol * 1.2)  # Ensure at least up to 100%
    if max_vol < 20:
        y_max = 20
    
    y_min = 0  # Start at zero
    
    # VOLATILITY PLOT (MAIN)
    fig = go.Figure()
    
    # Title with key metrics
    plot_title = (
        f"{selected_token} Annualized Volatility (5min) - 24h<br>"
        f"<span style='font-size: 14px; color: gray;'>Current: {current_vol:.1f}%, "
        f"Avg: {avg_vol:.1f}%, Max: {max_vol:.1f}%</span>"
    )
    
    # Add volatility line chart with color-coded markers
    fig.add_trace(
        go.Scatter(
            x=vol_data_pct.index,
            y=vol_data_pct['realized_vol'],
            mode='lines+markers',
            line=dict(color='rgba(100, 100, 180, 0.7)', width=3),
            marker=dict(
                color=vol_colors, 
                size=8,
                line=dict(width=1, color='rgba(0,0,0,0.5)')
            ),
            name="Volatility",
            hovertemplate="<b>%{x}</b><br>Vol: %{y:.1f}%<extra></extra>"
        )
    )
    
    # Add threshold lines for volatility regimes
    fig.add_shape(
        type="line",
        x0=vol_data_pct.index.min(),
        x1=vol_data_pct.index.max(),
        y0=30,
        y1=30,
        line=dict(color="rgba(0, 200, 0, 0.5)", width=1.5, dash="dash"),
    )
    
    fig.add_shape(
        type="line",
        x0=vol_data_pct.index.min(),
        x1=vol_data_pct.index.max(),
        y0=60,
        y1=60,
        line=dict(color="rgba(255, 200, 0, 0.5)", width=1.5, dash="dash"),
    )
    
    fig.add_shape(
        type="line",
        x0=vol_data_pct.index.min(),
        x1=vol_data_pct.index.max(),
        y0=100,
        y1=100,
        line=dict(color="rgba(255, 0, 0, 0.5)", width=1.5, dash="dash"),
    )
    
    # Add annotations for the threshold lines
    fig.add_annotation(
        x=vol_data_pct.index.max(),
        y=30,
        text="30% - Low",
        showarrow=False,
        font=dict(size=12, color="green"),
        xanchor="left",
        bgcolor="rgba(255,255,255,0.7)"
    )
    
    fig.add_annotation(
        x=vol_data_pct.index.max(),
        y=60,
        text="60% - Medium",
        showarrow=False,
        font=dict(size=12, color="darkorange"),
        xanchor="left",
        bgcolor="rgba(255,255,255,0.7)"
    )
    
    fig.add_annotation(
        x=vol_data_pct.index.max(),
        y=100,
        text="100% - Extreme",
        showarrow=False,
        font=dict(size=12, color="red"),
        xanchor="left",
        bgcolor="rgba(255,255,255,0.7)"
    )
    
    # Improve x-axis ticks to show more frequent time labels
    # For a 24-hour period, show every hour
    hourly_ticks = pd.date_range(
        start=vol_data_pct.index.min().floor('H'),
        end=vol_data_pct.index.max().ceil('H'),
        freq='1H'
    )
    
    # Filter to only include hours that are within our data range
    valid_ticks = [tick for tick in hourly_ticks if 
                  tick >= vol_data_pct.index.min() and 
                  tick <= vol_data_pct.index.max()]
    
    # Update layout for better readability
    fig.update_layout(
        title=dict(
            text=plot_title,
            font=dict(size=20)
        ),
        height=600,
        margin=dict(l=20, r=20, t=80, b=20),
        showlegend=False,
        hovermode="x unified",
        xaxis=dict(
            title="Time (Singapore)",
            showgrid=True,
            gridcolor='rgba(200,200,200,0.3)',
            tickvals=valid_ticks,  # Use hourly ticks
            ticktext=[tick.strftime('%H:%M<br>%m/%d') for tick in valid_ticks],
            tickangle=-45,
            tickfont=dict(size=12),
        ),
        yaxis=dict(
            title="Annualized Volatility (%)",
            range=[y_min, y_max],  # Set fixed range for better visibility
            showgrid=True,
            gridcolor='rgba(200,200,200,0.3)',
            zeroline=True,
            zerolinecolor='rgba(0,0,0,0.2)',
            titlefont=dict(size=16),
            tickfont=dict(size=14),
        ),
        plot_bgcolor='rgba(250,250,250,0.9)',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", size=14, color="black"),
    )
    
    # Display the figure
    st.plotly_chart(fig, use_container_width=True)
    
    # Display key statistics in a more prominent way
    st.subheader("Volatility Statistics")
    
    # Calculate key metrics
    avg_vol = vol_data_pct['realized_vol'].mean()
    max_vol = vol_data_pct['realized_vol'].max()
    min_vol = vol_data_pct['realized_vol'].min()
    current_vol = vol_data_pct['realized_vol'].iloc[-1]
    extreme_count = vol_data_pct['is_extreme'].sum()
    
    # Layout metrics in a clean grid
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Function to get color based on volatility
    def get_vol_color(vol):
        if vol < 30:
            return "green"
        elif vol < 60:
            return "orange"
        elif vol < 100:
            return "darkorange"
        else:
            return "red"
            
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
    
    # Show a table of the highest volatility periods
    st.subheader("Highest Volatility Periods")
    
    # Sort by volatility (highest first) and take top 10
    high_vol_periods = vol_data_pct.sort_values(by='realized_vol', ascending=False).head(10).copy()
    high_vol_periods['Time (SG)'] = high_vol_periods.index.strftime('%Y-%m-%d %H:%M')
    high_vol_periods['Volatility (%)'] = high_vol_periods['realized_vol'].round(1)
    high_vol_periods['Regime'] = high_vol_periods['vol_desc']
    
    # Select columns for display
    display_df = high_vol_periods[['Time (SG)', 'Volatility (%)', 'Regime']].reset_index(drop=True)
    
    # Add row numbering
    display_df.index = display_df.index + 1
    
    # Display the table
    st.dataframe(display_df, height=400, use_container_width=True)
    
    # Show a 5-minute OHLC chart of prices (cleaner than candlestick)
    st.subheader(f"{selected_token} Price (Last 24 Hours)")
    
    # Create price chart
    price_fig = go.Figure()
    
    # Add price line with high/low range as a shaded area
    price_fig.add_trace(
        go.Scatter(
            x=vol_data.index,
            y=vol_data['high'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip',
            name="High"
        )
    )
    
    price_fig.add_trace(
        go.Scatter(
            x=vol_data.index,
            y=vol_data['low'],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(0, 100, 180, 0.2)',
            showlegend=False,
            hoverinfo='skip',
            name="Low"
        )
    )
    
    price_fig.add_trace(
        go.Scatter(
            x=vol_data.index,
            y=vol_data['close'],
            mode='lines',
            name="Price",
            line=dict(color='rgba(0, 100, 180, 1)', width=2),
            hovertemplate="<b>%{x}</b><br>Price: %{y:,.2f} USDT<extra></extra>"
        )
    )
    
    # Update layout - use same hourly ticks for consistency
    price_fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False,
        hovermode="x unified",
        xaxis=dict(
            title="Time (Singapore)",
            showgrid=True,
            gridcolor='rgba(200,200,200,0.3)',
            tickvals=valid_ticks,  # Use hourly ticks
            ticktext=[tick.strftime('%H:%M<br>%m/%d') for tick in valid_ticks],
            tickangle=-45,
        ),
        yaxis=dict(
            title="Price (USDT)",
            showgrid=True,
            gridcolor='rgba(200,200,200,0.3)',
            zeroline=True,
            zerolinecolor='rgba(0,0,0,0.2)',
        ),
        plot_bgcolor='rgba(250,250,250,0.9)',
        paper_bgcolor='white',
    )
    
    # Display the price chart
    st.plotly_chart(price_fig, use_container_width=True)
    
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
        1. For each 5-minute window, we gather all 1-second price data points within that window
        2. We calculate the standard deviation of logarithmic returns from these 1-second points
        3. This standard deviation is then annualized by multiplying by √(31,536,000/300)
           - 31,536,000 = seconds in a year (60 * 60 * 24 * 365)
           - 300 = seconds in a 5-minute window
        
        **Interpretation**:
        - Each 5-minute window's volatility represents the actual price fluctuation within that specific period
        - Higher volatility generally indicates more risk and uncertainty
        - Volatility tends to cluster (periods of high volatility are often followed by more high volatility)
        - Extreme volatility events may indicate market disruptions or significant news
        """)
else:
    st.error(f"No data available for {selected_token} in the selected time period. Try another token or time range.")