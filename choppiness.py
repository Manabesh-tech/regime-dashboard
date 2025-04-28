import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
import time
import traceback

# Clear cache at startup to ensure fresh data
if hasattr(st, 'cache_data'):
    st.cache_data.clear()

# Page configuration
st.set_page_config(page_title="User-defined Interval Choppiness", layout="wide")

# Enhanced CSS for better UI
st.markdown("""
<style>
    .main .block-container {max-width: 98% !important; padding-top: 1rem !important;}
    h1, h2, h3 {margin-bottom: 0.5rem !important;}
    .stButton > button {width: 100%; font-weight: bold; height: 46px; font-size: 18px;}
    div.stProgress > div > div {height: 8px !important;}
    .metric-container {
        background-color: rgba(28, 131, 225, 0.1);
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .metric-title {
        font-weight: bold;
        margin-bottom: 5px;
    }
    .info-box {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .success {
        background-color: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .warning {
        background-color: #fff3cd;
        color: #856404;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .error {
        background-color: #f8d7da;
        color: #721c24;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Database configuration
DB_CONFIG = {
    'rollbit': {
        'url': "postgresql://public_rw:aTJ92^kl04hllk@aws-jp-tk-surf-pg-public.cluster-csteuf9lw8dv.ap-northeast-1.rds.amazonaws.com:5432/report_dev"
    },
    'surf': {
        'url': "postgresql://public_replication:866^FKC4hllk@aws-jp-tk-surf-pg-public.cluster-csteuf9lw8dv.ap-northeast-1.rds.amazonaws.com:5432/replication_report"
    }
}

# Create database engines
@st.cache_resource
def get_engine(platform='surf'):
    """Create database engine for the specified platform"""
    try:
        return create_engine(DB_CONFIG[platform]['url'], pool_size=5, max_overflow=10)
    except Exception as e:
        st.error(f"Error creating {platform} database engine: {e}")
        return None

@contextmanager
def get_session(platform='surf'):
    """Database session context manager for the specified platform"""
    engine = get_engine(platform)
    if not engine:
        yield None
        return
        
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    except Exception as e:
        st.error(f"Database error for {platform}: {e}")
        session.rollback()
    finally:
        session.close()

# Pre-defined pairs as a fallback
PREDEFINED_PAIRS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", 
    "AVAX/USDT", "DOGE/USDT", "ADA/USDT", "TRX/USDT", "DOT/USDT"
]

# Get available pairs from database
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_available_pairs():
    try:
        pairs = set()
        
        # Try to get pairs from Surf
        with get_session('surf') as session:
            if session:
                # Try the trade_pool_pairs table first
                try:
                    query = text("SELECT pair_name FROM trade_pool_pairs WHERE status = 1")
                    result = session.execute(query)
                    pairs_surf = [row[0] for row in result]
                    if pairs_surf:
                        return sorted(pairs_surf)
                except:
                    pass
                
                # If that doesn't work, try to get pairs from partition tables
                singapore_tz = pytz.timezone('Asia/Singapore')
                today = datetime.now(singapore_tz).strftime("%Y%m%d")
                yesterday = (datetime.now(singapore_tz) - timedelta(days=1)).strftime("%Y%m%d")
                
                for date in [today, yesterday]:
                    try:
                        table_name = f"oracle_price_log_partition_{date}"
                        query = text(f"""
                            SELECT DISTINCT pair_name 
                            FROM public."{table_name}" 
                            LIMIT 50
                        """)
                        result = session.execute(query)
                        pairs_from_table = [row[0] for row in result]
                        if pairs_from_table:
                            pairs.update(pairs_from_table)
                            break
                    except:
                        continue
        
        # If we found any pairs, return them, otherwise use predefined list
        if pairs:
            return sorted(list(pairs))
        return PREDEFINED_PAIRS
    
    except Exception as e:
        st.error(f"Error fetching available pairs: {e}")
        return PREDEFINED_PAIRS

# Check if a table exists
def table_exists(session, table_name):
    """Check if a table exists in the database"""
    check_query = text("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name = :table_name
        );
    """)
    return session.execute(check_query, {"table_name": table_name}).scalar()

# Get available partition tables
def get_partition_tables(session, platform, start_date, end_date):
    """
    Get list of partition tables that need to be queried based on date range.
    Returns a list of table names
    """
    # Ensure we have datetime objects
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    # Ensure timezone is properly set to Singapore
    singapore_tz = pytz.timezone('Asia/Singapore')
    if start_date.tzinfo is None:
        start_date = singapore_tz.localize(start_date)
    if end_date.tzinfo is None:
        end_date = singapore_tz.localize(end_date)
    
    # Convert to Singapore time
    start_date = start_date.astimezone(singapore_tz)
    end_date = end_date.astimezone(singapore_tz)
    
    # Generate list of dates between start and end
    dates = []
    current_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_date_day = end_date.replace(hour=0, minute=0, second=0, microsecond=0)
    
    while current_date <= end_date_day:
        dates.append(current_date.strftime("%Y%m%d"))
        current_date += timedelta(days=1)
    
    # Create table names
    table_names = [f"oracle_price_log_partition_{date}" for date in dates]
    
    # Verify which tables actually exist
    existing_tables = []
    
    for table in table_names:
        if table_exists(session, table):
            existing_tables.append(table)
    
    return existing_tables

# Fetch tick data with timestamps for a specific time period
def fetch_tick_data(pair_name, platform, hours=3, min_ticks=20000, progress_callback=None):
    """
    Fetch tick data for a specific pair from the specified platform.
    Ensures we get at least min_ticks data points or data from the last hours, whichever is more.
    """
    try:
        with get_session(platform) as session:
            if not session:
                return None
            
            # Calculate time range in Singapore time
            singapore_tz = pytz.timezone('Asia/Singapore')
            now = datetime.now(singapore_tz)
            start_time = now - timedelta(hours=hours)
            
            # Format for display and database
            start_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
            end_str = now.strftime("%Y-%m-%d %H:%M:%S")
            
            if progress_callback:
                progress_callback(0.1, f"Fetching {platform} data from {start_str} to {end_str} (SGT)")
            
            # Get relevant partition tables
            tables = get_partition_tables(session, platform, start_time, now)
            
            if progress_callback:
                progress_callback(0.2, f"Found {len(tables)} {platform} partition tables")
            
            if not tables:
                if progress_callback:
                    progress_callback(0.2, f"No {platform} tables found for this time range")
                return None
            
            # Build query for each table
            union_parts = []
            
            for table in tables:
                # For Surf data 
                if platform == 'surf':
                    query = f"""
                    SELECT 
                        pair_name,
                        created_at + INTERVAL '8 hour' AS timestamp_sgt,
                        final_price AS price
                    FROM 
                        public."{table}"
                    WHERE 
                        created_at >= '{start_str}'::timestamp - INTERVAL '8 hour'
                        AND created_at <= '{end_str}'::timestamp - INTERVAL '8 hour'
                        AND source_type = 0
                        AND pair_name = '{pair_name}'
                    """
                else:
                    # For Rollbit data
                    query = f"""
                    SELECT 
                        pair_name,
                        created_at + INTERVAL '8 hour' AS timestamp_sgt,
                        final_price AS price
                    FROM 
                        public."{table}"
                    WHERE 
                        created_at >= '{start_str}'::timestamp - INTERVAL '8 hour'
                        AND created_at <= '{end_str}'::timestamp - INTERVAL '8 hour'
                        AND source_type = 1
                        AND pair_name = '{pair_name}'
                    """
                union_parts.append(query)
            
            # Join with UNION and add ORDER BY
            complete_query = " UNION ".join(union_parts) + " ORDER BY timestamp_sgt DESC"  # DESC to get newest first
            
            # Add LIMIT to get at least min_ticks
            limited_query = complete_query + f" LIMIT {min_ticks * 2}"  # Get more than needed to account for potential duplicates
            
            if progress_callback:
                progress_callback(0.4, f"Executing query for {platform}...")
            
            # Execute query
            result = session.execute(text(limited_query))
            data = result.fetchall()
            
            if not data:
                if progress_callback:
                    progress_callback(0.5, f"No {platform} data found for {pair_name}")
                return None
            
            # Create DataFrame
            df = pd.DataFrame(data, columns=['pair_name', 'timestamp_sgt', 'price'])
            
            # Convert price to numeric
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            
            # Sort by timestamp (ascending)
            df = df.sort_values('timestamp_sgt', ascending=True)
            
            # Remove duplicates
            df = df.drop_duplicates(subset=['timestamp_sgt'])
            
            if progress_callback:
                progress_callback(0.8, f"Processing {len(df)} {platform} ticks...")
            
            return df
    
    except Exception as e:
        if progress_callback:
            progress_callback(1.0, f"{platform} error: {str(e)}")
        st.error(f"Error fetching {platform} tick data: {e}")
        return None

# Calculate choppiness for a dataframe EXACTLY like parameters.py does
def calculate_params_choppiness(prices, window_size=20):
    """
    Calculate choppiness exactly like parameters.py implementation.
    
    Args:
        prices: Series of price values
        window_size: Size of the rolling window
    
    Returns:
        Average choppiness value
    """
    try:
        # Convert to Series if it's not already
        if not isinstance(prices, pd.Series):
            prices = pd.Series(prices)
        
        # Calculate absolute differences
        diff = prices.diff().abs()
        
        # Calculate rolling sum of absolute changes
        sum_abs_changes = diff.rolling(window=window_size, min_periods=1).sum()
        
        # Calculate rolling price range (max - min)
        price_range = prices.rolling(window=window_size, min_periods=1).max() - prices.rolling(window=window_size, min_periods=1).min()
        
        # Handle zero price range
        if (price_range == 0).any():
            price_range = price_range.replace(0, 1e-10)
        
        # Calculate choppiness
        epsilon = 1e-10
        choppiness = 100 * sum_abs_changes / (price_range + epsilon)
        
        # Cap extreme values and handle NaN
        choppiness = np.minimum(choppiness, 1000)
        choppiness = choppiness.fillna(200)
        
        # Return mean
        return choppiness.mean()
    except Exception as e:
        st.error(f"Error in choppiness calculation: {e}")
        return 200.0  # Same default as parameters.py

# Calculate synchronized choppiness at intervals
def calculate_synchronized_choppiness(rollbit_df, surf_df, window_size=20, tick_count=5000, num_points=10, interval_minutes=5):
    """
    Calculate choppiness for both exchanges at exactly the same timestamps.
    Uses exact parameters.py implementation.
    
    Args:
        rollbit_df: DataFrame with Rollbit data
        surf_df: DataFrame with Surf data
        window_size: Size of the rolling window (default: 20)
        tick_count: Number of ticks to use for calculation (default: 5000)
        num_points: Number of points to calculate (default: 10)
        interval_minutes: Minutes between points (default: 5)
    
    Returns:
        Tuple of (timestamps, rollbit_choppiness, surf_choppiness)
    """
    # Validation checks
    if rollbit_df is None or surf_df is None:
        st.error("Missing data for one or both exchanges")
        return [], [], []
    
    if len(rollbit_df) < tick_count or len(surf_df) < tick_count:
        st.error(f"Not enough data: Rollbit has {len(rollbit_df)} ticks, Surf has {len(surf_df)} ticks. Need {tick_count}.")
        return [], [], []
    
    try:
        # Sort dataframes by timestamp
        rollbit_df = rollbit_df.sort_values('timestamp_sgt', ascending=True)
        surf_df = surf_df.sort_values('timestamp_sgt', ascending=True)
        
        # Get the most recent common timestamp (use the earlier of the two latest timestamps)
        latest_rollbit = rollbit_df['timestamp_sgt'].iloc[-1]
        latest_surf = surf_df['timestamp_sgt'].iloc[-1]
        latest_common = min(latest_rollbit, latest_surf)
        
        # Create evenly spaced timestamps going backwards
        timestamps = []
        for i in range(num_points):
            timestamps.append(latest_common - timedelta(minutes=interval_minutes * (num_points - 1 - i)))
        
        # Calculate choppiness for each exchange at each timestamp
        rollbit_values = []
        surf_values = []
        valid_timestamps = []
        
        for timestamp in timestamps:
            # Get Rollbit data before this timestamp
            rollbit_mask = rollbit_df['timestamp_sgt'] <= timestamp
            rollbit_previous = rollbit_df[rollbit_mask]
            
            # Get Surf data before this timestamp
            surf_mask = surf_df['timestamp_sgt'] <= timestamp
            surf_previous = surf_df[surf_mask]
            
            # Skip if either exchange doesn't have enough data
            if len(rollbit_previous) < tick_count or len(surf_previous) < tick_count:
                continue
            
            # Get most recent tick_count ticks for each exchange
            rollbit_recent = rollbit_previous.iloc[-tick_count:]
            surf_recent = surf_previous.iloc[-tick_count:]
            
            # Calculate choppiness using parameters.py method
            rollbit_choppiness = calculate_params_choppiness(rollbit_recent['price'], window_size)
            surf_choppiness = calculate_params_choppiness(surf_recent['price'], window_size)
            
            # Store results
            valid_timestamps.append(timestamp)
            rollbit_values.append(rollbit_choppiness)
            surf_values.append(surf_choppiness)
            
            # Debug - print values to verify
            # st.write(f"Timestamp: {timestamp}, Rollbit: {rollbit_choppiness}, Surf: {surf_choppiness}")
        
        return valid_timestamps, rollbit_values, surf_values
    
    except Exception as e:
        st.error(f"Error calculating synchronized choppiness: {str(e)}")
        st.error(traceback.format_exc())
        return [], [], []

# Main app
def main():
    # Get current Singapore time
    singapore_tz = pytz.timezone('Asia/Singapore')
    now_sg = datetime.now(singapore_tz)
    current_time_sg = now_sg.strftime("%Y-%m-%d %H:%M:%S")
    
    # Main title
    st.title("5-Minute Interval Choppiness: Rollbit vs Surf")
    st.markdown(f"<p style='text-align: center; font-size:14px; color:gray;'>Last updated: {current_time_sg} (SGT)</p>", unsafe_allow_html=True)
    
    st.markdown("Comparison of 5000-tick choppiness at 5-minute intervals. Each point represents the average 20-tick choppiness across 5000 ticks.")
    
    # Get available pairs
    available_pairs = get_available_pairs()
    
    # Create columns for layout
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Selection UI
        selected_pair = st.selectbox(
            "Select cryptocurrency pair",
            options=available_pairs,
            index=0 if "BTC/USDT" in available_pairs else 0
        )
    
    with col2:
        # Interval selector
        interval_minutes = st.number_input(
            "Minutes between points",
            min_value=1,
            max_value=30,
            value=5,
            help="Time interval between data points (minutes)"
        )
        
    with col3:
        # Hours to fetch
        hours_to_fetch = st.number_input(
            "Hours of data to fetch",
            min_value=1,
            max_value=24,
            value=3,
            help="How many hours of historical data to fetch"
        )
    
    # Create plot button
    if st.button("Generate 5-Minute Interval Choppiness Plot", use_container_width=True):
        # Progress bars for data fetching
        col_rollbit, col_surf = st.columns(2)
        
        with col_rollbit:
            st.write("Rollbit")
            progress_bar_rollbit = st.progress(0)
            
            def update_rollbit(progress, text):
                progress_bar_rollbit.progress(progress, text)
        
        with col_surf:
            st.write("Surf")
            progress_bar_surf = st.progress(0)
            
            def update_surf(progress, text):
                progress_bar_surf.progress(progress, text)
        
        # Fetch data for both platforms
        with st.spinner(f"Fetching {hours_to_fetch} hours of tick data..."):
            # We want to fetch enough data for all our points plus 5000 ticks for each
            min_ticks_needed = max(20000, (interval_minutes * 10) * 100)  # Estimate roughly 100 ticks per minute
            rollbit_df = fetch_tick_data(selected_pair, 'rollbit', hours_to_fetch, min_ticks_needed, update_rollbit)
            surf_df = fetch_tick_data(selected_pair, 'surf', hours_to_fetch, min_ticks_needed, update_surf)
        
        # Process data if available
        if rollbit_df is not None and surf_df is not None:
            st.success(f"Fetched {len(rollbit_df)} ticks from Rollbit and {len(surf_df)} ticks from Surf")
            
            # Fixed window size of 20 for choppiness calculation
            window_size = 20
            tick_count = 5000  # Use 5000 ticks for each calculation
            num_points = 10   # Calculate 10 points
            
            # Calculate synchronized choppiness with error handling
            with st.spinner(f"Calculating synchronized choppiness at {interval_minutes}-minute intervals..."):
                try:
                    timestamps, rollbit_chop, surf_chop = calculate_synchronized_choppiness(
                        rollbit_df, surf_df, window_size, tick_count, num_points, interval_minutes)
                    
                    # Log the calculation for debugging
                    st.info(f"Generated {len(timestamps)} synchronized timestamps")
                    
                    if len(timestamps) == 0:
                        st.error("No timestamps were generated. There may not be enough data.")
                        
                except Exception as e:
                    st.error(f"Error in choppiness calculation: {str(e)}")
                    st.error(traceback.format_exc())
                    timestamps, rollbit_chop, surf_chop = [], [], []
            
            # Check if we got any data points
            if len(timestamps) > 0 and len(rollbit_chop) > 0 and len(surf_chop) > 0:
                # Calculate the most recent (latest) values
                rollbit_latest = rollbit_chop[-1] if rollbit_chop else None
                surf_latest = surf_chop[-1] if surf_chop else None
                
                # Display latest values (most recent time point)
                st.subheader("Latest 5000-Tick Choppiness Values")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Rollbit",
                        f"{rollbit_latest:.2f}" if rollbit_latest is not None else "N/A", 
                        f"{rollbit_latest - surf_latest:.2f} vs Surf" if (rollbit_latest is not None and surf_latest is not None) else "N/A"
                    )
                with col2:
                    st.metric(
                        "Surf",
                        f"{surf_latest:.2f}" if surf_latest is not None else "N/A",
                        f"{surf_latest - rollbit_latest:.2f} vs Rollbit" if (rollbit_latest is not None and surf_latest is not None) else "N/A"
                    )
                
                # Create plot
                fig = go.Figure()
                
                # Format timestamps for better display
                formatted_timestamps = [t.strftime("%H:%M:%S") for t in timestamps]
                
                # Add Rollbit line
                fig.add_trace(go.Scatter(
                    x=formatted_timestamps,
                    y=rollbit_chop,
                    mode='lines+markers',
                    name='Rollbit Choppiness',
                    line=dict(color='red', width=2),
                    marker=dict(size=10, color='red'),
                    hoverinfo='text',
                    hovertext=[f'Time: {t}<br>Choppiness: {v:.2f}' 
                              for t, v in zip(formatted_timestamps, rollbit_chop)]
                ))
                
                # Add Surf line
                fig.add_trace(go.Scatter(
                    x=formatted_timestamps,
                    y=surf_chop,
                    mode='lines+markers',
                    name='Surf Choppiness',
                    line=dict(color='blue', width=2),
                    marker=dict(size=10, color='blue'),
                    hoverinfo='text',
                    hovertext=[f'Time: {t}<br>Choppiness: {v:.2f}' 
                              for t, v in zip(formatted_timestamps, surf_chop)]
                ))
                
                # Add horizontal line for reference value 250
                fig.add_shape(
                    type="line",
                    x0=0,
                    y0=250,
                    x1=1,
                    y1=250,
                    xref="paper",
                    line=dict(color="grey", width=1, dash="dash"),
                )
                
                # Calculate y-axis range with a 10% buffer
                all_values = rollbit_chop + surf_chop
                if all_values:
                    min_val = min(all_values) * 0.9
                    max_val = max(all_values) * 1.1
                else:
                    min_val, max_val = 100, 400
                
                # Add annotations for latest values on the chart
                fig.add_annotation(
                    xref='paper',
                    x=1.0,
                    y=rollbit_latest,
                    text=f"Latest Rollbit: {rollbit_latest:.2f}",
                    showarrow=True,
                    arrowhead=7,
                    ax=50,
                    ay=0,
                    font=dict(color="red"),
                    bgcolor="white",
                    bordercolor="red",
                    borderwidth=1
                )
                
                fig.add_annotation(
                    xref='paper',
                    x=1.0,
                    y=surf_latest,
                    text=f"Latest Surf: {surf_latest:.2f}",
                    showarrow=True,
                    arrowhead=7,
                    ax=50,
                    ay=0,
                    font=dict(color="blue"),
                    bgcolor="white",
                    bordercolor="blue",
                    borderwidth=1
                )
                
                # Layout
                fig.update_layout(
                    title=f"5000-Tick Choppiness at {interval_minutes}-Minute Intervals: {selected_pair}",
                    xaxis_title="Time (SGT)",
                    yaxis_title="5000-Tick Choppiness (20-tick window average)",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    height=600,
                    hovermode="closest",
                    yaxis=dict(
                        range=[min_val, max_val],
                    ),
                    plot_bgcolor='white',  # White background
                    xaxis=dict(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(220,220,220,0.5)',
                        zeroline=False,
                    ),
                    yaxis_gridwidth=1,
                    yaxis_gridcolor='rgba(220,220,220,0.5)',
                    margin=dict(l=50, r=50, t=80, b=50),
                )
                
                # Show plot
                st.plotly_chart(fig, use_container_width=True)
                
                # Display data collection time ranges
                st.subheader("Data Collection Time Ranges")
                time_data = []
                
                for i, timestamp in enumerate(timestamps):
                    # Get the corresponding 5000 ticks before this timestamp for both exchanges
                    rollbit_mask = rollbit_df['timestamp_sgt'] <= timestamp
                    rollbit_previous = rollbit_df[rollbit_mask]
                    rollbit_recent = rollbit_previous.iloc[-tick_count:] if len(rollbit_previous) >= tick_count else None
                    
                    surf_mask = surf_df['timestamp_sgt'] <= timestamp
                    surf_previous = surf_df[surf_mask]
                    surf_recent = surf_previous.iloc[-tick_count:] if len(surf_previous) >= tick_count else None
                    
                    if rollbit_recent is not None and surf_recent is not None:
                        time_data.append({
                            'Pair': selected_pair,
                            'Rollbit Start': rollbit_recent['timestamp_sgt'].iloc[0],
                            'Rollbit End': rollbit_recent['timestamp_sgt'].iloc[-1],
                            'Rollbit Count': len(rollbit_recent),
                            'Surf Start': surf_recent['timestamp_sgt'].iloc[0],
                            'Surf End': surf_recent['timestamp_sgt'].iloc[-1],
                            'Surf Count': len(surf_recent)
                        })
                
                if time_data:
                    time_df = pd.DataFrame(time_data)
                    
                    # Format datetime columns to be more readable
                    for col in time_df.columns:
                        if 'Start' in col or 'End' in col:
                            try:
                                time_df[col] = pd.to_datetime(time_df[col]).dt.strftime('%Y-%m-%d %H:%M:%S')
                            except:
                                pass
                    
                    st.dataframe(time_df, use_container_width=True)
                    st.info("Note: 'Start' is the oldest data point, 'End' is the most recent data point in the analysis.")
                
                # Display table of values
                st.subheader("Choppiness Values")
                
                # Create a dataframe with all the data points
                data = []
                
                for i in range(len(timestamps)):
                    formatted_time = timestamps[i].strftime("%H:%M:%S")
                    rollbit_value = rollbit_chop[i]
                    surf_value = surf_chop[i]
                    
                    data.append({
                        "Time": formatted_time,
                        "Rollbit": f"{rollbit_value:.2f}",
                        "Surf": f"{surf_value:.2f}",
                        "Difference": f"{(rollbit_value - surf_value):.2f}"
                    })
                
                # Create a DataFrame and display it
                df_results = pd.DataFrame(data)
                st.dataframe(df_results, use_container_width=True)
                
                # Add explanation of the chart
                with st.expander("About This Choppiness Plot"):
                    st.markdown("""
                    **How Choppiness Is Calculated:**
                    
                    1. For each point in time (spaced 5 minutes apart):
                       - We take the most recent 5000 ticks up to that point
                       - Calculate the choppiness using the exact same method as in parameters.py
                       - The calculation uses a 20-tick rolling window across all 5000 ticks
                    
                    2. The formula for calculating choppiness is:
                       `choppiness = 100 * sum_abs_changes / (price_range + epsilon)`
                       - Where `sum_abs_changes` is the sum of all absolute price changes in the window
                       - And `price_range` is the difference between max and min prices in the window
                    
                    3. The latest values (rightmost points) exactly match the values shown in parameters.py.
                    
                    **Interpretation:**
                    - Values below 200: Low choppiness (trending market)
                    - Values around 250: Moderate choppiness
                    - Values above 300: High choppiness (sideways/volatile market)
                    """)
            else:
                st.error("Not enough data to calculate choppiness at the requested intervals")
                st.info(f"""
                We need at least {tick_count} ticks before each time point for both exchanges.
                Try:
                1. Increasing the 'Hours of data to fetch'
                2. Reducing the interval between points
                3. Selecting a more actively traded pair
                """)
        else:
            if rollbit_df is None:
                st.error(f"Could not fetch Rollbit data for {selected_pair}")
            if surf_df is None:
                st.error(f"Could not fetch Surf data for {selected_pair}")

if __name__ == "__main__":
    main()