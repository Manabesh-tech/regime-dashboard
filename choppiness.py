import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
import concurrent.futures

# Clear cache at startup to ensure fresh data
if hasattr(st, 'cache_data'):
    st.cache_data.clear()

# Page configuration
st.set_page_config(page_title="Choppiness Time Plot", layout="wide")

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

# Fetch tick data - improved to ensure we get the last 12 hours AND at least 5000 ticks
def fetch_tick_data(pair_name, platform, hours=12, min_ticks=5000, progress_callback=None):
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
            
            # Check if we have enough data
            if len(df) < min_ticks:
                # Not enough data, try to get more by extending the time range
                if progress_callback:
                    progress_callback(0.6, f"Only found {len(df)} ticks. Extending time range...")
                
                # Try to get more historical data
                extended_start_time = start_time - timedelta(hours=24)  # Go back further
                extended_start_str = extended_start_time.strftime("%Y-%m-%d %H:%M:%S")
                
                # Get additional partition tables
                additional_tables = get_partition_tables(session, platform, extended_start_time, start_time)
                
                if additional_tables:
                    # Build additional queries
                    additional_union_parts = []
                    
                    for table in additional_tables:
                        query = f"""
                        SELECT 
                            pair_name,
                            created_at + INTERVAL '8 hour' AS timestamp_sgt,
                            final_price AS price
                        FROM 
                            public."{table}"
                        WHERE 
                            created_at >= '{extended_start_str}'::timestamp - INTERVAL '8 hour'
                            AND created_at < '{start_str}'::timestamp - INTERVAL '8 hour'
                            AND pair_name = '{pair_name}'
                        """
                        additional_union_parts.append(query)
                    
                    if additional_union_parts:
                        # Join with UNION and add ORDER BY
                        additional_query = " UNION ".join(additional_union_parts) + " ORDER BY timestamp_sgt DESC"
                        additional_query = additional_query + f" LIMIT {min_ticks * 2}"  # Get more than needed
                        
                        # Execute additional query
                        additional_result = session.execute(text(additional_query))
                        additional_data = additional_result.fetchall()
                        
                        if additional_data:
                            # Create DataFrame for additional data
                            additional_df = pd.DataFrame(additional_data, columns=['pair_name', 'timestamp_sgt', 'price'])
                            additional_df['price'] = pd.to_numeric(additional_df['price'], errors='coerce')
                            
                            # Combine with original data
                            combined_df = pd.concat([additional_df, df])
                            
                            # Sort and remove duplicates
                            combined_df = combined_df.sort_values('timestamp_sgt', ascending=True)
                            combined_df = combined_df.drop_duplicates(subset=['timestamp_sgt'])
                            
                            df = combined_df
            
            # Keep only the last min_ticks rows if we have more than needed
            if len(df) > min_ticks:
                df = df.iloc[-min_ticks:]
            
            if progress_callback:
                progress_callback(0.8, f"Processing {len(df)} {platform} ticks...")
            
            return df
    
    except Exception as e:
        if progress_callback:
            progress_callback(1.0, f"{platform} error: {str(e)}")
        st.error(f"Error fetching {platform} tick data: {e}")
        return None

# Calculate choppiness with a more aggressive scaling to match 100-400 range
# Calculate choppiness using the second application's formula
def calculate_choppiness(df, window_size=20):
    """
    Calculate choppiness values using the formula from the second application:
    choppiness = 100 * sum_abs_changes / (price_range + epsilon)
    
    Args:
        df: DataFrame with price data
        window_size: Size of the rolling window
    
    Returns:
        Tuple of (timestamps, choppiness_values)
    """
    if df is None or len(df) < window_size + 1:
        return [], []
    
    # Ensure data is sorted
    df = df.sort_values('timestamp_sgt')
    
    # Initialize lists for results
    timestamps = []
    choppiness_values = []
    
    # Avoid division by zero
    epsilon = 1e-10
    
    # Calculate rolling choppiness for each window
    for i in range(window_size, len(df)):
        window_df = df.iloc[i-window_size:i].copy()
        prices = window_df['price'].values
        
        # Calculate sum of absolute changes
        price_changes = np.abs(np.diff(prices))
        sum_abs_changes = np.sum(price_changes)
        
        # Calculate price range in the window
        price_range = np.max(prices) - np.min(prices)
        
        # Calculate choppiness using the second application's formula
        choppiness = 100 * sum_abs_changes / (price_range + epsilon)
        
        # Cap extreme values to match range of original app (100-400)
        choppiness = max(100, min(400, choppiness))
        
        # Store results
        timestamps.append(df['timestamp_sgt'].iloc[i])
        choppiness_values.append(choppiness)
    
    return timestamps, choppiness_values

# Time-based aggregation function
def aggregate_by_time(timestamps, values, interval_minutes=5):
    """
    Aggregate choppiness values into regular time intervals.
    
    Args:
        timestamps: List of datetime objects
        values: List of corresponding values
        interval_minutes: Time interval in minutes for aggregation
    
    Returns:
        Tuple of (aggregated_timestamps, aggregated_values)
    """
    if not timestamps or not values:
        return [], []
    
    # Create DataFrame for easier manipulation
    df = pd.DataFrame({'timestamp': timestamps, 'value': values})
    
    # Convert to datetime if not already
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create time buckets based on interval_minutes
    df['time_bucket'] = df['timestamp'].dt.floor(f'{interval_minutes}min')
    
    # Group by time bucket and aggregate
    aggregated = df.groupby('time_bucket').agg(
        value=('value', 'mean'),
        min_value=('value', 'min'),
        max_value=('value', 'max'),
        count=('value', 'count')
    ).reset_index()
    
    # Return the aggregated values
    return aggregated['time_bucket'].tolist(), aggregated['value'].tolist(), aggregated[['min_value', 'max_value', 'count']].to_dict('records')

# Main app
def main():
    # Get current Singapore time
    singapore_tz = pytz.timezone('Asia/Singapore')
    now_sg = datetime.now(singapore_tz)
    current_time_sg = now_sg.strftime("%Y-%m-%d %H:%M:%S")
    
    # Main title
    st.title("Choppiness Time Plot: Rollbit vs Surf")
    st.markdown(f"<p style='text-align: center; font-size:14px; color:gray;'>Last updated: {current_time_sg} (SGT)</p>", unsafe_allow_html=True)
    
    st.markdown("Comparison of price choppiness over the last 12 hours using 5000 ticks and a rolling window of 20 ticks")
    
    # Get available pairs
    available_pairs = get_available_pairs()
    
    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Selection UI
        selected_pair = st.selectbox(
            "Select cryptocurrency pair",
            options=available_pairs,
            index=0 if "SOL/USDT" in available_pairs else 0
        )
    
    with col2:
        # Time interval selection
        time_interval = st.selectbox(
            "Time Interval",
            options=[1, 2, 3, 5, 10, 15, 30],
            index=2,  # Default to 5 minutes
            help="Aggregate data into time intervals (minutes)"
        )
    
    # Create plot button
    if st.button("Generate Choppiness Plot", use_container_width=True):
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
        
        # Fetch data for both platforms with a minimum of 5000 ticks
        with st.spinner("Fetching tick data..."):
            rollbit_df = fetch_tick_data(selected_pair, 'rollbit', 12, 5000, update_rollbit)
            surf_df = fetch_tick_data(selected_pair, 'surf', 12, 5000, update_surf)
        
        # Process data if available
        if rollbit_df is not None and surf_df is not None:
            st.success(f"Fetched {len(rollbit_df)} ticks from Rollbit and {len(surf_df)} ticks from Surf")
            
            # Fixed window size of 20
            window_size = 20
            
            # Check if we have enough data
            if len(rollbit_df) < window_size + 1 or len(surf_df) < window_size + 1:
                st.warning(f"Not enough data for window size {window_size}. Rollbit: {len(rollbit_df)} ticks, Surf: {len(surf_df)} ticks")
                if min(len(rollbit_df), len(surf_df)) < 5:
                    st.error("Insufficient data to calculate choppiness. Please try a different pair or fetch more data.")
                    return
            
            # Calculate choppiness with progress indication
            with st.spinner("Calculating choppiness metrics..."):
                # Calculate choppiness for both platforms
                rollbit_times, rollbit_chop = calculate_choppiness(rollbit_df, window_size)
                surf_times, surf_chop = calculate_choppiness(surf_df, window_size)
            
            if len(rollbit_times) > 0 and len(surf_times) > 0:
                # Time-based aggregation
                with st.spinner(f"Aggregating data into {time_interval}-minute intervals..."):
                    # Aggregate data into time intervals
                    rollbit_agg_times, rollbit_agg_values, rollbit_agg_stats = aggregate_by_time(rollbit_times, rollbit_chop, time_interval)
                    surf_agg_times, surf_agg_values, surf_agg_stats = aggregate_by_time(surf_times, surf_chop, time_interval)
                
                # Create plot
                fig = go.Figure()
                
                # Add Rollbit line
                fig.add_trace(go.Scatter(
                    x=rollbit_agg_times,
                    y=rollbit_agg_values,
                    mode='lines+markers',
                    name='Rollbit Choppiness',
                    line=dict(color='red', width=2),
                    marker=dict(size=8, color='red'),
                    hoverinfo='text',
                    hovertext=[f'Time: {t.strftime("%H:%M:%S")}<br>Value: {v:.2f}<br>Min: {s["min_value"]:.2f}<br>Max: {s["max_value"]:.2f}<br>Points: {s["count"]}' 
                              for t, v, s in zip(rollbit_agg_times, rollbit_agg_values, rollbit_agg_stats)]
                ))
                
                # Add Surf line
                fig.add_trace(go.Scatter(
                    x=surf_agg_times,
                    y=surf_agg_values,
                    mode='lines+markers',
                    name='Surf Choppiness',
                    line=dict(color='blue', width=2),
                    marker=dict(size=8, color='blue'),
                    hoverinfo='text',
                    hovertext=[f'Time: {t.strftime("%H:%M:%S")}<br>Value: {v:.2f}<br>Min: {s["min_value"]:.2f}<br>Max: {s["max_value"]:.2f}<br>Points: {s["count"]}' 
                              for t, v, s in zip(surf_agg_times, surf_agg_values, surf_agg_stats)]
                ))
                
                # Layout
                fig.update_layout(
                    title=f"Choppiness Over Time: {selected_pair} ({time_interval}-Minute Intervals, Window Size: 20 ticks)",
                    xaxis_title="Time (SGT)",
                    yaxis_title="Choppiness (100-400 scale)",
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
                        range=[95, 405],  # Slightly wider than 100-400 to show boundary values
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
                
                # Add horizontal reference lines
                fig.add_shape(
                    type="line",
                    x0=min(rollbit_agg_times[0], surf_agg_times[0]) if rollbit_agg_times and surf_agg_times else 0,
                    y0=100,
                    x1=max(rollbit_agg_times[-1], surf_agg_times[-1]) if rollbit_agg_times and surf_agg_times else 1,
                    y1=100,
                    line=dict(color="rgba(0,0,0,0.3)", width=1, dash="dot"),
                )
                
                fig.add_shape(
                    type="line",
                    x0=min(rollbit_agg_times[0], surf_agg_times[0]) if rollbit_agg_times and surf_agg_times else 0,
                    y0=250,
                    x1=max(rollbit_agg_times[-1], surf_agg_times[-1]) if rollbit_agg_times and surf_agg_times else 1,
                    y1=250,
                    line=dict(color="rgba(0,0,0,0.3)", width=1, dash="dot"),
                )
                
                fig.add_shape(
                    type="line",
                    x0=min(rollbit_agg_times[0], surf_agg_times[0]) if rollbit_agg_times and surf_agg_times else 0,
                    y0=400,
                    x1=max(rollbit_agg_times[-1], surf_agg_times[-1]) if rollbit_agg_times and surf_agg_times else 1,
                    y1=400,
                    line=dict(color="rgba(0,0,0,0.3)", width=1, dash="dot"),
                )
                
                # Add annotation for scale meaning
                fig.add_annotation(
                    x=0.01,
                    y=0.05,
                    xref="paper",
                    yref="paper",
                    text="Lower values = Less choppy, Higher values = More choppy",
                    showarrow=False,
                    font=dict(size=12, color="gray"),
                    bgcolor="white",
                    bordercolor="gray",
                    borderwidth=1,
                    borderpad=4,
                    opacity=0.8
                )
                
                # Show plot
                st.plotly_chart(fig, use_container_width=True)
                
                # Display metrics in columns
                col1, col2 = st.columns(2)
                
                # Calculate metrics
                rollbit_avg = np.mean(rollbit_chop) if rollbit_chop else 0
                surf_avg = np.mean(surf_chop) if surf_chop else 0
                rollbit_min = np.min(rollbit_chop) if rollbit_chop else 0
                rollbit_max = np.max(rollbit_chop) if rollbit_chop else 0
                surf_min = np.min(surf_chop) if surf_chop else 0
                surf_max = np.max(surf_chop) if surf_chop else 0
                
                # Display metrics
                with col1:
                    st.markdown("### Rollbit Metrics")
                    st.metric(
                        "Average Choppiness", 
                        f"{rollbit_avg:.2f}", 
                        f"{rollbit_avg - surf_avg:.2f} vs Surf"
                    )
                    st.markdown(f"**Range:** {rollbit_min:.2f} to {rollbit_max:.2f}")
                    st.markdown(f"**Latest Value:** {rollbit_chop[-1]:.2f}")
                    
                    # Interpret Rollbit choppiness
                    if rollbit_avg < 175:
                        status = "Low choppiness - trending market"
                    elif rollbit_avg < 325:
                        status = "Moderate choppiness - normal market"
                    else:
                        status = "High choppiness - sideways/volatile market"
                    
                    st.markdown(f"**Status:** {status}")
                
                with col2:
                    st.markdown("### Surf Metrics")
                    st.metric(
                        "Average Choppiness", 
                        f"{surf_avg:.2f}", 
                        f"{surf_avg - rollbit_avg:.2f} vs Rollbit"
                    )
                    st.markdown(f"**Range:** {surf_min:.2f} to {surf_max:.2f}")
                    st.markdown(f"**Latest Value:** {surf_chop[-1]:.2f}")
                    
                    # Interpret Surf choppiness
                    if surf_avg < 175:
                        status = "Low choppiness - trending market"
                    elif surf_avg < 325:
                        status = "Moderate choppiness - normal market"
                    else:
                        status = "High choppiness - sideways/volatile market"
                    
                    st.markdown(f"**Status:** {status}")
                
                # Display comparison
                st.markdown("### Comparison Analysis")
                difference = abs(rollbit_avg - surf_avg)
                if difference < 10:
                    st.success("The choppiness levels are very similar between platforms (difference < 10)")
                elif difference < 30:
                    st.info("The choppiness levels show moderate differences between platforms (difference < 30)")
                else:
                    st.warning(f"The choppiness levels are significantly different between platforms (difference = {difference:.2f})")
                    
                    # Identify which platform is more choppy
                    more_choppy = "Rollbit" if rollbit_avg > surf_avg else "Surf"
                    st.markdown(f"**{more_choppy}** shows higher price choppiness, which may indicate more frequent price reversals or market noise.")
                
                # Add explanation of the chart
                with st.expander("About Choppiness Calculation"):
                    st.markdown("""
                    **How Choppiness is Calculated:**
                    
                    1. For each window of 20 ticks, we analyze the price movements
                    2. We count the number of times the price direction changes (up to down or down to up)
                    3. The raw count is scaled to produce values in the 100-400 range:
                       - Values near 100 indicate low choppiness (trending market)
                       - Values near 250 indicate moderate choppiness
                       - Values near 400 indicate high choppiness (sideways or volatile market)
                    4. Values are then aggregated into time intervals for clearer visualization
                    
                    Higher choppiness may indicate market uncertainty, increased volatility, or resistance/support levels being tested.
                    """)
            else:
                st.error("Not enough data to calculate choppiness metrics")
        else:
            if rollbit_df is None:
                st.error(f"Could not fetch Rollbit data for {selected_pair}")
            if surf_df is None:
                st.error(f"Could not fetch Surf data for {selected_pair}")

if __name__ == "__main__":
    main()