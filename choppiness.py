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

# Clear cache at startup to ensure fresh data
if hasattr(st, 'cache_data'):
    st.cache_data.clear()

# Page configuration
st.set_page_config(page_title="5000-Tick Choppiness Over Time", layout="wide")

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

# Fetch tick data with timestamps for a specific time period
def fetch_tick_data(pair_name, platform, hours=24, min_ticks=10000, progress_callback=None):
    """
    Fetch tick data for a specific pair from the specified platform.
    Ensures we get at least min_ticks data points or data from the last hours, whichever is more.
    Returns timestamps with data to track changes over time.
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
            
            if progress_callback:
                progress_callback(0.8, f"Processing {len(df)} {platform} ticks...")
            
            return df
    
    except Exception as e:
        if progress_callback:
            progress_callback(1.0, f"{platform} error: {str(e)}")
        st.error(f"Error fetching {platform} tick data: {e}")
        return None

# Calculate choppiness for a single window
def calculate_choppiness_for_window(prices, window_size=20):
    """
    Calculate choppiness for a specific window using parameters.py formula
    
    Args:
        prices: Series of price values
        window_size: Size of the rolling window
    
    Returns:
        Choppiness value
    """
    if len(prices) < window_size:
        return None
    
    try:
        # Calculate absolute differences
        diff = prices.diff().abs().dropna()
        
        # Sum of absolute changes
        sum_abs_changes = diff.sum()
        
        # Calculate price range
        price_range = prices.max() - prices.min()
        
        # Avoid division by zero
        epsilon = 1e-10
        
        # Calculate choppiness
        choppiness = 100 * sum_abs_changes / (price_range + epsilon)
        
        return choppiness
    except Exception as e:
        st.error(f"Error calculating window choppiness: {e}")
        return None

# Track 5000-tick choppiness over time
def track_choppiness_over_time(df, window_size=20, tick_count=5000):
    """
    Track how the 5000-tick choppiness average changes over time.
    
    Args:
        df: DataFrame with timestamp_sgt and price columns
        window_size: Size of the rolling window for choppiness calculation
        tick_count: Number of most recent ticks to use for each calculation
    
    Returns:
        Tuple of (timestamps, choppiness_values)
    """
    if df is None or len(df) < tick_count + window_size:
        st.write(f"Not enough data: {len(df) if df is not None else 0} ticks available, need at least {tick_count + window_size}")
        return [], []
    
    # Make sure df is sorted by timestamp (ascending)
    df = df.sort_values('timestamp_sgt')
    
    timestamps = []
    choppiness_values = []
    
    # Determine how many points we can show (between 5 and 20)
    # We want enough points to show a trend, but not so many that they're crowded
    available_points = len(df) - tick_count
    if available_points <= 0:
        st.write(f"Not enough data points: need more than {tick_count} ticks")
        return [], []
    
    # Target about 10-15 points on the timeline, but at least 3
    num_points = min(15, max(3, available_points // 500))
    
    # If we can't even show 3 points, adjust tick_count or reduce points
    if num_points < 3:
        if len(df) > 6000:
            # Use fewer ticks if we have enough data
            adjusted_tick_count = len(df) // 3
            num_points = 3
            st.info(f"Adjusting to use {adjusted_tick_count} ticks per point to show timeline")
            tick_count = adjusted_tick_count
        else:
            # Just use fewer points
            num_points = max(2, len(df) // tick_count)
            st.info(f"Limited data - showing {num_points} timeline points")
    
    # Calculate the step size between points
    step_size = max(1, (len(df) - tick_count) // num_points)
    
    # For debugging
    st.write(f"Total ticks: {len(df)}, Tick count: {tick_count}, Step size: {step_size}, Points: {num_points}")
    
    # Calculate choppiness at regular intervals
    for i in range(tick_count, len(df)+1, step_size):
        end_idx = min(i, len(df))
        start_idx = max(0, end_idx - tick_count)
        
        # Get the timestamp for this point (end of the window)
        timestamp = df['timestamp_sgt'].iloc[end_idx-1]
        
        # Get the ticks for this window
        window_df = df.iloc[start_idx:end_idx]
        
        # Calculate 20-tick choppiness values across these ticks
        chop_values = []
        
        # For each 20-tick window in these 5000 ticks
        for j in range(window_size, len(window_df)):
            # Get the price data for this 20-tick window
            price_window = window_df['price'].iloc[j-window_size:j]
            
            # Calculate choppiness for this window
            chop = calculate_choppiness_for_window(price_window, window_size)
            if chop is not None:
                chop_values.append(chop)
        
        # Calculate average choppiness across all 20-tick windows
        if chop_values:
            avg_choppiness = sum(chop_values) / len(chop_values)
            timestamps.append(timestamp)
            choppiness_values.append(avg_choppiness)
    
    return timestamps, choppiness_values

# Main app
def main():
    # Get current Singapore time
    singapore_tz = pytz.timezone('Asia/Singapore')
    now_sg = datetime.now(singapore_tz)
    current_time_sg = now_sg.strftime("%Y-%m-%d %H:%M:%S")
    
    # Main title
    st.title("5000-Tick Choppiness Over Time: Rollbit vs Surf")
    st.markdown(f"<p style='text-align: center; font-size:14px; color:gray;'>Last updated: {current_time_sg} (SGT)</p>", unsafe_allow_html=True)
    
    st.markdown("Tracking how the 5000-tick choppiness average (using 20-tick windows) changes over time")
    
    # Get available pairs
    available_pairs = get_available_pairs()
    
    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Selection UI
        selected_pair = st.selectbox(
            "Select cryptocurrency pair",
            options=available_pairs,
            index=0 if "BTC/USDT" in available_pairs else 0
        )
    
    with col2:
        # Hours to look back
        hours_lookback = st.selectbox(
            "Hours to Look Back",
            options=[4, 8, 12, 24],
            index=2,  # Default to 12 hours
            help="Amount of historical data to fetch (more hours = more data points)"
        )
    
    # Create plot button
    if st.button("Generate 5000-Tick Choppiness Plot", use_container_width=True):
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
        
        # Fetch data for both platforms with extra ticks to have enough history
        with st.spinner("Fetching tick data..."):
            # Need at least 10000 ticks to have a good timeline of 5000-tick windows
            rollbit_df = fetch_tick_data(selected_pair, 'rollbit', hours_lookback, 10000, update_rollbit)
            surf_df = fetch_tick_data(selected_pair, 'surf', hours_lookback, 10000, update_surf)
        
        # Process data if available
        if rollbit_df is not None and surf_df is not None:
            st.success(f"Fetched {len(rollbit_df)} ticks from Rollbit and {len(surf_df)} ticks from Surf")
            
            # Fixed window size of 20 for choppiness calculation
            window_size = 20
            
            # Check if we have enough data for at least the most recent calculation
            if len(rollbit_df) < window_size + 1 or len(surf_df) < window_size + 1:
                st.warning(f"Not enough data for 20-tick window calculation. Rollbit: {len(rollbit_df)} ticks, Surf: {len(surf_df)} ticks")
                return
            
            # Determine optimal tick count for timeline (default to 5000 if enough data)
            rollbit_tick_count = min(5000, max(1000, len(rollbit_df) // 2)) 
            surf_tick_count = min(5000, max(1000, len(surf_df) // 2))
            
            st.info(f"Using {rollbit_tick_count} ticks for Rollbit and {surf_tick_count} ticks for Surf timeline calculations")
            
            # Track choppiness over time
            with st.spinner("Calculating tick choppiness over time..."):
                rollbit_times, rollbit_chop = track_choppiness_over_time(rollbit_df, window_size, rollbit_tick_count)
                surf_times, surf_chop = track_choppiness_over_time(surf_df, window_size, surf_tick_count)
            
            if len(rollbit_times) > 0 and len(surf_times) > 0:
                # Calculate the most recent choppiness (should match parameters.py)
                rollbit_latest = rollbit_chop[-1] if rollbit_chop else None
                surf_latest = surf_chop[-1] if surf_chop else None
                
                # Display latest choppiness values
                st.subheader(f"Latest {rollbit_tick_count}-Tick Choppiness (Should match parameters.py)")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Rollbit Choppiness", f"{rollbit_latest:.4f}" if rollbit_latest else "N/A")
                with col2:
                    st.metric("Surf Choppiness", f"{surf_latest:.4f}" if surf_latest else "N/A")
                
                # Create plot
                fig = go.Figure()
                
                # Add Rollbit line
                fig.add_trace(go.Scatter(
                    x=rollbit_times,
                    y=rollbit_chop,
                    mode='lines+markers',
                    name='Rollbit Choppiness',
                    line=dict(color='red', width=2),
                    marker=dict(size=8, color='red'),
                    hoverinfo='text',
                    hovertext=[f'Time: {t.strftime("%H:%M:%S")}<br>Choppiness: {v:.2f}' 
                              for t, v in zip(rollbit_times, rollbit_chop)]
                ))
                
                # Add Surf line
                fig.add_trace(go.Scatter(
                    x=surf_times,
                    y=surf_chop,
                    mode='lines+markers',
                    name='Surf Choppiness',
                    line=dict(color='blue', width=2),
                    marker=dict(size=8, color='blue'),
                    hoverinfo='text',
                    hovertext=[f'Time: {t.strftime("%H:%M:%S")}<br>Choppiness: {v:.2f}' 
                              for t, v in zip(surf_times, surf_chop)]
                ))
                
                # Add horizontal line for latest Rollbit value
                if rollbit_latest:
                    fig.add_shape(
                        type="line",
                        x0=min(rollbit_times[0], surf_times[0]) if rollbit_times and surf_times else 0,
                        y0=rollbit_latest,
                        x1=max(rollbit_times[-1], surf_times[-1]) if rollbit_times and surf_times else 1,
                        y1=rollbit_latest,
                        line=dict(color="rgba(255,0,0,0.5)", width=1, dash="dash"),
                    )
                
                # Add horizontal line for latest Surf value
                if surf_latest:
                    fig.add_shape(
                        type="line",
                        x0=min(rollbit_times[0], surf_times[0]) if rollbit_times and surf_times else 0,
                        y0=surf_latest,
                        x1=max(rollbit_times[-1], surf_times[-1]) if rollbit_times and surf_times else 1,
                        y1=surf_latest,
                        line=dict(color="rgba(0,0,255,0.5)", width=1, dash="dash"),
                    )
                
                # Layout
                y_min = min(min(rollbit_chop), min(surf_chop)) * 0.9 if rollbit_chop and surf_chop else 100
                y_max = max(max(rollbit_chop), max(surf_chop)) * 1.1 if rollbit_chop and surf_chop else 400
                
                fig.update_layout(
                    title=f"Tick Choppiness Over Time: {selected_pair} (Window Size: 20 ticks)",
                    xaxis_title="Time (SGT)",
                    yaxis_title="Choppiness",
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
                        range=[y_min, y_max],
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
                
                # Add annotations for latest values
                if rollbit_latest:
                    fig.add_annotation(
                        x=0.01,
                        y=0.95,
                        xref="paper",
                        yref="paper",
                        text=f"Latest Rollbit: {rollbit_latest:.2f}",
                        showarrow=False,
                        font=dict(size=12, color="red"),
                        bgcolor="white",
                        bordercolor="red",
                        borderwidth=1,
                        borderpad=4,
                        opacity=0.8
                    )
                
                if surf_latest:
                    fig.add_annotation(
                        x=0.01,
                        y=0.90,
                        xref="paper",
                        yref="paper",
                        text=f"Latest Surf: {surf_latest:.2f}",
                        showarrow=False,
                        font=dict(size=12, color="blue"),
                        bgcolor="white",
                        bordercolor="blue",
                        borderwidth=1,
                        borderpad=4,
                        opacity=0.8
                    )
                
                # Show plot
                st.plotly_chart(fig, use_container_width=True)
                
                # Display metrics in columns
                col1, col2 = st.columns(2)
                
                # Calculate metrics
                rollbit_min = min(rollbit_chop) if rollbit_chop else 0
                rollbit_max = max(rollbit_chop) if rollbit_chop else 0
                surf_min = min(surf_chop) if surf_chop else 0
                surf_max = max(surf_chop) if surf_chop else 0
                
                # Display metrics
                with col1:
                    st.markdown("### Rollbit Metrics")
                    st.metric(
                        "Latest Choppiness", 
                        f"{rollbit_latest:.2f}" if rollbit_latest else "N/A", 
                        f"{rollbit_latest - surf_latest:.2f} vs Surf" if rollbit_latest and surf_latest else "N/A"
                    )
                    st.markdown(f"**Range:** {rollbit_min:.2f} to {rollbit_max:.2f}")
                    st.markdown(f"**Data Points:** {len(rollbit_times)}")
                
                with col2:
                    st.markdown("### Surf Metrics")
                    st.metric(
                        "Latest Choppiness", 
                        f"{surf_latest:.2f}" if surf_latest else "N/A", 
                        f"{surf_latest - rollbit_latest:.2f} vs Rollbit" if rollbit_latest and surf_latest else "N/A"
                    )
                    st.markdown(f"**Range:** {surf_min:.2f} to {surf_max:.2f}")
                    st.markdown(f"**Data Points:** {len(surf_times)}")
                
                # Add explanation of the chart
                with st.expander("About This Choppiness Plot"):
                    st.markdown("""
                    **How Choppiness Is Calculated:**
                    
                    1. For each point in time, we look at the most recent ticks
                    2. Within those ticks, we calculate the choppiness using a 20-tick window:
                       - Formula: `choppiness = 100 * sum_abs_changes / (price_range + epsilon)`
                       - Where `sum_abs_changes` is the sum of all absolute price changes in the window
                       - And `price_range` is the difference between max and min prices in the window
                    3. We average all the 20-tick choppiness values across the entire window
                    4. We track how this average changes over time
                    
                    **The latest values (rightmost points) should match the values in parameters.py.**
                    
                    This approach shows how the overall choppiness for each exchange changes
                    as new market data arrives, giving insight into evolving market conditions.
                    """)
            else:
                st.error("Not enough data to calculate a meaningful choppiness timeline")
                st.info("Try increasing 'Hours to Look Back' or selecting a more actively traded pair")
        else:
            if rollbit_df is None:
                st.error(f"Could not fetch Rollbit data for {selected_pair}")
            if surf_df is None:
                st.error(f"Could not fetch Surf data for {selected_pair}")

if __name__ == "__main__":
    main()