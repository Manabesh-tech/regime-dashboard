# Save this as pages/06_Trades_PNL_Table_Updated.py in your Streamlit app folder
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import pytz

st.set_page_config(
    page_title="User Trades & Platform PNL Table",
    page_icon="💰",
    layout="wide"
)

# --- DB CONFIG ---
try:
    db_config = st.secrets["database"]
    db_uri = (
        f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}"
        f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    )
    engine = create_engine(db_uri)
except Exception as e:
    st.error(f"Error connecting to the database: {e}")
    st.stop()

# --- UI Setup ---
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("User Trades & Platform PNL Table (30min)")
st.subheader("Active Trading Pairs - Last 24 Hours (Singapore Time)")

# Add a helpful explanation
with st.expander("About this dashboard"):
    st.markdown("""
    ### Updated Platform PNL Dashboard
    
    This dashboard has been updated to:
    1. **Show only active trading pairs** - delisted pairs are excluded
    2. **Calculate platform PNL directly from user PNL** - platform gains when users lose and vice versa
    3. **Provide component breakdowns** of platform PNL (from user trades, fees, and funding)
    4. **Filter pairs with recent activity** - focus on pairs that are actually being traded
    
    Platform PNL Formula: `platform_pnl = -1 * user_pnl + fees + funding`
    """)

# Define parameters for the 30-minute timeframe
timeframe = "30min"
lookback_days = 1  # 24 hours
expected_points = 48  # Expected data points per pair over 24 hours (24 hours * 2 intervals per hour)
singapore_timezone = pytz.timezone('Asia/Singapore')

# Get current time in Singapore timezone
now_utc = datetime.now(pytz.utc)
now_sg = now_utc.astimezone(singapore_timezone)
st.write(f"Current Singapore Time: {now_sg.strftime('%Y-%m-%d %H:%M:%S')}")

# Set thresholds for highlighting
high_trade_count_threshold = 100  # Number of trades considered "high activity"
high_pnl_threshold = 1000  # Platform PNL amount considered "high" (in USD)
low_pnl_threshold = -1000  # Platform PNL amount considered "low" (in USD)

# Fetch only active pairs from DB
@st.cache_data(ttl=600, show_spinner="Fetching active pairs...")
def fetch_active_pairs():
    # Modified query to only fetch currently active pairs
    query = """
    SELECT DISTINCT pair_name 
    FROM public.trade_pool_pairs 
    WHERE is_active = TRUE  -- Only include active pairs
    ORDER BY pair_name
    """
    try:
        df = pd.read_sql(query, engine)
        if df.empty:
            st.error("No active pairs found in the database.")
            return []
        return df['pair_name'].tolist()
    except Exception as e:
        # If the is_active column doesn't exist, fall back to the original query
        try:
            fallback_query = """
            SELECT DISTINCT pair_name 
            FROM public.trade_pool_pairs
            WHERE status = 1  -- Assuming status=1 means active
            ORDER BY pair_name
            """
            df = pd.read_sql(fallback_query, engine)
            if df.empty:
                st.error("No active pairs found in the database.")
                return []
            return df['pair_name'].tolist()
        except Exception as e2:
            st.error(f"Error fetching pairs: {e2}")
            return ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT", "PEPE/USDT"]  # Default fallback

all_pairs = fetch_active_pairs()

# Add a filter for pairs with recent activity
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    # Let user select pairs to display (or select all)
    select_all = st.checkbox("Select All Active Pairs", value=True)
    
with col2:
    # Add a filter for recent activity
    show_only_active = st.checkbox("Show Only Pairs with Recent Activity", value=True)
    
    if show_only_active:
        # Filter to pairs that had trades in the last 24 hours
        @st.cache_data(ttl=600)
        def get_recently_active_pairs():
            now_utc = datetime.now(pytz.utc)
            start_time_utc = now_utc - timedelta(days=lookback_days)
            
            query = f"""
            SELECT DISTINCT pair_name
            FROM public.trade_fill_fresh tf
            JOIN public.trade_pool_pairs tp ON tf.pair_id = tp.pair_id
            WHERE tf.created_at >= '{start_time_utc}'
            AND tf.taker_way IN (1, 2, 3, 4)
            ORDER BY pair_name
            """
            
            try:
                df = pd.read_sql(query, engine)
                if df.empty:
                    return []
                return df['pair_name'].tolist()
            except Exception as e:
                st.error(f"Error fetching active pairs: {e}")
                return all_pairs
        
        active_pairs = get_recently_active_pairs()
        pair_options = active_pairs
    else:
        pair_options = all_pairs
    
    if select_all:
        selected_pairs = pair_options
    else:
        selected_pairs = st.multiselect(
            "Select Pairs", 
            pair_options,
            default=pair_options[:5] if len(pair_options) > 5 else pair_options
        )

with col3:
    # Add a refresh button
    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.experimental_rerun()

if not selected_pairs:
    st.warning("Please select at least one pair")
    st.stop()

# Function to generate aligned 30-minute time blocks for the past 24 hours
def generate_aligned_time_blocks(current_time):
    """
    Generate fixed 30-minute time blocks for past 24 hours,
    aligned with standard 30-minute intervals (e.g., 4:00-4:30, 4:30-5:00)
    """
    # Round down to the nearest 30-minute mark
    if current_time.minute < 30:
        # Round down to XX:00
        latest_complete_block_end = current_time.replace(minute=0, second=0, microsecond=0)
    else:
        # Round down to XX:30
        latest_complete_block_end = current_time.replace(minute=30, second=0, microsecond=0)
    
    # Generate block labels for display
    blocks = []
    for i in range(48):  # 24 hours of 30-minute blocks
        block_end = latest_complete_block_end - timedelta(minutes=i*30)
        block_start = block_end - timedelta(minutes=30)
        block_label = f"{block_start.strftime('%H:%M')}"
        blocks.append((block_start, block_end, block_label))
    
    return blocks

# Generate aligned time blocks
aligned_time_blocks = generate_aligned_time_blocks(now_sg)
time_block_labels = [block[2] for block in aligned_time_blocks]

# Fetch trades data for the past 24 hours in 30min intervals
@st.cache_data(ttl=600, show_spinner="Fetching trade counts...")
def fetch_trade_counts(pair_name):
    # Get current time in Singapore timezone
    now_utc = datetime.now(pytz.utc)
    now_sg = now_utc.astimezone(singapore_timezone)
    start_time_sg = now_sg - timedelta(days=lookback_days)
    
    # Convert back to UTC for database query
    start_time_utc = start_time_sg.astimezone(pytz.utc)
    end_time_utc = now_sg.astimezone(pytz.utc)

    # Updated query to use trade_fill_fresh with consistent time handling
    # Explicitly adding 8 hours to UTC timestamps to match Singapore time
    query = f"""
    SELECT
        date_trunc('hour', created_at + INTERVAL '8 hour') + 
        INTERVAL '30 min' * FLOOR(EXTRACT(MINUTE FROM created_at + INTERVAL '8 hour')::INT / 30) 
        AS timestamp,
        COUNT(*) AS trade_count
    FROM public.trade_fill_fresh
    WHERE created_at BETWEEN '{start_time_utc}' AND '{end_time_utc}'
    AND pair_id IN (SELECT pair_id FROM public.trade_pool_pairs WHERE pair_name = '{pair_name}')
    AND taker_way IN (1, 2, 3, 4)  -- Exclude taker_way = 0 (funding fee deductions)
    GROUP BY
        date_trunc('hour', created_at + INTERVAL '8 hour') + 
        INTERVAL '30 min' * FLOOR(EXTRACT(MINUTE FROM created_at + INTERVAL '8 hour')::INT / 30)
    ORDER BY timestamp
    """
    
    try:
        print(f"[{pair_name}] Executing trade count query")
        df = pd.read_sql(query, engine)
        print(f"[{pair_name}] Trade count query executed. DataFrame shape: {df.shape}")
        
        if df.empty:
            print(f"[{pair_name}] No trade data found.")
            return None
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        
        # Format time label to match our aligned blocks (HH:MM format)
        df['time_label'] = df.index.strftime('%H:%M')
        
        return df
    except Exception as e:
        st.error(f"Error processing trade counts for {pair_name}: {e}")
        print(f"[{pair_name}] Error processing trade counts: {e}")
        return None
    
# Fetch platform PNL from user received PNL for the past 24 hours in 30min intervals
@st.cache_data(ttl=600, show_spinner="Calculating platform PNL from user trades...")
def fetch_platform_pnl_from_user_pnl(pair_name):
    # Get current time in Singapore timezone
    now_utc = datetime.now(pytz.utc)
    now_sg = now_utc.astimezone(singapore_timezone)
    start_time_sg = now_sg - timedelta(days=lookback_days)
    
    # Convert back to UTC for database query
    start_time_utc = start_time_sg.astimezone(pytz.utc)
    end_time_utc = now_sg.astimezone(pytz.utc)

    # This query aggregates user PNL (用户实际到手PNL) in 30-minute intervals
    # Platform PNL is the negative of user PNL (users' losses are platform's gains)
    query = f"""
    SELECT
        date_trunc('hour', created_at + INTERVAL '8 hour') + 
        INTERVAL '30 min' * FLOOR(EXTRACT(MINUTE FROM created_at + INTERVAL '8 hour')::INT / 30) 
        AS timestamp,
        SUM(-1 * (taker_pnl * collateral_price)) AS platform_pnl_from_user,
        SUM(taker_fee * collateral_price) AS platform_fee_income,
        COUNT(*) as trade_count,
        SUM(deal_vol * collateral_price) as trading_volume_usd
    FROM public.trade_fill_fresh
    WHERE created_at BETWEEN '{start_time_utc}' AND '{end_time_utc}'
    AND pair_id IN (SELECT pair_id FROM public.trade_pool_pairs WHERE pair_name = '{pair_name}')
    AND taker_way IN (1, 2, 3, 4)  -- Exclude taker_way = 0 (funding fee deductions)
    GROUP BY
        date_trunc('hour', created_at + INTERVAL '8 hour') + 
        INTERVAL '30 min' * FLOOR(EXTRACT(MINUTE FROM created_at + INTERVAL '8 hour')::INT / 30)
    ORDER BY timestamp
    """
    
    try:
        print(f"[{pair_name}] Executing platform PNL from user PNL query")
        df = pd.read_sql(query, engine)
        print(f"[{pair_name}] Platform PNL query executed. DataFrame shape: {df.shape}")
        
        if df.empty:
            print(f"[{pair_name}] No PNL data found.")
            return None
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        
        # Calculate total platform PNL (from user PNL + fees)
        df['platform_total_pnl'] = df['platform_pnl_from_user'] + df['platform_fee_income']
        
        # Store trade count and volume if available
        if 'trade_count' in df.columns:
            df['trade_count_from_pnl'] = df['trade_count']
        if 'trading_volume_usd' in df.columns:
            df['trading_volume_usd'] = df['trading_volume_usd']
        
        # Format time label to match our aligned blocks (HH:MM format)
        df['time_label'] = df.index.strftime('%H:%M')
        
        return df
    except Exception as e:
        st.error(f"Error processing platform PNL for {pair_name}: {e}")
        print(f"[{pair_name}] Error processing platform PNL: {e}")
        return None

# Additional query to fetch funding fees if needed
@st.cache_data(ttl=600, show_spinner="Calculating funding fees...")
def fetch_funding_fees(pair_name):
    # Get current time in Singapore timezone
    now_utc = datetime.now(pytz.utc)
    now_sg = now_utc.astimezone(singapore_timezone)
    start_time_sg = now_sg - timedelta(days=lookback_days)
    
    # Convert back to UTC for database query
    start_time_utc = start_time_sg.astimezone(pytz.utc)
    end_time_utc = now_sg.astimezone(pytz.utc)

    # This query gets funding fees (taker_way = 0 entries)
    query = f"""
    SELECT
        date_trunc('hour', created_at + INTERVAL '8 hour') + 
        INTERVAL '30 min' * FLOOR(EXTRACT(MINUTE FROM created_at + INTERVAL '8 hour')::INT / 30) 
        AS timestamp,
        SUM(-1 * funding_fee * collateral_price) AS platform_funding_pnl
    FROM public.trade_fill_fresh
    WHERE created_at BETWEEN '{start_time_utc}' AND '{end_time_utc}'
    AND pair_id IN (SELECT pair_id FROM public.trade_pool_pairs WHERE pair_name = '{pair_name}')
    AND taker_way = 0  -- Only funding fee entries
    GROUP BY
        date_trunc('hour', created_at + INTERVAL '8 hour') + 
        INTERVAL '30 min' * FLOOR(EXTRACT(MINUTE FROM created_at + INTERVAL '8 hour')::INT / 30)
    ORDER BY timestamp
    """
    
    try:
        print(f"[{pair_name}] Executing funding fees query")
        df = pd.read_sql(query, engine)
        print(f"[{pair_name}] Funding fees query executed. DataFrame shape: {df.shape}")
        
        if df.empty:
            print(f"[{pair_name}] No funding fee data found.")
            return None
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        
        # Format time label to match our aligned blocks (HH:MM format)
        df['time_label'] = df.index.strftime('%H:%M')
        
        return df
    except Exception as e:
        st.error(f"Error processing funding fees for {pair_name}: {e}")
        print(f"[{pair_name}] Error processing funding fees: {e}")
        return None

# Combine Trade Count and Platform PNL data for visualization
def combine_data(trade_data, pnl_data, funding_data=None):
    if trade_data is None and pnl_data is None and funding_data is None:
        return None
    
    # Create a DataFrame with time blocks as index
    time_blocks = pd.DataFrame(index=[block[2] for block in aligned_time_blocks])
    
    # Add trade count data if available
    if trade_data is not None and not trade_data.empty:
        for time_label in time_blocks.index:
            # Find matching rows in trade_data by time_label
            matching_rows = trade_data[trade_data['time_label'] == time_label]
            if not matching_rows.empty:
                time_blocks.at[time_label, 'trade_count'] = matching_rows['trade_count'].sum()
    
    # Add PNL data if available
    if pnl_data is not None and not pnl_data.empty:
        for time_label in time_blocks.index:
            # Find matching rows in pnl_data by time_label
            matching_rows = pnl_data[pnl_data['time_label'] == time_label]
            if not matching_rows.empty:
                time_blocks.at[time_label, 'platform_pnl_from_user'] = matching_rows['platform_pnl_from_user'].sum()
                time_blocks.at[time_label, 'platform_fee_income'] = matching_rows['platform_fee_income'].sum()
                time_blocks.at[time_label, 'platform_total_pnl'] = matching_rows['platform_total_pnl'].sum()
                
                # Add trading volume if available
                if 'trading_volume_usd' in matching_rows.columns:
                    time_blocks.at[time_label, 'trading_volume_usd'] = matching_rows['trading_volume_usd'].sum()
                
                # Use trade count from PNL query if available and main trade count is missing
                if 'trade_count_from_pnl' in matching_rows.columns and (
                    'trade_count' not in time_blocks.columns or pd.isna(time_blocks.at[time_label, 'trade_count'])
                ):
                    time_blocks.at[time_label, 'trade_count'] = matching_rows['trade_count_from_pnl'].sum()
    
    # Add funding fee data if available
    if funding_data is not None and not funding_data.empty:
        for time_label in time_blocks.index:
            # Find matching rows in funding_data by time_label
            matching_rows = funding_data[funding_data['time_label'] == time_label]
            if not matching_rows.empty:
                time_blocks.at[time_label, 'platform_funding_pnl'] = matching_rows['platform_funding_pnl'].sum()
                
                # If we have both PNL and funding data, update the total
                if 'platform_total_pnl' in time_blocks.columns:
                    time_blocks.at[time_label, 'platform_total_pnl'] += matching_rows['platform_funding_pnl'].sum()
                else:
                    time_blocks.at[time_label, 'platform_total_pnl'] = matching_rows['platform_funding_pnl'].sum()
    
    # Fill NaN values with 0
    time_blocks.fillna(0, inplace=True)
    
    return time_blocks

# Show progress bar while calculating
progress_bar = st.progress(0)
status_text = st.empty()

# Calculate trade count and platform PNL for each pair
pair_results = {}
for i, pair_name in enumerate(selected_pairs):
    try:
        progress_bar.progress((i) / len(selected_pairs))
        status_text.text(f"Processing {pair_name} ({i+1}/{len(selected_pairs)})")
        
        # Fetch trade count data
        trade_data = fetch_trade_counts(pair_name)
        
        # Fetch platform PNL data from user PNL
        pnl_data = fetch_platform_pnl_from_user_pnl(pair_name)
        
        # Fetch funding fee data
        funding_data = fetch_funding_fees(pair_name)
        
        # Combine data
        combined_data = combine_data(trade_data, pnl_data, funding_data)
        
        if combined_data is not None:
            pair_results[pair_name] = combined_data
    except Exception as e:
        st.error(f"Error processing pair {pair_name}: {e}")
        print(f"Error processing pair {pair_name} in main loop: {e}")

# Final progress update
progress_bar.progress(1.0)
status_text.text(f"Processed {len(pair_results)}/{len(selected_pairs)} pairs successfully")

# Create tables for display - Trade Count Table
if pair_results:
    # Create trade count table data
    trade_count_data = {}
    for pair_name, df in pair_results.items():
        if 'trade_count' in df.columns:
            trade_count_data[pair_name] = df['trade_count']
    
    # Create DataFrame with all pairs
    trade_count_table = pd.DataFrame(trade_count_data)
    
    # Apply the time blocks in the proper order (most recent first)
    available_times = set(trade_count_table.index)
    ordered_times = [t for t in time_block_labels if t in available_times]
    
    # If no matches are found in aligned blocks, fallback to the available times
    if not ordered_times and available_times:
        ordered_times = sorted(list(available_times), reverse=True)
    
    # Reindex with the ordered times
    trade_count_table = trade_count_table.reindex(ordered_times)
    
    # Round to integers - trade counts should be whole numbers
    trade_count_table = trade_count_table.round(0).astype('Int64')  # Using Int64 to handle NaN values properly
    
    def color_trade_cells(val):
        if pd.isna(val) or val == 0:
            return 'background-color: #f5f5f5; color: #666666;'  # Grey for missing/zero
        elif val < 10:  # Low activity
            intensity = max(0, min(255, int(255 * val / 10)))
            return f'background-color: rgba({255-intensity}, 255, {255-intensity}, 0.7); color: black'
        elif val < 50:  # Medium activity
            intensity = max(0, min(255, int(255 * (val - 10) / 40)))
            return f'background-color: rgba(255, 255, {255-intensity}, 0.7); color: black'
        elif val < high_trade_count_threshold:  # High activity
            intensity = max(0, min(255, int(255 * (val - 50) / 50)))
            return f'background-color: rgba(255, {255-(intensity//2)}, 0, 0.7); color: black'
        else:  # Very high activity
            return 'background-color: rgba(255, 0, 0, 0.7); color: white'