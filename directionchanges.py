# Save this as pages/06_Direction_Changes_Table.py in your Streamlit app folder

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pytz
import warnings
from sqlalchemy import create_engine

# Suppress warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Direction Changes Analysis",
    page_icon="🔄",
    layout="wide"
)

# Specify the exact pairs to analyze
PAIRS_TO_ANALYZE = [
    "PEPE/USDT", "PAXG/USDT", "DOGE/USDT", "BTC/USDT", "EOS/USDT",
    "BNB/USDT", "MERL/USDT", "FHE/USDT", "IP/USDT", "ORCA/USDT",
    "TRUMP/USDT", "LIBRA/USDT", "AI16Z/USDT", "OM/USDT", "TRX/USDT",
    "S/USDT", "PI/USDT", "JUP/USDT", "BABY/USDT", "PARTI/USDT",
    "ADA/USDT", "HYPE/USDT", "VIRTUAL/USDT", "SUI/USDT", "SATS/USDT",
    "XRP/USDT", "ORDI/USDT", "WIF/USDT", "VANA/USDT", "PENGU/USDT",
    "VINE/USDT", "GRIFFAIN/USDT", "MEW/USDT", "POPCAT/USDT", "FARTCOIN/USDT",
    "TON/USDT", "MELANIA/USDT", "SOL/USDT", "PNUT/USDT", "CAKE/USDT",
    "TST/USDT", "ETH/USDT"
]

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
    
    engine = create_engine(
        f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['database']}",
        isolation_level="AUTOCOMMIT",  # 设置自动提交模式
        pool_size=5,  # 连接池大小
        max_overflow=10,  # 最大溢出连接数
        pool_timeout=30,  # 连接超时时间
        pool_recycle=1800,  # 连接回收时间(30分钟)
        pool_pre_ping=True,  # 使用连接前先测试连接是否有效
        pool_use_lifo=True,  # 使用后进先出,减少空闲连接
        echo=False  # 不打印 SQL 语句
    )
except Exception as e:
    st.error(f"Error connecting to the database: {e}")
    st.stop()

# --- UI Setup ---
st.title("Direction Changes Analysis (10min Intervals)")
st.subheader("Selected Trading Pairs - Last 24 Hours (Singapore Time)")

# Define parameters
timeframe = "10min"  # Using 10-minute intervals as requested
lookback_days = 1  # 24 hours
expected_points = 144  # Expected data points per pair over 24 hours (6 per hour × 24 hours)
singapore_timezone = pytz.timezone('Asia/Singapore')

# Get current time in Singapore timezone
now_utc = datetime.now(pytz.utc)
now_sg = now_utc.astimezone(singapore_timezone)
st.write(f"Current Singapore Time: {now_sg.strftime('%Y-%m-%d %H:%M:%S')}")

# Function to get partition tables based on date range
def get_partition_tables(conn, start_date, end_date):
    """
    Get list of partition tables that need to be queried based on date range.
    Returns a list of table names (oracle_price_log_partition_v1)
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
    table_names = ["oracle_price_log_partition_v1"]
    
    # Verify which tables actually exist in the database
    cursor = conn.cursor()
    existing_tables = []
    
    for table in table_names:
        # Check if table exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = %s
            );
        """, (table,))
        
        if cursor.fetchone()[0]:
            existing_tables.append(table)
    
    cursor.close()
    
    if not existing_tables:
        st.warning(f"No partition tables found for the date range {start_date.date()} to {end_date.date()}")
    
    return existing_tables

# Function to check table data by hour using Singapore time
def check_table_data_by_hour(conn, table):
    """
    Check data available by hour in Singapore time for a given partition table
    """
    cursor = conn.cursor()
    
    try:
        # Query to get data by hour
        cursor.execute(f"""
            SELECT 
                EXTRACT(HOUR FROM created_at + INTERVAL '8 hour') AS hour,
                COUNT(*) AS record_count
            FROM 
                public.{table}
            WHERE 
                source_type = 0
            GROUP BY 
                hour
            ORDER BY 
                hour
        """)
        
        return cursor.fetchall()
    except Exception as e:
        print(f"Error checking hourly data for {table}: {e}")
        return []
    finally:
        cursor.close()

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
        # Use the correct time zone conversion for our query
        query = f"""
        SELECT 
            pair_name,
            created_at + INTERVAL '8 hour' AS timestamp_sg, 
            final_price AS price
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
    complete_query = " UNION ".join(union_parts) + " ORDER BY timestamp_sg"
    return complete_query

# UI Controls
col1, col2 = st.columns([3, 1])

with col1:
    # Let user select tokens to display (or select all)
    select_all = st.checkbox("Select All Tokens", value=True)
    
    if select_all:
        selected_tokens = PAIRS_TO_ANALYZE
    else:
        selected_tokens = st.multiselect(
            "Select Tokens", 
            PAIRS_TO_ANALYZE,
            default=PAIRS_TO_ANALYZE[:5] if len(PAIRS_TO_ANALYZE) > 5 else PAIRS_TO_ANALYZE
        )

with col2:
    # Add a refresh button
    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.experimental_rerun()

if not selected_tokens:
    st.warning("Please select at least one token")
    st.stop()

# Function to generate aligned 10-minute time blocks for the past 24 hours
def generate_aligned_time_blocks(current_time):
    """
    Generate fixed 10-minute time blocks for past 24 hours,
    aligned with standard 10-minute intervals (e.g., 4:00-4:10, 4:10-4:20)
    """
    # Round down to the nearest 10-minute mark
    minute = current_time.minute
    nearest_10min = (minute // 10) * 10
    latest_complete_block_end = current_time.replace(minute=nearest_10min, second=0, microsecond=0)
    
    # Generate block labels for display
    blocks = []
    for i in range(6 * 24):  # 24 hours of 10-minute blocks (6 per hour)
        block_end = latest_complete_block_end - timedelta(minutes=i*10)
        block_start = block_end - timedelta(minutes=10)
        block_label = f"{block_start.strftime('%H:%M')}"
        blocks.append((block_start, block_end, block_label))
    
    return blocks

# Generate aligned time blocks
aligned_time_blocks = generate_aligned_time_blocks(now_sg)
time_block_labels = [block[2] for block in aligned_time_blocks]

# Direction change classification
def classify_direction_changes(changes_count, interval_minutes=10):
    """
    Classify the number of direction changes within a 10-minute period
    """
    # Calculate theoretical max changes per minute (based on 500ms data)
    max_changes_per_minute = 60 / 0.5  # 120 potential changes per minute
    max_changes = max_changes_per_minute * interval_minutes * 0.5  # Theoretical maximum (50% of potential)
    
    if pd.isna(changes_count) or changes_count < 0:
        return ("UNKNOWN", 0, "Insufficient data")
    elif changes_count == 0:
        return ("NONE", 0, "No direction changes")
    elif changes_count < max_changes * 0.25:  # Less than 25% of theoretical max
        return ("LOW", 1, "Few direction changes")
    elif changes_count < max_changes * 0.50:  # Between 25% and 50% of theoretical max
        return ("MEDIUM", 2, "Moderate direction changes")
    elif changes_count < max_changes * 0.75:  # Between 50% and 75% of theoretical max
        return ("HIGH", 3, "Many direction changes")
    else:  # More than 75% of theoretical max
        return ("EXTREME", 4, "Extreme number of direction changes")

# Analyze price runs (similar to your CryptoMarketAnalyzer.analyze_price_runs method)
def analyze_price_runs(df):
    print("Analyzing price runs...")
    direction = df['direction'].values

    runs = {'up': [], 'down': [], 'flat': []}
    current_direction = 0 if direction[0] == 0 else (1 if direction[0] > 0 else -1)
    current_run = 1

    for i in range(1, len(direction)):
        if direction[i] == 0:
            continue
        elif direction[i] == current_direction:
            current_run += 1
        else:
            if current_direction == 1:
                runs['up'].append(current_run)
            elif current_direction == -1:
                runs['down'].append(current_run)
            elif current_direction == 0:
                runs['flat'].append(current_run)

            current_run = 1
            current_direction = direction[i]

    if current_direction == 1:
        runs['up'].append(current_run)
    elif current_direction == -1:
        runs['down'].append(current_run)
    elif current_direction == 0:
        runs['flat'].append(current_run)

    result = {
        'up_runs': {
            'count': len(runs['up']),
            'average_length': np.mean(runs['up']) if runs['up'] else 0,
            'max_length': max(runs['up']) if runs['up'] else 0
        },
        'down_runs': {
            'count': len(runs['down']),
            'average_length': np.mean(runs['down']) if runs['down'] else 0,
            'max_length': max(runs['down']) if runs['down'] else 0
        },
        'flat_runs': {
            'count': len(runs['flat']),
            'average_length': np.mean(runs['flat']) if runs['flat'] else 0,
            'max_length': max(runs['flat']) if runs['flat'] else 0
        },
        'total_runs': len(runs['up']) + len(runs['down']) + len(runs['flat'])
    }

    return result

# Check data availability in today's partition table
today_str = now_sg.date().strftime("%Y%m%d")
yesterday_str = (now_sg.date() - timedelta(days=1)).strftime("%Y%m%d")
today_table = "oracle_price_log_partition_v1"
yesterday_table = "oracle_price_log_partition_v1"

# Check tables exist
cursor = conn.cursor()
today_exists = False
yesterday_exists = False

cursor.execute("""
    SELECT EXISTS (
        SELECT FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name = %s
    );
""", (today_table,))
today_exists = cursor.fetchone()[0]

cursor.execute("""
    SELECT EXISTS (
        SELECT FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name = %s
    );
""", (yesterday_table,))
yesterday_exists = cursor.fetchone()[0]

cursor.close()

with st.expander("View Data Availability by Hour"):
    # Show data availability by hour
    if today_exists:
        today_data = check_table_data_by_hour(conn, today_table)
        st.write(f"Today's Data ({today_str}) by Hour (Singapore Time):")
        today_df = pd.DataFrame(today_data, columns=['Hour', 'Record Count'])
        st.dataframe(today_df)
    
    if yesterday_exists:
        yesterday_data = check_table_data_by_hour(conn, yesterday_table)
        st.write(f"Yesterday's Data ({yesterday_str}) by Hour (Singapore Time):")
        yesterday_df = pd.DataFrame(yesterday_data, columns=['Hour', 'Record Count'])
        st.dataframe(yesterday_df)

# Enhanced fetch and calculate function with correct time zone handling
@st.cache_data(ttl=600, show_spinner="Calculating direction changes...")
def fetch_and_calculate_direction_changes(token):
    # Get current time in Singapore timezone
    now_utc = datetime.now(pytz.utc)
    now_sg = now_utc.astimezone(singapore_timezone)
    
    # Calculate the lookback period
    start_time_sg = now_sg - timedelta(days=lookback_days)
    
    # Convert for database query (keep as Singapore time strings but we'll adjust in the query)
    start_time = start_time_sg.strftime("%Y-%m-%d %H:%M:%S")
    end_time = now_sg.strftime("%Y-%m-%d %H:%M:%S")

    # Get tables for today and yesterday
    tables_to_query = []
    if yesterday_exists:
        tables_to_query.append(yesterday_table)
    if today_exists:
        tables_to_query.append(today_table)
    
    if not tables_to_query:
        print(f"[{token}] No partition tables found for the specified date range")
        return None
    
    # Build query using all available partition tables
    query = build_query_for_partition_tables(
        tables_to_query,
        pair_name=token,
        start_time=start_time,
        end_time=end_time
    )
    
    try:
        print(f"[{token}] Executing query across {len(tables_to_query)} partition tables")
        df = pd.read_sql_query(query, conn)
        print(f"[{token}] Query executed. DataFrame shape: {df.shape}")

        if df.empty:
            print(f"[{token}] No data found.")
            return None

        # Rename column for clarity - timestamp_sg contains Singapore time
        df = df.rename(columns={'timestamp_sg': 'timestamp'})
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter to just the last 24 hours of data
        last_24h_cutoff = now_sg - timedelta(days=1)
        df = df[df['timestamp'] >= last_24h_cutoff]
        
        if df.empty:
            print(f"[{token}] No data in the last 24 hours.")
            return None
            
        df = df.set_index('timestamp').sort_index()
        
        # Calculate basic metrics (using same logic as CryptoMarketAnalyzer)
        df['price_change'] = df['price'].diff()
        df['abs_price_change'] = df['price_change'].abs()
        
        # Calculate direction
        df['direction'] = np.sign(df['price_change']).fillna(0)
        
        # Calculate direction changes (same logic as your code)
        df['prev_direction'] = df['direction'].shift(1).fillna(0)
        df['direction_change'] = ((df['direction'] != df['prev_direction']) &
                                 (df['direction'] != 0) &
                                 (df['prev_direction'] != 0)).astype(int)
        
        # Calculate the number of direction changes in each 10-minute window
        direction_changes = df['direction_change'].resample('10min', closed='left', label='left').sum()
        
        # Create a DataFrame with the results
        changes_df = direction_changes.to_frame(name='direction_changes')
        changes_df['original_datetime'] = changes_df.index
        
        # Create a DateTime column in string format for display
        changes_df['datetime_str'] = changes_df.index.strftime('%Y-%m-%d %H:%M')
        
        # Create time_label for display in the table (just HH:MM format)
        changes_df['time_label'] = changes_df.index.strftime('%H:%M')
        
        # Create a unique composite key using date and time
        changes_df['unique_key'] = changes_df.index.strftime('%Y-%m-%d_%H:%M')
        
        # Calculate the 24-hour average
        changes_df['avg_24h_changes'] = changes_df['direction_changes'].mean()
        
        # Classify direction changes
        changes_df['changes_info'] = changes_df['direction_changes'].apply(classify_direction_changes)
        changes_df['changes_regime'] = changes_df['changes_info'].apply(lambda x: x[0])
        changes_df['changes_desc'] = changes_df['changes_info'].apply(lambda x: x[2])
        
        # Also classify the 24-hour average
        changes_df['avg_changes_info'] = changes_df['avg_24h_changes'].apply(classify_direction_changes)
        changes_df['avg_changes_regime'] = changes_df['avg_changes_info'].apply(lambda x: x[0])
        changes_df['avg_changes_desc'] = changes_df['avg_changes_info'].apply(lambda x: x[2])
        
        # Calculate direction change rate (changes per minute)
        changes_df['changes_per_minute'] = changes_df['direction_changes'] / 10
        changes_df['avg_changes_per_minute'] = changes_df['avg_24h_changes'] / 10
        
        # Additional metrics
        # Calculate the percentage of time spent in each direction
        uptime_pct = (df[df['direction'] > 0].shape[0] / df.shape[0]) * 100 if df.shape[0] > 0 else 0
        downtime_pct = (df[df['direction'] < 0].shape[0] / df.shape[0]) * 100 if df.shape[0] > 0 else 0
        flattime_pct = (df[df['direction'] == 0].shape[0] / df.shape[0]) * 100 if df.shape[0] > 0 else 0
        
        changes_df['uptime_pct'] = uptime_pct
        changes_df['downtime_pct'] = downtime_pct
        changes_df['flattime_pct'] = flattime_pct
        
        # Calculate price runs
        runs = analyze_price_runs(df)
        avg_run_length = (runs['up_runs']['average_length'] + runs['down_runs']['average_length']) / 2
        changes_df['avg_run_length'] = avg_run_length
        changes_df['up_run_avg'] = runs['up_runs']['average_length']
        changes_df['down_run_avg'] = runs['down_runs']['average_length']
        changes_df['up_run_count'] = runs['up_runs']['count']
        changes_df['down_run_count'] = runs['down_runs']['count']
        
        print(f"[{token}] Successful Direction Changes Calculation")
        return changes_df
    except Exception as e:
        st.error(f"Error processing {token}: {e}")
        print(f"[{token}] Error processing: {e}")
        return None

# Show the blocks we're analyzing
with st.expander("View Time Blocks Being Analyzed"):
    time_blocks_df = pd.DataFrame([(b[0].strftime('%Y-%m-%d %H:%M'), b[1].strftime('%Y-%m-%d %H:%M'), b[2]) 
                                  for b in aligned_time_blocks], 
                                 columns=['Start Time', 'End Time', 'Block Label'])
    st.dataframe(time_blocks_df)

# Show progress bar while calculating
progress_bar = st.progress(0)
status_text = st.empty()

# Calculate direction changes for each token
token_results = {}
for i, token in enumerate(selected_tokens):
    try:
        progress_bar.progress((i) / len(selected_tokens))
        status_text.text(f"Processing {token} ({i+1}/{len(selected_tokens)})")
        result = fetch_and_calculate_direction_changes(token)
        if result is not None:
            token_results[token] = result
    except Exception as e:
        st.error(f"Error processing token {token}: {e}")
        print(f"Error processing token {token} in main loop: {e}")

# Final progress update
progress_bar.progress(1.0)
status_text.text(f"Processed {len(token_results)}/{len(selected_tokens)} tokens successfully")

# If we found any results, display the data coverage info
if token_results:
    # Get the earliest and latest timestamps across all results to show the actual data coverage
    all_timestamps = []
    for token, df in token_results.items():
        if not df.empty and 'original_datetime' in df.columns:
            all_timestamps.extend(df['original_datetime'].tolist())
    
    if all_timestamps:
        earliest_data = min(all_timestamps)
        latest_data = max(all_timestamps)
        
        # Display the data coverage info
        data_range_col1, data_range_col2 = st.columns(2)
        with data_range_col1:
            st.info(f"Data coverage: From {earliest_data.strftime('%Y-%m-%d %H:%M')} to {latest_data.strftime('%Y-%m-%d %H:%M')} (Singapore Time)")
        
        with data_range_col2:
            # Calculate the exact time difference for display and warning
            now_naive = now_sg.replace(tzinfo=None)
            latest_naive = latest_data.replace(tzinfo=None) if hasattr(latest_data, 'tzinfo') else latest_data
            
            # Calculate time difference in hours
            time_diff = now_naive - latest_naive
            hours_diff = time_diff.total_seconds() / 3600
            
            # Show a warning if the latest data is more than 2 hours old
            if hours_diff > 2:
                st.warning(f"⚠️ Latest data is {hours_diff:.1f} hours old")
            else:
                st.success(f"✅ Data is up to date (last update: {latest_data.strftime('%H:%M')})")

# Create table for display
if token_results:
    # Create table data - USING UNIQUE KEY TO PREVENT DUPLICATE INDICES
    table_data = {}
    
    for token, df in token_results.items():
        # Use the unique_key as the index
        changes_series = df.set_index('unique_key')['direction_changes']
        table_data[token] = changes_series
    
    # Create DataFrame with all tokens
    changes_table = pd.DataFrame(table_data)
    
    # Now, extract the date and time from the unique_key for display
    display_idx = []
    time_only_idx = []
    for idx in changes_table.index:
        parts = idx.split('_')
        date_part = parts[0]
        time_part = parts[1]
        display_idx.append(f"{date_part} {time_part}")
        time_only_idx.append(time_part)
    
    # Create a mapping from unique_key to datetime for sorting
    idx_datetime = {idx: pd.to_datetime(idx.replace('_', ' ')) for idx in changes_table.index}
    
    # Sort by datetime (most recent first)
    sorted_idx = sorted(changes_table.index, key=lambda x: idx_datetime[x], reverse=True)
    changes_table = changes_table.reindex(sorted_idx)
    
    # Update the display and time-only indices after sorting
    display_idx = [idx.replace('_', ' ') for idx in changes_table.index]
    time_only_idx = [idx.split('_')[1] for idx in changes_table.index]
    
    # Save the original index
    changes_table_with_orig_idx = changes_table.copy()
    
    # Set the display index for presentation
    changes_table.index = display_idx
    
    # Function to color cells based on number of direction changes
    def color_cells(val):
        if pd.isna(val):
            return 'background-color: #f5f5f5; color: #666666;'  # Grey for missing
        elif val == 0:
            return 'background-color: rgba(240, 240, 240, 0.7); color: black'  # Light grey for no changes
        elif val < 50:  # Low number of changes - green
            intensity = max(0, min(255, int(255 * val / 50)))
            return f'background-color: rgba({255-intensity}, 255, {255-intensity}, 0.7); color: black'
        elif val < 100:  # Medium number of changes - yellow
            intensity = max(0, min(255, int(255 * (val - 50) / 50)))
            return f'background-color: rgba(255, 255, {255-intensity}, 0.7); color: black'
        elif val < 200:  # High number of changes - orange
            intensity = max(0, min(255, int(255 * (val - 100) / 100)))
            return f'background-color: rgba(255, {255-(intensity//2)}, 0, 0.7); color: black'
        else:  # Extreme number of changes - red
            return 'background-color: rgba(255, 0, 0, 0.7); color: white'
    
    styled_table = changes_table.style.applymap(color_cells)
    st.markdown("## Direction Changes Table (10min timeframe, Last 24 hours, Singapore Time)")
    st.markdown("### Color Legend: <span style='color:green'>Few Changes</span>, <span style='color:#aaaa00'>Moderate</span>, <span style='color:orange'>Many</span>, <span style='color:red'>Extreme</span>", unsafe_allow_html=True)
    st.markdown("Values shown as absolute count of direction changes within each 10-minute interval")
    st.dataframe(styled_table, height=700, use_container_width=True)
    
    # Create ranking table based on average direction changes
    st.subheader("Direction Changes Ranking (24-Hour Average, Descending Order)")
    
    ranking_data = []
    for token, df in token_results.items():
        if not df.empty and 'avg_24h_changes' in df.columns and not df['avg_24h_changes'].isna().all():
            avg_changes = df['avg_24h_changes'].iloc[0]  # All rows have the same avg value
            changes_regime = df['avg_changes_desc'].iloc[0]
            max_changes = df['direction_changes'].max()
            min_changes = df['direction_changes'].min()
            # Calculate changes per minute
            avg_changes_per_min = df['avg_changes_per_minute'].iloc[0]
            
            # Get directional bias
            uptime_pct = df['uptime_pct'].iloc[0]
            downtime_pct = df['downtime_pct'].iloc[0]
            flattime_pct = df['flattime_pct'].iloc[0]
            
            # Get run information
            avg_run_length = df['avg_run_length'].iloc[0]
            up_run_avg = df['up_run_avg'].iloc[0]
            down_run_avg = df['down_run_avg'].iloc[0]
            
            # Determine directional bias text
            if uptime_pct > downtime_pct * 1.5:
                directional_bias = "Strong Upward"
            elif uptime_pct > downtime_pct * 1.1:
                directional_bias = "Slight Upward"
            elif downtime_pct > uptime_pct * 1.5:
                directional_bias = "Strong Downward"
            elif downtime_pct > uptime_pct * 1.1:
                directional_bias = "Slight Downward"
            else:
                directional_bias = "Neutral"
            
            ranking_data.append({
                'Token': token,
                'Avg Changes': round(avg_changes, 1),
                'Changes/Min': round(avg_changes_per_min, 1),
                'Regime': changes_regime,
                'Max Changes': round(max_changes),
                'Min Changes': round(min_changes),
                'Range': round(max_changes - min_changes),
                'Avg Run Length': round(avg_run_length, 1),
                'Up Run Avg': round(up_run_avg, 1),
                'Down Run Avg': round(down_run_avg, 1),
                'Uptime %': round(uptime_pct, 1),
                'Downtime %': round(downtime_pct, 1),
                'Flattime %': round(flattime_pct, 1),
                'Directional Bias': directional_bias
            })
    
    if ranking_data:
        ranking_df = pd.DataFrame(ranking_data)
        # Sort by average changes (high to low)
        ranking_df = ranking_df.sort_values(by='Avg Changes', ascending=False)
        # Add rank column
        ranking_df.insert(0, 'Rank', range(1, len(ranking_df) + 1))
        
        # Reset the index to remove it
        ranking_df = ranking_df.reset_index(drop=True)
        
        # Format ranking table with colors
        def color_regime(val):
            if 'Few' in val or 'None' in val:
                return 'color: green'
            elif 'Moderate' in val:
                return 'color: #aaaa00'
            elif 'Many' in val:
                return 'color: orange'
            elif 'Extreme' in val:
                return 'color: red'
            return ''
        
        def color_changes(val):
            if pd.isna(val):
                return ''
            elif val < 50:
                return 'color: green'
            elif val < 100:
                return 'color: #aaaa00'
            elif val < 200:
                return 'color: orange'
            else:
                return 'color: red'
        
        def color_directional_bias(val):
            if 'Strong Upward' in val:
                return 'color: #008800; font-weight: bold'
            elif 'Slight Upward' in val:
                return 'color: #008800'
            elif 'Strong Downward' in val:
                return 'color: #880000; font-weight: bold'
            elif 'Slight Downward' in val:
                return 'color: #880000'
            else:
                return 'color: #888888'
        
        # Apply styling
        styled_ranking = ranking_df.style\
            .applymap(color_regime, subset=['Regime'])\
            .applymap(color_changes, subset=['Avg Changes', 'Max Changes', 'Min Changes'])\
            .applymap(color_directional_bias, subset=['Directional Bias'])
        
        # Display the styled dataframe
        st.dataframe(styled_ranking, height=500, use_container_width=True)
    else:
        st.warning("No ranking data available.")
    
    # Identify tokens with high direction changes volatility
    st.subheader("Tokens with Highly Variable Direction Changes")
    
    # Calculate the coefficient of variation (std/mean) for direction changes
    variability_data = []
    for token, df in token_results.items():
        if not df.empty and 'direction_changes' in df.columns:
            changes_std = df['direction_changes'].std()
            changes_mean = df['direction_changes'].mean()
            changes_cv = changes_std / changes_mean if changes_mean > 0 else 0
            
            max_changes = df['direction_changes'].max()
            min_changes = df['direction_changes'].min()
            changes_range = max_changes - min_changes
            
            variability_data.append({
                'Token': token,
                'CoV': round(changes_cv, 2),  # Coefficient of Variation
                'Std Dev': round(changes_std, 1),
                'Mean Changes': round(changes_mean, 1),
                'Range': int(changes_range),
                'Max Changes': int(max_changes),
                'Min Changes': int(min_changes)
            })
    
    if variability_data:
        variability_df = pd.DataFrame(variability_data)
        # Sort by Coefficient of Variation (high to low)
        variability_df = variability_df.sort_values(by='CoV', ascending=False)
        
        # Reset the index to remove it
        variability_df = variability_df.reset_index(drop=True)
        
        # Display the dataframe
        st.dataframe(variability_df, height=300, use_container_width=True)
        
        # Create a bar chart of the top 10 most variable tokens
        top_variable_tokens = variability_df.head(10)
        
        fig = px.bar(
            top_variable_tokens, 
            x='Token', 
            y='CoV', 
            title='Top 10 Tokens by Direction Change Variability',
            labels={'CoV': 'Coefficient of Variation (Std/Mean)', 'Token': 'Token'},
            color='CoV',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No variability data available.")
    
    # Direction Changes Distribution
    st.subheader("24-Hour Direction Changes Overview (Singapore Time)")
    avg_changes = {}
    for token, df in token_results.items():
        if not df.empty and 'avg_24h_changes' in df.columns and not df['avg_24h_changes'].isna().all():
            avg = df['avg_24h_changes'].iloc[0]
            regime = df['avg_changes_desc'].iloc[0]
            avg_changes[token] = (avg, regime)
    
    if avg_changes:
        # Calculate statistics for the different regimes
        few_changes = sum(1 for v, r in avg_changes.values() if 'Few' in r or 'None' in r)
        moderate_changes = sum(1 for v, r in avg_changes.values() if 'Moderate' in r)
        many_changes = sum(1 for v, r in avg_changes.values() if 'Many' in r)
        extreme_changes = sum(1 for v, r in avg_changes.values() if 'Extreme' in r)
        total = few_changes + moderate_changes + many_changes + extreme_changes
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Few Changes", f"{few_changes} ({few_changes/total*100:.1f}%)" if total > 0 else "0 (0.0%)")
        col2.metric("Moderate Changes", f"{moderate_changes} ({moderate_changes/total*100:.1f}%)" if total > 0 else "0 (0.0%)")
        col3.metric("Many Changes", f"{many_changes} ({many_changes/total*100:.1f}%)" if total > 0 else "0 (0.0%)")
        col4.metric("Extreme Changes", f"{extreme_changes} ({extreme_changes/total*100:.1f}%)" if total > 0 else "0 (0.0%)")
        
        labels = ['Few Changes', 'Moderate Changes', 'Many Changes', 'Extreme Changes']
        values = [few_changes, moderate_changes, many_changes, extreme_changes]
        colors = ['rgba(100,255,100,0.8)', 'rgba(255,255,100,0.8)', 'rgba(255,165,0,0.8)', 'rgba(255,0,0,0.8)']
        
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, marker=dict(colors=colors, line=dict(color='#000000', width=2)), textinfo='label+percent', hole=.3)])
        fig.update_layout(
            title="Direction Changes Distribution (Singapore Time)",
            height=400,
            font=dict(color="#000000", size=12),
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Create columns for each direction changes category
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)
        
        with col1:
            st.markdown("### Tokens with Few Direction Changes")
            fc_tokens = [(t, v, r) for t, (v, r) in avg_changes.items() if 'Few' in r or 'None' in r]
            fc_tokens.sort(key=lambda x: x[1])
            if fc_tokens:
                for token, value, regime in fc_tokens:
                    st.markdown(f"- **{token}**: <span style='color:green'>{value:.1f}</span> changes/10min ({regime})", unsafe_allow_html=True)
            else:
                st.markdown("*No tokens in this category*")
        
        with col2:
            st.markdown("### Tokens with Moderate Direction Changes")
            mc_tokens = [(t, v, r) for t, (v, r) in avg_changes.items() if 'Moderate' in r]
            mc_tokens.sort(key=lambda x: x[1])
            if mc_tokens:
                for token, value, regime in mc_tokens:
                    st.markdown(f"- **{token}**: <span style='color:#aaaa00'>{value:.1f}</span> changes/10min ({regime})", unsafe_allow_html=True)
            else:
                st.markdown("*No tokens in this category*")
        
        with col3:
            st.markdown("### Tokens with Many Direction Changes")
            hc_tokens = [(t, v, r) for t, (v, r) in avg_changes.items() if 'Many' in r]
            hc_tokens.sort(key=lambda x: x[1])
            if hc_tokens:
                for token, value, regime in hc_tokens:
                    st.markdown(f"- **{token}**: <span style='color:orange'>{value:.1f}</span> changes/10min ({regime})", unsafe_allow_html=True)
            else:
                st.markdown("*No tokens in this category*")
        
        with col4:
            st.markdown("### Tokens with Extreme Direction Changes")
            ec_tokens = [(t, v, r) for t, (v, r) in avg_changes.items() if 'Extreme' in r]
            ec_tokens.sort(key=lambda x: x[1], reverse=True)
            if ec_tokens:
                for token, value, regime in ec_tokens:
                    st.markdown(f"- **{token}**: <span style='color:red'>{value:.1f}</span> changes/10min ({regime})", unsafe_allow_html=True)
            else:
                st.markdown("*No tokens in this category*")
    else:
        st.warning("No direction changes data available for the selected tokens.")
    
    # Final Summary
    st.subheader("Executive Summary of Direction Changes Analysis")
    
    # Generate a simple text summary
    if ranking_data:
        ranking_df = pd.DataFrame(ranking_data)
        top_changing_token = ranking_df.sort_values(by='Avg Changes', ascending=False).iloc[0]
        lowest_changing_token = ranking_df.sort_values(by='Avg Changes', ascending=True).iloc[0]
        
        most_variable_token = None
        if variability_data:
            var_df = pd.DataFrame(variability_data)
            most_variable_token = var_df.sort_values(by='CoV', ascending=False).iloc[0]
        
        st.write(f"""
        ### Key Findings:
        
        1. **{top_changing_token['Token']}** shows the highest average direction changes with **{top_changing_token['Avg Changes']}** changes per 10-minute period.
        
        2. **{lowest_changing_token['Token']}** shows the lowest average direction changes with **{lowest_changing_token['Avg Changes']}** changes per 10-minute period.
        """)
        
        if most_variable_token is not None:
            st.write(f"""
        3. **{most_variable_token['Token']}** shows the most variable direction change behavior with a Coefficient of Variation of **{most_variable_token['CoV']}**.
            """)
            
        # Add data coverage information to summary
        if all_timestamps:
            earliest_data = min(all_timestamps)
            latest_data = max(all_timestamps)
            
            # Calculate the exact time difference
            now_naive = now_sg.replace(tzinfo=None)
            latest_naive = latest_data.replace(tzinfo=None) if hasattr(latest_data, 'tzinfo') else latest_data
            
            time_diff = now_naive - latest_naive
            hours_diff = time_diff.total_seconds() / 3600
            
            st.write(f"""
        4. Data covers from **{earliest_data.strftime('%Y-%m-%d %H:%M')}** to **{latest_data.strftime('%Y-%m-%d %H:%M')}** (Singapore Time).
        
        5. Latest available data is **{hours_diff:.1f} hours** old.
            """)
    else:
        st.warning("Insufficient data to generate an executive summary.")

with st.expander("Understanding the Direction Changes Table"):
    st.markdown("""
    ### How to Read This Table
    This table shows the number of price direction changes for all selected tokens over the last 24 hours using 10-minute intervals.
    Each row represents a specific 10-minute time period, with times shown in Singapore time. The table is sorted with the most recent 10-minute period at the top.
    
    **Color coding:**
    - **Green** (< 50 changes): Few direction changes
    - **Yellow** (50-100 changes): Moderate direction changes
    - **Orange** (100-200 changes): Many direction changes
    - **Red** (> 200 changes): Extreme number of direction changes
    
    **The intensity of the color indicates the number of direction changes:**
    - Darker green = Fewer changes
    - Darker red = More changes
    
    **Ranking Table:**
    The ranking table sorts tokens by their 24-hour average direction changes from highest to lowest.
    
    ### Direction Change Metric Details
    Direction changes occur when the price movement switches from upward to downward or vice versa. A direction change is counted when:
    
    1. The previous price movement was positive (price increasing)
    2. The current price movement is negative (price decreasing)
    
    OR
    
    1. The previous price movement was negative (price decreasing) 
    2. The current price movement is positive (price increasing)
    
    The calculation ignores flat periods (no price change) and only counts true reversals in price movement.
    
    **Metrics explained:**
    - **Direction Changes**: Raw count of direction changes in the given time period
    - **Changes/Min**: Average number of direction changes per minute
    - **Directional Bias**: Overall tendency of price movement (Upward, Neutral, or Downward)
    - **Up/Down Ratio**: Ratio of time spent in upward movement vs. downward movement
    - **Avg Run Length**: Average number of consecutive price movements in the same direction
    
    **Note on data granularity:**
    The underlying data is sampled at ~500ms intervals, giving a theoretical maximum of 1,200 data points per 10-minute period. The analysis examines each price change to identify genuine direction reversals.
    """)