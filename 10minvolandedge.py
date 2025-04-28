import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import psycopg2
import pytz
from concurrent.futures import ThreadPoolExecutor
import time
import re

st.set_page_config(
    page_title="10-Minute System Edge & Volatility Matrix",
    page_icon="📊",
    layout="wide"
)

# --- PERFORMANCE OPTIONS ---
# Allow user to select resolution to speed up loading
resolution_options = {
    "10-minute": 10,
    "20-minute": 20,
    "30-minute": 30
}

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
st.title("System Edge & Volatility Matrix")
st.subheader("All Trading Pairs - Last 24 Hours (Singapore Time)")

# Add performance options in the sidebar
st.sidebar.header("Performance Settings")
resolution_minutes = st.sidebar.selectbox(
    "Select Time Resolution", 
    list(resolution_options.keys()), 
    index=0, 
    help="Higher resolution (lower minutes) provides more detail but may load slower"
)
resolution_min = resolution_options[resolution_minutes]
max_tokens = st.sidebar.slider("Maximum Tokens to Display", 5, 50, 20, help="Limit the number of tokens to improve performance")
parallel_workers = st.sidebar.slider("Parallel Processing Threads", 1, 8, 4, help="More threads can improve speed but may impact stability")

cache_ttl = st.sidebar.slider("Cache Duration (minutes)", 5, 120, 30, help="How long to keep data cached before refreshing")

# Get current time in Singapore timezone
singapore_timezone = pytz.timezone('Asia/Singapore')
now_utc = datetime.now(pytz.utc)
now_sg = now_utc.astimezone(singapore_timezone)
st.write(f"Current Singapore Time: {now_sg.strftime('%Y-%m-%d %H:%M:%S')}")

# Calculate timeframes
lookback_days = 1  # 24 hours
start_time_sg = now_sg - timedelta(days=lookback_days)
start_time_utc = start_time_sg.astimezone(pytz.UTC)

# Set thresholds
extreme_vol_threshold = 1.0  # 100% annualized volatility
high_edge_threshold = 0.5    # 50% house edge
negative_edge_threshold = -0.2  # -20% house edge (system losing)

# Calculate expected data points based on resolution
expected_points = int(24 * 60 / resolution_min)  # Number of intervals in 24 hours

# Function to get partition tables based on date range
@st.cache_data(ttl=60*cache_ttl, show_spinner=False)
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
        print(f"No partition tables found for the date range {start_date.date()} to {end_date.date()}")
    
    return existing_tables

# Function to build query across partition tables
def build_price_query(tables, pair_name, start_time, end_time):
    """
    Build a complete UNION query for multiple partition tables.
    This creates a complete, valid SQL query with correct WHERE clauses.
    """
    if not tables:
        return ""
        
    union_parts = []
    
    for table in tables:
        # Query for price data (source_type = 0)
        query = f"""
        SELECT 
            pair_name,
            created_at AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore' AS timestamp,
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

# Function to fetch house edge data for all pairs
@st.cache_data(ttl=60*cache_ttl, show_spinner=False)
def fetch_house_edge_data(resolution_min=10):
    """
    Fetch house edge data for all pairs with the specified resolution
    """
    # Calculate time range for the query
    end_time = now_sg
    start_time = end_time - timedelta(days=lookback_days)
    
    # Convert to UTC for database query
    start_time_utc = start_time.astimezone(pytz.UTC)
    
    # Build the SQL query for house edge calculation
    query = f"""
    SELECT
        t.pair_name,
        DATE_TRUNC('{resolution_min} minutes', t.sg_time) AS time_block,
        
        -- Calculate total PNL
        SUM(CASE WHEN t.taker_way IN (1, 2, 3, 4) THEN -1 * t.taker_pnl * t.collateral_price ELSE 0 END) +
        SUM(CASE WHEN t.taker_way = 0 THEN -1 * t.funding_fee * t.collateral_price ELSE 0 END) +
        SUM(-1 * t.rebate * t.collateral_price) AS total_platform_pnl,
        
        -- Calculate margin amount
        SUM(CASE WHEN t.taker_way IN (1, 3) THEN t.deal_vol * t.collateral_price ELSE 0 END) AS margin_amount,
        
        -- Calculate house edge with bounds (-1 to 1)
        CASE
            WHEN SUM(CASE WHEN t.taker_way IN (1, 3) THEN t.deal_vol * t.collateral_price ELSE 0 END) = 0 THEN 0
            WHEN (SUM(CASE WHEN t.taker_way IN (1, 2, 3, 4) THEN -1 * t.taker_pnl * t.collateral_price ELSE 0 END) +
                SUM(CASE WHEN t.taker_way = 0 THEN -1 * t.funding_fee * t.collateral_price ELSE 0 END) +
                SUM(-1 * t.rebate * t.collateral_price)) / 
                SUM(CASE WHEN t.taker_way IN (1, 3) THEN t.deal_vol * t.collateral_price ELSE 0 END) > 1 THEN 1
            WHEN (SUM(CASE WHEN t.taker_way IN (1, 2, 3, 4) THEN -1 * t.taker_pnl * t.collateral_price ELSE 0 END) +
                SUM(CASE WHEN t.taker_way = 0 THEN -1 * t.funding_fee * t.collateral_price ELSE 0 END) +
                SUM(-1 * t.rebate * t.collateral_price)) / 
                SUM(CASE WHEN t.taker_way IN (1, 3) THEN t.deal_vol * t.collateral_price ELSE 0 END) < -1 THEN -1
            ELSE (SUM(CASE WHEN t.taker_way IN (1, 2, 3, 4) THEN -1 * t.taker_pnl * t.collateral_price ELSE 0 END) +
                SUM(CASE WHEN t.taker_way = 0 THEN -1 * t.funding_fee * t.collateral_price ELSE 0 END) +
                SUM(-1 * t.rebate * t.collateral_price)) / 
                NULLIF(SUM(CASE WHEN t.taker_way IN (1, 3) THEN t.deal_vol * t.collateral_price ELSE 0 END), 0)
        END AS house_edge
    FROM
        (
            SELECT
                pair_name,
                taker_way,
                taker_fee_mode,
                taker_pnl,
                funding_fee,
                rebate,
                deal_vol,
                collateral_price,
                created_at,
                created_at + INTERVAL '8 hour' AS sg_time
            FROM
                trade_fill_fresh  -- Using trade_fill_fresh instead of surfv2_trade
            WHERE
                created_at >= NOW() - INTERVAL '24 hours'
                AND taker_fee_mode = 2
        ) t
    GROUP BY
        t.pair_name,
        DATE_TRUNC('{resolution_min} minutes', t.sg_time)
    ORDER BY
        t.pair_name,
        DATE_TRUNC('{resolution_min} minutes', t.sg_time)
    """
    
    try:
        print("Executing house edge query...")
        
        start_time = time.time()
        edge_df = pd.read_sql_query(query, conn)
        query_time = time.time() - start_time
        print(f"House edge query completed in {query_time:.2f} seconds. DataFrame shape: {edge_df.shape}")
        
        if edge_df.empty:
            print("No house edge data found.")
            return pd.DataFrame()
        
        # Convert timestamp to datetime
        edge_df['timestamp'] = pd.to_datetime(edge_df['time_block'])
        
        # Format the time label (HH:MM) for later joining
        edge_df['time_label'] = edge_df['timestamp'].dt.strftime('%H:%M')
        
        return edge_df
        
    except Exception as e:
        error_msg = str(e)
        print(f"Error fetching house edge data: {error_msg}")
        
        # Check if it's a specific table issue
        if "relation" in error_msg and "does not exist" in error_msg:
            table_name = re.search(r'"([^"]*)"', error_msg)
            if table_name:
                print(f"Table {table_name.group(1)} does not exist. Please check database schema.")
            
        return pd.DataFrame()

# Fetch all available tokens from DB
@st.cache_data(ttl=60*cache_ttl, show_spinner=False)
def fetch_all_tokens():
    """Get a list of all trading pairs/tokens"""
    try:
        cursor = conn.cursor()
        # Query tokens from trade_fill_fresh directly
        cursor.execute("""
        SELECT DISTINCT pair_name 
        FROM trade_fill_fresh 
        WHERE created_at >= NOW() - INTERVAL '24 hours'
        AND taker_fee_mode = 2
        ORDER BY pair_name
        """)
        tokens = [row[0] for row in cursor.fetchall()]
        cursor.close()
        
        if not tokens:
            # Try alternative approach from oracle price log
            print("No tokens found in trade_fill_fresh, trying oracle_price_log...")
            partition_tables = get_partition_tables(conn, start_time_sg, now_sg)
            
            if partition_tables:
                cursor = conn.cursor()
                cursor.execute(f"""
                SELECT DISTINCT pair_name 
                FROM public.{partition_tables[0]}
                WHERE source_type = 0
                ORDER BY pair_name
                """)
                tokens = [row[0] for row in cursor.fetchall()]
                cursor.close()
        
        return tokens
    except Exception as e:
        st.error(f"Error fetching tokens: {e}")
        return ["BTC", "ETH", "SOL", "DOGE", "PEPE"]  # Default fallback

# Function to calculate volatility metrics
def calculate_volatility_metrics(price_series):
    if price_series is None or len(price_series) < 2:
        return {
            'realized_vol': np.nan,
            'parkinson_vol': np.nan,
            'gk_vol': np.nan,
            'rs_vol': np.nan
        }
    
    try:
        # Calculate log returns
        log_returns = np.diff(np.log(price_series))
        
        # Annualization factor depends on resolution
        # 10min = 144 periods per day, 20min = 72, 30min = 48
        periods_per_day = int(24 * 60 / resolution_min)
        
        # Realized volatility - adjusted for selected timeframe
        realized_vol = np.std(log_returns) * np.sqrt(252 * periods_per_day)
        
        return {
            'realized_vol': realized_vol,
            'parkinson_vol': np.nan,
            'gk_vol': np.nan,
            'rs_vol': np.nan
        }
    except Exception as e:
        print(f"Error in volatility calculation: {e}")
        return {
            'realized_vol': np.nan,
            'parkinson_vol': np.nan,
            'gk_vol': np.nan,
            'rs_vol': np.nan
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

# House edge classification function
def classify_house_edge(edge):
    if pd.isna(edge):
        return ("UNKNOWN", 0, "Insufficient data")
    elif edge < negative_edge_threshold:  # System is losing money
        return ("NEGATIVE", 1, "System losing")
    elif edge < 0.1:  # Low edge
        return ("LOW", 2, "Low edge")
    elif edge < 0.3:  # Medium edge
        return ("MEDIUM", 3, "Medium edge")
    elif edge < high_edge_threshold:  # Good edge
        return ("GOOD", 4, "Good edge")
    else:  # High edge
        return ("HIGH", 5, "High edge")

# Function to generate aligned time blocks for the past 24 hours
def generate_aligned_time_blocks(current_time, resolution_min):
    """
    Generate fixed time blocks for past 24 hours,
    aligned with standard intervals (e.g., 10-minute, 20-minute)
    """
    # Round down to the nearest resolution mark
    minutes = current_time.minute
    rounded_minutes = (minutes // resolution_min) * resolution_min
    latest_complete_block_end = current_time.replace(minute=rounded_minutes, second=0, microsecond=0)
    
    # Calculate number of blocks in 24 hours
    blocks_per_day = int(24 * 60 / resolution_min)
    
    # Generate block labels for display
    blocks = []
    for i in range(blocks_per_day):
        block_end = latest_complete_block_end - timedelta(minutes=i*resolution_min)
        block_start = block_end - timedelta(minutes=resolution_min)
        block_label = f"{block_start.strftime('%H:%M')}"
        blocks.append((block_start, block_end, block_label))
    
    return blocks

# Generate aligned time blocks
aligned_time_blocks = generate_aligned_time_blocks(now_sg, resolution_min)
time_block_labels = [block[2] for block in aligned_time_blocks]

# Fetch and calculate volatility for a token
@st.cache_data(ttl=60*cache_ttl, show_spinner=False)
def fetch_and_calculate_volatility(token):
    # Convert for database query (keep as Singapore time strings as the query will handle timezone)
    start_time = start_time_sg.strftime("%Y-%m-%d %H:%M:%S")
    end_time = now_sg.strftime("%Y-%m-%d %H:%M:%S")

    # Get relevant partition tables
    partition_tables = get_partition_tables(conn, start_time_sg, now_sg)
    
    if not partition_tables:
        print(f"[{token}] No partition tables found for the specified date range")
        return None
        
    # Build query using partition tables
    query = build_price_query(
        partition_tables,
        pair_name=token,
        start_time=start_time,
        end_time=end_time
    )
    
    try:
        print(f"[{token}] Executing price query across {len(partition_tables)} partition tables")
        start_query_time = time.time()
        df = pd.read_sql_query(query, conn)
        query_time = time.time() - start_query_time
        print(f"[{token}] Query executed in {query_time:.2f} seconds. DataFrame shape: {df.shape}")

        if df.empty:
            print(f"[{token}] No price data found.")
            return None

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        
        # Create 1-minute OHLC data
        one_min_ohlc = df['final_price'].resample('1min').ohlc().dropna()
        if one_min_ohlc.empty:
            print(f"[{token}] No OHLC data after resampling.")
            return None
            
        # Calculate rolling volatility on 1-minute data
        rolling_window = max(10, int(resolution_min / 2))  # Adjust window based on resolution
        one_min_ohlc['realized_vol'] = one_min_ohlc['close'].rolling(window=rolling_window).apply(
            lambda x: calculate_volatility_metrics(x)['realized_vol']
        )
        
        # Resample to exactly the specified resolution
        resolution_vol = one_min_ohlc['realized_vol'].resample(f'{resolution_min}min', closed='left', label='left').mean().dropna()
        
        if resolution_vol.empty:
            print(f"[{token}] No {resolution_min}-min volatility data.")
            return None
            
        # Get last 24 hours
        periods_per_day = int(24 * 60 / resolution_min)
        last_24h_vol = resolution_vol.tail(periods_per_day)
        last_24h_vol = last_24h_vol.to_frame()
        
        # Store original datetime index for reference
        last_24h_vol['original_datetime'] = last_24h_vol.index
        
        # Format time label to match our aligned blocks (HH:MM format)
        last_24h_vol['time_label'] = last_24h_vol.index.strftime('%H:%M')
        
        # Calculate 24-hour average volatility
        last_24h_vol['avg_24h_vol'] = last_24h_vol['realized_vol'].mean()
        
        # Classify volatility
        last_24h_vol['vol_info'] = last_24h_vol['realized_vol'].apply(classify_volatility)
        last_24h_vol['vol_regime'] = last_24h_vol['vol_info'].apply(lambda x: x[0])
        last_24h_vol['vol_desc'] = last_24h_vol['vol_info'].apply(lambda x: x[2])
        
        # Also classify the 24-hour average
        last_24h_vol['avg_vol_info'] = last_24h_vol['avg_24h_vol'].apply(classify_volatility)
        last_24h_vol['avg_vol_regime'] = last_24h_vol['avg_vol_info'].apply(lambda x: x[0])
        last_24h_vol['avg_vol_desc'] = last_24h_vol['avg_vol_info'].apply(lambda x: x[2])
        
        # Flag extreme volatility events
        last_24h_vol['is_extreme'] = last_24h_vol['realized_vol'] >= extreme_vol_threshold
        
        print(f"[{token}] Successful Volatility Calculation")
        return last_24h_vol
    except Exception as e:
        print(f"[{token}] Error processing: {e}")
        return None

# Function to combine volatility and house edge data
def combine_volatility_and_edge(volatility_results, house_edge_data):
    """
    Combine volatility and house edge data into a single DataFrame for display.
    Returns a dictionary with tokens as keys and combined dataframes as values.
    """
    combined_results = {}
    
    for token in volatility_results.keys():
        # Get volatility data for this token
        vol_df = volatility_results[token]
        
        # Get house edge data for this token
        edge_df = house_edge_data[house_edge_data['pair_name'] == token].copy() if not house_edge_data.empty else pd.DataFrame()
        
        if vol_df is not None and not vol_df.empty:
            if not edge_df.empty:
                # Format timestamps for joining
                edge_df['time_label'] = edge_df['timestamp'].dt.strftime('%H:%M')
                
                # Merge on time_label
                merged = pd.merge(
                    vol_df.reset_index(), 
                    edge_df[['time_label', 'house_edge', 'total_platform_pnl', 'margin_amount']], 
                    on='time_label', 
                    how='outer'
                )
                
                # Classify house edge
                merged['edge_info'] = merged['house_edge'].apply(classify_house_edge)
                merged['edge_regime'] = merged['edge_info'].apply(lambda x: x[0] if x else "UNKNOWN")
                merged['edge_desc'] = merged['edge_info'].apply(lambda x: x[2] if x else "Insufficient data")
                
                # Set index back to timestamp
                if 'timestamp_x' in merged.columns:
                    merged.set_index('timestamp_x', inplace=True)
                elif 'index' in merged.columns:
                    merged.set_index('index', inplace=True)
                
                combined_results[token] = merged
            else:
                # If no edge data, still include volatility data
                vol_df['house_edge'] = np.nan
                vol_df['total_platform_pnl'] = np.nan
                vol_df['margin_amount'] = np.nan
                vol_df['edge_info'] = None
                vol_df['edge_regime'] = "UNKNOWN"
                vol_df['edge_desc'] = "Insufficient data"
                combined_results[token] = vol_df
            
    return combined_results

# Function to process tokens in parallel
def process_tokens_in_parallel(tokens, max_workers):
    """Process volatility calculations for multiple tokens in parallel"""
    results = {}
    
    def process_single_token(token):
        try:
            result = fetch_and_calculate_volatility(token)
            return (token, result)
        except Exception as e:
            print(f"Error processing {token}: {e}")
            return (token, None)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and gather results
        futures = [executor.submit(process_single_token, token) for token in tokens]
        
        # Create a progress bar
        progress_bar = st.progress(0)
        
        # Track completed futures
        completed = 0
        total = len(futures)
        
        # Wait for all futures to complete
        for future in futures:
            token, result = future.result()
            completed += 1
            progress_bar.progress(completed / total)
            
            if result is not None:
                results[token] = result
                
    return results

# Main code begins here
# Fetch all tokens
all_tokens = fetch_all_tokens()

# UI Controls
col1, col2 = st.columns([3, 1])

with col1:
    # Let user select tokens to display (or select all)
    select_all = st.checkbox("Select All Tokens", value=False)
    
    if select_all:
        selected_tokens = all_tokens[:max_tokens]  # Limit to max_tokens for performance
        if len(all_tokens) > max_tokens:
            st.info(f"Showing top {max_tokens} tokens for performance. Adjust max tokens in sidebar if needed.")
    else:
        default_tokens = all_tokens[:min(5, len(all_tokens))]
        selected_tokens = st.multiselect(
            "Select Tokens", 
            all_tokens,
            default=default_tokens
        )

with col2:
    # Add a refresh button
    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.experimental_rerun()
    
    # Show current settings
    st.caption(f"Resolution: {resolution_min} minutes")
    st.caption(f"Cache TTL: {cache_ttl} minutes")

if not selected_tokens:
    st.warning("Please select at least one token")
    st.stop()

# Show processing information
st.write(f"Processing {len(selected_tokens)} tokens with {resolution_min}-minute resolution...")

# Show the blocks we're analyzing
with st.expander("View Time Blocks Being Analyzed"):
    time_blocks_df = pd.DataFrame([(b[0].strftime('%Y-%m-%d %H:%M'), b[1].strftime('%Y-%m-%d %H:%M'), b[2]) 
                                  for b in aligned_time_blocks], 
                                 columns=['Start Time', 'End Time', 'Block Label'])
    st.dataframe(time_blocks_df)

# Initialize progress placeholder
progress_status = st.empty()
progress_status.info("Starting data processing...")

# Fetch house edge data (for all tokens)
progress_status.info("Fetching house edge data for all tokens...")
start_time = time.time()
house_edge_data = fetch_house_edge_data(resolution_min)
edge_time = time.time() - start_time
progress_status.info(f"House edge data fetched in {edge_time:.2f} seconds. Processing volatility...")

# Calculate volatility for each token in parallel
token_volatility_results = process_tokens_in_parallel(selected_tokens, parallel_workers)

# Combine volatility and house edge data
progress_status.info("Combining edge and volatility data...")
combined_results = combine_volatility_and_edge(token_volatility_results, house_edge_data)

# Final progress update
total_time = time.time() - start_time
progress_status.success(f"Processing complete in {total_time:.2f} seconds. Processed {len(combined_results)}/{len(selected_tokens)} tokens.")

# Create table for display
if combined_results:
    # Create table data for house edge and volatility together
    table_data = {}
    
    for token, df in combined_results.items():
        # Create a series with both volatility and house edge values
        # Format: "vol% / edge%"
        combined_series = pd.Series(
            [f"{(v*100):.1f}% / {(e*100):.1f}%" if not pd.isna(v) and not pd.isna(e) else 
             f"{(v*100):.1f}% / N/A" if not pd.isna(v) else
             f"N/A / {(e*100):.1f}%" if not pd.isna(e) else "N/A" 
             for v, e in zip(df['realized_vol'], df['house_edge'])],
            index=df['time_label']
        )
        table_data[token] = combined_series
    
    # Create DataFrame with all tokens
    combined_table = pd.DataFrame(table_data)
    
    # Apply the time blocks in the proper order (most recent first)
    available_times = set(combined_table.index)
    ordered_times = [t for t in time_block_labels if t in available_times]
    
    # If no matches are found in aligned blocks, fallback to the available times
    if not ordered_times and available_times:
        ordered_times = sorted(list(available_times), reverse=True)
    
    # Reindex with the ordered times
    combined_table = combined_table.reindex(ordered_times)
    
    # Create a separate table for visualization that shows numeric values
    numeric_data = {}
    for token, df in combined_results.items():
        # Create dataframes with the numeric values for coloring
        vol_series = pd.Series(df['realized_vol'] if 'realized_vol' in df else np.nan, 
                              index=df['time_label'] if 'time_label' in df else [])
        edge_series = pd.Series(df['house_edge'] if 'house_edge' in df else np.nan, 
                               index=df['time_label'] if 'time_label' in df else [])
        numeric_data[token] = {'vol': vol_series, 'edge': edge_series}
    
    # Function to color cells for combined data
    def color_combined_cells(val, token, time_label):
        if val == "N/A":
            return 'background-color: #f5f5f5; color: #666666;'  # Grey for missing
        
        # Extract vol and edge values
        vol_value = np.nan
        edge_value = np.nan
        
        if "N/A" not in val:
            # Format: "vol% / edge%"
            parts = val.split(" / ")
            if len(parts) == 2:
                try:
                    vol_value = float(parts[0].replace("%", "")) / 100
                    edge_value = float(parts[1].replace("%", "")) / 100
                except:
                    pass
        elif "/ N/A" in val:
            # Format: "vol% / N/A"
            try:
                vol_value = float(val.split(" / ")[0].replace("%", "")) / 100
            except:
                pass
        elif "N/A /" in val:
            # Format: "N/A / edge%"
            try:
                edge_value = float(val.split(" / ")[1].replace("%", "")) / 100
            except:
                pass
        
        # If we couldn't parse the values, try to get them from numeric_data
        if pd.isna(vol_value) or pd.isna(edge_value):
            try:
                if token in numeric_data and time_label in numeric_data[token]['vol'].index:
                    vol_value = numeric_data[token]['vol'].loc[time_label]
                if token in numeric_data and time_label in numeric_data[token]['edge'].index:
                    edge_value = numeric_data[token]['edge'].loc[time_label]
            except:
                pass
        
        if pd.isna(vol_value) and pd.isna(edge_value):
            return 'background-color: #f5f5f5; color: #666666;'  # Grey for missing
        
        # Create colors based on combinations of edge and volatility
        
        # Define edge colors (red to green)
        if pd.isna(edge_value):
            edge_color = "200,200,200"  # Grey for missing edge
        elif edge_value < negative_edge_threshold:  # Negative edge (system losing)
            edge_color = "255,100,100"  # Red
        elif edge_value < 0.1:  # Low edge
            edge_color = "255,200,100"  # Orange yellow
        elif edge_value < 0.3:  # Medium edge
            edge_color = "200,255,100"  # Yellow green
        elif edge_value < high_edge_threshold:  # Good edge
            edge_color = "150,255,150"  # Light green
        else:  # High edge
            edge_color = "100,255,100"  # Green
            
        # Define volatility intensity (transparency)
        if pd.isna(vol_value):
            vol_alpha = 0.5  # Medium transparency for missing vol
            text_color = "black"
        elif vol_value < 0.3:  # Low volatility
            vol_alpha = 0.4
            text_color = "black"
        elif vol_value < 0.6:  # Medium volatility
            vol_alpha = 0.6
            text_color = "black"
        elif vol_value < 1.0:  # High volatility
            vol_alpha = 0.8
            text_color = "black"
        else:  # Extreme volatility
            vol_alpha = 1.0
            text_color = "white"
            
        return f'background-color: rgba({edge_color}, {vol_alpha}); color: {text_color}'
    
    # Apply styling for the combined table
    styled_table = combined_table.style.apply(
        lambda x: pd.Series([color_combined_cells(val, x.name, idx) for idx, val in x.items()], index=x.index),
        axis=1
    )
    
    # Display the matrix
    st.markdown(f"## Combined {resolution_min}-Minute Edge and Volatility Matrix")
    st.markdown("### Values shown as: Volatility % / House Edge %")
    st.markdown("""
    #### Color Legend: 
    - **Background Color**: House Edge (Red = Negative, Yellow = Low, Green = High)
    - **Color Intensity**: Volatility (Darker = Higher Volatility)
    """)
    st.dataframe(styled_table, height=700, use_container_width=True)
    
    # Create a summary dashboard
    st.subheader("24-Hour Summary Dashboard")
    
    # Summary metrics
    summary_data = []
    for token, df in combined_results.items():
        if not df.empty:
            avg_vol = df['realized_vol'].mean() if 'realized_vol' in df and not df['realized_vol'].isna().all() else np.nan
            avg_edge = df['house_edge'].mean() if 'house_edge' in df and not df['house_edge'].isna().all() else np.nan
            total_pnl = df['total_platform_pnl'].sum() if 'total_platform_pnl' in df and not df['total_platform_pnl'].isna().all() else np.nan
            total_margin = df['margin_amount'].sum() if 'margin_amount' in df and not df['margin_amount'].isna().all() else np.nan
            
            # Get peak volatility time
            if 'realized_vol' in df and not df['realized_vol'].isna().all():
                max_vol_idx = df['realized_vol'].idxmax() 
                max_vol_time = df.loc[max_vol_idx, 'time_label'] if 'time_label' in df.columns else "N/A"
            else:
                max_vol_time = "N/A"
            
            # Get best edge time
            if 'house_edge' in df and not df['house_edge'].isna().all():
                max_edge_idx = df['house_edge'].idxmax()
                max_edge_time = df.loc[max_edge_idx, 'time_label'] if 'time_label' in df.columns else "N/A"
            else:
                max_edge_time = "N/A"
            
            summary_data.append({
                'Token': token,
                'Avg Vol (%)': (avg_vol * 100).round(1) if not pd.isna(avg_vol) else np.nan,
                'Avg Edge (%)': (avg_edge * 100).round(1) if not pd.isna(avg_edge) else np.nan,
                'Total PNL': int(total_pnl) if not pd.isna(total_pnl) else np.nan,
                'Total Margin': int(total_margin) if not pd.isna(total_margin) else np.nan,
                'Peak Vol Time': max_vol_time,
                'Best Edge Time': max_edge_time
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        
        # Sort by average edge (high to low)
        summary_df = summary_df.sort_values(by='Avg Edge (%)', ascending=False)
        
        # Add rank column
        summary_df.insert(0, 'Rank', range(1, len(summary_df) + 1))
        
        # Reset the index to remove it
        summary_df = summary_df.reset_index(drop=True)
        
        # Format summary table with colors
        def color_vol_column(val):
            if pd.isna(val):
                return ''
            elif val < 30:
                return 'color: green'
            elif val < 60:
                return 'color: #aaaa00'
            elif val < 100:
                return 'color: orange'
            else:
                return 'color: red'
        
        def color_edge_column(val):
            if pd.isna(val):
                return ''
            elif val < negative_edge_threshold * 100:
                return 'color: red'
            elif val < 10:
                return 'color: orange'
            elif val < 30:
                return 'color: #aaaa00'
            elif val < high_edge_threshold * 100:
                return 'color: lightgreen'
            else:
                return 'color: green; font-weight: bold'
        
        def color_pnl_column(val):
            if pd.isna(val):
                return ''
            elif val < 0:
                return 'color: red'
            else:
                return 'color: green'
        
        # Apply styling
        styled_summary = summary_df.style\
            .applymap(color_vol_column, subset=['Avg Vol (%)'])\
            .applymap(color_edge_column, subset=['Avg Edge (%)'])\
            .applymap(color_pnl_column, subset=['Total PNL'])
        
        # Display the styled dataframe
        st.dataframe(styled_summary, height=500, use_container_width=True)
        
        # Create summary metrics at the top
        total_pnl = sum(row['Total PNL'] for row in summary_data if not pd.isna(row['Total PNL']))
        
        # Calculate weighted average edge if we have valid data
        valid_data = [(row['Avg Edge (%)'], row['Total Margin']) for row in summary_data 
                     if not pd.isna(row['Avg Edge (%)']) and not pd.isna(row['Total Margin']) and row['Total Margin'] != 0]
        
        if valid_data:
            total_margin = sum(margin for _, margin in valid_data)
            if total_margin > 0:
                avg_portfolio_edge = sum(edge * margin / 100 for edge, margin in valid_data) / total_margin * 100
            else:
                avg_portfolio_edge = np.nan
        else:
            avg_portfolio_edge = np.nan
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Portfolio PNL (24h)", f"${total_pnl:,.2f}")
        col2.metric("Weighted Avg Edge", f"{avg_portfolio_edge:.2f}%" if not pd.isna(avg_portfolio_edge) else "N/A")
        col3.metric("Tokens Analyzed", f"{len(summary_data)}")
    else:
        st.warning("No summary data available.")
    
    # Identify and display opportunities
    st.subheader("Trading Opportunities")
    
    # 1. High Edge with Low Volatility (Good Risk/Reward)
    high_edge_tokens = [row for row in summary_data 
                        if not pd.isna(row['Avg Edge (%)']) 
                        and not pd.isna(row['Avg Vol (%)'])
                        and row['Avg Edge (%)'] > 20
                        and row['Avg Vol (%)'] < 60]
    
    # 2. Extreme volatility events
    extreme_vol_events = []
    for token, df in combined_results.items():
        if not df.empty and 'realized_vol' in df.columns:
            extreme_periods = df[df['realized_vol'] >= extreme_vol_threshold]
            for idx, row in extreme_periods.iterrows():
                vol_value = float(row['realized_vol']) if 'realized_vol' in row and not pd.isna(row['realized_vol']) else 0.0
                edge_value = float(row['house_edge']) if 'house_edge' in row and not pd.isna(row['house_edge']) else 0.0
                time_label = str(row['time_label']) if 'time_label' in row and not pd.isna(row['time_label']) else "Unknown"
                
                extreme_vol_events.append({
                    'Token': token,
                    'Time': time_label,
                    'Volatility (%)': round(vol_value * 100, 1),
                    'Edge (%)': round(edge_value * 100, 1),
                    'Full Timestamp': idx.strftime('%Y-%m-%d %H:%M') if hasattr(idx, 'strftime') else str(idx)
                })
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### High Edge / Low Volatility Pairs")
        if high_edge_tokens:
            for token in high_edge_tokens:
                edge = token['Avg Edge (%)']
                vol = token['Avg Vol (%)']
                edge_color = "green" if edge > 30 else "lightgreen"
                st.markdown(f"- **{token['Token']}**: Edge: <span style='color:{edge_color}'>{edge:.1f}%</span>, Vol: <span style='color:green'>{vol:.1f}%</span>", unsafe_allow_html=True)
        else:
            st.markdown("*No tokens in this category*")
    
    with col2:
        st.markdown("### Extreme Volatility Events")
        if extreme_vol_events:
            extreme_vol_events_sorted = sorted(extreme_vol_events, key=lambda x: x['Volatility (%)'], reverse=True)
            for event in extreme_vol_events_sorted[:5]:  # Show top 5
                token = event['Token']
                time = event['Time']
                vol = event['Volatility (%)']
                edge = event['Edge (%)']
                edge_color = "red" if edge < 0 else "green"
                st.markdown(f"- **{token}** at **{time}**: Vol: <span style='color:red'>{vol}%</span>, Edge: <span style='color:{edge_color}'>{edge}%</span>", unsafe_allow_html=True)
            
            if len(extreme_vol_events) > 5:
                st.markdown(f"*... and {len(extreme_vol_events) - 5} more extreme events*")
        else:
            st.markdown("*No events in this category*")
    
    # Visualize the relationship between edge and volatility
    st.subheader("Edge vs. Volatility Relationship")
    
    # Prepare data for scatter plot
    scatter_data = []
    for token, df in combined_results.items():
        if not df.empty:
            avg_vol = df['realized_vol'].mean() if 'realized_vol' in df and not df['realized_vol'].isna().all() else np.nan
            avg_edge = df['house_edge'].mean() if 'house_edge' in df and not df['house_edge'].isna().all() else np.nan
            if not pd.isna(avg_vol) and not pd.isna(avg_edge):
                scatter_data.append({
                    'Token': token,
                    'Volatility': avg_vol * 100,  # Convert to percentage
                    'Edge': avg_edge * 100        # Convert to percentage
                })
    
    if scatter_data:
        scatter_df = pd.DataFrame(scatter_data)
        
        # Create scatter plot
        fig = go.Figure()
        
        # Add scatter points
        fig.add_trace(go.Scatter(
            x=scatter_df['Volatility'],
            y=scatter_df['Edge'],
            mode='markers+text',
            marker=dict(
                size=12,
                color=scatter_df['Volatility'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Volatility (%)"),
                line=dict(width=1, color='black')
            ),
            text=scatter_df['Token'],
            textposition="top center",
            name='Tokens'
        ))
        
        # Add quadrant lines
        fig.add_shape(
            type="line",
            x0=0, x1=max(scatter_df['Volatility']) * 1.1,
            y0=0, y1=0,
            line=dict(color="black", width=1, dash="dash")
        )
        
        fig.add_shape(
            type="line",
            x0=60, x1=60,
            y0=min(scatter_df['Edge']) * 1.1, y1=max(scatter_df['Edge']) * 1.1,
            line=dict(color="black", width=1, dash="dash")
        )
        
        # Add quadrant labels
        fig.add_annotation(
            x=30, y=20,
            text="IDEAL:<br>High Edge, Low Vol",
            showarrow=False,
            font=dict(size=12, color="green")
        )
        
        fig.add_annotation(
            x=80, y=20,
            text="GOOD:<br>High Edge, High Vol",
            showarrow=False,
            font=dict(size=12, color="darkgreen")
        )
        
        fig.add_annotation(
            x=30, y=-20,
            text="POOR:<br>Negative Edge, Low Vol",
            showarrow=False,
            font=dict(size=12, color="red")
        )
        
        fig.add_annotation(
            x=80, y=-20,
            text="AVOID:<br>Negative Edge, High Vol",
            showarrow=False,
            font=dict(size=12, color="darkred")
        )
        
        # Update layout
        fig.update_layout(
            title="Average Edge vs. Volatility by Token (24h)",
            xaxis_title="Volatility (%)",
            yaxis_title="Edge (%)",
            height=600,
            template="plotly_white",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Insufficient data for scatter plot.")

with st.expander("Understanding the Matrix"):
    st.markdown(f"""
    ### How to Read This Table
    
    The {resolution_min}-Minute System Edge & Volatility Matrix shows combined data for each token across time intervals during the last 24 hours. Each cell contains both volatility and house edge information.
    
    #### Cell Values
    Each cell shows **Volatility % / House Edge %** for that specific token and time period.
    
    #### Color Coding
    - **Background Color** represents the house edge:
        - **Red**: Negative edge (system is losing money)
        - **Orange/Yellow**: Low/medium edge
        - **Green**: High edge (good for the system)
    
    - **Color Intensity** represents volatility:
        - **Lighter shades**: Lower volatility
        - **Darker shades**: Higher volatility
        - **White text**: Extreme volatility (≥100%)
    
    #### Trading Implications
    - **Green cells with light background**: High edge with low volatility - best trading opportunities
    - **Green cells with dark background**: High edge but high volatility - good but riskier
    - **Red cells with dark background**: Negative edge with high volatility - avoid these
    
    #### Technical Details
    - Volatility is calculated as the standard deviation of log returns, annualized 
    - House edge represents the system's profitability, calculated as (platform PNL / total margin)
    - Edge values range from -1 to 1, with positive values indicating system profit
    - All times are shown in Singapore timezone
    - Missing values (gray cells) indicate insufficient data
    """)