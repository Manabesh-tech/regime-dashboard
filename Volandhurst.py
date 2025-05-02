# Save this as pages/05_5min_Volatility_Table.py in your Streamlit app folder

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
    page_title="5min Volatility Table",
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
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("5-Minute Volatility Table")
st.subheader("All Trading Pairs - Last 12 Hours (Singapore Time)")

# Define parameters for the 5-minute timeframe
timeframe = "5min"
lookback_hours = 12  # 12 hours instead of 24
rolling_window = 10  # Reduced window size for 5min data to improve calculation speed
expected_points = 144  # Expected data points per pair over 12 hours (12 hours * 12 5-min periods per hour)
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
        # Query for Surf data (source_type = 0) with explicit timezone handling
        query = f"""
        SELECT 
            pair_name,
            created_at AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore' AS timestamp,
            final_price
        FROM 
            public.{table}
        WHERE 
            created_at >= '{start_time}'::timestamp AT TIME ZONE 'Asia/Singapore' AT TIME ZONE 'UTC'
            AND created_at <= '{end_time}'::timestamp AT TIME ZONE 'Asia/Singapore' AT TIME ZONE 'UTC'
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

# UI Controls - OPTIMIZED layout for better user experience
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    # Let user select tokens to display (or select all)
    select_all = st.checkbox("Select All Tokens", value=False)  # Default to false to reduce initial load
    
    if select_all:
        selected_tokens = all_tokens
    else:
        selected_tokens = st.multiselect(
            "Select Tokens", 
            all_tokens,
            default=all_tokens[:5] if len(all_tokens) > 5 else all_tokens
        )

with col2:
    # Add a refresh button
    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.experimental_rerun()

with col3:
    # Add option to adjust lookback period
    lookback_option = st.selectbox(
        "Lookback Period",
        options=[6, 12, 24],
        index=1,  # Default to 12 hours
        format_func=lambda x: f"{x} hours"
    )
    lookback_hours = lookback_option

# Add a debug expander to the UI
with st.expander("Debug Information", expanded=False):
    st.subheader("Time Range Information")
    st.write(f"Current Singapore Time: {now_sg.strftime('%Y-%m-%d %H:%M:%S')}")
    st.write(f"Requested Lookback Hours: {lookback_hours}")
    st.write(f"Query Start Time: {(now_sg - timedelta(hours=lookback_hours)).strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Add a button to show available partition tables
    if st.button("Check Available Partition Tables"):
        start_time_sg = now_sg - timedelta(hours=lookback_hours)
        partition_tables = get_partition_tables(conn, start_time_sg, now_sg)
        
        if partition_tables:
            st.success(f"Found {len(partition_tables)} partition tables:")
            st.write(partition_tables)
            
            # Query for the first and last timestamp in each table
            for table in partition_tables:
                try:
                    query = f"""
                    SELECT MIN(created_at AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') as min_time,
                           MAX(created_at AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') as max_time,
                           COUNT(*) as record_count
                    FROM public.{table}
                    WHERE source_type = 0
                    """
                    df = pd.read_sql_query(query, conn)
                    st.write(f"Table: {table}")
                    st.write(f"  Time range: {df['min_time'][0]} to {df['max_time'][0]}")
                    st.write(f"  Record count: {df['record_count'][0]}")
                except Exception as e:
                    st.error(f"Error querying table {table}: {e}")
        else:
            st.error("No partition tables found for the specified time range.")
    
    # Add a button to test data for a specific token
    test_token = st.text_input("Test token (e.g., BTC/USDT)", "BTC/USDT")
    if st.button("Test Data Availability"):
        start_time_sg = now_sg - timedelta(hours=lookback_hours)
        start_time = start_time_sg.strftime("%Y-%m-%d %H:%M:%S")
        end_time = now_sg.strftime("%Y-%m-%d %H:%M:%S")
        
        partition_tables = get_partition_tables(conn, start_time_sg, now_sg)
        if partition_tables:
            query = build_query_for_partition_tables(
                partition_tables,
                pair_name=test_token,
                start_time=start_time,
                end_time=end_time
            )
            
            try:
                df = pd.read_sql_query(query, conn)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                if not df.empty:
                    min_time = df['timestamp'].min()
                    max_time = df['timestamp'].max()
                    actual_hours = (max_time - min_time).total_seconds() / 3600
                    
                    st.success(f"Found {len(df)} records for {test_token}")
                    st.write(f"Time range: {min_time} to {max_time} ({actual_hours:.1f} hours)")
                    
                    # Sample of the data
                    st.write("Sample data (first 5 rows):")
                    st.dataframe(df.head())
                    
                    # Count of records per hour
                    df['hour'] = df['timestamp'].dt.floor('H')
                    hourly_counts = df.groupby('hour').size().reset_index(name='count')
                    hourly_counts.columns = ['Hour', 'Record Count']
                    st.write("Records per hour:")
                    st.dataframe(hourly_counts)
                else:
                    st.error(f"No data found for {test_token} in the specified time range.")
            except Exception as e:
                st.error(f"Error testing data availability: {e}")
        else:
            st.error("No partition tables found for the specified time range.")

if not selected_tokens:
    st.warning("Please select at least one token")
    st.stop()

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

# Function to generate aligned 5-minute time blocks for the past 12 hours
def generate_aligned_time_blocks(current_time, hours_back=12):
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
    for i in range(hours_back * 12):  # 12 hours of 5-minute blocks = 144 blocks
        block_end = latest_complete_block_end - timedelta(minutes=i*5)
        block_start = block_end - timedelta(minutes=5)
        block_label = f"{block_start.strftime('%H:%M')}"
        blocks.append((block_start, block_end, block_label))
    
    return blocks

# Generate aligned time blocks
aligned_time_blocks = generate_aligned_time_blocks(now_sg, lookback_hours)
time_block_labels = [block[2] for block in aligned_time_blocks]

# Fetch and calculate volatility for a token with 5min timeframe - OPTIMIZED query
@st.cache_data(ttl=300, show_spinner="Calculating volatility metrics...")
def fetch_and_calculate_volatility(token):
    # Get current time in Singapore timezone
    now_utc = datetime.now(pytz.utc)
    now_sg = now_utc.astimezone(singapore_timezone)
    
    # Extend the lookback period by 50% to ensure we get enough data
    extended_lookback = int(lookback_hours * 1.5)
    start_time_sg = now_sg - timedelta(hours=extended_lookback)
    
    # Convert for database query
    start_time = start_time_sg.strftime("%Y-%m-%d %H:%M:%S")
    end_time = now_sg.strftime("%Y-%m-%d %H:%M:%S")

    # Get relevant partition tables - search additional days if needed
    partition_tables = get_partition_tables(conn, start_time_sg, now_sg)
    
    if not partition_tables:
        # Try looking back one more day to find tables
        alt_start_time_sg = start_time_sg - timedelta(days=1)
        partition_tables = get_partition_tables(conn, alt_start_time_sg, now_sg)
        if partition_tables:
            print(f"[{token}] Found partition tables by extending search to {alt_start_time_sg.strftime('%Y-%m-%d')}")
        else:
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

        # Additional validation for timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        
        # Work directly with timestamps for 5min resampling
        price_data = df['final_price'].dropna()
        if price_data.empty:
            print(f"[{token}] No data after cleaning.")
            return None
        
        # Use more robust resampling with explicit timezone
        # This ensures time blocks align properly with Singapore time
        five_min_ohlc = price_data.resample('5min', closed='left', label='left').ohlc().dropna()
        
        if five_min_ohlc.empty:
            print(f"[{token}] No 5-min data after resampling.")
            return None
        
        # Calculate rolling volatility directly on 5-minute close prices
        five_min_ohlc['realized_vol'] = five_min_ohlc['close'].rolling(window=rolling_window).apply(
            lambda x: calculate_volatility_metrics(x)['realized_vol']
        )
        
        # Get exactly the requested number of hours worth of data
        blocks_needed = lookback_hours * 12  # Number of 5-minute blocks in lookback period
        
        # Take only the most recent blocks_needed points, or all if less than that
        recent_data = five_min_ohlc.tail(blocks_needed)
        last_period_vol = recent_data['realized_vol']
        
        if last_period_vol.empty:
            print(f"[{token}] No 5-min volatility data.")
            return None
        
        last_period_vol = last_period_vol.to_frame()
        
        # Store original datetime index for reference
        last_period_vol['original_datetime'] = last_period_vol.index
        
        # Format time label to match our aligned blocks (HH:MM format)
        last_period_vol['time_label'] = last_period_vol.index.strftime('%H:%M')
        
        # Calculate average volatility over lookback period
        last_period_vol['avg_period_vol'] = last_period_vol['realized_vol'].mean()
        
        # Classify volatility
        last_period_vol['vol_info'] = last_period_vol['realized_vol'].apply(classify_volatility)
        last_period_vol['vol_regime'] = last_period_vol['vol_info'].apply(lambda x: x[0])
        last_period_vol['vol_desc'] = last_period_vol['vol_info'].apply(lambda x: x[2])
        
        # Also classify the average
        last_period_vol['avg_vol_info'] = last_period_vol['avg_period_vol'].apply(classify_volatility)
        last_period_vol['avg_vol_regime'] = last_period_vol['avg_vol_info'].apply(lambda x: x[0])
        last_period_vol['avg_vol_desc'] = last_period_vol['avg_vol_info'].apply(lambda x: x[2])
        
        # Flag extreme volatility events
        last_period_vol['is_extreme'] = last_period_vol['realized_vol'] >= extreme_vol_threshold
        
        print(f"[{token}] Successful Volatility Calculation")
        return last_period_vol
    except Exception as e:
        st.error(f"Error processing {token}: {e}")
        print(f"[{token}] Error processing: {e}")
        return None

# Show the blocks we're analyzing - OPTIMIZED to be in expander for cleaner UI
with st.expander("View Time Blocks Being Analyzed", expanded=False):
    time_blocks_df = pd.DataFrame([(b[0].strftime('%Y-%m-%d %H:%M'), b[1].strftime('%Y-%m-%d %H:%M'), b[2]) 
                                  for b in aligned_time_blocks], 
                                 columns=['Start Time', 'End Time', 'Block Label'])
    st.dataframe(time_blocks_df)

# Optimize the progress bar with batched processing
progress_bar = st.progress(0)
status_text = st.empty()

# Process tokens in batches to increase perceived speed
token_results = {}
batch_size = min(5, len(selected_tokens))  # Process up to 5 tokens simultaneously
for i in range(0, len(selected_tokens), batch_size):
    batch = selected_tokens[i:i+batch_size]
    
    # Update progress
    progress_bar.progress(i / len(selected_tokens) if len(selected_tokens) > 0 else 0)
    status_text.text(f"Processing batch {i//batch_size + 1}/{(len(selected_tokens)-1)//batch_size + 1} ({len(batch)} tokens)")
    
    # Process tokens in current batch (could be parallelized in a future version)
    for token in batch:
        try:
            result = fetch_and_calculate_volatility(token)
            if result is not None:
                token_results[token] = result
        except Exception as e:
            st.error(f"Error processing token {token}: {e}")
            print(f"Error processing token {token} in main loop: {e}")

# Final progress update
progress_bar.progress(1.0)
status_text.text(f"Processed {len(token_results)}/{len(selected_tokens)} tokens successfully")

# Create table for display - OPTIMIZED display for 5min data
if token_results:
    # Create table data
    table_data = {}
    for token, df in token_results.items():
        vol_series = df.set_index('time_label')['realized_vol']
        table_data[token] = vol_series
    
    # Create DataFrame with all tokens
    vol_table = pd.DataFrame(table_data)
    
    # Apply the time blocks in the proper order (most recent first)
    available_times = set(vol_table.index)
    ordered_times = [t for t in time_block_labels if t in available_times]
    
    # If no matches are found in aligned blocks, fallback to the available times
    if not ordered_times and available_times:
        ordered_times = sorted(list(available_times), reverse=True)
    
    # Reindex with the ordered times
    vol_table = vol_table.reindex(ordered_times)
    
    # Convert from decimal to percentage and round to 1 decimal place
    vol_table = (vol_table * 100).round(1)
    
    def color_cells(val):
        if pd.isna(val):
            return 'background-color: #f5f5f5; color: #666666;'  # Grey for missing
        elif val < 30:  # Low volatility - green
            intensity = max(0, min(255, int(255 * val / 30)))
            return f'background-color: rgba({255-intensity}, 255, {255-intensity}, 0.7); color: black'
        elif val < 60:  # Medium volatility - yellow
            intensity = max(0, min(255, int(255 * (val - 30) / 30)))
            return f'background-color: rgba(255, 255, {255-intensity}, 0.7); color: black'
        elif val < 100:  # High volatility - orange
            intensity = max(0, min(255, int(255 * (val - 60) / 40)))
            return f'background-color: rgba(255, {255-(intensity//2)}, 0, 0.7); color: black'
        else:  # Extreme volatility - red
            return 'background-color: rgba(255, 0, 0, 0.7); color: white'
    
    styled_table = vol_table.style.applymap(color_cells)
    st.markdown(f"## Volatility Table (5min timeframe, Last {lookback_hours} hours, Singapore Time)")
    st.markdown("### Color Legend: <span style='color:green'>Low Vol</span>, <span style='color:#aaaa00'>Medium Vol</span>, <span style='color:orange'>High Vol</span>, <span style='color:red'>Extreme Vol</span>", unsafe_allow_html=True)
    st.markdown("Values shown as annualized volatility percentage")
    
    # OPTIMIZATION: Set a maximum height for the table to avoid overwhelming the page
    max_height = min(700, 100 + 20 * len(ordered_times))  # Base height + rows
    st.dataframe(styled_table, height=max_height, use_container_width=True)
    
    # Create ranking table based on average volatility
    st.subheader(f"Volatility Ranking ({lookback_hours}-Hour Average, Descending Order)")
    
    ranking_data = []
    for token, df in token_results.items():
        if not df.empty and 'avg_period_vol' in df.columns and not df['avg_period_vol'].isna().all():
            avg_vol = df['avg_period_vol'].iloc[0]  # All rows have the same avg value
            vol_regime = df['avg_vol_desc'].iloc[0]
            max_vol = df['realized_vol'].max()
            min_vol = df['realized_vol'].min()
            ranking_data.append({
                'Token': token,
                'Avg Vol (%)': (avg_vol * 100).round(1),
                'Regime': vol_regime,
                'Max Vol (%)': (max_vol * 100).round(1),
                'Min Vol (%)': (min_vol * 100).round(1),
                'Vol Range (%)': ((max_vol - min_vol) * 100).round(1)
            })
    
    if ranking_data:
        ranking_df = pd.DataFrame(ranking_data)
        # Sort by average volatility (high to low)
        ranking_df = ranking_df.sort_values(by='Avg Vol (%)', ascending=False)
        # Add rank column
        ranking_df.insert(0, 'Rank', range(1, len(ranking_df) + 1))
        
        # Reset the index to remove it
        ranking_df = ranking_df.reset_index(drop=True)
        
        # Format ranking table with colors
        def color_regime(val):
            if pd.isna(val) or not isinstance(val, str):
                return ''
            elif 'Low' in val:
                return 'color: green'
            elif 'Medium' in val:
                return 'color: #aaaa00'
            elif 'High' in val:
                return 'color: orange'
            elif 'Extreme' in val:
                return 'color: red'
            return ''
        
        def color_value(val):
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
        
        # Apply styling
        styled_ranking = ranking_df.style\
            .applymap(color_regime, subset=['Regime'])\
            .applymap(color_value, subset=['Avg Vol (%)', 'Max Vol (%)', 'Min Vol (%)'])
        
        # Display the styled dataframe
        st.dataframe(styled_ranking, height=min(500, 100 + 35 * len(ranking_df)), use_container_width=True)
    else:
        st.warning("No ranking data available.")
    
    # Identify and display extreme volatility events
    st.subheader("Extreme Volatility Events (>= 100% Annualized)")
    
    extreme_events = []
    for token, df in token_results.items():
        if not df.empty and 'is_extreme' in df.columns:
            extreme_periods = df[df['is_extreme']]
            for idx, row in extreme_periods.iterrows():
                # Safely access values with explicit casting to avoid attribute errors
                vol_value = float(row['realized_vol']) if not pd.isna(row['realized_vol']) else 0.0
                time_label = str(row['time_label']) if 'time_label' in row and not pd.isna(row['time_label']) else "Unknown"
                
                extreme_events.append({
                    'Token': token,
                    'Time': time_label,
                    'Volatility (%)': round(vol_value * 100, 1),
                    'Full Timestamp': idx.strftime('%Y-%m-%d %H:%M')
                })
    
    if extreme_events:
        extreme_df = pd.DataFrame(extreme_events)
        # Sort by volatility (highest first)
        extreme_df = extreme_df.sort_values(by='Volatility (%)', ascending=False)
        
        # Reset the index to remove it
        extreme_df = extreme_df.reset_index(drop=True)
        
        # Display the dataframe
        st.dataframe(extreme_df, height=min(300, 100 + 35 * len(extreme_df)), use_container_width=True)
        
        # Create a more visually appealing list of extreme events
        with st.expander("Extreme Volatility Events Detail", expanded=False):
            # Only process top 10 events if there are any
            top_events = extreme_events[:min(10, len(extreme_events))]
            for i, event in enumerate(top_events):
                token = event['Token']
                time = event['Time']
                vol = event['Volatility (%)']
                date = event['Full Timestamp'].split(' ')[0]
                
                st.markdown(f"**{i+1}. {token}** at **{time}** on {date}: <span style='color:red; font-weight:bold;'>{vol}%</span> volatility", unsafe_allow_html=True)
            
            if len(extreme_events) > 10:
                st.markdown(f"*... and {len(extreme_events) - 10} more extreme events*")
        
    else:
        st.info("No extreme volatility events detected in the selected tokens.")
    
    # Average Volatility Distribution - OPTIMIZED with simple metrics display
    # Using columns to make it more compact
    st.subheader(f"{lookback_hours}-Hour Average Volatility Overview (Singapore Time)")
    
    # OPTIMIZATION: Place metrics in a more compact layout
    col1, col2 = st.columns(2)
    
    with col1:
        avg_values = {}
        for token, df in token_results.items():
            if not df.empty and 'avg_period_vol' in df.columns and not df['avg_period_vol'].isna().all():
                avg = df['avg_period_vol'].iloc[0]  # All rows have the same avg value
                regime = df['avg_vol_desc'].iloc[0]
                avg_values[token] = (avg, regime)
        
        if avg_values:
            low_vol = sum(1 for v, r in avg_values.values() if v < 0.3)
            medium_vol = sum(1 for v, r in avg_values.values() if 0.3 <= v < 0.6)
            high_vol = sum(1 for v, r in avg_values.values() if 0.6 <= v < 1.0)
            extreme_vol = sum(1 for v, r in avg_values.values() if v >= 1.0)
            total = low_vol + medium_vol + high_vol + extreme_vol
            
            col1a, col1b = st.columns(2)
            col2a, col2b = st.columns(2)
            
            if total > 0:
                col1a.metric("Low Vol", f"{low_vol} ({low_vol/total*100:.1f}%)")
                col1b.metric("Medium Vol", f"{medium_vol} ({medium_vol/total*100:.1f}%)")
                col2a.metric("High Vol", f"{high_vol} ({high_vol/total*100:.1f}%)")
                col2b.metric("Extreme Vol", f"{extreme_vol} ({extreme_vol/total*100:.1f}%)")
            else:
                st.warning("No volatility data to calculate percentages")
        else:
            st.warning("No average volatility data available for the selected tokens.")
    
    with col2:
        # Simple pie chart for volatility distribution
        if 'avg_values' in locals() and avg_values and total > 0:
            labels = ['Low Vol', 'Medium Vol', 'High Vol', 'Extreme Vol']
            values = [low_vol, medium_vol, high_vol, extreme_vol]
            colors = ['rgba(100,255,100,0.8)', 'rgba(255,255,100,0.8)', 'rgba(255,165,0,0.8)', 'rgba(255,0,0,0.8)']
            
            fig = go.Figure(data=[go.Pie(labels=labels, values=values, marker=dict(colors=colors, line=dict(color='#000000', width=2)), textinfo='label+percent', hole=.3)])
            fig.update_layout(
                title=f"{lookback_hours}-Hour Average Volatility Distribution",
                height=300,
                font=dict(color="#000000", size=12),
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

    # OPTIMIZATION: Put detailed token lists in expanders to save space
    with st.expander("Token Volatility Categories", expanded=False):
        # Create columns for each volatility category
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)
        
        with col1:
            st.markdown("### Low Average Volatility Tokens")
            lv_tokens = [(t, v*100, r) for t, (v, r) in avg_values.items() if v < 0.3]
            lv_tokens.sort(key=lambda x: x[1])
            if lv_tokens:
                for token, value, regime in lv_tokens:
                    st.markdown(f"- **{token}**: <span style='color:green'>{value:.1f}%</span> ({regime})", unsafe_allow_html=True)
            else:
                st.markdown("*No tokens in this category*")
        
        with col2:
            st.markdown("### Medium Average Volatility Tokens")
            mv_tokens = [(t, v*100, r) for t, (v, r) in avg_values.items() if 0.3 <= v < 0.6]
            mv_tokens.sort(key=lambda x: x[1])
            if mv_tokens:
                for token, value, regime in mv_tokens:
                    st.markdown(f"- **{token}**: <span style='color:#aaaa00'>{value:.1f}%</span> ({regime})", unsafe_allow_html=True)
            else:
                st.markdown("*No tokens in this category*")
        
        with col3:
            st.markdown("### High Average Volatility Tokens")
            hv_tokens = [(t, v*100, r) for t, (v, r) in avg_values.items() if 0.6 <= v < 1.0]
            hv_tokens.sort(key=lambda x: x[1])
            if hv_tokens:
                for token, value, regime in hv_tokens:
                    st.markdown(f"- **{token}**: <span style='color:orange'>{value:.1f}%</span> ({regime})", unsafe_allow_html=True)
            else:
                st.markdown("*No tokens in this category*")
        
        with col4:
            st.markdown("### Extreme Average Volatility Tokens")
            ev_tokens = [(t, v*100, r) for t, (v, r) in avg_values.items() if v >= 1.0]
            ev_tokens.sort(key=lambda x: x[1], reverse=True)
            if ev_tokens:
                for token, value, regime in ev_tokens:
                    st.markdown(f"- **{token}**: <span style='color:red'>{value:.1f}%</span> ({regime})", unsafe_allow_html=True)
            else:
                st.markdown("*No tokens in this category*")
else:
    st.warning("No volatility data available for the selected tokens.")

with st.expander("Understanding the Volatility Table", expanded=False):
    st.markdown("""
    ### How to Read This Table
    This table shows annualized volatility values for all selected tokens over the last 12 hours using 5-minute bars.
    Each row represents a specific 5-minute time period, with times shown in Singapore time. The table is sorted with the most recent 5-minute period at the top.
    
    **Color coding:**
    - **Green** (< 30%): Low volatility
    - **Yellow** (30-60%): Medium volatility
    - **Orange** (60-100%): High volatility
    - **Red** (> 100%): Extreme volatility
    
    **The intensity of the color indicates the strength of the volatility:**
    - Darker green = Lower volatility
    - Darker red = Higher volatility
    
    **Ranking Table:**
    The ranking table sorts tokens by their average volatility over the selected lookback period, from highest to lowest.
    
    **Extreme Volatility Events:**
    These are specific 5-minute periods where a token's annualized volatility exceeded 100%.
    
    **Technical details:**
    - Volatility is calculated as the standard deviation of log returns, annualized to represent the expected price variation over a year
    - Values shown are in percentage (e.g., 50.0 means 50% annualized volatility)
    - The calculation uses a rolling window of 10 price points for 5-minute data (versus 20 for 30-minute data)
    - Missing values (light gray cells) indicate insufficient data for calculation
    """)