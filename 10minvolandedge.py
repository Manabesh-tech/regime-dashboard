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
    page_title="10-Minute Volatility Matrix",
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
st.title("10-Minute Volatility Matrix")
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

# Fetch all available tokens from DB
@st.cache_data(ttl=60*cache_ttl, show_spinner=False)
def fetch_all_tokens():
    """Get a list of all trading pairs/tokens"""
    try:
        # Try from oracle price log
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
        
        return []
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

# Calculate volatility for each token in parallel
start_time = time.time()
token_volatility_results = process_tokens_in_parallel(selected_tokens, parallel_workers)

# Final progress update
total_time = time.time() - start_time
progress_status.success(f"Processing complete in {total_time:.2f} seconds. Processed {len(token_volatility_results)}/{len(selected_tokens)} tokens.")

# Create table for display
if token_volatility_results:
    # Create table data for volatility
    table_data = {}
    
    for token, df in token_volatility_results.items():
        # Create a series with volatility values
        volatility_series = pd.Series(
            [f"{(v*100):.1f}%" if not pd.isna(v) else "N/A" 
             for v in df['realized_vol']],
            index=df['time_label']
        )
        table_data[token] = volatility_series
    
    # Create DataFrame with all tokens
    volatility_table = pd.DataFrame(table_data)
    
    # Apply the time blocks in the proper order (most recent first)
    available_times = set(volatility_table.index)
    ordered_times = [t for t in time_block_labels if t in available_times]
    
    # If no matches are found in aligned blocks, fallback to the available times
    if not ordered_times and available_times:
        ordered_times = sorted(list(available_times), reverse=True)
    
    # Reindex with the ordered times
    volatility_table = volatility_table.reindex(ordered_times)
    
    # Function to color cells based on volatility
    def color_volatility_cells(val):
        if val == "N/A":
            return 'background-color: #f5f5f5; color: #666666;'  # Grey for missing
        
        try:
            vol_value = float(val.replace("%", "")) / 100
        except:
            return 'background-color: #f5f5f5; color: #666666;'  # Grey for parsing error
        
        if pd.isna(vol_value):
            return 'background-color: #f5f5f5; color: #666666;'  # Grey for missing
        
        # Define volatility color gradient
        if vol_value < 0.3:  # Low volatility
            intensity = max(0, min(255, int(255 * vol_value / 0.3)))
            return f'background-color: rgba({255-intensity}, 255, {255-intensity}, 0.7); color: black'
        elif vol_value < 0.6:  # Medium volatility
            intensity = max(0, min(255, int(255 * (vol_value - 0.3) / 0.3)))
            return f'background-color: rgba(255, 255, {255-intensity}, 0.7); color: black'
        elif vol_value < 1.0:  # High volatility
            intensity = max(0, min(255, int(255 * (vol_value - 0.6) / 0.4)))
            return f'background-color: rgba(255, {255-(intensity//2)}, 0, 0.7); color: black'
        else:  # Extreme volatility
            return 'background-color: rgba(255, 0, 0, 0.7); color: white'
    
    # Apply styling
    styled_table = volatility_table.style.applymap(color_volatility_cells)
    
    # Display the matrix
    st.markdown(f"## {resolution_min}-Minute Volatility Matrix")
    st.markdown("### Values shown as annualized volatility percentages")
    st.markdown("""
    #### Color Legend: 
    - **Green**: Low volatility (<30%)
    - **Yellow**: Medium volatility (30-60%)
    - **Orange**: High volatility (60-100%)
    - **Red**: Extreme volatility (>100%)
    """)
    st.dataframe(styled_table, height=700, use_container_width=True)
    
    # Create a summary dashboard
    st.subheader("24-Hour Volatility Summary")
    
    # Summary metrics
    summary_data = []
    for token, df in token_volatility_results.items():
        if not df.empty:
            avg_vol = df['realized_vol'].mean() if 'realized_vol' in df and not df['realized_vol'].isna().all() else np.nan
            
            # Get peak volatility time
            if 'realized_vol' in df and not df['realized_vol'].isna().all():
                max_vol_idx = df['realized_vol'].idxmax() 
                max_vol_time = df.loc[max_vol_idx, 'time_label'] if 'time_label' in df.columns else "N/A"
                max_vol_value = df.loc[max_vol_idx, 'realized_vol'] if not pd.isna(df.loc[max_vol_idx, 'realized_vol']) else np.nan
            else:
                max_vol_time = "N/A"
                max_vol_value = np.nan
            
            summary_data.append({
                'Token': token,
                'Avg Vol (%)': (avg_vol * 100).round(1) if not pd.isna(avg_vol) else np.nan,
                'Peak Vol (%)': (max_vol_value * 100).round(1) if not pd.isna(max_vol_value) else np.nan,
                'Peak Vol Time': max_vol_time,
                'Volatility Regime': df['avg_vol_desc'].iloc[0] if not df['avg_vol_desc'].isna().all() else "Unknown"
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        
        # Sort by average volatility (high to low)
        summary_df = summary_df.sort_values(by='Avg Vol (%)', ascending=False)
        
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
        
        def color_regime_column(val):
            if "Low" in val:
                return 'color: green'
            elif "Medium" in val:
                return 'color: #aaaa00'
            elif "High" in val:
                return 'color: orange'
            elif "Extreme" in val:
                return 'color: red'
            else:
                return ''
        
        # Apply styling
        styled_summary = summary_df.style\
            .applymap(color_vol_column, subset=['Avg Vol (%)', 'Peak Vol (%)'])\
            .applymap(color_regime_column, subset=['Volatility Regime'])
        
        # Display the styled dataframe
        st.dataframe(styled_summary, height=500, use_container_width=True)
        
        # Volatility overview
        st.subheader("Volatility Breakdown")
        
        # Count tokens in each volatility regime
        low_vol = sum(1 for row in summary_data if not pd.isna(row['Avg Vol (%)']) and row['Avg Vol (%)'] < 30)
        medium_vol = sum(1 for row in summary_data if not pd.isna(row['Avg Vol (%)']) and 30 <= row['Avg Vol (%)'] < 60)
        high_vol = sum(1 for row in summary_data if not pd.isna(row['Avg Vol (%)']) and 60 <= row['Avg Vol (%)'] < 100)
        extreme_vol = sum(1 for row in summary_data if not pd.isna(row['Avg Vol (%)']) and row['Avg Vol (%)'] >= 100)
        
        total = low_vol + medium_vol + high_vol + extreme_vol
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Low Vol", f"{low_vol} ({low_vol/total*100:.1f}%)")
        col2.metric("Medium Vol", f"{medium_vol} ({medium_vol/total*100:.1f}%)")
        col3.metric("High Vol", f"{high_vol} ({high_vol/total*100:.1f}%)")
        col4.metric("Extreme Vol", f"{extreme_vol} ({extreme_vol/total*100:.1f}%)")
        
        # Create pie chart
        labels = ['Low Vol', 'Medium Vol', 'High Vol', 'Extreme Vol']
        values = [low_vol, medium_vol, high_vol, extreme_vol]
        colors = ['rgba(100,255,100,0.8)', 'rgba(255,255,100,0.8)', 'rgba(255,165,0,0.8)', 'rgba(255,0,0,0.8)']
        
        fig = go.Figure(data=[go.Pie(
            labels=labels, 
            values=values, 
            marker=dict(colors=colors, line=dict(color='#000000', width=2)), 
            textinfo='label+percent', 
            hole=.3
        )])
        fig.update_layout(
            title="24-Hour Volatility Distribution",
            height=400,
            font=dict(color="#000000", size=12),
        )
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning("No summary data available.")
    
    # Extreme volatility events
    st.subheader("Extreme Volatility Events")
    
    extreme_vol_events = []
    for token, df in token_volatility_results.items():
        if not df.empty and 'realized_vol' in df.columns:
            extreme_periods = df[df['realized_vol'] >= extreme_vol_threshold]
            for idx, row in extreme_periods.iterrows():
                vol_value = float(row['realized_vol']) if 'realized_vol' in row and not pd.isna(row['realized_vol']) else 0.0
                time_label = str(row['time_label']) if 'time_label' in row and not pd.isna(row['time_label']) else "Unknown"
                
                extreme_vol_events.append({
                    'Token': token,
                    'Time': time_label,
                    'Volatility (%)': round(vol_value * 100, 1),
                    'Full Timestamp': idx.strftime('%Y-%m-%d %H:%M') if hasattr(idx, 'strftime') else str(idx)
                })
    
    if extreme_vol_events:
        extreme_df = pd.DataFrame(extreme_vol_events)
        # Sort by volatility (highest first)
        extreme_df = extreme_df.sort_values(by='Volatility (%)', ascending=False)
        
        # Reset the index to remove it
        extreme_df = extreme_df.reset_index(drop=True)
        
        # Display the dataframe
        st.dataframe(extreme_df, height=300, use_container_width=True)
        
        # Create a more visually appealing list of extreme events
        st.markdown("### Top Extreme Volatility Events")
        
        # Only process top 10 events if there are any
        top_events = extreme_df.head(10).to_dict('records')
        for i, event in enumerate(top_events):
            token = event['Token']
            time = event['Time']
            vol = event['Volatility (%)']
            date = event['Full Timestamp'].split(' ')[0] if ' ' in event['Full Timestamp'] else ""
            
            st.markdown(f"**{i+1}. {token}** at **{time}** on {date}: <span style='color:red; font-weight:bold;'>{vol}%</span> volatility", unsafe_allow_html=True)
        
        if len(extreme_vol_events) > 10:
            st.markdown(f"*... and {len(extreme_vol_events) - 10} more extreme events*")
    else:
        st.info("No extreme volatility events detected in the selected tokens.")

with st.expander("Understanding the Volatility Matrix"):
    st.markdown(f"""
    ### How to Read This Table
    
    The {resolution_min}-Minute Volatility Matrix shows volatility data for each token across time intervals during the last 24 hours.
    
    #### Cell Values
    Each cell shows the **annualized volatility percentage** for that specific token and time period.
    
    #### Color Coding
    - **Green**: Low volatility (<30%)
    - **Yellow**: Medium volatility (30-60%)
    - **Orange**: High volatility (60-100%)
    - **Red**: Extreme volatility (>100%)
    
    #### Trading Implications
    - **Green areas**: Periods of low volatility, often suitable for range-bound trading strategies
    - **Yellow/Orange areas**: Periods of increased volatility, may require wider stops but offer more trading opportunities 
    - **Red areas**: Extreme volatility periods, requiring careful risk management but potentially offering significant opportunities
    
    #### Technical Details
    - Volatility is calculated as the standard deviation of log returns
    - Values are annualized to represent expected price variation over a year
    - Calculations use a rolling window approach on 1-minute price data
    - All times are shown in Singapore timezone
    - Missing values (gray cells) indicate insufficient data
    """)