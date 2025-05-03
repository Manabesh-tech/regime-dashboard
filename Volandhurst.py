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
st.title("5-Minute Volatility Table")
st.subheader("All Trading Pairs - Last 12 Hours (Singapore Time)")

# Define parameters for the 5-minute timeframe
timeframe = "5min"
lookback_hours = 12  # Fixed at 12 hours
expected_points = 144  # Expected data points per pair over 12 hours (12 hours * 12 5-min periods per hour)
singapore_timezone = pytz.timezone('Asia/Singapore')

# Get current time in Singapore timezone
now_utc = datetime.now(pytz.utc)
now_sg = now_utc.astimezone(singapore_timezone)
st.write(f"Current Singapore Time: {now_sg.strftime('%Y-%m-%d %H:%M:%S')}")

# Set extreme volatility threshold
extreme_vol_threshold = 1.0  # 100% annualized volatility

# Function to get partition tables based on date range
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
        # Query for Surf data (source_type = 0) with direct interval addition
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

# UI Controls
col1, col2 = st.columns([3, 1])

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
    if st.button("Refresh Data", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.experimental_rerun()

if not selected_tokens:
    st.warning("Please select at least one token")
    st.stop()

# Fetch and calculate volatility for a token with 5min timeframe
@st.cache_data(ttl=300, show_spinner="Calculating volatility metrics...")
def fetch_and_calculate_volatility(token):
    # Get current time in Singapore timezone
    now_utc = datetime.now(pytz.utc)
    now_sg = now_utc.astimezone(singapore_timezone)
    
    # Request 16 hours to ensure we get at least 12 hours (33% extra)
    extended_lookback = 16
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
        
        # Create 5-minute windows for consistent methodology with the plot script
        result = []
        
        # Create a rounded start and end time for the windows
        start_date = price_data.index.min().floor('5min')
        end_date = price_data.index.max().ceil('5min')
        five_min_periods = pd.date_range(start=start_date, end=end_date, freq='5min')
        
        for i in range(len(five_min_periods)-1):
            start_window = five_min_periods[i]
            end_window = five_min_periods[i+1]
            
            # Get price data in this window
            window_data = price_data[(price_data.index >= start_window) & (price_data.index < end_window)]
            
            if len(window_data) >= 2:  # Need at least 2 points for volatility
                # OHLC data
                window_open = window_data.iloc[0]
                window_high = window_data.max()
                window_low = window_data.min()
                window_close = window_data.iloc[-1]
                
                # Calculate volatility using 1-second data points
                # Log returns
                log_returns = np.diff(np.log(window_data.values))
                
                # Annualize: seconds in year / seconds in 5 minutes
                annualization_factor = np.sqrt(31536000 / 300)
                volatility = np.std(log_returns) * annualization_factor
                
                result.append({
                    'timestamp': start_window,
                    'time_label': start_window.strftime('%H:%M'),
                    'open': window_open,
                    'high': window_high, 
                    'low': window_low,
                    'close': window_close,
                    'realized_vol': volatility
                })
        
        # Create dataframe from results
        if not result:
            print(f"[{token}] Could not calculate volatility.")
            return None
            
        result_df = pd.DataFrame(result)
        
        # Get most recent 12 hours (144 5-minute periods)
        periods_needed = lookback_hours * 12
        result_df = result_df.tail(periods_needed)
        
        # Ensure we have a complete set of time blocks for the last 12 hours
        end_time = now_sg.replace(second=0, microsecond=0)
        minute = end_time.minute
        end_time = end_time.replace(minute=(minute // 5) * 5)  # Round to nearest 5 minutes
        
        start_time = end_time - timedelta(hours=lookback_hours)
        
        # Create the full range of expected timestamps
        full_time_range = pd.date_range(start=start_time, end=end_time, freq='5min')
        full_time_labels = [ts.strftime('%H:%M') for ts in full_time_range]
        
        # Match against what we have
        time_label_mapping = {row['time_label']: row for idx, row in result_df.iterrows()}
        
        # Create a complete dataframe
        complete_result = []
        for ts, label in zip(full_time_range, full_time_labels):
            if label in time_label_mapping:
                row_data = time_label_mapping[label]
                row_data['timestamp'] = ts  # Ensure we use the consistent timestamp
                complete_result.append(row_data)
            else:
                # Create an empty row
                complete_result.append({
                    'timestamp': ts,
                    'time_label': label,
                    'open': np.nan,
                    'high': np.nan,
                    'low': np.nan,
                    'close': np.nan,
                    'realized_vol': np.nan
                })
        
        complete_df = pd.DataFrame(complete_result)
        
        # Calculate average volatility over the period
        valid_vols = complete_df['realized_vol'].dropna()
        if not valid_vols.empty:
            avg_vol = valid_vols.mean()
            complete_df['avg_period_vol'] = avg_vol
            
            # Classify volatility
            complete_df['vol_info'] = complete_df['realized_vol'].apply(
                lambda x: ("UNKNOWN", 0, "Insufficient data") if pd.isna(x) else
                          ("LOW", 1, "Low volatility") if x < 0.3 else
                          ("MEDIUM", 2, "Medium volatility") if x < 0.6 else
                          ("HIGH", 3, "High volatility") if x < 1.0 else
                          ("EXTREME", 4, "Extreme volatility")
            )
            
            complete_df['vol_regime'] = complete_df['vol_info'].apply(lambda x: x[0])
            complete_df['vol_desc'] = complete_df['vol_info'].apply(lambda x: x[2])
            
            # Also classify the average
            avg_vol_info = ("UNKNOWN", 0, "Insufficient data") if pd.isna(avg_vol) else \
                           ("LOW", 1, "Low volatility") if avg_vol < 0.3 else \
                           ("MEDIUM", 2, "Medium volatility") if avg_vol < 0.6 else \
                           ("HIGH", 3, "High volatility") if avg_vol < 1.0 else \
                           ("EXTREME", 4, "Extreme volatility")
                           
            complete_df['avg_vol_regime'] = avg_vol_info[0]
            complete_df['avg_vol_desc'] = avg_vol_info[2]
            
            # Flag extreme volatility
            complete_df['is_extreme'] = complete_df['realized_vol'] >= extreme_vol_threshold
        
        print(f"[{token}] Successful Volatility Calculation")
        return complete_df
    except Exception as e:
        st.error(f"Error processing {token}: {e}")
        print(f"[{token}] Error processing: {e}")
        return None

# Generate time blocks for display
def generate_time_blocks(current_time, hours=12):
    """
    Generate full 12 hours of 5-minute blocks
    """
    blocks = []
    # Round down to the nearest 5 minutes
    minute = current_time.minute
    rounded_minute = (minute // 5) * 5
    end_time = current_time.replace(minute=rounded_minute, second=0, microsecond=0)
    
    # Generate 144 blocks (12 hours * 12 5-min periods per hour)
    for i in range(hours * 12):
        block_end = end_time - timedelta(minutes=i*5)
        block_start = block_end - timedelta(minutes=5)
        label = block_start.strftime('%H:%M')
        blocks.append((block_start, block_end, label))
    
    return blocks

# Generate all the 5-minute blocks we want to display
time_blocks = generate_time_blocks(now_sg)
block_labels = [block[2] for block in time_blocks]

# Show the blocks we're analyzing
with st.expander("View Time Blocks Being Analyzed", expanded=False):
    time_blocks_df = pd.DataFrame([(b[0].strftime('%Y-%m-%d %H:%M'), b[1].strftime('%Y-%m-%d %H:%M'), b[2]) 
                                  for b in time_blocks], 
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
    
    # Process tokens in current batch
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

# Create table for display
if token_results:
    # Create table data
    table_data = {}
    for token, df in token_results.items():
        vol_series = df.set_index('time_label')['realized_vol']
        table_data[token] = vol_series
    
    # Create DataFrame with all tokens
    vol_table = pd.DataFrame(table_data)
    
    # Apply the standard time blocks
    vol_table = vol_table.reindex(block_labels)
    
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
    
    # Style the table
    styled_table = vol_table.style.applymap(color_cells)
    st.markdown(f"## Volatility Table (5min timeframe, Last 12 hours, Singapore Time)")
    st.markdown("### Color Legend: <span style='color:green'>Low Vol</span>, <span style='color:#aaaa00'>Medium Vol</span>, <span style='color:orange'>High Vol</span>, <span style='color:red'>Extreme Vol</span>", unsafe_allow_html=True)
    st.markdown("Values shown as annualized volatility percentage")
    
    # Calculate how many time blocks actually contain data
    data_blocks = vol_table.count(axis=1).max() if not vol_table.empty else 0
    st.info(f"Displaying {len(vol_table)} time blocks covering 12 hours. Found data for approximately {data_blocks/12:.1f} hours.")
    
    # Set a maximum height for the table
    max_height = min(700, 100 + 20 * len(vol_table))  # Base height + rows
    st.dataframe(styled_table, height=max_height, use_container_width=True)
    
    # Create ranking table based on average volatility
    st.subheader(f"Volatility Ranking (12-Hour Average, Descending Order)")
    
    ranking_data = []
    for token, df in token_results.items():
        if not df.empty and 'avg_period_vol' in df.columns:
            # We should have at least one value that's not NA
            avg_vol = df['avg_period_vol'].iloc[0] if not pd.isna(df['avg_period_vol'].iloc[0]) else np.nan
            
            if not pd.isna(avg_vol):
                vol_regime = df['avg_vol_desc'].iloc[0] if 'avg_vol_desc' in df.columns else ""
                
                # Get max and min from non-NA values
                valid_vols = df['realized_vol'].dropna()
                if not valid_vols.empty:
                    max_vol = valid_vols.max()
                    min_vol = valid_vols.min()
                    
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
                    'Full Timestamp': row['timestamp'].strftime('%Y-%m-%d %H:%M') if 'timestamp' in row else "Unknown"
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
    st.subheader(f"12-Hour Average Volatility Overview (Singapore Time)")
    
    # OPTIMIZATION: Place metrics in a more compact layout
    col1, col2 = st.columns(2)
    
    with col1:
        avg_values = {}
        for token, df in token_results.items():
            if not df.empty and 'avg_period_vol' in df.columns:
                # Check for first row with avg_period_vol that's not NA
                if not pd.isna(df['avg_period_vol'].iloc[0]):
                    avg = df['avg_period_vol'].iloc[0]
                    regime = df['avg_vol_desc'].iloc[0] if 'avg_vol_desc' in df.columns else ""
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
                title=f"12-Hour Average Volatility Distribution",
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
    - Volatility is calculated using log returns of price data within each 5-minute period
    - Values shown are in percentage (e.g., 50.0 means 50% annualized volatility)
    - The volatility is annualized using sqrt(seconds in year / seconds in 5 minutes)
    - Missing values (light gray cells) indicate insufficient data for calculation
    """)