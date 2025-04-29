import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import pytz

st.set_page_config(
    page_title="Edge & Volatility Matrix",
    page_icon="📊",
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
st.title("Edge & Volatility Matrix (10min)")
st.subheader("All Trading Pairs - Last 24 Hours (Singapore Time)")

# Define parameters
timeframe = "10min"
lookback_days = 1  # 24 hours
singapore_timezone = pytz.timezone('Asia/Singapore')

# Get current time in Singapore timezone
now_utc = datetime.now(pytz.utc)
now_sg = now_utc.astimezone(singapore_timezone)
st.write(f"Current Singapore Time: {now_sg.strftime('%Y-%m-%d %H:%M:%S')}")

# Fetch all available pairs from DB
@st.cache_data(ttl=600, show_spinner="Fetching pairs...")
def fetch_all_pairs():
    query = "SELECT DISTINCT pair_name FROM public.trade_pool_pairs ORDER BY pair_name"
    try:
        df = pd.read_sql(query, engine)
        if df.empty:
            st.error("No pairs found in the database.")
            return []
        return df['pair_name'].tolist()
    except Exception as e:
        st.error(f"Error fetching pairs: {e}")
        return ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT", "PEPE/USDT"]  # Default fallback

all_pairs = fetch_all_pairs()

# UI Controls
col1, col2 = st.columns([3, 1])

with col1:
    # Let user select pairs to display (or select all)
    select_all = st.checkbox("Select All Pairs", value=False)
    
    if select_all:
        selected_pairs = all_pairs
    else:
        selected_pairs = st.multiselect(
            "Select Pairs", 
            all_pairs,
            default=all_pairs[:5] if len(all_pairs) > 5 else all_pairs
        )

with col2:
    # Add a refresh button
    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.experimental_rerun()

if not selected_pairs:
    st.warning("Please select at least one pair")
    st.stop()

# Function to generate aligned 10-minute time blocks for the past 24 hours
def generate_aligned_time_blocks(current_time):
    """
    Generate fixed 10-minute time blocks for past 24 hours,
    aligned with standard 10-minute intervals (e.g., 4:00-4:10, 4:10-4:20)
    """
    # Round down to the nearest 10-minute mark
    minute = current_time.minute
    rounded_minute = (minute // 10) * 10
    latest_complete_block_end = current_time.replace(minute=rounded_minute, second=0, microsecond=0)
    
    # Generate block labels for display
    blocks = []
    for i in range(144):  # 24 hours of 10-minute blocks
        block_end = latest_complete_block_end - timedelta(minutes=i*10)
        block_start = block_end - timedelta(minutes=10)
        block_label = f"{block_start.strftime('%H:%M')}"
        blocks.append((block_start, block_end, block_label))
    
    return blocks

# Generate aligned time blocks
aligned_time_blocks = generate_aligned_time_blocks(now_sg)
time_block_labels = [block[2] for block in aligned_time_blocks]

# Calculate Edge (PNL / Collateral) for each 10-minute time block
@st.cache_data(ttl=600, show_spinner="Calculating edge...")
def fetch_edge_data(pair_name):
    # Get current time in Singapore timezone
    now_utc = datetime.now(pytz.utc)
    now_sg = now_utc.astimezone(singapore_timezone)
    start_time_sg = now_sg - timedelta(days=lookback_days)
    
    # Convert back to UTC for database query
    start_time_utc = start_time_sg.astimezone(pytz.utc)
    end_time_utc = now_sg.astimezone(pytz.utc)

    # This query calculates edge (PNL / Collateral) for each 10-minute interval
    query = f"""
    WITH time_intervals AS (
      -- Generate 10-minute intervals for the past 24 hours
      SELECT
        generate_series(
          date_trunc('hour', '{start_time_utc}'::timestamp) + 
          INTERVAL '10 min' * floor(EXTRACT(MINUTE FROM '{start_time_utc}'::timestamp) / 10),
          '{end_time_utc}'::timestamp,
          INTERVAL '10 minutes'
        ) AS time_slot
    ),
    
    order_pnl AS (
      -- Calculate platform order PNL
      SELECT
        date_trunc('hour', created_at + INTERVAL '8 hour') + 
        INTERVAL '10 min' * FLOOR(EXTRACT(MINUTE FROM created_at + INTERVAL '8 hour')::INT / 10) AS timestamp,
        COALESCE(SUM(taker_pnl * collateral_price), 0) AS platform_order_pnl
      FROM
        public.trade_fill_fresh
      WHERE
        created_at BETWEEN '{start_time_utc}' AND '{end_time_utc}'
        AND pair_name = '{pair_name}'
        AND taker_way IN (1, 2, 3, 4)
      GROUP BY
        timestamp
    ),
    
    fee_data AS (
      -- Calculate fee revenue
      SELECT
        date_trunc('hour', created_at + INTERVAL '8 hour') + 
        INTERVAL '10 min' * FLOOR(EXTRACT(MINUTE FROM created_at + INTERVAL '8 hour')::INT / 10) AS timestamp,
        COALESCE(SUM(-1 * taker_fee * collateral_price), 0) AS fee_revenue
      FROM
        public.trade_fill_fresh
      WHERE
        created_at BETWEEN '{start_time_utc}' AND '{end_time_utc}'
        AND pair_name = '{pair_name}'
        AND taker_fee_mode = 1
        AND taker_way IN (1, 3)
      GROUP BY
        timestamp
    ),
    
    funding_pnl AS (
      -- Calculate funding fee PNL
      SELECT
        date_trunc('hour', created_at + INTERVAL '8 hour') + 
        INTERVAL '10 min' * FLOOR(EXTRACT(MINUTE FROM created_at + INTERVAL '8 hour')::INT / 10) AS timestamp,
        COALESCE(SUM(funding_fee * collateral_price), 0) AS funding_fee_pnl
      FROM
        public.trade_fill_fresh
      WHERE
        created_at BETWEEN '{start_time_utc}' AND '{end_time_utc}'
        AND pair_name = '{pair_name}'
        AND taker_way = 0
      GROUP BY
        timestamp
    ),
    
    sl_fees AS (
      -- Calculate stop loss fees
      SELECT
        date_trunc('hour', created_at + INTERVAL '8 hour') + 
        INTERVAL '10 min' * FLOOR(EXTRACT(MINUTE FROM created_at + INTERVAL '8 hour')::INT / 10) AS timestamp,
        COALESCE(SUM(-taker_sl_fee * collateral_price - maker_sl_fee), 0) AS sl_fee_pnl
      FROM
        public.trade_fill_fresh
      WHERE
        created_at BETWEEN '{start_time_utc}' AND '{end_time_utc}'
        AND pair_name = '{pair_name}'
      GROUP BY
        timestamp
    ),
    
    collateral_data AS (
      -- Calculate open collateral
      SELECT
        date_trunc('hour', created_at + INTERVAL '8 hour') + 
        INTERVAL '10 min' * FLOOR(EXTRACT(MINUTE FROM created_at + INTERVAL '8 hour')::INT / 10) AS timestamp,
        COALESCE(SUM(CASE WHEN taker_fee_mode = 2 AND taker_way IN (1, 3) 
                   THEN deal_vol * collateral_price ELSE 0 END), 0) AS total_collateral,
        array_agg(deal_price ORDER BY created_at) AS price_array
      FROM
        public.trade_fill_fresh
      WHERE
        created_at BETWEEN '{start_time_utc}' AND '{end_time_utc}'
        AND pair_name = '{pair_name}'
      GROUP BY
        timestamp
    )
    
    -- Combine all data sources
    SELECT
      ti.time_slot AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore' AS timestamp_sg,
      COALESCE(o.platform_order_pnl, 0) + 
      COALESCE(f.fee_revenue, 0) + 
      COALESCE(ff.funding_fee_pnl, 0) + 
      COALESCE(sl.sl_fee_pnl, 0) AS total_pnl,
      COALESCE(c.total_collateral, 0) AS total_collateral,
      c.price_array
    FROM
      time_intervals ti
    LEFT JOIN
      order_pnl o ON ti.time_slot = o.timestamp AT TIME ZONE 'Asia/Singapore' AT TIME ZONE 'UTC'
    LEFT JOIN
      fee_data f ON ti.time_slot = f.timestamp AT TIME ZONE 'Asia/Singapore' AT TIME ZONE 'UTC'
    LEFT JOIN
      funding_pnl ff ON ti.time_slot = ff.timestamp AT TIME ZONE 'Asia/Singapore' AT TIME ZONE 'UTC'
    LEFT JOIN
      sl_fees sl ON ti.time_slot = sl.timestamp AT TIME ZONE 'Asia/Singapore' AT TIME ZONE 'UTC'
    LEFT JOIN
      collateral_data c ON ti.time_slot = c.timestamp AT TIME ZONE 'Asia/Singapore' AT TIME ZONE 'UTC'
    ORDER BY
      ti.time_slot DESC
    """
    
    try:
        df = pd.read_sql(query, engine)
        
        if df.empty:
            return None
        
        # Process data
        df['timestamp_sg'] = pd.to_datetime(df['timestamp_sg'])
        df['time_label'] = df['timestamp_sg'].dt.strftime('%H:%M')
        
        # Calculate Edge
        df['edge'] = np.where(df['total_collateral'] > 0, 
                            df['total_pnl'] / df['total_collateral'], 
                            None)
        
        # Calculate volatility
        def calculate_volatility(prices):
            if prices is None or len(prices) < 2:
                return None
            
            try:
                # Handle PostgreSQL array formats
                if isinstance(prices, str):
                    prices = prices.strip('{}').split(',')
                    prices = [float(p) for p in prices if p and p != 'NULL' and p != 'None']
                else:
                    prices = [float(p) for p in prices if p]
                
                if len(prices) < 2:
                    return None
                    
                # Calculate log returns and volatility
                prices_array = np.array(prices)
                log_returns = np.diff(np.log(prices_array))
                
                # Calculate standard deviation and annualize (10-min intervals -> 6 * 24 * 365)
                std_dev = np.std(log_returns)
                volatility = std_dev * np.sqrt(6 * 24 * 365)
                
                return volatility
            except Exception as e:
                return None
        
        # Apply volatility calculation
        df['volatility'] = df['price_array'].apply(calculate_volatility)
        
        return df
    except Exception as e:
        st.error(f"Error calculating edge for {pair_name}: {e}")
        print(f"[{pair_name}] Error calculating edge: {e}")
        return None

# Show progress bar while calculating
progress_bar = st.progress(0)
status_text = st.empty()

# Calculate edge for each pair
pair_results = {}
for i, pair_name in enumerate(selected_pairs):
    try:
        progress_bar.progress((i) / len(selected_pairs))
        status_text.text(f"Processing {pair_name} ({i+1}/{len(selected_pairs)})")
        
        # Fetch edge data
        edge_data = fetch_edge_data(pair_name)
        
        if edge_data is not None:
            pair_results[pair_name] = edge_data
    except Exception as e:
        st.error(f"Error processing pair {pair_name}: {e}")
        print(f"Error processing pair {pair_name} in main loop: {e}")

# Final progress update
progress_bar.progress(1.0)
status_text.text(f"Processed {len(pair_results)}/{len(selected_pairs)} pairs successfully")

# Create tabs for Edge and Volatility
tab1, tab2, tab3 = st.tabs(["Edge", "Volatility", "Combined"])

# Helper function to format time labels for display
def format_time_labels():
    # Process time blocks to get unique, ordered labels
    all_time_labels = set()
    for pair, df in pair_results.items():
        all_time_labels.update(df['time_label'].tolist())
    
    # Find which time labels match our standard blocks
    ordered_times = []
    for t in time_block_labels:
        if t in all_time_labels:
            ordered_times.append(t)
    
    # If no matches found, use the existing times
    if not ordered_times:
        ordered_times = sorted(list(all_time_labels), reverse=True)
    
    return ordered_times

# Create an edge matrix that shows exact values
def create_edge_matrix(data_dict, time_labels):
    # Create a new DataFrame with time labels as columns and pairs as rows
    edge_matrix = pd.DataFrame(index=selected_pairs, columns=time_labels)
    
    # Fill in the matrix with edge values
    for pair_name, df in data_dict.items():
        for _, row in df.iterrows():
            time_label = row['time_label']
            if time_label in time_labels:
                edge_matrix.at[pair_name, time_label] = row['edge']
    
    # Return the matrix with both the index and columns reset
    return edge_matrix.reset_index().rename(columns={'index': 'pair_name'})

# Create a volatility matrix that shows exact values
def create_volatility_matrix(data_dict, time_labels):
    # Create a new DataFrame with time labels as columns and pairs as rows
    vol_matrix = pd.DataFrame(index=selected_pairs, columns=time_labels)
    
    # Fill in the matrix with volatility values
    for pair_name, df in data_dict.items():
        for _, row in df.iterrows():
            time_label = row['time_label']
            if time_label in time_labels:
                vol_matrix.at[pair_name, time_label] = row['volatility']
    
    # Return the matrix with both the index and columns reset
    return vol_matrix.reset_index().rename(columns={'index': 'pair_name'})

# Style functions
def color_edge_cells(val):
    if pd.isna(val) or val == 0:
        return 'background-color: #f5f5f5; color: #666666;'  # Gray for missing/zero
    elif val < -0.1:
        return 'background-color: rgba(180, 0, 0, 0.9); color: white;'  # Deep red
    elif val < -0.05:
        return 'background-color: rgba(255, 0, 0, 0.9); color: white;'  # Red
    elif val < -0.01:
        return 'background-color: rgba(255, 150, 150, 0.9); color: black;'  # Light red
    elif val < 0.01:
        return 'background-color: rgba(255, 255, 150, 0.9); color: black;'  # Yellow
    elif val < 0.05:
        return 'background-color: rgba(150, 255, 150, 0.9); color: black;'  # Light green
    elif val < 0.1:
        return 'background-color: rgba(0, 255, 0, 0.9); color: black;'  # Green
    else:
        return 'background-color: rgba(0, 180, 0, 0.9); color: white;'  # Deep green

def color_volatility_cells(val):
    if pd.isna(val) or val == 0:
        return 'background-color: #f5f5f5; color: #666666;'  # Gray for missing/zero
    elif val < 0.3:
        return 'background-color: rgba(0, 255, 0, 0.9); color: black;'  # Green
    elif val < 0.6:
        return 'background-color: rgba(255, 255, 0, 0.9); color: black;'  # Yellow
    elif val < 1.0:
        return 'background-color: rgba(255, 165, 0, 0.9); color: black;'  # Orange
    else:
        return 'background-color: rgba(255, 0, 0, 0.9); color: white;'  # Red

if pair_results:
    # Get ordered time labels
    ordered_times = format_time_labels()
    
    # With tab 1 (Edge)
    with tab1:
        st.markdown("## Edge Matrix (10min timeframe, Last 24 hours, Singapore Time)")
        st.markdown("### Edge = PNL / Total Open Collateral")
        
        # Create edge matrix
        edge_df = create_edge_matrix(pair_results, ordered_times)
        
        # Format values for display (convert to percentages)
        display_edge_df = edge_df.copy()
        for col in ordered_times:
            display_edge_df[col] = display_edge_df[col].apply(
                lambda x: f"{x*100:.1f}%" if not pd.isna(x) else "0"
            )
        
        # Style the dataframe
        edge_styled = edge_df.style.apply(
            lambda x: [color_edge_cells(val) for val in x], 
            axis=1, 
            subset=ordered_times
        )
        
        # Replace NaN with formatted values for display
        edge_styled.data = display_edge_df
        
        # Display the styled dataframe
        st.dataframe(edge_styled, height=600, use_container_width=True)
        
        # Legend
        st.markdown("**Edge Legend:** <span style='color:red'>Negative</span> | <span style='color:yellow'>Neutral</span> | <span style='color:green'>Positive</span>", unsafe_allow_html=True)
    
    # With tab 2 (Volatility)
    with tab2:
        st.markdown("## Volatility Matrix (10min timeframe, Last 24 hours, Singapore Time)")
        st.markdown("### Annualized Volatility = StdDev(Log Returns) * sqrt(trading periods per year)")
        
        # Create volatility matrix
        vol_df = create_volatility_matrix(pair_results, ordered_times)
        
        # Format values for display (convert to percentages)
        display_vol_df = vol_df.copy()
        for col in ordered_times:
            display_vol_df[col] = display_vol_df[col].apply(
                lambda x: f"{x*100:.1f}%" if not pd.isna(x) else "0"
            )
        
        # Style the dataframe
        vol_styled = vol_df.style.apply(
            lambda x: [color_volatility_cells(val) for val in x], 
            axis=1, 
            subset=ordered_times
        )
        
        # Replace NaN with formatted values for display
        vol_styled.data = display_vol_df
        
        # Display the styled dataframe
        st.dataframe(vol_styled, height=600, use_container_width=True)
        
        # Legend
        st.markdown("**Volatility Legend:** <span style='color:green'>Low</span> | <span style='color:yellow'>Medium</span> | <span style='color:orange'>High</span> | <span style='color:red'>Extreme</span>", unsafe_allow_html=True)
    
    # With tab 3 (Combined View)
    with tab3:
        st.markdown("## Combined View (Edge and Volatility)")
        st.markdown("### Edge values are shown as percentages, Volatility values are shown as percentages")
        
        # Create tabs for different combined views
        combined_tab1, combined_tab2 = st.tabs(["Edge Values", "Volatility Values"])
        
        with combined_tab1:
            st.markdown("### Edge Values (Percentages)")
            # We'll use the already formatted edge dataframe
            st.dataframe(edge_styled, height=600, use_container_width=True)
        
        with combined_tab2:
            st.markdown("### Volatility Values (Percentages)")
            # We'll use the already formatted volatility dataframe
            st.dataframe(vol_styled, height=600, use_container_width=True)

else:
    st.warning("No data available for the selected pairs. Try selecting different pairs or refreshing the data.")