import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import pytz

# Page configuration
st.set_page_config(page_title="Edge & Volatility Matrix", layout="wide")
st.title("Edge & Volatility Matrix (10min)")
st.subheader("All Trading Pairs - Last 24 Hours (Singapore Time)")

# DB connection
try:
    db_config = st.secrets["database"]
    db_uri = (
        f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}"
        f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    )
    engine = create_engine(db_uri)
except Exception as e:
    st.error(f"Database connection error: {e}")
    st.stop()

# Time parameters
singapore_tz = pytz.timezone('Asia/Singapore')
now_utc = datetime.now(pytz.utc)
now_sg = now_utc.astimezone(singapore_tz)
start_time_sg = now_sg - timedelta(days=1)

st.write(f"Current Singapore Time: {now_sg.strftime('%Y-%m-%d %H:%M:%S')}")

# Fetch pairs
@st.cache_data(ttl=600)
def fetch_pairs():
    query = "SELECT DISTINCT pair_name FROM public.trade_fill_fresh ORDER BY pair_name"
    try:
        df = pd.read_sql(query, engine)
        if df.empty:
            return []
        return df['pair_name'].tolist()
    except Exception as e:
        st.error(f"Error fetching pairs: {e}")
        return ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT", "PEPE/USDT"]

# Generate time slots
@st.cache_data(ttl=600)
def generate_time_slots(now_time, start_time):
    result_slots = []
    result_labels = []
    result_dates = []
    
    current_time = now_time
    while current_time >= start_time:
        # Round to nearest 10 minute mark
        minute_val = current_time.minute
        rounded_minute_val = (minute_val // 10) * 10
        slot_time = current_time.replace(minute=rounded_minute_val, second=0, microsecond=0)
        
        # Add to our result lists
        result_slots.append(slot_time)
        result_labels.append(slot_time.strftime("%H:%M"))
        result_dates.append(slot_time.strftime("%b %d"))
        
        # Move to previous time slot
        current_time -= timedelta(minutes=10)
    
    return result_slots, result_labels, result_dates

# Call the function and unpack the results
time_slots, time_labels, date_labels = generate_time_slots(now_sg, start_time_sg)

# UI controls
pairs = fetch_pairs()
col1, col2, col3 = st.columns([3, 1, 1])

with col1:
    search_term = st.text_input("Search pairs", "")
    if search_term:
        selected_pairs = [pair for pair in pairs if search_term.lower() in pair.lower()]
        if not selected_pairs:
            st.warning(f"No pairs found matching '{search_term}'")
            selected_pairs = pairs[:5] if len(pairs) >= 5 else pairs
    else:
        selected_pairs = pairs

with col2:
    select_all = st.checkbox("Select All Pairs", value=False)
    if select_all:
        selected_pairs = pairs

with col3:
    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.experimental_rerun()

# Function to fetch data and calculate edge and volatility
@st.cache_data(ttl=600)
def calculate_edge_volatility(pair_name):
    # Convert to UTC for database query
    start_time_utc = start_time_sg.astimezone(pytz.utc)
    end_time_utc = now_sg.astimezone(pytz.utc)
    
    # Query with improved join conditions to handle midnight transition better
    query = f"""
    WITH time_intervals AS (
      SELECT
        generate_series(
          date_trunc('hour', '{start_time_utc}'::timestamp) + 
          INTERVAL '10 min' * floor(EXTRACT(MINUTE FROM '{start_time_utc}'::timestamp) / 10),
          '{end_time_utc}'::timestamp,
          INTERVAL '10 minutes'
        ) AS slot
    ),
    
    sg_time_slots AS (
      SELECT 
        slot,
        (slot AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') AS slot_sg
      FROM time_intervals
    ),
    
    pnl_data AS (
      SELECT
        date_trunc('hour', created_at + INTERVAL '8 hour') + 
        INTERVAL '10 min' * FLOOR(EXTRACT(MINUTE FROM created_at + INTERVAL '8 hour')::INT / 10) AS time_slot_sg,
        
        -- Calculate total PNL based on your colleague's formula
        SUM(CASE WHEN taker_way IN (1, 2, 3, 4) THEN taker_pnl * collateral_price ELSE 0 END) +
        SUM(CASE WHEN taker_fee_mode = 1 AND taker_way IN (1, 3) THEN -1 * taker_fee * collateral_price ELSE 0 END) +
        SUM(CASE WHEN taker_way = 0 THEN funding_fee * collateral_price ELSE 0 END) +
        SUM(-taker_sl_fee * collateral_price - maker_sl_fee) AS total_pnl,
        
        -- Calculate collateral based on your colleague's formula
        SUM(CASE WHEN taker_fee_mode = 2 AND taker_way IN (1, 3) THEN deal_vol * collateral_price ELSE 0 END) AS total_collateral,
        
        -- Get price array for volatility calculation
        array_agg(deal_price ORDER BY created_at) AS price_array
      FROM
        public.trade_fill_fresh
      WHERE
        created_at BETWEEN '{start_time_utc}' AND '{end_time_utc}'
        AND pair_name = '{pair_name}'
      GROUP BY
        time_slot_sg
    )
    
    SELECT
      ts.slot_sg AS timestamp_sg,
      COALESCE(p.total_pnl, 0) AS total_pnl,
      COALESCE(p.total_collateral, 0) AS total_collateral,
      p.price_array
    FROM
      sg_time_slots ts
    LEFT JOIN
      pnl_data p ON ts.slot_sg = p.time_slot_sg
    ORDER BY
      ts.slot DESC  -- Order by time descending (newest first)
    """
    
    try:
        df = pd.read_sql(query, engine)
        
        # Format timestamp and create time_label
        df['timestamp_sg'] = pd.to_datetime(df['timestamp_sg'])
        df['time_label'] = df['timestamp_sg'].dt.strftime('%H:%M')
        
        # Calculate edge
        df['edge'] = np.where(df['total_collateral'] > 0, 
                            df['total_pnl'] / df['total_collateral'], 
                            None)
        
        # Calculate volatility
        def calculate_volatility(prices):
            if prices is None or len(prices) < 2:
                return None
            
            try:
                # Handle PostgreSQL array format
                if isinstance(prices, str):
                    prices = prices.strip('{}').split(',')
                    
                # Convert to float
                numeric_prices = [float(p) for p in prices if p and p != 'NULL' and p != 'None']
                
                if len(numeric_prices) < 2:
                    return None
                
                # Calculate log returns
                prices_array = np.array(numeric_prices)
                log_returns = np.diff(np.log(prices_array))
                
                # Calculate standard deviation and annualize
                volatility = np.std(log_returns) * np.sqrt(6 * 24 * 365)  # 10-min blocks
                
                return volatility
            except Exception as e:
                print(f"Volatility calculation error: {e}")
                return None
        
        df['volatility'] = df.apply(lambda row: calculate_volatility(row['price_array']), axis=1)
        
        return df
    
    except Exception as e:
        st.error(f"Error processing {pair_name}: {e}")
        return None

# NEW FUNCTION: Fetch market spread data for pairs
@st.cache_data(ttl=600)
def fetch_market_spread_data(pair_name):
    # Convert to UTC for database query
    start_time_utc = start_time_sg.astimezone(pytz.utc)
    end_time_utc = now_sg.astimezone(pytz.utc)
    
    query = f"""
    WITH time_intervals AS (
      SELECT
        generate_series(
          date_trunc('hour', '{start_time_utc}'::timestamp) + 
          INTERVAL '10 min' * floor(EXTRACT(MINUTE FROM '{start_time_utc}'::timestamp) / 10),
          '{end_time_utc}'::timestamp,
          INTERVAL '10 minutes'
        ) AS slot
    ),
    
    sg_time_slots AS (
      SELECT 
        slot,
        (slot AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') AS slot_sg
      FROM time_intervals
    ),
    
    spread_data AS (
      SELECT 
        date_trunc('hour', time_group + INTERVAL '8 hour') + 
        INTERVAL '10 min' * FLOOR(EXTRACT(MINUTE FROM time_group + INTERVAL '8 hour')::INT / 10) AS time_slot_sg,
        AVG(fee1) as avg_spread
      FROM 
        oracle_exchange_fee
      WHERE 
        time_group BETWEEN '{start_time_utc}' AND '{end_time_utc}'
        AND pair_name = '{pair_name}'
        AND source IN ('binanceFuture', 'gateFuture', 'hyperliquidFuture')
      GROUP BY 
        time_slot_sg
    )
    
    SELECT
      ts.slot_sg AS timestamp_sg,
      COALESCE(s.avg_spread, NULL) AS avg_spread
    FROM
      sg_time_slots ts
    LEFT JOIN
      spread_data s ON ts.slot_sg = s.time_slot_sg
    ORDER BY
      ts.slot DESC  -- Order by time descending (newest first)
    """
    
    try:
        df = pd.read_sql(query, engine)
        
        # Format timestamp and create time_label
        df['timestamp_sg'] = pd.to_datetime(df['timestamp_sg'])
        df['time_label'] = df['timestamp_sg'].dt.strftime('%H:%M')
        
        return df
    
    except Exception as e:
        st.error(f"Error fetching spreads for {pair_name}: {e}")
        return None

# Function to create spread matrix with pairs as columns and time as rows
def create_transposed_spread_matrix():
    # Create a DataFrame with time slots as rows and pairs as columns
    # Initialize with a list of time labels from our time slots (latest first)
    matrix_data = {
        'time_slot': time_labels,
        'date': date_labels
    }
    
    # Add data for each pair (using the pre-fetched spread_data)
    for pair in selected_pairs:
        if pair in spread_data:
            pair_df = spread_data[pair]
            
            # Create a series with spread values indexed by time_label
            spread_by_time = pd.Series(
                pair_df['avg_spread'].values,
                index=pair_df['time_label']
            )
            
            # Add this pair's data to the matrix
            matrix_data[pair] = [spread_by_time.get(time, None) for time in time_labels]
    
    # Convert to DataFrame
    df = pd.DataFrame(matrix_data)
    
    # Calculate data density for each pair to determine order
    data_counts = {}
    for pair in selected_pairs:
        if pair in df.columns and pair not in ['time_slot', 'date']:
            # Count non-null values for this pair
            data_counts[pair] = df[pair].notna().sum()
    
    # Sort pairs by data density (highest first)
    sorted_pairs = sorted(data_counts.keys(), key=lambda p: data_counts[p], reverse=True)
    
    # Reorder columns: first time_slot and date, then pairs by data density
    reordered_columns = ['time_slot', 'date'] + sorted_pairs
    
    # Return reordered DataFrame where pairs exist
    return df[reordered_columns]

# Function to display spread matrix with custom formatting and date separators
def display_spread_matrix(spread_df):
    # Create a DataFrame with formatted values
    formatted_df = spread_df.copy()
    
    # Process each numeric column
    for col in formatted_df.columns:
        if col not in ['time_slot', 'date']:
            # Format values as strings with basis points representation (1bp = 0.01%)
            formatted_df[col] = formatted_df[col].apply(
                lambda x: f"{x*10000:.2f}" if isinstance(x, (int, float)) and not pd.isna(x) else ""
            )
    
    # Add row numbers for better reference
    formatted_df = formatted_df.reset_index()
    
    # Create date separators
    date_changes = []
    current_date = None
    
    for idx, row in formatted_df.iterrows():
        if row['date'] != current_date:
            date_changes.append(idx)
            current_date = row['date']
    
    # Display using st.dataframe with custom configuration
    st.dataframe(
        formatted_df,
        height=600,
        use_container_width=True,
        column_config={
            "index": st.column_config.NumberColumn("#", width="small"),
            "time_slot": st.column_config.TextColumn("Time", width="small"),
            "date": st.column_config.TextColumn("Date", width="small")
        }
    )
    
    # Display date change indicators
    if date_changes:
        date_info = ", ".join([f"Row #{idx}" for idx in date_changes[1:]])
        if date_info:
            st.info(f"📅 Date changes at: {date_info}")
    
    return formatted_df

# Show pairs selected
st.write(f"Displaying data for {len(selected_pairs)} pairs")

# Progress bar
progress_bar = st.progress(0)
status_text = st.empty()

# Fetch data for all selected pairs
pair_data = {}
spread_data = {}  # Add storage for spread data too
for i, pair in enumerate(selected_pairs):
    progress_bar.progress(i / len(selected_pairs))
    status_text.text(f"Processing {pair} ({i+1}/{len(selected_pairs)})")
    
    # Fetch edge and volatility data
    data = calculate_edge_volatility(pair)
    if data is not None:
        pair_data[pair] = data
    
    # Fetch spread data
    spread = fetch_market_spread_data(pair)
    if spread is not None:
        spread_data[pair] = spread

progress_bar.progress(1.0)
status_text.text(f"Processed {len(pair_data)}/{len(selected_pairs)} pairs")

# Function to create edge matrix with pairs as columns and time as rows
def create_transposed_edge_matrix():
    # Create a DataFrame with time slots as rows and pairs as columns
    # Initialize with a list of time labels from our time slots (latest first)
    matrix_data = {
        'time_slot': time_labels,
        'date': date_labels
    }
    
    # Add data for each pair
    for pair in selected_pairs:
        if pair in pair_data:
            pair_df = pair_data[pair]
            
            # Create a series with edge values indexed by time_label
            edge_by_time = pd.Series(
                pair_df['edge'].values,
                index=pair_df['time_label']
            )
            
            # Add this pair's data to the matrix
            matrix_data[pair] = [edge_by_time.get(time, None) for time in time_labels]
    
    # Convert to DataFrame
    df = pd.DataFrame(matrix_data)
    
    # Calculate data density for each pair to determine order
    data_counts = {}
    for pair in selected_pairs:
        if pair in df.columns and pair not in ['time_slot', 'date']:
            # Count non-null values for this pair
            data_counts[pair] = df[pair].notna().sum()
    
    # Sort pairs by data density (highest first)
    sorted_pairs = sorted(data_counts.keys(), key=lambda p: data_counts[p], reverse=True)
    
    # Reorder columns: first time_slot and date, then pairs by data density
    reordered_columns = ['time_slot', 'date'] + sorted_pairs
    
    # Return reordered DataFrame
    return df[reordered_columns]

# Function to create volatility matrix with pairs as columns
def create_transposed_volatility_matrix():
    # Similar structure as edge matrix
    matrix_data = {
        'time_slot': time_labels,
        'date': date_labels
    }
    
    for pair in selected_pairs:
        if pair in pair_data:
            pair_df = pair_data[pair]
            
            # Create a series with volatility values indexed by time_label
            vol_by_time = pd.Series(
                pair_df['volatility'].values,
                index=pair_df['time_label']
            )
            
            # Add this pair's data to the matrix
            matrix_data[pair] = [vol_by_time.get(time, None) for time in time_labels]
    
    # Convert to DataFrame
    df = pd.DataFrame(matrix_data)
    
    # Calculate data density for each pair to determine order
    data_counts = {}
    for pair in selected_pairs:
        if pair in df.columns and pair not in ['time_slot', 'date']:
            # Count non-null values for this pair
            data_counts[pair] = df[pair].notna().sum()
    
    # Sort pairs by data density (highest first)
    sorted_pairs = sorted(data_counts.keys(), key=lambda p: data_counts[p], reverse=True)
    
    # Reorder columns: first time_slot and date, then pairs by data density
    reordered_columns = ['time_slot', 'date'] + sorted_pairs
    
    # Return reordered DataFrame
    return df[reordered_columns]

# Function to display edge matrix with custom formatting and date separators
def display_edge_matrix(edge_df):
    # Create a DataFrame with formatted values
    formatted_df = edge_df.copy()
    
    # Process each numeric column
    for col in formatted_df.columns:
        if col not in ['time_slot', 'date']:
            # Format values as strings with appropriate coloring
            formatted_df[col] = formatted_df[col].apply(
                lambda x: f"{x*100:.1f}%" if isinstance(x, (int, float)) and not pd.isna(x) else ""
            )
    
    # Add row numbers for better reference
    formatted_df = formatted_df.reset_index()
    
    # Create date separators
    date_changes = []
    current_date = None
    
    for idx, row in formatted_df.iterrows():
        if row['date'] != current_date:
            date_changes.append(idx)
            current_date = row['date']
    
    # Display using st.dataframe with custom configuration
    st.dataframe(
        formatted_df,
        height=600,
        use_container_width=True,
        column_config={
            "index": st.column_config.NumberColumn("#", width="small"),
            "time_slot": st.column_config.TextColumn("Time", width="small"),
            "date": st.column_config.TextColumn("Date", width="small")
        }
    )
    
    # Display date change indicators
    if date_changes:
        date_info = ", ".join([f"Row #{idx}" for idx in date_changes[1:]])
        if date_info:
            st.info(f"📅 Date changes at: {date_info}")
    
    return formatted_df

# Function to display volatility matrix with custom formatting and date separators
def display_volatility_matrix(vol_df):
    # Create a DataFrame with formatted values
    formatted_df = vol_df.copy()
    
    # Process each numeric column
    for col in formatted_df.columns:
        if col not in ['time_slot', 'date']:
            # Format values as strings
            formatted_df[col] = formatted_df[col].apply(
                lambda x: f"{x*100:.1f}%" if isinstance(x, (int, float)) and not pd.isna(x) else ""
            )
    
    # Add row numbers for better reference
    formatted_df = formatted_df.reset_index()
    
    # Create date separators
    date_changes = []
    current_date = None
    
    for idx, row in formatted_df.iterrows():
        if row['date'] != current_date:
            date_changes.append(idx)
            current_date = row['date']
    
    # Display using st.dataframe with column configuration
    st.dataframe(
        formatted_df,
        height=600,
        use_container_width=True,
        column_config={
            "index": st.column_config.NumberColumn("#", width="small"),
            "time_slot": st.column_config.TextColumn("Time", width="small"),
            "date": st.column_config.TextColumn("Date", width="small")
        }
    )
    
    # Display date change indicators
    if date_changes:
        date_info = ", ".join([f"Row #{idx}" for idx in date_changes[1:]])
        if date_info:
            st.info(f"📅 Date changes at: {date_info}")
    
    return formatted_df

# Create tabs
tab1, tab2, tab3 = st.tabs(["Edge", "Volatility", "Spreads"])

if pair_data:
    # Tab 1: Edge Matrix
    with tab1:
        st.markdown("## Edge Matrix (10min timeframe, Last 24 hours, Singapore Time)")
        st.markdown("### Edge = PNL / Total Open Collateral")
        
        # Create edge matrix with pairs as columns
        edge_df = create_transposed_edge_matrix()
        
        if not edge_df.empty:
            # Display the matrix without complex styling
            display_edge_matrix(edge_df)
            
            # Legend
            st.markdown("""
            **Edge Legend:**
            <span style='background-color:rgba(180, 0, 0, 0.9);color:white;padding:2px 6px;border-radius:3px;'>Very Negative (<-10%)</span>
            <span style='background-color:rgba(255, 0, 0, 0.9);color:white;padding:2px 6px;border-radius:3px;'>Negative (-10% to -5%)</span>
            <span style='background-color:rgba(255, 150, 150, 0.9);color:black;padding:2px 6px;border-radius:3px;'>Slightly Negative (-5% to -1%)</span>
            <span style='background-color:rgba(255, 255, 150, 0.9);color:black;padding:2px 6px;border-radius:3px;'>Neutral (-1% to 1%)</span>
            <span style='background-color:rgba(150, 255, 150, 0.9);color:black;padding:2px 6px;border-radius:3px;'>Slightly Positive (1% to 5%)</span>
            <span style='background-color:rgba(0, 255, 0, 0.9);color:black;padding:2px 6px;border-radius:3px;'>Positive (5% to 10%)</span>
            <span style='background-color:rgba(0, 180, 0, 0.9);color:white;padding:2px 6px;border-radius:3px;'>Very Positive (>10%)</span>
            """, unsafe_allow_html=True)
        else:
            st.warning("No edge data available for selected pairs.")
    
    # Tab 2: Volatility Matrix
    with tab2:
        st.markdown("## Volatility Matrix (10min timeframe, Last 24 hours, Singapore Time)")
        st.markdown("### Annualized Volatility = StdDev(Log Returns) * sqrt(trading periods per year)")
        
        # Create volatility matrix with pairs as columns
        vol_df = create_transposed_volatility_matrix()
        
        if not vol_df.empty:
            # Display the table without complex styling
            display_volatility_matrix(vol_df)
            
            # Legend
            st.markdown("""
            **Volatility Legend:**
            <span style='background-color:rgba(0, 180, 0, 0.9);color:white;padding:2px 6px;border-radius:3px;'>Low (<20%)</span>
            <span style='background-color:rgba(150, 255, 150, 0.9);color:black;padding:2px 6px;border-radius:3px;'>Medium-Low (20% to 50%)</span>
            <span style='background-color:rgba(255, 255, 150, 0.9);color:black;padding:2px 6px;border-radius:3px;'>Medium (50% to 100%)</span>
            <span style='background-color:rgba(255, 150, 0, 0.9);color:black;padding:2px 6px;border-radius:3px;'>Medium-High (100% to 150%)</span>
            <span style='background-color:rgba(255, 0, 0, 0.9);color:white;padding:2px 6px;border-radius:3px;'>High (>150%)</span>
            """, unsafe_allow_html=True)
        else:
            st.warning("No volatility data available for selected pairs.")
    
    # Tab 3: Spreads Matrix (NEW)
    with tab3:
        st.markdown("## Market Spreads Matrix (10min timeframe, Last 24 hours, Singapore Time)")
        st.markdown("### Spreads shown in basis points (1bp = 0.01%)")
        
        # Check if we have spread data
        if spread_data:
            # Create market spreads matrix
            spread_df = create_transposed_spread_matrix()
            
            if not spread_df.empty:
                # Display the matrix
                display_spread_matrix(spread_df)
                
                # Legend
                st.markdown("""
                **Spread Legend:**
                <span style='background-color:rgba(0, 180, 0, 0.9);color:white;padding:2px 6px;border-radius:3px;'>Very Low (<2.5)</span>
                <span style='background-color:rgba(150, 255, 150, 0.9);color:black;padding:2px 6px;border-radius:3px;'>Low (2.5 to 5)</span>
                <span style='background-color:rgba(255, 255, 150, 0.9);color:black;padding:2px 6px;border-radius:3px;'>Medium (5 to 10)</span>
                <span style='background-color:rgba(255, 150, 0, 0.9);color:black;padding:2px 6px;border-radius:3px;'>High (10 to 20)</span>
                <span style='background-color:rgba(255, 0, 0, 0.9);color:white;padding:2px 6px;border-radius:3px;'>Very High (>20)</span>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                **Notes:**
                - Market spreads are averaged from binanceFuture, gateFuture, and hyperliquidFuture
                - Spreads are shown in basis points (1bp = 0.01%)
                - Empty cells indicate no data for that time slot
                """)
            else:
                st.warning("No spread data available for selected pairs.")
        else:
            st.warning("No spread data available for selected pairs.")

else:
    st.warning("No data available for selected pairs. Try selecting different pairs or refreshing the data.")