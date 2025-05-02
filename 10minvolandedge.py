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
    query = """
    SELECT DISTINCT pair_name 
    FROM public.trade_fill_fresh 
    WHERE created_at > NOW() - INTERVAL '1 day'
    ORDER BY pair_name
    """
    
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
        # Default to just showing 5 pairs to avoid performance issues
        selected_pairs = pairs[:5] if len(pairs) >= 5 else pairs

with col2:
    select_all = st.checkbox("Select All Pairs", value=False)
    if select_all:
        # Add a max limit to avoid overloading
        max_pairs = 10
        if len(pairs) > max_pairs:
            st.warning(f"Limited to top {max_pairs} pairs for performance")
            selected_pairs = pairs[:max_pairs]
        else:
            selected_pairs = pairs

with col3:
    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.experimental_rerun()

# Function to fetch edge data only
@st.cache_data(ttl=600)
def fetch_edge_data(pair_name, start_time_sg, now_sg):
    # Convert to UTC for database query
    start_time_utc = start_time_sg.astimezone(pytz.utc)
    end_time_utc = now_sg.astimezone(pytz.utc)
    
    # Query for house edge
    edge_query = f"""
    WITH time_intervals AS (
      SELECT 
        generate_series(
          date_trunc('hour', '{start_time_utc}'::timestamp) + 
          INTERVAL '10 min' * floor(EXTRACT(MINUTE FROM '{start_time_utc}'::timestamp) / 10),
          '{end_time_utc}'::timestamp,
          INTERVAL '10 minutes'
        ) AS interval_start
    ),
    pnl_data AS (
      SELECT
        date_trunc('hour', created_at + INTERVAL '8 hour') + 
          INTERVAL '10 min' * floor(extract(minute from created_at + INTERVAL '8 hour') / 10) AS interval_start,
        SUM(-1 * taker_pnl * collateral_price) AS trading_pnl,
        SUM(CASE WHEN taker_fee_mode = 1 AND taker_way IN (1, 3) THEN taker_fee * collateral_price ELSE 0 END) AS taker_fee,
        SUM(CASE WHEN taker_way = 0 THEN -1 * funding_fee * collateral_price ELSE 0 END) AS funding_pnl,
        SUM(taker_sl_fee * collateral_price + maker_sl_fee) AS sl_fee
      FROM public.trade_fill_fresh
      WHERE created_at BETWEEN '{start_time_utc}' AND '{end_time_utc}'
        AND pair_name = '{pair_name}'
      GROUP BY 1
    ),
    collateral_data AS (
      SELECT
        date_trunc('hour', created_at + INTERVAL '8 hour') + 
          INTERVAL '10 min' * floor(extract(minute from created_at + INTERVAL '8 hour') / 10) AS interval_start,
        SUM(deal_vol * collateral_price) AS open_collateral
      FROM public.trade_fill_fresh
      WHERE taker_fee_mode = 2 AND taker_way IN (1, 3)
        AND created_at BETWEEN '{start_time_utc}' AND '{end_time_utc}'
        AND pair_name = '{pair_name}'
      GROUP BY 1
    ),
    rebate_data AS (
      SELECT
        date_trunc('hour', created_at + INTERVAL '8 hour') + 
          INTERVAL '10 min' * floor(extract(minute from created_at + INTERVAL '8 hour') / 10) AS interval_start,
        SUM(amount * coin_price) AS rebate_amount
      FROM public.user_cashbooks
      WHERE remark = '给邀请人返佣'
        AND created_at BETWEEN '{start_time_utc}' AND '{end_time_utc}'
      GROUP BY 1
    )
    SELECT
      ti.interval_start AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore' AS timestamp_sg,
      CASE
        WHEN COALESCE(cd.open_collateral, 0) = 0 THEN 0
        ELSE (COALESCE(pd.trading_pnl, 0) + 
              COALESCE(pd.taker_fee, 0) + 
              COALESCE(pd.funding_pnl, 0) + 
              COALESCE(pd.sl_fee, 0) - 
              COALESCE(rd.rebate_amount, 0)) / 
             cd.open_collateral
      END AS house_edge,
      COALESCE(pd.trading_pnl, 0) + 
        COALESCE(pd.taker_fee, 0) + 
        COALESCE(pd.funding_pnl, 0) + 
        COALESCE(pd.sl_fee, 0) - 
        COALESCE(rd.rebate_amount, 0) AS pnl,
      COALESCE(cd.open_collateral, 0) AS open_collateral
    FROM time_intervals ti
    LEFT JOIN pnl_data pd ON ti.interval_start = pd.interval_start
    LEFT JOIN collateral_data cd ON ti.interval_start = cd.interval_start
    LEFT JOIN rebate_data rd ON ti.interval_start = rd.interval_start
    ORDER BY timestamp_sg DESC
    """
    
    try:
        # Get edge data
        edge_df = pd.read_sql(edge_query, engine)
        
        # Format timestamps and create time_label for edge data
        edge_df['timestamp_sg'] = pd.to_datetime(edge_df['timestamp_sg'])
        edge_df['time_label'] = edge_df['timestamp_sg'].dt.strftime('%H:%M')
        edge_df['pair_name'] = pair_name
        
        # Rename house_edge to edge for consistency
        edge_df.rename(columns={'house_edge': 'edge'}, inplace=True)
        
        # Add volatility column (will be filled later if volatility data is available)
        edge_df['volatility'] = None
        
        return edge_df
    except Exception as e:
        st.error(f"Error processing {pair_name} edge data: {e}")
        # Return empty dataframe with required columns
        return pd.DataFrame(columns=[
            'timestamp_sg', 'edge', 'pnl', 'open_collateral', 
            'time_label', 'pair_name', 'volatility'
        ])

# Separate function to fetch price stats for volatility calculation
@st.cache_data(ttl=600)
def fetch_price_stats(pair_name, start_time_sg, now_sg):
    # Convert to UTC for database query
    start_time_utc = start_time_sg.astimezone(pytz.utc)
    end_time_utc = now_sg.astimezone(pytz.utc)
    
    # Super simple query to get min/max prices in 10-minute windows
    query = f"""
    WITH time_windows AS (
        SELECT 
            time_window,
            time_window + INTERVAL '10 minutes' AS time_window_end
        FROM (
            SELECT 
                generate_series(
                    '{start_time_utc}'::timestamp, 
                    '{end_time_utc}'::timestamp, 
                    '10 minutes'::interval
                ) AS time_window
        ) t
    )
    SELECT
        tw.time_window AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore' AS timestamp_sg,
        MIN(tf.deal_price) AS min_price,
        MAX(tf.deal_price) AS max_price,
        COUNT(*) AS trade_count
    FROM time_windows tw
    LEFT JOIN public.trade_fill_fresh tf ON
        tf.created_at >= tw.time_window AND
        tf.created_at < tw.time_window_end AND
        tf.pair_name = '{pair_name}'
    GROUP BY tw.time_window
    ORDER BY tw.time_window DESC
    """
    
    try:
        df = pd.read_sql(query, engine)
        
        # Format timestamp and create time_label
        df['timestamp_sg'] = pd.to_datetime(df['timestamp_sg'])
        df['time_label'] = df['timestamp_sg'].dt.strftime('%H:%M')
        
        # Calculate volatility estimate based on price range
        df['volatility'] = None
        mask = (df['min_price'] > 0) & (df['min_price'].notnull()) & (df['max_price'].notnull()) & (df['trade_count'] >= 3)
        if any(mask):
            # Calculate normalized price range
            price_range = (df.loc[mask, 'max_price'] - df.loc[mask, 'min_price']) / df.loc[mask, 'min_price']
            
            # Convert to annualized volatility (assuming range ≈ 4*stddev)
            # 144 = number of 10-min periods in a day, 365 = days in a year
            df.loc[mask, 'volatility'] = price_range * np.sqrt(144 * 365) / 4
        
        return df
    except Exception as e:
        st.error(f"Error fetching price stats for {pair_name}: {e}")
        return pd.DataFrame(columns=['timestamp_sg', 'min_price', 'max_price', 'trade_count', 'time_label', 'volatility'])

# Function to merge edge and volatility data
def merge_edge_volatility(edge_df, vol_df):
    if edge_df.empty or vol_df.empty:
        return edge_df
        
    # Merge on time_label
    result = edge_df.copy()
    
    # Update volatility values where time labels match
    for label in vol_df['time_label'].unique():
        vol_value = vol_df.loc[vol_df['time_label'] == label, 'volatility'].values
        if len(vol_value) > 0 and not pd.isna(vol_value[0]):
            result.loc[result['time_label'] == label, 'volatility'] = vol_value[0]
    
    return result

# Function to create matrix with pairs as columns
def create_transposed_matrix(pair_data, column_name, time_labels, date_labels, selected_pairs):
    # Create a DataFrame with time slots as rows and pairs as columns
    matrix_data = {
        'time_slot': time_labels,
        'date': date_labels
    }
    
    # Add data for each pair
    for pair in selected_pairs:
        if pair in pair_data:
            pair_df = pair_data[pair]
            
            # Create a series with values indexed by time_label
            if not pair_df.empty and column_name in pair_df.columns:
                value_by_time = pd.Series(
                    pair_df[column_name].values,
                    index=pair_df['time_label']
                )
                
                # Add this pair's data to the matrix
                matrix_data[pair] = [value_by_time.get(time, None) for time in time_labels]
            else:
                # No data for this pair
                matrix_data[pair] = [None] * len(time_labels)
        else:
            # Pair not in data
            matrix_data[pair] = [None] * len(time_labels)
    
    # Convert to DataFrame
    df = pd.DataFrame(matrix_data)
    
    # Calculate data density for each pair to determine order
    data_counts = {}
    for pair in selected_pairs:
        if pair in df.columns and pair not in ['time_slot', 'date']:
            # Count non-null values for this pair
            data_counts[pair] = df[pair].notna().sum()
    
    # Sort pairs by data density (highest first)
    sorted_pairs = sorted(data_counts.keys(), key=lambda p: data_counts.get(p, 0), reverse=True)
    
    # Reorder columns: first time_slot and date, then pairs by data density
    reordered_columns = ['time_slot', 'date'] + sorted_pairs
    
    # Return reordered DataFrame
    return df[reordered_columns]

# Function to display matrix with custom formatting
def display_matrix(df, format_func, height=600):
    # Create a DataFrame with formatted values
    formatted_df = df.copy()
    
    # Process each numeric column
    for col in formatted_df.columns:
        if col not in ['time_slot', 'date']:
            # Format values using the provided function
            formatted_df[col] = formatted_df[col].apply(
                lambda x: format_func(x) if pd.notna(x) else ""
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
        height=height,
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

# Show pairs selected
st.write(f"Displaying data for {len(selected_pairs)} pairs")

# Progress bar
progress_bar = st.progress(0)
status_text = st.empty()

# Fetch data for all selected pairs
pair_data = {}

status_text.text(f"Processing pairs... (0/{len(selected_pairs)})")
for i, pair in enumerate(selected_pairs):
    progress_bar.progress((i) / len(selected_pairs))
    status_text.text(f"Processing {pair} ({i+1}/{len(selected_pairs)})")
    
    # Fetch edge data
    edge_df = fetch_edge_data(pair, start_time_sg, now_sg)
    
    # Fetch price stats for volatility calculation
    vol_df = fetch_price_stats(pair, start_time_sg, now_sg)
    
    # Merge edge and volatility data
    result_df = merge_edge_volatility(edge_df, vol_df)
    
    # Store the result
    pair_data[pair] = result_df

progress_bar.progress(1.0)
status_text.text(f"Processed {len(pair_data)}/{len(selected_pairs)} pairs")

# Create tabs
tab1, tab2 = st.tabs(["Edge", "Volatility"])

if pair_data:
    # Tab 1: Edge Matrix
    with tab1:
        st.markdown("## Edge Matrix (10min timeframe, Last 24 hours, Singapore Time)")
        st.markdown("### Edge = (Trading PNL + Taker Fee + Funding PNL + SL Fee - Rebate Amount) / Open Collateral")
        
        # Create edge matrix with pairs as columns
        edge_df = create_transposed_matrix(pair_data, 'edge', time_labels, date_labels, selected_pairs)
        
        if not edge_df.empty:
            # Format function for edge values
            def format_edge(x):
                return f"{x*100:.1f}%"
            
            # Display the matrix with edge formatting
            display_matrix(edge_df, format_edge)
            
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
        st.markdown("### Annualized Volatility based on price range within 10-minute windows")
        
        # Create volatility matrix with pairs as columns
        vol_df = create_transposed_matrix(pair_data, 'volatility', time_labels, date_labels, selected_pairs)
        
        if not vol_df.empty:
            # Format function for volatility values
            def format_volatility(x):
                return f"{x*100:.1f}%"
            
            # Display the matrix with volatility formatting
            display_matrix(vol_df, format_volatility)
            
            # Legend
            st.markdown("""
            **Volatility Legend:**
            <span style='background-color:rgba(0, 180, 0, 0.9);color:white;padding:2px 6px;border-radius:3px;'>Low (<20%)</span>
            <span style='background-color:rgba(150, 255, 150, 0.9);color:black;padding:2px 6px;border-radius:3px;'>Medium-Low (20% to 50%)</span>
            <span style='background-color:rgba(255, 255, 150, 0.9);color:black;padding:2px 6px;border-radius:3px;'>Medium (50% to 100%)</span>
            <span style='background-color:rgba(255, 150, 0, 0.9);color:black;padding:2px 6px;border-radius:3px;'>Medium-High (100% to 150%)</span>
            <span style='background-color:rgba(255, 0, 0, 0.9);color:white;padding:2px 6px;border-radius:3px;'>High (>150%)</span>
            """, unsafe_allow_html=True)
            
            # Explanation of the volatility calculation
            st.markdown("""
            **Notes on Simplified Volatility Calculation:**
            - Uses high-low price range within each 10-minute window
            - Annualizes volatility based on the normalized price range
            - Empty cells indicate insufficient price data (fewer than 3 trades)
            """)
        else:
            st.warning("No volatility data available for selected pairs.")

else:
    st.warning("No data available for selected pairs. Try selecting different pairs or refreshing the data.")

# Add performance info in sidebar
st.sidebar.markdown("### Performance Info")
st.sidebar.write(f"Processed {len(selected_pairs)} pairs")
st.sidebar.write(f"Data cached for 10 minutes")