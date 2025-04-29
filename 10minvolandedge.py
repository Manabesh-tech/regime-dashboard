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

# Generate time slots in reverse order (latest first)
def generate_time_slots():
    slots = []
    current = now_sg
    end_time = start_time_sg
    
    while current >= end_time:
        # Round to nearest 10 minute mark
        minute = current.minute
        rounded_minute = (minute // 10) * 10
        slot = current.replace(minute=rounded_minute, second=0, microsecond=0)
        slots.append(slot)
        current -= timedelta(minutes=10)
    
    return slots

time_slots = generate_time_slots()
time_labels = [slot.strftime("%H:%M") for slot in time_slots]
date_labels = [slot.strftime("%b %d") for slot in time_slots]

# Function to fetch data and calculate edge and volatility
@st.cache_data(ttl=600)
def calculate_edge_volatility(pair_name):
    # Convert to UTC for database query
    start_time_utc = start_time_sg.astimezone(pytz.utc)
    end_time_utc = now_sg.astimezone(pytz.utc)
    
    # Query with your colleague's PNL calculation method
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
    
    pnl_data AS (
      SELECT
        date_trunc('hour', created_at + INTERVAL '8 hour') + 
        INTERVAL '10 min' * FLOOR(EXTRACT(MINUTE FROM created_at + INTERVAL '8 hour')::INT / 10) AS time_slot,
        
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
        time_slot
    )
    
    SELECT
      t.slot AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore' AS timestamp_sg,
      COALESCE(p.total_pnl, 0) AS total_pnl,
      COALESCE(p.total_collateral, 0) AS total_collateral,
      p.price_array
    FROM
      time_intervals t
    LEFT JOIN
      pnl_data p ON t.slot = p.time_slot AT TIME ZONE 'Asia/Singapore' AT TIME ZONE 'UTC'
    ORDER BY
      t.slot DESC  -- Order by time descending (newest first)
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

# Show pairs selected
st.write(f"Displaying data for {len(selected_pairs)} pairs")

# Progress bar
progress_bar = st.progress(0)
status_text = st.empty()

# Fetch data for all selected pairs
pair_data = {}
for i, pair in enumerate(selected_pairs):
    progress_bar.progress(i / len(selected_pairs))
    status_text.text(f"Processing {pair} ({i+1}/{len(selected_pairs)})")
    
    data = calculate_edge_volatility(pair)
    if data is not None:
        pair_data[pair] = data

progress_bar.progress(1.0)
status_text.text(f"Processed {len(pair_data)}/{len(selected_pairs)} pairs")

# Create tabs
tab1, tab2 = st.tabs(["Edge", "Volatility"])

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

# Function to display edge matrix with custom formatting and colors
def display_edge_matrix(edge_df):
    # Create a DataFrame with formatted values
    formatted_df = edge_df.copy()
    
    # Create a custom CSS for coloring cells based on value
    cell_css = """
    <style>
    .very-negative { background-color: rgba(180, 0, 0, 0.9); color: white; }
    .negative { background-color: rgba(255, 0, 0, 0.9); color: white; }
    .slightly-negative { background-color: rgba(255, 150, 150, 0.9); color: black; }
    .neutral { background-color: rgba(255, 255, 150, 0.9); color: black; }
    .slightly-positive { background-color: rgba(150, 255, 150, 0.9); color: black; }
    .positive { background-color: rgba(0, 255, 0, 0.9); color: black; }
    .very-positive { background-color: rgba(0, 180, 0, 0.9); color: white; }
    </style>
    """
    
    # Process each numeric column
    for col in formatted_df.columns:
        if col not in ['time_slot', 'date']:
            # Format values as strings
            formatted_df[col] = formatted_df[col].apply(
                lambda x: f"{x*100:.1f}%" if isinstance(x, (int, float)) and not pd.isna(x) else ""
            )
    
    # Display using st.dataframe with custom formatting
    st.dataframe(
        formatted_df,
        height=600,
        use_container_width=True,
        column_config={
            "time_slot": st.column_config.TextColumn("Time", width="small"),
            "date": st.column_config.TextColumn("Date", width="small")
        }
    )
    
    return formatted_df

# Function to display volatility matrix with custom formatting and column width
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
    
    # Display using st.dataframe with column configuration
    st.dataframe(
        formatted_df,
        height=600,
        use_container_width=True,
        column_config={
            "time_slot": st.column_config.TextColumn("Time", width="small"),
            "date": st.column_config.TextColumn("Date", width="small")
        }
    )
    
    return formatted_df

# Function to add date separators to DataFrame display
def add_date_separators(df):
    result_df = pd.DataFrame()
    current_date = None
    
    # Process each row
    for idx, row in df.iterrows():
        date = row['date']
        
        # Add date separator if it's a new date
        if date != current_date:
            # Create a separator row
            separator = pd.DataFrame([{col: '' for col in df.columns}])
            separator.iloc[0, df.columns.get_indexer(['time_slot'])[0]] = f"--- {date} ---"
            
            # Add separator to results
            result_df = pd.concat([result_df, separator])
            current_date = date
        
        # Add the actual data row
        result_df = pd.concat([result_df, pd.DataFrame([row])])
    
    return result_df

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

else:
    st.warning("No data available for selected pairs. Try selecting different pairs or refreshing the data.")