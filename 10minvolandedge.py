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
col1, col2 = st.columns([3, 1])

with col1:
    select_all = st.checkbox("Select All Pairs", value=False)
    if select_all:
        selected_pairs = pairs
    else:
        selected_pairs = st.multiselect(
            "Select Pairs", 
            pairs,
            default=pairs[:5] if len(pairs) >= 5 else pairs
        )

with col2:
    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.experimental_rerun()

if not selected_pairs:
    st.warning("Please select at least one pair")
    st.stop()

# Generate time slots
def generate_time_slots():
    slots = []
    current = start_time_sg
    while current < now_sg:
        # Round to nearest 10 minute mark
        minute = current.minute
        rounded_minute = (minute // 10) * 10
        slot = current.replace(minute=rounded_minute, second=0, microsecond=0)
        slots.append(slot.strftime("%H:%M"))
        current += timedelta(minutes=10)
    return slots

time_slots = generate_time_slots()

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
      t.slot
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
            if prices is None or not isinstance(prices, list) or len(prices) < 2:
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
        
        df['volatility'] = df['price_array'].apply(calculate_volatility)
        
        return df
    
    except Exception as e:
        st.error(f"Error processing {pair_name}: {e}")
        return None

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

# Create matrix tables
def create_matrix(data_dict, time_slots, metric):
    # Create DataFrame with proper index
    matrix = {}
    
    for pair_name in selected_pairs:
        if pair_name in data_dict:
            # Convert to dictionary for quick lookup
            df = data_dict[pair_name]
            value_lookup = dict(zip(df['time_label'], df[metric]))
            
            # Create row with values for each time slot
            row_values = {}
            for slot in time_slots:
                row_values[slot] = value_lookup.get(slot, None)
            
            # Add to matrix
            matrix[pair_name] = row_values
    
    # Convert to DataFrame
    return pd.DataFrame.from_dict(matrix, orient='index')

# Function to style edge cells
def format_edge(val):
    if pd.isna(val) or val == 0:
        return '0'
    return f"{val*100:.1f}%"

def color_edge(val):
    if pd.isna(val) or val == 0:
        return 'background-color: #f5f5f5; color: #666666;'
    elif val < -0.1:
        return 'background-color: rgba(180, 0, 0, 0.9); color: white;'
    elif val < -0.05:
        return 'background-color: rgba(255, 0, 0, 0.9); color: white;'
    elif val < -0.01:
        return 'background-color: rgba(255, 150, 150, 0.9); color: black;'
    elif val < 0.01:
        return 'background-color: rgba(255, 255, 150, 0.9); color: black;'
    elif val < 0.05:
        return 'background-color: rgba(150, 255, 150, 0.9); color: black;'
    elif val < 0.1:
        return 'background-color: rgba(0, 255, 0, 0.9); color: black;'
    else:
        return 'background-color: rgba(0, 180, 0, 0.9); color: white;'

# Function to style volatility cells
def format_volatility(val):
    if pd.isna(val) or val == 0:
        return '0'
    return f"{val*100:.1f}%"

def color_volatility(val):
    if pd.isna(val) or val == 0:
        return 'background-color: #f5f5f5; color: #666666;'
    elif val < 0.3:
        return 'background-color: rgba(0, 255, 0, 0.9); color: black;'
    elif val < 0.6:
        return 'background-color: rgba(255, 255, 0, 0.9); color: black;'
    elif val < 1.0:
        return 'background-color: rgba(255, 165, 0, 0.9); color: black;'
    else:
        return 'background-color: rgba(255, 0, 0, 0.9); color: white;'

if pair_data:
    # Get time slots found in the data
    all_times = set()
    for df in pair_data.values():
        all_times.update(df['time_label'].tolist())
    
    # Create ordered list of time slots
    ordered_slots = [t for t in time_slots if t in all_times]
    
    # Tab 1: Edge Matrix
    with tab1:
        st.markdown("## Edge Matrix (10min timeframe, Last 24 hours, Singapore Time)")
        st.markdown("### Edge = PNL / Total Open Collateral")
        
        # Create edge matrix
        edge_matrix = create_matrix(pair_data, ordered_slots, 'edge')
        
        if not edge_matrix.empty:
            # Format edge values
            formatted_edge = edge_matrix.applymap(format_edge)
            
            # Style the table
            styled_edge = edge_matrix.style.applymap(color_edge)
            styled_edge.data = formatted_edge
            
            # Display the table
            st.dataframe(styled_edge, height=600, use_container_width=True)
            
            # Legend
            st.markdown("**Edge Legend:** <span style='color:red'>Negative</span> | <span style='color:yellow'>Neutral</span> | <span style='color:green'>Positive</span>", unsafe_allow_html=True)
        else:
            st.warning("No edge data available for selected pairs.")
    
    # Tab 2: Volatility Matrix
    with tab2:
        st.markdown("## Volatility Matrix (10min timeframe, Last 24 hours, Singapore Time)")
        st.markdown("### Annualized Volatility = StdDev(Log Returns) * sqrt(trading periods per year)")
        
        # Create volatility matrix
        volatility_matrix = create_matrix(pair_data, ordered_slots, 'volatility')
        
        if not volatility_matrix.empty:
            # Format volatility values
            formatted_vol = volatility_matrix.applymap(format_volatility)
            
            # Style the table
            styled_vol = volatility_matrix.style.applymap(color_volatility)
            styled_vol.data = formatted_vol
            
            # Display the table
            st.dataframe(styled_vol, height=600, use_container_width=True)
            
            # Legend
            st.markdown("**Volatility Legend:** <span style='color:green'>Low</span> | <span style='color:yellow'>Medium</span> | <span style='color:orange'>High</span> | <span style='color:red'>Extreme</span>", unsafe_allow_html=True)
        else:
            st.warning("No volatility data available for selected pairs.")

else:
    st.warning("No data available for selected pairs. Try selecting different pairs or refreshing the data.")