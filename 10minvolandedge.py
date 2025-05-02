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
        # Default to just showing 3 pairs to avoid performance issues
        selected_pairs = pairs[:3] if len(pairs) >= 3 else pairs

with col2:
    select_all = st.checkbox("Select All Pairs", value=False)
    if select_all:
        # Add a max limit to avoid overloading
        max_pairs = 5
        if len(pairs) > max_pairs:
            st.warning(f"Limited to top {max_pairs} pairs for performance")
            selected_pairs = pairs[:max_pairs]
        else:
            selected_pairs = pairs

with col3:
    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.experimental_rerun()

# Show pairs selected
st.write(f"Displaying data for {len(selected_pairs)} pairs")

# Progress bar
progress_bar = st.progress(0)
status_text = st.empty()

# Create tabs
tab1, tab2 = st.tabs(["Edge", "Volatility"])

# Process each pair and display data
edge_data = {}
vol_data = {}

# Process each pair
for i, pair in enumerate(selected_pairs):
    progress_bar.progress(i / max(1, len(selected_pairs)))
    status_text.text(f"Processing {pair} ({i+1}/{len(selected_pairs)})")
    
    # Convert to UTC for database query
    start_time_utc = start_time_sg.astimezone(pytz.utc)
    end_time_utc = now_sg.astimezone(pytz.utc)
    
    # Edge Query 
    try:
        edge_query = f"""
        WITH time_intervals AS (
          SELECT 
            t.interval_start,
            t.interval_start + INTERVAL '10 minutes' AS interval_end
          FROM (
            SELECT 
              generate_series(
                '{start_time_utc}'::timestamp, 
                '{end_time_utc}'::timestamp, 
                '10 minutes'::interval
              ) AS interval_start
          ) t
        ),
        pnl_data AS (
          SELECT
            ti.interval_start,
            SUM(-1 * taker_pnl * collateral_price) AS trading_pnl,
            SUM(CASE WHEN taker_fee_mode = 1 AND taker_way IN (1, 3) THEN taker_fee * collateral_price ELSE 0 END) AS taker_fee,
            SUM(CASE WHEN taker_way = 0 THEN -1 * funding_fee * collateral_price ELSE 0 END) AS funding_pnl,
            SUM(taker_sl_fee * collateral_price + maker_sl_fee) AS sl_fee
          FROM time_intervals ti
          LEFT JOIN public.trade_fill_fresh tf ON
            tf.created_at >= ti.interval_start AND 
            tf.created_at < ti.interval_end AND
            tf.pair_name = '{pair}'
          GROUP BY ti.interval_start
        ),
        collateral_data AS (
          SELECT
            ti.interval_start,
            SUM(deal_vol * collateral_price) AS open_collateral
          FROM time_intervals ti
          LEFT JOIN public.trade_fill_fresh tf ON
            tf.created_at >= ti.interval_start AND 
            tf.created_at < ti.interval_end AND
            tf.pair_name = '{pair}' AND
            tf.taker_fee_mode = 2 AND 
            tf.taker_way IN (1, 3)
          GROUP BY ti.interval_start
        )
        SELECT
          ti.interval_start + INTERVAL '8 hour' AS timestamp_sg,
          CASE
            WHEN COALESCE(cd.open_collateral, 0) = 0 THEN 0
            ELSE (COALESCE(pd.trading_pnl, 0) + 
                  COALESCE(pd.taker_fee, 0) + 
                  COALESCE(pd.funding_pnl, 0) + 
                  COALESCE(pd.sl_fee, 0)) / 
                 cd.open_collateral
          END AS edge
        FROM time_intervals ti
        LEFT JOIN pnl_data pd ON ti.interval_start = pd.interval_start
        LEFT JOIN collateral_data cd ON ti.interval_start = cd.interval_start
        ORDER BY ti.interval_start DESC
        """
        
        edge_df = pd.read_sql(edge_query, engine)
        
        # Convert timestamp to Singapore timezone
        edge_df['timestamp_sg'] = pd.to_datetime(edge_df['timestamp_sg'])
        edge_df['time_label'] = edge_df['timestamp_sg'].dt.strftime('%H:%M')
        
        # Store in dictionary
        edge_data[pair] = edge_df
        
    except Exception as e:
        st.error(f"Error processing edge data for {pair}: {e}")
        edge_data[pair] = pd.DataFrame(columns=['timestamp_sg', 'edge', 'time_label'])
    
    # Volatility Query
    try:
        vol_query = f"""
        WITH time_periods AS (
          SELECT 
            t.period_start,
            t.period_start + INTERVAL '10 minutes' AS period_end
          FROM (
            SELECT 
              generate_series(
                '{start_time_utc}'::timestamp, 
                '{end_time_utc}'::timestamp, 
                '10 minutes'::interval
              ) AS period_start
          ) t
        ),
        price_stats AS (
          SELECT
            tp.period_start,
            MIN(tf.deal_price) AS min_price,
            MAX(tf.deal_price) AS max_price,
            COUNT(*) AS trade_count
          FROM time_periods tp
          LEFT JOIN public.trade_fill_fresh tf ON
            tf.created_at >= tp.period_start AND
            tf.created_at < tp.period_end AND
            tf.pair_name = '{pair}'
          GROUP BY tp.period_start
        )
        SELECT
          ps.period_start + INTERVAL '8 hour' AS timestamp_sg,
          ps.min_price,
          ps.max_price,
          ps.trade_count,
          CASE
            WHEN ps.min_price > 0 AND ps.min_price IS NOT NULL AND ps.max_price IS NOT NULL AND ps.trade_count >= 3
            THEN (ps.max_price - ps.min_price) / ps.min_price
            ELSE NULL
          END AS price_range
        FROM price_stats ps
        ORDER BY ps.period_start DESC
        """
        
        vol_df = pd.read_sql(vol_query, engine)
        
        # Convert timestamp to Singapore timezone
        vol_df['timestamp_sg'] = pd.to_datetime(vol_df['timestamp_sg'])
        vol_df['time_label'] = vol_df['timestamp_sg'].dt.strftime('%H:%M')
        
        # Calculate volatility from price range
        vol_df['volatility'] = None
        mask = vol_df['price_range'].notna() & (vol_df['trade_count'] >= 3)
        if any(mask):
            # Convert price range to annualized volatility (statistical rule of thumb: range ≈ 4*stddev)
            # 144 = number of 10-min periods in day, 365 = days in year
            vol_df.loc[mask, 'volatility'] = vol_df.loc[mask, 'price_range'] * np.sqrt(144 * 365) / 4
        
        # Store in dictionary
        vol_data[pair] = vol_df
        
    except Exception as e:
        st.error(f"Error processing volatility data for {pair}: {e}")
        vol_data[pair] = pd.DataFrame(columns=['timestamp_sg', 'min_price', 'max_price', 'trade_count', 'price_range', 'volatility', 'time_label'])

progress_bar.progress(1.0)
status_text.text(f"Processed {len(selected_pairs)} pairs")

# Create Edge Matrix
with tab1:
    st.markdown("## Edge Matrix (10min timeframe, Last 24 hours, Singapore Time)")
    
    # Create a matrix with time slots as rows and pairs as columns
    edge_matrix = pd.DataFrame({
        'Time': time_labels,
        'Date': date_labels
    })
    
    # Add edge data for each pair
    for pair in selected_pairs:
        if pair in edge_data:
            df = edge_data[pair]
            if not df.empty:
                # Create a mapping of time label to edge value
                edge_by_time = {}
                for _, row in df.iterrows():
                    if pd.notna(row['edge']):
                        edge_by_time[row['time_label']] = row['edge']
                
                # Add the pair's edge values to the matrix
                edge_values = []
                for time in time_labels:
                    edge_values.append(edge_by_time.get(time, None))
                
                edge_matrix[pair] = edge_values
    
    # Display the edge matrix
    st.dataframe(edge_matrix, use_container_width=True)
    
    # Legend
    st.markdown("""
    **Edge Legend:**
    - Very Negative: < -10%
    - Negative: -10% to -5%
    - Slightly Negative: -5% to -1%
    - Neutral: -1% to 1%
    - Slightly Positive: 1% to 5%
    - Positive: 5% to 10%
    - Very Positive: > 10%
    """)

# Create Volatility Matrix
with tab2:
    st.markdown("## Volatility Matrix (10min timeframe, Last 24 hours, Singapore Time)")
    
    # Create a matrix with time slots as rows and pairs as columns
    vol_matrix = pd.DataFrame({
        'Time': time_labels,
        'Date': date_labels
    })
    
    # Add volatility data for each pair
    for pair in selected_pairs:
        if pair in vol_data:
            df = vol_data[pair]
            if not df.empty:
                # Create a mapping of time label to volatility value
                vol_by_time = {}
                for _, row in df.iterrows():
                    if pd.notna(row['volatility']):
                        vol_by_time[row['time_label']] = row['volatility']
                
                # Add the pair's volatility values to the matrix
                vol_values = []
                for time in time_labels:
                    vol_values.append(vol_by_time.get(time, None))
                
                vol_matrix[pair] = vol_values
    
    # Display the volatility matrix
    st.dataframe(vol_matrix, use_container_width=True)
    
    # Legend
    st.markdown("""
    **Volatility Legend:**
    - Low: < 20%
    - Medium-Low: 20% to 50%
    - Medium: 50% to 100%
    - Medium-High: 100% to 150%
    - High: > 150%
    """)
    
    # Explanation
    st.markdown("""
    **Notes on Volatility Calculation:**
    - Based on price range (high-low) within each 10-minute window
    - Requires at least 3 trades in the interval
    - Annualized for easier comparison
    """)

# Add performance info in sidebar
st.sidebar.markdown("### Performance Info")
st.sidebar.write(f"Processed {len(selected_pairs)} pairs")
st.sidebar.write(f"Data cached for 10 minutes")