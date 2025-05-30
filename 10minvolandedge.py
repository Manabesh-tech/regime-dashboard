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
        f"postgresql://{db_config['user']}:{db_config['password']}"
        f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    )
    engine = create_engine(
        db_uri,
        isolation_level="AUTOCOMMIT",  # 设置自动提交模式
        pool_size=5,  # 连接池大小
        max_overflow=10,  # 最大溢出连接数
        pool_timeout=30,  # 连接超时时间
        pool_recycle=1800,  # 连接回收时间(30分钟)
        pool_pre_ping=True,  # 使用连接前先测试连接是否有效
        pool_use_lifo=True,  # 使用后进先出,减少空闲连接
        echo=False  # 不打印 SQL 语句
    )
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
tab1, tab2, tab3 = st.tabs(["Edge", "Volatility", "Spreads"])

# Process each pair and display data
edge_data = {}
vol_data = {}
spread_data = {}

# Process each pair
for i, pair in enumerate(selected_pairs):
    progress_percent = i / max(1, len(selected_pairs) * 3)  # Divide by 3 because we have 3 types of data
    progress_bar.progress(progress_percent)
    status_text.text(f"Processing {pair} - Edge Data ({i+1}/{len(selected_pairs)})")
    
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
    
    # Update progress
    progress_percent = (i / max(1, len(selected_pairs)) + 1/3)
    progress_bar.progress(progress_percent)
    status_text.text(f"Processing {pair} - Volatility Data ({i+1}/{len(selected_pairs)})")
    
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
    
    # Update progress
    progress_percent = (i / max(1, len(selected_pairs)) + 2/3)
    progress_bar.progress(progress_percent)
    status_text.text(f"Processing {pair} - Spread Data ({i+1}/{len(selected_pairs)})")
    
    # Market Spread Query
    try:
        spread_query = f"""
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
        spread_data AS (
          SELECT 
            ti.interval_start,
            AVG(fee1) as avg_spread
          FROM time_intervals ti
          LEFT JOIN oracle_exchange_fee oef ON
            oef.time_group >= ti.interval_start AND 
            oef.time_group < ti.interval_end AND
            oef.pair_name = '{pair}' AND
            oef.source IN ('binanceFuture', 'gateFuture', 'hyperliquidFuture')
          GROUP BY ti.interval_start
        )
        SELECT
          ti.interval_start + INTERVAL '8 hour' AS timestamp_sg,
          COALESCE(sd.avg_spread, NULL) AS avg_spread
        FROM time_intervals ti
        LEFT JOIN spread_data sd ON ti.interval_start = sd.interval_start
        ORDER BY ti.interval_start DESC
        """
        
        spread_df = pd.read_sql(spread_query, engine)
        
        # Convert timestamp to Singapore timezone
        spread_df['timestamp_sg'] = pd.to_datetime(spread_df['timestamp_sg'])
        spread_df['time_label'] = spread_df['timestamp_sg'].dt.strftime('%H:%M')
        
        # Store in dictionary
        spread_data[pair] = spread_df
        
    except Exception as e:
        st.error(f"Error processing spread data for {pair}: {e}")
        spread_data[pair] = pd.DataFrame(columns=['timestamp_sg', 'avg_spread', 'time_label'])

progress_bar.progress(1.0)
status_text.text(f"Processed {len(selected_pairs)} pairs")

# Function to calculate data density for pairs
def calculate_data_density(data_dict, value_column):
    data_counts = {}
    for pair, df in data_dict.items():
        if not df.empty and value_column in df.columns:
            # Count non-null values
            data_counts[pair] = df[value_column].notna().sum()
        else:
            data_counts[pair] = 0
    return data_counts

# Create Edge Matrix
with tab1:
    st.markdown("## Edge Matrix (10min timeframe, Last 24 hours, Singapore Time)")
    st.markdown("### Edge = (Trading PNL + Taker Fee + Funding PNL + SL Fee - Rebate Amount) / Open Collateral")
    
    # Create a matrix with time slots as rows and pairs as columns
    edge_matrix = pd.DataFrame({
        'Time': time_labels,
        'Date': date_labels
    })
    
    # Calculate data density for edge
    edge_counts = calculate_data_density(edge_data, 'edge')
    
    # Sort pairs by data density (highest first)
    sorted_pairs = sorted(selected_pairs, key=lambda p: edge_counts.get(p, 0), reverse=True)
    
    # Add edge data for pairs in density-sorted order
    for pair in sorted_pairs:
        if pair in edge_data:
            df = edge_data[pair]
            if not df.empty:
                # Create a mapping of time label to edge value
                edge_by_time = {}
                for _, row in df.iterrows():
                    if pd.notna(row['edge']):
                        # Format edge as percentage
                        edge_by_time[row['time_label']] = f"{row['edge']*100:.1f}%"
                
                # Add the pair's edge values to the matrix
                edge_values = []
                for time in time_labels:
                    edge_values.append(edge_by_time.get(time, ""))
                
                edge_matrix[pair] = edge_values
    
    # Display the edge matrix
    st.dataframe(edge_matrix, use_container_width=True)
    
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

# Create Volatility Matrix
with tab2:
    st.markdown("## Volatility Matrix (10min timeframe, Last 24 hours, Singapore Time)")
    st.markdown("### Annualized Volatility based on price range within 10-minute windows")
    
    # Create a matrix with time slots as rows and pairs as columns
    vol_matrix = pd.DataFrame({
        'Time': time_labels,
        'Date': date_labels
    })
    
    # Calculate data density for volatility
    vol_counts = calculate_data_density(vol_data, 'volatility')
    
    # Sort pairs by data density (highest first)
    sorted_pairs = sorted(selected_pairs, key=lambda p: vol_counts.get(p, 0), reverse=True)
    
    # Add volatility data for pairs in density-sorted order
    for pair in sorted_pairs:
        if pair in vol_data:
            df = vol_data[pair]
            if not df.empty:
                # Create a mapping of time label to volatility value
                vol_by_time = {}
                for _, row in df.iterrows():
                    if pd.notna(row['volatility']):
                        # Format volatility as percentage
                        vol_by_time[row['time_label']] = f"{row['volatility']*100:.1f}%"
                
                # Add the pair's volatility values to the matrix
                vol_values = []
                for time in time_labels:
                    vol_values.append(vol_by_time.get(time, ""))
                
                vol_matrix[pair] = vol_values
    
    # Display the volatility matrix
    st.dataframe(vol_matrix, use_container_width=True)
    
    # Legend
    st.markdown("""
    **Volatility Legend:**
    <span style='background-color:rgba(0, 180, 0, 0.9);color:white;padding:2px 6px;border-radius:3px;'>Low (<20%)</span>
    <span style='background-color:rgba(150, 255, 150, 0.9);color:black;padding:2px 6px;border-radius:3px;'>Medium-Low (20% to 50%)</span>
    <span style='background-color:rgba(255, 255, 150, 0.9);color:black;padding:2px 6px;border-radius:3px;'>Medium (50% to 100%)</span>
    <span style='background-color:rgba(255, 150, 0, 0.9);color:black;padding:2px 6px;border-radius:3px;'>Medium-High (100% to 150%)</span>
    <span style='background-color:rgba(255, 0, 0, 0.9);color:white;padding:2px 6px;border-radius:3px;'>High (>150%)</span>
    """, unsafe_allow_html=True)
    
    # Explanation
    st.markdown("""
    **Notes on Volatility Calculation:**
    - Based on price range (high-low) within each 10-minute window
    - Requires at least 3 trades in the interval
    - Annualized for easier comparison
    """)

# Create Spreads Matrix
with tab3:
    st.markdown("## Market Spreads Matrix (10min timeframe, Last 24 hours, Singapore Time)")
    st.markdown("### Spreads shown in basis points (1bp = 0.01%)")
    
    # Create a matrix with time slots as rows and pairs as columns
    spread_matrix = pd.DataFrame({
        'Time': time_labels,
        'Date': date_labels
    })
    
    # Calculate data density for spreads
    spread_counts = calculate_data_density(spread_data, 'avg_spread')
    
    # Sort pairs by data density (highest first)
    sorted_pairs = sorted(selected_pairs, key=lambda p: spread_counts.get(p, 0), reverse=True)
    
    # Add spread data for pairs in density-sorted order
    for pair in sorted_pairs:
        if pair in spread_data:
            df = spread_data[pair]
            if not df.empty:
                # Create a mapping of time label to spread value
                spread_by_time = {}
                for _, row in df.iterrows():
                    if pd.notna(row['avg_spread']):
                        # Format spread as basis points
                        spread_by_time[row['time_label']] = f"{row['avg_spread']*10000:.2f}"
                
                # Add the pair's spread values to the matrix
                spread_values = []
                for time in time_labels:
                    spread_values.append(spread_by_time.get(time, ""))
                
                spread_matrix[pair] = spread_values
    
    # Display the spreads matrix
    st.dataframe(spread_matrix, use_container_width=True)
    
    # Legend
    st.markdown("""
    **Spread Legend:**
    <span style='background-color:rgba(0, 180, 0, 0.9);color:white;padding:2px 6px;border-radius:3px;'>Very Low (<2.5)</span>
    <span style='background-color:rgba(150, 255, 150, 0.9);color:black;padding:2px 6px;border-radius:3px;'>Low (2.5 to 5)</span>
    <span style='background-color:rgba(255, 255, 150, 0.9);color:black;padding:2px 6px;border-radius:3px;'>Medium (5 to 10)</span>
    <span style='background-color:rgba(255, 150, 0, 0.9);color:black;padding:2px 6px;border-radius:3px;'>High (10 to 20)</span>
    <span style='background-color:rgba(255, 0, 0, 0.9);color:white;padding:2px 6px;border-radius:3px;'>Very High (>20)</span>
    """, unsafe_allow_html=True)
    
    # Notes
    st.markdown("""
    **Notes:**
    - Market spreads are averaged from binanceFuture, gateFuture, and hyperliquidFuture
    - Spreads are shown in basis points (1bp = 0.01%)
    - Empty cells indicate no data for that time slot
    """)

# Add performance info in sidebar
st.sidebar.markdown("### Performance Info")
st.sidebar.write(f"Processed {len(selected_pairs)} pairs")
st.sidebar.write(f"Data cached for 10 minutes")