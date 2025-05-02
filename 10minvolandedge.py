import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from datetime import datetime, timedelta
import pytz
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(page_title="Edge & Volatility Matrix", layout="wide")
st.title("Edge & Volatility Matrix (10min)")
st.subheader("All Trading Pairs - Last 24 Hours (Singapore Time)")

# Set SQL query timeout
@event.listens_for(Engine, "before_cursor_execute")
def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    conn.connection.set_isolation_level(0)
    cursor.execute("SET statement_timeout = 60000;")  # 60 seconds timeout

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

# Fetch pairs with activity filter
@st.cache_data(ttl=600)
def fetch_pairs():
    # More efficient query with activity filter
    query = """
    SELECT DISTINCT pair_name 
    FROM public.trade_fill_fresh 
    WHERE created_at > NOW() - INTERVAL '1 day'
    ORDER BY pair_name
    """
    
    try:
        with engine.connect().execution_options(timeout=30) as conn:
            df = pd.read_sql(query, conn)
            if df.empty:
                return []
            return df['pair_name'].tolist()
    except Exception as e:
        logger.error(f"Error fetching pairs: {e}")
        return ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT", "PEPE/USDT"]

# Generate time slots once
@st.cache_data(ttl=3600)
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
col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

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
        max_pairs = 20
        if len(pairs) > max_pairs:
            st.warning(f"Limited to top {max_pairs} pairs for performance")
            selected_pairs = pairs[:max_pairs]
        else:
            selected_pairs = pairs

with col3:
    # Add option to limit number of pairs
    max_selected = st.number_input("Max Pairs", min_value=1, max_value=50, value=10)
    if len(selected_pairs) > max_selected:
        selected_pairs = selected_pairs[:max_selected]
        st.info(f"Showing top {max_selected} pairs")

with col4:
    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.experimental_rerun()

# Function to batch fetch edge and volatility data
@st.cache_data(ttl=600)
def batch_fetch_edge_volatility(selected_pairs, start_time_sg, now_sg, batch_size=5):
    # Convert to UTC for database query
    start_time_utc = start_time_sg.astimezone(pytz.utc)
    end_time_utc = now_sg.astimezone(pytz.utc)
    
    all_results = {}
    
    # Process in smaller batches
    for i in range(0, len(selected_pairs), batch_size):
        batch_pairs = selected_pairs[i:i+batch_size]
        pairs_str = "', '".join(batch_pairs)
        
        # Edge query modified to handle multiple pairs
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
        pair_names AS (
          SELECT unnest(ARRAY['{pairs_str}']) AS pair_name
        ),
        time_pairs AS (
          SELECT 
            t.interval_start,
            p.pair_name
          FROM time_intervals t
          CROSS JOIN pair_names p
        ),
        pnl_data AS (
          SELECT
            date_trunc('hour', created_at + INTERVAL '8 hour') + 
              INTERVAL '10 min' * floor(extract(minute from created_at + INTERVAL '8 hour') / 10) AS interval_start,
            pair_name,
            SUM(-1 * taker_pnl * collateral_price) AS trading_pnl,
            SUM(CASE WHEN taker_fee_mode = 1 AND taker_way IN (1, 3) THEN taker_fee * collateral_price ELSE 0 END) AS taker_fee,
            SUM(CASE WHEN taker_way = 0 THEN -1 * funding_fee * collateral_price ELSE 0 END) AS funding_pnl,
            SUM(taker_sl_fee * collateral_price + maker_sl_fee) AS sl_fee
          FROM public.trade_fill_fresh
          WHERE created_at BETWEEN '{start_time_utc}' AND '{end_time_utc}'
            AND pair_name IN ('{pairs_str}')
          GROUP BY 1, 2
        ),
        collateral_data AS (
          SELECT
            date_trunc('hour', created_at + INTERVAL '8 hour') + 
              INTERVAL '10 min' * floor(extract(minute from created_at + INTERVAL '8 hour') / 10) AS interval_start,
            pair_name,
            SUM(deal_vol * collateral_price) AS open_collateral
          FROM public.trade_fill_fresh
          WHERE taker_fee_mode = 2 AND taker_way IN (1, 3)
            AND created_at BETWEEN '{start_time_utc}' AND '{end_time_utc}'
            AND pair_name IN ('{pairs_str}')
          GROUP BY 1, 2
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
          tp.interval_start AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore' AS timestamp_sg,
          tp.pair_name,
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
        FROM time_pairs tp
        LEFT JOIN pnl_data pd ON tp.interval_start = pd.interval_start AND tp.pair_name = pd.pair_name
        LEFT JOIN collateral_data cd ON tp.interval_start = cd.interval_start AND tp.pair_name = cd.pair_name
        LEFT JOIN rebate_data rd ON tp.interval_start = rd.interval_start
        ORDER BY tp.pair_name, timestamp_sg DESC
        """
        
        # Improved volatility query for multiple pairs
        volatility_query = f"""
        WITH all_time_intervals AS (
          -- Generate all 2-minute intervals for the past 24 hours for each pair
          SELECT 
            generate_series(
              date_trunc('hour', '{start_time_utc}'::timestamp) + 
              INTERVAL '2 min' * floor(EXTRACT(MINUTE FROM '{start_time_utc}'::timestamp) / 2),
              '{end_time_utc}'::timestamp,
              INTERVAL '2 minutes'
            ) AS interval_start,
            
            -- Group these 2-minute intervals into their parent 10-minute intervals
            date_trunc('hour', generate_series(
              date_trunc('hour', '{start_time_utc}'::timestamp) + 
              INTERVAL '2 min' * floor(EXTRACT(MINUTE FROM '{start_time_utc}'::timestamp) / 2),
              '{end_time_utc}'::timestamp,
              INTERVAL '2 minutes'
            )) + 
            INTERVAL '10 min' * floor(EXTRACT(MINUTE FROM generate_series(
              date_trunc('hour', '{start_time_utc}'::timestamp) + 
              INTERVAL '2 min' * floor(EXTRACT(MINUTE FROM '{start_time_utc}'::timestamp) / 2),
              '{end_time_utc}'::timestamp,
              INTERVAL '2 minutes'
            )) / 10) AS parent_interval,
            
            -- Add pair to the matrix
            unnest(ARRAY['{pairs_str}']) AS pair_name
        ),
        
        -- Get all trades for the requested pairs in the time range
        pair_trades AS (
          SELECT 
            created_at,
            pair_name,
            deal_price
          FROM 
            public.trade_fill_fresh
          WHERE 
            created_at BETWEEN '{start_time_utc}' AND '{end_time_utc}'
            AND pair_name IN ('{pairs_str}')
        ),
        
        -- Join trades to time intervals
        interval_prices AS (
          SELECT
            ati.interval_start,
            ati.parent_interval,
            ati.pair_name,
            (
              SELECT t.deal_price 
              FROM pair_trades t
              WHERE t.created_at < ati.interval_start + INTERVAL '2 minutes'
                AND t.created_at >= ati.interval_start
                AND t.pair_name = ati.pair_name
              ORDER BY t.created_at DESC
              LIMIT 1
            ) AS closing_price
          FROM all_time_intervals ati
        )
        
        -- Get all prices at 2-minute intervals, grouped by 10-minute parent intervals and pair
        SELECT
          parent_interval AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore' AS timestamp_sg,
          pair_name,
          array_agg(closing_price ORDER BY interval_start) FILTER (WHERE closing_price IS NOT NULL) AS price_array,
          count(closing_price) FILTER (WHERE closing_price IS NOT NULL) AS price_count
        FROM
          interval_prices
        GROUP BY
          parent_interval, pair_name
        ORDER BY
          pair_name, parent_interval DESC
        """
        
        try:
            # Execute the edge query with timeout
            with engine.connect().execution_options(timeout=60) as conn:
                edge_df = pd.read_sql(edge_query, conn)
                
                # Format timestamps and create time_label for edge data
                edge_df['timestamp_sg'] = pd.to_datetime(edge_df['timestamp_sg'])
                edge_df['time_label'] = edge_df['timestamp_sg'].dt.strftime('%H:%M')
                
                # Rename house_edge to edge for consistency
                edge_df.rename(columns={'house_edge': 'edge'}, inplace=True)
            
            # Execute the volatility query with timeout
            with engine.connect().execution_options(timeout=60) as conn:
                vol_df = pd.read_sql(volatility_query, conn)
                
                # Format timestamps
                vol_df['timestamp_sg'] = pd.to_datetime(vol_df['timestamp_sg'])
                vol_df['time_label'] = vol_df['timestamp_sg'].dt.strftime('%H:%M')
            
            # Process each pair in the batch
            for pair in batch_pairs:
                # Filter data for this specific pair
                pair_edge_df = edge_df[edge_df['pair_name'] == pair].copy()
                pair_vol_df = vol_df[vol_df['pair_name'] == pair].copy()
                
                # Calculate volatility with improved method
                volatility_results = []
                
                for _, row in pair_vol_df.iterrows():
                    try:
                        price_array = row['price_array']
                        price_count = row['price_count']
                        timestamp = row['timestamp_sg']
                        time_label = row['time_label']
                        
                        # Initialize volatility as None (will be updated if calculation succeeds)
                        volatility = None
                        
                        # Print debug info
                        logger.info(f"Processing volatility for {pair} at {time_label}")
                        logger.info(f"Price count: {price_count}")
                        logger.info(f"Price array type: {type(price_array)}")
                        
                        # Skip calculation if no price data
                        if price_array is None or price_count < 3:
                            logger.info(f"Insufficient price data for {pair} at {time_label}")
                        else:
                            # Parse the array - handle all possible formats
                            prices = []
                            
                            # Case 1: String representation like '{123,456,789}'
                            if isinstance(price_array, str):
                                try:
                                    # Strip curly braces and split
                                    clean_str = price_array.strip('{}')
                                    if clean_str:  # Only process if not empty
                                        prices_str = clean_str.split(',')
                                        for p in prices_str:
                                            if p and p.strip() != 'NULL' and p.strip() != 'None':
                                                try:
                                                    prices.append(float(p.strip()))
                                                except (ValueError, TypeError) as e:
                                                    logger.error(f"Could not convert {p} to float: {e}")
                                except Exception as e:
                                    logger.error(f"Error parsing string array: {e}")
                            
                            # Case 2: List or array
                            elif isinstance(price_array, (list, np.ndarray)):
                                for p in price_array:
                                    if p is not None:
                                        try:
                                            prices.append(float(p))
                                        except (ValueError, TypeError) as e:
                                            logger.error(f"Could not convert {p} to float: {e}")
                            
                            # Case 3: Other types (like PostgreSQL-specific array types)
                            else:
                                try:
                                    # Try converting to list first
                                    price_list = list(price_array)
                                    for p in price_list:
                                        if p is not None:
                                            try:
                                                prices.append(float(p))
                                            except (ValueError, TypeError) as e:
                                                logger.error(f"Could not convert {p} to float: {e}")
                                except Exception as e:
                                    logger.error(f"Could not convert price_array to list: {e}")
                                    # Last resort: try string conversion and parsing
                                    try:
                                        arr_str = str(price_array)
                                        # Clean up the string (remove brackets, etc.)
                                        clean_str = arr_str.strip('{}[]()').replace('\'', '').replace('"', '')
                                        if clean_str:
                                            parts = clean_str.split(',')
                                            for p in parts:
                                                if p and p.strip() != 'NULL' and p.strip() != 'None':
                                                    try:
                                                        prices.append(float(p.strip()))
                                                    except (ValueError, TypeError):
                                                        pass
                                    except Exception as nested_e:
                                        logger.error(f"Failed string parsing fallback: {nested_e}")
                            
                            # Log the parsed prices for debugging
                            logger.info(f"Parsed {len(prices)} valid prices: {prices[:5]}...")
                            
                            # Calculate volatility with robust error handling
                            if len(prices) >= 3:  # Need at least 3 price points
                                try:
                                    # Convert to numpy array and ensure they're floats
                                    prices_array = np.array(prices, dtype=np.float64)
                                    
                                    # Log min, max, mean prices to verify data quality
                                    logger.info(f"Price stats: min={np.min(prices_array):.2f}, max={np.max(prices_array):.2f}, mean={np.mean(prices_array):.2f}")
                                    
                                    # Calculate returns
                                    # Check for zero prices that would cause division error
                                    if np.any(prices_array[:-1] == 0):
                                        logger.warning(f"Zero prices detected for {pair} at {time_label}, skipping volatility calculation")
                                    else:
                                        # Calculate simple returns
                                        returns = np.diff(prices_array) / prices_array[:-1]
                                        
                                        # Standard deviation of returns
                                        std_dev = np.std(returns)
                                        
                                        # Annualize based on actual number of intervals
                                        actual_intervals = len(returns)
                                        # Scale to annual vol (assuming 5 intervals per 10min window, 6 per hour, 24 hours, 365 days)
                                        intervals_per_year = (5 * 6 * 24 * 365) / actual_intervals
                                        annualized_vol = std_dev * np.sqrt(intervals_per_year)
                                        
                                        logger.info(f"Calculated volatility: {annualized_vol*100:.2f}% from {actual_intervals} intervals")
                                        volatility = annualized_vol
                                        
                                except Exception as e:
                                    logger.error(f"Error in numpy calculation: {e}")
                            else:
                                logger.info(f"Not enough valid prices ({len(prices)}) for volatility calculation")
                        
                        # Add to results
                        volatility_results.append({
                            'timestamp_sg': timestamp,
                            'time_label': time_label,
                            'volatility': volatility
                        })
                        
                    except Exception as e:
                        logger.error(f"Error in volatility calculation for {pair}: {e}")
                        volatility_results.append({
                            'timestamp_sg': row['timestamp_sg'],
                            'time_label': row['time_label'],
                            'volatility': None
                        })
                
                # Convert volatility results to dataframe
                vol_results_df = pd.DataFrame(volatility_results)
                
                # Merge edge and volatility data
                if not pair_edge_df.empty and not vol_results_df.empty:
                    result_df = pd.merge(
                        pair_edge_df,
                        vol_results_df[['time_label', 'volatility']],
                        on='time_label',
                        how='left'
                    )
                else:
                    # Use edge data if available, otherwise create empty dataframe
                    if not pair_edge_df.empty:
                        result_df = pair_edge_df.copy()
                        result_df['volatility'] = None
                    else:
                        # Create empty dataframe with required columns
                        result_df = pd.DataFrame(columns=[
                            'timestamp_sg', 'pair_name', 'edge', 'pnl', 
                            'open_collateral', 'time_label', 'volatility'
                        ])
                
                # Store results for this pair
                all_results[pair] = result_df
        
        except Exception as e:
            logger.error(f"Error processing batch {batch_pairs}: {e}")
            # Create empty dataframes for pairs that failed
            for pair in batch_pairs:
                if pair not in all_results:
                    all_results[pair] = pd.DataFrame(columns=[
                        'timestamp_sg', 'pair_name', 'edge', 'pnl', 
                        'open_collateral', 'time_label', 'volatility'
                    ])
    
    return all_results

# Batch fetch market spread data
@st.cache_data(ttl=600)
def batch_fetch_market_spread_data(selected_pairs, start_time_sg, now_sg, batch_size=5):
    # Convert to UTC for database query
    start_time_utc = start_time_sg.astimezone(pytz.utc)
    end_time_utc = now_sg.astimezone(pytz.utc)
    
    all_results = {}
    
    # Process in smaller batches
    for i in range(0, len(selected_pairs), batch_size):
        batch_pairs = selected_pairs[i:i+batch_size]
        pairs_str = "', '".join(batch_pairs)
        
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
        
        pair_matrix AS (
          SELECT 
            ts.slot_sg,
            p.pair_name
          FROM sg_time_slots ts
          CROSS JOIN (SELECT unnest(ARRAY['{pairs_str}']) AS pair_name) p
        ),
        
        spread_data AS (
          SELECT 
            date_trunc('hour', time_group + INTERVAL '8 hour') + 
            INTERVAL '10 min' * FLOOR(EXTRACT(MINUTE FROM time_group + INTERVAL '8 hour')::INT / 10) AS time_slot_sg,
            pair_name,
            AVG(fee1) as avg_spread
          FROM 
            oracle_exchange_fee
          WHERE 
            time_group > NOW() - INTERVAL '1 day'
            AND pair_name IN ('{pairs_str}')
            AND source IN ('binanceFuture', 'gateFuture', 'hyperliquidFuture')
          GROUP BY 
            time_slot_sg, pair_name
        )
        
        SELECT
          pm.slot_sg AS timestamp_sg,
          pm.pair_name,
          COALESCE(s.avg_spread, NULL) AS avg_spread
        FROM
          pair_matrix pm
        LEFT JOIN
          spread_data s ON pm.slot_sg = s.time_slot_sg AND pm.pair_name = s.pair_name
        ORDER BY
          pm.pair_name, pm.slot_sg DESC
        """
        
        try:
            # Execute the query with timeout
            with engine.connect().execution_options(timeout=60) as conn:
                df = pd.read_sql(query, conn)
                
                # Format timestamp and create time_label
                df['timestamp_sg'] = pd.to_datetime(df['timestamp_sg'])
                df['time_label'] = df['timestamp_sg'].dt.strftime('%H:%M')
                
                # Split results by pair
                for pair in batch_pairs:
                    pair_df = df[df['pair_name'] == pair].copy()
                    all_results[pair] = pair_df
                    
        except Exception as e:
            logger.error(f"Error fetching spreads for batch {batch_pairs}: {e}")
            # Create empty dataframes for pairs that failed
            for pair in batch_pairs:
                if pair not in all_results:
                    all_results[pair] = pd.DataFrame(columns=[
                        'timestamp_sg', 'pair_name', 'avg_spread', 'time_label'
                    ])
    
    return all_results

# Function to create edge matrix with pairs as columns
def create_transposed_edge_matrix(pair_data, time_labels, date_labels, selected_pairs):
    # Create a DataFrame with time slots as rows and pairs as columns
    matrix_data = {
        'time_slot': time_labels,
        'date': date_labels
    }
    
    # Add data for each pair
    for pair in selected_pairs:
        if pair in pair_data:
            pair_df = pair_data[pair]
            
            # Create a series with edge values indexed by time_label
            if not pair_df.empty:
                edge_by_time = pd.Series(
                    pair_df['edge'].values,
                    index=pair_df['time_label']
                )
                
                # Add this pair's data to the matrix
                matrix_data[pair] = [edge_by_time.get(time, None) for time in time_labels]
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
    avail_columns = [col for col in reordered_columns if col in df.columns]
    
    # Return reordered DataFrame
    return df[avail_columns]

# Function to create volatility matrix with pairs as columns
def create_transposed_volatility_matrix(pair_data, time_labels, date_labels, selected_pairs):
    # Similar structure as edge matrix
    matrix_data = {
        'time_slot': time_labels,
        'date': date_labels
    }
    
    for pair in selected_pairs:
        if pair in pair_data:
            pair_df = pair_data[pair]
            
            # Create a series with volatility values indexed by time_label
            if not pair_df.empty and 'volatility' in pair_df.columns:
                # Handle case where volatility column exists but might have invalid values
                vol_values = []
                time_index = []
                
                # Process each row to ensure we have valid volatility values
                for idx, row in pair_df.iterrows():
                    try:
                        time_label = row['time_label']
                        vol_value = row['volatility']
                        
                        # Ensure volatility value is a valid number
                        if vol_value is not None and not pd.isna(vol_value):
                            try:
                                # Try to convert to float to verify it's a number
                                vol_float = float(vol_value)
                                vol_values.append(vol_float)
                                time_index.append(time_label)
                            except (ValueError, TypeError):
                                # Log but skip invalid values
                                logger.warning(f"Invalid volatility value for {pair} at {time_label}: {vol_value}")
                    except Exception as e:
                        logger.error(f"Error processing volatility for {pair}: {e}")
                
                # Create Series with valid values only
                vol_by_time = pd.Series(vol_values, index=time_index)
                
                # Add this pair's data to the matrix
                matrix_data[pair] = [vol_by_time.get(time, None) for time in time_labels]
            else:
                # No data for this pair or missing volatility column
                logger.warning(f"Missing volatility data for {pair}")
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
    avail_columns = [col for col in reordered_columns if col in df.columns]
    
    # Return reordered DataFrame
    return df[avail_columns]

# Function to create spread matrix with pairs as columns
def create_transposed_spread_matrix(spread_data, time_labels, date_labels, selected_pairs):
    # Create a DataFrame with time slots as rows and pairs as columns
    matrix_data = {
        'time_slot': time_labels,
        'date': date_labels
    }
    
    # Add data for each pair
    for pair in selected_pairs:
        if pair in spread_data:
            pair_df = spread_data[pair]
            
            # Create a series with spread values indexed by time_label
            if not pair_df.empty:
                spread_by_time = pd.Series(
                    pair_df['avg_spread'].values,
                    index=pair_df['time_label']
                )
                
                # Add this pair's data to the matrix
                matrix_data[pair] = [spread_by_time.get(time, None) for time in time_labels]
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
    avail_columns = [col for col in reordered_columns if col in df.columns]
    
    # Return reordered DataFrame
    return df[avail_columns]

# Function to display matrix with custom formatting
def display_matrix(df, format_func, height=600):
    # Create a DataFrame with formatted values
    formatted_df = df.copy()
    
    # Process each numeric column
    for col in formatted_df.columns:
        if col not in ['time_slot', 'date']:
            # Safer approach to formatting that handles potential non-numeric values
            def safe_format(x):
                try:
                    if pd.isna(x) or x is None:
                        return ""
                    return format_func(x)
                except (TypeError, ValueError):
                    return str(x)
            
            formatted_df[col] = formatted_df[col].apply(safe_format)
    
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

status_text.text("Fetching edge and volatility data...")
# Batch process all selected pairs (smaller batches to avoid timeout)
with st.spinner('Processing data...'):
    batch_size = 3  # Smaller batch size to reduce load
    pair_data = batch_fetch_edge_volatility(selected_pairs, start_time_sg, now_sg, batch_size)
    progress_bar.progress(0.5)
    
    status_text.text("Fetching market spread data...")
    spread_data = batch_fetch_market_spread_data(selected_pairs, start_time_sg, now_sg, batch_size)
    progress_bar.progress(1.0)
    
    status_text.text(f"Processed {len(pair_data)}/{len(selected_pairs)} pairs")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Edge", "Volatility", "Spreads"])

if pair_data:
    # Tab 1: Edge Matrix
    with tab1:
        st.markdown("## Edge Matrix (10min timeframe, Last 24 hours, Singapore Time)")
        st.markdown("### Edge = (Trading PNL + Taker Fee + Funding PNL + SL Fee - Rebate Amount) / Open Collateral")
        
        # Create edge matrix with pairs as columns
        edge_df = create_transposed_edge_matrix(pair_data, time_labels, date_labels, selected_pairs)
        
        if not edge_df.empty and len(edge_df.columns) > 2:  # Check if we have actual pair data columns
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
        st.markdown("### Annualized Volatility based on 2-minute price intervals")
        
        # Create volatility matrix with pairs as columns
        vol_df = create_transposed_volatility_matrix(pair_data, time_labels, date_labels, selected_pairs)
        
        if not vol_df.empty and len(vol_df.columns) > 2:  # Check if we have actual pair data columns
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
            **Notes on Improved Volatility Calculation:**
            - Uses 2-minute price intervals within each 10-minute window
            - Calculates standard deviation of price returns (not log returns)
            - Annualizes based on the actual number of observed intervals
            - Empty cells indicate insufficient price data for calculation (need at least 3 price points)
            """)
        else:
            st.warning("No volatility data available for selected pairs.")
    
    # Tab 3: Spreads Matrix
    with tab3:
        st.markdown("## Market Spreads Matrix (10min timeframe, Last 24 hours, Singapore Time)")
        st.markdown("### Spreads shown in basis points (1bp = 0.01%)")
        
        # Check if we have spread data
        if spread_data:
            # Create market spreads matrix
            spread_df = create_transposed_spread_matrix(spread_data, time_labels, date_labels, selected_pairs)
            
            if not spread_df.empty and len(spread_df.columns) > 2:  # Check if we have actual pair data columns
                # Format function for spread values
                def format_spread(x):
                    return f"{x*10000:.2f}"
                
                # Display the matrix with spread formatting
                display_matrix(spread_df, format_spread)
                
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

# Add metrics display
if st.checkbox("Show Pair Performance Metrics"):
    st.subheader("Pair Performance Metrics (Last 24 Hours)")
    
    # Compute metrics for each pair
    metrics_data = []
    
    for pair in selected_pairs:
        if pair in pair_data and not pair_data[pair].empty:
            df = pair_data[pair]
            
            # Calculate metrics
            avg_edge = df['edge'].mean() if 'edge' in df.columns else None
            avg_vol = df['volatility'].mean() if 'volatility' in df.columns else None
            
            # Calculate edge/vol ratio (risk-adjusted return)
            edge_vol_ratio = None
            if avg_edge is not None and avg_vol is not None and avg_vol > 0:
                edge_vol_ratio = avg_edge / avg_vol
            
            # Get average spread if available
            avg_spread = None
            if pair in spread_data and not spread_data[pair].empty:
                avg_spread = spread_data[pair]['avg_spread'].mean()
            
            # Add to metrics data
            metrics_data.append({
                'Pair': pair,
                'Avg Edge': f"{avg_edge*100:.2f}%" if avg_edge is not None else "N/A",
                'Avg Volatility': f"{avg_vol*100:.2f}%" if avg_vol is not None else "N/A",
                'Edge/Vol Ratio': f"{edge_vol_ratio:.4f}" if edge_vol_ratio is not None else "N/A",
                'Avg Spread (bps)': f"{avg_spread*10000:.2f}" if avg_spread is not None else "N/A"
            })
    
    # Display metrics table
    if metrics_data:
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)
    else:
        st.warning("No metrics data available.")

# Add execution time tracker
st.sidebar.markdown("### Performance Info")
st.sidebar.write(f"Processed {len(selected_pairs)} pairs")
st.sidebar.write(f"Data cached for 10 minutes")