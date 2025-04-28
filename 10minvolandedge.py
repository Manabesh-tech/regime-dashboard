import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import psycopg2
import pytz

st.set_page_config(
    page_title="10-Minute System Edge & Volatility Matrix",
    page_icon="📊",
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
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("10-Minute System Edge & Volatility Matrix")
st.subheader("All Trading Pairs - Last 24 Hours (Singapore Time)")

# Define parameters for the 10-minute timeframe
timeframe = "10min"
lookback_days = 1  # 24 hours
rolling_window = 10  # Window size for volatility calculation
expected_points = 144  # Expected data points per pair over 24 hours (144 10-min windows)
singapore_timezone = pytz.timezone('Asia/Singapore')

# Get current time in Singapore timezone
now_utc = datetime.now(pytz.utc)
now_sg = now_utc.astimezone(singapore_timezone)
st.write(f"Current Singapore Time: {now_sg.strftime('%Y-%m-%d %H:%M:%S')}")

# Set extreme volatility threshold and edge thresholds
extreme_vol_threshold = 1.0  # 100% annualized volatility
high_edge_threshold = 0.5    # 50% house edge
negative_edge_threshold = -0.2  # -20% house edge (system losing)

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
        # Query for Surf data (source_type = 0)
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
@st.cache_data(show_spinner="Fetching tokens...")
def fetch_all_tokens():
    # Calculate time range for the last 24 hours
    end_time = now_sg.strftime("%Y-%m-%d %H:%M:%S")
    start_time = (now_sg - timedelta(days=lookback_days)).strftime("%Y-%m-%d %H:%M:%S")
    
    # Get partition tables for this period
    partition_tables = get_partition_tables(conn, start_time, end_time)
    
    if not partition_tables:
        st.error("No partition tables found for the last 24 hours.")
        return []
    
    # Get distinct tokens from the first partition table
    cursor = conn.cursor()
    try:
        cursor.execute(f"""
        SELECT DISTINCT pair_name 
        FROM public.{partition_tables[0]}
        WHERE source_type = 0
        ORDER BY pair_name
        """)
        tokens = [row[0] for row in cursor.fetchall()]
        return tokens
    except Exception as e:
        st.error(f"Error fetching tokens: {e}")
        return ["BTC", "ETH", "SOL", "DOGE", "PEPE", "AI16Z"]  # Default fallback
    finally:
        cursor.close()

# Function to fetch 10-minute house edge data
@st.cache_data(ttl=600, show_spinner="Fetching house edge data...")
def fetch_house_edge_data():
    # Calculate time range for the query
    end_time = now_sg
    start_time = end_time - timedelta(days=lookback_days)
    
    # UTC timestamps for database query
    start_time_utc = start_time - timedelta(hours=8)  # Convert SG to UTC
    start_time_utc_str = start_time_utc.strftime("%Y-%m-%d %H:%M:%S.000 +00:00")
    
    # Build the SQL query for 10-minute house edge
    query = f"""
    SELECT
      "source"."pair_name" AS "pair_name",
      "source"."UTC+8" AS "timestamp",
      "source"."10min Platform PNL" AS "platform_pnl",
      "source"."house_edge_mapping" AS "house_edge"
    FROM
      (
        SELECT
          "source"."pair_name" AS "pair_name",
          DATE_TRUNC('10 minutes', CAST("source"."UTC+8" AS timestamp)) AS "UTC+8",
          SUM(-1 * "source"."1") + SUM(-1 * "source"."2") + SUM(-1 * "source"."3") AS "10min Platform PNL",
          CASE
            WHEN (CAST(SUM(-1 * "source"."1") + SUM(-1 * "source"."2") + SUM(-1 * "source"."3") AS float) / 
                  NULLIF(SUM("10min_margin"."margin_amount"), 0)) > 1 THEN 1
            WHEN (CAST(SUM(-1 * "source"."1") + SUM(-1 * "source"."2") + SUM(-1 * "source"."3") AS float) / 
                  NULLIF(SUM("10min_margin"."margin_amount"), 0)) < -1 THEN -1
            ELSE (CAST(SUM(-1 * "source"."1") + SUM(-1 * "source"."2") + SUM(-1 * "source"."3") AS float) / 
                  NULLIF(SUM("10min_margin"."margin_amount"), 0))
          END AS "house_edge_mapping"
        FROM
          (
            SELECT
              "source"."UTC+8" AS "UTC+8",
              "source"."pair_name" AS "pair_name",
              "source"."每分钟平台订单盈亏" AS "每分钟平台订单盈亏",
              COALESCE("source"."每分钟平台订单盈亏", 0) AS "1",
              COALESCE("每分钟平台资金费用盈亏 - UTC+8: 分"."每分钟平台资金费用盈亏", 0) AS "2",
              COALESCE("每分钟平台返佣支出 - UTC+8: 分"."每分钟平台返佣支出", 0) AS "3"
            FROM
              (
                SELECT
                  DATE_TRUNC('minute', CAST("source"."UTC+8" AS timestamp)) AS "UTC+8",
                  "source"."pair_name" AS "pair_name",
                  SUM("source"."taker_pnl" * "source"."collateral_price") AS "每分钟平台订单盈亏"
                FROM
                  (
                    SELECT
                      "public"."surfv2_trade"."pair_name" AS "pair_name",
                      "public"."surfv2_trade"."taker_way" AS "taker_way",
                      "public"."surfv2_trade"."collateral_price" AS "collateral_price",
                      "public"."surfv2_trade"."taker_fee_mode" AS "taker_fee_mode",
                      "public"."surfv2_trade"."taker_pnl" AS "taker_pnl",
                      "public"."surfv2_trade"."created_at" AS "created_at",
                      ("public"."surfv2_trade"."created_at" + INTERVAL '8 hour') AS "UTC+8"
                    FROM
                      "public"."surfv2_trade"
                  ) AS "source"
                WHERE
                  ("source"."taker_fee_mode" = 2)
                  AND (
                    ("source"."taker_way" = 1)
                    OR ("source"."taker_way" = 2)
                    OR ("source"."taker_way" = 3)
                    OR ("source"."taker_way" = 4)
                  )
                  AND (
                    "source"."UTC+8" >= timestamp with time zone '{start_time_utc_str}'
                  )
                GROUP BY
                  DATE_TRUNC('minute', CAST("source"."UTC+8" AS timestamp)),
                  "source"."pair_name"
                ORDER BY
                  DATE_TRUNC('minute', CAST("source"."UTC+8" AS timestamp)) ASC,
                  "source"."pair_name" ASC
              ) AS "source"
              LEFT JOIN (
                SELECT
                  DATE_TRUNC('minute', CAST("source"."UTC+8" AS timestamp)) AS "UTC+8",
                  "source"."pair_name" AS "pair_name",
                  SUM(
                    "source"."funding_fee" * "source"."collateral_price"
                  ) AS "每分钟平台资金费用盈亏"
                FROM
                  (
                    SELECT
                      "public"."surfv2_trade"."pair_name" AS "pair_name",
                      "public"."surfv2_trade"."taker_way" AS "taker_way",
                      "public"."surfv2_trade"."collateral_price" AS "collateral_price",
                      "public"."surfv2_trade"."taker_fee_mode" AS "taker_fee_mode",
                      "public"."surfv2_trade"."funding_fee" AS "funding_fee",
                      "public"."surfv2_trade"."created_at" AS "created_at",
                      ("public"."surfv2_trade"."created_at" + INTERVAL '8 hour') AS "UTC+8"
                    FROM
                      "public"."surfv2_trade"
                  ) AS "source"
                WHERE
                  ("source"."taker_fee_mode" = 2)
                  AND ("source"."taker_way" = 0)
                  AND (
                    "source"."UTC+8" >= timestamp with time zone '{start_time_utc_str}'
                  )
                GROUP BY
                  DATE_TRUNC('minute', CAST("source"."UTC+8" AS timestamp)),
                  "source"."pair_name"
                ORDER BY
                  DATE_TRUNC('minute', CAST("source"."UTC+8" AS timestamp)) ASC,
                  "source"."pair_name" ASC
              ) AS "每分钟平台资金费用盈亏 - UTC+8: 分" ON (
                DATE_TRUNC('minute', CAST("source"."UTC+8" AS timestamp)) = DATE_TRUNC(
                  'minute',
                  CAST("每分钟平台资金费用盈亏 - UTC+8: 分"."UTC+8" AS timestamp)
                )
              )
              AND (
                "source"."pair_name" = "每分钟平台资金费用盈亏 - UTC+8: 分"."pair_name"
              )
              LEFT JOIN (
                SELECT
                  DATE_TRUNC('minute', CAST("source"."UTC+8" AS timestamp)) AS "UTC+8",
                  "source"."pair_name" AS "pair_name",
                  SUM("source"."rebate" * "source"."collateral_price") AS "每分钟平台返佣支出"
                FROM
                  (
                    SELECT
                      "public"."surfv2_trade"."pair_name" AS "pair_name",
                      "public"."surfv2_trade"."collateral_price" AS "collateral_price",
                      "public"."surfv2_trade"."taker_fee_mode" AS "taker_fee_mode",
                      "public"."surfv2_trade"."rebate" AS "rebate",
                      "public"."surfv2_trade"."created_at" AS "created_at",
                      ("public"."surfv2_trade"."created_at" + INTERVAL '8 hour') AS "UTC+8"
                    FROM
                      "public"."surfv2_trade"
                  ) AS "source"
                WHERE
                  ("source"."taker_fee_mode" = 2)
                  AND (
                    "source"."UTC+8" >= timestamp with time zone '{start_time_utc_str}'
                  )
                GROUP BY
                  DATE_TRUNC('minute', CAST("source"."UTC+8" AS timestamp)),
                  "source"."pair_name"
                ORDER BY
                  DATE_TRUNC('minute', CAST("source"."UTC+8" AS timestamp)) ASC,
                  "source"."pair_name" ASC
              ) AS "每分钟平台返佣支出 - UTC+8: 分" ON (
                DATE_TRUNC('minute', CAST("source"."UTC+8" AS timestamp)) = DATE_TRUNC(
                  'minute',
                  CAST("每分钟平台返佣支出 - UTC+8: 分"."UTC+8" AS timestamp)
                )
              )
              AND (
                "source"."pair_name" = "每分钟平台返佣支出 - UTC+8: 分"."pair_name"
              )
          ) AS "source"
          LEFT JOIN (
            -- This subquery calculates opening margin amounts in 10-minute blocks
            SELECT
              DATE_TRUNC('10 minutes', CAST("source"."UTC+8" AS timestamp)) AS "UTC+8",
              "source"."pair_name" AS "pair_name",
              SUM("source"."deal_vol" * "source"."collateral_price") AS "margin_amount"
            FROM
              (
                SELECT
                  "public"."surfv2_trade"."pair_name" AS "pair_name",
                  "public"."surfv2_trade"."deal_vol" AS "deal_vol",
                  "public"."surfv2_trade"."taker_way" AS "taker_way",
                  "public"."surfv2_trade"."collateral_price" AS "collateral_price",
                  "public"."surfv2_trade"."taker_fee_mode" AS "taker_fee_mode",
                  "public"."surfv2_trade"."created_at" AS "created_at",
                  ("public"."surfv2_trade"."created_at" + INTERVAL '8 hour') AS "UTC+8"
                FROM
                  "public"."surfv2_trade"
              ) AS "source"
            WHERE
              ("source"."UTC+8" >= timestamp with time zone '{start_time_utc_str}')
              AND ("source"."taker_fee_mode" = 2)
              AND (
                ("source"."taker_way" = 1)
                OR ("source"."taker_way" = 3)
              )
            GROUP BY
              DATE_TRUNC('10 minutes', CAST("source"."UTC+8" AS timestamp)),
              "source"."pair_name"
            ORDER BY
              DATE_TRUNC('10 minutes', CAST("source"."UTC+8" AS timestamp)) ASC,
              "source"."pair_name" ASC
          ) AS "10min_margin" ON (
            "source"."pair_name" = "10min_margin"."pair_name"
          )
          AND (
            DATE_TRUNC('10 minutes', CAST("source"."UTC+8" AS timestamp)) = 
            DATE_TRUNC('10 minutes', CAST("10min_margin"."UTC+8" AS timestamp))
          )
        GROUP BY
          "source"."pair_name",
          DATE_TRUNC('10 minutes', CAST("source"."UTC+8" AS timestamp))
        ORDER BY
          "source"."pair_name" ASC,
          DATE_TRUNC('10 minutes', CAST("source"."UTC+8" AS timestamp)) ASC
      ) AS "source"
    WHERE
      "source"."UTC+8" >= (CURRENT_TIMESTAMP + INTERVAL '8 hour' - INTERVAL '24 hours')
    ORDER BY 
      "source"."pair_name" ASC,
      "source"."UTC+8" ASC
    """
    
    try:
        print("Executing house edge query...")
        edge_df = pd.read_sql_query(query, conn)
        print(f"House edge query completed. DataFrame shape: {edge_df.shape}")
        
        if edge_df.empty:
            print("No house edge data found.")
            return pd.DataFrame()
        
        # Convert timestamp to datetime
        edge_df['timestamp'] = pd.to_datetime(edge_df['timestamp'])
        
        # Format the time label (HH:MM) for later joining
        edge_df['time_label'] = edge_df['timestamp'].dt.strftime('%H:%M')
        
        return edge_df
        
    except Exception as e:
        st.error(f"Error fetching house edge data: {e}")
        print(f"Error fetching house edge data: {e}")
        return pd.DataFrame()

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
        
        # Realized volatility - adjusted for 10-minute timeframe (144 periods in a day)
        realized_vol = np.std(log_returns) * np.sqrt(252 * 144)  # Annualized volatility (10min bars)
        
        return {
            'realized_vol': realized_vol,
            'parkinson_vol': np.nan,  # Placeholder 
            'gk_vol': np.nan,         # Placeholder
            'rs_vol': np.nan          # Placeholder
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

# House edge classification function
def classify_house_edge(edge):
    if pd.isna(edge):
        return ("UNKNOWN", 0, "Insufficient data")
    elif edge < negative_edge_threshold:  # System is losing money
        return ("NEGATIVE", 1, "System losing")
    elif edge < 0.1:  # Low edge
        return ("LOW", 2, "Low edge")
    elif edge < 0.3:  # Medium edge
        return ("MEDIUM", 3, "Medium edge")
    elif edge < high_edge_threshold:  # Good edge
        return ("GOOD", 4, "Good edge")
    else:  # High edge
        return ("HIGH", 5, "High edge")

# Function to generate aligned 10-minute time blocks for the past 24 hours
def generate_aligned_time_blocks(current_time):
    """
    Generate fixed 10-minute time blocks for past 24 hours,
    aligned with standard 10-minute intervals
    """
    # Round down to the nearest 10-minute mark
    minutes = current_time.minute
    rounded_minutes = (minutes // 10) * 10
    latest_complete_block_end = current_time.replace(minute=rounded_minutes, second=0, microsecond=0)
    
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

# Fetch and calculate volatility for a token with 10min timeframe
@st.cache_data(ttl=600, show_spinner="Calculating volatility metrics...")
def fetch_and_calculate_volatility(token):
    # Get current time in Singapore timezone
    now_utc = datetime.now(pytz.utc)
    now_sg = now_utc.astimezone(singapore_timezone)
    start_time_sg = now_sg - timedelta(days=lookback_days)
    
    # Convert for database query (keep as Singapore time strings as the query will handle timezone)
    start_time = start_time_sg.strftime("%Y-%m-%d %H:%M:%S")
    end_time = now_sg.strftime("%Y-%m-%d %H:%M:%S")

    # Get relevant partition tables
    partition_tables = get_partition_tables(conn, start_time_sg, now_sg)
    
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

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        
        # Create 1-minute OHLC data
        one_min_ohlc = df['final_price'].resample('1min').ohlc().dropna()
        if one_min_ohlc.empty:
            print(f"[{token}] No OHLC data after resampling.")
            return None
            
        # Calculate rolling volatility on 1-minute data
        one_min_ohlc['realized_vol'] = one_min_ohlc['close'].rolling(window=rolling_window).apply(
            lambda x: calculate_volatility_metrics(x)['realized_vol']
        )
        
        # Resample to exactly 10min intervals aligned with clock
        ten_min_vol = one_min_ohlc['realized_vol'].resample('10min', closed='left', label='left').mean().dropna()
        
        if ten_min_vol.empty:
            print(f"[{token}] No 10-min volatility data.")
            return None
            
        # Get last 24 hours (144 10-minute bars)
        last_24h_vol = ten_min_vol.tail(144)  # Get up to last 144 periods (24 hours)
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
        st.error(f"Error processing {token}: {e}")
        print(f"[{token}] Error processing: {e}")
        return None

# Function to combine volatility and house edge data
def combine_volatility_and_edge(volatility_results, house_edge_data):
    """
    Combine volatility and house edge data into a single DataFrame for display.
    Returns a dictionary with tokens as keys and combined dataframes as values.
    """
    combined_results = {}
    
    for token in volatility_results.keys():
        # Get volatility data for this token
        vol_df = volatility_results[token]
        
        # Get house edge data for this token
        edge_df = house_edge_data[house_edge_data['pair_name'] == token].copy()
        
        if not edge_df.empty and not vol_df.empty:
            # Format timestamps for joining
            edge_df['time_label'] = edge_df['timestamp'].dt.strftime('%H:%M')
            
            # Merge on time_label
            merged = pd.merge(
                vol_df.reset_index(), 
                edge_df[['time_label', 'house_edge', 'platform_pnl']], 
                on='time_label', 
                how='outer'
            )
            
            # Classify house edge
            merged['edge_info'] = merged['house_edge'].apply(classify_house_edge)
            merged['edge_regime'] = merged['edge_info'].apply(lambda x: x[0] if x else "UNKNOWN")
            merged['edge_desc'] = merged['edge_info'].apply(lambda x: x[2] if x else "Insufficient data")
            
            # Set index back to timestamp
            if 'timestamp_x' in merged.columns:
                merged.set_index('timestamp_x', inplace=True)
            elif 'index' in merged.columns:
                merged.set_index('index', inplace=True)
            
            combined_results[token] = merged
            
    return combined_results

# UI Controls
col1, col2 = st.columns([3, 1])

with col1:
    # Let user select tokens to display (or select all)
    all_tokens = fetch_all_tokens()
    select_all = st.checkbox("Select All Tokens", value=True)
    
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
    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.experimental_rerun()

if not selected_tokens:
    st.warning("Please select at least one token")
    st.stop()

# Show the blocks we're analyzing
with st.expander("View Time Blocks Being Analyzed"):
    time_blocks_df = pd.DataFrame([(b[0].strftime('%Y-%m-%d %H:%M'), b[1].strftime('%Y-%m-%d %H:%M'), b[2]) 
                                  for b in aligned_time_blocks], 
                                 columns=['Start Time', 'End Time', 'Block Label'])
    st.dataframe(time_blocks_df)

# Show progress bar while calculating
progress_bar = st.progress(0)
status_text = st.empty()

# Calculate volatility for each token
token_volatility_results = {}
for i, token in enumerate(selected_tokens):
    try:
        progress_bar.progress((i) / len(selected_tokens))
        status_text.text(f"Processing volatility for {token} ({i+1}/{len(selected_tokens)})")
        result = fetch_and_calculate_volatility(token)
        if result is not None:
            token_volatility_results[token] = result
    except Exception as e:
        st.error(f"Error processing token {token} volatility: {e}")
        print(f"Error processing token {token} volatility in main loop: {e}")

# Fetch house edge data
status_text.text("Fetching house edge data...")
house_edge_data = fetch_house_edge_data()

# Combine volatility and house edge data
status_text.text("Combining volatility and house edge data...")
combined_results = combine_volatility_and_edge(token_volatility_results, house_edge_data)

# Final progress update
progress_bar.progress(1.0)
status_text.text(f"Processed {len(combined_results)}/{len(selected_tokens)} tokens successfully")

# Create table for display
if combined_results:
    # Create table data for house edge and volatility together
    table_data = {}
    
    for token, df in combined_results.items():
        # Create a series with both volatility and house edge values
        # Format: "vol% / edge%"
        combined_series = pd.Series(
            [f"{(v*100):.1f}% / {(e*100):.1f}%" if not pd.isna(v) and not pd.isna(e) else "N/A" 
             for v, e in zip(df['realized_vol'], df['house_edge'])],
            index=df['time_label']
        )
        table_data[token] = combined_series
    
    # Create DataFrame with all tokens
    combined_table = pd.DataFrame(table_data)
    
    # Apply the time blocks in the proper order (most recent first)
    available_times = set(combined_table.index)
    ordered_times = [t for t in time_block_labels if t in available_times]
    
    # If no matches are found in aligned blocks, fallback to the available times
    if not ordered_times and available_times:
        ordered_times = sorted(list(available_times), reverse=True)
    
    # Reindex with the ordered times
    combined_table = combined_table.reindex(ordered_times)
    
    # Create a separate table for visualization that shows numeric values
    numeric_data = {}
    for token, df in combined_results.items():
        # Create dataframes with the numeric values for coloring
        vol_series = pd.Series(df['realized_vol'], index=df['time_label'])
        edge_series = pd.Series(df['house_edge'], index=df['time_label'])
        numeric_data[token] = {'vol': vol_series, 'edge': edge_series}
    
    # Function to color cells for combined data
    def color_combined_cells(val, token, time_label):
        if val == "N/A":
            return 'background-color: #f5f5f5; color: #666666;'  # Grey for missing
        
        # Get volatility and house edge values
        try:
            vol = numeric_data[token]['vol'].loc[time_label]
            edge = numeric_data[token]['edge'].loc[time_label]
        except:
            return 'background-color: #f5f5f5; color: #666666;'  # Grey if key error
        
        if pd.isna(vol) or pd.isna(edge):
            return 'background-color: #f5f5f5; color: #666666;'  # Grey for missing
        
        # Create colors based on combinations of edge and volatility
        
        # Define edge colors (red to green)
        if edge < negative_edge_threshold:  # Negative edge (system losing)
            edge_color = "255,100,100"  # Red
        elif edge < 0.1:  # Low edge
            edge_color = "255,200,100"  # Orange yellow
        elif edge < 0.3:  # Medium edge
            edge_color = "200,255,100"  # Yellow green
        elif edge < high_edge_threshold:  # Good edge
            edge_color = "150,255,150"  # Light green
        else:  # High edge
            edge_color = "100,255,100"  # Green
            
        # Define volatility intensity (transparency)
        if vol < 0.3:  # Low volatility
            vol_alpha = 0.4
            text_color = "black"
        elif vol < 0.6:  # Medium volatility
            vol_alpha = 0.6
            text_color = "black"
        elif vol < 1.0:  # High volatility
            vol_alpha = 0.8
            text_color = "black"
        else:  # Extreme volatility
            vol_alpha = 1.0
            text_color = "white"
            
        return f'background-color: rgba({edge_color}, {vol_alpha}); color: {text_color}'
    
    # Apply styling for the combined table
    def style_combined_table(df):
        styles = []
        for token in df.columns:
            for time_label in df.index:
                val = df.loc[time_label, token]
                style = color_combined_cells(val, token, time_label)
                styles.append({'selector': f'.data-row-{df.index.get_loc(time_label)} .data-col-{df.columns.get_loc(token)}', 'props': [('background-color', style)]})
        return styles
    
    # Convert the combined_table to a styled version
    styled_table = combined_table.style.apply(
        lambda x: pd.Series([color_combined_cells(val, x.name, idx) for idx, val in x.items()], index=x.index),
        axis=1
    )
    
    # Display the matrix
    st.markdown("## Combined 10-Minute Edge and Volatility Matrix")
    st.markdown("### Values shown as: Volatility % / House Edge %")
    st.markdown("""
    #### Color Legend: 
    - **Background Color**: House Edge (Red = Negative, Yellow = Low, Green = High)
    - **Color Intensity**: Volatility (Darker = Higher Volatility)
    """)
    st.dataframe(styled_table, height=700, use_container_width=True)
    
    # Create a summary dashboard
    st.subheader("24-Hour Summary Dashboard")
    
    # Summary metrics
    summary_data = []
    for token, df in combined_results.items():
        if not df.empty:
            avg_vol = df['realized_vol'].mean() if 'realized_vol' in df.columns else np.nan
            avg_edge = df['house_edge'].mean() if 'house_edge' in df.columns else np.nan
            total_pnl = df['platform_pnl'].sum() if 'platform_pnl' in df.columns else np.nan
            
            # Get peak volatility time
            max_vol_idx = df['realized_vol'].idxmax() if 'realized_vol' in df.columns and not df['realized_vol'].isna().all() else None
            max_vol_time = df.loc[max_vol_idx, 'time_label'] if max_vol_idx and 'time_label' in df.columns else "N/A"
            
            # Get best edge time
            max_edge_idx = df['house_edge'].idxmax() if 'house_edge' in df.columns and not df['house_edge'].isna().all() else None
            max_edge_time = df.loc[max_edge_idx, 'time_label'] if max_edge_idx and 'time_label' in df.columns else "N/A"
            
            summary_data.append({
                'Token': token,
                'Avg Vol (%)': (avg_vol * 100).round(1) if not pd.isna(avg_vol) else np.nan,
                'Avg Edge (%)': (avg_edge * 100).round(1) if not pd.isna(avg_edge) else np.nan,
                'Total PNL': int(total_pnl) if not pd.isna(total_pnl) else np.nan,
                'Peak Vol Time': max_vol_time,
                'Best Edge Time': max_edge_time
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        
        # Sort by average edge (high to low)
        summary_df = summary_df.sort_values(by='Avg Edge (%)', ascending=False)
        
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
        
        def color_edge_column(val):
            if pd.isna(val):
                return ''
            elif val < negative_edge_threshold * 100:
                return 'color: red'
            elif val < 10:
                return 'color: orange'
            elif val < 30:
                return 'color: #aaaa00'
            elif val < high_edge_threshold * 100:
                return 'color: lightgreen'
            else:
                return 'color: green; font-weight: bold'
        
        def color_pnl_column(val):
            if pd.isna(val):
                return ''
            elif val < 0:
                return 'color: red'
            else:
                return 'color: green'
        
        # Apply styling
        styled_summary = summary_df.style\
            .applymap(color_vol_column, subset=['Avg Vol (%)'])\
            .applymap(color_edge_column, subset=['Avg Edge (%)'])\
            .applymap(color_pnl_column, subset=['Total PNL'])
        
        # Display the styled dataframe
        st.dataframe(styled_summary, height=500, use_container_width=True)
        
        # Create summary metrics at the top
        total_pnl = sum(row['Total PNL'] for row in summary_data if not pd.isna(row['Total PNL']))
        
        # Calculate weighted average edge if we have valid data
        valid_data = [(row['Avg Edge (%)'], row['Total PNL']) for row in summary_data 
                     if not pd.isna(row['Avg Edge (%)']) and not pd.isna(row['Total PNL']) and row['Total PNL'] != 0]
        
        if valid_data:
            total_abs_pnl = sum(abs(pnl) for _, pnl in valid_data)
            if total_abs_pnl > 0:
                avg_portfolio_edge = sum(edge * abs(pnl) for edge, pnl in valid_data) / total_abs_pnl
            else:
                avg_portfolio_edge = np.nan
        else:
            avg_portfolio_edge = np.nan
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Portfolio PNL (24h)", f"${total_pnl:,.2f}")
        col2.metric("Weighted Avg Edge", f"{avg_portfolio_edge:.2f}%" if not pd.isna(avg_portfolio_edge) else "N/A")
        col3.metric("Tokens Analyzed", f"{len(summary_data)}")
    else:
        st.warning("No summary data available.")
    
    # Identify and display opportunities
    st.subheader("Trading Opportunities")
    
    # 1. High Edge with Low Volatility (Good Risk/Reward)
    high_edge_tokens = [row for row in summary_data 
                        if not pd.isna(row['Avg Edge (%)']) 
                        and not pd.isna(row['Avg Vol (%)'])
                        and row['Avg Edge (%)'] > 20
                        and row['Avg Vol (%)'] < 60]
    
    # 2. Extreme volatility events
    extreme_vol_events = []
    for token, df in combined_results.items():
        if not df.empty and 'realized_vol' in df.columns and 'house_edge' in df.columns:
            extreme_periods = df[df['realized_vol'] >= extreme_vol_threshold]
            for idx, row in extreme_periods.iterrows():
                vol_value = float(row['realized_vol']) if not pd.isna(row['realized_vol']) else 0.0
                edge_value = float(row['house_edge']) if not pd.isna(row['house_edge']) else 0.0
                time_label = str(row['time_label']) if 'time_label' in row and not pd.isna(row['time_label']) else "Unknown"
                
                extreme_vol_events.append({
                    'Token': token,
                    'Time': time_label,
                    'Volatility (%)': round(vol_value * 100, 1),
                    'Edge (%)': round(edge_value * 100, 1),
                    'Full Timestamp': idx.strftime('%Y-%m-%d %H:%M') if hasattr(idx, 'strftime') else str(idx)
                })
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### High Edge / Low Volatility Pairs")
        if high_edge_tokens:
            for token in high_edge_tokens:
                edge = token['Avg Edge (%)']
                vol = token['Avg Vol (%)']
                edge_color = "green" if edge > 30 else "lightgreen"
                st.markdown(f"- **{token['Token']}**: Edge: <span style='color:{edge_color}'>{edge:.1f}%</span>, Vol: <span style='color:green'>{vol:.1f}%</span>", unsafe_allow_html=True)
        else:
            st.markdown("*No tokens in this category*")
    
    with col2:
        st.markdown("### Extreme Volatility Events")
        if extreme_vol_events:
            for event in extreme_vol_events[:5]:  # Show top 5
                token = event['Token']
                time = event['Time']
                vol = event['Volatility (%)']
                edge = event['Edge (%)']
                edge_color = "red" if edge < 0 else "green"
                st.markdown(f"- **{token}** at **{time}**: Vol: <span style='color:red'>{vol}%</span>, Edge: <span style='color:{edge_color}'>{edge}%</span>", unsafe_allow_html=True)
            
            if len(extreme_vol_events) > 5:
                st.markdown(f"*... and {len(extreme_vol_events) - 5} more extreme events*")
        else:
            st.markdown("*No events in this category*")
    
    # Visualize the relationship between edge and volatility
    st.subheader("Edge vs. Volatility Relationship")
    
    # Prepare data for scatter plot
    scatter_data = []
    for token, df in combined_results.items():
        if not df.empty and 'realized_vol' in df.columns and 'house_edge' in df.columns:
            avg_vol = df['realized_vol'].mean()
            avg_edge = df['house_edge'].mean()
            if not pd.isna(avg_vol) and not pd.isna(avg_edge):
                scatter_data.append({
                    'Token': token,
                    'Volatility': avg_vol * 100,  # Convert to percentage
                    'Edge': avg_edge * 100        # Convert to percentage
                })
    
    if scatter_data:
        scatter_df = pd.DataFrame(scatter_data)
        
        # Create scatter plot
        fig = go.Figure()
        
        # Add scatter points
        fig.add_trace(go.Scatter(
            x=scatter_df['Volatility'],
            y=scatter_df['Edge'],
            mode='markers+text',
            marker=dict(
                size=12,
                color=scatter_df['Volatility'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Volatility (%)"),
                line=dict(width=1, color='black')
            ),
            text=scatter_df['Token'],
            textposition="top center",
            name='Tokens'
        ))
        
        # Add quadrant lines
        fig.add_shape(
            type="line",
            x0=0, x1=max(scatter_df['Volatility']) * 1.1,
            y0=0, y1=0,
            line=dict(color="black", width=1, dash="dash")
        )
        
        fig.add_shape(
            type="line",
            x0=60, x1=60,
            y0=min(scatter_df['Edge']) * 1.1, y1=max(scatter_df['Edge']) * 1.1,
            line=dict(color="black", width=1, dash="dash")
        )
        
        # Add quadrant labels
        fig.add_annotation(
            x=30, y=20,
            text="IDEAL:<br>High Edge, Low Vol",
            showarrow=False,
            font=dict(size=12, color="green")
        )
        
        fig.add_annotation(
            x=80, y=20,
            text="GOOD:<br>High Edge, High Vol",
            showarrow=False,
            font=dict(size=12, color="darkgreen")
        )
        
        fig.add_annotation(
            x=30, y=-20,
            text="POOR:<br>Negative Edge, Low Vol",
            showarrow=False,
            font=dict(size=12, color="red")
        )
        
        fig.add_annotation(
            x=80, y=-20,
            text="AVOID:<br>Negative Edge, High Vol",
            showarrow=False,
            font=dict(size=12, color="darkred")
        )
        
        # Update layout
        fig.update_layout(
            title="Average Edge vs. Volatility by Token (24h)",
            xaxis_title="Volatility (%)",
            yaxis_title="Edge (%)",
            height=600,
            template="plotly_white",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Insufficient data for scatter plot.")

with st.expander("Understanding the Matrix"):
    st.markdown("""
    ### How to Read This Table
    
    The 10-Minute System Edge & Volatility Matrix shows combined data for each token across 10-minute time intervals during the last 24 hours. Each cell contains both volatility and house edge information.
    
    #### Cell Values
    Each cell shows **Volatility % / House Edge %** for that specific token and time period.
    
    #### Color Coding
    - **Background Color** represents the house edge:
        - **Red**: Negative edge (system is losing money)
        - **Orange/Yellow**: Low/medium edge
        - **Green**: High edge (good for the system)
    
    - **Color Intensity** represents volatility:
        - **Lighter shades**: Lower volatility
        - **Darker shades**: Higher volatility
        - **White text**: Extreme volatility (≥100%)
    
    #### Trading Implications
    - **Green cells with light background**: High edge with low volatility - best trading opportunities
    - **Green cells with dark background**: High edge but high volatility - good but riskier
    - **Red cells with dark background**: Negative edge with high volatility - avoid these
    
    #### Technical Details
    - Volatility is calculated as the standard deviation of log returns, annualized based on 10-minute intervals (144 per day)
    - House edge represents the system's profitability, calculated as (platform PNL / total margin) for each 10-minute period
    - Edge values range from -1 to 1, with positive values indicating system profit
    - All times are shown in Singapore timezone
    """)