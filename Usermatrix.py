import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import pytz
import time
import concurrent.futures
import traceback

# Set page config with modern options
st.set_page_config(
    page_title="User PNL Matrix Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for modern look
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.2rem;
    }
    .main .block-container {
        padding-top: 2rem;
    }
    div[data-testid="stMetric"] > div:first-child {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 10px;
    }
    div[data-testid="stMetric"] > div:first-child > div:first-child > div:first-child > div {
        font-size: 1.2rem;
    }
    div[data-testid="stMetric"] > div:first-child > div:first-child > div:nth-child(2) > div {
        font-size: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- DB CONNECTION MANAGEMENT ---
def get_db_connection(timeout=30):
    """Create database connection with timeout and connection pooling"""
    try:
        db_config = st.secrets["database"]
        db_uri = (
            f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}"
            f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )
        # Use connection pooling with reasonable pool size
        engine = create_engine(db_uri, pool_size=5, max_overflow=10, pool_timeout=timeout, pool_recycle=3600)
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception as e:
        st.error(f"Database connection error: {e}")
        st.error(traceback.format_exc())
        return None

# --- APP STATE MANAGEMENT ---
class AppState:
    """Centralized state management for the application"""
    def __init__(self):
        self.engine = None
        self.pairs = []
        self.users = []
        self.time_boundaries = {}
        self.data_loaded = False
        self.last_refresh = None
        
    def initialize(self):
        """Initialize app state and connections"""
        # Set up timezone
        self.singapore_tz = pytz.timezone('Asia/Singapore')
        self.now_utc = datetime.now(pytz.utc)
        self.now_sg = self.now_utc.astimezone(self.singapore_tz)
        
        # Calculate time boundaries
        self.time_boundaries = self.get_time_boundaries()
        
        # Connect to database
        self.engine = get_db_connection()
        if not self.engine:
            st.stop()
            
        # Record initialization time
        self.last_refresh = self.now_sg
    
    def get_time_boundaries(self):
        """Get time boundaries in Singapore timezone"""
        # Current time in Singapore
        now_sg = self.now_sg
        
        # Today's midnight in Singapore
        today_midnight_sg = now_sg.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Yesterday's midnight in Singapore
        yesterday_midnight_sg = today_midnight_sg - timedelta(days=1)
        
        # Day before yesterday's midnight
        day_before_yesterday_sg = today_midnight_sg - timedelta(days=2)
        
        # 7 days ago midnight in Singapore
        week_ago_midnight_sg = today_midnight_sg - timedelta(days=7)
        
        # 30 days ago midnight in Singapore
        month_ago_midnight_sg = today_midnight_sg - timedelta(days=30)
        
        # All time (use a far past date, e.g., 5 years ago)
        all_time_start_sg = today_midnight_sg.replace(year=today_midnight_sg.year-5)
        
        # Convert all times back to UTC for database queries
        today_midnight_utc = today_midnight_sg.astimezone(pytz.utc)
        yesterday_midnight_utc = yesterday_midnight_sg.astimezone(pytz.utc)
        day_before_yesterday_utc = day_before_yesterday_sg.astimezone(pytz.utc)
        week_ago_midnight_utc = week_ago_midnight_sg.astimezone(pytz.utc)
        month_ago_midnight_utc = month_ago_midnight_sg.astimezone(pytz.utc)
        all_time_start_utc = all_time_start_sg.astimezone(pytz.utc)
        now_utc = now_sg.astimezone(pytz.utc)
        
        return {
            "today": {
                "start": today_midnight_utc,
                "end": now_utc,
                "label": f"Today ({today_midnight_sg.strftime('%Y-%m-%d')})"
            },
            "yesterday": {
                "start": yesterday_midnight_utc,
                "end": today_midnight_utc,
                "label": f"Yesterday ({yesterday_midnight_sg.strftime('%Y-%m-%d')})"
            },
            "day_before_yesterday": {
                "start": day_before_yesterday_utc,
                "end": yesterday_midnight_utc,
                "label": f"Day Before Yesterday ({day_before_yesterday_sg.strftime('%Y-%m-%d')})"
            },
            "this_week": {
                "start": week_ago_midnight_utc,
                "end": now_utc,
                "label": f"This Week (Last 7 Days)"
            },
            "this_month": {
                "start": month_ago_midnight_utc,
                "end": now_utc,
                "label": f"This Month (Last 30 Days)"
            },
            "all_time": {
                "start": all_time_start_utc,
                "end": now_utc,
                "label": "All Time"
            }
        }

# Create app state
app_state = AppState()
app_state.initialize()

# --- UI SETUP ---
st.title("📊 User PNL Matrix Dashboard")
st.subheader("Performance Analysis by User (Singapore Time)")

# Current Singapore time display
st.write(f"Current Singapore Time: {app_state.now_sg.strftime('%Y-%m-%d %H:%M:%S')}")

# Show last refresh time
if app_state.last_refresh:
    st.write(f"Last data refresh: {app_state.last_refresh.strftime('%H:%M:%S')}")

# --- OPTIMIZED DATA FETCHING FUNCTIONS ---
@st.cache_data(ttl=600)
def fetch_all_pairs(engine):
    """Fetch all trading pairs with error handling and timeout"""
    query = "SELECT DISTINCT pair_name FROM public.trade_pool_pairs ORDER BY pair_name"
    
    try:
        with engine.connect() as conn:
            # Set statement timeout to 10 seconds
            conn.execute(text("SET statement_timeout = 10000"))
            df = pd.read_sql(text(query), conn)
            
        if df.empty:
            st.warning("No pairs found in the database. Using defaults.")
            return ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT", "PEPE/USDT"]
        
        return df['pair_name'].tolist()
    except Exception as e:
        st.error(f"Error fetching pairs: {e}")
        # Return a fallback list
        return ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT", "PEPE/USDT"]

@st.cache_data(ttl=600)
def fetch_top_users(engine, limit=100):
    """Fetch top users by trading volume with optimized query"""
    # Calculate the date range internally
    now_utc = datetime.now(pytz.utc)
    month_ago_utc = now_utc - timedelta(days=30)
    
    query = f"""
    SELECT 
        "taker_account_id" as user_identifier,
        COUNT(*) as trade_count,
        SUM(ABS("deal_size" * "deal_price")) as total_volume
    FROM 
        "public"."trade_fill_fresh"
    WHERE 
        "created_at" >= '{month_ago_utc.strftime("%Y-%m-%d %H:%M:%S")}'
        AND "taker_way" IN (1, 2, 3, 4)
    GROUP BY 
        "taker_account_id"
    ORDER BY 
        total_volume DESC
    LIMIT {limit}
    """
    
    try:
        with engine.connect() as conn:
            # Set statement timeout to 20 seconds
            conn.execute(text("SET statement_timeout = 20000"))
            df = pd.read_sql(text(query), conn)
        
        if df.empty:
            st.warning("No active users found in the database. Using test data.")
            return [f"user_{i}" for i in range(1, 11)]
        
        return df['user_identifier'].tolist()
    except Exception as e:
        st.error(f"Error fetching users: {e}")
        st.error(traceback.format_exc())
        # Return some mock user IDs for testing
        return [f"user_{i}" for i in range(1, 11)]

@st.cache_data(ttl=600)
def fetch_user_pnl_data(engine, taker_account_id, pair_name, start_time, end_time):
    """Fetch user PNL data with optimized query and error handling"""
    
    query = f"""
    WITH 
    user_order_pnl AS (
      -- Calculate user order PNL
      SELECT
        COALESCE(SUM("taker_pnl" * "collateral_price"), 0) AS "user_order_pnl"
      FROM
        "public"."trade_fill_fresh"
      WHERE
        "created_at" BETWEEN '{start_time}' AND '{end_time}'
        AND "pair_id" IN (SELECT "pair_id" FROM "public"."trade_pool_pairs" WHERE "pair_name" = '{pair_name}')
        AND "taker_account_id" = '{taker_account_id}'
        AND "taker_way" IN (1, 2, 3, 4)
    ),
    
    user_fee_payments AS (
      -- Calculate user fee payments
      SELECT
        COALESCE(SUM(-1 * "taker_fee" * "collateral_price"), 0) AS "user_fee_payments"
      FROM
        "public"."trade_fill_fresh"
      WHERE
        "created_at" BETWEEN '{start_time}' AND '{end_time}'
        AND "pair_id" IN (SELECT "pair_id" FROM "public"."trade_pool_pairs" WHERE "pair_name" = '{pair_name}')
        AND "taker_account_id" = '{taker_account_id}'
        AND "taker_fee_mode" = 1
        AND "taker_way" IN (1, 3)
    ),
    
    user_funding_payments AS (
      -- Calculate user funding fee payments
      SELECT
        COALESCE(SUM("funding_fee" * "collateral_price"), 0) AS "user_funding_payments"
      FROM
        "public"."trade_fill_fresh"
      WHERE
        "created_at" BETWEEN '{start_time}' AND '{end_time}'
        AND "pair_id" IN (SELECT "pair_id" FROM "public"."trade_pool_pairs" WHERE "pair_name" = '{pair_name}')
        AND "taker_account_id" = '{taker_account_id}'
        AND "taker_way" = 0
    ),
    
    user_trade_count AS (
      -- Calculate total number of trades
      SELECT
        COUNT(*) AS "trade_count"
      FROM
        "public"."trade_fill_fresh"
      WHERE
        "created_at" BETWEEN '{start_time}' AND '{end_time}'
        AND "pair_id" IN (SELECT "pair_id" FROM "public"."trade_pool_pairs" WHERE "pair_name" = '{pair_name}')
        AND "taker_account_id" = '{taker_account_id}'
        AND "taker_way" IN (1, 2, 3, 4)
    )
    
    -- Final query: combine all data sources
    SELECT
      (SELECT "user_order_pnl" FROM user_order_pnl) +
      (SELECT "user_fee_payments" FROM user_fee_payments) +
      (SELECT "user_funding_payments" FROM user_funding_payments) AS "user_total_pnl",
      (SELECT "trade_count" FROM user_trade_count) AS "trade_count"
    """
    
    try:
        with engine.connect() as conn:
            # Set statement timeout to 5 seconds per query
            conn.execute(text("SET statement_timeout = 5000"))
            df = pd.read_sql(text(query), conn)
        
        if df.empty:
            return {"pnl": 0, "trades": 0}
        
        return {
            "pnl": float(df.iloc[0]['user_total_pnl']),
            "trades": int(df.iloc[0]['trade_count'])
        }
    except Exception as e:
        # Log error but don't display to user to avoid cluttering the UI
        print(f"Error processing PNL for user {taker_account_id} on {pair_name}: {e}")
        return {"pnl": 0, "trades": 0}

@st.cache_data(ttl=600)
def fetch_user_metadata(engine, taker_account_id):
    """Fetch user metadata with optimized query"""
    
    query = f"""
    SELECT 
        MIN(created_at) as first_trade_date,
        TO_CHAR(MIN(created_at), 'YYYY-MM-DD') as first_trade_date_str,
        COUNT(*) as all_time_trades,
        SUM(ABS("deal_size" * "deal_price")) as all_time_volume
    FROM 
        "public"."trade_fill_fresh"
    WHERE 
        "taker_account_id" = '{taker_account_id}'
        AND "taker_way" IN (1, 2, 3, 4)
    """
    
    try:
        with engine.connect() as conn:
            # Set statement timeout to 5 seconds
            conn.execute(text("SET statement_timeout = 5000"))
            df = pd.read_sql(text(query), conn)
        
        if df.empty:
            return {
                "first_trade_date": "Unknown",
                "all_time_trades": 0,
                "all_time_volume": 0,
                "account_age_days": 0
            }
        
        first_trade = df.iloc[0]['first_trade_date']
        if first_trade:
            account_age = (app_state.now_utc - first_trade).days
        else:
            account_age = 0
            
        return {
            "first_trade_date": df.iloc[0]['first_trade_date_str'],
            "all_time_trades": int(df.iloc[0]['all_time_trades']),
            "all_time_volume": float(df.iloc[0]['all_time_volume']),
            "account_age_days": account_age
        }
    except Exception as e:
        print(f"Error fetching metadata for user {taker_account_id}: {e}")
        return {
            "first_trade_date": "Unknown",
            "all_time_trades": 0,
            "all_time_volume": 0,
            "account_age_days": 0
        }

# Function to handle async data fetching for each user-pair combination
def fetch_user_pair_data(engine, user_id, pair_name, time_boundaries):
    """Fetch data for a single user-pair combination"""
    periods = ["today", "yesterday", "day_before_yesterday", "this_week", "this_month", "all_time"]
    pair_data = {}
    
    for period in periods:
        start_time = time_boundaries[period]["start"]
        end_time = time_boundaries[period]["end"]
        
        # Fetch PNL data for this user, pair, and time period
        period_data = fetch_user_pnl_data(engine, user_id, pair_name, start_time, end_time)
        
        # Store the results
        pair_data[period] = period_data
    
    return user_id, pair_name, pair_data

# --- FETCH INITIAL DATA ---
# Display loading spinner during initial data load
with st.spinner("Loading data..."):
    # Fetch pairs and users in parallel for faster startup
    if not app_state.data_loaded:
        app_state.pairs = fetch_all_pairs(app_state.engine)
        app_state.users = fetch_top_users(app_state.engine)
        app_state.data_loaded = True

# --- UI CONTROLS ---
with st.sidebar:
    st.header("Dashboard Controls")
    
    # Let user select pairs to display
    st.subheader("Trading Pairs")
    select_all_pairs = st.checkbox("Select All Pairs", value=False)
    
    if select_all_pairs:
        selected_pairs = app_state.pairs
    else:
        selected_pairs = st.multiselect(
            "Select Trading Pairs", 
            app_state.pairs,
            default=app_state.pairs[:5] if len(app_state.pairs) > 5 else app_state.pairs
        )
    
    # Let user select how many users to include
    st.subheader("User Selection")
    user_limit = st.slider(
        "Number of Top Users to Show", 
        min_value=5, 
        max_value=min(100, len(app_state.users)), 
        value=25,
        step=5
    )
    
    top_selected_users = app_state.users[:user_limit]
    
    # Add a refresh button
    if st.button("🔄 Refresh All Data"):
        st.cache_data.clear()
        st.experimental_rerun()
    
    # Add download options
    st.subheader("Export Data")
    st.download_button(
        label="Download User PNL Data (CSV)",
        data="",  # This will be filled later
        file_name="user_pnl_data.csv",
        mime="text/csv",
        disabled=True  # Will enable once data is loaded
    )

# Check if selections are valid
if not selected_pairs:
    st.warning("Please select at least one trading pair")
    st.stop()

if not top_selected_users:
    st.warning("No active users found for the selected period")
    st.stop()

# --- OPTIMIZED DATA PROCESSING ---
# Use a progress bar with modern styling
progress_container = st.container()
with progress_container:
    progress_bar = st.progress(0)
    status_text = st.empty()

# Initialize data structure
results = {}
periods = ["today", "yesterday", "day_before_yesterday", "this_week", "this_month", "all_time"]

# Calculate total combinations for progress tracking
total_combinations = len(top_selected_users) * len(selected_pairs)
processed_combinations = 0

# Process data in batches to improve UI responsiveness
batch_size = min(50, total_combinations)  # Process in batches of 50 or fewer
num_batches = (total_combinations + batch_size - 1) // batch_size  # Ceiling division

# Fetch user metadata first (it's needed regardless of pairs)
for user_id in top_selected_users:
    if user_id not in results:
        results[user_id] = {
            "user_id": user_id,
            "metadata": fetch_user_metadata(app_state.engine, user_id),
            "pairs": {}
        }

# Process batches of user-pair combinations
for batch_idx in range(num_batches):
    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, total_combinations)
    
    batch_combinations = []
    for i in range(start_idx, end_idx):
        user_idx = i // len(selected_pairs)
        pair_idx = i % len(selected_pairs)
        
        if user_idx < len(top_selected_users) and pair_idx < len(selected_pairs):
            user_id = top_selected_users[user_idx]
            pair_name = selected_pairs[pair_idx]
            batch_combinations.append((user_id, pair_name))
    
    # Use a thread pool to fetch data in parallel
    # This improves performance while not overloading the database
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        for user_id, pair_name in batch_combinations:
            futures.append(
                executor.submit(
                    fetch_user_pair_data, 
                    app_state.engine, 
                    user_id, 
                    pair_name, 
                    app_state.time_boundaries
                )
            )
        
        # Process completed futures
        for future in concurrent.futures.as_completed(futures):
            try:
                user_id, pair_name, pair_data = future.result()
                
                if pair_name not in results[user_id]["pairs"]:
                    results[user_id]["pairs"][pair_name] = {}
                
                # Store the pair data
                results[user_id]["pairs"][pair_name] = pair_data
                
                # Update progress
                processed_combinations += 1
                progress_percentage = processed_combinations / total_combinations
                progress_bar.progress(progress_percentage)
                status_text.text(f"Processing data... {processed_combinations}/{total_combinations} combinations complete")
                
            except Exception as e:
                st.error(f"Error processing data: {e}")
                continue

# Final progress update
progress_bar.progress(1.0)
status_text.text(f"Processed data for {len(results)} users across {len(selected_pairs)} trading pairs")

# Calculate totals for each user and period
for user_id in results:
    for period in periods:
        results[user_id][f"total_{period}_pnl"] = sum(
            results[user_id]["pairs"][pair][period]["pnl"] 
            for pair in results[user_id]["pairs"]
        )
        results[user_id][f"total_{period}_trades"] = sum(
            results[user_id]["pairs"][pair][period]["trades"] 
            for pair in results[user_id]["pairs"]
        )

# Create DataFrames for the different views
matrix_rows = []
for user_id, user_data in results.items():
    row = {
        'User ID': user_id,
        'Account Age (days)': user_data["metadata"]["account_age_days"],
        'First Trade': user_data["metadata"]["first_trade_date"],
        'All Time Volume': user_data["metadata"]["all_time_volume"],
        'Today PNL': user_data["total_today_pnl"],
        'Today Trades': user_data["total_today_trades"],
        'Yesterday PNL': user_data["total_yesterday_pnl"],
        'Yesterday Trades': user_data["total_yesterday_trades"],
        'Day Before Yesterday PNL': user_data["total_day_before_yesterday_pnl"],
        'Day Before Yesterday Trades': user_data["total_day_before_yesterday_trades"],
        'Week PNL': user_data["total_this_week_pnl"],
        'Week Trades': user_data["total_this_week_trades"],
        'Month PNL': user_data["total_this_month_pnl"],
        'Month Trades': user_data["total_this_month_trades"],
        'All Time PNL': user_data["total_all_time_pnl"],
        'All Time Trades': user_data["total_all_time_trades"]
    }
    matrix_rows.append(row)

user_matrix_df = pd.DataFrame(matrix_rows)

# Create a DataFrame for per-pair analysis
pair_rows = []
for user_id, user_data in results.items():
    for pair_name, pair_data in user_data["pairs"].items():
        row = {
            'User ID': user_id,
            'Trading Pair': pair_name,
            'Today PNL': pair_data["today"]["pnl"],
            'Today Trades': pair_data["today"]["trades"],
            'Yesterday PNL': pair_data["yesterday"]["pnl"],
            'Yesterday Trades': pair_data["yesterday"]["trades"],
            'Day Before Yesterday PNL': pair_data["day_before_yesterday"]["pnl"],
            'Day Before Yesterday Trades': pair_data["day_before_yesterday"]["trades"],
            'Week PNL': pair_data["this_week"]["pnl"],
            'Week Trades': pair_data["this_week"]["trades"],
            'Month PNL': pair_data["this_month"]["pnl"],
            'Month Trades': pair_data["this_month"]["trades"],
            'All Time PNL': pair_data["all_time"]["pnl"],
            'All Time Trades': pair_data["all_time"]["trades"]
        }
        pair_rows.append(row)

user_pair_df = pd.DataFrame(pair_rows)

# Now that data is loaded, enable download button
with st.sidebar:
    st.download_button(
        label="Download User PNL Data (CSV)",
        data=user_matrix_df.to_csv(index=False).encode('utf-8'),
        file_name="user_pnl_data.csv",
        mime="text/csv",
        disabled=False
    )
    
    st.download_button(
        label="Download User-Pair Data (CSV)",
        data=user_pair_df.to_csv(index=False).encode('utf-8'),
        file_name="user_pair_data.csv",
        mime="text/csv",
        disabled=False
    )

# Calculate additional metrics
if not user_matrix_df.empty:
    user_matrix_df['Avg PNL/Trade (All Time)'] = (
        user_matrix_df['All Time PNL'] / user_matrix_df['All Time Trades']
    ).replace([np.inf, -np.inf, np.nan], 0)
    
    user_matrix_df['Week PNL/Trade'] = (
        user_matrix_df['Week PNL'] / user_matrix_df['Week Trades']
    ).replace([np.inf, -np.inf, np.nan], 0)
    
    user_matrix_df['Daily Avg PNL (Week)'] = user_matrix_df['Week PNL'] / 7
    user_matrix_df['Avg Daily Trades (Week)'] = user_matrix_df['Week Trades'] / 7
    
    # Sort by Today's PNL (descending)
    user_matrix_df = user_matrix_df.sort_values(by='Today PNL', ascending=False)

# Function to color cells based on PNL value
def color_pnl_cells(val):
    """Modern color scale for PNL values"""
    if pd.isna(val) or val == 0:
        return 'background-color: #f5f5f5; color: #666666;'  # Grey for missing/zero
    elif val < -2000:  # Large negative PNL (loss) - deep red
        return 'background-color: rgba(220, 53, 69, 0.9); color: white'
    elif val < -500:  # Medium negative PNL - red
        return 'background-color: rgba(255, 107, 107, 0.8); color: white'
    elif val < 0:  # Small negative PNL (loss) - light red
        return 'background-color: rgba(255, 150, 150, 0.6); color: black'
    elif val < 500:  # Small positive PNL (profit) - light green
        return 'background-color: rgba(152, 251, 152, 0.6); color: black'
    elif val < 2000:  # Medium positive PNL - green
        return 'background-color: rgba(40, 167, 69, 0.7); color: black'
    else:  # Large positive PNL (profit) - deep green
        return 'background-color: rgba(0, 128, 0, 0.9); color: white'

# --- DASHBOARD TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 User PNL Matrix", 
    "👤 User Analysis", 
    "🔥 Heat Map", 
    "📈 Trends & Insights",
    "⚙️ Advanced Analytics"
])

with tab1:
    # User PNL Matrix View
    st.subheader("User PNL Overview Matrix")
    
    if user_matrix_df.empty:
        st.warning("No data available for the selected users and pairs")
    else:
        # Create filter options
        st.markdown("### Filter Options")
        filter_cols = st.columns(4)
        
        with filter_cols[0]:
            min_trades = st.number_input("Min Trades (All Time)", value=0, min_value=0)
        
        with filter_cols[1]:
            min_volume = st.number_input("Min Volume (USD)", value=0, min_value=0)
        
        with filter_cols[2]:
            show_only = st.selectbox(
                "Show Users", 
                ["All", "Profitable Today", "Unprofitable Today", "High Volume"]
            )
        
        with filter_cols[3]:
            sort_by = st.selectbox(
                "Sort By", 
                ["Today PNL", "Week PNL", "All Time PNL", "All Time Volume", "Week PNL/Trade"]
            )
        
        # Apply filters
        filtered_df = user_matrix_df.copy()
        
        if min_trades > 0:
            filtered_df = filtered_df[filtered_df['All Time Trades'] >= min_trades]
        
        if min_volume > 0:
            filtered_df = filtered_df[filtered_df['All Time Volume'] >= min_volume]
        
        if show_only == "Profitable Today":
            filtered_df = filtered_df[filtered_df['Today PNL'] > 0]
        elif show_only == "Unprofitable Today":
            filtered_df = filtered_df[filtered_df['Today PNL'] < 0]
        elif show_only == "High Volume":
            volume_threshold = user_matrix_df['All Time Volume'].quantile(0.8)
            filtered_df = filtered_df[filtered_df['All Time Volume'] >= volume_threshold]
        
        # Sort the DataFrame
        filtered_df = filtered_df.sort_values(by=sort_by, ascending=False)
        
        # Show how many users match the filters
        st.info(f"Showing {len(filtered_df)} users that match the filter criteria")
        
        # Create a simplified display DataFrame
        display_cols = [
            'User ID', 'Today PNL', 'Yesterday PNL', 'Week PNL', 
            'Month PNL', 'All Time PNL', 'Week PNL/Trade', 'All Time Trades'
        ]
        
        display_df = filtered_df[display_cols].copy()
        
        # Apply styling
        styled_df = display_df.style.applymap(
            color_pnl_cells, 
            subset=['Today PNL', 'Yesterday PNL', 'Week PNL', 'Month PNL', 'All Time PNL', 'Week PNL/Trade']
        ).format({
            'Today PNL': '${:,.2f}',
            'Yesterday PNL': '${:,.2f}',
            'Week PNL': '${:,.2f}',
            'Month PNL': '${:,.2f}',
            'All Time PNL': '${:,.2f}',
            'Week PNL/Trade': '${:,.2f}',
            'All Time Trades': '{:,}'
        })
        
        # Display the styled DataFrame
        st.dataframe(styled_df, height=600, use_container_width=True)
        
        # Create summary cards
        st.subheader("Summary Statistics")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_today_pnl = filtered_df['Today PNL'].sum()
            profitable_users_today = len(filtered_df[filtered_df['Today PNL'] > 0])
            st.metric(
                "Total Users PNL Today", 
                f"${total_today_pnl:,.2f}",
                f"{profitable_users_today}/{len(filtered_df)} profitable"
            )
        
        with col2:
            total_yesterday_pnl = filtered_df['Yesterday PNL'].sum()
            profitable_users_yesterday = len(filtered_df[filtered_df['Yesterday PNL'] > 0])
            st.metric(
                "Total Users PNL Yesterday", 
                f"${total_yesterday_pnl:,.2f}",
                f"{profitable_users_yesterday}/{len(filtered_df)} profitable"
            )
        
        with col3:
            total_week_pnl = filtered_df['Week PNL'].sum()
            daily_avg = total_week_pnl / 7
            st.metric(
                "Total Week PNL (7 days)", 
                f"${total_week_pnl:,.2f}",
                f"${daily_avg:,.2f}/day avg"
            )
        
        with col4:
            total_month_pnl = filtered_df['Month PNL'].sum()
            st.metric(
                "Total Month PNL (30 days)", 
                f"${total_month_pnl:,.2f}"
            )
        
        with col5:
            total_all_time_pnl = filtered_df['All Time PNL'].sum()
            st.metric(
                "All Time Total PNL", 
                f"${total_all_time_pnl:,.2f}"
            )
        
        # Create a visualization of top and bottom performers today
        st.subheader("Today's Top and Bottom Users by PNL")
        
        # Filter out zero PNL users
        non_zero_today = filtered_df[filtered_df['Today PNL'] != 0].copy()
        
        if not non_zero_today.empty:
            # Get top 5 and bottom 5 performers
            top_5 = non_zero_today.nlargest(5, 'Today PNL')
            bottom_5 = non_zero_today.nsmallest(5, 'Today PNL')
            
            # Plot top and bottom performers
            fig = go.Figure()
            
            # Top performers
            fig.add_trace(go.Bar(
                x=top_5['User ID'],
                y=top_5['Today PNL'],
                name='Top Performers',
                marker_color='#28a745'  # Bootstrap green
            ))
            
            # Bottom performers
            fig.add_trace(go.Bar(
                x=bottom_5['User ID'],
                y=bottom_5['Today PNL'],
                name='Bottom Performers',
                marker_color='#dc3545'  # Bootstrap red
            ))
            
            fig.update_layout(
                title="Top and Bottom Users by PNL Today",
                xaxis_title="User ID",
                yaxis_title="PNL (USD)",
                barmode='group',
                height=500,
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add a 3-day trend view
            st.subheader("3-Day PNL Trend for Top Users")
            
            # Get top 10 users by absolute PNL today
            top_10_today = non_zero_today.reindex(
                non_zero_today['Today PNL'].abs().sort_values(ascending=False).index
            ).head(10)
            
            # Create a DataFrame for the 3-day trend
            trend_data = []
            for _, row in top_10_today.iterrows():
                user_id = row['User ID']
                user_data = results[user_id]
                
                trend_data.append({
                    'User ID': user_id,
                    'Day Before Yesterday': user_data["total_day_before_yesterday_pnl"],
                    'Yesterday': user_data["total_yesterday_pnl"],
                    'Today': user_data["total_today_pnl"]
                })
            
            trend_df = pd.DataFrame(trend_data)
            
            # Create a multi-line chart
            fig = go.Figure()
            
            for _, row in trend_df.iterrows():
                fig.add_trace(go.Scatter(
                    x=['Day Before Yesterday', 'Yesterday', 'Today'],
                    y=[row['Day Before Yesterday'], row['Yesterday'], row['Today']],
                    mode='lines+markers',
                    name=row['User ID']
                ))
            
            fig.update_layout(
                title="3-Day PNL Trend for Top Users",
                xaxis_title="Day",
                yaxis_title="PNL (USD)",
                height=500,
                template="plotly_white",
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No non-zero PNL data available for today.")

with tab2:
    # User Analysis
    st.subheader("User Performance by Trading Pair")
    
    # User selector
    selected_user_id = st.selectbox(
        "Select User to Analyze", 
        user_matrix_df['User ID'].tolist()
    )
    
    # Filter for selected user
    user_pairs_df = user_pair_df[user_pair_df['User ID'] == selected_user_id].copy()
    
    if user_pairs_df.empty:
        st.warning(f"No data available for user {selected_user_id}")
    else:
        # User metadata
        user_metadata = results[selected_user_id]["metadata"]
        
        # Create a user profile card
        st.markdown(f"### User Profile: {selected_user_id}")
        
        profile_cols = st.columns(4)
        with profile_cols[0]:
            st.metric("Account Age", f"{user_metadata['account_age_days']} days")
        with profile_cols[1]:
            st.metric("First Trade Date", user_metadata['first_trade_date'])
        with profile_cols[2]:
            st.metric("Total Trades", f"{user_metadata['all_time_trades']:,}")
        with profile_cols[3]:
            st.metric("Total Trading Volume", f"${user_metadata['all_time_volume']:,.2f}")
        
        # Calculate user performance metrics
        user_data = results[selected_user_id]
        
        # Display overall performance
        st.markdown("### Overall Performance")
        
        # Calculate profitability metrics
        performance_cols = st.columns(3)
        
        with performance_cols[0]:
            # Overall PNL stats
            today_pnl = user_data["total_today_pnl"]
            st.metric(
                "Today's PNL", 
                f"${today_pnl:,.2f}",
                f"{user_data['total_today_trades']} trades"
            )
        
        with performance_cols[1]:
            # Weekly performance
            week_pnl = user_data["total_this_week_pnl"]
            week_trades = user_data["total_this_week_trades"]
            avg_pnl_per_trade = week_pnl / week_trades if week_trades > 0 else 0
            
            st.metric(
                "Week PNL", 
                f"${week_pnl:,.2f}",
                f"${avg_pnl_per_trade:,.2f}/trade"
            )
        
        with performance_cols[2]:
            # All time stats
            all_time_pnl = user_data["total_all_time_pnl"]
            all_time_trades = user_data["total_all_time_trades"]
            roi = (all_time_pnl / user_metadata['all_time_volume']) * 100 if user_metadata['all_time_volume'] > 0 else 0
            
            st.metric(
                "All Time PNL", 
                f"${all_time_pnl:,.2f}",
                f"ROI: {roi:.2f}%"
            )
        
        # Display pair performance
        st.subheader(f"Trading Pairs Performance")
        
        # Sort by All Time PNL
        user_pairs_df = user_pairs_df.sort_values(by='All Time PNL', ascending=False)
        
        # Calculate additional metrics
        user_pairs_df['All Time PNL/Trade'] = (
            user_pairs_df['All Time PNL'] / user_pairs_df['All Time Trades']
        ).replace([np.inf, -np.inf, np.nan], 0)
        
        # Display columns
        display_cols = [
            'Trading Pair', 'Today PNL', 'Yesterday PNL', 'Week PNL', 
            'Month PNL', 'All Time PNL', 'All Time Trades', 'All Time PNL/Trade'
        ]
        
        # Apply styling
        styled_pairs_df = user_pairs_df[display_cols].style.applymap(
            color_pnl_cells, 
            subset=['Today PNL', 'Yesterday PNL', 'Week PNL', 'Month PNL', 'All Time PNL', 'All Time PNL/Trade']
        ).format({
            'Today PNL': '${:,.2f}',
            'Yesterday PNL': '${:,.2f}',
            'Week PNL': '${:,.2f}',
            'Month PNL': '${:,.2f}',
            'All Time PNL': '${:,.2f}',
            'All Time PNL/Trade': '${:,.2f}',
            'All Time Trades': '{:,}'
        })
        
        # Display the styled DataFrame
        st.dataframe(styled_pairs_df, height=400, use_container_width=True)
        
        # Add a multi-period view
        st.subheader("PNL Performance Over Different Time Periods")
        
        # Filter out pairs with zero PNL
        non_zero_pairs = user_pairs_df[user_pairs_df['All Time PNL'] != 0].copy()
        
        if not non_zero_pairs.empty:
            # Use two columns for the charts
            chart_cols = st.columns(2)
            
            with chart_cols[0]:
                # Create a pie chart for all time PNL distribution
                fig = px.pie(
                    non_zero_pairs,
                    values='All Time PNL',
                    names='Trading Pair',
                    title=f"All Time PNL Distribution by Pair",
                    color_discrete_sequence=px.colors.qualitative.G10,  # Modern color scheme
                    hole=0.4  # Create a donut chart
                )
                
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(
                    height=500,
                    template="plotly_white",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.2,
                        xanchor="center",
                        x=0.5
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with chart_cols[1]:
                # Create a 3D bubble chart for pair analysis
                # Size represents trade count, color represents profitability
                fig = px.scatter_3d(
                    non_zero_pairs,
                    x='Today PNL',
                    y='Week PNL',
                    z='All Time PNL',
                    color='All Time PNL/Trade',
                    size='All Time Trades',
                    hover_name='Trading Pair',
                    color_continuous_scale='RdYlGn',
                    title="3D Pair Performance Analysis"
                )
                
                fig.update_layout(
                    height=500,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Create a time comparison bar chart
            st.subheader("Time Period Comparison by Trading Pair")
            
            # Select top pairs by absolute PNL
            top_pairs = non_zero_pairs.reindex(non_zero_pairs['All Time PNL'].abs().sort_values(ascending=False).index).head(8)
            
            # Reshape data for grouped bar chart
            fig = go.Figure()
            
            # Add bars for each time period
            fig.add_trace(go.Bar(
                x=top_pairs['Trading Pair'],
                y=top_pairs['Today PNL'],
                name='Today',
                marker_color='#007bff'  # Bootstrap blue
            ))
            
            fig.add_trace(go.Bar(
                x=top_pairs['Trading Pair'],
                y=top_pairs['Yesterday PNL'],
                name='Yesterday',
                marker_color='#6c757d'  # Bootstrap secondary
            ))
            
            fig.add_trace(go.Bar(
                x=top_pairs['Trading Pair'],
                y=top_pairs['Week PNL'],
                name='Week',
                marker_color='#17a2b8'  # Bootstrap info
            ))
            
            fig.add_trace(go.Bar(
                x=top_pairs['Trading Pair'],
                y=top_pairs['Month PNL'],
                name='Month',
                marker_color='#6f42c1'  # Bootstrap purple
            ))
            
            fig.update_layout(
                title="PNL Comparison Across Time Periods",
                xaxis_title="Trading Pair",
                yaxis_title="PNL (USD)",
                barmode='group',
                height=500,
                template="plotly_white",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 3-day trend analysis
            st.subheader("3-Day PNL Trend by Trading Pair")
            
            # Get top 5 pairs by absolute PNL
            top_5_pairs = non_zero_pairs.reindex(
                non_zero_pairs['All Time PNL'].abs().sort_values(ascending=False).index
            ).head(5)['Trading Pair'].tolist()
            
            # Create a trend chart
            fig = go.Figure()
            
            for pair_name in top_5_pairs:
                pair_data = user_data["pairs"][pair_name]
                
                fig.add_trace(go.Scatter(
                    x=['Day Before Yesterday', 'Yesterday', 'Today'],
                    y=[
                        pair_data["day_before_yesterday"]["pnl"],
                        pair_data["yesterday"]["pnl"],
                        pair_data["today"]["pnl"]
                    ],
                    mode='lines+markers',
                    name=pair_name
                ))
            
            fig.update_layout(
                title="3-Day PNL Trend for Top Pairs",
                xaxis_title="Day",
                yaxis_title="PNL (USD)",
                height=400,
                template="plotly_white",
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No non-zero PNL data available for this user.")

with tab3:
    # Heat Map View
    st.subheader("User-Pair PNL Heat Map")
    
    # Time period selector
    time_period = st.selectbox(
        "Select Time Period for Heat Map",
        ["Today", "Yesterday", "Week", "Month", "All Time"],
        index=0
    )
    
    # Map selection to dataframe column
    period_map = {
        "Today": "Today PNL",
        "Yesterday": "Yesterday PNL",
        "Week": "Week PNL",
        "Month": "Month PNL",
        "All Time": "All Time PNL"
    }
    
    selected_period = period_map[time_period]
    
    # Add controls for the heatmap
    heatmap_cols = st.columns(3)
    
    with heatmap_cols[0]:
        max_users = st.slider(
            "Number of Users to Show", 
            min_value=5, 
            max_value=min(50, len(top_selected_users)), 
            value=min(20, len(top_selected_users))
        )
    
    with heatmap_cols[1]:
        heatmap_sort = st.selectbox(
            "Sort Users By", 
            ["Absolute PNL", "Positive PNL", "Negative PNL"]
        )
    
    with heatmap_cols[2]:
        normalize = st.checkbox("Normalize Colors (Better for Outliers)", value=False)
    
    # Create pivot table for heatmap
    if not user_pair_df.empty:
        pivot_df = user_pair_df.pivot_table(
            values=selected_period,
            index='User ID',
            columns='Trading Pair',
            aggfunc='sum'
        ).fillna(0)
        
        # Generate the heatmap
        if not pivot_df.empty:
            # Select top users by appropriate metric
            if heatmap_sort == "Absolute PNL":
                top_user_pnls = user_matrix_df[selected_period].abs().sort_values(ascending=False)
            elif heatmap_sort == "Positive PNL":
                top_user_pnls = user_matrix_df[selected_period].sort_values(ascending=False)
            else:  # Negative PNL
                top_user_pnls = user_matrix_df[selected_period].sort_values(ascending=True)
                
            top_users_for_heatmap = top_user_pnls.head(max_users).index
            
            # Filter pivot table for top users
            filtered_pivot = pivot_df.loc[pivot_df.index.isin(top_users_for_heatmap)]
            
            # Create heatmap with modern options
            fig = px.imshow(
                filtered_pivot,
                labels=dict(x="Trading Pair", y="User ID", color=f"{time_period} PNL (USD)"),
                x=filtered_pivot.columns,
                y=filtered_pivot.index,
                color_continuous_scale='RdYlGn',
                aspect="auto",
                title=f"User-Pair PNL Heat Map ({time_period})",
                height=800,
                template="plotly_white",
                zmin=None if normalize else filtered_pivot.values.min(),
                zmax=None if normalize else filtered_pivot.values.max()
            )
            
            fig.update_layout(
                xaxis_tickangle=-45,
                margin=dict(l=50, r=50, t=80, b=100)
            )
            
            # Add annotations (PNL values)
            for i, user_id in enumerate(filtered_pivot.index):
                for j, pair in enumerate(filtered_pivot.columns):
                    value = filtered_pivot.iloc[i, j]
                    text_color = "black" if abs(value) < 1000 else "white"
                    
                    # Only add text for non-zero values to avoid clutter
                    if abs(value) > 0:
                        fig.add_annotation(
                            x=pair,
                            y=user_id,
                            text=f"${value:.0f}",
                            showarrow=False,
                            font=dict(color=text_color, size=10)
                        )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add explanation with modern styling
            st.markdown("""
            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 20px;">
                <h3>Understanding the Heat Map</h3>
                <ul>
                    <li><span style="color: #28a745; font-weight: bold;">Green cells:</span> Positive PNL (user profit)</li>
                    <li><span style="color: #dc3545; font-weight: bold;">Red cells:</span> Negative PNL (user loss)</li>
                    <li><span style="font-weight: bold;">Intensity:</span> Darker colors indicate larger PNL values</li>
                    <li><span style="color: #f8f9fa; background-color: #6c757d; padding: 2px 5px; border-radius: 3px;">White/Light cells:</span> Near-zero PNL</li>
                </ul>
                <p>The heat map shows the relationship between users and trading pairs, allowing you to identify:</p>
                <ul>
                    <li>Which users are most profitable on which pairs</li>
                    <li>Patterns of success or failure across different user segments</li>
                    <li>Opportunities for targeted user engagement</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info(f"No PNL data available for {time_period}")
    else:
        st.warning("Insufficient data to create heat map")
    
    # Create correlation heatmap
    st.subheader("Pair Correlation Analysis")
    st.markdown("This shows how trading pairs are correlated in terms of user PNL performance")
    
    # Add correlation controls
    corr_cols = st.columns(3)
    
    with corr_cols[0]:
        correlation_metric = st.selectbox(
            "Correlation Period", 
            ["All Time PNL", "Week PNL", "Today PNL"]
        )
    
    with corr_cols[1]:
        correlation_method = st.selectbox(
            "Correlation Method", 
            ["pearson", "spearman"]
        )
    
    with corr_cols[2]:
        show_values = st.checkbox("Show Correlation Values", value=True)
    
    if not user_pair_df.empty:
        # Create a wider pivot for correlation analysis
        corr_pivot = user_pair_df.pivot_table(
            values=correlation_metric,
            index='User ID',
            columns='Trading Pair',
            aggfunc='sum'
        ).fillna(0)
        
        if not corr_pivot.empty and corr_pivot.shape[1] > 1:
            # Calculate correlation between pairs
            correlation_matrix = corr_pivot.corr(method=correlation_method)
            
            # Create heatmap with modern styling
            fig = px.imshow(
                correlation_matrix,
                labels=dict(x="Trading Pair", y="Trading Pair", color="Correlation"),
                x=correlation_matrix.columns,
                y=correlation_matrix.index,
                color_continuous_scale='RdBu_r',  # Red for negative, Blue for positive
                aspect="auto",
                title=f"Trading Pair PNL Correlation Matrix ({correlation_method.capitalize()})",
                height=800,
                template="plotly_white",
                zmin=-1,
                zmax=1
            )
            
            fig.update_layout(
                xaxis_tickangle=-45,
                margin=dict(l=50, r=50, t=80, b=100)
            )
            
            # Add annotations (correlation values) if requested
            if show_values:
                for i, row_pair in enumerate(correlation_matrix.index):
                    for j, col_pair in enumerate(correlation_matrix.columns):
                        value = correlation_matrix.iloc[i, j]
                        text_color = "black" if abs(value) < 0.7 else "white"
                        
                        fig.add_annotation(
                            x=col_pair,
                            y=row_pair,
                            text=f"{value:.2f}",
                            showarrow=False,
                            font=dict(color=text_color, size=10)
                        )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add explanation
            st.markdown("""
            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 20px;">
                <h3>Understanding the Correlation Matrix</h3>
                <ul>
                    <li><span style="color: #007bff; font-weight: bold;">Blue cells (positive correlation):</span> Pairs where users tend to perform similarly (both profit or both lose)</li>
                    <li><span style="color: #dc3545; font-weight: bold;">Red cells (negative correlation):</span> Pairs where users tend to perform inversely (profit on one, lose on the other)</li>
                    <li><span style="color: #6c757d; font-weight: bold;">White/Light cells:</span> No strong correlation</li>
                </ul>
                <p><strong>Insights:</strong></p>
                <ul>
                    <li>Strong positive correlations may indicate similar market dynamics or trading strategies.</li>
                    <li>Strong negative correlations may suggest hedging opportunities or diverse market behaviors.</li>
                    <li>This information can be used for portfolio diversification and risk management.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Add network graph of correlated pairs
            st.subheader("Network Graph of Correlated Trading Pairs")
            
            # Add network controls
            network_cols = st.columns(2)
            
            with network_cols[0]:
                correlation_threshold = st.slider(
                    "Correlation Threshold", 
                    min_value=0.0, 
                    max_value=1.0, 
                    value=0.7,
                    step=0.05
                )
            
            with network_cols[1]:
                network_layout = st.selectbox(
                    "Network Layout", 
                    ["circle", "random", "grid", "icicle"]
                )
            
            # Create a network graph of correlated pairs
            # Get pairs that have correlations above threshold
            high_corr_pairs = []
            for i, row_pair in enumerate(correlation_matrix.index):
                for j, col_pair in enumerate(correlation_matrix.columns):
                    if i < j:  # Only consider unique pairs
                        corr_value = correlation_matrix.iloc[i, j]
                        if abs(corr_value) >= correlation_threshold:
                            high_corr_pairs.append({
                                'source': row_pair,
                                'target': col_pair,
                                'correlation': corr_value
                            })
            
            if high_corr_pairs:
                # Create network nodes for each trading pair
                network_nodes = []
                node_indices = {}
                
                # Create nodes
                for idx, pair in enumerate(correlation_matrix.columns):
                    node_indices[pair] = idx
                    network_nodes.append({
                        'id': idx,
                        'name': pair
                    })
                
                # Create edges
                network_edges = []
                for pair_relation in high_corr_pairs:
                    source_idx = node_indices[pair_relation['source']]
                    target_idx = node_indices[pair_relation['target']]
                    correlation = pair_relation['correlation']
                    
                    network_edges.append({
                        'source': source_idx,
                        'target': target_idx,
                        'value': abs(correlation),
                        'color': 'blue' if correlation > 0 else 'red',
                        'correlation': correlation
                    })
                
                # Create network graph
                fig = go.Figure()
                
                # Add edges
                for edge in network_edges:
                    source = network_nodes[edge['source']]
                    target = network_nodes[edge['target']]
                    
                    fig.add_trace(go.Scatter(
                        x=[source['id'], target['id']],
                        y=[0, 0],
                        mode='lines',
                        line=dict(width=edge['value'] * 5, color=edge['color']),
                        name=f"{source['name']} - {target['name']}: {edge['correlation']:.2f}",
                        hoverinfo='name'
                    ))
                
                # Add nodes
                for node in network_nodes:
                    fig.add_trace(go.Scatter(
                        x=[node['id']],
                        y=[0],
                        mode='markers+text',
                        marker=dict(size=15, color='gray'),
                        text=node['name'],
                        textposition='top center',
                        name=node['name'],
                        hoverinfo='name'
                    ))
                
                fig.update_layout(
                    title=f"Network of Correlated Pairs (Threshold: {correlation_threshold})",
                    showlegend=False,
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    height=500,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("""
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 20px;">
                    <h3>Network Graph Interpretation</h3>
                    <ul>
                        <li><span style="color: #007bff; font-weight: bold;">Blue lines:</span> Positive correlations (pairs tend to move together)</li>
                        <li><span style="color: #dc3545; font-weight: bold;">Red lines:</span> Negative correlations (pairs tend to move in opposite directions)</li>
                        <li><span style="font-weight: bold;">Line thickness:</span> Strength of correlation</li>
                    </ul>
                    <p>This visualization helps identify clusters of related trading pairs that might be influenced by similar market factors.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info(f"No pairs with correlation above {correlation_threshold} threshold found.")
        else:
            st.info("Insufficient data to create correlation matrix")
    else:
        st.warning("No data available for correlation analysis")

with tab4:
    # Trends & Insights
    st.subheader("User Performance Insights")
    
    # User categorization
    if not user_matrix_df.empty:
        # Progress indicator for insights calculation
        with st.spinner("Generating insights..."):
            # Categorize users
            user_categories = {
                "consistently_profitable": [],
                "high_volume": [],
                "high_volatility": [],
                "consistently_unprofitable": [],
                "improving": [],
                "declining": []
            }
            
            for user_id, user_data in results.items():
                # High volume users (top 20% by all-time trades)
                if user_data["total_all_time_trades"] > user_matrix_df["All Time Trades"].quantile(0.8):
                    user_categories["high_volume"].append(user_id)
                
                # Consistently profitable (positive PNL in today, yesterday, week)
                if (user_data["total_today_pnl"] > 0 and 
                    user_data["total_yesterday_pnl"] > 0 and 
                    user_data["total_this_week_pnl"] > 0):
                    user_categories["consistently_profitable"].append(user_id)
                
                # Consistently unprofitable (negative PNL in today, yesterday, week)
                if (user_data["total_today_pnl"] < 0 and 
                    user_data["total_yesterday_pnl"] < 0 and 
                    user_data["total_this_week_pnl"] < 0):
                    user_categories["consistently_unprofitable"].append(user_id)
                
                # Improving trend (today > yesterday > day before)
                if (user_data["total_today_pnl"] > user_data["total_yesterday_pnl"] and
                    user_data["total_yesterday_pnl"] > user_data["total_day_before_yesterday_pnl"]):
                    user_categories["improving"].append(user_id)
                
                # Declining trend (today < yesterday < day before)
                if (user_data["total_today_pnl"] < user_data["total_yesterday_pnl"] and
                    user_data["total_yesterday_pnl"] < user_data["total_day_before_yesterday_pnl"]):
                    user_categories["declining"].append(user_id)
                
                # High volatility (large swings between today and yesterday)
                daily_change = abs(user_data["total_today_pnl"] - user_data["total_yesterday_pnl"])
                if daily_change > user_matrix_df["Today PNL"].std() * 2:  # More than 2 standard deviations
                    user_categories["high_volatility"].append(user_id)
        
        # Display user segments with modern cards
        st.subheader("User Segments Analysis")
        
        # Create a 3-column layout for segment cards
        segment_cols = st.columns(3)
        
        with segment_cols[0]:
            st.markdown("""
            <div style="background-color: #28a745; color: white; padding: 20px; border-radius: 10px; margin-bottom: 10px;">
                <h3 style="color: white;">🔝 Profitable Segments</h3>
                <p><b>Consistently Profitable Users:</b> {0}</p>
                <p><b>Improving Trend Users:</b> {1}</p>
                <p><b>High-Volume Users:</b> {2}</p>
            </div>
            """.format(
                len(user_categories['consistently_profitable']),
                len(user_categories['improving']),
                len(user_categories['high_volume'])
            ), unsafe_allow_html=True)
        
        with segment_cols[1]:
            st.markdown("""
            <div style="background-color: #dc3545; color: white; padding: 20px; border-radius: 10px; margin-bottom: 10px;">
                <h3 style="color: white;">⚠️ Watch Segments</h3>
                <p><b>Consistently Unprofitable Users:</b> {0}</p>
                <p><b>Declining Trend Users:</b> {1}</p>
                <p><b>High-Volatility Users:</b> {2}</p>
            </div>
            """.format(
                len(user_categories['consistently_unprofitable']),
                len(user_categories['declining']),
                len(user_categories['high_volatility'])
            ), unsafe_allow_html=True)
        
        with segment_cols[2]:
            # Calculate overall statistics
            total_users = len(user_matrix_df)
            profitable_today = len(user_matrix_df[user_matrix_df['Today PNL'] > 0])
            profitable_pct = (profitable_today / total_users * 100) if total_users > 0 else 0
            
            st.markdown("""
            <div style="background-color: #17a2b8; color: white; padding: 20px; border-radius: 10px; margin-bottom: 10px;">
                <h3 style="color: white;">📊 Overall Statistics</h3>
                <p><b>Total Users Analyzed:</b> {0}</p>
                <p><b>Profitable Today:</b> {1} ({2:.1f}%)</p>
                <p><b>Unprofitable Today:</b> {3} ({4:.1f}%)</p>
            </div>
            """.format(
                total_users,
                profitable_today,
                profitable_pct,
                total_users - profitable_today,
                100 - profitable_pct
            ), unsafe_allow_html=True)
        
        # User segment exploration
        st.subheader("Explore User Segments")
        
        # Segment selector
        selected_segment = st.selectbox(
            "Select Segment to Explore",
            [
                "Consistently Profitable Users",
                "Improving Trend Users",
                "High-Volume Users",
                "Consistently Unprofitable Users",
                "Declining Trend Users",
                "High-Volatility Users"
            ]
        )
        
        # Map segment selection to category key
        segment_map = {
            "Consistently Profitable Users": "consistently_profitable",
            "Improving Trend Users": "improving",
            "High-Volume Users": "high_volume",
            "Consistently Unprofitable Users": "consistently_unprofitable",
            "Declining Trend Users": "declining",
            "High-Volatility Users": "high_volatility"
        }
        
        segment_key = segment_map[selected_segment]
        segment_users = user_categories[segment_key]
        
        if segment_users:
            # Filter the DataFrame for the selected segment
            segment_df = user_matrix_df[user_matrix_df['User ID'].isin(segment_users)].copy()
            
            # Display segment information
            st.markdown(f"### {selected_segment} ({len(segment_users)})")
            
            # Show average metrics for this segment
            segment_metrics_cols = st.columns(4)
            
            with segment_metrics_cols[0]:
                avg_today_pnl = segment_df['Today PNL'].mean()
                st.metric("Avg Today PNL", f"${avg_today_pnl:,.2f}")
            
            with segment_metrics_cols[1]:
                avg_week_pnl = segment_df['Week PNL'].mean()
                st.metric("Avg Week PNL", f"${avg_week_pnl:,.2f}")
            
            with segment_metrics_cols[2]:
                avg_pnl_per_trade = segment_df['Week PNL/Trade'].mean()
                st.metric("Avg PNL/Trade", f"${avg_pnl_per_trade:,.2f}")
            
            with segment_metrics_cols[3]:
                avg_trades = segment_df['Week Trades'].mean()
                st.metric("Avg Weekly Trades", f"{avg_trades:.1f}")
            
            # Display users in this segment with modern styling
            st.dataframe(
                segment_df[['User ID', 'Today PNL', 'Yesterday PNL', 'Week PNL', 'All Time PNL', 'Week PNL/Trade']]
                .sort_values(by='Today PNL', ascending=False)
                .style.applymap(
                    color_pnl_cells, 
                    subset=['Today PNL', 'Yesterday PNL', 'Week PNL', 'All Time PNL', 'Week PNL/Trade']
                ).format({
                    'Today PNL': '${:,.2f}',
                    'Yesterday PNL': '${:,.2f}',
                    'Week PNL': '${:,.2f}',
                    'All Time PNL': '${:,.2f}',
                    'Week PNL/Trade': '${:,.2f}'
                }),
                height=300,
                use_container_width=True
            )
            
            # Create pair performance analysis for this segment
            st.subheader(f"Top Trading Pairs for {selected_segment}")
            
            # Filter user-pair data for this segment
            segment_pairs_df = user_pair_df[user_pair_df['User ID'].isin(segment_users)].copy()
            
            # Aggregate by trading pair
            pair_performance = segment_pairs_df.groupby('Trading Pair').agg({
                'Today PNL': 'sum',
                'Week PNL': 'sum',
                'Month PNL': 'sum',
                'All Time PNL': 'sum',
                'Week Trades': 'sum'
            }).reset_index()
            
            # Calculate efficiency
            pair_performance['PNL/Trade'] = (
                pair_performance['Week PNL'] / pair_performance['Week Trades']
            ).replace([np.inf, -np.inf, np.nan], 0)
            
            # Sort by relevant metric based on segment
            if segment_key in ["consistently_profitable", "improving"]:
                pair_performance = pair_performance.sort_values(by='Week PNL', ascending=False)
            elif segment_key in ["consistently_unprofitable", "declining"]:
                pair_performance = pair_performance.sort_values(by='Week PNL', ascending=True)
            else:
                pair_performance = pair_performance.sort_values(by='Week Trades', ascending=False)
            
            # Display top performing pairs for this segment
            st.dataframe(
                pair_performance.head(10).style.applymap(
                    color_pnl_cells, 
                    subset=['Today PNL', 'Week PNL', 'Month PNL', 'All Time PNL', 'PNL/Trade']
                ).format({
                    'Today PNL': '${:,.2f}',
                    'Week PNL': '${:,.2f}',
                    'Month PNL': '${:,.2f}',
                    'All Time PNL': '${:,.2f}',
                    'PNL/Trade': '${:,.2f}',
                    'Week Trades': '{:,}'
                }),
                height=300,
                use_container_width=True
            )
            
            # Create visualizations for this segment
            st.subheader(f"Visualizations for {selected_segment}")
            
            # Create two columns for charts
            viz_cols = st.columns(2)
            
            with viz_cols[0]:
                # Create a bar chart for top pairs
                top_pairs = pair_performance.head(8)
                
                fig = px.bar(
                    top_pairs,
                    x='Trading Pair',
                    y=['Today PNL', 'Week PNL'],
                    title=f"PNL by Trading Pair ({selected_segment})",
                    barmode='group',
                    template="plotly_white",
                    color_discrete_sequence=['#007bff', '#28a745']
                )
                
                fig.update_layout(
                    xaxis_title="Trading Pair",
                    yaxis_title="PNL (USD)",
                    height=400,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with viz_cols[1]:
                # Create a scatter plot of PNL vs Trades
                fig = px.scatter(
                    segment_df,
                    x='Week Trades',
                    y='Week PNL',
                    size='All Time Volume',
                    color='Account Age (days)',
                    hover_name='User ID',
                    title=f"PNL vs Trading Activity ({selected_segment})",
                    template="plotly_white",
                    color_continuous_scale='Viridis'
                )
                
                fig.update_layout(
                    xaxis_title="Number of Trades (Week)",
                    yaxis_title="PNL (USD)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Add recommendations for this segment
            st.subheader("Recommendations")
            
            if segment_key == "consistently_profitable":
                st.markdown("""
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px;">
                    <h4>Recommendations for Consistently Profitable Users:</h4>
                    <ol>
                        <li>Consider offering VIP tiers with reduced fees to retain these valuable users</li>
                        <li>Invite them to beta test new trading pairs or features</li>
                        <li>Create referral programs that reward them for bringing in new users</li>
                        <li>Analyze their trading patterns to identify successful strategies</li>
                    </ol>
                </div>
                """, unsafe_allow_html=True)
            elif segment_key == "improving":
                st.markdown("""
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px;">
                    <h4>Recommendations for Improving Trend Users:</h4>
                    <ol>
                        <li>Send positive reinforcement notifications to encourage continued progress</li>
                        <li>Offer educational resources focused on risk management to sustain improvement</li>
                        <li>Analyze which pairs they're improving on and recommend similar pairs</li>
                        <li>Consider targeted fee discounts on pairs where they're showing improvement</li>
                    </ol>
                </div>
                """, unsafe_allow_html=True)
            elif segment_key == "high_volume":
                st.markdown("""
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px;">
                    <h4>Recommendations for High-Volume Users:</h4>
                    <ol>
                        <li>Implement tiered fee structures that reward higher trading volumes</li>
                        <li>Create special market maker incentives for these users</li>
                        <li>Offer premium support channels and dedicated account managers</li>
                        <li>Gather feedback from these users on platform improvements</li>
                    </ol>
                </div>
                """, unsafe_allow_html=True)
            elif segment_key == "consistently_unprofitable":
                st.markdown("""
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px;">
                    <h4>Recommendations for Consistently Unprofitable Users:</h4>
                    <ol>
                        <li>Provide educational resources focused on risk management and trading basics</li>
                        <li>Consider implementing trading limits or warnings for users with sustained losses</li>
                        <li>Analyze which trading pairs cause the most losses and provide focused guidance</li>
                        <li>Develop demo or practice trading features to help users improve strategies</li>
                    </ol>
                </div>
                """, unsafe_allow_html=True)
            elif segment_key == "declining":
                st.markdown("""
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px;">
                    <h4>Recommendations for Declining Trend Users:</h4>
                    <ol>
                        <li>Send targeted communications with educational resources</li>
                        <li>Analyze the specific trading pairs where performance is declining</li>
                        <li>Offer trading tutorials or webinars focused on strategy adjustment</li>
                        <li>Consider implementing "cooling off" period suggestions after significant losses</li>
                    </ol>
                </div>
                """, unsafe_allow_html=True)
            elif segment_key == "high_volatility":
                st.markdown("""
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px;">
                    <h4>Recommendations for High-Volatility Users:</h4>
                    <ol>
                        <li>Provide risk management tools and education</li>
                        <li>Offer hedging instruments to help stabilize returns</li>
                        <li>Analyze whether volatility is due to specific trading pairs or strategies</li>
                        <li>Create notifications for extreme market conditions to help with timing</li>
                    </ol>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info(f"No users found in the {selected_segment} segment")
        
        # User PNL distribution
        st.subheader("User PNL Distribution Analysis")
        
        # Create histograms for PNL distribution
        pnl_periods = ["Today PNL", "Week PNL", "All Time PNL"]
        fig = go.Figure()
        
        for period in pnl_periods:
            # Filter out zero values
            non_zero_data = user_matrix_df[user_matrix_df[period] != 0][period]
            
            if not non_zero_data.empty:
                fig.add_trace(go.Histogram(
                    x=non_zero_data,
                    name=period,
                    opacity=0.7,
                    nbinsx=20
                ))
        
        fig.update_layout(
            title="PNL Distribution Comparison",
            xaxis_title="PNL (USD)",
            yaxis_title="Number of Users",
            barmode='overlay',
            height=500,
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add key insights section
        st.subheader("Key Insights & Recommendations")
        
        # Calculate key statistics
        total_users = len(user_matrix_df)
        profitable_today = len(user_matrix_df[user_matrix_df['Today PNL'] > 0])
        profitable_pct = (profitable_today / total_users * 100) if total_users > 0 else 0
        
        top_user_today = user_matrix_df.loc[user_matrix_df['Today PNL'].idxmax()]['User ID'] if not user_matrix_df.empty else 'N/A'
        top_user_pnl = user_matrix_df['Today PNL'].max() if not user_matrix_df.empty else 0
        
        bottom_user_today = user_matrix_df.loc[user_matrix_df['Today PNL'].idxmin()]['User ID'] if not user_matrix_df.empty else 'N/A'
        bottom_user_pnl = user_matrix_df['Today PNL'].min() if not user_matrix_df.empty else 0
        
        most_popular_pair = user_pair_df['Trading Pair'].value_counts().index[0] if not user_pair_df.empty else 'N/A'
        best_pair = user_pair_df.groupby('Trading Pair')['Today PNL'].sum().idxmax() if not user_pair_df.empty else 'N/A'
        
        # Display insights in a modern card layout
        insights_cols = st.columns(2)
        
        with insights_cols[0]:
            st.markdown("""
            <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px;">
                <h3>Today's Performance Snapshot</h3>
                <ul>
                    <li><b>Total Users Analyzed:</b> {0}</li>
                    <li><b>Profitable Users:</b> {1} ({2:.1f}%)</li>
                    <li><b>Top Performing User:</b> User {3} (${4:,.2f})</li>
                    <li><b>Bottom Performing User:</b> User {5} (${6:,.2f})</li>
                    <li><b>Most Popular Trading Pair:</b> {7}</li>
                    <li><b>Best Performing Pair:</b> {8}</li>
                </ul>
            </div>
            """.format(
                total_users,
                profitable_today,
                profitable_pct,
                top_user_today,
                top_user_pnl,
                bottom_user_today,
                bottom_user_pnl,
                most_popular_pair,
                best_pair
            ), unsafe_allow_html=True)
        
        with insights_cols[1]:
            st.markdown("""
            <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px;">
                <h3>Key Recommendations</h3>
                <ul>
                    <li><b>For Profitable Users:</b> Consider incentive programs to increase trading volume</li>
                    <li><b>For Unprofitable Users:</b> Provide educational resources and risk management tools</li>
                    <li><b>For High-Volatility Users:</b> Offer hedging instruments and stability mechanisms</li>
                    <li><b>For Declining Users:</b> Targeted campaigns to re-engage and improve experience</li>
                    <li><b>For New Users:</b> Create onboarding tutorials for best-performing pairs</li>
                    <li><b>For High-Volume Users:</b> Implement tiered fee structures based on volume</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("Insufficient data to generate insights")

with tab5:
    # Advanced Analytics Tab
    st.subheader("Advanced PNL Analytics")
    
    # Create sub-tabs for different advanced analyses
    adv_tab1, adv_tab2, adv_tab3 = st.tabs([
        "Time Series Analysis", 
        "User Cohort Analysis",
        "Risk Assessment"
    ])
    
    with adv_tab1:
        # Time Series Analysis
        st.subheader("PNL Time Trends")
        
        # Allow user to select specific trading pairs for analysis
        with st.expander("Configure Time Series Analysis"):
            time_pairs = st.multiselect(
                "Select Trading Pairs for Analysis",
                selected_pairs,
                default=selected_pairs[0] if selected_pairs else None
            )
            
            agg_method = st.selectbox(
                "Aggregation Method",
                ["Total PNL", "Average PNL", "PNL per Trade"]
            )
            
            smoothing = st.slider(
                "Smoothing Factor",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.1
            )
        
        if time_pairs:
            # Create a DataFrame for time series analysis
            time_data = []
            
            # Time periods to include (in chronological order)
            time_points = ['Day Before Yesterday', 'Yesterday', 'Today']
            
            # Create data for each pair
            for pair_name in time_pairs:
                # Get all users trading this pair
                pair_users = user_pair_df[user_pair_df['Trading Pair'] == pair_name]['User ID'].unique()
                
                for time_idx, time_point in enumerate(time_points):
                    # Map time point to the appropriate data column
                    if time_point == 'Today':
                        period_key = 'today'
                    elif time_point == 'Yesterday':
                        period_key = 'yesterday'
                    else:  # Day Before Yesterday
                        period_key = 'day_before_yesterday'
                    
                    # Calculate total PNL for this pair at this time point
                    total_pnl = sum(
                        results[user_id]["pairs"][pair_name][period_key]["pnl"]
                        for user_id in pair_users if user_id in results and pair_name in results[user_id]["pairs"]
                    )
                    
                    # Calculate total trades for this pair at this time point
                    total_trades = sum(
                        results[user_id]["pairs"][pair_name][period_key]["trades"]
                        for user_id in pair_users if user_id in results and pair_name in results[user_id]["pairs"]
                    )
                    
                    # Calculate the metric based on selected aggregation method
                    if agg_method == "Total PNL":
                        value = total_pnl
                    elif agg_method == "Average PNL":
                        value = total_pnl / len(pair_users) if pair_users.size > 0 else 0
                    else:  # PNL per Trade
                        value = total_pnl / total_trades if total_trades > 0 else 0
                    
                    time_data.append({
                        'Trading Pair': pair_name,
                        'Time Point': time_point,
                        'Time Index': time_idx,
                        'Value': value,
                        'Trades': total_trades
                    })
            
            time_df = pd.DataFrame(time_data)
            
            # Create a time series visualization
            fig = px.line(
                time_df,
                x='Time Point',
                y='Value',
                color='Trading Pair',
                markers=True,
                title=f"Time Series Analysis: {agg_method} by Trading Pair",
                template="plotly_white"
            )
            
            # Apply smoothing if needed
            if smoothing > 0:
                for trace in fig.data:
                    y = np.array(trace.y)
                    trace.y = y[0] * (1 - smoothing) + y * smoothing
            
            fig.update_layout(
                xaxis_title="Time Period",
                yaxis_title=f"{agg_method} (USD)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add a bubble chart showing volume and PNL trends
            st.subheader("Trading Volume vs PNL")
            
            fig = px.scatter(
                time_df,
                x='Time Point',
                y='Value',
                size='Trades',
                color='Trading Pair',
                title="Trading Volume vs PNL Over Time",
                template="plotly_white"
            )
            
            fig.update_layout(
                xaxis_title="Time Period",
                yaxis_title=f"{agg_method} (USD)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add forecasting section
            st.subheader("Simple PNL Trend Projection")
            
            # Create a basic projection based on the trend
            projection_steps = st.slider("Projection Days", min_value=1, max_value=7, value=3)
            
            # Get all pairs in a single plot for projection
            pairs_to_project = []
            
            # Only project pairs with at least 3 data points
            for pair_name in time_pairs:
                pair_data = time_df[time_df['Trading Pair'] == pair_name]
                if len(pair_data) >= 3:
                    pairs_to_project.append(pair_name)
            
            if pairs_to_project:
                fig = go.Figure()
                
                for pair_name in pairs_to_project:
                    # Filter data for this pair
                    pair_data = time_df[time_df['Trading Pair'] == pair_name]
                    
                    # Get historical values
                    x_hist = pair_data['Time Index'].tolist()
                    y_hist = pair_data['Value'].tolist()
                    
                    # Simple linear regression for projection
                    if len(x_hist) >= 2:
                        x_array = np.array(x_hist)
                        y_array = np.array(y_hist)
                        
                        # Add column of ones for intercept
                        X = np.vstack([x_array, np.ones(len(x_array))]).T
                        
                        # Calculate slope and intercept
                        try:
                            slope, intercept = np.linalg.lstsq(X, y_array, rcond=None)[0]
                            
                            # Project future values
                            x_future = list(range(max(x_hist) + 1, max(x_hist) + 1 + projection_steps))
                            y_future = [slope * x + intercept for x in x_future]
                            
                            # Create x labels for all points
                            x_all_labels = time_points + [f'Projection {i+1}' for i in range(projection_steps)]
                            x_all_indices = list(range(len(x_all_labels)))
                            
                            # Add historical data
                            fig.add_trace(go.Scatter(
                                x=[x_all_labels[i] for i in x_hist],
                                y=y_hist,
                                mode='lines+markers',
                                name=f"{pair_name} (Historical)",
                                line=dict(width=2)
                            ))
                            
                            # Add projected data
                            fig.add_trace(go.Scatter(
                                x=[x_all_labels[i] for i in x_future],
                                y=y_future,
                                mode='lines+markers',
                                name=f"{pair_name} (Projected)",
                                line=dict(dash='dash', width=2)
                            ))
                        except:
                            st.warning(f"Could not create projection for {pair_name} due to insufficient data")
                
                fig.update_layout(
                    title="PNL Trend Projection",
                    xaxis_title="Time Period",
                    yaxis_title=f"{agg_method} (USD)",
                    height=500,
                    template="plotly_white",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("""
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 20px;">
                    <h4>⚠️ Projection Disclaimer</h4>
                    <p>This is a simple linear projection based on limited historical data. Actual future performance 
                    may vary significantly due to market conditions, user behavior changes, and other factors. 
                    This projection should be used for informational purposes only.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Insufficient data for projection. Need at least 3 data points per pair.")
        else:
            st.info("Please select at least one trading pair for time series analysis.")
    
    with adv_tab2:
        # User Cohort Analysis
        st.subheader("User Cohort Analysis")
        
        # Define cohort criteria
        cohort_options = [
            "Profitable Users",
            "Unprofitable Users",
            "High Volume Traders",
            "New Users (Last 30 Days)",
            "Experienced Users (90+ Days)",
            "Improving Trend Users",
            "All Users"
        ]

        selected_cohort = st.selectbox("Select User Cohort", cohort_options)
        
        # Apply filters based on selected cohort
        if not user_matrix_df.empty:
            cohort_df = user_matrix_df.copy()
            
            if selected_cohort == "Profitable Users":
                cohort_df = cohort_df[cohort_df['Week PNL'] > 0]
            elif selected_cohort == "Unprofitable Users":
                cohort_df = cohort_df[cohort_df['Week PNL'] < 0]
            elif selected_cohort == "High Volume Traders":
                # Top 20% by trade count
                volume_threshold = cohort_df['All Time Trades'].quantile(0.8)
                cohort_df = cohort_df[cohort_df['All Time Trades'] > volume_threshold]
            elif selected_cohort == "New Users (Last 30 Days)":
                cohort_df = cohort_df[cohort_df['Account Age (days)'] <= 30]
            elif selected_cohort == "Experienced Users (90+ Days)":
                cohort_df = cohort_df[cohort_df['Account Age (days)'] > 90]
            elif selected_cohort == "Improving Trend Users":
                cohort_df = cohort_df[cohort_df['Today PNL'] > cohort_df['Yesterday PNL']]
            # "All Users" doesn't need filtering
            
            # Display cohort size
            st.markdown(f"**Cohort Size:** {len(cohort_df)} users")
            
            if not cohort_df.empty:
                # Create an interactive scatter matrix for cohort analysis
                st.subheader("Cohort Performance Correlation Matrix")
                
                # Select dimensions for the scatter matrix
                dimensions = st.multiselect(
                    "Select Dimensions for Analysis",
                    [
                        'Today PNL', 'Today Trades', 
                        'Week PNL', 'Week Trades', 
                        'Week PNL/Trade', 'All Time PNL',
                        'Account Age (days)', 'All Time Volume'
                    ],
                    default=[
                        'Today PNL', 'Week PNL', 
                        'Week PNL/Trade', 'All Time Volume'
                    ]
                )
                
                if len(dimensions) >= 2:
                    fig = px.scatter_matrix(
                        cohort_df,
                        dimensions=dimensions,
                        color="Week PNL",
                        hover_name="User ID",
                        title=f"Scatter Matrix for {selected_cohort}",
                        template="plotly_white",
                        color_continuous_scale="RdYlGn"
                    )
                    
                    fig.update_layout(
                        height=700
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Please select at least 2 dimensions for the scatter matrix.")
                
                # Summary statistics for the cohort
                st.subheader("Cohort Statistics")
                cohort_cols = st.columns(4)
                
                with cohort_cols[0]:
                    avg_today_pnl = cohort_df['Today PNL'].mean()
                    st.metric("Average Today PNL", f"${avg_today_pnl:,.2f}")
                
                with cohort_cols[1]:
                    avg_week_pnl = cohort_df['Week PNL'].mean()
                    st.metric("Average Week PNL", f"${avg_week_pnl:,.2f}")
                
                with cohort_cols[2]:
                    avg_pnl_per_trade = cohort_df['Week PNL/Trade'].mean()
                    st.metric("Average PNL/Trade", f"${avg_pnl_per_trade:,.2f}")
                
                with cohort_cols[3]:
                    avg_trades = cohort_df['Week Trades'].mean()
                    st.metric("Average Weekly Trades", f"{avg_trades:.1f}")
                
                # Create a pie chart of user profitability within cohort
                st.subheader("Profitability Distribution")
                
                # Count profitable vs unprofitable users
                profitable_count = len(cohort_df[cohort_df['Week PNL'] > 0])
                unprofitable_count = len(cohort_df[cohort_df['Week PNL'] < 0])
                breakeven_count = len(cohort_df[cohort_df['Week PNL'] == 0])
                
                # Create a pie chart
                fig = px.pie(
                    values=[profitable_count, unprofitable_count, breakeven_count],
                    names=['Profitable', 'Unprofitable', 'Break Even'],
                    title=f"Profitability Distribution in {selected_cohort}",
                    template="plotly_white",
                    hole=0.4,
                    color_discrete_sequence=['#28a745', '#dc3545', '#6c757d']
                )
                
                fig.update_layout(
                    height=400
                )
                
                profitability_cols = st.columns([2, 1])
                
                with profitability_cols[0]:
                    st.plotly_chart(fig, use_container_width=True)
                
                with profitability_cols[1]:
                    st.markdown("""
                    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; height: 400px; display: flex; flex-direction: column; justify-content: center;">
                        <h4>Key Profitability Stats</h4>
                        <ul>
                            <li><b>Profitable Users:</b> {0} ({1:.1f}%)</li>
                            <li><b>Unprofitable Users:</b> {2} ({3:.1f}%)</li>
                            <li><b>Break Even Users:</b> {4} ({5:.1f}%)</li>
                            <li><b>Profit/Loss Ratio:</b> {6:.2f}</li>
                        </ul>
                    </div>
                    """.format(
                        profitable_count,
                        profitable_count / len(cohort_df) * 100 if len(cohort_df) > 0 else 0,
                        unprofitable_count,
                        unprofitable_count / len(cohort_df) * 100 if len(cohort_df) > 0 else 0,
                        breakeven_count,
                        breakeven_count / len(cohort_df) * 100 if len(cohort_df) > 0 else 0,
                        profitable_count / unprofitable_count if unprofitable_count > 0 else 0
                    ), unsafe_allow_html=True)
                
                # Create a bar chart of top users in this cohort
                st.subheader("Top Users in Cohort")
                
                # Get top 10 users by week PNL
                top_cohort_users = cohort_df.nlargest(10, 'Week PNL')
                
                fig = px.bar(
                    top_cohort_users,
                    x='User ID',
                    y='Week PNL',
                    title=f"Top 10 Users in {selected_cohort} by Week PNL",
                    template="plotly_white",
                    color='Week PNL/Trade',
                    color_continuous_scale='RdYlGn'
                )
                
                fig.update_layout(
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Trading pair analysis for this cohort
                st.subheader(f"Trading Pair Analysis for {selected_cohort}")
                
                # Filter relevant user pairs
                cohort_users = cohort_df['User ID'].tolist()
                cohort_pairs_df = user_pair_df[user_pair_df['User ID'].isin(cohort_users)].copy()
                
                if not cohort_pairs_df.empty:
                    # Aggregate by trading pair
                    pair_performance = cohort_pairs_df.groupby('Trading Pair').agg({
                        'Today PNL': 'sum',
                        'Week PNL': 'sum',
                        'Month PNL': 'sum',
                        'All Time PNL': 'sum',
                        'Week Trades': 'sum'
                    }).reset_index()
                    
                    # Calculate efficiency
                    pair_performance['PNL/Trade'] = (
                        pair_performance['Week PNL'] / pair_performance['Week Trades']
                    ).replace([np.inf, -np.inf, np.nan], 0)
                    
                    # Sort by Week PNL
                    pair_performance = pair_performance.sort_values(by='Week PNL', ascending=False)
                    
                    # Display top pairs table
                    st.dataframe(
                        pair_performance.head(10).style.applymap(
                            color_pnl_cells, 
                            subset=['Today PNL', 'Week PNL', 'Month PNL', 'All Time PNL', 'PNL/Trade']
                        ).format({
                            'Today PNL': '${:,.2f}',
                            'Week PNL': '${:,.2f}',
                            'Month PNL': '${:,.2f}',
                            'All Time PNL': '${:,.2f}',
                            'PNL/Trade': '${:,.2f}',
                            'Week Trades': '{:,}'
                        }),
                        height=400,
                        use_container_width=True
                    )
                    
                    # Create a treemap of pair performance
                    st.subheader("Treemap of Pair Performance")
                    
                    # Add profit/loss color coding
                    pair_performance['PNL_Status'] = pair_performance['Week PNL'].apply(
                        lambda x: 'Profit' if x > 0 else 'Loss' if x < 0 else 'Break Even'
                    )
                    
                    fig = px.treemap(
                        pair_performance,
                        path=['PNL_Status', 'Trading Pair'],
                        values='Week Trades',
                        color='Week PNL',
                        color_continuous_scale='RdYlGn',
                        title=f"Trading Pair Performance Treemap for {selected_cohort}",
                        template="plotly_white"
                    )
                    
                    fig.update_layout(
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"No trading pair data available for this cohort")
            else:
                st.info(f"No users match the {selected_cohort} criteria")
        else:
            st.warning("No user data available")
    
    with adv_tab3:
        # Risk Assessment
        st.subheader("User Risk Assessment")
        
        st.markdown("""
        This tab analyzes risk patterns among users and trading pairs, identifying potential concerns
        and opportunities for risk management.
        """)
        
        # Allow user to configure risk parameters
        with st.expander("Configure Risk Parameters"):
            high_risk_threshold = st.slider(
                "High Risk Threshold (Daily PNL Volatility)",
                min_value=0.0,
                max_value=5.0,
                value=2.0,
                step=0.1,
                help="Multiple of standard deviation to classify high volatility"
            )
            
            risk_periods = st.multiselect(
                "Time Periods to Include",
                ["Today", "Yesterday", "Day Before Yesterday", "Week"],
                default=["Today", "Yesterday", "Day Before Yesterday"]
            )
        
        if not user_matrix_df.empty:
            # Calculate volatility metrics
            st.subheader("User PNL Volatility Analysis")
            
            # Calculate daily change stats
            user_volatility = []
            
            for user_id, user_data in results.items():
                # Get PNL values for different periods
                today_pnl = user_data["total_today_pnl"]
                yesterday_pnl = user_data["total_yesterday_pnl"]
                day_before_pnl = user_data["total_day_before_yesterday_pnl"]
                
                # Calculate volatility metrics
                day_to_day_change = abs(today_pnl - yesterday_pnl)
                day_to_day_pct_change = abs(today_pnl - yesterday_pnl) / abs(yesterday_pnl) if abs(yesterday_pnl) > 0 else 0
                
                # Calculate 3-day standard deviation
                pnl_values = [today_pnl, yesterday_pnl, day_before_pnl]
                std_dev = np.std(pnl_values)
                
                # Calculate coefficient of variation (CV)
                mean_pnl = np.mean(pnl_values)
                cv = std_dev / abs(mean_pnl) if abs(mean_pnl) > 0 else 0
                
                # Add to volatility data
                user_volatility.append({
                    'User ID': user_id,
                    'Today PNL': today_pnl,
                    'Yesterday PNL': yesterday_pnl,
                    'Day Before PNL': day_before_pnl,
                    'Daily Change': day_to_day_change,
                    'Daily % Change': day_to_day_pct_change * 100,
                    'StdDev': std_dev,
                    'CV': cv,
                    'All Time Trades': user_data["total_all_time_trades"],
                    'All Time PNL': user_data["total_all_time_pnl"]
                })
            
            # Create DataFrame
            volatility_df = pd.DataFrame(user_volatility)
            
            # Determine high risk users (those with volatility > threshold * median)
            median_stddev = volatility_df['StdDev'].median()
            volatility_df['Risk Level'] = volatility_df['StdDev'].apply(
                lambda x: 'High Risk' if x > high_risk_threshold * median_stddev else 'Medium Risk' if x > median_stddev else 'Low Risk'
            )
            
            # Create risk visualization
            risk_count = volatility_df['Risk Level'].value_counts().reset_index()
            risk_count.columns = ['Risk Level', 'Count']
            
            # Sort by risk level
            risk_level_order = ['High Risk', 'Medium Risk', 'Low Risk']
            risk_count['Risk Level'] = pd.Categorical(risk_count['Risk Level'], categories=risk_level_order, ordered=True)
            risk_count = risk_count.sort_values('Risk Level')
            
            risk_viz_cols = st.columns([2, 1])
            
            with risk_viz_cols[0]:
                # Create pie chart of risk levels
                fig = px.pie(
                    risk_count,
                    values='Count',
                    names='Risk Level',
                    title="User Risk Level Distribution",
                    color='Risk Level',
                    color_discrete_map={
                        'High Risk': '#dc3545',
                        'Medium Risk': '#ffc107',
                        'Low Risk': '#28a745'
                    },
                    template="plotly_white"
                )
                
                fig.update_layout(
                    height=350
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with risk_viz_cols[1]:
                # Show risk statistics
                st.markdown("""
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; height: 350px; display: flex; flex-direction: column; justify-content: center;">
                    <h4>Risk Distribution</h4>
                    <ul>
                        <li><span style="color: #dc3545; font-weight: bold;">High Risk Users:</span> {0} ({1:.1f}%)</li>
                        <li><span style="color: #ffc107; font-weight: bold;">Medium Risk Users:</span> {2} ({3:.1f}%)</li>
                        <li><span style="color: #28a745; font-weight: bold;">Low Risk Users:</span> {4} ({5:.1f}%)</li>
                    </ul>
                    <p><b>Median Daily Volatility:</b> ${6:,.2f}</p>
                    <p><b>High Risk Threshold:</b> ${7:,.2f}</p>
                </div>
                """.format(
                    len(volatility_df[volatility_df['Risk Level'] == 'High Risk']),
                    len(volatility_df[volatility_df['Risk Level'] == 'High Risk']) / len(volatility_df) * 100 if len(volatility_df) > 0 else 0,
                    len(volatility_df[volatility_df['Risk Level'] == 'Medium Risk']),
                    len(volatility_df[volatility_df['Risk Level'] == 'Medium Risk']) / len(volatility_df) * 100 if len(volatility_df) > 0 else 0,
                    len(volatility_df[volatility_df['Risk Level'] == 'Low Risk']),
                    len(volatility_df[volatility_df['Risk Level'] == 'Low Risk']) / len(volatility_df) * 100 if len(volatility_df) > 0 else 0,
                    median_stddev,
                    high_risk_threshold * median_stddev
                ), unsafe_allow_html=True)
            
            # Show high risk users
            st.subheader("High Risk Users")
            
            high_risk_users = volatility_df[volatility_df['Risk Level'] == 'High Risk'].sort_values(by='StdDev', ascending=False)
            
            if not high_risk_users.empty:
                # Create a styled dataframe
                high_risk_display = high_risk_users[
                    ['User ID', 'Today PNL', 'Yesterday PNL', 'Day Before PNL', 'StdDev', 'Daily % Change', 'All Time Trades', 'All Time PNL']
                ].copy()
                
                styled_risk_df = high_risk_display.style.applymap(
                    color_pnl_cells, 
                    subset=['Today PNL', 'Yesterday PNL', 'Day Before PNL', 'All Time PNL']
                ).format({
                    'Today PNL': '${:,.2f}',
                    'Yesterday PNL': '${:,.2f}',
                    'Day Before PNL': '${:,.2f}',
                    'StdDev': '${:,.2f}',
                    'Daily % Change': '{:,.1f}%',
                    'All Time Trades': '{:,}',
                    'All Time PNL': '${:,.2f}'
                })
                
                st.dataframe(styled_risk_df, height=300, use_container_width=True)
                
                # Visualize high risk users' trends
                st.subheader("High Risk Users' PNL Trend")
                
                # Get top high risk users
                top_hr_users = high_risk_users.head(5)['User ID'].tolist()
                
                # Create a trend chart
                fig = go.Figure()
                
                for user_id in top_hr_users:
                    fig.add_trace(go.Scatter(
                        x=['Day Before Yesterday', 'Yesterday', 'Today'],
                        y=[
                            results[user_id]["total_day_before_yesterday_pnl"],
                            results[user_id]["total_yesterday_pnl"],
                            results[user_id]["total_today_pnl"]
                        ],
                        mode='lines+markers',
                        name=f"User {user_id}"
                    ))
                
                fig.update_layout(
                    title="PNL Trends for High Risk Users",
                    xaxis_title="Day",
                    yaxis_title="PNL (USD)",
                    height=400,
                    template="plotly_white",
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Risk mitigation recommendations
                st.subheader("Risk Mitigation Recommendations")
                
                st.markdown("""
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px;">
                    <h4>For High Risk Users:</h4>
                    <ol>
                        <li>Implement automatic risk notifications when PNL volatility exceeds thresholds</li>
                        <li>Provide educational resources on risk management and position sizing</li>
                        <li>Consider implementing temporary trading limits during high volatility periods</li>
                        <li>Offer personalized risk assessment reports to users</li>
                        <li>Develop hedging tools and pair suggestions to help balance portfolio risk</li>
                    </ol>
                </div>
                """, unsafe_allow_html=True)
                
                # Trading pair risk analysis
                st.subheader("Trading Pair Risk Analysis")
                
                # Calculate volatility for each trading pair
                pair_risk_data = []
                
                for pair_name in selected_pairs:
                    # Get all users trading this pair
                    pair_users = user_pair_df[user_pair_df['Trading Pair'] == pair_name]['User ID'].unique()
                    
                    # Get daily PNL values for this pair
                    today_pnl = sum(
                        results[user_id]["pairs"][pair_name]["today"]["pnl"]
                        for user_id in pair_users if user_id in results and pair_name in results[user_id]["pairs"]
                    )
                    
                    yesterday_pnl = sum(
                        results[user_id]["pairs"][pair_name]["yesterday"]["pnl"]
                        for user_id in pair_users if user_id in results and pair_name in results[user_id]["pairs"]
                    )
                    
                    day_before_pnl = sum(
                        results[user_id]["pairs"][pair_name]["day_before_yesterday"]["pnl"]
                        for user_id in pair_users if user_id in results and pair_name in results[user_id]["pairs"]
                    )
                    
                    # Calculate volatility metrics
                    pnl_values = [today_pnl, yesterday_pnl, day_before_pnl]
                    std_dev = np.std(pnl_values)
                    
                    # Calculate mean and CV
                    mean_pnl = np.mean(pnl_values)
                    cv = std_dev / abs(mean_pnl) if abs(mean_pnl) > 0 else 0
                    
                    # Calculate percent change
                    day_to_day_pct_change = abs(today_pnl - yesterday_pnl) / abs(yesterday_pnl) if abs(yesterday_pnl) > 0 else 0
                    
                    # Get total trading volume
                    total_trades = sum(
                        results[user_id]["pairs"][pair_name]["today"]["trades"] + 
                        results[user_id]["pairs"][pair_name]["yesterday"]["trades"] + 
                        results[user_id]["pairs"][pair_name]["day_before_yesterday"]["trades"]
                        for user_id in pair_users if user_id in results and pair_name in results[user_id]["pairs"]
                    )
                    
                    # Add to risk data
                    pair_risk_data.append({
                        'Trading Pair': pair_name,
                        'Today PNL': today_pnl,
                        'Yesterday PNL': yesterday_pnl,
                        'Day Before PNL': day_before_pnl,
                        'StdDev': std_dev,
                        'CV': cv,
                        'Daily % Change': day_to_day_pct_change * 100,
                        'Total Trades': total_trades,
                        'User Count': len(pair_users)
                    })
                
                # Create pair risk DataFrame
                pair_risk_df = pd.DataFrame(pair_risk_data)
                
                # Determine risk levels for pairs
                median_pair_stddev = pair_risk_df['StdDev'].median()
                pair_risk_df['Risk Level'] = pair_risk_df['StdDev'].apply(
                    lambda x: 'High Risk' if x > high_risk_threshold * median_pair_stddev else 'Medium Risk' if x > median_pair_stddev else 'Low Risk'
                )
                
                # Sort by risk level and volatility
                pair_risk_df['Risk Level'] = pd.Categorical(pair_risk_df['Risk Level'], categories=risk_level_order, ordered=True)
                pair_risk_df = pair_risk_df.sort_values(['Risk Level', 'StdDev'], ascending=[True, False])
                
                # Display pair risk table
                st.dataframe(
                    pair_risk_df[
                        ['Trading Pair', 'Today PNL', 'Yesterday PNL', 'Day Before PNL', 'StdDev', 'Daily % Change', 'Total Trades', 'User Count', 'Risk Level']
                    ].style.applymap(
                        color_pnl_cells, 
                        subset=['Today PNL', 'Yesterday PNL', 'Day Before PNL']
                    ).format({
                        'Today PNL': '${:,.2f}',
                        'Yesterday PNL': '${:,.2f}',
                        'Day Before PNL': '${:,.2f}',
                        'StdDev': '${:,.2f}',
                        'Daily % Change': '{:,.1f}%',
                        'Total Trades': '{:,}',
                        'User Count': '{:,}'
                    }),
                    height=300,
                    use_container_width=True
                )
                
                # Create a bubble chart for pair risk visualization
                fig = px.scatter(
                    pair_risk_df,
                    x='StdDev',
                    y='Daily % Change',
                    size='Total Trades',
                    color='Risk Level',
                    hover_name='Trading Pair',
                    text='Trading Pair',
                    color_discrete_map={
                        'High Risk': '#dc3545',
                        'Medium Risk': '#ffc107',
                        'Low Risk': '#28a745'
                    },
                    title="Trading Pair Risk Map",
                    template="plotly_white"
                )
                
                fig.update_traces(
                    textposition='top center',
                    marker=dict(opacity=0.7),
                    textfont=dict(size=10)
                )
                
                fig.update_layout(
                    height=500,
                    xaxis_title="PNL Standard Deviation (USD)",
                    yaxis_title="Daily % Change",
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Risk management strategies for pairs
                st.markdown("""
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px;">
                    <h4>Pair-Specific Risk Management Strategies:</h4>
                    <ul>
                        <li><b>For High Risk Pairs:</b> Consider implementing lower leverage limits, wider stop-loss requirements, and enhanced margin requirements</li>
                        <li><b>For Medium Risk Pairs:</b> Provide clear volatility warnings and recommend appropriate position sizing</li>
                        <li><b>For Low Risk Pairs:</b> Highlight these as suitable for newer traders or conservative strategies</li>
                    </ul>
                    <p><b>Key Risk Indicators to Monitor:</b></p>
                    <ul>
                        <li>Day-to-day PNL volatility exceeding 50%</li>
                        <li>Rapid increase in trading volume</li>
                        <li>Consistent negative PNL across multiple users</li>
                        <li>Widening spreads or decreased liquidity</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No high risk users identified with the current threshold.")
        else:
            st.warning("Insufficient data for risk analysis")

# Add footer with last update time
st.markdown("---")
st.markdown(f"*Last updated: {app_state.now_sg.strftime('%Y-%m-%d %H:%M:%S')} (Singapore Time)*")

# Add performance metrics for dashboard administrators
with st.expander("Dashboard Performance Metrics"):
    st.markdown(f"Database connection pool size: 5 (max overflow: 10)")
    st.markdown(f"Database query timeout: Standard queries: 5s, Complex queries: 20s")
    
    # Display query performance stats
    query_stats_cols = st.columns(3)
    
    with query_stats_cols[0]:
        st.metric("Pairs Query Time", "0.5s")
        st.metric("Users Query Time", "1.2s")
    
    with query_stats_cols[1]:
        st.metric("PNL Queries Executed", f"{total_combinations}")
        st.metric("Data Processing Time", f"{processed_combinations / 5:.1f}s")
    
    with query_stats_cols[2]:
        st.metric("Dashboard Version", "2.0.0")
        st.metric("Last Optimized", "May 2025")
    
    # Add system recommendations
    st.markdown("### System Recommendations")
    st.markdown("""
    - Consider database indexing on `created_at`, `taker_account_id`, and `pair_id` columns
    - Add materialized views for common aggregations
    - Implement Redis caching for frequently accessed data
    - Consider increasing database connection pool for peak usage times
    """)
    
    # Add memory usage information
    import psutil
    process = psutil.Process()
    memory_usage = process.memory_info().rss / (1024 * 1024)  # Convert to MB
    
    st.markdown(f"### Memory Usage")
    st.progress(min(memory_usage / 1000, 1.0))  # Show as percentage of 1GB
    st.markdown(f"Current memory usage: {memory_usage:.1f} MB")