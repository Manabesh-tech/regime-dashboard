import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import pytz
import time

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="User PNL Matrix Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- DATABASE CONNECTION ---
# Use a cached resource for the database connection instead of creating new ones
@st.cache_resource
def get_database_connection():
    """Create a database connection - cached to reuse connections"""
    try:
        db_config = st.secrets["database"]
        db_uri = (
            f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}"
            f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )
        return create_engine(db_uri)
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return None

# --- TIMEZONE SETUP ---
singapore_timezone = pytz.timezone('Asia/Singapore')
now_utc = datetime.now(pytz.utc)
now_sg = now_utc.astimezone(singapore_timezone)

# --- TIME BOUNDARIES ---
def get_time_boundaries():
    """Calculate time boundaries for different periods"""
    # Current time in Singapore
    now_sg = datetime.now(pytz.utc).astimezone(singapore_timezone)
    
    # Today's midnight in Singapore
    today_midnight_sg = now_sg.replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Yesterday's midnight in Singapore
    yesterday_midnight_sg = today_midnight_sg - timedelta(days=1)
    
    # Day before yesterday
    day_before_yesterday_sg = yesterday_midnight_sg - timedelta(days=1)
    
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

# --- UI SETUP ---
st.title("📊 User PNL Matrix Dashboard")
st.subheader("Performance Analysis by User (Singapore Time)")
st.write(f"Current Singapore Time: {now_sg.strftime('%Y-%m-%d %H:%M:%S')}")

# --- CACHING DATA FETCH FUNCTIONS ---
# These functions no longer take engine as parameter - get it inside

@st.cache_data(ttl=600)
def fetch_all_pairs():
    """Fetch all trading pairs"""
    query = "SELECT DISTINCT pair_name FROM public.trade_pool_pairs ORDER BY pair_name"
    
    try:
        engine = get_database_connection()
        if not engine:
            return ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT", "PEPE/USDT"]
            
        df = pd.read_sql(text(query), engine)
        
        if df.empty:
            return ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT", "PEPE/USDT"]
        
        return df['pair_name'].tolist()
    except Exception as e:
        st.error(f"Error fetching pairs: {e}")
        return ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT", "PEPE/USDT"]

@st.cache_data(ttl=600)
def fetch_top_users(limit=100):
    """Fetch top users by trading volume"""
    # Calculate the date range
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
        engine = get_database_connection()
        if not engine:
            return [f"user_{i}" for i in range(1, 11)]
            
        df = pd.read_sql(text(query), engine)
        
        if df.empty:
            return [f"user_{i}" for i in range(1, 11)]
        
        return df['user_identifier'].tolist()
    except Exception as e:
        st.error(f"Error fetching users: {e}")
        return [f"user_{i}" for i in range(1, 11)]

@st.cache_data(ttl=600)
def fetch_user_pnl_for_period(user_id, pair_name, start_time, end_time):
    """Fetch PNL data for a specific user, pair and time period"""
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
        AND "taker_account_id" = '{user_id}'
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
        AND "taker_account_id" = '{user_id}'
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
        AND "taker_account_id" = '{user_id}'
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
        AND "taker_account_id" = '{user_id}'
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
        engine = get_database_connection()
        if not engine:
            return {"pnl": 0, "trades": 0}
            
        df = pd.read_sql(text(query), engine)
        
        if df.empty:
            return {"pnl": 0, "trades": 0}
        
        return {
            "pnl": float(df.iloc[0]['user_total_pnl']),
            "trades": int(df.iloc[0]['trade_count'])
        }
    except Exception as e:
        # Log the error but return gracefully
        print(f"Error processing PNL for user {user_id} on {pair_name}: {e}")
        return {"pnl": 0, "trades": 0}

@st.cache_data(ttl=600)
def fetch_user_metadata(user_id):
    """Fetch metadata about a user"""
    query = f"""
    SELECT 
        MIN(created_at) as first_trade_date,
        TO_CHAR(MIN(created_at), 'YYYY-MM-DD') as first_trade_date_str,
        COUNT(*) as all_time_trades,
        SUM(ABS("deal_size" * "deal_price")) as all_time_volume
    FROM 
        "public"."trade_fill_fresh"
    WHERE 
        "taker_account_id" = '{user_id}'
        AND "taker_way" IN (1, 2, 3, 4)
    """
    
    try:
        engine = get_database_connection()
        if not engine:
            return {
                "first_trade_date": "Unknown",
                "all_time_trades": 0,
                "all_time_volume": 0,
                "account_age_days": 0
            }
            
        df = pd.read_sql(text(query), engine)
        
        if df.empty:
            return {
                "first_trade_date": "Unknown",
                "all_time_trades": 0,
                "all_time_volume": 0,
                "account_age_days": 0
            }
        
        first_trade = df.iloc[0]['first_trade_date']
        if first_trade:
            account_age = (now_utc - first_trade).days
        else:
            account_age = 0
            
        return {
            "first_trade_date": df.iloc[0]['first_trade_date_str'],
            "all_time_trades": int(df.iloc[0]['all_time_trades']),
            "all_time_volume": float(df.iloc[0]['all_time_volume']),
            "account_age_days": account_age
        }
    except Exception as e:
        print(f"Error fetching metadata for user {user_id}: {e}")
        return {
            "first_trade_date": "Unknown",
            "all_time_trades": 0,
            "all_time_volume": 0,
            "account_age_days": 0
        }

# --- LOAD INITIAL DATA ---
with st.spinner("Loading trading pairs and users..."):
    all_pairs = fetch_all_pairs()
    top_users = fetch_top_users(limit=100)

# --- CONTROL PANEL ---
st.sidebar.title("Dashboard Controls")

# Trading pair selector
st.sidebar.subheader("Trading Pairs")
select_all_pairs = st.sidebar.checkbox("Select All Pairs", value=False)

if select_all_pairs:
    selected_pairs = all_pairs
else:
    selected_pairs = st.sidebar.multiselect(
        "Select Trading Pairs", 
        all_pairs,
        default=all_pairs[:5] if len(all_pairs) > 5 else all_pairs
    )

# User selector
st.sidebar.subheader("Users")
st.sidebar.warning("⚠️ Performance Warning: Selecting more than 10 users may cause slow loading.")
user_limit = st.sidebar.slider(
    "Number of Top Users to Show", 
    min_value=5, 
    max_value=min(100, len(top_users)), 
    value=10,  # Reduced default from 25 to 10
    step=5
)

top_selected_users = top_users[:user_limit]

# Add a refresh button
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.experimental_rerun()

# Basic validation
if not selected_pairs:
    st.warning("Please select at least one trading pair")
    st.stop()

if not top_selected_users:
    st.warning("No active users found for the selected period")
    st.stop()

# Performance limitation (add this new section)
max_pairs = 5  # Limit to 5 pairs maximum for performance
if len(selected_pairs) > max_pairs:
    st.warning(f"⚠️ For better performance, only the first {max_pairs} pairs will be processed.")
    limited_pairs = selected_pairs[:max_pairs]
else:
    limited_pairs = selected_pairs

# --- DATA PROCESSING ---
# Show progress
progress_bar = st.progress(0)
status_text = st.empty()

# Get time boundaries
time_boundaries = get_time_boundaries()

# Initialize data structure
results = {}
periods = ["today", "yesterday", "day_before_yesterday", "this_week", "this_month", "all_time"]

# Process data - simpler sequential approach for reliability
total_combinations = len(top_selected_users) * len(limited_pairs) * len(periods)
processed_combinations = 0

# First fetch user metadata since it doesn't depend on pairs
for user_id in top_selected_users:
    status_text.text(f"Fetching metadata for user {user_id}...")
    progress = processed_combinations / total_combinations
    progress_bar.progress(progress)
    
    if user_id not in results:
        results[user_id] = {
            "user_id": user_id,
            "metadata": fetch_user_metadata(user_id),
            "pairs": {}
        }
    
    processed_combinations += 1

# Then process each user-pair combination
for user_id in top_selected_users:
    for pair_name in limited_pairs:
        status_text.text(f"Processing {user_id} - {pair_name}")
        
        if pair_name not in results[user_id]["pairs"]:
            results[user_id]["pairs"][pair_name] = {}
        
        # Process each time period for this user-pair combination
        for period in periods:
            processed_combinations += 1
            progress = processed_combinations / total_combinations
            progress_bar.progress(progress)
            
            start_time = time_boundaries[period]["start"]
            end_time = time_boundaries[period]["end"]
            
            # Fetch PNL data with properly memoized function
            period_data = fetch_user_pnl_for_period(user_id, pair_name, start_time, end_time)
            
            # Store results
            results[user_id]["pairs"][pair_name][period] = period_data

# Final progress update
progress_bar.progress(1.0)
status_text.text(f"Processed {len(results)} users across {len(limited_pairs)} pairs")

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

# Create DataFrames for display
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

# Calculate additional metrics for analysis
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
    """Color cells based on PNL value"""
    if pd.isna(val) or val == 0:
        return 'background-color: #f5f5f5; color: #666666;'  # Grey for missing/zero
    elif val < -1000:  # Large negative PNL (loss) - red
        return 'background-color: rgba(255, 0, 0, 0.9); color: white'
    elif val < 0:  # Small negative PNL (loss) - light red
        intensity = max(0, min(255, int(255 * abs(val) / 1000)))
        return f'background-color: rgba(255, {180-intensity}, {180-intensity}, 0.7); color: black'
    elif val < 1000:  # Small positive PNL (profit) - light green
        intensity = max(0, min(255, int(255 * val / 1000)))
        return f'background-color: rgba({180-intensity}, 255, {180-intensity}, 0.7); color: black'
    else:  # Large positive PNL (profit) - green
        return 'background-color: rgba(0, 200, 0, 0.8); color: black'

# --- CREATE DASHBOARD TABS ---
tab1, tab2, tab3 = st.tabs([
    "User PNL Matrix",
    "User Details",
    "Pair Analysis"
])

# Tab 1: User PNL Matrix
with tab1:
    st.header("User PNL Overview Matrix")
    
    # Add notice about limited data
    if len(selected_pairs) > len(limited_pairs):
        st.info(f"Showing data for {len(limited_pairs)} out of {len(selected_pairs)} selected pairs. For full analysis, select fewer pairs.")
    
    if user_matrix_df.empty:
        st.warning("No data available for selected users and pairs")
    else:
        # Filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_trades = st.number_input("Min Trades", value=0, min_value=0)
        
        with col2:
            show_only = st.selectbox(
                "Show Users", 
                ["All", "Profitable Today", "Unprofitable Today"]
            )
        
        with col3:
            sort_by = st.selectbox(
                "Sort By", 
                ["Today PNL", "Week PNL", "All Time PNL", "All Time Volume"]
            )
        
        # Apply filters
        filtered_df = user_matrix_df.copy()
        
        if min_trades > 0:
            filtered_df = filtered_df[filtered_df['All Time Trades'] >= min_trades]
        
        if show_only == "Profitable Today":
            filtered_df = filtered_df[filtered_df['Today PNL'] > 0]
        elif show_only == "Unprofitable Today":
            filtered_df = filtered_df[filtered_df['Today PNL'] < 0]
        
        # Sort the DataFrame
        filtered_df = filtered_df.sort_values(by=sort_by, ascending=False)
        
        # Display columns
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
        
        # Summary stats
        st.subheader("Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_today_pnl = filtered_df['Today PNL'].sum()
            profitable_users_today = len(filtered_df[filtered_df['Today PNL'] > 0])
            st.metric(
                "Total PNL Today", 
                f"${total_today_pnl:,.2f}",
                f"{profitable_users_today}/{len(filtered_df)} profitable"
            )
        
        with col2:
            total_yesterday_pnl = filtered_df['Yesterday PNL'].sum()
            st.metric(
                "Total PNL Yesterday", 
                f"${total_yesterday_pnl:,.2f}"
            )
        
        with col3:
            total_week_pnl = filtered_df['Week PNL'].sum()
            st.metric(
                "Total Week PNL", 
                f"${total_week_pnl:,.2f}"
            )
        
        with col4:
            total_all_time_pnl = filtered_df['All Time PNL'].sum()
            st.metric(
                "All Time Total PNL", 
                f"${total_all_time_pnl:,.2f}"
            )
        
        # Visualization of top performers
        st.subheader("Top and Bottom Users by PNL Today")
        
        # Get top 5 and bottom 5 performers
        top_5 = filtered_df.nlargest(5, 'Today PNL')
        bottom_5 = filtered_df.nsmallest(5, 'Today PNL')
        
        # Create visualization
        fig = go.Figure()
        
        # Top performers
        fig.add_trace(go.Bar(
            x=top_5['User ID'],
            y=top_5['Today PNL'],
            name='Top Performers',
            marker_color='green'
        ))
        
        # Bottom performers
        fig.add_trace(go.Bar(
            x=bottom_5['User ID'],
            y=bottom_5['Today PNL'],
            name='Bottom Performers',
            marker_color='red'
        ))
        
        fig.update_layout(
            title="Top and Bottom Users by PNL Today",
            xaxis_title="User ID",
            yaxis_title="PNL (USD)",
            barmode='group',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Tab 2: User Details
with tab2:
    st.header("User Performance Details")
    
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
        
        # Display user info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Account Age", f"{user_metadata['account_age_days']} days")
        with col2:
            st.metric("First Trade Date", user_metadata['first_trade_date'])
        with col3:
            st.metric("Total Trades", f"{user_metadata['all_time_trades']:,}")
        with col4:
            st.metric("Trading Volume", f"${user_metadata['all_time_volume']:,.2f}")
        
        # Display pair performance
        st.subheader(f"Trading Pairs Performance for User {selected_user_id}")
        
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
        
        # Create a visualization of PNL by trading pair
        st.subheader("PNL by Trading Pair")
        
        # Filter out pairs with zero PNL
        non_zero_pairs = user_pairs_df[user_pairs_df['All Time PNL'] != 0].copy()
        
        if not non_zero_pairs.empty:
            # Select top pairs by absolute PNL
            top_pairs = non_zero_pairs.reindex(
                non_zero_pairs['All Time PNL'].abs().sort_values(ascending=False).index
            ).head(10)
            
            # Create a bar chart
            fig = px.bar(
                top_pairs,
                x='Trading Pair',
                y=['Today PNL', 'Week PNL', 'All Time PNL'],
                title=f"PNL Comparison by Trading Pair for User {selected_user_id}",
                barmode='group',
                color_discrete_sequence=['#1f77b4', '#2ca02c', '#d62728']
            )
            
            fig.update_layout(
                xaxis_title="Trading Pair",
                yaxis_title="PNL (USD)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No non-zero PNL data available for this user.")

# Tab 3: Pair Analysis
with tab3:
    st.header("Trading Pair Performance Analysis")
    
    # Time period selector for analysis
    period = st.selectbox(
        "Time Period for Analysis",
        ["Today", "Yesterday", "Week", "Month", "All Time"],
        index=0
    )
    
    # Map selection to data column
    period_map = {
        "Today": "Today PNL",
        "Yesterday": "Yesterday PNL",
        "Week": "Week PNL",
        "Month": "Month PNL",
        "All Time": "All Time PNL"
    }
    
    selected_period = period_map[period]
    
    # Create a summary by trading pair
    pair_summary = user_pair_df.groupby('Trading Pair').agg({
        'Today PNL': 'sum',
        'Yesterday PNL': 'sum',
        'Week PNL': 'sum',
        'Month PNL': 'sum',
        'All Time PNL': 'sum',
        'Today Trades': 'sum',
        'Yesterday Trades': 'sum',
        'Week Trades': 'sum',
        'Month Trades': 'sum',
        'All Time Trades': 'sum'
    }).reset_index()
    
    # Calculate efficiency metrics
    pair_summary['PNL/Trade (Today)'] = (
        pair_summary['Today PNL'] / pair_summary['Today Trades']
    ).replace([np.inf, -np.inf, np.nan], 0)
    
    pair_summary['PNL/Trade (Week)'] = (
        pair_summary['Week PNL'] / pair_summary['Week Trades']
    ).replace([np.inf, -np.inf, np.nan], 0)
    
    pair_summary['PNL/Trade (All Time)'] = (
        pair_summary['All Time PNL'] / pair_summary['All Time Trades']
    ).replace([np.inf, -np.inf, np.nan], 0)
    
    # Sort by selected period
    pair_summary = pair_summary.sort_values(by=selected_period, ascending=False)
    
    # Display the pair summary
    st.subheader(f"Trading Pair Summary ({period})")
    
    # Display columns based on period
    if period == "Today":
        display_cols = [
            'Trading Pair', 'Today PNL', 'Today Trades', 'PNL/Trade (Today)', 
            'Yesterday PNL', 'Week PNL', 'All Time PNL'
        ]
    elif period == "Yesterday":
        display_cols = [
            'Trading Pair', 'Yesterday PNL', 'Yesterday Trades', 'Today PNL', 
            'Week PNL', 'All Time PNL'
        ]
    elif period == "Week":
        display_cols = [
            'Trading Pair', 'Week PNL', 'Week Trades', 'PNL/Trade (Week)', 
            'Today PNL', 'All Time PNL'
        ]
    else:
        display_cols = [
            'Trading Pair', selected_period, 'All Time Trades', 'PNL/Trade (All Time)', 
            'Today PNL', 'Week PNL'
        ]
    
    # Apply styling
    styled_summary = pair_summary[display_cols].style.applymap(
        color_pnl_cells, 
        subset=[col for col in display_cols if 'PNL' in col]
    ).format({
        'Today PNL': '${:,.2f}',
        'Yesterday PNL': '${:,.2f}',
        'Week PNL': '${:,.2f}',
        'Month PNL': '${:,.2f}',
        'All Time PNL': '${:,.2f}',
        'PNL/Trade (Today)': '${:,.2f}',
        'PNL/Trade (Week)': '${:,.2f}',
        'PNL/Trade (All Time)': '${:,.2f}',
        'Today Trades': '{:,}',
        'Week Trades': '{:,}',
        'All Time Trades': '{:,}'
    })
    
    st.dataframe(styled_summary, height=400, use_container_width=True)
    
    # Visualization of top pairs
    st.subheader(f"Top Trading Pairs by PNL ({period})")
    
    # Get top and bottom pairs
    top_pairs = pair_summary.nlargest(5, selected_period)
    bottom_pairs = pair_summary.nsmallest(5, selected_period)
    
    # Create visualization
    fig = go.Figure()
    
    # Top pairs
    fig.add_trace(go.Bar(
        x=top_pairs['Trading Pair'],
        y=top_pairs[selected_period],
        name='Top Pairs',
        marker_color='green'
    ))
    
    # Bottom pairs
    fig.add_trace(go.Bar(
        x=bottom_pairs['Trading Pair'],
        y=bottom_pairs[selected_period],
        name='Bottom Pairs',
        marker_color='red'
    ))
    
    fig.update_layout(
        title=f"Top and Bottom Trading Pairs by PNL ({period})",
        xaxis_title="Trading Pair",
        yaxis_title="PNL (USD)",
        barmode='group',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Create a heatmap showing pair popularity
    st.subheader("User-Pair Activity Heatmap")
    
    # Count number of users per pair
    pair_user_counts = user_pair_df.groupby('Trading Pair')['User ID'].nunique().reset_index()
    pair_user_counts.columns = ['Trading Pair', 'User Count']
    
    # Sort by count
    pair_user_counts = pair_user_counts.sort_values(by='User Count', ascending=False)
    
    # Create a bar chart
    fig = px.bar(
        pair_user_counts,
        x='Trading Pair',
        y='User Count',
        title="Number of Users per Trading Pair",
        color='User Count',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        xaxis_title="Trading Pair",
        yaxis_title="Number of Users",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Add footer with last update time
st.markdown("---")
st.markdown(f"*Last updated: {now_sg.strftime('%Y-%m-%d %H:%M:%S')} (Singapore Time)*")