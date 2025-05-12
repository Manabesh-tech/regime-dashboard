import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import psycopg2
import warnings
import pytz
from plotly.subplots import make_subplots

# Suppress warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="User Trading Behavior Analysis",
    page_icon="📊",
    layout="wide"
)

# Configure database
def init_db_connection():
    # DB parameters - these should be stored in Streamlit secrets in production
    db_params = {
        'host': 'aws-jp-tk-surf-pg-public.cluster-csteuf9lw8dv.ap-northeast-1.rds.amazonaws.com',
        'port': 5432,
        'database': 'replication_report',
        'user': 'public_replication',
        'password': '866^FKC4hllk'
    }
    
    try:
        conn = psycopg2.connect(
            host=db_params['host'],
            port=db_params['port'],
            database=db_params['database'],
            user=db_params['user'],
            password=db_params['password']
        )
        return conn, db_params
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return None, db_params

# Initialize connection
conn, db_params = init_db_connection()

# Main title
st.title("User Trading Behavior Analysis Dashboard")
st.caption("This dashboard analyzes trading patterns and behaviors for users")

# Set up the timezone
singapore_timezone = pytz.timezone('Asia/Singapore')
now_utc = datetime.now(pytz.utc)
now_sg = now_utc.astimezone(singapore_timezone)
st.write(f"Current Singapore Time: {now_sg.strftime('%Y-%m-%d %H:%M:%S')}")

# Create tabs
tab1, tab2, tab3 = st.tabs(["All Users", "Trading Metrics", "User Analysis"])

# Function to fetch user trading metrics with improved trade type breakdown
@st.cache_data(ttl=600)
def fetch_trading_metrics():
    query = """
    SELECT
      taker_account_id,
      CONCAT(taker_account_id, '') AS user_id_str,
      COUNT(*) AS total_trades,
      COUNT(CASE WHEN taker_way IN (2, 4) AND taker_pnl > 0 THEN 1 END) AS winning_trades,
      COUNT(CASE WHEN taker_way IN (2, 4) AND taker_pnl < 0 THEN 1 END) AS losing_trades,
      COUNT(CASE WHEN taker_way IN (2, 4) AND taker_pnl = 0 THEN 1 END) AS break_even_trades,
      COUNT(CASE WHEN taker_way = 1 THEN 1 END) AS open_long_count,
      COUNT(CASE WHEN taker_way = 2 THEN 1 END) AS close_short_count,
      COUNT(CASE WHEN taker_way = 3 THEN 1 END) AS open_short_count,
      COUNT(CASE WHEN taker_way = 4 THEN 1 END) AS close_long_count,
      COUNT(CASE WHEN taker_way IN (1, 3) THEN 1 END) AS opening_positions,
      COUNT(CASE WHEN taker_way IN (2, 4) THEN 1 END) AS closing_positions,
      CAST(
        COUNT(CASE WHEN taker_way IN (2, 4) AND taker_pnl > 0 THEN 1 END) AS FLOAT
      ) / NULLIF(
        COUNT(CASE WHEN taker_way IN (2, 4) THEN 1 END), 0
      ) * 100 AS win_percentage,
      SUM(
        CASE
          WHEN taker_pnl > 0 THEN taker_pnl * collateral_price
          ELSE 0
        END
      ) AS total_profit,
      SUM(
        CASE
          WHEN taker_pnl < 0 THEN taker_pnl * collateral_price
          ELSE 0
        END
      ) AS total_loss,
      SUM(taker_pnl * collateral_price) AS net_pnl,
      SUM(taker_share_pnl * collateral_price) AS profit_share,
      AVG(leverage) AS avg_leverage,
      MAX(leverage) AS max_leverage,
      AVG(deal_size) AS avg_position_size,
      MAX(deal_size) AS max_position_size,
      COUNT(CASE WHEN taker_mode = 4 THEN 1 END) AS liquidations,
      STRING_AGG(DISTINCT pair_name, ', ') AS traded_pairs,
      MIN(created_at + INTERVAL '8 hour') AS first_trade,
      MAX(created_at + INTERVAL '8 hour') AS last_trade
    FROM
      public.trade_fill_fresh
    GROUP BY
      taker_account_id, user_id_str
    ORDER BY
      total_trades DESC;
    """
    
    try:
        df = pd.read_sql(query, conn)
        
        # Calculate derived metrics
        df['profit_factor'] = df.apply(
            lambda x: abs(x['total_profit']) / abs(x['total_loss']) if x['total_loss'] != 0 else 
                     (float('inf') if x['total_profit'] > 0 else 0), 
            axis=1
        )
        
        # Count number of different pairs traded
        df['num_pairs'] = df['traded_pairs'].apply(lambda x: len(x.split(',')) if isinstance(x, str) else 0)
        
        # Calculate trade reconciliation (to explain total_trades != winning + losing)
        df['reconciled_trades'] = df['winning_trades'] + df['losing_trades'] + df['break_even_trades'] + df['opening_positions']
        df['unreconciled_count'] = df['total_trades'] - df['reconciled_trades']
        
        return df
    except Exception as e:
        st.error(f"Error fetching trading metrics: {e}")
        import traceback
        traceback.print_exc()
        return None

# Function to fetch detailed trade data for a specific user
@st.cache_data(ttl=600, show_spinner=False)
def fetch_user_trade_details_v3(user_id):
    query = f"""
    SELECT
      pair_name,
      taker_way,
      CASE
        WHEN taker_way = 0 THEN 'Funding Fee'
        WHEN taker_way = 1 THEN 'Open Long'
        WHEN taker_way = 2 THEN 'Close Short'
        WHEN taker_way = 3 THEN 'Open Short'
        WHEN taker_way = 4 THEN 'Close Long'
      END AS trade_type,
      CASE
        WHEN taker_way IN (1, 3) THEN 'Entry'
        WHEN taker_way IN (2, 4) THEN 'Exit'
        ELSE 'Fee'
      END AS position_action,
      taker_mode,
      taker_fee_mode,
      CASE
        WHEN taker_mode = 1 THEN 'Active'
        WHEN taker_mode = 2 THEN 'Take Profit'
        WHEN taker_mode = 3 THEN 'Stop Loss'
        WHEN taker_mode = 4 THEN 'Liquidation'
      END AS order_type,
      deal_price AS entry_exit_price,
      leverage,
      leverage || 'x' AS leverage_display,
      deal_size AS size,
      deal_vol AS volume,
      collateral_amount AS collateral,
      collateral_price,
      taker_pnl,
      taker_share_pnl,
      taker_fee,
      taker_sl_fee,
      maker_sl_fee,
      funding_fee,
      -- Calculate values to match Metabase exactly:
      -- User Received PNL (no trade_pnl column!)
      ROUND(taker_pnl * collateral_price, 2) AS user_received_pnl,
      -- Platform Profit Share
      ROUND(taker_share_pnl * collateral_price, 2) AS profit_share,
      -- Calculate profit share percentage: profit_share / (user_received_pnl + profit_share)
      CASE 
        WHEN (taker_pnl * collateral_price + taker_share_pnl * collateral_price) != 0 THEN 
          ROUND((taker_share_pnl * collateral_price) / (taker_pnl * collateral_price + taker_share_pnl * collateral_price) * 100, 2)
        ELSE 0
      END AS profit_share_percent,
      created_at + INTERVAL '8 hour' AS trade_time,
      created_at
    FROM
      public.trade_fill_fresh
    WHERE
      CONCAT(taker_account_id, '') = '{user_id}'
    ORDER BY
      created_at DESC
    LIMIT 1000
    """
    
    try:
        df = pd.read_sql(query, conn)
        
        # Add post-processing calculations
        df['pre_profit_share_exit'] = None
        df['post_profit_share_exit'] = None
        df['percent_distance'] = None
        df['matched_entry_price'] = None
        
        # For exit trades, calculate pre/post profit share prices
        exit_mask = df['position_action'] == 'Exit'
        
        # Match entry prices for exits and calculate % distance
        for idx, row in df[exit_mask].iterrows():
            # Find the most recent entry for this pair
            entry_trades = df[(df['pair_name'] == row['pair_name']) & 
                            (df['position_action'] == 'Entry') & 
                            (df['created_at'] < row['created_at'])]
            
            if not entry_trades.empty:
                entry_price = entry_trades.iloc[0]['entry_exit_price']
                df.loc[idx, 'matched_entry_price'] = entry_price
                
                # Calculate % distance: (Pbefore - Pentry) / Pentry * 100
                # Where Pbefore is the exit price before profit share
                exit_price = row['entry_exit_price']
                if entry_price != 0:
                    # Calculate the pre-profit share price
                    if row['size'] != 0:
                        pre_profit_share_price = exit_price - (row['profit_share'] / row['size'])
                        df.loc[idx, 'pre_profit_share_exit'] = pre_profit_share_price
                        df.loc[idx, 'post_profit_share_exit'] = exit_price
                        df.loc[idx, 'percent_distance'] = round(((pre_profit_share_price - entry_price) / entry_price) * 100, 2)
                    else:
                        df.loc[idx, 'percent_distance'] = round(((exit_price - entry_price) / entry_price) * 100, 2)
        
        return df
    except Exception as e:
        st.error(f"Error fetching trade details for user {user_id}: {e}")
        import traceback
        traceback.print_exc()
        return None

# Function to count users per day (based on first trade date)
@st.cache_data(ttl=600)
def fetch_users_per_day():
    query = """
    SELECT
      DATE(MIN(created_at) + INTERVAL '8 hour') AS date,
      CONCAT(taker_account_id, '') AS user_id_str
    FROM
      public.trade_fill_fresh
    GROUP BY
      user_id_str
    ORDER BY
      date;
    """
    
    try:
        df = pd.read_sql(query, conn)
        # Aggregate to count users per day
        date_counts = df.groupby('date').size().reset_index(name='new_users')
        return date_counts
    except Exception as e:
        st.error(f"Error fetching users per day: {e}")
        return None

# Load data
with st.spinner("Loading user data..."):
    trading_metrics_df = fetch_trading_metrics()
    users_per_day_df = fetch_users_per_day()

# Check if we have data
if trading_metrics_df is not None:
    # Tab 1 - All Users
    with tab1:
        st.header("All Users")
        
        # Add search and filter options
        st.subheader("Search and Filter")
        col1, col2 = st.columns(2)
        
        with col1:
            search_id = st.text_input("Search by User ID")
        
        with col2:
            trader_filter = st.selectbox(
                "Filter by Trading Activity", 
                ["All", "High Activity (>100 trades)", "Medium Activity (10-100 trades)", "Low Activity (<10 trades)"]
            )
        
        # Apply filters
        filtered_df = trading_metrics_df.copy()
        
        if search_id:
            filtered_df = filtered_df[filtered_df['user_id_str'].str.contains(search_id, na=False)]
        
        if trader_filter != "All":
            if trader_filter == "High Activity (>100 trades)":
                filtered_df = filtered_df[filtered_df['total_trades'] > 100]
            elif trader_filter == "Medium Activity (10-100 trades)":
                filtered_df = filtered_df[(filtered_df['total_trades'] >= 10) & (filtered_df['total_trades'] <= 100)]
            else:  # Low Activity
                filtered_df = filtered_df[filtered_df['total_trades'] < 10]
        
        # Display users
        st.subheader(f"User List ({len(filtered_df)} users)")
        
        # Select columns to display - now with trade type breakdown
        display_cols = ['user_id_str', 'total_trades', 'winning_trades', 'losing_trades', 
                        'opening_positions', 'closing_positions', 'win_percentage', 'net_pnl']
        
        # Create display dataframe
        display_df = filtered_df[display_cols].copy()
        
        # Format columns
        display_df['net_pnl'] = display_df['net_pnl'].round(2)
        display_df['win_percentage'] = display_df['win_percentage'].round(2)
        
        # Show the dataframe
        st.dataframe(display_df, use_container_width=True)
        
        # Add explanation about trade types
        st.info("""
        **Understanding Trade Counts:**
        - **Total Trades**: All transactions (opening + closing positions)
        - **Winning Trades**: Only profitable position closes (taker_way 2 or 4 with positive PnL)
        - **Losing Trades**: Only unprofitable position closes (taker_way 2 or 4 with negative PnL)
        - **Opening Positions**: Trades that open new positions (taker_way 1 or 3)
        - **Closing Positions**: Trades that close positions (taker_way 2 or 4)
        
        This explains why Total Trades ≠ Winning Trades + Losing Trades (the difference is opening positions).
        """)
        
        # Add download button
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="user_list.csv",
            mime="text/csv"
        )
        
        # User statistics
        st.subheader("User Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_users = len(trading_metrics_df)
            st.metric("Total Users", f"{total_users:,}")
        
        with col2:
            active_users = len(trading_metrics_df[trading_metrics_df['total_trades'] > 10])
            st.metric("Active Users (>10 trades)", f"{active_users:,}")
        
        with col3:
            total_volume = trading_metrics_df['total_trades'].sum()
            st.metric("Total Trades", f"{total_volume:,}")
        
        with col4:
            profitable_users = len(trading_metrics_df[trading_metrics_df['net_pnl'] > 0])
            st.metric("Profitable Users", f"{profitable_users:,}")
        
        # User activity over time
        st.subheader("User Activity Over Time")
        
        if users_per_day_df is not None and len(users_per_day_df) > 0:
            try:
                # Create line chart
                fig = px.line(
                    users_per_day_df, 
                    x='date', 
                    y='new_users',
                    labels={'new_users': 'New Users', 'date': 'Date'},
                    title='Daily New User First Trades'
                )
                
                # Add markers to the line
                fig.update_traces(mode='lines+markers')
                
                # Display the chart
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating user activity chart: {e}")
                st.write("Debug info:")
                st.write(f"Available columns: {list(users_per_day_df.columns)}")
                st.write(f"Data types: {users_per_day_df.dtypes}")
                st.write("First few rows:")
                st.dataframe(users_per_day_df.head())
            
            try:
                # Create cumulative chart
                users_per_day_df['cumulative_users'] = users_per_day_df['new_users'].cumsum()
                
                fig2 = px.line(
                    users_per_day_df, 
                    x='date', 
                    y='cumulative_users',
                    labels={'cumulative_users': 'Total Users', 'date': 'Date'},
                    title='Cumulative User Growth (Based on First Trade)'
                )
                
                st.plotly_chart(fig2, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating cumulative users chart: {e}")
                st.write("Debug info:")
                st.write(f"Available columns: {list(users_per_day_df.columns)}")
                st.write(f"Data types: {users_per_day_df.dtypes}")
                st.write("First few rows:")
                st.dataframe(users_per_day_df.head())
        else:
            st.info("No user activity data available.")
    
    # Tab 2 - Trading Metrics
    with tab2:
        st.header("Trading Metrics")
        
        # Add filters
        st.subheader("Filter Trading Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_trades = st.number_input("Min Trades", min_value=0, value=0)
        
        with col2:
            sort_options = ["total_trades", "net_pnl", "win_percentage", "profit_factor", "opening_positions"]
            sort_by = st.selectbox("Sort By", sort_options, index=0)
        
        with col3:
            sort_order = st.selectbox("Sort Order", ["Descending", "Ascending"])
        
        # Apply filters
        metrics_df = trading_metrics_df.copy()
        
        if min_trades > 0:
            metrics_df = metrics_df[metrics_df['total_trades'] >= min_trades]
        
        # Sort the dataframe
        is_ascending = sort_order == "Ascending"
        metrics_df = metrics_df.sort_values(sort_by, ascending=is_ascending)
        
        # Display trading metrics table
        st.subheader(f"Trading Metrics ({len(metrics_df)} users)")
        
        # Create clean display dataframe with improved trade type breakdown
        metrics_cols = ['user_id_str', 'total_trades', 'winning_trades', 'losing_trades',
                        'opening_positions', 'closing_positions', 'win_percentage',
                        'total_profit', 'total_loss', 'net_pnl', 'profit_factor']
        
        metrics_display = metrics_df[metrics_cols].copy()
        
        # Format numeric columns
        for col in ['win_percentage', 'profit_factor']:
            metrics_display[col] = metrics_display[col].round(2)
        
        for col in ['total_profit', 'total_loss', 'net_pnl']:
            metrics_display[col] = metrics_display[col].round(2)
        
        # Show the dataframe
        st.dataframe(metrics_display, use_container_width=True)
        
        # Add download button
        csv = metrics_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Metrics CSV",
            data=csv,
            file_name="trading_metrics.csv",
            mime="text/csv"
        )
        
        # Performance distribution
        st.subheader("Performance Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                metrics_df,
                x='win_percentage',
                nbins=20,
                title='Win Percentage Distribution',
                labels={'win_percentage': 'Win Percentage (%)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Limit to reasonable range for better visualization
            filtered_pnl = metrics_df[metrics_df['net_pnl'].between(
                metrics_df['net_pnl'].quantile(0.05),
                metrics_df['net_pnl'].quantile(0.95)
            )]
            
            fig = px.histogram(
                filtered_pnl,
                x='net_pnl',
                nbins=20,
                title='Net PnL Distribution (5-95 percentile)',
                labels={'net_pnl': 'Net PnL (USD)'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3 - User Analysis
    with tab3:
        st.header("User Analysis")
        
        # Direct user search option
        search_col1, search_col2 = st.columns([3, 1])
        
        with search_col1:
            direct_user_search = st.text_input("Enter User ID to Analyze", 
                                              placeholder="Enter exact user ID here...")
        
        with search_col2:
            search_button = st.button("Search User", use_container_width=True)
        
        # Process direct search
        if direct_user_search and search_button:
            # Check if user exists
            user_exists = direct_user_search in trading_metrics_df['user_id_str'].values
            
            if user_exists:
                selected_user = direct_user_search
            else:
                st.error(f"User ID {direct_user_search} not found. Please check the ID and try again.")
                # Fall back to dropdown selection
                selected_user = st.selectbox(
                    "Select User from Dropdown Instead",
                    options=trading_metrics_df['user_id_str'].tolist(),
                    format_func=lambda x: f"User ID: {x}"
                )
        else:
            # Regular dropdown selection if no direct search
            selected_user = st.selectbox(
                "Or Select User from Dropdown",
                options=trading_metrics_df['user_id_str'].tolist(),
                format_func=lambda x: f"User ID: {x}"
            )
        
        # Get user data
        user_data = trading_metrics_df[trading_metrics_df['user_id_str'] == selected_user].iloc[0]
        
        # Fetch detailed trade data with new function name to bust cache
        user_trades = fetch_user_trade_details_v3(selected_user)
        
        # Display user summary
        st.subheader(f"User Summary: {selected_user}")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", f"{user_data['total_trades']:.0f}")
            st.metric("Win Percentage", f"{user_data['win_percentage']:.2f}%")
        
        with col2:
            st.metric("Net PnL", f"${user_data['net_pnl']:.2f}")
            st.metric("Profit Factor", f"{user_data['profit_factor']:.2f}")
        
        with col3:
            st.metric("Max Leverage", f"{user_data['max_leverage']:.2f}x")
            st.metric("Avg Position Size", f"{user_data['avg_position_size']:.2f}")
        
        with col4:
            st.metric("Liquidations", f"{user_data['liquidations']:.0f}")
            st.metric("Trading Pairs", f"{user_data['num_pairs']:.0f}")
        
        if user_trades is not None and len(user_trades) > 0:
            # Trade history
            st.subheader("Trade History")
            
            # Debug mode checkbox
            show_debug = st.checkbox("Show Debug Info", value=False)
            
            if show_debug:
                st.write("Debug Information:")
                st.write("Available columns in user_trades:")
                st.write(list(user_trades.columns))
                
                # Check if old columns exist
                if 'trade_pnl' in user_trades.columns:
                    st.warning("⚠️ Old column 'trade_pnl' still exists. Please click 'Force Clear Cache'.")
                
                # Just show the dataframe as is, without trying to select specific columns
                st.write("First 5 trades:")
                st.dataframe(user_trades.head(5))
            
            # Create two views - compact and detailed
            view_option = st.radio("Select View", ["Compact View", "Detailed View"], horizontal=True)
            
            if view_option == "Compact View":
                # Display enhanced compact view
                # Define columns we want to show - User Received PNL, not Trade PNL
                compact_cols = ['trade_time', 'pair_name', 'trade_type', 'position_action', 
                                'entry_exit_price', 'size', 'leverage_display', 
                                'user_received_pnl', 'profit_share', 'profit_share_percent']
                
                # Add percent_distance only if it exists (for exit trades)
                if 'percent_distance' in user_trades.columns:
                    compact_cols.append('percent_distance')
                
                # Filter out columns that might not exist
                available_cols = [col for col in compact_cols if col in user_trades.columns]
                
                # If we don't have the new columns, skip the table display
                if not available_cols:
                    st.error("⚠️ Column structure is outdated. Please click 'Force Clear Cache' in the sidebar and refresh the page.")
                else:
                    # Create display dataframe
                    display_df = user_trades[available_cols].copy()
                    
                    # Create column mapping
                    col_mapping = {
                        'trade_time': 'Trade Time',
                        'pair_name': 'Pair',
                        'trade_type': 'Trade Type',
                        'position_action': 'Action',
                        'entry_exit_price': 'Entry/Exit Price',
                        'size': 'Size',
                        'leverage_display': 'Leverage',
                        'user_received_pnl': 'User Received PNL',
                        'profit_share': 'Profit Share',
                        'profit_share_percent': 'Profit Share %',
                        'percent_distance': '% Distance'
                    }
                    
                    # Rename columns
                    display_df.columns = [col_mapping.get(col, col) for col in display_df.columns]
                    
                    # Format numeric columns - all PnL values to 2 decimals
                    if 'Entry/Exit Price' in display_df.columns:
                        display_df['Entry/Exit Price'] = display_df['Entry/Exit Price'].round(4)
                    if 'Size' in display_df.columns:
                        display_df['Size'] = display_df['Size'].round(2)
                    if 'User Received PNL' in display_df.columns:
                        display_df['User Received PNL'] = display_df['User Received PNL'].round(2)
                    if 'Profit Share' in display_df.columns:
                        display_df['Profit Share'] = display_df['Profit Share'].round(2)
                    if 'Profit Share %' in display_df.columns:
                        display_df['Profit Share %'] = display_df['Profit Share %'].round(2)
                    if '% Distance' in display_df.columns:
                        display_df['% Distance'] = display_df['% Distance'].round(2)
                    
                    st.dataframe(display_df, use_container_width=True)
            
            # PnL timeline with better error handling
            st.subheader("PnL Timeline")
            
            try:
                # Convert to datetime and sort
                user_trades['trade_time'] = pd.to_datetime(user_trades['trade_time'])
                user_trades = user_trades.sort_values('trade_time').reset_index(drop=True)
                
                # Make sure user_received_pnl is numeric
                user_trades['user_received_pnl'] = pd.to_numeric(user_trades['user_received_pnl'], errors='coerce').fillna(0)
                
                # Simple cumulative sum for exit trades only
                user_trades['pnl_contribution'] = 0.0
                exit_mask = user_trades['position_action'] == 'Exit'
                user_trades.loc[exit_mask, 'pnl_contribution'] = user_trades.loc[exit_mask, 'user_received_pnl']
                
                # Calculate cumulative PnL
                user_trades['cumulative_pnl'] = user_trades['pnl_contribution'].cumsum()
                
                # Debug info
                if show_debug:
                    st.write("Cumulative PnL Debug:")
                    st.write(f"Number of exit trades: {exit_mask.sum()}")
                    st.write(f"Sum of pnl_contribution: {user_trades['pnl_contribution'].sum():.2f}")
                    st.write(f"First few cumulative values: {user_trades['cumulative_pnl'].head()}")
                
                # Create line chart
                fig = px.line(
                    user_trades,
                    x='trade_time',
                    y='cumulative_pnl',
                    title='Cumulative PnL Over Time',
                    labels={'trade_time': 'Trade Time', 'cumulative_pnl': 'Cumulative PnL (USD)'}
                )
                
                # Add markers at each trade
                fig.update_traces(mode='lines+markers')
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Verify cumulative PnL
                final_cumulative_pnl = user_trades['cumulative_pnl'].iloc[-1]
                expected_net_pnl = user_data['net_pnl']
                sum_user_received_pnl = user_trades[exit_mask]['user_received_pnl'].sum()
                
                st.info(f"""
                **PnL Verification:**
                - Sum of User Received PnL (Exit trades only): ${sum_user_received_pnl:.2f}
                - Final Cumulative PnL: ${final_cumulative_pnl:.2f}
                - Expected Net PnL (from metrics): ${expected_net_pnl:.2f}
                - Difference: ${abs(final_cumulative_pnl - expected_net_pnl):.2f}
                
                The cumulative PnL is the sum of user_received_pnl for exit trades only.
                """)
                
            except Exception as e:
                st.error(f"Error creating PnL timeline: {str(e)}")
                st.write("Available columns:", list(user_trades.columns))
                if 'cumulative_pnl' in user_trades.columns:
                    st.write("cumulative_pnl exists but has issues")
                    st.write(f"Sample data: {user_trades[['trade_time', 'cumulative_pnl']].head()}")
                import traceback
                st.code(traceback.format_exc())
            
            # Add export functionality
            if st.button("Export Trade History to CSV"):
                # Prepare export data with all details
                export_df = user_trades.copy()
                
                # Rearrange columns for export
                export_cols = ['trade_time', 'pair_name', 'trade_type', 'position_action',
                              'order_type', 'leverage_display', 'size', 'collateral',
                              'entry_exit_price', 'matched_entry_price',
                              'pre_profit_share_exit', 'post_profit_share_exit',
                              'percent_distance', 'user_received_pnl', 
                              'profit_share', 'profit_share_percent']
                
                available_export_cols = [col for col in export_cols if col in export_df.columns]
                csv = export_df[available_export_cols].to_csv(index=False).encode('utf-8')
                
                st.download_button(
                    label="Download Trade History",
                    data=csv,
                    file_name=f"trade_history_{selected_user}.csv",
                    mime="text/csv"
                )
        else:
            st.warning("No trade data available for this user.")
else:
    st.error("Failed to load user data.")

# Add refresh button in sidebar
st.sidebar.title("Dashboard Controls")
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()
with col2:
    if st.button("Force Clear Cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Cache cleared!")
        st.rerun()

# Add dashboard info
st.sidebar.title("About This Dashboard")
st.sidebar.info("""
This dashboard provides comprehensive analysis of user behavior in the trading platform.

**Features:**
- View all users and their trading metrics
- Analyze performance by win rate, PnL, and other metrics
- Detailed user-level analysis with trading history
- Interactive charts and visualizations
- Direct user ID search

**Understanding PnL Calculations (Matching Metabase):**
- User Received PNL = taker_pnl * collateral_price
- Profit Share = taker_share_pnl * collateral_price
- Profit Share % = profit_share / (user_received_pnl + profit_share) * 100
- % Distance = (Pbefore - Pentry) / Pentry * 100

**Net PnL Calculation:**
The user's net PnL is the sum of User Received PNL from all exit trades.

Data is sourced from the trade_fill_fresh table in the database.
""")

# Show last update time
st.sidebar.markdown(f"*Last updated: {now_sg.strftime('%Y-%m-%d %H:%M:%S')} (SGT)*")