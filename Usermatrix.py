import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import psycopg2
import warnings
import pytz

# Suppress warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="User Behavior Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Always clear cache at startup to ensure fresh data
st.cache_data.clear()

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
st.title("User Behavior Analysis Dashboard")

# Set up the timezone
singapore_timezone = pytz.timezone('Asia/Singapore')
now_utc = datetime.now(pytz.utc)
now_sg = now_utc.astimezone(singapore_timezone)
st.write(f"Current Singapore Time: {now_sg.strftime('%Y-%m-%d %H:%M:%S')}")

# Create tabs
tab1, tab2, tab3 = st.tabs(["All Users", "Trading Metrics", "User Analysis"])

# Function to fetch all users
@st.cache_data(ttl=600)
def fetch_all_users():
    query = """
    SELECT
      account_id,
      total_points,
      login_days,
      referral_code,
      referrer_id,
      referral_num,
      referral_size,
      remark
    FROM
      public.user_info
    ORDER BY
      account_id;
    """
    
    try:
        df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        st.error(f"Error fetching user data: {e}")
        import traceback
        traceback.print_exc()
        return None

# Function to fetch user client status
@st.cache_data(ttl=600)
def fetch_user_client_status():
    query = """
    SELECT
      id,
      create_way,
      created_at + INTERVAL '8 hour' AS created_at,
      updated_at + INTERVAL '8 hour' AS updated_at,
      is_enable,
      can_order,
      can_withdraw,
      role_type
    FROM
      public.user_client
    ORDER BY
      id;
    """
    
    try:
        df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        st.error(f"Error fetching user client status: {e}")
        return None

# Function to fetch user trading metrics
@st.cache_data(ttl=600)
def fetch_user_trading_metrics():
    query = """
    SELECT
      taker_account_id,
      COUNT(*) AS total_trades,
      COUNT(CASE WHEN taker_way IN (2, 4) AND taker_pnl > 0 THEN 1 END) AS winning_trades,
      COUNT(CASE WHEN taker_way IN (2, 4) AND taker_pnl < 0 THEN 1 END) AS losing_trades,
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
      taker_account_id
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
        
        return df
    except Exception as e:
        st.error(f"Error fetching trading metrics: {e}")
        import traceback
        traceback.print_exc()
        return None

# Function to fetch detailed trade data for a specific user
@st.cache_data(ttl=600)
def fetch_user_trade_details(account_id):
    query = f"""
    SELECT
      pair_name,
      taker_way,
      CASE
        WHEN taker_way = 1 THEN 'Open Long'
        WHEN taker_way = 2 THEN 'Close Short'
        WHEN taker_way = 3 THEN 'Open Short'
        WHEN taker_way = 4 THEN 'Close Long'
      END AS trade_type,
      taker_mode,
      CASE
        WHEN taker_mode = 1 THEN 'Active'
        WHEN taker_mode = 2 THEN 'Take Profit'
        WHEN taker_mode = 3 THEN 'Stop Loss'
        WHEN taker_mode = 4 THEN 'Liquidation'
      END AS order_type,
      deal_price,
      deal_size,
      leverage,
      taker_pnl * collateral_price AS pnl_usd,
      taker_share_pnl * collateral_price AS profit_share_usd,
      created_at + INTERVAL '8 hour' AS trade_time
    FROM
      public.trade_fill_fresh
    WHERE
      taker_account_id = {account_id}
    ORDER BY
      created_at DESC
    LIMIT 1000
    """
    
    try:
        df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        st.error(f"Error fetching trade details for user {account_id}: {e}")
        return None

# Function to count users per day
@st.cache_data(ttl=600)
def fetch_users_per_day():
    query = """
    SELECT
      DATE(created_at + INTERVAL '8 hour') AS date,
      COUNT(*) AS new_users
    FROM
      public.user_client
    GROUP BY
      DATE(created_at + INTERVAL '8 hour')
    ORDER BY
      date;
    """
    
    try:
        df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        st.error(f"Error fetching users per day: {e}")
        return None

# Load data
with st.spinner("Loading user data..."):
    users_df = fetch_all_users()
    user_status_df = fetch_user_client_status()
    trading_metrics_df = fetch_user_trading_metrics()
    users_per_day_df = fetch_users_per_day()

# Combine the data
if users_df is not None and trading_metrics_df is not None:
    # Convert account_id to string to ensure matching formats
    users_df['account_id_str'] = users_df['account_id'].astype(str)
    trading_metrics_df['taker_account_id_str'] = trading_metrics_df['taker_account_id'].astype(str)
    
    # Merge user info with trading metrics
    merged_df = pd.merge(
        users_df, 
        trading_metrics_df, 
        left_on='account_id_str', 
        right_on='taker_account_id_str', 
        how='left'
    )
    
    # Add user client status if available
    if user_status_df is not None:
        user_status_df['id_str'] = user_status_df['id'].astype(str)
        merged_df = pd.merge(
            merged_df,
            user_status_df,
            left_on='account_id_str',
            right_on='id_str',
            how='left',
            suffixes=('', '_client')
        )
    
    # Fill NaN values for users who haven't traded
    trade_cols = ['total_trades', 'winning_trades', 'losing_trades', 'win_percentage', 
                 'total_profit', 'total_loss', 'net_pnl', 'profit_share',
                 'avg_leverage', 'max_leverage', 'avg_position_size', 'max_position_size',
                 'liquidations', 'profit_factor', 'num_pairs']
    
    for col in trade_cols:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].fillna(0)
    
    # Display in tabs
    with tab1:
        st.header("All Users")
        
        # Add search and filter options
        st.subheader("Search and Filter")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            search_id = st.text_input("Search by Account ID")
        
        with col2:
            if 'is_enable' in merged_df.columns:
                status_options = ["All", "Active (can trade)", "Inactive"]
                filter_status = st.selectbox("Filter by Status", status_options)
            else:
                filter_status = "All"
        
        with col3:
            trader_filter = st.selectbox(
                "Filter by Trading Activity", 
                ["All", "Has traded", "Never traded"]
            )
        
        # Apply filters
        filtered_df = merged_df.copy()
        
        if search_id:
            filtered_df = filtered_df[filtered_df['account_id_str'].str.contains(search_id, na=False)]
        
        if filter_status != "All" and 'is_enable' in filtered_df.columns and 'can_order' in filtered_df.columns:
            if filter_status == "Active (can trade)":
                filtered_df = filtered_df[(filtered_df['is_enable'] == True) & (filtered_df['can_order'] == True)]
            elif filter_status == "Inactive":
                filtered_df = filtered_df[(filtered_df['is_enable'] == False) | (filtered_df['can_order'] == False)]
        
        if trader_filter != "All":
            if trader_filter == "Has traded":
                filtered_df = filtered_df[filtered_df['total_trades'] > 0]
            else:  # "Never traded"
                filtered_df = filtered_df[filtered_df['total_trades'] == 0]
        
        # Display users
        st.subheader(f"User List ({len(filtered_df)} users)")
        
        # Select columns to display based on what's available
        display_cols = ['account_id']
        
        # Add user info columns if available
        if 'login_days' in filtered_df.columns:
            display_cols.append('login_days')
        if 'total_points' in filtered_df.columns:
            display_cols.append('total_points')
        if 'referral_code' in filtered_df.columns:
            display_cols.append('referral_code')
        
        # Add status columns if available
        if 'is_enable' in filtered_df.columns:
            display_cols.append('is_enable')
        if 'can_order' in filtered_df.columns:
            display_cols.append('can_order')
        if 'created_at' in filtered_df.columns:
            display_cols.append('created_at')
        
        # Add trading metrics
        if 'total_trades' in filtered_df.columns:
            display_cols.append('total_trades')
        if 'net_pnl' in filtered_df.columns:
            display_cols.append('net_pnl')
        
        # Create display dataframe
        display_df = filtered_df[display_cols].copy()
        
        # Format numeric columns
        numeric_cols = ['total_trades', 'net_pnl']
        for col in numeric_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].map(lambda x: f"{x:.2f}" if isinstance(x, float) else x)
        
        # Show the dataframe
        st.dataframe(display_df, use_container_width=True)
        
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
            total_users = len(users_df)
            st.metric("Total Users", f"{total_users:,}")
        
        with col2:
            if 'is_enable' in merged_df.columns and 'can_order' in merged_df.columns:
                active_users = len(merged_df[(merged_df['is_enable'] == True) & (merged_df['can_order'] == True)])
                st.metric("Active Users", f"{active_users:,}")
            else:
                st.metric("Active Users", "N/A")
        
        with col3:
            if 'total_trades' in merged_df.columns:
                trading_users = len(merged_df[merged_df['total_trades'] > 0])
                st.metric("Users Who Have Traded", f"{trading_users:,}")
            else:
                st.metric("Users Who Have Traded", "N/A")
        
        with col4:
            if 'net_pnl' in merged_df.columns:
                profitable_users = len(merged_df[merged_df['net_pnl'] > 0])
                st.metric("Profitable Users", f"{profitable_users:,}")
            else:
                st.metric("Profitable Users", "N/A")
        
        # User registration chart
        st.subheader("User Registration Over Time")
        
        if users_per_day_df is not None and len(users_per_day_df) > 0:
            # Create line chart
            fig = px.line(
                users_per_day_df, 
                x='date', 
                y='new_users',
                labels={'new_users': 'New Users', 'date': 'Date'},
                title='Daily New User Registrations'
            )
            
            # Add markers to the line
            fig.update_traces(mode='lines+markers')
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Create cumulative chart
            users_per_day_df['cumulative_users'] = users_per_day_df['new_users'].cumsum()
            
            fig2 = px.line(
                users_per_day_df, 
                x='date', 
                y='cumulative_users',
                labels={'cumulative_users': 'Total Users', 'date': 'Date'},
                title='Cumulative User Growth'
            )
            
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No user registration data available.")
    
    with tab2:
        st.header("Trading Metrics")
        
        # Filter to only users who have traded
        if 'total_trades' in merged_df.columns:
            traders_df = merged_df[merged_df['total_trades'] > 0].copy()
            
            if len(traders_df) > 0:
                # Add filters
                st.subheader("Filter Trading Users")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    min_trades = st.number_input("Min Trades", min_value=0, value=0)
                
                with col2:
                    sort_options = ["total_trades", "net_pnl", "win_percentage", "profit_factor", "max_leverage"]
                    sort_options = [opt for opt in sort_options if opt in traders_df.columns]
                    sort_by = st.selectbox("Sort By", sort_options, index=0)
                
                with col3:
                    sort_order = st.selectbox("Sort Order", ["Descending", "Ascending"])
                
                # Apply filters
                if min_trades > 0:
                    traders_df = traders_df[traders_df['total_trades'] >= min_trades]
                
                # Sort the dataframe
                is_ascending = sort_order == "Ascending"
                traders_df = traders_df.sort_values(sort_by, ascending=is_ascending)
                
                # Display trading metrics table
                st.subheader(f"Trading Metrics ({len(traders_df)} users)")
                
                # Create clean display dataframe
                metrics_cols = ['account_id']
                
                # Add available trading metrics
                for col in ['total_trades', 'winning_trades', 'losing_trades', 'win_percentage', 
                            'total_profit', 'total_loss', 'net_pnl', 'profit_share',
                            'avg_leverage', 'max_leverage', 'avg_position_size',
                            'liquidations', 'profit_factor', 'num_pairs']:
                    if col in traders_df.columns:
                        metrics_cols.append(col)
                
                metrics_df = traders_df[metrics_cols].copy()
                
                # Format numeric columns
                for col in metrics_cols:
                    if col in ['win_percentage', 'avg_leverage', 'profit_factor']:
                        metrics_df[col] = metrics_df[col].map(lambda x: f"{x:.2f}" if isinstance(x, float) else x)
                    elif col in ['total_profit', 'total_loss', 'net_pnl', 'profit_share', 'avg_position_size']:
                        metrics_df[col] = metrics_df[col].map(lambda x: f"{x:.2f}" if isinstance(x, float) else x)
                
                # Show the dataframe
                st.dataframe(metrics_df, use_container_width=True)
                
                # Add download button
                csv = traders_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Trading Metrics CSV",
                    data=csv,
                    file_name="trading_metrics.csv",
                    mime="text/csv"
                )
                
                # Performance distribution
                st.subheader("Performance Distribution")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'win_percentage' in traders_df.columns:
                        fig = px.histogram(
                            traders_df,
                            x='win_percentage',
                            nbins=20,
                            title='Win Percentage Distribution',
                            labels={'win_percentage': 'Win Percentage (%)'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if 'net_pnl' in traders_df.columns:
                        fig = px.histogram(
                            traders_df,
                            x='net_pnl',
                            nbins=20,
                            title='Net PnL Distribution',
                            labels={'net_pnl': 'Net PnL (USD)'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # Scatter plot of trades vs performance
                st.subheader("Trading Activity vs Performance")
                
                if all(col in traders_df.columns for col in ['total_trades', 'net_pnl']):
                    scatter_fig = px.scatter(
                        traders_df,
                        x='total_trades',
                        y='net_pnl',
                        size='max_leverage' if 'max_leverage' in traders_df.columns else None,
                        color='win_percentage' if 'win_percentage' in traders_df.columns else None,
                        hover_name='account_id',
                        hover_data=['total_trades', 'net_pnl', 'win_percentage', 'profit_factor'],
                        title='Trading Activity vs Performance',
                        labels={
                            'total_trades': 'Total Trades',
                            'net_pnl': 'Net PnL (USD)',
                            'max_leverage': 'Max Leverage',
                            'win_percentage': 'Win Percentage (%)'
                        }
                    )
                    st.plotly_chart(scatter_fig, use_container_width=True)
                
                # Top users by net PnL
                st.subheader("Top 10 Users by Net PnL")
                
                if 'net_pnl' in traders_df.columns:
                    top_users = traders_df.nlargest(10, 'net_pnl')
                    
                    fig = px.bar(
                        top_users,
                        x='account_id',
                        y='net_pnl',
                        title='Top 10 Users by Net PnL',
                        labels={'account_id': 'Account ID', 'net_pnl': 'Net PnL (USD)'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No users with trading activity found.")
        else:
            st.warning("Trading metrics data is not available.")
    
    with tab3:
        st.header("User Analysis")
        
        # User selection
        if 'total_trades' in merged_df.columns:
            traders = merged_df[merged_df['total_trades'] > 0]
            
            if len(traders) > 0:
                selected_user = st.selectbox(
                    "Select User for Analysis",
                    options=traders['account_id'].unique(),
                    format_func=lambda x: f"Account ID: {x}"
                )
                
                # Get user data
                user_data = merged_df[merged_df['account_id'] == selected_user].iloc[0]
                
                # Fetch detailed trade data
                user_trades = fetch_user_trade_details(selected_user)
                
                # Display user summary
                st.subheader(f"User Summary: {selected_user}")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if 'total_trades' in user_data:
                        st.metric("Total Trades", f"{user_data['total_trades']:.0f}")
                    if 'win_percentage' in user_data:
                        st.metric("Win Percentage", f"{user_data['win_percentage']:.2f}%")
                
                with col2:
                    if 'net_pnl' in user_data:
                        st.metric("Net PnL", f"${user_data['net_pnl']:.2f}")
                    if 'profit_factor' in user_data:
                        st.metric("Profit Factor", f"{user_data['profit_factor']:.2f}")
                
                with col3:
                    if 'max_leverage' in user_data:
                        st.metric("Max Leverage", f"{user_data['max_leverage']:.2f}x")
                    if 'avg_position_size' in user_data:
                        st.metric("Avg Position Size", f"{user_data['avg_position_size']:.2f}")
                
                with col4:
                    if 'liquidations' in user_data:
                        st.metric("Liquidations", f"{user_data['liquidations']:.0f}")
                    if 'num_pairs' in user_data:
                        st.metric("Trading Pairs", f"{user_data['num_pairs']:.0f}")
                
                # Add more user details
                with st.expander("Additional User Details"):
                    if 'login_days' in user_data:
                        st.write(f"Login Days: {user_data['login_days']}")
                    if 'total_points' in user_data:
                        st.write(f"Total Points: {user_data['total_points']}")
                    if 'referral_code' in user_data:
                        st.write(f"Referral Code: {user_data['referral_code']}")
                    if 'referral_num' in user_data:
                        st.write(f"Referrals: {user_data['referral_num']}")
                    if 'created_at' in user_data:
                        st.write(f"Account Created: {user_data['created_at']}")
                    if 'is_enable' in user_data:
                        st.write(f"Account Enabled: {'Yes' if user_data['is_enable'] else 'No'}")
                    if 'can_order' in user_data:
                        st.write(f"Can Place Orders: {'Yes' if user_data['can_order'] else 'No'}")
                
                if user_trades is not None and len(user_trades) > 0:
                    # Trade history
                    st.subheader("Trade History")
                    st.dataframe(user_trades, use_container_width=True)
                    
                    # PnL timeline
                    st.subheader("PnL Timeline")
                    
                    # Convert to datetime
                    user_trades['trade_time'] = pd.to_datetime(user_trades['trade_time'])
                    
                    # Sort chronologically
                    user_trades = user_trades.sort_values('trade_time')
                    
                    # Calculate cumulative PnL
                    user_trades['cumulative_pnl'] = user_trades['pnl_usd'].cumsum()
                    
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
                    
                    # Trade analysis
                    st.subheader("Trade Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Trade type distribution
                        trade_types = user_trades['trade_type'].value_counts().reset_index()
                        trade_types.columns = ['Trade Type', 'Count']
                        
                        fig = px.pie(
                            trade_types,
                            values='Count',
                            names='Trade Type',
                            title='Trade Type Distribution'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Order type distribution
                        order_types = user_trades['order_type'].value_counts().reset_index()
                        order_types.columns = ['Order Type', 'Count']
                        
                        fig = px.pie(
                            order_types,
                            values='Count',
                            names='Order Type',
                            title='Order Type Distribution'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Trading pairs analysis
                    st.subheader("Trading Pairs Analysis")
                    
                    # Get performance by pair
                    pair_performance = user_trades.groupby('pair_name').agg(
                        count=('pnl_usd', 'count'),
                        total_pnl=('pnl_usd', 'sum'),
                        avg_pnl=('pnl_usd', 'mean'),
                        win_rate=('pnl_usd', lambda x: (x > 0).mean() * 100),
                        avg_leverage=('leverage', 'mean')
                    ).reset_index()
                    
                    # Sort by trade count
                    pair_performance = pair_performance.sort_values('count', ascending=False)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("Performance by Trading Pair")
                        st.dataframe(pair_performance, use_container_width=True)
                    
                    with col2:
                        # Create bar chart of PnL by pair
                        fig = px.bar(
                            pair_performance,
                            x='pair_name',
                            y='total_pnl',
                            title='PnL by Trading Pair',
                            labels={'pair_name': 'Trading Pair', 'total_pnl': 'Total PnL (USD)'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Leverage analysis
                    st.subheader("Leverage Analysis")
                    
                    # Group by leverage
                    leverage_groups = pd.cut(user_trades['leverage'], bins=10)
                    leverage_analysis = user_trades.groupby(leverage_groups).agg(
                        count=('leverage', 'count'),
                        avg_pnl=('pnl_usd', 'mean'),
                        total_pnl=('pnl_usd', 'sum')
                    ).reset_index()
                    
                    leverage_analysis['leverage'] = leverage_analysis['leverage'].astype(str)
                    
                    # Create dual axis chart
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    fig.add_trace(
                        go.Bar(
                            x=leverage_analysis['leverage'],
                            y=leverage_analysis['count'],
                            name='Number of Trades'
                        ),
                        secondary_y=False
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=leverage_analysis['leverage'],
                            y=leverage_analysis['avg_pnl'],
                            name='Average PnL',
                            mode='lines+markers'
                        ),
                        secondary_y=True
                    )
                    
                    fig.update_layout(
                        title='Leverage Analysis',
                        xaxis_title='Leverage Range'
                    )
                    
                    fig.update_yaxes(title_text='Number of Trades', secondary_y=False)
                    fig.update_yaxes(title_text='Average PnL (USD)', secondary_y=True)
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No trade data available for this user.")
            else:
                st.warning("No users with trading activity found.")
        else:
            st.warning("User trading data is not available.")
else:
    st.error("Failed to load user data.")

# Add refresh button in sidebar
st.sidebar.title("Dashboard Controls")
if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()
    st.experimental_rerun()

# Add dashboard info
st.sidebar.title("About This Dashboard")
st.sidebar.info("""
This dashboard provides comprehensive analysis of user behavior in the trading platform.

**Features:**
- View all users and their basic information
- Analyze trading metrics and patterns
- Detailed user-level analysis
- Interactive charts and visualizations

Data is sourced from the replication_report database.
""")

# Show last update time
st.sidebar.markdown(f"*Last updated: {now_sg.strftime('%Y-%m-%d %H:%M:%S')} (SGT)*")

# Import missing libraries
from plotly.subplots import make_subplots