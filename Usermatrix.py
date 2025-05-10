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
    page_title="User ID Analysis Dashboard",
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
st.title("User ID Analysis Dashboard")

# Set up the timezone
singapore_timezone = pytz.timezone('Asia/Singapore')
now_utc = datetime.now(pytz.utc)
now_sg = now_utc.astimezone(singapore_timezone)
st.write(f"Current Singapore Time: {now_sg.strftime('%Y-%m-%d %H:%M:%S')}")

# Create tabs
tab1, tab2, tab3 = st.tabs(["All Users", "User Trading Metrics", "User Behavior Analysis"])

# Function to fetch all user IDs from user_client table
@st.cache_data(ttl=600)
def fetch_all_users():
    query = """
    SELECT
      id,
      account_id,
      email,
      created_at + INTERVAL '8 hour' AS created_at,
      updated_at + INTERVAL '8 hour' AS updated_at,
      status
    FROM
      public.user_client
    ORDER BY
      created_at DESC;
    """
    
    try:
        df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        st.error(f"Error fetching user IDs from user_client: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(columns=['id', 'account_id', 'email', 'created_at', 'updated_at', 'status'])

# Function to fetch user metrics from trade data
@st.cache_data(ttl=600)
def fetch_user_metrics():
    query = """
    SELECT
      CONCAT(taker_account_id, '') AS user_id,
      COUNT(CASE WHEN taker_way IN (2, 4) AND taker_pnl > 0 THEN 1 END) AS winning_trades,
      COUNT(CASE WHEN taker_way IN (2, 4) AND taker_pnl < 0 THEN 1 END) AS losing_trades,
      CAST(
        COUNT(CASE WHEN taker_way IN (2, 4) AND taker_pnl > 0 THEN 1 END) AS FLOAT
      ) / NULLIF(
        COUNT(CASE WHEN taker_way IN (2, 4) THEN 1 END), 0
      ) * 100 AS win_percentage,
      SUM(
        CASE
          WHEN taker_way IN (1, 2, 3, 4) AND (taker_pnl > 0 OR taker_share_pnl > 0)
          THEN COALESCE(taker_pnl, 0) * COALESCE(collateral_price, 0) + COALESCE(taker_share_pnl, 0) * COALESCE(collateral_price, 0)
          ELSE 0
        END
      ) + SUM(
        CASE
          WHEN taker_way = 0 AND funding_fee > 0
          THEN funding_fee * collateral_price
          ELSE 0
        END
      ) AS total_profit,
      SUM(
        CASE
          WHEN taker_way IN (1, 2, 3, 4) AND taker_pnl < 0
          THEN COALESCE(taker_pnl, 0) * COALESCE(collateral_price, 0)
          ELSE 0
        END
      ) + SUM(
        CASE
          WHEN taker_fee_mode = 1 AND taker_way IN (1, 3)
          THEN -1 * COALESCE(taker_fee, 0) * COALESCE(collateral_price, 0)
          ELSE 0
        END
      ) + SUM(
        CASE
          WHEN taker_way = 0 AND funding_fee < 0
          THEN COALESCE(funding_fee, 0) * COALESCE(collateral_price, 0)
          ELSE 0
        END
      ) + SUM(
        - COALESCE(taker_sl_fee, 0) * COALESCE(collateral_price, 0) - COALESCE(maker_sl_fee, 0)
      ) AS total_loss,
      SUM(taker_share_pnl * collateral_price) AS profit_share,
      STRING_AGG(DISTINCT pair_name, ', ' ORDER BY pair_name) AS traded_pairs,
      AVG(deal_size) AS average_size,
      MAX(leverage) AS max_leverage,
      COUNT(*) AS total_trades,
      COUNT(CASE WHEN taker_mode = 4 THEN 1 END) AS liquidations,
      MIN(created_at + INTERVAL '8 hour') AS first_trade,
      MAX(created_at + INTERVAL '8 hour') AS last_trade
    FROM
      public.trade_fill_fresh
    GROUP BY
      user_id
    """
    
    try:
        df = pd.read_sql(query, conn)
        
        # Replace NaN values with 0 for numerical columns
        numeric_cols = ['winning_trades', 'losing_trades', 'win_percentage', 'total_profit', 
                        'profit_share', 'total_loss', 'average_size', 'max_leverage',
                        'total_trades', 'liquidations']
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        # Calculate net profit
        df['net_profit'] = df['total_profit'] + df['total_loss']
        
        # Calculate profit factor
        df['profit_factor'] = abs(df['total_profit']) / abs(df['total_loss'].replace(0, 1))
        
        return df
    except Exception as e:
        st.error(f"Error fetching user metrics: {e}")
        import traceback
        traceback.print_exc()
        return None

# Function to fetch wallet addresses from surfv2_user_login_log
@st.cache_data(ttl=600)
def fetch_wallet_addresses():
    query = """
    SELECT 
      CONCAT(account_id, '') AS user_id,
      address,
      MAX(created_at + INTERVAL '8 hour') AS last_login
    FROM 
      public.surfv2_user_login_log
    GROUP BY
      user_id, address
    ORDER BY
      last_login DESC
    """
    
    try:
        df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        st.error(f"Error fetching wallet addresses: {e}")
        return None

# Function to fetch detailed trade data for a specific user
@st.cache_data(ttl=600)
def fetch_user_trade_details(user_id):
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
      CONCAT(taker_account_id, '') = '{user_id}'
    ORDER BY
      created_at DESC
    LIMIT 1000
    """
    
    try:
        df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        st.error(f"Error fetching trade details for user {user_id}: {e}")
        return None

# Load data
with st.spinner("Loading user data..."):
    users_df = fetch_all_users()
    user_metrics_df = fetch_user_metrics()
    wallet_df = fetch_wallet_addresses()

# Merge dataframes to get complete user data
if users_df is not None and user_metrics_df is not None:
    # First, convert account_id to string in users_df to match user_id format in user_metrics_df
    users_df['account_id_str'] = users_df['account_id'].astype(str)
    
    # Merge users_df with user_metrics_df
    complete_user_df = pd.merge(
        users_df, 
        user_metrics_df, 
        left_on='account_id_str', 
        right_on='user_id', 
        how='left'
    )
    
    # Replace NaN values with 0 for metrics
    numeric_cols = ['winning_trades', 'losing_trades', 'win_percentage', 'total_profit', 
                    'profit_share', 'total_loss', 'average_size', 'max_leverage',
                    'total_trades', 'liquidations', 'net_profit', 'profit_factor']
    complete_user_df[numeric_cols] = complete_user_df[numeric_cols].fillna(0)
    
    # If we have wallet data, add it too
    if wallet_df is not None:
        # Group wallet df by user_id and get the most recent address
        wallet_latest = wallet_df.sort_values('last_login', ascending=False).drop_duplicates('user_id')
        
        # Merge the wallet addresses into the complete user dataframe
        complete_user_df = pd.merge(
            complete_user_df, 
            wallet_latest[['user_id', 'address', 'last_login']], 
            left_on='account_id_str', 
            right_on='user_id', 
            how='left',
            suffixes=('', '_wallet')
        )
else:
    st.error("Failed to load user data.")
    complete_user_df = None

# Display data in tabs
if complete_user_df is not None:
    # Tab 1 - All Users
    with tab1:
        st.header("All Users")
        
        # Add search and filter options
        st.subheader("Search and Filter")
        col1, col2 = st.columns(2)
        
        with col1:
            search_id = st.text_input("Search by User ID/Email")
        
        with col2:
            filter_active = st.selectbox("Filter by Status", ["All", "Active", "Inactive"])
        
        # Apply filters
        filtered_df = complete_user_df.copy()
        
        if search_id:
            filtered_df = filtered_df[
                filtered_df['account_id_str'].str.contains(search_id, case=False, na=False) |
                filtered_df['email'].str.contains(search_id, case=False, na=False)
            ]
        
        if filter_active != "All":
            if filter_active == "Active":
                filtered_df = filtered_df[filtered_df['status'] == 1]
            else:
                filtered_df = filtered_df[filtered_df['status'] != 1]
        
        # Display user table with pagination
        st.subheader(f"User List ({len(filtered_df)} users)")
        
        # Create a clean display dataframe
        display_cols = ['id', 'account_id', 'email', 'status', 'created_at', 'address', 'total_trades', 'net_profit']
        display_df = filtered_df[display_cols].copy()
        
        # Format status
        display_df['status'] = display_df['status'].map({1: 'Active', 0: 'Inactive'})
        
        # Format dates
        if 'created_at' in display_df.columns:
            display_df['created_at'] = pd.to_datetime(display_df['created_at']).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Format numbers
        if 'net_profit' in display_df.columns:
            display_df['net_profit'] = display_df['net_profit'].round(2)
        
        # Show dataframe with pagination
        st.dataframe(display_df, use_container_width=True)
        
        # Add download button
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download All Users CSV",
            data=csv,
            file_name="user_list.csv",
            mime="text/csv"
        )
        
        # Display basic stats
        st.subheader("User Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Users", len(users_df))
        
        with col2:
            active_users = len(users_df[users_df['status'] == 1])
            st.metric("Active Users", active_users)
        
        with col3:
            trading_users = len(user_metrics_df[user_metrics_df['total_trades'] > 0])
            st.metric("Trading Users", trading_users)
        
        with col4:
            profitable_users = len(user_metrics_df[user_metrics_df['net_profit'] > 0])
            st.metric("Profitable Users", profitable_users)
        
        # Show registration chart
        st.subheader("User Registration Over Time")
        
        # Convert to datetime and create monthly registration data
        users_df['created_at'] = pd.to_datetime(users_df['created_at'])
        users_df['year_month'] = users_df['created_at'].dt.strftime('%Y-%m')
        
        # Group by month and count
        monthly_registrations = users_df.groupby('year_month').size().reset_index(name='count')
        monthly_registrations = monthly_registrations.sort_values('year_month')
        
        # Create bar chart
        fig = px.bar(
            monthly_registrations, 
            x='year_month', 
            y='count',
            labels={'count': 'New Users', 'year_month': 'Month'},
            title='Monthly User Registrations'
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2 - User Trading Metrics
    with tab2:
        st.header("User Trading Metrics")
        
        # Filter to only users who have traded
        trading_df = complete_user_df[complete_user_df['total_trades'] > 0].copy()
        
        if len(trading_df) > 0:
            # Add filters
            st.subheader("Filter Trading Users")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                min_trades = st.number_input("Min Trades", min_value=0, value=0)
            
            with col2:
                sort_by = st.selectbox(
                    "Sort By", 
                    ["net_profit", "total_trades", "win_percentage", "profit_factor", "max_leverage"]
                )
            
            with col3:
                sort_order = st.selectbox("Sort Order", ["Descending", "Ascending"])
            
            # Apply filters
            filtered_trading_df = trading_df[trading_df['total_trades'] >= min_trades]
            
            # Sort the dataframe
            is_ascending = sort_order == "Ascending"
            filtered_trading_df = filtered_trading_df.sort_values(sort_by, ascending=is_ascending)
            
            # Display trading metrics table
            st.subheader(f"Trading Metrics ({len(filtered_trading_df)} users)")
            
            # Select columns for display
            metrics_display_cols = [
                'account_id', 'email', 'address', 'total_trades', 'winning_trades', 
                'losing_trades', 'win_percentage', 'total_profit', 'total_loss', 
                'net_profit', 'profit_factor', 'max_leverage', 'average_size',
                'liquidations', 'first_trade', 'last_trade'
            ]
            
            # Create display dataframe with formatted values
            metrics_display_df = filtered_trading_df[metrics_display_cols].copy()
            
            # Format numbers and dates
            for col in ['win_percentage', 'total_profit', 'total_loss', 'net_profit', 'profit_factor', 'average_size']:
                if col in metrics_display_df.columns:
                    metrics_display_df[col] = metrics_display_df[col].round(2)
            
            for col in ['first_trade', 'last_trade']:
                if col in metrics_display_df.columns:
                    metrics_display_df[col] = pd.to_datetime(metrics_display_df[col]).dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Display dataframe
            st.dataframe(metrics_display_df, use_container_width=True)
            
            # Add download button
            metrics_csv = filtered_trading_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Trading Metrics CSV",
                data=metrics_csv,
                file_name="user_trading_metrics.csv",
                mime="text/csv"
            )
            
            # Display trading performance visualizations
            st.subheader("Trading Performance Distribution")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Win percentage distribution
                fig = px.histogram(
                    filtered_trading_df,
                    x="win_percentage",
                    nbins=20,
                    title="Win Percentage Distribution",
                    labels={"win_percentage": "Win Percentage (%)"}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Net profit distribution
                fig = px.histogram(
                    filtered_trading_df,
                    x="net_profit",
                    nbins=20,
                    title="Net Profit Distribution (USD)",
                    labels={"net_profit": "Net Profit (USD)"}
                )
                fig.update_layout(xaxis_range=[filtered_trading_df['net_profit'].min(), filtered_trading_df['net_profit'].max()])
                st.plotly_chart(fig, use_container_width=True)
            
            # Display trading activity over time
            st.subheader("Trading Activity")
            
            # Create scatter plot of trades vs profit
            fig = px.scatter(
                filtered_trading_df,
                x="total_trades",
                y="net_profit",
                size="max_leverage",
                color="win_percentage",
                hover_name="account_id",
                hover_data=["email", "win_percentage", "profit_factor"],
                title="Trading Activity vs Performance",
                labels={
                    "total_trades": "Total Trades",
                    "net_profit": "Net Profit (USD)",
                    "max_leverage": "Max Leverage",
                    "win_percentage": "Win Percentage (%)"
                },
                color_continuous_scale=px.colors.sequential.Viridis,
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show top users
            st.subheader("Top 10 Users by Net Profit")
            top_profit_df = trading_df.nlargest(10, 'net_profit')[['account_id', 'net_profit', 'win_percentage', 'total_trades']]
            
            fig = px.bar(
                top_profit_df,
                x="account_id",
                y="net_profit",
                title="Top 10 Users by Net Profit",
                labels={"account_id": "User ID", "net_profit": "Net Profit (USD)"}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No users with trading activity found.")
    
    # Tab 3 - User Behavior Analysis
    with tab3:
        st.header("User Behavior Analysis")
        
        # User selection dropdown
        users_with_trades = complete_user_df[complete_user_df['total_trades'] > 0]
        
        if len(users_with_trades) > 0:
            # Create a list of options with user ID and email
            user_options = []
            for _, row in users_with_trades.iterrows():
                display_text = f"{row['account_id_str']}"
                if pd.notna(row['email']) and row['email'] != '':
                    display_text += f" - {row['email']}"
                user_options.append({"label": display_text, "value": row['account_id_str']})
            
            selected_user = st.selectbox(
                "Select User to Analyze",
                options=[opt["value"] for opt in user_options],
                format_func=lambda x: next((opt["label"] for opt in user_options if opt["value"] == x), x)
            )
            
            # Get user metrics
            user_data = complete_user_df[complete_user_df['account_id_str'] == selected_user].iloc[0]
            
            # Fetch user trade details
            user_trades = fetch_user_trade_details(selected_user)
            
            # Display user summary
            st.subheader(f"User Summary: {selected_user}")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Trades", user_data['total_trades'])
                st.metric("Win Percentage", f"{user_data['win_percentage']:.2f}%")
            
            with col2:
                st.metric("Net Profit", f"${user_data['net_profit']:.2f}")
                st.metric("Profit Factor", f"{user_data['profit_factor']:.2f}")
            
            with col3:
                st.metric("Max Leverage", f"{user_data['max_leverage']:.2f}x")
                st.metric("Avg Position Size", f"{user_data['average_size']:.2f}")
            
            with col4:
                st.metric("Liquidations", user_data['liquidations'])
                st.metric("Traded Pairs", len(str(user_data['traded_pairs']).split(',')))
            
            if user_trades is not None and len(user_trades) > 0:
                # Display trade history
                st.subheader("Trade History")
                st.dataframe(user_trades, use_container_width=True)
                
                # Plot profit/loss timeline
                st.subheader("Profit/Loss Timeline")
                
                user_trades['trade_time'] = pd.to_datetime(user_trades['trade_time'])
                user_trades = user_trades.sort_values('trade_time')
                
                # Calculate cumulative P&L
                user_trades['cumulative_pnl'] = user_trades['pnl_usd'].cumsum()
                
                fig = px.line(
                    user_trades,
                    x="trade_time",
                    y="cumulative_pnl",
                    title="Cumulative Profit/Loss Over Time",
                    labels={"trade_time": "Trade Time", "cumulative_pnl": "Cumulative P&L (USD)"}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show trade type distribution
                st.subheader("Trade Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Trade direction distribution
                    direction_counts = user_trades['trade_type'].value_counts().reset_index()
                    direction_counts.columns = ['Trade Type', 'Count']
                    
                    fig = px.pie(
                        direction_counts,
                        values='Count',
                        names='Trade Type',
                        title="Trade Direction Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Order type distribution
                    order_counts = user_trades['order_type'].value_counts().reset_index()
                    order_counts.columns = ['Order Type', 'Count']
                    
                    fig = px.pie(
                        order_counts,
                        values='Count',
                        names='Order Type',
                        title="Order Type Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Show traded pairs
                st.subheader("Traded Pairs")
                
                # Get top traded pairs
                pair_counts = user_trades['pair_name'].value_counts().reset_index()
                pair_counts.columns = ['Pair', 'Count']
                
                # Calculate profit by pair
                pair_profit = user_trades.groupby('pair_name')['pnl_usd'].sum().reset_index()
                pair_profit.columns = ['Pair', 'Net Profit']
                
                # Merge data
                pair_analysis = pd.merge(pair_counts, pair_profit, on='Pair', how='left')
                
                # Display table and chart
                col1, col2 = st.columns(2)
                
                with col1:
                    st.dataframe(pair_analysis, use_container_width=True)
                
                with col2:
                    fig = px.bar(
                        pair_analysis,
                        x="Pair",
                        y="Net Profit",
                        title="Profit/Loss by Trading Pair",
                        labels={"Pair": "Trading Pair", "Net Profit": "Net Profit (USD)"}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Show trading by time of day
                st.subheader("Trading Patterns by Time")
                
                # Extract hour and weekday from trade time
                user_trades['hour'] = user_trades['trade_time'].dt.hour
                user_trades['weekday'] = user_trades['trade_time'].dt.day_name()
                
                # Create heatmap of trading activity by hour and weekday
                hour_day_counts = user_trades.groupby(['weekday', 'hour']).size().reset_index(name='count')
                
                # Define weekday order
                weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                
                # Create pivot table
                hour_day_pivot = hour_day_counts.pivot(index='weekday', columns='hour', values='count')
                hour_day_pivot = hour_day_pivot.reindex(weekday_order)
                
                # Fill NaN with 0
                hour_day_pivot = hour_day_pivot.fillna(0)
                
                # Create heatmap
                fig = px.imshow(
                    hour_day_pivot,
                    labels=dict(x="Hour of Day", y="Day of Week", color="Number of Trades"),
                    x=hour_day_pivot.columns,
                    y=hour_day_pivot.index,
                    title="Trading Activity by Hour and Day",
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show leverage distribution
                st.subheader("Leverage Analysis")
                
                leverage_data = user_trades['leverage'].value_counts().reset_index()
                leverage_data.columns = ['Leverage', 'Count']
                leverage_data = leverage_data.sort_values('Leverage')
                
                fig = px.bar(
                    leverage_data,
                    x="Leverage",
                    y="Count",
                    title="Leverage Distribution",
                    labels={"Leverage": "Leverage Used", "Count": "Number of Trades"}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show position size distribution
                st.subheader("Position Size Analysis")
                
                # Group position sizes into bins
                user_trades['size_bin'] = pd.cut(
                    user_trades['deal_size'],
                    bins=10,
                    labels=[f"Bin {i+1}" for i in range(10)]
                )
                
                size_data = user_trades.groupby('size_bin').agg(
                    count=('deal_size', 'count'),
                    min_size=('deal_size', 'min'),
                    max_size=('deal_size', 'max'),
                    avg_pnl=('pnl_usd', 'mean')
                ).reset_index()
                
                size_data['size_range'] = size_data.apply(
                    lambda x: f"{x['min_size']:.2f} - {x['max_size']:.2f}", axis=1
                )
                
                # Create dual-axis chart
                fig = go.Figure()
                
                # Add position count bars
                fig.add_trace(go.Bar(
                    x=size_data['size_range'],
                    y=size_data['count'],
                    name='Number of Trades',
                    marker_color='blue'
                ))
                
                # Add average PNL line
                fig.add_trace(go.Scatter(
                    x=size_data['size_range'],
                    y=size_data['avg_pnl'],
                    name='Average P&L',
                    marker_color='red',
                    mode='lines+markers',
                    yaxis='y2'
                ))
                
                # Update layout
                fig.update_layout(
                    title="Position Size Distribution vs Average P&L",
                    xaxis_title="Position Size Range",
                    yaxis_title="Number of Trades",
                    yaxis2=dict(
                        title="Average P&L (USD)",
                        overlaying="y",
                        side="right"
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No trade data available for this user.")
        else:
            st.warning("No users with trading activity found.")

# Add sidebar with info
st.sidebar.title("Dashboard Info")
st.sidebar.info("""
This dashboard provides comprehensive analysis of user IDs and trading behavior.

**Features:**
- View all user IDs from the system
- Analyze user trading metrics and patterns
- Identify user behavior and trading patterns
- Sort and filter users by various criteria
- Download data for further analysis

For any issues, please contact the data team.
""")

# Add refresh button
if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()
    st.experimental_rerun()

# Show last update time
st.sidebar.markdown(f"*Last refreshed: {now_sg.strftime('%Y-%m-%d %H:%M:%S')} (SGT)*")