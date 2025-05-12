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
@st.cache_data(ttl=600)
def fetch_user_trade_details(user_id):
    query = f"""
    WITH trade_data AS (
      SELECT
        pair_name,
        taker_way,
        CASE
          WHEN taker_way = 1 THEN 'Open Long'
          WHEN taker_way = 2 THEN 'Close Short'
          WHEN taker_way = 3 THEN 'Open Short'
          WHEN taker_way = 4 THEN 'Close Long'
        END AS trade_type,
        CASE
          WHEN taker_way IN (1, 3) THEN 'Entry'
          ELSE 'Exit'
        END AS position_action,
        taker_mode,
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
        collateral_amount AS collateral,
        taker_pnl * collateral_price AS trade_pnl,
        taker_share_pnl * collateral_price AS profit_share_usd,
        CASE 
          WHEN taker_pnl * collateral_price != 0 THEN 
            (taker_share_pnl * collateral_price) / (taker_pnl * collateral_price) * 100
          ELSE 0
        END AS profit_share_percent,
        created_at + INTERVAL '8 hour' AS trade_time,
        created_at,
        taker_account_id
      FROM
        public.trade_fill_fresh
      WHERE
        CONCAT(taker_account_id, '') = '{user_id}'
    ),
    entry_prices AS (
      SELECT
        taker_account_id,
        pair_name,
        created_at as entry_time,
        deal_price as entry_price,
        ROW_NUMBER() OVER (PARTITION BY taker_account_id, pair_name ORDER BY created_at DESC) as rn
      FROM
        public.trade_fill_fresh
      WHERE
        CONCAT(taker_account_id, '') = '{user_id}'
        AND taker_way IN (1, 3)
    )
    SELECT
      t.*,
      CASE 
        WHEN t.position_action = 'Exit' AND t.profit_share_usd > 0 THEN 
          t.entry_exit_price - (t.profit_share_usd / t.size)
        WHEN t.position_action = 'Exit' AND t.profit_share_usd < 0 THEN 
          t.entry_exit_price - (t.profit_share_usd / t.size)
        ELSE NULL
      END AS pre_profit_share_exit,
      CASE 
        WHEN t.position_action = 'Exit' THEN 
          t.entry_exit_price
        ELSE NULL
      END AS post_profit_share_exit,
      CASE 
        WHEN t.position_action = 'Exit' AND e.entry_price IS NOT NULL THEN 
          ((t.entry_exit_price - (t.profit_share_usd / t.size) - e.entry_price) / e.entry_price) * 100
        ELSE NULL
      END AS percent_move,
      e.entry_price AS matched_entry_price
    FROM
      trade_data t
    LEFT JOIN LATERAL (
      SELECT entry_price
      FROM entry_prices e
      WHERE e.taker_account_id = t.taker_account_id
        AND e.pair_name = t.pair_name
        AND e.entry_time < t.created_at
      ORDER BY e.entry_time DESC
      LIMIT 1
    ) e ON true
    ORDER BY
      t.created_at DESC
    LIMIT 1000
    """
    
    try:
        df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        st.error(f"Error fetching trade details for user {user_id}: {e}")
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

# Try to access user_client table (but don't fail if not available)
@st.cache_data(ttl=600)
def fetch_user_client_info():
    try:
        query = """
        SELECT
          id,
          CONCAT(id, '') AS id_str,
          create_way,
          created_at + INTERVAL '8 hour' AS created_at,
          updated_at + INTERVAL '8 hour' AS updated_at,
          is_enable,
          can_order,
          can_withdraw
        FROM
          public.user_client
        ORDER BY
          id;
        """
        df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        st.warning(f"Could not access user_client table (this is normal): {e}")
        return None

# Load data
with st.spinner("Loading user data..."):
    trading_metrics_df = fetch_trading_metrics()
    users_per_day_df = fetch_users_per_day()
    user_client_df = fetch_user_client_info()

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
        
        # Add explanation about trade reconciliation
        st.info("""
        **Trade Type Breakdown:**
        - Position Opens: taker_way = 1 (Open Long) or 3 (Open Short)
        - Position Closes: taker_way = 2 (Close Short) or 4 (Close Long)
        - Win/Loss only counts closed positions: winning_trades + losing_trades + break_even_trades = closing_positions
        
        Some users may show a pattern of all winning trades and no losing trades if they have only closed positions profitably.
        """)
        
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
        
        # Scatter plot of trades vs performance
        st.subheader("Trading Activity vs Performance")
        
        scatter_fig = px.scatter(
            metrics_df,
            x='total_trades',
            y='net_pnl',
            size='max_leverage',
            color='win_percentage',
            hover_name='user_id_str',
            hover_data=['winning_trades', 'losing_trades', 'opening_positions', 'profit_factor'],
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
        
        top_users = metrics_df.nlargest(10, 'net_pnl')
        
        fig = px.bar(
            top_users,
            x='user_id_str',
            y='net_pnl',
            title='Top 10 Users by Net PnL',
            labels={'user_id_str': 'User ID', 'net_pnl': 'Net PnL (USD)'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Trade type analysis - NEW SECTION
        st.subheader("Trade Type Analysis")
        
        # Calculate aggregated trade type metrics
        total_metrics = {
            'Opening Positions': metrics_df['opening_positions'].sum(),
            'Winning Trades': metrics_df['winning_trades'].sum(),
            'Losing Trades': metrics_df['losing_trades'].sum(),
            'Break-even Trades': metrics_df['break_even_trades'].sum(),
        }
        
        trade_type_df = pd.DataFrame({
            'Trade Type': list(total_metrics.keys()),
            'Count': list(total_metrics.values())
        })
        
        # Create bar chart of trade types
        fig = px.bar(
            trade_type_df,
            x='Trade Type',
            y='Count',
            title='Distribution of Trade Types Across All Users',
            color='Trade Type'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate percentages
        trade_type_pct = pd.DataFrame({
            'Trade Type': list(total_metrics.keys()),
            'Percentage': [v / sum(total_metrics.values()) * 100 for v in total_metrics.values()]
        })
        
        # Create pie chart
        fig = px.pie(
            trade_type_pct,
            values='Percentage',
            names='Trade Type',
            title='Trade Type Distribution (%)'
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
        
        # Fetch detailed trade data
        user_trades = fetch_user_trade_details(selected_user)
        
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
        
        # Add trade type breakdown
        st.subheader("Trade Type Breakdown")
        
        trade_types_col1, trade_types_col2 = st.columns([1, 1])
        
        with trade_types_col1:
            # Create a small dataframe to explain the trade breakdown
            trade_breakdown = pd.DataFrame({
                'Trade Type': ['Open Long', 'Close Short', 'Open Short', 'Close Long', 'Total'],
                'Count': [
                    user_data.get('open_long_count', 0),
                    user_data.get('close_short_count', 0),
                    user_data.get('open_short_count', 0),
                    user_data.get('close_long_count', 0),
                    user_data.get('total_trades', 0)
                ]
            })
            
            st.dataframe(trade_breakdown, use_container_width=True)
            
            # Add explanation for win/loss count
            win_loss_sum = user_data.get('winning_trades', 0) + user_data.get('losing_trades', 0) + user_data.get('break_even_trades', 0)
            open_positions = user_data.get('opening_positions', 0)
            
            st.info(f"""
            **Trade Count Breakdown:**
            - Total trades: {user_data.get('total_trades', 0)}
            - Position opens: {open_positions} ({100*open_positions/max(1, user_data.get('total_trades', 0)):.1f}% of total)
            - Winning trades: {user_data.get('winning_trades', 0)}
            - Losing trades: {user_data.get('losing_trades', 0)}
            - Break-even trades: {user_data.get('break_even_trades', 0)}
            
            Position closes (wins + losses + break-even): {win_loss_sum}
            """)
        
        with trade_types_col2:
            # Create pie chart of trade types
            trade_types_data = pd.DataFrame({
                'Trade Type': ['Open Long', 'Close Short', 'Open Short', 'Close Long'],
                'Count': [
                    user_data.get('open_long_count', 0),
                    user_data.get('close_short_count', 0),
                    user_data.get('open_short_count', 0),
                    user_data.get('close_long_count', 0)
                ]
            })
            
            fig = px.pie(
                trade_types_data,
                values='Count',
                names='Trade Type',
                title='Trade Types Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        if user_trades is not None and len(user_trades) > 0:
            # Trade history
            st.subheader("Trade History")
            
            # Create two views - compact and detailed
            view_option = st.radio("Select View", ["Compact View", "Detailed View"], horizontal=True)
            
            if view_option == "Compact View":
                # Display enhanced compact view with renamed columns
                compact_cols = ['trade_time', 'pair_name', 'trade_type', 'position_action', 
                                'entry_exit_price', 'size', 'leverage_display', 
                                'trade_pnl', 'profit_share_usd', 'profit_share_percent']
                
                # Rename columns for display
                display_df = user_trades[compact_cols].copy()
                display_df.columns = ['Trade Time', 'Pair', 'Trade Type', 'Action',
                                    'Entry/Exit Price', 'Size', 'Leverage',
                                    'Trade PNL', 'Profit Share', 'Profit Share %']
                
                # Format numeric columns
                display_df['Entry/Exit Price'] = display_df['Entry/Exit Price'].round(8)
                display_df['Size'] = display_df['Size'].round(2)
                display_df['Trade PNL'] = display_df['Trade PNL'].round(2)
                display_df['Profit Share'] = display_df['Profit Share'].round(2)
                display_df['Profit Share %'] = display_df['Profit Share %'].round(2)
                
                st.dataframe(display_df, use_container_width=True)
                
                # Show additional columns for exit trades
                exit_trades = user_trades[user_trades['position_action'] == 'Exit'].copy()
                if len(exit_trades) > 0:
                    st.subheader("Exit Trade Details")
                    exit_cols = ['trade_time', 'pair_name', 'pre_profit_share_exit', 
                               'post_profit_share_exit', 'matched_entry_price', 'percent_move']
                    exit_display = exit_trades[exit_cols].copy()
                    exit_display.columns = ['Trade Time', 'Pair', 'Pre Profit Share Exit',
                                          'Post Profit Share Exit', 'Entry Price', '% Move']
                    
                    # Format columns
                    for col in ['Pre Profit Share Exit', 'Post Profit Share Exit', 'Entry Price']:
                        exit_display[col] = exit_display[col].round(8)
                    exit_display['% Move'] = exit_display['% Move'].round(4)
                    
                    st.dataframe(exit_display, use_container_width=True)
                
            else:
                # Display detailed view with enhanced information
                for idx, trade in user_trades.iterrows():
                    # Create a styled container for each trade
                    with st.container():
                        # Header row with pair info
                        col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1])
                        
                        with col1:
                            st.write("**Pair**")
                            st.write(f"🪙 {trade['pair_name']}")
                        
                        with col2:
                            st.write("**Leverage**")
                            st.write(trade['leverage_display'])
                        
                        with col3:
                            st.write("**Size**")
                            st.write(f"{trade['size']:,.2f}")
                        
                        with col4:
                            st.write("**Collateral**")
                            st.write(f"💎 {trade['collateral']:,.4f}")
                        
                        with col5:
                            st.write("**Entry/Exit Price**")
                            st.write(f"{trade['entry_exit_price']:.8f}")
                        
                        with col6:
                            st.write("**Trade PNL**")
                            if trade['position_action'] == 'Exit':
                                pnl_color = "green" if trade['trade_pnl'] > 0 else "red"
                                st.write(f":{pnl_color}[{trade['trade_pnl']:+.2f}]")
                            else:
                                st.write("-")
                        
                        # Additional details row
                        col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
                        
                        with col1:
                            st.write(f"**Action**: {trade['trade_type']}")
                            st.write(f"**Time**: {trade['trade_time'].strftime('%Y-%m-%d %H:%M:%S')}")
                        
                        with col2:
                            st.write(f"**Order Type**: {trade['order_type']}")
                            if trade['position_action'] == 'Exit':
                                st.write(f"**% Move**: {trade['percent_move']:.4f}%" if pd.notna(trade['percent_move']) else "% Move: N/A")
                            else:
                                st.write("**% Move**: N/A (Entry)")
                        
                        with col3:
                            if trade['position_action'] == 'Exit':
                                st.write(f"**Pre Share Exit**: {trade['pre_profit_share_exit']:.8f}" if pd.notna(trade['pre_profit_share_exit']) else "Pre Share Exit: N/A")
                                st.write(f"**Post Share Exit**: {trade['post_profit_share_exit']:.8f}" if pd.notna(trade['post_profit_share_exit']) else "Post Share Exit: N/A")
                            else:
                                st.write("**Exit Prices**: N/A (Entry)")
                        
                        with col4:
                            if trade['position_action'] == 'Exit':
                                st.write(f"**Profit Share**: ${trade['profit_share_usd']:,.2f}")
                                st.write(f"**Share %**: {trade['profit_share_percent']:.2f}%")
                            else:
                                st.write("**Profit Share**: N/A")
                                st.write("**Share %**: N/A")
                        
                        # Show matched entry price for exits
                        if trade['position_action'] == 'Exit' and pd.notna(trade['matched_entry_price']):
                            st.write(f"**Original Entry Price**: {trade['matched_entry_price']:.8f}")
                        
                        st.divider()
            
            # Add export functionality
            if st.button("Export Trade History to CSV"):
                # Prepare export data with all details
                export_df = user_trades.copy()
                
                # Rearrange columns for export
                export_cols = ['trade_time', 'pair_name', 'trade_type', 'position_action',
                              'order_type', 'leverage_display', 'size', 'collateral',
                              'entry_exit_price', 'matched_entry_price',
                              'pre_profit_share_exit', 'post_profit_share_exit',
                              'percent_move', 'trade_pnl', 'profit_share_usd', 
                              'profit_share_percent']
                
                csv = export_df[export_cols].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Trade History",
                    data=csv,
                    file_name=f"trade_history_{selected_user}.csv",
                    mime="text/csv"
                )
            
            # PnL timeline
            st.subheader("PnL Timeline")
            
            # Convert to datetime
            user_trades['trade_time'] = pd.to_datetime(user_trades['trade_time'])
            
            # Sort chronologically
            user_trades = user_trades.sort_values('trade_time')
            
            # Calculate cumulative PnL
            user_trades['cumulative_pnl'] = user_trades['trade_pnl'].cumsum()
            
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
                count=('trade_pnl', 'count'),
                total_pnl=('trade_pnl', 'sum'),
                avg_pnl=('trade_pnl', 'mean'),
                win_rate=('trade_pnl', lambda x: (x > 0).mean() * 100),
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
            try:
                leverage_groups = pd.cut(user_trades['leverage'], bins=10)
                leverage_analysis = user_trades.groupby(leverage_groups).agg(
                    count=('leverage', 'count'),
                    avg_pnl=('trade_pnl', 'mean'),
                    total_pnl=('trade_pnl', 'sum')
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
            except Exception as e:
                st.warning(f"Could not create leverage analysis: {e}")
        else:
            st.warning("No trade data available for this user.")
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
- View all users and their trading metrics
- Analyze performance by win rate, PnL, and other metrics
- Detailed user-level analysis with trading history
- Interactive charts and visualizations
- Direct user ID search

**Understanding Trade Types:**
- Total trades = opening positions + closing positions
- Win/Loss percentages are only calculated on closed positions

Data is sourced from the trade_fill_fresh table in the database.
""")

# Show last update time
st.sidebar.markdown(f"*Last updated: {now_sg.strftime('%Y-%m-%d %H:%M:%S')} (SGT)*")