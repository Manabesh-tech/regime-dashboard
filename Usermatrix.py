import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import pytz
import time
import altair as alt
from collections import defaultdict

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="User Trading Style Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- DATABASE CONNECTION ---
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

# --- UI SETUP ---
st.title("📊 User Trading Style Analysis Dashboard")
st.subheader("Comprehensive Analysis of Trading Behavior")
st.write(f"Current Singapore Time: {now_sg.strftime('%Y-%m-%d %H:%M:%S')}")

# --- DATA FETCH FUNCTIONS ---
@st.cache_data(ttl=600)
def fetch_trading_data(date_range=30, limit=10000):
    """Fetch comprehensive trading data for analysis"""
    # Calculate the date range
    now_utc = datetime.now(pytz.utc)
    start_date = now_utc - timedelta(days=date_range)
    
    query = """
    SELECT 
        t.taker_account_id AS user_id,
        ul.address AS wallet_address,
        t.pair_name,
        t.taker_way,
        t.taker_mode,
        t.taker_fee_mode,
        t.dual_side,
        t.leverage,
        t.deal_price,
        t.deal_vol,
        t.deal_size,
        t.coin_name,
        t.taker_pnl * t.collateral_price AS pnl_usd,
        t.taker_fee * t.collateral_price AS fee_usd,
        t.taker_share_pnl * t.collateral_price AS share_pnl_usd,
        t.deal_vol * t.collateral_price AS margin_usd,
        CASE
            WHEN t.taker_way = 1 THEN 'Open Long'
            WHEN t.taker_way = 2 THEN 'Close Short'
            WHEN t.taker_way = 3 THEN 'Open Short'
            WHEN t.taker_way = 4 THEN 'Close Long'
        END AS trade_type,
        CASE
            WHEN t.taker_mode = 1 THEN 'Active'
            WHEN t.taker_mode = 2 THEN 'Take Profit'
            WHEN t.taker_mode = 3 THEN 'Stop Loss'
            WHEN t.taker_mode = 4 THEN 'Liquidation'
        END AS order_type,
        CASE
            WHEN p.margin_type = 1 THEN 'Isolated'
            WHEN p.margin_type = 2 THEN 'Cross'
        END AS margin_type,
        t.created_at
    FROM 
        public.surfv2_trade t
    LEFT JOIN 
        public.surfv2_user_login_log ul ON t.taker_account_id = ul.account_id
    LEFT JOIN 
        public.surfv2_position p ON t.taker_position = p.id
    WHERE 
        t.created_at >= %s
        AND (t.taker_way IN (1, 2, 3, 4))
        AND (t.taker_mode IN (1, 2, 3, 4))
    ORDER BY 
        t.created_at DESC
    LIMIT %s
    """
    
    try:
        engine = get_database_connection()
        if not engine:
            return pd.DataFrame()
            
        df = pd.read_sql_query(text(query), engine, params=[start_date, limit])
        
        if df.empty:
            return pd.DataFrame()
        
        # Convert timestamp to SG time
        df['created_at'] = pd.to_datetime(df['created_at']).dt.tz_localize('UTC').dt.tz_convert('Asia/Singapore')
        df['trade_date'] = df['created_at'].dt.date
        df['trade_hour'] = df['created_at'].dt.hour
        df['trade_day'] = df['created_at'].dt.day_name()
        
        # Add indicators for win/loss
        df['is_win'] = df['pnl_usd'] > 0
        df['is_loss'] = df['pnl_usd'] < 0
        
        # Categorize trade direction
        df['direction'] = 'Unknown'
        df.loc[df['taker_way'].isin([1, 4]), 'direction'] = 'Long'  # Open Long or Close Long
        df.loc[df['taker_way'].isin([2, 3]), 'direction'] = 'Short'  # Close Short or Open Short
        
        # Categorize trade action
        df['action'] = 'Unknown'
        df.loc[df['taker_way'].isin([1, 3]), 'action'] = 'Open'  # Open Long or Open Short
        df.loc[df['taker_way'].isin([2, 4]), 'action'] = 'Close'  # Close Short or Close Long
        
        return df
    except Exception as e:
        st.error(f"Error fetching trading data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=600)
def calculate_user_metrics(df):
    """Calculate trading metrics for each user"""
    if df.empty:
        return pd.DataFrame()
    
    user_metrics = []
    
    # Group by user
    for user_id, user_data in df.groupby('user_id'):
        # Basic metrics
        wallet_address = user_data['wallet_address'].iloc[0] if not user_data['wallet_address'].isna().all() else "Unknown"
        total_trades = len(user_data)
        first_trade = user_data['created_at'].min()
        last_trade = user_data['created_at'].max()
        
        # PNL metrics
        total_pnl = user_data['pnl_usd'].sum()
        total_fee = user_data['fee_usd'].sum()
        net_pnl = total_pnl - total_fee
        
        # Trading behavior
        win_trades = user_data[user_data['is_win']].shape[0]
        loss_trades = user_data[user_data['is_loss']].shape[0]
        win_rate = win_trades / total_trades if total_trades > 0 else 0
        
        # Direction preference
        long_trades = user_data[user_data['direction'] == 'Long'].shape[0]
        short_trades = user_data[user_data['direction'] == 'Short'].shape[0]
        long_pct = long_trades / total_trades if total_trades > 0 else 0
        
        # Order type breakdown
        active_orders = user_data[user_data['order_type'] == 'Active'].shape[0]
        tp_orders = user_data[user_data['order_type'] == 'Take Profit'].shape[0]
        sl_orders = user_data[user_data['order_type'] == 'Stop Loss'].shape[0]
        liq_orders = user_data[user_data['order_type'] == 'Liquidation'].shape[0]
        
        # Timing patterns
        avg_trades_per_day = total_trades / max((last_trade - first_trade).days, 1)
        
        # Risk metrics
        avg_leverage = user_data['leverage'].mean()
        max_leverage = user_data['leverage'].max()
        
        # Pair diversity
        unique_pairs = user_data['pair_name'].nunique()
        most_traded_pair = user_data['pair_name'].value_counts().index[0] if not user_data['pair_name'].empty else "None"
        
        # Average position sizes
        avg_position_size = user_data['margin_usd'].mean()
        max_position_size = user_data['margin_usd'].max()
        
        # Margin preferences
        isolated_margin_pct = user_data[user_data['margin_type'] == 'Isolated'].shape[0] / total_trades if total_trades > 0 else 0
        
        # Calculate trading style indicators
        # Scalper: Many trades, small position sizes, short timeframes
        # Swing trader: Fewer trades, larger position sizes
        # Day trader: Moderate number of trades, closes positions daily
        scalper_score = 0
        swing_trader_score = 0
        day_trader_score = 0
        
        if avg_trades_per_day > 5:
            scalper_score += 1
        if avg_position_size < 100:
            scalper_score += 1
        
        if avg_trades_per_day < 2:
            swing_trader_score += 1
        if avg_position_size > 500:
            swing_trader_score += 1
        
        if 2 <= avg_trades_per_day <= 5:
            day_trader_score += 1
        
        # Analyze order type usage
        uses_tp_sl = (tp_orders + sl_orders) / total_trades if total_trades > 0 else 0
        
        # Account for position open duration
        # Compute average time positions are held (for positions that are opened and closed)
        open_longs = user_data[user_data['trade_type'] == 'Open Long']
        close_longs = user_data[user_data['trade_type'] == 'Close Long']
        open_shorts = user_data[user_data['trade_type'] == 'Open Short']
        close_shorts = user_data[user_data['trade_type'] == 'Close Short']
        
        # Determine primary trading style
        trading_styles = {
            "Scalper": scalper_score,
            "Swing Trader": swing_trader_score,
            "Day Trader": day_trader_score
        }
        primary_style = max(trading_styles, key=trading_styles.get)
        
        # Create user metrics dictionary
        user_metric = {
            'user_id': user_id,
            'wallet_address': wallet_address,
            'total_trades': total_trades,
            'first_trade': first_trade,
            'last_trade': last_trade,
            'days_trading': (last_trade - first_trade).days,
            'total_pnl': total_pnl,
            'total_fee': total_fee,
            'net_pnl': net_pnl,
            'win_trades': win_trades,
            'loss_trades': loss_trades,
            'win_rate': win_rate,
            'long_trades': long_trades,
            'short_trades': short_trades,
            'long_pct': long_pct,
            'active_orders': active_orders,
            'tp_orders': tp_orders,
            'sl_orders': sl_orders,
            'liq_orders': liq_orders,
            'avg_trades_per_day': avg_trades_per_day,
            'avg_leverage': avg_leverage,
            'max_leverage': max_leverage,
            'unique_pairs': unique_pairs,
            'most_traded_pair': most_traded_pair,
            'avg_position_size': avg_position_size,
            'max_position_size': max_position_size,
            'isolated_margin_pct': isolated_margin_pct,
            'uses_tp_sl': uses_tp_sl,
            'primary_style': primary_style,
            'scalper_score': scalper_score,
            'swing_trader_score': swing_trader_score,
            'day_trader_score': day_trader_score
        }
        
        user_metrics.append(user_metric)
    
    return pd.DataFrame(user_metrics)

@st.cache_data(ttl=600)
def analyze_user_trading_patterns(df, user_id):
    """Analyze detailed trading patterns for a specific user"""
    if df.empty:
        return {}
    
    user_data = df[df['user_id'] == user_id]
    if user_data.empty:
        return {}
    
    # Trading time patterns
    hour_distribution = user_data['trade_hour'].value_counts().sort_index()
    day_distribution = user_data['trade_day'].value_counts()
    
    # Pair preferences over time
    pair_over_time = user_data.groupby(['trade_date', 'pair_name']).size().unstack(fill_value=0)
    
    # PNL analysis
    pnl_over_time = user_data.groupby('trade_date')['pnl_usd'].sum()
    cumulative_pnl = pnl_over_time.cumsum()
    
    # Win/loss streaks
    is_win = user_data['is_win'].astype(int).values
    is_loss = user_data['is_loss'].astype(int).values
    
    win_streaks = []
    loss_streaks = []
    
    current_win_streak = 0
    current_loss_streak = 0
    
    for win, loss in zip(is_win, is_loss):
        if win:
            current_win_streak += 1
            if current_loss_streak > 0:
                loss_streaks.append(current_loss_streak)
                current_loss_streak = 0
        elif loss:
            current_loss_streak += 1
            if current_win_streak > 0:
                win_streaks.append(current_win_streak)
                current_win_streak = 0
        else:
            # Neither win nor loss (e.g., breakeven)
            if current_win_streak > 0:
                win_streaks.append(current_win_streak)
                current_win_streak = 0
            if current_loss_streak > 0:
                loss_streaks.append(current_loss_streak)
                current_loss_streak = 0
    
    # Add the last streaks if they exist
    if current_win_streak > 0:
        win_streaks.append(current_win_streak)
    if current_loss_streak > 0:
        loss_streaks.append(current_loss_streak)
    
    max_win_streak = max(win_streaks) if win_streaks else 0
    max_loss_streak = max(loss_streaks) if loss_streaks else 0
    avg_win_streak = sum(win_streaks) / len(win_streaks) if win_streaks else 0
    avg_loss_streak = sum(loss_streaks) / len(loss_streaks) if loss_streaks else 0
    
    # Leverage patterns
    leverage_over_time = user_data.groupby('trade_date')['leverage'].mean()
    
    # Position size patterns
    position_size_over_time = user_data.groupby('trade_date')['margin_usd'].mean()
    
    # Order type usage over time
    order_types_over_time = user_data.groupby(['trade_date', 'order_type']).size().unstack(fill_value=0)
    
    # Risk management patterns
    # Analyze how user manages risk - TP/SL usage vs manual closing
    tp_sl_usage = user_data.groupby('trade_date')['order_type'].apply(
        lambda x: (x == 'Take Profit').sum() + (x == 'Stop Loss').sum()
    ) / user_data.groupby('trade_date')['order_type'].count()
    
    # Compute average time positions are held (if possible)
    # This is complex and would require matching open and close orders
    
    # Pair rotation - how often user changes trading pairs
    daily_unique_pairs = user_data.groupby('trade_date')['pair_name'].nunique()
    
    # Liquidity analysis - when does user get liquidated
    liquidations = user_data[user_data['order_type'] == 'Liquidation']
    liq_by_pair = liquidations['pair_name'].value_counts()
    
    # Return analysis results
    return {
        'hour_distribution': hour_distribution,
        'day_distribution': day_distribution,
        'pair_over_time': pair_over_time,
        'pnl_over_time': pnl_over_time,
        'cumulative_pnl': cumulative_pnl,
        'win_streaks': win_streaks,
        'loss_streaks': loss_streaks,
        'max_win_streak': max_win_streak,
        'max_loss_streak': max_loss_streak,
        'avg_win_streak': avg_win_streak,
        'avg_loss_streak': avg_loss_streak,
        'leverage_over_time': leverage_over_time,
        'position_size_over_time': position_size_over_time,
        'order_types_over_time': order_types_over_time,
        'tp_sl_usage': tp_sl_usage,
        'daily_unique_pairs': daily_unique_pairs,
        'liq_by_pair': liq_by_pair
    }

# --- CONTROL PANEL ---
st.sidebar.title("Dashboard Controls")

# Date range selector
st.sidebar.subheader("Time Period")
date_range = st.sidebar.slider(
    "Number of days to analyze", 
    min_value=7, 
    max_value=365, 
    value=30,
    step=1
)

# Limit number of trades to fetch
row_limit = st.sidebar.slider(
    "Maximum trades to process",
    min_value=1000,
    max_value=100000,
    value=50000,
    step=1000
)

# Add a refresh button
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.experimental_rerun()

# Fetch the data
with st.spinner("Loading trading data..."):
    df = fetch_trading_data(date_range, row_limit)
    if df.empty:
        st.warning("No trading data found for the selected period.")
        st.stop()
    
    user_metrics_df = calculate_user_metrics(df)
    if user_metrics_df.empty:
        st.warning("No user metrics could be calculated.")
        st.stop()
    
    # Sort users by trading volume
    user_metrics_df = user_metrics_df.sort_values(by='total_trades', ascending=False)

# --- CREATE DASHBOARD TABS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "Trading Style Overview",
    "User Comparison",
    "User Detail Analysis",
    "Trading Pairs Analysis"
])

# Tab 1: Trading Style Overview
with tab1:
    st.header("Trading Style Overview")
    
    # Display basic statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Users", f"{len(user_metrics_df)}")
    
    with col2:
        st.metric("Total Trades", f"{df.shape[0]:,}")
    
    with col3:
        avg_win_rate = user_metrics_df['win_rate'].mean()
        st.metric("Avg Win Rate", f"{avg_win_rate:.2%}")
    
    with col4:
        total_net_pnl = user_metrics_df['net_pnl'].sum()
        st.metric("Total Net PNL", f"${total_net_pnl:,.2f}")
    
    # Trading style distribution
    st.subheader("Trading Style Distribution")
    
    style_counts = user_metrics_df['primary_style'].value_counts()
    
    fig = px.pie(
        values=style_counts.values,
        names=style_counts.index,
        title="Distribution of Trading Styles",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Win rate distribution
    st.subheader("Win Rate Distribution")
    
    fig = px.histogram(
        user_metrics_df,
        x='win_rate',
        nbins=20,
        labels={'win_rate': 'Win Rate', 'count': 'Number of Users'},
        title="Distribution of User Win Rates",
        color_discrete_sequence=['#3498db']
    )
    
    fig.update_layout(xaxis=dict(tickformat=".0%"))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Long vs Short preference
    st.subheader("Long vs Short Trading Preference")
    
    fig = px.histogram(
        user_metrics_df,
        x='long_pct',
        nbins=20,
        labels={'long_pct': 'Long Position Percentage', 'count': 'Number of Users'},
        title="Distribution of Long vs Short Preference",
        color_discrete_sequence=['#2ecc71']
    )
    
    fig.update_layout(xaxis=dict(tickformat=".0%"))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # TP/SL usage distribution
    st.subheader("Take Profit / Stop Loss Usage")
    
    fig = px.histogram(
        user_metrics_df,
        x='uses_tp_sl',
        nbins=20,
        labels={'uses_tp_sl': 'TP/SL Usage Rate', 'count': 'Number of Users'},
        title="Distribution of TP/SL Usage",
        color_discrete_sequence=['#e74c3c']
    )
    
    fig.update_layout(xaxis=dict(tickformat=".0%"))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Leverage distribution
    st.subheader("Leverage Usage")
    
    fig = px.histogram(
        user_metrics_df,
        x='avg_leverage',
        nbins=20,
        labels={'avg_leverage': 'Average Leverage', 'count': 'Number of Users'},
        title="Distribution of Average Leverage Usage",
        color_discrete_sequence=['#9b59b6']
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Position size distribution
    st.subheader("Position Size Distribution")
    
    fig = px.histogram(
        user_metrics_df,
        x='avg_position_size',
        nbins=20,
        labels={'avg_position_size': 'Average Position Size (USD)', 'count': 'Number of Users'},
        title="Distribution of Average Position Sizes",
        color_discrete_sequence=['#f39c12']
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Most popular trading pairs
    st.subheader("Most Popular Trading Pairs")
    
    pair_counts = df['pair_name'].value_counts().reset_index()
    pair_counts.columns = ['Pair', 'Trade Count']
    
    fig = px.bar(
        pair_counts.head(10),
        x='Pair',
        y='Trade Count',
        title="Top 10 Most Traded Pairs",
        color='Trade Count',
        color_continuous_scale='Viridis'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Tab 2: User Comparison
with tab2:
    st.header("User Comparison")
    
    # Filter metrics
    min_trades = st.slider(
        "Minimum number of trades", 
        min_value=1, 
        max_value=100, 
        value=10,
        step=1
    )
    
    filtered_users = user_metrics_df[user_metrics_df['total_trades'] >= min_trades].copy()
    
    if filtered_users.empty:
        st.warning(f"No users with at least {min_trades} trades.")
    else:
        # Top users by various metrics
        st.subheader("Top Performers by PNL")
        
        # Calculate PNL per trade for fair comparison
        filtered_users['pnl_per_trade'] = filtered_users['net_pnl'] / filtered_users['total_trades']
        
        # Sort by net PNL
        top_pnl_users = filtered_users.sort_values(by='net_pnl', ascending=False).head(10)
        
        fig = px.bar(
            top_pnl_users,
            x='user_id',
            y='net_pnl',
            title="Top 10 Users by Net PNL",
            labels={'user_id': 'User ID', 'net_pnl': 'Net PNL (USD)'},
            color='win_rate',
            color_continuous_scale='RdYlGn',
            hover_data=['total_trades', 'win_rate', 'avg_leverage', 'primary_style']
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Top users by win rate
        st.subheader("Top Users by Win Rate")
        
        top_winrate_users = filtered_users.sort_values(by='win_rate', ascending=False).head(10)
        
        fig = px.bar(
            top_winrate_users,
            x='user_id',
            y='win_rate',
            title="Top 10 Users by Win Rate",
            labels={'user_id': 'User ID', 'win_rate': 'Win Rate'},
            color='net_pnl',
            color_continuous_scale='RdYlGn',
            hover_data=['total_trades', 'net_pnl', 'avg_leverage', 'primary_style']
        )
        
        fig.update_layout(yaxis=dict(tickformat=".0%"))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Trading style comparison
        st.subheader("Trading Style Performance Comparison")
        
        style_performance = filtered_users.groupby('primary_style').agg({
            'user_id': 'count',
            'win_rate': 'mean',
            'pnl_per_trade': 'mean',
            'avg_leverage': 'mean',
            'avg_position_size': 'mean',
            'uses_tp_sl': 'mean'
        }).reset_index()
        
        style_performance.columns = [
            'Trading Style', 'User Count', 'Avg Win Rate', 
            'Avg PNL per Trade', 'Avg Leverage', 'Avg Position Size', 'TP/SL Usage'
        ]
        
        # Display as a table
        st.dataframe(
            style_performance.style.format({
                'Avg Win Rate': '{:.2%}',
                'Avg PNL per Trade': '${:.2f}',
                'Avg Leverage': '{:.2f}x',
                'Avg Position Size': '${:.2f}',
                'TP/SL Usage': '{:.2%}'
            }),
            use_container_width=True
        )
        
        # Scatter plot of win rate vs PNL
        st.subheader("Win Rate vs. PNL per Trade")
        
        fig = px.scatter(
            filtered_users,
            x='win_rate',
            y='pnl_per_trade',
            color='primary_style',
            size='total_trades',
            hover_data=['user_id', 'total_trades', 'avg_leverage'],
            labels={
                'win_rate': 'Win Rate',
                'pnl_per_trade': 'PNL per Trade (USD)',
                'primary_style': 'Trading Style'
            },
            title="Relationship Between Win Rate and PNL per Trade"
        )
        
        fig.update_layout(xaxis=dict(tickformat=".0%"))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Leverage vs PNL
        st.subheader("Leverage vs. PNL per Trade")
        
        fig = px.scatter(
            filtered_users,
            x='avg_leverage',
            y='pnl_per_trade',
            color='primary_style',
            size='total_trades',
            hover_data=['user_id', 'win_rate', 'total_trades'],
            labels={
                'avg_leverage': 'Average Leverage',
                'pnl_per_trade': 'PNL per Trade (USD)',
                'primary_style': 'Trading Style'
            },
            title="Relationship Between Leverage and PNL per Trade"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Long vs Short preference and performance
        st.subheader("Long vs Short Trading Performance")
        
        # Create bins for long percentage
        filtered_users['long_pct_bin'] = pd.cut(
            filtered_users['long_pct'],
            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
        )
        
        long_short_perf = filtered_users.groupby('long_pct_bin').agg({
            'user_id': 'count',
            'win_rate': 'mean',
            'pnl_per_trade': 'mean'
        }).reset_index()
        
        long_short_perf.columns = ['Long Trade %', 'User Count', 'Avg Win Rate', 'Avg PNL per Trade']
        
        fig = px.bar(
            long_short_perf,
            x='Long Trade %',
            y='Avg PNL per Trade',
            title="PNL Performance by Long/Short Preference",
            color='Avg Win Rate',
            color_continuous_scale='RdYlGn',
            text='User Count'
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Tab 3: User Detail Analysis
with tab3:
    st.header("User Detail Analysis")
    
    # User selection
    selected_user = st.selectbox(
        "Select User to Analyze",
        options=user_metrics_df['user_id'].tolist(),
        format_func=lambda x: f"{x} ({user_metrics_df[user_metrics_df['user_id'] == x]['total_trades'].values[0]} trades)"
    )
    
    # Analyze user patterns
    user_patterns = analyze_user_trading_patterns(df, selected_user)
    
    if not user_patterns:
        st.warning(f"No detailed data available for user {selected_user}")
    else:
        # Get user metrics
        user_metric = user_metrics_df[user_metrics_df['user_id'] == selected_user].iloc[0]
        
        # User overview
        st.subheader("User Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", f"{user_metric['total_trades']:,}")
        
        with col2:
            st.metric("Net PNL", f"${user_metric['net_pnl']:,.2f}")
        
        with col3:
            st.metric("Win Rate", f"{user_metric['win_rate']:.2%}")
        
        with col4:
            st.metric("Avg Leverage", f"{user_metric['avg_leverage']:.2f}x")
        
        # Second row of metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Trading Style", f"{user_metric['primary_style']}")
        
        with col2:
            st.metric("Long/Short", f"{user_metric['long_pct']:.0%} Long")
        
        with col3:
            st.metric("TP/SL Usage", f"{user_metric['uses_tp_sl']:.2%}")
        
        with col4:
            st.metric("Avg Position Size", f"${user_metric['avg_position_size']:,.2f}")
        
        # PNL over time
        st.subheader("PNL Performance Over Time")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=user_patterns['cumulative_pnl'].index,
            y=user_patterns['cumulative_pnl'].values,
            mode='lines',
            name='Cumulative PNL',
            line=dict(color='green' if user_patterns['cumulative_pnl'].iloc[-1] > 0 else 'red')
        ))
        
        fig.add_trace(go.Bar(
            x=user_patterns['pnl_over_time'].index,
            y=user_patterns['pnl_over_time'].values,
            name='Daily PNL',
            marker_color=['green' if x > 0 else 'red' for x in user_patterns['pnl_over_time'].values]
        ))
        
        fig.update_layout(
            title=f"PNL Performance for User {selected_user}",
            xaxis_title="Date",
            yaxis_title="PNL (USD)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Trading activity patterns
        st.subheader("Trading Activity Patterns")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Hour distribution
            fig = px.bar(
                x=user_patterns['hour_distribution'].index,
                y=user_patterns['hour_distribution'].values,
                labels={'x': 'Hour of Day (SG Time)', 'y': 'Number of Trades'},
                title="Trading Activity by Hour",
                color=user_patterns['hour_distribution'].values,
                color_continuous_scale='Viridis'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Day distribution
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_counts = pd.Series(0, index=days_order)
            for day in user_patterns['day_distribution'].index:
                if day in day_counts.index:
                    day_counts[day] = user_patterns['day_distribution'][day]
            
            fig = px.bar(
                x=day_counts.index,
                y=day_counts.values,
                labels={'x': 'Day of Week', 'y': 'Number of Trades'},
                title="Trading Activity by Day of Week",
                color=day_counts.values,
                color_continuous_scale='Viridis'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Order type usage
        st.subheader("Order Type Usage")
        
        if 'order_types_over_time' in user_patterns and not user_patterns['order_types_over_time'].empty:
            # Sum up order types
            order_type_sums = user_patterns['order_types_over_time'].sum()
            
            fig = px.pie(
                values=order_type_sums.values,
                names=order_type_sums.index,
                title="Order Type Distribution",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Order types over time
            fig = px.bar(
                user_patterns['order_types_over_time'],
                labels={'value': 'Number of Orders', 'variable': 'Order Type'},
                title="Order Types Over Time",
                barmode='stack'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No order type data available for this user.")
        
        # Trading pairs analysis
        st.subheader("Trading Pairs Analysis")
        
        # Extract user's trading data
        user_trades = df[df['user_id'] == selected_user]
        
        # Get pair counts
        pair_counts = user_trades['pair_name'].value_counts().reset_index()
        pair_counts.columns = ['Pair', 'Trade Count']
        
        fig = px.bar(
            pair_counts.head(10),
            x='Pair',
            y='Trade Count',
            title="Most Traded Pairs",
            color='Trade Count',
            color_continuous_scale='Viridis'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # PNL by pair
        pair_pnl = user_trades.groupby('pair_name')['pnl_usd'].sum().reset_index()
        pair_pnl.columns = ['Pair', 'Total PNL']
        pair_pnl = pair_pnl.sort_values(by='Total PNL', ascending=False)
        
        fig = px.bar(
            pair_pnl.head(10),
            x='Pair',
            y='Total PNL',
            title="PNL by Trading Pair",
            color='Total PNL',
            color_continuous_scale='RdBu',
            color_continuous_midpoint=0
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Win rate by pair
        pair_wins = user_trades.groupby('pair_name')['is_win'].sum().reset_index()
        pair_trades = user_trades.groupby('pair_name').size().reset_index(name='Total Trades')
        
        pair_winrate = pd.merge(pair_wins, pair_trades, on='pair_name')
        pair_winrate['Win Rate'] = pair_winrate['is_win'] / pair_winrate['Total Trades']
        pair_winrate = pair_winrate.sort_values(by='Total Trades', ascending=False)
        
        # Filter pairs with at least 5 trades
        pair_winrate = pair_winrate[pair_winrate['Total Trades'] >= 5]
        
        if not pair_winrate.empty:
            fig = px.bar(
                pair_winrate.head(10),
                x='pair_name',
                y='Win Rate',
                title="Win Rate by Trading Pair (Min. 5 Trades)",
                color='Win Rate',
                color_continuous_scale='RdYlGn',
                text='Total Trades'
            )
            
            fig.update_layout(yaxis=dict(tickformat=".0%"))
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data to analyze win rates by pair.")
        
        # Risk management analysis
        st.subheader("Risk Management Analysis")
        
        # Leverage usage over time
        if 'leverage_over_time' in user_patterns and not user_patterns['leverage_over_time'].empty:
            fig = px.line(
                x=user_patterns['leverage_over_time'].index,
                y=user_patterns['leverage_over_time'].values,
                labels={'x': 'Date', 'y': 'Average Leverage'},
                title="Leverage Usage Over Time"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No leverage data available for this user.")
        
        # Position size over time
        if 'position_size_over_time' in user_patterns and not user_patterns['position_size_over_time'].empty:
            fig = px.line(
                x=user_patterns['position_size_over_time'].index,
                y=user_patterns['position_size_over_time'].values,
                labels={'x': 'Date', 'y': 'Average Position Size (USD)'},
                title="Position Size Over Time"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No position size data available for this user.")
        
        # Streak analysis
        st.subheader("Win/Loss Streak Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Max Win Streak", f"{user_patterns['max_win_streak']}")
        
        with col2:
            st.metric("Max Loss Streak", f"{user_patterns['max_loss_streak']}")
        
        # Streak distribution
        if user_patterns['win_streaks'] or user_patterns['loss_streaks']:
            win_streak_count = pd.Series(user_patterns['win_streaks']).value_counts().sort_index()
            loss_streak_count = pd.Series(user_patterns['loss_streaks']).value_counts().sort_index()
            
            # Combine into a dataframe
            streak_df = pd.DataFrame({
                'Win Streak': win_streak_count,
                'Loss Streak': loss_streak_count
            }).fillna(0)
            
            fig = px.bar(
                streak_df,
                barmode='group',
                labels={'value': 'Frequency', 'variable': 'Streak Type', 'index': 'Streak Length'},
                title="Win/Loss Streak Distribution"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No streak data available for this user.")
        
        # Trading behavior summary
        st.subheader("Trading Behavior Summary")
        
        # Calculate metrics for trading behavior
        trading_metrics = {
            "Trading Style": user_metric['primary_style'],
            "Win Rate": f"{user_metric['win_rate']:.2%}",
            "Avg Trades Per Day": f"{user_metric['avg_trades_per_day']:.2f}",
            "Typical Position Size": f"${user_metric['avg_position_size']:.2f}",
            "Leverage Usage": f"{user_metric['avg_leverage']:.2f}x",
            "Long/Short Preference": f"{user_metric['long_pct']:.0%} Long / {1-user_metric['long_pct']:.0%} Short",
            "TP/SL Usage": f"{user_metric['uses_tp_sl']:.2%}",
            "Pair Diversity": f"{user_metric['unique_pairs']} different pairs",
            "Most Traded Pair": user_metric['most_traded_pair'],
            "Risk Management": "Good" if user_metric['uses_tp_sl'] > 0.5 else "Moderate" if user_metric['uses_tp_sl'] > 0.2 else "Poor",
            "Liquidations": f"{user_metric['liq_orders']} times"
        }
        
        # Convert to dataframe
        trading_metrics_df = pd.DataFrame({
            'Metric': list(trading_metrics.keys()),
            'Value': list(trading_metrics.values())
        })
        
        st.table(trading_metrics_df)
        
        # Trading style recommendations
        st.subheader("Recommendations")
        
        # Generate recommendations based on metrics
        recommendations = []
        
        if user_metric['win_rate'] < 0.4:
            recommendations.append("Consider improving trade entry criteria to increase win rate.")
        
        if user_metric['uses_tp_sl'] < 0.3:
            recommendations.append("Increase usage of Take Profit and Stop Loss orders for better risk management.")
        
        if user_metric['liq_orders'] > 0:
            recommendations.append(f"Reduce leverage or use wider stop losses to avoid liquidations ({user_metric['liq_orders']} liquidations detected).")
        
        if user_metric['avg_leverage'] > 10:
            recommendations.append("High leverage detected. Consider reducing leverage to manage risk better.")
        
        if user_metric['unique_pairs'] < 3:
            recommendations.append("Consider diversifying to trade more pairs for better opportunities.")
        
        if user_metric['long_pct'] > 0.9 or user_metric['long_pct'] < 0.1:
            recommendations.append("Consider trading both long and short directions for more opportunities.")
        
        if not recommendations:
            recommendations.append("Trading patterns appear solid. Continue with current strategy while managing risk appropriately.")
        
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")

# Tab 4: Trading Pairs Analysis
with tab4:
    st.header("Trading Pairs Analysis")
    
    # Pair metrics
    pair_metrics = df.groupby('pair_name').agg({
        'user_id': pd.Series.nunique,
        'pnl_usd': ['sum', 'mean'],
        'leverage': 'mean',
        'margin_usd': 'mean',
        'is_win': 'mean'
    })
    
    pair_metrics.columns = [
        'Unique Users', 'Total PNL', 'Avg PNL per Trade',
        'Avg Leverage', 'Avg Position Size', 'Win Rate'
    ]
    
    pair_metrics = pair_metrics.reset_index()
    
    # Sort by trading volume (unique users)
    pair_metrics = pair_metrics.sort_values(by='Unique Users', ascending=False)
    
    # Pair selection
    selected_pair = st.selectbox(
        "Select Trading Pair to Analyze",
        options=pair_metrics['pair_name'].tolist(),
        format_func=lambda x: f"{x} ({pair_metrics[pair_metrics['pair_name'] == x]['Unique Users'].values[0]} users)"
    )
    
    # Filter data for selected pair
    pair_data = df[df['pair_name'] == selected_pair]
    
    if pair_data.empty:
        st.warning(f"No data available for pair {selected_pair}")
    else:
        # Pair overview
        st.subheader("Pair Overview")
        
        pair_metric = pair_metrics[pair_metrics['pair_name'] == selected_pair].iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", f"{len(pair_data):,}")
        
        with col2:
            st.metric("Unique Users", f"{pair_metric['Unique Users']:,}")
        
        with col3:
            st.metric("Win Rate", f"{pair_metric['Win Rate']:.2%}")
        
        with col4:
            st.metric("Total PNL", f"${pair_metric['Total PNL']:,.2f}")
        
        # Second row of metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Avg PNL per Trade", f"${pair_metric['Avg PNL per Trade']:,.2f}")
        
        with col2:
            st.metric("Avg Leverage", f"{pair_metric['Avg Leverage']:.2f}x")
        
        with col3:
            st.metric("Avg Position Size", f"${pair_metric['Avg Position Size']:,.2f}")
        
        with col4:
            # Calculate long percentage
            long_trades = pair_data[pair_data['direction'] == 'Long'].shape[0]
            total_trades = len(pair_data)
            long_pct = long_trades / total_trades if total_trades > 0 else 0
            st.metric("Long/Short Split", f"{long_pct:.0%} Long")
        
        # Trading volume over time
        st.subheader("Trading Volume Over Time")
        
        # Group by date and count trades
        volume_over_time = pair_data.groupby('trade_date').size()
        
        fig = px.bar(
            x=volume_over_time.index,
            y=volume_over_time.values,
            labels={'x': 'Date', 'y': 'Number of Trades'},
            title=f"Trading Volume for {selected_pair}",
            color=volume_over_time.values,
            color_continuous_scale='Viridis'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # PNL over time
        st.subheader("PNL Performance Over Time")
        
        pnl_over_time = pair_data.groupby('trade_date')['pnl_usd'].sum()
        
        fig = px.bar(
            x=pnl_over_time.index,
            y=pnl_over_time.values,
            labels={'x': 'Date', 'y': 'PNL (USD)'},
            title=f"PNL Performance for {selected_pair}",
            color=pnl_over_time.values,
            color_continuous_scale='RdBu',
            color_continuous_midpoint=0
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Trading direction analysis
        st.subheader("Trading Direction Analysis")
        
        # Calculate metrics by direction
        direction_metrics = pair_data.groupby('direction').agg({
            'pnl_usd': ['sum', 'mean'],
            'is_win': 'mean',
            'user_id': 'count'
        })
        
        direction_metrics.columns = [
            'Total PNL', 'Avg PNL per Trade', 'Win Rate', 'Trade Count'
        ]
        
        direction_metrics = direction_metrics.reset_index()
        
        # Create a bar chart
        fig = px.bar(
            direction_metrics,
            x='direction',
            y='Total PNL',
            color='Win Rate',
            text='Trade Count',
            labels={'direction': 'Direction', 'Total PNL': 'Total PNL (USD)'},
            title=f"PNL by Trading Direction for {selected_pair}",
            color_continuous_scale='RdYlGn'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Order type analysis
        st.subheader("Order Type Analysis")
        
        # Calculate metrics by order type
        order_metrics = pair_data.groupby('order_type').agg({
            'pnl_usd': ['sum', 'mean'],
            'is_win': 'mean',
            'user_id': 'count'
        })
        
        order_metrics.columns = [
            'Total PNL', 'Avg PNL per Trade', 'Win Rate', 'Trade Count'
        ]
        
        order_metrics = order_metrics.reset_index()
        
        # Create a bar chart
        fig = px.bar(
            order_metrics,
            x='order_type',
            y='Trade Count',
            color='Avg PNL per Trade',
            text='Win Rate',
            labels={'order_type': 'Order Type', 'Trade Count': 'Number of Trades'},
            title=f"Order Type Usage for {selected_pair}",
            color_continuous_scale='RdBu',
            color_continuous_midpoint=0
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # User performance on this pair
        st.subheader("Top Users on This Pair")
        
        # Group by user
        user_pair_metrics = pair_data.groupby('user_id').agg({
            'pnl_usd': 'sum',
            'is_win': 'mean',
            'margin_usd': 'mean',
            'leverage': 'mean',
            'user_id': 'count'
        })
        
        user_pair_metrics.columns = [
            'Total PNL', 'Win Rate', 'Avg Position Size',
            'Avg Leverage', 'Trade Count'
        ]
        
        user_pair_metrics = user_pair_metrics.reset_index()
        
        # Filter users with at least 5 trades
        user_pair_metrics = user_pair_metrics[user_pair_metrics['Trade Count'] >= 5]
        
        # Sort by PNL
        user_pair_metrics = user_pair_metrics.sort_values(by='Total PNL', ascending=False)
        
        if not user_pair_metrics.empty:
            # Create a table
            st.dataframe(
                user_pair_metrics.head(10).style.format({
                    'Total PNL': '${:.2f}',
                    'Win Rate': '{:.2%}',
                    'Avg Position Size': '${:.2f}',
                    'Avg Leverage': '{:.2f}x'
                }),
                use_container_width=True
            )
            
            # Create a scatter plot of win rate vs PNL
            fig = px.scatter(
                user_pair_metrics,
                x='Win Rate',
                y='Total PNL',
                size='Trade Count',
                color='Avg Leverage',
                hover_data=['user_id', 'Trade Count', 'Avg Position Size'],
                labels={
                    'Win Rate': 'Win Rate',
                    'Total PNL': 'Total PNL (USD)',
                    'Avg Leverage': 'Avg Leverage'
                },
                title=f"User Performance on {selected_pair}"
            )
            
            fig.update_layout(xaxis=dict(tickformat=".0%"))
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough user data to analyze performance on this pair.")
        
        # Leverage analysis
        st.subheader("Leverage Analysis")
        
        # Create leverage bins
        pair_data['leverage_bin'] = pd.cut(
            pair_data['leverage'],
            bins=[0, 2, 5, 10, 20, 50, 100, 1000],
            labels=['1-2x', '2-5x', '5-10x', '10-20x', '20-50x', '50-100x', '100x+']
        )
        
        # Calculate metrics by leverage bin
        leverage_metrics = pair_data.groupby('leverage_bin').agg({
            'pnl_usd': ['sum', 'mean'],
            'is_win': 'mean',
            'user_id': 'count'
        })
        
        leverage_metrics.columns = [
            'Total PNL', 'Avg PNL per Trade', 'Win Rate', 'Trade Count'
        ]
        
        leverage_metrics = leverage_metrics.reset_index()
        
        # Create a bar chart
        fig = px.bar(
            leverage_metrics,
            x='leverage_bin',
            y='Trade Count',
            color='Win Rate',
            labels={'leverage_bin': 'Leverage Range', 'Trade Count': 'Number of Trades'},
            title=f"Trade Distribution by Leverage for {selected_pair}",
            color_continuous_scale='RdYlGn',
            text='Avg PNL per Trade'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Hour analysis
        st.subheader("Trading Hour Analysis")
        
        # Calculate metrics by hour
        hour_metrics = pair_data.groupby('trade_hour').agg({
            'pnl_usd': ['sum', 'mean'],
            'is_win': 'mean',
            'user_id': 'count'
        })
        
        hour_metrics.columns = [
            'Total PNL', 'Avg PNL per Trade', 'Win Rate', 'Trade Count'
        ]
        
        hour_metrics = hour_metrics.reset_index()
        
        # Create a bar chart
        fig = px.bar(
            hour_metrics,
            x='trade_hour',
            y='Trade Count',
            color='Win Rate',
            labels={'trade_hour': 'Hour of Day (SG Time)', 'Trade Count': 'Number of Trades'},
            title=f"Trade Distribution by Hour for {selected_pair}",
            color_continuous_scale='RdYlGn'
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Add footer with last update time
st.markdown("---")
st.markdown(f"*Last updated: {now_sg.strftime('%Y-%m-%d %H:%M:%S')} (Singapore Time)*")