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
    # DB parameters
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

# Initialize session state for PnL values
if 'daily_pnl' not in st.session_state:
    st.session_state.daily_pnl = 0
if 'all_time_pnl' not in st.session_state:
    st.session_state.all_time_pnl = 0

# Function to fetch daily PnL
@st.cache_data(ttl=60)  # Cache for 1 minute for live data
def fetch_daily_pnl():
    query = """
    WITH DailyTradeData AS (
        SELECT
            DATE("public"."trade_fill_fresh"."created_at" + INTERVAL '+8 hours') AS "日期",
            
            -- 计算 "总和"，加上止损费用部分
            SUM(
                CASE
                    WHEN "public"."trade_fill_fresh"."taker_way" IN (1, 2, 3, 4) 
                         AND CONCAT("public"."trade_fill_fresh"."taker_account_id", '') NOT IN ('383645340185311232', '383645323663947776', '384014230417035264', '384014656596699136','384015011585812480','384015271796238336','384015526326947840')
                    THEN "public"."trade_fill_fresh"."taker_pnl" * "public"."trade_fill_fresh"."collateral_price"
                    ELSE 0
                END
            ) 
            + SUM(
                CASE
                    WHEN "public"."trade_fill_fresh"."taker_fee_mode" = 1 
                         AND "public"."trade_fill_fresh"."taker_way" IN (1, 3)
                         AND CONCAT("public"."trade_fill_fresh"."taker_account_id", '') NOT IN ('383645340185311232', '383645323663947776', '384014230417035264', '384014656596699136','384015011585812480','384015271796238336','384015526326947840')
                    THEN -1 * "public"."trade_fill_fresh"."taker_fee" * "public"."trade_fill_fresh"."collateral_price"
                    ELSE 0
                END
            ) 
            + SUM(
                CASE
                    WHEN "public"."trade_fill_fresh"."taker_way" = 0
                         AND CONCAT("public"."trade_fill_fresh"."taker_account_id", '') NOT IN ('383645340185311232', '383645323663947776', '384014230417035264', '384014656596699136','384015011585812480','384015271796238336','384015526326947840')
                    THEN "public"."trade_fill_fresh"."funding_fee" * "public"."trade_fill_fresh"."collateral_price"
                    ELSE 0
                END
            )
            + COALESCE(SUM(
                CASE
                    WHEN CONCAT("public"."trade_fill_fresh"."taker_account_id", '') NOT IN ('383645340185311232', '383645323663947776', '384014230417035264', '384014656596699136','384015011585812480','384015271796238336','384015526326947840')
                    THEN -1 * "public"."trade_fill_fresh"."taker_sl_fee" * "public"."trade_fill_fresh"."collateral_price" - "public"."trade_fill_fresh"."maker_sl_fee"
                    ELSE 0
                END
            ), 0) AS "总和"
        FROM
            "public"."trade_fill_fresh"
        WHERE
            "public"."trade_fill_fresh"."created_at" + INTERVAL '+8 hours' >= DATE_TRUNC('day', NOW() + INTERVAL '8 hours')
            AND "public"."trade_fill_fresh"."created_at" + INTERVAL '+8 hours' < DATE_TRUNC('day', NOW() + INTERVAL '8 hours') + INTERVAL '1 day'
        GROUP BY
            DATE("public"."trade_fill_fresh"."created_at" + INTERVAL '+8 hours')
    ),

    DailyCashbookData AS (
        SELECT
            DATE("public"."user_cashbooks"."created_at" + INTERVAL '+8 hours') AS "日期",
            SUM("public"."user_cashbooks"."amount" * "public"."user_cashbooks"."coin_price") AS "总返佣"
        FROM
            "public"."user_cashbooks"
        WHERE
            "public"."user_cashbooks"."remark" = '给邀请人返佣'
            AND CONCAT("public"."user_cashbooks"."account_id", '') NOT IN ('383645340185311232', '383645323663947776', '384014230417035264', '384014656596699136','384015011585812480','384015271796238336','384015526326947840')
            AND "public"."user_cashbooks"."created_at" + INTERVAL '+8 hours' >= DATE_TRUNC('day', NOW() + INTERVAL '8 hours')
            AND "public"."user_cashbooks"."created_at" + INTERVAL '+8 hours' < DATE_TRUNC('day', NOW() + INTERVAL '8 hours') + INTERVAL '1 day'
        GROUP BY
            DATE("public"."user_cashbooks"."created_at" + INTERVAL '+8 hours')
    )

    SELECT
        COALESCE(-1 * t."总和" - COALESCE(c."总返佣", 0), 0) AS "总平台盈亏扣除返佣"
    FROM
        DailyTradeData t
    LEFT JOIN
        DailyCashbookData c ON t."日期" = c."日期"
    ORDER BY
        t."日期" DESC
    LIMIT 1
    """
    
    try:
        result = pd.read_sql(query, conn)
        if not result.empty:
            return result.iloc[0]['总平台盈亏扣除返佣']
        return 0
    except Exception as e:
        st.error(f"Error fetching daily PnL: {e}")
        return 0

# Function to fetch live trades
@st.cache_data(ttl=30)  # Cache for 30 seconds for live data
def fetch_live_trades():
    query = """
    SELECT
      id,
      pair_name,
      deal_price,
      deal_vol,
      deal_size,
      leverage,
      coin_name,
      taker_type,
      taker_way,
      taker_mode,
      collateral_price,
      taker_fee_mode,
      taker_fee,
      taker_pnl,
      trigger_price,
      taker_share_pnl,
      CONCAT(taker_account_id, '') AS user_id_str,
      (created_at + INTERVAL '8 hour') AS trade_time,
      -- Calculated fields
      CASE
        WHEN taker_mode = 1 THEN '主动 (Active)'
        WHEN taker_mode = 2 THEN '止盈 (Take Profit)'
        WHEN taker_mode = 3 THEN '止损 (Stop Loss)'
        WHEN taker_mode = 4 THEN '爆仓 (Liq)'
      END AS taker_mode_display,
      CASE
        WHEN taker_way = 1 THEN '开多 (Open Long)'
        WHEN taker_way = 2 THEN '平空 (Close Short)'
        WHEN taker_way = 3 THEN '开空 (Open Short)'
        WHEN taker_way = 4 THEN '平多 (Close Long)'
      END AS taker_way_display,
      CASE
        WHEN taker_fee_mode = 1 THEN 'Flat'
        WHEN taker_fee_mode = 2 THEN 'Profit Share'
      END AS fee_mode_display,
      taker_pnl * collateral_price AS user_pnl_usd,
      taker_share_pnl * collateral_price AS platform_profit_share,
      (taker_pnl * collateral_price) + (taker_share_pnl * collateral_price) AS total_pnl,
      taker_fee * collateral_price AS flat_fee_usd,
      deal_vol * collateral_price AS volume_usd
    FROM
      public.trade_fill_fresh
    WHERE
      taker_way IN (1, 2, 3, 4)
      AND taker_mode IN (1, 2, 3, 4)
    ORDER BY
      created_at DESC
    LIMIT 100
    """
    
    try:
        df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        st.error(f"Error fetching live trades: {e}")
        return None

# Function to fetch all-time PnL
@st.cache_data(ttl=60)  # Cache for 1 minute for live data
def fetch_all_time_pnl():
    query = """
    SELECT
      -- Sum of (Platform PnL + Flat Fee Revenue + Funding Fee PnL + SL Fees) - Profit Share Rebates
      (
        -- Platform PnL from user trading
        (SELECT SUM(-1 * st.taker_pnl * st.collateral_price)
         FROM public.trade_fill_fresh st
         WHERE st.taker_way IN (1, 2, 3, 4)
           AND CONCAT(st.taker_account_id, '') NOT IN ('383645340185311232', '383645323663947776', '384014230417035264', '384014656596699136','384015011585812480','384015271796238336','384015526326947840'))
        +
        -- Taker Fee
        (SELECT SUM(st.taker_fee * st.collateral_price)
         FROM public.trade_fill_fresh st
         WHERE st.taker_fee_mode = 1
           AND st.taker_way IN (1, 3)
           AND CONCAT(st.taker_account_id, '') NOT IN ('383645340185311232', '383645323663947776', '384014230417035264', '384014656596699136','384015011585812480','384015271796238336','384015526326947840'))
        +
        -- Platform funding fee PnL
        (SELECT SUM(-1 * st.funding_fee * st.collateral_price)
         FROM public.trade_fill_fresh st
         WHERE st.taker_way = 0
           AND CONCAT(st.taker_account_id, '') NOT IN ('383645340185311232', '383645323663947776', '384014230417035264', '384014656596699136','384015011585812480','384015271796238336','384015526326947840'))
        +
        -- SL Fees (taker_sl_fee * collateral_price + maker_sl_fee)
        (SELECT SUM(st.taker_sl_fee * st.collateral_price + st.maker_sl_fee)
         FROM public.trade_fill_fresh st
         WHERE CONCAT(st.taker_account_id, '') NOT IN ('383645340185311232', '383645323663947776', '384014230417035264', '384014656596699136','384015011585812480','384015271796238336','384015526326947840'))
      )
      -
      -- 返佣（包含了Flat和Profit)
      (SELECT SUM(st.amount * coin_price)
       FROM public.user_cashbooks st
       WHERE st."remark" = '给邀请人返佣'
         AND CONCAT(st.account_id, '') NOT IN ('383645340185311232', '383645323663947776', '384014230417035264', '384014656596699136','384015011585812480','384015271796238336','384015526326947840')) AS "总平台盈亏扣除返佣"
    """
    
    try:
        result = pd.read_sql(query, conn)
        if not result.empty:
            return result.iloc[0]['总平台盈亏扣除返佣']
        return 0
    except Exception as e:
        st.error(f"Error fetching all-time PnL: {e}")
        return 0

# Function to fetch trading metrics
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
        
        return df
    except Exception as e:
        st.error(f"Error fetching trading metrics: {e}")
        return None

# Function to fetch detailed trade data for a specific user
@st.cache_data(ttl=600, show_spinner=False)
def fetch_user_trade_details_v4(user_id):
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
      -- User Received PNL (what user actually gets)
      ROUND(taker_pnl * collateral_price, 2) AS user_received_pnl,
      -- Platform Profit Share
      ROUND(taker_share_pnl * collateral_price, 2) AS profit_share,
      -- Calculate profit share percentage
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
                exit_price = row['entry_exit_price']
                if entry_price != 0:
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
        return None

# Function to count users per day
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
        date_counts = df.groupby('date').size().reset_index(name='new_users')
        return date_counts
    except Exception as e:
        st.error(f"Error fetching users per day: {e}")
        return None

# Main title
st.title("User Trading Behavior Analysis Dashboard")
st.caption("This dashboard analyzes trading patterns and behaviors for users")

# Set up the timezone
singapore_timezone = pytz.timezone('Asia/Singapore')
now_utc = datetime.now(pytz.utc)
now_sg = now_utc.astimezone(singapore_timezone)
st.write(f"Current Singapore Time: {now_sg.strftime('%Y-%m-%d %H:%M:%S')}")

# Load live data
with st.spinner("Loading live data..."):
    st.session_state.daily_pnl = fetch_daily_pnl()
    st.session_state.all_time_pnl = fetch_all_time_pnl()
    live_trades_df = fetch_live_trades()

# Create tabs with PnL display in Live Trades tab title
daily_pnl_display = f"${st.session_state.daily_pnl:,.2f}" if st.session_state.daily_pnl else "$0.00"
all_time_pnl_display = f"${st.session_state.all_time_pnl:,.2f}" if st.session_state.all_time_pnl else "$0.00"

tab1, tab2, tab3, tab4 = st.tabs([
    f"Live Trades | Daily: {daily_pnl_display} | All-Time: {all_time_pnl_display}",
    "All Users",
    "Trading Metrics",
    "User Analysis"
])

# Tab 1 - Live Trades
with tab1:
    st.header("Live Trading Activity & Platform PnL")
    
    # Refresh button
    col1, col2, col3 = st.columns([1, 1, 8])
    with col1:
        if st.button("🔄 Refresh", key="refresh_pnl"):
            st.cache_data.clear()
            st.rerun()
    
    # PnL metrics row
    pnl_col1, pnl_col2 = st.columns(2)
    
    with pnl_col1:
        # Daily PnL display
        st.subheader("Today's Platform PnL")
        daily_color = "green" if st.session_state.daily_pnl >= 0 else "red"
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; background-color: rgba(0,0,0,0.05); border-radius: 10px; margin: 10px 0;">
            <h1 style="color: {daily_color}; font-size: 36px; font-weight: bold; margin: 0;">
                ${st.session_state.daily_pnl:,.2f}
            </h1>
            <p style="font-size: 14px; color: #666; margin-top: 5px;">Since 00:00 SGT</p>
        </div>
        """, unsafe_allow_html=True)
    
    with pnl_col2:
        # All-time PnL display
        st.subheader("All-Time Platform PnL")
        all_time_color = "green" if st.session_state.all_time_pnl >= 0 else "red"
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; background-color: rgba(0,0,0,0.05); border-radius: 10px; margin: 10px 0;">
            <h1 style="color: {all_time_color}; font-size: 36px; font-weight: bold; margin: 0;">
                ${st.session_state.all_time_pnl:,.2f}
            </h1>
            <p style="font-size: 14px; color: #666; margin-top: 5px;">Since Platform Launch</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Live Trades section
    st.subheader("Live Trades (Latest 100)")
    
    # Fetch live trades
    live_trades_df = fetch_live_trades()
    
    if live_trades_df is not None and not live_trades_df.empty:
        # Create display dataframe with selected columns
        display_columns = [
            'trade_time',
            'user_id_str',
            'pair_name',
            'taker_way_display',
            'taker_mode_display',
            'deal_price',
            'deal_size',
            'leverage',
            'user_pnl_usd',
            'platform_profit_share',
            'flat_fee_usd',
            'fee_mode_display'
        ]
        
        # Rename columns for better display
        column_mapping = {
            'trade_time': 'Time',
            'user_id_str': 'User ID',
            'pair_name': 'Pair',
            'taker_way_display': 'Action',
            'taker_mode_display': 'Order Type',
            'deal_price': 'Price',
            'deal_size': 'Size',
            'leverage': 'Leverage',
            'user_pnl_usd': 'User PnL ($)',
            'platform_profit_share': 'Platform Share ($)',
            'flat_fee_usd': 'Flat Fee ($)',
            'fee_mode_display': 'Fee Mode'
        }
        
        live_display_df = live_trades_df[display_columns].copy()
        live_display_df.columns = [column_mapping.get(col, col) for col in live_display_df.columns]
        
        # Format numeric columns
        live_display_df['Price'] = live_display_df['Price'].round(4)
        live_display_df['Size'] = live_display_df['Size'].round(4)
        live_display_df['User PnL ($)'] = live_display_df['User PnL ($)'].round(2)
        live_display_df['Platform Share ($)'] = live_display_df['Platform Share ($)'].round(2)
        live_display_df['Flat Fee ($)'] = live_display_df['Flat Fee ($)'].round(2)
        live_display_df['Leverage'] = live_display_df['Leverage'].apply(lambda x: f"{x:.0f}x")
        
        # Format time to be more readable
        live_display_df['Time'] = pd.to_datetime(live_display_df['Time']).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Apply color to PnL columns
        def color_pnl(val):
            if isinstance(val, (int, float)):
                color = 'green' if val >= 0 else 'red'
                return f'color: {color}'
            return ''
        
        styled_df = live_display_df.style.applymap(
            color_pnl, 
            subset=['User PnL ($)']
        )
        
        # Display the dataframe
        st.dataframe(styled_df, use_container_width=True, height=600)
        
        # Trade statistics
        st.subheader("Recent Trade Statistics")
        
        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
        
        with stats_col1:
            total_trades = len(live_trades_df)
            st.metric("Total Trades (Last 100)", f"{total_trades:,}")
        
        with stats_col2:
            total_volume = live_trades_df['volume_usd'].sum()
            st.metric("Total Volume ($)", f"${total_volume:,.2f}")
        
        with stats_col3:
            avg_leverage = live_trades_df['leverage'].mean()
            st.metric("Average Leverage", f"{avg_leverage:.1f}x")
        
        with stats_col4:
            unique_users = live_trades_df['user_id_str'].nunique()
            st.metric("Active Users", f"{unique_users:,}")
        
        # Trade type breakdown
        st.subheader("Trade Type Breakdown")
        
        type_col1, type_col2 = st.columns(2)
        
        with type_col1:
            # Trade action breakdown
            action_counts = live_trades_df['taker_way_display'].value_counts()
            fig_action = px.pie(
                values=action_counts.values,
                names=action_counts.index,
                title="Trade Actions"
            )
            st.plotly_chart(fig_action, use_container_width=True)
        
        with type_col2:
            # Order type breakdown
            order_counts = live_trades_df['taker_mode_display'].value_counts()
            fig_order = px.pie(
                values=order_counts.values,
                names=order_counts.index,
                title="Order Types"
            )
            st.plotly_chart(fig_order, use_container_width=True)
    
    else:
        st.warning("No live trade data available.")
    
    # PnL Calculation Info
    with st.expander("PnL Calculation Details"):
        components_col1, components_col2 = st.columns(2)
        
        with components_col1:
            st.info("""
            **Platform PnL Components:**
            - User Trading PnL (Inverse)
            - Flat Fee Revenue
            - Funding Fee PnL
            - Stop Loss Fees
            - Minus: All Rebates
            """)
        
        with components_col2:
            st.info("""
            **Excluded Users:**
            - 383645340185311232
            - 383645323663947776
            - 384014230417035264
            - 384014656596699136
            - 384015011585812480
            - 384015271796238336
            - 384015526326947840
            """)
    
    # Auto-refresh note
    st.caption("Live trades refresh every 30 seconds. PnL data refreshes every 60 seconds.")

# Load trading metrics and user data
with st.spinner("Loading user data..."):
    trading_metrics_df = fetch_trading_metrics()
    users_per_day_df = fetch_users_per_day()

# Check if we have data
if trading_metrics_df is not None:
    # Tab 2 - All Users
    with tab2:
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
        
        display_cols = ['user_id_str', 'total_trades', 'winning_trades', 'losing_trades', 
                        'opening_positions', 'closing_positions', 'win_percentage', 'net_pnl']
        
        display_df = filtered_df[display_cols].copy()
        display_df['net_pnl'] = display_df['net_pnl'].round(2)
        display_df['win_percentage'] = display_df['win_percentage'].round(2)
        
        st.dataframe(display_df, use_container_width=True)
        
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
    
    # Tab 3 - Trading Metrics
    with tab3:
        st.header("Trading Metrics")
        
        # Add filters and display metrics
        metrics_df = trading_metrics_df.copy()
        
        # Display trading metrics table
        st.subheader(f"Trading Metrics ({len(metrics_df)} users)")
        
        metrics_cols = ['user_id_str', 'total_trades', 'winning_trades', 'losing_trades',
                        'opening_positions', 'closing_positions', 'win_percentage',
                        'total_profit', 'total_loss', 'net_pnl', 'profit_factor']
        
        metrics_display = metrics_df[metrics_cols].copy()
        
        # Format columns
        for col in ['win_percentage', 'profit_factor']:
            metrics_display[col] = metrics_display[col].round(2)
        
        for col in ['total_profit', 'total_loss', 'net_pnl']:
            metrics_display[col] = metrics_display[col].round(2)
        
        st.dataframe(metrics_display, use_container_width=True)
    
    # Tab 4 - User Analysis
    with tab4:
        st.header("User Analysis")
        
        # User selection
        selected_user = st.selectbox(
            "Select User",
            options=trading_metrics_df['user_id_str'].tolist(),
            format_func=lambda x: f"User ID: {x}"
        )
        
        # Get user data
        user_data = trading_metrics_df[trading_metrics_df['user_id_str'] == selected_user].iloc[0]
        
        # Fetch detailed trade data
        user_trades = fetch_user_trade_details_v4(selected_user)
        
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
            
            # Compact view
            compact_cols = ['trade_time', 'pair_name', 'trade_type', 'position_action', 
                            'entry_exit_price', 'size', 'leverage_display', 
                            'user_received_pnl', 'profit_share', 'profit_share_percent']
            
            if 'percent_distance' in user_trades.columns:
                compact_cols.append('percent_distance')
            
            available_cols = [col for col in compact_cols if col in user_trades.columns]
            display_df = user_trades[available_cols].copy()
            
            # Format columns
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
            
            display_df.columns = [col_mapping.get(col, col) for col in display_df.columns]
            
            # Format numeric columns
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
            
            # PnL timeline - FIXED VERSION
            st.subheader("PnL Timeline")
            
            # Create a dataframe for the chart
            chart_df = pd.DataFrame()
            chart_df['trade_time'] = pd.to_datetime(user_trades['trade_time'])
            chart_df['position_action'] = user_trades['position_action']
            chart_df['user_received_pnl'] = user_trades['user_received_pnl']
            
            # Sort by time
            chart_df = chart_df.sort_values('trade_time')
            
            # Calculate PnL contribution (only for exit trades)
            chart_df['pnl_contribution'] = chart_df.apply(
                lambda row: row['user_received_pnl'] if row['position_action'] == 'Exit' else 0, 
                axis=1
            )
            
            # Calculate cumulative PnL
            chart_df['cumulative_pnl'] = chart_df['pnl_contribution'].cumsum()
            
            # Create the chart
            fig = px.line(
                chart_df,
                x='trade_time',
                y='cumulative_pnl',
                title='Cumulative PnL Over Time',
                labels={'trade_time': 'Trade Time', 'cumulative_pnl': 'Cumulative PnL (USD)'}
            )
            
            # Add markers
            fig.update_traces(mode='lines+markers')
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Verify cumulative PnL
            final_cumulative_pnl = chart_df['cumulative_pnl'].iloc[-1]
            expected_net_pnl = user_data['net_pnl']
            exit_mask = chart_df['position_action'] == 'Exit'
            sum_user_received_pnl = chart_df[exit_mask]['user_received_pnl'].sum()
            
            st.info(f"""
            **PnL Verification:**
            - Sum of User Received PnL (Exit trades only): ${sum_user_received_pnl:.2f}
            - Final Cumulative PnL: ${final_cumulative_pnl:.2f}
            - Expected Net PnL (from metrics): ${expected_net_pnl:.2f}
            - Difference: ${abs(final_cumulative_pnl - expected_net_pnl):.2f}
            
            The cumulative PnL is the sum of user_received_pnl for exit trades only.
            """)
            
        else:
            st.warning("No trade data available for this user.")
else:
    st.error("Failed to load user data.")

# Add refresh button in sidebar
st.sidebar.title("Dashboard Controls")
if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# Add dashboard info
st.sidebar.title("About This Dashboard")
st.sidebar.info("""
This dashboard provides comprehensive analysis of user behavior and live platform PnL.

**Live Trades:**
- Shows real-time platform PnL
- Daily PnL reset at 00:00 SGT
- All-time PnL since launch
- Excludes specified test accounts

**PnL Calculations:**
- User Received PNL = taker_pnl * collateral_price
- Profit Share = taker_share_pnl * collateral_price
- % Distance = (Pbefore - Pentry) / Pentry * 100

The cumulative PnL is the sum of user_received_pnl for exit trades only.
""")