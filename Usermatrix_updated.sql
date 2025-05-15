import streamlit as st
import pandas as pd
import psycopg2
from datetime import datetime, timedelta
import time
import numpy as np

# Page config
st.set_page_config(
    page_title="Live PnL Dashboard",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
.big-font {
    font-size:48px !important;
    font-weight: bold;
    color: #1f77b4;
}
.metric-box {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
    margin: 10px 0;
}
.positive {
    color: #00cc44;
}
.negative {
    color: #ff3333;
}
</style>
""", unsafe_allow_html=True)

# Database connection function
@st.cache_resource
def get_connection():
    # Update these with your database credentials
    conn = psycopg2.connect(
        host=st.secrets["db_host"],
        port=st.secrets["db_port"],
        database=st.secrets["db_name"],
        user=st.secrets["db_user"],
        password=st.secrets["db_password"]
    )
    return conn

# Function to execute queries
def execute_query(query):
    try:
        conn = get_connection()
        df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        st.error(f"Error executing query: {e}")
        return None

# Main PnL calculations
def get_total_platform_pnl():
    query = """
    SELECT
      -- Sum of (Platform PnL + Flat Fee Revenue + Funding Fee PnL + SL Fees) - Profit Share Rebates
      (
        -- Platform PnL from user trading
        (SELECT COALESCE(SUM(-1 * st.taker_pnl * st.collateral_price), 0)
         FROM public.trade_fill_fresh st
         WHERE st.taker_way IN (1, 2, 3, 4)
           AND CONCAT(st.taker_account_id, '') NOT IN ('383645340185311232', '383645323663947776', '384014230417035264', '384014656596699136','384015011585812480','384015271796238336','384015526326947840'))
        +
        -- Taker Fee
        (SELECT COALESCE(SUM(st.taker_fee * st.collateral_price), 0)
         FROM public.trade_fill_fresh st
         WHERE st.taker_fee_mode = 1
           AND st.taker_way IN (1, 3)
           AND CONCAT(st.taker_account_id, '') NOT IN ('383645340185311232', '383645323663947776', '384014230417035264', '384014656596699136','384015011585812480','384015271796238336','384015526326947840'))
        +
        -- Platform funding fee PnL
        (SELECT COALESCE(SUM(-1 * st.funding_fee * st.collateral_price), 0)
         FROM public.trade_fill_fresh st
         WHERE st.taker_way = 0
           AND CONCAT(st.taker_account_id, '') NOT IN ('383645340185311232', '383645323663947776', '384014230417035264', '384014656596699136','384015011585812480','384015271796238336','384015526326947840'))
        +
        -- SL Fees (taker_sl_fee * collateral_price + maker_sl_fee)
        (SELECT COALESCE(SUM(st.taker_sl_fee * st.collateral_price + st.maker_sl_fee), 0)
         FROM public.trade_fill_fresh st
         WHERE CONCAT(st.taker_account_id, '') NOT IN ('383645340185311232', '383645323663947776', '384014230417035264', '384014656596699136','384015011585812480','384015271796238336','384015526326947840'))
      )
      -
      -- Rebates (includes Flat and Profit)
      (SELECT COALESCE(SUM(st.amount * coin_price), 0)
       FROM public.user_cashbooks st
       WHERE st."remark" = '给邀请人返佣'
         AND CONCAT(st.account_id, '') NOT IN ('383645340185311232', '383645323663947776', '384014230417035264', '384014656596699136','384015011585812480','384015271796238336','384015526326947840')) AS total_platform_pnl
    """
    result = execute_query(query)
    return result['total_platform_pnl'].iloc[0] if result is not None else 0

def get_today_platform_pnl():
    query = """
    SELECT
        (
            -- Platform PnL from user trading
            COALESCE((
                SELECT SUM(-1 * st.taker_pnl * st.collateral_price)
                FROM public.trade_fill_fresh st
                WHERE st.taker_way IN (1, 2, 3, 4)
                  AND (st.created_at + INTERVAL '+8 hours' >= CAST(NOW() + INTERVAL '+8 hours' AS DATE))
                  AND (st.created_at + INTERVAL '+8 hours' < CAST((NOW() + INTERVAL '+8 hours' + INTERVAL '1 day') AS DATE))
                  AND CONCAT(st.taker_account_id, '') NOT IN ('383645340185311232', '383645323663947776', '384014230417035264', '384014656596699136','384015011585812480','384015271796238336','384015526326947840')
            ), 0)
            +
            -- taker fee
            COALESCE((
                SELECT SUM(st.taker_fee * st.collateral_price)
                FROM public.trade_fill_fresh st
                WHERE st.taker_fee_mode = 1
                  AND st.taker_way IN (1, 3)
                  AND (st.created_at + INTERVAL '+8 hours' >= CAST(NOW() + INTERVAL '+8 hours' AS DATE))
                  AND (st.created_at + INTERVAL '+8 hours' < CAST((NOW() + INTERVAL '+8 hours' + INTERVAL '1 day') AS DATE))
                  AND CONCAT(st.taker_account_id, '') NOT IN ('383645340185311232', '383645323663947776', '384014230417035264', '384014656596699136','384015011585812480','384015271796238336','384015526326947840')
            ), 0)
            +
            -- Platform funding fee PnL
            COALESCE((
                SELECT SUM(-1 * st.funding_fee * st.collateral_price)
                FROM public.trade_fill_fresh st
                WHERE st.taker_way = 0
                  AND (st.created_at + INTERVAL '+8 hours' >= CAST(NOW() + INTERVAL '+8 hours' AS DATE))
                  AND (st.created_at + INTERVAL '+8 hours' < CAST((NOW() + INTERVAL '+8 hours' + INTERVAL '1 day') AS DATE))
                  AND CONCAT(st.taker_account_id, '') NOT IN ('383645340185311232', '383645323663947776', '384014230417035264', '384014656596699136','384015011585812480','384015271796238336','384015526326947840')
            ), 0)
            +
            -- taker_sl_fee * collateral_price + maker_sl_fee
            COALESCE((
                SELECT SUM(st.taker_sl_fee * st.collateral_price + st.maker_sl_fee)
                FROM public.trade_fill_fresh st
                WHERE (st.created_at + INTERVAL '+8 hours' >= CAST(NOW() + INTERVAL '+8 hours' AS DATE))
                  AND (st.created_at + INTERVAL '+8 hours' < CAST((NOW() + INTERVAL '+8 hours' + INTERVAL '1 day') AS DATE))
                  AND CONCAT(st.taker_account_id, '') NOT IN ('383645340185311232', '383645323663947776', '384014230417035264', '384014656596699136','384015011585812480','384015271796238336','384015526326947840')
            ), 0)
        )
        -
        -- Rebates (includes Flat and Profit)
        COALESCE((
            SELECT SUM(st.amount * coin_price)
            FROM public.user_cashbooks st
            WHERE st."remark" = '给邀请人返佣'
              AND (st.created_at + INTERVAL '+8 hours' >= CAST(NOW() + INTERVAL '+8 hours' AS DATE))
              AND (st.created_at + INTERVAL '+8 hours' < CAST((NOW() + INTERVAL '+8 hours' + INTERVAL '1 day') AS DATE))
              AND CONCAT(st.account_id, '') NOT IN ('383645340185311232', '383645323663947776', '384014230417035264', '384014656596699136','384015011585812480','384015271796238336','384015526326947840')
        ), 0) AS today_platform_pnl
    """
    result = execute_query(query)
    return result['today_platform_pnl'].iloc[0] if result is not None else 0

# Function to get detailed trade data
def get_trade_details():
    query = """
    SELECT
      "Surfv2 User Login Log - Taker Account__address" AS user_address,
      "pair_name",
      "taker_type",
      "leverage",
      "deal_size",
      "coin_name",
      "deal_vol",
      "Margin_USD" AS margin_usd,
      "User_PNL" AS user_pnl,
      "Taker_Mode_Map" AS taker_mode,
      "Taker_Way_Map" AS taker_way,
      "Platform_Profit_Share" AS platform_profit_share,
      "Flat Fee" AS flat_fee,
      "Time"
    FROM (
      SELECT
        "source"."Surfv2 User Login Log - Taker Account__address" AS "Surfv2 User Login Log - Taker Account__address",
        "source"."pair_name" AS "pair_name",
        "source"."taker_type" AS "taker_type",
        "source"."leverage" AS "leverage",
        "source"."deal_size" AS "deal_size",
        "source"."coin_name" AS "coin_name",
        "source"."deal_vol" AS "deal_vol",
        "source"."Margin_USD" AS "Margin_USD",
        "source"."User_PNL" AS "User_PNL",
        "source"."Taker_Mode_Map" AS "Taker_Mode_Map",
        "source"."Taker_Way_Map" AS "Taker_Way_Map",
        "source"."Platform_Profit_Share" AS "Platform_Profit_Share",
        "source"."Flat Fee" AS "Flat Fee",
        "source"."Time" AS "Time"
      FROM
        (
          SELECT
            "public"."surfv2_trade"."id" AS "id",
            "public"."surfv2_trade"."pair_name" AS "pair_name",
            "public"."surfv2_trade"."deal_price" AS "deal_price",
            "public"."surfv2_trade"."deal_vol" AS "deal_vol",
            "public"."surfv2_trade"."deal_size" AS "deal_size",
            "public"."surfv2_trade"."leverage" AS "leverage",
            "public"."surfv2_trade"."coin_name" AS "coin_name",
            "public"."surfv2_trade"."taker_type" AS "taker_type",
            "public"."surfv2_trade"."taker_fee" AS "taker_fee",
            "public"."surfv2_trade"."taker_pnl" AS "taker_pnl",
            "public"."surfv2_trade"."created_at" AS "created_at",
            "public"."surfv2_trade"."taker_share_pnl" AS "taker_share_pnl",
            CASE
              WHEN "public"."surfv2_trade"."taker_mode" = 1 THEN 'Active'
              WHEN "public"."surfv2_trade"."taker_mode" = 2 THEN 'Take Profit'
              WHEN "public"."surfv2_trade"."taker_mode" = 3 THEN 'Stop Loss'
              WHEN "public"."surfv2_trade"."taker_mode" = 4 THEN 'Liquidation'
            END AS "Taker_Mode_Map",
            "public"."surfv2_trade"."taker_pnl" * "public"."surfv2_trade"."collateral_price" AS "User_PNL",
            CASE
              WHEN "public"."surfv2_trade"."taker_way" = 1 THEN 'Open Long'
              WHEN "public"."surfv2_trade"."taker_way" = 2 THEN 'Close Short'
              WHEN "public"."surfv2_trade"."taker_way" = 3 THEN 'Open Short'
              WHEN "public"."surfv2_trade"."taker_way" = 4 THEN 'Close Long'
            END AS "Taker_Way_Map",
            "public"."surfv2_trade"."taker_share_pnl" * "public"."surfv2_trade"."collateral_price" AS "Platform_Profit_Share",
            TO_CHAR(("public"."surfv2_trade"."created_at" + INTERVAL '8 hour'), 'YYYY-MM-DD HH24:MI:SS') AS "Time",
            "public"."surfv2_trade"."taker_fee" * "public"."surfv2_trade"."collateral_price" AS "Flat Fee",
            "public"."surfv2_trade"."deal_vol" * "public"."surfv2_trade"."collateral_price" AS "Margin_USD",
            "Surfv2 User Login Log - Taker Account"."address" AS "Surfv2 User Login Log - Taker Account__address"
          FROM
            "public"."surfv2_trade"
           
          LEFT JOIN "public"."surfv2_user_login_log" AS "Surfv2 User Login Log - Taker Account" 
            ON "public"."surfv2_trade"."taker_account_id" = "Surfv2 User Login Log - Taker Account"."account_id"
          WHERE
            (
              ("public"."surfv2_trade"."taker_way" = 1)
              OR ("public"."surfv2_trade"."taker_way" = 2)
              OR ("public"."surfv2_trade"."taker_way" = 3)
              OR ("public"."surfv2_trade"."taker_way" = 4)
            )
           
         AND (
              ("public"."surfv2_trade"."taker_mode" = 1)
              OR ("public"."surfv2_trade"."taker_mode" = 4)
              OR ("public"."surfv2_trade"."taker_mode" = 2)
              OR ("public"."surfv2_trade"."taker_mode" = 3)
            )
        ) AS "source"
    ) AS "final"
    ORDER BY "Time" DESC
    LIMIT 100
    """
    
    result = execute_query(query)
    return result

# Main app layout
def main():
    st.title("🚀 Live PnL Dashboard")
    
    # Add refresh button
    col1, col2, col3 = st.columns([1, 1, 5])
    with col1:
        if st.button("🔄 Refresh", type="primary"):
            st.rerun()
    
    with col2:
        # Auto-refresh toggle
        auto_refresh = st.checkbox("Auto-refresh", value=True)
        
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(5)  # Refresh every 5 seconds
        st.rerun()
    
    # Display current time
    st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Main metrics
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Total Platform PnL")
        total_pnl = get_total_platform_pnl()
        color_class = "positive" if total_pnl >= 0 else "negative"
        st.markdown(f'<div class="metric-box"><p class="big-font {color_class}">${total_pnl:,.2f}</p></div>', 
                   unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Today's Platform PnL")
        today_pnl = get_today_platform_pnl()
        color_class = "positive" if today_pnl >= 0 else "negative"
        st.markdown(f'<div class="metric-box"><p class="big-font {color_class}">${today_pnl:,.2f}</p></div>', 
                   unsafe_allow_html=True)
    
    # Detailed trade data
    st.markdown("---")
    st.markdown("### Recent Trades")
    
    trade_data = get_trade_details()
    
    if trade_data is not None and not trade_data.empty:
        # Format numeric columns
        numeric_columns = ['leverage', 'deal_size', 'margin_usd', 'user_pnl', 
                          'platform_profit_share', 'flat_fee']
        
        for col in numeric_columns:
            if col in trade_data.columns:
                trade_data[col] = pd.to_numeric(trade_data[col], errors='coerce')
        
        # Format display
        display_data = trade_data.copy()
        display_data['margin_usd'] = display_data['margin_usd'].apply(lambda x: f'${x:,.2f}' if pd.notna(x) else '')
        display_data['user_pnl'] = display_data['user_pnl'].apply(lambda x: f'${x:,.2f}' if pd.notna(x) else '')
        display_data['platform_profit_share'] = display_data['platform_profit_share'].apply(lambda x: f'${x:,.2f}' if pd.notna(x) else '')
        display_data['flat_fee'] = display_data['flat_fee'].apply(lambda x: f'${x:,.2f}' if pd.notna(x) else '')
        
        # Display dataframe with formatting
        st.dataframe(
            display_data,
            column_config={
                "user_address": st.column_config.TextColumn("User Address", width="medium"),
                "pair_name": st.column_config.TextColumn("Pair", width="small"),
                "taker_type": st.column_config.NumberColumn("Taker Type", width="small"),
                "leverage": st.column_config.NumberColumn("Leverage", width="small"),
                "deal_size": st.column_config.NumberColumn("Deal Size", width="small"),
                "coin_name": st.column_config.TextColumn("Coin", width="small"),
                "deal_vol": st.column_config.NumberColumn("Volume", width="small"),
                "margin_usd": st.column_config.TextColumn("Margin USD", width="small"),
                "user_pnl": st.column_config.TextColumn("User PnL", width="small"),
                "taker_mode": st.column_config.TextColumn("Mode", width="small"),
                "taker_way": st.column_config.TextColumn("Direction", width="small"),
                "platform_profit_share": st.column_config.TextColumn("Platform Share", width="small"),
                "flat_fee": st.column_config.TextColumn("Flat Fee", width="small"),
                "Time": st.column_config.TextColumn("Time", width="medium"),
            },
            hide_index=True,
            use_container_width=True
        )
        
        # Summary statistics
        st.markdown("---")
        st.markdown("### Summary Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", len(trade_data))
        
        with col2:
            total_volume = trade_data['margin_usd'].sum()
            st.metric("Total Volume", f"${total_volume:,.2f}")
        
        with col3:
            avg_deal_size = trade_data['deal_size'].mean()
            st.metric("Avg Deal Size", f"{avg_deal_size:,.2f}")
        
        with col4:
            total_fees = trade_data['flat_fee'].sum()
            st.metric("Total Fees", f"${total_fees:,.2f}")
    else:
        st.warning("No trade data available")

if __name__ == "__main__":
    main()