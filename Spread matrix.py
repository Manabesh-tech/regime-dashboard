import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import pytz

# Page configuration
st.set_page_config(
    page_title="Exchange Spread Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Apply custom CSS styling - more minimal design with centered numeric columns and simplified styling
st.markdown("""
<style>
    .header-style {
        font-size:24px !important;
        font-weight: bold;
        padding: 10px 0;
    }
    .subheader-style {
        font-size:20px !important;
        font-weight: bold;
        padding: 5px 0;
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 15px;
        border: 1px solid #e9ecef;
    }
    /* Simplified tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 15px;
        background-color: #f5f5f5;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4682B4;
        color: white;
    }
    
    /* Center numeric columns in dataframes */
    .dataframe th, .dataframe td {
        text-align: center !important;
        font-family: monospace;  /* Use monospace for better number alignment */
    }
    /* First column (Token) remains left-aligned */
    .dataframe th:first-child, .dataframe td:first-child {
        text-align: left !important;
        font-family: inherit;  /* Use normal font for token names */
    }
    
    /* Bold formatting for spread values */
    .spread-value {
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- Database Configuration ---
try:
    # Try to get database config from secrets
    db_config = st.secrets["database"]
    db_uri = (
        f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}"
        f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    )
    engine = create_engine(
        db_uri,
        isolation_level="AUTOCOMMIT",  # 设置自动提交模式
        pool_size=5,  # 连接池大小
        max_overflow=10,  # 最大溢出连接数
        pool_timeout=30,  # 连接超时时间
        pool_recycle=1800,  # 连接回收时间(30分钟)
        pool_pre_ping=True,  # 使用连接前先测试连接是否有效
        pool_use_lifo=True,  # 使用后进先出,减少空闲连接
        echo=False  # 不打印 SQL 语句
    )
    st.sidebar.success("Connected to database successfully")
except Exception as e:
    # If secrets not available, allow manual entry
    st.sidebar.error(f"Error connecting to the database: {e}")
    
    # Provide manual connection option
    st.sidebar.header("Database Connection")
    db_user = st.sidebar.text_input("Database Username")
    db_password = st.sidebar.text_input("Database Password", type="password")
    db_host = st.sidebar.text_input("Database Host")
    db_port = st.sidebar.text_input("Database Port", "5432")
    db_name = st.sidebar.text_input("Database Name")
    
    if st.sidebar.button("Connect to Database"):
        try:
            db_uri = (
                f"postgresql+psycopg2://{db_user}:{db_password}"
                f"@{db_host}:{db_port}/{db_name}"
            )
            engine = create_engine(
                db_uri,
                isolation_level="AUTOCOMMIT",  # 设置自动提交模式
                pool_size=5,  # 连接池大小
                max_overflow=10,  # 最大溢出连接数
                pool_timeout=30,  # 连接超时时间
                pool_recycle=1800,  # 连接回收时间(30分钟)
                pool_pre_ping=True,  # 使用连接前先测试连接是否有效
                pool_use_lifo=True,  # 使用后进先出,减少空闲连接
                echo=False  # 不打印 SQL 语句
            )
            st.sidebar.success("Connected to database successfully")
        except Exception as e:
            st.sidebar.error(f"Failed to connect: {e}")
            st.stop()
    else:
        st.error("Please connect to the database to continue")
        st.stop()

# --- Constants and Configuration ---
# Define exchanges
exchanges = ["binanceFuture", "gateFuture", "hyperliquidFuture", "surfFuture"]
exchanges_display = {
    "binanceFuture": "Binance",
    "gateFuture": "Gate",
    "hyperliquidFuture": "Hyperliquid",
    "surfFuture": "SurfFuture"
}

# Define time parameters
interval_minutes = 10  # 10-minute intervals
singapore_timezone = pytz.timezone('Asia/Singapore')
lookback_days = 1  # Fixed to 1 day

# Use a fixed scale factor for consistency across all tabs
scale_factor = 10000
scale_label = "× 10,000"

# --- Utility Functions ---
# Function to convert time string to sortable minutes value
def time_to_minutes(time_str):
    hours, minutes = map(int, time_str.split(':'))
    return hours * 60 + minutes

# Function to determine if a token is a major
def is_major(token):
    majors = ["BTC", "ETH", "SOL", "XRP", "BNB"]
    for major in majors:
        if major in token:
            return True
    return False

# Function to get appropriate depth tiers based on token type
def get_depth_tiers(token):
    if is_major(token):
        return [50000, 100000, 200000, 500000]  # Majors: 50k, 100k, 200k, 500k
    else:
        return [20000, 50000, 100000, 200000]   # Altcoins: 20k, 50k, 100k, 200k

# Define mappings for depth labels that can be reused across tabs
fee_depth_map_major = {
    'fee1': '50K', 'fee2': '100K', 'fee3': '200K', 'fee4': '500K',
    'avg_fee1': '50K', 'avg_fee2': '100K', 'avg_fee3': '200K', 'avg_fee4': '500K'
}
fee_depth_map_altcoin = {
    'fee1': '20K', 'fee2': '50K', 'fee3': '100K', 'fee4': '200K',
    'avg_fee1': '20K', 'avg_fee2': '50K', 'avg_fee3': '100K', 'avg_fee4': '200K'
}

# Function to get depth label based on token type
def get_depth_label(fee_column, token):
    if is_major(token):
        return fee_depth_map_major.get(fee_column, fee_column)
    else:
        return fee_depth_map_altcoin.get(fee_column, fee_column)

# Function to format number with 2 decimal places and no trailing zeros
def format_number(x):
    if pd.isna(x):
        return ""
    # Format to 2 decimal places and remove trailing zeros
    formatted = f"{float(x):.2f}"
    # Remove trailing zeros and decimal point if it's a whole number
    return formatted.rstrip('0').rstrip('.') if '.' in formatted else formatted

# Function to apply bold formatting to spread values
def color_code_value(value, thresholds=None):
    """Apply bold formatting to spread values"""
    if pd.isna(value):
        return ""
    
    # Format with 2 decimal places and use bold
    formatted = format_number(value)
    return f'<span class="spread-value">{formatted}</span>'

# --- Data Fetching Functions ---
@st.cache_data(ttl=600, show_spinner="Fetching tokens...")
def fetch_all_tokens():
    """Fetch all available tokens from the database"""
    query = """
    SELECT DISTINCT pair_name 
    FROM oracle_exchange_fee 
    ORDER BY pair_name
    """
    try:
        df = pd.read_sql(query, engine)
        if df.empty:
            st.error("No tokens found in the database.")
            return []
        return df['pair_name'].tolist()
    except Exception as e:
        st.error(f"Error fetching tokens: {e}")
        print(f"Error fetching tokens: {e}")
        return ["BTC/USDT", "ETH/USDT", "SOL/USDT"]  # Default fallback

@st.cache_data(ttl=600, show_spinner="Fetching daily spread averages...")
def fetch_daily_spread_averages(tokens):
    """Fetch daily spread averages for multiple tokens (last 24 hours)"""
    try:
        # Get current time in Singapore timezone
        now_utc = datetime.now(pytz.utc)
        now_sg = now_utc.astimezone(singapore_timezone)
        start_time_sg = now_sg - timedelta(days=lookback_days)
        
        # Convert back to UTC for database query
        start_time_utc = start_time_sg.astimezone(pytz.utc)
        end_time_utc = now_sg.astimezone(pytz.utc)
        
        # Create placeholders for tokens
        tokens_str = "', '".join(tokens)

        # Query to get average fee data for all selected tokens
        # Special handling for SurfFuture: Use total_fee for all fee levels
        query = f"""
        SELECT 
            pair_name,
            source,
            CASE WHEN source = 'surfFuture' THEN AVG(total_fee) ELSE AVG(fee1) END as avg_fee1,
            CASE WHEN source = 'surfFuture' THEN AVG(total_fee) ELSE AVG(fee2) END as avg_fee2,
            CASE WHEN source = 'surfFuture' THEN AVG(total_fee) ELSE AVG(fee3) END as avg_fee3,
            CASE WHEN source = 'surfFuture' THEN AVG(total_fee) ELSE AVG(fee4) END as avg_fee4,
            AVG(total_fee) as avg_total_fee
        FROM 
            oracle_exchange_fee
        WHERE 
            pair_name IN ('{tokens_str}')
            AND time_group BETWEEN '{start_time_utc}' AND '{end_time_utc}'
            AND source IN ('binanceFuture', 'gateFuture', 'hyperliquidFuture', 'surfFuture')
        GROUP BY 
            pair_name, source
        ORDER BY 
            pair_name, source
        """
        
        df = pd.read_sql(query, engine)
        
        if df.empty:
            return None

        return df
            
    except Exception as e:
        st.error(f"Error fetching daily spread averages: {e}")
        print(f"Error fetching daily spread averages: {e}")
        return None

@st.cache_data(ttl=600, show_spinner="Fetching 10-minute spread data...")
def fetch_10min_spread_data(token):
    """Fetch 10-minute spread data for a specific token (last 24 hours)"""
    try:
        # Get current time in Singapore timezone
        now_utc = datetime.now(pytz.utc)
        now_sg = now_utc.astimezone(singapore_timezone)
        start_time_sg = now_sg - timedelta(days=lookback_days)
        
        # Convert back to UTC for database query
        start_time_utc = start_time_sg.astimezone(pytz.utc)
        end_time_utc = now_sg.astimezone(pytz.utc)

        # Query to get the fee data for the specified token
        # Special handling for SurfFuture: Use total_fee for all fee levels
        query = f"""
        SELECT 
            time_group AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore' AS timestamp,
            pair_name,
            source,
            CASE WHEN source = 'surfFuture' THEN total_fee ELSE fee1 END as fee1,
            CASE WHEN source = 'surfFuture' THEN total_fee ELSE fee2 END as fee2,
            CASE WHEN source = 'surfFuture' THEN total_fee ELSE fee3 END as fee3,
            CASE WHEN source = 'surfFuture' THEN total_fee ELSE fee4 END as fee4,
            total_fee
        FROM 
            oracle_exchange_fee
        WHERE 
            pair_name = '{token}'
            AND time_group BETWEEN '{start_time_utc}' AND '{end_time_utc}'
            AND source IN ('binanceFuture', 'gateFuture', 'hyperliquidFuture', 'surfFuture')
        ORDER BY 
            timestamp ASC,
            source
        """
        
        df = pd.read_sql(query, engine)
        
        if df.empty:
            return None

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
            
    except Exception as e:
        print(f"[{token}] Error fetching spread data: {e}")
        import traceback
        traceback.print_exc()
        return None

def calculate_matrix_data(avg_data):
    """Transform the average data into matrix format for display"""
    if avg_data is None or avg_data.empty:
        return None
    
    # Create pivot tables for each fee level
    matrix_data = {}
    fee_columns = [col for col in ['avg_fee1', 'avg_fee2', 'avg_fee3', 'avg_fee4'] 
                  if col in avg_data.columns]
    
    if not fee_columns:
        st.error("Required fee columns not found in the data. Check database schema.")
        return None
    
    for fee_col in fee_columns:
        try:
            # Convert the long format to wide format (pivot)
            pivot_df = avg_data.pivot(index='pair_name', columns='source', values=fee_col).reset_index()
            
            # Rename columns to display names
            for source in exchanges:
                if source in pivot_df.columns:
                    pivot_df = pivot_df.rename(columns={source: exchanges_display[source]})
            
            # Calculate non-surf average
            non_surf_columns = ['Binance', 'Gate', 'Hyperliquid']
            available_non_surf = [col for col in non_surf_columns if col in pivot_df.columns]
            
            if available_non_surf:
                pivot_df['Avg (Non-Surf)'] = pivot_df[available_non_surf].mean(axis=1)
            
            # Add a column indicating if SurfFuture is better than non-surf avg
            if 'SurfFuture' in pivot_df.columns and 'Avg (Non-Surf)' in pivot_df.columns:
                pivot_df['Surf Better'] = pivot_df['SurfFuture'] < pivot_df['Avg (Non-Surf)']
                
                # Calculate percentage improvement
                pivot_df['Improvement %'] = ((pivot_df['Avg (Non-Surf)'] - pivot_df['SurfFuture']) / 
                                            pivot_df['Avg (Non-Surf)'] * 100).round(2)
            
            # Store the pivot table for this fee level
            matrix_data[fee_col.replace('avg_', '')] = pivot_df
        except Exception as e:
            st.error(f"Error processing {fee_col}: {e}")
    
    return matrix_data

# --- Main Application ---
st.markdown('<div class="header-style">Exchange Spread Analysis Dashboard</div>', unsafe_allow_html=True)

# Fetch all available tokens
all_tokens = fetch_all_tokens()

# Sidebar - simplified, just refresh button
st.sidebar.header("Controls")

# Always select all tokens
selected_tokens = all_tokens

# Add a refresh button
if st.sidebar.button("Refresh Data", use_container_width=True):
    st.cache_data.clear()
    st.experimental_rerun()

# Create tabs for different analyses
tab1, tab2, tab3 = st.tabs(["50K/20K Analysis", "100K/50K Analysis", "Spread By Size"])

# Get daily spread data (used in all tabs)
daily_avg_data = fetch_daily_spread_averages(selected_tokens)
if daily_avg_data is None or daily_avg_data.empty:
    st.warning("No spread data available for the selected time period.")
    st.stop()

# Calculate matrix data for all fee levels
matrix_data = calculate_matrix_data(daily_avg_data)
if matrix_data is None or not matrix_data:
    st.warning("Unable to process spread data. Check log for details.")
    st.stop()

# === TAB 1: 50K/20K ANALYSIS (fee1) ===
with tab1:
    st.markdown('<div class="header-style">50K/20K Spread Analysis</div>', unsafe_allow_html=True)
    
    # Display explanation of depth tiers
    st.markdown("""
    <div class="info-box">
    <b>Trading Size Definition:</b><br>
    • <b>Major tokens</b> (BTC, ETH, SOL, XRP, BNB): 50K<br>
    • <b>Altcoin tokens</b>: 20K<br>
    <br>
    This tab shows daily averages of 10-minute spread data points at 50K/20K size.
    </div>
    """, unsafe_allow_html=True)
    
    if 'fee1' in matrix_data:
        df = matrix_data['fee1'].copy()
        
        st.markdown(f"<div class='info-box'><b>Note:</b> All spread values are multiplied by {scale_factor} ({scale_label}) for better readability.</div>", unsafe_allow_html=True)
        
        # Calculate numeric columns for scaling
        numeric_cols = [col for col in df.columns if col not in ['pair_name', 'Surf Better', 'Improvement %']]
        
        # Apply scaling
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col] * scale_factor
        
        # Format and display the dataframe
        display_df = df.copy()
        
        # Add token type column for clarity
        display_df['Token Type'] = display_df['pair_name'].apply(
            lambda x: 'Major' if is_major(x) else 'Altcoin'
        )
        
        # Sort by token type and then by name
        display_df = display_df.sort_values(by=['Token Type', 'pair_name'])
        
        # Define column order with SurfFuture at the end
        desired_order = ['pair_name', 'Token Type', 'Binance', 'Gate', 'Hyperliquid', 'Avg (Non-Surf)', 'SurfFuture']
        if 'Improvement %' in display_df.columns:
            desired_order.append('Improvement %')
        ordered_columns = [col for col in desired_order if col in display_df.columns]
        
        # Add Surf Better column if it exists (hidden, used for filtering)
        if 'Surf Better' in display_df.columns:
            ordered_columns.append('Surf Better')
        
        display_df = display_df[ordered_columns]
        
        # Rename columns for display
        display_df = display_df.rename(columns={'pair_name': 'Token'})
        
        # Apply formatting to numeric columns (with error checking)
        color_df = display_df.copy()
        for col in numeric_cols:
            if col in color_df.columns and col != 'Token Type':
                color_df[col] = color_df[col].apply(lambda x: color_code_value(x) if not pd.isna(x) else "")
        
        # Special formatting for improvement percentage
        if 'Improvement %' in color_df.columns:
            color_df['Improvement %'] = color_df['Improvement %'].apply(
                lambda x: f'<span style="color:green;font-weight:bold">+{x:.2f}%</span>' if x > 0 else 
                (f'<span style="color:red">-{abs(x):.2f}%</span>' if x < 0 else f'{x:.2f}%')
            )
        
        # Display the table with HTML formatting
        html_table = color_df.to_html(escape=False, index=False)
        st.markdown(html_table, unsafe_allow_html=True)
        
        # Visualization - Pie chart showing proportion of tokens where SurfFuture is better
        if 'Surf Better' in df.columns:
            st.markdown("### SurfFuture Performance Analysis")
            
            # Count tokens where SurfFuture is better
            surf_better_count = df['Surf Better'].sum()
            total_count = len(df)
            surf_worse_count = total_count - surf_better_count
            
            # Create pie chart
            fig = px.pie(
                values=[surf_better_count, surf_worse_count],
                names=['SurfFuture Better', 'Other Exchanges Better'],
                title="Proportion of Tokens Where SurfFuture Has Better Spreads",
                color_discrete_sequence=['#4CAF50', '#FFC107'],
                hole=0.4
            )
            
            # Update layout
            fig.update_layout(
                legend=dict(orientation='h', yanchor='bottom', y=-0.2),
                margin=dict(t=60, b=60, l=20, r=20),
                height=400
            )
            
            # Display percentage text in middle
            better_percentage = surf_better_count / total_count * 100 if total_count > 0 else 0
            fig.add_annotation(
                text=f"{better_percentage:.1f}%<br>Better",
                x=0.5, y=0.5,
                font_size=20,
                showarrow=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Show summary of SurfFuture performance
        if 'SurfFuture' in display_df.columns and 'Avg (Non-Surf)' in display_df.columns:
            surf_values = df['SurfFuture'].dropna()
            nonsurf_values = df['Avg (Non-Surf)'].dropna()
            
            if not surf_values.empty and not nonsurf_values.empty:
                # Match indices to compare only pairs with both values
                common_indices = surf_values.index.intersection(nonsurf_values.index)
                if len(common_indices) > 0:
                    surf_better_count = sum(surf_values.loc[common_indices] < nonsurf_values.loc[common_indices])
                    total_count = len(common_indices)
                    
                    # Calculate percentages
                    surf_better_pct = surf_better_count/total_count*100 if total_count > 0 else 0
                    
                    # Display summary box
                    st.markdown(f"""
                    <div class="info-box">
                    <b>SurfFuture Performance Summary:</b><br>
                    • SurfFuture has tighter spreads for {surf_better_count}/{total_count} tokens ({surf_better_pct:.1f}%)<br>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Calculate averages
                    surf_avg = surf_values.mean()
                    nonsurf_avg = nonsurf_values.mean()
                    
                    if surf_avg < nonsurf_avg:
                        improvement = ((nonsurf_avg - surf_avg) / nonsurf_avg) * 100
                        st.success(f"📉 **SurfFuture average spread ({format_number(surf_avg * scale_factor)}) is {improvement:.2f}% lower than other exchanges ({format_number(nonsurf_avg * scale_factor)})**")
                    
            # Calculate separate stats for majors and altcoins
            major_tokens_df = df[df['pair_name'].apply(is_major)]
            altcoin_tokens_df = df[~df['pair_name'].apply(is_major)]
            
            # For Major tokens
            if not major_tokens_df.empty:
                surf_major = major_tokens_df['SurfFuture'].dropna()
                nonsurf_major = major_tokens_df['Avg (Non-Surf)'].dropna()
                
                if not surf_major.empty and not nonsurf_major.empty:
                    common_indices = surf_major.index.intersection(nonsurf_major.index)
                    if len(common_indices) > 0:
                        surf_better_count = sum(surf_major.loc[common_indices] < nonsurf_major.loc[common_indices])
                        total_count = len(common_indices)
                        
                        if total_count > 0:
                            surf_major_avg = surf_major.mean()
                            nonsurf_major_avg = nonsurf_major.mean()
                            
                            st.markdown(f"""
                            <div class="info-box">
                            <b>Major Tokens (50K):</b><br>
                            • SurfFuture has tighter spreads for {surf_better_count}/{total_count} major tokens<br>
                            • SurfFuture average: {format_number(surf_major_avg)} vs Non-Surf average: {format_number(nonsurf_major_avg)}
                            </div>
                            """, unsafe_allow_html=True)
            
            # For Altcoin tokens
            if not altcoin_tokens_df.empty:
                surf_altcoin = altcoin_tokens_df['SurfFuture'].dropna()
                nonsurf_altcoin = altcoin_tokens_df['Avg (Non-Surf)'].dropna()
                
                if not surf_altcoin.empty and not nonsurf_altcoin.empty:
                    common_indices = surf_altcoin.index.intersection(nonsurf_altcoin.index)
                    if len(common_indices) > 0:
                        surf_better_count = sum(surf_altcoin.loc[common_indices] < nonsurf_altcoin.loc[common_indices])
                        total_count = len(common_indices)
                        
                        if total_count > 0:
                            surf_altcoin_avg = surf_altcoin.mean()
                            nonsurf_altcoin_avg = nonsurf_altcoin.mean()
                            
                            st.markdown(f"""
                            <div class="info-box">
                            <b>Altcoin Tokens (20K):</b><br>
                            • SurfFuture has tighter spreads for {surf_better_count}/{total_count} altcoin tokens<br>
                            • SurfFuture average: {format_number(surf_altcoin_avg)} vs Non-Surf average: {format_number(nonsurf_altcoin_avg)}
                            </div>
                            """, unsafe_allow_html=True)
                            
            # Bar chart comparing exchanges
            st.markdown("### Average Spread by Exchange")
            
            # Calculate average for each exchange
            exchange_avgs = {}
            for exchange in ['Binance', 'Gate', 'Hyperliquid', 'SurfFuture']:
                if exchange in df.columns:
                    values = df[exchange].dropna()
                    if not values.empty:
                        exchange_avgs[exchange] = values.mean()
            
            if exchange_avgs:
                # Create data frame for plotting
                avg_df = pd.DataFrame({
                    'Exchange': list(exchange_avgs.keys()),
                    'Average Spread': list(exchange_avgs.values())
                })
                
                # Sort by average spread (ascending)
                avg_df = avg_df.sort_values('Average Spread')
                
                # Create bar chart
                fig = px.bar(
                    avg_df,
                    x='Exchange',
                    y='Average Spread',
                    title=f"Average Spread by Exchange ({scale_label})",
                    color='Exchange',
                    text='Average Spread',
                    color_discrete_map={
                        'SurfFuture': '#4CAF50',
                        'Binance': '#2196F3',
                        'Gate': '#FFC107',
                        'Hyperliquid': '#FF5722'
                    }
                )
                
                # Format the bars
                fig.update_traces(
                    texttemplate='%{y:.2f}',
                    textposition='outside'
                )
                
                # Format the layout
                fig.update_layout(
                    xaxis_title="Exchange",
                    yaxis_title=f"Average Spread {scale_label}",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No fee1 (50K/20K) data available for analysis")

# === TAB 2: 100K/50K ANALYSIS (fee2) ===
with tab2:
    st.markdown('<div class="header-style">100K/50K Spread Analysis</div>', unsafe_allow_html=True)
    
    # Display explanation of depth tiers
    st.markdown("""
    <div class="info-box">
    <b>Trading Size Definition:</b><br>
    • <b>Major tokens</b> (BTC, ETH, SOL, XRP, BNB): 100K<br>
    • <b>Altcoin tokens</b>: 50K<br>
    <br>
    This tab shows daily averages of 10-minute spread data points at 100K/50K size.
    </div>
    """, unsafe_allow_html=True)
    
    if 'fee2' in matrix_data:
        df = matrix_data['fee2'].copy()
        
        st.markdown(f"<div class='info-box'><b>Note:</b> All spread values are multiplied by {scale_factor} ({scale_label}) for better readability.</div>", unsafe_allow_html=True)
        
        # Calculate numeric columns for scaling
        numeric_cols = [col for col in df.columns if col not in ['pair_name', 'Surf Better', 'Improvement %']]
        
        # Apply scaling
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col] * scale_factor
        
        # Format and display the dataframe
        display_df = df.copy()
        
        # Add token type column for clarity
        display_df['Token Type'] = display_df['pair_name'].apply(
            lambda x: 'Major' if is_major(x) else 'Altcoin'
        )
        
        # Sort by token type and then by name
        display_df = display_df.sort_values(by=['Token Type', 'pair_name'])
        
        # Define column order with SurfFuture at the end
        desired_order = ['pair_name', 'Token Type', 'Binance', 'Gate', 'Hyperliquid', 'Avg (Non-Surf)', 'SurfFuture']
        if 'Improvement %' in display_df.columns:
            desired_order.append('Improvement %')
        ordered_columns = [col for col in desired_order if col in display_df.columns]
        # Add Surf Better column if it exists (hidden, used for filtering)
        if 'Surf Better' in display_df.columns:
            ordered_columns.append('Surf Better')
        
        display_df = display_df[ordered_columns]
        
        # Rename columns for display
        display_df = display_df.rename(columns={'pair_name': 'Token'})
        
        # Apply formatting to numeric columns (with error checking)
        color_df = display_df.copy()
        for col in numeric_cols:
            if col in color_df.columns and col != 'Token Type':
                color_df[col] = color_df[col].apply(lambda x: color_code_value(x) if not pd.isna(x) else "")
        
        # Special formatting for improvement percentage
        if 'Improvement %' in color_df.columns:
            color_df['Improvement %'] = color_df['Improvement %'].apply(
                lambda x: f'<span style="color:green;font-weight:bold">+{x:.2f}%</span>' if x > 0 else 
                (f'<span style="color:red">-{abs(x):.2f}%</span>' if x < 0 else f'{x:.2f}%')
            )
        
        # Display the table with HTML formatting
        html_table = color_df.to_html(escape=False, index=False)
        st.markdown(html_table, unsafe_allow_html=True)
        
        # Visualization - Pie chart showing proportion of tokens where SurfFuture is better
        if 'Surf Better' in df.columns:
            st.markdown("### SurfFuture Performance Analysis")
            
            # Count tokens where SurfFuture is better
            surf_better_count = df['Surf Better'].sum()
            total_count = len(df)
            surf_worse_count = total_count - surf_better_count
            
            # Create pie chart
            fig = px.pie(
                values=[surf_better_count, surf_worse_count],
                names=['SurfFuture Better', 'Other Exchanges Better'],
                title="Proportion of Tokens Where SurfFuture Has Better Spreads",
                color_discrete_sequence=['#4CAF50', '#FFC107'],
                hole=0.4
            )
            
            # Update layout
            fig.update_layout(
                legend=dict(orientation='h', yanchor='bottom', y=-0.2),
                margin=dict(t=60, b=60, l=20, r=20),
                height=400
            )
            
            # Display percentage text in middle
            better_percentage = surf_better_count / total_count * 100 if total_count > 0 else 0
            fig.add_annotation(
                text=f"{better_percentage:.1f}%<br>Better",
                x=0.5, y=0.5,
                font_size=20,
                showarrow=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Show summary of SurfFuture performance
        if 'SurfFuture' in display_df.columns and 'Avg (Non-Surf)' in display_df.columns:
            surf_values = df['SurfFuture'].dropna()
            nonsurf_values = df['Avg (Non-Surf)'].dropna()
            
            if not surf_values.empty and not nonsurf_values.empty:
                # Match indices to compare only pairs with both values
                common_indices = surf_values.index.intersection(nonsurf_values.index)
                if len(common_indices) > 0:
                    surf_better_count = sum(surf_values.loc[common_indices] < nonsurf_values.loc[common_indices])
                    total_count = len(common_indices)
                    
                    # Calculate percentages
                    surf_better_pct = surf_better_count/total_count*100 if total_count > 0 else 0
                    
                    # Display summary box
                    st.markdown(f"""
                    <div class="info-box">
                    <b>SurfFuture Performance Summary:</b><br>
                    • SurfFuture has tighter spreads for {surf_better_count}/{total_count} tokens ({surf_better_pct:.1f}%)<br>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Calculate averages
                    surf_avg = surf_values.mean()
                    nonsurf_avg = nonsurf_values.mean()
                    
                    if surf_avg < nonsurf_avg:
                        improvement = ((nonsurf_avg - surf_avg) / nonsurf_avg) * 100
                        st.success(f"📉 **SurfFuture average spread ({format_number(surf_avg * scale_factor)}) is {improvement:.2f}% lower than other exchanges ({format_number(nonsurf_avg * scale_factor)})**")
                    
            # Calculate separate stats for majors and altcoins
            major_tokens_df = df[df['pair_name'].apply(is_major)]
            altcoin_tokens_df = df[~df['pair_name'].apply(is_major)]
            
            # For Major tokens
            if not major_tokens_df.empty:
                surf_major = major_tokens_df['SurfFuture'].dropna()
                nonsurf_major = major_tokens_df['Avg (Non-Surf)'].dropna()
                
                if not surf_major.empty and not nonsurf_major.empty:
                    common_indices = surf_major.index.intersection(nonsurf_major.index)
                    if len(common_indices) > 0:
                        surf_better_count = sum(surf_major.loc[common_indices] < nonsurf_major.loc[common_indices])
                        total_count = len(common_indices)
                        
                        if total_count > 0:
                            surf_major_avg = surf_major.mean()
                            nonsurf_major_avg = nonsurf_major.mean()
                            
                            st.markdown(f"""
                            <div class="info-box">
                            <b>Major Tokens (100K):</b><br>
                            • SurfFuture has tighter spreads for {surf_better_count}/{total_count} major tokens<br>
                            • SurfFuture average: {format_number(surf_major_avg * scale_factor)} vs Non-Surf average: {format_number(nonsurf_major_avg * scale_factor)}
                            </div>
                            """, unsafe_allow_html=True)
            
            # For Altcoin tokens
            if not altcoin_tokens_df.empty:
                surf_altcoin = altcoin_tokens_df['SurfFuture'].dropna()
                nonsurf_altcoin = altcoin_tokens_df['Avg (Non-Surf)'].dropna()
                
                if not surf_altcoin.empty and not nonsurf_altcoin.empty:
                    common_indices = surf_altcoin.index.intersection(nonsurf_altcoin.index)
                    if len(common_indices) > 0:
                        surf_better_count = sum(surf_altcoin.loc[common_indices] < nonsurf_altcoin.loc[common_indices])
                        total_count = len(common_indices)
                        
                        if total_count > 0:
                            surf_altcoin_avg = surf_altcoin.mean()
                            nonsurf_altcoin_avg = nonsurf_altcoin.mean()
                            
                            st.markdown(f"""
                            <div class="info-box">
                            <b>Altcoin Tokens (50K):</b><br>
                            • SurfFuture has tighter spreads for {surf_better_count}/{total_count} altcoin tokens<br>
                            • SurfFuture average: {format_number(surf_altcoin_avg * scale_factor)} vs Non-Surf average: {format_number(nonsurf_altcoin_avg * scale_factor)}
                            </div>
                            """, unsafe_allow_html=True)
                            
            # Bar chart comparing exchanges
            st.markdown("### Average Spread by Exchange")
            
            # Calculate average for each exchange
            exchange_avgs = {}
            for exchange in ['Binance', 'Gate', 'Hyperliquid', 'SurfFuture']:
                if exchange in df.columns:
                    values = df[exchange].dropna()
                    if not values.empty:
                        exchange_avgs[exchange] = values.mean()
            
            if exchange_avgs:
                # Create data frame for plotting
                avg_df = pd.DataFrame({
                    'Exchange': list(exchange_avgs.keys()),
                    'Average Spread': list(exchange_avgs.values())
                })
                
                # Sort by average spread (ascending)
                avg_df = avg_df.sort_values('Average Spread')
                
                # Create bar chart
                fig = px.bar(
                    avg_df,
                    x='Exchange',
                    y='Average Spread',
                    title=f"Average Spread by Exchange ({scale_label})",
                    color='Exchange',
                    text='Average Spread',
                    color_discrete_map={
                        'SurfFuture': '#4CAF50',
                        'Binance': '#2196F3',
                        'Gate': '#FFC107',
                        'Hyperliquid': '#FF5722'
                    }
                )
                
                # Format the bars
                fig.update_traces(
                    texttemplate='%{y:.2f}',
                    textposition='outside'
                )
                
                # Format the layout
                fig.update_layout(
                    xaxis_title="Exchange",
                    yaxis_title=f"Average Spread {scale_label}",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No fee2 (100K/50K) data available for analysis")

# === TAB 3: SPREAD BY SIZE (ALL SIZES) ===
with tab3:
    st.markdown('<div class="header-style">Spread Analysis by Size</div>', unsafe_allow_html=True)
    
    # Explanation of sizes
    st.markdown("""
    <div class="info-box">
    <b>Trading Size Definitions:</b><br>
    • <b>Major tokens</b> (BTC, ETH, SOL, XRP, BNB): 50K, 100K, 200K, 500K<br>
    • <b>Altcoin tokens</b>: 20K, 50K, 100K, 200K<br>
    <br>
    Tables show daily averages of 10-minute spread data points across different size tiers.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"<div class='info-box'><b>Note:</b> All spread values are multiplied by {scale_factor} ({scale_label}) for better readability.</div>", unsafe_allow_html=True)
    
    # Check if we have all the necessary fee levels
    if all(key in matrix_data for key in ['fee1', 'fee2', 'fee3', 'fee4']):
        # Group tokens as majors and altcoins for better organization
        major_tokens = [t for t in selected_tokens if is_major(t)]
        altcoin_tokens = [t for t in selected_tokens if not is_major(t)]
        
        # --- Process Major Tokens ---
        st.markdown("### Major Tokens Analysis")
        
        # Filter for major tokens in each fee level
        major_size_data = {}
        for fee_key in ['fee1', 'fee2', 'fee3', 'fee4']:
            if fee_key in matrix_data:
                # Get data for this fee level
                df = matrix_data[fee_key].copy()
                # Filter for major tokens only
                major_df = df[df['pair_name'].apply(is_major)].copy()
                if not major_df.empty:
                    # Apply scaling
                    numeric_cols = [col for col in major_df.columns if col not in ['pair_name', 'Surf Better', 'Improvement %']]
                    for col in numeric_cols:
                        if col in major_df.columns:
                            major_df[col] = major_df[col] * scale_factor
                    major_size_data[fee_key] = major_df
        
        # Create comprehensive table to show major token data across all size tiers
        if major_size_data:
            # Prepare combined dataframe for display
            combined_major_df = pd.DataFrame()
            
            # Start with token names from fee1 (which should have all tokens)
            if 'fee1' in major_size_data:
                combined_major_df['Token'] = major_size_data['fee1']['pair_name']
                
                # Add columns for each exchange across all size tiers
                for exchange in ['Binance', 'Gate', 'Hyperliquid', 'SurfFuture', 'Avg (Non-Surf)']:
                    for fee_key in ['fee1', 'fee2', 'fee3', 'fee4']:
                        if fee_key in major_size_data and exchange in major_size_data[fee_key].columns:
                            size_label = fee_depth_map_major[fee_key]
                            col_name = f"{exchange} {size_label}"
                            # Merge on pair_name
                            combined_major_df = pd.merge(
                                combined_major_df,
                                major_size_data[fee_key][['pair_name', exchange]].rename(columns={exchange: col_name}),
                                left_on='Token',
                                right_on='pair_name',
                                how='left'
                            )
                            # Drop the extra pair_name column
                            if 'pair_name' in combined_major_df.columns:
                                combined_major_df = combined_major_df.drop(columns=['pair_name'])
            
            # Format and display the combined dataframe
            if not combined_major_df.empty:
                # Format numeric columns
                for col in combined_major_df.columns:
                    if col != 'Token' and not pd.api.types.is_categorical_dtype(combined_major_df[col]):
                        combined_major_df[col] = combined_major_df[col].apply(
                            lambda x: color_code_value(x) if not pd.isna(x) else ""
                        )
                
                # Display the table
                st.markdown("#### Major Tokens - All Size Tiers")
                html_table = combined_major_df.to_html(escape=False, index=False)
                st.markdown(html_table, unsafe_allow_html=True)
                
                # Create comparison charts for each exchange
                st.markdown("#### Exchange Comparison by Size (Major Tokens)")
                
                # Prepare data for comparison charts by exchange
                exchange_data = {}
                
                # For each exchange, collect data across all size tiers
                for exchange in ['Binance', 'Gate', 'Hyperliquid', 'SurfFuture']:
                    exchange_comparison = []
                    for fee_key in ['fee1', 'fee2', 'fee3', 'fee4']:
                        if fee_key in major_size_data and exchange in major_size_data[fee_key].columns:
                            df = major_size_data[fee_key]
                            size_label = fee_depth_map_major[fee_key]
                            avg_value = df[exchange].mean()
                            exchange_comparison.append({
                                'Size': size_label,
                                'Spread': avg_value
                            })
                    if exchange_comparison:
                        exchange_data[exchange] = pd.DataFrame(exchange_comparison)
                
                # Create a multi-line chart comparing all exchanges across size tiers
                all_exchange_data = []
                for exchange, data in exchange_data.items():
                    for _, row in data.iterrows():
                        all_exchange_data.append({
                            'Exchange': exchange,
                            'Size': row['Size'],
                            'Spread': row['Spread']
                        })
                
                if all_exchange_data:
                    all_exchange_df = pd.DataFrame(all_exchange_data)
                    
                    # Sort by size tiers
                    size_order = ['50K', '100K', '200K', '500K']
                    all_exchange_df['Size'] = pd.Categorical(
                        all_exchange_df['Size'],
                        categories=size_order,
                        ordered=True
                    )
                    all_exchange_df = all_exchange_df.sort_values('Size')
                    
                    # Create line chart
                    fig = px.line(
                        all_exchange_df,
                        x='Size',
                        y='Spread',
                        color='Exchange',
                        markers=True,
                        title=f"Spread Comparison by Size - Major Tokens {scale_label}",
                        color_discrete_map={
                            'SurfFuture': '#4CAF50',
                            'Binance': '#2196F3',
                            'Gate': '#FFC107',
                            'Hyperliquid': '#FF5722'
                        }
                    )
                    
                    # Format the layout
                    fig.update_layout(
                        xaxis_title="Size Tier",
                        yaxis_title=f"Average Spread {scale_label}",
                        height=500,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        ),
                        yaxis=dict(
                            tickformat='.2f'  # Show only 2 decimal places
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Create grouped bar chart for exchange comparison
                st.markdown("#### Exchange Spreads by Size Tier (Major Tokens)")
                
                # Prepare data for grouped bar chart
                comparison_data = []
                for fee_key in ['fee1', 'fee2', 'fee3', 'fee4']:
                    if fee_key in major_size_data:
                        df = major_size_data[fee_key]
                        size_label = fee_depth_map_major[fee_key]
                        
                        for exchange in ['Binance', 'Gate', 'Hyperliquid', 'SurfFuture']:
                            if exchange in df.columns:
                                values = df[exchange].dropna()
                                if not values.empty:
                                    avg_value = values.mean()
                                    comparison_data.append({
                                        'Exchange': exchange,
                                        'Size': size_label,
                                        'Spread': avg_value
                                    })
                
                if comparison_data:
                    comparison_df = pd.DataFrame(comparison_data)
                    
                    # Sort by size tiers
                    size_order = ['50K', '100K', '200K', '500K']
                    comparison_df['Size'] = pd.Categorical(
                        comparison_df['Size'],
                        categories=size_order,
                        ordered=True
                    )
                    comparison_df = comparison_df.sort_values('Size')
                    
                    # Create grouped bar chart
                    fig = px.bar(
                        comparison_df,
                        x='Size',
                        y='Spread',
                        color='Exchange',
                        barmode='group',
                        title=f"Exchange Comparison by Size - Major Tokens {scale_label}",
                        color_discrete_map={
                            'SurfFuture': '#4CAF50',
                            'Binance': '#2196F3',
                            'Gate': '#FFC107',
                            'Hyperliquid': '#FF5722'
                        }
                    )
                    
                    # Format the layout
                    fig.update_layout(
                        xaxis_title="Size Tier",
                        yaxis_title=f"Average Spread {scale_label}",
                        height=500,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        ),
                        yaxis=dict(
                            tickformat='.2f'  # Show only 2 decimal places
                        )
                    )
                    
                    # Add value labels on bars
                    fig.update_traces(
                        texttemplate='%{y:.2f}',
                        textposition='outside'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        # --- Process Altcoin Tokens ---
        st.markdown("### Altcoin Tokens Analysis")
        
        # Filter for altcoin tokens in each fee level
        altcoin_size_data = {}
        for fee_key in ['fee1', 'fee2', 'fee3', 'fee4']:
            if fee_key in matrix_data:
                # Get data for this fee level
                df = matrix_data[fee_key].copy()
                # Filter for altcoin tokens only
                altcoin_df = df[~df['pair_name'].apply(is_major)].copy()
                if not altcoin_df.empty:
                    # Apply scaling
                    numeric_cols = [col for col in altcoin_df.columns if col not in ['pair_name', 'Surf Better', 'Improvement %']]
                    for col in numeric_cols:
                        if col in altcoin_df.columns:
                            altcoin_df[col] = altcoin_df[col] * scale_factor
                    altcoin_size_data[fee_key] = altcoin_df
        
        # Create comprehensive table to show altcoin token data across all size tiers
        if altcoin_size_data:
            # Prepare combined dataframe for display
            combined_altcoin_df = pd.DataFrame()
            
            # Start with token names from fee1 (which should have all tokens)
            if 'fee1' in altcoin_size_data:
                combined_altcoin_df['Token'] = altcoin_size_data['fee1']['pair_name']
                
                # Add columns for each exchange across all size tiers
                for exchange in ['Binance', 'Gate', 'Hyperliquid', 'SurfFuture', 'Avg (Non-Surf)']:
                    for fee_key in ['fee1', 'fee2', 'fee3', 'fee4']:
                        if fee_key in altcoin_size_data and exchange in altcoin_size_data[fee_key].columns:
                            size_label = fee_depth_map_altcoin[fee_key]
                            col_name = f"{exchange} {size_label}"
                            # Merge on pair_name
                            combined_altcoin_df = pd.merge(
                                combined_altcoin_df,
                                altcoin_size_data[fee_key][['pair_name', exchange]].rename(columns={exchange: col_name}),
                                left_on='Token',
                                right_on='pair_name',
                                how='left'
                            )
                            # Drop the extra pair_name column
                            if 'pair_name' in combined_altcoin_df.columns:
                                combined_altcoin_df = combined_altcoin_df.drop(columns=['pair_name'])
            
            # Format and display the combined dataframe
            if not combined_altcoin_df.empty:
                # Format numeric columns
                for col in combined_altcoin_df.columns:
                    if col != 'Token' and not pd.api.types.is_categorical_dtype(combined_altcoin_df[col]):
                        combined_altcoin_df[col] = combined_altcoin_df[col].apply(
                            lambda x: color_code_value(x) if not pd.isna(x) else ""
                        )
                
                # Display the table
                st.markdown("#### Altcoin Tokens - All Size Tiers")
                html_table = combined_altcoin_df.to_html(escape=False, index=False)
                st.markdown(html_table, unsafe_allow_html=True)
                
                # Create comparison charts for each exchange
                st.markdown("#### Exchange Comparison by Size (Altcoin Tokens)")
                
                # Prepare data for comparison charts by exchange
                exchange_data = {}
                
                # For each exchange, collect data across all size tiers
                for exchange in ['Binance', 'Gate', 'Hyperliquid', 'SurfFuture']:
                    exchange_comparison = []
                    for fee_key in ['fee1', 'fee2', 'fee3', 'fee4']:
                        if fee_key in altcoin_size_data and exchange in altcoin_size_data[fee_key].columns:
                            df = altcoin_size_data[fee_key]
                            size_label = fee_depth_map_altcoin[fee_key]
                            avg_value = df[exchange].mean()
                            exchange_comparison.append({
                                'Size': size_label,
                                'Spread': avg_value
                            })
                    if exchange_comparison:
                        exchange_data[exchange] = pd.DataFrame(exchange_comparison)
                
                # Create a multi-line chart comparing all exchanges across size tiers
                all_exchange_data = []
                for exchange, data in exchange_data.items():
                    for _, row in data.iterrows():
                        all_exchange_data.append({
                            'Exchange': exchange,
                            'Size': row['Size'],
                            'Spread': row['Spread']
                        })
                
                if all_exchange_data:
                    all_exchange_df = pd.DataFrame(all_exchange_data)
                    
                    # Sort by size tiers
                    size_order = ['20K', '50K', '100K', '200K']
                    all_exchange_df['Size'] = pd.Categorical(
                        all_exchange_df['Size'],
                        categories=size_order,
                        ordered=True
                    )
                    all_exchange_df = all_exchange_df.sort_values('Size')
                    
                    # Create line chart
                    fig = px.line(
                        all_exchange_df,
                        x='Size',
                        y='Spread',
                        color='Exchange',
                        markers=True,
                        title=f"Spread Comparison by Size - Altcoin Tokens {scale_label}",
                        color_discrete_map={
                            'SurfFuture': '#4CAF50',
                            'Binance': '#2196F3',
                            'Gate': '#FFC107',
                            'Hyperliquid': '#FF5722'
                        }
                    )
                    
                    # Format the layout
                    fig.update_layout(
                        xaxis_title="Size Tier",
                        yaxis_title=f"Average Spread {scale_label}",
                        height=500,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        ),
                        yaxis=dict(
                            tickformat='.2f'  # Show only 2 decimal places
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Create grouped bar chart for exchange comparison
                st.markdown("#### Exchange Spreads by Size Tier (Altcoin Tokens)")
                
                # Prepare data for grouped bar chart
                comparison_data = []
                for fee_key in ['fee1', 'fee2', 'fee3', 'fee4']:
                    if fee_key in altcoin_size_data:
                        df = altcoin_size_data[fee_key]
                        size_label = fee_depth_map_altcoin[fee_key]
                        
                        for exchange in ['Binance', 'Gate', 'Hyperliquid', 'SurfFuture']:
                            if exchange in df.columns:
                                values = df[exchange].dropna()
                                if not values.empty:
                                    avg_value = values.mean()
                                    comparison_data.append({
                                        'Exchange': exchange,
                                        'Size': size_label,
                                        'Spread': avg_value
                                    })
                
                if comparison_data:
                    comparison_df = pd.DataFrame(comparison_data)
                    
                    # Sort by size tiers
                    size_order = ['20K', '50K', '100K', '200K']
                    comparison_df['Size'] = pd.Categorical(
                        comparison_df['Size'],
                        categories=size_order,
                        ordered=True
                    )
                    comparison_df = comparison_df.sort_values('Size')
                    
                    # Create grouped bar chart
                    fig = px.bar(
                        comparison_df,
                        x='Size',
                        y='Spread',
                        color='Exchange',
                        barmode='group',
                        title=f"Exchange Comparison by Size - Altcoin Tokens {scale_label}",
                        color_discrete_map={
                            'SurfFuture': '#4CAF50',
                            'Binance': '#2196F3',
                            'Gate': '#FFC107',
                            'Hyperliquid': '#FF5722'
                        }
                    )
                    
                    # Format the layout
                    fig.update_layout(
                        xaxis_title="Size Tier",
                        yaxis_title=f"Average Spread {scale_label}",
                        height=500,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        ),
                        yaxis=dict(
                            tickformat='.2f'  # Show only 2 decimal places
                        )
                    )
                    
                    # Add value labels on bars
                    fig.update_traces(
                        texttemplate='%{y:.2f}',
                        textposition='outside'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Missing fee data for one or more size tiers. Cannot display complete analysis.")

# Footer with explanatory information
with st.expander("Understanding Exchange Spreads"):
    st.markdown("""
    ### About This Dashboard
    
    This dashboard provides comprehensive analysis of trading spreads across multiple cryptocurrency exchanges. 
    
    ### Key Concepts:
    
    - **Spread**: The difference between the buy and sell price, representing the cost of trading.
    
    - **Trading Sizes**: Different order sizes for analysis:
      - **Major tokens** (BTC, ETH, SOL, XRP, BNB): 50K, 100K, 200K, 500K
      - **Altcoin tokens**: 20K, 50K, 100K, 200K
    
    - **Fee Columns**: 
      - `fee1`, `fee2`, `fee3`, `fee4` correspond to spreads at different trading sizes
    
    - **Exchange Comparison**: The dashboard highlights when SurfFuture has tighter spreads (lower fees) than the average of other exchanges.
    
     ### Interpreting the Data:
    
    - Lower spread values indicate better pricing for traders
    - Bold formatting highlights spread values for better readability
    - All values are multiplied by 10,000 for better readability
    - The "Better" percentage shows how much lower SurfFuture's spread is compared to other exchanges (positive values mean SurfFuture is better)
    
    ### Data Source:
    
    Data is fetched from the `oracle_exchange_fee` table, with 10-minute interval data points that represent the average of previous 10 one-minute points.
    """)

# Execute the app
if __name__ == '__main__':
    pass  # The app is already running