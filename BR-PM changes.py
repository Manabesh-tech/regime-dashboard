import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import pytz
import traceback
import json
import math

# Page configuration
st.set_page_config(
    page_title="Surf vs Rollbit Parameters",
    page_icon="📊",
    layout="wide"
)

# Apply custom CSS styling
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
    .error-message {
        color: #d32f2f;
        background-color: #ffebee;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        border-left: 4px solid #d32f2f;
    }
    .success-message {
        color: #2e7d32;
        background-color: #e8f5e9;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        border-left: 4px solid #2e7d32;
    }
    .confirm-button {
        background-color: #f44336;
        color: white;
        font-weight: bold;
    }
    .action-button {
        background-color: #1976D2;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- Database Configuration ---
@st.cache_resource
def init_connection():
    try:
        # Try to get database config from secrets
        db_config = st.secrets["database"]
        db_uri = (
            f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}"
            f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )
        engine = create_engine(
            db_uri,
            pool_size=5,  # 连接池大小
            max_overflow=10,  # 最大溢出连接数
            pool_timeout=30,  # 连接超时时间
            pool_recycle=1800,  # 连接回收时间(30分钟)
            pool_pre_ping=True,  # 使用连接前先测试连接是否有效
            pool_use_lifo=True,  # 使用后进先出,减少空闲连接
            isolation_level="AUTOCOMMIT",  # 设置自动提交模式
            echo=False  # 不打印 SQL 语句
        )
        return engine
    except Exception as e:
        st.sidebar.error(f"Error connecting to the database: {e}")
        return None

# --- Session State Management ---
def init_session_state():
    """Initialize session state variables"""
    if 'backup_params' not in st.session_state:
        st.session_state.backup_params = None
    if 'has_applied_recommendations' not in st.session_state:
        st.session_state.has_applied_recommendations = False
    if 'show_confirm_dialog' not in st.session_state:
        st.session_state.show_confirm_dialog = False

# --- Utility Functions ---
def format_percent(value):
    """Format a value as a percentage with 2 decimal places"""
    if pd.isna(value) or value is None or value == 0:
        return "N/A"
    return f"{value * 100:.2f}%"

def format_number(value):
    """Format a number with comma separation"""
    if pd.isna(value) or value is None:
        return "N/A"
    return f"{value:,.0f}"

def is_major(token):
    """Determine if a token is a major token"""
    majors = ["BTC", "ETH", "SOL", "XRP", "BNB"]
    for major in majors:
        if major in token:
            return True
    return False

def safe_division(a, b, default=0.0):
    """Safely divide two numbers, handling zeros and None values"""
    if a is None or b is None or pd.isna(a) or pd.isna(b) or b == 0:
        return default
    return a / b

def check_null_or_zero(value):
    """Check if a value is NULL, None, NaN, or zero"""
    if value is None or pd.isna(value) or value == 0:
        return True
    return False

# --- Create Weekly Stats Table ---
def create_weekly_stats_table():
    """Create the spread_weekly_stats table if it doesn't exist"""
    try:
        engine = init_connection()
        if not engine:
            return False
        
        query = """
        CREATE TABLE IF NOT EXISTS spread_weekly_stats (
            pair_name VARCHAR(50) PRIMARY KEY,
            min_spread NUMERIC,
            max_spread NUMERIC,
            std_dev NUMERIC,
            updated_at TIMESTAMP
        );
        """
        
        with engine.connect() as conn:
            conn.execute(text(query))
        
        return True
    except Exception as e:
        st.error(f"Error creating weekly stats table: {e}")
        return False

# --- Data Fetching Functions ---
def fetch_current_parameters():
    try:
        engine = init_connection()
        if not engine:
            return None
        query = """
        SELECT
            pair_name,
            buffer_rate,
            position_multiplier,
            max_leverage
        FROM
            public.trade_pool_pairs
        WHERE
            status = 1
        ORDER BY
            pair_name
        """
        df = pd.read_sql(query, engine)
        if not df.empty:
            df['max_leverage'] = df['max_leverage'].fillna(100)
        return df if not df.empty else None
    except Exception as e:
        st.error(f"Error fetching current parameters: {e}")
        return None

@st.cache_data(ttl=600)
def fetch_rollbit_parameters():
    """Fetch Rollbit parameters for comparison"""
    try:
        engine = init_connection()
        if not engine:
            return None
            
        query = """
        SELECT * 
        FROM rollbit_pair_config 
        WHERE created_at = (SELECT max(created_at) FROM rollbit_pair_config)
        """
        
        df = pd.read_sql(query, engine)
        
        # Ensure we have the required columns and rename if needed
        if not df.empty:
            # Ensure we have bust_buffer to use as buffer_rate
            if 'bust_buffer' in df.columns and 'buffer_rate' not in df.columns:
                df['buffer_rate'] = df['bust_buffer']
            
        return df if not df.empty else None
    except Exception as e:
        st.error(f"Error fetching Rollbit parameters: {e}")
        return None

@st.cache_data(ttl=600)
def fetch_market_spread_data():
    """Fetch current market spread data for all tokens"""
    try:
        engine = init_connection()
        if not engine:
            return None
            
        # Get current time in Singapore timezone
        singapore_timezone = pytz.timezone('Asia/Singapore')
        now_utc = datetime.now(pytz.utc)
        now_sg = now_utc.astimezone(singapore_timezone)
        start_time_sg = now_sg - timedelta(days=1)
        
        # Convert back to UTC for database query
        start_time_utc = start_time_sg.astimezone(pytz.utc)
        end_time_utc = now_sg.astimezone(pytz.utc)
        
        query = f"""
        SELECT 
            pair_name,
            source,
            AVG(fee1) as avg_fee1
        FROM 
            oracle_exchange_fee
        WHERE 
            source IN ('binanceFuture', 'gateFuture', 'hyperliquidFuture')
            AND time_group BETWEEN '{start_time_utc}' AND '{end_time_utc}'
        GROUP BY 
            pair_name, source
        ORDER BY 
            pair_name, source
        """
        
        df = pd.read_sql(query, engine)
        return df if not df.empty else None
    except Exception as e:
        st.error(f"Error fetching market spread data: {e}")
        return None

@st.cache_data(ttl=600)
def fetch_spread_baselines():
    """Fetch spread baselines for comparison"""
    try:
        engine = init_connection()
        if not engine:
            return None
            
        query = """
        SELECT 
            pair_name,
            baseline_spread,
            updated_at
        FROM 
            spread_baselines
        ORDER BY 
            pair_name
        """
        
        df = pd.read_sql(query, engine)
        return df if not df.empty else None
    except Exception as e:
        st.error(f"Error fetching spread baselines: {e}")
        return None

@st.cache_data(ttl=600)
def fetch_weekly_spread_stats():
    """Fetch weekly spread statistics for the past 7 days"""
    try:
        engine = init_connection()
        if not engine:
            return None
            
        # Get current time in Singapore timezone and calculate 7 days ago
        singapore_timezone = pytz.timezone('Asia/Singapore')
        now_utc = datetime.now(pytz.utc)
        now_sg = now_utc.astimezone(singapore_timezone)
        start_time_sg = now_sg - timedelta(days=7)
        
        # Convert back to UTC for database query
        start_time_utc = start_time_sg.astimezone(pytz.utc)
        end_time_utc = now_sg.astimezone(pytz.utc)
        
        query = f"""
        SELECT 
            pair_name,
            MAX(fee1) as max_spread,
            MIN(fee1) as min_spread
        FROM 
            oracle_exchange_fee
        WHERE 
            source IN ('binanceFuture', 'gateFuture', 'hyperliquidFuture')
            AND time_group BETWEEN '{start_time_utc}' AND '{end_time_utc}'
        GROUP BY 
            pair_name
        ORDER BY 
            pair_name
        """
        
        df = pd.read_sql(query, engine)
        return df if not df.empty else None
    except Exception as e:
        st.error(f"Error fetching weekly spread stats: {e}")
        return None

def update_weekly_spread_stats(market_data_df):
    """Update weekly spread statistics based on current market data"""
    if market_data_df is None or market_data_df.empty:
        return False, "No market data available"
    
    try:
        engine = init_connection()
        if not engine:
            return False, "Database connection error"
        
        # Calculate current spreads by pair
        current_spreads = calculate_current_spreads(market_data_df)
        
        # Fetch existing stats
        existing_stats_df = fetch_spread_weekly_stats()
        existing_stats = {}
        
        if existing_stats_df is not None and not existing_stats_df.empty:
            for _, row in existing_stats_df.iterrows():
                existing_stats[row['pair_name']] = {
                    'min_spread': row['min_spread'],
                    'max_spread': row['max_spread'],
                    'std_dev': row['std_dev']
                }
        
        # Update statistics for each pair
        success_count = 0
        error_count = 0
        
        for pair_name, current_spread in current_spreads.items():
            try:
                # Get existing stats or initialize new ones
                if pair_name in existing_stats:
                    stats = existing_stats[pair_name]
                    
                    # Update min and max
                    min_spread = min(stats['min_spread'], current_spread)
                    max_spread = max(stats['max_spread'], current_spread)
                    
                    # Estimate std_dev as 1/4 of the range (normal distribution approximation)
                    std_dev = (max_spread - min_spread) / 4.0
                    if std_dev <= 0:
                        std_dev = current_spread * 0.05  # Fallback: 5% of current spread
                else:
                    # Initialize new stats with reasonable range around current value
                    min_spread = current_spread * 0.9  # 10% below current
                    max_spread = current_spread * 1.1  # 10% above current
                    std_dev = current_spread * 0.05     # 5% of current spread
                
                # Upsert the stats
                query = text("""
                INSERT INTO spread_weekly_stats 
                    (pair_name, min_spread, max_spread, std_dev, updated_at)
                VALUES 
                    (:pair_name, :min_spread, :max_spread, :std_dev, :updated_at)
                ON CONFLICT (pair_name) DO UPDATE 
                SET 
                    min_spread = LEAST(spread_weekly_stats.min_spread, :min_spread),
                    max_spread = GREATEST(spread_weekly_stats.max_spread, :max_spread),
                    std_dev = :std_dev,
                    updated_at = :updated_at
                """)
                
                with engine.connect() as conn:
                    conn.execute(
                        query, 
                        {
                            "pair_name": pair_name,
                            "min_spread": min_spread,
                            "max_spread": max_spread,
                            "std_dev": std_dev,
                            "updated_at": datetime.now()
                        }
                    )
                
                success_count += 1
            except Exception as e:
                error_count += 1
        
        return success_count > 0, f"Updated {success_count} pairs with {error_count} errors"
    except Exception as e:
        return False, f"Error updating weekly stats: {str(e)}"

def save_spread_baseline(pair_name, baseline_spread):
    """Save a new spread baseline to the database"""
    try:
        engine = init_connection()
        if not engine:
            return False
        
        # 确保 baseline_spread 是 Python 原生 float 类型
        if isinstance(baseline_spread, (np.float32, np.float64)):
            baseline_spread = float(baseline_spread)
        
        # Use SQLAlchemy text() for parameterized queries
        query = text("""
        INSERT INTO spread_baselines (pair_name, baseline_spread, updated_at)
        VALUES (:pair_name, :baseline_spread, :updated_at)
        ON CONFLICT (pair_name) DO UPDATE 
        SET baseline_spread = EXCLUDED.baseline_spread, 
            updated_at = EXCLUDED.updated_at
        """)
        
        # Execute with parameters
        with engine.connect() as conn:
            conn.execute(
                query, 
                {"pair_name": pair_name, "baseline_spread": baseline_spread, "updated_at": datetime.now()}
            )
            
        return True
    except Exception as e:
        error_details = traceback.format_exc()
        st.error(f"Error saving baseline spread for {pair_name}: {e}\n\nDetails: {error_details}")
        return False

def calculate_current_spreads(market_data):
    """Calculate current average non-SurfFuture spread for each token"""
    if market_data is None or market_data.empty:
        return {}
    
    # Group by pair_name and calculate average spread across all exchanges
    current_spreads = {}
    for pair, group in market_data.groupby('pair_name'):
        current_spreads[pair] = group['avg_fee1'].mean()
    
    return current_spreads

def reset_all_baselines(market_data_df):
    """Reset all baselines to current market spreads"""
    if market_data_df is None or market_data_df.empty:
        return False, 0, 0
    
    # Calculate current spreads
    current_spreads = calculate_current_spreads(market_data_df)
    
    success_count = 0
    error_count = 0
    
    # Update each baseline in the database
    for pair, spread in current_spreads.items():
        if save_spread_baseline(pair, spread):
            success_count += 1
        else:
            error_count += 1
    
    return success_count > 0, success_count, error_count

def render_complete_parameter_table(params_df, market_data_df, baselines_df, weekly_stats_df, sort_by="Pair Name"):
    """Render the complete parameter table with all pairs"""
    
    if params_df is None or params_df.empty:
        st.warning("No parameter data available.")
        return
    
    # Map sort option to column name
    sort_map = {
        "Pair Name": "pair_name",
        "Spread Change": "spread_change_pct"
    }
    
    # Add spread data to params_df
    data = []
    current_spreads = calculate_current_spreads(market_data_df) if market_data_df is not None else {}
    baselines = {}
    if baselines_df is not None and not baselines_df.empty:
        for _, row in baselines_df.iterrows():
            baselines[row['pair_name']] = row['baseline_spread']
    
    # Create weekly stats dict
    weekly_stats = {}
    if weekly_stats_df is not None and not weekly_stats_df.empty:
        for _, row in weekly_stats_df.iterrows():
            weekly_stats[row['pair_name']] = {
                'min_spread': row['min_spread'],
                'max_spread': row['max_spread']
            }
    
    for _, row in params_df.iterrows():
        pair_name = row['pair_name']
        current_spread = current_spreads.get(pair_name, None)
        baseline_spread = baselines.get(pair_name, None)
        
        # Get weekly stats
        weekly_min = None
        weekly_max = None
        if pair_name in weekly_stats:
            weekly_min = weekly_stats[pair_name]['min_spread']
            weekly_max = weekly_stats[pair_name]['max_spread']
            
        # Calculate spread change percentage
        spread_change_pct = None
        if current_spread is not None and baseline_spread is not None and baseline_spread > 0:
            spread_change_pct = ((current_spread / baseline_spread) - 1) * 100
            
        data.append({
            'pair_name': pair_name,
            'token_type': 'Major' if is_major(pair_name) else 'Altcoin',
            'buffer_rate': row['buffer_rate'],
            'position_multiplier': row['position_multiplier'],
            'current_spread': current_spread,
            'baseline_spread': baseline_spread,
            'weekly_min': weekly_min,
            'weekly_max': weekly_max,
            'spread_change_pct': spread_change_pct,
            'max_leverage': row['max_leverage']
        })
    
    df = pd.DataFrame(data)
    
    # Sort the DataFrame based on sort option
    sort_column = sort_map.get(sort_by, "pair_name")
    if sort_column == "spread_change_pct":
        sorted_df = df.sort_values(sort_column, ascending=False)
    else:
        sorted_df = df.sort_values(sort_column)
    
    # Highlight significant changes
    def highlight_changes(val):
        """Highlight significant changes in the parameters"""
        if isinstance(val, str) and "%" in val:
            try:
                num_val = float(val.strip('%').replace('+', '').replace('-', ''))
                if num_val > 5.0:
                    return 'background-color: #ffcccc'  # Red for significant changes
                elif num_val > 2.0:
                    return 'background-color: #ffffcc'  # Yellow for moderate changes
            except:
                pass
        return ''
    
    # Create a formatted DataFrame for display
    display_df = pd.DataFrame({
        'Pair': sorted_df['pair_name'],
        'Type': sorted_df['token_type'],
        'Current Spread': sorted_df['current_spread'].apply(lambda x: f"{x*10000:.2f}" if not pd.isna(x) else "N/A"),
        'Baseline Spread': sorted_df['baseline_spread'].apply(lambda x: f"{x*10000:.2f}" if not pd.isna(x) else "N/A"),
        'Weekly High': sorted_df['weekly_max'].apply(lambda x: f"{x*10000:.2f}" if not pd.isna(x) else "N/A"),
        'Weekly Low': sorted_df['weekly_min'].apply(lambda x: f"{x*10000:.2f}" if not pd.isna(x) else "N/A"),
        'Spread Change': sorted_df['spread_change_pct'].apply(
            lambda x: f"{x:+.2f}%" if not pd.isna(x) else "N/A"
        ),
        'Buffer Rate': sorted_df['buffer_rate'].apply(
            lambda x: f"{x*100:.3f}%" if not pd.isna(x) else "N/A"
        ),
        'Position Multiplier': sorted_df['position_multiplier'].apply(
            lambda x: f"{x:,.0f}" if not pd.isna(x) else "N/A"
        ),
        'Max Leverage': sorted_df['max_leverage'].apply(
            lambda x: f"{x:.0f}x" if not pd.isna(x) else "N/A"
        )
    })
    
    # Style the dataframe with highlighting
    styled_df = display_df.style.map(highlight_changes, subset=['Spread Change'])
    
    # Display with highlighting
    st.dataframe(styled_df, use_container_width=True)
    
    # Add a color legend below the table
    st.markdown("""
    <div style="margin-top: 10px; font-size: 0.8em;">
        <span style="background-color: #ffcccc; padding: 3px 8px;">Red</span>: Major spread change (> 5%)
        <span style="margin-left: 15px; background-color: #ffffcc; padding: 3px 8px;">Yellow</span>: Moderate spread change (> 2%)
    </div>
    """, unsafe_allow_html=True)

def render_rollbit_comparison(params_df, rollbit_df):
    """Render the Rollbit comparison tab"""
    if params_df is None or rollbit_df is None or params_df.empty or rollbit_df.empty:
        st.info("No data available for Rollbit comparison.")
        return
    
    # Merge the dataframes on pair_name
    merged_df = pd.merge(
        params_df[['pair_name', 'buffer_rate', 'position_multiplier']], 
        rollbit_df[['pair_name', 'buffer_rate', 'position_multiplier']], 
        on='pair_name', 
        how='inner',
        suffixes=('', '_rollbit')
    )
    
    if merged_df.empty:
        st.info("No matching pairs found for Rollbit comparison.")
        return
    
    # Buffer Rate Comparison
    st.markdown("### Buffer Rate Comparison")
    
    # Create buffer rate comparison table
    buffer_df = pd.DataFrame({
        'Pair': merged_df['pair_name'],
        'Type': merged_df['pair_name'].apply(lambda x: 'Major' if is_major(x) else 'Altcoin'),
        'SURF Buffer': merged_df['buffer_rate'].apply(
            lambda x: f"{x*100:.3f}%" if not pd.isna(x) else "N/A"
        ),
        'Rollbit Buffer': merged_df['buffer_rate_rollbit'].apply(
            lambda x: f"{x*100:.3f}%" if not pd.isna(x) else "N/A"
        )
    })
    
    # Add buffer ratio column
    buffer_ratio = []
    for _, row in merged_df.iterrows():
        if (not check_null_or_zero(row['buffer_rate']) and 
            not check_null_or_zero(row['buffer_rate_rollbit'])):
            ratio = safe_division(row['buffer_rate'], row['buffer_rate_rollbit'], None)
            buffer_ratio.append(f"{ratio:.2f}x" if ratio is not None else "N/A")
        else:
            buffer_ratio.append("N/A")
    
    buffer_df['Buffer Ratio'] = buffer_ratio
    
    # Display buffer rate comparison
    st.dataframe(buffer_df, use_container_width=True)
    
    # Position Multiplier Comparison
    st.markdown("### Position Multiplier Comparison")
    
    # Create position multiplier comparison table
    position_df = pd.DataFrame({
        'Pair': merged_df['pair_name'],
        'Type': merged_df['pair_name'].apply(lambda x: 'Major' if is_major(x) else 'Altcoin'),
        'SURF Position Mult.': merged_df['position_multiplier'].apply(
            lambda x: f"{x:,.0f}" if not pd.isna(x) else "N/A"
        ),
        'Rollbit Position Mult.': merged_df['position_multiplier_rollbit'].apply(
            lambda x: f"{x:,.0f}" if not pd.isna(x) else "N/A"
        )
    })
    
    # Add position ratio column
    position_ratio = []
    for _, row in merged_df.iterrows():
        if (not check_null_or_zero(row['position_multiplier']) and 
            not check_null_or_zero(row['position_multiplier_rollbit'])):
            ratio = safe_division(row['position_multiplier'], row['position_multiplier_rollbit'], None)
            position_ratio.append(f"{ratio:.2f}x" if ratio is not None else "N/A")
        else:
            position_ratio.append("N/A")
    
    position_df['Position Ratio'] = position_ratio
    
    # Display position multiplier comparison
    st.dataframe(position_df, use_container_width=True)
    
    # Add explanation
    st.markdown("""
    ### Understanding the Comparison
    
    This tab compares SURF's current parameters with Rollbit's parameters for matching tokens:
    
    - **Buffer Ratio**: SURF buffer rate ÷ Rollbit buffer rate. Values > 1 mean SURF is more conservative.
    - **Position Ratio**: SURF position multiplier ÷ Rollbit position multiplier. Values > 1 mean SURF allows larger positions.
    
    *Note: "N/A" is displayed when either SURF or Rollbit has null, zero, or missing values for comparison.*
    """)

# --- Main Application ---
def main():
    # Initialize session state
    init_session_state()
    
    st.markdown('<div class="header-style">Surf vs Rollbit Parameters</div>', unsafe_allow_html=True)

    # Sidebar controls
    st.sidebar.header("Controls")

    # Add a refresh button
    if st.sidebar.button("Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    # Add a reset baselines button
    if st.sidebar.button("Reset Baselines to Current Spreads", use_container_width=True):
        market_data_df = fetch_market_spread_data()
        if market_data_df is not None and not market_data_df.empty:
            success, count, errors = reset_all_baselines(market_data_df)
            if success:
                st.sidebar.success(f"Successfully reset {count} baselines")
                # Clear cache to refresh data
                st.cache_data.clear()
            else:
                st.sidebar.error(f"Failed to reset baselines. {errors} errors occurred.")
        else:
            st.sidebar.error("No market data available to reset baselines")

    # Create simplified tab navigation
    tabs = st.tabs(["Parameter Table", "Rollbit Comparison"])
    
    # Fetch data
    current_params_df = fetch_current_parameters()
    market_data_df = fetch_market_spread_data()
    baselines_df = fetch_spread_baselines()
    rollbit_df = fetch_rollbit_parameters()
    weekly_stats_df = fetch_weekly_spread_stats()

    # Process the data and render tabs
    if current_params_df is not None:
        # Render the appropriate tab content
        with tabs[0]:  # Parameter Table
            # Add sort options
            sort_by = st.selectbox(
                "Sort by:",
                options=["Pair Name", "Spread Change"],
                index=0
            )
            
            # Show parameter table
            st.markdown("### Parameter Table")
            render_complete_parameter_table(current_params_df, market_data_df, baselines_df, weekly_stats_df, sort_by)
            
        with tabs[1]:  # Rollbit Comparison
            render_rollbit_comparison(current_params_df, rollbit_df)
            
    else:
        st.error("Failed to load required data. Please check database connection and try refreshing.")

if __name__ == "__main__":
    main()
