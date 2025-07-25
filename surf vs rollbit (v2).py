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
    .mapping-info {
        background-color: #e3f2fd;
        padding: 8px;
        border-radius: 4px;
        margin: 5px 0;
        border-left: 3px solid #2196f3;
        font-size: 0.9em;
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

# --- Pair Mapping Configuration ---
def get_pair_mappings():
    """Define mappings between Surf and Rollbit pair names"""
    return {
        # Surf pair name -> Rollbit pair name
        "PUMP/USDT": "1000PUMP/USDT",
        # Add more mappings here as needed
        # "SURF_PAIR": "ROLLBIT_PAIR",
    }

def normalize_pair_for_comparison(surf_pair, rollbit_pairs, mappings):
    """
    Find the corresponding Rollbit pair for a given Surf pair
    Returns the Rollbit pair name if found, None otherwise
    """
    # Direct match
    if surf_pair in rollbit_pairs:
        return surf_pair
    
    # Check mappings
    if surf_pair in mappings:
        mapped_pair = mappings[surf_pair]
        if mapped_pair in rollbit_pairs:
            return mapped_pair
    
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
            max_leverage,
            rate_multiplier,
            rate_exponent
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
            df['rate_multiplier'] = df['rate_multiplier'].fillna(1.0)
            df['rate_exponent'] = df['rate_exponent'].fillna(1.0)
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
            
            # Fill missing values with defaults
            if 'rate_multiplier' not in df.columns:
                df['rate_multiplier'] = 1.0
            else:
                df['rate_multiplier'] = df['rate_multiplier'].fillna(1.0)
                
            if 'rate_exponent' not in df.columns:
                df['rate_exponent'] = 1.0
            else:
                df['rate_exponent'] = df['rate_exponent'].fillna(1.0)
            
        return df if not df.empty else None
    except Exception as e:
        st.error(f"Error fetching Rollbit parameters: {e}")
        return None

# Removed spread data fetching functions as they are no longer needed

# Removed spread-related functions as they are no longer needed

def render_complete_parameter_table(params_df, sort_by="Pair Name"):
    """Render the complete parameter table with all pairs"""
    
    if params_df is None or params_df.empty:
        st.warning("No parameter data available.")
        return
    
    # Map sort option to column name
    sort_map = {
        "Pair Name": "pair_name",
        "Token Type": "token_type"
    }
    
    # Prepare data for display
    data = []
    for _, row in params_df.iterrows():
        data.append({
            'pair_name': row['pair_name'],
            'token_type': 'Major' if is_major(row['pair_name']) else 'Altcoin',
            'buffer_rate': row['buffer_rate'],
            'position_multiplier': row['position_multiplier'],
            'rate_multiplier': row['rate_multiplier'],
            'rate_exponent': row['rate_exponent'],
            'max_leverage': row['max_leverage']
        })
    
    df = pd.DataFrame(data)
    
    # Sort the DataFrame based on sort option
    sort_column = sort_map.get(sort_by, "pair_name")
    sorted_df = df.sort_values(sort_column)
    
    # Create a formatted DataFrame for display
    display_df = pd.DataFrame({
        'Pair': sorted_df['pair_name'],
        'Type': sorted_df['token_type'],
        'Buffer Rate': sorted_df['buffer_rate'].apply(
            lambda x: f"{x*100:.3f}%" if not pd.isna(x) else "N/A"
        ),
        'Position Multiplier': sorted_df['position_multiplier'].apply(
            lambda x: f"{x:,.0f}" if not pd.isna(x) else "N/A"
        ),
        'Rate Multiplier': sorted_df['rate_multiplier'].apply(
            lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A"
        ),
        'Rate Exponent': sorted_df['rate_exponent'].apply(
            lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A"
        ),
        'Max Leverage': sorted_df['max_leverage'].apply(
            lambda x: f"{x:.0f}x" if not pd.isna(x) else "N/A"
        )
    })
    
    # Display the dataframe
    st.dataframe(display_df, use_container_width=True)

def render_rollbit_comparison(params_df, rollbit_df):
    """Render the Rollbit comparison tab with pair mapping support"""
    if params_df is None or rollbit_df is None or params_df.empty or rollbit_df.empty:
        st.info("No data available for Rollbit comparison.")
        return
    
    # Get pair mappings
    pair_mappings = get_pair_mappings()
    
    # Get list of available Rollbit pairs
    rollbit_pairs = set(rollbit_df['pair_name'].tolist())
    
    # Create mapping for comparison
    comparison_data = []
    mapped_pairs = []
    
    for _, surf_row in params_df.iterrows():
        surf_pair = surf_row['pair_name']
        
        # Find corresponding Rollbit pair
        rollbit_pair = normalize_pair_for_comparison(surf_pair, rollbit_pairs, pair_mappings)
        
        if rollbit_pair:
            # Get Rollbit data for this pair
            rollbit_row = rollbit_df[rollbit_df['pair_name'] == rollbit_pair].iloc[0]
            
            comparison_data.append({
                'surf_pair': surf_pair,
                'rollbit_pair': rollbit_pair,
                'is_mapped': rollbit_pair != surf_pair,
                'buffer_rate': surf_row['buffer_rate'],
                'buffer_rate_rollbit': rollbit_row['buffer_rate'],
                'position_multiplier': surf_row['position_multiplier'],
                'position_multiplier_rollbit': rollbit_row['position_multiplier'],
                'rate_multiplier': surf_row['rate_multiplier'],
                'rate_multiplier_rollbit': rollbit_row['rate_multiplier'],
                'rate_exponent': surf_row['rate_exponent'],
                'rate_exponent_rollbit': rollbit_row['rate_exponent']
            })
            
            if rollbit_pair != surf_pair:
                mapped_pairs.append((surf_pair, rollbit_pair))
    
    if not comparison_data:
        st.info("No matching pairs found for Rollbit comparison.")
        return
    
    # Display mapping information if there are any mapped pairs
    if mapped_pairs:
        st.markdown("### Pair Mappings")
        st.markdown('<div class="mapping-info">', unsafe_allow_html=True)
        st.markdown("**The following pairs are mapped for comparison:**")
        for surf_pair, rollbit_pair in mapped_pairs:
            st.markdown(f"• **{surf_pair}** (Surf) ↔ **{rollbit_pair}** (Rollbit)")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("---")
    
    merged_df = pd.DataFrame(comparison_data)
    
    # Create tabs for different parameter comparisons
    comp_tabs = st.tabs(["Buffer Rate", "Position Multiplier", "Rate Multiplier", "Rate Exponent"])
    
    with comp_tabs[0]:  # Buffer Rate Comparison
        st.markdown("### Buffer Rate Comparison")
        
        # Create buffer rate comparison table
        buffer_df = pd.DataFrame({
            'SURF Pair': merged_df['surf_pair'],
            'Rollbit Pair': merged_df['rollbit_pair'],
            'Type': merged_df['surf_pair'].apply(lambda x: 'Major' if is_major(x) else 'Altcoin'),
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
        
        buffer_df['Buffer Ratio (SURF/Rollbit)'] = buffer_ratio
        
        # Highlight mapped pairs
        def highlight_mapped_pairs(row):
            is_mapped = merged_df.iloc[row.name]['is_mapped']
            if is_mapped:
                return ['background-color: #e8f5e9'] * len(row)
            return [''] * len(row)
        
        # Display buffer rate comparison
        styled_buffer_df = buffer_df.style.apply(highlight_mapped_pairs, axis=1)
        st.dataframe(styled_buffer_df, use_container_width=True)
        
        st.markdown("""
        **Buffer Ratio Interpretation:**
        - Values > 1: SURF is more conservative (higher buffer rate)
        - Values < 1: SURF is more aggressive (lower buffer rate)
        - Values = 1: Both platforms have similar buffer rates
        
        <span style="background-color: #e8f5e9; padding: 2px 6px;">Green highlighting</span> indicates mapped pairs with different names.
        """, unsafe_allow_html=True)
    
    with comp_tabs[1]:  # Position Multiplier Comparison
        st.markdown("### Position Multiplier Comparison")
        
        # Create position multiplier comparison table
        position_df = pd.DataFrame({
            'SURF Pair': merged_df['surf_pair'],
            'Rollbit Pair': merged_df['rollbit_pair'],
            'Type': merged_df['surf_pair'].apply(lambda x: 'Major' if is_major(x) else 'Altcoin'),
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
        
        position_df['Position Ratio (SURF/Rollbit)'] = position_ratio
        
        # Display position multiplier comparison with highlighting
        styled_position_df = position_df.style.apply(highlight_mapped_pairs, axis=1)
        st.dataframe(styled_position_df, use_container_width=True)
        
        st.markdown("""
        **Position Ratio Interpretation:**
        - Values > 1: SURF allows larger positions relative to Rollbit
        - Values < 1: SURF allows smaller positions relative to Rollbit  
        - Values = 1: Both platforms have similar position multipliers
        
        <span style="background-color: #e8f5e9; padding: 2px 6px;">Green highlighting</span> indicates mapped pairs with different names.
        """, unsafe_allow_html=True)
    
    with comp_tabs[2]:  # Rate Multiplier Comparison
        st.markdown("### Rate Multiplier Comparison")
        
        # Create rate multiplier comparison table
        rate_mult_df = pd.DataFrame({
            'SURF Pair': merged_df['surf_pair'],
            'Rollbit Pair': merged_df['rollbit_pair'],
            'Type': merged_df['surf_pair'].apply(lambda x: 'Major' if is_major(x) else 'Altcoin'),
            'SURF Rate Mult.': merged_df['rate_multiplier'].apply(
                lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A"
            ),
            'Rollbit Rate Mult.': merged_df['rate_multiplier_rollbit'].apply(
                lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A"
            )
        })
        
        # Add rate multiplier ratio column
        rate_mult_ratio = []
        for _, row in merged_df.iterrows():
            if (not check_null_or_zero(row['rate_multiplier']) and 
                not check_null_or_zero(row['rate_multiplier_rollbit'])):
                ratio = safe_division(row['rate_multiplier'], row['rate_multiplier_rollbit'], None)
                rate_mult_ratio.append(f"{ratio:.4f}x" if ratio is not None else "N/A")
            else:
                rate_mult_ratio.append("N/A")
        
        rate_mult_df['Rate Mult. Ratio (SURF/Rollbit)'] = rate_mult_ratio
        
        # Display rate multiplier comparison with highlighting
        styled_rate_mult_df = rate_mult_df.style.apply(highlight_mapped_pairs, axis=1)
        st.dataframe(styled_rate_mult_df, use_container_width=True)
        
        st.markdown("""
        **Rate Multiplier Interpretation:**
        - Higher values: More aggressive rate scaling
        - Lower values: More conservative rate scaling
        - Rate multiplier affects how quickly rates change with market conditions
        
        <span style="background-color: #e8f5e9; padding: 2px 6px;">Green highlighting</span> indicates mapped pairs with different names.
        """, unsafe_allow_html=True)
    
    with comp_tabs[3]:  # Rate Exponent Comparison
        st.markdown("### Rate Exponent Comparison")
        
        # Create rate exponent comparison table
        rate_exp_df = pd.DataFrame({
            'SURF Pair': merged_df['surf_pair'],
            'Rollbit Pair': merged_df['rollbit_pair'],
            'Type': merged_df['surf_pair'].apply(lambda x: 'Major' if is_major(x) else 'Altcoin'),
            'SURF Rate Exp.': merged_df['rate_exponent'].apply(
                lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A"
            ),
            'Rollbit Rate Exp.': merged_df['rate_exponent_rollbit'].apply(
                lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A"
            )
        })
        
        # Add rate exponent ratio column
        rate_exp_ratio = []
        for _, row in merged_df.iterrows():
            if (not check_null_or_zero(row['rate_exponent']) and 
                not check_null_or_zero(row['rate_exponent_rollbit'])):
                ratio = safe_division(row['rate_exponent'], row['rate_exponent_rollbit'], None)
                rate_exp_ratio.append(f"{ratio:.4f}x" if ratio is not None else "N/A")
            else:
                rate_exp_ratio.append("N/A")
        
        rate_exp_df['Rate Exp. Ratio (SURF/Rollbit)'] = rate_exp_ratio
        
        # Display rate exponent comparison with highlighting
        styled_rate_exp_df = rate_exp_df.style.apply(highlight_mapped_pairs, axis=1)
        st.dataframe(styled_rate_exp_df, use_container_width=True)
        
        st.markdown("""
        **Rate Exponent Interpretation:**
        - Values > 1: Non-linear rate scaling (exponential growth)
        - Values = 1: Linear rate scaling
        - Values < 1: Logarithmic rate scaling (diminishing returns)
        - Rate exponent controls the shape of the rate curve
        
        <span style="background-color: #e8f5e9; padding: 2px 6px;">Green highlighting</span> indicates mapped pairs with different names.
        """, unsafe_allow_html=True)
    
    # Add overall comparison summary
    st.markdown("---")
    st.markdown("### Summary")
    total_pairs = len(comparison_data)
    mapped_count = len(mapped_pairs)
    direct_match_count = total_pairs - mapped_count
    
    st.markdown(f"""
    This comparison shows how SURF's risk parameters compare to Rollbit's for matching trading pairs:
    
    - **Total Comparable Pairs**: {total_pairs}
    - **Direct Matches**: {direct_match_count}
    - **Mapped Pairs**: {mapped_count}
    
    **Parameter Definitions:**
    - **Buffer Rate**: Risk buffer for position management
    - **Position Multiplier**: Maximum position size multiplier
    - **Rate Multiplier**: Scaling factor for rate calculations
    - **Rate Exponent**: Non-linear scaling parameter for rate curves
    
    *Note: "N/A" is displayed when either SURF or Rollbit has null, zero, or missing values for comparison.*
    """)

# --- Main Application ---
def main():
    # Initialize session state
    init_session_state()
    
    st.markdown('<div class="header-style">Surf vs Rollbit Parameters</div>', unsafe_allow_html=True)

    # Sidebar controls
    st.sidebar.header("Controls")

    # Add refresh button with improved functionality
    col1, col2 = st.sidebar.columns([3, 1])
    with col1:
        if st.button("🔄 Refresh Data", use_container_width=True, help="Clear cache and reload all data"):
            with st.spinner("Refreshing data..."):
                # Clear all cached data
                st.cache_data.clear()
                st.cache_resource.clear()
                # Force rerun to refresh the page
                st.rerun()
    
    with col2:
        # Show last refresh time
        current_time = datetime.now().strftime("%H:%M")
        st.markdown(f"<small>{current_time}</small>", unsafe_allow_html=True)

    # Show current pair mappings in sidebar
    st.sidebar.markdown("### Pair Mappings")
    pair_mappings = get_pair_mappings()
    if pair_mappings:
        for surf_pair, rollbit_pair in pair_mappings.items():
            st.sidebar.markdown(f"• {surf_pair} ↔ {rollbit_pair}")
    else:
        st.sidebar.markdown("*No custom mappings configured*")

    # Create simplified tab navigation
    tabs = st.tabs(["Parameter Table", "Rollbit Comparison"])
    
    # Fetch data
    current_params_df = fetch_current_parameters()
    rollbit_df = fetch_rollbit_parameters()

    # Process the data and render tabs
    if current_params_df is not None:
        # Render the appropriate tab content
        with tabs[0]:  # Parameter Table
            # Add sort options
            sort_by = st.selectbox(
                "Sort by:",
                options=["Pair Name", "Token Type"],
                index=0
            )
            
            # Show parameter table
            st.markdown("### Parameter Table")
            render_complete_parameter_table(current_params_df, sort_by)
            
        with tabs[1]:  # Rollbit Comparison
            render_rollbit_comparison(current_params_df, rollbit_df)
            
    else:
        st.error("Failed to load required data. Please check database connection and try refreshing.")

if __name__ == "__main__":
    main()