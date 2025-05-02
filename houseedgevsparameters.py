import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import math
import pytz
from sqlalchemy import create_engine, text

# Page configuration
st.set_page_config(
    page_title="House Edge Adjustment Dashboard",
    page_icon="📊",
    layout="wide"
)

# Apply CSS styles
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
    .highlight {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .warning {
        background-color: #ffe6e6;
        padding: 10px;
        border-radius: 5px;
    }
    .success {
        background-color: #e6ffe6;
        padding: 10px;
        border-radius: 5px;
    }
    .up {
        color: green;
    }
    .down {
        color: red;
    }
    .button-row {
        display: flex;
        justify-content: center;
        gap: 10px;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    .pair-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        border: 1px solid #e0e0e0;
    }
    .pair-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .pair-title {
        font-weight: bold;
        font-size: 18px;
        margin-bottom: 10px;
    }
    .metric-row {
        display: flex;
        justify-content: space-between;
        margin-bottom: 5px;
    }
    .metric-label {
        color: #666;
    }
    .metric-value {
        font-weight: 500;
    }
    .status-indicator {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
    }
    .status-green {
        background-color: #28a745;
    }
    .status-yellow {
        background-color: #ffc107;
    }
    .status-red {
        background-color: #dc3545;
    }
    .parameter-section {
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .info-tooltip {
        color: #888;
        font-size: 14px;
        margin-left: 5px;
    }
    .baseline-value {
        color: #666;
        font-style: italic;
    }
    .timer-container {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 15px;
        border: 1px solid #ddd;
    }
    .timer-label {
        font-weight: 500;
        color: #444;
    }
    .global-action-button {
        margin-top: 10px !important;
        margin-bottom: 10px !important;
    }
    .stButton > button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Function to handle navigation
def navigate_to(view, pair=None):
    """Handle navigation between views with rerun"""
    st.session_state.view_mode = view
    if pair:
        st.session_state.current_pair = pair
    if not st.session_state.get('navigating', False):
        st.session_state.navigating = True
        st.rerun()
    
# Initialize database connection
def init_connection():
    """
    Initialize database connection using Streamlit secrets.
    Returns SQLAlchemy engine or None if connection fails.
    """
    try:
        # Get database credentials from Streamlit secrets
        db_config = st.secrets["database"]
        db_uri = (
            f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}"
            f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )
        engine = create_engine(db_uri)
        return engine
    except Exception as e:
        st.sidebar.error(f"Database connection error: {e}")
        return None

# Fetch available trading pairs
def fetch_pairs():
    """
    Fetch all active trading pairs.
    Returns a list of pair names or default list.
    """
    # If in simulated data mode, return a fixed list
    if st.session_state.get('simulated_data_mode', True):
        return ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT", "XRP/USDT"]
    
    # Otherwise try to fetch from database
    engine = init_connection()
    if not engine:
        return ["BTC/USDT", "ETH/USDT", "SOL/USDT"]  # Default pairs if connection fails
    
    try:
        query = """
        SELECT DISTINCT pair_name 
        FROM public.trade_fill_fresh 
        WHERE created_at > NOW() - INTERVAL '1 day'
        ORDER BY pair_name
        """
        
        df = pd.read_sql(query, engine)
        if df.empty:
            return ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        return df['pair_name'].tolist()
    except Exception as e:
        st.error(f"Error fetching pairs: {e}")
        return ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

# Fetch current parameters for a specific pair
def fetch_current_parameters(pair_name):
    """
    Fetch current parameters for a specific pair.
    Returns a dictionary with all parameter values needed for fee calculation.
    """
    # If in simulated data mode, return fixed values
    if st.session_state.get('simulated_data_mode', True):
        return {
            "buffer_rate": 0.001,
            "position_multiplier": 1000,
            "max_leverage": 100,
            "rate_multiplier": 10000,
            "rate_exponent": 1,
            "pnl_base_rate": 0.0005
        }
    
    # Otherwise try to fetch from database
    engine = init_connection()
    if not engine:
        return {
            "buffer_rate": 0.001,
            "position_multiplier": 1000,
            "max_leverage": 100,
            "rate_multiplier": 10000,
            "rate_exponent": 1,
            "pnl_base_rate": 0.0005
        }
    
    try:
        query = f"""
        SELECT
            (leverage_config::jsonb->0->>'buffer_rate')::numeric AS buffer_rate,
            position_multiplier,
            max_leverage,
            rate_multiplier,
            rate_exponent,
            pnl_base_rate
        FROM
            public.trade_pool_pairs
        WHERE
            pair_name = '{pair_name}'
            AND status = 1
        """
        
        df = pd.read_sql(query, engine)
        if df.empty:
            return {
                "buffer_rate": 0.001,
                "position_multiplier": 1000,
                "max_leverage": 100,
                "rate_multiplier": 10000,
                "rate_exponent": 1,
                "pnl_base_rate": 0.0005
            }
        
        # Convert to dictionary
        params = {
            "buffer_rate": float(df['buffer_rate'].iloc[0]),
            "position_multiplier": float(df['position_multiplier'].iloc[0]),
            "max_leverage": float(df['max_leverage'].iloc[0]),
            "rate_multiplier": float(df['rate_multiplier'].iloc[0]),
            "rate_exponent": float(df['rate_exponent'].iloc[0]),
            "pnl_base_rate": float(df['pnl_base_rate'].iloc[0])
        }
        
        return params
    except Exception as e:
        st.error(f"Error fetching parameters for {pair_name}: {e}")
        return {
            "buffer_rate": 0.001,
            "position_multiplier": 10000,
            "max_leverage": 100,
            "rate_multiplier": 10000,
            "rate_exponent": 1,
            "pnl_base_rate": 0.0005
        }

# Calculate edge for a specific pair
def calculate_edge(pair_name, lookback_minutes=10):
    """
    Calculate house edge for a specific pair.
    Returns edge value or None if calculation fails.
    """
    # Check if we're in simulated data mode
    if st.session_state.get('simulated_data_mode', True):
        # Generate random edge values for testing
        import random
        return 0.001 + random.uniform(-0.0005, 0.0010)
    
    engine = init_connection()
    if engine is None:
        # For testing - generate random edge values when no connection
        import random
        return 0.001 + random.uniform(-0.0005, 0.0010)
    
    try:
        # Get current time in Singapore timezone
        singapore_tz = pytz.timezone('Asia/Singapore')
        now_utc = datetime.now(pytz.utc)
        now_sg = now_utc.astimezone(singapore_tz)
        
        # Calculate lookback period
        start_time_sg = now_sg - timedelta(minutes=lookback_minutes)
        
        # Convert to UTC for database query
        start_time_utc = start_time_sg.astimezone(pytz.utc)
        end_time_utc = now_sg.astimezone(pytz.utc)
        
        # Query for house edge calculation
        edge_query = f"""
        WITH pnl_data AS (
          SELECT
            SUM(-1 * taker_pnl * collateral_price) AS trading_pnl,
            SUM(CASE WHEN taker_fee_mode = 1 AND taker_way IN (1, 3) THEN taker_fee * collateral_price ELSE 0 END) AS taker_fee,
            SUM(CASE WHEN taker_way = 0 THEN -1 * funding_fee * collateral_price ELSE 0 END) AS funding_pnl,
            SUM(taker_sl_fee * collateral_price + maker_sl_fee) AS sl_fee
          FROM public.trade_fill_fresh
          WHERE created_at BETWEEN '{start_time_utc}' AND '{end_time_utc}'
            AND pair_name = '{pair_name}'
        ),
        collateral_data AS (
          SELECT
            SUM(deal_vol * collateral_price) AS open_collateral
          FROM public.trade_fill_fresh
          WHERE taker_fee_mode = 2 AND taker_way IN (1, 3)
            AND created_at BETWEEN '{start_time_utc}' AND '{end_time_utc}'
            AND pair_name = '{pair_name}'
        ),
        rebate_data AS (
          SELECT
            SUM(amount * coin_price) AS rebate_amount
          FROM public.user_cashbooks
          WHERE remark = '给邀请人返佣'
            AND created_at BETWEEN '{start_time_utc}' AND '{end_time_utc}'
        )
        SELECT
          CASE
            WHEN COALESCE(cd.open_collateral, 0) = 0 THEN 0
            ELSE (COALESCE(pd.trading_pnl, 0) + 
                  COALESCE(pd.taker_fee, 0) + 
                  COALESCE(pd.funding_pnl, 0) + 
                  COALESCE(pd.sl_fee, 0) - 
                  COALESCE(rd.rebate_amount, 0)) / 
                 cd.open_collateral
          END AS house_edge,
          COALESCE(pd.trading_pnl, 0) + 
            COALESCE(pd.taker_fee, 0) + 
            COALESCE(pd.funding_pnl, 0) + 
            COALESCE(pd.sl_fee, 0) - 
            COALESCE(rd.rebate_amount, 0) AS pnl,
          COALESCE(cd.open_collateral, 0) AS open_collateral
        FROM pnl_data pd
        CROSS JOIN collateral_data cd
        CROSS JOIN rebate_data rd
        """
        
        df = pd.read_sql(edge_query, engine)
        
        if df.empty or pd.isna(df['house_edge'].iloc[0]):
            # For testing - generate random edge values when no data
            import random
            return 0.001 + random.uniform(-0.0005, 0.0010)
        
        edge_value = float(df['house_edge'].iloc[0])
        
        # If the edge is exactly 0, set a small positive value
        # This avoids issues with initial reference being zero
        if edge_value == 0:
            edge_value = 0.001
            
        # For testing - add variation for testing
        import random
        edge_value += random.uniform(-0.0005, 0.0010)
        
        return edge_value
    
    except Exception as e:
        st.error(f"Error calculating edge for {pair_name}: {e}")
        # Return a random edge value for testing
        import random
        return 0.001 + random.uniform(-0.0005, 0.0010)

# Function to calculate fee for a percentage price move based on the Profit Share Model
def calculate_fee_for_move(move_pct, pnl_base_rate, position_multiplier, rate_multiplier=15000, 
                           rate_exponent=1, bet=1.0, leverage=1.0, debug=False):
    """
    Calculate fee for a percentage price move using the Profit Share Model formula.
    """
    # For negative or zero price moves, no fee is charged
    if move_pct <= 0:
        return 0, 0
    
    # Set initial price (fixed at 100000 as in spreadsheet)
    initial_price = 100000
    
    # Calculate price after move
    price_after_move = initial_price * (1 + move_pct/100)
    
    # Calculate price ratio
    price_ratio = price_after_move / initial_price
    
    # Calculate relative price change 
    relative_change = abs(price_ratio - 1)
    
    # Calculate the terms in the denominator
    term1 = (1 + 1/(relative_change * rate_multiplier)) ** rate_exponent
    term2 = (bet * leverage) / (1000000 * relative_change * position_multiplier)
    
    # Calculate P_close
    p_close = initial_price + (1 - pnl_base_rate) * (price_after_move - initial_price) / (term1 + term2)
    
    # Calculate hypothetical PnL
    hypothetical_pnl = relative_change * initial_price
    
    # Calculate fee amount
    fee_amount = price_after_move - p_close
    
    # Calculate fee as percentage of profit
    fee_percentage = (fee_amount / hypothetical_pnl) * 100 if hypothetical_pnl != 0 else 0
    
    if debug:
        debug_info = {
            "inputs": {
                "move_pct": move_pct,
                "pnl_base_rate": pnl_base_rate,
                "position_multiplier": position_multiplier,
                "rate_multiplier": rate_multiplier,
                "rate_exponent": rate_exponent,
                "bet": bet,
                "leverage": leverage
            },
            "calculations": {
                "initial_price": initial_price,
                "price_after_move": price_after_move,
                "price_ratio": price_ratio,
                "relative_change": relative_change,
                "term1": term1,
                "term2": term2,
                "p_close": p_close,
                "hypothetical_pnl": hypothetical_pnl,
                "fee_amount": fee_amount,
                "fee_percentage": fee_percentage
            }
        }
        return fee_amount, fee_percentage, debug_info
    
    return fee_amount, fee_percentage

# Function to update buffer rate based on edge comparison
def update_buffer_rate(current_buffer, edge, edge_ref, max_leverage, alpha_up=0.1, alpha_down=0.02):
    """
    Update buffer_rate based on edge comparison with proper scaling.
    Increases sharply when edge declines, decreases slowly when edge improves.
    """
    delta = edge - edge_ref
    upper_bound = 0.7 / max_leverage
    lower_bound = 0.0001
    
    # Calculate a normalized delta relative to the reference edge
    if edge_ref != 0:
        normalized_delta = delta / abs(edge_ref)  # Scale relative to reference
    else:
        normalized_delta = delta  # Fallback if reference is zero
    
    # Apply caps to avoid extreme adjustments
    normalized_delta = max(min(normalized_delta, 1.0), -1.0)
    
    # Asymmetric adjustment: fast up, slow down
    adjustment = alpha_up * abs(normalized_delta) * current_buffer if normalized_delta < 0 else -alpha_down * normalized_delta * current_buffer
    
    return max(lower_bound, min(upper_bound, current_buffer + adjustment))

# Function to update position multiplier based on edge comparison
def update_position_multiplier(current_multiplier, edge, edge_ref, alpha_up=0.02, alpha_down=0.1):
    """
    Update position_multiplier based on edge comparison with logarithmic scaling.
    """
    # Handle edge cases safely
    if edge_ref == 0 or current_multiplier <= 0:
        return max(1, min(14000, current_multiplier))
    
    delta = edge - edge_ref
    upper_bound = 14000
    lower_bound = 1
    
    # Calculate normalized delta relative to reference edge
    normalized_delta = delta / abs(edge_ref)
    
    # Cap normalized delta to avoid extreme adjustments
    normalized_delta = max(min(normalized_delta, 1.0), -1.0)
    
    # Convert to log space
    log_pm = np.log(current_multiplier)
    
    # Apply asymmetric adjustment in log space
    if normalized_delta < 0:
        # Edge declining - decrease PM more aggressively
        adjustment = -alpha_down * abs(normalized_delta)
    else:
        # Edge improving - increase PM more conservatively
        adjustment = alpha_up * normalized_delta
    
    # Apply adjustment in log space
    new_log_pm = log_pm + adjustment
    
    # Convert back to linear space
    new_pm = np.exp(new_log_pm)
    
    # Apply bounds
    return max(lower_bound, min(upper_bound, new_pm))

# Function to get status of a pair
def get_pair_status(current_edge, reference_edge, threshold=0.2):
    """
    Determine the status of a pair based on edge comparison.
    Returns 'green', 'yellow', or 'red' as the status.
    """
    if current_edge is None or reference_edge is None:
        return 'yellow'
    
    # Calculate percentage difference
    if reference_edge == 0:
        pct_diff = 0 if current_edge == 0 else float('inf')
    else:
        pct_diff = (current_edge - reference_edge) / abs(reference_edge)
    
    # Determine status based on percentage difference
    if pct_diff >= 0:
        return 'green'  # Edge is equal to or better than reference
    elif pct_diff >= -threshold:
        return 'yellow'  # Edge is worse than reference but within threshold
    else:
        return 'red'  # Edge is significantly worse than reference

# Initialize session state for an individual pair
def init_pair_state(pair_name):
    """Initialize session state for a specific pair."""
    if 'pair_data' not in st.session_state:
        st.session_state.pair_data = {}
    
    if pair_name not in st.session_state.pair_data:
        st.session_state.pair_data[pair_name] = {
            'initialized': False,
            'buffer_rate': 0.001,
            'pnl_base_rate': 0.1,
            'position_multiplier': 1000,
            'max_leverage': 100,
            'rate_multiplier': 15000,
            'rate_exponent': 1,
            'edge_history': [],  # List of (timestamp, edge) tuples
            'buffer_history': [],  # List of (timestamp, buffer_rate) tuples
            'multiplier_history': [],  # List of (timestamp, position_multiplier) tuples
            'fee_history': [],  # List of (timestamp, fee_for_01pct_move) tuples
            'current_edge': None,
            'reference_edge': None,
            'proposed_buffer_rate': None,
            'proposed_position_multiplier': None,
            'reference_buffer_rate': None,
            'reference_position_multiplier': None,
            'last_update_time': None,
            'params_changed': False,
            'current_fee_amount': None,
            'current_fee_percentage': None
        }

# Initialize session state variables
def init_session_state():
    """Initialize all session state variables to prevent duplicates."""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    
    if 'monitored_pairs' not in st.session_state:
        st.session_state.monitored_pairs = []
    
    if 'pair_data' not in st.session_state:
        st.session_state.pair_data = {}
    
    if 'current_pair' not in st.session_state:
        st.session_state.current_pair = None
    
    if 'lookback_minutes' not in st.session_state:
        st.session_state.lookback_minutes = 1  # Default to 1 minute for faster updates
    
    if 'buffer_alpha_up' not in st.session_state:
        st.session_state.buffer_alpha_up = 0.1
    
    if 'buffer_alpha_down' not in st.session_state:
        st.session_state.buffer_alpha_down = 0.02
    
    if 'multiplier_alpha_up' not in st.session_state:
        st.session_state.multiplier_alpha_up = 0.02
    
    if 'multiplier_alpha_down' not in st.session_state:
        st.session_state.multiplier_alpha_down = 0.1
    
    if 'history_length' not in st.session_state:
        st.session_state.history_length = 100  # Default history length
    
    if 'view_mode' not in st.session_state:
        st.session_state.view_mode = "Pairs Overview"  # Default view mode
    
    if 'last_refresh_time' not in st.session_state:
        st.session_state.last_refresh_time = datetime.now(pytz.utc)
    
    # Set the simulated data mode to true by default
    if 'simulated_data_mode' not in st.session_state:
        st.session_state.simulated_data_mode = True
        
    if 'pairs_with_changes' not in st.session_state:
        st.session_state.pairs_with_changes = []
    
    # Navigation tracker to prevent multiple reruns
    if 'navigating' not in st.session_state:
        st.session_state.navigating = False

# Function to get current time in Singapore timezone
def get_sg_time():
    """Get current time in Singapore timezone."""
    singapore_tz = pytz.timezone('Asia/Singapore')
    now_utc = datetime.now(pytz.utc)
    now_sg = now_utc.astimezone(singapore_tz)
    return now_sg

# Function to convert naive datetime to Singapore timezone
def to_singapore_time(dt):
    """Convert naive datetime to Singapore timezone datetime."""
    if dt is None:
        return None
    
    # Check if datetime is already timezone-aware
    if dt.tzinfo is None:
        # Make it timezone-aware by assuming it's UTC
        dt = dt.replace(tzinfo=pytz.utc)
    
    # Convert to Singapore time
    singapore_tz = pytz.timezone('Asia/Singapore')
    return dt.astimezone(singapore_tz)

# Function to format time display consistently
def format_time_display(dt):
    """Format a datetime for display."""
    if dt is None:
        return "N/A"
    if not isinstance(dt, datetime):
        return "N/A"
    try:
        # Basic formatting approach that works with or without timezone info
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except:
        return "N/A"

# Function to calculate and record fee for a 0.1% price move
def calculate_and_record_fee(pair_name, timestamp):
    """Calculate and record the fee percentage for a 0.1% price move for the given pair."""
    # Get required parameters
    pnl_base_rate = st.session_state.pair_data[pair_name]['pnl_base_rate']
    position_multiplier = st.session_state.pair_data[pair_name]['position_multiplier']
    
    # Get rate parameters from session state or fetch them
    if 'rate_multiplier' not in st.session_state.pair_data[pair_name]:
        # Fetch these parameters if not already in session state
        params = fetch_current_parameters(pair_name)
        rate_multiplier = params.get('rate_multiplier', 10000)
        rate_exponent = params.get('rate_exponent', 1)
        
        # Store in session state for future use
        st.session_state.pair_data[pair_name]['rate_multiplier'] = rate_multiplier
        st.session_state.pair_data[pair_name]['rate_exponent'] = rate_exponent
    else:
        rate_multiplier = st.session_state.pair_data[pair_name]['rate_multiplier']
        rate_exponent = st.session_state.pair_data[pair_name]['rate_exponent']
    
    # Calculate fee percentage
    fee_pct = calculate_fee_for_move(
        0.1, 
        pnl_base_rate, 
        position_multiplier,
        rate_multiplier,
        rate_exponent
    )
    
    # Record in fee history
    st.session_state.pair_data[pair_name]['fee_history'].append((timestamp, fee_pct))
    
    # Update current fee values in session state
    st.session_state.pair_data[pair_name]['current_fee_percentage'] = fee_pct
    
    return fee_pct

# Function to process edge data and calculate parameter updates for a specific pair
def process_edge_data(pair_name, timestamp=None):
    """
    Process a new edge data point and calculate parameter updates if needed.
    Returns True if parameters need to be changed, False otherwise.
    """
    # Ensure pair state is initialized
    init_pair_state(pair_name)
    
    if timestamp is None:
        timestamp = get_sg_time()
    elif timestamp.tzinfo is None:
        # Ensure timestamp is timezone-aware
        timestamp = timestamp.replace(tzinfo=pytz.utc)
        timestamp = to_singapore_time(timestamp)
    
    # Fetch new edge
    new_edge = calculate_edge(pair_name, st.session_state.lookback_minutes)
    
    # Skip updates if edge calculation failed, but don't return yet to update timestamp
    if new_edge is not None:
        # If new edge is zero, use a small positive value instead
        if new_edge == 0:
            new_edge = 0.001  # Use 0.1% as minimum edge
            
        # Add to edge history
        st.session_state.pair_data[pair_name]['edge_history'].append((timestamp, new_edge))
        
        # Update current edge
        st.session_state.pair_data[pair_name]['current_edge'] = new_edge
        
        # Get reference edge and current parameters
        edge_ref = st.session_state.pair_data[pair_name]['reference_edge']
        if edge_ref == 0:
            edge_ref = 0.001  # Use a small positive value if reference is zero
            
        current_buffer = st.session_state.pair_data[pair_name]['buffer_rate']
        current_multiplier = st.session_state.pair_data[pair_name]['position_multiplier']
        max_leverage = st.session_state.pair_data[pair_name]['max_leverage']
        
        # Get sensitivity parameters
        buffer_alpha_up = st.session_state.buffer_alpha_up
        buffer_alpha_down = st.session_state.buffer_alpha_down
        multiplier_alpha_up = st.session_state.multiplier_alpha_up
        multiplier_alpha_down = st.session_state.multiplier_alpha_down
        
        # Calculate proposed parameter updates
        new_buffer_rate = update_buffer_rate(
            current_buffer, new_edge, edge_ref, max_leverage,
            buffer_alpha_up, buffer_alpha_down
        )
        
        new_position_multiplier = update_position_multiplier(
            current_multiplier, new_edge, edge_ref,
            multiplier_alpha_up, multiplier_alpha_down
        )
        
        # Check if parameters would change significantly
        parameters_changed = (
            abs(new_buffer_rate - current_buffer) / current_buffer > 0.001 or 
            abs(new_position_multiplier - current_multiplier) / current_multiplier > 0.001
        )
        
        # Store proposed values if changes are needed
        if parameters_changed:
            st.session_state.pair_data[pair_name]['proposed_buffer_rate'] = new_buffer_rate
            st.session_state.pair_data[pair_name]['proposed_position_multiplier'] = new_position_multiplier
            st.session_state.pair_data[pair_name]['params_changed'] = True
            
            # Add to the list of pairs with changes
            if pair_name not in st.session_state.pairs_with_changes:
                st.session_state.pairs_with_changes.append(pair_name)
    else:
        parameters_changed = False
    
    # Update last update time regardless of whether edge calculation succeeded
    st.session_state.pair_data[pair_name]['last_update_time'] = timestamp
    
    return parameters_changed

# Function to update display parameters for a specific pair
def update_display_parameters(pair_name):
    """Apply the proposed parameter updates to display values for a specific pair."""
    # Ensure pair state is initialized
    init_pair_state(pair_name)
    
    if (st.session_state.pair_data[pair_name]['proposed_buffer_rate'] is not None and 
        st.session_state.pair_data[pair_name]['proposed_position_multiplier'] is not None):
        
        # Update parameters
        timestamp = get_sg_time()
        
        # Update buffer rate
        old_buffer = st.session_state.pair_data[pair_name]['buffer_rate']
        st.session_state.pair_data[pair_name]['buffer_rate'] = st.session_state.pair_data[pair_name]['proposed_buffer_rate']
        st.session_state.pair_data[pair_name]['buffer_history'].append((timestamp, st.session_state.pair_data[pair_name]['buffer_rate']))
        
        # Update position multiplier
        old_multiplier = st.session_state.pair_data[pair_name]['position_multiplier']
        st.session_state.pair_data[pair_name]['position_multiplier'] = st.session_state.pair_data[pair_name]['proposed_position_multiplier']
        st.session_state.pair_data[pair_name]['multiplier_history'].append((timestamp, st.session_state.pair_data[pair_name]['position_multiplier']))
        
        # Calculate and record fee for 0.1% move with updated parameters
        calculate_and_record_fee(pair_name, timestamp)
        
        # Update reference parameters to match the new values
        st.session_state.pair_data[pair_name]['reference_buffer_rate'] = st.session_state.pair_data[pair_name]['buffer_rate']
        st.session_state.pair_data[pair_name]['reference_position_multiplier'] = st.session_state.pair_data[pair_name]['position_multiplier']
        st.session_state.pair_data[pair_name]['reference_edge'] = st.session_state.pair_data[pair_name]['current_edge']
        
        # Mark that parameters have been changed
        st.session_state.pair_data[pair_name]['params_changed'] = False
        
        # Reset proposed values
        st.session_state.pair_data[pair_name]['proposed_buffer_rate'] = None
        st.session_state.pair_data[pair_name]['proposed_position_multiplier'] = None
        
        # Remove from list of pairs with changes
        if pair_name in st.session_state.pairs_with_changes:
            st.session_state.pairs_with_changes.remove(pair_name)
        
        return True, old_buffer, st.session_state.pair_data[pair_name]['buffer_rate'], old_multiplier, st.session_state.pair_data[pair_name]['position_multiplier']
    
    return False, None, None, None, None

# Function to reset parameters to reference values for a specific pair
def reset_to_reference_parameters(pair_name):
    """Reset parameters to reference values for a specific pair."""
    # Ensure pair state is initialized
    init_pair_state(pair_name)
    
    if (st.session_state.pair_data[pair_name]['reference_buffer_rate'] is not None and 
        st.session_state.pair_data[pair_name]['reference_position_multiplier'] is not None):
        
        # Update parameters
        timestamp = get_sg_time()
        
        # Update buffer rate
        old_buffer = st.session_state.pair_data[pair_name]['buffer_rate']
        st.session_state.pair_data[pair_name]['buffer_rate'] = st.session_state.pair_data[pair_name]['reference_buffer_rate']
        st.session_state.pair_data[pair_name]['buffer_history'].append((timestamp, st.session_state.pair_data[pair_name]['buffer_rate']))
        
        # Update position multiplier
        old_multiplier = st.session_state.pair_data[pair_name]['position_multiplier']
        st.session_state.pair_data[pair_name]['position_multiplier'] = st.session_state.pair_data[pair_name]['reference_position_multiplier']
        st.session_state.pair_data[pair_name]['multiplier_history'].append((timestamp, st.session_state.pair_data[pair_name]['position_multiplier']))
        
        # Calculate and record fee for 0.1% move
        calculate_and_record_fee(pair_name, timestamp)
        
        # Mark that parameters have been changed
        st.session_state.pair_data[pair_name]['params_changed'] = False
        
        # Reset proposed values
        st.session_state.pair_data[pair_name]['proposed_buffer_rate'] = None
        st.session_state.pair_data[pair_name]['proposed_position_multiplier'] = None
        
        # Remove from list of pairs with changes
        if pair_name in st.session_state.pairs_with_changes:
            st.session_state.pairs_with_changes.remove(pair_name)
        
        return True, old_buffer, st.session_state.pair_data[pair_name]['buffer_rate'], old_multiplier, st.session_state.pair_data[pair_name]['position_multiplier']
    
    return False, None, None, None, None

# Function to create edge plot with matplotlib for a specific pair
def create_edge_plot(pair_name):
    """Create a plot of house edge with reference line for a specific pair."""
    # Ensure pair state is initialized
    init_pair_state(pair_name)
    
    if len(st.session_state.pair_data[pair_name]['edge_history']) < 1:
        return None
    
    # Extract data for plotting
    timestamps = [t for t, _ in st.session_state.pair_data[pair_name]['edge_history']]
    edges = [e for _, e in st.session_state.pair_data[pair_name]['edge_history']]
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot edge line
    ax.plot(timestamps, edges, 'b-', label='House Edge')
    
    # Add reference line if available
    if st.session_state.pair_data[pair_name]['reference_edge'] is not None:
        ax.axhline(y=st.session_state.pair_data[pair_name]['reference_edge'], color='r', linestyle='--', label='Reference Edge')
    
    # Set title and labels
    ax.set_title(f'House Edge Monitoring - {pair_name}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Edge')
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.2%}'.format(x)))
    
    # Add legend
    ax.legend()
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Format dates on x-axis
    fig.autofmt_xdate()
    
    # Tight layout
    plt.tight_layout()
    
    return fig

# Function to create buffer rate and position multiplier plots for a specific pair
def create_parameter_plots(pair_name):
    """Create plots for buffer rate and position multiplier history for a specific pair."""
    # Ensure pair state is initialized
    init_pair_state(pair_name)
    
    buffer_fig, multiplier_fig = None, None
    
    # Buffer rate plot
    if len(st.session_state.pair_data[pair_name]['buffer_history']) >= 1:
        # Extract data for plotting
        buffer_times = [t for t, _ in st.session_state.pair_data[pair_name]['buffer_history']]
        buffer_rates = [r for _, r in st.session_state.pair_data[pair_name]['buffer_history']]
        
        # Create figure and axis
        buffer_fig, buffer_ax = plt.subplots(figsize=(10, 5))
        
        # Plot buffer rate line
        buffer_ax.plot(buffer_times, buffer_rates, 'g-', marker='o', label='Buffer Rate')
        
        # Add reference line if available
        if st.session_state.pair_data[pair_name]['reference_buffer_rate'] is not None:
            buffer_ax.axhline(y=st.session_state.pair_data[pair_name]['reference_buffer_rate'], color='r', 
                             linestyle='--', label='Reference Buffer Rate')
        
        # Set title and labels
        buffer_ax.set_title(f'Buffer Rate Adjustments - {pair_name}')
        buffer_ax.set_xlabel('Time')
        buffer_ax.set_ylabel('Buffer Rate')
        
        # Add legend
        buffer_ax.legend()
        
        # Add grid
        buffer_ax.grid(True, alpha=0.3)
        
        # Format dates on x-axis
        buffer_fig.autofmt_xdate()
        
        # Tight layout
        plt.tight_layout()
    
    # Position multiplier plot
    if len(st.session_state.pair_data[pair_name]['multiplier_history']) >= 1:
        # Extract data for plotting
        multiplier_times = [t for t, _ in st.session_state.pair_data[pair_name]['multiplier_history']]
        multipliers = [m for _, m in st.session_state.pair_data[pair_name]['multiplier_history']]
        
        # Create figure and axis
        multiplier_fig, multiplier_ax = plt.subplots(figsize=(10, 5))
        
        # Plot position multiplier line
        multiplier_ax.plot(multiplier_times, multipliers, 'm-', marker='o', label='Position Multiplier')
        
        # Add reference line if available
        if st.session_state.pair_data[pair_name]['reference_position_multiplier'] is not None:
            multiplier_ax.axhline(y=st.session_state.pair_data[pair_name]['reference_position_multiplier'], color='r', 
                                 linestyle='--', label='Reference Position Multiplier')
        
        # Set title and labels
        multiplier_ax.set_title(f'Position Multiplier Adjustments - {pair_name}')
        multiplier_ax.set_xlabel('Time')
        multiplier_ax.set_ylabel('Position Multiplier')
        
        # Add legend
        multiplier_ax.legend()
        
        # Add grid
        multiplier_ax.grid(True, alpha=0.3)
        
        # Format dates on x-axis
        multiplier_fig.autofmt_xdate()
        
        # Tight layout
        plt.tight_layout()
    
    return buffer_fig, multiplier_fig

# Function to create fee plot for a specific pair
def create_fee_plot(pair_name):
    """Create a plot of fee history for 0.1% price move for a specific pair."""
    # Ensure pair state is initialized
    init_pair_state(pair_name)
    
    if len(st.session_state.pair_data[pair_name]['fee_history']) < 1:
        return None
    
    # Extract data for plotting
    fee_times = [t for t, _ in st.session_state.pair_data[pair_name]['fee_history']]
    fees = [f for _, f in st.session_state.pair_data[pair_name]['fee_history']]
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot fee line
    ax.plot(fee_times, fees, 'r-', marker='o', label='Fee for 0.1% Move')
    
    # Set title and labels
    ax.set_title(f'Fee for 0.1% Price Move - {pair_name}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Fee Amount')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Format dates on x-axis
    fig.autofmt_xdate()
    
    # Tight layout
    plt.tight_layout()
    
    return fig

# Function to create fee percentage vs price move plot
def create_fee_curve_plot(pair_name):
    """Create a plot of fee percentage vs price move."""
    # Ensure pair state is initialized
    init_pair_state(pair_name)
    
    # Get required parameters
    pnl_base_rate = st.session_state.pair_data[pair_name]['pnl_base_rate']
    position_multiplier = st.session_state.pair_data[pair_name]['position_multiplier']
    rate_multiplier = st.session_state.pair_data[pair_name].get('rate_multiplier', 10000)
    rate_exponent = st.session_state.pair_data[pair_name].get('rate_exponent', 1)
    
    # Calculate fee across a range of move sizes
    move_sizes = np.linspace(0, 1, 101)  # Focus on positive moves only (where fees apply)
    current_fee_pcts = []
    
    for move in move_sizes:
        fee_pct = calculate_fee_for_move(
            move, 
            pnl_base_rate, 
            position_multiplier,
            rate_multiplier,
            rate_exponent
        )
        current_fee_pcts.append(fee_pct)
    
    # Create figure and axis for fee percentage
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot current fee percentage curve
    ax.plot(move_sizes, current_fee_pcts, 'g-', label='Current Fee Percentage')
    
    # Plot proposed fee percentage curve if available
    if (st.session_state.pair_data[pair_name]['proposed_buffer_rate'] is not None and 
        st.session_state.pair_data[pair_name]['proposed_position_multiplier'] is not None):
        
        proposed_buffer = st.session_state.pair_data[pair_name]['proposed_buffer_rate']
        proposed_multiplier = st.session_state.pair_data[pair_name]['proposed_position_multiplier']
        
        proposed_fee_pcts = []
        
        for move in move_sizes:
            fee_pct = calculate_fee_for_move(
                move, 
                pnl_base_rate, 
                proposed_multiplier,
                rate_multiplier,
                rate_exponent
            )
            proposed_fee_pcts.append(fee_pct)
        
        ax.plot(move_sizes, proposed_fee_pcts, 'r--', label='Proposed Fee Percentage')
    
    # Set title and labels
    ax.set_title(f'Fee Percentage vs. Price Move Size - {pair_name}')
    ax.set_xlabel('Price Move (%)')
    ax.set_ylabel('Fee (% of Profit)')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend()
    
    # Tight layout
    plt.tight_layout()
    
    return fig

# Function to create PM vs fee sensitivity plot
# Function to create PM vs fee sensitivity plot
def create_pm_fee_sensitivity_plot(pair_name):
    """Create a plot showing how PM changes affect fees at different PM values"""
    # Sample a range of PM values on logarithmic scale
    pm_values = np.logspace(0, 3.5, 20)  # From 1 to ~3000
    
    # Get current parameters
    pnl_base_rate = st.session_state.pair_data[pair_name]['pnl_base_rate']
    rate_multiplier = st.session_state.pair_data[pair_name].get('rate_multiplier', 10000)
    rate_exponent = st.session_state.pair_data[pair_name].get('rate_exponent', 1)
    
    # Calculate fee percentage for 0.1% move for each PM value
    fees = []
    for pm in pm_values:
        fee_pct = calculate_fee_for_move(
            0.1, 
            pnl_base_rate, 
            pm,
            rate_multiplier,
            rate_exponent
        )
        fees.append(fee_pct)
    
    # Create figure for sensitivity analysis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot PM vs Fee curve
    ax.semilogx(pm_values, fees, 'b-', marker='o')
    
    # Add current PM position
    current_pm = st.session_state.pair_data[pair_name]['position_multiplier']
    current_fee = calculate_fee_for_move(
        0.1, 
        pnl_base_rate, 
        current_pm,
        rate_multiplier,
        rate_exponent
    )
    ax.scatter([current_pm], [current_fee], color='red', s=100, zorder=5, label='Current PM')
    
    # Add proposed PM if available
    if st.session_state.pair_data[pair_name]['proposed_position_multiplier'] is not None:
        proposed_pm = st.session_state.pair_data[pair_name]['proposed_position_multiplier']
        # Calculate proposed fee
        proposed_fee = calculate_fee_for_move(
            0.1, 
            pnl_base_rate, 
            proposed_pm,
            rate_multiplier,
            rate_exponent
        )
        ax.scatter([proposed_pm], [proposed_fee], color='green', s=100, zorder=5, label='Proposed PM')
    
    # Set title and labels
    ax.set_title(f'Fee Sensitivity to Position Multiplier - {pair_name}')
    ax.set_xlabel('Position Multiplier (log scale)')
    ax.set_ylabel('Fee for 0.1% Move (%)')
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    return fig

# Function to create fee comparison table for a specific pair
def create_fee_comparison_table(pair_name):
    """Create a table comparing fees for different move sizes with current and proposed parameters."""
    # Ensure pair state is initialized
    init_pair_state(pair_name)
    
    move_sizes = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    
    # Get required parameters
    current_buffer = st.session_state.pair_data[pair_name]['buffer_rate']
    current_multiplier = st.session_state.pair_data[pair_name]['position_multiplier']
    
    # Get rate parameters
    if 'rate_multiplier' not in st.session_state.pair_data[pair_name]:
        # Fetch if not in session state
        params = fetch_current_parameters(pair_name)
        rate_multiplier = params.get('rate_multiplier', 10000)
        rate_exponent = params.get('rate_exponent', 1)
        
        # Store in session state
        st.session_state.pair_data[pair_name]['rate_multiplier'] = rate_multiplier
        st.session_state.pair_data[pair_name]['rate_exponent'] = rate_exponent
    else:
        rate_multiplier = st.session_state.pair_data[pair_name]['rate_multiplier']
        rate_exponent = st.session_state.pair_data[pair_name]['rate_exponent']
        
    # Get pnl_base_rate parameter
    pnl_base_rate = st.session_state.pair_data[pair_name]['pnl_base_rate']
    
    # Calculate fees with current parameters
    current_fee_pcts = []
    
    for move in move_sizes:
        fee_pct = calculate_fee_for_move(
            move, 
            pnl_base_rate, 
            current_multiplier,
            rate_multiplier,
            rate_exponent
        )
        current_fee_pcts.append(fee_pct)
    
    # Calculate fees with proposed parameters if available
    if (st.session_state.pair_data[pair_name]['proposed_buffer_rate'] is not None and 
        st.session_state.pair_data[pair_name]['proposed_position_multiplier'] is not None):
        
        proposed_buffer = st.session_state.pair_data[pair_name]['proposed_buffer_rate']
        proposed_multiplier = st.session_state.pair_data[pair_name]['proposed_position_multiplier']
        
        proposed_fee_pcts = []
        
        for move in move_sizes:
            fee_pct = calculate_fee_for_move(
                move, 
                pnl_base_rate, 
                proposed_multiplier,
                rate_multiplier,
                rate_exponent
            )
            proposed_fee_pcts.append(fee_pct)
        
        # Create dataframe for the table with both current and proposed
        fee_df = pd.DataFrame({
            'Move Size (%)': move_sizes,
            'Current Fee (%)': current_fee_pcts,
            'Proposed Fee (%)': proposed_fee_pcts,
            'Fee % Change': [(new - old) / old * 100 if old != 0 else float('inf') 
                              for new, old in zip(proposed_fee_pcts, current_fee_pcts)]
        })
    else:
        # Create dataframe with just current fees
        fee_df = pd.DataFrame({
            'Move Size (%)': move_sizes,
            'Current Fee (%)': current_fee_pcts
        })
    
    return fee_df

# Function to recommend position multiplier based on target fee
def recommend_position_multiplier(pair_name, target_fee_pct=None):
    """
    Recommend a position multiplier to achieve a target fee percentage.
    If no target specified, uses a sliding scale based on edge performance.
    """
    # If no target specified, calculate based on edge performance
    if target_fee_pct is None:
        current_edge = st.session_state.pair_data[pair_name]['current_edge']
        reference_edge = st.session_state.pair_data[pair_name]['reference_edge']
        
        if current_edge is None or reference_edge is None:
            return None, "Insufficient edge data for recommendation"
        
        # Calculate edge performance ratio
        edge_ratio = current_edge / reference_edge if reference_edge != 0 else 1
        
        # Adjust target fee based on edge performance
        # Better edge performance -> lower fees
        # Worse edge performance -> higher fees
        if edge_ratio >= 1.2:
            # Edge significantly better - recommend lower fees
            target_fee_pct = 15.0
        elif edge_ratio >= 1.0:
            # Edge at or above reference - moderate fees
            target_fee_pct = 25.0
        elif edge_ratio >= 0.8:
            # Edge slightly below reference - higher fees
            target_fee_pct = 35.0
        else:
            # Edge significantly below reference - much higher fees
            target_fee_pct = 45.0
    
    # Get parameters needed for fee calculation
    pnl_base_rate = st.session_state.pair_data[pair_name]['pnl_base_rate']
    rate_multiplier = st.session_state.pair_data[pair_name].get('rate_multiplier', 10000)
    rate_exponent = st.session_state.pair_data[pair_name].get('rate_exponent', 1)
    
    # Generate a range of PM values to search
    pm_values = np.logspace(0, 3.5, 100)  # From 1 to ~3000
    
    # Calculate fees for each PM value
    closest_pm = None
    min_diff = float('inf')
    
    for pm in pm_values:
        _, fee_pct = calculate_fee_for_move(
            0.1, 
            pnl_base_rate, 
            pm,
            rate_multiplier,
            rate_exponent
        )
        
        # Find PM that gives fee closest to target
        diff = abs(fee_pct - target_fee_pct)
        if diff < min_diff:
            min_diff = diff
            closest_pm = pm
    
    return closest_pm, f"Recommended PM for target fee of {target_fee_pct:.1f}%: {closest_pm:.1f}"

# Initialize a trading pair for monitoring
def initialize_pair(pair_name):
    """Initialize a trading pair for monitoring, including all required parameters."""
    # Initialize pair state if it doesn't exist
    init_pair_state(pair_name)
    
    # Fetch current parameters from database
    params = fetch_current_parameters(pair_name)
    
    # Update session state with current parameters
    st.session_state.pair_data[pair_name]['buffer_rate'] = params["buffer_rate"]
    st.session_state.pair_data[pair_name]['pnl_base_rate'] = params["pnl_base_rate"]
    st.session_state.pair_data[pair_name]['position_multiplier'] = params["position_multiplier"]
    st.session_state.pair_data[pair_name]['max_leverage'] = params["max_leverage"]
    st.session_state.pair_data[pair_name]['rate_multiplier'] = params["rate_multiplier"]
    st.session_state.pair_data[pair_name]['rate_exponent'] = params["rate_exponent"]
    
    # Save reference values
    st.session_state.pair_data[pair_name]['reference_buffer_rate'] = params["buffer_rate"]
    st.session_state.pair_data[pair_name]['reference_position_multiplier'] = params["position_multiplier"]
    
    # Reset history - use Singapore time
    timestamp = get_sg_time()
    st.session_state.pair_data[pair_name]['edge_history'] = []
    st.session_state.pair_data[pair_name]['buffer_history'] = [(timestamp, st.session_state.pair_data[pair_name]['buffer_rate'])]
    st.session_state.pair_data[pair_name]['multiplier_history'] = [(timestamp, st.session_state.pair_data[pair_name]['position_multiplier'])]
    
    # Calculate and record initial fee for 0.1% move
    fee_amount, fee_pct = calculate_and_record_fee(pair_name, timestamp)
    
    # Fetch initial reference edge based on selected lookback period
    initial_edge = calculate_edge(pair_name, st.session_state.lookback_minutes)
    if initial_edge is not None:
        # Use a small positive default if edge is zero
        if initial_edge == 0:
            initial_edge = 0.001  # Use 0.1% as minimum edge
        
        st.session_state.pair_data[pair_name]['reference_edge'] = initial_edge
        st.session_state.pair_data[pair_name]['current_edge'] = initial_edge
        st.session_state.pair_data[pair_name]['edge_history'] = [(timestamp, initial_edge)]
    
    # Reset proposed values
    st.session_state.pair_data[pair_name]['proposed_buffer_rate'] = None
    st.session_state.pair_data[pair_name]['proposed_position_multiplier'] = None
    
    # Reset params_changed flag
    st.session_state.pair_data[pair_name]['params_changed'] = False
    
    # Set last update time
    st.session_state.pair_data[pair_name]['last_update_time'] = timestamp
    
    # Mark pair as initialized
    st.session_state.pair_data[pair_name]['initialized'] = True
    
    return True

# Function to render the pair overview cards
def render_pair_overview():
    """Render overview cards for all monitored pairs."""
    if not st.session_state.monitored_pairs:
        st.info("No pairs are currently being monitored. Select a pair and click 'Add Pair' in the sidebar.")
        return
    
    st.markdown("### Monitored Trading Pairs")
    st.markdown("Select a pair below to view detailed analytics or click 'Monitor' to update parameters.")
    
    # Show changes summary if there are any pairs with pending changes
    if st.session_state.pairs_with_changes:
        with st.container():
            st.markdown(f"""
            <div style="background-color: #fff3cd; border-radius: 8px; padding: 15px; margin-bottom: 20px; border: 1px solid #ffeeba;">
                <h4 style="margin-top: 0; color: #856404;">⚠️ Parameter Updates Available</h4>
                <p>{len(st.session_state.pairs_with_changes)} pairs have recommended parameter updates that need to be applied.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Add "Apply All Changes" button
            if st.button("Apply All Recommended Changes", key="apply_all_changes", type="primary"):
                with st.spinner("Applying parameter updates to all pairs..."):
                    applied_count = 0
                    for pair_with_changes in st.session_state.pairs_with_changes.copy():
                        success, _, _, _, _ = update_display_parameters(pair_with_changes)
                        if success:
                            applied_count += 1
                
                st.success(f"Successfully applied updates to {applied_count} pairs.")
                # Reset navigation state and rerun
                st.session_state.navigating = False
                st.rerun()
    
    # Create a grid layout for pair cards (3 columns)
    columns = st.columns(3)
    
    # Get current time in Singapore timezone for comparison
    current_time = get_sg_time()
    
    # Render a card for each monitored pair
    for i, pair_name in enumerate(st.session_state.monitored_pairs):
        with columns[i % 3]:
            # Get pair data
            pair_data = st.session_state.pair_data.get(pair_name, {})
            current_edge = pair_data.get('current_edge')
            reference_edge = pair_data.get('reference_edge')
            buffer_rate = pair_data.get('buffer_rate')
            position_multiplier = pair_data.get('position_multiplier')
            fee_percentage = pair_data.get('current_fee_percentage')
            last_update = pair_data.get('last_update_time')
            params_changed = pair_data.get('params_changed', False)
            
            # Determine status indicator
            status = get_pair_status(current_edge, reference_edge)
            status_color = "green" if status == "green" else "orange" if status == "yellow" else "red"
            
            # Format edge values and last update time
            edge_display = f"{current_edge:.4%}" if current_edge is not None else "N/A"
            ref_display = f"{reference_edge:.4%}" if reference_edge is not None else "N/A"
            fee_display = f"{fee_percentage:.2f}%" if fee_percentage is not None else "N/A"
            
            # Calculate edge delta
            if current_edge is not None and reference_edge is not None:
                edge_delta = current_edge - reference_edge
                delta_pct = edge_delta / abs(reference_edge) * 100 if reference_edge != 0 else 0
                delta_color = "green" if edge_delta >= 0 else "red"
                delta_display = f"{edge_delta:.4%} ({delta_pct:+.2f}%)"
            else:
                delta_display = "N/A"
                delta_color = "gray"
            
            # Calculate last update time - ensure it's in Singapore time for comparison
            if last_update:
                sg_last_update = last_update
                if sg_last_update.tzinfo is None:
                    sg_last_update = to_singapore_time(sg_last_update)
                    
                time_since_update = current_time - sg_last_update
                if time_since_update.total_seconds() < 60:
                    update_display = f"{int(time_since_update.total_seconds())} seconds ago"
                elif time_since_update.total_seconds() < 3600:
                    update_display = f"{int(time_since_update.total_seconds() / 60)} minutes ago"
                else:
                    update_display = f"{int(time_since_update.total_seconds() / 3600)} hours ago"
            else:
                update_display = "Never"
            
            # Safely format values with proper error handling
            buffer_display = f"{buffer_rate:.6f}" if buffer_rate is not None else "N/A"
            multiplier_display = f"{position_multiplier:.1f}" if position_multiplier is not None else "N/A"
            
            # Create a container with custom styling for the card
            with st.container():
                # Highlight pairs with recommended parameter changes
                card_border = "#ffc107" if params_changed else "#e0e0e0"
                card_bg = "#fffbf0" if params_changed else "#f8f9fa"
                
                st.markdown(f"""
                <div style="background-color: {card_bg}; border-radius: 10px; padding: 15px; margin-bottom: 15px; border: 1px solid {card_border};">
                    <div style="font-weight: bold; font-size: 18px; margin-bottom: 10px;">
                        <span style="display: inline-block; width: 12px; height: 12px; border-radius: 50%; background-color: {status_color}; margin-right: 8px;"></span>
                        {pair_name} {" ⚠️" if params_changed else ""}
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <span style="color: #666;">Current Edge:</span>
                        <span style="font-weight: 500;">{edge_display}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <span style="color: #666;">Reference Edge:</span>
                        <span style="font-weight: 500;">{ref_display}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <span style="color: #666;">Delta:</span>
                        <span style="font-weight: 500; color: {delta_color};">{delta_display}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <span style="color: #666;">Buffer Rate:</span>
                        <span style="font-weight: 500;">{buffer_display}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <span style="color: #666;">Position Multiplier:</span>
                        <span style="font-weight: 500;">{multiplier_display}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <span style="color: #666;">Fee for 0.1% Move:</span>
                        <span style="font-weight: 500;">{fee_display}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <span style="color: #666;">Last Update:</span>
                        <span style="font-weight: 500;">{update_display}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Use standard Streamlit buttons instead of HTML buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"View Details", key=f"view_{pair_name}", on_click=navigate_to, args=("Pair Detail", pair_name)):
                        pass  # Navigation is handled by the on_click callback
                with col2:
                    button_label = f"Monitor{' ⚠️' if params_changed else ''}"
                    if st.button(button_label, key=f"monitor_{pair_name}", on_click=navigate_to, args=("Pair Monitor", pair_name)):
                        pass  # Navigation is handled by the on_click callback

# Function to render the detailed pair view
def render_pair_detail(pair_name):
    """Render detailed view for a specific pair."""
    # Ensure pair state is initialized
    init_pair_state(pair_name)
    
    # Check if the pair has been initialized properly
    if not st.session_state.pair_data[pair_name].get('initialized', False) or st.session_state.pair_data[pair_name].get('reference_edge') is None:
        st.warning(f"Pair {pair_name} has not been properly initialized. Please initialize it first.")
        return
    
    st.markdown(f"### Detailed Analytics: {pair_name}")
    
    # Create columns for key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Show current edge
    with col1:
        if st.session_state.pair_data[pair_name]['current_edge'] is not None:
            edge_delta = st.session_state.pair_data[pair_name]['current_edge'] - st.session_state.pair_data[pair_name]['reference_edge'] if st.session_state.pair_data[pair_name]['reference_edge'] is not None else None
            delta_color = "down" if edge_delta and edge_delta < 0 else "up"
            delta_str = f"<span class='{delta_color}'>{edge_delta:.4%}</span>" if edge_delta is not None else ""
            st.markdown(f"**Current Edge:** {st.session_state.pair_data[pair_name]['current_edge']:.4%} {delta_str}", unsafe_allow_html=True)
        else:
            st.markdown("**Current Edge:** N/A")
    
    # Show reference edge
    with col2:
        if st.session_state.pair_data[pair_name]['reference_edge'] is not None:
            st.markdown(f"**Reference Edge:** {st.session_state.pair_data[pair_name]['reference_edge']:.4%}")
        else:
            st.markdown("**Reference Edge:** N/A")
    
    # Show current buffer rate
    with col3:
        st.markdown(f"**Buffer Rate:** {st.session_state.pair_data[pair_name]['buffer_rate']:.6f}")
    
    # Show current position multiplier
    with col4:
        st.markdown(f"**Position Multiplier:** {st.session_state.pair_data[pair_name]['position_multiplier']:.1f}")
    
    # Show current fee for 0.1% move
    with col5:
        if st.session_state.pair_data[pair_name].get('current_fee_percentage') is not None:
            st.markdown(f"**Fee for 0.1% Move:** {st.session_state.pair_data[pair_name]['current_fee_percentage']:.2f}%")
        else:
            st.markdown("**Fee for 0.1% Move:** N/A")
    
    # Create tabbed view for detailed analytics
    detail_tabs = st.tabs(["Edge History", "Parameter History", "Fee Analysis", "PM Sensitivity"])
    
    # Edge History tab
    with detail_tabs[0]:
        # Create edge plot
        edge_plot = create_edge_plot(pair_name)
        if edge_plot is not None:
            st.pyplot(edge_plot)
        else:
            st.info("Not enough data points yet for edge visualization.")
            
        # Add raw edge history data display
        if st.checkbox("Show Raw Edge History Data", key=f"show_edge_history_{pair_name}"):
            if len(st.session_state.pair_data[pair_name]['edge_history']) > 0:
                edge_df = pd.DataFrame({
                    'Timestamp': [t for t, _ in st.session_state.pair_data[pair_name]['edge_history']],
                    'Edge': [f"{e:.6f}" for _, e in st.session_state.pair_data[pair_name]['edge_history']]
                })
                st.dataframe(edge_df, hide_index=True, use_container_width=True)
            else:
                st.info("No edge history data yet.")
    
    # Parameter History tab
    with detail_tabs[1]:
        # Create parameter plots
        buffer_fig, multiplier_fig = create_parameter_plots(pair_name)
        
        if buffer_fig is not None:
            st.pyplot(buffer_fig)
        else:
            st.info("Not enough data points yet for buffer rate visualization.")
            
        if multiplier_fig is not None:
            st.pyplot(multiplier_fig)
        else:
            st.info("Not enough data points yet for position multiplier visualization.")
            
        # Show fee history plot
        fee_fig = create_fee_plot(pair_name)
        if fee_fig is not None:
            st.pyplot(fee_fig)
        else:
            st.info("Not enough data points yet for fee visualization.")
            
        # Show raw parameter history data
        if st.checkbox("Show Raw Parameter History Data", key=f"show_param_history_{pair_name}"):
            # Create tabs for different history tables
            history_tabs = st.tabs(["Buffer History", "Multiplier History", "Fee History"])
            
            with history_tabs[0]:
                if len(st.session_state.pair_data[pair_name]['buffer_history']) > 0:
                    buffer_df = pd.DataFrame({
                        'Timestamp': [t for t, _ in st.session_state.pair_data[pair_name]['buffer_history']],
                        'Buffer Rate': [f"{r:.6f}" for _, r in st.session_state.pair_data[pair_name]['buffer_history']]
                    })
                    st.dataframe(buffer_df, hide_index=True, use_container_width=True)
                else:
                    st.info("No buffer rate history data yet.")
            
            with history_tabs[1]:
                if len(st.session_state.pair_data[pair_name]['multiplier_history']) > 0:
                    multiplier_df = pd.DataFrame({
                        'Timestamp': [t for t, _ in st.session_state.pair_data[pair_name]['multiplier_history']],
                        'Position Multiplier': [f"{m:.1f}" for _, m in st.session_state.pair_data[pair_name]['multiplier_history']]
                    })
                    st.dataframe(multiplier_df, hide_index=True, use_container_width=True)
                else:
                    st.info("No position multiplier history data yet.")
            
            with history_tabs[2]:
                if len(st.session_state.pair_data[pair_name]['fee_history']) > 0:
                    fee_df = pd.DataFrame({
                        'Timestamp': [t for t, _ in st.session_state.pair_data[pair_name]['fee_history']],
                        'Fee for 0.1% Move': [f"{f:.8f}" for _, f in st.session_state.pair_data[pair_name]['fee_history']]
                    })
                    st.dataframe(fee_df, hide_index=True, use_container_width=True)
                else:
                    st.info("No fee history data yet.")
    
    # Fee Analysis tab
    with detail_tabs[2]:
        st.markdown("### Fee Analysis")
        
        # Create fee curve plot
        fee_percentage_fig = create_fee_curve_plot(pair_name)
        if fee_percentage_fig is not None:
            st.markdown("#### Fee Percentage vs Price Move")
            st.pyplot(fee_percentage_fig)
        
        # Fee comparison table
        st.markdown("### Fee Comparison Table")
        
        # Create fee comparison table
        fee_df = create_fee_comparison_table(pair_name)
        st.dataframe(fee_df, use_container_width=True)
        
        # Fee equation details
        st.markdown("### Fee Equation")
        st.markdown("""
        The fee is calculated using the following equation:
        
        $Fee = -1 \\times \\frac{1 + Rate\\_Multiplier \\cdot |\\frac{P_t}{P_T} - 1|^{Rate\\_Exponent}}{1 + 10^6 \\cdot Position\\_Multiplier \\cdot |\\frac{P_t}{P_T} - 1|^{Rate\\_Exponent}} \\times \\frac{Bet \\times Leverage}{1 - Buffer\\_Rate} \\times (P_T - P_t)$
        
        Where:
        - $P_T$ is the initial price
        - $P_t$ is the price after the move
        - $Buffer\\_Rate$ is the buffer rate parameter
        - $Position\\_Multiplier$ is the position multiplier parameter
        - $Rate\\_Multiplier$ and $Rate\\_Exponent$ are additional parameters affecting the fee curve
        """)
        
    # PM Sensitivity tab
    with detail_tabs[3]:
        st.markdown("### Position Multiplier Sensitivity Analysis")
        
        # Show PM fee sensitivity plot
        sensitivity_fig = create_pm_fee_sensitivity_plot(pair_name)
        if sensitivity_fig is not None:
            st.pyplot(sensitivity_fig)
            
            # Add explanation
            st.markdown("""
            This plot shows how the fee for a 0.1% price move changes with different position multiplier values.
            - The x-axis uses a logarithmic scale to better visualize the relationship.
            - Lower position multiplier values result in higher fees.
            - The red point shows the current position multiplier.
            - If a proposed position multiplier exists, it is shown as a green point.
            """)
        
        # Show PM recommendation section
        st.markdown("### Position Multiplier Recommendations")
        
        # Add a selection for different target fees
        target_fees = [15, 20, 25, 30, 35, 40, 45, 50]
        target_fee = st.selectbox(
            "Select target fee percentage for 0.1% move:",
            options=target_fees,
            index=3,  # Default to 30%
            key=f"target_fee_select_{pair_name}"
        )
        
        # Calculate recommended PM for the selected target fee
        recommended_pm, recommendation_text = recommend_position_multiplier(pair_name, target_fee)
        
        if recommended_pm is not None:
            st.success(recommendation_text)
            
            # Show comparison with current PM
            current_pm = st.session_state.pair_data[pair_name]['position_multiplier']
            
            # Calculate percentage change
            pct_change = (recommended_pm - current_pm) / current_pm * 100
            change_text = f"Recommended PM is {pct_change:+.1f}% compared to current PM"
            
            st.info(f"Current PM: {current_pm:.1f}")
            st.info(f"Recommended PM: {recommended_pm:.1f} ({change_text})")
    
    # Button to return to overview
    if st.button("Return to Pairs Overview", type="secondary", on_click=navigate_to, args=("Pairs Overview",)):
        pass  # Navigation is handled by the on_click callback

# Function to render the parameter adjustment view
def render_pair_monitor(pair_name):
    """Render parameter monitoring and adjustment view for a specific pair."""
    # Ensure pair state is initialized
    init_pair_state(pair_name)
    
    # Check if the pair has been initialized properly
    if not st.session_state.pair_data[pair_name].get('initialized', False) or st.session_state.pair_data[pair_name].get('reference_edge') is None:
        st.warning(f"Pair {pair_name} has not been properly initialized. Please initialize it first.")
        return
    
    st.markdown(f"### Parameter Monitoring: {pair_name}")
    
    # Create columns for key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Show current edge
    with col1:
        if st.session_state.pair_data[pair_name]['current_edge'] is not None:
            edge_delta = st.session_state.pair_data[pair_name]['current_edge'] - st.session_state.pair_data[pair_name]['reference_edge'] if st.session_state.pair_data[pair_name]['reference_edge'] is not None else None
            delta_color = "down" if edge_delta and edge_delta < 0 else "up"
            delta_str = f"<span class='{delta_color}'>{edge_delta:.4%}</span>" if edge_delta is not None else ""
            st.markdown(f"**Current Edge:** {st.session_state.pair_data[pair_name]['current_edge']:.4%} {delta_str}", unsafe_allow_html=True)
        else:
            st.markdown("**Current Edge:** N/A")
    
    # Show reference edge
    with col2:
        if st.session_state.pair_data[pair_name]['reference_edge'] is not None:
            st.markdown(f"**Reference Edge:** {st.session_state.pair_data[pair_name]['reference_edge']:.4%}")
        else:
            st.markdown("**Reference Edge:** N/A")
    
    # Show current buffer rate
    with col3:
        st.markdown(f"**Buffer Rate:** {st.session_state.pair_data[pair_name]['buffer_rate']:.6f}")
    
    # Show current position multiplier
    with col4:
        st.markdown(f"**Position Multiplier:** {st.session_state.pair_data[pair_name]['position_multiplier']:.1f}")
    
    # Show current fee for 0.1% move
    with col5:
        if st.session_state.pair_data[pair_name].get('current_fee_percentage') is not None:
            st.markdown(f"**Fee for 0.1% Move:** {st.session_state.pair_data[pair_name]['current_fee_percentage']:.2f}%")
        else:
            st.markdown("**Fee for 0.1% Move:** N/A")
    
    # Show edge plot
    edge_plot = create_edge_plot(pair_name)
    if edge_plot is not None:
        st.pyplot(edge_plot)
    else:
        st.info("Not enough data points yet for edge visualization.")
    
    # Parameter update section
    param_col1, param_col2 = st.columns(2)
    
    with param_col1:
        # Manual update button
        if st.button("Fetch New Data", key=f"fetch_data_{pair_name}", help="Get the latest edge data and calculate parameter updates"):
            process_edge_data(pair_name)
            
            # Calculate fee with updated edge data
            timestamp = get_sg_time()
            calculate_and_record_fee(pair_name, timestamp)
            
            # Reset navigation state and rerun
            st.session_state.navigating = False
            st.rerun()
    
    with param_col2:
        # Reset to reference parameters button
        if st.button("Restore Baseline Parameters", 
                    type="secondary", 
                    key=f"reset_params_{pair_name}",
                    help="Discard current changes and restore the last saved reference parameters"):
            success, old_buffer, new_buffer, old_multiplier, new_multiplier = reset_to_reference_parameters(pair_name)
            if success:
                st.success("Parameters reset to reference values")
                # Reset navigation state and rerun
                st.session_state.navigating = False
                st.rerun()
            else:
                st.error("No reference parameters available")
    
    # Show parameter update notification
    if st.session_state.pair_data[pair_name].get('params_changed', False):
        st.markdown('<div class="warning">Parameter updates available. Review proposed changes below.</div>', unsafe_allow_html=True)
        
        # Parameter change details
        if st.session_state.pair_data[pair_name]['proposed_buffer_rate'] is not None and st.session_state.pair_data[pair_name]['proposed_position_multiplier'] is not None:
            delta_buffer = st.session_state.pair_data[pair_name]['proposed_buffer_rate'] - st.session_state.pair_data[pair_name]['buffer_rate']
            delta_multiplier = st.session_state.pair_data[pair_name]['proposed_position_multiplier'] - st.session_state.pair_data[pair_name]['position_multiplier']
            
            # Get rate parameters
            rate_multiplier = st.session_state.pair_data[pair_name].get('rate_multiplier', 10000)
            rate_exponent = st.session_state.pair_data[pair_name].get('rate_exponent', 1)
            
            # Calculate current and new fees - ensure proper calculation
            current_fee_amount, current_fee_pct = calculate_fee_for_move(
                0.1, 
                st.session_state.pair_data[pair_name]['pnl_base_rate'], 
                st.session_state.pair_data[pair_name]['position_multiplier'],
                rate_multiplier,
                rate_exponent
            )
            
            new_fee_amount, new_fee_pct = calculate_fee_for_move(
                0.1, 
                st.session_state.pair_data[pair_name]['pnl_base_rate'], 
                st.session_state.pair_data[pair_name]['proposed_position_multiplier'],
                rate_multiplier,
                rate_exponent
            )
            
            delta_fee_amount = new_fee_amount - current_fee_amount
            delta_fee_pct = new_fee_pct - current_fee_pct
            
            # Calculate percentage change safely
            if abs(current_fee_pct) > 1e-10:
                pct_fee_change = (delta_fee_pct / abs(current_fee_pct)) * 100
            else:
                pct_fee_change = 0 if abs(delta_fee_pct) < 1e-10 else float('inf')
            
            # Show proposed changes in a better formatted layout
            st.markdown("### Proposed Parameter Changes")
            
            st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Buffer Rate**")
                st.markdown(f"Current: {st.session_state.pair_data[pair_name]['buffer_rate']:.6f}")
                st.markdown(f"Proposed: {st.session_state.pair_data[pair_name]['proposed_buffer_rate']:.6f}")
                
                # Show change with color
                pct_change = delta_buffer/st.session_state.pair_data[pair_name]['buffer_rate']*100
                change_color = "up" if delta_buffer > 0 else "down"
                st.markdown(f"Change: {delta_buffer:.6f} (<span class='{change_color}'>{pct_change:+.2f}%</span>)", unsafe_allow_html=True)
            
            with col2:
                st.markdown("**Position Multiplier**")
                st.markdown(f"Current: {st.session_state.pair_data[pair_name]['position_multiplier']:.1f}")
                st.markdown(f"Proposed: {st.session_state.pair_data[pair_name]['proposed_position_multiplier']:.1f}")
                
                # Show change with color
                pct_change = delta_multiplier/st.session_state.pair_data[pair_name]['position_multiplier']*100
                change_color = "up" if delta_multiplier > 0 else "down"
                st.markdown(f"Change: {delta_multiplier:.1f} (<span class='{change_color}'>{pct_change:+.2f}%</span>)", unsafe_allow_html=True)
            
            with col3:
                st.markdown("**Fee for 0.1% Move**")
                st.markdown(f"Current: {current_fee_pct:.2f}%")
                st.markdown(f"Proposed: {new_fee_pct:.2f}%")
                
                # Show change with proper color and format
                change_color = "up" if delta_fee_pct > 0 else "down"
                if abs(pct_fee_change) != float('inf'):
                    st.markdown(f"Change: {delta_fee_pct:.2f}% (<span class='{change_color}'>{pct_fee_change:+.2f}%</span>)", unsafe_allow_html=True)
                else:
                    st.markdown(f"Change: {delta_fee_pct:.2f}%", unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show PM sensitivity visualization
            st.markdown("### Fee Sensitivity Analysis")
            sensitivity_fig = create_pm_fee_sensitivity_plot(pair_name)
            if sensitivity_fig is not None:
                st.pyplot(sensitivity_fig)
            
            # Add a view of fee changes for different move sizes
            with st.expander("View Fee Comparison for Different Move Sizes"):
                fee_df = create_fee_comparison_table(pair_name)
                st.dataframe(fee_df, use_container_width=True)
            
            # Update display parameters button - more prominent placement
            update_col1, update_col2 = st.columns([3, 1])
            
            with update_col1:
                st.info("Review the proposed changes carefully. If you apply these changes, they will become the new baseline parameters for this pair.")
                
            with update_col2:
                if st.button("Apply Changes", type="primary", key=f"update_params_{pair_name}", help="Apply proposed parameter updates and set them as the new baseline"):
                    # Apply the updates to display (not database)
                    success, old_buffer, new_buffer, old_multiplier, new_multiplier = update_display_parameters(pair_name)
                    
                    if success:
                        st.markdown('<div class="success">Parameters updated successfully! New baseline established.</div>', unsafe_allow_html=True)
                        # Reset navigation state and rerun
                        st.session_state.navigating = False
                        st.rerun()
    else:
        # No parameter changes to show
        st.info("No parameter changes needed at this time. The current edge is within acceptable bounds of the reference edge.")
        
        # Show the sensitivity analysis regardless
        st.markdown("### Fee Sensitivity Analysis")
        sensitivity_fig = create_pm_fee_sensitivity_plot(pair_name)
        if sensitivity_fig is not None:
            st.pyplot(sensitivity_fig)
    
    # Add a horizontal line to separate sections
    st.markdown("---")
    
    # Button to return to overview - at the bottom for consistent placement
    if st.button("Return to Pairs Overview", type="secondary", key=f"return_from_monitor_{pair_name}", on_click=navigate_to, args=("Pairs Overview",)):
        pass  # Navigation is handled by the on_click callback
        
# Main application
def main():
    # Initialize session state
    init_session_state()
    
    # Sidebar configuration
    st.sidebar.title("House Edge Control")
    
    # Add a note about refresh rates having been turned off
    st.sidebar.info("Auto-update functionality has been disabled to resolve performance issues. Use the manual refresh buttons instead.")
    
    # Parameter control section in sidebar
    st.sidebar.markdown("### Parameter Controls")
    
    # Always use simulated data for now
    st.session_state.simulated_data_mode = True
    
    # Pair selection
    st.sidebar.markdown("#### Trading Pair Selection")
    
    # Fetch available pairs
    available_pairs = fetch_pairs()
    
    # Pair selection dropdown
    selected_pair = st.sidebar.selectbox(
        "Select Pair", 
        available_pairs, 
        key="pair_select"
    )
    
    # Add pair button
    if st.sidebar.button("Add Pair for Monitoring", key="add_pair"):
        if selected_pair not in st.session_state.monitored_pairs:
            st.session_state.monitored_pairs.append(selected_pair)
            # Initialize the pair
            initialize_pair(selected_pair)
            st.sidebar.success(f"Added {selected_pair} to monitoring list!")
            # Reset navigation state and rerun
            st.session_state.navigating = False
            st.rerun()
        else:
            st.sidebar.warning(f"{selected_pair} is already being monitored.")
    
    # Add all pairs button
    if st.sidebar.button("Add All Pairs", key="add_all_pairs"):
        new_pairs_added = 0
        for pair in available_pairs:
            if pair not in st.session_state.monitored_pairs:
                st.session_state.monitored_pairs.append(pair)
                # Initialize the pair
                initialize_pair(pair)
                new_pairs_added += 1
        
        if new_pairs_added > 0:
            st.sidebar.success(f"Added {new_pairs_added} new pairs to monitoring list!")
            # Reset navigation state and rerun
            st.session_state.navigating = False
            st.rerun()
        else:
            st.sidebar.info("All available pairs are already being monitored.")
    
    # Remove pair button
    if st.session_state.monitored_pairs:
        remove_pair = st.sidebar.selectbox(
            "Select Pair to Remove",
            st.session_state.monitored_pairs,
            key="remove_pair_select"
        )
        
        if st.sidebar.button("Remove Pair", key="remove_pair"):
            st.session_state.monitored_pairs.remove(remove_pair)
            # Clean up pair data
            if remove_pair in st.session_state.pair_data:
                del st.session_state.pair_data[remove_pair]
            # Also remove from pairs with changes if present
            if remove_pair in st.session_state.pairs_with_changes:
                st.session_state.pairs_with_changes.remove(remove_pair)
            st.sidebar.success(f"Removed {remove_pair} from monitoring list!")
            # Reset navigation state and rerun
            st.session_state.navigating = False
            st.rerun()
    
    # Lookback period selection
    st.sidebar.markdown("#### Lookback Period")
    lookback_options = {"1 minute": 1, "5 minutes": 5, "10 minutes": 10}
    selected_interval = st.sidebar.radio(
        "Select edge calculation period:",
        options=list(lookback_options.keys()),
        key="interval_radio"
    )
    # Update lookback minutes in session state
    st.session_state.lookback_minutes = lookback_options[selected_interval]
    
    # Parameter adjustment sensitivity controls
    st.sidebar.markdown("#### Parameter Adjustment Controls")
    
    # Expand sensitivity controls in an expander
    with st.sidebar.expander("Adjustment Sensitivity Controls"):
        # Buffer rate adjustment sensitivity
        st.markdown("##### Buffer Rate Adjustment Sensitivity")
        buffer_alpha_up = st.slider(
            "Buffer Increase Rate (when edge decreases):",
            min_value=0.01, 
            max_value=0.5, 
            value=st.session_state.buffer_alpha_up,
            step=0.01,
            key="buffer_alpha_up_slider",
            help="Higher values make buffer rate increase more quickly when edge decreases"
        )
        
        buffer_alpha_down = st.slider(
            "Buffer Decrease Rate (when edge increases):",
            min_value=0.01, 
            max_value=0.2, 
            value=st.session_state.buffer_alpha_down,
            step=0.01,
            key="buffer_alpha_down_slider",
            help="Higher values make buffer rate decrease more quickly when edge increases"
        )
        
        st.session_state.buffer_alpha_up = buffer_alpha_up
        st.session_state.buffer_alpha_down = buffer_alpha_down
        
        st.markdown("##### Position Multiplier Adjustment Sensitivity")
        multiplier_alpha_up = st.slider(
            "Multiplier Increase Rate (when edge increases):",
            min_value=0.01, 
            max_value=0.2, 
            value=st.session_state.multiplier_alpha_up,
            step=0.01,
            key="multiplier_alpha_up_slider",
            help="Higher values make position multiplier increase more quickly when edge increases"
        )
        
        multiplier_alpha_down = st.slider(
            "Multiplier Decrease Rate (when edge decreases):",
            min_value=0.01, 
            max_value=0.5, 
            value=st.session_state.multiplier_alpha_down,
            step=0.01,
            key="multiplier_alpha_down_slider",
            help="Higher values make position multiplier decrease more quickly when edge decreases"
        )
        
        st.session_state.multiplier_alpha_up = multiplier_alpha_up
        st.session_state.multiplier_alpha_down = multiplier_alpha_down
    
    # Add a manual refresh button
    if st.sidebar.button("Manual Refresh", type="primary"):
        with st.spinner("Refreshing data..."):
            # Process edge data for all monitored pairs
            for pair_name in st.session_state.monitored_pairs:
                if st.session_state.pair_data.get(pair_name, {}).get('initialized', False):
                    process_edge_data(pair_name)
                    # Calculate and record fee
                    timestamp = get_sg_time()
                    calculate_and_record_fee(pair_name, timestamp)
        
        st.sidebar.success("Data refreshed successfully!")
        st.session_state.last_refresh_time = datetime.now(pytz.utc)
        
        # Reset navigation state to prevent duplicate reruns
        st.session_state.navigating = False
        st.rerun()
    
    # Display last refresh time
    last_refresh = st.session_state.last_refresh_time
    if last_refresh.tzinfo is None:
        last_refresh = last_refresh.replace(tzinfo=pytz.utc)
    singapore_last_refresh = to_singapore_time(last_refresh)
    refresh_time_str = singapore_last_refresh.strftime('%Y-%m-%d %H:%M:%S')
    st.sidebar.info(f"Last manual refresh: {refresh_time_str}")
    
    # Main content area based on view mode
    if st.session_state.view_mode == "Pairs Overview":
        # Header
        st.title("House Edge Adjustment Dashboard")
        st.markdown("Monitor and adjust trading fees based on real-time edge performance.")
        
        # Render the overview
        render_pair_overview()
    
    elif st.session_state.view_mode == "Pair Detail":
        # Check if we have a current pair selected
        if st.session_state.current_pair is None:
            st.warning("No pair selected. Please select a pair from the overview.")
            if st.button("Return to Overview"):
                st.session_state.view_mode = "Pairs Overview"
                st.rerun()
        else:
            render_pair_detail(st.session_state.current_pair)
    
    elif st.session_state.view_mode == "Pair Monitor":
        # Check if we have a current pair selected
        if st.session_state.current_pair is None:
            st.warning("No pair selected. Please select a pair from the overview.")
            if st.button("Return to Overview"):
                st.session_state.view_mode = "Pairs Overview"
                st.rerun()
        else:
            render_pair_monitor(st.session_state.current_pair)

if __name__ == "__main__":
    main()