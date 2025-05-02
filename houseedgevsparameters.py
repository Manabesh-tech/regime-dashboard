import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import math
import pytz
from sqlalchemy import create_engine, text

# Try to import auto-refresh, but don't error if not available
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    # Create a dummy function if the import fails
    def st_autorefresh(interval=0, key=None):
        pass

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
</style>
""", unsafe_allow_html=True)

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

# Fetch available trading pairs (simulated for stability)
def fetch_pairs():
    """
    Fetch all active trading pairs.
    Returns a list of pair names or default list.
    """
    # Force simulated data mode for stability
    if True:
        return ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT", "XRP/USDT"]
    
    # Original database code (disabled)
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

# Fetch current parameters for a specific pair (simulated)
def fetch_current_parameters(pair_name):
    """
    Fetch current parameters for a specific pair.
    Returns a dictionary with all parameter values needed for fee calculation.
    """
    # Force simulated data for stability
    return {
        "buffer_rate": 0.001,
        "position_multiplier": 1000,
        "max_leverage": 100,
        "rate_multiplier": 10000,
        "rate_exponent": 1,
        "pnl_base_rate": 0.0005
    }

# Calculate edge for a specific pair (simulated)
def calculate_edge(pair_name, lookback_minutes=10):
    """
    Calculate house edge for a specific pair.
    Returns simulated edge value.
    """
    # Generate random edge values for testing
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
    
    if 'auto_update' not in st.session_state:
        st.session_state.auto_update = False  # Default to disabled for stability
    
    if 'view_mode' not in st.session_state:
        st.session_state.view_mode = "Pairs Overview"  # Default view mode
    
    if 'last_refresh_time' not in st.session_state:
        st.session_state.last_refresh_time = datetime.now(pytz.utc)
    
    # Ensure next_update_time and last_update_time are aware of timezone
    current_time = datetime.now(pytz.utc)
    
    if 'next_update_time' not in st.session_state:
        st.session_state.next_update_time = current_time + timedelta(minutes=1)
    elif st.session_state.next_update_time.tzinfo is None:
        st.session_state.next_update_time = st.session_state.next_update_time.replace(tzinfo=pytz.utc)
    
    if 'last_update_time' not in st.session_state:
        st.session_state.last_update_time = current_time
    elif st.session_state.last_update_time.tzinfo is None:
        st.session_state.last_update_time = st.session_state.last_update_time.replace(tzinfo=pytz.utc)
    
    if 'last_auto_refresh' not in st.session_state:
        st.session_state.last_auto_refresh = current_time
    elif st.session_state.last_auto_refresh.tzinfo is None:
        st.session_state.last_auto_refresh = st.session_state.last_auto_refresh.replace(tzinfo=pytz.utc)
    
    if 'last_global_update' not in st.session_state:
        st.session_state.last_global_update = current_time
    elif st.session_state.last_global_update.tzinfo is None:
        st.session_state.last_global_update = st.session_state.last_global_update.replace(tzinfo=pytz.utc)
    
    if 'simulated_data_mode' not in st.session_state:
        st.session_state.simulated_data_mode = True
        
    if 'pairs_with_changes' not in st.session_state:
        st.session_state.pairs_with_changes = []
    
    # Set a flag to avoid rerun loops
    if 'rerun_counter' not in st.session_state:
        st.session_state.rerun_counter = 0

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
    """Calculate and record the fee for a 0.1% price move for the given pair."""
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
    
    # Calculate fee and percentage
    fee_amount, fee_pct = calculate_fee_for_move(
        0.1, 
        pnl_base_rate, 
        position_multiplier,
        rate_multiplier,
        rate_exponent
    )
    
    # Record in fee history
    st.session_state.pair_data[pair_name]['fee_history'].append((timestamp, fee_amount))
    
    # Update current fee values in session state
    st.session_state.pair_data[pair_name]['current_fee_amount'] = fee_amount
    st.session_state.pair_data[pair_name]['current_fee_percentage'] = fee_pct
    
    return fee_amount, fee_pct

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

# Function to batch update all monitored pairs
def batch_update_all_pairs():
    """Process edge data for all monitored pairs."""
    if not st.session_state.monitored_pairs:
        return []
    
    timestamp = datetime.now(pytz.utc)
    pairs_updated = []
    
    for pair_name in st.session_state.monitored_pairs:
        # Skip pairs that haven't been initialized
        if not st.session_state.pair_data.get(pair_name, {}).get('initialized', False):
            continue
        
        # Process edge data for this pair
        parameters_changed = process_edge_data(pair_name, timestamp)
        
        # If parameters changed, add to the list
        if parameters_changed:
            pairs_updated.append(pair_name)
    
    # Return list of pairs that need parameter updates
    return pairs_updated

# Function to update all monitored pairs on each page load
def update_pairs_on_load():
    """
    Updates all monitored pairs on each page load when needed.
    This ensures edge data is always fresh.
    """
    # Only update if we have monitored pairs and auto-update is enabled
    if not st.session_state.get('monitored_pairs', []) or not st.session_state.get('auto_update', False):
        return False
    
    # Get current time in UTC
    current_time = datetime.now(pytz.utc)
    
    # Initialize or fix last_update_time if needed
    if 'last_update_time' not in st.session_state:
        st.session_state.last_update_time = current_time
    elif st.session_state.last_update_time.tzinfo is None:
        st.session_state.last_update_time = st.session_state.last_update_time.replace(tzinfo=pytz.utc)
    
    # Calculate time since last update in minutes
    minutes_since_last_update = (current_time - st.session_state.last_update_time).total_seconds() / 60
    
    # Get current selected interval
    selected_interval = st.session_state.get('interval_radio', "1 minute")
    interval_mapping = {"1 minute": 1, "5 minutes": 5, "10 minutes": 10}
    current_lookback_mins = interval_mapping.get(selected_interval, 1)
    
    # Check if enough time has passed since last update
    # Also force update if the interval has changed
    needs_update = (minutes_since_last_update >= current_lookback_mins) or \
                    (st.session_state.get('lookback_minutes') != current_lookback_mins)
    
    if needs_update:
        print(f"Updating pairs on page load. Last update was {minutes_since_last_update:.2f} minutes ago.")
        
        # Update the session state with the current interval
        st.session_state.lookback_minutes = current_lookback_mins
        
        # Update all pairs
        pairs_updated = batch_update_all_pairs()
        
        # Update timing information
        st.session_state.last_update_time = current_time
        st.session_state.next_update_time = current_time + timedelta(minutes=current_lookback_mins)
        
        # Update fee calculations for all pairs
        timestamp = get_sg_time()
        for pair_name in st.session_state.monitored_pairs:
            if st.session_state.pair_data[pair_name].get('initialized', False):
                calculate_and_record_fee(pair_name, timestamp)
        
        # Update the last refresh time
        st.session_state.last_refresh_time = current_time
        
        # Update successful
        return True
    
    return False

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

# Function to render the pair overview cards (simplified for stability)
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
                    if st.button(f"View Details", key=f"view_{pair_name}"):
                        st.session_state.view_mode = "Pair Detail"
                        st.session_state.current_pair = pair_name
                
                with col2:
                    button_label = f"Monitor{' ⚠️' if params_changed else ''}"
                    if st.button(button_label, key=f"monitor_{pair_name}"):
                        st.session_state.view_mode = "Pair Monitor"
                        st.session_state.current_pair = pair_name

def main():
    # Initialize all session state variables to prevent duplicates
    init_session_state()
    
    # Set up auto-refresh logic - simplified for stability
    if st.session_state.auto_update:
        # Set a refresh interval based on update interval, but don't create a new refresh component
        # which could cause loops
        selected_interval = st.session_state.get('interval_radio', "1 minute")
        interval_mapping = {"1 minute": 1, "5 minutes": 5, "10 minutes": 10}
        current_lookback_mins = interval_mapping.get(selected_interval, 1)
        st.session_state.lookback_minutes = current_lookback_mins
        
        # Update next_update_time if needed
        current_time = datetime.now(pytz.utc)
        if 'next_update_time' not in st.session_state or st.session_state.next_update_time < current_time:
            st.session_state.next_update_time = current_time + timedelta(minutes=current_lookback_mins)
    
    # Update pairs on page load when needed - only if auto-update is enabled
    if st.session_state.auto_update:
        update_pairs_on_load()
    
    # Title and description
    st.markdown('<div class="header-style">House Edge Parameter Adjustment Dashboard</div>', unsafe_allow_html=True)
    st.markdown("""
    This dashboard monitors house edge and dynamically adjusts parameters to maintain exchange profitability.
    
    Parameters respond asymmetrically: buffer rate increases quickly when edge declines, while position multiplier
    uses logarithmic scaling for more balanced fee control across different parameter ranges.
    """)
    
    # Singapore time display
    singapore_tz = pytz.timezone('Asia/Singapore')
    now_utc = datetime.now(pytz.utc)
    now_sg = now_utc.astimezone(singapore_tz)
    st.markdown(f"**Current Singapore Time:** {now_sg.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create a container that will be shown on every run at the top
    update_metrics_container = st.container()
    
    # Get current time for consistent use throughout
    current_time = datetime.now(pytz.utc)
    current_time_sg = current_time.astimezone(singapore_tz)
    
    # Get current selected interval
    selected_interval = st.session_state.get('interval_radio', "1 minute")
    interval_mapping = {"1 minute": 1, "5 minutes": 5, "10 minutes": 10}
    current_lookback_mins = interval_mapping.get(selected_interval, 1)
    
    # Update session state with current interval
    st.session_state.lookback_minutes = current_lookback_mins
    
    # Ensure next_update_time is timezone-aware for comparison
    if 'next_update_time' not in st.session_state:
        next_update_time = current_time + timedelta(minutes=current_lookback_mins)
        st.session_state.next_update_time = next_update_time
    elif st.session_state.next_update_time.tzinfo is None:
        st.session_state.next_update_time = st.session_state.next_update_time.replace(tzinfo=pytz.utc)
    
    next_update_time = st.session_state.next_update_time
    next_update_time_sg = next_update_time.astimezone(singapore_tz)
    
    # Ensure last_update_time is timezone-aware for display
    if 'last_update_time' not in st.session_state:
        st.session_state.last_update_time = current_time
    elif st.session_state.last_update_time.tzinfo is None:
        st.session_state.last_update_time = st.session_state.last_update_time.replace(tzinfo=pytz.utc)
        
    last_update_time = st.session_state.last_update_time
    last_update_time_sg = last_update_time.astimezone(singapore_tz)
    
    # Display update status in the container with debug info
    with update_metrics_container:
        # Create a clean and clear status display
        st.markdown('<div class="timer-container">', unsafe_allow_html=True)
        
        # Create a more effective timer display
        st.markdown("### Update Status")
        
        status_col1, status_col2 = st.columns(2)
        
        with status_col1:
            st.markdown(f"**Last Update:** {last_update_time_sg.strftime('%H:%M:%S')} (SG)")
            st.markdown(f"**Next Update:** {next_update_time_sg.strftime('%H:%M:%S')} (SG)")
            
            # Show auto-update status with toggle
            auto_status = "Enabled" if st.session_state.get('auto_update', True) else "Disabled"
            status_color = "green" if st.session_state.get('auto_update', True) else "red"
            st.markdown(f"**Auto Update:** <span style='color:{status_color};'>{auto_status}</span>", unsafe_allow_html=True)
            
            # Show update interval
            st.markdown(f"**Update Interval:** {selected_interval}")
            
            # Show monitored pairs count with pending changes count
            pair_count = len(st.session_state.get('monitored_pairs', []))
            changes_count = len(st.session_state.get('pairs_with_changes', []))
            changes_text = f" ({changes_count} with changes)" if changes_count > 0 else ""
            st.markdown(f"**Monitored Pairs:** {pair_count}{changes_text}")
        
        with status_col2:
            # Calculate time to next update
            time_to_next = max(0, (next_update_time - current_time).total_seconds())
            minutes_to_next = int(time_to_next // 60)
            seconds_to_next = int(time_to_next % 60)
            
            # Show countdown with visual indicator
            if st.session_state.get('auto_update', True):
                st.markdown(f"**Next Auto-Update In:**")
                # Add a visual countdown progress bar
                progress_value = 1.0 - (time_to_next / (current_lookback_mins * 60))
                progress_value = max(0.0, min(1.0, progress_value))  # Ensure value is between 0 and 1
                st.progress(progress_value, text=f"{minutes_to_next}m {seconds_to_next}s")
            else:
                st.markdown(f"**Auto-Updates Disabled**")
                st.markdown("Use manual update button below")
            
            # Manual update button - always available
            if st.button("Update Now", key="force_update", type="primary"):
                # Force an immediate update
                with st.spinner("Forcing immediate update..."):
                    pairs_updated = batch_update_all_pairs()
                    
                    # Update timestamps
                    new_time = datetime.now(pytz.utc)
                    st.session_state.last_update_time = new_time
                    st.session_state.next_update_time = new_time + timedelta(minutes=current_lookback_mins)
                    
                    # Update fee calculations
                    timestamp = get_sg_time()
                    for pair_name in st.session_state.monitored_pairs:
                        if st.session_state.pair_data[pair_name].get('initialized', False):
                            calculate_and_record_fee(pair_name, timestamp)
                    
                    # Record update information
                    st.session_state['last_update_info'] = {
                        'time': new_time,
                        'pairs_updated': len(pairs_updated),
                        'trigger': 'manual'
                    }
        
        # Close timer container
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Status message based on auto-update
        if st.session_state.get('auto_update', True) and pair_count > 0:
            st.success(f"Auto-update is enabled with {selected_interval} interval. Next update at {next_update_time_sg.strftime('%H:%M:%S')} (SG time).")
        elif not st.session_state.get('auto_update', True):
            st.warning("Auto-update is disabled. Use the 'Update Now' button for manual updates.")
        elif pair_count == 0:
            st.info("Add trading pairs to begin monitoring.")
    
    # Sidebar controls
    st.sidebar.markdown('<div class="subheader-style">Trading Pair Configuration</div>', unsafe_allow_html=True)
    
    # Fetch available pairs (simplified for stability)
    pairs = fetch_pairs()
    
    # Select pair for initialization
    selected_pair = st.sidebar.selectbox(
        "Select Trading Pair",
        options=pairs,
        index=0,
        key="pair_selector"
    )
    
    # Update interval options
    update_interval_options = {
        "1 minute": 1,
        "5 minutes": 5,
        "10 minutes": 10
    }
    
    # Select update interval
    update_interval_selection = st.sidebar.radio(
        "Update Interval",
        options=list(update_interval_options.keys()),
        index=0,  # Default to 1 minute
        key="interval_radio"
    )
    
    # Initialize button and Add All Pairs button in a row
    init_col1, init_col2 = st.sidebar.columns(2)
    
    with init_col1:
        initialize_button = st.button(
            "Add Pair", 
            help=f"Add the selected pair {selected_pair} to monitoring",
            type="primary",
            key="init_button"
        )
    
    with init_col2:
        add_all_button = st.button(
            "Add All Pairs",
            help="Initialize and monitor all available trading pairs",
            type="secondary",
            key="add_all_button"
        )
    
    # Handle button actions
    if initialize_button:
        try:
            # Initialize pair directly inside this block
            pair_name = selected_pair
            
            # Initialize the pair with all parameters
            success = initialize_pair(pair_name)
            
            if success:
                # Add to monitored pairs if not already there
                if pair_name not in st.session_state.monitored_pairs:
                    st.session_state.monitored_pairs.append(pair_name)
                
                # Set as current pair
                st.session_state.current_pair = pair_name
                
                # Success message and reset timers
                st.sidebar.success(f"Started monitoring for {pair_name}")
                st.session_state.view_mode = "Pairs Overview"
                
                # Reset auto-update timer when adding a new pair
                current_time = datetime.now(pytz.utc)
                st.session_state.last_update_time = current_time
                st.session_state.next_update_time = current_time + timedelta(minutes=st.session_state.lookback_minutes)
            else:
                st.sidebar.error(f"Failed to initialize {pair_name}")
                
        except Exception as e:
            st.sidebar.error(f"Error adding pair: {str(e)}")
    
    if add_all_button:
        try:
            # Filter out pairs that are already being monitored
            unmonitored_pairs = [pair for pair in pairs if pair not in st.session_state.monitored_pairs]
            
            if not unmonitored_pairs:
                st.sidebar.info("All available pairs are already being monitored.")
            else:
                # Show a progress bar during initialization
                with st.sidebar:
                    progress_bar = st.progress(0)
                    total_pairs = len(unmonitored_pairs)
                    pairs_added = 0
                    
                    for i, p in enumerate(unmonitored_pairs):
                        try:
                            # Update progress
                            progress = (i+1) / total_pairs
                            progress_bar.progress(progress, text=f"Initializing {p}...")
                            
                            # Initialize the pair
                            success = initialize_pair(p)
                            
                            if success:
                                # Add to monitored pairs
                                if p not in st.session_state.monitored_pairs:
                                    st.session_state.monitored_pairs.append(p)
                                pairs_added += 1
                                
                        except Exception as e:
                            st.error(f"Error initializing {p}: {str(e)}")
                    
                    # Complete progress
                    progress_bar.progress(1.0, text="All pairs initialized!")
                    
                # Reset auto-update timer after adding all pairs
                current_time = datetime.now(pytz.utc)
                st.session_state.last_update_time = current_time
                st.session_state.next_update_time = current_time + timedelta(minutes=st.session_state.lookback_minutes)
                
                st.sidebar.success(f"Added {pairs_added} new pairs to monitoring")
                st.session_state.view_mode = "Pairs Overview"
        except Exception as e:
            st.sidebar.error(f"Error adding pairs: {str(e)}")
    
    # Batch operations section
    st.sidebar.markdown('<div class="subheader-style">Batch Operations</div>', unsafe_allow_html=True)
    
    # Batch update button
    if st.sidebar.button(
        "Update All Pairs Now", 
        help="Immediately fetch new edge data for all monitored pairs",
        key="batch_update_button",
        disabled=len(st.session_state.monitored_pairs) == 0
    ):
        # Perform the immediate update
        with st.spinner("Updating all pairs..."):
            current_time = datetime.now(pytz.utc)
            pairs_updated = batch_update_all_pairs()
            
            # Update timestamps
            st.session_state.last_update_time = current_time
            st.session_state.next_update_time = current_time + timedelta(minutes=st.session_state.lookback_minutes)
            
            # Update fee calculations
            timestamp = get_sg_time()
            for pair_name in st.session_state.monitored_pairs:
                if st.session_state.pair_data[pair_name].get('initialized', False):
                    calculate_and_record_fee(pair_name, timestamp)
            
            if pairs_updated:
                st.sidebar.warning(f"Parameter updates available for {len(pairs_updated)} pairs")
            else:
                st.sidebar.success("All pairs updated, no parameter changes needed")
    
    # Global apply changes button (only shown if changes are available)
    if st.session_state.pairs_with_changes:
        if st.sidebar.button(
            f"Apply All Changes ({len(st.session_state.pairs_with_changes)})", 
            help="Apply all recommended parameter changes across all pairs",
            key="global_apply_button",
            type="primary",
            disabled=len(st.session_state.pairs_with_changes) == 0
        ):
            with st.spinner("Applying parameter updates to all pairs..."):
                applied_count = 0
                for pair_with_changes in st.session_state.pairs_with_changes.copy():
                    success, old_buffer, new_buffer, old_multiplier, new_multiplier = update_display_parameters(pair_with_changes)
                    if success:
                        applied_count += 1
            
            if applied_count > 0:
                st.sidebar.success(f"Successfully applied updates to {applied_count} pairs")
            else:
                st.sidebar.info("No changes were applied")
    
    # Sensitivity parameters section
    st.sidebar.markdown('<div class="subheader-style">Sensitivity Parameters</div>', unsafe_allow_html=True)
    
    # Buffer rate increase sensitivity
    st.session_state.buffer_alpha_up = st.sidebar.slider(
        "Buffer Rate Increase Sensitivity",
        min_value=0.01, 
        max_value=0.5, 
        value=st.session_state.buffer_alpha_up,
        step=0.01,
        help="How quickly buffer rate increases when edge declines",
        key="buffer_up_slider"
    )
    
    # Buffer rate decrease sensitivity
    st.session_state.buffer_alpha_down = st.sidebar.slider(
        "Buffer Rate Decrease Sensitivity",
        min_value=0.001, 
        max_value=0.1, 
        value=st.session_state.buffer_alpha_down,
        step=0.001,
        help="How quickly buffer rate decreases when edge improves",
        key="buffer_down_slider"
    )
    
    # Position multiplier increase sensitivity
    st.session_state.multiplier_alpha_up = st.sidebar.slider(
        "Position Multiplier Increase Sensitivity",
        min_value=0.001, 
        max_value=0.1, 
        value=st.session_state.multiplier_alpha_up,
        step=0.001,
        help="How quickly position multiplier increases when edge improves",
        key="multiplier_up_slider"
    )
    
    # Position multiplier decrease sensitivity
    st.session_state.multiplier_alpha_down = st.sidebar.slider(
        "Position Multiplier Decrease Sensitivity",
        min_value=0.01, 
        max_value=0.5, 
        value=st.session_state.multiplier_alpha_down,
        step=0.01,
        help="How quickly position multiplier decreases when edge declines",
        key="multiplier_down_slider"
    )
    
    # History length slider
    st.session_state.history_length = st.sidebar.slider(
        "History Length (points)", 
        min_value=10, 
        max_value=1000, 
        value=st.session_state.history_length,
        help="Number of data points to keep in history",
        key="history_length_slider"
    )
    
    # Debug section for advanced users
    with st.sidebar.expander("Debug", expanded=False):
        st.write("Simulated data mode:", st.session_state.simulated_data_mode)
        st.write("Last update time:", format_time_display(st.session_state.last_update_time))
        st.write("Next update time:", format_time_display(st.session_state.next_update_time))
        st.write("View mode:", st.session_state.view_mode)
        st.write("Pairs with changes:", len(st.session_state.pairs_with_changes))
        
        # Force simulated data mode button
        if st.button("Toggle Simulated Data Mode", key="toggle_sim_mode"):
            st.session_state.simulated_data_mode = not st.session_state.simulated_data_mode
            st.write("Simulated data mode:", st.session_state.simulated_data_mode)
    
    # Main content area
    # Render different views based on current mode
    if st.session_state.view_mode == "Pairs Overview":
        render_pair_overview()
    elif st.session_state.view_mode == "Pair Detail" and st.session_state.current_pair is not None:
        # For now, just show a placeholder to prevent errors
        st.subheader(f"Detailed Analytics: {st.session_state.current_pair}")
        st.write("Detailed view is temporarily disabled in this simplified version.")
        
        # Button to return to overview
        if st.button("Return to Pairs Overview", type="secondary"):
            st.session_state.view_mode = "Pairs Overview"
            st.session_state.current_pair = None
    elif st.session_state.view_mode == "Pair Monitor" and st.session_state.current_pair is not None:
        # For now, just show a placeholder to prevent errors
        st.subheader(f"Parameter Monitoring: {st.session_state.current_pair}")
        st.write("Monitor view is temporarily disabled in this simplified version.")
        
        pair_name = st.session_state.current_pair
        if st.session_state.pair_data[pair_name].get('params_changed', False):
            st.markdown('<div class="warning">Parameter updates available.</div>', unsafe_allow_html=True)
            
            # Update display parameters button
            update_button = st.button("Apply Changes", type="primary", key=f"update_params_{pair_name}")
            
            if update_button:
                success, old_buffer, new_buffer, old_multiplier, new_multiplier = update_display_parameters(pair_name)
                if success:
                    st.success("Parameters updated successfully!")
        else:
            st.info("No parameter changes needed at this time.")
            
        # Button to return to overview
        if st.button("Return to Pairs Overview", type="secondary"):
            st.session_state.view_mode = "Pairs Overview"
            st.session_state.current_pair = None
    else:
        # Default to overview if mode is invalid
        st.session_state.view_mode = "Pairs Overview"
        render_pair_overview()
    
    # Prune history for all pairs if needed
    for pair_name in st.session_state.monitored_pairs:
        if pair_name in st.session_state.pair_data:
            pair_data = st.session_state.pair_data[pair_name]
            
            if len(pair_data.get('edge_history', [])) > st.session_state.history_length:
                pair_data['edge_history'] = pair_data['edge_history'][-st.session_state.history_length:]
            
            if len(pair_data.get('buffer_history', [])) > st.session_state.history_length:
                pair_data['buffer_history'] = pair_data['buffer_history'][-st.session_state.history_length:]
            
            if len(pair_data.get('multiplier_history', [])) > st.session_state.history_length:
                pair_data['multiplier_history'] = pair_data['multiplier_history'][-st.session_state.history_length:]
            
            if len(pair_data.get('fee_history', [])) > st.session_state.history_length:
                pair_data['fee_history'] = pair_data['fee_history'][-st.session_state.history_length:]

# Main entry point
if __name__ == "__main__":
    # Run the main application
    main()