import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import math
import pytz
from sqlalchemy import create_engine, text
from streamlit_autorefresh import st_autorefresh  # Import the auto-refresh component

# Page configuration
st.set_page_config(
    page_title="House Edge Adjustment Dashboard",
    page_icon="📊",
    layout="wide"
)

# Generate a unique refresh key based on the current time
refresh_key = f"autorefresh_{int(time.time())}"

# Initialize auto-refresh based on lookback minutes
def setup_auto_refresh():
    """Set up auto-refresh with the current settings"""
    if st.session_state.get('auto_update', True):
        # Get current value directly from the radio button
        selected_interval = st.session_state.get('interval_radio', "1 minute")
        interval_mapping = {"1 minute": 1, "5 minutes": 5, "10 minutes": 10}
        lookback_mins = interval_mapping.get(selected_interval, 1)
        
        # Update the session state
        st.session_state.lookback_minutes = lookback_mins
        
        # Calculate refresh rate in milliseconds (slightly shorter than full interval)
        refresh_rate = int(lookback_mins * 60 * 1000 * 0.9)
        
        # Setup the auto-refresh with a unique key to force refreshing
        st_autorefresh(interval=refresh_rate, key=refresh_key)
        
        # Reset timers
        current_time = datetime.now(pytz.utc)
        st.session_state.last_update_time = current_time
        st.session_state.next_update_time = current_time + timedelta(minutes=lookback_mins)
        
        print(f"Auto-refresh enabled with {refresh_rate}ms interval, next update in {lookback_mins} minutes")
        return True
    else:
        print("Auto-refresh disabled")
        return False

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
</style>
""", unsafe_allow_html=True)

# Initialize database connection
@st.cache_resource
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

# Fetch available trading pairs from database
@st.cache_data(ttl=60)  # Reduced from 600 to 60 seconds to refresh pairs more often
def fetch_pairs():
    """
    Fetch all active trading pairs from the database.
    Returns a list of pair names or default list if query fails.
    """
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
@st.cache_data(ttl=60)  # Reduced from 300 to 60 seconds to refresh parameters more often
def fetch_current_parameters(pair_name):
    """
    Fetch current parameters for a specific pair.
    Returns a dictionary with all parameter values needed for fee calculation.
    """
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
# Remove the cache to ensure fresh data every time
def calculate_edge(pair_name, lookback_minutes=10):
    """
    Calculate house edge for a specific pair using the SQL query.
    Returns edge value or None if calculation fails.
    """
    # Check if we're in simulated data mode
    if st.session_state.get('simulated_data_mode', False):
        # Generate random edge values for testing
        import random
        return 0.001 + random.uniform(-0.0005, 0.0010)
    
    engine = init_connection()
    if not engine:
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
        
        # Add debug logging
        print(f"Calculating edge for {pair_name} from {start_time_utc} to {end_time_utc}")
        
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
        
        # Add a debug log
        st.session_state.last_query_time = datetime.now(pytz.utc)
        
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
        
        print(f"Edge value for {pair_name}: {edge_value}")
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
    
    The formula calculates P_close as:
    P_close = initial_price + ((1 - pnl_base_rate) / (1 + (1/(abs(price_ratio - 1) * rate_multiplier))^rate_exponent 
                + (bet*leverage)/(10^6 * abs(price_ratio - 1) * position_multiplier))) * (price_after_move - initial_price)
    
    Where price_ratio = price_after_move/initial_price
    
    Args:
        move_pct: Price move percentage (positive for price increase, negative for decrease)
        buffer_rate: The base rate parameter (equivalent to pnl_base_rate in some documentation)
        position_multiplier: The position multiplier parameter
        rate_multiplier: Rate multiplier (default 15000)
        rate_exponent: Rate exponent (default 1)
        bet: Bet amount (default 1.0)
        leverage: Position leverage (default 1.0)
        debug: Whether to return additional debug information
        
    Returns:
        If debug=False: A tuple of (fee_amount, fee_percentage)
        If debug=True: A tuple of (fee_amount, fee_percentage, debug_info)
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

# Function to update buffer rate based on edge comparison with improved scaling
def update_buffer_rate(current_buffer, edge, edge_ref, max_leverage, alpha_up=0.1, alpha_down=0.02):
    """
    Update buffer_rate based on edge comparison with proper scaling.
    Increases sharply when edge declines, decreases slowly when edge improves.
    """
    delta = edge - edge_ref
    upper_bound = 0.7 / max_leverage
    lower_bound = 0.0001
    
    # Calculate a normalized delta relative to the reference edge
    # This scales the adjustment to the magnitude of the reference edge
    if edge_ref != 0:
        normalized_delta = delta / abs(edge_ref)  # Scale relative to reference
    else:
        normalized_delta = delta  # Fallback if reference is zero
    
    # Apply caps to avoid extreme adjustments
    normalized_delta = max(min(normalized_delta, 1.0), -1.0)
    
    # Asymmetric adjustment: fast up, slow down
    adjustment = alpha_up * abs(normalized_delta) * current_buffer if normalized_delta < 0 else -alpha_down * normalized_delta * current_buffer
    
    return max(lower_bound, min(upper_bound, current_buffer + adjustment))

# Function to update position multiplier based on edge comparison with improved scaling
def update_position_multiplier(current_multiplier, edge, edge_ref, alpha_up=0.02, alpha_down=0.1):
    """
    Update position_multiplier based on edge comparison with proper scaling.
    Decreases sharply when edge declines, increases slowly when edge improves.
    """
    delta = edge - edge_ref
    upper_bound = 14000
    lower_bound = 1
    
    # Calculate a normalized delta relative to the reference edge
    # This scales the adjustment to the magnitude of the reference edge
    if edge_ref != 0:
        normalized_delta = delta / abs(edge_ref)  # Scale relative to reference
    else:
        normalized_delta = delta  # Fallback if reference is zero
    
    # Apply caps to avoid extreme adjustments
    normalized_delta = max(min(normalized_delta, 1.0), -1.0)
    
    # Asymmetric adjustment: fast down, slow up
    adjustment = -alpha_down * abs(normalized_delta) * current_multiplier if normalized_delta < 0 else alpha_up * normalized_delta * current_multiplier
    
    return max(lower_bound, min(upper_bound, current_multiplier + adjustment))

# New function to get status of a pair
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
            'pnl_base_rate':0.1,
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

# Initialize session state variables if they don't exist already
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
        st.session_state.auto_update = True  # Enable auto-update by default
    
    if 'view_mode' not in st.session_state:
        st.session_state.view_mode = "Pairs Overview"  # Default view mode
    
    # Ensure next_update_time and last_update_time are aware of timezone
    if 'next_update_time' not in st.session_state:
        st.session_state.next_update_time = datetime.now(pytz.utc) + timedelta(minutes=1)
    elif st.session_state.next_update_time.tzinfo is None:
        st.session_state.next_update_time = st.session_state.next_update_time.replace(tzinfo=pytz.utc)
    
    if 'last_update_time' not in st.session_state:
        st.session_state.last_update_time = datetime.now(pytz.utc)
    elif st.session_state.last_update_time.tzinfo is None:
        st.session_state.last_update_time = st.session_state.last_update_time.replace(tzinfo=pytz.utc)
    
    if 'last_auto_refresh' not in st.session_state:
        st.session_state.last_auto_refresh = datetime.now(pytz.utc)
    elif st.session_state.last_auto_refresh.tzinfo is None:
        st.session_state.last_auto_refresh = st.session_state.last_auto_refresh.replace(tzinfo=pytz.utc)
    
    if 'last_global_update' not in st.session_state:
        st.session_state.last_global_update = datetime.now(pytz.utc)
    elif st.session_state.last_global_update.tzinfo is None:
        st.session_state.last_global_update = st.session_state.last_global_update.replace(tzinfo=pytz.utc)
    
    if 'simulated_data_mode' not in st.session_state:
        st.session_state.simulated_data_mode = False

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
    buffer_rate = st.session_state.pair_data[pair_name]['buffer_rate']
    position_multiplier = st.session_state.pair_data[pair_name]['position_multiplier']
    pnl_base_rate = st.session_state.pair_data[pair_name].get('pnl_base_rate', 0.1)  # Get base_rate with default
    
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
        rate_exponent, 
        
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
        
        # Log the edge update to aid debugging
        old_edge = st.session_state.pair_data[pair_name]['current_edge']
        edge_change = 0 if old_edge is None else ((new_edge - old_edge) / old_edge * 100 if old_edge != 0 else 0)
        
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
        # ADD THESE THREE LINES HERE:
        # Update reference parameters to match the new values
        st.session_state.pair_data[pair_name]['reference_buffer_rate'] = st.session_state.pair_data[pair_name]['buffer_rate']
        st.session_state.pair_data[pair_name]['reference_position_multiplier'] = st.session_state.pair_data[pair_name]['position_multiplier']
        st.session_state.pair_data[pair_name]['reference_edge'] = st.session_state.pair_data[pair_name]['current_edge']
        # Mark that parameters have been changed
        st.session_state.pair_data[pair_name]['params_changed'] = False
        
        # Reset proposed values
        st.session_state.pair_data[pair_name]['proposed_buffer_rate'] = None
        st.session_state.pair_data[pair_name]['proposed_position_multiplier'] = None
        
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

# Function to create fee curve plot for a specific pair
def create_fee_curve_plot(pair_name):
    """Create two plots: fee amount vs price move and fee percentage vs price move."""
    # Ensure pair state is initialized
    init_pair_state(pair_name)
    
    # Get required parameters
    buffer_rate = st.session_state.pair_data[pair_name]['buffer_rate']
    position_multiplier = st.session_state.pair_data[pair_name]['position_multiplier']
    rate_multiplier = st.session_state.pair_data[pair_name].get('rate_multiplier', 10000)
    rate_exponent = st.session_state.pair_data[pair_name].get('rate_exponent', 1)
    
    # Calculate fee across a range of move sizes
    move_sizes = np.linspace(0, 1, 101)  # Focus on positive moves only (where fees apply)
    current_fees = []
    current_fee_pcts = []
    
    for move in move_sizes:
        fee_amount, fee_pct = calculate_fee_for_move(
            move, 
            buffer_rate, 
            position_multiplier,
            rate_multiplier,
            rate_exponent
        )
        current_fees.append(fee_amount)
        current_fee_pcts.append(fee_pct)
    
    # Create figure and axis for fee amount
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot current fee curve
    ax1.plot(move_sizes, current_fees, 'b-', label='Current Fee Amount')
    
    # Plot proposed fee curve if available
    if (st.session_state.pair_data[pair_name]['proposed_buffer_rate'] is not None and 
        st.session_state.pair_data[pair_name]['proposed_position_multiplier'] is not None):
        
        proposed_buffer = st.session_state.pair_data[pair_name]['proposed_buffer_rate']
        proposed_multiplier = st.session_state.pair_data[pair_name]['proposed_position_multiplier']
        
        proposed_fees = []
        proposed_fee_pcts = []
        
        for move in move_sizes:
            fee_amount, fee_pct = calculate_fee_for_move(
                move, 
                proposed_buffer, 
                proposed_multiplier,
                rate_multiplier,
                rate_exponent
            )
            proposed_fees.append(fee_amount)
            proposed_fee_pcts.append(fee_pct)
        
        ax1.plot(move_sizes, proposed_fees, 'r--', label='Proposed Fee Amount')
    
    # Set title and labels
    ax1.set_title(f'Fee Amount vs. Price Move Size - {pair_name}')
    ax1.set_xlabel('Price Move (%)')
    ax1.set_ylabel('Fee Amount')
    
    # Add grid
    ax1.grid(True, alpha=0.3)
    
    # Add legend
    ax1.legend()
    
    # Tight layout
    plt.tight_layout()
    
    # Create figure and axis for fee percentage
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    # Plot current fee percentage curve
    ax2.plot(move_sizes, current_fee_pcts, 'g-', label='Current Fee Percentage')
    
    # Plot proposed fee percentage curve if available
    if (st.session_state.pair_data[pair_name]['proposed_buffer_rate'] is not None and 
        st.session_state.pair_data[pair_name]['proposed_position_multiplier'] is not None):
        ax2.plot(move_sizes, proposed_fee_pcts, 'r--', label='Proposed Fee Percentage')
    
    # Set title and labels
    ax2.set_title(f'Fee Percentage vs. Price Move Size - {pair_name}')
    ax2.set_xlabel('Price Move (%)')
    ax2.set_ylabel('Fee (% of Profit)')
    
    # Add grid
    ax2.grid(True, alpha=0.3)
    
    # Add legend
    ax2.legend()
    
    # Tight layout
    plt.tight_layout()
    
    return fig1, fig2

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
    
    # Calculate fees with current parameters
    current_fees = []
    current_fee_pcts = []
    
    for move in move_sizes:
        fee_amount, fee_pct = calculate_fee_for_move(
            move, 
            current_buffer, 
            current_multiplier,
            rate_multiplier,
            rate_exponent
        )
        current_fees.append(fee_amount)
        current_fee_pcts.append(fee_pct)
    
    # Calculate fees with proposed parameters if available
    if (st.session_state.pair_data[pair_name]['proposed_buffer_rate'] is not None and 
        st.session_state.pair_data[pair_name]['proposed_position_multiplier'] is not None):
        
        proposed_buffer = st.session_state.pair_data[pair_name]['proposed_buffer_rate']
        proposed_multiplier = st.session_state.pair_data[pair_name]['proposed_position_multiplier']
        
        proposed_fees = []
        proposed_fee_pcts = []
        
        for move in move_sizes:
            fee_amount, fee_pct = calculate_fee_for_move(
                move, 
                proposed_buffer, 
                proposed_multiplier,
                rate_multiplier,
                rate_exponent
            )
            proposed_fees.append(fee_amount)
            proposed_fee_pcts.append(fee_pct)
        
        # Create dataframe for the table with both current and proposed
        fee_df = pd.DataFrame({
            'Move Size (%)': move_sizes,
            'Current Fee Amount': current_fees,
            'Current Fee (%)': current_fee_pcts,
            'Proposed Fee Amount': proposed_fees,
            'Proposed Fee (%)': proposed_fee_pcts,
            'Fee % Change': [(new - old) / old * 100 if old != 0 else float('inf') 
                              for new, old in zip(proposed_fee_pcts, current_fee_pcts)]
        })
    else:
        # Create dataframe with just current fees
        fee_df = pd.DataFrame({
            'Move Size (%)': move_sizes,
            'Current Fee Amount': current_fees,
            'Current Fee (%)': current_fee_pcts
        })
    
    return fee_df

# Function to render the pair overview cards
def render_pair_overview():
    """Render overview cards for all monitored pairs."""
    if not st.session_state.monitored_pairs:
        st.info("No pairs are currently being monitored. Select a pair and click 'Add Pair' in the sidebar.")
        return
    
    st.markdown("### Monitored Trading Pairs")
    st.markdown("Select a pair below to view detailed analytics or click 'Monitor' to update parameters.")
    
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
                st.markdown(f"""
                <div style="background-color: #f8f9fa; border-radius: 10px; padding: 15px; margin-bottom: 15px; border: 1px solid #e0e0e0;">
                    <div style="font-weight: bold; font-size: 18px; margin-bottom: 10px;">
                        <span style="display: inline-block; width: 12px; height: 12px; border-radius: 50%; background-color: {status_color}; margin-right: 8px;"></span>
                        {pair_name}
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
                        st.rerun()
                with col2:
                    button_label = f"Monitor{' ⚠️' if params_changed else ''}"
                    if st.button(button_label, key=f"monitor_{pair_name}"):
                        st.session_state.view_mode = "Pair Monitor"
                        st.session_state.current_pair = pair_name
                        st.rerun()

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
    detail_tabs = st.tabs(["Edge History", "Parameter History", "Fee Analysis"])
    
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
        
        # Create fee curve plots
        fee_amount_fig, fee_percentage_fig = create_fee_curve_plot(pair_name)
        if fee_amount_fig is not None:
            st.markdown("#### Fee Amount vs Price Move")
            st.pyplot(fee_amount_fig)
        
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
    
    # Button to return to overview
    if st.button("Return to Pairs Overview", type="secondary"):
        st.session_state.view_mode = "Pairs Overview"
        st.session_state.current_pair = None
        st.rerun()

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
    
    # Add a row of action buttons
    hist_col1, hist_col2, hist_col3 = st.columns(3)
    
    with hist_col1:
        if st.button("View Edge History", key=f"view_edge_history_{pair_name}"):
            if len(st.session_state.pair_data[pair_name]['edge_history']) > 0:
                edge_df = pd.DataFrame({
                    'Timestamp': [t for t, _ in st.session_state.pair_data[pair_name]['edge_history']],
                    'Edge Value': [f"{e:.6f}" for _, e in st.session_state.pair_data[pair_name]['edge_history']]
                })
                st.dataframe(edge_df, hide_index=True, use_container_width=True)
            else:
                st.info("No edge history data yet.")
                
    with hist_col2:
        if st.button("View Buffer Rate History", key=f"view_buffer_history_{pair_name}"):
            if len(st.session_state.pair_data[pair_name]['buffer_history']) > 0:
                buffer_df = pd.DataFrame({
                    'Timestamp': [t for t, _ in st.session_state.pair_data[pair_name]['buffer_history']],
                    'Buffer Rate': [f"{r:.6f}" for _, r in st.session_state.pair_data[pair_name]['buffer_history']]
                })
                st.dataframe(buffer_df, hide_index=True, use_container_width=True)
            else:
                st.info("No buffer rate history data yet.")
                
    with hist_col3:
        if st.button("View Multiplier History", key=f"view_multiplier_history_{pair_name}"):
            if len(st.session_state.pair_data[pair_name]['multiplier_history']) > 0:
                multiplier_df = pd.DataFrame({
                    'Timestamp': [t for t, _ in st.session_state.pair_data[pair_name]['multiplier_history']],
                    'Position Multiplier': [f"{m:.1f}" for _, m in st.session_state.pair_data[pair_name]['multiplier_history']]
                })
                st.dataframe(multiplier_df, hide_index=True, use_container_width=True)
            else:
                st.info("No position multiplier history data yet.")
    
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
                st.session_state.pair_data[pair_name]['buffer_rate'], 
                st.session_state.pair_data[pair_name]['position_multiplier'],
                rate_multiplier,
                rate_exponent
            )
            
            new_fee_amount, new_fee_pct = calculate_fee_for_move(
                0.1, 
                st.session_state.pair_data[pair_name]['proposed_buffer_rate'], 
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
            
            # Show proposed changes
            st.markdown("### Proposed Parameter Changes")
            
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
            
            # Add a view of fee changes for different move sizes
            with st.expander("View Fee Comparison for Different Move Sizes"):
                fee_df = create_fee_comparison_table(pair_name)
                st.dataframe(fee_df, use_container_width=True)
        
        # Update display parameters button
        st.markdown('<div class="button-row">', unsafe_allow_html=True)
        update_button = st.button("Update Parameters", type="primary", key=f"update_params_{pair_name}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if update_button:
            # Apply the updates to display (not database)
            success, old_buffer, new_buffer, old_multiplier, new_multiplier = update_display_parameters(pair_name)
            
            if success:
                st.markdown('<div class="success">Display parameters updated successfully!</div>', unsafe_allow_html=True)
                st.rerun()
    else:
        # Manual update button
        st.markdown('<div class="button-row">', unsafe_allow_html=True)
        if st.button("Fetch New Data", key=f"fetch_data_{pair_name}"):
            process_edge_data(pair_name)
            
            # Calculate fee with updated edge data
            timestamp = get_sg_time()
            calculate_and_record_fee(pair_name, timestamp)
            
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Reset to reference parameters button
    st.markdown('<div class="button-row">', unsafe_allow_html=True)
    if st.button("Reset to Reference Parameters", type="secondary", key=f"reset_params_{pair_name}"):
        success, old_buffer, new_buffer, old_multiplier, new_multiplier = reset_to_reference_parameters(pair_name)
        if success:
            st.success("Parameters reset to reference values")
            st.rerun()
        else:
            st.error("No reference parameters available")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Button to return to overview
    if st.button("Return to Pairs Overview", type="secondary", key=f"return_from_monitor_{pair_name}"):
        st.session_state.view_mode = "Pairs Overview"
        st.session_state.current_pair = None
        st.rerun()

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

# Handle simulated data mode for testing without database
def enable_simulated_data_mode():
    """Enable simulated data mode for testing without database connection"""
    st.session_state['simulated_data_mode'] = True
    print("Simulated data mode enabled")

# Function to update all monitored pairs on each page load
def update_pairs_on_load():
    """
    Updates all monitored pairs on each page load when needed.
    This ensures edge data is always fresh.
    """
    # Only update if we have monitored pairs
    if not st.session_state.get('monitored_pairs', []):
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
    
    # Get current selected interval (always use current value, not stored value)
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
        
        # Update successful
        return True
    
    return False

def main():
    # Initialize all session state variables to prevent duplicates
    init_session_state()
    
    # Set up auto-refresh before anything else
    if st.session_state.get('auto_update', True):
        setup_auto_refresh()
    
    # If auto-update is enabled, update pairs on page load when needed
    if st.session_state.get('auto_update', True):
        update_pairs_on_load()
    
    # Title and description
    st.markdown('<div class="header-style">House Edge Parameter Adjustment Dashboard</div>', unsafe_allow_html=True)
    st.markdown("""
    This dashboard monitors house edge and dynamically adjusts buffer rate and position multiplier parameters 
    to maintain exchange profitability.
    
    Parameters respond asymmetrically: buffer rate increases quickly when edge declines and decreases slowly when edge improves, 
    while position multiplier does the opposite.
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
        st.markdown("### Update Status")
        
        status_col1, status_col2 = st.columns(2)
        
        with status_col1:
            st.markdown(f"**Last Update:** {last_update_time_sg.strftime('%H:%M:%S')} (SG)")
            st.markdown(f"**Next Update:** {next_update_time_sg.strftime('%H:%M:%S')} (SG)")
            
            # Show auto-update status
            auto_status = "Enabled" if st.session_state.get('auto_update', True) else "Disabled"
            st.markdown(f"**Auto Update:** {auto_status}")
            
            # Show update interval
            st.markdown(f"**Update Interval:** {selected_interval}")
            
            # Show monitored pairs count
            pair_count = len(st.session_state.get('monitored_pairs', []))
            st.markdown(f"**Monitored Pairs:** {pair_count}")
        
        with status_col2:
            # Calculate time to next update
            time_to_next = max(0, (next_update_time - current_time).total_seconds())
            minutes_to_next = int(time_to_next // 60)
            seconds_to_next = int(time_to_next % 60)
            
            # Show countdown
            st.markdown(f"**Time to Next Update:** {minutes_to_next}m {seconds_to_next}s")
            
            # Manual update button
            if st.button("Update Now", key="force_update"):
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
                    
                    # Rerun to refresh
                    st.rerun()
            
            # Add a visual countdown progress bar
            if time_to_next > 0:
                progress_value = 1.0 - (time_to_next / (current_lookback_mins * 60))
                progress_value = max(0.0, min(1.0, progress_value))  # Ensure value is between 0 and 1
                st.progress(progress_value, text=f"Next update in {minutes_to_next}m {seconds_to_next}s")
        
        # Status message based on auto-update
        if st.session_state.get('auto_update', True) and pair_count > 0:
            st.success(f"Auto-update is enabled with {selected_interval} interval. Next update at {next_update_time_sg.strftime('%H:%M:%S')} (SG time).")
        elif not st.session_state.get('auto_update', True):
            st.warning("Auto-update is disabled. Use the 'Update Now' button for manual updates.")
        elif pair_count == 0:
            st.info("Add trading pairs to begin monitoring.")
    
    # Sidebar controls
    st.sidebar.markdown('<div class="subheader-style">Trading Pair Configuration</div>', unsafe_allow_html=True)
    
    # Fetch available pairs
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
        key="interval_radio",
        on_change=lambda: setup_auto_refresh()  # Setup auto-refresh when interval changes
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
                
                st.rerun()
            else:
                st.sidebar.error(f"Failed to initialize {pair_name}")
                
        except Exception as e:
            st.sidebar.error(f"Error adding pair: {str(e)}")
            import traceback
            st.sidebar.error(traceback.format_exc())  # Display full traceback
    
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
                st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error adding pairs: {str(e)}")
            import traceback
            st.sidebar.error(traceback.format_exc())  # Display full traceback
    
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
        
        st.rerun()
    
    # Auto-update toggle
    st.session_state.auto_update = st.sidebar.checkbox(
        "Enable Auto-Update", 
        value=st.session_state.get('auto_update', True),
        help="Automatically update data at the selected interval",
        key="auto_update_checkbox"
    )
    
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
    
    # Main content area
    # Render different views based on current mode
    if st.session_state.view_mode == "Pairs Overview":
        render_pair_overview()
    elif st.session_state.view_mode == "Pair Detail" and st.session_state.current_pair is not None:
        render_pair_detail(st.session_state.current_pair)
    elif st.session_state.view_mode == "Pair Monitor" and st.session_state.current_pair is not None:
        render_pair_monitor(st.session_state.current_pair)
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
    # Check if we're in debug mode (no database connection)
    try:
        engine = init_connection()
        if engine is None:
            # No database connection, enable simulated data mode
            if 'simulated_data_mode' not in st.session_state:
                enable_simulated_data_mode()
    except:
        # Error connecting to database, enable simulated data mode
        if 'simulated_data_mode' not in st.session_state:
            enable_simulated_data_mode()
    
    # Run the main application
    main()