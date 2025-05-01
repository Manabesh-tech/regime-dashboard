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
@st.cache_data(ttl=600)
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
@st.cache_data(ttl=300)
def fetch_current_parameters(pair_name):
    """
    Fetch current buffer_rate and position_multiplier for a specific pair.
    Returns a dictionary with parameter values or defaults if query fails.
    """
    engine = init_connection()
    if not engine:
        return {
            "buffer_rate": 0.001,
            "position_multiplier": 10000,
            "max_leverage": 100
        }
    
    try:
        query = f"""
        SELECT
            (leverage_config::jsonb->0->>'buffer_rate')::numeric AS buffer_rate,
            position_multiplier,
            max_leverage
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
                "position_multiplier": 10000,
                "max_leverage": 100
            }
        
        # Convert to dictionary
        params = {
            "buffer_rate": float(df['buffer_rate'].iloc[0]),
            "position_multiplier": float(df['position_multiplier'].iloc[0]),
            "max_leverage": float(df['max_leverage'].iloc[0])
        }
        
        return params
    except Exception as e:
        st.error(f"Error fetching parameters for {pair_name}: {e}")
        return {
            "buffer_rate": 0.001,
            "position_multiplier": 10000,
            "max_leverage": 100
        }

# Calculate edge for a specific pair
@st.cache_data(ttl=30)  # Cache for 30 seconds to avoid repeated identical queries
def calculate_edge(pair_name, lookback_minutes=10):
    """
    Calculate house edge for a specific pair using the SQL query from the first file.
    Returns edge value or None if calculation fails.
    """
    engine = init_connection()
    if not engine:
        return None
    
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
        
        # Add a debug log
        st.session_state.last_query_time = datetime.now()
        
        df = pd.read_sql(edge_query, engine)
        
        if df.empty or pd.isna(df['house_edge'].iloc[0]):
            # Set a small default positive edge if calculation returns 0 or null
            return 0.001
        
        edge_value = float(df['house_edge'].iloc[0])
        
        # If the edge is exactly 0, set a small positive value
        # This avoids issues with initial reference being zero
        if edge_value == 0:
            return 0.001
        
        # Add random small variation for testing if needed
        # import random
        # edge_value += random.uniform(-0.001, 0.001)
        
        return edge_value
    
    except Exception as e:
        st.error(f"Error calculating edge for {pair_name}: {e}")
        # Return a default small positive edge in case of error
        return 0.001

# Function to calculate fee for a percentage price move
def calculate_fee_for_move(move_pct, buffer_rate, position_multiplier, rate_multiplier=0.5, 
                          base_rate=0.02, bet=1.0, leverage=1.0):
    """
    Calculate fee for a percentage price move using the fee equation.
    """
    # Convert percentage move to decimal
    move_decimal = move_pct / 100
    
    # Assuming PT = 1 (starting price) for simplicity
    PT = 1.0
    Pt = PT * (1 - move_decimal)  # Price after move
    
    # Calculate price difference and its absolute relative value
    price_diff = PT - Pt
    abs_relative_diff = abs(price_diff / PT)
    
    # Calculate numerator
    numerator = -bet * leverage * price_diff * (1 + rate_multiplier * abs_relative_diff)
    
    # Calculate denominator
    denominator = (1 - buffer_rate) * (1 + 1000000 * position_multiplier * abs_relative_diff)
    
    # Calculate fee
    fee = numerator / denominator
    
    return fee

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
            'position_multiplier': 10000,
            'max_leverage': 100,
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
            'params_changed': False
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
        st.session_state.lookback_minutes = 10  # Default to 10 minutes
    
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
        st.session_state.auto_update = False
    
    if 'view_mode' not in st.session_state:
        st.session_state.view_mode = "Pairs Overview"  # Default view mode

# Function to initialize or reset the system for a new pair
def initialize_system(pair_name, lookback_minutes):
    """Initialize or reset the system for a new pair."""
    # Initialize pair state if it doesn't exist
    init_pair_state(pair_name)
    
    # Fetch current parameters from database
    params = fetch_current_parameters(pair_name)
    
    # Update session state with current parameters
    st.session_state.pair_data[pair_name]['buffer_rate'] = params["buffer_rate"]
    st.session_state.pair_data[pair_name]['position_multiplier'] = params["position_multiplier"]
    st.session_state.pair_data[pair_name]['max_leverage'] = params["max_leverage"]
    
    # Save reference values
    st.session_state.pair_data[pair_name]['reference_buffer_rate'] = params["buffer_rate"]
    st.session_state.pair_data[pair_name]['reference_position_multiplier'] = params["position_multiplier"]
    
    # Reset history
    timestamp = datetime.now()
    st.session_state.pair_data[pair_name]['edge_history'] = []
    st.session_state.pair_data[pair_name]['buffer_history'] = [(timestamp, st.session_state.pair_data[pair_name]['buffer_rate'])]
    st.session_state.pair_data[pair_name]['multiplier_history'] = [(timestamp, st.session_state.pair_data[pair_name]['position_multiplier'])]
    
    # Calculate and record initial fee for 0.1% move
    initial_fee = calculate_fee_for_move(
        0.1, 
        st.session_state.pair_data[pair_name]['buffer_rate'], 
        st.session_state.pair_data[pair_name]['position_multiplier']
    )
    st.session_state.pair_data[pair_name]['fee_history'] = [(timestamp, initial_fee)]
    
    # Fetch initial reference edge based on selected lookback period
    initial_edge = calculate_edge(pair_name, lookback_minutes)
    if initial_edge is not None:
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
    
    # Add to monitored pairs if not already there
    if pair_name not in st.session_state.monitored_pairs:
        st.session_state.monitored_pairs.append(pair_name)
    
    # Set as current pair
    st.session_state.current_pair = pair_name

# Function to process edge data and calculate parameter updates for a specific pair
def process_edge_data(pair_name, timestamp=None):
    """
    Process a new edge data point and calculate parameter updates if needed.
    Returns True if parameters need to be changed, False otherwise.
    """
    # Ensure pair state is initialized
    init_pair_state(pair_name)
    
    if timestamp is None:
        timestamp = datetime.now()
    
    # Fetch new edge
    new_edge = calculate_edge(pair_name, st.session_state.lookback_minutes)
    
    # Skip updates if edge calculation failed, but don't return yet to update timestamp
    if new_edge is not None:
        # Add to edge history
        st.session_state.pair_data[pair_name]['edge_history'].append((timestamp, new_edge))
        
        # Log the edge update to aid debugging
        old_edge = st.session_state.pair_data[pair_name]['current_edge']
        edge_change = 0 if old_edge is None else ((new_edge - old_edge) / old_edge * 100 if old_edge != 0 else 0)
        
        # Update current edge
        st.session_state.pair_data[pair_name]['current_edge'] = new_edge
        
        # Get reference edge and current parameters
        edge_ref = st.session_state.pair_data[pair_name]['reference_edge']
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
        timestamp = datetime.now()
        
        # Update buffer rate
        old_buffer = st.session_state.pair_data[pair_name]['buffer_rate']
        st.session_state.pair_data[pair_name]['buffer_rate'] = st.session_state.pair_data[pair_name]['proposed_buffer_rate']
        st.session_state.pair_data[pair_name]['buffer_history'].append((timestamp, st.session_state.pair_data[pair_name]['buffer_rate']))
        
        # Update position multiplier
        old_multiplier = st.session_state.pair_data[pair_name]['position_multiplier']
        st.session_state.pair_data[pair_name]['position_multiplier'] = st.session_state.pair_data[pair_name]['proposed_position_multiplier']
        st.session_state.pair_data[pair_name]['multiplier_history'].append((timestamp, st.session_state.pair_data[pair_name]['position_multiplier']))
        
        # Calculate and record fee for 0.1% move
        fee_for_01pct_move = calculate_fee_for_move(
            0.1, 
            st.session_state.pair_data[pair_name]['buffer_rate'], 
            st.session_state.pair_data[pair_name]['position_multiplier']
        )
        st.session_state.pair_data[pair_name]['fee_history'].append((timestamp, fee_for_01pct_move))
        
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
        timestamp = datetime.now()
        
        # Update buffer rate
        old_buffer = st.session_state.pair_data[pair_name]['buffer_rate']
        st.session_state.pair_data[pair_name]['buffer_rate'] = st.session_state.pair_data[pair_name]['reference_buffer_rate']
        st.session_state.pair_data[pair_name]['buffer_history'].append((timestamp, st.session_state.pair_data[pair_name]['buffer_rate']))
        
        # Update position multiplier
        old_multiplier = st.session_state.pair_data[pair_name]['position_multiplier']
        st.session_state.pair_data[pair_name]['position_multiplier'] = st.session_state.pair_data[pair_name]['reference_position_multiplier']
        st.session_state.pair_data[pair_name]['multiplier_history'].append((timestamp, st.session_state.pair_data[pair_name]['position_multiplier']))
        
        # Calculate and record fee for 0.1% move
        fee_for_01pct_move = calculate_fee_for_move(
            0.1, 
            st.session_state.pair_data[pair_name]['buffer_rate'], 
            st.session_state.pair_data[pair_name]['position_multiplier']
        )
        st.session_state.pair_data[pair_name]['fee_history'].append((timestamp, fee_for_01pct_move))
        
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
    ax.set_ylabel('Fee')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Format dates on x-axis
    fig.autofmt_xdate()
    
    # Tight layout
    plt.tight_layout()
    
    return fig

# Function to create fee curve plot for a specific pair
def create_fee_curve_plot(pair_name):
    """Create a plot of fee vs price move for a specific pair."""
    # Ensure pair state is initialized
    init_pair_state(pair_name)
    
    # Calculate fee across a range of move sizes
    move_sizes = np.linspace(-1, 1, 201)
    current_fees = [calculate_fee_for_move(move, 
                                        st.session_state.pair_data[pair_name]['buffer_rate'], 
                                        st.session_state.pair_data[pair_name]['position_multiplier']) 
                   for move in move_sizes]
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot current fee curve
    ax.plot(move_sizes, current_fees, 'b-', label='Current Fee')
    
    # Plot proposed fee curve if available
    if (st.session_state.pair_data[pair_name]['proposed_buffer_rate'] is not None and 
        st.session_state.pair_data[pair_name]['proposed_position_multiplier'] is not None):
        proposed_fees = [calculate_fee_for_move(move, 
                                              st.session_state.pair_data[pair_name]['proposed_buffer_rate'], 
                                              st.session_state.pair_data[pair_name]['proposed_position_multiplier']) 
                       for move in move_sizes]
        ax.plot(move_sizes, proposed_fees, 'r--', label='Proposed Fee')
    
    # Set title and labels
    ax.set_title(f'Fee vs. Price Move Size - {pair_name}')
    ax.set_xlabel('Price Move (%)')
    ax.set_ylabel('Fee')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add vertical and horizontal lines at 0
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    # Add legend
    ax.legend()
    
    # Tight layout
    plt.tight_layout()
    
    return fig

# Function to create fee comparison table for a specific pair
def create_fee_comparison_table(pair_name):
    """Create a table comparing fees for different move sizes with current and proposed parameters for a specific pair."""
    # Ensure pair state is initialized
    init_pair_state(pair_name)
    
    move_sizes = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    
    # Calculate fees with current parameters
    current_fees = [calculate_fee_for_move(move, 
                                        st.session_state.pair_data[pair_name]['buffer_rate'], 
                                        st.session_state.pair_data[pair_name]['position_multiplier']) 
                   for move in move_sizes]
    
    # Calculate fees with proposed parameters if available
    if (st.session_state.pair_data[pair_name]['proposed_buffer_rate'] is not None and 
        st.session_state.pair_data[pair_name]['proposed_position_multiplier'] is not None):
        proposed_fees = [calculate_fee_for_move(move, 
                                              st.session_state.pair_data[pair_name]['proposed_buffer_rate'], 
                                              st.session_state.pair_data[pair_name]['proposed_position_multiplier']) 
                       for move in move_sizes]
        
        # Create dataframe for the table with both current and proposed
        fee_df = pd.DataFrame({
            'Move Size (%)': move_sizes,
            'Current Fee': current_fees,
            'Proposed Fee': proposed_fees,
            'Difference (%)': [(new - old) / old * 100 if old != 0 else float('inf') 
                              for new, old in zip(proposed_fees, current_fees)]
        })
    else:
        # Create dataframe with just current fees
        fee_df = pd.DataFrame({
            'Move Size (%)': move_sizes,
            'Current Fee': current_fees
        })
    
    return fee_df

# Function to render the pair overview cards
def render_pair_overview():
    """Render overview cards for all monitored pairs."""
    if not st.session_state.monitored_pairs:
        st.info("No pairs are currently being monitored. Select a pair and click 'Initialize Pair and Start Monitoring' in the sidebar.")
        return
    
    st.markdown("### Monitored Trading Pairs")
    st.markdown("Select a pair below to view detailed analytics or click 'Monitor' to update parameters.")
    
    # Create a grid layout for pair cards (3 columns)
    columns = st.columns(3)
    
    # Render a card for each monitored pair
    for i, pair_name in enumerate(st.session_state.monitored_pairs):
        with columns[i % 3]:
            # Get pair data
            pair_data = st.session_state.pair_data.get(pair_name, {})
            current_edge = pair_data.get('current_edge')
            reference_edge = pair_data.get('reference_edge')
            buffer_rate = pair_data.get('buffer_rate')
            position_multiplier = pair_data.get('position_multiplier')
            last_update = pair_data.get('last_update_time')
            params_changed = pair_data.get('params_changed', False)
            
            # Determine status indicator
            status = get_pair_status(current_edge, reference_edge)
            status_color = "green" if status == "green" else "orange" if status == "yellow" else "red"
            
            # Format edge values and last update time
            edge_display = f"{current_edge:.4%}" if current_edge is not None else "N/A"
            ref_display = f"{reference_edge:.4%}" if reference_edge is not None else "N/A"
            
            # Calculate edge delta
            if current_edge is not None and reference_edge is not None:
                edge_delta = current_edge - reference_edge
                delta_pct = edge_delta / abs(reference_edge) * 100 if reference_edge != 0 else 0
                delta_color = "green" if edge_delta >= 0 else "red"
                delta_display = f"{edge_delta:.4%} ({delta_pct:+.2f}%)"
            else:
                delta_display = "N/A"
                delta_color = "gray"
            
            # Calculate last update time
            if last_update:
                time_since_update = datetime.now() - last_update
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
        return
    
    timestamp = datetime.now()
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
    col1, col2, col3, col4 = st.columns(4)
    
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
    
    # Fee Analysis tab
    with detail_tabs[2]:
        st.markdown("### Fee Analysis")
        
        # Create fee curve plot
        fee_curve_plot = create_fee_curve_plot(pair_name)
        if fee_curve_plot is not None:
            st.pyplot(fee_curve_plot)
        
        # Fee comparison table
        st.markdown("### Fee Comparison Table")
        
        # Create fee comparison table
        fee_df = create_fee_comparison_table(pair_name)
        st.dataframe(fee_df, use_container_width=True)
        
        # Fee equation details
        st.markdown("### Fee Equation")
        st.markdown("""
        The fee is calculated using the following equation:
        
        $$Fee = \\frac{-Bet \\times Leverage \\times (P_T-P_t) \\times (1 + Rate Multiplier \\cdot |\\frac{P_T-P_t}{P_T}|)}{(1-Base Rate) \\cdot (1 + 10^6 \\cdot Position Multiplier \\cdot |\\frac{P_T-P_t}{P_T}|)}$$
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
    col1, col2, col3, col4 = st.columns(4)
    
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
    
    # Show edge plot
    edge_plot = create_edge_plot(pair_name)
    if edge_plot is not None:
        st.pyplot(edge_plot)
    else:
        st.info("Not enough data points yet for edge visualization.")
    
    # Show parameter update notification
    if st.session_state.pair_data[pair_name].get('params_changed', False):
        st.markdown('<div class="warning">Parameter updates available. Review proposed changes below.</div>', unsafe_allow_html=True)
        
        # Parameter change details
        if st.session_state.pair_data[pair_name]['proposed_buffer_rate'] is not None and st.session_state.pair_data[pair_name]['proposed_position_multiplier'] is not None:
            delta_buffer = st.session_state.pair_data[pair_name]['proposed_buffer_rate'] - st.session_state.pair_data[pair_name]['buffer_rate']
            delta_multiplier = st.session_state.pair_data[pair_name]['proposed_position_multiplier'] - st.session_state.pair_data[pair_name]['position_multiplier']
            
            # Calculate current and new fees
            current_fee = calculate_fee_for_move(0.1, st.session_state.pair_data[pair_name]['buffer_rate'], st.session_state.pair_data[pair_name]['position_multiplier'])
            new_fee = calculate_fee_for_move(0.1, st.session_state.pair_data[pair_name]['proposed_buffer_rate'], st.session_state.pair_data[pair_name]['proposed_position_multiplier'])
            delta_fee = new_fee - current_fee
            
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
                st.markdown(f"Current: {current_fee:.8f}")
                st.markdown(f"Proposed: {new_fee:.8f}")
                
                # Show change with color
                pct_change = delta_fee/current_fee*100 if current_fee != 0 else 0
                change_color = "up" if delta_fee > 0 else "down"
                st.markdown(f"Change: {delta_fee:.8f} (<span class='{change_color}'>{pct_change:+.2f}%</span>)", unsafe_allow_html=True)
        
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

# Add a heartbeat function to prevent browser sleep
def heartbeat():
    """Add a hidden heartbeat to prevent browser sleep."""
    current_time = datetime.now()
    
    # Create a hidden element that changes every second
    st.markdown(f"""
    <div style="display: none;">
        Heartbeat: {current_time.timestamp()}
    </div>
    """, unsafe_allow_html=True)

def main():
    # Initialize all session state variables to prevent duplicates
    init_session_state()
    
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
        index=0,  # Default to 1 minute for more frequent updates
        key="interval_radio"
    )
    
    lookback_minutes = update_interval_options[update_interval_selection]
    st.session_state.lookback_minutes = lookback_minutes
    
    # Initialize button
    initialize_button = st.sidebar.button(
        "Initialize Pair and Start Monitoring", 
        help=f"Initialize the selected pair {selected_pair} and begin monitoring",
        type="primary",
        key="init_button"
    )
    
    if initialize_button:
        initialize_system(selected_pair, lookback_minutes)
        st.sidebar.success(f"Started monitoring for {selected_pair}")
        st.session_state.view_mode = "Pairs Overview"
        
        # Reset the timer for auto-update
        st.session_state.last_global_update = datetime.now()
        st.rerun()
    
    # Add a timestamp to track the last global update
    if 'last_global_update' not in st.session_state:
        st.session_state.last_global_update = datetime.now()
    
    # Batch operations section
    st.sidebar.markdown('<div class="subheader-style">Batch Operations</div>', unsafe_allow_html=True)
    
    # Batch update button
    if st.sidebar.button(
        "Update All Pairs Now", 
        help="Immediately fetch new edge data for all monitored pairs",
        key="batch_update_button",
        disabled=len(st.session_state.monitored_pairs) == 0
    ):
        # Set a flag to force an update regardless of timer
        st.session_state.update_clicked = True
        
        # Perform the immediate update
        with st.spinner("Updating all pairs..."):
            pairs_updated = batch_update_all_pairs()
            st.session_state.last_global_update = datetime.now()
            
            if pairs_updated:
                st.sidebar.warning(f"Parameter updates available for {len(pairs_updated)} pairs")
            else:
                st.sidebar.success("All pairs updated, no parameter changes needed")
        
        st.rerun()
        
    # Show auto-update status and last update time
    current_time = datetime.now()
    time_since_update = (current_time - st.session_state.last_global_update).total_seconds()
    
    # Auto-update toggle with more visible status
    auto_update_col1, auto_update_col2 = st.sidebar.columns([3, 2])
    
    with auto_update_col1:
        st.session_state.auto_update = st.checkbox(
            "Auto-update data", 
            value=st.session_state.auto_update,
            help="Automatically fetch new edge data at the specified interval",
            key="auto_update_checkbox"
        )
    
    # Calculate next update time and display
    update_interval_seconds = lookback_minutes * 60
    time_to_next_update = max(0, update_interval_seconds - time_since_update)
    
    with auto_update_col2:
        if st.session_state.auto_update:
            if time_to_next_update < 60:
                st.info(f"Next update in {int(time_to_next_update)} sec")
            else:
                st.info(f"Next update in {int(time_to_next_update/60)} min")
    
    # Show last update time
    if st.session_state.last_global_update:
        if time_since_update < 60:
            update_text = f"Last update: {int(time_since_update)} seconds ago"
        elif time_since_update < 3600:
            update_text = f"Last update: {int(time_since_update/60)} minutes ago"
        else:
            update_text = f"Last update: {int(time_since_update/3600)} hours ago"
        st.sidebar.text(update_text)
    
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
    
    # Auto-update logic - check if it's time for an update
    if st.session_state.auto_update and len(st.session_state.monitored_pairs) > 0:
        # Check if it's time for an update
        if time_since_update >= update_interval_seconds:
            # Update all pairs
            with st.spinner("Updating all pairs..."):
                pairs_updated = batch_update_all_pairs()
                # Update the global update timestamp
                st.session_state.last_global_update = current_time
                
            # Log the update to help with debugging
            st.sidebar.success(f"Auto-updated at {current_time.strftime('%H:%M:%S')}")
            
            # Force refresh
            st.rerun()
    
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
    
    # Add heartbeat to prevent browser sleep
    if st.session_state.auto_update:
        heartbeat()

if __name__ == "__main__":
    main()