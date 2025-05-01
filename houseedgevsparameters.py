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
@st.cache_data(ttl=60)
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
        
        df = pd.read_sql(edge_query, engine)
        
        if df.empty or pd.isna(df['house_edge'].iloc[0]):
            # Set a small default positive edge if calculation returns 0 or null
            return 0.001
        
        edge_value = float(df['house_edge'].iloc[0])
        
        # If the edge is exactly 0, set a small positive value
        # This avoids issues with initial reference being zero
        if edge_value == 0:
            return 0.001
        
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

# Initialize session state variables
if 'buffer_rate' not in st.session_state:
    st.session_state.buffer_rate = 0.001

if 'position_multiplier' not in st.session_state:
    st.session_state.position_multiplier = 10000

if 'max_leverage' not in st.session_state:
    st.session_state.max_leverage = 100

if 'edge_history' not in st.session_state:
    st.session_state.edge_history = []  # List of (timestamp, edge) tuples

if 'buffer_history' not in st.session_state:
    st.session_state.buffer_history = []  # List of (timestamp, buffer_rate) tuples

if 'multiplier_history' not in st.session_state:
    st.session_state.multiplier_history = []  # List of (timestamp, position_multiplier) tuples

if 'fee_history' not in st.session_state:
    st.session_state.fee_history = []  # List of (timestamp, fee_for_01pct_move) tuples

if 'current_edge' not in st.session_state:
    st.session_state.current_edge = None

if 'reference_edge' not in st.session_state:
    st.session_state.reference_edge = None

if 'proposed_buffer_rate' not in st.session_state:
    st.session_state.proposed_buffer_rate = None

if 'proposed_position_multiplier' not in st.session_state:
    st.session_state.proposed_position_multiplier = None

if 'reference_buffer_rate' not in st.session_state:
    st.session_state.reference_buffer_rate = None

if 'reference_position_multiplier' not in st.session_state:
    st.session_state.reference_position_multiplier = None

if 'lookback_minutes' not in st.session_state:
    st.session_state.lookback_minutes = 10  # Default to 10 minutes

if 'last_update_time' not in st.session_state:
    st.session_state.last_update_time = None

if 'buffer_alpha_up' not in st.session_state:
    st.session_state.buffer_alpha_up = 0.1

if 'buffer_alpha_down' not in st.session_state:
    st.session_state.buffer_alpha_down = 0.02

if 'multiplier_alpha_up' not in st.session_state:
    st.session_state.multiplier_alpha_up = 0.02

if 'multiplier_alpha_down' not in st.session_state:
    st.session_state.multiplier_alpha_down = 0.1

if 'params_changed' not in st.session_state:
    st.session_state.params_changed = False

if 'history_length' not in st.session_state:
    st.session_state.history_length = 100  # Default history length

if 'auto_update' not in st.session_state:
    st.session_state.auto_update = False

# Function to update display parameters
def update_display_parameters():
    """Apply the proposed parameter updates to display values."""
    if (st.session_state.proposed_buffer_rate is not None and 
        st.session_state.proposed_position_multiplier is not None):
        
        # Update parameters
        timestamp = datetime.now()
        
        # Update buffer rate
        old_buffer = st.session_state.buffer_rate
        st.session_state.buffer_rate = st.session_state.proposed_buffer_rate
        st.session_state.buffer_history.append((timestamp, st.session_state.buffer_rate))
        
        # Update position multiplier
        old_multiplier = st.session_state.position_multiplier
        st.session_state.position_multiplier = st.session_state.proposed_position_multiplier
        st.session_state.multiplier_history.append((timestamp, st.session_state.position_multiplier))
        
        # Calculate and record fee for 0.1% move
        fee_for_01pct_move = calculate_fee_for_move(
            0.1, st.session_state.buffer_rate, st.session_state.position_multiplier
        )
        st.session_state.fee_history.append((timestamp, fee_for_01pct_move))
        
        # Mark that parameters have been changed
        st.session_state.params_changed = False
        
        # Reset proposed values
        st.session_state.proposed_buffer_rate = None
        st.session_state.proposed_position_multiplier = None
        
        return True, old_buffer, st.session_state.buffer_rate, old_multiplier, st.session_state.position_multiplier
    
    return False, None, None, None, None

# Function to reset parameters to reference values
def reset_to_reference_parameters():
    """Reset parameters to reference values."""
    if (st.session_state.reference_buffer_rate is not None and 
        st.session_state.reference_position_multiplier is not None):
        
        # Update parameters
        timestamp = datetime.now()
        
        # Update buffer rate
        old_buffer = st.session_state.buffer_rate
        st.session_state.buffer_rate = st.session_state.reference_buffer_rate
        st.session_state.buffer_history.append((timestamp, st.session_state.buffer_rate))
        
        # Update position multiplier
        old_multiplier = st.session_state.position_multiplier
        st.session_state.position_multiplier = st.session_state.reference_position_multiplier
        st.session_state.multiplier_history.append((timestamp, st.session_state.position_multiplier))
        
        # Calculate and record fee for 0.1% move
        fee_for_01pct_move = calculate_fee_for_move(
            0.1, st.session_state.buffer_rate, st.session_state.position_multiplier
        )
        st.session_state.fee_history.append((timestamp, fee_for_01pct_move))
        
        # Mark that parameters have been changed
        st.session_state.params_changed = False
        
        # Reset proposed values
        st.session_state.proposed_buffer_rate = None
        st.session_state.proposed_position_multiplier = None
        
        return True, old_buffer, st.session_state.buffer_rate, old_multiplier, st.session_state.position_multiplier
    
    return False, None, None, None, None

# Function to create edge plot with matplotlib
def create_edge_plot():
    """Create a plot of house edge with reference line."""
    if len(st.session_state.edge_history) < 1:
        return None
    
    # Extract data for plotting
    timestamps = [t for t, _ in st.session_state.edge_history]
    edges = [e for _, e in st.session_state.edge_history]
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot edge line
    ax.plot(timestamps, edges, 'b-', label='House Edge')
    
    # Add reference line if available
    if st.session_state.reference_edge is not None:
        ax.axhline(y=st.session_state.reference_edge, color='r', linestyle='--', label='Reference Edge')
    
    # Set title and labels
    ax.set_title('House Edge Monitoring')
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

# Function to create buffer rate and position multiplier plots
def create_parameter_plots():
    """Create plots for buffer rate and position multiplier history."""
    buffer_fig, multiplier_fig = None, None
    
    # Buffer rate plot
    if len(st.session_state.buffer_history) >= 1:
        # Extract data for plotting
        buffer_times = [t for t, _ in st.session_state.buffer_history]
        buffer_rates = [r for _, r in st.session_state.buffer_history]
        
        # Create figure and axis
        buffer_fig, buffer_ax = plt.subplots(figsize=(10, 5))
        
        # Plot buffer rate line
        buffer_ax.plot(buffer_times, buffer_rates, 'g-', marker='o', label='Buffer Rate')
        
        # Add reference line if available
        if st.session_state.reference_buffer_rate is not None:
            buffer_ax.axhline(y=st.session_state.reference_buffer_rate, color='r', 
                             linestyle='--', label='Reference Buffer Rate')
        
        # Set title and labels
        buffer_ax.set_title('Buffer Rate Adjustments')
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
    if len(st.session_state.multiplier_history) >= 1:
        # Extract data for plotting
        multiplier_times = [t for t, _ in st.session_state.multiplier_history]
        multipliers = [m for _, m in st.session_state.multiplier_history]
        
        # Create figure and axis
        multiplier_fig, multiplier_ax = plt.subplots(figsize=(10, 5))
        
        # Plot position multiplier line
        multiplier_ax.plot(multiplier_times, multipliers, 'm-', marker='o', label='Position Multiplier')
        
        # Add reference line if available
        if st.session_state.reference_position_multiplier is not None:
            multiplier_ax.axhline(y=st.session_state.reference_position_multiplier, color='r', 
                                 linestyle='--', label='Reference Position Multiplier')
        
        # Set title and labels
        multiplier_ax.set_title('Position Multiplier Adjustments')
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

# Function to create fee plot
def create_fee_plot():
    """Create a plot of fee history for 0.1% price move."""
    if len(st.session_state.fee_history) < 1:
        return None
    
    # Extract data for plotting
    fee_times = [t for t, _ in st.session_state.fee_history]
    fees = [f for _, f in st.session_state.fee_history]
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot fee line
    ax.plot(fee_times, fees, 'r-', marker='o', label='Fee for 0.1% Move')
    
    # Set title and labels
    ax.set_title('Fee for 0.1% Price Move')
    ax.set_xlabel('Time')
    ax.set_ylabel('Fee')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Format dates on x-axis
    fig.autofmt_xdate()
    
    # Tight layout
    plt.tight_layout()
    
    return fig

# Function to create fee curve plot
def create_fee_curve_plot():
    """Create a plot of fee vs price move."""
    # Calculate fee across a range of move sizes
    move_sizes = np.linspace(-1, 1, 201)
    current_fees = [calculate_fee_for_move(move, st.session_state.buffer_rate, 
                                        st.session_state.position_multiplier) 
                   for move in move_sizes]
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot current fee curve
    ax.plot(move_sizes, current_fees, 'b-', label='Current Fee')
    
    # Plot proposed fee curve if available
    if (st.session_state.proposed_buffer_rate is not None and 
        st.session_state.proposed_position_multiplier is not None):
        proposed_fees = [calculate_fee_for_move(move, st.session_state.proposed_buffer_rate, 
                                              st.session_state.proposed_position_multiplier) 
                       for move in move_sizes]
        ax.plot(move_sizes, proposed_fees, 'r--', label='Proposed Fee')
    
    # Set title and labels
    ax.set_title('Fee vs. Price Move Size')
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

# Function to create fee comparison table
def create_fee_comparison_table():
    """Create a table comparing fees for different move sizes with current and proposed parameters."""
    move_sizes = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    
    # Calculate fees with current parameters
    current_fees = [calculate_fee_for_move(move, st.session_state.buffer_rate, 
                                        st.session_state.position_multiplier) 
                   for move in move_sizes]
    
    # Calculate fees with proposed parameters if available
    if (st.session_state.proposed_buffer_rate is not None and 
        st.session_state.proposed_position_multiplier is not None):
        proposed_fees = [calculate_fee_for_move(move, st.session_state.proposed_buffer_rate, 
                                              st.session_state.proposed_position_multiplier) 
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

# Function to initialize or reset the system for a new pair
def initialize_system(pair_name, lookback_minutes):
    """Initialize or reset the system for a new pair."""
    # Fetch current parameters from database
    params = fetch_current_parameters(pair_name)
    
    # Update session state with current parameters
    st.session_state.buffer_rate = params["buffer_rate"]
    st.session_state.position_multiplier = params["position_multiplier"]
    st.session_state.max_leverage = params["max_leverage"]
    
    # Save reference values
    st.session_state.reference_buffer_rate = params["buffer_rate"]
    st.session_state.reference_position_multiplier = params["position_multiplier"]
    
    # Reset history
    timestamp = datetime.now()
    st.session_state.edge_history = []
    st.session_state.buffer_history = [(timestamp, st.session_state.buffer_rate)]
    st.session_state.multiplier_history = [(timestamp, st.session_state.position_multiplier)]
    
    # Calculate and record initial fee for 0.1% move
    initial_fee = calculate_fee_for_move(0.1, st.session_state.buffer_rate, st.session_state.position_multiplier)
    st.session_state.fee_history = [(timestamp, initial_fee)]
    
    # Fetch initial reference edge based on selected lookback period
    initial_edge = calculate_edge(pair_name, lookback_minutes)
    if initial_edge is not None:
        st.session_state.reference_edge = initial_edge
        st.session_state.current_edge = initial_edge
        st.session_state.edge_history.append((timestamp, initial_edge))
    
    # Reset proposed values
    st.session_state.proposed_buffer_rate = None
    st.session_state.proposed_position_multiplier = None
    
    # Reset params_changed flag
    st.session_state.params_changed = False
    
    # Set last update time
    st.session_state.last_update_time = timestamp

# Function to process edge data and calculate parameter updates
def process_edge_data(pair_name, timestamp=None):
    """
    Process a new edge data point and calculate parameter updates if needed.
    Returns True if parameters need to be changed, False otherwise.
    """
    if timestamp is None:
        timestamp = datetime.now()
    
    # Fetch new edge
    new_edge = calculate_edge(pair_name, st.session_state.lookback_minutes)
    if new_edge is None:
        return False  # Can't update without a valid edge
    
    # Add to edge history
    st.session_state.edge_history.append((timestamp, new_edge))
    st.session_state.current_edge = new_edge
    
    # Update last update time
    st.session_state.last_update_time = timestamp
    
    # Get reference edge and current parameters
    edge_ref = st.session_state.reference_edge
    current_buffer = st.session_state.buffer_rate
    current_multiplier = st.session_state.position_multiplier
    max_leverage = st.session_state.max_leverage
    
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
        st.session_state.proposed_buffer_rate = new_buffer_rate
        st.session_state.proposed_position_multiplier = new_position_multiplier
        st.session_state.params_changed = True
    
    return parameters_changed

def main():
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
    st.sidebar.markdown('<div class="subheader-style">Configuration</div>', unsafe_allow_html=True)
    
    # Fetch available pairs
    pairs = fetch_pairs()
    
    # Select pair
    selected_pair = st.sidebar.selectbox(
        "Select Trading Pair",
        options=pairs,
        index=0
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
        index=2  # Default to 10 minutes
    )
    
    lookback_minutes = update_interval_options[update_interval_selection]
    st.session_state.lookback_minutes = lookback_minutes
    
    # Sensitivity parameters
    st.sidebar.markdown('<div class="subheader-style">Sensitivity Parameters</div>', unsafe_allow_html=True)
    
    # Buffer rate increase sensitivity
    st.sidebar.markdown("**Buffer Rate Increase Sensitivity**")
    st.session_state.buffer_alpha_up = st.sidebar.slider(
        "Buffer Alpha Up",
        min_value=0.01, 
        max_value=0.5, 
        value=st.session_state.buffer_alpha_up,
        step=0.01,
        help="How quickly buffer rate increases when edge declines"
    )
    
    # Buffer rate decrease sensitivity
    st.sidebar.markdown("**Buffer Rate Decrease Sensitivity**")
    st.session_state.buffer_alpha_down = st.sidebar.slider(
        "Buffer Alpha Down",
        min_value=0.001, 
        max_value=0.1, 
        value=st.session_state.buffer_alpha_down,
        step=0.001,
        help="How quickly buffer rate decreases when edge improves"
    )
    
    # Position multiplier increase sensitivity
    st.sidebar.markdown("**Position Multiplier Increase Sensitivity**")
    st.session_state.multiplier_alpha_up = st.sidebar.slider(
        "Position Alpha Up",
        min_value=0.001, 
        max_value=0.1, 
        value=st.session_state.multiplier_alpha_up,
        step=0.001,
        help="How quickly position multiplier increases when edge improves"
    )
    
    # Position multiplier decrease sensitivity
    st.sidebar.markdown("**Position Multiplier Decrease Sensitivity**")
    st.session_state.multiplier_alpha_down = st.sidebar.slider(
        "Position Alpha Down",
        min_value=0.01, 
        max_value=0.5, 
        value=st.session_state.multiplier_alpha_down,
        step=0.01,
        help="How quickly position multiplier decreases when edge declines"
    )
    
    # Auto-update toggle
    st.sidebar.markdown('<div class="subheader-style">Update Controls</div>', unsafe_allow_html=True)
    
    st.session_state.auto_update = st.sidebar.checkbox(
        "Auto-update", 
        value=st.session_state.auto_update,
        help="Automatically fetch new edge data at the specified interval"
    )
    
    # History length (with specific key to prevent duplication)
    st.session_state.history_length = st.sidebar.slider(
        "History Length (points)", 
        min_value=10, 
        max_value=1000, 
        value=st.session_state.history_length,
        help="Number of data points to keep in history",
        key="history_length_slider"
    )
    
    # Initialize button
    initialize_button = st.sidebar.button(
        "Initialize System", 
        help="Initialize the system with the selected pair",
        type="primary"
    )
    
    if initialize_button:
        initialize_system(selected_pair, lookback_minutes)
        st.sidebar.success(f"System initialized for {selected_pair}")
        st.rerun()
    
    # Reset button
    if st.sidebar.button(
        "Reset to Reference", 
        help="Reset parameters to reference values",
        type="secondary"
    ):
        success, old_buffer, new_buffer, old_multiplier, new_multiplier = reset_to_reference_parameters()
        if success:
            st.sidebar.success("Parameters reset to reference values")
            st.rerun()
        else:
            st.sidebar.error("No reference parameters available")
    
    # Manual update button
    update_button = st.sidebar.button(
        "Fetch New Data", 
        help="Manually fetch new edge data",
        disabled=st.session_state.auto_update
    )
    
    # Main dashboard tabs
    tabs = st.tabs(["Monitoring", "Parameter History", "Fee Analysis"])
    
    # Monitoring tab
    with tabs[0]:
        # Check if the system has been initialized
        if st.session_state.reference_edge is None or len(st.session_state.edge_history) == 0:
            st.warning("Please initialize the system by clicking the 'Initialize System' button in the sidebar.")
        else:
            # Create columns for key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            # Show current edge
            with col1:
                if st.session_state.current_edge is not None:
                    edge_delta = st.session_state.current_edge - st.session_state.reference_edge if st.session_state.reference_edge is not None else None
                    delta_color = "down" if edge_delta and edge_delta < 0 else "up"
                    delta_str = f"<span class='{delta_color}'>{edge_delta:.4%}</span>" if edge_delta is not None else ""
                    st.markdown(f"**Current Edge:** {st.session_state.current_edge:.4%} {delta_str}", unsafe_allow_html=True)
                else:
                    st.markdown("**Current Edge:** N/A")
            
            # Show reference edge
            with col2:
                if st.session_state.reference_edge is not None:
                    st.markdown(f"**Reference Edge:** {st.session_state.reference_edge:.4%}")
                else:
                    st.markdown("**Reference Edge:** N/A")
            
            # Show current buffer rate
            with col3:
                st.markdown(f"**Buffer Rate:** {st.session_state.buffer_rate:.6f}")
            
            # Show current position multiplier
            with col4:
                st.markdown(f"**Position Multiplier:** {st.session_state.position_multiplier:.1f}")
            
            # Create edge plot
            edge_plot = create_edge_plot()
            if edge_plot is not None:
                st.pyplot(edge_plot)
            else:
                st.info("Not enough data points yet for edge visualization.")
            
            # Show parameter update notification
            if st.session_state.params_changed:
                st.markdown('<div class="warning">Parameter updates available. Review proposed changes below.</div>', unsafe_allow_html=True)
                
                # Parameter change details
                if st.session_state.proposed_buffer_rate is not None and st.session_state.proposed_position_multiplier is not None:
                    delta_buffer = st.session_state.proposed_buffer_rate - st.session_state.buffer_rate
                    delta_multiplier = st.session_state.proposed_position_multiplier - st.session_state.position_multiplier
                    
                    # Calculate current and new fees
                    current_fee = calculate_fee_for_move(0.1, st.session_state.buffer_rate, st.session_state.position_multiplier)
                    new_fee = calculate_fee_for_move(0.1, st.session_state.proposed_buffer_rate, st.session_state.proposed_position_multiplier)
                    delta_fee = new_fee - current_fee
                    
                    # Show proposed changes
                    st.markdown("### Proposed Parameter Changes")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**Buffer Rate**")
                        st.markdown(f"Current: {st.session_state.buffer_rate:.6f}")
                        st.markdown(f"Proposed: {st.session_state.proposed_buffer_rate:.6f}")
                        
                        # Show change with color
                        pct_change = delta_buffer/st.session_state.buffer_rate*100
                        change_color = "up" if delta_buffer > 0 else "down"
                        st.markdown(f"Change: {delta_buffer:.6f} (<span class='{change_color}'>{pct_change:+.2f}%</span>)", unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("**Position Multiplier**")
                        st.markdown(f"Current: {st.session_state.position_multiplier:.1f}")
                        st.markdown(f"Proposed: {st.session_state.proposed_position_multiplier:.1f}")
                        
                        # Show change with color
                        pct_change = delta_multiplier/st.session_state.position_multiplier*100
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
                update_button = st.button("Update Parameters", type="primary", key="update_params")
                st.markdown('</div>', unsafe_allow_html=True)
                
                if update_button:
                    # Apply the updates to display (not database)
                    success, old_buffer, new_buffer, old_multiplier, new_multiplier = update_display_parameters()
                    
                    if success:
                        st.markdown('<div class="success">Display parameters updated successfully!</div>', unsafe_allow_html=True)
                        st.rerun()
    
    # Parameter History tab
    with tabs[1]:
        if st.session_state.reference_edge is None:
            st.warning("Please initialize the system by clicking the 'Initialize System' button in the sidebar.")
        else:
            # Create parameter plots
            buffer_fig, multiplier_fig = create_parameter_plots()
            
            if buffer_fig is not None:
                st.pyplot(buffer_fig)
            else:
                st.info("Not enough data points yet for buffer rate visualization.")
                
            if multiplier_fig is not None:
                st.pyplot(multiplier_fig)
            else:
                st.info("Not enough data points yet for position multiplier visualization.")
                
            # Show fee history plot
            fee_fig = create_fee_plot()
            if fee_fig is not None:
                st.pyplot(fee_fig)
            else:
                st.info("Not enough data points yet for fee visualization.")
            
            # Show parameter history data
            if st.checkbox("Show Raw History Data"):
                # Create tabs for different history tables
                history_tabs = st.tabs(["Edge History", "Buffer History", "Multiplier History", "Fee History"])
                
                with history_tabs[0]:
                    if len(st.session_state.edge_history) > 0:
                        edge_df = pd.DataFrame({
                            'Timestamp': [t for t, _ in st.session_state.edge_history],
                            'Edge': [e for _, e in st.session_state.edge_history]
                        })
                        # Format edge as percentage
                        edge_df['Edge'] = edge_df['Edge'].map(lambda x: f"{x:.4%}")
                        st.dataframe(edge_df, use_container_width=True)
                    else:
                        st.info("No edge history data yet.")
                
                with history_tabs[1]:
                    if len(st.session_state.buffer_history) > 0:
                        buffer_df = pd.DataFrame({
                            'Timestamp': [t for t, _ in st.session_state.buffer_history],
                            'Buffer Rate': [r for _, r in st.session_state.buffer_history]
                        })
                        st.dataframe(buffer_df, use_container_width=True)
                    else:
                        st.info("No buffer rate history data yet.")
                
                with history_tabs[2]:
                    if len(st.session_state.multiplier_history) > 0:
                        multiplier_df = pd.DataFrame({
                            'Timestamp': [t for t, _ in st.session_state.multiplier_history],
                            'Position Multiplier': [m for _, m in st.session_state.multiplier_history]
                        })
                        st.dataframe(multiplier_df, use_container_width=True)
                    else:
                        st.info("No position multiplier history data yet.")
                
                with history_tabs[3]:
                    if len(st.session_state.fee_history) > 0:
                        fee_df = pd.DataFrame({
                            'Timestamp': [t for t, _ in st.session_state.fee_history],
                            'Fee for 0.1% Move': [f for _, f in st.session_state.fee_history]
                        })
                        st.dataframe(fee_df, use_container_width=True)
                    else:
                        st.info("No fee history data yet.")
    
    # Fee Analysis tab
    with tabs[2]:
        if st.session_state.reference_edge is None:
            st.warning("Please initialize the system by clicking the 'Initialize System' button in the sidebar.")
        else:
            st.markdown("### Fee Analysis")
            
            # Create fee curve plot
            fee_curve_plot = create_fee_curve_plot()
            if fee_curve_plot is not None:
                st.pyplot(fee_curve_plot)
            
            # Fee comparison table
            st.markdown("### Fee Comparison Table")
            
            # Create fee comparison table
            fee_df = create_fee_comparison_table()
            st.dataframe(fee_df, use_container_width=True)
            
            # Fee equation details
            st.markdown("### Fee Equation")
            st.markdown("""
            The fee is calculated using the following equation:
            
            $$Fee = \\frac{-Bet \\times Leverage \\times (P_T-P_t) \\times (1 + Rate Multiplier \\cdot |\\frac{P_T-P_t}{P_T}|)}{(1-Base Rate) \\cdot (1 + 10^6 \\cdot Position Multiplier \\cdot |\\frac{P_T-P_t}{P_T}|)}$$
            """)
    
    # Check if the system is already initialized
    if st.session_state.reference_edge is not None:
        # Update logic
        if update_button or (st.session_state.auto_update and st.session_state.last_update_time is not None):
            # Manual update button clicked or auto-update enabled
            current_time = datetime.now()
            
            # Determine if it's time to update
            should_update = False
            if update_button:
                # Manual update
                should_update = True
            elif st.session_state.auto_update:
                # Auto-update based on interval
                time_since_update = (current_time - st.session_state.last_update_time).total_seconds()
                update_interval_seconds = lookback_minutes * 60
                should_update = time_since_update >= update_interval_seconds
            
            if should_update:
                # Process new edge data
                process_edge_data(selected_pair, current_time)
                
                # Force refresh
                st.rerun()
    
    # Prune history if too long
    if len(st.session_state.edge_history) > st.session_state.history_length:
        st.session_state.edge_history = st.session_state.edge_history[-st.session_state.history_length:]
    
    if len(st.session_state.buffer_history) > st.session_state.history_length:
        st.session_state.buffer_history = st.session_state.buffer_history[-st.session_state.history_length:]
    
    if len(st.session_state.multiplier_history) > st.session_state.history_length:
        st.session_state.multiplier_history = st.session_state.multiplier_history[-st.session_state.history_length:]
    
    if len(st.session_state.fee_history) > st.session_state.history_length:
        st.session_state.fee_history = st.session_state.fee_history[-st.session_state.history_length:]
    
    # Auto-refresh if enabled
    if st.session_state.auto_update and st.session_state.reference_edge is not None:
        time.sleep(0.1)  # Brief pause to prevent excessive refreshing
        st.rerun()

if __name__ == "__main__":
    main()