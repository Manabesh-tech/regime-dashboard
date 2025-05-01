import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import math
import json
import os
import pytz
from sqlalchemy import create_engine, text

# Page configuration
st.set_page_config(
    page_title="Dynamic House Edge Adjustment Dashboard",
    page_icon="📊",
    layout="wide"
)

# Apply CSS styles to make the dashboard more attractive
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
    .info-box {
        background-color: #e6f7ff;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
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

# Initialize session state variables
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

if 'selected_pair' not in st.session_state:
    st.session_state.selected_pair = None

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

if 'last_update_time' not in st.session_state:
    st.session_state.last_update_time = None

if 'auto_update' not in st.session_state:
    st.session_state.auto_update = False

if 'params_changed' not in st.session_state:
    st.session_state.params_changed = False

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
    st.session_state.lookback_minutes = 10

# Function to calculate fee for a percentage price move
def calculate_fee_for_move(move_pct, buffer_rate, position_multiplier, rate_multiplier=0.5, 
                          base_rate=0.02, bet=1.0, leverage=1.0):
    """
    Calculate fee for a percentage price move using the fee equation.
    
    Fee = -Bet × Leverage × (PT-Pt) × (1 + Rate Multiplier ⋅ |PT-Pt/PT|) / 
          ((1-Base Rate) ⋅ (1 + 10^6 ⋅ Position Multiplier ⋅ |PT-Pt/PT|))
    
    Args:
        move_pct: Price move in percentage (e.g., 0.1 for 0.1%)
        buffer_rate: Buffer rate used in place of base rate
        position_multiplier: Position multiplier parameter
        rate_multiplier: Rate multiplier (default 0.5)
        base_rate: Base rate (default 0.02)
        bet: Bet amount (default 1.0)
        leverage: Leverage used (default 1.0)
        
    Returns:
        Fee amount
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

# Function to update buffer rate based on edge comparison
def update_buffer_rate(current_buffer, edge, edge_ref, max_leverage, alpha_up=0.1, alpha_down=0.02):
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

# Function to update position multiplier based on edge comparison
def update_position_multiplier(current_multiplier, edge, edge_ref, alpha_up=0.02, alpha_down=0.1):
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

# Function to process edge data and calculate parameter updates
def process_edge_data(edge, timestamp=None):
    """
    Process a new edge data point and calculate parameter updates if needed.
    Returns: Tuple of (parameters_changed, updated_buffer, updated_multiplier)
    """
    if timestamp is None:
        timestamp = datetime.now()
    
    # Add to edge history
    st.session_state.edge_history.append((timestamp, edge))
    
    # Use reference edge value for comparison
    edge_ref = st.session_state.reference_edge
    
    # Get current parameters
    current_buffer = st.session_state.buffer_rate
    current_multiplier = st.session_state.position_multiplier
    
    # Get adjustment parameters from sidebar
    max_leverage = st.session_state.max_leverage
    buffer_alpha_up = st.session_state.buffer_alpha_up
    buffer_alpha_down = st.session_state.buffer_alpha_down
    multiplier_alpha_up = st.session_state.multiplier_alpha_up
    multiplier_alpha_down = st.session_state.multiplier_alpha_down
    
    # Calculate updated parameters
    new_buffer_rate = update_buffer_rate(
        current_buffer, edge, edge_ref, max_leverage, 
        buffer_alpha_up, buffer_alpha_down
    )
    
    new_position_multiplier = update_position_multiplier(
        current_multiplier, edge, edge_ref,
        multiplier_alpha_up, multiplier_alpha_down
    )
    
    # Check if parameters changed
    parameters_changed = (abs(new_buffer_rate - current_buffer) > 1e-6 or 
                         abs(new_position_multiplier - current_multiplier) > 1e-6)
    
    # Store the current edge
    st.session_state.current_edge = edge
    
    # Store the proposed parameters
    st.session_state.proposed_buffer_rate = new_buffer_rate
    st.session_state.proposed_position_multiplier = new_position_multiplier
    
    return parameters_changed, new_buffer_rate, new_position_multiplier

# Function to update display parameters (not actual database)
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
    
    # Set initialization flag
    st.session_state.initialized = True
    st.session_state.last_update_time = timestamp
    st.session_state.params_changed = False

# Function to create edge plot using matplotlib
def create_edge_plot():
    """Create a plot of house edge with reference line using matplotlib."""
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
    ax.set_title(f'House Edge Monitoring - {st.session_state.selected_pair}')
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

# Function to create buffer rate plot using matplotlib
def create_buffer_plot():
    """Create a plot of buffer rate history using matplotlib."""
    if len(st.session_state.buffer_history) < 1:
        return None
    
    # Extract data for plotting
    timestamps = [t for t, _ in st.session_state.buffer_history]
    buffer_rates = [r for _, r in st.session_state.buffer_history]
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot buffer rate line
    ax.plot(timestamps, buffer_rates, 'g-', marker='o', label='Buffer Rate')
    
    # Set title and labels
    ax.set_title('Buffer Rate Adjustments')
    ax.set_xlabel('Time')
    ax.set_ylabel('Buffer Rate')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Format dates on x-axis
    fig.autofmt_xdate()
    
    # Tight layout
    plt.tight_layout()
    
    return fig

# Function to create position multiplier plot using matplotlib
def create_multiplier_plot():
    """Create a plot of position multiplier history using matplotlib."""
    if len(st.session_state.multiplier_history) < 1:
        return None
    
    # Extract data for plotting
    timestamps = [t for t, _ in st.session_state.multiplier_history]
    multipliers = [m for _, m in st.session_state.multiplier_history]
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot position multiplier line
    ax.plot(timestamps, multipliers, 'm-', marker='o', label='Position Multiplier')
    
    # Set title and labels
    ax.set_title('Position Multiplier Adjustments')
    ax.set_xlabel('Time')
    ax.set_ylabel('Position Multiplier')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Format dates on x-axis
    fig.autofmt_xdate()
    
    # Tight layout
    plt.tight_layout()
    
    return fig

# Function to create fee plot using matplotlib
def create_fee_plot():
    """Create a plot of fee history for 0.1% price move using matplotlib."""
    if len(st.session_state.fee_history) < 1:
        return None
    
    # Extract data for plotting
    timestamps = [t for t, _ in st.session_state.fee_history]
    fees = [f for _, f in st.session_state.fee_history]
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot fee line
    ax.plot(timestamps, fees, 'r-', marker='o', label='Fee for 0.1% Move')
    
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

# Function to make fee comparison table
def create_fee_comparison_table():
    """Create a table comparing fees for different move sizes with current parameters."""
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

# Function to create sensitivity analysis plot using matplotlib
def create_sensitivity_plot():
    """Create a plot showing fee sensitivity to different parameter combinations."""
    # Create range of parameter values
    buffer_range = np.linspace(0.0001, 0.005, 5)
    multiplier_range = np.linspace(1000, 14000, 5)
    
    # Create a mesh grid
    buffer_mesh, multiplier_mesh = np.meshgrid(buffer_range, multiplier_range)
    
    # Calculate fees for each combination
    fee_mesh = np.zeros_like(buffer_mesh)
    for i in range(buffer_mesh.shape[0]):
        for j in range(buffer_mesh.shape[1]):
            fee_mesh[i, j] = calculate_fee_for_move(0.1, buffer_mesh[i, j], multiplier_mesh[i, j])
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create contour plot
    contour = ax.contourf(buffer_mesh, multiplier_mesh, fee_mesh, 20, cmap='viridis')
    
    # Add colorbar
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label('Fee for 0.1% Move')
    
    # Mark current parameters
    ax.plot(st.session_state.buffer_rate, st.session_state.position_multiplier, 'ro', markersize=10,
           label='Current Parameters')
    
    # Mark proposed parameters if available
    if (st.session_state.proposed_buffer_rate is not None and 
        st.session_state.proposed_position_multiplier is not None):
        ax.plot(st.session_state.proposed_buffer_rate, st.session_state.proposed_position_multiplier, 
               'mo', markersize=10, label='Proposed Parameters')
    
    # Set title and labels
    ax.set_title('Fee Sensitivity to Parameter Combinations')
    ax.set_xlabel('Buffer Rate')
    ax.set_ylabel('Position Multiplier')
    
    # Add legend
    ax.legend()
    
    # Format axes
    ax.set_xlim(min(buffer_range), max(buffer_range))
    ax.set_ylim(min(multiplier_range), max(multiplier_range))
    
    # Tight layout
    plt.tight_layout()
    
    return fig

# Function to create fee curve plot using matplotlib
def create_fee_curve_plot():
    """Create a plot of fee vs price move for current parameters."""
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

def main():
    # Title and description
    st.markdown('<div class="header-style">Dynamic House Edge Adjustment Dashboard</div>', unsafe_allow_html=True)
    st.markdown("""
    This dashboard monitors house edge in real-time and dynamically calculates recommended adjustments for buffer rate 
    and position multiplier parameters to maintain exchange profitability.
    
    Parameters respond asymmetrically: buffer rate increases quickly when edge declines and decreases slowly when edge improves, 
    while position multiplier does the opposite.
    """)
    
    # Singapore time display
    singapore_tz = pytz.timezone('Asia/Singapore')
    now_utc = datetime.now(pytz.utc)
    now_sg = now_utc.astimezone(singapore_tz)
    st.markdown(f"**Current Singapore Time:** {now_sg.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Sidebar for parameters and controls
    st.sidebar.markdown('<div class="subheader-style">Trading Pair Selection</div>', unsafe_allow_html=True)
    
    # Fetch available pairs
    pairs = fetch_pairs()
    
    # Select pair
    selected_pair = st.sidebar.selectbox(
        "Select Trading Pair",
        options=pairs,
        index=0 if st.session_state.selected_pair is None else pairs.index(st.session_state.selected_pair) if st.session_state.selected_pair in pairs else 0
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
    
    # Check if pair changed
    if st.session_state.selected_pair != selected_pair:
        st.session_state.selected_pair = selected_pair
        # Initialize system for new pair
        initialize_system(selected_pair, lookback_minutes)
    
    # Adjustment sensitivity parameters
    st.sidebar.markdown('<div class="subheader-style">Adjustment Sensitivity</div>', unsafe_allow_html=True)
    
    st.session_state.buffer_alpha_up = st.sidebar.slider(
        "Buffer Rate Increase Sensitivity", 
        min_value=0.01, 
        max_value=0.5, 
        value=0.1,
        step=0.01,
        help="How quickly buffer rate increases when edge declines"
    )
    
    st.session_state.buffer_alpha_down = st.sidebar.slider(
        "Buffer Rate Decrease Sensitivity", 
        min_value=0.001, 
        max_value=0.1, 
        value=0.02,
        step=0.001,
        help="How quickly buffer rate decreases when edge improves"
    )
    
    st.session_state.multiplier_alpha_up = st.sidebar.slider(
        "Position Multiplier Increase Sensitivity", 
        min_value=0.001, 
        max_value=0.1, 
        value=0.02,
        step=0.001,
        help="How quickly position multiplier increases when edge improves"
    )
    
    st.session_state.multiplier_alpha_down = st.sidebar.slider(
        "Position Multiplier Decrease Sensitivity", 
        min_value=0.01, 
        max_value=0.5, 
        value=0.1,
        step=0.01,
        help="How quickly position multiplier decreases when edge declines"
    )
    
    # Auto-update toggle
    st.sidebar.markdown('<div class="subheader-style">Monitoring Controls</div>', unsafe_allow_html=True)
    
    st.session_state.auto_update = st.sidebar.checkbox("Auto-update", value=st.session_state.auto_update)
    
    # Calculate update interval in seconds
    update_interval_seconds = lookback_minutes * 60
    
    # For demo purposes, allow faster updates
    if st.sidebar.checkbox("Use Fast Updates for Demo", value=False):
        update_interval_seconds = st.sidebar.slider(
            "Demo Update Seconds", 
            min_value=5, 
            max_value=60, 
            value=15
        )
    
    # History length
    history_length = st.sidebar.slider(
        "History Length (points)", 
        min_value=10, 
        max_value=1000, 
        value=100
    )
    
    # Reset button
    if st.sidebar.button("Reset to Reference Parameters"):
        success, old_buffer, new_buffer, old_multiplier, new_multiplier = reset_to_reference_parameters()
        if success:
            st.sidebar.success("Parameters reset to reference values")
        else:
            st.sidebar.error("No reference parameters available")
    
    # Main content area
    if not st.session_state.initialized:
        st.warning("Initializing system for selected pair...")
        initialize_system(selected_pair, lookback_minutes)
        st.rerun()
    else:
        # Create tabs for different views
        tabs = st.tabs(["Monitoring", "Parameter History", "Fee Analysis"])
        
        # Check if it's time to update
        current_time = datetime.now()
        
        # Determine if we should fetch new data
        should_update = False
        if st.session_state.auto_update:
            if st.session_state.last_update_time is None:
                should_update = True
            else:
                time_since_update = (current_time - st.session_state.last_update_time).total_seconds()
                should_update = time_since_update >= update_interval_seconds
        
        # Fetch and process new edge data if it's time
        if should_update:
            # Fetch new edge data
            new_edge = calculate_edge(st.session_state.selected_pair, lookback_minutes)
            
            if new_edge is not None:
                # Process the data and calculate parameter updates
                params_changed, new_buffer, new_multiplier = process_edge_data(new_edge, current_time)
                
                # Update last update time
                st.session_state.last_update_time = current_time
                
                # Set params_changed flag in session state
                st.session_state.params_changed = params_changed
        
        # Monitoring tab
        with tabs[0]:
            # Show reference edge and parameters
            st.markdown(f"**Reference Values for {st.session_state.selected_pair}:**")
            ref_col1, ref_col2, ref_col3 = st.columns(3)
            
            with ref_col1:
                if st.session_state.reference_edge is not None:
                    st.info(f"Reference Edge: {st.session_state.reference_edge:.4%}")
            
            with ref_col2:
                if st.session_state.reference_buffer_rate is not None:
                    st.info(f"Reference Buffer Rate: {st.session_state.reference_buffer_rate:.6f}")
            
            with ref_col3:
                if st.session_state.reference_position_multiplier is not None:
                    st.info(f"Reference Position Multiplier: {st.session_state.reference_position_multiplier:.1f}")
            
            # Create columns for current metrics
            st.markdown("### Current Values")
            col1, col2, col3 = st.columns(3)
            
            # Show current edge
            with col1:
                if st.session_state.current_edge is not None:
                    edge_delta = st.session_state.current_edge - st.session_state.reference_edge if st.session_state.reference_edge is not None else None
                    st.metric(
                        "Current Edge", 
                        f"{st.session_state.current_edge:.4%}",
                        f"{edge_delta:.4%}" if edge_delta is not None else None,
                        delta_color="inverse"  # Negative delta is bad for edge
                    )
                else:
                    st.metric("Current Edge", "N/A")
            
            # Show current buffer rate
            with col2:
                st.metric("Buffer Rate", f"{st.session_state.buffer_rate:.6f}")
            
            # Show current position multiplier
            with col3:
                st.metric("Position Multiplier", f"{st.session_state.position_multiplier:.1f}")
            
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
                        st.markdown(f"Change: {delta_buffer:.6f} ({delta_buffer/st.session_state.buffer_rate*100:.2f}%)")
                    
                    with col2:
                        st.markdown("**Position Multiplier**")
                        st.markdown(f"Current: {st.session_state.position_multiplier:.1f}")
                        st.markdown(f"Proposed: {st.session_state.proposed_position_multiplier:.1f}")
                        st.markdown(f"Change: {delta_multiplier:.1f} ({delta_multiplier/st.session_state.position_multiplier*100:.2f}%)")
                    
                    with col3:
                        st.markdown("**Fee for 0.1% Move**")
                        st.markdown(f"Current: {current_fee:.8f}")
                        st.markdown(f"Proposed: {new_fee:.8f}")
                        st.markdown(f"Change: {delta_fee:.8f} ({delta_fee/current_fee*100:.2f}%)")
                
                # Update display parameters button
                if st.button("Update Display Parameters", type="primary"):
                    # Apply the updates to display (not database)
                    success, old_buffer, new_buffer, old_multiplier, new_multiplier = update_display_parameters()
                    
                    if success:
                        st.markdown('<div class="success">Display parameters updated successfully!</div>', unsafe_allow_html=True)
                        # Force refresh
                        time.sleep(0.5)
                        st.rerun()
            
            # Manual data fetch button
            if not st.session_state.auto_update:
                if st.button("Fetch New Data", key="manual_fetch"):
                    # Fetch new edge data
                    new_edge = calculate_edge(st.session_state.selected_pair, lookback_minutes)
                    
                    if new_edge is not None:
                        # Process the data and calculate parameter updates
                        params_changed, new_buffer, new_multiplier = process_edge_data(new_edge, current_time)
                        
                        # Update last update time
                        st.session_state.last_update_time = current_time
                        
                        # Set params_changed flag in session state
                        st.session_state.params_changed = params_changed
                        
                        # Force refresh
                        st.rerun()
                    else:
                        st.error(f"Failed to fetch edge data for {st.session_state.selected_pair}")
        
        # Parameter History tab
        with tabs[1]:
            # Create parameter plots
            buffer_fig = create_buffer_plot()
            multiplier_fig = create_multiplier_plot()
            fee_fig = create_fee_plot()
            
            if buffer_fig is not None:
                st.pyplot(buffer_fig)
            else:
                st.info("Not enough data points yet for buffer rate visualization.")
                
            if multiplier_fig is not None:
                st.pyplot(multiplier_fig)
            else:
                st.info("Not enough data points yet for position multiplier visualization.")
                
            if fee_fig is not None:
                st.pyplot(fee_fig)
            else:
                st.info("Not enough data points yet for fee visualization.")
            
            # History data tables
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
            st.markdown("### Fee Analysis with Current Parameters")
            
            # Fee comparison table
            fee_df = create_fee_comparison_table()
            st.dataframe(fee_df, use_container_width=True)
            
            # Create fee curve plot
            fee_curve_fig = create_fee_curve_plot()
            if fee_curve_fig is not None:
                st.pyplot(fee_curve_fig)
            
            # Parameter sensitivity analysis
            st.markdown("### Parameter Sensitivity Analysis")
            
            sensitivity_fig = create_sensitivity_plot()
            if sensitivity_fig is not None:
                st.pyplot(sensitivity_fig)
            
            # Fee equation details
            st.markdown("### Fee Equation")
            st.markdown("""
            The fee is calculated using the following equation:
            
            $$Fee = \\frac{-Bet \\times Leverage \\times (P_T-P_t) \\times (1 + Rate Multiplier \\cdot |\\frac{P_T-P_t}{P_T}|)}{(1-Base Rate) \\cdot (1 + 10^6 \\cdot Position Multiplier \\cdot |\\frac{P_T-P_t}{P_T}|)}$$
            
            Where:
            - The buffer rate is used in place of the base rate
            - For a 0.1% move, PT = 1 and Pt = 0.999
            - The rate multiplier is set to 0.5
            - The base fee rate is set to 0.02
            """)
        
        # Prune history if too long
        if len(st.session_state.edge_history) > history_length:
            st.session_state.edge_history = st.session_state.edge_history[-history_length:]
        
        if len(st.session_state.buffer_history) > history_length:
            st.session_state.buffer_history = st.session_state.buffer_history[-history_length:]
        
        if len(st.session_state.multiplier_history) > history_length:
            st.session_state.multiplier_history = st.session_state.multiplier_history[-history_length:]
        
        if len(st.session_state.fee_history) > history_length:
            st.session_state.fee_history = st.session_state.fee_history[-history_length:]
        
        # Auto-refresh if enabled
        if st.session_state.auto_update:
            time.sleep(0.1)  # Brief pause to prevent excessive refreshing
            st.rerun()

if __name__ == "__main__":
    main()