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
    page_title="Multi-Pair House Edge Monitoring Dashboard",
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
    .metric-positive {
        color: green;
    }
    .metric-negative {
        color: red;
    }
    .stButton>button {
        width: 100%;
    }
    .sensitivity-btn {
        font-size: 12px !important;
        height: 25px !important;
        padding: 0px 8px !important;
        margin: 2px !important;
        background-color: #f0f2f6;
    }
    .small-btn {
        height: 25px !important;
        padding: 0px 8px !important;
        margin: 2px !important;
        font-size: 10px !important;
    }
    .clickable-row {
        cursor: pointer;
    }
    .clickable-row:hover {
        background-color: #f0f2f6 !important;
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

# Initialize or get session state variables
def init_state():
    # Pair monitoring state
    if 'pairs_data' not in st.session_state:
        st.session_state.pairs_data = {}
    
    if 'monitored_pairs' not in st.session_state:
        st.session_state.monitored_pairs = []
    
    if 'selected_pair' not in st.session_state:
        st.session_state.selected_pair = None
    
    if 'last_update_time' not in st.session_state:
        st.session_state.last_update_time = None
    
    if 'auto_update' not in st.session_state:
        st.session_state.auto_update = False
    
    if 'lookback_minutes' not in st.session_state:
        st.session_state.lookback_minutes = 10
    
    if 'view_mode' not in st.session_state:
        st.session_state.view_mode = "table"  # "table" or "detail"
    
    # Default sensitivity values
    if 'default_buffer_alpha_up' not in st.session_state:
        st.session_state.default_buffer_alpha_up = 0.1
    
    if 'default_buffer_alpha_down' not in st.session_state:
        st.session_state.default_buffer_alpha_down = 0.02
    
    if 'default_multiplier_alpha_up' not in st.session_state:
        st.session_state.default_multiplier_alpha_up = 0.02
    
    if 'default_multiplier_alpha_down' not in st.session_state:
        st.session_state.default_multiplier_alpha_down = 0.1
    
    # For tracking parameter changes
    if 'params_changed' not in st.session_state:
        st.session_state.params_changed = {}

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
    
    Fee = -Bet × Leverage × (PT-Pt) × (1 + Rate Multiplier ⋅ |PT-Pt/PT|) / 
          ((1-Base Rate) ⋅ (1 + 10^6 ⋅ Position Multiplier ⋅ |PT-Pt/PT|))
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

# Initialize a new pair in the monitoring system
def initialize_pair(pair_name):
    """Initialize a pair for monitoring."""
    # Fetch current parameters from database
    params = fetch_current_parameters(pair_name)
    
    # Fetch initial edge based on selected lookback period
    initial_edge = calculate_edge(pair_name, st.session_state.lookback_minutes)
    if initial_edge is None:
        initial_edge = 0.001  # Default if calculation fails
    
    # Create timestamp
    timestamp = datetime.now()
    
    # Initialize pair data
    pair_data = {
        # Current parameter values
        "buffer_rate": params["buffer_rate"],
        "position_multiplier": params["position_multiplier"],
        "max_leverage": params["max_leverage"],
        
        # Reference values (initial values)
        "reference_edge": initial_edge,
        "reference_buffer_rate": params["buffer_rate"],
        "reference_position_multiplier": params["position_multiplier"],
        
        # Current edge
        "current_edge": initial_edge,
        
        # Sensitivity parameters (from defaults)
        "buffer_alpha_up": st.session_state.default_buffer_alpha_up,
        "buffer_alpha_down": st.session_state.default_buffer_alpha_down,
        "multiplier_alpha_up": st.session_state.default_multiplier_alpha_up,
        "multiplier_alpha_down": st.session_state.default_multiplier_alpha_down,
        
        # History data
        "edge_history": [(timestamp, initial_edge)],
        "buffer_history": [(timestamp, params["buffer_rate"])],
        "multiplier_history": [(timestamp, params["position_multiplier"])],
        
        # Proposed changes
        "proposed_buffer_rate": None,
        "proposed_position_multiplier": None,
        
        # Fee calculations
        "current_fee": calculate_fee_for_move(0.1, params["buffer_rate"], params["position_multiplier"]),
        "proposed_fee": None,
        
        # Last update time
        "last_update_time": timestamp
    }
    
    # Store in session state
    st.session_state.pairs_data[pair_name] = pair_data
    
    # Add to monitored pairs if not already there
    if pair_name not in st.session_state.monitored_pairs:
        st.session_state.monitored_pairs.append(pair_name)
    
    # Mark that there are no pending parameter changes
    st.session_state.params_changed[pair_name] = False
    
    return pair_data

# Update a pair's edge and calculate proposed parameter changes
def update_pair_edge(pair_name, timestamp=None):
    """
    Update a pair's edge and calculate proposed parameter changes.
    Returns True if parameters need to be changed, False otherwise.
    """
    if timestamp is None:
        timestamp = datetime.now()
    
    # Get pair data
    pair_data = st.session_state.pairs_data.get(pair_name)
    if pair_data is None:
        # Initialize if doesn't exist
        pair_data = initialize_pair(pair_name)
    
    # Fetch new edge
    new_edge = calculate_edge(pair_name, st.session_state.lookback_minutes)
    if new_edge is None:
        return False  # Can't update without a valid edge
    
    # Update edge history
    pair_data["edge_history"].append((timestamp, new_edge))
    pair_data["current_edge"] = new_edge
    
    # Update last update time
    pair_data["last_update_time"] = timestamp
    
    # Get reference edge and current parameters
    edge_ref = pair_data["reference_edge"]
    current_buffer = pair_data["buffer_rate"]
    current_multiplier = pair_data["position_multiplier"]
    max_leverage = pair_data["max_leverage"]
    
    # Get sensitivity parameters for this pair
    buffer_alpha_up = pair_data["buffer_alpha_up"]
    buffer_alpha_down = pair_data["buffer_alpha_down"]
    multiplier_alpha_up = pair_data["multiplier_alpha_up"]
    multiplier_alpha_down = pair_data["multiplier_alpha_down"]
    
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
    
    # Store proposed values
    if parameters_changed:
        pair_data["proposed_buffer_rate"] = new_buffer_rate
        pair_data["proposed_position_multiplier"] = new_position_multiplier
        
        # Calculate proposed fee
        pair_data["proposed_fee"] = calculate_fee_for_move(
            0.1, new_buffer_rate, new_position_multiplier
        )
        
        # Mark that parameters have changed
        st.session_state.params_changed[pair_name] = True
    
    # Save updated pair data
    st.session_state.pairs_data[pair_name] = pair_data
    
    return parameters_changed

# Apply proposed parameter changes for a pair
def apply_pair_changes(pair_name):
    """Apply proposed parameter changes for a pair."""
    # Get pair data
    pair_data = st.session_state.pairs_data.get(pair_name)
    if pair_data is None:
        return False
    
    # Check if we have proposed changes
    if (pair_data["proposed_buffer_rate"] is None or 
        pair_data["proposed_position_multiplier"] is None):
        return False
    
    # Apply changes
    timestamp = datetime.now()
    
    # Update buffer rate
    pair_data["buffer_rate"] = pair_data["proposed_buffer_rate"]
    pair_data["buffer_history"].append((timestamp, pair_data["buffer_rate"]))
    
    # Update position multiplier
    pair_data["position_multiplier"] = pair_data["proposed_position_multiplier"]
    pair_data["multiplier_history"].append((timestamp, pair_data["position_multiplier"]))
    
    # Update current fee
    pair_data["current_fee"] = pair_data["proposed_fee"]
    
    # Clear proposed values
    pair_data["proposed_buffer_rate"] = None
    pair_data["proposed_position_multiplier"] = None
    pair_data["proposed_fee"] = None
    
    # Mark that there are no pending parameter changes
    st.session_state.params_changed[pair_name] = False
    
    # Save updated pair data
    st.session_state.pairs_data[pair_name] = pair_data
    
    return True

# Reset parameters to reference values for a pair
def reset_pair_parameters(pair_name):
    """Reset parameters to reference values for a pair."""
    # Get pair data
    pair_data = st.session_state.pairs_data.get(pair_name)
    if pair_data is None:
        return False
    
    # Apply reset
    timestamp = datetime.now()
    
    # Reset buffer rate
    pair_data["buffer_rate"] = pair_data["reference_buffer_rate"]
    pair_data["buffer_history"].append((timestamp, pair_data["buffer_rate"]))
    
    # Reset position multiplier
    pair_data["position_multiplier"] = pair_data["reference_position_multiplier"]
    pair_data["multiplier_history"].append((timestamp, pair_data["position_multiplier"]))
    
    # Update current fee
    pair_data["current_fee"] = calculate_fee_for_move(
        0.1, pair_data["buffer_rate"], pair_data["position_multiplier"]
    )
    
    # Clear proposed values
    pair_data["proposed_buffer_rate"] = None
    pair_data["proposed_position_multiplier"] = None
    pair_data["proposed_fee"] = None
    
    # Mark that there are no pending parameter changes
    st.session_state.params_changed[pair_name] = False
    
    # Save updated pair data
    st.session_state.pairs_data[pair_name] = pair_data
    
    return True

# Create edge plot for a pair using matplotlib
def create_edge_plot(pair_name):
    """Create a plot of house edge with reference line for a specific pair."""
    # Get pair data
    pair_data = st.session_state.pairs_data.get(pair_name)
    if pair_data is None or len(pair_data["edge_history"]) < 1:
        return None
    
    # Extract data for plotting
    timestamps = [t for t, _ in pair_data["edge_history"]]
    edges = [e for _, e in pair_data["edge_history"]]
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot edge line
    ax.plot(timestamps, edges, 'b-', label='House Edge')
    
    # Add reference line
    ax.axhline(y=pair_data["reference_edge"], color='r', linestyle='--', label='Reference Edge')
    
    # Set title and labels
    ax.set_title(f'House Edge - {pair_name}')
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

# Create parameter plots for a pair using matplotlib
def create_parameter_plots(pair_name):
    """Create plots for buffer rate and position multiplier history for a specific pair."""
    # Get pair data
    pair_data = st.session_state.pairs_data.get(pair_name)
    if pair_data is None:
        return None, None
    
    # Check if we have enough data
    if len(pair_data["buffer_history"]) < 1 or len(pair_data["multiplier_history"]) < 1:
        return None, None
    
    # Extract data for buffer rate plot
    buffer_times = [t for t, _ in pair_data["buffer_history"]]
    buffer_rates = [r for _, r in pair_data["buffer_history"]]
    
    # Create buffer rate plot
    buffer_fig, buffer_ax = plt.subplots(figsize=(10, 5))
    
    # Plot buffer rate line
    buffer_ax.plot(buffer_times, buffer_rates, 'g-', marker='o', label='Buffer Rate')
    
    # Add reference line
    buffer_ax.axhline(y=pair_data["reference_buffer_rate"], color='r', linestyle='--', label='Reference')
    
    # Set title and labels
    buffer_ax.set_title(f'Buffer Rate - {pair_name}')
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
    
    # Extract data for position multiplier plot
    multiplier_times = [t for t, _ in pair_data["multiplier_history"]]
    multipliers = [m for _, m in pair_data["multiplier_history"]]
    
    # Create position multiplier plot
    multiplier_fig, multiplier_ax = plt.subplots(figsize=(10, 5))
    
    # Plot position multiplier line
    multiplier_ax.plot(multiplier_times, multipliers, 'm-', marker='o', label='Position Multiplier')
    
    # Add reference line
    multiplier_ax.axhline(y=pair_data["reference_position_multiplier"], color='r', linestyle='--', label='Reference')
    
    # Set title and labels
    multiplier_ax.set_title(f'Position Multiplier - {pair_name}')
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

# Create fee curve plot for a pair using matplotlib
def create_fee_curve_plot(pair_name):
    """Create a plot of fee vs price move for a specific pair."""
    # Get pair data
    pair_data = st.session_state.pairs_data.get(pair_name)
    if pair_data is None:
        return None
    
    # Calculate fee across a range of move sizes
    move_sizes = np.linspace(-1, 1, 201)
    current_fees = [calculate_fee_for_move(move, pair_data["buffer_rate"], pair_data["position_multiplier"]) 
                   for move in move_sizes]
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot current fee curve
    ax.plot(move_sizes, current_fees, 'b-', label='Current Fee')
    
    # Plot proposed fee curve if available
    if (pair_data["proposed_buffer_rate"] is not None and 
        pair_data["proposed_position_multiplier"] is not None):
        proposed_fees = [calculate_fee_for_move(move, pair_data["proposed_buffer_rate"], 
                                              pair_data["proposed_position_multiplier"]) 
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

# Create a table of all pair data
def create_pairs_table():
    """Create a DataFrame with data for all monitored pairs."""
    # Create list to hold row data
    rows = []
    
    # Add data for each monitored pair
    for pair_name in st.session_state.monitored_pairs:
        pair_data = st.session_state.pairs_data.get(pair_name)
        if pair_data is None:
            continue
        
        # Calculate edge change
        edge_change = pair_data["current_edge"] - pair_data["reference_edge"]
        edge_change_pct = edge_change / abs(pair_data["reference_edge"]) if pair_data["reference_edge"] != 0 else 0
        
        # Check if parameters have changed
        has_changes = st.session_state.params_changed.get(pair_name, False)
        
        # Calculate fee change if proposed changes exist
        fee_change_pct = None
        if pair_data["proposed_fee"] is not None and pair_data["current_fee"] != 0:
            fee_change_pct = (pair_data["proposed_fee"] - pair_data["current_fee"]) / pair_data["current_fee"] * 100
        
        # Create row
        row = {
            "Pair": pair_name,
            "Current Edge": pair_data["current_edge"],
            "Reference Edge": pair_data["reference_edge"],
            "Edge Change %": edge_change_pct * 100,  # Convert to percentage
            "Buffer Rate": pair_data["buffer_rate"],
            "Position Multiplier": pair_data["position_multiplier"],
            "Fee for 0.1%": pair_data["current_fee"],
            "Has Changes": has_changes,
            "Proposed Buffer": pair_data["proposed_buffer_rate"],
            "Proposed Multiplier": pair_data["proposed_position_multiplier"],
            "Proposed Fee": pair_data["proposed_fee"],
            "Fee Change %": fee_change_pct,
            "Last Updated": pair_data["last_update_time"]
        }
        
        rows.append(row)
    
    # Create DataFrame
    if not rows:
        return None
        
    df = pd.DataFrame(rows)
    
    # Sort by edge change (largest negative changes first)
    df = df.sort_values("Edge Change %", ascending=True)
    
    return df

def main():
    # Initialize session state
    init_state()
    
    # Title and description
    st.markdown('<div class="header-style">Multi-Pair House Edge Monitoring Dashboard</div>', unsafe_allow_html=True)
    st.markdown("""
    Monitor house edge across multiple trading pairs and dynamically adjust buffer rate and position multiplier parameters
    to maintain exchange profitability. Click on a pair to view detailed charts and parameter history.
    """)
    
    # Singapore time display
    singapore_tz = pytz.timezone('Asia/Singapore')
    now_utc = datetime.now(pytz.utc)
    now_sg = now_utc.astimezone(singapore_tz)
    st.markdown(f"**Current Singapore Time:** {now_sg.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Sidebar controls
    st.sidebar.markdown('<div class="subheader-style">Monitoring Controls</div>', unsafe_allow_html=True)
    
    # Fetch available pairs
    available_pairs = fetch_pairs()
    
    # Pair selection
    selected_pairs = st.sidebar.multiselect(
        "Select Pairs to Monitor",
        options=available_pairs,
        default=st.session_state.monitored_pairs if st.session_state.monitored_pairs else available_pairs[:3]
    )
    
    # Update monitored pairs list
    st.session_state.monitored_pairs = selected_pairs
    
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
    
    # Auto-update toggle
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
    
    # Default sensitivity parameters
    st.sidebar.markdown('<div class="subheader-style">Default Sensitivity</div>', unsafe_allow_html=True)
    
    st.session_state.default_buffer_alpha_up = st.sidebar.slider(
        "Buffer Rate Increase Sensitivity", 
        min_value=0.01, 
        max_value=0.5, 
        value=0.1,
        step=0.01,
        help="How quickly buffer rate increases when edge declines"
    )
    
    st.session_state.default_buffer_alpha_down = st.sidebar.slider(
        "Buffer Rate Decrease Sensitivity", 
        min_value=0.001, 
        max_value=0.1, 
        value=0.02,
        step=0.001,
        help="How quickly buffer rate decreases when edge improves"
    )
    
    st.session_state.default_multiplier_alpha_up = st.sidebar.slider(
        "Position Multiplier Increase Sensitivity", 
        min_value=0.001, 
        max_value=0.1, 
        value=0.02,
        step=0.001,
        help="How quickly position multiplier increases when edge improves"
    )
    
    st.session_state.default_multiplier_alpha_down = st.sidebar.slider(
        "Position Multiplier Decrease Sensitivity", 
        min_value=0.01, 
        max_value=0.5, 
        value=0.1,
        step=0.01,
        help="How quickly position multiplier decreases when edge declines"
    )
    
    # History length
    history_length = st.sidebar.slider(
        "History Length (points)", 
        min_value=10, 
        max_value=1000, 
        value=100
    )
    
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
    
    # Update all monitored pairs if it's time
    if should_update:
        for pair_name in st.session_state.monitored_pairs:
            update_pair_edge(pair_name, current_time)
        
        # Update last update time
        st.session_state.last_update_time = current_time
    
    # Manual update button
    if not st.session_state.auto_update:
        if st.sidebar.button("Update All Pairs", type="primary"):
            for pair_name in st.session_state.monitored_pairs:
                update_pair_edge(pair_name, current_time)
            
            # Update last update time
            st.session_state.last_update_time = current_time
            
            # Force refresh
            st.rerun()
    
    # Main content
    if st.session_state.view_mode == "table":
        # Multi-pair table view
        st.markdown('<div class="subheader-style">Multi-Pair Monitoring</div>', unsafe_allow_html=True)
        
        # Create table of all pair data
        pairs_df = create_pairs_table()
        
        if pairs_df is not None and not pairs_df.empty:
            # Create a formatted version for display
            display_df = pairs_df.copy()
            
            # Format columns
            display_df["Current Edge"] = display_df["Current Edge"].map(lambda x: f"{x:.4%}" if pd.notnull(x) else "N/A")
            display_df["Reference Edge"] = display_df["Reference Edge"].map(lambda x: f"{x:.4%}" if pd.notnull(x) else "N/A")
            display_df["Edge Change %"] = display_df["Edge Change %"].map(lambda x: f"{x:+.2f}%" if pd.notnull(x) else "N/A")
            display_df["Buffer Rate"] = display_df["Buffer Rate"].map(lambda x: f"{x:.6f}" if pd.notnull(x) else "N/A")
            display_df["Position Multiplier"] = display_df["Position Multiplier"].map(lambda x: f"{x:.1f}" if pd.notnull(x) else "N/A")
            display_df["Fee for 0.1%"] = display_df["Fee for 0.1%"].map(lambda x: f"{x:.8f}" if pd.notnull(x) else "N/A")
            display_df["Has Changes"] = display_df["Has Changes"].map(lambda x: "✓" if x else "")
            
            # Simplify the display columns
            display_columns = [
                "Pair", "Current Edge", "Reference Edge", "Edge Change %", 
                "Buffer Rate", "Position Multiplier", "Fee for 0.1%", "Has Changes"
            ]
            
            # Check if any pairs have changes
            has_any_changes = display_df["Has Changes"].any()
            
            # Show table
            st.write("Click on a pair to view detailed information and charts.")
            
            # Create clickable dataframe with callback
            clicked = None
            if not display_df.empty:
                clicked = st.dataframe(
                    display_df[display_columns],
                    use_container_width=True,
                    column_config={
                        "Edge Change %": st.column_config.NumberColumn(
                            "Edge Change %",
                            format="%.2f%%",
                            help="Percentage change from reference edge"
                        ),
                        "Has Changes": st.column_config.CheckboxColumn(
                            "Has Changes",
                            help="Indicates if parameter updates are available"
                        )
                    }
                )
            
            # Warning for pairs with changes
            if has_any_changes:
                st.warning("One or more pairs have parameter updates available.")
                
                # Get pairs with changes
                pairs_with_changes = pairs_df[pairs_df["Has Changes"]]["Pair"].tolist()
                
                # Create buttons to apply changes for each pair
                st.markdown("### Apply Parameter Updates")
                
                # Create columns for buttons
                cols = st.columns(3)
                
                for i, pair_name in enumerate(pairs_with_changes):
                    with cols[i % 3]:
                        if st.button(f"Update {pair_name}", key=f"update_{pair_name}"):
                            # Apply changes for this pair
                            if apply_pair_changes(pair_name):
                                st.success(f"Parameters updated for {pair_name}")
                                time.sleep(0.5)
                                st.rerun()
            
            # Pair detail view if a pair is selected
            if st.session_state.selected_pair is not None and st.session_state.selected_pair in st.session_state.monitored_pairs:
                st.markdown(f"## Detailed View: {st.session_state.selected_pair}")
                
                # Get pair data
                pair_data = st.session_state.pairs_data.get(st.session_state.selected_pair)
                
                if pair_data is not None:
                    # Create tabs for different views
                    tabs = st.tabs(["Parameter Summary", "Edge History", "Parameter History", "Fee Analysis"])
                    
                    # Parameter Summary tab
                    with tabs[0]:
                        # Create columns for key metrics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("**Edge Values**")
                            st.write(f"Current Edge: {pair_data['current_edge']:.4%}")
                            st.write(f"Reference Edge: {pair_data['reference_edge']:.4%}")
                            edge_delta = pair_data['current_edge'] - pair_data['reference_edge']
                            edge_delta_pct = edge_delta / abs(pair_data['reference_edge']) if pair_data['reference_edge'] != 0 else 0
                            st.write(f"Edge Change: {edge_delta:.4%} ({edge_delta_pct*100:+.2f}%)")
                        
                        with col2:
                            st.markdown("**Current Parameters**")
                            st.write(f"Buffer Rate: {pair_data['buffer_rate']:.6f}")
                            st.write(f"Position Multiplier: {pair_data['position_multiplier']:.1f}")
                            st.write(f"Max Leverage: {pair_data['max_leverage']}")
                        
                        with col3:
                            st.markdown("**Sensitivity Settings**")
                            
                            # Create a mini-form for updating sensitivity
                            cols = st.columns([3, 1, 1])
                            
                            # Buffer Rate Up
                            with cols[0]:
                                st.write("Buffer Up:")
                            with cols[1]:
                                if st.button("-", key=f"buf_up_down_{st.session_state.selected_pair}", help="Decrease sensitivity"):
                                    pair_data['buffer_alpha_up'] = max(0.01, pair_data['buffer_alpha_up'] - 0.01)
                                    st.session_state.pairs_data[st.session_state.selected_pair] = pair_data
                                    st.rerun()
                            with cols[2]:
                                if st.button("+", key=f"buf_up_up_{st.session_state.selected_pair}", help="Increase sensitivity"):
                                    pair_data['buffer_alpha_up'] = min(0.5, pair_data['buffer_alpha_up'] + 0.01)
                                    st.session_state.pairs_data[st.session_state.selected_pair] = pair_data
                                    st.rerun()
                            
                            st.write(f"Value: {pair_data['buffer_alpha_up']:.2f}")
                            
                            # Buffer Rate Down
                            cols = st.columns([3, 1, 1])
                            with cols[0]:
                                st.write("Buffer Down:")
                            with cols[1]:
                                if st.button("-", key=f"buf_down_down_{st.session_state.selected_pair}", help="Decrease sensitivity"):
                                    pair_data['buffer_alpha_down'] = max(0.001, pair_data['buffer_alpha_down'] - 0.001)
                                    st.session_state.pairs_data[st.session_state.selected_pair] = pair_data
                                    st.rerun()
                            with cols[2]:
                                if st.button("+", key=f"buf_down_up_{st.session_state.selected_pair}", help="Increase sensitivity"):
                                    pair_data['buffer_alpha_down'] = min(0.1, pair_data['buffer_alpha_down'] + 0.001)
                                    st.session_state.pairs_data[st.session_state.selected_pair] = pair_data
                                    st.rerun()
                            
                            st.write(f"Value: {pair_data['buffer_alpha_down']:.3f}")
                            
                            # Position Multiplier Up
                            cols = st.columns([3, 1, 1])
                            with cols[0]:
                                st.write("Position Up:")
                            with cols[1]:
                                if st.button("-", key=f"pos_up_down_{st.session_state.selected_pair}", help="Decrease sensitivity"):
                                    pair_data['multiplier_alpha_up'] = max(0.001, pair_data['multiplier_alpha_up'] - 0.001)
                                    st.session_state.pairs_data[st.session_state.selected_pair] = pair_data
                                    st.rerun()
                            with cols[2]:
                                if st.button("+", key=f"pos_up_up_{st.session_state.selected_pair}", help="Increase sensitivity"):
                                    pair_data['multiplier_alpha_up'] = min(0.1, pair_data['multiplier_alpha_up'] + 0.001)
                                    st.session_state.pairs_data[st.session_state.selected_pair] = pair_data
                                    st.rerun()
                            
                            st.write(f"Value: {pair_data['multiplier_alpha_up']:.3f}")
                            
                            # Position Multiplier Down
                            cols = st.columns([3, 1, 1])
                            with cols[0]:
                                st.write("Position Down:")
                            with cols[1]:
                                if st.button("-", key=f"pos_down_down_{st.session_state.selected_pair}", help="Decrease sensitivity"):
                                    pair_data['multiplier_alpha_down'] = max(0.01, pair_data['multiplier_alpha_down'] - 0.01)
                                    st.session_state.pairs_data[st.session_state.selected_pair] = pair_data
                                    st.rerun()
                            with cols[2]:
                                if st.button("+", key=f"pos_down_up_{st.session_state.selected_pair}", help="Increase sensitivity"):
                                    pair_data['multiplier_alpha_down'] = min(0.5, pair_data['multiplier_alpha_down'] + 0.01)
                                    st.session_state.pairs_data[st.session_state.selected_pair] = pair_data
                                    st.rerun()
                            
                            st.write(f"Value: {pair_data['multiplier_alpha_down']:.2f}")
                        
                        # Add buttons for actions
                        st.markdown("### Actions")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if st.button("Update Edge", key=f"update_edge_{st.session_state.selected_pair}"):
                                update_pair_edge(st.session_state.selected_pair)
                                st.rerun()
                        
                        with col2:
                            if st.session_state.params_changed.get(st.session_state.selected_pair, False):
                                if st.button("Apply Changes", key=f"apply_{st.session_state.selected_pair}"):
                                    apply_pair_changes(st.session_state.selected_pair)
                                    st.rerun()
                        
                        with col3:
                            if st.button("Reset Parameters", key=f"reset_{st.session_state.selected_pair}"):
                                reset_pair_parameters(st.session_state.selected_pair)
                                st.rerun()
                        
                        # Show proposed changes if available
                        if st.session_state.params_changed.get(st.session_state.selected_pair, False):
                            st.markdown("### Proposed Parameter Changes")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.markdown("**Buffer Rate**")
                                st.write(f"Current: {pair_data['buffer_rate']:.6f}")
                                st.write(f"Proposed: {pair_data['proposed_buffer_rate']:.6f}")
                                buffer_change = pair_data['proposed_buffer_rate'] - pair_data['buffer_rate']
                                buffer_change_pct = buffer_change / pair_data['buffer_rate'] * 100
                                st.write(f"Change: {buffer_change:.6f} ({buffer_change_pct:+.2f}%)")
                            
                            with col2:
                                st.markdown("**Position Multiplier**")
                                st.write(f"Current: {pair_data['position_multiplier']:.1f}")
                                st.write(f"Proposed: {pair_data['proposed_position_multiplier']:.1f}")
                                multiplier_change = pair_data['proposed_position_multiplier'] - pair_data['position_multiplier']
                                multiplier_change_pct = multiplier_change / pair_data['position_multiplier'] * 100
                                st.write(f"Change: {multiplier_change:.1f} ({multiplier_change_pct:+.2f}%)")
                            
                            with col3:
                                st.markdown("**Fee for 0.1% Move**")
                                st.write(f"Current: {pair_data['current_fee']:.8f}")
                                st.write(f"Proposed: {pair_data['proposed_fee']:.8f}")
                                fee_change = pair_data['proposed_fee'] - pair_data['current_fee']
                                fee_change_pct = fee_change / pair_data['current_fee'] * 100 if pair_data['current_fee'] != 0 else 0
                                st.write(f"Change: {fee_change:.8f} ({fee_change_pct:+.2f}%)")
                    
                    # Edge History tab
                    with tabs[1]:
                        # Create edge plot
                        edge_fig = create_edge_plot(st.session_state.selected_pair)
                        if edge_fig is not None:
                            st.pyplot(edge_fig)
                        else:
                            st.info("Not enough data points yet for edge visualization.")
                        
                        # Show edge history data
                        if len(pair_data["edge_history"]) > 0:
                            st.markdown("### Edge History Data")
                            edge_df = pd.DataFrame({
                                'Timestamp': [t for t, _ in pair_data["edge_history"]],
                                'Edge': [e for _, e in pair_data["edge_history"]]
                            })
                            # Format edge as percentage
                            edge_df['Edge'] = edge_df['Edge'].map(lambda x: f"{x:.4%}")
                            st.dataframe(edge_df, use_container_width=True)
                    
                    # Parameter History tab
                    with tabs[2]:
                        # Create parameter plots
                        buffer_fig, multiplier_fig = create_parameter_plots(st.session_state.selected_pair)
                        
                        if buffer_fig is not None:
                            st.pyplot(buffer_fig)
                        else:
                            st.info("Not enough data points yet for buffer rate visualization.")
                            
                        if multiplier_fig is not None:
                            st.pyplot(multiplier_fig)
                        else:
                            st.info("Not enough data points yet for position multiplier visualization.")
                        
                        # Show parameter history data
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if len(pair_data["buffer_history"]) > 0:
                                st.markdown("### Buffer Rate History")
                                buffer_df = pd.DataFrame({
                                    'Timestamp': [t for t, _ in pair_data["buffer_history"]],
                                    'Buffer Rate': [r for _, r in pair_data["buffer_history"]]
                                })
                                st.dataframe(buffer_df, use_container_width=True)
                        
                        with col2:
                            if len(pair_data["multiplier_history"]) > 0:
                                st.markdown("### Position Multiplier History")
                                multiplier_df = pd.DataFrame({
                                    'Timestamp': [t for t, _ in pair_data["multiplier_history"]],
                                    'Position Multiplier': [m for _, m in pair_data["multiplier_history"]]
                                })
                                st.dataframe(multiplier_df, use_container_width=True)
                    
                    # Fee Analysis tab
                    with tabs[3]:
                        st.markdown("### Fee Analysis")
                        
                        # Create fee curve plot
                        fee_fig = create_fee_curve_plot(st.session_state.selected_pair)
                        if fee_fig is not None:
                            st.pyplot(fee_fig)
                        
                        # Fee comparison table
                        st.markdown("### Fee for Different Move Sizes")
                        
                        # Calculate fees for different move sizes
                        move_sizes = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
                        
                        # Calculate current fees
                        current_fees = [calculate_fee_for_move(move, pair_data["buffer_rate"], pair_data["position_multiplier"]) 
                                      for move in move_sizes]
                        
                        # Create dataframe for fee table
                        if pair_data["proposed_buffer_rate"] is not None and pair_data["proposed_position_multiplier"] is not None:
                            # Calculate proposed fees
                            proposed_fees = [calculate_fee_for_move(move, pair_data["proposed_buffer_rate"], pair_data["proposed_position_multiplier"]) 
                                            for move in move_sizes]
                            
                            # Create dataframe with both current and proposed fees
                            fee_df = pd.DataFrame({
                                'Move Size (%)': move_sizes,
                                'Current Fee': current_fees,
                                'Proposed Fee': proposed_fees,
                                'Difference (%)': [(new - old) / old * 100 if old != 0 else 0 
                                                for new, old in zip(proposed_fees, current_fees)]
                            })
                        else:
                            # Create dataframe with just current fees
                            fee_df = pd.DataFrame({
                                'Move Size (%)': move_sizes,
                                'Current Fee': current_fees
                            })
                        
                        # Display fee table
                        st.dataframe(fee_df, use_container_width=True)
                        
                        # Fee equation explanation
                        st.markdown("### Fee Equation")
                        st.markdown("""
                        The fee is calculated using the following equation:
                        
                        $$Fee = \\frac{-Bet \\times Leverage \\times (P_T-P_t) \\times (1 + Rate Multiplier \\cdot |\\frac{P_T-P_t}{P_T}|)}{(1-Base Rate) \\cdot (1 + 10^6 \\cdot Position Multiplier \\cdot |\\frac{P_T-P_t}{P_T}|)}$$
                        """)
                
                # Button to return to table view
                if st.button("Return to Table View"):
                    st.session_state.selected_pair = None
                    st.session_state.view_mode = "table"
                    st.rerun()
            
            # Initialize any uninitialized pairs
            for pair_name in st.session_state.monitored_pairs:
                if pair_name not in st.session_state.pairs_data:
                    initialize_pair(pair_name)
            
            # Use expander for initial fetch and debug buttons
            with st.expander("Advanced Options"):
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Initialize/Reset All Pairs"):
                        for pair_name in st.session_state.monitored_pairs:
                            initialize_pair(pair_name)
                        st.rerun()
                
                with col2:
                    if st.button("Apply All Pending Changes"):
                        for pair_name in st.session_state.monitored_pairs:
                            if st.session_state.params_changed.get(pair_name, False):
                                apply_pair_changes(pair_name)
                        st.rerun()
                        
                # Show last update time        
                if st.session_state.last_update_time:
                    last_update_sg = st.session_state.last_update_time.astimezone(singapore_tz) if st.session_state.last_update_time.tzinfo else pytz.utc.localize(st.session_state.last_update_time).astimezone(singapore_tz)
                    st.write(f"Last data update: {last_update_sg.strftime('%Y-%m-%d %H:%M:%S')} (SG time)")
        else:
            st.info("No pairs are currently being monitored. Please select pairs from the sidebar.")
    
    # Check for row clicks to switch to detail view
    if st.session_state.view_mode == "table" and st.session_state.selected_pair is None:
        # Placeholder for pair selection
        # In a real implementation, we would handle clicks on the table
        # For now, we'll use a selectbox
        selected_detail_pair = st.selectbox(
            "Select a pair to view details:",
            options=["None"] + st.session_state.monitored_pairs,
            index=0,
            key="detail_pair_select"
        )
        
        if selected_detail_pair != "None":
            st.session_state.selected_pair = selected_detail_pair
            st.rerun()
    
    # Prune history if needed
    for pair_name in st.session_state.pairs_data:
        pair_data = st.session_state.pairs_data[pair_name]
        
        if len(pair_data["edge_history"]) > history_length:
            pair_data["edge_history"] = pair_data["edge_history"][-history_length:]
        
        if len(pair_data["buffer_history"]) > history_length:
            pair_data["buffer_history"] = pair_data["buffer_history"][-history_length:]
        
        if len(pair_data["multiplier_history"]) > history_length:
            pair_data["multiplier_history"] = pair_data["multiplier_history"][-history_length:]
        
        st.session_state.pairs_data[pair_name] = pair_data
    
    # Auto-refresh if enabled
    if st.session_state.auto_update:
        time.sleep(0.1)  # Brief pause to prevent excessive refreshing
        st.rerun()

if __name__ == "__main__":
    main()