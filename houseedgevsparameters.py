import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pytz

# Page configuration
st.set_page_config(
    page_title="House Edge Adjustment Dashboard",
    page_icon="📊",
    layout="wide"
)

# Apply basic CSS styles
st.markdown("""
<style>
    .header-style {
        font-size:24px !important;
        font-weight: bold;
        padding: 10px 0;
    }
    .success {
        background-color: #e6ffe6;
        padding: 10px;
        border-radius: 5px;
    }
    .warning {
        background-color: #fff3cd;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'monitored_pairs' not in st.session_state:
    st.session_state.monitored_pairs = []

if 'pair_data' not in st.session_state:
    st.session_state.pair_data = {}

if 'view_mode' not in st.session_state:
    st.session_state.view_mode = "Pairs Overview"

if 'current_pair' not in st.session_state:
    st.session_state.current_pair = None

# Get Singapore time
def get_sg_time():
    """Get current time in Singapore timezone."""
    singapore_tz = pytz.timezone('Asia/Singapore')
    now_utc = datetime.now(pytz.utc)
    now_sg = now_utc.astimezone(singapore_tz)
    return now_sg

# Initialize a pair
def initialize_pair(pair_name):
    """Initialize a trading pair with simulated data."""
    if pair_name not in st.session_state.pair_data:
        st.session_state.pair_data[pair_name] = {
            'initialized': True,
            'buffer_rate': 0.001,
            'pnl_base_rate': 0.1,
            'position_multiplier': 1000,
            'max_leverage': 100,
            'rate_multiplier': 15000,
            'rate_exponent': 1,
            'edge_history': [(get_sg_time(), 0.002)],
            'current_edge': 0.002,
            'reference_edge': 0.002,
            'last_update_time': get_sg_time(),
            'params_changed': False,
            'current_fee_percentage': 25.0
        }
    return True

# Set view mode
def set_view(mode, pair=None):
    """Set the view mode and current pair."""
    st.session_state.view_mode = mode
    if pair:
        st.session_state.current_pair = pair
    st.experimental_rerun()

# Main application
def main():
    # Title and description
    st.markdown('<div class="header-style">House Edge Parameter Adjustment Dashboard</div>', unsafe_allow_html=True)
    st.markdown("""
    This is a simplified version of the dashboard that monitors house edge and adjusts parameters.
    """)
    
    # Singapore time display
    now_sg = get_sg_time()
    st.markdown(f"**Current Singapore Time:** {now_sg.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Sidebar controls
    st.sidebar.title("Dashboard Controls")
    
    # Predefined pairs for testing
    pairs = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT", "XRP/USDT"]
    
    # Select pair for initialization
    selected_pair = st.sidebar.selectbox(
        "Select Trading Pair",
        options=pairs,
        index=0,
        key="pair_selector"
    )
    
    # Initialize button
    if st.sidebar.button("Add Pair", key="init_button"):
        if selected_pair not in st.session_state.monitored_pairs:
            initialize_pair(selected_pair)
            st.session_state.monitored_pairs.append(selected_pair)
            st.sidebar.success(f"Added {selected_pair} to monitoring")
            st.experimental_rerun()
        else:
            st.sidebar.info(f"{selected_pair} is already being monitored")
    
    # Main content area
    if st.session_state.view_mode == "Pairs Overview":
        render_overview()
    elif st.session_state.view_mode == "Pair Detail":
        render_detail(st.session_state.current_pair)
    else:
        st.session_state.view_mode = "Pairs Overview"
        render_overview()

# Render overview page
def render_overview():
    if not st.session_state.monitored_pairs:
        st.info("No pairs are currently being monitored. Select a pair and click 'Add Pair' in the sidebar.")
        return
    
    st.markdown("### Monitored Trading Pairs")
    st.markdown("Select a pair below to view detailed analytics.")
    
    # Create a grid layout for pair cards (3 columns)
    columns = st.columns(3)
    
    # Render a card for each monitored pair
    for i, pair_name in enumerate(st.session_state.monitored_pairs):
        with columns[i % 3]:
            pair_data = st.session_state.pair_data.get(pair_name, {})
            current_edge = pair_data.get('current_edge', 0)
            reference_edge = pair_data.get('reference_edge', 0)
            
            # Create a simple card
            with st.container():
                st.subheader(pair_name)
                st.markdown(f"**Current Edge:** {current_edge:.4%}")
                st.markdown(f"**Reference Edge:** {reference_edge:.4%}")
                
                # View button
                if st.button(f"View Details", key=f"view_{pair_name}"):
                    set_view("Pair Detail", pair_name)

# Render detail page
def render_detail(pair_name):
    if pair_name not in st.session_state.pair_data:
        st.warning("Pair data not found.")
        return
    
    st.markdown(f"### Detailed Analytics: {pair_name}")
    
    pair_data = st.session_state.pair_data[pair_name]
    
    # Display key metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Current Edge:** {pair_data['current_edge']:.4%}")
        st.markdown(f"**Buffer Rate:** {pair_data['buffer_rate']:.6f}")
        st.markdown(f"**Fee for 0.1% Move:** {pair_data['current_fee_percentage']:.2f}%")
    
    with col2:
        st.markdown(f"**Reference Edge:** {pair_data['reference_edge']:.4%}")
        st.markdown(f"**Position Multiplier:** {pair_data['position_multiplier']:.1f}")
        st.markdown(f"**Last Update:** {pair_data['last_update_time'].strftime('%H:%M:%S')}")
    
    # Create a simple plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot([0, 1, 2], [0.002, 0.0025, 0.0022], 'b-', label='House Edge')
    ax.axhline(y=0.002, color='r', linestyle='--', label='Reference Edge')
    ax.set_title(f'House Edge Monitoring - {pair_name}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Edge')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.2%}'.format(x)))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    st.pyplot(fig)
    
    # Return button
    if st.button("Return to Pairs Overview"):
        set_view("Pairs Overview")

# Run the app
if __name__ == "__main__":
    main()