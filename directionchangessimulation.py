import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time

# Set page config
st.set_page_config(page_title="Price Path Simulator", layout="wide")

# Main title
st.title("Price Path Simulator")

# Sidebar for controls
with st.sidebar:
    direction_changes_pct = st.slider(
        "Direction Changes (%)",
        min_value=5.0,
        max_value=95.0,
        value=45.0,
        step=5.0,
        help="How often the price changes direction"
    )
    
    choppiness = st.slider(
        "Choppiness",
        min_value=100.0,
        max_value=300.0,
        value=150.0,
        step=10.0,
        help="Price oscillation within a range. Higher values = more oscillation"
    )
    
    num_ticks = st.select_slider(
        "Number of Ticks",
        options=[100, 500, 1000, 5000],
        value=1000,
        help="Number of price ticks to simulate"
    )
    
    initial_price = st.number_input(
        "Initial Price",
        min_value=1.0,
        value=100.0,
        step=10.0,
        help="Starting price for all simulations"
    )
    
    num_simulations = st.number_input(
        "Number of Simulations",
        min_value=10,
        max_value=1000,
        value=100,
        step=10,
        help="Number of price paths to simulate"
    )
    
    run_button = st.button("Run Simulation", type="primary")

# Function to simulate a price path
def simulate_price_path(num_ticks, initial_price, direction_change_pct, target_choppiness):
    """Simulates a price path with the given direction change percentage and choppiness."""
    # Convert to proportion
    p_change = direction_change_pct / 100.0
    
    # Initialize
    prices = np.zeros(num_ticks)
    prices[0] = initial_price
    
    # Start with random direction
    current_direction = np.random.choice([-1, 1])
    
    # Window size for measuring local choppiness
    window_size = min(20, num_ticks // 10)
    
    # Base volatility - derived from initial price (we'll adjust dynamically)
    # Start with a conservative value
    base_volatility = initial_price * 0.001
    
    # Scaling factor to gradually reach target choppiness
    choppiness_factor = 1.0
    
    # Generate price path
    for i in range(1, num_ticks):
        # Calculate current choppiness if we have enough points
        if i >= window_size + 5:  # Need a bit more data for stable calculation
            window_prices = prices[max(0, i-window_size):i]
            
            # Calculate current choppiness
            price_range = max(window_prices) - min(window_prices)
            sum_movements = np.sum(np.abs(np.diff(window_prices)))
            
            # Avoid division by zero
            epsilon = 1e-10
            current_choppiness = (sum_movements / (price_range + epsilon) * 100)
            
            # More aggressive adjustment based on the difference
            if current_choppiness > 0:
                # If current is too high, reduce volatility
                if current_choppiness > target_choppiness * 1.05:
                    choppiness_factor = max(choppiness_factor * 0.95, 0.5)
                # If current is too low, increase volatility    
                elif current_choppiness < target_choppiness * 0.95:
                    choppiness_factor = min(choppiness_factor * 1.05, 2.0)
            
            # Apply the adjustment
            volatility = base_volatility * choppiness_factor
        else:
            volatility = base_volatility
        
        # Decide whether to change direction
        if np.random.random() < p_change:
            current_direction *= -1
        
        # Calculate price change
        price_change = volatility * current_direction
        
        # Update price
        prices[i] = prices[i-1] + price_change
        
        # Ensure price stays positive
        prices[i] = max(prices[i], 0.01)
    
    return pd.Series(prices)

# Calculate metrics on a price series
def calculate_metrics(prices):
    """Calculate key metrics for a price path."""
    # Direction changes
    price_changes = prices.diff().dropna()
    signs = np.sign(price_changes)
    direction_changes = (signs.shift(1) != signs).sum()
    total_periods = len(signs) - 1
    direction_change_pct = (direction_changes / total_periods * 100) if total_periods > 0 else 0
    
    # Calculate choppiness
    window = min(14, len(prices) // 10)  # Use a reasonable window size
    diff = prices.diff().abs()
    sum_abs_changes = diff.rolling(window, min_periods=1).sum()
    price_range = prices.rolling(window, min_periods=1).max() - prices.rolling(window, min_periods=1).min()
    
    # Avoid division by zero
    epsilon = 1e-10
    choppiness_values = 100 * sum_abs_changes / (price_range + epsilon)
    
    # Cap extreme values and handle NaN
    choppiness_values = np.minimum(choppiness_values, 1000)
    choppiness_values = choppiness_values.fillna(200)
    choppiness_avg = choppiness_values.mean()
    
    # Median price
    median_price = prices.median()
    
    # Zero crossings (median crossings)
    centered = prices - median_price
    signs = np.sign(centered)
    zero_crossings = (signs.shift(1) != signs).sum()
    
    # Points within +/- 0.5% of median
    upper_bound = median_price * 1.005
    lower_bound = median_price * 0.995
    points_in_range = ((prices >= lower_bound) & (prices <= upper_bound)).sum()
    percentage_in_range = (points_in_range / len(prices)) * 100
    
    return {
        'direction_changes': direction_change_pct,
        'choppiness': choppiness_avg,
        'median_price': median_price,
        'zero_crossings': zero_crossings,
        'points_in_range': percentage_in_range
    }

# Run multiple simulations
def run_simulations(num_simulations, num_ticks, initial_price, direction_changes_pct, choppiness):
    """Run multiple simulations and collect results."""
    paths = []
    metrics = {
        'direction_changes': [],
        'choppiness': [],
        'median_price': [],
        'zero_crossings': [],
        'points_in_range': []
    }
    
    # Progress bar
    progress = st.progress(0)
    
    for i in range(num_simulations):
        # Simulate path
        path = simulate_price_path(num_ticks, initial_price, direction_changes_pct, choppiness)
        paths.append(path)
        
        # Calculate metrics
        path_metrics = calculate_metrics(path)
        for key in metrics:
            metrics[key].append(path_metrics[key])
        
        # Update progress
        progress.progress((i + 1) / num_simulations)
    
    # Clear progress bar
    progress.empty()
    
    return {
        'paths': paths,
        'metrics': metrics
    }

# Main content
if run_button:
    # Start timing
    start_time = time.time()
    
    # Run simulations
    with st.spinner("Running simulations..."):
        results = run_simulations(
            num_simulations,
            num_ticks,
            initial_price,
            direction_changes_pct,
            choppiness
        )
    
    # Calculate execution time
    execution_time = time.time() - start_time
    st.success(f"Completed {num_simulations} simulations in {execution_time:.2f} seconds")
    
    # Display key metrics
    st.header("Key Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Median Final Price", f"${np.median(results['metrics']['median_price']):.2f}")
    
    with col2:
        st.metric("Avg Direction Changes", f"{np.mean(results['metrics']['direction_changes']):.1f}%")
    
    with col3:
        st.metric("Avg Choppiness", f"{np.mean(results['metrics']['choppiness']):.1f}")
    
    with col4:
        st.metric("Avg Zero Crossings", f"{int(np.mean(results['metrics']['zero_crossings']))}")
    
    with col5:
        st.metric("Avg % Within ±0.5% of Median", f"{np.mean(results['metrics']['points_in_range']):.1f}%")
    
    # Display example paths
    st.header("Example Paths")
    
    # Select 5 random paths
    num_examples = min(5, num_simulations)
    example_indices = np.random.choice(num_simulations, size=num_examples, replace=False)
    
    # Create figure
    fig = go.Figure()
    
    for i, idx in enumerate(example_indices):
        path = results['paths'][idx]
        dir_changes = results['metrics']['direction_changes'][idx]
        chop = results['metrics']['choppiness'][idx]
        zero_cross = results['metrics']['zero_crossings'][idx]
        in_range = results['metrics']['points_in_range'][idx]
        
        fig.add_trace(go.Scatter(
            y=path,
            mode='lines',
            name=f"Path {i+1} (DC: {dir_changes:.1f}%, CH: {chop:.1f}, ZC: {zero_cross}, Range: {in_range:.1f}%)"
        ))
    
    fig.update_layout(
        title="Example Price Paths",
        xaxis_title="Tick",
        yaxis_title="Price ($)",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Create metrics table
    st.header("Statistics")
    
    # Create dataframe
    stats_df = pd.DataFrame({
        'Path': range(1, num_simulations + 1),
        'Median Price ($)': results['metrics']['median_price'],
        'Direction Changes (%)': results['metrics']['direction_changes'],
        'Choppiness': results['metrics']['choppiness'],
        'Zero Crossings': results['metrics']['zero_crossings'],
        'Points Within ±0.5% (%)': results['metrics']['points_in_range']
    })
    
    st.dataframe(stats_df, height=400, use_container_width=True)
    
    # Export data option
    st.header("Export Data")
    
    csv = stats_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Statistics CSV",
        data=csv,
        file_name="price_path_statistics.csv",
        mime="text/csv"
    )
    
    # Option to export price paths
    if st.button("Prepare Price Paths for Export"):
        # Create dataframe with all paths
        paths_df = pd.DataFrame({f"Path_{i+1}": path for i, path in enumerate(results['paths'])})
        paths_df.insert(0, 'Tick', range(num_ticks))
        
        csv_paths = paths_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Price Paths CSV",
            data=csv_paths,
            file_name="price_paths.csv",
            mime="text/csv"
        )
else:
    st.info("Set parameters in the sidebar and click 'Run Simulation' to begin.")
    
    st.markdown("""
    ## About This Simulator
    
    This simplified tool generates price paths based on two key parameters:
    
    ### Direction Changes (%)
    Controls how often the price changes direction:
    - Low values (5-30%): Strong trending behavior
    - Medium values (30-60%): Mild trending with some reversals
    - High values (60-95%): Very choppy, oscillating price action
    
    ### Choppiness
    Controls how much the price oscillates within its range:
    - Low values (100-150): Cleaner price movements
    - Medium values (150-200): Moderate oscillation
    - High values (200-300): Significant oscillation within a range
    
    After simulation, you'll see:
    - Median price across all simulations
    - Number of zero crossings (times price crosses the median)
    - Percentage of points within ±0.5% of median price
    """)