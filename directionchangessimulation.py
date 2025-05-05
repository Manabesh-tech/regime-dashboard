import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# Set page config
st.set_page_config(
    page_title="Price Path Simulator",
    page_icon="📈",
    layout="wide"
)

# Main title and description
st.title("Price Path Simulator")
st.markdown("""
This tool simulates cryptocurrency price paths based on key market behavior metrics.
Simply adjust the parameters and generate 1,000 realistic price paths.
""")

# Sidebar for controls
with st.sidebar:
    st.header("Simulation Parameters")
    
    direction_changes_pct = st.slider(
        "Direction Changes (%)",
        min_value=5.0,
        max_value=95.0,
        value=45.0,
        step=5.0,
        help="Percentage of times the price changes direction. 50% is random walk, lower is more trending, higher is more reversal."
    )
    
    choppiness = st.slider(
        "Choppiness",
        min_value=100.0,
        max_value=300.0,
        value=150.0,
        step=10.0,
        help="Measures price oscillation within a range. Higher values indicate more oscillation."
    )
    
    trend_strength = st.slider(
        "Trend Strength",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.1,
        help="Measures how effectively price moves in a consistent direction. Lower values indicate more oscillation without net progress."
    )
    
    st.markdown("---")
    
    st.subheader("Additional Settings")
    
    num_ticks = st.select_slider(
        "Number of Ticks",
        options=[100, 500, 1000, 2500, 5000],
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
        max_value=5000,
        value=1000,
        step=100,
        help="Number of price paths to simulate"
    )
    
    run_button = st.button("Run Simulation", type="primary", use_container_width=True)

# Function to simulate a price path with target metrics
def simulate_price_path(num_ticks, initial_price, target_dir_change_pct, target_choppiness, target_trend_strength):
    """Simulates a price path with target metrics."""
    # Convert percentages to proportions
    target_dir_change = target_dir_change_pct / 100.0
    
    # Initialization
    prices = np.zeros(num_ticks)
    prices[0] = initial_price
    
    # Initialize with a random direction
    current_direction = np.random.choice([-1, 1])
    
    # Window size for calculating running metrics
    window_size = min(20, num_ticks // 10)
    
    # Base volatility (derived from the target metrics)
    base_volatility = initial_price * 0.002 * (target_choppiness / 150) * (1 / target_trend_strength)
    
    # Generate subsequent prices
    for i in range(1, num_ticks):
        # Calculate current metrics if we have enough data points
        if i >= window_size:
            window_prices = prices[max(0, i-window_size):i]
            
            # Calculate current direction changes
            price_changes = np.diff(window_prices)
            signs = np.sign(price_changes)
            direction_changes = (signs[1:] != signs[:-1]).sum()
            current_dir_change = direction_changes / (len(signs) - 1) if len(signs) > 1 else 0.5
            
            # Calculate current choppiness
            price_range = max(window_prices) - min(window_prices)
            sum_movements = np.sum(np.abs(np.diff(window_prices)))
            current_choppiness = (sum_movements / price_range * 100) if price_range > 0 else 150
            
            # Calculate current trend strength
            net_change = abs(window_prices[-1] - window_prices[0])
            current_trend_strength = (net_change / sum_movements) if sum_movements > 0 else 0.5
            
            # Adjust probabilities based on current metrics vs target metrics
            dir_change_adjustment = 0.1 * (target_dir_change - current_dir_change)
            trend_adjustment = 0.1 * (current_trend_strength - target_trend_strength)
            
            # Choppiness adjustment
            choppiness_ratio = target_choppiness / current_choppiness if current_choppiness > 0 else 1
            volatility_adjustment = min(max(choppiness_ratio * 0.5, 0.5), 1.5)
            
            # Combine adjustments
            p_change = max(0.01, min(0.99, target_dir_change + dir_change_adjustment - trend_adjustment))
            adjusted_volatility = base_volatility * volatility_adjustment
        else:
            # Default values for early iterations
            p_change = target_dir_change
            adjusted_volatility = base_volatility
        
        # Decide whether to change direction
        if np.random.random() < p_change:
            current_direction *= -1
        
        # Calculate the price change
        price_change = adjusted_volatility * current_direction
        
        # Update price
        prices[i] = prices[i-1] + price_change
        
        # Ensure price stays positive
        prices[i] = max(prices[i], 0.01)
    
    return pd.Series(prices)

# Functions to calculate metrics on a price series
def calculate_direction_changes(prices):
    """Calculate the percentage of times the price direction changes."""
    price_changes = prices.diff().dropna()
    signs = np.sign(price_changes)
    direction_changes = (signs.shift(1) != signs).sum()
    
    total_periods = len(signs) - 1
    if total_periods > 0:
        direction_change_pct = (direction_changes / total_periods) * 100
    else:
        direction_change_pct = 0
    
    return direction_change_pct

def count_zero_crossings(prices, reference_price=None):
    """Count how many times the price crosses the reference price (median by default)."""
    if reference_price is None:
        reference_price = prices.median()
    
    # Subtract reference price to center around zero
    centered = prices - reference_price
    
    # Count sign changes (zero crossings)
    signs = np.sign(centered)
    crossings = (signs.shift(1) != signs).sum()
    
    return crossings

def calculate_price_range_percentage(prices, percentage=0.5):
    """Calculate percentage of points within +/- x% of median price."""
    median_price = prices.median()
    upper_bound = median_price * (1 + percentage/100)
    lower_bound = median_price * (1 - percentage/100)
    
    # Count points within range
    points_in_range = ((prices >= lower_bound) & (prices <= upper_bound)).sum()
    
    # Calculate percentage
    percentage_in_range = (points_in_range / len(prices)) * 100
    
    return percentage_in_range

# Function to run multiple simulations and return statistics
def run_multiple_simulations(num_simulations, num_ticks, initial_price, 
                             target_dir_change_pct, target_choppiness, target_trend_strength):
    """Run multiple simulations and collect statistics."""
    all_paths = []
    direction_changes_actual = []
    median_prices = []
    zero_crossings = []
    range_percentages = []
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(num_simulations):
        # Update progress
        progress = int((i + 1) / num_simulations * 100)
        progress_bar.progress(progress)
        status_text.text(f"Simulating path {i+1}/{num_simulations} ({progress}%)")
        
        # Simulate path
        path = simulate_price_path(
            num_ticks, 
            initial_price, 
            target_dir_change_pct, 
            target_choppiness, 
            target_trend_strength
        )
        all_paths.append(path)
        
        # Calculate metrics
        direction_changes_actual.append(calculate_direction_changes(path))
        
        # Calculate median price
        median_price = path.median()
        median_prices.append(median_price)
        
        # Calculate zero crossings (median crossings)
        zero_crossings.append(count_zero_crossings(path, median_price))
        
        # Calculate percentage within +/- 0.5% of median
        range_percentages.append(calculate_price_range_percentage(path, 0.5))
    
    # Clear progress bar
    progress_bar.empty()
    status_text.empty()
    
    # Combine all statistics
    results = {
        'paths': all_paths,
        'direction_changes': direction_changes_actual,
        'median_prices': median_prices,
        'zero_crossings': zero_crossings,
        'range_percentages': range_percentages
    }
    
    return results

# Main interface
tab1, tab2, tab3 = st.tabs(["Example Paths", "Statistics", "Export Data"])

# Run simulation when button is clicked
if run_button:
    # Show a spinner while running simulations
    with st.spinner("Running simulations..."):
        start_time = time.time()
        results = run_multiple_simulations(
            num_simulations=num_simulations,
            num_ticks=num_ticks,
            initial_price=initial_price,
            target_dir_change_pct=direction_changes_pct,
            target_choppiness=choppiness,
            target_trend_strength=trend_strength
        )
        execution_time = time.time() - start_time
    
    # Store results in session state for tab access
    st.session_state.simulation_results = results
    st.session_state.simulation_params = {
        'direction_changes_pct': direction_changes_pct,
        'choppiness': choppiness,
        'trend_strength': trend_strength,
        'num_ticks': num_ticks,
        'initial_price': initial_price,
        'num_simulations': num_simulations,
        'execution_time': execution_time
    }
    
    # Key metrics at the top
    st.header("Simulation Results")
    
    # Display key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Median Final Price",
            f"${np.median(results['median_prices']):.2f}"
        )
    
    with col2:
        st.metric(
            "Avg Direction Changes",
            f"{np.mean(results['direction_changes']):.1f}%"
        )
    
    with col3:
        st.metric(
            "Avg Zero Crossings",
            f"{np.mean(results['zero_crossings']):.1f}"
        )
    
    with col4:
        st.metric(
            "Avg % Within ±0.5% of Median",
            f"{np.mean(results['range_percentages']):.1f}%"
        )
    
    # Tab 1: Example Paths
    with tab1:
        st.header("Example Price Paths")
        
        # Controls for visualization
        num_example_paths = st.slider(
            "Number of paths to show",
            min_value=1,
            max_value=20,
            value=5
        )
        
        # Random selection of paths
        random_indices = np.random.choice(len(results['paths']), size=min(num_example_paths, len(results['paths'])), replace=False)
        selected_paths = [results['paths'][i] for i in random_indices]
        selected_metrics = [
            (results['direction_changes'][i], results['zero_crossings'][i], results['range_percentages'][i]) 
            for i in random_indices
        ]
        
        # Create visualization with metrics in legend
        fig = go.Figure()
        
        for i, path in enumerate(selected_paths):
            dir_change, crossings, range_pct = selected_metrics[i]
            fig.add_trace(go.Scatter(
                y=path,
                mode='lines',
                name=f"Path {i+1} (DC: {dir_change:.1f}%, ZC: {crossings}, Range: {range_pct:.1f}%)"
            ))
        
        fig.update_layout(
            title="Example Price Paths",
            xaxis_title="Tick",
            yaxis_title="Price ($)",
            height=500,
            legend_title="Path (Metrics)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Examples of extreme cases
        st.subheader("Examples of Extreme Cases")
        
        # Find indices for extreme cases
        high_dir_change_idx = np.argmax(results['direction_changes'])
        low_dir_change_idx = np.argmin(results['direction_changes'])
        high_crossings_idx = np.argmax(results['zero_crossings'])
        low_crossings_idx = np.argmin(results['zero_crossings'])
        
        # Create a subplot grid
        extreme_fig = make_subplots(
            rows=2, 
            cols=2,
            subplot_titles=[
                f"Highest Direction Changes: {results['direction_changes'][high_dir_change_idx]:.1f}%",
                f"Lowest Direction Changes: {results['direction_changes'][low_dir_change_idx]:.1f}%",
                f"Most Zero Crossings: {results['zero_crossings'][high_crossings_idx]}",
                f"Fewest Zero Crossings: {results['zero_crossings'][low_crossings_idx]}"
            ],
            vertical_spacing=0.1
        )
        
        # Add each extreme case
        extreme_fig.add_trace(
            go.Scatter(y=results['paths'][high_dir_change_idx], mode='lines', line=dict(color='red')),
            row=1, col=1
        )
        extreme_fig.add_trace(
            go.Scatter(y=results['paths'][low_dir_change_idx], mode='lines', line=dict(color='blue')),
            row=1, col=2
        )
        extreme_fig.add_trace(
            go.Scatter(y=results['paths'][high_crossings_idx], mode='lines', line=dict(color='purple')),
            row=2, col=1
        )
        extreme_fig.add_trace(
            go.Scatter(y=results['paths'][low_crossings_idx], mode='lines', line=dict(color='green')),
            row=2, col=2
        )
        
        extreme_fig.update_layout(
            height=600,
            showlegend=False
        )
        
        st.plotly_chart(extreme_fig, use_container_width=True)
    
    # Tab 2: Statistics
    with tab2:
        st.header("Statistics")
        
        # Create dataframe with key statistics
        stats_df = pd.DataFrame({
            'Direction Changes (%)': results['direction_changes'],
            'Median Price ($)': results['median_prices'],
            'Zero Crossings': results['zero_crossings'],
            'Points Within ±0.5% of Median (%)': results['range_percentages']
        })
        
        # Display the dataframe with key stats
        st.dataframe(stats_df, height=400, use_container_width=True)
        
        # Create histogram for median prices
        fig_median = go.Figure()
        fig_median.add_trace(go.Histogram(
            x=results['median_prices'],
            nbinsx=20,
            marker_color='green',
            opacity=0.7
        ))
        fig_median.add_vline(
            x=initial_price,
            line_dash="dash",
            line_color="red",
            annotation_text="Initial Price",
            annotation_position="top right"
        )
        fig_median.update_layout(
            title="Distribution of Median Prices",
            xaxis_title="Median Price ($)",
            yaxis_title="Count",
            height=300
        )
        st.plotly_chart(fig_median, use_container_width=True)
        
        # Create side-by-side histograms
        col1, col2 = st.columns(2)
        
        with col1:
            # Zero crossings distribution
            fig_zc = go.Figure()
            fig_zc.add_trace(go.Histogram(
                x=results['zero_crossings'],
                nbinsx=20,
                marker_color='purple',
                opacity=0.7
            ))
            fig_zc.update_layout(
                title="Distribution of Zero Crossings",
                xaxis_title="Number of Zero Crossings",
                yaxis_title="Count",
                height=300
            )
            st.plotly_chart(fig_zc, use_container_width=True)
        
        with col2:
            # Range percentage distribution
            fig_range = go.Figure()
            fig_range.add_trace(go.Histogram(
                x=results['range_percentages'],
                nbinsx=20,
                marker_color='orange',
                opacity=0.7
            ))
            fig_range.update_layout(
                title="Distribution of Points Within ±0.5% of Median",
                xaxis_title="Percentage of Points (%)",
                yaxis_title="Count",
                height=300
            )
            st.plotly_chart(fig_range, use_container_width=True)
        
        # Summary statistics
        st.subheader("Summary Statistics")
        st.dataframe(stats_df.describe(), use_container_width=True)
    
    # Tab 3: Export Data
    with tab3:
        st.header("Export Data")
        
        # Export options
        st.subheader("Export Options")
        
        # Export statistics
        csv_stats = stats_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Statistics CSV",
            data=csv_stats,
            file_name=f"price_path_stats.csv",
            mime="text/csv",
            help="Download statistics for all simulations as a CSV file"
        )
        
        # Export example paths
        st.write("Export Example Price Paths:")
        num_paths_export = st.number_input(
            "Number of paths to export",
            min_value=1,
            max_value=num_simulations,
            value=min(10, num_simulations)
        )
        
        # Random selection for export
        export_indices = np.random.choice(len(results['paths']), size=num_paths_export, replace=False)
        export_paths = [results['paths'][i] for i in export_indices]
        
        # Create dataframe for export
        paths_df = pd.DataFrame({f"Path_{i+1}": path for i, path in enumerate(export_paths)})
        
        # Add tick column
        paths_df.insert(0, 'Tick', range(num_ticks))
        
        # Export as CSV
        csv_paths = paths_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Price Paths CSV",
            data=csv_paths,
            file_name=f"price_paths.csv",
            mime="text/csv",
            help="Download selected price paths as a CSV file"
        )
else:
    # Initial instructions when dashboard loads
    with tab1:
        st.info("Set your desired simulation parameters in the sidebar and click 'Run Simulation' to begin.")
        
        st.markdown("""
        ## How This Simulator Works
        
        This tool generates price paths based on three key market behavior metrics:
        
        ### Direction Changes (%)
        How frequently the price changes direction:
        - Lower values = stronger trends
        - Higher values = choppier price action
        
        ### Choppiness
        How much the price oscillates within a range:
        - Lower values = cleaner price movement
        - Higher values = more oscillation in a range
        
        ### Trend Strength
        How efficiently price moves in a direction:
        - Lower values = inefficient price action
        - Higher values = efficient directional movement
        
        After simulation, you'll see example paths and key metrics including:
        - Median price
        - Number of zero crossings (median crossings)
        - Percentage of points within ±0.5% of median price
        """)
    
    with tab2:
        st.info("Run a simulation to see statistics.")
    
    with tab3:
        st.info("Run a simulation to export the data.")

# Add footer
st.sidebar.markdown("---")
st.sidebar.markdown("*Price Path Simulator*")
st.sidebar.markdown(f"*Completed in {time.time() - st.session_state.get('start_time', time.time()):.2f} seconds*" if 'start_time' in st.session_state else "")