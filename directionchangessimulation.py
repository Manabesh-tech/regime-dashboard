import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# Set page config
st.set_page_config(
    page_title="Direction Changes Price Path Simulator",
    page_icon="📈",
    layout="wide"
)

# Main title and description
st.title("Direction Changes Price Path Simulator")
st.markdown("""
This dashboard simulates cryptocurrency price paths based on a specified direction changes percentage.
Direction changes measure how frequently the price reverses direction - a higher percentage means more choppy price action.

**How it works:**
- Enter the desired direction changes percentage (e.g., 45% means the price changes direction 45% of the time)
- Select the number of ticks to simulate
- The simulator will generate 1000 price paths with that characteristic
- You can view example paths, distribution statistics, and download the results
""")

# Sidebar for controls
with st.sidebar:
    st.header("Simulation Parameters")
    
    direction_changes_pct = st.slider(
        "Direction Changes Percentage (%)",
        min_value=0.0,
        max_value=100.0,
        value=45.0,
        step=1.0,
        help="Percentage of times the price changes direction. 50% is random walk, lower is more trending, higher is more reversal."
    )
    
    num_ticks = st.select_slider(
        "Number of Ticks",
        options=[100, 500, 1000, 2500, 5000, 10000],
        value=1000,
        help="Number of price ticks to simulate"
    )
    
    initial_price = st.number_input(
        "Initial Price",
        min_value=1.0,
        value=100.0,
        step=1.0,
        help="Starting price for all simulations"
    )
    
    price_volatility = st.slider(
        "Price Volatility (%)",
        min_value=0.01,
        max_value=5.0,
        value=0.5,
        step=0.01,
        help="Average size of price moves as a percentage"
    )
    
    trend_bias = st.slider(
        "Trend Bias (%)",
        min_value=-1.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        help="Bias towards upward (positive) or downward (negative) movement"
    )
    
    num_simulations = st.number_input(
        "Number of Simulations",
        min_value=10,
        max_value=5000,
        value=1000,
        step=10,
        help="Number of price paths to simulate"
    )
    
    run_button = st.button("Run Simulation", type="primary", use_container_width=True)

# Function to simulate a price path with a target direction change percentage
def simulate_price_path(num_ticks, initial_price, target_dir_change_pct, volatility_pct, trend_bias):
    """
    Simulates a price path with a target direction change percentage.
    
    Args:
        num_ticks: Number of price ticks to simulate
        initial_price: Starting price
        target_dir_change_pct: Target percentage of direction changes (0-100)
        volatility_pct: Price volatility as a percentage
        trend_bias: Bias towards upward or downward movement
    
    Returns:
        Pandas Series with the simulated price path
    """
    # Convert percentages to proportions
    target_dir_change = target_dir_change_pct / 100.0
    
    # Initialization
    prices = np.zeros(num_ticks)
    prices[0] = initial_price
    
    # Base probability of a direction change
    p_change = target_dir_change
    
    # Initialize with a random direction
    current_direction = np.random.choice([-1, 1])
    
    # Generate subsequent prices
    for i in range(1, num_ticks):
        # Decide whether to change direction
        if np.random.random() < p_change:
            current_direction *= -1
        
        # Calculate the price change
        volatility = initial_price * (volatility_pct / 100.0)
        price_change = volatility * current_direction
        
        # Add trend bias
        price_change += initial_price * (trend_bias / 100.0)
        
        # Update price
        prices[i] = prices[i-1] + price_change
        
        # Ensure price stays positive
        prices[i] = max(prices[i], 0.01)
    
    return pd.Series(prices)

# Function to calculate actual direction changes percentage
def calculate_direction_changes(prices):
    """
    Calculate the percentage of times the price direction changes.
    
    Args:
        prices: Series of price values
    
    Returns:
        Percentage of direction changes
    """
    price_changes = prices.diff().dropna()
    signs = np.sign(price_changes)
    direction_changes = (signs.shift(1) != signs).sum()
    
    total_periods = len(signs) - 1
    if total_periods > 0:
        direction_change_pct = (direction_changes / total_periods) * 100
    else:
        direction_change_pct = 0
    
    return direction_change_pct

# Function to run multiple simulations and return statistics
def run_multiple_simulations(num_simulations, num_ticks, initial_price, target_dir_change_pct, volatility_pct, trend_bias):
    """
    Run multiple simulations and collect statistics.
    
    Returns:
        Dictionary with simulation results and statistics
    """
    all_paths = []
    direction_changes_actual = []
    final_prices = []
    max_prices = []
    min_prices = []
    max_drawdowns = []
    volatilities = []
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(num_simulations):
        # Update progress
        progress = int((i + 1) / num_simulations * 100)
        progress_bar.progress(progress)
        status_text.text(f"Simulating path {i+1}/{num_simulations} ({progress}%)")
        
        # Simulate path
        path = simulate_price_path(num_ticks, initial_price, target_dir_change_pct, volatility_pct, trend_bias)
        all_paths.append(path)
        
        # Calculate statistics
        direction_changes_actual.append(calculate_direction_changes(path))
        final_prices.append(path.iloc[-1])
        max_prices.append(path.max())
        min_prices.append(path.min())
        
        # Calculate max drawdown
        peak = path.iloc[0]
        max_dd = 0
        for price in path:
            if price > peak:
                peak = price
            dd = (peak - price) / peak * 100 if peak > 0 else 0
            max_dd = max(max_dd, dd)
        max_drawdowns.append(max_dd)
        
        # Calculate volatility (standard deviation of returns)
        returns = path.pct_change().dropna()
        volatilities.append(returns.std() * 100)
    
    # Clear progress bar
    progress_bar.empty()
    status_text.empty()
    
    # Combine all statistics
    results = {
        'paths': all_paths,
        'direction_changes': direction_changes_actual,
        'final_prices': final_prices,
        'max_prices': max_prices,
        'min_prices': min_prices,
        'max_drawdowns': max_drawdowns,
        'volatilities': volatilities
    }
    
    return results

# Main interface
tab1, tab2, tab3, tab4 = st.tabs(["Simulation Results", "Example Paths", "Statistics", "Data Export"])

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
            volatility_pct=price_volatility,
            trend_bias=trend_bias
        )
        execution_time = time.time() - start_time
    
    # Store results in session state for tab access
    st.session_state.simulation_results = results
    st.session_state.simulation_params = {
        'direction_changes_pct': direction_changes_pct,
        'num_ticks': num_ticks,
        'initial_price': initial_price,
        'price_volatility': price_volatility,
        'trend_bias': trend_bias,
        'num_simulations': num_simulations,
        'execution_time': execution_time
    }
    
    # Display basic statistics
    with tab1:
        st.header("Simulation Results")
        st.write(f"Completed {num_simulations} simulations in {execution_time:.2f} seconds")
        
        # Create 3 columns for metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Average Direction Changes",
                f"{np.mean(results['direction_changes']):.2f}%",
                f"{np.mean(results['direction_changes']) - direction_changes_pct:.2f}%"
            )
            
            st.metric(
                "Median Final Price",
                f"${np.median(results['final_prices']):.2f}",
                f"{(np.median(results['final_prices']) - initial_price) / initial_price * 100:.2f}%"
            )
        
        with col2:
            st.metric(
                "Average Max Drawdown",
                f"{np.mean(results['max_drawdowns']):.2f}%"
            )
            
            st.metric(
                "Average Volatility",
                f"{np.mean(results['volatilities']):.2f}%"
            )
        
        with col3:
            st.metric(
                "% of Paths Ending Higher",
                f"{sum(p > initial_price for p in results['final_prices']) / num_simulations * 100:.2f}%"
            )
            
            st.metric(
                "% of Paths Ending Lower",
                f"{sum(p < initial_price for p in results['final_prices']) / num_simulations * 100:.2f}%"
            )
        
        # Direction changes distribution
        st.subheader("Direction Changes Distribution")
        fig1 = go.Figure()
        fig1.add_trace(go.Histogram(
            x=results['direction_changes'],
            nbinsx=20,
            marker_color='blue',
            opacity=0.7
        ))
        fig1.add_vline(
            x=direction_changes_pct,
            line_dash="dash",
            line_color="red",
            annotation_text="Target",
            annotation_position="top right"
        )
        fig1.update_layout(
            title="Distribution of Direction Changes Across Simulations",
            xaxis_title="Direction Changes %",
            yaxis_title="Count",
            height=400
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Final price distribution
        st.subheader("Final Price Distribution")
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=results['final_prices'],
            nbinsx=20,
            marker_color='green',
            opacity=0.7
        ))
        fig2.add_vline(
            x=initial_price,
            line_dash="dash",
            line_color="red",
            annotation_text="Initial Price",
            annotation_position="top right"
        )
        fig2.update_layout(
            title="Distribution of Final Prices Across Simulations",
            xaxis_title="Final Price ($)",
            yaxis_title="Count",
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)
        
    # Example paths visualization
    with tab2:
        st.header("Example Price Paths")
        
        # Controls for visualization
        num_example_paths = st.slider(
            "Number of example paths to show",
            min_value=1,
            max_value=100,
            value=10
        )
        
        # Random selection of paths
        random_indices = np.random.choice(len(results['paths']), size=min(num_example_paths, len(results['paths'])), replace=False)
        selected_paths = [results['paths'][i] for i in random_indices]
        selected_dir_changes = [results['direction_changes'][i] for i in random_indices]
        
        # Create visualization with actual direction changes percentages
        fig = go.Figure()
        
        for i, path in enumerate(selected_paths):
            fig.add_trace(go.Scatter(
                y=path,
                mode='lines',
                name=f"Path {i+1} ({selected_dir_changes[i]:.1f}%)"
            ))
        
        fig.update_layout(
            title=f"Example Price Paths with Target Direction Changes = {direction_changes_pct}%",
            xaxis_title="Tick",
            yaxis_title="Price ($)",
            height=600,
            legend_title="Path (Direction Changes %)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Comparison of path with highest and lowest direction changes
        st.subheader("Comparison: Most vs. Least Direction Changes")
        
        # Find indices of max and min direction changes
        max_idx = np.argmax(results['direction_changes'])
        min_idx = np.argmin(results['direction_changes'])
        
        # Create comparison chart
        comparison_fig = make_subplots(
            rows=2, 
            cols=1,
            subplot_titles=[
                f"Most Direction Changes: {results['direction_changes'][max_idx]:.2f}%",
                f"Least Direction Changes: {results['direction_changes'][min_idx]:.2f}%"
            ],
            vertical_spacing=0.12
        )
        
        # Add the most direction changes path
        comparison_fig.add_trace(
            go.Scatter(
                y=results['paths'][max_idx],
                mode='lines',
                name=f"Most Changes ({results['direction_changes'][max_idx]:.2f}%)",
                line=dict(color='red')
            ),
            row=1, col=1
        )
        
        # Add the least direction changes path
        comparison_fig.add_trace(
            go.Scatter(
                y=results['paths'][min_idx],
                mode='lines',
                name=f"Least Changes ({results['direction_changes'][min_idx]:.2f}%)",
                line=dict(color='blue')
            ),
            row=2, col=1
        )
        
        comparison_fig.update_layout(
            height=600,
            showlegend=True
        )
        
        st.plotly_chart(comparison_fig, use_container_width=True)
    
    # Statistics tab
    with tab3:
        st.header("Simulation Statistics")
        
        # Create dataframe with all statistics
        stats_df = pd.DataFrame({
            'Direction Changes (%)': results['direction_changes'],
            'Final Price ($)': results['final_prices'],
            'Max Price ($)': results['max_prices'],
            'Min Price ($)': results['min_prices'],
            'Max Drawdown (%)': results['max_drawdowns'],
            'Volatility (%)': results['volatilities'],
            'Return (%)': [(p - initial_price) / initial_price * 100 for p in results['final_prices']]
        })
        
        # Display the dataframe
        st.dataframe(stats_df, height=400, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Correlation Between Metrics")
        
        # Calculate correlation matrix
        corr_matrix = stats_df.corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='Viridis',
            zmin=-1, zmax=1
        ))
        
        fig.update_layout(
            title="Correlation Between Different Metrics",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        st.subheader("Summary Statistics")
        st.dataframe(stats_df.describe(), use_container_width=True)
        
        # Interpretation
        st.subheader("Interpretation")
        
        interpretation = f"""
        The target direction changes percentage was **{direction_changes_pct}%**, and the simulations achieved an average of **{np.mean(results['direction_changes']):.2f}%**.
        
        **Key observations:**
        - {'Higher' if np.mean(results['max_drawdowns']) > 20 else 'Lower'} direction changes typically lead to {'higher' if np.mean(results['max_drawdowns']) > 20 else 'lower'} drawdowns
        - The average volatility across all simulations was **{np.mean(results['volatilities']):.2f}%**
        - **{sum(p > initial_price for p in results['final_prices']) / num_simulations * 100:.1f}%** of paths ended higher than the starting price
        - The median final price was **${np.median(results['final_prices']):.2f}**
        
        This simulation demonstrates how different direction change percentages affect price movement patterns.
        """
        
        st.markdown(interpretation)
        
    # Data export tab
    with tab4:
        st.header("Data Export")
        
        # Export options
        st.subheader("Export Options")
        
        # Export statistics
        csv_stats = stats_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Statistics CSV",
            data=csv_stats,
            file_name=f"direction_changes_stats_{direction_changes_pct:.0f}pct.csv",
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
        export_dir_changes = [results['direction_changes'][i] for i in export_indices]
        
        # Create dataframe for export
        paths_df = pd.DataFrame({f"Path_{i+1}_{export_dir_changes[i]:.1f}pct": path for i, path in enumerate(export_paths)})
        
        # Add tick column
        paths_df.insert(0, 'Tick', range(num_ticks))
        
        # Export as CSV
        csv_paths = paths_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Price Paths CSV",
            data=csv_paths,
            file_name=f"direction_changes_paths_{direction_changes_pct:.0f}pct.csv",
            mime="text/csv",
            help="Download selected price paths as a CSV file"
        )
        
        # Export all data
        st.write("Export Complete Dataset (Warning: May be large)")
        
        if st.button("Prepare Full Dataset for Download"):
            # Create full dataframe (this could be very large)
            st.warning("Preparing full dataset - this may take a moment for large simulations...")
            
            # Create dataframe with all paths
            all_paths_df = pd.DataFrame({f"Path_{i+1}": path for i, path in enumerate(results['paths'])})
            
            # Add tick column
            all_paths_df.insert(0, 'Tick', range(num_ticks))
            
            # Export as CSV
            csv_all = all_paths_df.to_csv(index=False).encode('utf-8')
            
            # Create download button
            st.download_button(
                label="Download Complete Dataset",
                data=csv_all,
                file_name=f"all_paths_{direction_changes_pct:.0f}pct_{num_simulations}_sims.csv",
                mime="text/csv",
                help="Download all simulated price paths"
            )
            
            st.success(f"Full dataset prepared with {num_simulations} paths of {num_ticks} ticks each")
else:
    # Initial instructions when dashboard loads
    with tab1:
        st.info("Set your desired simulation parameters in the sidebar and click 'Run Simulation' to begin.")
        
        st.markdown("""
        ## Understanding Direction Changes
        
        **Direction changes** is a metric that measures how often price movement changes direction:
        
        - **Low values (0-30%)**: Strong trending behavior with few reversals
        - **Medium values (30-50%)**: Mild trending with some choppiness
        - **Values near 50%**: Similar to a random walk
        - **High values (60-100%)**: Very choppy, oscillating price action
        
        The simulation allows you to see what different price paths might look like with various direction change percentages.
        """)
        
        # Show example images for different direction change percentages
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### Low Direction Changes (25%)")
            st.markdown("*Strong trend with few reversals*")
            
        with col2:
            st.markdown("### Medium Direction Changes (50%)")
            st.markdown("*Random walk behavior*")
            
        with col3:
            st.markdown("### High Direction Changes (75%)")
            st.markdown("*Choppy, oscillating price action*")
            
    with tab2:
        st.info("Run a simulation to see example price paths.")
    
    with tab3:
        st.info("Run a simulation to see detailed statistics.")
    
    with tab4:
        st.info("Run a simulation to export the data.")

# Add explanation in the sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("About Direction Changes")
st.sidebar.markdown("""
**Direction changes** measures the frequency of price reversals:

- **Lower values** indicate stronger trends with fewer reversals
- **Higher values** indicate choppier, oscillating price action
- **50%** approximates a random walk

This metric is used by traders to understand market behavior and adjust trading strategies accordingly.
""")

# Add footer
st.sidebar.markdown("---")
st.sidebar.markdown("*Direction Changes Price Path Simulator*")
st.sidebar.markdown("*Created with Streamlit and Plotly*")