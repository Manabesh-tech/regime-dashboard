import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# Set page config
st.set_page_config(
    page_title="Multi-Metric Crypto Price Path Simulator",
    page_icon="📈",
    layout="wide"
)

# Main title and description
st.title("Multi-Metric Crypto Price Path Simulator")
st.markdown("""
This dashboard simulates cryptocurrency price paths based on three key metrics:

1. **Direction Changes (%)**: How frequently the price changes direction
2. **Choppiness**: How much the price oscillates within a range
3. **Trend Strength**: How effectively price moves in a consistent direction

By adjusting these metrics, you can simulate different market conditions and generate realistic price paths.
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
    
    choppiness = st.slider(
        "Choppiness",
        min_value=100.0,
        max_value=300.0,
        value=150.0,
        step=5.0,
        help="Measures price oscillation within a range. Higher values indicate more oscillation."
    )
    
    trend_strength = st.slider(
        "Trend Strength",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Measures how effectively price moves in a consistent direction. Lower values indicate more oscillation without net progress."
    )
    
    st.markdown("---")
    
    st.subheader("Additional Parameters")
    
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
    
    num_simulations = st.number_input(
        "Number of Simulations",
        min_value=10,
        max_value=5000,
        value=1000,
        step=10,
        help="Number of price paths to simulate"
    )
    
    run_button = st.button("Run Simulation", type="primary", use_container_width=True)

# Function to simulate a price path with target metrics
def simulate_price_path(num_ticks, initial_price, target_dir_change_pct, target_choppiness, target_trend_strength):
    """
    Simulates a price path with target metrics.
    
    Args:
        num_ticks: Number of price ticks to simulate
        initial_price: Starting price
        target_dir_change_pct: Target percentage of direction changes (0-100)
        target_choppiness: Target choppiness value (higher means more oscillation)
        target_trend_strength: Target trend strength (higher means stronger trend)
    
    Returns:
        Pandas Series with the simulated price path
    """
    # Convert percentages to proportions
    target_dir_change = target_dir_change_pct / 100.0
    
    # Initialization
    prices = np.zeros(num_ticks)
    prices[0] = initial_price
    
    # Initialize with a random direction
    current_direction = np.random.choice([-1, 1])
    
    # Window size for calculating running metrics (for adjustments)
    window_size = min(20, num_ticks // 10)
    
    # Base volatility (derived from the target metrics)
    # Higher choppiness or lower trend strength means higher volatility
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
            
            # Calculate current choppiness (simplified version)
            price_range = max(window_prices) - min(window_prices)
            sum_movements = np.sum(np.abs(np.diff(window_prices)))
            current_choppiness = (sum_movements / price_range * 100) if price_range > 0 else 150
            
            # Calculate current trend strength
            price_range = max(window_prices) - min(window_prices)
            net_change = abs(window_prices[-1] - window_prices[0])
            current_trend_strength = (net_change / sum_movements) if sum_movements > 0 else 0.5
            
            # Adjust probabilities based on current metrics vs target metrics
            # Direction changes adjustment
            dir_change_adjustment = 0.1 * (target_dir_change - current_dir_change)
            
            # Choppiness adjustment
            # If current choppiness is higher than target, reduce volatility
            # If current choppiness is lower than target, increase volatility
            choppiness_ratio = target_choppiness / current_choppiness if current_choppiness > 0 else 1
            volatility_adjustment = min(max(choppiness_ratio * 0.5, 0.5), 1.5)
            
            # Trend strength adjustment
            # If current trend strength is lower than target, reduce direction change probability
            # If current trend strength is higher than target, increase direction change probability
            trend_adjustment = 0.1 * (current_trend_strength - target_trend_strength)
            
            # Combine adjustments
            p_change = max(0.01, min(0.99, target_dir_change + dir_change_adjustment - trend_adjustment))
            
            # Adjust volatility
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

def calculate_choppiness(prices, window=14):
    """Calculate average Choppiness Index."""
    diff = prices.diff().abs()
    sum_abs_changes = diff.rolling(window, min_periods=1).sum()
    price_range = prices.rolling(window, min_periods=1).max() - prices.rolling(window, min_periods=1).min()
    
    # Avoid division by zero
    epsilon = 1e-10
    choppiness = 100 * sum_abs_changes / (price_range + epsilon)
    
    # Cap extreme values and handle NaN
    choppiness = np.minimum(choppiness, 1000)
    choppiness = choppiness.fillna(200)
    
    return choppiness.mean()

def calculate_trend_strength(prices, window=14):
    """Calculate average Trend Strength."""
    diff = prices.diff().abs()
    sum_abs_changes = diff.rolling(window, min_periods=1).sum()
    net_change = (prices - prices.shift(window)).abs()
    
    # Avoid division by zero
    epsilon = 1e-10
    trend_strength = np.where(
        sum_abs_changes > epsilon,
        net_change / (sum_abs_changes + epsilon),
        0.5
    )
    
    # Convert to pandas Series if it's a numpy array
    if isinstance(trend_strength, np.ndarray):
        trend_strength = pd.Series(trend_strength, index=net_change.index)
    
    # Handle NaN values
    trend_strength = pd.Series(trend_strength).fillna(0.5)
    
    return trend_strength.mean()

# Function to run multiple simulations and return statistics
def run_multiple_simulations(num_simulations, num_ticks, initial_price, target_dir_change_pct, 
                             target_choppiness, target_trend_strength):
    """
    Run multiple simulations and collect statistics.
    
    Returns:
        Dictionary with simulation results and statistics
    """
    all_paths = []
    direction_changes_actual = []
    choppiness_actual = []
    trend_strength_actual = []
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
        choppiness_actual.append(calculate_choppiness(path))
        trend_strength_actual.append(calculate_trend_strength(path))
        
        # Calculate additional statistics
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
        'choppiness': choppiness_actual,
        'trend_strength': trend_strength_actual,
        'final_prices': final_prices,
        'max_prices': max_prices,
        'min_prices': min_prices,
        'max_drawdowns': max_drawdowns,
        'volatilities': volatilities
    }
    
    return results

# Main interface
tab1, tab2, tab3, tab4 = st.tabs(["Simulation Results", "Example Paths", "Metrics Analysis", "Data Export"])

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
    
    # Display basic statistics in tab 1
    with tab1:
        st.header("Simulation Results")
        st.write(f"Completed {num_simulations} simulations in {execution_time:.2f} seconds")
        
        # Create columns for metrics summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Avg Direction Changes",
                f"{np.mean(results['direction_changes']):.2f}%",
                f"{np.mean(results['direction_changes']) - direction_changes_pct:.2f}%"
            )
            
            st.metric(
                "Avg Choppiness",
                f"{np.mean(results['choppiness']):.2f}",
                f"{np.mean(results['choppiness']) - choppiness:.2f}"
            )
            
            st.metric(
                "Avg Trend Strength",
                f"{np.mean(results['trend_strength']):.2f}",
                f"{np.mean(results['trend_strength']) - trend_strength:.2f}"
            )
        
        with col2:
            st.metric(
                "Median Final Price",
                f"${np.median(results['final_prices']):.2f}",
                f"{(np.median(results['final_prices']) - initial_price) / initial_price * 100:.2f}%"
            )
            
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
                "% Paths Ending Higher",
                f"{sum(p > initial_price for p in results['final_prices']) / num_simulations * 100:.2f}%"
            )
            
            st.metric(
                "% Paths Ending Lower",
                f"{sum(p < initial_price for p in results['final_prices']) / num_simulations * 100:.2f}%"
            )
            
            st.metric(
                "Avg Net Change",
                f"${np.mean([p - initial_price for p in results['final_prices']]):.2f}"
            )
        
        # Distribution of final prices
        st.subheader("Final Price Distribution")
        fig_prices = go.Figure()
        fig_prices.add_trace(go.Histogram(
            x=results['final_prices'],
            nbinsx=20,
            marker_color='green',
            opacity=0.7
        ))
        fig_prices.add_vline(
            x=initial_price,
            line_dash="dash",
            line_color="red",
            annotation_text="Initial Price",
            annotation_position="top right"
        )
        fig_prices.update_layout(
            title="Distribution of Final Prices Across Simulations",
            xaxis_title="Final Price ($)",
            yaxis_title="Count",
            height=400
        )
        st.plotly_chart(fig_prices, use_container_width=True)
        
        # Metrics distributions
        metrics_col1, metrics_col2 = st.columns(2)
        
        with metrics_col1:
            # Direction changes distribution
            fig_dir = go.Figure()
            fig_dir.add_trace(go.Histogram(
                x=results['direction_changes'],
                nbinsx=20,
                marker_color='blue',
                opacity=0.7
            ))
            fig_dir.add_vline(
                x=direction_changes_pct,
                line_dash="dash",
                line_color="red",
                annotation_text="Target",
                annotation_position="top right"
            )
            fig_dir.update_layout(
                title="Direction Changes Distribution",
                xaxis_title="Direction Changes %",
                yaxis_title="Count",
                height=300
            )
            st.plotly_chart(fig_dir, use_container_width=True)
        
        with metrics_col2:
            # Choppiness distribution
            fig_chop = go.Figure()
            fig_chop.add_trace(go.Histogram(
                x=results['choppiness'],
                nbinsx=20,
                marker_color='purple',
                opacity=0.7
            ))
            fig_chop.add_vline(
                x=choppiness,
                line_dash="dash",
                line_color="red",
                annotation_text="Target",
                annotation_position="top right"
            )
            fig_chop.update_layout(
                title="Choppiness Distribution",
                xaxis_title="Choppiness",
                yaxis_title="Count",
                height=300
            )
            st.plotly_chart(fig_chop, use_container_width=True)
        
        # Trend strength distribution
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Histogram(
            x=results['trend_strength'],
            nbinsx=20,
            marker_color='orange',
            opacity=0.7
        ))
        fig_trend.add_vline(
            x=trend_strength,
            line_dash="dash",
            line_color="red",
            annotation_text="Target",
            annotation_position="top right"
        )
        fig_trend.update_layout(
            title="Trend Strength Distribution",
            xaxis_title="Trend Strength",
            yaxis_title="Count",
            height=300
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    
    # Example paths visualization in tab 2
    with tab2:
        st.header("Example Price Paths")
        
        # Controls for visualization
        num_example_paths = st.slider(
            "Number of example paths to show",
            min_value=1,
            max_value=50,
            value=10
        )
        
        # Random selection of paths
        random_indices = np.random.choice(len(results['paths']), size=min(num_example_paths, len(results['paths'])), replace=False)
        selected_paths = [results['paths'][i] for i in random_indices]
        selected_metrics = [
            (results['direction_changes'][i], results['choppiness'][i], results['trend_strength'][i]) 
            for i in random_indices
        ]
        
        # Create visualization with metrics in legend
        fig = go.Figure()
        
        for i, path in enumerate(selected_paths):
            dir_change, chop, trend = selected_metrics[i]
            fig.add_trace(go.Scatter(
                y=path,
                mode='lines',
                name=f"Path {i+1} (DC: {dir_change:.1f}%, CH: {chop:.1f}, TS: {trend:.2f})"
            ))
        
        fig.update_layout(
            title=f"Example Price Paths (Target: DC={direction_changes_pct}%, CH={choppiness}, TS={trend_strength})",
            xaxis_title="Tick",
            yaxis_title="Price ($)",
            height=600,
            legend_title="Path (Metrics)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Examples of extreme cases
        st.subheader("Examples of Extreme Cases")
        
        # Find indices for extreme cases
        high_dir_change_idx = np.argmax(results['direction_changes'])
        low_dir_change_idx = np.argmin(results['direction_changes'])
        high_chop_idx = np.argmax(results['choppiness'])
        low_chop_idx = np.argmin(results['choppiness'])
        high_trend_idx = np.argmax(results['trend_strength'])
        low_trend_idx = np.argmin(results['trend_strength'])
        
        # Create a subplot grid
        extreme_fig = make_subplots(
            rows=3, 
            cols=2,
            subplot_titles=[
                f"Highest Direction Changes: {results['direction_changes'][high_dir_change_idx]:.2f}%",
                f"Lowest Direction Changes: {results['direction_changes'][low_dir_change_idx]:.2f}%",
                f"Highest Choppiness: {results['choppiness'][high_chop_idx]:.2f}",
                f"Lowest Choppiness: {results['choppiness'][low_chop_idx]:.2f}",
                f"Highest Trend Strength: {results['trend_strength'][high_trend_idx]:.2f}",
                f"Lowest Trend Strength: {results['trend_strength'][low_trend_idx]:.2f}"
            ],
            vertical_spacing=0.1
        )
        
        # Add each extreme case
        extreme_fig.add_trace(
            go.Scatter(y=results['paths'][high_dir_change_idx], mode='lines', name="Highest Dir Changes", line=dict(color='red')),
            row=1, col=1
        )
        extreme_fig.add_trace(
            go.Scatter(y=results['paths'][low_dir_change_idx], mode='lines', name="Lowest Dir Changes", line=dict(color='blue')),
            row=1, col=2
        )
        extreme_fig.add_trace(
            go.Scatter(y=results['paths'][high_chop_idx], mode='lines', name="Highest Choppiness", line=dict(color='purple')),
            row=2, col=1
        )
        extreme_fig.add_trace(
            go.Scatter(y=results['paths'][low_chop_idx], mode='lines', name="Lowest Choppiness", line=dict(color='green')),
            row=2, col=2
        )
        extreme_fig.add_trace(
            go.Scatter(y=results['paths'][high_trend_idx], mode='lines', name="Highest Trend Strength", line=dict(color='orange')),
            row=3, col=1
        )
        extreme_fig.add_trace(
            go.Scatter(y=results['paths'][low_trend_idx], mode='lines', name="Lowest Trend Strength", line=dict(color='brown')),
            row=3, col=2
        )
        
        extreme_fig.update_layout(
            height=800,
            showlegend=False
        )
        
        st.plotly_chart(extreme_fig, use_container_width=True)
    
    # Metrics analysis in tab 3
    with tab3:
        st.header("Metrics Analysis")
        
        # Create dataframe with all metrics and statistics
        stats_df = pd.DataFrame({
            'Direction Changes (%)': results['direction_changes'],
            'Choppiness': results['choppiness'],
            'Trend Strength': results['trend_strength'],
            'Final Price ($)': results['final_prices'],
            'Max Price ($)': results['max_prices'],
            'Min Price ($)': results['min_prices'],
            'Max Drawdown (%)': results['max_drawdowns'],
            'Volatility (%)': results['volatilities'],
            'Return (%)': [(p - initial_price) / initial_price * 100 for p in results['final_prices']]
        })
        
        # Display the dataframe
        st.subheader("All Metrics")
        st.dataframe(stats_df, height=400, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Correlation Between Metrics")
        
        # Calculate correlation matrix
        corr_matrix = stats_df.corr()
        
        # Create heatmap
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='Viridis',
            zmin=-1, zmax=1
        ))
        
        fig_corr.update_layout(
            title="Correlation Between Different Metrics",
            height=600
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Scatter plots matrix for key relationships
        st.subheader("Relationship Between Key Metrics")
        
        # Create 2x2 scatter plot matrix
        scatter_fig = make_subplots(
            rows=2, 
            cols=2,
            subplot_titles=[
                "Direction Changes vs Final Price",
                "Choppiness vs Final Price",
                "Trend Strength vs Final Price",
                "Direction Changes vs Drawdown"
            ]
        )
        
        # Add scatter plots
        scatter_fig.add_trace(
            go.Scatter(
                x=stats_df['Direction Changes (%)'],
                y=stats_df['Final Price ($)'],
                mode='markers',
                marker=dict(color='blue', size=5, opacity=0.6),
                name="DC vs Price"
            ),
            row=1, col=1
        )
        
        scatter_fig.add_trace(
            go.Scatter(
                x=stats_df['Choppiness'],
                y=stats_df['Final Price ($)'],
                mode='markers',
                marker=dict(color='green', size=5, opacity=0.6),
                name="Chop vs Price"
            ),
            row=1, col=2
        )
        
        scatter_fig.add_trace(
            go.Scatter(
                x=stats_df['Trend Strength'],
                y=stats_df['Final Price ($)'],
                mode='markers',
                marker=dict(color='purple', size=5, opacity=0.6),
                name="Trend vs Price"
            ),
            row=2, col=1
        )
        
        scatter_fig.add_trace(
            go.Scatter(
                x=stats_df['Direction Changes (%)'],
                y=stats_df['Max Drawdown (%)'],
                mode='markers',
                marker=dict(color='red', size=5, opacity=0.6),
                name="DC vs Drawdown"
            ),
            row=2, col=2
        )
        
        scatter_fig.update_layout(
            height=800,
            showlegend=False
        )
        
        st.plotly_chart(scatter_fig, use_container_width=True)
        
        # Summary statistics
        st.subheader("Summary Statistics")
        st.dataframe(stats_df.describe(), use_container_width=True)
        
        # Interpretation
        st.subheader("Interpretation")
        
        # Generate interpretation based on the results
        avg_dir_change = np.mean(results['direction_changes'])
        avg_choppiness = np.mean(results['choppiness'])
        avg_trend_strength = np.mean(results['trend_strength'])
        correlation_dc_price = corr_matrix.loc['Direction Changes (%)', 'Final Price ($)']
        correlation_chop_price = corr_matrix.loc['Choppiness', 'Final Price ($)']
        correlation_trend_price = corr_matrix.loc['Trend Strength', 'Final Price ($)']
        
        interpretation = f"""
        The simulation targeted these metrics:
        - Direction Changes: **{direction_changes_pct}%** (achieved: **{avg_dir_change:.2f}%**)
        - Choppiness: **{choppiness}** (achieved: **{avg_choppiness:.2f}**)
        - Trend Strength: **{trend_strength}** (achieved: **{avg_trend_strength:.2f}**)
        
        **Key observations:**
        - The correlation between Direction Changes and Final Price is **{correlation_dc_price:.2f}**
        - The correlation between Choppiness and Final Price is **{correlation_chop_price:.2f}**
        - The correlation between Trend Strength and Final Price is **{correlation_trend_strength:.2f}**
        
        {'Higher' if avg_dir_change > 50 else 'Lower'} direction changes typically lead to {'more volatile' if avg_dir_change > 50 else 'more trending'} price action.
        {'Higher' if avg_choppiness > 150 else 'Lower'} choppiness results in {'more oscillation within ranges' if avg_choppiness > 150 else 'cleaner price movements'}.
        {'Higher' if avg_trend_strength > 0.5 else 'Lower'} trend strength leads to {'stronger directional movements' if avg_trend_strength > 0.5 else 'more price inefficiency'}.
        
        **{sum(p > initial_price for p in results['final_prices']) / num_simulations * 100:.1f}%** of paths ended higher than the starting price, suggesting a {'bullish' if sum(p > initial_price for p in results['final_prices']) > num_simulations/2 else 'bearish'} bias in the simulations.
        
        This demonstrates how different combinations of these three metrics create distinct price behavior patterns.
        """
        
        st.markdown(interpretation)
    
    # Data export in tab 4
    with tab4:
        st.header("Data Export")
        
        # Export options
        st.subheader("Export Options")
        
        # Export statistics
        csv_stats = stats_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Statistics CSV",
            data=csv_stats,
            file_name=f"multi_metric_stats_DC{direction_changes_pct:.0f}_CH{choppiness:.0f}_TS{trend_strength:.2f}.csv",
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
        export_metrics = [
            f"DC{results['direction_changes'][i]:.1f}_CH{results['choppiness'][i]:.1f}_TS{results['trend_strength'][i]:.2f}"
            for i in export_indices
        ]
        
        # Create dataframe for export
        paths_df = pd.DataFrame({f"Path_{i+1}_{metric}": path for i, (path, metric) in enumerate(zip(export_paths, export_metrics))})
        
        # Add tick column
        paths_df.insert(0, 'Tick', range(num_ticks))
        
        # Export as CSV
        csv_paths = paths_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Price Paths CSV",
            data=csv_paths,
            file_name=f"multi_metric_paths_DC{direction_changes_pct:.0f}_CH{choppiness:.0f}_TS{trend_strength:.2f}.csv",
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
                file_name=f"all_paths_DC{direction_changes_pct:.0f}_CH{choppiness:.0f}_TS{trend_strength:.2f}_{num_simulations}_sims.csv",
                mime="text/csv",
                help="Download all simulated price paths"
            )
            
            st.success(f"Full dataset prepared with {num_simulations} paths of {num_ticks} ticks each")
else:
    # Initial instructions when dashboard loads
    with tab1:
        st.info("Set your desired simulation parameters in the sidebar and click 'Run Simulation' to begin.")
        
        st.markdown("""
        ## Understanding the Metrics
        
        These three metrics work together to describe different aspects of price behavior:
        
        ### 1. Direction Changes (%)
        Measures how frequently the price changes direction:
        - **Low values (0-30%)**: Strong trending behavior with few reversals
        - **Medium values (30-50%)**: Mild trending with some choppiness
        - **Values near 50%**: Similar to a random walk
        - **High values (60-100%)**: Very choppy, oscillating price action
        
        ### 2. Choppiness
        Measures price oscillation within a range:
        - **Low values (100-150)**: Cleaner price movements with less oscillation
        - **Medium values (150-200)**: Moderate oscillation within ranges
        - **High values (200+)**: Significant oscillation with price repeatedly covering the same range
        
        ### 3. Trend Strength
        Measures the directional strength of price movement:
        - **Low values (0.1-0.3)**: Price oscillates a lot without making much net progress
        - **Medium values (0.3-0.6)**: Moderate trending with some inefficiency
        - **High values (0.6-0.9)**: Strong trending with efficient directional movement
        """)
        
        # Show example visuals
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### Low Direction Changes")
            st.markdown("*Strong trend with few reversals*")
            
        with col2:
            st.markdown("### Medium Choppiness")
            st.markdown("*Moderate oscillation within ranges*")
            
        with col3:
            st.markdown("### High Trend Strength")
            st.markdown("*Efficient directional movement*")
    
    with tab2:
        st.info("Run a simulation to see example price paths.")
    
    with tab3:
        st.info("Run a simulation to see detailed metrics analysis.")
    
    with tab4:
        st.info("Run a simulation to export the data.")

# Add explanation in the sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("About These Metrics")
st.sidebar.markdown("""
**Direction Changes (%)** - Frequency of price reversals:
- Lower values = stronger trends
- Higher values = choppier price action
- 50% = random walk

**Choppiness** - Price oscillation within ranges:
- Lower values = cleaner price movement
- Higher values = more oscillation in a range

**Trend Strength** - Directional efficiency:
- Lower values = inefficient price action
- Higher values = efficient directional movement

These metrics together create realistic market behavior patterns.
""")

# Add footer
st.sidebar.markdown("---")
st.sidebar.markdown("*Multi-Metric Crypto Price Path Simulator*")
st.sidebar.markdown("*Created with Streamlit and Plotly*")