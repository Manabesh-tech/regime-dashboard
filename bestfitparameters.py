import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from sqlalchemy import create_engine
from scipy.optimize import minimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Zone-Based Buffer Parameter Fitting", page_icon="📊", layout="wide")

# --- UI Setup ---
st.title("Zone-Based Buffer Parameter Fitting Tool")

# DB connection
db_params = {
    'host': 'aws-jp-tk-surf-pg-public.cluster-csteuf9lw8dv.ap-northeast-1.rds.amazonaws.com',
    'port': 5432,
    'database': 'replication_report',
    'user': 'public_replication',
    'password': '866^FKC4hllk'
}

engine = create_engine(
    f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['database']}"
)

# Cache token list
@st.cache_data(ttl=3600)
def fetch_trading_pairs():
    query = """
    SELECT pair_name 
    FROM trade_pool_pairs 
    WHERE status = 1
    ORDER BY pair_name
    """
    df = pd.read_sql_query(query, engine)
    return df['pair_name'].tolist()

def remove_buffer_spikes(data, spike_time_ranges=None, use_time_based=True, use_statistical=True):
    """
    Remove or smooth out event-related spikes in buffer rates
    Can use time-based removal and/or statistical outlier detection
    """
    data_copy = data.copy()
    original_buffer = data_copy['buffer_rate'].copy()
    
    # Time-based spike removal
    if use_time_based and spike_time_ranges is None:
        # Define spike time ranges (8:25-8:35 PM and 8:35-8:45 PM)
        spike_time_ranges = [
            ('20:25', '20:35'),  # 8:25-8:35 PM
            ('20:35', '20:45'),  # 8:35-8:45 PM
        ]
    
    spike_mask = pd.Series(False, index=data_copy.index)
    
    if use_time_based and spike_time_ranges:
        for start_time, end_time in spike_time_ranges:
            time_mask = (data_copy.index.time >= pd.to_datetime(start_time).time()) & \
                       (data_copy.index.time <= pd.to_datetime(end_time).time())
            spike_mask |= time_mask
    
    # Statistical outlier detection
    if use_statistical:
        # Calculate rolling statistics
        rolling_mean = data_copy['buffer_rate'].rolling(window=30, center=True).mean()
        rolling_std = data_copy['buffer_rate'].rolling(window=30, center=True).std()
        
        # Identify outliers (more than 3 standard deviations from rolling mean)
        outlier_mask = np.abs(data_copy['buffer_rate'] - rolling_mean) > 3 * rolling_std
        spike_mask |= outlier_mask
    
    # Apply filtering
    if spike_mask.any():
        # For buffer rates during spike periods, use interpolation
        data_copy.loc[spike_mask, 'buffer_rate'] = np.nan
        data_copy['buffer_rate'] = data_copy['buffer_rate'].interpolate(method='linear')
        
        # Fill any remaining NaN values
        data_copy['buffer_rate'] = data_copy['buffer_rate'].bfill().ffill()
    
    return data_copy

# Get volatility and rollbit data
def get_zone(vol, p25, p50, p75, p95):
    """
    Determine zone based on volatility and percentiles
    Zone 1: vol >= p95
    Zone 2: p75 < vol <= p95
    Zone 3: p50 < vol <= p75
    Zone 4: vol <= p50
    """
    if vol >= p95:
        return 1
    elif vol > p75:
        return 2
    elif vol > p50:
        return 3
    else:
        return 4

@st.cache_data(ttl=60)
def get_combined_data(token, hours=3):
    now_sg = datetime.now(pytz.timezone('Asia/Singapore'))
    start_time_sg = now_sg - timedelta(hours=hours)
    
    # Get volatility data
    today_str = now_sg.strftime("%Y%m%d")
    yesterday_str = (now_sg - timedelta(days=1)).strftime("%Y%m%d")
    start_time_str = start_time_sg.strftime("%Y-%m-%d %H:%M:%S")
    end_time_str = now_sg.strftime("%Y-%m-%d %H:%M:%S")
    
    # Volatility query
    vol_query = f"""
    SELECT 
        created_at + INTERVAL '8 hour' AS timestamp,
        final_price
    FROM public.oracle_price_log_partition_{today_str}
    WHERE created_at >= '{start_time_str}'::timestamp - INTERVAL '8 hour'
    AND created_at <= '{end_time_str}'::timestamp - INTERVAL '8 hour'
    AND source_type = 0
    AND pair_name = '{token}'
    ORDER BY timestamp
    """
    
    try:
        price_df = pd.read_sql_query(vol_query, engine)
        
        # If no data, try yesterday's partition
        if price_df.empty:
            vol_query_yesterday = f"""
            SELECT 
                created_at + INTERVAL '8 hour' AS timestamp,
                final_price
            FROM public.oracle_price_log_partition_{yesterday_str}
            WHERE created_at >= '{start_time_str}'::timestamp - INTERVAL '8 hour'
            AND created_at <= '{end_time_str}'::timestamp - INTERVAL '8 hour'
            AND source_type = 0
            AND pair_name = '{token}'
            ORDER BY timestamp
            """
            price_df = pd.read_sql_query(vol_query_yesterday, engine)
        
        if price_df.empty:
            return None
        
        # Process price data
        price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
        price_df = price_df.set_index('timestamp').sort_index()
        
        # Resample to 500ms
        price_data = price_df['final_price'].resample('500ms').ffill().dropna()
        
        # Calculate 10-second volatility
        result = []
        start_date = price_data.index.min().floor('10s')
        end_date = price_data.index.max().ceil('10s')
        ten_sec_periods = pd.date_range(start=start_date, end=end_date, freq='10s')
        
        for i in range(len(ten_sec_periods)-1):
            start_window = ten_sec_periods[i]
            end_window = ten_sec_periods[i+1]
            
            window_data = price_data[(price_data.index >= start_window) & (price_data.index < end_window)]
            
            if len(window_data) >= 2:
                log_returns = np.diff(np.log(window_data.values))
                if len(log_returns) > 0:
                    annualization_factor = np.sqrt(3153600)
                    volatility = np.std(log_returns) * annualization_factor * 100  # Convert to percentage
                    
                    result.append({
                        'timestamp': start_window,
                        'volatility': volatility
                    })
        
        vol_df = pd.DataFrame(result).set_index('timestamp')
        
    except Exception as e:
        st.error(f"Error getting volatility: {e}")
        return None
    
    # Get Rollbit data
    rollbit_query = f"""
    SELECT 
        pair_name,
        bust_buffer * 100 AS buffer_rate,  -- Convert to percentage
        position_multiplier,
        created_at + INTERVAL '8 hour' AS timestamp
    FROM rollbit_pair_config 
    WHERE pair_name = '{token}'
    AND created_at >= '{start_time_str}'::timestamp - INTERVAL '8 hour'
    AND created_at <= '{end_time_str}'::timestamp - INTERVAL '8 hour'
    ORDER BY created_at
    """
    
    try:
        rollbit_df = pd.read_sql_query(rollbit_query, engine)
        if rollbit_df.empty:
            return None
            
        rollbit_df['timestamp'] = pd.to_datetime(rollbit_df['timestamp'])
        rollbit_df = rollbit_df.set_index('timestamp').sort_index()
        rollbit_df = rollbit_df.resample('10s').ffill()
        
    except Exception as e:
        st.error(f"Error getting Rollbit data: {e}")
        return None
    
    # Merge data
    combined = pd.merge(vol_df, rollbit_df, left_index=True, right_index=True, how='inner')
    
    return combined

def simulate_buffer_updates(volatilities, buffer_base, vol_base, k, percentiles):
    """
    Simulate buffer updates using the zone-based formula
    """
    p25, p50, p75, p95 = percentiles
    
    predicted_buffers = []
    last_zone = None
    buffer_rate = buffer_base
    
    for vol_t in volatilities:
        zone_t = get_zone(vol_t, p25, p50, p75, p95)
        
        if zone_t == 1 or zone_t != last_zone:
            # Update buffer rate
            buffer_rate = np.clip(
                buffer_base * (vol_t / vol_base) ** k,
                0.03, 0.08  # Converting to percentage (0.0003 -> 0.03%, 0.0008 -> 0.08%)
            )
            last_zone = zone_t
        
        predicted_buffers.append(buffer_rate)
    
    return np.array(predicted_buffers)

def fit_parameters(volatilities, actual_buffers, percentiles):
    """
    Fit buffer_base and k parameters using optimization
    Fixed vol_base as the 50th percentile (p50)
    """
    # Fixed vol_base to be the 50th percentile (p50)
    vol_base = percentiles[1]  # p50 is at index 1
    
    def objective(params):
        buffer_base, k = params
        
        predicted_buffers = simulate_buffer_updates(
            volatilities, buffer_base, vol_base, k, percentiles
        )
        
        # Mean squared error
        mse = np.mean((predicted_buffers - actual_buffers) ** 2)
        return mse
    
    # Initial guess - use mean of actual buffers as starting point
    initial_guess = [np.mean(actual_buffers), 0.5]
    
    # Bounds for optimization - wider bounds for better fitting
    bounds = [(0.03, 0.08), (0.01, 5.0)]  # buffer_base in percentage, k between 0.01 and 5
    
    # Optimize
    result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
    
    return result.x[0], result.x[1], vol_base

# Main UI
all_tokens = fetch_trading_pairs()

st.markdown("### Configuration")
col1, col2, col3 = st.columns([3, 1, 1])

with col1:
    selected_token = st.selectbox(
        "Select Token",
        all_tokens,
        index=all_tokens.index("BTC/USDT") if "BTC/USDT" in all_tokens else 0
    )

with col2:
    remove_spikes = st.checkbox("Remove Spikes", value=True, help="Remove event-related spikes around 8:30-8:40 PM")

with col3:
    if st.button("Analyze All Tokens", type="primary"):
        st.session_state.analyze_all = True

# Single token analysis
with st.spinner(f"Loading data for {selected_token}..."):
    data = get_combined_data(selected_token)

if data is not None and len(data) > 0:
    # Remove spikes if option is selected
    if remove_spikes:
        data = remove_buffer_spikes(data)
        st.info("Event-related spikes around 8:30-8:40 PM have been filtered out")
    
    # Get last 6000 values for percentile calculation (6000 10-sec intervals = ~16.7 hours)
    percentile_window = min(6000, len(data))
    percentile_data = data.tail(percentile_window)
    
    # Calculate percentiles
    all_vols = percentile_data['volatility'].values
    percentiles = (
        np.percentile(all_vols, 25),
        np.percentile(all_vols, 50),
        np.percentile(all_vols, 75),
        np.percentile(all_vols, 95)
    )
    
    # Get last 1000 values for fitting (1000 10-sec intervals = ~2.8 hours)
    fit_window = min(1000, len(data))
    recent_data = data.tail(fit_window)
    
    # Extract volatility and buffer
    volatilities = recent_data['volatility'].values
    buffers = recent_data['buffer_rate'].values
    
    # Fit parameters with fixed vol_base as p50
    buffer_base, k, vol_base = fit_parameters(volatilities, buffers, percentiles)
    
    # Calculate predicted buffers
    predicted_buffers = simulate_buffer_updates(
        volatilities, buffer_base, vol_base, k, percentiles
    )
    
    # Calculate R-squared and other metrics
    ss_res = np.sum((buffers - predicted_buffers) ** 2)
    ss_tot = np.sum((buffers - np.mean(buffers)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Additional diagnostics
    rmse = np.sqrt(np.mean((buffers - predicted_buffers) ** 2))
    mae = np.mean(np.abs(buffers - predicted_buffers))
    
    # Check if parameters hit bounds
    param_at_lower_bound = np.isclose(buffer_base, 0.03) or np.isclose(k, 0.01)
    param_at_upper_bound = np.isclose(buffer_base, 0.08) or np.isclose(k, 5.0)
    
    # Display results
    st.markdown(f"### {selected_token} - Parameter Fitting Results")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Buffer Base", f"{buffer_base:.4f}%")
    with col2:
        st.metric("K Parameter", f"{k:.3f}")
    with col3:
        st.metric("Vol Base (P50)", f"{vol_base:.2f}%")
    with col4:
        st.metric("R² Score", f"{r_squared:.3f}")
    
    # Show warnings if fit is poor
    if r_squared < 0:
        st.error("⚠️ Negative R² indicates the model is performing worse than using the mean. The formula may not match how this token's buffer is calculated.")
    elif r_squared < 0.3:
        st.warning("⚠️ Poor fit (R² < 0.3). Consider checking data quality or if this token uses a different formula.")
    
    if param_at_lower_bound or param_at_upper_bound:
        st.warning("⚠️ One or more parameters hit optimization bounds. Consider adjusting bounds.")
    
    # Additional metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("RMSE", f"{rmse:.4f}%")
    with col2:
        st.metric("MAE", f"{mae:.4f}%")
    with col3:
        st.metric("Buffer Range", f"{buffers.min():.4f}% - {buffers.max():.4f}%")
    with col4:
        st.metric("Vol Range", f"{volatilities.min():.1f}% - {volatilities.max():.1f}%")
    
    # Show percentiles
    st.markdown("### Percentile Values (Last 6000 points = ~16.7 hours)")
    pcol1, pcol2, pcol3, pcol4 = st.columns(4)
    with pcol1:
        st.metric("P25", f"{percentiles[0]:.2f}%")
    with pcol2:
        st.metric("P50", f"{percentiles[1]:.2f}%")
    with pcol3:
        st.metric("P75", f"{percentiles[2]:.2f}%")
    with pcol4:
        st.metric("P95", f"{percentiles[3]:.2f}%")
    
    # Create plot
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Volatility with Zones", "Buffer Rate: Actual vs Predicted")
    )
    
    # Plot volatility with zones
    fig.add_trace(
        go.Scatter(
            x=recent_data.index,
            y=volatilities,
            mode='lines',
            name='Volatility',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # Add percentile lines
    percentile_lines = [
        (percentiles[0], 'P25', 'green'),
        (percentiles[1], 'P50 (Vol Base)', 'orange'),
        (percentiles[2], 'P75', 'red'),
        (percentiles[3], 'P95', 'purple')
    ]
    
    for val, label, color in percentile_lines:
        fig.add_hline(
            y=val,
            line_dash="dash",
            line_color=color,
            annotation_text=label,
            annotation_position="right",
            row=1, col=1
        )
    
    # Plot actual vs predicted buffers
    fig.add_trace(
        go.Scatter(
            x=recent_data.index,
            y=buffers,
            mode='lines',
            name='Actual Buffer',
            line=dict(color='green', width=2)
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=recent_data.index,
            y=predicted_buffers,
            mode='lines',
            name='Predicted Buffer',
            line=dict(color='red', width=2, dash='dash')
        ),
        row=2, col=1
    )
    
    fig.update_layout(height=800, title_text=f"{selected_token} - Zone-Based Buffer Analysis")
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Volatility (%)", row=1, col=1)
    fig.update_yaxes(title_text="Buffer Rate (%)", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show table
    st.markdown("### Data Table (Last 1000 values = ~2.8 hours)")
    
    # Add zone information
    zones = [get_zone(vol, *percentiles) for vol in volatilities]
    
    # Create display dataframe (show last 200 rows for performance)
    display_volatilities = volatilities[-200:]
    display_zones = zones[-200:]
    display_buffers = buffers[-200:]
    display_predicted = predicted_buffers[-200:]
    display_index = recent_data.index[-200:]
    
    display_df = pd.DataFrame({
        'Timestamp': display_index,
        'Volatility (%)': display_volatilities,
        'Zone': display_zones,
        'Actual Buffer (%)': display_buffers,
        'Predicted Buffer (%)': display_predicted,
        'Error (%)': np.abs(display_buffers - display_predicted)
    })
    
    # Show table with formatting
    st.info("Showing last 200 rows of 1000 for display performance")
    st.dataframe(
        display_df.style.format({
            'Volatility (%)': '{:.2f}',
            'Actual Buffer (%)': '{:.4f}',
            'Predicted Buffer (%)': '{:.4f}',
            'Error (%)': '{:.4f}'
        }),
        use_container_width=True
    )
    
    # Zone distribution
    st.markdown("### Zone Distribution")
    zone_counts = pd.Series(zones).value_counts().sort_index()
    zone_df = pd.DataFrame({
        'Zone': zone_counts.index,
        'Count': zone_counts.values,
        'Percentage': (zone_counts.values / len(zones)) * 100
    })
    
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(zone_df)
    
    with col2:
        fig_pie = go.Figure(data=[go.Pie(
            labels=[f'Zone {z}' for z in zone_df['Zone']],
            values=zone_df['Count'],
            hole=0.3
        )])
        fig_pie.update_layout(title="Zone Distribution", height=300)
        st.plotly_chart(fig_pie, use_container_width=True)

else:
    st.error("No data available for the selected token")

# Analyze all tokens
if st.session_state.get('analyze_all', False):
    st.markdown("---")
    st.markdown("### Analyzing All Tokens")
    
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, token in enumerate(all_tokens):
        status_text.text(f"Processing {token}... ({i+1}/{len(all_tokens)})")
        progress_bar.progress((i + 1) / len(all_tokens))
        
        try:
            data = get_combined_data(token)
            
            if data is not None and len(data) > 50:  # Need enough data
                # Remove spikes if option is selected
                if remove_spikes:
                    data = remove_buffer_spikes(data)
                
                # Calculate percentiles
                percentile_window = min(6000, len(data))
                percentile_data = data.tail(percentile_window)
                all_vols = percentile_data['volatility'].values
                percentiles = (
                    np.percentile(all_vols, 25),
                    np.percentile(all_vols, 50),
                    np.percentile(all_vols, 75),
                    np.percentile(all_vols, 95)
                )
                
                # Get fitting data
                fit_window = min(1000, len(data))
                recent_data = data.tail(fit_window)
                volatilities = recent_data['volatility'].values
                buffers = recent_data['buffer_rate'].values
                
                # Fit parameters with fixed vol_base as p50
                buffer_base, k, vol_base = fit_parameters(volatilities, buffers, percentiles)
                
                # Calculate predicted buffers
                predicted_buffers = simulate_buffer_updates(
                    volatilities, buffer_base, vol_base, k, percentiles
                )
                
                # Calculate R-squared
                ss_res = np.sum((buffers - predicted_buffers) ** 2)
                ss_tot = np.sum((buffers - np.mean(buffers)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                results.append({
                    'Token': token,
                    'Buffer Base (%)': buffer_base,
                    'K': k,
                    'Vol Base (P50) (%)': vol_base,
                    'R²': r_squared,
                    'Data Points': len(recent_data)
                })
        except Exception as e:
            results.append({
                'Token': token,
                'Buffer Base (%)': np.nan,
                'K': np.nan,
                'Vol Base (P50) (%)': np.nan,
                'R²': np.nan,
                'Data Points': 0,
                'Error': str(e)
            })
    
    status_text.text("Analysis complete!")
    progress_bar.progress(1.0)
    
    # Display results
    results_df = pd.DataFrame(results)
    
    # Sort by R² score
    results_df = results_df.sort_values('R²', ascending=False)
    
    st.markdown("### All Tokens - Fitting Results")
    st.dataframe(
        results_df.style.format({
            'Buffer Base (%)': '{:.4f}',
            'K': '{:.3f}',
            'Vol Base (P50) (%)': '{:.2f}',
            'R²': '{:.3f}',
            'Data Points': '{:.0f}'
        }).highlight_max(subset=['R²'])
    )
    
    # Summary statistics
    st.markdown("### Summary Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_k = results_df['K'].mean()
        st.metric("Average K", f"{avg_k:.3f}")
    
    with col2:
        avg_r2 = results_df['R²'].mean()
        st.metric("Average R²", f"{avg_r2:.3f}")
    
    with col3:
        successful = len(results_df[results_df['R²'] > 0.5])
        st.metric("Good Fits (R² > 0.5)", f"{successful}/{len(results_df)}")
    
    # Reset analyze_all flag
    st.session_state.analyze_all = False