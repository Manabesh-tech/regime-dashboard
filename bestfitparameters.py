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

# Get volatility and rollbit data
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
    """
    def objective(params):
        buffer_base, k = params
        vol_base = volatilities[0]  # Use first volatility as base
        
        predicted_buffers = simulate_buffer_updates(
            volatilities, buffer_base, vol_base, k, percentiles
        )
        
        # Mean squared error
        mse = np.mean((predicted_buffers - actual_buffers) ** 2)
        return mse
    
    # Initial guess - use mean of actual buffers as starting point
    initial_guess = [np.mean(actual_buffers), 0.5]
    
    # Bounds for optimization
    bounds = [(0.03, 0.08), (0.1, 2.0)]  # buffer_base in percentage, k between 0.1 and 2
    
    # Optimize
    result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
    
    return result.x[0], result.x[1]

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
    fit_hours = st.number_input("Hours for fitting", min_value=1, max_value=24, value=3)

with col3:
    if st.button("Analyze All Tokens", type="primary"):
        st.session_state.analyze_all = True

# Single token analysis
with st.spinner(f"Loading data for {selected_token}..."):
    data = get_combined_data(selected_token, fit_hours)

if data is not None and len(data) > 0:
    # Get last 600 values for percentile calculation
    percentile_window = min(600, len(data))
    percentile_data = data.tail(percentile_window)
    
    # Calculate percentiles
    all_vols = percentile_data['volatility'].values
    percentiles = (
        np.percentile(all_vols, 25),
        np.percentile(all_vols, 50),
        np.percentile(all_vols, 75),
        np.percentile(all_vols, 95)
    )
    
    # Get last 200 values for fitting
    fit_window = min(200, len(data))
    recent_data = data.tail(fit_window)
    
    # Extract volatility and buffer
    volatilities = recent_data['volatility'].values
    buffers = recent_data['buffer_rate'].values
    
    # Fit parameters
    buffer_base, k = fit_parameters(volatilities, buffers, percentiles)
    
    # Calculate predicted buffers
    vol_base = volatilities[0]
    predicted_buffers = simulate_buffer_updates(
        volatilities, buffer_base, vol_base, k, percentiles
    )
    
    # Calculate R-squared
    ss_res = np.sum((buffers - predicted_buffers) ** 2)
    ss_tot = np.sum((buffers - np.mean(buffers)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Display results
    st.markdown(f"### {selected_token} - Parameter Fitting Results")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Buffer Base", f"{buffer_base:.4f}%")
    with col2:
        st.metric("K Parameter", f"{k:.3f}")
    with col3:
        st.metric("Vol Base", f"{vol_base:.2f}%")
    with col4:
        st.metric("R² Score", f"{r_squared:.3f}")
    
    # Show percentiles
    st.markdown("### Percentile Values (Last 600 points)")
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
        (percentiles[1], 'P50', 'orange'),
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
    st.markdown("### Data Table (Last 200 values)")
    
    # Add zone information
    zones = [get_zone(vol, *percentiles) for vol in volatilities]
    
    # Create display dataframe
    display_df = pd.DataFrame({
        'Timestamp': recent_data.index,
        'Volatility (%)': volatilities,
        'Zone': zones,
        'Actual Buffer (%)': buffers,
        'Predicted Buffer (%)': predicted_buffers,
        'Error (%)': np.abs(buffers - predicted_buffers)
    })
    
    # Show table with formatting
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
            data = get_combined_data(token, fit_hours)
            
            if data is not None and len(data) > 50:  # Need enough data
                # Calculate percentiles
                percentile_window = min(600, len(data))
                percentile_data = data.tail(percentile_window)
                all_vols = percentile_data['volatility'].values
                percentiles = (
                    np.percentile(all_vols, 25),
                    np.percentile(all_vols, 50),
                    np.percentile(all_vols, 75),
                    np.percentile(all_vols, 95)
                )
                
                # Get fitting data
                fit_window = min(200, len(data))
                recent_data = data.tail(fit_window)
                volatilities = recent_data['volatility'].values
                buffers = recent_data['buffer_rate'].values
                
                # Fit parameters
                buffer_base, k = fit_parameters(volatilities, buffers, percentiles)
                
                # Calculate predicted buffers
                vol_base = volatilities[0]
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
                    'Vol Base (%)': vol_base,
                    'R²': r_squared,
                    'Data Points': len(recent_data)
                })
        except Exception as e:
            results.append({
                'Token': token,
                'Buffer Base (%)': np.nan,
                'K': np.nan,
                'Vol Base (%)': np.nan,
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
            'Vol Base (%)': '{:.2f}',
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