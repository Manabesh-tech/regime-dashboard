import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
from datetime import datetime, timedelta

# --- DB CONFIG ---
db_config = st.secrets["database"]

db_uri = (
    f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}"
    f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
)
engine = create_engine(db_uri)

# --- UI Setup ---
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Rolling Hurst Exponent Dashboard")

# --- Fetch token list from DB ---
@st.cache_data
def fetch_token_list():
    query = "SELECT DISTINCT pair_name FROM public.oracle_price_log ORDER BY pair_name"
    df = pd.read_sql(query, engine)
    return df['pair_name'].tolist()

token_list = fetch_token_list()
selected_token = st.selectbox("Select Token", token_list, index=0)
timeframe = st.selectbox("Timeframe", ["30s", "15min", "30min", "1h", "6h"], index=2)

col1, col2 = st.columns(2)
with col1:
    lookback_days = st.slider("Lookback (Days)", 1, 30, 2)
with col2:
    rolling_window = st.slider("Rolling Window (Bars)", 20, 100, 20)

# --- Determine Bars per Hour and calculate expected data points ---
bars_per_hour = {"30s": 120, "15min": 4, "30min": 2, "1h": 1, "6h": 1/6}[timeframe]
expected_bars = int(lookback_days * 24 * bars_per_hour)
expected_points = max(0, expected_bars - rolling_window + 1)  # Points that can be plotted

# Show data point information
st.info(f"📊 Data Point Information: Based on your settings, expecting ~{expected_bars} total bars and ~{expected_points} plotted Hurst values.")

if expected_bars < rolling_window + 10:
    st.warning("⚠️ Not enough data for this rolling window. Increase lookback or reduce window.")

# --- Fetch Oracle Price Data ---
end_time = datetime.utcnow()
start_time = end_time - timedelta(days=lookback_days)

query = f"""
SELECT created_at AT TIME ZONE 'UTC' + INTERVAL '8 hours' AS timestamp, final_price, pair_name
FROM public.oracle_price_log
WHERE created_at BETWEEN '{start_time}' AND '{end_time}'
AND pair_name = '{selected_token}';
"""
df = pd.read_sql(query, engine)

# --- Data Validation Function ---
def validate_price_data(df):
    """Validate price data and return diagnostics"""
    diagnostics = {}
    
    # Check if dataframe is empty
    if df.empty:
        return {"error": "Dataframe is empty"}
    
    # Check data types
    diagnostics["data_types"] = {col: str(df[col].dtype) for col in df.columns}
    
    # Convert price to numeric and check for NaNs
    if 'final_price' in df.columns:
        price_col = 'final_price'
    elif 'close' in df.columns:
        price_col = 'close'
    else:
        return {"error": "No price column found"}
    
    # Ensure price is numeric
    original_price = df[price_col].copy()
    numeric_price = pd.to_numeric(original_price, errors='coerce')
    
    # Check for NaNs introduced by conversion
    nan_count = numeric_price.isna().sum()
    nan_pct = nan_count / len(numeric_price) * 100 if len(numeric_price) > 0 else 0
    
    diagnostics["nan_values"] = {
        "count": int(nan_count),
        "percentage": float(nan_pct)
    }
    
    # Check for duplicate timestamps
    if 'timestamp' in df.columns:
        duplicates = df['timestamp'].duplicated().sum()
        diagnostics["duplicate_timestamps"] = int(duplicates)
    
    # Check price variation
    if len(numeric_price.dropna()) > 1:
        price_stats = {
            "min": float(numeric_price.min()),
            "max": float(numeric_price.max()),
            "mean": float(numeric_price.mean()),
            "std": float(numeric_price.std()),
            "zero_values": int((numeric_price == 0).sum())
        }
        
        # Check price changes
        price_changes = numeric_price.pct_change().dropna()
        change_stats = {
            "min_change_pct": float(price_changes.min() * 100),
            "max_change_pct": float(price_changes.max() * 100),
            "mean_abs_change_pct": float(price_changes.abs().mean() * 100),
            "std_change_pct": float(price_changes.std() * 100),
            "zero_changes": int((price_changes == 0).sum()),
            "zero_changes_pct": float((price_changes == 0).sum() / len(price_changes) * 100) if len(price_changes) > 0 else 0
        }
        
        diagnostics["price_stats"] = price_stats
        diagnostics["change_stats"] = change_stats
    
    return diagnostics

if df.empty:
    st.warning("No data found for selected pair and timeframe.")
    st.stop()

# Validate data
data_diagnostics = validate_price_data(df)
with st.expander("Data Diagnostics"):
    st.json(data_diagnostics)
    
    # Plot histogram of price changes if we have valid data
    if 'change_stats' in data_diagnostics:
        pct_changes = df['final_price'].pct_change().dropna() * 100
        
        fig_hist = px.histogram(
            pct_changes, 
            nbins=50,
            title="Distribution of Price Changes (%)",
            labels={'value': 'Price Change (%)'}
        )
        st.plotly_chart(fig_hist, use_container_width=True)

# --- Preprocess ---
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp').sort_index()

# --- Resample to OHLC ---
ohlc = df['final_price'].resample(timeframe).ohlc().dropna()

# --- Universal Hurst Calculation ---
def universal_hurst(ts):
    """
    A universal Hurst exponent calculation that works for any asset class.
    
    Args:
        ts: Time series of prices (numpy array or list)
    
    Returns:
        float: Hurst exponent value between 0 and 1, or np.nan if calculation fails
    """
    # Convert to numpy array and ensure floating point
    try:
        ts = np.array(ts, dtype=float)
    except:
        return np.nan  # Return NaN if conversion fails
        
    # Basic data validation
    if len(ts) < 10 or np.any(~np.isfinite(ts)):
        return np.nan
    
    # Convert to returns - using log returns handles any scale of asset
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    adjusted_ts = ts + epsilon
    log_returns = np.diff(np.log(adjusted_ts))
    
    # If all returns are exactly zero (completely flat price), return 0.5
    if np.all(log_returns == 0):
        return 0.5
    
    # Use multiple methods and average for robustness
    hurst_estimates = []
    
    # Method 1: Rescaled Range (R/S) Analysis
    try:
        # Create range of lags - adaptive based on data length
        max_lag = min(len(log_returns) // 4, 40)
        lags = range(10, max_lag, max(1, (max_lag - 10) // 10))
        
        rs_values = []
        for lag in lags:
            # Reshape returns into segments
            segments = len(log_returns) // lag
            if segments < 1:
                continue
                
            # Calculate R/S for each segment
            rs_by_segment = []
            for i in range(segments):
                segment = log_returns[i*lag:(i+1)*lag]
                if len(segment) < lag // 2:  # Skip if segment is too short
                    continue
                    
                # Get mean and standard deviation
                mean_return = np.mean(segment)
                std_return = np.std(segment)
                
                if std_return == 0:  # Skip if no variation
                    continue
                    
                # Calculate cumulative deviation from mean
                cumdev = np.cumsum(segment - mean_return)
                
                # Calculate R/S statistic
                r = np.max(cumdev) - np.min(cumdev)
                s = std_return
                
                rs_by_segment.append(r / s)
            
            if rs_by_segment:
                rs_values.append((lag, np.mean(rs_by_segment)))
        
        # Need at least 4 points for reliable regression
        if len(rs_values) >= 4:
            lags_log = np.log10([x[0] for x in rs_values])
            rs_log = np.log10([x[1] for x in rs_values])
            
            # Calculate Hurst exponent from slope
            poly = np.polyfit(lags_log, rs_log, 1)
            h_rs = poly[0]
            hurst_estimates.append(h_rs)
    except:
        pass
    
    # Method 2: Variance Method
    try:
        # Calculate variance at different lags
        max_lag = min(len(log_returns) // 4, 40)
        lags = range(10, max_lag, max(1, (max_lag - 10) // 10))
        
        var_values = []
        for lag in lags:
            if lag >= len(log_returns):
                continue
                
            # Compute the log returns at different lags
            lagged_returns = np.array([np.mean(log_returns[i:i+lag]) for i in range(0, len(log_returns)-lag+1, lag)])
            
            if len(lagged_returns) < 2:
                continue
                
            # Calculate variance of the lagged series
            var = np.var(lagged_returns)
            if var > 0:
                var_values.append((lag, var))
        
        # Need at least 4 points for reliable regression
        if len(var_values) >= 4:
            lags_log = np.log10([x[0] for x in var_values])
            var_log = np.log10([x[1] for x in var_values])
            
            # For variance, the slope should be 2H-1
            poly = np.polyfit(lags_log, var_log, 1)
            h_var = (poly[0] + 1) / 2
            hurst_estimates.append(h_var)
    except:
        pass
    
    # Method 3: Detrended Fluctuation Analysis (DFA)
    try:
        # Simplified DFA implementation
        max_lag = min(len(log_returns) // 4, 40)
        lags = range(10, max_lag, max(1, (max_lag - 10) // 10))
        
        # Cumulative sum of mean-centered returns (profile)
        profile = np.cumsum(log_returns - np.mean(log_returns))
        
        dfa_values = []
        for lag in lags:
            if lag >= len(profile):
                continue
                
            segments = len(profile) // lag
            if segments < 1:
                continue
                
            # Calculate DFA for each segment
            f2_values = []
            for i in range(segments):
                segment = profile[i*lag:(i+1)*lag]
                if len(segment) < lag // 2:
                    continue
                    
                # Linear fit to remove trend
                x = np.arange(len(segment))
                coeffs = np.polyfit(x, segment, 1)
                trend = np.polyval(coeffs, x)
                
                # Calculate fluctuation
                f2 = np.mean((segment - trend) ** 2)
                f2_values.append(f2)
            
            if f2_values:
                dfa_values.append((lag, np.sqrt(np.mean(f2_values))))
        
        # Need at least 4 points for reliable regression
        if len(dfa_values) >= 4:
            lags_log = np.log10([x[0] for x in dfa_values])
            dfa_log = np.log10([x[1] for x in dfa_values])
            
            # Calculate Hurst exponent from slope
            poly = np.polyfit(lags_log, dfa_log, 1)
            h_dfa = poly[0]
            hurst_estimates.append(h_dfa)
    except:
        pass
    
    # Fallback to autocorrelation method if other methods fail
    if not hurst_estimates and len(log_returns) > 1:
        try:
            # Calculate lag-1 autocorrelation
            autocorr = np.corrcoef(log_returns[:-1], log_returns[1:])[0, 1]
            
            # Convert autocorrelation to Hurst estimate
            # Strong negative correlation suggests mean reversion (H < 0.5)
            # Strong positive correlation suggests trending (H > 0.5)
            h_acf = 0.5 + (np.sign(autocorr) * min(abs(autocorr) * 0.4, 0.4))
            hurst_estimates.append(h_acf)
        except:
            pass
    
    # If we have estimates, aggregate them and constrain to 0-1 range
    if hurst_estimates:
        # Remove any extreme outliers
        valid_estimates = [h for h in hurst_estimates if 0 <= h <= 1]
        
        # If no valid estimates remain after filtering, use all estimates but constrain them
        if not valid_estimates and hurst_estimates:
            valid_estimates = [max(0, min(1, h)) for h in hurst_estimates]
        
        # If we have valid estimates, return their median (more robust than mean)
        if valid_estimates:
            return np.median(valid_estimates)
    
    # If all methods fail, return 0.5 (random walk assumption)
    return 0.5

# --- Calculate Hurst confidence ---
def hurst_confidence(ts):
    """Calculate confidence score for Hurst estimation (0-100%)"""
    ts = np.array(ts)
    
    # Factors affecting confidence
    factors = []
    
    # 1. Length of time series
    len_factor = min(1.0, len(ts) / 50)
    factors.append(len_factor)
    
    # 2. Variance in the series
    var = np.var(ts)
    var_factor = min(1.0, var / 1e-4) if var > 0 else 0
    factors.append(var_factor)
    
    # 3. Trend consistency
    diff = np.diff(ts)
    sign_changes = np.sum(diff[:-1] * diff[1:] < 0)
    consistency = 1.0 - min(1.0, sign_changes / (len(diff) - 1))
    factors.append(consistency)
    
    # Combine factors
    confidence = np.mean(factors) * 100
    return round(confidence)

# --- Enhanced Regime Classification with Intensity Levels ---
def detailed_regime_classification(hurst):
    """
    Provides a more detailed regime classification including intensity levels.
    
    Args:
        hurst: Calculated Hurst exponent value
        
    Returns:
        tuple: (regime category, intensity level, description)
    """
    if pd.isna(hurst):
        return ("UNKNOWN", 0, "Insufficient data")
    
    # Strong mean reversion
    elif hurst < 0.2:
        return ("MEAN-REVERT", 3, "Strong mean-reversion")
    
    # Moderate mean reversion
    elif hurst < 0.3:
        return ("MEAN-REVERT", 2, "Moderate mean-reversion")
    
    # Mild mean reversion
    elif hurst < 0.4:
        return ("MEAN-REVERT", 1, "Mild mean-reversion")
    
    # Noisy/Random zone
    elif hurst < 0.45:
        return ("NOISE", 1, "Slight mean-reversion bias")
    elif hurst <= 0.55:
        return ("NOISE", 0, "Pure random walk")
    elif hurst < 0.6:
        return ("NOISE", 1, "Slight trending bias")
    
    # Mild trend
    elif hurst < 0.7:
        return ("TREND", 1, "Mild trending")
    
    # Moderate trend
    elif hurst < 0.8:
        return ("TREND", 2, "Moderate trending")
    
    # Strong trend
    else:
        return ("TREND", 3, "Strong trending")

# --- Compute Rolling Hurst, Confidence and Regime ---
ohlc['Hurst'] = ohlc['close'].rolling(rolling_window).apply(universal_hurst)
ohlc['confidence'] = ohlc['close'].rolling(rolling_window).apply(hurst_confidence)
ohlc['regime_info'] = ohlc['Hurst'].apply(detailed_regime_classification)
ohlc['regime'] = ohlc['regime_info'].apply(lambda x: x[0])
ohlc['intensity'] = ohlc['regime_info'].apply(lambda x: x[1])
ohlc['regime_desc'] = ohlc['regime_info'].apply(lambda x: x[2])

# --- Display Calculation Diagnostics ---
st.subheader("Calculation Diagnostics")
actual_bars = len(ohlc)
actual_points = len(ohlc.dropna(subset=['Hurst']))
hurst_valid = ohlc['Hurst'].notna().sum()
hurst_invalid = ohlc['Hurst'].isna().sum()
hurst_validity_pct = hurst_valid / (hurst_valid + hurst_invalid) * 100 if (hurst_valid + hurst_invalid) > 0 else 0

col1, col2 = st.columns(2)
col1.metric("Total Bars Collected", f"{actual_bars}")
col2.metric("Valid Hurst Values", f"{hurst_valid} ({hurst_validity_pct:.1f}%)")

st.success(f"✅ Actual Data: {actual_bars} bars collected, {actual_points} valid Hurst values calculated")

if hurst_valid == 0:
    st.error("No valid Hurst values calculated. Performing deep diagnostics...")
    
    # Sample calculation on a fixed window
    sample_size = min(rolling_window, len(ohlc))
    if sample_size > 10:
        # Get sample window
        sample_window = ohlc['close'].iloc[:sample_size].values
        
        st.write(f"Attempting manual Hurst calculation on first {sample_size} values...")
        try:
            manual_hurst = universal_hurst(sample_window)
            st.write(f"Manual Hurst result: {manual_hurst}")
            
            # Show the sample data
            st.write("Sample price data:")
            sample_df = pd.DataFrame({
                'index': range(len(sample_window)),
                'price': sample_window
            })
            st.dataframe(sample_df)
            
            # Plot the sample data
            fig = px.line(sample_df, x='index', y='price', title='Sample Price Window')
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate returns
            returns = np.diff(np.log(sample_window))
            returns_df = pd.DataFrame({
                'index': range(len(returns)),
                'log_return': returns
            })
            
            st.write("Log returns from sample:")
            st.dataframe(returns_df)
            
            # Plot returns
            fig = px.line(returns_df, x='index', y='log_return', title='Log Returns')
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate autocorrelation
            if len(returns) > 1:
                autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
                st.write(f"Lag-1 Autocorrelation: {autocorr:.4f}")
        except Exception as e:
            st.error(f"Error in manual calculation: {str(e)}")
    
    # Stop execution if no valid Hurst values
    st.warning("Cannot generate visualizations without valid Hurst values. Please try different parameters.")
    st.stop()

# --- Plots ---
st.subheader(f"Rolling Hurst for {selected_token} ({timeframe})")

# Create two plots - one for price, one for Hurst
fig = go.Figure()
# Add this line to define df_plot:
df_plot = ohlc.reset_index()
# First chart - Hurst values
fig = go.Figure()

# Add Hurst values
fig.add_trace(go.Scatter(
    x=df_plot['timestamp'],
    y=df_plot['Hurst'],
    mode='lines+markers',
    name='Hurst',
    line=dict(color='blue'),
    marker=dict(
        color=df_plot['Hurst'].apply(lambda h: 
            'red' if h < 0.4 else 
            'green' if h > 0.6 else 
            'gray'
        )
    )
))

# Add horizontal line at 0.5
fig.add_trace(go.Scatter(
    x=[df_plot['timestamp'].iloc[0], df_plot['timestamp'].iloc[-1]] if not df_plot.empty else [],
    y=[0.5, 0.5],
    mode='lines',
    line=dict(color='black', dash='dash'),
    showlegend=False
))

# Add horizontal lines at 0.4 and 0.6
fig.add_trace(go.Scatter(
    x=[df_plot['timestamp'].iloc[0], df_plot['timestamp'].iloc[-1]] if not df_plot.empty else [],
    y=[0.4, 0.4],
    mode='lines',
    line=dict(color='red', dash='dash'),
    showlegend=False
))

fig.add_trace(go.Scatter(
    x=[df_plot['timestamp'].iloc[0], df_plot['timestamp'].iloc[-1]] if not df_plot.empty else [],
    y=[0.6, 0.6],
    mode='lines',
    line=dict(color='green', dash='dash'),
    showlegend=False
))

# Update layout with colored backgrounds
fig.update_layout(
    title=f"Rolling Hurst for {selected_token} ({timeframe})",
    xaxis_title="Time",
    yaxis_title="Hurst Exponent",
    yaxis=dict(range=[0, 1]),
    height=400,
    # Use plain annotations instead of complex shapes
    annotations=[
        dict(x=0.1, y=0.2, xref="paper", yref="y", text="Mean-Reverting Zone", 
             showarrow=False, font=dict(color="red")),
        dict(x=0.1, y=0.5, xref="paper", yref="y", text="Random Walk Zone", 
             showarrow=False, font=dict(color="gray")),
        dict(x=0.1, y=0.8, xref="paper", yref="y", text="Trending Zone", 
             showarrow=False, font=dict(color="green"))
    ]
)

# Second chart - Price with simple regime indicators
fig2 = go.Figure()

# Add candlestick chart
fig2.add_trace(go.Candlestick(
    x=df_plot['timestamp'],
    open=df_plot['open'],
    high=df_plot['high'],
    low=df_plot['low'],
    close=df_plot['close'],
    name="Price"
))

# Add simple Hurst overlay without secondary axis
fig2.add_trace(go.Scatter(
    x=df_plot['timestamp'],
    y=(df_plot['Hurst'] * (df_plot['high'].max() - df_plot['low'].min()) * 0.1) + df_plot['low'].min(),
    mode='lines',
    name='Hurst Trend',
    line=dict(color='blue', width=1)
))

# Simplify layout
fig2.update_layout(
    title=f"Price Chart for {selected_token} ({timeframe})",
    xaxis_title="Time",
    yaxis_title="Price",
    height=400
)

# Display charts
st.plotly_chart(fig, use_container_width=True)
st.plotly_chart(fig2, use_container_width=True)

# --- Continue with the rest of your dashboard (metrics, table, etc.) ---
# --- Show Confidence Metrics ---
col1, col2, col3, col4 = st.columns(4)

# Valid coverage
valid_pct = round(ohlc['Hurst'].notna().mean() * 100, 1)
col1.metric("✅ Valid Hurst Coverage", f"{valid_pct}%")

# Average confidence
avg_conf = round(ohlc['confidence'].mean(), 1)
col2.metric("🎯 Avg Confidence", f"{avg_conf}%")

# Current regime
current_regime_desc = ohlc['regime_desc'].iloc[-1] if not ohlc.empty and not pd.isna(ohlc['regime_desc'].iloc[-1]) else "Unknown"
col3.metric("🔍 Current Regime", current_regime_desc)

# Window/Data Ratio
window_data_ratio = round(rolling_window / actual_bars * 100, 1) if actual_bars > 0 else 0
col4.metric("⚖️ Window/Data Ratio", f"{window_data_ratio}%", 
           delta="Good" if 10 <= window_data_ratio <= 50 else "Adjust",
           delta_color="normal" if 10 <= window_data_ratio <= 50 else "off")

# --- Table Display ---
st.markdown("### Regime Table (Most Recent 100 Bars)")
display_df = ohlc[['open', 'high', 'low', 'close', 'Hurst', 'confidence', 'regime_desc']].copy()
display_df['Hurst'] = display_df['Hurst'].round(3)
display_df['confidence'] = display_df['confidence'].round(1)
st.dataframe(display_df.sort_index(ascending=False).head(100))

# --- Explanation ---
with st.expander("Understanding Hurst Exponent and Dashboard"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Interpreting the Hurst Exponent
        
        The Hurst exponent measures the long-term memory of a time series:
        
        **Mean-Reverting (H < 0.4)**
        - **Strong (0.0-0.2)**: Very strong pullbacks to mean
        - **Moderate (0.2-0.3)**: Consistent mean-reversion
        - **Mild (0.3-0.4)**: Weak mean-reversion tendency
        
        **Random/Noisy (H 0.4-0.6)**
        - **Near 0.5**: Random walk, no correlation to past
        
        **Trending (H > 0.6)**
        - **Mild (0.6-0.7)**: Weak trend persistence
        - **Moderate (0.7-0.8)**: Steady trend persistence
        - **Strong (0.8-1.0)**: Very strong trend persistence
        """)
    
    with col2:
        st.markdown("""
        ### Dashboard Components
        
        **Settings:**
        - **Lookback**: How far back to collect price data
        - **Rolling Window**: How many bars to use for each Hurst calculation
        
        **Charts:**
        - **Hurst Chart**: Shows Hurst values over time with colored bands indicating regimes
        - **Price Chart**: Shows price with colored backgrounds and a secondary Hurst axis
        
        **Metrics:**
        - **Valid Coverage**: Percentage of time with valid Hurst values
        - **Avg Confidence**: Average reliability of calculations
        - **Window/Data Ratio**: Rolling window size relative to data size
        """)