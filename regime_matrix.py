import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import plotly.graph_objects as go
import plotly.express as px
from sqlalchemy import create_engine
import concurrent.futures
from functools import lru_cache

# --- Setup ---
st.set_page_config(layout="wide")
st.title("📈 Currency Pair Trend Matrix Dashboard")

# Create tabs for Matrix View, Summary Table, Filters/Settings, and Global Summary
tab1, tab2, tab3, tab4 = st.tabs(["Matrix View", "Pair-Specific Summary Table", "Filter by Regime", "Global Regime Summary"])

# --- DB CONFIG ---
@st.cache_resource
def get_database_connection():
    db_config = st.secrets["database"]
    db_uri = f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    return create_engine(db_uri,isolation_level="AUTOCOMMIT")

engine = get_database_connection()

# --- Detailed Regime Color Map with Intensities ---
color_map = {
    "MEAN-REVERT": {
        3: "rgba(255,0,0,0.7)",      # Strong Mean-Reversion
        2: "rgba(255,50,50,0.6)",    # Moderate Mean-Reversion
        1: "rgba(255,100,100,0.5)",  # Mild Mean-Reversion
        0: "rgba(255,150,150,0.4)"   # (Not used)
    },
    "NOISE": {
        0: "rgba(200,200,200,0.5)",  # Pure Random Walk
        1: "rgba(220,220,255,0.4)"   # Slight bias (either direction)
    },
    "TREND": {
        3: "rgba(0,180,0,0.7)",      # Strong Trend
        2: "rgba(50,200,50,0.6)",    # Moderate Trend
        1: "rgba(100,220,100,0.5)",  # Mild Trend
        0: "rgba(150,255,150,0.4)"   # (Not used)
    }
}

# Emoji indicators for regimes
regime_emojis = {
    "Strong mean-reversion": "⬇️⬇️⬇️",
    "Moderate mean-reversion": "⬇️⬇️",
    "Mild mean-reversion": "⬇️",
    "Slight mean-reversion bias": "↘️",
    "Pure random walk": "↔️",
    "Slight trending bias": "↗️",
    "Mild trending": "⬆️",
    "Moderate trending": "⬆️⬆️",
    "Strong trending": "⬆️⬆️⬆️",
    "Insufficient data": "❓",
}

# --- Optimized Hurst & Regime Logic ---
@lru_cache(maxsize=128)
def universal_hurst(ts_tuple):
    """
    A universal Hurst exponent calculation that works for any asset class.
    
    Args:
        ts_tuple: Time series of prices as a tuple (for caching)
    
    Returns:
        float: Hurst exponent value between 0 and 1, or np.nan if calculation fails
    """
    # Convert tuple to numpy array
    ts = np.array(ts_tuple, dtype=float)
    
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
    
    # Use lag-1 autocorrelation as primary method (fastest)
    try:
        if len(log_returns) > 1:
            autocorr = np.corrcoef(log_returns[:-1], log_returns[1:])[0, 1]
            h_acf = 0.5 + (np.sign(autocorr) * min(abs(autocorr) * 0.4, 0.4))
            return max(0, min(1, h_acf))  # Constrain to [0,1]
    except:
        pass
    
    # If autocorrelation fails, return random walk assumption
    return 0.5

def batch_calculate_hurst(df, window_size):
    """
    Calculate Hurst exponent for all windows at once instead of rolling
    
    Args:
        df: DataFrame with 'close' column
        window_size: Size of the rolling window
        
    Returns:
        Series: Hurst values for each window
    """
    n = len(df)
    if n < window_size:
        return pd.Series([np.nan] * n)
    
    hurst_values = []
    
    # For first window-1 positions, we don't have enough data
    hurst_values.extend([np.nan] * (window_size - 1))
    
    # Calculate Hurst for each complete window
    for i in range(window_size - 1, n):
        window = df.iloc[i - window_size + 1:i + 1]['close'].values
        hurst = universal_hurst(tuple(window))
        hurst_values.append(hurst)
    
    return pd.Series(hurst_values, index=df.index)

# --- Calculate Hurst confidence ---
def hurst_confidence(ts):
    """Calculate confidence score for Hurst estimation (0-100%)"""
    ts = np.array(ts)
    
    # Simple factors affecting confidence
    len_factor = min(1.0, len(ts) / 50)
    var = np.var(ts) if len(ts) > 1 else 0
    var_factor = min(1.0, var / 1e-4) if var > 0 else 0
    
    # Simple confidence calculation
    confidence = np.mean([len_factor, var_factor]) * 100
    return round(confidence)

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
    
def get_recommended_settings(timeframe):
    """Returns recommended lookback and window settings for a given timeframe"""
    recommendations = {
        "30s": {"lookback_min": 1, "lookback_ideal": 2, "window_min": 30, "window_ideal": 50},
        "15min": {"lookback_min": 2, "lookback_ideal": 3, "window_min": 20, "window_ideal": 30},
        "30min": {"lookback_min": 3, "lookback_ideal": 4, "window_min": 20, "window_ideal": 30},
        "1h": {"lookback_min": 5, "lookback_ideal": 7, "window_min": 20, "window_ideal": 30},
        "4h": {"lookback_min": 10, "lookback_ideal": 14, "window_min": 20, "window_ideal": 30},
        "6h": {"lookback_min": 14, "lookback_ideal": 21, "window_min": 20, "window_ideal": 30}
    }
    
    return recommendations.get(timeframe, {"lookback_min": 3, "lookback_ideal": 7, "window_min": 20, "window_ideal": 30})

# --- Bulk data fetching ---
@st.cache_data(ttl=300)  # Cache data for 5 minutes
def fetch_price_data_bulk(pairs, lookback_days):
    """
    Fetch price data for multiple pairs in a single query
    
    Args:
        pairs: List of pairs to fetch
        lookback_days: Days to look back
        
    Returns:
        DataFrame: Prices for all pairs
    """
    if not pairs:
        return pd.DataFrame()
        
    # Format the IN clause with proper SQL escaping
    pairs_str = "','".join(pairs)
    
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=lookback_days)

    query = f"""
    SELECT created_at AT TIME ZONE 'UTC' + INTERVAL '8 hours' AS timestamp, 
           final_price, pair_name
    FROM public.oracle_price_log
    WHERE created_at BETWEEN '{start_time}' AND '{end_time}'
    AND pair_name IN ('{pairs_str}');
    """
    
    try:
        df = pd.read_sql(query, engine)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return pd.DataFrame()

def process_pair_data(df, timeframe, rolling_window):
    """
    Process data for a single pair and timeframe
    
    Args:
        df: DataFrame with price data for a single pair
        timeframe: Timeframe for resampling
        rolling_window: Size of rolling window
        
    Returns:
        DataFrame: Processed OHLC data with Hurst values
    """
    if df.empty:
        return None
    
    # Convert timestamp and sort
    df = df.set_index('timestamp').sort_index()
    
    # Resample to OHLC
    ohlc = df['final_price'].resample(timeframe).ohlc().dropna()
    
    # If not enough data for the window size, return None
    if len(ohlc) < rolling_window:
        return None
    
    # Calculate Hurst with optimized batch calculation
    ohlc['Hurst'] = batch_calculate_hurst(ohlc, rolling_window)
    
    # Calculate confidence only for the last value to save processing
    last_window = ohlc['close'].iloc[-rolling_window:].values if len(ohlc) >= rolling_window else []
    last_confidence = hurst_confidence(last_window) if len(last_window) == rolling_window else 0
    ohlc['confidence'] = 0  # Initialize
    if len(ohlc) >= rolling_window:
        ohlc.iloc[-1, ohlc.columns.get_loc('confidence')] = last_confidence
    
    # Apply the enhanced regime classification only to the last value
    last_hurst = ohlc['Hurst'].iloc[-1] if not ohlc.empty else np.nan
    last_regime_info = detailed_regime_classification(last_hurst)
    
    # Initialize regime columns
    ohlc['regime'] = np.nan
    ohlc['intensity'] = np.nan
    ohlc['regime_desc'] = np.nan
    
    # Set the last values
    if not ohlc.empty:
        ohlc.iloc[-1, ohlc.columns.get_loc('regime')] = last_regime_info[0]
        ohlc.iloc[-1, ohlc.columns.get_loc('intensity')] = last_regime_info[1]
        ohlc.iloc[-1, ohlc.columns.get_loc('regime_desc')] = last_regime_info[2]
    
    return ohlc

# --- Sidebar Parameters ---
@st.cache_data
def fetch_token_list():
    query = "SELECT DISTINCT pair_name FROM public.oracle_price_log ORDER BY pair_name"
    df = pd.read_sql(query, engine)
    return df['pair_name'].tolist()

# Keep only the most essential controls in the sidebar
all_pairs = fetch_token_list()
selected_pairs = st.sidebar.multiselect("Select Currency Pairs", all_pairs, default=all_pairs[:5] if len(all_pairs) >= 5 else all_pairs)
timeframes = ["30s","15min", "30min", "1h", "4h", "6h"]
selected_timeframes = st.sidebar.multiselect("Select Timeframes", timeframes, default=["15min", "1h"] if "15min" in timeframes and "1h" in timeframes else timeframes[:2] if len(timeframes) >= 2 else timeframes)

# IMPORTANT: Define sliders in sidebar (essential settings)
col1, col2 = st.sidebar.columns(2)
lookback_days = col1.slider("Lookback (Days)", 1, 30, 7)  # Default to 7 days
rolling_window = col2.slider("Rolling Window (Bars)", 20, 100, 30)

# Display dynamic recommendations in sidebar
if selected_timeframes:
    st.sidebar.markdown("### Recommended Settings")
    settings_text = ""
    for tf in selected_timeframes:
        rec = get_recommended_settings(tf)
        settings_text += f"""
        - **{tf}**: {rec['lookback_min']}-{rec['lookback_ideal']} day lookback, {rec['window_min']}-{rec['window_ideal']} bar window
        """
    
    st.sidebar.markdown(settings_text)

    # Auto-suggestion for current settings
    recommended_lookbacks = []
    recommended_windows = []
    
    for tf in selected_timeframes:
        rec = get_recommended_settings(tf)
        recommended_lookbacks.append(rec["lookback_min"])
        recommended_windows.append(rec["window_ideal"])
    
    # Only proceed if we have recommendations
    if recommended_lookbacks and recommended_windows:
        rec_lookback = max(recommended_lookbacks)
        rec_window = min(recommended_windows)
    
        if lookback_days < rec_lookback:
            st.sidebar.warning(f"⚠️ Current lookback ({lookback_days} days) may be too short for {max(selected_timeframes, key=lambda x: get_recommended_settings(x)['lookback_min'])}")
            
            # Add a button to auto-apply the recommended settings
            if st.sidebar.button(f"Apply Recommended Settings ({rec_lookback} days lookback, {rec_window} bar window)"):
                lookback_days = rec_lookback
                rolling_window = rec_window
                st.experimental_rerun()

# --- Color Code Legend ---
with st.sidebar.expander("Legend: Regime Colors", expanded=True):
    st.markdown("""
    ### Mean-Reverting
    - <span style='background-color:rgba(255,0,0,0.7);padding:3px'>**Strong Mean-Reverting ⬇️⬇️⬇️**</span>  
    - <span style='background-color:rgba(255,50,50,0.6);padding:3px'>**Moderate Mean-Reverting ⬇️⬇️**</span>  
    - <span style='background-color:rgba(255,100,100,0.5);padding:3px'>**Mild Mean-Reverting ⬇️**</span>  
    - <span style='background-color:rgba(255,150,150,0.4);padding:3px'>**Slight Mean-Reverting Bias ↘️**</span>  
    
    ### Random/Noise
    - <span style='background-color:rgba(200,200,200,0.5);padding:3px'>**Pure Random Walk ↔️**</span>  
    
    ### Trending
    - <span style='background-color:rgba(150,255,150,0.4);padding:3px'>**Slight Trending Bias ↗️**</span>  
    - <span style='background-color:rgba(100,220,100,0.5);padding:3px'>**Mild Trending ⬆️**</span>  
    - <span style='background-color:rgba(50,200,50,0.6);padding:3px'>**Moderate Trending ⬆️⬆️**</span>  
    - <span style='background-color:rgba(0,180,0,0.7);padding:3px'>**Strong Trending ⬆️⬆️⬆️**</span>  
    """, unsafe_allow_html=True)

# --- Batch Data Processing ---
@st.cache_data(ttl=300)
def get_hurst_data_batch(pairs, timeframes, lookback_days, rolling_window):
    """
    Process Hurst data for multiple pairs and timeframes in batch
    
    Args:
        pairs: List of pairs to process
        timeframes: List of timeframes to process
        lookback_days: Days to look back
        rolling_window: Size of rolling window
        
    Returns:
        dict: Processed data for each pair and timeframe
    """
    if not pairs or not timeframes:
        return {}
    
    # Fetch all price data at once
    bulk_data = fetch_price_data_bulk(pairs, lookback_days)
    
    if bulk_data.empty:
        return {}
    
    # Process each pair and timeframe
    results = {}
    
    for pair in pairs:
        results[pair] = {}
        # Filter data for this pair
        pair_data = bulk_data[bulk_data['pair_name'] == pair].copy()
        
        if pair_data.empty:
            continue
            
        # Drop the pair_name column as it's no longer needed
        pair_data = pair_data.drop(columns=['pair_name'])
        
        for tf in timeframes:
            # Process data for this timeframe
            ohlc = process_pair_data(pair_data, tf, rolling_window)
            results[pair][tf] = ohlc
    
    return results

# --- Analyze Data & Generate Summary ---
def generate_summary_data(batch_results):
    """
    Generate summary data from batch results
    
    Args:
        batch_results: Dict of processed data for each pair and timeframe
        
    Returns:
        list: Summary data for each pair
    """
    summary_data = []
    
    for pair, timeframe_data in batch_results.items():
        pair_data = {"Pair": pair}
        
        for tf, ohlc in timeframe_data.items():
            if ohlc is None or ohlc.empty or pd.isna(ohlc['Hurst'].iloc[-1]):
                pair_data[tf] = {"Hurst": np.nan, "Regime": "UNKNOWN", "Description": "Insufficient data"}
            else:
                # Get last values
                pair_data[tf] = {
                    "Hurst": ohlc['Hurst'].iloc[-1],
                    "Regime": ohlc['regime'].iloc[-1],
                    "Description": ohlc['regime_desc'].iloc[-1],
                    "Emoji": regime_emojis.get(ohlc['regime_desc'].iloc[-1], ""),
                    "Valid_Pct": (ohlc['Hurst'].notna().sum() / len(ohlc)) * 100,
                    "Confidence": ohlc['confidence'].iloc[-1]
                }
        
        summary_data.append(pair_data)
    
    return summary_data

# --- Process data in background ---
if selected_pairs and selected_timeframes:
    # Cache data loading with a status indicator
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
        st.session_state.processing_started = False
    
    if not st.session_state.processing_started:
        st.session_state.processing_started = True
        with st.spinner("Loading data and calculating Hurst exponents..."):
            # Get batch results once
            batch_results = get_hurst_data_batch(selected_pairs, selected_timeframes, lookback_days, rolling_window)
            st.session_state.batch_results = batch_results
            
            # Generate summary data
            summary_data = generate_summary_data(batch_results)
            st.session_state.summary_data = summary_data
            
            st.session_state.data_loaded = True
        
# --- Display Matrix View ---
with tab1:
    # Add a refresh button at the top of the Matrix View tab
    refresh_clicked = st.button("Refresh Analysis")
    if refresh_clicked:
        st.session_state.data_loaded = False
        st.session_state.processing_started = False
        st.experimental_rerun()
    
    if not selected_pairs or not selected_timeframes:
        st.warning("Please select at least one pair and timeframe")
    elif not st.session_state.data_loaded:
        st.info("Processing data... please wait")
    else:
        # Get batch results from session state
        batch_results = st.session_state.batch_results
        
        # Check if current settings match recommendations
        needs_adjustment = False
        problematic_timeframes = []
        
        for tf in selected_timeframes:
            rec = get_recommended_settings(tf)
            if lookback_days < rec["lookback_min"] or rolling_window > rec["window_ideal"] * 1.5:
                needs_adjustment = True
                problematic_timeframes.append(tf)
        
        if needs_adjustment and problematic_timeframes:
            recommendation_text = "Recommended settings:\n"
            for tf in problematic_timeframes:
                rec = get_recommended_settings(tf)
                recommendation_text += f"- {tf}: {rec['lookback_min']}-{rec['lookback_ideal']} day lookback, {rec['window_min']}-{rec['window_ideal']} bar window\n"
            
            st.warning(f"""
            ⚠️ **Your current settings may result in insufficient data for some timeframes.**
            
            {recommendation_text}
            
            Current settings: {lookback_days} days lookback, {rolling_window} bar window
            """)
            
        # Display pairs and charts
        for pair in selected_pairs:
            if pair not in batch_results:
                continue
                
            st.markdown(f"### 📌 {pair}")
            cols = st.columns(len(selected_timeframes))

            for i, tf in enumerate(selected_timeframes):
                with cols[i]:
                    st.markdown(f"**{tf}**")
                    
                    if tf not in batch_results[pair]:
                        st.write("No data")
                        continue
                        
                    ohlc = batch_results[pair][tf]

                    if ohlc is None or ohlc.empty:
                        st.write("No data")
                        continue

                    # Calculate data quality metrics
                    valid_data_pct = (ohlc['Hurst'].notna().sum() / len(ohlc)) * 100
                    
                    # Diagnostic information for insufficient data
                    if valid_data_pct < 30:
                        if len(ohlc) < rolling_window * 2:
                            suggestion = "⚠️ Need more data. Increase lookback period."
                        elif rolling_window > 40:
                            suggestion = "⚠️ Window too large. Try a smaller rolling window."
                        else:
                            suggestion = f"⚠️ Low valid data ({valid_data_pct:.1f}%)."
                        st.warning(suggestion)

                    # Check if we have a valid Hurst value
                    if pd.isna(ohlc['Hurst'].iloc[-1]):
                        st.error("Insufficient data for Hurst calculation")
                        continue
                    
                    # Chart 
                    fig = go.Figure()

                    # Background regime color - simplified for performance
                    # Only color the last few bars for performance
                    last_regime = ohlc['regime'].iloc[-1] if not pd.isna(ohlc['regime'].iloc[-1]) else "UNKNOWN"
                    last_intensity = ohlc['intensity'].iloc[-1] if not pd.isna(ohlc['intensity'].iloc[-1]) else 0
                    
                    if last_regime in color_map and last_intensity in color_map[last_regime]:
                        shade_color = color_map[last_regime][last_intensity]
                    else:
                        shade_color = "rgba(200,200,200,0.3)"
                        
                    # Add a background for the entire chart based on last regime
                    fig.add_shape(
                        type="rect",
                        x0=ohlc.index[0],
                        y0=0,
                        x1=ohlc.index[-1],
                        y1=1,
                        yref="paper",
                        fillcolor=shade_color,
                        opacity=0.5,
                        layer="below",
                        line_width=0
                    )

                    # Price line
                    fig.add_trace(go.Scatter(
                        x=ohlc.index, 
                        y=ohlc['close'], 
                        mode='lines', 
                        line=dict(color='black', width=1.5), 
                        name='Price'))

                    # Add Hurst line on secondary y-axis - only for last few values
                    # Subsample the data for better performance
                    max_points = 100  # Maximum points to plot for performance
                    step = max(1, len(ohlc) // max_points)
                    
                    # Only add Hurst line if we have calculated values
                    valid_hurst = ohlc[ohlc['Hurst'].notna()]
                    if not valid_hurst.empty:
                        # Subsample for performance
                        subsampled_hurst = valid_hurst.iloc[::step]
                        
                        fig.add_trace(go.Scatter(
                            x=subsampled_hurst.index,
                            y=subsampled_hurst['Hurst'],
                            mode='lines',
                            line=dict(color='blue', width=2, dash='dot'),
                            name='Hurst',
                            yaxis='y2'
                        ))

                    # Determine color based on regime
                    if last_regime == "MEAN-REVERT":
                         title_color = "red"
                    elif last_regime == "TREND":
                         title_color = "green"
                    else:
                         title_color = "gray"

                    # Current regime info
                    current_hurst = ohlc['Hurst'].iloc[-1]
                    current_desc = ohlc['regime_desc'].iloc[-1] if not pd.isna(ohlc['regime_desc'].iloc[-1]) else "Unknown"

                    # Add emoji to description
                    emoji = regime_emojis.get(current_desc, "")
                    display_text = f"{current_desc} {emoji}" if not pd.isna(current_hurst) else "Unknown"
                    hurst_text = f"Hurst: {current_hurst:.2f}" if not pd.isna(current_hurst) else "Hurst: n/a"

                     # Add data quality info
                    quality_text = f"Valid data: {valid_data_pct:.1f}%"

                    fig.update_layout(
                        title=dict(
                            text=f"<b>{display_text}</b><br><sub>{hurst_text} | {quality_text}</sub>",
                            font=dict(color=title_color, size=14, family="Arial, sans-serif")
                        ),
                        margin=dict(l=5, r=5, t=60, b=5),
                        height=220,
                        hovermode="x unified",
                        yaxis=dict(
                            title="Price",
                            titlefont=dict(size=10),
                            showgrid=True,
                            gridcolor='rgba(230,230,230,0.5)'
                        ),
                        yaxis2=dict(
                            title="Hurst",
                            titlefont=dict(color="blue", size=10),
                            tickfont=dict(color="blue", size=8),
                            anchor="x",
                            overlaying="y",
                            side="right",
                            range=[0, 1],
                            showgrid=False
                        ),
                        xaxis=dict(
                            showgrid=True,
                            gridcolor='rgba(230,230,230,0.5)'
                        ),
                        showlegend=False,
                        plot_bgcolor='white'
                    )

                    # Add reference lines for Hurst thresholds
                    fig.add_shape(
                         type="line",
                         x0=ohlc.index[0],
                         y0=0.4,
                         x1=ohlc.index[-1],
                         y1=0.4,
                        line=dict(color="red", width=1.5, dash="dash"),
                        yref="y2"
                    )

                    fig.add_shape(
                        type="line",
                        x0=ohlc.index[0],
                        y0=0.6,
                        x1=ohlc.index[-1],
                        y1=0.6,
                        line=dict(color="green", width=1.5, dash="dash"),
                        yref="y2"
                    )

                    st.plotly_chart(fig, use_container_width=True) 

# --- Summary Table ---
with tab2:
    if not selected_pairs or not selected_timeframes:
        st.warning("Please select at least one pair and timeframe")
    elif not st.session_state.data_loaded:
        st.info("Processing data... please wait")
    else:
        # Get summary data from session state
        summary_data = st.session_state.summary_data
        
        if not summary_data:
            st.warning("No summary data available")
        else:
            st.subheader("🔍 Pair-Specific Summary Table")
            
            # Recommended settings based on current data
            st.info(f"""
            ### Data Quality & Recommended Settings
            
            - **Low timeframes (15min)**: 2-3 day lookback, 20-30 bar window
            - **Medium timeframes (1h)**: 5-7 day lookback, 20-30 bar window
            - **High timeframes (6h)**: 14+ day lookback, 20-30 bar window
            
            Your current settings: **{lookback_days} days lookback** with **{rolling_window} bar window**
            """.format(lookback_days, rolling_window))
            
            # Create a formatted HTML table for better visualization
            html_table = "<table style='width:100%; border-collapse: collapse;'>"
            
            # Header row
            html_table += "<tr style='background-color:#f2f2f2;'>"
            html_table += "<th style='padding:10px; border:1px solid #ddd;'>Pair</th>"
            
            for tf in selected_timeframes:
                html_table += f"<th style='padding:10px; border:1px solid #ddd;'>{tf}</th>"
            
            html_table += "</tr>"
            
            # Data rows
            for item in summary_data:
                html_table += "<tr>"
                html_table += f"<td style='padding:10px; border:1px solid #ddd; font-weight:bold;'>{item['Pair']}</td>"
                
                for tf in selected_timeframes:
                    if tf in item:
                        regime_data = item[tf]
                        
                        # Determine cell background color based on regime
                        if "Regime" in regime_data:
                            if regime_data["Regime"] == "MEAN-REVERT":
                                bg_color = "rgba(255,200,200,0.5)"
                            elif regime_data["Regime"] == "TREND":
                                bg_color = "rgba(200,255,200,0.5)"
                            else:
                                bg_color = "rgba(220,220,220,0.3)"
                        else:
                            bg_color = "rgba(220,220,220,0.3)"
                        
                        # Format Hurst value
                        hurst_val = f"{regime_data['Hurst']:.2f}" if "Hurst" in regime_data and not pd.isna(regime_data["Hurst"]) else "n/a"
                        
                        # Add emoji if available
                        emoji = regime_data.get("Emoji", "")
                        
                        # Add data quality info if available
                        valid_pct = regime_data.get("Valid_Pct", 0)
                        quality_text = ""
                        if valid_pct < 30 and valid_pct > 0:
                            quality_text = f"<br><small style='color:orange;'>Low quality: {valid_pct:.1f}%</small>"
                        
                        html_table += f"<td style='padding:10px; border:1px solid #ddd; background-color:{bg_color};'>"
                        html_table += f"{regime_data['Description']} {emoji}<br><small>Hurst: {hurst_val}</small>{quality_text}"
                        html_table += "</td>"
                    else:
                        html_table += "<td style='padding:10px; border:1px solid #ddd;'>No data</td>"
                
                html_table += "</tr>"
            
            html_table += "</table>"
            
            # Display the HTML table
            st.markdown(html_table, unsafe_allow_html=True)
            
            # Add a downloadable CSV version
            csv_data = []
            header = ["Pair"]
            
            for tf in selected_timeframes:
                header.append(f"{tf}_Regime")
                header.append(f"{tf}_Hurst")
                header.append(f"{tf}_Valid_Pct")
            
            csv_data.append(header)
            
            for item in summary_data:
                row = [item["Pair"]]
                
                for tf in selected_timeframes:
                    if tf in item:
                        row.append(item[tf]["Description"])
                        row.append(item[tf]["Hurst"] if "Hurst" in item[tf] and not pd.isna(item[tf]["Hurst"]) else "")
                        row.append(item[tf].get("Valid_Pct", ""))
                    else:
                        row.append("No data")
                        row.append("")
                        row.append("")
                
                csv_data.append(row)
            
            # Convert to DataFrame for download
            csv_df = pd.DataFrame(csv_data[1:], columns=csv_data[0])
            
            # Add download button
            st.download_button(
                label="Download as CSV",
                data=csv_df.to_csv(index=False),
                file_name=f"market_regimes_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )

# --- Filter by Regime Tab ---
with tab3:
    st.header("Find Currency Pairs by Regime")
    
    st.info("""
    This tab allows you to search across **all currency pairs** in the database to find those 
    matching specific market regimes, regardless of the pairs selected in the sidebar.
    """)
    
    # Select timeframe
    filter_timeframe = st.selectbox(
        "Select Timeframe to Analyze",
        timeframes,
        index=timeframes.index("1h") if "1h" in timeframes else 0
    )
    
    # Select regime to filter by
    filter_regime = st.multiselect(
        "Show Only Pairs with These Regimes:", 
        ["All Regimes","Strong mean-reversion", "Moderate mean-reversion", "Mild mean-reversion", "Slight mean-reversion bias",
         "Pure random walk", 
         "Slight trending bias", "Mild trending", "Moderate trending", "Strong trending"],
        default=["All Regimes"]
    )
    
    # Sorting options
    sort_option_filter = st.selectbox(
        "Sort Results By:",
        ["Name", "Most Trending (Highest Hurst)", "Most Mean-Reverting (Lowest Hurst)", "Data Quality"],
        index=0
    )
    
    # Data quality filter
    min_data_quality = st.slider("Minimum Data Quality (%)", 0, 100, 30)
    
    # Use current lookback/window or set custom ones
    use_custom_params = st.checkbox("Use Custom Parameters (instead of sidebar settings)", value=False)
    
    if use_custom_params:
        custom_col1, custom_col2 = st.columns(2)
        custom_lookback = custom_col1.slider("Custom Lookback (Days)", 1, 30, 
                                            get_recommended_settings(filter_timeframe)["lookback_ideal"])
        custom_window = custom_col2.slider("Custom Window (Bars)", 20, 100, 
                                          get_recommended_settings(filter_timeframe)["window_ideal"])
    
    # Button to run the filter
    if st.button("Find Matching Pairs"):
        # Show a spinner while processing
        with st.spinner("Analyzing all currency pairs..."):
            # Get the complete list of pairs from database 
            all_available_pairs = fetch_token_list()
            
            # Determine which parameters to use
            if use_custom_params:
                actual_lookback = custom_lookback
                actual_window = custom_window
            else:
                actual_lookback = lookback_days
                actual_window = rolling_window
            
            # Batch process all pairs with parallelization
            # Show loading visualization
            progress_bar = st.progress(0)
            
            # Process in batches for better performance
            batch_size = 5  # Process 5 pairs at a time
            num_batches = (len(all_available_pairs) + batch_size - 1) // batch_size
            
            regime_results = []
            
            for i in range(0, num_batches):
                # Update progress
                progress_bar.progress(i / num_batches)
                
                # Get batch of pairs
                batch_start = i * batch_size
                batch_end = min((i + 1) * batch_size, len(all_available_pairs))
                batch_pairs = all_available_pairs[batch_start:batch_end]
                
                # Process batch
                batch_results = get_hurst_data_batch(batch_pairs, [filter_timeframe], actual_lookback, actual_window)
                
                # Extract results
                for pair in batch_pairs:
                    if pair not in batch_results or filter_timeframe not in batch_results[pair]:
                        regime_results.append({
                            "Pair": pair,
                            "Regime": "Insufficient data",
                            "Hurst": np.nan,
                            "Data Quality": 0,
                            "Emoji": "❓"
                        })
                        continue
                    
                    ohlc = batch_results[pair][filter_timeframe]
                    
                    if ohlc is None or ohlc.empty or pd.isna(ohlc['Hurst'].iloc[-1]):
                        regime_results.append({
                            "Pair": pair,
                            "Regime": "Insufficient data",
                            "Hurst": np.nan,
                            "Data Quality": 0,
                            "Emoji": "❓"
                        })
                        continue
                    
                    # Calculate data quality
                    data_quality = (ohlc['Hurst'].notna().sum() / len(ohlc)) * 100
                    
                    regime_info = {
                        "Pair": pair,
                        "Regime": ohlc['regime_desc'].iloc[-1] if not pd.isna(ohlc['regime_desc'].iloc[-1]) else "Unknown",
                        "Hurst": ohlc['Hurst'].iloc[-1],
                        "Data Quality": data_quality,
                        "Emoji": regime_emojis.get(ohlc['regime_desc'].iloc[-1], "")
                    }
                    
                    if "All Regimes" in filter_regime or not filter_regime or regime_info["Regime"] in filter_regime:
                        regime_results.append(regime_info)
            
            # Complete progress
            progress_bar.progress(1.0)
            
            # Filter by data quality
            regime_results = [
                result for result in regime_results 
                if result['Data Quality'] >= min_data_quality
            ]
            
            # Sorting logic
            if sort_option_filter == "Most Trending (Highest Hurst)":
                regime_results.sort(key=lambda x: x["Hurst"] if not pd.isna(x["Hurst"]) else -np.inf, reverse=True)
            elif sort_option_filter == "Most Mean-Reverting (Lowest Hurst)":
                regime_results.sort(key=lambda x: x["Hurst"] if not pd.isna(x["Hurst"]) else np.inf)
            elif sort_option_filter == "Data Quality":
                regime_results.sort(key=lambda x: x["Data Quality"], reverse=True)
            else:  # Sort by name
                regime_results.sort(key=lambda x: x["Pair"])
                
            # Display results
            if regime_results:
                st.success(f"Found {len(regime_results)} matching pairs")
                
                # Create a DataFrame for better display
                results_df = pd.DataFrame(regime_results)
                
                # Define a function to apply styling to the table
                def highlight_regimes(val):
                    color = "white"
                    if "mean-reversion" in str(val).lower():
                        color = "rgba(255,200,200,0.5)"
                    elif "trend" in str(val).lower():
                        color = "rgba(200,255,200,0.5)"
                    elif "random" in str(val).lower():
                        color = "rgba(220,220,220,0.5)"
                    return f'background-color: {color}'
                
                # Apply styling and display the table
                st.dataframe(
                    results_df.style.applymap(highlight_regimes, subset=['Regime']),
                    height=600
                )
                
                # Option to select these pairs in the main view
                if st.button(f"Add these {len(regime_results)} pairs to sidebar selection"):
                    # Get the pairs to add
                    pairs_to_add = [item["Pair"] for item in regime_results]
                    # Convert to set to avoid duplicates and combine with existing selection
                    updated_pairs = list(set(selected_pairs + pairs_to_add))
                    # Update session state to persist across reruns
                    if 'selected_pairs' not in st.session_state:
                        st.session_state.selected_pairs = updated_pairs
                    else:
                        st.session_state.selected_pairs = updated_pairs
                    st.experimental_rerun()
            else:
                st.warning("No currency pairs match your filter criteria")

# --- Global Regime Summary Tab ---
# --- Global Regime Summary Tab ---
# --- Global Regime Summary Tab ---
# --- Global Regime Summary Tab ---
# --- Global Regime Summary Tab ---
with tab4:
    st.header("Global Regime Summary")
    
    # Add a simple debug message
    st.write("This tab calculates regimes for all pairs across selected timeframes")
    
    # Independent controls for global summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        global_timeframes = st.multiselect(
            "Select Timeframes", 
            timeframes, 
            default=[timeframes[0]] if timeframes else [],
            key="global_timeframes_select_unique"
        )
    
    with col2:
        global_lookback = st.slider(
            "Lookback (Days)", 
            1, 30, 3,
            key="global_lookback_slider_unique"
        )
    
    with col3:
        global_window = st.slider(
            "Rolling Window (Bars)", 
            20, 100, 25,
            key="global_window_slider_unique"
        )
    
    # Ensure we have a session state key for this tab
    if 'global_results' not in st.session_state:
        st.session_state.global_results = None
    
    # Add a debug option
    debug_mode = st.checkbox("Enable debug mode", value=False, key="global_debug_unique")
    
    # Define styling function properly (outside of any conditional blocks)
    def style_regime_table(df):
        # Create a copy with styled values
        styled_df = df.copy()
        
        # Define CSS for the entire dataframe
        styles = [
            # Increase overall font size
            dict(selector="th", props=[("font-size", "16px"), ("font-weight", "bold"), 
                  ("background-color", "#f2f2f2"), ("padding", "12px")]),
            dict(selector="td", props=[("font-size", "15px"), ("padding", "10px")])
        ]
        
        # Apply styling based on regime
        def color_regimes(val):
            if pd.isna(val):
                return 'background-color: #f0f0f0; color: #666666;'
            
            if "mean-reversion" in str(val).lower():
                intensity = "strong" if "strong" in str(val).lower() else \
                           "moderate" if "moderate" in str(val).lower() else \
                           "mild" if "mild" in str(val).lower() else "slight"
                
                if intensity == "strong":
                    return 'background-color: rgba(255,0,0,0.7); color: white; font-weight: bold;'
                elif intensity == "moderate":
                    return 'background-color: rgba(255,50,50,0.6); color: white; font-weight: bold;'
                elif intensity == "mild":
                    return 'background-color: rgba(255,100,100,0.5); color: black;'
                else:  # slight
                    return 'background-color: rgba(255,150,150,0.4); color: black;'
                    
            elif "trending" in str(val).lower():
                intensity = "strong" if "strong" in str(val).lower() else \
                           "moderate" if "moderate" in str(val).lower() else \
                           "mild" if "mild" in str(val).lower() else "slight"
                
                if intensity == "strong":
                    return 'background-color: rgba(0,180,0,0.7); color: white; font-weight: bold;'
                elif intensity == "moderate":
                    return 'background-color: rgba(50,200,50,0.6); color: white; font-weight: bold;'
                elif intensity == "mild":
                    return 'background-color: rgba(100,220,100,0.5); color: black;'
                else:  # slight
                    return 'background-color: rgba(150,255,150,0.4); color: black;'
                    
            elif "random" in str(val).lower():
                return 'background-color: rgba(200,200,200,0.5); color: black;'
                
            elif "no data" in str(val).lower() or "insufficient" in str(val).lower():
                return 'background-color: #f0f0f0; color: #999999; font-style: italic;'
                
            else:
                return ''
        
        # Return the styled dataframe
        return styled_df.style.set_table_styles(styles).applymap(color_regimes)
    
    # Generate button with a unique key
    generate_button = st.button("Generate Global Regime Summary", key="unique_global_generate_btn")
    
    if debug_mode:
        st.write(f"Button state: {generate_button}")
    
    if generate_button:
        if debug_mode:
            st.write("Button clicked!")
            
        if not global_timeframes:
            st.warning("Please select at least one timeframe")
        else:
            try:
                # Create a placeholder for the progress bar
                progress_placeholder = st.empty()
                progress_bar = progress_placeholder.progress(0)
                
                # Get all pairs for testing
                all_pairs = fetch_token_list()
                if debug_mode:
                    st.write(f"Found {len(all_pairs)} pairs")
                    if debug_mode:
                        all_pairs = all_pairs[:10]  # Limit to 10 pairs for debugging
                        st.write(f"Using {len(all_pairs)} pairs for debug")
                
                # Process in smaller batches for reliability
                batch_size = 5
                rows = []
                
                # Process each batch
                for i in range(0, len(all_pairs), batch_size):
                    batch = all_pairs[i:i+batch_size]
                    progress_bar.progress(i / len(all_pairs) if len(all_pairs) > 0 else 0)
                    
                    if debug_mode:
                        st.write(f"Processing batch {i//batch_size + 1}/{(len(all_pairs) + batch_size - 1)//batch_size}")
                    
                    # Get data for this batch
                    for pair in batch:
                        row = {"Pair": pair}
                        
                        # Process each timeframe
                        for tf in global_timeframes:
                            # Simplified logic - just calculate the Hurst value directly
                            try:
                                # Get price data
                                end_time = datetime.now(timezone.utc)
                                start_time = end_time - timedelta(days=global_lookback)
                                
                                query = f"""
                                SELECT created_at AT TIME ZONE 'UTC' + INTERVAL '8 hours' AS timestamp, 
                                       final_price 
                                FROM public.oracle_price_log 
                                WHERE created_at BETWEEN '{start_time}' AND '{end_time}'
                                AND pair_name = '{pair}';
                                """
                                
                                df = pd.read_sql(query, engine)
                                
                                if df.empty:
                                    row[tf] = "No data"
                                    continue
                                
                                # Process data
                                df['timestamp'] = pd.to_datetime(df['timestamp'])
                                df = df.set_index('timestamp').sort_index()
                                
                                # Resample to OHLC
                                ohlc = df['final_price'].resample(tf).ohlc().dropna()
                                
                                if len(ohlc) < global_window:
                                    row[tf] = "Insufficient data"
                                    continue
                                
                                # Calculate Hurst on the last window
                                last_window = ohlc['close'].tail(global_window).values
                                hurst = universal_hurst(tuple(last_window))
                                
                                if pd.isna(hurst):
                                    row[tf] = "Calculation failed"
                                    continue
                                
                                # Get regime
                                regime_info = detailed_regime_classification(hurst)
                                regime_desc = regime_info[2]
                                emoji = regime_emojis.get(regime_desc, "")
                                
                                # Store result
                                row[tf] = f"{regime_desc} {emoji} (H:{hurst:.2f})"
                                
                            except Exception as e:
                                if debug_mode:
                                    st.write(f"Error processing {pair} - {tf}: {str(e)}")
                                row[tf] = f"Error: {str(e)[:20]}"
                        
                        rows.append(row)
                
                # Complete progress
                progress_bar.progress(1.0)
                
                # Create DataFrame
                results_df = pd.DataFrame(rows)
                
                # Store in session state
                st.session_state.global_results = results_df
                
                # Success message
                st.success(f"Analysis complete for {len(rows)} pairs")
                
                # Display the enhanced table with increased height
                st.dataframe(
                    style_regime_table(results_df),
                    height=800,  # Increase height to show more rows
                    use_container_width=True
                )
                
                # Add option to view full-screen
                st.markdown("""
                <style>
                    .fullscreen-button {
                        background-color: #4CAF50;
                        border: none;
                        color: white;
                        padding: 10px 20px;
                        text-align: center;
                        text-decoration: none;
                        display: inline-block;
                        font-size: 16px;
                        margin: 4px 2px;
                        cursor: pointer;
                        border-radius: 4px;
                    }
                </style>

                <a href="#" class="fullscreen-button" onclick="document.querySelector('iframe[title=\"streamlit_app\"]').requestFullscreen(); return false;">View Table Fullscreen</a>
                """, unsafe_allow_html=True)
                
                # Add download options
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "Download as CSV",
                    data=csv,
                    file_name=f"global_regime_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    key="global_download_btn"
                )
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                if debug_mode:
                    import traceback
                    st.code(traceback.format_exc())
                    
    # Display previous results if they exist
    elif st.session_state.global_results is not None:
        st.success("Showing previously generated results")
        
        # Display using the same styling
        st.dataframe(
            style_regime_table(st.session_state.global_results),
            height=800,
            use_container_width=True
        )
