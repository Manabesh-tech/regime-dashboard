# Save this as pages/07_Market_Response_Analysis.py in your Streamlit app folder

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import pytz

st.set_page_config(
    page_title="Market Response Analysis",
    page_icon="📊",
    layout="wide"
)

# --- DB CONFIG ---
try:
    db_config = st.secrets["database"]
    db_uri = (
        f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}"
        f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    )
    engine = create_engine(
        db_uri,
        isolation_level="AUTOCOMMIT",  # 设置自动提交模式
        pool_size=5,  # 连接池大小
        max_overflow=10,  # 最大溢出连接数
        pool_timeout=30,  # 连接超时时间
        pool_recycle=1800,  # 连接回收时间(30分钟)
        pool_pre_ping=True,  # 使用连接前先测试连接是否有效
        pool_use_lifo=True,  # 使用后进先出,减少空闲连接
        echo=False  # 不打印 SQL 语句
    )
except Exception as e:
    st.error(f"Error connecting to the database: {e}")
    st.stop()

# --- UI Setup ---
st.title("Market Response Analysis")
st.subheader("Analyzing how altcoins respond to BTC movements")

# Global parameters
atr_periods = 14  # Standard ATR uses 14 periods
timeframe = "30min"  # Using 30-minute intervals as requested
lookback_days = 1  # 24 hours
singapore_timezone = pytz.timezone('Asia/Singapore')
epsilon = 1e-10  # Small constant to avoid division by zero

# Get current time in Singapore timezone
now_utc = datetime.now(pytz.utc)
now_sg = now_utc.astimezone(singapore_timezone)
st.write(f"Current Singapore Time: {now_sg.strftime('%Y-%m-%d %H:%M:%S')}")

# Debug controls
enable_debug = st.checkbox("Enable Debug Mode", value=True)

def debug_print(message):
    if enable_debug:
        st.write(f"DEBUG: {message}")

# Function to get current partition table name
def get_current_partition_table():
    today = datetime.now().strftime("%Y%m%d")
    return "oracle_price_log_partition_v1"

# Function to fetch available tokens
@st.cache_data(ttl=600, show_spinner="Fetching tokens...")
def fetch_available_tokens():
    table_name = get_current_partition_table()
    
    query = f"""
    SELECT pair_name, COUNT(*) as row_count
    FROM public.{table_name}
    GROUP BY pair_name
    HAVING COUNT(*) > 100
    ORDER BY row_count DESC
    """
    
    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
            return df['pair_name'].tolist()
    except Exception as e:
        st.error(f"Error fetching tokens: {e}")
        return ["BTC/USDT", "ETH/USDT", "SOL/USDT"]  # Default fallback

all_tokens = fetch_available_tokens()

# Make sure reference token is available
reference_token = "BTC/USDT"
if reference_token not in all_tokens:
    st.warning(f"{reference_token} data not found. Analysis may be limited.")
    # Try to find an alternative BTC token
    btc_alternatives = [t for t in all_tokens if "BTC" in t]
    if btc_alternatives:
        reference_token = btc_alternatives[0]
        st.info(f"Using {reference_token} as reference instead.")
    elif all_tokens:
        reference_token = all_tokens[0]
        st.warning(f"No BTC token found. Using {reference_token} as reference instead.")
    else:
        st.error("No tokens with data found. Cannot perform analysis.")
        st.stop()

# UI Controls
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    # Let user select tokens to display (or select all)
    select_all = st.checkbox("Select All Tokens", value=False)
    
    if select_all:
        selected_tokens = all_tokens
    else:
        default_tokens = [t for t in all_tokens[:10] if t != reference_token][:9]
        default_tokens = [reference_token] + default_tokens
        
        selected_tokens = st.multiselect(
            "Select Tokens", 
            all_tokens,
            default=default_tokens
        )

with col2:
    # Add more configuration options
    atr_periods = st.number_input("ATR Periods", min_value=5, max_value=30, value=14)
    lookback_days = st.number_input("Lookback Days", min_value=1, max_value=7, value=1)

with col3:
    # Add a refresh button
    if st.button("Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.experimental_rerun()
    
    # Add timeframe selector
    timeframe = st.selectbox(
        "Timeframe",
        options=["15min", "30min", "1H", "2H", "4H"],
        index=1  # Default to 30min
    )

if not selected_tokens:
    st.warning("Please select at least one token")
    st.stop()

# Ensure reference token is selected
if reference_token not in selected_tokens:
    selected_tokens = [reference_token] + selected_tokens
    st.info(f"Added {reference_token} to selection as it's required for ratio calculations")

# Function to generate aligned time blocks for the past 24 hours
def generate_aligned_time_blocks(current_time, interval=30):
    """
    Generate fixed time blocks for past 24 hours,
    aligned with standard intervals
    
    Args:
        current_time: Current time in Singapore timezone
        interval: Interval in minutes (default: 30)
    """
    # Determine the number of blocks based on interval
    blocks_per_day = 24 * 60 // interval
    
    # Round down to the nearest interval mark
    minutes_past_hour = current_time.minute
    closest_block = (minutes_past_hour // interval) * interval
    latest_complete_block_end = current_time.replace(
        minute=closest_block, 
        second=0, 
        microsecond=0
    )
    
    # Generate block labels for display
    blocks = []
    for i in range(blocks_per_day):  # 24 hours of blocks
        block_end = latest_complete_block_end - timedelta(minutes=i*interval)
        block_start = block_end - timedelta(minutes=interval)
        block_label = f"{block_start.strftime('%H:%M')}"
        blocks.append((block_start, block_end, block_label))
    
    return blocks

# Convert timeframe to minutes for aligned blocks
interval_map = {"15min": 15, "30min": 30, "1H": 60, "2H": 120, "4H": 240}
interval_minutes = interval_map.get(timeframe, 30)

# Generate aligned time blocks
aligned_time_blocks = generate_aligned_time_blocks(now_sg, interval=interval_minutes)
time_block_labels = [block[2] for block in aligned_time_blocks]

# Calculate ATR for a given DataFrame - Improved for small values
def calculate_atr(ohlc_df, period=14, min_threshold=0.0001):
    """
    Calculate the Average True Range (ATR) for a price series
    - Enhanced to handle small values better
    """
    df = ohlc_df.copy()
    
    # Make sure the input data has enough rows
    if len(df) < period + 1:
        debug_print(f"Warning: Not enough data for ATR calculation. Only {len(df)} rows.")
        
    # Ensure minimum values to prevent division by very small numbers
    df['high'] = df['high'].clip(lower=min_threshold)
    df['low'] = df['low'].clip(lower=min_threshold)
    df['close'] = df['close'].clip(lower=min_threshold)
    
    # Calculate High-Low, High-Close(prev), and Low-Close(prev)
    df['previous_close'] = df['close'].shift(1)
    
    # Use absolute values for True Range calculation
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['previous_close'])
    df['tr3'] = abs(df['low'] - df['previous_close'])
    
    # Calculate the True Range
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Calculate the ATR using exponential weighted average for more responsiveness
    df['atr'] = df['tr'].ewm(span=period, adjust=False).mean()
    
    # For percentage-based ATR - this helps with comparing tokens of different prices
    if 'close' in df.columns:
        df['atr_pct'] = df['atr'] / df['close'] * 100
    
    return df

# Calculate beta using numpy - Improved for stability and handling NaN values
def calculate_beta(token_returns, ref_returns, min_data_points=3):
    """
    Calculate the beta coefficient
    beta = cov(token, reference) / var(reference)
    """
    # Drop NaN values and align the series
    clean_data = pd.concat([token_returns, ref_returns], axis=1).dropna()
    
    if len(clean_data) < min_data_points:  # Need at least X data points for meaningful calculation
        return None, None, None
    
    try:
        token_rets = clean_data.iloc[:, 0].values
        ref_rets = clean_data.iloc[:, 1].values
        
        # Check for all zero values
        if np.all(np.abs(ref_rets) < epsilon) or np.all(np.abs(token_rets) < epsilon):
            return None, None, None
        
        # Calculate covariance and variance with safeguards
        covariance = np.cov(token_rets, ref_rets, ddof=1)[0, 1]
        ref_variance = np.var(ref_rets, ddof=1)
        
        if ref_variance < epsilon:
            return None, None, None
            
        # Calculate beta
        beta = covariance / ref_variance
        
        # Calculate correlation
        correlation = np.corrcoef(token_rets, ref_rets)[0, 1]
        r_squared = correlation ** 2
        
        # Calculate alpha (intercept)
        alpha = np.mean(token_rets) - beta * np.mean(ref_rets)
        
        # Check for extreme values that might indicate calculation problems
        if abs(beta) > 10 or abs(alpha) > 20:
            debug_print(f"Warning: Calculated extreme beta ({beta}) or alpha ({alpha})")
            if abs(beta) > 20:  # Cap extremely high betas
                beta = np.sign(beta) * 20
        
        return beta, alpha, r_squared
    except Exception as e:
        debug_print(f"Error calculating beta: {e}")
        return None, None, None

# Fetch price data from the partition table
@st.cache_data(ttl=600, show_spinner="Fetching data...")
def fetch_price_data(token, start_time_utc, end_time_utc):
    table_name = get_current_partition_table()
    
    query = f"""
    SELECT 
        created_at AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore' AS timestamp, 
        final_price
    FROM public.{table_name}
    WHERE pair_name = '{token}'
    AND created_at BETWEEN '{start_time_utc}' AND '{end_time_utc}'
    ORDER BY created_at
    """
    
    debug_print(f"Executing query for {token}: {query}")
    
    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
            debug_print(f"Found {len(df)} rows for {token}")
            return df
    except Exception as e:
        st.error(f"Error fetching {token} data: {e}")
        return pd.DataFrame()

# Fetch and calculate metrics - With improved handling of small values and data quality checks
@st.cache_data(ttl=600, show_spinner="Calculating metrics...")
def fetch_and_calculate_metrics(token):
    # Get time range for data fetch
    now_utc = datetime.now(pytz.utc)
    now_sg = now_utc.astimezone(singapore_timezone)
    start_time_sg = now_sg - timedelta(days=lookback_days + 1)  # Extra day for calculations
    
    # Convert back to UTC for database query
    start_time_utc = start_time_sg.astimezone(pytz.utc)
    end_time_utc = now_sg.astimezone(pytz.utc)

    # Fetch price data
    df = fetch_price_data(token, start_time_utc, end_time_utc)
    
    if df.empty:
        debug_print(f"No data found for {token}")
        return None

    # Prepare data for analysis
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()
    
    # Basic data quality checks
    if df['final_price'].isna().sum() > 0:
        debug_print(f"Warning: {df['final_price'].isna().sum()} NaN values found for {token}")
        df['final_price'] = df['final_price'].fillna(method='ffill')
    
    # Check for price jumps that might indicate bad data
    df['price_change_pct'] = df['final_price'].pct_change().abs() * 100
    suspicious_jumps = df[df['price_change_pct'] > 20]  # Flag >20% moves as suspicious
    if not suspicious_jumps.empty and enable_debug:
        debug_print(f"Potential data quality issues detected for {token}: {len(suspicious_jumps)} suspicious price jumps")
    
    # Check for very small price values
    avg_price = df['final_price'].mean()
    is_very_small_price = avg_price < 0.001
    
    # Apply scaling for very small prices if needed
    scaling_factor = 1
    if is_very_small_price:
        scaling_factor = 1000
        df['final_price'] = df['final_price'] * scaling_factor
        debug_print(f"Applied scaling factor of {scaling_factor} to {token} due to very small price (avg: {avg_price})")
    
    # Show sample of raw data
    if enable_debug:
        debug_print(f"Sample of raw data for {token}:")
        st.dataframe(df.head())
        
        # Show more detailed statistics
        price_stats = {
            'min': df['final_price'].min(),
            'max': df['final_price'].max(),
            'mean': df['final_price'].mean(),
            'median': df['final_price'].median(),
            'std': df['final_price'].std(),
            'is_very_small': is_very_small_price,
            'scaling_factor': scaling_factor
        }
        debug_print(f"Price statistics for {token}: {price_stats}")
    
    # Create OHLC data required for ATR calculation
    df_resampled = df['final_price'].resample(timeframe).ohlc().dropna()
    
    if df_resampled.empty:
        debug_print(f"No OHLC data after resampling for {token}")
        return None
    
    # Show sample of OHLC data
    if enable_debug:
        debug_print(f"Sample of OHLC data for {token}:")
        st.dataframe(df_resampled.head())
    
    # Calculate returns for beta calculation
    df_resampled['returns'] = df_resampled['close'].pct_change() * 100  # percentage returns
    
    # Calculate ATR with improved handling of small values
    df_atr = calculate_atr(df_resampled, period=atr_periods, min_threshold=0.0001*scaling_factor)
    
    # Show ATR statistics
    if enable_debug:
        debug_print(f"ATR statistics for {token}:")
        st.write({
            'min_atr': df_atr['atr'].min(),
            'max_atr': df_atr['atr'].max(),
            'mean_atr': df_atr['atr'].mean(),
            'median_atr': df_atr['atr'].median(),
            'std_atr': df_atr['atr'].std(),
            'atr_as_pct_of_price': (df_atr['atr'].mean() / df_atr['close'].mean()) * 100 if df_atr['close'].mean() > 0 else 0
        })
    
    # Create a DataFrame with the results
    metrics_df = df_atr[['open', 'high', 'low', 'close', 'atr', 'atr_pct', 'returns']].copy()
    metrics_df['token'] = token
    metrics_df['original_datetime'] = metrics_df.index
    metrics_df['time_label'] = metrics_df.index.strftime('%H:%M')
    metrics_df['scaling_factor'] = scaling_factor  # Store scaling factor for reference
    
    # Calculate the 24-hour average ATR
    metrics_df['avg_24h_atr'] = metrics_df['atr'].mean()
    metrics_df['avg_24h_atr_pct'] = metrics_df['atr_pct'].mean()
    
    return metrics_df

# Show the blocks we're analyzing
with st.expander("View Time Blocks Being Analyzed"):
    time_blocks_df = pd.DataFrame([(b[0].strftime('%Y-%m-%d %H:%M'), b[1].strftime('%Y-%m-%d %H:%M'), b[2]) 
                                  for b in aligned_time_blocks], 
                                 columns=['Start Time', 'End Time', 'Block Label'])
    st.dataframe(time_blocks_df)

# Show progress bar while calculating
progress_bar = st.progress(0)
status_text = st.empty()

# Calculate metrics for each token
token_results = {}
for i, token in enumerate(selected_tokens):
    try:
        progress_bar.progress(i / len(selected_tokens))
        status_text.text(f"Processing {token} ({i+1}/{len(selected_tokens)})")
        result = fetch_and_calculate_metrics(token)
        if result is not None:
            token_results[token] = result
    except Exception as e:
        st.error(f"Error processing token {token}: {e}")
        debug_print(f"Full error: {str(e)}")

# Final progress update
progress_bar.progress(1.0)
status_text.text(f"Processed {len(token_results)}/{len(selected_tokens)} tokens successfully")

# Debug - Show ATR values for all tokens for comparison
if enable_debug and token_results:
    st.subheader("Debug - ATR Values Comparison")
    atr_stats = {}
    for token, df in token_results.items():
        atr_stats[token] = {
            'min': df['atr'].min(),
            'max': df['atr'].max(),
            'mean': df['atr'].mean(),
            'median': df['atr'].median(),
            'atr_pct_mean': df['atr_pct'].mean(),  # Show as percentage of price
            'scaling_factor': df['scaling_factor'].iloc[0]  # Show scaling factor
        }
    st.dataframe(pd.DataFrame(atr_stats).T)

# Create tabs for ATR Ratio and Beta analysis
# IMPORTANT: Create tabs BEFORE trying to use them
tab1, tab2 = st.tabs(["ATR Ratio Analysis", "Beta Analysis"])

# Helper function to categorize volatility
def get_volatility_category(ratio):
    if ratio < 0.5:
        return "Much Less Volatile"
    elif ratio < 0.9:
        return "Less Volatile"
    elif ratio < 1.1:
        return "Similar to Reference"
    elif ratio < 2.0:
        return "More Volatile"
    else:
        return "Much More Volatile"

# Helper function to categorize beta
def get_beta_category(beta):
    if beta is None:
        return "Insufficient Data"
    elif beta < 0:
        return "Inverse Response"
    elif beta < 0.3:
        return "Very Low Response"
    elif beta < 0.7:
        return "Low Response"
    elif beta < 1.3:
        return "Medium Response"
    elif beta < 2.0:
        return "High Response"
    else:
        return "Very High Response"

# Define a safe formatting function
def safe_format(val):
    if pd.isna(val):
        return ""
    try:
        return f"{val:.4f}"
    except (ValueError, TypeError):
        return str(val)

# TAB 1: ATR RATIO ANALYSIS
with tab1:
    st.header("ATR Ratio Analysis")
    st.write(f"Analyzing how volatile each token is compared to {reference_token}")
    
    if token_results and reference_token in token_results:
        reference_df = token_results[reference_token]
        
        # Debug - Show reference token ATR
        if enable_debug:
            st.write(f"Reference token ({reference_token}) ATR stats:")
            st.write({
                'min': reference_df['atr'].min(),
                'max': reference_df['atr'].max(),
                'mean': reference_df['atr'].mean(),
                'scaling_factor': reference_df['scaling_factor'].iloc[0]
            })
        
        # Create table data for ATR ratios
        ratio_table_data = {}
        for token, df in token_results.items():
            if token != reference_token:  # Skip reference token as we're comparing others to it
                # Merge with reference token ATR data on the time index
                merged_df = pd.merge(
                    df[['time_label', 'atr', 'atr_pct', 'close']], 
                    reference_df[['time_label', 'atr', 'atr_pct', 'close']], 
                    on='time_label', 
                    suffixes=('', '_ref')
                )
                
                # Debug - Show merged dataframe for first token
                if enable_debug and token == next(iter([t for t in token_results if t != reference_token]), None):
                    st.write(f"Sample of merged data for {token}:")
                    st.dataframe(merged_df.head())
                
                # Calculate the ratio using both raw ATR and percentage-based ATR
                # Using percentage-based ATR helps normalize for price differences
                merged_df['atr_ratio'] = merged_df['atr'] / (merged_df['atr_ref'] + epsilon)
                merged_df['atr_pct_ratio'] = merged_df['atr_pct'] / (merged_df['atr_pct_ref'] + epsilon)
                
                # If using raw ATR ratios, apply scaling factor correction
                # This is important if tokens have very different scales
                token_scale = df['scaling_factor'].iloc[0]
                ref_scale = reference_df['scaling_factor'].iloc[0]
                if token_scale != ref_scale:
                    scale_correction = ref_scale / token_scale
                    merged_df['atr_ratio'] = merged_df['atr_ratio'] * scale_correction
                    debug_print(f"Applied scaling correction of {scale_correction} to ATR ratio for {token}")
                
                # Debug - Check for division by zero or very small numbers
                if enable_debug:
                    zero_refs = (merged_df['atr_ref'] < epsilon).sum()
                    small_refs = (merged_df['atr_ref'] < 0.0001).sum()
                    if zero_refs > 0 or small_refs > 0:
                        st.warning(f"Warning for {token}: {zero_refs} zero reference ATRs, {small_refs} very small reference ATRs")
                
                # Use percentage-based ATR ratio to create the final series
                # This is generally more accurate for comparing tokens with different price ranges
                ratio_series = merged_df.set_index('time_label')['atr_pct_ratio']
                ratio_table_data[token] = ratio_series
        
        # Create DataFrame with all token ratios
        ratio_table = pd.DataFrame(ratio_table_data)
        
        # Apply the time blocks in the proper order (most recent first)
        available_times = set(ratio_table.index)
        ordered_times = [t for t in time_block_labels if t in available_times]
        
        # If no matches are found in aligned blocks, fallback to the available times
        if not ordered_times and available_times:
            ordered_times = sorted(list(available_times), reverse=True)
        
        # Reindex with the ordered times if they exist
        if ordered_times:
            ratio_table = ratio_table.reindex(ordered_times)
        
        # Debug - Show raw ratio table
        if enable_debug:
            st.write("Raw ATR ratio table:")
            st.dataframe(ratio_table)
        
        # Clean table for display - handle inf, -inf and NaN values
        ratio_table_clean = ratio_table.replace([np.inf, -np.inf], np.nan)
        
        # Clip extreme values for better visualization
        ratio_table_clean = ratio_table_clean.clip(lower=0, upper=10)
        
        # Function to color cells based on ATR ratio
        def color_ratio_cells(val):
            if pd.isna(val):
                return 'background-color: #f5f5f5; color: #666666;'  # Grey for missing
            elif val < 0.5:  # Much less volatile than reference
                return 'background-color: rgba(0, 0, 255, 0.7); color: white'  # Blue
            elif val < 0.9:  # Less volatile than reference
                return 'background-color: rgba(173, 216, 230, 0.7); color: black'  # Light blue
            elif val < 1.1:  # Similar to reference
                return 'background-color: rgba(255, 255, 255, 0.7); color: black'  # White/transparent
            elif val < 2.0:  # More volatile than reference
                return 'background-color: rgba(255, 165, 0, 0.7); color: black'  # Orange
            else:  # Much more volatile than reference
                return 'background-color: rgba(255, 0, 0, 0.7); color: white'  # Red
        
        # Try to display styled table with error handling
        try:
            # Show with increased decimal precision using safe formatter
            styled_table = ratio_table_clean.style.applymap(color_ratio_cells).format(formatter=safe_format)
            st.markdown(f"## ATR Ratio Table ({timeframe} timeframe, Last {lookback_days} day(s), Singapore Time)")
            st.markdown(f"### Reference Token: {reference_token}")
            st.markdown("### Color Legend: <span style='color:blue'>Much Less Volatile</span>, <span style='color:lightblue'>Less Volatile</span>, <span style='color:black'>Similar to Reference</span>, <span style='color:orange'>More Volatile</span>, <span style='color:red'>Much More Volatile</span>", unsafe_allow_html=True)
            st.markdown(f"Values shown as ratio of token's ATR to {reference_token}'s ATR within each time interval")
            st.dataframe(styled_table, height=700, use_container_width=True)
        except Exception as e:
            st.error(f"Error displaying styled ratio table: {str(e)}")
            st.write("Displaying unstyled ratio table instead:")
            st.dataframe(ratio_table_clean, height=700, use_container_width=True)
        
        # Create ranking table based on average ATR ratio
        st.subheader(f"ATR Ratio Ranking ({lookback_days}-Day Average, Descending Order)")
        
        ranking_data = []
        for token, ratio_series in ratio_table_data.items():
            if not ratio_series.empty:
                # Filter out outliers for more stable statistics
                filtered_series = ratio_series.replace([np.inf, -np.inf], np.nan).dropna()
                if not filtered_series.empty:
                    filtered_series = filtered_series.clip(lower=0.1, upper=10)
                    
                    avg_ratio = filtered_series.mean()
                    min_ratio = filtered_series.min()
                    max_ratio = filtered_series.max()
                    range_ratio = max_ratio - min_ratio
                    std_ratio = filtered_series.std()
                    cv_ratio = std_ratio / avg_ratio if avg_ratio > 0 else 0  # Coefficient of variation
                    
                    # Calculate time periods where token outperforms reference
                    outperformance_periods = (filtered_series > 1.5).sum()
                    outperformance_pct = (outperformance_periods / len(filtered_series)) * 100 if len(filtered_series) > 0 else 0
                    
                    ranking_data.append({
                        'Token': token,
                        'Avg ATR Ratio': round(avg_ratio, 4),
                        'Max ATR Ratio': round(max_ratio, 4),
                        'Min ATR Ratio': round(min_ratio, 4),
                        'Range': round(range_ratio, 4),
                        'Std Dev': round(std_ratio, 4),
                        'CoV': round(cv_ratio, 4),
                        'Outperform %': round(outperformance_pct, 1),
                        'Volatility Category': get_volatility_category(avg_ratio)
                    })
        
        if ranking_data:
            ranking_df = pd.DataFrame(ranking_data)
            # Sort by average ATR ratio (high to low)
            ranking_df = ranking_df.sort_values(by='Avg ATR Ratio', ascending=False)
            # Add rank column
            ranking_df.insert(0, 'Rank', range(1, len(ranking_df) + 1))
            
            # Reset the index to remove it
            ranking_df = ranking_df.reset_index(drop=True)
            
            # Display the styled dataframe
            st.dataframe(ranking_df, height=500, use_container_width=True)
            
            # Create breakout amplification chart
            st.subheader("Breakout Amplification Analysis")
            
            # Create a bar chart of tokens by average ATR ratio
            tokens_by_ratio = ranking_df.sort_values(by='Avg ATR Ratio', ascending=False).head(15)
            
            fig = px.bar(
                tokens_by_ratio, 
                x='Token', 
                y='Avg ATR Ratio', 
                title=f'Top 15 Tokens by Average ATR Ratio to {reference_token}',
                labels={'Avg ATR Ratio': f'Average ATR Ratio (Token/{reference_token})', 'Token': 'Token'},
                color='Avg ATR Ratio',
                color_continuous_scale='Reds'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Visualize ATR ratio variability
            tokens_by_variability = ranking_df.sort_values(by='CoV', ascending=False).head(15)
            
            fig = px.bar(
                tokens_by_variability, 
                x='Token', 
                y='CoV', 
                title='Top 15 Tokens by ATR Ratio Variability',
                labels={'CoV': 'Coefficient of Variation (Std/Mean)', 'Token': 'Token'},
                color='CoV',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # ATR Ratio explainer
            with st.expander("Understanding ATR Ratio Analysis"):
                st.markdown(f"""
                ### How to Read the ATR Ratio Matrix
                This matrix shows the ATR (Average True Range) of each cryptocurrency pair compared to {reference_token}'s ATR over the last {lookback_days} day(s) using {timeframe} intervals.
                
                **The ATR Ratio is calculated as:**
                ```
                ATR Ratio = Token's ATR / {reference_token}'s ATR
                ```
                
                **For better comparability between tokens of different price ranges, we use:**
                ```
                ATR Percentage Ratio = (Token's ATR / Token's Price) / ({reference_token}'s ATR / {reference_token}'s Price)
                ```
                
                **What the ratios mean:**
                - **Ratio = 1.0**: The token has the same volatility as {reference_token}
                - **Ratio > 1.0**: The token is more volatile than {reference_token}
                - **Ratio < 1.0**: The token is less volatile than {reference_token}
                
                **Color coding:**
                - **Blue** (< 0.5): Much less volatile than {reference_token}
                - **Light Blue** (0.5 - 0.9): Less volatile than {reference_token}
                - **White** (0.9 - 1.1): Similar volatility to {reference_token}
                - **Orange** (1.1 - 2.0): More volatile than {reference_token}
                - **Red** (> 2.0): Much more volatile than {reference_token}
                
                ### Trading Applications
                
                **Breakout Identification:**
                When {reference_token} shows a significant price movement or breakout:
                
                1. Tokens with consistently high ATR ratios (greater than 1.5) are likely to show even larger moves
                2. Tokens with high "Outperform %" values consistently amplify {reference_token}'s movements
                3. The "Range" value shows how much the token's behavior varies
                
                **Coefficient of Variation (CoV):**
                - Lower CoV means the token's volatility relative to {reference_token} is more consistent
                - Higher CoV means the relationship is more variable and less predictable
                """)
        else:
            st.warning("No ranking data available.")
    else:
        st.error(f"Reference token data ({reference_token}) is required for ATR ratio calculations. Please ensure it is selected and data is available.")

# TAB 2: BETA ANALYSIS
with tab2:
    st.header("Beta Analysis")
    st.write(f"Analyzing how much each token moves per 1% move in {reference_token} (accounting for correlation)")
    
    if token_results and reference_token in token_results:
        reference_returns = token_results[reference_token]['returns']
        
        # Debug - Show reference token returns
        if enable_debug:
            st.write(f"Reference token ({reference_token}) returns stats:")
            st.write({
                'min': reference_returns.min(),
                'max': reference_returns.max(),
                'mean': reference_returns.mean(),
                'std': reference_returns.std(),
                'zero_returns_pct': (abs(reference_returns) < epsilon).mean() * 100
            })
        
        # Create table data for betas
        beta_table_data = {}
        beta_values = {}
        
        for token, df in token_results.items():
            if token != reference_token:  # Skip reference token as we're comparing others to it
                token_returns = df['returns']
                
                # Debug - Show token returns
                if enable_debug and token == next(iter([t for t in token_results if t != reference_token]), None):
                    st.write(f"Sample returns for {token}:")
                    st.write({
                        'min': token_returns.min(),
                        'max': token_returns.max(),
                        'mean': token_returns.mean(),
                        'std': token_returns.std(),
                        'zero_returns_pct': (abs(token_returns) < epsilon).mean() * 100
                    })
                
                # Filter out extreme return values for more stable calculations
                max_return = 50  # Cap at 50% for single period return
                filtered_token_returns = token_returns.clip(lower=-max_return, upper=max_return)
                filtered_ref_returns = reference_returns.clip(lower=-max_return, upper=max_return)
                
                # Calculate overall beta for the entire period
                overall_beta, overall_alpha, overall_r_squared = calculate_beta(
                    filtered_token_returns, filtered_ref_returns, min_data_points=5
                )
                
                # Calculate correlation
                clean_data = pd.concat([filtered_token_returns, filtered_ref_returns], axis=1).dropna()
                overall_corr = clean_data.iloc[:, 0].corr(clean_data.iloc[:, 1]) if len(clean_data) > 1 else np.nan
                
                # Calculate rolling betas for each time period
                beta_by_time = {}
                
                # Group data by time label (hour:minute)
                time_groups = df.groupby('time_label')
                
                for time_label, group in time_groups:
                    # Get reference data for the same time period
                    ref_group = token_results[reference_token][token_results[reference_token]['time_label'] == time_label]
                    
                    if not group.empty and not ref_group.empty:
                        # Get returns and filter extremes
                        period_token_returns = group['returns'].clip(lower=-max_return, upper=max_return)
                        period_ref_returns = ref_group['returns'].clip(lower=-max_return, upper=max_return)
                        
                        # Calculate beta for this time period
                        period_beta, _, _ = calculate_beta(
                            period_token_returns, period_ref_returns, min_data_points=3
                        )
                        beta_by_time[time_label] = period_beta
                
                # Convert to series
                beta_series = pd.Series(beta_by_time)
                beta_table_data[token] = beta_series
                
                # Store overall metrics
                beta_values[token] = {
                    'beta': overall_beta,
                    'alpha': overall_alpha,
                    'r_squared': overall_r_squared,
                    'correlation': overall_corr
                }
        
        # Create DataFrame with all token betas
        beta_table = pd.DataFrame(beta_table_data)
        
        # Debug - Show raw beta table
        if enable_debug:
            st.write("Raw beta table:")
            st.dataframe(beta_table)
        
        # Apply the time blocks in the proper order (most recent first)
        available_times = set(beta_table.index)
        ordered_times = [t for t in time_block_labels if t in available_times]
        
        # Reindex with the ordered times if they exist
        if ordered_times:
            beta_table = beta_table.reindex(ordered_times)
        
        # CRITICAL FIX: Clean the beta table before styling by replacing inf, -inf with NaN
        # and clipping any extremely high/low values
        beta_table_clean = beta_table.replace([np.inf, -np.inf], np.nan)
        
        # Optional: Clip extreme values
        max_beta_display = 10  # Limit to +/- 10 for display purposes
        beta_table_clean = beta_table_clean.clip(lower=-max_beta_display, upper=max_beta_display)
        
        # Function to color cells based on beta value
        def color_beta_cells(val):
            if pd.isna(val):
                return 'background-color: #f5f5f5; color: #666666;'  # Grey for missing
            elif val < 0:  # Negative beta (moves opposite to reference)
                return 'background-color: rgba(255, 0, 255, 0.7); color: white'  # Purple
            elif val < 0.5:  # Low beta
                return 'background-color: rgba(0, 0, 255, 0.7); color: white'  # Blue
            elif val < 0.9:  # Moderate beta
                return 'background-color: rgba(173, 216, 230, 0.7); color: black'  # Light blue
            elif val < 1.1:  # Similar to reference
                return 'background-color: rgba(255, 255, 255, 0.7); color: black'  # White/transparent
            elif val < 2.0:  # High beta
                return 'background-color: rgba(255, 165, 0, 0.7); color: black'  # Orange
            else:  # Very high beta
                return 'background-color: rgba(255, 0, 0, 0.7); color: white'  # Red
        
        # Try to display the styled table with error handling
        try:
            styled_beta_table = beta_table_clean.style.applymap(color_beta_cells).format(formatter=safe_format)
            st.markdown(f"## Beta Coefficient Table ({timeframe} intervals, Last {lookback_days} day(s), Singapore Time)")
            st.markdown(f"### Reference Token: {reference_token}")
            st.markdown("### Color Legend: <span style='color:purple'>Negative Beta</span>, <span style='color:blue'>Low Beta</span>, <span style='color:lightblue'>Moderate Beta</span>, <span style='color:black'>Similar to Reference</span>, <span style='color:orange'>High Beta</span>, <span style='color:red'>Very High Beta</span>", unsafe_allow_html=True)
            st.markdown(f"Values shown as Beta coefficient (how much token moves per 1% move in {reference_token})")
            st.dataframe(styled_beta_table, height=700, use_container_width=True)
        except Exception as e:
            st.error(f"Error displaying styled beta table: {str(e)}")
            st.write("Displaying unstyled beta table instead:")
            st.dataframe(beta_table_clean, height=700, use_container_width=True)
        
        # Create ranking table based on overall beta
        st.subheader(f"Beta Coefficient Ranking ({lookback_days}-Day, Descending Order)")
        
        beta_ranking_data = []
        for token, values in beta_values.items():
            beta = values['beta']
            r_squared = values['r_squared']
            alpha = values['alpha']
            correlation = values['correlation']
            
            # Skip if beta calculation failed
            if beta is None:
                continue
                
            # Get statistics for this token if available
            if token in beta_table_data:
                rolling_betas = beta_table_data[token].dropna()
                
                # Filter out extreme values for more reliable statistics
                filtered_betas = rolling_betas.replace([np.inf, -np.inf], np.nan).dropna()
                if not filtered_betas.empty:
                    filtered_betas = filtered_betas.clip(lower=-5, upper=5)
                    
                    max_beta = filtered_betas.max() if not filtered_betas.empty else np.nan
                    min_beta = filtered_betas.min() if not filtered_betas.empty else np.nan
                    beta_range = max_beta - min_beta if not np.isnan(max_beta) and not np.isnan(min_beta) else np.nan
                    beta_std = filtered_betas.std() if not filtered_betas.empty else np.nan
                    
                    # Calculate how often beta is within reasonable range
                    reliable_betas = ((filtered_betas >= 0.5) & (filtered_betas <= 2.0)).mean() * 100
                else:
                    max_beta = min_beta = beta_range = beta_std = reliable_betas = np.nan
            else:
                max_beta = min_beta = beta_range = beta_std = reliable_betas = np.nan
            
            beta_ranking_data.append({
                'Token': token,
                'Beta': round(beta, 4),
                'Alpha (%)': round(alpha, 4) if alpha is not None else np.nan,
                'R²': round(r_squared, 4) if r_squared is not None else np.nan,
                'Correlation': round(correlation, 4),
                'Max Beta': round(max_beta, 4),
                'Min Beta': round(min_beta, 4),
                'Beta Range': round(beta_range, 4),
                'Beta Std Dev': round(beta_std, 4),
                'Stable Beta %': round(reliable_betas, 1) if not np.isnan(reliable_betas) else np.nan,
                'Response Category': get_beta_category(beta)
            })
        
        if beta_ranking_data:
            beta_ranking_df = pd.DataFrame(beta_ranking_data)
            # Sort by beta (high to low)
            beta_ranking_df = beta_ranking_df.sort_values(by='Beta', ascending=False)
            # Add rank column
            beta_ranking_df.insert(0, 'Rank', range(1, len(beta_ranking_df) + 1))
            
            # Reset the index to remove it
            beta_ranking_df = beta_ranking_df.reset_index(drop=True)
            
            # Display the styled dataframe
            st.dataframe(beta_ranking_df, height=500, use_container_width=True)
            
            # Create visualization
            st.subheader("Beta vs. Correlation Analysis")
            
            # Scatter plot of beta vs correlation
            fig = px.scatter(
                beta_ranking_df,
                x='Beta',
                y='Correlation',
                title=f'Beta vs. Correlation with {reference_token}',
                labels={
                    'Beta': 'Beta Coefficient',
                    'Correlation': f'Correlation with {reference_token}'
                },
                color='R²',
                size='Stable Beta %',
                hover_name='Token',
                color_continuous_scale='Viridis'
            )
            
            # Add a vertical line at x=1 (same as reference)
            fig.add_vline(x=1, line_dash="dash", line_color="gray", annotation_text="Same as Reference")
            
            # Add a horizontal line at y=0 (no correlation)
            fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="No Correlation")
            
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Create bar chart of top betas
            fig2 = px.bar(
                beta_ranking_df.head(15),
                x='Token',
                y='Beta',
                title='Top 15 Tokens by Beta Coefficient',
                color='Beta',
                color_continuous_scale='Reds'
            )
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
            
            # Alpha Analysis
            st.subheader("Alpha Analysis - Tokens that Outperform Reference")
            
            # Sort by Alpha
            alpha_df = beta_ranking_df.sort_values(by='Alpha (%)', ascending=False)
            
            # Create bar chart of top alphas
            fig3 = px.bar(
                alpha_df.head(15),
                x='Token',
                y='Alpha (%)',
                title=f'Top 15 Tokens by Alpha (Outperformance vs {reference_token})',
                color='Alpha (%)',
                color_continuous_scale='Greens'
            )
            fig3.update_layout(height=400)
            st.plotly_chart(fig3, use_container_width=True)
            
            # Add a combined analysis with both beta and ATR ratio
            if 'ratio_table' in locals() and 'ranking_df' in locals():
                st.subheader("Combined ATR Ratio and Beta Analysis")
                
                # Merge the data
                combined_df = pd.merge(
                    ranking_df[['Token', 'Avg ATR Ratio', 'Volatility Category']], 
                    beta_ranking_df[['Token', 'Beta', 'Correlation', 'Response Category']], 
                    on='Token',
                    how='inner'
                )
                
                if not combined_df.empty:
                    # Create a scatter plot
                    fig4 = px.scatter(
                        combined_df,
                        x='Beta',
                        y='Avg ATR Ratio',
                        title=f'Beta vs ATR Ratio Analysis (Relative to {reference_token})',
                        color='Correlation',
                        size='Avg ATR Ratio',
                        hover_name='Token',
                        text='Token',
                        color_continuous_scale='Viridis'
                    )
                    
                    # Add reference lines
                    fig4.add_vline(x=1, line_dash="dash", line_color="gray")
                    fig4.add_hline(y=1, line_dash="dash", line_color="gray")
                    
                    # Add annotations for quadrants
                    fig4.add_annotation(x=0.5, y=1.5, text="High Volatility, Low Beta",
                                       showarrow=False, font=dict(size=12))
                    fig4.add_annotation(x=1.5, y=1.5, text="High Volatility, High Beta",
                                       showarrow=False, font=dict(size=12))
                    fig4.add_annotation(x=0.5, y=0.5, text="Low Volatility, Low Beta",
                                       showarrow=False, font=dict(size=12))
                    fig4.add_annotation(x=1.5, y=0.5, text="Low Volatility, High Beta",
                                       showarrow=False, font=dict(size=12))
                    
                    fig4.update_traces(textposition='top center')
                    fig4.update_layout(height=700)
                    st.plotly_chart(fig4, use_container_width=True)
                    
                    with st.expander("Understanding the Combined Analysis"):
                        st.markdown(f"""
                        ### Beta vs ATR Ratio Quadrant Analysis
                        
                        This scatter plot combines both beta and ATR ratio metrics to create a more complete picture of how tokens respond to {reference_token}.
                        
                        **The four quadrants represent:**
                        
                        1. **Top Right (High Beta, High ATR Ratio)**: Tokens that are both highly volatile and strongly correlated with {reference_token}. These tokens tend to amplify {reference_token}'s moves in both directions.
                        
                        2. **Top Left (Low Beta, High ATR Ratio)**: Tokens that are highly volatile but don't necessarily move in tandem with {reference_token}. They may have high volatility for token-specific reasons.
                        
                        3. **Bottom Right (High Beta, Low ATR Ratio)**: Tokens that closely follow {reference_token}'s moves but with less overall volatility. These may be more established tokens with strong BTC correlation.
                        
                        4. **Bottom Left (Low Beta, Low ATR Ratio)**: Tokens with both low volatility and weak correlation to {reference_token}. These tend to be more independent from market-wide moves.
                        
                        **Trading applications:**
                        
                        - For leveraged breakout trading after a {reference_token} move, focus on tokens in the **Top Right** quadrant
                        - For more conservative plays with good correlation, look at the **Bottom Right** quadrant
                        - Tokens in the **Top Left** may present opportunities for token-specific catalysts regardless of {reference_token}'s movement
                        """)
            
            # Beta explainer
            with st.expander("Understanding Beta Analysis"):
                st.markdown(f"""
                ### How to Read the Beta Coefficient Matrix
                
                This matrix shows how much each token moves relative to a 1% move in {reference_token}, taking into account the correlation between the token and {reference_token}.
                
                **The Beta Coefficient is calculated using covariance and variance:**
                ```
                Beta = Covariance(Token, {reference_token}) / Variance({reference_token})
                ```
                
                **What the beta values mean:**
                - **Beta = 1.0**: The token moves exactly the same as {reference_token} (1% when it moves 1%)
                - **Beta = 2.0**: The token moves twice as much as {reference_token} (2% when it moves 1%)
                - **Beta = 0.5**: The token moves half as much as {reference_token} (0.5% when it moves 1%)
                - **Beta < 0**: The token moves in the opposite direction to {reference_token}
                
                **Other important metrics:**
                - **Alpha**: The token's excess return over what would be predicted by {reference_token}'s movements alone
                - **R²**: How well {reference_token}'s movements explain the token's movements (higher means stronger relationship)
                - **Correlation**: Linear correlation between token and {reference_token} returns
                - **Stable Beta %**: Percentage of time the token's beta is within a reasonable range (0.5 to 2.0)
                
                ### Trading Applications
                
                **For Breakout Trading:**
                1. Look for tokens with high Beta (>1.5) AND high Correlation (>0.7) for amplified moves in the same direction
                2. Tokens with high Beta but lower Correlation may move strongly but less predictably
                3. Tokens with high positive Alpha tend to outperform {reference_token} over time
                
                **Risk Management:**
                - Higher Beta tokens will experience larger drawdowns when {reference_token} falls
                - Beta Range shows how consistent the relationship is (lower means more consistent)
                - A high Stable Beta % indicates a more reliable relationship with {reference_token}
                """)
        else:
            st.warning("No beta data available.")
    else:
        st.error(f"Reference token data ({reference_token}) is required for beta calculations. Please ensure it is selected and data is available.")

# Add disclaimer
st.markdown("---")
st.caption("Disclaimer: This analysis is for informational purposes only and should not be considered as financial advice.")