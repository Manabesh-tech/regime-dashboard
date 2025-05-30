# Save this as pages/05_Daily_Volatility_Table.py in your Streamlit app folder

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import pytz

st.set_page_config(
    page_title="Daily Volatility Table",
    page_icon="📈",
    layout="wide"
)

# --- DB CONFIG ---
try:
    db_config = st.secrets["database"]
    db_uri = (
        f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}"
        f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    )
    engine = create_engine(db_uri,    isolation_level="AUTOCOMMIT")
except Exception as e:
    st.error(f"Error connecting to the database: {e}")
    st.stop()

# --- UI Setup ---
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Daily Volatility Table (30min)")
st.subheader("All Trading Pairs - Last 24 Hours (Singapore Time)")

# Define parameters for the 30-minute timeframe
timeframe = "30min"
lookback_days = 1  # 24 hours
rolling_window = 20  # Window size for volatility calculation
expected_points = 48  # Expected data points per pair over 24 hours
singapore_timezone = pytz.timezone('Asia/Singapore')

# Get current time in Singapore timezone
now_utc = datetime.now(pytz.utc)
now_sg = now_utc.astimezone(singapore_timezone)
st.write(f"Current Singapore Time: {now_sg.strftime('%Y-%m-%d %H:%M:%S')}")

# Set extreme volatility threshold
extreme_vol_threshold = 1.0  # 100% annualized volatility

# Fetch all available tokens from DB
@st.cache_data(show_spinner="Fetching tokens...")
def fetch_all_tokens():
    query = "SELECT DISTINCT pair_name FROM public.oracle_price_log ORDER BY pair_name"
    try:
        df = pd.read_sql(query, engine)
        if df.empty:
            st.error("No tokens found in the database.")
            return []
        return df['pair_name'].tolist()
    except Exception as e:
        st.error(f"Error fetching tokens: {e}")
        return ["BTC", "ETH", "SOL", "DOGE", "PEPE", "AI16Z"]  # Default fallback

all_tokens = fetch_all_tokens()

# UI Controls
col1, col2 = st.columns([3, 1])

with col1:
    # Let user select tokens to display (or select all)
    select_all = st.checkbox("Select All Tokens", value=True)
    
    if select_all:
        selected_tokens = all_tokens
    else:
        selected_tokens = st.multiselect(
            "Select Tokens", 
            all_tokens,
            default=all_tokens[:5] if len(all_tokens) > 5 else all_tokens
        )

with col2:
    # Add a refresh button
    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.experimental_rerun()

if not selected_tokens:
    st.warning("Please select at least one token")
    st.stop()

# Function to calculate various volatility metrics
def calculate_volatility_metrics(price_series):
    if price_series is None or len(price_series) < 2:
        return {
            'realized_vol': np.nan,
            'parkinson_vol': np.nan,
            'gk_vol': np.nan,
            'rs_vol': np.nan
        }
    
    try:
        # Calculate log returns
        log_returns = np.diff(np.log(price_series))
        
        # 1. Standard deviation of returns (realized volatility)
        realized_vol = np.std(log_returns) * np.sqrt(252 * 48)  # Annualized volatility (30min bars)
        
        # For other volatility metrics, need OHLC data
        # For simplicity, we'll focus on realized volatility for now
        # But the structure allows adding more volatility metrics
        
        return {
            'realized_vol': realized_vol,
            'parkinson_vol': np.nan,  # Placeholder for Parkinson volatility
            'gk_vol': np.nan,         # Placeholder for Garman-Klass volatility
            'rs_vol': np.nan          # Placeholder for Rogers-Satchell volatility
        }
    except Exception as e:
        print(f"Error in volatility calculation: {e}")
        return {
            'realized_vol': np.nan,
            'parkinson_vol': np.nan,
            'gk_vol': np.nan,
            'rs_vol': np.nan
        }

# Volatility classification function
def classify_volatility(vol):
    if pd.isna(vol):
        return ("UNKNOWN", 0, "Insufficient data")
    elif vol < 0.30:  # 30% annualized volatility threshold for low volatility
        return ("LOW", 1, "Low volatility")
    elif vol < 0.60:  # 60% annualized volatility threshold for medium volatility
        return ("MEDIUM", 2, "Medium volatility")
    elif vol < 1.00:  # 100% annualized volatility threshold for high volatility
        return ("HIGH", 3, "High volatility")
    else:
        return ("EXTREME", 4, "Extreme volatility")

# Function to generate aligned 30-minute time blocks for the past 24 hours
def generate_aligned_time_blocks(current_time):
    """
    Generate fixed 30-minute time blocks for past 24 hours,
    aligned with standard 30-minute intervals (e.g., 4:00-4:30, 4:30-5:00)
    """
    # Round down to the nearest 30-minute mark
    if current_time.minute < 30:
        # Round down to XX:00
        latest_complete_block_end = current_time.replace(minute=0, second=0, microsecond=0)
    else:
        # Round down to XX:30
        latest_complete_block_end = current_time.replace(minute=30, second=0, microsecond=0)
    
    # Generate block labels for display
    blocks = []
    for i in range(48):  # 24 hours of 30-minute blocks
        block_end = latest_complete_block_end - timedelta(minutes=i*30)
        block_start = block_end - timedelta(minutes=30)
        block_label = f"{block_start.strftime('%H:%M')}"
        blocks.append((block_start, block_end, block_label))
    
    return blocks

# Generate aligned time blocks
aligned_time_blocks = generate_aligned_time_blocks(now_sg)
time_block_labels = [block[2] for block in aligned_time_blocks]

# Fetch and calculate volatility for a token with 30min timeframe
@st.cache_data(ttl=600, show_spinner="Calculating volatility metrics...")
def fetch_and_calculate_volatility(token):
    # Get current time in Singapore timezone
    now_utc = datetime.now(pytz.utc)
    now_sg = now_utc.astimezone(singapore_timezone)
    start_time_sg = now_sg - timedelta(days=lookback_days)
    
    # Convert back to UTC for database query
    start_time_utc = start_time_sg.astimezone(pytz.utc)
    end_time_utc = now_sg.astimezone(pytz.utc)

    query = f"""
    SELECT 
        created_at AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore' AS timestamp, 
        final_price, 
        pair_name
    FROM public.oracle_price_log
    WHERE created_at BETWEEN '{start_time_utc}' AND '{end_time_utc}'
    AND pair_name = '{token}';
    """
    try:
        print(f"[{token}] Executing query: {query}")
        df = pd.read_sql(query, engine)
        print(f"[{token}] Query executed. DataFrame shape: {df.shape}")

        if df.empty:
            print(f"[{token}] No data found.")
            return None

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        
        # Create 1-minute OHLC data
        one_min_ohlc = df['final_price'].resample('1min').ohlc().dropna()
        if one_min_ohlc.empty:
            print(f"[{token}] No OHLC data after resampling.")
            return None
            
        # Calculate rolling volatility on 1-minute data
        one_min_ohlc['realized_vol'] = one_min_ohlc['close'].rolling(window=rolling_window).apply(
            lambda x: calculate_volatility_metrics(x)['realized_vol']
        )
        
        # Resample to exactly 30min intervals aligned with clock
        thirty_min_vol = one_min_ohlc['realized_vol'].resample('30min', closed='left', label='left').mean().dropna()
        
        if thirty_min_vol.empty:
            print(f"[{token}] No 30-min volatility data.")
            return None
            
        # Get last 24 hours (48 30-minute bars)
        last_24h_vol = thirty_min_vol.tail(48)  # Get up to last 48 periods (24 hours)
        last_24h_vol = last_24h_vol.to_frame()
        
        # Store original datetime index for reference
        last_24h_vol['original_datetime'] = last_24h_vol.index
        
        # Format time label to match our aligned blocks (HH:MM format)
        last_24h_vol['time_label'] = last_24h_vol.index.strftime('%H:%M')
        
        # Calculate 24-hour average volatility
        last_24h_vol['avg_24h_vol'] = last_24h_vol['realized_vol'].mean()
        
        # Classify volatility
        last_24h_vol['vol_info'] = last_24h_vol['realized_vol'].apply(classify_volatility)
        last_24h_vol['vol_regime'] = last_24h_vol['vol_info'].apply(lambda x: x[0])
        last_24h_vol['vol_desc'] = last_24h_vol['vol_info'].apply(lambda x: x[2])
        
        # Also classify the 24-hour average
        last_24h_vol['avg_vol_info'] = last_24h_vol['avg_24h_vol'].apply(classify_volatility)
        last_24h_vol['avg_vol_regime'] = last_24h_vol['avg_vol_info'].apply(lambda x: x[0])
        last_24h_vol['avg_vol_desc'] = last_24h_vol['avg_vol_info'].apply(lambda x: x[2])
        
        # Flag extreme volatility events
        last_24h_vol['is_extreme'] = last_24h_vol['realized_vol'] >= extreme_vol_threshold
        
        print(f"[{token}] Successful Volatility Calculation")
        return last_24h_vol
    except Exception as e:
        st.error(f"Error processing {token}: {e}")
        print(f"[{token}] Error processing: {e}")
        return None

# Show the blocks we're analyzing
with st.expander("View Time Blocks Being Analyzed"):
    time_blocks_df = pd.DataFrame([(b[0].strftime('%Y-%m-%d %H:%M'), b[1].strftime('%Y-%m-%d %H:%M'), b[2]) 
                                  for b in aligned_time_blocks], 
                                 columns=['Start Time', 'End Time', 'Block Label'])
    st.dataframe(time_blocks_df)

# Show progress bar while calculating
progress_bar = st.progress(0)
status_text = st.empty()

# Calculate volatility for each token
token_results = {}
for i, token in enumerate(selected_tokens):
    try:
        progress_bar.progress((i) / len(selected_tokens))
        status_text.text(f"Processing {token} ({i+1}/{len(selected_tokens)})")
        result = fetch_and_calculate_volatility(token)
        if result is not None:
            token_results[token] = result
    except Exception as e:
        st.error(f"Error processing token {token}: {e}")
        print(f"Error processing token {token} in main loop: {e}")

# Final progress update
progress_bar.progress(1.0)
status_text.text(f"Processed {len(token_results)}/{len(selected_tokens)} tokens successfully")

# Create table for display
if token_results:
    # Create table data
    table_data = {}
    for token, df in token_results.items():
        vol_series = df.set_index('time_label')['realized_vol']
        table_data[token] = vol_series
    
    # Create DataFrame with all tokens
    vol_table = pd.DataFrame(table_data)
    
    # Apply the time blocks in the proper order (most recent first)
    available_times = set(vol_table.index)
    ordered_times = [t for t in time_block_labels if t in available_times]
    
    # If no matches are found in aligned blocks, fallback to the available times
    if not ordered_times and available_times:
        ordered_times = sorted(list(available_times), reverse=True)
    
    # Reindex with the ordered times
    vol_table = vol_table.reindex(ordered_times)
    
    # Convert from decimal to percentage and round to 1 decimal place
    vol_table = (vol_table * 100).round(1)
    
    def color_cells(val):
        if pd.isna(val):
            return 'background-color: #f5f5f5; color: #666666;'  # Grey for missing
        elif val < 30:  # Low volatility - green
            intensity = max(0, min(255, int(255 * val / 30)))
            return f'background-color: rgba({255-intensity}, 255, {255-intensity}, 0.7); color: black'
        elif val < 60:  # Medium volatility - yellow
            intensity = max(0, min(255, int(255 * (val - 30) / 30)))
            return f'background-color: rgba(255, 255, {255-intensity}, 0.7); color: black'
        elif val < 100:  # High volatility - orange
            intensity = max(0, min(255, int(255 * (val - 60) / 40)))
            return f'background-color: rgba(255, {255-(intensity//2)}, 0, 0.7); color: black'
        else:  # Extreme volatility - red
            return 'background-color: rgba(255, 0, 0, 0.7); color: white'
    
    styled_table = vol_table.style.applymap(color_cells)
    st.markdown("## Volatility Table (30min timeframe, Last 24 hours, Singapore Time)")
    st.markdown("### Color Legend: <span style='color:green'>Low Vol</span>, <span style='color:#aaaa00'>Medium Vol</span>, <span style='color:orange'>High Vol</span>, <span style='color:red'>Extreme Vol</span>", unsafe_allow_html=True)
    st.markdown("Values shown as annualized volatility percentage")
    st.dataframe(styled_table, height=700, use_container_width=True)
    
    # Create ranking table based on average volatility
    st.subheader("Volatility Ranking (24-Hour Average, Descending Order)")
    
    ranking_data = []
    for token, df in token_results.items():
        if not df.empty and 'avg_24h_vol' in df.columns and not df['avg_24h_vol'].isna().all():
            avg_vol = df['avg_24h_vol'].iloc[0]  # All rows have the same avg value
            vol_regime = df['avg_vol_desc'].iloc[0]
            max_vol = df['realized_vol'].max()
            min_vol = df['realized_vol'].min()
            ranking_data.append({
                'Token': token,
                'Avg Vol (%)': (avg_vol * 100).round(1),
                'Regime': vol_regime,
                'Max Vol (%)': (max_vol * 100).round(1),
                'Min Vol (%)': (min_vol * 100).round(1),
                'Vol Range (%)': ((max_vol - min_vol) * 100).round(1)
            })
    
    if ranking_data:
        ranking_df = pd.DataFrame(ranking_data)
        # Sort by average volatility (high to low)
        ranking_df = ranking_df.sort_values(by='Avg Vol (%)', ascending=False)
        # Add rank column
        ranking_df.insert(0, 'Rank', range(1, len(ranking_df) + 1))
        
        # Reset the index to remove it
        ranking_df = ranking_df.reset_index(drop=True)
        
        # Format ranking table with colors
        def color_regime(val):
            if 'Low' in val:
                return 'color: green'
            elif 'Medium' in val:
                return 'color: #aaaa00'
            elif 'High' in val:
                return 'color: orange'
            elif 'Extreme' in val:
                return 'color: red'
            return ''
        
        def color_value(val):
            if pd.isna(val):
                return ''
            elif val < 30:
                return 'color: green'
            elif val < 60:
                return 'color: #aaaa00'
            elif val < 100:
                return 'color: orange'
            else:
                return 'color: red'
        
        # Apply styling
        styled_ranking = ranking_df.style\
            .applymap(color_regime, subset=['Regime'])\
            .applymap(color_value, subset=['Avg Vol (%)', 'Max Vol (%)', 'Min Vol (%)'])
        
        # Display the styled dataframe
        st.dataframe(styled_ranking, height=500, use_container_width=True)
    else:
        st.warning("No ranking data available.")
    
    # Identify and display extreme volatility events
    st.subheader("Extreme Volatility Events (>= 100% Annualized)")
    
    extreme_events = []
    for token, df in token_results.items():
        if not df.empty and 'is_extreme' in df.columns:
            extreme_periods = df[df['is_extreme']]
            for idx, row in extreme_periods.iterrows():
                # Safely access values with explicit casting to avoid attribute errors
                vol_value = float(row['realized_vol']) if not pd.isna(row['realized_vol']) else 0.0
                time_label = str(row['time_label']) if 'time_label' in row and not pd.isna(row['time_label']) else "Unknown"
                
                extreme_events.append({
                    'Token': token,
                    'Time': time_label,
                    'Volatility (%)': round(vol_value * 100, 1),
                    'Full Timestamp': idx.strftime('%Y-%m-%d %H:%M')
                })
    
    if extreme_events:
        extreme_df = pd.DataFrame(extreme_events)
        # Sort by volatility (highest first)
        extreme_df = extreme_df.sort_values(by='Volatility (%)', ascending=False)
        
        # Reset the index to remove it
        extreme_df = extreme_df.reset_index(drop=True)
        
        # Display the dataframe
        st.dataframe(extreme_df, height=300, use_container_width=True)
        
        # Create a more visually appealing list of extreme events
        st.markdown("### Extreme Volatility Events Detail")
        
        # Only process top 10 events if there are any
        top_events = extreme_events[:min(10, len(extreme_events))]
        for i, event in enumerate(top_events):
            token = event['Token']
            time = event['Time']
            vol = event['Volatility (%)']
            date = event['Full Timestamp'].split(' ')[0]
            
            st.markdown(f"**{i+1}. {token}** at **{time}** on {date}: <span style='color:red; font-weight:bold;'>{vol}%</span> volatility", unsafe_allow_html=True)
        
        if len(extreme_events) > 10:
            st.markdown(f"*... and {len(extreme_events) - 10} more extreme events*")
        
    else:
        st.info("No extreme volatility events detected in the selected tokens.")
    
    # 24-Hour Average Volatility Distribution
    st.subheader("24-Hour Average Volatility Overview (Singapore Time)")
    avg_values = {}
    for token, df in token_results.items():
        if not df.empty and 'avg_24h_vol' in df.columns and not df['avg_24h_vol'].isna().all():
            avg = df['avg_24h_vol'].iloc[0]  # All rows have the same avg value
            regime = df['avg_vol_desc'].iloc[0]
            avg_values[token] = (avg, regime)
    
    if avg_values:
        low_vol = sum(1 for v, r in avg_values.values() if v < 0.3)
        medium_vol = sum(1 for v, r in avg_values.values() if 0.3 <= v < 0.6)
        high_vol = sum(1 for v, r in avg_values.values() if 0.6 <= v < 1.0)
        extreme_vol = sum(1 for v, r in avg_values.values() if v >= 1.0)
        total = low_vol + medium_vol + high_vol + extreme_vol
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Low Vol", f"{low_vol} ({low_vol/total*100:.1f}%)")
        col2.metric("Medium Vol", f"{medium_vol} ({medium_vol/total*100:.1f}%)")
        col3.metric("High Vol", f"{high_vol} ({high_vol/total*100:.1f}%)")
        col4.metric("Extreme Vol", f"{extreme_vol} ({extreme_vol/total*100:.1f}%)")
        
        labels = ['Low Vol', 'Medium Vol', 'High Vol', 'Extreme Vol']
        values = [low_vol, medium_vol, high_vol, extreme_vol]
        colors = ['rgba(100,255,100,0.8)', 'rgba(255,255,100,0.8)', 'rgba(255,165,0,0.8)', 'rgba(255,0,0,0.8)']
        
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, marker=dict(colors=colors, line=dict(color='#000000', width=2)), textinfo='label+percent', hole=.3)])
        fig.update_layout(
            title="24-Hour Average Volatility Distribution (Singapore Time)",
            height=400,
            font=dict(color="#000000", size=12),
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Create columns for each volatility category
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)
        
        with col1:
            st.markdown("### Low Average Volatility Tokens")
            lv_tokens = [(t, v*100, r) for t, (v, r) in avg_values.items() if v < 0.3]
            lv_tokens.sort(key=lambda x: x[1])
            if lv_tokens:
                for token, value, regime in lv_tokens:
                    st.markdown(f"- **{token}**: <span style='color:green'>{value:.1f}%</span> ({regime})", unsafe_allow_html=True)
            else:
                st.markdown("*No tokens in this category*")
        
        with col2:
            st.markdown("### Medium Average Volatility Tokens")
            mv_tokens = [(t, v*100, r) for t, (v, r) in avg_values.items() if 0.3 <= v < 0.6]
            mv_tokens.sort(key=lambda x: x[1])
            if mv_tokens:
                for token, value, regime in mv_tokens:
                    st.markdown(f"- **{token}**: <span style='color:#aaaa00'>{value:.1f}%</span> ({regime})", unsafe_allow_html=True)
            else:
                st.markdown("*No tokens in this category*")
        
        with col3:
            st.markdown("### High Average Volatility Tokens")
            hv_tokens = [(t, v*100, r) for t, (v, r) in avg_values.items() if 0.6 <= v < 1.0]
            hv_tokens.sort(key=lambda x: x[1])
            if hv_tokens:
                for token, value, regime in hv_tokens:
                    st.markdown(f"- **{token}**: <span style='color:orange'>{value:.1f}%</span> ({regime})", unsafe_allow_html=True)
            else:
                st.markdown("*No tokens in this category*")
        
        with col4:
            st.markdown("### Extreme Average Volatility Tokens")
            ev_tokens = [(t, v*100, r) for t, (v, r) in avg_values.items() if v >= 1.0]
            ev_tokens.sort(key=lambda x: x[1], reverse=True)
            if ev_tokens:
                for token, value, regime in ev_tokens:
                    st.markdown(f"- **{token}**: <span style='color:red'>{value:.1f}%</span> ({regime})", unsafe_allow_html=True)
            else:
                st.markdown("*No tokens in this category*")
    else:
        st.warning("No average volatility data available for the selected tokens.")

with st.expander("Understanding the Volatility Table"):
    st.markdown("""
    ### How to Read This Table
    This table shows annualized volatility values for all selected tokens over the last 24 hours using 30-minute bars.
    Each row represents a specific 30-minute time period, with times shown in Singapore time. The table is sorted with the most recent 30-minute period at the top.
    
    **Color coding:**
    - **Green** (< 30%): Low volatility
    - **Yellow** (30-60%): Medium volatility
    - **Orange** (60-100%): High volatility
    - **Red** (> 100%): Extreme volatility
    
    **The intensity of the color indicates the strength of the volatility:**
    - Darker green = Lower volatility
    - Darker red = Higher volatility
    
    **Ranking Table:**
    The ranking table sorts tokens by their 24-hour average volatility from highest to lowest.
    
    **Extreme Volatility Events:**
    These are specific 30-minute periods where a token's annualized volatility exceeded 100%.
    
    **Technical details:**
    - Volatility is calculated as the standard deviation of log returns, annualized to represent the expected price variation over a year
    - Values shown are in percentage (e.g., 50.0 means 50% annualized volatility)
    - The calculation uses a rolling window of 20 one-minute price points
    - The 24-hour average section shows the mean volatility across all 48 30-minute periods
    - Missing values (light gray cells) indicate insufficient data for calculation
    """)