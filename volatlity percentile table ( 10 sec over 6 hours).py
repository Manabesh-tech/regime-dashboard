import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from sqlalchemy import create_engine
import concurrent.futures
from functools import partial

st.set_page_config(page_title="Volatility Percentile Table", page_icon="📊", layout="wide")

# --- UI Setup ---
st.title("Volatility Percentile Table - All Surf Pairs")

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

# Function to calculate volatility for a single token
def calculate_volatility_percentiles(token, engine):
    try:
        sg_tz = pytz.timezone('Asia/Singapore')
        now_sg = datetime.now(sg_tz)
        start_time_sg = now_sg - timedelta(hours=6)
        
        # Get partitions
        today_str = now_sg.strftime("%Y%m%d")
        yesterday_str = (now_sg - timedelta(days=1)).strftime("%Y%m%d")
        
        start_time_str = start_time_sg.strftime("%Y-%m-%d %H:%M:%S")
        end_time_str = now_sg.strftime("%Y-%m-%d %H:%M:%S")
        
        # Try today's partition first
        query = f"""
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
        
        df = pd.read_sql_query(query, engine)
        
        # If we don't have enough data, try yesterday's partition too
        if df.empty or len(df) < 10:
            query_yesterday = f"""
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
            try:
                df_yesterday = pd.read_sql_query(query_yesterday, engine)
                df = pd.concat([df_yesterday, df]).drop_duplicates().sort_values('timestamp')
            except:
                pass
        
        if df.empty:
            return None
        
        # Process timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        
        # Resample to 500ms
        price_data = df['final_price'].resample('500ms').ffill().dropna()
        
        if len(price_data) < 2:
            return None
        
        # Calculate 10-second volatilities
        result = []
        start_date = price_data.index.min().floor('10s')
        end_date = price_data.index.max().ceil('10s')
        ten_sec_periods = pd.date_range(start=start_date, end=end_date, freq='10s')
        
        for i in range(len(ten_sec_periods)-1):
            start_window = ten_sec_periods[i]
            end_window = ten_sec_periods[i+1]
            
            window_data = price_data[(price_data.index >= start_window) & (price_data.index < end_window)]
            
            if len(window_data) >= 2:
                # Calculate volatility for 10-second window
                log_returns = np.diff(np.log(window_data.values))
                if len(log_returns) > 0:
                    annualization_factor = np.sqrt(3153600)  # For 10-second windows
                    volatility = np.std(log_returns) * annualization_factor
                    result.append(volatility)
        
        if not result:
            return None
        
        # Convert to percentage
        vol_pct = np.array(result) * 100
        
        # Calculate percentiles
        percentiles = {
            'pair': token,
            '50_pctile': np.percentile(vol_pct, 50),
            '75_pctile': np.percentile(vol_pct, 75),
            '90_pctile': np.percentile(vol_pct, 90)
        }
        
        return percentiles
        
    except Exception as e:
        return None

# Main execution
with st.spinner("Fetching all trading pairs..."):
    all_tokens = fetch_trading_pairs()

st.write(f"Total pairs: {len(all_tokens)}")

# Calculate percentiles for all pairs with parallel processing
results = []

# Create multiple engines for parallel processing
engines = [create_engine(
    f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['database']}"
) for _ in range(10)]

def process_token(token, engine_idx):
    engine = engines[engine_idx % len(engines)]
    result = calculate_volatility_percentiles(token, engine)
    return result

# Process tokens in parallel
with st.spinner(f"Processing {len(all_tokens)} pairs..."):
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for idx, token in enumerate(all_tokens):
            future = executor.submit(process_token, token, idx)
            futures.append(future)
        
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

# Create DataFrame
if results:
    df_results = pd.DataFrame(results)
    
    # Sort by pair name
    df_results = df_results.sort_values('pair')
    
    # Style the dataframe
    st.markdown("### Volatility Percentiles (Last 6 Hours)")
    st.markdown("*Based on 10-second window annualized volatility calculations*")
    
    # Create a styled version
    def color_cells(val):
        if val > 100:
            color = '#FF6B6B'  # Red for high volatility
        elif val > 50:
            color = '#FFA500'  # Orange for medium
        else:
            color = '#4CAF50'  # Green for low
        return f'background-color: {color}; color: white'
    
    styled_df = df_results.style.format({
        '50_pctile': '{:.1f}%',
        '75_pctile': '{:.1f}%',
        '90_pctile': '{:.1f}%'
    }).rename(columns={
        'pair': 'Pair',
        '50_pctile': '50th %ile',
        '75_pctile': '75th %ile',
        '90_pctile': '90th %ile'
    }).applymap(color_cells, subset=['50_pctile', '75_pctile', '90_pctile'])
    
    st.dataframe(styled_df, use_container_width=True, height=600)
    
    # Summary statistics
    st.markdown("### Summary Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Average 50th %ile", f"{df_results['50_pctile'].mean():.1f}%")
        st.metric("Pairs with 50th > 50%", f"{(df_results['50_pctile'] > 50).sum()}")
    
    with col2:
        st.metric("Average 75th %ile", f"{df_results['75_pctile'].mean():.1f}%")
        st.metric("Pairs with 75th > 100%", f"{(df_results['75_pctile'] > 100).sum()}")
    
    with col3:
        st.metric("Average 90th %ile", f"{df_results['90_pctile'].mean():.1f}%")
        st.metric("Pairs with 90th > 150%", f"{(df_results['90_pctile'] > 150).sum()}")
    
    # Option to download
    csv = df_results.to_csv(index=False)
    sg_tz = pytz.timezone('Asia/Singapore')
    timestamp = datetime.now(sg_tz).strftime('%Y%m%d_%H%M%S')
    st.download_button(
        "Download CSV",
        csv,
        f"volatility_percentiles_{timestamp}.csv",
        "text/csv",
        key='download-csv'
    )
else:
    st.error("No data could be calculated for any pairs")

# Close engines
for engine in engines:
    engine.dispose()

# Refresh button
if st.button("Refresh Data", type="primary"):
    st.cache_data.clear()
    st.rerun()

# Display current time
sg_tz = pytz.timezone('Asia/Singapore')
st.write(f"Last updated: {datetime.now(sg_tz).strftime('%Y-%m-%d %H:%M:%S')} (Singapore Time)")