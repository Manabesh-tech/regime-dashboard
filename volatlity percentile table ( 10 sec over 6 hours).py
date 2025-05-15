import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from sqlalchemy import create_engine

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

# Function to calculate volatility for a single token - sequential version
@st.cache_data(ttl=300)  # Cache for 5 minutes
def calculate_all_volatilities():
    all_tokens = fetch_trading_pairs()
    results = []
    
    sg_tz = pytz.timezone('Asia/Singapore')
    now_sg = datetime.now(sg_tz)
    start_time_sg = now_sg - timedelta(hours=6)
    
    # Get partitions
    today_str = now_sg.strftime("%Y%m%d")
    yesterday_str = (now_sg - timedelta(days=1)).strftime("%Y%m%d")
    
    start_time_str = start_time_sg.strftime("%Y-%m-%d %H:%M:%S")
    end_time_str = now_sg.strftime("%Y-%m-%d %H:%M:%S")
    
    progress_container = st.container()
    progress_bar = progress_container.progress(0)
    status_text = progress_container.empty()
    
    for idx, token in enumerate(all_tokens):
        # Update progress
        progress = (idx + 1) / len(all_tokens)
        progress_bar.progress(progress)
        status_text.text(f"Processing {idx + 1}/{len(all_tokens)}: {token}")
        
        try:
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
                continue
            
            # Process timestamps
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').sort_index()
            
            # Resample to 500ms
            price_data = df['final_price'].resample('500ms').ffill().dropna()
            
            if len(price_data) < 2:
                continue
            
            # Calculate 10-second volatilities
            volatility_values = []
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
                        volatility_values.append(volatility)
            
            if not volatility_values:
                continue
            
            # Convert to percentage
            vol_pct = np.array(volatility_values) * 100
            
            # Calculate percentiles
            percentiles = {
                'pair': token,
                '50_pctile': np.percentile(vol_pct, 50),
                '75_pctile': np.percentile(vol_pct, 75),
                '95_pctile': np.percentile(vol_pct, 95)
            }
            
            results.append(percentiles)
            
        except Exception as e:
            st.warning(f"Error processing {token}: {str(e)}")
            continue
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    return results

# Main execution
st.write("Calculating volatility percentiles for all trading pairs...")

# Calculate all volatilities
results = calculate_all_volatilities()

# Create DataFrame
if results:
    df_results = pd.DataFrame(results)
    
    # Sort by pair name
    df_results = df_results.sort_values('pair')
    
    # Style the dataframe
    st.markdown("### Volatility Percentiles (Last 6 Hours)")
    st.markdown("*Based on 10-second window annualized volatility calculations*")
    
    # Create a styled dataframe with color coding
    def style_volatility(val):
        """Apply color based on volatility value"""
        try:
            # Extract numeric value
            num_val = float(val)
                
            if num_val > 100:
                color = '#ff4444'  # Red
            elif num_val > 50:
                color = '#ff8800'  # Orange
            else:
                color = '#44ff44'  # Green
            return f'background-color: {color}; color: black'
        except:
            return ''
    
    # First rename columns
    display_df = df_results.rename(columns={
        'pair': 'Pair',
        '50_pctile': '50th %ile',
        '75_pctile': '75th %ile',
        '95_pctile': '95th %ile'
    })
    
    # Then format and style
    styled_df = display_df.style.format({
        '50th %ile': '{:.1f}%',
        '75th %ile': '{:.1f}%',
        '95th %ile': '{:.1f}%'
    })
    
    # Apply styling to percentile columns
    styled_df = styled_df.applymap(
        style_volatility,
        subset=['50th %ile', '75th %ile', '95th %ile']
    )
    
    # Display the styled dataframe
    st.dataframe(
        styled_df,
        use_container_width=True,
        height=600
    )
    
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
        st.metric("Average 95th %ile", f"{df_results['95_pctile'].mean():.1f}%")
        st.metric("Pairs with 95th > 150%", f"{(df_results['95_pctile'] > 150).sum()}")
    
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

# Refresh button
if st.button("Refresh Data", type="primary"):
    st.cache_data.clear()
    st.rerun()

# Display current time
sg_tz = pytz.timezone('Asia/Singapore')
st.write(f"Last updated: {datetime.now(sg_tz).strftime('%Y-%m-%d %H:%M:%S')} (Singapore Time)")