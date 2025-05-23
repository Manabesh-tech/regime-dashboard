import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import pytz

# Page configuration
st.set_page_config(page_title="Volatility Dashboard", page_icon="📊", layout="wide")

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

# Tab 1: Manual Buffer Rate Update
def manual_buffer_rate_tab():
    st.header("Manual Buffer Rate Update")
    
    # Input columns
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        init_vol = st.number_input("Initial Vol (%)", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
    
    with col2:
        init_buffer = st.number_input("Initial Buffer", min_value=0.0, max_value=10.0, value=4.5, step=0.1)
    
    with col3:
        coeff1 = st.number_input("Coefficient 1", min_value=0.0, max_value=10.0, value=2.0, step=0.5)
    
    with col4:
        coeff2 = st.number_input("Coefficient 2", min_value=0.0, max_value=10.0, value=3.0, step=0.5)
    
    with col5:
        min_buffer = st.number_input("Min Buffer", min_value=0.0, max_value=10.0, value=4.0, step=0.1)
        max_buffer = st.number_input("Max Buffer", min_value=0.0, max_value=10.0, value=9.0, step=0.1)
    
    # Calculation for different final volatilities
    final_vols = np.linspace(0, 100, 21)
    expected_buffers = []
    
    for final_vol in final_vols:
        # Calculation as per the given equation
        expected_buf = min(
            max(
                init_vol + coeff1 * (final_vol - init_vol) + coeff2 * (final_vol - init_vol)**2, 
                min_buffer
            ), 
            max_buffer
        )
        expected_buffers.append(expected_buf)
    
    # Create Plotly figure using Plotly Express
    fig = px.line(
        x=final_vols, 
        y=expected_buffers, 
        labels={
            'x': 'Final Volatility (%)', 
            'y': 'Expected Surf Buffer'
        },
        title='Expected Surf Buffer vs Final Volatility'
    )
    
    fig.update_traces(mode='lines+markers')
    
    fig.update_layout(
        height=600,
        width=1000
    )
    
    # Display the plot
    st.plotly_chart(fig, use_container_width=True)
    
    # Display calculation details
    st.write("### Calculation Details")
    df_details = pd.DataFrame({
        'Final Vol (%)': final_vols,
        'Expected Surf Buffer': expected_buffers
    })
    st.dataframe(df_details, use_container_width=True)

# Shared function for calculating volatilities
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

@st.cache_data(ttl=300)
def calculate_volatilities(hours=3):
    all_tokens = fetch_trading_pairs()
    results = []
    
    sg_tz = pytz.timezone('Asia/Singapore')
    now_sg = datetime.now(sg_tz)
    start_time_sg = now_sg - timedelta(hours=hours)
    
    # Get partitions
    today_str = now_sg.strftime("%Y%m%d")
    yesterday_str = (now_sg - timedelta(days=1)).strftime("%Y%m%d")
    
    start_time_str = start_time_sg.strftime("%Y-%m-%d %H:%M:%S")
    end_time_str = now_sg.strftime("%Y-%m-%d %H:%M:%S")
    
    progress_bar = st.progress(0)
    
    for idx, token in enumerate(all_tokens):
        # Update progress
        progress_bar.progress((idx + 1) / len(all_tokens))
        
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
            
            # If we don't have enough data, try yesterday's partition
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
                '25_pctile': int(np.percentile(vol_pct, 25)),
                '50_pctile': int(np.percentile(vol_pct, 50)),
                '75_pctile': int(np.percentile(vol_pct, 75)),
                '95_pctile': int(np.percentile(vol_pct, 95))
            }
            
            results.append(percentiles)
            
        except Exception as e:
            st.warning(f"Error processing {token}: {str(e)}")
            continue
    
    progress_bar.empty()
    return results

# Tab 2: Automatic Buffer Rate Update
def automatic_buffer_rate_tab():
    st.header("Volatility and Buffer Rate Update")
    
    # Calculate 3-hour volatilities
    results = calculate_volatilities(hours=3)
    
    if not results:
        st.error("Could not fetch volatility data")
        return
    
    # Create DataFrame
    df_results = pd.DataFrame(results)
    
    # Sort by 50th percentile to get ranks
    df_results['50_pctile_rank'] = df_results['50_pctile'].rank(method='dense')
    
    # Scale init buffer between 4 and 6 while maintaining ranks
    max_rank = df_results['50_pctile_rank'].max()
    df_results['init_buffer'] = 4 + (df_results['50_pctile_rank'] - 1) * (6 - 4) / (max_rank - 1)
    
    # Display results
    display_df = df_results.rename(columns={
        'pair': 'Pair',
        '50_pctile': '50th %ile Vol',
        '50_pctile_rank': 'Volatility Rank',
        'init_buffer': 'Init Buffer',
        '25_pctile': '25th %ile Vol',
        '75_pctile': '75th %ile Vol',
        '95_pctile': '95th %ile Vol'
    })
    
    styled_df = display_df.style.format({
        '50th %ile Vol': '{:d}%',
        '25th %ile Vol': '{:d}%',
        '75th %ile Vol': '{:d}%',
        '95th %ile Vol': '{:d}%',
        'Init Buffer': '{:.2f}'
    })
    
    st.dataframe(
        styled_df,
        use_container_width=True,
        height=600
    )
    
    # Scatter plot to show relationship
    fig = px.scatter(
        df_results, 
        x='50_pctile', 
        y='init_buffer', 
        hover_data=['pair'],
        labels={
            '50_pctile': '50th Percentile Volatility (%)', 
            'init_buffer': 'Initial Buffer'
        },
        title='Init Buffer vs 50th Percentile Volatility'
    )
    
    fig.update_layout(height=600)
    
    st.plotly_chart(fig, use_container_width=True)

# Main app
def main():
    # Tabs
    tab1, tab2 = st.tabs([
        "Manual Buffer Rate Update", 
        "Automatic Buffer Rate Update"
    ])
    
    with tab1:
        manual_buffer_rate_tab()
    
    with tab2:
        automatic_buffer_rate_tab()

# Run the app
if __name__ == "__main__":
    main()