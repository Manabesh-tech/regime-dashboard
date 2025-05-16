# Save this as optimized_10sec_and_20sec_volatility_with_uat.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import psycopg2
import pytz
from sqlalchemy import create_engine

st.set_page_config(page_title="Volatility Plot with Rollbit & UAT", page_icon="📈", layout="wide")

# --- UI Setup ---
st.title("Volatility Plot with Rollbit & UAT Buffer Comparison")

# Production DB connection
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

conn = psycopg2.connect(**db_params)

# UAT DB connection - DIFFERENT DATABASE
uat_db_params = {
    'host': 'aws-jp-tk-surf-pg-public.cluster-csteuf9lw8dv.ap-northeast-1.rds.amazonaws.com',  
    'port': 5432,
    'database': 'report_dev',  # Different database
    'user': 'public_rw',     # Different user
    'password': 'aTJ92^kl04hllk'  # Different password
}

try:
    uat_engine = create_engine(
        f"postgresql://{uat_db_params['user']}:{uat_db_params['password']}@{uat_db_params['host']}:{uat_db_params['port']}/{uat_db_params['database']}"
    )
    uat_connection_status = True
except Exception as e:
    st.error(f"Could not establish UAT database connection: {e}")
    uat_connection_status = False

# Cache token list for longer
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_trading_pairs():
    query = """
    SELECT pair_name 
    FROM trade_pool_pairs 
    WHERE status in (1,2)
    ORDER BY pair_name
    """
    df = pd.read_sql_query(query, uat_engine)
    return df['pair_name'].tolist()

all_tokens = fetch_trading_pairs()

col1, col2, col3 = st.columns([3, 1, 1])

with col1:
    default_token = "BTC/USDT" if "BTC/USDT" in all_tokens else all_tokens[0]
    selected_token = st.selectbox(
        "Select Token",
        all_tokens,
        index=all_tokens.index(default_token) if default_token in all_tokens else 0
    )

with col2:
    if st.button("Refresh Data", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

with col3:
    # Add option to select which volatility charts to display
    display_options = st.multiselect(
        "Display Charts",
        ["10-Second Volatility", "20-Second Volatility"],
        default=["10-Second Volatility", "20-Second Volatility"]
    )

sg_tz = pytz.timezone('Asia/Singapore')
now_sg = datetime.now(sg_tz)
st.write(f"Current time (Singapore): {now_sg.strftime('%Y-%m-%d %H:%M:%S')}")

# Optimized Rollbit fetch - resample to 10 seconds (PRODUCTION)
@st.cache_data(ttl=60)
def fetch_rollbit_parameters_10sec(token, hours=3):
    """Fetch Rollbit parameters with 10-second resolution"""
    try:
        # 移除 token 名称中的 'PROD'
        clean_token = token.replace('PROD', '')
        
        now_sg = datetime.now(pytz.timezone('Asia/Singapore'))
        start_time_sg = now_sg - timedelta(hours=hours)
        
        start_str = start_time_sg.strftime("%Y-%m-%d %H:%M:%S")
        end_str = now_sg.strftime("%Y-%m-%d %H:%M:%S")

        query = f"""
         SELECT 
            pair_name,
            bust_buffer AS buffer_rate,
            position_multiplier,
            created_at + INTERVAL '8 hour' AS timestamp
        FROM rollbit_pair_config 
        WHERE pair_name = '{clean_token}'
        AND created_at >= '{start_str}'::timestamp - INTERVAL '8 hour'
        AND created_at <= '{end_str}'::timestamp - INTERVAL '8 hour'
        ORDER BY created_at
        """

        df = pd.read_sql_query(query, engine)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').sort_index()
            # Resample to 10 seconds
            df = df.resample('10s').ffill()
        return df
    except Exception as e:
        st.error(f"Error fetching Rollbit parameters: {e}")
        return None

# NEW: Fetch UAT buffer rates from DIFFERENT DATABASE
@st.cache_data(ttl=60)
def fetch_uat_buffer_rates_10sec(token, hours=3):
    """Fetch UAT buffer rates with 10-second resolution from UAT DATABASE"""
    if not uat_connection_status:
        return pd.DataFrame()
    
    try:
        now_sg = datetime.now(pytz.timezone('Asia/Singapore'))
        start_time_sg = now_sg - timedelta(hours=hours)
        
        start_str = start_time_sg.strftime("%Y-%m-%d %H:%M:%S")
        end_str = now_sg.strftime("%Y-%m-%d %H:%M:%S")

        # Debug: Check what data we're getting
        query = f"""
        SELECT 
            pair_name,
            buffer_rate AS buffer_rate,
            created_at + INTERVAL '8 hour' AS timestamp
        FROM trade_pair_risk_history 
        WHERE pair_name = '{token}'
        AND created_at >= '{start_str}'::timestamp - INTERVAL '8 hour'
        AND created_at <= '{end_str}'::timestamp - INTERVAL '8 hour'
        ORDER BY created_at
        """

        df = pd.read_sql_query(query, uat_engine)
        
        # Debug info
        if df.empty:
            st.warning(f"No UAT data found for {token}")
            # Try to see what tokens are available
            check_query = f"""
            SELECT DISTINCT pair_name 
            FROM trade_pair_risk_history 
            WHERE created_at >= '{start_str}'::timestamp - INTERVAL '8 hour'
            LIMIT 10
            """
            available = pd.read_sql_query(check_query, uat_engine)
            st.info(f"Available tokens in UAT: {available['pair_name'].tolist()}")
        else:
            st.success(f"Found {len(df)} UAT records for {token}")
            
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').sort_index()
            # Resample to 10 seconds and forward fill missing values
            df = df.resample('10s').ffill()
            
            # Remove any potential NaN values
            df = df.dropna()
            
        return df
    except Exception as e:
        st.error(f"Error fetching UAT buffer rates: {e}")
        return pd.DataFrame()

# Function to get raw price data (used by both 10-sec and 20-sec volatility)
@st.cache_data(ttl=30)
def get_raw_price_data(token, hours=3):
    now_sg = datetime.now(pytz.timezone('Asia/Singapore'))
    start_time_sg = now_sg - timedelta(hours=hours)
    
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
    
    try:
        df = pd.read_sql_query(query, uat_engine)
        
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
    except Exception as e:
        st.error(f"Query error for {token}: {str(e)}")
        return None
    
    if df.empty:
        return None
    
    # Process timestamps
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()
    
    # Resample to 500ms
    price_data = df['final_price'].resample('500ms').ffill().dropna()
    
    return price_data

# 10-second volatility calculation
@st.cache_data(ttl=30)
def get_volatility_data_10sec(price_data):
    if price_data is None or len(price_data) < 2:
        return None, None
    
    # Create 10-second windows
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
                annualization_factor = np.sqrt(3153600)  # For 10-second windows (31,536,000 seconds in a year / 10)
                volatility = np.std(log_returns) * annualization_factor
                
                result.append({
                    'timestamp': start_window,
                    'realized_vol': volatility
                })
    
    if not result:
        return None, None
    
    result_df = pd.DataFrame(result).set_index('timestamp')
    
    # Calculate percentiles for the last 3 hours
    vol_pct = result_df['realized_vol'] * 100
    if len(vol_pct) > 0:
        percentiles = {
            'p25': np.percentile(vol_pct, 25),
            'p50': np.percentile(vol_pct, 50),
            'p75': np.percentile(vol_pct, 75),
            'p95': np.percentile(vol_pct, 95)
        }
    else:
        percentiles = {'p25': 0, 'p50': 0, 'p75': 0, 'p95': 0}
    
    return result_df, percentiles

# NEW: 20-second volatility calculation
@st.cache_data(ttl=30)
def get_volatility_data_20sec(price_data):
    if price_data is None or len(price_data) < 2:
        return None, None
    
    # Create 20-second windows
    result = []
    start_date = price_data.index.min().floor('20s')
    end_date = price_data.index.max().ceil('20s')
    twenty_sec_periods = pd.date_range(start=start_date, end=end_date, freq='20s')
    
    for i in range(len(twenty_sec_periods)-1):
        start_window = twenty_sec_periods[i]
        end_window = twenty_sec_periods[i+1]
        
        window_data = price_data[(price_data.index >= start_window) & (price_data.index < end_window)]
        
        if len(window_data) >= 2:
            # Calculate volatility for 20-second window
            log_returns = np.diff(np.log(window_data.values))
            if len(log_returns) > 0:
                annualization_factor = np.sqrt(1576800)  # For 20-second windows (31,536,000 seconds in a year / 20)
                volatility = np.std(log_returns) * annualization_factor
                
                result.append({
                    'timestamp': start_window,
                    'realized_vol': volatility
                })
    
    if not result:
        return None, None
    
    result_df = pd.DataFrame(result).set_index('timestamp')
    
    # Calculate percentiles for the last 3 hours
    vol_pct = result_df['realized_vol'] * 100
    if len(vol_pct) > 0:
        percentiles = {
            'p25': np.percentile(vol_pct, 25),
            'p50': np.percentile(vol_pct, 50),
            'p75': np.percentile(vol_pct, 75),
            'p95': np.percentile(vol_pct, 95)
        }
    else:
        percentiles = {'p25': 0, 'p50': 0, 'p75': 0, 'p95': 0}
    
    return result_df, percentiles

# Main chart section
with st.spinner(f"Loading data for {selected_token}..."):
    # Fetch raw price data once
    raw_price_data = get_raw_price_data(selected_token)
    
    # Calculate volatilities
    vol_data_10sec, percentiles_10sec = get_volatility_data_10sec(raw_price_data) if "10-Second Volatility" in display_options else (None, None)
    vol_data_20sec, percentiles_20sec = get_volatility_data_20sec(raw_price_data) if "20-Second Volatility" in display_options else (None, None)
    
    # Fetch parameters
    rollbit_params = fetch_rollbit_parameters_10sec(selected_token)
    uat_buffer = fetch_uat_buffer_rates_10sec(selected_token)

# Determine how many charts to show based on user selection
active_charts = []
if "10-Second Volatility" in display_options and vol_data_10sec is not None:
    active_charts.append(("10-Second", vol_data_10sec, percentiles_10sec))
if "20-Second Volatility" in display_options and vol_data_20sec is not None:
    active_charts.append(("20-Second", vol_data_20sec, percentiles_20sec))

if len(active_charts) > 0:
    # Calculate how many rows we need (vol charts + buffer charts)
    num_rows = len(active_charts) + 2  # Add 2 for Rollbit and UAT buffer
    
    # Create row heights based on number of charts
    row_heights = [0.3] * len(active_charts) + [0.2, 0.2]  # Volatility charts + buffer charts
    
    # Create subplots
    fig = make_subplots(
        rows=num_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=row_heights
    )
    
    # Create hover templates
    hover_template = (
        "<b>Time: %{x}</b><br>" +
        "Volatility: %{customdata[0]:.1f}%<br>" +
        "Rollbit Buffer: %{customdata[1]:.3f}%<br>" +
        "UAT Buffer: %{customdata[2]:.3f}%<br>" +
        "<extra></extra>"
    )
    
    # Create title components
    title_parts = [f"{selected_token} Analysis Dashboard<br>"]
    subtitle_parts = []
    
    # Add volatility charts
    current_row = 1
    for chart_name, vol_data, percentiles in active_charts:
        # Convert to percentage
        vol_data_pct = vol_data.copy()
        vol_data_pct['realized_vol'] = vol_data_pct['realized_vol'] * 100
        
        # Key metrics for this volatility
        current_vol = vol_data_pct['realized_vol'].iloc[-1]
        avg_vol = vol_data_pct['realized_vol'].mean()
        max_vol = vol_data_pct['realized_vol'].max()
        min_vol = vol_data_pct['realized_vol'].min()
        
        # Calculate current percentile
        all_vols = vol_data_pct['realized_vol'].values
        current_percentile = (all_vols < current_vol).mean() * 100
        
        # Add to subtitle
        subtitle_parts.append(f"{chart_name} Vol: {current_vol:.1f}% ({current_percentile:.0f}th)")
        
        # Process into combined dataframe
        combined_data = vol_data_pct.copy()
        
        # Add Rollbit data if available
        if rollbit_params is not None and not rollbit_params.empty:
            rollbit_params['buffer_rate_pct'] = rollbit_params['buffer_rate'] * 100
            combined_data = pd.merge(
                combined_data,
                rollbit_params[['buffer_rate_pct']],
                left_index=True,
                right_index=True,
                how='left',
                suffixes=('', '_rollbit')
            )
            combined_data['buffer_rate_pct'] = combined_data['buffer_rate_pct'].ffill()
        else:
            combined_data['buffer_rate_pct'] = np.nan
        
        # Add UAT data if available
        if uat_buffer is not None and not uat_buffer.empty:
            uat_buffer['uat_buffer_rate_pct'] = uat_buffer['buffer_rate'] * 100
            combined_data = pd.merge(
                combined_data,
                uat_buffer[['uat_buffer_rate_pct']],
                left_index=True,
                right_index=True,
                how='left'
            )
            combined_data['uat_buffer_rate_pct'] = combined_data['uat_buffer_rate_pct'].ffill()
        else:
            combined_data['uat_buffer_rate_pct'] = np.nan
        
        # Create unified hover data
        customdata = np.column_stack((
            combined_data['realized_vol'],
            combined_data['buffer_rate_pct'].fillna(0),
            combined_data['uat_buffer_rate_pct'].fillna(0)
        ))
        
        # Add volatility trace
        fig.add_trace(
            go.Scatter(
                x=combined_data.index,
                y=combined_data['realized_vol'],
                mode='lines',
                line=dict(color='blue', width=2),
                name=f"{chart_name} Volatility (%)",
                customdata=customdata,
                hovertemplate=hover_template,
                showlegend=True
            ),
            row=current_row, col=1
        )
        
        # Add percentile lines
        percentile_lines = [
            ('p25', '#2ECC71', '25th'),
            ('p50', '#3498DB', '50th'),
            ('p75', '#F39C12', '75th'),
            ('p95', '#E74C3C', '95th')
        ]

        for key, color, label in percentile_lines:
            fig.add_hline(
                y=percentiles[key],
                line_dash="dash",
                line_color=color,
                line_width=2,
                annotation_text=f"{label}: {percentiles[key]:.1f}%",
                annotation_position="left",
                annotation_font_color=color,
                row=current_row, col=1
            )
        
        # Update y-axis range
        fig.update_yaxes(
            title_text=f"{chart_name} Volatility (%)",
            row=current_row, col=1,
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray',
            range=[0, max(max_vol * 1.1, percentiles['p95'] * 1.1, 5)]
        )
        
        current_row += 1
    
    # Add buffer rate charts
    # Panel 1: Rollbit Buffer Rate
    if rollbit_params is not None and not rollbit_params.empty:
        rollbit_params['buffer_rate_pct'] = rollbit_params['buffer_rate'] * 100
        
        # Create unified hover data
        customdata_rollbit = np.column_stack((
            np.zeros(len(rollbit_params)),  # Placeholder for volatility
            rollbit_params['buffer_rate_pct'],
            np.zeros(len(rollbit_params))   # Placeholder for UAT buffer
        ))
        
        fig.add_trace(
            go.Scatter(
                x=rollbit_params.index,
                y=rollbit_params['buffer_rate_pct'],
                mode='lines+markers',
                line=dict(color='darkgreen', width=3),
                marker=dict(size=4),
                name="Rollbit Buffer Rate (%)",
                customdata=customdata_rollbit,
                hovertemplate=hover_template,
                showlegend=True
            ),
            row=current_row, col=1
        )
        latest_rollbit_buffer = rollbit_params['buffer_rate_pct'].iloc[-1]
        subtitle_parts.append(f"Rollbit Buffer: {latest_rollbit_buffer:.3f}%")
    else:
        fig.add_annotation(
            x=0.5,
            y=0.5,
            text="No Rollbit data available",
            showarrow=False,
            font=dict(size=12),
            xref=f"x{current_row} domain",
            yref=f"y{current_row} domain",
            row=current_row, col=1
        )
        latest_rollbit_buffer = None
    
    current_row += 1
    
    # Panel 2: UAT Buffer Rate
    if uat_buffer is not None and not uat_buffer.empty:
        uat_buffer['uat_buffer_rate_pct'] = uat_buffer['buffer_rate'] * 100
        
        # Create unified hover data
        customdata_uat = np.column_stack((
            np.zeros(len(uat_buffer)),  # Placeholder for volatility
            np.zeros(len(uat_buffer)),  # Placeholder for Rollbit buffer
            uat_buffer['uat_buffer_rate_pct']
        ))
        
        fig.add_trace(
            go.Scatter(
                x=uat_buffer.index,
                y=uat_buffer['uat_buffer_rate_pct'],
                mode='lines+markers',
                line=dict(color='purple', width=3),
                marker=dict(size=4),
                name="UAT Buffer Rate (%)",
                customdata=customdata_uat,
                hovertemplate=hover_template,
                showlegend=True,
                connectgaps=False  # Don't connect gaps in data
            ),
            row=current_row, col=1
        )
        
        # Get the latest non-null value
        latest_uat_values = uat_buffer['uat_buffer_rate_pct'].dropna()
        latest_uat_buffer = latest_uat_values.iloc[-1] if len(latest_uat_values) > 0 else None
        
        if latest_uat_buffer is not None:
            subtitle_parts.append(f"UAT Buffer: {latest_uat_buffer:.3f}%")
    else:
        fig.add_annotation(
            x=0.5,
            y=0.5,
            text="No UAT data available",
            showarrow=False,
            font=dict(size=12),
            xref=f"x{current_row} domain",
            yref=f"y{current_row} domain",
            row=current_row, col=1
        )
        latest_uat_buffer = None
    
    # Find the min and max values across both buffer rates to create a consistent scale
    buffer_min = 0
    buffer_max = 0.075  # Fixed maximum at 7.5 basis points
    
    # Update y-axes for buffer charts
    fig.update_yaxes(
        title_text="Rollbit Buffer (%)",
        row=num_rows-1, col=1,
        tickformat=".4f",
        showgrid=True,
        gridwidth=1,
        gridcolor='LightGray',
        range=[buffer_min, buffer_max]
    )
    
    fig.update_yaxes(
        title_text="UAT Buffer (%)", 
        row=num_rows, col=1,
        tickformat=".4f",
        showgrid=True,
        gridwidth=1,
        gridcolor='LightGray',
        range=[buffer_min, buffer_max],  # Same exact range as Rollbit panel
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor='gray'
    )
    
    # Complete the title
    title_text = title_parts[0] + f"<sub>{' | '.join(subtitle_parts)}</sub>"
    
    # Update layout
    fig.update_layout(
        title=title_text,
        height=200 * num_rows,  # Dynamic height based on number of rows
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="x unified",
        plot_bgcolor='white',
        paper_bgcolor='white',
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="black",
            font_size=12
        ),
        xaxis=dict(
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikethickness=2,
            spikecolor="gray",
            spikedash="solid"
        )
    )
    
    # Update all x-axes to have spikes
    for i in range(1, num_rows + 1):
        fig.update_xaxes(
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikethickness=2,
            spikecolor="gray",
            spikedash="solid",
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray',
            row=i, col=1
        )
    
    # X-axis labels only on bottom
    fig.update_xaxes(title_text="Time (Singapore)", row=num_rows, col=1, tickformat="%H:%M:%S<br>%m/%d")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add chart descriptions
    st.markdown("### Chart Descriptions")
    
    # Description for volatility charts
    vol_row = 1
    for chart_name, vol_data, percentiles in active_charts:
        st.markdown(f"**Panel {vol_row}**: {chart_name} Annualized Volatility")
        st.markdown(f"Shows the annualized volatility calculated using {chart_name.lower()} price windows. The percentile lines indicate historical volatility levels over the past 3 hours.")
        vol_row += 1
    
    # Description for buffer charts
    st.markdown(f"**Panel {vol_row}**: Rollbit Buffer Rate (%)")
    st.markdown("Displays the Rollbit buffer rate from production database over time. This rate determines the risk management parameters for trading.")
    vol_row += 1
    
    st.markdown(f"**Panel {vol_row}**: UAT Buffer Rate (%)")
    st.markdown("Shows the UAT (test environment) buffer rate for comparison with production Rollbit rates. This helps verify if test parameters align with production.")
    
    # Display metrics for each volatility window
    for chart_name, vol_data, percentiles in active_charts:
        vol_data_pct = vol_data.copy()
        vol_data_pct['realized_vol'] = vol_data_pct['realized_vol'] * 100
        
        current_vol = vol_data_pct['realized_vol'].iloc[-1]
        avg_vol = vol_data_pct['realized_vol'].mean()
        max_vol = vol_data_pct['realized_vol'].max()
        min_vol = vol_data_pct['realized_vol'].min()
        
        all_vols = vol_data_pct['realized_vol'].values
        current_percentile = (all_vols < current_vol).mean() * 100
        
        st.markdown(f"### {chart_name} Volatility Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current", f"{current_vol:.1f}%", f"{current_percentile:.0f}th %ile")
        with col2:
            st.metric("Average", f"{avg_vol:.1f}%")
        with col3:
            st.metric("Max", f"{max_vol:.1f}%")
        with col4:
            st.metric("Min", f"{min_vol:.1f}%")
        
        # Percentile display
        st.markdown(f"### {chart_name} Percentiles (3h)")
        pcol1, pcol2, pcol3, pcol4 = st.columns(4)
        with pcol1:
            st.metric("25th", f"{percentiles['p25']:.1f}%")
        with pcol2:
            st.metric("50th", f"{percentiles['p50']:.1f}%")
        with pcol3:
            st.metric("75th", f"{percentiles['p75']:.1f}%")
        with pcol4:
            st.metric("95th", f"{percentiles['p95']:.1f}%")
    
    # Current parameters section
    st.markdown("### Current Parameters")
    
    # Buffer rate comparison
    bcol1, bcol2, bcol3 = st.columns(3)
    
    with bcol1:
        if latest_rollbit_buffer is not None:
            st.metric("Rollbit Buffer Rate", f"{latest_rollbit_buffer:.3f}%")
        else:
            st.metric("Rollbit Buffer Rate", "N/A")
    
    with bcol2:
        if latest_uat_buffer is not None:
            # Use more decimal places for small values
            if latest_uat_buffer < 0.1:
                st.metric("UAT Buffer Rate", f"{latest_uat_buffer:.4f}%")
            else:
                st.metric("UAT Buffer Rate", f"{latest_uat_buffer:.3f}%")
        else:
            st.metric("UAT Buffer Rate", "N/A")
    
    with bcol3:
        if latest_rollbit_buffer is not None and latest_uat_buffer is not None:
            diff = latest_uat_buffer - latest_rollbit_buffer
            diff_pct = (diff / latest_rollbit_buffer * 100) if latest_rollbit_buffer != 0 else 0
            # Use more decimal places for small differences
            if abs(diff) < 0.01:
                st.metric("UAT vs Rollbit", f"{diff:.4f}%", f"{diff_pct:.1f}% diff")
            else:
                st.metric("UAT vs Rollbit", f"{diff:.3f}%", f"{diff_pct:.1f}% diff")
        else:
            st.metric("UAT vs Rollbit", "N/A")

else:
    st.error("No volatility data available for the selected token")