# Save this as simple_volatility_plot_with_rollbit_historical.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import psycopg2
import pytz

st.set_page_config(page_title="5min Volatility Plot with Rollbit", page_icon="📈", layout="wide")

# --- UI Setup ---
st.title("5-Minute Volatility Plot with Historical Rollbit Parameters")

# DB connection
db_params = {
    'host': 'aws-jp-tk-surf-pg-public.cluster-csteuf9lw8dv.ap-northeast-1.rds.amazonaws.com',
    'port': 5432,
    'database': 'replication_report',
    'user': 'public_replication',
    'password': '866^FKC4hllk'
}

conn = psycopg2.connect(**db_params)

# Get available tokens
def fetch_trading_pairs():
    query = """
    SELECT pair_name 
    FROM trade_pool_pairs 
    WHERE status = 1
    ORDER BY pair_name
    """

    df = pd.read_sql_query(query, conn)
    return df['pair_name'].tolist()

# Get all tokens
all_tokens = fetch_trading_pairs()

# UI Controls
col1, col2 = st.columns([3, 1])

with col1:
    # Select token
    default_token = "BTC/USDT" if "BTC/USDT" in all_tokens else all_tokens[0]
    selected_token = st.selectbox(
        "Select Token",
        all_tokens,
        index=all_tokens.index(default_token) if default_token in all_tokens else 0
    )

with col2:
    # Refresh button
    if st.button("Refresh Data", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.experimental_rerun()

# Singapore time
sg_tz = pytz.timezone('Asia/Singapore')
now_utc = datetime.now(pytz.utc)
now_sg = now_utc.astimezone(sg_tz)
st.write(f"Current time (Singapore): {now_sg.strftime('%Y-%m-%d %H:%M:%S')}")

# Fetch historical Rollbit parameters
@st.cache_data(ttl=300)
def fetch_rollbit_parameters_historical(token, hours=24):
    """Fetch historical Rollbit parameters for the selected token"""
    try:
        # Time range matching volatility data
        now_sg = datetime.now(pytz.timezone('Asia/Singapore'))
        start_time_sg = now_sg - timedelta(hours=hours+1)  # Extra hour for buffer

        # Convert to UTC for database query
        start_time_utc = start_time_sg.astimezone(pytz.utc)
        end_time_utc = now_sg.astimezone(pytz.utc)

        query = """
        SELECT 
            pair_name,
            buffer_rate,
            position_multiplier,
            created_at + INTERVAL '8 hour' AS timestamp
        FROM rollbit_pair_config 
        WHERE pair_name = %s
        AND created_at >= %s::timestamp - INTERVAL '8 hour'
        AND created_at <= %s::timestamp - INTERVAL '8 hour'
        ORDER BY created_at
        """

        df = pd.read_sql_query(query, conn, params=(
            token,
            start_time_utc.strftime("%Y-%m-%d %H:%M:%S"),
            end_time_utc.strftime("%Y-%m-%d %H:%M:%S")
        ))

        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').sort_index()
            return df
        return None
    except Exception as e:
        st.error(f"Error fetching Rollbit parameters: {e}")
        return None

# Get partition tables
def get_partition_tables(start_date, end_date):
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)

    # Remove timezone
    start_date = start_date.replace(tzinfo=None)
    end_date = end_date.replace(tzinfo=None)

    # Generate all dates
    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date.strftime("%Y%m%d"))
        current_date += timedelta(days=1)

    # Table names
    table_names = [f"oracle_price_log_partition_{date}" for date in dates]

    # Check which tables exist
    cursor = conn.cursor()
    if table_names:
        table_list_str = "', '".join(table_names)
        cursor.execute(f"""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('{table_list_str}')
        """)

        existing_tables = [row[0] for row in cursor.fetchall()]
    cursor.close()

    return existing_tables

# Build query for partition tables - Modified for 500ms data
def build_query(tables, token, start_time, end_time):
    if not tables:
        return ""

    union_parts = []
    for table in tables:
        # IMPORTANT: Add 8 hours to convert to Singapore time
        query = f"""
        SELECT 
            pair_name,
            created_at + INTERVAL '8 hour' AS timestamp,
            final_price
        FROM 
            public.{table}
        WHERE 
            created_at >= '{start_time}'::timestamp - INTERVAL '8 hour'
            AND created_at <= '{end_time}'::timestamp - INTERVAL '8 hour'
            AND source_type = 0
            AND pair_name = '{token}'
        ORDER BY created_at
        """
        union_parts.append(query)

    return " UNION ALL ".join(union_parts) + " ORDER BY timestamp"

# Calculate volatility
@st.cache_data(ttl=60)  # Short cache to ensure fresh data
def get_volatility_data(token, hours=24):
    # Time range
    now_sg = datetime.now(pytz.timezone('Asia/Singapore'))
    start_time_sg = now_sg - timedelta(hours=hours+1)  # Extra hour for buffer

    # Get relevant partition tables (today and yesterday)
    start_date = start_time_sg.replace(tzinfo=None)
    end_date = now_sg.replace(tzinfo=None)
    partition_tables = get_partition_tables(start_date, end_date)

    if not partition_tables:
        st.error(f"No data tables found for {start_date} to {end_date}")
        return None

    # Convert to strings for query
    start_time_str = start_time_sg.strftime("%Y-%m-%d %H:%M:%S")
    end_time_str = now_sg.strftime("%Y-%m-%d %H:%M:%S")

    # Build and execute query
    query = build_query(partition_tables, token, start_time_str, end_time_str)
    df = pd.read_sql_query(query, conn)

    if df.empty:
        st.error(f"No data found for {token}")
        return None

    # Process timestamps
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()

    # Resample to 500ms intervals
    price_data = df['final_price'].resample('500ms').ffill().dropna()

    # Create 5-minute windows
    result = []
    start_date = price_data.index.min().floor('5min')
    end_date = price_data.index.max().ceil('5min')
    five_min_periods = pd.date_range(start=start_date, end=end_date, freq='5min')

    for i in range(len(five_min_periods)-1):
        start_window = five_min_periods[i]
        end_window = five_min_periods[i+1]

        # Get price data in this window
        window_data = price_data[(price_data.index >= start_window) & (price_data.index < end_window)]

        if len(window_data) >= 2:  # Need at least 2 points for volatility
            # OHLC data
            window_open = window_data.iloc[0]
            window_high = window_data.max()
            window_low = window_data.min()
            window_close = window_data.iloc[-1]

            # Calculate volatility using 500ms data points
            # Log returns
            log_returns = np.diff(np.log(window_data.values))

            # Annualize: 500ms intervals in year / 500ms intervals in 5 minutes
            # There are 63,072,000 half-seconds in a year and 600 half-seconds in 5 minutes
            annualization_factor = np.sqrt(63072000 / 600)
            volatility = np.std(log_returns) * annualization_factor

            result.append({
                'timestamp': start_window,
                'open': window_open,
                'high': window_high,
                'low': window_low,
                'close': window_close,
                'realized_vol': volatility,
                'data_points': len(window_data),  # Track how many 500ms points we have
                'actual_data_interval': 0.5  # 500ms
            })

    # Create dataframe and get last 24 hours of data
    if not result:
        st.error(f"Could not calculate volatility for {token}")
        return None

    result_df = pd.DataFrame(result).set_index('timestamp')
    periods_needed = hours * 12  # 12 5-minute periods per hour

    # Get most recent data
    return result_df.tail(periods_needed)

# Get data for selected token
with st.spinner(f"Calculating volatility and fetching Rollbit parameters for {selected_token}..."):
    vol_data = get_volatility_data(selected_token)
    rollbit_params = fetch_rollbit_parameters_historical(selected_token)

# Create the plot
if vol_data is not None and not vol_data.empty:
    # Convert to percentage
    vol_data_pct = vol_data.copy()
    vol_data_pct['realized_vol'] = vol_data_pct['realized_vol'] * 100

    # Key metrics
    avg_vol = vol_data_pct['realized_vol'].mean()
    max_vol = vol_data_pct['realized_vol'].max()
    current_vol = vol_data_pct['realized_vol'].iloc[-1]
    avg_data_points = vol_data_pct['data_points'].mean()

    # Create figure with secondary y-axis
    fig = go.Figure()

    # Color coding for volatility
    colors = []
    for val in vol_data_pct['realized_vol']:
        if pd.isna(val):
            colors.append('gray')
        elif val < 30:
            colors.append('green')
        elif val < 60:
            colors.append('gold')
        elif val < 100:
            colors.append('orange')
        else:
            colors.append('red')

    # Add volatility line (left y-axis)
    fig.add_trace(
        go.Scatter(
            x=vol_data_pct.index,
            y=vol_data_pct['realized_vol'],
            mode='lines+markers',
            line=dict(color='blue', width=2),
            marker=dict(color=colors, size=7),
            name="Volatility (%)",
            hovertemplate="<b>Time: %{x}</b><br>Volatility: %{y:.1f}%<br>Buffer Rate: N/A<br>Position Mult: N/A<extra></extra>",
            showlegend=False
        )
    )

    # Add threshold lines for volatility
    for threshold, color, label in [(30, "green", "Low"), (60, "gold", "Medium"), (100, "red", "Extreme")]:
        fig.add_shape(
            type="line",
            x0=vol_data_pct.index.min(),
            x1=vol_data_pct.index.max(),
            y0=threshold,
            y1=threshold,
            line=dict(color=color, width=1, dash="dash"),
            yref="y"
        )

    # Add Rollbit parameters if available
    if rollbit_params is not None and not rollbit_params.empty:
        # Resample Rollbit data to 5-minute intervals to match volatility data
        rollbit_resampled = rollbit_params.resample('5min').ffill()

        # Merge with volatility data to ensure aligned timestamps
        combined_data = pd.merge(
            vol_data_pct,
            rollbit_resampled,
            left_index=True,
            right_index=True,
            how='left',
            suffixes=('', '_rollbit')
        )

        # Forward fill any missing Rollbit values
        combined_data['buffer_rate'] = combined_data['buffer_rate'].ffill()
        combined_data['position_multiplier'] = combined_data['position_multiplier'].ffill()

        # Convert buffer rate to percentage
        combined_data['buffer_rate_pct'] = combined_data['buffer_rate'] * 100

        # Add buffer rate line (right y-axis)
        fig.add_trace(
            go.Scatter(
                x=combined_data.index,
                y=combined_data['buffer_rate_pct'],
                mode='lines+markers',
                line=dict(color='darkgreen', width=2),
                name="Rollbit Buffer Rate (%)",
                yaxis='y2',
                customdata=np.column_stack((
                    combined_data['realized_vol'],
                    combined_data['buffer_rate_pct'],
                    combined_data['position_multiplier']
                )),
                hovertemplate="<b>%{x}</b><br>" +
                              "Volatility: %{customdata[0]:.1f}%<br>" +
                              "Buffer Rate: %{customdata[1]:.3f}%<br>" +
                              "Position Mult: %{customdata[2]:,.0f}<extra></extra>"
            )
        )

        # Add position multiplier line (right y-axis)
        # Scale down position multiplier for better visualization
        scale_factor = 100000  # Adjust this based on your data range
        fig.add_trace(
            go.Scatter(
                x=combined_data.index,
                y=combined_data['position_multiplier'] / scale_factor,
                mode='lines+markers',
                line=dict(color='darkblue', width=2),
                name=f"Rollbit Position Multiplier (÷{scale_factor:,})",
                yaxis='y2',
                customdata=np.column_stack((
                    combined_data['realized_vol'],
                    combined_data['buffer_rate_pct'],
                    combined_data['position_multiplier']
                )),
                hovertemplate="<b>%{x}</b><br>" +
                              "Volatility: %{customdata[0]:.1f}%<br>" +
                              "Buffer Rate: %{customdata[1]:.3f}%<br>" +
                              "Position Mult: %{customdata[2]:,.0f}<extra></extra>"
            )
        )

    # Update layout with dual y-axes
    fig.update_layout(
        title=f"{selected_token} Volatility vs Rollbit Parameters<br>" +
              f"<sub>Current Vol: {current_vol:.1f}%, Avg: {avg_vol:.1f}%, Max: {max_vol:.1f}%</sub>",
        height=700,
        showlegend=True,
        hovermode="x unified",
        yaxis=dict(
            title="Annualized Volatility (%)",
            side="left",
            range=[0, max(120, max_vol * 1.2)]
        ),
        yaxis2=dict(
            title="Rollbit Parameters",
            side="right",
            overlaying="y",
            anchor="x"
        ),
        xaxis=dict(
            title="Time (Singapore)",
            tickformat="%H:%M<br>%m/%d",
            tickangle=-45
        )
    )

    # Display chart
    st.plotly_chart(fig, use_container_width=True)

    # Display data range information
    st.markdown(f"### Data Time Range")
    st.markdown(f"- First data point: {vol_data.index[0].strftime('%Y-%m-%d %H:%M')}")
    st.markdown(f"- Last data point: {vol_data.index[-1].strftime('%Y-%m-%d %H:%M')}")
    st.markdown(f"- Most recent: {(now_sg - vol_data.index[-1]).total_seconds() / 60:.1f} minutes ago")
    st.markdown(f"- Data granularity: 500ms")

    # Display current Rollbit parameters if available
    if rollbit_params is not None and not rollbit_params.empty:
        st.markdown(f"### Current Rollbit Parameters for {selected_token}")
        latest_params = rollbit_params.iloc[-1]
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Buffer Rate", f"{latest_params.get('buffer_rate', 0) * 100:.3f}%")
        with col2:
            st.metric("Position Multiplier", f"{latest_params.get('position_multiplier', 0):,.0f}")

    # Display highest volatility periods
    st.subheader("Highest Volatility Periods")
    top_periods = vol_data_pct.sort_values(by='realized_vol', ascending=False).head(5)
    top_periods['Time'] = top_periods.index.strftime('%Y-%m-%d %H:%M')
    top_periods['Volatility (%)'] = top_periods['realized_vol'].round(1)

    st.table(top_periods[['Time', 'Volatility (%)']])

    # If no Rollbit data, still show volatility with just volatility values
    fig.add_trace(
        go.Scatter(
            x=vol_data_pct.index,
            y=vol_data_pct['realized_vol'],
            mode='lines+markers',
            line=dict(color='blue', width=2),
            marker=dict(color=colors, size=7),
            name="Volatility (%)",
            hovertemplate="<b>Time: %{x}</b><br>Volatility: %{y:.1f}%<br>Buffer Rate: N/A<br>Position Mult: N/A<extra></extra>",
            showlegend=False
        )
    )
    
    # 简化注释设置
    fig.add_annotation(
        x=vol_data_pct.index[len(vol_data_pct)//2],
        y=0.5,
        text="No Rollbit data available",
        showarrow=False,
        font=dict(size=12),
        row=1, col=1
    )

    # 修改百分位线注释
    for i, (key, color, label) in enumerate(percentile_lines):
        if y_min <= percentiles[key] <= y_max and percentiles[key] > 0:
            fig.add_shape(
                type="line",
                x0=vol_data_pct.index.min(),
                x1=vol_data_pct.index.max(),
                y0=percentiles[key],
                y1=percentiles[key],
                line=dict(color=color, width=3, dash="dash"),
                row=1, col=1
            )
            
            # 简化注释设置
            fig.add_annotation(
                x=vol_data_pct.index[10],
                y=percentiles[key],
                text=f"{label}: {percentiles[key]:.1f}%",
                showarrow=False,
                font=dict(size=9),
                xanchor="left",
                yanchor="middle",
                row=1, col=1
            )
            
            fig.add_annotation(
                x=vol_data_pct.index[-10],
                y=percentiles[key],
                text=f"{label}: {percentiles[key]:.1f}%",
                showarrow=False,
                font=dict(size=9),
                xanchor="right",
                yanchor="middle",
                row=1, col=1
            )

else:
    st.error("No volatility data available for the selected token")
