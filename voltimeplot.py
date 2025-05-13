# Save this as complete_volatility_with_levels.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import psycopg2
import pytz
from sqlalchemy import create_engine

st.set_page_config(page_title="5min Volatility Plot with Rollbit", page_icon="📈", layout="wide")

# --- UI Setup ---
st.title("5-Minute Volatility Plot with Historical Rollbit Parameters")

# DB connection using SQLAlchemy for better pandas compatibility
db_params = {
    'host': 'aws-jp-tk-surf-pg-public.cluster-csteuf9lw8dv.ap-northeast-1.rds.amazonaws.com',
    'port': 5432,
    'database': 'replication_report',
    'user': 'public_replication',
    'password': '866^FKC4hllk'
}

# Create SQLAlchemy engine for pandas
engine = create_engine(
    f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['database']}"
)

# Keep psycopg2 connection for cursor operations
conn = psycopg2.connect(**db_params)

# Get available tokens
def fetch_trading_pairs():
    query = """
    SELECT pair_name 
    FROM trade_pool_pairs 
    WHERE status = 1
    ORDER BY pair_name
    """

    df = pd.read_sql_query(query, engine)
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
        st.rerun()

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
        start_time_utc = start_time_sg.replace(tzinfo=None)
        end_time_utc = now_sg.replace(tzinfo=None)

        # Format timestamps
        start_str = start_time_utc.strftime("%Y-%m-%d %H:%M:%S")
        end_str = end_time_utc.strftime("%Y-%m-%d %H:%M:%S")

        query = f"""
        SELECT 
            pair_name,
            bust_buffer AS buffer_rate,
            position_multiplier,
            created_at + INTERVAL '8 hour' AS timestamp
        FROM rollbit_pair_config 
        WHERE pair_name = '{token}'
        AND created_at >= '{start_str}'::timestamp - INTERVAL '8 hour'
        AND created_at <= '{end_str}'::timestamp - INTERVAL '8 hour'
        ORDER BY created_at
        """

        df = pd.read_sql_query(query, engine)

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
        # Note: Don't add ORDER BY to individual queries before UNION
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
        """
        union_parts.append(query)

    return " UNION ALL ".join(union_parts) + " ORDER BY timestamp"

# Calculate volatility with percentiles
@st.cache_data(ttl=60)  # Short cache to ensure fresh data
def get_volatility_data_with_percentiles(token, display_hours=24, history_days=3):
    # Time range - only 3 days for faster loading
    now_sg = datetime.now(pytz.timezone('Asia/Singapore'))
    start_time_sg = now_sg - timedelta(days=history_days)

    # Get relevant partition tables (today and yesterday)
    start_date = start_time_sg.replace(tzinfo=None)
    end_date = now_sg.replace(tzinfo=None)
    partition_tables = get_partition_tables(start_date, end_date)

    if not partition_tables:
        st.error(f"No data tables found for {start_date} to {end_date}")
        return None, None

    # Convert to strings for query
    start_time_str = start_time_sg.strftime("%Y-%m-%d %H:%M:%S")
    end_time_str = now_sg.strftime("%Y-%m-%d %H:%M:%S")

    # Build and execute query
    query = build_query(partition_tables, token, start_time_str, end_time_str)

    with st.spinner(f"Loading data..."):
        df = pd.read_sql_query(query, engine)

    if df.empty:
        st.error(f"No data found for {token}")
        return None, None

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
        return None, None

    result_df = pd.DataFrame(result).set_index('timestamp')

    # Split into display and historical data
    display_periods = display_hours * 12  # 12 5-minute periods per hour
    display_df = result_df.tail(display_periods)

    return display_df, result_df

# Get data for selected token
with st.spinner(f"Calculating volatility and fetching Rollbit parameters for {selected_token}..."):
    vol_data, historical_vol_data = get_volatility_data_with_percentiles(selected_token)
    rollbit_params = fetch_rollbit_parameters_historical(selected_token)

# Create the plot
if vol_data is not None and not vol_data.empty and historical_vol_data is not None and not historical_vol_data.empty:
    # Convert to percentage
    vol_data_pct = vol_data.copy()
    vol_data_pct['realized_vol'] = vol_data_pct['realized_vol'] * 100

    historical_vol_pct = historical_vol_data.copy()
    historical_vol_pct['realized_vol'] = historical_vol_pct['realized_vol'] * 100

    # Calculate percentiles from historical data
    percentiles = {
        'p25': np.percentile(historical_vol_pct['realized_vol'], 25),
        'p50': np.percentile(historical_vol_pct['realized_vol'], 50),
        'p75': np.percentile(historical_vol_pct['realized_vol'], 75),
        'p95': np.percentile(historical_vol_pct['realized_vol'], 95)
    }

    # Key metrics
    avg_vol = vol_data_pct['realized_vol'].mean()
    max_vol = vol_data_pct['realized_vol'].max()
    current_vol = vol_data_pct['realized_vol'].iloc[-1]
    current_percentile = (historical_vol_pct['realized_vol'] < current_vol).mean() * 100
    avg_data_points = vol_data_pct['data_points'].mean()

    # Create subplots with 3 rows
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,  # This is key for synchronized hover
        vertical_spacing=0.05,
        subplot_titles=(
            f"{selected_token} Annualized Volatility (5min windows, 500ms data)",
            "Rollbit Buffer Rate (%)",
            "Rollbit Position Multiplier"
        ),
        row_heights=[0.4, 0.3, 0.3]
    )

    # Color coding for volatility based on percentiles
    colors = []
    for val in vol_data_pct['realized_vol']:
        if pd.isna(val):
            colors.append('gray')
        elif val < percentiles['p25']:
            colors.append('darkgreen')  # Very low
        elif val < percentiles['p50']:
            colors.append('green')  # Low
        elif val < percentiles['p75']:
            colors.append('gold')  # Normal
        elif val < percentiles['p95']:
            colors.append('orange')  # Elevated
        else:
            colors.append('red')  # High

    # Process Rollbit data if available
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

        # Create combined hover data
        combined_hover = []
        for i in range(len(combined_data)):
            hover_text = (
                    f"<b>Time: {combined_data.index[i].strftime('%Y-%m-%d %H:%M')}</b><br>" +
                    f"Volatility: {combined_data['realized_vol'].iloc[i]:.1f}%<br>" +
                    f"Buffer Rate: {combined_data['buffer_rate_pct'].iloc[i]:.3f}%<br>" +
                    f"Position Mult: {combined_data['position_multiplier'].iloc[i]:,.0f}"
            )
            combined_hover.append(hover_text)

        # Panel 1: Volatility (now with combined hover data)
        fig.add_trace(
            go.Scatter(
                x=combined_data.index,
                y=combined_data['realized_vol'],
                mode='lines+markers',
                line=dict(color='blue', width=2),
                marker=dict(color=colors[:len(combined_data)], size=7),  # Adjust colors length
                name="Volatility (%)",
                hovertemplate=combined_hover,
                showlegend=False
            ),
            row=1, col=1
        )

        # Panel 2: Buffer Rate
        fig.add_trace(
            go.Scatter(
                x=combined_data.index,
                y=combined_data['buffer_rate_pct'],
                mode='lines+markers',
                line=dict(color='darkgreen', width=3),
                marker=dict(size=8),
                name="Buffer Rate (%)",
                hovertemplate=combined_hover,
                showlegend=False
            ),
            row=2, col=1
        )

        # Panel 3: Position Multiplier
        fig.add_trace(
            go.Scatter(
                x=combined_data.index,
                y=combined_data['position_multiplier'],
                mode='lines+markers',
                line=dict(color='darkblue', width=3),
                marker=dict(size=8),
                name="Position Multiplier",
                hovertemplate=combined_hover,
                showlegend=False
            ),
            row=3, col=1
        )
    else:
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
            ),
            row=1, col=1
        )

        # Add notes if no Rollbit data
        fig.add_annotation(
            x=vol_data_pct.index[len(vol_data_pct)//2],
            y=0.5,
            text="No Rollbit data available",
            showarrow=False,
            font=dict(size=12),
            row=1, col=1
        )

    # Calculate better y-axis range for volatility
    vol_min = vol_data_pct['realized_vol'].min()
    vol_max = vol_data_pct['realized_vol'].max()

    # Look at actual data range
    data_range = vol_max - vol_min

    # Start from 0
    y_min = 0

    # Determine y_max based on volatility levels
    if vol_max < 5:
        # Low volatility assets (like BTC) - use tight scaling
        y_max = vol_max * 1.2
        # But ensure we show at least up to 5% for context
        y_max = max(y_max, 5)
    elif vol_max < 50:
        # Medium volatility - add 15% padding
        y_max = vol_max * 1.15
    else:
        # High volatility assets (like PNUT) - use appropriate scaling
        # Add 10% padding but ensure we capture the full range
        y_max = vol_max * 1.1

        # For extremely high volatility, ensure the scale is appropriate
        if vol_max > 100:
            # Round up to nearest 50 for cleaner scale
            y_max = ((vol_max // 50) + 1) * 50

    # Make sure all percentiles that matter are visible
    if percentiles['p95'] > y_max:
        y_max = percentiles['p95'] * 1.1

    # Add percentile lines for volatility with cleaner labels
    percentile_lines = [
        ('p25', 'green', '25th'),
        ('p50', 'blue', '50th'),
        ('p75', 'gold', '75th'),
        ('p95', 'red', '95th')
    ]

    # Calculate actual visible range for better annotation placement
    visible_range = y_max - y_min

    for i, (key, color, label) in enumerate(percentile_lines):
        # Only show percentiles that are within visible range and make sense
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

    # Update layout with enhanced hover line
    fig.update_layout(
        title=f"{selected_token} Analysis Dashboard<br>" +
              f"<sub>Current Volatility: {current_vol:.1f}% ({current_percentile:.0f}th percentile)</sub>",
        height=900,
        showlegend=False,
        hovermode="x unified",  # This creates the unified vertical hover line
        plot_bgcolor='white',
        paper_bgcolor='white',
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="black",
            font_size=14
        ),
        # Make the hover line more visible
        spikedistance=1000,
        xaxis=dict(
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            showline=True,
            spikethickness=2,
            spikecolor="gray",
            spikedash="solid"
        )
    )

    # Update axes
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

    # Configure spike lines for all x-axes
    for i in range(1, 4):
        fig.update_xaxes(
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikethickness=2,
            spikecolor="gray",
            spikedash="solid",
            row=i, col=1
        )

    # Only show x-axis label on the bottom plot
    fig.update_xaxes(title_text="Time (Singapore)", row=3, col=1, tickformat="%H:%M<br>%m/%d", tickangle=-45)

    # Update y-axis labels with better scaling for volatility
    # Determine appropriate tick interval based on range
    if y_max < 5:
        dtick = 0.5  # 0.5% increments for very low volatility
    elif y_max < 20:
        dtick = 2.0  # 2% increments for low volatility
    elif y_max < 50:
        dtick = 5.0  # 5% increments for medium volatility
    elif y_max < 100:
        dtick = 10.0  # 10% increments for high volatility
    else:
        dtick = 25.0  # 25% increments for extreme volatility

    fig.update_yaxes(
        title_text="Volatility (%)",
        row=1, col=1,
        range=[y_min, y_max],
        tickformat=".1f",
        dtick=dtick,
        showgrid=True,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinecolor='rgba(128,128,128,0.5)'
    )
    fig.update_yaxes(title_text="Buffer Rate (%)", row=2, col=1, tickformat=".3f")
    fig.update_yaxes(title_text="Position Multiplier", row=3, col=1, tickformat=",")

    # Display chart
    st.plotly_chart(fig, use_container_width=True)

    # Display volatility interpretation
    st.markdown("### Volatility Status")

    if current_vol < percentiles['p25']:
        status = "🟩 **Very Low**"
        interpretation = f"Current volatility ({current_vol:.1f}%) is below the 25th percentile"
    elif current_vol < percentiles['p50']:
        status = "🟢 **Low**"
        interpretation = f"Current volatility ({current_vol:.1f}%) is below the median"
    elif current_vol < percentiles['p75']:
        status = "🟡 **Normal**"
        interpretation = f"Current volatility ({current_vol:.1f}%) is in the normal range"
    elif current_vol < percentiles['p95']:
        status = "🟠 **Elevated**"
        interpretation = f"Current volatility ({current_vol:.1f}%) is above the 75th percentile"
    else:
        status = "🔴 **High**"
        interpretation = f"Current volatility ({current_vol:.1f}%) is above the 95th percentile"

    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown(status)
    with col2:
        st.markdown(interpretation)

    # Display percentile metrics
    st.markdown("### Volatility Percentiles (3-day)")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("25th %ile", f"{percentiles['p25']:.1f}%")
    with col2:
        st.metric("Median", f"{percentiles['p50']:.1f}%")
    with col3:
        st.metric("75th %ile", f"{percentiles['p75']:.1f}%")
    with col4:
        st.metric("95th %ile", f"{percentiles['p95']:.1f}%")

    # Display current metrics
    if rollbit_params is not None and not rollbit_params.empty:
        st.markdown("### Current Metrics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Current Volatility", f"{current_vol:.1f}%")

        with col2:
            st.metric("Average Volatility", f"{avg_vol:.1f}%")

        with col3:
            latest_buffer = rollbit_params['buffer_rate'].iloc[-1] * 100
            st.metric("Current Buffer Rate", f"{latest_buffer:.3f}%")

        with col4:
            latest_pos_mult = rollbit_params['position_multiplier'].iloc[-1]
            st.metric("Current Position Multiplier", f"{latest_pos_mult:,.0f}")

else:
    st.error("No volatility data available for the selected token")
