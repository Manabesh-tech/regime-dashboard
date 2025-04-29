import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import pytz

st.set_page_config(
    page_title="Edge & Volatility Matrix",
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
    engine = create_engine(db_uri)
except Exception as e:
    st.error(f"Error connecting to the database: {e}")
    st.stop()

# --- UI Setup ---
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Edge & Volatility Matrix (10min)")
st.subheader("All Trading Pairs - Last 24 Hours (Singapore Time)")

# Define parameters
timeframe = "10min"
lookback_days = 1  # 24 hours
expected_points = 144  # Expected data points per pair over 24 hours (24 hours * 6 intervals per hour)
singapore_timezone = pytz.timezone('Asia/Singapore')

# Get current time in Singapore timezone
now_utc = datetime.now(pytz.utc)
now_sg = now_utc.astimezone(singapore_timezone)
st.write(f"Current Singapore Time: {now_sg.strftime('%Y-%m-%d %H:%M:%S')}")

# Fetch all available pairs from DB
@st.cache_data(ttl=600, show_spinner="Fetching pairs...")
def fetch_all_pairs():
    query = "SELECT DISTINCT pair_name FROM public.trade_pool_pairs ORDER BY pair_name"
    try:
        df = pd.read_sql(query, engine)
        if df.empty:
            st.error("No pairs found in the database.")
            return []
        return df['pair_name'].tolist()
    except Exception as e:
        st.error(f"Error fetching pairs: {e}")
        return ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT", "PEPE/USDT"]  # Default fallback

all_pairs = fetch_all_pairs()

# UI Controls
col1, col2 = st.columns([3, 1])

with col1:
    # Let user select pairs to display (or select all)
    select_all = st.checkbox("Select All Pairs", value=False)
    
    if select_all:
        selected_pairs = all_pairs
    else:
        selected_pairs = st.multiselect(
            "Select Pairs", 
            all_pairs,
            default=all_pairs[:5] if len(all_pairs) > 5 else all_pairs
        )

with col2:
    # Add a refresh button
    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.experimental_rerun()

if not selected_pairs:
    st.warning("Please select at least one pair")
    st.stop()

# Function to generate aligned 10-minute time blocks for the past 24 hours
def generate_aligned_time_blocks(current_time):
    """
    Generate fixed 10-minute time blocks for past 24 hours,
    aligned with standard 10-minute intervals (e.g., 4:00-4:10, 4:10-4:20)
    """
    # Round down to the nearest 10-minute mark
    minute = current_time.minute
    rounded_minute = (minute // 10) * 10
    latest_complete_block_end = current_time.replace(minute=rounded_minute, second=0, microsecond=0)
    
    # Generate block labels for display
    blocks = []
    for i in range(144):  # 24 hours of 10-minute blocks
        block_end = latest_complete_block_end - timedelta(minutes=i*10)
        block_start = block_end - timedelta(minutes=10)
        block_label = f"{block_start.strftime('%H:%M')}"
        blocks.append((block_start, block_end, block_label))
    
    return blocks

# Generate aligned time blocks
aligned_time_blocks = generate_aligned_time_blocks(now_sg)
time_block_labels = [block[2] for block in aligned_time_blocks]

# Calculate Edge (PNL / Collateral) for each 10-minute time block
@st.cache_data(ttl=600, show_spinner="Calculating edge...")
def fetch_edge_data(pair_name):
    # Get current time in Singapore timezone
    now_utc = datetime.now(pytz.utc)
    now_sg = now_utc.astimezone(singapore_timezone)
    start_time_sg = now_sg - timedelta(days=lookback_days)
    
    # Convert back to UTC for database query
    start_time_utc = start_time_sg.astimezone(pytz.utc)
    end_time_utc = now_sg.astimezone(pytz.utc)

    # This query calculates edge (PNL / Collateral) for each 10-minute interval
    query = f"""
    WITH time_intervals AS (
      -- Generate 10-minute intervals for the past 24 hours
      SELECT
        generate_series(
          date_trunc('hour', '{start_time_utc}'::timestamp) + 
          INTERVAL '10 min' * floor(EXTRACT(MINUTE FROM '{start_time_utc}'::timestamp) / 10),
          '{end_time_utc}'::timestamp,
          INTERVAL '10 minutes'
        ) AS time_slot
    ),
    
    order_pnl AS (
      -- Calculate platform order PNL
      SELECT
        date_trunc('hour', created_at + INTERVAL '8 hour') + 
        INTERVAL '10 min' * FLOOR(EXTRACT(MINUTE FROM created_at + INTERVAL '8 hour')::INT / 10) AS timestamp,
        COALESCE(SUM(taker_pnl * collateral_price), 0) AS platform_order_pnl
      FROM
        public.trade_fill_fresh
      WHERE
        created_at BETWEEN '{start_time_utc}' AND '{end_time_utc}'
        AND pair_name = '{pair_name}'
        AND taker_way IN (1, 2, 3, 4)
      GROUP BY
        timestamp
    ),
    
    fee_data AS (
      -- Calculate fee revenue
      SELECT
        date_trunc('hour', created_at + INTERVAL '8 hour') + 
        INTERVAL '10 min' * FLOOR(EXTRACT(MINUTE FROM created_at + INTERVAL '8 hour')::INT / 10) AS timestamp,
        COALESCE(SUM(-1 * taker_fee * collateral_price), 0) AS fee_revenue
      FROM
        public.trade_fill_fresh
      WHERE
        created_at BETWEEN '{start_time_utc}' AND '{end_time_utc}'
        AND pair_name = '{pair_name}'
        AND taker_fee_mode = 1
        AND taker_way IN (1, 3)
      GROUP BY
        timestamp
    ),
    
    funding_pnl AS (
      -- Calculate funding fee PNL
      SELECT
        date_trunc('hour', created_at + INTERVAL '8 hour') + 
        INTERVAL '10 min' * FLOOR(EXTRACT(MINUTE FROM created_at + INTERVAL '8 hour')::INT / 10) AS timestamp,
        COALESCE(SUM(funding_fee * collateral_price), 0) AS funding_fee_pnl
      FROM
        public.trade_fill_fresh
      WHERE
        created_at BETWEEN '{start_time_utc}' AND '{end_time_utc}'
        AND pair_name = '{pair_name}'
        AND taker_way = 0
      GROUP BY
        timestamp
    ),
    
    sl_fees AS (
      -- Calculate stop loss fees
      SELECT
        date_trunc('hour', created_at + INTERVAL '8 hour') + 
        INTERVAL '10 min' * FLOOR(EXTRACT(MINUTE FROM created_at + INTERVAL '8 hour')::INT / 10) AS timestamp,
        COALESCE(SUM(-taker_sl_fee * collateral_price - maker_sl_fee), 0) AS sl_fee_pnl
      FROM
        public.trade_fill_fresh
      WHERE
        created_at BETWEEN '{start_time_utc}' AND '{end_time_utc}'
        AND pair_name = '{pair_name}'
      GROUP BY
        timestamp
    ),
    
    collateral_data AS (
      -- Calculate open collateral
      SELECT
        date_trunc('hour', created_at + INTERVAL '8 hour') + 
        INTERVAL '10 min' * FLOOR(EXTRACT(MINUTE FROM created_at + INTERVAL '8 hour')::INT / 10) AS timestamp,
        COALESCE(SUM(CASE WHEN taker_fee_mode = 2 AND taker_way IN (1, 3) 
                   THEN deal_vol * collateral_price ELSE 0 END), 0) AS total_collateral,
        array_agg(deal_price ORDER BY created_at) AS price_array
      FROM
        public.trade_fill_fresh
      WHERE
        created_at BETWEEN '{start_time_utc}' AND '{end_time_utc}'
        AND pair_name = '{pair_name}'
      GROUP BY
        timestamp
    )
    
    -- Combine all data sources
    SELECT
      ti.time_slot AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore' AS timestamp_sg,
      COALESCE(o.platform_order_pnl, 0) + 
      COALESCE(f.fee_revenue, 0) + 
      COALESCE(ff.funding_fee_pnl, 0) + 
      COALESCE(sl.sl_fee_pnl, 0) AS total_pnl,
      COALESCE(c.total_collateral, 0) AS total_collateral,
      c.price_array
    FROM
      time_intervals ti
    LEFT JOIN
      order_pnl o ON ti.time_slot = o.timestamp AT TIME ZONE 'Asia/Singapore' AT TIME ZONE 'UTC'
    LEFT JOIN
      fee_data f ON ti.time_slot = f.timestamp AT TIME ZONE 'Asia/Singapore' AT TIME ZONE 'UTC'
    LEFT JOIN
      funding_pnl ff ON ti.time_slot = ff.timestamp AT TIME ZONE 'Asia/Singapore' AT TIME ZONE 'UTC'
    LEFT JOIN
      sl_fees sl ON ti.time_slot = sl.timestamp AT TIME ZONE 'Asia/Singapore' AT TIME ZONE 'UTC'
    LEFT JOIN
      collateral_data c ON ti.time_slot = c.timestamp AT TIME ZONE 'Asia/Singapore' AT TIME ZONE 'UTC'
    ORDER BY
      ti.time_slot DESC
    """
    
    try:
        df = pd.read_sql(query, engine)
        
        if df.empty:
            return None
        
        # Process data
        df['timestamp_sg'] = pd.to_datetime(df['timestamp_sg'])
        df['time_label'] = df['timestamp_sg'].dt.strftime('%H:%M')
        
        # Calculate Edge
        df['edge'] = np.where(df['total_collateral'] > 0, 
                            df['total_pnl'] / df['total_collateral'], 
                            None)
        
        # Calculate volatility
        def calculate_volatility(prices):
            if prices is None or len(prices) < 2:
                return None
            
            try:
                # Handle PostgreSQL array formats
                if isinstance(prices, str):
                    prices = prices.strip('{}').split(',')
                    prices = [float(p) for p in prices if p and p != 'NULL' and p != 'None']
                else:
                    prices = [float(p) for p in prices if p]
                
                if len(prices) < 2:
                    return None
                    
                # Calculate log returns and volatility
                prices_array = np.array(prices)
                log_returns = np.diff(np.log(prices_array))
                
                # Calculate standard deviation and annualize (10-min intervals -> 6 * 24 * 365)
                std_dev = np.std(log_returns)
                volatility = std_dev * np.sqrt(6 * 24 * 365)
                
                return volatility
            except Exception as e:
                return None
        
        # Apply volatility calculation
        df['volatility'] = df['price_array'].apply(calculate_volatility)
        
        return df
    except Exception as e:
        st.error(f"Error calculating edge for {pair_name}: {e}")
        print(f"[{pair_name}] Error calculating edge: {e}")
        return None

# Show progress bar while calculating
progress_bar = st.progress(0)
status_text = st.empty()

# Calculate edge for each pair
pair_results = {}
for i, pair_name in enumerate(selected_pairs):
    try:
        progress_bar.progress((i) / len(selected_pairs))
        status_text.text(f"Processing {pair_name} ({i+1}/{len(selected_pairs)})")
        
        # Fetch edge data
        edge_data = fetch_edge_data(pair_name)
        
        if edge_data is not None:
            pair_results[pair_name] = edge_data
    except Exception as e:
        st.error(f"Error processing pair {pair_name}: {e}")
        print(f"Error processing pair {pair_name} in main loop: {e}")

# Final progress update
progress_bar.progress(1.0)
status_text.text(f"Processed {len(pair_results)}/{len(selected_pairs)} pairs successfully")

# Create tabs for Edge and Volatility
tab1, tab2, tab3 = st.tabs(["Edge", "Volatility", "Combined"])

# Function to color edge values
def edge_color(val):
    if pd.isna(val) or val is None:
        return '#f5f5f5'  # Gray for missing/zero
    elif val < -0.1:
        return 'rgba(180, 0, 0, 0.9)'  # Deep red
    elif val < -0.05:
        return 'rgba(255, 0, 0, 0.9)'  # Red
    elif val < -0.01:
        return 'rgba(255, 150, 150, 0.9)'  # Light red
    elif val < 0.01:
        return 'rgba(255, 255, 150, 0.9)'  # Yellow
    elif val < 0.05:
        return 'rgba(150, 255, 150, 0.9)'  # Light green
    elif val < 0.1:
        return 'rgba(0, 255, 0, 0.9)'  # Green
    else:
        return 'rgba(0, 180, 0, 0.9)'  # Deep green

# Function to color volatility values
def vol_color(val):
    if pd.isna(val) or val is None:
        return '#f5f5f5'  # Gray for missing/zero
    elif val < 0.3:
        return 'rgba(0, 255, 0, 0.9)'  # Green
    elif val < 0.6:
        return 'rgba(255, 255, 0, 0.9)'  # Yellow
    elif val < 1.0:
        return 'rgba(255, 165, 0, 0.9)'  # Orange
    else:
        return 'rgba(255, 0, 0, 0.9)'  # Red

# Format values for display
def format_value(val, type='edge'):
    if pd.isna(val) or val is None:
        return '-'
    
    if type == 'edge':
        return f"{val*100:.2f}%"
    else:  # volatility
        return f"{val*100:.1f}%"

# Function to create matrix and handle duplicates
def create_time_matrix(data_dict, time_labels, metric):
    # Create a dict of dicts to handle potential duplicates
    matrix_data = {}
    
    for pair_name, df in data_dict.items():
        matrix_data[pair_name] = {}
        
        # Group by time_label and take the average for any duplicates
        grouped = df.groupby('time_label')[metric].mean()
        
        # Fill the matrix with values
        for time_label in time_labels:
            if time_label in grouped.index:
                matrix_data[pair_name][time_label] = grouped[time_label]
            else:
                matrix_data[pair_name][time_label] = None
    
    return matrix_data

if pair_results:
    # Create a list of all time labels we have data for
    all_time_labels = set()
    for pair, df in pair_results.items():
        all_time_labels.update(df['time_label'].tolist())
    
    # Find which time labels match our standard blocks
    ordered_times = []
    for t in time_block_labels:
        if t in all_time_labels:
            ordered_times.append(t)
    
    # If no matches found, use the existing times
    if not ordered_times:
        ordered_times = sorted(list(all_time_labels), reverse=True)
    
    # With tab 1 (Edge)
    with tab1:
        st.markdown("## Edge Matrix (10min timeframe, Last 24 hours, Singapore Time)")
        st.markdown("### Edge = PNL / Total Open Collateral")
        
        # Create edge matrix
        edge_matrix = create_time_matrix(pair_results, ordered_times, 'edge')
        
        # Create visualization
        fig = go.Figure()
        
        # Add each pair to the matrix
        for pair_name, time_data in edge_matrix.items():
            # Extract values in the correct order
            edge_values = [time_data.get(t) for t in ordered_times]
            
            # Add trace for hover
            fig.add_trace(go.Heatmap(
                z=[edge_values],
                x=ordered_times,
                y=[pair_name],
                colorscale=[[0, 'rgba(255,255,255,0)']],  # Transparent
                showscale=False,
                hoverinfo='text',
                text=[[format_value(v) for v in edge_values]],
                hovertemplate='<b>%{y}</b><br>Time: %{x}<br>Edge: %{text}<extra></extra>'
            ))
            
            # Add colored rectangles and text
            for i, val in enumerate(edge_values):
                if not pd.isna(val) and val is not None:
                    # Add colored rectangle
                    fig.add_shape(
                        type="rect",
                        x0=i-0.5, x1=i+0.5,
                        y0=-0.5, y1=0.5,
                        xref="x", yref=f"y{len(fig.data)}",
                        fillcolor=edge_color(val),
                        line=dict(width=1, color='white'),
                    )
                    
                    # Add text annotation
                    fig.add_annotation(
                        x=i, y=0,
                        text=format_value(val),
                        showarrow=False,
                        font=dict(
                            color='black' if abs(val) < 0.05 else 'white',
                            size=10
                        ),
                        xref="x", yref=f"y{len(fig.data)}",
                    )
        
        # Update layout
        fig.update_layout(
            title="Edge Matrix (10min intervals, Last 24 hours)",
            xaxis=dict(
                title="Time (Singapore)",
                tickangle=45,
                side="top"
            ),
            yaxis=dict(
                title="Trading Pair",
                autorange="reversed"
            ),
            height=max(500, len(pair_results) * 40),
            margin=dict(t=50, l=120, r=20, b=50),
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Legend
        st.markdown("**Edge Legend:** <span style='color:red'>Negative</span> | <span style='color:yellow'>Neutral</span> | <span style='color:green'>Positive</span>", unsafe_allow_html=True)
    
    # With tab 2 (Volatility)
    with tab2:
        st.markdown("## Volatility Matrix (10min timeframe, Last 24 hours, Singapore Time)")
        st.markdown("### Annualized Volatility = StdDev(Log Returns) * sqrt(trading periods per year)")
        
        # Create volatility matrix
        vol_matrix = create_time_matrix(pair_results, ordered_times, 'volatility')
        
        # Create visualization
        fig = go.Figure()
        
        # Add each pair to the matrix
        for pair_name, time_data in vol_matrix.items():
            # Extract values in the correct order
            vol_values = [time_data.get(t) for t in ordered_times]
            
            # Add trace for hover
            fig.add_trace(go.Heatmap(
                z=[vol_values],
                x=ordered_times,
                y=[pair_name],
                colorscale=[[0, 'rgba(255,255,255,0)']],  # Transparent
                showscale=False,
                hoverinfo='text',
                text=[[format_value(v, 'volatility') for v in vol_values]],
                hovertemplate='<b>%{y}</b><br>Time: %{x}<br>Volatility: %{text}<extra></extra>'
            ))
            
            # Add colored rectangles and text
            for i, val in enumerate(vol_values):
                if not pd.isna(val) and val is not None:
                    # Add colored rectangle
                    fig.add_shape(
                        type="rect",
                        x0=i-0.5, x1=i+0.5,
                        y0=-0.5, y1=0.5,
                        xref="x", yref=f"y{len(fig.data)}",
                        fillcolor=vol_color(val),
                        line=dict(width=1, color='white'),
                    )
                    
                    # Add text annotation
                    fig.add_annotation(
                        x=i, y=0,
                        text=format_value(val, 'volatility'),
                        showarrow=False,
                        font=dict(
                            color='black' if val < 0.6 else 'white',
                            size=10
                        ),
                        xref="x", yref=f"y{len(fig.data)}",
                    )
        
        # Update layout
        fig.update_layout(
            title="Volatility Matrix (10min intervals, Last 24 hours)",
            xaxis=dict(
                title="Time (Singapore)",
                tickangle=45,
                side="top"
            ),
            yaxis=dict(
                title="Trading Pair",
                autorange="reversed"
            ),
            height=max(500, len(pair_results) * 40),
            margin=dict(t=50, l=120, r=20, b=50),
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Legend
        st.markdown("**Volatility Legend:** <span style='color:green'>Low</span> | <span style='color:yellow'>Medium</span> | <span style='color:orange'>High</span> | <span style='color:red'>Extreme</span>", unsafe_allow_html=True)
    
    # With tab 3 (Combined View)
    with tab3:
        st.markdown("## Combined Edge & Volatility Matrix (10min intervals, Last 24 hours)")
        st.markdown("### Each cell shows: Edge (top) and Volatility (bottom)")
        
        # Create visualization
        fig = go.Figure()
        
        # Add each pair to the matrix
        for pair_name in pair_results.keys():
            edge_data = edge_matrix[pair_name]
            vol_data = vol_matrix[pair_name]
            
            # Extract values in the correct order
            edge_values = [edge_data.get(t) for t in ordered_times]
            vol_values = [vol_data.get(t) for t in ordered_times]
            
            # Add placeholder trace
            fig.add_trace(go.Heatmap(
                z=[np.zeros(len(ordered_times))],
                x=ordered_times,
                y=[pair_name],
                colorscale=[[0, 'rgba(255,255,255,0)']],  # Transparent
                showscale=False,
                hoverinfo='none'
            ))
            
            # Add split colored rectangles for each cell
            for i in range(len(ordered_times)):
                edge_val = edge_values[i]
                vol_val = vol_values[i]
                
                # Top half - Edge
                if not pd.isna(edge_val) and edge_val is not None:
                    fig.add_shape(
                        type="rect",
                        x0=i-0.5, x1=i+0.5,
                        y0=0, y1=0.5,  # Top half
                        xref="x", yref=f"y{len(fig.data)}",
                        fillcolor=edge_color(edge_val),
                        line=dict(width=1, color='white'),
                    )
                    
                    # Add edge text
                    fig.add_annotation(
                        x=i, y=0.25,  # Top quarter
                        text=format_value(edge_val),
                        showarrow=False,
                        font=dict(
                            color='black' if abs(edge_val) < 0.05 else 'white',
                            size=9
                        ),
                        xref="x", yref=f"y{len(fig.data)}",
                    )
                
                # Bottom half - Volatility
                if not pd.isna(vol_val) and vol_val is not None:
                    fig.add_shape(
                        type="rect",
                        x0=i-0.5, x1=i+0.5,
                        y0=-0.5, y1=0,  # Bottom half
                        xref="x", yref=f"y{len(fig.data)}",
                        fillcolor=vol_color(vol_val),
                        line=dict(width=1, color='white'),
                    )
                    
                    # Add volatility text
                    fig.add_annotation(
                        x=i, y=-0.25,  # Bottom quarter
                        text=format_value(vol_val, 'volatility'),
                        showarrow=False,
                        font=dict(
                            color='black' if vol_val < 0.6 else 'white',
                            size=9
                        ),
                        xref="x", yref=f"y{len(fig.data)}",
                    )
                
                # Add separator line
                if (not pd.isna(edge_val) and edge_val is not None) or (not pd.isna(vol_val) and vol_val is not None):
                    fig.add_shape(
                        type="line",
                        x0=i-0.5, x1=i+0.5,
                        y0=0, y1=0,  # Middle
                        xref="x", yref=f"y{len(fig.data)}",
                        line=dict(width=1, color='white'),
                    )
            
            # Add hover information
            hover_texts = []
            for i in range(len(ordered_times)):
                edge_val = edge_values[i]
                vol_val = vol_values[i]
                edge_text = format_value(edge_val) if not pd.isna(edge_val) and edge_val is not None else '-'
                vol_text = format_value(vol_val, 'volatility') if not pd.isna(vol_val) and vol_val is not None else '-'
                hover_texts.append(f"Edge: {edge_text}<br>Vol: {vol_text}")
            
            # Add invisible trace for hover text
            fig.add_trace(go.Scatter(
                x=ordered_times,
                y=[0] * len(ordered_times),
                mode='markers',
                marker=dict(size=0, color='rgba(0,0,0,0)'),
                hoverinfo='text',
                text=hover_texts,
                hovertemplate=f'<b>{pair_name}</b><br>Time: %{{x}}<br>%{{text}}<extra></extra>',
                showlegend=False
            ))
        
        # Update layout
        fig.update_layout(
            title="Combined Edge & Volatility Matrix (10min intervals, Last 24 hours)",
            xaxis=dict(
                title="Time (Singapore)",
                tickangle=45,
                side="top"
            ),
            yaxis=dict(
                title="Trading Pair",
                autorange="reversed"
            ),
            height=max(500, len(pair_results) * 40),
            margin=dict(t=50, l=120, r=20, b=50),
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Legend
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Edge (Top):** <span style='color:red'>Negative</span> | <span style='color:yellow'>Neutral</span> | <span style='color:green'>Positive</span>", unsafe_allow_html=True)
        with col2:
            st.markdown("**Volatility (Bottom):** <span style='color:green'>Low</span> | <span style='color:yellow'>Medium</span> | <span style='color:orange'>High</span> | <span style='color:red'>Extreme</span>", unsafe_allow_html=True)

else:
    st.warning("No data available for the selected pairs. Try selecting different pairs or refreshing the data.")