import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import pytz

# Page configuration
st.set_page_config(page_title="Edge & Volatility Matrix", layout="wide")

# --- UI Setup ---
st.title("Edge & Volatility Matrix (10min slots, 24 hours)")

# Time parameters
singapore_timezone = pytz.timezone('Asia/Singapore')
now_utc = datetime.now(pytz.utc)
now_sg = now_utc.astimezone(singapore_timezone)
start_time_sg = now_sg - timedelta(days=1)  # 24 hours back

# --- DB CONFIG ---
@st.cache_resource
def init_connection():
    try:
        db_config = st.secrets["database"]
        db_uri = f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        return create_engine(db_uri)
    except Exception as e:
        st.error(f"Database connection error: {e}")
        # Use your fallback credentials here if needed
        db_uri = "postgresql+psycopg2://username:password@host:port/database"
        return create_engine(db_uri)

engine = init_connection()

# Function to fetch pairs
@st.cache_data(ttl=600)
def fetch_pairs():
    query = "SELECT DISTINCT pair_name FROM public.trade_fill_fresh ORDER BY pair_name"
    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        return df['pair_name'].tolist() if not df.empty else []
    except Exception as e:
        st.error(f"Error fetching pairs: {e}")
        return []

# Generate time blocks
def generate_time_blocks(current_time):
    minute = current_time.minute
    rounded_minute = (minute // 10) * 10
    latest_block_end = current_time.replace(minute=rounded_minute, second=0, microsecond=0)
    
    blocks = []
    for i in range(144):  # 24 hours of 10-minute blocks
        block_end = latest_block_end - timedelta(minutes=i*10)
        block_start = block_end - timedelta(minutes=10)
        block_label = f"{block_start.strftime('%H:%M')}"
        blocks.append((block_start, block_end, block_label))
    
    return blocks

time_blocks = generate_time_blocks(now_sg)
time_labels = [block[2] for block in time_blocks]

# UI Controls
pairs = fetch_pairs()
col1, col2 = st.columns([3, 1])

with col1:
    select_all = st.checkbox("Select All Pairs", value=False)
    if select_all:
        selected_pairs = pairs
    else:
        default_pairs = pairs[:5] if len(pairs) >= 5 else pairs
        selected_pairs = st.multiselect("Select Pairs", pairs, default=default_pairs)

with col2:
    if st.button("Refresh Data", key="refresh"):
        st.cache_data.clear()
        st.rerun()

if not selected_pairs:
    st.warning("Please select at least one pair")
    st.stop()

# Function to calculate volatility
def calculate_volatility(prices):
    if prices is None or len(prices) < 2:
        return None
    
    try:
        # Handle PostgreSQL array formats
        if isinstance(prices, str):
            prices = prices.strip('{}').split(',')
            prices = [float(p) for p in prices if p]
        else:
            prices = [float(p) for p in prices if p]
        
        if len(prices) < 2:
            return None
            
        # Convert to numpy array
        prices = np.array(prices)
        
        # Calculate log returns
        log_returns = np.diff(np.log(prices))
        
        # Calculate standard deviation of returns and annualize
        std_dev = np.std(log_returns)
        annualized_vol = std_dev * np.sqrt(6 * 24 * 365)  # 10-min blocks
        
        return annualized_vol
    except Exception as e:
        print(f"Volatility calculation error: {e}")
        return None

# Function to fetch edge and volatility data
@st.cache_data(ttl=600)
def fetch_data(pair_name):
    # Convert to UTC for query
    start_time_utc = start_time_sg.astimezone(pytz.utc)
    end_time_utc = now_sg.astimezone(pytz.utc)

    # Debug information
    st.sidebar.write(f"Fetching data for {pair_name}")
    st.sidebar.write(f"Time range: {start_time_utc} to {end_time_utc}")

    query = f"""
    WITH time_blocks AS (
      SELECT
        generate_series(
          date_trunc('hour', '{start_time_utc}'::timestamp) + 
          INTERVAL '10 min' * floor(EXTRACT(MINUTE FROM '{start_time_utc}'::timestamp) / 10),
          '{end_time_utc}'::timestamp,
          INTERVAL '10 minutes'
        ) AS block_start
    ),
    pnl_data AS (
      SELECT
        date_trunc('hour', created_at) + 
        INTERVAL '10 min' * floor(EXTRACT(MINUTE FROM created_at) / 10) AS time_slot,
        (
          SELECT SUM(CombinedResult) FROM (
            -- Order PNL
            SELECT SUM(taker_pnl * collateral_price) AS CombinedResult
            FROM public.trade_fill_fresh inner_t
            WHERE inner_t.created_at BETWEEN 
                  date_trunc('hour', t.created_at) + INTERVAL '10 min' * floor(EXTRACT(MINUTE FROM t.created_at) / 10) 
                  AND date_trunc('hour', t.created_at) + INTERVAL '10 min' * floor(EXTRACT(MINUTE FROM t.created_at) / 10) + INTERVAL '10 minutes'
              AND inner_t.pair_name = '{pair_name}'
              AND inner_t.taker_way IN (1, 2, 3, 4)

            UNION ALL

            -- Fee revenue (negative because it's income for platform)
            SELECT SUM(-1 * taker_fee * collateral_price) AS CombinedResult
            FROM public.trade_fill_fresh inner_t
            WHERE inner_t.created_at BETWEEN 
                  date_trunc('hour', t.created_at) + INTERVAL '10 min' * floor(EXTRACT(MINUTE FROM t.created_at) / 10) 
                  AND date_trunc('hour', t.created_at) + INTERVAL '10 min' * floor(EXTRACT(MINUTE FROM t.created_at) / 10) + INTERVAL '10 minutes'
              AND inner_t.pair_name = '{pair_name}'
              AND inner_t.taker_fee_mode = 1 
              AND inner_t.taker_way IN (1, 3)

            UNION ALL

            -- Funding fees
            SELECT SUM(funding_fee * collateral_price) AS CombinedResult
            FROM public.trade_fill_fresh inner_t
            WHERE inner_t.created_at BETWEEN 
                  date_trunc('hour', t.created_at) + INTERVAL '10 min' * floor(EXTRACT(MINUTE FROM t.created_at) / 10) 
                  AND date_trunc('hour', t.created_at) + INTERVAL '10 min' * floor(EXTRACT(MINUTE FROM t.created_at) / 10) + INTERVAL '10 minutes'
              AND inner_t.pair_name = '{pair_name}'
              AND inner_t.taker_way = 0

            UNION ALL

            -- Stop loss fees
            SELECT SUM(-taker_sl_fee * collateral_price - maker_sl_fee) AS CombinedResult
            FROM public.trade_fill_fresh inner_t
            WHERE inner_t.created_at BETWEEN 
                  date_trunc('hour', t.created_at) + INTERVAL '10 min' * floor(EXTRACT(MINUTE FROM t.created_at) / 10) 
                  AND date_trunc('hour', t.created_at) + INTERVAL '10 min' * floor(EXTRACT(MINUTE FROM t.created_at) / 10) + INTERVAL '10 minutes'
              AND inner_t.pair_name = '{pair_name}'
          ) AS CombinedResults
        ) AS total_pnl,
        
        -- Collateral calculation as per your colleague's query
        SUM(
          CASE 
            WHEN taker_fee_mode = 2 AND taker_way IN (1, 3) 
            THEN deal_vol * collateral_price 
            ELSE 0 
          END
        ) AS total_collateral,
        
        array_agg(deal_price ORDER BY created_at) AS price_array
      FROM
        public.trade_fill_fresh t
      WHERE
        created_at BETWEEN '{start_time_utc}' AND '{end_time_utc}'
        AND pair_name = '{pair_name}'
      GROUP BY
        time_slot
    )
    SELECT
      tb.block_start AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore' AS timestamp,
      COALESCE(pd.total_pnl, 0) AS total_pnl,
      COALESCE(pd.total_collateral, 0) AS total_collateral,
      pd.price_array
    FROM
      time_blocks tb
    LEFT JOIN
      pnl_data pd ON tb.block_start = pd.time_slot
    ORDER BY
      tb.block_start
    """
    
    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        
        if df.empty:
            st.sidebar.write(f"No data found for {pair_name}")
            return None
        
        # Debug count
        st.sidebar.write(f"Retrieved {len(df)} rows for {pair_name}")
        
        # Process data
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['time_label'] = df['timestamp'].dt.strftime('%H:%M')
        
        # Calculate edge directly using total_pnl and total_collateral
        df['edge'] = np.where(df['total_collateral'] > 0, df['total_pnl'] / df['total_collateral'], None)
        df['volatility'] = df['price_array'].apply(calculate_volatility)
        
        # Show data summary
        non_null_edges = df['edge'].count()
        non_null_vols = df['volatility'].count()
        st.sidebar.write(f"Non-null edges: {non_null_edges}, Non-null volatilities: {non_null_vols}")
        
        return df
    except Exception as e:
        st.sidebar.error(f"Error fetching data for {pair_name}: {str(e)}")
        return None

# Progress indicator
progress = st.progress(0)
status = st.empty()

# Fetch data for all selected pairs
results = {}
for i, pair in enumerate(selected_pairs):
    progress.progress((i) / len(selected_pairs))
    status.text(f"Processing {pair} ({i+1}/{len(selected_pairs)})")
    
    data = fetch_data(pair)
    if data is not None:
        results[pair] = data

progress.progress(1.0)
status.text(f"Processed {len(results)}/{len(selected_pairs)} pairs")

if not results:
    st.error("No data available for selected pairs")
    st.stop()

# Color functions
def edge_color(value):
    if pd.isna(value) or value is None:
        return '#f5f5f5'  # Light gray
        
    if value < -0.1:
        return 'rgba(180, 0, 0, 0.9)'  # Deep red
    elif value < -0.05:
        return 'rgba(255, 0, 0, 0.9)'  # Red
    elif value < -0.01:
        return 'rgba(255, 150, 150, 0.9)'  # Light red
    elif value < 0.01:
        return 'rgba(255, 255, 150, 0.9)'  # Yellow
    elif value < 0.05:
        return 'rgba(150, 255, 150, 0.9)'  # Light green
    elif value < 0.1:
        return 'rgba(0, 255, 0, 0.9)'  # Green
    else:
        return 'rgba(0, 180, 0, 0.9)'  # Deep green

def vol_color(value):
    if pd.isna(value) or value is None:
        return '#f5f5f5'  # Light gray
        
    if value < 0.3:
        return 'rgba(0, 255, 0, 0.9)'  # Green
    elif value < 0.6:
        return 'rgba(255, 255, 0, 0.9)'  # Yellow
    elif value < 1.0:
        return 'rgba(255, 165, 0, 0.9)'  # Orange
    else:
        return 'rgba(255, 0, 0, 0.9)'  # Red

def format_value(value, type='edge'):
    if pd.isna(value) or value is None:
        return '-'
    
    if type == 'edge':
        return f"{value*100:.2f}%"
    else:  # volatility
        return f"{value*100:.1f}%"

# Prepare time grid
def prepare_time_grid(results):
    # Create a unique list of all time_labels from all pairs
    all_times = set()
    for pair_df in results.values():
        all_times.update(pair_df['time_label'].tolist())
    
    # Get intersection with our standard time blocks
    ordered_times = []
    for t in time_labels:
        if t in all_times:
            ordered_times.append(t)
    
    # If no matches, just use the available times
    if not ordered_times:
        ordered_times = sorted(list(all_times), reverse=True)
    
    return ordered_times

# Create a grid of pairs and times
def create_matrix_data(results, ordered_times, metric='edge'):
    # Create matrix [pair x time] with the metric values
    matrix_data = {}
    
    for pair, df in results.items():
        matrix_data[pair] = {}
        
        # Initialize all time slots to None
        for time_label in ordered_times:
            matrix_data[pair][time_label] = None
        
        # Fill in the data we have
        for _, row in df.iterrows():
            time_label = row['time_label']
            if time_label in ordered_times:
                matrix_data[pair][time_label] = row[metric]
    
    return matrix_data

# Create tabs
tab1, tab2, tab3 = st.tabs(["Edge", "Volatility", "Combined"])

# Get ordered time grid
ordered_times = prepare_time_grid(results)

# Edge tab 
with tab1:
    st.markdown("## Edge Matrix (10min timeframe, Last 24 hours, Singapore Time)")
    st.markdown("### Edge = PNL / Total Open Collateral")
    
    # Create edge data
    edge_matrix = create_matrix_data(results, ordered_times, 'edge')
    
    # Create heatmap
    fig = go.Figure()
    
    # Add data for each pair
    for i, (pair, time_data) in enumerate(edge_matrix.items()):
        # Extract values keeping ordered time
        values = [time_data[t] for t in ordered_times]
        texts = [format_value(v) for v in values]
        
        # Add trace for hover
        fig.add_trace(go.Heatmap(
            z=[values],
            x=ordered_times,
            y=[pair],
            colorscale=[[0, 'rgba(255,255,255,0)']],  # Transparent
            showscale=False,
            text=[texts],
            hovertemplate='<b>%{y}</b><br>Time: %{x}<br>Edge: %{text}<extra></extra>'
        ))
        
        # Add colored rectangles and text
        for j, val in enumerate(values):
            if not pd.isna(val):
                # Add colored rectangle
                fig.add_shape(
                    type="rect", x0=j-0.5, x1=j+0.5, y0=-0.5, y1=0.5,
                    xref=f"x", yref=f"y{len(fig.data)}",
                    fillcolor=edge_color(val),
                    line=dict(width=1, color='white'),
                )
                
                # Add text
                fig.add_annotation(
                    x=j, y=0, text=format_value(val),
                    showarrow=False,
                    font=dict(color='black' if abs(val) < 0.05 else 'white', size=10),
                    xref=f"x", yref=f"y{len(fig.data)}",
                )
    
    # Layout
    fig.update_layout(
        title="Edge Matrix (10min intervals, Last 24 hours)",
        xaxis=dict(title="Time (Singapore)", tickangle=45, side="top"),
        yaxis=dict(title="Trading Pair", autorange="reversed"),
        height=max(500, len(edge_matrix) * 30),
        margin=dict(t=50, l=120, r=20, b=50),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Simple legend
    st.markdown("**Edge Legend:** <span style='color:red'>Negative</span> | <span style='color:yellow'>Neutral</span> | <span style='color:green'>Positive</span>", unsafe_allow_html=True)

# Volatility tab
with tab2:
    st.markdown("## Volatility Matrix (10min timeframe, Last 24 hours, Singapore Time)")
    st.markdown("### Annualized Volatility = StdDev(Log Returns) * sqrt(trading periods per year)")
    
    # Create volatility data
    vol_matrix = create_matrix_data(results, ordered_times, 'volatility')
    
    # Create heatmap
    fig = go.Figure()
    
    # Add data for each pair
    for i, (pair, time_data) in enumerate(vol_matrix.items()):
        # Extract values keeping ordered time
        values = [time_data[t] for t in ordered_times]
        texts = [format_value(v, 'volatility') for v in values]
        
        # Add trace for hover
        fig.add_trace(go.Heatmap(
            z=[values],
            x=ordered_times,
            y=[pair],
            colorscale=[[0, 'rgba(255,255,255,0)']],  # Transparent
            showscale=False,
            text=[texts],
            hovertemplate='<b>%{y}</b><br>Time: %{x}<br>Volatility: %{text}<extra></extra>'
        ))
        
        # Add colored rectangles and text
        for j, val in enumerate(values):
            if not pd.isna(val):
                # Add colored rectangle
                fig.add_shape(
                    type="rect", x0=j-0.5, x1=j+0.5, y0=-0.5, y1=0.5,
                    xref=f"x", yref=f"y{len(fig.data)}",
                    fillcolor=vol_color(val),
                    line=dict(width=1, color='white'),
                )
                
                # Add text
                fig.add_annotation(
                    x=j, y=0, text=format_value(val, 'volatility'),
                    showarrow=False,
                    font=dict(color='black' if val < 0.6 else 'white', size=10),
                    xref=f"x", yref=f"y{len(fig.data)}",
                )
    
    # Layout
    fig.update_layout(
        title="Volatility Matrix (10min intervals, Last 24 hours)",
        xaxis=dict(title="Time (Singapore)", tickangle=45, side="top"),
        yaxis=dict(title="Trading Pair", autorange="reversed"),
        height=max(500, len(vol_matrix) * 30),
        margin=dict(t=50, l=120, r=20, b=50),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Simple legend
    st.markdown("**Volatility Legend:** <span style='color:green'>Low</span> | <span style='color:yellow'>Medium</span> | <span style='color:orange'>High</span> | <span style='color:red'>Extreme</span>", unsafe_allow_html=True)

# Combined tab
with tab3:
    st.markdown("## Combined Edge & Volatility Matrix (10min intervals, Last 24 hours)")
    st.markdown("### Each cell shows: Edge (top) and Volatility (bottom)")
    
    # Create data for combined visualization
    combined_matrix = {}
    for pair in results.keys():
        combined_matrix[pair] = {}
        for time_label in ordered_times:
            edge_val = edge_matrix[pair][time_label] if pair in edge_matrix and time_label in edge_matrix[pair] else None
            vol_val = vol_matrix[pair][time_label] if pair in vol_matrix and time_label in vol_matrix[pair] else None
            combined_matrix[pair][time_label] = {'edge': edge_val, 'volatility': vol_val}
    
    # Create visualization
    fig = go.Figure()
    
    # Loop through pairs
    for pair, time_data in combined_matrix.items():
        # Extract values in order
        values = [time_data[t] for t in ordered_times]
        
        # Add base trace
        fig.add_trace(go.Heatmap(
            z=[np.zeros(len(values))],
            x=ordered_times,
            y=[pair],
            colorscale=[[0, 'rgba(255,255,255,0)']],
            showscale=False,
            hoverinfo='none',
        ))
        
        # Add split rectangles
        for i, val_dict in enumerate(values):
            edge_val = val_dict.get('edge')
            vol_val = val_dict.get('volatility')
            
            # Top half - edge
            if not pd.isna(edge_val):
                fig.add_shape(
                    type="rect", x0=i-0.5, x1=i+0.5, y0=0, y1=0.5,
                    xref=f"x", yref=f"y{len(fig.data)}",
                    fillcolor=edge_color(edge_val),
                    line=dict(width=1, color='white'),
                )
                
                # Add edge text
                fig.add_annotation(
                    x=i, y=0.25, text=format_value(edge_val),
                    showarrow=False,
                    font=dict(color='black' if abs(edge_val) < 0.05 else 'white', size=9),
                    xref=f"x", yref=f"y{len(fig.data)}",
                )
            
            # Bottom half - volatility
            if not pd.isna(vol_val):
                fig.add_shape(
                    type="rect", x0=i-0.5, x1=i+0.5, y0=-0.5, y1=0,
                    xref=f"x", yref=f"y{len(fig.data)}",
                    fillcolor=vol_color(vol_val),
                    line=dict(width=1, color='white'),
                )
                
                # Add volatility text
                fig.add_annotation(
                    x=i, y=-0.25, text=format_value(vol_val, 'volatility'),
                    showarrow=False,
                    font=dict(color='black' if vol_val < 0.6 else 'white', size=9),
                    xref=f"x", yref=f"y{len(fig.data)}",
                )
            
            # Add separator line
            fig.add_shape(
                type="line", x0=i-0.5, x1=i+0.5, y0=0, y1=0,
                xref=f"x", yref=f"y{len(fig.data)}",
                line=dict(width=1, color='white'),
            )
            
        # Add hover data
        hover_texts = []
        for val_dict in values:
            edge_val = val_dict.get('edge')
            vol_val = val_dict.get('volatility')
            edge_text = format_value(edge_val) if not pd.isna(edge_val) else '-'
            vol_text = format_value(vol_val, 'volatility') if not pd.isna(vol_val) else '-'
            hover_texts.append(f"Edge: {edge_text}<br>Vol: {vol_text}")
        
        # Add invisible trace for hover (corrected)
        fig.add_trace(go.Scatter(
            x=ordered_times,
            y=[0] * len(ordered_times),
            mode='markers',
            marker=dict(size=0, color='rgba(0,0,0,0)'),
            hoverinfo='text',
            text=hover_texts,
            hovertemplate=f'<b>{pair}</b><br>Time: %{{x}}<br>%{{text}}<extra></extra>',
            showlegend=False
        ))
    
    # Layout
    fig.update_layout(
        title="Combined Edge & Volatility Matrix (10min intervals, Last 24 hours)",
        xaxis=dict(title="Time (Singapore)", tickangle=45, side="top"),
        yaxis=dict(title="Trading Pair", autorange="reversed"),
        height=max(500, len(combined_matrix) * 30),
        margin=dict(t=50, l=120, r=20, b=50),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Legend
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Edge (Top):** <span style='color:red'>Negative</span> | <span style='color:yellow'>Neutral</span> | <span style='color:green'>Positive</span>", unsafe_allow_html=True)
    with col2:
        st.markdown("**Volatility (Bottom):** <span style='color:green'>Low</span> | <span style='color:yellow'>Medium</span> | <span style='color:orange'>High</span> | <span style='color:red'>Extreme</span>", unsafe_allow_html=True)