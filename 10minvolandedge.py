import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import pytz
import time

# Page config
st.set_page_config(page_title="Edge & Volatility Matrix", layout="wide")
st.title("Edge & Volatility Matrix (10min slots, 24 hours)")

# DB Connection
@st.cache_resource
def init_connection():
    try:
        db_config = st.secrets["database"]
        db_uri = f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        return create_engine(db_uri)
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return None

engine = init_connection()
if not engine:
    st.error("Failed to connect to database")
    st.stop()

# Create time slots for the past 24 hours
sg_tz = pytz.timezone('Asia/Singapore')
now_sg = datetime.now(sg_tz)
start_time_sg = now_sg - timedelta(days=1)

# Generate 10-minute time slots
time_slots = []
current_slot = start_time_sg.replace(minute=(start_time_sg.minute//10)*10, second=0, microsecond=0)
while current_slot < now_sg:
    time_slots.append(current_slot)
    current_slot += timedelta(minutes=10)

# Get pairs
@st.cache_data(ttl=600)
def get_pairs():
    try:
        query = "SELECT DISTINCT pair_name FROM public.trade_fill_fresh ORDER BY pair_name"
        return pd.read_sql(query, engine)['pair_name'].tolist()
    except Exception as e:
        st.error(f"Error getting pairs: {e}")
        return []

pairs = get_pairs()

# UI controls
col1, col2 = st.columns([3, 1])
with col1:
    select_all = st.checkbox("Select All Pairs", value=False)
    if select_all:
        selected_pairs = pairs
    else:
        default_pairs = pairs[:5] if len(pairs) >= 5 else pairs
        selected_pairs = st.multiselect("Select Pairs", pairs, default=default_pairs)
        
with col2:
    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.experimental_rerun()

if not selected_pairs:
    st.warning("Please select at least one pair")
    st.stop()

# Fetch data for each pair
@st.cache_data(ttl=600)
def get_data_for_pair(pair_name):
    """Get edge and volatility for a pair"""
    try:
        # Convert to UTC for database query
        start_time_utc = start_time_sg.astimezone(pytz.utc)
        end_time_utc = now_sg.astimezone(pytz.utc)
        
        # Simple query that directly fetches trade data for 10-minute slots
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
        
        trade_data AS (
            SELECT
                date_trunc('hour', created_at) + 
                INTERVAL '10 min' * floor(EXTRACT(MINUTE FROM created_at) / 10) AS time_slot,
                
                -- Order PNL
                SUM(taker_pnl * collateral_price) AS order_pnl,
                
                -- Fee revenue (negative because it's income for platform)
                SUM(CASE WHEN taker_fee_mode = 1 AND taker_way IN (1, 3) 
                    THEN -1 * taker_fee * collateral_price ELSE 0 END) AS fee_pnl,
                
                -- Funding fees
                SUM(CASE WHEN taker_way = 0 
                    THEN funding_fee * collateral_price ELSE 0 END) AS funding_pnl,
                
                -- Stop loss fees
                SUM(-taker_sl_fee * collateral_price - maker_sl_fee) AS sl_pnl,
                
                -- Open collateral
                SUM(CASE WHEN taker_fee_mode = 2 AND taker_way IN (1, 3) 
                    THEN deal_vol * collateral_price ELSE 0 END) AS collateral,
                
                -- Deal prices for volatility calc
                array_agg(deal_price ORDER BY created_at) AS prices
            FROM
                public.trade_fill_fresh
            WHERE
                created_at BETWEEN '{start_time_utc}' AND '{end_time_utc}'
                AND pair_name = '{pair_name}'
            GROUP BY
                time_slot
        )
        
        SELECT
            tb.block_start AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore' AS time_sg,
            td.order_pnl,
            td.fee_pnl,
            td.funding_pnl,
            td.sl_pnl,
            td.collateral,
            td.prices
        FROM
            time_blocks tb
        LEFT JOIN
            trade_data td ON tb.block_start = td.time_slot
        ORDER BY
            tb.block_start
        """
        
        df = pd.read_sql(query, engine)
        
        if df.empty:
            return pd.DataFrame()
        
        # Calculate total PNL and edge
        df['total_pnl'] = df['order_pnl'].fillna(0) + df['fee_pnl'].fillna(0) + \
                          df['funding_pnl'].fillna(0) + df['sl_pnl'].fillna(0)
                          
        df['edge'] = np.where(df['collateral'] > 0, 
                              df['total_pnl'] / df['collateral'], 
                              None)
        
        # Calculate volatility from price data
        def calc_volatility(prices):
            if prices is None or not isinstance(prices, list) or len(prices) < 2:
                return None
            try:
                # Convert prices to numeric
                if isinstance(prices, str):  # Handle PostgreSQL array format
                    prices = prices.strip('{}').split(',')
                prices = [float(p) for p in prices if p and p != 'NULL' and p != 'None']
                
                if len(prices) < 2:
                    return None
                
                # Calculate log returns and volatility
                prices_array = np.array(prices)
                log_returns = np.diff(np.log(prices_array))
                std_dev = np.std(log_returns)
                
                # Annualize (10-min intervals -> 144 intervals per day * 365 days)
                volatility = std_dev * np.sqrt(144 * 365)
                
                return volatility
            except Exception as e:
                print(f"Volatility calculation error: {e}")
                return None
        
        # Apply volatility calculation
        df['volatility'] = df['prices'].apply(calc_volatility)
        
        # Format time and create time_label
        df['time_sg'] = pd.to_datetime(df['time_sg'])
        df['time_label'] = df['time_sg'].dt.strftime('%H:%M')
        
        return df
    
    except Exception as e:
        st.error(f"Error fetching data for {pair_name}: {e}")
        return pd.DataFrame()

# Progress bar
progress_bar = st.progress(0)
status_text = st.empty()

# Fetch data for all pairs
all_data = {}
for i, pair in enumerate(selected_pairs):
    progress_bar.progress(i / len(selected_pairs))
    status_text.text(f"Processing {pair} ({i+1}/{len(selected_pairs)})")
    
    df = get_data_for_pair(pair)
    if not df.empty:
        all_data[pair] = df

progress_bar.progress(1.0)
status_text.text(f"Finished processing {len(all_data)}/{len(selected_pairs)} pairs")

if not all_data:
    st.error("No data found for any selected pairs")
    st.stop()

# Create a unified time grid across all pairs
all_times = set()
for pair, df in all_data.items():
    all_times.update(df['time_label'].tolist())

time_grid = sorted(all_times)

# Create tabs
tab1, tab2, tab3 = st.tabs(["Edge", "Volatility", "Combined"])

# Helper functions for coloring
def edge_color(value):
    if pd.isna(value) or value is None:
        return '#f5f5f5'  # Light gray for empty
    
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
        return '#f5f5f5'  # Light gray for empty
    
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

# Tab 1: Edge Matrix
with tab1:
    st.markdown("## Edge Matrix (10min timeframe, Last 24 hours, Singapore Time)")
    st.markdown("### Edge = PNL / Total Open Collateral")
    
    # Create edge matrix
    edge_matrix = go.Figure()
    
    # Add edge data for each pair
    for pair in selected_pairs:
        if pair in all_data:
            df = all_data[pair]
            if df.empty:
                continue
                
            # Create a map of time_label -> edge value
            edge_map = dict(zip(df['time_label'], df['edge']))
            
            # Create arrays for visualization
            edge_values = []
            for time_label in time_grid:
                edge_values.append(edge_map.get(time_label, None))
            
            # Add pair to matrix
            edge_matrix.add_trace(go.Heatmap(
                z=[edge_values],  # Each row is a pair
                x=time_grid,      # X-axis is time
                y=[pair],         # Y-axis is pair name
                colorscale=[[0, 'rgba(255,255,255,0)']],  # Transparent (we'll add custom colors)
                showscale=False,
                hoverinfo='text',
                text=[[format_value(val) for val in edge_values]],
                hovertemplate='<b>%{y}</b><br>Time: %{x}<br>Edge: %{text}<extra></extra>'
            ))
            
            # Add custom colored rectangles and text
            for i, val in enumerate(edge_values):
                if not pd.isna(val) and val is not None:
                    # Add colored rectangle
                    edge_matrix.add_shape(
                        type="rect",
                        x0=i-0.5, x1=i+0.5,
                        y0=-0.5, y1=0.5,
                        xref="x", yref=f"y{len(edge_matrix.data)}",
                        fillcolor=edge_color(val),
                        line=dict(width=1, color='white'),
                    )
                    
                    # Add text label
                    edge_matrix.add_annotation(
                        x=i, y=0,
                        text=format_value(val),
                        showarrow=False,
                        font=dict(
                            color='black' if abs(val) < 0.05 else 'white',
                            size=10
                        ),
                        xref="x", yref=f"y{len(edge_matrix.data)}",
                    )
    
    # Update layout
    edge_matrix.update_layout(
        title="Edge Matrix (10min intervals)",
        xaxis=dict(
            title="Time (Singapore)",
            tickangle=45,
            side="top"
        ),
        yaxis=dict(
            title="Trading Pair",
            autorange="reversed"
        ),
        height=max(500, len(selected_pairs) * 40),
        margin=dict(t=50, l=120, r=20, b=50),
    )
    
    st.plotly_chart(edge_matrix, use_container_width=True)
    
    # Legend
    st.markdown("**Edge Legend:** <span style='color:red'>Negative</span> | <span style='color:yellow'>Neutral</span> | <span style='color:green'>Positive</span>", unsafe_allow_html=True)

# Tab 2: Volatility Matrix
with tab2:
    st.markdown("## Volatility Matrix (10min timeframe, Last 24 hours, Singapore Time)")
    st.markdown("### Annualized Volatility = StdDev(Log Returns) * sqrt(trading periods per year)")
    
    # Create volatility matrix
    vol_matrix = go.Figure()
    
    # Add volatility data for each pair
    for pair in selected_pairs:
        if pair in all_data:
            df = all_data[pair]
            if df.empty:
                continue
                
            # Create a map of time_label -> volatility value
            vol_map = dict(zip(df['time_label'], df['volatility']))
            
            # Create arrays for visualization
            vol_values = []
            for time_label in time_grid:
                vol_values.append(vol_map.get(time_label, None))
            
            # Add pair to matrix
            vol_matrix.add_trace(go.Heatmap(
                z=[vol_values],   # Each row is a pair
                x=time_grid,      # X-axis is time
                y=[pair],         # Y-axis is pair name
                colorscale=[[0, 'rgba(255,255,255,0)']],  # Transparent (we'll add custom colors)
                showscale=False,
                hoverinfo='text',
                text=[[format_value(val, 'volatility') for val in vol_values]],
                hovertemplate='<b>%{y}</b><br>Time: %{x}<br>Volatility: %{text}<extra></extra>'
            ))
            
            # Add custom colored rectangles and text
            for i, val in enumerate(vol_values):
                if not pd.isna(val) and val is not None:
                    # Add colored rectangle
                    vol_matrix.add_shape(
                        type="rect",
                        x0=i-0.5, x1=i+0.5,
                        y0=-0.5, y1=0.5,
                        xref="x", yref=f"y{len(vol_matrix.data)}",
                        fillcolor=vol_color(val),
                        line=dict(width=1, color='white'),
                    )
                    
                    # Add text label
                    vol_matrix.add_annotation(
                        x=i, y=0,
                        text=format_value(val, 'volatility'),
                        showarrow=False,
                        font=dict(
                            color='black' if val < 0.6 else 'white',
                            size=10
                        ),
                        xref="x", yref=f"y{len(vol_matrix.data)}",
                    )
    
    # Update layout
    vol_matrix.update_layout(
        title="Volatility Matrix (10min intervals)",
        xaxis=dict(
            title="Time (Singapore)",
            tickangle=45,
            side="top"
        ),
        yaxis=dict(
            title="Trading Pair",
            autorange="reversed"
        ),
        height=max(500, len(selected_pairs) * 40),
        margin=dict(t=50, l=120, r=20, b=50),
    )
    
    st.plotly_chart(vol_matrix, use_container_width=True)
    
    # Legend
    st.markdown("**Volatility Legend:** <span style='color:green'>Low</span> | <span style='color:yellow'>Medium</span> | <span style='color:orange'>High</span> | <span style='color:red'>Extreme</span>", unsafe_allow_html=True)

# Tab 3: Combined View
with tab3:
    st.markdown("## Combined Edge & Volatility Matrix (10min intervals, Last 24 hours)")
    st.markdown("### Each cell shows: Edge (top) and Volatility (bottom)")
    
    # Create combined matrix
    combined_matrix = go.Figure()
    
    # Add data for each pair
    for pair in selected_pairs:
        if pair in all_data:
            df = all_data[pair]
            if df.empty:
                continue
                
            # Create maps of time_label -> value
            edge_map = dict(zip(df['time_label'], df['edge']))
            vol_map = dict(zip(df['time_label'], df['volatility']))
            
            # Create arrays for visualization
            combined_values = []
            for time_label in time_grid:
                combined_values.append({
                    'edge': edge_map.get(time_label, None),
                    'vol': vol_map.get(time_label, None)
                })
            
            # Add placeholder for this pair
            combined_matrix.add_trace(go.Heatmap(
                z=[np.zeros(len(time_grid))],  # Just placeholder values
                x=time_grid,
                y=[pair],
                colorscale=[[0, 'rgba(255,255,255,0)']],  # Transparent
                showscale=False,
                hoverinfo='none'
            ))
            
            # Add custom rectangles for each cell
            for i, val_dict in enumerate(combined_values):
                edge_val = val_dict['edge']
                vol_val = val_dict['vol']
                
                # Top half: Edge
                if not pd.isna(edge_val) and edge_val is not None:
                    combined_matrix.add_shape(
                        type="rect",
                        x0=i-0.5, x1=i+0.5,
                        y0=0, y1=0.5,  # Top half
                        xref="x", yref=f"y{len(combined_matrix.data)}",
                        fillcolor=edge_color(edge_val),
                        line=dict(width=1, color='white'),
                    )
                    
                    # Edge value text
                    combined_matrix.add_annotation(
                        x=i, y=0.25,  # Top quarter
                        text=format_value(edge_val),
                        showarrow=False,
                        font=dict(
                            color='black' if abs(edge_val) < 0.05 else 'white',
                            size=9
                        ),
                        xref="x", yref=f"y{len(combined_matrix.data)}",
                    )
                
                # Bottom half: Volatility
                if not pd.isna(vol_val) and vol_val is not None:
                    combined_matrix.add_shape(
                        type="rect",
                        x0=i-0.5, x1=i+0.5,
                        y0=-0.5, y1=0,  # Bottom half
                        xref="x", yref=f"y{len(combined_matrix.data)}",
                        fillcolor=vol_color(vol_val),
                        line=dict(width=1, color='white'),
                    )
                    
                    # Volatility value text
                    combined_matrix.add_annotation(
                        x=i, y=-0.25,  # Bottom quarter
                        text=format_value(vol_val, 'volatility'),
                        showarrow=False,
                        font=dict(
                            color='black' if vol_val < 0.6 else 'white',
                            size=9
                        ),
                        xref="x", yref=f"y{len(combined_matrix.data)}",
                    )
                
                # Add dividing line
                if (not pd.isna(edge_val) and edge_val is not None) or (not pd.isna(vol_val) and vol_val is not None):
                    combined_matrix.add_shape(
                        type="line",
                        x0=i-0.5, x1=i+0.5,
                        y0=0, y1=0,  # Middle
                        xref="x", yref=f"y{len(combined_matrix.data)}",
                        line=dict(width=1, color='white'),
                    )
            
            # Add hover information
            hover_texts = []
            for val_dict in combined_values:
                edge_val = val_dict['edge']
                vol_val = val_dict['vol']
                edge_text = format_value(edge_val) if not pd.isna(edge_val) and edge_val is not None else '-'
                vol_text = format_value(vol_val, 'volatility') if not pd.isna(vol_val) and vol_val is not None else '-'
                hover_texts.append(f"Edge: {edge_text}<br>Vol: {vol_text}")
            
            # Add invisible hover trace
            combined_matrix.add_trace(go.Scatter(
                x=time_grid,
                y=[0] * len(time_grid),
                mode='markers',
                marker=dict(size=0, color='rgba(0,0,0,0)'),
                hoverinfo='text',
                text=hover_texts,
                hovertemplate=f'<b>{pair}</b><br>Time: %{{x}}<br>%{{text}}<extra></extra>',
                showlegend=False,
                yaxis=f"y{len(combined_matrix.data)-1}"
            ))
    
    # Update layout
    combined_matrix.update_layout(
        title="Combined Edge & Volatility Matrix (10min intervals)",
        xaxis=dict(
            title="Time (Singapore)",
            tickangle=45,
            side="top"
        ),
        yaxis=dict(
            title="Trading Pair",
            autorange="reversed"
        ),
        height=max(500, len(selected_pairs) * 40),
        margin=dict(t=50, l=120, r=20, b=50),
    )
    
    st.plotly_chart(combined_matrix, use_container_width=True)
    
    # Legend
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Edge (Top):** <span style='color:red'>Negative</span> | <span style='color:yellow'>Neutral</span> | <span style='color:green'>Positive</span>", unsafe_allow_html=True)
    with col2:
        st.markdown("**Volatility (Bottom):** <span style='color:green'>Low</span> | <span style='color:yellow'>Medium</span> | <span style='color:orange'>High</span> | <span style='color:red'>Extreme</span>", unsafe_allow_html=True)