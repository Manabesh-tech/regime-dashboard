import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import pytz
import time

# Page configuration
st.set_page_config(
    page_title="Edge & Volatility Matrix",
    page_icon="📊",
    layout="wide"
)

# --- UI Setup ---
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Edge & Volatility Matrix")
st.subheader("All Trading Pairs - Last 24 Hours (Singapore Time)")

# Define parameters for the 10-minute timeframe
timeframe = "10min"
lookback_days = 1  # 24 hours
expected_points = 144  # Expected data points per pair over 24 hours (24 hours * 6 intervals per hour)
singapore_timezone = pytz.timezone('Asia/Singapore')

# Get current time in Singapore timezone
now_utc = datetime.now(pytz.utc)
now_sg = now_utc.astimezone(singapore_timezone)
st.write(f"Current Singapore Time: {now_sg.strftime('%Y-%m-%d %H:%M:%S')}")

# --- DB CONFIG ---
@st.cache_resource
def init_connection():
    """Initialize database connection"""
    try:
        # Try to get database config from secrets
        db_config = st.secrets["database"]
        db_uri = (
            f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}"
            f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )
        return create_engine(db_uri)
    except Exception as e:
        st.error(f"Error connecting to the database: {e}")
        # Fallback to hardcoded credentials (these would need to be replaced with actual values)
        db_uri = (
            "postgresql+psycopg2://user:password@host:port/database"
        )
        return create_engine(db_uri)

# Create database connection
engine = init_connection()

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

# Fetch all available pairs from DB
@st.cache_data(ttl=600, show_spinner="Fetching pairs...")
def fetch_all_pairs():
    """Fetch all available trading pairs"""
    query = "SELECT DISTINCT pair_name FROM public.trade_fill_fresh ORDER BY pair_name"
    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        if df.empty:
            st.error("No pairs found in the database.")
            return []
        return df['pair_name'].tolist()
    except Exception as e:
        st.error(f"Error fetching pairs: {e}")
        # Return some example pairs as fallback
        return ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT", "PEPE/USDT"]

all_pairs = fetch_all_pairs()

# UI Controls
col1, col2 = st.columns([3, 1])

with col1:
    # Let user select pairs to display (or select all)
    select_all = st.checkbox("Select All Pairs", value=False)
    
    if select_all:
        selected_pairs = all_pairs
    else:
        default_pairs = all_pairs[:5] if len(all_pairs) >= 5 else all_pairs
        selected_pairs = st.multiselect(
            "Select Pairs", 
            all_pairs,
            default=default_pairs
        )

with col2:
    # Add a refresh button
    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.experimental_rerun()

if not selected_pairs:
    st.warning("Please select at least one pair")
    st.stop()

# Function to fetch data for edge and volatility calculation
@st.cache_data(ttl=600, show_spinner="Fetching data...")
def fetch_edge_and_volatility_data(pair_name):
    """Fetch data for edge and volatility calculation for a specific pair"""
    # Get current time in Singapore timezone
    now_utc = datetime.now(pytz.utc)
    now_sg = now_utc.astimezone(singapore_timezone)
    start_time_sg = now_sg - timedelta(days=lookback_days)
    
    # Convert back to UTC for database query
    start_time_utc = start_time_sg.astimezone(pytz.utc)
    end_time_utc = now_sg.astimezone(pytz.utc)

    # Query to fetch data for edge calculation (PNL and collateral)
    query = f"""
    WITH time_intervals AS (
      -- Generate 10-minute intervals for the past 24 hours in Singapore time
      SELECT
        generate_series(
          date_trunc('hour', '{start_time_utc}'::timestamp) + 
          INTERVAL '10 min' * floor(EXTRACT(MINUTE FROM '{start_time_utc}'::timestamp) / 10),
          '{end_time_utc}'::timestamp,
          INTERVAL '10 minutes'
        ) AS slot_start
    ),
    slot_data AS (
      -- Calculate PNL, collateral, and get prices for volatility calculation
      SELECT
        '{pair_name}' AS pair_name,
        date_trunc('hour', created_at AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') + 
        INTERVAL '10 min' * floor(EXTRACT(MINUTE FROM created_at AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') / 10) AS time_slot,
        SUM(taker_pnl * collateral_price) AS order_pnl,
        SUM(funding_fee * collateral_price) AS funding_pnl,
        SUM(rebate * collateral_price) AS rebate_pnl,
        SUM(CASE WHEN taker_way IN (1, 3) THEN collateral_amount * collateral_price ELSE 0 END) AS total_collateral,
        array_agg(deal_price ORDER BY created_at) AS price_array
      FROM
        trade_fill_fresh
      WHERE
        created_at BETWEEN '{start_time_utc}' AND '{end_time_utc}'
        AND pair_name = '{pair_name}'
      GROUP BY
        date_trunc('hour', created_at AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') + 
        INTERVAL '10 min' * floor(EXTRACT(MINUTE FROM created_at AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore') / 10)
    )
    SELECT
      ti.slot_start AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore' AS timestamp,
      COALESCE(s.pair_name, '{pair_name}') AS pair_name,
      COALESCE(s.order_pnl, 0) AS order_pnl,
      COALESCE(s.funding_pnl, 0) AS funding_pnl,
      COALESCE(s.rebate_pnl, 0) AS rebate_pnl,
      COALESCE(s.total_collateral, 0) AS total_collateral,
      s.price_array
    FROM
      time_intervals ti
    LEFT JOIN
      slot_data s ON ti.slot_start = s.time_slot AT TIME ZONE 'Asia/Singapore' AT TIME ZONE 'UTC'
    ORDER BY
      ti.slot_start DESC
    """
    
    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        
        if df.empty:
            return None
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Calculate total PNL
        df['total_pnl'] = df['order_pnl'] + df['funding_pnl'] + df['rebate_pnl']
        
        # Calculate edge = PNL / collateral
        df['edge'] = np.where(df['total_collateral'] > 0, 
                             df['total_pnl'] / df['total_collateral'], 
                             0)
        
        # For time slots with no trades (and thus no collateral), set edge to None
        df.loc[df['total_collateral'] == 0, 'edge'] = None
        
                    # Calculate volatility - convert price arrays to pandas series
        df['volatility'] = df.apply(lambda row: calculate_volatility(row['price_array']), axis=1)
        
        # Format time label to match our aligned blocks (HH:MM format)
        df['time_label'] = df['timestamp'].dt.strftime('%H:%M')
        
        return df
    
    except Exception as e:
        st.error(f"Error fetching data for {pair_name}: {e}")
        return None

def calculate_volatility(price_array):
    """Calculate annualized volatility from an array of prices"""
    if price_array is None or len(price_array) < 2:
        return None
    
    try:
        # Convert PostgreSQL array to Python list of floats
        if isinstance(price_array, str):
            # If returned as string (e.g., '{1,2,3}'), strip braces and split
            price_array = price_array.strip('{}').split(',')
            prices = [float(p) for p in price_array if p]
        else:
            # If returned as list
            prices = [float(p) for p in price_array if p]
        
        if len(prices) < 2:
            return None
            
        # Convert to numpy array
        prices = np.array(prices)
        
        # Calculate log returns
        log_returns = np.diff(np.log(prices))
        
        # Calculate standard deviation of returns
        std_dev = np.std(log_returns)
        
        # Annualize (10-minute intervals -> 6*24*365 intervals per year)
        annualized_vol = std_dev * np.sqrt(6 * 24 * 365)
        
        return annualized_vol
    
    except Exception as e:
        print(f"Error in volatility calculation: {e}")
        return None

# Define color scales and thresholds
def map_edge_to_color(edge_value):
    """Map edge value to a color scale"""
    if pd.isna(edge_value) or edge_value is None:
        return '#f5f5f5'  # Light gray for no data
        
    if edge_value < -0.1:  # Very negative edge (deep red)
        return f'rgba(180, 0, 0, 0.9)'
    elif edge_value < -0.05:  # Negative edge (red)
        return f'rgba(255, 0, 0, 0.9)'
    elif edge_value < -0.01:  # Slightly negative edge (light red)
        return f'rgba(255, 150, 150, 0.9)'
    elif edge_value < 0.01:  # Near zero edge (yellow)
        return f'rgba(255, 255, 150, 0.9)'
    elif edge_value < 0.05:  # Slightly positive edge (light green)
        return f'rgba(150, 255, 150, 0.9)'
    elif edge_value < 0.1:  # Positive edge (green)
        return f'rgba(0, 255, 0, 0.9)'
    else:  # Very positive edge (deep green)
        return f'rgba(0, 180, 0, 0.9)'

def map_volatility_to_color(vol_value):
    """Map volatility value to a color scale"""
    if pd.isna(vol_value) or vol_value is None:
        return '#f5f5f5'  # Light gray for no data
        
    if vol_value < 0.3:  # Low volatility (green)
        return f'rgba(0, 255, 0, 0.9)'
    elif vol_value < 0.6:  # Medium volatility (yellow)
        return f'rgba(255, 255, 0, 0.9)'
    elif vol_value < 1.0:  # High volatility (orange)
        return f'rgba(255, 165, 0, 0.9)'
    else:  # Very high volatility (red)
        return f'rgba(255, 0, 0, 0.9)'

def format_value(value, type='edge'):
    """Format edge or volatility values for display"""
    if pd.isna(value) or value is None:
        return '-'
    
    if type == 'edge':
        # Format as percentage with 2 decimal places
        return f"{value*100:.2f}%"
    else:  # volatility
        # Format as percentage with 2 decimal places
        return f"{value*100:.1f}%"

# Show progress bar while calculating
progress_bar = st.progress(0)
status_text = st.empty()

# Calculate edge and volatility for each pair
pair_results = {}
for i, pair_name in enumerate(selected_pairs):
    try:
        progress_bar.progress((i) / len(selected_pairs))
        status_text.text(f"Processing {pair_name} ({i+1}/{len(selected_pairs)})")
        
        # Fetch data for edge and volatility calculation
        df = fetch_edge_and_volatility_data(pair_name)
        
        if df is not None:
            pair_results[pair_name] = df
    except Exception as e:
        st.error(f"Error processing pair {pair_name}: {e}")
        print(f"Error processing pair {pair_name}: {e}")

# Final progress update
progress_bar.progress(1.0)
status_text.text(f"Processed {len(pair_results)}/{len(selected_pairs)} pairs successfully")

# Create heatmap matrix for edge
if pair_results:
    # Create tabs for Edge and Volatility
    tab1, tab2, tab3 = st.tabs(["Edge", "Volatility", "Combined View"])
    
    with tab1:
        st.markdown("## Edge Matrix (10min timeframe, Last 24 hours, Singapore Time)")
        st.markdown("### Edge = PNL / Total Open Collateral")
        st.markdown("### Color Legend: <span style='color:red'>Negative Edge</span>, <span style='color:yellow'>Neutral</span>, <span style='color:green'>Positive Edge</span>", unsafe_allow_html=True)
        
        # Create edge data
        edge_data = {}
        for pair_name, df in pair_results.items():
            if 'edge' in df.columns:
                edge_series = df.set_index('time_label')['edge']
                edge_data[pair_name] = edge_series
        
        # Create DataFrame with all pairs
        edge_df = pd.DataFrame(edge_data)
        
        # Apply the time blocks in the proper order (most recent first)
        available_times = set(edge_df.index)
        ordered_times = [t for t in time_block_labels if t in available_times]
        
        # If no matches are found in aligned blocks, fallback to the available times
        if not ordered_times and available_times:
            ordered_times = sorted(list(available_times), reverse=True)
        
        # Reindex with the ordered times
        edge_df = edge_df.reindex(ordered_times)
        
        # Create a plotly heatmap
        fig = go.Figure()
        
        # Create heatmap
        for pair_name in edge_df.columns:
            edge_values = edge_df[pair_name].values
            colors = [map_edge_to_color(val) for val in edge_values]
            
            # Create text for hover info
            text = [format_value(val) for val in edge_values]
            
            fig.add_trace(go.Heatmap(
                z=[edge_values],
                x=edge_df.index,
                y=[pair_name],
                colorscale=[[0, 'rgba(255,255,255,0)']], # Transparent colorscale
                showscale=False,
                text=[text],
                hoverinfo='text',
                hoverlabel=dict(bgcolor='white'),
                hovertemplate='<b>%{y}</b><br>Time: %{x}<br>Edge: %{text}<extra></extra>'
            ))
            
            # Add colored rectangles
            for i, val in enumerate(edge_values):
                if not pd.isna(val):
                    fig.add_shape(
                        type="rect",
                        x0=i-0.5,
                        x1=i+0.5,
                        y0=-0.5,
                        y1=0.5,
                        xref=f"x",
                        yref=f"y{len(fig.data)}",
                        fillcolor=map_edge_to_color(val),
                        line=dict(width=1, color='white'),
                    )
                    
                    # Add text annotation for value
                    formatted_val = format_value(val)
                    fig.add_annotation(
                        x=i,
                        y=0,
                        text=formatted_val,
                        showarrow=False,
                        font=dict(
                            color='black' if abs(val) < 0.05 else 'white',
                            size=10
                        ),
                        xref=f"x",
                        yref=f"y{len(fig.data)}",
                    )
        
        # Customize layout
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
            height=max(600, len(edge_df.columns) * 40),  # Dynamic height based on number of pairs
            margin=dict(t=50, l=120, r=20, b=50),  # Add margin for labels
            showlegend=False,
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Create a summary table
        st.markdown("## Edge Summary Statistics")
        
        # Calculate statistics
        edge_summary = {}
        for pair_name in edge_df.columns:
            values = edge_df[pair_name].dropna()
            if not values.empty:
                edge_summary[pair_name] = {
                    'Mean Edge': values.mean(),
                    'Max Edge': values.max(),
                    'Min Edge': values.min(),
                    'Median Edge': values.median(),
                    'Std Dev': values.std(),
                    'Positive Count': (values > 0).sum(),
                    'Negative Count': (values < 0).sum(),
                    'Data Points': len(values)
                }
        
        if edge_summary:
            summary_df = pd.DataFrame(edge_summary).T
            
            # Format the values
            formatted_summary = summary_df.copy()
            formatted_summary['Mean Edge'] = formatted_summary['Mean Edge'].apply(lambda x: f"{x*100:.2f}%")
            formatted_summary['Max Edge'] = formatted_summary['Max Edge'].apply(lambda x: f"{x*100:.2f}%")
            formatted_summary['Min Edge'] = formatted_summary['Min Edge'].apply(lambda x: f"{x*100:.2f}%")
            formatted_summary['Median Edge'] = formatted_summary['Median Edge'].apply(lambda x: f"{x*100:.2f}%")
            formatted_summary['Std Dev'] = formatted_summary['Std Dev'].apply(lambda x: f"{x*100:.2f}%")
            
            # Add win rate
            formatted_summary['Win Rate'] = (summary_df['Positive Count'] / summary_df['Data Points']).apply(lambda x: f"{x*100:.1f}%")
            
            # Reorder columns
            formatted_summary = formatted_summary[['Mean Edge', 'Median Edge', 'Min Edge', 'Max Edge', 'Std Dev', 'Win Rate', 'Positive Count', 'Negative Count', 'Data Points']]
            
                            # Color cells based on values
            def color_edge_cells(val, column):
                if 'Edge' in column and '%' in str(val):
                    try:
                        value = float(val.strip('%'))/100
                        if value > 0.05:
                            return 'background-color: rgba(0, 180, 0, 0.3)'
                        elif value > 0:
                            return 'background-color: rgba(150, 255, 150, 0.3)'
                        elif value > -0.05:
                            return 'background-color: rgba(255, 150, 150, 0.3)'
                        else:
                            return 'background-color: rgba(255, 0, 0, 0.3)'
                    except Exception:
                        return ''
                elif column == 'Win Rate' and '%' in str(val):
                    try:
                        value = float(val.strip('%'))/100
                        if value > 0.7:
                            return 'background-color: rgba(0, 180, 0, 0.3)'
                        elif value > 0.5:
                            return 'background-color: rgba(150, 255, 150, 0.3)'
                        elif value > 0.3:
                            return 'background-color: rgba(255, 255, 150, 0.3)'
                        else:
                            return 'background-color: rgba(255, 150, 150, 0.3)'
                    except Exception:
                        return ''
                return ''
            
            # Apply styling
            styled_summary = formatted_summary.style.apply(lambda x: [color_edge_cells(val, col) for val, col in zip(x, formatted_summary.columns)], axis=1)
            
            # Display the summary table
            st.dataframe(styled_summary, use_container_width=True)
    
    with tab2:
        st.markdown("## Volatility Matrix (10min timeframe, Last 24 hours, Singapore Time)")
        st.markdown("### Annualized Volatility = StdDev(Log Returns) * sqrt(trading periods per year)")
        st.markdown("### Color Legend: <span style='color:green'>Low Vol</span>, <span style='color:yellow'>Medium Vol</span>, <span style='color:orange'>High Vol</span>, <span style='color:red'>Extreme Vol</span>", unsafe_allow_html=True)
        
        # Create volatility data
        vol_data = {}
        for pair_name, df in pair_results.items():
            if 'volatility' in df.columns:
                vol_series = df.set_index('time_label')['volatility']
                vol_data[pair_name] = vol_series
        
        # Create DataFrame with all pairs
        vol_df = pd.DataFrame(vol_data)
        
        # Apply the time blocks in the proper order (most recent first)
        vol_df = vol_df.reindex(ordered_times)
        
        # Create a plotly heatmap for volatility
        fig = go.Figure()
        
        # Create heatmap
        for pair_name in vol_df.columns:
            vol_values = vol_df[pair_name].values
            colors = [map_volatility_to_color(val) for val in vol_values]
            
            # Create text for hover info
            text = [format_value(val, 'volatility') for val in vol_values]
            
            fig.add_trace(go.Heatmap(
                z=[vol_values],
                x=vol_df.index,
                y=[pair_name],
                colorscale=[[0, 'rgba(255,255,255,0)']], # Transparent colorscale
                showscale=False,
                text=[text],
                hoverinfo='text',
                hoverlabel=dict(bgcolor='white'),
                hovertemplate='<b>%{y}</b><br>Time: %{x}<br>Volatility: %{text}<extra></extra>'
            ))
            
            # Add colored rectangles
            for i, val in enumerate(vol_values):
                if not pd.isna(val):
                    fig.add_shape(
                        type="rect",
                        x0=i-0.5,
                        x1=i+0.5,
                        y0=-0.5,
                        y1=0.5,
                        xref=f"x",
                        yref=f"y{len(fig.data)}",
                        fillcolor=map_volatility_to_color(val),
                        line=dict(width=1, color='white'),
                    )
                    
                    # Add text annotation for value
                    formatted_val = format_value(val, 'volatility')
                    fig.add_annotation(
                        x=i,
                        y=0,
                        text=formatted_val,
                        showarrow=False,
                        font=dict(
                            color='black' if val < 0.6 else 'white',
                            size=10
                        ),
                        xref=f"x",
                        yref=f"y{len(fig.data)}",
                    )
        
        # Customize layout
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
            height=max(600, len(vol_df.columns) * 40),  # Dynamic height based on number of pairs
            margin=dict(t=50, l=120, r=20, b=50),  # Add margin for labels
            showlegend=False,
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Create a summary table for volatility
        st.markdown("## Volatility Summary Statistics")
        
        # Calculate statistics
        vol_summary = {}
        for pair_name in vol_df.columns:
            values = vol_df[pair_name].dropna()
            if not values.empty:
                vol_summary[pair_name] = {
                    'Mean Vol': values.mean(),
                    'Max Vol': values.max(),
                    'Min Vol': values.min(),
                    'Median Vol': values.median(),
                    'Std Dev': values.std(),
                    'High Vol Count': (values >= 0.6).sum(),
                    'Low Vol Count': (values < 0.6).sum(),
                    'Data Points': len(values)
                }
        
        if vol_summary:
            vol_summary_df = pd.DataFrame(vol_summary).T
            
            # Format the values
            formatted_vol_summary = vol_summary_df.copy()
            formatted_vol_summary['Mean Vol'] = formatted_vol_summary['Mean Vol'].apply(lambda x: f"{x*100:.1f}%")
            formatted_vol_summary['Max Vol'] = formatted_vol_summary['Max Vol'].apply(lambda x: f"{x*100:.1f}%")
            formatted_vol_summary['Min Vol'] = formatted_vol_summary['Min Vol'].apply(lambda x: f"{x*100:.1f}%")
            formatted_vol_summary['Median Vol'] = formatted_vol_summary['Median Vol'].apply(lambda x: f"{x*100:.1f}%")
            formatted_vol_summary['Std Dev'] = formatted_vol_summary['Std Dev'].apply(lambda x: f"{x*100:.1f}%")
            
            # Add volatility profile
            formatted_vol_summary['High Vol %'] = (vol_summary_df['High Vol Count'] / vol_summary_df['Data Points']).apply(lambda x: f"{x*100:.1f}%")
            
            # Reorder columns
            formatted_vol_summary = formatted_vol_summary[['Mean Vol', 'Median Vol', 'Min Vol', 'Max Vol', 'Std Dev', 'High Vol %', 'High Vol Count', 'Low Vol Count', 'Data Points']]
            
            # Color cells based on values
            def color_vol_cells(val, column):
                if 'Vol' in column and '%' in str(val) and not 'Count' in column and not 'High Vol %' in column:
                    try:
                        value = float(val.strip('%'))/100
                        if value < 0.3:
                            return 'background-color: rgba(0, 255, 0, 0.3)'
                        elif value < 0.6:
                            return 'background-color: rgba(255, 255, 0, 0.3)'
                        elif value < 1.0:
                            return 'background-color: rgba(255, 165, 0, 0.3)'
                        else:
                            return 'background-color: rgba(255, 0, 0, 0.3)'
                    except Exception:
                        return ''
                elif column == 'High Vol %' and '%' in str(val):
                    try:
                        value = float(val.strip('%'))/100
                        if value < 0.3:
                            return 'background-color: rgba(0, 255, 0, 0.3)'
                        elif value < 0.5:
                            return 'background-color: rgba(255, 255, 0, 0.3)'
                        elif value < 0.7:
                            return 'background-color: rgba(255, 165, 0, 0.3)'
                        else:
                            return 'background-color: rgba(255, 0, 0, 0.3)'
                    except Exception:
                        return ''
                return ''
            
            # Apply styling
            styled_vol_summary = formatted_vol_summary.style.apply(lambda x: [color_vol_cells(val, col) for val, col in zip(x, formatted_vol_summary.columns)], axis=1)
            
            # Display the summary table
            st.dataframe(styled_vol_summary, use_container_width=True)
    
    with tab3:
        st.markdown("## Combined Edge & Volatility Matrix (10min timeframe, Last 24 hours)")
        st.markdown("### Each cell shows: Edge (top) and Volatility (bottom)")
        
        # Create combined data
        combined_data = {}
        for pair_name in selected_pairs:
            if pair_name in pair_results:
                df = pair_results[pair_name]
                if 'edge' in df.columns and 'volatility' in df.columns:
                    combined_series = df.set_index('time_label').apply(
                        lambda row: {
                            'edge': row['edge'], 
                            'volatility': row['volatility']
                        }, 
                        axis=1
                    )
                    combined_data[pair_name] = combined_series
        
        # Create DataFrame with all pairs
        combined_df = pd.DataFrame(combined_data)
        
        # Apply the time blocks in the proper order (most recent first)
        combined_df = combined_df.reindex(ordered_times)
        
        # Create a plotly figure for combined view
        fig = go.Figure()
        
        # Create visualization
        for pair_name in combined_df.columns:
            combined_values = combined_df[pair_name].values
            
            fig.add_trace(go.Heatmap(
                z=[np.zeros(len(combined_values))],  # Placeholder
                x=combined_df.index,
                y=[pair_name],
                colorscale=[[0, 'rgba(255,255,255,0)']], # Transparent colorscale
                showscale=False,
                hoverinfo='none',
            ))
            
            # Add rectangles with split colors
            for i, val_dict in enumerate(combined_values):
                if val_dict is not None:
                    edge_val = val_dict.get('edge')
                    vol_val = val_dict.get('volatility')
                    
                    # Add split colored rectangle - top half for edge, bottom half for volatility
                    if not pd.isna(edge_val):
                        fig.add_shape(
                            type="rect",
                            x0=i-0.5,
                            x1=i+0.5,
                            y0=0,  # Center of the cell
                            y1=0.5,  # Top half
                            xref=f"x",
                            yref=f"y{len(fig.data)}",
                            fillcolor=map_edge_to_color(edge_val),
                            line=dict(width=1, color='white'),
                        )
                    
                    if not pd.isna(vol_val):
                        fig.add_shape(
                            type="rect",
                            x0=i-0.5,
                            x1=i+0.5,
                            y0=-0.5,  # Bottom half
                            y1=0,  # Center of the cell
                            xref=f"x",
                            yref=f"y{len(fig.data)}",
                            fillcolor=map_volatility_to_color(vol_val),
                            line=dict(width=1, color='white'),
                        )
                    
                    # Add text annotation for edge value (top)
                    if not pd.isna(edge_val):
                        formatted_edge = format_value(edge_val)
                        fig.add_annotation(
                            x=i,
                            y=0.25,  # Top quarter
                            text=formatted_edge,
                            showarrow=False,
                            font=dict(
                                color='black' if abs(edge_val) < 0.05 else 'white',
                                size=9
                            ),
                            xref=f"x",
                            yref=f"y{len(fig.data)}",
                        )
                    
                    # Add text annotation for volatility value (bottom)
                    if not pd.isna(vol_val):
                        formatted_vol = format_value(vol_val, 'volatility')
                        fig.add_annotation(
                            x=i,
                            y=-0.25,  # Bottom quarter
                            text=formatted_vol,
                            showarrow=False,
                            font=dict(
                                color='black' if vol_val < 0.6 else 'white',
                                size=9
                            ),
                            xref=f"x",
                            yref=f"y{len(fig.data)}",
                        )
                    
                    # Add separator line in the middle
                    fig.add_shape(
                        type="line",
                        x0=i-0.5,
                        x1=i+0.5,
                        y0=0,
                        y1=0,
                        xref=f"x",
                        yref=f"y{len(fig.data)}",
                        line=dict(width=1, color='white'),
                    )
            
            # Add hover text with both values
            hover_text = []
            for val_dict in combined_values:
                if val_dict is not None:
                    edge_val = val_dict.get('edge')
                    vol_val = val_dict.get('volatility')
                    edge_text = format_value(edge_val) if not pd.isna(edge_val) else '-'
                    vol_text = format_value(vol_val, 'volatility') if not pd.isna(vol_val) else '-'
                    hover_text.append(f"Edge: {edge_text}<br>Vol: {vol_text}")
                else:
                    hover_text.append("No data")
            
            # Add invisible trace for hover
            fig.add_trace(go.Scatter(
                x=combined_df.index,
                y=[0] * len(combined_df.index),
                mode='markers',
                marker=dict(
                    size=0,
                    color='rgba(0,0,0,0)'
                ),
                hoverinfo='text',
                text=hover_text,
                hovertemplate='<b>%{y}</b><br>Time: %{x}<br>%{text}<extra></extra>',
                showlegend=False,
                yaxis=f"y{len(fig.data)-1}"
            ))
        
        # Customize layout
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
            height=max(600, len(combined_df.columns) * 40),  # Dynamic height based on number of pairs
            margin=dict(t=50, l=120, r=20, b=50),  # Add margin for labels
            showlegend=False,
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Create a legend for the combined view
        st.markdown("### Color Legend")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Edge Colors (Top Half):**")
            st.markdown("<span style='display:inline-block; width:15px; height:15px; background-color:rgba(180, 0, 0, 0.9)'></span> Very Negative (< -10%)", unsafe_allow_html=True)
            st.markdown("<span style='display:inline-block; width:15px; height:15px; background-color:rgba(255, 0, 0, 0.9)'></span> Negative (-10% to -5%)", unsafe_allow_html=True)
            st.markdown("<span style='display:inline-block; width:15px; height:15px; background-color:rgba(255, 150, 150, 0.9)'></span> Slightly Negative (-5% to -1%)", unsafe_allow_html=True)
            st.markdown("<span style='display:inline-block; width:15px; height:15px; background-color:rgba(255, 255, 150, 0.9)'></span> Neutral (-1% to 1%)", unsafe_allow_html=True)
            st.markdown("<span style='display:inline-block; width:15px; height:15px; background-color:rgba(150, 255, 150, 0.9)'></span> Slightly Positive (1% to 5%)", unsafe_allow_html=True)
            st.markdown("<span style='display:inline-block; width:15px; height:15px; background-color:rgba(0, 255, 0, 0.9)'></span> Positive (5% to 10%)", unsafe_allow_html=True)
            st.markdown("<span style='display:inline-block; width:15px; height:15px; background-color:rgba(0, 180, 0, 0.9)'></span> Very Positive (> 10%)", unsafe_allow_html=True)
        
        with col2:
            st.markdown("**Volatility Colors (Bottom Half):**")
            st.markdown("<span style='display:inline-block; width:15px; height:15px; background-color:rgba(0, 255, 0, 0.9)'></span> Low (< 30%)", unsafe_allow_html=True)
            st.markdown("<span style='display:inline-block; width:15px; height:15px; background-color:rgba(255, 255, 0, 0.9)'></span> Medium (30% to 60%)", unsafe_allow_html=True)
            st.markdown("<span style='display:inline-block; width:15px; height:15px; background-color:rgba(255, 165, 0, 0.9)'></span> High (60% to 100%)", unsafe_allow_html=True)
            st.markdown("<span style='display:inline-block; width:15px; height:15px; background-color:rgba(255, 0, 0, 0.9)'></span> Extreme (> 100%)", unsafe_allow_html=True)
        
        # Create comparison analysis
        st.markdown("## Edge vs. Volatility Analysis")
        
        # Create scatter plot of edge vs volatility
        scatter_data = []
        for pair_name in selected_pairs:
            if pair_name in pair_results:
                df = pair_results[pair_name]
                if 'edge' in df.columns and 'volatility' in df.columns:
                    # Filter out rows with NaN values
                    valid_data = df.dropna(subset=['edge', 'volatility'])
                    
                    for _, row in valid_data.iterrows():
                        scatter_data.append({
                            'pair_name': pair_name,
                            'time_label': row['time_label'],
                            'edge': row['edge'],
                            'volatility': row['volatility']
                        })
        
        if scatter_data:
            # Create DataFrame for scatter plot
            scatter_df = pd.DataFrame(scatter_data)
            
            # Create scatter plot
            fig = px.scatter(
                scatter_df, 
                x='volatility', 
                y='edge',
                color='pair_name',
                hover_name='pair_name',
                hover_data=['time_label', 'edge', 'volatility'],
                labels={
                    'edge': 'Edge (PNL / Collateral)',
                    'volatility': 'Volatility (Annualized)',
                    'pair_name': 'Trading Pair'
                },
                title='Edge vs. Volatility Scatter Plot'
            )
            
            # Format axis labels
            fig.update_layout(
                xaxis_title="Volatility (Annualized)",
                yaxis_title="Edge (PNL / Collateral)",
                xaxis_tickformat='.1%',
                yaxis_tickformat='.1%',
                height=600
            )
            
            # Add reference lines
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.add_vline(x=0.6, line_dash="dash", line_color="gray")
            
            # Add quadrant labels
            fig.add_annotation(
                x=0.3, y=0.05,
                text="Low Vol, Positive Edge<br>(Ideal)",
                showarrow=False,
                font=dict(color="green")
            )
            fig.add_annotation(
                x=0.3, y=-0.05,
                text="Low Vol, Negative Edge<br>(Caution)",
                showarrow=False,
                font=dict(color="red")
            )
            fig.add_annotation(
                x=0.9, y=0.05,
                text="High Vol, Positive Edge<br>(Monitor)",
                showarrow=False,
                font=dict(color="orange")
            )
            fig.add_annotation(
                x=0.9, y=-0.05,
                text="High Vol, Negative Edge<br>(High Risk)",
                showarrow=False,
                font=dict(color="red")
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate correlation between edge and volatility
            overall_corr = scatter_df[['edge', 'volatility']].corr().iloc[0, 1]
            
            # Calculate correlation by pair
            pair_corrs = {}
            for pair_name, group in scatter_df.groupby('pair_name'):
                if len(group) >= 5:  # Require at least 5 data points
                    pair_corrs[pair_name] = group[['edge', 'volatility']].corr().iloc[0, 1]
            
            # Display correlation analysis
            st.markdown(f"### Correlation Analysis: Edge vs. Volatility")
            st.markdown(f"**Overall Correlation**: {overall_corr:.3f}")
            
            if pair_corrs:
                corr_df = pd.DataFrame(
                    {'Correlation': pair_corrs}
                ).sort_values('Correlation', ascending=False)
                
                # Format and color cells
                def color_correlation(val):
                    if abs(val) > 0.7:
                        if val > 0:
                            return 'background-color: rgba(0, 180, 0, 0.3)'
                        else:
                            return 'background-color: rgba(255, 0, 0, 0.3)'
                    elif abs(val) > 0.3:
                        if val > 0:
                            return 'background-color: rgba(150, 255, 150, 0.3)'
                        else:
                            return 'background-color: rgba(255, 150, 150, 0.3)'
                    else:
                        return 'background-color: rgba(255, 255, 150, 0.3)'
                
                styled_corr = corr_df.style.format({
                    'Correlation': '{:.3f}'
                }).applymap(color_correlation)
                
                st.markdown("**Correlation by Pair**")
                st.dataframe(styled_corr, use_container_width=True)
                
                # Interpretation
                st.markdown("""
                **Interpretation:**
                - **Positive correlation**: Higher volatility tends to be associated with higher edge
                - **Negative correlation**: Higher volatility tends to be associated with lower edge
                - **Near-zero correlation**: No clear relationship between volatility and edge
                """)
            
            # Quadrant Analysis
            st.markdown("### Quadrant Analysis")
            
            # Define quadrants
            scatter_df['quadrant'] = scatter_df.apply(
                lambda row: 'Low Vol, Positive Edge' if row['volatility'] < 0.6 and row['edge'] > 0 else
                            'Low Vol, Negative Edge' if row['volatility'] < 0.6 and row['edge'] <= 0 else
                            'High Vol, Positive Edge' if row['volatility'] >= 0.6 and row['edge'] > 0 else
                            'High Vol, Negative Edge',
                axis=1
            )
            
            # Count observations in each quadrant
            quadrant_counts = scatter_df['quadrant'].value_counts()
            
            # Calculate percentage
            quadrant_pcts = (quadrant_counts / len(scatter_df) * 100).round(1)
            
            # Combine counts and percentages
            quadrant_df = pd.DataFrame({
                'Count': quadrant_counts,
                'Percentage': quadrant_pcts
            })
            
            # Format percentage column
            quadrant_df['Percentage'] = quadrant_df['Percentage'].apply(lambda x: f"{x:.1f}%")
            
            # Reorder rows
            quadrant_df = quadrant_df.reindex([
                'Low Vol, Positive Edge',
                'Low Vol, Negative Edge',
                'High Vol, Positive Edge',
                'High Vol, Negative Edge'
            ])
            
            # Display quadrant analysis
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.dataframe(quadrant_df, use_container_width=True)
            
            with col2:
                # Create pie chart
                fig = go.Figure(data=[go.Pie(
                    labels=quadrant_df.index,
                    values=quadrant_df['Count'],
                    textinfo='percent+label',
                    marker=dict(
                        colors=['rgba(0, 255, 0, 0.7)', 'rgba(255, 255, 0, 0.7)', 
                                'rgba(255, 165, 0, 0.7)', 'rgba(255, 0, 0, 0.7)']
                    )
                )])
                
                fig.update_layout(
                    title="Distribution of Edge-Volatility Quadrants",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)

# Add explanation for the dashboard
with st.expander("Understanding the Edge & Volatility Matrix"):
    st.markdown("""
    ## 📊 How to Use This Dashboard
    
    This dashboard provides a comprehensive view of trading edge and volatility across all selected trading pairs using 10-minute intervals over the past 24 hours (Singapore time).
    
    ### Key Metrics
    
    - **Edge**: Calculated as PNL / Total Open Collateral
      - Positive edge: The trading pair is profitable
      - Negative edge: The trading pair is unprofitable
      
    - **Volatility**: Annualized standard deviation of log returns
      - Low volatility (<30%): Stable price movement
      - Medium volatility (30-60%): Moderate price movement
      - High volatility (60-100%): Significant price movement
      - Extreme volatility (>100%): Very high price movement
    
    ### Tabs Explanation
    
    1. **Edge Tab**: Shows a heatmap of edge values for each pair across time
    2. **Volatility Tab**: Shows a heatmap of volatility values for each pair across time
    3. **Combined View Tab**: Displays both edge (top half) and volatility (bottom half) in each cell
    
    ### Interpreting the Combined View
    
    The ideal scenario is low volatility with positive edge (top left quadrant in the scatter plot):
    
    - **Low Vol, Positive Edge**: Stable and profitable pairs
    - **Low Vol, Negative Edge**: Stable but unprofitable pairs
    - **High Vol, Positive Edge**: Profitable but potentially risky pairs
    - **High Vol, Negative Edge**: Unprofitable and risky pairs
    
    ### Using This Information
    
    - Identify pairs with consistently positive edge for increased allocation
    - Monitor pairs with high volatility but positive edge for potential opportunities
    - Consider reducing exposure to pairs with negative edge, especially those with high volatility
    - Track changes in edge and volatility patterns over time to detect shifts in market behavior
    
    ### Technical Details
    
    - PNL includes order PNL, funding fees, and rebate payments
    - Volatility is calculated from price data within each 10-minute interval and annualized
    - The time blocks are aligned to clock times (e.g., 00:00-00:10, 00:10-00:20)
    - Singapore timezone (UTC+8) is used throughout
    """
    )
else:
    st.warning("No data available for the selected pairs. Try selecting different pairs or refreshing the data.")