# Save this as optimized_5sec_volatility_rankings.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import psycopg2
import pytz
from sqlalchemy import create_engine
from scipy.stats import spearmanr

st.set_page_config(page_title="5sec Volatility Plot with Rollbit", page_icon="📈", layout="wide")

# --- UI Setup ---
st.title("5-Second Volatility Plot with Rollbit Rankings")

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

conn = psycopg2.connect(**db_params)

# Cache token list for longer
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_trading_pairs():
    query = """
    SELECT pair_name 
    FROM trade_pool_pairs 
    WHERE status = 1
    ORDER BY pair_name
    """
    df = pd.read_sql_query(query, engine)
    return df['pair_name'].tolist()

all_tokens = fetch_trading_pairs()

# Create tabs
tab1, tab2 = st.tabs(["Volatility Chart", "Rankings"])

with tab1:
    col1, col2 = st.columns([3, 1])

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

    sg_tz = pytz.timezone('Asia/Singapore')
    now_sg = datetime.now(sg_tz)
    st.write(f"Current time (Singapore): {now_sg.strftime('%Y-%m-%d %H:%M:%S')}")

# Optimized Rollbit fetch - resample to 5 seconds
@st.cache_data(ttl=60)
def fetch_rollbit_parameters_5sec(token, hours=3):
    """Fetch Rollbit parameters with 5-second resolution"""
    try:
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
        WHERE pair_name = '{token}'
        AND created_at >= '{start_str}'::timestamp - INTERVAL '8 hour'
        AND created_at <= '{end_str}'::timestamp - INTERVAL '8 hour'
        ORDER BY created_at
        """

        df = pd.read_sql_query(query, engine)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').sort_index()
            # Resample to 5 seconds
            df = df.resample('5s').ffill()
        return df
    except Exception as e:
        st.error(f"Error fetching Rollbit parameters: {e}")
        return None

# 5-second volatility calculation
@st.cache_data(ttl=30)
def get_volatility_data_5sec(token, hours=3):
    now_sg = datetime.now(pytz.timezone('Asia/Singapore'))
    start_time_sg = now_sg - timedelta(hours=hours)
    
    # Get partitions
    today_str = now_sg.strftime("%Y%m%d")
    yesterday_str = (now_sg - timedelta(days=1)).strftime("%Y%m%d")
    
    tables = [f"oracle_price_log_partition_{yesterday_str}", f"oracle_price_log_partition_{today_str}"]
    
    start_time_str = start_time_sg.strftime("%Y-%m-%d %H:%M:%S")
    end_time_str = now_sg.strftime("%Y-%m-%d %H:%M:%S")
    
    # Build query for both tables
    queries = []
    for table in tables:
        query = f"""
        SELECT 
            created_at + INTERVAL '8 hour' AS timestamp,
            final_price
        FROM public.{table}
        WHERE created_at >= '{start_time_str}'::timestamp - INTERVAL '8 hour'
        AND created_at <= '{end_time_str}'::timestamp - INTERVAL '8 hour'
        AND source_type = 0
        AND pair_name = '{token}'
        """
        queries.append(query)
    
    # Union query
    final_query = " UNION ALL ".join(queries) + " ORDER BY timestamp"
    
    # Execute query
    try:
        df = pd.read_sql_query(final_query, engine)
    except:
        # If yesterday's table doesn't exist, only use today
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
    
    if df.empty:
        return None, None
    
    # Process timestamps
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()
    
    # Resample to 500ms
    price_data = df['final_price'].resample('500ms').ffill().dropna()
    
    # Create 5-second windows
    result = []
    start_date = price_data.index.min().floor('5s')
    end_date = price_data.index.max().ceil('5s')
    five_sec_periods = pd.date_range(start=start_date, end=end_date, freq='5s')
    
    for i in range(len(five_sec_periods)-1):
        start_window = five_sec_periods[i]
        end_window = five_sec_periods[i+1]
        
        window_data = price_data[(price_data.index >= start_window) & (price_data.index < end_window)]
        
        if len(window_data) >= 2:
            # Calculate volatility for 5-second window
            log_returns = np.diff(np.log(window_data.values))
            annualization_factor = np.sqrt(6307200)  # For 5-second windows
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
    percentiles = {
        'p25': np.percentile(vol_pct, 25),
        'p50': np.percentile(vol_pct, 50),
        'p75': np.percentile(vol_pct, 75),
        'p95': np.percentile(vol_pct, 95)
    }
    
    return result_df, percentiles

# Get volatility for multiple time windows (rankings)
@st.cache_data(ttl=60)
def get_volatility_snapshots(token, hours=[1, 2, 3]):
    """Get average 5-second volatility for different time windows"""
    snapshots = {}
    
    for h in hours:
        vol_data, _ = get_volatility_data_5sec(token, hours=h)
        
        if vol_data is not None and not vol_data.empty:
            # Get data for the specific hour window
            now_sg = datetime.now(pytz.timezone('Asia/Singapore'))
            start_time = now_sg - timedelta(hours=h)
            end_time = now_sg - timedelta(hours=h-1) if h > 1 else now_sg
            
            window_data = vol_data[(vol_data.index >= start_time) & (vol_data.index < end_time)]
            if not window_data.empty:
                avg_vol = window_data['realized_vol'].mean() * 100  # Convert to percentage
                snapshots[f'hour_{h}'] = avg_vol
            else:
                snapshots[f'hour_{h}'] = None
        else:
            snapshots[f'hour_{h}'] = None
    
    return snapshots

@st.cache_data(ttl=60)
def get_rankings_5sec():
    """Get volatility rankings with 3 hourly snapshots"""
    
    results = []
    
    for token in all_tokens[:30]:  # Limit to 30 for speed
        try:
            snapshots = get_volatility_snapshots(token)
            
            if snapshots['hour_1'] is not None:
                results.append({
                    'pair_name': token,
                    'vol_1h': snapshots['hour_1'],
                    'vol_2h': snapshots['hour_2'],
                    'vol_3h': snapshots['hour_3']
                })
        except:
            continue
    
    if not results:
        return pd.DataFrame(), pd.DataFrame()
    
    vol_df = pd.DataFrame(results)
    
    # Sort and rank by 1h volatility
    vol_df = vol_df.sort_values('vol_1h', ascending=False)
    vol_df['rank_1h'] = range(1, len(vol_df) + 1)
    
    # Sort and rank by 2h volatility if data exists
    if 'vol_2h' in vol_df.columns and vol_df['vol_2h'].notna().any():
        vol_df = vol_df.sort_values('vol_2h', ascending=False)
        vol_df['rank_2h'] = range(1, len(vol_df) + 1)
    
    # Sort and rank by 3h volatility if data exists
    if 'vol_3h' in vol_df.columns and vol_df['vol_3h'].notna().any():
        vol_df = vol_df.sort_values('vol_3h', ascending=False)
        vol_df['rank_3h'] = range(1, len(vol_df) + 1)
    
    # Sort back by 1h ranking
    vol_df = vol_df.sort_values('rank_1h')
    
    # Rollbit ranking
    rollbit_query = """
    WITH latest_config AS (
        SELECT 
            pair_name,
            bust_buffer AS buffer_rate,
            position_multiplier,
            ROW_NUMBER() OVER (PARTITION BY pair_name ORDER BY created_at DESC) as rn
        FROM rollbit_pair_config
        WHERE created_at >= NOW() - INTERVAL '1 hour'
    )
    SELECT 
        pair_name,
        buffer_rate * 100 as buffer_rate_pct,
        position_multiplier,
        ROW_NUMBER() OVER (ORDER BY buffer_rate DESC) as buffer_rank
    FROM latest_config
    WHERE rn = 1
    ORDER BY buffer_rate DESC
    LIMIT 30
    """
    
    rollbit_df = pd.read_sql_query(rollbit_query, engine)
    
    return vol_df, rollbit_df

# Main chart section
with tab1:
    with st.spinner(f"Loading data for {selected_token}..."):
        vol_data, percentiles = get_volatility_data_5sec(selected_token)
        rollbit_params = fetch_rollbit_parameters_5sec(selected_token)

    if vol_data is not None and not vol_data.empty:
        # Convert to percentage
        vol_data_pct = vol_data.copy()
        vol_data_pct['realized_vol'] = vol_data_pct['realized_vol'] * 100

        # Key metrics
        current_vol = vol_data_pct['realized_vol'].iloc[-1]
        avg_vol = vol_data_pct['realized_vol'].mean()
        max_vol = vol_data_pct['realized_vol'].max()
        min_vol = vol_data_pct['realized_vol'].min()
        
        # Calculate current percentile
        all_vols = vol_data_pct['realized_vol'].values
        current_percentile = (all_vols < current_vol).mean() * 100

        # Create subplots with 3 rows
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                f"{selected_token} Annualized Volatility (5sec windows)",
                "Rollbit Buffer Rate (%)",
                "Rollbit Position Multiplier"
            ),
            row_heights=[0.4, 0.3, 0.3]
        )

        # Process Rollbit data if available
        if rollbit_params is not None and not rollbit_params.empty:
            # Merge with volatility data to ensure aligned timestamps
            combined_data = pd.merge(
                vol_data_pct,
                rollbit_params,
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
            
            # Create unified hover data
            hover_template = (
                "<b>Time: %{x}</b><br>" +
                "Volatility: %{customdata[0]:.1f}%<br>" +
                "Buffer Rate: %{customdata[1]:.3f}%<br>" +
                "Position Mult: %{customdata[2]:,.0f}<br>" +
                "<extra></extra>"
            )
            
            customdata = np.column_stack((
                combined_data['realized_vol'],
                combined_data['buffer_rate_pct'],
                combined_data['position_multiplier']
            ))
            
            # Panel 1: Volatility
            fig.add_trace(
                go.Scatter(
                    x=combined_data.index,
                    y=combined_data['realized_vol'],
                    mode='lines',
                    line=dict(color='blue', width=2),
                    name="Volatility (%)",
                    customdata=customdata,
                    hovertemplate=hover_template,
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
                    marker=dict(size=4),
                    name="Buffer Rate (%)",
                    customdata=customdata,
                    hovertemplate=hover_template,
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
                    marker=dict(size=4),
                    name="Position Multiplier",
                    customdata=customdata,
                    hovertemplate=hover_template,
                    showlegend=False
                ),
                row=3, col=1
            )
            
            # Get latest values for title
            latest_buffer = combined_data['buffer_rate_pct'].iloc[-1]
            latest_pos_mult = combined_data['position_multiplier'].iloc[-1]
            
            # Auto-scale y-axes
            buffer_min = combined_data['buffer_rate_pct'].min()
            buffer_max = combined_data['buffer_rate_pct'].max()
            pos_mult_min = combined_data['position_multiplier'].min()
            pos_mult_max = combined_data['position_multiplier'].max()
            
        else:
            # If no Rollbit data, only show volatility
            hover_template = (
                "<b>Time: %{x}</b><br>" +
                "Volatility: %{y:.1f}%<br>" +
                "Buffer Rate: N/A<br>" +
                "Position Mult: N/A<br>" +
                "<extra></extra>"
            )
            
            fig.add_trace(
                go.Scatter(
                    x=vol_data_pct.index,
                    y=vol_data_pct['realized_vol'],
                    mode='lines',
                    line=dict(color='blue', width=2),
                    name="Volatility (%)",
                    hovertemplate=hover_template,
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # Add notes if no Rollbit data
            fig.add_annotation(
                x=0.5,
                y=0.5,
                text="No Rollbit data available",
                showarrow=False,
                font=dict(size=12),
                xref="x2 domain",
                yref="y2 domain",
                row=2, col=1
            )
            
            fig.add_annotation(
                x=0.5,
                y=0.5,
                text="No Rollbit data available",
                showarrow=False,
                font=dict(size=12),
                xref="x3 domain",
                yref="y3 domain",
                row=3, col=1
            )
            
            latest_buffer = None
            latest_pos_mult = None
            buffer_min = buffer_max = 0
            pos_mult_min = pos_mult_max = 1

        # Add percentile lines
        percentile_lines = [
            ('p25', '#2ECC71', '25th'),  # Green
            ('p50', '#3498DB', '50th'),  # Blue
            ('p75', '#F39C12', '75th'),  # Orange
            ('p95', '#E74C3C', '95th')   # Red
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
                row=1, col=1
            )

        # Create title
        if latest_buffer is not None and latest_pos_mult is not None:
            title_text = f"{selected_token} Analysis Dashboard (5-second windows)<br>" + \
                        f"<sub>Current Vol: {current_vol:.1f}% ({current_percentile:.0f}th percentile) | Buffer: {latest_buffer:.3f}% | Pos Mult: {latest_pos_mult:,.0f}</sub>"
        else:
            title_text = f"{selected_token} Analysis Dashboard (5-second windows)<br>" + \
                        f"<sub>Current Vol: {current_vol:.1f}% ({current_percentile:.0f}th percentile)</sub>"

        # Update layout
        fig.update_layout(
            title=title_text,
            height=800,
            showlegend=False,
            hovermode="x unified",
            plot_bgcolor='white',
            paper_bgcolor='white',
            hoverlabel=dict(
                bgcolor="white",
                bordercolor="black",
                font_size=12
            ),
            # Enable spike lines
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
        for i in range(1, 4):
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

        # Update y-axes with auto-scaling
        fig.update_yaxes(
            title_text="Volatility (%)",
            row=1, col=1,
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray',
            range=[0, max(max_vol * 1.1, percentiles['p95'] * 1.1, 5)]
        )
        
        if latest_buffer is not None:
            fig.update_yaxes(
                title_text="Buffer Rate (%)",
                row=2, col=1,
                tickformat=".3f",
                showgrid=True,
                gridwidth=1,
                gridcolor='LightGray',
                range=[buffer_min * 0.95, buffer_max * 1.05] if buffer_max > buffer_min else None
            )
            
            fig.update_yaxes(
                title_text="Position Multiplier",
                row=3, col=1,
                tickformat=",",
                showgrid=True,
                gridwidth=1,
                gridcolor='LightGray',
                range=[pos_mult_min * 0.95, pos_mult_max * 1.05] if pos_mult_max > pos_mult_min else None
            )
        
        # X-axis labels only on bottom
        fig.update_xaxes(title_text="Time (Singapore)", row=3, col=1, tickformat="%H:%M:%S<br>%m/%d")

        st.plotly_chart(fig, use_container_width=True)

        # Metrics display with percentiles
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
        st.markdown("### Percentiles (3h)")
        pcol1, pcol2, pcol3, pcol4 = st.columns(4)
        with pcol1:
            st.metric("25th", f"{percentiles['p25']:.1f}%")
        with pcol2:
            st.metric("50th", f"{percentiles['p50']:.1f}%")
        with pcol3:
            st.metric("75th", f"{percentiles['p75']:.1f}%")
        with pcol4:
            st.metric("95th", f"{percentiles['p95']:.1f}%")

        # Current Rollbit metrics if available
        if rollbit_params is not None and not rollbit_params.empty and latest_buffer is not None:
            st.markdown("### Current Rollbit Parameters")
            rcol1, rcol2 = st.columns(2)
            with rcol1:
                st.metric("Current Buffer Rate", f"{latest_buffer:.3f}%")
            with rcol2:
                st.metric("Current Position Multiplier", f"{latest_pos_mult:,.0f}")

    else:
        st.error("No volatility data available for the selected token")

# Rankings tab
with tab2:
    st.markdown("## Token Rankings (5-second volatility)")
    
    # Main rankings
    with st.spinner("Loading rankings..."):
        vol_ranking, rollbit_ranking = get_rankings_5sec()
    
    # Volatility ranking with 3 hourly snapshots
    st.markdown("### Surf Tokens - Volatility Ranking (3 Hourly Snapshots)")
    st.markdown("*Average 5-second volatility over different time windows*")
    
    if not vol_ranking.empty:
        vol_display = vol_ranking.copy()
        
        # Format the columns
        vol_display['vol_1h'] = vol_display['vol_1h'].apply(lambda x: f"{x:.2f}%" if x is not None else "N/A")
        vol_display['vol_2h'] = vol_display['vol_2h'].apply(lambda x: f"{x:.2f}%" if x is not None else "N/A")
        vol_display['vol_3h'] = vol_display['vol_3h'].apply(lambda x: f"{x:.2f}%" if x is not None else "N/A")
        
        # Create column names based on available data
        columns_map = {
            'pair_name': 'Token',
            'vol_1h': 'Vol 1h (%)',
            'vol_2h': 'Vol 2h (%)',
            'vol_3h': 'Vol 3h (%)',
            'rank_1h': 'Rank 1h',
            'rank_2h': 'Rank 2h',
            'rank_3h': 'Rank 3h'
        }
        
        # Only rename columns that exist
        existing_columns = [col for col in columns_map.keys() if col in vol_display.columns]
        vol_display = vol_display[existing_columns].rename(columns=columns_map)
        
        # Select display columns based on what exists
        display_columns = ['Rank 1h', 'Token', 'Vol 1h (%)']
        if 'Vol 2h (%)' in vol_display.columns:
            display_columns.append('Vol 2h (%)')
        if 'Vol 3h (%)' in vol_display.columns:
            display_columns.append('Vol 3h (%)')
        
        vol_display = vol_display[display_columns]
        
        st.dataframe(vol_display, hide_index=True, use_container_width=True)
        
        # Add stability indicator
        st.markdown("*If rankings are consistent across time windows, the volatility patterns are stable*")
    else:
        st.warning("No volatility data available")
    
    # Rollbit ranking
    st.markdown("### Rollbit Buffer Ranking")
    if not rollbit_ranking.empty:
        rollbit_display = rollbit_ranking.copy()
        rollbit_display['buffer_rate_pct'] = rollbit_display['buffer_rate_pct'].apply(lambda x: f"{x:.3f}%")
        rollbit_display['position_multiplier'] = rollbit_display['position_multiplier'].apply(lambda x: f"{x:,.0f}")
        rollbit_display.columns = ['Token', 'Buffer Rate (%)', 'Position Multiplier', 'Rank']
        rollbit_display = rollbit_display[['Rank', 'Token', 'Buffer Rate (%)', 'Position Multiplier']]
        st.dataframe(rollbit_display, hide_index=True, use_container_width=True)
    else:
        st.warning("No Rollbit data available")
    
    if st.button("Refresh Rankings", type="primary"):
        st.cache_data.clear()
        st.rerun()