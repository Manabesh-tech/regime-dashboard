# Save this as optimized_1min_volatility_rankings.py

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

st.set_page_config(page_title="1min Volatility Plot with Rollbit", page_icon="📈", layout="wide")

# --- UI Setup ---
st.title("1-Minute Volatility Plot with Rollbit Rankings")

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

# Optimized Rollbit fetch - only get last value and minimal history
@st.cache_data(ttl=300)
def fetch_rollbit_parameters_simplified(token, hours=18):
    """Fetch simplified Rollbit parameters"""
    try:
        now_sg = datetime.now(pytz.timezone('Asia/Singapore'))
        start_time_sg = now_sg - timedelta(hours=hours)
        
        start_str = start_time_sg.strftime("%Y-%m-%d %H:%M:%S")
        end_str = now_sg.strftime("%Y-%m-%d %H:%M:%S")

        # Only get hourly samples to reduce data
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
    except Exception as e:
        st.error(f"Error fetching Rollbit parameters: {e}")
        return None

# Simplified volatility calculation - only 12 hours
@st.cache_data(ttl=60)
def get_volatility_data_simplified(token, hours=12):
    now_sg = datetime.now(pytz.timezone('Asia/Singapore'))
    start_time_sg = now_sg - timedelta(hours=hours+1)  # Extra hour for buffer
    
    # Only get today's partition to speed up
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
    
    # Create 1-minute windows
    result = []
    start_date = price_data.index.min().floor('1min')
    end_date = price_data.index.max().ceil('1min')
    one_min_periods = pd.date_range(start=start_date, end=end_date, freq='1min')
    
    for i in range(len(one_min_periods)-1):
        start_window = one_min_periods[i]
        end_window = one_min_periods[i+1]
        
        window_data = price_data[(price_data.index >= start_window) & (price_data.index < end_window)]
        
        if len(window_data) >= 2:
            # Calculate volatility
            log_returns = np.diff(np.log(window_data.values))
            annualization_factor = np.sqrt(63072000 / 120)  # For 1-minute windows
            volatility = np.std(log_returns) * annualization_factor
            
            result.append({
                'timestamp': start_window,
                'realized_vol': volatility
            })
    
    if not result:
        return None, None
    
    result_df = pd.DataFrame(result).set_index('timestamp')
    
    # Calculate percentiles for the last 12 hours
    vol_pct = result_df['realized_vol'] * 100
    percentiles = {
        'p25': np.percentile(vol_pct, 25),
        'p50': np.percentile(vol_pct, 50),
        'p75': np.percentile(vol_pct, 75),
        'p95': np.percentile(vol_pct, 95)
    }
    
    return result_df, percentiles

# Get common tokens between Surf and Rollbit
@st.cache_data(ttl=300)
def get_common_tokens_volatility():
    """Get volatility ranking for tokens that exist in both Surf and Rollbit"""
    # First get all Rollbit tokens
    rollbit_query = """
    SELECT DISTINCT pair_name
    FROM rollbit_pair_config
    WHERE created_at >= NOW() - INTERVAL '1 day'
    """
    rollbit_tokens_df = pd.read_sql_query(rollbit_query, engine)
    rollbit_tokens = set(rollbit_tokens_df['pair_name'].tolist())
    
    # Get Surf tokens (from our all_tokens list)
    surf_tokens = set(all_tokens)
    
    # Find common tokens
    common_tokens = list(surf_tokens.intersection(rollbit_tokens))
    
    # Calculate volatility for common tokens
    results = []
    for token in common_tokens[:30]:  # Limit to 30 for speed
        try:
            vol_data, _ = get_volatility_data_simplified(token, hours=12)
            
            if vol_data is not None and not vol_data.empty:
                avg_vol = vol_data['realized_vol'].mean() * 100  # Convert to percentage
                results.append({
                    'pair_name': token,
                    'avg_volatility': avg_vol
                })
        except:
            continue
    
    common_vol_df = pd.DataFrame(results).sort_values('avg_volatility', ascending=False)
    common_vol_df['vol_rank'] = range(1, len(common_vol_df) + 1)
    
    return common_vol_df
@st.cache_data(ttl=300)
def get_rankings_simplified():
    """Get both volatility and Rollbit rankings"""
    
    # Get all tokens volatility for ranking
    results = []
    
    for token in all_tokens[:50]:  # Limit to top 50 for speed
        try:
            vol_data, _ = get_volatility_data_simplified(token, hours=12)
            
            if vol_data is not None and not vol_data.empty:
                avg_vol = vol_data['realized_vol'].mean() * 100  # Convert to percentage
                results.append({
                    'pair_name': token,
                    'avg_volatility': avg_vol
                })
        except:
            continue
    
    vol_df = pd.DataFrame(results).sort_values('avg_volatility', ascending=False)
    vol_df['vol_rank'] = range(1, len(vol_df) + 1)
    
    # Rollbit ranking
    rollbit_query = """
    WITH latest_config AS (
        SELECT 
            pair_name,
            bust_buffer AS buffer_rate,
            position_multiplier,
            ROW_NUMBER() OVER (PARTITION BY pair_name ORDER BY created_at DESC) as rn
        FROM rollbit_pair_config
        WHERE created_at >= NOW() - INTERVAL '1 day'
    )
    SELECT 
        pair_name,
        buffer_rate * 100 as buffer_rate_pct,
        position_multiplier,
        ROW_NUMBER() OVER (ORDER BY buffer_rate DESC) as buffer_rank
    FROM latest_config
    WHERE rn = 1
    ORDER BY buffer_rate DESC
    LIMIT 50
    """
    
    rollbit_df = pd.read_sql_query(rollbit_query, engine)
    
    return vol_df, rollbit_df

# Main chart section
with tab1:
    with st.spinner(f"Loading data for {selected_token}..."):
        vol_data, percentiles = get_volatility_data_simplified(selected_token)
        rollbit_params = fetch_rollbit_parameters_simplified(selected_token)

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
                f"{selected_token} Annualized Volatility (1min windows)",
                "Rollbit Buffer Rate (%)",
                "Rollbit Position Multiplier"
            ),
            row_heights=[0.4, 0.3, 0.3]
        )

        # Process Rollbit data if available
        if rollbit_params is not None and not rollbit_params.empty:
            # Resample Rollbit data to 1-minute intervals to match volatility data
            rollbit_resampled = rollbit_params.resample('1min').ffill()
            
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
                    marker=dict(size=6),
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
                    marker=dict(size=6),
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
            title_text = f"{selected_token} Analysis Dashboard<br>" + \
                        f"<sub>Current Vol: {current_vol:.1f}% ({current_percentile:.0f}th percentile) | Buffer: {latest_buffer:.3f}% | Pos Mult: {latest_pos_mult:,.0f}</sub>"
        else:
            title_text = f"{selected_token} Analysis Dashboard<br>" + \
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
        fig.update_xaxes(title_text="Time (Singapore)", row=3, col=1, tickformat="%H:%M<br>%m/%d")

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
        st.markdown("### Percentiles (12h)")
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
    st.markdown("## Token Rankings")
    
    # Main rankings in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### All Surf Tokens - Volatility Ranking (12h)")
        with st.spinner("Loading volatility rankings..."):
            vol_ranking, rollbit_ranking = get_rankings_simplified()
        
        if not vol_ranking.empty:
            vol_display = vol_ranking.copy()
            vol_display['avg_volatility'] = vol_display['avg_volatility'].apply(lambda x: f"{x:.2f}%")
            vol_display.columns = ['Token', 'Avg Volatility (%)', 'Rank']
            vol_display = vol_display[['Rank', 'Token', 'Avg Volatility (%)']]
            st.dataframe(vol_display, hide_index=True, use_container_width=True)
        else:
            st.warning("No volatility data available")
    
    with col2:
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
    
    # Common tokens ranking section
    st.markdown("### Common Tokens (Surf + Rollbit) - Volatility Ranking")
    st.markdown("*Tokens that exist in both Surf and Rollbit platforms*")
    
    with st.spinner("Loading common tokens volatility ranking..."):
        common_vol_ranking = get_common_tokens_volatility()
    
    if not common_vol_ranking.empty:
        # Get Rollbit rankings for common tokens
        common_tokens_list = common_vol_ranking['pair_name'].tolist()
        rollbit_common_query = f"""
        WITH latest_config AS (
            SELECT 
                pair_name,
                bust_buffer AS buffer_rate,
                position_multiplier,
                ROW_NUMBER() OVER (PARTITION BY pair_name ORDER BY created_at DESC) as rn
            FROM rollbit_pair_config
            WHERE created_at >= NOW() - INTERVAL '1 day'
            AND pair_name IN ('{"','".join(common_tokens_list)}')
        )
        SELECT 
            pair_name,
            buffer_rate * 100 as buffer_rate_pct,
            position_multiplier
        FROM latest_config
        WHERE rn = 1
        ORDER BY buffer_rate DESC
        """
        
        rollbit_common_df = pd.read_sql_query(rollbit_common_query, engine)
        rollbit_common_df['buffer_rank'] = range(1, len(rollbit_common_df) + 1)
        
        # Merge volatility and buffer rankings
        comparison_df = pd.merge(
            common_vol_ranking,
            rollbit_common_df,
            on='pair_name',
            how='inner'
        )
        
        # Display the comparison table
        display_df = comparison_df[['vol_rank', 'pair_name', 'avg_volatility', 'buffer_rate_pct', 'buffer_rank']].copy()
        display_df['rank_diff'] = display_df['buffer_rank'] - display_df['vol_rank']
        display_df['avg_volatility'] = display_df['avg_volatility'].apply(lambda x: f"{x:.2f}%")
        display_df['buffer_rate_pct'] = display_df['buffer_rate_pct'].apply(lambda x: f"{x:.3f}%")
        display_df.columns = ['Vol Rank', 'Token', 'Avg Vol (%)', 'Buffer Rate (%)', 'Buffer Rank', 'Rank Diff']
        
        # Color code the rank difference
        def style_rank_diff(val):
            if val > 5:
                return 'color: red'
            elif val < -5:
                return 'color: green'
            else:
                return 'color: black'
        
        styled_df = display_df.style.applymap(style_rank_diff, subset=['Rank Diff'])
        st.dataframe(styled_df, hide_index=True, use_container_width=True)
        
        # Calculate correlation for common tokens
        corr, _ = spearmanr(comparison_df['vol_rank'], comparison_df['buffer_rank'])
        st.metric("Common Tokens Rank Correlation", f"{corr:.3f}")
    else:
        st.warning("No common tokens found between Surf and Rollbit")
    
    if st.button("Refresh Rankings", type="primary"):
        st.cache_data.clear()
        st.rerun()
