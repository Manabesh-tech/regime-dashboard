# Save this as optimized_10sec_volatility.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import psycopg2
import pytz
from sqlalchemy import create_engine

st.set_page_config(page_title="10sec Volatility Plot with Rollbit", page_icon="📈", layout="wide")

# --- UI Setup ---
st.title("10-Second Volatility Plot with Rollbit")

# DB connection
db_params = {
    'host': 'aws-jp-tk-surf-pg-public.cluster-csteuf9lw8dv.ap-northeast-1.rds.amazonaws.com',
    'port': 5432,
    'database': 'replication_report',
    'user': 'public_replication',
    'password': '866^FKC4hllk'
}

engine = create_engine(
    f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['database']}",
    isolation_level="AUTOCOMMIT",  # 设置自动提交模式
    pool_size=5,  # 连接池大小
    max_overflow=10,  # 最大溢出连接数
    pool_timeout=30,  # 连接超时时间
    pool_recycle=1800,  # 连接回收时间(30分钟)
    pool_pre_ping=True,  # 使用连接前先测试连接是否有效
    pool_use_lifo=True,  # 使用后进先出,减少空闲连接
    echo=False  # 不打印 SQL 语句
)

conn = psycopg2.connect(**db_params)
conn.autocommit = True
# Cache token list for longer
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_trading_pairs():
    query = """
    SELECT 
        pair_name 
    FROM 
        trade_pool_pairs 
    WHERE 
        status = 1
    ORDER BY 
        pair_name
    """
    df = pd.read_sql_query(query, engine)
    return df['pair_name'].tolist()

all_tokens = fetch_trading_pairs()

# Create tabs for our app
tab1, tab2 = st.tabs(["Volatility Analysis", "Buffer Rate Comparison"])

with tab1:
    col1, col2 = st.columns([3, 1])

    with col1:
        default_token = "BTC/USDT" if "BTC/USDT" in all_tokens else all_tokens[0]
        selected_token = st.selectbox(
            "Select Token",
            all_tokens,
            index=all_tokens.index(default_token) if default_token in all_tokens else 0,
            key="token_select_tab1"
        )

    with col2:
        if st.button("Refresh Data", type="primary", use_container_width=True, key="refresh_button_tab1"):
            st.cache_data.clear()
            st.rerun()

    sg_tz = pytz.timezone('Asia/Singapore')
    now_sg = datetime.now(sg_tz)
    st.write(f"Current time (Singapore): {now_sg.strftime('%Y-%m-%d %H:%M:%S')}")

# Optimized Rollbit fetch - resample to 10 seconds
@st.cache_data(ttl=60)
def fetch_rollbit_parameters_10sec(token, hours=3):
    """Fetch Rollbit parameters with 10-second resolution"""
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
        FROM 
            rollbit_pair_config
        WHERE 
            pair_name = '{token}'
            AND created_at >= '{start_str}'::timestamp - INTERVAL '8 hour'
            AND created_at <= '{end_str}'::timestamp - INTERVAL '8 hour'
        ORDER BY 
            created_at
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

# Fetch Surf parameters with 10-second resolution
@st.cache_data(ttl=60)
def fetch_surf_parameters_10sec(token, hours=3):
    """Fetch Surf parameters with 10-second resolution"""
    try:
        now_sg = datetime.now(pytz.timezone('Asia/Singapore'))
        start_time_sg = now_sg - timedelta(hours=hours)
        
        start_str = start_time_sg.strftime("%Y-%m-%d %H:%M:%S")
        end_str = now_sg.strftime("%Y-%m-%d %H:%M:%S")

        query = f"""
        SELECT
            pair_name,
            buffer_rate,
            position_multiplier,
            created_at + INTERVAL '8 hour' AS timestamp
        FROM 
            trade_pair_risk_history
        WHERE 
            pair_name = '{token}'
            AND created_at >= '{start_str}'::timestamp - INTERVAL '8 hour'
            AND created_at <= '{end_str}'::timestamp - INTERVAL '8 hour'
        ORDER BY 
            created_at
        """

        df = pd.read_sql_query(query, engine)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').sort_index()
            # Resample to 10 seconds
            df = df.resample('10s').ffill()
        return df
    except Exception as e:
        st.error(f"Error fetching Surf parameters: {e}")
        return None

# 10-second volatility calculation
@st.cache_data(ttl=30)
def get_volatility_data_10sec(token, hours=3):
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
    FROM 
        public.oracle_price_log_partition_{today_str}
    WHERE 
        created_at >= '{start_time_str}'::timestamp - INTERVAL '8 hour'
        AND created_at <= '{end_time_str}'::timestamp - INTERVAL '8 hour'
        AND source_type = 0
        AND pair_name = '{token}'
    ORDER BY 
        timestamp
    """
    
    try:
        df = pd.read_sql_query(query, engine)
        
        # If we don't have enough data, try yesterday's partition too
        if df.empty or len(df) < 10:
            query_yesterday = f"""
            SELECT
                created_at + INTERVAL '8 hour' AS timestamp,
                final_price
            FROM 
                public.oracle_price_log_partition_{yesterday_str}
            WHERE 
                created_at >= '{start_time_str}'::timestamp - INTERVAL '8 hour'
                AND created_at <= '{end_time_str}'::timestamp - INTERVAL '8 hour'
                AND source_type = 0
                AND pair_name = '{token}'
            ORDER BY 
                timestamp
            """
            try:
                df_yesterday = pd.read_sql_query(query_yesterday, engine)
                df = pd.concat([df_yesterday, df]).drop_duplicates().sort_values('timestamp')
            except:
                pass
    except Exception as e:
        st.error(f"Query error for {token}: {str(e)}")
        return None, None
    
    if df.empty:
        return None, None
    
    # Process timestamps
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()
    
    # Resample to 500ms
    price_data = df['final_price'].resample('500ms').ffill().dropna()
    
    if len(price_data) < 2:
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

# Main chart section in Tab 1
with tab1:
    with st.spinner(f"Loading data for {selected_token}..."):
        vol_data, percentiles = get_volatility_data_10sec(selected_token)
        rollbit_params = fetch_rollbit_parameters_10sec(selected_token)

    if vol_data is not None and not vol_data.empty:
        # Convert to percentage
        vol_data_pct = vol_data.copy()
        vol_data_pct['realized_vol'] = vol_data_pct['realized_vol'] * 100
        vol_data_pct = vol_data_pct.sort_index()

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
                f"{selected_token} Annualized Volatility (10sec windows)",
                "Rollbit Buffer Rate (%)",
                "Rollbit Position Multiplier"
            ),
            row_heights=[0.4, 0.3, 0.3]
        )

        # 1. 波动率图表（只用vol_data_pct）
        fig.add_trace(
            go.Scatter(
                x=vol_data_pct.index,
                y=vol_data_pct['realized_vol'],
                mode='lines',
                line=dict(color='blue', width=2),
                name="Volatility (%)",
                hovertemplate="<b>Time: %{x}</b><br>Volatility: %{y:.1f}%<extra></extra>",
                showlegend=False
            ),
            row=1, col=1
        )

        # 2. Rollbit Buffer Rate & Position Multiplier（只用rollbit_params）
        if rollbit_params is not None and not rollbit_params.empty:
            rollbit_params = rollbit_params.sort_index()
            fig.add_trace(
                go.Scatter(
                    x=rollbit_params.index,
                    y=rollbit_params['buffer_rate'] * 100,
                    mode='lines+markers',
                    line=dict(color='darkgreen', width=3),
                    marker=dict(size=4),
                    name="Buffer Rate (%)",
                    hovertemplate="<b>Time: %{x}</b><br>Buffer Rate: %{y:.3f}%<extra></extra>",
                    showlegend=False
                ),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=rollbit_params.index,
                    y=rollbit_params['position_multiplier'],
                    mode='lines+markers',
                    line=dict(color='darkblue', width=3),
                    marker=dict(size=4),
                    name="Position Multiplier",
                    hovertemplate="<b>Time: %{x}</b><br>Position Mult: %{y:,.0f}<extra></extra>",
                    showlegend=False
                ),
                row=3, col=1
            )
            latest_buffer = rollbit_params['buffer_rate'].iloc[-1] * 100
            latest_pos_mult = rollbit_params['position_multiplier'].iloc[-1]
            buffer_min = rollbit_params['buffer_rate'].min() * 100
            buffer_max = rollbit_params['buffer_rate'].max() * 100
            pos_mult_min = rollbit_params['position_multiplier'].min()
            pos_mult_max = rollbit_params['position_multiplier'].max()
        else:
            fig.add_annotation(
                x=0.5, y=0.5, text="No Rollbit data available", showarrow=False,
                font=dict(size=12), xref="x2 domain", yref="y2 domain", row=2, col=1
            )
            fig.add_annotation(
                x=0.5, y=0.5, text="No Rollbit data available", showarrow=False,
                font=dict(size=12), xref="x3 domain", yref="y3 domain", row=3, col=1
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
            title_text = f"{selected_token} Analysis Dashboard (10-second windows)<br>" + \
                        f"<sub>Current Vol: {current_vol:.1f}% ({current_percentile:.0f}th percentile) | Buffer: {latest_buffer:.3f}% | Pos Mult: {latest_pos_mult:,.0f}</sub>"
        else:
            title_text = f"{selected_token} Analysis Dashboard (10-second windows)<br>" + \
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
            xaxis=dict(
                showspikes=True,
                spikemode="across",
                spikesnap="cursor",
                spikethickness=2,
                spikecolor="gray",
                spikedash="solid"
            )
        )
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
        fig.update_xaxes(
            range=[vol_data_pct.index.min(), vol_data_pct.index.max()],
            title_text="Time (Singapore)", row=3, col=1, tickformat="%H:%M:%S<br>%m/%d"
        )
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
        if rollbit_params is not None and not rollbit_params.empty and latest_buffer is not None:
            st.markdown("### Current Rollbit Parameters")
            rcol1, rcol2 = st.columns(2)
            with rcol1:
                st.metric("Current Buffer Rate", f"{latest_buffer:.3f}%")
            with rcol2:
                st.metric("Current Position Multiplier", f"{latest_pos_mult:,.0f}")
    else:
        st.error("No volatility data available for the selected token")

# Buffer Comparison Tab
with tab2:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        default_token = "BTC/USDT" if "BTC/USDT" in all_tokens else all_tokens[0]
        selected_token_tab2 = st.selectbox(
            "Select Token",
            all_tokens,
            index=all_tokens.index(default_token) if default_token in all_tokens else 0,
            key="token_select_tab2"
        )
    
    with col2:
        hours_to_show = st.selectbox(
            "Time Window", 
            [1, 3, 6, 12, 24],
            index=1,
            key="hours_selector"
        )
        
        if st.button("Refresh Data", type="primary", use_container_width=True, key="refresh_button_tab2"):
            st.cache_data.clear()
            st.rerun()
            
    st.write(f"Current time (Singapore): {now_sg.strftime('%Y-%m-%d %H:%M:%S')}")
    
    with st.spinner(f"Loading comparison data for {selected_token_tab2}..."):
        # Fetch both Rollbit and Surf parameters
        rollbit_params = fetch_rollbit_parameters_10sec(selected_token_tab2, hours=hours_to_show)
        surf_params = fetch_surf_parameters_10sec(selected_token_tab2, hours=hours_to_show)
    
    if (rollbit_params is not None and not rollbit_params.empty and 
        surf_params is not None and not surf_params.empty):
        
        # Merge the dataframes to align timestamps
        # First create common datetime index
        start_time = min(rollbit_params.index.min(), surf_params.index.min())
        end_time = max(rollbit_params.index.max(), surf_params.index.max())
        common_index = pd.date_range(start=start_time, end=end_time, freq='10s')
        
        # Reindex both dataframes to this common index
        rollbit_reindexed = rollbit_params.reindex(common_index, method='ffill')
        surf_reindexed = surf_params.reindex(common_index, method='ffill')
        
        # Merge them into a single dataframe
        comparison_df = pd.DataFrame({
            'timestamp': common_index,
            'rollbit_buffer': rollbit_reindexed['buffer_rate'],
            'surf_buffer': surf_reindexed['buffer_rate'],
            'rollbit_pos_mult': rollbit_reindexed['position_multiplier'],
            'surf_pos_mult': surf_reindexed['position_multiplier']
        }).set_index('timestamp')
        
        # Calculate buffer rate difference (as percentage points)
        comparison_df['buffer_diff'] = (comparison_df['surf_buffer'] - comparison_df['rollbit_buffer']) * 100
        comparison_df['buffer_ratio'] = comparison_df['surf_buffer'] / comparison_df['rollbit_buffer']
        
        # Calculate position multiplier ratio
        comparison_df['pos_mult_ratio'] = comparison_df['surf_pos_mult'] / comparison_df['rollbit_pos_mult']
        
        # Convert buffer rates to percentages for display
        comparison_df['rollbit_buffer_pct'] = comparison_df['rollbit_buffer'] * 100
        comparison_df['surf_buffer_pct'] = comparison_df['surf_buffer'] * 100
        
        # Create subplots with 3 rows
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                f"{selected_token_tab2} Buffer Rate Comparison",
                "Buffer Rate Difference (Surf - Rollbit)",
                "Position Multiplier Comparison"
            ),
            row_heights=[0.33, 0.33, 0.34]
        )
        
        # Panel 1: Buffer Rate Comparison
        fig.add_trace(
            go.Scatter(
                x=comparison_df.index,
                y=comparison_df['rollbit_buffer_pct'],
                mode='lines',
                line=dict(color='red', width=2),
                name="Rollbit Buffer (%)"
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=comparison_df.index,
                y=comparison_df['surf_buffer_pct'],
                mode='lines',
                line=dict(color='blue', width=2),
                name="Surf Buffer (%)"
            ),
            row=1, col=1
        )
        
        # Panel 2: Buffer Rate Difference
        fig.add_trace(
            go.Scatter(
                x=comparison_df.index,
                y=comparison_df['buffer_diff'],
                mode='lines',
                line=dict(color='purple', width=2),
                name="Buffer Diff (pp)",
                fill='tozeroy'
            ),
            row=2, col=1
        )
        
        # Add a horizontal line at zero for reference
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="gray",
            line_width=1,
            row=2, col=1
        )
        
        # Panel 3: Position Multiplier Comparison
        fig.add_trace(
            go.Scatter(
                x=comparison_df.index,
                y=comparison_df['rollbit_pos_mult'],
                mode='lines',
                line=dict(color='red', width=2),
                name="Rollbit Pos Mult"
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=comparison_df.index,
                y=comparison_df['surf_pos_mult'],
                mode='lines',
                line=dict(color='blue', width=2),
                name="Surf Pos Mult"
            ),
            row=3, col=1
        )
        
        # Get latest values for metrics
        latest_rollbit_buffer = comparison_df['rollbit_buffer_pct'].iloc[-1]
        latest_surf_buffer = comparison_df['surf_buffer_pct'].iloc[-1]
        latest_buffer_diff = comparison_df['buffer_diff'].iloc[-1]
        latest_buffer_ratio = comparison_df['buffer_ratio'].iloc[-1]
        
        latest_rollbit_pos = comparison_df['rollbit_pos_mult'].iloc[-1]
        latest_surf_pos = comparison_df['surf_pos_mult'].iloc[-1]
        latest_pos_ratio = comparison_df['pos_mult_ratio'].iloc[-1]
        
        # Create title
        title_text = f"{selected_token_tab2} Buffer Rate Comparison<br>" + \
                    f"<sub>Current Buffer: Surf {latest_surf_buffer:.3f}% vs Rollbit {latest_rollbit_buffer:.3f}% | Diff: {latest_buffer_diff:+.3f}pp | Ratio: {latest_buffer_ratio:.2f}x</sub>"
        
        # Update layout
        fig.update_layout(
            title=title_text,
            height=800,
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
        
        # Update y-axes
        fig.update_yaxes(
            title_text="Buffer Rate (%)",
            row=1, col=1,
            tickformat=".3f",
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray'
        )
        
        fig.update_yaxes(
            title_text="Difference (percentage points)",
            row=2, col=1,
            tickformat=".3f",
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray'
        )
        
        fig.update_yaxes(
            title_text="Position Multiplier",
            row=3, col=1,
            tickformat=",",
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray'
        )
        
        # X-axis labels only on bottom
        fig.update_xaxes(title_text="Time (Singapore)", row=3, col=1, tickformat="%H:%M:%S<br>%m/%d")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Current metrics display
        st.markdown("### Current Buffer Rate Comparison")
        bcol1, bcol2, bcol3 = st.columns(3)
        with bcol1:
            st.metric("Surf Buffer", f"{latest_surf_buffer:.3f}%")
        with bcol2:
            st.metric("Rollbit Buffer", f"{latest_rollbit_buffer:.3f}%")
        with bcol3:
            st.metric("Difference (Surf - Rollbit)", f"{latest_buffer_diff:+.3f}pp")
            
        st.markdown("### Current Position Multiplier Comparison")
        pcol1, pcol2, pcol3 = st.columns(3)
        with pcol1:
            st.metric("Surf Position Mult", f"{latest_surf_pos:,.0f}")
        with pcol2:
            st.metric("Rollbit Position Mult", f"{latest_rollbit_pos:,.0f}")
        with pcol3:
            # Format the ratio metric with + if Surf > Rollbit
            ratio_text = f"{latest_pos_ratio:.2f}x"
            ratio_delta = f"{'+' if latest_pos_ratio > 1 else ''}{(latest_pos_ratio - 1) * 100:.1f}%"
            st.metric("Ratio (Surf ÷ Rollbit)", ratio_text, ratio_delta)
            
    else:
        st.warning(f"Insufficient data to compare Surf and Rollbit parameters for {selected_token_tab2}.")
        
        # Check which one is missing
        if rollbit_params is None or rollbit_params.empty:
            st.error("No Rollbit parameter data available.")
        if surf_params is None or surf_params.empty:
            st.error("No Surf parameter data available.")
            
        st.info("Try selecting a different token or time window, or check if both platforms have data for this token.")
