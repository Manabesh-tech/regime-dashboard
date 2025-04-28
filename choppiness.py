import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import pytz
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager

# Clear cache at startup to ensure fresh data
st.cache_data.clear()

# Page configuration
st.set_page_config(
    page_title="Crypto Tick Data Plotter",
    page_icon="📈",
    layout="wide",
)

# Enhanced CSS for better UI
st.markdown("""
<style>
    .main .block-container {max-width: 98% !important; padding-top: 1rem !important;}
    h1, h2, h3 {margin-bottom: 0.5rem !important;}
    .stButton > button {width: 100%; font-weight: bold; height: 46px; font-size: 18px;}
    
    /* Improved tabs styling */
    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px;
        color: #000000;
        font-size: 18px;
        font-weight: 600;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 0 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4 !important;
        color: white !important;
    }
    
    /* Info box styling */
    .time-info {
        background-color: #e7f1ff;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    
    /* Selection container styling */
    .selection-container {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    
    /* Progress bar styling */
    div.stProgress > div > div {height: 8px !important;}
    
    /* Plot container styling */
    .plot-container {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

# Database configuration
DB_CONFIG = {
    'replication': {
        'url': "postgresql://public_replication:866^FKC4hllk@aws-jp-tk-surf-pg-public.cluster-csteuf9lw8dv.ap-northeast-1.rds.amazonaws.com:5432/replication_report"
    }
}

# Create database engine
@st.cache_resource
def get_engine():
    """Create database engine"""
    try:
        return create_engine(DB_CONFIG['replication']['url'], pool_size=5, max_overflow=10)
    except Exception as e:
        st.error(f"Error creating database engine: {e}")
        return None

@contextmanager
def get_session():
    """Database session context manager"""
    engine = get_engine()
    if not engine:
        yield None
        return
        
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    except Exception as e:
        st.error(f"Database error: {e}")
        session.rollback()
    finally:
        session.close()

# Pre-defined pairs as a fallback
PREDEFINED_PAIRS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", 
    "AVAX/USDT", "DOGE/USDT", "ADA/USDT", "TRX/USDT", "DOT/USDT"
]

# Get available pairs from the database
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_available_pairs():
    try:
        with get_session() as session:
            if not session:
                return PREDEFINED_PAIRS
            
            query = text("SELECT pair_name FROM trade_pool_pairs WHERE status = 1")
            result = session.execute(query)
            pairs = [row[0] for row in result]
            
            return sorted(pairs) if pairs else PREDEFINED_PAIRS
    
    except Exception as e:
        st.error(f"Error fetching available pairs: {e}")
        return PREDEFINED_PAIRS

def get_partition_tables(session, start_date, end_date):
    """
    Get list of partition tables that need to be queried based on date range.
    Returns a list of table names (oracle_price_log_partition_YYYYMMDD)
    """
    # Ensure we have datetime objects
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    # Ensure timezone is properly set to Singapore
    singapore_tz = pytz.timezone('Asia/Singapore')
    if start_date.tzinfo is None:
        start_date = singapore_tz.localize(start_date)
    if end_date.tzinfo is None:
        end_date = singapore_tz.localize(end_date)
    
    # Convert to Singapore time
    start_date = start_date.astimezone(singapore_tz)
    end_date = end_date.astimezone(singapore_tz)
    
    # Remove timezone for database compatibility
    start_date = start_date.replace(tzinfo=None)
    end_date = end_date.replace(tzinfo=None)
    
    # Generate list of dates between start and end
    dates = []
    current_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_date_day = end_date.replace(hour=0, minute=0, second=0, microsecond=0)
    
    while current_date <= end_date_day:
        dates.append(current_date.strftime("%Y%m%d"))
        current_date += timedelta(days=1)
    
    # Create table names
    table_names = [f"oracle_price_log_partition_{date}" for date in dates]
    
    # Verify which tables actually exist
    existing_tables = []
    
    for table in table_names:
        check_table = text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = %s
            );
        """)
        
        if session.execute(check_table, (table,)).scalar():
            existing_tables.append(table)
    
    return existing_tables

def build_union_query(tables, pair_name, start_time, end_time):
    """Build a UNION query across multiple partition tables."""
    if not tables:
        return ""
    
    # Format dates for query
    if isinstance(start_time, datetime):
        start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
    else:
        start_time_str = start_time
        
    if isinstance(end_time, datetime):
        end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
    else:
        end_time_str = end_time
    
    union_parts = []
    
    for table in tables:
        query = f"""
        SELECT 
            pair_name,
            created_at + INTERVAL '8 hour' AS timestamp_sgt,
            final_price AS price
        FROM 
            public.{table}
        WHERE 
            created_at >= '{start_time_str}'::timestamp - INTERVAL '8 hour'
            AND created_at <= '{end_time_str}'::timestamp - INTERVAL '8 hour'
            AND pair_name = '{pair_name}'
        """
        union_parts.append(query)
    
    # Join with UNION and add ORDER BY
    complete_query = " UNION ".join(union_parts) + " ORDER BY timestamp_sgt DESC"
    return complete_query

def fetch_tick_data(pair_name, hours=24, num_ticks=5000, progress_callback=None):
    """
    Fetch tick data for a specific pair.
    
    Args:
        pair_name: The cryptocurrency pair to fetch
        hours: Hours to look back (default 24)
        num_ticks: Number of most recent ticks to fetch (default 5000)
        progress_callback: Optional callback for progress updates
        
    Returns:
        DataFrame with tick data and analysis info
    """
    try:
        with get_session() as session:
            if not session:
                return None
            
            # Calculate time range in Singapore time
            singapore_tz = pytz.timezone('Asia/Singapore')
            now = datetime.now(singapore_tz)
            start_time = now - timedelta(hours=hours)
            
            # Format for display and database
            start_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
            end_str = now.strftime("%Y-%m-%d %H:%M:%S")
            
            # Update progress if callback provided
            if progress_callback:
                progress_callback(0.1, f"Analyzing data from {start_str} to {end_str} (SGT)")
            
            # Get relevant partition tables
            tables = get_partition_tables(session, start_time, now)
            
            if progress_callback:
                progress_callback(0.2, f"Found {len(tables)} partition tables to query")
            
            if not tables:
                if progress_callback:
                    progress_callback(0.2, "No partition tables found for the specified time range")
                return None
            
            # Build and execute query
            query = build_union_query(tables, pair_name, start_str, end_str)
            
            if progress_callback:
                progress_callback(0.4, f"Fetching data for {pair_name}...")
            
            if not query:
                return None
                
            result = session.execute(text(query))
            data = result.fetchall()
            
            if not data:
                if progress_callback:
                    progress_callback(0.5, "No data found for the specified pair")
                return None
            
            # Create DataFrame
            df = pd.DataFrame(data, columns=['pair_name', 'timestamp_sgt', 'price'])
            
            # Convert price to numeric
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            
            # Convert timestamp to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp_sgt']):
                df['timestamp_sgt'] = pd.to_datetime(df['timestamp_sgt'])
            
            # Sort by timestamp (newest first) and take the specified number of ticks
            df = df.sort_values('timestamp_sgt', ascending=False)
            
            if len(df) > num_ticks:
                df = df.iloc[:num_ticks]
            
            if progress_callback:
                progress_callback(0.8, f"Processing {len(df)} ticks...")
            
            # Analyze the data
            if len(df) > 1:
                # Calculate tick-to-tick changes
                df['price_change'] = df['price'].diff(-1)  # Negative diff because newest is first
                df['pct_change'] = (df['price_change'] / df['price'].shift(-1)) * 100
                
                # Calculate time between ticks
                df['time_diff'] = df['timestamp_sgt'].diff(-1).dt.total_seconds()
                
                # Get timestamp range
                newest_time = df['timestamp_sgt'].iloc[0]
                oldest_time = df['timestamp_sgt'].iloc[-1]
                total_duration = (newest_time - oldest_time).total_seconds()
                
                # Calculate statistics
                tick_stats = {
                    'pair': pair_name,
                    'count': len(df),
                    'newest_time': newest_time.strftime("%Y-%m-%d %H:%M:%S"),
                    'oldest_time': oldest_time.strftime("%Y-%m-%d %H:%M:%S"),
                    'duration_seconds': total_duration,
                    'duration_formatted': f"{total_duration//3600:.0f}h {(total_duration%3600)//60:.0f}m {total_duration%60:.0f}s",
                    'avg_tick_seconds': df['time_diff'].mean(),
                    'min_price': df['price'].min(),
                    'max_price': df['price'].max(),
                    'volatility': df['pct_change'].std()
                }
            else:
                tick_stats = {
                    'pair': pair_name,
                    'count': len(df),
                    'newest_time': None,
                    'oldest_time': None,
                    'duration_seconds': 0,
                    'duration_formatted': "N/A",
                    'avg_tick_seconds': 0,
                    'min_price': df['price'].min() if len(df) > 0 else None,
                    'max_price': df['price'].max() if len(df) > 0 else None,
                    'volatility': 0
                }
            
            if progress_callback:
                progress_callback(1.0, "Data processing complete")
            
            return {
                'data': df,
                'stats': tick_stats
            }
    
    except Exception as e:
        if progress_callback:
            progress_callback(1.0, f"Error: {str(e)}")
        st.error(f"Error fetching tick data: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None

def create_price_plot(data_dict, plot_title=None, height=600):
    """Create an interactive price chart with Plotly"""
    if not data_dict or 'data' not in data_dict:
        return None
    
    df = data_dict['data']
    stats = data_dict['stats']
    
    if df is None or len(df) < 2:
        return None
    
    # Create a subplot with 2 rows
    fig = make_subplots(
        rows=2, 
        cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=(
            f"{stats['pair']} Price Chart ({stats['count']} ticks)", 
            "Tick Time Interval (seconds)"
        )
    )
    
    # Reverse order for better display (oldest to newest)
    df = df.sort_values('timestamp_sgt')
    
    # Add price line
    fig.add_trace(
        go.Scatter(
            x=df['timestamp_sgt'],
            y=df['price'],
            mode='lines',
            name='Price',
            line=dict(color='blue'),
        ),
        row=1, col=1
    )
    
    # Add time diff scatter plot
    fig.add_trace(
        go.Scatter(
            x=df['timestamp_sgt'][:-1],
            y=df['time_diff'],
            mode='markers',
            name='Tick Interval',
            marker=dict(
                color='orange',
                size=5,
                opacity=0.7
            ),
        ),
        row=2, col=1
    )
    
    # Add a horizontal line for average tick time
    fig.add_shape(
        type="line",
        x0=df['timestamp_sgt'].iloc[0],
        y0=stats['avg_tick_seconds'],
        x1=df['timestamp_sgt'].iloc[-1],
        y1=stats['avg_tick_seconds'],
        line=dict(
            color="red",
            width=1,
            dash="dash",
        ),
        row=2, col=1
    )
    
    # Customize layout
    fig.update_layout(
        height=height,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        hovermode="x unified",
        xaxis=dict(
            title="",
            gridcolor='lightgray',
            showgrid=True,
            zeroline=False,
        ),
        yaxis=dict(
            title="Price",
            gridcolor='lightgray',
            showgrid=True,
            zeroline=False,
        ),
        xaxis2=dict(
            title="Singapore Time (SGT)",
            gridcolor='lightgray',
            showgrid=True,
            zeroline=False,
        ),
        yaxis2=dict(
            title="Seconds",
            gridcolor='lightgray',
            showgrid=True,
            zeroline=False,
        ),
        margin=dict(l=50, r=50, t=50, b=50),
        plot_bgcolor='white',
    )
    
    # Add a title if provided
    if plot_title:
        fig.update_layout(title=dict(text=plot_title, x=0.5, xanchor='center'))
    
    return fig

def plot_tick_comparison(data_dict_a, data_dict_b=None, height=700):
    """Create a comparison plot between two pairs or a single pair"""
    # If only one dataset is provided
    if data_dict_b is None:
        return create_price_plot(data_dict_a, height=height)
    
    # Check if we have valid data for both pairs
    if not data_dict_a or not data_dict_b or 'data' not in data_dict_a or 'data' not in data_dict_b:
        return None
    
    df_a = data_dict_a['data']
    stats_a = data_dict_a['stats']
    df_b = data_dict_b['data']
    stats_b = data_dict_b['stats']
    
    if df_a is None or df_b is None or len(df_a) < 2 or len(df_b) < 2:
        return None
    
    # Create a subplot with 2 rows
    fig = make_subplots(
        rows=2, 
        cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.6, 0.4],
        subplot_titles=(
            f"Normalized Price Comparison ({stats_a['pair']} vs {stats_b['pair']})", 
            "Tick Time Intervals (seconds)"
        )
    )
    
    # Reverse order for better display (oldest to newest)
    df_a = df_a.sort_values('timestamp_sgt')
    df_b = df_b.sort_values('timestamp_sgt')
    
    # Normalize prices for better comparison
    df_a = df_a.copy()
    df_b = df_b.copy()
    df_a['norm_price'] = df_a['price'] / df_a['price'].iloc[0] * 100
    df_b['norm_price'] = df_b['price'] / df_b['price'].iloc[0] * 100
    
    # Add normalized price lines
    fig.add_trace(
        go.Scatter(
            x=df_a['timestamp_sgt'],
            y=df_a['norm_price'],
            mode='lines',
            name=f"{stats_a['pair']} (normalized)",
            line=dict(color='blue'),
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df_b['timestamp_sgt'],
            y=df_b['norm_price'],
            mode='lines',
            name=f"{stats_b['pair']} (normalized)",
            line=dict(color='red'),
        ),
        row=1, col=1
    )
    
    # Add time diff scatter plots
    fig.add_trace(
        go.Scatter(
            x=df_a['timestamp_sgt'][:-1],
            y=df_a['time_diff'],
            mode='markers',
            name=f"{stats_a['pair']} Tick Interval",
            marker=dict(
                color='blue',
                size=4,
                opacity=0.7
            ),
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df_b['timestamp_sgt'][:-1],
            y=df_b['time_diff'],
            mode='markers',
            name=f"{stats_b['pair']} Tick Interval",
            marker=dict(
                color='red',
                size=4,
                opacity=0.7
            ),
        ),
        row=2, col=1
    )
    
    # Add horizontal lines for average tick times
    fig.add_shape(
        type="line",
        x0=min(df_a['timestamp_sgt'].iloc[0], df_b['timestamp_sgt'].iloc[0]),
        y0=stats_a['avg_tick_seconds'],
        x1=max(df_a['timestamp_sgt'].iloc[-1], df_b['timestamp_sgt'].iloc[-1]),
        y1=stats_a['avg_tick_seconds'],
        line=dict(
            color="blue",
            width=1,
            dash="dash",
        ),
        row=2, col=1
    )
    
    fig.add_shape(
        type="line",
        x0=min(df_a['timestamp_sgt'].iloc[0], df_b['timestamp_sgt'].iloc[0]),
        y0=stats_b['avg_tick_seconds'],
        x1=max(df_a['timestamp_sgt'].iloc[-1], df_b['timestamp_sgt'].iloc[-1]),
        y1=stats_b['avg_tick_seconds'],
        line=dict(
            color="red",
            width=1,
            dash="dash",
        ),
        row=2, col=1
    )
    
    # Add annotations for average tick times
    fig.add_annotation(
        x=max(df_a['timestamp_sgt'].iloc[-1], df_b['timestamp_sgt'].iloc[-1]),
        y=stats_a['avg_tick_seconds'],
        text=f"Avg: {stats_a['avg_tick_seconds']:.2f}s",
        showarrow=False,
        font=dict(color="blue"),
        xanchor="right",
        row=2, col=1
    )
    
    fig.add_annotation(
        x=max(df_a['timestamp_sgt'].iloc[-1], df_b['timestamp_sgt'].iloc[-1]),
        y=stats_b['avg_tick_seconds'],
        text=f"Avg: {stats_b['avg_tick_seconds']:.2f}s",
        showarrow=False,
        font=dict(color="red"),
        xanchor="right",
        row=2, col=1
    )
    
    # Customize layout
    fig.update_layout(
        height=height,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        hovermode="x unified",
        xaxis=dict(
            title="",
            gridcolor='lightgray',
            showgrid=True,
            zeroline=False,
        ),
        yaxis=dict(
            title="Normalized Price (base=100)",
            gridcolor='lightgray',
            showgrid=True,
            zeroline=False,
        ),
        xaxis2=dict(
            title="Singapore Time (SGT)",
            gridcolor='lightgray', 
            showgrid=True,
            zeroline=False,
        ),
        yaxis2=dict(
            title="Seconds",
            gridcolor='lightgray',
            showgrid=True,
            zeroline=False,
        ),
        margin=dict(l=50, r=50, t=70, b=50),
        plot_bgcolor='white',
    )
    
    return fig

def display_tick_stats(data_dict):
    """Display statistics about the tick data"""
    if not data_dict or 'stats' not in data_dict:
        return
    
    stats = data_dict['stats']
    
    # Create a formatted display of statistics
    st.markdown(f"""
    <div class="time-info">
        <h4>Tick Data Statistics: {stats['pair']}</h4>
        <p><strong>Total Ticks:</strong> {stats['count']}</p>
        <p><strong>Time Range:</strong> {stats['oldest_time']} to {stats['newest_time']} (SGT)</p>
        <p><strong>Duration:</strong> {stats['duration_formatted']} ({stats['duration_seconds']:.1f} seconds)</p>
        <p><strong>Average Time Between Ticks:</strong> {stats['avg_tick_seconds']:.3f} seconds</p>
        <p><strong>Price Range:</strong> {stats['min_price']} to {stats['max_price']}</p>
        <p><strong>Volatility (StdDev of % changes):</strong> {stats['volatility']:.4f}%</p>
    </div>
    """, unsafe_allow_html=True)

def main():
    # Get current Singapore time
    singapore_tz = pytz.timezone('Asia/Singapore')
    now_sg = datetime.now(singapore_tz)
    current_time_sg = now_sg.strftime("%Y-%m-%d %H:%M:%S")
    
    # Main title
    st.title("Crypto Tick Data Plotter")
    st.markdown(f"<p style='text-align: center; font-size:14px; color:gray;'>Last updated: {current_time_sg} (SGT)</p>", unsafe_allow_html=True)
    
    # Create tabs for different analysis modes
    tab1, tab2 = st.tabs(["Single Pair Analysis", "Pair Comparison"])
    
    # Get available pairs
    available_pairs = get_available_pairs()
    
    # Tab 1: Single Pair Analysis
    with tab1:
        st.markdown("""
        <div class="selection-container">
            <h3>Single Crypto Pair Analysis</h3>
            <p>Select a cryptocurrency pair and analyze its tick data with a detailed price chart and statistics.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            pair = st.selectbox(
                "Select Cryptocurrency Pair",
                options=available_pairs,
                index=0 if "BTC/USDT" in available_pairs else 0
            )
        
        with col2:
            hours = st.number_input(
                "Hours of Data",
                min_value=1,
                max_value=48,
                value=24,
                step=1,
                help="How many hours of data to analyze"
            )
        
        with col3:
            num_ticks = st.number_input(
                "Number of Ticks",
                min_value=100,
                max_value=10000,
                value=5000,
                step=100,
                help="How many of the most recent ticks to analyze"
            )
        
        analyze_button = st.button("Analyze Pair", key="analyze_single")
        
        if analyze_button:
            # Clear cache to ensure fresh data
            st.cache_data.clear()
            
            # Show analysis start time
            analysis_start_time = datetime.now(singapore_tz).strftime("%Y-%m-%d %H:%M:%S")
            st.markdown(f"<p style='text-align: center; font-size:14px; color:green;'>Analysis started at: {analysis_start_time} (SGT)</p>", unsafe_allow_html=True)
            
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Define progress callback function
            def update_progress(progress, text):
                progress_bar.progress(progress, text)
            
            # Fetch and process data
            with st.spinner(f"Fetching tick data for {pair}..."):
                data = fetch_tick_data(pair, hours, num_ticks, update_progress)
            
            if data and 'data' in data and len(data['data']) > 0:
                # Display statistics
                display_tick_stats(data)
                
                # Create and show plot
                fig = create_price_plot(data)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Show data table (hidden by default)
                with st.expander("Show Raw Data Table"):
                    st.dataframe(
                        data['data'].sort_values('timestamp_sgt', ascending=False),
                        use_container_width=True
                    )
                
                # Download button for CSV
                csv = data['data'].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download CSV Data",
                    data=csv,
                    file_name=f"{pair.replace('/', '_')}_tick_data_{now_sg.strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                
                # Show analysis end time
                analysis_end_time = datetime.now(singapore_tz).strftime("%Y-%m-%d %H:%M:%S")
                st.markdown(f"<p style='text-align: center; font-size:14px; color:green;'>Analysis completed at: {analysis_end_time} (SGT)</p>", unsafe_allow_html=True)
            else:
                st.error(f"No tick data found for {pair} in the specified time range.")
    
    # Tab 2: Pair Comparison
    with tab2:
        st.markdown("""
        <div class="selection-container">
            <h3>Compare Two Crypto Pairs</h3>
            <p>Compare tick data between two different cryptocurrency pairs to analyze relative price movements and trading activity.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h4>First Pair</h4>", unsafe_allow_html=True)
            pair1 = st.selectbox(
                "Select First Pair",
                options=available_pairs,
                index=0 if "BTC/USDT" in available_pairs else 0
            )
        
        with col2:
            st.markdown("<h4>Second Pair</h4>", unsafe_allow_html=True)
            pair2 = st.selectbox(
                "Select Second Pair",
                options=available_pairs,
                index=1 if "ETH/USDT" in available_pairs else 1
            )
        
        col3, col4, col5 = st.columns([1, 1, 2])
        
        with col3:
            hours_compare = st.number_input(
                "Hours of Data",
                min_value=1,
                max_value=48,
                value=24,
                step=1,
                help="How many hours of data to analyze",
                key="hours_compare"
            )
        
        with col4:
            ticks_compare = st.number_input(
                "Number of Ticks",
                min_value=100,
                max_value=10000,
                value=5000,
                step=100,
                help="How many of the most recent ticks to analyze",
                key="ticks_compare"
            )
        
        with col5:
            compare_button = st.button("Compare Pairs", use_container_width=True)
        
        if compare_button:
            # Clear cache to ensure fresh data
            st.cache_data.clear()
            
            # Show analysis start time
            analysis_start_time = datetime.now(singapore_tz).strftime("%Y-%m-%d %H:%M:%S")
            st.markdown(f"<p style='text-align: center; font-size:14px; color:green;'>Comparison started at: {analysis_start_time} (SGT)</p>", unsafe_allow_html=True)
            
            # Create progress bars
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown(f"<h4>{pair1}</h4>", unsafe_allow_html=True)
                progress_bar1 = st.progress(0)
                status_text1 = st.empty()
                
                def update_progress1(progress, text):
                    progress_bar1.progress(progress, text)
            
            with col_b:
                st.markdown(f"<h4>{pair2}</h4>", unsafe_allow_html=True)
                progress_bar2 = st.progress(0)
                status_text2 = st.empty()
                
                def update_progress2(progress, text):
                    progress_bar2.progress(progress, text)
            
            # Fetch data for both pairs
            with st.spinner("Fetching and comparing data..."):
                data1 = fetch_tick_data(pair1, hours_compare, ticks_compare, update_progress1)
                data2 = fetch_tick_data(pair2, hours_compare, ticks_compare, update_progress2)
            
            # Check if we have valid data for both pairs
            if (data1 and 'data' in data1 and len(data1['data']) > 0 and 
                data2 and 'data' in data2 and len(data2['data']) > 0):
                
                # Display statistics for both pairs side by side
                col_stats1, col_stats2 = st.columns(2)
                
                with col_stats1:
                    display_tick_stats(data1)
                
                with col_stats2:
                    display_tick_stats(data2)
                
                # Create and display comparison plot
                fig = plot_tick_comparison(data1, data2)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Show analysis end time
                analysis_end_time = datetime.now(singapore_tz).strftime("%Y-%m-%d %H:%M:%S")
                st.markdown(f"<p style='text-align: center; font-size:14px; color:green;'>Comparison completed at: {analysis_end_time} (SGT)</p>", unsafe_allow_html=True)
                
                # Show data tables (hidden by default)
                with st.expander(f"Show Raw Data Tables"):
                    tab_a, tab_b = st.tabs([f"{pair1} Data", f"{pair2} Data"])
                    
                    with tab_a:
                        st.dataframe(
                            data1['data'].sort_values('timestamp_sgt', ascending=False),
                            use_container_width=True
                        )
                        
                        # Download button for CSV
                        csv1 = data1['data'].to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label=f"Download {pair1} CSV Data",
                            data=csv1,
                            file_name=f"{pair1.replace('/', '_')}_tick_data_{now_sg.strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                            key="download1"
                        )
                    
                    with tab_b:
                        st.dataframe(
                            data2['data'].sort_values('timestamp_sgt', ascending=False),
                            use_container_width=True
                        )
                        
                        # Download button for CSV
                        csv2 = data2['data'].to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label=f"Download {pair2} CSV Data",
                            data=csv2,
                            file_name=f"{pair2.replace('/', '_')}_tick_data_{now_sg.strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                            key="download2"
                        )
            else:
                if not data1 or 'data' not in data1 or len(data1['data']) == 0:
                    st.error(f"No tick data found for {pair1} in the specified time range.")
                if not data2 or 'data' not in data2 or len(data2['data']) == 0:
                    st.error(f"No tick data found for {pair2} in the specified time range.")

    # Add footer information
    st.markdown("""
    <div style='margin-top: 30px; padding: 10px; border-top: 1px solid #ddd; text-align: center; color: #666;'>
        <p>This dashboard fetches and analyzes cryptocurrency tick data with automatic timezone handling and table transitions.</p>
        <p>All times are displayed in Singapore Time (SGT/UTC+8).</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()