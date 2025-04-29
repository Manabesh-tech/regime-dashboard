import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
import time
import altair as alt
import threading

# Clear cache at startup to ensure fresh data
st.cache_data.clear()

# Page configuration
st.set_page_config(
    page_title="Cross-Exchange Depth Tier Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced CSS for better table readability
st.markdown("""
<style>
    .block-container {padding: 0 !important;}
    .main .block-container {max-width: 98% !important;}
    h1, h2, h3 {margin: 0 !important; padding: 0 !important;}
    .stButton > button {width: 100%; font-weight: bold; height: 46px; font-size: 18px;}
    div.stProgress > div > div {height: 5px !important;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Table styling */
    .dataframe {
        font-size: 18px !important;
        width: 100% !important;
    }

    .dataframe th {
        font-weight: 700 !important;
        background-color: #f0f2f6 !important;
    }

    .dataframe td {
        font-weight: 500 !important;
    }

    /* Highlight top tiers */
    .dataframe tr:nth-child(1) {
        background-color: #d4f7d4 !important; /* Green for top tier */
    }
    .dataframe tr:nth-child(2) {
        background-color: #e6f7ff !important; /* Light blue for second tier */
    }
    .dataframe tr:nth-child(3) {
        background-color: #fff9e6 !important; /* Light yellow for third tier */
    }
    
    /* Exchange tag styling */
    .exchange-tag {
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: bold;
        margin-right: 5px;
    }
    .exchange-binance {
        background-color: #f3ba2f;
        color: black;
    }
    .exchange-hyperliquid {
        background-color: #333333;
        color: white;
    }
    .exchange-bybit {
        background-color: #b50c5c;
        color: white;
    }
    .exchange-okx {
        background-color: #121212;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Database configuration
DB_CONFIG = {
    'report_dev': {
        'url': "postgresql://[DB_USERNAME]:[DB_PASSWORD]@[DB_HOST]:5432/report_dev"
    }
}

# Create database engine
@st.cache_resource
def get_engine(db_name='report_dev'):
    """Create database engine
    Args:
        db_name (str): Database name, default is 'report_dev'
    Returns:
        engine: SQLAlchemy engine
    """
    try:
        # For now, use direct connection details to avoid setup complexity
        # In production, these should be moved to Streamlit secrets
        db_url = "postgresql://public_rw:aTJ92^kl04hllk@aws-jp-tk-surf-pg-public.cluster-csteuf9lw8dv.ap-northeast-1.rds.amazonaws.com:5432/report_dev"
        return create_engine(db_url, pool_size=5, max_overflow=10)
    except Exception as e:
        st.error(f"Error creating database engine: {e}")
        return None

@contextmanager
def get_session(db_name='report_dev'):
    """Database session context manager
    Args:
        db_name (str): Database name, default is 'report_dev'
    Yields:
        session: SQLAlchemy session
    """
    engine = get_engine(db_name)
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

# Get available pairs from the database
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_available_pairs():
    """Fetch available trading pairs from the database
    Returns:
        list: List of available trading pairs
    """
    try:
        with get_session() as session:
            if not session:
                return []

            # Query the oracle_exchange_price table to get unique pairs
            query = text("""
                SELECT DISTINCT pair_name 
                FROM uat_oracle_exchange_price_partition_v1_20250429
                WHERE created_at >= NOW() - INTERVAL '24 hours'
                ORDER BY pair_name
            """)
            
            result = session.execute(query)
            pairs = [row[0] for row in result]
            return sorted(pairs) if pairs else []

    except Exception as e:
        st.error(f"Error fetching available pairs: {e}")
        return []

# Get available exchanges from the database
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_available_exchanges():
    """Fetch available exchanges from the database
    Returns:
        list: List of available exchanges
    """
    try:
        with get_session() as session:
            if not session:
                return []

            # Query the oracle_exchange_price table to get unique exchanges
            query = text("""
                SELECT DISTINCT exchange_name 
                FROM uat_oracle_exchange_price_partition_v1_20250429
                WHERE created_at >= NOW() - INTERVAL '24 hours'
                ORDER BY exchange_name
            """)
            
            result = session.execute(query)
            exchanges = [row[0] for row in result]
            return exchanges if exchanges else []

    except Exception as e:
        st.error(f"Error fetching available exchanges: {e}")
        return []

# Class to handle cross-exchange depth tier analysis
class CrossExchangeAnalyzer:
    """Analyzer for depth tiers across multiple exchanges"""
    
    def __init__(self):
        # Fixed point count (5000 points as requested)
        self.point_count = 5000
        
        # Map depth tiers to their nominal values
        self.depth_tier_values = {
            '1k': 1000,
            '3k': 3000,
            '5k': 5000,
            '7k': 7000,
            '10k': 10000,
            '50k': 50000,
            '100k': 100000,
            '200k': 200000,
            '300k': 300000,
            '400k': 400000,
            '500k': 500000,
            '600k': 600000,
            '700k': 700000,
            '800k': 800000,
            '900k': 900000,
            '1000k': 1000000,
            '2000k': 2000000,
            '3000k': 3000000,
            '4000k': 4000000,
            '5000k': 5000000,
            '6000k': 6000000,
            '7000k': 7000000,
            '8000k': 8000000,
            '9000k': 9000000,
            '10000k': 10000000,
        }
        
        # Reverse mapping for easier lookup
        self.nominal_to_tier = {v: k for k, v in self.depth_tier_values.items()}
        
        # Store analysis results
        self.results = None
        self.analysis_time_range = None
        self.last_updated = None
        
    def fetch_and_analyze(self, pair_name, hours=24, progress_bar=None):
        """Fetch data and analyze depth tiers across all exchanges
        
        Args:
            pair_name: Cryptocurrency pair to analyze
            hours: Hours of data to look back (default 24 hours)
            progress_bar: Optional progress bar to update
        
        Returns:
            bool: Success status
        """
        try:
            with get_session() as session:
                if not session:
                    return False

                # Calculate time range
                singapore_tz = pytz.timezone('Asia/Singapore')
                now = datetime.now(singapore_tz)
                start_time = now - timedelta(hours=hours)

                # Format for display
                start_str_display = start_time.strftime("%Y-%m-%d %H:%M:%S")
                end_str_display = now.strftime("%Y-%m-%d %H:%M:%S")

                # Store time range for display
                self.analysis_time_range = {
                    'start': start_str_display,
                    'end': end_str_display,
                    'timezone': 'SGT'
                }
                
                # Store last updated time
                self.last_updated = now.strftime("%Y-%m-%d %H:%M:%S")

                # Format for database query (without timezone)
                start_str = start_time.strftime("%Y-%m-%d %H:%M:%S")

                if progress_bar:
                    progress_bar.progress(0.05, text=f"Fetching data from {start_str_display} to {end_str_display} (SGT)")

                # Query to fetch data from all exchanges for this pair
                query = text("""
                    SELECT 
                        exchange_name,
                        pair_name,
                        depth,
                        price,
                        created_at,
                        TO_CHAR(created_at + INTERVAL '8 hour', 'YYYY-MM-DD HH24:MI:SS.MS') AS timestamp_sgt
                    FROM 
                        uat_oracle_exchange_price_partition_v1_20250429
                    WHERE 
                        pair_name = :pair_name
                        AND created_at >= :start_time
                    ORDER BY 
                        created_at DESC
                """)
                
                # Execute query with parameters
                if progress_bar:
                    progress_bar.progress(0.1, text="Executing database query...")
                
                result = session.execute(
                    query,
                    {
                        "pair_name": pair_name,
                        "start_time": start_str
                    }
                )
                
                # Convert to DataFrame
                all_data = result.fetchall()
                
                if not all_data:
                    if progress_bar:
                        progress_bar.progress(1.0, text="No data found for the specified pair")
                    return False
                
                if progress_bar:
                    progress_bar.progress(0.2, text=f"Processing {len(all_data)} data points...")
                
                # Create DataFrame from the query results
                columns = ['exchange_name', 'pair_name', 'depth', 'price', 'created_at', 'timestamp_sgt']
                df = pd.DataFrame(all_data, columns=columns)
                
                # Convert columns to appropriate types
                df['depth'] = pd.to_numeric(df['depth'], errors='coerce')
                df['price'] = pd.to_numeric(df['price'], errors='coerce')
                
                # Get unique exchanges and depths
                exchanges = df['exchange_name'].unique()
                depths = df['depth'].unique()
                
                if progress_bar:
                    progress_bar.progress(0.3, text=f"Analyzing {len(exchanges)} exchanges and {len(depths)} depth tiers...")
                
                # Process data for each exchange and depth tier
                results = []
                
                # Get available depth tiers (convert numeric depth to tier name)
                available_tiers = []
                for depth in depths:
                    if depth in self.nominal_to_tier:
                        available_tiers.append(self.nominal_to_tier[depth])
                
                # Track global choppiness winners for win rate calculation
                choppiness_winners = []
                
                # First pass: Calculate choppiness for each exchange and tier
                for exchange in exchanges:
                    exchange_df = df[df['exchange_name'] == exchange].copy()
                    
                    for depth in depths:
                        # Skip if depth doesn't correspond to a known tier
                        if depth not in self.nominal_to_tier:
                            continue
                            
                        tier_name = self.nominal_to_tier[depth]
                        
                        # Get data for this exchange and depth
                        tier_df = exchange_df[exchange_df['depth'] == depth].copy()
                        
                        # Skip if insufficient data
                        if len(tier_df) < self.point_count * 0.6:
                            continue
                            
                        # Calculate metrics for this tier
                        metrics = self._calculate_metrics(tier_df, 'price', self.point_count)
                        
                        if not metrics:
                            continue
                            
                        # Store results with exchange and tier info
                        metrics['exchange'] = exchange
                        metrics['tier'] = tier_name
                        metrics['depth'] = depth
                        metrics['data_points'] = len(tier_df)
                        
                        # Calculate win rate separately in second pass
                        metrics['win_rate'] = 0
                        
                        # Store choppiness for determining winners
                        choppiness_winners.append({
                            'exchange': exchange,
                            'tier': tier_name,
                            'depth': depth,
                            'choppiness': metrics['choppiness']
                        })
                        
                        # Add to results
                        results.append(metrics)
                
                if not results:
                    if progress_bar:
                        progress_bar.progress(1.0, text="Insufficient data for analysis")
                    return False
                
                # Sort choppiness winners by choppiness (highest first)
                choppiness_winners = sorted(choppiness_winners, key=lambda x: x['choppiness'], reverse=True)
                
                if progress_bar:
                    progress_bar.progress(0.5, text="Calculating win rates and efficiency scores...")
                
                # Create a map for quick lookup of exchange+tier
                metrics_map = {f"{m['exchange']}:{m['tier']}": m for m in results}
                
                # Assign win rate based on choppiness ranking
                total_tiers = len(choppiness_winners)
                for i, winner in enumerate(choppiness_winners):
                    # Calculate win rate (100% for top tier, decreasing for others)
                    win_rate = 100.0 - (i / total_tiers * 100.0)
                    
                    # Update the metrics
                    key = f"{winner['exchange']}:{winner['tier']}"
                    if key in metrics_map:
                        metrics_map[key]['win_rate'] = win_rate
                
                # Calculate efficiency scores
                for metrics in results:
                    # Ensure minimum dropout rate to avoid division by zero
                    dropout_rate = max(0.1, metrics['dropout_rate'])
                    
                    # Calculate efficiency as win_rate / dropout_rate
                    metrics['efficiency'] = metrics['win_rate'] / dropout_rate
                
                # Convert results to DataFrame and sort by efficiency
                results_df = pd.DataFrame(results)
                results_df = results_df.sort_values('efficiency', ascending=False)
                
                # Store results
                self.results = results_df
                
                if progress_bar:
                    progress_bar.progress(1.0, text="Analysis complete!")
                
                return True
                
        except Exception as e:
            if progress_bar:
                progress_bar.progress(1.0, text=f"Error: {str(e)}")
            st.error(f"Error in analysis: {e}")
            import traceback
            st.error(traceback.format_exc())
            return False
    
    def _calculate_metrics(self, df, price_col, point_count):
        """Calculate metrics for a specific exchange and tier
        
        Args:
            df: DataFrame with price data
            price_col: Column name containing price data
            point_count: Number of points to analyze
            
        Returns:
            dict: Metrics including choppiness and dropout rate
        """
        try:
            # Calculate dropout rate (percentage of zero or NaN values)
            total_points = len(df)
            nan_or_zero_count = (df[price_col].isna() | (df[price_col] == 0)).sum()
            dropout_rate = (nan_or_zero_count / total_points) * 100 if total_points > 0 else 100
            
            # Get valid prices for further calculations
            prices = df[price_col].dropna()
            prices = prices[prices > 0]
            
            if len(prices) < point_count * 0.6:  # Allow some flexibility for missing data
                return None
                
            # Take only the needed number of points
            prices = prices.iloc[:min(point_count, len(prices))].copy()
            
            # Calculate metrics
            # Mean price
            mean_price = prices.mean()
            
            # Direction changes
            price_changes = prices.diff().dropna()
            signs = np.sign(price_changes)
            direction_changes = (signs.shift(1) != signs).sum()
            direction_change_pct = (direction_changes / (len(signs) - 1)) * 100 if len(signs) > 1 else 0
            
            # Choppiness (using a window-based approach)
            window = min(20, point_count // 10)
            diff = prices.diff().abs()
            sum_abs_changes = diff.rolling(window, min_periods=1).sum()
            price_range = prices.rolling(window, min_periods=1).max() - prices.rolling(window, min_periods=1).min()
            
            # Avoid division by zero
            epsilon = 1e-10
            choppiness_values = 100 * sum_abs_changes / (price_range + epsilon)
            
            # Cap extreme values
            choppiness_values = np.minimum(choppiness_values, 1000)
            
            # Mean choppiness
            choppiness = choppiness_values.mean()
            
            # Tick ATR
            tick_atr = price_changes.abs().mean()
            tick_atr_pct = (tick_atr / mean_price) * 100
            
            return {
                'choppiness': choppiness,
                'dropout_rate': dropout_rate,
                'direction_change_pct': direction_change_pct,
                'tick_atr_pct': tick_atr_pct
            }
            
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return None
    
    def get_global_rankings(self):
        """Get global rankings of all exchange-tier combinations
        
        Returns:
            DataFrame: Ranked tiers across all exchanges
        """
        if self.results is None:
            return None
            
        # Create display DataFrame with relevant columns
        display_df = self.results.copy()
        
        # Create combined exchange:tier column
        display_df['exchange_tier'] = display_df.apply(
            lambda row: f"{row['exchange']}:{row['tier']}", axis=1
        )
        
        # Select columns for display
        columns = [
            'exchange_tier', 'exchange', 'tier', 'depth', 'efficiency', 
            'win_rate', 'dropout_rate', 'choppiness', 'direction_change_pct', 
            'tick_atr_pct', 'data_points'
        ]
        
        display_df = display_df[columns]
        
        # Sort by efficiency (descending)
        display_df = display_df.sort_values('efficiency', ascending=False)
        
        return display_df

# Auto-refresh manager
class AutoRefreshManager:
    """Manages auto-refresh of the analysis"""
    
    def __init__(self, refresh_interval=600):  # Default: 10 minutes (600 seconds)
        self.refresh_interval = refresh_interval
        self.should_refresh = threading.Event()
        self.refresh_thread = None
        self.running = False
        
    def start(self, callback):
        """Start the auto-refresh thread
        
        Args:
            callback: Function to call when refresh is needed
        """
        if self.running:
            return
            
        self.running = True
        self.should_refresh.clear()
        
        def refresh_loop():
            while self.running:
                # Wait for the specified interval
                if self.should_refresh.wait(timeout=self.refresh_interval):
                    # If set, clear the event and continue
                    self.should_refresh.clear()
                    continue
                
                # Trigger the callback
                callback()
        
        self.refresh_thread = threading.Thread(target=refresh_loop)
        self.refresh_thread.daemon = True
        self.refresh_thread.start()
        
    def stop(self):
        """Stop the auto-refresh thread"""
        self.running = False
        if self.refresh_thread:
            self.should_refresh.set()  # Wake up the thread
            self.refresh_thread.join(timeout=1.0)
            self.refresh_thread = None
            
    def trigger_refresh(self):
        """Trigger an immediate refresh"""
        self.should_refresh.set()

# Format number with commas
def format_number(num):
    """Format number with thousand separators
    
    Args:
        num: Number to format
        
    Returns:
        str: Formatted string
    """
    if num is None:
        return "N/A"
    try:
        # Convert to int and format with commas
        return f"{int(float(num)):,}"
    except:
        return str(num)

# Display rankings table
def display_rankings_table(df):
    """Display the global rankings table
    
    Args:
        df: DataFrame with ranking data
    """
    if df is None or len(df) == 0:
        st.info("No ranking data available.")
        return
        
    # Make a copy for display formatting
    display_df = df.copy()
    
    # Format exchange_tier with styled HTML tags
    def format_exchange_tier(row):
        exchange = row['exchange']
        tier = row['tier']
        
        # Map exchange to CSS class
        exchange_class = f"exchange-{exchange.lower()}"
        
        return f"""<span class="exchange-tag {exchange_class}">{exchange}</span> {tier}"""
    
    # Create styled exchange_tier column for display
    display_df['Exchange:Tier'] = display_df.apply(format_exchange_tier, axis=1)
    
    # Rename and format columns
    display_df = display_df.rename(columns={
        'efficiency': 'Efficiency Score', 
        'win_rate': 'Win Rate (%)', 
        'dropout_rate': 'Dropout Rate (%)',
        'choppiness': 'Choppiness',
        'direction_change_pct': 'Direction Changes (%)',
        'tick_atr_pct': 'Tick ATR (%)',
        'tier': 'Tier',
        'exchange': 'Exchange',
        'depth': 'Depth',
        'data_points': 'Data Points'
    })
    
    # Format numeric columns
    for col in display_df.columns:
        if col in ['Win Rate (%)', 'Dropout Rate (%)', 'Direction Changes (%)']:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.1f}%" if not pd.isna(x) else "N/A"
            )
        elif col == 'Tick ATR (%)':
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A"
            )
        elif col == 'Efficiency Score':
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A"
            )
        elif col == 'Choppiness':
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.1f}" if not pd.isna(x) else "N/A"
            )
        elif col == 'Depth':
            display_df[col] = display_df[col].apply(
                lambda x: f"{int(x):,}" if not pd.isna(x) else "N/A"
            )
    
    # Select columns for display
    display_columns = [
        'Exchange:Tier', 'Efficiency Score', 'Win Rate (%)', 
        'Dropout Rate (%)', 'Choppiness', 'Depth'
    ]
    
    # Show the table with all needed context
    st.markdown("### Global Tier Rankings Across All Exchanges")
    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 8px; border-radius: 5px; margin-bottom: 15px;">
        <p style="margin: 3px 0;"><strong>Efficiency Score</strong> = Win Rate (%) / Dropout Rate (%)</p>
        <p style="margin: 3px 0;"><strong>Win Rate</strong>: Based on relative choppiness ranking (100% for top tier)</p>
        <p style="margin: 3px 0;"><strong>Dropout Rate</strong>: Percentage of missing or zero values in the feed</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.dataframe(
        display_df[display_columns],
        use_container_width=True,
        height=min(800, 100 + (len(display_df) * 35))
    )
    
    # Show fallback recommendations
    if len(display_df) >= 3:
        st.markdown("### Recommended Tier Fallback Strategy")
        
        # Get top 3 tiers
        top_tiers = display_df.iloc[:3]
        
        # Display recommendation
        st.markdown(f"""
        <div style="background-color: #e6f7ff; padding: 10px; border-radius: 5px; margin-bottom: 15px;">
            <h4 style="margin: 0;">Cross-Exchange Tier Strategy:</h4>
            <p style="margin: 5px 0;"><strong>Primary Tier:</strong> {top_tiers.iloc[0]['Exchange:Tier']}</p>
            <p style="margin: 5px 0;"><strong>Fallback Tier 1:</strong> {top_tiers.iloc[1]['Exchange:Tier']}</p>
            <p style="margin: 5px 0;"><strong>Fallback Tier 2:</strong> {top_tiers.iloc[2]['Exchange:Tier']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        ### Implementation Guide
        
        For real-time implementation, use this approach:
        
        1. **Multi-Exchange Monitoring**: Subscribe to all three tiers across the different exchanges
        2. **Zero-Tolerance Fallback**: If primary tier returns zero/missing value, IMMEDIATELY use data from fallback tier 1
        3. **Global Ranking**: The recommended tiers are ranked across ALL exchanges based on efficiency
        4. **Auto-Update**: This analysis refreshes every 10 minutes to adapt to changing market conditions
        """)

def main():
    # Get current Singapore time
    singapore_tz = pytz.timezone('Asia/Singapore')
    now_sg = datetime.now(singapore_tz)
    current_time_sg = now_sg.strftime("%Y-%m-%d %H:%M:%S")

    # Initialize session state for tracking auto-refresh
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = CrossExchangeAnalyzer()
    
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = None
        
    if 'refresh_manager' not in st.session_state:
        st.session_state.refresh_manager = AutoRefreshManager(refresh_interval=600)  # 10 minutes
        
    if 'selected_pair' not in st.session_state:
        st.session_state.selected_pair = None
        
    if 'auto_refresh_enabled' not in st.session_state:
        st.session_state.auto_refresh_enabled = False

    # Main layout
    st.markdown("<h1 style='text-align: center; font-size:28px; margin-bottom: 10px;'>Cross-Exchange Depth Tier Analyzer</h1>", unsafe_allow_html=True)

    # Display current time
    st.markdown(f"<p style='text-align: center; font-size:14px; color:gray;'>Current time: {current_time_sg} (SGT)</p>", unsafe_allow_html=True)

    # Get available pairs
    available_pairs = get_available_pairs()

    # Top controls area
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        selected_pair = st.selectbox(
            "Select Pair",
            available_pairs,
            index=0 if available_pairs and len(available_pairs) > 0 else None
        )
        st.session_state.selected_pair = selected_pair

    with col2:
        auto_refresh = st.checkbox("Auto-refresh (10 min)", value=st.session_state.auto_refresh_enabled)
        
        # Update auto-refresh state
        if auto_refresh != st.session_state.auto_refresh_enabled:
            st.session_state.auto_refresh_enabled = auto_refresh
            if auto_refresh:
                # Start auto-refresh
                def refresh_callback():
                    # This will trigger a rerun with the current selected pair
                    st.session_state.last_refresh = datetime.now()
                    st.experimental_rerun()
                
                st.session_state.refresh_manager.start(refresh_callback)
            else:
                # Stop auto-refresh
                st.session_state.refresh_manager.stop()

    with col3:
        run_analysis = st.button("ANALYZE NOW", use_container_width=True)

    # Main content area
    if (run_analysis or st.session_state.last_refresh is not None) and selected_pair:
        # Reset last_refresh to avoid immediate rerun
        st.session_state.last_refresh = None
        
        # Show analysis time
        analysis_start_time = datetime.now(singapore_tz).strftime("%Y-%m-%d %H:%M:%S")
        st.markdown(f"<p style='text-align: center; font-size:14px; color:green;'>Analysis started at: {analysis_start_time} (SGT)</p>", unsafe_allow_html=True)

        # Create progress bar
        progress_bar = st.progress(0, text="Starting analysis...")

        # Run the analysis
        success = st.session_state.analyzer.fetch_and_analyze(
            selected_pair,
            hours=24,  # Fixed 24 hour lookback
            progress_bar=progress_bar
        )

        if success:
            # Display time range
            if hasattr(st.session_state.analyzer, 'analysis_time_range') and st.session_state.analyzer.analysis_time_range:
                time_range = st.session_state.analyzer.analysis_time_range
                st.markdown(f"""
                <div style="background-color: #e6f3ff; padding: 10px; border-radius: 5px; margin-bottom: 20px;">
                    <h3 style="margin: 0;">Analysis Time Range</h3>
                    <p style="margin: 5px 0;"><strong>From:</strong> {time_range['start']} ({time_range['timezone']})</p>
                    <p style="margin: 5px 0;"><strong>To:</strong> {time_range['end']} ({time_range['timezone']})</p>
                    <p style="margin: 5px 0;"><strong>Last Updated:</strong> {st.session_state.analyzer.last_updated} ({time_range['timezone']})</p>
                </div>
                """, unsafe_allow_html=True)

            # Get and display global rankings
            rankings = st.session_state.analyzer.get_global_rankings()
            display_rankings_table(rankings)

            # Show analysis completion time
            analysis_end_time = datetime.now(singapore_tz).strftime("%Y-%m-%d %H:%M:%S")
            st.markdown(f"<p style='text-align: center; font-size:14px; color:green;'>Analysis completed at: {analysis_end_time} (SGT)</p>", unsafe_allow_html=True)
            
            if st.session_state.auto_refresh_enabled:
                next_refresh = datetime.now() + timedelta(minutes=10)
                next_refresh_str = next_refresh.strftime("%H:%M:%S")
                st.markdown(f"<p style='text-align: center; font-size:14px; color:gray;'>Next auto-refresh scheduled at: {next_refresh_str}</p>", unsafe_allow_html=True)

        else:
            progress_bar.empty()
            st.error(f"Failed to analyze {selected_pair}. Please try another pair.")

    else:
        # Welcome message
        st.info("Select a pair and click ANALYZE NOW to find the optimal depth tier with cross-exchange efficiency ranking.")
        
        # Instructions
        st.markdown("""
        ### About This Tool
        
        This analyzer helps you find the best depth tier across all exchanges by:
        
        1. **Global Ranking**: Ranks all tiers from all exchanges in a single list
        2. **Efficiency Score**: Calculates Win Rate / Dropout Rate for optimal tier selection
        3. **Fallback Strategy**: Recommends primary and fallback tiers for robust implementation
        4. **Auto-Refresh**: Updates analysis every 10 minutes when enabled
        
        **Note**: Data is fetched from the `uat_oracle_exchange_price_partition_v1_20250429` table with a 24-hour lookback period.
        """)

if __name__ == "__main__":
    main()