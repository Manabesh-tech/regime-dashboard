import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
import time
import threading

# Clear cache at startup to ensure fresh data
st.cache_data.clear()

# Page configuration
st.set_page_config(
    page_title="Global Tier Efficiency Analyzer",
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
        font-size: 16px !important;
        width: 100% !important;
    }

    .dataframe th {
        font-weight: 700 !important;
        background-color: #f0f2f6 !important;
    }

    .dataframe td {
        font-weight: 500 !important;
    }
    
    /* Exchange-tier tag styling */
    .exchange-tier {
        display: inline-flex;
        align-items: center;
        font-size: 14px;
    }
    
    .exchange-tag {
        display: inline-block;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: bold;
        margin-right: 5px;
        font-size: 14px;
    }
    .exchange-binancefuture {
        background-color: #f3ba2f;
        color: black;
    }
    .exchange-hyperliquidfuture {
        background-color: #333333;
        color: white;
    }
    .exchange-bybitfuture {
        background-color: #b50c5c;
        color: white;
    }
    .exchange-okxfuture {
        background-color: #121212;
        color: white;
    }
    .exchange-mexcfuture {
        background-color: #0052FF;
        color: white;
    }
    .exchange-gatefuture {
        background-color: #3366CC;
        color: white;
    }
    .exchange-bitgetfuture {
        background-color: #FF6600;
        color: white;
    }
    
    .tier-tag {
        display: inline-block;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: bold;
        background-color: #E0E0E0;
        color: #000;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Create database engine
@st.cache_resource
def get_engine():
    """Create database engine
    Returns:
        engine: SQLAlchemy engine
    """
    try:
        # Use the correct database details from the working SQL connection
        user = "public_rw"
        password = "aTJ92^kl04hllk"
        host = "aws-jp-tk-surf-pg-public.cluster-csteuf9lw8dv.ap-northeast-1.rds.amazonaws.com"
        port = 5432
        database = "report_dev"
        
        # Construct connection URL
        db_url = f"postgresql://{user}:{password}@{host}:{port}/{database}"
        return create_engine(db_url, pool_size=5, max_overflow=10)
    except Exception as e:
        st.error(f"Error creating database engine: {e}")
        return None

@contextmanager
def get_session():
    """Database session context manager
    Yields:
        session: SQLAlchemy session
    """
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

# Get available pairs from the database
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_available_pairs():
    """Fetch available trading pairs from the database
    Returns:
        list: List of available trading pairs
    """
    # Provide a default list in case database query fails
    default_pairs = ["BTC", "SOL", "ETH", "TRUMP"]
    
    try:
        with get_session() as session:
            if not session:
                return default_pairs

            # Get current date for table name
            current_date = datetime.now().strftime("%Y%m%d")
            table_name = f"oracle_exchange_price_partition_v1_{current_date}"
            
            # Query the oracle_exchange_price table to get unique pairs
            query = text(f"""
                SELECT DISTINCT pair_name 
                FROM {table_name}
                ORDER BY pair_name
            """)
            
            result = session.execute(query)
            pairs = [row[0] for row in result]
            
            # Return the sorted pairs or default list if empty
            return sorted(pairs) if pairs else default_pairs

    except Exception as e:
        st.error(f"Error fetching available pairs: {e}")
        return default_pairs  # Return default pairs on error

# Class to handle global tier analysis
class GlobalTierAnalyzer:
    """Analyzer for tiers across all exchanges globally"""
    
    def __init__(self):
        # Fixed point count (5000 points as requested)
        self.point_count = 5000
        
        # Define available price tiers
        self.tier_columns = [
            'price_1', 'price_2', 'price_3', 'price_4', 'price_5',
            'price_6', 'price_7', 'price_8', 'price_9', 'price_10',
            'price_11', 'price_12', 'price_13', 'price_14', 'price_15', 
            'price_16', 'price_17', 'price_18', 'price_19', 'price_20',
            'price_21', 'price_22', 'price_23', 'price_24', 'price_25',
            'price_26', 'price_27', 'price_28', 'price_29'
        ]
        
        # Map columns to actual tier values
        self.tier_values = {
            'price_26': '1k',
            'price_27': '3k',
            'price_28': '5k',
            'price_29': '7k',
            'price_1': '10k',
            'price_2': '50k',
            'price_3': '100k',
            'price_4': '200k',
            'price_5': '300k',
            'price_6': '400k',
            'price_7': '500k',
            'price_8': '600k',
            'price_9': '700k',
            'price_10': '800k',
            'price_11': '900k',
            'price_12': '1000k',
            'price_13': '2000k',
            'price_14': '3000k',
            'price_15': '4000k',
            'price_16': '5000k',
            'price_17': '6000k',
            'price_18': '7000k',
            'price_19': '8000k',
            'price_20': '9000k',
            'price_21': '10000k',
            'price_22': '11000k',
            'price_23': '12000k',
            'price_24': '13000k',
            'price_25': '14000k',
        }
        
        # Store analysis results
        self.results = None
        self.analysis_time_range = None
        self.last_updated = None
    
    def fetch_and_analyze(self, pair_name, hours=24, progress_bar=None):
        """Fetch data and analyze all exchange-tier combinations
        
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

                # Calculate time range (for display only)
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

                if progress_bar:
                    progress_bar.progress(0.05, text=f"Fetching data for {pair_name}...")

                # Get current date for table name
                current_date = datetime.now().strftime("%Y%m%d")
                table_name = f"oracle_exchange_price_partition_v1_{current_date}"
                
                # Join all tier columns for the query
                price_columns = ", ".join(self.tier_columns)
                
                # Query to fetch data for all tiers across all exchanges
                query = text(f"""
                    SELECT 
                        source as exchange_name,
                        pair_name,
                        created_at,
                        TO_CHAR(created_at + INTERVAL '8 hour', 'YYYY-MM-DD HH24:MI:SS.MS') AS timestamp_sgt,
                        all_bid,
                        all_ask,
                        {price_columns}
                    FROM 
                        {table_name}
                    WHERE 
                        pair_name = :pair_name
                    ORDER BY 
                        created_at DESC
                    LIMIT 10000
                """)
                
                # Execute query with parameters
                if progress_bar:
                    progress_bar.progress(0.1, text=f"Executing database query for table:{table_name}...")
                
                result = session.execute(
                    query,
                    {
                        "pair_name": pair_name
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
                columns = ['exchange_name', 'pair_name', 'created_at', 'timestamp_sgt', 'all_bid', 'all_ask'] + self.tier_columns
                df = pd.DataFrame(all_data, columns=columns)
                
                # Convert columns to appropriate types
                for column in self.tier_columns:
                    df[column] = pd.to_numeric(df[column], errors='coerce')
                
                df['all_bid'] = pd.to_numeric(df['all_bid'], errors='coerce')
                df['all_ask'] = pd.to_numeric(df['all_ask'], errors='coerce')
                
                # Get unique exchanges
                exchanges = df['exchange_name'].unique()
                
                if progress_bar:
                    progress_bar.progress(0.3, text=f"Analyzing {len(exchanges)} exchanges across all tiers...")
                
                # Process data for each exchange and tier
                results = []
                
                # Process each exchange
                exchange_count = len(exchanges)
                for i, exchange in enumerate(exchanges):
                    if progress_bar:
                        # Update progress based on exchanges processed
                        progress = 0.3 + (i / exchange_count) * 0.6
                        progress_bar.progress(progress, text=f"Analyzing {exchange} tiers...")
                    
                    # Filter data for this exchange
                    exchange_df = df[df['exchange_name'] == exchange].copy()
                    
                    # Skip if not enough data
                    if len(exchange_df) < self.point_count * 0.3:  # Allow very flexible threshold
                        continue
                    
                    # Process each tier for this exchange
                    for tier_col in self.tier_columns:
                        # Skip if this tier isn't named in our mapping
                        if tier_col not in self.tier_values:
                            continue
                            
                        # Get tier name from mapping
                        tier_name = self.tier_values[tier_col]
                        
                        # Calculate metrics for this exchange-tier
                        metrics = self._calculate_metrics(exchange_df, tier_col, self.point_count)
                        
                        if not metrics:
                            continue
                            
                        # Store results with exchange and tier info
                        metrics['exchange'] = exchange
                        metrics['tier'] = tier_name
                        metrics['exchange_tier'] = f"{exchange}:{tier_name}"
                        metrics['data_points'] = len(exchange_df)
                        
                        # Calculate efficiency using Choppiness * (100% - Dropout Rate)
                        choppiness = metrics['choppiness']
                        dropout_rate = metrics['dropout_rate']
                        
                        # Calculate efficiency score
                        metrics['efficiency'] = choppiness * ((100 - dropout_rate) / 100)
                        
                        # Add to results
                        results.append(metrics)
                
                if not results:
                    if progress_bar:
                        progress_bar.progress(1.0, text="Insufficient data for analysis")
                    return False
                
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
        """Calculate metrics for a specific exchange-tier combination
        
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
            
            # If 100% of the values are missing, skip this tier
            if dropout_rate >= 99.9:
                return None
                
            # Get valid prices for further calculations
            prices = df[price_col].dropna()
            prices = prices[prices > 0]
            
            # Check if we have enough data after filtering
            if len(prices) < max(100, point_count * 0.1):  # Very flexible threshold for global analysis
                return None
                
            # Take only the needed number of points
            prices = prices.iloc[:min(point_count, len(prices))].copy()
            
            # Calculate metrics
            # Direction changes
            price_changes = prices.diff().dropna()
            signs = np.sign(price_changes)
            direction_changes = (signs.shift(1) != signs).sum()
            direction_change_pct = (direction_changes / (len(signs) - 1)) * 100 if len(signs) > 1 else 0
            
            # Choppiness (using a window-based approach)
            window = min(20, len(prices) // 10)
            diff = prices.diff().abs()
            sum_abs_changes = diff.rolling(window, min_periods=1).sum()
            price_range = prices.rolling(window, min_periods=1).max() - prices.rolling(window, min_periods=1).min()
            
            # Avoid division by zero
            epsilon = 1e-10
            choppiness_values = 100 * sum_abs_changes / (price_range + epsilon)
            
            # Raw choppiness values without capping
            choppiness = choppiness_values.mean()
            
            # Get total bid/ask for liquidity data if available
            avg_bid = df['all_bid'].mean() if 'all_bid' in df.columns else None
            avg_ask = df['all_ask'].mean() if 'all_ask' in df.columns else None
            
            # Use min of bid/ask as total liquidity
            if avg_bid is not None and avg_ask is not None:
                avg_liquidity = min(avg_bid, avg_ask)
            else:
                avg_liquidity = None
            
            return {
                'choppiness': choppiness,
                'dropout_rate': dropout_rate,
                'direction_change_pct': direction_change_pct,
                'avg_liquidity': avg_liquidity,
                'actual_points': len(prices)
            }
            
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return None
            
    def get_global_rankings(self):
        """Get global rankings of all exchange-tier combinations
        
        Returns:
            DataFrame: Globally ranked exchange-tier combinations
        """
        if self.results is None:
            return None
            
        # Create display DataFrame with relevant columns
        display_df = self.results.copy()
        
        # Select columns for display
        columns = [
            'exchange', 'tier', 'exchange_tier', 'efficiency', 
            'choppiness', 'dropout_rate', 'direction_change_pct', 
            'avg_liquidity', 'data_points', 'actual_points'
        ]
        
        # Filter to columns that exist in the dataframe
        available_columns = [col for col in columns if col in display_df.columns]
        display_df = display_df[available_columns]
        
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

# Display global rankings table
def display_global_rankings(df):
    """Display the global exchange-tier rankings table
    
    Args:
        df: DataFrame with ranking data
    """
    if df is None or len(df) == 0:
        st.info("No ranking data available.")
        return
        
    # Make a copy for display formatting
    display_df = df.copy()
    
    # Format exchange with styled HTML tags
    def format_exchange_tier(row):
        exchange = row['exchange']
        tier = row['tier']
        
        # Map exchange to CSS class
        exchange_class = f"exchange-{exchange.lower()}"
        
        # Create HTML for exchange-tier combination
        return f"""
        <span class="exchange-tier">
            <span class="exchange-tag {exchange_class}">{exchange}</span>
            <span class="tier-tag">{tier}</span>
        </span>
        """
    
    # Create styled exchange-tier column for display
    display_df['Exchange:Tier'] = display_df.apply(format_exchange_tier, axis=1)
    
    # Rename and format columns
    display_df = display_df.rename(columns={
        'efficiency': 'Efficiency Score', 
        'choppiness': 'Choppiness', 
        'dropout_rate': 'Dropout Rate (%)',
        'direction_change_pct': 'Direction Changes (%)',
        'avg_liquidity': 'Avg Liquidity',
        'data_points': 'Total Points',
        'actual_points': 'Valid Points'
    })
    
    # Select columns for display
    display_columns = [
        'Exchange:Tier', 'Efficiency Score', 'Choppiness', 
        'Dropout Rate (%)', 'Avg Liquidity', 'Valid Points'
    ]
    
    # Filter to columns that exist
    available_display_columns = [col for col in display_columns if col in display_df.columns]
    
    # Show the table with all needed context
    st.markdown("### Global Tier Rankings Across All Exchanges")
    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 8px; border-radius: 5px; margin-bottom: 15px;">
        <p style="margin: 3px 0;"><strong>Efficiency Score</strong> = Choppiness × (100% - Dropout Rate)</p>
        <p style="margin: 3px 0;"><strong>Choppiness</strong>: Raw choppiness value measuring price oscillation</p>
        <p style="margin: 3px 0;"><strong>Dropout Rate (%)</strong>: Percentage of missing or zero values in the price feed</p>
        <p style="margin: 3px 0;"><strong>Valid Points</strong>: Number of non-zero data points used for analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Convert to HTML and display with styled tags
    st.markdown(display_df[available_display_columns].to_html(escape=False, index=False), unsafe_allow_html=True)
    
    # Show top recommendations
    if len(display_df) >= 3:
        st.markdown("### Recommended Global Tier Fallback Strategy")
        
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
        
        1. **Multi-Exchange Tier Monitoring**: Subscribe to all three exchange-tier combinations simultaneously
        2. **Zero-Tolerance Fallback**: If primary tier returns zero/missing value, IMMEDIATELY use data from fallback tier 1
        3. **Global Ranking**: These recommendations are based on a unified ranking across ALL exchanges and ALL tiers
        4. **Raw Values**: The efficiency score uses raw values: Choppiness × (100% - Dropout Rate)
        """)

def main():
    # Get current Singapore time
    singapore_tz = pytz.timezone('Asia/Singapore')
    now_sg = datetime.now(singapore_tz)
    current_time_sg = now_sg.strftime("%Y-%m-%d %H:%M:%S")

    # Initialize session state for tracking auto-refresh
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = GlobalTierAnalyzer()
    
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = None
        
    if 'refresh_manager' not in st.session_state:
        st.session_state.refresh_manager = AutoRefreshManager(refresh_interval=600)  # 10 minutes
        
    if 'selected_pair' not in st.session_state:
        st.session_state.selected_pair = None
        
    if 'auto_refresh_enabled' not in st.session_state:
        st.session_state.auto_refresh_enabled = False

    # Main layout
    st.markdown("<h1 style='text-align: center; font-size:28px; margin-bottom: 10px;'>Global Tier Efficiency Analyzer</h1>", unsafe_allow_html=True)

    # Display current time
    st.markdown(f"<p style='text-align: center; font-size:14px; color:gray;'>Current time: {current_time_sg} (SGT)</p>", unsafe_allow_html=True)

    # Get available pairs but handle error case better
    try:
        available_pairs = get_available_pairs()
    except:
        available_pairs = ["BTC", "SOL", "ETH", "TRUMP"]  # Default fallback
    
    # Ensure we have a valid list
    if not available_pairs or not isinstance(available_pairs, list):
        available_pairs = ["BTC", "SOL", "ETH", "TRUMP"]  # Emergency fallback

    # Top controls area
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        # Make sure we have a valid default index
        default_index = 0 if available_pairs and len(available_pairs) > 0 else None
        
        # Use a try-except block to handle potential errors
        try:
            selected_pair = st.selectbox(
                "Select Pair",
                options=available_pairs,
                index=default_index
            )
            st.session_state.selected_pair = selected_pair
        except Exception as e:
            st.error(f"Error displaying pairs: {e}")
            selected_pair = "BTC"  # Force a default selection
            st.write(f"Using default pair: {selected_pair}")
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
                    <h3 style="margin: 0;">Analysis Time Range (For Reference)</h3>
                    <p style="margin: 5px 0;"><strong>From:</strong> {time_range['start']} ({time_range['timezone']})</p>
                    <p style="margin: 5px 0;"><strong>To:</strong> {time_range['end']} ({time_range['timezone']})</p>
                    <p style="margin: 5px 0;"><strong>Last Updated:</strong> {st.session_state.analyzer.last_updated} ({time_range['timezone']})</p>
                </div>
                """, unsafe_allow_html=True)

            # Get and display global rankings
            rankings = st.session_state.analyzer.get_global_rankings()
            display_global_rankings(rankings)

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
        st.info("Select a pair and click ANALYZE NOW to find the optimal tier across all exchanges.")
        
        # Instructions
        st.markdown("""
        ### About This Tool
        
        This analyzer creates a global ranking of all exchange-tier combinations:
        
        1. **Unified Ranking**: Every exchange-tier combination ranked in a single global list
        2. **Raw Efficiency Formula**: Efficiency = Choppiness × (100% - Dropout Rate)
        3. **No Normalization**: Uses raw values without capping or normalization
        4. **Comprehensive Analysis**: Analyzes all available tiers across all exchanges
        
        This approach gives you a complete view of which specific exchange-tier combination 
        provides the optimal blend of high choppiness and low dropout rate.
        """)

if __name__ == "__main__":
    main()