import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import psycopg2
import warnings
import pytz


# Suppress warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="SURF Environment Analysis",
    page_icon="📊",
    layout="wide"
)

# Always clear cache at startup to ensure fresh data
st.cache_data.clear()

# Configure database
def init_db_connection():
    # DB parameters - these should be stored in Streamlit secrets in production
    db_params = {
        'host': 'aws-jp-tk-surf-pg-public.cluster-csteuf9lw8dv.ap-northeast-1.rds.amazonaws.com',
        'port': 5432,
        'database': 'report_dev',
        'user': 'public_rw',
        'password': 'aTJ92^kl04hllk'
    }
    
    try:
        conn = psycopg2.connect(
            host=db_params['host'],
            port=db_params['port'],
            database=db_params['database'],
            user=db_params['user'],
            password=db_params['password'],
        )
        conn.autocommit = True
        return conn, db_params
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return None, db_params

# Initialize connection
conn, db_params = init_db_connection()

# Main title
st.title("SURF Environment Analysis Dashboard")

# Set up the timezone
singapore_timezone = pytz.timezone('Asia/Singapore')
now_utc = datetime.now(pytz.utc)
now_sg = now_utc.astimezone(singapore_timezone)
st.write(f"Current Singapore Time: {now_sg.strftime('%Y-%m-%d %H:%M:%S')}")

# SURF Analyzer class
class SurfAnalyzer:
    """Specialized analyzer for calculating metrics across different SURF environments"""
    
    def __init__(self):
        self.environment_data = {}  # Will store data from different SURF environments
        self.all_environments = ['UAT']
        
        # Metrics to calculate
        self.metrics = [
            'direction_changes',   # Frequency of price direction reversals (%)
            'choppiness',          # Measures price oscillation within a range
            'tick_atr_pct',        # ATR % (Average True Range as percentage of mean price)
            'trend_strength'       # Measures directional strength
        ]
        
        # Display names for metrics (for printing)
        self.metric_display_names = {
            'direction_changes': 'Direction Changes (%)',
            'choppiness': 'Choppiness',
            'tick_atr_pct': 'Tick ATR %',
            'trend_strength': 'Trend Strength'
        }
        
        # Point counts to analyze
        self.point_counts = [500, 1500, 2500, 5000]
        
        # Initialize data structures
        for metric in self.metrics:
            self.environment_data[metric] = {point: {} for point in self.point_counts}
        
        # Initialize timestamp_ranges structure
        self.environment_data['timestamp_ranges'] = {point: {} for point in self.point_counts}

    def _get_partition_tables(self, conn, start_date, end_date):
        """
        Get list of partition tables that need to be queried based on date range.
        Returns a list of table names (oracle_price_log_partition_v1)
        """
        # Convert to datetime objects if they're strings
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str) and end_date:
            end_date = pd.to_datetime(end_date)
        elif end_date is None:
            # Use explicit Singapore timezone when getting current date
            singapore_tz = pytz.timezone('Asia/Singapore')
            end_date = datetime.now(singapore_tz)
            
        # Ensure timezone is explicitly set to Singapore
        singapore_tz = pytz.timezone('Asia/Singapore')
        if start_date.tzinfo is None:
            start_date = singapore_tz.localize(start_date)
        if end_date.tzinfo is None:
            end_date = singapore_tz.localize(end_date)
        
        # Convert to Singapore time
        start_date = start_date.astimezone(singapore_tz)
        end_date = end_date.astimezone(singapore_tz)
        
        # Remove timezone after conversion for compatibility with database
        start_date = start_date.replace(tzinfo=None)
        end_date = end_date.replace(tzinfo=None)
                
        # Generate list of dates between start and end
        current_date = start_date
        dates = []
        
        while current_date <= end_date:
            dates.append(current_date.strftime("%Y%m%d"))
            current_date += timedelta(days=1)
        
        # Create table names from dates
        table_names = ["oracle_price_log_partition_v1"]
        
        # Debug info
        st.write(f"Looking for tables: {table_names}")
        
        # Verify which tables actually exist in the database
        cursor = conn.cursor()
        existing_tables = []
        
        for table in table_names:
            # Check if table exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = %s
                );
            """, (table,))
            
            if cursor.fetchone()[0]:
                existing_tables.append(table)
        
        cursor.close()
        
        if not existing_tables:
            st.warning(f"No partition tables found for the date range {start_date.date()} to {end_date.date()}")
        
        return existing_tables

    def _build_query_for_partition_tables(self, tables, pair_name, start_time, end_time, environment):
        """
        Build a complete UNION query for multiple partition tables.
        This creates a complete, valid SQL query with correct WHERE clauses.
        """
        if not tables:
            return ""
        
        # Convert the times to datetime objects if they're strings
        if isinstance(start_time, str):
            start_time = pd.to_datetime(start_time)
        if isinstance(end_time, str):
            end_time = pd.to_datetime(end_time)
        
        # Format with timezone information explicitly
        start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
        end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
            
        union_parts = []
        
        for table in tables:
            # For PROD data
            if environment == 'PROD':
                query = f"""
                SELECT 
                    pair_name,
                    created_at + INTERVAL '8 hour' AS timestamp,
                    final_price AS price
                FROM 
                    public.{table}
                WHERE 
                    created_at >= '{start_time_str}'::timestamp - INTERVAL '8 hour'
                    AND created_at <= '{end_time_str}'::timestamp - INTERVAL '8 hour'
                    AND source_type = 0
                    AND pair_name = '{pair_name}'
                """
            # For SIT data
            elif environment == 'SIT':
                query = f"""
                SELECT 
                    pair_name,
                    created_at + INTERVAL '8 hour' AS timestamp,
                    final_price AS price
                FROM 
                    public.{table}
                WHERE 
                    created_at >= '{start_time_str}'::timestamp - INTERVAL '8 hour'
                    AND created_at <= '{end_time_str}'::timestamp - INTERVAL '8 hour'
                    AND source_type = 0
                    AND pair_name = '{pair_name}'
                """
            # For UAT data
            else:  # UAT
                query = f"""
                SELECT 
                    pair_name,
                    created_at + INTERVAL '8 hour' AS timestamp,
                    final_price AS price
                FROM 
                    public.{table}
                WHERE 
                    created_at >= '{start_time_str}'::timestamp - INTERVAL '8 hour'
                    AND created_at <= '{end_time_str}'::timestamp - INTERVAL '8 hour'
                    AND source_type = 0
                    AND pair_name = '{pair_name}'
                """
            
            union_parts.append(query)
        
        # Join with UNION and add ORDER BY at the end
        complete_query = " UNION ".join(union_parts) + " ORDER BY timestamp DESC"
        return complete_query

    def fetch_and_analyze(self, conn, pairs_to_analyze, hours=24):
        """
        Fetch data for SURF environments, analyze metrics.
        
        Args:
            conn: Database connection
            pairs_to_analyze: List of coin pairs to analyze
            hours: Hours to look back for data retrieval
        """
        # Environments to compare
        environments_to_compare = ['UAT']
        
        # Use explicit Singapore timezone for all time calculations
        singapore_tz = pytz.timezone('Asia/Singapore')
        now = datetime.now(singapore_tz)
        
        # Calculate times in Singapore timezone
        end_time = now.strftime("%Y-%m-%d %H:%M:%S")
        start_time = (now - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
        
        st.info(f"Retrieving data from the last {hours} hours")
        st.write(f"Start time: {start_time} (SGT)")
        st.write(f"End time: {end_time} (SGT)")
        
        try:
            # Get relevant partition tables for this time range
            partition_tables = self._get_partition_tables(conn, start_time, end_time)
            
            if not partition_tables:
                # If no tables found, try looking one day earlier (for edge cases)
                st.warning("No tables found for the specified range, trying to look back one more day...")
                alt_start_time = (now - timedelta(hours=hours+24)).strftime("%Y-%m-%d %H:%M:%S")
                partition_tables = self._get_partition_tables(conn, alt_start_time, end_time)
                
                if not partition_tables:
                    st.error("No data tables available for the selected time range, even with extended lookback.")
                    return None
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Build all queries first to minimize time between executions
            all_queries = {}
            for i, pair in enumerate(pairs_to_analyze):
                progress_bar.progress((i) / len(pairs_to_analyze) / 3)  # First third for query building
                status_text.text(f"Building queries for {pair} ({i+1}/{len(pairs_to_analyze)})")
                
                all_queries[pair] = {}
                for environment in environments_to_compare:
                    query = self._build_query_for_partition_tables(
                        partition_tables,
                        pair_name=pair,
                        start_time=start_time,
                        end_time=end_time,
                        environment=environment
                    )
                    all_queries[pair][environment] = query
            
            # Execute all queries in quick succession
            pair_data = {}
            for i, pair in enumerate(pairs_to_analyze):
                progress_bar.progress(0.33 + (i) / len(pairs_to_analyze) / 3)  # Second third for query execution
                status_text.text(f"Executing queries for {pair} ({i+1}/{len(pairs_to_analyze)})")
                
                pair_data[pair] = {}
                
                # Execute queries for each environment back-to-back for each pair
                for environment in environments_to_compare:
                    query = all_queries[pair][environment]
                    if query:
                        try:
                            df = pd.read_sql_query(query, conn)
                            if len(df) > 0:
                                pair_data[pair][environment] = df
                                st.write(f"Found {len(df)} records for SURF_{environment}_{pair}")
                            else:
                                st.warning(f"No data found for SURF_{environment}_{pair}")
                        except Exception as e:
                            st.error(f"Database query error for SURF_{environment}_{pair}: {e}")
            
            # Process the data for analysis
            for i, pair in enumerate(pairs_to_analyze):
                progress_bar.progress(0.67 + (i) / len(pairs_to_analyze) / 3)  # Final third for processing
                status_text.text(f"Analyzing {pair} ({i+1}/{len(pairs_to_analyze)})")
                
                # Process data for each environment
                coin_key = pair.replace('/', '_')
                for environment in environments_to_compare:
                    if environment in pair_data[pair]:
                        self._process_price_data(pair_data[pair][environment], 'timestamp', 'price', coin_key, environment)
            
            # Final progress update
            progress_bar.progress(1.0)
            status_text.text(f"Processing complete!")
            
            return self.environment_data
                
        except Exception as e:
            st.error(f"Error fetching and processing data: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _process_price_data(self, data, timestamp_col, price_col, coin_key, environment):
        """Process price data for a cryptocurrency and calculate metrics for specified point counts."""
        try:
            # Extract price data
            filtered_df = data.copy()
            prices = pd.to_numeric(filtered_df[price_col], errors='coerce')
            prices = prices.dropna()
            
            if len(prices) < 100:  # Minimum threshold for meaningful analysis
                return
            
            # Calculate metrics for each point count
            for point_count in self.point_counts:
                if len(prices) >= point_count:
                    # Use the most recent N points
                    sample = prices.iloc[:point_count]
                    
                    # Get timestamp range information (first the timestamp column must exist)
                    if timestamp_col in filtered_df.columns:
                        sample_timestamps = filtered_df[timestamp_col].iloc[:point_count]
                        start_time = sample_timestamps.iloc[-1] if not sample_timestamps.empty else None
                        end_time = sample_timestamps.iloc[0] if not sample_timestamps.empty else None
                    else:
                        start_time, end_time = None, None
                    
                    # Calculate mean price for ATR percentage calculation
                    mean_price = sample.mean()
                    
                    # Calculate each metric
                    direction_changes = self._calculate_direction_changes(sample)
                    choppiness = self._calculate_choppiness(sample, min(20, point_count // 10))
                    
                    # Calculate tick ATR
                    true_ranges = sample.diff().abs().dropna()
                    tick_atr = true_ranges.mean()
                    tick_atr_pct = (tick_atr / mean_price) * 100  # Convert to percentage of mean price
                    
                    # Calculate trend strength
                    trend_strength = self._calculate_trend_strength(sample, min(20, point_count // 10))
                    
                    # Store results in the metrics dictionary
                    for metric, value in [
                        ('direction_changes', direction_changes),
                        ('choppiness', choppiness),
                        ('tick_atr_pct', tick_atr_pct),
                        ('trend_strength', trend_strength)
                    ]:
                        if coin_key not in self.environment_data[metric][point_count]:
                            self.environment_data[metric][point_count][coin_key] = {}
                        self.environment_data[metric][point_count][coin_key][environment] = value
                    
                    # Store timestamp range
                    if coin_key not in self.environment_data['timestamp_ranges'][point_count]:
                        self.environment_data['timestamp_ranges'][point_count][coin_key] = {}
                    
                    self.environment_data['timestamp_ranges'][point_count][coin_key][environment] = {
                        'start': start_time,
                        'end': end_time,
                        'count': len(sample)
                    }
        except Exception as e:
            st.error(f"Error processing {coin_key}: {e}")
    
    def _calculate_direction_changes(self, prices):
        """Calculate the percentage of times the price direction changes."""
        try:
            price_changes = prices.diff().dropna()
            signs = np.sign(price_changes)
            direction_changes = (signs.shift(1) != signs).sum()
            
            total_periods = len(signs) - 1
            if total_periods > 0:
                direction_change_pct = (direction_changes / total_periods) * 100
            else:
                direction_change_pct = 0
            
            return direction_change_pct
        except Exception as e:
            return 50.0  # Return a reasonable default
    
    def _calculate_choppiness(self, prices, window):
        """
        Calculate average Choppiness Index using fixed 20-tick windows over most recent ticks.
        This matches the implementation requested.
        """
        try:
            # Use the most recent prices
            recent_prices = prices
            
            # Define fixed window size for choppiness calculation
            window_size_choppiness = 20
            all_choppiness_values = []
            
            # Process each valid 20-tick window
            for i in range(len(recent_prices) - window_size_choppiness + 1):
                window_prices = recent_prices.iloc[i:i+window_size_choppiness]
                
                # Skip if not a complete window
                if len(window_prices) < window_size_choppiness:
                    continue
                    
                # Calculate sum of absolute changes
                diff = window_prices.diff().dropna().abs()
                sum_abs_changes = diff.sum()
                
                # Calculate price range
                price_range = window_prices.max() - window_prices.min()
                
                # Avoid division by zero
                if price_range > 0:
                    # Calculate choppiness and cap at 1000
                    choppiness = 100 * sum_abs_changes / price_range
                    choppiness = min(choppiness, 1000)
                    all_choppiness_values.append(choppiness)
            
            # Calculate average choppiness across all valid windows
            if all_choppiness_values:
                avg_choppiness = sum(all_choppiness_values) / len(all_choppiness_values)
            else:
                avg_choppiness = 0
                
            return avg_choppiness
            
        except Exception as e:
            st.error(f"Error calculating choppiness: {e}")
            return 200.0  # Return a reasonable default value
    
    def _calculate_trend_strength(self, prices, window):
        """Calculate average Trend Strength."""
        try:
            diff = prices.diff().abs()
            sum_abs_changes = diff.rolling(window, min_periods=1).sum()
            net_change = (prices - prices.shift(window)).abs()
            
            # Avoid division by zero
            epsilon = 1e-10
            trend_strength = net_change / (sum_abs_changes + epsilon)
            
            # Handle NaN values
            trend_strength = pd.Series(trend_strength).fillna(0.5)
            
            return trend_strength.mean()
        except Exception as e:
            return 0.5  # Return a reasonable default value
    
    def create_metrics_table(self, metric_name, point_count):
        """Create a table showing the specified metric for all pairs across environments."""
        if point_count not in self.environment_data[metric_name]:
            return None
            
        # Collect metric data
        table_data = []
        for coin_key, environments in self.environment_data[metric_name][point_count].items():
            row = {'Pair': coin_key.replace('_', '/')}
            
            # Add data for each environment
            for env in self.all_environments:
                if env in environments:
                    row[f'{env}'] = environments[env]
                else:
                    row[f'{env}'] = None
                    
            table_data.append(row)
        
        # Create the dataframe
        if table_data:
            metrics_df = pd.DataFrame(table_data)
            return metrics_df
        
        return None
    
    def create_timestamp_range_table(self, point_count, pairs):
        """Create a table showing the time range for data collection."""
        if point_count not in self.environment_data['timestamp_ranges']:
            return None
            
        # Collect timestamp data
        time_data = []
        for pair in pairs:
            coin_key = pair.replace('/', '_')
            if coin_key in self.environment_data['timestamp_ranges'][point_count]:
                row = {'Pair': pair}
                
                # Add data for each environment
                for env in self.all_environments:
                    if env in self.environment_data['timestamp_ranges'][point_count][coin_key]:
                        env_range = self.environment_data['timestamp_ranges'][point_count][coin_key][env]
                        row[f'{env} Start'] = env_range['start']
                        row[f'{env} End'] = env_range['end']
                        row[f'{env} Count'] = env_range['count']
                
                time_data.append(row)
        
        # Create dataframe
        if time_data:
            time_df = pd.DataFrame(time_data)
            
            # Format datetime columns to be more readable
            for col in time_df.columns:
                if 'Start' in col or 'End' in col:
                    try:
                        time_df[col] = pd.to_datetime(time_df[col]).dt.strftime('%Y-%m-%d %H:%M:%S')
                    except:
                        pass
                        
            return time_df
        
        return None

# Function to fetch trading pairs from database
@st.cache_data(ttl=600)
def fetch_trading_pairs():
    query = """
    SELECT pair_name 
    FROM trade_pool_pairs 
    WHERE status in(1,2)
    ORDER BY pair_name
    """
    
    try:
        df = pd.read_sql(query, conn)
        return df['pair_name'].tolist()
    except Exception as e:
        st.error(f"Error fetching trading pairs: {e}")
        return ["BTC/USDT", "ETH/USDT"]  # Default pairs if database query fails

# Get trading pairs from database
all_pairs = fetch_trading_pairs()

# Setup sidebar with options
with st.sidebar:
    st.header("Analysis Parameters")
    
    # Initialize session state for selections if not present
    if 'selected_pairs' not in st.session_state:
        st.session_state.selected_pairs = ["ETH/USDT", "BTC/USDT"]  # Default selection
    
    # Create buttons with styling
    st.markdown("### Quick Selection")

    # Main selection buttons in a single row
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Select All Pairs", type="primary", use_container_width=True):
            st.session_state.selected_pairs = all_pairs
            st.rerun()

    with col2:
        if st.button("Clear All", type="secondary", use_container_width=True):
            st.session_state.selected_pairs = []
            st.rerun()

    # Additional options in a new row
    col3, col4 = st.columns(2)

    with col3:
        if st.button("Major Coins", use_container_width=True):
            st.session_state.selected_pairs = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT"]
            st.rerun()

    with col4:
        if st.button("Default Pairs", use_container_width=True):
            st.session_state.selected_pairs = ["ETH/USDT", "BTC/USDT"]
            st.rerun()

    st.markdown("---")  # Add a separator
    
    # Create the form
    with st.form("surf_analysis_form"):
        # Data retrieval window
        hours = st.number_input(
            "Hours to Look Back (for data retrieval)",
            min_value=1,
            max_value=168,
            value=8,
            help="How many hours of historical data to retrieve. This ensures enough data for point-based analysis."
        )
        
        st.info("Analysis will be performed on the most recent data points: 500, 1500, 2500, and 5000 points regardless of time span.")
        
        # Create multiselect for pairs
        selected_pairs = st.multiselect(
            "Select Pairs to Analyze",
            options=all_pairs,
            default=st.session_state.selected_pairs,
            help="Select one or more cryptocurrency pairs to analyze"
        )
        
        # Add submit button
        submit_button = st.form_submit_button("Analyze SURF Data")
        
        # Update session state when form is submitted
        if submit_button:
            st.session_state.selected_pairs = selected_pairs
            st.session_state.hours = hours
            st.session_state.analyze_clicked = True
            st.rerun()

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Choppiness", "Direction Changes", "Trend Strength", "ATR"])

# Check if we should run the analysis
if st.session_state.get('analyze_clicked', False):
    # Clear cache and previous data at start of analysis to ensure fresh data
    st.cache_data.clear()
    
    # Clear previous results from the page
    for tab in [tab1, tab2, tab3, tab4]:
        with tab:
            st.empty()  # Clear the tab content
    
    if not conn:
        st.error("Database connection not available.")
    elif not st.session_state.selected_pairs:
        st.error("Please enter at least one pair to analyze.")
    else:
        # Initialize analyzer
        analyzer = SurfAnalyzer()
        
        # Run analysis
        st.header("Analyzing SURF Environments")
        
        # Add a progress container
        progress_container = st.empty()
        with progress_container.container():
            st.info("Starting analysis... This may take a few minutes depending on the number of pairs selected.")
            
            with st.spinner("Fetching and analyzing data..."):
                results = analyzer.fetch_and_analyze(
                    conn=conn,
                    pairs_to_analyze=st.session_state.selected_pairs,
                    hours=st.session_state.hours
                )
            
            # Clear the progress container after analysis is complete
            progress_container.empty()
        
        if results:
            # Metrics tables for each tab and point count
            metrics_mapping = {
                tab1: 'choppiness',
                tab2: 'direction_changes',
                tab3: 'trend_strength',
                tab4: 'tick_atr_pct'
            }
            
            # Display metrics tables in each tab
            for tab, metric_name in metrics_mapping.items():
                with tab:
                    st.header(f"{analyzer.metric_display_names[metric_name]} Analysis")
                    
                    # Create subtabs for different point counts
                    point_tabs = st.tabs([f"{point_count} Points" for point_count in analyzer.point_counts])
                    
                    for i, point_count in enumerate(analyzer.point_counts):
                        with point_tabs[i]:
                            # Show time range information
                            st.subheader(f"Data Collection Time Ranges ({point_count} Points)")
                            time_df = analyzer.create_timestamp_range_table(point_count, st.session_state.selected_pairs)
                            
                            if time_df is not None:
                                st.dataframe(time_df, use_container_width=True)
                                st.info("Note: 'Start' is the oldest data point, 'End' is the most recent data point in the analysis.")
                            
                            # Show metric table
                            st.subheader(f"{analyzer.metric_display_names[metric_name]} Values")
                            metrics_df = analyzer.create_metrics_table(metric_name, point_count)
                            
                            if metrics_df is not None:
                                # Display unstyled table to avoid styling errors
                                st.dataframe(metrics_df, height=400, use_container_width=True)
                                
                                # Add download button for CSV
                                csv_data = metrics_df.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label=f"Download {metric_name}_{point_count} CSV",
                                    data=csv_data,
                                    file_name=f"{metric_name}_{point_count}_comparison.csv",
                                    mime="text/csv"
                                )
                                
                                # Add interpretation guide
                                st.subheader("Interpretation Guide")
                                
                                if metric_name == 'choppiness':
                                    interpretation_data = {
                                        'Range': ['< 50', '50-100', '100-200', '200-300', '> 300'],
                                        'Interpretation': [
                                            'Very directional price movement (strong trend)',
                                            'Directional price movement (clear trend)',
                                            'Moderately choppy price action',
                                            'Very choppy price action',
                                            'Extremely choppy, range-bound price action'
                                        ]
                                    }
                                    st.table(pd.DataFrame(interpretation_data))
                                
                                elif metric_name == 'direction_changes':
                                    interpretation_data = {
                                        'Range (%)': ['< 20', '20-35', '35-45', '45-55', '55-65', '> 65'],
                                        'Interpretation': [
                                            'Very strong directional consistency (strong trend)',
                                            'Strong directional consistency (clear trend)',
                                            'Moderate directional consistency (likely trending)',
                                            'Neutral - balanced directional changes (like random walk)',
                                            'Frequent direction changes (somewhat choppy)',
                                            'Very frequent direction changes (highly choppy/noisy)'
                                        ]
                                    }
                                    st.table(pd.DataFrame(interpretation_data))
                                
                                elif metric_name == 'trend_strength':
                                    interpretation_data = {
                                        'Range': ['< 0.1', '0.1-0.2', '0.2-0.4', '0.4-0.6', '> 0.6'],
                                        'Interpretation': [
                                            'Very weak trending (mostly oscillation)',
                                            'Weak trending behavior',
                                            'Moderate trending behavior',
                                            'Strong trending behavior',
                                            'Very strong trending behavior'
                                        ]
                                    }
                                    st.table(pd.DataFrame(interpretation_data))
                                
                                elif metric_name == 'tick_atr_pct':
                                    interpretation_data = {
                                        'Range (%)': ['< 0.05', '0.05-0.1', '0.1-0.25', '0.25-0.5', '0.5-1.0', '> 1.0'],
                                        'Interpretation': [
                                            'Extremely low volatility/range',
                                            'Very low volatility/range',
                                            'Low volatility/range',
                                            'Moderate volatility/range',
                                            'High volatility/range',
                                            'Very high volatility/range'
                                        ]
                                    }
                                    st.table(pd.DataFrame(interpretation_data))
                            else:
                                st.warning(f"No {metric_name} data available for {point_count} points")
        else:
            st.error("Failed to analyze data. Please try again with different parameters.")
        
        # Reset the analyze_clicked flag
        st.session_state.analyze_clicked = False

# Add explanation in the sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("About This Dashboard")
st.sidebar.markdown("""
This dashboard analyzes cryptocurrency prices across different SURF environments (UAT, PROD, SIT) and calculates the following metrics:

- **Choppiness**: Measures price oscillation within a range. Lower values indicate directional price movement.
- **Direction Changes (%)**: Frequency of price reversals. Lower values indicate more consistent price direction.
- **Trend Strength**: Measures directional price strength. Higher values indicate stronger trends.
- **ATR %**: Average True Range as percentage of mean price. Measures volatility.

The dashboard calculates these metrics at 500, 1500, 2500, and 5000 tick points to provide a comprehensive view of price behavior.
""")

# Add footer
st.sidebar.markdown("---")
st.sidebar.markdown(f"*Last updated: {now_sg.strftime('%Y-%m-%d %H:%M:%S')} (SGT)*")
