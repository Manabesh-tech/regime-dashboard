import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import psycopg2
import warnings
import pytz

# Suppress warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Crypto Exchange Analysis Dashboard",
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
        'database': 'replication_report',
        'user': 'public_replication',
        'password': '866^FKC4hllk'
    }
    
    try:
        conn = psycopg2.connect(
            host=db_params['host'],
            port=db_params['port'],
            database=db_params['database'],
            user=db_params['user'],
            password=db_params['password']
        )
        return conn, db_params
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return None, db_params

# Initialize connection
conn, db_params = init_db_connection()

# Main title
st.title("Crypto Exchange Analysis Dashboard")

# Create tabs
tab1, tab2 = st.tabs(["Parameter Comparison", "Rankings & Analysis"])

# Set up the timezone
singapore_timezone = pytz.timezone('Asia/Singapore')
now_utc = datetime.now(pytz.utc)
now_sg = now_utc.astimezone(singapore_timezone)
st.write(f"Current Singapore Time: {now_sg.strftime('%Y-%m-%d %H:%M:%S')}")

# Exchange comparison class
class ExchangeAnalyzer:
    """Specialized analyzer for comparing metrics between Surf and Rollbit"""
    
    def __init__(self):
        self.exchange_data = {}  # Will store data from different exchanges
        self.all_exchanges = ['rollbit', 'surf']  # Only Rollbit and Surf
        
        # Metrics to calculate and compare
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
        
        # Short names for metrics (for tables to avoid overflow)
        self.metric_short_names = {
            'direction_changes': 'Dir Chg',
            'choppiness': 'Chop',
            'tick_atr_pct': 'ATR%',
            'trend_strength': 'Trend'
        }
        
        # Point counts to analyze - updated to the requested values
        self.point_counts = [500, 1500, 2500, 5000]
        
        # The desired direction for each metric (whether higher or lower is better)
        self.metric_desired_direction = {
            'direction_changes': 'lower',  
            'choppiness': 'lower',        
            'tick_atr_pct': 'lower',       
            'trend_strength': 'lower'     
        }
        
        # Initialize exchange_data structure
        for metric in self.metrics:
            self.exchange_data[metric] = {point: {} for point in self.point_counts}
        
        # Initialize timestamp_ranges structure
        self.exchange_data['timestamp_ranges'] = {point: {} for point in self.point_counts}

    def _get_partition_tables(self, conn, start_date, end_date):
        """
        Get list of partition tables that need to be queried based on date range.
        Returns a list of table names (oracle_price_log_partition_YYYYMMDD)
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
        table_names = [f"oracle_price_log_partition_{date}" for date in dates]
        
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

    def _build_query_for_partition_tables(self, tables, pair_name, start_time, end_time, exchange):
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
            # For Surf data (production)
            if exchange == 'surf':
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
            else:
                # For Rollbit data
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
                    AND source_type = 1
                    AND pair_name = '{pair_name}'
                """
            
            union_parts.append(query)
        
        # Join with UNION and add ORDER BY at the end
        complete_query = " UNION ".join(union_parts) + " ORDER BY timestamp DESC"
        return complete_query

    def fetch_and_analyze(self, conn, pairs_to_analyze, hours=24):
        """
        Fetch data for Surf and Rollbit, analyze metrics, and calculate rankings.
        
        Args:
            conn: Database connection
            pairs_to_analyze: List of coin pairs to analyze
            hours: Hours to look back for data retrieval
        """
        # Always compare rollbit and surf
        exchanges_to_compare = ['rollbit', 'surf']
        primary_exchange = 'rollbit'
        secondary_exchange = 'surf'
        
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
                for exchange in exchanges_to_compare:
                    query = self._build_query_for_partition_tables(
                        partition_tables,
                        pair_name=pair,
                        start_time=start_time,
                        end_time=end_time,
                        exchange=exchange
                    )
                    all_queries[pair][exchange] = query
            
            # Execute all queries in quick succession
            pair_data = {}
            for i, pair in enumerate(pairs_to_analyze):
                progress_bar.progress(0.33 + (i) / len(pairs_to_analyze) / 3)  # Second third for query execution
                status_text.text(f"Executing queries for {pair} ({i+1}/{len(pairs_to_analyze)})")
                
                pair_data[pair] = {}
                
                # Execute rollbit and surf queries back-to-back for each pair
                for exchange in exchanges_to_compare:
                    query = all_queries[pair][exchange]
                    if query:
                        try:
                            df = pd.read_sql_query(query, conn)
                            if len(df) > 0:
                                pair_data[pair][exchange] = df
                                st.write(f"Found {len(df)} records for {exchange.upper()}_{pair}")
                            else:
                                st.warning(f"No data found for {exchange.upper()}_{pair}")
                        except Exception as e:
                            st.error(f"Database query error for {exchange.upper()}_{pair}: {e}")
            
            # Process the data for analysis
            for i, pair in enumerate(pairs_to_analyze):
                progress_bar.progress(0.67 + (i) / len(pairs_to_analyze) / 3)  # Final third for processing
                status_text.text(f"Analyzing {pair} ({i+1}/{len(pairs_to_analyze)})")
                
                # Process data for both exchanges
                coin_key = pair.replace('/', '_')
                for exchange in exchanges_to_compare:
                    if exchange in pair_data[pair]:
                        self._process_price_data(pair_data[pair][exchange], 'timestamp', 'price', coin_key, exchange)
            
            # Final progress update
            progress_bar.progress(1.0)
            status_text.text(f"Processing complete!")
            
            # Create comparison results
            comparison_results = self._create_comparison_results(primary_exchange, secondary_exchange)
            
            # Create individual rankings
            individual_rankings = {}
            for exchange in exchanges_to_compare:
                individual_rankings[exchange] = self._create_individual_rankings(exchange)
            
            return {
                'comparison_results': comparison_results,
                'individual_rankings': individual_rankings,
                'raw_data': self.exchange_data
            }
                
        except Exception as e:
            st.error(f"Error fetching and processing data: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _process_price_data(self, data, timestamp_col, price_col, coin_key, exchange):
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
                    
                    # Calculate each metric with improved error handling
                    direction_changes = self._calculate_direction_changes(sample)
                    choppiness = self._calculate_choppiness(sample, min(20, point_count // 10))
                    
                    # Calculate tick ATR
                    true_ranges = sample.diff().abs().dropna()
                    tick_atr = true_ranges.mean()
                    tick_atr_pct = (tick_atr / mean_price) * 100  # Convert to percentage of mean price
                    
                    # Calculate trend strength
                    trend_strength = self._calculate_trend_strength(sample, min(20, point_count // 10))
                    
                    # Store results in the metrics dictionary
                    if coin_key not in self.exchange_data['direction_changes'][point_count]:
                        self.exchange_data['direction_changes'][point_count][coin_key] = {}
                    if coin_key not in self.exchange_data['choppiness'][point_count]:
                        self.exchange_data['choppiness'][point_count][coin_key] = {}
                    if coin_key not in self.exchange_data['tick_atr_pct'][point_count]:
                        self.exchange_data['tick_atr_pct'][point_count][coin_key] = {}
                    if coin_key not in self.exchange_data['trend_strength'][point_count]:
                        self.exchange_data['trend_strength'][point_count][coin_key] = {}
                    if coin_key not in self.exchange_data['timestamp_ranges'][point_count]:
                        self.exchange_data['timestamp_ranges'][point_count][coin_key] = {}
                    
                    # Store metrics
                    self.exchange_data['direction_changes'][point_count][coin_key][exchange] = direction_changes
                    self.exchange_data['choppiness'][point_count][coin_key][exchange] = choppiness
                    self.exchange_data['tick_atr_pct'][point_count][coin_key][exchange] = tick_atr_pct
                    self.exchange_data['trend_strength'][point_count][coin_key][exchange] = trend_strength
                    
                    # Store timestamp range
                    self.exchange_data['timestamp_ranges'][point_count][coin_key][exchange] = {
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
            return 50.0  # Return a reasonable default instead of zero
    
    def _calculate_choppiness(self, prices, window):
        """Calculate average Choppiness Index with improved error handling."""
        try:
            diff = prices.diff().abs()
            sum_abs_changes = diff.rolling(window, min_periods=1).sum()
            price_range = prices.rolling(window, min_periods=1).max() - prices.rolling(window, min_periods=1).min()
            
            # Check for zero price range
            if (price_range == 0).any():
                # Replace zeros with a small value to avoid division by zero
                price_range = price_range.replace(0, 1e-10)
            
            # Avoid division by zero
            epsilon = 1e-10
            choppiness = 100 * sum_abs_changes / (price_range + epsilon)
            
            # Cap extreme values and handle NaN
            choppiness = np.minimum(choppiness, 1000)
            choppiness = choppiness.fillna(200)  # Replace NaN with a reasonable default
            
            return choppiness.mean()
        except Exception as e:
            return 200.0  # Return a reasonable default value
    
    def _calculate_trend_strength(self, prices, window):
        """Calculate average Trend Strength with improved error handling."""
        try:
            diff = prices.diff().abs()
            sum_abs_changes = diff.rolling(window, min_periods=1).sum()
            net_change = (prices - prices.shift(window)).abs()
            
            # Avoid division by zero
            epsilon = 1e-10
            
            # Check if sum_abs_changes is close to zero
            trend_strength = np.where(
                sum_abs_changes > epsilon,
                net_change / (sum_abs_changes + epsilon),
                0.5  # Default value when there's no change
            )
            
            # Convert to pandas Series if it's a numpy array
            if isinstance(trend_strength, np.ndarray):
                trend_strength = pd.Series(trend_strength, index=net_change.index)
            
            # Handle NaN values
            trend_strength = pd.Series(trend_strength).fillna(0.5)
            
            return trend_strength.mean()
        except Exception as e:
            return 0.5  # Return a reasonable default value
    
    def _create_comparison_results(self, primary_exchange, secondary_exchange):
        """Create comparison results between the two exchanges for all point counts."""
        comparison_results = {}
        
        for point_count in self.point_counts:
            comparison_data = []
            
            # Get all coins that have data for secondary exchange
            all_coins = set()
            for metric in self.metrics:
                for coin, exchanges in self.exchange_data[metric][point_count].items():
                    if secondary_exchange in exchanges:
                        all_coins.add(coin)
            
            # For each coin, calculate relative performance for each metric
            for coin in all_coins:
                row = {'Coin': coin.replace('_', '/')}
                relative_scores = []
                
                for metric in self.metrics:
                    # Check if we have data for secondary exchange for this metric
                    if (coin in self.exchange_data[metric][point_count] and
                        secondary_exchange in self.exchange_data[metric][point_count][coin]):
                        
                        secondary_value = self.exchange_data[metric][point_count][coin][secondary_exchange]
                        
                        # Add to the row
                        row[f'{self.metric_short_names[metric]} {secondary_exchange.upper()}'] = secondary_value
                        
                        # If we have primary exchange data, add it and calculate relative scores
                        if primary_exchange in self.exchange_data[metric][point_count][coin]:
                            primary_value = self.exchange_data[metric][point_count][coin][primary_exchange]
                            
                            # Calculate absolute and percentage difference
                            abs_diff = secondary_value - primary_value
                            pct_diff = (abs_diff / primary_value * 100) if primary_value != 0 else 0
                            
                            # Calculate relative performance score (100 means equal to primary, >100 means better)
                            if metric == 'trend_strength':
                                # For trend_strength, lower is better, so inverse the ratio
                                if secondary_value == 0:
                                    # Edge case
                                    relative_score = 100
                                else:
                                    relative_score = (primary_value / secondary_value) * 100
                            else:
                                # For all other metrics, higher is better
                                if primary_value == 0:
                                    # Edge case
                                    relative_score = 100 if secondary_value == 0 else 200
                                else:
                                    relative_score = (secondary_value / primary_value) * 100
                            
                            relative_scores.append(relative_score)
                            
                            # Add primary exchange data and differences
                            row[f'{self.metric_short_names[metric]} {primary_exchange.upper()}'] = primary_value
                            row[f'{self.metric_short_names[metric]} Diff'] = abs_diff
                            row[f'{self.metric_short_names[metric]} Diff %'] = pct_diff
                            row[f'{self.metric_short_names[metric]} Score'] = relative_score
                        else:
                            # If no primary exchange data, just show secondary exchange data
                            row[f'{self.metric_short_names[metric]} {primary_exchange.upper()}'] = None
                            row[f'{self.metric_short_names[metric]} Diff'] = None
                            row[f'{self.metric_short_names[metric]} Diff %'] = None
                            row[f'{self.metric_short_names[metric]} Score'] = None
                
                # Calculate overall relative score (average of individual scores)
                if relative_scores:
                    row['Overall Score'] = sum(relative_scores) / len(relative_scores)
                else:
                    row['Overall Score'] = None
                
                comparison_data.append(row)
            
            # Create DataFrame and sort by relative score (highest first)
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                comparison_df = comparison_df.sort_values('Overall Score', ascending=False)
                comparison_results[point_count] = comparison_df
            else:
                comparison_results[point_count] = None
        
        return comparison_results
    
    def _create_individual_rankings(self, exchange):
        """Create rankings for a specific exchange across all metrics and point counts."""
        rankings = {}
        
        for point_count in self.point_counts:
            point_rankings = {}
            
            for metric in self.metrics:
                # Get all coins that have data for this exchange and metric
                coin_data = {}
                for coin, exchanges in self.exchange_data[metric][point_count].items():
                    if exchange in exchanges:
                        coin_data[coin] = exchanges[exchange]
                
                if coin_data:
                    # Create DataFrame
                    df = pd.DataFrame({
                        'Coin': [coin.replace('_', '/') for coin in coin_data.keys()],
                        'Value': list(coin_data.values())
                    })
                    
                    # Sort based on metric (ascending or descending)
                    # For trend_strength, lower is better
                    # For all other metrics, higher is now better
                    ascending = True if metric == 'trend_strength' else False
                    
                    df = df.sort_values('Value', ascending=ascending)
                    
                    # Add rank column
                    df.insert(0, 'Rank', range(1, len(df) + 1))
                    
                    point_rankings[metric] = df
            
            rankings[point_count] = point_rankings
        
        return rankings
    
    def create_parameter_comparison_table(self):
        """
        Create a comprehensive table with all metrics for all pairs across all point counts.
        This is for tab 1 to display a huge table of parameters.
        """
        # Always use rollbit as primary and surf as secondary
        primary_exchange = 'rollbit'
        secondary_exchange = 'surf'
        
        # First, collect all coins that have data for secondary exchange
        all_coins = set()
        for metric in self.metrics:
            for point_count in self.point_counts:
                for coin in self.exchange_data[metric][point_count].keys():
                    if secondary_exchange in self.exchange_data[metric][point_count][coin]:
                        all_coins.add(coin)
        
        # Create a huge dataframe with all metrics
        rows = []
        for coin in sorted(all_coins):
            row = {'Coin': coin.replace('_', '/')}
            
            # Add metrics for each point count
            for point_count in self.point_counts:
                for metric in self.metrics:
                    # Check if we have secondary exchange data
                    if (coin in self.exchange_data[metric][point_count] and 
                        secondary_exchange in self.exchange_data[metric][point_count][coin]):
                        row[f'{self.metric_short_names[metric]} {secondary_exchange.upper()} ({point_count})'] = self.exchange_data[metric][point_count][coin][secondary_exchange]
                    else:
                        row[f'{self.metric_short_names[metric]} {secondary_exchange.upper()} ({point_count})'] = None
                    
                    # Check if we have primary exchange data
                    if (coin in self.exchange_data[metric][point_count] and 
                        primary_exchange in self.exchange_data[metric][point_count][coin]):
                        row[f'{self.metric_short_names[metric]} {primary_exchange.upper()} ({point_count})'] = self.exchange_data[metric][point_count][coin][primary_exchange]
                    else:
                        row[f'{self.metric_short_names[metric]} {primary_exchange.upper()} ({point_count})'] = None
            
            rows.append(row)
        
        # Create the dataframe
        if rows:
            comparison_df = pd.DataFrame(rows)
            return comparison_df
        else:
            return None
    
    def create_timestamp_range_table(self, point_count, pairs):
        """Create a table showing the time range for data collection."""
        if point_count not in self.exchange_data['timestamp_ranges']:
            return None
            
        # Collect timestamp data
        time_data = []
        for pair in pairs:
            coin_key = pair.replace('/', '_')
            if coin_key in self.exchange_data['timestamp_ranges'][point_count]:
                row = {'Pair': pair}
                
                # Add rollbit data if available
                if 'rollbit' in self.exchange_data['timestamp_ranges'][point_count][coin_key]:
                    rollbit_range = self.exchange_data['timestamp_ranges'][point_count][coin_key]['rollbit']
                    row['Rollbit Start'] = rollbit_range['start']
                    row['Rollbit End'] = rollbit_range['end']
                    row['Rollbit Count'] = rollbit_range['count']
                
                # Add surf data if available
                if 'surf' in self.exchange_data['timestamp_ranges'][point_count][coin_key]:
                    surf_range = self.exchange_data['timestamp_ranges'][point_count][coin_key]['surf']
                    row['Surf Start'] = surf_range['start']
                    row['Surf End'] = surf_range['end']
                    row['Surf Count'] = surf_range['count']
                
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
    WHERE status = 1
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

# Setup sidebar with simplified options
with st.sidebar:
    st.header("Analysis Parameters")
    
    # Initialize session state for selections if not present
    if 'selected_pairs' not in st.session_state:
        st.session_state.selected_pairs = ["ETH/USDT", "BTC/USDT"]  # Default selection
    
    # Create buttons with more prominent styling
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
    with st.form("exchange_comparison_form"):
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
        submit_button = st.form_submit_button("Analyze Exchanges")
        
        # Update session state when form is submitted
        if submit_button:
            st.session_state.selected_pairs = selected_pairs
            st.session_state.hours = hours
            st.session_state.analyze_clicked = True
            st.rerun()

# Check if we should run the analysis
if st.session_state.get('analyze_clicked', False):
    # Clear cache and previous data at start of analysis to ensure fresh data
    st.cache_data.clear()
    
    # Clear previous results from the page
    for tab in [tab1, tab2]:
        with tab:
            st.empty()  # Clear the tab content
    
    if not conn:
        st.error("Database connection not available.")
    elif not st.session_state.selected_pairs:
        st.error("Please enter at least one pair to analyze.")
    else:
        # Initialize analyzer
        analyzer = ExchangeAnalyzer()
        
        # Run analysis
        st.header("Comparing ROLLBIT vs SURF")
        
        # Add a progress container
        progress_container = st.empty()
        with progress_container.container():
            st.info("Starting analysis... This may take a few minutes depending on the number of pairs selected.")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("Fetching and analyzing data..."):
                results = analyzer.fetch_and_analyze(
                    conn=conn,
                    pairs_to_analyze=st.session_state.selected_pairs,
                    hours=st.session_state.hours
                )
            
            # Clear the progress container after analysis is complete
            progress_container.empty()
        
        if results:
            # Create parameter comparison table for Tab 1
            with tab1:
                st.header("Parameter Comparison Table")
                st.write("This table shows all metrics for all pairs across different point counts.")
                
                comparison_table = analyzer.create_parameter_comparison_table()
                
                if comparison_table is not None:
                    # Style the table
                    def style_comparison_table(val):
                        """Style cells, highlight differences."""
                        if pd.isna(val):
                            return 'background-color: #f2f2f2'  # Light gray for missing values
                        return ''
                    
                    # Display the table with horizontal scrolling
                    st.dataframe(
                        comparison_table.style.applymap(style_comparison_table),
                        height=600,
                        use_container_width=True,
                    )
                    
                    # Add download button for the table
                    csv = comparison_table.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"parameter_comparison_rollbit_vs_surf.csv",
                        mime="text/csv"
                    )
                    
                    # Add time range information
                    st.subheader("Data Collection Time Ranges")
                    st.write("This shows the exact time range for data analyzed at each point count:")
                    
                    for point_count in analyzer.point_counts:
                        st.write(f"#### {point_count} Points")
                        time_df = analyzer.create_timestamp_range_table(point_count, st.session_state.selected_pairs)
                        
                        if time_df is not None:
                            st.dataframe(time_df, use_container_width=True)
                            st.info("Note: 'Start' is the oldest data point, 'End' is the most recent data point in the analysis.")
                        else:
                            st.warning(f"No timestamp data available for {point_count} points")
                else:
                    st.warning("No comparison data available.")
            
            # Rankings and analysis for Tab 2
            with tab2:
                st.header("Rankings & Analysis")
                
                if results['comparison_results']:
                    # Create subtabs for different point counts
                    valid_point_counts = [count for count in analyzer.point_counts if count in results['comparison_results'] and results['comparison_results'][count] is not None]
                    
                    if valid_point_counts:
                        point_count_tabs = st.tabs([f"{count} Points" for count in valid_point_counts])
                        
                        for i, point_count in enumerate(valid_point_counts):
                            with point_count_tabs[i]:
                                df = results['comparison_results'][point_count]
                                
                                if df is not None and not df.empty:
                                    # Style the DataFrame for relative scores
                                    def highlight_scores(val):
                                        try:
                                            if isinstance(val.name, str) and 'Score' in val.name:
                                                if val > 130:
                                                    return 'background-color: #60b33c; color: white; font-weight: bold'
                                                elif val > 110:
                                                    return 'background-color: #a0d995; color: black'
                                                elif val > 90:
                                                    return 'background-color: #f1f1aa; color: black'
                                                elif val > 70:
                                                    return 'background-color: #ffc299; color: black'
                                                else:
                                                    return 'background-color: #ff8080; color: black; font-weight: bold'
                                        except:
                                            pass
                                        return ''
                                    
                                    # Display data collection time range
                                    st.subheader("Data Collection Time Ranges")
                                    time_df = analyzer.create_timestamp_range_table(point_count, st.session_state.selected_pairs)
                                    
                                    if time_df is not None:
                                        st.dataframe(time_df, height=300, use_container_width=True)
                                        st.info("Note: 'Start' is the oldest data point, 'End' is the most recent data point in the analysis.")
                                    
                                    # Display the data
                                    st.subheader(f"Relative Performance: {point_count} Points")
                                    st.dataframe(
                                        df.style.applymap(highlight_scores),
                                        height=400,
                                        use_container_width=True
                                    )
                                    
                                    # Create visualization
                                    st.subheader(f"Top and Bottom Performers")
                                    
                                    # Convert Overall Score to numeric, replacing None with 0
                                    df['Overall Score'] = pd.to_numeric(df['Overall Score'], errors='coerce').fillna(0)
                                    
                                    # Top 10 and bottom 10 coins
                                    top_10 = df.nlargest(10, 'Overall Score')
                                    bottom_10 = df.nsmallest(10, 'Overall Score')
                                    
                                    # Combined visualization
                                    fig = go.Figure()
                                    
                                    # Add top 10
                                    fig.add_trace(go.Bar(
                                        x=top_10['Coin'],
                                        y=top_10['Overall Score'],
                                        name='Top Performers',
                                        marker_color='green'
                                    ))
                                    
                                    # Add bottom 10
                                    fig.add_trace(go.Bar(
                                        x=bottom_10['Coin'],
                                        y=bottom_10['Overall Score'],
                                        name='Bottom Performers',
                                        marker_color='red'
                                    ))
                                    
                                    # Add reference line at 100
                                    fig.add_shape(
                                        type="line",
                                        x0=-0.5,
                                        y0=100,
                                        x1=len(top_10) + len(bottom_10) - 0.5,
                                        y1=100,
                                        line=dict(
                                            color="black",
                                            width=2,
                                            dash="dash",
                                        )
                                    )
                                    
                                    fig.update_layout(
                                        title=f"SURF Performance Relative to ROLLBIT (100 = Equal)",
                                        xaxis_title="Coin",
                                        yaxis_title="Relative Performance Score",
                                        barmode='group',
                                        height=500
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True, key=f"performance_{point_count}_chart")
                                    
                                    # Add metric-by-metric analysis
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        # Individual rankings for Rollbit
                                        st.subheader("ROLLBIT Rankings")
                                        
                                        if point_count in results['individual_rankings']['rollbit']:
                                            # Create subtabs for each metric
                                            valid_metrics_primary = [m for m in analyzer.metrics 
                                                                   if m in results['individual_rankings']['rollbit'][point_count] 
                                                                   and results['individual_rankings']['rollbit'][point_count][m] is not None]
                                            
                                            if valid_metrics_primary:
                                                metric_tabs_primary = st.tabs([analyzer.metric_display_names[m] for m in valid_metrics_primary])
                                                
                                                for j, metric in enumerate(valid_metrics_primary):
                                                    with metric_tabs_primary[j]:
                                                        metric_df = results['individual_rankings']['rollbit'][point_count][metric]
                                                        if not metric_df.empty:
                                                            st.dataframe(metric_df, height=300, use_container_width=True)
                                                            
                                                            # Bar chart of top 10
                                                            top_10_metric = metric_df.head(10)
                                                            fig = px.bar(
                                                                top_10_metric, 
                                                                x='Coin', 
                                                                y='Value',
                                                                title=f"Top 10 by {analyzer.metric_display_names[metric]}",
                                                                color='Rank',
                                                                color_continuous_scale='Viridis_r'
                                                            )
                                                            st.plotly_chart(fig, use_container_width=True, key=f"rollbit_{point_count}_{metric}_chart")
                                        else:
                                            st.warning(f"No ranking data available for ROLLBIT at {point_count} points")
                                    
                                    with col2:
                                        # Individual rankings for Surf
                                        st.subheader("SURF Rankings")
                                        
                                        if point_count in results['individual_rankings']['surf']:
                                            # Create subtabs for each metric
                                            valid_metrics_secondary = [m for m in analyzer.metrics 
                                                                     if m in results['individual_rankings']['surf'][point_count] 
                                                                     and results['individual_rankings']['surf'][point_count][m] is not None]
                                            
                                            if valid_metrics_secondary:
                                                metric_tabs_secondary = st.tabs([analyzer.metric_display_names[m] for m in valid_metrics_secondary])
                                                
                                                for j, metric in enumerate(valid_metrics_secondary):
                                                    with metric_tabs_secondary[j]:
                                                        metric_df = results['individual_rankings']['surf'][point_count][metric]
                                                        if not metric_df.empty:
                                                            st.dataframe(metric_df, height=300, use_container_width=True)
                                                            
                                                            # Bar chart of top 10
                                                            top_10_metric = metric_df.head(10)
                                                            fig = px.bar(
                                                                top_10_metric, 
                                                                x='Coin', 
                                                                y='Value',
                                                                title=f"Top 10 by {analyzer.metric_display_names[metric]}",
                                                                color='Rank',
                                                                color_continuous_scale='Viridis_r'
                                                            )
                                                            st.plotly_chart(fig, use_container_width=True, key=f"surf_{point_count}_{metric}_chart")
                                        else:
                                            st.warning(f"No ranking data available for SURF at {point_count} points")
                                else:
                                    st.warning(f"No data available for {point_count} points")
                    else:
                        st.warning("No valid data available for any point count.")
                else:
                    st.warning("No comparison results available.")
        else:
            st.error("Failed to analyze data. Please try again with different parameters.")
        
        # Reset the analyze_clicked flag
        st.session_state.analyze_clicked = False

# Add explanation in the sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("About This Dashboard")
st.sidebar.markdown("""
This dashboard analyzes cryptocurrency prices between Rollbit and Surf exchanges and calculates various metrics:

- **Direction Changes (%)**: Frequency of price reversals
- **Choppiness**: Measures price oscillation within a range
- **Tick ATR %**: Average True Range as percentage of mean price
- **Trend Strength**: Measures directional price strength

The dashboard compares these metrics and provides rankings and visualizations for various point counts (500, 1500, 2500, and 5000).
""")

# Add footer
st.sidebar.markdown("---")
st.sidebar.markdown(f"*Last updated: {now_sg.strftime('%Y-%m-%d %H:%M:%S')} (SGT)*")
