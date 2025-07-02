import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import psycopg2
import warnings
import pytz
import time

# Suppress warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Crypto Price Stability Analysis",
    page_icon="📊",
    layout="wide"
)

# Define constants for analysis
INTERVAL_MINUTES = 15  # 15-minute intervals
TOLERANCE_PERCENTAGE = 0.5  # ±0.5% tolerance

# Configure database connection
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
st.title("Cryptocurrency Price Stability Analysis")

# Set up the timezone
singapore_timezone = pytz.timezone('Asia/Singapore')
now_utc = datetime.now(pytz.utc)
now_sg = now_utc.astimezone(singapore_timezone)
st.write(f"Current Singapore Time: {now_sg.strftime('%Y-%m-%d %H:%M:%S')}")

class PriceStabilityAnalyzer:
    """Analyzer for cryptocurrency price stability within intervals"""
    
    def __init__(self):
        self.interval_minutes = INTERVAL_MINUTES
        self.tolerance_percentage = TOLERANCE_PERCENTAGE
        
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

    def _build_query_for_partition_tables(self, tables, pair_name, start_time, end_time):
        """
        Build a complete UNION query for multiple partition tables.
        Surf data only (source_type = 0)
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
            # For Surf data (production) only
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
        complete_query = " UNION ".join(union_parts) + " ORDER BY timestamp ASC"
        return complete_query

    def analyze_price_stability(self, df, interval_minutes=15, tolerance_percentage=0.5):
        """
        Analyze price stability within 15-minute intervals.
        
        Args:
            df: DataFrame with 'timestamp' and 'price' columns
            interval_minutes: Size of the time interval in minutes (default 15)
            tolerance_percentage: Percentage tolerance around median (default 0.5%)
            
        Returns:
            DataFrame with intervals and percentage of data points within tolerance
        """
        if df.empty:
            return pd.DataFrame()
        
        # Ensure timestamp is in datetime format
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create interval labels (15-minute bins)
        df['interval'] = df['timestamp'].dt.floor(f'{interval_minutes}min')
        
        # Group by interval
        interval_groups = df.groupby('interval')
        
        results = []
        for interval, group in interval_groups:
            if len(group) >= 3:  # Need at least a few points to make meaningful calculation
                prices = group['price'].values
                med_price = np.median(prices)
                
                # Calculate upper and lower bounds (±0.5% of median)
                lower_bound = med_price * (1 - tolerance_percentage/100)
                upper_bound = med_price * (1 + tolerance_percentage/100)
                
                # Count how many points fall within the range
                in_range = ((prices >= lower_bound) & (prices <= upper_bound)).sum()
                percentage_in_range = (in_range / len(prices)) * 100
                
                results.append({
                    'interval': interval,
                    'median_price': med_price,
                    'points_count': len(prices),
                    'percentage_in_range': percentage_in_range
                })
        
        result_df = pd.DataFrame(results)
        return result_df

    def fetch_and_analyze_data(self, conn, pairs_to_analyze, hours=24, status_elements=None):
        """
        Fetch data for selected pairs and analyze stability in 15-minute intervals.
        Only for Surf exchange (source_type = 0)
        
        Args:
            conn: Database connection
            pairs_to_analyze: List of coin pairs to analyze
            hours: Hours to look back for data retrieval
            status_elements: Dict with UI elements for displaying progress
            
        Returns:
            Dictionary of DataFrames with stability analysis for each pair
        """
        # Use explicit Singapore timezone for all time calculations
        singapore_tz = pytz.timezone('Asia/Singapore')
        now = datetime.now(singapore_tz)
        
        # Calculate times in Singapore timezone
        end_time = now.strftime("%Y-%m-%d %H:%M:%S")
        start_time = (now - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
        
        # Update status
        if status_elements:
            status_elements['main_status'].markdown("### 📊 Analysis Status: Initializing...")
            status_elements['progress_bar'].progress(0.05)
            status_elements['details'].info(f"Retrieving data from the last {hours} hours")
            status_elements['sub_status'].write(f"Start time: {start_time} (SGT)")
            status_elements['sub_status'].write(f"End time: {end_time} (SGT)")
            time.sleep(0.5)  # Brief pause to show status
        
        try:
            # Get relevant partition tables for this time range
            if status_elements:
                status_elements['main_status'].markdown("### 📊 Analysis Status: Finding partition tables...")
                status_elements['progress_bar'].progress(0.1)
                
            partition_tables = self._get_partition_tables(conn, start_time, end_time)
            
            if not partition_tables:
                # If no tables found, try looking one day earlier (for edge cases)
                if status_elements:
                    status_elements['main_status'].markdown("### 📊 Analysis Status: No tables found, looking back further...")
                    status_elements['details'].warning("No tables found for the specified range, trying to look back one more day...")
                    status_elements['progress_bar'].progress(0.15)
                    
                alt_start_time = (now - timedelta(hours=hours+24)).strftime("%Y-%m-%d %H:%M:%S")
                partition_tables = self._get_partition_tables(conn, alt_start_time, end_time)
                
                if not partition_tables:
                    if status_elements:
                        status_elements['main_status'].markdown("### ❌ Analysis Status: No data available")
                        status_elements['details'].error("No data tables available for the selected time range, even with extended lookback.")
                        status_elements['progress_bar'].progress(1.0)
                    return None
            
            # Update status with found tables
            if status_elements:
                status_elements['main_status'].markdown("### 📊 Analysis Status: Found partition tables")
                status_elements['details'].success(f"Found {len(partition_tables)} partition tables")
                status_elements['sub_status'].write(f"Tables: {', '.join(partition_tables)}")
                status_elements['progress_bar'].progress(0.2)
                time.sleep(0.5)  # Brief pause to show status
            
            # Build and execute queries for each pair
            stability_results = {}
            daily_averages = {}
            
            total_pairs = len(pairs_to_analyze)
            
            for i, pair in enumerate(pairs_to_analyze):
                # Calculate progress percentage (20% baseline + up to 70% for processing pairs)
                progress_pct = 0.2 + (i / total_pairs * 0.7)
                
                # Update status
                if status_elements:
                    status_elements['main_status'].markdown(f"### 📊 Analysis Status: Processing pair {i+1} of {total_pairs}")
                    status_elements['progress_bar'].progress(progress_pct)
                    status_elements['details'].info(f"Analyzing {pair}")
                    status_elements['sub_status'].write(f"Building database query...")
                
                # Build query
                query = self._build_query_for_partition_tables(
                    partition_tables,
                    pair_name=pair,
                    start_time=start_time,
                    end_time=end_time
                )
                
                if query:
                    try:
                        # Update status
                        if status_elements:
                            status_elements['sub_status'].write(f"Executing database query for {pair}...")
                        
                        # Execute query
                        df = pd.read_sql_query(query, conn)
                        
                        if len(df) > 0:
                            # Update status
                            if status_elements:
                                status_elements['sub_status'].write(f"Found {len(df)} price records for {pair}")
                            
                            # Analyze price stability
                            if status_elements:
                                status_elements['sub_status'].write(f"Calculating price stability metrics...")
                                
                            stability_df = self.analyze_price_stability(
                                df, 
                                interval_minutes=self.interval_minutes,
                                tolerance_percentage=self.tolerance_percentage
                            )
                            
                            if not stability_df.empty:
                                stability_results[pair] = stability_df
                                
                                # Calculate daily average
                                daily_avg = stability_df['percentage_in_range'].mean()
                                daily_averages[pair] = daily_avg
                                
                                # Update status
                                if status_elements:
                                    status_elements['sub_status'].write(f"Daily average: {daily_avg:.2f}% of prices within ±{self.tolerance_percentage}% of interval median")
                                    status_elements['sub_status'].write(f"✅ {pair} analysis complete")
                            else:
                                # Update status
                                if status_elements:
                                    status_elements['sub_status'].write(f"⚠️ No stability data calculated for {pair}")
                        else:
                            # Update status
                            if status_elements:
                                status_elements['sub_status'].write(f"⚠️ No data found for {pair}")
                    except Exception as e:
                        # Update status
                        if status_elements:
                            status_elements['sub_status'].write(f"❌ Error processing {pair}: {str(e)}")
                
                # Brief pause between pairs to show status updates
                time.sleep(0.2)
            
            # Final progress update
            if status_elements:
                status_elements['main_status'].markdown("### ✅ Analysis Status: Complete!")
                status_elements['progress_bar'].progress(1.0)
                status_elements['details'].success(f"Successfully analyzed {len(stability_results)} pairs")
                status_elements['sub_status'].write(f"Analysis complete! Rendering results...")
                time.sleep(1)  # Pause to show completion status
            
            return {
                'stability_results': stability_results,
                'daily_averages': daily_averages
            }
                
        except Exception as e:
            if status_elements:
                status_elements['main_status'].markdown("### ❌ Analysis Status: Error")
                status_elements['details'].error(f"Error fetching and processing data: {str(e)}")
                status_elements['progress_bar'].progress(1.0)
            import traceback
            traceback.print_exc()
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

# Create tabs for different views
tab1, tab2 = st.tabs(["Time Series Analysis", "Daily Rankings"])

# Setup sidebar with analysis options
with st.sidebar:
    st.header("Analysis Parameters")
    
    # Initialize session state for selections if not present
    if 'selected_pairs' not in st.session_state:
        st.session_state.selected_pairs = ["BTC/USDT", "ETH/USDT"]  # Default selection
    
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
    with st.form("price_stability_form"):
        # Data retrieval window
        hours = st.number_input(
            "Hours to Look Back",
            min_value=1,
            max_value=168,
            value=24,
            help="How many hours of historical data to retrieve"
        )
        
        # Create multiselect for pairs
        selected_pairs = st.multiselect(
            "Select Pairs to Analyze",
            options=all_pairs,
            default=st.session_state.selected_pairs,
            help="Select one or more cryptocurrency pairs to analyze"
        )
        
        # Add submit button
        submit_button = st.form_submit_button("Analyze Price Stability")
        
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
        analyzer = PriceStabilityAnalyzer()
        
        # Run analysis with highly visible progress tracking
        st.header(f"Price Stability Analysis for SURF")
        
        # Create large, highly visible progress indicators
        progress_container = st.container()
        
        with progress_container:
            # Create all status elements
            main_status = st.empty()  # For the main status header
            progress_bar = st.progress(0)  # Main progress bar
            details = st.empty()    # For detailed status messages
            sub_status = st.empty()  # For sub-status messages
            
            # Package all status elements for easy updating
            status_elements = {
                'main_status': main_status,
                'progress_bar': progress_bar,
                'details': details,
                'sub_status': sub_status
            }
            
            # Initialize status
            main_status.markdown("### 🚀 Analysis Status: Starting...")
            details.info("Initializing price stability analysis...")
            
            # Call the analyzer with status elements
            results = analyzer.fetch_and_analyze_data(
                conn=conn,
                pairs_to_analyze=st.session_state.selected_pairs,
                hours=st.session_state.hours,
                status_elements=status_elements
            )
            
            # Keep progress visible for a moment to show completion
            time.sleep(1)
        
        if results:
            # Store results in session state
            st.session_state.stability_results = results['stability_results']
            st.session_state.daily_averages = results['daily_averages']
            
            # Show final completion message with count
            st.success(f"Analysis complete! Analyzed {len(results['stability_results'])} pairs successfully.")
        else:
            st.error("Failed to analyze data. Please try again with different parameters.")
        
        # Reset the analyze_clicked flag
        st.session_state.analyze_clicked = False

# Display results if available
if 'stability_results' in st.session_state and 'daily_averages' in st.session_state:
    stability_results = st.session_state.stability_results
    daily_averages = st.session_state.daily_averages
    
    # Tab 1: Time Series Analysis
    with tab1:
        st.header("Time Series Analysis")
        st.write(f"Percentage of prices within ±{TOLERANCE_PERCENTAGE}% of {INTERVAL_MINUTES}-minute interval median")
        
        # Create time series plots for each pair
        for pair, df in stability_results.items():
            st.subheader(f"{pair} Stability")
            
            # Create line chart
            fig = px.line(
                df,
                x='interval',
                y='percentage_in_range',
                title=f"{pair} - Percentage of prices within ±{TOLERANCE_PERCENTAGE}% of interval median",
                labels={
                    'interval': 'Time (15-minute intervals)',
                    'percentage_in_range': '% within ±0.5% of median'
                }
            )
            
            # Add reference line at 100%
            fig.add_shape(
                type="line",
                x0=df['interval'].min(),
                y0=100,
                x1=df['interval'].max(),
                y1=100,
                line=dict(
                    color="green",
                    width=1,
                    dash="dash",
                )
            )
            
            # Add reference line at 50%
            fig.add_shape(
                type="line",
                x0=df['interval'].min(),
                y0=50,
                x1=df['interval'].max(),
                y1=50,
                line=dict(
                    color="red",
                    width=1,
                    dash="dash",
                )
            )
            
            # Update layout
            fig.update_layout(
                height=400,
                xaxis_title="Time (15-minute intervals)",
                yaxis_title="Percentage within ±0.5%",
                yaxis=dict(
                    range=[0, 105]  # Set y-axis range from 0 to 105%
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add daily average
            daily_avg = daily_averages.get(pair, 0)
            st.write(f"Daily average: {daily_avg:.2f}% of prices within ±{TOLERANCE_PERCENTAGE}% of interval median")
            
            # Add download button for the data
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"Download {pair} Data",
                data=csv,
                file_name=f"price_stability_surf_{pair.replace('/', '_')}.csv",
                mime="text/csv"
            )
            
            st.markdown("---")
    
    # Tab 2: Daily Rankings
    with tab2:
        st.header("Daily Ranking by Price Stability")
        st.write(f"Average percentage of prices within ±{TOLERANCE_PERCENTAGE}% of {INTERVAL_MINUTES}-minute interval median")
        
        # Create DataFrame with daily averages
        if daily_averages:
            avg_df = pd.DataFrame({
                'Pair': list(daily_averages.keys()),
                'Average % within ±0.5%': list(daily_averages.values())
            })
            
            # Sort by average (descending)
            avg_df = avg_df.sort_values('Average % within ±0.5%', ascending=False)
            
            # Add rank column
            avg_df.insert(0, 'Rank', range(1, len(avg_df) + 1))
            
            # Display as table
            st.dataframe(avg_df, use_container_width=True)
            
            # Create bar chart
            fig = px.bar(
                avg_df,
                x='Pair',
                y='Average % within ±0.5%',
                title=f"Daily Average Stability Ranking (SURF)",
                labels={
                    'Pair': 'Cryptocurrency Pair',
                    'Average % within ±0.5%': '% within ±0.5% of median'
                },
                color='Average % within ±0.5%',
                color_continuous_scale='Viridis'
            )
            
            # Update layout
            fig.update_layout(
                height=500,
                xaxis_title="Cryptocurrency Pair",
                yaxis_title="Daily Average % within ±0.5%",
                yaxis=dict(
                    range=[0, max(avg_df['Average % within ±0.5%']) * 1.1]  # Set y-axis range with some padding
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add download button for the rankings
            csv = avg_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Rankings",
                data=csv,
                file_name=f"price_stability_rankings_surf.csv",
                mime="text/csv"
            )
        else:
            st.warning("No daily averages calculated.")

# Add explanation in the sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("About This Analysis")
st.sidebar.markdown(f"""
This dashboard analyzes cryptocurrency price stability by:

1. Dividing price data into {INTERVAL_MINUTES}-minute intervals
2. Calculating the median price for each interval
3. Measuring what percentage of prices fall within ±{TOLERANCE_PERCENTAGE}% of the median
4. Plotting these percentages over time
5. Calculating daily averages and ranking the coins by stability

Higher percentages indicate more stable prices within the tolerance range.
""")

# Add footer
st.sidebar.markdown("---")
st.sidebar.markdown(f"*Last updated: {now_sg.strftime('%Y-%m-%d %H:%M:%S')} (SGT)*")