import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
import time

# Clear cache at startup to ensure fresh data
st.cache_data.clear()

# Page configuration - absolute minimum for speed
st.set_page_config(
    page_title="Aggregated Depth Tier Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"  # Start with sidebar collapsed for speed
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

    /* Make tabs much bigger and more prominent */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
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

    /* Improved table styling */
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

    /* Highlight top tier */
    .dataframe tr:first-child {
        background-color: #e6f7ff !important;
    }
    
    /* Highlight % Time Highest Choppiness cells based on value */
    .highlight-cell-high {
        background-color: #d4f7d4 !important;  /* Light green */
    }
    .highlight-cell-medium {
        background-color: #ffffd4 !important;  /* Light yellow */
    }
    .highlight-cell-low {
        background-color: #ffd4d4 !important;  /* Light red */
    }
</style>
""", unsafe_allow_html=True)

# Database configuration
DB_CONFIG = {
    'main': {
        'url': "postgresql://public_rw:aTJ92^kl04hllk@aws-jp-tk-surf-pg-public.cluster-csteuf9lw8dv.ap-northeast-1.rds.amazonaws.com:5432/report_dev"
    },
    'replication': {
        'url': "postgresql://public_replication:866^FKC4hllk@aws-jp-tk-surf-pg-public.cluster-csteuf9lw8dv.ap-northeast-1.rds.amazonaws.com:5432/replication_report"
    }
}

# Create database engine
@st.cache_resource
def get_engine(use_replication=True):
    """Create database engine
    Args:
        use_replication (bool): Whether to use replication database connection, default is True (use replication database)
    Returns:
        engine: SQLAlchemy engine
    """
    try:
        config = DB_CONFIG['replication' if use_replication else 'main']
        return create_engine(config['url'], pool_size=5, max_overflow=10)
    except Exception as e:
        st.error(f"Error creating database engine: {e}")
        return None

@contextmanager
def get_session(use_replication=True):
    """Database session context manager
    Args:
        use_replication (bool): Whether to use replication database connection, default is True (use replication database)
    Yields:
        session: SQLAlchemy session
    """
    engine = get_engine(use_replication)
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

# Pre-defined pairs as a fast fallback
PREDEFINED_PAIRS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "AVAX/USDT", "DOGE/USDT", "ADA/USDT", "TRX/USDT", "DOT/USDT"
]

# Get available pairs from the replication database
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_available_pairs():
    try:
        with get_session() as session:
            if not session:
                return PREDEFINED_PAIRS  # Fallback to predefined pairs

            query = text("SELECT pair_name FROM trade_pool_pairs WHERE status = 1")
            result = session.execute(query)
            pairs = [row[0] for row in result]

            # Return sorted pairs or fallback to predefined list if empty
            return sorted(pairs) if pairs else PREDEFINED_PAIRS

    except Exception as e:
        st.error(f"Error fetching available pairs: {e}")
        return PREDEFINED_PAIRS  # Fallback to predefined pairs

# Get current bid/ask data
def get_current_bid_ask(pair_name, use_replication=True):
    try:
        with get_session(use_replication=use_replication) as session:
            if not session:
                return None

            # Get the most recent partition table
            today = datetime.now().strftime("%Y%m%d")
            table_name = f'oracle_order_book_level_price_data_partition_v5_{today}'

            # Check if table exists
            check_table = text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = :table_name
                );
            """)

            if not session.execute(check_table, {"table_name": table_name}).scalar():
                # Try yesterday if today doesn't exist
                yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
                table_name = f'oracle_order_book_level_price_data_partition_v5_{yesterday}'

                # Check if yesterday's table exists
                if not session.execute(check_table, {"table_name": table_name}).scalar():
                    return None

            # Use the exact SQL you provided
            query = text(f"""
            SELECT 
                pair_name,
                TO_CHAR(created_at + INTERVAL '8 hour', 'YYYY-MM-DD HH24:MI:SS.MS') AS "UTC+8",
                all_bid,
                all_ask
            FROM 
                public."{table_name}"
            WHERE 
                pair_name = :pair_name
            ORDER BY 
                created_at DESC
            LIMIT 1
            """)

            result = session.execute(query, {"pair_name": pair_name}).fetchone()

            if result:
                return {
                    "pair": result[0],
                    "time": result[1],
                    "all_bid": result[2],
                    "all_ask": result[3]
                }
            return None

    except Exception as e:
        st.error(f"Error getting bid/ask data: {e}")
        return None

# Enhanced version of the depth tier analyzer with time-based choppiness tracking
class EnhancedDepthTierAnalyzer:
    """
    Enhanced analyzer for liquidity depth tiers with time-based choppiness tracking
    """

    def __init__(self):
        # Update point counts to match the other file
        self.point_counts = [500, 1500, 2500, 5000]

        # Initialize analysis time range
        self.analysis_time_range = None

        # Initialize time ranges for each point count
        self.point_time_ranges = {point: None for point in self.point_counts}

        # Define depth tiers

        # Define depth tiers
        self.depth_tier_columns = [
            'price_1', 'price_2', 'price_3', 'price_4', 'price_5',
            'price_6', 'price_7', 'price_8', 'price_9', 'price_10',
            'price_11', 'price_12', 'price_13', 'price_14', 'price_15',
            'price_16','price_17','price_18','price_19','price_20','price_21',
            'price_22','price_23','price_24','price_25','price_26','price_27','price_28','price_29','price_30'
        ]

        # Map column names to actual depth values
        self.depth_tier_values = {
            'price_1':'1k',
            'price_2':'3k',
            'price_3':'5k',
            'price_4':'7k',
            'price_5': '10k',
            'price_6': '50k',
            'price_7': '100k',
            'price_8': '200k',
            'price_9': '300k',
            'price_10': '400k',
            'price_11': '500k',
            'price_12': '600k',
            'price_13': '700k',
            'price_14': '800k',
            'price_15': '900k',
            'price_16': '1M',
            'price_17': '1.5M',
            'price_18': '2M',
            'price_19': '3M',
            'price_20': '4M',
            'price_21':'5M',
            'price_22':'6M',
            'price_23':'7M',
            'price_24': '8M',
            'price_25':'9M',
            'price_26':'10M',
            'price_27':'11M',
            'price_28':'12M',
            'price_29':'13M',
            'price_30':'14M',
        }

        # Metrics to calculate
        self.metrics = [
            'direction_changes',   # Frequency of price direction reversals (%)
            'choppiness',          # Measures price oscillation within a range
            'tick_atr_pct',        # ATR %
            'trend_strength'       # Measures directional strength
        ]

        # Display names for metrics
        self.metric_display_names = {
            'direction_changes': 'Direction Changes (%)',
            'choppiness': 'Choppiness',
            'tick_atr_pct': 'Tick ATR %',
            'trend_strength': 'Trend Strength'
        }

        # Store results
        self.results = {point: None for point in self.point_counts}
        
        # New: Store historical choppiness data for time-based analysis
        self.historical_choppiness = {}
        
        # New: Store time highest percentages
        self.time_highest_choppiness = {point: {} for point in self.point_counts}
        
        # New: Store validity rate data
        self.validity_rates = {point: {} for point in self.point_counts}

    def fetch_and_analyze(self, pair_name, hours=24, progress_bar=None, use_replication=True, time_intervals=12):
        """Fetch data and calculate metrics for each depth tier with time-based analysis
        
        Args:
            pair_name: Cryptocurrency pair to analyze
            hours: Hours of data to look back (default 24 hours)
            progress_bar: Optional progress bar to update
            use_replication: Whether to use replication database
            time_intervals: Number of time intervals to split data into for historical analysis
        """
        try:
            with get_session(use_replication=use_replication) as session:
                if not session:
                    return False

                # Calculate time range in Singapore time
                singapore_tz = pytz.timezone('Asia/Singapore')
                now = datetime.now(singapore_tz)
                start_time = now - timedelta(hours=hours)

                # Format for display
                start_str_display = start_time.strftime("%Y-%m-%d %H:%M:%S")
                end_str_display = now.strftime("%Y-%m-%d %H:%M:%S")

                # Store the time range for display later
                self.analysis_time_range = {
                    'start': start_str_display,
                    'end': end_str_display,
                    'timezone': 'SGT'
                }

                # Format for database query (without timezone)
                start_str = start_time.strftime("%Y-%m-%d %H:%M:%S")

                # Display the exact time range being analyzed
                if progress_bar:
                    progress_bar.progress(0.05, text=f"Analyzing data from {start_str_display} to {end_str_display} (SGT)")

                # Get data tables covering the time range
                table_dates = []
                current_date = now.date()
                
                # Add yesterday and today's tables if within the time range
                yesterday = current_date - timedelta(days=1)
                
                # Always check today's table first
                table_dates.append(current_date.strftime("%Y%m%d"))
                
                # Check if we need yesterday's data too
                if start_time.date() <= yesterday:
                    table_dates.append(yesterday.strftime("%Y%m%d"))
                
                # Check if we need even earlier data
                if hours > 48:
                    for i in range(2, min(7, hours//24 + 2)):  # Limit to a week of lookback
                        past_date = current_date - timedelta(days=i)
                        if start_time.date() <= past_date:
                            table_dates.append(past_date.strftime("%Y%m%d"))

                if progress_bar:
                    progress_bar.progress(0.1, text=f"Searching data across {len(table_dates)} tables")

                # Collect data from all tables
                all_data = []
                
                for table_date in table_dates:
                    table_name = f"oracle_order_book_level_price_data_partition_v5_{table_date}"
                    
                    # Check if table exists
                    check_table = text("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_schema = 'public' 
                            AND table_name = :table_name
                        );
                    """)
                    
                    if not session.execute(check_table, {"table_name": table_name}).scalar():
                        continue  # Skip if table doesn't exist
                        
                    if progress_bar:
                        progress_bar.progress(0.1, text=f"Fetching data from {table_name}...")
                        
                    # Query to fetch data from this table
                    query = text(f"""
                        SELECT
                            pair_name,
                            created_at,
                            TO_CHAR(created_at + INTERVAL '8 hour', 'YYYY-MM-DD HH24:MI:SS.MS') AS timestamp_sgt,
                            {', '.join(self.depth_tier_columns)}
                        FROM
                            public."{table_name}"
                        WHERE
                            pair_name = :pair_name
                            AND created_at >= :start_time
                        ORDER BY created_at DESC
                    """)
                    
                    # Execute query with parameters
                    result = session.execute(
                        query,
                        {
                            "pair_name": pair_name,
                            "start_time": start_str
                        }
                    )
                    
                    # Add to our collected data
                    table_data = result.fetchall()
                    if table_data:
                        all_data.extend(table_data)
                        if progress_bar:
                            progress_bar.progress(0.1, text=f"Found {len(table_data)} rows in {table_name}")
                
                # Sort all collected data by timestamp
                all_data.sort(key=lambda x: x[1], reverse=True)  # Sort by created_at (index 1)
                
                if not all_data:
                    if progress_bar:
                        progress_bar.progress(0.2, text="No data found for the specified pair")
                    return False

                if len(all_data) < min(self.point_counts):
                    if progress_bar:
                        progress_bar.progress(0.2, text=f"Insufficient data: found {len(all_data)} rows, need at least {min(self.point_counts)}")
                    return False

                if progress_bar:
                    progress_bar.progress(0.3, text=f"Processing {len(all_data)} data points...")

                # Convert to DataFrame for faster processing
                columns = ['pair_name', 'created_at', 'timestamp_sgt'] + self.depth_tier_columns
                all_df = pd.DataFrame(all_data, columns=columns)
                
                # Process each point count using the pre-fetched data
                for i, point_count in enumerate(self.point_counts):
                    if progress_bar:
                        progress_bar.progress((i / len(self.point_counts)) * 0.6 + 0.3,
                                              text=f"Processing {point_count} points...")

                    if len(all_df) >= point_count:
                        # Get and store the time range for this specific point count
                        point_df = all_df.iloc[:point_count].copy()

                        if 'timestamp_sgt' in point_df.columns:
                            newest_time = point_df['timestamp_sgt'].iloc[0]
                            oldest_time = point_df['timestamp_sgt'].iloc[-1]

                            self.point_time_ranges[point_count] = {
                                'newest': newest_time,
                                'oldest': oldest_time,
                                'count': len(point_df)
                            }
                        
                        # Calculate validity rates for each depth tier
                        tier_validity_rates = {}
                        for column in self.depth_tier_columns:
                            if column in point_df.columns:
                                # Convert to numeric and count valid entries (non-zero values)
                                numeric_values = pd.to_numeric(point_df[column], errors='coerce')
                                valid_count = (numeric_values > 0).sum()
                                total_count = len(numeric_values)
                                validity_rate = (valid_count / total_count) * 100 if total_count > 0 else 0
                                
                                # Store the validity rate
                                tier = self.depth_tier_values[column]
                                tier_validity_rates[tier] = validity_rate
                                
                        # Store the validity rates for this point count
                        self.validity_rates[point_count] = tier_validity_rates
                        
                        # Only perform time-based choppiness analysis for 5000 points to improve speed
                        if point_count == 5000:
                            # New: Historical analysis by time intervals
                            # Divide data into time intervals for historical analysis
                            interval_size = min(len(all_df) // time_intervals, point_count)
                            
                            if interval_size < 50:  # Ensure meaningful sample size
                                time_intervals = max(1, len(all_df) // 50)
                                interval_size = min(len(all_df) // time_intervals, point_count)
                            
                            # Initialize storage for this point count's historical analysis
                            self.historical_choppiness[point_count] = {
                                'intervals': [],
                                'tier_data': {}
                            }
                            
                            # Track which tier had highest choppiness in each interval
                            highest_choppiness_counts = {}
                            
                            # Process each time interval
                            for j in range(time_intervals):
                                start_idx = j * interval_size
                                end_idx = min((j + 1) * interval_size, len(all_df))
                                
                                if end_idx - start_idx < interval_size * 0.5:  # Skip if too small
                                    continue
                                    
                                interval_df = all_df.iloc[start_idx:end_idx].copy()
                                
                                # Get interval time range for reference
                                interval_range = {
                                    'start': interval_df['timestamp_sgt'].iloc[-1],
                                    'end': interval_df['timestamp_sgt'].iloc[0]
                                }
                                
                                self.historical_choppiness[point_count]['intervals'].append(interval_range)
                                
                                # Calculate choppiness for each tier in this interval
                                interval_results = {}
                                highest_choppiness = 0
                                highest_tier = None
                                
                                for column in self.depth_tier_columns:
                                    # Extract price data for this tier
                                    if column in interval_df.columns:
                                        # Make a clean copy of the data for this specific tier
                                        df_tier = interval_df[['pair_name', column]].copy()
                                        
                                        # Calculate choppiness metric
                                        metrics = self._calculate_metrics(df_tier, column, end_idx - start_idx)
                                        
                                        if metrics and 'choppiness' in metrics:
                                            tier = self.depth_tier_values[column]
                                            
                                            # Store the choppiness value
                                            if tier not in self.historical_choppiness[point_count]['tier_data']:
                                                self.historical_choppiness[point_count]['tier_data'][tier] = []
                                                
                                            self.historical_choppiness[point_count]['tier_data'][tier].append(metrics['choppiness'])
                                            
                                            # Track which tier had highest choppiness in this interval
                                            if metrics['choppiness'] > highest_choppiness:
                                                highest_choppiness = metrics['choppiness']
                                                highest_tier = tier
                                
                                # Increment count for the tier with highest choppiness
                                if highest_tier:
                                    if highest_tier not in highest_choppiness_counts:
                                        highest_choppiness_counts[highest_tier] = 0
                                    highest_choppiness_counts[highest_tier] += 1
                            
                            # Calculate percentage of time each tier had highest choppiness
                            total_intervals = len(self.historical_choppiness[point_count]['intervals'])
                            
                            if total_intervals > 0:
                                for tier, count in highest_choppiness_counts.items():
                                    percentage = (count / total_intervals) * 100
                                    self.time_highest_choppiness[point_count][tier] = percentage

                        # Process each depth tier for the main results
                        tier_results = {}

                        for column in self.depth_tier_columns:
                            # Extract price data for this tier
                            if column in all_df.columns:
                                # Make a clean copy of the data for this specific tier
                                df_tier = all_df[['pair_name', column]].copy()

                                # Calculate metrics using the correct method
                                metrics = self._calculate_metrics(df_tier, column, point_count)
                                if metrics:
                                    tier = self.depth_tier_values[column]
                                    tier_results[tier] = metrics
                                    
                                    # Add validity rate
                                    if tier in self.validity_rates[point_count]:
                                        tier_results[tier]['validity_rate'] = self.validity_rates[point_count][tier]
                                    
                                    # Add time highest percentage only for 5000 points
                                    if point_count == 5000 and tier in self.time_highest_choppiness[point_count]:
                                        tier_results[tier]['time_highest_choppiness'] = self.time_highest_choppiness[point_count][tier]
                                    
                                    # We're no longer calculating winrate

                        # Convert to DataFrame and sort appropriately
                        if point_count == 5000:
                            # For 5000 points, sort by time_highest_choppiness
                            self.results[point_count] = self._create_results_table(tier_results, sort_by='time_highest_choppiness')
                        else:
                            # For other points, sort by choppiness
                            self.results[point_count] = self._create_results_table(tier_results, sort_by='choppiness')

                if progress_bar:
                    progress_bar.progress(1.0, text="Analysis complete!")

                # Check if we got any results
                has_results = False
                for pc in self.point_counts:
                    if self.results[pc] is not None:
                        has_results = True
                        break

                return has_results

        except Exception as e:
            if progress_bar:
                progress_bar.progress(1.0, text=f"Error: {str(e)}")
            st.error(f"Error in analysis: {e}")
            import traceback
            st.error(traceback.format_exc())
            return False

    def _calculate_metrics(self, df, price_col, point_count):
        """Calculate raw metrics without any normalization or scoring"""
        try:
            # Convert to numeric and drop any NaN values
            prices = pd.to_numeric(df[price_col], errors='coerce').dropna()
            prices = prices[prices>0]
            if len(prices) < point_count * 0.8:  # Allow some flexibility for missing data
                return None

            # Take only the needed number of points
            prices = prices.iloc[:point_count].copy()

            # Calculate mean price for ATR percentage calculation
            mean_price = prices.mean()

            # Direction changes
            price_changes = prices.diff().dropna()
            signs = np.sign(price_changes)
            direction_changes = (signs.shift(1) != signs).sum()
            direction_change_pct = (direction_changes / (len(signs) - 1)) * 100 if len(signs) > 1 else 0

            # Choppiness
            window = min(20, point_count // 10)
            diff = prices.diff().abs()
            sum_abs_changes = diff.rolling(window, min_periods=1).sum()
            price_range = prices.rolling(window, min_periods=1).max() - prices.rolling(window, min_periods=1).min()

            # Avoid division by zero
            epsilon = 1e-10
            choppiness_values = 100 * sum_abs_changes / (price_range + epsilon)

            # Cap extreme values
            choppiness_values = np.minimum(choppiness_values, 1000)

            # Calculate mean choppiness
            choppiness = choppiness_values.mean()

            # Tick ATR
            tick_atr = price_changes.abs().mean()
            tick_atr_pct = (tick_atr / mean_price) * 100

            # Trend strength
            net_change = (prices - prices.shift(window)).abs()
            trend_strength = (net_change / (sum_abs_changes + epsilon)).dropna().mean()

            return {
                'direction_changes': direction_change_pct,
                'choppiness': choppiness,
                'tick_atr_pct': tick_atr_pct,
                'trend_strength': trend_strength
            }

        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return None
            
    # The winrate calculation method has been removed

    def _create_results_table(self, tier_results, sort_by='choppiness'):
        """Create a results table including time highest percentages
        
        Args:
            tier_results: Dictionary of tier results
            sort_by: Column to sort by (default: choppiness)
        """
        if not tier_results:
            return None

        # Create DataFrame directly
        data = []
        for tier, metrics in tier_results.items():
            row = {'Tier': tier}
            row.update(metrics)
            data.append(row)

        df = pd.DataFrame(data)

        # Sort by specified column if it exists
        if sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=False)
        elif 'choppiness' in df.columns:
            # Fallback to choppiness
            df = df.sort_values('choppiness', ascending=False)
        else:
            # If choppiness is not available, try another metric
            for metric in ['direction_changes', 'tick_atr_pct']:
                if metric in df.columns:
                    df = df.sort_values(metric, ascending=False)
                    break

        return df

# Enhanced table display function with time highest percentage and validity rate
def create_point_count_table(analyzer, point_count):
    """Creates a clean, readable table with time highest choppiness percentages and validity rates"""
    if analyzer.results[point_count] is None:
        st.info(f"No data available for {point_count} points analysis.")
        return

    # Display specific point count time range if available
    if hasattr(analyzer, 'point_time_ranges') and analyzer.point_time_ranges.get(point_count):
        point_range = analyzer.point_time_ranges[point_count]
        st.markdown(f"""
        <div style="background-color: #f0f7ff; padding: 8px; border-radius: 5px; margin-bottom: 15px;">
            <h4 style="margin: 0 0 5px 0;">Specific Time Range for {point_count} Points:</h4>
            <p style="margin: 3px 0;"><strong>Newest data point:</strong> {point_range['newest']} (SGT)</p>
            <p style="margin: 3px 0;"><strong>Oldest data point:</strong> {point_range['oldest']} (SGT)</p>
            <p style="margin: 3px 0;"><strong>Total data points:</strong> {point_range['count']}</p>
        </div>
        """, unsafe_allow_html=True)
        
    # Check how many time intervals were analyzed
    interval_count = 0
    if hasattr(analyzer, 'historical_choppiness') and point_count in analyzer.historical_choppiness:
        interval_count = len(analyzer.historical_choppiness[point_count]['intervals'])
        if interval_count > 0:
            st.markdown(f"**Time Analysis:** Analyzed {interval_count} time intervals across the period")

    df = analyzer.results[point_count]

    # Make a clean copy for display
    display_df = df.copy()

    # Select only the columns we want to display
    display_columns = ['Tier', 'validity_rate', 'time_highest_choppiness', 'choppiness', 'direction_changes', 'tick_atr_pct', 'trend_strength']
    
    # Filter to columns that exist in the dataframe
    available_columns = [col for col in display_columns if col in display_df.columns]
    display_df = display_df[available_columns]

    # Rename columns for better display
    column_renames = {
        'validity_rate': 'Validity Rate (%)',
        'time_highest_choppiness': '% Time Highest Choppiness',
        'direction_changes': 'Direction Changes (%)',
        'choppiness': 'Choppiness',
        'tick_atr_pct': 'Tick ATR %',
        'trend_strength': 'Trend Strength'
    }
    
    # Only rename columns that exist
    renames = {col: column_renames[col] for col in column_renames if col in display_df.columns}
    display_df = display_df.rename(columns=renames)

    # Format numeric columns with appropriate decimal places
    for col in display_df.columns:
        if col != 'Tier':
            if col in ['Validity Rate (%)', '% Time Highest Choppiness']:
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.1f}%" if not pd.isna(x) else "0.0%"
                )
            elif col == 'Tick ATR %':
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A"
                )
            elif col == 'Trend Strength':
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A"
                )
            else:
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.1f}" if not pd.isna(x) else "N/A"
                )

    # Display the top tier as recommendation
    top_tier = display_df.iloc[0]['Tier']
    st.markdown(f"### Recommended Depth Tier: **{top_tier}**")
    
    # Add explanatory text
    explanation_text = '<div style="background-color: #f7f7f7; padding: 10px; border-radius: 5px; margin-bottom: 15px;">'
    
    # Add validity rate explanation
    explanation_text += '<p style="margin: 5px 0;"><strong>Validity Rate (%)</strong>: Percentage of non-zero price feed values for each tier in the analyzed data</p>'
    
    if '% Time Highest Choppiness' in display_df.columns:
        explanation_text += '<p style="margin: 5px 0;"><strong>% Time Highest Choppiness</strong>: Percentage of time intervals where this tier had the highest choppiness value compared to other tiers</p>'
            
    explanation_text += '</div>'
    st.markdown(explanation_text, unsafe_allow_html=True)

    # Show the full table with enhanced styling
    st.dataframe(
        display_df,
        use_container_width=True,
        height=min(800, 100 + (len(display_df) * 35))  # Adaptive height
    )

# Format number with commas (e.g., 1,234,567)
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

def main():
    # Get current Singapore time
    singapore_tz = pytz.timezone('Asia/Singapore')
    now_sg = datetime.now(singapore_tz)
    current_time_sg = now_sg.strftime("%Y-%m-%d %H:%M:%S")

    # Main layout - super streamlined
    st.markdown("<h1 style='text-align: center; font-size:28px; margin-bottom: 10px;'>Enhanced Liquidity Depth Tier Analyzer</h1>", unsafe_allow_html=True)

    # Display current Singapore time to confirm updates
    st.markdown(f"<p style='text-align: center; font-size:14px; color:gray;'>Last updated: {current_time_sg} (SGT)</p>", unsafe_allow_html=True)

    # Get available pairs from the database
    available_pairs = get_available_pairs()

    # Main selection area - simplified with fixed 24 hours
    col1, col2 = st.columns([2, 1])

    with col1:
        selected_pair = st.selectbox(
            "Select Pair",
            available_pairs,
            index=0 if available_pairs else None
        )

    with col2:
        run_analysis = st.button("ANALYZE", use_container_width=True)
        
    # Fixed analysis at 24 hours
    selected_hours = 24

    # Main content
    if run_analysis and selected_pair:
        # Clear cache before analysis to ensure fresh data
        st.cache_data.clear()

        # Show analysis time in Singapore timezone
        analysis_start_time = datetime.now(singapore_tz).strftime("%Y-%m-%d %H:%M:%S")
        st.markdown(f"<p style='text-align: center; font-size:14px; color:green;'>Analysis started at: {analysis_start_time} (SGT) for 24 hours of data</p>", unsafe_allow_html=True)

        # Get current bid/ask data
        bid_ask_data = get_current_bid_ask(selected_pair)

        if bid_ask_data:
            # Display in a box at the top
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 20px;">
                <h3 style="margin: 0;">Current Market Data: {selected_pair}</h3>
                <p style="margin: 5px 0;"><strong>UTC+8:</strong> {bid_ask_data['time']}</p>
                <p style="margin: 5px 0;"><strong>Total Bid:</strong> {format_number(bid_ask_data['all_bid'])}</p>
                <p style="margin: 5px 0;"><strong>Total Ask:</strong> {format_number(bid_ask_data['all_ask'])}</p>
            </div>
            """, unsafe_allow_html=True)

        # Simple explanation of metrics (much shorter)
        st.markdown(f"""
        **Analysis Period:** Looking back 24 hours
        
        **Key Metrics:**
        - **Validity Rate (%):** Percentage of non-zero price feed values in each tier
        - **% Time Highest Choppiness:** Percentage of time this tier had the highest choppiness
        - **Choppiness:** Oscillation intensity within price range
        - **Direction Changes (%):** Frequency of price direction reversals
        - **Tick ATR %:** Average tick-to-tick price change percentage
        - **Trend Strength:** Lower values indicate more choppy/mean-reverting behavior
        """)

        # Set up tabs for results - updated to match point counts
        tabs = st.tabs(["500 POINTS", "1,500 POINTS", "2,500 POINTS", "5,000 POINTS"])

        # Create progress bar
        progress_bar = st.progress(0, text="Starting analysis...")

        # Initialize analyzer and run analysis
        analyzer = EnhancedDepthTierAnalyzer()
        
        # Calculate appropriate time intervals based on hours
        time_intervals = max(12, selected_hours * 2)  # More intervals for longer periods
        
        success = analyzer.fetch_and_analyze(
            selected_pair, 
            hours=selected_hours, 
            progress_bar=progress_bar,
            time_intervals=time_intervals
        )

        if success:
            # Display the time range used for analysis
            if hasattr(analyzer, 'analysis_time_range') and analyzer.analysis_time_range:
                st.markdown(f"""
                <div style="background-color: #e6f3ff; padding: 10px; border-radius: 5px; margin-bottom: 20px;">
                    <h3 style="margin: 0;">Overall Data Time Range</h3>
                    <p style="margin: 5px 0;"><strong>From:</strong> {analyzer.analysis_time_range['start']} ({analyzer.analysis_time_range['timezone']})</p>
                    <p style="margin: 5px 0;"><strong>To:</strong> {analyzer.analysis_time_range['end']} ({analyzer.analysis_time_range['timezone']})</p>
                </div>
                """, unsafe_allow_html=True)

            # Display results for each point count (updated point counts)
            with tabs[0]:
                create_point_count_table(analyzer, 500)

            with tabs[1]:
                create_point_count_table(analyzer, 1500)

            with tabs[2]:
                create_point_count_table(analyzer, 2500)

            with tabs[3]:
                create_point_count_table(analyzer, 5000)

            # Show analysis completion time in Singapore timezone
            analysis_end_time = datetime.now(singapore_tz).strftime("%Y-%m-%d %H:%M:%S")
            st.markdown(f"<p style='text-align: center; font-size:14px; color:green;'>Analysis completed at: {analysis_end_time} (SGT)</p>", unsafe_allow_html=True)

        else:
            progress_bar.empty()
            st.error(f"Failed to analyze {selected_pair}. Please try another pair or time range.")

    else:
        # Minimal welcome message
        st.info("Select a pair and time range, then click ANALYZE to find the optimal depth tier.")

if __name__ == "__main__":
    main()