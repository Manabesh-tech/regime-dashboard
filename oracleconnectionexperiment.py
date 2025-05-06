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

# Page configuration
st.set_page_config(
    page_title="Enhanced Global Tier Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced CSS for better visual presentation
st.markdown("""
<style>
    .block-container {padding: 0 !important;}
    .main .block-container {max-width: 98% !important;}
    h1, h2, h3 {margin: 0 !important; padding: 0 !important;}
    .stButton > button {width: 100%; font-weight: bold; height: 46px; font-size: 18px;}
    div.stProgress > div > div {height: 5px !important;}
    
    /* Better table styling */
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
    .dataframe tr:first-child {
        background-color: #e6f7ff !important;
    }
</style>
""", unsafe_allow_html=True)

# Create database engine
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
    "BTC", "ETH", "SOL", "BNB", "XRP",
    "AVAX", "DOGE", "ADA", "TRX", "DOT",
    "LINK", "MATIC", "SUI", "PNUT", "TRUMP"
]

# Get available pairs from the database
def get_available_pairs():
    """Fallback to predefined pairs instead of querying the database"""
    # Due to database connection issues, we'll use the predefined list
    return PREDEFINED_PAIRS

def _calculate_trend_strength(prices, window):
    """Calculate average Trend Strength - measures the directional strength of price movements.
    
    Trend Strength is calculated as the ratio of net price change to the sum of all price movements.
    A higher value indicates stronger trend (closer to 1), lower values indicate choppy price action.
    """
    try:
        # Calculate absolute tick-to-tick changes
        diff = prices.diff().abs()
        
        # Get sum of all absolute changes over the window
        sum_abs_changes = diff.rolling(window=window, min_periods=1).sum()
        
        # Calculate the absolute net change over the window
        net_change = prices.diff(periods=window).abs()
        
        # Avoid division by zero
        epsilon = 1e-10
        
        # Calculate trend strength: net change / sum of all changes
        # This ranges from 0 to 1 (choppy to trending)
        trend_strength = net_change / (sum_abs_changes + epsilon)
        
        # Handle NaN values and ensure reasonable bounds
        trend_strength = trend_strength.fillna(0.5)
        
        return trend_strength.mean()
    except Exception as e:
        # Return a reasonable default on error
        return 0.5

def analyze_tiers(pair_name, progress_bar=None):
    """Analyze all exchange-tier combinations with time-based windows"""
    # Setup metadata dictionary to return
    metadata = {
        'time_span_hours': 0,
        'time_span_seconds': 0,
        'total_data_points': 0,
        'avg_interval_ms': 0,
        'exchange_counts': {},
        'theoretical_max': 0
    }
    
    try:
        with get_session() as session:
            if not session:
                return None, metadata

            # Define tier columns and mapping
            tier_columns = [
                'price_1', 'price_2', 'price_3', 'price_4', 'price_5',
                'price_6', 'price_7', 'price_8', 'price_9', 'price_10',
                'price_11', 'price_12', 'price_13', 'price_14', 'price_15'
            ]
            
            tier_values = { 
                'price_1': '100k', 
                'price_2': '300k', 
                'price_3': '500k', 
                'price_4': '1M', 
                'price_5': '1.5M', 
                'price_6': '2M', 
                'price_7': '2.5M', 
                'price_8': '3M', 
                'price_9': '3.5M', 
                'price_10': '4M', 
                'price_11': '4.5M', 
                'price_12': '5M', 
                'price_13': '5.5M', 
                'price_14': '6M', 
                'price_15': '6.5M'
            }
            
            # Join all tier columns for the query
            price_columns = ", ".join(tier_columns)
            
            # Get Singapore time for proper date handling
            singapore_tz = pytz.timezone('Asia/Singapore')
            now_sg = datetime.now(singapore_tz)
            
            # Explicitly set the exact 24-hour period we want to analyze
            end_time = now_sg.replace(tzinfo=None)
            start_time = end_time - timedelta(hours=24)
            
            if progress_bar:
                progress_bar.progress(0.05, text=f"Analyzing data from {start_time} to {end_time} (24 hour period)")
            
            # Get current date in Singapore time
            current_date_sg = now_sg.strftime("%Y%m%d")
            
            # Add 3 days prior for data collection to ensure we get 24 hours coverage
            dates_to_check = []
            for i in range(4):  # Today, yesterday, day before, and day before that
                check_date = (now_sg - timedelta(days=i)).strftime("%Y%m%d")
                dates_to_check.append(check_date)
            
            if progress_bar:
                progress_bar.progress(0.1, text=f"Checking tables for dates: {', '.join(dates_to_check)}")
            
            # Check which tables exist and collect data from all of them
            all_data = []
            
            for date in dates_to_check:
                table_name = f"oracle_exchange_price_partition_v1_{date}"
                
                # Check if table exists
                table_exists_query = text(f"""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = :table_name
                    );
                """)
                
                if not session.execute(table_exists_query, {"table_name": table_name}).scalar():
                    if progress_bar:
                        progress_bar.progress(0.1, text=f"Table {table_name} does not exist, trying next date...")
                    continue
                
                # Get timestamp from 24 hours ago in UTC
                utc_24h_ago = now_sg.replace(tzinfo=None) - timedelta(hours=24)
                
                # Query with optimized filters and fewer columns to reduce data transfer
                if progress_bar:
                    progress_bar.progress(0.1, text=f"Fetching data from {table_name} (optimized)...")
                
                # Only fetch columns we need for better performance
                query = text(f"""
                    WITH latest_data AS (
                        SELECT 
                            source as exchange_name,
                            created_at,
                            {price_columns}
                        FROM 
                            {table_name}
                        WHERE 
                            pair_name = :pair_name
                            AND created_at >= :start_time
                            AND created_at <= :end_time
                            AND source IN ('binanceFuture', 'bitgetFuture', 'okxFuture', 'bybitFuture', 
                                          'gateFuture', 'mexcFuture', 'hyperliquidFuture')
                        ORDER BY 
                            created_at DESC
                        LIMIT 300000
                    )
                    SELECT * FROM latest_data
                    ORDER BY 
                        exchange_name,
                        created_at DESC
                """)
                
                # Only fetch the minimum data needed for analysis
                result = session.execute(query, {"pair_name": pair_name, "start_time": start_time, "end_time": end_time})
                table_data = result.fetchall()
                
                if table_data:
                    all_data.extend(table_data)
                    if progress_bar:
                        progress_bar.progress(0.2, text=f"Found {len(table_data)} rows in {table_name}")
                            
                    # Skip if we have enough data - reduced to 250,000 for faster processing
                    if len(all_data) >= 250000:  # Reduced from 600,000
                        break
            
            if not all_data:
                if progress_bar:
                    progress_bar.progress(1.0, text=f"No data found for {pair_name}")
                return None, metadata
            
            # Update metadata with data points count
            metadata['total_data_points'] = len(all_data)
            
            # Calculate time span info
            if len(all_data) > 0:
                newest_timestamp = max(row[1] for row in all_data)
                oldest_timestamp = min(row[1] for row in all_data)
                
                # Update time span in metadata
                metadata['time_span_seconds'] = (newest_timestamp - oldest_timestamp).total_seconds()
                metadata['time_span_hours'] = metadata['time_span_seconds'] / 3600
                metadata['avg_interval_ms'] = (metadata['time_span_seconds'] * 1000) / len(all_data)
                metadata['theoretical_max'] = int(metadata['time_span_seconds'] * 2)  # 2 ticks per second (500ms)
                
                # Count data points per exchange
                exchange_counts = {}
                for row in all_data:
                    exchange = row[0]  # exchange_name is the first column
                    if exchange not in exchange_counts:
                        exchange_counts[exchange] = 0
                    exchange_counts[exchange] += 1
                
                metadata['exchange_counts'] = exchange_counts
                
                # Find largest exchange by data count
                if exchange_counts:
                    max_exchange = max(exchange_counts.items(), key=lambda x: x[1])
                    
                    if progress_bar:
                        progress_bar.progress(0.3, 
                            text=f"Processing {len(all_data):,} data points spanning {metadata['time_span_hours']:.2f} hours " + 
                                 f"(avg interval: {metadata['avg_interval_ms']:.1f}ms)")
                        progress_bar.progress(0.31, 
                            text=f"Largest exchange: {max_exchange[0]} with {max_exchange[1]:,} data points")
            
            # Create DataFrame
            columns = ['exchange_name', 'created_at'] + tier_columns
            df = pd.DataFrame(all_data, columns=columns)
            
            # Convert numeric columns
            for col in tier_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Get unique exchanges
            exchanges = df['exchange_name'].unique()
            
            if progress_bar:
                progress_bar.progress(0.4, text=f"Found {len(exchanges)} exchanges")
            
            # Initialize trackers
            win_counts = {}  # To count how many times each tier wins
            exchange_tier_choppiness = {}  # Store the overall choppiness average
            exchange_tier_trend_strength = {}  # Store the overall trend strength average
            exchange_tier_chop_trend_ratio = {}  # Store the choppiness/trend strength ratio
            exchange_tier_dropout = {}  # Store the overall dropout rate
            exchange_tier_valid_points = {}  # Store the count of valid data points
            exchange_tier_validity = {}  # Store the validity rate
            exchange_choppiness_by_range = {}  # Store choppiness by range group
            exchange_trend_strength_by_range = {}  # Store trend strength by range group
            
            # Define range groups for time-based windows
            range_groups = list(range(1, 11))  # 1 to 10 groups
            
            # Process each exchange separately
            for exchange in exchanges:
                # Filter data for this exchange
                exchange_df = df[df['exchange_name'] == exchange].copy()
                
                # Sort by timestamp to ensure proper order
                exchange_df.sort_values('created_at', ascending=False, inplace=True)
                
                # Calculate number of windows for this exchange
                window_size = 5000
                total_points = len(exchange_df)
                num_windows = total_points // window_size
                
                if num_windows == 0:
                    continue
                
                if progress_bar:
                    progress_bar.progress(0.5, text=f"Processing {num_windows} windows for {exchange}...")
                
                # Process each window for this exchange
                for window_idx in range(num_windows):
                    if progress_bar:
                        progress_bar.progress(0.5 + (0.4 * window_idx / num_windows), 
                                             text=f"Analyzing window {window_idx+1}/{num_windows} for {exchange}")
                    
                    # Calculate start and end indices for this window
                    start_idx = window_idx * window_size
                    end_idx = (window_idx + 1) * window_size
                    
                    # Get the data for this window
                    window_df = exchange_df.iloc[start_idx:end_idx].copy()
                    
                    # Assign a range group to this window based on its time
                    range_group = window_idx % len(range_groups) + 1
                    
                    # Skip if window is too small
                    if len(window_df) < 0.8 * window_size:
                        continue
                    
                    # Track best choppiness for this window
                    best_choppiness = 0
                    best_tier = None
                    
                    # Track all choppiness values for diagnostic
                    window_tier_choppiness = {}
                    window_tier_trend_strength = {}
                    window_tier_chop_trend_ratio = {}
                    
                    # Process each tier
                    for tier_col in tier_columns:
                        if tier_col not in window_df.columns:
                            continue
                            
                        tier_name = tier_values.get(tier_col, tier_col)
                        exchange_tier_key = f"{exchange}:{tier_name}"
                        
                        # Calculate dropout rate for this window
                        total_points_in_window = len(window_df)
                        nan_or_zero = (window_df[tier_col].isna() | (window_df[tier_col] <= 0)).sum()
                        valid_points = total_points_in_window - nan_or_zero
                        dropout_rate = (nan_or_zero / total_points_in_window) * 100
                        
                        # Initialize validity tracking if not already done
                        if exchange_tier_key not in exchange_tier_validity:
                            exchange_tier_validity[exchange_tier_key] = 0
                            
                        # Track overall dropout rate
                        if exchange_tier_key not in exchange_tier_dropout:
                            exchange_tier_dropout[exchange_tier_key] = []
                        exchange_tier_dropout[exchange_tier_key].append(dropout_rate)
                        
                        # Track valid points count
                        if exchange_tier_key not in exchange_tier_valid_points:
                            exchange_tier_valid_points[exchange_tier_key] = 0
                        exchange_tier_valid_points[exchange_tier_key] += valid_points
                        
                        # Calculate validity rate (valid ticks / actual exchange points)
                        exchange_total_points = metadata['exchange_counts'].get(exchange, 0)
                        if exchange_total_points > 0:
                            validity_rate = (exchange_tier_valid_points[exchange_tier_key] / exchange_total_points) * 100
                            exchange_tier_validity[exchange_tier_key] = validity_rate
                        
                        # Skip if dropout rate is too high
                        if dropout_rate > 90:
                            continue
                        
                        # Make sure we have enough consecutive valid prices
                        # Get valid prices
                        prices = window_df[tier_col].dropna()
                        prices = prices[prices > 0]
                        
                        # Skip if not enough data
                        if len(prices) < 0.8 * window_size:
                            continue
                        
                        # Calculate choppiness with 20-tick window - improved to match SQL
                        window_size_choppiness = 20
                        all_choppiness_values = []
                        
                        # Process each valid 20-tick window (similar to SQL approach)
                        for i in range(len(prices) - window_size_choppiness + 1):
                            window_prices = prices.iloc[i:i+window_size_choppiness]
                            
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
                        
                        # Calculate trend strength
                        trend_strength = _calculate_trend_strength(prices, window_size_choppiness)
                        
                        # Calculate choppiness/trend strength ratio
                        # Higher values indicate more choppiness relative to trend strength
                        if trend_strength > 0:
                            chop_trend_ratio = avg_choppiness / (trend_strength * 100)  # Scaling for easier reading
                        else:
                            chop_trend_ratio = avg_choppiness  # If trend_strength is 0, just use choppiness
                        
                        # Store for diagnostics
                        window_tier_choppiness[tier_name] = avg_choppiness
                        window_tier_trend_strength[tier_name] = trend_strength
                        window_tier_chop_trend_ratio[tier_name] = chop_trend_ratio
                        
                        # Track overall choppiness and trend strength
                        if exchange_tier_key not in exchange_tier_choppiness:
                            exchange_tier_choppiness[exchange_tier_key] = []
                        exchange_tier_choppiness[exchange_tier_key].append(avg_choppiness)
                        
                        if exchange_tier_key not in exchange_tier_trend_strength:
                            exchange_tier_trend_strength[exchange_tier_key] = []
                        exchange_tier_trend_strength[exchange_tier_key].append(trend_strength)
                        
                        if exchange_tier_key not in exchange_tier_chop_trend_ratio:
                            exchange_tier_chop_trend_ratio[exchange_tier_key] = []
                        exchange_tier_chop_trend_ratio[exchange_tier_key].append(chop_trend_ratio)
                        
                        # Check if this is the best choppiness for this window
                        if avg_choppiness > best_choppiness:
                            best_choppiness = avg_choppiness
                            best_tier = exchange_tier_key
                    
                    # Save choppiness and trend strength values for this range group by tier
                    for tier_col in tier_columns:
                        if tier_col not in window_df.columns:
                            continue
                            
                        tier_name = tier_values.get(tier_col, tier_col)
                        exchange_tier_key = f"{exchange}:{tier_name}"
                        
                        # Store choppiness for this range group
                        if exchange_tier_key not in exchange_choppiness_by_range:
                            exchange_choppiness_by_range[exchange_tier_key] = {}
                        
                        if window_tier_choppiness.get(tier_name) is not None:
                            exchange_choppiness_by_range[exchange_tier_key][range_group] = window_tier_choppiness[tier_name]
                            
                        # Store trend strength for this range group
                        if exchange_tier_key not in exchange_trend_strength_by_range:
                            exchange_trend_strength_by_range[exchange_tier_key] = {}
                        
                        if window_tier_trend_strength.get(tier_name) is not None:
                            exchange_trend_strength_by_range[exchange_tier_key][range_group] = window_tier_trend_strength[tier_name]
                    
                    # Record the winner for this window
                    if best_tier:
                        if best_tier not in win_counts:
                            win_counts[best_tier] = 0
                        win_counts[best_tier] += 1
                
                # After processing all windows for this exchange, update validity rates
                for tier_col in tier_columns:
                    tier_name = tier_values.get(tier_col, tier_col)
                    exchange_tier_key = f"{exchange}:{tier_name}"
                    
                    # Use the exchange's total points for validity calculation
                    exchange_total_points = metadata['exchange_counts'].get(exchange, 0)
                    valid_points = exchange_tier_valid_points.get(exchange_tier_key, 0)
                    
                    # Calculate validity rate
                    if exchange_total_points > 0:
                        validity_rate = (valid_points / exchange_total_points) * 100
                        exchange_tier_validity[exchange_tier_key] = validity_rate
            
            if progress_bar:
                progress_bar.progress(0.9, text="Calculating final rankings...")
            
            # Calculate final metrics
            total_wins = sum(win_counts.values())
            
            if total_wins == 0:
                if progress_bar:
                    progress_bar.progress(1.0, text="No valid winners found in any window")
                return None, metadata
            
            # Create average choppiness across all exchanges for each tier
            tier_to_avg_choppiness = {}
            tier_to_avg_trend_strength = {}
            tier_to_avg_chop_trend_ratio = {}
            
            # Calculate average choppiness per tier across all exchanges
            for exchange_tier_key, choppiness_values in exchange_tier_choppiness.items():
                _, tier = exchange_tier_key.split(':', 1)
                if tier not in tier_to_avg_choppiness:
                    tier_to_avg_choppiness[tier] = []
                tier_to_avg_choppiness[tier].extend(choppiness_values)
            
            # Calculate average trend strength per tier across all exchanges
            for exchange_tier_key, trend_strength_values in exchange_tier_trend_strength.items():
                _, tier = exchange_tier_key.split(':', 1)
                if tier not in tier_to_avg_trend_strength:
                    tier_to_avg_trend_strength[tier] = []
                tier_to_avg_trend_strength[tier].extend(trend_strength_values)
                
            # Calculate average chop/trend ratio per tier across all exchanges
            for exchange_tier_key, ratio_values in exchange_tier_chop_trend_ratio.items():
                _, tier = exchange_tier_key.split(':', 1)
                if tier not in tier_to_avg_chop_trend_ratio:
                    tier_to_avg_chop_trend_ratio[tier] = []
                tier_to_avg_chop_trend_ratio[tier].extend(ratio_values)
            
            # Calculate averages
            tier_avg_choppiness = {}
            tier_avg_trend_strength = {}
            tier_avg_chop_trend_ratio = {}
            
            for tier, choppiness_list in tier_to_avg_choppiness.items():
                if choppiness_list:
                    tier_avg_choppiness[tier] = sum(choppiness_list) / len(choppiness_list)
                    
            for tier, trend_strength_list in tier_to_avg_trend_strength.items():
                if trend_strength_list:
                    tier_avg_trend_strength[tier] = sum(trend_strength_list) / len(trend_strength_list)
                    
            for tier, ratio_list in tier_to_avg_chop_trend_ratio.items():
                if ratio_list:
                    tier_avg_chop_trend_ratio[tier] = sum(ratio_list) / len(ratio_list)
            
            # Sort tiers by average choppiness
            sorted_tiers = sorted(tier_avg_choppiness.items(), key=lambda x: x[1], reverse=True)
            
            # Log the top tiers by average choppiness
            if progress_bar and sorted_tiers:
                top_tiers_str = ", ".join([f"{tier}: {chop:.1f}" for tier, chop in sorted_tiers[:5]])
                progress_bar.progress(0.95, text=f"Top 5 tiers by avg choppiness: {top_tiers_str}")
            
            # Create results list
            results = []
            
            # Process each exchange:tier that was analyzed
            for exchange_tier_key in set(exchange_tier_choppiness.keys()) | set(win_counts.keys()):
                # Calculate win rate
                win_rate = (win_counts.get(exchange_tier_key, 0) / total_wins) * 100
                win_rate = round(win_rate, 1)
                
                # Get valid points
                valid_points = exchange_tier_valid_points.get(exchange_tier_key, 0)
                
                # Calculate validity rate (already calculated above)
                validity_rate = exchange_tier_validity.get(exchange_tier_key, 0)
                validity_rate = round(validity_rate, 1)
                
                # Calculate average choppiness across all windows
                choppiness_values = exchange_tier_choppiness.get(exchange_tier_key, [0])
                
                # Use the most recent choppiness as current choppiness
                current_choppiness = choppiness_values[0] if choppiness_values else 0
                current_choppiness = round(current_choppiness, 1)
                
                # Calculate average choppiness across all windows from the past 24 hours
                avg_choppiness = sum(choppiness_values) / len(choppiness_values) if choppiness_values else 0
                avg_choppiness = round(avg_choppiness, 1)
                
                # Calculate trend strength metrics
                trend_strength_values = exchange_tier_trend_strength.get(exchange_tier_key, [0.5])
                current_trend_strength = trend_strength_values[0] if trend_strength_values else 0.5
                current_trend_strength = round(current_trend_strength, 3)
                
                avg_trend_strength = sum(trend_strength_values) / len(trend_strength_values) if trend_strength_values else 0.5
                avg_trend_strength = round(avg_trend_strength, 3)
                
                # Calculate choppiness/trend strength ratio 
                ratio_values = exchange_tier_chop_trend_ratio.get(exchange_tier_key, [0])
                current_ratio = ratio_values[0] if ratio_values else 0
                current_ratio = round(current_ratio, 2)
                
                avg_ratio = sum(ratio_values) / len(ratio_values) if ratio_values else 0
                avg_ratio = round(avg_ratio, 2)
                
                # Calculate average dropout rate across all windows
                dropout_values = exchange_tier_dropout.get(exchange_tier_key, [100])
                avg_dropout = sum(dropout_values) / len(dropout_values) if dropout_values else 100
                avg_dropout = round(avg_dropout, 1)
                
                # Calculate efficiency score (win rate * validity rate)
                efficiency = (win_rate * validity_rate) / 100
                efficiency = round(efficiency, 1)
                
                # Parse exchange and tier from the key
                exchange, tier = exchange_tier_key.split(':', 1)
                
                # Store result
                results.append({
                    'exchange': exchange,
                    'tier': tier,
                    'exchange_tier': exchange_tier_key,
                    'current_choppiness': current_choppiness,
                    'avg_choppiness': avg_choppiness,
                    'current_trend_strength': current_trend_strength,
                    'avg_trend_strength': avg_trend_strength,
                    'current_chop_trend_ratio': current_ratio,
                    'avg_chop_trend_ratio': avg_ratio,
                    'dropout_rate': avg_dropout,
                    'win_rate': win_rate,
                    'validity_rate': validity_rate,
                    'efficiency': efficiency,
                    'valid_points': valid_points,
                    'rank': 0  # Will be filled in later
                })
            
            # Convert to DataFrame and sort
            if not results:
                if progress_bar:
                    progress_bar.progress(1.0, text="No valid tiers found")
                return None, metadata
                
            results_df = pd.DataFrame(results)
            
            # Sort by efficiency score (primary) and current_choppiness (secondary for ties)
            results_df = results_df.sort_values(['efficiency', 'current_choppiness'], ascending=[False, False])
            results_df['rank'] = range(1, len(results_df) + 1)
            
            if progress_bar:
                progress_bar.progress(1.0, text="Analysis complete!")
                
            return results_df, metadata
            
    except Exception as e:
        if progress_bar:
            progress_bar.progress(1.0, text=f"Error: {str(e)}")
        st.error(f"Analysis error: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, metadata

# Format numbers for display
def format_column(value, column_name):
    """Format numbers according to their type"""
    if pd.isna(value):
        return "N/A"
        
    if column_name in ['dropout_rate', 'win_rate', 'efficiency', 'validity_rate']:
        return f"{value:.1f}%"
    elif column_name in ['current_choppiness', 'avg_choppiness']:
        return f"{value:.1f}"
    elif column_name in ['current_trend_strength', 'avg_trend_strength']:
        return f"{value:.3f}"
    elif column_name in ['current_chop_trend_ratio', 'avg_chop_trend_ratio']:
        return f"{value:.2f}"
    elif column_name == 'valid_points':
        return f"{int(value):,}"
    elif column_name == 'rank':
        return f"{int(value)}"
    else:
        return str(value)
        
# Calculate exchange-specific metrics 
def calculate_exchange_metrics(rankings_df):
    """
    Calculate average metrics per exchange across all tiers
    """
    if rankings_df is None or rankings_df.empty:
        return None
        
    # Create a copy to avoid modifying the original
    df = rankings_df.copy()
    
    # Extract exchange from the exchange:tier column
    if 'exchange_tier' in df.columns:
        df['exchange'] = df['exchange_tier'].apply(lambda x: x.split(':', 1)[0] if ':' in str(x) else x)
    
    # Group by exchange and calculate average metrics
    exchange_metrics = df.groupby('exchange').agg({
        'current_choppiness': 'mean',
        'avg_choppiness': 'mean',
        'current_trend_strength': 'mean',
        'avg_trend_strength': 'mean',
        'current_chop_trend_ratio': 'mean',
        'avg_chop_trend_ratio': 'mean',
        'dropout_rate': 'mean',
        'win_rate': 'mean',
        'validity_rate': 'mean',
        'efficiency': 'mean',
        'valid_points': 'sum'
    }).reset_index()
    
    # Round values for display
    for col in exchange_metrics.columns:
        if col != 'exchange' and col != 'valid_points':
            exchange_metrics[col] = exchange_metrics[col].round(2)
    
    # Sort by average choppiness (descending)
    exchange_metrics = exchange_metrics.sort_values('avg_choppiness', ascending=False)
    
    return exchange_metrics

# Main function
def main():
    # Get current time in Singapore
    singapore_tz = pytz.timezone('Asia/Singapore')
    now_sg = datetime.now(singapore_tz)
    current_time_sg = now_sg.strftime("%Y-%m-%d %H:%M:%S")
    
    # Page header
    st.markdown("<h1 style='text-align: center; font-size:28px; margin-bottom: 10px;'>Enhanced Global Tier Analyzer</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; font-size:14px; color:gray;'>Last updated: {current_time_sg} (SGT)</p>", unsafe_allow_html=True)
    
    # Try to get available pairs - silently
    try:
        available_pairs = get_available_pairs()
    except:
        available_pairs = ["BTC", "SOL", "ETH", "DOGE", "XRP", "PNUT", "SUI"]
    
    # Pair selection - simple dropdown
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_pair = st.selectbox(
            "Select Pair",
            options=available_pairs,
            index=0 if available_pairs else None
        )
    
    with col2:
        run_analysis = st.button("ANALYZE NOW", use_container_width=True)
    
    # Explanation of metrics
    st.markdown("""
    **Key Metrics:**
    - **Win Rate:** Percentage of 5000-tick windows where this tier had the highest choppiness
    - **Validity Rate:** Percentage of exchange's data points that have valid data for this tier
    - **Efficiency Score:** Win Rate % × Validity Rate % ÷ 100
    - **Current Choppiness:** Most recent choppiness value using 20-tick rolling window
    - **Avg Choppiness:** Average choppiness across all windows
    - **Trend Strength:** Directional movement strength (0-1, where higher means stronger trend)
    - **Chop/Trend Ratio:** Ratio of choppiness to trend strength (higher values indicate more volatile price action)
    - **Dropout Rate:** Percentage of time tier has missing or invalid data
    - **Valid Ticks:** Number of valid data points analyzed for this tier
    """)
    
    # Run analysis
    if run_analysis and selected_pair:
        # Clear any cached data before new analysis
        st.cache_data.clear()
        
        # Show analysis start time in Singapore timezone
        analysis_start_time = datetime.now(singapore_tz).strftime("%Y-%m-%d %H:%M:%S")
        st.markdown(f"<p style='text-align: center; font-size:14px; color:green;'>Analysis started at: {analysis_start_time} (SGT)</p>", unsafe_allow_html=True)
        
        # Create progress bar
        progress_bar = st.progress(0, text="Starting analysis...")
        
        # Run analysis
        rankings, analysis_metadata = analyze_tiers(selected_pair, progress_bar)
        
        if rankings is not None and not rankings.empty:
            # Display detailed time and data coverage information
            st.header("Analysis Coverage Details")
            
            # Create columns for metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Time Period Analyzed", f"{analysis_metadata.get('time_span_hours', 0):.2f} hours")
            
            with col2:
                st.metric("Total Data Points", f"{analysis_metadata.get('total_data_points', 0):,}")
            
            with col3:
                st.metric("Avg Data Interval", f"{analysis_metadata.get('avg_interval_ms', 0):.1f}ms")
            
            # Show exchange-specific counts
            if analysis_metadata.get('exchange_counts'):
                st.subheader("Data Points per Exchange")
                exchange_data = []
                time_span_seconds = analysis_metadata.get('time_span_seconds', 0)
                
                for exchange, count in analysis_metadata.get('exchange_counts', {}).items():
                    # Calculate average interval for this exchange
                    exchange_interval = 0
                    if time_span_seconds > 0 and count > 0:
                        exchange_interval = (time_span_seconds * 1000) / count
                    
                    # Calculate coverage percentage
                    coverage_pct = 0
                    if time_span_seconds > 0:
                        theoretical_max_for_exchange = time_span_seconds * 2  # 2 ticks per second
                        coverage_pct = (count / theoretical_max_for_exchange) * 100
                    
                    exchange_data.append({
                        "Exchange": exchange,
                        "Data Points": f"{count:,}",
                        "Avg Interval": f"{exchange_interval:.1f}ms",
                        "Coverage %": f"{coverage_pct:.1f}%"
                    })
                
                if exchange_data:
                    # Create dataframe and display
                    exchange_df = pd.DataFrame(exchange_data)
                    st.dataframe(exchange_df, use_container_width=True, hide_index=True)
            
            # Calculate theoretical vs actual data coverage
            theoretical_max = analysis_metadata.get('theoretical_max', 0)
            if theoretical_max > 0:
                total_points = analysis_metadata.get('total_data_points', 0)
                coverage_percent = (total_points / theoretical_max) * 100
                st.markdown(f"**Overall Data Density:** {coverage_percent:.1f}% of theoretical maximum ({theoretical_max:,} points at 500ms intervals)")
            
            # Show results
            st.header("Global Tier Rankings")
            st.markdown("**Efficiency Formula:** Win Rate % × (100% - Dropout Rate %)")
            
            # Verify win rates sum to approximately 100%
            total_win_rate = rankings['win_rate'].sum()
            st.markdown(f"**Total Win Rate:** {total_win_rate:.1f}% (Should be approximately 100%)")
            
            # Format for display
            display_df = rankings.copy()
            
            # Rename and select columns for display
            display_df = display_df.rename(columns={
                'exchange_tier': 'Exchange:Tier',
                'efficiency': 'Efficiency Score',
                'current_choppiness': 'Current Choppiness',
                'avg_choppiness': 'Avg Choppiness',
                'current_trend_strength': 'Current Trend Str',
                'avg_trend_strength': 'Avg Trend Str',
                'current_chop_trend_ratio': 'Current Chop/Trend',
                'avg_chop_trend_ratio': 'Avg Chop/Trend',
                'dropout_rate': 'Dropout Rate (%)',
                'win_rate': 'Win Rate (%)',
                'validity_rate': 'Validity Rate (%)',
                'valid_points': 'Valid Ticks',
                'rank': 'Rank'
            })
            
            # Select columns for display
            display_columns = [
                'Rank',
                'Exchange:Tier', 
                'Efficiency Score',
                'Win Rate (%)',
                'Validity Rate (%)',
                'Dropout Rate (%)',
                'Current Choppiness',
                'Avg Choppiness',
                'Current Trend Str',
                'Avg Trend Str',
                'Avg Chop/Trend',
                'Valid Ticks'
            ]
            
            # Filter to columns that exist
            available_columns = [col for col in display_columns if col in display_df.columns]
            display_df = display_df[available_columns]
            
            # Format numeric columns
            for col in display_df.columns:
                if col != 'Exchange:Tier':
                    col_name = col.lower().replace(' ', '_').replace('(%)', '').replace(':', '_').replace('/', '_')
                    display_df[col] = display_df[col].apply(lambda x: format_column(x, col_name))
            
            # Show table
            st.dataframe(
                display_df,
                use_container_width=True,
                height=min(800, 100 + (len(display_df) * 35)),  # Adaptive height
                hide_index=True  # Hide the index column
            )
            
            # Show top recommendations
            st.header("Recommended Tiers")
            top_count = min(3, len(display_df))
            top_tiers = display_df.iloc[:top_count]
            
            for i in range(top_count):
                if i == 0:
                    st.markdown(f"**Primary Tier:** {top_tiers.iloc[i]['Exchange:Tier']} (Rank {top_tiers.iloc[i]['Rank']})")
                else:
                    st.markdown(f"**Fallback Tier {i}:** {top_tiers.iloc[i]['Exchange:Tier']} (Rank {top_tiers.iloc[i]['Rank']})")
            
            # Display analysis completion time
            analysis_end_time = datetime.now(singapore_tz).strftime("%Y-%m-%d %H:%M:%S")
            st.markdown(f"<p style='text-align: center; font-size:14px; color:green;'>Analysis completed at: {analysis_end_time} (SGT)</p>", unsafe_allow_html=True)
        else:
            st.error(f"No valid data found for {selected_pair}. Please try another pair.")
    else:
        st.info("Select a pair and click ANALYZE NOW to find the optimal exchange and depth tier combination.")

if __name__ == "__main__":
    main()