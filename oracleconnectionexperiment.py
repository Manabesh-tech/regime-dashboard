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

# Get available pairs from the database
def get_available_pairs():
    """Fetch available trading pairs"""
    default_pairs = ["BTC", "SOL", "ETH", "DOGE", "XRP"]
    
    try:
        with get_session() as session:
            if not session:
                return default_pairs

            # Get current date for table name
            current_date = datetime.now().strftime("%Y%m%d")
            table_name = f"oracle_exchange_price_partition_v1_{current_date}"
            
            # Query
            query = text(f"""
                SELECT DISTINCT pair_name 
                FROM {table_name}
                ORDER BY pair_name
            """)
            
            result = session.execute(query)
            pairs = [row[0] for row in result]
            
            # If current date doesn't have data, try yesterday
            if not pairs:
                yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
                table_name = f"oracle_exchange_price_partition_v1_{yesterday}"
                query = text(f"""
                    SELECT DISTINCT pair_name 
                    FROM {table_name}
                    ORDER BY pair_name
                """)
                result = session.execute(query)
                pairs = [row[0] for row in result]
            
            return sorted(pairs) if pairs else default_pairs

    except Exception as e:
        st.error(f"Error fetching pairs: {e}")
        return default_pairs

def analyze_tiers(pair_name, progress_bar=None):
    """Analyze all exchange-tier combinations with time-based windows"""
    try:
        with get_session() as session:
            if not session:
                return None

            # Define tier columns and mapping
            tier_columns = [
                'price_1', 'price_2', 'price_3', 'price_4', 'price_5',
                'price_6', 'price_7', 'price_8', 'price_9', 'price_10',
                'price_11', 'price_12', 'price_13', 'price_14', 'price_15'
            ]
            
            tier_values = {
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
            
            # Join all tier columns for the query
            price_columns = ", ".join(tier_columns)
            
            # Get Singapore time for proper date handling
            singapore_tz = pytz.timezone('Asia/Singapore')
            now_sg = datetime.now(singapore_tz)
            
            # Get current date in Singapore time
            current_date_sg = now_sg.strftime("%Y%m%d")
            
            # Add 2 days prior for data collection
            dates_to_check = []
            for i in range(3):  # Today, yesterday, day before
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
                
                # Query to fetch data from this table - fetch all data from the table
                if progress_bar:
                    progress_bar.progress(0.1, text=f"Fetching data from {table_name}...")
                
                query = text(f"""
                    SELECT 
                        source as exchange_name,
                        created_at,
                        {price_columns}
                    FROM 
                        {table_name}
                    WHERE 
                        pair_name = :pair_name
                    ORDER BY 
                        created_at DESC
                """)
                
                result = session.execute(query, {"pair_name": pair_name})
                table_data = result.fetchall()
                
                if table_data:
                    all_data.extend(table_data)
                    if progress_bar:
                        progress_bar.progress(0.2, text=f"Found {len(table_data)} rows in {table_name}")
                        
                # Stop if we have enough data
                if len(all_data) >= 50000:  # This should be enough for multiple 5000-tick windows
                    break
            
            if not all_data:
                if progress_bar:
                    progress_bar.progress(1.0, text=f"No data found for {pair_name}")
                return None
                
            if progress_bar:
                progress_bar.progress(0.3, text=f"Processing {len(all_data)} data points...")
            
            # Create DataFrame
            columns = ['exchange_name', 'created_at'] + tier_columns
            df = pd.DataFrame(all_data, columns=columns)
            
            # Sort by timestamp to ensure proper order
            df.sort_values('created_at', ascending=False, inplace=True)
            
            # Convert numeric columns
            for col in tier_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Get unique exchanges
            exchanges = df['exchange_name'].unique()
            
            if progress_bar:
                progress_bar.progress(0.4, text=f"Found {len(exchanges)} exchanges")
            
            # Non-overlapping 5000-tick windows
            window_size = 5000
            total_points = len(df)
            num_windows = total_points // window_size
            
            # Lower the minimum window size requirement if needed
            if num_windows < 3 and total_points >= 1000:
                # If we don't have enough data for 5000-tick windows, try smaller ones
                window_size = min(total_points // 3, 2500)  # Ensure at least 3 windows if possible
                num_windows = total_points // window_size
                if progress_bar:
                    progress_bar.progress(0.5, text=f"Using smaller window size ({window_size}). Created {num_windows} windows.")
            
            # Accept even just 1 window if that's all we have
            if num_windows == 0 and total_points >= 1000:
                window_size = total_points
                num_windows = 1
                if progress_bar:
                    progress_bar.progress(0.5, text=f"Using all available data as single window ({window_size} points)")
                
            if num_windows == 0:
                if progress_bar:
                    progress_bar.progress(1.0, text=f"Insufficient data: found {total_points} points, need at least 1000")
                return None
                
            if progress_bar:
                progress_bar.progress(0.5, text=f"Creating {num_windows} non-overlapping windows...")
                
            # Initialize variables to track tier performance
            win_counts = {}  # To count how many times each tier wins
            exchange_tier_choppiness = {}  # Store the overall choppiness average
            exchange_tier_dropout = {}     # Store the overall dropout rate
            
            # Process each 5000-tick window
            for window_idx in range(num_windows):
                if progress_bar:
                    progress_bar.progress(0.5 + (0.4 * window_idx / num_windows), 
                                         text=f"Analyzing window {window_idx+1}/{num_windows}")
                
                # Calculate start and end indices for this window (non-overlapping)
                start_idx = window_idx * window_size
                end_idx = (window_idx + 1) * window_size
                
                if end_idx > total_points:
                    end_idx = total_points  # Use whatever data is left
                
                # Get the data for this window
                window_df = df.iloc[start_idx:end_idx].copy()
                
                # Skip if window is too small
                if len(window_df) < 0.8 * window_size:  # Need at least 80% of full window
                    continue
                
                # Track best choppiness for this window
                best_choppiness = 0
                best_tier = None
                
                # Process each exchange and tier in this window
                for exchange in exchanges:
                    # Filter for this exchange
                    exchange_df = window_df[window_df['exchange_name'] == exchange].copy()
                    
                    if len(exchange_df) < 0.8 * window_size:  # Need at least 80% of data points
                        continue
                    
                    # Process each tier
                    for tier_col in tier_columns:
                        tier_name = tier_values.get(tier_col, tier_col)
                        exchange_tier_key = f"{exchange}:{tier_name}"
                        
                        # Calculate dropout rate for this window
                        total_points_in_window = len(exchange_df)
                        nan_or_zero = (exchange_df[tier_col].isna() | (exchange_df[tier_col] <= 0)).sum()
                        dropout_rate = (nan_or_zero / total_points_in_window) * 100
                        
                        # Track overall dropout rate (average across all windows)
                        if exchange_tier_key not in exchange_tier_dropout:
                            exchange_tier_dropout[exchange_tier_key] = []
                        exchange_tier_dropout[exchange_tier_key].append(dropout_rate)
                        
                        # Skip if dropout rate is too high
                        if dropout_rate > 90:
                            continue
                        
                        # Get valid prices
                        prices = exchange_df[tier_col].dropna()
                        prices = prices[prices > 0]
                        
                # Skip if not enough data or too high dropout rate
                if len(prices) < 0.5 * window_size or dropout_rate > 95:  # More lenient
                            continue
                        
                        # Calculate choppiness with 20-tick window
                        window_size_choppiness = 20  # As specified in the instructions
                        diff = prices.diff().dropna()
                        
                        if len(diff) < window_size_choppiness:
                            continue
                        
                        # Calculate sum of absolute changes
                        sum_abs_changes = diff.abs().rolling(window_size_choppiness, min_periods=1).sum()
                        
                        # Calculate price range
                        price_range = prices.rolling(window_size_choppiness, min_periods=1).max() - prices.rolling(window_size_choppiness, min_periods=1).min()
                        
                        # Avoid division by zero
                        epsilon = 1e-10
                        choppiness_values = 100 * sum_abs_changes / (price_range + epsilon)
                        
                        # Cap extreme values
                        choppiness_values = np.minimum(choppiness_values, 1000)
                        
                        # Calculate mean choppiness for this 5000-tick window
                        avg_choppiness = choppiness_values.mean()
                        
                        # Track overall choppiness (average across all windows)
                        if exchange_tier_key not in exchange_tier_choppiness:
                            exchange_tier_choppiness[exchange_tier_key] = []
                        exchange_tier_choppiness[exchange_tier_key].append(avg_choppiness)
                        
                        # Check if this is the best choppiness for this window
                        if avg_choppiness > best_choppiness:
                            best_choppiness = avg_choppiness
                            best_tier = exchange_tier_key
                
                # Count win for this window
                if best_tier:
                    if best_tier not in win_counts:
                        win_counts[best_tier] = 0
                    win_counts[best_tier] += 1
                    
                    # Log which tier won for the first few windows to diagnose
                    if window_idx < 5 and progress_bar:
                        progress_bar.progress(0.5 + (0.4 * window_idx / num_windows), 
                                             text=f"Window {window_idx+1}: Winner is {best_tier} with choppiness {best_choppiness:.1f}")
            
            if progress_bar:
                progress_bar.progress(0.9, text="Calculating final rankings...")
            
            # Calculate final metrics
            total_wins = sum(win_counts.values())
            
            if total_wins == 0:
                if progress_bar:
                    progress_bar.progress(1.0, text="No valid winners found in any window")
                return None
            
            # Create results list
            results = []
            
            # Add average choppiness across all exchanges for each tier
            tier_to_avg_choppiness = {}
            
            # Calculate average choppiness per tier across all exchanges
            for exchange_tier_key, choppiness_values in exchange_tier_choppiness.items():
                _, tier = exchange_tier_key.split(':', 1)
                if tier not in tier_to_avg_choppiness:
                    tier_to_avg_choppiness[tier] = []
                tier_to_avg_choppiness[tier].extend(choppiness_values)
            
            # Calculate averages
            tier_avg_choppiness = {}
            for tier, choppiness_list in tier_to_avg_choppiness.items():
                if choppiness_list:
                    tier_avg_choppiness[tier] = sum(choppiness_list) / len(choppiness_list)
            
            # Sort tiers by average choppiness
            sorted_tiers = sorted(tier_avg_choppiness.items(), key=lambda x: x[1], reverse=True)
            
            # Log the top tiers by average choppiness
            if progress_bar:
                top_tiers_str = ", ".join([f"{tier}: {chop:.1f}" for tier, chop in sorted_tiers[:5]])
                progress_bar.progress(0.95, text=f"Top 5 tiers by avg choppiness: {top_tiers_str}")
            
            # Process each exchange:tier that was analyzed
            for exchange_tier_key in set(exchange_tier_choppiness.keys()) | set(win_counts.keys()):
                # Calculate win rate
                win_rate = (win_counts.get(exchange_tier_key, 0) / total_wins) * 100
                win_rate = round(win_rate, 1)
                
                # Calculate average choppiness across all windows
                choppiness_values = exchange_tier_choppiness.get(exchange_tier_key, [0])
                avg_choppiness = sum(choppiness_values) / len(choppiness_values)
                avg_choppiness = round(avg_choppiness, 1)
                
                # Calculate average dropout rate across all windows
                dropout_values = exchange_tier_dropout.get(exchange_tier_key, [100])
                avg_dropout = sum(dropout_values) / len(dropout_values)
                avg_dropout = round(avg_dropout, 1)
                
                # Calculate efficiency score
                efficiency = (win_rate * (100 - avg_dropout)) / 100
                efficiency = round(efficiency, 1)
                
                # Parse exchange and tier from the key
                exchange, tier = exchange_tier_key.split(':', 1)
                
                # Store result
                results.append({
                    'exchange': exchange,
                    'tier': tier,
                    'exchange_tier': exchange_tier_key,
                    'choppiness': avg_choppiness,
                    'dropout_rate': avg_dropout,
                    'win_rate': win_rate,
                    'efficiency': efficiency,
                    'rank': 0  # Will be filled in later
                })
            
            # Convert to DataFrame and sort
            if not results:
                if progress_bar:
                    progress_bar.progress(1.0, text="No valid tiers found")
                return None
                
            results_df = pd.DataFrame(results)
            
            # Sort by efficiency score and assign ranks
            results_df = results_df.sort_values('efficiency', ascending=False)
            results_df['rank'] = range(1, len(results_df) + 1)
            
            if progress_bar:
                progress_bar.progress(1.0, text="Analysis complete!")
                
            return results_df
            
    except Exception as e:
        if progress_bar:
            progress_bar.progress(1.0, text=f"Error: {str(e)}")
        st.error(f"Analysis error: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None

# Format numbers for display
def format_column(value, column_name):
    """Format numbers according to their type"""
    if pd.isna(value):
        return "N/A"
        
    if column_name in ['dropout_rate', 'win_rate', 'efficiency']:
        return f"{value:.1f}%"
    elif column_name == 'choppiness':
        return f"{value:.1f}"
    elif column_name == 'valid_points':
        return f"{int(value):,}"
    elif column_name == 'rank':
        return f"{int(value)}"
    else:
        return str(value)

# Main function
def main():
    # Get current time in Singapore
    singapore_tz = pytz.timezone('Asia/Singapore')
    now_sg = datetime.now(singapore_tz)
    current_time_sg = now_sg.strftime("%Y-%m-%d %H:%M:%S")
    
    # Page header
    st.markdown("<h1 style='text-align: center; font-size:28px; margin-bottom: 10px;'>Enhanced Global Tier Analyzer</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; font-size:14px; color:gray;'>Last updated: {current_time_sg} (SGT)</p>", unsafe_allow_html=True)
    
    # Try to get available pairs
    try:
        available_pairs = get_available_pairs()
    except:
        available_pairs = ["BTC/USDT", "SOL/USDT", "ETH/USDT", "DOGE/USDT", "XRP/USDT"]
    
    # Pair selection
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
    - **Efficiency Score:** Win Rate % × (100% - Dropout Rate %)
    - **Choppiness:** Average value using 20-tick rolling window calculation
    - **Dropout Rate:** Percentage of time tier has missing or invalid data
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
        rankings = analyze_tiers(selected_pair, progress_bar)
        
        if rankings is not None and not rankings.empty:
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
                'choppiness': 'Choppiness',
                'dropout_rate': 'Dropout Rate (%)',
                'win_rate': 'Win Rate (%)',
                'rank': 'Rank'
            })
            
            # Select columns for display
            display_columns = [
                'Rank',
                'Exchange:Tier', 
                'Efficiency Score',
                'Win Rate (%)',
                'Dropout Rate (%)',
                'Choppiness'
            ]
            
            # Filter to columns that exist
            available_columns = [col for col in display_columns if col in display_df.columns]
            display_df = display_df[available_columns]
            
            # Format numeric columns
            for col in display_df.columns:
                if col != 'Exchange:Tier':
                    col_name = col.lower().replace(' ', '_').replace('(%)', '').replace(':', '_')
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