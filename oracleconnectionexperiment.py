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
    default_pairs = ["BTC/USDT", "SOL/USDT", "ETH/USDT", "DOGE/USDT", "XRP/USDT"]
    
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

def analyze_tiers(pair_name, progress_bar=None, time_intervals=24):
    """Analyze all exchange-tier combinations with time-based choppiness tracking"""
    try:
        with get_session() as session:
            if not session:
                return None

            # Get current date for table name
            current_date = datetime.now().strftime("%Y%m%d")
            table_name = f"oracle_exchange_price_partition_v1_{current_date}"
            
            # Check if today's table exists
            table_exists_query = text(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = :table_name
                );
            """)
            
            if not session.execute(table_exists_query, {"table_name": table_name}).scalar():
                # Try yesterday if today doesn't exist
                yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
                table_name = f"oracle_exchange_price_partition_v1_{yesterday}"
                
                # Check if yesterday's table exists
                if not session.execute(table_exists_query, {"table_name": table_name}).scalar():
                    if progress_bar:
                        progress_bar.progress(1.0, text="No data tables found")
                    return None
            
            # Define tier columns and mapping (using the same scheme as in singlepairtieranalyzer.py)
            tier_columns = [
                'price_1', 'price_2', 'price_3', 'price_4', 'price_5',
                'price_6', 'price_7', 'price_8', 'price_9', 'price_10',
                'price_11', 'price_12', 'price_13', 'price_14', 'price_15'
            ]
            
            tier_values = {
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
                'price_12': '1M',
                'price_13': '2M',
                'price_14': '3M',
                'price_15': '4M',
            }
            
            # Join all tier columns for the query
            price_columns = ", ".join(tier_columns)
            
            if progress_bar:
                progress_bar.progress(0.1, text="Fetching data...")
            
            # Query to fetch data, also get created_at for time-based analysis
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
                LIMIT 5000
            """)
            
            result = session.execute(query, {"pair_name": pair_name})
            all_data = result.fetchall()
            
            if not all_data:
                if progress_bar:
                    progress_bar.progress(1.0, text=f"No data found for {pair_name}")
                return None
            
            if progress_bar:
                progress_bar.progress(0.3, text=f"Processing {len(all_data)} data points...")
            
            # Create DataFrame
            columns = ['exchange_name', 'created_at'] + tier_columns
            df = pd.DataFrame(all_data, columns=columns)
            
            # Convert numeric columns
            for col in tier_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Get unique exchanges
            exchanges = df['exchange_name'].unique()

            # Initialize dictionaries to store results
            exchange_tier_choppiness = {}  # Store choppiness for each exchange:tier
            exchange_tier_dropout = {}     # Store dropout rate for each exchange:tier
            
            # Track which tier had highest choppiness in each time interval
            highest_choppiness_counts = {}
            
            # Process each exchange and tier to calculate choppiness
            total_exchanges = len(exchanges)
            for i, exchange in enumerate(exchanges):
                # Update progress
                if progress_bar:
                    progress_bar.progress(0.3 + (0.3 * i / total_exchanges), 
                                        text=f"Calculating choppiness for {exchange} ({i+1}/{total_exchanges})")
                
                # Filter for this exchange
                exchange_df = df[df['exchange_name'] == exchange].copy()
                
                # Process each tier
                for tier_col in tier_columns:
                    # Get tier name
                    tier_name = tier_values.get(tier_col, tier_col)
                    exchange_tier_key = f"{exchange}:{tier_name}"
                    
                    # Calculate dropout rate
                    total_points = len(exchange_df)
                    nan_or_zero = (exchange_df[tier_col].isna() | (exchange_df[tier_col] <= 0)).sum()
                    dropout_rate = (nan_or_zero / total_points) * 100 if total_points > 0 else 100
                    
                    # Store dropout rate
                    exchange_tier_dropout[exchange_tier_key] = round(dropout_rate, 1)
                    
                    # Skip completely empty tiers
                    if dropout_rate >= 99.9:
                        continue
                    
                    # Get valid prices
                    prices = exchange_df[tier_col].dropna()
                    prices = prices[prices > 0]
                    
                    # Skip if not enough data
                    if len(prices) < 100:
                        continue
                    
                    # Calculate choppiness
                    window = min(20, len(prices) // 10)
                    diff = prices.diff().dropna()
                    
                    if len(diff) < window:
                        continue
                    
                    # Calculate sum of absolute changes
                    sum_abs_changes = diff.abs().rolling(window, min_periods=1).sum()
                    
                    # Calculate price range
                    price_range = prices.rolling(window, min_periods=1).max() - prices.rolling(window, min_periods=1).min()
                    
                    # Avoid division by zero
                    epsilon = 1e-10
                    choppiness_values = 100 * sum_abs_changes / (price_range + epsilon)
                    
                    # Cap extreme values
                    choppiness_values = np.minimum(choppiness_values, 1000)
                    
                    # Calculate mean choppiness
                    choppiness = choppiness_values.mean()
                    
                    # Store choppiness
                    exchange_tier_choppiness[exchange_tier_key] = round(choppiness, 1)
            
            # Divide data into time intervals to calculate win rate
            if progress_bar:
                progress_bar.progress(0.6, text="Calculating win rates across time intervals...")
                
            # Calculate time intervals (each interval represents a sample of data)
            total_data_points = len(df)
            points_per_interval = max(200, total_data_points // time_intervals)
            actual_intervals = total_data_points // points_per_interval
            
            if progress_bar:
                progress_bar.progress(0.65, text=f"Analyzing {actual_intervals} time intervals...")
            
            # Track which exchange:tier had highest choppiness in each interval
            interval_winners = {}
            
            # Process each time interval
            for j in range(actual_intervals):
                start_idx = j * points_per_interval
                end_idx = min((j + 1) * points_per_interval, total_data_points)
                
                if end_idx - start_idx < points_per_interval * 0.5:  # Skip if too small
                    continue
                    
                interval_df = df.iloc[start_idx:end_idx].copy()
                
                # Find highest choppiness for this interval across all exchange:tier combinations
                highest_choppiness = 0
                highest_tier = None
                
                # Process each exchange in this interval
                for exchange in exchanges:
                    # Filter for this exchange
                    exchange_interval_df = interval_df[interval_df['exchange_name'] == exchange].copy()
                    
                    if len(exchange_interval_df) < 50:  # Skip if not enough data
                        continue
                    
                    # Process each tier
                    for tier_col in tier_columns:
                        tier_name = tier_values.get(tier_col, tier_col)
                        exchange_tier_key = f"{exchange}:{tier_name}"
                        
                        # Skip if we already know this tier has high dropout rate
                        if exchange_tier_key in exchange_tier_dropout and exchange_tier_dropout[exchange_tier_key] > 90:
                            continue
                            
                        # Get valid prices for this tier in this interval
                        prices = pd.to_numeric(exchange_interval_df[tier_col], errors='coerce').dropna()
                        prices = prices[prices > 0]
                        
                        # Skip if not enough data
                        if len(prices) < 50:
                            continue
                        
                        # Calculate choppiness
                        window = min(20, len(prices) // 5)
                        diff = prices.diff().dropna()
                        
                        if len(diff) < window:
                            continue
                        
                        # Calculate sum of absolute changes
                        sum_abs_changes = diff.abs().rolling(window, min_periods=1).sum()
                        
                        # Calculate price range
                        price_range = prices.rolling(window, min_periods=1).max() - prices.rolling(window, min_periods=1).min()
                        
                        # Avoid division by zero
                        epsilon = 1e-10
                        choppiness_values = 100 * sum_abs_changes / (price_range + epsilon)
                        
                        # Cap extreme values
                        choppiness_values = np.minimum(choppiness_values, 1000)
                        
                        # Calculate mean choppiness for this interval
                        interval_choppiness = choppiness_values.mean()
                        
                        # Check if this is the highest choppiness so far
                        if interval_choppiness > highest_choppiness:
                            highest_choppiness = interval_choppiness
                            highest_tier = exchange_tier_key
                
                # Record the winner for this interval
                if highest_tier:
                    if highest_tier not in interval_winners:
                        interval_winners[highest_tier] = 0
                    interval_winners[highest_tier] += 1
            
            # Calculate win rates
            if progress_bar:
                progress_bar.progress(0.8, text="Calculating final rankings...")
                
            total_intervals = sum(interval_winners.values())
            
            # Calculate win rate for each exchange:tier
            win_rates = {}
            for tier, count in interval_winners.items():
                win_rate = (count / total_intervals) * 100 if total_intervals > 0 else 0
                win_rates[tier] = round(win_rate, 1)
            
            # Create final results
            results = []
            
            # Process each exchange:tier that has choppiness data
            for exchange_tier, choppiness in exchange_tier_choppiness.items():
                # Get dropout rate
                dropout_rate = exchange_tier_dropout.get(exchange_tier, 100)
                
                # Get win rate (0 if this tier never won an interval)
                win_rate = win_rates.get(exchange_tier, 0)
                
                # Calculate efficiency score
                run_rate = 100 - dropout_rate
                efficiency = (win_rate * run_rate) / 100
                efficiency = round(efficiency, 1)
                
                # Parse exchange and tier from the key
                exchange, tier = exchange_tier.split(':', 1)
                
                # Store result
                results.append({
                    'exchange': exchange,
                    'tier': tier,
                    'exchange_tier': exchange_tier,
                    'choppiness': choppiness,
                    'dropout_rate': dropout_rate,
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
    - **Win Rate:** Percentage of time intervals where this tier had the highest choppiness
    - **Efficiency Score:** Win Rate % × (100% - Dropout Rate %)
    - **Choppiness:** Measures price oscillation intensity
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
        
        # Determine time intervals (approximately 24 time intervals)
        time_intervals = 24
        
        # Run analysis
        rankings = analyze_tiers(selected_pair, progress_bar, time_intervals)
        
        if rankings is not None and not rankings.empty:
            # Show results
            st.header("Global Tier Rankings")
            st.markdown("**Efficiency Formula:** Win Rate % × (100% - Dropout Rate %)")
            
            # Verify win rates sum to approximately 100%
            total_win_rate = rankings['win_rate'].sum()
            st.markdown(f"**Total Win Rate:** {total_win_rate:.1f}% (Sum of all tier win rates)")
            
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