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
    
    /* Highlight cells based on value */
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

def analyze_tiers(pair_name, progress_bar=None):
    """Analyze all exchange-tier combinations"""
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
            
            # Process each exchange and tier
            results = []
            
            # Calculate time intervals for win rate
            timeframes = [10, 20, 50, 100, 200]
            
            total_exchanges = len(exchanges)
            for i, exchange in enumerate(exchanges):
                # Update progress
                if progress_bar:
                    progress_bar.progress(0.3 + (0.6 * i / total_exchanges), 
                                         text=f"Analyzing {exchange} ({i+1}/{total_exchanges})")
                
                # Filter for this exchange
                exchange_df = df[df['exchange_name'] == exchange].copy()
                
                # Process each tier
                for tier_col in tier_columns:
                    # Get tier name
                    tier_name = tier_values.get(tier_col, tier_col)
                    
                    # Calculate metrics
                    # 1. Dropout rate - percentage of missing or zero values
                    total_points = len(exchange_df)
                    nan_or_zero = (exchange_df[tier_col].isna() | (exchange_df[tier_col] <= 0)).sum()
                    dropout_rate = (nan_or_zero / total_points) * 100 if total_points > 0 else 100
                    
                    # Skip completely empty tiers
                    if dropout_rate >= 99.9:
                        continue
                    
                    # 2. Get valid prices
                    prices = exchange_df[tier_col].dropna()
                    prices = prices[prices > 0]
                    
                    # Skip if not enough data
                    if len(prices) < 100:
                        continue
                    
                    # 3. Calculate choppiness - using same algorithm as singlepairtieranalyzer.py
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
                    
                    # 4. Calculate direction changes (%)
                    signs = np.sign(diff)
                    direction_changes = (signs.shift(1) != signs).sum()
                    direction_change_pct = (direction_changes / (len(signs) - 1)) * 100 if len(signs) > 1 else 0
                    
                    # 5. Calculate tick ATR %
                    mean_price = prices.mean()
                    tick_atr = diff.abs().mean()
                    tick_atr_pct = (tick_atr / mean_price) * 100
                    
                    # 6. Calculate corrected efficiency score: run_rate * (100-dropout_rate)
                    # Run rate = % of time the tier has valid data (opposite of dropout rate)
                    run_rate = 100 - dropout_rate
                    efficiency = (run_rate * choppiness) / 100
                    
                    # 7. Calculate win rate
                    win_rates = []
                    for tf in timeframes:
                        if len(prices) <= tf:
                            continue
                            
                        # Calculate price changes over this timeframe
                        price_shifts = prices.shift(-tf) - prices
                        increases = (price_shifts > 0).sum()
                        decreases = (price_shifts < 0).sum()
                        total = increases + decreases
                        
                        if total > 0:
                            win_rate = (increases / total) * 100
                            win_rates.append(win_rate)
                    
                    # Average winrate across different timeframes
                    avg_win_rate = sum(win_rates) / len(win_rates) if win_rates else 50.0
                    
                    # Store result
                    results.append({
                        'exchange': exchange,
                        'tier': tier_name,
                        'exchange_tier': f"{exchange}:{tier_name}",
                        'choppiness': choppiness,
                        'direction_changes': direction_change_pct,
                        'tick_atr_pct': tick_atr_pct,
                        'dropout_rate': dropout_rate,
                        'run_rate': run_rate,
                        'efficiency': efficiency,
                        'win_rate': avg_win_rate,
                        'valid_points': len(prices)
                    })
            
            if progress_bar:
                progress_bar.progress(0.9, text="Ranking results...")
            
            # Convert to DataFrame and sort
            if not results:
                if progress_bar:
                    progress_bar.progress(1.0, text="No valid tiers found")
                return None
                
            results_df = pd.DataFrame(results)
            
            # Sort by efficiency, then by win_rate as secondary factor
            results_df = results_df.sort_values(['efficiency', 'win_rate'], ascending=[False, False])
            
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
        
    if column_name in ['run_rate', 'dropout_rate', 'direction_changes', 'win_rate']:
        return f"{value:.1f}%"
    elif column_name == 'tick_atr_pct':
        return f"{value:.4f}"
    elif column_name == 'choppiness':
        return f"{value:.1f}"
    elif column_name == 'efficiency':
        return f"{value:.2f}"
    elif column_name == 'valid_points':
        return f"{int(value):,}"
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
    - **Efficiency Score:** Choppiness × Run Rate % (higher is better)
    - **Run Rate:** Percentage of time tier has valid data (opposite of Dropout Rate)
    - **Choppiness:** Measures price oscillation intensity (higher means more active)
    - **Win Rate:** Percentage of profitable price movements across multiple timeframes
    - **Direction Changes:** Frequency of price direction reversals (%)
    - **Tick ATR %:** Average tick-to-tick price change percentage
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
            st.markdown("**Efficiency Formula:** Choppiness × Run Rate %")
            
            # Format for display
            display_df = rankings.copy()
            
            # Rename and select columns for display
            display_df = display_df.rename(columns={
                'exchange_tier': 'Exchange:Tier',
                'efficiency': 'Efficiency Score',
                'choppiness': 'Choppiness',
                'dropout_rate': 'Dropout Rate (%)',
                'run_rate': 'Run Rate (%)',
                'win_rate': 'Win Rate (%)',
                'direction_changes': 'Direction Changes (%)',
                'tick_atr_pct': 'Tick ATR %',
                'valid_points': 'Valid Points'
            })
            
            # Select columns for display
            display_columns = [
                'Exchange:Tier', 
                'Efficiency Score', 
                'Run Rate (%)',
                'Choppiness', 
                'Win Rate (%)',
                'Direction Changes (%)',
                'Tick ATR %',
                'Valid Points'
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
                height=min(800, 100 + (len(display_df) * 35))  # Adaptive height
            )
            
            # Show top recommendations
            st.header("Recommended Tiers")
            top_count = min(3, len(display_df))
            top_tiers = display_df.iloc[:top_count]
            
            for i in range(top_count):
                if i == 0:
                    st.markdown(f"**Primary Tier:** {top_tiers.iloc[i]['Exchange:Tier']}")
                else:
                    st.markdown(f"**Fallback Tier {i}:** {top_tiers.iloc[i]['Exchange:Tier']}")
            
            # Display analysis completion time
            analysis_end_time = datetime.now(singapore_tz).strftime("%Y-%m-%d %H:%M:%S")
            st.markdown(f"<p style='text-align: center; font-size:14px; color:green;'>Analysis completed at: {analysis_end_time} (SGT)</p>", unsafe_allow_html=True)
        else:
            st.error(f"No valid data found for {selected_pair}. Please try another pair.")
    else:
        st.info("Select a pair and click ANALYZE NOW to find the optimal exchange and depth tier combination.")

if __name__ == "__main__":
    main()