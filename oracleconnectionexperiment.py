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
    page_title="Global Tier Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Basic CSS
st.markdown("""
<style>
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
    default_pairs = ["BTC", "SOL", "ETH", "TRUMP"]
    
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
            
            # Define tier columns and mapping
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
                'price_12': '1000k',
                'price_13': '2000k',
                'price_14': '3000k',
                'price_15': '4000k',
            }
            
            # Join all tier columns for the query
            price_columns = ", ".join(tier_columns)
            
            if progress_bar:
                progress_bar.progress(0.1, text="Fetching data...")
            
            # Query to fetch data
            query = text(f"""
                SELECT 
                    source as exchange_name,
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
                return None
            
            if progress_bar:
                progress_bar.progress(0.3, text="Processing data...")
            
            # Create DataFrame
            columns = ['exchange_name'] + tier_columns
            df = pd.DataFrame(all_data, columns=columns)
            
            # Convert numeric
            for col in tier_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Get unique exchanges
            exchanges = df['exchange_name'].unique()
            
            # Process each exchange and tier
            results = []
            
            for exchange in exchanges:
                # Filter for this exchange
                exchange_df = df[df['exchange_name'] == exchange].copy()
                
                # Process each tier
                for tier_col in tier_columns:
                    # Get tier name
                    tier_name = tier_values.get(tier_col, tier_col)
                    
                    # Calculate metrics
                    # 1. Dropout rate
                    total_points = len(exchange_df)
                    nan_or_zero = (exchange_df[tier_col].isna() | (exchange_df[tier_col] == 0)).sum()
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
                    
                    # 3. Calculate choppiness
                    window = min(20, len(prices) // 10)
                    diff = prices.diff().dropna()
                    
                    if len(diff) < window:
                        continue
                        
                    sum_abs_changes = diff.abs().rolling(window, min_periods=1).sum()
                    price_range = prices.rolling(window, min_periods=1).max() - prices.rolling(window, min_periods=1).min()
                    
                    # Avoid division by zero
                    epsilon = 1e-10
                    choppiness_values = 100 * sum_abs_changes / (price_range + epsilon)
                    choppiness = choppiness_values.mean()
                    
                    # 4. Calculate efficiency
                    efficiency = choppiness * ((100 - dropout_rate) / 100)
                    
                    # Store result
                    results.append({
                        'exchange': exchange,
                        'tier': tier_name,
                        'exchange_tier': f"{exchange}:{tier_name}",
                        'choppiness': choppiness,
                        'dropout_rate': dropout_rate,
                        'efficiency': efficiency,
                        'valid_points': len(prices)
                    })
            
            if progress_bar:
                progress_bar.progress(0.8, text="Ranking results...")
            
            # Convert to DataFrame and sort
            if not results:
                return None
                
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values('efficiency', ascending=False)
            
            if progress_bar:
                progress_bar.progress(1.0, text="Done!")
                
            return results_df
            
    except Exception as e:
        st.error(f"Analysis error: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None

# Main function
def main():
    st.title("Global Tier Analyzer")
    
    # Get current time
    singapore_tz = pytz.timezone('Asia/Singapore')
    now_sg = datetime.now(singapore_tz)
    current_time_sg = now_sg.strftime("%Y-%m-%d %H:%M:%S")
    st.write(f"Current time: {current_time_sg} (SGT)")
    
    # Try to get available pairs
    try:
        available_pairs = get_available_pairs()
    except:
        available_pairs = ["BTC", "SOL", "ETH", "TRUMP"]
    
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
    
    # Run analysis
    if run_analysis and selected_pair:
        progress_bar = st.progress(0, text="Starting analysis...")
        
        # Run analysis
        rankings = analyze_tiers(selected_pair, progress_bar)
        
        if rankings is not None and not rankings.empty:
            # Show results
            st.header("Global Tier Rankings")
            st.markdown("**Efficiency Formula:** Choppiness × (100% - Dropout Rate)")
            
            # Format for display
            display_df = rankings.copy()
            display_df['Exchange:Tier'] = display_df['exchange_tier']
            
            # Rename columns
            display_df = display_df.rename(columns={
                'efficiency': 'Efficiency Score',
                'choppiness': 'Choppiness',
                'dropout_rate': 'Dropout Rate (%)',
                'valid_points': 'Valid Points'
            })
            
            # Show table
            st.dataframe(
                display_df[['Exchange:Tier', 'Efficiency Score', 'Choppiness', 'Dropout Rate (%)', 'Valid Points']],
                use_container_width=True
            )
            
            # Show top recommendations
            st.header("Recommended Tiers")
            top_tiers = display_df.iloc[:3]
            
            st.markdown(f"""
            **Primary Tier:** {top_tiers.iloc[0]['Exchange:Tier']}  
            **Fallback Tier 1:** {top_tiers.iloc[1]['Exchange:Tier']}  
            **Fallback Tier 2:** {top_tiers.iloc[2]['Exchange:Tier']}
            """)
        else:
            st.error("No valid data found for analysis.")
    else:
        st.info("Select a pair and click ANALYZE NOW.")

if __name__ == "__main__":
    main()