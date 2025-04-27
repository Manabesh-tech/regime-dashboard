import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager

# Page configuration - absolute minimum for speed
st.set_page_config(
    page_title="Depth Tier Analyzer",
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
            table_name = f'oracle_order_book_level_price_data_partition_v4_{today}'
            
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
                table_name = f'oracle_order_book_level_price_data_partition_v4_{yesterday}'
                
                # Check if yesterday's table exists
                if not session.execute(check_table, {"table_name": table_name}).scalar():
                    return None
            
            # Use the exact SQL you provided
            query = text(f"""
            SELECT DISTINCT ON (p.pair_name)
              p.pair_name,
              TO_CHAR(p.utc8, 'YYYY-MM-DD HH24:MI:SS.MS') AS "UTC+8",
              p.all_bid,
              p.all_ask
            FROM (
              SELECT
                pair_name,
                (created_at + INTERVAL '8 hour') AS utc8,
                all_bid,
                all_ask
              FROM
                public."{table_name}"
              WHERE
                pair_name = :pair_name
            ) AS p
            ORDER BY
              p.pair_name ASC,
              p.utc8 DESC
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

# Simplified version of the depth tier analyzer
class SimplifiedDepthTierAnalyzer:
    """
    Simplified analyzer for liquidity depth tiers - raw metrics only, no scoring
    """

    def __init__(self):
        self.point_counts = [500, 5000, 10000, 50000]

        # Define depth tiers
        self.depth_tier_columns = [
            'price_1', 'price_2', 'price_3', 'price_4', 'price_5',
            'price_6', 'price_7', 'price_8', 'price_9', 'price_10',
            'price_11', 'price_12', 'price_13', 'price_14', 'price_15',
            'price_16','price_17','price_18','price_19','price_20','price_21',
            'price_22','price_23','price_24','price_25','price_26','price_27','price_28','price_29'
        ]

        # Map column names to actual depth values
        self.depth_tier_values = {
            'price_26':'1k',
            'price_27':'3k',
            'price_28':'5k',
            'price_29':'7k',
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
            'price_16':'5000k',
            'price_17':'6000k',
            'price_18':'7000k',
            'price_19': '8000k',
            'price_20':'9000k',
            'price_21':'10000k',
            'price_22':'11000k',
            'price_23':'12000k',
            'price_24':'13000k',
            'price_25':'14000k',

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

    def fetch_and_analyze(self, pair_name, hours=24, progress_bar=None, use_replication=True):
        """Fetch data and calculate metrics for each depth tier"""
        try:
            with get_session(use_replication=use_replication) as session:
                if not session:
                    return False
                
                # Get current day's partition table (most likely to have data)
                today = datetime.now().strftime("%Y%m%d")
                table_name = f"oracle_order_book_level_price_data_partition_v4_{today}"
                
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
                    table_name = f"oracle_order_book_level_price_data_partition_v4_{yesterday}"
                    
                    # Check if yesterday's table exists
                    if not session.execute(check_table, {"table_name": table_name}).scalar():
                        return False
                
                # Calculate time range
                start_time = datetime.now() - timedelta(hours=hours)
                start_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
                
                # Fetch all data at once with a single query for the max point count
                max_points = max(self.point_counts)
                
                query = text(f"""
                    SELECT
                        pair_name,
                        {', '.join(self.depth_tier_columns)}
                    FROM
                        public."{table_name}"
                    WHERE
                        pair_name = :pair_name
                        AND created_at >= :start_time
                    ORDER BY created_at DESC
                    LIMIT :limit
                """)
                
                # Execute query with parameters
                result = session.execute(
                    query,
                    {
                        "pair_name": pair_name,
                        "start_time": start_str,
                        "limit": max_points + 1000
                    }
                )
                
                # Fetch all rows and create DataFrame
                columns = ['pair_name'] + self.depth_tier_columns
                all_data = result.fetchall()
                
                if not all_data or len(all_data) < min(self.point_counts):
                    return False
                
                # Convert to DataFrame for faster processing
                all_df = pd.DataFrame(all_data, columns=columns)
                
                # Process each point count using the pre-fetched data
                for i, point_count in enumerate(self.point_counts):
                    if progress_bar:
                        progress_bar.progress((i / len(self.point_counts)) * 0.9 + 0.1, 
                                          text=f"Processing {point_count} points...")
                    
                    if len(all_df) >= point_count:
                        # Process each depth tier separately
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
                        
                        # Convert to DataFrame and sort by a primary metric
                        self.results[point_count] = self._create_results_table(tier_results)
                
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
            st.error(f"Error in analysis: {e}")
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
            return None

    def _create_results_table(self, tier_results):
        """Create a simple results table without scoring or normalization"""
        if not tier_results:
            return None

        # Create DataFrame directly
        data = []
        for tier, metrics in tier_results.items():
            row = {'Tier': tier}
            row.update(metrics)
            data.append(row)

        df = pd.DataFrame(data)

        # Sort by direction changes (higher is better) as the primary metric
        # You can change this to sort by another metric if you prefer
        if 'direction_changes' in df.columns:
            df = df.sort_values('direction_changes', ascending=False)
        else:
            # If direction_changes is not available, try another metric
            for metric in ['choppiness', 'tick_atr_pct']:
                if metric in df.columns:
                    df = df.sort_values(metric, ascending=False)
                    break

        return df

# Table-only display function - simplified
def create_point_count_table(analyzer, point_count):
    """Creates a clean, readable table of raw metrics without scoring"""
    if analyzer.results[point_count] is None:
        st.info(f"No data available for {point_count} points analysis.")
        return

    df = analyzer.results[point_count]

    # Make a clean copy for display
    display_df = df.copy()

    # Select only the columns we want to display
    display_columns = ['Tier', 'direction_changes', 'choppiness', 'tick_atr_pct', 'trend_strength']
    display_df = display_df[display_columns]

    # Rename columns for better display
    display_df = display_df.rename(columns={
        'direction_changes': 'Direction Changes (%)',
        'choppiness': 'Choppiness',
        'tick_atr_pct': 'Tick ATR %',
        'trend_strength': 'Trend Strength'
    })

    # Format numeric columns with appropriate decimal places
    for col in display_df.columns:
        if col != 'Tier':
            if col == 'Tick ATR %':
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

    # Show the full table with enhanced styling
    st.dataframe(
        display_df,
        use_container_width=True,
        height=min(800, 100 + (len(display_df) * 35))  # Adaptive height
    )

# Format number with commas (e.g., 1,234,567)
def format_number(num):
    """格式化数字显示，添加千位分隔符
    Args:
        num: 要格式化的数字
    Returns:
        str: 格式化后的字符串
    """
    if num is None:
        return "N/A"
    try:
        # Convert to int and format with commas
        return f"{int(float(num)):,}"
    except:
        return str(num)

def main():
    # Main layout - super streamlined
    st.markdown("<h1 style='text-align: center; font-size:28px; margin-bottom: 10px;'>Liquidity Depth Tier Analyzer</h1>", unsafe_allow_html=True)

    # Get available pairs from the database
    available_pairs = get_available_pairs()

    # Main selection area
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        selected_pair = st.selectbox(
            "Select Pair",
            available_pairs,
            index=0 if available_pairs else None
        )

    with col2:
        run_analysis = st.button("ANALYZE", use_container_width=True)

    # Main content
    if run_analysis and selected_pair:
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
        st.markdown("""
        **Metrics:** Direction Changes (%), Choppiness, Tick ATR %, and Trend Strength.
        Higher values of the first three metrics and lower values of Trend Strength typically indicate better trading conditions.
        """)

        # Set up tabs for results
        tabs = st.tabs(["500 POINTS", "5,000 POINTS", "10,000 POINTS", "50,000 POINTS"])

        # Create progress bar
        progress_bar = st.progress(0, text="Starting analysis...")

        # Initialize analyzer and run analysis
        analyzer = SimplifiedDepthTierAnalyzer()
        success = analyzer.fetch_and_analyze(selected_pair, 24, progress_bar)

        if success:
            # Display results for each point count
            with tabs[0]:
                create_point_count_table(analyzer, 500)

            with tabs[1]:
                create_point_count_table(analyzer, 5000)

            with tabs[2]:
                create_point_count_table(analyzer, 10000)

            with tabs[3]:
                create_point_count_table(analyzer, 50000)

        else:
            progress_bar.empty()
            st.error(f"Failed to analyze {selected_pair}. Please try another pair.")

    else:
        # Minimal welcome message
        st.info("Select a pair and click ANALYZE to find the optimal depth tier.")

if __name__ == "__main__":
    main()
