import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import pytz
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Page configuration
st.set_page_config(
    page_title="Market Spreads vs Volatility Analysis",
    layout="wide"
)

st.title("Enhanced Market Spreads vs Volatility Analysis")
st.markdown("Advanced analysis of the relationship between market spreads and volatility with improved estimation techniques")

# DB connection
try:
    db_config = st.secrets["database"]
    db_uri = (
        f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}"
        f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    )
    engine = create_engine(db_uri)
except Exception as e:
    st.error(f"Database connection error: {e}")
    db_uri = None  # For local testing without DB

# Fetch active pairs
@st.cache_data(ttl=600)
def fetch_active_pairs():
    """Fetch pairs that have been active in the last 30 days"""
    if db_uri is None:
        return ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "BNB/USDT"]
    
    query = """
    SELECT DISTINCT pair_name 
    FROM public.trade_fill_fresh 
    WHERE created_at > NOW() - INTERVAL '30 days'
    ORDER BY pair_name
    """
    
    try:
        df = pd.read_sql(query, engine)
        if df.empty:
            return ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "BNB/USDT"]
        return df['pair_name'].tolist()
    except Exception as e:
        st.error(f"Error fetching pairs: {e}")
        return ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "BNB/USDT"]

# Define timeframes for analysis
def get_timeframe_options():
    return {
        "Last 24 Hours": 1,
        "Last 3 Days": 3, 
        "Last 7 Days": 7,
        "Last 14 Days": 14,
        "Last 30 Days": 30
    }

# Fetch historical spread data with improved query
@st.cache_data(ttl=600)
def fetch_historical_spreads(pair_name, days=7):
    """Fetch historical spread data for a given pair with better filtering"""
    if db_uri is None:
        # Generate mock data for testing without DB
        dates = pd.date_range(end=datetime.now(), periods=24*days, freq='H')
        mock_data = {
            'hour': dates,
            'binanceFuture': np.random.uniform(0.0001, 0.001, len(dates)),
            'gateFuture': np.random.uniform(0.0001, 0.0015, len(dates)),
            'hyperliquidFuture': np.random.uniform(0.0001, 0.002, len(dates)),
            'avg_spread': np.random.uniform(0.0002, 0.0012, len(dates))
        }
        return pd.DataFrame(mock_data).set_index('hour')
    
    # Calculate time range
    end_time = datetime.now(pytz.utc)
    start_time = end_time - timedelta(days=days)
    
    query = f"""
    SELECT 
        time_group,
        source,
        fee1 as spread
    FROM 
        oracle_exchange_fee
    WHERE 
        time_group BETWEEN '{start_time}' AND '{end_time}'
        AND pair_name = '{pair_name}'
        AND source IN ('binanceFuture', 'gateFuture', 'hyperliquidFuture')
        AND fee1 > 0.00001  -- Filter out extremely low values that might be errors
        AND fee1 < 0.05     -- Filter out extremely high values that might be errors
    ORDER BY 
        time_group
    """
    
    try:
        df = pd.read_sql(query, engine)
        if df.empty:
            # Try alternative query if no data found
            alt_query = f"""
            SELECT 
                time_group,
                source,
                fee1 as spread
            FROM 
                oracle_exchange_fee
            WHERE 
                time_group BETWEEN '{start_time}' AND '{end_time}'
                AND pair_name = '{pair_name}'
                AND fee1 > 0.00001
                AND fee1 < 0.05
            ORDER BY 
                time_group
            """
            df = pd.read_sql(alt_query, engine)
            if df.empty:
                return None
            
        # Convert time_group to datetime
        df['timestamp'] = pd.to_datetime(df['time_group'])
        
        # Resample to hourly intervals
        df['hour'] = df['timestamp'].dt.floor('H')
        hourly_df = df.groupby(['hour', 'source']).agg({
            'spread': 'mean'
        }).reset_index()
        
        # Pivot to get one column per source
        pivot_df = hourly_df.pivot(index='hour', columns='source', values='spread')
        
        # Add average spread across all sources
        pivot_df['avg_spread'] = pivot_df.mean(axis=1)
        
        return pivot_df
    except Exception as e:
        st.error(f"Error fetching historical spreads for {pair_name}: {e}")
        return None

# Fetch historical prices with robust fallback methods
@st.cache_data(ttl=600)
def fetch_historical_prices(pair_name, days=7):
    """Fetch historical price data with multiple fallback methods to ensure data availability"""
    if db_uri is None:
        # Generate mock data for testing without DB
        dates = pd.date_range(end=datetime.now(), periods=24*days*12, freq='5min')
        price_base = 100 if "BTC" in pair_name else (3000 if "ETH" in pair_name else 50)
        price_volatility = 0.05 if "BTC" in pair_name else (0.07 if "ETH" in pair_name else 0.1)
        
        # Generate price movement with realistic patterns
        random_walk = np.cumprod(1 + np.random.normal(0, price_volatility/np.sqrt(12*24), len(dates)))
        prices = price_base * random_walk
        
        # Create mock data with first, last, high, low, etc.
        mock_data = {
            'interval': dates,
            'deal_price_first': prices,
            'deal_price_last': prices * (1 + np.random.normal(0, 0.001, len(dates))),
            'deal_price_mean': prices * (1 + np.random.normal(0, 0.0005, len(dates))),
            'deal_price_max': prices * (1 + np.random.uniform(0.001, 0.01, len(dates))),
            'deal_price_min': prices * (1 - np.random.uniform(0.001, 0.01, len(dates))),
            'deal_price_count': np.random.randint(10, 100, len(dates)),
            'deal_price_std': prices * np.random.uniform(0.0001, 0.005, len(dates))
        }
        return pd.DataFrame(mock_data)
    
    # Calculate time range
    end_time = datetime.now(pytz.utc)
    start_time = end_time - timedelta(days=days)
    
    # Try main query first - trade_fill_fresh
    query = f"""
    SELECT 
        created_at,
        deal_price
    FROM 
        public.trade_fill_fresh
    WHERE 
        created_at BETWEEN '{start_time}' AND '{end_time}'
        AND pair_name = '{pair_name}'
    ORDER BY 
        created_at
    """
    
    try:
        df = pd.read_sql(query, engine)
        
        # If not enough data, try with a wider time range
        if len(df) < 100:
            wider_start_time = end_time - timedelta(days=days*2)
            wider_query = f"""
            SELECT 
                created_at,
                deal_price
            FROM 
                public.trade_fill_fresh
            WHERE 
                created_at BETWEEN '{wider_start_time}' AND '{end_time}'
                AND pair_name = '{pair_name}'
            ORDER BY 
                created_at
            """
            df = pd.read_sql(wider_query, engine)
        
        # If still not enough data, try oracle_pricing
        if len(df) < 100:
            alt_query = f"""
            SELECT 
                time_group as created_at,
                price as deal_price
            FROM 
                oracle_pricing
            WHERE 
                time_group BETWEEN '{start_time}' AND '{end_time}'
                AND pair_name = '{pair_name}'
            ORDER BY 
                time_group
            """
            oracle_df = pd.read_sql(alt_query, engine)
            
            # Combine if we got data from both sources
            if not oracle_df.empty:
                if df.empty:
                    df = oracle_df
                else:
                    df = pd.concat([df, oracle_df]).drop_duplicates(subset=['created_at'])
        
        # If still not enough data, try one more source
        if len(df) < 100:
            try:
                # Try market_tick data
                market_query = f"""
                SELECT 
                    created_at,
                    last_price as deal_price
                FROM 
                    market_tick
                WHERE 
                    created_at BETWEEN '{start_time}' AND '{end_time}'
                    AND pair_name = '{pair_name}'
                ORDER BY 
                    created_at
                """
                market_df = pd.read_sql(market_query, engine)
                
                if not market_df.empty:
                    if df.empty:
                        df = market_df
                    else:
                        df = pd.concat([df, market_df]).drop_duplicates(subset=['created_at'])
            except:
                pass
                
        # Try one last source - order books
        if len(df) < 100:
            try:
                # Get data from order books (average of bid/ask)
                order_query = f"""
                SELECT 
                    created_at,
                    (best_bid_price + best_ask_price) / 2 as deal_price
                FROM 
                    order_book_snapshot
                WHERE 
                    created_at BETWEEN '{start_time}' AND '{end_time}'
                    AND pair_name = '{pair_name}'
                ORDER BY 
                    created_at
                """
                order_df = pd.read_sql(order_query, engine)
                
                if not order_df.empty:
                    if df.empty:
                        df = order_df
                    else:
                        df = pd.concat([df, order_df]).drop_duplicates(subset=['created_at'])
            except:
                pass
                
        if df.empty:
            return None
            
        # Convert created_at to datetime
        df['timestamp'] = pd.to_datetime(df['created_at'])
        
        # Detect and filter outliers (prices that are more than 3 std devs from mean)
        if len(df) > 5:
            mean_price = df['deal_price'].mean()
            std_price = df['deal_price'].std()
            if std_price > 0:  # Avoid division by zero
                lower_bound = mean_price - (3 * std_price)
                upper_bound = mean_price + (3 * std_price)
                df = df[(df['deal_price'] > lower_bound) & (df['deal_price'] < upper_bound)]
        
        # Use 5-minute intervals for more granular analysis
        df['interval'] = df['timestamp'].dt.floor('5min')
        
        # Group by interval to get aggregated price data
        interval_df = df.groupby('interval').agg({
            'deal_price': ['first', 'last', 'mean', 'max', 'min', 'count']
        })
        
        # Flatten multi-index columns
        interval_df.columns = ['_'.join(col).strip() for col in interval_df.columns.values]
        interval_df = interval_df.reset_index()
        
        # Add a standard deviation column
        if len(df) > 1:
            std_dev = []
            for interval in interval_df['interval']:
                interval_data = df[df['interval'] == interval]['deal_price']
                if len(interval_data) > 1:
                    std_dev.append(interval_data.std())
                else:
                    std_dev.append(0)
            interval_df['deal_price_std'] = std_dev
        else:
            interval_df['deal_price_std'] = 0
            
        # Ensure we have enough intervals
        if len(interval_df) < 5:
            return None
            
        return interval_df
    except Exception as e:
        st.error(f"Error fetching historical prices for {pair_name}: {e}")
        return None

def calculate_volatility(price_df, window_minutes=60):
    """Calculate rolling volatility with multiple timeframes and methods for more accurate measurement"""
    if price_df is None or len(price_df) < 10:
        return None
    
    # Create a copy to avoid warnings
    vol_df = price_df.copy()
    
    # Sort by time to ensure correct calculations
    vol_df = vol_df.sort_values('interval')
    
    # Calculate returns (with sanity check to avoid division by zero or negative prices)
    vol_df['prev_price'] = vol_df['deal_price_last'].shift(1)
    vol_df = vol_df[vol_df['prev_price'] > 0]  # Filter out zero or negative previous prices
    
    # Use log returns for better statistical properties
    vol_df['log_return'] = np.log(vol_df['deal_price_last'] / vol_df['prev_price'])
    vol_df['pct_return'] = (vol_df['deal_price_last'] / vol_df['prev_price']) - 1
    
    # Add interval-specific volatility (using price std dev within interval)
    vol_df['interval_volatility'] = vol_df['deal_price_std'] / vol_df['deal_price_mean']
    
    # Filter out extreme returns (more than 5 standard deviations)
    if len(vol_df) > 5:  # Need enough data for meaningful statistics
        mean_return = vol_df['log_return'].mean()
        std_return = vol_df['log_return'].std()
        if std_return > 0:  # Avoid division by zero
            lower_bound = mean_return - (5 * std_return)
            upper_bound = mean_return + (5 * std_return)
            vol_df = vol_df[(vol_df['log_return'] > lower_bound) & (vol_df['log_return'] < upper_bound)]
    
    # Calculate multiple timeframe volatility measures
    timeframes = [
        # Very short-term (intraday)
        ('volatility_vshort', 3),       # ~15 min
        # Short-term
        ('volatility_short', 6),        # ~30 min
        # Medium-term
        ('volatility_medium', 12),      # ~1 hour
        # Standard (original implementation)
        ('volatility_std', max(2, int(window_minutes / 5))),
        # Long-term
        ('volatility_long', 72),        # ~6 hours
        # Very long-term
        ('volatility_vlong', 144)       # ~12 hours
    ]
    
    # Add Parkinson's volatility (uses high-low range)
    if 'deal_price_max' in vol_df.columns and 'deal_price_min' in vol_df.columns:
        vol_df['log_high_low'] = np.log(vol_df['deal_price_max'] / vol_df['deal_price_min'])
        vol_df['parkinson_vol'] = np.sqrt(vol_df['log_high_low'].rolling(window=12).mean() / (4 * np.log(2))) * np.sqrt(12 * 24 * 365)
    
    # Add Garman-Klass volatility if we have open/close/high/low
    if 'deal_price_first' in vol_df.columns:
        vol_df['log_close_open'] = np.log(vol_df['deal_price_last'] / vol_df['deal_price_first'])
        if 'deal_price_max' in vol_df.columns and 'deal_price_min' in vol_df.columns:
            vol_df['garman_klass_term1'] = 0.5 * np.power(np.log(vol_df['deal_price_max'] / vol_df['deal_price_min']), 2)
            vol_df['garman_klass_term2'] = (2*np.log(2) - 1) * np.power(vol_df['log_close_open'], 2)
            vol_df['garman_klass_vol'] = np.sqrt(vol_df[['garman_klass_term1', 'garman_klass_term2']].sum(axis=1).rolling(window=12).mean()) * np.sqrt(12 * 24 * 365)
    
    # Calculate each volatility measure with its specific timeframe
    for vol_name, window_size in timeframes:
        # Ensure window size is reasonable
        actual_window = min(window_size, max(2, len(vol_df) // 3))
        
        # Calculate standard deviation-based volatility (annualized)
        vol_df[vol_name] = vol_df['log_return'].rolling(window=actual_window).std() * np.sqrt(12 * 24 * 365)
    
    # Add EWMA volatility with different decay factors
    spans = [
        ('volatility_ewma_fast', 6),     # Fast decay (recent points matter more)
        ('volatility_ewma_medium', 12),  # Medium decay
        ('volatility_ewma_slow', 24)     # Slow decay (longer history matters)
    ]
    
    for vol_name, span in spans:
        vol_df[vol_name] = vol_df['log_return'].ewm(span=span).std() * np.sqrt(12 * 24 * 365)
    
    # Add Yang-Zhang volatility estimator if we have the required data
    # This is considered one of the best estimators for financial volatility
    if all(col in vol_df.columns for col in ['deal_price_first', 'deal_price_last', 'deal_price_max', 'deal_price_min']):
        # Overnight volatility
        vol_df['log_open_close_prev'] = np.log(vol_df['deal_price_first'] / vol_df['deal_price_last'].shift(1))
        vol_df['overnight_vol'] = vol_df['log_open_close_prev'].rolling(window=12).var()
        
        # Open-to-close volatility
        vol_df['log_open_close'] = np.log(vol_df['deal_price_last'] / vol_df['deal_price_first'])
        vol_df['open_close_vol'] = vol_df['log_open_close'].rolling(window=12).var()
        
        # Rogers-Satchell volatility
        vol_df['log_high_open'] = np.log(vol_df['deal_price_max'] / vol_df['deal_price_first'])
        vol_df['log_high_close'] = np.log(vol_df['deal_price_max'] / vol_df['deal_price_last'])
        vol_df['log_low_open'] = np.log(vol_df['deal_price_min'] / vol_df['deal_price_first'])
        vol_df['log_low_close'] = np.log(vol_df['deal_price_min'] / vol_df['deal_price_last'])
        
        vol_df['rs_vol'] = vol_df['log_high_open'] * vol_df['log_high_close'] + vol_df['log_low_open'] * vol_df['log_low_close']
        vol_df['rs_vol'] = vol_df['rs_vol'].rolling(window=12).mean()
        
        # Calculate Yang-Zhang volatility
        k = 0.34 / (1.34 + (12 + 1) / (12 - 1))
        vol_df['yang_zhang_vol'] = vol_df['overnight_vol'] + k * vol_df['open_close_vol'] + (1 - k) * vol_df['rs_vol']
        vol_df['yang_zhang_vol'] = np.sqrt(vol_df['yang_zhang_vol']) * np.sqrt(12 * 24 * 365)
    
    # Handle missing values without using inplace=True
    vol_columns = [col for col in vol_df.columns if 'volatility' in col or 'vol' in col]
    for col in vol_columns:
        if col in vol_df.columns:
            vol_df[col] = vol_df[col].bfill().ffill()
    
    # Add timestamps in different formats for easier joining
    vol_df['hour'] = vol_df['interval'].dt.floor('H')
    vol_df['minute'] = vol_df['interval'].dt.floor('min')
    
    return vol_df

def combine_spread_volatility_data(spread_df, vol_df):
    """Combine spread and volatility data with improved matching logic"""
    if spread_df is None or vol_df is None:
        return None
    
    # Ensure the indices are aligned
    spread_df = spread_df.reset_index()
    vol_df = vol_df.reset_index()
    
    # Create hourly timestamps for joining
    if 'hour' not in spread_df.columns and 'hour' in spread_df.index.names:
        spread_df['hour'] = spread_df.index.get_level_values('hour')
    elif 'hour' not in spread_df.columns:
        spread_df['hour'] = pd.to_datetime(spread_df['hour'])
    
    # Get all volatility columns
    vol_cols = [col for col in vol_df.columns if 'volatility' in col or 'vol' in col]
    
    # Create an hourly grouped volatility dataframe
    agg_dict = {col: 'mean' for col in vol_cols}
    agg_dict.update({
        'log_return': ['mean', 'std'],
        'deal_price_mean': 'mean'
    })
    
    hourly_vol_df = vol_df.groupby('hour').agg(agg_dict)
    
    # Flatten multi-index columns if needed
    if isinstance(hourly_vol_df.columns, pd.MultiIndex):
        hourly_vol_df.columns = ['_'.join(col).strip() for col in hourly_vol_df.columns.values]
    
    hourly_vol_df = hourly_vol_df.reset_index()
    
    # Merge on the hour
    merged_df = pd.merge(
        spread_df,
        hourly_vol_df,
        left_on='hour',
        right_on='hour',
        how='inner'
    )
    
    # If we don't have enough data points after merging, try a more flexible approach
    if len(merged_df) < 5:
        # Convert times to unix timestamps for proximity matching
        spread_times = pd.to_datetime(spread_df['hour']).astype(int) // 10**9
        vol_times = pd.to_datetime(hourly_vol_df['hour']).astype(int) // 10**9
        
        # Create a proximity matrix
        proximity_pairs = []
        for s_idx, s_time in enumerate(spread_times):
            for v_idx, v_time in enumerate(vol_times):
                time_diff = abs(s_time - v_time)
                if time_diff < 3600:  # Within 1 hour
                    proximity_pairs.append((s_idx, v_idx, time_diff))
        
        # Sort by time difference
        proximity_pairs.sort(key=lambda x: x[2])
        
        # Create merged dataframe from closest matches
        matched_data = []
        used_s = set()
        used_v = set()
        
        for s_idx, v_idx, _ in proximity_pairs:
            if s_idx not in used_s and v_idx not in used_v:
                s_row = spread_df.iloc[s_idx].to_dict()
                v_row = hourly_vol_df.iloc[v_idx].to_dict()
                combined_row = {**s_row, **v_row}
                matched_data.append(combined_row)
                used_s.add(s_idx)
                used_v.add(v_idx)
        
        if matched_data:
            merged_df = pd.DataFrame(matched_data)
    
    return merged_df

def get_vol_timeframe(vol_name):
    """Return the approximate timeframe for a volatility measure based on its name"""
    if 'vshort' in vol_name:
        return "~15 min"
    elif 'short' in vol_name:
        return "~30 min"
    elif 'medium' in vol_name:
        return "~1 hour"
    elif 'std' in vol_name:
        return "~1 hour"
    elif 'long' in vol_name:
        return "~6 hours"
    elif 'vlong' in vol_name:
        return "~12 hours"
    elif 'ewma_fast' in vol_name:
        return "Recent weighted"
    elif 'ewma_medium' in vol_name:
        return "Medium weighted"
    elif 'ewma_slow' in vol_name:
        return "Long weighted"
    elif 'parkinson' in vol_name:
        return "Range-based"
    elif 'garman_klass' in vol_name:
        return "OHLC-based"
    elif 'yang_zhang' in vol_name:
        return "Advanced OHLC"
    else:
        return "Unknown"

def analyze_relationship(combined_df):
    """Analyze the relationship between spreads and volatility with more detailed diagnostics"""
    if combined_df is None or len(combined_df) < 10:
        return None
    
    # Find all volatility measures
    vol_cols = [col for col in combined_df.columns if 'volatility' in col or 'vol' in col]
    
    # Make sure we have at least one volatility column
    if not vol_cols:
        return None
    
    # Create correlation matrix with all volatility measures
    corr_columns = ['avg_spread'] + vol_cols
    
    # Make sure all required columns exist
    available_columns = [col for col in corr_columns if col in combined_df.columns]
    
    if len(available_columns) < 2:  # Need at least spread and one vol measure
        return None
    
    # Drop rows with NaN values for correlation calculation
    clean_df_corr = combined_df[available_columns].dropna()
    if len(clean_df_corr) < 5:  # Need at least a few data points
        st.warning("Not enough valid data points for correlation analysis")
        return None
        
    correlation = clean_df_corr.corr()
    
    # Determine which volatility measure has the strongest correlation with spread
    spread_correlations = correlation.loc['avg_spread', vol_cols]
    
    # Get both strongest positive and negative correlations
    positive_corr = spread_correlations[spread_correlations > 0]
    negative_corr = spread_correlations[spread_correlations < 0]
    
    # Find strongest correlations in each direction
    best_positive_col = None if positive_corr.empty else positive_corr.idxmax()
    best_positive_corr = None if positive_corr.empty else positive_corr.max()
    
    best_negative_col = None if negative_corr.empty else negative_corr.idxmin()
    best_negative_corr = None if negative_corr.empty else negative_corr.min()
    
    # Determine which correlation is stronger (by absolute value)
    if best_positive_corr is None and best_negative_corr is None:
        # No correlations at all
        return None
    elif best_positive_corr is None:
        # Only negative correlations
        best_vol_col = best_negative_col
        best_corr = best_negative_corr
    elif best_negative_corr is None:
        # Only positive correlations
        best_vol_col = best_positive_col
        best_corr = best_positive_corr
    else:
        # Both exist, choose the stronger one by absolute value
        if abs(best_positive_corr) >= abs(best_negative_corr):
            best_vol_col = best_positive_col
            best_corr = best_positive_corr
        else:
            best_vol_col = best_negative_col
            best_corr = best_negative_corr
    
    # Store correlation data for all volatility measures
    all_correlations = []
    for col in vol_cols:
        corr_value = correlation.loc['avg_spread', col]
        all_correlations.append({
            'measure': col,
            'correlation': corr_value,
            'abs_correlation': abs(corr_value),
            'timeframe': get_vol_timeframe(col)
        })
    
    all_correlations_df = pd.DataFrame(all_correlations)
    if not all_correlations_df.empty:
        all_correlations_df = all_correlations_df.sort_values('abs_correlation', ascending=False)
    
    # For regression, use the volatility measure with the strongest correlation
    reg_df = combined_df[['avg_spread', best_vol_col]].dropna()
    
    if len(reg_df) < 5:  # Need at least a few data points
        st.warning("Not enough valid data points for regression analysis")
        return {
            'correlation': correlation,
            'best_volatility_measure': best_vol_col,
            'best_volatility_correlation': best_corr,
            'all_correlations': all_correlations_df,
            'correlation_direction': 'positive' if best_corr > 0 else 'negative',
            'regression_coefficient': None,
            'r2': None,
            'lead_lag': None,
            'best_lag': None
        }
    
    # Simple linear regression with cleaned data
    X = reg_df[best_vol_col].values.reshape(-1, 1)
    y = reg_df['avg_spread'].values
    
    try:
        model = LinearRegression()
        model.fit(X, y)
        
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        
        # Calculate lead-lag relationship (which one leads?)
        leads_lags = []
        for lag in range(-12, 13, 1):  # -12 to +12 hours
            if lag < 0:
                # Spread leads volatility
                spread_shifted = combined_df['avg_spread'].shift(-lag)
                lag_corr = spread_shifted.corr(combined_df[best_vol_col])
                leads_lags.append({
                    'lag': lag,
                    'correlation': lag_corr,
                    'description': f"Spread leads volatility by {-lag} hours"
                })
            else:
                # Volatility leads spread
                vol_shifted = combined_df[best_vol_col].shift(lag)
                lag_corr = combined_df['avg_spread'].corr(vol_shifted)
                leads_lags.append({
                    'lag': lag,
                    'correlation': lag_corr,
                    'description': f"Volatility leads spread by {lag} hours"
                })
        
        leads_lags_df = pd.DataFrame(leads_lags)
        
        # Filter out NaN correlation values
        leads_lags_df = leads_lags_df.dropna(subset=['correlation'])
        
        if leads_lags_df.empty:
            best_lag = None
        else:
            # Find the lag with the highest correlation
            best_lag_idx = leads_lags_df['correlation'].abs().idxmax()
            best_lag = leads_lags_df.loc[best_lag_idx]
        
        return {
            'correlation': correlation,
            'best_volatility_measure': best_vol_col,
            'best_volatility_correlation': best_corr,
            'all_correlations': all_correlations_df,
            'correlation_direction': 'positive' if best_corr > 0 else 'negative',
            'regression_coefficient': model.coef_[0],
            'r2': r2,
            'lead_lag': leads_lags_df,
            'best_lag': best_lag
        }
    
    except Exception as e:
        st.error(f"Error in regression analysis: {e}")
        return {
            'correlation': correlation,
            'best_volatility_measure': best_vol_col,
            'best_volatility_correlation': best_corr,
            'all_correlations': all_correlations_df,
            'correlation_direction': 'positive' if best_corr > 0 else 'negative',
            'regression_coefficient': None,
            'r2': None,
            'lead_lag': None,
            'best_lag': None
        }

def display_all_correlations(results, pair_name):
    """Display detailed correlation analysis between spread and all volatility measures"""
    if results is None or 'all_correlations' not in results or results['all_correlations'] is None:
        st.warning("No correlation data available to display")
        return
        
    all_corrs = results['all_correlations']
    
    st.markdown("### Correlation Analysis by Volatility Timeframe")
    st.markdown(f"Showing all correlations between spread and volatility measures for {pair_name}")
    
    # Format for display
    display_df = all_corrs.copy()
    display_df = display_df.rename(columns={
        'measure': 'Volatility Measure',
        'correlation': 'Correlation', 
        'abs_correlation': 'Abs. Correlation',
        'timeframe': 'Timeframe'
    })
    
    # Format the measure names
    display_df['Volatility Measure'] = display_df['Volatility Measure'].apply(
        lambda x: x.replace('_', ' ').title()
    )
    
    st.dataframe(display_df)
    
    # Create a correlation bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by timeframe for better visualization
    plot_df = all_corrs.sort_values('measure')
    
    # Use a more visually appealing style with color coding
    bars = ax.bar(plot_df['measure'].apply(lambda x: x.replace('volatility_', 'vol_')), 
                 plot_df['correlation'], 
                 color=[('g' if c >= 0 else 'r') for c in plot_df['correlation']])
    
    # Add a horizontal line at y=0
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Highlight the best correlation
    best_idx = plot_df['abs_correlation'].idxmax()
    if not pd.isna(best_idx):
        best_measure = plot_df.loc[best_idx, 'measure']
        best_index = plot_df['measure'].tolist().index(best_measure)
        bars[best_index].set_color('blue')
        bars[best_index].set_edgecolor('black')
        bars[best_index].set_linewidth(1.5)
    
    # Format x-axis labels
    plt.xticks(rotation=45, ha='right')
    plt.ylim(-1, 1)
    
    # Add labels and title
    ax.set_xlabel('Volatility Measure')
    ax.set_ylabel('Correlation with Spread')
    ax.set_title(f'Spread-Volatility Correlation by Measure for {pair_name}')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Add interpretation
    st.markdown("### Interpretation")
    
    best_measure = all_corrs.iloc[0]['measure'].replace('_', ' ').title()
    best_corr = all_corrs.iloc[0]['correlation']
    best_timeframe = all_corrs.iloc[0]['timeframe']
    
    if abs(best_corr) > 0.7:
        strength = "very strong"
    elif abs(best_corr) > 0.5:
        strength = "strong"
    elif abs(best_corr) > 0.3:
        strength = "moderate"
    else:
        strength = "weak"
    
    direction = "positive" if best_corr > 0 else "negative"
    
    st.markdown(f"""
    - **Best Volatility Measure**: {best_measure} ({best_timeframe})
    - **Correlation Strength**: {strength} {direction} correlation ({best_corr:.3f})
    
    #### What this means:
    
    A {direction} correlation means that as volatility increases, spreads tend to {"increase" if direction == "positive" else "decrease"}.
    """)
    
    if direction == "negative":
        st.markdown("""
        **Negative correlation is unusual but can occur when:**
        
        1. **Market makers are very active**: They might tighten spreads during high volatility to capture more trading volume
        2. **Exchange incentives**: Some exchanges offer rebates for market makers who maintain tight spreads during volatile periods
        3. **Time lag effects**: There might be a delay between volatility changes and spread responses
        4. **Volatility calculation method**: Different volatility measures capture different aspects of price movement
        
        Try examining the lead-lag relationship to see if volatility changes precede spread changes by several hours.
        """)

def plot_spread_volatility(combined_df, pair_name):
    """Plot spread vs. volatility relationship with more compact layout"""
    if combined_df is None or len(combined_df) < 5:
        st.warning(f"Not enough data for {pair_name} to create plots")
        return
    
    # Identify which volatility measures are available
    vol_cols = [col for col in combined_df.columns if 'volatility' in col or 'vol' in col]
    if not vol_cols:
        st.warning(f"No volatility measures available for {pair_name}")
        return
    
    # Find the volatility measure with strongest correlation to spread
    correlations = {}
    for col in vol_cols:
        corr = combined_df['avg_spread'].corr(combined_df[col])
        if not pd.isna(corr):
            correlations[col] = corr
    
    if not correlations:
        st.warning(f"Cannot calculate correlations for {pair_name}")
        return
        
    # Use the volatility measure with the strongest correlation
    best_vol_col = max(correlations.items(), key=lambda x: abs(x[1]))[0]
    best_corr = correlations[best_vol_col]
    
    # Create a copy and ensure datetime format for x-axis
    plot_df = combined_df.copy()
    plot_df['hour'] = pd.to_datetime(plot_df['hour'])
    
    # Filter outliers for better visualization
    q1_spread = plot_df['avg_spread'].quantile(0.05)
    q3_spread = plot_df['avg_spread'].quantile(0.95)
    iqr_spread = q3_spread - q1_spread
    lower_bound = q1_spread - (1.5 * iqr_spread)
    upper_bound = q3_spread + (1.5 * iqr_spread)
    plot_df = plot_df[(plot_df['avg_spread'] >= lower_bound) & (plot_df['avg_spread'] <= upper_bound)]
    
    # Same for volatility
    q1_vol = plot_df[best_vol_col].quantile(0.05)
    q3_vol = plot_df[best_vol_col].quantile(0.95)
    iqr_vol = q3_vol - q1_vol
    lower_bound = q1_vol - (1.5 * iqr_vol)
    upper_bound = q3_vol + (1.5 * iqr_vol)
    plot_df = plot_df[(plot_df[best_vol_col] >= lower_bound) & (plot_df[best_vol_col] <= upper_bound)]
    
    # Sort by time for proper time series display
    plot_df = plot_df.sort_values('hour')
    
    # Create a more compact figure - reduced height and simplified layout
    # Using just 2 panels instead of 3, with smaller size
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), dpi=100)
    
    # First subplot - Time series
    measure_name = best_vol_col.replace('_', ' ').title()
    ax1.set_title(f"Spread vs. {measure_name} for {pair_name}", 
                 fontsize=12, fontweight='bold')
    
    # Plot spread on left y-axis
    spread_line = ax1.plot(plot_df['hour'], plot_df['avg_spread'] * 10000, 'b-', 
                          label='Spread (bps)', linewidth=2)
    ax1.set_ylabel('Spread (basis points)', fontsize=10)
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Create twin axis for volatility
    ax1_twin = ax1.twinx()
    vol_line = ax1_twin.plot(plot_df['hour'], plot_df[best_vol_col] * 100, 'r-', 
                           label=f'{measure_name} (%)', linewidth=2)
    ax1_twin.set_ylabel('Annualized Volatility (%)', fontsize=10, color='r')
    ax1_twin.tick_params(axis='y', labelcolor='r')
    
    # Combine legends
    lines = spread_line + vol_line
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', fontsize=9)
    
    # Format x-axis to show dates nicely
    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Second subplot - Scatter plot with best volatility measure
    clean_df = plot_df[['avg_spread', best_vol_col]].dropna()
    
    if len(clean_df) < 5:
        ax2.text(0.5, 0.5, "Not enough data points for regression analysis", 
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax2.transAxes, fontsize=10)
    else:
        # Scatter plot with cleaned data
        ax2.scatter(clean_df[best_vol_col] * 100, clean_df['avg_spread'] * 10000, alpha=0.6, s=30)
        ax2.set_xlabel('Annualized Volatility (%)', fontsize=10)
        ax2.set_ylabel('Spread (basis points)', fontsize=10)
        ax2.set_title(f'Spread vs. Volatility Correlation', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(labelsize=8)
        
        # Add regression line using clean data
        X = clean_df[best_vol_col].values.reshape(-1, 1)
        y = clean_df['avg_spread'].values
        
        try:
            model = LinearRegression()
            model.fit(X, y)
            
            x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
            y_pred = model.predict(x_range)
            
            ax2.plot(x_range * 100, y_pred * 10000, 'r-', linewidth=1.5,
                     label=f'y = {model.coef_[0]*100:.4f}x + {model.intercept_*10000:.4f}')
            ax2.legend(fontsize=8)
            
            # Add correlation coefficient
            ax2.annotate(f"Correlation: {best_corr:.4f}", xy=(0.02, 0.95), xycoords='axes fraction',
                         fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        except Exception as e:
            ax2.text(0.5, 0.5, f"Error in regression: {str(e)}", 
                     horizontalalignment='center', verticalalignment='center',
                     transform=ax2.transAxes, fontsize=8)
    
    plt.tight_layout()
    return fig

def plot_lead_lag(lead_lag_df, pair_name):
    """Plot lead-lag relationship with more compact visualization"""
    if lead_lag_df is None or lead_lag_df.empty:
        return None
    
    # Filter out NaN correlations
    clean_df = lead_lag_df.dropna(subset=['correlation']).copy()
    
    if clean_df.empty or len(clean_df) < 3:
        return None
    
    fig, ax = plt.subplots(figsize=(7, 4), dpi=100)
    
    # Use a more visually appealing style
    bars = ax.bar(clean_df['lag'], clean_df['correlation'], width=0.7, 
             color=[('g' if c >= 0 else 'r') for c in clean_df['correlation']])
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Add vertical line at x=0
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Find best lag (highest absolute correlation)
    best_lag_idx = clean_df['correlation'].abs().idxmax()
    best_lag = clean_df.loc[best_lag_idx]
    
    # Highlight the best lag bar
    if not pd.isna(best_lag_idx):
        bar_index = clean_df.index.get_loc(best_lag_idx)
        bars[bar_index].set_color('blue')
        bars[bar_index].set_edgecolor('black')
        bars[bar_index].set_linewidth(1.5)
    
    # Add best lag annotation
    if not pd.isna(best_lag['lag']):
        ax.annotate(f"Best lag: {best_lag['lag']} hours\nCorr: {best_lag['correlation']:.3f}",
                    xy=(best_lag['lag'], best_lag['correlation']),
                    xytext=(0, 20 if best_lag['correlation'] > 0 else -20),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue'),
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3),
                    fontsize=9)
    
    # Improve axis labels and title
    ax.set_xlabel('Lag (hours)', fontsize=10)
    ax.set_ylabel('Correlation', fontsize=10)
    ax.set_title(f'Lead-Lag Relationship for {pair_name}', fontsize=11)
    ax.tick_params(labelsize=8)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3)
    
    # Add interpretation text based on the best lag
    if best_lag['lag'] < -1:
        interpretation = f"Spread changes lead volatility changes by {-best_lag['lag']} hours"
    elif best_lag['lag'] > 1:
        interpretation = f"Volatility changes lead spread changes by {best_lag['lag']} hours"
    else:
        interpretation = "Spread and volatility changes occur nearly simultaneously"
    
    ax.text(0.02, 0.02, interpretation, transform=ax.transAxes, fontsize=9,
           bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

def explain_lead_lag_concept():
    """Provide a clear explanation of what 'best lag' means"""
    with st.expander("What is Lead-Lag Analysis?"):
        st.markdown("""
        ### Understanding Lead-Lag Analysis
        
        Lead-lag analysis helps determine **which metric changes first** - spreads or volatility:
        
        - **Negative Lag (e.g., -3 hours)**: Spread changes typically precede volatility changes by the specified number of hours
        - **Positive Lag (e.g., +2 hours)**: Volatility changes typically precede spread changes by the specified number of hours
        - **Near Zero Lag**: Changes in both metrics occur nearly simultaneously
        
        ### Why This Matters
        
        Understanding lead-lag relationships helps you:
        
        1. **Predict Changes**: If one metric consistently leads the other, you can anticipate changes
        2. **Optimize Timing**: Adjust buffer rates proactively rather than reactively
        3. **Choose Metrics**: Base your buffer strategy on the leading indicator for better results
        
        ### How It's Calculated
        
        We shift one time series relative to the other and measure correlation at each lag.
        The lag with the strongest correlation is considered the "best lag."
        """)

def process_all_pairs(selected_pairs, days, vol_window, min_data_points):
    """Process all selected pairs with better error handling and reporting"""
    results_dict = {}
    successful_pairs = []
    failed_pairs = []
    data_issues = []
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, pair in enumerate(selected_pairs):
        # Update progress
        progress = (i + 1) / len(selected_pairs)
        progress_bar.progress(progress)
        status_text.text(f"Processing {pair} ({i+1}/{len(selected_pairs)})")
        
        try:
            # Get historical spread data
            spread_df = fetch_historical_spreads(pair, days=days)
            
            # Get historical price data for volatility
            price_df = fetch_historical_prices(pair, days=days)
            
            spread_status = "✓" if spread_df is not None and not spread_df.empty else "✗"
            price_status = "✓" if price_df is not None and not price_df.empty else "✗"
            
            if price_df is not None and not price_df.empty:
                vol_df = calculate_volatility(price_df, window_minutes=vol_window)
                vol_status = "✓" if vol_df is not None and not vol_df.empty else "✗"
            else:
                vol_df = None
                vol_status = "✗"
            
            # Combine data
            if spread_df is not None and vol_df is not None:
                combined_df = combine_spread_volatility_data(spread_df, vol_df)
                combined_status = "✓" if combined_df is not None and len(combined_df) >= min_data_points else "✗"
                
                if combined_df is not None and len(combined_df) >= min_data_points:
                    # Analyze relationship
                    results = analyze_relationship(combined_df)
                    if results is not None:
                        results_dict[pair] = results
                        successful_pairs.append(pair)
                    else:
                        failed_pairs.append(pair)
                        data_issues.append({
                            'pair': pair,
                            'spread_data': spread_status,
                            'price_data': price_status,
                            'volatility_calc': vol_status,
                            'combined_data': combined_status,
                            'issue': "Analysis failed"
                        })
                else:
                    failed_pairs.append(pair)
                    data_issues.append({
                        'pair': pair,
                        'spread_data': spread_status,
                        'price_data': price_status,
                        'volatility_calc': vol_status,
                        'combined_data': combined_status,
                        'issue': f"Not enough matched data points ({0 if combined_df is None else len(combined_df)}, need {min_data_points})"
                    })
            else:
                failed_pairs.append(pair)
                combined_status = "✗"
                data_issues.append({
                    'pair': pair,
                    'spread_data': spread_status,
                    'price_data': price_status,
                    'volatility_calc': vol_status,
                    'combined_data': combined_status,
                    'issue': "Missing spread or volatility data"
                })
        except Exception as e:
            failed_pairs.append(pair)
            data_issues.append({
                'pair': pair,
                'spread_data': "?",
                'price_data': "?",
                'volatility_calc': "?",
                'combined_data': "?",
                'issue': f"Error: {str(e)}"
            })
    
    # Clear progress display
    progress_bar.empty()
    status_text.empty()
    
    return results_dict, successful_pairs, failed_pairs, data_issues

def render_individual_analysis_tab(selected_pairs, days, vol_window, min_data_points):
    """Render the individual pair analysis tab with improved layout and all pairs"""
    st.markdown("## Individual Pair Analysis")
    st.markdown(f"Analyzing the relationship between market spreads and volatility over {days} days")
    
    # Process all pairs
    results_dict, successful_pairs, failed_pairs, data_issues = process_all_pairs(
        selected_pairs, days, vol_window, min_data_points
    )
    
    # Display processing summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Pairs", len(selected_pairs))
    with col2:
        st.metric("Successful Analyses", len(successful_pairs))
    with col3:
        st.metric("Failed Analyses", len(failed_pairs))
    
    # Add lead-lag explanation
    explain_lead_lag_concept()
    
    # Create tabs for successful and failed pairs
    results_tabs = st.tabs(["Successful Pairs", "Failed Pairs"])
    
    with results_tabs[0]:
        if successful_pairs:
            # Create a selection widget for successful pairs
            pair_to_view = st.selectbox("Select a pair to view analysis", successful_pairs)
            
            if pair_to_view:
                st.markdown(f"### {pair_to_view}")
                
                # Fetch and process data to display results
                with st.spinner(f"Loading analysis for {pair_to_view}..."):
                    spread_df = fetch_historical_spreads(pair_to_view, days=days)
                    price_df = fetch_historical_prices(pair_to_view, days=days)
                    vol_df = calculate_volatility(price_df, window_minutes=vol_window)
                    combined_df = combine_spread_volatility_data(spread_df, vol_df)
                    
                    # Display detailed correlation analysis
                    if pair_to_view in results_dict:
                        display_all_correlations(results_dict[pair_to_view], pair_to_view)
                    
                    # Display results side by side
                    col1, col2 = st.columns([3, 2])
                    
                    with col1:
                        # Plot relationship
                        fig = plot_spread_volatility(combined_df, pair_to_view)
                        if fig:
                            st.pyplot(fig)
                    
                    with col2:
                        # Summary stats
                        if pair_to_view in results_dict and 'correlation' in results_dict[pair_to_view]:
                            results = results_dict[pair_to_view]
                            
                            st.markdown("#### Correlation Analysis")
                            
                            if results.get('best_volatility_measure'):
                                best_vol = results['best_volatility_measure'].replace('_', ' ').title()
                                st.metric("Best Volatility Measure", best_vol)
                                st.metric("Correlation", f"{results['best_volatility_correlation']:.4f}")
                            
                            # Regression results
                            st.markdown("#### Regression Results")
                            if results['regression_coefficient'] is not None:
                                st.metric("Coefficient", f"{results['regression_coefficient']:.6f}")
                                st.metric("R² Score", f"{results['r2']:.4f}")
                            
                            # Lead-lag results
                            st.markdown("#### Lead-Lag Analysis")
                            if results['best_lag'] is not None:
                                st.markdown(f"**{results['best_lag']['description']}**")
                                st.metric("Lag Correlation", f"{results['best_lag']['correlation']:.4f}")
                            
                            # Plot lead-lag relationship
                            if results and 'lead_lag' in results and results['lead_lag'] is not None:
                                lead_lag_fig = plot_lead_lag(results['lead_lag'], pair_to_view)
                                if lead_lag_fig:
                                    st.pyplot(lead_lag_fig)
        else:
            st.warning("No successful analyses to display.")
    
    with results_tabs[1]:
        if failed_pairs:
            st.markdown("### Pairs with Data Issues")
            
            # Create a table of failed pairs with reasons
            failed_df = pd.DataFrame(data_issues)
            st.dataframe(failed_df)
            
            # Show tips for fixing data issues
            st.markdown("""
            #### Tips for Fixing Data Issues
            
            1. **Increase the timeframe** to capture more historical data
            2. **Reduce the minimum data points** threshold
            3. **Check database connections** for the specific pair
            4. **Verify the pair is still active** in your system
            """)
        else:
            st.success("All pairs were analyzed successfully!")

def consolidate_results(results_dict):
    """Consolidate analysis results into a summary dataframe"""
    summary_data = []
    
    for pair, result in results_dict.items():
        if result and 'correlation' in result:
            try:
                best_vol_col = result.get('best_volatility_measure', 'volatility_24h')
                best_vol_corr = result.get('best_volatility_correlation', 
                                       result['correlation'].loc['avg_spread', best_vol_col])
                
                # Handle case where best_lag might be None
                best_lag = None
                best_lag_corr = None
                
                if result['best_lag'] is not None:
                    best_lag = result['best_lag']['lag']
                    best_lag_corr = result['best_lag']['correlation']
                
                r2 = result['r2']
                
                summary_data.append({
                    'pair_name': pair,
                    'volatility_measure': best_vol_col,
                    'spread_volatility_correlation': best_vol_corr,
                    'best_lag_hours': best_lag,
                    'best_lag_correlation': best_lag_corr,
                    'r2_score': r2,
                })
            except Exception as e:
                st.error(f"Error consolidating results for {pair}: {e}")
    
    return pd.DataFrame(summary_data)

def render_summary_tab(results_dict, selected_pairs, days, vol_window, min_data_points):
    """Render the summary tab with results across all pairs"""
    st.markdown("## Summary Results")
    st.markdown("Comparison of spread-volatility relationships across all analyzed pairs")
    
    # Process all pairs if results_dict is empty
    if not results_dict:
        results_dict, successful_pairs, failed_pairs, _ = process_all_pairs(
            selected_pairs, days, vol_window, min_data_points
        )
    
    # Consolidate results
    summary_df = consolidate_results(results_dict)
    
    if not summary_df.empty:
        # Sort by correlation strength (avoiding abs() function)
        summary_df['abs_correlation'] = summary_df['spread_volatility_correlation'].apply(lambda x: abs(x) if pd.notna(x) else 0)
        summary_df = summary_df.sort_values(by='abs_correlation', ascending=False)
        
        # Create a nicer display dataframe
        display_df = summary_df.copy()
        display_df = display_df.rename(columns={
            'pair_name': 'Pair',
            'volatility_measure': 'Best Volatility Measure',
            'spread_volatility_correlation': 'Correlation',
            'best_lag_hours': 'Best Lag (hours)',
            'best_lag_correlation': 'Lag Correlation',
            'r2_score': 'R² Score'
        })
        
        # Format the volatility measure names
        display_df['Best Volatility Measure'] = display_df['Best Volatility Measure'].apply(
            lambda x: x.replace('_', ' ').title() if isinstance(x, str) else 'N/A'
        )
        
        # Display summary table (without the abs column)
        st.dataframe(display_df.drop(columns=['abs_correlation']))
        
        # Create correlation distribution plot
        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
        valid_corrs = summary_df['spread_volatility_correlation'].dropna()
        ax.hist(valid_corrs, bins=10, alpha=0.7)
        ax.set_xlabel('Correlation: Spread vs. Volatility')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Spread-Volatility Correlations Across Pairs')
        ax.axvline(x=0, color='r', linestyle='--')
        mean_corr = valid_corrs.mean()
        ax.axvline(x=mean_corr, color='g', linestyle='-', label=f'Mean: {mean_corr:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Create lag distribution plot
        valid_lags = summary_df['best_lag_hours'].dropna()
        if len(valid_lags) > 0:
            fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
            ax.hist(valid_lags, bins=10, alpha=0.7)
            ax.set_xlabel('Best Lag (hours)')
            ax.set_ylabel('Count')
            ax.set_title('Distribution of Lead-Lag Relationships')
            ax.axvline(x=0, color='r', linestyle='--')
            mean_lag = valid_lags.mean()
            ax.axvline(x=mean_lag, color='g', linestyle='-', label=f'Mean: {mean_lag:.1f} hours')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        # Overall findings
        st.markdown("### Key Findings")
        
        # Calculate aggregate statistics
        avg_corr = valid_corrs.mean() if len(valid_corrs) > 0 else 0
        avg_abs_corr = valid_corrs.abs().mean() if len(valid_corrs) > 0 else 0
        avg_lag = valid_lags.mean() if len(valid_lags) > 0 else 0
        
        st.write(f"**Average Correlation:** {avg_corr:.4f} (Absolute: {avg_abs_corr:.4f})")
        
        if len(valid_lags) > 0:
            st.write(f"**Average Optimal Lag:** {avg_lag:.2f} hours")
            
            # Interpret results
            if avg_lag < -1:
                st.write(f"**Overall Finding:** Spread changes typically precede volatility changes by {-avg_lag:.1f} hours")
            elif avg_lag > 1:
                st.write(f"**Overall Finding:** Volatility changes typically precede spread changes by {avg_lag:.1f} hours")
            else:
                st.write("**Overall Finding:** Spread and volatility changes occur nearly simultaneously")
        
        if avg_abs_corr > 0.7:
            st.write("**Correlation Strength:** Very strong correlation between spreads and volatility")
        elif avg_abs_corr > 0.5:
            st.write("**Correlation Strength:** Strong correlation between spreads and volatility")
        elif avg_abs_corr > 0.3:
            st.write("**Correlation Strength:** Moderate correlation between spreads and volatility")
        else:
            st.write("**Correlation Strength:** Weak correlation between spreads and volatility")
            
        # Recommendations based on findings
        st.markdown("### Recommendations for Buffer Rate Strategy")
        
        if avg_abs_corr > 0.5:
            st.markdown("""
            Based on the strong correlation between spreads and volatility, you could use **either metric** 
            as the foundation for your dynamic buffer rate adjustment system. However, since spreads
            directly impact trading costs and liquidity, they may be a more intuitive basis for your strategy.
            """)
        else:
            st.markdown("""
            The correlation between spreads and volatility is relatively weak, suggesting you should
            **carefully choose** which metric to base your buffer rate adjustments on:
            
            - **Market Spreads:** More directly tied to trading costs and market depth
            - **Volatility:** More predictive of potential price swings and liquidation risks
            
            Consider using both in your formula with different weights.
            """)
            
        if avg_lag < -1:
            st.markdown(f"""
            Since spread changes typically precede volatility changes by {-avg_lag:.1f} hours,
            using spreads as your primary metric could provide an **early warning system**
            for incoming volatility changes.
            """)
        elif avg_lag > 1:
            st.markdown(f"""
            Since volatility changes typically precede spread changes by {avg_lag:.1f} hours,
            using volatility as your primary metric could help you **anticipate spread changes**
            and adjust buffer rates proactively.
            """)
            
        # Add example implementation
        st.markdown("### Example Implementation")
        
        if avg_abs_corr > 0.3:
            weight_spread = 0.7
            weight_vol = 0.3
        else:
            weight_spread = 0.9
            weight_vol = 0.1
            
        st.code(f"""
# Example dynamic buffer rate adjustment function using both metrics
def calculate_dynamic_buffer_rate(market_spread, volatility, token_type):
    # Base parameters
    spread_multiplier = 12.0  # Based on market analysis
    volatility_weight = {weight_vol:.1f}  # Weight for volatility
    spread_weight = {weight_spread:.1f}   # Weight for spread
    
    # Base buffer varies by token type
    if token_type == 'Major':
        base_buffer = 0.02  # 2%
    elif token_type == 'Stablecoin':
        base_buffer = 0.01  # 1%
    else:  # Altcoin
        base_buffer = 0.025  # 2.5%
    
    # Calculate buffer rate components
    spread_component = market_spread * spread_multiplier
    volatility_component = volatility * 0.10  # Convert volatility to buffer contribution
    
    # Weighted combination
    buffer_rate = (spread_component * spread_weight) + (volatility_component * volatility_weight) + base_buffer
    
    # Apply reasonable bounds
    buffer_rate = max(0.01, min(0.10, buffer_rate))
    
    return buffer_rate
        """, language="python")
    else:
        st.warning("No results available for summary. Please analyze some pairs first.")

def calculate_dynamic_buffer_rate(market_spread, volatility, token_type):
    """Calculate dynamic buffer rate based on market spread, volatility, and token type
    
    Args:
        market_spread (float): Current market spread (decimal, e.g. 0.0015 for 15 bps)
        volatility (float): Current volatility measure (decimal, annualized)
        token_type (str): 'Major', 'Stablecoin', or 'Altcoin'
        
    Returns:
        float: Recommended buffer rate (decimal, e.g. 0.025 for 2.5%)
    """
    # Base parameters
    spread_multiplier = 12.0  # Based on market analysis
    
    # Different volatility multipliers based on token type
    if token_type == 'Major':
        base_buffer = 0.02  # 2%
        volatility_multiplier = 0.08
        spread_weight = 0.7
        volatility_weight = 0.3
    elif token_type == 'Stablecoin':
        base_buffer = 0.01  # 1%
        volatility_multiplier = 0.05
        spread_weight = 0.8
        volatility_weight = 0.2
    else:  # Altcoin
        base_buffer = 0.025  # 2.5%
        volatility_multiplier = 0.12
        spread_weight = 0.6
        volatility_weight = 0.4
    
    # Calculate buffer rate components
    spread_component = market_spread * spread_multiplier
    volatility_component = volatility * volatility_multiplier
    
    # Weighted combination
    buffer_rate = (spread_component * spread_weight) + (volatility_component * volatility_weight) + base_buffer
    
    # Apply reasonable bounds
    buffer_rate = max(0.01, min(0.10, buffer_rate))
    
    return buffer_rate

def backtest_buffer_strategy(combined_df, pair_name, token_type='Altcoin'):
    """Backtest how a dynamic buffer strategy would have performed"""
    if combined_df is None or len(combined_df) < 10:
        return None
    
    # Find best volatility measure
    vol_cols = [col for col in combined_df.columns if 'volatility' in col or 'vol' in col]
    correlations = {}
    for col in vol_cols:
        corr = combined_df['avg_spread'].corr(combined_df[col])
        if not pd.isna(corr):
            correlations[col] = abs(corr)
    
    if not correlations:
        return None
        
    # Use the volatility measure with the strongest correlation
    best_vol_col = max(correlations.items(), key=lambda x: x[1])[0]
    
    # Create a copy for backtesting
    backtest_df = combined_df.copy()
    
    # Determine token type if not provided
    if token_type is None:
        if 'BTC' in pair_name or 'ETH' in pair_name:
            token_type = 'Major'
        elif 'USDT' in pair_name or 'USDC' in pair_name:
            token_type = 'Stablecoin'
        else:
            token_type = 'Altcoin'
    
    # Calculate dynamic buffer rates
    backtest_df['dynamic_buffer'] = backtest_df.apply(
        lambda row: calculate_dynamic_buffer_rate(
            row['avg_spread'], 
            row[best_vol_col], 
            token_type
        ), 
        axis=1
    )
    
    # Calculate a naive baseline that uses only spread * multiplier
    backtest_df['naive_buffer'] = backtest_df['avg_spread'] * 12.0 + 0.02
    
    # Calculate % change in spreads to simulate potential losses
    backtest_df['spread_chg_1h'] = backtest_df['avg_spread'].pct_change(1)
    backtest_df['spread_chg_2h'] = backtest_df['avg_spread'].pct_change(2)
    backtest_df['spread_chg_3h'] = backtest_df['avg_spread'].pct_change(3)
    
    # Simulate how many times our buffer would have been insufficient
    # by checking if spread changed by more than buffer within N hours
    for hrs in [1, 2, 3]:
        spread_chg_col = f'spread_chg_{hrs}h'
        backtest_df[f'dynamic_buffer_breach_{hrs}h'] = abs(backtest_df[spread_chg_col]) > backtest_df['dynamic_buffer']
        backtest_df[f'naive_buffer_breach_{hrs}h'] = abs(backtest_df[spread_chg_col]) > backtest_df['naive_buffer']
    
    # Create summary statistics
    results = {
        'pair': pair_name,
        'token_type': token_type,
        'avg_dynamic_buffer': backtest_df['dynamic_buffer'].mean(),
        'avg_naive_buffer': backtest_df['naive_buffer'].mean(),
        'max_dynamic_buffer': backtest_df['dynamic_buffer'].max(),
        'min_dynamic_buffer': backtest_df['dynamic_buffer'].min(),
        'volatility_measure': best_vol_col
    }
    
    # Add breach statistics
    for hrs in [1, 2, 3]:
        dynamic_breach_col = f'dynamic_buffer_breach_{hrs}h'
        naive_breach_col = f'naive_buffer_breach_{hrs}h'
        
        results[f'dynamic_breaches_{hrs}h'] = backtest_df[dynamic_breach_col].sum()
        results[f'dynamic_breach_rate_{hrs}h'] = backtest_df[dynamic_breach_col].mean()
        
        results[f'naive_breaches_{hrs}h'] = backtest_df[naive_breach_col].sum()
        results[f'naive_breach_rate_{hrs}h'] = backtest_df[naive_breach_col].mean()
    
    return results, backtest_df

def display_backtest_results(backtest_results, backtest_df, pair_name):
    """Display the results of the buffer rate strategy backtest"""
    if backtest_results is None or backtest_df is None:
        st.warning("Not enough data for backtesting")
        return
    
    st.markdown(f"### Buffer Strategy Backtest Results for {pair_name}")
    
    # Display summary statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Avg Dynamic Buffer", f"{backtest_results['avg_dynamic_buffer']*100:.2f}%")
    with col2:
        st.metric("Avg Naive Buffer", f"{backtest_results['avg_naive_buffer']*100:.2f}%")
    with col3:
        st.metric("Buffer Difference", f"{(backtest_results['avg_dynamic_buffer'] - backtest_results['avg_naive_buffer'])*100:.2f}%")
    
    # Show breach statistics
    st.markdown("#### Breach Analysis (% of time buffer was insufficient)")
    
    breach_data = []
    for hrs in [1, 2, 3]:
        breach_data.append({
            'timeframe': f"{hrs} hour{'s' if hrs > 1 else ''}",
            'dynamic_rate': backtest_results[f'dynamic_breach_rate_{hrs}h'] * 100,
            'naive_rate': backtest_results[f'naive_breach_rate_{hrs}h'] * 100
        })
    
    breach_df = pd.DataFrame(breach_data)
    
    # Create a bar chart for breaches
    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
    x = np.arange(len(breach_df))
    width = 0.35
    
    bar1 = ax.bar(x - width/2, breach_df['dynamic_rate'], width, label='Dynamic Buffer Strategy')
    bar2 = ax.bar(x + width/2, breach_df['naive_rate'], width, label='Naive Buffer Strategy')
    
    ax.set_xlabel('Time Window')
    ax.set_ylabel('Breach Rate (%)')
    ax.set_title('Buffer Breach Rates by Strategy')
    ax.set_xticks(x)
    ax.set_xticklabels(breach_df['timeframe'])
    ax.legend()
    
    for bars in [bar1, bar2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    st.pyplot(fig)
    
    # Plot the buffer rates over time
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    
    # Plot both buffer strategies
    ax.plot(backtest_df['hour'], backtest_df['dynamic_buffer']*100, 'b-', label='Dynamic Buffer (%)')
    ax.plot(backtest_df['hour'], backtest_df['naive_buffer']*100, 'r--', label='Naive Buffer (%)')
    
    # Format x-axis to show dates nicely
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, fontsize=8)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Buffer Rate (%)')
    ax.set_title(f'Buffer Rate Strategies Over Time - {pair_name}')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Show efficiency comparison
    avg_dynamic = backtest_results['avg_dynamic_buffer']
    avg_naive = backtest_results['avg_naive_buffer']
    
    dyn_breach_2h = backtest_results['dynamic_breach_rate_2h']
    naive_breach_2h = backtest_results['naive_breach_rate_2h']
    
    if avg_dynamic < avg_naive and dyn_breach_2h <= naive_breach_2h:
        st.success(f"""
        **Strategy Assessment: EXCELLENT**
        
        The dynamic buffer strategy achieves **lower average buffer rates** ({avg_dynamic*100:.2f}% vs {avg_naive*100:.2f}%)
        while maintaining equal or better protection against spread movements.
        """)
    elif avg_dynamic < avg_naive and dyn_breach_2h > naive_breach_2h:
        st.warning(f"""
        **Strategy Assessment: GOOD**
        
        The dynamic buffer strategy uses **lower average buffer rates** ({avg_dynamic*100:.2f}% vs {avg_naive*100:.2f}%)
        but has slightly higher breach rates. This may be an acceptable tradeoff for efficiency.
        """)
    elif avg_dynamic > avg_naive and dyn_breach_2h < naive_breach_2h:
        st.info(f"""
        **Strategy Assessment: CONSERVATIVE**
        
        The dynamic buffer strategy uses **higher average buffer rates** ({avg_dynamic*100:.2f}% vs {avg_naive*100:.2f}%)
        but provides better protection against spread movements.
        """)
    else:
        st.error(f"""
        **Strategy Assessment: NEEDS IMPROVEMENT**
        
        The dynamic buffer strategy uses **higher average buffer rates** ({avg_dynamic*100:.2f}% vs {avg_naive*100:.2f}%)
        without providing better protection. Consider adjusting the parameters.
        """)
    
    # Recommendations for improving the strategy
    st.markdown("### Strategy Optimization Recommendations")
    
    vol_measure = backtest_results['volatility_measure'].replace('_', ' ').title()
    token_type = backtest_results['token_type']
    
    st.markdown(f"""
    Based on the backtest results for {pair_name} ({token_type}):
    
    1. **Volatility Measure**: {vol_measure} showed the strongest correlation with spread
    2. **Buffer Allocation**: 
        - Base buffer: {token_type}-specific base rate
        - Spread component: Using 12.0x multiplier
        - Volatility component: Using token-type specific weight
        
    **Fine-tuning suggestions:**
    """)
    
    if dyn_breach_2h > 0.05:  # More than 5% breach rate
        st.markdown("""
        - Increase the base buffer rate slightly
        - Consider increasing the volatility weight for better responsiveness
        """)
    elif avg_dynamic > 0.05:  # Average buffer over 5%
        st.markdown("""
        - Reduce the spread multiplier slightly
        - Implement a cap on maximum buffer rates during extreme volatility
        """)
    else:
        st.markdown("""
        - Current parameters seem well-balanced
        - Consider testing with different lead-lag values to be more proactive
        """)

def filter_and_display_tokens(token_types, token_results):
    """Filter and display token-specific results"""
    filtered_results = [r for r in token_results if r['token_type'] in token_types]
    
    if not filtered_results:
        st.info(f"No results available for the selected token types")
        return
    
    # Create a dataframe for comparison
    df = pd.DataFrame(filtered_results)
    
    # Format for display
    display_df = df[['pair', 'token_type', 'avg_dynamic_buffer', 'avg_naive_buffer', 
                     'dynamic_breach_rate_2h', 'naive_breach_rate_2h']]
    
    # Format percentages
    display_df['avg_dynamic_buffer'] = display_df['avg_dynamic_buffer'] * 100
    display_df['avg_naive_buffer'] = display_df['avg_naive_buffer'] * 100
    display_df['dynamic_breach_rate_2h'] = display_df['dynamic_breach_rate_2h'] * 100
    display_df['naive_breach_rate_2h'] = display_df['naive_breach_rate_2h'] * 100
    
    # Rename columns
    display_df = display_df.rename(columns={
        'pair': 'Pair',
        'token_type': 'Token Type',
        'avg_dynamic_buffer': 'Avg Dynamic Buffer (%)',
        'avg_naive_buffer': 'Avg Naive Buffer (%)',
        'dynamic_breach_rate_2h': 'Dynamic Breach Rate (%)',
        'naive_breach_rate_2h': 'Naive Breach Rate (%)'
    })
    
    # Calculate efficiency metrics
    display_df['Buffer Reduction (%)'] = display_df['Avg Naive Buffer (%)'] - display_df['Avg Dynamic Buffer (%)']
    display_df['Breach Rate Change (%)'] = display_df['Dynamic Breach Rate (%)'] - display_df['Naive Breach Rate (%)']
    
    # Sort by token type and then by buffer reduction
    display_df = display_df.sort_values(['Token Type', 'Buffer Reduction (%)'], ascending=[True, False])
    
    st.dataframe(display_df)
    
    # Display average savings by token type
    st.markdown("### Average Efficiency Gains by Token Type")
    
    token_stats = df.groupby('token_type').agg({
        'avg_dynamic_buffer': 'mean',
        'avg_naive_buffer': 'mean',
        'dynamic_breach_rate_2h': 'mean',
        'naive_breach_rate_2h': 'mean'
    })
    
    # Format for visualization
    token_stats['buffer_reduction'] = token_stats['avg_naive_buffer'] - token_stats['avg_dynamic_buffer']
    token_stats['breach_rate_change'] = token_stats['dynamic_breach_rate_2h'] - token_stats['naive_breach_rate_2h']
    
    # Create a comparison chart
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    
    x = np.arange(len(token_stats.index))
    width = 0.35
    
    bar1 = ax.bar(x - width/2, token_stats['avg_dynamic_buffer']*100, width, label='Dynamic Buffer')
    bar2 = ax.bar(x + width/2, token_stats['avg_naive_buffer']*100, width, label='Naive Buffer')
    
    ax.set_xlabel('Token Type')
    ax.set_ylabel('Average Buffer Rate (%)')
    ax.set_title('Average Buffer Rates by Token Type')
    ax.set_xticks(x)
    ax.set_xticklabels(token_stats.index)
    ax.legend()
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Show breach rate comparison
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    
    bar1 = ax.bar(x - width/2, token_stats['dynamic_breach_rate_2h']*100, width, label='Dynamic Buffer')
    bar2 = ax.bar(x + width/2, token_stats['naive_breach_rate_2h']*100, width, label='Naive Buffer')
    
    ax.set_xlabel('Token Type')
    ax.set_ylabel('Average 2h Breach Rate (%)')
    ax.set_title('Buffer Breach Rates by Token Type')
    ax.set_xticks(x)
    ax.set_xticklabels(token_stats.index)
    ax.legend()
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Summary findings
    st.markdown("### Key Findings by Token Type")
    
    for token_type in token_stats.index:
        buffer_reduction = token_stats.loc[token_type, 'buffer_reduction'] * 100
        breach_change = token_stats.loc[token_type, 'breach_rate_change'] * 100
        
        st.markdown(f"**{token_type}**:")
        
        if buffer_reduction > 0:
            st.markdown(f"- **Buffer Reduction**: {buffer_reduction:.2f}% lower than naive approach")
        else:
            st.markdown(f"- **Buffer Increase**: {-buffer_reduction:.2f}% higher than naive approach")
            
        if breach_change <= 0:
            st.markdown(f"- **Breach Protection**: {-breach_change:.2f}% fewer breaches than naive approach")
        else:
            st.markdown(f"- **Breach Increase**: {breach_change:.2f}% more breaches than naive approach")
        
        # Token-specific recommendations
        if buffer_reduction > 0 and breach_change <= 0:
            st.markdown("- **Recommendation**: Current parameters work well ✅")
        elif buffer_reduction > 0 and breach_change > 0:
            st.markdown("- **Recommendation**: Slight increase in volatility multiplier may help ⚠️")
        elif buffer_reduction <= 0 and breach_change <= 0:
            st.markdown("- **Recommendation**: Parameters are too conservative, can be optimized ⚠️")
        else:
            st.markdown("- **Recommendation**: Strategy not effective, needs significant adjustment ❌")

def render_strategy_backtest_tab(selected_pairs, days, vol_window, min_data_points):
    """Render the buffer strategy backtest tab"""
    st.markdown("## Buffer Strategy Backtest")
    st.markdown("Testing how our dynamic buffer rate strategy would have performed")
    
    # Create backtest settings
    col1, col2 = st.columns(2)
    
    with col1:
        selected_token_types = st.multiselect(
            "Filter by Token Type",
            options=["Major", "Stablecoin", "Altcoin"],
            default=["Major", "Altcoin"]
        )
    
    with col2:
        backtest_pair = st.selectbox(
            "Select a specific pair to analyze in detail",
            options=selected_pairs,
            index=0 if selected_pairs else None
        )
    
    # Process selected pair for detailed analysis
    if backtest_pair:
        with st.spinner(f"Running backtest for {backtest_pair}..."):
            # Fetch data
            spread_df = fetch_historical_spreads(backtest_pair, days=days)
            price_df = fetch_historical_prices(backtest_pair, days=days)
            
            # Determine token type
            token_type = None
            if any(token in backtest_pair for token in ['BTC', 'ETH']):
                token_type = 'Major'
            elif any(token in backtest_pair for token in ['USDT', 'USDC']):
                token_type = 'Stablecoin'
            else:
                token_type = 'Altcoin'
            
            # Calculate volatility
            if price_df is not None:
                vol_df = calculate_volatility(price_df, window_minutes=vol_window)
                
                # Combine data
                if spread_df is not None and vol_df is not None:
                    combined_df = combine_spread_volatility_data(spread_df, vol_df)
                    
                    # Run backtest
                    if combined_df is not None and len(combined_df) >= min_data_points:
                        backtest_results, backtest_df = backtest_buffer_strategy(
                            combined_df, backtest_pair, token_type
                        )
                        
                        # Display results
                        display_backtest_results(backtest_results, backtest_df, backtest_pair)
                    else:
                        st.warning(f"Not enough data for {backtest_pair}")
                else:
                    st.warning(f"Missing spread or volatility data for {backtest_pair}")
            else:
                st.warning(f"No price data available for {backtest_pair}")
    
    # Process all pairs and display aggregate results by token type
    if selected_token_types:
        st.markdown("## Aggregate Results by Token Type")
        st.markdown("Comparing the performance of our dynamic buffer strategy across different token types")
        
        # Process all pairs
        with st.spinner("Running backtest across all pairs..."):
            all_results = []
            
            for pair in selected_pairs:
                # Determine token type
                token_type = 'Altcoin'  # Default
                if any(token in pair for token in ['BTC', 'ETH']):
                    token_type = 'Major'
                elif any(token in pair for token in ['USDT', 'USDC']):
                    token_type = 'Stablecoin'
                
                # Skip if token type not selected
                if token_type not in selected_token_types:
                    continue
                
                # Fetch data
                spread_df = fetch_historical_spreads(pair, days=days)
                price_df = fetch_historical_prices(pair, days=days)
                
                # Calculate volatility
                if price_df is not None:
                    vol_df = calculate_volatility(price_df, window_minutes=vol_window)
                    
                    # Combine data
                    if spread_df is not None and vol_df is not None:
                        combined_df = combine_spread_volatility_data(spread_df, vol_df)
                        
                        # Run backtest
                        if combined_df is not None and len(combined_df) >= min_data_points:
                            backtest_result, _ = backtest_buffer_strategy(
                                combined_df, pair, token_type
                            )
                            
                            if backtest_result is not None:
                                all_results.append(backtest_result)
            
            # Display token type analysis
            if all_results:
                filter_and_display_tokens(selected_token_types, all_results)
            else:
                st.warning("No backtest results available")

# Main app
def main():
    # Sidebar for global settings
    st.sidebar.header("Analysis Settings")
    
    # Timeframe selection
    timeframe_options = get_timeframe_options()
    selected_timeframe = st.sidebar.selectbox(
        "Select Analysis Timeframe",
        options=list(timeframe_options.keys()),
        index=2  # Default to 7 days
    )
    days = timeframe_options[selected_timeframe]
    
    # Fetch active pairs
    all_pairs = fetch_active_pairs()
    
    # Allow user to select specific pairs or categories
    pair_selection = st.sidebar.radio(
        "Select Pairs to Analyze",
        options=["Major Pairs", "All Active Pairs", "Custom Selection"]
    )
    
    # Define major pairs
    major_pairs = [p for p in all_pairs if any(major in p for major in 
                                           ["BTC", "ETH", "SOL", "XRP", "BNB", "DOGE"])]
    
    if pair_selection == "Major Pairs":
        selected_pairs = major_pairs
    elif pair_selection == "All Active Pairs":
        selected_pairs = all_pairs
    else:  # Custom Selection
        default_selection = major_pairs[:3]  # Default to first 3 major pairs
        selected_pairs = st.sidebar.multiselect(
            "Select specific trading pairs",
            options=all_pairs,
            default=default_selection
        )
    
    # Add volatility timeframe settings
    st.sidebar.header("Volatility Settings")
    vol_window = st.sidebar.slider(
        "Volatility Window (minutes)", 
        min_value=5, 
        max_value=240, 
        value=60,
        step=5,
        help="Time window used for calculating rolling volatility. Shorter windows capture more recent market movements."
    )
    
    # Add minimum data points threshold
    min_data_points = st.sidebar.slider(
        "Minimum Data Points",
        min_value=5,
        max_value=50,
        value=10,
        help="Minimum number of data points required for analysis. Increase for more reliable results, decrease for more coverage."
    )
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["Individual Pair Analysis", "Summary Results", "Strategy Backtest"])
    
    # Individual Pair Analysis Tab
    with tab1:
        render_individual_analysis_tab(selected_pairs, days, vol_window, min_data_points)
    
    # Summary Results Tab
    with tab2:
        render_summary_tab({}, selected_pairs, days, vol_window, min_data_points)
    
    # Strategy Backtest Tab
    with tab3:
        render_strategy_backtest_tab(selected_pairs, days, vol_window, min_data_points)

if __name__ == "__main__":
    main()