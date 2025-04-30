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

st.title("Market Spreads vs Volatility Analysis")
st.markdown("Analyzing the relationship between market spreads and volatility to optimize buffer rates")

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
    st.stop()

# Fetch active pairs
@st.cache_data(ttl=600)
def fetch_active_pairs():
    """Fetch pairs that have been active in the last 30 days"""
    query = """
    SELECT DISTINCT pair_name 
    FROM public.trade_fill_fresh 
    WHERE created_at > NOW() - INTERVAL '30 days'
    ORDER BY pair_name
    """
    
    try:
        df = pd.read_sql(query, engine)
        if df.empty:
            return []
        return df['pair_name'].tolist()
    except Exception as e:
        st.error(f"Error fetching pairs: {e}")
        return ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

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

# Calculate historical volatility with improved granularity and method
@st.cache_data(ttl=600)
def fetch_historical_prices(pair_name, days=7):
    """Fetch historical price data with multiple fallback methods to ensure data availability"""
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
    """Calculate rolling volatility from price data with more granular timeframes"""
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
    
    # Convert window_minutes to number of 5-minute intervals
    window_size = max(2, int(window_minutes / 5))
    
    # Ensure window size is reasonable given available data
    window_size = min(window_size, len(vol_df) // 3)
    window_size = max(2, window_size)  # Ensure at least 2
    
    # Calculate multiple volatility measures
    # 1. Short-term (5-min intervals rolling)
    vol_df['volatility_short'] = vol_df['log_return'].rolling(window=3).std() * np.sqrt(12 * 24 * 365)
    
    # 2. Medium-term (approximately 1-hour rolling)
    vol_df['volatility_medium'] = vol_df['log_return'].rolling(window=window_size).std() * np.sqrt(12 * 24 * 365)
    
    # 3. Full period volatility
    vol_df['volatility_full'] = vol_df['log_return'].expanding().std() * np.sqrt(12 * 24 * 365)
    
    # 4. EWMA volatility (gives more weight to recent observations)
    vol_df['volatility_ewma'] = vol_df['log_return'].ewm(span=window_size).std() * np.sqrt(12 * 24 * 365)
    
    # Handle missing values without using inplace=True
    for col in ['volatility_short', 'volatility_medium', 'volatility_full', 'volatility_ewma']:
        vol_df[col] = vol_df[col].bfill().ffill()
    
    # Add timestamps in different formats for easier joining
    vol_df['hour'] = vol_df['interval'].dt.floor('H')
    
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
    
    # Create an hourly grouped volatility dataframe
    hourly_vol_df = vol_df.groupby('hour').agg({
        'volatility_short': 'mean',
        'volatility_medium': 'mean',
        'volatility_full': 'mean',
        'volatility_ewma': 'mean',
        'interval_volatility': 'mean',
        'log_return': ['mean', 'std'],
        'deal_price_mean': 'mean'
    })
    
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

def analyze_relationship(combined_df):
    """Analyze the relationship between spreads and volatility with multiple measures"""
    if combined_df is None or len(combined_df) < 10:
        return None
    
    # Create correlation matrix with all volatility measures
    vol_cols = [col for col in combined_df.columns if 'volatility' in col]
    corr_columns = ['avg_spread'] + vol_cols
    
    # Make sure all required columns exist
    available_columns = [col for col in corr_columns if col in combined_df.columns]
    
    if len(available_columns) < 3:  # Need at least spread and two vol measures
        return None
    
    # Drop rows with NaN values for correlation calculation
    clean_df_corr = combined_df[available_columns].dropna()
    if len(clean_df_corr) < 5:  # Need at least a few data points
        st.warning("Not enough valid data points for correlation analysis")
        return None
        
    correlation = clean_df_corr.corr()
    
    # Determine which volatility measure has the strongest correlation with spread
    spread_correlations = correlation.loc['avg_spread', vol_cols]
    best_vol_col = spread_correlations.abs().idxmax()
    best_corr = spread_correlations[best_vol_col]
    
    # For regression, use the volatility measure with the strongest correlation
    reg_df = combined_df[['avg_spread', best_vol_col]].dropna()
    
    if len(reg_df) < 5:  # Need at least a few data points
        st.warning("Not enough valid data points for regression analysis")
        return {
            'correlation': correlation,
            'best_volatility_measure': best_vol_col,
            'best_volatility_correlation': best_corr,
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
            'regression_coefficient': None,
            'r2': None,
            'lead_lag': None,
            'best_lag': None
        }

def plot_spread_volatility(combined_df, pair_name):
    """Plot spread vs. volatility relationship with more compact layout"""
    if combined_df is None or len(combined_df) < 5:
        st.warning(f"Not enough data for {pair_name} to create plots")
        return
    
    # Identify which volatility measures are available
    vol_cols = [col for col in combined_df.columns if 'volatility' in col]
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
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    
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
    
    fig, ax = plt.subplots(figsize=(7, 4))
    
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

def fetch_rollbit_buffer_vs_spread(days=30):
    """Analyze if Rollbit buffer rates correlate with market spreads"""
    query = f"""
    WITH rollbit_data AS (
        SELECT 
            created_at,
            pair_name,
            bust_buffer
        FROM 
            rollbit_pair_config
        WHERE 
            created_at > NOW() - INTERVAL '{days} days'
    ),
    
    daily_spreads AS (
        SELECT 
            date_trunc('day', time_group) as day,
            pair_name,
            AVG(fee1) as avg_spread
        FROM 
            oracle_exchange_fee
        WHERE 
            time_group > NOW() - INTERVAL '{days} days'
            AND source IN ('binanceFuture', 'gateFuture', 'hyperliquidFuture')
        GROUP BY 
            day, pair_name
    )
    
    SELECT 
        r.created_at,
        r.pair_name,
        r.bust_buffer,
        s.avg_spread
    FROM 
        rollbit_data r
    JOIN 
        daily_spreads s ON r.pair_name = s.pair_name
        AND date_trunc('day', r.created_at) = s.day
    ORDER BY
        r.pair_name, r.created_at
    """
    
    try:
        df = pd.read_sql(query, engine)
        if df.empty:
            return None
        
        # Calculate correlation by pair
        correlations = []
        for pair, group in df.groupby('pair_name'):
            if len(group) > 5:  # Need reasonable sample size
                corr = group['bust_buffer'].corr(group['avg_spread'])
                correlations.append({
                    'pair_name': pair,
                    'correlation': corr,
                    'data_points': len(group),
                    'rollbit_buffer_avg': group['bust_buffer'].mean() * 100,  # Convert to percentage
                    'spread_avg': group['avg_spread'].mean() * 10000,  # Convert to basis points
                    'ratio': (group['bust_buffer'].mean() / group['avg_spread'].mean()) if group['avg_spread'].mean() > 0 else None
                })
        
        return pd.DataFrame(correlations)
    
    except Exception as e:
        st.error(f"Error analyzing Rollbit buffer vs spread: {e}")
        return None

def analyze_rollbit_buffer_adjustment_pattern(days=30):
    """Analyze how Rollbit adjusts buffer rates - by size, frequency, and direction"""
    query = f"""
    WITH rollbit_changes AS (
        SELECT 
            pair_name,
            created_at,
            bust_buffer,
            position_multiplier,
            LAG(bust_buffer) OVER (PARTITION BY pair_name ORDER BY created_at) AS prev_buffer,
            LAG(position_multiplier) OVER (PARTITION BY pair_name ORDER BY created_at) AS prev_multiplier
        FROM 
            rollbit_pair_config
        WHERE 
            created_at > NOW() - INTERVAL '{days} days'
    )
    
    SELECT 
        pair_name,
        created_at,
        bust_buffer,
        prev_buffer,
        CASE 
            WHEN prev_buffer IS NOT NULL THEN (bust_buffer - prev_buffer) / prev_buffer * 100
            ELSE NULL
        END AS buffer_pct_change,
        position_multiplier,
        prev_multiplier,
        CASE 
            WHEN prev_multiplier IS NOT NULL THEN (position_multiplier - prev_multiplier) / prev_multiplier * 100
            ELSE NULL
        END AS multiplier_pct_change
    FROM 
        rollbit_changes
    WHERE 
        prev_buffer IS NOT NULL OR prev_multiplier IS NOT NULL
    ORDER BY 
        pair_name, created_at
    """
    
    try:
        df = pd.read_sql(query, engine)
        if df.empty:
            return None
        
        # Analyze patterns in the adjustments
        adjustments = []
        for pair, group in df.groupby('pair_name'):
            # Buffer adjustments
            buffer_changes = group['buffer_pct_change'].dropna()
            
            # Skip pairs with insufficient data
            if len(buffer_changes) < 3:
                continue
                
            # Calculate statistics
            avg_buffer_change = buffer_changes.abs().mean()
            std_buffer_change = buffer_changes.std()
            max_buffer_increase = buffer_changes.max()
            max_buffer_decrease = buffer_changes.min()
            
            # Calculate frequency of adjustments
            first_date = group['created_at'].min()
            last_date = group['created_at'].max()
            days_span = (last_date - first_date).total_seconds() / (24 * 3600)
            
            adjustments_count = len(buffer_changes)
            avg_days_between = days_span / adjustments_count if adjustments_count > 0 else None
            
            adjustments.append({
                'pair_name': pair,
                'data_points': len(group),
                'days_analyzed': days_span,
                'buffer_adjustments_count': adjustments_count,
                'avg_days_between_adjustments': avg_days_between,
                'avg_buffer_change_pct': avg_buffer_change,
                'std_buffer_change_pct': std_buffer_change,
                'max_buffer_increase_pct': max_buffer_increase,
                'max_buffer_decrease_pct': max_buffer_decrease,
                'increases_count': len(buffer_changes[buffer_changes > 0]),
                'decreases_count': len(buffer_changes[buffer_changes < 0]),
                'buffer_current': group['bust_buffer'].iloc[-1] * 100,  # Convert to percentage
                'buffer_min': group['bust_buffer'].min() * 100,
                'buffer_max': group['bust_buffer'].max() * 100
            })
        
        return pd.DataFrame(adjustments)
        
    except Exception as e:
        st.error(f"Error analyzing Rollbit adjustment patterns: {e}")
        return None

def analyze_rollbit_by_token_type(df):
    """Analyze Rollbit's buffer strategies by token type"""
    if df is None or df.empty:
        return None
    
    # Define token categories
    major_tokens = ["BTC", "ETH", "SOL", "XRP", "BNB", "DOGE"]
    stablecoins = ["USDT", "USDC", "DAI"]
    
    # Add token type column
    df['token_type'] = 'Altcoin'  # Default
    
    for idx, row in df.iterrows():
        pair = row['pair_name']
        
        # Check if any major token is in the pair
        if any(token in pair for token in major_tokens):
            df.at[idx, 'token_type'] = 'Major'
            
        # Check if any stablecoin is in the pair
        if any(token in pair for token in stablecoins):
            df.at[idx, 'token_type'] = 'Stablecoin'
    
    # Group by token type and calculate averages
    token_analysis = df.groupby('token_type').agg({
        'buffer_current': 'mean',
        'avg_buffer_change_pct': 'mean',
        'avg_days_between_adjustments': 'mean',
        'buffer_min': 'mean',
        'buffer_max': 'mean'
    }).reset_index()
    
    return token_analysis

def display_rollbit_strategy_summary(df_changes, df_correlations):
    """Display a summary of Rollbit's buffer rate strategy"""
    if df_changes is None or df_changes.empty:
        st.warning("Not enough data to analyze Rollbit's strategy")
        return
    
    # Create a summary card
    st.markdown("### Rollbit Buffer Rate Strategy Summary")
    
    # Overall stats
    st.markdown("#### Overall Adjustment Patterns")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_adjustment = df_changes['avg_buffer_change_pct'].mean()
        st.metric("Avg Buffer Change", f"{avg_adjustment:.2f}%")
        
    with col2:
        avg_time = df_changes['avg_days_between_adjustments'].mean()
        st.metric("Avg Days Between Adjustments", f"{avg_time:.1f}")
        
    with col3:
        increase_vs_decrease = df_changes['increases_count'].sum() / max(1, df_changes['decreases_count'].sum())
        st.metric("Increase:Decrease Ratio", f"{increase_vs_decrease:.2f}")
    
    # Token type analysis
    token_analysis = analyze_rollbit_by_token_type(df_changes)
    
    if token_analysis is not None and not token_analysis.empty:
        st.markdown("#### Buffer Strategies by Token Type")
        
        # Create a readable dataframe for display
        display_df = token_analysis.copy()
        display_df = display_df.rename(columns={
            'buffer_current': 'Current Buffer (%)',
            'avg_buffer_change_pct': 'Avg Change (%)',
            'avg_days_between_adjustments': 'Days Between Adjustments',
            'buffer_min': 'Min Buffer (%)',
            'buffer_max': 'Max Buffer (%)'
        })
        
        # Format numeric columns
        for col in display_df.columns:
            if col != 'token_type':
                display_df[col] = display_df[col].map(lambda x: f"{x:.2f}")
        
        st.dataframe(display_df)
    
    # Correlation with market spreads
    if df_correlations is not None and not df_correlations.empty:
        st.markdown("#### Correlation with Market Spreads")
        
        avg_corr = df_correlations['correlation'].mean()
        abs_avg_corr = df_correlations['correlation'].abs().mean()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Average Correlation", f"{avg_corr:.4f}")
            
        with col2:
            st.metric("Average Absolute Correlation", f"{abs_avg_corr:.4f}")
        
        # Create ratio analysis
        if 'ratio' in df_correlations.columns:
            ratios = df_correlations['ratio'].dropna()
            
            if not ratios.empty:
                avg_ratio = ratios.mean()
                st.markdown(f"#### Buffer to Spread Ratio: {avg_ratio:.2f}x")
                st.markdown("""
                *This ratio represents how much larger Rollbit's buffer rates are compared to market spreads.
                A higher ratio means more conservative (higher margin requirements).*
                """)
    
    # Key insights
    st.markdown("### Key Insights on Rollbit's Strategy")
    
    # Generate insights based on the data
    insights = []
    
    if df_changes['avg_days_between_adjustments'].mean() < 7:
        insights.append("Rollbit frequently adjusts buffer rates (less than weekly on average)")
    else:
        insights.append("Rollbit makes infrequent buffer rate adjustments (typically weekly or longer)")
    
    if df_changes['avg_buffer_change_pct'].mean() < 5:
        insights.append("Rollbit typically makes small incremental adjustments to buffer rates")
    else:
        insights.append("Rollbit makes substantial adjustments to buffer rates when they change")
    
    if token_analysis is not None and not token_analysis.empty:
        # Compare major vs altcoin buffers
        major_row = token_analysis[token_analysis['token_type'] == 'Major']
        alt_row = token_analysis[token_analysis['token_type'] == 'Altcoin']
        
        if not major_row.empty and not alt_row.empty:
            major_buffer = major_row['buffer_current'].iloc[0]
            alt_buffer = alt_row['buffer_current'].iloc[0]
            
            if major_buffer < alt_buffer:
                insights.append(f"Rollbit uses lower buffer rates for major tokens compared to altcoins")
            else:
                insights.append(f"Rollbit uses similar or higher buffer rates for major tokens compared to altcoins")
    
    if df_correlations is not None and not df_correlations.empty:
        avg_abs_corr = df_correlations['correlation'].abs().mean()
        
        if avg_abs_corr > 0.5:
            insights.append("Strong correlation between Rollbit's buffer rates and market spreads")
        elif avg_abs_corr > 0.3:
            insights.append("Moderate correlation between Rollbit's buffer rates and market spreads")
        else:
            insights.append("Weak correlation between Rollbit's buffer rates and market spreads")
    
    # Display insights
    for i, insight in enumerate(insights):
        st.markdown(f"**{i+1}. {insight}**")
    
    # Implementation recommendation
    st.markdown("### Strategy Implementation Recommendation")
    st.markdown("""
    Based on this analysis, consider implementing a buffer rate adjustment strategy that:
    
    1. Adjusts buffer rates in response to significant changes in market spreads
    2. Uses different base buffer rates for major coins vs altcoins
    3. Makes incremental adjustments rather than large changes
    4. Implements a formula relating buffer rate to market spread with a safety multiplier
    """)
    
    formula_col1, formula_col2 = st.columns([2, 1])
    
    with formula_col1:
        st.latex(r"\text{Buffer Rate} = \text{Market Spread} \times \text{Safety Multiplier} + \text{Base Buffer}")
    
    with formula_col2:
        st.markdown("""
        Where:
        - Safety Multiplier = 10-15
        - Base Buffer depends on token type
        """)

def plot_rollbit_comparison(pair, days):
    """Compare your buffer rates with Rollbit's over time"""
    # Fetch Rollbit data
    query = f"""
    SELECT 
        created_at,
        pair_name,
        bust_buffer,
        position_multiplier
    FROM 
        rollbit_pair_config
    WHERE 
        pair_name = '{pair}'
        AND created_at > NOW() - INTERVAL '{days} days'
    ORDER BY 
        created_at
    """
    
    try:
        rollbit_df = pd.read_sql(query, engine)
        
        # Fetch SURF data
        query = f"""
        SELECT 
            updated_at as created_at,
            pair_name,
            buffer_rate,
            position_multiplier
        FROM 
            public.trade_pool_pairs
        WHERE 
            pair_name = '{pair}'
            AND updated_at > NOW() - INTERVAL '{days} days'
        ORDER BY 
            created_at
        """
        
        surf_df = pd.read_sql(query, engine)
        
        # Check if we have data
        if rollbit_df.empty:
            st.warning(f"No Rollbit data available for {pair}")
            return None
            
        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        
        # Plot buffer rates
        ax1.set_title(f"Buffer Rate Comparison for {pair}")
        
        if not rollbit_df.empty:
            # Convert to percent for better readability
            ax1.plot(rollbit_df['created_at'], rollbit_df['bust_buffer'] * 100, 'r-', 
                    label='Rollbit Buffer (%)')
        
        if not surf_df.empty:
            # Make sure dates are aligned properly
            ax1.plot(surf_df['created_at'], surf_df['buffer_rate'] * 100, 'b-', 
                    label='SURF Buffer (%)')
        else:
            # If no SURF data, add a note
            ax1.text(0.5, 0.5, "No SURF buffer rate data available", 
                     horizontalalignment='center', verticalalignment='center',
                     transform=ax1.transAxes, bbox=dict(facecolor='yellow', alpha=0.5))
        
        ax1.set_ylabel('Buffer Rate (%)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot position multipliers
        ax2.set_title(f"Position Multiplier Comparison for {pair}")
        
        if not rollbit_df.empty:
            ax2.plot(rollbit_df['created_at'], rollbit_df['position_multiplier'], 'r-', 
                    label='Rollbit Position Multiplier')
        
        if not surf_df.empty:
            ax2.plot(surf_df['created_at'], surf_df['position_multiplier'], 'b-', 
                    label='SURF Position Multiplier')
        else:
            # If no SURF data, add a note
            ax2.text(0.5, 0.5, "No SURF position multiplier data available", 
                     horizontalalignment='center', verticalalignment='center',
                     transform=ax2.transAxes, bbox=dict(facecolor='yellow', alpha=0.5))
        
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Position Multiplier')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add data information section below the plot
        if not rollbit_df.empty:
            rollbit_min = rollbit_df['bust_buffer'].min() * 100
            rollbit_max = rollbit_df['bust_buffer'].max() * 100
            rollbit_avg = rollbit_df['bust_buffer'].mean() * 100
            
            st.markdown(f"""
            ### Rollbit Buffer Rate Statistics
            - **Min Buffer Rate:** {rollbit_min:.3f}%
            - **Max Buffer Rate:** {rollbit_max:.3f}%
            - **Average Buffer Rate:** {rollbit_avg:.3f}%
            - **Number of Records:** {len(rollbit_df)}
            """)
        
        plt.tight_layout()
        return fig
    
    except Exception as e:
        st.error(f"Error fetching comparison data: {e}")
        return None

def render_rollbit_tab(days):
    st.markdown("## Rollbit Buffer Strategy Analysis")
    st.markdown("Analyzing Rollbit's buffer rate strategies to understand their adjustment approach")
    
    # Create tabs for different analysis types
    rollbit_tabs = st.tabs([
        "Individual Pair Analysis", 
        "Buffer Adjustment Patterns", 
        "Spread Correlation", 
        "Recommendations"
    ])
    
    # Tab 1: Individual Pair Analysis
    with rollbit_tabs[0]:
        st.markdown("### Compare Rollbit's Buffer Rate Strategies for Individual Pairs")
        
        # Fetch active pairs
        active_pairs = fetch_active_pairs()
        
        # Select a pair for Rollbit comparison
        rollbit_pair = st.selectbox(
            "Select a pair for Rollbit comparison",
            options=active_pairs,
            index=0 if active_pairs else None
        )
        
        if rollbit_pair:
            # Show historical comparison
            rollbit_fig = plot_rollbit_comparison(rollbit_pair, days)
            if rollbit_fig:
                st.pyplot(rollbit_fig)
            else:
                st.warning(f"No data available for {rollbit_pair}")
                
            # Show additional market data
            st.markdown("### Recent Market Data for Reference")
            
            # Fetch market spread data
            spread_df = fetch_historical_spreads(rollbit_pair, days=min(7, days))
            
            if spread_df is not None and not spread_df.empty:
                avg_spread = spread_df['avg_spread'].mean() * 10000  # Convert to basis points
                spread_volatility = spread_df['avg_spread'].std() * 10000
                
                st.markdown(f"**Average Market Spread:** {avg_spread:.2f} basis points")
                st.markdown(f"**Spread Volatility:** {spread_volatility:.2f} basis points")
            else:
                st.warning("No market spread data available")
    
    # Tab 2: Buffer Adjustment Patterns
    with rollbit_tabs[1]:
        st.markdown("### Rollbit's Buffer Rate Adjustment Patterns")
        
        # Analyze how Rollbit adjusts buffer rates
        adjustment_patterns = analyze_rollbit_buffer_adjustment_pattern(days)
        
        if adjustment_patterns is not None and not adjustment_patterns.empty:
            # Filter to most active pairs
            active_adjustments = adjustment_patterns[adjustment_patterns['buffer_adjustments_count'] > 1]
            
            if not active_adjustments.empty:
                # Sort by number of adjustments
                active_adjustments = active_adjustments.sort_values('buffer_adjustments_count', ascending=False)
                
                # Format for display
                display_df = active_adjustments.copy()
                display_df = display_df[['pair_name', 'buffer_adjustments_count', 'avg_days_between_adjustments', 
                                        'avg_buffer_change_pct', 'buffer_current', 'buffer_min', 'buffer_max']]
                
                display_df = display_df.rename(columns={
                    'pair_name': 'Pair',
                    'buffer_adjustments_count': 'Adjustments',
                    'avg_days_between_adjustments': 'Days Between',
                    'avg_buffer_change_pct': 'Avg Change (%)',
                    'buffer_current': 'Current (%)',
                    'buffer_min': 'Min (%)',
                    'buffer_max': 'Max (%)'
                })
                
                st.dataframe(display_df)
                
                # Create token type analysis
                token_analysis = analyze_rollbit_by_token_type(active_adjustments)
                
                if token_analysis is not None and not token_analysis.empty:
                    st.markdown("### Buffer Rate Strategies by Token Type")
                    
                    # Format token analysis for display
                    token_display = token_analysis.copy()
                    token_display = token_display.rename(columns={
                        'token_type': 'Token Type',
                        'buffer_current': 'Current Buffer (%)',
                        'avg_buffer_change_pct': 'Avg Change (%)',
                        'avg_days_between_adjustments': 'Days Between',
                        'buffer_min': 'Min Buffer (%)',
                        'buffer_max': 'Max Buffer (%)'
                    })
                    
                    st.dataframe(token_display)
                    
                    # Plot comparison of buffer rates by token type
                    if len(token_analysis) > 1:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        # Plot min, current, and max buffer rates by token type
                        token_types = token_analysis['token_type']
                        x = np.arange(len(token_types))
                        width = 0.25
                        
                        ax.bar(x - width, token_analysis['buffer_min'], width, label='Min Buffer', color='green', alpha=0.7)
                        ax.bar(x, token_analysis['buffer_current'], width, label='Current Buffer', color='blue', alpha=0.7)
                        ax.bar(x + width, token_analysis['buffer_max'], width, label='Max Buffer', color='red', alpha=0.7)
                        
                        ax.set_xlabel('Token Type')
                        ax.set_ylabel('Buffer Rate (%)')
                        ax.set_title('Buffer Rates by Token Type')
                        ax.set_xticks(x)
                        ax.set_xticklabels(token_types)
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        
                        st.pyplot(fig)
            else:
                st.warning("Not enough adjustment data to analyze patterns")
        else:
            st.warning("No adjustment data available for analysis")
    
    # Tab 3: Spread Correlation
    with rollbit_tabs[2]:
        st.markdown("### Correlation Between Rollbit's Buffer Rates and Market Spreads")
        
        # Analyze if Rollbit buffer rates correlate with market spreads
        buffer_spread_corr = fetch_rollbit_buffer_vs_spread(days)
        
        if buffer_spread_corr is not None and not buffer_spread_corr.empty:
            # Create a copy for display
            display_corr = buffer_spread_corr.copy()
            
            # Add absolute correlation for sorting
            display_corr['abs_correlation'] = display_corr['correlation'].abs()
            
            # Sort by absolute correlation
            display_corr = display_corr.sort_values('abs_correlation', ascending=False)
            
            # Format for display
            display_corr = display_corr[['pair_name', 'correlation', 'rollbit_buffer_avg', 'spread_avg', 'ratio', 'data_points']]
            display_corr = display_corr.rename(columns={
                'pair_name': 'Pair',
                'correlation': 'Correlation',
                'rollbit_buffer_avg': 'Rollbit Buffer (%)',
                'spread_avg': 'Market Spread (bps)',
                'ratio': 'Buffer/Spread Ratio',
                'data_points': 'Data Points'
            })
            
            st.dataframe(display_corr)
            
            # Plot correlation distribution
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Use a histogram instead of KDE for more stability
            ax.hist(buffer_spread_corr['correlation'], bins=10, alpha=0.7, color='blue', edgecolor='black')
            
            ax.set_xlabel('Correlation: Rollbit Buffer vs Market Spread')
            ax.set_ylabel('Count')
            ax.set_title('Distribution of Buffer-Spread Correlations')
            ax.axvline(x=0, color='r', linestyle='--')
            
            # Add mean line
            mean_corr = buffer_spread_corr['correlation'].mean()
            ax.axvline(x=mean_corr, color='g', linestyle='-', 
                     label=f'Mean: {mean_corr:.3f}')
            
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            # Add ratio analysis
            if 'ratio' in buffer_spread_corr.columns:
                valid_ratios = buffer_spread_corr['ratio'].dropna()
                
                if not valid_ratios.empty:
                    st.markdown("### Buffer Rate to Market Spread Ratio")
                    
                    avg_ratio = valid_ratios.mean()
                    median_ratio = valid_ratios.median()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Average Ratio", f"{avg_ratio:.2f}x")
                    
                    with col2:
                        st.metric("Median Ratio", f"{median_ratio:.2f}x")
                    
                    st.markdown("""
                    **Interpretation:** This ratio represents how many times larger Rollbit's buffer rates are
                    compared to market spreads. A higher ratio means more conservative margin requirements.
                    """)
                    
                    # Plot ratio distribution
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Filter extreme values for better visualization
                    plotable_ratios = valid_ratios[valid_ratios < valid_ratios.quantile(0.95)]
                    
                    ax.hist(plotable_ratios, bins=10, alpha=0.7, color='green', edgecolor='black')
                    ax.set_xlabel('Buffer/Spread Ratio')
                    ax.set_ylabel('Count')
                    ax.set_title('Distribution of Buffer to Spread Ratios')
                    ax.axvline(x=avg_ratio, color='r', linestyle='-', 
                             label=f'Mean: {avg_ratio:.2f}x')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
        else:
            st.warning("No correlation data available for analysis")
    
    # Tab 4: Strategy Recommendations
    with rollbit_tabs[3]:
        st.markdown("### Dynamic Buffer Rate Strategy Recommendations")
        
        # Display strategy summary and recommendations
        adjustment_patterns = analyze_rollbit_buffer_adjustment_pattern(days)
        buffer_spread_corr = fetch_rollbit_buffer_vs_spread(days)
        
        display_rollbit_strategy_summary(adjustment_patterns, buffer_spread_corr)
                
        # Add implementation suggestions
        st.markdown("### Implementation Example")
        
        st.code("""
# Example dynamic buffer rate adjustment function
def calculate_dynamic_buffer_rate(market_spread, token_type):
    # Base parameters
    safety_multiplier = 12.0  # Based on Rollbit's average ratio
    
    # Base buffer varies by token type
    if token_type == 'Major':
        base_buffer = 0.02  # 2%
    elif token_type == 'Stablecoin':
        base_buffer = 0.01  # 1%
    else:  # Altcoin
        base_buffer = 0.025  # 2.5%
    
    # Calculate buffer rate based on market spread
    buffer_rate = (market_spread * safety_multiplier) + base_buffer
    
    # Apply reasonable bounds
    buffer_rate = max(0.01, min(0.10, buffer_rate))
    
    return buffer_rate
        """, language="python")
        
        st.markdown("""
        This implementation would run:
        1. **Periodically** (e.g., every few hours) to check for significant spread changes
        2. **Incrementally** making small adjustments rather than large jumps
        3. **Differentially** across token types, with appropriate safety multipliers
        """)
        
        # Visualization of the strategy
        st.markdown("### Visualization of the Recommended Strategy")
        
        # Create example visualization
        spreads = np.linspace(0.0001, 0.005, 100)  # 1 to 50 basis points
        
        major_buffers = [(s * 12.0) + 0.02 for s in spreads]
        alt_buffers = [(s * 15.0) + 0.025 for s in spreads]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(spreads * 10000, np.array(major_buffers) * 100, 'b-', label='Major Tokens', linewidth=2)
        ax.plot(spreads * 10000, np.array(alt_buffers) * 100, 'r-', label='Altcoins', linewidth=2)
        
        ax.set_xlabel('Market Spread (basis points)')
        ax.set_ylabel('Buffer Rate (%)')
        ax.set_title('Recommended Dynamic Buffer Rate Strategy')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        st.pyplot(fig)

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
    tab1, tab2, tab3 = st.tabs(["Individual Pair Analysis", "Summary Results", "Rollbit Analysis"])
    
    # Individual Pair Analysis Tab
    with tab1:
        render_individual_analysis_tab(selected_pairs, days, vol_window, min_data_points)
    
    # Summary Results Tab
    with tab2:
        st.markdown("## Summary Results")
        st.markdown("Comparison of spread-volatility relationships across all analyzed pairs")
        
        # Process all pairs to get results
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
            fig, ax = plt.subplots(figsize=(10, 6))
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
                fig, ax = plt.subplots(figsize=(10, 6))
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
                st.write("**Correlation Strength:** Moderate correlation between spreads and volatility")
            elif avg_abs_corr > 0.3:
                st.write("**Correlation Strength:** Weak correlation between spreads and volatility")
            else:
                st.write("**Correlation Strength:** Very weak correlation between spreads and volatility")
                
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
    spread_multiplier = 12.0  # Based on Rollbit's average ratio
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
    
    # Rollbit Analysis Tab        
    with tab3:
        render_rollbit_tab(days)

if __name__ == "__main__":
    main()