import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import pytz
import statsmodels.api as sm
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

# Fetch historical spread data
@st.cache_data(ttl=600)
def fetch_historical_spreads(pair_name, days=7):
    """Fetch historical spread data for a given pair"""
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
    ORDER BY 
        time_group
    """
    
    try:
        df = pd.read_sql(query, engine)
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

# Calculate historical volatility
@st.cache_data(ttl=600)
def fetch_historical_prices(pair_name, days=7):
    """Fetch historical price data for volatility calculation"""
    # Calculate time range
    end_time = datetime.now(pytz.utc)
    start_time = end_time - timedelta(days=days)
    
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
        if df.empty:
            return None
            
        # Convert created_at to datetime
        df['timestamp'] = pd.to_datetime(df['created_at'])
        
        # Resample to hourly intervals
        df['hour'] = df['timestamp'].dt.floor('H')
        hourly_df = df.groupby('hour').agg({
            'deal_price': ['first', 'last', 'mean', 'max', 'min', 'count']
        })
        
        # Flatten multi-index columns
        hourly_df.columns = ['_'.join(col).strip() for col in hourly_df.columns.values]
        hourly_df = hourly_df.reset_index()
        
        return hourly_df
    except Exception as e:
        st.error(f"Error fetching historical prices for {pair_name}: {e}")
        return None

def calculate_volatility(price_df, window=24):
    """Calculate rolling volatility from price data"""
    if price_df is None or len(price_df) < 2:
        return None
    
    # Calculate returns
    price_df['log_return'] = np.log(price_df['deal_price_last'] / price_df['deal_price_last'].shift(1))
    
    # Calculate rolling volatility (annualized)
    price_df['volatility_1h'] = price_df['log_return'].rolling(window=1).std() * np.sqrt(24 * 365)
    price_df['volatility_24h'] = price_df['log_return'].rolling(window=window).std() * np.sqrt(24 * 365)
    
    # Fill first value with a reasonable estimate
    price_df['volatility_1h'].fillna(method='bfill', inplace=True)
    price_df['volatility_24h'].fillna(price_df['volatility_1h'], inplace=True)
    
    return price_df

def combine_spread_volatility_data(spread_df, vol_df):
    """Combine spread and volatility data into a single dataframe"""
    if spread_df is None or vol_df is None:
        return None
    
    # Ensure the indices are aligned
    spread_df = spread_df.reset_index()
    
    # Merge on the hour
    merged_df = pd.merge(
        spread_df,
        vol_df[['hour', 'volatility_1h', 'volatility_24h', 'log_return', 'deal_price_mean']], 
        left_on='hour', 
        right_on='hour',
        how='inner'
    )
    
    return merged_df

def analyze_relationship(combined_df):
    """Analyze the relationship between spreads and volatility"""
    if combined_df is None or len(combined_df) < 10:
        return None
    
    # Create correlation matrix
    corr_columns = ['avg_spread', 'volatility_1h', 'volatility_24h']
    available_columns = [col for col in corr_columns if col in combined_df.columns]
    
    if len(available_columns) < 2:
        return None
    
    correlation = combined_df[available_columns].corr()
    
    # Simple linear regression
    X = combined_df['volatility_24h'].values.reshape(-1, 1)
    y = combined_df['avg_spread'].values
    
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
            correlation = spread_shifted.corr(combined_df['volatility_24h'])
            leads_lags.append({
                'lag': lag,
                'correlation': correlation,
                'description': f"Spread leads volatility by {-lag} hours"
            })
        else:
            # Volatility leads spread
            vol_shifted = combined_df['volatility_24h'].shift(lag)
            correlation = combined_df['avg_spread'].corr(vol_shifted)
            leads_lags.append({
                'lag': lag,
                'correlation': correlation,
                'description': f"Volatility leads spread by {lag} hours"
            })
    
    leads_lags_df = pd.DataFrame(leads_lags)
    
    # Find the lag with the highest correlation
    best_lag = leads_lags_df.loc[leads_lags_df['correlation'].abs().idxmax()]
    
    return {
        'correlation': correlation,
        'regression_coefficient': model.coef_[0],
        'r2': r2,
        'lead_lag': leads_lags_df,
        'best_lag': best_lag
    }

def plot_spread_volatility(combined_df, pair_name):
    """Plot spread vs. volatility relationship"""
    if combined_df is None or len(combined_df) < 5:
        st.warning(f"Not enough data for {pair_name} to create plots")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # First subplot - Time series
    ax1.set_title(f"Spread vs. Volatility for {pair_name}")
    ax1.plot(combined_df['hour'], combined_df['avg_spread'] * 10000, 'b-', label='Spread (bps)')
    ax1.set_ylabel('Spread (basis points)')
    ax1.legend(loc='upper left')
    
    ax3 = ax1.twinx()
    ax3.plot(combined_df['hour'], combined_df['volatility_24h'] * 100, 'r-', label='Volatility (%)')
    ax3.set_ylabel('Annualized Volatility (%)')
    ax3.legend(loc='upper right')
    
    # Second subplot - Scatter plot
    ax2.scatter(combined_df['volatility_24h'] * 100, combined_df['avg_spread'] * 10000, alpha=0.5)
    ax2.set_xlabel('Annualized Volatility (%)')
    ax2.set_ylabel('Spread (basis points)')
    ax2.set_title('Spread vs. Volatility Correlation')
    
    # Add regression line
    X = combined_df['volatility_24h'].values.reshape(-1, 1)
    y = combined_df['avg_spread'].values
    
    try:
        model = LinearRegression()
        model.fit(X, y)
        
        x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_pred = model.predict(x_range)
        
        ax2.plot(x_range * 100, y_pred * 10000, 'r-', 
                 label=f'y = {model.coef_[0]:.6f}x + {model.intercept_:.6f}')
        ax2.legend()
    except:
        pass
    
    plt.tight_layout()
    return fig

def plot_lead_lag(lead_lag_df, pair_name):
    """Plot lead-lag relationship"""
    if lead_lag_df is None or lead_lag_df.empty:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(lead_lag_df['lag'], lead_lag_df['correlation'])
    ax.set_xlabel('Lag (hours)')
    ax.set_ylabel('Correlation')
    ax.set_title(f'Lead-Lag Relationship for {pair_name}')
    ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='r', linestyle='-', alpha=0.3)
    
    # Add best lag annotation
    best_lag = lead_lag_df.loc[lead_lag_df['correlation'].abs().idxmax()]
    ax.annotate(f"Best lag: {best_lag['lag']} hours\nCorr: {best_lag['correlation']:.3f}",
                xy=(best_lag['lag'], best_lag['correlation']),
                xytext=(0, 20),
                textcoords='offset points',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3'),
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3))
    
    plt.tight_layout()
    return fig

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
        if rollbit_df.empty:
            st.warning(f"No Rollbit data available for {pair}")
            return None
            
        # Fetch your data
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
        if rollbit_df.empty and surf_df.empty:
            return None
        
        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        
        # Plot buffer rates
        ax1.set_title(f"Buffer Rate Comparison for {pair}")
        
        if not rollbit_df.empty:
            ax1.plot(rollbit_df['created_at'], rollbit_df['bust_buffer'] * 100, 'r-', 
                    label='Rollbit Buffer (%)')
        
        if not surf_df.empty:
            ax1.plot(surf_df['created_at'], surf_df['buffer_rate'] * 100, 'b-', 
                    label='SURF Buffer (%)')
        
        ax1.set_ylabel('Buffer Rate (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot position multipliers
        ax2.set_title(f"Position Multiplier Comparison for {pair}")
        
        if not rollbit_df.empty:
            ax2.plot(rollbit_df['created_at'], rollbit_df['position_multiplier'], 'r-', 
                    label='Rollbit Position Multiplier')
        
        if not surf_df.empty:
            ax2.plot(surf_df['created_at'], surf_df['position_multiplier'], 'b-', 
                    label='SURF Position Multiplier')
        
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Position Multiplier')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    except Exception as e:
        st.error(f"Error fetching comparison data: {e}")
        return None

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
                    'data_points': len(group)
                })
        
        return pd.DataFrame(correlations)
    
    except Exception as e:
        st.error(f"Error analyzing Rollbit buffer vs spread: {e}")
        return None

def fetch_rollbit_buffer_vs_volatility(days=30):
    """Analyze if Rollbit buffer rates correlate with market volatility"""
    # This would require fetching volatility data as well
    st.warning("This feature is under development")
    return None

def consolidate_results(results_dict):
    """Consolidate analysis results into a summary dataframe"""
    summary_data = []
    
    for pair, result in results_dict.items():
        if result and 'correlation' in result:
            try:
                spread_vol_corr = result['correlation'].loc['avg_spread', 'volatility_24h']
                best_lag = result['best_lag']['lag']
                best_lag_corr = result['best_lag']['correlation']
                r2 = result['r2']
                
                summary_data.append({
                    'pair_name': pair,
                    'spread_volatility_correlation': spread_vol_corr,
                    'best_lag_hours': best_lag,
                    'best_lag_correlation': best_lag_corr,
                    'r2_score': r2,
                })
            except:
                pass
    
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
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["Individual Pair Analysis", "Summary Results", "Rollbit Analysis"])
    
    with tab1:
        st.markdown("## Individual Pair Analysis")
        st.markdown(f"Analyzing the relationship between market spreads and volatility over {selected_timeframe}")
        
        # Process each selected pair
        results_dict = {}
        
        for pair in selected_pairs:
            st.markdown(f"### {pair}")
            
            # Fetch data
            with st.spinner(f"Fetching data for {pair}..."):
                # Get historical spread data
                spread_df = fetch_historical_spreads(pair, days=days)
                
                # Get historical price data for volatility
                price_df = fetch_historical_prices(pair, days=days)
                
                if price_df is not None and not price_df.empty:
                    vol_df = calculate_volatility(price_df)
                else:
                    vol_df = None
                
                # Combine data
                if spread_df is not None and vol_df is not None:
                    combined_df = combine_spread_volatility_data(spread_df, vol_df)
                    
                    # Analyze relationship
                    results = analyze_relationship(combined_df)
                    results_dict[pair] = results
                    
                    # Plot relationship
                    fig = plot_spread_volatility(combined_df, pair)
                    if fig:
                        st.pyplot(fig)
                    
                    # Plot lead-lag relationship
                    if results and 'lead_lag' in results:
                        lead_lag_fig = plot_lead_lag(results['lead_lag'], pair)
                        if lead_lag_fig:
                            st.pyplot(lead_lag_fig)
                    
                    # Summary stats
                    if results and 'correlation' in results:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Correlation Matrix**")
                            st.dataframe(results['correlation'])
                        
                        with col2:
                            st.markdown("**Regression Results**")
                            st.write(f"Coefficient: {results['regression_coefficient']:.6f}")
                            st.write(f"R² Score: {results['r2']:.4f}")
                            st.write(f"Best lag: {results['best_lag']['description']}")
                            st.write(f"Lead-lag correlation: {results['best_lag']['correlation']:.4f}")
                else:
                    st.warning(f"Not enough data available for {pair}")
    
    with tab2:
        st.markdown("## Summary Results")
        st.markdown("Comparison of spread-volatility relationships across all analyzed pairs")
        
        # Consolidate results
        summary_df = consolidate_results(results_dict)
        
        if not summary_df.empty:
            # Sort by correlation strength
            summary_df = summary_df.sort_values(by='spread_volatility_correlation', 
                                              key=abs, ascending=False)
            
            # Display summary table
            st.dataframe(summary_df)
            
            # Create correlation distribution plot
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(summary_df['spread_volatility_correlation'], kde=True, ax=ax)
            ax.set_xlabel('Correlation: Spread vs. Volatility')
            ax.set_ylabel('Count')
            ax.set_title('Distribution of Spread-Volatility Correlations Across Pairs')
            ax.axvline(x=0, color='r', linestyle='--')
            ax.axvline(x=summary_df['spread_volatility_correlation'].mean(), 
                     color='g', linestyle='-', label=f'Mean: {summary_df["spread_volatility_correlation"].mean():.3f}')
            ax.legend()
            st.pyplot(fig)
            
            # Create lag distribution plot
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(summary_df['best_lag_hours'], kde=True, bins=10, ax=ax)
            ax.set_xlabel('Best Lag (hours)')
            ax.set_ylabel('Count')
            ax.set_title('Distribution of Lead-Lag Relationships')
            ax.axvline(x=0, color='r', linestyle='--')
            ax.axvline(x=summary_df['best_lag_hours'].mean(), 
                     color='g', linestyle='-', label=f'Mean: {summary_df["best_lag_hours"].mean():.1f} hours')
            ax.legend()
            st.pyplot(fig)
            
            # Overall findings
            st.markdown("### Key Findings")
            
            # Calculate aggregate statistics
            avg_corr = summary_df['spread_volatility_correlation'].mean()
            abs_avg_corr = summary_df['spread_volatility_correlation'].abs().mean()
            avg_lag = summary_df['best_lag_hours'].mean()
            
            st.write(f"**Average Correlation:** {avg_corr:.4f} (Absolute: {abs_avg_corr:.4f})")
            st.write(f"**Average Optimal Lag:** {avg_lag:.2f} hours")
            
            # Interpret results
            if avg_lag < -1:
                st.write(f"**Overall Finding:** Spread changes typically precede volatility changes by {-avg_lag:.1f} hours")
            elif avg_lag > 1:
                st.write(f"**Overall Finding:** Volatility changes typically precede spread changes by {avg_lag:.1f} hours")
            else:
                st.write("**Overall Finding:** Spread and volatility changes occur nearly simultaneously")
            
            if abs_avg_corr > 0.7:
                st.write("**Correlation Strength:** Very strong correlation between spreads and volatility")
            elif abs_avg_corr > 0.5:
                st.write("**Correlation Strength:** Moderate correlation between spreads and volatility")
            elif abs_avg_corr > 0.3:
                st.write("**Correlation Strength:** Weak correlation between spreads and volatility")
            else:
                st.write("**Correlation Strength:** Very weak correlation between spreads and volatility")
        else:
            st.warning("No results available for summary. Please analyze some pairs first.")
            
    with tab3:
        st.markdown("## Rollbit Analysis")
        st.markdown("Analyzing Rollbit's buffer rate and position multiplier strategies")
        
        # Select a pair for Rollbit comparison
        rollbit_pair = st.selectbox(
            "Select a pair for Rollbit comparison",
            options=all_pairs,
            index=0 if all_pairs else None
        )
        
        if rollbit_pair:
            # Show historical comparison
            rollbit_fig = plot_rollbit_comparison(rollbit_pair, days)
            if rollbit_fig:
                st.pyplot(rollbit_fig)
            
            # Analyze if Rollbit buffer rates correlate with market spreads
            st.markdown("### Rollbit Buffer Rate vs Market Spread Correlation")
            buffer_spread_corr = fetch_rollbit_buffer_vs_spread(days)
            
            if buffer_spread_corr is not None and not buffer_spread_corr.empty:
                buffer_spread_corr = buffer_spread_corr.sort_values(
                    by='correlation', key=abs, ascending=False)
                st.dataframe(buffer_spread_corr)
                
                # Plot correlation distribution
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(buffer_spread_corr['correlation'], kde=True, ax=ax)
                ax.set_xlabel('Correlation: Rollbit Buffer vs Market Spread')
                ax.set_ylabel('Count')
                ax.set_title('Distribution of Buffer-Spread Correlations')
                ax.axvline(x=0, color='r', linestyle='--')
                ax.axvline(x=buffer_spread_corr['correlation'].mean(), 
                         color='g', linestyle='-', 
                         label=f'Mean: {buffer_spread_corr["correlation"].mean():.3f}')
                ax.legend()
                st.pyplot(fig)
                
                # Interpretation
                avg_corr = buffer_spread_corr['correlation'].mean()
                abs_avg_corr = buffer_spread_corr['correlation'].abs().mean()
                
                st.write(f"**Average Correlation:** {avg_corr:.4f}")
                st.write(f"**Average Absolute Correlation:** {abs_avg_corr:.4f}")
                
                if abs_avg_corr > 0.7:
                    st.write("**Finding:** Rollbit's buffer rates are strongly correlated with market spreads")
                elif abs_avg_corr > 0.5:
                    st.write("**Finding:** Rollbit's buffer rates are moderately correlated with market spreads")
                elif abs_avg_corr > 0.3:
                    st.write("**Finding:** Rollbit's buffer rates are weakly correlated with market spreads")
                else:
                    st.write("**Finding:** Rollbit's buffer rates show very little correlation with market spreads")
            else:
                st.warning("No data available for Rollbit buffer vs spread analysis")
            
            # Analysis of Rollbit's buffer rates vs volatility would go here
            st.markdown("### Rollbit Buffer Rate vs Market Volatility Correlation")
            fetch_rollbit_buffer_vs_volatility(days)
            
        else:
            st.warning("Please select a pair for Rollbit comparison")

if __name__ == "__main__":
    main()