import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import pytz
try:
    import pkg_resources
except ImportError:
    import importlib.metadata as pkg_resources

st.set_page_config(
    page_title="JUP Bollinger Bands Strategy Analyzer",
    page_icon="📈",
    layout="wide"
)

# --- DB CONFIG ---
try:
    db_config = st.secrets["database"]
    db_uri = (
        f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}"
        f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    )
    engine = create_engine(
        db_uri,
        isolation_level="AUTOCOMMIT",  # 设置自动提交模式
        pool_size=5,  # 连接池大小
        max_overflow=10,  # 最大溢出连接数
        pool_timeout=30,  # 连接超时时间
        pool_recycle=1800,  # 连接回收时间(30分钟)
        pool_pre_ping=True,  # 使用连接前先测试连接是否有效
        pool_use_lifo=True,  # 使用后进先出,减少空闲连接
        echo=False  # 不打印 SQL 语句
    )
except Exception as e:
    st.error(f"Error connecting to the database: {e}")
    st.stop()

# --- UI Setup ---
st.title("Jupiter (JUP) Bollinger Bands Trading Analysis")
st.subheader("Short-Term Trading Opportunity Scanner")

# Define parameters
singapore_timezone = pytz.timezone('Asia/Singapore')

# Get current time in Singapore timezone
now_utc = datetime.now(pytz.utc)
now_sg = now_utc.astimezone(singapore_timezone)
st.write(f"Current Singapore Time: {now_sg.strftime('%Y-%m-%d %H:%M:%S')}")

# Add a debug mode checkbox
debug_mode = st.sidebar.checkbox("Debug Mode", value=False)

# Fetch all available tokens from DB
@st.cache_data(show_spinner="Fetching tokens...")
def fetch_all_tokens():
    query = "SELECT DISTINCT pair_name FROM public.oracle_price_log ORDER BY pair_name"
    try:
        df = pd.read_sql(query, engine)
        if df.empty:
            st.error("No tokens found in the database.")
            return []
        return df['pair_name'].tolist()
    except Exception as e:
        st.error(f"Error fetching tokens: {e}")
        return ["JUP", "BTC", "ETH", "SOL", "DOGE", "PEPE", "AI16Z"]  # Default fallback with JUP at the top

all_tokens = fetch_all_tokens()

# UI Controls for analysis parameters
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    # Let user select tokens to analyze (with JUP preselected)
    default_token = "JUP" if "JUP" in all_tokens else all_tokens[0]
    selected_token = st.selectbox(
        "Select Token to Analyze", 
        all_tokens,
        index=all_tokens.index(default_token) if default_token in all_tokens else 0
    )

with col2:
    # Bollinger Bands parameters
    st.subheader("Bollinger Parameters")
    bb_period = st.slider("BB Period", min_value=5, max_value=50, value=20, step=1)
    bb_std_dev = st.slider("BB Standard Deviations", min_value=1.0, max_value=4.0, value=2.0, step=0.1)

with col3:
    # Backtest parameters
    st.subheader("Time Parameters")
    
    timeframe_options = ["Days", "Hours", "Minutes"]
    timeframe_type = st.selectbox("Lookback Unit", timeframe_options, index=1)  # Default to Hours
    
    if timeframe_type == "Days":
        lookback_value = st.slider("Lookback Days", min_value=1, max_value=30, value=3, step=1)
        lookback_minutes = lookback_value * 1440  # Convert to minutes
    elif timeframe_type == "Hours":
        lookback_value = st.slider("Lookback Hours", min_value=1, max_value=48, value=24, step=1)
        lookback_minutes = lookback_value * 60  # Convert to minutes
    else:  # Minutes
        lookback_value = st.slider("Lookback Minutes", min_value=15, max_value=1440, value=240, step=15)
        lookback_minutes = lookback_value
    
    # Candle timeframe for analysis
    candle_options = ["1m", "3m", "5m", "15m", "30m", "1h"]
    candle_timeframe = st.selectbox("Analysis Timeframe", candle_options, index=0)  # Default to 1-minute

# Time filtering options
with st.expander("Time of Day Filtering (Singapore Time)"):
    apply_time_filter = st.checkbox("Filter by Time of Day", value=True)
    if apply_time_filter:
        col1, col2 = st.columns(2)
        with col1:
            start_hour = st.slider("Start Hour (24h)", min_value=0, max_value=23, value=11, step=1)
        with col2:
            end_hour = st.slider("End Hour (24h)", min_value=0, max_value=23, value=17, step=1)
        
        st.info(f"Analyzing trades between {start_hour}:00 and {end_hour}:00 Singapore time")

# Strategy parameters
strategy_options = {
    "BB Bounce": "Trade when price bounces off the bands",
    "BB Breakout": "Trade when price breaks through the bands",
    "BB Squeeze": "Trade when bands narrow and then expand",
    "All Strategies": "Test all strategies and show best"
}

selected_strategy = st.selectbox("Strategy Type", list(strategy_options.keys()), index=0)  # Default to BB Bounce
st.markdown(strategy_options[selected_strategy])

# Add trade duration analysis options
with st.expander("Trade Settings"):
    col1, col2 = st.columns(2)
    with col1:
        max_hold_time = st.slider("Max Hold Time (minutes)", min_value=1, max_value=60, value=5, step=1)
    with col2:
        take_profit_pct = st.slider("Take Profit (%)", min_value=0.1, max_value=10.0, value=3.0, step=0.1)
        stop_loss_pct = st.slider("Stop Loss (%)", min_value=0.1, max_value=10.0, value=1.5, step=0.1)

# Add a refresh button
if st.button("Refresh Data"):
    st.cache_data.clear()
    st.experimental_rerun()

# Function to calculate Bollinger Bands
def calculate_bollinger_bands(df, period=20, std_dev=2.0):
    """
    Calculate Bollinger Bands for the provided dataframe
    """
    df['bb_middle'] = df['price'].rolling(window=period).mean()
    rolling_std = df['price'].rolling(window=period).std(ddof=0)
    df['bb_upper'] = df['bb_middle'] + (rolling_std * std_dev)
    df['bb_lower'] = df['bb_middle'] - (rolling_std * std_dev)
    
    # Calculate bandwidth
    df['bb_bandwidth'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # Calculate %B (where price is within the bands)
    df['bb_percent_b'] = (df['price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Calculate rate of change for bandwidth (for squeeze detection)
    df['bandwidth_change'] = df['bb_bandwidth'].pct_change(periods=3)
    
    # Calculate rate of change for price
    df['price_change_pct'] = df['price'].pct_change() * 100
    
    return df

# Define trading strategies with specific focus on short-term trades
def bb_bounce_strategy(df, take_profit_pct=3.0, stop_loss_pct=1.5, max_hold_time=5):
    """
    Strategy: Buy when price touches lower band, sell when price bounces back
    Enhanced for shorter timeframes with take profit, stop loss, and max hold time
    """
    # Initialize columns
    df['position'] = 0
    df['signal'] = 0
    df['entry_price'] = np.nan
    df['exit_price'] = np.nan
    df['trade_duration'] = np.nan  # in bars
    df['exit_reason'] = ""
    df['trade_profit_pct'] = np.nan
    
    # Generate buy signals (1) when price touches or goes below the lower band
    # AND price is near a support level (indicated by previous lows)
    lower_band_touch = df['price'] <= df['bb_lower']
    
    # Simple support detection: price is within 1% of recent lows
    window_size = min(20, len(df))
    df['recent_min'] = df['price'].rolling(window=window_size).min()
    near_support = (df['price'] / df['recent_min'] - 1) < 0.01
    
    # Oversold condition using %B
    oversold = df['bb_percent_b'] < 0.05
    
    # Combine conditions
    df.loc[lower_band_touch & near_support & oversold, 'signal'] = 1
    
    # Track trades with take profit, stop loss and max hold time
    in_position = False
    entry_index = None
    entry_price = None
    position_count = 0
    
    for i in range(1, len(df)):
        idx = df.index[i]
        prev_idx = df.index[i-1]
        
        # New buy signal
        if df.loc[idx, 'signal'] == 1 and not in_position:
            in_position = True
            entry_index = i
            entry_price = df.loc[idx, 'price']
            df.loc[idx, 'position'] = 1
            df.loc[idx, 'entry_price'] = entry_price
            position_count += 1
            
        # Check for exit conditions if in a position
        elif in_position:
            current_price = df.loc[idx, 'price']
            bars_in_trade = i - entry_index
            
            # Calculate current P&L for this position
            current_profit_pct = (current_price / entry_price - 1) * 100
            
            # Exit conditions
            take_profit_hit = current_profit_pct >= take_profit_pct
            stop_loss_hit = current_profit_pct <= -stop_loss_pct
            max_hold_reached = bars_in_trade >= max_hold_time
            middle_band_reached = current_price >= df.loc[idx, 'bb_middle']
            
            # Determine if we should exit
            exit_triggered = take_profit_hit or stop_loss_hit or max_hold_reached or middle_band_reached
            
            if exit_triggered:
                # Record exit information
                df.loc[idx, 'position'] = 0  # Exit position
                df.loc[idx, 'exit_price'] = current_price
                df.loc[idx, 'trade_duration'] = bars_in_trade
                df.loc[idx, 'trade_profit_pct'] = current_profit_pct
                
                # Record exit reason
                if take_profit_hit:
                    df.loc[idx, 'exit_reason'] = "Take Profit"
                elif stop_loss_hit:
                    df.loc[idx, 'exit_reason'] = "Stop Loss"
                elif max_hold_reached:
                    df.loc[idx, 'exit_reason'] = "Max Hold Time"
                elif middle_band_reached:
                    df.loc[idx, 'exit_reason'] = "Middle Band"
                
                # Reset position tracking
                in_position = False
                entry_index = None
                entry_price = None
            else:
                # Still in the position
                df.loc[idx, 'position'] = 1
        
        # Propagate position status if no changes
        elif df.loc[prev_idx, 'position'] == 1 and not in_position:
            # This handles cases where we have position=1 from previous logic but not tracked in our loop
            in_position = True
            entry_index = i-1
            entry_price = df.loc[prev_idx, 'price']
    
    return df

def bb_breakout_strategy(df, take_profit_pct=3.0, stop_loss_pct=1.5, max_hold_time=5):
    """
    Strategy: Buy when price breaks above upper band with momentum, sell on reversal
    Enhanced for shorter timeframes with take profit, stop loss, and max hold time
    """
    # Initialize columns
    df['position'] = 0
    df['signal'] = 0
    df['entry_price'] = np.nan
    df['exit_price'] = np.nan
    df['trade_duration'] = np.nan
    df['exit_reason'] = ""
    df['trade_profit_pct'] = np.nan
    
    # Generate buy signals when price breaks above the upper band with momentum
    # Momentum confirmed by increasing volume and price acceleration
    upper_band_break = df['price'] > df['bb_upper']
    
    # Price momentum: current bar gain is greater than previous bar
    df['price_momentum'] = df['price_change_pct'] > df['price_change_pct'].shift(1)
    
    # Combine conditions
    df.loc[upper_band_break & df['price_momentum'], 'signal'] = 1
    
    # Track trades with take profit, stop loss and max hold time
    in_position = False
    entry_index = None
    entry_price = None
    
    for i in range(1, len(df)):
        idx = df.index[i]
        prev_idx = df.index[i-1]
        
        # New buy signal
        if df.loc[idx, 'signal'] == 1 and not in_position:
            in_position = True
            entry_index = i
            entry_price = df.loc[idx, 'price']
            df.loc[idx, 'position'] = 1
            df.loc[idx, 'entry_price'] = entry_price
            
        # Check for exit conditions if in a position
        elif in_position:
            current_price = df.loc[idx, 'price']
            bars_in_trade = i - entry_index
            
            # Calculate current P&L for this position
            current_profit_pct = (current_price / entry_price - 1) * 100
            
            # Exit conditions
            take_profit_hit = current_profit_pct >= take_profit_pct
            stop_loss_hit = current_profit_pct <= -stop_loss_pct
            max_hold_reached = bars_in_trade >= max_hold_time
            momentum_reversal = df.loc[idx, 'price'] < df.loc[idx, 'bb_upper'] and df.loc[prev_idx, 'price'] >= df.loc[prev_idx, 'bb_upper']
            
            # Determine if we should exit
            exit_triggered = take_profit_hit or stop_loss_hit or max_hold_reached or momentum_reversal
            
            if exit_triggered:
                # Record exit information
                df.loc[idx, 'position'] = 0  # Exit position
                df.loc[idx, 'exit_price'] = current_price
                df.loc[idx, 'trade_duration'] = bars_in_trade
                df.loc[idx, 'trade_profit_pct'] = current_profit_pct
                
                # Record exit reason
                if take_profit_hit:
                    df.loc[idx, 'exit_reason'] = "Take Profit"
                elif stop_loss_hit:
                    df.loc[idx, 'exit_reason'] = "Stop Loss"
                elif max_hold_reached:
                    df.loc[idx, 'exit_reason'] = "Max Hold Time"
                elif momentum_reversal:
                    df.loc[idx, 'exit_reason'] = "Momentum Reversal"
                
                # Reset position tracking
                in_position = False
                entry_index = None
                entry_price = None
            else:
                # Still in the position
                df.loc[idx, 'position'] = 1
        
        # Propagate position status if no changes
        elif df.loc[prev_idx, 'position'] == 1 and not in_position:
            in_position = True
            entry_index = i-1
            entry_price = df.loc[prev_idx, 'price']
    
    return df

def bb_squeeze_strategy(df, take_profit_pct=3.0, stop_loss_pct=1.5, max_hold_time=5):
    """
    Strategy: Buy when bands narrow and then expand with price moving upward
    Enhanced for shorter timeframes with take profit, stop loss, and max hold time
    """
    # Initialize columns
    df['position'] = 0
    df['signal'] = 0
    df['entry_price'] = np.nan
    df['exit_price'] = np.nan
    df['trade_duration'] = np.nan
    df['exit_reason'] = ""
    df['trade_profit_pct'] = np.nan
    
    # Define squeeze condition (when bandwidth is in bottom 20% of its range over lookback period)
    lookback = min(len(df), 100)
    
    # Safely calculate bandwidth percentile - handle empty dataframes
    if len(df) > lookback:
        df['bandwidth_percentile'] = df['bb_bandwidth'].rolling(window=lookback).apply(
            lambda x: np.percentile(x, 20) if len(x) > 0 else np.nan
        )
    else:
        # If we don't have enough data for the lookback, use a smaller window
        safe_lookback = max(5, len(df) // 2)
        df['bandwidth_percentile'] = df['bb_bandwidth'].rolling(window=safe_lookback).apply(
            lambda x: np.percentile(x, 20) if len(x) > 0 else np.nan
        )
    
    # Identify squeeze - handle NaN values safely
    df['in_squeeze'] = (df['bb_bandwidth'] <= df['bandwidth_percentile']).fillna(False)
    
    # Identify squeeze exit with upward momentum
    df['squeeze_exit_long'] = (df['in_squeeze'].shift(1).fillna(False) & 
                              ~df['in_squeeze'] & 
                              (df['price'] > df['bb_middle']) & 
                              (df['price_change_pct'] > 0))
    
    # Generate signals
    df.loc[df['squeeze_exit_long'], 'signal'] = 1
    
    # Track trades with take profit, stop loss and max hold time
    in_position = False
    entry_index = None
    entry_price = None
    
    for i in range(1, len(df)):
        idx = df.index[i]
        prev_idx = df.index[i-1]
        
        # New buy signal
        if df.loc[idx, 'signal'] == 1 and not in_position:
            in_position = True
            entry_index = i
            entry_price = df.loc[idx, 'price']
            df.loc[idx, 'position'] = 1
            df.loc[idx, 'entry_price'] = entry_price
            
        # Check for exit conditions if in a position
        elif in_position:
            current_price = df.loc[idx, 'price']
            bars_in_trade = i - entry_index
            
            # Calculate current P&L for this position
            current_profit_pct = (current_price / entry_price - 1) * 100
            
            # Exit conditions
            take_profit_hit = current_profit_pct >= take_profit_pct
            stop_loss_hit = current_profit_pct <= -stop_loss_pct
            max_hold_reached = bars_in_trade >= max_hold_time
            upper_band_reached = current_price >= df.loc[idx, 'bb_upper']
            
            # Determine if we should exit
            exit_triggered = take_profit_hit or stop_loss_hit or max_hold_reached or upper_band_reached
            
            if exit_triggered:
                # Record exit information
                df.loc[idx, 'position'] = 0  # Exit position
                df.loc[idx, 'exit_price'] = current_price
                df.loc[idx, 'trade_duration'] = bars_in_trade
                df.loc[idx, 'trade_profit_pct'] = current_profit_pct
                
                # Record exit reason
                if take_profit_hit:
                    df.loc[idx, 'exit_reason'] = "Take Profit"
                elif stop_loss_hit:
                    df.loc[idx, 'exit_reason'] = "Stop Loss"
                elif max_hold_reached:
                    df.loc[idx, 'exit_reason'] = "Max Hold Time"
                elif upper_band_reached:
                    df.loc[idx, 'exit_reason'] = "Upper Band"
                
                # Reset position tracking
                in_position = False
                entry_index = None
                entry_price = None
            else:
                # Still in the position
                df.loc[idx, 'position'] = 1
        
        # Propagate position status if no changes
        elif df.loc[prev_idx, 'position'] == 1 and not in_position:
            in_position = True
            entry_index = i-1
            entry_price = df.loc[prev_idx, 'price']
    
    return df

# Calculate strategy returns - FIXED VERSION
 # This is the key function that needs to be replaced in your code
def calculate_returns(df):
    """
    Calculate returns and equity curve directly from completed trades
    """
    # Filter for completed trades (those with both entry and exit prices)
    completed_trades = df[(df['entry_price'].notna()) & (df['exit_price'].notna())].copy()
    
    if completed_trades.empty:
        df['trade_return'] = 0.0
        df['cum_strategy_return'] = 0.0
        return df
    
    # Initialize the trade return column for all rows
    df['trade_return'] = 0.0
    
    # Track cumulative returns manually
    initial_capital = 1.0
    current_capital = initial_capital
    returns_by_time = {}
    
    # Process each completed trade sequentially
    for idx, row in completed_trades.iterrows():
        # Calculate return for this trade (already in percent format)
        trade_profit_pct = row['trade_profit_pct']
        
        # Convert percentage to decimal for calculation
        trade_return = trade_profit_pct / 100.0
        
        # Record this return at the exit point
        df.loc[idx, 'trade_return'] = trade_return
        
        # Update our capital
        current_capital *= (1 + trade_return)
        
        # Store the cumulative return at this timestamp
        returns_by_time[idx] = current_capital - initial_capital
    
    # Create a series of cumulative returns
    cum_returns = pd.Series(returns_by_time)
    
    # Forward fill the cumulative returns to all timestamps
    full_index = df.index
    cum_returns = cum_returns.reindex(full_index).ffill().fillna(0)
    
    # Store in dataframe
    df['cum_strategy_return'] = cum_returns
    
    return df   

# Calculate performance metrics - FIXED VERSION
def calculate_performance_metrics(df):
    """
    Calculate performance metrics for the strategy with focus on short-term trading
    """
    # Filter for completed trades (where both entry and exit exist)
    completed_trades = df[(df['entry_price'].notna()) & (df['exit_price'].notna())].copy()
    
    if completed_trades.empty:
        return {
            'total_return': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'avg_trade_return': 0,
            'max_drawdown': 0,
            'total_trades': 0,
            'avg_trade_duration': 0
        }
    
    # Calculate trade metrics
    trade_profits = completed_trades['trade_profit_pct'].fillna(0)
    winning_trades = trade_profits[trade_profits > 0]
    losing_trades = trade_profits[trade_profits <= 0]
    
    # Total return
    total_return = df['cum_strategy_return'].iloc[-1] if 'cum_strategy_return' in df.columns else 0
    
    # Win rate
    total_trades = len(trade_profits)
    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
    
    # Profit factor
    total_profit = winning_trades.sum() if len(winning_trades) > 0 else 0
    total_loss = abs(losing_trades.sum()) if len(losing_trades) > 0 else 0.00001  # Small value to avoid division by zero
    profit_factor = total_profit / total_loss if total_loss > 0 else 0
    
    # Average return per trade
    avg_trade_return = trade_profits.mean() if len(trade_profits) > 0 else 0
    
    # Maximum drawdown
    if 'cum_strategy_return' in df.columns:
        df['peak'] = df['cum_strategy_return'].cummax()
        df['drawdown'] = df['peak'] - df['cum_strategy_return']
        max_drawdown = df['drawdown'].max() if len(df['drawdown']) > 0 else 0
    else:
        max_drawdown = 0
    
    # Average trade duration
    avg_trade_duration = completed_trades['trade_duration'].mean() if 'trade_duration' in completed_trades.columns else 0
    
    # Exit reasons count
    exit_reasons = completed_trades['exit_reason'].value_counts().to_dict() if 'exit_reason' in completed_trades.columns else {}
    
    if debug_mode:
        st.write("Performance metrics calculation complete")
        st.write(f"Win rate: {win_rate:.2f}, Profit factor: {profit_factor:.2f}")
        st.write(f"Avg trade return: {avg_trade_return:.2f}%, Total trades: {total_trades}")
    
    return {
        'total_return': total_return,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_trade_return': avg_trade_return, 
        'max_drawdown': max_drawdown,
        'total_trades': total_trades,
        'avg_trade_duration': avg_trade_duration,
        'exit_reasons': exit_reasons
    }

# Function to resample data to desired timeframe
def resample_data(df, timeframe):
    """
    Resample data to the desired timeframe
    """
    # Define the resample rule based on the timeframe
    if timeframe == "1m":
        rule = "1min"
    elif timeframe == "3m":
        rule = "3min"
    elif timeframe == "5m":
        rule = "5min"
    elif timeframe == "15m":
        rule = "15min"
    elif timeframe == "30m":
        rule = "30min"
    elif timeframe == "1h":
        rule = "1H"
    else:
        rule = "1min"  # Default to 1-minute
    
    # Resample OHLCV data
    resampled = df.resample(rule).agg({
        'price': 'last'
    })
    
    return resampled.dropna()

# Fetch price data and run strategy
@st.cache_data(ttl=600, show_spinner="Fetching price data...")
def fetch_price_data_and_run_strategy(token, lookback_minutes, bb_period, bb_std_dev, strategy_name, 
                                     take_profit_pct, stop_loss_pct, max_hold_time, candle_timeframe,
                                     apply_time_filter=False, start_hour=None, end_hour=None):
    """
    Fetch price data and run the selected trading strategy
    """
    # Get current time in Singapore timezone
    now_utc = datetime.now(pytz.utc)
    now_sg = now_utc.astimezone(singapore_timezone)
    start_time_sg = now_sg - timedelta(minutes=lookback_minutes)
    
    # Convert back to UTC for database query
    start_time_utc = start_time_sg.astimezone(pytz.utc)
    end_time_utc = now_sg.astimezone(pytz.utc)

    query = f"""
    SELECT 
        created_at AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore' AS timestamp, 
        final_price AS price, 
        pair_name
    FROM public.oracle_price_log
    WHERE created_at BETWEEN '{start_time_utc}' AND '{end_time_utc}'
    AND pair_name = '{token}';
    """
    
    try:
        df = pd.read_sql(query, engine)

        if df.empty:
            if debug_mode:
                st.error(f"No data found for {token} in the database")
            return None, None, None

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        
        if debug_mode:
            st.write(f"Initial data points: {len(df)}")
            st.write(f"Data range: {df.index.min()} to {df.index.max()}")
        
        # Apply time of day filter if requested
        if apply_time_filter and start_hour is not None and end_hour is not None:
            before_filter = len(df)
            # Filter by hour of day in Singapore time
            df = df[(df.index.hour >= start_hour) & (df.index.hour <= end_hour)]
            after_filter = len(df)
            
            if debug_mode:
                st.write(f"Time filtered: {before_filter} -> {after_filter} points")
            
            if df.empty:
                st.warning(f"No data available for {token} during the selected hours.")
                return None, None, None
        
        # Store original data for visualization
        original_df = df.copy()
        
        # Resample to desired timeframe
        df = resample_data(df, candle_timeframe)
        
        if debug_mode:
            st.write(f"After resampling to {candle_timeframe}: {len(df)} data points")
        
        # Calculate Bollinger Bands
        df = calculate_bollinger_bands(df, period=bb_period, std_dev=bb_std_dev)
        
        # Run the selected strategy
        if strategy_name == "BB Bounce":
            df = bb_bounce_strategy(df, take_profit_pct, stop_loss_pct, max_hold_time)
        elif strategy_name == "BB Breakout":
            df = bb_breakout_strategy(df, take_profit_pct, stop_loss_pct, max_hold_time)
        elif strategy_name == "BB Squeeze":
            df = bb_squeeze_strategy(df, take_profit_pct, stop_loss_pct, max_hold_time)
        elif strategy_name == "All Strategies":
            # Run all strategies and keep the best one
            strategies = {
                "BB Bounce": bb_bounce_strategy,
                "BB Breakout": bb_breakout_strategy,
                "BB Squeeze": bb_squeeze_strategy
            }
            
            best_return = -float('inf')
            best_df = None
            best_strat_name = None
            
            for strat_name, strat_func in strategies.items():
                if debug_mode:
                    st.write(f"Testing strategy: {strat_name}")
                
                temp_df = df.copy()
                temp_df = strat_func(temp_df, take_profit_pct, stop_loss_pct, max_hold_time)
                temp_df = calculate_returns(temp_df)
                
                # Count completed trades
                completed_trades = temp_df[(temp_df['entry_price'].notna()) & (temp_df['exit_price'].notna())]
                num_trades = len(completed_trades)
                
                if debug_mode:
                    st.write(f"{strat_name}: {num_trades} trades identified")
                
                if num_trades > 0:
                    # Calculate total profit
                    total_profit = completed_trades['trade_profit_pct'].sum()
                    
                    if debug_mode:
                        st.write(f"{strat_name} total profit: {total_profit:.2f}%")
                    
                    if total_profit > best_return:
                        best_return = total_profit
                        best_df = temp_df
                        best_strat_name = strat_name
            
            if best_df is not None:
                df = best_df
                strategy_name = best_strat_name
                
                if debug_mode:
                    st.write(f"Selected best strategy: {strategy_name} with return: {best_return:.2f}%")
            else:
                strategy_name = "No valid strategy found"
                
                if debug_mode:
                    st.warning("No valid trades found across all strategies")
        
        # Calculate returns
        df = calculate_returns(df)
        
        # Calculate performance metrics
        metrics = calculate_performance_metrics(df)
        metrics['strategy_name'] = strategy_name
        
        return df, metrics, original_df
    
    except Exception as e:
        st.error(f"Error processing {token}: {e}")
        if debug_mode:
            import traceback
            st.write(traceback.format_exc())
        return None, None, None

# Show spinner while processing
with st.spinner(f"Analyzing {selected_token} with {selected_strategy} strategy..."):
    df, metrics, original_df = fetch_price_data_and_run_strategy(
        selected_token, 
        lookback_minutes, 
        bb_period, 
        bb_std_dev, 
        selected_strategy,
        take_profit_pct,
        stop_loss_pct,
        max_hold_time,
        candle_timeframe,
        apply_time_filter,
        start_hour,
        end_hour
    )

# Display results
if df is not None and metrics is not None:
    # Main metrics
    st.header(f"{selected_token} Analysis Results - {metrics['strategy_name']} Strategy")
    
    # Performance metrics in neat columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Win Rate", f"{metrics['win_rate']*100:.1f}%", 
                 delta=f"{(metrics['win_rate']-0.5)*100:.1f}%" if metrics['win_rate'] > 0 else None)
        st.metric("Total Trades", f"{metrics['total_trades']}")
    
    with col2:
        st.metric("Avg Trade Return", f"{metrics['avg_trade_return']:.2f}%")
        st.metric("Avg Trade Duration", f"{metrics['avg_trade_duration']:.1f} bars")
    
    with col3:
        st.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
        st.metric("Total Return", f"{metrics['total_return']*100:.2f}%")
    
    with col4:
        st.metric("Max Drawdown", f"{metrics['max_drawdown']*100:.2f}%")
        if 'exit_reasons' in metrics and metrics['exit_reasons']:
            top_exit = max(metrics['exit_reasons'].items(), key=lambda x: x[1])[0]
            st.metric("Top Exit Reason", top_exit)
    
    # Price chart with Bollinger Bands and trades
    st.subheader("Price Chart with Bollinger Bands and Trades")
    
    fig = go.Figure()
    
    # Add price
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['price'],
        mode='lines',
        name='Price',
        line=dict(color='blue', width=1)
    ))
    
    # Add Bollinger Bands
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['bb_upper'],
        mode='lines',
        line=dict(color='rgba(173, 216, 230, 0.7)', width=1),
        name='Upper Band'
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['bb_lower'],
        mode='lines',
        line=dict(color='rgba(173, 216, 230, 0.7)', width=1),
        fill='tonexty',
        fillcolor='rgba(173, 216, 230, 0.3)',
        name='Lower Band'
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['bb_middle'],
        mode='lines',
        line=dict(color='rgba(173, 216, 230, 0.9)', width=1, dash='dash'),
        name='Middle Band'
    ))
    
    # Add entry points
    entries = df[df['entry_price'].notna()]
    if not entries.empty:
        fig.add_trace(go.Scatter(
            x=entries.index,
            y=entries['price'],
            mode='markers',
            marker=dict(color='green', size=10, symbol='triangle-up'),
            name='Entry Points'
        ))
    
    # Add exit points
    exits = df[df['exit_price'].notna()]
    if not exits.empty:
        fig.add_trace(go.Scatter(
            x=exits.index,
            y=exits['price'],
            mode='markers',
            marker=dict(color='red', size=10, symbol='triangle-down'),
            name='Exit Points'
        ))
    
    # Highlight squeeze periods if using squeeze strategy
    if selected_strategy == "BB Squeeze" or (selected_strategy == "All Strategies" and metrics['strategy_name'] == "BB Squeeze"):
        if 'in_squeeze' in df.columns:
            squeeze_periods = df[df['in_squeeze']]
            if not squeeze_periods.empty:
                fig.add_trace(go.Scatter(
                    x=squeeze_periods.index,
                    y=squeeze_periods['bb_lower'] * 0.99,  # Just below lower band for visibility
                    mode='markers',
                    marker=dict(color='purple', size=5, symbol='square'),
                    name='Squeeze Periods'
                ))
    
    fig.update_layout(
        title=f"{selected_token} Price with Bollinger Bands ({candle_timeframe} timeframe)",
        xaxis_title="Time",
        yaxis_title="Price",
        height=600,
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display equity curve
    if 'cum_strategy_return' in df.columns:
        st.subheader("Strategy Equity Curve")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['cum_strategy_return'] * 100,
            mode='lines',
            name='Equity Curve',
            line=dict(color='green', width=2)
        ))
        
        fig.update_layout(
            title=f"{selected_token} Strategy Performance ({metrics['strategy_name']})",
            xaxis_title="Time",
            yaxis_title="Cumulative Return (%)",
            height=400,
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Trade analysis
    st.subheader("Trade Analysis")
    
    # Filter for completed trades
    completed_trades = df[(df['entry_price'].notna()) & (df['exit_price'].notna())].copy()
    
    if not completed_trades.empty:
        # Calculate additional metrics for each trade
        completed_trades['time_of_day'] = completed_trades.index.hour
        
        # Display trade distribution by hour
        hour_counts = completed_trades.groupby('time_of_day').size()
        hour_returns = completed_trades.groupby('time_of_day')['trade_profit_pct'].mean()
        
        # Combine into a single DataFrame
        hour_analysis = pd.DataFrame({
            'Trade Count': hour_counts,
            'Avg Return (%)': hour_returns
        }).fillna(0)
        
        # Plot trade distribution by hour
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=hour_analysis.index,
            y=hour_analysis['Trade Count'],
            name='Trade Count',
            marker_color='blue',
            opacity=0.7,
            yaxis='y'
        ))
        
        fig.add_trace(go.Scatter(
            x=hour_analysis.index,
            y=hour_analysis['Avg Return (%)'],
            name='Avg Return (%)',
            marker_color='green',
            mode='lines+markers',
            yaxis='y2'
        ))
        
        fig.update_layout(
            title=f"Trade Distribution by Hour of Day (Singapore Time)",
            xaxis=dict(
                title='Hour of Day',
                tickmode='linear',
                tickvals=list(range(0, 24)),
                ticktext=[f"{h:02d}:00" for h in range(0, 24)]
            ),
            yaxis=dict(
                title='Number of Trades',
                titlefont=dict(color='blue'),
                tickfont=dict(color='blue')
            ),
            yaxis2=dict(
                title='Avg Return (%)',
                titlefont=dict(color='green'),
                tickfont=dict(color='green'),
                anchor='x',
                overlaying='y',
                side='right'
            ),
            height=400,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Trade duration analysis
        if 'trade_duration' in completed_trades.columns:
            st.subheader("Trade Duration Analysis")
            
            fig = px.histogram(
                completed_trades, 
                x='trade_duration',
                nbins=20,
                labels={'trade_duration': 'Trade Duration (bars)'},
                title=f"Distribution of Trade Durations for {selected_token}",
                color_discrete_sequence=['blue']
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Analyze if shorter or longer trades are more profitable
            duration_bins = [0, 1, 2, 3, 5, 10, max(completed_trades['trade_duration']) + 1]
            completed_trades['duration_category'] = pd.cut(completed_trades['trade_duration'], bins=duration_bins, 
                                                          labels=[f"{duration_bins[i]}-{duration_bins[i+1]-1}" for i in range(len(duration_bins)-1)])
            
            duration_analysis = completed_trades.groupby('duration_category')['trade_profit_pct'].agg(['mean', 'count']).reset_index()
            duration_analysis.columns = ['Duration (bars)', 'Avg Return (%)', 'Trade Count']
            
            fig = px.bar(
                duration_analysis,
                x='Duration (bars)',
                y='Avg Return (%)',
                text='Trade Count',
                color='Avg Return (%)',
                color_continuous_scale='RdYlGn',
                title='Returns by Trade Duration'
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Determine optimal trade duration based on data
            if not duration_analysis.empty:
                best_duration = duration_analysis.loc[duration_analysis['Avg Return (%)'].idxmax()]
                
                st.info(f"""
                **Optimal Trade Duration:**
                
                The most profitable trade duration is **{best_duration['Duration (bars)']} bars** with an average return of **{best_duration['Avg Return (%)']:.2f}%**.
                
                This duration had **{int(best_duration['Trade Count'])}** trades during the analyzed period.
                """)
        
        # Exit reason analysis
        if 'exit_reason' in completed_trades.columns:
            st.subheader("Exit Reason Analysis")
            
            exit_counts = completed_trades['exit_reason'].value_counts().reset_index()
            exit_counts.columns = ['Exit Reason', 'Count']
            
            exit_returns = completed_trades.groupby('exit_reason')['trade_profit_pct'].mean().reset_index()
            exit_returns.columns = ['Exit Reason', 'Avg Return (%)']
            
            exit_analysis = exit_counts.merge(exit_returns, on='Exit Reason')
            
            fig = px.bar(
                exit_analysis,
                x='Exit Reason',
                y='Count',
                color='Avg Return (%)',
                color_continuous_scale='RdYlGn',
                text='Count',
                title='Trade Distribution by Exit Reason'
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Determine best exit reason
            if not exit_analysis.empty:
                best_exit = exit_analysis.loc[exit_analysis['Avg Return (%)'].idxmax()]
                
                st.info(f"""
                **Most Profitable Exit Reason:**
                
                The most profitable exit reason is **{best_exit['Exit Reason']}** with an average return of **{best_exit['Avg Return (%)']:.2f}%**.
                
                This exit reason occurred **{int(best_exit['Count'])}** times during the analyzed period.
                """)
    
    # Trade table
    st.subheader("Trade Log")
    trade_log = df[(df['entry_price'].notna()) | (df['exit_price'].notna())].copy()
    
    if not trade_log.empty:
        # Format the trade log for display
        trade_log['time'] = trade_log.index.strftime('%Y-%m-%d %H:%M:%S')
        trade_log['price'] = trade_log['price'].round(4)
        trade_log['entry_price'] = trade_log['entry_price'].round(4)
        trade_log['exit_price'] = trade_log['exit_price'].round(4)
        trade_log['trade_profit_pct'] = trade_log['trade_profit_pct'].round(2)
        
        # Select columns to display
        display_columns = ['time', 'price', 'entry_price', 'exit_price', 'trade_profit_pct', 'trade_duration', 'exit_reason']
        trade_log_display = trade_log[display_columns].fillna('')
        
        # Rename columns for display
        trade_log_display.columns = ['Time', 'Price', 'Entry Price', 'Exit Price', 'Profit (%)', 'Duration (bars)', 'Exit Reason']
        
        # Display the trade log
        st.dataframe(trade_log_display, use_container_width=True)
    
    # Strategy sensitivity analysis
    st.subheader("Strategy Parameter Sensitivity")
    
    # Create tabs for parameter analysis
    tab1, tab2 = st.tabs(["Period Sensitivity", "Standard Deviation Sensitivity"])
    
    with tab1:
        # Analyze sensitivity to BB period
        periods_to_test = [10, 15, 20, 25, 30]
        period_results = []
        
        for period in periods_to_test:
            with st.spinner(f"Testing period {period}..."):
                test_df, test_metrics, _ = fetch_price_data_and_run_strategy(
                    selected_token, 
                    lookback_minutes, 
                    period,  # Vary the period
                    bb_std_dev, 
                    selected_strategy,
                    take_profit_pct,
                    stop_loss_pct,
                    max_hold_time,
                    candle_timeframe,
                    apply_time_filter,
                    start_hour,
                    end_hour
                )
                
                if test_df is not None and test_metrics is not None:
                    if debug_mode:
                        st.write(f"Period {period}: {test_metrics['total_trades']} trades")
                        
                    period_results.append({
                        'Period': period,
                        'Win Rate': test_metrics['win_rate'] * 100,
                        'Avg Return (%)': test_metrics['avg_trade_return'],
                        'Total Trades': test_metrics['total_trades'],
                        'Profit Factor': test_metrics['profit_factor']
                    })
        
        if period_results:
            period_df = pd.DataFrame(period_results)
            
            # Plot sensitivity to BB period
            fig = go.Figure()
            
            # Add bars for Win Rate
            fig.add_trace(go.Bar(
                x=period_df['Period'],
                y=period_df['Win Rate'],
                name='Win Rate (%)',
                marker_color='blue',
                opacity=0.7,
                yaxis='y'
            ))
            
            # Add lines for Avg Return
            fig.add_trace(go.Scatter(
                x=period_df['Period'],
                y=period_df['Avg Return (%)'],
                name='Avg Return (%)',
                marker_color='green',
                mode='lines+markers',
                yaxis='y2'
            ))
            
            # Add total trades as text
            for i, row in period_df.iterrows():
                fig.add_annotation(
                    x=row['Period'],
                    y=row['Win Rate'] + 5,
                    text=f"{int(row['Total Trades'])} trades",
                    showarrow=False,
                    font=dict(size=10)
                )
            
            fig.update_layout(
                title="Sensitivity to Bollinger Band Period",
                xaxis=dict(
                    title='Period',
                    tickmode='array',
                    tickvals=period_df['Period']
                ),
                yaxis=dict(
                    title='Win Rate (%)',
                    titlefont=dict(color='blue'),
                    tickfont=dict(color='blue')
                ),
                yaxis2=dict(
                    title='Avg Return (%)',
                    titlefont=dict(color='green'),
                    tickfont=dict(color='green'),
                    anchor='x',
                    overlaying='y',
                    side='right'
                ),
                height=400,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Determine optimal period
            if not period_df.empty:
                # Avoid division by zero
                period_df['Score'] = period_df['Win Rate'] * period_df['Avg Return (%)'] * (period_df['Profit Factor'] + 0.0001)
                best_period = period_df.loc[period_df['Score'].idxmax()]
                
                st.info(f"""
                **Optimal Bollinger Band Period:**
                
                The most effective period is **{int(best_period['Period'])}** with:
                - Win Rate: **{best_period['Win Rate']:.1f}%**
                - Average Return: **{best_period['Avg Return (%)']:.2f}%**
                - Profit Factor: **{best_period['Profit Factor']:.2f}**
                - Total Trades: **{int(best_period['Total Trades'])}**
                """)
    
    with tab2:
        # Analyze sensitivity to BB standard deviation
        stds_to_test = [1.5, 2.0, 2.5, 3.0]
        std_results = []
        
        for std in stds_to_test:
            with st.spinner(f"Testing standard deviation {std}..."):
                test_df, test_metrics, _ = fetch_price_data_and_run_strategy(
                    selected_token, 
                    lookback_minutes, 
                    bb_period, 
                    std,  # Vary the standard deviation
                    selected_strategy,
                    take_profit_pct,
                    stop_loss_pct,
                    max_hold_time,
                    candle_timeframe,
                    apply_time_filter,
                    start_hour,
                    end_hour
                )
                
                if test_df is not None and test_metrics is not None:
                    if debug_mode:
                        st.write(f"Std Dev {std}: {test_metrics['total_trades']} trades")
                        
                    std_results.append({
                        'Std Dev': std,
                        'Win Rate': test_metrics['win_rate'] * 100,
                        'Avg Return (%)': test_metrics['avg_trade_return'],
                        'Total Trades': test_metrics['total_trades'],
                        'Profit Factor': test_metrics['profit_factor']
                    })
        
        if std_results:
            std_df = pd.DataFrame(std_results)
            
            # Plot sensitivity to BB standard deviation
            fig = go.Figure()
            
            # Add bars for Win Rate
            fig.add_trace(go.Bar(
                x=std_df['Std Dev'],
                y=std_df['Win Rate'],
                name='Win Rate (%)',
                marker_color='blue',
                opacity=0.7,
                yaxis='y'
            ))
            
            # Add lines for Avg Return
            fig.add_trace(go.Scatter(
                x=std_df['Std Dev'],
                y=std_df['Avg Return (%)'],
                name='Avg Return (%)',
                marker_color='green',
                mode='lines+markers',
                yaxis='y2'
            ))
            
            # Add total trades as text
            for i, row in std_df.iterrows():
                fig.add_annotation(
                    x=row['Std Dev'],
                    y=row['Win Rate'] + 5,
                    text=f"{int(row['Total Trades'])} trades",
                    showarrow=False,
                    font=dict(size=10)
                )
            
            fig.update_layout(
                title="Sensitivity to Bollinger Band Standard Deviation",
                xaxis=dict(
                    title='Standard Deviation',
                    tickmode='array',
                    tickvals=std_df['Std Dev']
                ),
                yaxis=dict(
                    title='Win Rate (%)',
                    titlefont=dict(color='blue'),
                    tickfont=dict(color='blue')
                ),
                yaxis2=dict(
                    title='Avg Return (%)',
                    titlefont=dict(color='green'),
                    tickfont=dict(color='green'),
                    anchor='x',
                    overlaying='y',
                    side='right'
                ),
                height=400,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Determine optimal standard deviation
            if not std_df.empty:
                # Avoid division by zero
                std_df['Score'] = std_df['Win Rate'] * std_df['Avg Return (%)'] * (std_df['Profit Factor'] + 0.0001)
                best_std = std_df.loc[std_df['Score'].idxmax()]
                
                st.info(f"""
                **Optimal Bollinger Band Standard Deviation:**
                
                The most effective standard deviation is **{best_std['Std Dev']}** with:
                - Win Rate: **{best_std['Win Rate']:.1f}%**
                - Average Return: **{best_std['Avg Return (%)']:.2f}%**
                - Profit Factor: **{best_std['Profit Factor']:.2f}**
                - Total Trades: **{int(best_std['Total Trades'])}**
                """)
    
    # Strategy recommendations
    st.subheader("Strategy Recommendations")
    
    # Generate recommendations based on the analysis
    recommendations = []
    
    # Time-based recommendations
    if 'hour_analysis' in locals() and not hour_analysis.empty:
        best_hour = hour_analysis['Avg Return (%)'].idxmax()
        recommendations.append(f"Focus trading activity around {best_hour:02d}:00 Singapore time, which shows the highest average returns.")
    
    # Parameter recommendations
    if 'best_period' in locals():
        recommendations.append(f"Use a Bollinger Band period of {int(best_period['Period'])} for optimal results.")
    
    if 'best_std' in locals():
        recommendations.append(f"Set Bollinger Band standard deviation to {best_std['Std Dev']} for best performance.")
    
    # Trade duration recommendations
    if 'best_duration' in locals():
        recommendations.append(f"Target trade durations of {best_duration['Duration (bars)']} bars for maximum profitability.")
    
    # Exit strategy recommendations
    if 'best_exit' in locals():
        recommendations.append(f"Prioritize {best_exit['Exit Reason']} as your exit strategy, as it shows the highest average returns.")
    
    # Display recommendations
    for i, rec in enumerate(recommendations):
        st.markdown(f"{i+1}. {rec}")
    
    # Summary of findings
    st.subheader("Executive Summary")
    
    win_rate_desc = "excellent" if metrics['win_rate'] > 0.7 else "good" if metrics['win_rate'] > 0.5 else "poor"
    profit_factor_desc = "excellent" if metrics['profit_factor'] > 3 else "good" if metrics['profit_factor'] > 1.5 else "poor"
    
    # Determine if the strategy seems viable
    viable = metrics['win_rate'] > 0.5 and metrics['profit_factor'] > 1.5 and metrics['total_trades'] >= 5
    
    conclusion = f"""
    Based on the analysis of {selected_token} using the {metrics['strategy_name']} strategy with Bollinger Bands:
    
    - The strategy shows a {win_rate_desc} win rate of {metrics['win_rate']*100:.1f}% across {metrics['total_trades']} trades.
    - The profit factor is {profit_factor_desc} at {metrics['profit_factor']:.2f}.
    - Average trade return is {metrics['avg_trade_return']:.2f}% with holding periods of approximately {metrics['avg_trade_duration']:.1f} bars.
    
    **Conclusion:** This strategy {"appears viable and could be considered for real trading with appropriate risk management" if viable else "shows potential but needs further optimization or may not be suitable in current market conditions"}.
    """
    
    st.markdown(conclusion)
    
    # Download results as CSV
    if 'completed_trades' in locals() and not completed_trades.empty:
        csv = completed_trades.to_csv().encode('utf-8')
        st.download_button(
            label="Download Trade Log as CSV",
            data=csv,
            file_name=f"{selected_token}_{metrics['strategy_name']}_{candle_timeframe}_analysis.csv",
            mime="text/csv"
        )
else:
    st.warning(f"No results available for {selected_token} with the selected parameters. Try adjusting the time period or strategy settings.")

# Add helpful information about the analyzer
with st.expander("About This Analysis Tool"):
    st.markdown("""
    ### JUP Bollinger Bands Trading Analysis
    
    This tool is designed to analyze the effectiveness of Bollinger Band strategies for short-term trading, with a focus on Jupiter (JUP) and other cryptocurrencies.
    
    #### Implemented Strategies:
    
    1. **BB Bounce Strategy:**
       - Buys when price touches or goes below the lower Bollinger Band while near support
       - Sells when price reaches the middle band, hits take profit, stop loss, or max hold time
       - Works best in ranging markets with consistent volatility
    
    2. **BB Breakout Strategy:**
       - Buys when price breaks above the upper Bollinger Band with momentum
       - Sells when momentum reverses, hits take profit, stop loss, or max hold time
       - Works best in trending markets with increasing volatility
    
    3. **BB Squeeze Strategy:**
       - Identifies when Bollinger Bands narrow (low volatility) and then expand
       - Buys when bands start to expand with price moving upward
       - Sells when price reaches the upper band, hits take profit, stop loss, or max hold time
       - Works well for capturing the beginning of new trends
    
    #### Analysis Features:
    
    - Intraday analysis with 1-minute to 1-hour timeframes
    - Time-of-day filtering to focus on specific trading windows
    - Trade duration tracking and profitability analysis
    - Strategy parameter optimization
    - Detailed trade logging with entry/exit reasons
    
    #### Limitations:
    
    - Backtest results are theoretical and don't account for slippage, fees, or liquidity
    - Past performance is not indicative of future results
    - The analysis does not consider fundamental factors or market news
    
    This tool is for research purposes only and does not constitute financial advice.
    """)