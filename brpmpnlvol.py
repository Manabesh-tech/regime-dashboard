import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz
from sqlalchemy import create_engine, text
import matplotlib.pyplot as plt
import time

# Page configuration
st.set_page_config(
    page_title="Parameter Adjustment Dashboard",
    page_icon="📊",
    layout="wide"
)

# Apply CSS styles
st.markdown("""
<style>
    .header-style {
        font-size:24px !important;
        font-weight: bold;
        padding: 10px 0;
    }
    .subheader-style {
        font-size:20px !important;
        font-weight: bold;
        padding: 5px 0;
    }
    .highlight {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .warning {
        background-color: #ffe6e6;
        padding: 10px;
        border-radius: 5px;
    }
    .success {
        background-color: #e6ffe6;
        padding: 10px;
        border-radius: 5px;
    }
    .indicator-green {
        color: green;
        font-weight: bold;
    }
    .indicator-yellow {
        color: #FFA500;
        font-weight: bold;
    }
    .indicator-red {
        color: red;
        font-weight: bold;
    }
    .pair-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        border: 1px solid #e0e0e0;
    }
    .pair-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .metric-row {
        display: flex;
        justify-content: space-between;
        margin-bottom: 5px;
    }
    .metric-label {
        color: #666;
    }
    .metric-value {
        font-weight: 500;
    }
    .parameter-section {
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .dashboard-container {
        display: flex;
        flex-direction: column;
        gap: 20px;
    }
    .metrics-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }
    .metric-card {
        background-color: white;
        border-radius: 5px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        min-width: 180px;
        flex: 1;
    }
    .metric-card h4 {
        margin-top: 0;
        color: #555;
        font-size: 14px;
    }
    .metric-card p {
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 0;
    }
    .tier-0 {
        background-color: #e6ffe6;
    }
    .tier-1 {
        background-color: #fff4e6;
    }
    .tier-2 {
        background-color: #ffe6e6;
    }
    .reset-button {
        background-color: #f44336;
        color: white;
        border: none;
        padding: 10px 15px;
        border-radius: 5px;
        cursor: pointer;
    }
    .reset-button:hover {
        background-color: #d32f2f;
    }
    .update-timer {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
        text-align: center;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Singapore timezone for consistent time handling
SG_TZ = pytz.timezone('Asia/Singapore')

# Initialize session state variables
def init_session_state():
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False

    if 'last_daily_reset' not in st.session_state:
        st.session_state.last_daily_reset = get_sg_time().date()

    if 'pair_data' not in st.session_state:
        st.session_state.pair_data = {}

    if 'current_pair' not in st.session_state:
        st.session_state.current_pair = None
        
    if 'last_auto_update' not in st.session_state:
        st.session_state.last_auto_update = get_sg_time()
        
    if 'auto_update_enabled' not in st.session_state:
        st.session_state.auto_update_enabled = True
        
    # Add a force refresh counter to ensure timer updates
    if 'refresh_counter' not in st.session_state:
        st.session_state.refresh_counter = 0

    # Configuration values
    if 'vol_threshold_1' not in st.session_state:
        st.session_state.vol_threshold_1 = 50  # Default: 50% increase over daily average
    
    if 'vol_threshold_2' not in st.session_state:
        st.session_state.vol_threshold_2 = 100  # Default: 100% increase over daily average
    
    if 'parameter_adjustment_pct' not in st.session_state:
        st.session_state.parameter_adjustment_pct = 20  # Default: 20% adjustment
    
    if 'pnl_threshold_major_1' not in st.session_state:
        st.session_state.pnl_threshold_major_1 = -200  # Default -200 for major pairs
    
    if 'pnl_threshold_major_2' not in st.session_state:
        st.session_state.pnl_threshold_major_2 = -400  # Default -400 for major pairs
    
    if 'pnl_threshold_alt_1' not in st.session_state:
        st.session_state.pnl_threshold_alt_1 = -100  # Default -100 for alts
    
    if 'pnl_threshold_alt_2' not in st.session_state:
        st.session_state.pnl_threshold_alt_2 = -200  # Default -200 for alts
    
    if 'is_major_pairs' not in st.session_state:
        st.session_state.is_major_pairs = {
            "BTC/USDT": True,
            "ETH/USDT": True,
            "BNB/USDT": True,
            "SOL/USDT": True,
            "XRP/USDT": True,
            "ADA/USDT": True,
            "DOGE/USDT": True
        }  # Default major pairs

    st.session_state.initialized = True

# Function to get current time in Singapore timezone
def get_sg_time():
    """Get current time in Singapore timezone."""
    now_utc = datetime.now(pytz.utc)
    now_sg = now_utc.astimezone(SG_TZ)
    return now_sg

# Check if daily reset is needed
def check_daily_reset():
    today = get_sg_time().date()
    if today > st.session_state.last_daily_reset:
        # Reset PnL indicators for all pairs
        for pair_name in st.session_state.pair_data:
            st.session_state.pair_data[pair_name]['pnl_cumulative'] = 0
            st.session_state.pair_data[pair_name]['pnl_adjustment_tier'] = 0
        
        # Update last reset date
        st.session_state.last_daily_reset = today
        return True
    return False

# Check if auto-update is needed
def check_auto_update():
    now = get_sg_time()
    last_update = st.session_state.last_auto_update
    
    # Make sure last_update is timezone-aware
    if last_update.tzinfo is None:
        last_update = pytz.utc.localize(last_update).astimezone(SG_TZ)
        
    time_diff = (now - last_update).total_seconds() / 60
    
    if time_diff >= 5 and st.session_state.auto_update_enabled:
        st.session_state.last_auto_update = now
        return True
    return False

# Database connection functions
def init_connection():
    """Initialize database connection using Streamlit secrets."""
    try:
        # Get database credentials from Streamlit secrets
        db_config = st.secrets["database"]
        db_uri = (
            f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}"
            f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )
        engine = create_engine(db_uri)
        return engine
    except Exception as e:
        st.sidebar.error(f"Database connection error: {e}")
        # Fall back to direct connection parameters as provided in volatility plot code
        try:
            db_params = {
                'host': 'aws-jp-tk-surf-pg-public.cluster-csteuf9lw8dv.ap-northeast-1.rds.amazonaws.com',
                'port': 5432,
                'database': 'replication_report',
                'user': 'public_replication',
                'password': '866^FKC4hllk'
            }
            conn_str = f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['database']}"
            engine = create_engine(conn_str)
            return engine
        except Exception as e2:
            st.error(f"Alternative connection failed: {e2}")
            return None

# Fetch available trading pairs
def fetch_pairs():
    """Fetch all active trading pairs from the database."""
    engine = init_connection()
    if not engine:
        st.error("Failed to connect to database. Check your connection settings.")
        return []
    
    try:
        query = """
        SELECT DISTINCT pair_name 
        FROM public.trade_pool_pairs 
        WHERE status = 1
        ORDER BY pair_name
        """
        
        df = pd.read_sql(query, engine)
        if df.empty:
            st.warning("No active trading pairs found in the database.")
            return []
        return df['pair_name'].tolist()
    except Exception as e:
        st.error(f"Error fetching pairs: {e}")
        return []

# Fetch current parameters for a specific pair
def fetch_current_parameters(pair_name):
    """Fetch current parameters for a specific pair from the database."""
    engine = init_connection()
    if not engine:
        st.error("Failed to connect to database. Check your connection settings.")
        return {
            "buffer_rate": 0.001,
            "position_multiplier": 1000
        }
    
    try:
        query = f"""
        SELECT
            (leverage_config::jsonb->0->>'buffer_rate')::numeric AS buffer_rate,
            position_multiplier
        FROM
            public.trade_pool_pairs
        WHERE
            pair_name = '{pair_name}'
            AND status = 1
        """
        
        df = pd.read_sql(query, engine)
        if df.empty:
            st.warning(f"No parameter data found for {pair_name}. Using default values.")
            return {
                "buffer_rate": 0.001,
                "position_multiplier": 1000
            }
        
        # Convert to dictionary
        params = {
            "buffer_rate": float(df['buffer_rate'].iloc[0]),
            "position_multiplier": float(df['position_multiplier'].iloc[0])
        }
        
        return params
    except Exception as e:
        st.error(f"Error fetching parameters for {pair_name}: {e}")
        return {
            "buffer_rate": 0.001,
            "position_multiplier": 1000
        }

# Initialize a new pair in session state
def init_pair_state(pair_name):
    """Initialize session state for a specific pair."""
    if 'pair_data' not in st.session_state:
        st.session_state.pair_data = {}
    
    if pair_name not in st.session_state.pair_data:
        # Get current parameters
        params = fetch_current_parameters(pair_name)
        
        # Initialize pair data
        st.session_state.pair_data[pair_name] = {
            'initialized': True,
            'buffer_rate': params["buffer_rate"],
            'position_multiplier': params["position_multiplier"],
            'base_buffer_rate': params["buffer_rate"],  # Store original values as base
            'base_position_multiplier': params["position_multiplier"],
            'volatility_history': [],  # List of (timestamp, volatility) tuples
            'daily_avg_volatility': None,
            'current_volatility': None,
            'vol_adjustment_tier': 0,  # 0: normal, 1: tier 1 adjustment, 2: tier 2 adjustment
            'pnl_history': [],  # List of (timestamp, pnl) tuples
            'pnl_cumulative': 0,  # Running PnL total
            'pnl_adjustment_tier': 0,  # 0: normal, 1: tier 1 adjustment, 2: tier 2 adjustment
            'last_update_time': get_sg_time(),
            'parameter_history': []  # List of (timestamp, buffer_rate, position_multiplier, reason) tuples
        }
        
        # Determine if this is a major pair
        if pair_name not in st.session_state.is_major_pairs:
            # Check if it matches common major pair patterns
            is_major = (
                pair_name == "BTC/USDT" or 
                pair_name == "ETH/USDT" or 
                pair_name == "BNB/USDT" or 
                pair_name == "SOL/USDT" or
                pair_name == "XRP/USDT" or
                pair_name == "ADA/USDT" or
                pair_name == "DOGE/USDT"
            )
            st.session_state.is_major_pairs[pair_name] = is_major

# Calculate volatility
def calculate_volatility(pair_name, hours=24):
    """Calculate and return the volatility data for a pair."""
    engine = init_connection()
    if not engine:
        st.error("Failed to connect to database. Check your connection settings.")
        return None, None, None
    
    try:
        # Get current time in Singapore timezone
        now_sg = get_sg_time()
        start_time_sg = now_sg - timedelta(hours=hours+1)  # Extra hour for buffer
        
        # Get relevant partition tables (today and yesterday)
        start_date = start_time_sg.replace(tzinfo=None)
        end_date = now_sg.replace(tzinfo=None)
        
        # Generate all dates
        dates = []
        current_date = start_date
        while current_date <= end_date:
            dates.append(current_date.strftime("%Y%m%d"))
            current_date += timedelta(days=1)
        
        # Table names
        table_names = [f"oracle_price_log_partition_{date}" for date in dates]
        
        # Check which tables exist
        with engine.connect() as connection:
            if table_names:
                table_list_str = "', '".join(table_names)
                query = f"""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name IN ('{table_list_str}')
                """
                result = connection.execute(text(query))
                existing_tables = [row[0] for row in result]
        
        if not existing_tables:
            # Fallback to using the regular oracle_price_log table
            start_str = start_time_sg.strftime('%Y-%m-%d %H:%M:%S')
            end_str = now_sg.strftime('%Y-%m-%d %H:%M:%S')
            
            query = f"""
            SELECT 
                created_at AT TIME ZONE 'UTC' AT TIME ZONE 'Asia/Singapore' AS timestamp,
                final_price
            FROM 
                public.oracle_price_log
            WHERE 
                created_at BETWEEN '{start_str}'::timestamp - INTERVAL '8 hour'
                AND '{end_str}'::timestamp - INTERVAL '8 hour'
                AND pair_name = '{pair_name}'
                AND source_type = 0
            ORDER BY 
                created_at
            """
            
            price_df = pd.read_sql(query, engine)
        else:
            # Use the same approach as the volatility plot code
            union_parts = []
            for table in existing_tables:
                # Convert to strings for query with Singapore time
                start_str = start_time_sg.strftime('%Y-%m-%d %H:%M:%S')
                end_str = now_sg.strftime('%Y-%m-%d %H:%M:%S')
                
                # Add 8 hours to convert to Singapore time - copying the exact approach from voltimeplot.py
                query_part = f"""
                SELECT 
                    pair_name,
                    created_at + INTERVAL '8 hour' AS timestamp,
                    final_price
                FROM 
                    public.{table}
                WHERE 
                    created_at >= '{start_str}'::timestamp - INTERVAL '8 hour'
                    AND created_at <= '{end_str}'::timestamp - INTERVAL '8 hour'
                    AND source_type = 0
                    AND pair_name = '{pair_name}'
                """
                union_parts.append(query_part)
            
            full_query = " UNION ".join(union_parts) + " ORDER BY timestamp"
            price_df = pd.read_sql(full_query, engine)
        
        if price_df.empty:
            st.warning(f"No price data found for {pair_name} in the specified time period.")
            return None, None, None
        
        # Convert timestamp to pandas datetime
        price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
        price_df = price_df.set_index('timestamp').sort_index()
        
        # Make sure we have the final_price column
        price_column = 'final_price'
        if price_column not in price_df.columns and len(price_df.columns) > 0:
            price_column = price_df.columns[0]  # Use the first available column
        
        # Create 5-minute windows exactly like in voltimeplot.py
        result = []
        start_time = price_df.index.min().floor('5min')
        end_time = price_df.index.max().ceil('5min')
        periods = pd.date_range(start=start_time, end=end_time, freq='5min')
        
        for i in range(len(periods) - 1):
            start_period = periods[i]
            end_period = periods[i+1]
            
            # Get data in this window
            window_data = price_df[(price_df.index >= start_period) & (price_df.index < end_period)]
            
            if len(window_data) >= 2:  # Need at least 2 points for volatility
                # Calculate log returns
                log_returns = np.diff(np.log(window_data[price_column].values))
                
                # Annualize: seconds in year / seconds in 5 minutes
                annualization_factor = np.sqrt(31536000 / 300)
                volatility = np.std(log_returns) * annualization_factor
                
                result.append({
                    'timestamp': start_period,
                    'realized_vol': volatility
                })
        
        if not result:
            st.warning(f"Could not calculate volatility for {pair_name}")
            return None, None, None
        
        # Create DataFrame
        vol_df = pd.DataFrame(result).set_index('timestamp')
        
        # Get most recent volatility
        current_vol = vol_df['realized_vol'].iloc[-1]
        
        # Calculate daily average volatility (24 hours)
        daily_avg = vol_df['realized_vol'].mean()
        
        return vol_df, current_vol, daily_avg
    
    except Exception as e:
        st.error(f"Error calculating volatility for {pair_name}: {e}")
        return None, None, None

# Calculate PnL for a specific pair
def calculate_pnl(pair_name, hours=24):
    """Calculate PnL for a specific pair for the last specified hours."""
    engine = init_connection()
    if not engine:
        st.error("Failed to connect to database. Check your connection settings.")
        return None, 0
    
    try:
        # Get current time in Singapore timezone
        now_sg = get_sg_time()
        start_time_sg = now_sg - timedelta(hours=hours)
        
        # Convert to UTC for database query (subtract 8 hours for proper time alignment)
        start_time_utc = start_time_sg - timedelta(hours=8)
        now_utc = now_sg - timedelta(hours=8)
        
        # Format timestamps for query
        start_str = start_time_utc.strftime('%Y-%m-%d %H:%M:%S')
        end_str = now_utc.strftime('%Y-%m-%d %H:%M:%S')
        
        # Query for PnL data with proper time conversion
        query = f"""
        WITH pnl_data AS (
          SELECT
            created_at + INTERVAL '8 hour' AS timestamp,
            (-1 * taker_pnl * collateral_price) AS pnl_value
          FROM 
            public.trade_fill_fresh
          WHERE 
            created_at BETWEEN '{start_str}'::timestamp AND '{end_str}'::timestamp
            AND pair_name = '{pair_name}'
        )
        SELECT 
            timestamp,
            pnl_value,
            SUM(pnl_value) OVER (ORDER BY timestamp) AS cumulative_pnl
        FROM 
            pnl_data
        ORDER BY 
            timestamp
        """
        
        pnl_df = pd.read_sql(query, engine)
        
        if pnl_df.empty:
            return None, 0
        
        # Calculate total PnL for the period
        total_pnl = pnl_df['pnl_value'].sum()
        
        return pnl_df, total_pnl
    
    except Exception as e:
        st.error(f"Error calculating PnL for {pair_name}: {e}")
        return None, 0

# Update pair volatility and PnL data
def update_pair_data(pair_name):
    """Update volatility and PnL data for a specific pair."""
    # Ensure pair state is initialized
    init_pair_state(pair_name)
    
    # Get current time
    current_time = get_sg_time()
    
    # Calculate volatility
    vol_df, current_vol, daily_avg = calculate_volatility(pair_name)
    
    # Update volatility information
    if vol_df is not None and current_vol is not None and daily_avg is not None:
        st.session_state.pair_data[pair_name]['volatility_history'].append((current_time, current_vol))
        st.session_state.pair_data[pair_name]['current_volatility'] = current_vol
        st.session_state.pair_data[pair_name]['daily_avg_volatility'] = daily_avg
        
        # Calculate percent increase from daily average
        if daily_avg > 0:
            vol_increase_pct = ((current_vol - daily_avg) / daily_avg) * 100
        else:
            vol_increase_pct = 0
        
        # Determine volatility adjustment tier
        old_tier = st.session_state.pair_data[pair_name]['vol_adjustment_tier']
        
        if vol_increase_pct >= st.session_state.vol_threshold_2:
            new_tier = 2
        elif vol_increase_pct >= st.session_state.vol_threshold_1:
            new_tier = 1
        else:
            new_tier = 0
        
        st.session_state.pair_data[pair_name]['vol_adjustment_tier'] = new_tier
        
        # Log tier change
        if old_tier != new_tier:
            reason = f"Volatility {'increased' if new_tier > old_tier else 'decreased'} (Current: {current_vol:.4f}, Daily Avg: {daily_avg:.4f}, Change: {vol_increase_pct:.2f}%)"
            st.session_state.pair_data[pair_name]['parameter_history'].append((
                current_time, 
                st.session_state.pair_data[pair_name]['buffer_rate'],
                st.session_state.pair_data[pair_name]['position_multiplier'],
                reason
            ))
    
    # Calculate PnL
    pnl_df, period_pnl = calculate_pnl(pair_name)
    
    # Update PnL information
    if pnl_df is not None:
        st.session_state.pair_data[pair_name]['pnl_history'].append((current_time, period_pnl))
        
        # Update cumulative PnL
        st.session_state.pair_data[pair_name]['pnl_cumulative'] += period_pnl
        cumulative_pnl = st.session_state.pair_data[pair_name]['pnl_cumulative']
        
        # Determine PnL adjustment tier based on pair type (major or altcoin)
        old_tier = st.session_state.pair_data[pair_name]['pnl_adjustment_tier']
        
        if st.session_state.is_major_pairs.get(pair_name, False):
            # Major pair thresholds
            if cumulative_pnl <= st.session_state.pnl_threshold_major_2:
                new_tier = 2
            elif cumulative_pnl <= st.session_state.pnl_threshold_major_1:
                new_tier = 1
            else:
                new_tier = 0
        else:
            # Alt thresholds
            if cumulative_pnl <= st.session_state.pnl_threshold_alt_2:
                new_tier = 2
            elif cumulative_pnl <= st.session_state.pnl_threshold_alt_1:
                new_tier = 1
            else:
                new_tier = 0
        
        st.session_state.pair_data[pair_name]['pnl_adjustment_tier'] = new_tier
        
        # Log tier change
        if old_tier != new_tier:
            reason = f"PnL {'decreased' if new_tier > old_tier else 'increased'} (Cumulative PnL: {cumulative_pnl:.2f})"
            st.session_state.pair_data[pair_name]['parameter_history'].append((
                current_time, 
                st.session_state.pair_data[pair_name]['buffer_rate'],
                st.session_state.pair_data[pair_name]['position_multiplier'],
                reason
            ))
    
    # Update last update time
    st.session_state.pair_data[pair_name]['last_update_time'] = current_time
    
    # Apply parameter adjustments based on current tiers
    adjust_parameters(pair_name)

# Apply parameter adjustments based on volatility and PnL tiers
def adjust_parameters(pair_name):
    """Apply parameter adjustments based on volatility and PnL tiers."""
    # Get current adjustment tiers
    vol_tier = st.session_state.pair_data[pair_name]['vol_adjustment_tier']
    pnl_tier = st.session_state.pair_data[pair_name]['pnl_adjustment_tier']
    
    # Get base parameters
    base_br = st.session_state.pair_data[pair_name]['base_buffer_rate']
    base_pm = st.session_state.pair_data[pair_name]['base_position_multiplier']
    
    # Determine the overall tier (take the worse of the two)
    overall_tier = max(vol_tier, pnl_tier)
    
    # Apply adjustments based on overall tier
    if overall_tier == 0:
        # Normal conditions - use base parameters
        new_br = base_br
        new_pm = base_pm
    elif overall_tier == 1:
        # Tier 1 adjustment - 20% worse
        adjustment_pct = st.session_state.parameter_adjustment_pct / 100
        new_br = base_br * (1 + adjustment_pct)  # Increase buffer rate
        new_pm = base_pm * (1 - adjustment_pct)  # Decrease position multiplier
    elif overall_tier == 2:
        # Tier 2 adjustment - 40% worse
        adjustment_pct = (st.session_state.parameter_adjustment_pct * 2) / 100
        new_br = base_br * (1 + adjustment_pct)  # Increase buffer rate
        new_pm = base_pm * (1 - adjustment_pct)  # Decrease position multiplier
    
    # Update parameters
    st.session_state.pair_data[pair_name]['buffer_rate'] = new_br
    st.session_state.pair_data[pair_name]['position_multiplier'] = new_pm

# Reset PnL for a specific pair
def reset_pnl(pair_name):
    """Reset the PnL indicator for a specific pair."""
    if pair_name in st.session_state.pair_data:
        st.session_state.pair_data[pair_name]['pnl_cumulative'] = 0
        st.session_state.pair_data[pair_name]['pnl_adjustment_tier'] = 0
        
        # Log the reset
        current_time = get_sg_time()
        reason = "Manual PnL reset"
        st.session_state.pair_data[pair_name]['parameter_history'].append((
            current_time,