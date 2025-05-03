import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz
from sqlalchemy import create_engine, text
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

    # Add view mode: 'table' or 'detail'
    if 'view_mode' not in st.session_state:
        st.session_state.view_mode = 'table'

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
        
    if 'update_status' not in st.session_state:
        st.session_state.update_status = ""

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

# Database connection functions - with caching to improve performance
@st.cache_resource(ttl=3600)  # Cache for 1 hour
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

# Fetch available trading pairs - cache the results to avoid repeated database calls
@st.cache_data(ttl=300)  # Cache for 5 minutes
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

# Fetch current parameters for multiple pairs at once to reduce database calls
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_current_parameters_batch(pair_names):
    """Fetch current parameters for multiple pairs from the database."""
    engine = init_connection()
    if not engine:
        st.error("Failed to connect to database. Check your connection settings.")
        return {}
    
    # Default parameters
    default_params = {
        "buffer_rate": 0.001,
        "position_multiplier": 1000
    }
    
    if not pair_names:
        return {}
    
    try:
        # Create a comma-separated list of pair names
        pair_list = "', '".join(pair_names)
        
        query = f"""
        SELECT
            pair_name,
            (leverage_config::jsonb->0->>'buffer_rate')::numeric AS buffer_rate,
            position_multiplier
        FROM
            public.trade_pool_pairs
        WHERE
            pair_name IN ('{pair_list}')
            AND status = 1
        """
        
        df = pd.read_sql(query, engine)
        
        # Convert to dictionary
        params_dict = {}
        for pair_name in pair_names:
            pair_df = df[df['pair_name'] == pair_name]
            if not pair_df.empty:
                params_dict[pair_name] = {
                    "buffer_rate": float(pair_df['buffer_rate'].iloc[0]),
                    "position_multiplier": float(pair_df['position_multiplier'].iloc[0])
                }
            else:
                params_dict[pair_name] = default_params.copy()
        
        return params_dict
    except Exception as e:
        st.error(f"Error fetching parameters: {e}")
        return {pair_name: default_params.copy() for pair_name in pair_names}

# Initialize multiple pairs at once to reduce database calls
def init_pairs_batch(pair_names):
    """Initialize session state for multiple pairs at once."""
    if 'pair_data' not in st.session_state:
        st.session_state.pair_data = {}
    
    # Fetch parameters for all pairs at once
    params_batch = fetch_current_parameters_batch(pair_names)
    
    for pair_name in pair_names:
        if pair_name not in st.session_state.pair_data:
            # Get parameters for this pair
            params = params_batch.get(pair_name, {
                "buffer_rate": 0.001,
                "position_multiplier": 1000
            })
            
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
                'parameter_history': [],  # List of (timestamp, buffer_rate, position_multiplier, reason) tuples
                'recommended_buffer_rate': params["buffer_rate"],  # Added recommended values
                'recommended_position_multiplier': params["position_multiplier"]
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

# Optimized volatility calculation with caching
@st.cache_data(ttl=300)  # Cache for 5 minutes
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

# Optimized PnL calculation with caching
@st.cache_data(ttl=300)  # Cache for 5 minutes
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

# Batch update for multiple pairs to improve efficiency
def batch_update_pairs(pair_names):
    """Update volatility and PnL data for multiple pairs efficiently."""
    if not pair_names:
        return
    
    # Initialize all pairs first in batch
    init_pairs_batch(pair_names)
    
    # Get current time
    current_time = get_sg_time()
    
    # Process all pairs
    with st.spinner(f"Updating {len(pair_names)} pairs..."):
        for pair_name in pair_names:
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
            
            # Calculate recommended parameters
            calculate_recommended_parameters(pair_name)
            
            # Apply parameter adjustments based on current tiers
            adjust_parameters(pair_name)

# Calculate recommended parameters based on volatility and PnL tiers
def calculate_recommended_parameters(pair_name):
    """Calculate recommended parameters based on volatility and PnL tiers."""
    # Get current adjustment tiers
    vol_tier = st.session_state.pair_data[pair_name]['vol_adjustment_tier']
    pnl_tier = st.session_state.pair_data[pair_name]['pnl_adjustment_tier']
    
    # Get base parameters
    base_br = st.session_state.pair_data[pair_name]['base_buffer_rate']
    base_pm = st.session_state.pair_data[pair_name]['base_position_multiplier']
    
    # Determine the overall tier (take the worse of the two)
    overall_tier = max(vol_tier, pnl_tier)
    
    # Calculate recommended parameters based on overall tier
    if overall_tier == 0:
        # Normal conditions - use base parameters
        recommended_br = base_br
        recommended_pm = base_pm
    elif overall_tier == 1:
        # Tier 1 adjustment - 20% worse
        adjustment_pct = st.session_state.parameter_adjustment_pct / 100
        recommended_br = base_br * (1 + adjustment_pct)  # Increase buffer rate
        recommended_pm = base_pm * (1 - adjustment_pct)  # Decrease position multiplier
    elif overall_tier == 2:
        # Tier 2 adjustment - 40% worse
        adjustment_pct = (st.session_state.parameter_adjustment_pct * 2) / 100
        recommended_br = base_br * (1 + adjustment_pct)  # Increase buffer rate
        recommended_pm = base_pm * (1 - adjustment_pct)  # Decrease position multiplier
    
    # Update recommended parameters
    st.session_state.pair_data[pair_name]['recommended_buffer_rate'] = recommended_br
    st.session_state.pair_data[pair_name]['recommended_position_multiplier'] = recommended_pm

# Apply parameter adjustments based on volatility and PnL tiers
def adjust_parameters(pair_name):
    """Apply parameter adjustments based on volatility and PnL tiers."""
    # Get the recommended values
    recommended_br = st.session_state.pair_data[pair_name]['recommended_buffer_rate']
    recommended_pm = st.session_state.pair_data[pair_name]['recommended_position_multiplier']
    
    # Update parameters
    st.session_state.pair_data[pair_name]['buffer_rate'] = recommended_br
    st.session_state.pair_data[pair_name]['position_multiplier'] = recommended_pm

# Reset PnL for a specific pair
def reset_pnl(pair_name):
    """Reset the PnL indicator for a specific pair."""
    if pair_name in st.session_state.pair_data:
        st.session_state.pair_data[pair_name]['pnl_cumulative'] = 0
        st.session_state.pair_data[pair_name]['pnl_adjustment_tier'] = 0
        
        # Log the reset
        current_time = get_sg_time()
        reason = "Manual PnL reset"
        st.session_state.pair_data[pair_name]['parameter_history'].append(
            (current_time, 
            st.session_state.pair_data[pair_name]['buffer_rate'],
            st.session_state.pair_data[pair_name]['position_multiplier'],
            reason)
        )
        
        # Re-calculate recommended parameters
        calculate_recommended_parameters(pair_name)
        
        # Re-apply parameter adjustments
        adjust_parameters(pair_name)
        
        return True
    return False

# Reset all PnL - improved to handle batches
def reset_all_pnl(pairs=None):
    """Reset PnL for all pairs or a specific list of pairs."""
    reset_count = 0
    pairs_to_reset = pairs if pairs else st.session_state.pair_data.keys()
    
    for pair_name in pairs_to_reset:
        if reset_pnl(pair_name):
            reset_count += 1
    
    return reset_count

# Function to create volatility plot
def create_volatility_plot(pair_name):
    """Create a plot of volatility data."""
    if pair_name not in st.session_state.pair_data:
        return None
    
    # Get volatility data
    vol_df, current_vol, daily_avg = calculate_volatility(pair_name)
    
    if vol_df is None:
        return None
    
    # Convert to percentage for display
    vol_df_pct = vol_df.copy()
    vol_df_pct['realized_vol'] = vol_df_pct['realized_vol'] * 100  # Convert to percentage
    
    # Create figure
    fig = go.Figure()
    
    # Add volatility line
    fig.add_trace(
        go.Scatter(
            x=vol_df_pct.index,
            y=vol_df_pct['realized_vol'],
            mode='lines',
            line=dict(color='blue', width=2),
            name="Realized Volatility"
        )
    )
    
    # Add daily average line
    if daily_avg is not None:
        daily_avg_pct = daily_avg * 100
        fig.add_trace(
            go.Scatter(
                x=[vol_df_pct.index.min(), vol_df_pct.index.max()],
                y=[daily_avg_pct, daily_avg_pct],
                mode='lines',
                line=dict(color='green', width=2, dash='dash'),
                name=f"Daily Average: {daily_avg_pct:.2f}%"
            )
        )
        
        # Add threshold lines
        threshold1_pct = daily_avg_pct * (1 + st.session_state.vol_threshold_1/100)
        threshold2_pct = daily_avg_pct * (1 + st.session_state.vol_threshold_2/100)
        
        fig.add_trace(
            go.Scatter(
                x=[vol_df_pct.index.min(), vol_df_pct.index.max()],
                y=[threshold1_pct, threshold1_pct],
                mode='lines',
                line=dict(color='orange', width=2, dash='dash'),
                name=f"Threshold 1 (+{st.session_state.vol_threshold_1}%): {threshold1_pct:.2f}%"
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=[vol_df_pct.index.min(), vol_df_pct.index.max()],
                y=[threshold2_pct, threshold2_pct],
                mode='lines',
                line=dict(color='red', width=2, dash='dash'),
                name=f"Threshold 2 (+{st.session_state.vol_threshold_2}%): {threshold2_pct:.2f}%"
            )
        )
    
    # Update layout
    fig.update_layout(
        title=f"{pair_name} Realized Volatility (5-minute)",
        xaxis_title="Time",
        yaxis_title="Annualized Volatility (%)",
        height=500,
        hovermode="x unified"
    )
    
    return fig

# Function to create PnL plot
def create_pnl_plot(pair_name):
    """Create a plot of PnL data."""
    if pair_name not in st.session_state.pair_data:
        return None
    
    # Get PnL data
    pnl_df, _ = calculate_pnl(pair_name)
    
    if pnl_df is None or pnl_df.empty:
        return None
    
    # Create figure
    fig = go.Figure()
    
    # Add PnL line
    fig.add_trace(
        go.Scatter(
            x=pnl_df['timestamp'],
            y=pnl_df['cumulative_pnl'],
            mode='lines',
            line=dict(color='blue', width=2),
            name="Cumulative PnL"
        )
    )
    
    # Add threshold lines
    is_major = st.session_state.is_major_pairs.get(pair_name, False)
    threshold1 = st.session_state.pnl_threshold_major_1 if is_major else st.session_state.pnl_threshold_alt_1
    threshold2 = st.session_state.pnl_threshold_major_2 if is_major else st.session_state.pnl_threshold_alt_2
    
    fig.add_trace(
        go.Scatter(
            x=[pnl_df['timestamp'].min(), pnl_df['timestamp'].max()],
            y=[threshold1, threshold1],
            mode='lines',
            line=dict(color='orange', width=2, dash='dash'),
            name=f"Threshold 1: {threshold1}"
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=[pnl_df['timestamp'].min(), pnl_df['timestamp'].max()],
            y=[threshold2, threshold2],
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name=f"Threshold 2: {threshold2}"
        )
    )
    
    # Update layout
    fig.update_layout(
        title=f"{pair_name} Cumulative PnL",
        xaxis_title="Time",
        yaxis_title="PnL Value",
        height=500,
        hovermode="x unified"
    )
    
    return fig

# Function to create parameter history plot
def create_parameter_history_plot(pair_name):
    """Create a plot of parameter adjustment history."""
    if pair_name not in st.session_state.pair_data or not st.session_state.pair_data[pair_name]['parameter_history']:
        return None, None
    
    # Extract data
    history = st.session_state.pair_data[pair_name]['parameter_history']
    timestamps = [entry[0] for entry in history]
    buffer_rates = [entry[1] for entry in history]
    position_multipliers = [entry[2] for entry in history]
    
    # Create buffer rate figure
    br_fig = go.Figure()
    br_fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=buffer_rates,
            mode='lines+markers',
            name="Buffer Rate"
        )
    )
    
    br_fig.update_layout(
        title=f"{pair_name} Buffer Rate History",
        xaxis_title="Time",
        yaxis_title="Buffer Rate",
        height=400
    )
    
    # Create position multiplier figure
    pm_fig = go.Figure()
    pm_fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=position_multipliers,
            mode='lines+markers',
            name="Position Multiplier"
        )
    )
    
    pm_fig.update_layout(
        title=f"{pair_name} Position Multiplier History",
        xaxis_title="Time",
        yaxis_title="Position Multiplier",
        height=400
    )
    
    return br_fig, pm_fig

# Calculate time until next auto-update
def time_until_next_update():
    """Calculate time until next auto-update."""
    if not st.session_state.auto_update_enabled:
        return None
    
    now = get_sg_time()
    last_update = st.session_state.last_auto_update
    
    # Make sure last_update is timezone-aware
    if last_update.tzinfo is None:
        last_update = pytz.utc.localize(last_update).astimezone(SG_TZ)
    
    elapsed_seconds = (now - last_update).total_seconds()
    
    if elapsed_seconds >= 300:  # 5 minutes in seconds
        return 0
    
    return 300 - elapsed_seconds  # Time remaining in seconds

# Optimized function to generate pairs table data
@st.cache_data(ttl=30)  # Short cache to ensure UI responsiveness
def generate_pairs_table_data(filter_type=None, tier_filter=None, search_term=None):
    """Generate data for the pairs table with filtering."""
    # Get available pairs
    available_pairs = fetch_pairs()
    
    # Apply pair type filter
    if filter_type == 'major':
        available_pairs = [p for p in available_pairs if st.session_state.is_major_pairs.get(p, False)]
    elif filter_type == 'alt':
        available_pairs = [p for p in available_pairs if not st.session_state.is_major_pairs.get(p, False)]
    
    # Apply search filter
    if search_term:
        available_pairs = [p for p in available_pairs if search_term.lower() in p.lower()]
    
    # Initialize pairs if needed
    init_pairs_batch(available_pairs)
    
    # Generate table data
    table_data = []
    
    for pair_name in available_pairs:
        pair_data = st.session_state.pair_data.get(pair_name, {})
        
        # Get pair type
        is_major = st.session_state.is_major_pairs.get(pair_name, False)
        pair_type = "Major" if is_major else "Alt"
        
        # Get volatility data
        current_vol = pair_data.get('current_volatility')
        daily_avg_vol = pair_data.get('daily_avg_volatility')
        vol_change_pct = 0
        if current_vol is not None and daily_avg_vol is not None and daily_avg_vol > 0:
            vol_change_pct = ((current_vol - daily_avg_vol) / daily_avg_vol) * 100
        
        # Get PnL data
        cumulative_pnl = pair_data.get('pnl_cumulative', 0)
        
        # Get parameters
        current_br = pair_data.get('buffer_rate')
        base_br = pair_data.get('base_buffer_rate')
        recommended_br = pair_data.get('recommended_buffer_rate')
        
        current_pm = pair_data.get('position_multiplier')
        base_pm = pair_data.get('base_position_multiplier')
        recommended_pm = pair_data.get('recommended_position_multiplier')
        
        # Get tiers
        vol_tier = pair_data.get('vol_adjustment_tier', 0)
        pnl_tier = pair_data.get('pnl_adjustment_tier', 0)
        overall_tier = max(vol_tier, pnl_tier)
        
        # Apply tier filter if specified
        if tier_filter is not None and overall_tier != tier_filter:
            continue
        
        # Get last update time
        last_update = pair_data.get('last_update_time', None)
        last_update_str = last_update.strftime('%Y-%m-%d %H:%M:%S') if last_update else "Never"
        
        # Add to table data
        table_data.append({
            'pair_name': pair_name,
            'pair_type': pair_type,
            'current_vol': current_vol * 100 if current_vol is not None else None,  # Convert to percentage
            'daily_avg_vol': daily_avg_vol * 100 if daily_avg_vol is not None else None,  # Convert to percentage
            'vol_change_pct': vol_change_pct,
            'cumulative_pnl': cumulative_pnl,
            'current_br': current_br,
            'base_br': base_br,
            'recommended_br': recommended_br,
            'current_pm': current_pm,
            'base_pm': base_pm,
            'recommended_pm': recommended_pm,
            'overall_tier': overall_tier,
            'last_update': last_update_str
        })
    
    return pd.DataFrame(table_data)

# Properly working countdown timer using JavaScript
def render_countdown_timer():
    """Render a JavaScript-based countdown timer."""
    remaining_seconds = time_until_next_update()
    if remaining_seconds is not None:
        # Calculate minutes and seconds for initial display
        minutes, seconds = divmod(int(remaining_seconds), 60)
        
        # Get current time and next update time
        now_sg = get_sg_time()
        next_update_time = now_sg + timedelta(seconds=remaining_seconds)
        
        st.markdown(f"""
        <div class="update-timer">
            <div>Current time (SGT): {now_sg.strftime('%H:%M:%S')}</div>
            <div>Next auto-update in: {minutes:02d}:{seconds:02d}</div>
            <div>Next update at: {next_update_time.strftime('%H:%M:%S')}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Use Streamlit's auto-refresh for simplicity
        if remaining_seconds <= 5 and remaining_seconds > 0:
            st.rerun()

# Function to handle pair selection from the table
def select_pair(pair_name):
    """Set the selected pair and switch to detail view."""
    st.session_state.current_pair = pair_name
    st.session_state.view_mode = 'detail'
    # Update page for selected pair
    batch_update_pairs([pair_name])
    st.rerun()

# Function to render the detailed dashboard for a specific pair
def render_detail_dashboard(pair_name):
    """Render the detailed dashboard for a specific pair."""
    st.markdown(f"## Parameter Dashboard: {pair_name}")
    
    # Back to table button
    if st.button("Back to All Pairs"):
        st.session_state.view_mode = 'table'
        st.session_state.current_pair = None
        st.rerun()
    
    # Reset PnL and Update Data buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Reset PnL", key=f"reset_pnl_{pair_name}"):
            reset_pnl(pair_name)
            st.success("PnL reset successfully!")
            st.rerun()
    
    with col2:
        if st.button("Update Data", key=f"update_{pair_name}", type="primary"):
            batch_update_pairs([pair_name])
            st.success("Data updated successfully!")
            st.rerun()
    
    # Display pair type (Major or Alt)
    is_major = st.session_state.is_major_pairs.get(pair_name, False)
    pair_type = "Major" if is_major else "Alt"
    
    # Let user change pair type
    new_is_major = st.checkbox(
        "This is a major pair", 
        value=is_major,
        key=f"is_major_{pair_name}"
    )
    
    if new_is_major != is_major:
        st.session_state.is_major_pairs[pair_name] = new_is_major
        # Recalculate recommended parameters
        calculate_recommended_parameters(pair_name)
        adjust_parameters(pair_name)
        st.success(f"Changed {pair_name} to {('Major' if new_is_major else 'Alt')} pair.")
        st.rerun()
    
    st.markdown(f"**Pair Type**: {pair_type} (PnL Thresholds: {st.session_state.pnl_threshold_major_1}/{st.session_state.pnl_threshold_major_2 if is_major else st.session_state.pnl_threshold_alt_1}/{st.session_state.pnl_threshold_alt_2})")
    
    # Current metrics section
    st.markdown("### Current Metrics")
    
    # Create a grid of metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_vol = st.session_state.pair_data[pair_name].get('current_volatility')
        daily_avg = st.session_state.pair_data[pair_name].get('daily_avg_volatility')
        
        if current_vol is not None and daily_avg is not None:
            vol_pct = current_vol * 100
            avg_pct = daily_avg * 100
            vol_increase = ((current_vol - daily_avg) / daily_avg) * 100 if daily_avg > 0 else 0
            
            vol_class = "indicator-green"
            if vol_increase >= st.session_state.vol_threshold_2:
                vol_class = "indicator-red"
            elif vol_increase >= st.session_state.vol_threshold_1:
                vol_class = "indicator-yellow"
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>Current Volatility</h4>
                <p class="{vol_class}">{vol_pct:.2f}%</p>
                <small>Daily Avg: {avg_pct:.2f}% ({vol_increase:+.2f}%)</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card">
                <h4>Current Volatility</h4>
                <p>N/A</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        cumulative_pnl = st.session_state.pair_data[pair_name].get('pnl_cumulative', 0)
        threshold1 = st.session_state.pnl_threshold_major_1 if is_major else st.session_state.pnl_threshold_alt_1
        threshold2 = st.session_state.pnl_threshold_major_2 if is_major else st.session_state.pnl_threshold_alt_2
        
        pnl_class = "indicator-green"
        if cumulative_pnl <= threshold2:
            pnl_class = "indicator-red"
        elif cumulative_pnl <= threshold1:
            pnl_class = "indicator-yellow"
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>Cumulative PnL</h4>
            <p class="{pnl_class}">{cumulative_pnl:.2f}</p>
            <small>Thresholds: {threshold1} / {threshold2}</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Get current buffer rate and base buffer rate
        current_br = st.session_state.pair_data[pair_name].get('buffer_rate')
        base_br = st.session_state.pair_data[pair_name].get('base_buffer_rate')
        recommended_br = st.session_state.pair_data[pair_name].get('recommended_buffer_rate')
        
        if current_br is not None and base_br is not None:
            br_pct_change = ((current_br - base_br) / base_br) * 100
            
            br_class = "indicator-green"
            if br_pct_change > 0:
                br_class = "indicator-red" if br_pct_change >= 30 else "indicator-yellow"
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>Buffer Rate</h4>
                <p class="{br_class}">{current_br:.6f}</p>
                <small>Base: {base_br:.6f} ({br_pct_change:+.2f}%)</small>
                <small>Recommended: {recommended_br:.6f}</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card">
                <h4>Buffer Rate</h4>
                <p>N/A</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        # Get current position multiplier and base position multiplier
        current_pm = st.session_state.pair_data[pair_name].get('position_multiplier')
        base_pm = st.session_state.pair_data[pair_name].get('base_position_multiplier')
        recommended_pm = st.session_state.pair_data[pair_name].get('recommended_position_multiplier')
        
        if current_pm is not None and base_pm is not None:
            pm_pct_change = ((current_pm - base_pm) / base_pm) * 100
            
            pm_class = "indicator-green"
            if pm_pct_change < 0:
                pm_class = "indicator-red" if pm_pct_change <= -30 else "indicator-yellow"
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>Position Multiplier</h4>
                <p class="{pm_class}">{current_pm:.1f}</p>
                <small>Base: {base_pm:.1f} ({pm_pct_change:+.2f}%)</small>
                <small>Recommended: {recommended_pm:.1f}</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card">
                <h4>Position Multiplier</h4>
                <p>N/A</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Display adjustment tier information
    vol_tier = st.session_state.pair_data[pair_name].get('vol_adjustment_tier', 0)
    pnl_tier = st.session_state.pair_data[pair_name].get('pnl_adjustment_tier', 0)
    overall_tier = max(vol_tier, pnl_tier)
    
    tier_descriptions = {
        0: "Normal - Base parameters applied",
        1: f"Tier 1 - Parameters {st.session_state.parameter_adjustment_pct}% worse than base",
        2: f"Tier 2 - Parameters {st.session_state.parameter_adjustment_pct * 2}% worse than base"
    }
    
    tier_classes = {
        0: "tier-0",
        1: "tier-1",
        2: "tier-2"
    }
    
    st.markdown(f"""
    <div class="metric-card {tier_classes[overall_tier]}">
        <h4>Current Adjustment Tier</h4>
        <p>Tier {overall_tier}</p>
        <small>{tier_descriptions[overall_tier]}</small>
        <p>Volatility Tier: {vol_tier} | PnL Tier: {pnl_tier}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display charts in tabs
    tabs = st.tabs(["Volatility", "PnL", "Parameter History"])
    
    with tabs[0]:
        vol_fig = create_volatility_plot(pair_name)
        if vol_fig:
            st.plotly_chart(vol_fig, use_container_width=True)
        else:
            st.info("No volatility data available.")
    
    with tabs[1]:
        pnl_fig = create_pnl_plot(pair_name)
        if pnl_fig:
            st.plotly_chart(pnl_fig, use_container_width=True)
        else:
            st.info("No PnL data available.")
    
    with tabs[2]:
        br_fig, pm_fig = create_parameter_history_plot(pair_name)
        if br_fig and pm_fig:
            st.plotly_chart(br_fig, use_container_width=True)
            st.plotly_chart(pm_fig, use_container_width=True)
            
            # Also show history as a table
            history = st.session_state.pair_data[pair_name]['parameter_history']
            if history:
                st.markdown("### Parameter Adjustment History")
                history_df = pd.DataFrame(history, columns=['Timestamp', 'Buffer Rate', 'Position Multiplier', 'Reason'])
                st.dataframe(history_df, use_container_width=True)
        else:
            st.info("No parameter adjustment history available.")

# Function to render the main table view dashboard
def render_table_dashboard():
    """Render the main dashboard with all pairs in a simplified table."""
    st.title("Volatility & PnL Parameter Adjustment System")
    
    # Auto-update timer
    render_countdown_timer()
    
    # Add filtering options
    st.markdown("### Filter Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Filter by pair type
        pair_type_filter = st.selectbox(
            "Pair Type",
            ["All", "Major Only", "Alt Only"]
        )
        
        # Store the current filter in session state for auto-update
        filter_type = None
        if pair_type_filter == "Major Only":
            filter_type = "major"
        elif pair_type_filter == "Alt Only":
            filter_type = "alt"
    
    with col2:
        # Filter by tier
        tier_filter_options = ["All", "Tier 0 (Normal)", "Tier 1", "Tier 2"]
        tier_filter_selection = st.selectbox(
            "Adjustment Tier",
            tier_filter_options
        )
        
        # Convert to numeric tier
        tier_filter = None
        if tier_filter_selection == "Tier 0 (Normal)":
            tier_filter = 0
        elif tier_filter_selection == "Tier 1":
            tier_filter = 1
        elif tier_filter_selection == "Tier 2":
            tier_filter = 2
    
    with col3:
        # Search by pair name
        search_term = st.text_input("Search Pair Name")
    
    # Generate filtered table data
    filtered_df = generate_pairs_table_data(
        filter_type=filter_type, 
        tier_filter=tier_filter, 
        search_term=search_term
    )
    
    # Update buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        auto_update = st.checkbox(
            "Auto-update (5 min)", 
            value=st.session_state.auto_update_enabled
        )
        st.session_state.auto_update_enabled = auto_update
    
    with col2:
        if st.button("Update All Pairs", type="primary"):
            # Get all pairs that match the current filters
            available_pairs = filtered_df['pair_name'].tolist() if not filtered_df.empty else []
            
            with st.spinner(f"Updating {len(available_pairs)} pairs..."):
                # Use batch update for better performance
                batch_update_pairs(available_pairs)
            
            st.success(f"Updated {len(available_pairs)} pairs successfully!")
            st.rerun()
    
    with col3:
        if st.button("Reset All PnL"):
            with st.spinner("Resetting PnL for all pairs..."):
                # Get filtered pairs
                pairs_to_reset = filtered_df['pair_name'].tolist() if not filtered_df.empty else []
                reset_count = reset_all_pnl(pairs_to_reset)
            
            st.success(f"Reset PnL for {reset_count} pairs!")
            st.rerun()
    
    # Display the status message
    if st.session_state.update_status:
        st.info(st.session_state.update_status)
        # Clear status after displaying
        st.session_state.update_status = ""
    
    # Display the table using a simplified approach
    st.markdown("### All Trading Pairs")
    st.markdown("Click on any row to view detailed information")
    
    if not filtered_df.empty:
        # Format the data for better display
        display_df = filtered_df.copy()
        
        # Format percentage columns
        display_df['current_vol'] = display_df['current_vol'].apply(lambda x: f"{x:.2f}%" if x is not None else "N/A")
        display_df['daily_avg_vol'] = display_df['daily_avg_vol'].apply(lambda x: f"{x:.2f}%" if x is not None else "N/A")
        display_df['vol_change_pct'] = display_df['vol_change_pct'].apply(lambda x: f"{x:+.2f}%" if x is not None else "N/A")
        
        # Format numeric columns
        display_df['cumulative_pnl'] = display_df['cumulative_pnl'].apply(lambda x: f"{x:.2f}" if x is not None else "N/A")
        display_df['current_br'] = display_df['current_br'].apply(lambda x: f"{x:.6f}" if x is not None else "N/A")
        display_df['recommended_br'] = display_df['recommended_br'].apply(lambda x: f"{x:.6f}" if x is not None else "N/A")
        display_df['current_pm'] = display_df['current_pm'].apply(lambda x: f"{x:.1f}" if x is not None else "N/A")
        display_df['recommended_pm'] = display_df['recommended_pm'].apply(lambda x: f"{x:.1f}" if x is not None else "N/A")
        
        # Keep only the columns we want to display
        display_cols = [
            'pair_name', 'pair_type', 'current_vol', 'daily_avg_vol', 
            'vol_change_pct', 'cumulative_pnl', 'overall_tier', 
            'current_br', 'recommended_br', 'current_pm', 'recommended_pm', 
            'last_update'
        ]
        
        display_df = display_df[display_cols]
        
        # Create clickable dataframe with selection callback
        st.dataframe(
            display_df, 
            use_container_width=True,
            hide_index=True
        )
        
        # Create a separate selection mechanism using selectbox
        st.markdown("### Select a pair to view details")
        selected_pair = st.selectbox(
            "Choose a pair",
            options=display_df['pair_name'].tolist(),
            index=0 if not display_df.empty else None
        )
        
        if st.button("View Detailed Dashboard", key="view_details"):
            select_pair(selected_pair)
    else:
        st.warning("No pairs match the selected filters.")

# Sidebar configuration
def render_sidebar():
    """Render sidebar configuration options."""
    st.sidebar.title("Configuration")
    
    # Volatility threshold settings
    st.sidebar.markdown("### Volatility Thresholds")
    vol_threshold_1 = st.sidebar.slider(
        "Volatility Threshold 1 (%)",
        min_value=10,
        max_value=200,
        value=st.session_state.vol_threshold_1,
        step=5,
        help="Percentage increase over daily average to trigger Tier 1 adjustment"
    )
    
    vol_threshold_2 = st.sidebar.slider(
        "Volatility Threshold 2 (%)",
        min_value=vol_threshold_1 + 10,
        max_value=300,
        value=max(st.session_state.vol_threshold_2, vol_threshold_1 + 10),
        step=5,
        help="Percentage increase over daily average to trigger Tier 2 adjustment"
    )
    
    # PnL threshold settings
    st.sidebar.markdown("### PnL Thresholds for Major Pairs")
    pnl_threshold_major_1 = st.sidebar.number_input(
        "PnL Threshold 1 (Major)",
        value=st.session_state.pnl_threshold_major_1,
        step=50,
        help="PnL threshold to trigger Tier 1 adjustment for major pairs"
    )
    
    pnl_threshold_major_2 = st.sidebar.number_input(
        "PnL Threshold 2 (Major)",
        value=min(st.session_state.pnl_threshold_major_2, pnl_threshold_major_1 - 50),
        step=50,
        help="PnL threshold to trigger Tier 2 adjustment for major pairs"
    )
    
    st.sidebar.markdown("### PnL Thresholds for Alt Pairs")
    pnl_threshold_alt_1 = st.sidebar.number_input(
        "PnL Threshold 1 (Alts)",
        value=st.session_state.pnl_threshold_alt_1,
        step=20,
        help="PnL threshold to trigger Tier 1 adjustment for altcoin pairs"
    )
    
    pnl_threshold_alt_2 = st.sidebar.number_input(
        "PnL Threshold 2 (Alts)",
        value=min(st.session_state.pnl_threshold_alt_2, pnl_threshold_alt_1 - 20),
        step=20,
        help="PnL threshold to trigger Tier 2 adjustment for altcoin pairs"
    )
    
    # Parameter adjustment percentage
    st.sidebar.markdown("### Parameter Adjustment")
    parameter_adjustment_pct = st.sidebar.slider(
        "Parameter Adjustment (%)",
        min_value=5,
        max_value=50,
        value=st.session_state.parameter_adjustment_pct,
        step=5,
        help="Percentage to adjust parameters at each tier"
    )
    
    # Update session state with new settings
    st.session_state.vol_threshold_1 = vol_threshold_1
    st.session_state.vol_threshold_2 = vol_threshold_2
    st.session_state.pnl_threshold_major_1 = pnl_threshold_major_1
    st.session_state.pnl_threshold_major_2 = pnl_threshold_major_2
    st.session_state.pnl_threshold_alt_1 = pnl_threshold_alt_1
    st.session_state.pnl_threshold_alt_2 = pnl_threshold_alt_2
    st.session_state.parameter_adjustment_pct = parameter_adjustment_pct
    
    # Apply settings button
    if st.sidebar.button("Apply Settings"):
        # Re-apply parameter adjustments to all pairs
        for pair_name in st.session_state.pair_data:
            calculate_recommended_parameters(pair_name)
            adjust_parameters(pair_name)
        st.sidebar.success("Settings applied to all pairs!")
        st.rerun()

# Main application
def main():
    # Initialize session state
    init_session_state()
    
    # Check if daily reset is needed
    if check_daily_reset():
        st.success("Daily PnL reset completed.")
    
    # Check if auto-update is needed
    if check_auto_update():
        # Get the current view mode
        if st.session_state.view_mode == 'table':
            # If we're on the table view, update all pairs
            st.session_state.update_status = "Auto-updating pairs..."
            
            # Get available pairs
            available_pairs = fetch_pairs()
            
            # Limit the number of pairs to update to improve performance
            # This helps avoid timeout issues during auto-refresh
            max_pairs_to_update = 20
            if len(available_pairs) > max_pairs_to_update:
                # Prioritize major pairs and pairs with higher tiers
                major_pairs = [p for p in available_pairs if st.session_state.is_major_pairs.get(p, False)]
                major_pairs = major_pairs[:max_pairs_to_update]
                
                if len(major_pairs) < max_pairs_to_update:
                    # Add some alt pairs if we have room
                    alt_pairs = [p for p in available_pairs if not st.session_state.is_major_pairs.get(p, False)]
                    alt_pairs = alt_pairs[:max_pairs_to_update-len(major_pairs)]
                    pairs_to_update = major_pairs + alt_pairs
                else:
                    pairs_to_update = major_pairs[:max_pairs_to_update]
                    
                batch_update_pairs(pairs_to_update)
                st.session_state.update_status = f"Auto-updated {len(pairs_to_update)} pairs. Full update recommended."
            else:
                # Update all pairs if the number is manageable
                batch_update_pairs(available_pairs)
                st.session_state.update_status = f"Auto-updated all {len(available_pairs)} pairs."
        else:
            # We're in detail view, just update the current pair
            current_pair = st.session_state.current_pair
            if current_pair:
                batch_update_pairs([current_pair])
                st.session_state.update_status = f"Auto-updated {current_pair}."
        
        st.rerun()
    
    # Render sidebar
    render_sidebar()
    
    # View selection
    if st.session_state.view_mode == 'detail' and st.session_state.current_pair:
        # Render detailed view for selected pair
        render_detail_dashboard(st.session_state.current_pair)
    else:
        # Render table view
        render_table_dashboard()

if __name__ == "__main__":
    main()