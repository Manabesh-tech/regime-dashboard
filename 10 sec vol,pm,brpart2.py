

# Save this as optimized_10sec_volatility.py import streamlit as st import pandas as pd import numpy as np import plotly.graph_objects as go from plotly.subplots import make_subplots from datetime import datetime, timedelta import psycopg2 import pytz from sqlalchemy import create_engine

pasted

consider this.

below i want to add buffer rates from UAT mode to test if it its in line with Rollbit buffer rates as they change with time.

here is the sql for it

i just care about buffer rate from this table and timeplot of it alongwith the rest
the only catch is this table exists in different database
following are the details of it

[database]user = "report_dev"password = "aTJ92^kl04hllk"host = "http://aws-jp-tk-surf-pg-public.cluster-csteuf9lw8dv.ap-northeast-1.rds.amazonaws.com/"port = 5432database = "public_rw"

other stuff is in production database the details of which are in secret file already.
is everything clear?

Yes, I understand! You want to add UAT buffer rates alongside the existing Rollbit buffer rates to compare them over time. The UAT buffer rates come from a different database with the credentials you provided.

Here's the optimized code that adds UAT buffer rates to your existing dashboard:


Optimized 10sec Volatility with UAT Buffer Rates
Code 
I've updated your dashboard to include UAT buffer rates alongside the Rollbit buffer rates. Here are the key changes:

Added UAT Database Connection: Created a separate connection to the public_rw database with the credentials you provided.
New Function: Added fetch_uat_buffer_rates_10sec() that queries the leverage_config table from the UAT database and resamples the buffer rates to 10-second intervals.
Expanded Dashboard Layout:
Changed from 3 to 4 panels
Added a dedicated panel for UAT Buffer Rate (in purple)
Adjusted panel heights to accommodate the new visualization
Updated Hover Information: The hover tooltip now shows both Rollbit and UAT buffer rates for easy comparison.
Buffer Rate Comparison: Added a new metrics section that shows:
Current Rollbit Buffer Rate
Current UAT Buffer Rate
The difference between UAT and Rollbit (absolute and percentage)
Visual Improvements:
UAT buffer rate is displayed in purple to distinguish it from Rollbit (dark green)
Both buffer rate panels use the same scale formatting (3 decimal places)
Auto-scaling for both buffer rate panels
The dashboard now provides a comprehensive view to compare how UAT buffer rates track against Rollbit buffer rates over time, making it easy to verify if they're in line with each other.




Error fetching UAT buffer rates: (psycopg2.OperationalError) connection to server at "aws-jp-tk-surf-pg-public.cluster-csteuf9lw8dv.ap-northeast-1.rds.amazonaws.com" (52.199.27.96), port 5432 failed: FATAL: password authentication failed for user "report_dev" connection to server at "aws-jp-tk-surf-pg-public.cluster-csteuf9lw8dv.ap-northeast-1.rds.amazonaws.com" (52.199.27.96), port 5432 failed: FATAL: password authentication failed for user "report_dev"
(Background on this error at: https://sqlalche.me/e/20/e3q8)

I see the authentication error. Looking at the host URL you provided, there's an issue - the host should not include "http://" prefix. Let me fix that:


Optimized 10sec Volatility with UAT Buffer Rates
Code ∙ Version 2 
I notice the UAT database is using the same host as the production database. This is correct - PostgreSQL can have multiple databases on the same server. The different user credentials and database name should distinguish between them.

If you're still getting authentication errors, could you please verify:

The password is correct (it's different from the production password)
The database name public_rw is correct
The user report_dev has access to this database
You might also want to test the connection directly using a tool like psql or pgAdmin to verify the credentials:

bash
psql -h aws-jp-tk-surf-pg-public.cluster-csteuf9lw8dv.ap-northeast-1.rds.amazonaws.com -p 5432 -U report_dev -d public_rw




i clearly told you that UAT database is different and i gave you username and password also for it

You're absolutely right - I apologize for the confusion. You provided different credentials for the UAT database. Let me check the credentials again based on what you originally provided:

User: report_dev
Password: aTJ92^kl04hllk
Database: public_rw
Host: You mentioned it was at http://aws-jp-tk-surf-pg-public.cluster-csteuf9lw8dv.ap-northeast-1.rds.amazonaws.com/
The issue is likely the http:// prefix in the host URL. PostgreSQL connections should not include the protocol prefix. Let me update the code to remove it:


Optimized 10sec Volatility with UAT Buffer Rates
Code ∙ Version 3 
Wait, I see the code already has the correct host without the http:// prefix. The authentication error suggests the credentials might not be correct. Could you please double-check:

Is the password exactly aTJ92^kl04hllk? (No extra spaces or characters)
Is the database name exactly public_rw?
Is the username exactly report_dev?
If these are all correct and you're still getting authentication errors, it's possible that:

The user might not exist on that database server
The password might be incorrect
The user might not have permissions to access the public_rw database
Could you verify these credentials work when connecting directly to the database?




the attached file had details from production database the details of which was already provided 

i am just adding the uat version from a different database

didnt you understand what i am trying to do?

I completely understand what you're trying to do - you want to add UAT buffer rates from a different database to compare with the production Rollbit buffer rates that are already working.

I see the confusion now. You're saying that the UAT database is indeed different from the production database. Let me verify:

Production database (already in your code):

Host: aws-jp-tk-surf-pg-public.cluster-csteuf9lw8dv.ap-northeast-1.rds.amazonaws.com
Database: replication_report
User: public_replication
Password: 866^FKC4hllk
UAT database (what you provided):

Host: Same server but different database
Database: public_rw
User: report_dev
Password: aTJ92^kl04hllk
The authentication error suggests the credentials for the UAT database aren't working. Can you confirm:

Are these the exact credentials for the UAT database?
Should the UAT database be on the same server/host as production, or is it on a different server?
If the UAT database is on a completely different server, please provide the correct host/server address for it.




in this latest code , i dont even see the earlier stuff i was seeing which was being extracted from production database only

so this combined code is worse and shows me nothing

You're absolutely right - I apologize. Let me fix this issue. The code should be working with the production database as it was before, and only adding the UAT buffer rates as an additional comparison. Let me check what's happening:


Optimized 10sec Volatility with UAT Buffer Rates
Code ∙ Version 4 
Let me also fix the main visualization logic to ensure the production data still shows even if UAT data fails:


Optimized 10sec Volatility with UAT Buffer Rates
Code ∙ Version 5 
Now let me also ensure the production panels still show correctly even when UAT fails:


Optimized 10sec Volatility with UAT Buffer Rates
Code ∙ Version 6 
Now I need to fix the panel rendering logic to adapt to whether we have UAT data or not:


Optimized 10sec Volatility with UAT Buffer Rates
Code ∙ Version 7 
Now let me fix the axes update logic and the height:


Optimized 10sec Volatility with UAT Buffer Rates
Code ∙ Version 8 
Let me also fix the y-axes update logic:


Optimized 10sec Volatility with UAT Buffer Rates
Code ∙ Version 9 
Now the code should work correctly:

It will show all production data as before - volatility, Rollbit buffer rates, and position multiplier
It will gracefully handle UAT database connection failures - if the UAT connection fails, it will still show all production data with 3 panels instead of 4
If UAT data is available, it will show in a 4th panel for comparison
The key improvements:

The UAT fetch function now returns an empty DataFrame instead of None on error
The chart adapts to show 3 or 4 panels based on whether UAT data is available
The production data will always show regardless of UAT connection status
Error messages will appear for UAT connection issues but won't break the dashboard
If the UAT database credentials are still not working, the dashboard will continue to function with just the production data. Once you get the correct UAT credentials, that data will appear automatically.




i pasted this , github said i have made no changes???

If GitHub shows no changes, it means the code hasn't been modified from what's already there. Let me create a completely new version with the correct changes that adds UAT functionality while preserving all the original production functionality:


Optimized 10sec Volatility with UAT Buffer Rates
Code 
This is a complete rewrite that:

Preserves all original production functionality - volatility, Rollbit buffer rates, and position multiplier from the production database
Adds UAT buffer rate comparison - from the separate public_rw database with different credentials
Handles failures gracefully - if UAT connection fails, the dashboard still works with just production data
Shows clear database separation - comments clearly indicate which database each function uses
Adapts the layout dynamically - shows 3 or 4 panels depending on available data
Key changes from the original:

Added UAT database connection with the credentials you provided
Added UAT buffer rate fetching from the leverage_config table
Dynamic layout that adjusts panels based on available data
Buffer rate compariso




# Save this as optimized_10sec_volatility_with_uat.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import psycopg2
import pytz
from sqlalchemy import create_engine

st.set_page_config(page_title="10sec Volatility Plot with Rollbit & UAT", page_icon="📈", layout="wide")

# --- UI Setup ---
st.title("10-Second Volatility Plot with Rollbit & UAT Buffer Comparison")

# Production DB connection
db_params = {
    'host': 'aws-jp-tk-surf-pg-public.cluster-csteuf9lw8dv.ap-northeast-1.rds.amazonaws.com',
    'port': 5432,
    'database': 'replication_report',
    'user': 'public_replication',
    'password': '866^FKC4hllk'
}

engine = create_engine(
    f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['database']}"
)

conn = psycopg2.connect(**db_params)

# UAT DB connection - DIFFERENT DATABASE
uat_db_params = {
    'host': 'aws-jp-tk-surf-pg-public.cluster-csteuf9lw8dv.ap-northeast-1.rds.amazonaws.com',
    'port': 5432,
    'database': 'public_rw',  # Different database
    'user': 'report_dev',     # Different user
    'password': 'aTJ92^kl04hllk'  # Different password
}

try:
    uat_engine = create_engine(
        f"postgresql://{uat_db_params['user']}:{uat_db_params['password']}@{uat_db_params['host']}:{uat_db_params['port']}/{uat_db_params['database']}"
    )
    uat_connection_status = True
except Exception as e:
    st.error(f"Could not establish UAT database connection: {e}")
    uat_connection_status = False

# Cache token list for longer
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_trading_pairs():
    query = """
    SELECT pair_name 
    FROM trade_pool_pairs 
    WHERE status = 1
    ORDER BY pair_name
    """
    df = pd.read_sql_query(query, engine)
    return df['pair_name'].tolist()

all_tokens = fetch_trading_pairs()

col1, col2 = st.columns([3, 1])

with col1:
    default_token = "BTC/USDT" if "BTC/USDT" in all_tokens else all_tokens[0]
    selected_token = st.selectbox(
        "Select Token",
        all_tokens,
        index=all_tokens.index(default_token) if default_token in all_tokens else 0
    )

with col2:
    if st.button("Refresh Data", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

sg_tz = pytz.timezone('Asia/Singapore')
now_sg = datetime.now(sg_tz)
st.write(f"Current time (Singapore): {now_sg.strftime('%Y-%m-%d %H:%M:%S')}")

# Optimized Rollbit fetch - resample to 10 seconds (PRODUCTION)
@st.cache_data(ttl=60)
def fetch_rollbit_parameters_10sec(token, hours=3):
    """Fetch Rollbit parameters with 10-second resolution from PRODUCTION"""
    try:
        now_sg = datetime.now(pytz.timezone('Asia/Singapore'))
        start_time_sg = now_sg - timedelta(hours=hours)
        
        start_str = start_time_sg.strftime("%Y-%m-%d %H:%M:%S")
        end_str = now_sg.strftime("%Y-%m-%d %H:%M:%S")

        query = f"""
         SELECT 
            pair_name,
            bust_buffer AS buffer_rate,
            position_multiplier,
            created_at + INTERVAL '8 hour' AS timestamp
        FROM rollbit_pair_config 
        WHERE pair_name = '{token}'
        AND created_at >= '{start_str}'::timestamp - INTERVAL '8 hour'
        AND created_at <= '{end_str}'::timestamp - INTERVAL '8 hour'
        ORDER BY created_at
        """

        df = pd.read_sql_query(query, engine)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').sort_index()
            # Resample to 10 seconds
            df = df.resample('10s').ffill()
        return df
    except Exception as e:
        st.error(f"Error fetching Rollbit parameters: {e}")
        return None

# NEW: Fetch UAT buffer rates from DIFFERENT DATABASE
@st.cache_data(ttl=60)
def fetch_uat_buffer_rates_10sec(token, hours=3):
    """Fetch UAT buffer rates with 10-second resolution from UAT DATABASE"""
    if not uat_connection_status:
        return pd.DataFrame()
    
    try:
        now_sg = datetime.now(pytz.timezone('Asia/Singapore'))
        start_time_sg = now_sg - timedelta(hours=hours)
        
        start_str = start_time_sg.strftime("%Y-%m-%d %H:%M:%S")
        end_str = now_sg.strftime("%Y-%m-%d %H:%M:%S")

        query = f"""
        SELECT 
            pair_name,
            buffer AS buffer_rate,
            created_at + INTERVAL '8 hour' AS timestamp
        FROM leverage_config 
        WHERE pair_name = '{token}'
        AND created_at >= '{start_str}'::timestamp - INTERVAL '8 hour'
        AND created_at <= '{end_str}'::timestamp - INTERVAL '8 hour'
        ORDER BY created_at
        """

        df = pd.read_sql_query(query, uat_engine)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').sort_index()
            # Resample to 10 seconds
            df = df.resample('10s').ffill()
        return df
    except Exception as e:
        st.error(f"Error fetching UAT buffer rates: {e}")
        return pd.DataFrame()

# 10-second volatility calculation (PRODUCTION)
@st.cache_data(ttl=30)
def get_volatility_data_10sec(token, hours=3):
    now_sg = datetime.now(pytz.timezone('Asia/Singapore'))
    start_time_sg = now_sg - timedelta(hours=hours)
    
    # Get partitions
    today_str = now_sg.strftime("%Y%m%d")
    yesterday_str = (now_sg - timedelta(days=1)).strftime("%Y%m%d")
    
    start_time_str = start_time_sg.strftime("%Y-%m-%d %H:%M:%S")
    end_time_str = now_sg.strftime("%Y-%m-%d %H:%M:%S")
    
    # Try today's partition first
    query = f"""
    SELECT 
        created_at + INTERVAL '8 hour' AS timestamp,
        final_price
    FROM public.oracle_price_log_partition_{today_str}
    WHERE created_at >= '{start_time_str}'::timestamp - INTERVAL '8 hour'
    AND created_at <= '{end_time_str}'::timestamp - INTERVAL '8 hour'
    AND source_type = 0
    AND pair_name = '{token}'
    ORDER BY timestamp
    """
    
    try:
        df = pd.read_sql_query(query, engine)
        
        # If we don't have enough data, try yesterday's partition too
        if df.empty or len(df) < 10:
            query_yesterday = f"""
            SELECT 
                created_at + INTERVAL '8 hour' AS timestamp,
                final_price
            FROM public.oracle_price_log_partition_{yesterday_str}
            WHERE created_at >= '{start_time_str}'::timestamp - INTERVAL '8 hour'
            AND created_at <= '{end_time_str}'::timestamp - INTERVAL '8 hour'
            AND source_type = 0
            AND pair_name = '{token}'
            ORDER BY timestamp
            """
            try:
                df_yesterday = pd.read_sql_query(query_yesterday, engine)
                df = pd.concat([df_yesterday, df]).drop_duplicates().sort_values('timestamp')
            except:
                pass
    except Exception as e:
        st.error(f"Query error for {token}: {str(e)}")
        return None, None
    
    if df.empty:
        return None, None
    
    # Process timestamps
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()
    
    # Resample to 500ms
    price_data = df['final_price'].resample('500ms').ffill().dropna()
    
    if len(price_data) < 2:
        return None, None
    
    # Create 10-second windows
    result = []
    start_date = price_data.index.min().floor('10s')
    end_date = price_data.index.max().ceil('10s')
    ten_sec_periods = pd.date_range(start=start_date, end=end_date, freq='10s')
    
    for i in range(len(ten_sec_periods)-1):
        start_window = ten_sec_periods[i]
        end_window = ten_sec_periods[i+1]
        
        window_data = price_data[(price_data.index >= start_window) & (price_data.index < end_window)]
        
        if len(window_data) >= 2:
            # Calculate volatility for 10-second window
            log_returns = np.diff(np.log(window_data.values))
            if len(log_returns) > 0:
                annualization_factor = np.sqrt(3153600)  # For 10-second windows (31,536,000 seconds in a year / 10)
                volatility = np.std(log_returns) * annualization_factor
                
                result.append({
                    'timestamp': start_window,
                    'realized_vol': volatility
                })
    
    if not result:
        return None, None
    
    result_df = pd.DataFrame(result).set_index('timestamp')
    
    # Calculate percentiles for the last 3 hours
    vol_pct = result_df['realized_vol'] * 100
    if len(vol_pct) > 0:
        percentiles = {
            'p25': np.percentile(vol_pct, 25),
            'p50': np.percentile(vol_pct, 50),
            'p75': np.percentile(vol_pct, 75),
            'p95': np.percentile(vol_pct, 95)
        }
    else:
        percentiles = {'p25': 0, 'p50': 0, 'p75': 0, 'p95': 0}
    
    return result_df, percentiles

# Main chart section
with st.spinner(f"Loading data for {selected_token}..."):
    vol_data, percentiles = get_volatility_data_10sec(selected_token)
    rollbit_params = fetch_rollbit_parameters_10sec(selected_token)
    uat_buffer = fetch_uat_buffer_rates_10sec(selected_token)

if vol_data is not None and not vol_data.empty:
    # Convert to percentage
    vol_data_pct = vol_data.copy()
    vol_data_pct['realized_vol'] = vol_data_pct['realized_vol'] * 100

    # Key metrics
    current_vol = vol_data_pct['realized_vol'].iloc[-1]
    avg_vol = vol_data_pct['realized_vol'].mean()
    max_vol = vol_data_pct['realized_vol'].max()
    min_vol = vol_data_pct['realized_vol'].min()
    
    # Calculate current percentile
    all_vols = vol_data_pct['realized_vol'].values
    current_percentile = (all_vols < current_vol).mean() * 100

    # Determine how many rows we need
    has_uat_data = uat_buffer is not None and not uat_buffer.empty
    has_rollbit_data = rollbit_params is not None and not rollbit_params.empty
    
    if has_uat_data and has_rollbit_data:
        num_rows = 4
        subplot_titles = (
            f"{selected_token} Annualized Volatility (10sec windows)",
            "Rollbit Buffer Rate (%)",
            "UAT Buffer Rate (%)", 
            "Rollbit Position Multiplier"
        )
        row_heights = [0.3, 0.23, 0.23, 0.24]
    else:
        num_rows = 3
        subplot_titles = (
            f"{selected_token} Annualized Volatility (10sec windows)",
            "Rollbit Buffer Rate (%)",
            "Rollbit Position Multiplier"
        )
        row_heights = [0.4, 0.3, 0.3]

    # Create subplots
    fig = make_subplots(
        rows=num_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=subplot_titles,
        row_heights=row_heights
    )

    # Process all data into combined dataframe
    combined_data = vol_data_pct.copy()
    
    # Add Rollbit data if available
    if has_rollbit_data:
        rollbit_params['buffer_rate_pct'] = rollbit_params['buffer_rate'] * 100
        combined_data = pd.merge(
            combined_data,
            rollbit_params[['buffer_rate_pct', 'position_multiplier']],
            left_index=True,
            right_index=True,
            how='left',
            suffixes=('', '_rollbit')
        )
        combined_data['buffer_rate_pct'] = combined_data['buffer_rate_pct'].ffill()
        combined_data['position_multiplier'] = combined_data['position_multiplier'].ffill()
    else:
        combined_data['buffer_rate_pct'] = np.nan
        combined_data['position_multiplier'] = np.nan
    
    # Add UAT data if available
    if has_uat_data:
        uat_buffer['uat_buffer_rate_pct'] = uat_buffer['buffer_rate'] * 100
        combined_data = pd.merge(
            combined_data,
            uat_buffer[['uat_buffer_rate_pct']],
            left_index=True,
            right_index=True,
            how='left'
        )
        combined_data['uat_buffer_rate_pct'] = combined_data['uat_buffer_rate_pct'].ffill()
    else:
        combined_data['uat_buffer_rate_pct'] = np.nan
    
    # Create unified hover data
    hover_template = (
        "<b>Time: %{x}</b><br>" +
        "Volatility: %{customdata[0]:.1f}%<br>" +
        "Rollbit Buffer: %{customdata[1]:.3f}%<br>" +
        "UAT Buffer: %{customdata[2]:.3f}%<br>" +
        "Position Mult: %{customdata[3]:,.0f}<br>" +
        "<extra></extra>"
    )
    
    customdata = np.column_stack((
        combined_data['realized_vol'],
        combined_data['buffer_rate_pct'].fillna(0),
        combined_data['uat_buffer_rate_pct'].fillna(0),
        combined_data['position_multiplier'].fillna(0)
    ))
    
    # Panel 1: Volatility (always shown)
    fig.add_trace(
        go.Scatter(
            x=combined_data.index,
            y=combined_data['realized_vol'],
            mode='lines',
            line=dict(color='blue', width=2),
            name="Volatility (%)",
            customdata=customdata,
            hovertemplate=hover_template,
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Panel 2: Rollbit Buffer Rate
    if has_rollbit_data:
        fig.add_trace(
            go.Scatter(
                x=combined_data.index,
                y=combined_data['buffer_rate_pct'],
                mode='lines+markers',
                line=dict(color='darkgreen', width=3),
                marker=dict(size=4),
                name="Rollbit Buffer Rate (%)",
                customdata=customdata,
                hovertemplate=hover_template,
                showlegend=False
            ),
            row=2, col=1
        )
        latest_rollbit_buffer = combined_data['buffer_rate_pct'].iloc[-1]
    else:
        fig.add_annotation(
            x=0.5,
            y=0.5,
            text="No Rollbit data available",
            showarrow=False,
            font=dict(size=12),
            xref="x2 domain",
            yref="y2 domain",
            row=2, col=1
        )
        latest_rollbit_buffer = None
    
    # Panel 3: UAT Buffer Rate (only if we have UAT data)
    if has_uat_data:
        uat_row = 3
        fig.add_trace(
            go.Scatter(
                x=combined_data.index,
                y=combined_data['uat_buffer_rate_pct'],
                mode='lines+markers',
                line=dict(color='purple', width=3),
                marker=dict(size=4),
                name="UAT Buffer Rate (%)",
                customdata=customdata,
                hovertemplate=hover_template,
                showlegend=False
            ),
            row=uat_row, col=1
        )
        latest_uat_buffer = combined_data['uat_buffer_rate_pct'].iloc[-1]
        pos_mult_row = 4
    else:
        latest_uat_buffer = None
        pos_mult_row = 3
    
    # Panel 4 (or 3): Position Multiplier
    if has_rollbit_data:
        fig.add_trace(
            go.Scatter(
                x=combined_data.index,
                y=combined_data['position_multiplier'],
                mode='lines+markers',
                line=dict(color='darkblue', width=3),
                marker=dict(size=4),
                name="Position Multiplier",
                customdata=customdata,
                hovertemplate=hover_template,
                showlegend=False
            ),
            row=pos_mult_row, col=1
        )
        latest_pos_mult = combined_data['position_multiplier'].iloc[-1]
    else:
        fig.add_annotation(
            x=0.5,
            y=0.5,
            text="No Rollbit data available",
            showarrow=False,
            font=dict(size=12),
            xref=f"x{pos_mult_row} domain",
            yref=f"y{pos_mult_row} domain",
            row=pos_mult_row, col=1
        )
        latest_pos_mult = None
    
    # Add percentile lines to volatility panel
    percentile_lines = [
        ('p25', '#2ECC71', '25th'),
        ('p50', '#3498DB', '50th'),
        ('p75', '#F39C12', '75th'),
        ('p95', '#E74C3C', '95th')
    ]

    for key, color, label in percentile_lines:
        fig.add_hline(
            y=percentiles[key],
            line_dash="dash",
            line_color=color,
            line_width=2,
            annotation_text=f"{label}: {percentiles[key]:.1f}%",
            annotation_position="left",
            annotation_font_color=color,
            row=1, col=1
        )

    # Create title
    title_parts = [f"{selected_token} Analysis Dashboard (10-second windows)<br>"]
    subtitle_parts = [f"Current Vol: {current_vol:.1f}% ({current_percentile:.0f}th percentile)"]
    
    if latest_rollbit_buffer is not None:
        subtitle_parts.append(f"Rollbit Buffer: {latest_rollbit_buffer:.3f}%")
    if latest_uat_buffer is not None:
        subtitle_parts.append(f"UAT Buffer: {latest_uat_buffer:.3f}%")
    if latest_pos_mult is not None:
        subtitle_parts.append(f"Pos Mult: {latest_pos_mult:,.0f}")
    
    title_text = title_parts[0] + f"<sub>{' | '.join(subtitle_parts)}</sub>"

    # Update layout
    fig.update_layout(
        title=title_text,
        height=1000 if has_uat_data else 800,
        showlegend=False,
        hovermode="x unified",
        plot_bgcolor='white',
        paper_bgcolor='white',
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="black",
            font_size=12
        ),
        xaxis=dict(
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikethickness=2,
            spikecolor="gray",
            spikedash="solid"
        )
    )

    # Update all x-axes to have spikes
    for i in range(1, num_rows + 1):
        fig.update_xaxes(
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikethickness=2,
            spikecolor="gray",
            spikedash="solid",
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray',
            row=i, col=1
        )

    # Update y-axes with auto-scaling
    fig.update_yaxes(
        title_text="Volatility (%)",
        row=1, col=1,
        showgrid=True,
        gridwidth=1,
        gridcolor='LightGray',
        range=[0, max(max_vol * 1.1, percentiles['p95'] * 1.1, 5)]
    )
    
    # Rollbit buffer rate
    if has_rollbit_data:
        buffer_min = combined_data['buffer_rate_pct'].min()
        buffer_max = combined_data['buffer_rate_pct'].max()
        fig.update_yaxes(
            title_text="Rollbit Buffer (%)",
            row=2, col=1,
            tickformat=".3f",
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray',
            range=[buffer_min * 0.95, buffer_max * 1.05] if buffer_max > buffer_min else None
        )
    
    # UAT buffer rate
    if has_uat_data:
        uat_buffer_min = combined_data['uat_buffer_rate_pct'].min()
        uat_buffer_max = combined_data['uat_buffer_rate_pct'].max()
        fig.update_yaxes(
            title_text="UAT Buffer (%)", 
            row=3, col=1,
            tickformat=".3f",
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray',
            range=[uat_buffer_min * 0.95, uat_buffer_max * 1.05] if uat_buffer_max > uat_buffer_min else None
        )
    
    # Position multiplier
    if has_rollbit_data:
        pos_mult_min = combined_data['position_multiplier'].min()
        pos_mult_max = combined_data['position_multiplier'].max()
        fig.update_yaxes(
            title_text="Position Multiplier",
            row=pos_mult_row, col=1,
            tickformat=",",
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray',
            range=[pos_mult_min * 0.95, pos_mult_max * 1.05] if pos_mult_max > pos_mult_min else None
        )
    
    # X-axis labels only on bottom
    fig.update_xaxes(title_text="Time (Singapore)", row=num_rows, col=1, tickformat="%H:%M:%S<br>%m/%d")

    st.plotly_chart(fig, use_container_width=True)

    # Metrics display with percentiles
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current", f"{current_vol:.1f}%", f"{current_percentile:.0f}th %ile")
    with col2:
        st.metric("Average", f"{avg_vol:.1f}%")
    with col3:
        st.metric("Max", f"{max_vol:.1f}%")
    with col4:
        st.metric("Min", f"{min_vol:.1f}%")
    
    # Percentile display
    st.markdown("### Percentiles (3h)")
    pcol1, pcol2, pcol3, pcol4 = st.columns(4)
    with pcol1:
        st.metric("25th", f"{percentiles['p25']:.1f}%")
    with pcol2:
        st.metric("50th", f"{percentiles['p50']:.1f}%")
    with pcol3:
        st.metric("75th", f"{percentiles['p75']:.1f}%")
    with pcol4:
        st.metric("95th", f"{percentiles['p95']:.1f}%")

    # Current parameters section
    st.markdown("### Current Parameters")
    
    # Buffer rate comparison
    if has_rollbit_data or has_uat_data:
        bcol1, bcol2, bcol3 = st.columns(3)
        
        with bcol1:
            if latest_rollbit_buffer is not None:
                st.metric("Rollbit Buffer Rate", f"{latest_rollbit_buffer:.3f}%")
            else:
                st.metric("Rollbit Buffer Rate", "N/A")
        
        with bcol2:
            if latest_uat_buffer is not None:
                st.metric("UAT Buffer Rate", f"{latest_uat_buffer:.3f}%")
            else:
                st.metric("UAT Buffer Rate", "N/A")
        
        with bcol3:
            if latest_rollbit_buffer is not None and latest_uat_buffer is not None:
                diff = latest_uat_buffer - latest_rollbit_buffer
                diff_pct = (diff / latest_rollbit_buffer * 100) if latest_rollbit_buffer != 0 else 0
                st.metric("UAT vs Rollbit", f"{diff:.3f}%", f"{diff_pct:.1f}% diff")
            else:
                st.metric("UAT vs Rollbit", "N/A")
    
    # Position multiplier
    if latest_pos_mult is not None:
        st.metric("Current Position Multiplier", f"{latest_pos_mult:,.0f}")

else:
    st.error("No volatility data available for the selected token")
