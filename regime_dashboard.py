import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sqlalchemy import create_engine
from datetime import datetime, timedelta

# --- DB CONFIG ---
db_config = st.secrets["database"]

db_uri = (
    f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}"
    f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
)
engine = create_engine(db_uri)

# --- UI Sidebar ---
st.title("Rolling tHurst Exponent Dashboard")

# --- Fetch token list from DB ---
@st.cache_data
def fetch_token_list():
    query = "SELECT DISTINCT pair_name FROM public.oracle_price_log ORDER BY pair_name"
    df = pd.read_sql(query, engine)
    return df['pair_name'].tolist()

token_list = fetch_token_list()
selected_token = st.selectbox("Select Token", token_list, index=0)
timeframe = st.selectbox("Timeframe", ["30s", "15min", "30min", "1h", "6h"], index=2)
lookback_days = st.slider("Lookback (Days)", 1, 30, 2)
rolling_window = st.slider("Rolling Window (Bars)", 5, 50, 11)

# --- Fetch Oracle Price Data ---
end_time = datetime.utcnow()
start_time = end_time - timedelta(days=lookback_days)

query = f"""
SELECT created_at AT TIME ZONE 'UTC' + INTERVAL '8 hours' AS timestamp, final_price, pair_name
FROM public.oracle_price_log
WHERE created_at BETWEEN '{start_time}' AND '{end_time}'
AND pair_name = '{selected_token}';
"""
df = pd.read_sql(query, engine)

if df.empty:
    st.warning("No data found for selected pair and timeframe.")
    st.stop()

# --- Preprocess ---
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp').sort_index()

# --- Resample to OHLC ---
ohlc = df['final_price'].resample(timeframe).ohlc().dropna()

# --- More Robust Hurst Calculation ---
def compute_hurst(ts):
    ts = np.array(ts)
    if len(ts) < 20 or np.std(ts) == 0:
        return np.nan
    lags = range(2, 20)
    tau = []
    for lag in lags:
        diff = ts[lag:] - ts[:-lag]
        if len(diff) == 0 or np.std(diff) == 0:
            return np.nan
        tau.append(np.std(diff))
    if len(tau) == 0 or any(t == 0 for t in tau):
        return np.nan
    try:
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]
    except:
        return np.nan

# --- Debug Display: Sample Input Series ---
st.subheader(f"Rolling tHurst for {selected_token} ({timeframe})")
st.caption("Sample Close Prices (first 20 rows):")
st.write(ohlc['close'].head(20).tolist())

# --- Compute Rolling tHurst ---
ohlc['tHurst'] = ohlc['close'].rolling(rolling_window).apply(compute_hurst)

# --- Plot ---
if ohlc['tHurst'].dropna().empty:
    st.warning("⚠️ No valid tHurst values computed. Price may be too flat or lookback too short.")
else:
    fig = px.line(ohlc.reset_index(), x='timestamp', y='tHurst',
                  title=f"Rolling tHurst for {selected_token} ({timeframe})")
    fig.update_layout(yaxis_title="tHurst Exponent", xaxis_title="Time")
    st.plotly_chart(fig, use_container_width=True)

# --- Optional: Show table ---
st.dataframe(ohlc.tail(50))
