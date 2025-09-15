import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from sqlalchemy import create_engine, text
import warnings

warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Crypto Exchange Analysis",
    page_icon="📊",
    layout="wide"
)

# Clear cache
st.cache_data.clear()

# Database connection
@st.cache_resource
def get_engine():
    return create_engine(
        "postgresql://public_replication:866^FKC4hllk@aws-jp-tk-surf-pg-public.cluster-csteuf9lw8dv.ap-northeast-1.rds.amazonaws.com:5432/replication_report",
        pool_pre_ping=True
    )

engine = get_engine()

# Token mapping
def map_token(surf_token):
    mapping = {"PUMP/USDT": "1000PUMP/USDT"}
    return mapping.get(surf_token, surf_token)

# Main analyzer
class Analyzer:
    def __init__(self):
        self.point_counts = [500, 1500, 2500, 5000]
        self.data = {}
        self.health = {}
    
    def fetch(self, pairs):
        sg_tz = pytz.timezone('Asia/Singapore')
        now = datetime.now(sg_tz)
        table = f"oracle_price_log_partition_{now.strftime('%Y%m%d')}"
        
        start = (now - timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')
        end = now.strftime('%Y-%m-%d %H:%M:%S')
        
        results = {}
        for pair in pairs:
            results[pair] = {}
            
            # Rollbit query
            rq = f"""
                SELECT created_at + INTERVAL '8 hour' AS timestamp, final_price AS price
                FROM {table}
                WHERE source_type = 1 AND pair_name = '{map_token(pair)}'
                AND created_at >= '{start}'::timestamp - INTERVAL '8 hour'
                AND created_at <= '{end}'::timestamp - INTERVAL '8 hour'
                ORDER BY timestamp DESC
            """
            
            # Surf query
            sq = f"""
                SELECT created_at + INTERVAL '8 hour' AS timestamp, final_price AS price
                FROM {table}
                WHERE source_type = 0 AND pair_name = '{pair}'
                AND created_at >= '{start}'::timestamp - INTERVAL '8 hour'
                AND created_at <= '{end}'::timestamp - INTERVAL '8 hour'
                ORDER BY timestamp DESC
            """
            
            try:
                with engine.connect() as conn:
                    rdf = pd.read_sql_query(text(rq), conn)
                    sdf = pd.read_sql_query(text(sq), conn)
                    
                    if not rdf.empty:
                        results[pair]['rollbit'] = rdf
                        st.success(f"✓ Rollbit {pair}: {len(rdf)} points")
                    if not sdf.empty:
                        results[pair]['surf'] = sdf
                        st.success(f"✓ Surf {pair}: {len(sdf)} points")
            except Exception as e:
                st.error(f"Error {pair}: {e}")
        
        return results
    
    def calc_metrics(self, prices, points):
        if len(prices) < points:
            return None
        
        sample = prices.iloc[-points:]
        
        # Choppiness
        w = min(20, points // 10)
        hr = sample.rolling(w).max()
        lr = sample.rolling(w).min()
        tr = hr - lr
        ac = sample.diff().abs().rolling(w).sum()
        chop = 100 * (ac / tr.replace(0, 1e-10))
        chop = chop.replace([np.inf, -np.inf], np.nan).mean()
        
        # ATR%
        atr = sample.diff().abs().mean()
        atr_pct = (atr / sample.mean()) * 100
        
        # Trend
        nc = abs(sample.iloc[-1] - sample.iloc[-w])
        sc = sample.diff().abs().rolling(w).sum().iloc[-1]
        trend = nc / sc if sc > 0 else 0
        
        return {'choppiness': chop, 'tick_atr_pct': atr_pct, 'trend_strength': trend}
    
    def calc_health(self, df):
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['price'] = pd.to_numeric(df['price'])
        df = df.set_index('timestamp').sort_index()
        
        cutoff = df.index.max() - pd.Timedelta(minutes=10)
        recent = df[df.index >= cutoff]
        
        if len(recent) < 10:
            return None
        
        ohlc = recent['price'].resample('1T').ohlc().dropna()
        
        if len(ohlc) == 0:
            return None
        
        metrics = {}
        doji = 0
        wps = []
        ws = []
        hwc = 0
        wbr = []
        
        for _, r in ohlc.iterrows():
            body = abs(r['close'] - r['open'])
            tr = r['high'] - r['low']
            
            if tr > 0:
                uw = r['high'] - max(r['open'], r['close'])
                lw = min(r['open'], r['close']) - r['low']
                tw = uw + lw
                
                bp = (body / tr) * 100
                wp = (tw / tr) * 100
                
                if bp < 30:
                    doji += 1
                if wp > 50:
                    hwc += 1
                
                wps.append(wp)
                ws.append(tw)
                
                if body > 0:
                    wbr.append(tw / body)
                else:
                    wbr.append(10.0)
        
        n = len(ohlc)
        metrics['doji_pct'] = (doji / n) * 100 if n > 0 else 0
        metrics['avg_wick_pct'] = np.mean(wps) if wps else 0
        metrics['avg_wick_size'] = np.mean(ws) if ws else 0
        metrics['high_wick_count'] = hwc
        metrics['max_wick'] = max(ws) if ws else 0
        metrics['wick_body_ratio'] = np.mean(wbr) if wbr else 0
        
        return metrics
    
    def analyze(self, data):
        for pair, exs in data.items():
            key = pair.replace('/', '_')
            
            if key not in self.data:
                self.data[key] = {}
            
            for ex, df in exs.items():
                prices = pd.Series(df['price'].values, dtype=float)
                
                if ex not in self.data[key]:
                    self.data[key][ex] = {}
                
                for pc in self.point_counts:
                    m = self.calc_metrics(prices, pc)
                    if m:
                        self.data[key][ex][pc] = m
                
                h = self.calc_health(df)
                if h:
                    if key not in self.health:
                        self.health[key] = {}
                    self.health[key][ex] = h
    
    def get_comparison_df(self, pc):
        rows = []
        for key, exs in self.data.items():
            coin = key.replace('_', '/')
            row = {'Coin': coin}
            
            for m in ['choppiness', 'tick_atr_pct', 'trend_strength']:
                ms = m.replace('tick_atr_pct', 'ATR%').replace('choppiness', 'Chop').replace('trend_strength', 'Trend')
                
                rv = None
                sv = None
                
                if 'rollbit' in exs and pc in exs['rollbit']:
                    rv = exs['rollbit'][pc].get(m)
                if 'surf' in exs and pc in exs['surf']:
                    sv = exs['surf'][pc].get(m)
                
                row[f'{ms} ROLLBIT'] = rv
                row[f'{ms} SURF'] = sv
                
                if rv is not None and sv is not None:
                    row[f'{ms} Diff'] = sv - rv
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def get_health_df(self):
        rows = []
        
        for key in sorted(self.health.keys()):
            coin = key.replace('_', '/')
            
            # Rollbit row
            r = {
                'Coin': coin,
                'Exchange': 'Rollbit',
                'Doji %': None,
                'Avg Wick %': None,
                'Avg Wick Size': None,
                'High Wick Count': None,
                'Max Wick': None,
                'Wick/Body Ratio': None
            }
            
            if 'rollbit' in self.health[key]:
                d = self.health[key]['rollbit']
                r['Doji %'] = round(d['doji_pct'], 2)
                r['Avg Wick %'] = round(d['avg_wick_pct'], 2)
                r['Avg Wick Size'] = round(d['avg_wick_size'], 6)
                r['High Wick Count'] = int(d['high_wick_count'])
                r['Max Wick'] = round(d['max_wick'], 6)
                r['Wick/Body Ratio'] = round(d['wick_body_ratio'], 2)
            
            # Surf row
            s = {
                'Coin': coin,
                'Exchange': 'Surf',
                'Doji %': None,
                'Avg Wick %': None,
                'Avg Wick Size': None,
                'High Wick Count': None,
                'Max Wick': None,
                'Wick/Body Ratio': None
            }
            
            if 'surf' in self.health[key]:
                d = self.health[key]['surf']
                s['Doji %'] = round(d['doji_pct'], 2)
                s['Avg Wick %'] = round(d['avg_wick_pct'], 2)
                s['Avg Wick Size'] = round(d['avg_wick_size'], 6)
                s['High Wick Count'] = int(d['high_wick_count'])
                s['Max Wick'] = round(d['max_wick'], 6)
                s['Wick/Body Ratio'] = round(d['wick_body_ratio'], 2)
            
            rows.append(r)
            rows.append(s)
        
        return pd.DataFrame(rows) if rows else None

# UI
st.title("Crypto Exchange Analysis Dashboard")
st.write(f"Time: {datetime.now(pytz.timezone('Asia/Singapore')).strftime('%Y-%m-%d %H:%M:%S')} SGT")

# ONLY 2 TABS
tab1, tab2 = st.tabs(["Parameter Comparison", "Coin Health (1-min Wicks)"])

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    @st.cache_data
    def get_pairs():
        try:
            q = "SELECT DISTINCT pair_name FROM trade_pool_pairs WHERE status IN (1,2) ORDER BY pair_name"
            with engine.connect() as c:
                r = c.execute(text(q))
                return [row[0] for row in r]
        except:
            return ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    
    all_pairs = get_pairs()
    
    # Buttons
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Major"):
            st.session_state.p = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT"]
    with c2:
        if st.button("All"):
            st.session_state.p = all_pairs
    with c3:
        if st.button("Clear"):
            st.session_state.p = []
    
    if 'p' not in st.session_state:
        st.session_state.p = ["BTC/USDT", "ETH/USDT"]
    
    selected = st.multiselect("Select Pairs", all_pairs, default=st.session_state.p)
    
    st.info("Uses last 1 hour of data")
    
    analyze = st.button("🔍 Analyze", type="primary")

# Analysis
if analyze and selected:
    a = Analyzer()
    
    with st.spinner("Analyzing..."):
        data = a.fetch(selected)
        a.analyze(data)
        
        # Tab 1
        with tab1:
            st.header("Parameter Comparison")
            
            for pc in a.point_counts:
                with st.expander(f"{pc} Points", expanded=(pc==500)):
                    df = a.get_comparison_df(pc)
                    if not df.empty:
                        st.dataframe(df, use_container_width=True)
        
        # Tab 2
        with tab2:
            st.header("Coin Health Analysis")
            st.write("1-minute candles over last 10 minutes")
            
            df = a.get_health_df()
            
            if df is not None:
                def style(row):
                    if row['Exchange'] == 'Rollbit':
                        return ['background-color: #e3f2fd'] * len(row)
                    return ['background-color: #fff3e0'] * len(row)
                
                st.dataframe(
                    df.style.apply(style, axis=1),
                    use_container_width=True,
                    height=800,
                    column_config={
                        "High Wick Count": st.column_config.NumberColumn(format="%d")
                    }
                )
                
                csv = df.to_csv(index=False)
                st.download_button("Download", csv, "health.csv")

elif analyze:
    st.warning("Select at least one pair")