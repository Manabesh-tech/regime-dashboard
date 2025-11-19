import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine

st.set_page_config(page_title="SurfV2 5-Min Volatility (First 48h)", page_icon="📊", layout="wide")

st.title("SurfV2 Pairs - 5-Min Volatility in First 48 Hours")

# Database connection parameters (same as other dashboards)
db_params = {
    "host": "aws-jp-tk-surf-pg-public.cluster-csteuf9lw8dv.ap-northeast-1.rds.amazonaws.com",
    "port": 5432,
    "database": "replication_report",
    "user": "public_replication",
    "password": "866^FKC4hllk",
}

engine = create_engine(
    f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['database']}",
    isolation_level="AUTOCOMMIT",  # Enable auto-commit to avoid long-running transactions
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800,
    pool_pre_ping=True,
    pool_use_lifo=True,
    echo=False,
)


@st.cache_data(ttl=600)
def fetch_surfv2_pairs(days_back: int = 7) -> pd.DataFrame:
    """Fetch SurfV2 trade pairs created within the last N days."""
    query = f"""
    SELECT
      id,
      pair_id,
      pair_name,
      status,
      created_at
    FROM public.trade_pool_pairs
    WHERE status = 1
      AND created_at IS NOT NULL
      AND created_at >= NOW() - INTERVAL '{days_back} days'
    ORDER BY created_at
    """
    return pd.read_sql_query(query, engine)


def get_partition_tables_for_range(start_sg: datetime, end_sg: datetime) -> list[str]:
    """Return existing oracle_price_log_partition_YYYYMMDD tables for the given SG time range."""
    start_date = start_sg.date()
    end_date = end_sg.date()

    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)

    if not dates:
        return []

    table_names = [f"oracle_price_log_partition_{d}" for d in dates]

    if not table_names:
        return []

    table_list_str = "', '".join(table_names)
    query = f"""
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'public'
      AND table_name IN ('{table_list_str}')
    """

    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn)

    existing = df["table_name"].tolist() if not df.empty else []
    return existing


def compute_5min_vol_for_pair(pair_name: str, created_at_utc: datetime) -> tuple[float | None, float | None]:
    """Compute max and average 5-min annualized volatility in the first 48h after pair creation.

    Returns (max_vol, avg_vol) as decimals (e.g. 1.0 = 100%).
    """
    if created_at_utc is None:
        return None, None

    if isinstance(created_at_utc, pd.Timestamp):
        created_at_utc = created_at_utc.to_pydatetime()

    # Work in Singapore time (UTC+8) like other dashboards
    start_sg = created_at_utc + timedelta(hours=8)
    end_sg = start_sg + timedelta(hours=48)

    partition_tables = get_partition_tables_for_range(start_sg, end_sg)
    if not partition_tables:
        return None, None

    start_str = start_sg.strftime("%Y-%m-%d %H:%M:%S")
    end_str = end_sg.strftime("%Y-%m-%d %H:%M:%S")

    union_parts: list[str] = []
    for table in partition_tables:
        q = f"""
        SELECT
          created_at + INTERVAL '8 hour' AS timestamp,
          final_price
        FROM public.{table}
        WHERE created_at >= '{start_str}'::timestamp - INTERVAL '8 hour'
          AND created_at <= '{end_str}'::timestamp - INTERVAL '8 hour'
          AND source_type = 0
          AND pair_name = '{pair_name}'
        """
        union_parts.append(q)

    if not union_parts:
        return None, None

    full_query = " UNION ALL ".join(union_parts) + " ORDER BY timestamp"

    with engine.connect() as conn:
        df = pd.read_sql_query(full_query, conn)

    if df.empty:
        return None, None

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()

    price_series = df["final_price"].dropna()
    if price_series.empty:
        return None, None

    # Build 5-minute windows in Singapore time
    start_window = price_series.index.min().floor("5min")
    end_window = price_series.index.max().ceil("5min")
    periods = pd.date_range(start=start_window, end=end_window, freq="5min")

    vols: list[float] = []
    for i in range(len(periods) - 1):
        w_start = periods[i]
        w_end = periods[i + 1]
        window_data = price_series[(price_series.index >= w_start) & (price_series.index < w_end)]
        if len(window_data) < 2:
            continue

        log_returns = np.diff(np.log(window_data.values))
        if len(log_returns) == 0:
            continue

        # Annualization factor: seconds in a year / seconds in 5 minutes
        annualization_factor = np.sqrt(31536000.0 / 300.0)
        vol = float(np.std(log_returns) * annualization_factor)
        vols.append(vol)

    if not vols:
        return None, None

    vols_arr = np.array(vols)
    max_vol = float(vols_arr.max())
    avg_vol = float(vols_arr.mean())
    return max_vol, avg_vol


st.sidebar.header("Settings")
days_back = st.sidebar.number_input(
    "Filter pairs created within the last N days",
    min_value=1,
    max_value=365,
    value=7,
    step=1,
)

st.write("This dashboard computes the 5-minute annualized volatility for each SurfV2 pair in the first 48 hours after its creation.")

if st.button("Calculate 5-Min Volatility for All Loaded Pairs", type="primary"):
    pairs_df = fetch_surfv2_pairs(days_back=int(days_back))

    if pairs_df.empty:
        st.error("No pairs found in trade_pool_pairs table for the selected creation date window.")
        st.stop()

    st.write(
        f"Loaded {len(pairs_df)} pairs from trade_pool_pairs created within the last {int(days_back)} days."
    )

    results = []
    progress = st.progress(0.0)

    for idx, row in pairs_df.iterrows():
        pair_name = row["pair_name"]
        created_at = row["created_at"]

        max_vol, avg_vol = compute_5min_vol_for_pair(pair_name, created_at)

        results.append(
            {
                "Token": pair_name,
                "Created At": created_at,
                "Max 5min Vol (%)": max_vol * 100 if max_vol is not None else None,
                "Avg 5min Vol (%)": avg_vol * 100 if avg_vol is not None else None,
            }
        )

        progress.progress((idx + 1) / len(pairs_df))

    result_df = pd.DataFrame(results)

    # Sort by max volatility descending for easier inspection
    if not result_df.empty:
        result_df = result_df.sort_values(by="Max 5min Vol (%)", ascending=False)

    st.subheader("5-Min Volatility in First 48 Hours After Pair Creation")
    st.dataframe(result_df, use_container_width=True)

    st.caption(
        "Volatility numbers are annualized and expressed in percentage terms. "
        "They are computed using 5-minute windows on Surf oracle prices (source_type = 0)."
    )
else:
    st.info("Set the filters in the left sidebar, then click the button above to calculate volatility.")
