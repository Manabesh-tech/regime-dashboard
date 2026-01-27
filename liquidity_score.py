import json
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import psycopg2
import requests
import streamlit as st

# ---------- Paths & Config ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
CRAWLER_DIR = os.path.join(PROJECT_ROOT, "crawler")
RESULT_DIR = os.path.join(CRAWLER_DIR, "result")
HISTORY_PATH = os.path.join(RESULT_DIR, "history.json")
SCHEDULER_STATE_PATH = os.path.join(PROJECT_ROOT, "scheduler_state.json")

DB_CONFIG = {
    "host": "aws-jp-tk-surf-pg-public.cluster-csteuf9lw8dv.ap-northeast-1.rds.amazonaws.com",
    "port": 5432,
    "dbname": "replication_report",
    "user": "public_replication",
    "password": "866^FKC4hllk",
}

SQL_LIQUIDITY = """
WITH latest_data AS (
    SELECT *
    FROM oracle_exchange_spread
    WHERE source IN (
        'binanceFuture', 'hyperliquidFuture', 'gateFuture',
        'bybitFuture', 'mexcFuture', 'bitgetFuture', 'okxFuture'
    )
      AND time_group = (SELECT MAX(time_group) FROM oracle_exchange_spread)
      AND amount::numeric IN (1000, 5000, 10000, 20000, 50000, 200000)
),
ranked AS (
    SELECT *,
           ROW_NUMBER() OVER (
               PARTITION BY pair_name, amount
               ORDER BY 
                   CASE source
                       WHEN 'binanceFuture' THEN 1
                       WHEN 'hyperliquidFuture' THEN 2
                       WHEN 'gateFuture' THEN 3
                       WHEN 'bybitFuture' THEN 4
                       WHEN 'mexcFuture' THEN 5
                       WHEN 'bitgetFuture' THEN 6
                       WHEN 'okxFuture' THEN 7
                       ELSE 100
                   END
           ) AS rn
    FROM latest_data
)
SELECT 
    pair_name,
    MAX(CASE WHEN amount::numeric = 1000 THEN fee::numeric * 10000 END) AS "1k"
FROM ranked
WHERE rn = 1
GROUP BY pair_name
ORDER BY "1k" DESC;
"""

UTC8 = timezone(timedelta(hours=8))


# ---------- Remote GitHub helpers (for Cloud refresh) ----------

GITHUB_OWNER = os.environ.get("GITHUB_OWNER", "czx51")
GITHUB_REPO = os.environ.get("GITHUB_REPO", "regime-dashboard")
GITHUB_RAW_BASE = os.environ.get(
    "GITHUB_RAW_BASE",
    f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}/main",
)
GITHUB_REF = os.environ.get("GITHUB_REF", "main")


def _http_get_text(url: str, timeout: float = 10.0) -> str | None:
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200 and r.text:
            return r.text
    except Exception:
        pass
    return None


def _try_load_remote_crawler_results() -> tuple[Dict | None, Dict | None]:
    base = f"{GITHUB_RAW_BASE}/crawler/result"
    ts = int(time.time())
    oi_txt = _http_get_text(f"{base}/openinterest.json?t={ts}")
    tv_txt = _http_get_text(f"{base}/tradingvolume.json?t={ts}")

    oi_data: Dict | None = None
    tv_data: Dict | None = None
    if oi_txt:
        try:
            oi_data = json.loads(oi_txt)
        except Exception:
            oi_data = None
    if tv_txt:
        try:
            tv_data = json.loads(tv_txt)
        except Exception:
            tv_data = None
    return oi_data, tv_data


def _try_load_remote_history() -> Dict | None:
    base = f"{GITHUB_RAW_BASE}/crawler/result"
    ts = int(time.time())
    txt = _http_get_text(f"{base}/history.json?t={ts}")
    if not txt:
        return None
    try:
        data = json.loads(txt)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _try_load_remote_scheduler_state() -> Dict | None:
    base = f"{GITHUB_RAW_BASE}/"
    ts = int(time.time())
    txt = _http_get_text(f"{base}/scheduler_state.json?t={ts}")
    if not txt:
        return None
    try:
        data = json.loads(txt)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _dispatch_and_wait_scheduler(wait_seconds: int = 600, poll_interval: float = 10.0) -> bool:
    """Trigger liquidity_monitor_scheduler workflow and wait until latest run succeeds.

    This mirrors the pattern used in currency_leverage_collection: we use a GitHub PAT
    (from st.secrets["GITHUB_PAT"] or env GITHUB_PAT) to dispatch the workflow and
    then poll its runs until a successful completion or timeout.
    """

    try:
        # Try to read token from Streamlit secrets first (supports nested [github] table),
        # then fall back to environment variable GITHUB_PAT.
        token = None

        # Preferred: [github].token in secrets.toml
        try:
            github_section = st.secrets["github"]  # type: ignore[attr-defined]
            try:
                # support both attribute-like and dict-like access
                token = getattr(github_section, "token", None) or github_section["token"]
            except Exception:
                # fallback: try common alternate key
                token = github_section.get("token") or github_section.get("GITHUB_PAT")  # type: ignore[call-arg]
        except Exception:
            token = None

        # Fallback: top-level GITHUB_PAT in secrets
        if not token:
            try:
                token = st.secrets["GITHUB_PAT"]  # type: ignore[attr-defined]
            except Exception:
                token = None

        # Last resort: environment variable
        if not token:
            token = os.environ.get("GITHUB_PAT","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")

        if not token:
            st.error(
                "GITHUB_PAT (or [github].token) is not configured. Data Refresh can only "
                "trigger the GitHub Actions scheduler when this token is set (e.g. in "
                ".streamlit/secrets.toml or Streamlit Cloud app secrets)."
            )
            return False

        headers = {"Accept": "application/vnd.github+json", "Authorization": f"token {token}"}

        # Dispatch liquidity_monitor_scheduler.yml
        dispatch_url = (
            f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/actions/"
            f"workflows/liquidity_monitor.yml/dispatches"
        )
        resp = requests.post(
            dispatch_url,
            headers=headers,
            json={"ref": GITHUB_REF},
            timeout=15,
        )
        if resp.status_code not in (201, 202, 204):
            st.error(f"Dispatch failed: {resp.status_code} {resp.text}")
            return False

        # Poll the latest run status
        runs_api = (
            f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/actions/"
            f"workflows/liquidity_monitor.yml/runs?per_page=1"
        )
        t0 = time.time()
        while time.time() - t0 < wait_seconds:
            time.sleep(poll_interval)
            try:
                r = requests.get(runs_api, headers=headers, timeout=15)
                if r.status_code == 200:
                    js = r.json() or {}
                    runs = js.get("workflow_runs") or []
                    if runs:
                        latest = runs[0]
                        status = (latest.get("status") or "").lower()
                        conclusion = (latest.get("conclusion") or "").lower()
                        if status == "completed" and conclusion == "success":
                            return True
            except Exception:
                pass
        return False
    except Exception as e:
        st.error(f"Dispatch error: {e}")
        return False


# ---------- Data helpers ----------

def pair_to_symbol(pair: str) -> str:
    """Map pair name (e.g. BTCUSDT, BTC-USDT, BTC/USDT) to base symbol.

    We try to be robust to separators like '-', '_' and '/'.
    """
    # First cut off anything after '/', e.g. "BTC/USDT" -> "BTC"
    s = pair.upper().split("/")[0]
    # Remove common separators
    s = s.replace("-", "").replace("_", "")
    for suffix in ["USDT", "USD", "PERP", "USDC"]:
        if s.endswith(suffix):
            return s[: -len(suffix)]
    return s


def load_liquidity_from_db() -> pd.DataFrame:
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        df = pd.read_sql(SQL_LIQUIDITY, conn)
    finally:
        conn.close()

    df["pair_name"] = df["pair_name"].astype(str)
    df["1k"] = pd.to_numeric(df["1k"], errors="coerce")
    df["symbol"] = df["pair_name"].apply(pair_to_symbol)
    return df


def run_crawlers_for_symbols(symbols: List[str]) -> None:
    if not symbols:
        return

    symbols_arg = ",".join(sorted(set(s.upper() for s in symbols)))
    main_crawler = os.path.join(CRAWLER_DIR, "main_crawler.py")
    import subprocess

    oi_workers = 2
    oi_progress_prefix = os.path.join(RESULT_DIR, "oi_progress")
    oi_progress_files = [f"{oi_progress_prefix}_{i}.json" for i in range(1, oi_workers + 1)]
    tv_progress_file = os.path.join(RESULT_DIR, "tv_progress.json")

    progress_container = st.empty()
    with progress_container:
        st.write("Open interest crawler")
        oi_bar = st.progress(0)
        st.write("Trading volume crawler")
        tv_bar = st.progress(0)

    python_exe = sys.executable or "python"

    cmd_oi = [
        python_exe,
        main_crawler,
        "--symbols",
        symbols_arg,
        "--oi-workers",
        str(oi_workers),
        "--mode",
        "oi",
        "--oi-progress-prefix",
        oi_progress_prefix,
    ]
    cmd_tv = [
        python_exe,
        main_crawler,
        "--mode",
        "tv",
        "--tv-progress-file",
        tv_progress_file,
    ]

    proc_oi = subprocess.Popen(cmd_oi, cwd=CRAWLER_DIR)
    proc_tv = subprocess.Popen(cmd_tv, cwd=CRAWLER_DIR)

    def _read_progress(path: str) -> tuple[int, int]:
        if not os.path.exists(path):
            return 0, 0
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            done = int(data.get("done", 0) or 0)
            total = int(data.get("total", 0) or 0)
            return max(done, 0), max(total, 0)
        except Exception:
            return 0, 0

    oi_done = False
    tv_done = False
    oi_progress = 0
    tv_progress = 0

    while not (oi_done and tv_done):
        time.sleep(0.5)

        if not oi_done:
            if proc_oi.poll() is not None:
                oi_done = True
                oi_bar.progress(100)
            else:
                total_done = 0
                total_total = 0
                for p in oi_progress_files:
                    d, t = _read_progress(p)
                    total_done += d
                    total_total += t
                if total_total > 0 and total_done >= 0:
                    pct = int(min(100, max(1, total_done / total_total * 100)))
                    oi_progress = pct
                oi_bar.progress(oi_progress)

        if not tv_done:
            if proc_tv.poll() is not None:
                tv_done = True
                tv_bar.progress(100)
            else:
                d, t = _read_progress(tv_progress_file)
                if t > 0 and d >= 0:
                    pct = int(min(100, max(1, d / t * 100)))
                    tv_progress = pct
                tv_bar.progress(tv_progress)

    ret_oi = proc_oi.returncode
    ret_tv = proc_tv.returncode

    if ret_oi != 0:
        st.error(f"Open interest crawler failed with exit code {ret_oi}")
    if ret_tv != 0:
        st.error(f"Trading volume crawler failed with exit code {ret_tv}")

    # 所有任务结束后，自动移除进度条
    progress_container.empty()


def load_crawler_results() -> Tuple[Dict, Dict]:
    """Load crawler results from database."""
    oi_data: Dict = {}
    tv_data: Dict = {}

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        try:
            # 加载持仓数据
            oi_df = pd.read_sql("""
                SELECT symbol, oi_change_1h, oi_change_4h, oi_change_24h, 
                       oi_volume_ratio, exchange_data
                FROM oracle_open_interest
            """, conn)
            for _, row in oi_df.iterrows():
                symbol = str(row["symbol"]).upper()
                # 使用 row[] 访问，如果字段不存在会抛出异常，这样可以及时发现问题
                oi_change_1h = row["oi_change_1h"] if pd.notna(row.get("oi_change_1h")) else None
                oi_change_4h = row["oi_change_4h"] if pd.notna(row.get("oi_change_4h")) else None
                oi_change_24h = row["oi_change_24h"] if pd.notna(row.get("oi_change_24h")) else None
                oi_volume_ratio = row["oi_volume_ratio"] if pd.notna(row.get("oi_volume_ratio")) else None
                exchange_data_str = row["exchange_data"] if pd.notna(row.get("exchange_data")) else None

                oi_data[symbol] = {
                    "持仓变化（1小时）": oi_change_1h,
                    "持仓变化（4小时）": oi_change_4h,
                    "持仓变化（24小时）": oi_change_24h,
                    "持仓/24小时成交额": oi_volume_ratio,
                    "交易所数据": json.loads(exchange_data_str) if exchange_data_str else [],
                }

            # 加载交易量数据
            tv_df = pd.read_sql("""
                SELECT symbol, volume_24h_percent, volume_24h_raw, volume_24h,
                       market_cap_raw, market_cap
                FROM oracle_trading_volume
            """, conn)
            for _, row in tv_df.iterrows():
                symbol = str(row["symbol"]).upper()
                tv_data[symbol] = {
                    "成交额24h%": row.get("volume_24h_percent"),
                    "成交额24h_raw": row.get("volume_24h_raw"),
                    "成交额24h": row.get("volume_24h"),
                    "总市值_raw": row.get("market_cap_raw"),
                    "总市值": row.get("market_cap"),
                }
        finally:
            conn.close()
    except Exception as e:  # noqa: BLE001
        print(f"[app] Failed to load from database: {e!r}")

    return oi_data, tv_data


def _parse_percent(text: str | None) -> float | None:
    """解析百分比字符串，如 "+0.21%" 或 "-1.33%"，返回浮点数。"""
    if text is None:
        return None
    t = str(text).strip().replace("%", "")
    if t == "":
        return None
    try:
        return float(t)
    except ValueError:
        return None


def parse_openinterest_24h(oi_data: Dict) -> pd.DataFrame:
    rows = []
    for symbol, v in oi_data.items():
        if not isinstance(v, dict):
            continue

        # 获取原始值
        raw_1h = v.get("持仓变化（1小时）")
        raw_4h = v.get("持仓变化（4小时）")
        raw_24h = v.get("持仓变化（24小时）")
        raw_ratio = v.get("持仓/24小时成交额")

        # 解析百分比值：如果值是 None、空字符串或只包含空白字符，则返回 None
        val_1h = _parse_percent(raw_1h) if raw_1h and str(raw_1h).strip() else None
        val_4h = _parse_percent(raw_4h) if raw_4h and str(raw_4h).strip() else None
        val_24h = _parse_percent(raw_24h) if raw_24h and str(raw_24h).strip() else None

        ratio = None
        if raw_ratio is not None:
            t = str(raw_ratio).strip().replace(",", "")
            if t != "":
                try:
                    ratio = float(t)
                except ValueError:
                    ratio = None

        rows.append(
            {
                "symbol": symbol.upper(),
                "oi_1h_change": val_1h,
                "oi_4h_change": val_4h,
                "oi_24h_change": val_24h,
                "oi_vol_ratio_24h": ratio,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "symbol",
                "oi_1h_change",
                "oi_4h_change",
                "oi_24h_change",
                "oi_vol_ratio_24h",
            ]
        )
    return pd.DataFrame(rows)


def parse_tradingvolume_24h(tv_data: Dict) -> pd.DataFrame:
    rows = []
    for symbol, v in tv_data.items():
        if not isinstance(v, dict):
            continue
        change_24h = v.get("成交额24h%") or ""
        val = _parse_percent(change_24h)
        volume_abs = v.get("成交额24h")
        mcap = v.get("总市值")
        rows.append(
            {
                "symbol": symbol.upper(),
                "vol_24h_change": val,
                "volume_24h": volume_abs,
                "market_cap": mcap,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["symbol", "vol_24h_change", "volume_24h", "market_cap"])
    return pd.DataFrame(rows)


def zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mean = s.mean()
    std = s.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=series.index)
    return (s - mean) / std


def mad_winsorize(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Winsorize a series using median +/- threshold * MAD.

    Values whose absolute deviation from the median is greater than
    ``threshold * MAD`` are clipped to the corresponding bounds.
    NaNs are preserved.
    """

    s = pd.to_numeric(series, errors="coerce")
    median = s.median()
    mad = (s - median).abs().median()
    if mad == 0 or np.isnan(mad):
        # No variability or cannot compute MAD -> return as-is
        return s

    lower = median - threshold * mad
    upper = median + threshold * mad
    return s.clip(lower, upper)


def write_last_start(ts: datetime) -> None:
    data = {"last_start_utc": ts.astimezone(timezone.utc).isoformat()}
    try:
        with open(SCHEDULER_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        # Best-effort; UI should not crash because of scheduler state
        pass


def compute_scores(df_base: pd.DataFrame, df_oi: pd.DataFrame, df_tv: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df_base.copy()
    # 确保所有 DataFrame 的 symbol 列都是字符串类型且大写，以便正确合并
    df["symbol"] = df["symbol"].astype(str).str.upper()
    if not df_oi.empty:
        df_oi["symbol"] = df_oi["symbol"].astype(str).str.upper()
    if not df_tv.empty:
        df_tv["symbol"] = df_tv["symbol"].astype(str).str.upper()

    df = df.merge(df_oi, on="symbol", how="left")
    df = df.merge(df_tv, on="symbol", how="left")

    # 对每个指标做 z-score，缺失值（None/NaN）视为 0
    # 对 oi_24h_change 和 vol_24h_change 在 z-score 前先进行基于 MAD 的去极值化
    df["volume_24h"] = pd.to_numeric(df.get("volume_24h"), errors="coerce")
    df["market_cap"] = pd.to_numeric(df.get("market_cap"), errors="coerce")

    df["z_1k"] = zscore(df["1k"]).fillna(0.0)

    oi_24h_winsor = mad_winsorize(df["oi_24h_change"], threshold=3.0)
    vol_24h_winsor = mad_winsorize(df["vol_24h_change"], threshold=3.0)

    df["z_oi_24h"] = zscore(oi_24h_winsor).fillna(0.0)
    df["z_vol_24h"] = zscore(vol_24h_winsor).fillna(0.0)
    df["z_volume_24h"] = zscore(df["volume_24h"]).fillna(0.0)
    df["z_market_cap"] = zscore(df["market_cap"]).fillna(0.0)

    df["score"] = (
            -df["z_1k"] * 0.25
            + df["z_volume_24h"] * 0.25
            + df["z_market_cap"] * 0.25
            + df["z_oi_24h"] * 0.125
            + df["z_vol_24h"] * 0.125
    )

    df_scores = df[["symbol", "pair_name", "score"]].copy()
    df_scores = df_scores.sort_values("score", ascending=True).reset_index(drop=True)

    df_raw = df[[
        "symbol",
        "pair_name",
        "1k",
        "oi_1h_change",
        "oi_4h_change",
        "oi_24h_change",
        "oi_vol_ratio_24h",
        "vol_24h_change",
        "volume_24h",
        "market_cap",
        "z_1k",
        "z_oi_24h",
        "z_vol_24h",
        "z_volume_24h",
        "z_market_cap",
        "score",
    ]].copy()

    return df_scores, df_raw


def _show_overview_metrics(df_scores_view: pd.DataFrame, df_raw_view: pd.DataFrame) -> None:
    total_symbols = int(len(df_scores_view))
    full_data_mask = (~df_raw_view["oi_24h_change"].isna()) & (~df_raw_view["vol_24h_change"].isna())
    full_symbols = int(full_data_mask.sum())

    median_score = float(df_scores_view["score"].median()) if not df_scores_view.empty else 0.0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Symbols", total_symbols)
    with col2:
        st.metric("Symbols with full data", full_symbols)
    with col3:
        st.metric("Median score", f"{median_score:.3f}")


def _style_raw_table(
        df_raw_view: pd.DataFrame,
        df_raw_full: pd.DataFrame | None = None,
) -> "pd.io.formats.style.Styler":  # type: ignore[name-defined]
    if df_raw_view.empty:
        return df_raw_view.style

    base = df_raw_full if df_raw_full is not None else df_raw_view

    styled = df_raw_view.style.format(
        {
            "1k": "{:.1f}",
            "oi_1h_change": "{:+.2f}%",
            "oi_4h_change": "{:+.2f}%",
            "oi_24h_change": "{:+.2f}%",
            "oi_vol_ratio_24h": "{:.4f}",
            "vol_24h_change": "{:+.2f}%",
            "volume_24h": "{:.0f}",
            "market_cap": "{:.0f}",
            "z_1k": "{:+.2f}",
            "z_oi_24h": "{:+.2f}",
            "z_vol_24h": "{:+.2f}",
            "z_volume_24h": "{:+.2f}",
            "z_market_cap": "{:+.2f}",
            "score": "{:+.3f}",
        }
    )

    # 使用全量数据的最小/最大值来固定颜色区间，保证搜索前后颜色语义一致
    vmin_z1 = float(base["z_1k"].min()) if "z_1k" in base.columns else None
    vmax_z1 = float(base["z_1k"].max()) if "z_1k" in base.columns else None
    vmin_score = float(base["score"].min()) if "score" in base.columns else None
    vmax_score = float(base["score"].max()) if "score" in base.columns else None
    vmin_z_oi = float(base["z_oi_24h"].min()) if "z_oi_24h" in base.columns else None
    vmax_z_oi = float(base["z_oi_24h"].max()) if "z_oi_24h" in base.columns else None
    vmin_z_vol = float(base["z_vol_24h"].min()) if "z_vol_24h" in base.columns else None
    vmax_z_vol = float(base["z_vol_24h"].max()) if "z_vol_24h" in base.columns else None
    vmin_z_vol_abs = float(base["z_volume_24h"].min()) if "z_volume_24h" in base.columns else None
    vmax_z_vol_abs = float(base["z_volume_24h"].max()) if "z_volume_24h" in base.columns else None
    vmin_z_mcap = float(base["z_market_cap"].min()) if "z_market_cap" in base.columns else None
    vmax_z_mcap = float(base["z_market_cap"].max()) if "z_market_cap" in base.columns else None

    styled = styled.background_gradient(
        subset=["z_1k"],
        cmap="RdYlGn_r",
        vmin=vmin_z1,
        vmax=vmax_z1,
    )
    styled = styled.background_gradient(
        subset=["score"],
        cmap="RdYlGn",
        vmin=vmin_score,
        vmax=vmax_score,
    )
    styled = styled.background_gradient(
        subset=["z_oi_24h"],
        cmap="RdYlGn",
        vmin=vmin_z_oi,
        vmax=vmax_z_oi,
    )
    styled = styled.background_gradient(
        subset=["z_vol_24h"],
        cmap="RdYlGn",
        vmin=vmin_z_vol,
        vmax=vmax_z_vol,
    )
    if "z_volume_24h" in df_raw_view.columns:
        styled = styled.background_gradient(
            subset=["z_volume_24h"],
            cmap="RdYlGn",
            vmin=vmin_z_vol_abs,
            vmax=vmax_z_vol_abs,
        )
    if "z_market_cap" in df_raw_view.columns:
        styled = styled.background_gradient(
            subset=["z_market_cap"],
            cmap="RdYlGn",
            vmin=vmin_z_mcap,
            vmax=vmax_z_mcap,
        )
    return styled


def _load_history() -> Dict[str, list]:
    """Load history data from database."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        try:
            query = """
            SELECT 
                symbol,
                ts,
                spread_1k,
                score,
                oi_24h_change,
                vol_24h_change,
                volume_24h,
                market_cap,
                oi_exchanges
            FROM oracle_liquidity_history
            WHERE ts >= NOW() - INTERVAL '7 days'
            ORDER BY symbol, ts ASC
            """
            df = pd.read_sql(query, conn)

            if not df.empty:
                # 转换为原来的格式：{symbol: [records...]}
                history: Dict[str, list] = {}
                for symbol in df["symbol"].unique():
                    symbol_df = df[df["symbol"] == symbol].copy()
                    records = []
                    for _, row in symbol_df.iterrows():
                        rec = {
                            "ts": row["ts"].isoformat() if pd.notna(row["ts"]) else None,
                            "spread_1k": row["spread_1k"] if pd.notna(row["spread_1k"]) else None,
                            "score": row["score"] if pd.notna(row["score"]) else None,
                            "oi_24h_change": row["oi_24h_change"] if pd.notna(row["oi_24h_change"]) else None,
                            "vol_24h_change": row["vol_24h_change"] if pd.notna(row["vol_24h_change"]) else None,
                            "volume_24h": row["volume_24h"] if pd.notna(row["volume_24h"]) else None,
                            "market_cap": row["market_cap"] if pd.notna(row["market_cap"]) else None,
                        }
                        # 处理 oi_exchanges JSONB 字段
                        if pd.notna(row["oi_exchanges"]):
                            if isinstance(row["oi_exchanges"], dict):
                                rec["oi_exchanges"] = row["oi_exchanges"]
                            elif isinstance(row["oi_exchanges"], str):
                                try:
                                    rec["oi_exchanges"] = json.loads(row["oi_exchanges"])
                                except json.JSONDecodeError:
                                    rec["oi_exchanges"] = {}
                            else:
                                rec["oi_exchanges"] = {}
                        else:
                            rec["oi_exchanges"] = {}
                        records.append(rec)
                    history[symbol] = records
                return history
        finally:
            conn.close()
    except Exception as e:  # noqa: BLE001
        print(f"[app] Failed to load history from database: {e!r}")

    return {}


def _build_history_df(history: Dict[str, list], symbol: str) -> pd.DataFrame:
    sym = symbol.upper()
    records = history.get(sym, [])
    if not isinstance(records, list) or not records:
        return pd.DataFrame(
            columns=[
                "ts",
                "spread_1k",
                "score",
                "oi_24h_change",
                "vol_24h_change",
                "volume_24h",
                "market_cap",
            ]
        )
    df = pd.DataFrame(records)
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
        df = df.sort_values("ts")
        df["ts"] = df["ts"] + pd.Timedelta(hours=8)
    return df


def _render_history_chart(df_hist: pd.DataFrame, symbol: str) -> None:
    metrics = [
        ("1k spread", "spread_1k"),
        ("Liquidity score", "score"),
        ("Volume 24h change", "vol_24h_change"),
        ("Volume 24h", "volume_24h"),
        ("Market cap", "market_cap"),
    ]

    if "hist_metric_idx" not in st.session_state:
        st.session_state["hist_metric_idx"] = 0

    idx = st.session_state["hist_metric_idx"]
    name, col = metrics[idx]

    df_plot = df_hist.dropna(subset=[col]).copy()
    if df_plot.empty:
        st.info("No historical data for this metric yet.")
        return

    df_plot = df_plot.sort_values("ts")
    fig = px.line(df_plot, x="ts", y=col, markers=True, template="plotly_white")
    fig.update_traces(
        hovertemplate="Time=%{x|%Y-%m-%d %H:%M}<br>" + f"{name}=" + "%{y:.3f}<extra></extra>"
    )
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=360)

    col_left, col_center, col_right = st.columns([1, 6, 1])
    with col_left:
        if st.button("←", key="hist_prev"):
            st.session_state["hist_metric_idx"] = (st.session_state["hist_metric_idx"] - 1) % len(metrics)
    with col_center:
        st.caption(f"{symbol.upper()} - {name}")
        st.plotly_chart(fig, use_container_width=True)
    with col_right:
        if st.button("→", key="hist_next"):
            st.session_state["hist_metric_idx"] = (st.session_state["hist_metric_idx"] + 1) % len(metrics)


def _render_oi_exchanges_history(df_hist: pd.DataFrame, symbol: str) -> None:
    """Render multi-curve history for total OI 24h change and per-exchange OI 24h change.

    It expects history records to contain an optional "oi_exchanges" field with
    a mapping {exchange_name: percent_change} for each timestamp.
    """

    if "oi_exchanges" not in df_hist.columns:
        return

    rows: list[dict[str, object]] = []
    for _, row in df_hist.iterrows():
        ts = row.get("ts")
        total_val = row.get("oi_24h_change")
        if ts is None:
            continue

        # 总体 OI 变化一条曲线
        rows.append({"ts": ts, "series": "Total", "value": total_val})

        ex_map = row.get("oi_exchanges") or {}
        if isinstance(ex_map, dict):
            for ex_name, ex_val in ex_map.items():
                rows.append({"ts": ts, "series": str(ex_name), "value": ex_val})

    if not rows:
        return

    df_long = pd.DataFrame(rows)
    df_long = df_long.dropna(subset=["value"]).copy()
    if df_long.empty:
        return

    df_long = df_long.sort_values("ts")
    fig = px.line(
        df_long,
        x="ts",
        y="value",
        color="series",
        markers=True,
        template="plotly_white",
    )
    fig.update_traces(
        hovertemplate="Time=%{x|%Y-%m-%d %H:%M}<br>Series=%{legendgroup}<br>Value=%{y:.3f}%<extra></extra>"
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=360,
        legend_title_text="OI 24h change",
    )

    st.caption(f"{symbol.upper()} - OI 24h change by exchange")
    st.plotly_chart(fig, use_container_width=True)


# ---------- Streamlit UI ----------

st.set_page_config(page_title="Crypto Liquidity Monitor", layout="wide")

if "last_updated" not in st.session_state:
    st.session_state["last_updated"] = None

# 每次渲染时尽量从远端 scheduler_state.json 刷新一次最新时间，失败时退回本地文件
remote_state = _try_load_remote_scheduler_state()
if isinstance(remote_state, dict):
    ts_str = remote_state.get("last_start_utc")
    if isinstance(ts_str, str):
        try:
            st.session_state["last_updated"] = datetime.fromisoformat(ts_str)
        except Exception:
            pass
elif st.session_state["last_updated"] is None and os.path.exists(SCHEDULER_STATE_PATH):
    try:
        with open(SCHEDULER_STATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        ts_str = data.get("last_start_utc")
        if ts_str:
            st.session_state["last_updated"] = datetime.fromisoformat(ts_str)
    except Exception:
        pass
if "df_scores" not in st.session_state:
    st.session_state["df_scores"] = None
if "df_raw" not in st.session_state:
    st.session_state["df_raw"] = None

with st.sidebar:
    st.title("Liquidity Monitor")
    # 仅在侧边栏中，将按钮样式改为红色背景、白色文字
    st.markdown(
        """
        <style>
        div[data-testid="stSidebar"] div.stButton > button {
            background-color: #ff4d4f;
            color: white;
            border: none;
        }
        div[data-testid="stSidebar"] div.stButton > button:hover {
            background-color: #ff7875;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Data Refresh button hidden
    # if st.button("Data Refresh", use_container_width=True):
    #     with st.status("Triggering scheduler and waiting for new data...", expanded=True):
    #         ok = _dispatch_and_wait_scheduler()
    #         if ok:
    #             st.success("Data updated")
    #             # Clear cached dataframes and cached functions if any
    #             st.session_state["df_scores"] = None
    #             st.session_state["df_raw"] = None
    #             try:
    #                 st.cache_data.clear()
    #             except Exception:
    #                 pass
    #             st.rerun()
    #         else:
    #             st.warning("Timed out waiting for scheduler run to complete")
    refresh_clicked = False
    page = st.radio("Page", ("Introduction", "Liquidity Score", "Raw Data", "History plot"))

    # History plot page uses symbol selector in the sidebar and does not show search box
    if page in ("Liquidity Score", "Raw Data"):
        query = st.text_input("Search symbol / pair")
    else:
        query = ""

    # History plot page: select symbol in the sidebar
    if page == "History plot":
        hist_data_sidebar = _load_history()
        symbols_hist = sorted(hist_data_sidebar.keys()) if isinstance(hist_data_sidebar, dict) else []
        if symbols_hist:
            default_symbol = symbols_hist[0]
            current_symbol = st.session_state.get("hist_sidebar_symbol", default_symbol)
            try:
                default_index = symbols_hist.index(current_symbol)
            except ValueError:
                default_index = 0
            st.selectbox("Select symbol", symbols_hist, index=default_index, key="hist_sidebar_symbol")
        else:
            st.caption("History data is not available yet. It will be populated by the scheduler runs.")

    if st.session_state["last_updated"] is not None:
        ts = st.session_state["last_updated"].astimezone(UTC8)
        st.caption(f"Last updated (UTC+8): {ts.strftime('%Y-%m-%d %H:%M:%S')}")

st.title("Crypto Liquidity Scoring")

intro = (
    "Crypto Liquidity Monitor tracks the liquidity conditions of a set of perpetual futures.\n\n"
    "**Data sources**\n"
    "- 1k cost (`1k`): latest snapshot from the PostgreSQL table `oracle_exchange_spread`.\n"
    "- Open interest and volume changes (and OI/volume ratio): scraped from Coinglass via Playwright.\n"
    "- GitHub Actions runs `scheduler.py` on a schedule, refreshing crawler results and maintaining the last 7 days in the `liquidity_history` database table.\n\n"
    "**Liquidity score**\n"
    "For each symbol we use three indicators: 1k cost, OI 24h change, and volume 24h change.\n"
    "Before computing z-scores, we winsorize OI 24h change and volume 24h change using a median-absolute-deviation (MAD) rule: values beyond median ± 3×MAD are clipped to the corresponding bounds.\n"
    "Then all three indicators are z-scored cross-sectionally, and we take the negative of 1k cost, so that:\n"
    "    score = -z(1k) + z(oi_24h_change) + z(vol_24h_change).\n"
    "Lower scores indicate worse liquidity.\n\n"
    "**Pages**\n"
    "- *Liquidity Score*: aggregated liquidity score by symbol, with search by symbol / pair and a Data Refresh button in the sidebar.\n"
    "- *Raw Data*: underlying indicators (including `oi_vol_ratio_24h`, the ratio of open interest to 24h trading volume), their z-scores, and the final score, for debugging and deeper analysis.\n"
    "- *History plot*: for a single symbol, time series of 1k cost, score and volume 24h change, plus total OI 24h change and per-exchange OI 24h change curves.\n\n"
    "These views together let you quickly see, in both cross-section and time series, which symbols are experiencing deteriorating or improving liquidity."
)

if page == "Introduction":
    st.markdown(intro)
    st.stop()

if st.session_state["df_scores"] is None:
    with st.spinner("Loading data from database and latest crawler results..."):
        base_df = load_liquidity_from_db()
        oi_json, tv_json = load_crawler_results()
        if not oi_json or not tv_json:
            st.warning(
                "Crawler result files are not available yet. "
                "Please wait for the GitHub Actions scheduler to run at least once."
            )
        df_oi = parse_openinterest_24h(oi_json)
        df_tv = parse_tradingvolume_24h(tv_json)
        df_scores, df_raw = compute_scores(base_df, df_oi, df_tv)
        st.session_state["df_scores"] = df_scores
        st.session_state["df_raw"] = df_raw

df_scores = st.session_state["df_scores"]
df_raw = st.session_state["df_raw"]

if df_scores is None or df_raw is None:
    st.error("No data available.")
    st.stop()

if query:
    q = query.strip().upper()
    mask_scores = df_scores["symbol"].str.upper().str.contains(q) | df_scores["pair_name"].str.upper().str.contains(q)
    mask_raw = df_raw["symbol"].str.upper().str.contains(q) | df_raw["pair_name"].str.upper().str.contains(q)
    df_scores_view = df_scores[mask_scores]
    df_raw_view = df_raw[mask_raw]
else:
    df_scores_view = df_scores
    df_raw_view = df_raw

if page == "Liquidity Score":
    st.subheader("Overview")
    _show_overview_metrics(df_scores_view, df_raw_view)
    st.subheader("Liquidity score by symbol")
    st.dataframe(df_scores_view, use_container_width=True)

elif page == "Raw Data":
    st.subheader("Source data")
    styled_raw = _style_raw_table(df_raw_view, df_raw)
    st.dataframe(styled_raw, use_container_width=True)

elif page == "History plot":
    history = _load_history()
    if not history:
        st.info("History data is not available yet. It will be populated by the scheduler runs.")
        st.stop()

    symbols_hist = sorted(history.keys()) if isinstance(history, dict) else []
    if not symbols_hist:
        st.info("No symbols available for history display.")
        st.stop()

    symbol = st.session_state.get("hist_sidebar_symbol") or symbols_hist[0]
    df_hist = _build_history_df(history, symbol)
    if df_hist.empty:
        st.info("No history data for this symbol yet.")
    else:
        st.subheader(f"History plot - {symbol.upper()}")
        _render_history_chart(df_hist, symbol)
        _render_oi_exchanges_history(df_hist, symbol)
