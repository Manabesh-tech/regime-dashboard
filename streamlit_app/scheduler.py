import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import psycopg2
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CRAWLER_DIR = os.path.join(PROJECT_ROOT, "crawler")
RESULT_DIR = os.path.join(CRAWLER_DIR, "result")
STATE_PATH = os.path.join(PROJECT_ROOT, "scheduler_state.json")
HISTORY_PATH = os.path.join(RESULT_DIR, "history.json")

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

UTC = timezone.utc


def pair_to_symbol(pair: str) -> str:
    """Map pair name (e.g. BTCUSDT, BTC-USDT, BTC/USDT) to base symbol."""
    s = pair.upper().split("/")[0]
    s = s.replace("-", "").replace("_", "")
    for suffix in ["USDT", "USD", "PERP", "USDC"]:
        if s.endswith(suffix):
            return s[: -len(suffix)]
    return s


def load_symbols() -> list[str]:
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        df = pd.read_sql(SQL_LIQUIDITY, conn)
    finally:
        conn.close()
    df["pair_name"] = df["pair_name"].astype(str)
    df["symbol"] = df["pair_name"].apply(pair_to_symbol)
    symbols = sorted(df["symbol"].dropna().unique().tolist())
    return symbols


def run_crawlers(symbols: list[str]) -> None:
    if not symbols:
        return
    symbols_arg = ",".join(sorted(set(s.upper() for s in symbols)))
    main_crawler = os.path.join(CRAWLER_DIR, "main_crawler.py")
    cmd = ["python", main_crawler, "--symbols", symbols_arg]
    print(f"[scheduler] Running crawlers: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=CRAWLER_DIR, check=False)


def _parse_percent(text: str) -> float | None:
    if text is None:
        return None
    t = str(text).strip().replace("%", "")
    if t == "":
        return None
    try:
        return float(t)
    except ValueError:
        return None


def parse_openinterest_24h(oi_data: dict) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for symbol, v in oi_data.items():
        if not isinstance(v, dict):
            continue

        raw_24h = v.get("持仓变化（24小时）") or ""
        val_24h = _parse_percent(raw_24h)

        rows.append({"symbol": str(symbol).upper(), "oi_24h_change": val_24h})

    if not rows:
        return pd.DataFrame(columns=["symbol", "oi_24h_change"])
    return pd.DataFrame(rows)


def parse_tradingvolume_24h(tv_data: dict) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for symbol, v in tv_data.items():
        if not isinstance(v, dict):
            continue
        change_24h = v.get("成交额24h%") or ""
        val = _parse_percent(change_24h)
        volume_abs = v.get("成交额24h")
        mcap = v.get("总市值")
        rows.append(
            {
                "symbol": str(symbol).upper(),
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


def compute_scores(df_base: pd.DataFrame, df_oi: pd.DataFrame, df_tv: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df_base.copy()
    df = df.merge(df_oi, on="symbol", how="left")
    df = df.merge(df_tv, on="symbol", how="left")

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
        "oi_24h_change",
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


def update_history(now: datetime) -> None:
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        try:
            df_base = pd.read_sql(SQL_LIQUIDITY, conn)
        finally:
            conn.close()
    except Exception as e:  # noqa: BLE001
        print(f"[scheduler] Failed to load liquidity from DB for history: {e!r}")
        return

    df_base["pair_name"] = df_base["pair_name"].astype(str)
    df_base["1k"] = pd.to_numeric(df_base["1k"], errors="coerce")
    df_base["symbol"] = df_base["pair_name"].apply(pair_to_symbol)

    oi_path = os.path.join(RESULT_DIR, "openinterest.json")
    tv_path = os.path.join(RESULT_DIR, "tradingvolume.json")
    oi_json: dict = {}
    tv_json: dict = {}
    if os.path.exists(oi_path):
        try:
            with open(oi_path, "r", encoding="utf-8") as f:
                oi_json = json.load(f)
        except json.JSONDecodeError:
            oi_json = {}
    if os.path.exists(tv_path):
        try:
            with open(tv_path, "r", encoding="utf-8") as f:
                tv_json = json.load(f)
        except json.JSONDecodeError:
            tv_json = {}

    # 解析每个 symbol 下各交易所的 24h OI 变化（来自 openinterest_crawler 写入的 "交易所数据" 字段）
    oi_exchanges_map: dict[str, dict[str, float | None]] = {}
    for sym_key, v in (oi_json or {}).items():
        if not isinstance(v, dict):
            continue
        raw_list = v.get("交易所数据")
        if not isinstance(raw_list, list) or not raw_list:
            continue
        ex_map: dict[str, float | None] = {}
        for item in raw_list:
            if not isinstance(item, dict):
                continue
            ex_name = str(item.get("交易所", "")).strip()
            raw_val = item.get("持仓变化（24小时）") or ""
            if not ex_name:
                continue
            val = _parse_percent(str(raw_val))
            ex_map[ex_name] = val
        if ex_map:
            oi_exchanges_map[str(sym_key).upper()] = ex_map

    df_oi = parse_openinterest_24h(oi_json)
    df_tv = parse_tradingvolume_24h(tv_json)
    df_scores, df_raw = compute_scores(df_base, df_oi, df_tv)

    history: dict[str, list[dict[str, object]]] = {}
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                history = json.load(f)
        except json.JSONDecodeError:
            history = {}

    ts = now.astimezone(UTC).isoformat()

    for _, row in df_raw.iterrows():
        sym = str(row["symbol"]).upper()
        spread = row["1k"]
        oi_24h = row["oi_24h_change"]
        vol_24h = row["vol_24h_change"]
        score = row["score"]
        volume_abs = row.get("volume_24h")
        mcap = row.get("market_cap")
        ex_map_row = oi_exchanges_map.get(sym, {})

        def _to_scalar(x: object) -> object:
            if isinstance(x, (float, int)):
                if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
                    return None
                return float(x)
            try:
                if pd.isna(x):  # type: ignore[func-returns-value]
                    return None
            except Exception:
                pass
            return x

        # 将每个交易所的数值也转换为纯 float / None，保证 JSON 可序列化
        oi_exchanges_clean: dict[str, float | None] = {}
        if isinstance(ex_map_row, dict):
            for ex_name, ex_val in ex_map_row.items():
                if ex_val is None:
                    oi_exchanges_clean[ex_name] = None
                else:
                    oi_exchanges_clean[ex_name] = float(ex_val)

        rec = {
            "ts": ts,
            "spread_1k": _to_scalar(spread),
            "score": _to_scalar(score),
            "oi_24h_change": _to_scalar(oi_24h),
            "vol_24h_change": _to_scalar(vol_24h),
            "volume_24h": _to_scalar(volume_abs),
            "market_cap": _to_scalar(mcap),
            "oi_exchanges": oi_exchanges_clean,
        }

        lst = history.get(sym, [])
        if lst and isinstance(lst[-1], dict) and lst[-1].get("ts") == ts:
            continue
        lst.append(rec)
        history[sym] = lst

    cutoff = now - timedelta(days=7)
    for sym, lst in list(history.items()):
        if not isinstance(lst, list):
            continue
        new_lst = []
        for item in lst:
            if not isinstance(item, dict):
                new_lst.append(item)
                continue
            ts_str = item.get("ts")
            if not isinstance(ts_str, str):
                new_lst.append(item)
                continue
            try:
                dt = datetime.fromisoformat(ts_str)
            except Exception:
                new_lst.append(item)
                continue
            if dt >= cutoff:
                new_lst.append(item)
        if new_lst:
            history[sym] = new_lst
        else:
            del history[sym]

    os.makedirs(RESULT_DIR, exist_ok=True)
    tmp_path = HISTORY_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, HISTORY_PATH)
    print(f"[scheduler] Updated history at {HISTORY_PATH}")


def read_last_start() -> datetime | None:
    if not os.path.exists(STATE_PATH):
        return None
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        ts = data.get("last_start_utc")
        if not ts:
            return None
        return datetime.fromisoformat(ts).astimezone(UTC)
    except Exception:
        return None


def write_last_start(ts: datetime) -> None:
    data = {"last_start_utc": ts.astimezone(UTC).isoformat()}
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f)


def main_once() -> None:
    """Run scheduler logic once.

    This is intended for environments like GitHub Actions, which already provide
    periodic scheduling (e.g. cron). It reuses the 10-minute guard against
    running too close to a manual refresh.
    """

    now = datetime.now(UTC)
    last_start = read_last_start()
    if last_start is not None and (now - last_start) < timedelta(minutes=10):
        print(
            f"[scheduler] Skipping run at {now.isoformat()} because last_start={last_start.isoformat()} (<10min)"
        )
        return

    print(f"[scheduler] Starting one-shot run at {now.isoformat()}")
    symbols = load_symbols()
    write_last_start(now)
    run_crawlers(symbols)
    update_history(now)


def main_loop() -> None:
    print("[scheduler] Starting hourly scheduler loop")
    last_hour_triggered: int | None = None

    while True:
        now = datetime.now(UTC)
        if now.minute == 0 and now.second < 10:
            if last_hour_triggered != now.hour:
                last_start = read_last_start()
                if last_start is None or (now - last_start) >= timedelta(minutes=10):
                    print(f"[scheduler] Top-of-hour trigger at {now.isoformat()}")
                    symbols = load_symbols()
                    write_last_start(now)
                    run_crawlers(symbols)
                    update_history(now)
                else:
                    print(
                        f"[scheduler] Skipping auto-run at {now.isoformat()} because last_start={last_start.isoformat()}"
                    )
                last_hour_triggered = now.hour
        time.sleep(10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["once", "loop"],
        default="once",
        help="Run once (for cron / GitHub Actions) or as a long-running loop.",
    )
    args = parser.parse_args()

    if args.mode == "loop":
        main_loop()
    else:
        main_once()
