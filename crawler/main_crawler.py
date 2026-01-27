import argparse
import json
import math
import os
import subprocess
import sys
from typing import List, Any


THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _split_symbols(symbols_arg: str, workers: int) -> List[str]:
    """Split comma-separated symbols into N roughly equal shards.

    Each shard will be joined back into a comma-separated string and passed
    to openinterest_crawler.py via --symbols.
    """

    raw = [s.strip() for s in symbols_arg.split(",") if s.strip()]
    if not raw:
        return []

    workers = max(1, min(workers, len(raw)))
    chunk_size = max(1, math.ceil(len(raw) / workers))

    shards: List[str] = []
    for i in range(0, len(raw), chunk_size):
        shard = ",".join(raw[i : i + chunk_size])
        shards.append(shard)
    return shards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--symbols",
        type=str,
        default="",
        help="Comma-separated symbols to pass to openinterest_crawler (e.g. BTC,ETH)",
    )
    parser.add_argument(
        "--oi-workers",
        type=int,
        default=1,
        help="Number of parallel openinterest_crawler workers (processes).",
    )
    parser.add_argument(
        "--oi-progress-prefix",
        type=str,
        default="",
        help="Prefix for openinterest progress files (e.g. result/oi_progress)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        help="Which crawlers to run: oi, tv, or both.",
    )
    parser.add_argument(
        "--tv-progress-file",
        type=str,
        default="",
        help="Path for trading volume progress file.",
    )
    args = parser.parse_args()

    symbols_arg = args.symbols.strip()
    workers = max(1, args.oi_workers)
    oi_progress_prefix = args.oi_progress_prefix.strip()
    mode = (args.mode or "both").lower()
    run_oi = mode in ("oi", "both")
    run_tv = mode in ("tv", "both")
    tv_progress_file = args.tv_progress_file.strip() or None

    processes: List[tuple[str, subprocess.Popen]] = []
    oi_output_files: List[str] = []

    # ---------- openinterest_crawler workers ----------
    if run_oi:
        oi_path = os.path.join(THIS_DIR, "openinterest_crawler.py")
        if not os.path.exists(oi_path):
            print(f"[main_crawler] Script not found: {oi_path}")
        else:
            if symbols_arg and workers > 1:
                shards = _split_symbols(symbols_arg, workers)
                for idx, shard in enumerate(shards, start=1):
                    cmd = [sys.executable, oi_path, "--symbols", shard]
                    if oi_progress_prefix:
                        progress_file = f"{oi_progress_prefix}_{idx}.json"
                        cmd.extend(["--progress-file", progress_file])
                    output_rel = os.path.join("result", f"openinterest_{idx}.json")
                    cmd.extend(["--output", output_rel])
                    oi_output_files.append(os.path.join(THIS_DIR, output_rel))
                    print(f"[main_crawler] Starting OI worker {idx}/{len(shards)}: {' '.join(cmd)}")
                    p = subprocess.Popen(cmd, cwd=THIS_DIR)
                    processes.append((f"openinterest_crawler.py[{idx}]", p))
            else:
                cmd = [sys.executable, oi_path]
                if symbols_arg:
                    cmd.extend(["--symbols", symbols_arg])
                if oi_progress_prefix:
                    progress_file = f"{oi_progress_prefix}_1.json"
                    cmd.extend(["--progress-file", progress_file])
                output_rel = os.path.join("result", "openinterest_1.json")
                cmd.extend(["--output", output_rel])
                oi_output_files.append(os.path.join(THIS_DIR, output_rel))
                print(f"[main_crawler] Starting: {' '.join(cmd)}")
                p = subprocess.Popen(cmd, cwd=THIS_DIR)
                processes.append(("openinterest_crawler.py", p))

    # ---------- tradingvolume_crawler (single process) ----------
    if run_tv:
        tv_path = os.path.join(THIS_DIR, "tradingvolume_crawler.py")
        if not os.path.exists(tv_path):
            print(f"[main_crawler] Script not found: {tv_path}")
        else:
            cmd = [sys.executable, tv_path]
            if tv_progress_file:
                cmd.extend(["--progress-file", tv_progress_file])
            print(f"[main_crawler] Starting: {' '.join(cmd)}")
            p = subprocess.Popen(cmd, cwd=THIS_DIR)
            processes.append(("tradingvolume_crawler.py", p))

    # ---------- wait for all ----------
    for name, p in processes:
        ret = p.wait()
        print(f"[main_crawler] Script {name} finished with exit code {ret}")

    # ---------- merge openinterest outputs ----------
    if run_oi and oi_output_files:
        merged: dict[str, Any] = {}
        for path in oi_output_files:
            try:
                if not os.path.exists(path):
                    continue
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    merged.update(data)
            except Exception:
                continue

        result_dir = os.path.join(THIS_DIR, "result")
        os.makedirs(result_dir, exist_ok=True)
        final_path = os.path.join(result_dir, "openinterest.json")

        # 根据第一轮结果，找出需要重试的 symbol（全部字段为空或缺失）
        retry_symbols: list[str] = []
        if symbols_arg:
            desired = [s.strip().upper() for s in symbols_arg.split(",") if s.strip()]
            for sym in desired:
                v = merged.get(sym)
                if not isinstance(v, dict):
                    retry_symbols.append(sym)
                    continue
                fields = [
                    v.get("持仓变化（1小时）"),
                    v.get("持仓变化（4小时）"),
                    v.get("持仓变化（24小时）"),
                    v.get("持仓/24小时成交额"),
                ]
                if all((str(f or "").strip() == "") for f in fields):
                    retry_symbols.append(sym)

        # 对需要重试的 symbol 再单线程跑一遍 openinterest_crawler
        retry_output_abs: str | None = None
        if retry_symbols:
            unique_retry = sorted(set(retry_symbols))
            symbols_arg_retry = ",".join(unique_retry)
            retry_output_rel = os.path.join("result", "openinterest_retry.json")
            retry_output_abs = os.path.join(THIS_DIR, retry_output_rel)

            cmd = [sys.executable, oi_path, "--symbols", symbols_arg_retry, "--output", retry_output_rel]
            print(
                f"[main_crawler] Retrying {len(unique_retry)} symbols in single-thread OI pass: {' '.join(cmd)}"
            )
            subprocess.run(cmd, cwd=THIS_DIR, check=False)

            # 将重试结果覆盖合并到 merged
            try:
                if os.path.exists(retry_output_abs):
                    with open(retry_output_abs, "r", encoding="utf-8") as f:
                        retry_data = json.load(f)
                    if isinstance(retry_data, dict):
                        merged.update(retry_data)
            except Exception:
                pass

        # 写入最终的 openinterest.json（原子写入）
        tmp_path = final_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, final_path)
        print(f"[main_crawler] Merged openinterest outputs into {final_path}")

        # 清理各个 worker 和重试的临时输出文件，只保留最终的 openinterest.json
        for path in oi_output_files:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                continue
        if retry_symbols and retry_output_abs:
            try:
                if os.path.exists(retry_output_abs):
                    os.remove(retry_output_abs)
            except Exception:
                pass

        # 清理 openinterest 相关进度文件 oi_progress_*.json
        if oi_progress_prefix:
            # 这里不强依赖实际 shard 数量，按 workers 上限尝试删除，不存在就跳过
            for i in range(1, workers + 1):
                prog_path = f"{oi_progress_prefix}_{i}.json"
                try:
                    if os.path.exists(prog_path):
                        os.remove(prog_path)
                except Exception:
                    continue

    # 清理 trading volume 进度文件 tv_progress.json
    if tv_progress_file:
        try:
            if os.path.exists(tv_progress_file):
                os.remove(tv_progress_file)
        except Exception:
            pass


if __name__ == "__main__":
    main()
