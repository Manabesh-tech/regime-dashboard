import json
import os
import subprocess
import sys
from typing import Dict, Any

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError, sync_playwright


URL = "https://www.coinglass.com/zh/BitcoinOpenInterest"


def _ensure_playwright_browsers() -> None:
    """Best-effort download of Playwright browsers (chromium) if missing.

    On Streamlit Cloud only the Python package is installed by default. Running
    ``python -m playwright install chromium`` here makes sure the headless
    browser executable exists. If this fails we silently ignore the error so
    that local runs (where GitHub Actions already installs browsers) are not
    affected.
    """

    try:
        subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        # Best-effort; if this fails Playwright will still raise a clear error later.
        pass


def _update_progress(progress_file: str | None, done: int, total: int) -> None:
    """Write progress JSON to the given file in an atomic way.

    Format: {"done": int, "total": int}
    """

    if not progress_file:
        return

    try:
        os.makedirs(os.path.dirname(progress_file), exist_ok=True)
        tmp_path = progress_file + ".tmp"
        payload = {"done": done, "total": total}
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp_path, progress_file)
    except Exception:
        # Best-effort; do not break crawler because of progress IO
        pass


def fetch_openinterest_per_coin(
    url: str = URL,
    headless: bool = True,
    timeout_ms: int = 30000,
    per_coin_delay_ms: int = 800,
    target_symbols: dict[str, str] | None = None,
    progress_file: str | None = None,
) -> Dict[str, Any]:
    """Fetch open interest change metrics per coin from CoinGlass.

    This follows your最初的稳定版本逻辑：单次 goto + 固定 5 秒等待，
    仅在遍历按钮时使用可选的 target_symbols 做筛选。
    """

    _ensure_playwright_browsers()

    with sync_playwright() as playwright:
        # 使用更接近真实浏览器的配置，减少 headless 被识别为爬虫的概率
        browser = playwright.chromium.launch(
            headless=headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-sandbox",
            ],
            ignore_default_args=["--enable-automation"],
        )

        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1920, "height": 1080},
            device_scale_factor=1,
            is_mobile=False,
            has_touch=False,
        )

        # 隐藏 navigator.webdriver
        context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
        )

        page = context.new_page()
        page.goto(url, wait_until="networkidle", timeout=timeout_ms)
        print(f"[openinterest_crawler] Loaded URL: {page.url}")
        print(f"[openinterest_crawler] Page title: {page.title()}")

        # 显式等待币种按钮出现，替代固定 5 秒死等
        page.wait_for_selector("button.cg-tab-item, button.MuiButton-root.cg-tab-item", timeout=timeout_ms)

        buttons = page.query_selector_all("button.cg-tab-item, button.MuiButton-root.cg-tab-item")
        print(f"[openinterest_crawler] Found {len(buttons)} coin buttons")
        if not buttons:
            browser.close()
            return {}

        # 先根据 target_symbols 过滤出本次实际要处理的按钮集合，用于精确计算 total
        selected: list[tuple[int, str, Any]] = []
        for index, button in enumerate(buttons):
            raw_name = button.inner_text().strip()
            name = raw_name or f"UNKNOWN_{index}"

            if target_symbols is not None:
                key = name.upper()
                if key not in target_symbols:
                    continue

            selected.append((index, name, button))

        total = len(selected)
        if total == 0:
            _update_progress(progress_file, 0, 0)
            browser.close()
            return {}

        results: Dict[str, Any] = {}

        for i, (index, name, button) in enumerate(selected, start=1):
            button.click()

            if per_coin_delay_ms > 0:
                page.wait_for_timeout(per_coin_delay_ms)

            # 第二行通常是该币种整体汇总数据
            row_selector = "div.ant-table-wrapper table tbody tr:nth-child(2)"

            try:
                cell_1h = page.wait_for_selector(
                    f"{row_selector} td:nth-child(6)",
                    timeout=5000,
                )
                text_1h = cell_1h.inner_text().strip()
            except PlaywrightTimeoutError:
                text_1h = ""

            cell_4h = page.query_selector(f"{row_selector} td:nth-child(7)")
            text_4h = cell_4h.inner_text().strip() if cell_4h else ""

            cell_24h = page.query_selector(f"{row_selector} td:nth-child(8)")
            text_24h = cell_24h.inner_text().strip() if cell_24h else ""

            cell_oi_vol = page.query_selector(f"{row_selector} td:nth-child(9)")
            text_oi_vol = cell_oi_vol.inner_text().strip() if cell_oi_vol else ""

            results[name] = {
                "持仓变化（1小时）": text_1h,
                "持仓变化（4小时）": text_4h,
                "持仓变化（24小时）": text_24h,
                "持仓/24小时成交额": text_oi_vol,
            }

            # 抓取表格第 3-7 行的交易所名称和“持仓变化（24小时）”列
            # 对应你的 XPath：tbody/tr[3-7]/td[2] 为交易所列，td[8] 为持仓变化（24小时）
            exchange_results: list[dict[str, str]] = []
            for j in range(3, 8):
                row_sel = f"div.ant-table-wrapper table tbody tr:nth-child({j})"
                # 交易所名称：优先取 div.symbol-name 文本
                cell_ex = page.query_selector(f"{row_sel} td:nth-child(2) div.symbol-name")
                if cell_ex is None:
                    cell_ex = page.query_selector(f"{row_sel} td:nth-child(2)")
                ex_name = cell_ex.inner_text().strip() if cell_ex else ""

                cell_24h_ex = page.query_selector(f"{row_sel} td:nth-child(8)")
                ex_24h = cell_24h_ex.inner_text().strip() if cell_24h_ex else ""

                if ex_name or ex_24h:
                    exchange_results.append(
                        {
                            "交易所": ex_name,
                            "持仓变化（24小时）": ex_24h,
                        }
                    )

            results[name]["交易所数据"] = exchange_results

            _update_progress(progress_file, i, total)

        browser.close()

    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--symbols",
        type=str,
        default="",
        help="Comma-separated symbols to fetch (e.g. BTC,ETH,SOL)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Path to output JSON file (optional)",
    )
    parser.add_argument(
        "--progress-file",
        type=str,
        default="",
        help="Path to a JSON file where progress will be written (optional)",
    )
    args = parser.parse_args()

    target_symbols = None
    if args.symbols:
        raw_list = [s.strip() for s in args.symbols.split(",") if s.strip()]
        target_symbols = {s.upper(): s for s in raw_list}

    progress_file = args.progress_file or None
    data = fetch_openinterest_per_coin(target_symbols=target_symbols, progress_file=progress_file)

    if args.output:
        output_path = args.output
        base_dir = os.path.dirname(output_path)
        if base_dir:
            os.makedirs(base_dir, exist_ok=True)
    else:
        result_dir = os.path.join(os.path.dirname(__file__), "result")
        os.makedirs(result_dir, exist_ok=True)
        output_path = os.path.join(result_dir, "openinterest.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[openinterest_crawler] Saved result to {output_path}")


if __name__ == "__main__":
    main()
