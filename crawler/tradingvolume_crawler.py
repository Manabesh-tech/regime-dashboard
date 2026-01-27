import json
import os
import subprocess
import sys
from typing import Dict, Any

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError, sync_playwright


URL = "https://www.coinglass.com/zh"


def _ensure_playwright_browsers() -> None:
    """Best-effort download of Playwright browsers (chromium) if missing."""

    try:
        subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
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


def _parse_chinese_number(text: str | None) -> float | None:
    """Parse numbers like "$973.50亿", "$1.80万亿" or "97,350,039,150.694" into floats.

    返回基础货币单位下的数值（不带万/亿等），解析失败时返回 None。
    """

    if text is None:
        return None

    t = str(text).strip()
    if not t:
        return None

    # 去掉货币符号和空格
    if t.startswith("$") or t.startswith("￥"):
        t = t[1:]
    t = t.replace(" ", "")

    # 去掉千分位逗号
    t = t.replace(",", "")

    multiplier = 1.0
    # 处理中文单位：万亿、亿、万
    if t.endswith("万亿"):
        multiplier = 1e12
        t = t[:-2]
    elif t.endswith("亿"):
        multiplier = 1e8
        t = t[:-1]
    elif t.endswith("万"):
        multiplier = 1e4
        t = t[:-1]

    try:
        base = float(t)
    except ValueError:
        return None

    return base * multiplier


def _parse_current_page(page, results: Dict[str, Any]) -> None:
    rows = page.query_selector_all("div.ant-table-wrapper table tbody tr[data-row-key]")
    print(f"[tradingvolume_crawler] Parsing page, found {len(rows)} rows")

    for row in rows:
        symbol = row.get_attribute("data-row-key") or ""
        if not symbol:
            name_el = row.query_selector(".symbol-name")
            symbol = name_el.inner_text().strip() if name_el else ""
        if not symbol:
            continue
        if symbol in results:
            continue

        percent_text = ""
        volume_text = ""
        mcap_text = ""
        volume_val: float | None = None
        mcap_val: float | None = None

        for attempt in range(2):
            try:
                # 成交额 24h 百分比（原有逻辑，位于第 8 列）
                td_pct = row.query_selector("td:nth-child(8) .Number")
                if not td_pct:
                    td_pct = row.query_selector("td:nth-child(8)")
                if td_pct:
                    text = td_pct.inner_text().strip()
                    if text:
                        percent_text = text

                # 24 小时成交额（第 7 列）
                td_vol = row.query_selector("td:nth-child(7) .Number")
                if not td_vol:
                    td_vol = row.query_selector("td:nth-child(7)")
                if td_vol:
                    aria = (td_vol.get_attribute("aria-label") or "").strip()
                    txt = td_vol.inner_text().strip()
                    if txt:
                        volume_text = txt
                    # 优先用 aria-label 里的纯数字，其次解析展示文本中的万/亿等
                    source = aria or txt
                    if source and volume_val is None:
                        volume_val = _parse_chinese_number(source)

                # 总市值（第 9 列）
                td_mcap = row.query_selector("td:nth-child(9) .Number")
                if not td_mcap:
                    td_mcap = row.query_selector("td:nth-child(9)")
                if td_mcap:
                    aria_m = (td_mcap.get_attribute("aria-label") or "").strip()
                    txt_m = td_mcap.inner_text().strip()
                    if txt_m:
                        mcap_text = txt_m
                    source_m = aria_m or txt_m
                    if source_m and mcap_val is None:
                        mcap_val = _parse_chinese_number(source_m)

                # 成功读取后无需再等太久
                if percent_text and (volume_val is not None) and (mcap_val is not None):
                    break
            except PlaywrightTimeoutError:
                # 允许短暂的重试
                pass
            if attempt == 0:
                page.wait_for_timeout(200)

        results[symbol] = {
            "成交额24h%": percent_text,
            "成交额24h_raw": volume_text,
            "成交额24h": volume_val,
            "总市值_raw": mcap_text,
            "总市值": mcap_val,
        }


def _goto_next_page(page) -> bool:
    pagination = page.query_selector("ul.rc-pagination")
    if not pagination:
        return False

    active = pagination.query_selector("li.rc-pagination-item-active")
    if not active:
        return False

    title = (active.get_attribute("title") or active.inner_text() or "").strip()
    if not title.isdigit():
        return False

    current_page = int(title)
    # 优先尝试直接跳到下一个数字页码
    next_li = pagination.query_selector(f"li.rc-pagination-item[title='{current_page + 1}']")
    target = None
    if next_li:
        target = next_li.query_selector("button") or next_li
    else:
        # 当页码较多时，Coinglass 可能只显示部分数字 + “下一页”箭头
        # 此时退而求其次，点击通用的 next 按钮
        next_btn = pagination.query_selector(
            "li.rc-pagination-next:not(.rc-pagination-disabled)"
        )
        if next_btn:
            target = next_btn.query_selector("button") or next_btn

    if not target:
        return False

    target.click()
    try:
        page.wait_for_selector(
            "div.ant-table-wrapper table tbody tr[data-row-key]",
            timeout=5000,
        )
    except PlaywrightTimeoutError:
        return False
    return True


def fetch_trading_volume_24h_percent(
    url: str = URL,
    headless: bool = True,
    timeout_ms: int = 30000,
    progress_file: str | None = None,
) -> Dict[str, Any]:
    _ensure_playwright_browsers()
    with sync_playwright() as playwright:
        # 使用与 openinterest_crawler 相同的伪装配置，降低 headless 模式下被识别为爬虫的概率
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

        context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
        )

        page = context.new_page()
        try:
            page.goto(url, wait_until="networkidle", timeout=timeout_ms)
        except PlaywrightTimeoutError:
            print(
                f"[tradingvolume_crawler] Timeout when loading {url} with timeout_ms={timeout_ms}; returning empty result"
            )
            browser.close()
            return {}

        print(f"[tradingvolume_crawler] Loaded URL: {page.url}")
        print(f"[tradingvolume_crawler] Page title: {page.title()}")
        page.wait_for_selector(
            "div.ant-table-wrapper table tbody tr[data-row-key]",
            timeout=timeout_ms,
        )

        try:
            dropdown = page.query_selector("div.rc-select-selector")
            if dropdown:
                dropdown.click()
                page.wait_for_timeout(500)
                option = page.query_selector("div.rc-select-item-option[title='100'], div.rc-select-item-option:has-text(\"100\")")
                if option:
                    option.click()
                    page.wait_for_timeout(1000)
                    print("[tradingvolume_crawler] Set page size to 100")
        except Exception as e:  # noqa: BLE001
            print(f"[tradingvolume_crawler] Failed to set page size to 100: {e!r}")

        # 预估总页数，便于进度条展示
        max_page: int | None = None
        try:
            pagination = page.query_selector("ul.rc-pagination")
            if pagination:
                items = pagination.query_selector_all("li.rc-pagination-item")
                pages: list[int] = []
                for item in items:
                    t = (item.get_attribute("title") or item.inner_text() or "").strip()
                    if t.isdigit():
                        pages.append(int(t))
                if pages:
                    max_page = max(pages)
        except Exception:
            max_page = None

        results: Dict[str, Any] = {}
        visited_pages: set[int] = set()

        while True:
            active = page.query_selector("ul.rc-pagination li.rc-pagination-item-active")
            if active:
                title = (active.get_attribute("title") or active.inner_text() or "").strip()
                current_page = int(title) if title.isdigit() else None
            else:
                current_page = None

            if current_page is not None:
                if current_page in visited_pages:
                    break
                visited_pages.add(current_page)
                print(f"[tradingvolume_crawler] At page {current_page}")

            # 更新进度：已访问页数 / 预估总页数
            done_pages = len(visited_pages)
            total_pages = max_page or done_pages
            _update_progress(progress_file, done_pages, total_pages)

            _parse_current_page(page, results)

            if not _goto_next_page(page):
                break

        browser.close()

    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--progress-file",
        type=str,
        default="",
        help="Path to a JSON file where progress will be written (optional)",
    )
    args = parser.parse_args()

    progress_file = args.progress_file or None
    data = fetch_trading_volume_24h_percent(progress_file=progress_file)

    result_dir = os.path.join(os.path.dirname(__file__), "result")
    os.makedirs(result_dir, exist_ok=True)
    output_path = os.path.join(result_dir, "tradingvolume.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[tradingvolume_crawler] Saved result to {output_path}")


if __name__ == "__main__":
    main()
