import os
import json
import time
import math
import requests
import datetime as dt
from typing import Dict, Any, List

# -----------------------------
# CONFIG
# -----------------------------
API_KEY = "YdHjgU3cLma7X2WghK0pGLFAjup0hb86" # os.getenv("POLYGON_API_KEY", "YOUR_POLYGON_API_KEY")
UNDERLYING = "QQQ"     # proxy for NQ
VOL_INDEX  = "VIX"     # or "VXN" for Nasdaq 100 vol index
MONTHS_BACK = 24
OUTPUT_DIR = "data/marketdata"
RATE_LIMIT_SLEEP = 1.0   # seconds between requests (tune if you get 429s)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# HTTP helpers (with simple retry/backoff)
# -----------------------------
def _get(url: str, params: Dict[str, Any], retries: int = 5) -> requests.Response:
    last_err = None
    for i in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=20)
            if resp.status_code == 429:
                # rate limited: backoff & retry
                sleep_for = RATE_LIMIT_SLEEP * (2 ** i)
                print(f"⏳ 429 rate-limit from {url}. Sleeping {sleep_for:.1f}s...")
                time.sleep(sleep_for)
                continue
            resp.raise_for_status()
            return resp
        except Exception as e:
            last_err = e
            sleep_for = RATE_LIMIT_SLEEP * (2 ** i)
            print(f"⚠️ HTTP error ({e}). Retry {i+1}/{retries} in {sleep_for:.1f}s")
            time.sleep(sleep_for)
    raise last_err if last_err else RuntimeError("Request failed")

# -----------------------------
# Polygon Aggs v2 fetchers
# -----------------------------
def get_daily_aggs(ticker: str, start: str, end: str) -> List[Dict[str, Any]]:
    """
    Fetch daily aggregates for a ticker over a date range [start, end].
    Returns a list of bars: { t, o, h, l, c, v, n, vw } with t = unix ms.
    """
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": API_KEY
    }
    resp = _get(url, params)
    data = resp.json()
    return data.get("results", []) or []

def ms_to_datestr(ms: int) -> str:
    # Polygon returns epoch ms in UTC. Convert to YYYY-MM-DD
    return dt.datetime.utcfromtimestamp(ms / 1000).strftime("%Y-%m-%d")

# -----------------------------
# Main
# -----------------------------
def download_underlying_and_vix(months_back: int = MONTHS_BACK):
    today = dt.date.today()
    start_date = today - dt.timedelta(days=int(round(30.0 * months_back)))
    start_str = start_date.strftime("%Y-%m-%d")
    end_str   = today.strftime("%Y-%m-%d")

    print(f"Fetching daily aggs for {UNDERLYING} from {start_str} to {end_str} ...")
    qqq_bars = get_daily_aggs(UNDERLYING, start_str, end_str)
    print(f"→ got {len(qqq_bars)} daily bars for {UNDERLYING}")

    print(f"Fetching daily aggs for {VOL_INDEX} from {start_str} to {end_str} ...")
    vix_bars = get_daily_aggs(VOL_INDEX, start_str, end_str)
    print(f"→ got {len(vix_bars)} daily bars for {VOL_INDEX}")

    # Index bars by date string for easy joins
    qqq_by_date: Dict[str, Dict[str, Any]] = {}
    for b in qqq_bars:
        d = ms_to_datestr(b["t"])
        qqq_by_date[d] = b

    vix_by_date: Dict[str, Dict[str, Any]] = {}
    for b in vix_bars:
        d = ms_to_datestr(b["t"])
        vix_by_date[d] = b

    # Iterate through *QQQ* trading days and write one file per day
    saved = 0
    for day, b in qqq_by_date.items():
        out_path = os.path.join(OUTPUT_DIR, f"{UNDERLYING}_{day}.json")
        if os.path.exists(out_path):
            # skip if already saved
            continue

        qqq = {
            "symbol": UNDERLYING,
            "date": day,
            "open":  b.get("o"),
            "high":  b.get("h"),
            "low":   b.get("l"),
            "close": b.get("c"),
            "volume": b.get("v"),
            "vwap":   b.get("vw"),
        }

        vb = vix_by_date.get(day, {})
        vix = {
            "symbol": VOL_INDEX,
            "date": day,
            "open":  vb.get("o"),
            "high":  vb.get("h"),
            "low":   vb.get("l"),
            "close": vb.get("c"),
            "volume": vb.get("v"),
            "vwap":   vb.get("vw"),
        } if vb else {}

        record = {
            "date": day,
            "underlying": qqq,
            "vix": vix
        }

        with open(out_path, "w") as f:
            json.dump(record, f, indent=2)
        saved += 1

    print(f"✅ Saved {saved} daily snapshots to {OUTPUT_DIR}")

if __name__ == "__main__":
    if API_KEY == "YOUR_POLYGON_API_KEY":
        raise SystemExit("Please set POLYGON_API_KEY env var or edit the script with your key.")
    download_underlying_and_vix()
