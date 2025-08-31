import os
import json
import requests
import datetime
from dataclasses import asdict
from typing import List, Optional

from models import OptionContract, Greeks
from market_context import MarketContext

# -------------------------------
# CONFIG
# -------------------------------
API_KEY = "YdHjgU3cLma7X2WghK0pGLFAjup0hb86"
UNDERLYING = "QQQ"   # proxy for NQ
MONTHS_BACK = 24
OUTPUT_DIR = "data/chains"

# -------------------------------
# UTILS: save/load option chains
# -------------------------------

def _option_to_dict(opt: OptionContract) -> dict:
    d = asdict(opt)
    if d.get("greeks") is None:
        d.pop("greeks", None)
    return d

def _dict_to_option(d: dict) -> OptionContract:
    g: Optional[Greeks] = None
    if "greeks" in d and d["greeks"] is not None:
        g = Greeks(**d["greeks"])
    return OptionContract(
        strike=float(d["strike"]),
        right=str(d["right"]),
        expiry=str(d["expiry"]),
        bid=float(d["bid"]),
        ask=float(d["ask"]),
        mid=float(d["mid"]),
        volume=int(d.get("volume", 0)),
        open_interest=int(d.get("open_interest", 0)),
        greeks=g,
    )

def save_option_chain_json(options: List[OptionContract], underlying: str, expiry_yyyymmdd: str) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"{underlying}_{expiry_yyyymmdd}.json")
    payload = [_option_to_dict(o) for o in options]
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path

def load_option_chain_json(path: str) -> List[OptionContract]:
    with open(path, "r") as f:
        payload = json.load(f)
    return [_dict_to_option(d) for d in payload]

# -------------------------------
# POLYGON HELPERS
# -------------------------------

def polygon_symbol(symbol: str, expiry: str, right: str, strike: float) -> str:
    """Build Polygon option ticker symbol, e.g. O:QQQ250822C00450000"""
    yymmdd = datetime.datetime.strptime(expiry, "%Y-%m-%d").strftime("%y%m%d")
    right = right.upper()[0]
    strike_int = int(strike * 1000)
    return f"O:{symbol}{yymmdd}{right}{strike_int:08d}"

def get_underlying_close(symbol: str, date: str) -> float:
    url = f"https://api.polygon.io/v1/open-close/{symbol}/{date}"
    resp = requests.get(url, params={"apiKey": API_KEY})
    if resp.status_code == 200:
        return resp.json().get("close", 0)
    return 0

def get_contract_ohlc(ticker: str, date: str):
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{date}/{date}"
    resp = requests.get(url, params={"apiKey": API_KEY})
    if resp.status_code == 200:
        results = resp.json().get("results", [])
        return results[0] if results else None
    return None

# -------------------------------
# MAIN LOOP
# -------------------------------

def download_chains():
    end = datetime.date.today()
    start = end - datetime.timedelta(days=30 * MONTHS_BACK)

    day = start
    while day <= end:
        # Skip weekends
        if day.weekday() >= 5:
            day += datetime.timedelta(days=1)
            continue

        day_str = day.strftime("%Y-%m-%d")
        expiry_yyyymmdd = day.strftime("%Y%m%d")

        spot = get_underlying_close(UNDERLYING, day_str)
        if spot == 0:
            print(f"⚠️ No close price for {day_str}, skipping")
            day += datetime.timedelta(days=1)
            continue

        # Build strikes ±10% around spot
        strike_inc = 5
        lower = int((spot * 0.9) // strike_inc) * strike_inc
        upper = int((spot * 1.1) // strike_inc) * strike_inc
        strikes = list(range(lower, upper + strike_inc, strike_inc))

        options = []
        for strike in strikes:
            for right in ["C", "P"]:
                sym = polygon_symbol(UNDERLYING, day_str, right, strike)
                ohlc = get_contract_ohlc(sym, day_str)
                if not ohlc:
                    continue
                bid = ohlc["o"]
                ask = ohlc["c"]
                mid = (bid + ask) / 2 if bid and ask else 0
                options.append(
                    OptionContract(
                        strike=float(strike),
                        right=right,
                        expiry=expiry_yyyymmdd,
                        bid=float(bid),
                        ask=float(ask),
                        mid=mid,
                        volume=ohlc.get("v", 0),
                        open_interest=0,
                        greeks=None,
                    )
                )

        if options:
            path = save_option_chain_json(options, UNDERLYING, expiry_yyyymmdd)
            print(f"✅ {day_str}: Saved {len(options)} contracts -> {path}")
        else:
            print(f"⚠️ {day_str}: No contracts built")

        day += datetime.timedelta(days=1)

if __name__ == "__main__":
    download_chains()
