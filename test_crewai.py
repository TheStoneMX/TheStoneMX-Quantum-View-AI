
import requests
import datetime
from models import OptionContract, Greeks
from market_context import MarketContext

API_KEY = "YdHjgU3cLma7X2WghK0pGLFAjup0hb86"
UNDERLYING = "QQQ"   # proxy for NQ futures options
TARGET_DATE = "2025-08-21"  # YYYY-MM-DD

# ------------------------------------------
# Helpers
# ------------------------------------------

def polygon_symbol(symbol: str, expiry: str, right: str, strike: float) -> str:
    """Build Polygon option ticker symbol, e.g. O:QQQ250822C00450000"""
    yymmdd = datetime.datetime.strptime(expiry, "%Y-%m-%d").strftime("%y%m%d")
    right = right.upper()[0]   # "C" or "P"
    strike_int = int(strike * 1000)
    return f"O:{symbol}{yymmdd}{right}{strike_int:08d}"

def get_underlying_close(symbol: str, date: str) -> float:
    """Fetch underlying close price for target date"""
    url = f"https://api.polygon.io/v1/open-close/{symbol}/{date}"
    resp = requests.get(url, params={"apiKey": API_KEY})
    resp.raise_for_status()
    data = resp.json()
    return data.get("close", 0)

def get_contract_ohlc(ticker: str, date: str):
    """Fetch OHLC for a single option contract on given date"""
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{date}/{date}"
    resp = requests.get(url, params={"apiKey": API_KEY})
    resp.raise_for_status()
    results = resp.json().get("results", [])
    return results[0] if results else None

# ------------------------------------------
# Main
# ------------------------------------------

# Step 1: get underlying close on target date
spot = get_underlying_close(UNDERLYING, TARGET_DATE)
print(f"{UNDERLYING} close on {TARGET_DATE}: {spot}")

# Step 2: generate strikes (±10% around spot, nearest 5 increments)
strike_increment = 5
lower = int((spot * 0.9) // strike_increment) * strike_increment
upper = int((spot * 1.1) // strike_increment) * strike_increment
strikes = list(range(lower, upper + strike_increment, strike_increment))

# Step 3: build option symbols
contracts = []
for strike in strikes:
    for right in ["C", "P"]:
        contracts.append(polygon_symbol(UNDERLYING, TARGET_DATE, right, strike))

print(f"Built {len(contracts)} contract symbols")

# Step 4: fetch OHLC for each
options = []
for csym in contracts:
    ohlc = get_contract_ohlc(csym, TARGET_DATE)
    if not ohlc:
        continue

    # Parse symbol
    details = csym.split(":")[1]  # e.g. QQQ250821C00450000
    expiry = "20" + details[3:9]  # convert YYMMDD → YYYYMMDD
    right = details[9]
    strike = int(details[10:]) / 1000

    bid = ohlc["o"]   # open as proxy
    ask = ohlc["c"]   # close as proxy
    mid = (bid + ask) / 2 if bid and ask else 0

    opt = OptionContract(
        strike=float(strike),
        right=right,
        expiry=expiry,
        bid=float(bid),
        ask=float(ask),
        mid=mid,
        volume=ohlc.get("v", 0),
        open_interest=0,  # Polygon doesn’t provide OI in aggs
        greeks=None
    )
    options.append(opt)

print(f"Fetched {len(options)} option contracts for {UNDERLYING} {TARGET_DATE}")

# Step 5: feed into MarketContext
ctx = MarketContext()
ctx.update_option_chain(options)

print("✅ Option chain loaded into MarketContext")