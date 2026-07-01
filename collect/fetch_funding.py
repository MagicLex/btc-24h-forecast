"""Fetch BTC perpetual funding-rate history from Binance futures.

Funding is paid every 8h on the perp; a high positive rate means longs pay shorts
(crowded, leveraged longs -> contrarian bearish), negative means the opposite. Real
positioning signal, and the full history is available (unlike the futures/data/*
open-interest endpoints, which only keep 30 days -- those are left to a v2 online FG).

fapi/v1/fundingRate returns <=1000 rows per call; we page forward from the start.
Cached to data/funding.parquet, resumable.

    python collect/fetch_funding.py
"""
import os
import time

import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
CACHE = os.path.join(DATA, "funding.parquet")
URL = "https://fapi.binance.com/fapi/v1/fundingRate"
START_MS = 1567296000000  # Sep 2019, the BTCUSDT perp's first funding prints


def fetch():
    s = requests.Session()
    start = START_MS
    frames = []
    if os.path.exists(CACHE):
        have = pd.read_parquet(CACHE)
        frames.append(have)
        start = int(have["funding_time"].max()) + 1
        print(f"cache: {len(have)} funding points up to "
              f"{pd.to_datetime(have['funding_time'].max(), unit='ms', utc=True)}")

    while True:
        r = s.get(URL, params={"symbol": "BTCUSDT", "startTime": start, "limit": 1000}, timeout=20)
        r.raise_for_status()
        rows = r.json()
        if not rows:
            break
        df = pd.DataFrame([[int(x["fundingTime"]), float(x["fundingRate"])] for x in rows],
                          columns=["funding_time", "funding_rate"])
        frames.append(df)
        last = int(df["funding_time"].max())
        print(f"  pulled {len(df)} to {pd.to_datetime(last, unit='ms', utc=True)}", flush=True)
        if len(rows) < 1000:
            break
        start = last + 1
        time.sleep(0.25)

    out = (pd.concat(frames, ignore_index=True)
             .drop_duplicates(subset=["funding_time"])
             .sort_values("funding_time")
             .reset_index(drop=True))
    os.makedirs(DATA, exist_ok=True)
    out.to_parquet(CACHE)
    print(f"cached {len(out)} funding points")
    return out


if __name__ == "__main__":
    fetch()
