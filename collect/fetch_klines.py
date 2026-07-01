"""Fetch hourly OHLCV from Binance, paginated and resumable. BTC and ETH.

Binance klines are free, no key, 1000 bars per call. We page backward from now
until we have `hours` of history, cache the raw bars per symbol to
data/klines_<sym>.parquet, and on re-run extend/refresh from cache. One public
endpoint, one interval -- boring on purpose.

    python collect/fetch_klines.py --symbol BTCUSDT --hours 20000
    python collect/fetch_klines.py --symbol ETHUSDT --hours 20000

If Binance is unreachable, set BTC_SOURCE=coinbase (300 bars/call, BTC/ETH only).
"""
import argparse
import os
import time

import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

BINANCE = "https://api.binance.com/api/v3/klines"
COINBASE = "https://api.exchange.coinbase.com/products/{product}/candles"
COLS = ["open_time", "open", "high", "low", "close", "volume"]
HOUR_MS = 3600_000
_COINBASE_PRODUCT = {"BTCUSDT": "BTC-USD", "ETHUSDT": "ETH-USD"}


def _cache(symbol):
    return os.path.join(DATA, f"klines_{symbol.lower()}.parquet")


def _binance_page(s, symbol, end_ms, limit=1000):
    start_ms = end_ms - limit * HOUR_MS
    r = s.get(BINANCE, params={"symbol": symbol, "interval": "1h",
                               "startTime": start_ms, "endTime": end_ms,
                               "limit": limit}, timeout=20)
    r.raise_for_status()
    rows = [[int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])]
            for k in r.json()]
    return pd.DataFrame(rows, columns=COLS)


def _coinbase_page(s, symbol, end_ms, limit=300):
    end = end_ms // 1000
    start = end - limit * 3600
    url = COINBASE.format(product=_COINBASE_PRODUCT[symbol])
    r = s.get(url, params={"granularity": 3600, "start": start, "end": end}, timeout=20)
    r.raise_for_status()
    # coinbase: [time, low, high, open, close, volume], descending
    rows = [[int(k[0]) * 1000, float(k[3]), float(k[2]), float(k[1]), float(k[4]), float(k[5])]
            for k in r.json()]
    return pd.DataFrame(rows, columns=COLS).sort_values("open_time").reset_index(drop=True)


def fetch(hours, symbol="BTCUSDT", source=None):
    source = source or os.environ.get("BTC_SOURCE", "binance")
    page = _coinbase_page if source == "coinbase" else _binance_page
    step = 300 if source == "coinbase" else 1000
    cache = _cache(symbol)
    s = requests.Session()
    now_ms = int(time.time() * 1000) // HOUR_MS * HOUR_MS
    target_ms = now_ms - hours * HOUR_MS

    frames = []
    stop_ms = target_ms  # page backward no further than this
    if os.path.exists(cache):
        have = pd.read_parquet(cache)
        frames.append(have)
        cmin, cmax = int(have["open_time"].min()), int(have["open_time"].max())
        print(f"[{symbol}] cache: {len(have)} bars up to "
              f"{pd.to_datetime(cmax, unit='ms', utc=True)}")
        # cache already reaches back to target -> only fill the forward gap (now..cmax)
        if cmin <= target_ms:
            stop_ms = cmax

    end_ms = now_ms
    while end_ms > stop_ms:
        df = page(s, symbol, end_ms, step)
        if df.empty:
            break
        frames.append(df)
        oldest = int(df["open_time"].min())
        print(f"  [{symbol}] pulled {len(df)} bars back to "
              f"{pd.to_datetime(oldest, unit='ms', utc=True)}", flush=True)
        if oldest >= end_ms:
            break
        end_ms = oldest
        time.sleep(0.25)

    out = (pd.concat(frames, ignore_index=True)
             .drop_duplicates(subset=["open_time"])
             .sort_values("open_time")
             .reset_index(drop=True))
    os.makedirs(DATA, exist_ok=True)
    out.to_parquet(cache)
    span = pd.to_datetime(out["open_time"], unit="ms", utc=True)
    print(f"[{symbol}] cached {len(out)} bars, {span.min()} .. {span.max()}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--hours", type=int, default=20000)
    ap.add_argument("--source", default=None)
    a = ap.parse_args()
    fetch(a.hours, a.symbol, a.source)
