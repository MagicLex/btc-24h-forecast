"""Fetch daily BTC on-chain fundamentals from blockchain.com charts (free, no key).

Network health and usage: hashrate (security/miner commitment), difficulty,
transaction count (usage), mempool size (congestion), miners' revenue. Daily series,
years of history. These are exogenous fundamentals a pure price model never sees.

Each chart is one endpoint returning {values: [{x: unix_s, y: float}, ...]}. Cached
to data/onchain.parquet, one column per metric, indexed by day.

    python collect/fetch_onchain.py
"""
import os
import time

import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
CACHE = os.path.join(DATA, "onchain.parquet")
BASE = "https://api.blockchain.info/charts/{chart}"

# chart id -> column name
CHARTS = {
    "hash-rate": "hashrate",
    "difficulty": "difficulty",
    "n-transactions": "n_transactions",
    "mempool-size": "mempool_size",
    "miners-revenue": "miners_revenue",
}


def _chart(s, chart):
    """Page in 2-year chunks: the API downsamples long timespans (timespan=all
    returns weekly-ish points), but <=2 years stays daily."""
    frames = []
    for start in ("2019-01-01", "2021-01-01", "2023-01-01", "2025-01-01"):
        r = s.get(BASE.format(chart=chart),
                  params={"timespan": "2years", "start": start, "format": "json"},
                  timeout=25)
        r.raise_for_status()
        v = r.json().get("values", [])
        frames.append(pd.DataFrame([(int(p["x"]), float(p["y"])) for p in v],
                                   columns=["ts", "val"]))
        time.sleep(0.2)
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["ts"])


def fetch():
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0"})
    merged = None
    for chart, col in CHARTS.items():
        df = _chart(s, chart)
        # normalise to day (UTC midnight) so different charts align
        df["day"] = (df["ts"] // 86400) * 86400
        df = df.groupby("day", as_index=False)["val"].last().rename(columns={"val": col})
        print(f"  {col}: {len(df)} days", flush=True)
        merged = df if merged is None else merged.merge(df, on="day", how="outer")
        time.sleep(0.25)

    merged = merged.sort_values("day").reset_index(drop=True)
    merged = merged.ffill()  # carry last known daily value forward across any gap
    merged["event_time"] = pd.to_datetime(merged["day"], unit="s", utc=True)
    os.makedirs(DATA, exist_ok=True)
    merged.to_parquet(CACHE)
    print(f"cached {len(merged)} on-chain days, "
          f"{merged['event_time'].min()} .. {merged['event_time'].max()}")
    return merged


if __name__ == "__main__":
    fetch()
