"""Fetch the daily Crypto Fear & Greed index from alternative.me (free, no key).

A 0-100 composite (volatility, momentum, social, dominance, trends). Extreme fear
has historically preceded bounces and extreme greed preceded pullbacks, so it is a
contrarian sentiment signal exogenous to price. Full history back to 2018.

    python collect/fetch_sentiment.py
"""
import os

import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
CACHE = os.path.join(DATA, "sentiment.parquet")
URL = "https://api.alternative.me/fng/"


def fetch():
    s = requests.Session()
    r = s.get(URL, params={"limit": 0, "format": "json"}, timeout=25)  # limit=0 -> all history
    r.raise_for_status()
    data = r.json()["data"]
    df = pd.DataFrame([(int(x["timestamp"]), int(x["value"])) for x in data],
                      columns=["ts", "fng_value"])
    df["day"] = (df["ts"] // 86400) * 86400
    df = df.groupby("day", as_index=False)["fng_value"].last().sort_values("day").reset_index(drop=True)
    df["event_time"] = pd.to_datetime(df["day"], unit="s", utc=True)
    os.makedirs(DATA, exist_ok=True)
    df.to_parquet(CACHE)
    print(f"cached {len(df)} F&G days, {df['event_time'].min()} .. {df['event_time'].max()}")
    return df


if __name__ == "__main__":
    fetch()
