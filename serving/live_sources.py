"""Live trailing-window pulls of the five sources, for serving.

Same upstream endpoints as the collectors, but sized for request time: one call per
source, just enough history to fill every rolling window in the feature modules
(longest: 240h funding z-score, 168h price channel, 30d on-chain z + 14d momentum).
No parquet caching -- the predictor pod is stateless.
"""
import time

import pandas as pd
import requests

KLINES = "https://api.binance.com/api/v3/klines"
FUNDING = "https://fapi.binance.com/fapi/v1/fundingRate"
ONCHAIN = "https://api.blockchain.info/charts/{chart}"
FNG = "https://api.alternative.me/fng/"
HOUR_MS = 3600_000

ONCHAIN_CHARTS = {
    "hash-rate": "hashrate",
    "difficulty": "difficulty",
    "n-transactions": "n_transactions",
    "mempool-size": "mempool_size",
    "miners-revenue": "miners_revenue",
}


def klines(s, symbol, hours=700):
    now_ms = int(time.time() * 1000) // HOUR_MS * HOUR_MS
    r = s.get(KLINES, params={"symbol": symbol, "interval": "1h",
                              "startTime": now_ms - hours * HOUR_MS,
                              "endTime": now_ms, "limit": 1000}, timeout=20)
    r.raise_for_status()
    rows = [[int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])]
            for k in r.json()]
    return pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume"])


def funding(s):
    r = s.get(FUNDING, params={"symbol": "BTCUSDT", "limit": 1000}, timeout=20)  # ~333 days
    r.raise_for_status()
    return pd.DataFrame([[int(x["fundingTime"]), float(x["fundingRate"])] for x in r.json()],
                        columns=["funding_time", "funding_rate"])


def onchain(s):
    merged = None
    for chart, col in ONCHAIN_CHARTS.items():
        r = s.get(ONCHAIN.format(chart=chart),
                  params={"timespan": "6months", "format": "json"}, timeout=25)
        r.raise_for_status()
        df = pd.DataFrame([(int(p["x"]), float(p["y"])) for p in r.json().get("values", [])],
                          columns=["ts", "val"])
        df["day"] = (df["ts"] // 86400) * 86400
        df = df.groupby("day", as_index=False)["val"].last().rename(columns={"val": col})
        merged = df if merged is None else merged.merge(df, on="day", how="outer")
    merged = merged.sort_values("day").ffill().reset_index(drop=True)
    merged["event_time"] = pd.to_datetime(merged["day"], unit="s", utc=True)
    return merged


def sentiment(s):
    r = s.get(FNG, params={"limit": 60, "format": "json"}, timeout=20)
    r.raise_for_status()
    df = pd.DataFrame([(int(x["timestamp"]), int(x["value"])) for x in r.json()["data"]],
                      columns=["ts", "fng_value"])
    df["day"] = (df["ts"] // 86400) * 86400
    df = df.groupby("day", as_index=False)["fng_value"].last().sort_values("day").reset_index(drop=True)
    df["event_time"] = pd.to_datetime(df["day"], unit="s", utc=True)
    return df


def pull_all(session=None):
    s = session or requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0"})
    return {
        "btc": klines(s, "BTCUSDT"),
        "eth": klines(s, "ETHUSDT"),
        "funding": funding(s),
        "onchain": onchain(s),
        "sentiment": sentiment(s),
    }
