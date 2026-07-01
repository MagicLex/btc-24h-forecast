"""Feature pipeline (F stage): five sources -> five feature groups, one per source.

The join does NOT happen here. Each source gets its own FG at its own cadence, and
the feature view (autoresearch/search.py) joins them: hourly FGs on `open_time`,
daily FGs on a data-day key with the +1d publication lag baked in (hourly
`prev_day_ms` == daily `day_ms`). The store owns the join; this script only ingests.

  btc_price_1h         hourly   OHLCV + technicals + forward-24h label + prev_day_ms
  btc_funding_1h       hourly   perp funding positioning features
  crypto_cross_1h      hourly   BTC/ETH cross-market features
  btc_onchain_1d       daily    network fundamentals (event_time = day+1, publish lag)
  market_sentiment_1d  daily    Fear & Greed sentiment (same lag)

`features.assemble_features` (the same per-source modules, merged in pandas) is for
the ONLINE path only: the KServe predictor can't read the offline FV, so it rebuilds
the identical vector live. One module per source keeps both paths skew-free.

    python pipelines/feature_pipeline.py --hours 20000
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

# As a Hopsworks job this file runs from an uploaded copy in Resources/jobs/<name>/,
# not from the repo, so anchor ROOT on wherever the repo actually lives on /hopsfs.
import glob

def _find_root():
    cand = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for p in [cand] + sorted(glob.glob("/hopsfs/Users/*/btc-24h-forecast")):
        if os.path.exists(os.path.join(p, "features.py")):
            return p
    raise RuntimeError("repo root with features.py not found")

ROOT = _find_root()
sys.path.insert(0, ROOT)
import btc_features                                                  # noqa: E402
import funding_features                                              # noqa: E402
import cross_features                                                # noqa: E402
import onchain_features                                              # noqa: E402
import sentiment_features                                            # noqa: E402
from btc_features import HORIZON                                     # noqa: E402
from collect.fetch_klines import fetch as fetch_klines               # noqa: E402
from collect.fetch_funding import fetch as fetch_funding             # noqa: E402
from collect.fetch_onchain import fetch as fetch_onchain             # noqa: E402
from collect.fetch_sentiment import fetch as fetch_sentiment         # noqa: E402

LABEL = "fwd_return_24h"
BAR_COLS = ["open", "high", "low", "close", "volume"]
DAY_MS = 86400_000

FG_PRICE = "btc_price_1h"
FG_FUNDING = "btc_funding_1h"
FG_CROSS = "crypto_cross_1h"
FG_ONCHAIN = "btc_onchain_1d"
FG_SENTIMENT = "market_sentiment_1d"


def _event_time(df):
    df["event_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df


def build_frames(hours):
    btc = fetch_klines(hours, "BTCUSDT")
    eth = fetch_klines(hours, "ETHUSDT")
    funding = fetch_funding()
    onchain = fetch_onchain()
    sentiment = fetch_sentiment()

    price = btc_features.compute_features(btc)
    close = price["close"].astype(float)
    price[LABEL] = np.log(close.shift(-HORIZON) / close)
    # join key toward the daily FGs: yesterday's data day (publication lag in the key)
    price["prev_day_ms"] = (price["open_time"] // DAY_MS - 1) * DAY_MS
    price = _event_time(price)
    price = price[["open_time", "event_time", "prev_day_ms"] + BAR_COLS
                  + btc_features.FEATURE_NAMES + [LABEL]]
    price = price.dropna().reset_index(drop=True)

    fund = funding_features.build(funding, btc["open_time"].values)
    fund = _event_time(fund).dropna().reset_index(drop=True)

    cross = cross_features.build(btc, eth)
    cross = _event_time(cross).dropna().reset_index(drop=True)

    onc = onchain_features.build(onchain)
    onc = (onc.rename(columns={"avail_time": "event_time"})
              [["day_ms", "event_time"] + onchain_features.FEATURE_NAMES]
              .dropna().reset_index(drop=True))

    sent = sentiment_features.build(sentiment)
    sent = (sent.rename(columns={"avail_time": "event_time"})
                [["day_ms", "event_time"] + sentiment_features.FEATURE_NAMES]
                .dropna().reset_index(drop=True))

    return price, fund, cross, onc, sent


def insert(hours):
    import hopsworks
    price, fund, cross, onc, sent = build_frames(hours)
    proj = hopsworks.login()
    fs = proj.get_feature_store()

    specs = [
        (FG_PRICE, price, ["open_time"],
         "Hourly BTC/USDT bars: OHLCV, causal technicals, forward-24h log-return "
         "label, prev_day_ms join key toward the daily FGs."),
        (FG_FUNDING, fund, ["open_time"],
         "BTC perp funding positioning features on the hourly grid (rate, trailing "
         "mean/z/cum, positive-share). Contrarian leverage signal."),
        (FG_CROSS, cross, ["open_time"],
         "BTC/ETH cross-market features: ETH returns, spread, ETHBTC momentum, "
         "ETH vol, rolling BTC-ETH correlation."),
        (FG_ONCHAIN, onc, ["day_ms"],
         "Daily BTC on-chain fundamentals (hashrate, difficulty, tx count, mempool, "
         "miners revenue). event_time = day+1: published with a 1-day lag."),
        (FG_SENTIMENT, sent, ["day_ms"],
         "Daily Crypto Fear & Greed sentiment features (level, changes, trend). "
         "event_time = day+1: published with a 1-day lag."),
    ]
    for name, df, pk, desc in specs:
        fg = fs.get_or_create_feature_group(
            name=name, version=1, description=desc, primary_key=pk,
            event_time="event_time", online_enabled=False)
        fg.insert(df, write_options={"wait_for_job": False})
        print(f"  {name}: {len(df)} rows inserted (materializing async)")
    print("all five feature groups written; the feature view owns the join.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=20000)
    a = ap.parse_args()
    insert(a.hours)
