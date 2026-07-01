"""Causal on-chain fundamental features (daily), point-in-time safe.

Network security (hashrate), mining economics (difficulty, miners' revenue), usage
(transaction count) and congestion (mempool size). Slow-moving fundamentals a price
model never sees.

Point-in-time: a day's on-chain aggregate is only fully known after that day ends, so
we publish it with a +1 day lag (`avail_time` = day + 1d). The pipeline / predictor
backward-asof-joins on `avail_time`, so an hourly bar on day D uses day D-1's values.
No same-day leak.
"""
import numpy as np
import pandas as pd

FEATURE_NAMES = ["hashrate_log", "hashrate_mom_7d", "difficulty_mom_14d",
                 "ntx_z_30d", "mempool_log", "mempool_mom_7d", "miners_rev_mom_7d"]
DAY = 86400_000  # ms


def build(onchain_df):
    d = onchain_df.sort_values("event_time").reset_index(drop=True).copy()
    out = pd.DataFrame()
    out["hashrate_log"] = np.log(d["hashrate"].astype(float))
    out["hashrate_mom_7d"] = d["hashrate"].astype(float).pct_change(7)
    out["difficulty_mom_14d"] = d["difficulty"].astype(float).pct_change(14)
    ntx = d["n_transactions"].astype(float)
    out["ntx_z_30d"] = (ntx - ntx.rolling(30, min_periods=7).mean()) / \
        ntx.rolling(30, min_periods=7).std().replace(0, np.nan)
    mem = d["mempool_size"].astype(float)
    out["mempool_log"] = np.log(mem.replace(0, np.nan))
    out["mempool_mom_7d"] = mem.pct_change(7)
    out["miners_rev_mom_7d"] = d["miners_revenue"].astype(float).pct_change(7)
    # +1 day publish lag: value for day D is usable from D+1
    out["avail_time"] = d["event_time"] + pd.Timedelta(days=1)
    # equi-join key for the feature view: the data day, epoch ms. An hourly bar on
    # day D joins day_ms == D-1, which is the same row the backward asof would pick.
    out["day_ms"] = (d["event_time"].astype("int64") // 10**6).astype("int64")
    return out.dropna(subset=["avail_time"]).reset_index(drop=True)
