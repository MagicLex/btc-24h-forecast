"""Causal sentiment features from the daily Crypto Fear & Greed index.

0-100 composite. Contrarian read: extreme fear (<25) has historically preceded
bounces, extreme greed (>75) pullbacks. Level, short changes, trend, and signed
distance from neutral.

Point-in-time: the day's index is published at day's end, so we lag +1 day
(`avail_time` = day + 1d) and backward-asof-join, same as the on-chain block. An
hourly bar on day D reads day D-1's index. No same-day leak.
"""
import pandas as pd

FEATURE_NAMES = ["fng_value", "fng_chg_1d", "fng_chg_7d", "fng_ma_7d", "fng_dist_50"]


def build(fng_df):
    d = fng_df.sort_values("event_time").reset_index(drop=True).copy()
    v = d["fng_value"].astype(float)
    out = pd.DataFrame()
    out["fng_value"] = v
    out["fng_chg_1d"] = v.diff(1)
    out["fng_chg_7d"] = v.diff(7)
    out["fng_ma_7d"] = v.rolling(7, min_periods=2).mean()
    out["fng_dist_50"] = v - 50.0
    out["avail_time"] = d["event_time"] + pd.Timedelta(days=1)
    # equi-join key for the feature view (data day, epoch ms); see onchain_features.
    out["day_ms"] = (d["event_time"].astype("int64") // 10**6).astype("int64")
    return out.dropna(subset=["avail_time"]).reset_index(drop=True)
