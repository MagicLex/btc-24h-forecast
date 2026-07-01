"""Causal funding-rate features on the hourly bar grid.

Shared by the feature pipeline and the predictor, so no train/serve skew. Funding
prints every 8h; we carry the last known rate forward to each hour (backward asof =
the rate known at the bar's close) and roll trailing stats over hours.

Signal reading: persistently high positive funding = crowded, leveraged longs
paying to hold = contrarian bearish; deeply negative = crowded shorts.
"""
import numpy as np
import pandas as pd

FEATURE_NAMES = ["funding_rate", "funding_ma_72h", "funding_z_240h",
                 "funding_cum_24h", "funding_pos_frac_72h"]


def build(funding_df, open_times):
    """open_times: 1-D array of bar open_time (int ms). Returns DataFrame keyed by open_time."""
    f = funding_df.sort_values("funding_time").reset_index(drop=True)
    grid = pd.DataFrame({"open_time": np.sort(np.asarray(open_times, dtype="int64"))})
    # backward asof: each hour gets the most recent funding rate known at/at-or-before it
    merged = pd.merge_asof(grid, f.rename(columns={"funding_time": "open_time"}),
                           on="open_time", direction="backward")
    r = merged["funding_rate"].astype(float)
    out = pd.DataFrame({"open_time": merged["open_time"]})
    out["funding_rate"] = r
    out["funding_ma_72h"] = r.rolling(72, min_periods=9).mean()
    mu = r.rolling(240, min_periods=30).mean()
    sd = r.rolling(240, min_periods=30).std().replace(0, np.nan)
    out["funding_z_240h"] = ((r - mu) / sd)
    out["funding_cum_24h"] = r.rolling(24, min_periods=3).sum()
    out["funding_pos_frac_72h"] = (r > 0).rolling(72, min_periods=9).mean()
    return out
