"""Assemble the full multi-source feature frame -- ONLINE path + offline fallback.

Offline, the five per-source feature groups are joined by the FEATURE VIEW (the
store owns the join; see pipelines/feature_pipeline.py). This assembler exists for
the paths that cannot read that FV: the KServe predictor, which rebuilds the vector
live from the same five raw sources, and the local training fallback while a fresh
FG materializes. It is built from the SAME per-source feature modules the pipeline
inserts from, so the two paths cannot skew.

Point-in-time discipline:
  - price / funding / cross are hourly -> exact join on open_time.
  - on-chain / sentiment are daily and lagged +1 day in their feature modules, then
    backward merge_asof on time -> an hourly bar reads only strictly-earlier daily data.

The forward-return label is NOT added here (it needs future bars); the pipeline adds it.
"""
import pandas as pd

import btc_features
import funding_features
import cross_features
import onchain_features
import sentiment_features

BAR_COLS = ["open", "high", "low", "close", "volume"]
ALL_FEATURE_NAMES = (btc_features.FEATURE_NAMES
                     + funding_features.FEATURE_NAMES
                     + cross_features.FEATURE_NAMES
                     + onchain_features.FEATURE_NAMES
                     + sentiment_features.FEATURE_NAMES)


def _asof_daily(frame, daily_feats, names):
    """Backward merge_asof of a +1d-lagged daily feature block onto the hourly frame."""
    right = daily_feats.sort_values("avail_time").reset_index(drop=True)
    merged = pd.merge_asof(frame.sort_values("event_time"), right[["avail_time"] + names],
                           left_on="event_time", right_on="avail_time", direction="backward")
    return merged.drop(columns=["avail_time"])


def assemble_features(btc_df, eth_df, funding_df, onchain_df, sentiment_df):
    price = btc_features.compute_features(btc_df)
    price = price[["open_time"] + BAR_COLS + btc_features.FEATURE_NAMES]

    fund = funding_features.build(funding_df, price["open_time"].values)
    cross = cross_features.build(btc_df, eth_df)

    frame = (price.merge(fund, on="open_time", how="left")
                  .merge(cross, on="open_time", how="left"))
    frame["event_time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)

    frame = _asof_daily(frame, onchain_features.build(onchain_df),
                        onchain_features.FEATURE_NAMES)
    frame = _asof_daily(frame, sentiment_features.build(sentiment_df),
                        sentiment_features.FEATURE_NAMES)

    cols = ["open_time", "event_time"] + BAR_COLS + ALL_FEATURE_NAMES
    return frame[cols].sort_values("open_time").reset_index(drop=True)
