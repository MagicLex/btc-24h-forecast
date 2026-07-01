"""Shared, causal feature extraction for hourly BTC bars.

ONE extractor, used by both the feature pipeline that fills the offline feature
group and the predictor that serves a live forecast, so training and serving cannot
skew.

Hard rule: every feature on bar `t` is knowable at the close of bar `t`. Everything
is a `shift`/`rolling` over the past. No feature peeks at a future bar. The target
(the forward 24h return) is NOT computed here -- it needs future bars and only
exists offline; see `pipelines.feature_pipeline.add_label`.

Input: a DataFrame of hourly OHLCV sorted ascending by `open_time`, columns:
    open_time (int ms, UTC), open, high, low, close, volume   (floats)

Output of `compute_features(df)`: the same frame plus one column per FEATURE_NAMES.
Rows without enough history to fill every feature carry NaN and are dropped by the
caller (the feature pipeline) or ignored by the predictor (which uses only the last
fully-formed row).
"""
import numpy as np
import pandas as pd

# Horizon of the forecast, in hours. The label is the log return from close[t] to
# close[t + HORIZON]. Kept here so the label builder and the app agree on one number.
HORIZON = 24

# Lags (hours) at which we read the trailing log return.
_RET_LAGS = [1, 2, 3, 6, 12, 24, 48, 72, 168]
# Windows (hours) for rolling volatility and moving-average momentum.
_VOL_WINDOWS = [24, 72, 168]
_MA_WINDOWS = [24, 72, 168]

FEATURE_NAMES = (
    [f"ret_{h}h" for h in _RET_LAGS]
    + [f"vol_{w}h" for w in _VOL_WINDOWS]
    + [f"mom_{w}h" for w in _MA_WINDOWS]
    + ["rsi_14", "vol_z_24h", "range_24h", "pos_in_range_168h",
       "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
)


def _rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def compute_features(df):
    """Attach FEATURE_NAMES to an ascending hourly OHLCV frame. Pure causal."""
    df = df.sort_values("open_time").reset_index(drop=True).copy()
    close = df["close"].astype(float)
    logret = np.log(close).diff()  # hourly log return, causal

    for h in _RET_LAGS:
        # cumulative log return over the trailing h hours, known at bar t
        df[f"ret_{h}h"] = np.log(close / close.shift(h))
    for w in _VOL_WINDOWS:
        df[f"vol_{w}h"] = logret.rolling(w).std()
    for w in _MA_WINDOWS:
        df[f"mom_{w}h"] = close / close.rolling(w).mean() - 1.0

    df["rsi_14"] = _rsi(close, 14)

    vol = df["volume"].astype(float)
    vmean = vol.rolling(24).mean()
    vstd = vol.rolling(24).std().replace(0, np.nan)
    df["vol_z_24h"] = ((vol - vmean) / vstd).fillna(0.0)

    # realized range over the last 24h relative to price
    hi24 = df["high"].astype(float).rolling(24).max()
    lo24 = df["low"].astype(float).rolling(24).min()
    df["range_24h"] = (hi24 - lo24) / close

    # where the current close sits inside the trailing 168h high-low channel (0..1)
    hi168 = df["high"].astype(float).rolling(168).max()
    lo168 = df["low"].astype(float).rolling(168).min()
    df["pos_in_range_168h"] = ((close - lo168) / (hi168 - lo168).replace(0, np.nan)).clip(0, 1)

    # cyclical calendar features from the bar's own timestamp (no future info)
    ts = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    hour = ts.dt.hour
    dow = ts.dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    return df


def latest_feature_row(df):
    """Features for the most recent fully-formed bar, as a 1-row DataFrame.

    Used by the predictor: it fetches the trailing window, computes features, and
    scores the last row that has no NaN in any feature.
    """
    feats = compute_features(df)
    valid = feats.dropna(subset=FEATURE_NAMES)
    if valid.empty:
        return None
    return valid.iloc[[-1]].reset_index(drop=True)
