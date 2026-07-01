"""Causal cross-crypto features: BTC vs ETH.

ETH is BTC's highest-beta liquid cousin. Their return spread and rolling correlation
carry regime information a BTC-only model misses: leadership rotation (ETH leading up
often precedes broad crypto risk-on), and correlation breakdowns that flag idiosyncratic
BTC moves. Shared by pipeline and predictor.

Inputs are two ascending hourly OHLCV frames (open_time, close) for BTC and ETH.
"""
import numpy as np
import pandas as pd

FEATURE_NAMES = ["eth_ret_24h", "eth_ret_72h", "btc_eth_spread_24h",
                 "ethbtc_mom_72h", "eth_vol_24h", "corr_btc_eth_72h"]


def build(btc_df, eth_df):
    b = btc_df[["open_time", "close"]].rename(columns={"close": "btc"})
    e = eth_df[["open_time", "close"]].rename(columns={"close": "eth"})
    m = b.merge(e, on="open_time", how="inner").sort_values("open_time").reset_index(drop=True)
    btc = m["btc"].astype(float)
    eth = m["eth"].astype(float)
    btc_lr = np.log(btc).diff()
    eth_lr = np.log(eth).diff()

    out = pd.DataFrame({"open_time": m["open_time"]})
    out["eth_ret_24h"] = np.log(eth / eth.shift(24))
    out["eth_ret_72h"] = np.log(eth / eth.shift(72))
    out["btc_eth_spread_24h"] = np.log(btc / btc.shift(24)) - out["eth_ret_24h"]
    ratio = eth / btc
    out["ethbtc_mom_72h"] = ratio / ratio.shift(72) - 1.0
    out["eth_vol_24h"] = eth_lr.rolling(24).std()
    out["corr_btc_eth_72h"] = btc_lr.rolling(72).corr(eth_lr)
    return out
