# Honesty notes

What this system claims, and what it does not.

## The claim

There isn't one, and that is the point. Over ~57k hourly bars (2019-2026), 46
multi-source causal features (price technicals, perp funding positioning, ETH
cross-market, on-chain fundamentals, Fear & Greed sentiment) and two rounds of
model search, **no model beats the random-walk baseline on next-24h MAE** (best:
-3.1% lift). The residue of signal is directional only (~52.6% out-of-sample vs a
50% coin flip) with no usable magnitude. That is the honest ceiling for 24h BTC
with public data; anyone claiming much more is leaking the future or overfitting
a backtest. One caveat the other way: our "sentiment" input (Fear & Greed) is
partly derived from price itself, so genuinely exogenous sentiment (news/social
text) remains untested here.

## Why the numbers can be trusted

- **Walk-forward only.** TimeSeriesSplit with a 24-bar embargo between train and
  test. The label spans the next 24 bars, so neighbouring rows share futures; a
  shuffled K-fold would leak. Nothing here is scored on bars the model saw.
- **Point-in-time features.** Every hourly feature is computable at the bar's
  close. Daily series (on-chain, sentiment) carry a +1 day publication lag baked
  into the join key: a bar on day D reads day D-1's values.
- **One code path for features.** The per-source feature modules are shared by the
  offline pipeline and the live predictor; the offline join is done by the feature
  view, the online vector by the same modules. No train/serve skew.
- **Fee-aware, non-overlapping backtest.** Positions change at most every 24h and
  pay 4 bps per flip. Overlapping-window backtests double-count the same move.

## What it does NOT claim

- It does not claim tradable alpha. A thin MAE edge over persistence is a modelling
  result, not a strategy; execution, slippage, funding costs and regime breaks eat
  edges this size.
- The backtest is not a promise. It is the out-of-sample residue of a model search;
  the search itself selects for backtest luck (one champion out of seven configs).
- Open interest / long-short positioning are absent from v1: Binance only retains
  30 days of them, which cannot support multi-year training. They belong to a v2
  online feature group filled going-forward.
