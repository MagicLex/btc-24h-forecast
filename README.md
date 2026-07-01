# btc-24h-forecast

Live BTC next-24h return forecaster on Hopsworks. Five public data sources, one
feature view, a walk-forward autoresearch loop, a KServe endpoint that rebuilds the
feature vector live, and a Streamlit front-end with a trade suggestion it politely
begs you not to follow.

> **THIS IS DANGEROUS AND NOT FINANCIAL ADVICE.** A model with a thin edge over a
> random walk still loses often. This is an ML systems demo, not a strategy.

## System

```mermaid
flowchart LR
    subgraph sources
      B[Binance BTC 1h] --> F
      E[Binance ETH 1h] --> F
      FU[Perp funding] --> F
      OC[blockchain.com on-chain 1d] --> F
      SG[Fear & Greed 1d] --> F
    end
    subgraph Feature
      F[feature_pipeline.py] --> FG1[(btc_price_1h)]
      F --> FG2[(btc_funding_1h)]
      F --> FG3[(crypto_cross_1h)]
      F --> FG4[(btc_onchain_1d)]
      F --> FG5[(market_sentiment_1d)]
    end
    FG1 & FG2 & FG3 & FG4 & FG5 --> FV[btc_fwd24_fv<br/>join in the store]
    subgraph Training
      FV --> AR[autoresearch walk-forward] --> M[(btc_fwd24<br/>model registry)]
      AR --> LB[(btc_leaderboard)]
      AR --> SC[(btc_scored OOS)]
    end
    subgraph Inference
      M --> D[btcforecaster KServe] --> APP[btclive Streamlit]
      SC --> APP
      LB --> APP
    end
```

The five feature groups each keep their own cadence; the **feature view owns the
join** (hourly on `open_time`, daily on a lagged day key -- a bar on day D reads day
D-1's on-chain/sentiment values, publication lag baked into the key). The serving
endpoint rebuilds the identical 46-feature vector from the same shared modules,
live, at request time.

## Results (walk-forward, out-of-sample)

<!-- METRICS -->

![MAE per fold](assets/mae_per_fold.png)
![Backtest](assets/backtest.png)
![Pred vs actual](assets/pred_vs_actual.png)
![Feature importance](assets/feature_importance.png)

Every number is walk-forward (`TimeSeriesSplit`, 24-bar embargo -- the label spans
24 bars, a shuffled K-fold would leak). The backtest is non-overlapping and pays 4
bps per position flip. See [docs/HONESTY.md](docs/HONESTY.md) for what this does
and does not claim.

## Features (46, all causal)

| block | features | source |
|---|---|---|
| technicals (23) | multi-lag returns, rolling vol, MA momentum, RSI, volume z, ranges, calendar | Binance BTC 1h |
| funding (5) | rate, 72h mean, 240h z, 24h cum, positive-share | Binance perp |
| cross (6) | ETH returns, BTC-ETH spread, ETHBTC momentum, ETH vol, rolling corr | Binance ETH 1h |
| on-chain (7) | hashrate, difficulty, tx count, mempool, miners revenue (levels+momenta) | blockchain.com |
| sentiment (5) | Fear & Greed level, changes, trend, distance from neutral | alternative.me |

Open interest / long-short ratios are deliberately absent: Binance retains only 30
days, which cannot support multi-year training. They belong to a v2 online FG.

## Run it

```bash
make features      # ingest all five sources into their FGs (~7 years hourly)
make train-job     # walk-forward autoresearch as a Hopsworks job -> champion + images
make serve         # deploy the champion as the btcforecaster KServe endpoint
make app           # deploy the Streamlit front-end
make smoke         # poke the endpoint
```

Schedule `pipelines/feature_pipeline.py` hourly (`hops job schedule`) to keep the
store fresh; the endpoint pulls its trailing window live and needs no refresh.
