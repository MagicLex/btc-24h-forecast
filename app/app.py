"""BTC 24h forecaster -- Streamlit front for the btcforecaster KServe endpoint.

Thin client: the endpoint computes the forecast (it rebuilds the training feature
vector live); this app renders it and adds the honest context around it -- the MAE
error band, the persistence baseline, the out-of-sample backtest from the scored
feature group, the autoresearch leaderboard, and a position-sizing suggestion with
a disclaimer the size of a house. It loads no model and pins no ML stack.
"""
import math
import time

import hopsworks
import numpy as np
import pandas as pd
import requests
import streamlit as st

DEPLOYMENT = "btcforecaster"
CHAMPION = "btc_fwd24"
SCORED_FG = "btc_scored"
LEADERBOARD_FG = "btc_leaderboard"
HORIZON = 24
FEE = 0.0004
KELLY_FRACTION = 0.25   # quarter-Kelly
KELLY_CAP = 0.10        # never suggest more than 10% of bankroll

st.set_page_config(page_title="BTC 24h forecast", page_icon="chart_with_upwards_trend",
                   layout="wide")

DISCLAIMER = (
    "**THIS IS DANGEROUS AND NOT FINANCIAL ADVICE.** This is a machine-learning "
    "demo. Crypto is extremely volatile; a model with a few percent of edge over a "
    "random walk still loses often and can be wrong for weeks. Do not trade money "
    "you cannot afford to lose. Nothing here is a recommendation."
)


@st.cache_resource
def _project():
    return hopsworks.login()


@st.cache_data(ttl=300, show_spinner="Calling the forecaster...")
def get_forecast():
    proj = _project()
    dep = proj.get_model_serving().get_deployment(DEPLOYMENT)
    return dep.predict(inputs=[{}])["predictions"][0]


@st.cache_data(ttl=300)
def get_model_metrics():
    proj = _project()
    mr = proj.get_model_registry()
    version = max(m.version for m in mr.get_models(CHAMPION))
    return mr.get_model(CHAMPION, version=version).training_metrics


@st.cache_data(ttl=3600, show_spinner="Loading out-of-sample history...")
def get_scored():
    fs = _project().get_feature_store()
    df = fs.get_feature_group(SCORED_FG, version=1).read()
    return df.sort_values("open_time").reset_index(drop=True)


@st.cache_data(ttl=3600)
def get_leaderboard():
    fs = _project().get_feature_store()
    df = fs.get_feature_group(LEADERBOARD_FG, version=1).read()
    return df.sort_values("cv_mae").reset_index(drop=True)


@st.cache_data(ttl=300)
def get_recent_price(hours=336):
    now_ms = int(time.time() * 1000) // 3600_000 * 3600_000
    r = requests.get("https://api.binance.com/api/v3/klines",
                     params={"symbol": "BTCUSDT", "interval": "1h",
                             "startTime": now_ms - hours * 3600_000, "limit": 1000},
                     timeout=20)
    r.raise_for_status()
    df = pd.DataFrame([[int(k[0]), float(k[4])] for k in r.json()],
                      columns=["open_time", "close"])
    df["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df


def backtest(scored):
    """Same non-overlapping fee-aware backtest as autoresearch (every HORIZON bars)."""
    s = scored.iloc[::HORIZON].reset_index(drop=True)
    pos = np.sign(s["y_pred"].values)
    turns = np.abs(np.diff(np.concatenate([[0], pos])))
    strat = pos * s["y_true"].values - turns * FEE
    out = pd.DataFrame({"time": pd.to_datetime(s["event_time"], utc=True),
                        "model long/short": np.exp(np.cumsum(strat)),
                        "buy & hold": np.exp(np.cumsum(s["y_true"].values))})
    sharpe = float(np.mean(strat) / (np.std(strat) + 1e-9) * math.sqrt(365))
    hit = float(np.mean(np.sign(s["y_pred"]) == np.sign(s["y_true"])))
    return out.set_index("time"), sharpe, hit


st.title("BTC -- next-24h forecast")
st.error(DISCLAIMER, icon="⚠️")

try:
    fc = get_forecast()
    metrics = get_model_metrics()
except Exception as e:
    st.warning(f"Forecaster unreachable: {e}")
    st.stop()

pred = float(fc["pred_return_24h"])
close = float(fc["close"])
mae = float(metrics.get("cv_mae", 0.015))
base_mae = float(metrics.get("baseline_mae", mae))
dir_acc = float(metrics.get("cv_dir_acc", 0.5))

target = close * math.exp(pred)
lo, hi = close * math.exp(pred - mae), close * math.exp(pred + mae)

# ---- headline ----------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("BTC now", f"${close:,.0f}", help=f"last closed hourly bar ({fc['asof']})")
c2.metric("24h forecast", f"${target:,.0f}", f"{(math.exp(pred)-1)*100:+.2f}%")
c3.metric("MAE band (68-ish%)", f"${lo:,.0f} - ${hi:,.0f}",
          help="forecast +/- the walk-forward MAE of the champion model")
c4.metric("directional accuracy (OOS)", f"{dir_acc*100:.1f}%",
          f"{(dir_acc-0.5)*100:+.1f} pts vs coin flip")

# ---- trade suggestion --------------------------------------------------------
st.subheader("Suggested trade")
signal_floor = 0.25 * mae   # below this the signal is noise vs the error bar
if pred > signal_floor:
    side, color = "LONG", "green"
elif pred < -signal_floor:
    side, color = "SHORT", "red"
else:
    side, color = "FLAT -- signal below noise floor", "gray"

ret_std = max(mae * 1.35, 1e-6)   # rough sigma from MAE (normal: sigma ~ 1.25*MAE)
kelly = pred / (ret_std ** 2)
stake_frac = float(np.clip(KELLY_FRACTION * abs(kelly), 0.0, KELLY_CAP)) if "FLAT" not in side else 0.0

left, right = st.columns([1, 1])
with left:
    st.markdown(f"### :{color}[{side}]")
    st.caption(f"model edge {pred*100:+.3f}% vs noise floor +/-{signal_floor*100:.3f}% "
               f"(a quarter of the MAE)")
with right:
    bankroll = st.number_input("bankroll (USD)", min_value=0.0, value=1000.0, step=100.0)
    st.metric("suggested stake (quarter-Kelly, hard-capped)",
              f"${bankroll * stake_frac:,.0f}",
              f"{stake_frac*100:.1f}% of bankroll")
st.error(DISCLAIMER, icon="⚠️")

# ---- price chart with forecast -----------------------------------------------
st.subheader("Last 14 days + forecast")
price = get_recent_price()
chart = price[["time", "close"]].rename(columns={"close": "BTC close"}).set_index("time")
st.line_chart(chart, height=260)
st.caption(f"forecast for +24h: ${target:,.0f} (band ${lo:,.0f} - ${hi:,.0f})")

# ---- honest model card --------------------------------------------------------
st.subheader("Is this model actually better than nothing?")
lift = (base_mae - mae) / base_mae * 100 if base_mae else 0.0
verdict = ("the model has a thin real edge in magnitude."
           if lift > 0 else
           "**the random walk wins on magnitude** -- what survives is a thin "
           "directional edge, which is why the stake suggestion above is tiny "
           "and usually FLAT.")
st.markdown(
    f"- champion walk-forward **MAE {mae:.5f}** vs random-walk baseline "
    f"**{base_mae:.5f}** -> **{lift:+.2f}% lift**: {verdict}\n"
    f"- directional accuracy **{dir_acc*100:.1f}%** out-of-sample (coin flip = 50%).\n"
    f"- every number here is walk-forward: the model never saw the bars it is scored on."
)

try:
    scored = get_scored()
    eq, sharpe, hit = backtest(scored)
    st.subheader("Out-of-sample backtest (fees included)")
    st.line_chart(eq, height=260)
    st.caption(f"non-overlapping 24h positions, {FEE*1e4:.0f} bps per flip. "
               f"Sharpe {sharpe:.2f}, hit rate {hit*100:.1f}%. Past performance "
               f"predicts nothing.")
except Exception as e:
    st.info(f"backtest data not available yet: {e}")

try:
    lb = get_leaderboard()
    st.subheader("Autoresearch leaderboard")
    st.dataframe(lb[["config", "cv_mae", "baseline_mae", "mae_lift_pct",
                     "cv_dir_acc", "cv_r2"]], use_container_width=True, hide_index=True)
except Exception as e:
    st.info(f"leaderboard not available yet: {e}")

st.error(DISCLAIMER, icon="⚠️")
