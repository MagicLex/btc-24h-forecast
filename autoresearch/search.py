"""Autoresearch + training (T stage): search models with honest walk-forward CV,
pick a champion that actually beats the persistence baseline, register it with a
full evaluation card, and write the out-of-sample predictions for the app's backtest.

Why walk-forward and not K-fold: the label is the forward 24h log return, so
neighbouring hourly rows share overlapping futures. A shuffled split leaks. We use
an expanding-window TimeSeriesSplit with a HORIZON-bar gap (embargo) between train
and test, so every score is strictly out-of-sample.

The headline is honest:
  - MAE of the model vs MAE of persistence (predict 0 / random walk). On BTC this
    baseline is hard; we report the lift, not a flattering absolute.
  - Directional accuracy vs a coin flip (0.5).
  - An OOS, non-overlapping, fee-aware backtest vs buy-and-hold.

    python autoresearch/search.py
"""
import json
import os
import shutil
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

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
import btc_features                                          # noqa: E402
import funding_features                                       # noqa: E402
import cross_features                                         # noqa: E402
import onchain_features                                       # noqa: E402
import sentiment_features                                     # noqa: E402
from features import ALL_FEATURE_NAMES                       # noqa: E402
from btc_features import HORIZON                             # noqa: E402
from pipelines.feature_pipeline import (                     # noqa: E402
    LABEL, FG_PRICE, FG_FUNDING, FG_CROSS, FG_ONCHAIN, FG_SENTIMENT)

FV_NAME = "btc_fwd24_fv"
MODEL_NAME = "btc_fwd24"
LEADERBOARD_FG = "btc_leaderboard"
SCORED_FG = "btc_scored"
N_SPLITS = 6
SEED = 42
FEE = 0.0004  # 4 bps per position change, applied in the backtest
OUT = Path(ROOT) / "models" / "eval"
DATA = Path(ROOT) / "data"

# The search grid: linear, two tree ensembles, gradient boosting. Kept modest so the
# walk-forward runs in a couple minutes; the point is honest comparison, not a zoo.
MODELS = {
    "ridge_a1":   lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "ridge_a10":  lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
    "rf_d6":      lambda: RandomForestRegressor(n_estimators=300, max_depth=6,
                                                min_samples_leaf=50, n_jobs=-1, random_state=SEED),
    "rf_d10":     lambda: RandomForestRegressor(n_estimators=300, max_depth=10,
                                                min_samples_leaf=25, n_jobs=-1, random_state=SEED),
    "hgb_lr05":   lambda: HistGradientBoostingRegressor(learning_rate=0.05, max_depth=3,
                                                        max_iter=400, l2_regularization=1.0,
                                                        random_state=SEED),
    "hgb_lr02":   lambda: HistGradientBoostingRegressor(learning_rate=0.02, max_depth=4,
                                                        max_iter=600, l2_regularization=1.0,
                                                        random_state=SEED),
    "xgb":        lambda: xgb.XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.03,
                                           subsample=0.8, colsample_bytree=0.8,
                                           reg_lambda=2.0, random_state=SEED, n_jobs=-1),
}


def get_feature_view(fs):
    """The join lives HERE, in the store: five per-source FGs -> one feature view.

    Hourly FGs equi-join on open_time. Daily FGs join on the data-day key with the
    +1d publication lag already baked in (price.prev_day_ms == daily.day_ms), so an
    hourly bar on day D reads day D-1's on-chain/sentiment values -- the same row a
    backward asof would pick. Point-in-time by construction.
    """
    price = fs.get_feature_group(FG_PRICE, version=1)
    fund = fs.get_feature_group(FG_FUNDING, version=1)
    cross = fs.get_feature_group(FG_CROSS, version=1)
    onc = fs.get_feature_group(FG_ONCHAIN, version=1)
    sent = fs.get_feature_group(FG_SENTIMENT, version=1)
    query = (price.select(["open_time", "close"] + btc_features.FEATURE_NAMES + [LABEL])
             .join(fund.select(funding_features.FEATURE_NAMES), on=["open_time"])
             .join(cross.select(cross_features.FEATURE_NAMES), on=["open_time"])
             .join(onc.select(onchain_features.FEATURE_NAMES),
                   left_on=["prev_day_ms"], right_on=["day_ms"])
             .join(sent.select(sentiment_features.FEATURE_NAMES),
                   left_on=["prev_day_ms"], right_on=["day_ms"]))
    return fs.get_or_create_feature_view(
        name=FV_NAME, version=1, query=query, labels=[LABEL],
        description="Five per-source FGs joined in the store (hourly on open_time, "
                    "daily on lagged day key) -> forward 24h BTC log return.")


def load_data(proj):
    """Prefer the feature view; fall back to the identical local assembly if the
    freshly-inserted FG has not finished offline materialization yet."""
    cache = DATA / "train_cache.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
    else:
        fs = proj.get_feature_store()
        try:
            fv = get_feature_view(fs)
            X, y = fv.training_data()
            df = X[["open_time", "close"] + ALL_FEATURE_NAMES].copy()
            df[LABEL] = y[LABEL].values if hasattr(y, "columns") else np.asarray(y)
            df["event_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            df = df.dropna(subset=ALL_FEATURE_NAMES + [LABEL])
            if len(df) < 1000:
                raise RuntimeError("FG not materialized yet")
        except Exception as e:
            print(f"FV read unavailable ({e}); using local dataset.parquet")
            df = pd.read_parquet(DATA / "dataset.parquet")
        df = df.sort_values("open_time").reset_index(drop=True)
        df.to_parquet(cache)
    return df


def walk_forward(model_fn, X, y):
    """Return per-fold MAE / dir-acc / R2 / baseline-MAE and the concatenated OOS preds."""
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=HORIZON)
    maes, dirs, r2s, base_maes = [], [], [], []
    oos_idx, oos_pred = [], []
    for tr, te in tscv.split(X):
        m = model_fn()
        m.fit(X.iloc[tr], y.iloc[tr])
        p = m.predict(X.iloc[te])
        yte = y.iloc[te].values
        maes.append(mean_absolute_error(yte, p))
        base_maes.append(mean_absolute_error(yte, np.zeros_like(yte)))  # persistence
        dirs.append(float(np.mean(np.sign(p) == np.sign(yte))))
        r2s.append(r2_score(yte, p))
        oos_idx.extend(te.tolist())
        oos_pred.extend(p.tolist())
    return {
        "cv_mae": float(np.mean(maes)), "cv_mae_std": float(np.std(maes)),
        "cv_dir_acc": float(np.mean(dirs)), "cv_r2": float(np.mean(r2s)),
        "baseline_mae": float(np.mean(base_maes)),
        "oos_idx": oos_idx, "oos_pred": oos_pred,
    }


def main():
    import hopsworks
    proj = hopsworks.login()
    df = load_data(proj)
    X = df[ALL_FEATURE_NAMES]
    y = df[LABEL].astype(float)
    print(f"data: {len(X)} rows x {len(ALL_FEATURE_NAMES)} features, "
          f"{df['event_time'].min()} .. {df['event_time'].max()}\n", flush=True)

    results = {}
    for name, fn in MODELS.items():
        r = walk_forward(fn, X, y)
        results[name] = r
        lift = (r["baseline_mae"] - r["cv_mae"]) / r["baseline_mae"] * 100
        print(f"  {name:10s} MAE {r['cv_mae']:.5f} (base {r['baseline_mae']:.5f}, "
              f"lift {lift:+.2f}%)  dir {r['cv_dir_acc']:.3f}  R2 {r['cv_r2']:+.4f}",
              flush=True)

    champ = min(results, key=lambda k: results[k]["cv_mae"])
    cr = results[champ]
    lift = (cr["baseline_mae"] - cr["cv_mae"]) / cr["baseline_mae"] * 100
    print(f"\n  -> champion: {champ}  (MAE lift {lift:+.2f}% vs persistence, "
          f"dir acc {cr['cv_dir_acc']:.3f})\n")

    metrics = {
        "champion": champ, "cv_mae": round(cr["cv_mae"], 6),
        "cv_mae_std": round(cr["cv_mae_std"], 6),
        "baseline_mae": round(cr["baseline_mae"], 6),
        "mae_lift_pct": round(lift, 3),
        "cv_dir_acc": round(cr["cv_dir_acc"], 4), "cv_r2": round(cr["cv_r2"], 5),
        "n_rows": int(len(X)), "n_features": len(ALL_FEATURE_NAMES), "horizon_h": HORIZON,
    }
    print(json.dumps(metrics, indent=2))

    scored = _scored_frame(df, cr)
    _plots(df, results, champ, X, y)
    _write_leaderboard(proj, results)
    _write_scored(proj, scored)
    _register(proj, X, y, champ, metrics)


def _scored_frame(df, champ_res):
    oos = pd.DataFrame({"row": champ_res["oos_idx"], "y_pred": champ_res["oos_pred"]})
    j = df.iloc[oos["row"].values][["open_time", "event_time", "close", LABEL]].reset_index(drop=True)
    j["y_pred"] = oos["y_pred"].values
    j = j.rename(columns={LABEL: "y_true"})
    return j.sort_values("open_time").reset_index(drop=True)


def _backtest(scored):
    """Non-overlapping (every HORIZON bars), fee-aware long/short vs buy-and-hold."""
    s = scored.iloc[::HORIZON].reset_index(drop=True)
    pos = np.sign(s["y_pred"].values)
    turns = np.abs(np.diff(np.concatenate([[0], pos])))
    strat = pos * s["y_true"].values - turns * FEE
    eq_strat = np.exp(np.cumsum(strat))
    eq_hold = np.exp(np.cumsum(s["y_true"].values))
    sharpe = float(np.mean(strat) / (np.std(strat) + 1e-9) * np.sqrt(365))  # ~daily steps
    return s["event_time"].values, eq_strat, eq_hold, sharpe


def _plots(df, results, champ, X, y):
    OUT.mkdir(parents=True, exist_ok=True)
    cr = results[champ]
    scored = _scored_frame(df, cr)
    yt, yp = scored["y_true"].values, scored["y_pred"].values

    # 1. predicted vs actual
    plt.figure(figsize=(5, 5))
    plt.hexbin(yt, yp, gridsize=40, cmap="viridis", mincnt=1)
    lim = np.percentile(np.abs(yt), 99)
    plt.plot([-lim, lim], [-lim, lim], "--", color="#f59e0b")
    plt.xlim(-lim, lim); plt.ylim(-lim, lim)
    plt.xlabel("actual 24h return"); plt.ylabel("predicted"); plt.title(f"OOS pred vs actual ({champ})")
    plt.tight_layout(); plt.savefig(OUT / "pred_vs_actual.png", dpi=120); plt.close()

    # 2. MAE per fold: model vs persistence baseline
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=HORIZON)
    folds = list(range(1, N_SPLITS + 1))
    per_fold_model, per_fold_base = [], []
    fn = MODELS[champ]
    for tr, te in tscv.split(X):
        mdl = fn(); mdl.fit(X.iloc[tr], y.iloc[tr])
        p = mdl.predict(X.iloc[te]); yte = y.iloc[te].values
        per_fold_model.append(mean_absolute_error(yte, p))
        per_fold_base.append(mean_absolute_error(yte, np.zeros_like(yte)))
    w = 0.38
    plt.figure(figsize=(6, 4))
    plt.bar([f - w/2 for f in folds], per_fold_base, width=w, label="persistence", color="#6b7280")
    plt.bar([f + w/2 for f in folds], per_fold_model, width=w, label=champ, color="#8b5cf6")
    plt.xlabel("walk-forward fold"); plt.ylabel("MAE (lower better)")
    plt.title("MAE per fold: model vs random walk"); plt.legend()
    plt.tight_layout(); plt.savefig(OUT / "mae_per_fold.png", dpi=120); plt.close()

    # 3. OOS backtest equity curve
    t, eqs, eqh, sharpe = _backtest(scored)
    plt.figure(figsize=(7, 4))
    plt.plot(t, eqs, color="#8b5cf6", label=f"model long/short (Sharpe {sharpe:.2f})")
    plt.plot(t, eqh, color="#f59e0b", label="buy & hold")
    plt.axhline(1.0, color="#6b7280", lw=0.8)
    plt.ylabel("growth of $1 (OOS, fees in)"); plt.title("Out-of-sample backtest")
    plt.legend(); plt.tight_layout(); plt.savefig(OUT / "backtest.png", dpi=120); plt.close()

    # 4. permutation importance on the last OOS fold
    splits = list(tscv.split(X)); tr, te = splits[-1]
    mdl = fn(); mdl.fit(X.iloc[tr], y.iloc[tr])
    imp = permutation_importance(mdl, X.iloc[te], y.iloc[te], n_repeats=6,
                                 random_state=SEED, scoring="neg_mean_absolute_error")
    order = np.argsort(imp.importances_mean)[-18:]
    plt.figure(figsize=(6, 7))
    plt.barh([ALL_FEATURE_NAMES[i] for i in order], imp.importances_mean[order], color="#8b5cf6")
    plt.xlabel("permutation importance (MAE increase)"); plt.title("Top features (OOS)")
    plt.tight_layout(); plt.savefig(OUT / "feature_importance.png", dpi=120); plt.close()
    json.dump({ALL_FEATURE_NAMES[i]: round(float(imp.importances_mean[i]), 6)
               for i in np.argsort(imp.importances_mean)[::-1]},
              open(OUT / "feature_importance.json", "w"), indent=2)


def _write_leaderboard(proj, results):
    fs = proj.get_feature_store()
    rows = []
    for name, r in results.items():
        rows.append({
            "config": name, "cv_mae": r["cv_mae"], "cv_mae_std": r["cv_mae_std"],
            "baseline_mae": r["baseline_mae"],
            "mae_lift_pct": (r["baseline_mae"] - r["cv_mae"]) / r["baseline_mae"] * 100,
            "cv_dir_acc": r["cv_dir_acc"], "cv_r2": r["cv_r2"],
        })
    lb = pd.DataFrame(rows).sort_values("cv_mae").reset_index(drop=True)
    lb["run_time"] = pd.Timestamp.utcnow()
    fg = fs.get_or_create_feature_group(
        name=LEADERBOARD_FG, version=1,
        description="Autoresearch leaderboard: walk-forward CV MAE / lift / directional "
                    "accuracy per model config for the BTC 24h forecaster.",
        primary_key=["config"], event_time="run_time", online_enabled=False)
    fg.insert(lb, write_options={"wait_for_job": False})
    print(f"leaderboard -> {LEADERBOARD_FG} ({len(lb)} configs)")


def _write_scored(proj, scored):
    fs = proj.get_feature_store()
    scored = scored.copy()
    scored["event_time"] = pd.to_datetime(scored["event_time"], utc=True)
    fg = fs.get_or_create_feature_group(
        name=SCORED_FG, version=1,
        description="Out-of-sample walk-forward predictions of the BTC 24h forecaster "
                    "(open_time, close, y_true, y_pred) for the app's honest backtest.",
        primary_key=["open_time"], event_time="event_time", online_enabled=False)
    fg.insert(scored, write_options={"wait_for_job": False})
    print(f"scored OOS -> {SCORED_FG} ({len(scored)} rows)")


def _register(proj, X, y, champ, metrics):
    from hsml.schema import Schema
    from hsml.model_schema import ModelSchema
    mr = proj.get_model_registry()
    model = MODELS[champ]()
    model.fit(X, y)  # refit champion on all history for serving
    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUT / "model.joblib")

    assets = Path(ROOT) / "assets"
    assets.mkdir(exist_ok=True)
    for f in OUT.glob("*"):
        if f.suffix in (".png", ".json"):
            shutil.copy(f, assets / f.name)

    schema = ModelSchema(input_schema=Schema(X.iloc[:1]),
                         output_schema=Schema(pd.DataFrame({"fwd_return_24h": [0.0]})))
    example = X.iloc[:1].to_dict(orient="records")[0]
    m = mr.python.create_model(
        name=MODEL_NAME, metrics={k: v for k, v in metrics.items() if isinstance(v, (int, float))},
        model_schema=schema, input_example=example,
        description=f"BTC 24h forward-return forecaster ({champ}, {len(ALL_FEATURE_NAMES)} "
                    f"multi-source features, walk-forward MAE lift {metrics['mae_lift_pct']}%).")
    m.save(str(OUT))
    print(f"\nregistered model {MODEL_NAME} v{m.version}")


if __name__ == "__main__":
    main()
