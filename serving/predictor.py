"""KServe predictor: live 24h-ahead BTC return forecast.

At request time the endpoint pulls a trailing window of all five sources, rebuilds
the exact training feature vector with the same shared modules (bundled next to the
model, no skew), and returns the champion's forward-24h log-return prediction plus
the context the app needs to render a forecast.

    predict(inputs=[{}])  ->  [{"pred_return_24h": -0.004, "close": 60190.6,
                                "open_time": 1782928800000, "asof": "..."}]

Sources are cached for 5 minutes so a busy app does not hammer the upstreams; the
bar only changes hourly anyway.
"""
import os
import sys
import time

import joblib

_PATH = os.environ.get("MODEL_FILES_PATH", "/mnt/models")
sys.path.insert(0, _PATH)
from features import assemble_features, ALL_FEATURE_NAMES    # noqa: E402
import live_sources                                          # noqa: E402

CACHE_TTL_S = 300


class Predict:
    def __init__(self):
        self.model = joblib.load(os.path.join(_PATH, "model.joblib"))
        self._cached = None
        self._cached_at = 0.0

    def _forecast(self):
        now = time.monotonic()
        if self._cached is not None and now - self._cached_at < CACHE_TTL_S:
            return self._cached
        src = live_sources.pull_all()
        frame = assemble_features(src["btc"], src["eth"], src["funding"],
                                  src["onchain"], src["sentiment"])
        valid = frame.dropna(subset=ALL_FEATURE_NAMES)
        row = valid.iloc[[-1]]
        pred = float(self.model.predict(row[ALL_FEATURE_NAMES])[0])
        out = {
            "pred_return_24h": round(pred, 6),
            "close": float(row["close"].iloc[0]),
            "open_time": int(row["open_time"].iloc[0]),
            "asof": str(row["event_time"].iloc[0]),
        }
        self._cached, self._cached_at = out, now
        return out

    def predict(self, inputs):
        if isinstance(inputs, dict):
            inputs = inputs.get("instances") or inputs.get("inputs") or [{}]
        return [self._forecast() for _ in (inputs or [{}])]
