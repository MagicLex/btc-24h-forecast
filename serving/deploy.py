"""Deploy the champion as an always-on KServe endpoint.

The serving artifact bundles model.joblib + every shared feature module +
live_sources.py, so the endpoint rebuilds the exact training feature vector from
live upstreams at request time. Poke it with anything, get the current 24h forecast.

Env: stock pandas-inference-pipeline first. The champion trains in the managed
pandas-training-pipeline job env; if the two ship different sklearn versions the
smoke predict will warn/fail -> clone and pin (tools/build_envs.py).
"""
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CHAMPION = "btc_fwd24"
SERVE_MODEL = "btc_fwd24_serving"
DEPLOYMENT = "btcforecaster"
SERVE_ENV = os.environ.get("SERVE_ENV", "pandas-inference-pipeline")
BUNDLE_MODULES = ["features.py", "btc_features.py", "funding_features.py",
                  "cross_features.py", "onchain_features.py", "sentiment_features.py"]
# /hopsfs/Users/<u>/<proj>/serving/deploy.py -> project-relative predictor path
_rel = __file__.split("/hopsfs/", 1)[1].rsplit("/", 1)[0] if "/hopsfs/" in __file__ else ""
PREDICTOR = f"/Projects/createnew/{_rel}/predictor.py" if _rel \
    else os.path.join(ROOT, "serving", "predictor.py")


def _schema():
    import pandas as pd
    from hsml.schema import Schema
    from hsml.model_schema import ModelSchema
    inp = Schema(pd.DataFrame({"trigger": [1]}))
    out = Schema(pd.DataFrame({"pred_return_24h": [0.0], "close": [60000.0]}))
    return ModelSchema(input_schema=inp, output_schema=out)


def _bundle(proj, version=None):
    mr = proj.get_model_registry()
    if version is None:
        version = max(m.version for m in mr.get_models(CHAMPION))
    champ = mr.get_model(CHAMPION, version=version)
    src = champ.download()
    d = os.path.join(ROOT, "serving", "_artifact")
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)
    shutil.copy(os.path.join(src, "model.joblib"), os.path.join(d, "model.joblib"))
    for mod in BUNDLE_MODULES:
        shutil.copy(os.path.join(ROOT, mod), os.path.join(d, mod))
    shutil.copy(os.path.join(ROOT, "serving", "live_sources.py"),
                os.path.join(d, "live_sources.py"))
    return d, version


def deploy_champion(proj):
    mr = proj.get_model_registry()
    d, version = _bundle(proj)
    sm = mr.python.create_model(SERVE_MODEL, model_schema=_schema(),
                                description=f"serving bundle of {CHAMPION} v{version}")
    sm.save(d)
    dep = sm.deploy(name=DEPLOYMENT, script_file=PREDICTOR, environment=SERVE_ENV)
    dep.start(await_running=600)
    return dep


if __name__ == "__main__":
    import hopsworks
    p = hopsworks.login()
    dep = deploy_champion(p)
    print("deployment:", dep.name, "| running:", dep.is_running())
    try:
        print("smoke:", dep.predict(inputs=[{}]))
    except Exception as e:
        print("smoke predict pending:", e)
