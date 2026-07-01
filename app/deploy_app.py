"""Deploy the forecaster front-end as a Hopsworks Streamlit app.

Run from a Hopsworks terminal where this repo lives under the project FUSE mount.
The app is a thin client of the btcforecaster endpoint plus feature-store reads
(scored + leaderboard FGs) -- no pickle, no pinned ML stack, stock app base env.
"""
from pathlib import Path

import hopsworks

APP_NAME = "btclive"
ENV_NAME = "python-app-pipeline"

# /hopsfs/<...project-relative...>/app/deploy_app.py -> project-relative path
rel = str(Path(__file__).resolve()).split("/hopsfs/", 1)[1]
APP_PATH = str(Path(rel).parent / "app.py")


def main():
    project = hopsworks.login()
    apps = project.get_app_api()
    print(f"app_path={APP_PATH} env={ENV_NAME}", flush=True)
    app = apps.get_app(APP_NAME)
    if app is None:
        app = apps.create_app(name=APP_NAME, app_path=APP_PATH,
                              environment=ENV_NAME, memory=2048, cores=1.0)
    app.run(await_serving=True)
    print("serving:", app.serving)
    print("URL:", app.get_url())


if __name__ == "__main__":
    main()
