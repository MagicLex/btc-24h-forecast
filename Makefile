# BTC 24h forecaster -- FTI pipelines on Hopsworks
# Feature (5 FGs) -> Training (walk-forward autoresearch) -> Inference (KServe + app)

HOURS ?= 62000

features:            ## ingest all five sources into their feature groups
	python3 pipelines/feature_pipeline.py --hours $(HOURS)

features-job:        ## same, as a Hopsworks job (hourly schedule candidate)
	hops job deploy btc-features pipelines/feature_pipeline.py \
		--env python-feature-pipeline --run --wait --overwrite

train-job:           ## walk-forward autoresearch + champion registration, as a job
	hops job deploy btc-autoresearch autoresearch/search.py \
		--env pandas-training-pipeline --run --wait --overwrite

serve:               ## deploy the champion as the btcforecaster KServe endpoint
	python3 serving/deploy.py

app:                 ## deploy the Streamlit front-end
	python3 app/deploy_app.py

smoke:               ## poke the endpoint
	python3 -c "import hopsworks; d=hopsworks.login().get_model_serving().get_deployment('btcforecaster'); print(d.predict(inputs=[{}]))"

help:
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/  --/'
.PHONY: features features-job train-job serve app smoke help
