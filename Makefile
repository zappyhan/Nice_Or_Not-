# Convenience targets. `make demo` is the 60-second path from clone to report.
PYTHON ?= python3
export PYTHONPATH := src

.PHONY: install test train train-yelp inspect score serve demo docker clean

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest tests -q

train:                ## Train on the bundled simulator (no download required)
	$(PYTHON) -m reputation.pipeline.train --source synthetic --n-businesses 120 --months 36

inspect:              ## Preflight the Yelp download: which cities, how big
	$(PYTHON) -m reputation.data.inspect --data-dir data

train-yelp:           ## Train on the real Yelp Open Dataset in ./data
	$(PYTHON) -m reputation.pipeline.train --source yelp --city Philadelphia

score:
	$(PYTHON) -m reputation.pipeline.score --top 10

serve:
	$(PYTHON) -m uvicorn reputation.api.main:app --reload --port 8000

demo: train score

docker:
	docker build -t reputation-early-warning:latest .

clean:
	rm -rf artifacts/*.joblib artifacts/*.json .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
