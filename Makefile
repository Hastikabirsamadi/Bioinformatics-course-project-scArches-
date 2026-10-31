PY ?= .venv/bin/python

.PHONY: env run smoke aggregate
env:
	python3 -m venv .venv && .venv/bin/pip install -r requirements.lock
run:        ## full experiment: 4 variants x 3 seeds (use a GPU, see colab_run.ipynb)
	./run_all.sh
smoke:      ## tiny CPU test of the whole pipeline
	OUT=results_smoke/runs SEEDS="0" ./run_all.sh --epochs 3 --max_cells 3000
aggregate:
	$(PY) aggregate.py results
