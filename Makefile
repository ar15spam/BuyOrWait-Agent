PYTHON := $(shell [ -x venv/bin/python ] && echo venv/bin/python || echo python3)

.PHONY: test check run score smoke run-llm

test:
	$(PYTHON) -m pytest tests -q

check:
	$(PYTHON) -m py_compile code/*.py code/extract/*.py code/evaluation/*.py tests/*.py

run:
	$(PYTHON) code/main.py --requests dataset/requests.csv --out dataset/output.csv

score:
	$(PYTHON) code/main.py --requests dataset/sample_requests.csv --out /tmp/pred_sample.csv
	$(PYTHON) -m code.evaluation.score --pred /tmp/pred_sample.csv

smoke:
	$(PYTHON) -m code.evaluation.smoke

run-llm:
	$(PYTHON) code/main.py --requests dataset/requests.csv --out dataset/output.csv --messages --explain
