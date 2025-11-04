PY := .venv/bin/python

setup:
	python -m venv .venv && source .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

train-mnist:
	$(PY) -m src.cli.train

lint:
	ruff check || true

clean:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} \;
