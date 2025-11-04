# ============================================================
#
# Automates environment setup, model training, uncertainty scoring,
# and metric evaluation for all dataset–model combinations.
# Ensures reproducibility, consistency, and easy execution
# during experiments and defense demonstrations.
#
# Example:
#   make setup
#   make train-mnist-shufflenet
#   make score-mnist-shufflenet
#   make eval-mnist-shufflenet
#
# ============================================================

PY := .venv/bin/python

# ---------- ENVIRONMENT ----------
setup:
	python -m venv .venv && source .venv/bin/activate && pip install -U pip && pip install -r requirements.txt && pip install -e .

update:
	pip install -r requirements.txt

# ---------- TRAINING ----------
train-mnist-shufflenet:
	$(PY) -m src.cli.train --dataset mnist --model shufflenet_v2_0_5

train-mnist-mobilenet:
	$(PY) -m src.cli.train --dataset mnist --model mobilenet_v3_small

train-fashion-shufflenet:
	$(PY) -m src.cli.train --dataset fashion_mnist --model shufflenet_v2_0_5

train-fashion-mobilenet:
	$(PY) -m src.cli.train --dataset fashion_mnist --model mobilenet_v3_small

train-cifar-mobilenet:
	$(PY) -m src.cli.train --dataset cifar10 --model mobilenet_v3_small

train-cifar-efficientnet:
	$(PY) -m src.cli.train --dataset cifar10 --model efficientnet_v2_s

# ---------- SCORING ----------
score-mnist-shufflenet:
	$(PY) -m src.cli.score --dataset mnist --model shufflenet_v2_0_5

score-mnist-mobilenet:
	$(PY) -m src.cli.score --dataset mnist --model mobilenet_v3_small

score-fashion-shufflenet:
	$(PY) -m src.cli.score --dataset fashion_mnist --model shufflenet_v2_0_5

score-fashion-mobilenet:
	$(PY) -m src.cli.score --dataset fashion_mnist --model mobilenet_v3_small

score-cifar-mobilenet:
	$(PY) -m src.cli.score --dataset cifar10 --model mobilenet_v3_small

score-cifar-efficientnet:
	$(PY) -m src.cli.score --dataset cifar10 --model efficientnet_v2_s

# ---------- EVALUATION ----------
eval-mnist-shufflenet:
	$(PY) -m src.cli.evaluate --dataset mnist --model shufflenet_v2_0_5

eval-mnist-mobilenet:
	$(PY) -m src.cli.evaluate --dataset mnist --model mobilenet_v3_small

eval-fashion-shufflenet:
	$(PY) -m src.cli.evaluate --dataset fashion_mnist --model shufflenet_v2_0_5

eval-fashion-mobilenet:
	$(PY) -m src.cli.evaluate --dataset fashion_mnist --model mobilenet_v3_small

eval-cifar-mobilenet:
	$(PY) -m src.cli.evaluate --dataset cifar10 --model mobilenet_v3_small

eval-cifar-efficientnet:
	$(PY) -m src.cli.evaluate --dataset cifar10 --model efficientnet_v2_s

# ---------- UTILITIES ----------
lint:
	ruff check || true

clean:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} \;