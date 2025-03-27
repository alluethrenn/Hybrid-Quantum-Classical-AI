# Set Python environment (modify if using virtualenv/conda)
PYTHON=python

# Paths
CONFIG=configs/config.yaml
TRAIN_SCRIPT=src/quantum_ml/train.py
EVAL_SCRIPT=src/quantum_ml/evaluate.py

# Targets
.PHONY: train eval clean setup

# Run training
train:
	$(PYTHON) $(TRAIN_SCRIPT) --config $(CONFIG)

# Run evaluation
eval:
	$(PYTHON) $(EVAL_SCRIPT) --config $(CONFIG)

# Clean logs & models
clean:
	rm -rf results/models/* results/logs/*

# Setup environment (install dependencies)
setup:
	$(PYTHON) -m pip install -r requirements.txt
