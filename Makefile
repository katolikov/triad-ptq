.PHONY: help install test smoke sweep_llm sweep_cnn plots all clean

help:
	@echo "Targets:"
	@echo "  install     uv sync (Python 3.11 + torch + transformers + ...)"
	@echo "  test        pytest"
	@echo "  smoke       SmolLM-135M end-to-end"
	@echo "  sweep_llm   FP32, RTN, AWQ-like, TRIAD on Tier-1 LLMs"
	@echo "  sweep_cnn   same on Tier-1 CNNs (ImageNetV2 matched-frequency, 5K)"
	@echo "  plots       regenerate tables + plots from results/tables/*.json"
	@echo "  all         install + test + smoke + sweep_llm + sweep_cnn + plots"

install:
	uv sync --no-dev
	uv add --dev pytest pytest-xdist tabulate

test:
	uv run pytest -q

smoke:
	uv run python experiments/01_calibrate_smollm.py

sweep_llm:
	uv run python experiments/10_compare_all_models.py --models smollm-135 smollm-360 tinyllama

sweep_cnn:
	uv run python experiments/20_compare_cnns.py \
	  --models mobilenet_v2 efficientnet_b0 mobilevit_s --n-eval 5000 --n-calib 64

plots:
	uv run python experiments/30_make_plots.py

all: install test smoke sweep_cnn sweep_llm plots
