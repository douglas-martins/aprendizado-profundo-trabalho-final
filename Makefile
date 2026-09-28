UV := uv
RUN := $(UV) run
PYTHON := $(RUN) python
PYTEST := $(RUN) pytest
RUFF := $(RUN) ruff
MUTMUT := $(RUN) mutmut
TARGET_DATA := /Users/ghostface/Projects/mestrado/methane-detection/data

.PHONY: sync data-link test coverage lint format docstring-coverage mutation \
	train train-recovery evaluate evaluate-scenes precompute-cache confirm-raw \
	build-r2-manifest data-quality calibrate-thresholds idle-power \
	figure-architecture figure-eda figure-class-balance figure-data-quality \
	figure-pr-curves figure-training-curves figure-threshold-effect \
	figure-efficiency figure-qualitative figures clean-generated

TRAIN_ARGS ?= dataset=starcop_mini
RECOVERY_ARGS ?= architecture=E2 dataset=starcop_raw tier=raw-full
EVALUATE_ARGS ?= architecture=E1 dataset=starcop_mini tier=mini
EVALUATE_SCENES_ARGS ?= architecture=E2 checkpoint_tier=raw-full eval_tier=raw-full
PRECOMPUTE_CACHE_ARGS ?= dataset=starcop_mini splits=train,val,test
CONFIRM_RAW_ARGS ?= dataset=starcop_raw
POWER_SECONDS ?= 30

sync:
	$(UV) sync

data-link:
	rm -rf data/starcop_mini data/starcop_raw
	ln -s $(TARGET_DATA)/starcop_mini data/starcop_mini
	ln -s $(TARGET_DATA)/starcop_raw data/starcop_raw
	rm -rf data/processed/starcop_mini data/processed/starcop_raw
	ln -s $(TARGET_DATA)/processed/starcop_mini data/processed/starcop_mini
	ln -s $(TARGET_DATA)/processed/starcop_raw data/processed/starcop_raw

test:
	$(PYTEST) -v

coverage:
	$(PYTEST) --cov=src --cov-report=term-missing --cov-report=xml --junitxml=junit.xml

lint:
	$(RUFF) check src conftest.py
	$(RUFF) format --check src conftest.py

format:
	$(RUFF) check --fix src conftest.py
	$(RUFF) format src conftest.py

docstring-coverage:
	$(RUN) interrogate -v src

mutation:
	rm -rf mutants .mutmut-cache
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $(MUTMUT) run

train:
	$(PYTHON) -m training.train $(TRAIN_ARGS)

train-recovery:
	$(PYTHON) -m training.train_with_recovery $(RECOVERY_ARGS)

evaluate:
	$(PYTHON) -m evaluation.evaluate $(EVALUATE_ARGS)

evaluate-scenes:
	$(PYTHON) -m evaluation.evaluate_scenes $(EVALUATE_SCENES_ARGS)

precompute-cache:
	$(PYTHON) -m data.precompute_patch_cache $(PRECOMPUTE_CACHE_ARGS)

confirm-raw:
	$(PYTHON) -m data.confirm_raw $(CONFIRM_RAW_ARGS)

build-r2-manifest:
	$(PYTHON) -m data.build_r2_manifest

data-quality:
	$(PYTHON) -m data.data_quality

calibrate-thresholds:
	$(PYTHON) -m evaluation.threshold_calibration

idle-power:
	$(PYTHON) -m utils.power_meter $(POWER_SECONDS)

figure-architecture:
	$(PYTHON) -m visualization.architecture_diagram

figure-eda:
	$(PYTHON) -m visualization.eda

figure-class-balance:
	$(PYTHON) -m visualization.class_balance

figure-data-quality: data-quality

figure-pr-curves:
	$(PYTHON) -m visualization.pr_curve_plots

figure-training-curves:
	$(PYTHON) -m visualization.training_curves

figure-threshold-effect:
	$(PYTHON) -m visualization.threshold_effect_plot

figure-efficiency:
	$(PYTHON) -m visualization.efficiency_plot

figure-qualitative:
	$(PYTHON) -m visualization.qualitative_predictions

figures: figure-architecture figure-eda figure-class-balance figure-data-quality \
	figure-pr-curves figure-training-curves figure-threshold-effect figure-efficiency \
	figure-qualitative

clean-generated:
	find figures checkpoints logs patch_cache run_state scene_scores -mindepth 1 ! -name .gitkeep -exec rm -rf {} +
