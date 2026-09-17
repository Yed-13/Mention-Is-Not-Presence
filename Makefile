.PHONY: test verify expansion-data baselines diagnostics summarize-expansion all

PYTHON ?= python3

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -p 'test_*.py'

verify:
	PYTHONPATH=src $(PYTHON) src/verify_artifact.py

expansion-data:
	PYTHONPATH=src $(PYTHON) src/prepare_expansion_experiments.py --root .

baselines:
	PYTHONPATH=src $(PYTHON) src/lexical_baselines.py --train data/synthetic/train.jsonl --test data/synthetic/test.jsonl --method majority --predictions results/raw/majority_predictions.jsonl --metrics results/metrics/majority_metrics.json
	PYTHONPATH=src $(PYTHON) src/lexical_baselines.py --train data/synthetic/train.jsonl --test data/synthetic/test.jsonl --method char_ngram_nb --predictions results/raw/char_ngram_nb_predictions.jsonl --metrics results/metrics/char_ngram_nb_metrics.json

diagnostics:
	PYTHONPATH=src $(PYTHON) src/analyze_diagnostics.py --gold data/synthetic/test.jsonl --prediction base_direct results/raw/direct_predictions.jsonl --prediction base_guideline results/raw/guideline_predictions.jsonl --prediction qlora_direct results/raw/qlora_s17_direct_predictions.jsonl --prediction qlora_guideline results/raw/qlora_s17_guideline_predictions.jsonl --out results/metrics/template_diagnostics.json

summarize-expansion:
	PYTHONPATH=src $(PYTHON) src/summarize_expansion.py

all: test verify
