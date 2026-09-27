.PHONY: setup download run lint test check

setup:
	bash scripts/setup.sh

download:
	.venv/bin/python scripts/download_models.py flux-klein-4b

run:
	bash scripts/run.sh

lint:
	.venv/bin/ruff check .

test:
	.venv/bin/pytest

check: lint test
