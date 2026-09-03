run:
	uv run python -m src

install:
	uv sync

debug:
	uv run python -m pdb -m src

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

lint:
	uv run flake8 src/
	uv run mypy src/ --config-file=pyproject.toml --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs \
		--check-untyped-defs

lint-strict:
	uv run flake8 src/
	uv run mypy src/ --config-file=pyproject.toml --strict

.PHONY: install run debug clean lint lint-strict