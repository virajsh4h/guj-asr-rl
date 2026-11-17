PY ?= python

.PHONY: test report verify clean

test:
	$(PY) -m pytest -q

report:
	$(PY) -m gujasr report --out results/harness

verify:
	$(PY) -m gujasr verify --db results/harness.sqlite

clean:
	rm -rf .pytest_cache **/__pycache__ results/harness
