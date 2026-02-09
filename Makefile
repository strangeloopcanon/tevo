# Keep recipes portable across macOS + Ubuntu CI (GitHub runners don't include /bin/zsh by default).
SHELL := /bin/sh
PYTHON ?= python3
VENV := .venv
VENV_BIN := $(VENV)/bin
PIP := $(VENV_BIN)/pip
FMT := $(VENV_BIN)/black
LINT := $(VENV_BIN)/ruff
TYPECHECK := $(VENV_BIN)/mypy
PYTEST := $(VENV_BIN)/pytest
BANDIT := $(VENV_BIN)/bandit
DETECT_SECRETS := $(VENV_BIN)/detect-secrets
PIP_AUDIT := $(VENV_BIN)/pip-audit
APP := $(VENV_BIN)/python -m transformer_evolution_llm
AGENT_MODE ?= baseline

.PHONY: setup check format format-check lint type security test llm-live deps-audit all release clean

$(VENV):
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

setup: $(VENV)
	uv pip install --python $(VENV_BIN)/python -e ".[dev]"
	@echo "AGENT_MODE=$(AGENT_MODE)" > .env.agent_mode

format:
	$(FMT) src tests

format-check:
	$(FMT) --check src tests

lint:
	$(LINT) check src tests

type:
	$(TYPECHECK)

security:
	$(BANDIT) -q -r src
	$(DETECT_SECRETS) scan --baseline .secrets.baseline

check: format-check lint type security

test:
	$(PYTEST)

llm-live:
	$(VENV_BIN)/python -m transformer_evolution_llm.llm_live_stub

deps-audit:
	$(PIP_AUDIT)

all: check test llm-live deps-audit

release:
	@echo "Release automation is only available in production mode."
	@echo "Set AGENT_MODE=production and supply the release workflow to continue."

clean:
	rm -rf $(VENV) .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov dist build
