UV ?= uv
PYTEST ?= $(UV) run pytest
RUFF ?= $(UV) run ruff
MYPY ?= $(UV) run mypy
EVAL_REPORTS ?= runs/evals/ci
BASELINES ?= evals/baselines

.PHONY: sync format lint type test live-test ui-check check clean eval-candidates eval-gates baselines nightly

sync:
	$(UV) sync

format:
	$(RUFF) format src tests
	$(RUFF) check --fix src tests

lint:
	$(RUFF) format --check src tests
	$(RUFF) check src tests

type:
	$(MYPY) src

test:
	$(PYTEST) -q

live-test:
	MG_TEST_OLLAMA=1 $(PYTEST) tests/live -q

ui-check:
	$(PYTEST) tests/web -q
	$(UV) run python tests/web/smoke_ui.py

check: lint type test

eval-candidates:
	rm -rf $(EVAL_REPORTS)
	mkdir -p $(EVAL_REPORTS)
	$(UV) run mg eval node --suite evals/mock/js-specialist --provider mock \
		--reports-dir $(EVAL_REPORTS)/runs --json > $(EVAL_REPORTS)/node.json
	$(UV) run mg eval routing --suite evals/mock/triage --provider mock \
		--reports-dir $(EVAL_REPORTS)/runs --json > $(EVAL_REPORTS)/routing.json
	$(UV) run mg eval e2e --suite evals/e2e/composed-assistant \
		--graph examples/composed-assistant.graph.yaml --provider mock --profile local-8gb \
		--reports-dir $(EVAL_REPORTS)/runs \
		--case js-debounce --case js-types --case general-rust \
		--case story-lighthouse --case story-misrouted \
		--json > $(EVAL_REPORTS)/e2e.json
	$(UV) run python evals/safety/run.py --reports-dir $(EVAL_REPORTS)/runs \
		--json > $(EVAL_REPORTS)/safety.json

# Run all offline comparisons, collect every result, then fail once at the end. That way one
# failing level does not hide the others from the PR summary. Safety is a measured hard gate:
# a violating candidate fails even when its quality metrics improve.
eval-gates: eval-candidates
	@mkdir -p $(EVAL_REPORTS)/comparisons
	@status=0; \
	for level in node routing e2e safety; do \
		echo "== $$level =="; \
		$(UV) run mg eval compare --baseline $(BASELINES)/$$level.json \
			--candidate $(EVAL_REPORTS)/$$level.json \
			--reports-dir $(EVAL_REPORTS)/comparisons \
			--json > $(EVAL_REPORTS)/compare-$$level.json || status=1; \
	done; \
	exit $$status

baselines:
	$(UV) run python evals/baselines/update.py

nightly:
	MG_EVAL_LIVE=1 $(UV) run mg eval node --suite evals/mock/js-specialist --provider ollama \
		--live --reports-dir runs/evals/nightly --fail-on-gate
	MG_EVAL_LIVE=1 $(UV) run mg eval routing --suite evals/mock/triage --provider ollama \
		--live --reports-dir runs/evals/nightly --fail-on-gate
	MG_EVAL_LIVE=1 $(UV) run mg eval e2e --suite evals/e2e/composed-assistant \
		--graph examples/composed-assistant.graph.yaml --provider ollama --live \
		--reports-dir runs/evals/nightly --fail-on-gate

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage dist build
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
