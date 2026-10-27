# Contributing to Model Graph

Model Graph is a local-first Python project for composing small language models into a routed,
evaluated graph. Contributions are welcome, but a change to a graph, node, model, or prompt is a
behavioral change: it needs reproducible evidence, not just code that runs.

By participating, you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## License of contributions

The project is licensed under the **GNU General Public License, version 3 or later**
(`GPL-3.0-or-later`). See [LICENSE](LICENSE).

Contributions are accepted under the same license (inbound = outbound). There is no separate CLA.
Every commit must be signed off under the [Developer Certificate of Origin 1.1](https://developercertificate.org/):

```text
git commit -s -m "your change"
```

The sign-off certifies that you wrote the change or have the right to submit it under this
project's license. Do not contribute code, model weights, datasets, or prompts whose license is
incompatible with GPLv3 or whose provenance you cannot explain.

## Development setup

Requirements:

- Python **3.12+**
- [`uv`](https://docs.astral.sh/uv/)
- Optional, for live local models: [Ollama](https://ollama.com/) and the model tags listed in
  `registry/models/`
- Optional, for the GUI: a modern browser; no Node build is required

```bash
git clone https://github.com/Voidoz/model-graph.git
cd model-graph
uv sync
make check
```

Run the first offline example:

```bash
uv run mg validate graphs/hello-triage.graph.yaml
uv run mg run graphs/hello-triage.graph.yaml --provider mock
uv run mg ui            # http://127.0.0.1:8765/
```

The project keeps its generated runs, caches, and local virtual environment out of git. Do not
commit model weights, run traces, datasets, secrets, or `.env` files.

## Checks

Run these before opening a pull request:

```bash
make check          # format, lint, mypy, offline tests
make eval-gates     # offline node/routing/e2e/safety gates
make ui-check       # web tests and GUI smoke test
```

Useful individual targets:

```bash
make format         # ruff format + safe fixes
make lint           # ruff format --check + ruff check
make type           # mypy
make test           # pytest
make audit          # network-dependent dependency audit
make live-test      # opt-in Ollama test; requires MG_TEST_OLLAMA=1
```

`make check` must stay offline and deterministic. A test that needs a live model, network, or GPU
belongs in an opt-in marker or a separate target.

## What to contribute

- **Runtime and compiler:** deterministic execution, validation, budgets, tracing, replay,
  caching, concurrency, and error handling.
- **Graphs and examples:** a new starter graph must validate, have a documented use case, and
  state its hardware assumptions.
- **Models and adapters:** include a manifest with license/provenance; never vendor weights.
- **Evaluation:** suites, baselines, calibration, and honest reports. A baseline change is a
  reviewed change, not a side effect of an unrelated pull request.
- **Documentation and tutorials:** commands must be copy-pasteable and should run offline unless
  clearly marked live.
- **Security:** reader-side validation, escaping, redaction, and threat-model improvements. See
  [SECURITY.md](SECURITY.md).

## Behavioral changes need evidence

For a prompt, model, routing, topology, tool, or training change, attach:

1. the exact command and commit;
2. the dataset/eval suite and its hash;
3. a baseline and candidate report;
4. quality, validity, cost, and latency together, with uncertainty where available;
5. the slices or rare cases that could regress.

The local gate is:

```bash
uv run mg eval compare --baseline evals/baselines/<level>.json \
  --candidate <report.json>
```

`mg compare`/`mg eval compare` is not a substitute for a live quality claim. If you cannot
measure something on your hardware, say so and leave the claim out.

For a model swap or deployment change, use the revision workflow:

```bash
uv run mg graph diff <base> <candidate>
uv run mg change explain <change-id>
uv run mg inspect --html <run-dir> --graph <graph.yaml> --out /tmp/review.html \
  --change <change-id>
uv run mg promote --revision <revision> --dry-run
```

Do not mutate a live deployment directly or bypass missing evidence.

## Data and privacy

- Use synthetic prompts and traces in examples and tests.
- Do not submit real user data, personal data, credentials, or proprietary prompts.
- If you contribute a dataset or model card, record consent, license, source, and preprocessing.
- Secret scanners and redaction are fallible. Treat traces, artifacts, and cache entries as
  sensitive local data.

## Code style

- Keep the contracts in `schemas/` and the documentation in `docs/` in sync with behavior.
- Add tests for new behavior, error paths, and security boundaries.
- Prefer the standard library and existing dependencies. A new dependency needs a concrete reason
  and a ticket.
- Keep the mock path deterministic. Do not hide network or model calls in a unit test.
- The GUI is loopback-only and has no authentication. Do not add a remote-hosting assumption or
  inject untrusted content with `innerHTML`.

## Pull requests

- Keep each pull request focused; unrelated cleanup belongs in a separate change.
- Describe the user-visible behavior, the commands run, and the test results.
- Link the issue or design note when there is one.
- Expect review to ask for evidence, not just a passing test.
- Do not force-push a shared branch after review without saying why.

## Security reports

Do not open a public issue for a vulnerability. Follow [SECURITY.md](SECURITY.md).

## Questions

Open a GitHub discussion or issue with the smallest reproducible graph, command, and trace that
shows the problem. Remove secrets and real data first.
