# ADR-0002: Python core, language-neutral contracts

- **Status:** Accepted
- **Date:** 2026-10-22
- **Deciders:** Project maintainers
- **Related:** [architecture.md](../architecture.md), [schemas/](../../schemas/)

## Context

The project needs:

- a runtime that calls model providers and schedules graph execution;
- data tooling for traces, datasets, and evaluation;
- training recipes in the Python ML ecosystem (PyTorch, PEFT, datasets);
- a graph format that other tools, languages, and AI workflows can read and write;
- eventually, a browser UI.

A single-language implementation would be simpler, but choosing the wrong language for the
core would either exclude the ML ecosystem or lock the contracts to an implementation.

## Decision

1. **Python 3.12+ is the reference implementation language** for the compiler, runtime,
   CLI, evaluation harness, and training recipes.
2. **Contracts are language-neutral.** Graph, model manifest, envelope, and ticket shapes
   are defined as JSON Schema 2020-12 and YAML/JSON documents. The schemas are canonical for
   external consumers; Python models mirror them and a parity test enforces agreement.
3. **All model I/O goes through process or network boundaries.** Adapters are HTTP clients
   or local subprocess clients; there is no in-process assumption about where a model runs.
4. **The CLI (`mg`) is the first interface.** A library API is provided for Python users;
   an OpenAI-compatible local daemon is a later facade over the same runtime.
5. **A future UI is a separate TypeScript application** that consumes trace/IR JSON and
   the CLI or daemon API. It does not reimplement graph semantics.

Tooling: `uv` for environments, `pydantic` v2 for runtime models, `PyYAML` with duplicate
key detection, `httpx` for providers, `pytest` for tests, `ruff` for lint/format, `mypy`
for types, `jsonschema` for contract tests. A dependency is added only when the ticket
justifies it.

## Consequences

**Positive**

- Direct access to the ML ecosystem for evaluation and training.
- Graph authors can write plain YAML; tool authors can target JSON Schema in any language.
- The runtime can be embedded in notebooks and scripts, which matters for research.
- A UI can be built without forking the core.

**Negative**

- Python's packaging and dependency management remain a friction point for users.
- Performance of the scheduler is adequate but not exceptional; model calls dominate anyway.
- Two representations (Pydantic models and JSON Schema) can drift without the parity test.
- A future UI in another language will duplicate some presentation logic, though not
  semantics.

**Neutral**

- Rust or Go could replace hot paths later behind the same provider and CLI contracts.

## Alternatives considered

| Alternative | Why rejected |
| --- | --- |
| TypeScript core | Better UI synergy, weaker ML/training ecosystem, and local model tooling is more awkward. |
| Rust core | Excellent performance and distribution, but slows iteration and excludes most contributors and ML libraries. |
| Single JSON Schema-first codegen pipeline | Attractive in theory; codegen adds build complexity before the contracts are stable. Parity tests give most of the benefit now. |
| Notebook-only prototype | Fast to start, but not composable, testable, or shippable. |

## Follow-ups

- [T-0001](../../.scratch/tickets/T-0001-dev-environment.md) establishes the toolchain.
- [T-0002](../../.scratch/tickets/T-0002-contract-tests.md) adds JSON Schema/Pydantic parity
  tests and a fixture corpus.
- Revisit the daemon API shape after the CLI runtime is stable.
- Keep the Python API surface small; the CLI and documents are the stable contract.
