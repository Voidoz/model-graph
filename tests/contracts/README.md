# Contract tests and fixtures

The four JSON Schemas in `schemas/` are canonical. These tests make the contracts
executable and keep the Pydantic mirrors in `src/modelgraph/contracts/` from drifting.

```text
uv run pytest tests/contracts -q
uv run pytest tests/contracts -q -k "invalid or parity"
make check
```

Everything runs offline and uses only the repository's existing dependencies.

## Fixture layout

```text
tests/fixtures/contracts/
  graph/
    valid/*.graph.yaml
    invalid/*.graph.yaml        # one sibling *.expected.json each
    warnings/*.graph.yaml       # schema-valid graphs that produce semantic warnings
  manifest/
    valid/*.manifest.yaml
    invalid/*.manifest.yaml
  envelope/
    valid/*.envelope.json
    invalid/*.envelope.json
  ticket/
    valid/*.ticket.md           # YAML frontmatter, as used by .scratch/tickets
    invalid/*.ticket.md
  models/                       # small registry for cross-reference checks
  graphs/                       # inner/outer subgraph corpus
  duplicate-keys/*.yaml
```

Naming conventions:

- A valid fixture passes its JSON Schema and its Pydantic model, and is semantically
  coherent where it is also used by compiler tests.
- An invalid fixture fails its JSON Schema. Its `.expected.json` sidecar records the exact
  JSON path and a stable message fragment:
  ```json
  {"path": "spec.provider", "message": "is not one of"}
  ```
  Tests assert both, so an invalid fixture cannot silently fail for the wrong reason.
- Every schema keeps at least three valid and three invalid fixtures. Add a fixture and its
  expectation together; `test_invalid_fixtures_have_expectations` rejects stale sidecars.
- Ticket fixtures are Markdown files because the ticket contract is YAML frontmatter. The
  repository copy of `.scratch/tickets/_template.md` is checked to exist and is explicitly
  excluded from the `T-*.md` validation glob.

## Parity rules

`test_parity.py` maps each canonical schema object to its Pydantic model and compares the
constraints named in T-0002:

- required fields;
- enum and `const` values (including array items);
- string patterns (including nullable fields, array items, and map values).

Lengths, numeric bounds, and defaults are intentionally outside this comparison. The
`json_schema_extra` enum on `PathEntry.kind` mirrors `NodeKind`; it is paired with a
validator so the Pydantic model rejects the same values even though the runtime constructs
the field from `node.kind.value`.

Four tests derive a deliberately drifted model (missing required field, widened enum,
dropped pattern, dropped `const`) and assert the comparison reports it. Never weaken a
canonical schema to make a fixture pass.

## Semantic checks outside the contract suite

- `test_cross_refs.py` only resolves model and subgraph references through the fixture
  `Registry`. Full graph semantics (run by `validate_graph`) live in `tests/compiler/`.
- `test_v15.py` loads the conflicting-exit-type fixture, confirms it is schema-valid, and
  asserts the compiler's semantic validator emits the documented `MG-1016` warning.
- `test_loader_duplicates.py` asserts duplicate YAML keys raise `MG-1001` naming the key and
  line, including nested maps and sequence items.
