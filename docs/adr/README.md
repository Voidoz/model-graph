# ADR index

Architectural decisions are recorded here once they are accepted. An ADR is not a design
doc; it captures a choice, the alternatives, and the consequences the project must live
with. A changed decision gets a new ADR that supersedes the old one; accepted ADRs are not
rewritten except for typos and link fixes.

Format: **Status**, **Date**, **Context**, **Decision**, **Consequences**, **Alternatives
considered**, **Follow-ups**.

| ADR | Title | Status | Date |
| --- | --- | --- | --- |
| [0001](0001-language-as-the-wire.md) | Language is the inter-node wire format | Accepted | 2026-10-22 |
| [0002](0002-python-core-neutral-contracts.md) | Python core, language-neutral contracts | Accepted | 2026-10-22 |
| [0003](0003-four-node-kinds.md) | Four node kinds, no arbitrary code nodes | Accepted | 2026-10-22 |
| [0004](0004-typed-artifacts.md) | Semantic artifact types on a text substrate | Accepted | 2026-10-22 |
| [0005](0005-compile-time-graph-runtime-routing.md) | Compile-time graph, runtime routing | Accepted | 2026-10-22 |
| [0006](0006-structured-router-decisions.md) | Structured router decisions with confidence and abstain | Accepted | 2026-10-22 |
| [0007](0007-scoped-blackboard-trace.md) | Scoped blackboard and content-addressed trace | Accepted | 2026-10-22 |
| [0008](0008-hard-budgets.md) | Hard budgets and loop guards by default | Accepted | 2026-10-22 |
| [0009](0009-local-first-provider-neutral.md) | Local-first, provider-neutral model binding | Accepted | 2026-10-22 |
| [0010](0010-evaluation-gated-contributions.md) | Evaluation-gated contributions and per-node ownership | Accepted | 2026-10-22 |
| [0011](0011-gpu-resident-backends.md) | GPU-resident execution as an optional backend | Accepted | 2026-10-22 |

## Proposed or open

None yet. Decisions that would change the architecture are tracked as questions in
[`../../.scratch/ideas/open-questions.md`](../../.scratch/ideas/open-questions.md) until
they are ready to become ADRs.

## Writing a new ADR

1. Copy the structure of an existing ADR; use the next number.
2. State the decision in one sentence in the title.
3. Include the alternatives you actually considered and why they lost.
4. State the negative consequences honestly; an ADR with no downside is not finished.
5. Add it to the table above and link it from the affected docs.
6. If it supersedes an ADR, set the old status to `Superseded by ADR-XXXX` and link forward.
