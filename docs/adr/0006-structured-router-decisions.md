# ADR-0006: Structured router decisions with confidence and abstain

- **Status:** Accepted
- **Date:** 2026-10-22
- **Deciders:** Project maintainers
- **Related:** [protocol.md](../protocol.md), [evaluation.md](../evaluation.md), ADR-0005

## Context

A router can produce many kinds of output:

- free text naming a model ("this looks like a coding task, send it to the coder");
- a JSON object with a target;
- a probability distribution over routes;
- a classifier label.

Free text is easy to prompt but difficult to parse, audit, and evaluate. A bare target
without a confidence makes it impossible to decide when to abstain or fall back. A
probability distribution is ideal but not available from most chat models. A classifier is
efficient but restricts the router's input and expressive power.

Routing is also the highest-risk component: a wrong route can make an excellent specialist
useless. The router contract must make wrongness visible and recoverable.

## Decision

**Every router returns a structured decision with a target or abstention, a confidence, and
a reason.**

```json
{"decision": "route", "to": "coding", "confidence": 0.91, "reason": "...", "note": "..."}
{"decision": "abstain", "confidence": 0.42, "reason": "..."}
```

Requirements:

1. **Structured output.** JSON-only, validated against the decision shape. Invalid output
   gets one repair attempt, then a typed error.
2. **Confidence is required** and interpreted as a calibrated probability-like estimate.
3. **Abstention is a first-class outcome**, not an error. Policy decides whether it becomes
   `error`, `fallback`, or `retry`.
4. **A route must be in the menu.** The router cannot invent targets.
5. **Reason is recorded** (bounded length) for auditing and training.
6. **Optional note** lets the router pass a short hint to the child without replacing the
   user request.
7. **Routers are evaluated as routers** (oracle agreement, confusion, calibration, abstain
   rate), not only by end-to-end quality.
8. **Routers can be implemented by any adapter** that produces this shape, including a
   classifier or a fine-tuned SLM. The contract is the interface, not the model type.

## Consequences

**Positive**

- The runtime can enforce thresholds and fallbacks deterministically.
- Traces explain why a route was chosen; failures can be attributed to routing.
- Decisions become training labels for the flywheel.
- Confusion matrices and calibration reports are possible.
- Router implementations are swappable without changing graphs.

**Negative**

- Confidence from a chat model is often poorly calibrated; the project must measure and
  correct it, and cannot assume it.
- JSON-only output adds prompt overhead and fails more often on very small models.
- A mandatory reason consumes tokens and can be post-hoc rationalization.
- The menu format and decision schema must stay in sync with prompts and training data.

**Neutral**

- Logit-based routers could map probabilities into the same shape; the contract does not
  prevent them.

## Alternatives considered

| Alternative | Why rejected |
| --- | --- |
| Free-text routing with regex parsing | Brittle, hard to evaluate, no confidence, invites prompt injection. |
| Target-only JSON | Cannot abstain or fall back gracefully; no signal for calibration. |
| Full probability distribution over children | Ideal, but unavailable from most chat models and expensive to prompt. Could be an optional adapter later. |
| Classifier-only routers | Efficient but limits input to features; fine as an implementation behind the contract. |
| Router answers directly when unsure | Blurs the role and makes quality attribution harder; deferred behind `allow_direct_answer`. |

## Follow-ups

- [T-0008](../../.scratch/tickets/T-0008-router-executor.md) implements the decision parser,
  repair, threshold, and fallback policy.
- [T-0020](../../.scratch/tickets/T-0020-node-eval.md) implements calibration and routing
  metrics.
- [T-0024](../../.scratch/tickets/T-0024-router-training.md) trains routers to produce this
  contract.
- Revisit `allow_direct_answer` only if evaluation shows a real category of requests that no
  child should handle.
