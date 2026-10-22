# Vision

## The problem with one model for everything

A frontier model is trained to be acceptable at everything. That is a remarkable
engineering achievement and an awkward product: most users need a narrow slice of its
capability, but they pay for all of it — in training compute, in inference cost, in
latency, and in the inability to inspect or change the parts that matter to them.

Three costs follow from the monolith:

1. **Training cost is centralized.** Improving a model generally means a larger run on
   more data and more GPUs. A domain expert with a valuable dataset and one workstation
   cannot meaningfully improve the system.
2. **Inference cost is paid wholesale.** A deployment that only needs code reasoning
   still carries the weights, context budget, and serving cost of a generalist.
3. **Improvement is entangled.** A change that helps one domain can regress another,
   and there is no clean boundary where a specialist can be swapped or evaluated.

Mixture-of-experts models address part of this inside a single training run, but the
experts are still trained together, served together, and owned by one organization.

## The proposition

Turn a model system into a **graph of small, independently trainable nodes connected by
natural language**:

- **Deciders (routers)** inspect a request and choose which child gets it.
- **Specialists (workers)** do the work and write a named result.
- **Joiners** combine results from multiple branches.
- **Subgraphs** package a whole graph as a node, so composition can nest.

Because the only thing crossing an edge is language, any model that can follow a prompt
can participate. A router does not need logits from a worker; a worker does not need the
router's weights. Each node can be trained, evaluated, versioned, and replaced on its
own. A user assembles the system the way they would assemble a team.

This is the "sectors of the brain" intuition made practical. The analogy is useful for
composition and routing. It is not literal, and the design says so plainly (§The brain
analogy, below).

## Why this can reduce cost

| Lever | Mechanism |
| --- | --- |
| Cheaper training | Each node trains on a narrow data slice. LoRA/QLoRA on consumer hardware can improve a 1–8B specialist. Contributors do not need a data center. |
| Cheaper inference | Only the branches a request needs are executed; unused subgraphs are pruned and their weights never loaded. Specialists can be small because the task is narrow. |
| Better per-task quality | A 3B model fine-tuned on one domain can beat a generalist on that domain, and routing can compose several such wins. |
| Reuse | A well-evaluated node can be used in many graphs. The community accumulates components, not just checkpoints. |
| Replaceable parts | Regressions are localized. A bad specialist can be rolled back without rolling back the system. |

These are hypotheses, not guarantees. The costs that can *increase* — routing overhead,
handoff token loss, extra hops, duplicated context — are treated as first-class in the
architecture and evaluation plan.

## What success looks like

A user should be able to:

1. Write a graph in YAML with three levels: a domain decider, a sub-domain decider, and
   two specialists. Run it locally against models they already have.
2. Add a specialist without changing the runtime: register a model manifest, add a node
   and an edge, rerun the validator, run the node's eval.
3. Disable a capability (no GPU, no network, no non-commercial licenses) and get a
   pruned graph that still runs, with an explicit report of what was removed.
4. Inspect every routing decision and every prompt, replay a run byte-for-byte against
   pinned model versions, and diff two graphs.
5. Train a specialist on a single 24GB GPU, submit it with an eval report, and see
   whether it improves a target node without regressing the end-to-end suite.
6. Compile a graph to a fused or distilled deployment target and compare it with a
   quality-matched dense LLM on TTFT, throughput, cost, and VRAM.
7. Swap a node's model or adapter, see the impact and eval delta, promote it atomically, and
   roll back without editing the running revision.

The project-level north-star metric is **quality per dollar at matched task difficulty**.
Secondary metrics: task success, TTFT, p95 latency, throughput, routing accuracy, prune ratio,
and the number of independently contributed nodes that pass evaluation gates. Performance
claims are always paired with a quality-matched dense baseline ([performance.md](performance.md)).

## The core hypotheses

These are deliberately falsifiable. The roadmap schedules cheap tests of each before
large investments.

| ID | Hypothesis | Evidence that would support it | What would falsify it |
| --- | --- | --- | --- |
| H1 | A routed graph of small specialists can match a much larger generalist on a narrow suite. | A 1–8B expert graph reaches parity within 2 points at ≤40% cost of the baseline. | Routing + handoff loses more than 5 points versus the best node run alone. |
| H2 | Routing can be learned accurately enough from system traces. | ≥90% agreement with oracle routing on separable domains. | Confusion between domains stays high, or abstain rate makes the router useless. |
| H3 | Capability-aware pruning produces real savings without surprising quality loss. | ≥30% cost reduction for a fixed deployment profile at parity. | Pruned graphs fail on tasks users actually send, not just benchmark tasks. |
| H4 | Independent contributors can improve nodes on consumer hardware. | A 24GB-GPU LoRA improvement of ≥5 points on a node suite that survives end-to-end gating. | Gains vanish out-of-distribution, or eval gating cannot detect regressions. |
| H5 | A GPU-resident backend removes most non-model overhead, and a shared-base adapter graph approaches single-model latency. | B1 transition p95 <15ms excluding prefill; B2 <5ms and TTFT within 1.25× of a single-model baseline; zero weight reloads. | VRAM or prefill dominates, the fused base constrains quality, or the measured gain over portable execution is negligible. |
| H6 | A compiled deployment target can match a quality-matched dense LLM on latency and throughput for decomposable tasks. | Compiled fused TTFT ≤1.25× and throughput ≥0.9× dense; distilled target ≥1.0× throughput; cost ≤1.0× at matched quality. | The graph loses after prefix/route compilation and batching, or distillation cannot hold quality on rare routes. |

**Kill criteria.** If, across three different domains, oracle routing beats the best
single model by less than 5 quality points at equal cost, the cost-reduction thesis is
weak for those domains. The project should then pivot toward composition as an
inspection, ownership, and evaluation tool — still useful, but a different claim.

## The brain analogy, precisely

Useful similarities:

- **Segregation.** Different regions handle different kinds of processing.
- **Routing.** A request is sent to the region that can handle it, not broadcast to all.
- **Working memory.** A shared, limited context that nodes read from and write to.
- **Hierarchy.** Broad categorization precedes fine specialization.
- **Local learning.** Components can change without retraining the whole system.

Important differences, which bound the design:

- **There is no shared latent representation.** Neurons exchange high-dimensional
  signals; nodes exchange text. The channel is low-bandwidth and lossy. This is the
  fundamental constraint of the architecture, not a detail.
- **There is no global workspace or recurrent loop** in v0. The envelope is a snapshot,
  not a recurrent mental state.
- **There is no plasticity.** Weights do not adapt during a run. Learning is offline.
- **Credit assignment is black-box.** A wrong final answer may have come from a router,
  a worker, a handoff, or the composition. The trace format exists to make this
  debuggable; end-to-end backpropagation does not exist.

The design response is **semantic compression**: each handoff carries a compact,
structured artifact (a plan, a patch, a report) rather than a raw transcript, and each
node receives only the view it needs. Language is the medium; disciplined artifacts are
how we limit its loss.

## Who this is for

| Persona | What they need | What Model Graph gives them |
| --- | --- | --- |
| Local hobbyist | Run a capable assistant on their own hardware without a data center. | Small nodes, local adapters, pruning, plain-language graph definitions. |
| Domain expert | Encode expertise without training a foundation model. | A node contract, a fine-tuning kit, and a node-level eval suite. |
| Application developer | Control cost and behavior of an AI feature. | Budgeted execution, replayable traces, explicit routing, fallbacks. |
| Researcher | Study composition, routing, and distillation. | A neutral graph format, trace corpus, and reproducible evaluation harness. |
| Component author | Distribute a model others can trust and use. | Manifests, model cards, eval reports, versioned nodes, registry. |

## What this is not

- **Not a new foundation model.** It composes existing ones.
- **Not an agent framework with arbitrary tool loops.** v0 graphs are acyclic and
  text-only. Tools and code execution are deliberately deferred (see `Q-004`).
- **Not a latent-space architecture.** No hidden-state sharing or cross-model KV-cache
  reuse, and no cross-model gradient flow. Prefix caching within one model/adapter set is an
  optimization, not latent communication. See [ADR-0001](adr/0001-language-as-the-wire.md).
- **Not one forward pass.** GPU-resident backends remove transport, reload, and
  re-tokenization overhead, but every node still performs its own prefill. "Runs like a single
  model" is a measured performance target, not a structural identity
  ([gpu-execution.md](gpu-execution.md)).
- **Not a guarantee of lower cost.** Routing can be wrong and handoffs can be expensive.
  The evaluation plan exists to measure when composition wins and when it does not.
- **Not a marketplace (yet).** The registry is local-first; payments and hosted serving
  are out of scope for v0.
- **Not a replacement for a frontier model on open-ended work.** The intended sweet spot
  is decomposable, repetitive, evaluable work.

## Optionality

If the thesis holds, the same primitives support several futures: a package manager for
capabilities, a training flywheel where traces become routers and specialists, a
hardware-aware runtime that places nodes on the machines that can serve them, and a
community benchmark where the best node for a task is discoverable and auditable. None
of those are required by the core design; all of them become possible if the graph,
manifest, trace, and evaluation contracts are stable.
