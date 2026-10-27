# Security

This document is the threat model, the enforced baseline, and the release checklist for
Model Graph. It is intentionally narrower than a security guarantee: the runtime reduces
known classes of accidental and adversarial failure, and it states plainly what it does not
do.

The protocol rules live in [protocol.md §11](protocol.md#11-security-rules). Execution
budgets and persistence are described in [execution.md](execution.md). The implementation is
in [`src/modelgraph/security.py`](../src/modelgraph/security.py),
[`src/modelgraph/runtime/renderer.py`](../src/modelgraph/runtime/renderer.py), and
[`src/modelgraph/runtime/trace.py`](../src/modelgraph/runtime/trace.py).

## 1. Assets and trust boundaries

| Asset | Why it matters | Where it lives |
| --- | --- | --- |
| Request text, artifacts, routing notes | Unstructured text that can carry instructions, secrets, or PII | Envelope, rendered prompts, traces |
| Graph YAML | Node instructions, descriptions, menu text, and templates are **code-equivalent** | Graph source, compiled IR |
| Model manifests | Provider, endpoint, model ID, license, resource claims | Registry |
| Run profiles | Network ceiling, provider/license allowlists, budget overrides | Profile files |
| Traces, manifests, envelopes | Debugging evidence and future training data; can quote all of the above | `runs/<run-id>/` |
| Artifact blobs | The requested output; may legitimately contain user data or secrets | `runs/<run-id>/artifacts/` |
| Credentials | Provider keys and keychain references | Environment/keychain only, never manifests |
| Dependencies and weights | Supply chain | `uv.lock`, manifest metadata |

Trust boundaries:

1. **Graph author → runtime.** Graph files are trusted code-equivalent input only after
   review. A hostile graph can change node instructions, routing menus, artifact contracts,
   and model references even though it cannot execute Python through the YAML loader.
2. **User/artifact content → model.** Content is always data. Delimiters and precedence
   reduce injection risk; they do not make a model immune.
3. **Model output → runtime.** Output is parsed into artifacts or router decisions. Router
   targets are validated against the compiled menu; artifact names against `writes`. Output
   is not trusted to obey the protocol.
4. **Runtime → local disk.** Persistence is where accidental leakage becomes durable. The
   trace, envelope, and manifest are redacted by default; artifact blobs are not rewritten.
5. **Registry/profile → run.** Manifests cannot execute code or smuggle credentials.
   Profiles can filter providers, licenses, and network access, but the local GUI and CLI
   have no authentication.

## 2. Threat model

### 2.1 Prompt injection via request, artifact, routing note, or future retrieval

**Threat.** User text, upstream artifact content, a router note, or (later) retrieved
documents contain instructions such as "ignore your system prompt", role-play framing, fake
delimiters, or a forged route/artifact block. A small or weakly aligned model may comply.

**Controls**

- Untrusted text is never interpolated into a system message. System messages contain only
  graph-author text (node instruction, strategy, contract) and renderer boilerplate.
- Untrusted blocks are explicitly delimited and labelled `trust="untrusted"`.
- Every router, worker, and joiner system message states the precedence rule: system
  instructions and the compiled route policy override anything inside untrusted content.
  The joiner system message also states that the strategy is fixed by the graph.
- Protocol control markers inside content are neutralized (`<<<MG:` and the
  request/artifact/routing-note/routes/route tags are escaped), so content cannot close a
  block early or forge an artifact block. Ordinary prose and code are unchanged.
- Routers choose only from the compiled menu. A target named in content, or emitted by the
  model, that is not in the menu is a protocol error (`MG-2101`), never a route. The trace
  records the failed decision, not a fabricated edge.
- Router notes are capped (1,000 characters per note, 4,000 total) and passed as delimited
  hints, never as the task. A note cannot replace the request.
- Untrusted blocks are size-capped (request 32,000 characters, each artifact 32,000, all
  artifacts 64,000). Truncation is visible (`[[truncated: ...]]`) and recorded in
  `RenderedPrompt.truncated` and as `render.truncated` trace events.

**Limitations**

- Delimiters and precedence statements are mitigation, not immunity. A model can still
  follow injected text; test with the actual target model, including small models.
- There is no output classifier, instruction-hierarchy enforcement inside the model, or
  sandbox. A node that is asked to produce code can still produce dangerous code.
- Retrieval is not implemented in v0. When it is added, retrieved content must use the same
  untrusted path; do not add it as system text.
- A hostile **graph author** can write malicious node instructions directly. That is why
  graphs are treated as code-equivalent and reviewed; no renderer can defend against a
  trusted system message that is itself malicious.

### 2.2 Untrusted graphs and model manifests

**Threat.** A graph from an untrusted source changes system instructions, routes to an
attacker-favoured model, exfiltrates context, or points a "local" provider at a remote host.
A model manifest smuggles a credential, a command, or a network destination.

**Enforced at load time**

- YAML/JSON uses a safe loader with duplicate-key rejection; Python object tags are
  rejected. Graph and manifest schemas use `additionalProperties: false`, so unknown fields
  (for example `command`) are validation errors.
- Registry loading applies a manifest trust policy (`modelgraph.security`):
  - credential-shaped values are rejected; author e-mail addresses are not treated as
    credentials;
  - keys that describe executable behavior (`command`, `script`, `shell`, `entrypoint`,
    `postinstall`, `preinstall`, `subprocess`, …) are rejected;
  - `spec.endpoint` must be `http`/`https`, must not embed credentials, and must be loopback
    when `resources.network_access` is `loopback` (and may not be set when it is `none`);
  - the mock provider may not configure an endpoint;
  - `license.weights_uri` is restricted to `hf`, `https`, `http`, `file`, `ollama`, or `s3`;
    `license.url` must be `http`/`https`.
- Profiles declare `providers`, `licenses`, and a `network` ceiling. Pruning removes models
  outside those allowlists or above the network/resource/cost ceiling. A profile with a
  missing allowlist is still accepted for compatibility, but `mg run` prints a warning and
  the absence is visible in the run manifest.

**Limitations**

- **There is no sandbox.** A reviewed graph is trusted; an unreviewed graph is code-equivalent.
  Review instructions, model references, menus, and endpoints before running it.
- **There is no signing.** Manifests and weights are not cryptographically verified
  (deferred to the registry work, `T-0026`). A manifest hash pins content, not authorship.
- The trust policy stops concrete configuration attacks. It cannot prove a model is benign
  or that a license claim is true.
- The runtime never downloads weights from `weights_uri`; the URI is metadata. A future
  downloader must add scheme/hash/signature checks before use.

### 2.3 Secret and PII leakage

**Threat.** A request, model response, router reason, provider error, or manifest field
contains an API key, token, private key, e-mail address, or home path. It is then persisted
in a trace, manifest, envelope, cache, dataset, or error message, or copied into a log.

**Controls**

- The default patterns are conservative and shared by trace persistence, dataset export, and
  the scanner: private keys, OpenAI-style keys, GitHub tokens, AWS access keys, bearer
  tokens, e-mail addresses, and home paths.
- Trace JSONL is redacted **on write**; in-memory events are not mutated. The run manifest
  redacts `request.task`, router `path[].reason`, and `error`; the envelope is recursively
  redacted; `RunOutcome.error` is redacted before it is returned or printed by `mg run`.
- Structured log fields are scrubbed with a content-shaped key rule **and** the same default
  secret/PII pattern set before a JSONL or console line is written, so a secret echoed into a
  field such as `reason` or `note` does not reach the log.
- Redaction markers keep the pattern name and a truncated SHA-256 digest, never the matched
  text. `runs/<run-id>/redaction.json` reports pattern names, counts, field paths, and
  digests only.
- `uv run python scripts/scan_secrets.py runs/` scans a tree and reports file, line, pattern,
  and digest without printing matched text. It exits 1 on findings. Credential patterns run
  by default; `--include-pii` adds the noisier e-mail/home-path patterns.
- `mg data export` applies a configurable redaction pass before writing datasets, with its
  own report. See [training.md](training.md).

**Limitations**

- Redaction is **pattern-based**. Novel secret formats, encoded secrets, names, phone
  numbers, and other PII are not detected. It is a safety net, not a DLP product.
- **Artifact blobs are not rewritten.** They are content-addressed run data; rewriting them
  would invalidate hashes and replay. Treat the entire run directory as sensitive and protect
  it with filesystem permissions/backups.
- Hidden models can copy a secret from an untrusted block into an output. The trace redacts
  the copy, but the artifact blob still contains it.
- Redaction can **break trace replay**: a redacted request/response no longer reproduces the
  original artifact hash. That is an accepted trade-off for sensitive runs; replay reports a
  mismatch rather than pretending the run is identical.
- Evaluation reports (`runs/evals/...`) persist prompts and responses as evaluation evidence.
  They are not redacted by the runtime hook and should be treated as sensitive.
- The response cache follows the same default pattern set: a response that matches a
  secret/PII pattern is not cached at all, and the cache stores response text plus key
  metadata but never request messages, prompts, or credentials. It is local and per-user;
  it is not a shared location and has no encryption at rest. See
  [execution.md §7](execution.md#7-caching).
- Credentials belong in environment variables or a keychain. Manifests and graph files are
  rejected if they contain credential-shaped values, but logs from external libraries are
  outside the runtime's control.

### 2.4 Budget and denial-of-service abuse

**Threat.** A graph, profile, provider, or scripted response causes unbounded calls,
recursive subgraphs, huge prompts, or runaway fan-out, exhausting tokens, money, memory, or
wall-clock time.

**Controls**

- `max_steps`, `max_depth`, `max_total_tokens`, `max_cost_usd`, and `max_wall_seconds` are
  hard limits checked before calls and charged after. A subgraph shares the parent's budget
  and checks `depth + 1` before entry (`MG-4006`).
- Transport failures retry at most twice (three attempts total). Invalid router decisions and
  malformed artifacts get **one** repair attempt. A single executor has a hard cap of eight
  provider calls per node as defense in depth, and the router's configured
  `routing.max_retries` is now respected (0 means no low-confidence retry).
- A `selection: single` router activates exactly one menu entry. `top_k`/`all` routers can
  amplify work, so the scheduler reserves the estimated first-call cost of the whole ready
  batch atomically and refuses it before any branch starts if it does not fit
  (`MG-4002`/`MG-4003`). In-flight branch calls are bounded by `max_concurrency` (default
  4); the step, token, cost, and wall budgets are shared and lock-protected.
- Untrusted content is size-capped (see §2.1) before it becomes prompt tokens.
- Profile overrides tighten by default. `--allow-budget-override` can raise a bucket above
  the graph default, but never above the schema-aligned hard ceilings
  (`max_steps` 64, `max_depth` 8, `max_total_tokens` 10,000,000, `max_wall_seconds` 86,400).
- A pre-flight estimate vetoes a call that cannot fit the remaining token budget (`MG-4002`)
  before the provider is invoked.

**Limitations**

- Wall-clock cancellation is cooperative. An adapter that cannot cancel a request can
  overrun the wall-clock limit until the call returns; the runtime fails immediately after.
- A provider can under-report usage. The runtime charges what is reported and records the
  provider metadata; it cannot audit the vendor.
- Fan-out pre-flight estimates the first call of each branch, not the branch's entire
  downstream cost. A subgraph branch has no direct call, so its inner scheduler reserves
  its own entry; the outer batch cannot see inside it. Later calls are still vetoed one at
  a time by the hard budget; a provider that reports more usage than estimated can overrun
  the reservation before the next check. Join/wall timeouts are cooperative: an in-flight
  provider call is allowed to return before the run fails.
- A branch is not started speculatively. If a later branch cannot fit, the scheduler has
  still spent the earlier branches' budget unless the whole batch was reserved together;
  the ready-batch reservation covers the fan-out's first calls, which is the amplification
  the veto is designed to prevent.
- The CLI can still be invoked with an operator-supplied huge `--max-tokens`; that is an
  explicit local operator action, not remote input.

### 2.5 Supply chain

**Threat.** A compromised dependency, a malicious model manifest, a poisoned weights source,
or a hostile provider endpoint.

**Controls**

- Runtime dependencies are pinned in `uv.lock`; CI installs with `uv sync --locked`.
- `make audit` exports the locked runtime requirements and runs `pip-audit` (via `uvx`, so
  the auditor is not a project dependency). It is **not** part of `make check`; the default
  check stays offline and deterministic.
- CI runs the audit in a separate `dependency-audit` job. A failure blocks merge/release.
- Manifest trust rules reject executable fields, credentials, unexpected endpoint schemes,
  and local-provider endpoints that contradict their declared network access.
- Provider adapters normalize errors and never log API keys; manifests reference credentials
  by environment/keychain, not value.
- Weights are referenced by manifest and hash, never committed to git.

**Limitations**

- Dependencies are audited against known advisories at CI time; zero-day or unpublished
  vulnerabilities are not detected.
- There is no SBOM/provenance attestation, no manifest/weight signing, and no registry
  signature verification yet (`T-0026`).
- A model's behaviour and license claims are not verified; evaluation measures quality, not
  safety or legal terms.

## 3. Failure policy for dependency audit

`make audit` (and the CI `dependency-audit` job) fails the release when `pip-audit` reports a
vulnerable runtime dependency. The response is:

1. Determine whether the vulnerable package/version is reachable in the supported execution
   paths.
2. Upgrade or pin to a fixed version, run `uv lock`/`uv sync`, and rerun `make check` and
   `make audit`.
3. If no fix exists, a temporary waiver is allowed only with (a) a linked issue describing
   reachability/impact, (b) an expiry date, and (c) the narrowest `--ignore-vuln` entry in the
   `audit` target or CI step. Waivers are reviewable code, not a silent allowlist.
4. Record the decision in the release notes/ticket. Do not disable the job or mark it
   non-blocking.

The audit is network-dependent by design and never runs as part of the offline `make check`.

## 4. Release security checklist

Run these from a clean checkout before tagging a public release. Check each item or record
the exception in the release notes.

- [ ] `make check` passes (ruff, mypy, full offline test suite).
- [ ] `uv run pytest tests/security -q` passes.
- [ ] `make eval-gates` passes, including the non-waivable safety gate.
- [ ] `uv run python scripts/scan_secrets.py runs/ examples/ registry/ profiles/` reports
      clean (add `--include-pii` when the release ships run data), or every finding is
      understood and removed/rotated. Historical local evidence under an ignored `runs/`
      directory is not a release artifact; scan the run directories being shipped.
- [ ] `make audit` passes, or a documented waiver with issue and expiry exists (§3).
- [ ] `uv run mg registry verify` passes for every versioned contribution: schema-valid
      manifest, complete card, passing eval report with matching evidence/graph/suite hashes,
      weights hash when present, license policy, and a current deterministic index.
- [ ] No credentials, API keys, private keys, tokens, or real user data are in the repo,
      examples, profiles, manifests, tests, or committed run data.
- [ ] Every graph and model manifest distributed with the release is reviewed as
      code-equivalent: instructions, routes, model refs, endpoints, licenses, and weights
      URIs.
- [ ] Release profiles set `providers`, `licenses`, and `network`; `no-network` still fails
      closed where it should.
- [ ] The GUI is documented and deployed as loopback-only (no auth) and not exposed.
- [ ] Known limitations (§5) are present in the release notes or linked security doc.
- [ ] The threat model and protocol security rules still match the implementation; update
      them in the same change as any control.
- [ ] Offline tests remain offline: no default test requires a model server, network, or
      real credentials.

## 5. Known limitations (explicit)

1. **No sandbox.** There is no tool/code-execution node in v0 and no process/filesystem
   isolation for model output. Do not run generated code.
2. **No signing.** Manifests, graphs, and weights are not cryptographically signed or
   verified.
3. **No immunity to prompt injection.** Delimiters, precedence, marker neutralization, and
   menu validation reduce risk; a model may still comply with injected instructions.
4. **No PII detection engine.** Redaction is a conservative pattern set, not a classifier.
5. **Artifacts are not redacted in place.** Run directories are sensitive data.
6. **No authentication or authorization.** The CLI and local GUI trust the local user; the
   GUI must stay on loopback.
7. **No encryption at rest or key management.** Traces, artifacts, and datasets rely on
   filesystem and operator controls.
8. **Replay is not guaranteed after redaction.** A sensitive run may no longer reproduce
   byte-identical artifacts; this is reported as a replay mismatch.
9. **No penetration test or bug bounty.** Report suspected vulnerabilities privately to the
   maintainers; do not open a public issue with a live credential.

## 6. Related work

- Prompt and router rules: [protocol.md §3, §5, §11](protocol.md#11-security-rules)
- Budgets, traces, replay: [execution.md](execution.md)
- Dataset export redaction and consent: [training.md](training.md)
- Profiles and pruning: [profiles.md](profiles.md)
- Open questions: `Q-001` (license), `Q-004` (tools/code execution), `Q-010` (data
  privacy/export), `Q-003` (hosted trust model)
- Tracked work: `T-0026` (registry and verification), `T-0029` (this hardening),
  `T-0030` (observability), `T-0016` (cache)
