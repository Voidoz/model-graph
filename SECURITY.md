# Security policy

## Supported versions

Security fixes are made on the current `main` branch and the most recent release. Older
development snapshots are not supported.

## Reporting a vulnerability

Please do **not** open a public issue for a suspected vulnerability.

Use GitHub's private vulnerability reporting / Security Advisories for
[Voidoz/model-graph](https://github.com/Voidoz/model-graph/security/advisories/new). Include:

- the affected version or commit;
- the smallest graph, command, and trace that reproduces the issue;
- the impact and any known preconditions;
- whether the issue affects the local GUI, runtime, providers, cache, or a model manifest;
- a suggested fix or mitigation, if you have one.

Do not include real credentials, personal data, or third-party confidential content. If a secret
is already exposed in a trace or artifact, rotate it and report the exposure without copying the
secret into the report.

We will acknowledge a report, investigate, and coordinate disclosure. There is no bug bounty.

## Security model and known limitations

The threat model is documented in [`docs/security.md`](docs/security.md). In brief:

- The GUI is local-only, binds to loopback, and has **no authentication**. Do not expose it to a
  network.
- Delimiters and precedence statements reduce prompt-injection risk; they do not make a model
  immune.
- Runtime redaction covers persisted traces, envelopes, manifests, and errors. Artifact blobs,
  evaluation reports, and user-supplied files are not rewritten.
- The response cache is local and unencrypted. Prompts/API keys are not stored in entries, but
  cached responses are sensitive local data.
- There is no sandbox for tools or code execution in v0.
- Model manifests and revisions are not cryptographically signed.
- Dependency auditing is a separate `make audit`/CI job; the offline test gate does not have
  network access.

If a limitation is unacceptable for your deployment, do not deploy it that way; open an issue to
discuss the design rather than silently disabling a control.
