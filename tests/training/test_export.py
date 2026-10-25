"""Offline tests for trace-to-dataset export.

All fixtures are synthetic mock runs created in ``tmp_path``. No real user data, network, or
provider is used.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from modelgraph.cli import app
from modelgraph.compiler import compile_graph
from modelgraph.data import (
    DatasetExportError,
    export_dataset,
    load_dataset,
    validate_dataset_example,
    validate_dataset_manifest,
)
from modelgraph.providers.mock import MockProvider, MockScenario
from modelgraph.runtime.engine import run_graph
from modelgraph.util import sha256_text

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
TRIAGE_GRAPH = EXAMPLES / "hello-triage.graph.yaml"
JS_SUITE = ROOT / "evals" / "mock" / "js-specialist"
CONSENT = "synthetic operator consent for local-training fixture"
SECRET_TASK = (
    "Fix /home/oscar/private/demo.ts using sk-proj-abcdefghijklmnop and email alice@example.com."
)
SECRET_RESPONSE = (
    "export const fixed = true; // sk-proj-abcdefghijklmnop\n"
    "// contact alice@example.com or use ghp_0123456789abcdefghijklmnopqrstuvwx\n"
)
SECRET_VALUES = (
    "sk-proj-abcdefghijklmnop",
    "alice@example.com",
    "ghp_0123456789abcdefghijklmnopqrstuvwx",
    "/home/oscar/private",
)
runner = CliRunner()


def _compile() -> object:
    compiled = compile_graph(TRIAGE_GRAPH, strict_models=True)
    assert compiled.ok, compiled.errors
    assert compiled.graph is not None
    return compiled.graph


def _make_run(
    tmp_path: Path,
    task: str,
    *,
    route: str = "coding",
    response: str = "export const answer = 42;\n",
    failure_node: str | None = None,
) -> Path:
    decision = json.dumps(
        {
            "decision": "route",
            "to": route,
            "confidence": 0.92,
            "reason": "synthetic routing decision",
        }
    )
    scenario = MockScenario(
        responses={"triage": [decision], route: [response]},
        failures={failure_node: "MG-3001"} if failure_node else {},
    )
    # This suite deliberately exercises the dataset exporter's own redaction pass, so the
    # fixture runs opt out of the runtime's default pre-persistence trace redaction. Real
    # runs keep the runtime default; see docs/security.md.
    outcome = run_graph(
        _compile(),
        task,
        lambda model: MockProvider(scenario, max_context=model.spec.context_window),
        runs_dir=tmp_path / "runs",
        redact_persisted=False,
    )
    assert outcome.run_dir is not None
    return outcome.run_dir


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    return path


def _dataset_text(path: Path) -> str:
    return "\n".join(
        item.read_text(encoding="utf-8") for item in sorted(path.rglob("*")) if item.is_file()
    )


def test_export_produces_schema_valid_jsonl_and_manifest(tmp_path: Path) -> None:
    run = _make_run(tmp_path, SECRET_TASK, response=SECRET_RESPONSE)
    out = tmp_path / "dataset"
    result = export_dataset(
        [run],
        out,
        consent=CONSENT,
        created_at="2026-10-25T00:00:00Z",
    )

    assert result.examples == 2
    assert result.counts == {"train": 2, "validation": 0, "test": 0}
    bundle = load_dataset(out)
    assert validate_dataset_manifest(bundle.manifest) == []
    assert len(bundle.examples) == 2
    for example in bundle.examples:
        assert validate_dataset_example(example) == []
        assert example["consent"]["recorded"] is True
        assert example["label"]["outcome"] == "unknown"
        assert example["label"]["source"] == "unknown"

    kinds = {example["kind"] for example in bundle.examples}
    assert kinds == {"node", "routing"}
    node = next(example for example in bundle.examples if example["kind"] == "node")
    routing = next(example for example in bundle.examples if example["kind"] == "routing")
    assert node["source"]["node"] == "coding"
    assert node["input"]["rendered"] is True
    assert len(node["input"]["messages"]) == 2
    assert node["output"]["finish_reason"] == "stop"
    assert node["output"]["artifacts"][0]["name"] == "answer"
    assert routing["source"]["node"] == "triage"
    assert routing["input"]["menu"]["available"] is True
    assert set(routing["input"]["menu"]["routes"]) == {"coding", "story"}
    assert routing["output"]["chosen_route"] == "coding"
    assert routing["output"]["chosen_decision"] == "route"

    spec = bundle.manifest["spec"]
    assert spec["sources"][0]["license"] == "unknown"
    assert spec["counts"]["total"] == 2
    assert spec["examples"]["node"] == 1
    assert spec["examples"]["routing"] == 1
    assert spec["examples"]["by_node"] == {"coding": 1, "triage": 1}
    assert spec["examples"]["by_route"] == {"coding": 1}
    assert spec["hash"] == spec["files"]["examples.jsonl"]
    assert spec["splits"]["hashes"]["train"] == spec["files"]["splits/train.jsonl"]
    assert spec["eval_contamination"]["status"] == "not-checked"
    assert spec["consent"] == {
        "required": True,
        "recorded": True,
        "statement": CONSENT,
        "scope": "local-training",
    }
    assert (out / "examples.jsonl").exists()
    assert (out / "splits" / "train.jsonl").exists()
    assert (out / "redaction-report.json").exists()
    assert (out / "splits.json").exists()


def test_redaction_removes_secrets_and_reports_patterns(tmp_path: Path) -> None:
    run = _make_run(tmp_path, SECRET_TASK, response=SECRET_RESPONSE)
    out = tmp_path / "dataset"
    result = export_dataset([run], out, consent=CONSENT, created_at="2026-10-25T00:00:00Z")

    text = _dataset_text(out)
    for secret in SECRET_VALUES:
        assert secret not in text
    report = json.loads((out / "redaction-report.json").read_text(encoding="utf-8"))
    assert report["enabled"] is True
    assert report["status"] == "redacted"
    assert report["clean"] is False
    assert report["total_redactions"] > 0
    counts = {pattern["name"]: pattern["count"] for pattern in report["patterns"]}
    assert counts["openai-api-key"] > 0
    assert counts["email"] > 0
    assert counts["home-path"] > 0
    assert any("input.messages" in field for field in report["patterns"][1]["fields"])
    assert result.manifest["spec"]["redaction"]["status"] == "redacted"
    for example in load_dataset(out).examples:
        assert example["redaction"]["redacted"] is True
        assert example["redaction"]["patterns"]
        for artifact in example["output"].get("artifacts", []):
            assert artifact["hash"] == sha256_text(artifact["content"])


def test_custom_redaction_config_can_drop_fields(tmp_path: Path) -> None:
    run = _make_run(tmp_path, "A task with alice@example.com", response="answer\n")
    config = _write_json(
        tmp_path / "redaction.json",
        {
            "default_action": "hash",
            "include_defaults": False,
            "patterns": [
                {
                    "name": "email-only",
                    "pattern": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
                }
            ],
            "fields": [{"name": "labeler", "action": "drop"}],
        },
    )
    out = tmp_path / "dataset"
    export_dataset(
        [run],
        out,
        consent=CONSENT,
        redact=str(config),
        created_at="2026-10-25T00:00:00Z",
    )
    text = _dataset_text(out)
    assert "alice@example.com" not in text
    report = json.loads((out / "redaction-report.json").read_text(encoding="utf-8"))
    patterns = {item["name"] for item in report["patterns"]}
    assert patterns == {"email-only"}
    assert report["patterns"][0]["count"] > 0


def test_export_without_consent_fails_closed(tmp_path: Path) -> None:
    run = _make_run(tmp_path, "A plain task")
    out = tmp_path / "dataset"
    with pytest.raises(DatasetExportError, match="consent is required"):
        export_dataset([run], out)
    assert not out.exists()

    result = runner.invoke(
        app,
        ["data", "export", "--run", str(run), "--out", str(out), "--require-consent"],
    )
    assert result.exit_code == 3
    assert "consent is required" in result.stderr
    assert not out.exists()


def test_failure_traces_require_opt_in_and_are_labeled_failure(tmp_path: Path) -> None:
    run = _make_run(tmp_path, "This worker fails", failure_node="coding")
    out = tmp_path / "dataset"
    with pytest.raises(DatasetExportError, match="no eligible runs"):
        export_dataset([run], out, consent=CONSENT)

    result = export_dataset(
        [run],
        out,
        consent=CONSENT,
        include_failures=True,
        created_at="2026-10-25T00:00:00Z",
    )
    assert result.examples == 2
    bundle = load_dataset(out)
    for example in bundle.examples:
        assert example["label"]["outcome"] == "failure"
        assert example["label"]["source"] == "run_outcome"
        assert example["source"]["run_outcome"] == "failed"
    failed_node = next(example for example in bundle.examples if example["kind"] == "node")
    assert failed_node["output"]["error"]["code"] == "MG-3001"
    assert failed_node["output"]["finish_reason"] is None
    assert "Failure traces were included explicitly" in "\n".join(
        bundle.manifest["spec"]["limitations"]
    )
    assert bundle.manifest["spec"]["counts"]["by_outcome"] == {"failure": 2}


def test_checks_and_human_labels_supply_evidence_not_guesses(tmp_path: Path) -> None:
    run = _make_run(tmp_path, "Write exportable code", response="export const ok = true;\n")
    checks = _write_json(
        tmp_path / "checks.json",
        {
            "checks": [
                {
                    "run_id": run.name,
                    "node": "coding",
                    "checks": [
                        {"type": "contains", "artifact": "answer", "value": "export"},
                        {"type": "artifact", "artifact": "answer", "min_bytes": 5},
                    ],
                }
            ]
        },
    )
    labels = _write_json(
        tmp_path / "labels.json",
        {
            "labels": [
                {
                    "run_id": run.name,
                    "kind": "routing",
                    "node": "triage",
                    "route": "coding",
                    "labeler": "synthetic-reviewer",
                }
            ]
        },
    )
    out = tmp_path / "dataset"
    export_dataset(
        [run],
        out,
        consent=CONSENT,
        checks=[checks],
        labels=[labels],
        created_at="2026-10-25T00:00:00Z",
    )
    bundle = load_dataset(out)
    node = next(example for example in bundle.examples if example["kind"] == "node")
    routing = next(example for example in bundle.examples if example["kind"] == "routing")
    assert node["label"]["outcome"] == "success"
    assert node["label"]["source"] == "eval_check"
    assert len(node["label"]["checks"]) == 2
    assert all(check["passed"] for check in node["label"]["checks"])
    assert routing["label"]["outcome"] == "success"
    assert routing["label"]["source"] == "human_label"
    assert routing["label"]["route"] == "coding"
    assert routing["label"]["should_abstain"] is False
    assert routing["label"]["labeler"] == "synthetic-reviewer"


def test_splits_group_by_source_and_near_duplicates_do_not_leak(tmp_path: Path) -> None:
    shared = "Write a TypeScript debounce helper for a search box"
    run_a = _make_run(tmp_path, shared + ".")
    run_b = _make_run(tmp_path, shared + "!")
    run_c = _make_run(
        tmp_path,
        "Tell me a story about a lighthouse keeper and a storm.",
        route="story",
        response="The light answered the storm.\n",
    )
    run_d = _make_run(tmp_path, shared + "?")
    labels = _write_json(
        tmp_path / "groups.json",
        {
            "labels": [
                {"run_id": run_a.name, "node": "*", "group": "repo-a"},
                {"run_id": run_b.name, "node": "*", "group": "repo-b"},
                {"run_id": run_c.name, "node": "*", "group": "repo-c"},
                {"run_id": run_d.name, "node": "*", "group": "repo-d"},
            ]
        },
    )
    plan = _write_json(
        tmp_path / "plan.json",
        {
            "seed": "frozen-seed",
            "ratios": {"train": 0.8, "test": 0.1, "validation": 0.1},
            "groups": {
                "group:repo-a": {"split": "train"},
                "group:repo-b": {"split": "train"},
                "group:repo-c": {"split": "test"},
                "group:repo-d": {"split": "test"},
            },
        },
    )
    out = tmp_path / "dataset"
    first = export_dataset(
        [run_a, run_b, run_c, run_d],
        out,
        consent=CONSENT,
        labels=[labels],
        split_plan=plan,
        created_at="2026-10-25T00:00:00Z",
    )
    bundle = load_dataset(out)
    assert first.manifest["spec"]["preprocessing"]["dedupe"]["near_removed"] >= 2
    assert first.manifest["spec"]["splits"]["frozen_test"] is True
    assert first.manifest["spec"]["splits"]["groups"]["group:repo-c"]["split"] == "test"
    assert first.manifest["spec"]["splits"]["groups"]["group:repo-a"]["split"] == "train"

    group_splits: dict[str, set[str]] = {}
    for example in bundle.examples:
        group_splits.setdefault(example["group"], set()).add(example["split"])
    assert all(len(splits) == 1 for splits in group_splits.values())
    assert group_splits["group:repo-c"] == {"test"}
    assert any(example["split"] == "test" for example in bundle.examples)

    near_signature = "Write a TypeScript debounce helper for a search box.".lower()
    matches = [
        example
        for example in bundle.examples
        if str(example["input"]["task"]).lower().rstrip(".!?") == near_signature.rstrip(".!?")
    ]
    assert len(matches) == 2
    assert {example["kind"] for example in matches} == {"node", "routing"}
    assert len({example["group"] for example in matches}) == 1
    assert len({example["split"] for example in matches}) == 1
    assert (
        sum(
            example["split"] == "test"
            for example in bundle.examples
            if example["group"] == "group:repo-c"
        )
        == 2
    )

    first_examples = (out / "examples.jsonl").read_bytes()
    second = tmp_path / "dataset-again"
    export_dataset(
        [run_a, run_b, run_c, run_d],
        second,
        consent=CONSENT,
        labels=[labels],
        split_plan=out / "splits.json",
        created_at="2026-10-25T00:00:00Z",
    )
    assert (second / "examples.jsonl").read_bytes() == first_examples
    assert (second / "splits" / "test.jsonl").read_bytes() == (
        out / "splits" / "test.jsonl"
    ).read_bytes()
    second_manifest = load_dataset(second).manifest
    second_manifest["spec"]["provenance"]["command"] = first.manifest["spec"]["provenance"][
        "command"
    ]
    assert first.manifest == second_manifest


def test_export_is_deterministic_modulo_created_at(tmp_path: Path) -> None:
    run = _make_run(tmp_path, "Deterministic task")
    out = tmp_path / "dataset"
    first = export_dataset([run], out, consent=CONSENT, created_at="2026-10-25T00:00:00Z")
    examples = (out / "examples.jsonl").read_bytes()
    split_files = {
        split: (out / "splits" / f"{split}.jsonl").read_bytes()
        for split in ("train", "validation", "test")
    }
    second = export_dataset([run], out, consent=CONSENT, created_at="2026-10-25T00:00:00Z")
    assert (out / "examples.jsonl").read_bytes() == examples
    assert {
        split: (out / "splits" / f"{split}.jsonl").read_bytes()
        for split in ("train", "validation", "test")
    } == split_files
    assert first.manifest == second.manifest
    assert first.manifest["spec"]["hash"] == second.manifest["spec"]["hash"]


def test_contamination_check_excludes_eval_case_tasks(tmp_path: Path) -> None:
    task = "Write a TypeScript debounce function that preserves the last arguments."
    run = _make_run(tmp_path, task)
    out = tmp_path / "dataset"
    result = export_dataset(
        [run],
        out,
        consent=CONSENT,
        eval_suites=[JS_SUITE],
        created_at="2026-10-25T00:00:00Z",
    )
    bundle = load_dataset(out)
    contamination = bundle.manifest["spec"]["eval_contamination"]
    assert contamination["status"] == "checked"
    assert contamination["excluded"] == 2
    assert contamination["matches"]
    assert contamination["matches"][0]["case_id"] == "js-001"
    assert bundle.examples == []
    assert result.manifest["spec"]["counts"]["total"] == 0
    assert bundle.manifest["spec"]["examples"]["total"] == 0


def test_cli_data_export_reports_summary(tmp_path: Path) -> None:
    run = _make_run(tmp_path, "CLI export task")
    out = tmp_path / "dataset"
    result = runner.invoke(
        app,
        [
            "data",
            "export",
            "--run",
            str(run),
            "--out",
            str(out),
            "--consent",
            CONSENT,
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "dataset:" in result.stdout
    assert "examples:" in result.stdout
    assert "node:" in result.stdout
    assert "routing:" in result.stdout
    assert "redaction:" in result.stdout
    assert (out / "dataset.json").exists()
    assert load_dataset(out).manifest["spec"]["consent"]["statement"] == CONSENT

    as_json = runner.invoke(
        app,
        [
            "data",
            "export",
            "--run",
            str(run),
            "--out",
            str(tmp_path / "dataset-json"),
            "--consent",
            CONSENT,
            "--json",
        ],
    )
    assert as_json.exit_code == 0, as_json.stdout + as_json.stderr
    payload = json.loads(as_json.stdout)
    assert payload["example_count"] == 2
    assert payload["counts"]["train"] == 2
