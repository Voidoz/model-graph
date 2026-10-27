"""Build valid (and deliberately invalid) registry contributions for tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from modelgraph.registry_store import REPORT_SCHEMA, file_hash

ROOT = Path(__file__).resolve().parents[2]
HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64
HASH_C = "sha256:" + "c" * 64

DEFAULT_CARD = """# Test specialist 0.1.0

## Base model

Qwen/Qwen2.5-Coder-3B-Instruct, apache-2.0.

## Training data

A consented synthetic T-0023 export with frozen splits; license CC0-1.0.

## Method

QLoRA SFT rehearsal with a deterministic fixture.

## Hyperparameters

rank=16, alpha=32, learning_rate=0.0002.

## Evaluation

Node suite and graph regression pass with the model pinned.

## Limitations

Narrow data may regress off-distribution; this is a fixture, not a quality claim.

## License

Model weights: apache-2.0. Dataset: CC0-1.0. This is a declaration, not legal advice.
"""


def manifest_document(
    *,
    model_id: str = "js-specialist-test",
    version: str = "0.1.0",
    license_name: str = "apache-2.0",
    weights_sha256: str | None = None,
) -> dict[str, Any]:
    integrity: dict[str, Any] = {"revision": "fixture@1"}
    if weights_sha256 is not None:
        integrity["sha256"] = weights_sha256
    return {
        "apiVersion": "modelgraph/v0",
        "kind": "Model",
        "metadata": {
            "id": model_id,
            "name": "Test specialist",
            "version": version,
            "description": "A registry test fixture model; not a real trained model.",
            "labels": {"role": "worker", "method": "qlora"},
            "authors": ["Model Graph contributors"],
        },
        "spec": {
            "provider": "mock",
            "model": "synthetic-specialist://test",
            "context_window": 32768,
            "max_output_tokens": 1024,
            "capabilities": ["chat"],
            "cost": {"pricing": "free"},
            "resources": {"ram_gb": 0.2, "cpu_only": True, "network_access": "none"},
            "license": {
                "name": license_name,
                "notes": "Fixture declaration, not legal advice.",
            },
            "integrity": integrity,
            "eval": {"suite": "fixture/node", "report": "eval-report.json"},
        },
    }


def _write_evidence(
    directory: Path, level: str, *, suite_hash: str, graph_hash: str
) -> dict[str, dict[str, str]]:
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "schema": "modelgraph/eval-report/v0",
                "suite": {"hash": suite_hash},
                "graph": {"hash": graph_hash},
                "gate": {"passed": True},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(
        json.dumps({"schema": "modelgraph/eval-manifest/v0", "level": level}) + "\n",
        encoding="utf-8",
    )
    return {
        "report": {"path": str(report_path), "hash": file_hash(report_path)},
        "run_manifest": {"path": str(manifest_path), "hash": file_hash(manifest_path)},
    }


def make_contribution(
    directory: Path,
    *,
    model_id: str = "js-specialist-test",
    version: str = "0.1.0",
    license_name: str = "apache-2.0",
    include_card: bool = True,
    include_report: bool = True,
    passed: bool = True,
    card_text: str | None = None,
    report_overrides: dict[str, Any] | None = None,
    manifest_overrides: dict[str, Any] | None = None,
    weights: bool = False,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    weights_sha: str | None = None
    if weights:
        weights_path = directory / "weights" / "adapter.bin"
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        weights_path.write_bytes(b"not a real adapter; test fixture bytes")
        weights_sha = file_hash(weights_path).removeprefix("sha256:")
    manifest = manifest_document(
        model_id=model_id,
        version=version,
        license_name=license_name,
        weights_sha256=weights_sha,
    )
    if manifest_overrides:
        _deep_update(manifest, manifest_overrides)
    (directory / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    if include_card:
        (directory / "card.md").write_text(card_text or DEFAULT_CARD, encoding="utf-8")
    if include_report:
        node = _write_evidence(
            directory / "eval" / "node", "node", suite_hash=HASH_A, graph_hash=HASH_C
        )
        graph = _write_evidence(
            directory / "eval" / "graph", "graph", suite_hash=HASH_B, graph_hash=HASH_C
        )
        report: dict[str, Any] = {
            "schema": REPORT_SCHEMA,
            "model": {"id": model_id, "version": version},
            "command": (
                "mg eval node --suite evals/mock/js-specialist --graph pinned.graph.yaml "
                "--provider mock"
            ),
            "reproduce": (
                "mg eval node --suite evals/mock/js-specialist --graph pinned.graph.yaml "
                "--provider mock"
            ),
            "suite": {
                "node": {"id": "fixture/node", "version": "1.0.0", "hash": HASH_A},
                "graph": {"id": "fixture/e2e", "version": "1.0.0", "hash": HASH_B},
            },
            "graph": {
                "id": "coding-specialist",
                "version": "1.0.0",
                "hash": HASH_C,
                "pinned_model": f"{model_id}@{version}",
            },
            "evidence": {
                "node": {
                    "report": {
                        "path": "eval/node/report.json",
                        "hash": node["report"]["hash"],
                    },
                    "run_manifest": {
                        "path": "eval/node/manifest.json",
                        "hash": node["run_manifest"]["hash"],
                    },
                },
                "graph": {
                    "report": {
                        "path": "eval/graph/report.json",
                        "hash": graph["report"]["hash"],
                    },
                    "run_manifest": {
                        "path": "eval/graph/manifest.json",
                        "hash": graph["run_manifest"]["hash"],
                    },
                },
            },
            "seeds": {"training": 20261025, "evaluation": 1234},
            "hardware": {"tier": "cpu", "gpu": "fixture", "vram_gb": 0},
            "results": {
                "node": {
                    "passed": passed,
                    "cases": 6,
                    "task_success": 1.0 if passed else 0.0,
                    "validity": 1.0,
                    "protocol_error_rate": 0.0,
                },
                "graph": {
                    "passed": passed,
                    "cases": 3,
                    "task_success": 1.0 if passed else 0.0,
                    "validity": 1.0,
                    "protocol_error_rate": 0.0,
                },
            },
            "passed": passed,
            "generated_at": "2026-10-25T00:00:00Z",
            "limitations": ["Fixture only; mock provider; not a quality result."],
        }
        if weights and weights_sha is not None:
            report["weights"] = {
                "path": "weights/adapter.bin",
                "hash": f"sha256:{weights_sha}",
            }
        if report_overrides:
            _deep_update(report, report_overrides)
        (directory / "eval-report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return directory


def _deep_update(target: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


def write_profile(path: Path, licenses: list[str]) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "name": "registry-test",
                "network": "none",
                "providers": ["mock"],
                "licenses": licenses,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path
