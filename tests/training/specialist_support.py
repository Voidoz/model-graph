"""Shared synthetic T-0023 dataset fixture for the specialist tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from modelgraph.compiler import compile_graph
from modelgraph.data import export_dataset
from modelgraph.providers.mock import MockProvider, MockScenario
from modelgraph.registry import Registry
from modelgraph.runtime.engine import run_graph

ROOT = Path(__file__).resolve().parents[2]
GRAPH = ROOT / "examples" / "coding-specialist.graph.yaml"
SUITE = ROOT / "evals/mock/js-specialist"
CONSENT = "synthetic test consent for the specialist pipeline"


def build_dataset(
    tmp_path: Path,
    *,
    count: int = 6,
    license_id: str = "CC0-1.0",
    consent: str | None = CONSENT,
    require_consent: bool = True,
) -> Path:
    """Export real mock runs through the T-0023 exporter; no network or real model."""
    compiled = compile_graph(
        GRAPH,
        registry=Registry.from_paths([ROOT / "registry/models"], [ROOT / "examples"]),
        strict_models=True,
        graph_paths=[GRAPH.parent],
    )
    assert compiled.ok and compiled.graph is not None, compiled.errors
    runs: list[Path] = []
    for index in range(count):
        decision = json.dumps(
            {
                "decision": "route",
                "to": "js-specialist",
                "confidence": 0.95,
                "reason": "test fixture",
            }
        )
        answer = (
            "export function fixture(input: readonly string[]): string[] {\n"
            "  return input.map((value) => value.trim());\n"
            f"}}\n// fixture {index:04d}\n"
        )
        scenario = MockScenario(responses={"coding-triage": [decision], "js-specialist": [answer]})
        outcome = run_graph(
            compiled.graph,
            f"Write a TypeScript helper that trims labels. Fixture ref {index + 1:04d}.",
            lambda model, scenario=scenario: MockProvider(
                scenario, max_context=model.spec.context_window
            ),
            runs_dir=tmp_path / "runs",
            run_id=f"specialist-test-{index:04d}",
            seed=7,
        )
        assert outcome.outcome == "succeeded", outcome.error
        assert outcome.run_dir is not None
        runs.append(outcome.run_dir)
    result = export_dataset(
        runs,
        tmp_path / "dataset",
        dataset_id="js-specialist-test",
        dataset_name="Specialist test fixture",
        dataset_version="1.0.0",
        dataset_description="Mock runs exported for the specialist training tests.",
        task="Implement a TypeScript helper.",
        license_id=license_id,
        consent=consent,
        require_consent=require_consent,
        eval_suites=[SUITE],
        dedupe="exact",
        group_by="run",
        split_ratio="0.7,0.15,0.15",
        split_seed="specialist-test",
        split_plan=tmp_path / "splits.json",
        created_at="2026-10-25T00:00:00Z",
    )
    return result.dataset_dir


def write_config(
    tmp_path: Path,
    dataset: Path,
    *,
    output: Path | None = None,
    method: str = "lora",
    mode: str = "synthetic",
    model_id: str = "js-specialist-test",
    version: str = "0.1.0",
    license_id: str = "CC0-1.0",
) -> Path:
    """Write a small valid specialist config that points at the fixture dataset."""
    config: dict[str, Any] = {
        "schema": "modelgraph/specialist-training-config/v0",
        "name": "specialist test rehearsal",
        "method": method,
        "mode": mode,
        "dataset": str(dataset),
        "node": "js-specialist",
        "output_dir": str(output or (tmp_path / "out")),
        "seed": 20261025,
        "created_at": "2026-10-25T00:00:00Z",
        "model": {
            "id": model_id,
            "name": "Specialist test rehearsal",
            "version": version,
            "description": "A deterministic test artifact for the specialist pipeline.",
            "base": "Qwen/Qwen2.5-Coder-3B-Instruct",
            "base_uri": "hf://Qwen/Qwen2.5-Coder-3B-Instruct",
            "base_license": "apache-2.0",
            "provider": "mock",
            "model": f"synthetic-specialist://{model_id}@{version}",
            "context_window": 32768,
            "max_output_tokens": 1024,
            "resources": {
                "ram_gb": 0.2,
                "disk_gb": 0.1,
                "cpu_only": True,
                "network_access": "none",
            },
        },
        "hyperparameters": {
            "rank": 8,
            "alpha": 16,
            "dropout": 0.05,
            "learning_rate": 0.0002,
            "epochs": 2,
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "max_seq_length": 1024,
            "quantization": "none",
            "target_modules": ["q_proj", "v_proj"],
            "optimizer": "paged_adamw_8bit",
            "scheduler": "cosine",
            "warmup_ratio": 0.03,
            "gradient_checkpointing": True,
        },
        "hardware": {
            "tier": "cpu",
            "gpu": "CPU test",
            "vram_gb": 0,
            "cpu_only": True,
            "notes": "test",
            "expected_data": "6 fixture examples",
        },
        "evaluation": {
            "node_suite": "evals/mock/js-specialist",
            "node_graph": "examples/coding-specialist.graph.yaml",
            "node": "js-specialist",
            "graph_suite": "evals/e2e/coding-specialist",
            "graph_graph": "examples/coding-specialist.graph.yaml",
            "provider": "mock",
            "reports_dir": str(tmp_path / "eval-runs"),
        },
        "promotion": {"require_passing_eval": True},
    }
    if method == "distill":
        config["distillation"] = {
            "teacher": "recorded-dataset-output",
            "teacher_license": license_id,
            "filter": "recorded",
            "min_bytes": 40,
            "include_failures": False,
        }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path
