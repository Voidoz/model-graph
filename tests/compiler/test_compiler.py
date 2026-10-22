from pathlib import Path

import yaml

from modelgraph.compiler import compile_graph
from modelgraph.contracts.loader import load_document
from modelgraph.registry import Registry

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
MODELS = EXAMPLES / "models"


def _write(tmp_path: Path, name: str, data: dict) -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _hello() -> dict:
    return load_document(EXAMPLES / "hello-triage.graph.yaml")


def test_examples_compile() -> None:
    for path in sorted(EXAMPLES.glob("*.graph.yaml")):
        result = compile_graph(path, strict_models=True)
        assert result.ok, result.errors
        assert result.graph is not None
        assert result.graph.graph_hash.startswith("sha256:")


def test_hash_is_stable() -> None:
    first = compile_graph(EXAMPLES / "hello-triage.graph.yaml", strict_models=True)
    second = compile_graph(EXAMPLES / "hello-triage.graph.yaml", strict_models=True)
    assert first.graph and second.graph
    assert first.graph.graph_hash == second.graph.graph_hash


def test_cycle_is_rejected(tmp_path: Path) -> None:
    data = _hello()
    data["spec"]["edges"].append(
        {"from": "story", "to": "triage", "when": "loop back", "artifact": "text"}
    )
    path = _write(tmp_path, "cycle.graph.yaml", data)
    result = compile_graph(path, strict_models=True)
    assert not result.ok
    assert any("cycle" in error for error in result.errors)


def test_unreachable_node_is_rejected(tmp_path: Path) -> None:
    data = _hello()
    data["spec"]["nodes"].append(
        {
            "id": "orphan",
            "kind": "worker",
            "description": "An orphaned worker.",
            "model": "coder-7b",
            "instruction": "Say hello.",
            "writes": [{"name": "orphan_answer", "type": "text"}],
        }
    )
    path = _write(tmp_path, "orphan.graph.yaml", data)
    result = compile_graph(path, strict_models=True)
    assert not result.ok
    assert any("unreachable" in error for error in result.errors)


def test_missing_read_is_rejected(tmp_path: Path) -> None:
    data = _hello()
    for node in data["spec"]["nodes"]:
        if node["id"] == "coding":
            node["reads"] = ["missing_artifact"]
    path = _write(tmp_path, "missing-read.graph.yaml", data)
    result = compile_graph(path, strict_models=True)
    assert not result.ok
    assert any("missing_artifact" in error for error in result.errors)


def test_unresolved_model_strict_and_offline(tmp_path: Path) -> None:
    path = EXAMPLES / "hello-triage.graph.yaml"
    strict = compile_graph(path, registry=Registry(), strict_models=True)
    assert not strict.ok
    assert any("not in the registry" in error for error in strict.errors)
    offline = compile_graph(path, registry=Registry(), strict_models=False)
    assert offline.ok
    assert any("not in the registry" in warning for warning in offline.warnings)


def test_conflicting_exit_type_warns() -> None:
    result = compile_graph(EXAMPLES / "hello-triage.graph.yaml", strict_models=True)
    assert result.ok
    assert any("conflicting types" in warning for warning in result.warnings)


def test_subgraph_compiles() -> None:
    result = compile_graph(EXAMPLES / "composed-assistant.graph.yaml", strict_models=True)
    assert result.ok
    assert result.graph is not None
    assert "coding-specialist@1.0.0" in result.graph.subgraphs
    assert result.graph.subgraphs["coding-specialist@1.0.0"].graph_hash.startswith("sha256:")


def test_registry_resolves_examples() -> None:
    registry = Registry.from_paths([MODELS], [EXAMPLES])
    assert registry.resolve_model("router-small") is not None
    assert registry.resolve_model("router-small@1.0.0") is not None
    assert registry.resolve_model("router-small@9.9.9") is None
    assert registry.resolve_graph("coding-specialist@1.0.0") is not None
