"""Offline tests for the local GUI server and JSON API."""

from __future__ import annotations

import http.client
import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from modelgraph.cli import app
from modelgraph.registry import Registry
from modelgraph.web import create_server, is_loopback_host
from modelgraph.web.server import ModelGraphHTTPServer

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
MODELS = EXAMPLES / "models"
DEMO_TASK = "Write a debounce function in TypeScript."
runner = CliRunner()


@contextmanager
def _running_server(**kwargs: Any) -> Iterator[ModelGraphHTTPServer]:
    server = create_server("127.0.0.1", 0, **kwargs)
    thread = threading.Thread(
        target=server.serve_forever,
        kwargs={"poll_interval": 0.01},
        daemon=True,
    )
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def api(tmp_path: Path) -> Iterator[ModelGraphHTTPServer]:
    with _running_server(runs_dir=tmp_path / "runs") as server:
        yield server


def _request(
    server: ModelGraphHTTPServer,
    method: str,
    path: str,
    *,
    body: Any | None = None,
    raw: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, str, bytes]:
    request_headers = dict(headers or {})
    payload: bytes | None
    if raw is not None:
        payload = raw
    elif body is not None:
        payload = json.dumps(body).encode("utf-8")
        request_headers.setdefault("Content-Type", "application/json")
    else:
        payload = None
    host, port = server.server_address[:2]
    connection = http.client.HTTPConnection(str(host), int(port), timeout=10)
    try:
        connection.request(method, path, body=payload, headers=request_headers)
        response = connection.getresponse()
        data = response.read()
        return response.status, response.getheader("Content-Type", ""), data
    finally:
        connection.close()


def _json(data: bytes) -> dict[str, Any]:
    payload = json.loads(data.decode("utf-8"))
    assert isinstance(payload, dict)
    return payload


def _error(data: bytes) -> dict[str, Any]:
    payload = _json(data)
    error = payload["error"]
    assert isinstance(error["code"], str) and error["code"].startswith("MG-")
    assert isinstance(error["message"], str) and error["message"]
    return payload


def _create_run(
    server: ModelGraphHTTPServer,
    *,
    graph_id: str = "hello-triage",
    task: str = DEMO_TASK,
    provider: str = "mock",
) -> dict[str, Any]:
    status, content_type, data = _request(
        server,
        "POST",
        "/api/runs",
        body={"graph_id": graph_id, "task": task, "provider": provider},
    )
    assert status == 201, data
    assert content_type.startswith("application/json")
    return _json(data)


def test_health(api: ModelGraphHTTPServer) -> None:
    status, content_type, data = _request(api, "GET", "/api/health")
    assert status == 200
    assert content_type.startswith("application/json")
    payload = _json(data)
    assert payload == {"ok": True, "version": "0.6.0", "ui": "1"}


def test_graphs_list_and_detail(api: ModelGraphHTTPServer) -> None:
    status, _, data = _request(api, "GET", "/api/graphs")
    assert status == 200
    graphs = {item["id"]: item for item in _json(data)["graphs"]}
    assert {"hello-triage", "composed-assistant"} <= set(graphs)

    hello = graphs["hello-triage"]
    assert hello["name"] == "Hello Triage"
    assert hello["version"] == "1.0.0"
    assert hello["node_count"] == 3
    assert hello["edge_count"] == 2
    assert hello["hash"].startswith("sha256:")
    assert hello["source"].endswith("hello-triage.graph.yaml")
    assert isinstance(hello["warnings"], list)
    assert isinstance(hello["description"], str) and hello["description"]

    status, _, data = _request(api, "GET", "/api/graphs/hello-triage")
    assert status == 200
    detail = _json(data)
    assert detail["graph"]["id"] == "hello-triage"
    assert set(detail["graph"]["nodes"]) == {"triage", "coding", "story"}
    assert detail["warnings"] == hello["warnings"]

    status, _, data = _request(api, "GET", "/api/graphs/hello-triage@1.0.0")
    assert status == 200
    assert _json(data)["graph"]["version"] == "1.0.0"


def test_models_list(api: ModelGraphHTTPServer) -> None:
    status, _, data = _request(api, "GET", "/api/models")
    assert status == 200
    models = {item["id"]: item for item in _json(data)["models"]}
    assert {"router-small", "coder-7b", "story-7b"} <= set(models)
    assert models["router-small"]["provider"] == "ollama"
    assert models["router-small"]["model"] == "qwen2.5:1.5b-instruct"
    assert models["router-small"]["license"] == "apache-2.0"
    assert models["router-small"]["description"]


def test_mock_run_round_trip(api: ModelGraphHTTPServer) -> None:
    created = _create_run(api)
    run_id = created["run_id"]
    assert created["outcome"] == "succeeded"
    assert created["run_dir"]
    assert [entry["node"] for entry in created["path"]] == ["triage", "coding"]
    assert created["path"][0]["decision"] == "route"
    assert created["path"][0]["to"] == "coding"
    assert created["usage"]["steps"] == 2
    assert created["artifacts"][0]["name"] == "answer"
    artifact_content = created["artifacts"][0]["content"]
    assert artifact_content

    status, _, data = _request(api, "GET", f"/api/runs/{run_id}")
    assert status == 200
    manifest = _json(data)
    assert manifest["run_id"] == run_id
    assert manifest["outcome"] == "succeeded"
    assert manifest["graph"]["id"] == "hello-triage"
    assert manifest["request"]["task"] == DEMO_TASK

    status, _, data = _request(api, "GET", f"/api/runs/{run_id}/trace")
    assert status == 200
    events = _json(data)["events"]
    assert events[0]["event"] == "run.started"
    assert any(event["event"] == "decision.made" for event in events)
    assert any(event["event"] == "provider.response" for event in events)

    status, _, data = _request(api, "GET", f"/api/runs/{run_id}/artifacts/answer")
    assert status == 200
    artifact = _json(data)["artifact"]
    assert artifact["name"] == "answer"
    assert artifact["content"] == artifact_content
    assert artifact["hash"].startswith("sha256:")

    status, _, data = _request(api, "GET", f"/api/runs/{run_id}/artifacts/missing")
    assert status == 404
    _error(data)

    status, _, data = _request(api, "GET", "/api/runs?limit=10")
    assert status == 200
    runs = _json(data)["runs"]
    listed = next(item for item in runs if item["run_id"] == run_id)
    assert listed["outcome"] == "succeeded"
    assert listed["graph_id"] == "hello-triage"
    assert listed["graph_version"] == "1.0.0"
    assert listed["tokens"] == created["usage"]["total_tokens"]
    assert listed["cost_usd"] == created["usage"]["cost_usd"]
    assert listed["started_at"]
    assert Path(listed["dir"]).is_dir()


def test_replay_round_trip(api: ModelGraphHTTPServer) -> None:
    created = _create_run(api)
    status, _, data = _request(
        api,
        "POST",
        f"/api/runs/{created['run_id']}/replay",
        body={},
    )
    assert status == 200, data
    payload = _json(data)
    assert payload["ok"] is True
    assert payload["run_id"] == created["run_id"]
    assert payload["replay_id"].startswith("replay_")
    assert payload["mismatches"] == []


def test_replay_hash_mismatch_is_409(api: ModelGraphHTTPServer) -> None:
    created = _create_run(api)
    run_dir = Path(created["run_dir"])
    manifest_path = run_dir / "run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["graph"]["hash"] = "sha256:" + "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    status, content_type, data = _request(
        api,
        "POST",
        f"/api/runs/{created['run_id']}/replay",
        body={},
    )
    assert status == 409
    assert content_type.startswith("application/json")
    error = _error(data)["error"]
    assert "graph hash changed" in error["message"]


def test_replay_can_explicitly_allow_ir_mismatch(api: ModelGraphHTTPServer) -> None:
    created = _create_run(api)
    run_dir = Path(created["run_dir"])
    manifest_path = run_dir / "run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["graph"]["hash"] = "sha256:" + "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    status, _, data = _request(
        api,
        "POST",
        f"/api/runs/{created['run_id']}/replay",
        body={"allow_ir_mismatch": True},
    )
    assert status == 200, data
    payload = _json(data)
    assert payload["ok"] is True
    assert payload["mismatches"] == ["graph hash mismatch allowed by flag"]


def test_invalid_requests_are_400(api: ModelGraphHTTPServer) -> None:
    cases: list[tuple[bytes | None, Any]] = [
        (b"{", None),
        (None, []),
        (None, {}),
        (None, {"graph_id": "hello-triage"}),
        (None, {"task": DEMO_TASK}),
        (None, {"graph_id": "hello-triage", "task": DEMO_TASK, "provider": "remote"}),
    ]
    for raw, body in cases:
        status, content_type, data = _request(api, "POST", "/api/runs", raw=raw, body=body)
        assert status == 400, data
        assert content_type.startswith("application/json")
        _error(data)
        assert b"Traceback" not in data
        assert b"<html" not in data.lower()

    status, _, data = _request(api, "GET", "/api/runs?limit=abc")
    assert status == 400
    _error(data)
    status, _, data = _request(api, "GET", "/api/runs?limit=0")
    assert status == 400
    _error(data)


def test_unknown_resources_are_404(api: ModelGraphHTTPServer) -> None:
    paths = [
        "/api/nope",
        "/api/graphs/does-not-exist",
        "/api/runs/run_missing",
        "/api/runs/run_missing/trace",
        "/api/runs/run_missing/artifacts/answer",
        "/api/runs/..%2f..%2fetc/passwd",
    ]
    for path in paths:
        status, content_type, data = _request(api, "GET", path)
        assert status == 404, (path, data)
        assert content_type.startswith("application/json")
        _error(data)

    status, _, data = _request(api, "POST", "/api/runs/run_missing/replay", body={})
    assert status == 404
    _error(data)

    status, _, data = _request(
        api,
        "POST",
        "/api/runs",
        body={
            "graph_id": "does-not-exist",
            "task": DEMO_TASK,
            "provider": "mock",
        },
    )
    assert status == 404
    _error(data)


def test_method_not_allowed_is_405(api: ModelGraphHTTPServer) -> None:
    status, content_type, data = _request(api, "DELETE", "/api/health")
    assert status == 405
    assert content_type.startswith("application/json")
    _error(data)


def _provider_failure_registry(tmp_path: Path) -> Registry:
    models = tmp_path / "models"
    graphs = tmp_path / "graphs"
    models.mkdir()
    graphs.mkdir()
    (models / "bogus.yaml").write_text(
        """
apiVersion: modelgraph/v0
kind: Model
metadata:
  id: bogus-model
  name: Bogus Model
  version: 1.0.0
  description: A manifest with a provider the runtime cannot build.
spec:
  provider: custom
  model: bogus
  context_window: 4096
  license:
    name: apache-2.0
""".lstrip(),
        encoding="utf-8",
    )
    (graphs / "broken.graph.yaml").write_text(
        """
apiVersion: modelgraph/v0
kind: Graph
metadata:
  id: broken-provider
  name: Broken Provider
  version: 1.0.0
  description: A one-node graph whose provider cannot be constructed.
spec:
  entry: work
  inputs:
    - name: task
      type: text
      required: true
  exit: [answer]
  nodes:
    - id: work
      kind: worker
      description: Echo the request back.
      model: bogus-model
      reads: [task]
      instruction: Reply with the request.
      writes:
        - name: answer
          type: text
  edges: []
""".lstrip(),
        encoding="utf-8",
    )
    return Registry.from_paths([models], [graphs])


def test_provider_failure_is_502(tmp_path: Path) -> None:
    registry = _provider_failure_registry(tmp_path)
    with _running_server(
        runs_dir=tmp_path / "runs",
        registry=registry,
    ) as server:
        status, content_type, data = _request(
            server,
            "POST",
            "/api/runs",
            body={
                "graph_id": "broken-provider",
                "task": "anything",
                "provider": "auto",
            },
        )
        assert status == 502, data
        assert content_type.startswith("application/json")
        error = _error(data)["error"]
        assert error["code"] == "MG-3005"
        assert "provider" in error["message"].lower()
        assert b"Traceback" not in data


def test_static_files_and_traversal(tmp_path: Path) -> None:
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<!doctype html><title>MG</title>", encoding="utf-8")
    (static / "app.js").write_text("console.log('ok');\n", encoding="utf-8")
    secret = tmp_path / "secret.txt"
    secret.write_text("do-not-serve", encoding="utf-8")

    with _running_server(
        runs_dir=tmp_path / "runs",
        static_dir=static,
    ) as server:
        status, content_type, data = _request(server, "GET", "/")
        assert status == 200
        assert content_type.startswith("text/html")
        assert b"MG" in data

        status, content_type, data = _request(server, "GET", "/static/app.js")
        assert status == 200
        assert content_type.startswith("text/javascript")
        assert b"console.log" in data

        status, _, data = _request(server, "GET", "/static/missing.js")
        assert status == 404
        _error(data)

        traversal = [
            "/../secret.txt",
            "/static/../secret.txt",
            "/static/%2e%2e/secret.txt",
            "/static/..%5csecret.txt",
            "/static/%2e%2e%2fsecret.txt",
            "/static/....//secret.txt",
        ]
        for path in traversal:
            status, _, data = _request(server, "GET", path)
            assert status == 404, (path, data)
            assert b"do-not-serve" not in data
            _error(data)


def test_missing_frontend_files_return_clean_404(tmp_path: Path) -> None:
    with _running_server(
        runs_dir=tmp_path / "runs",
        static_dir=tmp_path / "does-not-exist",
    ) as server:
        status, content_type, data = _request(server, "GET", "/")
        assert status == 404
        assert content_type.startswith("application/json")
        _error(data)
        assert b"Traceback" not in data


def test_create_server_defaults_to_loopback(tmp_path: Path) -> None:
    server = create_server(port=0, runs_dir=tmp_path)
    try:
        assert server.server_address[0] == "127.0.0.1"
        assert server.server_address[1] > 0
    finally:
        server.server_close()

    assert is_loopback_host("localhost")
    assert is_loopback_host("127.0.0.1")
    assert is_loopback_host("::1")
    assert is_loopback_host("[::1]")
    assert not is_loopback_host("0.0.0.0")
    assert not is_loopback_host("192.168.0.1")

    with pytest.raises(ValueError, match="loopback-only"):
        create_server(host="0.0.0.0", port=0, runs_dir=tmp_path)


def test_cli_ui_help_and_loopback_guard() -> None:
    help_result = runner.invoke(app, ["ui", "--help"])
    assert help_result.exit_code == 0
    assert "--port" in help_result.output
    assert "--open" in help_result.output
    assert "--no-browser" in help_result.output

    remote = runner.invoke(app, ["ui", "--host", "0.0.0.0", "--port", "0"])
    assert remote.exit_code == 2
    assert "loopback" in remote.output
