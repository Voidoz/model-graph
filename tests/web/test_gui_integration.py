"""End-to-end, offline integration tests for the local GUI (T-0045).

The tests start a real ``create_server`` instance on an ephemeral loopback port and use
stdlib HTTP clients, so they cover routing, JSON envelopes, static assets, a complete mock
run (manifest/trace/artifact/replay), and the security boundary without a browser or network.
Static checks cover the no-external-asset and no-HTML-sink rules plus JS syntax when Node is
available.
"""

from __future__ import annotations

import http.client
import json
import os
import re
import select
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pytest

from modelgraph.registry import Registry
from modelgraph.web import create_server, is_loopback_host
from modelgraph.web.server import ModelGraphHTTPServer

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "src" / "modelgraph" / "web" / "static"
DEMO_TASK = "Write a debounce function in TypeScript."
PAGES = ("index.html", "graph.html", "run.html", "models.html", "editor.html")
SVG_NAMESPACE = "http://www.w3.org/2000/svg"


# --------------------------------------------------------------------------- helpers


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
def gui(tmp_path: Path) -> Iterator[ModelGraphHTTPServer]:
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
) -> tuple[int, dict[str, str], bytes]:
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
    connection = http.client.HTTPConnection(str(host), int(port), timeout=15)
    try:
        connection.request(method, path, body=payload, headers=request_headers)
        response = connection.getresponse()
        data = response.read()
        response_headers = {key.lower(): value for key, value in response.getheaders()}
        return response.status, response_headers, data
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
    assert b"Traceback" not in data
    assert b"<html" not in data.lower()
    return payload


def _create_run(server: ModelGraphHTTPServer, **overrides: Any) -> dict[str, Any]:
    request = {
        "graph_id": "hello-triage",
        "task": DEMO_TASK,
        "provider": "mock",
    }
    request.update(overrides)
    status, headers, data = _request(server, "POST", "/api/runs", body=request)
    assert status == 201, data
    assert headers["content-type"].startswith("application/json")
    return _json(data)


def _asset_refs(html: str) -> set[str]:
    return set(re.findall(r'(?:src|href)="(/static/[^"]+)"', html))


# --------------------------------------------------------------------------- page audit


class PageAudit(HTMLParser):
    """Collect the accessibility and asset facts the GUI contract relies on."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: list[str] = []
        self.duplicate_ids: list[str] = []
        self.scripts: list[str] = []
        self.stylesheets: list[str] = []
        self.refs: list[str] = []
        self.controls: list[tuple[str, str | None]] = []
        self.buttons: list[str | None] = []
        self.label_targets: set[str] = set()
        self.inline_handlers: list[str] = []
        self.inline_scripts = 0
        self.inline_styles = 0
        self.lang: str | None = None
        self.has_viewport = False
        self.has_skip_link = False
        self.main_ids: list[str] = []
        self._seen_ids: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "html":
            self.lang = values.get("lang")
        ident = values.get("id")
        if ident:
            if ident in self._seen_ids:
                self.duplicate_ids.append(ident)
            self._seen_ids.add(ident)
            self.ids.append(ident)
            if tag == "main":
                self.main_ids.append(ident)
        for name, value in attrs:
            if name.startswith("on") and value is not None:
                self.inline_handlers.append(f"{tag}[{name}]")
        if tag == "meta" and values.get("name") == "viewport":
            self.has_viewport = True
        if tag == "a" and values.get("class") == "skip-link" and values.get("href") == "#main":
            self.has_skip_link = True
        if tag == "script":
            src = values.get("src")
            if src:
                self.scripts.append(src)
            else:
                self.inline_scripts += 1
        if tag == "style":
            self.inline_styles += 1
        if tag == "link" and values.get("rel") == "stylesheet":
            self.stylesheets.append(values.get("href") or "")
        if tag == "label" and values.get("for"):
            self.label_targets.add(values["for"])
        if tag in {"input", "select", "textarea"} and values.get("type") != "hidden":
            self.controls.append((tag, values.get("id")))
        if tag == "button":
            self.buttons.append(values.get("type"))
        for name in ("src", "href"):
            value = values.get(name)
            if value:
                self.refs.append(value)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)


def _audit(path: Path) -> PageAudit:
    parser = PageAudit()
    parser.feed(path.read_text(encoding="utf-8"))
    parser.close()
    return parser


# --------------------------------------------------------------------------- API tests


def test_health_and_catalog_contract(gui: ModelGraphHTTPServer) -> None:
    status, headers, data = _request(gui, "GET", "/api/health")
    assert status == 200
    assert headers["content-type"].startswith("application/json")
    assert _json(data) == {"ok": True, "version": "1.0.0", "ui": "1"}

    status, _, data = _request(gui, "GET", "/api/graphs")
    assert status == 200
    graphs = {item["id"]: item for item in _json(data)["graphs"]}
    assert {"hello-triage", "composed-assistant", "coding-specialist"} <= set(graphs)
    hello = graphs["hello-triage"]
    assert hello["name"] == "Hello Triage"
    assert hello["version"] == "1.0.0"
    assert hello["node_count"] == 3
    assert hello["edge_count"] == 2
    assert hello["hash"].startswith("sha256:")
    assert hello["source"].endswith("hello-triage.graph.yaml")
    assert any("MG-1016" in warning for warning in hello["warnings"])

    status, _, data = _request(gui, "GET", "/api/graphs/hello-triage")
    assert status == 200
    detail = _json(data)
    assert detail["name"] == "Hello Triage"
    assert detail["hash"].startswith("sha256:")
    assert detail["warnings"] == hello["warnings"]
    assert detail["graph"]["id"] == "hello-triage"
    assert set(detail["graph"]["nodes"]) == {"triage", "coding", "story"}

    status, _, data = _request(gui, "GET", "/api/graphs/hello-triage@1.0.0")
    assert status == 200
    assert _json(data)["graph"]["version"] == "1.0.0"

    status, _, data = _request(gui, "GET", "/api/models")
    assert status == 200
    models = {item["id"]: item for item in _json(data)["models"]}
    assert {"router-small", "coder-7b", "story-7b"} <= set(models)
    assert models["router-small"]["provider"] == "ollama"
    assert models["router-small"]["license"] == "apache-2.0"


def test_static_pages_assets_and_security_headers(gui: ModelGraphHTTPServer) -> None:
    for page in ("/", "/index.html", "/graph.html", "/run.html", "/models.html", "/editor.html"):
        status, headers, data = _request(gui, "GET", page)
        assert status == 200, (page, data)
        assert headers["content-type"].startswith("text/html")
        assert b"Model Graph" in data
        csp = headers["content-security-policy"]
        assert "default-src 'self'" in csp
        assert "unsafe-inline" not in csp
        assert headers["x-content-type-options"] == "nosniff"
        assert headers["x-frame-options"] == "DENY"
        assert headers["referrer-policy"] == "no-referrer"

    status, headers, data = _request(gui, "HEAD", "/")
    assert status == 200
    assert headers["content-type"].startswith("text/html")
    assert data == b""

    checked: set[str] = set()
    for page in PAGES:
        html = (STATIC / page).read_text(encoding="utf-8")
        for ref in _asset_refs(html):
            checked.add(ref)
            status, headers, data = _request(gui, "GET", ref)
            assert status == 200, (page, ref, data)
            if ref.endswith(".js"):
                assert headers["content-type"].startswith("text/javascript")
            elif ref.endswith(".css"):
                assert headers["content-type"].startswith("text/css")
    assert {"/static/common.js", "/static/styles.css"} <= checked
    assert {
        "/static/graph.js",
        "/static/run.js",
        "/static/models.js",
        "/static/editor.js",
        "/static/editor.css",
    } <= checked


def test_mock_run_round_trip(gui: ModelGraphHTTPServer) -> None:
    created = _create_run(gui)
    run_id = created["run_id"]
    assert run_id.startswith("run_")
    assert created["outcome"] == "succeeded"
    assert Path(created["run_dir"]).is_dir()
    assert [entry["node"] for entry in created["path"]] == ["triage", "coding"]
    assert created["usage"]["steps"] == 2
    artifact_content = created["artifacts"][0]["content"]
    assert artifact_content

    status, _, data = _request(gui, "GET", f"/api/runs/{run_id}")
    assert status == 200
    manifest = _json(data)
    assert manifest["run_id"] == run_id
    assert manifest["outcome"] == "succeeded"
    assert manifest["graph"]["id"] == "hello-triage"
    assert manifest["request"]["task"] == DEMO_TASK
    assert manifest["usage"]["total_tokens"] == created["usage"]["total_tokens"]

    status, _, data = _request(gui, "GET", f"/api/runs/{run_id}/trace")
    assert status == 200
    events = _json(data)["events"]
    assert events[0]["event"] == "run.started"
    assert any(event["event"] == "decision.made" for event in events)
    assert any(event["event"] == "provider.response" for event in events)

    status, _, data = _request(gui, "GET", f"/api/runs/{run_id}/artifacts/answer")
    assert status == 200
    artifact = _json(data)["artifact"]
    assert artifact["name"] == "answer"
    assert artifact["content"] == artifact_content
    assert artifact["hash"].startswith("sha256:")

    status, _, data = _request(gui, "GET", "/api/runs?limit=10")
    assert status == 200
    listed = next(item for item in _json(data)["runs"] if item["run_id"] == run_id)
    assert listed["graph_id"] == "hello-triage"
    assert listed["tokens"] == created["usage"]["total_tokens"]
    assert Path(listed["dir"]).is_dir()

    status, headers, data = _request(
        gui,
        "POST",
        f"/api/runs/{run_id}/replay",
        body={},
    )
    assert status == 200, data
    assert headers["content-type"].startswith("application/json")
    replay = _json(data)
    assert replay["ok"] is True
    assert replay["run_id"] == run_id
    assert replay["replay_id"].startswith("replay_")
    assert replay["mismatches"] == []


def test_run_history_is_newest_first_by_started_at(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    fixtures = (
        ("run_old", "2026-01-01T00:00:00Z", 2_000_000_000),
        ("run_new", "2026-02-01T00:00:00Z", 1_000_000_000),
    )
    for run_id, started_at, mtime in fixtures:
        run_dir = runs / run_id
        run_dir.mkdir()
        (run_dir / "run.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "outcome": "succeeded",
                    "started_at": started_at,
                    "graph": {"id": "hello-triage", "version": "1.0.0"},
                    "usage": {"total_tokens": 10, "cost_usd": 0.0},
                }
            ),
            encoding="utf-8",
        )
        os.utime(run_dir, (mtime, mtime))

    with _running_server(runs_dir=runs) as server:
        status, _, data = _request(server, "GET", "/api/runs")
        assert status == 200
        # The older run directory has the newer mtime; started_at must still win.
        assert [item["run_id"] for item in _json(data)["runs"]] == ["run_new", "run_old"]


def test_error_envelopes_are_json_and_hide_internals(gui: ModelGraphHTTPServer) -> None:
    cases: list[tuple[str, str, dict[str, Any] | None, bytes | None, int]] = [
        ("GET", "/api/nope", None, None, 404),
        ("GET", "/api/graphs/does-not-exist", None, None, 404),
        ("GET", "/api/runs/run_missing", None, None, 404),
        ("GET", "/api/runs/run_missing/trace", None, None, 404),
        ("GET", "/api/runs/run_missing/artifacts/answer", None, None, 404),
        ("GET", "/api/runs?limit=abc", None, None, 400),
        ("GET", "/api/runs?limit=0", None, None, 400),
        ("POST", "/api/runs", None, b"{", 400),
        ("POST", "/api/runs", [], None, 400),
        ("POST", "/api/runs", {}, None, 400),
        ("POST", "/api/runs", {"task": DEMO_TASK}, None, 400),
        (
            "POST",
            "/api/runs",
            {"graph_id": "hello-triage", "task": DEMO_TASK, "provider": "remote"},
            None,
            400,
        ),
        ("POST", "/api/runs/run_missing/replay", {}, None, 404),
        ("DELETE", "/api/health", None, None, 405),
    ]
    for method, path, body, raw, expected in cases:
        status, headers, data = _request(gui, method, path, body=body, raw=raw)
        assert status == expected, (method, path, data)
        assert headers["content-type"].startswith("application/json")
        _error(data)

    status, headers, data = _request(gui, "GET", "/static/does-not-exist.js")
    assert status == 404
    assert headers["content-type"].startswith("application/json")
    _error(data)

    status, headers, data = _request(gui, "PUT", "/api/graphs")
    assert status == 405
    assert headers["content-type"].startswith("application/json")
    _error(data)


def test_replay_mismatch_is_409(gui: ModelGraphHTTPServer) -> None:
    created = _create_run(gui)
    manifest_path = Path(created["run_dir"]) / "run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["graph"]["hash"] = "sha256:" + "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    status, headers, data = _request(
        gui,
        "POST",
        f"/api/runs/{created['run_id']}/replay",
        body={},
    )
    assert status == 409
    assert headers["content-type"].startswith("application/json")
    error = _error(data)["error"]
    assert error["code"] == "MG-9004"
    assert "graph hash changed" in error["message"]

    allowed = _request(
        gui,
        "POST",
        f"/api/runs/{created['run_id']}/replay",
        body={"allow_ir_mismatch": True},
    )
    status, _, data = allowed
    assert status == 200, data
    assert _json(data)["mismatches"] == ["graph hash mismatch allowed by flag"]


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
    with _running_server(runs_dir=tmp_path / "runs", registry=registry) as server:
        status, headers, data = _request(
            server,
            "POST",
            "/api/runs",
            body={"graph_id": "broken-provider", "task": "anything", "provider": "auto"},
        )
        assert status == 502, data
        assert headers["content-type"].startswith("application/json")
        error = _error(data)["error"]
        assert error["code"] == "MG-3005"
        assert "provider" in error["message"].lower()


def test_loopback_only_and_traversal_rejection(
    gui: ModelGraphHTTPServer,
    tmp_path: Path,
) -> None:
    assert gui.server_address[0] == "127.0.0.1"
    assert gui.server_address[1] > 0
    with pytest.raises(ValueError, match="loopback-only"):
        create_server(host="0.0.0.0", port=0, runs_dir=tmp_path)

    assert is_loopback_host("localhost")
    assert is_loopback_host("127.0.0.1")
    assert is_loopback_host("::1")
    assert is_loopback_host("[::1]")
    assert not is_loopback_host("192.168.0.1")
    assert not is_loopback_host("0.0.0.0")

    marker = "Compose small, independently trainable"
    assert marker in (ROOT / "README.md").read_text(encoding="utf-8")
    traversal = (
        "/../README.md",
        "/static/../README.md",
        "/static/%2e%2e/README.md",
        "/static/%2e%2e%2fREADME.md",
        "/static/..%2fREADME.md",
        "/static/%2e%2e%5cREADME.md",
        "/static/....//README.md",
        "/static/%2e%2e%2f%2e%2e%2fREADME.md",
        "/.%2e/README.md",
    )
    for path in traversal:
        status, headers, data = _request(gui, "GET", path)
        assert status == 404, (path, data)
        assert headers["content-type"].startswith("application/json")
        assert marker.encode() not in data
        _error(data)

    status, _, data = _request(gui, "GET", "/api/runs/..%2f..%2fetc%2fpasswd")
    assert status == 404
    _error(data)


def test_missing_frontend_returns_json_or_browser_page(tmp_path: Path) -> None:
    with _running_server(
        runs_dir=tmp_path / "runs",
        static_dir=tmp_path / "does-not-exist",
    ) as server:
        status, headers, data = _request(server, "GET", "/")
        assert status == 404
        assert headers["content-type"].startswith("application/json")
        _error(data)

        status, headers, data = _request(
            server,
            "GET",
            "/",
            headers={"Accept": "text/html,application/xhtml+xml"},
        )
        assert status == 404
        assert headers["content-type"].startswith("text/html")
        assert b"GUI assets were not found" in data
        assert b"Traceback" not in data


# --------------------------------------------------------------------------- static checks


def test_no_external_assets_and_no_unsafe_sinks() -> None:
    assets = (
        sorted(STATIC.glob("*.html")) + sorted(STATIC.glob("*.css")) + sorted(STATIC.glob("*.js"))
    )
    assert assets
    forbidden_sinks = (
        "innerHTML",
        "outerHTML",
        "insertAdjacentHTML",
        "document.write",
        "eval(",
        "new Function",
    )
    for path in assets:
        source = path.read_text(encoding="utf-8")
        checked = source.replace(SVG_NAMESPACE, "")
        assert "http://" not in checked, f"external http URL in {path.name}"
        assert "https://" not in checked, f"external https URL in {path.name}"
        assert "//cdn." not in checked.lower(), f"CDN reference in {path.name}"
        assert "javascript:" not in checked.lower(), f"javascript: URL in {path.name}"
        if path.suffix == ".js":
            for sink in forbidden_sinks:
                assert sink not in source, f"unsafe DOM sink {sink!r} in {path.name}"
        if path.suffix == ".css":
            assert "@import" not in source, f"@import in {path.name}"
            assert "url(" not in source.lower(), f"external url() in {path.name}"
        if path.suffix == ".html":
            assert not re.search(r"\son[a-z]+\s*=", source, re.IGNORECASE), (
                f"inline event handler in {path.name}"
            )
            assert "style=" not in source, f"inline style attribute in {path.name}"

    for page in PAGES:
        html = (STATIC / page).read_text(encoding="utf-8")
        for ref in _asset_refs(html):
            assert (STATIC / ref.removeprefix("/static/")).is_file(), (
                f"{page} references missing asset {ref}"
            )


def test_javascript_syntax_when_node_is_available() -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is not installed; skipping JavaScript syntax checks")
    for path in sorted(STATIC.glob("*.js")):
        result = subprocess.run(
            [node, "--check", str(path)],
            capture_output=True,
            check=False,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"{path.name}: {result.stderr}"


def test_pages_have_accessibility_basics() -> None:
    for page in PAGES:
        audit = _audit(STATIC / page)
        assert audit.lang == "en", f"{page} is missing lang=en"
        assert audit.has_viewport, f"{page} is missing the viewport meta tag"
        assert audit.main_ids == ["main"], f"{page} must have exactly one main element"
        assert audit.has_skip_link, f"{page} is missing the skip link"
        assert "mg-nav" in audit.ids, f"{page} is missing the shared navigation host"
        assert not audit.duplicate_ids, f"{page} has duplicate ids: {audit.duplicate_ids}"
        assert not audit.inline_handlers, f"{page} uses inline handlers: {audit.inline_handlers}"
        assert audit.inline_scripts == 0, f"{page} uses an inline script"
        assert audit.inline_styles == 0, f"{page} uses an inline style block"
        for tag, ident in audit.controls:
            assert ident, f"{page} has an unlabeled {tag}"
            assert ident in audit.label_targets, f"{page} has no label for #{ident}"
        for button_type in audit.buttons:
            assert button_type in {"button", "submit", "reset"}, (
                f"{page} has a button without an explicit type"
            )
        for ref in audit.refs:
            assert not ref.startswith(("http://", "https://", "//", "ftp:")), (
                f"{page} references external URL {ref}"
            )
            if ref.startswith("data:"):
                assert ref == "data:,", f"{page} has unexpected data URL {ref}"

    for css in sorted(STATIC.glob("*.css")):
        source = css.read_text(encoding="utf-8")
        assert ":focus-visible" in source, f"{css.name} has no visible focus style"
        assert "prefers-reduced-motion: reduce" in source, f"{css.name} has no reduced-motion rule"


def test_pages_load_the_shared_shell_and_page_script() -> None:
    audits = {page: _audit(STATIC / page) for page in PAGES}
    for page, audit in audits.items():
        assert "/static/styles.css" in audit.stylesheets, page
        assert "/static/common.js" in audit.scripts, page
        assert audit.scripts.index("/static/common.js") < len(audit.scripts) - 1, page

    assert "/static/index.js" in audits["index.html"].scripts
    assert "/static/graph.js" in audits["graph.html"].scripts
    assert "/static/run.js" in audits["run.html"].scripts
    assert "/static/run.css" in audits["run.html"].stylesheets
    assert "/static/models.js" in audits["models.html"].scripts
    assert "/static/models.css" in audits["models.html"].stylesheets
    assert "/static/editor.js" in audits["editor.html"].scripts
    assert "/static/editor.css" in audits["editor.html"].stylesheets

    # Run history links to the console's recorded-run loader; keep that path wired.
    models_js = (STATIC / "models.js").read_text(encoding="utf-8")
    run_js = (STATIC / "run.js").read_text(encoding="utf-8")
    assert "run.html?id=" in models_js
    assert 'get("id")' in run_js
    assert "/api/runs/" in run_js


# --------------------------------------------------------------------------- CLI smoke


def test_cli_ui_starts_serves_health_and_stops(tmp_path: Path) -> None:
    if os.name != "posix":
        pytest.skip("The SIGINT CLI smoke test is POSIX-only")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "modelgraph",
            "ui",
            "--port",
            "0",
            "--no-browser",
            "--runs-dir",
            str(tmp_path / "runs"),
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    output: list[str] = []
    url = ""
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            ready, _, _ = select.select([process.stdout], [], [], 0.2)
            if not ready:
                if process.poll() is not None:
                    break
                continue
            line = process.stdout.readline()
            if not line:
                break
            output.append(line)
            match = re.search(r"http://127\.0\.0\.1:\d+/", line)
            if match:
                url = match.group(0)
                break
        assert url, "mg ui did not print a loopback URL:\n" + "".join(output)

        with urllib.request.urlopen(url + "api/health", timeout=10) as response:
            assert response.status == 200
            assert json.loads(response.read().decode("utf-8"))["ok"] is True

        process.send_signal(signal.SIGINT)
        remaining, _ = process.communicate(timeout=15)
        output.append(remaining or "")
        assert process.returncode == 0, "".join(output)
        assert "stopping" in "".join(output)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)
