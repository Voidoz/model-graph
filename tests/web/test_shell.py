"""Static checks for the GUI shell, dashboard, and graph canvas.

These tests do not need the HTTP server or a browser. They verify asset
presence, local-only assets, safe DOM rendering, required page hooks, and (when
Node is available) JavaScript syntax plus the pure layout algorithm.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "src" / "modelgraph" / "web" / "static"

INDEX_HTML = STATIC / "index.html"
GRAPH_HTML = STATIC / "graph.html"
STYLES = STATIC / "styles.css"
COMMON_JS = STATIC / "common.js"
INDEX_JS = STATIC / "index.js"
GRAPH_JS = STATIC / "graph.js"

PAGES = (INDEX_HTML, GRAPH_HTML)
JS_FILES = (COMMON_JS, INDEX_JS, GRAPH_JS)
STATIC_FILES = (INDEX_HTML, GRAPH_HTML, STYLES, COMMON_JS, INDEX_JS, GRAPH_JS)

SVG_NAMESPACE = "http://www.w3.org/2000/svg"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def node_binary() -> str:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is not installed; skipping JavaScript checks")
    return node


def run_node(script: str) -> str:
    result = subprocess.run(
        [node_binary(), "-e", script],
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_required_static_files_exist() -> None:
    for path in STATIC_FILES:
        assert path.is_file(), f"missing static asset: {path}"


def test_html_loads_local_styles_and_page_scripts() -> None:
    index = read(INDEX_HTML)
    graph = read(GRAPH_HTML)

    assert '<link rel="stylesheet" href="/static/styles.css"' in index
    assert '<link rel="stylesheet" href="/static/styles.css"' in graph
    assert '<script src="/static/common.js" defer></script>' in index
    assert '<script src="/static/common.js" defer></script>' in graph
    assert '<script src="/static/index.js" defer></script>' in index
    assert '<script src="/static/graph.js" defer></script>' in graph


def test_pages_and_scripts_use_local_urls_only() -> None:
    for path in STATIC_FILES:
        source = read(path)
        checked = source.replace(SVG_NAMESPACE, "")
        assert "http://" not in checked, f"external http URL in {path.name}"
        assert "https://" not in checked, f"external https URL in {path.name}"
        assert "//cdn." not in checked, f"CDN reference in {path.name}"

    assert "url(" not in read(STYLES).lower(), "external URL reference in styles.css"

    for page in PAGES:
        source = read(page)
        for reference in re.findall(r'(?:src|href)="([^"]+)"', source):
            assert not reference.startswith(("http://", "https://", "//")), (
                f"external asset reference {reference!r} in {page.name}"
            )


def test_required_page_hooks_are_present() -> None:
    index = read(INDEX_HTML)
    for element_id in (
        "mg-nav",
        "main",
        "graphs",
        "graphs-status",
        "graphs-error",
        "graphs-empty",
        "graphs-summary",
    ):
        assert re.search(rf'\bid="{re.escape(element_id)}"', index), (
            f"index.html is missing #{element_id}"
        )

    graph = read(GRAPH_HTML)
    for element_id in (
        "mg-nav",
        "main",
        "graph-status",
        "graph-error",
        "graph-empty",
        "graph-content",
        "graph-canvas",
        "graph-legend",
        "node-details",
        "graph-warnings",
        "warning-count",
        "graph-run-link",
    ):
        assert re.search(rf'\bid="{re.escape(element_id)}"', graph), (
            f"graph.html is missing #{element_id}"
        )

    assert "run.html?graph=" in read(GRAPH_JS)
    assert "models.html" in read(COMMON_JS)
    assert "run.html" in read(COMMON_JS)
    assert '"New run", href: "run.html"' in read(COMMON_JS)


def test_shared_mg_helpers_are_exported() -> None:
    common = read(COMMON_JS)
    for helper in (
        "MG.api",
        "MG.el",
        "MG.text",
        "MG.number",
        "MG.nav",
        "MG.showError",
        "MG.showLoading",
        "MG.warningList",
    ):
        assert helper in common, f"common.js does not export {helper}"


def test_no_unsafe_dom_sinks_or_inline_handlers() -> None:
    for path in JS_FILES:
        source = read(path)
        for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval("):
            assert sink not in source, f"unsafe DOM sink {sink!r} in {path.name}"
        assert "new Function" not in source, f"new Function in {path.name}"

    for page in PAGES:
        source = read(page)
        assert not re.search(r"\son[a-z]+\s*=", source, re.IGNORECASE), (
            f"inline event handler in {page.name}"
        )


def test_shell_has_focus_styles_and_responsive_rules() -> None:
    css = read(STYLES)
    assert ":focus-visible" in css
    assert "@media (max-width: 820px)" in css
    assert "@media (max-width: 480px)" in css
    assert "--sans: system-ui" in css
    assert "font-family: var(--sans)" in css


def test_javascript_syntax_when_node_is_available() -> None:
    node = node_binary()
    for path in JS_FILES:
        result = subprocess.run(
            [node, "--check", str(path)],
            capture_output=True,
            check=False,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"{path.name}: {result.stderr}"


def test_svg_layout_handles_small_fixture_when_node_is_available() -> None:
    script = f"""
const api = require({json.dumps(str(GRAPH_JS))});
const graph = {{
  id: "hello-triage",
  entry: "triage",
  exit: ["answer"],
  nodes: {{
    story: {{ id: "story", kind: "worker", model_ref: "story-7b",
              writes: [{{ name: "answer", type: "text" }}] }},
    triage: {{ id: "triage", kind: "router", model_ref: "router-small" }},
    coding: {{ id: "coding", kind: "worker", model_ref: "coder-7b",
              writes: [{{ name: "answer", type: "code" }}] }}
  }},
  edges: [
    {{ from: "triage", to: "coding", when: "Code, debugging, architecture, or implementation." }},
    {{ from: "triage", to: "story", when: "Fiction, narrative, characters, or a poem." }}
  ]
}};
const layout = api.computeLayout(graph);
const long = "The user asks for code, debugging, architecture, or technical implementation.";
process.stdout.write(JSON.stringify({{
  width: layout.width,
  height: layout.height,
  layers: layout.layers,
  positions: layout.positions,
  edgeCount: layout.edges.length,
  shortened: api.shorten(long, 24)
}}));
"""
    payload = json.loads(run_node(script))

    assert payload["layers"] == {"triage": 0, "coding": 1, "story": 1}
    assert payload["edgeCount"] == 2
    assert set(payload["positions"]) == {"triage", "coding", "story"}
    assert payload["positions"]["coding"]["y"] > payload["positions"]["triage"]["y"]
    assert payload["positions"]["coding"]["y"] == payload["positions"]["story"]["y"]
    assert payload["positions"]["coding"]["x"] < payload["positions"]["story"]["x"]
    assert payload["width"] >= 560
    assert payload["height"] > 0
    assert payload["shortened"].endswith("…")
    assert len(payload["shortened"]) <= 24
