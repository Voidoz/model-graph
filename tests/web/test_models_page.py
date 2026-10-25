"""Static contract checks for the model catalog and run history page (T-0044)."""

from __future__ import annotations

import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest

STATIC_DIR = Path(__file__).resolve().parents[2] / "src" / "modelgraph" / "web" / "static"
MODELS_HTML = STATIC_DIR / "models.html"
MODELS_JS = STATIC_DIR / "models.js"
MODELS_CSS = STATIC_DIR / "models.css"

REQUIRED_IDS = {
    "main",
    "mg-nav",
    "models-section",
    "models-heading",
    "models-status",
    "models-list",
    "models-empty",
    "models-error",
    "history-section",
    "history-heading",
    "history-status",
    "outcome-filter",
    "history-refresh",
    "history-table-wrap",
    "history-rows",
    "history-empty",
    "history-error",
}

FORBIDDEN_JS_SINKS = (
    "innerHTML",
    "outerHTML",
    "insertAdjacentHTML",
    "document.write",
    "eval(",
    "new Function",
)


class AssetParser(HTMLParser):
    """Collect start tags, their attributes, and element ids from the page."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags: list[tuple[str, dict[str, str]]] = []
        self.ids: set[str] = set()
        self.text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {name.lower(): value or "" for name, value in attrs}
        self.tags.append((tag.lower(), values))
        if "id" in values:
            self.ids.add(values["id"])

    def handle_data(self, data: str) -> None:
        self.text.append(data)


def parse_page() -> AssetParser:
    parser = AssetParser()
    parser.feed(MODELS_HTML.read_text(encoding="utf-8"))
    parser.close()
    return parser


def attributes_by_id(parser: AssetParser) -> dict[str, tuple[str, dict[str, str]]]:
    return {attrs["id"]: (tag, attrs) for tag, attrs in parser.tags if attrs.get("id")}


@pytest.mark.parametrize("path", [MODELS_HTML, MODELS_JS, MODELS_CSS])
def test_required_files_exist(path: Path) -> None:
    assert path.is_file(), f"missing required asset: {path}"
    assert path.stat().st_size > 0, f"empty required asset: {path}"


def test_page_structure_and_required_hooks() -> None:
    parser = parse_page()
    assert parser.ids >= REQUIRED_IDS
    assert all(element_id in parser.ids for element_id in REQUIRED_IDS)

    html_tag = next(attrs for tag, attrs in parser.tags if tag == "html")
    assert html_tag.get("lang") == "en"
    assert 'name="viewport"' in MODELS_HTML.read_text(encoding="utf-8")
    assert "<title>" in MODELS_HTML.read_text(encoding="utf-8")
    assert "Model Graph" in "".join(parser.text)


def test_scripts_styles_and_shared_helpers_are_local() -> None:
    parser = parse_page()
    scripts = [attrs["src"] for tag, attrs in parser.tags if tag == "script"]
    stylesheets = [attrs["href"] for tag, attrs in parser.tags if tag == "link"]

    assert scripts == ["/static/common.js", "/static/models.js"]
    assert stylesheets == ["/static/styles.css", "/static/models.css"]
    assert scripts.index("/static/common.js") < scripts.index("/static/models.js")


def test_empty_and_error_states_are_present() -> None:
    parser = parse_page()
    by_id = attributes_by_id(parser)

    for element_id in ("models-empty", "models-error", "history-empty", "history-error"):
        assert "hidden" in by_id[element_id][1], f"{element_id} must start hidden"

    for element_id in ("models-status", "history-status"):
        tag, attrs = by_id[element_id]
        assert attrs.get("role") == "status"
        assert attrs.get("aria-live") == "polite"

    for element_id in ("models-error", "history-error"):
        assert by_id[element_id][1].get("role") == "alert"


def test_history_controls_are_labelled_and_bounded() -> None:
    parser = parse_page()
    by_id = attributes_by_id(parser)

    labels = [attrs for tag, attrs in parser.tags if tag == "label"]
    assert any(label.get("for") == "outcome-filter" for label in labels)
    assert by_id["outcome-filter"][0] == "select"
    assert by_id["history-refresh"][1].get("type") == "button"
    assert by_id["history-rows"][0] == "tbody"

    js = MODELS_JS.read_text(encoding="utf-8")
    assert "const RUN_HISTORY_LIMIT = 50;" in js
    assert "/api/runs?limit=${RUN_HISTORY_LIMIT}" in js


def test_page_has_no_external_assets_or_embedded_handlers() -> None:
    parser = parse_page()
    url_attributes = {"src", "href", "action", "formaction", "poster", "data"}
    forbidden_embeds = {"img", "iframe", "object", "embed", "video", "audio", "source", "base"}

    for tag, attrs in parser.tags:
        assert tag not in forbidden_embeds, f"unexpected embedded asset element: {tag}"
        for name, value in attrs.items():
            assert not name.startswith("on"), f"inline event handler on <{tag}>: {name}"
            if name in url_attributes:
                lowered = value.strip().lower()
                assert not lowered.startswith(
                    ("http://", "https://", "//", "data:", "javascript:", "ftp:")
                ), f"external or unsafe URL in <{tag} {name}>: {value}"


def test_css_has_no_external_assets() -> None:
    css = MODELS_CSS.read_text(encoding="utf-8").lower()
    assert "@import" not in css
    assert "url(" not in css
    assert "http://" not in css
    assert "https://" not in css


def test_js_uses_shared_helpers_and_no_raw_network_calls() -> None:
    js = MODELS_JS.read_text(encoding="utf-8")
    for helper in ("MG.api(", "MG.el(", "MG.text(", "MG.number(", 'MG.nav("models")'):
        assert helper in js, f"models.js should use the shared helper: {helper}"
    assert "fetch(" not in js
    assert "XMLHttpRequest" not in js
    assert "http://" not in js
    assert "https://" not in js


def test_js_reads_api_fields_defensively() -> None:
    js = MODELS_JS.read_text(encoding="utf-8")
    # The API contract flattens some run manifest fields; older/newer manifests do not.
    for token in (
        "item.usage",
        "usage.total_tokens",
        "usage.cost_usd",
        "item.budget",
        "item.graph",
        "item.graph_id",
        "item.graph_version",
        "item.started_at",
        "firstDefined",
    ):
        assert token in js, f"models.js should read run fields defensively: {token}"

    assert 'MG.api("/api/models")' in js
    assert "run.html?id=" in js
    assert "state.runs.slice().sort" in js


def test_js_has_no_unsafe_html_sinks() -> None:
    js = MODELS_JS.read_text(encoding="utf-8")
    for sink in FORBIDDEN_JS_SINKS:
        assert sink not in js, f"models.js must not use {sink}"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not available")
def test_models_js_syntax_with_node() -> None:
    node = shutil.which("node")
    assert node is not None
    result = subprocess.run(
        [node, "--check", str(MODELS_JS)],
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
