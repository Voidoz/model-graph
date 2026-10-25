"""Static contract tests for the GUI run console (T-0043).

These tests do not start a server or a browser. They check that the page is complete, that it
references only local assets, that every element the script looks up exists, and that the script
never pipes API data into an HTML sink. A JavaScript syntax check runs when Node is available.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "src" / "modelgraph" / "web" / "static"
RUN_HTML = STATIC / "run.html"
RUN_JS = STATIC / "run.js"
RUN_CSS = STATIC / "run.css"

REQUIRED_IDS = {
    "mg-nav",
    "main",
    "run-form",
    "run-fields",
    "graph-select",
    "graph-help",
    "task-input",
    "example-task",
    "provider-select",
    "provider-help",
    "run-button",
    "run-status",
    "form-error",
    "error-panel",
    "error-heading",
    "error-code",
    "error-context",
    "error-message",
    "error-hint",
    "result-panel",
    "result-outcome",
    "result-run-id",
    "result-graph",
    "result-provider",
    "result-run-dir",
    "run-dir-link",
    "copy-run-dir",
    "manifest-link",
    "raw-trace-link",
    "run-error",
    "usage-list",
    "path-timeline",
    "artifacts",
    "trace-status",
    "trace-note",
    "trace-table",
    "trace-body",
    "replay-button",
    "replay-status",
    "replay-result",
    "replay-verdict",
    "replay-detail",
    "replay-meta",
    "replay-mismatches",
}


class PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: list[str] = []
        self.duplicate_ids: list[str] = []
        self._seen_ids: set[str] = set()
        self.script_srcs: list[str] = []
        self.stylesheet_hrefs: list[str] = []
        self.link_hrefs: list[str] = []
        self.inline_scripts = 0
        self.inline_styles = 0
        self.inline_handlers: list[str] = []
        self.attributes: dict[str, dict[str, str | None]] = {}
        self.select_options: dict[str, list[dict[str, object]]] = {}
        self._select_stack: list[str | None] = []
        self.has_doctype = False
        self.lang: str | None = None

    def handle_decl(self, decl: str) -> None:
        if decl.lower().startswith("doctype"):
            self.has_doctype = True

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "html":
            self.lang = values.get("lang")
        self._record_ids(tag, values)
        self._record_assets(tag, attrs, values)
        self._record_form_controls(tag, values)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if tag == "select" and self._select_stack:
            self._select_stack.pop()

    def _record_ids(self, tag: str, values: dict[str, str | None]) -> None:
        ident = values.get("id")
        if not ident:
            return
        if ident in self._seen_ids:
            self.duplicate_ids.append(ident)
        self._seen_ids.add(ident)
        self.ids.append(ident)
        self.attributes[ident] = values

    def _record_assets(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
        values: dict[str, str | None],
    ) -> None:
        for name, value in attrs:
            if name.startswith("on") and value is not None:
                self.inline_handlers.append(f"{tag}[{name}]")
        if tag == "script":
            src = values.get("src")
            if src:
                self.script_srcs.append(src)
            else:
                self.inline_scripts += 1
        if tag == "style":
            self.inline_styles += 1
        if tag == "link":
            if values.get("rel") == "stylesheet":
                self.stylesheet_hrefs.append(values.get("href") or "")
            self.link_hrefs.append(values.get("href") or "")
        if tag == "a" and values.get("href"):
            self.link_hrefs.append(values.get("href") or "")

    def _record_form_controls(self, tag: str, values: dict[str, str | None]) -> None:
        if tag == "select":
            self._select_stack.append(values.get("id"))
            if values.get("id"):
                self.select_options.setdefault(values["id"], [])
        elif tag == "option" and self._select_stack and self._select_stack[-1]:
            self.select_options[self._select_stack[-1]].append(
                {
                    "value": values.get("value"),
                    "selected": "selected" in values,
                }
            )


@pytest.fixture(scope="module")
def page() -> PageParser:
    parser = PageParser()
    parser.feed(RUN_HTML.read_text(encoding="utf-8"))
    parser.close()
    return parser


def test_run_assets_exist() -> None:
    for path in (RUN_HTML, RUN_JS, RUN_CSS):
        assert path.is_file(), f"missing run console asset: {path}"
        assert path.stat().st_size > 500, f"run console asset is suspiciously small: {path}"


def test_html_is_a_complete_local_page(page: PageParser) -> None:
    assert page.has_doctype
    assert page.lang == "en"
    assert 'href="/static/styles.css"' in RUN_HTML.read_text(encoding="utf-8")
    assert "/static/common.js" in page.script_srcs
    assert "/static/run.js" in page.script_srcs
    assert "/static/styles.css" in page.stylesheet_hrefs
    assert "/static/run.css" in page.stylesheet_hrefs
    assert page.inline_scripts == 0, "run.html must not contain inline scripts"
    assert page.inline_styles == 0, "run.html must not contain inline styles"
    assert page.inline_handlers == [], "run.html must not use inline event handlers"


def test_html_has_required_ids_without_duplicates(page: PageParser) -> None:
    missing = sorted(REQUIRED_IDS - set(page.ids))
    assert not missing, f"run.html is missing required ids: {missing}"
    assert not page.duplicate_ids, f"duplicate ids: {page.duplicate_ids}"


def test_provider_select_defaults_to_mock(page: PageParser) -> None:
    options = page.select_options.get("provider-select", [])
    assert [option["value"] for option in options] == ["mock", "auto"]
    assert options[0]["selected"] is True
    assert options[1]["selected"] is False


def test_graph_and_task_are_required(page: PageParser) -> None:
    assert "required" in page.attributes["graph-select"]
    assert "required" in page.attributes["task-input"]
    assert "disabled" in page.attributes["graph-select"]


def test_all_referenced_urls_are_local(page: PageParser) -> None:
    refs = [*page.script_srcs, *page.stylesheet_hrefs, *page.link_hrefs]
    for ref in refs:
        assert ref, "empty asset reference"
        if ref == "data:,":
            continue
        assert not ref.startswith(("http://", "https://", "//", "data:", "ftp:")), (
            f"external asset reference in run.html: {ref}"
        )


def test_script_references_only_existing_element_ids(page: PageParser) -> None:
    source = RUN_JS.read_text(encoding="utf-8")
    looked_up = set(re.findall(r"""(?:byId|getElementById)\(\s*["']([^"']+)["']\s*\)""", source))
    missing = sorted(looked_up - set(page.ids))
    assert not missing, f"run.js looks up ids missing from run.html: {missing}"


def test_script_uses_the_shared_ui_helpers() -> None:
    source = RUN_JS.read_text(encoding="utf-8")
    assert "MG.api(" in source
    assert "MG.el(" in source
    assert "MG.nav(" in source
    assert ".textContent" in source


def test_script_covers_the_run_api_contract() -> None:
    source = RUN_JS.read_text(encoding="utf-8")
    for endpoint in (
        '"/api/graphs"',
        '"/api/runs"',
        '"/api/runs/" + encodeURIComponent(runId) + "/trace"',
        '"/api/runs/" + encodeURIComponent(runId) + "/replay"',
        '"/artifacts/"',
    ):
        assert endpoint in source, f"run.js does not call {endpoint}"
    for field in ("graph_id", "task", "provider", "run_id", "run_dir", "outcome", "usage", "path"):
        assert field in source, f"run.js does not handle the {field!r} field"
    for status in ("400", "404", "409", "500", "502"):
        assert status in source, f"run.js does not handle HTTP {status}"


def test_script_has_no_html_injection_or_eval_sinks() -> None:
    source = RUN_JS.read_text(encoding="utf-8")
    for sink in (
        "innerHTML",
        "outerHTML",
        "insertAdjacentHTML",
        "document.write",
        "eval(",
        "new Function(",
    ):
        assert sink not in source, f"run.js must not use {sink}"
    assert not re.search(r"\balert\s*\(", source), "run.js must not use alert()"


def test_css_is_self_contained() -> None:
    source = RUN_CSS.read_text(encoding="utf-8")
    assert "@import" not in source
    assert "http://" not in source
    assert "https://" not in source
    assert "url(" not in source


def test_no_external_network_assets_in_any_asset() -> None:
    for path in (RUN_HTML, RUN_JS, RUN_CSS):
        source = path.read_text(encoding="utf-8")
        assert "https://" not in source, f"{path.name} references an external URL"
        assert "http://" not in source, f"{path.name} references an external URL"
        assert "cdn." not in source.lower(), f"{path.name} references a CDN"


def test_javascript_syntax_with_node() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    completed = subprocess.run(
        [node, "--check", str(RUN_JS)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
