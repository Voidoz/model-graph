"""Optional OpenTelemetry export: absent path, allowlisted spans, and adapter shape."""

from __future__ import annotations

import importlib.util
import json
from typing import Any

import pytest

from modelgraph import observability
from modelgraph.errors import ModelGraphError, RuntimeFailure
from modelgraph.observability import (
    NODE_SPAN_NAME,
    RUN_SPAN_NAME,
    NullSpanExporter,
    OpenTelemetrySpanExporter,
    RecordingSpanExporter,
    SpanRecord,
    build_span_exporter,
    node_span,
    opentelemetry_available,
    run_span,
    safe_span_attributes,
)


def test_core_path_works_when_opentelemetry_is_absent(monkeypatch) -> None:
    real_find_spec = importlib.util.find_spec
    real_import = importlib.import_module

    def fake_find_spec(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "opentelemetry" or name.startswith("opentelemetry."):
            return None
        return real_find_spec(name, *args, **kwargs)

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "opentelemetry.trace" or name.startswith("opentelemetry."):
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(observability.importlib, "import_module", fake_import)

    assert opentelemetry_available() is False
    assert isinstance(build_span_exporter({}), NullSpanExporter)
    assert isinstance(build_span_exporter({"MG_OTEL": "off"}), NullSpanExporter)
    with pytest.raises(ModelGraphError) as excinfo:
        build_span_exporter({"MG_OTEL": "on"})
    assert excinfo.value.code == "MG-5000"
    assert "opentelemetry-api" in excinfo.value.message


def test_exporter_is_off_by_default() -> None:
    assert isinstance(build_span_exporter({}), NullSpanExporter)
    assert isinstance(build_span_exporter({"MG_OTEL_ENABLED": "0"}), NullSpanExporter)


def test_unknown_otel_mode_is_rejected() -> None:
    with pytest.raises(ModelGraphError) as excinfo:
        build_span_exporter({"MG_OTEL": "collector-please"})
    assert excinfo.value.code == "MG-5000"


def test_span_attributes_allowlist_drops_content() -> None:
    safe = safe_span_attributes(
        {
            "run_id": "run_1",
            "node": "triage",
            "depth": 0,
            "prompt": "SECRET PROMPT",
            "artifact": "SECRET ARTIFACT",
            "text": "SECRET TEXT",
            "api_key": "sk-secret",
            "messages": [{"content": "SECRET"}],
            "nested": {"anything": "dropped"},
        }
    )
    assert safe == {"run_id": "run_1", "node": "triage", "depth": 0}
    assert "SECRET" not in json.dumps(safe)


def test_recording_exporter_run_and_node_spans_have_ids_and_timings() -> None:
    exporter = RecordingSpanExporter()
    with (
        run_span(
            exporter,
            "run_1",
            graph_id="hello-triage",
            graph_hash="sha256:" + "0" * 64,
            target="portable",
            profile="local-8gb",
            # Content-shaped arguments must be dropped, not exported.
            prompt="SECRET PROMPT",
            task="SECRET TASK",
            artifact="SECRET ARTIFACT",
        ),
        node_span(
            exporter,
            "run_1",
            "triage",
            kind="router",
            depth=0,
            step=1,
            provider="mock",
            model="router-small",
            content="SECRET CONTENT",
        ),
    ):
        pass

    assert [span.name for span in exporter.spans] == [RUN_SPAN_NAME, NODE_SPAN_NAME]
    run_record, node_record = exporter.spans
    assert run_record.attributes["run_id"] == "run_1"
    assert run_record.attributes["graph_id"] == "hello-triage"
    assert run_record.attributes["target"] == "portable"
    assert node_record.attributes["node"] == "triage"
    assert node_record.attributes["kind"] == "router"
    assert node_record.attributes["depth"] == 0
    assert node_record.attributes["provider"] == "mock"
    assert all(span.status == "ok" for span in exporter.spans)
    assert all(span.duration_seconds is not None for span in exporter.spans)
    assert all(span.duration_seconds >= 0.0 for span in exporter.spans if span.duration_seconds)
    assert all(span.finished_at is not None for span in exporter.spans)
    serialized = json.dumps([span.attributes for span in exporter.spans])
    assert "SECRET" not in serialized
    assert "prompt" not in serialized
    assert "artifact" not in serialized


def test_error_span_records_status_and_code() -> None:
    exporter = RecordingSpanExporter()
    with pytest.raises(RuntimeFailure), node_span(exporter, "run_1", "worker", kind="worker"):
        raise RuntimeFailure("bad artifact", details={"code": "MG-2202"})
    record = exporter.spans[0]
    assert isinstance(record, SpanRecord)
    assert record.status == "error"
    assert record.error_code == "MG-2202"


class _FakeSpan:
    def __enter__(self) -> _FakeSpan:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None


class _FakeTracer:
    """An OpenTelemetry-shaped tracer: no SDK and no collector required."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def start_as_current_span(
        self, name: str, *, attributes: dict[str, Any] | None = None
    ) -> _FakeSpan:
        self.calls.append((name, attributes or {}))
        return _FakeSpan()


def test_otel_adapter_uses_a_tracer_shape_without_the_dependency() -> None:
    tracer = _FakeTracer()
    exporter = OpenTelemetrySpanExporter(tracer)
    with run_span(exporter, "run_otel", graph_id="hello-triage", prompt="SECRET PROMPT"):
        pass
    with node_span(exporter, "run_otel", "triage", kind="router", text="SECRET TEXT"):
        pass

    assert [name for name, _ in tracer.calls] == [RUN_SPAN_NAME, NODE_SPAN_NAME]
    assert tracer.calls[0][1]["run_id"] == "run_otel"
    assert tracer.calls[0][1]["graph_id"] == "hello-triage"
    assert tracer.calls[1][1]["node"] == "triage"
    assert "SECRET" not in json.dumps([attributes for _, attributes in tracer.calls])


def test_real_otel_is_absent_or_explicitly_skipped() -> None:
    """The core path is the one exercised in CI; no collector or SDK is required."""
    if opentelemetry_available():
        pytest.skip("OpenTelemetry is installed; the absent path is covered by simulation")
    with pytest.raises(ModelGraphError):
        OpenTelemetrySpanExporter.from_environment()
