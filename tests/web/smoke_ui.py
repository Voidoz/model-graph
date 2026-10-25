"""Headless HTTP smoke check for the local GUI.

Run with ``uv run python tests/web/smoke_ui.py`` (``make ui-check`` does this after the web
tests). It binds an ephemeral loopback port, requests the pages and core API endpoints, and
prints the URL it used. No browser and no network access are required.
"""

from __future__ import annotations

import json
import tempfile
import threading
import urllib.request
from pathlib import Path
from typing import Any

from modelgraph.web import create_server


def _get(base: str, path: str) -> tuple[int, str, bytes]:
    with urllib.request.urlopen(base + path, timeout=10) as response:
        return response.status, response.headers.get_content_type(), response.read()


def _json(body: bytes) -> dict[str, Any]:
    value = json.loads(body.decode("utf-8"))
    assert isinstance(value, dict), "expected a JSON object"
    return value


def main() -> int:
    with tempfile.TemporaryDirectory() as temp:
        server = create_server("127.0.0.1", 0, runs_dir=Path(temp) / "runs")
        thread = threading.Thread(
            target=server.serve_forever,
            kwargs={"poll_interval": 0.01},
            daemon=True,
        )
        thread.start()
        try:
            host, port = server.server_address[:2]
            base = f"http://{host}:{port}"

            status, content_type, body = _get(base, "/api/health")
            assert status == 200 and content_type == "application/json"
            assert _json(body)["ok"] is True

            _, _, body = _get(base, "/api/graphs")
            assert _json(body)["graphs"], "the default registry returned no graphs"

            _, _, body = _get(base, "/api/models")
            assert _json(body)["models"], "the default registry returned no model manifests"

            _, content_type, body = _get(base, "/")
            assert content_type == "text/html"
            assert b"Model Graph" in body

            _, content_type, body = _get(base, "/static/common.js")
            assert content_type == "text/javascript"
            assert b"MG.api" in body

            print(f"GUI smoke OK: {base}/ (health, graphs, models, dashboard, static asset)")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
