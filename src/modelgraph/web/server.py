"""Local stdlib HTTP server for the Model Graph GUI.

The server is intentionally small: it serves the JSON API and static frontend files over
loopback and never accepts filesystem paths from the browser. All run, graph, and model
semantics come from the existing registry, compiler, runtime, and replay modules.
"""

from __future__ import annotations

import contextlib
import ipaddress
import json
import logging
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast
from urllib.parse import unquote, urlsplit

from modelgraph import __version__
from modelgraph.paths import runs_dir as default_runs_dir
from modelgraph.registry import Registry, default_registry
from modelgraph.web.api import (
    INVALID_REQUEST,
    METHOD_NOT_ALLOWED,
    NOT_FOUND,
    REQUEST_TOO_LARGE,
    ApiError,
    WebContext,
    error_payload,
    handle_api,
)

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_BODY_BYTES = 1_000_000

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".png": "image/png",
    ".webp": "image/webp",
    ".woff2": "font/woff2",
}

_MISSING_UI_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>UI not found · Model Graph</title>
</head>
<body>
<h1>The GUI assets were not found</h1>
<p>This server is running, but its static frontend directory is missing or incomplete.</p>
<p>Reinstall the package or run <code>mg ui</code> from a source checkout, then
<a href="/">return to the dashboard</a>.</p>
</body>
</html>
""".encode()


def is_loopback_host(host: str) -> bool:
    """True for localhost and loopback IP literals, including bracketed IPv6."""
    value = host.strip().strip("[]").rstrip(".").lower()
    if not value:
        return False
    if value == "localhost":
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


class ModelGraphHTTPServer(ThreadingHTTPServer):
    """A threaded HTTP server bound to one local registry and runs directory."""

    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        context: WebContext,
        static_dir: Path,
    ) -> None:
        super().__init__(server_address, _RequestHandler)
        self.context = context
        self.static_dir = static_dir


def create_server(
    host: str = "127.0.0.1",
    port: int = 0,
    runs_dir: str | Path | None = None,
    *,
    registry: Registry | None = None,
    static_dir: str | Path | None = None,
) -> ModelGraphHTTPServer:
    """Bind a loopback HTTP server and return it without serving.

    ``port=0`` asks the OS for a free port; read the chosen value from
    ``server.server_address``. The caller owns the lifecycle and must call
    ``shutdown()`` / ``server_close()``.
    """
    if not is_loopback_host(host):
        raise ValueError(f"refusing to bind {host!r}: the GUI server is loopback-only")
    context = WebContext(
        registry=registry if registry is not None else default_registry(),
        runs_dir=Path(runs_dir) if runs_dir is not None else default_runs_dir(),
    )
    static_root = Path(static_dir) if static_dir is not None else STATIC_DIR
    return ModelGraphHTTPServer((host, port), context, static_root)


class _RequestHandler(BaseHTTPRequestHandler):
    server_version = f"modelgraph/{__version__}"
    protocol_version = "HTTP/1.1"

    @property
    def _app(self) -> ModelGraphHTTPServer:
        return cast(ModelGraphHTTPServer, self.server)

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler name
        self._dispatch("GET")

    def do_HEAD(self) -> None:  # noqa: N802 - stdlib handler name
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler name
        self._dispatch("POST")

    def do_PUT(self) -> None:  # noqa: N802 - stdlib handler name
        self._method_not_allowed()

    def do_PATCH(self) -> None:  # noqa: N802 - stdlib handler name
        self._method_not_allowed()

    def do_DELETE(self) -> None:  # noqa: N802 - stdlib handler name
        self._method_not_allowed()

    def do_OPTIONS(self) -> None:  # noqa: N802 - stdlib handler name
        self._method_not_allowed()

    def _method_not_allowed(self) -> None:
        self._send_json(
            405,
            error_payload(METHOD_NOT_ALLOWED, f"method {self.command} is not allowed"),
        )

    def _dispatch(self, method: str) -> None:
        path = urlsplit(self.path).path
        is_api = path == "/api" or path.startswith("/api/")
        if is_api:
            try:
                body = self._read_body()
            except ApiError as exc:
                self._send_json(exc.status, error_payload(exc.code, exc.message))
                return
            response = handle_api(self._app.context, method, self.path, body)
            self._send_json(response.status, response.payload)
            return
        if method != "GET":
            self._send_json(
                405,
                error_payload(METHOD_NOT_ALLOWED, "static assets are read-only"),
            )
            return
        self._serve_static()

    def _read_body(self) -> bytes:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            return b""
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ApiError(400, INVALID_REQUEST, "invalid Content-Length header") from exc
        if length < 0:
            raise ApiError(400, INVALID_REQUEST, "invalid Content-Length header")
        if length > MAX_BODY_BYTES:
            raise ApiError(413, REQUEST_TOO_LARGE, "request body is too large")
        return self.rfile.read(length)

    def _serve_static(self) -> None:
        request_path = unquote(urlsplit(self.path).path)
        resolved = _resolve_static(self._app.static_dir, request_path)
        if resolved is None:
            self._send_missing_asset()
            return
        try:
            body = resolved.read_bytes()
        except OSError:
            self._send_missing_asset()
            return
        self._send(200, body, _content_type(resolved))

    def _send_missing_asset(self) -> None:
        # A browser navigating to a missing page gets a readable page; API clients and asset
        # requests keep the JSON envelope. Neither form reflects the request path.
        accept = self.headers.get("Accept", "")
        if "text/html" in accept:
            self._send(404, _MISSING_UI_HTML, "text/html; charset=utf-8")
        else:
            self._send_json(404, error_payload(NOT_FOUND, "asset not found"))

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        # The frontend has no inline scripts/styles, no external assets, and only calls its
        # own origin. Keep it that way at the browser boundary as well as in review.
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
            "connect-src 'self'; font-src 'self'; object-src 'none'; base-uri 'none'; "
            "form-action 'self'; frame-ancestors 'none'",
        )
        self.end_headers()
        if self.command == "HEAD":
            return
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):
            self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        # Request bodies (which contain user prompts) are never logged at info level.
        logger.debug("%s - %s", self.address_string(), format % args)


def _resolve_static(root: Path, request_path: str) -> Path | None:
    """Map a URL path to a file under ``root``, rejecting traversal and hidden files."""
    if "\x00" in request_path or "\\" in request_path:
        return None
    if request_path in {"", "/"}:
        relative = "index.html"
    else:
        trimmed = request_path.lstrip("/")
        if trimmed.startswith("static/"):
            trimmed = trimmed[len("static/") :]
        if not trimmed:
            return None
        parts = trimmed.split("/")
        if any(not part or part in {".", ".."} or part.startswith(".") for part in parts):
            return None
        relative = "/".join(parts)

    try:
        root_resolved = root.resolve()
        candidate = (root_resolved / relative).resolve()
    except OSError:
        return None
    if not candidate.is_relative_to(root_resolved):
        return None
    if not candidate.is_file():
        return None
    return candidate


def _content_type(path: Path) -> str:
    override = _CONTENT_TYPES.get(path.suffix.lower())
    if override is not None:
        return override
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"
