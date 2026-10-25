/* Shared helpers for the local Model Graph GUI.
 *
 * No framework, no build step, no network calls except the local JSON API.
 * API-derived strings are rendered with text nodes only; this file never
 * writes markup strings or evaluates code.
 */
(function (global) {
  "use strict";

  var MG = global.MG || {};

  /**
   * Error raised by MG.api. `code` is the API error code when the server sent
   * one, otherwise a local MG-* code.
   */
  class ApiError extends Error {
    constructor(code, message, status) {
      super(message);
      this.name = "ApiError";
      this.code = code || "MG-UNKNOWN";
      this.status = typeof status === "number" ? status : 0;
    }
  }

  function appendChildren(parent, children) {
    children.forEach(function (child) {
      if (child === null || child === undefined || child === false) return;
      if (Array.isArray(child)) {
        appendChildren(parent, child);
        return;
      }
      if (typeof child === "string" || typeof child === "number" || typeof child === "bigint") {
        parent.appendChild(document.createTextNode(String(child)));
        return;
      }
      if (child && typeof child.nodeType === "number") {
        parent.appendChild(child);
      }
    });
  }

  /**
   * Create an element.
   *
   * attrs: `class`, `text`, `dataset`, `style` (object), `on*` listeners, or a
   * plain attribute. `text` and string children are inserted as text nodes.
   */
  function el(tag, attrs) {
    const element = document.createElement(tag);
    if (attrs && typeof attrs === "object") {
      Object.keys(attrs).forEach(function (key) {
        const value = attrs[key];
        if (value === null || value === undefined || value === false) return;
        if (key === "class" || key === "className") {
          element.className = String(value);
        } else if (key === "text") {
          element.textContent = String(value);
        } else if (key === "dataset" && typeof value === "object") {
          Object.keys(value).forEach(function (dataKey) {
            element.dataset[dataKey] = String(value[dataKey]);
          });
        } else if (key === "style" && typeof value === "object") {
          Object.keys(value).forEach(function (styleKey) {
            element.style[styleKey] = String(value[styleKey]);
          });
        } else if (key.indexOf("on") === 0 && typeof value === "function") {
          element.addEventListener(key.slice(2).toLowerCase(), value);
        } else if (value === true) {
          element.setAttribute(key, "");
        } else {
          element.setAttribute(key, String(value));
        }
      });
    }
    appendChildren(element, Array.prototype.slice.call(arguments, 2));
    return element;
  }

  /** Format untrusted API text for display. Never returns markup. */
  function text(value) {
    if (value === null || value === undefined) return "—";
    if (typeof value === "string") {
      const trimmed = value.trim();
      return trimmed ? value : "—";
    }
    if (typeof value === "number") return number(value);
    if (typeof value === "boolean") return value ? "Yes" : "No";
    if (Array.isArray(value)) {
      const parts = value.map(text).filter(function (part) {
        return part !== "—";
      });
      return parts.length ? parts.join(", ") : "—";
    }
    return String(value);
  }

  /** Format a finite number with the browser locale; non-numbers become "—". */
  function number(value, options) {
    const parsed = typeof value === "number" ? value : Number(value);
    if (!Number.isFinite(parsed)) return "—";
    const formatOptions = Object.assign(
      { maximumFractionDigits: 2 },
      options && typeof options === "object" ? options : {},
    );
    try {
      return new Intl.NumberFormat(undefined, formatOptions).format(parsed);
    } catch (error) {
      return String(parsed);
    }
  }

  /** Shorten long identifiers/hashes without losing the original in `title`. */
  function short(value, maxLength) {
    if (value === null || value === undefined) return "—";
    const raw = String(value);
    const limit = Number.isFinite(maxLength) && maxLength > 4 ? maxLength : 18;
    if (raw.length <= limit) return raw;
    return raw.slice(0, limit - 1) + "…";
  }

  /** Warnings may be strings or structured objects; render a safe one-line label. */
  function warningText(value) {
    if (typeof value === "string") return value;
    if (value && typeof value === "object") {
      const code = value.code || value.id || value.rule;
      const message = value.message || value.text || value.detail;
      if (code && message) return "[" + String(code) + "] " + String(message);
      if (message) return String(message);
      if (code) return String(code);
      try {
        return JSON.stringify(value);
      } catch (error) {
        return "Unrecognized warning";
      }
    }
    return text(value);
  }

  /** Build a warning list (or a muted "none" line) from an API warning array. */
  function warningList(warnings) {
    const items = Array.isArray(warnings) ? warnings : [];
    if (!items.length) return el("p", { class: "muted", text: "No warnings." });
    const list = el("ul", { class: "warning-list" });
    items.forEach(function (warning) {
      list.appendChild(el("li", { text: warningText(warning) }));
    });
    return list;
  }

  /** Fetch JSON from the local API and normalize failures to ApiError. */
  async function api(path, options) {
    const request = Object.assign({}, options || {});
    const headers = Object.assign({ Accept: "application/json" }, request.headers || {});
    if (request.body !== undefined && request.body !== null && typeof request.body !== "string") {
      headers["Content-Type"] = "application/json";
      request.body = JSON.stringify(request.body);
    }
    request.headers = headers;

    let response;
    try {
      response = await fetch(path, request);
    } catch (error) {
      throw new ApiError(
        "MG-NETWORK",
        "Cannot reach the local Model Graph server. Is `mg ui` running?",
        0,
      );
    }

    let raw = "";
    try {
      raw = await response.text();
    } catch (error) {
      throw new ApiError("MG-RESPONSE", "The server response could not be read.", response.status);
    }

    let payload = null;
    if (raw) {
      try {
        payload = JSON.parse(raw);
      } catch (error) {
        throw new ApiError(
          "MG-BAD-JSON",
          "The server returned a response that was not JSON (HTTP " + response.status + ").",
          response.status,
        );
      }
    }

    if (!response.ok) {
      const envelope = payload && typeof payload === "object" ? payload.error : null;
      const code = envelope && envelope.code ? envelope.code : "HTTP-" + response.status;
      const message =
        envelope && envelope.message
          ? envelope.message
          : "The request failed with HTTP " + response.status + ".";
      throw new ApiError(code, message, response.status);
    }
    if (payload === null) {
      throw new ApiError("MG-EMPTY", "The server returned an empty response.", response.status);
    }
    return payload;
  }

  const NAV_ITEMS = [
    { id: "dashboard", label: "Dashboard", href: "index.html" },
    { id: "models", label: "Models & runs", href: "models.html" },
    { id: "run", label: "New run", href: "run.html" },
  ];

  /** Render the shared top navigation into #mg-nav. */
  function nav(active) {
    const container = typeof document === "undefined" ? null : document.getElementById("mg-nav");
    if (!container) return null;

    container.textContent = "";
    const brand = el(
      "a",
      { class: "brand", href: "index.html", "aria-label": "Model Graph dashboard" },
      el("span", { class: "brand-mark", "aria-hidden": "true" }),
      el("span", { class: "brand-name", text: "Model Graph" }),
    );
    const links = el(
      "nav",
      { class: "nav-links", "aria-label": "Primary" },
      el(
        "ul",
        { class: "nav-list" },
        NAV_ITEMS.map(function (item) {
          const isActive =
            active === item.id || (active === "graph" && item.id === "dashboard");
          const attrs = { class: "nav-link", href: item.href, text: item.label };
          if (isActive) attrs["aria-current"] = "page";
          return el("li", {}, el("a", attrs));
        }),
      ),
    );
    container.appendChild(
      el(
        "div",
        { class: "nav-inner" },
        brand,
        links,
        el("span", { class: "nav-local", text: "local only" }),
      ),
    );
    return container;
  }

  function stateContainer(container) {
    if (!container) return null;
    container.hidden = false;
    container.textContent = "";
    return container;
  }

  /** Show a compact loading state inside an existing status container. */
  function showLoading(container, message) {
    const target = stateContainer(container);
    if (!target) return;
    target.appendChild(el("span", { class: "spinner", "aria-hidden": "true" }));
    target.appendChild(el("span", { text: message || "Loading…" }));
  }

  /** Render an API or contract error, optionally with a retry button. */
  function showError(container, error, options) {
    const target = stateContainer(container);
    if (!target) return;
    const opts = options || {};
    const code = error && error.code ? String(error.code) : "MG-ERROR";
    const message =
      error && error.message ? String(error.message) : "Something unexpected went wrong.";

    target.appendChild(
      el("h2", { class: "state-title", text: opts.title || "Could not load this page" }),
    );
    target.appendChild(
      el(
        "p",
        { class: "error-message" },
        el("code", { class: "error-code", text: code }),
        el("span", { text: message }),
      ),
    );
    if (typeof opts.onRetry === "function") {
      target.appendChild(
        el("button", { class: "button secondary", type: "button", onClick: opts.onRetry }, "Try again"),
      );
    }
  }

  /** Render an empty state. `action` may be a link built by the caller. */
  function showEmpty(container, title, message, action) {
    const target = stateContainer(container);
    if (!target) return;
    target.appendChild(el("h2", { class: "state-title", text: title || "Nothing here yet" }));
    if (message) target.appendChild(el("p", { class: "state-message", text: message }));
    if (action) target.appendChild(action);
  }

  function ready(callback) {
    if (typeof document === "undefined") return;
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", callback, { once: true });
    } else {
      callback();
    }
  }

  MG.ApiError = ApiError;
  MG.api = api;
  MG.el = el;
  MG.text = text;
  MG.number = number;
  MG.short = short;
  MG.warningText = warningText;
  MG.warningList = warningList;
  MG.nav = nav;
  MG.showLoading = showLoading;
  MG.showError = showError;
  MG.showEmpty = showEmpty;
  MG.ready = ready;

  global.MG = MG;
  if (typeof module !== "undefined" && module.exports) module.exports = MG;
})(typeof globalThis !== "undefined" ? globalThis : this);
