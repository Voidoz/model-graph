/*
 * Run console: run a compiled graph, then inspect the path, artifacts, usage,
 * raw trace, and replay result. All API-derived text is inserted as text
 * content; this file never builds markup from an API response.
 */
(function () {
  "use strict";

  var MG = window.MG;
  var MAX_TRACE_ROWS = 1000;
  var state = {
    graphs: [],
    running: false,
    replaying: false,
    currentRun: null,
    timer: null,
    startedAt: 0,
    requestedMissing: "",
    linkedGraphId: "",
  };
  var els = {};

  function byId(id) {
    return document.getElementById(id);
  }

  function init() {
    if (
      !MG ||
      typeof MG.api !== "function" ||
      typeof MG.el !== "function" ||
      typeof MG.nav !== "function"
    ) {
      var status = byId("run-status");
      if (status) {
        status.textContent =
          "The shared UI library (common.js) did not load. Reload the page; if this persists, " +
          "verify that mg ui is serving the static assets.";
      }
      return;
    }

    cacheElements();
    try {
      MG.nav("run");
    } catch (err) {
      /* Navigation chrome is not required for the console to work. */
    }
    bindEvents();
    loadGraphs();
    loadLinkedRun();
  }

  function cacheElements() {
    els.form = byId("run-form");
    els.fields = byId("run-fields");
    els.graphSelect = byId("graph-select");
    els.graphHelp = byId("graph-help");
    els.taskInput = byId("task-input");
    els.exampleTask = byId("example-task");
    els.providerSelect = byId("provider-select");
    els.providerHelp = byId("provider-help");
    els.runButton = byId("run-button");
    els.runStatus = byId("run-status");
    els.formError = byId("form-error");

    els.errorPanel = byId("error-panel");
    els.errorHeading = byId("error-heading");
    els.errorCode = byId("error-code");
    els.errorContext = byId("error-context");
    els.errorMessage = byId("error-message");
    els.errorHint = byId("error-hint");

    els.resultPanel = byId("result-panel");
    els.resultOutcome = byId("result-outcome");
    els.resultRunId = byId("result-run-id");
    els.resultGraph = byId("result-graph");
    els.resultProvider = byId("result-provider");
    els.resultTask = byId("result-task");
    els.resultRunDir = byId("result-run-dir");
    els.runDirLink = byId("run-dir-link");
    els.copyRunDir = byId("copy-run-dir");
    els.manifestLink = byId("manifest-link");
    els.rawTraceLink = byId("raw-trace-link");
    els.runError = byId("run-error");
    els.usageList = byId("usage-list");
    els.pathTimeline = byId("path-timeline");
    els.artifacts = byId("artifacts");
    els.traceStatus = byId("trace-status");
    els.traceNote = byId("trace-note");
    els.traceBody = byId("trace-body");
    els.replayButton = byId("replay-button");
    els.replayStatus = byId("replay-status");
    els.replayResult = byId("replay-result");
    els.replayVerdict = byId("replay-verdict");
    els.replayDetail = byId("replay-detail");
    els.replayMeta = byId("replay-meta");
    els.replayMismatches = byId("replay-mismatches");
  }

  function bindEvents() {
    els.form.addEventListener("submit", handleSubmit);
    els.graphSelect.addEventListener("change", function () {
      state.requestedMissing = "";
      updateGraphHelp();
    });
    els.providerSelect.addEventListener("change", updateProviderHelp);
    els.exampleTask.addEventListener("click", useExampleTask);
    els.copyRunDir.addEventListener("click", copyRunDir);
    els.replayButton.addEventListener("click", handleReplay);
    window.addEventListener("pagehide", stopTimer);
  }

  /* ---------------------------------------------------------------- helpers */

  function make(tag, attrs, text) {
    var element = MG.el(tag, attrs || {});
    var className = attrs && attrs["class"];
    if (className && element.classList) {
      String(className)
        .split(/\s+/)
        .forEach(function (name) {
          if (name) {
            element.classList.add(name);
          }
        });
    }
    if (text !== undefined && text !== null) {
      element.textContent = String(text);
    }
    return element;
  }

  function clear(element) {
    while (element.firstChild) {
      element.removeChild(element.firstChild);
    }
  }

  function display(value, fallback) {
    if (value === null || value === undefined || value === "") {
      return fallback || "—";
    }
    if (typeof value === "string") {
      return value;
    }
    if (typeof value === "number" || typeof value === "boolean") {
      return String(value);
    }
    try {
      return JSON.stringify(value);
    } catch (err) {
      return String(value);
    }
  }

  function shortText(value, limit) {
    var text = display(value);
    if (text.length <= limit) {
      return text;
    }
    return text.slice(0, limit - 1) + "…";
  }

  function toNumber(value) {
    if (typeof value === "number" && isFinite(value)) {
      return value;
    }
    if (typeof value === "string" && value.trim() !== "" && isFinite(Number(value))) {
      return Number(value);
    }
    return null;
  }

  function formatCount(value) {
    var number = toNumber(value);
    if (number === null) {
      return "—";
    }
    return Math.round(number).toLocaleString();
  }

  function formatCost(value) {
    var number = toNumber(value);
    if (number === null) {
      return "—";
    }
    return "$" + number.toFixed(6);
  }

  function formatSeconds(value) {
    var number = toNumber(value);
    if (number === null) {
      return "—";
    }
    return number.toFixed(2) + " s";
  }

  function formatConfidence(value) {
    var number = toNumber(value);
    if (number === null) {
      return "—";
    }
    number = Math.max(0, Math.min(1, number));
    return Math.round(number * 100) + "%";
  }

  function formatTimestamp(value) {
    if (typeof value === "string" && value) {
      return value.replace("T", " ").replace("Z", " UTC");
    }
    if (typeof value === "number" && isFinite(value)) {
      return new Date(value).toISOString();
    }
    return "—";
  }

  function errorStatus(err) {
    if (!err || typeof err !== "object") {
      return null;
    }
    var raw = err.status;
    if (raw === undefined || raw === null) {
      raw = err.statusCode;
    }
    if (raw === undefined || raw === null) {
      raw = err.response && err.response.status;
    }
    var number = Number(raw);
    if (!isFinite(number)) {
      return null;
    }
    return number === 0 || number >= 100 ? number : null;
  }

  function errorCode(err) {
    if (err && typeof err.code === "string" && err.code) {
      return err.code;
    }
    if (err && err.error && typeof err.error.code === "string") {
      return err.error.code;
    }
    return "";
  }

  function errorMessage(err) {
    if (!err) {
      return "Unknown error.";
    }
    if (typeof err === "string") {
      return err;
    }
    if (typeof err.message === "string" && err.message) {
      return err.message;
    }
    if (err.error && typeof err.error.message === "string" && err.error.message) {
      return err.error.message;
    }
    try {
      return JSON.stringify(err);
    } catch (error) {
      return String(err);
    }
  }

  /* ------------------------------------------------------------------ form */

  function currentGraph() {
    var id = els.graphSelect.value;
    for (var index = 0; index < state.graphs.length; index += 1) {
      if (state.graphs[index].id === id) {
        return state.graphs[index];
      }
    }
    return null;
  }

  function setGraphHelp(message, isError) {
    els.graphHelp.textContent = message;
    if (els.graphHelp.classList) {
      els.graphHelp.classList.toggle("is-error", Boolean(isError));
    }
  }

  function setFormError(message) {
    els.formError.textContent = message || "";
    els.formError.hidden = !message;
  }

  function setRunStatus(message) {
    els.runStatus.textContent = message || "";
  }

  function setTraceStatus(message, isError) {
    els.traceStatus.textContent = message || "";
    if (els.traceStatus.classList) {
      els.traceStatus.classList.toggle("is-error", Boolean(isError));
    }
  }

  function setReplayStatus(message, isError) {
    els.replayStatus.textContent = message || "";
    if (els.replayStatus.classList) {
      els.replayStatus.classList.toggle("is-error", Boolean(isError));
    }
  }

  function loadGraphs() {
    els.graphSelect.disabled = true;
    els.runButton.disabled = true;
    clear(els.graphSelect);
    els.graphSelect.appendChild(make("option", { value: "", selected: "" }, "Loading graphs…"));
    setGraphHelp("Loading compiled graphs from the local registry…");

    MG.api("/api/graphs")
      .then(function (payload) {
        var graphs = payload && Array.isArray(payload.graphs) ? payload.graphs : [];
        state.graphs = graphs.filter(function (graph) {
          return graph && typeof graph === "object" && typeof graph.id === "string" && graph.id;
        });
        fillGraphSelect();
      })
      .catch(function (err) {
        state.graphs = [];
        clear(els.graphSelect);
        els.graphSelect.appendChild(make("option", { value: "", selected: "" }, "Graphs unavailable"));
        els.graphSelect.disabled = true;
        els.runButton.disabled = true;
        var code = errorCode(err);
        setGraphHelp(
          "Could not load graphs" +
            (code ? " [" + code + "]" : "") +
            ": " +
            errorMessage(err),
          true
        );
        showErrorPanel(err, {
          title: "Could not load graphs",
          hint:
            "The graph list comes from the local registry. Confirm that mg ui is running, then " +
            "reload the page.",
        });
      });
  }

  function fillGraphSelect() {
    clear(els.graphSelect);
    if (!state.graphs.length) {
      var empty = make("option", { value: "", selected: "" }, "No graphs found");
      els.graphSelect.appendChild(empty);
      els.graphSelect.disabled = true;
      els.runButton.disabled = true;
      setGraphHelp(
        "No compilable graphs were found in the registry. Add a graph to the default registry, " +
          "then reload the page."
      );
      return;
    }

    state.graphs.forEach(function (graph) {
      var name = display(graph.name, graph.id);
      var version = display(graph.version, "");
      var label = name + " — " + graph.id + (version ? "@" + version : "");
      els.graphSelect.appendChild(make("option", { value: graph.id }, label));
    });

    els.graphSelect.disabled = false;
    els.runButton.disabled = state.running;

    var requested = "";
    try {
      requested = new URLSearchParams(window.location.search).get("graph") || "";
    } catch (err) {
      requested = "";
    }
    if (!requested) {
      requested = state.linkedGraphId;
    }
    if (requested) {
      var match = null;
      state.graphs.forEach(function (graph) {
        if (
          graph.id === requested ||
          (graph.version && graph.id + "@" + graph.version === requested)
        ) {
          match = graph;
        }
      });
      if (match) {
        els.graphSelect.value = match.id;
      } else {
        state.requestedMissing = requested;
      }
    }
    updateGraphHelp();
  }

  function linkedRunIdFromLocation() {
    try {
      return (new URLSearchParams(window.location.search).get("id") || "").trim();
    } catch (err) {
      return "";
    }
  }

  function selectAvailableGraph(graphId) {
    if (!graphId || !els.graphSelect || els.graphSelect.disabled) {
      return;
    }
    for (var index = 0; index < state.graphs.length; index += 1) {
      if (state.graphs[index].id === graphId) {
        els.graphSelect.value = graphId;
        updateGraphHelp();
        return;
      }
    }
  }

  /*
   * Run history links to run.html?id=<run_id>. Load the recorded manifest so the
   * console can show its path, usage, artifacts, trace, and replay controls
   * without re-running the graph.
   */
  function loadLinkedRun() {
    var runId = linkedRunIdFromLocation();
    if (!runId) {
      return;
    }
    hideErrorPanel();
    setRunStatus("Loading recorded run " + runId + "…");
    MG.api("/api/runs/" + encodeURIComponent(runId))
      .then(function (manifest) {
        if (state.running) {
          return undefined;
        }
        if (!manifest || typeof manifest !== "object") {
          throw new Error("The server returned an invalid run manifest.");
        }
        var resolvedId =
          typeof manifest.run_id === "string" && manifest.run_id ? manifest.run_id : runId;
        if (!manifest.run_id) {
          manifest.run_id = resolvedId;
        }
        var graph = manifest.graph && typeof manifest.graph === "object" ? manifest.graph : {};
        var graphId = typeof graph.id === "string" ? graph.id : "";
        var request = {
          graphId: graphId,
          graphLabel: graphId,
          task:
            manifest.request && typeof manifest.request.task === "string"
              ? manifest.request.task
              : "",
          provider: "recorded",
        };
        state.linkedGraphId = graphId;
        renderResult(manifest, request, { focus: false });
        hideErrorPanel();
        setRunStatus("Viewing recorded run " + resolvedId + ".");
        if (graphId) {
          selectAvailableGraph(graphId);
        }
        return loadTrace(resolvedId);
      })
      .catch(function (err) {
        var code = errorCode(err);
        setRunStatus(
          "Could not load recorded run " + runId + (code ? " [" + code + "]" : "") + "."
        );
        showErrorPanel(err, {
          title: "Could not load recorded run",
          context: "Run: " + runId + " (from the page URL)",
          hint:
            "The run may have been deleted or the URL may be wrong. Choose a run from the " +
            "Models & runs page, or start a new run above.",
        });
      });
  }

  function updateGraphHelp() {
    var graph = currentGraph();
    if (!graph) {
      setGraphHelp("Select a graph to run.");
      return;
    }
    var parts = [];
    if (state.requestedMissing) {
      parts.push(
        'The requested graph "' +
          state.requestedMissing +
          '" is not in the registry. Showing the selected graph instead.'
      );
    }
    if (graph.description) {
      parts.push(String(graph.description));
    }
    var counts = [];
    if (graph.node_count !== undefined) {
      counts.push(formatCount(graph.node_count) + " nodes");
    }
    if (graph.edge_count !== undefined) {
      counts.push(formatCount(graph.edge_count) + " edges");
    }
    if (counts.length) {
      parts.push(counts.join(" · "));
    }
    if (Array.isArray(graph.warnings) && graph.warnings.length) {
      parts.push("Warnings: " + graph.warnings.map(function (warning) {
        return display(warning);
      }).join(" "));
    }
    setGraphHelp(parts.join(" "));
  }

  function updateProviderHelp() {
    if (els.providerSelect.value === "mock") {
      els.providerHelp.textContent =
        "The mock provider is fast, offline, and deterministic. Use it to inspect the console " +
        "without a model server.";
    } else {
      els.providerHelp.textContent =
        "The auto provider uses each node's configured local provider (for example Ollama). " +
        "It is slower and fails if the model server or model is unavailable.";
    }
  }

  function useExampleTask() {
    if (state.running) {
      return;
    }
    els.taskInput.value = "Write a debounce function in TypeScript.";
    els.taskInput.focus();
  }

  function setRunning(isRunning) {
    state.running = isRunning;
    els.runButton.disabled = isRunning || !state.graphs.length;
    els.runButton.textContent = isRunning ? "Running…" : "Run graph";
    if (els.fields) {
      els.fields.disabled = isRunning;
    }
    if (els.form) {
      els.form.setAttribute("aria-busy", isRunning ? "true" : "false");
    }
    if (isRunning) {
      var seconds = 0;
      setRunStatus(
        "Running… 0s. The server executes the graph synchronously; this can take a while."
      );
      stopTimer();
      state.timer = window.setInterval(function () {
        seconds += 1;
        if (!state.running) {
          return;
        }
        setRunStatus(
          "Running… " +
            seconds +
            "s. The server executes the graph synchronously; this can take a while."
        );
      }, 1000);
    } else {
      stopTimer();
    }
  }

  function stopTimer() {
    if (state.timer !== null) {
      window.clearInterval(state.timer);
      state.timer = null;
    }
  }

  function handleSubmit(event) {
    event.preventDefault();
    if (state.running) {
      return;
    }
    setFormError("");
    hideErrorPanel();

    var graph = currentGraph();
    var task = els.taskInput.value.trim();
    var provider = els.providerSelect.value;
    if (!graph) {
      setFormError("Choose a graph before running.");
      els.graphSelect.focus();
      return;
    }
    if (!task) {
      setFormError("Enter a task. The task is sent to the graph's model nodes.");
      els.taskInput.focus();
      return;
    }
    if (provider !== "mock" && provider !== "auto") {
      setFormError("Choose the mock or auto provider.");
      els.providerSelect.focus();
      return;
    }

    var request = {
      graphId: graph.id,
      graphLabel: display(graph.name, graph.id),
      task: task,
      provider: provider,
    };

    hideResult();
    state.currentRun = null;
    state.running = true;
    state.startedAt = Date.now();
    setRunning(true);

    MG.api("/api/runs", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify({
        graph_id: request.graphId,
        task: request.task,
        provider: request.provider,
      }),
    })
      .then(function (payload) {
        if (!payload || typeof payload !== "object") {
          throw new Error("The server returned an empty or invalid run response.");
        }
        setRunning(false);
        renderResult(payload, request);

        var elapsed = payload.usage && toNumber(payload.usage.elapsed_seconds);
        var client = (Date.now() - state.startedAt) / 1000;
        var status = "Run finished: " + display(payload.outcome, "unknown") + " (";
        status += client.toFixed(2) + "s in the browser";
        if (elapsed !== null) {
          status += ", " + elapsed.toFixed(2) + "s server-reported";
        }
        status += ").";

        var runId = typeof payload.run_id === "string" ? payload.run_id : "";
        if (!runId) {
          setRunStatus(status + " Trace and replay are unavailable (no run id).");
          setTraceStatus("Trace unavailable: the run response did not include a run id.", true);
          return undefined;
        }
        setRunStatus(status);
        return loadTrace(runId);
      })
      .catch(function (err) {
        setRunning(false);
        showRunError(err, request);
      });
  }

  /* ---------------------------------------------------------------- result */

  function outcomeClass(outcome) {
    if (outcome === "succeeded") {
      return "badge badge-ok";
    }
    if (outcome === "budget_exhausted") {
      return "badge badge-warn";
    }
    return "badge badge-bad";
  }

  function hideResult() {
    els.resultPanel.hidden = true;
    els.replayResult.hidden = true;
    if (els.replayStatus) {
      setReplayStatus("");
    }
    stopTimer();
  }

  function hideErrorPanel() {
    els.errorPanel.hidden = true;
  }

  function focusPanel(element) {
    if (!element) {
      return;
    }
    try {
      element.focus({ preventScroll: false });
    } catch (err) {
      element.focus();
    }
  }

  function resultGraphLabel(request) {
    var id = request.graphId || "";
    var label = request.graphLabel || "";
    if (!id) {
      return label || "—";
    }
    if (!label || label === id) {
      return id;
    }
    return label + " (" + id + ")";
  }

  function providerLabel(provider) {
    if (provider === "mock") {
      return "mock — offline";
    }
    if (provider === "auto") {
      return "auto — local providers";
    }
    return "recorded — from the run manifest";
  }

  function renderResult(payload, request, options) {
    var opts = options || {};
    if (!payload || typeof payload !== "object") {
      showErrorPanel(new Error("The server returned an empty or invalid run response."), {
        title: "Run failed",
        context: request.graphId + " · " + request.provider,
      });
      return;
    }

    var runId = typeof payload.run_id === "string" && payload.run_id ? payload.run_id : null;
    var runDir = typeof payload.run_dir === "string" && payload.run_dir ? payload.run_dir : null;
    state.currentRun = {
      runId: runId,
      runDir: runDir,
      graphId: request.graphId,
      provider: request.provider,
    };

    els.resultPanel.hidden = false;
    els.resultOutcome.textContent = display(payload.outcome, "unknown");
    els.resultOutcome.className = outcomeClass(payload.outcome);
    els.resultRunId.textContent = display(runId);
    els.resultGraph.textContent = resultGraphLabel(request);
    els.resultProvider.textContent = providerLabel(request.provider);
    if (els.resultTask) {
      els.resultTask.textContent = display(request.task, "—");
    }
    renderRunDir(runDir);
    renderManifestLinks(runId);
    renderUsage(payload.usage);
    renderPath(payload.path);
    renderArtifacts(payload.artifacts, state.currentRun);
    renderInlineError(payload.error);
    resetReplay(runId);
    setTraceStatus("Loading trace…");
    clear(els.traceBody);
    els.traceNote.hidden = true;
    if (opts.focus !== false) {
      focusPanel(els.resultPanel);
    }
  }

  function renderRunDir(runDir) {
    els.resultRunDir.textContent = display(runDir);
    if (runDir) {
      els.copyRunDir.hidden = false;
      var url = fileUrl(runDir);
      if (url) {
        els.runDirLink.href = url;
        els.runDirLink.hidden = false;
      } else {
        els.runDirLink.hidden = true;
      }
    } else {
      els.copyRunDir.hidden = true;
      els.runDirLink.hidden = true;
    }
  }

  function fileUrl(path) {
    if (typeof path !== "string" || !path) {
      return "";
    }
    var normalized = path.replace(/\\/g, "/");
    if (/^[A-Za-z]:\//.test(normalized)) {
      normalized = "/" + normalized;
    }
    if (normalized.charAt(0) !== "/") {
      return "";
    }
    var encoded = normalized
      .split("/")
      .map(function (part) {
        return encodeURIComponent(part);
      })
      .join("/");
    return "file://" + encoded;
  }

  function renderManifestLinks(runId) {
    if (!runId) {
      els.manifestLink.hidden = true;
      els.rawTraceLink.hidden = true;
      return;
    }
    els.manifestLink.href = "/api/runs/" + encodeURIComponent(runId);
    els.manifestLink.hidden = false;
    els.rawTraceLink.href = "/api/runs/" + encodeURIComponent(runId) + "/trace";
    els.rawTraceLink.hidden = false;
  }

  function renderUsage(usage) {
    clear(els.usageList);
    if (!usage || typeof usage !== "object") {
      usage = {};
    }
    var rows = [
      ["Steps", formatCount(usage.steps)],
      ["Input tokens", formatCount(usage.input_tokens)],
      ["Output tokens", formatCount(usage.output_tokens)],
      ["Total tokens", formatCount(usage.total_tokens)],
      ["Cost", formatCost(usage.cost_usd)],
      ["Elapsed", formatSeconds(usage.elapsed_seconds)],
    ];
    rows.forEach(function (row) {
      var cell = make("div", {});
      cell.appendChild(make("dt", {}, row[0]));
      cell.appendChild(make("dd", {}, row[1]));
      els.usageList.appendChild(cell);
    });
  }

  function confidenceBar(confidence) {
    var number = toNumber(confidence);
    var percent = number === null ? 0 : Math.max(0, Math.min(1, number)) * 100;
    var wrapper = make("div", {
      "class": "confidence-bar",
      role: "img",
      "aria-label": "Confidence " + Math.round(percent) + "%",
    });
    var fill = make("span", { "class": "confidence-fill" });
    fill.style.width = percent + "%";
    wrapper.appendChild(fill);
    return wrapper;
  }

  function renderPath(path) {
    clear(els.pathTimeline);
    if (!Array.isArray(path) || !path.length) {
      els.pathTimeline.appendChild(
        make("li", { "class": "timeline-empty" }, "No routing path was recorded.")
      );
      return;
    }

    path.forEach(function (entry, index) {
      var item = entry && typeof entry === "object" ? entry : {};
      var row = make("li", { "class": "timeline-item" });
      var head = make("div", { "class": "timeline-head" });
      head.appendChild(make("span", { "class": "step-index" }, String(index + 1)));
      head.appendChild(make("span", { "class": "node-name" }, display(item.node, "(unnamed node)")));
      head.appendChild(make("span", { "class": "badge badge-kind" }, display(item.kind, "node")));
      row.appendChild(head);

      if (item.kind === "router" || item.decision) {
        var meta = make("p", { "class": "timeline-meta" });
        meta.appendChild(
          make("span", { "class": "decision" }, "Decision: " + display(item.decision, "unknown"))
        );
        if (item.to) {
          meta.appendChild(make("span", { "class": "target" }, "→ " + item.to));
        }
        if (item.confidence !== undefined && item.confidence !== null) {
          meta.appendChild(
            make(
              "span",
              { "class": "confidence" },
              "Confidence: " + formatConfidence(item.confidence)
            )
          );
        }
        row.appendChild(meta);
        if (item.confidence !== undefined && item.confidence !== null) {
          row.appendChild(confidenceBar(item.confidence));
        }
        if (item.reason) {
          row.appendChild(make("p", { "class": "reason" }, "Reason: " + item.reason));
        }
      } else if (item.kind === "subgraph") {
        row.appendChild(
          make(
            "p",
            { "class": "timeline-meta" },
            "Subgraph: " + display(item.graph, "unknown") + "."
          )
        );
      } else {
        row.appendChild(make("p", { "class": "timeline-meta" }, "Completed."));
      }

      els.pathTimeline.appendChild(row);
    });
  }

  function renderArtifacts(artifacts, run) {
    clear(els.artifacts);
    if (!Array.isArray(artifacts) || !artifacts.length) {
      els.artifacts.appendChild(
        make("p", { "class": "empty" }, "This run produced no exit artifacts.")
      );
      return;
    }

    artifacts.forEach(function (rawArtifact) {
      var artifact = rawArtifact && typeof rawArtifact === "object" ? rawArtifact : {};
      var card = make("article", { "class": "artifact", "aria-label": "Artifact " + display(artifact.name) });
      var head = make("div", { "class": "artifact-head" });
      head.appendChild(
        make("h4", { "class": "artifact-name" }, display(artifact.name, "(unnamed artifact)"))
      );
      head.appendChild(make("span", { "class": "badge badge-kind" }, display(artifact.type, "artifact")));
      var copy = make("button", { "class": "button secondary small", type: "button" }, "Copy");
      head.appendChild(copy);
      card.appendChild(head);

      var meta = [];
      if (artifact.producer) {
        meta.push("producer " + display(artifact.producer));
      }
      if (artifact.media_type) {
        meta.push(display(artifact.media_type));
      }
      if (artifact.hash) {
        meta.push(shortText(artifact.hash, 32));
      }
      if (meta.length) {
        var metaLine = make("p", { "class": "artifact-meta" });
        if (artifact.hash) {
          metaLine.title = String(artifact.hash);
        }
        metaLine.textContent = meta.join(" · ");
        card.appendChild(metaLine);
      }

      var pre = make("pre", { "class": "artifact-content", tabindex: "0" });
      if (typeof artifact.content === "string") {
        pre.textContent = artifact.content;
      } else {
        pre.textContent = "(content not included in the run response)";
        if (run && run.runId && artifact.name) {
          hydrateArtifact(pre, run.runId, artifact.name);
        }
      }
      copy.addEventListener("click", function () {
        copyText(pre.textContent, copy);
      });
      card.appendChild(pre);
      els.artifacts.appendChild(card);
    });
  }

  function hydrateArtifact(pre, runId, name) {
    MG.api("/api/runs/" + encodeURIComponent(runId) + "/artifacts/" + encodeURIComponent(name))
      .then(function (payload) {
        var content =
          payload && payload.artifact && typeof payload.artifact.content === "string"
            ? payload.artifact.content
            : null;
        if (content === null) {
          pre.textContent = "(artifact content unavailable)";
          return;
        }
        pre.textContent = content;
      })
      .catch(function (err) {
        var code = errorCode(err);
        pre.textContent =
          "Could not load artifact content" +
          (code ? " [" + code + "]" : "") +
          ": " +
          errorMessage(err);
      });
  }

  function renderInlineError(error) {
    if (!error) {
      els.runError.hidden = true;
      els.runError.textContent = "";
      return;
    }
    var message = typeof error === "string" ? error : display(error.message, "The run failed.");
    var code = errorCode(error);
    els.runError.textContent = (code ? "[" + code + "] " : "") + message;
    els.runError.hidden = false;
  }

  /* ----------------------------------------------------------------- trace */

  function loadTrace(runId) {
    setTraceStatus("Loading trace…");
    clear(els.traceBody);
    return MG.api("/api/runs/" + encodeURIComponent(runId) + "/trace")
      .then(function (payload) {
        if (!state.currentRun || state.currentRun.runId !== runId) {
          return;
        }
        var events = payload && Array.isArray(payload.events) ? payload.events : null;
        if (!events) {
          throw new Error("The trace response did not contain an events array.");
        }
        renderTrace(events);
        if (!events.length) {
          els.traceBody.appendChild(
            make("tr", {}, make("td", { colspan: "5", "class": "empty" }, "No events recorded."))
          );
          setTraceStatus("The trace is empty.");
        } else {
          setTraceStatus(
            "Loaded " + events.length + " trace event" + (events.length === 1 ? "" : "s") + "."
          );
        }
      })
      .catch(function (err) {
        if (!state.currentRun || state.currentRun.runId !== runId) {
          return;
        }
        var code = errorCode(err);
        setTraceStatus(
          "Could not load trace" + (code ? " [" + code + "]" : "") + ": " + errorMessage(err),
          true
        );
        els.traceBody.appendChild(
          make(
            "tr",
            {},
            make(
              "td",
              { colspan: "5", "class": "empty" },
              "Trace unavailable; the run result above is still valid."
            )
          )
        );
      });
  }

  function renderTrace(events) {
    clear(els.traceBody);
    var shown = Math.min(events.length, MAX_TRACE_ROWS);
    for (var index = 0; index < shown; index += 1) {
      var event = events[index];
      if (!event || typeof event !== "object") {
        event = { value: event };
      }
      var row = make("tr", {});
      row.appendChild(make("td", { "class": "trace-index" }, String(index + 1)));
      row.appendChild(make("td", { "class": "trace-time" }, formatTimestamp(event.ts)));
      row.appendChild(
        make("td", { "class": "trace-event" }, display(event.event, "(unnamed event)"))
      );
      row.appendChild(make("td", { "class": "trace-node" }, display(event.node)));
      var detailsCell = make("td", { "class": "trace-details" });
      var details = make("details", { "class": "trace-raw" });
      details.appendChild(make("summary", {}, traceSummary(event)));
      var pre = make("pre", { "class": "trace-json", tabindex: "0" });
      try {
        pre.textContent = JSON.stringify(event, null, 2);
      } catch (err) {
        pre.textContent = display(event);
      }
      details.appendChild(pre);
      detailsCell.appendChild(details);
      row.appendChild(detailsCell);
      els.traceBody.appendChild(row);
    }
    if (events.length > MAX_TRACE_ROWS) {
      els.traceNote.hidden = false;
      els.traceNote.textContent =
        "Showing the first " +
        MAX_TRACE_ROWS +
        " of " +
        events.length +
        " events. Use the raw trace link for the complete file.";
    } else {
      els.traceNote.hidden = true;
      els.traceNote.textContent = "";
    }
  }

  function traceSummary(event) {
    var keys = [
      "decision",
      "to",
      "confidence",
      "threshold",
      "code",
      "message",
      "outcome",
      "name",
      "type",
      "provider",
      "model",
      "attempt",
      "tokens",
      "total_tokens",
      "hash",
      "prompt_hash",
      "depth",
      "step",
      "reason",
    ];
    var parts = [];
    for (var index = 0; index < keys.length && parts.length < 4; index += 1) {
      var key = keys[index];
      var value = event[key];
      if (value === undefined || value === null || value === "") {
        continue;
      }
      var text = typeof value === "object" ? JSON.stringify(value) : String(value);
      text = text.replace(/\s+/g, " ");
      if (text.length > 54) {
        text = text.slice(0, 53) + "…";
      }
      parts.push(key + "=" + text);
    }
    return parts.length ? parts.join(" · ") : "raw event";
  }

  /* ---------------------------------------------------------------- replay */

  function resetReplay(runId) {
    els.replayResult.hidden = true;
    clear(els.replayMismatches);
    els.replayMismatches.hidden = true;
    els.replayVerdict.textContent = "";
    els.replayDetail.textContent = "";
    els.replayMeta.textContent = "";
    setReplayStatus("");
    updateReplayAvailability();
    if (!runId) {
      els.replayButton.disabled = true;
    }
  }

  function updateReplayAvailability() {
    var available = !!(state.currentRun && state.currentRun.runId);
    els.replayButton.disabled = !available || state.running || state.replaying;
  }

  function handleReplay() {
    if (!state.currentRun || !state.currentRun.runId || state.running || state.replaying) {
      return;
    }
    var runId = state.currentRun.runId;
    state.replaying = true;
    updateReplayAvailability();
    els.replayResult.hidden = true;
    clear(els.replayMismatches);
    setReplayStatus("Replaying the recorded run…");

    MG.api("/api/runs/" + encodeURIComponent(runId) + "/replay", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: "{}",
    })
      .then(function (payload) {
        if (!state.currentRun || state.currentRun.runId !== runId) {
          return;
        }
        renderReplay(payload);
      })
      .catch(function (err) {
        if (!state.currentRun || state.currentRun.runId !== runId) {
          return;
        }
        var status = errorStatus(err);
        var refused = status === 409;
        var title = refused ? "Replay refused" : "Replay failed";
        var detail = refused
          ? "The server refused to replay this run because it no longer matches the recorded run " +
            "(for example, the compiled graph or an artifact changed). The recorded run was not " +
            "changed. The server reported:"
          : "The replay did not complete. The server reported:";
        setReplayStatus(title + ".", true);
        renderReplayResult(title, "is-bad", detail, [errorMessage(err)], errorCode(err), []);
      })
      .finally(function () {
        state.replaying = false;
        updateReplayAvailability();
      });
  }

  function renderReplay(payload) {
    if (!payload || typeof payload !== "object") {
      setReplayStatus("Replay failed: the server returned an invalid response.", true);
      return;
    }
    var mismatches = Array.isArray(payload.mismatches)
      ? payload.mismatches.map(function (mismatch) {
          return display(mismatch);
        })
      : [];
    var ok = payload.ok === true;
    var verdict;
    var tone;
    var detail;
    if (ok && !mismatches.length) {
      verdict = "Replay OK";
      tone = "is-ok";
      detail =
        "No mismatches: the outcome, routing path, and exit artifact hashes match the recorded run.";
    } else if (ok) {
      verdict = "Replay OK with mismatches";
      tone = "is-warn";
      detail =
        "The server reported overall success but listed differences. Treat the mismatches below " +
        "as real.";
    } else {
      verdict = "Replay MISMATCH";
      tone = "is-bad";
      detail = "The replay completed but did not match the recorded run.";
    }
    var meta = [];
    if (payload.run_id) {
      meta.push("run " + display(payload.run_id));
    }
    if (payload.replay_id) {
      meta.push("replay " + display(payload.replay_id));
    }
    if (payload.outcome) {
      meta.push("outcome " + display(payload.outcome));
    }
    setReplayStatus(verdict + ".");
    renderReplayResult(verdict, tone, detail, mismatches, errorCode(payload), meta);
  }

  function renderReplayResult(verdict, tone, detail, mismatches, code, meta) {
    els.replayResult.hidden = false;
    els.replayVerdict.textContent = (code ? "[" + code + "] " : "") + verdict;
    els.replayVerdict.className = "replay-verdict " + tone;
    els.replayDetail.textContent = detail;
    els.replayMeta.textContent = Array.isArray(meta) ? meta.join(" · ") : meta || "";
    clear(els.replayMismatches);
    if (Array.isArray(mismatches) && mismatches.length) {
      els.replayMismatches.hidden = false;
      mismatches.forEach(function (mismatch) {
        els.replayMismatches.appendChild(make("li", {}, display(mismatch)));
      });
    } else {
      els.replayMismatches.hidden = true;
    }
  }

  /* ----------------------------------------------------------------- error */

  function titleForStatus(status) {
    if (status === 0) {
      return "Server unreachable";
    }
    if (status === 400) {
      return "Run rejected";
    }
    if (status === 404) {
      return "Not found";
    }
    if (status === 409) {
      return "Conflict";
    }
    if (status === 500) {
      return "Server error";
    }
    if (status === 502) {
      return "Provider failure";
    }
    return "Run failed";
  }

  function hintForStatus(status, err) {
    if (status === 0) {
      return (
        "The GUI could not reach the local server. Confirm that mg ui is still running, then " +
        "reload the page."
      );
    }
    if (status === 400) {
      return (
        "The server rejected the request as invalid. Check the selected graph and the task, then " +
        "run again."
      );
    }
    if (status === 404) {
      return "The graph or run was not found. Reload the page and choose a graph from the list.";
    }
    if (status === 409) {
      return "The request conflicted with the recorded state; no changes were applied.";
    }
    if (status === 500) {
      return "The local server failed. Check the mg ui terminal output for details.";
    }
    if (status === 502) {
      return (
        "A model provider failed. Try the mock provider, or check that the local model server " +
        "and the configured model are available."
      );
    }
    return (
      "The server returned an error or an unexpected response. Check the mg ui terminal output " +
      "for details."
    );
  }

  function showErrorPanel(err, options) {
    options = options || {};
    var status = errorStatus(err);
    var title = options.title || titleForStatus(status);
    var hint = options.hint || hintForStatus(status, err);
    var code = errorCode(err);
    els.errorHeading.textContent = title;
    els.errorCode.textContent = code || "error";
    els.errorContext.textContent = options.context || "";
    els.errorContext.hidden = !options.context;
    els.errorMessage.textContent = errorMessage(err);
    els.errorHint.textContent = hint;
    els.errorPanel.hidden = false;
    focusPanel(els.errorPanel);
  }

  function showRunError(err, request) {
    var status = errorStatus(err);
    showErrorPanel(err, {
      title: titleForStatus(status),
      context:
        "Graph: " + request.graphId + " · provider: " + request.provider + " · no result was returned.",
      hint: hintForStatus(status, err),
    });
    setRunStatus(titleForStatus(status) + ". See the error panel for the server's message.");
  }

  /* ------------------------------------------------------------- clipboard */

  function copyRunDir() {
    if (!state.currentRun || !state.currentRun.runDir) {
      return;
    }
    copyText(state.currentRun.runDir, els.copyRunDir);
  }

  function copyText(value, button) {
    var text = typeof value === "string" ? value : display(value, "");
    var original = button.textContent;
    function restoreSoon() {
      window.setTimeout(function () {
        button.textContent = original;
      }, 1600);
    }
    function succeeded() {
      button.textContent = "Copied";
      restoreSoon();
    }
    function failed() {
      button.textContent = "Copy failed";
      restoreSoon();
    }
    function fallback() {
      if (legacyCopy(text)) {
        succeeded();
      } else {
        failed();
      }
    }

    if (navigator.clipboard && typeof navigator.clipboard.writeText === "function") {
      try {
        var pending = navigator.clipboard.writeText(text);
        if (pending && typeof pending.then === "function") {
          pending.then(succeeded).catch(fallback);
        } else {
          succeeded();
        }
      } catch (err) {
        fallback();
      }
    } else {
      fallback();
    }
  }

  function legacyCopy(text) {
    var area = document.createElement("textarea");
    area.value = text;
    area.setAttribute("readonly", "");
    area.style.position = "fixed";
    area.style.top = "-1000px";
    area.style.opacity = "0";
    document.body.appendChild(area);
    var copied = false;
    try {
      area.select();
      copied = typeof document.execCommand === "function" && document.execCommand("copy");
    } catch (err) {
      copied = false;
    }
    document.body.removeChild(area);
    return copied;
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
