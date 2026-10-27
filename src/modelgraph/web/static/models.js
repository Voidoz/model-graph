/*
 * T-0044 — model catalog and run history.
 *
 * Vanilla JavaScript, no build step. Every value that comes from the local API is
 * inserted as a text node through the shared MG helpers; this file never parses
 * API-derived strings as markup and never creates markup with string concatenation.
 */
(function () {
  "use strict";

  const MG = window.MG;
  const RUN_HISTORY_LIMIT = 50;

  const OUTCOME_CLASSES = Object.freeze({
    succeeded: "badge--succeeded",
    success: "badge--succeeded",
    failed: "badge--failed",
    failure: "badge--failed",
    error: "badge--failed",
    cancelled: "badge--warn",
    canceled: "badge--warn",
    partial: "badge--warn",
    timeout: "badge--warn",
  });

  const state = {
    runs: [],
    outcomes: [],
    outcomeFilter: "all",
    loadingRuns: false,
    pullJob: null,
    pullTimer: null,
  };

  function byId(id) {
    return document.getElementById(id);
  }

  function boot() {
    if (
      !MG ||
      typeof MG.api !== "function" ||
      typeof MG.el !== "function" ||
      typeof MG.nav !== "function"
    ) {
      showFatal("The shared UI helpers did not load. Reload the page or restart `mg ui`.");
      return;
    }

    try {
      MG.nav("models");
    } catch (failure) {
      // Navigation is cosmetic; the catalog and history must still render without it.
      console.warn("Model Graph navigation failed to render.", failure);
    }

    const refresh = byId("history-refresh");
    if (refresh) {
      refresh.addEventListener("click", loadRuns);
    }

    const filter = byId("outcome-filter");
    if (filter) {
      filter.addEventListener("change", onOutcomeFilterChange);
    }

    const pullForm = byId("pull-form");
    if (pullForm) {
      pullForm.addEventListener("submit", startPull);
    }

    loadInstalledModels();
    loadModels();
    loadRuns();
  }

  function showFatal(message) {
    const main = byId("main");
    if (!main) {
      return;
    }
    const box = document.createElement("div");
    box.className = "models-error";
    box.setAttribute("role", "alert");
    const title = document.createElement("p");
    title.className = "models-state__title";
    title.textContent = message;
    box.appendChild(title);
    main.replaceChildren(box);
  }

  function displayValue(value, fallback) {
    if (typeof value === "string") {
      const trimmed = value.trim();
      return trimmed === "" ? fallback : trimmed;
    }
    if (typeof value === "number" && Number.isFinite(value)) {
      return String(value);
    }
    return fallback;
  }

  function firstDefined() {
    for (let index = 0; index < arguments.length; index += 1) {
      const value = arguments[index];
      if (value !== null && value !== undefined && value !== "") {
        return value;
      }
    }
    return undefined;
  }

  function licenseName(license) {
    if (typeof license === "string" && license.trim() !== "") {
      return license.trim();
    }
    if (license && typeof license === "object") {
      if (typeof license.name === "string" && license.name.trim() !== "") {
        return license.name.trim();
      }
      if (typeof license.id === "string" && license.id.trim() !== "") {
        return license.id.trim();
      }
    }
    return "Unknown license";
  }

  function modelsFrom(payload) {
    if (Array.isArray(payload)) {
      return payload;
    }
    if (payload && typeof payload === "object" && Array.isArray(payload.models)) {
      return payload.models;
    }
    return null;
  }

  function runsFrom(payload) {
    if (Array.isArray(payload)) {
      return payload;
    }
    if (payload && typeof payload === "object" && Array.isArray(payload.runs)) {
      return payload.runs;
    }
    return null;
  }

  function describeError(failure, fallback) {
    if (typeof failure === "string" && failure.trim() !== "") {
      return failure.trim();
    }
    if (!failure) {
      return fallback;
    }
    const code = typeof failure.code === "string" ? failure.code.trim() : "";
    const message =
      typeof failure.message === "string" && failure.message.trim() !== ""
        ? failure.message.trim()
        : fallback;
    return code ? `${code}: ${message}` : message;
  }

  function setStatus(element, message) {
    if (element) {
      element.textContent = message;
    }
  }

  function showError(box, titleText, detailText, retry) {
    if (!box) {
      return;
    }
    const children = [
      MG.el("p", { class: "models-state__title" }, MG.text(titleText)),
      MG.el("p", { class: "models-state__detail" }, MG.text(detailText)),
    ];
    if (typeof retry === "function") {
      const button = MG.el("button", { class: "models-button", type: "button" }, "Try again");
      button.addEventListener("click", retry);
      children.push(button);
    }
    box.replaceChildren(...children);
    box.hidden = false;
  }

  /* ------------------------------------------------------------- local pulls */

  async function loadInstalledModels() {
    const target = byId("installed-models");
    const status = byId("installed-status");
    if (!target || !status) return;
    target.replaceChildren();
    setStatus(status, "Checking installed models…");
    try {
      const payload = await MG.api("/api/models/installed");
      const installed = Array.isArray(payload && payload.installed) ? payload.installed : [];
      if (payload && payload.available === false) {
        setStatus(status, "Ollama is unavailable.");
        target.appendChild(
          MG.el(
            "p",
            { class: "models-state__detail" },
            MG.text(payload.detail || "Start Ollama to pull and run local models."),
          ),
        );
        return;
      }
      setStatus(
        status,
        installed.length
          ? `${installed.length} installed Ollama model${installed.length === 1 ? "" : "s"}.`
          : "No Ollama models installed yet.",
      );
      if (installed.length) {
        target.replaceChildren(
          ...installed.map((name) =>
            MG.el("span", { class: "installed-model", title: name }, MG.text(name)),
          ),
        );
      }
    } catch (failure) {
      setStatus(status, "Installed models unavailable.");
      target.appendChild(
        MG.el("p", { class: "models-state__detail" }, MG.text(describeError(failure, ""))),
      );
    }
  }

  async function startPull(event) {
    if (event && typeof event.preventDefault === "function") event.preventDefault();
    if (state.pullJob) return;
    const input = byId("pull-name");
    const button = byId("pull-submit");
    const status = byId("pull-status");
    const log = byId("pull-log");
    const name = input && typeof input.value === "string" ? input.value.trim() : "";
    if (!name) {
      setStatus(status, "Enter an Ollama model tag first.");
      if (input) input.focus();
      return;
    }
    if (button) button.disabled = true;
    if (log) {
      log.hidden = true;
      log.textContent = "";
    }
    setStatus(status, `Starting pull for ${name}…`);
    try {
      const job = await MG.api("/api/models/pull", {
        method: "POST",
        body: { name },
      });
      state.pullJob = job;
      renderPullJob(job);
      schedulePullPoll(job.job_id);
    } catch (failure) {
      state.pullJob = null;
      setStatus(status, describeError(failure, "Could not start the model pull."));
      if (button) button.disabled = false;
    }
  }

  function schedulePullPoll(jobId) {
    if (state.pullTimer) window.clearTimeout(state.pullTimer);
    state.pullTimer = window.setTimeout(() => pollPull(jobId), 900);
  }

  async function pollPull(jobId) {
    const button = byId("pull-submit");
    try {
      const job = await MG.api(`/api/models/pull/${encodeURIComponent(jobId)}`);
      state.pullJob = job;
      renderPullJob(job);
      if (job.status === "running") {
        schedulePullPoll(jobId);
        return;
      }
      if (button) button.disabled = false;
      state.pullJob = null;
      state.pullTimer = null;
      if (job.status === "succeeded") {
        await loadInstalledModels();
      }
    } catch (failure) {
      if (button) button.disabled = false;
      state.pullJob = null;
      state.pullTimer = null;
      setStatus(byId("pull-status"), describeError(failure, "Model pull status failed."));
    }
  }

  function renderPullJob(job) {
    const status = byId("pull-status");
    const log = byId("pull-log");
    const lines = Array.isArray(job && job.lines) ? job.lines : [];
    const current = job && job.status ? String(job.status) : "running";
    if (log) {
      log.hidden = lines.length === 0;
      log.textContent = lines.join("\n");
      log.scrollTop = log.scrollHeight;
    }
    if (!status) return;
    if (current === "running") {
      setStatus(status, `Pulling ${job.model}… this can take a while.`);
    } else if (current === "succeeded") {
      setStatus(status, `Pulled ${job.model}. It is available to Ollama and local graphs.`);
    } else {
      setStatus(status, job.error || `Pull failed for ${job.model}.`);
    }
  }

  /* ------------------------------------------------------------------ models */

  async function loadModels() {
    const list = byId("models-list");
    const empty = byId("models-empty");
    const box = byId("models-error");
    const status = byId("models-status");
    if (!list || !empty || !box || !status) {
      return;
    }

    empty.hidden = true;
    box.hidden = true;
    box.replaceChildren();
    list.hidden = false;
    list.replaceChildren();
    setStatus(status, "Loading models…");

    try {
      const payload = await MG.api("/api/models");
      const models = modelsFrom(payload);
      if (models === null) {
        list.hidden = true;
        setStatus(status, "Model catalog unavailable.");
        showError(
          box,
          "Could not read the model catalog.",
          "The server returned an unexpected response for /api/models.",
          loadModels,
        );
        return;
      }

      list.replaceChildren(...models.map(modelCard));
      if (models.length === 0) {
        list.hidden = true;
        empty.hidden = false;
        setStatus(status, "No models in the local registry.");
      } else {
        setStatus(
          status,
          `${models.length} model${models.length === 1 ? "" : "s"} in the local registry.`,
        );
      }
    } catch (failure) {
      list.hidden = true;
      empty.hidden = true;
      setStatus(status, "Model catalog unavailable.");
      showError(
        box,
        "Could not load the model catalog.",
        describeError(failure, "The /api/models request failed."),
        loadModels,
      );
    }
  }

  function modelCard(model) {
    const item = model && typeof model === "object" ? model : {};
    const id = displayValue(item.id, "Unknown model");
    const version = displayValue(item.version, "unversioned");
    const provider = displayValue(item.provider, "Unknown provider");
    const modelName = displayValue(item.model, "Unresolved model");
    const description = displayValue(item.description, "No description provided.");

    return MG.el(
      "article",
      { class: "model-card" },
      MG.el(
        "div",
        { class: "model-card__head" },
        MG.el("h3", { class: "model-card__title" }, MG.text(id)),
        MG.el(
          "span",
          { class: "badge badge--neutral", title: "Model version" },
          "version ",
          MG.text(version),
        ),
      ),
      MG.el(
        "dl",
        { class: "model-facts" },
        ...fact("Provider", provider),
        ...fact("Model", modelName),
        ...fact("License", licenseName(item.license)),
      ),
      MG.el("p", { class: "model-card__description" }, MG.text(description)),
    );
  }

  function fact(label, value) {
    return [
      MG.el("dt", {}, label),
      MG.el("dd", {}, MG.el("code", {}, MG.text(value))),
    ];
  }

  /* -------------------------------------------------------------- run history */

  async function loadRuns() {
    if (state.loadingRuns) {
      return;
    }
    state.loadingRuns = true;

    const refresh = byId("history-refresh");
    if (refresh) {
      refresh.disabled = true;
    }
    const box = byId("history-error");
    if (box) {
      box.hidden = true;
      box.replaceChildren();
    }
    const status = byId("history-status");
    setStatus(status, "Loading recent runs…");

    try {
      const payload = await MG.api(`/api/runs?limit=${RUN_HISTORY_LIMIT}`);
      const runs = runsFrom(payload);
      if (runs === null) {
        throw new Error("The server returned an unexpected response for /api/runs.");
      }
      state.runs = runs;
      populateOutcomeFilter(runs);
      renderRuns();
    } catch (failure) {
      state.runs = [];
      state.outcomes = [];
      state.outcomeFilter = "all";
      const rows = byId("history-rows");
      if (rows) {
        rows.replaceChildren();
      }
      const wrap = byId("history-table-wrap");
      if (wrap) {
        wrap.hidden = true;
      }
      const empty = byId("history-empty");
      if (empty) {
        empty.hidden = true;
      }
      populateOutcomeFilter([]);
      setStatus(status, "Run history unavailable.");
      showError(
        box,
        "Could not load run history.",
        describeError(failure, "The /api/runs request failed."),
        loadRuns,
      );
    } finally {
      state.loadingRuns = false;
      if (refresh) {
        refresh.disabled = false;
      }
    }
  }

  function runRecord(run) {
    const item = run && typeof run === "object" ? run : {};
    const graph =
      item.graph && typeof item.graph === "object"
        ? item.graph
        : typeof item.graph === "string"
          ? { id: item.graph, version: item.graph_version }
          : null;
    const usage = item.usage && typeof item.usage === "object" ? item.usage : null;
    const budget = item.budget && typeof item.budget === "object" ? item.budget : null;

    return {
      runId: displayValue(item.run_id, ""),
      outcome: displayValue(item.outcome, "unknown"),
      graphId: displayValue(graph ? graph.id : firstDefined(item.graph_id), "unknown graph"),
      graphVersion: displayValue(
        graph ? graph.version : firstDefined(item.graph_version),
        "",
      ),
      tokens: firstDefined(item.tokens, usage && usage.total_tokens),
      cost: firstDefined(item.cost_usd, usage && usage.cost_usd, budget && budget.used_cost_usd),
      startedAt: displayValue(item.started_at, ""),
    };
  }

  function outcomeClass(outcome) {
    const key = String(outcome || "").toLowerCase();
    return OUTCOME_CLASSES[key] || "badge--unknown";
  }

  function encodeRunId(runId) {
    try {
      return encodeURIComponent(runId);
    } catch (failure) {
      return null;
    }
  }

  function runLink(runId) {
    if (!runId) {
      return MG.el("span", { class: "models-muted" }, "Unknown run");
    }
    const encoded = encodeRunId(runId);
    if (encoded === null) {
      return MG.el("span", { class: "run-id" }, MG.text(runId));
    }
    return MG.el(
      "a",
      { class: "run-id", href: `run.html?id=${encoded}` },
      MG.text(runId),
    );
  }

  function formatTokens(value) {
    if (value === null || value === undefined || value === "") {
      return MG.el("span", { class: "models-muted" }, "—");
    }
    const numeric = Number(value);
    if (Number.isFinite(numeric)) {
      return MG.el("span", { class: "numeric" }, MG.number(numeric));
    }
    return MG.el("span", { class: "numeric" }, MG.text(String(value)));
  }

  function formatCost(value) {
    if (value === null || value === undefined || value === "") {
      return MG.el("span", { class: "models-muted" }, "—");
    }
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) {
      return MG.el("span", { class: "numeric" }, MG.text(String(value)));
    }
    const digits = numeric > 0 && numeric < 0.01 ? 4 : 2;
    return MG.el("span", { class: "numeric" }, `$${numeric.toFixed(digits)}`);
  }

  function formatStarted(value) {
    if (!value) {
      return MG.el("span", { class: "models-muted" }, "Unknown");
    }
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
      return MG.el("span", { class: "time" }, MG.text(value));
    }
    return MG.el(
      "time",
      { class: "time", datetime: value },
      MG.text(date.toLocaleString()),
    );
  }

  function runRow(run) {
    const record = runRecord(run);
    const graphValue = MG.el(
      "span",
      { class: "graph-value" },
      MG.el("span", { class: "graph-id" }, MG.text(record.graphId)),
    );
    if (record.graphVersion) {
      graphValue.append(MG.el("span", { class: "graph-version" }, MG.text(record.graphVersion)));
    }
    const graphCell = MG.el("td", { "data-label": "Graph" }, graphValue);

    return MG.el(
      "tr",
      {},
      MG.el("td", { "data-label": "Run" }, runLink(record.runId)),
      MG.el(
        "td",
        { "data-label": "Outcome" },
        MG.el(
          "span",
          { class: `badge ${outcomeClass(record.outcome)}` },
          MG.text(record.outcome),
        ),
      ),
      graphCell,
      MG.el("td", { "data-label": "Tokens" }, formatTokens(record.tokens)),
      MG.el("td", { "data-label": "Cost" }, formatCost(record.cost)),
      MG.el("td", { "data-label": "Started" }, formatStarted(record.startedAt)),
    );
  }

  function timestamp(value) {
    if (typeof value === "number" && Number.isFinite(value)) {
      return value;
    }
    if (typeof value !== "string" || value.trim() === "") {
      return Number.NEGATIVE_INFINITY;
    }
    const parsed = Date.parse(value);
    return Number.isFinite(parsed) ? parsed : Number.NEGATIVE_INFINITY;
  }

  function startedTimestamp(run) {
    if (!run || typeof run !== "object") {
      return Number.NEGATIVE_INFINITY;
    }
    const started = timestamp(run.started_at);
    return started === Number.NEGATIVE_INFINITY ? timestamp(run.finished_at) : started;
  }

  function renderRuns() {
    const rows = byId("history-rows");
    const wrap = byId("history-table-wrap");
    const empty = byId("history-empty");
    if (!rows || !wrap || !empty) {
      return;
    }

    // The API sorts newest-first, but sorting again keeps the page honest if a
    // server response is ordered differently or a started_at value is missing.
    const ordered = state.runs.slice().sort((left, right) => {
      return startedTimestamp(right) - startedTimestamp(left);
    });
    const visible =
      state.outcomeFilter === "all"
        ? ordered
        : ordered.filter((run) => runRecord(run).outcome === state.outcomeFilter);

    rows.replaceChildren(...visible.map(runRow));

    if (visible.length === 0) {
      wrap.hidden = true;
      empty.hidden = false;
      if (ordered.length === 0) {
        empty.replaceChildren(
          MG.el("p", { class: "models-state__title" }, "No runs recorded yet."),
          MG.el(
            "p",
            { class: "models-state__detail" },
            "Run a graph from ",
            MG.el("a", { href: "run.html" }, "the run console"),
            " to build a history. Recorded runs appear here automatically.",
          ),
        );
      } else {
        empty.replaceChildren(
          MG.el("p", { class: "models-state__title" }, "No runs match this outcome."),
          MG.el(
            "p",
            { class: "models-state__detail" },
            "Choose “All outcomes” to see every recorded run.",
          ),
        );
      }
    } else {
      wrap.hidden = false;
      empty.hidden = true;
      empty.replaceChildren();
    }

    updateHistoryStatus(visible.length, ordered.length);
  }

  function updateHistoryStatus(visible, total) {
    const status = byId("history-status");
    if (!status) {
      return;
    }
    if (total === 0) {
      setStatus(status, "No runs recorded on this machine.");
      return;
    }
    if (visible === total) {
      setStatus(status, `${total} run${total === 1 ? "" : "s"} · newest first`);
      return;
    }
    setStatus(status, `${visible} of ${total} runs · newest first`);
  }

  function populateOutcomeFilter(runs) {
    const select = byId("outcome-filter");
    if (!select) {
      return;
    }

    const seen = new Set();
    for (const run of runs) {
      seen.add(runRecord(run).outcome);
    }
    const outcomes = Array.from(seen).sort((left, right) => left.localeCompare(right));
    state.outcomes = outcomes;

    const options = [MG.el("option", { value: "all" }, "All outcomes")];
    outcomes.forEach((outcome, index) => {
      // The option value is a local index, never API text.
      options.push(MG.el("option", { value: String(index) }, MG.text(outcome)));
    });
    select.replaceChildren(...options);
    select.disabled = outcomes.length === 0;

    let selected = "all";
    if (state.outcomeFilter !== "all") {
      const index = outcomes.indexOf(state.outcomeFilter);
      if (index >= 0) {
        selected = String(index);
      }
    }
    select.value = selected;
    state.outcomeFilter =
      selected === "all" ? "all" : (outcomes[Number(selected)] || "all");
  }

  function onOutcomeFilterChange(event) {
    const select = event && event.target ? event.target : byId("outcome-filter");
    const value = select && typeof select.value === "string" ? select.value : "all";
    if (value === "all") {
      state.outcomeFilter = "all";
    } else {
      state.outcomeFilter = state.outcomes[Number(value)] || "all";
    }
    renderRuns();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot, { once: true });
  } else {
    boot();
  }
})();
