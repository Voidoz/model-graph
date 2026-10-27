/*
 * Local graph editor. No framework, no build step, no markup strings from
 * API data: every value is inserted through MG.el / textContent.
 */
(function () {
  "use strict";

  const MG = window.MG;
  const state = {
    graphId: null,
    source: "",
    busy: false,
  };

  function byId(id) {
    return document.getElementById(id);
  }

  function setStatus(message) {
    const node = byId("editor-status");
    if (node) node.textContent = message;
  }

  function showError(message) {
    const box = byId("editor-error");
    if (!box) return;
    box.textContent = message;
    box.hidden = !message;
  }

  function busy(value) {
    state.busy = value;
    ["validate-graph", "save-candidate", "run-candidate"].forEach(function (id) {
      const button = byId(id);
      if (button) button.disabled = value;
    });
  }

  function graphIdFromQuery() {
    const params = new URLSearchParams(window.location.search);
    return (params.get("id") || "").trim();
  }

  async function boot() {
    if (!MG || typeof MG.api !== "function" || typeof MG.el !== "function") {
      setStatus("The shared UI helpers did not load. Restart `mg ui`.");
      return;
    }
    try {
      MG.nav("graphs");
    } catch (failure) {
      console.warn("Navigation failed to render.", failure);
    }

    byId("validate-graph").addEventListener("click", validate);
    byId("save-candidate").addEventListener("click", saveCandidate);
    byId("run-candidate").addEventListener("click", runCandidate);
    byId("new-graph").addEventListener("click", newGraph);
    byId("graph-yaml").addEventListener("input", function () {
      setStatus("Unsaved changes.");
      showError("");
    });

    state.graphId = graphIdFromQuery() || null;
    if (state.graphId) {
      await loadGraph(state.graphId);
    } else {
      await loadTemplate();
    }
  }

  async function loadGraph(id) {
    setStatus("Loading graph source…");
    try {
      const payload = await MG.api(`/api/graphs/${encodeURIComponent(id)}/source`);
      state.graphId = payload.graph_id || id;
      setEditorValue(payload.yaml || "");
      updateLinks();
      await validate();
    } catch (failure) {
      showError(describe(failure, "Could not load the graph source."));
      setStatus("Graph source unavailable.");
    }
  }

  async function loadTemplate() {
    setStatus("Loading a blank graph…");
    try {
      const payload = await MG.api("/api/graph-template");
      state.graphId = null;
      setEditorValue(payload.yaml || "");
      updateLinks();
      await validate();
    } catch (failure) {
      showError(describe(failure, "Could not load the graph template."));
    }
  }

  function setEditorValue(value) {
    const editor = byId("graph-yaml");
    if (editor) editor.value = value;
    state.source = value;
  }

  function updateLinks() {
    const view = byId("view-graph");
    const crumb = byId("editor-graph-link");
    if (state.graphId) {
      const href = "graph.html?id=" + encodeURIComponent(state.graphId);
      if (view) {
        view.href = href;
        view.hidden = false;
      }
      if (crumb) crumb.href = href;
    } else {
      if (view) view.hidden = true;
      if (crumb) crumb.href = "index.html";
    }
  }

  async function newGraph() {
    if (state.busy) return;
    await loadTemplate();
    setStatus("New graph template loaded.");
  }

  async function validate() {
    if (state.busy) return null;
    const text = byId("graph-yaml").value;
    if (!text.trim()) {
      showError("The graph YAML is empty.");
      return null;
    }
    busy(true);
    showError("");
    setStatus("Validating…");
    try {
      const result = await MG.api("/api/graphs/validate", {
        method: "POST",
        body: { yaml: text },
      });
      state.source = text;
      renderValidation(result);
      if (!result.valid) {
        setStatus("Validation found errors.");
      } else {
        setStatus("Valid graph.");
      }
      return result;
    } catch (failure) {
      showError(describe(failure, "Validation failed."));
      setStatus("Validation unavailable.");
      return null;
    } finally {
      busy(false);
    }
  }

  function renderValidation(result) {
    const output = byId("validation-output");
    const preview = byId("graph-preview");
    if (!output || !preview) return;
    const children = [];
    if (result.valid) {
      children.push(MG.el("p", { class: "editor-ok" }, "Valid graph."));
    } else {
      children.push(MG.el("p", { class: "editor-bad" }, "Invalid graph."));
    }
    const errors = Array.isArray(result.errors) ? result.errors : [];
    const warnings = Array.isArray(result.warnings) ? result.warnings : [];
    if (errors.length) {
      children.push(MG.el("h3", {}, "Errors"));
      children.push(MG.el("ul", {}, ...errors.map((item) => MG.el("li", {}, MG.text(item)))));
    }
    if (warnings.length) {
      children.push(MG.el("h3", {}, "Warnings"));
      children.push(MG.el("ul", {}, ...warnings.map((item) => MG.el("li", {}, MG.text(item)))));
    }
    if (!errors.length && !warnings.length) {
      children.push(MG.el("p", { class: "muted" }, "No diagnostics."));
    }
    output.replaceChildren(...children);
    renderPreview(preview, result.graph);
  }

  function nodeList(graph) {
    if (!graph) return [];
    if (Array.isArray(graph.nodes)) return graph.nodes;
    if (graph.nodes && typeof graph.nodes === "object") {
      return Object.keys(graph.nodes).map((key) =>
        Object.assign({ id: key }, graph.nodes[key] || {}),
      );
    }
    return [];
  }

  function renderPreview(container, graph) {
    if (!graph) {
      container.replaceChildren(MG.el("p", { class: "muted" }, "No compiled graph."));
      return;
    }
    const nodes = nodeList(graph).sort((left, right) =>
      String(left.id).localeCompare(String(right.id)),
    );
    const edges = Array.isArray(graph.edges) ? graph.edges : [];
    const parts = [
      MG.el(
        "p",
        { class: "muted" },
        `entry ${MG.text(graph.entry)} · ${nodes.length} nodes · ${edges.length} edges`,
      ),
    ];
    if (nodes.length) {
      parts.push(
        MG.el(
          "ul",
          { class: "editor-list" },
          ...nodes.map((node) =>
            MG.el(
              "li",
              {},
              MG.el("strong", {}, MG.text(node.id)),
              " ",
              MG.el("span", { class: "muted" }, MG.text(node.kind)),
              node.model || node.model_ref
                ? MG.el("span", { class: "editor-model" }, MG.text(node.model || node.model_ref))
                : null,
            ),
          ),
        ),
      );
    }
    if (edges.length) {
      parts.push(MG.el("h3", {}, "Edges"));
      parts.push(
        MG.el(
          "ul",
          { class: "editor-list" },
          ...edges.map((edge) =>
            MG.el(
              "li",
              {},
              MG.el("code", {}, MG.text(`${edge.from} → ${edge.to}`)),
              edge.when ? MG.el("div", { class: "muted" }, MG.text(edge.when)) : null,
            ),
          ),
        ),
      );
    }
    container.replaceChildren(...parts);
  }

  async function saveCandidate() {
    if (state.busy) return;
    const text = byId("graph-yaml").value;
    busy(true);
    showError("");
    setStatus("Saving candidate…");
    try {
      const result = await MG.api("/api/graphs/candidates", {
        method: "POST",
        body: { yaml: text },
      });
      const output = byId("candidate-output");
      if (output) {
        output.replaceChildren(
          MG.el("p", { class: "editor-ok" }, "Candidate saved locally."),
          MG.el("p", {}, "ID: ", MG.el("code", {}, MG.text(result.candidate_id))),
          MG.el("p", {}, "Path: ", MG.el("code", {}, MG.text(result.path))),
          MG.el("p", { class: "muted" }, "This is not a live deployment."),
          ...(Array.isArray(result.warnings) && result.warnings.length
            ? [MG.el("ul", {}, ...result.warnings.map((item) => MG.el("li", {}, MG.text(item))))]
            : []),
        );
      }
      setStatus("Candidate saved; the live graph is unchanged.");
    } catch (failure) {
      showError(describe(failure, "Could not save the candidate."));
      setStatus("Save failed.");
    } finally {
      busy(false);
    }
  }

  async function runCandidate() {
    if (state.busy) return;
    const text = byId("graph-yaml").value;
    const task = byId("task-input").value;
    if (!task.trim()) {
      showError("Enter a smoke-test task.");
      return;
    }
    busy(true);
    showError("");
    setStatus("Running candidate with the mock provider…");
    try {
      const result = await MG.api("/api/graphs/run-candidate", {
        method: "POST",
        body: { yaml: text, task, provider: "mock" },
      });
      const output = byId("run-output");
      if (output) {
        const artifacts = Array.isArray(result.artifacts) ? result.artifacts : [];
        output.replaceChildren(
          MG.el("p", { class: "editor-ok" }, `Run ${MG.text(result.outcome)}.`),
          MG.el("p", {}, "Run: ", MG.el("code", {}, MG.text(result.run_id))),
          ...artifacts.map((artifact) =>
            MG.el(
              "details",
              {},
              MG.el("summary", {}, `${artifact.name} (${artifact.type})`),
              MG.el("pre", { class: "editor-artifact" }, MG.text(artifact.content)),
            ),
          ),
        );
      }
      setStatus("Mock run completed.");
    } catch (failure) {
      showError(describe(failure, "Candidate run failed."));
      setStatus("Run failed.");
    } finally {
      busy(false);
    }
  }

  function describe(failure, fallback) {
    if (!failure) return fallback;
    if (typeof failure === "string") return failure;
    const code = failure.code ? `${failure.code}: ` : "";
    return code + (failure.message || fallback);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot, { once: true });
  } else {
    boot();
  }
})();
