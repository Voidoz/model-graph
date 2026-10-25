/* Dashboard graph catalog. */
(function () {
  "use strict";

  const MG = window.MG;

  function idOf(graph) {
    return graph && graph.id !== undefined && graph.id !== null ? String(graph.id) : "";
  }

  function warningsOf(graph) {
    return Array.isArray(graph && graph.warnings) ? graph.warnings : [];
  }

  function countOf(graph, countKey, collectionKey) {
    const direct = graph ? Number(graph[countKey]) : NaN;
    if (Number.isFinite(direct)) return direct;
    const collection = graph ? graph[collectionKey] : null;
    if (Array.isArray(collection)) return collection.length;
    if (collection && typeof collection === "object") return Object.keys(collection).length;
    return 0;
  }

  function stat(label, value) {
    return MG.el(
      "div",
      { class: "stat" },
      MG.el("dt", { text: label }),
      MG.el("dd", { text: MG.number(value) }),
    );
  }

  function graphCard(graph) {
    const id = idOf(graph);
    const version = graph && graph.version ? String(graph.version) : "";
    const name = graph && graph.name ? String(graph.name) : id || "Untitled graph";
    const description = graph && graph.description ? String(graph.description) : "";
    const warnings = warningsOf(graph);
    const openHref = id ? "graph.html?id=" + encodeURIComponent(id) : "index.html";
    const runHref = id ? "run.html?graph=" + encodeURIComponent(id) : "run.html";
    const domId = "graph-" + (id || "untitled").replace(/[^a-zA-Z0-9_-]+/g, "-");

    const card = MG.el("article", { class: "card graph-card", "aria-labelledby": domId });

    const titleLink = MG.el("a", { href: openHref, text: name });
    const head = MG.el(
      "div",
      { class: "graph-card-head" },
      MG.el("h2", { class: "graph-card-title", id: domId }, titleLink),
      MG.el("p", { class: "graph-id", text: version ? id + "@" + version : id }),
    );
    card.appendChild(head);

    card.appendChild(
      MG.el("p", {
        class: "graph-desc",
        text: description || "No description was provided for this graph.",
      }),
    );

    card.appendChild(
      MG.el(
        "dl",
        { class: "graph-stats" },
        stat("Nodes", countOf(graph, "node_count", "nodes")),
        stat("Edges", countOf(graph, "edge_count", "edges")),
      ),
    );

    if (warnings.length) {
      const summary = MG.el(
        "summary",
        {},
        MG.el("span", {
          class: "badge badge-warning",
          text: warnings.length + (warnings.length === 1 ? " warning" : " warnings"),
        }),
      );
      card.appendChild(
        MG.el(
          "details",
          { class: "card-warnings" },
          summary,
          MG.warningList(warnings),
        ),
      );
    }

    if (graph && graph.hash) {
      card.appendChild(
        MG.el("p", {
          class: "graph-id hash",
          title: String(graph.hash),
          text: MG.short(String(graph.hash), 30),
        }),
      );
    }

    card.appendChild(
      MG.el(
        "div",
        { class: "card-actions" },
        MG.el("a", { class: "button small", href: openHref, text: "Open graph" }),
        MG.el("a", { class: "button small secondary", href: runHref, text: "Run" }),
      ),
    );

    return card;
  }

  async function load() {
    const status = document.getElementById("graphs-status");
    const errorBox = document.getElementById("graphs-error");
    const emptyBox = document.getElementById("graphs-empty");
    const grid = document.getElementById("graphs");
    const summary = document.getElementById("graphs-summary");
    if (!status || !errorBox || !emptyBox || !grid) return;

    errorBox.hidden = true;
    errorBox.textContent = "";
    emptyBox.hidden = true;
    if (summary) {
      summary.hidden = true;
      summary.textContent = "";
    }
    grid.textContent = "";
    MG.showLoading(status, "Loading compiled graphs…");

    try {
      const payload = await MG.api("/api/graphs");
      const graphs = payload && Array.isArray(payload.graphs) ? payload.graphs : null;
      if (!graphs) {
        throw new MG.ApiError(
          "MG-CONTRACT",
          "The graph list response did not contain a graphs array.",
        );
      }

      status.hidden = true;
      if (!graphs.length) {
        emptyBox.hidden = false;
        return;
      }

      let nodeTotal = 0;
      let edgeTotal = 0;
      let warningTotal = 0;
      graphs.forEach(function (graph) {
        nodeTotal += countOf(graph, "node_count", "nodes");
        edgeTotal += countOf(graph, "edge_count", "edges");
        warningTotal += warningsOf(graph).length;
        grid.appendChild(graphCard(graph));
      });
      grid.hidden = false;

      if (summary) {
        const parts = [
          MG.number(graphs.length) + (graphs.length === 1 ? " graph" : " graphs"),
          MG.number(nodeTotal) + (nodeTotal === 1 ? " node" : " nodes"),
          MG.number(edgeTotal) + (edgeTotal === 1 ? " edge" : " edges"),
        ];
        if (warningTotal) {
          parts.push(MG.number(warningTotal) + (warningTotal === 1 ? " warning" : " warnings"));
        }
        summary.textContent = parts.join(" · ");
        summary.hidden = false;
      }
    } catch (error) {
      status.hidden = true;
      MG.showError(errorBox, error, {
        title: "Could not load graphs",
        onRetry: load,
      });
    }
  }

  MG.ready(function () {
    MG.nav("dashboard");
    load();
  });
})();
