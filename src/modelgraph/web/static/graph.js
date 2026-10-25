/* Graph canvas, deterministic layered layout, and node details. */
(function (global) {
  "use strict";

  const MG = global.MG || (global.MG = {});
  const SVG_NS = "http://www.w3.org/2000/svg";
  const NODE_WIDTH = 190;
  const NODE_HEIGHT = 78;
  const COLUMN_GAP = 56;
  const LAYER_GAP = 96;
  const MARGIN_X = 36;
  const MARGIN_Y = 36;
  const MIN_WIDTH = 660;
  const KINDS = ["router", "worker", "joiner", "subgraph"];

  function asNodeList(graph) {
    const raw = graph && graph.nodes;
    if (Array.isArray(raw)) {
      return raw
        .filter(function (node) {
          return node && node.id !== undefined && node.id !== null;
        })
        .map(function (node) {
          return Object.assign({}, node, { id: String(node.id) });
        })
        .sort(function (left, right) {
          return left.id < right.id ? -1 : left.id > right.id ? 1 : 0;
        });
    }
    if (raw && typeof raw === "object") {
      return Object.keys(raw)
        .sort()
        .map(function (key) {
          const node = raw[key] || {};
          return Object.assign({}, node, {
            id: String(node.id !== undefined && node.id !== null ? node.id : key),
          });
        });
    }
    return [];
  }

  function asEdgeList(graph) {
    const raw = graph && Array.isArray(graph.edges) ? graph.edges : [];
    return raw
      .map(function (edge) {
        if (!edge || typeof edge !== "object") return null;
        const from = edge.from !== undefined ? edge.from : edge.from_;
        const to = edge.to;
        if (from === undefined || from === null || to === undefined || to === null) return null;
        return {
          from: String(from),
          to: String(to),
          when: typeof edge.when === "string" ? edge.when : null,
          examples: Array.isArray(edge.examples) ? edge.examples.map(String) : [],
          artifact: edge.artifact === undefined ? null : edge.artifact,
          priority: Number.isFinite(Number(edge.priority)) ? Number(edge.priority) : 0,
        };
      })
      .filter(Boolean);
  }

  /**
   * Deterministic top-down layered layout.
   *
   * Layer assignment is the longest path from any source, with the entry at
   * layer 0. Nodes in one layer are sorted by id. No physics, no randomness.
   */
  function computeLayout(graph, options) {
    const opts = Object.assign(
      {
        nodeWidth: NODE_WIDTH,
        nodeHeight: NODE_HEIGHT,
        columnGap: COLUMN_GAP,
        layerGap: LAYER_GAP,
        marginX: MARGIN_X,
        marginY: MARGIN_Y,
        minWidth: MIN_WIDTH,
      },
      options || {},
    );
    const nodes = asNodeList(graph);
    const byId = {};
    nodes.forEach(function (node) {
      byId[node.id] = node;
    });
    const ids = nodes.map(function (node) {
      return node.id;
    });
    if (!ids.length) {
      return {
        width: opts.minWidth,
        height: opts.marginY * 2 + opts.nodeHeight,
        positions: {},
        layers: {},
        nodes: [],
        edges: [],
        nodeWidth: opts.nodeWidth,
        nodeHeight: opts.nodeHeight,
      };
    }

    const edges = asEdgeList(graph).filter(function (edge) {
      return byId[edge.from] && byId[edge.to] && edge.from !== edge.to;
    });

    const incoming = {};
    const outgoing = {};
    ids.forEach(function (id) {
      incoming[id] = 0;
      outgoing[id] = [];
    });
    edges.forEach(function (edge) {
      outgoing[edge.from].push(edge.to);
      incoming[edge.to] += 1;
    });

    const layer = {};
    ids.forEach(function (id) {
      layer[id] = 0;
    });

    const queue = ids
      .filter(function (id) {
        return incoming[id] === 0;
      })
      .sort();
    const ordered = [];
    while (queue.length) {
      queue.sort();
      const current = queue.shift();
      ordered.push(current);
      outgoing[current]
        .slice()
        .sort()
        .forEach(function (target) {
          layer[target] = Math.max(layer[target], layer[current] + 1);
          incoming[target] -= 1;
          if (incoming[target] === 0) queue.push(target);
        });
    }

    const reached = new Set(ordered);
    const unreached = ids.filter(function (id) {
      return !reached.has(id);
    });
    if (unreached.length) {
      const base = ordered.length
        ? Math.max.apply(
            null,
            ordered.map(function (id) {
              return layer[id];
            }),
          ) + 1
        : 0;
      unreached.forEach(function (id) {
        layer[id] = base;
      });
    }

    if (graph && graph.entry && layer[graph.entry] !== undefined) {
      layer[graph.entry] = 0;
    }

    const maxLayer = Math.max.apply(
      null,
      ids.map(function (id) {
        return layer[id];
      }),
    );
    const grouped = [];
    for (let index = 0; index <= maxLayer; index += 1) grouped.push([]);
    ids
      .slice()
      .sort()
      .forEach(function (id) {
        grouped[layer[id]].push(id);
      });

    const widest = Math.max.apply(
      null,
      grouped.map(function (group) {
        return group.length;
      }),
    );
    const contentWidth =
      widest * opts.nodeWidth + Math.max(0, widest - 1) * opts.columnGap;
    const width = Math.max(opts.minWidth, contentWidth + opts.marginX * 2);
    const height =
      opts.marginY * 2 +
      (maxLayer + 1) * opts.nodeHeight +
      maxLayer * opts.layerGap;

    const positions = {};
    grouped.forEach(function (group, layerIndex) {
      const groupWidth =
        group.length * opts.nodeWidth + Math.max(0, group.length - 1) * opts.columnGap;
      const startX = (width - groupWidth) / 2;
      group.forEach(function (id, columnIndex) {
        positions[id] = {
          x: startX + columnIndex * (opts.nodeWidth + opts.columnGap),
          y: opts.marginY + layerIndex * (opts.nodeHeight + opts.layerGap),
          width: opts.nodeWidth,
          height: opts.nodeHeight,
          layer: layerIndex,
          column: columnIndex,
        };
      });
    });

    const laidOutEdges = edges.map(function (edge) {
      return Object.assign({}, edge, {
        fromPos: positions[edge.from],
        toPos: positions[edge.to],
      });
    });

    return {
      width: width,
      height: height,
      positions: positions,
      layers: Object.assign({}, layer),
      nodes: ids.map(function (id) {
        return Object.assign({ node: byId[id], id: id }, positions[id]);
      }),
      edges: laidOutEdges,
      nodeWidth: opts.nodeWidth,
      nodeHeight: opts.nodeHeight,
    };
  }

  function shorten(value, maxLength) {
    const limit = Number.isFinite(maxLength) && maxLength > 4 ? maxLength : 40;
    const raw = String(value === undefined || value === null ? "" : value)
      .replace(/\s+/g, " ")
      .trim();
    if (raw.length <= limit) return raw;
    return raw.slice(0, limit - 1).trimEnd() + "…";
  }

  function kindClass(kind) {
    const known = KINDS.indexOf(kind) >= 0 ? kind : "unknown";
    return "kind-" + known;
  }

  function kindLabel(kind) {
    if (!kind) return "unknown";
    const value = String(kind);
    return value.charAt(0).toUpperCase() + value.slice(1);
  }

  /* ------------------------------------------------------------------ */
  /* SVG rendering                                                       */
  /* ------------------------------------------------------------------ */

  function svgNode(tag, attrs) {
    const element = document.createElementNS(SVG_NS, tag);
    if (attrs && typeof attrs === "object") {
      Object.keys(attrs).forEach(function (key) {
        const value = attrs[key];
        if (value === null || value === undefined || value === false) return;
        element.setAttribute(key, String(value));
      });
    }
    for (let index = 2; index < arguments.length; index += 1) {
      const child = arguments[index];
      if (child === null || child === undefined || child === false) continue;
      if (Array.isArray(child)) {
        child.forEach(function (item) {
          if (item !== null && item !== undefined) element.appendChild(item);
        });
      } else if (typeof child === "string" || typeof child === "number") {
        element.appendChild(document.createTextNode(String(child)));
      } else {
        element.appendChild(child);
      }
    }
    return element;
  }

  function svgText(tag, attrs, value) {
    const element = svgNode(tag, attrs);
    element.textContent = String(value === undefined || value === null ? "" : value);
    return element;
  }

  function addMarker(defs, id, color) {
    const marker = svgNode("marker", {
      id: id,
      viewBox: "0 0 10 8",
      refX: "9",
      refY: "4",
      markerWidth: "9",
      markerHeight: "7",
      markerUnits: "userSpaceOnUse",
      orient: "auto",
    });
    marker.appendChild(svgNode("path", { d: "M 0 0 L 10 4 L 0 8 z", fill: color }));
    defs.appendChild(marker);
  }

  function edgeGeometry(edge) {
    const from = edge.fromPos;
    const to = edge.toPos;
    if (!from || !to) return null;
    if (to.layer > from.layer) {
      const startX = from.x + from.width / 2;
      const startY = from.y + from.height;
      const endX = to.x + to.width / 2;
      const endY = to.y;
      const midY = (startY + endY) / 2;
      return {
        d:
          "M " +
          startX +
          " " +
          startY +
          " C " +
          startX +
          " " +
          midY +
          ", " +
          endX +
          " " +
          midY +
          ", " +
          endX +
          " " +
          endY,
        labelX: (startX + endX) / 2,
        labelY: (startY + endY) / 2,
      };
    }

    /* Same-layer or backward edge (invalid in a v0 DAG, but render defensively). */
    const startX = from.x + from.width;
    const startY = from.y + from.height / 2;
    const endX = to.x + to.width;
    const endY = to.y + to.height / 2;
    const bow = Math.max(36, Math.abs(endY - startY) / 2 + 24);
    return {
      d:
        "M " +
        startX +
        " " +
        startY +
        " C " +
        (startX + bow) +
        " " +
        startY +
        ", " +
        (endX + bow) +
        " " +
        endY +
        ", " +
        endX +
        " " +
        endY,
      labelX: Math.max(startX, endX) + bow * 0.72,
      labelY: (startY + endY) / 2,
    };
  }

  function clear(node) {
    if (node) node.textContent = "";
  }

  function renderLegend(container, graphNodes) {
    if (!container) return;
    clear(container);
    const present = new Set(
      graphNodes.map(function (node) {
        return String(node.kind || "unknown");
      }),
    );
    const kinds = KINDS.filter(function (kind) {
      return present.has(kind);
    });
    const unknown = Array.from(present).filter(function (kind) {
      return KINDS.indexOf(kind) < 0;
    });
    kinds.concat(unknown).forEach(function (kind) {
      container.appendChild(
        MG.el(
          "li",
          {},
          MG.el("span", {
            class: "legend-swatch " + kindClass(kind),
            "aria-hidden": "true",
          }),
          MG.el("span", { text: kindLabel(kind) }),
        ),
      );
    });
  }

  function renderSvg(svg, graph, layout, warnings) {
    clear(svg);
    const graphName = graph.name ? String(graph.name) : graph.id ? String(graph.id) : "graph";
    const title = svgText("title", { id: "graph-canvas-title" }, "Layered graph for " + graphName);
    svg.appendChild(title);
    svg.setAttribute("viewBox", "0 0 " + layout.width + " " + layout.height);
    svg.setAttribute("width", String(layout.width));
    svg.setAttribute("height", String(layout.height));

    const defs = svgNode("defs");
    addMarker(defs, "mg-arrow", "#8592a5");
    addMarker(defs, "mg-arrow-active", "#0a5cc2");
    svg.appendChild(defs);

    const edgeLayer = svgNode("g", { class: "mg-edge-layer" });
    const nodeLayer = svgNode("g", { class: "mg-node-layer" });
    svg.appendChild(edgeLayer);
    svg.appendChild(nodeLayer);

    const entry = graph.entry ? String(graph.entry) : "";
    const exits = new Set(
      (Array.isArray(graph.exit) ? graph.exit : []).map(function (name) {
        return String(name);
      }),
    );
    const fallback = graph.fallback ? String(graph.fallback) : "";
    const nodeElements = {};
    const edgeElements = [];
    const labelsByTarget = {};

    function deselectAll() {
      Object.keys(nodeElements).forEach(function (id) {
        nodeElements[id].classList.remove("is-selected");
        nodeElements[id].removeAttribute("aria-current");
      });
      edgeElements.forEach(function (element) {
        element.classList.remove("is-selected");
        element.removeAttribute("aria-current");
        const line = element.querySelector(".edge-line");
        if (line) line.setAttribute("marker-end", "url(#mg-arrow)");
      });
    }

    function selectNode(node, element) {
      deselectAll();
      element.classList.add("is-selected");
      element.setAttribute("aria-current", "true");
      showNodeDetails(node, {
        graph: graph,
        warnings: warnings,
        isEntry: node.id === entry,
        isExit: nodeHasExit(node, exits),
        isFallback: node.id === fallback,
      });
    }

    function selectEdge(edge, element) {
      deselectAll();
      element.classList.add("is-selected");
      element.setAttribute("aria-current", "true");
      const line = element.querySelector(".edge-line");
      if (line) line.setAttribute("marker-end", "url(#mg-arrow-active)");
      showEdgeDetails(edge, graph);
    }

    layout.edges.forEach(function (edge, index) {
      const geometry = edgeGeometry(edge);
      if (!geometry) return;
      const fromName = edge.from;
      const toName = edge.to;
      const whenText = edge.when ? String(edge.when) : "";
      const label = whenText ? shorten(whenText, 34) : "";
      const aria =
        "Edge from " +
        fromName +
        " to " +
        toName +
        (whenText ? ": " + whenText : "");

      const group = svgNode("g", {
        class: "mg-edge",
        tabindex: "0",
        role: "button",
        "aria-label": aria,
        "data-edge-index": String(index),
      });
      const edgeTitle = svgText("title", {}, aria);
      group.appendChild(edgeTitle);
      group.appendChild(
        svgNode("path", { class: "edge-hit", d: geometry.d }),
      );
      group.appendChild(
        svgNode("path", {
          class: "edge-line",
          d: geometry.d,
          "marker-end": "url(#mg-arrow)",
        }),
      );

      if (label) {
        let labelX = geometry.labelX;
        let labelY = geometry.labelY;
        if (edge.toPos.layer > edge.fromPos.layer) {
          /* Keep sibling labels apart: anchor each label just above its target. */
          const stack = labelsByTarget[edge.to] || 0;
          labelsByTarget[edge.to] = stack + 1;
          labelX = edge.toPos.x + edge.toPos.width / 2;
          labelY = Math.max(
            edge.fromPos.y + edge.fromPos.height + 20,
            edge.toPos.y - 18 - stack * 22,
          );
        }
        const labelWidth = Math.min(250, Math.max(42, label.length * 6.1 + 14));
        const clampedX = Math.max(
          labelWidth / 2 + 6,
          Math.min(layout.width - labelWidth / 2 - 6, labelX),
        );
        group.appendChild(
          svgNode("rect", {
            class: "edge-label-bg",
            x: clampedX - labelWidth / 2,
            y: labelY - 10,
            width: labelWidth,
            height: 20,
            rx: 6,
          }),
        );
        group.appendChild(
          svgText(
            "text",
            {
              class: "edge-label",
              x: clampedX,
              y: labelY + 3.5,
              "text-anchor": "middle",
            },
            label,
          ),
        );
      }

      group.addEventListener("click", function () {
        selectEdge(edge, group);
      });
      group.addEventListener("keydown", function (event) {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          selectEdge(edge, group);
        }
      });

      edgeElements.push(group);
      edgeLayer.appendChild(group);
    });

    layout.nodes.forEach(function (item) {
      const node = item.node;
      const id = item.id;
      const kind = node.kind ? String(node.kind) : "unknown";
      const isEntry = id === entry;
      const isExit = nodeHasExit(node, exits);
      const badges = [];
      if (isEntry) badges.push("ENTRY");
      if (isExit) badges.push("EXIT");
      const subtitle = node.model_ref
        ? String(node.model_ref)
        : node.graph_ref
          ? String(node.graph_ref)
          : kind === "subgraph"
            ? "nested graph"
            : "no model";

      const group = svgNode("g", {
        class:
          "mg-node " +
          kindClass(kind) +
          (isEntry ? " is-entry" : "") +
          (isExit ? " is-exit" : ""),
        transform: "translate(" + item.x + " " + item.y + ")",
        tabindex: "0",
        role: "button",
        "aria-label":
          kindLabel(kind) +
          " node " +
          id +
          (node.description ? ". " + String(node.description) : ""),
        "data-node-id": id,
      });
      group.appendChild(
        svgText(
          "title",
          {},
          kindLabel(kind) +
            " " +
            id +
            (subtitle ? " · " + subtitle : "") +
            (node.description ? "\n" + String(node.description) : ""),
        ),
      );
      group.appendChild(
        svgNode("rect", {
          class: "node-body",
          x: "0",
          y: "0",
          width: String(layout.nodeWidth),
          height: String(layout.nodeHeight),
          rx: "11",
        }),
      );
      group.appendChild(
        svgNode("rect", {
          class: "node-accent",
          x: "0",
          y: "0",
          width: "6",
          height: String(layout.nodeHeight),
          rx: "3",
        }),
      );
      group.appendChild(
        svgText(
          "text",
          { class: "node-kind", x: "16", y: "20" },
          kind.toUpperCase(),
        ),
      );
      group.appendChild(
        svgText(
          "text",
          { class: "node-id", x: "16", y: "43" },
          shorten(id, 22),
        ),
      );
      group.appendChild(
        svgText(
          "text",
          { class: "node-sub", x: "16", y: "61" },
          shorten(subtitle, 27),
        ),
      );

      if (badges.length) {
        const badgeText = badges.join(" · ");
        const badgeWidth = badgeText.length * 4.9 + 12;
        const badgeX = layout.nodeWidth - 10 - badgeWidth;
        group.appendChild(
          svgNode("rect", {
            class: "node-badge-bg",
            x: String(badgeX),
            y: "9",
            width: String(badgeWidth),
            height: "14",
            rx: "7",
          }),
        );
        group.appendChild(
          svgText(
            "text",
            {
              class: "node-badge",
              x: String(badgeX + badgeWidth / 2),
              y: "19",
              "text-anchor": "middle",
            },
            badgeText,
          ),
        );
      }

      group.addEventListener("click", function () {
        selectNode(node, group);
      });
      group.addEventListener("keydown", function (event) {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          selectNode(node, group);
        }
      });

      nodeElements[id] = group;
      nodeLayer.appendChild(group);
    });

    const entryNode = layout.nodes.filter(function (item) {
      return item.id === entry;
    })[0];
    if (entryNode && nodeElements[entryNode.id]) {
      selectNode(entryNode.node, nodeElements[entryNode.id]);
    } else if (layout.nodes.length && nodeElements[layout.nodes[0].id]) {
      selectNode(layout.nodes[0].node, nodeElements[layout.nodes[0].id]);
    }

    return { nodeElements: nodeElements, edgeElements: edgeElements };
  }

  function nodeHasExit(node, exits) {
    if (!exits || !exits.size) return false;
    const writes = Array.isArray(node.writes) ? node.writes : [];
    return writes.some(function (write) {
      const name =
        typeof write === "string"
          ? write
          : write && write.name !== undefined && write.name !== null
            ? String(write.name)
            : "";
      return exits.has(name);
    });
  }

  /* ------------------------------------------------------------------ */
  /* Details panel                                                       */
  /* ------------------------------------------------------------------ */

  function textNode(value) {
    return document.createTextNode(MG.text(value));
  }

  function appendDetailValue(element, value) {
    if (Array.isArray(value)) {
      value.forEach(function (item, index) {
        if (index) element.appendChild(document.createTextNode(", "));
        appendDetailValue(element, item);
      });
      return;
    }
    if (value && typeof value.nodeType === "number") {
      element.appendChild(value);
      return;
    }
    element.appendChild(textNode(value));
  }

  function detailRow(list, label, value) {
    list.appendChild(MG.el("dt", { text: label }));
    const dd = MG.el("dd");
    appendDetailValue(dd, value);
    list.appendChild(dd);
  }

  function section(title, child) {
    return MG.el(
      "section",
      { class: "detail-section" },
      MG.el("h4", { text: title }),
      child,
    );
  }

  function writesList(writes) {
    const items = Array.isArray(writes) ? writes : [];
    if (!items.length) return MG.el("p", { class: "muted", text: "No artifacts written." });
    const list = MG.el("ul", { class: "plain-list" });
    items.forEach(function (write) {
      const entry = typeof write === "string" ? { name: write, type: "text" } : write || {};
      const meta = [
        entry.type ? String(entry.type) : "unknown type",
        entry.media_type ? String(entry.media_type) : "",
        entry.strict_json ? "strict json" : "",
      ].filter(Boolean);
      list.appendChild(
        MG.el(
          "li",
          {},
          MG.el("code", { text: entry.name ? String(entry.name) : "unnamed" }),
          MG.el("span", { class: "muted", text: " · " + meta.join(" · ") }),
        ),
      );
    });
    return list;
  }

  function mapList(mapping) {
    if (!mapping || typeof mapping !== "object" || !Object.keys(mapping).length) {
      return MG.el("p", { class: "muted", text: "No mapping." });
    }
    const list = MG.el("ul", { class: "plain-list" });
    Object.keys(mapping)
      .sort()
      .forEach(function (key) {
        list.appendChild(
          MG.el(
            "li",
            {},
            MG.el("code", { text: key }),
            MG.el("span", { class: "muted", text: " → " }),
            MG.el("code", { text: String(mapping[key]) }),
          ),
        );
      });
    return list;
  }

  function limitsSection(limits) {
    if (!limits || typeof limits !== "object" || !Object.keys(limits).length) return null;
    const list = MG.el("dl", { class: "detail-list" });
    Object.keys(limits).forEach(function (key) {
      const value = limits[key];
      if (value === null || value === undefined) return;
      const formatted =
        typeof value === "boolean"
          ? MG.text(value)
          : typeof value === "object"
            ? MG.text(value)
            : value;
      detailRow(list, key.replace(/_/g, " "), formatted);
    });
    return list;
  }

  function showNodeDetails(node, context) {
    const container = document.getElementById("node-details");
    if (!container) return;
    clear(container);
    const graph = context.graph || {};
    const warnings = Array.isArray(context.warnings) ? context.warnings : [];
    const kind = node.kind ? String(node.kind) : "unknown";
    const id = node.id ? String(node.id) : "unknown";

    const chips = MG.el("div", { class: "chip-row" });
    if (context.isEntry) chips.appendChild(MG.el("span", { class: "badge badge-entry", text: "Entry" }));
    if (context.isExit) chips.appendChild(MG.el("span", { class: "badge badge-exit", text: "Exit writer" }));
    if (context.isFallback) chips.appendChild(MG.el("span", { class: "badge", text: "Fallback" }));

    container.appendChild(
      MG.el(
        "div",
        { class: "details-head" },
        MG.el("h3", { class: "details-title", text: id }),
        MG.el("span", { class: "kind-pill " + kindClass(kind), text: kindLabel(kind) }),
      ),
    );
    if (chips.childNodes.length) container.appendChild(chips);
    container.appendChild(
      MG.el("p", {
        class: "detail-desc",
        text: node.description ? String(node.description) : "No description provided.",
      }),
    );

    const list = MG.el("dl", { class: "detail-list" });
    detailRow(list, "Id", id);
    detailRow(list, "Kind", kindLabel(kind));
    if (node.model_ref) detailRow(list, "Model", String(node.model_ref));
    if (node.graph_ref) detailRow(list, "Graph ref", String(node.graph_ref));
    if (node.model && typeof node.model === "object") {
      const modelLabel = [node.model.provider, node.model.model].filter(Boolean).join(" / ");
      if (modelLabel) detailRow(list, "Model detail", modelLabel);
    }
    if (node.model_hash) {
      const hash = String(node.model_hash);
      detailRow(
        list,
        "Model hash",
        MG.el("span", { class: "mono hash", title: hash, text: MG.short(hash, 22) }),
      );
    }
    if (node.strategy && kind === "joiner") detailRow(list, "Strategy", String(node.strategy));
    detailRow(list, "Reads", MG.text(Array.isArray(node.reads) ? node.reads : []));
    container.appendChild(list);

    container.appendChild(section("Writes", writesList(node.writes)));

    const instruction = node.instruction ? String(node.instruction) : "";
    container.appendChild(
      section(
        "Instruction",
        instruction
          ? MG.el("pre", { class: "code-block", text: instruction })
          : MG.el("p", {
              class: "muted",
              text: "No instruction recorded. Routers decide; subgraphs delegate.",
            }),
      ),
    );

    if (kind === "router") {
      const routing = node.routing && typeof node.routing === "object" ? node.routing : {};
      const routingList = MG.el("dl", { class: "detail-list" });
      detailRow(routingList, "Threshold", MG.text(routing.confidence_threshold));
      detailRow(routingList, "Low confidence", MG.text(routing.on_low_confidence));
      detailRow(routingList, "Retries", MG.text(routing.max_retries));
      detailRow(routingList, "Selection", MG.text(routing.selection));
      detailRow(routingList, "Direct answer", MG.text(routing.allow_direct_answer));
      container.appendChild(section("Routing", routingList));
      if (node.routing_card) {
        container.appendChild(
          section(
            "Routing card",
            MG.el("pre", { class: "code-block", text: String(node.routing_card) }),
          ),
        );
      }
    }

    if (kind === "subgraph") {
      container.appendChild(section("Input map", mapList(node.input_map)));
      container.appendChild(section("Output map", mapList(node.output_map)));
    }

    const limits = limitsSection(node.limits);
    if (limits) container.appendChild(section("Limits", limits));

    let warningContent;
    if (warnings.length) {
      warningContent = MG.el("div", { class: "warning-stack" }, MG.warningList(warnings));
      warningContent.appendChild(
        MG.el("p", {
          class: "muted detail-note",
          text: "The API reports graph-level warnings; none are attached to individual nodes.",
        }),
      );
    } else {
      warningContent = MG.el("p", { class: "muted", text: "No warnings reported for this graph." });
    }
    container.appendChild(section("Warnings", warningContent));

    if (graph.defaults || graph.inputs) {
      const graphInfo = MG.el("dl", { class: "detail-list" });
      if (Array.isArray(graph.inputs)) {
        detailRow(
          graphInfo,
          "Graph inputs",
          graph.inputs.map(function (input) {
            return input && input.name ? String(input.name) : String(input);
          }),
        );
      }
      if (graph.defaults && graph.defaults.max_steps !== undefined) {
        detailRow(graphInfo, "Max steps", MG.text(graph.defaults.max_steps));
      }
      container.appendChild(section("Graph contract", graphInfo));
    }
  }

  function showEdgeDetails(edge, graph) {
    const container = document.getElementById("node-details");
    if (!container) return;
    clear(container);
    container.appendChild(
      MG.el(
        "div",
        { class: "details-head" },
        MG.el("h3", { class: "details-title", text: edge.from + " → " + edge.to }),
        MG.el("span", { class: "kind-pill", text: "Edge" }),
      ),
    );
    container.appendChild(
      MG.el("p", { class: "detail-desc", text: "Directed routing edge." }),
    );

    const list = MG.el("dl", { class: "detail-list" });
    detailRow(list, "From", edge.from);
    detailRow(list, "To", edge.to);
    detailRow(list, "Artifact", MG.text(edge.artifact));
    detailRow(list, "Priority", MG.text(edge.priority));
    container.appendChild(list);

    container.appendChild(
      section(
        "When",
        MG.el("p", {
          text: edge.when ? String(edge.when) : "No when label recorded for this edge.",
        }),
      ),
    );
    if (edge.examples && edge.examples.length) {
      const examples = MG.el("ul", { class: "plain-list" });
      edge.examples.forEach(function (example) {
        examples.appendChild(MG.el("li", { text: String(example) }));
      });
      container.appendChild(section("Examples", examples));
    }

    if (graph && graph.entry) {
      container.appendChild(
        section(
          "Graph",
          MG.el("p", {
            class: "muted",
            text:
              "Entry " +
              String(graph.entry) +
              (Array.isArray(graph.exit) ? " · exits: " + graph.exit.join(", ") : ""),
          }),
        ),
      );
    }
  }

  /* ------------------------------------------------------------------ */
  /* Page bootstrap                                                      */
  /* ------------------------------------------------------------------ */

  function graphIdFromLocation() {
    const params = new URLSearchParams(window.location.search);
    const id = params.get("id") || params.get("graph") || "";
    return id.trim();
  }

  function uniqueWarnings(payload, graph) {
    const values = [];
    if (payload && Array.isArray(payload.warnings)) values.push.apply(values, payload.warnings);
    if (graph && Array.isArray(graph.warnings)) values.push.apply(values, graph.warnings);
    const seen = new Set();
    return values.filter(function (warning) {
      const label = MG.warningText(warning);
      if (seen.has(label)) return false;
      seen.add(label);
      return true;
    });
  }

  function chip(label, value, className) {
    return MG.el("span", {
      class: "badge" + (className ? " " + className : ""),
      text: label + ": " + MG.text(value),
    });
  }

  function renderHeader(graph, warnings, detailName, detailHash) {
    const id = graph.id ? String(graph.id) : graphIdFromLocation();
    const version = graph.version ? String(graph.version) : "";
    const name = detailName || (graph.name ? String(graph.name) : "");
    const hash = detailHash || (graph.hash ? String(graph.hash) : "");
    const title = name || id || "Graph";
    const titleElement = document.getElementById("graph-title");
    const idVersion = document.getElementById("graph-id-version");
    const description = document.getElementById("graph-description");
    const crumb = document.getElementById("graph-crumb");
    const meta = document.getElementById("graph-meta");
    const runLink = document.getElementById("graph-run-link");

    document.title = title + " · Model Graph";
    if (titleElement) titleElement.textContent = title;
    if (idVersion) idVersion.textContent = version ? id + "@" + version : id;
    if (description) {
      description.textContent = graph.description
        ? String(graph.description)
        : "No description was provided for this graph.";
    }
    if (crumb) crumb.textContent = id || "Graph";

    if (runLink) {
      runLink.setAttribute("href", "run.html?graph=" + encodeURIComponent(id));
    }

    if (meta) {
      clear(meta);
      const nodes = Array.isArray(graph.nodes)
        ? graph.nodes
        : graph.nodes && typeof graph.nodes === "object"
          ? Object.keys(graph.nodes)
          : [];
      const edges = Array.isArray(graph.edges) ? graph.edges : [];
      meta.appendChild(chip("Nodes", nodes.length));
      meta.appendChild(chip("Edges", edges.length));
      meta.appendChild(chip("Entry", graph.entry || "—"));
      meta.appendChild(chip("Exit", Array.isArray(graph.exit) ? graph.exit : []));
      if (graph.fallback) meta.appendChild(chip("Fallback", graph.fallback));
      if (graph.apiVersion) meta.appendChild(chip("API", graph.apiVersion));
      if (hash) {
        meta.appendChild(
          MG.el("span", {
            class: "badge",
            title: hash,
            text: "Hash: " + MG.short(hash, 16),
          }),
        );
      }
      if (warnings.length) {
        meta.appendChild(
          chip("Warnings", warnings.length, "badge-warning"),
        );
      }
    }
  }

  function renderWarningPanel(warnings) {
    const panel = document.getElementById("graph-warnings");
    const count = document.getElementById("warning-count");
    if (panel) {
      clear(panel);
      panel.appendChild(MG.warningList(warnings));
    }
    if (count) {
      count.textContent = String(warnings.length);
      count.className = warnings.length ? "badge badge-warning" : "badge";
    }
  }

  async function loadGraph() {
    const id = graphIdFromLocation();
    const status = document.getElementById("graph-status");
    const errorBox = document.getElementById("graph-error");
    const emptyBox = document.getElementById("graph-empty");
    const content = document.getElementById("graph-content");
    const crumb = document.getElementById("graph-crumb");
    if (!status || !errorBox || !emptyBox || !content) return;

    errorBox.hidden = true;
    errorBox.textContent = "";
    if (crumb) crumb.textContent = id || "Graph";

    if (!id) {
      status.hidden = true;
      content.hidden = true;
      emptyBox.hidden = false;
      document.title = "Graph · Model Graph";
      return;
    }

    emptyBox.hidden = true;
    content.hidden = true;
    MG.showLoading(status, "Loading graph " + id + "…");

    try {
      const payload = await MG.api("/api/graphs/" + encodeURIComponent(id));
      const graph = payload && payload.graph;
      if (!graph || typeof graph !== "object") {
        throw new MG.ApiError(
          "MG-CONTRACT",
          "The graph detail response did not contain a graph object.",
        );
      }
      const warnings = uniqueWarnings(payload, graph);
      renderHeader(
        graph,
        warnings,
        payload && typeof payload.name === "string" ? payload.name : "",
        payload && typeof payload.hash === "string" ? payload.hash : "",
      );
      renderWarningPanel(warnings);

      const svg = document.getElementById("graph-canvas");
      const legend = document.getElementById("graph-legend");
      if (!svg) {
        throw new MG.ApiError("MG-CONTRACT", "The graph canvas element is missing.");
      }
      const layout = computeLayout(graph);
      renderSvg(svg, graph, layout, warnings);
      renderLegend(
        legend,
        Object.keys(layout.positions).map(function (nodeId) {
          const positioned = layout.nodes.filter(function (item) {
            return item.id === nodeId;
          })[0];
          return positioned ? positioned.node : { id: nodeId, kind: "unknown" };
        }),
      );

      status.hidden = true;
      content.hidden = false;
    } catch (error) {
      status.hidden = true;
      content.hidden = true;
      MG.showError(errorBox, error, {
        title: "Could not load graph",
        onRetry: loadGraph,
      });
    }
  }

  const GraphApi = {
    NODE_WIDTH: NODE_WIDTH,
    NODE_HEIGHT: NODE_HEIGHT,
    computeLayout: computeLayout,
    asNodeList: asNodeList,
    asEdgeList: asEdgeList,
    shorten: shorten,
    kindClass: kindClass,
    kindLabel: kindLabel,
  };
  MG.Graph = Object.assign(MG.Graph || {}, GraphApi);
  if (typeof module !== "undefined" && module.exports) module.exports = GraphApi;

  if (typeof document !== "undefined") {
    MG.ready(function () {
      MG.nav("graph");
      loadGraph();
    });
  }
})(typeof globalThis !== "undefined" ? globalThis : this);
