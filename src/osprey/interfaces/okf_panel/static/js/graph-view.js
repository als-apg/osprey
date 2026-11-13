// @ts-check
/*
 * OKF Knowledge Panel — the Graph view: the facility graph, drawn.
 *
 * The view reads `/api/graph` once, the first time it is opened, and draws
 * one of the two cuts graph-model.js makes of it: the overview (kinds and
 * device classes, with the relations between them counted) or the
 * neighbourhood of one focus node. Its parts, all in index.html's
 * `#graph-view`:
 *
 *   sidebar     the drawing (overview or neighbourhood), the search hits, the
 *               depth, and the legend, whose relation rows are the filter
 *   stage       the drawing and its controls (graph-stage.js)
 *   inspector   the picked node (graph-inspector.js)
 *
 * A click picks a node, a double-click (or "Focus here") draws its
 * neighbourhood, a right-click opens the menu of what it landed on
 * (graph-menus.js). This module holds the state the three parts share and is
 * the only one that changes it.
 */

import { copyText } from '/design-system/js/clipboard.js';
import { openContextMenu } from '/design-system/js/context-menu.js';
import { initPanelSidebar } from '/design-system/js/panel-sidebar.js';
import { renderState } from '/design-system/js/panel-state.js';
import { el } from './helpers.js';
import { nodeRow, renderInspector } from './graph-inspector.js';
import { edgeMenu, nodeMenu, stageMenu } from './graph-menus.js';
import {
  count,
  indexGraph,
  KIND_NAMES,
  layout,
  neighbourhood,
  NODE_KINDS,
  nodeOfRecord,
  overview,
  relationStyle,
  searchNodes,
} from './graph-model.js';
import { createGraphStage, shape, svg } from './graph-stage.js';

/** @typedef {import('./graph-model.js').GraphIndex} GraphIndex */
/** @typedef {import('./graph-model.js').DrawnGraph} DrawnGraph */
/** @typedef {import('./graph-model.js').NodeKey} NodeKey */
/** @typedef {import('./graph-model.js').Point} Point */
/** @typedef {'overview' | 'hood'} Mode */

/**
 * @typedef {object} GraphViewHandles
 * @property {(conceptId: string) => void} onPage open a knowledge page
 * @property {(nodeId: string | null) => void} onLocation the view now shows
 *   this node's neighbourhood, or the overview (null)
 */

/**
 * @typedef {object} GraphViewApi
 * @property {(nodeId: string | null) => Promise<void>} show Draw a node's
 *   neighbourhood, or the overview (null). An id the graph does not hold
 *   draws the overview.
 * @property {(kind: string, id: string) => Promise<boolean>} showRecord Draw
 *   the neighbourhood of the node that stands for a facility record; false
 *   when the graph holds none.
 * @property {(query: string) => Promise<void>} search List the nodes a query
 *   names in the sidebar; an empty query clears the list.
 * @property {() => string | null} location The focus node's id, or null on
 *   the overview.
 * @property {() => void} resetWidth The sidebar goes back to its default width.
 */

const STORAGE_KEY = 'osprey-okf-graph-sidebar';
const SPACING = { overview: 120, hood: 70 };
const GRAVITY = { overview: 0.25, hood: 0.04 };
const DEFAULT_HOPS = 2;
/** How long the status line says what was copied, in ms. */
const COPIED_FOR = 2000;

/**
 * @param {string} id
 * @returns {HTMLElement}
 */
function need(id) {
  const node = document.getElementById(id);
  if (!node) throw new Error('okf_panel: missing required element #' + id);
  return node;
}

/**
 * Press one button of a segmented control.
 * @param {HTMLElement} seg
 * @param {string} attr the data attribute that names a button
 * @param {string} value
 */
function pressSeg(seg, attr, value) {
  for (const button of seg.querySelectorAll('button')) {
    button.setAttribute('aria-pressed', String(button.getAttribute(attr) === value));
  }
}

/**
 * A legend swatch.
 * @param {Record<string, string>} attrs
 * @param {SVGElement} mark
 * @returns {Node}
 */
function swatch(attrs, mark) {
  return svg('svg', { class: 'okf-graph-swatch', 'aria-hidden': 'true', ...attrs }, [mark]);
}

/**
 * Wire the Graph view. Call once at boot, when the panel serves a graph;
 * nothing is fetched until the view is first shown.
 * @param {GraphViewHandles} handles
 * @returns {GraphViewApi}
 */
export function createGraphView({ onPage, onLocation }) {
  const stageEl = need('graph-stage');
  const inspector = need('graph-inspector');
  const modesEl = need('graph-modes');
  const hopsEl = need('graph-hops');
  const depthEl = need('graph-depth');
  const hitsEl = need('graph-hits');
  const relationsEl = need('graph-relations');
  const kindsEl = need('graph-kinds');
  const contextEl = document.getElementById('graph-context');
  const chipEl = document.getElementById('graph-chip');

  const sidebar = initPanelSidebar({
    sidebar: need('graph-sidebar'),
    handle: document.getElementById('graph-resizer'),
    toggle: document.getElementById('graph-toggle'),
    storageKey: STORAGE_KEY,
    defaultOpen: true,
  });

  const stage = createGraphStage(stageEl, {
    onSelect: select,
    onFocus(key) {
      if (typeof key === 'number') focusOn(key);
    },
    onRelayout: () => redraw({ fresh: true }),
    onMenu: openMenu,
  });

  /** @type {GraphIndex | null} */
  let graph = null;
  /** @type {Promise<GraphIndex | null> | null} */
  let loading = null;

  /** @type {Mode} */
  let mode = 'overview';
  /** @type {number | null} */
  let focus = null;
  let hops = DEFAULT_HOPS;
  /** @type {Set<string>} */
  let relations = new Set();
  /** @type {Set<number>} */
  const expanded = new Set();
  /** @type {NodeKey | null} */
  let selected = null;
  /** @type {{overview: Map<NodeKey, Point>, hood: Map<NodeKey, Point>}} */
  const positions = { overview: new Map(), hood: new Map() };
  /** @type {DrawnGraph} */
  let drawn = { nodes: [], edges: [], hidden: 0 };
  /** The legend's checkbox of each relation. @type {Map<string, HTMLInputElement>} */
  const relationBoxes = new Map();
  /** @type {ReturnType<typeof setTimeout> | null} */
  let copiedTimer = null;

  // -- loading ---------------------------------------------------------------

  /** @returns {Promise<GraphIndex | null>} the graph, or null when it did not load */
  function load() {
    if (graph) return Promise.resolve(graph);
    if (loading) return loading;
    renderState(stageEl, { icon: 'refresh', message: 'Loading the graph' });
    loading = (async () => {
      try {
        const resp = await fetch('/api/graph');
        if (!resp.ok) throw new Error('HTTP ' + resp.status);
        graph = indexGraph(await resp.json());
      } catch {
        loading = null;
        renderState(stageEl, {
          icon: 'triangle-alert',
          message: 'The graph did not load',
          action: { label: 'Retry', onClick: () => void show(location()) },
        });
        return null;
      }
      if (!graph.nodes.length) {
        renderState(stageEl, {
          icon: 'search',
          message: 'The facility graph is empty',
          detail: 'The facility file names no place, device or channel.',
        });
        return graph;
      }
      relations = new Set(graph.relationCounts.keys());
      stage.build(relations);
      buildLegend(graph);
      if (chipEl) {
        const label = chipEl.querySelector('.osprey-chip-label') ?? chipEl;
        label.textContent = `${count(graph.nodes.length)} nodes · built from the facility file`;
      }
      return graph;
    })();
    return loading;
  }

  // -- drawing ---------------------------------------------------------------

  function renderStage(refit = false) {
    stage.render(
      { drawn, positions: positions[mode], grouped: mode === 'overview', focus, selected },
      { refit }
    );
  }

  function showInspector() {
    if (!graph) return;
    renderInspector(
      inspector,
      { graph, drawn, grouped: mode === 'overview', focus, selected },
      {
        onFocus: focusOn,
        onPage,
        onExpand(i) {
          expanded.add(i);
          redraw({ refit: true });
        },
      }
    );
  }

  /**
   * Cut the graph for the current state, place it and draw it.
   * @param {{fresh?: boolean, refit?: boolean}} [opts] `fresh` forgets where
   *   the nodes were; `refit` fits the result to the stage
   */
  function redraw({ fresh = false, refit = fresh } = {}) {
    if (!graph || !stage.built()) return;
    drawn =
      mode === 'hood' && focus !== null
        ? neighbourhood(graph, { focus, hops, relations, expanded })
        : overview(graph, relations);
    layout(drawn, positions[mode], {
      spacing: SPACING[mode],
      gravity: GRAVITY[mode],
      pin: mode === 'hood' ? focus : null,
      fresh,
    });
    renderStage(refit);
    showChrome(graph);
    showInspector();
  }

  /**
   * The toolbar's context, the depth control and the stage's status line.
   * @param {GraphIndex} index
   */
  function showChrome(index) {
    const name = mode === 'hood' && focus !== null ? index.nodes[focus].label : null;
    if (contextEl) {
      contextEl.textContent = name === null ? 'Overview' : 'Neighbourhood of ' + name;
      contextEl.title = contextEl.textContent;
    }
    depthEl.hidden = mode !== 'hood';
    if (name === null) {
      stage.setStatus(
        `${count(index.nodes.length)} nodes and ${count(index.edges.length)} relations, ` +
          'grouped by type'
      );
      return;
    }
    const more = drawn.hidden ? ` · ${count(drawn.hidden)} more not shown` : '';
    stage.setStatus(
      `${count(drawn.nodes.length)} nodes · ${count(drawn.edges.length)} relations around ` +
        `${name} (${hops} ${hops === 1 ? 'hop' : 'hops'})${more}`
    );
  }

  // -- state changes ---------------------------------------------------------

  /** @param {NodeKey | null} key */
  function select(key) {
    if (key === selected) return;
    selected = key;
    renderStage();
    showInspector();
  }

  /**
   * Draw a node's neighbourhood.
   * @param {number} i
   */
  function focusOn(i) {
    mode = 'hood';
    focus = i;
    selected = i;
    expanded.clear();
    positions.hood.clear();
    pressSeg(modesEl, 'data-mode', mode);
    redraw({ fresh: true });
    if (graph) onLocation(graph.nodes[i].id);
  }

  function showOverview() {
    mode = 'overview';
    selected = null;
    pressSeg(modesEl, 'data-mode', mode);
    redraw({ refit: true });
    onLocation(null);
  }

  /** The node a neighbourhood starts on when none was picked: the first device. */
  function firstFocus() {
    if (!graph) return 0;
    if (typeof selected === 'number') return selected;
    const device = graph.nodes.findIndex((node) => node.kind === 'device');
    return device < 0 ? 0 : device;
  }

  /**
   * Draw these relations only; the legend's checkboxes follow.
   * @param {Set<string>} next
   */
  function setRelations(next) {
    relations = new Set(next);
    for (const [relation, box] of relationBoxes) box.checked = relations.has(relation);
    redraw();
  }

  /** @param {number} next */
  function setHops(next) {
    hops = next;
    pressSeg(hopsEl, 'data-hops', String(hops));
  }

  /** Every relation, the default depth, labels on, nothing picked, laid out afresh. */
  function reset() {
    if (!graph) return;
    relations = new Set(graph.relationCounts.keys());
    for (const box of relationBoxes.values()) box.checked = true;
    setHops(DEFAULT_HOPS);
    expanded.clear();
    selected = null;
    stage.setLabels(true);
    redraw({ fresh: true });
  }

  /**
   * Copy text and say so in the status line for a moment.
   * @param {string} text
   * @param {string} what
   */
  async function copy(text, what) {
    const ok = await copyText(text);
    stage.setStatus(ok ? `Copied ${what}` : 'Copy failed');
    if (copiedTimer !== null) clearTimeout(copiedTimer);
    copiedTimer = setTimeout(() => {
      copiedTimer = null;
      if (graph) showChrome(graph);
    }, COPIED_FOR);
  }

  // -- the right-click menus -------------------------------------------------

  /**
   * @param {import('./graph-stage.js').MenuTarget} target
   * @param {Point} at
   */
  function openMenu(target, at) {
    if (!graph) return;
    const view = {
      graph,
      drawn,
      grouped: mode === 'overview',
      focus,
      relations,
      labels: stage.labels(),
    };
    const actions = {
      select,
      focus: focusOn,
      expand(/** @type {number} */ i) {
        expanded.add(i);
        redraw({ refit: true });
      },
      page: onPage,
      copy: (/** @type {string} */ text, /** @type {string} */ what) => void copy(text, what),
      setRelations,
      fit: () => stage.fit(),
      relayout: () => redraw({ fresh: true }),
      setLabels: (/** @type {boolean} */ on) => stage.setLabels(on),
      reset,
    };
    /** @type {[string, import('/design-system/js/context-menu.js').MenuItem[]]} */
    const [ariaLabel, items] =
      target.kind === 'node'
        ? ['Node', nodeMenu(view, target.key, actions)]
        : target.kind === 'edge'
          ? ['Relation', edgeMenu(view, target.edge, actions)]
          : ['Graph', stageMenu(view, actions)];
    if (!items.length) return;
    openContextMenu({ x: at.x, y: at.y, anchorEl: stageEl, ariaLabel, items });
  }

  modesEl.addEventListener('click', (ev) => {
    const button = ev.target instanceof Element ? ev.target.closest('button') : null;
    const next = button?.getAttribute('data-mode');
    if (!graph || !next || next === mode) return;
    if (next === 'hood') focusOn(focus ?? firstFocus());
    else showOverview();
  });

  hopsEl.addEventListener('click', (ev) => {
    const button = ev.target instanceof Element ? ev.target.closest('button') : null;
    const next = Number(button?.getAttribute('data-hops'));
    if (!next || next === hops) return;
    setHops(next);
    redraw({ refit: true });
  });

  // -- the sidebar's legend --------------------------------------------------

  /** @param {GraphIndex} index */
  function buildLegend(index) {
    relationsEl.replaceChildren(
      ...[...index.relationCounts].map(([relation, n]) => {
        const box = /** @type {HTMLInputElement} */ (el('input', { type: 'checkbox' }));
        box.checked = true;
        relationBoxes.set(relation, box);
        box.addEventListener('change', () => {
          if (box.checked) relations.add(relation);
          else relations.delete(relation);
          redraw();
        });
        const line = svg('path', {
          class: 'okf-graph-edge',
          'data-relation': relationStyle(relation),
          d: 'M1 4L21 4',
        });
        return el('label', { class: 'okf-graph-legend-row' }, [
          box,
          swatch({ viewBox: '0 0 22 8' }, line),
          el('span', { class: 'okf-graph-legend-name', text: relation }),
          el('span', { class: 'okf-graph-legend-count', text: count(n) }),
        ]);
      })
    );
    kindsEl.replaceChildren(
      ...NODE_KINDS.filter((kind) => index.kindCounts.has(kind)).map((kind) =>
        el('div', { class: 'okf-graph-legend-row' }, [
          swatch({ viewBox: '-8 -8 16 16', 'data-kind': kind }, shape(kind, 5)),
          el('span', { class: 'okf-graph-legend-name', text: KIND_NAMES[kind] }),
          el('span', {
            class: 'okf-graph-legend-count',
            text: count(index.kindCounts.get(kind) || 0),
          }),
        ])
      )
    );
  }

  // -- the API ---------------------------------------------------------------

  /** @returns {string | null} */
  function location() {
    return graph && mode === 'hood' && focus !== null ? graph.nodes[focus].id : null;
  }

  /** @param {string | null} nodeId */
  async function show(nodeId) {
    const index = await load();
    if (!index || !stage.built()) return;
    const i = nodeId === null ? undefined : index.byId.get(nodeId);
    if (i !== undefined) {
      if (mode !== 'hood' || focus !== i) focusOn(i);
      else redraw();
    } else if (mode !== 'overview' || !drawn.nodes.length) {
      showOverview();
    } else {
      redraw();
    }
  }

  return {
    show,
    location,
    async showRecord(kind, id) {
      const index = await load();
      if (!index || !stage.built()) return false;
      const i = nodeOfRecord(index, kind, id);
      if (i < 0) {
        await show(location());
        return false;
      }
      focusOn(i);
      return true;
    },
    async search(query) {
      const q = query.trim();
      if (!q) {
        hitsEl.hidden = true;
        hitsEl.replaceChildren();
        return;
      }
      const index = await load();
      if (!index) return;
      const hits = searchNodes(index, q);
      hitsEl.replaceChildren(
        el('div', { class: 'osprey-section-label', text: 'Search' }),
        ...(hits.length
          ? hits.map((i) => nodeRow(index, i, focusOn))
          : [el('p', { class: 'okf-graph-more', text: 'No node matches' })])
      );
      hitsEl.hidden = false;
      sidebar.open();
    },
    resetWidth() {
      sidebar.resetWidth();
    },
  };
}
