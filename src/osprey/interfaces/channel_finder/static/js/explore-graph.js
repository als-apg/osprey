// @ts-check
/**
 * OSPREY Channel Finder — Graph Explore (the facility ontology view).
 *
 * The graph paradigm has no channel database to browse, so Explore shows what
 * the store actually holds: the device-class taxonomy, each class carrying the
 * number of devices rolled up under it. The panel also names its provenance —
 * the store the numbers came from and the tools the assistant queries it with —
 * so an operator can tell a stale corpus from a live one at a glance.
 *
 * Three states, one panel: a drawn tree, an informational pane (the store is
 * reachable but unseeded), and the same informational pane carrying the store's
 * own remedy (the read failed). Only the third is a fault, and none of them is
 * styled as an error, because a correctly configured deployment that has simply
 * not been seeded yet must not read as a broken one.
 *
 * The endpoint is read with `fetch` rather than `api.js`'s `fetchJSON`: a 503
 * from the graph routes carries `error_type` and `suggestions` beside `detail`,
 * and those suggestions ARE the operator's remedy. `fetchJSON` keeps only
 * `detail`, and `api.js` is shared by every other view, so the body is parsed
 * here instead of widening the shared helper for one caller.
 *
 * Layout is delegated to graph-tree-layout.js (pure, unit-tested); this module
 * only measures text, builds SVG, and owns the fetch lifecycle.
 */

import { state } from './state.js';
import { esc, messageOf } from './utils.js';
import { refreshStatsBadges } from './stats-badges.js';
import { layoutForest } from './graph-tree-layout.js';

/** @typedef {import('./graph-tree-layout.js').LayoutNode} LayoutNode */
/** @typedef {(text: string) => number} TextMeasurer */

/**
 * What one ontology request produced: the payload, a rendered failure, or
 * `null` when the request was abandoned because the view went away.
 * @typedef {{ok: true, data: any} | {ok: false, detail: string, suggestions: string[]} | null} OntologyResult
 */

const ONTOLOGY_PATH = '/api/graph/ontology';
const SVG_NS = 'http://www.w3.org/2000/svg';

/**
 * Legend tints, cycled in order so each relationship type in the store's
 * vocabulary is visually distinct. Token names only — the CSS reads them
 * through an inline `--c` custom property with its own fallback, so this file
 * never names a colour value.
 */
const LEGEND_TOKENS = [
  '--color-accent',
  '--color-accent-secondary',
  '--color-success',
  '--ansi-magenta',
  '--ansi-cyan',
  '--color-warning',
];

const ROW_HEIGHT = 32;
const NODE_HEIGHT = 24;
const NODE_PAD_X = 10;
const NAME_GAP = 10;
/** Free space between the widest node in a column and the next column. */
const COLUMN_GAP = 56;
const SVG_PAD = 8;
const PILL_HEIGHT = 14;
const PILL_PAD_X = 7;
const PILL_MIN_WIDTH = 22;
/** Width per character when no canvas measurer is available (tests, headless). */
const FALLBACK_CHAR_WIDTH = 7;
const MEASURE_FONT = '12px monospace';

const LOADING_HTML =
  '<div class="loading-center"><div class="loading-spinner"></div> Loading the facility ontology&hellip;</div>';
const EMPTY_TITLE = 'The graph store is reachable, but holds no facility corpus yet.';
const PANEL_SUBTITLE =
  'Device classes in the graph store, each showing the number of devices under it and its subclasses.';

/** The element the panel is mounted into, or null when nothing is mounted. */
/** @type {HTMLElement|null} */
let paneEl = null;

/** Controller for the in-flight ontology request, if any. */
/** @type {AbortController|null} */
let controller = null;

/**
 * Bumped by every load and by unmount, so a reply that arrives after the view
 * moved on is discarded instead of overwriting whatever is mounted now.
 */
let generation = 0;

// ---------------------------------------------------------------------------
// Mount / unmount
// ---------------------------------------------------------------------------

/**
 * Render the graph explorer into `content` and load the ontology.
 *
 * @param {HTMLElement} content - The pane the explore dispatcher owns.
 * @returns {Promise<void>} Resolves once the first render has settled.
 */
export async function mountGraph(content) {
  paneEl = content;
  content.innerHTML = shellHtml();
  await loadOntology();
}

/**
 * Tear the panel down: abandon any in-flight request and clear the pane.
 * Safe to call when nothing is mounted.
 * @returns {void}
 */
export function unmountGraph() {
  abortInFlight();
  generation += 1;
  if (paneEl) paneEl.innerHTML = '';
  paneEl = null;
}

/** Abort the in-flight ontology request, if there is one. */
function abortInFlight() {
  if (controller) {
    controller.abort();
    controller = null;
  }
}

// ---------------------------------------------------------------------------
// Panel shell
// ---------------------------------------------------------------------------

/**
 * The panel chrome: title, provenance badge, subtitle with the tool chips, and
 * the body every later render replaces.
 * @returns {string} Markup for the panel.
 */
function shellHtml() {
  return `
    <div class="graph-panel" data-pipeline="graph">
      <div class="graph-panel-head">
        <div class="graph-panel-title">Facility ontology</div>
        ${storeBadgeHtml()}
        <div class="graph-panel-sub">${esc(PANEL_SUBTITLE)}${toolChipsHtml()}</div>
      </div>
      <div class="graph-panel-body">${LOADING_HTML}</div>
    </div>
  `;
}

/**
 * Name the store behind these numbers. Both halves are optional: a store seeded
 * from a TTL file is named `file @ uri`, one seeded another way by its URI
 * alone. A missing half is left out rather than printed as an empty word.
 * @returns {string} Markup for the badge, or '' when nothing is known.
 */
function storeBadgeHtml() {
  const store = state.graphStore;
  const uri = store && store.uri ? store.uri : '';
  const file = store && store.ttl_filename ? store.ttl_filename : '';
  const label = file && uri ? `${file} @ ${uri}` : (file || uri);
  if (!label) return '';
  return `<div class="graph-store-badge" title="Graph store"><code>${esc(label)}</code></div>`;
}

/**
 * The tools the assistant reads this same store with, so the reader knows the
 * panel and the agent are looking at one corpus.
 * @returns {string} Markup for the chips, or '' when none are reported.
 */
function toolChipsHtml() {
  const tools = state.tools || [];
  if (tools.length === 0) return '';
  const chips = tools.map((tool) => `<span class="graph-tool-chip">${esc(tool)}</span>`).join(' ');
  return ` The assistant queries it with ${chips}`;
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

/**
 * Fetch the ontology and render whichever of the three states it implies.
 * @returns {Promise<void>} Resolves once the body has been rendered.
 */
async function loadOntology() {
  const pane = paneEl;
  if (!pane) return;
  const body = pane.querySelector('.graph-panel-body');
  if (!body) return;

  body.innerHTML = LOADING_HTML;
  abortInFlight();
  generation += 1;
  const mine = generation;
  controller = new AbortController();

  const result = await requestOntology(controller.signal);

  // A newer load, or an unmount, happened while this one was in flight.
  if (mine !== generation || paneEl !== pane) return;
  controller = null;
  if (result === null) return;
  if (!result.ok) {
    renderInfo(body, result.detail, result.suggestions);
    return;
  }
  renderPayload(body, result.data);
}

/**
 * Ask the ontology endpoint, keeping the remedy fields a failure carries.
 *
 * @param {AbortSignal} signal - Abort signal for the request.
 * @returns {Promise<OntologyResult>} The payload, a failure, or null if aborted.
 */
async function requestOntology(signal) {
  try {
    const resp = await fetch(ONTOLOGY_PATH, { signal });
    const body = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      return {
        ok: false,
        detail: String(body.detail || resp.statusText || 'The graph store did not answer.'),
        suggestions: stringList(body.suggestions),
      };
    }
    return { ok: true, data: body };
  } catch (e) {
    if (e instanceof Error && e.name === 'AbortError') return null;
    return { ok: false, detail: messageOf(e), suggestions: [] };
  }
}

/**
 * Coerce an untyped payload field into a list of display strings.
 * @param {unknown} value - The field as the server sent it.
 * @returns {string[]} The strings it held, or an empty list.
 */
function stringList(value) {
  return Array.isArray(value) ? value.map((item) => String(item)) : [];
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/**
 * Draw a successful payload: the tree, or the seed pane when it holds nothing.
 *
 * @param {Element} body - The panel body to render into.
 * @param {any} data - The ontology payload.
 * @returns {void}
 */
function renderPayload(body, data) {
  const classes = Array.isArray(data && data.classes) ? data.classes : [];
  if (classes.length === 0) {
    renderInfo(body, EMPTY_TITLE, stringList(data && data.suggestions));
    return;
  }

  body.innerHTML = '';
  const types = stringList(data.relationship_types);
  if (types.length > 0) body.appendChild(buildLegend(types));
  if (data.truncated === true) body.appendChild(buildTruncated(classes.length));
  body.appendChild(buildSvg(classes));
}

/**
 * Render the informational pane: a headline and one line per remedy, with a
 * Retry that re-asks both the ontology and the header statistics — the two
 * reads that fail together when the store is down, so they recover together.
 *
 * @param {Element} body - The panel body to render into.
 * @param {string} detail - What happened, in the store's own words.
 * @param {string[]} suggestions - One remedy per line.
 * @returns {void}
 */
function renderInfo(body, detail, suggestions) {
  body.innerHTML = '';
  const pane = document.createElement('div');
  pane.className = 'explore-unknown explore-unknown--info';

  const title = document.createElement('div');
  title.className = 'explore-unknown-title';
  title.textContent = detail;

  const text = document.createElement('div');
  text.className = 'explore-unknown-body';
  for (const suggestion of suggestions) {
    const line = document.createElement('div');
    line.textContent = suggestion;
    text.appendChild(line);
  }

  const retry = document.createElement('button');
  retry.className = 'btn btn-secondary btn-sm';
  retry.id = 'graph-retry';
  retry.type = 'button';
  retry.textContent = 'Retry';
  retry.addEventListener('click', () => {
    void refreshStatsBadges();
    void loadOntology();
  });
  text.appendChild(retry);

  pane.append(title, text);
  body.appendChild(pane);
}

/**
 * One tinted chip per relationship type in the store's vocabulary.
 *
 * @param {string[]} types - Relationship type names.
 * @returns {HTMLElement} The legend row.
 */
function buildLegend(types) {
  const legend = document.createElement('div');
  legend.className = 'graph-legend';
  types.forEach((type, index) => {
    const chip = document.createElement('span');
    chip.className = 'graph-legend-chip';
    chip.style.setProperty('--c', `var(${LEGEND_TOKENS[index % LEGEND_TOKENS.length]})`);
    chip.textContent = type;
    legend.appendChild(chip);
  });
  return legend;
}

/**
 * The caution shown when the store held more classes than the query returned.
 *
 * @param {number} shown - How many classes the tree below draws.
 * @returns {HTMLElement} The caution strip.
 */
function buildTruncated(shown) {
  const strip = document.createElement('div');
  strip.className = 'graph-truncated';
  strip.textContent =
    `Showing ${shown.toLocaleString()} classes: the store holds more than this query returned, `
    + 'so the tree below is a partial view of the ontology.';
  return strip;
}

// ---------------------------------------------------------------------------
// SVG tree
// ---------------------------------------------------------------------------

/**
 * Lay the classes out and draw them: edges first, so the boxes sit on top.
 *
 * The column width is derived from the widest measured node rather than fixed,
 * so a corpus with long class names cannot overlap its own columns.
 *
 * @param {any[]} classes - Ontology classes from the endpoint.
 * @returns {SVGElement} The tree.
 */
function buildSvg(classes) {
  const measureText = makeTextMeasurer();
  /** @type {(name: string, rollup: number) => number} */
  const measure = (name, rollup) =>
    NODE_PAD_X * 2 + measureText(name) + NAME_GAP + pillWidth(rollup, measureText);

  let widest = 0;
  for (const cls of classes) {
    widest = Math.max(widest, measure(displayName(cls), Number(cls && cls.rollup) || 0));
  }
  const layout = layoutForest(classes, {
    rowHeight: ROW_HEIGHT,
    colWidth: Math.ceil(widest) + COLUMN_GAP,
    nodeHeight: NODE_HEIGHT,
    measure,
  });

  const width = layout.width + SVG_PAD * 2;
  const height = layout.height + SVG_PAD * 2;
  const svg = svgEl('svg', {
    class: 'graph-tree',
    width,
    height,
    viewBox: `0 0 ${width} ${height}`,
    role: 'img',
    'aria-label': `Device class tree: ${layout.nodes.length} classes`,
  });

  const root = svgEl('g', { transform: `translate(${SVG_PAD}, ${SVG_PAD})` });
  for (const edge of layout.edges) {
    root.appendChild(svgEl('path', { class: 'g-edge subclassof', d: edge.d }));
  }
  for (const node of layout.nodes) root.appendChild(buildNode(node, measureText));
  svg.appendChild(root);
  return svg;
}

/**
 * One class: its box, its name, its rolled-up device count, and a tooltip
 * carrying what the box has no room for.
 *
 * The count pill is nested one group deeper than the box so the box rule
 * (`.g-node > rect`) cannot claim it — the stylesheet addresses the pill by
 * class alone.
 *
 * @param {LayoutNode} node - A placed layout node.
 * @param {TextMeasurer} measureText - Text measurer, for the pill width.
 * @returns {SVGElement} The node group.
 */
function buildNode(node, measureText) {
  const group = svgEl('g', { class: node.isRoot ? 'g-node root' : 'g-node' });
  const middle = node.y + node.height / 2;

  group.appendChild(svgEl('rect', {
    x: node.x, y: node.y, width: node.width, height: node.height,
  }));

  const name = svgEl('text', { class: 'g-node-name', x: node.x + NODE_PAD_X, y: middle });
  name.textContent = node.name;
  group.appendChild(name);

  const label = node.rollup.toLocaleString();
  const width = pillWidth(node.rollup, measureText);
  const left = node.x + node.width - NODE_PAD_X - width;
  const pill = svgEl('g', { class: 'g-node-count-wrap' });
  pill.appendChild(svgEl('rect', {
    class: 'g-node-count-pill',
    x: left, y: middle - PILL_HEIGHT / 2, width, height: PILL_HEIGHT,
  }));
  const count = svgEl('text', { class: 'g-node-count', x: left + width / 2, y: middle });
  count.textContent = label;
  pill.appendChild(count);
  group.appendChild(pill);

  const tooltip = svgEl('title');
  tooltip.textContent = tooltipFor(node);
  group.appendChild(tooltip);
  return group;
}

/**
 * The node's tooltip: its URI, the labels the corpus also knows it by, and the
 * parents it is NOT drawn under — the one thing a tree cannot show about a
 * class with several parents.
 *
 * @param {LayoutNode} node - A placed layout node.
 * @returns {string} Newline-separated tooltip text.
 */
function tooltipFor(node) {
  const lines = [node.name, node.uri];
  if (node.altLabel.length > 0) lines.push(`also known as: ${node.altLabel.join(', ')}`);
  if (node.extraParents.length > 0) {
    lines.push(`also under: ${node.extraParents.map(shortName).join(', ')}`);
  }
  return lines.join('\n');
}

/**
 * Create an SVG element with attributes.
 *
 * @param {string} name - Tag name.
 * @param {Record<string, string|number>} [attrs] - Attributes to set.
 * @returns {SVGElement} The element.
 */
function svgEl(name, attrs = {}) {
  const node = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
  return node;
}

// ---------------------------------------------------------------------------
// Measurement
// ---------------------------------------------------------------------------

/**
 * Build a text measurer. A canvas gives real glyph widths in a browser; where
 * there is none, or where it reports nothing (headless DOMs return zero-width
 * text), fall back to a character-count estimate so the layout stays defined
 * and deterministic rather than collapsing to zero-width boxes.
 *
 * @returns {TextMeasurer} Width of a string, in px.
 */
function makeTextMeasurer() {
  /** @type {CanvasRenderingContext2D|null} */
  let ctx = null;
  try {
    const canvas = /** @type {HTMLCanvasElement} */ (document.createElement('canvas'));
    ctx = typeof canvas.getContext === 'function' ? canvas.getContext('2d') : null;
    if (ctx) {
      ctx.font = MEASURE_FONT;
      if (!(ctx.measureText('M').width > 0)) ctx = null;
    }
  } catch {
    ctx = null;
  }
  return (text) => {
    if (ctx) {
      const width = ctx.measureText(text).width;
      if (Number.isFinite(width) && width > 0) return width;
    }
    return text.length * FALLBACK_CHAR_WIDTH;
  };
}

/**
 * Width of the count pill, wide enough for its digits and never narrower than
 * a readable minimum.
 *
 * @param {number} rollup - The device count.
 * @param {TextMeasurer} measureText - Text measurer.
 * @returns {number} Pill width, px.
 */
function pillWidth(rollup, measureText) {
  return Math.max(PILL_MIN_WIDTH, measureText(rollup.toLocaleString()) + PILL_PAD_X * 2);
}

/**
 * The name the layout will use for a class, mirrored here so the pre-pass that
 * sizes the columns measures exactly what the renderer later draws.
 *
 * @param {any} cls - A raw ontology class.
 * @returns {string} Its display name.
 */
function displayName(cls) {
  if (cls && typeof cls.name === 'string' && cls.name !== '') return cls.name;
  return cls && typeof cls.uri === 'string' ? cls.uri : '';
}

/**
 * Shorten a class URI to its trailing fragment for display.
 *
 * @param {string} uri - A class URI.
 * @returns {string} The text after the last '/' or '#'.
 */
function shortName(uri) {
  const cut = Math.max(uri.lastIndexOf('/'), uri.lastIndexOf('#'));
  return cut >= 0 ? uri.slice(cut + 1) : uri;
}
