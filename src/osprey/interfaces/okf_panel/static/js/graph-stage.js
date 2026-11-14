// @ts-check
/*
 * OKF Knowledge Panel — the Graph view's stage: one cut of the facility
 * graph, drawn as SVG.
 *
 * The stage draws the scene it is given (graph-model.js's drawn nodes and
 * edges, and where each node is) and owns everything about looking at it:
 * the view transform (pan, wheel zoom, fit), dragging a node, dimming what
 * is not joined to the node pointed at, and the content strip (relation
 * labels, zoom, fit, re-layout). What a click, a double-click or a right-click
 * means is the caller's: the stage only reports them, a right-click with what
 * it landed on (a node, a relation, or the empty stage).
 *
 * Every colour and line style is the stylesheet's, keyed by `data-kind` and
 * `data-relation`; nothing here sets a style.
 */

import { createContentStrip } from '/design-system/js/content-strip.js';
import { declinesToBrowser } from '/design-system/js/context-menu.js';
import { el } from './helpers.js';
import { count, fitTransform, relationStyle, zoomAt } from './graph-model.js';

/** @typedef {import('./graph-model.js').DrawnGraph} DrawnGraph */
/** @typedef {import('./graph-model.js').DrawnNode} DrawnNode */
/** @typedef {import('./graph-model.js').DrawnEdge} DrawnEdge */
/** @typedef {import('./graph-model.js').NodeKey} NodeKey */
/** @typedef {import('./graph-model.js').NodeKind} NodeKind */
/** @typedef {import('./graph-model.js').Point} Point */

/**
 * @typedef {object} Scene what the stage draws
 * @property {DrawnGraph} drawn
 * @property {Map<NodeKey, Point>} positions where each drawn node is; a
 *   dragged node's point is moved in place
 * @property {boolean} grouped the scene is the overview, not a neighbourhood
 * @property {NodeKey | null} focus the node a neighbourhood is drawn around
 * @property {NodeKey | null} selected the picked node
 */

/**
 * @typedef {object} StageHandles
 * @property {(key: NodeKey | null) => void} onSelect a node was clicked, or
 *   the background (null)
 * @property {(key: NodeKey) => void} onFocus a node was double-clicked
 * @property {() => void} onRelayout "Lay out again" was pressed
 * @property {(target: MenuTarget, at: Point) => void} [onMenu] a right-click,
 *   with what it landed on and where in the viewport
 */

/**
 * @typedef {{kind: 'node', key: NodeKey} | {kind: 'edge', edge: DrawnEdge} | {kind: 'stage'}} MenuTarget
 */

/**
 * @typedef {object} GraphStage
 * @property {(relations: Iterable<string>) => void} build Replace the host's
 *   content with the empty drawing; `relations` are the ones an edge may show.
 * @property {() => boolean} built
 * @property {(scene: Scene, opts?: {refit?: boolean}) => void} render Draw a
 *   scene; `refit` fits it to the stage.
 * @property {(text: string) => void} setStatus
 * @property {() => void} fit Fit the drawing to the stage.
 * @property {() => boolean} labels whether relation labels are drawn
 * @property {(on: boolean) => void} setLabels
 */

const SVG_NS = 'http://www.w3.org/2000/svg';
/** Channel labels are dropped from a neighbourhood with more nodes than this. */
const CHANNEL_LABEL_LIMIT = 70;
/** Relation labels are dropped from a neighbourhood with more edges than this. */
const EDGE_LABEL_LIMIT = 90;
/** A revealed node keeps this far from the stage's edge. */
const REVEAL_MARGIN = 80;
const HINT = 'Click: details · Double-click: focus · Right-click: menu · Drag: move · Wheel: zoom';

/**
 * @param {string} tag
 * @param {Record<string, string | number>} [attrs]
 * @param {(Node | null)[]} [children]
 * @returns {SVGElement}
 */
export function svg(tag, attrs, children) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const key in attrs) node.setAttribute(key, String(attrs[key]));
  for (const child of children || []) if (child) node.appendChild(child);
  return node;
}

/**
 * A node's mark: a square for a place, a diamond for a signal, a circle
 * otherwise (a class's is drawn as a ring by the stylesheet).
 * @param {NodeKind} kind
 * @param {number} r
 * @returns {SVGElement}
 */
export function shape(kind, r) {
  if (kind === 'place') {
    return svg('rect', {
      class: 'okf-graph-shape',
      x: -r,
      y: -r,
      width: 2 * r,
      height: 2 * r,
      rx: 2,
    });
  }
  if (kind === 'signal') {
    const d = r * 1.25;
    return svg('path', { class: 'okf-graph-shape', d: `M0 ${-d}L${d} 0L0 ${d}L${-d} 0Z` });
  }
  return svg('circle', { class: 'okf-graph-shape', r });
}

/**
 * The point `distance` from `from` on the way to `to`.
 * @param {Point} from
 * @param {Point} to
 * @param {number} distance
 * @returns {Point}
 */
function toward(from, to, distance) {
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const length = Math.hypot(dx, dy) || 1;
  return { x: from.x + (dx / length) * distance, y: from.y + (dy / length) * distance };
}

/**
 * @param {EventTarget | null} target
 * @returns {NodeKey | null} the key of the node an event landed on
 */
function keyAt(target) {
  const g = target instanceof Element ? target.closest('.okf-graph-node') : null;
  const raw = g ? g.getAttribute('data-key') : null;
  if (raw === null) return null;
  return /^\d+$/.test(raw) ? Number(raw) : raw;
}

/**
 * @param {HTMLElement} host
 * @param {StageHandles} handles
 * @returns {GraphStage}
 */
export function createGraphStage(host, { onSelect, onFocus, onRelayout, onMenu }) {
  /** @type {Scene} */
  let scene = {
    drawn: { nodes: [], edges: [], hidden: 0 },
    positions: new Map(),
    grouped: true,
    focus: null,
    selected: null,
  };
  let labels = true;
  let tf = { x: 0, y: 0, k: 1 };
  /** The drawing has not been fitted to a laid-out stage yet. */
  let fitOwed = false;

  /** @type {SVGElement | null} */
  let surface = null;
  /** @type {SVGElement | null} */
  let viewport = null;
  /** @type {SVGElement | null} */
  let edgeLayer = null;
  /** @type {SVGElement | null} */
  let nodeLayer = null;
  /** @type {HTMLElement | null} */
  let statusEl = null;
  /** @type {Map<NodeKey, {g: SVGElement, node: DrawnNode}>} */
  const nodeEls = new Map();
  /**
   * @type {{path: SVGElement, hit: SVGElement, label: SVGElement | null, edge: DrawnEdge,
   *   bend: number}[]}
   */
  let edgeEls = [];
  /** @type {{setPressed: (id: string, pressed: boolean) => void} | null} */
  let strip = null;

  /** @param {Iterable<string>} relations */
  function build(relations) {
    const styles = new Set(['other', ...[...relations].map(relationStyle)]);
    const markers = [...styles].map((style) =>
      svg(
        'marker',
        {
          id: 'okf-graph-arrow-' + style,
          viewBox: '0 0 8 8',
          refX: 8,
          refY: 4,
          // In drawing units, so a heavy line's arrowhead is no larger.
          markerUnits: 'userSpaceOnUse',
          markerWidth: 9,
          markerHeight: 9,
          orient: 'auto-start-reverse',
        },
        [svg('path', { class: 'okf-graph-arrow', 'data-relation': style, d: 'M0 0L8 4L0 8Z' })]
      )
    );
    edgeLayer = svg('g');
    nodeLayer = svg('g');
    viewport = svg('g', {}, [edgeLayer, nodeLayer]);
    surface = svg('svg', { class: 'okf-graph-svg', role: 'img', 'aria-label': 'Facility graph' }, [
      svg('defs', {}, markers),
      viewport,
    ]);
    // The stage pans and drags on pointer events, so a touch must not scroll the page.
    surface.style.touchAction = 'none';
    statusEl = el('div', { class: 'okf-graph-status', role: 'status' });
    host.replaceChildren(surface, statusEl, el('div', { class: 'okf-graph-hint', text: HINT }));

    strip = createContentStrip(host, {
      label: 'Graph controls',
      groups: [
        {
          id: 'modes',
          items: [{ id: 'labels', icon: 'types', title: 'Relation labels', mode: true }],
        },
        {
          id: 'view',
          items: [
            { id: 'zoom-in', icon: 'zoom-in', title: 'Zoom in' },
            { id: 'zoom-out', icon: 'zoom-out', title: 'Zoom out' },
            { id: 'fit', icon: 'autoscale', title: 'Fit to view' },
          ],
        },
        { id: 'layout', items: [{ id: 'relayout', icon: 'refresh', title: 'Lay out again' }] },
      ],
      onAction(id, pressed) {
        if (id === 'labels') {
          setLabels(Boolean(pressed));
        } else if (id === 'zoom-in' || id === 'zoom-out') {
          const factor = id === 'zoom-in' ? 1.25 : 0.8;
          setTransform(zoomAt(tf, factor, host.clientWidth / 2, host.clientHeight / 2));
        } else if (id === 'fit') {
          fit();
        } else if (id === 'relayout') {
          onRelayout();
        }
      },
    });
    strip.setPressed('labels', true);
    wirePointer(surface);
    if (typeof ResizeObserver !== 'undefined') {
      new ResizeObserver(() => {
        if (fitOwed) fit();
        else reveal();
      }).observe(host);
    }
  }

  /** @param {boolean} on */
  function setLabels(on) {
    labels = on;
    strip?.setPressed('labels', on);
    draw();
  }

  /** @param {{x: number, y: number, k: number}} next */
  function setTransform(next) {
    tf = next;
    viewport?.setAttribute('transform', `translate(${tf.x} ${tf.y}) scale(${tf.k})`);
  }

  /**
   * Pan the picked node back onto the stage: the inspector that opens for it
   * takes the stage's right edge.
   */
  function reveal() {
    const at = scene.selected === null ? undefined : scene.positions.get(scene.selected);
    if (!at || !host.clientWidth) return;
    const over = at.x * tf.k + tf.x - (host.clientWidth - REVEAL_MARGIN);
    if (over > 0) setTransform({ ...tf, x: tf.x - over });
  }

  /** Fit the drawing to the stage; owed again while the stage has no size. */
  function fit() {
    const points = scene.drawn.nodes.flatMap((node) => scene.positions.get(node.key) ?? []);
    // A small drawing is not blown up to fill the stage: its labels scale too.
    const next = fitTransform(points, host.clientWidth, host.clientHeight, { maxScale: 1.15 });
    fitOwed = next === null;
    if (next) setTransform(next);
  }

  /** Build the drawing's elements from the scene. */
  function draw() {
    if (!edgeLayer || !nodeLayer) return;
    const { drawn, grouped, focus, selected } = scene;
    nodeEls.clear();

    // Edges that join the same two nodes bend apart, so each can be read.
    /** @type {Map<string, number>} */
    const shared = new Map();
    const pairOf = (/** @type {DrawnEdge} */ edge) =>
      String(edge.a) < String(edge.b) ? `${edge.a}|${edge.b}` : `${edge.b}|${edge.a}`;
    for (const edge of drawn.edges) shared.set(pairOf(edge), (shared.get(pairOf(edge)) || 0) + 1);
    /** @type {Map<string, number>} */
    const seen = new Map();
    const withLabels = labels && (grouped || drawn.edges.length <= EDGE_LABEL_LIMIT);

    edgeEls = drawn.edges.map((edge, index) => {
      const pair = pairOf(edge);
      const nth = seen.get(pair) || 0;
      seen.set(pair, nth + 1);
      const total = shared.get(pair) || 1;
      // The bend is measured from the pair's sorted order, so two edges in
      // opposite directions bend to opposite sides.
      const side = String(edge.a) < String(edge.b) ? 1 : -1;
      const bend = total > 1 ? (nth - (total - 1) / 2) * 26 * side : 0;
      const style = relationStyle(edge.relation);
      const path = svg('path', {
        class: 'okf-graph-edge',
        'data-relation': style,
        'data-a': edge.a,
        'data-b': edge.b,
        'marker-end': `url(#okf-graph-arrow-${style})`,
      });
      // The stylesheet draws the line; a counted edge only says how heavy.
      if (edge.width) path.style.setProperty('--okf-edge-width', edge.width.toFixed(2));
      const label = withLabels
        ? svg('text', { class: 'okf-graph-edge-label', 'text-anchor': 'middle' })
        : null;
      if (label) label.textContent = edge.label;
      // A line is too thin to point at: an unpainted wide one takes the pointer.
      const hit = svg('path', { class: 'okf-graph-edge-hit', 'data-edge': index });
      return { path, hit, label, edge, bend };
    });
    edgeLayer.replaceChildren(
      ...edgeEls.map((entry) => entry.hit),
      ...edgeEls.map((entry) => entry.path),
      ...edgeEls.flatMap((entry) => (entry.label ? [entry.label] : []))
    );

    const quietChannels = !grouped && drawn.nodes.length > CHANNEL_LABEL_LIMIT;
    nodeLayer.replaceChildren(
      ...drawn.nodes.map((node) => {
        const title = svg('title');
        title.textContent = node.label;
        const g = svg(
          'g',
          { class: 'okf-graph-node', 'data-kind': node.kind, 'data-key': node.key },
          [
            title,
            node.key === focus && !grouped
              ? svg('circle', { class: 'okf-graph-focus', r: node.r + 5 })
              : null,
            shape(node.kind, node.r),
          ]
        );
        if (node.key === selected) g.setAttribute('data-selected', '');
        if (!(quietChannels && node.kind === 'channel') || node.key === selected) {
          const text = svg('text', { class: 'okf-graph-label', x: node.r + 4, y: 3 });
          text.textContent = node.label;
          g.appendChild(text);
        }
        if (node.badge) {
          const badge = svg('text', { class: 'okf-graph-badge', x: node.r + 4, y: -node.r });
          badge.textContent = '+' + count(node.badge);
          g.appendChild(badge);
        }
        nodeEls.set(node.key, { g, node });
        return g;
      })
    );
    place();
  }

  /** Put every element where its node is. */
  function place() {
    const at = scene.positions;
    for (const [key, { g }] of nodeEls) {
      const p = at.get(key);
      if (p) g.setAttribute('transform', `translate(${p.x.toFixed(1)} ${p.y.toFixed(1)})`);
    }
    for (const { path, hit, label, edge, bend } of edgeEls) {
      const a = at.get(edge.a);
      const b = at.get(edge.b);
      const from = nodeEls.get(edge.a);
      const to = nodeEls.get(edge.b);
      if (!a || !b || !from || !to) continue;
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const length = Math.hypot(dx, dy) || 1;
      // The control point sits off the chord's middle, on its normal.
      const cx = (a.x + b.x) / 2 - (dy / length) * bend * 2;
      const cy = (a.y + b.y) / 2 + (dx / length) * bend * 2;
      const start = toward(a, { x: cx, y: cy }, from.node.r);
      // The arrow's tip stops at the target's edge.
      const end = toward(b, { x: cx, y: cy }, to.node.r + 2);
      const d =
        `M${start.x.toFixed(1)} ${start.y.toFixed(1)}Q${cx.toFixed(1)} ${cy.toFixed(1)} ` +
        `${end.x.toFixed(1)} ${end.y.toFixed(1)}`;
      path.setAttribute('d', d);
      hit.setAttribute('d', d);
      if (label) {
        label.setAttribute('x', (0.25 * start.x + 0.5 * cx + 0.25 * end.x).toFixed(1));
        label.setAttribute('y', (0.25 * start.y + 0.5 * cy + 0.25 * end.y - 3).toFixed(1));
      }
    }
  }

  /**
   * Pick, focus, drag, pan and zoom.
   * @param {SVGElement} target
   */
  function wirePointer(target) {
    /** @type {{key: NodeKey | null, x: number, y: number, moved: boolean} | null} */
    let drag = null;
    /** A drag just ended: the click that follows it picks nothing. */
    let dragged = false;
    // The press captures the pointer, so the click and the double-click that
    // follow are delivered to the drawing, not to the node: the node is the
    // one that was pressed.
    /** @type {NodeKey | null} */
    let pressed = null;

    target.addEventListener('pointerdown', (event) => {
      const ev = /** @type {PointerEvent} */ (event);
      if (ev.button !== 0) return;
      pressed = keyAt(ev.target);
      drag = { key: pressed, x: ev.clientX, y: ev.clientY, moved: false };
      dragged = false;
      target.setPointerCapture?.(ev.pointerId);
    });
    target.addEventListener('pointermove', (event) => {
      const ev = /** @type {PointerEvent} */ (event);
      if (!drag) return;
      const dx = ev.clientX - drag.x;
      const dy = ev.clientY - drag.y;
      if (!drag.moved && Math.hypot(dx, dy) < 3) return;
      drag.moved = true;
      drag.x = ev.clientX;
      drag.y = ev.clientY;
      if (drag.key === null) {
        setTransform({ ...tf, x: tf.x + dx, y: tf.y + dy });
        return;
      }
      const p = scene.positions.get(drag.key);
      if (!p) return;
      p.x += dx / tf.k;
      p.y += dy / tf.k;
      place();
    });
    const end = () => {
      dragged = Boolean(drag && drag.moved);
      drag = null;
    };
    target.addEventListener('pointerup', end);
    target.addEventListener('pointercancel', end);

    target.addEventListener('click', (ev) => {
      if (dragged) {
        dragged = false;
        return;
      }
      onSelect(keyAt(ev.target) ?? pressed);
    });
    target.addEventListener('dblclick', (ev) => {
      const key = keyAt(ev.target) ?? pressed;
      if (key !== null) onFocus(key);
    });
    target.addEventListener(
      'wheel',
      (event) => {
        const ev = /** @type {WheelEvent} */ (event);
        ev.preventDefault();
        const box = target.getBoundingClientRect();
        setTransform(
          zoomAt(tf, ev.deltaY < 0 ? 1.12 : 0.89, ev.clientX - box.left, ev.clientY - box.top)
        );
      },
      { passive: false }
    );
    target.addEventListener('contextmenu', (event) => {
      const ev = /** @type {MouseEvent} */ (event);
      if (!onMenu || declinesToBrowser(ev)) return;
      ev.preventDefault();
      const at = { x: ev.clientX, y: ev.clientY };
      const key = keyAt(ev.target);
      const hit = ev.target instanceof Element ? ev.target.closest('.okf-graph-edge-hit') : null;
      const entry = hit ? edgeEls[Number(hit.getAttribute('data-edge'))] : undefined;
      if (key !== null) onMenu({ kind: 'node', key }, at);
      else if (entry) onMenu({ kind: 'edge', edge: entry.edge }, at);
      else onMenu({ kind: 'stage' }, at);
    });
    // Pointing at a node dims what is not joined to it.
    target.addEventListener('mouseover', (ev) => highlight(keyAt(ev.target)));
    target.addEventListener('mouseleave', () => highlight(null));
  }

  /** @param {NodeKey | null} key the node pointed at, or null for none */
  function highlight(key) {
    /** @type {Set<NodeKey> | null} */
    let near = null;
    if (key !== null) {
      near = new Set([key]);
      for (const { edge } of edgeEls) {
        if (edge.a === key) near.add(edge.b);
        if (edge.b === key) near.add(edge.a);
      }
    }
    for (const [k, { g }] of nodeEls) g.toggleAttribute('data-dim', !!near && !near.has(k));
    for (const { path, label, edge } of edgeEls) {
      const dim = key !== null && edge.a !== key && edge.b !== key;
      path.toggleAttribute('data-dim', dim);
      label?.toggleAttribute('data-dim', dim);
    }
  }

  return {
    build,
    built: () => surface !== null,
    render(next, { refit = false } = {}) {
      scene = next;
      draw();
      if (refit || fitOwed) fit();
    },
    setStatus(text) {
      if (statusEl) statusEl.textContent = text;
    },
    fit,
    labels: () => labels,
    setLabels,
  };
}
