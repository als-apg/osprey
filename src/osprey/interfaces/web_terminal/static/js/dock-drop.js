// @ts-check
/* OSPREY Web Terminal — Dock Drop Bridge
 *
 * The engine (the vendored dock-engine bundle) decides every drop on the
 * dock; dockview decides nothing. This module:
 *
 *   - prevents every dockview overlay (the api's and each group model's, wired
 *     on onDidAddGroup and unwired on onDidRemoveGroup) and every group model's
 *     own drop, so dockview never moves anything by itself;
 *   - reads a drag's source from the drag sources' own state (dock-tab.js's
 *     dockDragSource: a rail row's `{new}` or `{tab}` from rail-drag.js) and
 *     maps dockview's own header drag — the bar is the drag handle — to
 *     `{tile}` when no source was recorded first;
 *   - raises the drag shield (the `.dock-dragging` class on .main-container and
 *     the overlay iframes' inline pointer-events, registered by dock-iframe.js
 *     through setDragShield) from the same start, and lowers it at the end;
 *   - listens for dragover and drop in the capture phase on the dock root (its
 *     sashes included), runs the engine's hitTest and paints the resultRect
 *     preview (a dragleave out of the root clears it); a drop runs planDrop
 *     through the layout holder, which holds the apply while the drag is live
 *     and runs it once the drag has ended.
 *
 * One refusal sits on top of the engine's zones, before a target is shown:
 * every tile holds one panel (dock-policy.js), so a tile's header and centre —
 * the engine's zones that would add a panel to the tile — have no target for
 * any source. A drop lands on a tile's edge, which splits it, or on a dock
 * edge, which makes a new tile along that side; the Agent pane's tile splits
 * like any other and never takes a panel.
 *
 * A drag with no recognised source (a file, text, another page's payload) gets
 * nothing: no preventDefault, no preview, no shield, no drop.
 */

import { hitTest, keysOf, planDrop, resultRect, tileIdsOf } from '/static/vendor/dock-engine.mjs';
import { getLayoutHolder } from './dock-apply.js';
import {
  clearDockDrag,
  dockDragSource,
  dockDragStart,
  measureTabRects,
  onDockDragStart,
} from './dock-tab.js';

/** @typedef {import('/static/vendor/dock-engine.mjs').Source} Source */
/** @typedef {import('/static/vendor/dock-engine.mjs').Target} Target */
/** @typedef {import('/static/vendor/dock-engine.mjs').Layout} Layout */
/** @typedef {import('./dock-apply.js').LayoutHolder} LayoutHolder */
/** @typedef {{ x: number, y: number, width: number, height: number }} Rect */

/** The preview layer's class (dockview-overrides.css). */
export const PREVIEW_CLASS = 'dock-drop-preview';
/** The class on .main-container while a dock drag is live (terminal.css). */
export const DRAGGING_CLASS = 'dock-dragging';

/**
 * @typedef {object} DockDropOptions
 * @property {() => LayoutHolder | null} [holder]  the page's holder; getLayoutHolder by default
 */

/** @type {((on: boolean) => void) | null} */
let iframeShield = null;

/**
 * Register the overlay iframes' inline pointer shield (dock-iframe.js's
 * setIframePointerShield): the `.dock-dragging` class rule cannot reach past
 * their inline `pointer-events: auto`. A registration seam rather than an
 * import: dock-iframe imports the workspace, which imports this module.
 * @param {((on: boolean) => void) | null} fn
 */
export function setDragShield(fn) {
  iframeShield = fn;
}

/** @param {boolean} on */
function shield(on) {
  if (typeof document === 'undefined') return;
  document.querySelector('.main-container')?.classList.toggle(DRAGGING_CLASS, on);
  iframeShield?.(on);
}

/**
 * The preview style of a target, as in the runtime's drop preview: an edge
 * paints the half the item lands in, a dock-edge strip (an outer target) the
 * new strip it lands in. A border target has no pointer zone (only the
 * agent's drop names it), and a zone target is never admitted, so neither
 * paints anything.
 * @param {Target} target
 * @returns {'split' | null}
 */
export function previewKind(target) {
  return 'edge' in target || 'outer' in target ? 'split' : null;
}

/**
 * Whether a drag may land on `target`: every edge and dock-edge target, never
 * a zone target — a tile holds one panel, so nothing drops into one.
 * @param {Target} target
 * @returns {boolean}
 */
export function admitted(target) {
  return !('zone' in target);
}

/**
 * How far a tile's card sits inside the engine's tile rect: half the gutter
 * between cards (dockview-overrides.css pads every leaf view by that much), so
 * the highlight lands on the card rather than on the gutter around it.
 * @param {HTMLElement} root
 * @returns {number}
 */
function cardInset(root) {
  const gutter = parseFloat(getComputedStyle(root).getPropertyValue('--wt-gutter'));
  return Number.isFinite(gutter) && gutter > 0 ? gutter / 2 : 0;
}

/** @param {Element} el @returns {Rect} */
function rectOf(el) {
  const r = el.getBoundingClientRect();
  return { x: r.left, y: r.top, width: r.width, height: r.height };
}

/**
 * Whether `source` names something in `layout`; a {new} key is any key.
 * @param {Source} source
 * @param {Layout} layout
 * @returns {boolean}
 */
function knownSource(source, layout) {
  if ('tab' in source) return keysOf(layout).includes(source.tab);
  if ('tile' in source) return tileIdsOf(layout).includes(source.tile);
  return typeof source.new === 'string' && source.new !== '';
}

/**
 * Wire the drop bridge onto a live dock. Returns the teardown.
 * @param {any} api  the DockviewApi
 * @param {HTMLElement} root  the dock root (#dock-root)
 * @param {DockDropOptions} [options]
 * @returns {() => void}
 */
export function initDockDrop(api, root, options = {}) {
  const holderOf = options.holder ?? getLayoutHolder;
  /** @type {(() => void) | null} */
  let lowerDrag = null;
  /** @type {HTMLElement | null} */
  let layer = null;
  let previewKey = 'null';

  // --- the preview ------------------------------------------------------------------

  const clearPreview = () => {
    previewKey = 'null';
    layer?.remove();
    layer = null;
  };

  /**
   * @param {Target | null} target
   * @param {import('/static/vendor/dock-engine.mjs').ResultRect | null} result
   */
  const showPreview = (target, result) => {
    const key = JSON.stringify([target, result]);
    if (key === previewKey) return;
    if (target === null || result === null) {
      clearPreview();
      return;
    }
    previewKey = key;
    const parent = /** @type {HTMLElement | null} */ (root.offsetParent ?? root.parentElement);
    if (!parent) return;
    const origin = parent.getBoundingClientRect();
    if (!layer || layer.parentElement !== parent) {
      layer?.remove();
      layer = document.createElement('div');
      layer.className = PREVIEW_CLASS;
      layer.setAttribute('aria-hidden', 'true');
      parent.appendChild(layer);
    }
    const kind = previewKind(target);
    if (kind === null) {
      clearPreview();
      return;
    }
    // One rect stays mounted while the pointer moves between zones, so the CSS
    // glides it from one zone to the next; the placeholder is its child.
    let box = /** @type {HTMLElement | null} */ (layer.querySelector(`.${PREVIEW_CLASS}__rect`));
    if (!box) {
      box = document.createElement('div');
      const placeholder = document.createElement('div');
      placeholder.className = `${PREVIEW_CLASS}__placeholder`;
      box.appendChild(placeholder);
    }
    box.className = `${PREVIEW_CLASS}__rect ${PREVIEW_CLASS}__rect--${kind}`;
    const { rect } = result;
    const inset = Math.min(cardInset(root), rect.width / 2, rect.height / 2);
    Object.assign(box.style, {
      left: `${rect.x + inset - origin.left}px`,
      top: `${rect.y + inset - origin.top}px`,
      width: `${rect.width - 2 * inset}px`,
      height: `${rect.height - 2 * inset}px`,
    });
    layer.replaceChildren(box);
  };

  // --- the live drag ------------------------------------------------------------------

  /** Marks the drag live once; its end lowers it and lets the holder apply. */
  const raiseDrag = () => {
    holderOf()?.setDragLive(true);
    if (lowerDrag) return;
    shield(true);
    /** @type {ReturnType<typeof setTimeout> | null} */
    let timer = null;
    // A turn later: the root's own drop handler still reads the source, and
    // dockview's drag source still exists for its own dragend.
    const end = () => {
      if (timer === null) timer = setTimeout(lower, 0);
    };
    const onKey = (/** @type {KeyboardEvent} */ e) => {
      if (e.key === 'Escape') end();
    };
    // A drag source detached mid-drag never gets its dragend: no native drag
    // delivers pointer or mouse moves, so a pointerdown, or a move with no
    // button held, means the drag is long over. Drop its source and shield too.
    const onStray = (/** @type {MouseEvent} */ e) => {
      if (e.type === 'mousemove' && e.buttons !== 0) return;
      clearDockDrag();
      end();
    };
    const lower = () => {
      document.removeEventListener('drop', end, true);
      document.removeEventListener('dragend', end, true);
      document.removeEventListener('keydown', onKey, true);
      document.removeEventListener('pointerdown', onStray, true);
      document.removeEventListener('mousemove', onStray, true);
      if (timer !== null) clearTimeout(timer);
      lowerDrag = null;
      clearPreview();
      shield(false);
      holderOf()?.setDragLive(false);
    };
    lowerDrag = lower;
    document.addEventListener('drop', end, true);
    document.addEventListener('dragend', end, true);
    document.addEventListener('keydown', onKey, true);
    document.addEventListener('pointerdown', onStray, true);
    document.addEventListener('mousemove', onStray, true);
  };

  /**
   * The live drag's source, or null for a foreign drag (or none).
   * @param {Layout} layout
   * @returns {Source | null}
   */
  const currentSource = (layout) => {
    const own = dockDragSource();
    return own && knownSource(own, layout) ? own : null;
  };

  /**
   * The target and result a drag of `source` at the event's point means.
   * @param {DragEvent} event
   * @param {Layout} layout
   * @param {Source} source
   */
  const targetAt = (event, layout, source) => {
    const dockRect = rectOf(root);
    const tabRects = measureTabRects(root);
    const point = { x: event.clientX, y: event.clientY };
    const target = hitTest(layout, dockRect, tabRects, point, source);
    if (target === null || !admitted(target)) return { target: null, result: null };
    return { target, result: resultRect(layout, dockRect, tabRects, source, target) };
  };

  /** @param {DragEvent} event */
  const onDragOver = (event) => {
    const layout = holderOf()?.layout ?? null;
    const source = layout === null ? null : currentSource(layout);
    if (layout === null || source === null) return;
    event.preventDefault();
    if (event.dataTransfer) event.dataTransfer.dropEffect = 'move';
    raiseDrag();
    const { target, result } = targetAt(event, layout, source);
    showPreview(target, result);
  };

  /**
   * A drag leaving the dock (to the page beside it, or out of the window: no
   * relatedTarget) leaves no highlight behind. A move between two elements of
   * the dock is not a leave; a later dragover over the dock paints again.
   * @param {DragEvent} event
   */
  const onDragLeave = (event) => {
    const next = /** @type {Node | null} */ (event.relatedTarget);
    if (next !== null && root.contains(next)) return;
    clearPreview();
  };

  /** @param {DragEvent} event */
  const onDrop = (event) => {
    const holder = holderOf();
    const layout = holder?.layout ?? null;
    const source = layout === null ? null : currentSource(layout);
    if (!holder || layout === null || source === null) return;
    event.preventDefault();
    const { target } = targetAt(event, layout, source);
    clearDockDrag();
    clearPreview();
    if (target === null) return;
    // Held while the drag is live: the holder applies once it has ended.
    holder.run((current, ctx) => planDrop(current, source, target, ctx.dock));
  };

  // --- dockview decides nothing ----------------------------------------------------------

  /** @param {{ preventDefault(): void }} event */
  const prevent = (event) => event.preventDefault();
  /** @type {Map<string, Array<{ dispose(): void }>>} */
  const wired = new Map();
  /** @param {any} group */
  const wire = (group) => {
    const model = group?.model;
    if (!model || wired.has(group.id)) return;
    /** @type {Array<{ dispose(): void }>} */
    const subscriptions = [];
    const on = (/** @type {string} */ name, /** @type {(e: any) => void} */ fn) => {
      if (typeof model[name] === 'function') subscriptions.push(model[name](fn));
    };
    on('onWillShowOverlay', prevent);
    on('onWillDrop', prevent);
    // The bar is the drag handle: dockview's own header drag is the tile.
    const tileStart = (/** @type {any} */ e) => {
      const id = e?.group?.id ?? e?.panel?.group?.id ?? group.id;
      dockDragStart({ tile: String(id) }, null);
    };
    on('onTabDragStart', tileStart);
    on('onGroupDragStart', tileStart);
    wired.set(group.id, subscriptions);
  };
  /** @param {any} group */
  const unwire = (group) => {
    for (const subscription of wired.get(group?.id) ?? []) subscription?.dispose?.();
    wired.delete(group?.id);
  };
  for (const group of api.groups ?? []) wire(group);
  /** @type {Array<{ dispose(): void } | undefined>} */
  const subscriptions = [];
  const onApi = (/** @type {string} */ name, /** @type {(e: any) => void} */ fn) => {
    if (typeof api[name] === 'function') subscriptions.push(api[name](fn));
  };
  onApi('onDidAddGroup', wire);
  onApi('onDidRemoveGroup', unwire);
  onApi('onWillShowOverlay', prevent);
  const unsubscribeStart = onDockDragStart(() => raiseDrag());

  root.addEventListener('dragover', onDragOver, true);
  root.addEventListener('dragleave', onDragLeave, true);
  root.addEventListener('drop', onDrop, true);

  return () => {
    root.removeEventListener('dragover', onDragOver, true);
    root.removeEventListener('dragleave', onDragLeave, true);
    root.removeEventListener('drop', onDrop, true);
    unsubscribeStart();
    for (const subscription of subscriptions) subscription?.dispose?.();
    for (const group of [...wired.keys()]) unwire({ id: group });
    lowerDrag?.();
    clearPreview();
  };
}
