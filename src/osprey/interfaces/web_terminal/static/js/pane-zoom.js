// @ts-check
/* OSPREY Web Terminal — zoom one pane, not the page
 *
 * Ctrl+wheel over a pane (and a trackpad pinch, which Chrome and Firefox send
 * as Ctrl+wheel, and Safari as gesture events) zooms that pane alone; the
 * browser's own zoom is left to everywhere else (the sidebar, the bars) and to
 * Ctrl+= / Ctrl+− / Ctrl+0. On macOS Cmd+wheel does it too: Cmd is the zoom
 * key there, and no Mac browser zooms on Cmd+wheel. A wheel notch steps through the browser's levels
 * ({@link ZOOM_LEVELS}); a pinch zooms smoothly. Each tile's ⋯ menu has
 * "Zoom ▸" ({@link zoomMenuRow}), and its bar shows the level while it is not
 * 100% ({@link watchZoomChip}); a click on that chip goes back to 100%.
 *
 * A zoom belongs to what a pane shows: the Agent pane, each panel by its id,
 * and each artifact by itself. An artifact's tile and the Artifacts tile while
 * it shows that artifact share one zoom; the Artifacts tile follows the
 * artifact it shows ({@link setGalleryArtifact}). An artifact never zoomed is
 * at 100%. The zooms are the operator's, kept in this browser under a key
 * scoped per user, the {@link MAX_STORED} most recently set.
 *
 * HOW A PANE ZOOMS
 *   - The Agent's chat: CSS `zoom` on `#operator-container`.
 *   - The Agent's terminal: xterm's font size (terminal.js subscribes).
 *   - A panel frame: CSS `zoom` on the iframe element. The frame then lays out
 *     in a viewport 1/z of its box at z times the pixel ratio, as the browser's
 *     zoom would, and so does every frame inside it (an artifact's sandboxed
 *     content too). `zoom` also scales the frame's own px geometry, so a
 *     docked frame's box is placed divided by z ({@link sizeFrame}); a frame
 *     in a window of its own fills it by percentages, which `zoom` leaves as
 *     they are ({@link fillFrame}).
 *
 * WHERE CTRL+WHEEL IS HEARD: on the hub's document and each pane window's
 * ({@link installZoomWheel}), and in each panel frame on this origin, hooked
 * as it loads. An opaque-origin frame (an artifact's plot or table, a panel
 * the hub serves sandboxed) cannot be listened to: there the browser zooms
 * the page, and the ⋯ menu zooms the pane.
 *
 * @module pane-zoom
 */

import { isMacPlatform } from '/design-system/js/platform.js';
import { scopedStorageKey } from '/design-system/js/storage-scope.js';
import { queryAllDocuments } from './attached-window.js';
import { TERMINAL_RAIL_ID } from './panel-catalog.js';

/** @typedef {import('/design-system/js/context-menu.js').MenuItem} MenuItem */

/** Base key of the persisted zooms, scoped per user. */
export const ZOOM_STORAGE_KEY = 'osprey-pane-zoom';

/** The levels a wheel notch steps through, as the browser's own zoom does. */
export const ZOOM_LEVELS = [0.5, 0.67, 0.75, 0.8, 0.9, 1, 1.1, 1.25, 1.5, 1.75, 2];

/** The levels the ⋯ menu offers. */
export const MENU_LEVELS = [0.75, 0.9, 1, 1.1, 1.25, 1.5, 1.75, 2];

const MIN_ZOOM = ZOOM_LEVELS[0];
const MAX_ZOOM = ZOOM_LEVELS[ZOOM_LEVELS.length - 1];

/** A wheel delta at least this large (in pixels) is a mouse notch, not a pinch. */
const NOTCH_DELTA = 50;

/** The chat's root: what the Agent's chat zooms. */
const CHAT_SELECTOR = '#operator-container';

/** The Agent pane's content, in the hub or in its own window. */
const AGENT_SELECTOR = '.terminal-panel';

/** How many zooms are kept: one per artifact adds up; the oldest set goes first. */
export const MAX_STORED = 200;

/** The panel frames the hub owns (panel-iframe-factory.js). */
const FRAME_SELECTOR = 'iframe.panel-iframe';

/** @type {Record<string, number> | null} kind → zoom, read on first use */
let zooms = null;
/** @type {Set<(kind: string) => void>} */
const listeners = new Set();
/** Documents already listening. @type {WeakSet<Document>} */
const hooked = new WeakSet();
/**
 * The Artifacts tile's panel id, and the tile key of the artifact it shows.
 * The ids come from the caller (sidebar-artifacts.js): the modules that name
 * them reach the terminal, which imports this module, and importing them here
 * would close a cycle that leaves terminal.js half-initialised.
 * @type {{id: string, shown: string | null}}
 */
let gallery = { id: '', shown: null };

/**
 * What a tile key, panel id or rail id zooms as: the Agent pane, the artifact
 * the Artifacts tile shows (the tile itself while it shows none), or the key.
 * @param {string} key
 * @returns {string}
 */
export function zoomKindOf(key) {
  if (key === TERMINAL_RAIL_ID) return 'agent';
  if (key === gallery.id) return gallery.shown ?? key;
  return key;
}

/**
 * The Artifacts tile shows another artifact (or none): it takes that
 * artifact's zoom.
 * @param {string} galleryId  the Artifacts tile's panel id
 * @param {string | null} artifactKey  the tile key of the artifact it shows
 */
export function setGalleryArtifact(galleryId, artifactKey) {
  const before = zoomKindOf(galleryId);
  gallery = { id: galleryId, shown: artifactKey };
  const kind = zoomKindOf(galleryId);
  if (kind === before) return;
  applyZoom(kind);
  for (const fn of [...listeners]) fn(kind);
}

/** @returns {Record<string, number>} */
function stored() {
  if (zooms) return zooms;
  zooms = {};
  try {
    const raw = JSON.parse(localStorage.getItem(scopedStorageKey(ZOOM_STORAGE_KEY)) ?? '{}');
    for (const [kind, z] of Object.entries(raw ?? {})) {
      if (typeof z === 'number' && z >= MIN_ZOOM && z <= MAX_ZOOM) zooms[kind] = z;
    }
  } catch {
    /* unreadable: every pane at 100% */
  }
  return zooms;
}

/**
 * @param {string} key  a tile key, panel id or rail id
 * @returns {number} the pane's zoom, 1 by default
 */
export function paneZoom(key) {
  return stored()[zoomKindOf(key)] ?? 1;
}

/** @param {number} z @returns {string} e.g. "125%" */
export const zoomLabel = (z) => `${Math.round(z * 100)}%`;

/**
 * Zoom what a pane shows, clamped to 50–200%, and keep it: last in the
 * stored order, which drops the oldest past {@link MAX_STORED}.
 * @param {string} key  a tile key, panel id or rail id
 * @param {number} z
 */
export function setPaneZoom(key, z) {
  const kind = zoomKindOf(key);
  const next = Math.round(Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, z)) * 100) / 100;
  if (next === paneZoom(key)) return;
  const all = stored();
  delete all[kind];
  if (next !== 1) all[kind] = next;
  for (const old of Object.keys(all).slice(0, -MAX_STORED)) delete all[old];
  try {
    localStorage.setItem(scopedStorageKey(ZOOM_STORAGE_KEY), JSON.stringify(all));
  } catch {
    /* storage blocked: the zoom holds for this page */
  }
  applyZoom(kind);
  for (const fn of [...listeners]) fn(kind);
}

/**
 * Hear every zoom change.
 * @param {(kind: string) => void} fn  told the kind that changed
 * @returns {() => void} stops listening
 */
export function onPaneZoomChange(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

/**
 * The next level a wheel notch reaches from `z`.
 * @param {number} z  @param {1 | -1} dir  in or out
 */
function nextLevel(z, dir) {
  if (dir > 0) return ZOOM_LEVELS.find((l) => l > z + 0.001) ?? MAX_ZOOM;
  return [...ZOOM_LEVELS].reverse().find((l) => l < z - 0.001) ?? MIN_ZOOM;
}

// ---- Applying a zoom ---------------------------------------------------------

/** @param {string} kind  apply a kind's zoom to the panes on screen */
function applyZoom(kind) {
  if (kind === 'agent') {
    const z = paneZoom(TERMINAL_RAIL_ID);
    for (const el of queryAllDocuments(CHAT_SELECTOR)) {
      /** @type {HTMLElement} */ (el).style.zoom = z === 1 ? '' : String(z);
    }
    return;
  }
  for (const frame of /** @type {HTMLIFrameElement[]} */ (queryAllDocuments(FRAME_SELECTOR))) {
    if (zoomKindOf(frame.dataset.panelId ?? '') !== kind) continue;
    const box = frame.dataset.zoomBox;
    if (box === 'fill') fillFrame(frame);
    else if (box)
      sizeFrame(
        frame,
        .../** @type {[number, number, number, number]} */ (box.split(' ').map(Number))
      );
  }
}

/**
 * Place a docked panel frame at a rectangle (px, relative to its overlay)
 * at its pane's zoom: `zoom` scales the frame's own lengths, so they are set
 * divided by it.
 * @param {HTMLIFrameElement} frame
 * @param {number} left  @param {number} top  @param {number} width  @param {number} height
 */
export function sizeFrame(frame, left, top, width, height) {
  const z = paneZoom(frame.dataset.panelId ?? '');
  frame.dataset.zoomBox = [left, top, width, height].map(Math.round).join(' ');
  frame.style.zoom = z === 1 ? '' : String(z);
  frame.style.left = `${Math.round(left) / z}px`;
  frame.style.top = `${Math.round(top) / z}px`;
  frame.style.width = `${Math.round(width) / z}px`;
  frame.style.height = `${Math.round(height) / z}px`;
}

/**
 * Zoom a panel frame that fills a window of its own by percentages.
 * @param {HTMLIFrameElement} frame
 */
export function fillFrame(frame) {
  const z = paneZoom(frame.dataset.panelId ?? '');
  frame.dataset.zoomBox = 'fill';
  frame.style.zoom = z === 1 ? '' : String(z);
}

// ---- Ctrl+wheel and pinch ------------------------------------------------------

/**
 * A Ctrl+wheel over a pane: a mouse notch steps a level, a pinch scales.
 * `deltaMode` is read before any delta: Firefox reports a mouse notch in
 * lines only to a page that asks for the mode first, and in pixels (which
 * can fall under {@link NOTCH_DELTA}) to one that does not.
 * @param {WheelEvent} event  @param {string} key
 */
function zoomByWheel(event, key) {
  event.preventDefault();
  event.stopPropagation();
  const lines = event.deltaMode !== 0;
  if (!event.deltaY) return;
  const z = paneZoom(key);
  const notch = lines || Math.abs(event.deltaY) >= NOTCH_DELTA;
  setPaneZoom(
    key,
    notch ? nextLevel(z, event.deltaY < 0 ? 1 : -1) : z * Math.exp(-event.deltaY / 100)
  );
}

/**
 * Listen on one document for Ctrl+wheel and Safari's pinch, over the pane
 * `keyAt` names for an event's target.
 * @param {Document} doc  @param {(target: EventTarget | null) => string | null} keyAt
 */
function listen(doc, keyAt) {
  if (hooked.has(doc)) return;
  hooked.add(doc);
  const options = { capture: true, passive: false };
  doc.addEventListener(
    'wheel',
    (event) => {
      const zoomKey = event.ctrlKey || (event.metaKey && isMacPlatform());
      const key = zoomKey ? keyAt(event.target) : null;
      if (key) zoomByWheel(event, key);
    },
    options
  );
  // Safari: a pinch is gesturestart / gesturechange, `scale` from the start.
  let base = 1;
  /** @type {string | null} */
  let pinched = null;
  doc.addEventListener(
    'gesturestart',
    (event) => {
      pinched = keyAt(event.target);
      if (!pinched) return;
      event.preventDefault();
      base = paneZoom(pinched);
    },
    options
  );
  doc.addEventListener(
    'gesturechange',
    (event) => {
      if (!pinched) return;
      event.preventDefault();
      setPaneZoom(pinched, base * /** @type {any} */ (event).scale);
    },
    options
  );
}

/** @param {HTMLIFrameElement} frame  listen in a panel frame on this origin */
function hookFrame(frame) {
  const key = frame.dataset.panelId;
  /** @type {Document | null} */
  let doc = null;
  try {
    doc = frame.contentDocument;
  } catch {
    /* another origin: not ours to hear */
  }
  if (key && doc) listen(doc, () => key);
}

/**
 * Hear Ctrl+wheel on a document of the hub's (the page, or a pane window):
 * over the Agent pane there, and in every panel frame it holds as each loads.
 * Idempotent per document.
 * @param {Document} doc
 */
export function installZoomWheel(doc) {
  if (hooked.has(doc)) return;
  listen(doc, (target) =>
    /** @type {Element | null} */ (target)?.closest?.(AGENT_SELECTOR) ? TERMINAL_RAIL_ID : null
  );
  doc
    .querySelectorAll(FRAME_SELECTOR)
    .forEach((f) => hookFrame(/** @type {HTMLIFrameElement} */ (f)));
  // `load` does not bubble, but a capture listener on the document sees it.
  doc.addEventListener(
    'load',
    (event) => {
      const target = /** @type {Element | null} */ (event.target);
      if (target?.matches?.(FRAME_SELECTOR)) hookFrame(/** @type {HTMLIFrameElement} */ (target));
    },
    true
  );
}

/** Start: the chat at its zoom, Ctrl+wheel heard on this page. */
export function initPaneZoom() {
  applyZoom('agent');
  installZoomWheel(document);
}

// ---- The ⋯ row and the bar chip ---------------------------------------------

/**
 * A tile's "Zoom ▸" row, its level as the value.
 * @param {string} key  a tile key, panel id or rail id
 * @returns {MenuItem}
 */
export function zoomMenuRow(key) {
  const z = paneZoom(key);
  return {
    label: 'Zoom',
    value: zoomLabel(z),
    submenu: MENU_LEVELS.map((level) => ({
      label: zoomLabel(level),
      checked: Math.abs(level - z) < 0.005,
      run: () => setPaneZoom(key, level),
    })),
  };
}

/**
 * A tile bar's zoom chip, first in its action cluster while the pane is not at
 * 100%, showing the level; a click goes back to 100%.
 * @param {HTMLElement} actions  the bar's action cluster
 * @param {string} key  a tile key, panel id or rail id
 * @param {{dispose: () => void}[]} disposables  the bar's; the chip's listener joins them
 */
export function watchZoomChip(actions, key, disposables) {
  const chip = actions.ownerDocument.createElement('button');
  chip.type = 'button';
  chip.className = 'tile-tab-zoom';
  chip.addEventListener('click', (event) => {
    if (event.defaultPrevented) return;
    event.preventDefault();
    setPaneZoom(key, 1);
  });
  const paint = () => {
    const z = paneZoom(key);
    if (z === 1) {
      chip.remove();
      return;
    }
    chip.textContent = zoomLabel(z);
    chip.title = `Zoomed to ${zoomLabel(z)}. Click for 100%.`;
    chip.setAttribute('aria-label', `Zoom ${zoomLabel(z)}, back to 100%`);
    if (chip.parentElement !== actions) actions.prepend(chip);
  };
  paint();
  disposables.push({ dispose: onPaneZoomChange(paint) });
}

/** Forget the stored zooms and listeners (tests). */
export function resetPaneZoom() {
  zooms = null;
  gallery = { id: '', shown: null };
  listeners.clear();
}
