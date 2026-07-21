// @ts-check
/* OSPREY Web Terminal — Dock Iframe Adapter (overlay-fallback path)
 *
 * The dock-spike (tests/interfaces/web_terminal/spike_dockview_iframe.py) proved
 * that dockview RE-PARENTS a panel's content element on every cross-group move,
 * which detaches+reattaches any <iframe> inside it and forces a full document
 * reload — destroying the embedded app's live JS state. A splitter resize does
 * NOT re-parent, so pure resizing is safe; only regrouping reloads. Verdict:
 * FAIL → native iframe panels are unsafe, take the bounded overlay path.
 *
 * MECHANISM (overlay fallback)
 * ----------------------------
 * The service-panel iframes owned by panel-manager.js do NOT live inside their
 * dockview panels. They live in a single persistent OVERLAY layer that is a
 * plain child of .main-container — a DOM node dockview never manages and never
 * re-parents, so the iframes keep their document (and state) for the life of the
 * page. For each managed panel we add an EMPTY placeholder dockview panel; on
 * dockview layout / active-panel change we copy that placeholder group's content
 * rectangle onto the overlay iframe via inline geometry. Dragging a placeholder
 * to a new dock position therefore only moves the (empty) placeholder; dockview
 * reloads nothing, and the overlay iframe simply re-follows the new rectangle.
 * Scope: iframe panels only; geometry synced on layout/resize events only;
 * floating groups and maximize are out of scope for the overlay bound.
 *
 * SETTLE-ONLY SYNC (deliberate bound, not a gap): geometry re-syncs only on
 * dockview's settle events — onDidLayoutChange (panel add/move/close and
 * sash-END), onDidActivePanelChange, and window resize. dockview emits no
 * per-frame layout event during a live sash drag, and a per-pointer-frame/rAF
 * follow is deliberately excluded, so during a sash drag an overlay iframe holds
 * its last rectangle and lags the sash, then SNAPS to the correct geometry the
 * instant the drag ends. (The terminal's own continuous refit is separate — it
 * rides terminal.js's ResizeObserver, not this adapter.)
 *
 * The overlay layer is pointer-events:none so clicks fall through to the dock
 * chrome (tab strips, sashes) it visually covers; each iframe re-enables pointer
 * events for itself. During a dockview drag the iframes are made transparent to
 * the pointer (mirroring dock-workspace.js's own shield) so a separate-document
 * iframe can't swallow the drag events dockview needs for drop detection — the
 * shared .dock-dragging CSS rule can't reach these iframes because their inline
 * pointer-events wins, so the shield is re-applied here inline.
 *
 * FALLBACK: if no DockviewApi is available (dock shell absent or failed to
 * init), the adapter degrades to the pre-dock behavior — iframes mount in the
 * host element panel-manager passes in (#panel-content) and show/hide is a plain
 * display toggle, with no overlay and no placeholders.
 *
 * ADAPTER API (consumed by panel-manager.js and dock-sync.js)
 * -----------------------------------------------------------
 * All functions are keyed by panel-manager's panel id. See the JSDoc on each
 * export for the exact signature its consumer wires against.
 */

import {
  getDockApi,
  onDragGesture,
  defaultServiceWidth,
  setServiceRedock,
} from './dock-workspace.js';
import { PLACEHOLDER_PREFIX } from './dock-reconcile.js';

/**
 * One tracked service panel: its cached iframe (created/owned by panel-manager),
 * the id of the empty dockview placeholder it follows, the tab title, whether it
 * is currently meant to be on screen (false once closed/hidden), and the last
 * synced size (to throttle resize re-dispatch to real size changes).
 * @typedef {object} ManagedPanel
 * @property {HTMLIFrameElement} iframe
 * @property {string} placeholderId
 * @property {string} title
 * @property {boolean} visible
 * @property {number} [lastW]
 * @property {number} [lastH]
 */

const OVERLAY_CLASS = 'dock-iframe-overlay';
// Component name for the empty placeholder panels. dock-workspace.js's
// createComponent doesn't know it, so it returns its neutral fallback element —
// exactly the empty content node we want (we measure the group, never this node).
// This module owns the placeholder lifecycle; dock-sync.js imports both these
// ids to resolve/pre-create placeholders against the same namespace.
export const PLACEHOLDER_COMPONENT = 'dock-iframe-placeholder';
// The placeholder id namespace is defined in dock-reconcile.js (reconcile()
// treats it as always-recreatable); re-exported here as the adapter's public
// constant so dock-sync.js keeps one import site for the adapter contract.
export { PLACEHOLDER_PREFIX };
// dockview panel id of the native terminal/chat card (dock-core-shell), used as
// the anchor for the first placeholder: services open LEFT of the terminal at
// the classic 60/40 split, in expert and (locked) simple mode alike.
const TERMINAL_PANEL_ID = 'terminal';

/** @type {HTMLElement | null} */
let overlayEl = null;
/** @type {HTMLElement | null} */
let fallbackHostEl = null;
// dockview event disposables, retained for the page lifetime (never torn down —
// the adapter lives as long as the workspace shell does).
/** @type {any[]} */
const disposers = [];
/** @type {Map<string, ManagedPanel>} */
const managed = new Map();
let dockWired = false;

/** @param {string} id */
function placeholderIdFor(id) {
  return PLACEHOLDER_PREFIX + id;
}

/**
 * Resolve (and, once dockview is ready, wire) the adapter's dock coupling. Safe
 * to call on every operation: it no-ops after the first successful wiring and
 * returns null while no DockviewApi exists yet (fallback mode). panel-manager
 * inits the adapter before dock-workspace runs, so the api genuinely arrives
 * late; every entry point re-checks through here rather than caching at init.
 * @returns {any} the DockviewApi, or null in fallback mode
 */
function ensureDock() {
  const api = getDockApi();
  if (api && !dockWired) {
    ensureOverlay();
    if (overlayEl) {
      disposers.push(
        api.onDidLayoutChange(syncGeometry),
        api.onDidActivePanelChange(syncGeometry),
      );
      wireDragShield(api);
      window.addEventListener('resize', syncGeometry);
      // After a default-layout rebuild (reset / restore-fallback) the grid holds
      // only the terminal — re-dock every visible managed panel's placeholder so
      // its overlay iframe has a rectangle to follow again.
      setServiceRedock(redockVisiblePanels);
      dockWired = true;
    }
  }
  return overlayEl ? api : null;
}

/** Create the persistent overlay layer inside .main-container (once). */
function ensureOverlay() {
  if (overlayEl) return;
  const container = document.querySelector('.main-container');
  if (!(container instanceof HTMLElement)) return;
  const el = document.createElement('div');
  el.className = OVERLAY_CLASS;
  Object.assign(el.style, {
    position: 'absolute',
    top: '0',
    left: '0',
    right: '0',
    bottom: '0',
    pointerEvents: 'none',
    zIndex: '5',
  });
  container.appendChild(el);
  overlayEl = el;
}

/**
 * Re-apply dock-workspace.js's pointer shield to the overlay iframes. The shared
 * `.main-container.dock-dragging iframe { pointer-events:none }` rule can't win
 * against the inline `pointer-events:auto` these iframes carry, so we toggle it
 * inline across the whole drag gesture (start → any natural terminator).
 * @param {any} api
 */
function wireDragShield(api) {
  disposers.push(...onDragGesture(api, {
    onStart: () => { for (const e of managed.values()) e.iframe.style.pointerEvents = 'none'; },
    onEnd: () => { for (const e of managed.values()) e.iframe.style.pointerEvents = 'auto'; },
  }));
}

/** @param {HTMLIFrameElement} iframe */
function styleOverlayIframe(iframe) {
  Object.assign(iframe.style, {
    position: 'absolute',
    margin: '0',
    border: 'none',
    padding: '0',
    display: 'none',
    pointerEvents: 'auto',
    background: 'var(--bg-primary)',
    opacity: '1',
  });
}

/**
 * Ensure the empty placeholder dockview panel for a managed panel exists. The
 * first placeholder opens the service group LEFT of the terminal at the 60/40
 * split; the rest stack into that same group (so the initial arrangement
 * mirrors the single-active rail — and, in locked simple mode, preserves the
 * synthesized single tab-stack even for panels that dock late). In expert mode
 * the operator is free to drag any of them out into their own group.
 * @param {any} api
 * @param {ManagedPanel} entry
 */
function ensurePlaceholder(api, entry) {
  if (api.getPanel(entry.placeholderId)) return;
  /** @type {any} */
  const opts = { id: entry.placeholderId, component: PLACEHOLDER_COMPONENT, title: entry.title };
  const anchor = firstExistingPlaceholderId(api, entry.placeholderId);
  if (anchor) {
    opts.position = { referencePanel: anchor, direction: 'within' };
  } else if (api.getPanel(TERMINAL_PANEL_ID)) {
    opts.position = { referencePanel: TERMINAL_PANEL_ID, direction: 'left' };
    opts.initialWidth = defaultServiceWidth();
  }
  try {
    api.addPanel(opts);
  } catch (err) {
    console.error('dock-iframe: addPanel failed for', entry.placeholderId, err);
  }
}

/**
 * Re-create the placeholder for every visible managed panel, drop orphaned
 * placeholders, and re-sync geometry. Registered with dock-workspace's
 * setServiceRedock, which fires after every layout apply (default rebuild,
 * stored-layout restore, mode-flip restore).
 */
function redockVisiblePanels() {
  const api = getDockApi();
  if (!api || !overlayEl) return;
  for (const entry of managed.values()) {
    if (entry.visible) ensurePlaceholder(api, entry);
  }
  pruneOrphanPlaceholders(api);
  syncGeometry();
}

/** The service-id universe panel-manager knows, or null before it has loaded.
 *  @type {Set<string> | null} */
let knownServiceIds = null;

/**
 * Tell the adapter which service panels exist. reconcile() deliberately keeps
 * every `iframe:` placeholder in a stored layout (they may simply not be
 * created yet this session), so a placeholder whose service was genuinely
 * removed from the deployment survives the restore — this is the cleanup for
 * that case, called by panel-manager once its registry is loaded (and again on
 * runtime registration). Prunes immediately and on every later layout apply.
 * @param {Iterable<string>} ids
 */
export function setKnownServicePanels(ids) {
  knownServiceIds = new Set(ids);
  const api = ensureDock();
  if (api) {
    pruneOrphanPlaceholders(api);
    syncGeometry();
  }
}

/**
 * Remove placeholder panels whose service id is neither known to panel-manager
 * nor managed by this adapter. No-op until setKnownServicePanels has run —
 * before the registry loads, "unknown" would just mean "not fetched yet".
 * @param {any} api
 */
function pruneOrphanPlaceholders(api) {
  if (!knownServiceIds) return;
  const panels = Array.isArray(api.panels) ? [...api.panels] : [];
  for (const panel of panels) {
    const id = typeof panel?.id === 'string' ? panel.id : '';
    if (!id.startsWith(PLACEHOLDER_PREFIX)) continue;
    const sid = id.slice(PLACEHOLDER_PREFIX.length);
    if (knownServiceIds.has(sid) || managed.has(sid)) continue;
    try {
      api.removePanel(panel);
    } catch (err) {
      console.error('dock-iframe: orphan placeholder removal failed for', id, err);
    }
  }
}

/**
 * @param {any} api
 * @param {string} exceptId
 * @returns {string | null} another managed placeholder id that already exists
 */
function firstExistingPlaceholderId(api, exceptId) {
  for (const e of managed.values()) {
    if (e.placeholderId !== exceptId && api.getPanel(e.placeholderId)) return e.placeholderId;
  }
  return null;
}

/**
 * @param {any} api
 * @param {ManagedPanel} entry
 */
function removePlaceholder(api, entry) {
  const panel = api.getPanel(entry.placeholderId);
  if (panel) {
    try {
      api.removePanel(panel);
    } catch (err) {
      console.error('dock-iframe: removePanel failed for', entry.placeholderId, err);
    }
  }
}

/** @param {HTMLIFrameElement} iframe */
function dispatchResize(iframe) {
  if (!iframe.contentWindow) return;
  try {
    iframe.contentWindow.dispatchEvent(new Event('resize'));
  } catch { /* cross-origin — nothing we can do */ }
}

/**
 * Copy each managed placeholder's group content rectangle onto its overlay
 * iframe (inline geometry, relative to the overlay's own origin). An iframe is
 * shown only when its panel is visible AND its placeholder is the active tab in
 * its group; anything hidden behind another tab, closed, or lacking a live
 * placeholder is display:none. Only fires on dockview settle events, never per
 * pointer frame. No-op in fallback mode (no overlay / no api).
 */
function syncGeometry() {
  const api = getDockApi();
  if (!api || !overlayEl) return;
  const base = overlayEl.getBoundingClientRect();
  for (const entry of managed.values()) {
    const { iframe } = entry;
    const panel = entry.visible ? api.getPanel(entry.placeholderId) : null;
    const group = panel?.group;
    const content = group?.element?.querySelector('.dv-content-container');
    if (!panel || !content || group.activePanel !== panel) {
      iframe.style.display = 'none';
      continue;
    }
    const r = content.getBoundingClientRect();
    const w = Math.round(r.width);
    const h = Math.round(r.height);
    iframe.style.left = Math.round(r.left - base.left) + 'px';
    iframe.style.top = Math.round(r.top - base.top) + 'px';
    iframe.style.width = w + 'px';
    iframe.style.height = h + 'px';
    iframe.style.display = '';
    if (entry.lastW !== w || entry.lastH !== h) {
      entry.lastW = w;
      entry.lastH = h;
      dispatchResize(iframe);
    }
  }
}

// ---- Public API -----------------------------------------------------------

/**
 * Initialize the adapter. Records the fallback mount host (panel-manager's
 * #panel-content, used only when no dockview is present) and wires dockview if
 * it is already up. Idempotent-friendly: dock wiring completes lazily on the
 * first operation once the DockviewApi appears.
 * @param {{ fallbackHost?: HTMLElement | null }} [options]
 */
export function initDockIframeAdapter({ fallbackHost = null } = {}) {
  fallbackHostEl = fallbackHost;
  ensureDock();
}

/**
 * Take ownership of a panel-manager iframe: mount it in the overlay (or the
 * fallback host), create its placeholder dock panel, and geometry-sync it. This
 * is the panel-manager lifecycle seam — called from createIframe() right after
 * the element is built. The iframe's sandbox attrs, src, load listener and the
 * theme/mode/session postMessage protocol all stay in panel-manager; the adapter
 * only owns where the element lives and how big it is.
 * @param {string} panelId       panel-manager rail id
 * @param {HTMLIFrameElement} iframe
 * @param {{ title?: string }} [options]  dock tab title (defaults to the id)
 */
export function adoptIframe(panelId, iframe, { title = panelId } = {}) {
  const api = ensureDock();
  let entry = managed.get(panelId);
  if (entry) {
    // Re-adopt (a caller re-homing a panel's iframe): detach the previously adopted
    // element first so replacing the ref can't orphan a live iframe in the DOM.
    if (entry.iframe !== iframe) entry.iframe.remove();
    entry.iframe = iframe;
    entry.title = title;
    entry.visible = true;
  } else {
    entry = { iframe, placeholderId: placeholderIdFor(panelId), title, visible: true };
    managed.set(panelId, entry);
  }
  if (api && overlayEl) {
    styleOverlayIframe(iframe);
    overlayEl.appendChild(iframe);
    ensurePlaceholder(api, entry);
    syncGeometry();
  } else if (fallbackHostEl) {
    fallbackHostEl.appendChild(iframe);
  }
}

/**
 * Bring a panel forward: make its placeholder the active tab in its group and
 * reveal it. In fallback mode this is a classic single-active toggle across the
 * managed host. Consumed by panel-manager's activateTab().
 * @param {string} panelId
 */
export function focusPanel(panelId) {
  const entry = managed.get(panelId);
  if (!entry) return;
  entry.visible = true;
  const api = ensureDock();
  if (api && overlayEl) {
    ensurePlaceholder(api, entry);
    api.getPanel(entry.placeholderId)?.api?.setActive?.();
    syncGeometry();
  } else {
    for (const [id, e] of managed) e.iframe.style.display = id === panelId ? '' : 'none';
  }
}

/**
 * Hide a panel (closed / tab hidden): suppress its overlay iframe and drop its
 * placeholder dock panel so no empty tab lingers. The iframe element itself is
 * kept (cached) so a later show/focus re-reveals it with its state intact.
 * Consumed by panel-manager's hide paths.
 * @param {string} panelId
 */
export function hidePanel(panelId) {
  const entry = managed.get(panelId);
  if (!entry) return;
  entry.visible = false;
  entry.iframe.style.display = 'none';
  const api = ensureDock();
  if (api && overlayEl) {
    removePlaceholder(api, entry);
    syncGeometry();
  }
}
