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
 * The adapter never touches dockview's panels itself: a placeholder enters or
 * leaves the dock as one operation on the layout holder (dock-apply.js), by
 * the placement rules in dock-open.js, and the holder applies the result.
 * The adapter reads the groups back to place its iframes.
 *
 * LIVE SYNC: geometry re-syncs on dockview's settle events — onDidLayoutChange
 * (panel add/move/close and sash-END), onDidActivePanelChange, and window
 * resize — and, between settles, via a ResizeObserver on each managed
 * placeholder's group content container. dockview emits no per-frame layout
 * event during a live sash drag, but the drag resizes those containers every
 * frame, so the observer follows the sash continuously instead of the
 * iframe freezing and snapping at mouseup. Settle events additionally REWIRE
 * the observed set (groups are created/destroyed as panels move); the observer
 * callback only re-applies geometry, never rewires, so it cannot recurse.
 * (The terminal's own continuous refit is separate — it rides terminal.js's
 * ResizeObserver, not this adapter.)
 *
 * The overlay layer is pointer-events:none so clicks fall through to the dock
 * chrome (tile bars, sashes) it visually covers; each iframe re-enables pointer
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
 * NOT-ANSWERING NOTICE: each managed panel has a `.tile-notice` sibling element
 * in the overlay, drawn over its tile body with the iframe's geometry. It
 * carries the time the caller hands in, stays laid out but empty while the
 * panel answers (a polite live region present before its text), is hidden
 * wherever the iframe is hidden, and takes no pointer events. Reachability is
 * held per panel id, so it survives a tile closing and is shown again when the
 * tile returns. In fallback mode the host carries one notice for whichever
 * panel it is showing.
 *
 * ADAPTER API (consumed by panel-manager.js and dock-sync.js)
 * -----------------------------------------------------------
 * All functions are keyed by panel-manager's panel id. See the JSDoc on each
 * export for the exact signature its consumer wires against.
 */

import { applyingSessionLayout, getDockApi, setServiceRedock } from './dock-workspace.js';
import { setDragShield } from './dock-drop.js';
import {
  PLACEHOLDER_COMPONENT,
  PLACEHOLDER_PREFIX,
  TERMINAL_KEY,
  getLayoutHolder,
} from './dock-apply.js';
import { closeKeys, openPanel } from './dock-open.js';
import { keysOf } from '/static/vendor/dock-engine.mjs';
import { flashElement } from '/design-system/js/highlight.js';
import { isPoppedOut } from './popout.js';

/**
 * One tracked service panel: its cached iframe (created/owned by panel-manager),
 * the id of the empty dockview placeholder it follows, the tile title, whether it
 * is currently meant to be on screen (false once closed/hidden), the last synced
 * size (to throttle resize re-dispatch to real size changes), and its lazily
 * created agent-glow overlay (see glowPanel).
 * @typedef {object} ManagedPanel
 * @property {HTMLIFrameElement} iframe
 * @property {string} placeholderId
 * @property {string} title
 * @property {boolean} visible
 * @property {number} [lastW]
 * @property {number} [lastH]
 * @property {HTMLElement} [glowEl]
 * @property {HTMLElement} [noticeEl]
 */

const OVERLAY_CLASS = 'dock-iframe-overlay';
// The placeholder id namespace and component are the layout holder's
// (dock-apply.js adds and strips the prefix, and serializes the component);
// re-exported here as the adapter's public contract for the modules that
// resolve placeholders by id.
export { PLACEHOLDER_COMPONENT, PLACEHOLDER_PREFIX };

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
/**
 * Panel id → the display time its backend stopped answering, for panels not
 * answering now. Held apart from `managed` so a panel can be marked before its
 * tile exists. An id has an entry only between the caller's "not answering"
 * and its "answering again".
 * @type {Map<string, string>}
 */
const notAnsweringSince = new Map();
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
      disposers.push(api.onDidLayoutChange(syncGeometry), api.onDidActivePanelChange(syncGeometry));
      // The drop bridge raises the drag shield over the overlay iframes for
      // every dock drag (a tile's bar, a sidebar row).
      setDragShield(setIframePointerShield);
      window.addEventListener('resize', syncGeometry);
      // After a whole-layout apply (reset, a stored-layout restore) the managed
      // set is settled against what the layout holds.
      setServiceRedock(reconcilePlaceholders);
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
 * Toggle the inline pointer shield over every managed overlay iframe. The
 * shared `.main-container.dock-dragging iframe { pointer-events:none }` rule
 * can't win against the inline `pointer-events:auto` these iframes carry, so
 * the toggle must be inline. Registered with the drop bridge (dock-drop.js),
 * which raises it for every dock drag and lowers it at the drag's end.
 *
 * SASH drags need no shield here — and must not get one. dockview-core's own
 * sash handling already disables pointer-events on every iframe in the
 * document for the whole gesture (disableIframePointEvents in its splitview),
 * by SNAPSHOTTING each iframe's inline pointer-events value on sash
 * pointerdown and restoring the snapshot on pointerup. A second, adapter-side
 * shield running in the capture phase sets 'none' before dockview snapshots,
 * so dockview records 'none' as the "original" value and its restore
 * re-freezes every overlay iframe after the drag — permanently, since each
 * further drag re-snapshots the poisoned value. That was the issue-638
 * panel freeze; dock-iframe.test.mjs pins the gesture contract.
 * @param {boolean} on
 */
export function setIframePointerShield(on) {
  for (const e of managed.values()) e.iframe.style.pointerEvents = on ? 'none' : 'auto';
}

/**
 * Corner rounding of a tile BODY drawn over the dock: an overlay covers a
 * group's `.dv-content-container`, which sits under the tile bar inside the
 * card (dockview-overrides.css), so only its two bottom corners meet the
 * card's `--radius-xl` corners. The iframe, its notice and its glow each take
 * it, because the overlay is outside the card and its `overflow: hidden` clip.
 */
const TILE_BODY_RADIUS = '0 0 var(--radius-xl) var(--radius-xl)';

/** @param {HTMLIFrameElement} iframe */
function styleOverlayIframe(iframe) {
  Object.assign(iframe.style, {
    position: 'absolute',
    margin: '0',
    border: 'none',
    padding: '0',
    display: 'none',
    pointerEvents: 'auto',
    background: 'var(--surface-card)',
    borderRadius: TILE_BODY_RADIUS,
    opacity: '1',
  });
}

/**
 * The service keys the layout holder's dock holds right now — every key but
 * the terminal's. Empty before the holder exists.
 * @returns {Set<string>}
 */
function dockedServiceKeys() {
  const keys = new Set(keysOf(getLayoutHolder()?.layout ?? null));
  keys.delete(TERMINAL_KEY);
  return keys;
}

/**
 * Put a managed panel's placeholder in the dock through the layout holder, by
 * the open rule (dock-open.js's openPanel): a panel already in the dock stays
 * where it is; otherwise it takes the service tile the operator used last,
 * replacing the panel that tile showed — nothing resizes, and the replaced
 * panel keeps its sidebar row (its iframe stays cached here; the next
 * geometry pass conceals it). With no service tile at all it opens the first
 * one LEFT of the agent pane at the classic 60/40 split (or against the
 * grid's left edge when the agent pane's tile is closed too). No-op without a
 * holder (fallback mode).
 * @param {string} panelId
 * @param {ManagedPanel} entry
 */
function place(panelId, entry) {
  const holder = getLayoutHolder();
  if (!holder) return;
  holder.setPanelRecord(panelId, { title: entry.title });
  holder.run((layout, ctx) => openPanel(layout, panelId, ctx));
}

/**
 * Take a managed panel off screen: mark it not-visible and hide its iframe and
 * its notice. No geometry pass is needed for the notice to vanish with its tile.
 * @param {ManagedPanel} entry
 */
function concealEntry(entry) {
  entry.visible = false;
  entry.iframe.style.display = 'none';
  if (entry.noticeEl) entry.noticeEl.style.display = 'none';
}

/**
 * Settle the managed set after a whole-layout apply (a default rebuild, a
 * stored-layout restore): orphaned placeholders leave the layout, geometry is
 * re-synced — which shows every managed panel the applied layout docks and
 * conceals every one it lacks — and panel-manager's placeholders-applied
 * handler runs, told whether the layout is a session's own (`asLeft`).
 * Registered with dock-workspace's setServiceRedock.
 */
function reconcilePlaceholders() {
  const api = getDockApi();
  if (!api || !overlayEl) return;
  pruneOrphanPlaceholders();
  syncGeometry();
  placeholdersAppliedHandler?.({ asLeft: applyingSessionLayout() });
}

/**
 * Handler run after every layout apply, once the placeholders are final —
 * registered by panel-manager, which fills a restored placeholder with its
 * panel's iframe. A stored layout brings back a placeholder for every tile
 * that was open, but only the landing panel's activation creates an iframe;
 * without this the other restored tiles would stay empty cards until their
 * sidebar rows are clicked.
 * @type {((options: { asLeft: boolean }) => void) | null}
 */
let placeholdersAppliedHandler = null;

/** @param {((options: { asLeft: boolean }) => void) | null} fn */
export function setPlaceholdersAppliedHandler(fn) {
  placeholdersAppliedHandler = fn;
}

/** The service-id universe panel-manager knows, or null before it has loaded.
 *  @type {Set<string> | null} */
let knownServiceIds = null;

/** The server-visible service ids — RAIL MEMBERSHIP: a LIVE reference to
 *  panel-manager's own set (kept current on every SSE visibility change), or
 *  null before the registry has loaded. Used to prune restored placeholders
 *  whose panel is no longer a rail member: under one panel per tile such a
 *  placeholder would be a whole ghost tile.
 *  @type {Set<string> | null} */
let serverVisibleIds = null;

/**
 * Hand the adapter a live reference to the server-visible panel-id set. Called
 * by panel-manager once its registry is seeded (before setKnownServicePanels,
 * whose prune consumes it). Null-safe: before this runs, visibility pruning is
 * skipped entirely — an early layout restore must not drop every placeholder
 * just because the set has not been seeded yet.
 * @param {Set<string>} ids
 */
export function setServerVisiblePanels(ids) {
  serverVisibleIds = ids;
}

/**
 * Tell the adapter which service panels exist. reconcile() deliberately keeps
 * every `iframe:` placeholder in a stored layout (they may simply not be
 * created yet this session), so a placeholder whose service was genuinely
 * removed from the deployment survives the restore — this is the cleanup for
 * that case, called by panel-manager once its registry is loaded (and again
 * by panel-lifecycle's addPanel on runtime registration). Prunes immediately
 * and on every later layout apply.
 * @param {Iterable<string>} ids
 */
export function setKnownServicePanels(ids) {
  knownServiceIds = new Set(ids);
  const api = ensureDock();
  if (api) {
    pruneOrphanPlaceholders();
    syncGeometry();
  }
}

/**
 * Take out of the layout the placeholders that must not hold a tile: services
 * neither known to panel-manager nor managed here (removed from the
 * deployment); once the membership set is seeded, services no longer in the
 * rail that are not currently being shown locally (a restored placeholder for
 * a removed panel is a ghost tile under the one-panel-per-tile model); and
 * services living in their own pop-out window (a layout stored before the
 * pop-out and restored after the window was adopted would put an empty tile
 * back). The `managed` visible flag covers the window where a just-added
 * panel's membership POST has not echoed back into the server set yet. One
 * holder operation for all of them. No-op until setKnownServicePanels has run
 * — before the registry loads, "unknown" would just mean "not fetched yet".
 */
function pruneOrphanPlaceholders() {
  const holder = getLayoutHolder();
  if (!knownServiceIds || !holder) return;
  /** @type {string[]} */
  const orphans = [];
  for (const sid of dockedServiceKeys()) {
    const unknown = !knownServiceIds.has(sid) && !managed.has(sid);
    const closed = !!serverVisibleIds && !serverVisibleIds.has(sid) && !managed.get(sid)?.visible;
    if (unknown || closed || isPoppedOut(sid)) orphans.push(sid);
  }
  if (orphans.length > 0) holder.run((layout) => closeKeys(layout, orphans));
}

/** @param {HTMLIFrameElement} iframe */
function dispatchResize(iframe) {
  if (!iframe.contentWindow) return;
  try {
    iframe.contentWindow.dispatchEvent(new Event('resize'));
  } catch {
    /* cross-origin — nothing we can do */
  }
}

/**
 * Live follower for the stretch between dockview settle events: observes each
 * managed placeholder's group content container, whose per-frame resize during
 * a sash drag has no dockview event.
 * Rewired by syncGeometry() on every settle — groups come and go as panels
 * move. The callback re-applies geometry only; it never rewires, so the
 * initial fire observe() triggers cannot recurse into another rewire.
 * @type {ResizeObserver | null}
 */
let contentObserver = null;

/** @param {any} api */
function rewireContentObservers(api) {
  if (!contentObserver) contentObserver = new ResizeObserver(() => applyGeometry());
  contentObserver.disconnect();
  for (const entry of managed.values()) {
    if (!entry.visible) continue;
    const content = api
      .getPanel(entry.placeholderId)
      ?.group?.element?.querySelector('.dv-content-container');
    if (content) contentObserver.observe(content);
  }
}

/**
 * Settle-event sync: re-apply all overlay geometry, then re-point the live
 * content observers at the placeholders' current groups. No-op in fallback
 * mode (no overlay / no api).
 */
function syncGeometry() {
  const api = getDockApi();
  if (!api || !overlayEl) return;
  applyGeometry();
  rewireContentObservers(api);
}

/**
 * Copy each managed placeholder's group content rectangle onto its overlay
 * iframe (inline geometry, relative to the overlay's own origin). The held
 * layout decides what shows: a managed panel it docks is visible — even one
 * concealed earlier, when a later layout (a restore, a reset, a preset, an
 * agent arrangement) gives it a tile again — and a panel it no longer holds
 * was replaced in its tile (the open rule) or left out by a layout applied
 * whole, so it is concealed, its iframe cached for its sidebar row. An
 * iframe is shown only when its panel is visible AND its placeholder is live
 * in its group; anything else is display:none.
 */
function applyGeometry() {
  const api = getDockApi();
  if (!api || !overlayEl) return;
  const base = overlayEl.getBoundingClientRect();
  const docked = dockedServiceKeys();
  /** @param {HTMLElement} el @param {DOMRect} r @param {DOMRect} origin */
  const place = (el, r, origin) => {
    el.style.left = Math.round(r.left - origin.left) + 'px';
    el.style.top = Math.round(r.top - origin.top) + 'px';
    el.style.width = Math.round(r.width) + 'px';
    el.style.height = Math.round(r.height) + 'px';
  };
  for (const [panelId, entry] of managed) {
    if (docked.has(panelId)) entry.visible = true;
    else if (entry.visible) concealEntry(entry);
    const { iframe } = entry;
    const panel = entry.visible ? api.getPanel(entry.placeholderId) : null;
    const group = panel?.group;
    const content = group?.element?.querySelector('.dv-content-container');
    if (!panel || !content || group.activePanel !== panel) {
      iframe.style.display = 'none';
      if (entry.noticeEl) entry.noticeEl.style.display = 'none';
      continue;
    }
    const r = content.getBoundingClientRect();
    const w = Math.round(r.width);
    const h = Math.round(r.height);
    place(iframe, r, base);
    iframe.style.display = '';
    // The notice spans the whole content rectangle: that is what the wash covers.
    const notice = ensureNoticeEl(panelId, entry, overlayEl);
    place(notice, r, base);
    notice.style.display = '';
    paintNotice(notice, panelId);
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
 * @param {{ title?: string }} [options]  dock tile title (defaults to the id)
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
    place(panelId, entry);
    syncGeometry();
  } else if (fallbackHostEl) {
    fallbackHostEl.appendChild(iframe);
  }
}

/**
 * Bring a panel forward: make its placeholder's tile the active one and
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
    place(panelId, entry);
    api.getPanel(entry.placeholderId)?.api?.setActive?.();
    syncGeometry();
  } else {
    for (const [id, e] of managed) e.iframe.style.display = id === panelId ? '' : 'none';
    fallbackShownId = panelId;
    paintFallbackNotice();
  }
}

/**
 * Hide a panel (closed / removed from the rail): suppress its overlay iframe and take its
 * placeholder out of the layout so no empty tile lingers. The iframe element
 * itself is kept (cached) so a later show/focus re-reveals it with its state
 * intact. Also handles panels this adapter never adopted (no managed entry) —
 * a restored placeholder for a panel closed on another client still needs its
 * tile removed. Consumed by panel-manager's, panel-sse's and
 * panel-placement's hide paths.
 * @param {string} panelId
 */
export function hidePanel(panelId) {
  const entry = managed.get(panelId);
  if (entry) concealEntry(entry);
  const api = ensureDock();
  if (!api || !overlayEl) clearFallbackShown(panelId);
  if (api && overlayEl) {
    getLayoutHolder()?.run((layout) => closeKeys(layout, [panelId]));
    syncGeometry();
  }
}

/**
 * Mark a panel vacated WITHOUT touching its dock placeholder — the complement
 * of hidePanel for the human tile-close gesture, where the tile bar's own
 * close handler takes the placeholder out through the holder (dock-tab.js,
 * dock-workspace's closeDockPanel). Only the adapter-side state is cleared:
 * the overlay iframe is concealed (cached for a later re-activation) and the
 * entry marked not-visible so nothing re-materializes it. Geometry
 * re-syncs on the removal's own onDidLayoutChange, so none is forced here.
 * @param {string} panelId
 */
export function concealPanel(panelId) {
  const entry = managed.get(panelId);
  if (!entry) return;
  concealEntry(entry);
  if (!ensureDock()) clearFallbackShown(panelId);
}

/**
 * Show or clear the not-answering notice over a panel's tile.
 *
 * The caller decides reachability; the adapter only shows it. The state is
 * held per panel id, so a call made before the tile exists, or while it is
 * closed, takes effect when the tile is next on screen. A repeated identical
 * call changes nothing.
 * @param {string} panelId
 * @param {boolean} reachable
 * @param {string} [since] - the time the backend stopped answering, already formatted
 *   for display; required when `reachable` is false
 */
export function setPanelReachable(panelId, reachable, since) {
  const before = notAnsweringSince.get(panelId);
  if (reachable) notAnsweringSince.delete(panelId);
  else notAnsweringSince.set(panelId, since ?? '');
  if (notAnsweringSince.get(panelId) === before) return;
  if (ensureDock() && overlayEl) applyGeometry();
  else paintFallbackNotice();
}

// ---- Agent attribution glow ------------------------------------------------

/** Class of the per-panel glow element; styled in css/terminal.css. */
const GLOW_CLASS = 'tile-glow';

/**
 * The glow element for a managed panel, created on first use and reused after.
 * It is a sibling of the overlay iframes rather than anything inside them: the
 * iframes are separate documents this stylesheet cannot reach, and an
 * `.agent-flash` on the iframe element itself would clip the embedded app to
 * the flash's border-radius. The element is transparent and pointer-inert at
 * rest, so a spent glow can simply stay parked in the overlay.
 * @param {ManagedPanel} entry
 * @returns {HTMLElement}
 */
function ensureGlowEl(entry) {
  if (entry.glowEl?.isConnected) return entry.glowEl;
  const el = document.createElement('div');
  el.className = GLOW_CLASS;
  // Inline, so it outranks the flash class's own radius: the glow traces the
  // tile body's rounded corners (see TILE_BODY_RADIUS).
  el.style.borderRadius = TILE_BODY_RADIUS;
  /** @type {HTMLElement} */ (overlayEl).appendChild(el);
  entry.glowEl = el;
  return el;
}

/**
 * Flash the agent-activity glow over a panel's TILE BODY — the visual companion
 * to the rail entry's flash, so an agent action reads on the panel it actually
 * touched and not only on a ~20px rail row.
 *
 * No-ops unless the panel is genuinely on screen: it must be managed, visible,
 * hold a live placeholder, and be the panel that placeholder's group shows.
 * Anything else has no rectangle to glow, and glowing the tile a hidden panel
 * sits behind would attribute the action to the wrong panel.
 *
 * The rectangle is read inside a requestAnimationFrame: a glow commonly follows
 * the activation that created the tile, and dockview's geometry only lands once
 * the layout settles — reading synchronously would measure a zero-sized (or
 * stale) group.
 *
 * FALLBACK (no dockview): there are no tiles, so the single mounted host
 * (#panel-content) is the panel body and takes the flash directly.
 * @param {string} panelId
 */
export function glowPanel(panelId) {
  const api = ensureDock();
  if (!api || !overlayEl) {
    if (fallbackHostEl) flashElement(fallbackHostEl);
    return;
  }
  if (!managed.has(panelId)) return;
  requestAnimationFrame(() => flashTileGlow(panelId));
}

/**
 * Deferred half of glowPanel: re-resolve the panel (the tile may have moved,
 * closed, or lost focus during the frame), copy its group content rectangle
 * onto the glow element with applyGeometry's math, and fire the flash.
 * @param {string} panelId
 */
function flashTileGlow(panelId) {
  const dockApi = getDockApi();
  const entry = managed.get(panelId);
  if (!dockApi || !overlayEl || !entry?.visible) return;
  const panel = dockApi.getPanel(entry.placeholderId);
  const group = panel?.group;
  const content = group?.element?.querySelector('.dv-content-container');
  if (!panel || !content || group.activePanel !== panel) return;
  const base = overlayEl.getBoundingClientRect();
  const r = content.getBoundingClientRect();
  const el = ensureGlowEl(entry);
  el.style.left = Math.round(r.left - base.left) + 'px';
  el.style.top = Math.round(r.top - base.top) + 'px';
  el.style.width = Math.round(r.width) + 'px';
  el.style.height = Math.round(r.height) + 'px';
  flashElement(el);
}

// ---- Not-answering notice ----------------------------------------------------

/** Class of the per-panel notice element; styled in css/terminal.css. */
const NOTICE_CLASS = 'tile-notice';

/** @type {string | null} the panel the fallback host is showing */
let fallbackShownId = null;
/** @type {HTMLElement | null} the fallback host's one notice */
let fallbackNoticeEl = null;

/**
 * A bare notice element: a polite live region whose text child carries the
 * message. Inline pointer-events:none, as the overlay layer itself states it,
 * so the page underneath stays scrollable and clickable.
 * @returns {HTMLElement}
 */
function buildNoticeEl() {
  const el = document.createElement('div');
  el.className = NOTICE_CLASS;
  el.setAttribute('role', 'status');
  el.setAttribute('aria-live', 'polite');
  el.style.pointerEvents = 'none';
  el.style.position = 'absolute';
  const text = document.createElement('span');
  text.className = 'tile-notice-text';
  el.appendChild(text);
  return el;
}

/**
 * The notice element for a managed panel, created on first use under `parent`
 * and reused after. It is a sibling of the overlay iframe rather than anything
 * inside it: the iframe is another document this stylesheet cannot reach.
 * @param {string} panelId
 * @param {ManagedPanel} entry
 * @param {HTMLElement} parent
 * @returns {HTMLElement}
 */
function ensureNoticeEl(panelId, entry, parent) {
  if (entry.noticeEl?.isConnected && entry.noticeEl.parentElement === parent) {
    return entry.noticeEl;
  }
  const el = buildNoticeEl();
  el.dataset.panel = panelId;
  el.style.borderRadius = TILE_BODY_RADIUS;
  el.style.overflow = 'hidden';
  parent.appendChild(el);
  entry.noticeEl = el;
  return el;
}

/**
 * Write a panel's reachability onto a notice element: the time and message
 * while it is not answering, empty (but still laid out) while it answers.
 * @param {HTMLElement} el
 * @param {string | null} panelId
 */
function paintNotice(el, panelId) {
  const since = panelId === null ? undefined : notAnsweringSince.get(panelId);
  const text = /** @type {HTMLElement} */ (el.querySelector('.tile-notice-text'));
  if (since !== undefined) {
    el.dataset.unreachable = '';
    text.textContent = `Backend not answering since ${since} — this view may be out of date.`;
  } else {
    delete el.dataset.unreachable;
    text.textContent = '';
  }
}

/**
 * The fallback host's one notice, rebuilt and prepended whenever it is no
 * longer in the host (the empty state wipes the host's children).
 * @returns {HTMLElement}
 */
function ensureFallbackNotice() {
  const host = /** @type {HTMLElement} */ (fallbackHostEl);
  if (fallbackNoticeEl?.isConnected && fallbackNoticeEl.parentElement === host) {
    return fallbackNoticeEl;
  }
  const el = buildNoticeEl();
  host.prepend(el);
  fallbackNoticeEl = el;
  return el;
}

/** Paint the fallback host's notice for the panel the host is showing. */
function paintFallbackNotice() {
  if (!fallbackHostEl) return;
  const el = ensureFallbackNotice();
  el.dataset.panel = fallbackShownId ?? '';
  paintNotice(el, fallbackShownId);
}

/**
 * Fallback mode: the host stops showing `panelId`, so its notice no longer
 * speaks for it.
 * @param {string} panelId
 */
function clearFallbackShown(panelId) {
  if (fallbackShownId !== panelId) return;
  fallbackShownId = null;
  paintFallbackNotice();
}

// ---- Focus exit -------------------------------------------------------------

/**
 * Where a panel sits in the host's TAB ORDER, for a panel that hands keyboard
 * focus back to the host (`canopy-host-focus-exit`, handled in app.js).
 *
 * An overlay iframe lives in the overlay layer, whose place in the document has
 * nothing to do with where its tile is drawn, so the iframe itself cannot say
 * what comes before or after the panel. Its placeholder can: the placeholder
 * group's content container sits in the dock exactly where the tile is, after
 * that tile's header bar. That container is the anchor in dock mode. In
 * fallback mount the iframe is in the panel host in normal flow and is its own
 * anchor.
 *
 * `skip` is true for anything in the overlay layer, which a host walk must
 * step over: other panels' overlay iframes are tabbable in the DOM but drawn
 * somewhere else, so stopping on one would send focus to another tile. (In
 * fallback mount the other panels' iframes are display:none, so not tabbable.)
 *
 * @param {MessageEventSource | null} source  the `event.source` of the message
 * @returns {{ anchor: Element, skip: (el: Element) => boolean } | null}
 *   null when the source is not a managed panel's frame, or the panel has no
 *   live placeholder to anchor on
 */
export function panelFocusAnchor(source) {
  const entry = source && [...managed.values()].find((e) => e.iframe.contentWindow === source);
  if (!entry) return null;
  const skip = (/** @type {Element} */ el) => !!overlayEl?.contains(el);
  if (getDockApi() && overlayEl && entry.iframe.parentElement === overlayEl) {
    const group = getDockApi().getPanel(entry.placeholderId)?.group;
    const anchor = group?.element?.querySelector('.dv-content-container') ?? group?.element;
    return anchor ? { anchor, skip } : null;
  }
  return entry.iframe.isConnected ? { anchor: entry.iframe, skip } : null;
}
