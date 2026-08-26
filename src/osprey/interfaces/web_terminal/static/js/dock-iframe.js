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
import { flashElement } from '/design-system/js/highlight.js';

/**
 * One tracked service panel: its cached iframe (created/owned by panel-manager),
 * the id of the empty dockview placeholder it follows, the tab title, whether it
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
        api.onDidActivePanelChange(trackActiveServiceGroup),
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
 * Toggle the inline pointer shield over every managed overlay iframe. The
 * shared `.main-container.dock-dragging iframe { pointer-events:none }` rule
 * can't win against the inline `pointer-events:auto` these iframes carry, so
 * the toggle must be inline. Exported for rail-drag.js, whose HTML5 drags
 * start outside dockview and therefore outside onDragGesture's coverage.
 * @param {boolean} on
 */
export function setIframePointerShield(on) {
  for (const e of managed.values()) e.iframe.style.pointerEvents = on ? 'none' : 'auto';
}

/**
 * Re-apply dock-workspace.js's pointer shield to the overlay iframes across
 * every dockview-initiated drag gesture (start → any natural terminator).
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
 * @param {any} api
 */
function wireDragShield(api) {
  disposers.push(...onDragGesture(api, {
    onStart: () => setIframePointerShield(true),
    onEnd: () => setIframePointerShield(false),
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
 * The dockview group id that most recently held the focused service placeholder.
 * Tracked on every active-panel change so a placement can target "the service
 * tile the operator last worked in" even while the terminal is focused. May go
 * stale across layout rebuilds — consumers re-validate through the live api.
 * @type {string | null}
 */
let lastServiceGroupId = null;

/** Record the focused service tile's group on every active-panel change. */
function trackActiveServiceGroup() {
  const api = getDockApi();
  const active = api?.activePanel;
  if (typeof active?.id === 'string' && active.id.startsWith(PLACEHOLDER_PREFIX)) {
    lastServiceGroupId = active.group?.id ?? lastServiceGroupId;
  }
}

/** @param {any} group @returns {boolean} whether the group holds a service placeholder */
function groupHasServicePlaceholder(group) {
  return (group?.panels ?? []).some(
    (/** @type {any} */ p) => typeof p?.id === 'string' && p.id.startsWith(PLACEHOLDER_PREFIX),
  );
}

/**
 * The service tile a placement should target: the focused service tile's group,
 * else the last one the operator focused (re-validated — layouts rebuild), else
 * any existing service tile. Null when no service tile exists at all.
 * @param {any} api
 * @returns {any} a dockview group, or null
 */
function targetServiceGroup(api) {
  const active = api.activePanel;
  if (typeof active?.id === 'string' && active.id.startsWith(PLACEHOLDER_PREFIX)
      && active.group) {
    return active.group;
  }
  if (lastServiceGroupId) {
    const group = (api.groups ?? []).find((/** @type {any} */ g) => g?.id === lastServiceGroupId);
    if (group && groupHasServicePlaceholder(group)) return group;
  }
  for (const e of managed.values()) {
    const group = api.getPanel(e.placeholderId)?.group;
    if (group) return group;
  }
  for (const p of (Array.isArray(api.panels) ? api.panels : [])) {
    if (typeof p?.id === 'string' && p.id.startsWith(PLACEHOLDER_PREFIX) && p.group) {
      return p.group;
    }
  }
  return null;
}

/**
 * Ensure the empty placeholder dockview panel for a managed panel exists,
 * enforcing the one-panel-per-tile model. Placement by intent:
 *
 *   'replace' (an activation — rail click, agent focus): the new placeholder
 *   takes over the target service tile and the tile's previous occupant is
 *   evicted (removed + reported through the replaced-panel handler, so the
 *   panel manager can close it server-side). The rail is the tab system; an
 *   activation switches the tile rather than stacking a tab into it.
 *
 *   'beside' (a redock after a layout rebuild): the placeholder docks as a
 *   NEW tile beside the target group — re-materializing several visible
 *   panels must not have them evict one another.
 *
 * With no service tile at all, either intent opens the first one LEFT of the
 * terminal at the classic 60/40 split (or against the grid's left edge when
 * the terminal tile is closed too).
 * @param {any} api
 * @param {ManagedPanel} entry
 * @param {'replace'|'beside'} [intent]
 */
function ensurePlaceholder(api, entry, intent = 'replace') {
  if (api.getPanel(entry.placeholderId)) return;
  /** @type {any} */
  const opts = { id: entry.placeholderId, component: PLACEHOLDER_COMPONENT, title: entry.title };
  const target = targetServiceGroup(api);
  if (target) {
    opts.position = intent === 'replace'
      ? { referenceGroup: target, direction: 'within' }
      : { referenceGroup: target, direction: 'right' };
  } else if (api.getPanel(TERMINAL_PANEL_ID)) {
    opts.position = { referencePanel: TERMINAL_PANEL_ID, direction: 'left' };
    opts.initialWidth = defaultServiceWidth();
  } else {
    // No usable reference at all (terminal tile closed too): split against the
    // grid edge rather than stacking into the active group — one panel per
    // tile is an invariant, addPanel's default would violate it.
    opts.position = { direction: 'left' };
  }
  try {
    api.addPanel(opts);
  } catch (err) {
    console.error('dock-iframe: addPanel failed for', entry.placeholderId, err);
    return;
  }
  if (intent === 'replace' && target) evictReplacedOccupants(api, entry.placeholderId);
}

/**
 * Optional observer of replace-evictions, called with the evicted panel's
 * service id. Eviction is a purely LOCAL vacate under the launcher-rail model
 * (the evicted panel keeps its rail membership; the server is not told), so
 * nothing registers this in production — it remains as a test/diagnostic seam.
 * @type {((serviceId: string) => void) | null}
 */
let replacedHandler = null;

/** @param {((serviceId: string) => void) | null} fn */
export function setReplacedPanelHandler(fn) {
  replacedHandler = fn;
}

/**
 * Remove every OTHER service placeholder from the group that just received a
 * 'replace' placement — the tile's previous occupant(s). Native panels (the
 * terminal) are never evicted. The evicted iframe is concealed but kept cached
 * so a later re-activation restores its state; the eviction changes no rail or
 * server state.
 * @param {any} api
 * @param {string} keptPlaceholderId
 */
function evictReplacedOccupants(api, keptPlaceholderId) {
  const group = api.getPanel(keptPlaceholderId)?.group;
  if (!group) return;
  const others = (group.panels ?? []).filter(
    (/** @type {any} */ p) =>
      typeof p?.id === 'string' && p.id.startsWith(PLACEHOLDER_PREFIX) && p.id !== keptPlaceholderId,
  );
  for (const panel of others) {
    const serviceId = panel.id.slice(PLACEHOLDER_PREFIX.length);
    try {
      api.removePanel(panel);
    } catch (err) {
      console.error('dock-iframe: replace eviction failed for', panel.id, err);
      continue;
    }
    const evicted = managed.get(serviceId);
    if (evicted) {
      evicted.visible = false;
      evicted.iframe.style.display = 'none';
    }
    replacedHandler?.(serviceId);
  }
}

/**
 * Re-create the placeholder for every visible managed panel, drop orphaned
 * placeholders, and re-sync geometry. Registered with dock-workspace's
 * setServiceRedock, which fires after every layout apply (default rebuild,
 * stored-layout restore, mode-flip restore). Re-docking uses 'beside'
 * placement — materializing several visible panels into tiles must not have
 * them evict one another.
 */
function redockVisiblePanels() {
  const api = getDockApi();
  if (!api || !overlayEl) return;
  for (const entry of managed.values()) {
    if (entry.visible) ensurePlaceholder(api, entry, 'beside');
  }
  pruneOrphanPlaceholders(api);
  syncGeometry();
}

/** The service-id universe panel-manager knows, or null before it has loaded.
 *  @type {Set<string> | null} */
let knownServiceIds = null;

/** The server-visible service ids — RAIL MEMBERSHIP: a LIVE reference to
 *  panel-manager's own set (kept current on every SSE visibility change), or
 *  null before the registry has loaded. Used to prune restored placeholders
 *  whose panel is no longer a rail member: under one-panel-per-tile such a
 *  placeholder would be a whole ghost tile, not just a hidden tab.
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
    pruneOrphanPlaceholders(api);
    syncGeometry();
  }
}

/**
 * Remove placeholder panels that must not hold a tile: services neither known
 * to panel-manager nor managed here (removed from the deployment), and — once
 * the membership set is seeded — services no longer in the rail that are not
 * currently being shown locally (a restored placeholder for a removed panel is
 * a ghost tile under the one-panel-per-tile model). The `managed` visible flag
 * covers the window where a just-added panel's membership POST has not echoed
 * back into the server set yet. No-op until setKnownServicePanels has run —
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
    const unknown = !knownServiceIds.has(sid) && !managed.has(sid);
    const closed = !!serverVisibleIds
      && !serverVisibleIds.has(sid)
      && !managed.get(sid)?.visible;
    if (!unknown && !closed) continue;
    try {
      api.removePanel(panel);
    } catch (err) {
      console.error('dock-iframe: orphan placeholder removal failed for', id, err);
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
    const content = api.getPanel(entry.placeholderId)?.group?.element
      ?.querySelector('.dv-content-container');
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
 * iframe (inline geometry, relative to the overlay's own origin). An iframe is
 * shown only when its panel is visible AND its placeholder is the active tab in
 * its group; anything hidden behind another tab, closed, or lacking a live
 * placeholder is display:none.
 */
function applyGeometry() {
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
 * placeholder dock panel so no empty tile lingers. The iframe element itself is
 * kept (cached) so a later show/focus re-reveals it with its state intact.
 * Also handles panels this adapter never adopted (no managed entry) — a
 * restored placeholder for a panel closed on another client still needs its
 * tile removed. Consumed by panel-manager's, panel-sse's and
 * panel-placement's hide paths.
 * @param {string} panelId
 */
export function hidePanel(panelId) {
  const entry = managed.get(panelId);
  if (entry) {
    entry.visible = false;
    entry.iframe.style.display = 'none';
  }
  const api = ensureDock();
  if (api && overlayEl) {
    const placeholderId = entry?.placeholderId ?? placeholderIdFor(panelId);
    const panel = api.getPanel(placeholderId);
    if (panel) {
      try {
        api.removePanel(panel);
      } catch (err) {
        console.error('dock-iframe: removePanel failed for', placeholderId, err);
      }
    }
    syncGeometry();
  }
}

/**
 * Mark a panel vacated WITHOUT touching its dock placeholder — the complement
 * of hidePanel for the human tile-close gesture, where dockview itself is
 * already removing the placeholder (the close click's own bubble handler).
 * Only the adapter-side state is cleared: the overlay iframe is concealed
 * (cached for a later re-activation) and the entry marked not-visible so
 * redocks stop re-materializing it. Geometry re-syncs on the removal's own
 * onDidLayoutChange, so none is forced here.
 * @param {string} panelId
 */
export function concealPanel(panelId) {
  const entry = managed.get(panelId);
  if (!entry) return;
  entry.visible = false;
  entry.iframe.style.display = 'none';
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
  /** @type {HTMLElement} */ (overlayEl).appendChild(el);
  entry.glowEl = el;
  return el;
}

/**
 * Flash the agent-activity glow over a panel's TILE BODY — the visual companion
 * to the rail entry's flash, so an agent action reads on the panel it actually
 * touched and not only on a ~20px rail tab.
 *
 * No-ops unless the panel is genuinely on screen: it must be managed, visible,
 * hold a live placeholder, and be the ACTIVE tab in that placeholder's group.
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
