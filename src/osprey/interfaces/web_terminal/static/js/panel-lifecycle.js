// @ts-check
/* OSPREY Web Terminal — Panel Lifecycle
 *
 * HOW a panel comes to exist and stay alive. Extracted from panel-manager.js,
 * which owns the core of the panel state machine (tile occupancy, iframe
 * lifecycle, the active-panel policy, the command-palette accessors); the
 * server→client half — the /api/files/events SSE subscription and its frame
 * dispatcher — is panel-sse.js, which calls back into this module's addPanel
 * and ensureRailMembership. This module owns the birth-and-liveness half of
 * it — a panel's cold state, the rail render and membership entries it
 * appears through, its config fetch, and the health polling that decides
 * when it may fill a tile.
 *
 * Leaf modules are imported directly; everything that lives in panel-manager's
 * private state arrives as injected deps registered once at init
 * ({@link initPanelLifecycle}) — the same seam pattern panel-placement.js and
 * panel-menu-policy.js use. That injection is what keeps this module out of an
 * import cycle: panel-manager and panel-sse import it, never the reverse. It
 * also keeps this module unit-testable through panel-manager's and
 * panel-sse's own surfaces, since it holds no panel state of its own.
 */

import { fetchJSON } from './api.js';
import { setKnownServicePanels } from './dock-iframe.js';
import { PANELS, TERMINAL_RAIL_ID, TERMINAL_RAIL_LABEL } from './panel-catalog.js';
import { startHealthPolling as startPolling } from './panel-health.js';
import { railOptions, RAIL_MENU_HINT } from './panel-menu-policy.js';
import { createRail, addEntry, setActive, setEntryEnabled } from './panel-rail.js';
import { updateStatusBar } from './panel-status-bar.js';

/** @typedef {import('./panel-catalog.js').Panel} Panel */
/** @typedef {import('./panel-manager.js').PanelState} PanelState */
/** @typedef {import('./panel-manager.js').PanelRegisterEvent} PanelRegisterEvent */

/**
 * The panel-manager state this module reads and mutates. Containers arrive by
 * live reference (they are never reassigned); reassigned scalars arrive as
 * getters, so a value read here is always the one panel-manager holds now.
 * @typedef {object} LifecycleDeps
 * @property {Record<string, PanelState>} panelState - per-panel state, keyed by panel id
 * @property {Set<string>} memberSet - the server-owned rail membership set
 * @property {() => HTMLElement} getRailEl - the rail nav the entries live in
 * @property {() => string | null} getActive - the locally surfaced panel id
 * @property {() => void} ensureActive - give the empty workspace slot to the best panel available
 */

/** @type {LifecycleDeps | null} */
let deps = null;

/**
 * Register the panel-manager state every function here works through. Called
 * once from initPanelManager, before the rail is rendered.
 * @param {LifecycleDeps} lifecycleDeps
 */
export function initPanelLifecycle(lifecycleDeps) {
  deps = lifecycleDeps;
}

/**
 * The registered deps, or a hard failure. Lifecycle work before init is a
 * wiring bug, not a runtime condition to degrade around — panel-manager
 * registers the deps in the same init that mounts the rail.
 * @returns {LifecycleDeps}
 */
function ctx() {
  if (!deps) throw new Error('panel-lifecycle: initPanelLifecycle() has not run');
  return deps;
}

/** A panel's cold state — shared by the init loop and runtime addPanel().
 *  @returns {PanelState} */
export function freshPanelState() {
  return { url: null, healthy: false, iframe: null, pollTimer: null, polling: false, configLoaded: false };
}

/**
 * Destructive full render of the rail: the terminal entry first (the session
 * tile is the workspace's anchor), then every MEMBER service panel in PANELS
 * order — non-members have no entry at all. The terminal entry is enabled
 * after the render (it never health-polls, so it would stay disabled).
 */
export function renderRail() {
  const railEl = ctx().getRailEl();
  createRail(
    railEl,
    [
      { id: TERMINAL_RAIL_ID, label: TERMINAL_RAIL_LABEL },
      ...PANELS.filter((p) => ctx().memberSet.has(p.id)).map(
        (p) => ({ id: p.id, label: p.label, hint: RAIL_MENU_HINT })
      ),
    ],
    railOptions(),
  );
  setEntryEnabled(railEl, TERMINAL_RAIL_ID, true);
}

/**
 * Re-apply a rail entry's live state after it was (re)built cold by addEntry —
 * enabled from the panel's health, and the active accent when this panel is the
 * surfaced one (a "+"-menu reveal activates locally BEFORE the membership echo
 * rebuilds the entry).
 * @param {string} panelId
 */
function applyEntryState(panelId) {
  const c = ctx();
  const ps = c.panelState[panelId];
  if (!ps) return;
  if (ps.healthy) setEntryEnabled(c.getRailEl(), panelId, true);
  if (c.getActive() === panelId) setActive(c.getRailEl(), panelId);
}

/**
 * Give a panel its rail entry as a MEMBER: record the membership and append the
 * entry (membership IS the rail — there is no dimmed in-between state). The
 * entry is built cold, so its live health/active state is re-applied after the
 * add. Idempotent: addEntry no-ops for an id that already has an entry.
 * @param {string} panelId
 */
export function ensureRailMembership(panelId) {
  const c = ctx();
  c.memberSet.add(panelId);
  const spec = PANELS.find((p) => p.id === panelId);
  if (!spec) return;
  addEntry(c.getRailEl(), { id: spec.id, label: spec.label, hint: RAIL_MENU_HINT }, railOptions());
  applyEntryState(panelId);
}

/**
 * Register a runtime panel and append its rail entry without wiping existing ones.
 *
 * spec shape (matches the panel_register SSE broadcast payload):
 *   { id, label, url, healthEndpoint, path }
 *
 * Guard: if panelState[id] already exists (re-register), refresh the url
 * in-place rather than duplicating the entry or state.
 * @param {PanelRegisterEvent} spec
 */
export function addPanel(spec) {
  const { panelState } = ctx();
  if (panelState[spec.id]) {
    // Re-registration: update url so subsequent navigation stays consistent
    if (spec.url) panelState[spec.id].url = spec.url;
    return;
  }

  const normalized = {
    id: spec.id,
    label: spec.label || spec.id.toUpperCase(),
    configEndpoint: null,
    healthEndpoint: spec.healthEndpoint || null,
    statusBarId: null,
    path: spec.path || '/',
  };
  PANELS.push(normalized);
  // Keep the adapter's known-service set current (never orphan a runtime panel).
  setKnownServicePanels(PANELS.map((p) => p.id));

  panelState[spec.id] = freshPanelState();

  // Append exactly one entry. addEntry is non-destructive — never a full
  // re-render — so every live entry keeps its active/disabled/LED state, and it
  // is idempotent by id, which also guards the re-register path.
  addEntry(ctx().getRailEl(), { id: normalized.id, label: normalized.label, hint: RAIL_MENU_HINT }, railOptions());

  // Seed url and health, mirroring the custom-panel block in initPanelManager
  if (spec.url) {
    const ps = panelState[spec.id];
    ps.url = spec.url;
    ps.configLoaded = true;
    if (!spec.healthEndpoint) {
      assumeHealthy(normalized);
    } else {
      startHealthPolling(normalized);
    }
  }
}

// ---- Panel Initialization ----

/** @param {Panel} panel */
export async function initPanel(panel) {
  const c = ctx();
  const state = c.panelState[panel.id];
  // Custom/runtime panels carry no config endpoint; their url arrives via
  // /api/panels. Skip the fetch and leave the panel disabled until then.
  if (!panel.configEndpoint) { state.configLoaded = true; return; }

  try {
    const config = await fetchJSON(panel.configEndpoint);
    // Artifact server returns { url }, ARIEL returns { url, available }
    if (config.url && (config.available === undefined || config.available)) {
      state.url = config.url;
    }
  } catch {
    // Config endpoint not available — panel stays disabled
  } finally {
    state.configLoaded = true;
  }

  if (state.url) {
    // External panels (healthEndpoint === null) skip health polling —
    // mark healthy immediately so the tab is enabled.
    if (panel.healthEndpoint == null) {  // null or undefined → skip polling
      assumeHealthy(panel);
    } else {
      startHealthPolling(panel);
    }
  }
  // Re-evaluate on every settle, including the no-url case: this panel may be
  // the default that another panel's health poll was waiting on.
  c.ensureActive();
}

// ---- Health Polling ----

/**
 * Poll-settle hook handed to panel-health.js's timing machinery: reflect the
 * new health on the status bar, and on the FIRST healthy settle enable the
 * entry and let the shared policy decide whether the newly-healthy panel
 * should take an empty slot. The rail itself shows no per-poll readout — the
 * SYSTEM panel's `web_panels` category is where liveness is reported.
 * @param {Panel} panel
 * @param {boolean} wasHealthy
 */
function onHealthSettled(panel, wasHealthy) {
  const c = ctx();
  updateStatusBar(panel, c.panelState[panel.id]);
  if (c.panelState[panel.id].healthy && !wasHealthy) {
    setEntryEnabled(c.getRailEl(), panel.id, true);
    c.ensureActive();
  }
}

/** @param {Panel} panel  Start panel-health's polling loop with this module's hook. */
export function startHealthPolling(panel) {
  startPolling(panel, ctx().panelState[panel.id], onHealthSettled);
}

// ---- Entry State ----

/**
 * A panel with no health endpoint is assumed permanently healthy: mark it so
 * and enable its rail entry. Consolidates the built-in, custom-config, and
 * runtime-addPanel paths so none can leave an entry inert forever.
 * @param {Panel} panel
 */
export function assumeHealthy(panel) {
  const c = ctx();
  c.panelState[panel.id].healthy = true;
  setEntryEnabled(c.getRailEl(), panel.id, true);
}
