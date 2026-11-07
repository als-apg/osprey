// @ts-check
/* OSPREY Web Terminal — Panel Lifecycle
 *
 * HOW a panel comes to exist and stay alive. Extracted from panel-manager.js,
 * which owns the core of the panel state machine (tile occupancy, iframe
 * lifecycle, the active-panel policy, the command-palette accessors); the
 * server→client half — the /api/files/events SSE subscription and its frame
 * dispatcher — is panel-sse.js, which calls back into this module's addPanel
 * and ensureRailMembership. This module owns the birth-and-liveness half of
 * it — a panel's cold state, the sidebar rows and membership entries it
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

import { apiRequest, fetchJSON } from './api.js';
import { setKnownServicePanels, setPanelReachable } from './dock-iframe.js';
import { setPanelServed } from './panel-availability.js';
import {
  PANELS,
  TERMINAL_RAIL_ID,
  TERMINAL_RAIL_LABEL,
  panelIconName,
  titleCase,
} from './panel-catalog.js';
import { startHealthPolling as startPolling, stopHealthPolling } from './panel-health.js';
import { buildEmbedSrc, isUrlWithinMount } from './panel-iframe-sync.js';
import { railOptions, RAIL_MENU_HINT } from './panel-menu-policy.js';
import { isDetached } from './dock-apply.js';
import {
  clockTime,
  createRail,
  addEntry,
  placeEntry,
  setActive,
  setEntryEnabled,
  setEntryDetached,
  setEntryReachable,
  setEntryStatus,
} from './panel-rail.js';
import { paintAgentRow } from './sidebar.js';

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
 * @property {(id: string, options?: {userInitiated?: boolean}) => void} activate - surface a healthy panel
 * @property {(id: string) => void} navigatePending - load the address a panel was pointed at while it was not answering
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
  return {
    url: null,
    healthy: false,
    iframe: null,
    pollTimer: null,
    polling: false,
    configLoaded: false,
    settled: false,
    misses: 0,
    missSince: null,
  };
}

/**
 * Set where a panel is served, or `null` for nowhere: the one write of
 * `PanelState.url`, so the page's followers of it (panel-availability.js) hear
 * every change.
 * @param {string} panelId
 * @param {string | null} url
 */
export function publishPanelUrl(panelId, url) {
  ctx().panelState[panelId].url = url;
  setPanelServed(panelId, url !== null);
}

/** The load-time flags buildEmbedSrc stamps on a panel's address, which are
 *  not part of the page it shows. */
const EMBED_FLAGS = ['embedded', 'theme'];

/**
 * The page a panel's frame shows, as a root-relative address inside its own
 * mount, or null when the hub cannot read it: no frame yet, or an opaque
 * (sandboxed) or cross-origin panel, whose location the browser keeps from
 * the hub. The embed flags are left off, so the address is the page alone.
 * @param {string} panelId
 * @returns {string | null}
 */
export function getPanelPage(panelId) {
  const state = ctx().panelState[panelId];
  if (!state?.iframe || !state.url) return null;
  let url;
  try {
    url = new URL(String(state.iframe.contentWindow?.location.href));
  } catch {
    return null;
  }
  if (url.origin !== window.location.origin) return null;
  for (const flag of EMBED_FLAGS) url.searchParams.delete(flag);
  const page = url.pathname + url.search + url.hash;
  return isUrlWithinMount(page, state.url) ? page : null;
}

/**
 * Bring a panel back to a page it was on (a session's layout,
 * session-layout.js) without surfacing it: nothing when it shows that page
 * already, or for a panel not served or a page outside its mount. A panel not
 * framed yet opens there when it is.
 * @param {string} panelId
 * @param {string} page  root-relative, as {@link getPanelPage} reads it
 */
export function restorePanelPage(panelId, page) {
  const state = ctx().panelState[panelId];
  if (!state?.url || !isUrlWithinMount(page, state.url)) return;
  if (getPanelPage(panelId) === page) return;
  state.pendingUrl = page;
  ctx().navigatePending(panelId);
}

/**
 * A catalog panel as its sidebar row draws it: the label in title case, the
 * panel's glyph, and the context-menu hint.
 * @param {{id: string, label: string}} panel
 * @returns {import('./panel-rail.js').RailPanel}
 */
function railPanel(panel) {
  return {
    id: panel.id,
    label: titleCase(panel.label),
    icon: panelIconName(panel.id),
    hint: RAIL_MENU_HINT,
  };
}

/**
 * Destructive full render of the sidebar rows: the Agent row first (the
 * pane is the workspace's anchor), then every MEMBER service panel in the
 * server's rail order, each in the operator's group for it — non-members
 * have no row at all. The Agent row is enabled after the render (it never
 * health-polls, so it would stay disabled), and a row whose panel is popped
 * out gets its mark back, since the render builds rows cold.
 */
export function renderRail() {
  const railEl = ctx().getRailEl();
  createRail(
    railEl,
    [
      { id: TERMINAL_RAIL_ID, label: TERMINAL_RAIL_LABEL, icon: panelIconName(TERMINAL_RAIL_ID) },
      ...[...ctx().memberSet]
        .map((id) => PANELS.find((p) => p.id === id))
        .filter((p) => p !== undefined)
        .map(railPanel),
    ],
    railOptions()
  );
  setEntryEnabled(railEl, TERMINAL_RAIL_ID, true);
  for (const id of [TERMINAL_RAIL_ID, ...ctx().memberSet]) {
    if (isDetached(id)) setEntryDetached(railEl, id, true);
  }
}

/**
 * A pane arrived in its window or came back from it: mark its row. The Sessions
 * row's own painter has the last word on that row's tips.
 * @param {string} id
 * @param {boolean} detached
 */
export function markDetachedRow(id, detached) {
  setEntryDetached(ctx().getRailEl(), id, detached);
  if (id === TERMINAL_RAIL_ID) paintAgentRow();
}

/**
 * Put the rail in the server's order — the body behind the `rail_order` frame
 * and the reconnect resync. Membership is untouched: ids in `order` that are not
 * members are skipped, and a member missing from `order` keeps its place ahead
 * of the ordered ones. The member set is re-sequenced too, so a later full
 * render draws the same rail. Idempotent.
 * @param {string[]} order
 */
export function applyRailOrder(order) {
  const c = ctx();
  const ordered = order.filter((id) => c.memberSet.has(id));
  const rest = [...c.memberSet].filter((id) => !ordered.includes(id));
  c.memberSet.clear();
  for (const id of [...rest, ...ordered]) c.memberSet.add(id);
  // Each row moves within its own group (the Agent row stays first); moving
  // every ordered row to the end in turn puts each group in `order`.
  for (const id of ordered) placeEntry(c.getRailEl(), id, null);
}

/**
 * Re-apply a sidebar row's live state after addEntry (re)built or re-showed it
 * — enabled from the panel's health, the active accent when this panel is
 * the surfaced one (a "+"-menu reveal activates locally BEFORE the membership
 * echo adds the row), and the detached mark while its window is open.
 * @param {string} panelId
 */
function applyEntryState(panelId) {
  const c = ctx();
  const ps = c.panelState[panelId];
  if (!ps) return;
  if (ps.healthy || ps.misses > 0) setEntryEnabled(c.getRailEl(), panelId, true);
  if (ps.misses >= MISSES_BEFORE_UNREACHABLE) {
    setEntryReachable(c.getRailEl(), panelId, false, ps.missSince ?? undefined);
  }
  if (ps.failedMessage)
    setEntryStatus(c.getRailEl(), panelId, { failed: true, message: ps.failedMessage });
  if (c.getActive() === panelId) setActive(c.getRailEl(), panelId);
  if (isDetached(panelId)) setEntryDetached(c.getRailEl(), panelId, true);
}

/**
 * Give a panel its sidebar row as a MEMBER: record the membership and add the
 * row (membership IS the rail — there is no dimmed in-between state). A new
 * row is built cold, so its live health/active state is re-applied after the
 * add. Idempotent: addEntry no-ops for an id that already has a row.
 * @param {string} panelId
 */
export function ensureRailMembership(panelId) {
  const c = ctx();
  c.memberSet.add(panelId);
  const spec = PANELS.find((p) => p.id === panelId);
  if (!spec) return;
  addEntry(c.getRailEl(), railPanel(spec), railOptions());
  applyEntryState(panelId);
}

/**
 * Register a runtime panel and add its sidebar row without wiping existing ones.
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
    if (spec.url) publishPanelUrl(spec.id, spec.url);
    return;
  }

  const normalized = {
    id: spec.id,
    label: titleCase(spec.label || spec.id.toUpperCase()),
    configEndpoint: null,
    healthEndpoint: spec.healthEndpoint || null,
    path: spec.path || '/',
  };
  PANELS.push(normalized);
  // Keep the adapter's known-service set current (never orphan a runtime panel).
  setKnownServicePanels(PANELS.map((p) => p.id));

  panelState[spec.id] = freshPanelState();

  // Add exactly one row. addEntry is non-destructive — never a full re-render
  // — so every live row keeps its active/disabled state, and it is idempotent
  // by id, which also guards the re-register path.
  addEntry(ctx().getRailEl(), railPanel(normalized), railOptions());

  // Seed url and health, mirroring the custom-panel block in initPanelManager
  if (spec.url) {
    publishPanelUrl(spec.id, spec.url);
    panelState[spec.id].configLoaded = true;
    if (!spec.healthEndpoint) {
      assumeHealthy(normalized);
    } else {
      startHealthPolling(normalized);
    }
  }
}

// ---- Panel Initialization ----

/**
 * How long one config read may wait for an answer. A browser queues requests
 * behind its per-host connection cap, and a read stuck in that queue would
 * otherwise hold the panel's entry disabled for the life of the page.
 */
const CONFIG_READ_TIMEOUT_MS = 5000;

/** The wait before retrying an unanswered config read; doubles up to the cap. */
const CONFIG_RETRY_MS = 1000;
const CONFIG_RETRY_MAX_MS = 15000;

/**
 * Read a panel's config endpoint once.
 * @param {string} endpoint
 * @returns {Promise<any>} the config, or `undefined` when the read failed or
 *   timed out (a 200 answer, even one with no url, is never `undefined`)
 */
async function readConfig(endpoint) {
  try {
    return await fetchJSON(endpoint, { signal: AbortSignal.timeout(CONFIG_READ_TIMEOUT_MS) });
  } catch {
    return undefined;
  }
}

/**
 * Read the panel's config and act on it. A read that fails or times out is
 * retried with backoff until one is answered, with the entry disabled in the
 * meantime, so a request that was stalled or refused at page load does not
 * leave the panel dead until the next reload. An answer is final, including
 * one that says the panel has no url.
 * @param {Panel} panel
 */
export async function initPanel(panel) {
  const c = ctx();
  const state = c.panelState[panel.id];
  // Custom/runtime panels carry no config endpoint; their url arrives via
  // /api/panels. Skip the fetch and leave the panel disabled until then.
  if (!panel.configEndpoint) {
    state.configLoaded = true;
    return;
  }

  let delay = CONFIG_RETRY_MS;
  let config = await readConfig(panel.configEndpoint);
  while (config === undefined) {
    // Settled for the active-panel policy, which waits on every panel's first
    // read: this one is unavailable until a retry is answered.
    if (!state.configLoaded) {
      state.configLoaded = true;
      state.settled = true;
      c.ensureActive();
    }
    await new Promise((resolve) => setTimeout(resolve, delay));
    delay = Math.min(delay * 2, CONFIG_RETRY_MAX_MS);
    // A url pushed by the server in the meantime (addPanel) supersedes the read.
    if (state.url) return;
    config = await readConfig(panel.configEndpoint);
  }
  const url = servedUrl(config);
  if (url) publishPanelUrl(panel.id, url);
  state.configLoaded = true;

  // A sidecar reports its start outcome with its config: a failed one gets
  // its clickable failed entry, a starting one is followed until it settles.
  if (config?.state === 'failed') showFailed(panel, config.message);
  else if (config?.state === 'starting') followStart(panel);
  else if (state.url) followLiveness(panel);
  // No address and no start to follow: there is nothing to ask, so the first
  // verdict is in.
  else state.settled = true;
  // Re-evaluate on every settle, including the no-url case: this panel may be
  // the default that another panel's health poll was waiting on.
  c.ensureActive();
}

/**
 * The url a config answer says the panel is served at. The artifact server
 * returns { url }, ARIEL returns { url, available }.
 * @param {any} config
 * @returns {string | null}
 */
function servedUrl(config) {
  return config?.url && (config.available === undefined || config.available) ? config.url : null;
}

/**
 * Start tracking a served panel's liveness. External panels (healthEndpoint
 * null or undefined) skip health polling: mark them healthy at once so the tab
 * is enabled.
 * @param {Panel} panel
 */
function followLiveness(panel) {
  if (panel.healthEndpoint == null) assumeHealthy(panel);
  else startHealthPolling(panel);
}

/**
 * Read a panel's config again once the hub may have restarted (the event
 * stream reopened), since the answer {@link initPanel} took as final holds only
 * for that hub. A panel that gained a url starts up; one that lost it stops:
 * its entry goes back to `.disabled` and a tile it holds carries the notice. A
 * sidecar is skipped: its own start and health paths already follow it.
 * @param {Panel} panel
 */
export async function rereadPanelConfig(panel) {
  const c = ctx();
  const state = c.panelState[panel.id];
  if (!state || !panel.configEndpoint || panel.startEndpoint) return;
  const config = await readConfig(panel.configEndpoint);
  if (config === undefined) return; // no answer: the next reopen asks again
  const url = servedUrl(config);
  if (Boolean(url) === Boolean(state.url)) return;
  if (url) {
    publishPanelUrl(panel.id, url);
    setPanelReachable(panel.id, true);
    if (state.iframe) state.iframe.src = buildEmbedSrc(url);
    followLiveness(panel);
    c.ensureActive();
    return;
  }
  stopHealthPolling(state);
  publishPanelUrl(panel.id, null);
  state.healthy = false;
  if (state.misses >= MISSES_BEFORE_UNREACHABLE) setEntryReachable(c.getRailEl(), panel.id, true);
  state.misses = 0;
  state.missSince = null;
  setEntryEnabled(c.getRailEl(), panel.id, false);
  setPanelReachable(panel.id, false, clockTime(Date.now()));
}

// ---- Health Polling ----

/**
 * Consecutive unanswered polls, counted only after a panel has answered once,
 * before its rail entry is marked unreachable. At the 10 s maintenance cadence
 * two misses mean ~20 s of silence, so one slow answer never flickers the rail.
 */
const MISSES_BEFORE_UNREACHABLE = 2;

/**
 * Poll-settle hook handed to panel-health.js's timing machinery. The rail shows
 * no per-poll readout — coarse reachability only:
 *
 *   - the FIRST healthy settle enables the entry and lets the shared policy
 *     decide whether the newly-healthy panel should take an empty slot;
 *   - a panel that answered before and then misses MISSES_BEFORE_UNREACHABLE
 *     polls in a row is marked unreachable on both of its surfaces: the rail
 *     entry (dimmed, still clickable, tooltip "not answering since HH:MM") and
 *     its tile (a notice over the body, the same time);
 *   - the next healthy settle clears both;
 *   - a sidecar that answered before and then misses a poll is checked once
 *     against its config endpoint, and one that reports `failed` shows its
 *     failed entry instead.
 *
 * A panel that never answered keeps its `.disabled` boot state and is never
 * counted, and its tile, which never opens, carries no notice. Liveness
 * detail lives in the Health panel's `web_panels` category.
 * @param {Panel} panel
 * @param {boolean} wasHealthy
 */
function onHealthSettled(panel, wasHealthy) {
  const c = ctx();
  const state = c.panelState[panel.id];
  // The first answer, healthy or not, is the verdict the landing waits on.
  state.settled = true;
  if (state.healthy) {
    if (state.misses >= MISSES_BEFORE_UNREACHABLE) setEntryReachable(c.getRailEl(), panel.id, true);
    // Unconditional: showFailed resets misses while the tile keeps its notice,
    // so a retried sidecar's first answer must still clear the tile.
    setPanelReachable(panel.id, true);
    state.misses = 0;
    state.missSince = null;
    if (!wasHealthy) {
      setEntryEnabled(c.getRailEl(), panel.id, true);
      c.navigatePending(panel.id);
      if (state.activateOnHealthy) {
        // The operator asked for this panel by clicking its failed entry.
        state.activateOnHealthy = false;
        c.activate(panel.id, { userInitiated: true });
      } else c.ensureActive();
    }
    return;
  }
  if (wasHealthy && panel.startEndpoint && panel.configEndpoint) {
    // A sidecar that stopped answering may have died: ask the terminal once,
    // so a live page shows the failed entry without a reload.
    fetchJSON(panel.configEndpoint)
      .then((config) => {
        if (config?.state !== 'failed') return;
        stopHealthPolling(state);
        publishPanelUrl(panel.id, null);
        state.healthy = false;
        showFailed(panel, config.message);
      })
      .catch(() => {});
  }
  if (!wasHealthy && state.misses === 0) return; // never answered: stays .disabled
  state.misses += 1;
  if (state.missSince === null) state.missSince = Date.now();
  if (state.misses === MISSES_BEFORE_UNREACHABLE) {
    setEntryReachable(c.getRailEl(), panel.id, false, state.missSince);
    setPanelReachable(panel.id, false, clockTime(state.missSince));
  }
}

/** @param {Panel} panel  Start panel-health's polling loop with this module's hook. */
export function startHealthPolling(panel) {
  startPolling(panel, ctx().panelState[panel.id], onHealthSettled);
}

// ---- Sidecar Start Recovery ----

/**
 * A sidecar's start outcome as this page follows it: `starting` while the
 * config says so, `failed` with the server's sentence, `running` once the url
 * is published again.
 * @typedef {{ state: 'starting' | 'failed' | 'running', message: string | null }} StartState
 */

/** @type {Set<(panelId: string, start: StartState) => void>} */
const startFollowers = new Set();

/**
 * Hear every start outcome of every sidecar panel from now on (the panel's
 * tile draws its starting and failed states from it).
 * @param {(panelId: string, start: StartState) => void} fn
 * @returns {() => void} stops following
 */
export function onPanelStart(fn) {
  startFollowers.add(fn);
  return () => startFollowers.delete(fn);
}

/**
 * @param {string} panelId
 * @param {StartState['state']} state
 * @param {string | null} message
 */
function tellStart(panelId, state, message) {
  for (const fn of startFollowers) fn(panelId, { state, message });
}

/**
 * Show a sidecar's failed entry: dimmed, clickable, the server's sentence as
 * its tooltip. The message is kept on the state so a rebuilt entry shows it too.
 * @param {Panel} panel
 * @param {string | null} message
 */
function showFailed(panel, message) {
  const c = ctx();
  const state = c.panelState[panel.id];
  state.failedMessage = message;
  state.settled = true;
  // The failed entry names the reason; an unreachable notice would hide it.
  if (state.misses >= MISSES_BEFORE_UNREACHABLE) setEntryReachable(c.getRailEl(), panel.id, true);
  state.misses = 0;
  state.missSince = null;
  setEntryStatus(c.getRailEl(), panel.id, { failed: true, message });
  tellStart(panel.id, 'failed', message);
}

/**
 * Start a failed sidecar again: POST its start endpoint, show the server's
 * "is starting" sentence on a disabled entry, then follow the start until it
 * settles. The click on a failed entry lands here; the entry is no longer
 * `.failed` once it has, so a second click sends nothing.
 * @param {Panel} panel
 */
export async function retryPanelStart(panel) {
  const c = ctx();
  const state = c.panelState[panel.id];
  if (!panel.startEndpoint || !state) return;
  const lastMessage = state.failedMessage ?? null;
  state.failedMessage = null;
  setEntryStatus(c.getRailEl(), panel.id, { failed: false, message: null });
  setEntryEnabled(c.getRailEl(), panel.id, false);
  let answer;
  try {
    answer = await apiRequest(panel.startEndpoint, { errorPrefix: 'Could not start the panel' });
  } catch {
    showFailed(panel, lastMessage);
    return;
  }
  state.activateOnHealthy = true;
  settleStart(panel, answer, lastMessage);
}

/**
 * Follow a start already in flight (a page loaded while the sidecar started).
 * @param {Panel} panel
 */
function followStart(panel) {
  settleStart(panel, { state: 'starting', message: null }, null);
}

/**
 * Restart a running sidecar: POST `endpoint` (the server stops it and starts
 * a fresh one), stop polling the old server, then follow the new start as a
 * retry does. A refused POST throws and leaves the panel as it was.
 * @param {Panel} panel
 * @param {string} endpoint
 */
export async function restartPanelServer(panel, endpoint) {
  const c = ctx();
  const state = c.panelState[panel.id];
  if (!state) return;
  const answer = await apiRequest(endpoint, { errorPrefix: 'Could not restart the panel' });
  stopHealthPolling(state);
  // Not healthy until the new server answers: its first answer enables the
  // entry again (onHealthSettled).
  state.healthy = false;
  state.failedMessage = null;
  setEntryEnabled(c.getRailEl(), panel.id, false);
  settleStart(panel, answer, null);
}

/**
 * Apply one config answer during a start, polling the config endpoint once a
 * second while it still says `starting`. `running` publishes the url, reloads
 * an open iframe (same proxy path, new backend) and starts health polling;
 * `failed` shows the new reason; a fetch error restores the last known one.
 * @param {Panel} panel
 * @param {any} answer
 * @param {string | null} lastMessage
 */
function settleStart(panel, answer, lastMessage) {
  const c = ctx();
  const state = c.panelState[panel.id];
  if (answer?.state === 'starting') {
    setEntryStatus(c.getRailEl(), panel.id, { failed: false, message: answer.message ?? null });
    tellStart(panel.id, 'starting', answer.message ?? null);
    setTimeout(() => {
      if (!panel.configEndpoint) return;
      fetchJSON(panel.configEndpoint).then(
        (next) => settleStart(panel, next, lastMessage),
        () => showFailed(panel, lastMessage)
      );
    }, 1000);
    return;
  }
  if (answer?.state === 'failed') {
    showFailed(panel, answer.message ?? lastMessage);
    return;
  }
  setEntryStatus(c.getRailEl(), panel.id, { failed: false, message: null });
  if (!answer?.url || !answer.available) {
    showFailed(panel, lastMessage);
    return;
  }
  publishPanelUrl(panel.id, answer.url);
  tellStart(panel.id, 'running', null);
  if (state.iframe) state.iframe.src = buildEmbedSrc(answer.url);
  stopHealthPolling(state);
  startHealthPolling(panel);
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
  c.panelState[panel.id].settled = true;
  setEntryEnabled(c.getRailEl(), panel.id, true);
}
