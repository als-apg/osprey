// @ts-check
/* OSPREY Web Terminal — Session Layout
 *
 * Each session keeps its own web layout on the hub, and coming back to a
 * session brings it back: which panels are open, in which tile, their sizes,
 * the active tab, the Agent pane, and the page each panel was on where the hub
 * can read it.
 *
 * The layout is the dock's own stored form (dock-workspace.js's `currentDock`,
 * the value a reload restores from localStorage) with `pages` beside it: panel
 * id → the root-relative page its frame showed (panel-lifecycle's
 * `getPanelPage`). A first-party panel's page is read from its frame; an
 * opaque or cross-origin panel's is not readable and it reopens at its start
 * page, and the Artifacts viewer keeps its own per-session last artifact.
 *
 * The session is the active row's transcript (session-model.js). While the
 * tab is on it, every settled dock change and every panel page load writes its
 * layout (`PUT /api/sessions/{id}/layout`, debounced); leaving it writes once
 * more from the screen it left. Arriving reads the new session's layout
 * (`GET`) and applies it; a session with none yet — an older one, or a new one
 * — keeps the layout on screen, which becomes its own. Changes made while a
 * layout is applying, and for {@link SETTLE_MS} after, are the apply's own and
 * are never written back.
 *
 * The sidebar's width and fold state are per-user chrome, not part of a
 * session's layout.
 */

import { fetchJSON, withPrefix } from './api.js';
import { activeRow, subscribe as subscribeToSessions } from './session-model.js';
import { applyDock, currentDock, getDockApi, setLayoutRestoredHook } from './dock-workspace.js';
import { getPanelPage, restorePanelPage } from './panel-lifecycle.js';

/** Debounce of a layout write, so a gesture's burst of changes writes once:
 *  the cadence of the dock's own localStorage write (dock-workspace.js). */
export const SAVE_DEBOUNCE_MS = 150;

/** A keepalive request's body must stay under the browser's 64 KiB budget. */
const KEEPALIVE_MAX_CHARS = 60_000;

/** How long after an apply the dock's changes are still the apply's own. */
export const SETTLE_MS = 400;

/** The panel whose page is its own per-session last artifact, not an address. */
const ARTIFACTS_PANEL_ID = 'artifacts';

/**
 * @typedef {object} SessionLayout
 * @property {2} version
 * @property {any} layout
 * @property {Record<string, any>} panels
 * @property {Record<string, string>} pages
 */

let started = false;
/** The transcript the tab is on, as last followed. @type {string | null} */
let following = null;
/** The transcript writes go to: `following` once its layout has been read. @type {string | null} */
let target = null;
let generation = 0;
let applying = false;
/** @type {ReturnType<typeof setTimeout> | null} */
let saveTimer = null;
/** @type {ReturnType<typeof setTimeout> | null} */
let settleTimer = null;
/** The last layout known to be stored, per transcript, as {@link stableJson}. */
const stored = new Map();
/** @type {(() => void)[]} */
const teardown = [];

/**
 * Start following sessions. A page without a dock (the fallback host) has no
 * layout to keep, so nothing starts. The first session is followed once the
 * boot layout has settled, so a reload's own restore is never mistaken for a
 * change. Idempotent.
 */
export function initSessionLayout() {
  if (started) return;
  const api = getDockApi();
  if (!api) return;
  started = true;
  const disposers = [api.onDidLayoutChange(onDockChange), api.onDidActivePanelChange(onDockChange)];
  teardown.push(() => disposers.forEach((d) => d?.dispose?.()));
  document.addEventListener('load', onFrameLoad, true);
  window.addEventListener('pagehide', onLeave);
  document.addEventListener('visibilitychange', onVisibility);
  teardown.push(() => {
    document.removeEventListener('load', onFrameLoad, true);
    window.removeEventListener('pagehide', onLeave);
    document.removeEventListener('visibilitychange', onVisibility);
  });
  setLayoutRestoredHook(() => {
    if (!started) return;
    teardown.push(subscribeToSessions(follow));
    follow();
  });
}

/** Stop following and forget everything (tests). */
export function resetSessionLayout() {
  for (const off of teardown.splice(0)) off();
  if (saveTimer) clearTimeout(saveTimer);
  if (settleTimer) clearTimeout(settleTimer);
  saveTimer = settleTimer = null;
  stored.clear();
  started = applying = false;
  following = target = null;
  generation += 1;
}

/**
 * @param {string} tid
 * @returns {string}
 */
function layoutPath(tid) {
  return `/api/sessions/${encodeURIComponent(tid)}/layout`;
}

/**
 * JSON with object keys sorted, so two equal layouts compare equal whatever
 * order their keys were written in.
 * @param {unknown} value
 * @returns {string}
 */
function stableJson(value) {
  return JSON.stringify(value, (_key, v) =>
    v && typeof v === 'object' && !Array.isArray(v)
      ? Object.fromEntries(
          Object.keys(v)
            .sort()
            .map((k) => [k, v[k]])
        )
      : v
  );
}

/**
 * The layout on screen, or null without a dock.
 * @returns {SessionLayout | null}
 */
function snapshot() {
  const dock = currentDock();
  if (!dock) return null;
  /** @type {Record<string, string>} */
  const pages = {};
  for (const id of Object.keys(dock.panels ?? {})) {
    if (id === ARTIFACTS_PANEL_ID) continue;
    const page = getPanelPage(id);
    if (page) pages[id] = page;
  }
  return { ...dock, pages };
}

// ---- Following the session ----

/** Follow the active row's transcript: leave the old one, arrive at the new. */
function follow() {
  const tid = activeRow()?.transcript_id ?? null;
  if (tid === following) return;
  // The screen is still the session being left: write it from there.
  save();
  following = tid;
  target = null;
  const gen = ++generation;
  // A new session the listing has not caught up with has no transcript to
  // keep a layout under yet; it keeps the screen, and saves it on arrival.
  if (tid) void arrive(tid, gen);
}

/**
 * Read a session's layout and apply it; with none kept, the layout on screen
 * becomes its own. A read that fails (the hub unreachable, a session not this
 * operator's) neither applies nor overwrites.
 * @param {string} tid
 * @param {number} gen
 */
async function arrive(tid, gen) {
  /** @type {any} */
  let saved;
  try {
    saved = (await fetchJSON(layoutPath(tid)))?.layout ?? null;
  } catch (err) {
    console.warn('[session-layout] could not read the session layout', err);
    return;
  }
  if (gen !== generation) return;
  target = tid;
  if (saved && apply(saved)) {
    stored.set(tid, stableJson(saved));
    return;
  }
  save();
}

/**
 * Apply a stored layout: the dock, then each panel's page. The dock's changes
 * until {@link SETTLE_MS} after are the apply's own.
 * @param {any} saved
 * @returns {boolean} whether the dock took it
 */
function apply(saved) {
  if (saveTimer) clearTimeout(saveTimer);
  saveTimer = null;
  applying = true;
  try {
    if (!applyDock(saved)) {
      applying = false;
      return false;
    }
    const pages = saved.pages && typeof saved.pages === 'object' ? saved.pages : {};
    for (const [id, page] of Object.entries(pages)) {
      if (id !== ARTIFACTS_PANEL_ID && typeof page === 'string') restorePanelPage(id, page);
    }
  } finally {
    if (applying) {
      if (settleTimer) clearTimeout(settleTimer);
      settleTimer = setTimeout(() => {
        settleTimer = null;
        applying = false;
      }, SETTLE_MS);
    }
  }
  return true;
}

// ---- Writing ----

function onDockChange() {
  if (applying || !target) return;
  if (saveTimer) return;
  saveTimer = setTimeout(() => {
    saveTimer = null;
    save();
  }, SAVE_DEBOUNCE_MS);
}

/** A panel frame finished loading a page: its page may have changed. @param {Event} event */
function onFrameLoad(event) {
  const el = event.target;
  if (el instanceof HTMLIFrameElement && el.classList.contains('panel-iframe')) onDockChange();
}

function onLeave() {
  save();
}

function onVisibility() {
  if (document.visibilityState === 'hidden') onLeave();
}

/**
 * Write the layout on screen as the followed session's, unless it is what is
 * stored already or a layout is applying. The write is `keepalive` (when it
 * fits the browser's keepalive budget), so a reload or a closed tab right
 * after a change does not cancel it and the next page reads what was left.
 */
function save() {
  if (saveTimer) clearTimeout(saveTimer);
  saveTimer = null;
  const tid = target;
  if (!tid || applying) return;
  const value = snapshot();
  if (!value) return;
  const text = stableJson(value);
  if (stored.get(tid) === text) return;
  stored.set(tid, text);
  const body = JSON.stringify({ layout: value });
  void fetch(withPrefix(layoutPath(tid)), {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body,
    keepalive: body.length < KEEPALIVE_MAX_CHARS,
  })
    .then((resp) => {
      if (resp.ok) return;
      if (stored.get(tid) === text) stored.delete(tid);
      console.warn(`[session-layout] saving the session layout failed (HTTP ${resp.status})`);
    })
    .catch((err) => {
      if (stored.get(tid) === text) stored.delete(tid);
      console.warn('[session-layout] saving the session layout failed', err);
    });
}
