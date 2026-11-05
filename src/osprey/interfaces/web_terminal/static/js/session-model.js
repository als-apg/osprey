/**
 * The client's session model: the operator's sessions, which one the tab is
 * on, and the guarded way to move between them.
 *
 * The session pointer (`session-pointer.js`) is the single source of truth for
 * the tab's session. This module subscribes to it, so every key change — the
 * Agent bar's `+` on either surface, a `session_switched`, a
 * {@link switchTo}, page load — does the same three things: PUTs last-active
 * (debounced), clears the pick, and re-derives {@link activeRow}. A key the
 * listing has not caught up with gets a synthesized "New session" row.
 *
 * Rows come from the hub's `GET /api/sessions`, paged by `next_cursor`. Only a
 * page's first listing waits on the record's drain (`drain=1`); {@link refresh}
 * is throttled and merges page 1 without dropping the later pages.
 *
 * The pick is the artifact the tab shows for the session it is on: one value,
 * cleared on every key change, set by {@link switchTo} (`artifactId`, else the
 * row's `last_artifact_id`) and by the sidebar.
 */

import { apiRequest, createEventSource, fetchJSON, pageEventsUrl } from './api.js';
import { getPointer, subscribe as subscribeToPointer } from './session-pointer.js';
import { showConfirm } from './posture-confirm.js';
import { isStreaming, newConversation, stopTurn } from './chat.js';
import { onChat } from './chat-events.js';
import { resumeSession, startNewSession } from './sessions.js';
import { isTerminalSurface } from './agent-surface.js';

/** The hub's session listing. */
export const SESSIONS_PATH = '/api/sessions';

/** Debounce of the last-active PUT, so a burst of key changes writes once. */
export const LAST_ACTIVE_DEBOUNCE_MS = 300;

/** Throttle of {@link refresh}. */
export const REFRESH_THROTTLE_MS = 1000;

/** How long a "Stop & switch" waits for the stopped turn to end. */
export const STOP_TIMEOUT_MS = 10_000;

/** Title of a synthesized row for a key the listing does not have yet. */
export const NEW_SESSION_TITLE = 'New session';

/**
 * @typedef {object} SessionRow
 * @property {string|null} transcript_id - null only on a synthesized row
 * @property {string|null} session_key
 * @property {boolean} [current_for_key]
 * @property {string} [title]
 * @property {string} [naming]
 * @property {string|null} [last_artifact_id]
 * @property {string} [state]
 * @property {boolean} [truncated]
 * @property {boolean} [synthesized]
 */

/** @type {SessionRow[]} */
let loaded = [];
/** @type {string|null} */
let nextCursor = null;
let more = false;
/** @type {string} */
let record = 'complete';
let firstLoaded = false;
/** @type {string|null} */
let pickId = null;
/** The pick {@link switchTo} chose, applied when the pointer reaches its key. */
let pendingPick = /** @type {{key: string, id: string|null} | null} */ (null);
/** @type {Set<(change: {reason: string}) => void>} */
const listeners = new Set();
/** @type {(() => void)[]} */
const unsubscribers = [];
let active = false;
/** @type {ReturnType<typeof setTimeout> | null} */
let lastActiveTimer = null;
/** @type {ReturnType<typeof setTimeout> | null} */
let refreshTimer = null;
let lastRefreshAt = -Infinity;
/** @type {Promise<void> | null} */
let inFlight = null;
let generation = 0;

/** @param {string} reason */
function notify(reason) {
  for (const fn of [...listeners]) {
    try {
      fn({ reason });
    } catch (err) {
      console.error('[session-model] listener failed', err);
    }
  }
}

/**
 * @param {any} body
 * @returns {SessionRow[]}
 */
function rowsOf(body) {
  return Array.isArray(body?.sessions) ? body.sessions : [];
}

/**
 * Start the model: subscribe to the pointer and the hub's session frames, and
 * load the first page. Idempotent.
 * @returns {Promise<void>}
 */
export function initSessionModel() {
  if (active) return inFlight ?? Promise.resolve();
  active = true;
  unsubscribers.push(subscribeToPointer(onPointer));
  unsubscribers.push(onChat('turn-end', () => refresh()));
  unsubscribers.push(createEventSource(pageEventsUrl(), { onMessage: onHubFrame }).stop);
  return loadFirst();
}

/** Stop the model and forget everything (tests, teardown). */
export function resetSessionModel() {
  for (const off of unsubscribers.splice(0)) off();
  if (lastActiveTimer) clearTimeout(lastActiveTimer);
  if (refreshTimer) clearTimeout(refreshTimer);
  lastActiveTimer = refreshTimer = null;
  listeners.clear();
  loaded = [];
  nextCursor = null;
  more = false;
  record = 'complete';
  firstLoaded = false;
  pickId = null;
  pendingPick = null;
  lastRefreshAt = -Infinity;
  inFlight = null;
  generation += 1;
  active = false;
}

/** @param {string|null} key */
function onPointer(key) {
  pickId = pendingPick && pendingPick.key === key ? pendingPick.id : null;
  pendingPick = null;
  if (key) {
    if (lastActiveTimer) clearTimeout(lastActiveTimer);
    lastActiveTimer = setTimeout(() => {
      lastActiveTimer = null;
      void apiRequest(`${SESSIONS_PATH}/last-active`, {
        method: 'PUT',
        json: { session_key: key },
        errorPrefix: 'Saving the last session failed',
      }).catch((err) => console.warn('[session-model]', err.message));
    }, LAST_ACTIVE_DEBOUNCE_MS);
  }
  notify('pointer');
  if (firstLoaded) refresh();
}

/** @param {any} data */
function onHubFrame(data) {
  if (data?.type === 'session_transcript') {
    if (loaded.some((r) => r.session_key === data.session_key) || data.session_key === getPointer())
      refresh();
  } else if (data?.type === 'session_named') {
    const row = loaded.find((r) => r.transcript_id === data.transcript_id);
    if (!row) return;
    row.title = data.title;
    row.naming = data.naming;
    notify('named');
  }
}

/**
 * The loaded rows, newest first; the synthesized active row leads when the
 * listing does not have the pointer's key yet.
 * @returns {SessionRow[]}
 */
export function rows() {
  const current = activeRow();
  return current?.synthesized ? [current, ...loaded] : [...loaded];
}

/**
 * The row the tab is on: the loaded row current for the pointer's key, else
 * one synthesized from the key. Null with no pointer.
 * @returns {SessionRow|null}
 */
export function activeRow() {
  const key = getPointer();
  if (!key) return null;
  const row = loaded.find((r) => r.session_key === key && r.current_for_key);
  return (
    row ?? {
      transcript_id: null,
      session_key: key,
      current_for_key: true,
      title: NEW_SESSION_TITLE,
      last_artifact_id: null,
      synthesized: true,
    }
  );
}

/** Whether more sessions wait behind `next_cursor`. */
export function hasMore() {
  return more && nextCursor !== null;
}

/** The record state of the last listing: complete | incomplete | unavailable. */
export function recordState() {
  return record;
}

/**
 * Listen for model changes (`pointer`, `rows`, `named`, `pick`, `switched`).
 * @param {(change: {reason: string}) => void} fn
 * @returns {() => void}
 */
export function subscribe(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

/** The picked artifact id for the active session, or null. */
export function pick() {
  return pickId;
}

/**
 * Set (or clear) the pick for the active session.
 * @param {string|null} id
 */
export function setPick(id) {
  if (pickId === id) return;
  pickId = id;
  notify('pick');
}

/**
 * Record an artifact as its owner session's last-open one (no owner: none).
 * @param {{id: string, owner?: string|null} | null} entry
 */
export function recordLastArtifact(entry) {
  if (!entry?.owner) return;
  void apiRequest(`${SESSIONS_PATH}/${encodeURIComponent(entry.owner)}/last-artifact`, {
    method: 'PUT',
    json: { artifact_id: entry.id },
    errorPrefix: 'Saving the last artifact failed',
  }).catch((err) => console.warn('[session-model]', err.message));
}

/**
 * The loaded row that owns an artifact (by the server's `owner`), or null.
 * @param {{owner?: string|null}|null|undefined} artifact
 * @returns {SessionRow|null}
 */
export function sessionOfArtifact(artifact) {
  const owner = artifact?.owner;
  if (!owner) return null;
  return loaded.find((r) => r.transcript_id === owner) ?? null;
}

/** @param {string|null} cursor @param {boolean} drain */
function listUrl(cursor, drain) {
  const params = new URLSearchParams();
  if (cursor) params.set('cursor', cursor);
  if (drain) params.set('drain', '1');
  const query = params.toString();
  return query ? `${SESSIONS_PATH}?${query}` : SESSIONS_PATH;
}

/** First page, waiting on the drain. */
function loadFirst() {
  const gen = generation;
  inFlight = fetchJSON(listUrl(null, true))
    .then((body) => {
      if (gen !== generation) return;
      loaded = rowsOf(body);
      nextCursor = body?.next_cursor ?? null;
      more = Boolean(body?.more);
      record = body?.record ?? 'complete';
      firstLoaded = true;
      notify('rows');
    })
    .catch((err) => {
      if (gen !== generation) return;
      firstLoaded = true;
      record = 'unavailable';
      console.warn('[session-model] listing failed', err);
      notify('rows');
    })
    .finally(() => {
      if (gen === generation) inFlight = null;
    });
  lastRefreshAt = Date.now();
  return inFlight;
}

/**
 * Load the next page of sessions (its first listing waits on the drain).
 * @returns {Promise<boolean>} whether a page was appended
 */
export async function loadMore() {
  if (!hasMore() || inFlight) return false;
  const gen = generation;
  const cursor = nextCursor;
  /** @type {any} */
  let body;
  try {
    inFlight = fetchJSON(listUrl(cursor, true));
    body = await inFlight;
  } catch (err) {
    console.warn('[session-model] next page failed', err);
    return false;
  } finally {
    if (gen === generation) inFlight = null;
  }
  if (gen !== generation) return false;
  const seen = new Set(loaded.map((r) => r.transcript_id));
  loaded = [...loaded, ...rowsOf(body).filter((r) => !seen.has(r.transcript_id))];
  nextCursor = body?.next_cursor ?? null;
  more = Boolean(body?.more);
  notify('rows');
  return true;
}

/**
 * Re-list page 1 (throttled, no drain) and merge it in front of the rows
 * already loaded, keeping the later pages.
 * @returns {void}
 */
export function refresh() {
  if (!active || refreshTimer) return;
  const wait = Math.max(0, lastRefreshAt + REFRESH_THROTTLE_MS - Date.now());
  refreshTimer = setTimeout(() => {
    refreshTimer = null;
    void runRefresh();
  }, wait);
}

async function runRefresh() {
  if (inFlight) await inFlight.catch(() => {});
  const gen = generation;
  lastRefreshAt = Date.now();
  /** @type {any} */
  let body;
  try {
    body = await fetchJSON(listUrl(null, false));
  } catch (err) {
    console.warn('[session-model] refresh failed', err);
    return;
  }
  if (gen !== generation) return;
  const first = rowsOf(body);
  const ids = new Set(first.map((r) => r.transcript_id));
  const rest = loaded.slice(first.length).filter((r) => !ids.has(r.transcript_id));
  // A row that left page 1 because newer ones pushed it down stays loaded.
  const pushedDown = loaded.slice(0, first.length).filter((r) => !ids.has(r.transcript_id));
  loaded = [...first, ...pushedDown, ...rest];
  if (!firstLoaded) {
    nextCursor = body?.next_cursor ?? null;
    more = Boolean(body?.more);
  }
  record = body?.record ?? record;
  firstLoaded = true;
  notify('rows');
}

/**
 * Stop the chat's streaming turn and wait for it to end: true once the
 * console is idle, false when it still streams after `timeoutMs`.
 * @param {number} [timeoutMs]
 * @returns {Promise<boolean>}
 */
export function stopAndWait(timeoutMs = STOP_TIMEOUT_MS) {
  return new Promise((resolve) => {
    let off = () => {};
    const finish = (/** @type {boolean} */ value) => {
      clearTimeout(timer);
      off();
      resolve(value);
    };
    const timer = setTimeout(() => finish(!isStreaming()), timeoutMs);
    off = onChat('busy', () => {
      if (!isStreaming()) finish(true);
    });
    void Promise.resolve(stopTurn())
      .catch(() => {})
      .then(() => {
        if (!isStreaming()) finish(true);
      });
  });
}

/**
 * Ask before cutting a streaming chat turn. Resolves true when the turn
 * stopped (or nothing streamed), false when the operator cancelled or the
 * stop timed out.
 * @param {string} name
 * @returns {Promise<boolean>}
 */
function guardStreaming(name) {
  if (isTerminalSurface() || !isStreaming()) return Promise.resolve(true);
  return new Promise((resolve) => {
    let settled = false;
    const finish = (/** @type {boolean} */ value) => {
      if (settled) return;
      settled = true;
      resolve(value);
    };
    showConfirm({
      title: 'Stop the turn?',
      body: [['The Agent is answering. Stop the turn and switch to "', { em: name }, '"?']],
      live: null,
      confirmLabel: 'Stop & switch',
      focus: 'cancel',
      onConfirm: (ui) => {
        if (ui.confirm) ui.confirm.disabled = true;
        void stopAndWait().then((stopped) => {
          if (stopped) {
            finish(true);
            ui.done();
            return;
          }
          ui.error.textContent = 'The turn did not stop. Still on this session.';
          ui.error.hidden = false;
          if (ui.confirm) ui.confirm.disabled = false;
          finish(false);
        });
      },
      onDismiss: () => finish(false),
    });
  });
}

/**
 * Move the tab to a session: the guarded entry for gestures.
 *
 * A no-op for the active row (an `artifactId` still becomes the pick). While
 * the chat streams it asks first. A row current for its key moves the pointer
 * to that key; any other transcript is entered (`POST …/enter`) and the
 * pointer moves to the minted key. `{new: true}` starts a new conversation.
 * @param {SessionRow | {new: true}} target
 * @param {{artifactId?: string|null, reason?: string}} [options]
 * @returns {Promise<boolean>} whether the tab moved
 */
export async function switchTo(target, { artifactId = null } = {}) {
  if ('new' in target && target.new) {
    if (!(await guardStreaming(NEW_SESSION_TITLE))) return false;
    if (isTerminalSurface()) await startNewSession();
    else newConversation();
    notify('switched');
    return true;
  }
  const row = /** @type {SessionRow} */ (target);
  const current = activeRow();
  const isActive =
    current &&
    ((row.transcript_id && row.transcript_id === current.transcript_id) ||
      (current.synthesized && row.session_key === current.session_key && row.current_for_key));
  if (isActive) {
    if (artifactId) setPick(artifactId);
    return false;
  }
  if (!(await guardStreaming(row.title || 'Untitled session'))) return false;
  let key = row.current_for_key ? row.session_key : null;
  if (!key) {
    try {
      const body = await apiRequest(
        `${SESSIONS_PATH}/${encodeURIComponent(String(row.transcript_id))}/enter`,
        { method: 'POST', errorPrefix: 'Opening the session failed' }
      );
      key = body?.session_key ?? null;
    } catch (err) {
      console.warn('[session-model]', err instanceof Error ? err.message : err);
      return false;
    }
    if (!key) return false;
    // The entered transcript is now current for the minted key.
    row.session_key = key;
    row.current_for_key = true;
  }
  const chosen = artifactId || row.last_artifact_id || null;
  pendingPick = { key, id: chosen };
  await resumeSession(key);
  // The chat moves the pointer at once; the terminal moves it when the socket
  // confirms, and the pending pick waits for that.
  if (getPointer() === key) {
    pendingPick = null;
    pickId = chosen;
  }
  notify('switched');
  return true;
}
