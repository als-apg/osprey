// @ts-check
/**
 * panel — wires the trajectory panel together: the stream feeds the session
 * holder state and the latest snapshot, the controls call the API client, and
 * every change re-renders the views.
 *
 * All I/O goes through `stream.js` (the event stream), `api.js` (every
 * request) and `session.js` (whether this tab holds the target). The outcome
 * of record is the stream: an API answer only adds a notice (a refusal with
 * its code, or "outcome pending — see state") and drops the session on a
 * lost-session refusal.
 *
 * While a continuous run this tab holds is in progress, a heartbeat is sent
 * every {@link HEARTBEAT_MS}. The lease is a dead-man: when the heartbeat
 * stops (the tab closed, crashed, or was throttled while hidden) the run ends
 * `lease_expired`, and the panel shows that reason prominently.
 *
 * The session is never released on unload; Release is an explicit control.
 *
 * @module panel
 */

import { createApi, putBackBody } from './api.js';
import { createSession } from './session.js';
import { openStream } from './stream.js';
import {
  actionView,
  headerView,
  loopArgs,
  loopControls,
  loopForm,
  noticeFromEvent,
  noticeFromResult,
  noticesView,
  referencesView,
  setLoopEnabled,
  shouldHeartbeat,
  statusView,
  stepView,
} from './views.js';

/** The continuous heartbeat period. */
export const HEARTBEAT_MS = 5000;

/** How many notices the list keeps, newest first. */
export const MAX_NOTICES = 12;

/**
 * @typedef {object} PanelOptions
 * @property {HTMLElement} root  The element the panel renders into.
 * @property {import('./stream.js').EventSourceCtorLike} [EventSourceCtor]
 * @property {import('./api.js').FetchLike} [fetchImpl]
 * @property {(fn: () => void, ms: number) => unknown} [setIntervalImpl]
 * @property {(id: any) => void} [clearIntervalImpl]
 */

/**
 * @typedef {object} PanelHandle
 * @property {() => any} snapshot   The latest snapshot, or null.
 * @property {() => import('./views.js').Notice[]} notices
 * @property {() => boolean} heartbeating
 * @property {() => void} close     Close the stream and stop the heartbeat.
 */

/**
 * Start the panel in *root*.
 * @param {PanelOptions} options
 * @returns {PanelHandle}
 */
export function startPanel({
  root,
  EventSourceCtor,
  fetchImpl,
  setIntervalImpl = (fn, ms) => setInterval(fn, ms),
  clearIntervalImpl = (id) => clearInterval(id),
}) {
  /** @type {any} */
  let snapshot = null;
  /** @type {string} */
  let connection = 'connecting';
  /** @type {import('./views.js').Notice[]} */
  let notices = [];
  /** @type {import('./views.js').Notice | null} */
  let ended = null;
  /** @type {{ names: string[], current: string | null } | null} */
  let references = null;
  /** @type {any[] | null} */
  let shots = null;
  /** @type {unknown} */
  let heartbeatTimer = null;
  let listsLoaded = false;

  const session = createSession({ onChange: render });
  const api = createApi({ getClientId: session.clientId, fetchImpl });

  const form = loopForm({
    onSetup: () => act('setup', () => api.setup(form.bunchClean.checked)),
    onShot: () => {
      const { n_shots, average } = loopArgs(form);
      act('shot', () => api.shot({ n_shots, average }));
    },
    onApply: () => {
      if (snapshot?.step) act('apply', () => api.apply(snapshot.step.digest));
    },
    onPutBack: () => {
      const body = snapshot ? putBackBody(snapshot) : null;
      if (body) act('put back', () => api.putBack(body));
    },
    onContinuous: () => {
      ended = null;
      act('continuous', () => api.continuous(loopArgs(form)));
    },
  });

  const header = root.appendChild(document.createElement('div'));
  const noticeArea = root.appendChild(document.createElement('div'));
  const actionArea = root.appendChild(document.createElement('div'));
  root.appendChild(form.root);
  const stepArea = root.appendChild(document.createElement('div'));
  const statusArea = root.appendChild(document.createElement('div'));
  const referenceArea = root.appendChild(document.createElement('div'));

  /** @param {import('./views.js').Notice | null} notice */
  function notify(notice) {
    if (notice == null) return;
    notices = [notice, ...notices].slice(0, MAX_NOTICES);
  }

  /**
   * Note a call's result: a notice when it needs one, and the session dropped
   * on a lost-session refusal.
   * @param {string} label
   * @param {import('./api.js').Result} result
   */
  function absorb(label, result) {
    if (result.kind === 'refused') session.refused(result.code);
    notify(noticeFromResult(label, result));
  }

  /**
   * Run one call made by a control, note its result, and re-render.
   * @param {string} label
   * @param {() => Promise<import('./api.js').Result>} call
   */
  async function act(label, call) {
    const result = await call();
    absorb(label, result);
    if (label === 'continuous' && result.kind === 'ok') syncHeartbeat(true);
    render();
    return result;
  }

  async function openOrTakeOver(/** @type {'open' | 'takeover'} */ request) {
    session.begin(request);
    render();
    const result = request === 'open' ? await api.openSession() : await api.takeOver();
    if (result.kind === 'ok') session.granted(result.body);
    else session.failed();
    notify(noticeFromResult(request === 'open' ? 'open session' : 'take over', result));
    render();
  }

  async function release() {
    const result = await api.release();
    if (result.kind === 'ok') session.released();
    absorb('release', result);
    render();
  }

  async function refreshLists() {
    const [refs, listed] = await Promise.all([api.references(), api.shots()]);
    if (refs.kind === 'ok') references = refs.body;
    if (listed.kind === 'ok') shots = listed.body.shots;
    render();
  }

  async function sendHeartbeat() {
    const result = await api.heartbeat();
    if (result.kind === 'refused') {
      absorb('heartbeat', result);
      render();
    }
  }

  /** @param {boolean} [force]  Start it even before the snapshot shows the run. */
  function syncHeartbeat(force = false) {
    const want = force || shouldHeartbeat(snapshot, session.holding());
    if (want && heartbeatTimer == null) {
      heartbeatTimer = setIntervalImpl(() => void sendHeartbeat(), HEARTBEAT_MS);
    } else if (!want && heartbeatTimer != null) {
      clearIntervalImpl(heartbeatTimer);
      heartbeatTimer = null;
    }
  }

  /**
   * @param {string} type
   * @param {any} payload
   */
  function onEvent(type, payload) {
    session.onEvent(type, payload);
    if (type === 'snapshot') {
      snapshot = payload?.snapshot ?? snapshot;
      syncHeartbeat();
    } else {
      const notice = noticeFromEvent(type, payload);
      if (notice?.kind === 'ended') ended = notice;
      notify(notice);
    }
    // A shot stores a new shot; the first snapshot of an admitted owner loads
    // the lists once.
    const listsStale = type === 'shot' || (type === 'snapshot' && !listsLoaded);
    if (listsStale && snapshot?.caller != null) {
      listsLoaded = true;
      void refreshLists();
    }
    render();
  }

  function render() {
    const controls = session.controls(snapshot);
    const loop = loopControls(snapshot, { holding: session.holding() });
    header.replaceChildren(headerView(snapshot, connection, session.holding()));
    noticeArea.replaceChildren(noticesView(ended, notices));
    actionArea.replaceChildren(
      actionView(snapshot, controls, loop, {
        onOpen: () => void openOrTakeOver('open'),
        onTakeOver: () => void openOrTakeOver('takeover'),
        onRelease: () => void release(),
        onStop: () => void act('stop', () => api.stop()),
      }),
    );
    setLoopEnabled(form, loop);
    stepArea.replaceChildren(stepView(snapshot?.step ?? null));
    statusArea.replaceChildren(statusView(snapshot));
    referenceArea.replaceChildren(
      referencesView(references, shots, snapshot, loop, {
        onSave: (name, shotId) =>
          void act('save reference', () => api.saveReference(name, shotId)).then((r) => {
            if (r.kind === 'ok') void refreshLists();
          }),
        onSelect: (name) => void act('select reference', () => api.selectReference(name)),
        onRefresh: () => void refreshLists(),
      }),
    );
  }

  const stream = openStream({
    onEvent,
    onState: (state) => {
      connection = state;
      render();
    },
    EventSourceCtor,
  });
  render();

  return {
    snapshot: () => snapshot,
    notices: () => notices,
    heartbeating: () => heartbeatTimer != null,
    close() {
      stream.close();
      if (heartbeatTimer != null) clearIntervalImpl(heartbeatTimer);
      heartbeatTimer = null;
    },
  };
}

const mount = typeof document === 'undefined' ? null : document.getElementById('panel-body');
if (mount && typeof globalThis.EventSource === 'function') startPanel({ root: mount });
