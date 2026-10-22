// @ts-check
/**
 * panel — the composition root of the trajectory panel.
 *
 * It builds the shell ({@link shellView}): one host per `data-area`, and
 * mounts each area module into its host with `mount(host, ctx)`. The modules
 * never reach into each other; what they share travels through {@link
 * AreaContext}: the API client, the latest snapshot and its stream, the
 * stream's events by type, the render adapter, the keyed `view` store, the
 * notice list, and the continuous run's start and stop (which own the
 * heartbeat).
 *
 * The shell's own parts are the header strip (machine facts, session
 * controls, notices), the status strip with Stop, and the centre tabs.
 * Embedded in the web-terminal hub and not in Simple mode, the session
 * controls and the centre tabs are handed to the hub's tile bar through
 * `contributeHeader`, and the hub's clicks come back through
 * `onHeaderAction` to the same functions the in-body controls call; the
 * in-body copies are hidden by CSS in exactly that case. The status strip is
 * always in the body.
 *
 * All I/O goes through `stream.js` (the event stream), `api.js` (every
 * request) and `session.js` (whether this tab holds the target). The outcome
 * of record is the stream: an API answer only adds a notice (a refusal with
 * its code, or "outcome pending — see state"), and any caller's lost-session
 * refusal drops the session.
 *
 * While a continuous run this tab holds is in progress, a heartbeat is sent
 * every {@link HEARTBEAT_MS}. The lease is a dead-man: when the heartbeat
 * stops (the tab closed, crashed, or was throttled while hidden) the run ends
 * `lease_expired`, and the panel shows that reason prominently.
 *
 * The session is never released on unload; End session is an explicit control.
 *
 * @module panel
 */

import { contributeHeader, onHeaderAction } from '/design-system/js/header-contrib.js';
import { onModeChange } from '/design-system/js/frame-params.js';

import { createApi } from './api.js';
import { mount as mountChain } from './chain.js';
import { mount as mountCorrection } from './correction.js';
import { mount as mountDrawer } from './drawer.js';
import { mount as mountPlots } from './plots.js';
import { mount as mountRecords } from './records.js';
import { mount as mountReferences } from './references.js';
import { createRenderAdapter } from './render.js';
import { createSession } from './session.js';
import { mount as mountSetup } from './setup.js';
import { openStream } from './stream.js';
import {
  CENTRE_TABS,
  CENTRE_TABS_ITEM,
  createView,
  headerItems,
  headerView,
  noticeFromEvent,
  noticeFromResult,
  noticesView,
  sessionButtons,
  sessionView,
  shellView,
  shouldHeartbeat,
  showTab,
  statusView,
} from './views.js';

/** The continuous heartbeat period. */
export const HEARTBEAT_MS = 5000;

/** How many notices the list keeps, newest first. */
export const MAX_NOTICES = 12;

/**
 * What every area module is handed.
 * @typedef {object} AreaContext
 * @property {import('./api.js').Api} api
 * @property {() => any} snapshot  The latest snapshot, or null.
 * @property {(fn: (snapshot: any) => void) => () => void} onSnapshot
 *   Runs *fn* with each snapshot as it arrives; answers the unsubscribe.
 * @property {(type: string, fn: (payload: any) => void) => () => void} onEvent
 *   Runs *fn* with each stream event of *type*; answers the unsubscribe.
 * @property {any} adapter  The render adapter (`render.js`), the one owner of Plotly.
 * @property {import('./views.js').ViewStore} view  The keyed store the areas share.
 * @property {(notice: import('./views.js').Notice | null) => void} notify  Adds to the header's notices.
 * @property {(body: import('./api.js').ChainBody) => Promise<import('./api.js').Result>} startContinuous
 *   Starts a continuous run and its heartbeat.
 * @property {() => Promise<import('./api.js').Result>} stopContinuous  Stops it.
 * @property {() => boolean} holding  Whether this tab holds the session.
 * @property {(name: string) => HTMLElement | null} area  Another area's host, by `data-area`.
 */

/**
 * @typedef {(host: HTMLElement, ctx: AreaContext) => void} Mount
 */

/** Which module fills which area host. */
export const AREA_MODULES = Object.freeze({
  setup: mountSetup,
  chain: mountChain,
  plots: mountPlots,
  records: mountRecords,
  references: mountReferences,
  correction: mountCorrection,
  drawer: mountDrawer,
});

/**
 * @typedef {object} PanelOptions
 * @property {HTMLElement} root  The element the panel renders into.
 * @property {import('./stream.js').EventSourceCtorLike} [EventSourceCtor]
 * @property {import('./api.js').FetchLike} [fetchImpl]
 * @property {(fn: () => void, ms: number) => unknown} [setIntervalImpl]
 * @property {(id: any) => void} [clearIntervalImpl]
 * @property {any} [adapter]  The render adapter; one over the document when omitted.
 * @property {Record<string, Mount>} [modules]  Overrides {@link AREA_MODULES} entries.
 */

/**
 * @typedef {object} PanelHandle
 * @property {() => any} snapshot   The latest snapshot, or null.
 * @property {() => import('./views.js').Notice[]} notices
 * @property {() => boolean} heartbeating
 * @property {() => import('./views.js').CentreTab} tab  The centre tab showing.
 * @property {(tab: import('./views.js').CentreTab) => void} showTab
 * @property {(id: string, value?: string) => void} headerAction  What a hub tile-bar click does.
 * @property {() => void} close     Close the stream and stop the heartbeat.
 */

/**
 * @template T
 * @returns {{ add: (fn: (value: T) => void) => () => void, emit: (value: T) => void }}
 */
function listeners() {
  /** @type {Set<(value: T) => void>} */
  const fns = new Set();
  return {
    add(fn) {
      fns.add(fn);
      return () => void fns.delete(fn);
    },
    emit(value) {
      for (const fn of [...fns]) fn(value);
    },
  };
}

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
  adapter,
  modules = {},
}) {
  /** @type {any} */
  let snapshot = null;
  /** @type {string} */
  let connection = 'connecting';
  /** @type {import('./views.js').Notice[]} */
  let notices = [];
  /** @type {import('./views.js').Notice | null} */
  let ended = null;
  /** @type {unknown} */
  let heartbeatTimer = null;
  /** @type {import('./views.js').CentreTab} */
  let activeTab = 'raw';
  let closed = false;

  // A change of holding re-renders the areas too: they read `ctx.holding()`
  // on a snapshot, and the snapshot naming the new holder may arrive before
  // the grant does.
  const session = createSession({
    onChange: () => {
      render();
      if (snapshot != null) snapshots.emit(snapshot);
    },
  });
  const api = createApi({
    getClientId: session.clientId,
    fetchImpl,
    onRefused: (refusal) => void session.refused(refusal.code),
  });
  const snapshots = listeners();
  /** @type {Map<string, ReturnType<typeof listeners<any>>>} */
  const events = new Map();

  const shell = shellView((tab) => setTab(tab));
  root.replaceChildren(shell.root);

  /** @param {import('./views.js').Notice | null} notice */
  function notify(notice) {
    if (notice == null) return;
    notices = [notice, ...notices].slice(0, MAX_NOTICES);
    render();
  }

  /**
   * Run one call made by a control and note its result as a notice when it
   * needs one (the API client already dropped the session on a lost-session
   * refusal).
   * @param {string} label
   * @param {() => Promise<import('./api.js').Result>} call
   */
  async function act(label, call) {
    const result = await call();
    notify(noticeFromResult(label, result));
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
    await act('end session', async () => result);
  }

  /** @param {string} name  A session control's inventory name. */
  function onSession(name) {
    if (name === 'session-open') void openOrTakeOver('open');
    else if (name === 'session-takeover') void openOrTakeOver('takeover');
    else if (name === 'session-end') void release();
  }

  async function sendHeartbeat() {
    const result = await api.heartbeat();
    if (result.kind === 'refused') await act('heartbeat', async () => result);
  }

  /** @param {boolean} [force]  Start it even before the snapshot shows the run. */
  function syncHeartbeat(force = false) {
    const want = !closed && (force || shouldHeartbeat(snapshot, session.holding()));
    if (want && heartbeatTimer == null) {
      heartbeatTimer = setIntervalImpl(() => void sendHeartbeat(), HEARTBEAT_MS);
    } else if (!want && heartbeatTimer != null) {
      clearIntervalImpl(heartbeatTimer);
      heartbeatTimer = null;
    }
  }

  /** @param {import('./views.js').CentreTab} tab */
  function setTab(tab) {
    activeTab = tab;
    showTab(shell, tab);
    publish();
  }

  function publish() {
    if (closed) return;
    contributeHeader(headerItems(sessionButtons(snapshot, session.controls(snapshot)), activeTab));
  }

  /**
   * A tile-bar click: the session controls and the centre tabs.
   * @param {string} id
   * @param {string} [value]
   */
  function headerAction(id, value) {
    if (closed) return;
    if (id === CENTRE_TABS_ITEM) {
      const entry = CENTRE_TABS.find((t) => t.name === value);
      if (entry) setTab(/** @type {import('./views.js').CentreTab} */ (entry.tab));
    } else {
      onSession(id);
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
      if (notice) notices = [notice, ...notices].slice(0, MAX_NOTICES);
    }
    render();
    if (type === 'snapshot') snapshots.emit(snapshot);
    events.get(type)?.emit(payload);
  }

  function render() {
    if (closed) return;
    const controls = session.controls(snapshot);
    const buttons = sessionButtons(snapshot, controls);
    /** @type {import('./views.js').Notice[]} */
    const served = (snapshot?.notices ?? []).map((/** @type {string} */ text) => ({ kind: 'info', title: text }));
    shell.areas.header.replaceChildren(headerView(snapshot, session.holding()));
    shell.areas.session.replaceChildren(sessionView(buttons, onSession));
    shell.areas.notices.replaceChildren(noticesView(ended, [...served, ...notices]));
    shell.areas.status.replaceChildren(
      statusView(snapshot, connection, controls, () => void act('stop', () => api.stop())));
    publish();
  }

  /** @type {AreaContext} */
  const ctx = {
    api,
    snapshot: () => snapshot,
    onSnapshot: (fn) => snapshots.add(fn),
    onEvent(type, fn) {
      const set = events.get(type) ?? listeners();
      events.set(type, set);
      return set.add(fn);
    },
    adapter: adapter ?? createRenderAdapter({ root: root.ownerDocument }),
    view: createView(),
    notify,
    async startContinuous(body) {
      ended = null;
      const result = await act('continuous', () => api.continuous(body));
      if (result.kind === 'ok') syncHeartbeat(true);
      return result;
    },
    stopContinuous: () => act('stop', () => api.stop()),
    holding: () => session.holding(),
    area: (name) => shell.areas[name] ?? null,
  };

  /** @type {Record<string, Mount>} */
  const mounts = { ...AREA_MODULES, ...modules };
  for (const [name, mount] of Object.entries(mounts)) {
    const host = shell.areas[name];
    if (host) mount(host, ctx);
  }

  onHeaderAction(headerAction);
  onModeChange(() => publish());
  setTab(activeTab);

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
    tab: () => activeTab,
    showTab: setTab,
    headerAction,
    close() {
      stream.close();
      if (heartbeatTimer != null) clearIntervalImpl(heartbeatTimer);
      heartbeatTimer = null;
      closed = true;
    },
  };
}

const body = typeof document === 'undefined' ? null : document.getElementById('panel-body');
if (body && typeof globalThis.EventSource === 'function') startPanel({ root: body });
