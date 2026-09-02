/* OSPREY Web Terminal — Bar item builders.
 *
 * One module for every item BODY the header and the status bar can show. The
 * catalog (bar-catalog.js) says what a type is; the host (bar-host.js) says
 * where its shell goes; this says what is inside the shell and what it does
 * while it is there. Items are added here, one `defineBarItem()` per type, and
 * nothing else in the interface has to learn that a new type exists.
 *
 * ---- The lifecycle problem this module owns ----
 *
 * A bar item is not a static body. The connection dot listens, the clock and
 * the terminal-size readout run intervals, the stopwatch counts while it runs.
 * Every one of them is ATTACH-SCOPED: it must stop the moment its shell leaves
 * the bar, because the host does not destroy a folded item — it parks the node
 * in the hidden pool, alive, where a live subscription would keep updating a
 * body nobody can see, for the rest of the page's life.
 *
 * So the house rule, established with the connection item and binding on every
 * item added after it: EVERY ATTACH-SCOPED SUBSCRIPTION COMES FROM A
 * DISPOSER-RETURNING API. A builder does not register a listener and hope
 * something later remembers it; it is handed the way to stop at the moment it
 * starts (`api.js`'s `onConnectionStateChange`), and it returns that disposer
 * with its body. An item that cannot hand back a disposer has not finished
 * being written.
 *
 * `defineBarItem()` is the seam that enforces it: a factory returns
 * `{node, dispose}` rather than a bare node, and this module runs `dispose`
 * on detach, on rebuild, and on teardown.
 *
 * ---- What "detach" means today ----
 *
 * bar-host.js has no detach hook to subscribe to: parking a shell is an
 * `appendChild` into `#bar-item-pool` and nothing else. Rather than ask every
 * item to guess, this module watches the pool for arrivals and runs the same
 * idempotent pass, {@link syncBarItems}, that a caller may also run by hand
 * after a `reconcile()`. Both routes converge on one function, so an
 * `onItemDetach` seam in bar-host later replaces the observer by calling
 * `syncBarItems()` — one line, no item changes.
 *
 * Re-attach is the mirror image: on detach the shell's `data-bar-built` stamp
 * is cleared, so the next placement rebuilds the body through the builder
 * instead of re-hanging the bar's chrome on a disposed subscription. That is
 * what makes a folded-then-unfolded item show the CURRENT state rather than
 * the state it froze at.
 *
 * ---- The one thing that must NOT be attach-scoped ----
 *
 * A body is disposable; what the operator MEASURED with it is not. The
 * stopwatch's elapsed time lives in a module-level map keyed by the item's
 * layout key (`ctx.key`), not in the body and not in the shell, so folding the
 * item away, moving it from the header to the status bar, or removing it and
 * putting it back all leave the count running. Per page load, deliberately:
 * nothing is persisted, because "8 hours 12 minutes" restored from yesterday's
 * localStorage is a measurement of nothing.
 */

import { isLive, poolElement, registerItemBuilder } from './bar-host.js';
import { PANEL_HEALTH_STATUS_BAR_IDS } from './bar-catalog.js';
import { PANELS } from './panel-catalog.js';
import { getConnectionState, onConnectionStateChange } from './api.js';
import { getTerminalDimensions } from './terminal.js';

/** @typedef {import('./bar-host.js').BarBuildContext} BarBuildContext */
/** @typedef {import('./bar-host.js').BarRoot} BarRoot */
/** @typedef {import('./bar-catalog.js').BarItemOptions} BarItemOptions */
/** @typedef {import('./api.js').ConnState} ConnState */
/** @typedef {import('./api.js').ConnectionState} ConnectionState */

/**
 * What an item factory returns. `node` is the body (null renders an empty
 * shell); `dispose` stops everything the body started and is optional only for
 * an item that started nothing at all.
 * @typedef {object} BarItemInstance
 * @property {Node | null} node
 * @property {() => void} [dispose]
 */

/** @typedef {(ctx: BarBuildContext) => BarItemInstance} BarItemFactory */

/** Live items, by the shell they are mounted in. @type {Map<HTMLElement, () => void>} */
const instances = new Map();

/** The pool currently under observation, so re-arming is idempotent. */
/** @type {Element | null} */
let watchedPool = null;
/** @type {MutationObserver | null} */
let poolObserver = null;

/* ---- lifecycle ---- */

/**
 * Register one type's body factory and take over its attach/detach lifecycle.
 * Building a body is an attach; a rebuild (density or option change) is a
 * detach followed by an attach, so the previous instance is disposed first.
 * @param {string} type
 * @param {BarItemFactory} factory
 * @returns {() => void} unregister
 */
export function defineBarItem(type, factory) {
  const unregister = registerItemBuilder(type, (ctx) => {
    disposeShell(ctx.shell);
    const instance = factory(ctx);
    if (instance.dispose) instances.set(ctx.shell, instance.dispose);
    watchPool(ctx.shell.ownerDocument);
    return instance.node;
  });
  return () => {
    unregister();
    for (const shell of Array.from(instances.keys())) {
      if (shell.dataset.barItem === type) disposeShell(shell);
    }
  };
}

/**
 * Dispose every item whose shell is no longer live — parked in the pool by a
 * fold or a reconcile, or gone from the document entirely. Idempotent and
 * synchronous: safe to call after every `reconcile()`, and safe to call twice.
 *
 * Attaching is deliberately NOT done here. A body is built in exactly one
 * place, bar-host's builder path, so there is no second implementation of the
 * build signature to keep in step; clearing `data-bar-built` is what hands the
 * rebuild back to it.
 * @param {BarRoot} [root]
 */
export function syncBarItems(root = document) {
  watchPool(root);
  for (const shell of Array.from(instances.keys())) {
    if (shell.isConnected && isLive(shell)) continue;
    disposeShell(shell);
    delete shell.dataset.barBuilt;
  }
}

/**
 * Dispose every live item and stop watching the pool. The teardown entry
 * point — a page leaving, or a test starting from a clean module.
 */
export function disposeBarItems() {
  for (const shell of Array.from(instances.keys())) {
    disposeShell(shell);
    delete shell.dataset.barBuilt;
  }
  if (poolObserver) poolObserver.disconnect();
  poolObserver = null;
  watchedPool = null;
}

/**
 * Run and forget one shell's disposer. A throwing disposer must not strand the
 * others: the whole point of the pass is that every item stops.
 * @param {HTMLElement} shell
 */
function disposeShell(shell) {
  const dispose = instances.get(shell);
  if (!dispose) return;
  instances.delete(shell);
  try {
    dispose();
  } catch (err) {
    console.error(`[bar-items] disposer for "${shell.dataset.barItem}" threw`, err);
  }
}

/**
 * Watch the item pool so a parked shell disposes on its own, with no
 * cooperation from whoever called `reconcile()`. Re-arms when the pool node
 * changes (a re-rendered document), and is a no-op where MutationObserver is
 * absent — {@link syncBarItems} is still the deterministic path.
 * @param {BarRoot | null} root
 */
function watchPool(root) {
  if (!root || typeof MutationObserver === 'undefined') return;
  const pool = poolElement(root);
  if (!pool || pool === watchedPool) return;
  if (poolObserver) poolObserver.disconnect();
  poolObserver = new MutationObserver(() => syncBarItems(root));
  poolObserver.observe(pool, { childList: true });
  watchedPool = pool;
}

/* ---- connection ---- */

/**
 * The dot's modifier per connection state. `connecting` is the bare dot: it is
 * neither healthy nor failed, and the muted default says exactly that.
 * @type {Readonly<Record<ConnState, string>>}
 */
const CONNECTION_DOT_CLASS = Object.freeze({
  connected: ' live',
  connecting: '',
  disconnected: ' error',
});

/**
 * The WebSocket health dot, moved out of the hardcoded status bar and into the
 * catalog. Passive by design: it reports, it is not a control, so it carries
 * no `.bar-item-btn` — a button that does nothing on click is a worse lie than
 * a dot. The skin is the status bar's own `.status-dot` (files.css), unchanged,
 * so the item reads identically wherever the operator puts it and the two
 * densities are handled entirely by the host.
 *
 * Seeds SYNCHRONOUSLY from `getConnectionState()`. A dot that waited for the
 * next transition would sit grey on a healthy session — every reconnect the
 * item missed while folded is a transition that will never be repeated.
 * @param {BarBuildContext} ctx
 * @returns {BarItemInstance}
 */
function buildConnection(ctx) {
  const doc = ctx.shell.ownerDocument;
  const body = doc.createElement('span');
  body.className = 'status-item bar-connection';
  body.setAttribute('role', 'status');
  const dot = doc.createElement('span');
  dot.className = 'status-dot';
  const label = doc.createElement('span');
  label.textContent = 'WS';
  body.append(dot, label);

  /** @param {ConnectionState} state */
  const render = (state) => {
    dot.className = `status-dot${CONNECTION_DOT_CLASS[state.ws]}`;
    body.title = `WebSocket: ${state.ws}`;
    body.setAttribute('aria-label', `WebSocket ${state.ws}`);
  };

  render(getConnectionState());
  return { node: body, dispose: onConnectionStateChange(render) };
}

defineBarItem('connection', buildConnection);

/* ---- panel health ---- */

/**
 * One health dot, built to be indistinguishable from the server's.
 *
 * `index.html` renders these dots itself for every deployment that serves a
 * declaring panel, and bar-host ADOPTS that body — so this builder runs only
 * where the server rendered no body to adopt (a second panel-health entry in
 * one layout, or a document assembled without the SSR). Both bodies therefore
 * have to be the same body: same `.status-item` skin, same id, same label, and
 * `hidden` until a health poll settles. `panel-status-bar.js` resolves the dot
 * by ITS id and nothing resolves the inner span, so the inner span carries no
 * id of its own — the same call the connection item made about `#ws-dot`.
 * @param {Document} doc
 * @param {string} id - the panel's `statusBarId`
 * @param {string} label
 * @param {HTMLElement | null} previous - the outgoing dot for this id, if any
 * @returns {HTMLElement}
 */
function healthDot(doc, id, label, previous) {
  const item = doc.createElement('div');
  item.className = 'status-item';
  item.id = id;
  // Seeded from the body being replaced, for the reason the connection item
  // seeds from `getConnectionState()`: panel-health.js settles a healthy panel
  // once and then polls every 10 s, and it exposes no state to read back. A
  // rebuilt dot that started from scratch would go dark for up to ten seconds
  // after a host move, on a panel that never stopped being healthy.
  item.hidden = previous ? previous.hidden : true;
  const dot = doc.createElement('span');
  dot.className = previous?.querySelector('.status-dot')?.className ?? 'status-dot';
  const text = doc.createElement('span');
  text.textContent = label;
  item.append(dot, text);
  return item;
}

/**
 * One dot per ENABLED built-in panel that declares a `statusBarId`.
 *
 * "Enabled" needs no fetch of its own: `panel-manager.js` filters `PANELS` in
 * place against `/api/panels` at init, so the live array IS this deployment's
 * panels. A disabled panel has no dot at all rather than a grey one — a dot for
 * a panel nobody serves could only ever sit dark. The catalog's CLOSED id set
 * is the second half: it is frozen from the SHIPPED catalog at import time, so
 * a panel registered at runtime cannot mint a dot even if it declares an id.
 *
 * Ids are unique per document, so a dot whose id is already owned by another
 * shell is SKIPPED rather than duplicated — the client half of the server's
 * "one node, one home" rule for adopted chrome, and what keeps a layout naming
 * panel-health twice from leaving `getElementById` pointing at the wrong dot.
 * An item left with nothing to show returns a hidden body rather than an empty
 * one, so bar-host's mirror collapses the shell and the bar spends no gap on
 * it.
 *
 * No subscription, and so no disposer: the dots are written from OUTSIDE, by
 * `panel-status-bar.js`, off the health poll the rail already runs. There is no
 * per-item timer to leak, and nothing this item starts has to stop when the
 * shell is parked.
 * @param {BarBuildContext} ctx
 * @returns {BarItemInstance}
 */
function buildPanelHealth(ctx) {
  const doc = ctx.shell.ownerDocument;
  const body = doc.createDocumentFragment();
  for (const panel of PANELS) {
    const id = panel.statusBarId;
    if (!id || !PANEL_HEALTH_STATUS_BAR_IDS.includes(id)) continue;
    const existing = doc.getElementById(id);
    if (existing && !ctx.shell.contains(existing)) continue;
    body.appendChild(healthDot(doc, id, panel.label, existing));
  }
  if (body.firstChild) return { node: body };
  const empty = doc.createElement('span');
  empty.hidden = true;
  return { node: empty };
}

defineBarItem('panel-health', buildPanelHealth);

/* ---- shared formatting ---- */

/**
 * Two digits, always. `9:4` is not a time.
 * @param {number} value
 * @returns {string}
 */
function pad2(value) {
  return String(value).padStart(2, '0');
}

/* ---- clock ---- */

/**
 * One tick a second at every setting, seconds option off included. A clock
 * that ticked once a minute would repaint up to 59 s after the minute it
 * claims to show; the render compares its own string first and writes nothing
 * when nothing changed, so the accurate timer is also the cheap one.
 */
const CLOCK_TICK_MS = 1000;

/** What the readout is, in one phrase, per zone. */
const CLOCK_TITLE = Object.freeze({
  local: 'Local time',
  utc: 'UTC',
  both: 'Local · UTC',
});

/**
 * The zone option, narrowed. An unknown or absent value is local time — the
 * catalog default, and the only answer that is never actively wrong.
 * @param {BarItemOptions} options
 * @returns {'local' | 'utc' | 'both'}
 */
function clockZone(options) {
  const zone = options.zone;
  return zone === 'utc' || zone === 'both' ? zone : 'local';
}

/**
 * The local zone, shortened for a bar: `Europe/Berlin` reads as `Berlin`. An
 * empty string where `Intl` refuses — an unlabelled local clock is still a
 * correct clock, and a label reading `undefined` is not.
 * @returns {string}
 */
function localZoneLabel() {
  try {
    const zone = Intl.DateTimeFormat().resolvedOptions().timeZone;
    if (!zone) return '';
    return (zone.split('/').at(-1) ?? zone).replace(/_/g, ' ');
  } catch {
    return '';
  }
}

/**
 * `HH:MM`, or `HH:MM:SS` with the seconds option on.
 * @param {Date} now
 * @param {boolean} utc
 * @param {boolean} seconds
 * @returns {string}
 */
function formatClock(now, utc, seconds) {
  const hours = utc ? now.getUTCHours() : now.getHours();
  const minutes = utc ? now.getUTCMinutes() : now.getMinutes();
  const secs = utc ? now.getUTCSeconds() : now.getSeconds();
  return `${pad2(hours)}:${pad2(minutes)}${seconds ? `:${pad2(secs)}` : ''}`;
}

/**
 * The wall clock, moved out of `initStatusBar()`'s hardcoded 1 s interval and
 * into the catalog, where it gains the zone and seconds options that interval
 * could not express.
 *
 * `role="timer"` rather than the connection item's `role="status"`: both are
 * live regions, but `timer` is implicitly `aria-live="off"`, and a status
 * region that re-announced the time once a second would make a screen reader
 * unusable. The value stays in the body's TEXT, with no `aria-label` over it,
 * so the accessible name is the readout itself (`14:32 UTC`) rather than a
 * description of it.
 *
 * Density is the zone suffix. A local clock in the 20 px status bar is just
 * "the time" and the suffix is noise; a UTC or dual clock keeps its label at
 * both densities, because an unmarked UTC readout is not terse, it is wrong.
 * @param {BarBuildContext} ctx
 * @returns {BarItemInstance}
 */
function buildClock(ctx) {
  const doc = ctx.shell.ownerDocument;
  const zone = clockZone(ctx.options);
  const seconds = ctx.options.seconds === true;

  const body = doc.createElement('span');
  body.className = 'status-item bar-clock';
  body.setAttribute('role', 'timer');
  body.title = CLOCK_TITLE[zone];
  const time = doc.createElement('span');
  time.className = 'bar-clock-time';
  body.appendChild(time);

  const zoneLabel = zone === 'local' ? localZoneLabel() : 'UTC';
  if (zoneLabel && (zone !== 'local' || ctx.density === 'comfortable')) {
    const label = doc.createElement('span');
    label.className = 'bar-clock-zone';
    label.textContent = zoneLabel;
    body.appendChild(label);
  }

  const render = () => {
    const now = new Date();
    const text =
      zone === 'both'
        ? `${formatClock(now, false, seconds)} · ${formatClock(now, true, seconds)}`
        : formatClock(now, zone === 'utc', seconds);
    if (time.textContent !== text) time.textContent = text;
  };

  render();
  const timer = setInterval(render, CLOCK_TICK_MS);
  return { node: body, dispose: () => clearInterval(timer) };
}

defineBarItem('clock', buildClock);

/* ---- terminal size ---- */

/**
 * The old `initStatusBar()` cadence, kept. `terminal.js` owns xterm's
 * `onResize` and exposes no subscription — only `getTerminalDimensions()` —
 * so a poll is the whole of what is on offer today. Half a second is fast
 * enough to look live while a panel is dragged and slow enough to be free.
 *
 * Task 1.13 adds the `getTerminalDimensions` WINDOW seam (for
 * `docs/screenshots/contact_sheet.py`, which reads the fitted size out of the
 * now-deleted `#term-dims`). That is a second reader of the same getter, not a
 * replacement for this import: when a real resize subscription appears on
 * `terminal.js`, this poll becomes `onTerminalResize(render)` and the disposer
 * it hands back goes where `clearInterval` is now.
 */
const TERMINAL_SIZE_POLL_MS = 500;

/**
 * The fitted `cols×rows` readout.
 *
 * No live region at all, unlike the clock and the connection dot: the value
 * changes on every frame of a panel drag, and an `aria-live` region would turn
 * one resize into a burst of announcements. The size is a fact to be read, not
 * an event to be told about, so it sits in plain text with the spelled-out
 * form in `title`.
 *
 * An em dash before the terminal is up. The item may legitimately be placed on
 * a page whose terminal never initialises, and a blank body would read as a
 * broken item rather than as "nothing to report".
 * @param {BarBuildContext} ctx
 * @returns {BarItemInstance}
 */
function buildTerminalSize(ctx) {
  const doc = ctx.shell.ownerDocument;
  const body = doc.createElement('span');
  body.className = 'status-item bar-terminal-size';

  // Density is the standing label: 28 px of header has room to say what the
  // number is, 20 px of status bar does not, and `80x24` is self-describing.
  if (ctx.density === 'comfortable') {
    const label = doc.createElement('span');
    label.className = 'bar-terminal-size-label';
    label.textContent = 'Term';
    body.appendChild(label);
  }
  const value = doc.createElement('span');
  value.className = 'bar-terminal-size-value';
  body.appendChild(value);

  let shown = '';
  const render = () => {
    const dims = getTerminalDimensions();
    const text = dims ? `${dims.cols}×${dims.rows}` : '—';
    if (text === shown) return;
    shown = text;
    value.textContent = text;
    body.title = dims
      ? `Terminal size: ${dims.cols} columns × ${dims.rows} rows`
      : 'Terminal size: not ready';
  };

  render();
  const timer = setInterval(render, TERMINAL_SIZE_POLL_MS);
  return { node: body, dispose: () => clearInterval(timer) };
}

defineBarItem('terminal-size', buildTerminalSize);

/* ---- stopwatch ---- */

/**
 * One stopwatch's reading. `startedAt` is the epoch ms of the current run and
 * null while stopped; `accumulated` is everything measured before it. Elapsed
 * is derived from the clock on every render rather than counted up by the
 * interval, so a throttled background tab resumes with the right number
 * instead of the number of ticks the browser felt like delivering.
 * @typedef {object} StopwatchState
 * @property {number | null} startedAt
 * @property {number} accumulated
 */

/** Elapsed time per item key. See the module header. @type {Map<string, StopwatchState>} */
const stopwatches = new Map();

const STOPWATCH_TICK_MS = 1000;

/**
 * This item's reading, created on first sight. Keyed by `ctx.key` — bar-host's
 * `data-bar-key`, which is stable across pool round-trips and host moves —
 * and NOT by the shell, which is per-mount by design and would reset the count
 * on every fold.
 * @param {string} key
 * @returns {StopwatchState}
 */
function stopwatchState(key) {
  let state = stopwatches.get(key);
  if (!state) {
    state = { startedAt: null, accumulated: 0 };
    stopwatches.set(key, state);
  }
  return state;
}

/**
 * @param {StopwatchState} state
 * @returns {number} milliseconds measured so far
 */
function stopwatchElapsed(state) {
  return state.accumulated + (state.startedAt === null ? 0 : Date.now() - state.startedAt);
}

/**
 * `MM:SS`, growing an hours field only once there are hours to show. A leading
 * `00:` on every reading is two characters of bar spent on nothing.
 * @param {number} ms
 * @returns {string}
 */
function formatElapsed(ms) {
  const total = Math.max(0, Math.floor(ms / 1000));
  const minutesSeconds = `${pad2(Math.floor(total / 60) % 60)}:${pad2(total % 60)}`;
  const hours = Math.floor(total / 3600);
  return hours > 0 ? `${hours}:${minutesSeconds}` : minutesSeconds;
}

/**
 * A click-to-start, click-to-stop, right-click-to-reset stopwatch.
 *
 * The only bar item that is a real control, so it is the only one wearing
 * `.bar-item-btn`, and being a toggle it carries `aria-pressed` rather than a
 * label that has to be re-read to learn whether it is running.
 *
 * The click handler does NOT stop propagation. The mock did, because in the
 * mock every chip owned a popover; here, swallowing the event would stop the
 * document-level outside-click handlers that other items rely on to close
 * THEIR popovers, so clicking the stopwatch would leave a menu open.
 *
 * The interval runs only while the stopwatch does. A stopped stopwatch has
 * nothing to repaint, and the disposer stops the ticker either way — folding a
 * RUNNING stopwatch parks the body and stops its timer, while the reading
 * itself keeps accruing in `stopwatches`, because it is measured from the wall
 * clock and not from the ticks.
 * @param {BarBuildContext} ctx
 * @returns {BarItemInstance}
 */
function buildStopwatch(ctx) {
  const doc = ctx.shell.ownerDocument;
  const state = stopwatchState(ctx.key);

  const chip = doc.createElement('button');
  chip.type = 'button';
  chip.className = 'bar-item-btn bar-stopwatch';
  const icon = doc.createElement('span');
  icon.className = 'bar-stopwatch-icon';
  icon.setAttribute('aria-hidden', 'true');
  const value = doc.createElement('span');
  value.className = 'bar-stopwatch-time';
  chip.append(icon, value);

  /** @type {ReturnType<typeof setInterval> | null} */
  let timer = null;
  const stopTicking = () => {
    if (timer === null) return;
    clearInterval(timer);
    timer = null;
  };

  const render = () => {
    const running = state.startedAt !== null;
    const text = formatElapsed(stopwatchElapsed(state));
    if (value.textContent !== text) value.textContent = text;
    icon.textContent = running ? '■' : '▶';
    chip.setAttribute('aria-pressed', String(running));
    chip.setAttribute('aria-label', `Stopwatch ${text}, ${running ? 'running' : 'stopped'}`);
    chip.title = running
      ? 'Running — click to stop'
      : state.accumulated > 0
        ? 'Paused — click to resume, right-click to reset'
        : 'Click to start';
  };

  const sync = () => {
    render();
    if (state.startedAt === null) stopTicking();
    else if (timer === null) timer = setInterval(render, STOPWATCH_TICK_MS);
  };

  chip.addEventListener('click', () => {
    if (state.startedAt === null) {
      state.startedAt = Date.now();
    } else {
      state.accumulated = stopwatchElapsed(state);
      state.startedAt = null;
    }
    sync();
  });

  chip.addEventListener('contextmenu', (event) => {
    event.preventDefault();
    state.startedAt = null;
    state.accumulated = 0;
    sync();
  });

  sync();
  return { node: chip, dispose: stopTicking };
}

defineBarItem('stopwatch', buildStopwatch);
