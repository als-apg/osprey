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
 * A bar item is not a static body. The clock runs an interval, the stopwatch
 * counts while it runs, the system-health chip follows its poll.
 * Every one of them is ATTACH-SCOPED: it must stop the moment its shell leaves
 * the bar, because the host does not destroy a folded item — it parks the node
 * in the hidden pool, alive, where a live subscription would keep updating a
 * body nobody can see, for the rest of the page's life.
 *
 * So the house rule, binding on every item: EVERY ATTACH-SCOPED SUBSCRIPTION
 * COMES FROM A DISPOSER-RETURNING API. A builder does not register a listener
 * or arm a timer and hope something later remembers it; it is handed the way
 * to stop at the moment it starts, and it returns that disposer with its
 * body. An item that cannot hand back a disposer has not finished being
 * written.
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
import { defaultOptions } from './bar-catalog.js';

/** @typedef {import('./bar-host.js').BarBuildContext} BarBuildContext */
/** @typedef {import('./bar-host.js').BarRoot} BarRoot */
/** @typedef {import('./bar-catalog.js').BarItemOptions} BarItemOptions */

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

/** Every registered factory, by type — what {@link previewBarItem} builds from. */
/** @type {Map<string, BarItemFactory>} */
const factories = new Map();

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
  factories.set(type, factory);
  const unregister = registerItemBuilder(type, (ctx) => {
    disposeShell(ctx.shell);
    const instance = factory(ctx);
    if (instance.dispose) instances.set(ctx.shell, instance.dispose);
    watchPool(ctx.shell.ownerDocument);
    return instance.node;
  });
  return () => {
    unregister();
    if (factories.get(type) === factory) factories.delete(type);
    for (const shell of Array.from(instances.keys())) {
      if (shell.dataset.barItem === type) disposeShell(shell);
    }
  };
}

/**
 * Build one type's body OUTSIDE the bars, for the customize sheet's tiles: the
 * same factory, at the type's default options, so a tile shows the item as it
 * will look rather than a name for it. The instance is the caller's — it is
 * not registered against a shell, so the pool watcher never sees it, and the
 * returned `dispose` is the only thing that stops it.
 * @param {string} type
 * @param {Document} doc
 * @param {import('./bar-catalog.js').BarDensity} density
 * @returns {BarItemInstance | null} null when no factory renders the type
 */
export function previewBarItem(type, doc, density) {
  const factory = factories.get(type);
  if (!factory) return null;
  const shell = doc.createElement('div');
  shell.className = 'bar-item';
  shell.dataset.barItem = type;
  const instance = factory({
    type,
    key: `preview:${type}`,
    density,
    options: defaultOptions(type),
    shell,
  });
  return { node: instance.node, dispose: instance.dispose ?? (() => {}) };
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
  none: 'Local time',
  local: 'Local time',
  utc: 'UTC',
  both: 'Local · UTC',
});

/**
 * The zone option, narrowed. An unknown or absent value is the plain local
 * clock — the catalog default, and the only answer that is never actively
 * wrong. `none` and `local` show the same time; `local` adds the zone's name.
 * @param {BarItemOptions} options
 * @returns {'none' | 'local' | 'utc' | 'both'}
 */
function clockZone(options) {
  const zone = options.zone;
  return zone === 'local' || zone === 'utc' || zone === 'both' ? zone : 'none';
}

/**
 * The format option, narrowed: 12-hour with AM/PM only when asked for.
 * @param {BarItemOptions} options
 * @returns {boolean} whether to render the 12-hour cycle
 */
function clockHour12(options) {
  return options.format === '12h';
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
 * `HH:MM`, or `HH:MM:SS` with the seconds option on; `H:MM AM` on the
 * 12-hour cycle, where the hour drops its leading zero the way a wall clock
 * does and the meridiem carries what the missing 13–23 would have said.
 * @param {Date} now
 * @param {boolean} utc
 * @param {boolean} seconds
 * @param {boolean} hour12
 * @returns {string}
 */
function formatClock(now, utc, seconds, hour12) {
  const hours = utc ? now.getUTCHours() : now.getHours();
  const minutes = utc ? now.getUTCMinutes() : now.getMinutes();
  const secs = utc ? now.getUTCSeconds() : now.getSeconds();
  const hour = hour12 ? String(hours % 12 || 12) : pad2(hours);
  const meridiem = hour12 ? (hours < 12 ? ' AM' : ' PM') : '';
  return `${hour}:${pad2(minutes)}${seconds ? `:${pad2(secs)}` : ''}${meridiem}`;
}

/**
 * The wall clock, moved out of `initStatusBar()`'s hardcoded 1 s interval and
 * into the catalog, where it gains the zone and seconds options that interval
 * could not express.
 *
 * `role="timer"` rather than `role="status"`: both are
 * live regions, but `timer` is implicitly `aria-live="off"`, and a status
 * region that re-announced the time once a second would make a screen reader
 * unusable. The value stays in the body's TEXT, with no `aria-label` over it,
 * so the accessible name is the readout itself (`14:32 UTC`) rather than a
 * description of it.
 *
 * The suffix follows the zone option. `none`, the default, is the plain
 * clock and never carries one. `local` names the zone, but only at
 * comfortable density: in the 20 px status bar a local clock is just "the
 * time" and the name is noise. A UTC or dual clock keeps its label at both
 * densities, because an unmarked UTC readout is not terse, it is wrong.
 * @param {BarBuildContext} ctx
 * @returns {BarItemInstance}
 */
function buildClock(ctx) {
  const doc = ctx.shell.ownerDocument;
  const zone = clockZone(ctx.options);
  const seconds = ctx.options.seconds === true;
  const hour12 = clockHour12(ctx.options);

  const body = doc.createElement('span');
  body.className = 'status-item bar-clock';
  body.setAttribute('role', 'timer');
  body.title = CLOCK_TITLE[zone];
  const time = doc.createElement('span');
  time.className = 'bar-clock-time';
  body.appendChild(time);

  const zoneLabel = zone === 'none' ? '' : zone === 'local' ? localZoneLabel() : 'UTC';
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
        ? `${formatClock(now, false, seconds, hour12)} · ${formatClock(now, true, seconds, hour12)}`
        : formatClock(now, zone === 'utc', seconds, hour12);
    if (time.textContent !== text) time.textContent = text;
  };

  render();
  const timer = setInterval(render, CLOCK_TICK_MS);
  return { node: body, dispose: () => clearInterval(timer) };
}

defineBarItem('clock', buildClock);

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

/* ---- feedback ---- */

/** The rail's own feedback control, which feedback-boot.js binds the dialog to. */
const FEEDBACK_RAIL_BUTTON_ID = 'panel-feedback-btn';

/** The rail's speech-bubble glyph, so the item and the rail read as one control. */
const FEEDBACK_ICON_PATH =
  'M4.6 2.7H11.4A2.1 2.1 0 0 1 13.5 4.8V9.3A2.1 2.1 0 0 1 11.4 11.4H7.0L3.5 14.1' +
  'L4.4 11.4A2.1 2.1 0 0 1 2.5 9.3V4.8A2.1 2.1 0 0 1 4.6 2.7Z';

const SVG_NS = 'http://www.w3.org/2000/svg';

/**
 * The feedback button.
 *
 * The dialog belongs to the rail's `#panel-feedback-btn` — feedback-boot.js
 * binds it there and nowhere else, and the rail control is on every page
 * whether or not the rail shows it. So this item is a second way to press
 * that button, not a second dialog: a click here is forwarded to the rail.
 *
 * Icon and label at header density; the label alone in the status bar, where
 * the docs link beside it is a bare word too.
 * @param {BarBuildContext} ctx
 * @returns {BarItemInstance}
 */
function buildFeedback(ctx) {
  const doc = ctx.shell.ownerDocument;
  const chip = doc.createElement('button');
  chip.type = 'button';
  chip.className = 'bar-item-btn bar-feedback';
  chip.title = 'Send feedback';
  if (ctx.density === 'comfortable') {
    const svg = doc.createElementNS(SVG_NS, 'svg');
    svg.setAttribute('class', 'bar-feedback-icon');
    svg.setAttribute('viewBox', '0 0 16 16');
    svg.setAttribute('aria-hidden', 'true');
    const path = doc.createElementNS(SVG_NS, 'path');
    path.setAttribute('d', FEEDBACK_ICON_PATH);
    svg.appendChild(path);
    chip.appendChild(svg);
  }
  const label = doc.createElement('span');
  label.textContent = 'Feedback';
  chip.appendChild(label);
  chip.addEventListener('click', () => {
    doc.getElementById(FEEDBACK_RAIL_BUTTON_ID)?.click();
  });
  return { node: chip };
}

defineBarItem('feedback', buildFeedback);

/* ---- space ---- */

/**
 * The space's edit-mode furniture: what it is set to, and the grip that sets
 * it. Nothing shows outside edit mode — bars.css hides both there, and a
 * space is then exactly the empty stretch it always was. The width is
 * rendered from the option so a rebuild after a resize shows the stored
 * value; the drag gesture (bar-customize-drag.js) writes the label live while
 * the pointer is down and commits through the option when it lifts.
 *
 * Three bodies, not one: the label's box clips its own overflow (a narrow
 * space must not spill "1200 px" over its neighbour), and the two grips hang
 * half outside the shell's edges — so they cannot live inside that clipped
 * box, or the half the pointer reaches for would be cut away with them. A
 * grip at EACH end, because which end reads as "the edge of the space"
 * depends on what the space sits next to: an operator narrowing the gap
 * before the search bar reaches for the end nearest the items they are
 * moving. Either end sets the same width.
 * @param {BarBuildContext} ctx
 * @returns {BarItemInstance}
 */
function buildSpace(ctx) {
  const doc = ctx.shell.ownerDocument;
  const width = ctx.options.width;
  const body = doc.createDocumentFragment();
  const readout = doc.createElement('span');
  readout.className = 'bar-space';
  const label = doc.createElement('span');
  label.className = 'bar-space-label';
  label.textContent = spaceLabel(typeof width === 'number' ? width : 0);
  readout.appendChild(label);
  body.appendChild(readout);
  for (const edge of ['start', 'end']) {
    const grip = doc.createElement('span');
    grip.className = 'bar-space-grip';
    grip.dataset.edge = edge;
    grip.title = 'Drag to set the width';
    grip.setAttribute('aria-hidden', 'true');
    body.appendChild(grip);
  }
  return { node: body };
}

/**
 * How a space names its setting: the fill glyph, or a width.
 * @param {number} width - 0 for the flexible space
 * @returns {string}
 */
export function spaceLabel(width) {
  return width > 0 ? `${Math.round(width)} px` : '⟷';
}

defineBarItem('space', buildSpace);
