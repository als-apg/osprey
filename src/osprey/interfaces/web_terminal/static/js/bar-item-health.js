/* OSPREY Web Terminal — The system-health bar item.
 *
 * One item, in its own module like the Bluesky queue: a dot for the worst
 * outcome across the health checks the SYSTEM panel runs, with a card that
 * lists them and opens the panel. It registers through `defineBarItem()`
 * exactly as the items in bar-items.js do and follows the same house rule:
 * every subscription comes from a disposer-returning API.
 *
 * Imported for its side effect by app.js, right after bar-items.js, so the
 * type has a builder before the first reconcile asks for one.
 */

import { withPrefix } from './api.js';
import { registerBarPopover } from './bar-host.js';
import { defineBarItem } from './bar-items.js';

/** @typedef {import('./bar-host.js').BarBuildContext} BarBuildContext */
/** @typedef {import('./bar-items.js').BarItemInstance} BarItemInstance */

/**
 * The SYSTEM panel's id, and so the proxy the report is read through:
 * `/panel/system-health/checks` is the terminal's own route (routes/proxy.py),
 * which forwards to the health sidecar. The item is offered exactly where that
 * panel is enabled (the catalog's `systemHealthAvailable`), so a page holding
 * this body always has the route.
 */
const HEALTH_PANEL_ID = 'system-health';
const HEALTH_API = `/panel/${HEALTH_PANEL_ID}/checks`;

/** Poll cadence when the envelope names none, seconds — the sidecar's own default. */
const HEALTH_DEFAULT_INTERVAL_S = 60;
/** Poll cadence while the sidecar's first suite is still running, seconds. */
const HEALTH_WARMING_INTERVAL_S = 3;
/** Floor on the cadence, so a misconfigured interval cannot turn into a busy loop. */
const HEALTH_MIN_INTERVAL_S = 5;

/**
 * Severity order of a check's status, worst last. The same table the SYSTEM
 * dashboard keeps beside itself (`helpers.js`), restated here because that
 * bundle is served through the panel proxy, not to this page.
 * @type {Readonly<Record<string, number>>}
 */
const STATUS_RANK = Object.freeze({ ok: 0, skip: 1, warning: 2, error: 3 });

/**
 * One check as the envelope carries it.
 * @typedef {object} HealthCheck
 * @property {string} name
 * @property {string} category
 * @property {string} status - `ok` | `warning` | `error` | `skip`
 * @property {string} message
 * @property {string} [value]
 */

/**
 * The report as the last poll described it.
 * @typedef {object} HealthSnapshot
 * @property {HealthCheck[]} results
 * @property {string} summary - the sidecar's one-line summary
 * @property {boolean} warming - the first suite has not finished yet
 * @property {boolean} stale - the cached report is older than one interval
 * @property {number} intervalS - the sidecar's refresh cadence
 * @property {boolean} reachable - the last poll got an envelope back
 * @property {boolean} seen - at least one poll has completed this page
 * @property {number} receivedAt - epoch ms of the last envelope, 0 for none
 */

/** @type {HealthSnapshot} */
let healthSnapshot = emptySnapshot();

function emptySnapshot() {
  return {
    results: [],
    summary: '',
    warming: false,
    stale: false,
    intervalS: HEALTH_DEFAULT_INTERVAL_S,
    reachable: false,
    seen: false,
    receivedAt: 0,
  };
}

/**
 * Everyone reading the report on this page. ONE poller serves them all: the
 * item is a singleton, but the customize sheet builds a second, previewing
 * instance while it is open, and two pollers for one fact is one too many.
 * @type {Set<() => void>}
 */
const healthListeners = new Set();
/** @type {ReturnType<typeof setTimeout> | null} */
let healthTimer = null;
/** @type {AbortController | null} */
let healthInFlight = null;

function notifyHealth() {
  for (const listener of healthListeners) {
    try {
      listener();
    } catch (err) {
      console.error('[bar-items] system-health listener threw', err);
    }
  }
}

/**
 * Read one envelope into the snapshot. Anything not in the shape the sidecar
 * documents reads as its absence, never as a throw.
 * @param {unknown} raw
 */
function applyHealthEnvelope(raw) {
  const frame = raw && typeof raw === 'object' ? /** @type {Record<string, any>} */ (raw) : {};
  const results = Array.isArray(frame.results)
    ? frame.results.filter(
        /** @param {unknown} r @returns {r is HealthCheck} */
        (r) => Boolean(r) && typeof r === 'object' && typeof (/** @type {any} */ (r).status) === 'string'
      )
    : [];
  const interval = Number(frame.interval_s);
  healthSnapshot = {
    results,
    summary: typeof frame.summary === 'string' ? frame.summary : '',
    warming: frame.warming === true,
    stale: frame.stale === true,
    intervalS: Number.isFinite(interval) && interval > 0 ? interval : HEALTH_DEFAULT_INTERVAL_S,
    reachable: true,
    seen: true,
    receivedAt: Date.now(),
  };
}

/** Seconds until the next poll, from what the last envelope said. */
function nextHealthDelayS() {
  if (!healthSnapshot.reachable) return HEALTH_DEFAULT_INTERVAL_S;
  if (healthSnapshot.warming) return HEALTH_WARMING_INTERVAL_S;
  return Math.max(HEALTH_MIN_INTERVAL_S, Math.round(healthSnapshot.intervalS));
}

function scheduleHealthPoll() {
  if (healthTimer !== null || healthListeners.size === 0) return;
  healthTimer = setTimeout(() => {
    healthTimer = null;
    void pollHealth();
  }, nextHealthDelayS() * 1000);
}

async function pollHealth() {
  if (healthInFlight || typeof fetch === 'undefined') return;
  const controller = typeof AbortController === 'undefined' ? null : new AbortController();
  healthInFlight = controller;
  try {
    const response = await fetch(withPrefix(HEALTH_API), {
      cache: 'no-store',
      signal: controller?.signal,
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    applyHealthEnvelope(await response.json());
  } catch {
    if (controller?.signal.aborted) return;
    healthSnapshot = { ...healthSnapshot, reachable: false, seen: true };
  } finally {
    if (healthInFlight === controller) healthInFlight = null;
  }
  notifyHealth();
  scheduleHealthPoll();
}

function stopHealthPolling() {
  if (healthTimer !== null) {
    clearTimeout(healthTimer);
    healthTimer = null;
  }
  if (healthInFlight) {
    healthInFlight.abort();
    healthInFlight = null;
  }
}

/**
 * Follow the report. The poller starts with the first listener and stops with
 * the last, so a page with the item folded away asks the sidecar nothing.
 * @param {() => void} listener
 * @returns {() => void} dispose
 */
function subscribeHealth(listener) {
  healthListeners.add(listener);
  if (healthListeners.size === 1) void pollHealth();
  return () => {
    healthListeners.delete(listener);
    if (healthListeners.size === 0) stopHealthPolling();
  };
}

/**
 * The worst status in a list of checks, `ok` for none.
 * @param {readonly HealthCheck[]} checks
 * @returns {string}
 */
function worstOf(checks) {
  let worst = 'ok';
  for (const check of checks) {
    if ((STATUS_RANK[check.status] ?? 0) > (STATUS_RANK[worst] ?? 0)) worst = check.status;
  }
  return worst;
}

/**
 * A status as a dot tone.
 * @param {string} status
 * @returns {'off' | 'ok' | 'warn' | 'err'}
 */
function toneOf(status) {
  if (status === 'error') return 'err';
  if (status === 'warning') return 'warn';
  if (status === 'ok') return 'ok';
  return 'off';
}

/**
 * One reading of the snapshot for the chip and the card: a tone for the dot
 * and a word for the state.
 * @typedef {object} HealthReading
 * @property {'off' | 'ok' | 'warn' | 'err'} tone
 * @property {string} word
 */

/**
 * @param {HealthSnapshot} snap
 * @returns {HealthReading}
 */
function readHealth(snap) {
  if (!snap.seen) return { tone: 'off', word: 'checking' };
  if (!snap.reachable) return { tone: 'err', word: 'unreachable' };
  if (snap.warming) return { tone: 'off', word: 'warming' };
  // A stale report is still the report: the sidecar flags one as stale for the
  // seconds between its refresh-ahead kick and the suite finishing, on every
  // cycle, and a dot that went grey each minute would say nothing. The card's
  // note carries the staleness instead.
  const counted = snap.results.filter((r) => r.status !== 'skip');
  if (counted.length === 0) return { tone: 'off', word: 'no checks' };
  const errors = counted.filter((r) => r.status === 'error').length;
  if (errors) return { tone: 'err', word: `${errors} error${errors === 1 ? '' : 's'}` };
  const warnings = counted.filter((r) => r.status === 'warning').length;
  if (warnings) return { tone: 'warn', word: `${warnings} warning${warnings === 1 ? '' : 's'}` };
  return { tone: 'ok', word: 'ok' };
}

/**
 * Humanize a check name: drop the leading `category.` and title-case the
 * rest, so `epics.beam_current` reads `Beam Current`. The dashboard's rule.
 * @param {string} name
 * @returns {string}
 */
function checkTitle(name) {
  const dot = name.indexOf('.');
  const bare = dot > -1 ? name.slice(dot + 1) : name;
  return bare.replace(/_/g, ' ').replace(/\b[a-z]/g, (c) => c.toUpperCase());
}

/**
 * The checks grouped by category, first-seen order kept.
 * @param {readonly HealthCheck[]} checks
 * @returns {Map<string, HealthCheck[]>}
 */
function byCategory(checks) {
  /** @type {Map<string, HealthCheck[]>} */
  const groups = new Map();
  for (const check of checks) {
    const key = typeof check.category === 'string' ? check.category : '';
    const bucket = groups.get(key);
    if (bucket) bucket.push(check);
    else groups.set(key, [check]);
  }
  return groups;
}

/**
 * The time of the last envelope, as the card's note says it.
 * @param {number} epochMs
 * @returns {string}
 */
function clockOf(epochMs) {
  const date = new Date(epochMs);
  const pad = (/** @type {number} */ n) => String(n).padStart(2, '0');
  return `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
}

/**
 * The system-health item: the worst outcome across the SYSTEM panel's checks
 * as one dot, and — on a click — the checks themselves.
 *
 * The chip is a dot, the panel's name at the header's density, and the
 * outcome in a word where `text` asks for it. It never acts on a click: it
 * opens. The card lists one row per category (`detail: categories`) or every
 * check (`detail: checks`), each with its own dot, and ends in **Open
 * SYSTEM**, which is the whole dashboard.
 *
 * The poll is shared (see `subscribeHealth`) and the subscription is
 * attach-scoped: parking the shell stops it, placing the shell back restarts
 * it and repaints from the envelope that then arrives.
 * @param {BarBuildContext} ctx
 * @returns {BarItemInstance}
 */
function buildSystemHealth(ctx) {
  const doc = ctx.shell.ownerDocument;
  const showStatus = ctx.options.text === 'status';
  const listChecks = ctx.options.detail === 'checks';
  const showName = ctx.density === 'comfortable';

  const body = doc.createDocumentFragment();
  const chip = doc.createElement('button');
  chip.type = 'button';
  chip.className = 'bar-item-btn bar-health';
  chip.setAttribute('aria-haspopup', 'true');
  chip.setAttribute('aria-expanded', 'false');
  const dot = doc.createElement('span');
  dot.className = 'bar-health-dot';
  dot.setAttribute('aria-hidden', 'true');
  chip.appendChild(dot);
  if (showName) {
    const name = doc.createElement('span');
    name.className = 'bar-health-name';
    name.textContent = 'System';
    chip.appendChild(name);
  }
  const text = doc.createElement('span');
  text.className = 'bar-health-text';
  text.hidden = !showStatus;
  chip.appendChild(text);
  body.appendChild(chip);

  const pop = doc.createElement('div');
  pop.className = 'bar-pop bar-health-pop';
  pop.hidden = true;
  pop.setAttribute('role', 'dialog');
  pop.setAttribute('aria-label', 'System health');
  body.appendChild(pop);

  let open = false;

  const renderChip = () => {
    const reading = readHealth(healthSnapshot);
    dot.dataset.tone = reading.tone;
    if (showStatus && text.textContent !== reading.word) text.textContent = reading.word;
    const parts = [`System health: ${reading.word}`];
    if (healthSnapshot.summary) parts.push(healthSnapshot.summary);
    chip.title = parts.join(' · ');
    chip.setAttribute('aria-label', chip.title);
  };

  /**
   * @param {string} tone
   * @param {string} name
   * @param {string} count
   * @param {string} aside
   */
  const row = (tone, name, count, aside) => {
    const line = doc.createElement('div');
    line.className = 'bar-health-row';
    const mark = doc.createElement('span');
    mark.className = 'bar-health-dot';
    mark.dataset.tone = tone;
    mark.setAttribute('aria-hidden', 'true');
    const label = doc.createElement('span');
    label.className = 'bar-health-row-name';
    label.textContent = name;
    const tally = doc.createElement('span');
    tally.className = 'bar-health-row-count';
    tally.textContent = count;
    const note = doc.createElement('span');
    note.className = 'bar-health-row-aside';
    note.textContent = aside;
    note.title = aside;
    line.append(mark, label, tally, note);
    return line;
  };

  const renderPop = () => {
    if (!open) return;
    const snap = healthSnapshot;
    const reading = readHealth(snap);
    pop.replaceChildren();
    const eyebrow = doc.createElement('div');
    eyebrow.className = 'bar-pop-eyebrow';
    eyebrow.textContent = 'System health';
    const title = doc.createElement('div');
    title.className = 'bar-health-pop-title';
    const titleDot = doc.createElement('span');
    titleDot.className = 'bar-health-dot';
    titleDot.dataset.tone = reading.tone;
    titleDot.setAttribute('aria-hidden', 'true');
    const titleWord = doc.createElement('span');
    titleWord.textContent = reading.word;
    title.append(titleDot, titleWord);
    if (snap.summary && snap.reachable) {
      const summary = doc.createElement('span');
      summary.className = 'bar-health-pop-summary';
      summary.textContent = snap.summary;
      title.appendChild(summary);
    }
    pop.append(eyebrow, title);

    const rows = doc.createElement('div');
    rows.className = 'bar-health-rows';
    if (snap.reachable && !snap.warming && snap.results.length > 0) {
      if (listChecks) {
        for (const check of snap.results) {
          rows.appendChild(
            row(
              toneOf(check.status),
              checkTitle(String(check.name ?? '')),
              '',
              typeof check.value === 'string' && check.value ? check.value : String(check.message ?? '')
            )
          );
        }
      } else {
        for (const [category, checks] of byCategory(snap.results)) {
          const counted = checks.filter((c) => c.status !== 'skip');
          const passed = counted.filter((c) => c.status === 'ok').length;
          const worst = worstOf(checks);
          const loudest = checks.find((c) => c.status === worst && worst !== 'ok');
          rows.appendChild(
            row(
              toneOf(worst),
              category.replace(/_/g, ' '),
              counted.length ? `${passed}/${counted.length}` : 'skipped',
              loudest ? String(loudest.message ?? '') : ''
            )
          );
        }
      }
    }
    if (rows.childElementCount > 0) pop.appendChild(rows);

    const note = doc.createElement('div');
    note.className = 'bar-pop-note bar-health-note';
    if (!snap.seen) note.textContent = 'Checking…';
    else if (!snap.reachable) note.textContent = 'The SYSTEM panel could not be reached.';
    else if (snap.warming) note.textContent = 'First scan in progress…';
    else if (snap.stale) note.textContent = `Data may be stale · read ${clockOf(snap.receivedAt)}`;
    else note.textContent = `Read ${clockOf(snap.receivedAt)}`;
    pop.appendChild(note);

    const foot = doc.createElement('div');
    foot.className = 'bar-pop-foot';
    const openPanel = doc.createElement('button');
    openPanel.type = 'button';
    openPanel.className = 'bar-btn';
    openPanel.textContent = 'Open SYSTEM';
    openPanel.addEventListener('click', (event) => {
      event.stopPropagation();
      close();
      void import('./panel-manager.js').then((manager) => manager.showPanel(HEALTH_PANEL_ID));
    });
    foot.appendChild(openPanel);
    pop.appendChild(foot);
  };

  /** @param {Event} event */
  const onDocumentClick = (event) => {
    const target = /** @type {Node | null} */ (event.target);
    if (target && (chip.contains(target) || pop.contains(target))) return;
    close();
  };
  /** @param {KeyboardEvent} event */
  const onKeydown = (event) => {
    if (event.key === 'Escape') close();
  };

  function close() {
    if (!open) return;
    open = false;
    pop.hidden = true;
    pop.replaceChildren();
    chip.setAttribute('aria-expanded', 'false');
    doc.removeEventListener('click', onDocumentClick, true);
    doc.removeEventListener('keydown', onKeydown);
  }

  chip.addEventListener('click', (event) => {
    event.stopPropagation();
    if (open) {
      close();
      return;
    }
    open = true;
    pop.hidden = false;
    chip.setAttribute('aria-expanded', 'true');
    // Left half of the window: hang the card off the left edge instead, the
    // same rule the options popover applies to itself.
    const rect = chip.getBoundingClientRect();
    const view = doc.defaultView;
    pop.classList.toggle('is-left', Boolean(view) && rect.left < (view?.innerWidth ?? 0) / 2);
    renderPop();
    doc.addEventListener('click', onDocumentClick, true);
    doc.addEventListener('keydown', onKeydown);
  });

  const unsubscribe = subscribeHealth(() => {
    renderChip();
    renderPop();
  });
  const unregister = registerBarPopover(ctx.shell, close);
  renderChip();

  return {
    node: body,
    dispose: () => {
      close();
      unregister();
      unsubscribe();
    },
  };
}

defineBarItem('system-health', buildSystemHealth);
