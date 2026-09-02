/* OSPREY Web Terminal — The Bluesky-queue bar item.
 *
 * One item, in its own module because its body is the largest in the bar: a
 * live view of the Bluesky queue (what the run engine is doing, the running
 * plan and its progress, how many plans wait) with a popover that lists the
 * queue and, where the item's options allow, acts on it. It registers through
 * `defineBarItem()` exactly as the items in bar-items.js do and follows the
 * same house rule: every subscription comes from a disposer-returning API.
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
 * The Bluesky panel's id, and so the proxy every request below goes through:
 * `/panel/bluesky/...` is the terminal's own route (routes/proxy.py), which
 * forwards to the sidecar with the operator's entitlement attached. The item
 * is offered exactly where that panel is declared (the catalog's
 * `blueskyAvailable`), so a page holding this body always has the route.
 *
 * Lane 1 only, deliberately: the sidecar's lane axis is a per-document choice
 * the panel makes from its own URL, and the bar has no URL of its own. A
 * bare path is lane 1 on the sidecar side (`lane-client.js`).
 */
const QUEUE_PANEL_ID = 'bluesky';
const QUEUE_API = `/panel/${QUEUE_PANEL_ID}`;

/**
 * Manager states in which the queue is draining (or about to) — the same
 * list `queue-client.js` keeps beside the panel, mirroring the bridge's
 * `QUEUE_ACTIVE_MANAGER_STATES`. Kept in step by name.
 * @type {readonly string[]}
 */
const QUEUE_ACTIVE_MANAGER_STATES = Object.freeze([
  'starting_queue',
  'executing_queue',
  'executing_task',
  'paused',
]);

/** Longest the stream waits between reconnects, ms. */
const QUEUE_RECONNECT_MAX_MS = 30000;

/**
 * The queue as the last frame described it. `status` is the bridge's bounded
 * summary (`available`, `manager_state`, `items_in_queue`,
 * `queue_stop_pending`, …); `items` the pending plans; `runningItem` the one
 * in motion, carrying `progress` when the bridge knows it.
 * @typedef {object} QueueSnapshot
 * @property {Record<string, any> | null} status
 * @property {Array<Record<string, any>>} items
 * @property {Record<string, any> | null} runningItem
 * @property {boolean} connected - the stream is open right now
 * @property {boolean} seen - at least one frame has arrived this page
 */

/** @type {QueueSnapshot} */
let queueSnapshot = { status: null, items: [], runningItem: null, connected: false, seen: false };

/**
 * Everyone reading the queue on this page. ONE stream serves them all: the
 * item is a singleton, but the customize sheet builds a second, previewing
 * instance while it is open, and two sockets for one fact is one too many.
 * @type {Set<() => void>}
 */
const queueListeners = new Set();
/** @type {EventSource | null} */
let queueSource = null;
/** @type {ReturnType<typeof setTimeout> | null} */
let queueReconnect = null;
let queueAttempt = 0;

function notifyQueue() {
  for (const listener of queueListeners) {
    try {
      listener();
    } catch (err) {
      console.error('[bar-items] plan-queue listener threw', err);
    }
  }
}

/** @param {unknown} raw */
function applyQueueFrame(raw) {
  if (!raw || typeof raw !== 'object') return;
  const frame = /** @type {Record<string, any>} */ (raw);
  queueSnapshot = {
    status: frame.status && typeof frame.status === 'object' ? frame.status : null,
    items: Array.isArray(frame.items) ? frame.items : [],
    runningItem:
      frame.running_item && typeof frame.running_item === 'object' ? frame.running_item : null,
    connected: true,
    seen: true,
  };
  notifyQueue();
}

function openQueueStream() {
  if (queueSource || typeof EventSource === 'undefined') return;
  const source = new EventSource(withPrefix(`${QUEUE_API}/queue/events`));
  queueSource = source;
  source.onopen = () => {
    queueAttempt = 0;
    if (!queueSnapshot.connected) {
      queueSnapshot = { ...queueSnapshot, connected: true };
      notifyQueue();
    }
  };
  source.onmessage = (event) => {
    try {
      applyQueueFrame(JSON.parse(event.data));
    } catch {
      // A heartbeat or a frame this build cannot read; the next one will do.
    }
  };
  source.onerror = () => {
    if (queueSnapshot.connected) {
      queueSnapshot = { ...queueSnapshot, connected: false };
      notifyQueue();
    }
    // readyState 2 is CLOSED: the browser has given up on this source for
    // good (a proxy 502 while the sidecar restarts), so the retry is ours.
    if (source.readyState === 2 && queueSource === source) {
      source.close();
      queueSource = null;
      if (queueListeners.size > 0 && queueReconnect === null) {
        const delay = Math.min(1000 * 2 ** queueAttempt, QUEUE_RECONNECT_MAX_MS);
        queueAttempt += 1;
        queueReconnect = setTimeout(() => {
          queueReconnect = null;
          openQueueStream();
        }, delay);
      }
    }
  };
}

function closeQueueStream() {
  if (queueReconnect !== null) {
    clearTimeout(queueReconnect);
    queueReconnect = null;
  }
  queueAttempt = 0;
  if (queueSource) {
    queueSource.close();
    queueSource = null;
  }
  if (queueSnapshot.connected) queueSnapshot = { ...queueSnapshot, connected: false };
}

/**
 * Follow the queue. The stream opens with the first listener and closes with
 * the last, so a page with the item folded away holds no socket.
 * @param {() => void} listener
 * @returns {() => void} dispose
 */
function subscribeQueue(listener) {
  queueListeners.add(listener);
  openQueueStream();
  return () => {
    queueListeners.delete(listener);
    if (queueListeners.size === 0) closeQueueStream();
  };
}

/**
 * One reading of the snapshot for the chip and the popover: a tone for the
 * dot, a word for the state, a count for the corner.
 * @typedef {object} QueueReading
 * @property {'off' | 'idle' | 'active' | 'warn' | 'err'} tone
 * @property {string} word - what the state is, in one word
 * @property {string} plan - the running plan's name, or ''
 * @property {string} count - `2 queued`, `3/10`, or ''
 * @property {boolean} active
 * @property {boolean} stopPending
 */

/**
 * @param {QueueSnapshot} snap
 * @returns {QueueReading}
 */
function readQueue(snap) {
  const status = snap.status;
  const running = snap.runningItem;
  const plan = running && typeof running.name === 'string' ? running.name : '';
  const queued = snap.items.length;
  if (!snap.seen) {
    return { tone: 'off', word: 'queue', plan: '', count: '', active: false, stopPending: false };
  }
  if (!status || status.available !== true) {
    return {
      tone: 'err',
      word: 'unavailable',
      plan: '',
      count: '',
      active: false,
      stopPending: false,
    };
  }
  const state = typeof status.manager_state === 'string' ? status.manager_state : '';
  const active = QUEUE_ACTIVE_MANAGER_STATES.includes(state);
  const stopPending = status.queue_stop_pending === true;
  const progress = running && running.progress && typeof running.progress === 'object'
    ? running.progress
    : null;
  let count = '';
  if (active && progress && typeof progress.rows_seen === 'number') {
    count =
      typeof progress.expected_points === 'number'
        ? `${progress.rows_seen}/${progress.expected_points}`
        : `${progress.rows_seen} pts`;
  } else if (queued > 0) {
    count = `${queued} queued`;
  }
  let word = 'idle';
  let tone = /** @type {QueueReading['tone']} */ ('idle');
  if (state === 'paused') {
    word = 'paused';
    tone = 'warn';
  } else if (stopPending && active) {
    word = 'stopping';
    tone = 'warn';
  } else if (state === 'starting_queue') {
    word = 'starting';
    tone = 'active';
  } else if (active) {
    word = 'running';
    tone = 'active';
  } else if (state && state !== 'idle') {
    // creating_environment, closing_environment, … — the manager is busy
    // with something that is not a plan.
    word = state.replace(/_/g, ' ');
    tone = 'warn';
  }
  return { tone, word, plan, count, active, stopPending };
}

/**
 * Post one queue write through the panel's proxy and report the outcome in
 * the bridge's own words. Nothing here decides whether a write is allowed:
 * the bridge answers, and its refusal is shown verbatim.
 * @param {string} path
 * @param {Record<string, unknown>} body
 * @returns {Promise<{ok: boolean, message: string}>}
 */
async function queueWrite(path, body) {
  /** @type {Response} */
  let response;
  try {
    response = await fetch(withPrefix(`${QUEUE_API}${path}`), {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    });
  } catch {
    return { ok: false, message: 'The Bluesky panel could not be reached.' };
  }
  /** @type {any} */
  let parsed = null;
  try {
    parsed = await response.json();
  } catch {
    parsed = null;
  }
  if (response.ok) return { ok: true, message: '' };
  const detail = parsed && typeof parsed === 'object' ? parsed.detail : null;
  if (detail && typeof detail === 'object' && typeof detail.detail === 'string') {
    return { ok: false, message: detail.detail };
  }
  if (typeof detail === 'string' && detail.trim() !== '') return { ok: false, message: detail };
  return { ok: false, message: `The bridge refused the request (HTTP ${response.status}).` };
}

/**
 * The plan-queue item: what the Bluesky queue is doing, and — where its
 * options allow — the way to act on it without opening the panel.
 *
 * The chip is a dot, a word and a count, and it never acts on a click: it
 * opens. Every action lives in the popover, labelled, with the row it applies
 * to on screen — the same rule the control-target chip follows, for the same
 * reason. `controls` decides which actions the popover offers: none (Open
 * Bluesky alone), the plain stop, or the panel's full set. The two halts keep
 * the panel's shape exactly: a plain stop fires on the first click, because
 * friction in front of a halt is friction in the wrong place; withdrawing a
 * pending stop and the abort are two-step, because both send hardware
 * somewhere. Neither halt is ever disabled — the bridge answers a stop in
 * every state by design.
 *
 * The stream is shared (see `subscribeQueue`) and the subscription is
 * attach-scoped: parking the shell closes it, placing the shell back reopens
 * it and repaints from the frame that then arrives.
 * @param {BarBuildContext} ctx
 * @returns {BarItemInstance}
 */
function buildPlanQueue(ctx) {
  const doc = ctx.shell.ownerDocument;
  const controls = String(ctx.options.controls ?? 'none');
  const showPlan = ctx.options.progress !== false;
  const showCount = ctx.options.count !== false;

  const body = doc.createDocumentFragment();
  const chip = doc.createElement('button');
  chip.type = 'button';
  chip.className = 'bar-item-btn bar-queue';
  chip.setAttribute('aria-haspopup', 'true');
  chip.setAttribute('aria-expanded', 'false');
  const dot = doc.createElement('span');
  dot.className = 'bar-queue-dot';
  dot.setAttribute('aria-hidden', 'true');
  const text = doc.createElement('span');
  text.className = 'bar-queue-text';
  const count = doc.createElement('span');
  count.className = 'bar-queue-count';
  chip.append(dot, text, count);
  body.appendChild(chip);

  const pop = doc.createElement('div');
  pop.className = 'bar-pop bar-queue-pop';
  pop.hidden = true;
  pop.setAttribute('role', 'dialog');
  pop.setAttribute('aria-label', 'Bluesky queue');
  body.appendChild(pop);

  let open = false;
  let stopArmed = false;
  let abortArmed = false;
  let note = '';

  const renderChip = () => {
    const reading = readQueue(queueSnapshot);
    dot.dataset.tone = reading.tone;
    const shown = showPlan && reading.plan ? reading.plan : reading.word;
    if (text.textContent !== shown) text.textContent = shown;
    const corner = showCount ? reading.count : '';
    if (count.textContent !== corner) count.textContent = corner;
    count.hidden = corner === '';
    const parts = [`Bluesky queue: ${reading.word}`];
    if (reading.plan) parts.push(reading.plan);
    if (reading.count) parts.push(reading.count);
    if (!queueSnapshot.connected) parts.push('stream not connected');
    chip.title = parts.join(' · ');
    chip.setAttribute('aria-label', chip.title);
  };

  /**
   * @param {string} cls
   * @param {string} label
   * @param {() => void} onClick
   */
  const button = (cls, label, onClick) => {
    const el = doc.createElement('button');
    el.type = 'button';
    el.className = cls;
    el.textContent = label;
    el.addEventListener('click', (event) => {
      event.stopPropagation();
      onClick();
    });
    return el;
  };

  /** @param {string} path @param {Record<string, unknown>} payload */
  const write = (path, payload) => {
    note = '';
    renderPop();
    void queueWrite(path, payload).then((outcome) => {
      note = outcome.ok ? '' : outcome.message;
      renderPop();
    });
  };

  const renderPop = () => {
    if (!open) return;
    const reading = readQueue(queueSnapshot);
    pop.replaceChildren();
    const eyebrow = doc.createElement('div');
    eyebrow.className = 'bar-pop-eyebrow';
    eyebrow.textContent = 'Bluesky queue';
    const title = doc.createElement('div');
    title.className = 'bar-queue-pop-title';
    const titleDot = doc.createElement('span');
    titleDot.className = 'bar-queue-dot';
    titleDot.dataset.tone = reading.tone;
    titleDot.setAttribute('aria-hidden', 'true');
    const titleWord = doc.createElement('span');
    titleWord.textContent = queueSnapshot.connected ? reading.word : 'not connected';
    title.append(titleDot, titleWord);
    pop.append(eyebrow, title);

    const rows = doc.createElement('div');
    rows.className = 'bar-queue-rows';
    const row = (/** @type {string} */ name, /** @type {string} */ aside) => {
      const line = doc.createElement('div');
      line.className = 'bar-queue-row';
      const left = doc.createElement('span');
      left.className = 'bar-queue-row-name';
      left.textContent = name;
      const right = doc.createElement('span');
      right.className = 'bar-queue-row-aside';
      right.textContent = aside;
      line.append(left, right);
      rows.appendChild(line);
    };
    if (queueSnapshot.runningItem) {
      row(reading.plan || 'plan', reading.active ? reading.count || 'running' : 'running');
    }
    const pending = queueSnapshot.items;
    for (const item of pending.slice(0, 5)) {
      row(typeof item.name === 'string' ? item.name : 'plan', 'queued');
    }
    if (pending.length > 5) row(`+${pending.length - 5} more`, '');
    if (!queueSnapshot.runningItem && pending.length === 0) row('Nothing queued', '');
    pop.appendChild(rows);

    if (note) {
      const line = doc.createElement('div');
      line.className = 'bar-pop-note bar-queue-note';
      line.textContent = note;
      pop.appendChild(line);
    }

    const foot = doc.createElement('div');
    foot.className = 'bar-pop-foot';
    foot.appendChild(
      button('bar-btn', 'Open Bluesky', () => {
        close();
        void import('./panel-manager.js').then((manager) => manager.showPanel(QUEUE_PANEL_ID));
      })
    );
    if (controls === 'full') {
      const start = button('bar-btn', 'Start', () => write('/queue/start', {}));
      start.disabled = reading.active || pending.length === 0 || reading.tone === 'err';
      foot.appendChild(start);
    }
    if (controls === 'stop' || controls === 'full') {
      if (reading.stopPending) {
        foot.appendChild(
          button(
            stopArmed ? 'bar-btn is-danger' : 'bar-btn',
            stopArmed ? 'Confirm — the queue keeps draining' : 'Withdraw stop',
            () => {
              if (!stopArmed) {
                stopArmed = true;
                renderPop();
                return;
              }
              stopArmed = false;
              write('/queue/stop', { cancel: true });
            }
          )
        );
      } else {
        foot.appendChild(
          button('bar-btn', 'Stop after current item', () => {
            stopArmed = false;
            write('/queue/stop', { cancel: false });
          })
        );
      }
    }
    if (controls === 'full') {
      foot.appendChild(
        button('bar-btn is-danger', abortArmed ? 'Confirm abort' : 'Abort running plan', () => {
          if (!abortArmed) {
            abortArmed = true;
            renderPop();
            return;
          }
          abortArmed = false;
          write('/queue/abort', {});
        })
      );
    }
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
    stopArmed = false;
    abortArmed = false;
    note = '';
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

  const unsubscribe = subscribeQueue(() => {
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

defineBarItem('bluesky-queue', buildPlanQueue);
