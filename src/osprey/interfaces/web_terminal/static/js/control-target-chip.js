// @ts-check
/* OSPREY Web Terminal — Control-Target Header Chip
 *
 * The one-glance answer to "if the agent writes now, where does it land, and
 * will it be refused?". A 28 px chip in the global header reading
 * `● Rehearsal · writes on ▾`: the name of the machine this session stands
 * on, the effective write state on THAT machine, and a dot whose colour says
 * how much a write there matters and whose shape says who is holding writes
 * back. The popover behind it (control-target-popover.js) lists every
 * configured target and owns every gesture that CHANGES something; this module
 * owns the chip, the read, and the state every renderer shares.
 *
 * ONE truth: `GET /api/terminal/posture?session_id=`. The chip renders what
 * that route says and never what this page last did — a posture narrowed in
 * another tab, a target switched by the agent mid-turn, and a session restored
 * after a container restart all reach the operator the same way. Every mutation
 * anywhere therefore ends in a re-read, and so does every refusal.
 *
 * **Nothing here parses prose.** A control-target switch is announced on the
 * agent-activity stream as `{tool: 'control_target_set', target: {kind:
 * 'config', detail: 'live → va · success'}}`, and that sentence is the agent's
 * narration, not this session's state: it is broadcast to every open client,
 * it names targets in the controls server's words, and a failure and a success
 * differ by one word inside it. So the frame is a REFETCH HINT and only that.
 * The chip renders `last_switch` from the route, matched by `request_id` — a
 * value this browser minted, which cannot be another tab's switch.
 *
 * Three refresh rates, for three different questions:
 *
 * - 5 s idle, because a switch the AGENT makes fires no session change and no
 *   gesture here; without a poll the chip would keep naming the machine the
 *   operator left, which is the one failure this surface exists to prevent.
 * - 500 ms while a switch request this browser wrote is outstanding — the
 *   reconciler picks the request up on its own 1 s poll, and an operator who
 *   just clicked Switch is watching the word `switching…`.
 * - the hint above, for everything else.
 *
 * And one deadline that is ours alone: a controls server that has died answers
 * nothing at all, so no `last_switch` for that `request_id` will ever be
 * published. After `REQUEST_TTL_S` — the same TTL the reconciler drops a
 * request at — the client SYNTHESISES `request_expired` locally rather than
 * showing `switching…` forever. It is marked as synthesised, and a real
 * outcome for that id still wins if one turns up.
 *
 * Both topologies use this unchanged: every request goes through api.js's
 * `withPrefix` chokepoint, so the multi-user per-user mount (`/u/<name>/…`) is
 * already handled.
 */

import { displayName, statePhrase } from './control-target-facts.js';
import { withPrefix, createEventSource } from './api.js';
import { AGENT_ACTIVITY_FRAME } from './activity-format.js';
import { getCurrentSessionId, onSessionChange } from './terminal.js';

/**
 * One row of the roster (`targets[]` of `GET /api/terminal/posture`).
 * @typedef {object} TargetRow
 * @property {string} target  the config name (`live` / `va` / `standin`) — a
 *   state key, never display text
 * @property {string} label  what the controls server calls that machine
 * @property {string} short_label  the chip's word: LIVE / STAND-IN / VIRTUAL /
 *   SIMULATED, derived server-side from `real_machine` and the label's shape
 * @property {string} kind  the plain-language word for the same derivation
 *   (`live machine` / `stand-in` / `virtual accelerator` / `simulated`)
 * @property {string} endpoint
 * @property {boolean} real_machine  true for the facility's own machine AND
 *   for a stand-in — both get every limit and prompt hardware gets
 * @property {boolean} active  the target this session stands on
 * @property {boolean} is_baseline
 * @property {boolean} available_now  whether a Switch is offered
 * @property {string|null} reason  the switch tool's own refusal code when not
 * @property {string|null} reason_detail  the eligibility verdict's operator
 *   sentence for that code — the popover's tooltip, never its row text
 * @property {boolean} ceiling_writes  what the persona render permits
 * @property {'writes'|'sandbox'} posture  this session's own narrowing
 * @property {boolean} effective  the whole rule the connector applies
 * @property {{state: string, role: string|null, probed_at: string|null,
 *             age_s: number|null, role_detail?: Record<string, string>}} reachability
 */

/**
 * The payload of `GET /api/terminal/posture`.
 * @typedef {object} PostureView
 * @property {string} session_id
 * @property {string} session_target
 * @property {boolean} store_available
 * @property {boolean} enforceable
 * @property {string|null} enforceable_reason
 * @property {boolean} execution_in_flight
 * @property {LastSwitch|null} last_switch
 * @property {{state: string, at?: string}|null} last_posture_realign
 * @property {TargetRow[]} targets
 */

/**
 * The outcome of the most recent switch request, as published by the
 * reconciler — or, for a request nothing ever answered, as synthesised here.
 * @typedef {object} LastSwitch
 * @property {string} request_id
 * @property {string} [target]
 * @property {'success'|'refused'|'failed'|'expired'|string} status
 * @property {string|null} [reason]  the refusal word (`request_expired` rides
 *   on `expired`)
 * @property {string|null} [detail]
 * @property {string} [at]
 * @property {number|null} [age_s]
 * @property {boolean} [synthesized]  true only for the local expiry below
 */

/** The idle re-read. See the module docstring for why it exists at all. */
export const IDLE_POLL_MS = 5000;

/** The re-read while a switch request is outstanding. */
export const FAST_POLL_MS = 500;

/**
 * How long a switch request may go unanswered before the client calls it
 * expired. Matches the reconciler's own TTL: past this the request file is
 * dropped unread, so `switching…` would never end on its own.
 */
export const REQUEST_TTL_S = 30;

/**
 * The activity tool whose frame means "the control target may have moved".
 * Mirrors `osprey.mcp_server.http.TARGET_SWITCH_TOOL`; a rename on either side
 * costs a refetch hint, never a wrong render (the route stays the truth).
 */
export const TARGET_SWITCH_TOOL = 'control_target_set';

/**
 * Dispatched on the chip element when the operator clicks it. The chip owns
 * `aria-expanded` and nothing else; the popover module listens for this and
 * opens or closes itself, then calls {@link setExpanded} for any dismissal it
 * drives on its own (outside click, Escape, a completed switch).
 *
 * `detail.expanded` carries the state the chip has just moved TO.
 */
export const CHIP_TOGGLE_EVENT = 'osprey-control-target-toggle';

/**
 * The plain-language `kind` the route publishes → the `data-target-kind` value
 * the stylesheet keys the dot and the border on. Two vocabularies on purpose:
 * the route's is what the popover SHOWS a simple-mode operator, this one is a
 * CSS selector value.
 */
/** @type {Record<string, string>} */
const KIND_ATTR = {
  'live machine': 'live',
  'stand-in': 'standin',
  'virtual accelerator': 'va',
  simulated: 'simulated',
};

/**
 * The chip and the popover travel together inside one positioning context —
 * the popover is `position: absolute` under its trigger (terminal.css
 * `.ctc-anchor`), and `.header-actions` is not itself positioned.
 * @type {HTMLElement|null}
 */
let anchor = null;
/** @type {HTMLButtonElement|null} */
let chip = null;
/** @type {HTMLElement|null} */
let shortEl = null;
/** @type {HTMLElement|null} */
let stateEl = null;

/** The last payload the route answered. Null = nothing known yet. */
/** @type {PostureView|null} */
let view = null;

/** The request this browser is waiting on, if any. */
/** @type {{requestId: string, target: string|null, startedAt: number}|null} */
let pending = null;

/**
 * A `request_expired` this client minted because nothing answered. Cleared the
 * moment the route publishes any outcome for that same `request_id` — a slow
 * reconciler that lands late still gets the last word.
 */
/** @type {LastSwitch|null} */
let synthesized = null;

/**
 * Monotonic id for reads, so a slow one cannot overwrite a newer answer.
 *
 * The session guard does not cover this: the id is unchanged. A 5 s tick fires
 * a GET, the operator confirms a switch, the POST's own re-read lands — and the
 * tick's older GET resolves afterwards carrying the PRE-switch roster,
 * repainting the chip as the machine the session just left. Every read takes a
 * ticket here and drops its answer if a later read has since been issued.
 */
let readSeq = 0;

/** @type {ReturnType<typeof setInterval>|null} */
let idleTimer = null;
/** @type {ReturnType<typeof setInterval>|null} */
let fastTimer = null;
/** @type {{stop: () => void}|null} */
let sse = null;

/** Render subscribers (the popover). */
/** @type {((state: PostureView|null) => void)[]} */
let listeners = [];

/** False after teardown, so a session-change callback cannot revive a dead chip. */
let mounted = false;

/** `onSessionChange` has no unsubscribe; subscribe once per page, ever. */
let sessionSubscribed = false;

/* ---- mount ---- */

/**
 * Mount the chip into the global header and keep it current.
 *
 * Idempotent: a second call re-renders rather than mounting a second chip.
 * Safe to call before any session exists — the chip stays hidden until the
 * terminal reports one, because a roster is a fact about a session and there
 * is nothing honest to show without one.
 *
 * @param {{eventSourceFactory?: typeof createEventSource}} [opts]
 *   `eventSourceFactory` is injectable for tests the way session.js's
 *   `wireActivityStrip` injects it — happy-dom has no EventSource.
 */
export function initControlTargetChip({ eventSourceFactory = createEventSource } = {}) {
  const actions = document.querySelector('.header-actions');
  if (!actions) return;

  mounted = true;

  if (!chip || !actions.contains(chip)) {
    anchor = document.createElement('div');
    anchor.className = 'ctc-anchor';
    anchor.hidden = true;
    chip = buildChip();
    anchor.appendChild(chip);
    // Before the palette trigger: the chip answers "where am I standing",
    // which is read before anything is looked up. insertBefore with a null
    // reference node appends, so a header without the palette still mounts it.
    actions.insertBefore(anchor, actions.querySelector('#command-palette-btn'));
  }

  if (!sessionSubscribed) {
    sessionSubscribed = true;
    // Every path that settles a session id — a `session_info` frame, a switch,
    // the resume-liveness timer — funnels through terminal.js's
    // notifySessionChange, so this is the one subscription needed.
    onSessionChange(() => {
      if (mounted) void refetch();
    });
  }

  if (!sse) sse = subscribeSwitchHints(eventSourceFactory);

  void refetch();
}

/**
 * Unmount the chip and release everything it holds: both timers, the activity
 * subscription, the render subscribers and the DOM node.
 *
 * The session-change subscription cannot be released (terminal.js offers no
 * unsubscribe), so `mounted` gates its callback instead.
 */
export function teardownControlTargetChip() {
  mounted = false;
  stopIdlePolling();
  stopFastPolling();
  if (sse) {
    sse.stop();
    sse = null;
  }
  // The anchor goes with it, popover and all: a positioning context with
  // nothing to position is a stray flex child in the header's action gap.
  if (anchor) anchor.remove();
  else chip?.remove();
  anchor = null;
  chip = null;
  shortEl = null;
  stateEl = null;
  view = null;
  pending = null;
  synthesized = null;
  listeners = [];
}

/** @returns {HTMLButtonElement} */
function buildChip() {
  const el = document.createElement('button');
  el.type = 'button';
  el.className = 'control-target-chip';
  el.id = 'control-target-chip';
  el.hidden = true;
  el.setAttribute('aria-haspopup', 'true');
  el.setAttribute('aria-expanded', 'false');

  const dot = span('ctc-dot');
  shortEl = span('ctc-short');
  const sep = span('ctc-sep');
  sep.textContent = '·';
  stateEl = span('ctc-state');
  const caret = span('ctc-caret');
  caret.textContent = '▾';
  for (const decoration of [dot, sep, caret]) decoration.setAttribute('aria-hidden', 'true');

  el.append(dot, shortEl, sep, stateEl, caret);
  el.addEventListener('click', onChipClick);
  return el;
}

/** @param {string} cls @returns {HTMLElement} */
function span(cls) {
  const el = document.createElement('span');
  el.className = cls;
  return el;
}

/**
 * Status and action are separate gestures: the chip only opens. Every change
 * is a labelled action inside the popover, where the row it applies to is on
 * screen — a header chip that toggled something on click would be a one-click
 * path to arming writes on a real machine.
 */
function onChipClick() {
  if (!chip) return;
  const expanded = chip.getAttribute('aria-expanded') !== 'true';
  chip.setAttribute('aria-expanded', String(expanded));
  chip.dispatchEvent(
    new CustomEvent(CHIP_TOGGLE_EVENT, { bubbles: true, detail: { expanded } })
  );
}

/* ---- the read ---- */

/**
 * One request path for the chip and for the popover's POSTs, owning its own
 * error contract.
 *
 * Not api.js's `apiRequest`: these routes raise `HTTPException` with a DICT
 * detail (`{error, message}` — routes/websocket.py), and `apiRequest` builds
 * its Error from `detail.detail`, which stringifies an object to
 * "[object Object]". That is precisely the wording an operator most needs —
 * the refusal word under a Switch, the reason a toggle 409'd — so this reads
 * the body itself and unwraps the sentence.
 *
 * @param {string} path
 * @param {{method?: string, json?: any}} [opts]
 * @returns {Promise<any>} the parsed body
 * @throws {Error} on a non-OK response, carrying the server's own message
 */
export async function targetRequest(path, { method = 'GET', json } = {}) {
  /** @type {RequestInit} */
  const init = { method, cache: 'no-store' };
  if (json !== undefined) {
    init.headers = { 'Content-Type': 'application/json' };
    init.body = JSON.stringify(json);
  }
  const resp = await fetch(withPrefix(path), init);
  const body = await resp.json().catch(() => null);
  if (!resp.ok) throw new Error(refusalMessage(body, resp.status));
  return body;
}

/**
 * The most specific human sentence a refusal body carries. Handles all three
 * shapes rather than guessing at one: FastAPI's dict detail (what these routes
 * raise), the plain string detail every other route uses, and a body with
 * neither — where the status code is all there is to say.
 * @param {any} body @param {number} status @returns {string}
 */
export function refusalMessage(body, status) {
  const detail = body && typeof body === 'object' ? body.detail : null;
  if (typeof detail === 'string' && detail.trim()) return detail;
  if (detail && typeof detail === 'object' && typeof detail.message === 'string') {
    if (detail.message.trim()) return detail.message;
  }
  if (body && typeof body === 'object' && typeof body.message === 'string' && body.message.trim()) {
    return body.message;
  }
  return `Could not read this session's control target (HTTP ${status}).`;
}

/**
 * Re-read the roster for the current session and repaint.
 *
 * The one refresh path: mount, session change, idle tick, fast tick, activity
 * hint and every popover mutation all land here.
 * @returns {Promise<void>}
 */
export async function refetch() {
  if (!chip) return;
  const sessionId = getCurrentSessionId();
  if (!sessionId) {
    stopIdlePolling();
    stopFastPolling();
    pending = null;
    view = null;
    render();
    return;
  }
  if (!pending) startIdlePolling();

  const seq = ++readSeq;
  try {
    const data = await targetRequest(
      `/api/terminal/posture?session_id=${encodeURIComponent(sessionId)}`
    );
    // The session can change while the read is in flight (a switch, a failover
    // resume); a stale answer must never repaint the chip.
    if (getCurrentSessionId() !== sessionId) return;
    // A newer read has been issued since this one left — most importantly the
    // re-read a mutation does — so this answer is already history.
    if (seq !== readSeq) return;
    view = data;
    reconcilePending();
  } catch (err) {
    // Same stale guards as the success path: a read that failed for a session
    // this page has already left, or that a newer read superseded, must not
    // blank the chip.
    if (getCurrentSessionId() !== sessionId || seq !== readSeq) return;
    // Unknown beats wrong: a chip that cannot read the roster says nothing
    // rather than naming a machine nobody confirmed.
    console.error('osprey web_terminal: could not read the control-target roster', err);
    view = null;
  }
  render();
}

/**
 * Decide what the answer just read means for the request we are waiting on.
 *
 * Landing is matched by `request_id` and by nothing else. Matching on the
 * TARGET would land on another tab's switch to the same machine, and matching
 * on "the session_target changed" would land on a switch the agent made for
 * its own reasons — both of which end `switching…` on a request still in
 * flight.
 */
function reconcilePending() {
  const landed = view?.last_switch?.request_id;
  if (synthesized && landed === synthesized.request_id) synthesized = null;
  if (!pending) return;
  if (landed === pending.requestId) {
    pending = null;
    stopFastPolling();
    startIdlePolling();
    return;
  }
  expireIfOverdue();
}

/**
 * Call an unanswered request expired once its TTL has passed.
 *
 * The reconciler drops a request older than this unread, and a controls server
 * that has died never reads one at all — in both cases no outcome is coming and
 * `switching…` would be permanent. The synthesised outcome carries the same
 * word the reconciler would have published (`request_expired`) and is flagged
 * `synthesized` so a renderer can tell a local deadline from a server verdict.
 */
function expireIfOverdue() {
  if (!pending) return;
  if (Date.now() - pending.startedAt < REQUEST_TTL_S * 1000) return;
  synthesized = {
    request_id: pending.requestId,
    target: pending.target ?? undefined,
    status: 'expired',
    reason: 'request_expired',
    detail: null,
    at: new Date().toISOString(),
    age_s: 0,
    synthesized: true,
  };
  pending = null;
  stopFastPolling();
  startIdlePolling();
}

/* ---- polling ---- */

/**
 * Start the idle re-read, if it is not already running.
 *
 * The tick stops itself once there is nothing left to paint: no session, or a
 * chip no longer in the document. Checking the DOM rather than trusting a
 * teardown call means no code path can leak a timer that outlives the chip.
 */
function startIdlePolling() {
  if (idleTimer !== null) return;
  idleTimer = setInterval(() => {
    if (!chip || !chip.isConnected || !getCurrentSessionId()) {
      stopIdlePolling();
      return;
    }
    if (pending) return; // the fast poll owns this window
    void refetch();
  }, IDLE_POLL_MS);
}

/** Stop the idle re-read. Idempotent. */
function stopIdlePolling() {
  if (idleTimer === null) return;
  clearInterval(idleTimer);
  idleTimer = null;
}

/**
 * Start the fast re-read for the duration of one outstanding request.
 *
 * The deadline is checked BEFORE the read, so an expiry lands on time even
 * while every GET is failing — a dead controls server is exactly the case this
 * timer exists for, and it may well have taken the route down with it.
 */
function startFastPolling() {
  if (fastTimer !== null) return;
  fastTimer = setInterval(() => {
    if (!chip || !chip.isConnected || !getCurrentSessionId()) {
      stopFastPolling();
      return;
    }
    if (!pending) {
      stopFastPolling();
      return;
    }
    expireIfOverdue();
    if (!pending) {
      render();
      return;
    }
    void refetch();
  }, FAST_POLL_MS);
}

/** Stop the fast re-read. Idempotent. */
function stopFastPolling() {
  if (fastTimer === null) return;
  clearInterval(fastTimer);
  fastTimer = null;
}

/* ---- the refetch hint ---- */

/**
 * Subscribe to the agent-activity stream for switch announcements.
 *
 * A HINT and only a hint: the frame's `detail` is the agent's narration of a
 * switch that may belong to any client on this stream, so nothing is read out
 * of it — the tool name alone triggers a re-read, and the route answers what
 * actually happened to THIS session.
 *
 * Its own subscription rather than a seam through panel-manager, following
 * session.js's `wireActivityStrip`: the chip lives in the global header and
 * must work on a page where no panel workspace ever boots.
 * @param {typeof createEventSource} factory
 */
function subscribeSwitchHints(factory) {
  return factory('/api/files/events', {
    onMessage: (data) => {
      if (!data || typeof data !== 'object') return;
      if (data.type !== AGENT_ACTIVITY_FRAME) return;
      if (data.tool !== TARGET_SWITCH_TOOL) return;
      if (mounted) void refetch();
    },
  });
}

/* ---- render ---- */

/**
 * The roster as every renderer should read it: the route's answer, with a
 * locally synthesised expiry standing in for a `last_switch` that is never
 * coming. Null until the first successful read.
 * @returns {PostureView|null}
 */
export function getState() {
  if (!view) return null;
  if (synthesized && view.last_switch?.request_id !== synthesized.request_id) {
    return { ...view, last_switch: synthesized };
  }
  return view;
}

/**
 * Be told after every render. Called with the same value {@link getState}
 * returns, including `null` for "nothing known".
 * @param {(state: PostureView|null) => void} fn
 * @returns {() => void} unsubscribe
 */
export function subscribe(fn) {
  listeners.push(fn);
  return () => {
    listeners = listeners.filter((other) => other !== fn);
  };
}

/**
 * Note that a switch request this browser wrote is outstanding.
 *
 * Called by the popover with the `request_id` its POST returned. Arms the fast
 * poll and the local expiry deadline, and puts the chip in `switching…` — the
 * only piece of chip state that does not come from the route, and it is a
 * question ("what did I ask for?") the route cannot answer.
 * @param {string} requestId
 * @param {string|null} [target]  the row that was asked for, so a synthesised
 *   expiry can name it
 */
export function markPending(requestId, target = null) {
  if (!requestId) return;
  pending = { requestId, target, startedAt: Date.now() };
  synthesized = null;
  stopIdlePolling();
  startFastPolling();
  render();
}

/** Whether a switch request is outstanding right now. */
export function isPending() {
  return pending !== null;
}

/** The chip element, for the popover to hang its listeners on. */
export function getChipElement() {
  return chip;
}

/**
 * The positioning context the popover mounts into (`.ctc-anchor`, the chip's
 * own parent). The popover is `position: absolute` under its trigger, and
 * `.header-actions` is not positioned — appending the popover anywhere else
 * would anchor it to the page.
 */
export function getAnchorElement() {
  return anchor;
}

/** Whether the chip is currently showing itself as open. */
export function isExpanded() {
  return chip?.getAttribute('aria-expanded') === 'true';
}

/**
 * Mirror the popover's open state onto the chip. The popover dismisses itself
 * for reasons the chip never sees (an outside click, Escape); without this the
 * chip would keep claiming `aria-expanded=true` and the hover/open styling
 * would stay lit over a closed popover.
 * @param {boolean} expanded
 */
export function setExpanded(expanded) {
  chip?.setAttribute('aria-expanded', String(Boolean(expanded)));
}

/**
 * The row the chip speaks for: the one this session stands on.
 *
 * Two fallbacks, in the order of how much they claim. The baseline row is what
 * the deployment would put this session on and is the route's own answer when
 * nothing has been published; the first row is a last resort for a roster whose
 * shape this module did not anticipate. Both are better than an empty chip,
 * which would read as "no control system here".
 * @param {PostureView} state
 * @returns {TargetRow|null}
 */
function activeRow(state) {
  const rows = Array.isArray(state.targets) ? state.targets : [];
  if (!rows.length) return null;
  return (
    rows.find((r) => r.active) ??
    rows.find((r) => r.target === state.session_target) ??
    rows.find((r) => r.is_baseline) ??
    rows[0]
  );
}

/**
 * The `data-target-kind` value for one row.
 *
 * The route already made this decision (from `real_machine` and the label's
 * SHAPE, never from the target name, so a stand-in never renders as the
 * facility's own machine); this restates the mapping onto CSS values and keeps
 * the same derivation as a fallback for a row that carries no `kind`. The
 * fallback fails loud on purpose — an unrecognised real machine is `live`, the
 * direction this stack must fail in.
 *
 * Exported for the popover, which keys each ROW's dot and tints on the same
 * attribute: one derivation for the chip and the row it speaks for, so a
 * stand-in cannot render as the facility's own machine in one of them.
 * @param {TargetRow} row
 * @returns {string}
 */
export function kindAttr(row) {
  const published = KIND_ATTR[String(row.kind || '').trim().toLowerCase()];
  if (published) return published;
  const label = String(row.label || '').trim().toLowerCase();
  if (row.real_machine) return label.includes('(stand-in)') ? 'standin' : 'live';
  if (label.startsWith('virtual accelerator')) return 'va';
  return 'simulated';
}

/**
 * The effective state word for one row: `writes`, `sandbox` or `read-only`.
 *
 * Three terms collapse to one word here, and which of the two "no" cases it is
 * matters: `sandbox` is the operator's own narrowing and one click undoes it,
 * `read-only` is the deployment's ceiling (or a read-only run) and no gesture
 * in this interface will move it. A single "no" word would leave an operator
 * clicking a toggle that was never going to open.
 *
 * Exported for the popover, whose rows and confirms have to name the state in
 * the same three words the chip does — the confirm for a switch tells the
 * operator the posture they will land in, and it must be the word the chip
 * shows a moment later.
 * @param {TargetRow} row
 * @returns {'writes'|'sandbox'|'read-only'}
 */
export function stateWord(row) {
  if (row.effective) return 'writes';
  return row.posture === 'sandbox' ? 'sandbox' : 'read-only';
}

/**
 * Paint the chip from {@link getState}, then tell the subscribers.
 *
 * Everything visual is a data attribute; not one colour name is spelled here.
 * The stylesheet owns the whole map (which dot is filled, half or hollow, and
 * that only a live machine with writes armed tints the border), so the loudness
 * rules stay in one file and stay reviewable as a set.
 */
function render() {
  const state = getState();
  if (chip && shortEl && stateEl) {
    const row = state ? activeRow(state) : null;
    if (!state || !row) {
      // The ANCHOR is what hides: hiding the chip alone would leave a
      // zero-width flex child holding open the header's action gap.
      if (anchor) anchor.hidden = true;
      chip.hidden = true;
      chip.removeAttribute('data-pending');
    } else {
      const word = stateWord(row);
      if (anchor) anchor.hidden = false;
      chip.hidden = false;
      chip.dataset.targetKind = kindAttr(row);
      chip.dataset.state = word;
      // Enforceable is a question about the ANCHOR and the store, not about
      // the ceiling: it says whether a narrowing recorded here would actually
      // reach the agent. When it would not, the chip is dimmed and still opens
      // — the roster is worth reading even where the toggles govern nothing.
      chip.dataset.enforceable = String(Boolean(state.enforceable && state.store_available));
      if (pending) chip.dataset.pending = 'true';
      else chip.removeAttribute('data-pending');
      shortEl.textContent = displayName(row, kindAttr(row));
      // `data-state` keeps the real state under a pending request: the dot and
      // the border still describe the machine the session is on until the
      // switch actually lands.
      stateEl.textContent = pending ? 'switching…' : statePhrase(word);
      chip.title = row.label || row.target;
      chip.setAttribute(
        'aria-label',
        `Control target: ${displayName(row, kindAttr(row))} · ${pending ? 'switching' : statePhrase(word)}`
      );
    }
  }
  for (const fn of listeners) {
    try {
      fn(state);
    } catch (err) {
      // One broken subscriber must not stop the others, and must not leave the
      // chip unpainted — it is already painted by the time we get here.
      console.error('osprey web_terminal: a control-target subscriber threw', err);
    }
  }
}
