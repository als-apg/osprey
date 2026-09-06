// @ts-check
/* OSPREY Web Terminal — Control-Target Header Chip
 *
 * The one-glance answer to "if the agent writes now, where does it land, and
 * will it be refused?". A 28 px chip in the global header reading
 * `● Rehearsal · writes on ▾`: the name of the machine this deployment stands
 * on, the effective write state on THAT machine, and a dot whose colour says
 * how much a write there matters and whose shape says who is holding writes
 * back. The popover behind it (control-target-popover.js) lists every
 * configured target and owns every gesture that CHANGES something; this module
 * owns the chip, the read, and the state every renderer shares.
 *
 * ONE truth: `GET /api/terminal/posture`. The chip renders what that route
 * says and never what this page last did — a posture narrowed in another tab,
 * a target switched by the agent mid-turn, and a browser reloaded after a
 * container restart all reach the operator the same way. Every mutation
 * anywhere therefore ends in a re-read, and so does every refusal.
 *
 * **The roster is a fact about the DEPLOYMENT, not about a session.** One
 * control-context record says where this deployment's control system points,
 * so the read carries no session id and nothing here subscribes to a session
 * change. That is also what lets this module load on a page that has no
 * terminal at all: THIS module imports only `control-target-facts.js`,
 * `api.js` and `activity-format.js` — a list the suite pins by reading this
 * file — and the JupyterLab bar mounts it by handing
 * {@link initControlTargetChip} a `host` of its own. (The bar's own closure is
 * wider: it adds the popover, `confirm-skip.js` and `posture-confirm.js`.)
 *
 * **Nothing here parses prose.** A control-target switch is announced on the
 * agent-activity stream as `{tool: 'control_target_set', target: {kind:
 * 'config', detail: 'live → va · success'}}`, and that sentence is the agent's
 * narration, not the deployment's state: it is broadcast to every open client,
 * it names targets in the controls server's words, and a failure and a success
 * differ by one word inside it. So the frame is a REFETCH HINT and only that.
 *
 * Three refresh rates, for three different questions:
 *
 * - the push, `{type: 'control_context'}`, raised by the owning terminal on
 *   any change to the record or to a controls server's report. This is the
 *   normal path and it is why the chip follows a switch the agent made within
 *   a tick rather than within five seconds.
 * - 5 s idle, as the fallback — and as the whole story on a deployment served
 *   without a broadcaster, where no frame is ever pushed.
 * - 500 ms while a switch request this browser wrote is outstanding, because
 *   an operator who just clicked Switch is watching the word `switching…`.
 *
 * And one deadline that is ours alone: a deployment that has died answers
 * nothing at all, so no outcome for that `request_id` will ever be published.
 * After `REQUEST_TTL_S` the client SYNTHESISES `request_expired` locally
 * rather than showing `switching…` forever. It is marked as synthesised, and a
 * real terminus for that id still wins if one turns up.
 *
 * Both topologies use this unchanged: every request goes through api.js's
 * `withPrefix` chokepoint, so the multi-user per-user mount (`/u/<name>/…`) is
 * already handled.
 */

import {
  REASON_SWITCH_FAILED,
  contextWritable,
  displayName,
  resolvePendingSwitch,
  statePhrase,
  switchFailureNote,
} from './control-target-facts.js';
import { withPrefix, createEventSource } from './api.js';
import { AGENT_ACTIVITY_FRAME } from './activity-format.js';

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
 * @property {boolean} active  the target the deployment stands on
 * @property {boolean} is_baseline
 * @property {boolean} available_now  whether a Switch is offered
 * @property {string|null} reason  the switch tool's own refusal code when not
 * @property {string|null} reason_detail  the eligibility verdict's operator
 *   sentence for that code — the popover's tooltip, never its row text
 * @property {boolean} ceiling_writes  what the persona render permits
 * @property {'writes'|'sandbox'} posture  the deployment's recorded narrowing
 * @property {boolean} effective  the whole rule the connector applies
 * @property {{state: string, role: string|null, probed_at: string|null,
 *             age_s: number|null, role_detail?: Record<string, string>}} reachability
 */

/**
 * One live controls server, as `servers[]` of `GET /api/terminal/posture`
 * publishes it. The fleet is published rather than collapsed because "has the
 * switch landed" is a question about every server at once, and a single
 * verdict could not name the one that is stuck.
 * @typedef {object} ServerRow
 * @property {number} pid
 * @property {string|null} session
 * @property {string|null} applied_target  where its connector host actually
 *   is, `null` until it has got anywhere
 * @property {number|null} applied_generation  the generation it arrived at
 * @property {{status: 'applying'|'applied'|'failed'|string,
 *             generation?: number|null, detail?: string|null}|null} last_switch
 *   its progress through the swap the record is coordinating
 * @property {{state: string}|null} last_posture_realign
 * @property {string|null} updated_at
 */

/**
 * The payload of `GET /api/terminal/posture`.
 * @typedef {object} PostureView
 * @property {null} session_id  the route echoes the query, and this read sends
 *   none — so it is always `null` here. Nothing may key behaviour on it
 * @property {string} control_target
 * @property {number|null} generation  the record's, and what a live server's
 *   `applied_generation` is compared against
 * @property {{kind: string, pid: number|null, port: number|null,
 *             self: boolean}|null} owner
 * @property {ServerRow[]} servers
 * @property {boolean} store_available
 * @property {{pid: number|null, target: string, session: string|null,
 *             surface: string, kernel_id: string|null, started_at: string,
 *             age_s: number|null}[]} execution_in_flight
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
 * @property {'applied'|'refused'|'failed'|'expired'|string} status  the
 *   record's own vocabulary; `failed` and `expired` are this client's two
 *   syntheses
 * @property {number|null} [generation]  the record's generation after the
 *   answer — `null` is what makes a terminus a REFUSAL, since a refusal moves
 *   neither target nor generation
 * @property {string|null} [reason]  the refusal word (`request_expired` rides
 *   on `expired`, `switch_failed` on `failed`)
 * @property {string|null} [detail]
 * @property {string} [at]
 * @property {number|null} [age_s]
 * @property {boolean} [synthesized]  true only for the two blocks this client
 *   mints itself (see {@link synthesized})
 */

/** The idle re-read. See the module docstring for why it exists at all. */
export const IDLE_POLL_MS = 5000;

/** The re-read while a switch request is outstanding. */
export const FAST_POLL_MS = 500;

/**
 * How long a switch request may go unanswered before the client calls it
 * expired. A deployment that has stopped answering publishes no outcome and
 * moves no generation, so `switching…` would never end on its own.
 */
export const REQUEST_TTL_S = 30;

/**
 * The push the owning terminal raises when the control-context record or any
 * controls server's report has changed (`CONTROL_CONTEXT_FRAME` in
 * control_context_owner.py). Payload-free on purpose: every reader of the
 * context reads the record, and a frame carrying a copy of it would be a
 * second answer to "what is the context now".
 */
export const CONTROL_CONTEXT_FRAME = 'control_context';

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
 * the popover is `position: absolute` under its trigger and `.ctc-anchor` is
 * `position: relative` (terminal.css), so the pair keeps its own anchoring
 * wherever the layout parks the item's shell.
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
/** @type {{requestId: string, target: string|null, generation: number|null,
 *          startedAt: number}|null} */
let pending = null;

/**
 * An outcome this client minted because no published one is coming. Two, and
 * only two, are ever minted here — both are readings of what the route said,
 * never a second authority on what happened:
 *
 * - `request_expired`, for a request the deployment never answered. Dropped
 *   the moment a real terminus for that id turns up, so a slow answer that
 *   lands late still gets the last word.
 * - `switch_failed`, for a switch the record ACCEPTED and a controls server
 *   then failed to apply. There is no terminus coming for that — the record
 *   already wrote its own, saying applied — so it is dropped instead when the
 *   deployment moves past the generation it is about.
 */
/** @type {LastSwitch|null} */
let synthesized = null;

/**
 * Monotonic id for reads, so a slow one cannot overwrite a newer answer.
 *
 * A 5 s tick fires a GET, the operator confirms a switch, the POST's own
 * re-read lands — and the tick's older GET resolves afterwards carrying the
 * PRE-switch roster, repainting the chip as the machine the deployment just
 * left. Every read takes a ticket here and drops its answer if a later read
 * has since been issued.
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

/** False after teardown, so a late frame cannot revive a dead chip. */
let mounted = false;

/* ---- mount ---- */

/**
 * Mount the chip into a host and keep it current.
 *
 * WHERE the chip sits is the layout's answer, not this module's. On the
 * terminal page the header is a bar-item host and the `control-target` item
 * has a `.bar-item` shell that the server renders and bar-host.js reconciles
 * into the operator's order, so the chip only ever APPENDS into that shell.
 * It does not position itself relative to any neighbour — the palette trigger
 * it used to sit in front of now lives inside a shell of its own and is no
 * longer a child of the host, so a positional insert against it would both
 * misplace the chip and throw.
 *
 * `.header-actions` remains as a fallback host for a header rendered without
 * shells; appending there degrades to "somewhere in the action run", which is
 * strictly better than not mounting the chip at all.
 *
 * A caller that already knows its host passes one. That is how the JupyterLab
 * bar mounts the chip: its page has neither shell nor `.header-actions`, and
 * an element it owns is the only honest answer to where the chip goes.
 *
 * Idempotent: a second call re-renders rather than mounting a second chip, and
 * re-homes the existing anchor if the host turned up after a fallback mount.
 *
 * @param {{host?: HTMLElement|null, eventSourceFactory?: typeof createEventSource}} [opts]
 *   `eventSourceFactory` is injectable for tests the way session.js's
 *   `wireActivityStrip` injects it — happy-dom has no EventSource.
 */
export function initControlTargetChip({ host, eventSourceFactory = createEventSource } = {}) {
  const mountPoint = /** @type {HTMLElement|null} */ (
    host ??
      document.querySelector('[data-bar-item="control-target"]') ??
      document.querySelector('.header-actions')
  );
  if (!mountPoint) return;

  mounted = true;

  if (!chip || !anchor || !anchor.isConnected) {
    anchor = document.createElement('div');
    anchor.className = 'ctc-anchor';
    anchor.hidden = true;
    chip = buildChip();
    anchor.appendChild(chip);
    mountPoint.appendChild(anchor);
  } else if (anchor.parentElement !== mountPoint) {
    // A host that arrived after a fallback mount. Move the whole positioning
    // context rather than rebuilding it: the popover, the click listener and
    // the chip's current paint all ride along, and a rebuild would leave the
    // old chip orphaned in the header.
    mountPoint.appendChild(anchor);
  }

  if (!sse) sse = subscribeRefetchHints(eventSourceFactory);

  void refetch();
}

/**
 * Unmount the chip and release everything it holds: both timers, the event
 * subscription, the render subscribers and the DOM node.
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
  return `Could not read the control target (HTTP ${status}).`;
}

/**
 * Re-read the roster and repaint.
 *
 * The one refresh path: mount, the pushed frame, the activity hint, the idle
 * tick, the fast tick and every popover mutation all land here. The read
 * carries no session id — one record answers for the whole deployment.
 * @returns {Promise<void>}
 */
export async function refetch() {
  if (!chip) return;
  if (!pending) startIdlePolling();

  const seq = ++readSeq;
  try {
    const data = await targetRequest('/api/terminal/posture');
    // A newer read has been issued since this one left — most importantly the
    // re-read a mutation does — so this answer is already history.
    if (seq !== readSeq) return;
    view = data;
    reconcilePending();
  } catch (err) {
    // Same stale guard as the success path: a read a newer one superseded must
    // not blank the chip.
    if (seq !== readSeq) return;
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
 * The rule itself is {@link resolvePendingSwitch}, which is pure and lives
 * with the rest of the derived facts; all this does is act on its four
 * answers. It is never matched on the TARGET — that would land on another
 * tab's switch to the same machine — nor on "the control_target changed",
 * which would land on a switch the agent made for its own reasons.
 */
function reconcilePending() {
  dropOvertakenSynthesis();
  if (!pending) return;
  const outcome = resolvePendingSwitch(view, pending);
  if (outcome.state === 'waiting') {
    expireIfOverdue();
    return;
  }
  if (outcome.state === 'failed') {
    synthesized = {
      request_id: pending.requestId,
      target: pending.target ?? undefined,
      status: 'failed',
      generation: pending.generation,
      reason: REASON_SWITCH_FAILED,
      detail: switchFailureNote(outcome.pid, outcome.detail),
      at: new Date().toISOString(),
      age_s: 0,
      synthesized: true,
    };
  }
  settlePending();
}

/**
 * Drop a synthesised outcome the deployment has overtaken.
 *
 * **A terminus alone is not enough to retire one**, and that is the whole
 * point. Under the fleet rule the record's `applied` terminus is normally
 * ALREADY in view when a deadline passes — the wait was on the servers, not on
 * the record — so retiring on it would flip the chip to `✓ switched` while a
 * server is still on the old generation. That is the exact misreport this task
 * exists to prevent, and a connector rebuild outlasting `REQUEST_TTL_S` is an
 * ordinary way to reach it.
 *
 * So the question asked here is the one that would have ended the wait, put to
 * the block instead of to a pending request: has the fleet arrived
 * ({@link resolvePendingSwitch} answering `applied`), has the record given a
 * verdict that moved nothing (`answered` — a refusal that landed after the
 * deadline), or has the deployment moved on to a later generation? Any of the
 * three retires it; nothing else does.
 *
 * The exception is a block with **no generation**, which is a request the POST
 * answered without one. There is no fleet criterion to apply, so any published
 * outcome for that request is the answer — the pre-fleet rule, kept for exactly
 * the case the fleet rule cannot speak to.
 */
function dropOvertakenSynthesis() {
  if (!synthesized) return;
  const then = typeof synthesized.generation === 'number' ? synthesized.generation : null;
  if (then === null) {
    if (view?.last_switch?.request_id === synthesized.request_id) synthesized = null;
    return;
  }
  const now = view?.generation;
  if (typeof now === 'number' && now > then) {
    synthesized = null;
    return;
  }
  const settled = resolvePendingSwitch(view, {
    requestId: synthesized.request_id,
    generation: then,
  });
  if (settled.state === 'applied' || settled.state === 'answered') synthesized = null;
}

/** Stop waiting: back to the idle cadence, and repaint without `switching…`. */
function settlePending() {
  pending = null;
  stopFastPolling();
  startIdlePolling();
}

/**
 * Call an unanswered request expired once its TTL has passed.
 *
 * A deployment that has died publishes no terminus and moves no generation, so
 * nothing the rule reads will ever change and `switching…` would be permanent.
 * The synthesised outcome is flagged `synthesized` so a renderer can tell a
 * local deadline from a published verdict.
 *
 * It carries the `generation` it was waiting on, which is what
 * {@link dropOvertakenSynthesis} needs to ask the fleet whether the switch has
 * landed since. Deliberately NOT a snapshot of what the view said at mint
 * time: reads in flight when the deadline passes make any such flag a guess,
 * and the question can simply be asked again later.
 */
function expireIfOverdue() {
  if (!pending) return;
  if (Date.now() - pending.startedAt < REQUEST_TTL_S * 1000) return;
  synthesized = {
    request_id: pending.requestId,
    target: pending.target ?? undefined,
    status: 'expired',
    generation: pending.generation,
    reason: 'request_expired',
    detail: null,
    at: new Date().toISOString(),
    age_s: 0,
    synthesized: true,
  };
  settlePending();
}

/* ---- polling ---- */

/**
 * Start the idle re-read, if it is not already running.
 *
 * The FALLBACK, not the normal path: the owning terminal pushes
 * {@link CONTROL_CONTEXT_FRAME} on any change, and this tick is what covers a
 * deployment served without a broadcaster and a stream that dropped without
 * anyone noticing.
 *
 * It stops itself once there is nothing left to paint — a chip no longer in
 * the document. Checking the DOM rather than trusting a teardown call means no
 * code path can leak a timer that outlives the chip.
 */
function startIdlePolling() {
  if (idleTimer !== null) return;
  idleTimer = setInterval(() => {
    if (!chip || !chip.isConnected) {
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
    if (!chip || !chip.isConnected) {
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

/* ---- the refetch hints ---- */

/**
 * Subscribe to the server event stream for anything that means "read again".
 *
 * Two frames qualify, and both are HINTS and only hints — nothing is read out
 * of either, because the route is the only thing that knows what the context
 * now is:
 *
 * - `{type: 'control_context'}`, pushed by the owning terminal when the record
 *   or any controls server's report has changed. This is the one that matters:
 *   it covers a switch, a posture toggle, and a server arriving at the
 *   generation the operator is waiting on, whoever caused them.
 * - an `agent_activity` frame naming the switch tool, which stays because it
 *   is broadcast on deployments whose terminal owns no context to push from.
 *   Its `detail` is the agent's narration of a switch that may belong to any
 *   client on this stream, so only the tool name is read.
 *
 * One subscription, and api.js shares one socket per URL across the page's
 * modules, so the chip adds no connection of its own to the browser's per-host
 * cap. Its own subscription rather than a seam through panel-manager,
 * following session.js's `wireActivityStrip`: the chip must work on a page
 * where no panel workspace ever boots.
 * @param {typeof createEventSource} factory
 */
function subscribeRefetchHints(factory) {
  return factory('/api/files/events', {
    onMessage: (data) => {
      if (!data || typeof data !== 'object') return;
      if (!isRefetchHint(data)) return;
      if (mounted) void refetch();
    },
  });
}

/** @param {any} frame @returns {boolean} */
function isRefetchHint(frame) {
  if (frame.type === CONTROL_CONTEXT_FRAME) return true;
  return frame.type === AGENT_ACTIVITY_FRAME && frame.tool === TARGET_SWITCH_TOOL;
}

/* ---- render ---- */

/**
 * The roster as every renderer should read it: the route's answer, with a
 * locally synthesised outcome standing in for a `last_switch` that would
 * otherwise misdescribe what happened. Null until the first successful read.
 *
 * The overlay is unconditional because {@link dropOvertakenSynthesis} has
 * already removed any synthesis the route has overtaken — including the case
 * that matters most: a terminus saying `applied` for the very request a
 * controls server then failed to apply, which would read as `✓ switched` on a
 * switch that did not land.
 * @returns {PostureView|null}
 */
export function getState() {
  if (!view) return null;
  return synthesized ? { ...view, last_switch: synthesized } : view;
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
 * Called by the popover with what its POST returned. Arms the fast poll and
 * the local expiry deadline, and puts the chip in `switching…` — the only
 * piece of chip state that does not come from the route, and it is a question
 * ("what did I ask for?") the route cannot answer.
 *
 * The `generation` is what makes the wait end honestly: the record's terminus
 * says the switch was accepted, and this is the number every live controls
 * server has to report before it has actually landed
 * ({@link resolvePendingSwitch}). Without one, only a refusal or the local TTL
 * can end the wait.
 * @param {string} requestId
 * @param {string|null} [target]  the row that was asked for, so a synthesised
 *   outcome can name it
 * @param {number|null} [generation]  the generation the 202 answered with
 */
export function markPending(requestId, target = null, generation = null) {
  if (!requestId) return;
  pending = { requestId, target, generation, startedAt: Date.now() };
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
 * own parent). The popover is `position: absolute` under its trigger and only
 * `.ctc-anchor` is guaranteed `position: relative` — appending the popover
 * anywhere else would anchor it to whatever ancestor happens to be positioned.
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
 * The row the chip speaks for: the one the deployment stands on.
 *
 * Two fallbacks, in the order of how much they claim. The baseline row is what
 * the deployment would stand on and is the route's own answer when
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
    rows.find((r) => r.target === state.control_target) ??
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
 * The kind of machine the deployment stands on — `live`, `standin`, `va` or
 * `simulated` — or `null` while nothing has been read yet.
 *
 * The chip's own two steps, the row it speaks for and that row's kind, handed
 * out as one answer. A consumer that phrases what a read or a write MEANS here
 * is describing the machine the chip names a few pixels away, so it must not
 * walk the roster a second time: a second derivation is how the two come to
 * disagree, and disagreeing about whether this is the facility's own machine
 * is the one thing this surface exists to prevent.
 *
 * Recomputed per call from {@link getState}, so pairing it with
 * {@link subscribe} follows a switch without any state of its own.
 * @returns {string|null}
 */
export function activeKind() {
  const state = getState();
  if (!state) return null;
  const row = activeRow(state);
  return row ? kindAttr(row) : null;
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
      // A question about the CONTEXT, not about the ceiling: whether a change
      // made here would be written at all, or refused because there is no
      // context to write or because another terminal holds it. When it would
      // be refused the chip is dimmed and still opens — the roster is worth
      // reading even where every gesture on it is refused.
      chip.dataset.enforceable = String(contextWritable(state));
      if (pending) chip.dataset.pending = 'true';
      else chip.removeAttribute('data-pending');
      shortEl.textContent = displayName(row, kindAttr(row));
      // `data-state` keeps the real state under a pending request: the dot and
      // the border still describe the machine the deployment is on until the
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
