// @ts-check
/* OSPREY Web Terminal — Control-Target Chip
 *
 * The one-glance answer to "if the agent writes now, where does it land, and
 * will it be refused?". A chip reading `● Rehearsal · writes asked ▾`: the name
 * of the machine this deployment stands on, the effective write state on THAT
 * machine, and a dot whose colour says how much a write there matters and
 * whose shape says who is holding writes back. On the hub it is the button of
 * the sidebar's control-system block (sidebar-target.js lays it out); on a
 * popped-out JupyterLab page it is a 28 px chip in the Lab menu bar
 * (control-target-lab-bar.js). The popover behind it
 * (control-target-popover.js) lists every configured target and owns every
 * gesture that CHANGES something; this module owns the chip, the read, and the
 * state every renderer shares.
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
 * `api.js` and `activity-format.js` — the JupyterLab bar suite pins the
 * closure this module belongs to — and the JupyterLab bar mounts it by handing
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
  STOP_NOT_TAKEN_SENTENCE,
  contextWritable,
  displayName,
  isPosition,
  machineKind,
  reasonPhrase,
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
 * @property {string} short_label  the chip's word: LIVE / STAND-IN / VIRTUAL,
 *   derived server-side from `machine_kind`
 * @property {'real'|'rehearsal'|'simulator'} machine_kind  the write verdict's
 *   kind, read from the controls server's published report (C-KIND) — a
 *   stand-in no live server vouched for is `real`
 * @property {string} [display_name]  the deployment's configured name, or `''`
 * @property {string} endpoint
 * @property {boolean} real_machine  true for the facility's own machine AND
 *   for a stand-in. Nothing here keys on it: the kind is `machine_kind`'s
 * @property {boolean} active  the target the deployment stands on
 * @property {boolean} is_baseline
 * @property {boolean} available_now  whether a Switch is offered
 * @property {string|null} reason  the switch tool's own refusal code when not
 * @property {string|null} reason_detail  the eligibility verdict's operator
 *   sentence for that code — the popover's tooltip, never its row text
 * @property {'off'|'asked'|'allowed'|null} switch  this login's stored position
 *   for the machine; `null` when the record cannot be read
 * @property {boolean} prompt  whether this login's next write here prompts: the
 *   effective position, so a stored `allowed` capped at `asked` answers true
 * @property {boolean} write_allowed  the verdict a new write by this login
 *   would meet
 * @property {string|null} write_reason  the verdict's refusal reason, or null
 * @property {string|null} write_phrase  that refusal's operator sentence —
 *   what stopped the write and who lifts it (C-PHRASE)
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
 * @property {number[]} [children]  the connector-host PIDs it holds; an empty
 *   list is a server serving nothing, which the convergence wait passes over
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
 * @property {string|null} switch_epoch  the login's switch epoch as read, a
 *   decimal string kept opaque (nanoseconds exceed a JS number's exact range);
 *   a turn-on carries it back unchanged so a stop that landed since refuses it
 *   (C-POST)
 * @property {boolean} any_armed  whether any machine's switch is armed for this
 *   login — what enables "Turn all writes off"
 * @property {{readonly_run?: boolean, tier?: string}} [locks]  the locks no
 *   chip gesture lifts
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
 * Dispatched on the chip element when the operator clicks it. The chip flips
 * its own `aria-expanded` and nothing else; the popover module listens for
 * this and opens or closes itself, and the design system's popover.js keeps
 * `aria-expanded` in step for every close it drives on its own (a press
 * outside, Escape, a resize).
 *
 * `detail.expanded` carries the state the chip has just moved TO.
 */
export const CHIP_TOGGLE_EVENT = 'osprey-control-target-toggle';

/**
 * The chip's wrapper inside its host: what hides while there is nothing to
 * say, so an empty box never holds the host's layout open.
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

/* ---- mount ---- */

/**
 * Mount the chip into `host` and keep it current.
 *
 * WHERE the chip sits is the caller's answer, not this module's: the hub hands
 * it the sidebar's control-system block (`#sidebar-target`), and the
 * JupyterLab bar an element of its own. The chip only ever APPENDS into its
 * host; it never positions itself relative to a neighbour.
 *
 * Idempotent: a second call re-renders rather than mounting a second chip, and
 * re-homes the existing anchor when called with a different host.
 *
 * @param {{host?: HTMLElement|null}} [opts]
 */
export function initControlTargetChip({ host } = {}) {
  if (!host) return;

  if (!chip || !anchor || !anchor.isConnected) {
    anchor = document.createElement('div');
    anchor.className = 'ctc-anchor';
    anchor.hidden = true;
    chip = buildChip();
    anchor.appendChild(chip);
    host.appendChild(anchor);
  } else if (anchor.parentElement !== host) {
    // A different host. Move the wrapper rather than rebuilding it: the click
    // listener and the chip's current paint ride along, and a rebuild would
    // leave the old chip orphaned.
    host.appendChild(anchor);
  }

  if (!sse) sse = subscribeRefetchHints();

  void refetch();
}

/**
 * Unmount the chip and release everything it holds: both timers, the event
 * subscription, the render subscribers and the DOM node.
 */
export function teardownControlTargetChip() {
  stopIdlePolling();
  stopFastPolling();
  if (sse) {
    sse.stop();
    sse = null;
  }
  // The wrapper goes with it: an empty one is a stray box in its host.
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
  el.setAttribute('aria-haspopup', 'dialog');
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
 * screen — a chip that toggled something on click would be a one-click
 * path to arming writes on a real machine.
 */
function onChipClick() {
  if (!chip) return;
  const expanded = chip.getAttribute('aria-expanded') !== 'true';
  chip.setAttribute('aria-expanded', String(expanded));
  chip.dispatchEvent(new CustomEvent(CHIP_TOGGLE_EVENT, { bubbles: true, detail: { expanded } }));
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
 * A refusal keeps its status and its `detail` beside the sentence: the posture
 * route answers `{detail: {reason, phrase?, epoch?, stamped?, taken?}}`, and a
 * stamped 409 is retried with the epoch it carries, which a message alone
 * could not say.
 *
 * @param {string} path
 * @param {{method?: string, json?: any}} [opts]
 * @returns {Promise<any>} the parsed body
 * @throws {TargetRequestError} on a non-OK response, carrying the server's own
 *   sentence, the status and the detail
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
  if (!resp.ok) {
    const detail = body && typeof body === 'object' ? body.detail : null;
    throw new TargetRequestError(refusalMessage(body, resp.status), resp.status, detail ?? null);
  }
  return body;
}

/**
 * A refused request: the sentence to show, plus what the route answered it
 * with. `detail` is the body's `detail` as sent — a dict for these routes, a
 * string for most others, a list for a body FastAPI could not parse — or
 * `null` when the body carried none.
 */
export class TargetRequestError extends Error {
  /** @param {string} message @param {number} status @param {any} detail */
  constructor(message, status, detail) {
    super(message);
    this.name = 'TargetRequestError';
    /** @type {number} */
    this.status = status;
    /** @type {any} */
    this.detail = detail;
  }
}

/* ---- the switch gesture (C-POST) ---- */

/**
 * What a turn-on a stop overtook says. The 409 carries no phrase of its own:
 * it is an answer about the request, and the operator is owed the next step.
 */
export const STOPPED_SINCE_SENTENCE =
  'Writes were turned off after this turn-on began — nothing was turned on. Click again to turn them on.';

/**
 * POST one switch gesture: `{session_id, target, state, epoch, waived}`, where
 * `epoch` is the `switch_epoch` string the chip read (`null` when it read none),
 * sent back exactly as read: it is never turned into a number. `waived` says the
 * confirm was skipped; the route refuses it on a move no confirm may be skipped
 * for.
 *
 * A 409 that says it just STAMPED the record's first epoch is retried once
 * with that epoch and the same `waived` — the route never retries itself. Any
 * other 409 is a stop that landed after the gesture began, and is thrown: a
 * turn-on never silently re-arms after a stop.
 * @param {string|null} sessionId
 * @param {string} target  a configured target name, or `all` (off only)
 * @param {'off'|'asked'|'allowed'} position
 * @param {string|null} epoch
 * @param {boolean} [waived]
 * @returns {Promise<any>} the route's `{target, state, switch_epoch}`
 * @throws {TargetRequestError} the refusal, after at most one retry
 */
export async function postSwitch(sessionId, target, position, epoch, waived = false) {
  /** @param {string|null} at */
  const send = (at) =>
    targetRequest('/api/terminal/posture', {
      method: 'POST',
      json: { session_id: sessionId, target, state: position, epoch: at, waived },
    });
  try {
    return await send(epoch);
  } catch (err) {
    const detail = err instanceof TargetRequestError ? err.detail : null;
    const stamped =
      err instanceof TargetRequestError &&
      err.status === 409 &&
      detail?.stamped === true &&
      typeof detail.epoch === 'string';
    if (!stamped) throw err;
    return send(detail.epoch);
  }
}

/**
 * The sentence a refused switch gesture shows, and whether the confirm that
 * raised it should close. The route's `phrase` wins; a 409 is a stop that
 * overtook the gesture (the dialog's question no longer holds); a 503 on a
 * stop is the stop sentence even with no phrase — the one thing the operator
 * must not miss is that writes are still on.
 * @param {unknown} err
 * @param {'off'|'asked'|'allowed'} to
 * @returns {{message: string, close: boolean}}
 */
export function switchRefusal(err, to) {
  const e = err instanceof TargetRequestError ? err : null;
  const detail =
    e?.detail && typeof e.detail === 'object' && !Array.isArray(e.detail) ? e.detail : {};
  if (e?.status === 409) return { message: STOPPED_SINCE_SENTENCE, close: true };
  /** @param {string} message */
  const say = (message) => ({ message, close: false });
  if (typeof detail.phrase === 'string' && detail.phrase.trim()) return say(detail.phrase);
  if (e?.status === 503 && to === 'off') return say(STOP_NOT_TAKEN_SENTENCE);
  if (typeof detail.reason === 'string' && detail.reason) return say(reasonPhrase(detail.reason));
  return say(err instanceof Error ? err.message : String(err));
}

/**
 * The most specific human sentence a refusal body carries. Handles every
 * shape rather than guessing at one: the posture route's dict detail (its
 * `phrase`), the target route's (its `message`), the plain string detail every
 * other route uses, and a body with neither — where the status code is all
 * there is to say.
 * @param {any} body @param {number} status @returns {string}
 */
function refusalMessage(body, status) {
  const detail = body && typeof body === 'object' ? body.detail : null;
  if (typeof detail === 'string' && detail.trim()) return detail;
  if (detail && typeof detail === 'object' && !Array.isArray(detail)) {
    for (const key of ['phrase', 'message']) {
      if (typeof detail[key] === 'string' && detail[key].trim()) return detail[key];
    }
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
 * cap. Its own subscription rather than a seam through panel-manager: the
 * chip must work on a page where no panel workspace ever boots.
 */
function subscribeRefetchHints() {
  return createEventSource('/api/files/events', {
    onMessage: (data) => {
      if (!data || typeof data !== 'object') return;
      if (!isRefetchHint(data)) return;
      void refetch();
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
 * The `data-target-kind` value for one row: its `machine_kind`, exactly as GET
 * published it.
 *
 * The route already made this decision from the controls server's report
 * (C-KIND) — never from the target name or the label's shape — so a `standin`
 * no live server vouched for arrives as `real` and renders as the real
 * machine. A row with no recognised `machine_kind` reads as `real` too: the
 * loud answer is the direction this stack must fail in.
 *
 * Exported for the popover, which keys each ROW's dot and tints on the same
 * attribute: one derivation for the chip and the row it speaks for, so a
 * stand-in cannot render as the facility's own machine in one of them.
 * @param {TargetRow} row
 * @returns {string}
 */
export function kindAttr(row) {
  return machineKind(row);
}

/**
 * The state word for one row: `off`, `asked`, `allowed` or `locked`.
 *
 * The EFFECTIVE state, derived from `switch`, `write_allowed` and `prompt` only
 * — never from the legacy `posture` mirror. `off` is a switch that is off, which
 * one click undoes; `locked` is an armed switch that is still refused (its row
 * shows `write_phrase` and keeps the off control), or a switch nobody can read.
 * A single "no" word would leave an operator clicking a toggle that was never
 * going to open. Otherwise `prompt` decides, so a stored `allowed` capped at
 * `asked` reads `asked`: the word says what the next write meets, not what was
 * stored. A row that does not say reads `allowed`, the louder of the two.
 *
 * Exported for the popover, whose rows and confirms have to name the state in
 * the same words the chip does — the confirm for a switch tells the operator
 * the state they will land in, and it must be the word the chip shows a
 * moment later.
 * @param {TargetRow} row
 * @returns {'off'|'asked'|'allowed'|'locked'}
 */
export function stateWord(row) {
  if (row?.switch === 'off') return 'off';
  if (!isPosition(row?.switch) || row.write_allowed !== true) return 'locked';
  return row.prompt ? 'asked' : 'allowed';
}

/**
 * The kind of machine the deployment stands on — `real`, `rehearsal` or
 * `simulator` — or `null` while nothing has been read yet.
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
 * What the chip says right now: the machine's name, the write state in words
 * (`switching…` under a pending request), and the kind and state word its
 * attributes carry — or null while nothing is known. The one derivation the
 * chip paints from and any surface that repeats the chip's words reads (the
 * sidebar block's tooltip), so the two cannot word the same machine apart.
 * @returns {{name: string, state: string, kind: string, word: 'off'|'asked'|'allowed'|'locked'}|null}
 */
export function targetReading() {
  const state = getState();
  const row = state ? activeRow(state) : null;
  if (!row) return null;
  const kind = kindAttr(row);
  const word = stateWord(row);
  return {
    name: displayName(row, kind),
    state: pending ? 'switching…' : statePhrase(word),
    kind,
    word,
  };
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
    const reading = targetReading();
    const row = state ? activeRow(state) : null;
    if (!state || !row || !reading) {
      // The ANCHOR is what hides: hiding the chip alone would leave an empty
      // positioning context holding its host's layout open.
      if (anchor) anchor.hidden = true;
      chip.hidden = true;
      chip.removeAttribute('data-pending');
    } else {
      if (anchor) anchor.hidden = false;
      chip.hidden = false;
      chip.dataset.targetKind = reading.kind;
      chip.dataset.state = reading.word;
      // A question about the CONTEXT, not about the switch: whether a target
      // switch made here would be written at all, or refused because there is
      // no context to write or because another terminal holds it. When it
      // would be refused the chip is dimmed and still opens — the roster is
      // worth reading, and this login's own switches still work.
      chip.dataset.enforceable = String(contextWritable(state));
      if (pending) chip.dataset.pending = 'true';
      else chip.removeAttribute('data-pending');
      shortEl.textContent = reading.name;
      // `data-state` keeps the real state under a pending request: the dot and
      // the border still describe the machine the deployment is on until the
      // switch actually lands.
      stateEl.textContent = reading.state;
      chip.title = row.label || row.target;
      chip.setAttribute(
        'aria-label',
        `Control target: ${reading.name} · ${pending ? 'switching' : reading.state}`
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
