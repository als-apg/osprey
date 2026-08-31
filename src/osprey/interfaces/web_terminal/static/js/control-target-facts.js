// @ts-check
/* OSPREY Web Terminal — Control-Target Derived Facts
 *
 * The pure half of the control-target popover: every function here derives a
 * fact an operator reads — a refusal phrase, a lock reason, the head note, a
 * reachability word — from the state the chip publishes, and nothing here
 * touches the DOM, the chip, or any module state. The popover
 * (control-target-popover.js) owns the rows, the gestures and the confirms;
 * this module owns what the rows SAY, so the wording can be read (and tested)
 * without a popover on screen.
 */

/** `available_now` reason a chat session's rows carry (routes/websocket.py). */
export const REASON_CHAT_SESSION = 'chat_session';

/** The refusal word for a row whose Switch is missing because the store is. */
export const REASON_STORE_UNAVAILABLE = 'store_unavailable';

/**
 * Refusal code → the short phrase the operator reads.
 *
 * The route publishes the switch tool's machine codes so the popover and the
 * agent keep agreeing about the same refusal; what an OPERATOR reads is this
 * map's phrase, with the server's own sentence (`reason_detail`) on the
 * element's `title`. A code this map does not know renders verbatim — failing
 * informative is better than a blank where the reason should be, and it is how
 * a future code reaches the operator before this file has a phrase for it.
 *
 * The three `not configured` codes are one phrase on purpose: all three mean
 * the deployment has not authored this machine yet, which on a stock render is
 * the live target's DELIBERATE state (authoring it is the go-live edit), and
 * the distinction between them belongs to the tooltip, not the row.
 * @type {Record<string, string>}
 */
export const REASON_PHRASES = {
  connector_block_missing: 'not configured',
  gateways_missing: 'not configured',
  probe_channel_missing: 'not configured',
  target_unresolvable: 'unavailable',
  limits_posture: 'needs strict limits',
  operator_ack_missing: 'needs gateway ack',
  archive_belongs_to_standin: 'archive conflict',
  invented_history: 'no archive',
  standin_not_deployed: 'stand-in not deployed',
  selected_role_missing: 'no endpoint for role',
  [REASON_STORE_UNAVAILABLE]: 'store unavailable',
};

/**
 * The operator phrase for one refusal code. Sentences pass through untouched —
 * the gesture notes hold the server's own refusal sentences as well as codes,
 * and a sentence is already the most operator-readable form there is.
 * @param {unknown} code
 * @returns {string}
 */
export function reasonPhrase(code) {
  const word = String(code ?? '');
  return REASON_PHRASES[word] || word;
}

/**
 * Whether this session is a chat.
 *
 * A chat has no PTY and so no controls server of its own to address a switch
 * request to, which the route says by giving EVERY row `chat_session` as its
 * unavailability reason. Its toggles are untouched — the posture store is
 * keyed on the session, not on the topology.
 * @param {any[]} rows
 */
export function isChatSession(rows) {
  return rows.length > 0 && rows.every((row) => row.reason === REASON_CHAT_SESSION);
}

/**
 * Whether nothing this session can do would arm this row.
 *
 * The signature of a read-only run, read off the columns the route publishes:
 * the render's ceiling is up and this session has not narrowed the row, and
 * writes are STILL off. `effective` is `ceiling ∧ ¬readonly_run ∧ entry ≠
 * sandbox`, so with the other two terms true only the run can be holding it —
 * and the toggle is a readout either way, which is the whole of what the lock
 * has to say.
 * @param {any} row
 */
export function writesHeldByTheRun(row) {
  return Boolean(row.ceiling_writes) && row.posture !== 'sandbox' && !row.effective;
}

/**
 * Why this row's toggle cannot move, or `null` when it can.
 *
 * Ordered from the widest cause to the narrowest, so an operator reads the one
 * they could act on: a store that cannot record anything outranks a run that
 * would ignore it, which outranks a persona that never armed this target,
 * which outranks a gateway table with nowhere to narrow TO.
 * @param {any} row
 * @param {any} state
 * @returns {string|null}
 */
export function lockReason(row, state) {
  if (!state?.store_available) return 'store unavailable';
  if (!state.enforceable) return 'not enforceable';
  if (writesHeldByTheRun(row)) return 'readonly run';
  if (!row.ceiling_writes) return 'persona ceiling';
  // Narrowing this row would select a gateway role the deployment has not
  // configured. The route only reports it for a row a narrowing would CHANGE,
  // so when it is set the only move the toggle offers is the blocked one.
  if (row.narrowing_refusal) return 'no read-only endpoint';
  return null;
}

/**
 * The right of the head, which is empty in the plain case: absence is the good
 * news. Same order as {@link lockReason} — the widest fact about the session
 * that is not plain.
 * @param {any} state
 * @param {any[]} rows
 * @returns {{text: string, tone: string|null}}
 */
export function headNote(state, rows) {
  if (!state.store_available) return { text: 'posture store unavailable', tone: 'error' };
  if (!state.enforceable) {
    return { text: `not enforceable · ${state.enforceable_reason ?? 'unknown'}`, tone: 'warn' };
  }
  if (rows.length > 0 && rows.every(writesHeldByTheRun)) {
    return { text: 'readonly run · deployment-wide', tone: 'warn' };
  }
  if (isChatSession(rows)) return { text: 'chat session', tone: null };
  if (state.execution_in_flight) return { text: 'execution running', tone: 'warn' };
  return { text: '', tone: null };
}

/**
 * The reachability word and the age beside it.
 *
 * ONE vocabulary in the visible line, in the words an operator who has never
 * met a gateway can read: `connected`, `unreachable`, and `unknown` for
 * everything the prober could not vouch for. The measured word — `reached`,
 * `down`, `stale`, `not_applicable` — and the role it was measured on stay on
 * the element's `title`, which is where simple mode's LED-only row keeps the
 * whole sentence too.
 * @param {any} reachability
 * @returns {{state: string, text: string, title: string}}
 */
export function reachText(reachability) {
  const rc = reachability && typeof reachability === 'object' ? reachability : {};
  const measured = typeof rc.state === 'string' && rc.state ? rc.state : 'unknown';
  const word =
    measured === 'reached' ? 'connected' : measured === 'down' ? 'unreachable' : 'unknown';
  const age = typeof rc.age_s === 'number' ? ` · ${rc.age_s} s` : '';
  const parts = [`${measured}${age}`];
  if (rc.role) parts.push(`${rc.role} endpoint`);
  if (measured === 'stale') parts.push('last probe older than the prober interval');
  return { state: measured, text: `${word}${age}`, title: parts.join(' · ') };
}
