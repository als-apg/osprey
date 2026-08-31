// @ts-check
/* OSPREY Web Terminal — Control-Target Derived Facts
 *
 * The pure half of the control-target popover: every function here derives a
 * fact an operator reads — a machine's display name and what writing to it
 * means, a state phrase, a refusal phrase, a lock reason, the banner note, a
 * reachability word — from the state the chip publishes, and nothing here
 * touches the DOM, the chip, or any module state. The popover
 * (control-target-popover.js) owns the rows, the gestures and the confirms;
 * this module owns what the rows SAY, so the wording can be read (and tested)
 * without a popover on screen.
 *
 * **One naming rule.** Every machine is named by what writing to it does and
 * what it is for — never by how it is wired. The deployment may override the
 * name per target (`control_system.target_display_names`, published by the
 * route as `display_name`); the defaults live in {@link KIND_WORDS}. The
 * server's own label ("LIVE MACHINE (stand-in)") stays on tooltips, where the
 * implementation vocabulary belongs.
 *
 * **No process claims.** Nothing here says whether a write will ask for
 * approval, what limits apply, or who is prompted — all of that is deployment
 * configuration this module cannot see. What it may state is consequence:
 * writes move hardware, or nothing moves.
 */

/** `available_now` reason a chat session's rows carry (routes/websocket.py). */
export const REASON_CHAT_SESSION = 'chat_session';

/** The refusal word for a row whose Switch is missing because the store is. */
export const REASON_STORE_UNAVAILABLE = 'store_unavailable';

/**
 * The operator word for each machine kind, keyed on the `data-target-kind`
 * value {@link module:control-target-chip.kindAttr} derives. Used wherever a
 * row has no configured `display_name`. Consequence-first on purpose: an
 * operator who has never met a soft-IOC can still tell the machine that moves
 * from the ones that cannot.
 * @type {Record<string, string>}
 */
export const KIND_WORDS = {
  live: 'Real machine',
  standin: 'Rehearsal',
  va: 'Simulator',
  simulated: 'Demo',
};

/**
 * The one-line descriptor under each name: what writing to this machine does.
 * These are statements of consequence, not of process — nothing about
 * approvals or limits, which are configuration this file cannot see.
 * @type {Record<string, string>}
 */
export const KIND_DESCRIPTORS = {
  live: 'Writes move hardware',
  standin: "Copy of the real machine's controls · nothing moves",
  va: 'Physics model · nothing moves',
  simulated: 'Mock data · nothing moves',
};

/**
 * The `reason` codes that all mean "the deployment has not authored this
 * machine yet". On a stock render this is the live target's DELIBERATE state —
 * authoring it is the go-live edit — so the descriptor says "not set up"
 * rather than implying a fault.
 */
export const NOT_SET_UP_REASONS = new Set([
  'connector_block_missing',
  'gateways_missing',
  'probe_channel_missing',
]);

/**
 * The name a row renders: the deployment's configured name where one is set,
 * the kind's own word otherwise. The server label survives untouched for the
 * tooltip; falling back to it here would put "LIVE MACHINE (stand-in)" back on
 * the one surface this vocabulary exists to clean up, so the label is last.
 * @param {any} row
 * @param {string} kind  a `data-target-kind` value (kindAttr's answer)
 * @returns {string}
 */
export function displayName(row, kind) {
  const configured = String(row?.display_name || '').trim();
  if (configured) return configured;
  return KIND_WORDS[kind] || String(row?.label || row?.target || '');
}

/**
 * The consequence line under the name. A live machine nothing has authored
 * yet reads "Not set up yet" instead of promising hardware that is not there.
 * @param {any} row
 * @param {string} kind
 * @returns {string}
 */
export function descriptor(row, kind) {
  if (kind === 'live' && NOT_SET_UP_REASONS.has(String(row?.reason || ''))) {
    return 'Not set up yet';
  }
  return KIND_DESCRIPTORS[kind] || '';
}

/**
 * The tone the descriptor renders in: `hazard` for the one line that names
 * hardware moving, `null` for every line that promises nothing moves. A live
 * machine nothing has authored yet is not a hazard — its descriptor says "not
 * set up", and painting that red would flag a healthy render as broken.
 * @param {any} row
 * @param {string} kind
 * @returns {'hazard'|null}
 */
export function descriptorTone(row, kind) {
  return kind === 'live' && descriptor(row, kind) === KIND_DESCRIPTORS.live ? 'hazard' : null;
}

/**
 * The display phrase for a state word. The internal three-word vocabulary
 * (`writes` / `sandbox` / `read-only`, {@link module:control-target-chip.stateWord})
 * stays the wire and CSS truth; what an operator reads is on / off / locked —
 * off is the state you put it in and one click undoes, locked is the
 * deployment's and no click here moves it.
 * @param {'writes'|'sandbox'|'read-only'|string} word
 * @returns {string}
 */
export function statePhrase(word) {
  if (word === 'writes') return 'writes on';
  if (word === 'sandbox') return 'writes off';
  return 'writes locked';
}

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
 * The three `not set up` codes are one phrase on purpose: all three mean the
 * deployment has not authored this machine yet, and the distinction between
 * them belongs to the tooltip, not the row.
 * @type {Record<string, string>}
 */
export const REASON_PHRASES = {
  connector_block_missing: 'not set up',
  gateways_missing: 'not set up',
  probe_channel_missing: 'not set up',
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
 * unavailability reason. Its write toggles are untouched — the posture store
 * is keyed on the session, not on the topology.
 * @param {any[]} rows
 */
export function isChatSession(rows) {
  return rows.length > 0 && rows.every((row) => row.reason === REASON_CHAT_SESSION);
}

/**
 * Whether nothing this session can do would turn this row's writes on.
 *
 * The signature of a read-only run, read off the columns the route publishes:
 * the render's ceiling is up and this session has not narrowed the row, and
 * writes are STILL off. `effective` is `ceiling ∧ ¬readonly_run ∧ entry ≠
 * sandbox`, so with the other two terms true only the run can be holding it.
 * @param {any} row
 */
export function writesHeldByTheRun(row) {
  return Boolean(row.ceiling_writes) && row.posture !== 'sandbox' && !row.effective;
}

/**
 * Why this row's writes cannot be turned on or off, or `null` when they can.
 *
 * Ordered from the widest cause to the narrowest, so an operator reads the one
 * they could act on: a store that cannot record anything outranks a run that
 * would ignore it, which outranks a deployment that never arms this target,
 * which outranks a gateway table with nowhere to narrow TO. Spoken in the
 * operator's words; the machine vocabulary stays in the route payload.
 * @param {any} row
 * @param {any} state
 * @returns {string|null}
 */
export function lockReason(row, state) {
  if (!state?.store_available) return 'changes cannot be recorded right now';
  if (!state.enforceable) return 'changes here would not reach the agent';
  if (writesHeldByTheRun(row)) return 'the whole deployment is running read-only';
  if (!row.ceiling_writes) return 'kept read-only by the deployment';
  // Narrowing this row would select a gateway role the deployment has not
  // configured. The route only reports it for a row a narrowing would CHANGE,
  // so when it is set the only move on offer is the blocked one.
  if (row.narrowing_refusal) return 'no read-only endpoint configured';
  return null;
}

/**
 * The banner across the top of the popover, which does not exist in the plain
 * case: absence is the good news, and only a session that is not plain gets a
 * sentence. Same order as {@link lockReason} — the widest abnormal fact wins.
 * @param {any} state
 * @param {any[]} rows
 * @returns {{text: string, tone: string}|null}
 */
export function bannerNote(state, rows) {
  if (!state.store_available) {
    return { text: 'Changes cannot be recorded right now — the posture store is unavailable.', tone: 'error' };
  }
  if (!state.enforceable) {
    return { text: 'Changes here will not reach the agent yet.', tone: 'warn' };
  }
  if (rows.length > 0 && rows.every(writesHeldByTheRun)) {
    return { text: 'The whole deployment is running read-only.', tone: 'warn' };
  }
  return null;
}

/**
 * The reachability exception, or `null` for every state that needs no words.
 *
 * "Connected" on every row is noise an operator learns to skip; the only
 * reachability fact that changes a decision is that a machine is NOT
 * answering, so that is the only one rendered. The measured word (`down`,
 * `stale`), the age and the role stay on the element's `title` for whoever
 * runs the deployment.
 * @param {any} reachability
 * @returns {{state: string, text: string, title: string}|null}
 */
export function reachException(reachability) {
  const rc = reachability && typeof reachability === 'object' ? reachability : {};
  const measured = typeof rc.state === 'string' && rc.state ? rc.state : 'unknown';
  if (measured !== 'down' && measured !== 'stale') return null;
  const age = typeof rc.age_s === 'number' ? ` · ${rc.age_s} s` : '';
  const parts = [`${measured}${age}`];
  if (rc.role) parts.push(`${rc.role} endpoint`);
  if (measured === 'stale') parts.push('last probe older than the prober interval');
  return {
    state: measured,
    text: measured === 'down' ? 'not answering' : 'may be stale',
    title: parts.join(' · '),
  };
}

/**
 * One emphasised run inside a confirm body line. The popover renders it as a
 * `<strong>`; kept as data here so this module stays DOM-free and the whole
 * of a confirm's wording can be asserted without a dialog on screen.
 * @typedef {string | {em: string}} ConfirmRun
 */

/**
 * The consequence sentence a confirm about the facility's own machine
 * carries, and only that machine: writing there moves hardware. Deliberately
 * the whole of what the confirms say about safety — whether a write prompts
 * for approval, and what limits apply, are deployment configuration this
 * module cannot see, so it makes no claim about them.
 * @param {string} kind
 * @returns {string|null}
 */
export function hardwareNote(kind) {
  return kind === 'live' ? 'Real machine — writes move hardware.' : null;
}

/**
 * The tooltip a machine's name carries: the server's own label (the
 * implementation truth — "LIVE MACHINE (stand-in)"), the endpoint, and the
 * measured reachability. The at-rest surface names consequences; hover is
 * where the machine vocabulary lives.
 * @param {any} row
 * @returns {string}
 */
export function identTitle(row) {
  const parts = [];
  if (row.label) parts.push(String(row.label));
  if (row.endpoint) parts.push(String(row.endpoint));
  const reach = reachException(row.reachability);
  if (reach) parts.push(reach.title);
  return parts.join(' · ');
}

/**
 * Everything the turn-writes-on confirm says. Only this direction asks —
 * turning off removes reach and is undone by a click; turning on is the
 * gesture after which a write the agent makes can land. The title names the
 * machine, so the body does not name it again: one line for the scope and
 * the endpoint, one for when it takes hold.
 * @param {any} row
 * @param {string} kind
 * @returns {{title: string, body: ConfirmRun[][], live: string|null, confirmLabel: string}}
 */
export function turnOnConfirm(row, kind) {
  return {
    title: `Turn writes on for ${displayName(row, kind)}?`,
    body: [
      ['For ', { em: 'your session' }, row.endpoint ? ` · ${row.endpoint}.` : '.'],
      ['Takes effect at the next write — nothing restarts.'],
    ],
    live: hardwareNote(kind),
    confirmLabel: 'Turn writes on',
  };
}

/**
 * Everything the switch confirm says. The first line states the consequence —
 * where every control read and write goes next. The second names the write
 * state the session will ARRIVE in, because writes on/off is per machine and
 * does not travel; the word is stateWord's own, so the dialog and the chip a
 * moment later can never disagree.
 * @param {any} row
 * @param {string} kind
 * @param {'writes'|'sandbox'|'read-only'} word  stateWord's answer for the row
 * @returns {{title: string, body: ConfirmRun[][], live: string|null, confirmLabel: string}}
 */
export function switchConfirm(row, kind, word) {
  const arrival =
    word === 'writes'
      ? ['Writes are ', { em: 'on' }, ' there for your session.']
      : word === 'sandbox'
        ? [
            'Writes are ',
            { em: 'off' },
            ' there for your session — nothing moves until you turn them on.',
          ]
        : ['Writes are ', { em: 'locked' }, ' read-only there by the deployment.'];
  return {
    title: `Switch to ${displayName(row, kind)}?`,
    body: [
      ['All control reads and writes go to ', { em: String(row.endpoint || row.target) }, '.'],
      arrival,
    ],
    live: word === 'writes' ? hardwareNote(kind) : null,
    confirmLabel: 'Switch',
  };
}
