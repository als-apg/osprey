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

/** The refusal word for a row whose Switch is missing because the store is. */
export const REASON_STORE_UNAVAILABLE = 'store_unavailable';

/**
 * The refusal word for a row whose Switch is missing because something else
 * holds the control context. The same literal the write routes answer 409
 * with, so the row and the refusal an operator would have got by clicking
 * agree. Its phrase is deliberately kind-neutral; {@link contextHolderWords}
 * is what names the holder.
 */
const REASON_CONTEXT_OWNED_ELSEWHERE = 'context_owned_elsewhere';

/**
 * What can own the control context (`OWNER_WEB_TERMINAL` /
 * `OWNER_CONTROLS_SERVER` in osprey_connectors/control_context.py). A web
 * terminal is a page an operator can open, and its `port` is where; a controls
 * server is an agent's own process and carries no port, so it is named by pid.
 */
export const OWNER_WEB_TERMINAL = 'web_terminal';
export const OWNER_CONTROLS_SERVER = 'controls_server';

/**
 * The RECORD's word for a switch request it accepted (`SWITCH_APPLIED` in
 * osprey_connectors/control_context.py). Its sibling `SWITCH_REFUSED` is not
 * mirrored here: a refusal is recognised by its `generation` being null, which
 * is what makes it one, and any other non-applied verdict is read the same way.
 */
export const SWITCH_APPLIED = 'applied';

/**
 * A controls SERVER's own word for "I was asked for that generation and could
 * not get there" (`REPORT_FAILED` in osprey_connectors/control_context.py,
 * written as `SWITCH_FAILED` in mcp_server/control_system/target_state.py —
 * the reader's spelling and the writer's for one word). Deliberately distinct
 * from {@link SWITCH_APPLIED}, which is a request's terminus rather than a
 * server's progress.
 *
 * The one server-row status this module reads: `applying` and `applied` are
 * both answered by the generation comparison in {@link resolvePendingSwitch},
 * and only a failure is news that comparison would wait forever for.
 */
export const REPORT_FAILED = 'failed';

/** The refusal word a switch the fleet never applied is rendered under. */
export const REASON_SWITCH_FAILED = 'switch_failed';

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
 * The "read" capability phrase first contact and the tour put in front of an
 * operator, keyed on the same `data-target-kind` value as {@link KIND_WORDS}.
 * Keyed rather than fixed because the sentence is a promise about where the
 * numbers come from: a demo connector saying "read live machine values" would
 * be the one claim this vocabulary exists to prevent.
 * @type {Record<string, string>}
 */
export const KIND_READ_PHRASES = {
  live: 'read live machine values',
  standin: 'read values from the rehearsal copy',
  va: 'read values from the simulator',
  simulated: 'read demo data',
};

/**
 * The one-line descriptor under each name: what writing to this machine does.
 * These are statements of consequence, not of process — nothing about
 * approvals or limits, which are configuration this file cannot see.
 * @type {Record<string, string>}
 */
const KIND_DESCRIPTORS = {
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
const NOT_SET_UP_REASONS = new Set([
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
  [REASON_CONTEXT_OWNED_ELSEWHERE]: 'held elsewhere',
  [REASON_SWITCH_FAILED]: 'not applied',
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
 * Whether a change made here would actually be written.
 *
 * Two ways it would not, and both are refusals the write routes already make,
 * so a surface that offered the gesture anyway would be sending the operator
 * to find out by clicking:
 *
 * - there is no control context to write to (`store_available: false`, or an
 *   `owner` of `null` — nothing owns the context anywhere). Both answer 503.
 * - this terminal FOLLOWS another one (`owner.self === false`). That answers
 *   409 naming the owner's pid and port.
 *
 * `owner` absent rather than null is read as "this payload does not say", and
 * only the store decides — an older route must not turn every row read-only.
 * @param {any} state
 * @returns {boolean}
 */
export function contextWritable(state) {
  if (!state?.store_available) return false;
  if (state.owner === null) return false;
  return state.owner?.self !== false;
}

/**
 * The refusal code a row carries when the CONTEXT is why it has no Switch,
 * `''` when the context is not the reason. The row's own `reason` still wins;
 * this is the fallback that says which of the two context refusals it would
 * have been.
 * @param {any} state
 * @returns {string}
 */
export function contextRefusalCode(state) {
  if (contextWritable(state)) return '';
  return state?.owner?.self === false
    ? REASON_CONTEXT_OWNED_ELSEWHERE
    : REASON_STORE_UNAVAILABLE;
}

/**
 * Who holds the control context, in the two words a row has space for, and the
 * longer phrase the banner uses.
 *
 * Keyed on `owner.kind`, because a context owned by an agent's controls server
 * is not a terminal an operator can open — a banner naming a terminal there
 * sends them looking for a page that does not exist. Mirrors
 * `_OWNER_KIND_WORDS` in control_context_owner.py, which the write routes'
 * own 409 sentence is built from, so the two refusals for one situation agree.
 *
 * An unrecognised kind is named as a process rather than guessed at: the pid
 * is true whatever it is.
 * @param {any} owner  the payload's `owner` block
 * @returns {{short: string, long: string}}
 */
export function contextHolderWords(owner) {
  const pid = owner?.pid;
  const port = owner?.port;
  if (owner?.kind === OWNER_CONTROLS_SERVER) {
    return {
      short: 'the controls server',
      long: pid
        ? `The controls server (pid ${pid}) holds the control context.`
        : 'The controls server holds the control context.',
    };
  }
  if (owner?.kind === OWNER_WEB_TERMINAL) {
    return {
      short: 'another terminal',
      // The port is the whole point of the line: it is where the operator goes
      // to make the change this terminal will refuse.
      long: port
        ? `Another terminal on port ${port} holds the control context.`
        : 'Another terminal holds the control context.',
    };
  }
  return {
    short: 'another process',
    long: pid
      ? `Another process (pid ${pid}) holds the control context.`
      : 'Another process holds the control context.',
  };
}

/**
 * The short phrase standing where a row's Switch would be when the CONTEXT is
 * why it has no Switch, `''` when the context is not the reason.
 * @param {any} state
 * @returns {string}
 */
export function contextRefusalPhrase(state) {
  const code = contextRefusalCode(state);
  if (!code) return '';
  if (code === REASON_STORE_UNAVAILABLE) return REASON_PHRASES[REASON_STORE_UNAVAILABLE];
  return contextHolderWords(state?.owner).short;
}

/**
 * Whether the run, and nothing an operator can do here, is holding this row's
 * writes off.
 *
 * Read from the route's own `readonly_run`, never inferred from the columns:
 * "ceiling up, not narrowed, still off" is also what a posture store that
 * failed to resolve leaves behind, and that must not be reported as the
 * deployment running read-only. A row the deployment never armed is not held
 * by the run either — the ceiling holds it, and {@link lockReason} says so.
 * @param {any} row
 * @param {any} state
 */
function writesHeldByTheRun(row, state) {
  return Boolean(state?.readonly_run) && Boolean(row.ceiling_writes);
}

/**
 * Why this row's writes cannot be turned on or off, or `null` when they can.
 *
 * Ordered from the widest cause to the narrowest, so an operator reads the one
 * they could act on: a context nothing here can write outranks a run that
 * would ignore the write, which outranks a deployment that never arms this
 * target, which outranks a gateway table with nowhere to narrow TO. Spoken in
 * the operator's words; the machine vocabulary stays in the route payload.
 * @param {any} row
 * @param {any} state
 * @returns {string|null}
 */
export function lockReason(row, state) {
  if (!contextWritable(state)) {
    return state?.owner?.self === false
      ? `held by ${contextHolderWords(state.owner).short}`
      : 'changes cannot be recorded right now';
  }
  if (writesHeldByTheRun(row, state)) return 'the whole deployment is running read-only';
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
 * @returns {{text: string, tone: string}|null}
 */
export function bannerNote(state) {
  if (state?.owner?.self === false) {
    return { text: contextHolderWords(state.owner).long, tone: 'warn' };
  }
  if (!contextWritable(state)) {
    return { text: 'Changes cannot be recorded right now — the posture store is unavailable.', tone: 'error' };
  }
  // The run alone decides this line. Keying it on every row would hide the
  // banner from a readonly run with one unarmed row — that row is held by the
  // ceiling, not the run, but the run is still the widest fact on the page.
  if (state.readonly_run) {
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
function hardwareNote(kind) {
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
 * gesture after which a write the agent makes can land. The body names no
 * endpoint: the one the roster carries is where reads go under the recorded
 * posture, not where the write this dialog allows would land.

 * The scope line is the first thing it says, because one control context per
 * deployment means arming writes here arms them for every session, notebook
 * kernel and hook — not for the page the click was made on.
 * @param {any} row
 * @param {string} kind
 * @returns {{title: string, body: ConfirmRun[][], live: string|null, confirmLabel: string}}
 */
export function turnOnConfirm(row, kind) {
  return {
    title: `Turn writes on for ${displayName(row, kind)}?`,
    body: [
      ['For ', { em: 'every session on this deployment' }, '.'],
      ['Takes effect at the next write — nothing restarts.'],
    ],
    live: hardwareNote(kind),
    confirmLabel: 'Turn writes on',
  };
}

/**
 * Everything the switch confirm says. The first line states the consequence —
 * where control goes next, naming writes only when the deployment arrives able
 * to make them. The second names the write state it will ARRIVE in, because
 * writes on/off is per machine and does not travel; the word is stateWord's
 * own, so the dialog and the chip a moment later can never disagree. Neither
 * line claims a scope: the switch moves the whole deployment, and saying so
 * twice per dialog is filler.
 * @param {any} row
 * @param {string} kind
 * @param {'writes'|'sandbox'|'read-only'} word  stateWord's answer for the row
 * @returns {{title: string, body: ConfirmRun[][], live: string|null, confirmLabel: string}}
 */
export function switchConfirm(row, kind, word) {
  const arrival =
    word === 'writes'
      ? ['Writes are ', { em: 'on' }, ' there.']
      : word === 'sandbox'
        ? ['Writes are ', { em: 'off' }, ' there — nothing moves until you turn them on.']
        : ['Writes are ', { em: 'locked' }, ' read-only there by the deployment.'];
  return {
    title: `Switch to ${displayName(row, kind)}?`,
    body: [
      [
        word === 'writes' ? 'All control reads and writes go to ' : 'All control reads go to ',
        { em: String(row.endpoint || row.target) },
        '.',
      ],
      arrival,
    ],
    live: word === 'writes' ? hardwareNote(kind) : null,
    confirmLabel: 'Switch',
  };
}

/**
 * Whether one confirm may offer "don't ask again". Per gesture and per
 * machine kind, because the two dialogs guard different things: a switch is
 * recoverable (switch back, and the write state does not travel), so its
 * ceremony may be waived anywhere; turning writes on is the gesture after
 * which a write can land, and on a live machine that write moves hardware —
 * that one dialog keeps asking, always.
 * @param {'switch'|'writes-on'} gesture
 * @param {string} kind  kindAttr's answer for the row
 * @returns {boolean}
 */
export function confirmSkippable(gesture, kind) {
  return gesture === 'switch' || kind !== 'live';
}

/* ---- has the switch this browser asked for landed? ---------------------- */

/**
 * *value* as a non-negative integer, or `null`. Booleans excluded, because
 * `true >= 1` and a payload that ever sends one must not read as generation 1.
 * @param {unknown} value
 * @returns {number|null}
 */
function nonNegativeInt(value) {
  if (typeof value !== 'number' || !Number.isInteger(value) || value < 0) return null;
  return value;
}

/**
 * The switch request this browser is waiting on.
 * @typedef {object} PendingSwitch
 * @property {string} requestId  the id the POST answered with
 * @property {number|null} [generation]  the generation that POST said the
 *   record would carry once the fleet has followed it, `null` when the answer
 *   named none
 */

/**
 * What the roster just read says about that request.
 * @typedef {object} PendingOutcome
 * @property {'waiting'|'answered'|'applied'|'failed'} state
 * @property {number|null} pid  the server that failed, for `failed` only
 * @property {string|null} detail  that server's own sentence, where it sent one
 */

/**
 * Whether the switch this browser asked for has landed, and what happened.
 *
 * A switch is not one event but two: the RECORD moves (the terminus names the
 * request and bumps the generation), and then every live controls server
 * rebuilds its connector host and reports the generation it arrived at. The
 * operator is waiting on the second — a chip that stopped at the terminus
 * would say the deployment is on the new machine while a server is still on the
 * old one.
 *
 * So the rule is a comparison, not a memory: **pending is done when every live
 * server HOLDING A CONNECTOR reports the generation it was asked for.** That
 * survives a page reload, a second tab and a missed frame, because nothing in
 * it depends on having seen the terminus go by.
 *
 * Holding a connector is the row's `children` list, and only an explicit empty
 * list exempts a row: a server serving nothing cannot still be touching the
 * old target, and whatever child it launches comes up on the record's values —
 * its report is not owed. This is the deployment's own convergence stance
 * (`converged()` ignores a null binding as "behind, not wrong") said once more
 * at the chip. A row that does not say — no `children` field at all — is
 * waited on as every row always was.
 *
 * Three answers the comparison cannot give on its own:
 *
 * - a **refusal**, or any other verdict that is not `applied`, moves neither
 *   target nor generation, so no server will ever report it. It is matched by
 *   `request_id` on the terminus — a `generation` of `null` is exactly what
 *   makes a terminus a refusal.
 * - a **failure** is one server saying it was asked for that generation and
 *   could not get there. Without it the comparison would wait for a report
 *   that is not coming, so the pid is named and the wait ends.
 * - **nobody to wait for.** There is one controls server per agent session, so
 *   `servers: []` is a normal state — a deployment before its first agent
 *   session, between sessions, or one whose only server died mid-switch. There
 *   the record's own terminus is the whole answer, because a server that
 *   starts later adopts the record's generation on the way up. Waiting there
 *   would manufacture 30 s of `switching…` and then a false expiry on every
 *   switch made before the agent is running. A fleet whose every row is
 *   childless is the same state with the reports still on disk, and gets the
 *   same answer.
 *
 * `[].every()` is true, so the empty case is decided BEFORE the comparison and
 * on the terminus, never by the comparison — with neither a live server nor a
 * matching terminus nobody has answered at all, and that is still `waiting`.
 * Nothing here invents a deadline; the client's own TTL owns a request the
 * deployment never answers.
 *
 * @param {any} view  the payload of `GET /api/terminal/posture`
 * @param {PendingSwitch|null} pending
 * @returns {PendingOutcome}
 */
export function resolvePendingSwitch(view, pending) {
  /** @type {PendingOutcome} */
  const waiting = { state: 'waiting', pid: null, detail: null };
  /** @type {PendingOutcome} */
  const answered = { state: 'answered', pid: null, detail: null };
  if (!view || !pending || !pending.requestId) return waiting;

  const terminus = view.last_switch;
  const ours = Boolean(terminus) && terminus.request_id === pending.requestId;
  const terminusGeneration = ours ? nonNegativeInt(terminus.generation) : null;
  if (ours && (terminusGeneration === null || terminus.status !== SWITCH_APPLIED)) {
    return answered;
  }

  const generation = nonNegativeInt(pending.generation);
  if (generation === null) return waiting;

  const rows = Array.isArray(view.servers) ? view.servers : [];

  // Every row is scanned for a failure — a childless server that launched
  // toward the record and could not come up files `failed` too, and that
  // verdict is this request's news even though its row owes no arrival.
  for (const row of rows) {
    const block = row && typeof row.last_switch === 'object' ? row.last_switch : null;
    if (!block || block.status !== REPORT_FAILED) continue;
    // A block naming another generation is about a swap the fleet has already
    // moved past; it says nothing about the one being waited on.
    if (nonNegativeInt(block.generation) !== generation) continue;
    const detail = typeof block.detail === 'string' && block.detail.trim() ? block.detail : null;
    return { state: 'failed', pid: nonNegativeInt(row.pid), detail };
  }

  // Only a row that explicitly says `children: []` is exempt from arrival.
  const holding = rows.filter(
    (/** @type {any} */ row) => !(Array.isArray(row?.children) && row.children.length === 0)
  );
  if (!holding.length) {
    // No connector to converge. The record accepting this request at or past
    // the generation asked for is the whole of what "landed" can mean here.
    return terminusGeneration !== null && terminusGeneration >= generation
      ? { state: 'applied', pid: null, detail: null }
      : waiting;
  }

  const arrived = holding.every((/** @type {any} */ row) => {
    const applied = nonNegativeInt(row?.applied_generation);
    return applied !== null && applied >= generation;
  });
  return arrived ? { state: 'applied', pid: null, detail: null } : waiting;
}

/**
 * The sentence under a switch the record accepted and a controls server then
 * failed to apply. Names the pid, because "the switch did not land" without
 * one leaves nobody to look at; carries the server's own words where it sent
 * any, since it is the only process that knows why.
 * @param {number|null} pid
 * @param {string|null} [detail]
 * @returns {string}
 */
export function switchFailureNote(pid, detail = null) {
  const who = pid === null ? 'A controls server' : `Controls server pid ${pid}`;
  const said = String(detail || '').trim();
  return said ? `${who}: ${said}` : `${who} did not apply the switch.`;
}

/**
 * The localStorage key BASE under which a waived confirm is remembered — one
 * slot per gesture per machine, so the first switch to each machine still
 * explains itself once. Callers resolve it through `scopedStorageKey()`
 * (storage-scope.js): on a multi-user mount one operator's waiver must not
 * silence the dialog for everyone else.
 * @param {'switch'|'writes-on'} gesture
 * @param {any} row
 * @returns {string}
 */
export function confirmSkipKeyBase(gesture, row) {
  return `osprey-ctc-skip-confirm:${gesture}:${row.target}`;
}
