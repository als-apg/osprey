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
 * route as `display_name`); the defaults live in {@link KIND_WORDS}, keyed on
 * the `machine_kind` GET publishes (`real` | `rehearsal` | `simulator`). The
 * server's own label stays on tooltips, where the implementation vocabulary
 * belongs.
 *
 * **One process claim.** The switch's position decides whether a write prompts
 * first, and the row publishes it, so that is the one thing about process this
 * module says. What limits apply and who approves are deployment configuration
 * it cannot see. Beyond that it states consequence: writes move hardware, or
 * nothing moves.
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
 * The three switch positions, narrowest first: a move to a later one widens.
 * @type {readonly ('off'|'asked'|'allowed')[]}
 */
export const POSITIONS = ['off', 'asked', 'allowed'];

/**
 * Whether *value* is one of the {@link POSITIONS}: exactly, no folding.
 * @param {unknown} value
 * @returns {value is 'off'|'asked'|'allowed'}
 */
export function isPosition(value) {
  return POSITIONS.includes(/** @type {any} */ (value));
}

/**
 * The machine kinds, as GET publishes them in each row's `machine_kind`
 * (C-KIND) and as the stylesheet keys `data-target-kind` on. One vocabulary
 * for the wire, the tables below and the CSS, so no surface translates.
 */
export const KIND_REAL = 'real';
export const KIND_REHEARSAL = 'rehearsal';
export const KIND_SIMULATOR = 'simulator';

/**
 * A row's machine kind: its published `machine_kind` when it is one of the
 * three, the real machine otherwise — never derived from the target name or
 * the label, and failing toward the loud answer. The one normalisation;
 * {@link module:control-target-chip.kindAttr} is this, said for the CSS.
 * @param {any} row
 * @returns {string}
 */
export function machineKind(row) {
  const published = String(row?.machine_kind || '')
    .trim()
    .toLowerCase();
  return published === KIND_REHEARSAL || published === KIND_SIMULATOR ? published : KIND_REAL;
}

/**
 * The operator word for each machine kind, keyed on the `data-target-kind`
 * value {@link module:control-target-chip.kindAttr} derives. Used wherever a
 * row has no configured `display_name`. Consequence-first on purpose: an
 * operator who has never met a soft-IOC can still tell the machine that moves
 * from the ones that cannot.
 * @type {Record<string, string>}
 */
export const KIND_WORDS = {
  [KIND_REAL]: 'Live',
  [KIND_REHEARSAL]: 'Rehearsal',
  [KIND_SIMULATOR]: 'Simulator',
};

/**
 * The banner each machine kind is named by beside the machine — what a write
 * there does, in the owner's words (C-WORDS, D4). A restated copy of
 * `osprey_connectors.write_target.KIND_BANNERS`; a test pins the two. A
 * confirm carries the real machine's (red) and the rehearsal's (amber); the
 * simulator's confirm carries none ({@link hardwareNote}): nothing about it
 * needs saying before a click.
 * @type {Record<string, string>}
 */
export const KIND_BANNERS = {
  [KIND_REAL]: 'Live — writes move hardware',
  [KIND_REHEARSAL]: 'Rehearsal — nothing moves',
  [KIND_SIMULATOR]: 'Simulator — nothing moves',
};

/**
 * The sentence every widening carries (C-WORDS): in its confirm, or — when the
 * confirm was waived — once in the popover's status line after the move lands.
 * Never beside the off control: off reaches running sandboxes.
 */
export const TURN_ON_SENTENCE =
  'Runs already going — a sandbox run or a notebook cell — keep the state they started with; the next one follows.';

/**
 * What an operator reads when a turn-off could not be recorded (C-POST 503 on
 * off): writes are still on, which is the one thing they must not miss.
 */
export const STOP_NOT_TAKEN_SENTENCE =
  'The stop did not take effect — writes are still on. Retry; if it fails again, ask the deployment admin.';

/**
 * What stands in for a position above the login's tier: as the row's reason
 * ({@link lockReason}) and as the phrase of the route's `tier_caps` refusal.
 * `asked` is the one ceiling that caps a position: `allowed` is the top one, and
 * `never` is a lock on every armed position with a sentence of its own.
 */
const TIER_CAPS_SENTENCE = 'your tier caps writes at asked';

/**
 * The "read" capability phrase first contact and the tour put in front of an
 * operator, keyed on the same `data-target-kind` value as {@link KIND_WORDS}.
 * Keyed rather than fixed because the sentence is a promise about where the
 * numbers come from: the simulator saying "read values from the real machine" would be
 * the one claim this vocabulary exists to prevent.
 * @type {Record<string, string>}
 */
export const KIND_READ_PHRASES = {
  [KIND_REAL]: 'read values from the live machine',
  [KIND_REHEARSAL]: 'read values from the rehearsal copy',
  [KIND_SIMULATOR]: 'read values from the simulator',
};

/**
 * The one-line descriptor under each name: what writing to this machine does.
 * These are statements of consequence, not of process — nothing about
 * approvals or limits, which are configuration this file cannot see.
 * @type {Record<string, string>}
 */
const KIND_DESCRIPTORS = {
  [KIND_REAL]: 'Writes move hardware',
  [KIND_REHEARSAL]: "Copy of the live machine's controls · nothing moves",
  [KIND_SIMULATOR]: 'Physics model · nothing moves',
};

/**
 * The `reason` codes that all mean "the deployment has not authored this
 * machine yet". On a stock render this is the real machine's DELIBERATE state —
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
 * tooltip; it is the kind's banner, longer than a name, so the label is
 * last.
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
 * The consequence line under the name. A real machine nothing has authored
 * yet reads "Not set up yet" instead of promising hardware that is not there.
 * @param {any} row
 * @param {string} kind
 * @returns {string}
 */
export function descriptor(row, kind) {
  if (kind === KIND_REAL && NOT_SET_UP_REASONS.has(String(row?.reason || ''))) {
    return 'Not set up yet';
  }
  return KIND_DESCRIPTORS[kind] || '';
}

/**
 * The tone the descriptor renders in: `hazard` for the one line that names
 * hardware moving, `null` for every line that promises nothing moves. A real
 * machine nothing has authored yet is not a hazard — its descriptor says "not
 * set up", and painting that red would flag a healthy render as broken.
 * @param {any} row
 * @param {string} kind
 * @returns {'hazard'|null}
 */
export function descriptorTone(row, kind) {
  return kind === KIND_REAL && descriptor(row, kind) === KIND_DESCRIPTORS[KIND_REAL]
    ? 'hazard'
    : null;
}

/**
 * The display phrase for a state word ({@link module:control-target-chip.stateWord}):
 * asked writes prompt first, allowed writes do not, off is the state you put it
 * in and one click undoes, and locked is an armed switch still refused (or a
 * switch that cannot be read), whose row says why. Anything unrecognised reads
 * as locked — the direction that promises nothing.
 * @param {'off'|'asked'|'allowed'|'locked'|string} word
 * @returns {string}
 */
export function statePhrase(word) {
  if (isPosition(word)) return `writes ${word}`;
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
 *
 * `connector_not_switchable` is deliberately NOT one of them: that block is
 * authored and the deployment is running on it — the switch just has no way to
 * dial its protocol — so the row says what is unsupported instead of implying
 * the deployer left something out.
 * @type {Record<string, string>}
 */
export const REASON_PHRASES = {
  connector_block_missing: 'not set up',
  gateways_missing: 'not set up',
  probe_channel_missing: 'not set up',
  connector_not_switchable: 'switching not supported',
  target_unresolvable: 'unavailable',
  operator_ack_missing: 'needs gateway ack',
  archive_belongs_to_standin: 'archive conflict',
  invented_history: 'no archive',
  standin_not_deployed: 'stand-in not deployed',
  reaches_live_machine: 'points at the live machine',
  selected_role_missing: 'no endpoint for role',
  [REASON_STORE_UNAVAILABLE]: 'store unavailable',
  [REASON_CONTEXT_OWNED_ELSEWHERE]: 'held elsewhere',
  [REASON_SWITCH_FAILED]: 'not applied',
  // Minted by the chip for a request nothing answered before its deadline.
  request_expired: 'no answer',
  tier_caps: TIER_CAPS_SENTENCE,
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
 * Whether a TARGET SWITCH made here would actually be written. Posture
 * gestures do not ask this: a follower turns its login's switch on and off
 * exactly as the holder does (C-POST), so {@link lockReason} never reads the
 * owner.
 *
 * Two ways a switch would not be written, and both are refusals the target
 * route already makes, so a surface that offered the gesture anyway would be
 * sending the operator to find out by clicking:
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
  return state?.owner?.self === false ? REASON_CONTEXT_OWNED_ELSEWHERE : REASON_STORE_UNAVAILABLE;
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
 * The verdict reasons a turn-on lifts: the switch itself being off, or a stop
 * that came after a turn-on began. Every other `write_reason` is a lock no
 * chip gesture moves, and its `write_phrase` already says who lifts it.
 */
export const LIFTED_BY_TURN_ON = new Set(['switch_off', 'stopped_since']);

/**
 * Why this row's writes cannot be moved to *position*, or `null` when they can
 * — read from C-GET alone (`switch`, `write_reason`, `write_phrase`, `locks`).
 *
 * Turning off is never locked by what this answers: off is refused for no
 * tier, run or follower (C-POST), and a locked row keeps its off control.
 * Nor does the owner enter it — a follower turns its login's switch exactly as
 * the holder does; only target switching stays the holder's
 * ({@link contextWritable}).
 *
 * Ordered widest first: no record to write, then a record nobody can read,
 * then the verdict's own phrase for a lock no click here lifts, then the two
 * published locks (a readonly run, a login whose tier never writes) for a row
 * whose verdict named the switch first, and last the tier's ceiling, which
 * locks `allowed` alone: a capped login still arms at `asked`.
 * @param {any} row
 * @param {any} state
 * @param {'asked'|'allowed'} [position]  the armed position asked about
 * @returns {string|null}
 */
export function lockReason(row, state, position = 'asked') {
  if (state?.store_available === false) return 'changes cannot be recorded right now';
  const reason = typeof row?.write_reason === 'string' ? row.write_reason : '';
  const phrase = typeof row?.write_phrase === 'string' ? row.write_phrase : '';
  if (!isPosition(row?.switch)) {
    return phrase || 'the write switch cannot be read';
  }
  if (reason && !LIFTED_BY_TURN_ON.has(reason)) return phrase || reason;
  const locks = state?.locks && typeof state.locks === 'object' ? state.locks : {};
  if (locks.readonly_run || state?.readonly_run) return 'the whole deployment is running read-only';
  if (locks.tier === 'never') return "this login's write tier does not allow writes";
  if (locks.tier === 'asked' && position === 'allowed') return TIER_CAPS_SENTENCE;
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
    return {
      text: 'Changes cannot be recorded right now — the posture store is unavailable.',
      tone: 'error',
    };
  }
  // The run alone decides this line, never the rows: it is the widest fact on
  // the page whatever each row's verdict names first.
  if (state.readonly_run || state.locks?.readonly_run) {
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
 * The banner a confirm carries ({@link KIND_BANNERS}): the real machine's
 * "writes move hardware", the rehearsal's "nothing moves", nothing for the
 * simulator. It says what a write does to the machine; whether the write
 * prompts first is the confirm's body, from the position. What limits apply is
 * deployment configuration this module cannot see, so it makes no claim about
 * them. The dialog's `data-target-kind` decides its tone (red, amber).
 * @param {string} kind
 * @returns {string|null}
 */
function hardwareNote(kind) {
  return kind === KIND_SIMULATOR ? null : KIND_BANNERS[kind] || null;
}

/**
 * The tooltip a machine's name carries: the row's label (its kind's banner,
 * such as "Rehearsal — nothing moves"), the endpoint, and the measured
 * reachability. The at-rest surface names consequences; hover is
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
 * Everything a widening's confirm says, for a move to *to*. Only a widening
 * asks — a narrowing removes reach and is undone by a click; a widening is the
 * gesture after which a write the agent makes can land, or land unprompted.
 * The body names no endpoint: the one the roster carries is where reads go, not
 * where the write this dialog allows would land.
 *
 * The scope line is the first thing it says, because a switch is one login's:
 * moving it here moves it for every session, notebook and agent run of the
 * signed-in login — and for nobody else's. Then whether writes to this machine
 * will still prompt, the one thing the two positions differ on, then the
 * turn-on sentence (C-WORDS): runs already going keep the state they started
 * with. The move's start adds nothing: off→allowed and asked→allowed arrive at
 * the same place.
 * @param {any} row
 * @param {string} kind
 * @param {'asked'|'allowed'} [to]  the position the move arrives at
 * @returns {{title: string, body: ConfirmRun[][], live: string|null,
 *            confirmLabel: string, targetKind: string}}
 */
export function turnOnConfirm(row, kind, to = 'asked') {
  const name = displayName(row, kind);
  const allowed = to === 'allowed';
  const position = allowed ? 'allowed' : 'asked';
  return {
    title: `Set writes to ${position} on ${name}?`,
    body: [
      ['For ', { em: 'your login' }, ' — every session, notebook and agent run you start.'],
      allowed
        ? ['Writes to ', { em: name }, ' will run with ', { em: 'no prompt' }, '.']
        : ['Writes to ', { em: name }, ' will ', { em: 'still prompt' }, ' before each one runs.'],
      [TURN_ON_SENTENCE],
    ],
    live: hardwareNote(kind),
    confirmLabel: `Set writes to ${position}`,
    targetKind: kind,
  };
}

/**
 * The clause that follows `Writes are <word>` in the switch confirm's arrival
 * line, per word stateWord can answer; a word not listed ends the sentence.
 * @type {Record<string, string>}
 */
const ARRIVAL_ENDINGS = {
  asked: ' there — each one prompts first.',
  allowed: ' there — they run with no prompt.',
  off: ' there — nothing moves until you turn them on.',
};

/**
 * Everything the switch confirm says. The first line states the consequence —
 * where control goes next, naming writes only when the deployment arrives able
 * to make them. The second names the write state it will ARRIVE in, because a
 * position is per machine and does not travel; the word is stateWord's own, so
 * the dialog and the chip a moment later can never disagree. Neither line
 * claims a scope: the switch moves the whole deployment, and saying so twice
 * per dialog is filler.
 * @param {any} row
 * @param {string} kind
 * @param {'off'|'asked'|'allowed'|'locked'} word  stateWord's answer for the row
 * @returns {{title: string, body: ConfirmRun[][], live: string|null,
 *            confirmLabel: string, targetKind: string}}
 */
export function switchConfirm(row, kind, word) {
  const armed = word === 'asked' || word === 'allowed';
  const known = armed || word === 'off';
  return {
    title: `Switch to ${displayName(row, kind)}?`,
    body: [
      [
        armed ? 'All control reads and writes go to ' : 'All control reads go to ',
        { em: String(row.endpoint || row.target) },
        '.',
      ],
      ['Writes are ', { em: known ? word : 'locked' }, known ? ARRIVAL_ENDINGS[word] : ' there.'],
    ],
    live: armed ? hardwareNote(kind) : null,
    confirmLabel: 'Switch',
    targetKind: kind,
  };
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
 * The localStorage key BASES under which a waived confirm is remembered: one
 * slot for every move of the writes switch, to asked or allowed on any machine,
 * and one for every switch of target. Callers resolve them through
 * `scopedStorageKey()` (storage-scope.js): on a multi-user mount one operator's
 * waiver must not silence the dialog for everyone else.
 */
export const SKIP_KEY_MOVE = 'osprey-ctc-skip-confirm:move';
export const SKIP_KEY_SWITCH = 'osprey-ctc-skip-confirm:switch';
