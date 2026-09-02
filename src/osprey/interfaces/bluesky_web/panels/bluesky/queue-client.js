// @ts-check
/**
 * BLUESKY panel — queue logic.
 *
 * Everything in this module is pure (or, for `createQueueStream`, injectable):
 * the reducers that turn the bridge's SSE frames into panel state, the
 * predicates that decide which queue control is live, the refusal classifier,
 * and the progress formatter. `queue-view.js` owns every DOM node; this module owns
 * every decision, so the wire contract is unit-testable with plain objects.
 *
 * Wire shapes this module speaks (bridge `queue.py`, relayed verbatim by the
 * sidecar's `queue_relay.py`):
 *
 * - `GET /queue/events` frames: `{type: "hello"|"queue", status, items,
 *   running_item}` — FULL snapshots, never diffs, so a frame replaces state
 *   rather than patching it.
 * - `status` is the manager's bounded summary: `{available, manager_state,
 *   items_in_queue, items_in_history, running_item_uid, plan_queue_uid,
 *   plan_history_uid, queue_stop_pending, queue_autostart_enabled, ...}`, or
 *   `{available: false, reason}` when the manager could not be read at all.
 * - `items` are the manager's own item documents (`item_uid`, `name`,
 *   `kwargs`, `meta.osprey_run_id`); the running item may carry `progress`.
 *
 * Refusals are the one shape worth stating twice: the queue relay does NOT
 * unwrap the bridge's envelope, so the discriminator is `detail.code` and the
 * operator-facing sentence is `detail.detail`. A panel reading a TOP-LEVEL
 * `code` finds nothing and silently mis-classifies every refusal — see
 * `classifyQueueRefusal`.
 */

/**
 * Manager states in which the queue is draining (or about to). Mirrors
 * `queue_backend.QUEUE_ACTIVE_MANAGER_STATES` — the same predicate the bridge
 * arms on, kept in step by name.
 * @type {readonly string[]}
 */
export const QUEUE_ACTIVE_MANAGER_STATES = Object.freeze([
  'starting_queue',
  'executing_queue',
  'executing_task',
  'paused',
]);

/** Run statuses a run record can no longer leave (`runs.py`). */
export const TERMINAL_RUN_STATUSES = Object.freeze(['completed', 'stopped', 'error']);

// The three controls of the queue control strip read as one row of verbs:
// Start · Stop · Abort. What each one does to the running plan is said in its
// tooltip and, for the two that confirm, in the note that renders on arm.
export const START_LABEL = 'Start';
export const STOP_LABEL = 'Stop';
// Not "Cancel pending stop": in a queue UI "Cancel" scans as MORE halting, so
// that label pointed an operator's glance in the opposite direction from what
// the button does. The resting label has to be right on its own — the two-step
// confirm is a backstop, not the explanation.
export const WITHDRAW_STOP_LABEL = 'Withdraw stop — queue keeps draining';
export const CONFIRM_WITHDRAW_STOP_LABEL = 'Confirm — the queue keeps draining';

// The emergency halt. "Abort" rather than "Stop now" because the queue already
// has a Stop and the two must not read as degrees of the same thing: this one
// throws away the rest of the running plan.
export const ABORT_LABEL = 'Abort';
// The armed second step of BOTH confirming buttons (Abort, Clear) is the one
// word "Confirm": the button turns to the caution colour and the note under
// the strip says what the second click does. panel.css floors every strip
// button at this label's width, the longest any of them shows, so no swap
// resizes a box: Abort's width positions Stop beside it, and a label that
// grew on arming used to drag Stop sideways at the exact moment an operator
// is choosing between the two halts — the armed Abort then covered the
// pixels Stop had just vacated, so a click aimed at Stop committed the abort.
export const CONFIRM_LABEL = 'Confirm';
export const CONFIRM_ABORT_LABEL = CONFIRM_LABEL;
// Removes every WAITING plan; the running one is untouched (that is Abort).
export const CLEAR_LABEL = 'Clear';
export const CONFIRM_CLEAR_LABEL = CONFIRM_LABEL;

/**
 * Plain-language tooltips for the four controls — one sentence each, what the
 * click does, nothing else.
 */
export const CONTROL_HINTS = Object.freeze({
  start: 'Runs the queued plans, and any plan added later.',
  stop: 'Lets the current plan finish, then stops the queue.',
  abort: 'Stops the running plan right now.',
  clear: 'Removes every waiting plan. The running plan is not touched.',
  clearHistory: 'Removes every completed run from the list. Run data is kept.',
  removeRun: 'Removes this run from the list. Its data is kept.',
});

/** @typedef {Record<string, unknown>} QueueStatus */
/** @typedef {Record<string, any>} QueueItem */

/** @typedef {{
 *   status: QueueStatus|null,
 *   items: QueueItem[],
 *   runningItem: QueueItem|null,
 *   connected: boolean,
 *   frames: number,
 * }} QueueState */

/** @returns {QueueState} */
export function createInitialQueueState() {
  return { status: null, items: [], runningItem: null, connected: false, frames: 0 };
}

/**
 * Fold one SSE frame into panel state.
 *
 * Both frame types carry a whole snapshot, so the reduction is a replacement.
 * Anything that isn't a recognized frame is ignored and the previous state is
 * returned by identity — a malformed frame must never blank an operator's
 * queue view.
 *
 * @param {QueueState} state
 * @param {any} frame
 * @returns {QueueState}
 */
export function reduceQueueFrame(state, frame) {
  if (!frame || typeof frame !== 'object') return state;
  if (frame.type !== 'hello' && frame.type !== 'queue') return state;
  const status = frame.status && typeof frame.status === 'object' ? frame.status : null;
  const running =
    frame.running_item && typeof frame.running_item === 'object' ? frame.running_item : null;
  return {
    status,
    items: Array.isArray(frame.items) ? frame.items.filter(isItem) : [],
    runningItem: running,
    connected: true,
    frames: state.frames + 1,
  };
}

/**
 * @param {any} value
 * @returns {value is QueueItem}
 */
function isItem(value) {
  return Boolean(value) && typeof value === 'object';
}

/**
 * Whether the manager's *history* moved between two status summaries — the
 * cue to re-fetch `GET /runs` rather than polling it on a timer. Both keys are
 * checked because a rotated history can leave the count unchanged while the
 * uid moves, and a `null` (unavailable) summary must not read as "changed"
 * forever.
 *
 * @param {QueueStatus|null} prev
 * @param {QueueStatus|null} next
 * @returns {boolean}
 */
export function historyChanged(prev, next) {
  if (next === null) return false;
  if (prev === null) return true;
  return (
    prev.plan_history_uid !== next.plan_history_uid ||
    prev.items_in_history !== next.items_in_history ||
    prev.running_item_uid !== next.running_item_uid ||
    // The bridge's own key: a run removed from the OSPREY view moves none of
    // the manager's uids, so this is the only sign another panel did it.
    prev.runs_removed !== next.runs_removed
  );
}

/**
 * Whether the manager is ARMED: it drains whatever it holds, and whatever
 * arrives, without a further start. This is queueserver's autostart mode,
 * which the bridge switches on for a deployment whose queue is meant to be
 * running by default and which `POST /queue/start` re-enables after a stop.
 *
 * Read from the manager, never assumed from config: the flag the manager
 * reports is the only truth about what an enqueue does next.
 *
 * @param {QueueStatus|null} status
 * @returns {boolean}
 */
export function queueArmed(status) {
  return Boolean(status) && status?.available === true && status?.queue_autostart_enabled === true;
}

/** @typedef {'ok'|'info'|'warn'|'err'|'neutral'} BadgeTone */

/**
 * The manager's state as a badge, in the operator's vocabulary.
 *
 * The queue server's own word for a manager with nothing to do is `idle`. To
 * an operator that reads as "waiting for input", which is the wrong picture
 * for both things it can mean: a queue that was never started (`stopped`),
 * and an armed queue that will run the next plan it is given (`ready`). The
 * badge names which; the raw manager word rides along as `raw` for the
 * tooltip, so the manager's own state is always one hover away. A state
 * string this bundle does not know is shown raw, in the caution tone — an
 * unknown state is doubt, and doubt is not painted calm.
 *
 * `hint` is the badge's tooltip: one plain sentence saying what the state
 * means for the operator, followed by the manager's own word.
 *
 * @param {QueueStatus|null} status
 * @returns {{label: string, tone: BadgeTone, note: string|null, raw: string|null, hint: string}}
 */
export function describeQueueStatus(status) {
  if (!status || status.available !== true) {
    const reason = status && typeof status.reason === 'string' ? status.reason : null;
    return {
      label: 'unavailable',
      tone: 'err',
      note: reason
        ? `The queue manager could not be read (${reason}).`
        : 'The queue manager could not be read.',
      raw: null,
      hint: 'The queue server cannot be reached.',
    };
  }
  const raw = typeof status.manager_state === 'string' ? status.manager_state : null;
  if (raw === null) {
    return {
      label: 'unknown',
      tone: 'warn',
      note: 'The manager reported no state.',
      raw,
      hint: 'The queue server reported no state.',
    };
  }
  /** @param {string} sentence */
  const hint = (sentence) => `${sentence} (manager: ${raw})`;
  if (status.queue_stop_pending === true) {
    return {
      label: 'stopping after current',
      tone: 'warn',
      note: null,
      raw,
      hint: hint('The current plan finishes, then the queue stops.'),
    };
  }
  switch (raw) {
    case 'starting_queue':
      return { label: 'starting', tone: 'info', note: null, raw, hint: hint('A plan is about to run.') };
    case 'executing_queue':
    case 'executing_task':
      return { label: 'running', tone: 'info', note: null, raw, hint: hint('A plan is running now.') };
    case 'paused':
      return { label: 'paused', tone: 'warn', note: null, raw, hint: hint('The running plan is paused.') };
    case 'idle':
      if (!queueArmed(status)) {
        return {
          label: 'stopped',
          tone: 'neutral',
          note: null,
          raw,
          hint: hint('The queue is stopped. Plans you add wait until Start.'),
        };
      }
      // Autostart drains only once the worker environment is open: an armed
      // manager with no environment is not ready, whatever its flag says.
      if (status.worker_environment_exists === false) {
        return {
          label: 'environment closed',
          tone: 'warn',
          note: 'The queue is started, but the worker environment is not open yet.',
          raw,
          hint: hint('The queue is started, but the worker is not ready yet.'),
        };
      }
      return {
        label: 'ready',
        tone: 'ok',
        note: null,
        raw,
        hint: hint('The queue is running. A plan you add runs right away.'),
      };
    default:
      return { label: raw, tone: 'warn', note: null, raw, hint: hint('Unfamiliar state.') };
  }
}

/** @typedef {{disabled: boolean, reason: string|null}} ControlState */
/** @typedef {{
 *   body: {cancel: boolean},
 *   arming: boolean,
 *   note: string|null,
 * }} StopControlState */

/**
 * Which queue controls are live, and why not when they aren't.
 *
 * Deliberately NOT a local copy of the bridge's arming policy: nothing here
 * asks whether this deployment is armed or can execute. These are usability
 * gates only (don't offer "start" with nothing to start), and every one of
 * them is a superset of what the bridge enforces — the bridge's refusal is
 * still the answer, surfaced verbatim.
 *
 * THE STOP CONTROL HAS NO DISABLED STATE, and that is why this shape has no
 * `disabled` key to set. The bridge leaves a plain stop completely ungated by
 * design — halting must never have a failure mode — so every reason a panel
 * might think of for greying it out is wrong: an unreadable manager summary is
 * precisely when an operator most needs to send a halt, and a queue that looks
 * idle here may have started between two frames. Whatever this panel knows is
 * advisory only, and rides along as `note` (a tooltip), never as a gate.
 *
 * The asymmetry worth naming: a plain stop is never an arming action, but
 * *withdrawing* a pending stop is — it un-halts a queue a human halted and
 * lets it keep draining toward hardware. That branch is marked `arming`, which
 * is what drives the panel's two-step confirm and its caution styling.
 *
 * This control does NOT abort the running item; `abortControl` below is the
 * separate emergency halt for that.
 *
 * @param {QueueState} state
 * @returns {{start: ControlState, stop: StopControlState}}
 */
export function queueControls(state) {
  const status = state.status;
  const available = Boolean(status) && status?.available === true;
  const stopPending = Boolean(status) && status?.queue_stop_pending === true;
  const managerState = typeof status?.manager_state === 'string' ? status.manager_state : null;
  const active = managerState !== null && QUEUE_ACTIVE_MANAGER_STATES.includes(managerState);

  // Start ARMS the queue: it drains what is queued now and whatever arrives
  // later, so an empty stopped queue is a legitimate thing to start. It is
  // dead only when there is nothing to arm — already armed, already draining,
  // or a manager that cannot be read.
  /** @type {ControlState} */
  let start;
  if (!available) {
    start = { disabled: true, reason: 'The queue manager could not be read.' };
  } else if (active) {
    start = { disabled: true, reason: 'The queue is already running.' };
  } else if (queueArmed(status)) {
    start = { disabled: true, reason: 'The queue is already started.' };
  } else {
    start = { disabled: false, reason: null };
  }

  /** @type {StopControlState} */
  const stop = stopPending
    ? {
        body: { cancel: true },
        arming: true,
        note: 'The queue will keep draining after the current item.',
      }
    : {
        body: { cancel: false },
        arming: false,
        note: !available
          ? 'The queue manager summary could not be read; the stop is still sent.'
          : active
            ? null
            : 'The queue is not running; a stop is still accepted.',
      };

  return { start, stop };
}

/**
 * The stop button's label, across the plain stop and the two-step withdrawal.
 *
 * `confirmArmed` is the panel's transient "clicked once" state and only means
 * anything on the arming branch — a plain stop is a single click, because
 * putting a confirm in front of a halt is exactly the wrong place for
 * friction.
 *
 * @param {StopControlState} stop
 * @param {boolean} confirmArmed
 * @returns {string}
 */
export function stopButtonLabel(stop, confirmArmed) {
  if (!stop.arming) return STOP_LABEL;
  return confirmArmed ? CONFIRM_WITHDRAW_STOP_LABEL : WITHDRAW_STOP_LABEL;
}

/**
 * The stop button's class.
 *
 * `.confirm` is the caution treatment, and it means ARMED-AND-CONSEQUENTIAL —
 * the same meaning it carries on the Plans view's Add-to-queue and
 * discard-draft buttons, which paint it only once their confirm is armed. A
 * plain stop never gets it: styling the safe halt as the dangerous action and
 * the hardware-arming withdrawal as the routine one is precisely backwards.
 *
 * @param {StopControlState} stop
 * @param {boolean} confirmArmed
 * @returns {string}
 */
export function stopButtonClass(stop, confirmArmed) {
  return stop.arming && confirmArmed ? 'btn confirm' : 'btn';
}

/** @typedef {{note: string|null, running: boolean}} AbortControlState */

/**
 * The abort control's advisory state. Like the stop control, it has NO
 * `disabled` key, and for a stronger version of the same reason: this is the
 * only surface that halts a plan already moving hardware. Every state a panel
 * might grey it out in — an unreadable manager summary, a frame that showed no
 * running item — is a state where the panel's knowledge is one poll stale and
 * the machine may be scanning. What this panel knows rides as a tooltip.
 *
 * `running` is advisory only: it drives the wording, never a gate. When it is
 * false the abort is still sent, and the bridge answers with its own
 * `nothing_running`, which is the honest source for that claim.
 *
 * @param {QueueState} state
 * @returns {AbortControlState}
 */
export function abortControl(state) {
  const status = state.status;
  const available = Boolean(status) && status?.available === true;
  const running = state.runningItem !== null;

  if (!available) {
    return {
      note: 'The queue manager summary could not be read; the abort is still sent.',
      running,
    };
  }
  return {
    note: running
      ? 'Click again to abort: the rest of the running plan is dropped, and hardware stays where it is.'
      : 'No plan is running in the last frame seen; the abort is still sent and the bridge answers.',
    running,
  };
}

/**
 * The Clear control: drop every waiting plan. A usability gate only, like
 * Start — it is dead when there is provably nothing to clear, and it never
 * touches the running item (that is Abort). Two-step in the view, because
 * the plans it removes are somebody's staged work.
 *
 * @param {QueueState} state
 * @returns {ControlState}
 */
export function clearControl(state) {
  const status = state.status;
  const available = Boolean(status) && status?.available === true;
  if (!available) return { disabled: true, reason: 'The queue manager could not be read.' };
  if (state.items.length === 0) return { disabled: true, reason: 'Nothing is waiting.' };
  return { disabled: false, reason: null };
}

/**
 * Whether the two halts must be on screen on EVERY tab right now.
 *
 * This gates VISIBILITY, never function — the halts stay undisabled and
 * always-sendable whenever they are on screen (see `queueControls` /
 * `abortControl` for why they have no disabled state). What this predicate
 * decides is whether a tab other than Queue may show the queue as a badge
 * alone, and it answers no only when the panel affirmatively knows the halts
 * are dormant. The default for every other state — no frame yet, a dropped
 * stream, an unreadable manager, a state string this bundle has never heard
 * of — is PINNED. Reasons to stay visible are enumerated; hiding is what has
 * to be earned.
 *
 * Dormant means nothing is moving: manager idle, no running item, no pending
 * stop. An ARMED idle queue is dormant by this test — the machine is still —
 * and the moment an item lands the manager leaves idle, so the next frame
 * (≤ 1 s) pins the halts on whichever tab the operator is looking at.
 *
 * A pending stop pins them even though nothing is "running" in the dangerous
 * sense: the stop button is in its withdrawal mode, and a control an operator
 * armed must not vanish under them. The pinning states are, deliberately, a
 * superset of the states that keep either confirm-armed flag set in
 * queue-view.js — so a half-armed confirm can never be hidden.
 *
 * @param {QueueState} state
 * @returns {boolean}
 */
export function haltsPinned(state) {
  if (!state.connected) return true;
  const status = state.status;
  if (!status || status.available !== true) return true;
  const managerState = typeof status.manager_state === 'string' ? status.manager_state : null;
  if (managerState === null) return true;
  if (managerState !== 'idle') return true;
  if (status.queue_stop_pending === true) return true;
  if (state.runningItem !== null) return true;
  return false;
}

/**
 * Which of the strip's controls are shown, given the tab on screen.
 *
 * ONE set of controls serves every tab (they render outside the view
 * containers, so a halt is never a tab switch away). The Queue tab is the
 * control panel and shows all of them in every state. The other two tabs show
 * the queue as a badge while it is dormant, and grow the two halts the moment
 * anything is moving or in doubt (`haltsPinned`). Start never leaves the
 * Queue tab: arming a queue is a deliberate act, not an always-at-hand one.
 *
 * `dormant` is the halts' visual weight: true when the panel affirmatively
 * knows nothing is moving, so Stop and Abort may render quiet. They stay
 * live and clickable either way — a greyed halt is still a halt.
 *
 * @param {QueueState} state
 * @param {string} view The active view id (`plans` | `queue` | `results`).
 * @returns {{start: boolean, halts: boolean, clear: boolean, dormant: boolean}}
 */
export function stripControls(state, view) {
  const pinned = haltsPinned(state);
  if (view === 'queue') return { start: true, halts: true, clear: true, dormant: !pinned };
  return { start: false, halts: pinned, clear: false, dormant: !pinned };
}

/**
 * The Plans view's enqueue button, in words that match what the click does.
 *
 * Adding to an ARMED idle queue runs the plan — so the button says `Run`. In
 * every other state the item waits its turn, and the note says for what: the
 * running plan, or a start the operator has to give from the Queue tab. An
 * unreadable manager promises nothing (the capability banner speaks there).
 *
 * @param {QueueStatus|null} status
 * @returns {{label: string, note: string}}
 */
export function enqueuePresentation(status) {
  if (!status || status.available !== true) return { label: 'Add to queue', note: '' };
  const managerState = typeof status.manager_state === 'string' ? status.manager_state : null;
  const active = managerState !== null && QUEUE_ACTIVE_MANAGER_STATES.includes(managerState);
  if (active) return { label: 'Add to queue', note: 'Runs after the current plan.' };
  if (queueArmed(status)) return { label: 'Run', note: 'Runs now.' };
  return { label: 'Add to queue', note: 'The queue is stopped. Start it from the Queue tab.' };
}

/**
 * The abort button's label across its two-step confirm.
 *
 * The plain stop is one click on purpose (friction in front of a halt is
 * friction in the wrong place) and this control deliberately does NOT copy
 * that. The two are not the same kind of halt: "stop after the current item"
 * is cheap and withdrawable, while an abort throws away the rest of a plan and
 * cannot be undone — and the two buttons sit side by side, which makes a slip
 * from one to the other the likely error. One extra click is the whole cost,
 * on a control that is live from first paint and never disabled.
 *
 * @param {boolean} confirmArmed
 * @returns {string}
 */
export function abortButtonLabel(confirmArmed) {
  return confirmArmed ? CONFIRM_ABORT_LABEL : ABORT_LABEL;
}

/**
 * The abort button's class.
 *
 * `.confirm` is the caution treatment and means ARMED-AND-CONSEQUENTIAL — the
 * same sense the Plans view's discard-draft carries, where the consequence is
 * destruction rather than arming. It is applied only on the armed second step:
 * at rest this is a halt, and painting a halt as the dangerous action is the
 * inversion the sibling control already had to fix.
 *
 * @param {boolean} confirmArmed
 * @returns {string}
 */
export function abortButtonClass(confirmArmed) {
  return confirmArmed ? 'btn confirm' : 'btn';
}

/**
 * Banner tone for an abort that SUCCEEDED. Never `ok`.
 *
 * `writeOutcomeTone` reserves green for outcomes that cannot move hardware,
 * which an abort technically satisfies — and green would still be wrong. A
 * successful abort means a plan was destroyed and the machine is sitting at
 * whatever position it stopped at; that is a moment to register, not to
 * celebrate.
 *
 * @returns {'warn'}
 */
export function abortOutcomeTone() {
  return 'warn';
}

/**
 * The sentence shown after a successful abort.
 *
 * Leads with the consequence, then relays the bridge's own `msg` VERBATIM when
 * it sent one — the manager's wording is the only account of what it actually
 * did, and substituting a local sentence for it is how a panel ends up
 * describing an event differently from the agent.
 *
 * `abort_pending` is reported as still unwinding, never as a failure: the
 * bridge only sets it on a body whose `aborted` is true.
 *
 * @param {any} body
 * @returns {string}
 */
export function abortSuccessMessage(body) {
  const parts = [
    'Abort sent — the running plan was stopped where it was. Hardware is left ' +
      'wherever the plan stopped; check positions before anything else runs.',
  ];
  if (body && typeof body === 'object' && body.abort_pending === true) {
    parts.push('The manager is still unwinding the run.');
  }
  const msg = body && typeof body === 'object' ? body.msg : null;
  if (typeof msg === 'string' && msg.trim() !== '') parts.push(`Manager: ${msg}`);
  return parts.join(' ');
}

/** @typedef {{hidden: boolean, message: string}} EmptyState */

/**
 * What the queue list says when it shows no rows.
 *
 * Gated on KNOWLEDGE, not on count. "Nothing queued." is a positive claim
 * about the manager's contents, and this panel is only entitled to make it
 * when it actually read them: the bridge sends `items: []` alongside an
 * `available: false` summary whenever the manager could not be read, so
 * counting rows alone renders an outage as an empty queue. An operator who
 * believes that enqueues a duplicate of work that is already pending.
 *
 * The same reasoning covers startup, before any frame has arrived — the panel
 * does not know yet, and says so.
 *
 * @param {QueueState} state
 * @returns {EmptyState}
 */
export function queueEmptyState(state) {
  if (state.items.length > 0) return { hidden: true, message: '' };
  const available = Boolean(state.status) && state.status?.available === true;
  return available
    ? { hidden: false, message: 'Nothing queued.' }
    : { hidden: false, message: 'Queue contents unknown — the manager could not be read.' };
}

/**
 * What the completed-runs list says when it shows no rows.
 *
 * Same rule as `queueEmptyState`: until a `GET /runs` fetch has actually
 * landed, "No completed runs yet." would be a claim about history this panel
 * has not read. `loaded` tracks whether any fetch ever succeeded, not whether
 * the last one did — a transient failure after a good read leaves the last
 * known list on screen, which is still the best answer available.
 *
 * @param {Array<unknown>} records
 * @param {boolean} loaded
 * @returns {EmptyState}
 */
export function historyEmptyState(records, loaded) {
  if (records.length > 0) return { hidden: true, message: '' };
  return loaded
    ? { hidden: false, message: 'No completed runs yet.' }
    : { hidden: false, message: 'Completed runs could not be loaded.' };
}

/**
 * The history Clear control: drop every completed run from the list. A
 * usability gate only, like the queue's Clear — dead when the list is
 * provably empty or was never read. Two-step in the view.
 *
 * @param {ReturnType<typeof historyRecords>} records
 * @param {boolean} loaded
 * @returns {ControlState}
 */
export function historyClearControl(records, loaded) {
  if (!loaded) return { disabled: true, reason: 'Completed runs could not be loaded.' };
  if (records.length === 0) return { disabled: true, reason: 'No completed runs.' };
  return { disabled: false, reason: null };
}

/**
 * Banner tone for a queue write that SUCCEEDED.
 *
 * Green is reserved for outcomes that cannot move hardware. An action that
 * arms — starting the queue, or withdrawing a pending stop so it keeps
 * draining — reports in the caution tone even when it worked, because "it
 * worked" is the moment the operator most needs to register what is now
 * running.
 *
 * @param {boolean} arming
 * @returns {'ok'|'warn'}
 */
export function writeOutcomeTone(arming) {
  return arming ? 'warn' : 'ok';
}

/**
 * Body for moving `items[index]` one place towards the front, or `null` when
 * it is already there. Placement is relative to the neighbour's uid rather
 * than an index, so a concurrent add/remove between render and click moves the
 * item next to the same neighbour instead of into a stale slot.
 *
 * @param {QueueItem[]} items
 * @param {number} index
 * @returns {{before_uid: string}|null}
 */
export function moveUpBody(items, index) {
  const neighbour = index > 0 ? items[index - 1] : null;
  const uid = neighbour && typeof neighbour.item_uid === 'string' ? neighbour.item_uid : null;
  return uid === null ? null : { before_uid: uid };
}

/**
 * Body for moving `items[index]` one place towards the back, or `null` at the
 * end. See `moveUpBody` for why this is uid-relative.
 *
 * @param {QueueItem[]} items
 * @param {number} index
 * @returns {{after_uid: string}|null}
 */
export function moveDownBody(items, index) {
  const neighbour = index >= 0 && index < items.length - 1 ? items[index + 1] : null;
  const uid = neighbour && typeof neighbour.item_uid === 'string' ? neighbour.item_uid : null;
  return uid === null ? null : { after_uid: uid };
}

/**
 * The OSPREY run id the enqueue path stamped into an item's metadata, or
 * `null` for an item enqueued out of band (a `qserver` CLI, another client).
 * Such an item is real queue work and is shown, but it has no run to open in
 * the Results view.
 *
 * @param {QueueItem|null} item
 * @returns {string|null}
 */
export function itemRunId(item) {
  const meta = item && typeof item.meta === 'object' ? item.meta : null;
  const runId = meta ? meta.osprey_run_id : null;
  return typeof runId === 'string' && runId !== '' ? runId : null;
}

/**
 * The run that just left the running slot, if one did.
 *
 * Compares the running item across two queue frames by OSPREY run id: when the
 * previous frame was running a plan and the next frame shows that run gone —
 * the slot empty, or already handed to the next item — the finished run's id
 * is returned. `null` in every other case: nothing was running, the same run
 * is still going, or the item carried no OSPREY run id (enqueued out of band,
 * so there is no run for the Results view to open).
 *
 * The completion cue for the queue view's auto-follow: a run that finishes
 * with nothing selected is followed in the Results view rather than dropping
 * silently into the Completed-runs list — on a fast machine the whole run
 * fits between two glances, and this is what keeps its result from being
 * invisible afterwards.
 *
 * @param {QueueItem|null} previousRunning
 * @param {QueueItem|null} nextRunning
 * @returns {string|null}
 */
export function finishedRunId(previousRunning, nextRunning) {
  const previousId = itemRunId(previousRunning);
  if (previousId === null) return null;
  return itemRunId(nextRunning) === previousId ? null : previousId;
}

/**
 * A one-line summary of an item's parameters. Queueserver item `kwargs` ARE
 * the plan's params, unwrapped — there is no `params` envelope to open.
 *
 * @param {QueueItem|null} item
 * @param {number} [maxKeys]
 * @returns {string}
 */
export function itemParamSummary(item, maxKeys = 3) {
  const kwargs = item && typeof item.kwargs === 'object' && item.kwargs ? item.kwargs : null;
  if (kwargs === null) return '';
  const entries = Object.entries(kwargs);
  if (entries.length === 0) return '';
  const shown = entries.slice(0, maxKeys).map(([key, value]) => `${key}=${formatParam(value)}`);
  if (entries.length > maxKeys) shown.push(`+${entries.length - maxKeys} more`);
  return shown.join(', ');
}

/**
 * @param {unknown} value
 * @returns {string}
 */
function formatParam(value) {
  if (Array.isArray(value)) return `[${value.length}]`;
  if (value === null || value === undefined) return '—';
  if (typeof value === 'object') return '{…}';
  if (typeof value === 'number') return String(Number(value.toFixed(6)));
  const text = String(value);
  return text.length > 24 ? `${text.slice(0, 23)}…` : text;
}

/** @typedef {{
 *   mode: 'determinate'|'indeterminate',
 *   percent: number|null,
 *   label: string,
 *   complete: boolean,
 * }} ProgressView */

/**
 * How to render a running item's progress.
 *
 * The rule that matters: an unknown denominator renders INDETERMINATE, never
 * 0%. Progress is absent whenever the document plane knows nothing about the
 * run (including, legitimately, a run still executing after a bridge restart —
 * live rows are in-process only), and `fraction` is null whenever the expected
 * point count is unknown, which is the COMMON case for agent-authored session
 * plans. Rendering either as "0%" would show a working plan as a stalled one.
 *
 * `rows_seen` counts documents, not stored rows, so it can legitimately exceed
 * what the data table holds; and `fraction: 1.0` with `complete: false` is the
 * normal last-event state (the run's stop document has not landed yet), not a
 * stuck run — it reads as "finishing", never as an error.
 *
 * @param {any} progress
 * @returns {ProgressView}
 */
export function describeProgress(progress) {
  if (!progress || typeof progress !== 'object') {
    return { mode: 'indeterminate', percent: null, label: 'running', complete: false };
  }
  const rowsSeen = typeof progress.rows_seen === 'number' ? progress.rows_seen : 0;
  const complete = progress.complete === true;
  const fraction = typeof progress.fraction === 'number' ? progress.fraction : null;
  const points = `${rowsSeen} point${rowsSeen === 1 ? '' : 's'}`;

  if (fraction === null) {
    return {
      mode: 'indeterminate',
      percent: null,
      label: complete ? `${points} — finished` : `${points} so far`,
      complete,
    };
  }

  const expected = typeof progress.expected_points === 'number' ? progress.expected_points : null;
  const of = expected === null ? points : `${rowsSeen} of ${expected} points`;
  const tail = complete ? ' — finished' : fraction >= 1 ? ' — finishing' : '';
  return {
    mode: 'determinate',
    percent: Math.round(fraction * 100),
    label: `${of}${tail}`,
    complete,
  };
}

/** @typedef {{
 *   code: string|null,
 *   message: string,
 *   capability: any,
 *   managerState: string|null,
 * }} QueueRefusal */

/**
 * Classify a non-OK queue response.
 *
 * `detail.code` is the discriminator and `detail.detail` is the sentence to
 * show the operator VERBATIM — it is the bridge's own wording, and it carries
 * the remedy (the `osprey set connector=…` flip command on a
 * browse-only refusal, which arming header is missing on
 * `launch_token_required`). Rewording it here would put a second, drifting
 * copy of the bridge's policy in the panel.
 *
 * Nothing reads a top-level `code`: the queue relay hands the bridge's
 * envelope through untouched, so `code` only ever exists one level down. The
 * sidecar's own 502 is the one refusal with a plain-string detail.
 *
 * @param {number} status
 * @param {any} body
 * @returns {QueueRefusal}
 */
export function classifyQueueRefusal(status, body) {
  const detail = body && typeof body === 'object' ? body.detail : null;

  if (detail && typeof detail === 'object') {
    const code = typeof detail.code === 'string' ? detail.code : null;
    const message =
      typeof detail.detail === 'string' && detail.detail.trim() !== ''
        ? detail.detail
        : `The bridge refused the request (HTTP ${status}).`;
    return {
      code,
      message,
      capability: detail.capability ?? null,
      managerState: typeof detail.manager_state === 'string' ? detail.manager_state : null,
    };
  }

  if (typeof detail === 'string' && detail.trim() !== '') {
    return { code: null, message: detail, capability: null, managerState: null };
  }

  return {
    code: null,
    message: `The bridge refused the request (HTTP ${status}).`,
    capability: null,
    managerState: null,
  };
}

/**
 * The abort's one benign refusal: there was no plan to stop. Nothing is moving,
 * so this is an honest answer rather than a failed halt.
 */
export const NOTHING_RUNNING_CODE = 'nothing_running';

/**
 * Refusal codes that mean A HALT DID NOT LAND — the plan may still be moving
 * hardware right now. These are failures wherever they surface, not "the system
 * working as designed", so they never share the by-design amber.
 *
 * Only `abort_pause_timeout` is listed, and deliberately so: the abort
 * composition is its sole producer, so it carries the same meaning on every
 * route that could ever relay it. The abort's *other* failure codes
 * (`manager_unreachable`, `queue_request_rejected`) are absent because other
 * routes mint them as by-design degraded-state refusals; escalating them
 * globally would paint working deployments red. `abortRefusalTone` escalates
 * them on the halt path alone, where they unambiguously mean the plan was not
 * stopped.
 *
 * @type {ReadonlyArray<string>}
 */
export const HALT_FAILURE_REFUSAL_CODES = ['abort_pause_timeout'];

/**
 * Banner tone for a refusal. An arming refusal is a *warning*, not an error:
 * the deployment is working exactly as designed and the operator has an action
 * (arm it, or flip the connector). A genuine failure is an error.
 *
 * A failed halt is a genuine failure and is rendered as one: a coded refusal in
 * `HALT_FAILURE_REFUSAL_CODES` says the plan may still be moving hardware, which
 * is the opposite of "working as designed" and must not read as routine amber.
 *
 * @param {string|null} code
 * @returns {'warn'|'err'}
 */
export function refusalTone(code) {
  if (code === null) return 'err';
  return HALT_FAILURE_REFUSAL_CODES.includes(code) ? 'err' : 'warn';
}

/**
 * Banner tone for a refusal ON THE EMERGENCY-ABORT PATH.
 *
 * Every refusal of an abort means the machine was not stopped — the manager
 * never paused, never answered, or refused the abort outright — so the halt
 * path escalates the whole vocabulary to `err` rather than only the one code
 * `refusalTone` can safely escalate everywhere. This is per-call severity, not
 * a second copy of the code table: the same code stays amber on the routes
 * where it genuinely is by design.
 *
 * The single carve-out is `nothing_running`. It is not a failed halt: nothing
 * was running, so nothing is moving, and painting the reassuring answer red
 * would train operators to discount the colour that matters here.
 *
 * @param {string|null} code
 * @returns {'warn'|'err'}
 */
export function abortRefusalTone(code) {
  return code === NOTHING_RUNNING_CODE ? 'warn' : 'err';
}

/**
 * The completed runs to list, from `GET /runs`.
 *
 * `GET /runs` is the whole OSPREY-visible view — running, then pending, then
 * history — and the queue half above already shows the first two, so only
 * terminal records land here. Every field is read defensively: the record
 * shape is the bridge's, records are skipped rather than trusted when the one
 * key this panel needs (`id`) is missing, and no other key is required.
 *
 * @param {unknown} runs
 * @returns {Array<{id: string, status: string, planName: string|null, error: string|null}>}
 */
export function historyRecords(runs) {
  if (!Array.isArray(runs)) return [];
  const records = [];
  for (const run of runs) {
    if (!run || typeof run !== 'object') continue;
    const id = typeof run.id === 'string' && run.id !== '' ? run.id : null;
    if (id === null) continue;
    const status = typeof run.status === 'string' ? run.status : 'unknown';
    if (!TERMINAL_RUN_STATUSES.includes(status)) continue;
    records.push({
      id,
      status,
      planName: typeof run.plan_name === 'string' ? run.plan_name : null,
      error: typeof run.error === 'string' ? run.error : null,
    });
  }
  return records;
}

// ---------------------------------------------------------------------------
// SSE transport
// ---------------------------------------------------------------------------

const _INITIAL_BACKOFF_MS = 1000;
const _MAX_BACKOFF_MS = 15000;

/** @typedef {{close: () => void}} QueueStream */

/**
 * Subscribe to the queue's SSE stream with reconnect backoff.
 *
 * Backoff resets on a received FRAME rather than on `onopen`: a proxy that
 * accepts the connection and immediately drops it would otherwise count as a
 * success and reconnect at a fixed interval forever. `onConnectionChange(false)`
 * fires on every drop so the panel can mark itself stale instead of showing a
 * frozen queue as live.
 *
 * @param {string} url
 * @param {{
 *   onFrame: (frame: any) => void,
 *   onConnectionChange?: (connected: boolean) => void,
 *   EventSourceCtor?: typeof EventSource,
 * }} deps
 * @returns {QueueStream}
 */
export function createQueueStream(url, { onFrame, onConnectionChange, EventSourceCtor }) {
  const Ctor = EventSourceCtor || EventSource;
  let closed = false;
  let backoff = _INITIAL_BACKOFF_MS;
  /** @type {EventSource|null} */
  let source = null;
  /** @type {ReturnType<typeof setTimeout>|null} */
  let retryTimer = null;

  function connect() {
    if (closed) return;
    source = new Ctor(url);
    source.onmessage = (event) => {
      backoff = _INITIAL_BACKOFF_MS;
      if (onConnectionChange) onConnectionChange(true);
      try {
        onFrame(JSON.parse(event.data));
      } catch {
        // Malformed frame: ignore rather than kill the stream handler.
      }
    };
    source.onerror = () => {
      if (closed) return;
      if (source) source.close();
      if (onConnectionChange) onConnectionChange(false);
      const delay = backoff;
      backoff = Math.min(backoff * 2, _MAX_BACKOFF_MS);
      retryTimer = setTimeout(connect, delay);
    };
  }

  connect();

  return {
    close() {
      closed = true;
      if (retryTimer !== null) clearTimeout(retryTimer);
      if (source) source.close();
    },
  };
}
