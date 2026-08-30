// @ts-check
/* OSPREY Web Terminal — Control-Target Popover
 *
 * The panel behind the header chip: one row per configured control target,
 * and every gesture that CHANGES where this session writes. The chip
 * (control-target-chip.js) owns the read, the poll and the state; this module
 * owns the rows and the actions, and never fetches the roster itself — it
 * subscribes to the chip and re-renders from `getState()`, so the popover and
 * the chip can never disagree about the same machine.
 *
 * Each row answers the whole of what an operator needs to act on it: which
 * machine, where it points, whether anything is reaching it, the ceiling the
 * deployment rendered, this session's own narrowing, and what a switch would
 * do. Status and action stay in separate columns because reading the popover
 * and changing it are different gestures — and every element carries one fact,
 * once: the row's single identity line is the server's label, which already
 * says what kind of machine it names.
 *
 * **One DOM, two densities.** `html[data-ui-mode]` is a CSS concern and only a
 * CSS concern: every row renders the endpoint and role line, the reachability
 * word and age, the baseline tag, the lock reason and the foot note, in both
 * modes, and terminal.css hides what simple mode drops. Nothing here reads
 * `data-ui-mode` — a JS branch on it would make a live toggle-back rebuild the
 * DOM, and would put the density rule in two files that could drift.
 *
 * **The popover stays open beneath a confirm.** Arming writes and switching
 * both raise a `.posture-modal-overlay` at `--z-modal`, a full layer above the
 * popover's `--z-sticky`. The outside-click handler ignores clicks inside that
 * overlay, Escape dismisses an open confirm before it closes the popover, and
 * a confirmed change re-renders the rows in place. An operator changing
 * several rows never has to reopen it.
 *
 * **Only widening confirms.** Arming a target (read-only → writes) and
 * switching onto one are the two gestures that can end with a write landing
 * somewhere new, so both ask. Narrowing and `Sandbox everything` only ever
 * remove reach and apply on click.
 *
 * Every request goes through the chip's `targetRequest`, which unwraps this
 * route family's dict-detail refusals and runs through api.js's `withPrefix`
 * chokepoint — so the multi-user per-user mount (`/u/<name>/…`) is handled and
 * the operator reads the server's own wording rather than "[object Object]".
 */

import { fadeOutOverlay, mountOverlay } from './modal-overlay.js';
import {
  CHIP_TOGGLE_EVENT,
  getAnchorElement,
  getChipElement,
  getState,
  isPending,
  kindAttr,
  markPending,
  refetch,
  setExpanded,
  stateWord,
  subscribe,
  targetRequest,
} from './control-target-chip.js';

/**
 * The `target` value that narrows every configured target at once. Mirrors
 * `ALL_TARGETS` in routes/websocket.py; there is deliberately no matching
 * "arm everything", because each target's ceiling is its own.
 */
export const ALL_TARGETS = 'all';

/**
 * How recent a `last_switch` has to be to still be shown on its row.
 *
 * The route publishes the outcome and its age and leaves the call to the
 * renderer, because it is a question about what the operator is looking at
 * rather than about what happened: an outcome is news for about as long as
 * someone is still watching for it, and history afterwards. Past this the row
 * simply stops carrying it, so a popover opened an hour later describes the
 * machine rather than re-announcing a switch nobody is waiting on.
 */
export const OUTCOME_MAX_AGE_S = 60;

/** `available_now` reason a chat session's rows carry (routes/websocket.py). */
const REASON_CHAT_SESSION = 'chat_session';

/** The refusal word for a row whose Switch is missing because the store is. */
const REASON_STORE_UNAVAILABLE = 'store_unavailable';

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
const REASON_PHRASES = {
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
function reasonPhrase(code) {
  const word = String(code ?? '');
  return REASON_PHRASES[word] || word;
}

/** @type {HTMLElement|null} */
let popover = null;
/** @type {(() => void)|null} */
let unsubscribe = null;
/** Whether the popover is showing. Mirrored onto the chip's `aria-expanded`. */
let open = false;

/**
 * The confirm currently on screen, if any. One at a time: both confirms are
 * raised from a row of the same popover, and a second overlay would bury the
 * first without dismissing it.
 * @type {HTMLElement|null}
 */
let activeConfirm = null;

/**
 * The target this browser's outstanding switch request named.
 *
 * The chip owns whether a request is outstanding ({@link isPending}); which
 * ROW it was for is a question only the client that minted it can answer, and
 * it is what decides which row reads `switching…`.
 * @type {string|null}
 */
let pendingTarget = null;

/**
 * What the last gesture on a row did, when the ROUTE cannot say.
 *
 * Two things live here and nothing else: a refusal the POST came back with
 * (the operator clicked something and is owed the server's own sentence), and
 * a target `Sandbox everything` reported in `skipped`. Both are facts about a
 * click, not about the session, so they are cleared by the next gesture rather
 * than by the next read — a re-render from the 5 s poll must not silently
 * swallow the reason a click did nothing.
 * @type {Map<string, string>}
 */
const gestureNotes = new Map();

/** Whether a POST from this popover is in flight (one gesture at a time). */
let posting = false;

/**
 * The handles a confirm hands to the gesture it raised: where a refusal goes,
 * the two buttons to lock while the POST is out, and the one dismissal path.
 * @typedef {object} ConfirmUi
 * @property {HTMLElement} error
 * @property {HTMLButtonElement} confirm
 * @property {HTMLButtonElement} cancel
 * @property {() => void} done
 */

/* ---- mount ---- */

/**
 * Mount the popover under the header chip and keep it current.
 *
 * Idempotent, and a no-op on a page with no chip: the chip hides itself until
 * the terminal reports a session, and mounts into `.header-actions`, so this
 * has nothing to hang under on a page that renders no header actions.
 *
 * @returns {{open: () => void, close: () => void, isOpen: () => boolean}|null}
 */
export function initControlTargetPopover() {
  const chip = getChipElement();
  const anchor = getAnchorElement();
  if (!chip || !anchor) return null;

  if (!popover || !anchor.contains(popover)) {
    popover = document.createElement('div');
    popover.className = 'ctc-popover';
    popover.id = 'control-target-popover';
    popover.setAttribute('aria-label', 'Control target');
    anchor.appendChild(popover);
    chip.setAttribute('aria-controls', popover.id);
    // The chip flips its own `aria-expanded` and announces the click; it never
    // decides what the popover does with it.
    chip.addEventListener(CHIP_TOGGLE_EVENT, onChipToggle);
  }

  if (!unsubscribe) unsubscribe = subscribe(() => render());
  render();
  return { open: openPopover, close: closePopover, isOpen: () => open };
}

/**
 * Unmount the popover and release everything it holds: the subscription, the
 * document listeners, any confirm still on screen, and the node.
 */
export function teardownControlTargetPopover() {
  dismissConfirm();
  detachDocumentListeners();
  unsubscribe?.();
  unsubscribe = null;
  getChipElement()?.removeEventListener(CHIP_TOGGLE_EVENT, onChipToggle);
  popover?.remove();
  popover = null;
  open = false;
  posting = false;
  pendingTarget = null;
  gestureNotes.clear();
}

/** @param {Event} event */
function onChipToggle(event) {
  const expanded = /** @type {CustomEvent} */ (event).detail?.expanded;
  if (expanded) openPopover();
  else closePopover();
}

/* ---- open / close ---- */

/**
 * A click anywhere that is not the chip, the popover, or a confirm raised by
 * it. Capture phase, so it dismisses before the click does anything else.
 * @param {MouseEvent} event
 */
function onDocumentClick(event) {
  const target = /** @type {Node|null} */ (event.target);
  if (!target) return;
  const anchor = getAnchorElement();
  if (anchor?.contains(target)) return;
  // The confirm mounts on `document.body`, outside the anchor, and is a full
  // layer ABOVE the popover: clicking inside it is the operator answering the
  // question the popover asked, not leaving the popover.
  if (target instanceof Element && target.closest('.posture-modal-overlay')) return;
  closePopover();
}

/**
 * Escape, in the order the things on screen were opened: a confirm first, the
 * popover only when none is up. Cancelling a confirm and losing the rows it
 * was about in the same keystroke would undo the whole point of leaving the
 * popover open beneath it.
 * @param {KeyboardEvent} event
 */
function onDocumentKeydown(event) {
  if (event.key !== 'Escape') return;
  if (activeConfirm) {
    event.stopPropagation();
    dismissConfirm();
    return;
  }
  closePopover();
  getChipElement()?.focus();
}

function attachDocumentListeners() {
  document.addEventListener('click', onDocumentClick, true);
  document.addEventListener('keydown', onDocumentKeydown, true);
}

function detachDocumentListeners() {
  document.removeEventListener('click', onDocumentClick, true);
  document.removeEventListener('keydown', onDocumentKeydown, true);
}

/** Show the popover, rendered from the chip's current answer. */
function openPopover() {
  if (!popover || open) return;
  open = true;
  render();
  popover.classList.add('open');
  setExpanded(true);
  attachDocumentListeners();
}

/** Hide the popover, and any confirm it raised. */
function closePopover() {
  if (!open) return;
  open = false;
  dismissConfirm();
  popover?.classList.remove('open');
  setExpanded(false);
  detachDocumentListeners();
}

/* ---- derived facts ---- */

/**
 * Whether this session is a chat.
 *
 * A chat has no PTY and so no controls server of its own to address a switch
 * request to, which the route says by giving EVERY row `chat_session` as its
 * unavailability reason. Its toggles are untouched — the posture store is
 * keyed on the session, not on the topology.
 * @param {any[]} rows
 */
function isChatSession(rows) {
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
function writesHeldByTheRun(row) {
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
function headNote(state, rows) {
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
function reachText(reachability) {
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

/**
 * What the last switch did to THIS row, when it is still news.
 *
 * Matched on the target the outcome names, never on "the session moved": the
 * chip matches the request by `request_id` and hands the outcome on, and a
 * row that did not take part in it has nothing to report.
 * @param {any} row
 * @param {any} state
 * @returns {{status: string, text: string, title?: string}|null}
 */
function switchOutcome(row, state) {
  if (isPending() && pendingTarget === row.target) {
    return { status: 'pending', text: 'switching…' };
  }
  const last = state.last_switch;
  if (!last || last.target !== row.target) return null;
  if (typeof last.age_s === 'number' && last.age_s > OUTCOME_MAX_AGE_S) return null;
  if (last.status === 'success') {
    const age = typeof last.age_s === 'number' ? ` · ${last.age_s} s ago` : '';
    return { status: 'success', text: `✓ switched${age}` };
  }
  // refused / failed / expired render the operator phrase for the word the
  // gate (or the client's own deadline, for a request nothing ever answered)
  // put on them, with the gate's own sentence on the title where it sent one.
  return {
    status: last.status === 'expired' ? 'expired' : 'refused',
    text: `✗ ${reasonPhrase(last.reason || last.status)}`,
    title: typeof last.detail === 'string' && last.detail ? last.detail : undefined,
  };
}

/* ---- render ---- */

/** @param {string} tag @param {string} [cls] @param {string} [text] */
function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
}

/**
 * A button, typed and with its `type` set. Every button here lives inside no
 * form, but an unset `type` is `submit` and a stray form anywhere above would
 * make each of these navigate.
 * @param {string} cls @param {string} text @returns {HTMLButtonElement}
 */
function button(cls, text) {
  const node = /** @type {HTMLButtonElement} */ (el('button', cls, text));
  node.type = 'button';
  return node;
}

/**
 * Repaint the popover from the chip's current answer.
 *
 * Whole-subtree, on every render: the rows are a projection of one payload and
 * nothing in them is worth diffing, and rebuilding is what lets a confirmed
 * change land in place without the popover closing. Rendering while closed is
 * cheap and keeps the first frame after an open correct.
 */
function render() {
  if (!popover) return;
  const state = getState();
  popover.replaceChildren();
  if (!state) return;
  const rows = Array.isArray(state.targets) ? state.targets : [];

  const head = el('div', 'ctc-head');
  const title = el('span', 'ctc-head-title', 'Control target · this session');
  title.id = 'ctc-head-title';
  head.append(title);
  popover.setAttribute('aria-labelledby', title.id);
  const note = el('span', 'ctc-head-note');
  const { text: noteText, tone } = headNote(state, rows);
  note.textContent = noteText;
  if (tone) note.dataset.tone = tone;
  head.append(note);
  popover.append(head);

  const list = el('div', 'ctc-rows');
  for (const row of rows) list.append(renderRow(row, state, rows));
  popover.append(list);

  // A gesture that named every target has no row of its own to report on.
  const allNote = gestureNotes.get(ALL_TARGETS);
  if (allNote) {
    const outcome = el('div', 'ctc-outcome', `✗ ${reasonPhrase(allNote)}`);
    outcome.dataset.status = 'refused';
    popover.append(outcome);
  }

  popover.append(renderFoot(state, rows));
}

/**
 * One target's row: the dot, who it is, the posture this session holds on it,
 * and the one action available.
 * @param {any} row
 * @param {any} state
 * @param {any[]} rows
 */
function renderRow(row, state, rows) {
  const word = stateWord(row);
  const lock = lockReason(row, state);
  const node = el('div', 'ctc-row');
  node.dataset.target = row.target;
  node.dataset.targetKind = kindAttr(row);
  node.dataset.state = word;
  node.dataset.real = String(Boolean(row.real_machine));
  node.dataset.active = String(Boolean(row.active));

  node.append(el('span', 'ctc-dot'));

  const ident = el('div', 'ctc-ident');
  // The row's ONE identity line: the server's label, which already carries the
  // kind of machine it names ("LIVE MACHINE (stand-in)", "virtual accelerator
  // (simulation)"). A subtitle restating it would be the same fact twice on
  // the one surface whose job is to be read at a glance.
  const name = el('div', 'ctc-name');
  name.append(el('span', 'ctc-label', row.label || row.target));
  if (row.active) name.append(el('span', 'ctc-tag ctc-tag-current', 'current'));
  ident.append(name);

  const meta = el('div', 'ctc-meta');
  meta.append(el('span', 'ctc-endpoint', row.endpoint || ''));
  const reach = reachText(row.reachability);
  if (row.reachability?.role) meta.append(el('span', 'ctc-role', String(row.reachability.role)));
  ident.append(meta);

  const reachNode = el('div', 'ctc-reach');
  reachNode.dataset.state = reach.state;
  reachNode.title = reach.title;
  reachNode.append(el('i', 'ctc-reach-dot'));
  reachNode.append(el('span', 'ctc-reach-text', reach.text));
  if (row.is_baseline) reachNode.append(el('span', 'ctc-baseline', 'baseline'));
  ident.append(reachNode);

  const outcome = switchOutcome(row, state);
  if (outcome) {
    const line = el('div', 'ctc-outcome', outcome.text);
    line.dataset.status = outcome.status;
    if (outcome.title) line.title = outcome.title;
    ident.append(line);
  }
  const gestureNote = gestureNotes.get(row.target);
  if (gestureNote) {
    const line = el('div', 'ctc-outcome', `✗ ${reasonPhrase(gestureNote)}`);
    line.dataset.status = 'refused';
    ident.append(line);
  }
  // A narrowing on the target the session is ON only reaches the agent once
  // the connector is rebuilt, and that waits for the run in flight. Said out
  // loud, rather than leaving a toggle that appears to have done nothing.
  if (row.active && state.last_posture_realign?.state === 'pending') {
    const line = el('div', 'ctc-outcome', 'read-only applies after the running execution finishes');
    line.dataset.status = 'realign';
    ident.append(line);
  }
  node.append(ident);

  node.append(renderPosture(row, state, word, lock));
  node.append(renderAction(row, state, rows));
  return node;
}

/**
 * Column 3: the two-segment posture toggle, and the reason it cannot move.
 *
 * Locked, the toggle keeps showing which state holds — it is the readout as
 * well as the control — and only loses its affordance. The reason goes both
 * under it and on its `title`, because simple mode drops the line and keeps
 * the tooltip.
 * @param {any} row
 * @param {any} state
 * @param {string} word
 * @param {string|null} lock
 */
function renderPosture(row, state, word, lock) {
  const posture = el('div', 'ctc-posture');
  const toggle = el('div', 'ctc-toggle');
  toggle.setAttribute('role', 'group');
  toggle.setAttribute('aria-label', `Session posture on ${row.label || row.target}`);
  toggle.dataset.locked = String(Boolean(lock));
  if (lock) toggle.title = lock;

  for (const seg of ['writes', 'read-only']) {
    const segment = button('ctc-seg', seg);
    segment.dataset.seg = seg;
    const pressed = seg === 'writes' ? word === 'writes' : word !== 'writes';
    segment.setAttribute('aria-pressed', String(pressed));
    if (lock) {
      segment.disabled = true;
      segment.title = lock;
    } else if (!pressed) {
      segment.addEventListener('click', (event) => {
        event.stopPropagation();
        if (seg === 'writes') confirmArming(row, state);
        else void setPosture(row.target, 'sandbox', state);
      });
    }
    toggle.append(segment);
  }
  posture.append(toggle);
  if (lock) posture.append(el('span', 'ctc-lock', lock));
  return posture;
}

/**
 * Column 4: Switch, or the phrase for why there is no Switch.
 *
 * The refusal is keyed on the switch tool's own machine code, published by the
 * route — so the popover and the agent agree about the same refusal — and
 * rendered as {@link REASON_PHRASES}' operator phrase, with the route's
 * `reason_detail` sentence (falling back to the code) on the `title`. The gap
 * where the button would be is explained rather than merely empty, and never
 * in a vocabulary only the tool speaks.
 * @param {any} row
 * @param {any} state
 * @param {any[]} rows
 */
function renderAction(row, state, rows) {
  const action = el('div', 'ctc-action');
  // The current row says so with its `current` tag; a chat session has no
  // controls server to address a request to; and one request is outstanding at
  // a time, so while it is out no row offers a second.
  if (row.active || isChatSession(rows) || isPending()) return action;
  if (row.available_now && state.store_available) {
    const swap = button('ctc-switch', 'Switch');
    swap.title = `Switch this session to ${row.label || row.target}`;
    swap.addEventListener('click', (event) => {
      event.stopPropagation();
      confirmSwitch(row, state);
    });
    action.append(swap);
    return action;
  }
  const code = row.reason || (state.store_available ? '' : REASON_STORE_UNAVAILABLE);
  const reason = el('span', 'ctc-reason', reasonPhrase(code));
  const detail = typeof row.reason_detail === 'string' && row.reason_detail ? row.reason_detail : '';
  if (detail || code) reason.title = detail || String(code);
  action.append(reason);
  return action;
}

/**
 * The foot: the one-gesture narrowing, and the sentence that bounds the whole
 * popover. `Sandbox everything` only ever removes reach, so it applies on
 * click; it is disabled when there is nothing left to lift.
 * @param {any} state
 * @param {any[]} rows
 */
function renderFoot(state, rows) {
  const foot = el('div', 'ctc-foot');
  const all = button('ctc-sandbox-all', 'Sandbox everything');
  const liftable = rows.some((row) => !lockReason(row, state) && stateWord(row) === 'writes');
  all.disabled = !liftable;
  all.addEventListener('click', (event) => {
    event.stopPropagation();
    void setPosture(ALL_TARGETS, 'sandbox', state);
  });
  foot.append(all);
  foot.append(el('span', 'ctc-foot-note', "Nothing here changes the deployment's config."));
  return foot;
}

/* ---- gestures ---- */

/**
 * Narrow or widen one target (or every target), then re-read.
 *
 * The re-read is the whole point: the store is shared by every tab and
 * survives a restart, so what the popover shows next is what the server says,
 * never what this click intended. A refusal is kept on the row it was for,
 * because the operator is looking at that row and the server's own sentence is
 * more specific than anything this module could invent.
 * A gesture raised from a confirm passes that dialog's `ui`, and a refusal
 * then stays inside it: the dialog is where the operator is looking, nothing
 * was applied, and dismissing it to put the reason on a row behind would hide
 * the answer to the question they had just been asked.
 * @param {string} target  a configured target name, or {@link ALL_TARGETS}
 * @param {'sandbox'|'writes'} posture
 * @param {any} state  the payload the operator was shown
 * @param {ConfirmUi} [ui]  the confirm this gesture was raised from, if any
 * @returns {Promise<void>}
 */
async function setPosture(target, posture, state, ui) {
  if (posting) return;
  posting = true;
  gestureNotes.clear();
  if (ui) {
    ui.confirm.disabled = true;
    ui.cancel.disabled = true;
  }
  try {
    const body = await targetRequest('/api/terminal/posture', {
      method: 'POST',
      json: { session_id: state.session_id, target, posture },
    });
    // `all` narrows what it can and reports the rest rather than dropping it:
    // a target that stayed writable is exactly what an operator who just
    // clicked "Sandbox everything" must not be left believing otherwise about.
    for (const skip of Array.isArray(body?.skipped) ? body.skipped : []) {
      if (skip?.target) gestureNotes.set(String(skip.target), String(skip.reason || 'skipped'));
    }
  } catch (err) {
    posting = false;
    const message = err instanceof Error ? err.message : String(err);
    if (ui) {
      ui.error.textContent = message;
      ui.error.hidden = false;
      ui.confirm.disabled = false;
      ui.cancel.disabled = false;
    } else {
      gestureNotes.set(target, message);
    }
    // The refusal may itself be news about the render, so re-read rather than
    // keeping whatever the rows showed.
    await refetch();
    render();
    return;
  }
  posting = false;
  ui?.done();
  await refetch();
  render();
}

/**
 * Ask the controls server to switch, then hand the request to the chip.
 *
 * The route accepts and answers `202` with a `request_id`; nothing has
 * switched yet. The chip owns what happens next — the 500 ms poll, matching
 * the outcome by that id, and calling it expired if nothing ever answers — so
 * all this does is record which row is waiting.
 * @param {any} row
 * @param {any} state
 * @param {ConfirmUi} ui
 * @returns {Promise<void>}
 */
async function requestSwitch(row, state, ui) {
  ui.confirm.disabled = true;
  ui.cancel.disabled = true;
  posting = true;
  gestureNotes.clear();
  let body;
  try {
    body = await targetRequest('/api/terminal/target', {
      method: 'POST',
      json: { session_id: state.session_id, target: row.target },
    });
  } catch (err) {
    posting = false;
    // The refusal stays in the dialog: it is where the operator is looking,
    // and nothing was requested, so there is nothing to watch for.
    ui.error.textContent = err instanceof Error ? err.message : String(err);
    ui.error.hidden = false;
    ui.confirm.disabled = false;
    ui.cancel.disabled = false;
    await refetch();
    return;
  }
  posting = false;
  ui.done();
  pendingTarget = row.target;
  markPending(String(body?.request_id || ''), row.target);
  await refetch();
  render();
}

/* ---- confirms ---- */

/** @param {string} text */
function strong(text) {
  return el('strong', undefined, text);
}

/**
 * Build and show one confirm over the popover, which stays open beneath it.
 *
 * Structure and lifecycle mirror the badge-era dialog (posture-badge.js): the
 * overlay is appended to `document.body`, `.visible` lands on the next frame,
 * and one `done()` runs on every dismissal path. What differs is where Escape
 * is handled — this popover owns it, so a confirm and the rows behind it are
 * dismissed in the order they were opened.
 * @param {{title: string, body: (string|Node)[][], live: string|null,
 *          confirmLabel: string, onConfirm: (ui: ConfirmUi) => void}} spec
 */
function showConfirm(spec) {
  dismissConfirm();
  // A dialog dismissed a moment ago is still in the DOM, fading out. Drop it
  // now rather than stacking a second overlay on top of it.
  for (const stale of document.querySelectorAll('.posture-modal-overlay[data-closing]')) {
    stale.remove();
  }

  const overlay = el('div', 'posture-modal-overlay');
  const dialog = el('div', 'posture-modal');
  dialog.setAttribute('role', 'dialog');
  dialog.setAttribute('aria-modal', 'true');
  dialog.setAttribute('aria-labelledby', 'posture-modal-title');

  const heading = el('div', 'posture-modal-title', spec.title);
  heading.id = 'posture-modal-title';

  const body = el('div', 'posture-modal-body');
  // Assembled node by node (no innerHTML) so the emphasis can sit on the label
  // and the state word without any string ever being parsed as markup.
  for (const line of spec.body) {
    const paragraph = el('p');
    paragraph.append(...line);
    body.append(paragraph);
  }
  if (spec.live) body.append(el('div', 'posture-modal-live', spec.live));

  const error = el('div', 'posture-modal-error');
  error.setAttribute('role', 'alert');
  error.hidden = true;

  const actions = el('div', 'posture-modal-actions');
  const cancel = button('posture-modal-cancel', 'Cancel');
  const confirm = button('posture-modal-confirm', spec.confirmLabel);
  if (spec.live) confirm.dataset.live = 'true';
  actions.append(cancel, confirm);

  dialog.append(heading, body, error, actions);
  overlay.append(dialog);
  activeConfirm = overlay;
  mountOverlay(overlay);

  cancel.addEventListener('click', () => dismissConfirm());
  confirm.addEventListener('click', () =>
    spec.onConfirm({ error, confirm, cancel, done: dismissConfirm })
  );
  confirm.focus();
}

/** Take the confirm off the screen, if one is up. Idempotent. */
function dismissConfirm() {
  const overlay = activeConfirm;
  activeConfirm = null;
  if (!overlay) return;
  // `data-closing` is what tells "still up" from "on its way out" — to a
  // reader, to a test, and to the stale sweep in showConfirm.
  overlay.dataset.closing = '1';
  fadeOutOverlay(overlay);
}

/**
 * The confirm for arming one target.
 *
 * Only this direction asks. Narrowing removes reach and is undone by a click;
 * arming is the gesture after which a write the agent makes can land.
 *
 * The title already names the machine, so the body does not name it again:
 * each line is one fact the title does not carry — the scope and the endpoint,
 * then the guards that stay up and when it takes hold.
 * @param {any} row
 * @param {any} state
 */
function confirmArming(row, state) {
  const label = row.label || row.target;
  showConfirm({
    title: `Allow writes on ${label}?`,
    body: [
      ['Arms writes for ', strong('this session'), ` · ${row.endpoint}.`],
      [
        'Per-write approval and channel limits still apply. Takes effect on the next write — ' +
          'no restart.',
      ],
    ],
    live: kindAttr(row) === 'live' ? 'Real machine. A confirmed write moves hardware.' : null,
    confirmLabel: 'Allow writes',
    onConfirm: (ui) => void setPosture(row.target, 'writes', state, ui),
  });
}

/**
 * The confirm for switching this session onto another machine.
 *
 * It names the posture the session will ARRIVE in, because that is the fact
 * the chip will read a moment later and the one an operator is most likely to
 * assume travels with them: posture is per-target, and it does not. The word
 * is {@link stateWord}'s own, so the dialog and the chip a moment later can
 * never disagree. The title already names the machine, so the body does not
 * name it again.
 * @param {any} row
 * @param {any} state
 */
function confirmSwitch(row, state) {
  const label = row.label || row.target;
  const word = stateWord(row);
  const live =
    kindAttr(row) === 'live'
      ? word === 'writes'
        ? 'Real machine, writes armed. The next write the agent makes lands on hardware.'
        : 'Real machine. Writes are sandboxed on it for this session.'
      : null;
  showConfirm({
    title: `Switch to ${label}?`,
    body: [
      ['Arrives in the ', strong(word), ` posture · ${row.endpoint}.`],
      ['The conversation continues.'],
    ],
    live,
    confirmLabel: 'Switch',
    onConfirm: (ui) => void requestSwitch(row, state, ui),
  });
}
