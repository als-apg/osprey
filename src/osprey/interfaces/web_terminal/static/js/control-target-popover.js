// @ts-check
/* OSPREY Web Terminal — Control-Target Popover
 *
 * The panel behind the header chip, answering the three questions an operator
 * opens it with: which machine is the agent on and does writing there move
 * anything real; are writes on, off (their own doing, one click undoes it) or
 * locked (the deployment's, no click here moves it); and where else can the
 * session go. The chip (control-target-chip.js) owns the read, the poll and
 * the state; this module owns the card, the rows and the actions, and never
 * fetches the roster itself — it subscribes to the chip and re-renders from
 * `getState()`, so the popover and the chip can never disagree about the same
 * machine.
 *
 * **Two shapes, not one list.** The machine the agent stands on is a card —
 * name, ⓘ, and its writes control. Every other machine is a row: name, ⓘ,
 * its writes control, and Switch to. Where a control cannot act it stays
 * visible and says why on hover; what a machine is and what writing there
 * does — the consequence line, the server's own label, the endpoint — live
 * behind the name's ⓘ tooltip, never at rest.
 *
 * **One control per machine.** The write state is this login's switch for
 * that machine, one of three positions (`off` / `asked` / `allowed`), and the
 * control is a `radiogroup` whose `aria-checked` is always the STORED position
 * — never the focus and never a dialog's question. Arrows and Home/End move
 * focus only (roving tabindex, skipping disabled options); Space, Enter or a
 * click commit. A narrowing applies at once: it only ever takes reach away,
 * and no lock holds `off`. Every widening
 * confirms first, and the option it asked for reads `data-pending` while the
 * dialog is up. `off` is never disabled; an armed position a lock holds stays
 * on screen, disabled, with the reason on hover. So a stored `allowed` the
 * login's tier caps reads `asked` beside a checked, disabled `allowed`.
 *
 * **Focus survives a rebuild.** Every poll and push repaints the whole
 * subtree, so {@link render} re-focuses the option that had focus, and a
 * dismissed confirm hands focus back to the option that raised it.
 *
 * **The epoch the chip read.** Every switch gesture carries the
 * `switch_epoch` of the payload the operator was shown, as the decimal string
 * it was read as — never a number, which would round it. A stop that landed
 * since refuses a widening (409 `stopped_since`): the popover re-reads and the
 * operator clicks again — a widening never silently re-arms after a stop. The
 * one exception is a 409 that says it just STAMPED the record's first epoch,
 * which is retried once with that epoch.
 *
 * **No process claims.** Beyond the confirm's own sentence about the position
 * being chosen, nothing rendered here says what limits apply — that is
 * deployment configuration this module cannot see. Confirms state scope,
 * endpoint and consequence only.
 *
 * **A confirm can be waived, sometimes.** {@link confirmSkippable} decides. A
 * target switch and a move to `asked` on the rehearsal or the simulator carry
 * a "don't ask again" checkbox — per gesture, per machine, per render
 * (scopedStorageKey), remembered only when the operator confirms. A move to
 * `asked` on the real machine and a move to `allowed` anywhere keep asking
 * always. A waived dialog is re-shown by holding Shift on the gesture, which
 * is also where the waiver is undone.
 *
 * **The popover stays open beneath a confirm.** Both confirms raise a
 * `.posture-modal-overlay` at `--z-modal`, a full layer above the popover's
 * `--z-sticky`. The outside-click handler ignores clicks inside that overlay,
 * Escape dismisses an open confirm before it closes the popover, and a
 * confirmed change re-renders in place. An operator changing several rows
 * never has to reopen it.
 *
 * Every request goes through the chip's `targetRequest`, which unwraps this
 * route family's dict-detail refusals and runs through api.js's `withPrefix`
 * chokepoint — so the multi-user per-user mount (`/u/<name>/…`) is handled and
 * the operator reads the server's own wording rather than "[object Object]".
 */

import {
  POSITIONS,
  SWITCH_APPLIED,
  bannerNote,
  confirmSkipKeyBase,
  confirmSkippable,
  contextRefusalCode,
  contextRefusalPhrase,
  contextWritable,
  descriptor,
  descriptorTone,
  displayName,
  identTitle,
  lockReason,
  reachException,
  reasonPhrase,
  switchConfirm,
  TURN_ON_SENTENCE,
  turnOnConfirm,
} from './control-target-facts.js';
import { confirmSkipped } from './confirm-skip.js';
import {
  button,
  el,
  focusOption,
  focusedOption,
  renderSwitch,
  writePhraseLine,
} from './control-target-popover-switch.js';
import { dismissConfirm, isConfirmUp, showConfirm } from './posture-confirm.js';

import {
  CHIP_TOGGLE_EVENT,
  getAnchorElement,
  getChipElement,
  getState,
  isPending,
  kindAttr,
  markPending,
  postSwitch,
  refetch,
  setExpanded,
  stateWord,
  subscribe,
  switchRefusal,
  targetRequest,
} from './control-target-chip.js';

/**
 * The `target` value that turns every machine in this login's record off at
 * once. Mirrors `ALL_TARGETS` in routes/websocket.py; there is deliberately no
 * matching "turn everything on" — the route refuses it.
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

/** @type {HTMLElement|null} */
let popover = null;
/** @type {(() => void)|null} */
let unsubscribe = null;
/** Whether the popover is showing. Mirrored onto the chip's `aria-expanded`. */
let open = false;

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
 * One thing lives here: a refusal the POST came back with — the operator
 * clicked something and is owed the server's own sentence. It is a fact about
 * a click, not about the session, so it is cleared by the next gesture rather
 * than by the next read — a re-render from the 5 s poll must not silently
 * swallow the reason a click did nothing.
 * @type {Map<string, string>}
 */
const gestureNotes = new Map();

/**
 * The popover's status line: the widening sentence, after a move whose
 * confirm was waived lands — the one place it can be read when no dialog said
 * it. Shown until the next gesture or the popover closes; never beside the
 * off control.
 * @type {string|null}
 */
let statusNote = null;

/** Whether a POST from this popover is in flight (one gesture at a time). */
let posting = false;
/** A stop pressed while another POST was out, run when that one settles. */
/** @type {(() => Promise<void>)|null} */
let queuedStop = null;
const GESTURE_DEADLINE_MS = 15000;
const BUSY_SENTENCE = 'Another change is still being sent. Try again in a moment.';
const DEADLINE_SENTENCE = 'No answer from the server. The re-read shows what the record holds.';

/**
 * The widening whose confirm is on screen, if any.
 *
 * The option it asked for reads `data-pending` until the dialog is answered;
 * `aria-checked` stays on the stored position, because nothing has been
 * applied. Cleared on every dismissal path — {@link dismissConfirm} is the one
 * chokepoint.
 * @type {{target: string, position: 'asked'|'allowed'}|null}
 */
let confirmingMove = null;

/**
 * The handles a confirm hands to the gesture it raised — see
 * posture-confirm.js, which owns the dialog itself.
 * @typedef {import('./posture-confirm.js').ConfirmUi} ConfirmUi
 */

/* ---- mount ---- */

/**
 * Mount the popover under the header chip and keep it current.
 *
 * Idempotent, and a no-op on a page with no chip: the chip mounts into its
 * bar-item shell, `.header-actions`, or a host its caller passes, so on a page
 * with none of those there is nothing to hang under.
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
  queuedStop = null;
  pendingTarget = null;
  confirmingMove = null;
  statusNote = null;
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
  if (isConfirmUp()) {
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
  statusNote = null;
  dismissConfirm();
  popover?.classList.remove('open');
  setExpanded(false);
  detachDocumentListeners();
}

/* ---- derived facts (the pure ones live in control-target-facts.js) ---- */

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
  // No age in the copy: `age_s` decides WHETHER the line renders (the
  // freshness window above), never what it says. A ticking counter would
  // restate the mechanism, and every re-render — the 5 s poll, every
  // gesture's re-read — would bump it in front of the operator.
  // The RECORD's word for an accepted switch is `applied` — never `success`,
  // which nothing on this path has ever published. The returned `status` is a
  // different vocabulary: it is the `data-status` value terminal.css keys the
  // outcome line on, the same split kindAttr makes between the route's words
  // and the stylesheet's.
  if (last.status === SWITCH_APPLIED) return { status: 'success', text: '✓ switched' };
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

/**
 * Repaint the popover from the chip's current answer.
 *
 * Whole-subtree, on every render: the card and rows are a projection of one
 * payload and nothing in them is worth diffing, and rebuilding is what lets a
 * confirmed change land in place without the popover closing. Rendering while
 * closed is cheap and keeps the first frame after an open correct.
 */
function render() {
  if (!popover) return;
  const focused = focusedOption(popover);
  const state = getState();
  popover.replaceChildren();
  if (!state) return;
  const rows = Array.isArray(state.targets) ? state.targets : [];

  const banner = bannerNote(state);
  if (banner) {
    const note = el('div', 'ctc-banner', banner.text);
    note.dataset.tone = banner.tone;
    popover.append(note);
  }

  const active = rows.find((row) => row.active) || null;
  if (active) popover.append(renderCard(active, state));

  const others = rows.filter((row) => row !== active);
  if (others.length > 0) {
    popover.append(el('div', 'ctc-list-title', active ? 'Other machines' : 'Machines'));
    const list = el('div', 'ctc-rows');
    for (const row of others) list.append(renderRow(row, state));
    popover.append(list);
  }

  // A gesture that named every target has no row of its own to report on.
  const allNote = gestureNotes.get(ALL_TARGETS);
  if (allNote) {
    const outcome = el('div', 'ctc-outcome', `✗ ${reasonPhrase(allNote)}`);
    outcome.dataset.status = 'refused';
    popover.append(outcome);
  }

  if (statusNote) {
    const status = el('div', 'ctc-outcome ctc-status', statusNote);
    status.dataset.status = 'notice';
    popover.append(status);
  }

  popover.append(renderFoot(state));
  if (focused) focusOption(popover, focused.target, focused.position);
}

/**
 * The card: the machine the agent stands on. Name, consequence line, and the
 * writes switch that both states and changes the write state.
 * @param {any} row
 * @param {any} state
 */
function renderCard(row, state) {
  const kind = kindAttr(row);
  const word = stateWord(row);
  const lock = lockReason(row, state);
  const card = el('div', 'ctc-card');
  card.dataset.target = row.target;
  card.dataset.targetKind = kind;
  card.dataset.state = word;

  card.append(el('div', 'ctc-card-eyebrow', 'The agent is on'));

  const title = el('div', 'ctc-card-title');
  title.dataset.targetKind = kind;
  title.dataset.state = word;
  title.append(el('span', 'ctc-dot'));
  title.append(el('span', 'ctc-name', displayName(row, kind)));
  const info = infoAffordance(row, kind);
  if (info) title.append(info);
  card.append(title);

  card.append(writesSwitch(row, state, word));
  const phrase = writePhraseLine(row, word, lock);
  if (phrase) card.append(phrase);

  // The reach exception sits on the title line: the switch carries the write
  // state, so the card has no separate state line to put it on.
  const reach = reachException(row.reachability);
  if (reach) {
    const bad = el('span', 'ctc-reach-word', reach.text);
    bad.dataset.state = reach.state;
    bad.title = reach.title;
    title.append(bad);
  }

  for (const line of outcomeLines(row, state)) card.append(line);
  // Turning writes off on the machine the agent is ON only reaches it once
  // the connector is rebuilt, and that waits for the run in flight. Said out
  // loud, rather than leaving a button that appears to have done nothing.
  if (state.last_posture_realign?.state === 'pending') {
    const line = el('div', 'ctc-outcome', 'takes effect when the running execution finishes');
    line.dataset.status = 'realign';
    card.append(line);
  }
  return card;
}

/**
 * One other machine's row: who it is and what writing there means, its
 * writes switch, and Switch to.
 * @param {any} row
 * @param {any} state
 */
function renderRow(row, state) {
  const kind = kindAttr(row);
  const word = stateWord(row);
  const lock = lockReason(row, state);
  const node = el('div', 'ctc-row');
  node.dataset.target = row.target;
  node.dataset.targetKind = kind;
  node.dataset.state = word;

  const ident = el('div', 'ctc-ident');
  const name = el('div', 'ctc-name-line');
  name.append(el('span', 'ctc-name', displayName(row, kind)));
  const info = infoAffordance(row, kind);
  if (info) name.append(info);
  const reach = reachException(row.reachability);
  if (reach) {
    const bad = el('span', 'ctc-reach-word', reach.text);
    bad.dataset.state = reach.state;
    bad.title = reach.title;
    name.append(bad);
  }
  ident.append(name);
  const phrase = writePhraseLine(row, word, lock);
  if (phrase) ident.append(phrase);
  for (const line of outcomeLines(row, state)) ident.append(line);
  node.append(ident);

  node.append(writesSwitch(row, state, word));
  node.append(renderAction(row, state));
  return node;
}

/**
 * The ⓘ beside a name, and the tooltip behind it.
 *
 * Two facts in one place: what writing to this machine does (the consequence
 * line — the one that names hardware moving keeps its error tone), then the
 * machine vocabulary the name's `title` used to carry (the server's own
 * label, the endpoint, the measured reachability). A real element shown on
 * hover and keyboard focus rather than the browser's `title` bubble, which
 * is delayed, unstyled and mouse-only — the design system's shared info tip
 * (`.osprey-info` / `.osprey-tip`), with this popover's own classes kept for
 * the two line tones. Returns `null` for a row with nothing to say.
 * @param {any} row
 * @param {string} kind
 * @returns {HTMLElement|null}
 */
function infoAffordance(row, kind) {
  const what = descriptor(row, kind);
  const ident = identTitle(row);
  if (!what && !ident) return null;
  const info = button('osprey-info ctc-info', 'i');
  info.setAttribute('aria-label', `About ${displayName(row, kind)}`);
  const tip = el('span', 'osprey-tip ctc-tip');
  tip.setAttribute('role', 'tooltip');
  if (what) {
    const line = el('span', 'ctc-tip-what', what);
    const tone = descriptorTone(row, kind);
    if (tone) line.dataset.tone = tone;
    tip.append(line);
  }
  if (ident) tip.append(el('span', 'ctc-tip-ident', ident));
  info.append(tip);
  return info;
}

/**
 * The feedback lines a row or the card carries: the last switch's outcome
 * while it is news, and the reason the last click did nothing.
 * @param {any} row
 * @param {any} state
 * @returns {HTMLElement[]}
 */
function outcomeLines(row, state) {
  const lines = [];
  const outcome = switchOutcome(row, state);
  if (outcome) {
    const line = el('div', 'ctc-outcome', outcome.text);
    line.dataset.status = outcome.status;
    if (outcome.title) line.title = outcome.title;
    lines.push(line);
  }
  const gestureNote = gestureNotes.get(row.target);
  if (gestureNote) {
    const line = el('div', 'ctc-outcome', `✗ ${reasonPhrase(gestureNote)}`);
    line.dataset.status = 'refused';
    lines.push(line);
  }
  return lines;
}

/**
 * The Switch slot: the button, or the phrase for why there is no button.
 *
 * The refusal is keyed on the switch tool's own machine code, published by the
 * route — so the popover and the agent agree about the same refusal — and
 * rendered as {@link REASON_PHRASES}' operator phrase, with the route's
 * `reason_detail` sentence (falling back to the code) on the `title`.
 * @param {any} row
 * @param {any} state
 */
function renderAction(row, state) {
  const action = el('div', 'ctc-action');
  // One request is outstanding at a time, so while it is out no row offers a
  // second.
  if (isPending()) return action;
  if (row.available_now && contextWritable(state)) {
    const swap = button('ctc-switch', 'Switch to');
    swap.title = `Move all control reads and writes to ${displayName(row, kindAttr(row))}`;
    swap.addEventListener('click', (event) => {
      event.stopPropagation();
      confirmSwitchTo(row, state, event.shiftKey);
    });
    action.append(swap);
    return action;
  }
  // The row's own reason wins; the context supplies the fallback, and names
  // its holder itself — a two-word chip reading "another terminal" over an
  // agent's controls server would send the operator to a page that does not
  // exist.
  const code = row.reason || contextRefusalCode(state);
  const phrase = row.reason ? reasonPhrase(code) : contextRefusalPhrase(state);
  const reason = el('span', 'ctc-reason', phrase);
  const detail =
    typeof row.reason_detail === 'string' && row.reason_detail ? row.reason_detail : '';
  if (detail || code) reason.title = detail || String(code);
  action.append(reason);
  return action;
}

/**
 * The foot: all-off, and the popover's scope said once. `Turn all writes off`
 * only ever removes reach, so it applies on click; it is enabled exactly when
 * the read says some switch of this login's is armed (`any_armed`) — including a
 * machine no longer configured, which no row shows.
 *
 * The note beside it is the popover's scope, said once at rest: the switch is
 * the login's, so a stop recorded here reaches every session, notebook kernel
 * and hook of that login, not just the page it was made on.
 * @param {any} state
 */
function renderFoot(state) {
  const foot = el('div', 'ctc-foot');
  const all = button('ctc-all-off', 'Turn all writes off');
  all.disabled = state.any_armed !== true;
  all.addEventListener('click', (event) => {
    event.stopPropagation();
    void setSwitch(ALL_TARGETS, 'off', state);
  });
  foot.append(all);
  foot.append(el('span', 'ctc-foot-note', 'Applies to every session of this login'));
  return foot;
}

/**
 * One machine's writes control, wired to this popover's gestures.
 * @param {any} row @param {any} state @param {string} word
 */
function writesSwitch(row, state, word) {
  return renderSwitch(row, state, word, confirmingMove, (to, reshow) =>
    moveSwitch(row, to, state, reshow)
  );
}

/* ---- gestures ---- */

/**
 * POST one gesture and settle the UI around it, then re-read.
 *
 * The re-read is the whole point: the store is shared by every tab and
 * survives a restart, so what the popover shows next is what the server says,
 * never what this click intended. A refusal is kept where the operator is
 * looking: inside the confirm that raised the gesture when there is one
 * (nothing was applied, and dismissing it to put the reason on a row behind
 * would hide the answer to the question they had just been asked), otherwise
 * on `noteKey`'s row — the server's own sentence is more specific than
 * anything this module could invent. The one refusal that closes the confirm
 * is one that says the question it asked no longer holds (`close`).
 * @param {() => Promise<any>} send  the request
 * @param {string} noteKey  the row a dialog-less refusal lands on
 * @param {ConfirmUi|undefined} ui  the confirm this gesture came from, if any
 * @param {(body: any) => void} onAccepted  what this gesture means, applied
 *   before the confirm is dismissed and the state re-read
 * @param {(err: unknown) => {message: string, close: boolean}} [refusal]
 * @returns {Promise<void>}
 */
async function postGesture(send, noteKey, ui, onAccepted, refusal = plainRefusal, stop = false) {
  if (posting) {
    // A stop is never dropped: it runs the moment the outstanding POST
    // settles. Anything else waits for a fresh click, and a confirm says why.
    if (stop) queuedStop = () => postGesture(send, noteKey, ui, onAccepted, refusal, true);
    else if (ui) {
      ui.error.textContent = BUSY_SENTENCE;
      ui.error.hidden = false;
    }
    return;
  }
  posting = true;
  gestureNotes.clear();
  statusNote = null;
  if (ui) {
    ui.confirm.disabled = true;
    ui.cancel.disabled = true;
  }
  try {
    onAccepted(await withDeadline(send()));
    ui?.done();
  } catch (err) {
    const { message, close } = refusal(err);
    if (ui && !close) {
      ui.error.textContent = message;
      ui.error.hidden = false;
      ui.confirm.disabled = false;
      ui.cancel.disabled = false;
    } else {
      ui?.done();
      gestureNotes.set(noteKey, message);
    }
  }
  posting = false;
  const next = queuedStop;
  queuedStop = null;
  // The queued stop re-reads and renders on its own way out.
  if (next) return next();
  // A refusal may itself be news about the render, so every outcome re-reads.
  await refetch();
  render();
}

/**
 * Settle a gesture's request within {@link GESTURE_DEADLINE_MS}, so one hung
 * POST cannot hold the in-flight guard — and every later stop — forever. The
 * request itself may still land; the re-read shows what the record holds.
 * @template T
 * @param {Promise<T>} request
 * @returns {Promise<T>}
 */
function withDeadline(request) {
  /** @type {ReturnType<typeof setTimeout>|undefined} */
  let timer;
  const deadline = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(DEADLINE_SENTENCE)), GESTURE_DEADLINE_MS);
  });
  return Promise.race([request, deadline]).finally(() => clearTimeout(timer));
}

/** @param {unknown} err @returns {{message: string, close: boolean}} */
function plainRefusal(err) {
  return { message: err instanceof Error ? err.message : String(err), close: false };
}

/**
 * POST one machine's position for this login, or every machine's off
 * (C-POST), carrying the `switch_epoch` of the payload the operator was shown
 * — {@link postSwitch} owns the one stamped retry. A stop that overtook a
 * widening closes its confirm and wants a new click on what the re-read shows;
 * a stop that was not taken says so, and the re-read keeps showing the machine
 * armed — nothing here paints a position the record does not hold.
 * {@link moveSwitch} decides whether a move gets here directly or by a confirm.
 * @param {string} target  a configured target name, or {@link ALL_TARGETS}
 * @param {'off'|'asked'|'allowed'} to
 * @param {any} state  the payload the operator was shown
 * @param {ConfirmUi} [ui]  the confirm this gesture was raised from, if any
 * @param {boolean} [waived]  a widening whose confirm was skipped: the route
 *   checks the claim, and its sentence goes to the status line once it lands
 * @returns {Promise<void>}
 */
function setSwitch(target, to, state, ui, waived = false) {
  const epoch = typeof state.switch_epoch === 'string' ? state.switch_epoch : null;
  return postGesture(
    () => postSwitch(state.session_id ?? null, target, to, epoch, waived),
    target,
    ui,
    () => {
      if (waived) statusNote = TURN_ON_SENTENCE;
    },
    (err) => switchRefusal(err, to),
    to === 'off'
  );
}

/**
 * Route one committed option: the stored position is a no-op, a widening goes
 * through its confirm, and anything else POSTs at once. `off` is always a
 * narrowing, including from a record nobody can read.
 * @param {any} row
 * @param {'off'|'asked'|'allowed'} to
 * @param {any} state  the payload the operator was shown
 * @param {boolean} [reshow]  Shift was held: show the dialog even when waived
 */
function moveSwitch(row, to, state, reshow = false) {
  if (to === row.switch) return;
  if (to === 'off' || POSITIONS.indexOf(to) < POSITIONS.indexOf(row.switch)) {
    void setSwitch(row.target, to, state);
  } else {
    confirmWiden(row, to, state, reshow);
  }
}

/**
 * Ask for the switch, then hand the request to the chip.
 *
 * The route answers `202` with a `request_id` and the `generation` the record
 * will carry; nothing has switched yet. The chip owns what happens next — the
 * 500 ms poll, waiting until every live controls server reports that
 * generation, and the local deadline if nothing ever answers — so all this
 * does is record which row is waiting and hand over both values.
 *
 * **A request for the target the deployment is already on is not a wait.** It
 * is accepted at the CURRENT generation and writes no terminus, because
 * nothing moved. Arming a pending switch on it would leave the chip in
 * `switching…` until the TTL and then report `request_expired` for a request
 * that succeeded. Only reachable as a race — the active row offers no Switch —
 * but the answer says plainly that it happened, so it is read.
 * @param {any} row
 * @param {any} state
 * @param {ConfirmUi} [ui]  the confirm this gesture was raised from, if any
 * @returns {Promise<void>}
 */
function requestSwitch(row, state, ui) {
  const json = { session_id: state.session_id, target: row.target };
  const send = () => targetRequest('/api/terminal/target', { method: 'POST', json });
  return postGesture(send, row.target, ui, (body) => {
    const generation = typeof body?.generation === 'number' ? body.generation : null;
    const held = row.target === state.control_target;
    if (held && generation !== null && generation === state.generation) {
      pendingTarget = null;
      return;
    }
    pendingTarget = row.target;
    markPending(String(body?.request_id || ''), row.target, generation);
  });
}

/* ---- confirms (the dialog itself lives in posture-confirm.js) ---- */

/**
 * Release the parked option on every dismissal path: cancel, Escape, a
 * confirmed gesture's done(), the popover closing over it. Registered as the
 * widening confirm's `onDismiss`, which posture-confirm.js runs exactly once
 * per dialog. Focus returns to the option that raised the dialog, unless the
 * popover closed with it.
 */
function releaseParkedSwitch() {
  const move = confirmingMove;
  if (!move) return;
  confirmingMove = null;
  render();
  if (open) focusOption(popover, move.target, move.position);
}

/**
 * The confirm for one widening — or, when its waiver is recorded and Shift is
 * not held, the move applied directly. {@link confirmSkippable} decides which
 * moves take a waiver; the rest have no key to read or write.
 * Wording lives in control-target-facts.js ({@link turnOnConfirm}); this
 * only wires the gesture.
 * @param {any} row
 * @param {'asked'|'allowed'} to
 * @param {any} state
 * @param {boolean} [reshow]  Shift was held: show the dialog even when waived
 */
function confirmWiden(row, to, state, reshow = false) {
  const kind = kindAttr(row);
  const keyBase =
    to === 'asked' && confirmSkippable('writes-asked', kind)
      ? confirmSkipKeyBase('writes-asked', row)
      : null;
  if (keyBase && !reshow && confirmSkipped(keyBase)) {
    void setSwitch(row.target, to, state, undefined, true);
    return;
  }
  showConfirm({
    ...turnOnConfirm(row, kind, to),
    skipKeyBase: keyBase,
    onConfirm: (ui) => void setSwitch(row.target, to, state, ui),
    onDismiss: releaseParkedSwitch,
  });
  // After showConfirm: its stale-dialog sweep dismisses (and releases) any
  // previous confirm first. The render is what marks the option beneath the
  // dialog.
  confirmingMove = { target: row.target, position: to };
  render();
}

/**
 * The confirm for switching the deployment onto another machine — or, when its
 * waiver is recorded and Shift is not held, the request posted directly.
 * Wording lives in control-target-facts.js ({@link switchConfirm}); this
 * only wires the gesture.
 * @param {any} row
 * @param {any} state
 * @param {boolean} [reshow]  Shift was held: show the dialog even when waived
 */
function confirmSwitchTo(row, state, reshow = false) {
  const keyBase = confirmSkipKeyBase('switch', row);
  if (!reshow && confirmSkipped(keyBase)) {
    void requestSwitch(row, state);
    return;
  }
  showConfirm({
    ...switchConfirm(row, kindAttr(row), stateWord(row)),
    skipKeyBase: keyBase,
    onConfirm: (ui) => void requestSwitch(row, state, ui),
  });
}
