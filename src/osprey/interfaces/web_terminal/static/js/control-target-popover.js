// @ts-check
/* OSPREY Web Terminal — Control-Target Popover
 *
 * The panel behind the control-target chip, answering the three questions an
 * operator opens it with: which machine is the agent on and does writing
 * there move anything real; are writes on, off (their own doing, one click
 * undoes it) or locked (the deployment's, no click here moves it); and where
 * else can the session go. The chip (control-target-chip.js) owns the read,
 * the poll and the state; this module owns the panel, the gestures and the
 * confirms, and control-target-rows.js builds the card, the rows and the foot
 * it shows. It never fetches the roster itself — it subscribes to the chip and
 * re-renders from `getState()`, so the popover and the chip can never disagree
 * about the same machine.
 *
 * **A design-system popover.** The panel is a body-level `.osprey-popover`
 * (`#control-target-popover`, `role=dialog`) that the design system's
 * popover.js places and dismisses like every other popover in the shell:
 * beside the sidebar, 8 px right of its edge and level with the chip; under a
 * chip on a page bar (the JupyterLab bar), right-aligned below it. A press
 * outside, Escape, a window resize or blur close it, and so does a change of
 * the sidebar's width, which moves the edge it was placed against. Its title,
 * row and separator lines and its switches are the design system's
 * primitives; `.ctc-popover` scopes this panel's own rules in terminal.css.
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
 * **A confirm can be waived.** Every widening carries one "don't ask again"
 * checkbox covering every move to `asked` or `allowed` on every machine, and
 * every target switch carries another; both are per render (scopedStorageKey)
 * and remembered only when the operator confirms. A waived dialog is re-shown
 * by holding Shift on the gesture; "Ask again" under the foot clears both.
 *
 * **The popover stays open beneath a confirm.** Both confirms raise a
 * `.posture-modal-overlay` at `--z-modal`, a layer above the popover. The
 * popover is opened with popover.js's `ignore` set to that overlay, so a press
 * inside the confirm is not a press outside the popover; Escape dismisses an
 * open confirm before it closes the popover; and a confirmed change
 * re-renders in place. An operator changing several rows never has to reopen
 * it.
 *
 * Every request goes through the chip's `targetRequest`, which unwraps this
 * route family's dict-detail refusals and runs through api.js's `withPrefix`
 * chokepoint — so the multi-user per-user mount (`/u/<name>/…`) is handled and
 * the operator reads the server's own wording rather than "[object Object]".
 */

import {
  POSITIONS,
  bannerNote,
  SKIP_KEY_MOVE,
  SKIP_KEY_SWITCH,
  reasonPhrase,
  switchConfirm,
  TURN_ON_SENTENCE,
  turnOnConfirm,
} from './control-target-facts.js';
import { confirmSkipped, rememberConfirmSkip } from './confirm-skip.js';
import { el, focusOption, focusedOption } from './control-target-popover-switch.js';
import { dismissConfirm, isConfirmUp, showConfirm } from './posture-confirm.js';
import {
  closePopover as closeShellPopover,
  openPopover as openShellPopover,
} from '/design-system/js/popover.js';

import {
  CHIP_TOGGLE_EVENT,
  getChipElement,
  getState,
  kindAttr,
  markPending,
  postSwitch,
  refetch,
  stateWord,
  subscribe,
  switchRefusal,
  targetRequest,
} from './control-target-chip.js';
import { renderCard, renderFoot, renderRow } from './control-target-rows.js';

/**
 * The `target` value that turns every machine in this login's record off at
 * once. Mirrors `ALL_TARGETS` in routes/websocket.py; there is deliberately no
 * matching "turn everything on" — the route refuses it.
 */
export const ALL_TARGETS = 'all';

/**
 * The overlay posture-confirm.js raises. A press inside it answers the
 * question the popover asked; it never closes the popover.
 */
const CONFIRM_OVERLAY = '.posture-modal-overlay';

/** @type {HTMLElement|null} */
let popover = null;
/** @type {(() => void)|null} */
let unsubscribe = null;
/** The chip the toggle listener is on, so a re-mounted chip is re-wired. */
/** @type {HTMLElement|null} */
let wiredChip = null;
/** Watches the popover's `hidden`, the one signal every close path shares. */
/** @type {MutationObserver|null} */
let closeWatch = null;

/**
 * The target this browser's outstanding switch request named.
 *
 * The chip owns whether a request is outstanding (`isPending`); which
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
 * Mount the popover and keep it current.
 *
 * Idempotent, and a no-op on a page with no chip: the chip mounts into the
 * host its caller passes (the sidebar's control-system block, the JupyterLab
 * bar), and on a page that mounted none there is nothing to open it from.
 *
 * @returns {{open: () => void, close: () => void, isOpen: () => boolean}|null}
 */
export function initControlTargetPopover() {
  const chip = getChipElement();
  if (!chip) return null;

  if (!popover || !popover.isConnected) {
    popover = document.createElement('div');
    popover.className = 'osprey-popover ctc-popover';
    popover.id = 'control-target-popover';
    popover.setAttribute('role', 'dialog');
    popover.setAttribute('aria-label', 'Control target');
    // Focus can rest on the panel itself when a re-render removes the control
    // that held it, rather than falling out to <body>.
    popover.tabIndex = -1;
    popover.hidden = true;
    document.body.append(popover);
    // popover.js closes the panel for reasons this module never sees (a press
    // outside, Escape, a resize, another popover opening). Every one of them
    // sets `hidden`, so watching it is how a close always takes its confirm
    // and its status line with it.
    closeWatch = new MutationObserver(() => {
      if (popover?.hidden) onClosed();
    });
    closeWatch.observe(popover, { attributes: true, attributeFilter: ['hidden'] });
    document.addEventListener('keydown', onEscapeConfirmFirst, true);
    document.addEventListener('osprey-collapsible-change', closePopover);
  }

  if (wiredChip !== chip) {
    // The chip flips its own `aria-expanded` and announces the click; it never
    // decides what the popover does with it.
    wiredChip?.removeEventListener(CHIP_TOGGLE_EVENT, onChipToggle);
    chip.addEventListener(CHIP_TOGGLE_EVENT, onChipToggle);
    chip.setAttribute('aria-controls', popover.id);
    wiredChip = chip;
  }

  if (!unsubscribe) unsubscribe = subscribe(() => render());
  render();
  return { open: openPopover, close: closePopover, isOpen };
}

/**
 * Unmount the popover and release everything it holds: the subscription, the
 * document listeners, any confirm still on screen, and the node.
 */
export function teardownControlTargetPopover() {
  dismissConfirm();
  closeWatch?.disconnect();
  closeWatch = null;
  document.removeEventListener('keydown', onEscapeConfirmFirst, true);
  document.removeEventListener('osprey-collapsible-change', closePopover);
  unsubscribe?.();
  unsubscribe = null;
  wiredChip?.removeEventListener(CHIP_TOGGLE_EVENT, onChipToggle);
  wiredChip = null;
  popover?.remove();
  popover = null;
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

/** @returns {boolean} whether the popover is showing */
function isOpen() {
  return Boolean(popover && !popover.hidden);
}

/**
 * Show the popover, rendered from the chip's current answer, and put focus on
 * its first control: the panel sits at <body> level, so Tab from the chip
 * would never reach it.
 */
function openPopover() {
  const chip = getChipElement();
  if (!popover || !chip || isOpen()) return;
  statusNote = null;
  render();
  const sidebar = chip.closest('#sidebar');
  openShellPopover(
    popover,
    chip,
    sidebar
      ? { place: 'beside-right', edge: sidebar, ignore: CONFIRM_OVERLAY }
      : { place: 'below-end', ignore: CONFIRM_OVERLAY }
  );
  firstControl()?.focus({ preventScroll: true });
}

/** Hide the popover; {@link onClosed} takes any confirm it raised with it. */
function closePopover() {
  if (popover && isOpen()) closeShellPopover(popover);
}

/** What every close leaves behind: no confirm, no status line. */
function onClosed() {
  statusNote = null;
  dismissConfirm();
}

/**
 * Escape, in the order the things on screen were opened: a confirm the open
 * popover raised first, the popover only when none is up. Capture phase, so it
 * runs before popover.js's own Escape, which would close the popover and its
 * rows in the same keystroke as the confirm about them.
 * @param {KeyboardEvent} event
 */
function onEscapeConfirmFirst(event) {
  if (event.key !== 'Escape' || !isOpen() || !isConfirmUp()) return;
  event.stopImmediatePropagation();
  dismissConfirm();
}

/**
 * The control focus starts on: the first writes control's tab stop (its stored
 * position, or the nearest one below it no lock holds), else the first enabled
 * button (an ⓘ before it would open its tip on arrival).
 * @returns {HTMLElement|null}
 */
function firstControl() {
  const found =
    popover?.querySelector('.ctc-position[tabindex="0"]:not(:disabled)') ??
    popover?.querySelector('button:not(:disabled)');
  return found instanceof HTMLElement ? found : null;
}

/**
 * What the card and rows need from this module for one render: the facts about
 * the last gestures, and the gestures their controls run (control-target-rows.js).
 * @returns {import('./control-target-rows.js').RowContext}
 */
function rowContext() {
  return {
    pendingTarget,
    confirmingMove,
    gestureNotes,
    onMove: (row, to, state, reshow) => moveSwitch(row, to, state, reshow),
    onSwitchTo: (row, state, reshow) => confirmSwitchTo(row, state, reshow),
    onAllOff: (state) => void setSwitch(ALL_TARGETS, 'off', state),
  };
}

/* ---- render ---- */

/**
 * Repaint the popover from the chip's current answer.
 *
 * Whole-subtree, on every render: the card and rows are a projection of one
 * payload and nothing in them is worth diffing, and rebuilding is what lets a
 * confirmed change land in place without the popover closing. Rendering while
 * closed is cheap and keeps the first frame after an open correct. The control
 * that held focus is found again in the new subtree (the 5 s poll re-renders
 * under a keyboard user): a writes option by its machine and position, any
 * other control by its machine and class, and the panel itself takes focus
 * when that control is gone.
 */
function render() {
  if (!popover) return;
  const option = focusedOption(popover);
  const focused = option ? null : focusKey();
  const state = getState();
  popover.replaceChildren();
  if (!state) return;
  const rows = Array.isArray(state.targets) ? state.targets : [];
  const ctx = rowContext();

  const banner = bannerNote(state);
  if (banner) {
    const note = el('div', 'ctc-banner', banner.text);
    note.dataset.tone = banner.tone;
    popover.append(note);
  }

  const active = rows.find((row) => row.active) || null;
  if (active) {
    popover.append(el('div', 'osprey-popover-title', 'The agent is on'));
    popover.append(renderCard(active, state, ctx));
  }

  const others = rows.filter((row) => row !== active);
  if (others.length > 0) {
    if (active) popover.append(separator());
    popover.append(el('div', 'osprey-popover-title', active ? 'Other machines' : 'Machines'));
    const list = el('div', 'ctc-rows');
    for (const row of others) list.append(renderRow(row, state, ctx));
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

  popover.append(separator());
  popover.append(renderFoot(state, ctx));
  if (confirmSkipped(SKIP_KEY_MOVE) || confirmSkipped(SKIP_KEY_SWITCH)) {
    popover.append(renderAskAgain());
  }
  if (option) focusOption(popover, option.target, option.position);
  else restoreFocus(focused);
}

/**
 * The one way back from "don't ask again" that needs no Shift: says the dialogs
 * are waived and clears both slots.
 * @returns {HTMLElement}
 */
function renderAskAgain() {
  const line = el('div', 'ctc-ask-again', 'Not asking first on any machine.');
  const link = /** @type {HTMLButtonElement} */ (el('button', 'ctc-ask-again-link', 'Ask again'));
  link.type = 'button';
  link.addEventListener('click', () => {
    rememberConfirmSkip(SKIP_KEY_MOVE, false);
    rememberConfirmSkip(SKIP_KEY_SWITCH, false);
    render();
  });
  line.append(' ', link);
  return line;
}

/** @returns {HTMLElement} the design system's separator line */
function separator() {
  const sep = el('div', 'osprey-popover-sep');
  sep.setAttribute('role', 'separator');
  return sep;
}

/**
 * Where focus is inside the popover, as something a rebuilt subtree can match:
 * the machine its card or row is about, and the control's own class.
 * @returns {{target: string|null, cls: string}|null}
 */
function focusKey() {
  const focused = document.activeElement;
  if (!popover || !(focused instanceof HTMLElement) || focused === popover) return null;
  if (!popover.contains(focused) || !focused.classList.length) return null;
  const owner = /** @type {HTMLElement|null} */ (focused.closest('[data-target]'));
  return { target: owner?.dataset.target ?? null, cls: focused.classList[0] };
}

/**
 * Put focus back on the control {@link focusKey} described, or on the panel.
 * @param {{target: string|null, cls: string}|null} key
 */
function restoreFocus(key) {
  if (!key || !popover) return;
  const scope =
    key.target === null
      ? popover
      : popover.querySelector(`[data-target="${CSS.escape(key.target)}"]`);
  const again = scope?.querySelector(`.${CSS.escape(key.cls)}`);
  const usable = again instanceof HTMLElement && !again.matches(':disabled');
  (usable ? again : popover).focus({ preventScroll: true });
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
  if (isOpen()) focusOption(popover, move.target, move.position);
}

/**
 * The confirm for one widening — or, when the waiver is recorded and Shift is
 * not held, the move applied directly.
 * Wording lives in control-target-facts.js ({@link turnOnConfirm}); this
 * only wires the gesture.
 * @param {any} row
 * @param {'asked'|'allowed'} to
 * @param {any} state
 * @param {boolean} [reshow]  Shift was held: show the dialog even when waived
 */
function confirmWiden(row, to, state, reshow = false) {
  if (!reshow && confirmSkipped(SKIP_KEY_MOVE)) {
    void setSwitch(row.target, to, state, undefined, true);
    return;
  }
  showConfirm({
    ...turnOnConfirm(row, kindAttr(row), to),
    skipKeyBase: SKIP_KEY_MOVE,
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
  if (!reshow && confirmSkipped(SKIP_KEY_SWITCH)) {
    void requestSwitch(row, state);
    return;
  }
  showConfirm({
    ...switchConfirm(row, kindAttr(row), stateWord(row)),
    skipKeyBase: SKIP_KEY_SWITCH,
    onConfirm: (ui) => void requestSwitch(row, state, ui),
  });
}
