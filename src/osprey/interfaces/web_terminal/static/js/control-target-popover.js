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
 * name, consequence line, write state, and the one button that changes it.
 * Every other machine is a row: name and consequence, a state pill, a verb,
 * and Switch to. Where a control cannot act it stays visible and says why on
 * hover; the machine vocabulary (lock codes, endpoints, the server's own
 * label) lives on tooltips, never at rest.
 *
 * **Verbs, not toggles.** Turning writes off applies on click — it only ever
 * removes reach. Turning writes on confirms first, because it is the gesture
 * after which a write can land somewhere new. The button names the outcome,
 * so that asymmetry is visible before the click.
 *
 * **No process claims.** Nothing rendered here says whether a write will ask
 * for approval or what limits apply — that is deployment configuration this
 * module cannot see. Confirms state scope, endpoint and consequence only.
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
  REASON_STORE_UNAVAILABLE,
  bannerNote,
  descriptor,
  descriptorTone,
  displayName,
  identTitle,
  isChatSession,
  lockReason,
  reachException,
  reasonPhrase,
  statePhrase,
  switchConfirm,
  turnOnConfirm,
} from './control-target-facts.js';
import { fadeOutOverlay, mountOverlay } from './modal-overlay.js';

// Re-exported so the popover's public surface stays what it was before the
// derived-facts split; the wording itself lives in control-target-facts.js.
export { lockReason };
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
 * "turn everything on", because each target's ceiling is its own.
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
 * The confirm currently on screen, if any. One at a time: both confirms are
 * raised from the same popover, and a second overlay would bury the first
 * without dismissing it.
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
 * a target `Turn all writes off` reported in `skipped`. Both are facts about a
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
 * Whole-subtree, on every render: the card and rows are a projection of one
 * payload and nothing in them is worth diffing, and rebuilding is what lets a
 * confirmed change land in place without the popover closing. Rendering while
 * closed is cheap and keeps the first frame after an open correct.
 */
function render() {
  if (!popover) return;
  const state = getState();
  popover.replaceChildren();
  if (!state) return;
  const rows = Array.isArray(state.targets) ? state.targets : [];

  const banner = bannerNote(state, rows);
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
    for (const row of others) list.append(renderRow(row, state, rows));
    popover.append(list);
  }

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
 * The card: the machine the agent stands on. Name, consequence line, the
 * write state in a sentence, and the one button that changes it.
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
  card.dataset.real = String(Boolean(row.real_machine));

  card.append(el('div', 'ctc-card-eyebrow', 'The agent is on'));

  const title = el('div', 'ctc-card-title');
  title.dataset.targetKind = kind;
  title.dataset.state = word;
  title.append(el('span', 'ctc-dot'));
  title.append(el('span', 'ctc-name', displayName(row, kind)));
  title.title = identTitle(row);
  card.append(title);

  card.append(renderVerb(row, state, word, lock, 'ctc-card-verb'));

  const desc = descriptor(row, kind);
  if (desc) card.append(descLine(row, kind, desc));

  const sub = el('div', 'ctc-card-state');
  const phrase = el('span', 'ctc-state-phrase', statePhrase(word));
  phrase.dataset.state = word;
  if (lock) phrase.title = lock;
  sub.append(phrase);
  if (word === 'sandbox') sub.append(el('span', 'ctc-state-note', '— you turned them off'));
  const reach = reachException(row.reachability);
  if (reach) {
    const bad = el('span', 'ctc-reach-word', reach.text);
    bad.dataset.state = reach.state;
    bad.title = reach.title;
    sub.append(bad);
  }
  card.append(sub);

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
 * One other machine's row: who it is and what writing there means, the state
 * pill, the verb that changes it, and Switch to.
 * @param {any} row
 * @param {any} state
 * @param {any[]} rows
 */
function renderRow(row, state, rows) {
  const kind = kindAttr(row);
  const word = stateWord(row);
  const lock = lockReason(row, state);
  const node = el('div', 'ctc-row');
  node.dataset.target = row.target;
  node.dataset.targetKind = kind;
  node.dataset.state = word;
  node.dataset.real = String(Boolean(row.real_machine));

  const ident = el('div', 'ctc-ident');
  const name = el('div', 'ctc-name-line');
  name.append(el('span', 'ctc-name', displayName(row, kind)));
  name.title = identTitle(row);
  const reach = reachException(row.reachability);
  if (reach) {
    const bad = el('span', 'ctc-reach-word', reach.text);
    bad.dataset.state = reach.state;
    bad.title = reach.title;
    name.append(bad);
  }
  ident.append(name);
  const desc = descriptor(row, kind);
  if (desc) ident.append(descLine(row, kind, desc));
  for (const line of outcomeLines(row, state)) ident.append(line);
  node.append(ident);

  const pill = el('span', 'ctc-pill', word === 'writes' ? 'writes on' : word === 'sandbox' ? 'writes off' : 'locked');
  pill.dataset.state = word;
  if (lock) pill.title = lock;
  node.append(pill);

  node.append(renderVerb(row, state, word, lock, 'ctc-verb'));
  node.append(renderAction(row, state, rows));
  return node;
}

/**
 * The consequence line, toned: the one descriptor that names hardware moving
 * carries the error tone; every "nothing moves" (and "not set up") stays
 * muted.
 * @param {any} row
 * @param {string} kind
 * @param {string} desc
 * @returns {HTMLElement}
 */
function descLine(row, kind, desc) {
  const line = el('div', 'ctc-desc', desc);
  const tone = descriptorTone(row, kind);
  if (tone) line.dataset.tone = tone;
  return line;
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
 * The verb that changes a machine's write state, named for its outcome.
 *
 * Locked it stays on screen, disabled, with the reason on hover — the gap
 * where the control would be is explained rather than merely empty. Turning
 * off applies on click; turning on confirms first.
 * @param {any} row
 * @param {any} state
 * @param {string} word
 * @param {string|null} lock
 * @param {string} cls
 * @returns {HTMLButtonElement}
 */
function renderVerb(row, state, word, lock, cls) {
  const on = word === 'writes';
  const verb = button(cls, on ? 'Turn writes off' : 'Turn writes on');
  verb.dataset.direction = on ? 'off' : 'on';
  if (lock) {
    verb.disabled = true;
    verb.title = lock;
    return verb;
  }
  verb.addEventListener('click', (event) => {
    event.stopPropagation();
    if (on) void setPosture(row.target, 'sandbox', state);
    else confirmTurnOn(row, state);
  });
  return verb;
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
 * @param {any[]} rows
 */
function renderAction(row, state, rows) {
  const action = el('div', 'ctc-action');
  // A chat session has no controls server to address a request to; and one
  // request is outstanding at a time, so while it is out no row offers a
  // second.
  if (isChatSession(rows) || isPending()) return action;
  if (row.available_now && state.store_available) {
    const swap = button('ctc-switch', 'Switch to');
    swap.title = `Move this session's reads and writes to ${displayName(row, kindAttr(row))}`;
    swap.addEventListener('click', (event) => {
      event.stopPropagation();
      confirmSwitchTo(row, state);
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
 * The foot: the one-gesture narrowing, and the popover's scope said once.
 * `Turn all writes off` only ever removes reach, so it applies on click; it is
 * disabled when there is nothing left to turn off.
 * @param {any} state
 * @param {any[]} rows
 */
function renderFoot(state, rows) {
  const foot = el('div', 'ctc-foot');
  const all = button('ctc-all-off', 'Turn all writes off');
  const liftable = rows.some((row) => !lockReason(row, state) && stateWord(row) === 'writes');
  all.disabled = !liftable;
  all.addEventListener('click', (event) => {
    event.stopPropagation();
    void setPosture(ALL_TARGETS, 'sandbox', state);
  });
  foot.append(all);
  foot.append(el('span', 'ctc-foot-note', 'Your session only'));
  return foot;
}

/* ---- gestures ---- */

/**
 * Turn one target's writes off or on (or every target's off), then re-read.
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
    // a target whose writes stayed on is exactly what an operator who just
    // clicked "Turn all writes off" must not be left believing otherwise
    // about.
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
 * @param {{title: string,
 *          body: (import('./control-target-facts.js').ConfirmRun)[][],
 *          live: string|null, confirmLabel: string,
 *          onConfirm: (ui: ConfirmUi) => void}} spec
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
  // Assembled node by node (no innerHTML) so the emphasis can sit on the name
  // and the state word without any string ever being parsed as markup. An
  // `{em}` run is the facts module's DOM-free spelling of a `<strong>`.
  for (const line of spec.body) {
    const paragraph = el('p');
    for (const run of line) {
      paragraph.append(typeof run === 'string' ? run : strong(run.em));
    }
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
 * The confirm for turning one machine's writes on. Wording lives in
 * control-target-facts.js ({@link turnOnConfirm}); this only wires the
 * gesture.
 * @param {any} row
 * @param {any} state
 */
function confirmTurnOn(row, state) {
  showConfirm({
    ...turnOnConfirm(row, kindAttr(row)),
    onConfirm: (ui) => void setPosture(row.target, 'writes', state, ui),
  });
}

/**
 * The confirm for switching this session onto another machine. Wording lives
 * in control-target-facts.js ({@link switchConfirm}); this only wires the
 * gesture.
 * @param {any} row
 * @param {any} state
 */
function confirmSwitchTo(row, state) {
  showConfirm({
    ...switchConfirm(row, kindAttr(row), stateWord(row)),
    onConfirm: (ui) => void requestSwitch(row, state, ui),
  });
}
