// @ts-check
/* OSPREY Web Terminal — Posture Confirm Dialog
 *
 * The `.posture-modal` overlay both control-target confirms are built from,
 * split out of control-target-popover.js: that module owns what the dialogs
 * SAY and what confirming does; this one owns the mechanics — build, mount,
 * the one-at-a-time rule, the "don't ask again" checkbox, and the single
 * dismissal chokepoint.
 *
 * Structure and lifecycle mirror the badge-era dialog (posture-badge.js): the
 * overlay is appended to `document.body`, `.visible` lands on the next frame,
 * and one dismissal path runs on every way out. One confirm at a time: both
 * are raised from the same popover, and a second overlay would bury the first
 * without dismissing it.
 */

import { fadeOutOverlay, mountOverlay } from './modal-overlay.js';
import { confirmSkipped, rememberConfirmSkip } from './confirm-skip.js';

/**
 * The handles a confirm hands to the gesture it raised: where a refusal goes,
 * the two buttons to lock while the POST is out, and the one dismissal path.
 * @typedef {object} ConfirmUi
 * @property {HTMLElement} error
 * @property {HTMLButtonElement} confirm
 * @property {HTMLButtonElement} cancel
 * @property {() => void} done
 */

/**
 * The confirm currently on screen, if any, and the dismissal callback its
 * raiser registered.
 * @type {{overlay: HTMLElement, onDismiss?: () => void}|null}
 */
let active = null;

/** @param {string} tag @param {string} [cls] @param {string} [text] */
function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
}

/** Whether a confirm is up (one on its way out no longer counts). */
export function isConfirmUp() {
  return active !== null;
}

/**
 * Build and show one confirm. The caller's popover stays open beneath it.
 *
 * A spec with a `skipKeyBase` renders the "don't ask again" checkbox above
 * the actions, pre-ticked when the waiver is already recorded (the Shift
 * re-show is where it can be unticked), and the waiver is written the moment
 * the operator confirms — before the POST, because the checkbox answers "ask
 * me next time?", not "did this request succeed?".
 * @param {{title: string,
 *          body: (import('./control-target-facts.js').ConfirmRun)[][],
 *          live: string|null, confirmLabel: string,
 *          skipKeyBase?: string|null,
 *          onConfirm: (ui: ConfirmUi) => void,
 *          onDismiss?: () => void}} spec
 */
export function showConfirm(spec) {
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
      paragraph.append(typeof run === 'string' ? run : el('strong', undefined, run.em));
    }
    body.append(paragraph);
  }
  if (spec.live) body.append(el('div', 'posture-modal-live', spec.live));

  const error = el('div', 'posture-modal-error');
  error.setAttribute('role', 'alert');
  error.hidden = true;

  /** @type {HTMLInputElement|null} */
  let skipBox = null;
  /** @type {HTMLElement|null} */
  let skipRow = null;
  if (spec.skipKeyBase) {
    skipRow = el('label', 'posture-modal-skip');
    skipBox = /** @type {HTMLInputElement} */ (document.createElement('input'));
    skipBox.type = 'checkbox';
    skipBox.checked = confirmSkipped(spec.skipKeyBase);
    skipRow.append(skipBox, el('span', undefined, "Don't ask again for this machine"));
  }

  const actions = el('div', 'posture-modal-actions');
  const cancel = /** @type {HTMLButtonElement} */ (el('button', 'posture-modal-cancel', 'Cancel'));
  cancel.type = 'button';
  const confirm = /** @type {HTMLButtonElement} */ (
    el('button', 'posture-modal-confirm', spec.confirmLabel)
  );
  confirm.type = 'button';
  if (spec.live) confirm.dataset.live = 'true';
  actions.append(cancel, confirm);

  dialog.append(heading, body, error);
  if (skipRow) dialog.append(skipRow);
  dialog.append(actions);
  overlay.append(dialog);
  active = { overlay, onDismiss: spec.onDismiss };
  mountOverlay(overlay);

  cancel.addEventListener('click', () => dismissConfirm());
  confirm.addEventListener('click', () => {
    if (spec.skipKeyBase && skipBox) rememberConfirmSkip(spec.skipKeyBase, skipBox.checked);
    spec.onConfirm({ error, confirm, cancel, done: dismissConfirm });
  });
  confirm.focus();
}

/**
 * Take the confirm off the screen, if one is up. Idempotent, and the ONE
 * chokepoint: cancel, Escape, a confirmed gesture's done(), the popover
 * closing over it — every path lands here, so the raiser's `onDismiss` (the
 * popover un-parking a switch knob) runs exactly once per dialog.
 */
export function dismissConfirm() {
  const current = active;
  active = null;
  if (!current) return;
  // `data-closing` is what tells "still up" from "on its way out" — to a
  // reader, to a test, and to the stale sweep in showConfirm.
  current.overlay.dataset.closing = '1';
  fadeOutOverlay(current.overlay);
  current.onDismiss?.();
}
