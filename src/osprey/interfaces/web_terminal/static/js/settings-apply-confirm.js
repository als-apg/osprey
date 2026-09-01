// @ts-check
/* OSPREY Web Terminal — Apply-Settings Confirm Gate
 *
 * The "Apply Settings?" dialog between the Apply button and applySettings(),
 * split out of settings.js: that module owns loading, rendering and writing
 * the config; this one owns the gate — show the dialog, run the apply on
 * confirm, and the "don't ask again" waiver. A waived Apply runs directly;
 * Shift-clicking Apply re-shows the dialog (pre-ticked, so unticking there is
 * the undo). The waiver is recorded only when the operator confirms — a
 * cancel with the box ticked must not mute the dialog.
 *
 * The overlay markup lives in index.html inside the agent panel; everything
 * here queries within the panel handed to init.
 */

import { confirmSkipped, rememberConfirmSkip } from './confirm-skip.js';

/** Per persona via confirm-skip.js's scopedStorageKey resolution. */
const SKIP_KEY_BASE = 'osprey-settings-apply-skip-confirm';

/** @param {HTMLElement} panel */
function overlay(panel) {
  return panel.querySelector('.settings-confirm-overlay');
}

/** @param {HTMLElement} panel @returns {HTMLInputElement|null} */
function skipBox(panel) {
  const box = panel.querySelector('.settings-confirm-skip-box');
  return box instanceof HTMLInputElement ? box : null;
}

/** Hide the dialog, if up. applySettings() runs this before it saves. @param {HTMLElement} panel */
export function hideApplyConfirm(panel) {
  overlay(panel)?.classList.remove('visible');
}

/**
 * Wire the Apply button, and the dialog's Confirm/Cancel, to the gate.
 * @param {HTMLElement} panel  the agent tab panel holding all four elements
 * @param {() => void} onApply  runs applySettings (which hides the dialog)
 */
export function initApplyConfirmGate(panel, onApply) {
  const applyBtn = panel.querySelector('.settings-apply-btn');
  if (applyBtn) applyBtn.addEventListener('click', (event) => {
    if (!(/** @type {MouseEvent} */ (event).shiftKey) && confirmSkipped(SKIP_KEY_BASE)) {
      onApply();
      return;
    }
    // The checkbox mirrors the recorded waiver every time the dialog opens.
    const box = skipBox(panel);
    if (box) box.checked = confirmSkipped(SKIP_KEY_BASE);
    overlay(panel)?.classList.add('visible');
  });

  const confirmBtn = panel.querySelector('.settings-confirm-btn');
  if (confirmBtn) confirmBtn.addEventListener('click', () => {
    rememberConfirmSkip(SKIP_KEY_BASE, skipBox(panel)?.checked ?? false);
    onApply();
  });

  const cancelBtn = panel.querySelector('.settings-cancel-btn');
  if (cancelBtn) cancelBtn.addEventListener('click', () => hideApplyConfirm(panel));
}
