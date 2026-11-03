// @ts-check
/* OSPREY Logbook — modal DOM builders.
 *
 * Builds the logbook entry composer overlay from the static template and wires
 * its internal controls, and builds the success card shown after a submit.
 * State (the cached overlay, the model list, the artifact picker contents)
 * stays in logbook.js, which passes in the callbacks this module invokes.
 */

import { LOGBOOK_MODAL_HTML } from './logbook-template.js';

/**
 * Create the hidden `.logbook-overlay`, append it to the body and wire close
 * (button and backdrop), the detail toggle and the artifact-scope radios.
 *
 * @param {{ onClose: () => void, onChooseArtifacts: () => void }} handlers
 *   onClose runs on the close button or a backdrop click; onChooseArtifacts
 *   runs when the "choose" scope is selected, after the picker is shown.
 * @returns {HTMLElement} the overlay
 */
export function buildLogbookModal({ onClose, onChooseArtifacts }) {
  const overlay = document.createElement('div');
  overlay.className = 'logbook-overlay';
  overlay.style.display = 'none';
  overlay.innerHTML = LOGBOOK_MODAL_HTML;

  document.body.appendChild(overlay);

  // Close handlers
  /** @type {HTMLElement} */ (overlay.querySelector('.logbook-modal-close')).addEventListener(
    'click',
    onClose
  );
  overlay.addEventListener('click', function (e) {
    if (e.target === overlay) onClose();
  });

  // Detail toggle buttons
  overlay.querySelectorAll('.logbook-detail-btn').forEach(function (btn) {
    btn.addEventListener('click', function () {
      overlay.querySelectorAll('.logbook-detail-btn').forEach(function (b) {
        b.classList.remove('active');
      });
      btn.classList.add('active');
    });
  });

  // Artifact scope radio → show/hide picker
  /** @type {NodeListOf<HTMLInputElement>} */
  (overlay.querySelectorAll('input[name="logbook-artifact-scope"]')).forEach(function (radio) {
    radio.addEventListener('change', function () {
      const picker = document.getElementById('logbook-artifact-picker');
      if (!picker) return;
      if (radio.value === 'choose' && radio.checked) {
        picker.style.display = '';
        onChooseArtifacts();
      } else {
        picker.style.display = 'none';
      }
    });
  });

  return overlay;
}

/**
 * The "Draft created" card that replaces the form body on a successful submit.
 *
 * Built as DOM rather than an interpolated HTML string for the same reason the
 * artifact picker assigns its checkbox value as a property: the two server
 * strings reach text and an href, and property assignment bypasses HTML
 * parsing entirely, so neither can break out of its slot.
 *
 * @param {string} draftId
 * @param {string} url
 * @returns {HTMLElement}
 */
export function buildSuccessCard(draftId, url) {
  const card = document.createElement('div');
  card.style.cssText = 'text-align:center; padding:var(--art-space-6); color:var(--color-success);';

  const heading = document.createElement('div');
  heading.style.cssText = 'font-size:var(--art-text-xl); margin-bottom:var(--art-space-2);';
  heading.textContent = 'Draft created';

  const meta = document.createElement('div');
  meta.style.cssText = 'font-size:var(--art-text-sm); color:var(--text-secondary);';
  meta.appendChild(document.createTextNode(draftId));
  meta.appendChild(document.createElement('br'));

  const link = document.createElement('a');
  link.href = url;
  link.target = '_blank';
  link.rel = 'noopener';
  link.style.color = 'var(--color-accent-light)';
  link.textContent = 'Open in ARIEL';
  meta.appendChild(link);

  card.append(heading, meta);
  return card;
}
