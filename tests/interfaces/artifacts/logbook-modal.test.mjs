// @ts-check
/**
 * Unit tests for the logbook composer's DOM builders (logbook-modal.js):
 *   npx vitest run tests/interfaces/artifacts/logbook-modal.test.mjs
 *
 * buildLogbookModal mounts the overlay from the static template and wires its
 * internal controls to the callbacks logbook.js passes in; buildSuccessCard
 * renders the two server strings of a created draft as text and a property.
 */

import { test, expect, describe, afterEach, vi } from 'vitest';

import {
  buildLogbookModal,
  buildSuccessCard,
} from '../../../src/osprey/interfaces/artifacts/static/js/logbook-modal.js';

afterEach(() => {
  document.body.innerHTML = '';
});

/** Build the modal with spy callbacks. */
function build() {
  const onClose = vi.fn();
  const onChooseArtifacts = vi.fn();
  const overlay = buildLogbookModal({ onClose, onChooseArtifacts });
  return { overlay, onClose, onChooseArtifacts };
}

/**
 * Select one artifact-scope radio the way a click does.
 * @param {HTMLElement} overlay
 * @param {string} value
 */
function chooseScope(overlay, value) {
  const radio = /** @type {HTMLInputElement} */ (
    overlay.querySelector(`input[name="logbook-artifact-scope"][value="${value}"]`)
  );
  radio.checked = true;
  radio.dispatchEvent(new Event('change'));
}

describe('buildLogbookModal', () => {
  test('mounts one hidden overlay holding the template', () => {
    const { overlay } = build();

    expect(document.querySelectorAll('.logbook-overlay')).toHaveLength(1);
    expect(overlay.parentNode).toBe(document.body);
    expect(overlay.style.display).toBe('none');
    expect(overlay.querySelector('.logbook-modal')).not.toBeNull();
  });

  test('the close button calls onClose', () => {
    const { overlay, onClose } = build();

    /** @type {HTMLElement} */ (overlay.querySelector('.logbook-modal-close')).click();

    expect(onClose).toHaveBeenCalledTimes(1);
  });

  test('a backdrop click calls onClose and a click inside the dialog does not', () => {
    const { overlay, onClose } = build();

    /** @type {HTMLElement} */ (overlay.querySelector('.logbook-modal')).click();
    expect(onClose).not.toHaveBeenCalled();

    overlay.click();
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  test('a detail button click leaves exactly that button active', () => {
    const { overlay } = build();
    const detailed = /** @type {HTMLElement} */ (
      overlay.querySelector('.logbook-detail-btn[data-level="detailed"]')
    );

    detailed.click();

    const active = overlay.querySelectorAll('.logbook-detail-btn.active');
    expect(active).toHaveLength(1);
    expect(active[0]).toBe(detailed);
  });

  test('choosing "choose" shows the picker and calls onChooseArtifacts; another scope hides it', () => {
    const { overlay, onChooseArtifacts } = build();
    const picker = /** @type {HTMLElement} */ (document.getElementById('logbook-artifact-picker'));

    chooseScope(overlay, 'choose');
    expect(picker.style.display).toBe('');
    expect(onChooseArtifacts).toHaveBeenCalledTimes(1);

    chooseScope(overlay, 'all');
    expect(picker.style.display).toBe('none');
    expect(onChooseArtifacts).toHaveBeenCalledTimes(1);
  });
});

describe('buildSuccessCard', () => {
  test('renders the draft id as text and the url as the link href', () => {
    const draftId = '<script>alert("x")</script>"\'';
    const url = 'https://ariel.example/draft?id="1"&x=<b>';

    const card = buildSuccessCard(draftId, url);

    expect(card.querySelector('script')).toBeNull();
    expect(card.textContent).toContain(draftId);
    const link = /** @type {HTMLAnchorElement} */ (card.querySelector('a'));
    expect(link.getAttribute('href')).toBe(url);
    expect(link.rel).toBe('noopener');
    expect(link.target).toBe('_blank');
  });
});
