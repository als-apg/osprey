/**
 * Unit tests for design_system/static/js/components/segmented.js
 * (`.osprey-seg`).
 *
 *   npx vitest run tests/interfaces/design_system/js/components/segmented.test.mjs
 */

import { test, expect, describe, afterEach } from 'vitest';

import { wireSegmented, setSegmented } from '/design-system/js/components/segmented.js';

afterEach(() => {
  document.body.innerHTML = '';
});

function segMarkup() {
  document.body.innerHTML = `
    <div class="osprey-seg" role="group" aria-label="Run as">
      <button type="button" data-value="manual" aria-pressed="true">Manual</button>
      <button type="button" data-value="continuous" aria-pressed="false">Continuous</button>
      <button type="button" data-value="off" aria-pressed="false" disabled>Off</button>
    </div>`;
  return /** @type {HTMLElement} */ (document.querySelector('.osprey-seg'));
}

describe('segmented control', () => {
  test('a click presses that button alone and reports its value', () => {
    const seg = segMarkup();
    /** @type {string[]} */
    const seen = [];
    const dispose = wireSegmented(seg, (value) => seen.push(value));
    /** @type {HTMLButtonElement} */ (seg.querySelector('[data-value="continuous"]')).click();
    const pressed = [...seg.querySelectorAll('button[aria-pressed="true"]')].map(
      (b) => /** @type {HTMLElement} */ (b).dataset.value
    );
    expect(pressed).toEqual(['continuous']);
    expect(seen).toEqual(['continuous']);
    dispose();
  });

  test('clicking the pressed button reports nothing', () => {
    const seg = segMarkup();
    /** @type {string[]} */
    const seen = [];
    const dispose = wireSegmented(seg, (value) => seen.push(value));
    /** @type {HTMLButtonElement} */ (seg.querySelector('[data-value="manual"]')).click();
    expect(seen).toEqual([]);
    dispose();
  });

  test('a disabled button is never pressed', () => {
    const seg = segMarkup();
    /** @type {string[]} */
    const seen = [];
    const dispose = wireSegmented(seg, (value) => seen.push(value));
    const off = /** @type {HTMLButtonElement} */ (seg.querySelector('[data-value="off"]'));
    off.removeAttribute('disabled');
    off.setAttribute('aria-disabled', 'true');
    off.click();
    expect(off.getAttribute('aria-pressed')).toBe('false');
    expect(seen).toEqual([]);
    dispose();
  });

  test('arrow keys move the pressed button', () => {
    const seg = segMarkup();
    /** @type {string[]} */
    const seen = [];
    const dispose = wireSegmented(seg, (value) => seen.push(value));
    const first = /** @type {HTMLButtonElement} */ (seg.querySelector('[data-value="manual"]'));
    first.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true }));
    expect(seen).toEqual(['continuous']);
    const second = /** @type {HTMLButtonElement} */ (
      seg.querySelector('[data-value="continuous"]')
    );
    second.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft', bubbles: true }));
    expect(seen).toEqual(['continuous', 'manual']);
    dispose();
  });

  test('setSegmented presses a value without reporting it', () => {
    const seg = segMarkup();
    setSegmented(seg, 'continuous');
    expect(seg.querySelector('[aria-pressed="true"]')?.getAttribute('data-value')).toBe(
      'continuous'
    );
  });
});
