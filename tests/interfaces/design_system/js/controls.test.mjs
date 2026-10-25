/**
 * Unit tests for the shared control behaviours:
 *   - design_system/static/js/components/disclosure.js  (step-row options,
 *     any `.disclosure-btn[aria-controls]`)
 *   - design_system/static/js/components/segmented.js   (`.seg`)
 *
 *   npx vitest run tests/interfaces/design_system/js/controls.test.mjs
 */

import { test, expect, describe, afterEach } from 'vitest';

import { wireDisclosures, setDisclosure } from '/design-system/js/components/disclosure.js';
import { wireSegmented, setSegmented } from '/design-system/js/components/segmented.js';

afterEach(() => {
  document.body.innerHTML = '';
});

function stepMarkup() {
  document.body.innerHTML = `
    <div class="step-list">
      <div class="step" id="s1">
        <div class="step-row">
          <span class="step-name"><span class="name">Inject</span></span>
          <button class="disclosure-btn btn-icon" aria-expanded="false" aria-controls="s1-opts"></button>
        </div>
        <div class="step-options" id="s1-opts" hidden></div>
      </div>
    </div>`;
  return {
    btn: /** @type {HTMLButtonElement} */ (document.querySelector('.disclosure-btn')),
    panel: /** @type {HTMLElement} */ (document.getElementById('s1-opts')),
    step: /** @type {HTMLElement} */ (document.getElementById('s1')),
  };
}

describe('disclosure', () => {
  test('a click opens the controlled panel and marks the step open', () => {
    const { btn, panel, step } = stepMarkup();
    const dispose = wireDisclosures(document.body);
    btn.click();
    expect(btn.getAttribute('aria-expanded')).toBe('true');
    expect(panel.hidden).toBe(false);
    expect(step.classList.contains('is-open')).toBe(true);
    btn.click();
    expect(btn.getAttribute('aria-expanded')).toBe('false');
    expect(panel.hidden).toBe(true);
    expect(step.classList.contains('is-open')).toBe(false);
    dispose();
  });

  test('a disclosure fires a toggle event with its new state', () => {
    const { btn } = stepMarkup();
    const dispose = wireDisclosures(document.body);
    /** @type {boolean[]} */
    const seen = [];
    btn.addEventListener('disclosure:toggle', (/** @type {any} */ e) => seen.push(e.detail.open));
    btn.click();
    btn.click();
    expect(seen).toEqual([true, false]);
    dispose();
  });

  test('dispose removes the handler', () => {
    const { btn, panel } = stepMarkup();
    wireDisclosures(document.body)();
    btn.click();
    expect(panel.hidden).toBe(true);
  });

  test('setDisclosure sets a state directly', () => {
    const { btn, panel, step } = stepMarkup();
    setDisclosure(btn, true);
    expect(panel.hidden).toBe(false);
    expect(step.classList.contains('is-open')).toBe(true);
    setDisclosure(btn, false);
    expect(panel.hidden).toBe(true);
  });

  test('a button with no controlled element only flips aria-expanded', () => {
    document.body.innerHTML = '<button class="disclosure-btn" aria-expanded="false"></button>';
    const btn = /** @type {HTMLButtonElement} */ (document.querySelector('button'));
    const dispose = wireDisclosures(document.body);
    btn.click();
    expect(btn.getAttribute('aria-expanded')).toBe('true');
    dispose();
  });
});

function segMarkup() {
  document.body.innerHTML = `
    <div class="seg" role="group" aria-label="Run as">
      <button type="button" data-value="manual" aria-pressed="true">Manual</button>
      <button type="button" data-value="continuous" aria-pressed="false">Continuous</button>
      <button type="button" data-value="off" aria-pressed="false" disabled>Off</button>
    </div>`;
  return /** @type {HTMLElement} */ (document.querySelector('.seg'));
}

describe('segmented control', () => {
  test('a click presses that button alone and reports its value', () => {
    const seg = segMarkup();
    /** @type {string[]} */
    const seen = [];
    const dispose = wireSegmented(seg, (value) => seen.push(value));
    /** @type {HTMLButtonElement} */ (seg.querySelector('[data-value="continuous"]')).click();
    const pressed = [...seg.querySelectorAll('button[aria-pressed="true"]')].map(
      (b) => /** @type {HTMLElement} */ (b).dataset.value,
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
    const second = /** @type {HTMLButtonElement} */ (seg.querySelector('[data-value="continuous"]'));
    second.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft', bubbles: true }));
    expect(seen).toEqual(['continuous', 'manual']);
    dispose();
  });

  test('setSegmented presses a value without reporting it', () => {
    const seg = segMarkup();
    setSegmented(seg, 'continuous');
    expect(seg.querySelector('[aria-pressed="true"]')?.getAttribute('data-value')).toBe('continuous');
  });
});
