/**
 * Unit tests for design_system/static/js/components/disclosure.js
 * (step-row options, any `.osprey-disclosure[aria-controls]`).
 *
 *   npx vitest run tests/interfaces/design_system/js/components/disclosure.test.mjs
 */

import { test, expect, describe, afterEach } from 'vitest';

import { wireDisclosures, setDisclosure } from '/design-system/js/components/disclosure.js';

afterEach(() => {
  document.body.innerHTML = '';
});

function stepMarkup() {
  document.body.innerHTML = `
    <div class="osprey-step-list">
      <div class="osprey-step" id="s1">
        <div class="osprey-step-row">
          <span class="osprey-step-name"><span class="osprey-step-title">Inject</span></span>
          <button class="osprey-disclosure icon" aria-expanded="false" aria-controls="s1-opts"></button>
        </div>
        <div class="osprey-step-options" id="s1-opts" hidden></div>
      </div>
    </div>`;
  return {
    btn: /** @type {HTMLButtonElement} */ (document.querySelector('.osprey-disclosure')),
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
    document.body.innerHTML = '<button class="osprey-disclosure" aria-expanded="false"></button>';
    const btn = /** @type {HTMLButtonElement} */ (document.querySelector('button'));
    const dispose = wireDisclosures(document.body);
    btn.click();
    expect(btn.getAttribute('aria-expanded')).toBe('true');
    dispose();
  });
});
