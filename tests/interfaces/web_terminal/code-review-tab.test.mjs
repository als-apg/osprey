// @ts-check
/**
 * The Settings drawer's Review tab (code-review-tab.js):
 *   npx vitest run tests/interfaces/web_terminal/code-review-tab.test.mjs
 *
 * The tab renders `GET /api/code-review` as it is: on, off or unreadable, the
 * server's warning verbatim, the rules file's text, and a locked toggle.
 */

import { test, expect, beforeEach, afterEach, vi } from 'vitest';

import {
  _resetCodeReviewTabForTest,
  initCodeReviewTab,
  refreshCodeReviewTab,
  reviewPaletteEntries,
} from '../../../src/osprey/interfaces/web_terminal/static/js/code-review-tab.js';

const ON = {
  enabled: true,
  level: 'block',
  time_limit: 60,
  prompt: 'data/code_review/reviewer.md',
  rules: '## Rules\n\n- Read SR:BEAM:USER_MODE first.\n',
  warning: null,
  can_change: false,
};

/** @type {any} */
let payload;

beforeEach(() => {
  document.body.innerHTML = `
    <div id="settings-drawer">
      <div class="drawer-tabs">
        <button class="drawer-tab" data-tab="tab-safety">Safety</button>
        <button class="drawer-tab" data-tab="tab-review">Review</button>
      </div>
      <div id="tab-review"><div id="code-review-section"></div></div>
    </div>`;
  payload = { ...ON };
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => ({ ok: true, status: 200, statusText: 'OK', json: async () => payload }))
  );
  _resetCodeReviewTabForTest();
  initCodeReviewTab();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

/** @param {string} selector */
const text = (selector) => document.querySelector(selector)?.textContent ?? '';

test('opening the tab reads the section and shows it', async () => {
  document.getElementById('tab-review')?.dispatchEvent(new CustomEvent('drawer:tab-activate'));
  await vi.waitFor(() => expect(text('.code-review-state')).toBe('On'));
  expect(text('.code-review-value')).toBe('60 s');
  expect(text('.code-review-path')).toBe('data/code_review/reviewer.md');
  expect(text('.code-review-rules-heading')).toBe('## Rules');
  expect(text('.code-review-rules')).toContain('- Read SR:BEAM:USER_MODE first.');
  expect(document.querySelector('.code-review-warning')).toBeNull();
});

test('off is shown as Off, the tab still present', async () => {
  payload = { ...ON, enabled: false };
  await refreshCodeReviewTab();
  expect(text('.code-review-state')).toBe('Off');
  const toggle = /** @type {HTMLInputElement} */ (document.querySelector('.toggle-switch input'));
  expect(toggle.checked).toBe(false);
});

test('a non-admin sees the locked toggle, a read-only badge and the ask-your-admin line', async () => {
  await refreshCodeReviewTab();
  const toggle = /** @type {HTMLInputElement} */ (document.querySelector('.toggle-switch input'));
  expect(toggle.disabled).toBe(true);
  expect(toggle.checked).toBe(true);
  expect(text('.code-review-lock')).toContain('Set by your deployment admin');
  expect(text('.code-review-section-head .read-only')).toBe('READ-ONLY');
  expect(text('.code-review-foot')).toBe('Ask your deployment admin to change these rules.');
});

test('the admin tier is told where the section is changed, not offered a dead control', async () => {
  payload = { ...ON, can_change: true };
  await refreshCodeReviewTab();
  const toggle = /** @type {HTMLInputElement} */ (document.querySelector('.toggle-switch input'));
  expect(toggle.disabled).toBe(true);
  expect(text('.code-review-lock')).toContain('Set in the profile');
  expect(text('.code-review-foot')).toContain('Edit the rules file in the profile');
});

test("the server's warning leads the tab, verbatim, as text", async () => {
  const warning =
    "The rules file cannot be used (<b>gone</b>), so every review fails and the agent's Python and notebook code is blocked; restore x.";
  payload = { ...ON, rules: null, warning };
  await refreshCodeReviewTab();
  const notice = document.querySelector('#code-review-section > :first-child');
  expect(notice?.classList.contains('code-review-warning')).toBe(true);
  expect(notice?.textContent).toBe(warning);
  expect(notice?.querySelector('b')).toBeNull();
  expect(text('.code-review-rules')).toBe('The rules file cannot be used.');
});

test('an unreadable section says so instead of On or Off', async () => {
  payload = { ...ON, enabled: null, time_limit: null, prompt: null, rules: null, warning: 'x' };
  await refreshCodeReviewTab();
  expect(text('.code-review-state')).toBe('Cannot be read');
  expect(text('.code-review-value')).toBe('—');
  expect(document.querySelector('.code-review-path')).toBeNull();
});

test('a failed read leaves a line, not a blank tab', async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => ({ ok: false, status: 500, statusText: 'x', json: async () => ({}) }))
  );
  await refreshCodeReviewTab();
  expect(text('#code-review-section')).toBe('Code review settings unavailable.');
});

test('the palette row for the rules opens the Review tab', () => {
  expect(reviewPaletteEntries()).toEqual([
    {
      name: 'code-review/rules',
      label: 'Code review rules',
      category: '',
      tabId: 'tab-review',
      tabLabel: 'Review',
      description: "The rules the code reviewer judges the agent's code against",
    },
  ]);
  document.querySelector('[data-tab="tab-review"]')?.remove();
  expect(reviewPaletteEntries()).toEqual([]);
});

test('a rule and the indented lines that continue it are one span', async () => {
  const { rulesSpans } =
    await import('../../../src/osprey/interfaces/web_terminal/static/js/code-review-tab.js');
  expect(rulesSpans('## Rules\n\n- writes a corrector\n  must read USER_MODE\nTail\n')).toEqual([
    ['code-review-rules-heading', '## Rules'],
    ['', ''],
    ['code-review-rule', '- writes a corrector\n  must read USER_MODE'],
    ['', 'Tail'],
  ]);
});
