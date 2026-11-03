// @ts-check
/**
 * The Settings drawer's Review tab (code-review-tab.js):
 *   npx vitest run tests/interfaces/web_terminal/code-review-tab.test.mjs
 *
 * The tab renders `GET /api/code-review` as it is: on, off or unreadable, the
 * server's warning verbatim, the rules file's text, and either code review's
 * runtime safety switch (the Safety tab's own row) or a locked toggle.
 */

import { test, expect, beforeEach, afterEach, vi } from 'vitest';

import {
  _resetCodeReviewTabForTest,
  initCodeReviewTab,
  refreshCodeReviewTab,
  reviewPaletteEntries,
} from '../../../src/osprey/interfaces/web_terminal/static/js/code-review-tab.js';
import { dismissConfirm } from '../../../src/osprey/interfaces/web_terminal/static/js/posture-confirm.js';

const ON = {
  enabled: true,
  level: 'block',
  time_limit: 60,
  prompt: 'data/code_review/reviewer.md',
  rules: '## Rules\n\n- Read SR:BEAM:USER_MODE first.\n',
  warning: null,
  can_change: false,
  switch: null,
};

const SWITCH = {
  name: 'code_review',
  label: 'Code review',
  note: "Meanwhile the agent's Python and notebook code runs unreviewed.",
  on: true,
  off_by: null,
  off_at: null,
};

/** @type {any} */
let payload;
/** @type {Array<{url: string, init: any}>} */
let posts = [];

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
    vi.fn(async (/** @type {string} */ url, /** @type {any} */ init) => {
      if (init?.method === 'POST') posts.push({ url, init });
      return { ok: true, status: 200, statusText: 'OK', json: async () => payload };
    })
  );
  posts = [];
  _resetCodeReviewTabForTest();
  initCodeReviewTab();
});

afterEach(() => {
  dismissConfirm();
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
  expect(text('.safety-switch-lock')).toContain('Set by your deployment admin');
  expect(text('.code-review-section-head .read-only')).toBe('READ-ONLY');
  expect(text('.code-review-foot')).toBe('Ask your deployment admin to change these rules.');
});

test('the admin tier is told where the section is changed, not offered a dead control', async () => {
  payload = { ...ON, can_change: true };
  await refreshCodeReviewTab();
  const toggle = /** @type {HTMLInputElement} */ (document.querySelector('.toggle-switch input'));
  expect(toggle.disabled).toBe(true);
  expect(text('.safety-switch-lock')).toContain('Set in the profile');
  expect(text('.code-review-foot')).toContain('Edit the rules file in the profile');
});

test('with the switch registered, an admin moves it with the Safety tab row', async () => {
  payload = { ...ON, can_change: true, switch: SWITCH };
  await refreshCodeReviewTab();
  const toggle = /** @type {HTMLInputElement} */ (
    document.querySelector('[data-switch="code_review"] input')
  );
  expect(toggle.disabled).toBe(false);
  expect(toggle.checked).toBe(true);
  expect(text('.safety-switch-state')).toBe('On, as the profile sets it.');
  expect(text('.safety-switch-lock')).toBe('');
  expect(text('.code-review-foot')).toBe(
    'Turning review off or back on is recorded in the audit trail with your name. ' +
      'Edit the rules file in the profile; the next review reads it.'
  );

  toggle.checked = false;
  toggle.dispatchEvent(new Event('change'));
  const dialog = document.querySelector('.posture-modal');
  expect(dialog?.textContent).toContain('Turn Code review off?');
  expect(dialog?.textContent).toContain('runs unreviewed');
  const confirm = dialog?.querySelector('.posture-modal-confirm');
  if (!(confirm instanceof HTMLButtonElement)) throw new Error('no confirm button');
  confirm.click();
  await vi.waitFor(() => expect(posts).toHaveLength(1));
  expect(posts[0].url).toContain('/api/safety-switches/code_review');
  expect(JSON.parse(posts[0].init.body)).toEqual({ state: 'off' });
});

test('with the switch off, every tier reads who turned it off; a non-admin cannot move it', async () => {
  payload = {
    ...ON,
    switch: { ...SWITCH, on: false, off_by: 'alice', off_at: 1_700_000_000 },
  };
  await refreshCodeReviewTab();
  const toggle = /** @type {HTMLInputElement} */ (
    document.querySelector('[data-switch="code_review"] input')
  );
  expect(toggle.disabled).toBe(true);
  expect(toggle.checked).toBe(false);
  expect(text('.safety-switch-state')).toContain('Code review is off in this terminal by alice');
  expect(text('.safety-switch-lock')).toContain('Set by your deployment admin');
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
