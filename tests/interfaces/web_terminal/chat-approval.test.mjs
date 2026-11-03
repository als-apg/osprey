// @ts-check
/**
 * Unit tests for the approval card (chat-approval.js):
 *   npx vitest run tests/interfaces/web_terminal/chat-approval.test.mjs
 *
 * The card names the machine a write lands on, from the frame's `target`
 * (C-FRAME), with the same `data-target-kind` the chip sets; the stylesheet
 * owns the colour, so these tests assert the attribute and the words only.
 */

import { test, expect, vi } from 'vitest';

import { createApprovalCard } from '../../../src/osprey/interfaces/web_terminal/static/js/chat-approval.js';

/**
 * A tool-kind permission request, as the operator session sends it.
 * @param {Record<string, unknown>} [extra]
 */
const ask = (extra = {}) => ({
  type: 'permission_request',
  request_id: 'req-1',
  kind: 'tool',
  tool_name: 'Write channel',
  tool_name_raw: 'mcp__controls__channel_write',
  tool_use_id: 'tu-1',
  input: { channel: 'SR:DCCT', value: 1 },
  reason: null,
  can_allow_session: false,
  ...extra,
});

const hooks = () => ({ answer: vi.fn(async () => ({})), onSettled: vi.fn() });

/** @param {HTMLElement} el */
const targetLine = (el) => /** @type {HTMLElement | null} */ (el.querySelector('.op-card-target'));

test.each([
  ['real', 'live', 'Live — writes move hardware'],
  ['rehearsal', 'standin', 'Rehearsal — nothing moves'],
  ['simulator', 'va', 'Simulator — nothing moves'],
])('a %s write names its machine and kind', (kind, key, words) => {
  const card = createApprovalCard(
    ask({ target: { key, machine_kind: kind, kind_words: words } }),
    hooks()
  );
  const line = targetLine(card.el);
  expect(line).not.toBeNull();
  expect(line?.dataset.targetKind).toBe(kind);
  expect(line?.textContent).toContain(key);
  expect(line?.textContent).toContain(words);
});

test('a call that writes nothing carries no target line', () => {
  const card = createApprovalCard(ask({ target: null }), hooks());
  expect(targetLine(card.el)).toBeNull();
  expect(card.el.querySelector('[data-target-kind]')).toBeNull();
});

test('a request without a target field carries no target line', () => {
  const card = createApprovalCard(ask(), hooks());
  expect(targetLine(card.el)).toBeNull();
});

test('an unknown kind reads as the real machine, words and all', () => {
  const card = createApprovalCard(
    ask({ target: { key: 'odd', machine_kind: 'bogus', kind_words: 'Rehearsal — nothing moves' } }),
    hooks()
  );
  const line = targetLine(card.el);
  expect(line?.dataset.targetKind).toBe('real');
  expect(line?.textContent).toContain('Live — writes move hardware');
  expect(line?.textContent).not.toContain('nothing moves');
});

test('the target line sits after the reason, before the summary', () => {
  const card = createApprovalCard(
    ask({
      reason: 'Needs a write',
      target: {
        key: 'standin',
        machine_kind: 'rehearsal',
        kind_words: 'Rehearsal — nothing moves',
      },
    }),
    hooks()
  );
  const body = /** @type {HTMLElement} */ (card.el.querySelector('.op-card-body'));
  const order = [...body.children].map((c) => c.className.split(' ')[0]);
  expect(order.indexOf('op-card-reason')).toBeLessThan(order.indexOf('op-card-target'));
  expect(order.indexOf('op-card-target')).toBeLessThan(order.indexOf('op-card-summary'));
});

test('a plan card carries no target line', () => {
  const card = createApprovalCard(
    ask({
      kind: 'plan',
      input: { plan: 'do it' },
      target: {
        key: 'live',
        machine_kind: 'real',
        kind_words: 'Live — writes move hardware',
      },
    }),
    hooks()
  );
  expect(targetLine(card.el)).toBeNull();
});

test('the allow-for-session control is unchanged by a target', () => {
  const target = {
    key: 'standin',
    machine_kind: 'rehearsal',
    kind_words: 'Rehearsal — nothing moves',
  };
  const labels = (/** @type {HTMLElement} */ el) =>
    [...el.querySelectorAll('.op-card-actions button')].map((b) => b.textContent);
  expect(labels(createApprovalCard(ask({ target, can_allow_session: true }), hooks()).el)).toEqual([
    'Deny',
    'Allow for this session',
    'Allow',
  ]);
  expect(labels(createApprovalCard(ask({ target }), hooks()).el)).toEqual(['Deny', 'Allow']);
});

test('allowing a rehearsal write posts the plain answer', async () => {
  const h = hooks();
  const card = createApprovalCard(
    ask({
      target: {
        key: 'standin',
        machine_kind: 'rehearsal',
        kind_words: 'Rehearsal — nothing moves',
      },
    }),
    h
  );
  /** @type {HTMLButtonElement} */ (card.el.querySelector('.op-btn-primary')).click();
  await vi.waitFor(() => expect(h.onSettled).toHaveBeenCalled());
  expect(h.answer).toHaveBeenCalledWith({ allow: true });
});
