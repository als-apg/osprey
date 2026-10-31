// @ts-check
/**
 * Unit tests for the control-target popover's card, rows and foot
 * (control-target-rows.js):
 *   npx vitest run tests/interfaces/web_terminal/control-target-rows.test.mjs
 *
 * The module builds DOM from one roster payload and a RowContext; it holds no
 * state and sends nothing. These tests hand it a context of spies and check
 * the design-system primitives each part is built on, the facts the context
 * carries, and which context callback each control runs. The gestures behind
 * the callbacks are pinned by control-target-popover.test.mjs.
 */

import { test, expect, describe, beforeEach, vi } from 'vitest';

import {
  OUTCOME_MAX_AGE_S,
  renderCard,
  renderFoot,
  renderRow,
} from '../../../src/osprey/interfaces/web_terminal/static/js/control-target-rows.js';

/** A writable, reachable machine row, as GET /api/terminal/posture publishes it. */
const rowOf = (/** @type {object} */ o = {}) => ({
  target: 'va',
  label: 'Simulator — nothing moves',
  short_label: 'VIRTUAL',
  machine_kind: 'simulator',
  display_name: '',
  endpoint: '127.0.0.1:10064',
  real_machine: false,
  switch: 'on',
  write_allowed: true,
  write_reason: null,
  write_phrase: null,
  active: false,
  is_baseline: false,
  available_now: true,
  reason: null,
  reason_detail: null,
  reachability: { state: 'reached', role: 'write_access', age_s: 3, role_detail: {} },
  ...o,
});

/** A roster whose context this terminal owns, so Switch to is offered. */
const stateOf = (/** @type {object} */ o = {}) => ({
  session_id: null,
  control_target: 'standin',
  generation: 3,
  owner: { kind: 'web_terminal', pid: 1000, port: 8080, self: true },
  store_available: true,
  readonly_run: false,
  any_on: true,
  locks: { readonly_run: false, tier: 'readwrite' },
  last_switch: null,
  targets: [],
  ...o,
});

/** @type {import('../../../src/osprey/interfaces/web_terminal/static/js/control-target-rows.js').RowContext} */
let ctx;

beforeEach(() => {
  ctx = {
    pendingTarget: null,
    confirmingTarget: null,
    gestureNotes: new Map(),
    onTurnOff: vi.fn(),
    onTurnOn: vi.fn(),
    onSwitchTo: vi.fn(),
    onAllOff: vi.fn(),
  };
});

describe('renderRow', () => {
  test('is a design-system popover row with a dot, a switch and a primary Switch to', () => {
    const node = renderRow(rowOf(), stateOf(), ctx);

    expect(node.classList.contains('ctc-row')).toBe(true);
    expect(node.classList.contains('osprey-popover-item')).toBe(true);
    // A plain row: it holds buttons of its own, so it is no menu item.
    expect(node.hasAttribute('role')).toBe(false);
    expect(node.dataset.target).toBe('va');
    expect(node.dataset.targetKind).toBe('simulator');
    expect(node.dataset.state).toBe('on');
    expect(node.querySelector('.ctc-name-line > .osprey-dot.ctc-dot')).not.toBeNull();
    const toggle = /** @type {HTMLElement} */ (node.querySelector('.ctc-toggle'));
    expect(toggle.getAttribute('role')).toBe('switch');
    expect(toggle.getAttribute('aria-checked')).toBe('true');
    expect(toggle.querySelector(':scope > .osprey-switch')).not.toBeNull();
    const swap = /** @type {HTMLElement} */ (node.querySelector('.ctc-switch'));
    expect(swap.classList.contains('osprey-btn')).toBe(true);
    expect(swap.classList.contains('primary')).toBe(true);
  });

  test('a switch that is on runs onTurnOff; one that is off runs onTurnOn with Shift', () => {
    const on = rowOf();
    /** @type {HTMLElement} */ (renderRow(on, stateOf(), ctx).querySelector('.ctc-toggle')).click();
    expect(ctx.onTurnOff).toHaveBeenCalledWith(on, expect.anything());

    const off = rowOf({ switch: 'off', write_allowed: false, write_reason: 'switch_off' });
    const toggle = /** @type {HTMLElement} */ (
      renderRow(off, stateOf(), ctx).querySelector('.ctc-toggle')
    );
    toggle.dispatchEvent(new MouseEvent('click', { bubbles: true, shiftKey: true }));
    expect(ctx.onTurnOn).toHaveBeenCalledWith(off, expect.anything(), true);
  });

  test('Switch to runs onSwitchTo', () => {
    const row = rowOf();
    /** @type {HTMLElement} */ (
      renderRow(row, stateOf(), ctx).querySelector('.ctc-switch')
    ).click();
    expect(ctx.onSwitchTo).toHaveBeenCalledWith(row, expect.anything(), false);
  });

  test('the row whose confirm is up parks its switch', () => {
    ctx.confirmingTarget = 'va';
    const node = renderRow(rowOf({ switch: 'off', write_allowed: false }), stateOf(), ctx);
    expect(/** @type {HTMLElement} */ (node.querySelector('.ctc-toggle')).dataset.pending).toBe(
      'on'
    );
  });

  test("the last click's refusal is shown on its row", () => {
    ctx.gestureNotes.set('va', 'No controls server yet.');
    const node = renderRow(rowOf(), stateOf(), ctx);
    const lines = [...node.querySelectorAll('.ctc-outcome')].map((n) => n.textContent);
    expect(lines).toContain('✗ No controls server yet.');
  });

  test('an outcome older than the freshness window is not shown', () => {
    const last = { target: 'va', status: 'applied', age_s: OUTCOME_MAX_AGE_S + 1 };
    const node = renderRow(rowOf(), stateOf({ last_switch: last }), ctx);
    expect(node.querySelector('.ctc-outcome')).toBeNull();

    const fresh = renderRow(rowOf(), stateOf({ last_switch: { ...last, age_s: 1 } }), ctx);
    expect(fresh.querySelector('.ctc-outcome')?.textContent).toBe('✓ switched');
  });
});

describe('renderCard', () => {
  test('the machine the agent stands on: dot, name and switch, no eyebrow', () => {
    const card = renderCard(rowOf({ active: true }), stateOf(), ctx);

    expect(card.classList.contains('ctc-card')).toBe(true);
    expect(card.querySelector('.ctc-card-title > .osprey-dot.ctc-dot')).not.toBeNull();
    expect(card.querySelector('.ctc-name')?.textContent).toBe('Simulator');
    expect(card.querySelector('.ctc-toggle > .osprey-switch')).not.toBeNull();
    expect(card.querySelector('.ctc-card-eyebrow')).toBeNull();
  });
});

describe('renderFoot', () => {
  test('Turn all writes off is a design-system button that runs onAllOff', () => {
    const state = stateOf();
    const foot = renderFoot(state, ctx);
    const all = /** @type {HTMLButtonElement} */ (foot.querySelector('.ctc-all-off'));

    expect(all.classList.contains('osprey-btn')).toBe(true);
    all.click();
    expect(ctx.onAllOff).toHaveBeenCalledWith(state);
  });

  test('is disabled when no switch of this login is on', () => {
    const all = renderFoot(stateOf({ any_on: false }), ctx).querySelector('.ctc-all-off');
    expect(/** @type {HTMLButtonElement} */ (all).disabled).toBe(true);
  });
});
