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
  switch: 'asked',
  prompt: true,
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
  any_armed: true,
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
    confirmingMove: null,
    gestureNotes: new Map(),
    onMove: vi.fn(),
    onSwitchTo: vi.fn(),
    onAllOff: vi.fn(),
  };
});

describe('renderRow', () => {
  test('is a design-system popover row with a dot, a writes control and a primary Switch to', () => {
    const node = renderRow(rowOf(), stateOf(), ctx);

    expect(node.classList.contains('ctc-row')).toBe(true);
    expect(node.classList.contains('osprey-popover-item')).toBe(true);
    // A plain row: it holds buttons of its own, so it is no menu item.
    expect(node.hasAttribute('role')).toBe(false);
    expect(node.dataset.target).toBe('va');
    expect(node.dataset.targetKind).toBe('simulator');
    expect(node.dataset.state).toBe('asked');
    expect(node.querySelector('.ctc-name-line > .ctc-machine-glyph')).not.toBeNull();
    const group = /** @type {HTMLElement} */ (node.querySelector('.ctc-positions'));
    expect(group.getAttribute('role')).toBe('radiogroup');
    const checked = [...group.querySelectorAll('[role="radio"][aria-checked="true"]')];
    expect(checked.map((n) => /** @type {HTMLElement} */ (n).dataset.position)).toEqual(['asked']);
    const swap = /** @type {HTMLElement} */ (node.querySelector('.ctc-switch'));
    expect(swap.classList.contains('osprey-btn')).toBe(true);
    expect(swap.classList.contains('primary')).toBe(true);
  });

  test('a committed position runs onMove with that position, and Shift as reshow', () => {
    const armed = rowOf();
    /** @type {HTMLElement} */ (
      renderRow(armed, stateOf(), ctx).querySelector('[data-position="off"]')
    ).click();
    expect(ctx.onMove).toHaveBeenCalledWith(armed, 'off', expect.anything(), false);

    const off = rowOf({ switch: 'off', write_allowed: false, write_reason: 'switch_off' });
    const option = /** @type {HTMLElement} */ (
      renderRow(off, stateOf(), ctx).querySelector('[data-position="asked"]')
    );
    option.dispatchEvent(new MouseEvent('click', { bubbles: true, shiftKey: true }));
    expect(ctx.onMove).toHaveBeenCalledWith(off, 'asked', expect.anything(), true);
  });

  test('Switch to runs onSwitchTo', () => {
    const row = rowOf();
    /** @type {HTMLElement} */ (
      renderRow(row, stateOf(), ctx).querySelector('.ctc-switch')
    ).click();
    expect(ctx.onSwitchTo).toHaveBeenCalledWith(row, expect.anything(), false);
  });

  test('the row whose confirm is up marks the position it asked for', () => {
    ctx.confirmingMove = { target: 'va', position: 'asked' };
    const node = renderRow(rowOf({ switch: 'off', write_allowed: false }), stateOf(), ctx);
    const pending = [...node.querySelectorAll('[data-pending="true"]')];
    expect(pending.map((n) => /** @type {HTMLElement} */ (n).dataset.position)).toEqual(['asked']);
    // Nothing has been applied: the stored position stays checked.
    expect(
      /** @type {HTMLElement} */ (node.querySelector('[aria-checked="true"]')).dataset.position
    ).toBe('off');
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
  test('the machine the agent stands on: dot, name and writes control, no eyebrow', () => {
    const card = renderCard(rowOf({ active: true }), stateOf(), ctx);

    expect(card.classList.contains('ctc-card')).toBe(true);
    expect(card.querySelector('.ctc-card-title > .ctc-machine-glyph')).not.toBeNull();
    expect(card.querySelector('.ctc-name')?.textContent).toBe('Simulator');
    expect(card.querySelector('.ctc-switchwrap [role="radiogroup"]')).not.toBeNull();
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

  test('is disabled when no switch of this login is armed', () => {
    const all = renderFoot(stateOf({ any_armed: false }), ctx).querySelector('.ctc-all-off');
    expect(/** @type {HTMLButtonElement} */ (all).disabled).toBe(true);
  });
});
