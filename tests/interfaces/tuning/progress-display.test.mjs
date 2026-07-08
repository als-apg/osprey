// @ts-check
/**
 * Behavioral tests for the OSPREY Tuning progress-display module
 * (progress-display.js): the view layer that subscribes to the state
 * singleton and renders the run-status badge and the LHS/BO results table.
 *
 * Neither `updateResultsTable` nor `updateStatusBadge` is exported — both are
 * driven through the real subscription wiring. `initProgressDisplay()` binds
 * the module's `onStateChanged` handler to the store's
 * `optimizationStateChanged` event, so pushing an optimization state via
 * `store.setOptimizationState(...)` exercises the same fan-out the live panel
 * uses (badge, then plot, then table, then logs).
 *
 * The module imports the same module-level `state` singleton this suite drives.
 * To keep the shared singleton and the module's subscriptions independent per
 * test, each test clears `sessionStorage`, calls `vi.resetModules()`, and
 * re-imports BOTH modules from the fresh graph so `store` is the exact instance
 * `progress-display.js` subscribed to. Without the reset, every
 * `initProgressDisplay()` would stack another live `onStateChanged` listener on
 * the surviving singleton.
 *
 * DOM guard, happy-dom environment (configured globally):
 *   npx vitest run tests/interfaces/tuning/progress-display.test.mjs
 *
 * @module tests/interfaces/tuning/progress-display
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';

const STATE_PATH = '../../../src/osprey/interfaces/tuning/static/js/state.js';
const DISPLAY_PATH = '../../../src/osprey/interfaces/tuning/static/js/progress-display.js';

// plots.js reaches for the vendored `Plotly` global at runtime. This suite
// omits the `#optimization-plot` container (so `updatePlot` returns before it
// would ever touch Plotly), but a defensive no-op stub keeps the module inert
// regardless of code path.
/** @type {any} */ (globalThis).Plotly = {
  newPlot() {},
  react() {},
  on() {},
  removeAllListeners() {},
};

/** @type {typeof import('../../../src/osprey/interfaces/tuning/static/js/state.js').state} */
let store;

beforeEach(async () => {
  // Fresh module graph per test: drops listeners registered by a prior
  // `initProgressDisplay()` and rebuilds default state. Clear sessionStorage
  // first so the store constructor's `tuning_jobId` read starts clean.
  sessionStorage.clear();
  document.body.innerHTML = `
    <div id="run-status-badge"><span class="status-dot"></span><span class="status-text"></span></div>
    <table><tbody id="results-table-body"></tbody></table>
    <div id="results-pagination"></div>
    <textarea id="log-output"></textarea>
    <select id="display-mode"></select>
    <button id="apply-point-btn"></button>
  `;

  vi.resetModules();
  ({ state: store } = await import(STATE_PATH));
  /** @type {typeof import('../../../src/osprey/interfaces/tuning/static/js/progress-display.js')} */
  const display = await import(DISPLAY_PATH);
  display.initProgressDisplay();
});

/** @returns {HTMLElement} */
function tableBody() {
  return /** @type {HTMLElement} */ (document.getElementById('results-table-body'));
}

/** The three status-badge nodes: [dot, text].
 *  @returns {{ dot: HTMLElement, text: HTMLElement }} */
function badgeParts() {
  const badge = /** @type {HTMLElement} */ (document.getElementById('run-status-badge'));
  return {
    dot: /** @type {HTMLElement} */ (badge.querySelector('.status-dot')),
    text: /** @type {HTMLElement} */ (badge.querySelector('.status-text')),
  };
}

describe('updateResultsTable — row rendering', () => {
  it('renders one row per LHS/BO data point with sequential iteration numbers', () => {
    store.setOptimizationState({
      status: 'RUNNING',
      lhs_data: [{ objective_value: 1 }, { objective_value: 2 }],
      bo_data: [{ objective_value: 3 }],
    });

    const rows = tableBody().querySelectorAll('tr');
    expect(rows).toHaveLength(3);
    // First column is the 1-based iteration index, continuous across phases.
    const iters = Array.from(rows, (r) => r.querySelectorAll('td')[0].textContent);
    expect(iters).toEqual(['1', '2', '3']);
  });

  it('labels LHS rows then BO rows in the phase column', () => {
    store.setOptimizationState({
      status: 'RUNNING',
      lhs_data: [{ objective_value: 1 }],
      bo_data: [{ objective_value: 2 }],
    });

    const rows = tableBody().querySelectorAll('tr');
    const phases = Array.from(rows, (r) => r.querySelectorAll('td')[2].textContent);
    expect(phases).toEqual(['LHS', 'BO']);
  });

  it('renders a numeric objective_value formatted via toFixed(6)', () => {
    store.setOptimizationState({
      status: 'RUNNING',
      lhs_data: [{ objective_value: 1.5 }],
      bo_data: [],
    });

    const cell = tableBody().querySelector('tr')?.querySelectorAll('td')[1];
    expect(cell?.textContent).toBe('1.500000');
  });

  it('falls back to the escaped objective string when objective_value is non-numeric', () => {
    store.setOptimizationState({
      status: 'RUNNING',
      lhs_data: [{ objective: 'n/a', objective_value: null }],
      bo_data: [],
    });

    const cell = tableBody().querySelector('tr')?.querySelectorAll('td')[1];
    expect(cell?.textContent).toBe('n/a');
  });

  it('renders the placeholder dash when neither objective_value nor objective is present', () => {
    store.setOptimizationState({
      status: 'RUNNING',
      lhs_data: [{}],
      bo_data: [],
    });

    const cell = tableBody().querySelector('tr')?.querySelectorAll('td')[1];
    expect(cell?.textContent).toBe('--');
  });

  it('shows the empty-state row when there is no data', () => {
    store.setOptimizationState({ status: 'RUNNING', lhs_data: [], bo_data: [] });

    const body = tableBody();
    expect(body.querySelector('tr.empty-row')).not.toBeNull();
    expect(body.textContent).toContain('No results yet');
  });
});

describe('updateResultsTable — hostile input stays inert', () => {
  it('escapes a script/attribute-breakout objective so no live element is built', () => {
    const payload = '"><img src=x onerror=alert(1)><script>alert(1)</script>';
    store.setOptimizationState({
      status: 'RUNNING',
      lhs_data: [{ objective: payload, objective_value: null }],
      bo_data: [],
    });

    const body = tableBody();
    // Parsed DOM check: the payload survives only as inert text, never markup.
    expect(body.querySelector('img, script')).toBeNull();
    for (const el of body.querySelectorAll('*')) {
      for (const attr of el.attributes) {
        expect(attr.name.toLowerCase().startsWith('on')).toBe(false);
      }
    }
    // textContent decodes the escaped entities back to the literal payload.
    expect(body.textContent).toContain(payload);
  });
});

describe('updateStatusBadge — status transitions', () => {
  /** @type {Array<[string, string, string]>} status, expected dot class, expected label */
  const cases = [
    ['RUNNING', 'running', 'Running'],
    ['PAUSED', 'paused', 'Paused'],
    ['COMPLETED', 'live', 'Completed'],
    ['CANCELLED', 'warning', 'Cancelled'],
    ['ERROR', 'error', 'Error'],
  ];

  for (const [status, dotClass, label] of cases) {
    it(`sets the ${dotClass} dot and "${label}" text for ${status}`, () => {
      store.setOptimizationState({ status });

      const { dot, text } = badgeParts();
      expect(dot.classList.contains('status-dot')).toBe(true);
      expect(dot.classList.contains(dotClass)).toBe(true);
      expect(text.textContent).toBe(label);
    });
  }

  it('resets the previous state class on each transition', () => {
    store.setOptimizationState({ status: 'RUNNING' });
    expect(badgeParts().dot.classList.contains('running')).toBe(true);

    store.setOptimizationState({ status: 'PAUSED' });
    const { dot, text } = badgeParts();
    // The prior 'running' modifier is cleared; only the new modifier remains.
    expect(dot.classList.contains('running')).toBe(false);
    expect(dot.classList.contains('paused')).toBe(true);
    expect(text.textContent).toBe('Paused');
  });

  it('shows the Idle label for an unrecognized status without a modifier class', () => {
    store.setOptimizationState({ status: 'SOMETHING_ELSE' });

    const { dot, text } = badgeParts();
    expect(text.textContent).toBe('Idle');
    // Only the base class remains for the default branch.
    expect(dot.className).toBe('status-dot');
  });
});
