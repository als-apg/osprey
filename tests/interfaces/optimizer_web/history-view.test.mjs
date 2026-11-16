/**
 * Tests for the table of stored runs (history-view.js): the rows it shows
 * under the runs list's filter, newest first, and the run a row opens.
 */

import { describe, test, expect, vi } from 'vitest';

import { envDetails } from './_fixtures.mjs';
import {
  historyRows,
  mountHistoryView,
} from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/history-view.js';
import { inspectorFrame } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/shell.js';
import {
  deriveScreen,
  initialState,
  reduce,
} from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/store.js';

/** @typedef {import('../../../src/osprey/interfaces/optimizer_web/panels/optimizer/store.js').StoreEvent} StoreEvent */

/** @param {string} timestamp @param {Record<string, unknown>} [overrides] */
const stored = (timestamp, overrides = {}) => ({
  run_id: `injector/${timestamp}`,
  environment: 'injector',
  timestamp,
  objective_name: 'charge',
  has_model: false,
  job_id: `job-${timestamp}`,
  ...overrides,
});

const RESULTS = [
  stored('2026-11-15_09-00-00', { objective_name: 'loss' }),
  stored('2026-11-16_14-12-00', { has_model: true }),
  stored('2026-11-16_10-34-00', { environment: 'other', objective_name: 'rms' }),
];

function context() {
  const stage = document.createElement('section');
  const inspector = document.createElement('aside');
  const setPrimary = vi.fn();
  const actions = { select: vi.fn(), setFilter: vi.fn(), newRun: vi.fn() };
  const ctx = /** @type {any} */ ({
    stage,
    inspector,
    actions,
    shell: {
      setPrimary,
      fillInspector: (/** @type {any} */ parts) =>
        inspector.replaceChildren(...inspectorFrame(parts, () => {})),
      setDrawer: vi.fn(),
      setDrawerTop: vi.fn(),
    },
  });
  return { ctx, stage, inspector, setPrimary, actions };
}

/** @param {StoreEvent[]} events */
const at = (events) => {
  /** @type {StoreEvent[]} */
  const all = [{ type: 'select', selection: { kind: 'history' } }, ...events];
  const state = all.reduce(reduce, initialState());
  return { state, screen: deriveScreen(state) };
};

/** @param {StoreEvent[]} events */
function show(events) {
  const c = context();
  const view = mountHistoryView(c.ctx);
  const { state, screen } = at(events);
  view.update(state, screen);
  return { ...c, view, state, screen };
}

/** @param {HTMLElement} stage */
const rowIds = (stage) =>
  Array.from(stage.querySelectorAll('tbody tr')).map(
    (r) => /** @type {HTMLElement} */ (r).dataset.run
  );

describe('the table of stored runs', () => {
  test('lists every stored run, newest first, with the names the panel has read', () => {
    const { stage, screen } = show([
      { type: 'results', list: RESULTS, error: null },
      { type: 'env/loaded', name: 'injector', details: envDetails(), presets: {}, error: null },
    ]);
    expect(screen.id).toBe('history');
    expect(rowIds(stage)).toEqual([
      'injector/2026-11-16_14-12-00',
      'injector/2026-11-16_10-34-00',
      'injector/2026-11-15_09-00-00',
    ]);
    const first = stage.querySelector('tbody tr');
    // The objective's display name, with its identifier second.
    expect(first?.textContent).toContain('Charge');
    expect(first?.textContent).toContain('charge');
  });

  test('shows only the runs the runs list filter lets through', () => {
    const { stage } = show([
      { type: 'results', list: RESULTS, error: null },
      { type: 'filter', text: 'rms' },
    ]);
    expect(rowIds(stage)).toEqual(['injector/2026-11-16_10-34-00']);
  });

  test('a filter that matches nothing says so and clears through the shared filter', () => {
    const { stage, actions } = show([
      { type: 'results', list: RESULTS, error: null },
      { type: 'filter', text: 'nothing like this' },
    ]);
    expect(rowIds(stage)).toEqual([]);
    expect(stage.textContent).toContain('No past run matches the filter');
    /** @type {HTMLElement} */ (stage.querySelector('#clear-filter-btn')).click();
    expect(actions.setFilter).toHaveBeenCalledWith('');
  });

  test('a click or Enter on a row opens that run', () => {
    const { stage, actions } = show([{ type: 'results', list: RESULTS, error: null }]);
    const rows = /** @type {HTMLElement[]} */ (Array.from(stage.querySelectorAll('tbody tr')));
    rows[1].click();
    expect(actions.select).toHaveBeenLastCalledWith({
      kind: 'run',
      runId: 'injector/2026-11-16_10-34-00',
    });
    rows[0].dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
    expect(actions.select).toHaveBeenLastCalledWith({
      kind: 'run',
      runId: 'injector/2026-11-16_14-12-00',
    });
    expect(rows[0].tabIndex).toBe(0);
  });

  test('shows no column the stored-run list does not supply', () => {
    const rows = historyRows(at([{ type: 'results', list: RESULTS, error: null }]).state, 0);
    expect(Object.keys(rows[0]).sort()).toEqual(
      ['clock', 'day', 'environment', 'job', 'model', 'objective', 'objectiveRaw', 'runId'].sort()
    );
  });

  test('says when the list is being read, failed, or empty', () => {
    expect(show([]).stage.textContent).toContain('Reading past runs');
    expect(show([{ type: 'results', list: null, error: 'boom' }]).stage.textContent).toContain(
      'boom'
    );
    expect(show([{ type: 'results', list: [], error: null }]).stage.textContent).toContain(
      'No stored run yet'
    );
  });

  test('offers a new run as the primary action, and a poll with nothing new redraws nothing', () => {
    const { stage, setPrimary, actions, view, state, screen } = show([
      { type: 'results', list: RESULTS, error: null },
    ]);
    setPrimary.mock.calls.at(-1)?.[0].onClick();
    expect(actions.newRun).toHaveBeenCalledTimes(1);
    const table = stage.querySelector('table');
    view.update({ ...state }, { ...screen });
    expect(stage.querySelector('table')).toBe(table);
  });
});
