// @ts-check
/**
 * Behavioral tests for the OSPREY Tuning centralized state store (state.js):
 * the event-emitter singleton that replaced Dash's dcc.Store components.
 *
 * Exercises the public surface it exports — the `state` singleton's
 * on/off/emit event fan-out, every setter's state-mutation-plus-notify
 * contract, the derived `pageState` recomputation, and `resetForNewRun`.
 *
 * The module exports a module-level singleton whose constructor also reads
 * `sessionStorage`. To keep tests independent (fresh listeners, fresh state,
 * no persisted job id), each test re-imports the module under
 * `vi.resetModules()` after clearing `sessionStorage` in `beforeEach`.
 *
 * Pure-logic guard, happy-dom environment (configured globally):
 *   npx vitest run tests/interfaces/tuning/state.test.mjs
 *
 * @module tests/interfaces/tuning/state
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';

const MODULE_PATH = '../../../src/osprey/interfaces/tuning/static/js/state.js';

/** @type {typeof import('../../../src/osprey/interfaces/tuning/static/js/state.js').state} */
let store;

beforeEach(async () => {
  // Fresh module instance per test: clears any listeners registered by a
  // prior test and rebuilds default field state. Clear sessionStorage first so
  // the constructor's `tuning_jobId` read starts from a clean slate.
  sessionStorage.clear();
  vi.resetModules();
  ({ state: store } = await import(MODULE_PATH));
});

describe('event emitter — on/emit fan-out', () => {
  it('invokes a subscribed listener with the emitted payload', () => {
    const listener = vi.fn();
    store.on('customEvent', listener);

    store.emit('customEvent', { value: 42 });

    expect(listener).toHaveBeenCalledTimes(1);
    expect(listener).toHaveBeenCalledWith({ value: 42 });
  });

  it('fans out a single emit to every subscribed listener', () => {
    const first = vi.fn();
    const second = vi.fn();
    const third = vi.fn();
    store.on('customEvent', first);
    store.on('customEvent', second);
    store.on('customEvent', third);

    store.emit('customEvent', 'payload');

    for (const listener of [first, second, third]) {
      expect(listener).toHaveBeenCalledTimes(1);
      expect(listener).toHaveBeenCalledWith('payload');
    }
  });

  it('does not invoke listeners subscribed to a different event', () => {
    const listener = vi.fn();
    store.on('eventA', listener);

    store.emit('eventB', null);

    expect(listener).not.toHaveBeenCalled();
  });

  it('emitting an event with no listeners is a no-op', () => {
    expect(() => store.emit('neverSubscribed', 1)).not.toThrow();
  });

  it('isolates a throwing listener so later listeners still fire', () => {
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {});
    const boom = vi.fn(() => {
      throw new Error('listener boom');
    });
    const survivor = vi.fn();
    store.on('customEvent', boom);
    store.on('customEvent', survivor);

    store.emit('customEvent', 'x');

    expect(boom).toHaveBeenCalledTimes(1);
    expect(survivor).toHaveBeenCalledTimes(1);
    expect(consoleError).toHaveBeenCalled();
    consoleError.mockRestore();
  });

  it('off() unsubscribes a listener so it no longer fires', () => {
    const listener = vi.fn();
    store.on('customEvent', listener);
    store.off('customEvent', listener);

    store.emit('customEvent', 'x');

    expect(listener).not.toHaveBeenCalled();
  });

  it('off() only removes the named listener, leaving others attached', () => {
    const kept = vi.fn();
    const removed = vi.fn();
    store.on('customEvent', kept);
    store.on('customEvent', removed);
    store.off('customEvent', removed);

    store.emit('customEvent', 'x');

    expect(removed).not.toHaveBeenCalled();
    expect(kept).toHaveBeenCalledTimes(1);
  });
});

describe('setters — mutate state and notify subscribers', () => {
  it('setJobId updates the accessor, notifies, and persists to sessionStorage', () => {
    const listener = vi.fn();
    store.on('jobChanged', listener);

    store.setJobId('job-123');

    expect(store.jobId).toBe('job-123');
    expect(listener).toHaveBeenCalledWith('job-123');
    expect(sessionStorage.getItem('tuning_jobId')).toBe('job-123');
  });

  it('setJobId(null) clears the accessor and removes the persisted id', () => {
    store.setJobId('job-123');
    store.setJobId(null);

    expect(store.jobId).toBeNull();
    expect(sessionStorage.getItem('tuning_jobId')).toBeNull();
  });

  it('setEnvironment updates accessors and emits env + details', () => {
    const listener = vi.fn();
    store.on('environmentChanged', listener);
    const details = { id: 'env-1' };

    store.setEnvironment('sim', details);

    expect(store.environment).toBe('sim');
    expect(store.environmentDetails).toBe(details);
    expect(listener).toHaveBeenCalledWith({ env: 'sim', details });
  });

  it('setOptimizationState merges a partial update and emits the merged state', () => {
    const listener = vi.fn();
    store.on('optimizationStateChanged', listener);

    store.setOptimizationState({ status: 'RUNNING', current_iteration: 3 });

    expect(store.optimizationState.status).toBe('RUNNING');
    expect(store.optimizationState.current_iteration).toBe(3);
    // Untouched fields retain their defaults after the merge.
    expect(store.optimizationState.total_iterations).toBe(0);
    expect(listener).toHaveBeenCalledWith(store.optimizationState);
  });

  it('setSelectedPoint updates the accessor and emits the point', () => {
    const listener = vi.fn();
    store.on('selectedPointChanged', listener);
    const point = { x: 1, y: 2 };

    store.setSelectedPoint(point);

    expect(store.selectedPoint).toBe(point);
    expect(listener).toHaveBeenCalledWith(point);
  });

  it('setDisplayMode updates the accessor and emits the mode', () => {
    const listener = vi.fn();
    store.on('displayModeChanged', listener);

    store.setDisplayMode('raw');

    expect(store.displayMode).toBe('raw');
    expect(listener).toHaveBeenCalledWith('raw');
  });

  it('setActiveTab updates the accessor and emits the tab id', () => {
    const listener = vi.fn();
    store.on('tabChanged', listener);

    store.setActiveTab('results-tab');

    expect(store.activeTab).toBe('results-tab');
    expect(listener).toHaveBeenCalledWith('results-tab');
  });
});

describe('variable table setters', () => {
  it('setVariableTableData stores a copy and emits the new array', () => {
    const listener = vi.fn();
    store.on('variableTableChanged', listener);
    const rows = [{ pv_name: 'A' }, { pv_name: 'B' }];

    store.setVariableTableData(rows);

    expect(store.variableTableData).toEqual(rows);
    // Stored value is a copy, not the caller's array reference.
    expect(store.variableTableData).not.toBe(rows);
    expect(listener).toHaveBeenCalledWith(store.variableTableData);
  });

  it('addVariable appends a row and notifies', () => {
    const listener = vi.fn();
    store.setVariableTableData([{ pv_name: 'A' }]);
    store.on('variableTableChanged', listener);

    store.addVariable({ pv_name: 'B' });

    expect(store.variableTableData.map((v) => v.pv_name)).toEqual(['A', 'B']);
    expect(listener).toHaveBeenCalledTimes(1);
  });

  it('removeVariable drops the matching row by pv_name', () => {
    store.setVariableTableData([{ pv_name: 'A' }, { pv_name: 'B' }]);

    store.removeVariable('A');

    expect(store.variableTableData.map((v) => v.pv_name)).toEqual(['B']);
  });

  it('updateVariable merges updates into the matching row and notifies', () => {
    const listener = vi.fn();
    store.setVariableTableData([{ pv_name: 'A', min: 0 }]);
    store.on('variableTableChanged', listener);

    store.updateVariable('A', { min: 5, max: 10 });

    expect(store.variableTableData[0]).toEqual({ pv_name: 'A', min: 5, max: 10 });
    expect(listener).toHaveBeenCalledTimes(1);
  });

  it('updateVariable on an unknown pv_name is a no-op that does not notify', () => {
    const listener = vi.fn();
    store.setVariableTableData([{ pv_name: 'A' }]);
    store.on('variableTableChanged', listener);

    store.updateVariable('missing', { min: 5 });

    expect(store.variableTableData).toEqual([{ pv_name: 'A' }]);
    expect(listener).not.toHaveBeenCalled();
  });
});

describe('derived pageState', () => {
  it('starts idle with no environment: cannot start or cancel', () => {
    expect(store.pageState.canStart).toBe(false);
    expect(store.pageState.canCancel).toBe(false);
    expect(store.pageState.formDisabled).toBe(false);
  });

  it('allows start once an environment is selected while idle', () => {
    store.setEnvironment('sim');

    expect(store.pageState.canStart).toBe(true);
  });

  it('recomputes and emits pageState when the run starts', () => {
    const listener = vi.fn();
    store.setEnvironment('sim');
    store.setJobId('job-1');
    store.on('pageStateChanged', listener);

    store.setOptimizationState({ status: 'RUNNING' });

    expect(store.pageState.canPause).toBe(true);
    expect(store.pageState.canStart).toBe(false);
    expect(store.pageState.formDisabled).toBe(true);
    expect(store.pageState.pollingEnabled).toBe(true);
    expect(listener).toHaveBeenCalledWith(store.pageState);
  });
});

describe('resetForNewRun', () => {
  it('clears job, optimization state, and selection, and emits reset events', () => {
    store.setJobId('job-1');
    store.setOptimizationState({ status: 'RUNNING', current_iteration: 7 });
    store.setSelectedPoint({ x: 1 });

    const optimizationListener = vi.fn();
    const selectedListener = vi.fn();
    const resetListener = vi.fn();
    store.on('optimizationStateChanged', optimizationListener);
    store.on('selectedPointChanged', selectedListener);
    store.on('resetForNewRun', resetListener);

    store.resetForNewRun();

    expect(store.jobId).toBeNull();
    expect(store.optimizationState.status).toBe('IDLE');
    expect(store.optimizationState.current_iteration).toBe(0);
    expect(store.selectedPoint).toBeNull();
    expect(optimizationListener).toHaveBeenCalledWith(store.optimizationState);
    expect(selectedListener).toHaveBeenCalledWith(null);
    expect(resetListener).toHaveBeenCalledWith(null);
  });
});

describe('test isolation — fresh singleton per test', () => {
  // Module-scoped so it persists ACROSS the two tests below. The first test
  // registers a listener on its store instance that bumps this counter; the
  // second test proves that listener did NOT survive onto the fresh singleton
  // the beforeEach re-import produced (if it had leaked, emitting the same
  // event would bump the counter again).
  let priorProbeCalls = 0;

  it('registers a probe listener on this test\'s store instance', () => {
    store.on('isolationProbe', () => { priorProbeCalls += 1; });
    store.emit('isolationProbe', 1);
    expect(priorProbeCalls).toBe(1); // fires within this test
  });

  it('does not observe listeners registered in a prior test', () => {
    const callsBefore = priorProbeCalls;
    const listener = vi.fn();
    store.on('isolationProbe', listener);

    store.emit('isolationProbe', 1);

    // Fresh listener fires exactly once...
    expect(listener).toHaveBeenCalledTimes(1);
    // ...and the prior test's listener is gone (counter unchanged), proving the
    // beforeEach resetModules() handed us a brand-new store with no carryover.
    expect(priorProbeCalls).toBe(callsBefore);
  });

  it('starts from default state with no persisted job id', () => {
    expect(store.jobId).toBeNull();
    expect(store.optimizationState.status).toBe('IDLE');
    expect(store.variableTableData).toEqual([]);
  });
});
