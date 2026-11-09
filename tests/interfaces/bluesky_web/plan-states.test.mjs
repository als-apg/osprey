/**
 * Unit tests for plan-states.js: what the Plans view's two panes say when
 * they have no plan to show. Pure decisions, no DOM.
 *
 *   npx vitest run tests/interfaces/bluesky_web/plan-states.test.mjs
 */

import { describe, expect, test } from 'vitest';

import {
  planDetailState,
  planListState,
  queueReach,
} from '../../../src/osprey/interfaces/bluesky_web/panels/bluesky/plan-states.js';

describe('queueReach', () => {
  test('no frame yet, or a dropped stream, is connecting', () => {
    expect(queueReach(null)).toBe('connecting');
  });

  test('a summary the bridge could not read is unreachable', () => {
    expect(queueReach({ available: false, reason: 'manager_unreachable' })).toBe('unreachable');
    expect(queueReach({})).toBe('unreachable');
  });

  test('a readable summary is ok', () => {
    expect(queueReach({ available: true, manager_state: 'idle' })).toBe('ok');
  });
});

describe('planListState', () => {
  test('loading, until the first answer lands', () => {
    expect(planListState({ load: 'loading', total: 0, shown: 0 })).toEqual({
      icon: 'refresh',
      message: 'Loading plans…',
    });
  });

  test('a failed read and an empty catalog say the same: nothing to offer', () => {
    const none = { icon: 'file-text', message: 'No plans available.' };
    expect(planListState({ load: 'error', total: 0, shown: 0 })).toEqual(none);
    expect(planListState({ load: 'ready', total: 0, shown: 0 })).toEqual(none);
  });

  test('a filter that hides every plan says so, and is the operator’s to undo', () => {
    expect(planListState({ load: 'ready', total: 3, shown: 0 })).toEqual({
      icon: 'search',
      message: 'No plan matches the filter.',
    });
  });

  test('rows to show need no state', () => {
    expect(planListState({ load: 'ready', total: 3, shown: 2 })).toBeNull();
  });
});

describe('planDetailState', () => {
  test('a listed catalog only waits for a pick, whatever the queue says', () => {
    for (const queue of /** @type {const} */ (['connecting', 'unreachable', 'ok'])) {
      expect(planDetailState({ load: 'ready', total: 2, queue })).toEqual({
        icon: 'file-text',
        message: 'Select a plan to see its parameters.',
      });
    }
  });

  test('connecting says the halts are live, and offers Retry', () => {
    expect(planDetailState({ load: 'loading', total: 0, queue: 'connecting' })).toEqual({
      icon: 'radio',
      message: 'Connecting to the queue server…',
      detail: 'Stop and Abort are live while this connects.',
      retry: true,
    });
  });

  test('an unreadable manager is unreachable, with Retry and the halts still sent', () => {
    const state = planDetailState({ load: 'loading', total: 0, queue: 'unreachable' });
    expect(state).toMatchObject({
      icon: 'triangle-alert',
      message: 'Queue server unreachable',
      retry: true,
    });
    expect(state.detail).toContain('Stop and Abort are still sent');
  });

  test('a failed plan read with no queue stream either is one unreachable bridge', () => {
    expect(planDetailState({ load: 'error', total: 0, queue: 'connecting' }).message).toBe(
      'Queue server unreachable'
    );
  });

  test('a failed plan read beside a readable queue blames the plans, not the queue', () => {
    expect(planDetailState({ load: 'error', total: 0, queue: 'ok' })).toEqual({
      icon: 'triangle-alert',
      message: 'Plans could not be loaded.',
      retry: true,
    });
  });

  test('a readable queue while the plans load, or with none to load', () => {
    expect(planDetailState({ load: 'loading', total: 0, queue: 'ok' })).toEqual({
      icon: 'refresh',
      message: 'Loading plans…',
    });
    expect(planDetailState({ load: 'ready', total: 0, queue: 'ok' })).toEqual({
      icon: 'file-text',
      message: 'No plans available.',
    });
  });

  test('no state ever carries technical detail: no status code, no reason', () => {
    const states = [];
    for (const load of /** @type {const} */ (['loading', 'ready', 'error'])) {
      for (const queue of /** @type {const} */ (['connecting', 'unreachable', 'ok'])) {
        states.push(planDetailState({ load, total: 0, queue }));
      }
    }
    for (const state of states) expect(JSON.stringify(state)).not.toMatch(/\d{3}|_/);
  });
});
