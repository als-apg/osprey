// @ts-check
/**
 * Unit tests for the Activity page's view state and what a press does to it
 * (web_terminal/static/js/activity-log-state.js):
 *   npx vitest run tests/interfaces/web_terminal/activity-log-state.test.mjs
 */
import { describe, expect, test } from 'vitest';

import {
  applyAction,
  newUiState,
  resetForSession,
  revealCall,
} from '../../../src/osprey/interfaces/web_terminal/static/js/activity-log-state.js';
import { nodes } from '../_support/activity-log.mjs';

/** @param {string[]} [toggles] */
const ctx = (toggles = []) => ({ nodes: nodes(), toggles });

describe('newUiState', () => {
  test('a fresh page shows the whole timeline, nothing open', () => {
    const ui = newUiState();
    expect(ui).toMatchObject({ view: 'timeline', filter: 'all', scope: 'all', q: '' });
    expect(ui.sort).toEqual({ k: 'ts', dir: 1 });
    expect([ui.open.size, ui.closed.size, ui.full.size]).toEqual([0, 0, 0]);
  });

  test('opens on the view a link names', () => {
    expect(newUiState('safety').view).toBe('safety');
  });
});

describe('applyAction', () => {
  test('a view tab switches the view', () => {
    const ui = newUiState();
    expect(applyAction(ui, { view: 'tools' }, ctx())).toBe(true);
    expect(ui.view).toBe('tools');
  });

  test('a view the page does not have changes nothing', () => {
    const ui = newUiState();
    expect(applyAction(ui, { view: 'everything' }, ctx())).toBe(false);
    expect(ui.view).toBe('timeline');
  });

  test('a filter and a scope are taken as pressed', () => {
    const ui = newUiState();
    expect(applyAction(ui, { filter: 'errors' }, ctx())).toBe(true);
    expect(applyAction(ui, { scope: 'mcp' }, ctx())).toBe(true);
    expect([ui.filter, ui.scope]).toEqual(['errors', 'mcp']);
  });

  test('a row opens, then closes', () => {
    const ui = newUiState();
    applyAction(ui, { toggle: 'r1' }, ctx());
    expect(ui.open.has('r1')).toBe(true);
    applyAction(ui, { toggle: 'r1' }, ctx());
    expect(ui.open.has('r1')).toBe(false);
  });

  test('a subagent group collapses, then opens again', () => {
    const ui = newUiState();
    applyAction(ui, { collapse: 'a1' }, ctx());
    expect(ui.closed.has('a1')).toBe(true);
    applyAction(ui, { collapse: 'a1' }, ctx());
    expect(ui.closed.has('a1')).toBe(false);
  });

  test('a stat goes to its view', () => {
    const ui = newUiState();
    expect(applyAction(ui, { go: 'tools:all' }, ctx())).toBe(true);
    expect([ui.view, ui.filter]).toEqual(['tools', 'all']);
  });

  test('a stat that goes to the timeline sets its filter', () => {
    const ui = newUiState('tools');
    applyAction(ui, { go: 'timeline:errors' }, ctx());
    expect([ui.view, ui.filter]).toEqual(['timeline', 'errors']);
  });

  test('a column sorts ascending first, then flips', () => {
    const ui = newUiState();
    applyAction(ui, { sort: 'tool' }, ctx());
    expect(ui.sort).toEqual({ k: 'tool', dir: 1 });
    applyAction(ui, { sort: 'tool' }, ctx());
    expect(ui.sort).toEqual({ k: 'tool', dir: -1 });
    applyAction(ui, { sort: 'ts' }, ctx());
    expect(ui.sort).toEqual({ k: 'ts', dir: 1 });
  });

  test('expand all opens every call at any depth, every fold shown, and every group', () => {
    const ui = newUiState();
    ui.closed.add('a1');
    expect(applyAction(ui, { act: 'expand' }, ctx(['t0:1']))).toBe(true);
    for (const key of ['r1', 'w1', 'f1', 'o1', 't0:1']) expect(ui.open.has(key)).toBe(true);
    // A subagent is a group, not a row that opens.
    expect(ui.open.has('a1')).toBe(false);
    expect(ui.closed.size).toBe(0);
  });

  test('collapse all closes every row and leaves the groups as they are', () => {
    const ui = newUiState();
    ui.open.add('r1').add('t0:1');
    ui.closed.add('a1');
    expect(applyAction(ui, { act: 'collapse' }, ctx())).toBe(true);
    expect(ui.open.size).toBe(0);
    expect(ui.closed.has('a1')).toBe(true);
  });

  test('a jump shows the call on the timeline', () => {
    const ui = newUiState('safety');
    expect(applyAction(ui, { jump: 'w1' }, ctx())).toBe(true);
    expect(ui.view).toBe('timeline');
    expect(ui.open.has('w1')).toBe(true);
  });

  test.each([
    ['copy link', { act: 'copy-link' }],
    ['refresh', { act: 'refresh' }],
    ['a session', { session: 'aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee' }],
    ['nothing', {}],
  ])('%s is not a state change', (_name, data) => {
    const ui = newUiState();
    expect(applyAction(ui, data, ctx())).toBe(false);
  });
});

describe('revealCall', () => {
  test('clears what would hide the call and opens it', () => {
    const ui = newUiState('tools');
    ui.filter = 'errors';
    ui.q = 'dcct';
    revealCall(ui, nodes(), 'r1');
    expect(ui).toMatchObject({ view: 'timeline', filter: 'all', q: '' });
    expect(ui.open.has('r1')).toBe(true);
  });

  test('opens the subagent group a call is inside', () => {
    const ui = newUiState();
    ui.closed.add('a1');
    revealCall(ui, nodes(), 'w1');
    expect(ui.closed.has('a1')).toBe(false);
    expect(ui.open.has('w1')).toBe(true);
  });

  test('opens an orphan group by the name the page gives it', () => {
    const ui = newUiState();
    ui.closed.add('agent:agent-2');
    revealCall(ui, nodes(), 'o1');
    expect(ui.closed.has('agent:agent-2')).toBe(false);
  });

  test('a call the session does not hold still lands on the timeline', () => {
    const ui = newUiState('artifacts');
    revealCall(ui, nodes(), 'gone');
    expect(ui.view).toBe('timeline');
  });
});

describe('resetForSession', () => {
  test('drops what belonged to the session and keeps the view', () => {
    const ui = newUiState('tools');
    ui.filter = 'errors';
    ui.q = 'beam';
    ui.scope = 'mcp';
    ui.sort = { k: 'tool', dir: -1 };
    ui.open.add('r1');
    ui.closed.add('a1');
    ui.full.set('r1', 'whole');
    resetForSession(ui);
    expect(ui).toMatchObject({ view: 'tools', filter: 'all', q: '', scope: 'mcp' });
    expect(ui.sort).toEqual({ k: 'tool', dir: -1 });
    expect([ui.open.size, ui.closed.size, ui.full.size]).toEqual([0, 0, 0]);
  });
});
