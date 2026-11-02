/**
 * Unit tests for chat-tool-runs.js -- how a turn's tool rows fold into runs.
 *
 * Consecutive calls of one kind fold into one run row; anything else breaks the
 * run. An approval card takes its call out of the run so the card stays in
 * view. A run with a call still running is live, a failure inside it surfaces
 * on its row, and the one row that pulses is the turn's current action.
 *
 * happy-dom environment (configured globally in vitest.config.js):
 *   npx vitest run tests/interfaces/web_terminal/chat-tool-runs.test.mjs
 */

import { describe, test, expect } from 'vitest';

import { qs } from '../_support/dom.mjs';

import { createTurnActions } from '../../../src/osprey/interfaces/web_terminal/static/js/chat-tool-runs.js';

/**
 * A tool call as the stream carries it.
 * @param {string} id
 * @param {string} raw - the raw tool name
 * @param {Record<string, unknown>} [input]
 */
function call(id, raw, input = {}) {
  return { type: 'tool_use', tool_name: raw, tool_name_raw: raw, tool_use_id: id, input };
}

/** A fresh entry with its actions. */
function setup() {
  const entry = document.createElement('div');
  entry.className = 'op-entry assistant';
  return { entry, actions: createTurnActions(entry) };
}

/** The top-level children of the entry, in order, by what they are. */
const rowsOf = (/** @type {HTMLElement} */ entry) =>
  [...entry.children].map((el) => {
    if (el.classList.contains('op-run')) return 'run';
    if (el.classList.contains('op-agent')) return 'agent';
    if (el.classList.contains('op-tool')) return 'tool';
    return el.tagName.toLowerCase();
  });

describe('folding', () => {
  test('consecutive calls of one kind fold into one run row that counts and names them', () => {
    const { entry, actions } = setup();
    actions.addTool(call('t1', 'mcp__osprey__channel_read', { channel: 'SR:C01:BPM' }));
    actions.addTool(call('t2', 'mcp__osprey__channel_read', { channel: 'SR:C02:BPM' }));
    actions.addTool(call('t3', 'mcp__osprey__channel_read', { channel: 'SR:C03:BPM' }));

    expect(rowsOf(entry)).toEqual(['run']);
    const run = qs(entry, '.op-run');
    expect(qs(run, '.op-row-label').textContent).toBe('Read 3 channels');
    expect(qs(run, '.op-row-detail').textContent).toBe('SR:C01:BPM, SR:C02:BPM, SR:C03:BPM');
    expect(run.querySelectorAll('.op-run-list > .op-tool')).toHaveLength(3);
    // Folded by default; the stack shows one glyph per call.
    expect(qs(run, '.op-row-body').hidden).toBe(true);
    expect(run.querySelectorAll('.op-row-stack > .op-icon')).toHaveLength(3);
  });

  test.each([
    ['the operator ran every skill', [{ by: 'operator' }, { by: 'operator' }], 'Ran 2 skills'],
    ['the agent chose them', [{}, {}], 'Used 2 skills'],
    ['the two are mixed', [{ by: 'operator' }, {}], 'Used 2 skills'],
  ])('a run of skills reads by who ran them: %s', (_case, inputs, label) => {
    const { entry, actions } = setup();
    inputs.forEach((by, i) => {
      actions.addTool(call(`s${i}`, 'Skill', { skill: `diagnose-${i}`, ...by }));
    });

    expect(qs(entry, '.op-run .op-row-label').textContent).toBe(label);
  });

  test('a single call is its own row, not a run of one', () => {
    const { entry, actions } = setup();
    actions.addTool(call('t1', 'Read', { file_path: 'a.py' }));
    expect(rowsOf(entry)).toEqual(['tool']);
  });

  test("the stack shows at most four glyphs, and a run of mixed tools keeps its kind's phrase", () => {
    const { entry, actions } = setup();
    for (const id of ['t1', 't2', 't3', 't4', 't5']) {
      actions.addTool(
        call(id, id === 't3' ? 'mcp__osprey__list_channels' : 'mcp__osprey__channel_read')
      );
    }
    const run = qs(entry, '.op-run');
    expect(run.querySelectorAll('.op-row-stack > .op-icon')).toHaveLength(4);
    expect(qs(run, '.op-row-label').textContent).toBe('Made 5 control calls');
  });

  test('a call of another kind breaks the run and starts its own', () => {
    const { entry, actions } = setup();
    actions.addTool(call('t1', 'Read'));
    actions.addTool(call('t2', 'Read'));
    actions.addTool(call('t3', 'Bash'));
    actions.addTool(call('t4', 'Read'));
    actions.addTool(call('t5', 'Read'));

    expect(rowsOf(entry)).toEqual(['run', 'tool', 'run']);
  });

  test('a control write never folds: not with reads, and not with another write', () => {
    const { entry, actions } = setup();
    actions.addTool(call('t1', 'mcp__osprey__channel_read'));
    actions.addTool(call('t2', 'mcp__osprey__channel_write'));
    actions.addTool(call('t3', 'mcp__osprey__channel_write'));
    actions.addTool(call('t4', 'mcp__osprey__channel_read'));
    actions.addTool(call('t5', 'mcp__osprey__channel_read'));

    expect(rowsOf(entry)).toEqual(['tool', 'tool', 'tool', 'run']);
    expect(entry.children[1].getAttribute('data-kind')).toBe('control-write');
    expect(entry.children[2].getAttribute('data-kind')).toBe('control-write');
    expect(qs(entry, '.op-run .op-row-label').textContent).toBe('Read 2 channels');
  });

  test('a subagent, reasoning, text or a card between calls breaks the run', () => {
    const { entry, actions } = setup();
    actions.addTool(call('t1', 'Read'));
    actions.breakRun();
    actions.addTool(call('t2', 'Read'));
    actions.addAgent(
      { type: 'tool_use', tool_name_raw: 'Agent', input: { subagent_type: 'scout' } },
      () => {}
    );
    actions.addTool(call('t3', 'Read'));

    expect(rowsOf(entry)).toEqual(['tool', 'tool', 'agent', 'tool']);
  });
});

describe('an approval breaks the run', () => {
  test('the last call leaves the run for its card, and the run before it stays folded', () => {
    const { entry, actions } = setup();
    actions.addTool(call('t1', 'Read'));
    actions.addTool(call('t2', 'Read'));
    const third = actions.addTool(call('t3', 'Read'));
    const card = document.createElement('section');
    third.file(card);

    // The run of two, the ejected row, then the card under it.
    expect(rowsOf(entry)).toEqual(['run', 'tool', 'section']);
    expect(entry.children[1]).toBe(third.el);
    expect(entry.children[2]).toBe(card);
    expect(qs(entry, '.op-run .op-row-label').textContent).toBe('Read 2 files');
    // The run is broken: a later call of the same kind starts afresh.
    actions.addTool(call('t4', 'Read'));
    expect(entry.lastElementChild?.classList.contains('op-tool')).toBe(true);
  });

  test("a run left with one call is that call's own row again", () => {
    const { entry, actions } = setup();
    const first = actions.addTool(call('t1', 'Read'));
    const second = actions.addTool(call('t2', 'Read'));
    second.file(document.createElement('section'));

    expect(entry.children[0]).toBe(first.el);
    expect(entry.children[1]).toBe(second.el);
    expect(entry.querySelector('.op-run')).toBeNull();
  });

  test('a call in the middle splits the run around it', () => {
    const { entry, actions } = setup();
    actions.addTool(call('t1', 'Read'));
    actions.addTool(call('t2', 'Read'));
    const third = actions.addTool(call('t3', 'Read'));
    actions.addTool(call('t4', 'Read'));
    actions.addTool(call('t5', 'Read'));
    third.file(document.createElement('section'));

    expect(rowsOf(entry)).toEqual(['run', 'tool', 'section', 'run']);
    const runs = entry.querySelectorAll('.op-run');
    expect(qs(runs[0], '.op-row-label').textContent).toBe('Read 2 files');
    expect(qs(runs[1], '.op-row-label').textContent).toBe('Read 2 files');
    // The calls after it are still the open run: the next call joins them.
    actions.addTool(call('t6', 'Read'));
    expect(qs(runs[1], '.op-row-label').textContent).toBe('Read 3 files');
  });
});

describe('state', () => {
  test('a run is live while a call in it runs, and settles when the last one lands', () => {
    const { entry, actions } = setup();
    const a = actions.addTool(call('t1', 'Read'));
    const b = actions.addTool(call('t2', 'Read'));
    const run = qs(entry, '.op-run');
    expect(run.dataset.state).toBe('running');

    a.finish({ is_error: false, preview: 'ok' });
    expect(run.dataset.state).toBe('running');
    b.finish({ is_error: false, preview: 'ok' });
    expect(run.dataset.state).toBe('done');
    expect(run.querySelector('.op-row-status .op-icon')).not.toBeNull();
  });

  test('a failure inside a run surfaces on the run row', () => {
    const { entry, actions } = setup();
    const a = actions.addTool(call('t1', 'Read'));
    const b = actions.addTool(call('t2', 'Read'));
    a.finish({ is_error: true, preview: 'ENOENT' });
    b.finish({ is_error: false, preview: 'ok' });

    const run = qs(entry, '.op-run');
    expect(run.dataset.state).toBe('error');
    expect(qs(run, '.op-run-failed').hidden).toBe(false);
    expect(qs(run, '.op-run-failed').textContent).toBe('1 failed');
  });

  test('a growing run stays live, and settles once its calls have all landed', () => {
    const { entry, actions } = setup();
    const a = actions.addTool(call('t1', 'Read'));
    a.finish({ is_error: false, preview: 'ok' });
    const b = actions.addTool(call('t2', 'Read'));
    const run = qs(entry, '.op-run');
    expect(run.dataset.state).toBe('running');
    b.finish({ is_error: false, preview: 'ok' });
    expect(run.dataset.state).toBe('done');
  });

  test('only the action that started last pulses; the pulse passes back when it settles', () => {
    const { entry, actions } = setup();
    const thinking = document.createElement('div');
    thinking.className = 'op-row op-thinking';
    entry.append(thinking);
    actions.live(thinking);
    expect(thinking.classList.contains('pulse')).toBe(true);

    const a = actions.addTool(call('t1', 'Read'));
    expect(thinking.classList.contains('pulse')).toBe(false);
    expect(a.el.classList.contains('pulse')).toBe(true);

    // A second call folds the first into a run: the run row carries the pulse.
    const b = actions.addTool(call('t2', 'Read'));
    const run = qs(entry, '.op-run');
    expect(run.classList.contains('pulse')).toBe(true);
    expect(a.el.classList.contains('pulse')).toBe(false);
    expect(entry.querySelectorAll('.pulse')).toHaveLength(1);

    b.finish({ is_error: false, preview: 'ok' });
    a.finish({ is_error: false, preview: 'ok' });
    // Nothing of the run runs: the pulse returns to the thought still open.
    expect(run.classList.contains('pulse')).toBe(false);
    expect(thinking.classList.contains('pulse')).toBe(true);
    actions.settled(thinking);
    expect(entry.querySelectorAll('.pulse')).toHaveLength(0);
  });
});

describe('what the turn did', () => {
  test('counts the calls and approvals, and keeps each glyph once in order of first use', () => {
    const { actions } = setup();
    actions.addTool(call('t1', 'Read'));
    actions.addTool(call('t2', 'Read'));
    actions.addTool(call('t3', 'Bash'));
    actions.addAgent({ type: 'tool_use', tool_name_raw: 'Agent', input: {} }, () => {});
    actions.countApproval();

    expect(actions.stats()).toEqual({
      actions: 4,
      approvals: 1,
      icons: ['read', 'bash', 'agent'],
    });
  });
});
