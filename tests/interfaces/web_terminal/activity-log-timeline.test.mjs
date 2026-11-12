// @ts-check
/**
 * Unit tests for the Activity page's Timeline view
 * (web_terminal/static/js/activity-log-timeline.js):
 *   npx vitest run tests/interfaces/web_terminal/activity-log-timeline.test.mjs
 */
import { afterEach, beforeEach, describe, expect, test } from 'vitest';

import { summarize } from '../../../src/osprey/interfaces/web_terminal/static/js/activity-log-model.js';
import { renderTimeline } from '../../../src/osprey/interfaces/web_terminal/static/js/activity-log-timeline.js';
import { call, dom, payload, texts, uiState } from '../_support/activity-log.mjs';
import { qs } from '../_support/dom.mjs';
import { stampFacilityZone } from '../_support/facility-zone.mjs';

/** @typedef {import('../../../src/osprey/interfaces/web_terminal/static/js/activity-log-model.js').ActivityNode} ActivityNode */
/** @typedef {import('../../../src/osprey/interfaces/web_terminal/static/js/activity-log-model.js').ActivityPayload} ActivityPayload */
/** @typedef {import('../../../src/osprey/interfaces/web_terminal/static/js/activity-log-model.js').ArtifactRow} ArtifactRow */
/** @typedef {import('../_support/activity-log.mjs').UiState} UiState */

const SCRIPT = '<script>alert(1)</script>';

beforeEach(() => stampFacilityZone('UTC'));
afterEach(() => stampFacilityZone(null));

/**
 * One view of *p* as a detached element.
 * @param {(payload: ActivityPayload, model: ReturnType<typeof summarize>, ui: UiState) => string} render
 * @param {ActivityPayload} [p]
 * @param {Partial<UiState>} [ui]
 */
function view(render, p = payload(), ui = {}) {
  return dom(render(p, summarize(p), uiState(ui)));
}

/**
 * A session of *nodes* after one prompt.
 * @param {ActivityNode[]} nodes
 * @param {Partial<NonNullable<ActivityPayload['session']>>} [session]
 */
function sessionOf(nodes, session = {}) {
  /** @type {ActivityNode} */
  const prompt = { t: 'prompt', ts: '2026-11-12T10:00:00+00:00', text: 'Go' };
  return payload(session, { nodes: [prompt, ...nodes] });
}

/** The call keys of the tool rows under *root*, in order. @param {ParentNode} root */
function rowKeys(root) {
  return [...root.querySelectorAll('.act-row[data-kind="tool"][data-toggle]')].map(
    (row) => /** @type {HTMLElement} */ (row).dataset.toggle
  );
}

describe('renderTimeline', () => {
  test('a turn is a prompt and what followed it', () => {
    const timeline = view(renderTimeline);
    expect(timeline.querySelectorAll('.act-turn')).toHaveLength(2);
    expect(texts(timeline, '.act-who')).toEqual(['alice asked', 'alice asked']);
    expect(texts(timeline, '.act-prompt-text')).toEqual([
      'Read the beam current',
      '/status detailed',
    ]);
    expect(qs(timeline, '.act-prompt-text code').textContent).toBe('/status');
  });

  test('a prompt is escaped', () => {
    const timeline = view(
      renderTimeline,
      payload({}, { nodes: [{ t: 'prompt', ts: '', text: SCRIPT }] })
    );
    expect(timeline.querySelector('script')).toBeNull();
    expect(qs(timeline, '.act-prompt-text').textContent).toBe(SCRIPT);
  });

  test('an answer is text, marked for the markdown upgrade', () => {
    const timeline = view(
      renderTimeline,
      sessionOf([{ t: 'answer', ts: '2026-11-12T10:00:50+00:00', text: `**401.2 mA** ${SCRIPT}` }])
    );
    expect(timeline.querySelector('script')).toBeNull();
    expect(qs(timeline, '.act-md[data-md]').textContent).toBe(`**401.2 mA** ${SCRIPT}`);
  });

  test('thinking, a subagent brief and a note with a body are folded', () => {
    const timeline = view(renderTimeline);
    expect(texts(timeline, 'button.act-row[data-kind="thinking"] .act-think')).toEqual([
      'I will read the DCCT first.',
    ]);
    expect(texts(timeline, 'button.act-row[data-kind="note"] .act-muted')).toEqual([
      'Task given to the subagent',
      'Subagent reported back',
    ]);
    expect(timeline.querySelector('.act-fold')).toBeNull();
  });

  test('an opened fold shows its text', () => {
    const closed = view(renderTimeline);
    const id = qs(closed, 'button.act-row[data-kind="thinking"]').dataset.toggle ?? '';
    const open = view(renderTimeline, payload(), { open: new Set([id]) });
    expect(qs(open, '.act-body-block .act-fold').textContent).toBe('I will read the DCCT first.');
  });

  test('fold ids are unique across the timeline', () => {
    const ids = [...view(renderTimeline).querySelectorAll('[data-toggle]')].map(
      (row) => /** @type {HTMLElement} */ (row).dataset.toggle
    );
    expect(new Set(ids).size).toBe(ids.length);
  });

  test('a note without a body is a plain row with its size', () => {
    const timeline = view(
      renderTimeline,
      sessionOf([
        {
          t: 'note',
          ts: '2026-11-12T10:00:03+00:00',
          icon: 'book',
          text: 'Skill loaded: orbit',
          size: 2117,
        },
      ])
    );
    const note = qs(timeline, 'div.act-row[data-kind="note"]');
    expect(qs(note, '.act-muted').textContent).toBe('Skill loaded: orbit');
    expect(note.textContent).toContain('2.1 KB');
    expect(note.querySelector('svg')).not.toBeNull();
  });

  test('a subagent is a group that holds its calls', () => {
    const group = qs(view(renderTimeline), '.act-group[id="row-a1"]');
    const head = qs(group, 'button.act-group-head');
    expect(head.dataset.collapse).toBe('a1');
    expect(head.getAttribute('aria-expanded')).toBe('true');
    expect(qs(head, '.act-srv').textContent).toBe('subagent');
    expect(qs(head, '.act-tool').textContent).toBe('logbook-search');
    expect(head.textContent).toContain('Find the last orbit bump');
    expect(head.textContent).toContain('2 calls');
    expect(head.textContent).toContain('41.0 s');
    expect(qs(head, '.osprey-chip--err').textContent).toBe('1 failed');
    expect(rowKeys(group)).toEqual(['w1', 'f1']);
  });

  test('a collapsed group hides its calls', () => {
    const timeline = view(renderTimeline, payload(), { closed: new Set(['a1']) });
    const group = qs(timeline, '.act-group[id="row-a1"]');
    expect(qs(group, '.act-group-head').getAttribute('aria-expanded')).toBe('false');
    expect(group.querySelector('.act-steps')).toBeNull();
    expect(rowKeys(timeline)).not.toContain('w1');
  });

  test('an orphan subagent group shows its calls', () => {
    const group = qs(view(renderTimeline), '.act-group[id="row-agent:agent-2"]');
    const head = qs(group, '.act-group-head');
    expect(head.dataset.collapse).toBe('agent:agent-2');
    expect(qs(head, '.act-tool').textContent).toBe('explorer');
    expect(head.textContent).toContain('1 call');
    expect(head.textContent).not.toContain('1 calls');
    expect(rowKeys(group)).toEqual(['o1']);
  });

  test('a turn ends with its model requests, time and cost, subagents included', () => {
    const timeline = view(renderTimeline);
    expect(texts(timeline, '.act-turn-foot')).toEqual([
      'Turn: 3 model requests, 4.5 s model time, $0.17',
    ]);
  });

  test('model requests before the first prompt are not called a turn', () => {
    const timeline = view(
      renderTimeline,
      payload(
        {},
        {
          nodes: [
            { t: 'usage', ts: '2026-11-12T09:59:46+00:00', cost: 0.000224, ms: 2053 },
            { t: 'prompt', ts: '2026-11-12T10:00:00+00:00', text: 'Go' },
            { t: 'usage', ts: '2026-11-12T10:00:04+00:00', cost: 0.03, ms: 3880 },
          ],
        }
      )
    );
    expect(texts(timeline, '.act-turn-foot')).toEqual([
      'Before the first prompt: 1 model request, 2.1 s model time, $0.00',
      'Turn: 1 model request, 3.9 s model time, $0.03',
    ]);
  });

  test('a run of task-list calls folds into one row', () => {
    const p = sessionOf([
      call('t1', { name: 'TaskCreate', tool: 'TaskCreate' }),
      { t: 'usage', ts: '2026-11-12T10:00:02+00:00', cost: 0.01, ms: 100 },
      call('t2', { name: 'TaskUpdate', tool: 'TaskUpdate' }),
      call('t3', { name: 'TodoWrite', tool: 'TodoWrite' }),
      call('r1'),
    ]);
    const closed = view(renderTimeline, p);
    const fold = qs(closed, 'button.act-row[data-kind="note"]');
    expect(qs(fold, '.act-muted').textContent).toBe('Task list: 3 updates');
    expect(rowKeys(closed)).toEqual(['r1']);
    const open = view(renderTimeline, p, { open: new Set([fold.dataset.toggle ?? '']) });
    expect(rowKeys(qs(open, '.act-body-block .act-steps'))).toEqual(['t1', 't2', 't3']);
  });

  test('one task-list call alone stays a call', () => {
    const timeline = view(
      renderTimeline,
      sessionOf([call('t1', { name: 'TaskCreate', tool: 'TaskCreate' })])
    );
    expect(rowKeys(timeline)).toEqual(['t1']);
    expect(timeline.textContent).not.toContain('Task list:');
  });

  test('the bar: five filters, the clock the times are on, expand and collapse', () => {
    const timeline = view(renderTimeline, payload(), { filter: 'errors' });
    expect(texts(timeline, '.osprey-seg [data-filter]')).toEqual([
      'All',
      'Conversation',
      'Tool calls',
      'Approvals',
      'Errors',
    ]);
    expect(qs(timeline, '[data-filter][aria-pressed="true"]').dataset.filter).toBe('errors');
    expect(qs(timeline, '.act-sub .hint').textContent).toBe('Times in UTC');
    expect(timeline.querySelector('[data-act="expand"]')).not.toBeNull();
    expect(timeline.querySelector('[data-act="collapse"]')).not.toBeNull();
  });

  test('errors: the failed calls, flat', () => {
    const timeline = view(renderTimeline, payload(), { filter: 'errors' });
    expect(timeline.querySelector('.act-turn')).toBeNull();
    expect(rowKeys(timeline)).toEqual(['f1']);
  });

  test('errors, and none failed', () => {
    const timeline = view(renderTimeline, sessionOf([call('r1')]), { filter: 'errors' });
    expect(qs(timeline, '.osprey-state-message').textContent).toBe(
      'No failed calls in this session'
    );
  });

  test('approvals: the calls the operator was asked about', () => {
    expect(rowKeys(view(renderTimeline, payload(), { filter: 'approvals' }))).toEqual(['w1']);
    const none = view(renderTimeline, sessionOf([call('r1')]), { filter: 'approvals' });
    expect(qs(none, '.osprey-state-message').textContent).toBe(
      'No call in this session asked the operator'
    );
  });

  test('tool calls: every call at any depth, without the subagent groups', () => {
    expect(rowKeys(view(renderTimeline, payload(), { filter: 'tools' }))).toEqual([
      'r1',
      'w1',
      'f1',
      'o1',
    ]);
    const none = view(renderTimeline, sessionOf([]), { filter: 'tools' });
    expect(qs(none, '.osprey-state-message').textContent).toBe('No tool calls in this session');
  });

  test('conversation: prompts and answers only', () => {
    const timeline = view(renderTimeline, payload(), { filter: 'conversation' });
    expect(timeline.querySelectorAll('.act-prompt')).toHaveLength(2);
    expect(timeline.querySelectorAll('.act-md')).toHaveLength(1);
    expect(rowKeys(timeline)).toEqual([]);
    expect(timeline.querySelector('.act-group')).toBeNull();
    expect(timeline.querySelector('.act-turn-foot')).toBeNull();
  });

  test('a search keeps what holds the text, at any depth', () => {
    const timeline = view(renderTimeline, payload(), { q: 'dcct' });
    expect(timeline.querySelector('.act-turn')).toBeNull();
    expect(texts(timeline, '.act-think')).toEqual(['I will read the DCCT first.']);
    expect(rowKeys(timeline)).toEqual(['f1']);
  });

  test('a search finds prompts and answers too', () => {
    const timeline = view(renderTimeline, payload(), { q: 'beam current' });
    expect(texts(timeline, '.act-prompt-text')).toEqual(['Read the beam current']);
    expect(timeline.querySelectorAll('.act-md')).toHaveLength(1);
  });

  test('a search that matches nothing says so, escaped', () => {
    const timeline = view(renderTimeline, payload(), { q: '<zzz>' });
    expect(qs(timeline, '.osprey-state-message').textContent).toBe(
      'Nothing in this session matches “<zzz>”'
    );
  });

  test('a session with no entries yet says so', () => {
    const timeline = view(
      renderTimeline,
      payload({ start: null, end: null, entries: 0 }, { nodes: [] })
    );
    expect(qs(timeline, '.osprey-state-message').textContent).toBe(
      'No activity in this session yet'
    );
  });

  test('a call without a result is running only while the session is live', () => {
    const pending = call('p1', { result: null, running: true, elapsed_ms: null });
    const live = view(renderTimeline, sessionOf([pending], { state: 'live' }));
    expect(qs(live, '.act-running').textContent).toBe('running');
    const idle = view(renderTimeline, sessionOf([pending]));
    expect(idle.querySelector('.act-running')).toBeNull();
  });

  test('no inline styles', () => {
    const p = sessionOf([
      call('t1', { name: 'TaskCreate', tool: 'TaskCreate' }),
      call('t2', { name: 'TaskUpdate', tool: 'TaskUpdate' }),
    ]);
    const closed = view(renderTimeline, p);
    const id = qs(closed, 'button.act-row[data-kind="note"]').dataset.toggle ?? '';
    expect(view(renderTimeline, p, { open: new Set([id]) }).querySelector('[style]')).toBeNull();
    expect(view(renderTimeline).querySelector('[style]')).toBeNull();
  });
});
