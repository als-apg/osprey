// @ts-check
/**
 * Unit tests for the Activity page's derivations
 * (web_terminal/static/js/activity-log-model.js):
 *   npx vitest run tests/interfaces/web_terminal/activity-log-model.test.mjs
 */
import { describe, expect, test } from 'vitest';

import {
  callKey,
  findCallPath,
  flatCalls,
  matches,
  payloadSignature,
  summarize,
  turnsOf,
  walk,
} from '../../../src/osprey/interfaces/web_terminal/static/js/activity-log-model.js';
import { call, nodes, payload } from '../_support/activity-log.mjs';

/** @typedef {import('../../../src/osprey/interfaces/web_terminal/static/js/activity-log-model.js').ToolNode} ToolNode */

describe('walk', () => {
  test('visits every node at any depth, in timeline order', () => {
    /** @type {string[]} */
    const seen = [];
    walk(nodes(), (node) => seen.push(node.t === 'tool' ? callKey(node) : node.t));
    expect(seen).toEqual([
      'prompt',
      'thinking',
      'r1',
      'usage',
      'a1',
      'brief',
      'w1',
      'usage',
      'f1',
      'answer',
      'usage',
      'prompt',
      'note',
      'agent:agent-2',
      'o1',
    ]);
  });
});

describe('summarize', () => {
  test('counts calls at any depth, subagent calls included', () => {
    const m = summarize(payload());
    expect(m.tools.map((n) => n.id)).toEqual(['r1', 'a1', 'w1', 'f1', 'o1']);
    expect(m.subs.map((s) => s.id)).toEqual(['agent-1', 'agent-2']);
    expect(m.approvals.map((n) => n.id)).toEqual(['w1']);
    expect(m.errors.map((n) => n.id)).toEqual(['f1']);
    expect(m.gated.map((n) => n.id)).toEqual(['w1', 'f1']);
    expect(m.decided).toBe(3);
    expect(m.prompts).toBe(2);
  });

  test('an orphan group is a subagent, not a call; its calls count', () => {
    const m = summarize(payload());
    expect(m.tools.some((n) => n.orphan)).toBe(false);
    expect(m.subs.some((s) => s.id === 'agent-2')).toBe(true);
    expect(m.tools.some((n) => n.id === 'o1')).toBe(true);
  });

  test('sums model cost and model time over every usage entry', () => {
    const m = summarize(payload());
    expect(m.cost).toBeCloseTo(0.17, 10);
    expect(m.modelMs).toBe(4500);
  });

  test('the session length is end minus start', () => {
    expect(summarize(payload()).ms).toBe(125000);
  });

  test('no session, or a session without entries, summarizes to zeros', () => {
    const none = summarize({ session: null, nodes: [], artifacts: [], record: 'complete' });
    expect(none).toEqual({
      tools: [],
      subs: [],
      approvals: [],
      errors: [],
      gated: [],
      decided: 0,
      prompts: 0,
      cost: 0,
      modelMs: 0,
      ms: null,
    });
    expect(summarize(payload({ start: null, end: null }, { nodes: [] })).ms).toBeNull();
  });
});

describe('turnsOf', () => {
  test('a prompt opens a turn; what follows belongs to it', () => {
    const turns = turnsOf(nodes());
    expect(turns).toHaveLength(2);
    expect(turns[0].prompt?.text).toBe('Read the beam current');
    expect(turns[0].nodes.map((n) => n.t)).toEqual([
      'thinking',
      'tool',
      'usage',
      'tool',
      'answer',
      'usage',
    ]);
    expect(turns[1].prompt?.command).toBe('/status');
    expect(turns[1].nodes.map((n) => n.t)).toEqual(['note', 'tool']);
  });

  test('entries before the first prompt form a turn without one', () => {
    const [first, second] = turnsOf([
      { t: 'note', ts: '2026-11-12T10:00:00+00:00', icon: 'info', text: 'Harness instruction' },
      { t: 'prompt', ts: '2026-11-12T10:00:01+00:00', text: 'hello' },
    ]);
    expect(first.prompt).toBeNull();
    expect(first.nodes).toHaveLength(1);
    expect(second.prompt?.text).toBe('hello');
    expect(second.nodes).toEqual([]);
  });

  test('no entries, no turns', () => {
    expect(turnsOf([])).toEqual([]);
  });
});

describe('matches', () => {
  const all = nodes();
  const read = /** @type {ToolNode} */ (all[2]);
  const sub = /** @type {ToolNode} */ (all[4]);

  test('an empty query matches everything', () => {
    expect(matches(read, '')).toBe(true);
    expect(matches(all[0], '  ')).toBe(true);
  });

  test('a call matches on name, arguments and result text, in any case', () => {
    expect(matches(read, 'read')).toBe(true);
    expect(matches(read, 'CURRENT.TXT')).toBe(true);
    expect(matches(read, 'ok')).toBe(true);
    expect(matches(read, 'dcct')).toBe(false);
  });

  test('a subagent call matches on its type and description', () => {
    expect(matches(sub, 'orbit bump')).toBe(true);
    expect(matches(sub, 'logbook-search')).toBe(true);
  });

  test('a failed call matches on its error message', () => {
    const failed = /** @type {ToolNode} */ (sub.sub?.children[3]);
    expect(matches(failed, 'timed out')).toBe(true);
  });

  test('text entries match on text, command and body', () => {
    expect(matches(all[0], 'beam')).toBe(true);
    expect(matches(all[7], '/status')).toBe(true);
    expect(matches(all[8], '4711')).toBe(true);
    expect(matches(all[5], '401.2')).toBe(true);
    expect(matches(all[5], 'quadrupole')).toBe(false);
  });

  test('a usage entry matches no query', () => {
    expect(matches(all[3], '0.1')).toBe(false);
  });
});

describe('flatCalls', () => {
  test('every call at any depth, in timeline order', () => {
    expect(flatCalls(nodes()).map((n) => n.id)).toEqual(['r1', 'a1', 'w1', 'f1', 'o1']);
    expect(flatCalls(nodes(), { filter: 'tools' }).map((n) => n.id)).toEqual([
      'r1',
      'a1',
      'w1',
      'f1',
      'o1',
    ]);
  });

  test('approvals and errors narrow it', () => {
    expect(flatCalls(nodes(), { filter: 'approvals' }).map((n) => n.id)).toEqual(['w1']);
    expect(flatCalls(nodes(), { filter: 'errors' }).map((n) => n.id)).toEqual(['f1']);
  });

  test('a query narrows it further', () => {
    expect(flatCalls(nodes(), { q: 'hcm1' }).map((n) => n.id)).toEqual(['w1']);
    expect(flatCalls(nodes(), { filter: 'errors', q: 'hcm1' })).toEqual([]);
  });
});

describe('findCallPath', () => {
  test('a top-level call has no parents', () => {
    expect(findCallPath(nodes(), 'r1')).toEqual([]);
  });

  test('a subagent call names the calls that hold it', () => {
    expect(findCallPath(nodes(), 'f1')).toEqual(['a1']);
  });

  test('an orphan group, having no call id, is named by its subagent', () => {
    const orphan = /** @type {ToolNode} */ (nodes()[9]);
    expect(callKey(orphan)).toBe('agent:agent-2');
    expect(callKey(call('r1'))).toBe('r1');
    expect(findCallPath(nodes(), 'o1')).toEqual(['agent:agent-2']);
    expect(findCallPath(nodes(), 'agent:agent-2')).toEqual([]);
  });

  test('an unknown id is null', () => {
    expect(findCallPath(nodes(), 'nope')).toBeNull();
    expect(findCallPath([], 'r1')).toBeNull();
  });
});

describe('payloadSignature', () => {
  test('equal payloads have equal signatures', () => {
    expect(payloadSignature(payload())).toBe(payloadSignature(payload()));
  });

  test.each([
    ['entries', payload({ entries: 32 })],
    ['end', payload({ end: '2026-11-12T10:02:06+00:00' })],
    ['state', payload({ state: 'live' })],
    ['title', payload({ title: 'Beam current' })],
    ['record', payload({}, { record: 'incomplete' })],
    ['record_reason', payload({}, { record_reason: 'wait mark not passed' })],
    [
      'artifacts',
      payload(
        {},
        {
          artifacts: [
            {
              id: 'x1',
              title: 'Orbit',
              call: 'r1',
              tool: 'Read',
              agent: 'main',
              ts: '2026-11-12T10:00:03+00:00',
              form: 'plot',
              form_label: 'Plot',
              type: 'plot_html',
              size: 10,
            },
          ],
        }
      ),
    ],
  ])('a changed %s changes the signature', (_name, changed) => {
    expect(payloadSignature(changed)).not.toBe(payloadSignature(payload()));
  });

  test('a call that got its result changes the signature', () => {
    const running = payload();
    const read = /** @type {ToolNode} */ (running.nodes[2]);
    read.result = null;
    read.running = true;
    expect(payloadSignature(running)).not.toBe(payloadSignature(payload()));
  });

  test('no session has a signature too', () => {
    const empty = {
      session: null,
      nodes: [],
      artifacts: [],
      record: /** @type {const} */ ('complete'),
    };
    expect(payloadSignature(empty)).toBe(payloadSignature({ ...empty }));
    expect(payloadSignature(empty)).not.toBe(payloadSignature(payload()));
  });
});
