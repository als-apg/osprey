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
} from '../../../src/osprey/interfaces/web_terminal/static/js/activity-log-model.js';

/** @typedef {import('../../../src/osprey/interfaces/web_terminal/static/js/activity-log-model.js').ToolNode} ToolNode */
/** @typedef {import('../../../src/osprey/interfaces/web_terminal/static/js/activity-log-model.js').ActivityNode} ActivityNode */
/** @typedef {import('../../../src/osprey/interfaces/web_terminal/static/js/activity-log-model.js').ActivityPayload} ActivityPayload */

/**
 * A call as the server sends it, with the given fields over the defaults.
 * @param {string} id
 * @param {Partial<ToolNode>} [over]
 * @returns {ToolNode}
 */
function call(id, over = {}) {
  return {
    t: 'tool',
    ts: '2026-11-12T10:00:01+00:00',
    id,
    key: `cc:alice:s:${id}`,
    name: 'Read',
    server: '',
    tool: 'Read',
    agent: 'main',
    input: {},
    verdict: 'allowed',
    source: 'config',
    duration_ms: null,
    elapsed_ms: 12,
    target: null,
    posture: null,
    approval: null,
    result: { text: 'ok', size: 2, cut: false, key: `cc:alice:s:${id}:r` },
    error: false,
    errmsg: '',
    trail: [],
    incomplete: false,
    running: false,
    orphan: false,
    ...over,
  };
}

/** @type {ToolNode['approval']} */
const APPROVAL = {
  asked_ts: '2026-11-12T10:00:10+00:00',
  decided_ts: '2026-11-12T10:00:27.4+00:00',
  wait_ms: 17400,
  outcome: 'approved',
  approver: 'human',
  permission_mode: 'default',
  position: null,
};

/** @returns {ActivityNode[]} */
function nodes() {
  const write = call('w1', {
    name: 'mcp__controls__channel_write',
    server: 'controls',
    tool: 'channel_write',
    agent: 'logbook-search',
    input: { channel: 'SR:HCM1:SP', value: 0.5 },
    verdict: 'approved',
    approval: APPROVAL,
    target: 'virtual',
    trail: [{ ts: '2026-11-12T10:00:10+00:00', surface: 'controls', decision: 'ask', reason: '' }],
  });
  const failed = call('f1', {
    name: 'mcp__bluesky__run_plan',
    server: 'bluesky',
    tool: 'run_plan',
    agent: 'logbook-search',
    verdict: 'none',
    error: true,
    errmsg: 'Plan failed: Detector DCCT timed out',
    result: { text: '{"error": true}', size: 15, cut: false, key: 'k' },
    trail: [
      {
        ts: '2026-11-12T10:00:40+00:00',
        surface: 'bluesky',
        decision: 'refused',
        reason: 'tool_error',
      },
    ],
  });
  const sub = call('a1', {
    name: 'Agent',
    tool: 'Agent',
    input: { subagent_type: 'logbook-search', description: 'Find the last orbit bump' },
    result: null,
    sub: {
      id: 'agent-1',
      type: 'logbook-search',
      description: 'Find the last orbit bump',
      ms: 41000,
      children: [
        { t: 'brief', ts: '2026-11-12T10:00:05+00:00', text: 'Search the logbook' },
        write,
        { t: 'usage', ts: '2026-11-12T10:00:30+00:00', cost: 0.02, ms: 900 },
        failed,
      ],
    },
  });
  // A subagent whose parent call the record does not hold: no id, no key.
  const orphan = call('', {
    key: '',
    name: 'Agent',
    tool: 'Agent',
    verdict: 'none',
    result: null,
    orphan: true,
    sub: {
      id: 'agent-2',
      type: 'explorer',
      description: '',
      ms: null,
      children: [call('o1', { agent: 'explorer', verdict: 'none' })],
    },
  });
  return [
    { t: 'prompt', ts: '2026-11-12T10:00:00+00:00', text: 'Read the beam current' },
    { t: 'thinking', ts: '2026-11-12T10:00:00.5+00:00', text: 'I will read the DCCT first.' },
    call('r1', { input: { file_path: '/data/current.txt' } }),
    { t: 'usage', ts: '2026-11-12T10:00:02+00:00', cost: 0.1, ms: 2100 },
    sub,
    { t: 'answer', ts: '2026-11-12T10:00:50+00:00', text: 'The beam current is **401.2 mA**.' },
    { t: 'usage', ts: '2026-11-12T10:00:51+00:00', cost: 0.05, ms: 1500 },
    { t: 'prompt', ts: '2026-11-12T10:02:00+00:00', text: 'detailed', command: '/status' },
    {
      t: 'note',
      ts: '2026-11-12T10:02:01+00:00',
      icon: 'reply',
      text: 'Subagent reported back',
      size: 20,
      body: 'Found entry 4711',
    },
    orphan,
  ];
}

/**
 * @param {Partial<NonNullable<ActivityPayload['session']>>} [session]
 * @param {Partial<ActivityPayload>} [over]
 * @returns {ActivityPayload}
 */
function payload(session = {}, over = {}) {
  return {
    session: {
      id: 'aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee',
      session_key: null,
      title: 'Read the beam current',
      identity: 'alice',
      start: '2026-11-12T10:00:00+00:00',
      end: '2026-11-12T10:02:05+00:00',
      state: 'idle',
      entries: 31,
      kinds: { prompt: 2, tool_call: 5 },
      target: 'virtual',
      posture: 'writes',
      retention_days: 14,
      ...session,
    },
    nodes: nodes(),
    artifacts: [],
    record: 'complete',
    ...over,
  };
}

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
