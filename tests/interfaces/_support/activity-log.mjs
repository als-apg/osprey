// @ts-check
/**
 * Activity-page fixtures for interface tests: one `/api/session-activity`
 * answer as the server sends it, with a call inside a subagent, a failed
 * call, an approval and an orphan subagent group.
 */

/** @typedef {import('../../../src/osprey/interfaces/web_terminal/static/js/activity-log-model.js').ToolNode} ToolNode */
/** @typedef {import('../../../src/osprey/interfaces/web_terminal/static/js/activity-log-model.js').ActivityNode} ActivityNode */
/** @typedef {import('../../../src/osprey/interfaces/web_terminal/static/js/activity-log-model.js').ActivityPayload} ActivityPayload */

/**
 * A call as the server sends it, with the given fields over the defaults.
 * @param {string} id
 * @param {Partial<ToolNode>} [over]
 * @returns {ToolNode}
 */
export function call(id, over = {}) {
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

/** @type {NonNullable<ToolNode['approval']>} */
export const APPROVAL = {
  asked_ts: '2026-11-12T10:00:10+00:00',
  decided_ts: '2026-11-12T10:00:27.4+00:00',
  wait_ms: 17400,
  outcome: 'approved',
  approver: 'human',
  permission_mode: 'default',
  position: null,
};

/** @returns {ActivityNode[]} */
export function nodes() {
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
export function payload(session = {}, over = {}) {
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

/** @typedef {import('../../../src/osprey/interfaces/web_terminal/static/js/activity-log-rows.js').UiState} UiState */

/**
 * The page state a renderer reads, with the given fields over a fresh page's.
 * @param {Partial<UiState>} [over]
 * @returns {UiState}
 */
export function uiState(over = {}) {
  return {
    view: 'timeline',
    filter: 'all',
    scope: 'all',
    q: '',
    sort: { k: 'ts', dir: 1 },
    open: new Set(),
    closed: new Set(),
    full: new Map(),
    ...over,
  };
}

/**
 * A renderer's HTML string as a detached element to query.
 * @param {string} html
 * @returns {HTMLDivElement}
 */
export function dom(html) {
  const host = document.createElement('div');
  host.innerHTML = html;
  return host;
}

/**
 * The trimmed text of every match of *selector* under *root*.
 * @param {ParentNode} root
 * @param {string} selector
 * @returns {string[]}
 */
export function texts(root, selector) {
  return [...root.querySelectorAll(selector)].map((node) => (node.textContent ?? '').trim());
}
