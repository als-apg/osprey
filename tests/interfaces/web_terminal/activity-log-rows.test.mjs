// @ts-check
/**
 * Unit tests for the Activity page's rows
 * (web_terminal/static/js/activity-log-rows.js):
 *   npx vitest run tests/interfaces/web_terminal/activity-log-rows.test.mjs
 */
import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';

import {
  foldRow,
  iconHtml,
  serverIcon,
  stateBlock,
  statusCell,
  toolDetail,
  toolRow,
  trailList,
  verdictChip,
} from '../../../src/osprey/interfaces/web_terminal/static/js/activity-log-rows.js';
import { APPROVAL, call, dom, texts, uiState } from '../_support/activity-log.mjs';
import { qs } from '../_support/dom.mjs';
import { stampFacilityZone } from '../_support/facility-zone.mjs';

const SCRIPT = '<script>alert(1)</script>';

beforeEach(() => stampFacilityZone('UTC'));
afterEach(() => stampFacilityZone(null));

/** An approved write through the controls gate. */
function approvedWrite() {
  return call('w1', {
    name: 'mcp__controls__channel_write',
    server: 'controls',
    tool: 'channel_write',
    input: { channel: 'SR:HCM1:SP', value: 0.5 },
    verdict: 'approved',
    approval: APPROVAL,
    target: 'virtual',
    posture: 'writes',
    duration_ms: 840,
    trail: [
      { ts: '2026-11-12T10:00:27.5+00:00', surface: 'controls', decision: 'allowed', reason: '' },
      { ts: '2026-11-12T10:00:10+00:00', surface: 'controls', decision: 'ask', reason: '' },
      {
        ts: '2026-11-12T10:00:27.4+00:00',
        surface: 'hook_approval',
        decision: 'allowed',
        reason: '',
      },
    ],
  });
}

describe('iconHtml', () => {
  test('is the design-system icon as markup', () => {
    const svg = qs(dom(iconHtml('check')), 'svg');
    expect(svg.classList.contains('osprey-ic')).toBe(true);
  });

  test('a server has its icon; an unknown one the wrench', () => {
    expect(serverIcon('controls')).toBe(iconHtml('crosshair'));
    expect(serverIcon('bluesky')).toBe(iconHtml('bluesky'));
    expect(serverIcon('my_own_server')).toBe(iconHtml('wrench'));
    expect(serverIcon('')).toBe(iconHtml('wrench'));
  });
});

describe('verdictChip', () => {
  test('a call with no decision entry says "not recorded"', () => {
    const chip = dom(verdictChip(call('x', { verdict: 'none' }), true));
    expect(chip.textContent).toBe('not recorded');
    expect(qs(chip, '.act-verdict').dataset.v).toBe('none');
  });

  test('the short form shows only what the operator was part of', () => {
    expect(verdictChip(call('x', { verdict: 'none' }), false)).toBe('');
    expect(verdictChip(call('x', { verdict: 'allowed' }), false)).toBe('');
    expect(verdictChip(approvedWrite(), false)).toContain('Approved by operator');
  });

  test('allowed by the rules', () => {
    const chip = dom(verdictChip(call('x'), true));
    expect(chip.textContent).toBe('Allowed');
    expect(qs(chip, '.act-verdict').dataset.v).toBe('allowed');
  });

  test('approved carries the wait', () => {
    const chip = qs(dom(verdictChip(approvedWrite(), true)), '.osprey-chip');
    expect(chip.classList.contains('osprey-chip--warn')).toBe(true);
    expect(chip.textContent).toBe('Approved by operator · 17.4 s');
  });

  test('asked and denied', () => {
    const asked = qs(dom(verdictChip(call('x', { verdict: 'asked' }), false)), '.osprey-chip');
    expect(asked.textContent).toBe('Waiting for approval');
    expect(asked.classList.contains('osprey-chip--warn')).toBe(true);
    const denied = qs(dom(verdictChip(call('x', { verdict: 'denied' }), false)), '.osprey-chip');
    expect(denied.textContent).toBe('Denied');
    expect(denied.classList.contains('osprey-chip--err')).toBe(true);
  });
});

describe('trailList', () => {
  test('a failed tool is a server error, not a refusal', () => {
    const failed = call('f1', {
      server: 'bluesky',
      tool: 'run_plan',
      error: true,
      trail: [
        {
          ts: '2026-11-12T10:00:40+00:00',
          surface: 'bluesky',
          decision: 'refused',
          reason: 'tool_error',
        },
      ],
    });
    const list = dom(trailList(failed));
    expect(texts(list, 'li span')).toEqual([
      'Call issued by the agent',
      'The bluesky server returned an error',
    ]);
    expect(qs(list, 'li:last-child').dataset.tone).toBe('err');
    expect(list.textContent).not.toContain('Refused');
  });

  test('an approved write: asked, approved, passed the gate, executed, in time order', () => {
    const list = dom(trailList(approvedWrite()));
    expect(texts(list, 'li span')).toEqual([
      'Call issued by the agent',
      'Approval requested from the operator',
      'Approved by the operator after 17.4 s',
      'Passed the controls server gate, target virtual',
      'Executed in 840 ms',
    ]);
    expect(texts(list, 'li time')).toEqual([
      '10:00:01.000',
      '10:00:10.000',
      '10:00:27.400',
      '10:00:27.500',
      '',
    ]);
  });

  test('a refusal names the gate and its reason', () => {
    const refused = call('x', {
      verdict: 'denied',
      trail: [
        {
          ts: '2026-11-12T10:00:02+00:00',
          surface: 'controls',
          decision: 'refused',
          reason: 'limits',
        },
      ],
    });
    expect(texts(dom(trailList(refused)), 'li span')[1]).toBe(
      'Refused by the controls server gate (limits)'
    );
  });

  test('a declined approval', () => {
    const declined = call('x', {
      verdict: 'denied',
      approval: { ...APPROVAL, outcome: 'denied', wait_ms: 2000 },
    });
    expect(texts(dom(trailList(declined)), 'li span')).toContain(
      'Denied by the operator after 2.0 s'
    );
  });

  test('an approval for this call only says so', () => {
    const once = { ...approvedWrite(), source: 'user_temporary' };
    expect(texts(dom(trailList(once)), 'li span')).toContain(
      'Approved by the operator, this call only, after 17.4 s'
    );
  });

  test('a failed execution says so', () => {
    const failed = call('x', { error: true, duration_ms: 1200 });
    expect(texts(dom(trailList(failed)), 'li span')).toContain('Executed in 1.2 s, failed');
  });

  test('a gate name is escaped', () => {
    const odd = call('x', {
      trail: [
        { ts: '2026-11-12T10:00:02+00:00', surface: SCRIPT, decision: 'allowed', reason: '' },
      ],
    });
    expect(dom(trailList(odd)).querySelector('script')).toBeNull();
  });
});

describe('toolRow', () => {
  test('a tool name and an argument never reach the markup unescaped', () => {
    const row = dom(
      toolRow(call('x1', { name: SCRIPT, tool: SCRIPT, input: { q: SCRIPT } }), uiState())
    );
    expect(row.querySelector('script')).toBeNull();
    expect(qs(row, '.act-tool').textContent).toBe(SCRIPT);
    expect(qs(row, '.act-args').textContent).toBe(`q=${SCRIPT}`);
  });

  test('a call id with a quote stays one attribute', () => {
    const row = qs(dom(toolRow(call('a"b<c'), uiState())), '.act-row');
    expect(row.dataset.toggle).toBe('a"b<c');
    expect(row.id).toBe('row-a"b<c');
  });

  test('a row is a button that names its call and shows time, server, tool and arguments', () => {
    const row = qs(dom(toolRow(approvedWrite(), uiState())), 'button.act-row');
    expect(row.getAttribute('type')).toBe('button');
    expect(row.dataset.toggle).toBe('w1');
    expect(row.getAttribute('aria-expanded')).toBe('false');
    expect(qs(row, '.act-t').textContent).toBe('10:00:01');
    expect(qs(row, '.act-srv').textContent).toBe('controls');
    expect(qs(row, '.act-tool').textContent).toBe('channel_write');
    expect(qs(row, '.act-args').textContent).toBe('channel=SR:HCM1:SP  value=0.5');
    expect(row.querySelector('.act-i svg')).not.toBeNull();
    expect(qs(row, '.act-dur').textContent).toBe('840 ms');
    expect(row.querySelector('.act-ok')).not.toBeNull();
  });

  test('a built-in tool has no server label and a soft call-to-result duration', () => {
    const row = dom(toolRow(call('r1'), uiState()));
    expect(row.querySelector('.act-srv')).toBeNull();
    const duration = qs(row, '.act-dur');
    expect(duration.textContent).toBe('12 ms');
    expect(duration.classList.contains('soft')).toBe(true);
  });

  test('an error row shows the error message in place of the arguments', () => {
    const failed = call('f1', {
      input: { plan: 'count' },
      error: true,
      errmsg: 'Plan failed: Detector DCCT timed out',
    });
    const row = qs(dom(toolRow(failed, uiState())), '.act-row');
    expect(row.dataset.tone).toBe('err');
    expect(qs(row, '.act-args').textContent).toBe('Plan failed: Detector DCCT timed out');
    expect(row.querySelector('.act-bad')).not.toBeNull();
    expect(row.querySelector('.act-ok')).toBeNull();
  });

  test('a call without a result is running while the session is live', () => {
    const pending = call('p1', { result: null, running: true, elapsed_ms: null });
    const row = dom(toolRow(pending, uiState({ live: true })));
    expect(qs(row, '.act-running').textContent).toBe('running');
    expect(row.querySelector('.act-bad')).toBeNull();
    expect(row.textContent).not.toContain('–');
  });

  test('the same call in a session that is over has no result, and is not failed', () => {
    const pending = call('p1', { result: null, running: true, elapsed_ms: null });
    const row = dom(toolRow(pending, uiState()));
    expect(row.querySelector('.act-running')).toBeNull();
    expect(row.querySelector('.act-bad')).toBeNull();
    expect(qs(row, '[title="No result in the record"]').textContent).toBe('–');
  });

  test('a subagent call is completed once it returned, running until then', () => {
    const sub = { id: 'agent-1', type: 'explorer', description: '', ms: 4000, children: [] };
    const done = call('a1', { tool: 'Agent', result: null, sub });
    expect(dom(statusCell(done, true)).querySelector('.act-ok')).not.toBeNull();
    const atWork = call('a1', { tool: 'Agent', result: null, running: true, sub });
    expect(qs(dom(statusCell(atWork, true)), '.act-running').textContent).toBe('running');
    expect(dom(statusCell(atWork, true)).querySelector('.act-ok')).toBeNull();
    const left = dom(statusCell(atWork, false));
    expect(left.querySelector('.act-ok')).toBeNull();
    expect(qs(left, '[title="No result in the record"]').textContent).toBe('–');
  });

  test('a call the record holds only part of is marked incomplete', () => {
    const row = dom(toolRow(call('i1', { incomplete: true, result: null }), uiState()));
    const chip = qs(row, '.osprey-chip--warn');
    expect(chip.textContent).toBe('Incomplete');
  });

  test('an approval tints the row', () => {
    expect(qs(dom(toolRow(approvedWrite(), uiState())), '.act-row').dataset.tone).toBe('warn');
  });

  test('an open row carries its detail; a closed one does not', () => {
    const closed = dom(toolRow(call('r1'), uiState()));
    expect(closed.querySelector('.act-body-block')).toBeNull();
    const open = dom(toolRow(call('r1'), uiState({ open: new Set(['r1']) })));
    expect(qs(open, '.act-row').getAttribute('aria-expanded')).toBe('true');
    expect(open.querySelector('.act-body-block .act-detail')).not.toBeNull();
  });
});

describe('toolDetail', () => {
  test('single-line arguments are one JSON block', () => {
    const detail = dom(toolDetail(call('r1', { input: { path: '/a', n: 2 } }), uiState()));
    expect(texts(detail, 'h4')[0]).toBe('Arguments');
    expect(JSON.parse(qs(detail, 'pre.act-code').textContent ?? '')).toEqual({ path: '/a', n: 2 });
  });

  test('a multi-line argument gets its own block with its line count', () => {
    const code = 'a = 1\nb = 2\nprint(a + b)';
    const detail = dom(toolDetail(call('p1', { input: { code, timeout: 5 } }), uiState()));
    expect(texts(detail, 'h4').slice(0, 2)).toEqual(['Arguments', 'code3 lines']);
    const blocks = detail.querySelectorAll('pre.act-code');
    expect(JSON.parse(blocks[0].textContent ?? '')).toEqual({ timeout: 5 });
    expect(blocks[1].textContent).toBe(code);
  });

  test('a result that is no JSON is shown as the text it is', () => {
    const text = `<persisted-output>saved to /tmp/x</persisted-output> ${SCRIPT}`;
    const node = call('r1', { result: { text, size: 2117, cut: false, key: 'k1' } });
    const detail = dom(toolDetail(node, uiState()));
    expect(detail.querySelector('script')).toBeNull();
    expect(qs(detail, '.act-result').textContent).toBe(text);
    expect(detail.textContent).toContain('2.1 KB');
    expect(detail.querySelector('[data-full]')).toBeNull();
  });

  test('a cut result offers the whole one by its record key', () => {
    const node = call('r1', {
      result: { text: 'x'.repeat(2400), size: 12288, cut: true, key: 'cc:alice:s:r1:r' },
    });
    const detail = dom(toolDetail(node, uiState()));
    const button = qs(detail, 'button[data-full]');
    expect(button.dataset.full).toBe('r1');
    expect(button.dataset.key).toBe('cc:alice:s:r1:r');
    expect(button.textContent).toBe('Load full result (12.0 KB)');
    expect(detail.textContent).toContain('Showing the first 2400 characters');
  });

  test('a loaded full result replaces the cut one', () => {
    const node = call('r1', {
      result: { text: 'x'.repeat(2400), size: 12288, cut: true, key: 'k1' },
    });
    const detail = dom(toolDetail(node, uiState({ full: new Map([['r1', 'the whole result']]) })));
    expect(qs(detail, '.act-result').textContent).toBe('the whole result');
    expect(detail.querySelector('[data-full]')).toBeNull();
  });

  test('no result: not in the record, or not returned yet while live', () => {
    const pending = call('p1', { result: null, running: true });
    expect(dom(toolDetail(pending, uiState())).textContent).toContain(
      'No result in the record for this call.'
    );
    expect(dom(toolDetail(pending, uiState({ live: true }))).textContent).toContain(
      'The call has not returned yet.'
    );
  });

  test('an incomplete call says what is missing', () => {
    const detail = dom(toolDetail(call('i1', { incomplete: true, result: null }), uiState()));
    expect(detail.textContent).toContain('The record has not received all of this call.');
  });

  test('a subagent call has no result block of its own', () => {
    const sub = call('a1', {
      name: 'Agent',
      tool: 'Agent',
      result: null,
      sub: { id: 'agent-1', type: 'explorer', description: 'Look', ms: 10, children: [] },
    });
    expect(texts(dom(toolDetail(sub, uiState())), 'h4')).not.toContain('Result');
  });

  test('the safety facts of a call', () => {
    const detail = dom(toolDetail(approvedWrite(), uiState()));
    const facts = Object.fromEntries(
      texts(detail, '.act-kv dt').map((label, i) => [label, texts(detail, '.act-kv dd')[i]])
    );
    expect(facts).toEqual({
      Decision: 'Asked the operator, approved',
      Writes: 'Enabled for this session',
      Target: 'virtual',
      Tool: 'MCP server controls',
      Agent: 'main',
      'Call id': 'w1',
    });
    expect(detail.querySelector('.act-trail')).not.toBeNull();
  });

  test('a built-in call with no decision entry', () => {
    const detail = dom(toolDetail(call('r1', { verdict: 'none', source: null }), uiState()));
    const labels = texts(detail, '.act-kv dt');
    const values = texts(detail, '.act-kv dd');
    expect(values[labels.indexOf('Decision')]).toBe('Not recorded');
    expect(values[labels.indexOf('Tool')]).toBe('Built into the agent');
    expect(labels).not.toContain('Writes');
    expect(labels).not.toContain('Target');
    expect(detail.querySelector('.act-trail')).toBeNull();
  });

  test('allowed by configuration is worded as such', () => {
    const detail = dom(toolDetail(call('r1'), uiState()));
    expect(texts(detail, '.act-kv dd')[0]).toBe('Allowed by configuration');
  });
});

describe('foldRow', () => {
  const row = { id: 't0:1', kind: 'thinking', icon: 'bulb', ts: '2026-11-12T10:00:00+00:00' };

  test('closed: a button, and the body is never built', () => {
    const body = vi.fn(() => '<p>inside</p>');
    const fold = dom(foldRow({ ...row, label: '<span>Plan</span>', body }, uiState()));
    const button = qs(fold, 'button.act-row');
    expect(button.dataset.kind).toBe('thinking');
    expect(button.dataset.toggle).toBe('t0:1');
    expect(button.getAttribute('aria-expanded')).toBe('false');
    expect(qs(button, '.act-t').textContent).toBe('10:00:00');
    expect(body).not.toHaveBeenCalled();
    expect(fold.querySelector('.act-body-block')).toBeNull();
  });

  test('open: the body follows the row', () => {
    const fold = dom(
      foldRow(
        { ...row, label: '<span>Plan</span>', body: () => '<p>inside</p>' },
        uiState({ open: new Set(['t0:1']) })
      )
    );
    expect(qs(fold, 'button.act-row').getAttribute('aria-expanded')).toBe('true');
    expect(qs(fold, '.act-body-block').textContent).toBe('inside');
  });
});

describe('stateBlock', () => {
  test('is the design-system state with an optional action', () => {
    const block = qs(
      dom(
        stateBlock('database', 'No <b>record</b>', 'Try later', {
          label: 'Try again',
          act: 'refresh',
        })
      ),
      '.osprey-state'
    );
    expect(block.getAttribute('role')).toBe('status');
    expect(qs(block, '.osprey-state-message').textContent).toBe('No <b>record</b>');
    expect(qs(block, '.osprey-state-detail').textContent).toBe('Try later');
    const button = qs(block, 'button.osprey-btn');
    expect(button.dataset.act).toBe('refresh');
    expect(button.textContent).toBe('Try again');
  });

  test('a message alone', () => {
    const block = dom(stateBlock('check', 'No failed calls in this session'));
    expect(block.querySelector('.osprey-state-detail')).toBeNull();
    expect(block.querySelector('button')).toBeNull();
  });
});
