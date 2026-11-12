// @ts-check
/**
 * Unit tests for the Activity page's Tool calls, Safety and Artifacts views
 * (web_terminal/static/js/activity-log-views.js):
 *   npx vitest run tests/interfaces/web_terminal/activity-log-views.test.mjs
 */
import { afterEach, beforeEach, describe, expect, test } from 'vitest';

import { summarize } from '../../../src/osprey/interfaces/web_terminal/static/js/activity-log-model.js';
import {
  renderArtifacts,
  renderSafety,
  renderTools,
} from '../../../src/osprey/interfaces/web_terminal/static/js/activity-log-views.js';
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

describe('renderTools', () => {
  test('every call is a table row, in record order', () => {
    const tools = view(renderTools);
    expect(texts(tools, 'thead th')).toEqual([
      'Time ↑',
      'Tool',
      'Arguments',
      'Agent',
      'Decision',
      'Duration',
      'Result',
      '',
    ]);
    const rows = [...tools.querySelectorAll('tr.row')].map(
      (row) => /** @type {HTMLElement} */ (row).dataset.toggle
    );
    expect(rows).toEqual(['r1', 'a1', 'w1', 'f1', 'o1']);
  });

  test('a row: time, tool, arguments, agent, decision, duration, result size', () => {
    const tools = view(renderTools);
    const cells = texts(qs(tools, 'tr.row[data-toggle="w1"]'), 'td');
    expect(cells.slice(0, 5)).toEqual([
      '10:00:01',
      'controlschannel_write',
      'channel=SR:HCM1:SP  value=0.5',
      'logbook-search',
      'Approved by operator · 17.4 s',
    ]);
    // The name keeps to its column; the tooltip carries a long one whole.
    const name = /** @type {HTMLElement} */ (qs(tools, 'tr.row[data-toggle="w1"] td.act-name'));
    expect(name.title).toBe('mcp__controls__channel_write');
    expect(texts(qs(tools, 'tr.row[data-toggle="r1"]'), 'td').slice(3, 7)).toEqual([
      'main',
      'Allowed',
      '12 ms',
      '2 B',
    ]);
    expect(texts(qs(tools, 'tr.row[data-toggle="a1"]'), 'td')[2]).toBe('Find the last orbit bump');
    expect(texts(qs(tools, 'tr.row[data-toggle="f1"]'), 'td')[4]).toBe('not recorded');
  });

  test('scopes carry their counts and narrow the table', () => {
    const tools = view(renderTools, payload(), { scope: 'mcp' });
    expect(texts(tools, '[data-scope]')).toEqual([
      'All5',
      'Control system and services2',
      'Built in3',
    ]);
    expect(qs(tools, '[data-scope][aria-pressed="true"]').dataset.scope).toBe('mcp');
    expect(tools.querySelectorAll('tr.row')).toHaveLength(2);
    expect(qs(tools, '.act-sub .hint').textContent).toBe('2 of 5 calls');
  });

  test('sorting by duration, longest first', () => {
    const p = sessionOf([
      call('a', { duration_ms: 5 }),
      call('b', { duration_ms: 500 }),
      call('c', { duration_ms: null, elapsed_ms: 50 }),
    ]);
    const tools = view(renderTools, p, { sort: { k: 'dur', dir: -1 } });
    const rows = [...tools.querySelectorAll('tr.row')].map(
      (row) => /** @type {HTMLElement} */ (row).dataset.toggle
    );
    expect(rows).toEqual(['b', 'c', 'a']);
    expect(qs(tools, 'th[data-sort="dur"]').textContent).toBe('Duration ↓');
    expect(qs(tools, 'th[data-sort="ts"]').textContent).toBe('Time');
  });

  test('a search narrows the table; nothing left says so', () => {
    expect(view(renderTools, payload(), { q: 'hcm1' }).querySelectorAll('tr.row')).toHaveLength(1);
    const none = view(renderTools, payload(), { q: 'zzz' });
    expect(qs(none, '.osprey-state-message').textContent).toBe('No tool call matches');
    expect(none.querySelector('table')).toBeNull();
  });

  test('an open row is followed by its detail', () => {
    const tools = view(renderTools, payload(), { open: new Set(['w1']) });
    const row = qs(tools, 'tr.row[data-toggle="w1"]');
    expect(row.getAttribute('aria-expanded')).toBe('true');
    const detail = /** @type {HTMLElement} */ (row.nextElementSibling);
    expect(detail.classList.contains('detail')).toBe(true);
    expect(qs(detail, 'td').getAttribute('colspan')).toBe('8');
    expect(detail.querySelector('.act-detail')).not.toBeNull();
  });

  test('a tool name is escaped', () => {
    const tools = view(renderTools, sessionOf([call('x', { name: SCRIPT, tool: SCRIPT })]));
    expect(tools.querySelector('script')).toBeNull();
  });
});

describe('renderSafety', () => {
  test('no decision and no gate entry in the record', () => {
    const safety = view(
      renderSafety,
      sessionOf([call('r1', { verdict: 'none' }), call('r2', { verdict: 'none' })])
    );
    expect(qs(safety, '.osprey-state-message').textContent).toBe(
      'No safety decisions in the record for this session'
    );
    expect(safety.querySelector('.act-cards')).toBeNull();
  });

  test('three counts: allowed by configuration, asked, denied', () => {
    const safety = view(renderSafety);
    expect(texts(safety, '.act-card h3')).toEqual([
      'Allowed by configuration',
      'Asked the operator',
      'Denied',
    ]);
    expect(texts(safety, '.act-card .act-card-n')).toEqual(['2', '1', '0']);
    expect(texts(safety, '.act-card p')).toEqual([
      "Calls the deployment's rules permit without asking.",
      '1 approved, 0 declined.',
      'No call was refused.',
    ]);
  });

  test('an approval: what was asked, how long the operator took, and the way to it', () => {
    const approval = qs(view(renderSafety), '.act-approval');
    expect(qs(approval, 'h3').textContent).toContain('channel_write');
    const labels = texts(approval, '.act-kv dt');
    const values = texts(approval, '.act-kv dd');
    expect(values[labels.indexOf('Arguments')]).toBe('channel=SR:HCM1:SP  value=0.5');
    expect(values[labels.indexOf('Target')]).toBe('virtual');
    expect(values[labels.indexOf('Agent')]).toBe('logbook-search');
    expect(values[labels.indexOf('Waited')]).toBe('17.4 s for alice');
    expect(qs(approval, 'button[data-jump]').dataset.jump).toBe('w1');
    expect(approval.querySelector('.act-trail')).not.toBeNull();
  });

  test('the calls a server gate filed an entry for', () => {
    const safety = view(renderSafety);
    const rows = [...safety.querySelectorAll('table.act-table tr.row')];
    expect(rows.map((row) => /** @type {HTMLElement} */ (row).dataset.jump)).toEqual(['w1', 'f1']);
    expect(texts(rows[0], 'td')[3]).toBe('controls');
    expect(texts(rows[1], 'td')[3]).toBe('bluesky');
    expect(texts(safety, '.act-h2 .osprey-count')).toContain('2');
  });

  test('gate entries without any approval', () => {
    const gated = call('g1', {
      server: 'controls',
      tool: 'channel_read',
      posture: 'writes',
      trail: [
        { ts: '2026-11-12T10:00:02+00:00', surface: 'controls', decision: 'allowed', reason: '' },
      ],
    });
    const safety = view(renderSafety, sessionOf([gated]));
    expect(safety.querySelector('.act-approval')).toBeNull();
    expect(safety.textContent).toContain('No call needed an approval.');
    expect(texts(qs(safety, 'tr.row[data-jump="g1"]'), 'td')[4]).toBe('enabled');
  });
});

describe('renderArtifacts', () => {
  /** @type {ArtifactRow} */
  const plot = {
    id: 'art-1',
    title: 'Orbit <b>bump</b>',
    call: 'w1',
    tool: 'channel_write',
    agent: 'logbook-search',
    ts: '2026-11-12T10:00:42+00:00',
    form: 'plot',
    form_label: 'Plot',
    type: 'plot_html',
    size: 2117,
  };

  test('none', () => {
    const artifacts = view(renderArtifacts);
    expect(qs(artifacts, '.osprey-state-message').textContent).toBe('No artifacts in this session');
  });

  test('an artifact: title, what it is, who saved it, the way to its call, and Open', () => {
    const card = qs(view(renderArtifacts, payload({}, { artifacts: [plot] })), '.act-art');
    expect(card.querySelector('b')).toBeNull();
    expect(qs(card, 'h3').textContent).toBe('Orbit <b>bump</b>');
    expect(texts(card, 'p span')).toEqual([
      'Plot',
      'plot_html',
      '2.1 KB',
      'Saved 10:00:42 by channel_write in subagent logbook-search',
    ]);
    const jump = qs(card, 'button[data-jump]');
    expect(jump.dataset.jump).toBe('w1');
    expect(jump.textContent).toBe('Show in timeline');
    const open = qs(card, 'button[data-open-artifact]');
    expect(open.dataset.openArtifact).toBe('art-1');
    expect(open.textContent).toBe('Open');
  });

  test('an artifact whose call the record does not hold has no way to the timeline', () => {
    const loose = { ...plot, call: 'gone', tool: '', agent: '' };
    const card = qs(view(renderArtifacts, payload({}, { artifacts: [loose] })), '.act-art');
    expect(card.querySelector('[data-jump]')).toBeNull();
    expect(texts(card, 'p span')[3]).toBe('Saved 10:00:42');
  });
});
