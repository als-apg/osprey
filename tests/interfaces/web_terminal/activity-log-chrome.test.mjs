// @ts-check
/**
 * Unit tests for the Activity page's chrome: header, stat strip, session list
 * and record state (web_terminal/static/js/activity-log-chrome.js):
 *   npx vitest run tests/interfaces/web_terminal/activity-log-chrome.test.mjs
 */
import { afterEach, beforeEach, describe, expect, test } from 'vitest';

import {
  recordBanner,
  recordChip,
  recordPopover,
  renderHeader,
  renderLiveState,
  renderSidebar,
  renderStats,
  viewCounts,
} from '../../../src/osprey/interfaces/web_terminal/static/js/activity-log-chrome.js';
import { summarize } from '../../../src/osprey/interfaces/web_terminal/static/js/activity-log-model.js';
import { call, dom, payload, texts } from '../_support/activity-log.mjs';
import { qs } from '../_support/dom.mjs';
import { stampFacilityZone } from '../_support/facility-zone.mjs';

const SCRIPT = '<script>alert(1)</script>';

beforeEach(() => stampFacilityZone('UTC'));
afterEach(() => stampFacilityZone(null));

/** The date of the fixture session as the viewer's locale writes it. */
const DAY = new Intl.DateTimeFormat(undefined, {
  day: 'numeric',
  month: 'short',
  year: 'numeric',
  timeZone: 'UTC',
}).format(new Date('2026-11-12T10:00:00Z'));

describe('recordChip', () => {
  test('one chip per record state', () => {
    expect(recordChip('complete')).toEqual({
      tone: 'ok',
      label: 'Record complete',
      icon: 'database',
    });
    expect(recordChip('incomplete')).toEqual({
      tone: 'warn',
      label: 'Record incomplete',
      icon: 'triangle-alert',
    });
    expect(recordChip('unavailable').tone).toBe('err');
    expect(recordChip('unavailable').label).toBe('Record unavailable');
  });
});

describe('recordPopover', () => {
  test('what the record holds of the session, and until when', () => {
    expect(recordPopover(payload())).toEqual({
      title: 'Agent record',
      text: 'Everything this session wrote has reached the agent record.',
      rows: [
        { label: 'Entries', value: 31 },
        { label: 'Holds', value: '2 prompt, 5 tool call' },
        { label: 'Kept until', value: '2026-11-26, then in the archive' },
        { label: 'Session', value: 'aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee', copy: true },
      ],
    });
  });

  test('kinds with no entries are left out', () => {
    const content = recordPopover(payload({ kinds: { prompt: 1, tool_result: 0, audit: 3 } }));
    expect(content.rows?.[1]).toEqual({ label: 'Holds', value: '1 prompt, 3 audit' });
  });

  test('a record that is behind, or does not answer', () => {
    expect(recordPopover(payload({}, { record: 'incomplete' })).text).toBe(
      'The record is behind this session. Some entries have not arrived yet.'
    );
    expect(recordPopover(payload({}, { record: 'unavailable' })).text).toBe(
      'The record service did not answer.'
    );
  });

  test('no session, or one with no entries yet, has no rows it cannot fill', () => {
    expect(recordPopover(payload({}, { session: null })).rows).toEqual([]);
    const fresh = recordPopover(payload({ start: null, end: null, entries: 0, kinds: {} }));
    expect(fresh.rows).toEqual([
      { label: 'Entries', value: 0 },
      { label: 'Session', value: 'aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee', copy: true },
    ]);
  });
});

describe('recordBanner', () => {
  test('only an incomplete record has one', () => {
    expect(recordBanner('complete')).toBe('');
    expect(recordBanner('unavailable')).toBe('');
    const banner = qs(dom(recordBanner('incomplete')), '.act-banner');
    expect(banner.textContent).toContain('Entries may be missing.');
    expect(banner.querySelector('svg')).not.toBeNull();
  });
});

describe('viewCounts', () => {
  test('the counts on the view switcher', () => {
    const p = payload({}, { artifacts: [] });
    expect(viewCounts(p, summarize(p))).toEqual({ tools: 5, safety: 1, artifacts: 0 });
  });
});

describe('renderLiveState', () => {
  test('a live session', () => {
    const chip = qs(dom(renderLiveState(payload({ state: 'live' }).session)), '.osprey-chip--info');
    expect(chip.textContent).toBe('Live');
    expect(chip.querySelector('.act-live-dot')).not.toBeNull();
  });

  test('a session that is not live shows its last entry', () => {
    const state = dom(renderLiveState(payload().session));
    expect(state.textContent).toBe('Last entry 10:02');
    expect(state.querySelector('.osprey-chip')).toBeNull();
  });

  test('nothing to say without a session or an entry', () => {
    expect(renderLiveState(null)).toBe('');
    expect(renderLiveState(payload({ end: null }).session)).toBe('');
  });
});

describe('renderHeader', () => {
  test('title, operator, range, duration, target and posture', () => {
    const p = payload();
    const head = qs(dom(renderHeader(p, summarize(p))), '.act-head');
    expect(qs(head, 'h1').textContent).toBe('Read the beam current');
    expect(texts(head, '.act-meta > span:not(.sep)')).toEqual([
      'alice',
      `${DAY}, 10:00 to 10:02 UTC`,
      '2 min 5 s',
      'Target virtual',
      'Writes enabled',
    ]);
    expect(qs(head, '.act-meta .osprey-chip').classList.contains('osprey-chip--warn')).toBe(true);
  });

  test('a live session has a start and no end', () => {
    const p = payload({ state: 'live' });
    const head = dom(renderHeader(p, summarize(p)));
    expect(texts(head, '.act-meta > span:not(.sep)')[1]).toBe(`${DAY}, since 10:00 UTC`);
  });

  test('another posture is named as it is; none is left out', () => {
    const readOnly = payload({ posture: 'read_only' });
    const chip = qs(dom(renderHeader(readOnly, summarize(readOnly))), '.act-meta .osprey-chip');
    expect(chip.textContent).toBe('read_only');
    expect(chip.classList.contains('osprey-chip--warn')).toBe(false);
    const none = payload({ posture: null, target: null });
    const head = dom(renderHeader(none, summarize(none)));
    expect(head.querySelector('.osprey-chip')).toBeNull();
    expect(head.textContent).not.toContain('Target');
  });

  test('a session with no entries yet: title and operator only', () => {
    const fresh = payload({ start: null, end: null, target: null, posture: null }, { nodes: [] });
    const head = dom(renderHeader(fresh, summarize(fresh)));
    expect(texts(head, '.act-meta > span:not(.sep)')).toEqual(['alice']);
  });

  test('a title is escaped', () => {
    const p = payload({ title: SCRIPT });
    const head = dom(renderHeader(p, summarize(p)));
    expect(head.querySelector('script')).toBeNull();
    expect(qs(head, 'h1').textContent).toBe(SCRIPT);
  });

  test('the link to this session can be copied', () => {
    const p = payload();
    const button = qs(dom(renderHeader(p, summarize(p))), 'button[data-act="copy-link"]');
    expect(button.getAttribute('title')).toBe('Copy link to this session');
    expect(button.getAttribute('aria-label')).toBe('Copy link to this session');
  });

  test('no session, no header', () => {
    const p = payload({}, { session: null, nodes: [] });
    expect(renderHeader(p, summarize(p))).toBe('');
  });
});

describe('renderStats', () => {
  test('eight cells, each a way into a view', () => {
    const p = payload();
    const stats = dom(renderStats(p, summarize(p)));
    const cells = [...stats.querySelectorAll('button.act-stat')].map((cell) => [
      qs(cell, '.act-stat-l').textContent,
      qs(cell, '.act-stat-n').textContent,
      /** @type {HTMLElement} */ (cell).dataset.go,
    ]);
    expect(cells).toEqual([
      ['Prompts', '2', 'timeline:conversation'],
      ['Tool calls', '5', 'tools:all'],
      ['Approvals', '1', 'safety:all'],
      ['Failed calls', '1', 'timeline:errors'],
      ['Subagents', '2', 'timeline:all'],
      ['Artifacts', '0', 'artifacts:all'],
      ['Model time', '4.5 s', 'timeline:all'],
      ['Model cost', '$0.17', 'timeline:all'],
    ]);
  });

  test('approvals and failures are toned only when there are any', () => {
    const p = payload();
    const toned = dom(renderStats(p, summarize(p)));
    expect(qs(toned, '[data-go="safety:all"]').dataset.tone).toBe('warn');
    expect(qs(toned, '[data-go="timeline:errors"]').dataset.tone).toBe('err');
    const quiet = payload({}, { nodes: [call('r1')] });
    const plain = dom(renderStats(quiet, summarize(quiet)));
    expect(qs(plain, '[data-go="safety:all"]').dataset.tone).toBe('');
    expect(qs(plain, '[data-go="timeline:errors"]').dataset.tone).toBe('');
  });

  test('nothing measured is a dash, not a zero', () => {
    const quiet = payload({}, { nodes: [call('r1')] });
    const numbers = texts(dom(renderStats(quiet, summarize(quiet))), '.act-stat-n');
    expect(numbers.slice(6)).toEqual(['–', '–']);
  });
});

describe('renderSidebar', () => {
  const NOW = Date.parse('2026-11-12T12:00:00Z');
  const rows = [
    {
      transcript_id: 'aaaa',
      title: 'Read the beam current',
      state: 'live',
      first_ts: '2026-11-12T10:00:00+00:00',
      last_ts: '2026-11-12T10:02:05+00:00',
    },
    {
      transcript_id: 'bbbb',
      title: `Orbit ${SCRIPT}`,
      state: 'idle',
      first_ts: '2026-11-12T08:00:00+00:00',
      last_ts: '2026-11-12T08:30:00+00:00',
    },
    {
      transcript_id: 'cccc',
      title: 'Yesterday shift',
      state: 'idle',
      first_ts: '2026-11-11T22:00:00+00:00',
      last_ts: '2026-11-11T23:15:00+00:00',
    },
  ];

  test('sessions grouped by day, newest day first as the server sends them', () => {
    const list = dom(renderSidebar(rows, 'bbbb', NOW));
    expect(texts(list, '.osprey-section-label')).toEqual(['Today', 'Yesterday']);
    expect(
      [...list.querySelectorAll('button.osprey-list-row')].map(
        (row) => /** @type {HTMLElement} */ (row).dataset.session
      )
    ).toEqual(['aaaa', 'bbbb', 'cccc']);
    expect(texts(list, '.osprey-list-row time')).toEqual(['10:02', '08:30', '23:15']);
  });

  test('the selection is marked, and only it', () => {
    const list = dom(renderSidebar(rows, 'bbbb', NOW));
    expect(
      [...list.querySelectorAll('.osprey-list-row')].map((row) => row.getAttribute('aria-selected'))
    ).toEqual(['false', 'true', 'false']);
  });

  test('a title is escaped, in the row and in its tooltip', () => {
    const list = dom(renderSidebar(rows, null, NOW));
    expect(list.querySelector('script')).toBeNull();
    const row = qs(list, '[data-session="bbbb"]');
    expect(qs(row, '.osprey-list-title').textContent).toBe(`Orbit ${SCRIPT}`);
    expect(row.getAttribute('title')).toBe(`Orbit ${SCRIPT}`);
  });

  test('a live session carries the live dot', () => {
    const list = dom(renderSidebar(rows, null, NOW));
    expect(qs(list, '[data-session="aaaa"]').querySelector('.act-live-dot')).not.toBeNull();
    expect(qs(list, '[data-session="bbbb"]').querySelector('.act-live-dot')).toBeNull();
  });

  test('a session the record holds no entry of yet is listed without a time', () => {
    const fresh = {
      transcript_id: 'dddd',
      title: 'New session',
      state: 'live',
      first_ts: null,
      last_ts: null,
    };
    const list = dom(renderSidebar([fresh, ...rows], 'dddd', NOW));
    const row = qs(list, '[data-session="dddd"]');
    expect(row.getAttribute('aria-selected')).toBe('true');
    expect(qs(row, 'time').textContent).toBe('');
    expect(texts(list, '.osprey-section-label')).toEqual(['Today', 'Yesterday']);
  });

  test('no sessions', () => {
    const list = dom(renderSidebar([], null, NOW));
    expect(qs(list, '.osprey-state-message').textContent).toBe('No sessions in the record');
  });
});
