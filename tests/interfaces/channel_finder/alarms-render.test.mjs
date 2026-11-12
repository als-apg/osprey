// @ts-check
/**
 * The Alarms markup and its pure helpers (alarms-render.js): filters, counts,
 * the tree, times, the feed chip, and escaping.
 *   npx vitest run tests/interfaces/channel_finder/alarms-render.test.mjs
 */

import { test, expect, describe } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import {
  activeSummary,
  bandHtml,
  bannerHtml,
  buildTree,
  feedChip,
  filterRows,
  listHtml,
  sinceHtml,
  sinceText,
  tileCounts,
  treeHtml,
} from '../../../src/osprey/interfaces/channel_finder/static/js/alarms-render.js';

const ROUTES = JSON.parse(
  readFileSync(join(import.meta.dirname, 'fixtures/directory_routes.json'), 'utf8')
);
const ALARMS = ROUTES.alarms;
const [MAJOR_ROW, MINOR_ROW] = ALARMS.rows;

/** @param {any[]} rows */
const pvs = (rows) => rows.map((row) => row.pv);

describe('filterRows', () => {
  test('no filter keeps every row in the served order', () => {
    expect(pvs(filterRows(ALARMS.rows, { severity: null, node: null }))).toEqual([
      MAJOR_ROW.pv,
      MINOR_ROW.pv,
    ]);
  });

  test('a severity, a flag and a tree node each remove rows', () => {
    expect(pvs(filterRows(ALARMS.rows, { severity: 'MINOR', node: null }))).toEqual([MINOR_ROW.pv]);
    expect(pvs(filterRows(ALARMS.rows, { severity: 'latched', node: null }))).toEqual([
      MAJOR_ROW.pv,
    ]);
    expect(pvs(filterRows(ALARMS.rows, { severity: 'unack', node: null }))).toEqual([MAJOR_ROW.pv]);
    expect(pvs(filterRows(ALARMS.rows, { severity: null, node: '/Demo/SR/RF' }))).toEqual([
      MAJOR_ROW.pv,
    ]);
  });

  test('a node matches whole segments, not a prefix of one', () => {
    expect(filterRows(ALARMS.rows, { severity: null, node: '/Demo/SR/R' })).toEqual([]);
  });

  test('an _ACK severity filters under its severity', () => {
    const row = { ...MINOR_ROW, severity: 'MINOR_ACK' };
    expect(filterRows([row], { severity: 'MINOR', node: null })).toHaveLength(1);
  });
});

describe('counts', () => {
  test('tiles count the severities and the flags', () => {
    expect(tileCounts(ALARMS)).toEqual({
      MAJOR: 1,
      INVALID: 0,
      MINOR: 1,
      UNDEFINED: 0,
      unack: 1,
      latched: 1,
    });
  });

  test('a count that is not a number counts nothing', () => {
    const counts = tileCounts({ ...ALARMS, counts: { MAJOR: '<b>1</b>', MINOR_ACK: 2 } });
    expect(counts.MAJOR).toBe(0);
    expect(counts.MINOR).toBe(2);
  });

  test('the active summary leaves OK out and names the worst', () => {
    expect(activeSummary(ALARMS)).toEqual({ active: 2, worst: 'MAJOR' });
    expect(activeSummary({ ...ALARMS, counts: { MAJOR: 2, INVALID: 1 } })).toEqual({
      active: 3,
      worst: 'INVALID',
    });
    expect(activeSummary({ ...ALARMS, counts: { OK: 4 } })).toEqual({ active: 0, worst: 'NONE' });
  });
});

describe('buildTree', () => {
  test('the path segments, without the PV, nested and sorted', () => {
    const nodes = buildTree(ALARMS.rows);
    expect(nodes.map((node) => [node.path, node.depth, node.worst, node.count])).toEqual([
      ['/Demo', 0, 'MAJOR', 2],
      ['/Demo/SR', 1, 'MAJOR', 2],
      ['/Demo/SR/RF', 2, 'MAJOR', 1],
      ['/Demo/SR/Vacuum', 2, 'MINOR', 1],
    ]);
  });

  test('no path on any row is no tree', () => {
    expect(
      buildTree(ALARMS.rows.map((/** @type {any} */ row) => ({ ...row, path: null })))
    ).toEqual([]);
  });

  test('a node name is text, never markup', () => {
    const row = { ...MAJOR_ROW, path: '/<img src=x onerror=alert(1)>/X' };
    const host = document.createElement('div');
    host.innerHTML = treeHtml(buildTree([row]), null);
    expect(host.querySelector('img')).toBeNull();
    expect(host.textContent).toContain('<img src=x onerror=alert(1)>');
  });
});

describe('times and the feed', () => {
  const now = Date.parse('2026-11-06T12:40:00.000Z');

  test.each([
    ['2026-11-06T12:39:55.000Z', '5 s'],
    ['2026-11-06T12:02:00.000Z', '38 min'],
    ['2026-11-06T10:40:00.000Z', '2 h'],
    ['2026-11-06T10:35:00.000Z', '2 h 5 min'],
    ['2026-11-04T09:40:00.000Z', '2 d 3 h'],
    [null, ''],
    ['not a time', ''],
  ])('%s is %s ago', (time, text) => {
    expect(sinceText(time, now)).toBe(text);
  });

  test('a time becomes a placeholder the view ticks, never a clock reading', () => {
    expect(sinceHtml('2026-11-06T12:39:56.000Z')).toBe(
      '<span data-since="2026-11-06T12:39:56.000Z"></span>'
    );
    expect(sinceHtml(null)).toBe('');
    expect(sinceHtml('not a time')).toBe('');
  });

  test('a live feed names its last contact, a stale one says so', () => {
    expect(feedChip({ ...ALARMS, last_contact: '2026-11-06T12:39:56.000Z' })).toEqual({
      html: 'feed live · last contact <span data-since=""></span> ago',
      tone: 'ok',
    });
    expect(feedChip({ ...ALARMS, last_contact: null })).toEqual({ html: 'feed live', tone: 'ok' });
    expect(feedChip({ ...ALARMS, stale: true })).toEqual({ html: 'feed stale', tone: 'warn' });
  });

  test('the band does not carry the last contact, which the view writes in place', () => {
    const at = (/** @type {string} */ time) => bandHtml({ ...ALARMS, last_contact: time }, null);
    expect(at('2026-11-06T12:39:56.000Z')).toBe(at('2026-11-06T12:40:01.000Z'));
    expect(at('2026-11-06T12:39:56.000Z')).not.toContain('2026-11-06');
  });

  test('the banner names the last contact when known', () => {
    expect(bannerHtml(null)).not.toContain('No contact since');
    expect(bannerHtml('2026-11-06T12:00:00.000Z')).toContain('No contact since');
  });
});

describe('listHtml', () => {
  test('rows keep the given order and their names are text', () => {
    const rows = [
      { ...MAJOR_ROW, pv: 'B', severity: 'INVALID' },
      { ...MAJOR_ROW, pv: '<b>A</b>', severity: 'MINOR' },
    ];
    const host = document.createElement('div');
    host.innerHTML = listHtml(rows, { selected: 'B', values: null, stale: false, cut: null });
    const shown = [...host.querySelectorAll('tbody tr')].map(
      (tr) => /** @type {HTMLElement} */ (tr).dataset.name
    );
    expect(shown).toEqual(['B', '<b>A</b>']);
    expect(host.querySelector('tbody b')).toBeNull();
    expect(host.querySelector('tr[aria-selected="true"]')?.getAttribute('data-name')).toBe('B');
  });

  test('a cut answer says how many of the active alarms it shows', () => {
    const model = { selected: null, values: null, stale: false };
    const host = document.createElement('div');
    host.innerHTML = listHtml([MAJOR_ROW], { ...model, cut: { served: 1, active: 1200 } });
    expect(host.querySelector('.alarms-cut')?.textContent).toBe(
      'Worst 1 of 1,200 active alarms shown'
    );
    host.innerHTML = listHtml([MAJOR_ROW], { ...model, cut: null });
    expect(host.querySelector('.alarms-cut')).toBeNull();
  });

  test('the For cell holds the start time to tick, or a dash without one', () => {
    const rows = [
      { ...MAJOR_ROW, pv: 'A' },
      { ...MAJOR_ROW, pv: 'B', time: null },
    ];
    const host = document.createElement('div');
    host.innerHTML = listHtml(rows, { selected: null, values: null, stale: false, cut: null });
    const [a, b] = [...host.querySelectorAll('.alarms-for')];
    expect(a.querySelector('[data-since]')?.getAttribute('data-since')).toBe(MAJOR_ROW.time);
    expect(b.textContent).toBe('—');
  });
});
