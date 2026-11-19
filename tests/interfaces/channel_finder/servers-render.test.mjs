// @ts-check
/**
 * The IOCs markup (servers-render.js): times and ages, the outcome statements,
 * the count, the header, the band, the filters and the table, drawn from the
 * shared route fixture's fleet answer.
 *   npx vitest run tests/interfaces/channel_finder/servers-render.test.mjs
 */

import './epics-words.mjs';
import { test, expect, describe } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import {
  FINDINGS,
  STATES,
  ageText,
  bandHtml,
  countHtml,
  filterRows,
  headerHtml,
  outcomeStatement,
  skeletonHtml,
  tableHtml,
  timeText,
} from '../../../src/osprey/interfaces/channel_finder/static/js/servers-render.js';

const ROUTES = JSON.parse(
  readFileSync(join(import.meta.dirname, 'fixtures/directory_routes.json'), 'utf8')
);

/** The fixture's snapshot: demo-ioc-rf silent and expected running, then demo-ioc-bpm alive. */
const SNAPSHOT = ROUTES.fleet.snapshot;
/** @type {any[]} */
const ROWS = SNAPSHOT.servers;

/** @param {string} html */
function dom(html) {
  const host = document.createElement('div');
  host.innerHTML = html;
  return host;
}

describe('times', () => {
  test('timeText writes a UTC time to the second', () => {
    expect(timeText(1700000000)).toBe('2023-11-14 22:13:20 UTC');
  });

  test('ageText is the duration between the epoch and now', () => {
    expect(ageText(1700000000, 1700000000 * 1000 + 90_000)).toBe('1 min');
    expect(ageText(1700000000, 1700000000 * 1000 + 5_000)).toBe('5 s');
  });
});

describe('outcomeStatement', () => {
  test('each non-ok outcome has its own message and says nothing about IOCs running', () => {
    const messages = ['no_snapshot', 'not_refreshed', 'cannot_list', 'stale', 'unavailable'].map(
      (outcome) => outcomeStatement(/** @type {any} */ ({ outcome, target: 'live' }))
    );
    expect(messages.map((m) => m.message)).toEqual([
      'No snapshot yet',
      'Target not refreshed',
      'Cannot list the IOCs',
      'Snapshot stale',
      'server fleet unavailable',
    ]);
    for (const { detail } of messages) {
      expect(detail).toContain('target live');
      expect(detail).toContain('This says nothing about whether any IOC runs.');
    }
  });

  test('a reason leads the detail of cannot_list', () => {
    const { detail } = outcomeStatement(
      /** @type {any} */ ({ outcome: 'cannot_list', target: 'live', reason: 'directory down' })
    );
    expect(detail.startsWith('directory down. ')).toBe(true);
  });

  test('a stale answer names when its snapshot was taken, and only when it knows', () => {
    const taken = outcomeStatement(
      /** @type {any} */ ({
        outcome: 'stale',
        target: 'live',
        snapshot: { generated_at: 1700000000 },
      })
    );
    expect(taken.detail).toContain('it was taken at 2023-11-14 22:13:20 UTC');
    const unknown = outcomeStatement(/** @type {any} */ ({ outcome: 'stale', target: 'live' }));
    expect(unknown.detail).not.toContain('taken at');
  });
});

describe('skeleton and count', () => {
  test('the skeleton carries the header, band, state and list hosts', () => {
    const root = dom(skeletonHtml());
    for (const cls of ['servers-header', 'servers-band', 'servers-state', 'servers-list']) {
      expect(root.querySelector(`.${cls}`)).not.toBeNull();
    }
    expect(root.querySelector('.servers-header')?.hasAttribute('hidden')).toBe(true);
  });

  test('the count says "of" only when rows are filtered out', () => {
    expect(countHtml(1200, 1200)).toBe('<strong>1,200</strong> IOCs');
    expect(countHtml(3, 1200)).toBe('<strong>3</strong> of 1,200 IOCs');
  });
});

describe('headerHtml', () => {
  test('names the target, the time taken, an age to tick, the read path and each source', () => {
    const root = dom(headerHtml(SNAPSHOT));
    expect(root.querySelector('.servers-taken strong')?.textContent).toBe('live');
    expect(root.textContent).toContain('2023-11-14 22:13:20 UTC');
    expect(root.querySelector('[data-age]')?.getAttribute('data-age')).toBe('1700000000');
    const path = root.querySelector('[data-role="read-path"]');
    expect(path?.className).toContain('osprey-chip--ok');
    expect(path?.getAttribute('title')).toContain('SR:PROBE');
    expect(root.querySelector('[data-source="directory"]')?.textContent).toBe('directory: ok');
  });

  test('a down read path says every IOC is unknown; a missing one is unverified', () => {
    const down = dom(
      headerHtml({ ...SNAPSHOT, read_path: { status: 'down', channel: 'SR:PROBE' } })
    );
    expect(down.querySelector('[data-role="read-path"]')?.textContent).toBe(
      'read path down: every IOC unknown'
    );
    const none = dom(headerHtml({ ...SNAPSHOT, read_path: undefined }));
    expect(none.querySelector('[data-role="read-path"]')?.textContent).toBe('read path unverified');
  });

  test('a failed source carries its reason, escaped', () => {
    const root = dom(
      headerHtml({
        ...SNAPSHOT,
        sources: [{ name: 'roster', outcome: 'cannot_read', reason: '<b>gone</b>' }],
      })
    );
    const chip = root.querySelector('[data-source="roster"]');
    expect(chip?.className).toContain('osprey-chip--warn');
    expect(chip?.textContent).toBe('roster: cannot_read · <b>gone</b>');
    expect(chip?.querySelector('b')).toBeNull();
  });
});

describe('bandHtml', () => {
  test('one tile per state and per finding, in order, with counts and the pressed one marked', () => {
    const root = dom(bandHtml(SNAPSHOT, { pressed: 'state:silent', grouped: false }));
    const tiles = [...root.querySelectorAll('[data-tile]')];
    expect(tiles.map((t) => t.getAttribute('data-tile'))).toEqual([
      ...STATES.map(([key]) => `state:${key}`),
      ...FINDINGS.map(([key]) => `finding:${key}`),
    ]);
    const silent = root.querySelector('[data-tile="state:silent"]');
    expect(silent?.getAttribute('aria-pressed')).toBe('true');
    expect(silent?.querySelector('.servers-tile-n')?.textContent).toBe('1');
    expect(root.querySelector('[data-tile="state:stalled"]')?.classList.contains('is-zero')).toBe(
      true
    );
    expect(root.querySelectorAll('[aria-pressed="true"]')).toHaveLength(1);
    expect(root.querySelector('[data-action="refresh"]')).not.toBeNull();
  });

  test('the grouping toggle names the snapshot key and is absent without one', () => {
    const grouped = dom(bandHtml(SNAPSHOT, { pressed: null, grouped: true }));
    const toggle = grouped.querySelector('[data-action="group"]');
    expect(toggle?.textContent).toBe('Group by host');
    expect(toggle?.getAttribute('aria-pressed')).toBe('true');
    const none = dom(bandHtml({ ...SNAPSHOT, group_by: null }, { pressed: null, grouped: false }));
    expect(none.querySelector('[data-action="group"]')).toBeNull();
  });
});

describe('filterRows', () => {
  test('no filter keeps every row in the served order', () => {
    expect(filterRows(ROWS, { tile: null, text: '' }).map((r) => r.name)).toEqual([
      'demo-ioc-rf',
      'demo-ioc-bpm',
    ]);
  });

  test('a state tile and a finding tile each keep only their rows', () => {
    expect(filterRows(ROWS, { tile: 'state:alive', text: '' }).map((r) => r.name)).toEqual([
      'demo-ioc-bpm',
    ]);
    expect(
      filterRows(ROWS, { tile: 'finding:expected_running_not_alive', text: '' }).map((r) => r.name)
    ).toEqual(['demo-ioc-rf']);
  });

  test('the text matches name or host, case-insensitively, and combines with a tile', () => {
    expect(filterRows(ROWS, { tile: null, text: '  HOST-01 ' }).map((r) => r.name)).toEqual([
      'demo-ioc-bpm',
    ]);
    expect(filterRows(ROWS, { tile: 'state:silent', text: 'bpm' })).toEqual([]);
  });
});

describe('tableHtml', () => {
  test('one row per IOC: a link to its page, host, state chip, age, findings, directory, sources', () => {
    const root = dom(tableHtml(ROWS, null));
    expect(root.querySelectorAll('tbody')).toHaveLength(1);
    const rf = root.querySelector('tr[data-name="demo-ioc-rf"]');
    expect(rf?.querySelector('a')?.getAttribute('href')).toContain('demo-ioc-rf');
    const chip = rf?.querySelector('[data-state="silent"]');
    expect(chip?.className).toContain('osprey-chip--err');
    expect(chip?.getAttribute('title')).toBe('heartbeat did not answer');
    expect(rf?.querySelector('[data-age]')?.getAttribute('data-age')).toBe('1699999400');
    expect(rf?.querySelector('[data-finding]')?.textContent).toBe('Expected running, not alive');
    const cells = [...(rf?.querySelectorAll('td') ?? [])].map((td) => td.textContent);
    expect(cells[1]).toBe('ioc-host-02');
    expect(cells[5]).toBe('Active');
    expect(cells[6]).toBe('directory');
  });

  test('an unlisted IOC says so; a missing host and since are dashes; names are escaped', () => {
    const root = dom(
      tableHtml(
        [{ ...ROWS[0], name: '<i>x</i>', host: null, since: null, listed: false, findings: [] }],
        null
      )
    );
    const cells = [...root.querySelectorAll('tbody td')].map((td) => td.textContent);
    expect(cells[0]).toBe('<i>x</i>');
    expect(cells[1]).toBe('—');
    expect(cells[3]).toBe('—');
    expect(cells[5]).toBe('not listed');
    expect(root.querySelector('tbody i')).toBeNull();
  });

  test('grouping gathers rows under each group heading in first-appearance order', () => {
    const rows = [
      { ...ROWS[0], host: 'b' },
      { ...ROWS[1], host: 'a' },
      { ...ROWS[0], name: 'third', host: 'b' },
      { ...ROWS[1], name: 'nohost', host: null },
    ];
    const root = dom(tableHtml(rows, 'host'));
    const bodies = [...root.querySelectorAll('tbody')];
    expect(bodies.map((b) => b.getAttribute('data-group'))).toEqual(['b', 'a', '']);
    expect(
      [...bodies[0].querySelectorAll('tr[data-name]')].map((r) => r.getAttribute('data-name'))
    ).toEqual(['demo-ioc-rf', 'third']);
    expect(bodies[0].querySelector('.osprey-count')?.textContent).toBe('2');
    expect(bodies[2].querySelector('.servers-group th')?.textContent).toContain('no host');
  });
});
