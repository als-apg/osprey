// @ts-check
/**
 * The rows of the sidebar's artifact lists (sidebar-artifact-rows.js):
 *   npx vitest run tests/interfaces/web_terminal/sidebar-artifact-rows.test.mjs
 *
 * Pins the two groupings (Types by artifact type, Activity by facility day),
 * the title filter, the row markup the list primitives style, the time on the
 * facility clock, the empty lines, and the ⋯ menu's rows. Expected times are
 * computed with Intl directly, in the stamped facility zone.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import { icon } from '../../../src/osprey/interfaces/design_system/static/js/icons.js';

const rows =
  await import('../../../src/osprey/interfaces/web_terminal/static/js/sidebar-artifact-rows.js');

const NOW = Date.UTC(2026, 9, 30, 12, 0);

/**
 * @param {string} id
 * @param {string} type
 * @param {string} timestamp
 * @param {object} [extra]
 */
const entry = (id, type, timestamp, extra = {}) => ({
  id,
  title: `Title ${id}`,
  artifact_type: type,
  timestamp,
  ...extra,
});

const READY = { shownId: null, loaded: true, failed: false };

/** @param {Node[]} nodes */
function texts(nodes) {
  return nodes.map((node) => /** @type {HTMLElement} */ (node).textContent);
}

/** @param {Node[]} nodes */
function heads(nodes) {
  return nodes
    .filter((node) => /** @type {HTMLElement} */ (node).classList.contains('osprey-list-head'))
    .map((node) => node.textContent);
}

/** @param {Node[]} nodes */
function rowIds(nodes) {
  return nodes
    .filter((node) => /** @type {HTMLElement} */ (node).classList.contains('osprey-list-row'))
    .map((node) => /** @type {HTMLElement} */ (node).dataset.id);
}

beforeEach(() => {
  vi.useFakeTimers({ toFake: ['Date'] });
  vi.setSystemTime(NOW);
  document.documentElement.setAttribute('data-facility-timezone', 'UTC');
});

afterEach(() => {
  vi.useRealTimers();
  document.documentElement.removeAttribute('data-facility-timezone');
  document.body.replaceChildren();
});

describe('the Types grouping', () => {
  test('groups by artifact type in a fixed order, an unknown type under Other', () => {
    const entries = [
      entry('w', 'weird', '2026-10-30T11:00:00Z'),
      entry('d', 'json', '2026-10-30T10:00:00Z'),
      entry('n', 'notebook', '2026-10-30T09:00:00Z'),
      entry('t', 'table_html', '2026-10-30T08:00:00Z'),
      entry('p', 'plot_png', '2026-10-30T07:00:00Z'),
      entry('r', 'markdown', '2026-10-30T06:00:00Z'),
    ];
    const nodes = rows.artifactRows(entries, { ...READY, view: 'types', filter: '' });
    expect(heads(nodes)).toEqual(['Reports', 'Plots', 'Tables', 'Notebooks', 'Data', 'Other']);
    expect(rowIds(nodes)).toEqual(['r', 'p', 't', 'n', 'd', 'w']);
    expect(rows.drawnOrder(entries, 'types').map((a) => a.id)).toEqual([
      'r',
      'p',
      't',
      'n',
      'd',
      'w',
    ]);
  });

  test('each group draws its kind icon; an empty group draws no head', () => {
    const nodes = rows.artifactRows([entry('p', 'plot_html', '2026-10-30T07:00:00Z')], {
      ...READY,
      view: 'types',
      filter: '',
    });
    expect(heads(nodes)).toEqual(['Plots']);
    const svg = /** @type {HTMLElement} */ (nodes[1]).querySelector('svg');
    expect(svg?.innerHTML).toBe(icon('chart-line').innerHTML);
  });
});

describe('the Activity grouping', () => {
  test('groups by the facility day: Today, Yesterday, Earlier', () => {
    const entries = [
      entry('today', 'markdown', '2026-10-30T01:00:00Z'),
      entry('yesterday', 'markdown', '2026-10-29T23:00:00Z'),
      entry('older', 'markdown', '2026-10-20T10:00:00Z'),
    ];
    const nodes = rows.artifactRows(entries, { ...READY, view: 'activity', filter: '' });
    expect(heads(nodes)).toEqual(['Today', 'Yesterday', 'Earlier']);
    expect(rowIds(nodes)).toEqual(['today', 'yesterday', 'older']);
  });

  test('the day is the facility day, not the viewer day', () => {
    document.documentElement.setAttribute('data-facility-timezone', 'Asia/Tokyo');
    // 16:00 UTC on the 29th is already the 30th in Tokyo.
    const nodes = rows.artifactRows([entry('a', 'markdown', '2026-10-29T16:00:00Z')], {
      ...READY,
      view: 'activity',
      filter: '',
    });
    expect(heads(nodes)).toEqual(['Today']);
  });
});

describe('a row', () => {
  test('is a listbox option carrying the id, the title and the time', () => {
    const a = entry('a1', 'markdown', '2026-10-30T02:48:00Z');
    const [, row] = rows.artifactRows([a], { ...READY, shownId: 'a1', view: 'types', filter: '' });
    const button = /** @type {HTMLButtonElement} */ (row);
    expect(button.tagName).toBe('BUTTON');
    expect(button.type).toBe('button');
    expect(button.className).toBe('osprey-list-row');
    expect(button.getAttribute('role')).toBe('option');
    expect(button.getAttribute('aria-selected')).toBe('true');
    expect(button.dataset.id).toBe('a1');
    expect(button.title).toBe('Title a1');
    expect(button.querySelector('.osprey-list-title')?.textContent).toBe('Title a1');
    const time = button.querySelector('time');
    expect(time?.getAttribute('datetime')).toBe('2026-10-30T02:48:00Z');
    expect(time?.textContent).toBe(
      new Intl.DateTimeFormat(undefined, {
        hour: '2-digit',
        minute: '2-digit',
        timeZone: 'UTC',
      }).format(Date.UTC(2026, 9, 30, 2, 48))
    );
  });

  test('an older artifact shows its date; one with no title shows its file name', () => {
    const a = entry('a2', 'markdown', '2026-10-27T10:00:00Z', { title: '', filename: 'notes.md' });
    const [, row] = rows.artifactRows([a], { ...READY, view: 'types', filter: '' });
    expect(/** @type {HTMLElement} */ (row).getAttribute('aria-selected')).toBe('false');
    expect(/** @type {HTMLElement} */ (row).querySelector('.osprey-list-title')?.textContent).toBe(
      'notes.md'
    );
    expect(/** @type {HTMLElement} */ (row).querySelector('time')?.textContent).toBe(
      new Intl.DateTimeFormat(undefined, {
        month: 'short',
        day: 'numeric',
        timeZone: 'UTC',
      }).format(Date.UTC(2026, 9, 27, 10))
    );
  });

  test('a hostile title stays text', () => {
    const a = entry('x', 'markdown', '2026-10-30T02:48:00Z', {
      title: '<img src=x onerror=alert(1)>',
    });
    const [, row] = rows.artifactRows([a], { ...READY, view: 'types', filter: '' });
    expect(/** @type {HTMLElement} */ (row).querySelector('img')).toBeNull();
    expect(/** @type {HTMLElement} */ (row).textContent).toContain('<img src=x');
  });
});

describe('the filter and the empty lines', () => {
  const entries = [
    entry('a', 'markdown', '2026-10-30T02:00:00Z', { title: 'Beam Tunes' }),
    entry('b', 'markdown', '2026-10-30T01:00:00Z', { title: 'Orbit' }),
  ];

  test('filters by title, case-insensitively', () => {
    const nodes = rows.artifactRows(entries, { ...READY, view: 'types', filter: '  TUNES ' });
    expect(rowIds(nodes)).toEqual(['a']);
  });

  test('a filter matching nothing says so', () => {
    const nodes = rows.artifactRows(entries, { ...READY, view: 'types', filter: 'zzz' });
    expect(texts(nodes)).toEqual(['No artifacts match']);
  });

  test('nothing before the first page; then empty, or unavailable after a failure', () => {
    expect(rows.artifactRows([], { ...READY, loaded: false, view: 'types', filter: '' })).toEqual(
      []
    );
    expect(texts(rows.artifactRows([], { ...READY, view: 'types', filter: '' }))).toEqual([
      'No artifacts yet',
    ]);
    expect(
      texts(rows.artifactRows([], { ...READY, failed: true, view: 'types', filter: '' }))
    ).toEqual(['Artifacts are unavailable']);
    expect(texts(rows.runRows([], READY))).toEqual(['No script runs yet']);
    expect(rows.runRows([], { ...READY, loaded: false })).toEqual([]);
  });
});

describe('the Script runs rows', () => {
  test('newest first, no heads, the bash icon, the shown one selected', () => {
    const nodes = rows.runRows(
      [
        entry('r2', 'notebook', '2026-10-30T02:00:00Z'),
        entry('r1', 'json', '2026-10-30T01:00:00Z'),
      ],
      { ...READY, shownId: 'r1' }
    );
    expect(heads(nodes)).toEqual([]);
    expect(rowIds(nodes)).toEqual(['r2', 'r1']);
    const selected = nodes.map((node) =>
      /** @type {HTMLElement} */ (node).getAttribute('aria-selected')
    );
    expect(selected).toEqual(['false', 'true']);
    expect(/** @type {HTMLElement} */ (nodes[0]).querySelector('svg')?.innerHTML).toBe(
      icon('bash').innerHTML
    );
  });
});

describe('rowIn', () => {
  test('finds a row by id without a selector, whatever the id holds', () => {
    const list = document.createElement('div');
    list.append(...rows.runRows([entry('a"]b', 'json', '2026-10-30T01:00:00Z')], READY));
    expect(rows.rowIn(list, 'a"]b')?.dataset.id).toBe('a"]b');
    expect(rows.rowIn(list, 'missing')).toBeNull();
  });
});

describe('the ⋯ menu', () => {
  test('gets Group by, All sessions and Refresh once, with an accessible name', () => {
    const menu = document.createElement('div');
    rows.fillArtifactMenu(menu);
    rows.fillArtifactMenu(menu);
    const items = [...menu.querySelectorAll('.osprey-popover-item')];
    expect(items.map((item) => item.textContent)).toEqual([
      'Types',
      'Activity',
      'All sessions',
      'Refresh',
    ]);
    expect(items.map((item) => item.getAttribute('role'))).toEqual([
      'menuitemradio',
      'menuitemradio',
      'menuitemcheckbox',
      'menuitem',
    ]);
    expect(items.map((item) => item.getAttribute('data-art-view'))).toEqual([
      'types',
      'activity',
      null,
      null,
    ]);
    for (const item of items.slice(0, 3)) {
      expect(item.getAttribute('aria-checked')).toBe('false');
      expect(item.querySelector('.osprey-popover-check')).not.toBeNull();
    }
    const title = menu.querySelector('.osprey-popover-title');
    expect(title?.textContent).toBe('Group by');
    expect(title?.id).toBeTruthy();
    const group = menu.querySelector('[role="group"]');
    expect(group?.getAttribute('aria-labelledby')).toBe(title?.id);
    expect([...(group?.querySelectorAll('[role="menuitemradio"]') ?? [])]).toEqual(
      items.slice(0, 2)
    );
    expect(menu.querySelectorAll('.osprey-popover-sep')).toHaveLength(1);
    expect(menu.querySelector('.osprey-popover-sep')?.getAttribute('role')).toBe('separator');
    expect(menu.getAttribute('aria-label')).toBe('Artifact options');
  });

  test('keeps a label the skeleton gave it', () => {
    const menu = document.createElement('div');
    menu.setAttribute('aria-label', 'Options');
    rows.fillArtifactMenu(menu);
    expect(menu.getAttribute('aria-label')).toBe('Options');
  });
});
