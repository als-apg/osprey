// @ts-check
/**
 * The rows of the sidebar's artifact tree (sidebar-artifact-rows.js):
 *   npx vitest run tests/interfaces/web_terminal/sidebar-artifact-rows.test.mjs
 *
 * Pins the row markup the list primitives style, the time on the facility
 * clock, and the session tree's rows. Expected times are computed with Intl
 * directly, in the stamped facility zone.
 */

import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

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

/** @param {Node[]} nodes */
function texts(nodes) {
  return nodes.map((node) => /** @type {HTMLElement} */ (node).textContent);
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

describe('a row', () => {
  test('is a listbox option carrying the id, the title and the time', () => {
    const a = entry('a1', 'markdown', '2026-10-30T02:48:00Z');
    const button = rows.artifactRow(a, { shownId: 'a1' });
    expect(button.tagName).toBe('BUTTON');
    expect(button.type).toBe('button');
    expect(button.classList.contains('osprey-list-row')).toBe(true);
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
    const row = rows.artifactRow(a, { shownId: null });
    expect(row.getAttribute('aria-selected')).toBe('false');
    expect(row.querySelector('.osprey-list-title')?.textContent).toBe('notes.md');
    expect(row.querySelector('time')?.textContent).toBe(
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
    const row = rows.artifactRow(a, { shownId: null });
    expect(row.querySelector('img')).toBeNull();
    expect(row.textContent).toContain('<img src=x');
  });
});

describe('rowIn', () => {
  test('finds a row by id without a selector, whatever the id holds', () => {
    const list = document.createElement('div');
    list.append(rows.artifactRow(entry('a"]b', 'json', '2026-10-30T01:00:00Z'), { shownId: null }));
    expect(rows.rowIn(list, 'a"]b')?.dataset.id).toBe('a"]b');
    expect(rows.rowIn(list, 'missing')).toBeNull();
  });
});

/* ------------------------------------------------------------------------ */
/* The session tree: Agent row → sessions → form groups → runs and artifacts */
/* ------------------------------------------------------------------------ */

/** The shared /api/type-registry fixture the gallery suite reads too. */
const REGISTRY = JSON.parse(
  readFileSync(
    resolve(dirname(fileURLToPath(import.meta.url)), '../fixtures/type-registry.json'),
    'utf8'
  )
);

const TID = '11111111-1111-4111-8111-111111111111';

/**
 * A session row as GET /api/sessions lists it.
 * @param {object} [extra]
 */
/** @typedef {import('../../../src/osprey/interfaces/web_terminal/static/js/sidebar-artifact-rows.js').SessionRow} SessionRow */

/** @param {Partial<SessionRow>} [extra] @returns {SessionRow} */
const session = (extra = {}) => ({
  transcript_id: TID,
  title: 'Tune the orbit',
  naming: 'done',
  state: 'idle',
  truncated: false,
  ...extra,
});

/**
 * A run head as the listing returns it under runs=grouped.
 * @param {string} id
 * @param {string | null} status
 * @param {object[]} parts
 */
const head = (id, status, parts = [], extra = {}) =>
  entry(id, 'json', '2026-10-30T10:00:00Z', {
    form: 'runs',
    form_label: 'Runs',
    exec_id: `exec-${id}`,
    run_part: 'result',
    run_status: status,
    run_parts: parts,
    ...extra,
  });

/** @param {Element} el */
const cls = (el) => [...el.classList];

/** @param {Node[]} nodes */
function treeRowIds(nodes) {
  return nodes
    .filter((n) => n instanceof HTMLElement && n.classList.contains('osprey-list-row'))
    .map((n) => /** @type {HTMLElement} */ (n).dataset.id);
}

describe('the form groups', () => {
  test('come from the shared registry: base forms in order, then extension types', () => {
    const forms = rows.treeForms(REGISTRY);
    expect(forms.map((f) => f.label)).toEqual([
      'Runs',
      'Plots',
      'Trends',
      'Tables',
      'Documents',
      'Dashboards',
      'Images',
      'Data',
      'Beam Log',
      'Orbit Fit',
    ]);
    // No format names (Plotly, Matplotlib, Markdown, …) are group labels.
    const formats = Object.values(REGISTRY.artifact_types).map((t) => t.label);
    for (const f of forms) expect(formats).not.toContain(f.label);
    // An extension type draws its base form's icon.
    expect(forms.find((f) => f.key === 'beam_log')?.icon).toBe(
      forms.find((f) => f.key === 'tables')?.icon
    );
  });

  test('without a registry the base forms still draw', () => {
    expect(rows.treeForms(null).map((f) => f.key)).toEqual([
      'runs',
      'plots',
      'trends',
      'tables',
      'documents',
      'dashboards',
      'images',
      'data',
    ]);
  });
});

describe('the Agent row disclosure', () => {
  test('a chevron that says whether the sessions show', () => {
    const open = rows.agentDisclosure(true);
    expect(open.getAttribute('data-tree-agent-disc')).toBe('');
    expect(open.getAttribute('aria-expanded')).toBe('true');
    expect(open.querySelector('svg')).not.toBeNull();
    expect(rows.agentDisclosure(false).getAttribute('aria-expanded')).toBe('false');
  });
});

describe('a session row', () => {
  test('carries the title, the server count and the session id at the first indent', () => {
    const row = rows.sessionRow(session(), { active: false, open: false, count: 7 });
    expect(row.dataset.sessionId).toBe(TID);
    expect(row.querySelector('.osprey-tree-title')?.textContent).toBe('Tune the orbit');
    expect(row.querySelector('.osprey-tree-n')?.textContent).toBe('7');
    expect(cls(row)).toContain('osprey-tree-session');
    expect(cls(row)).toContain('osprey-tree-d0');
    expect(row.getAttribute('aria-expanded')).toBe('false');
    expect(row.getAttribute('role')).toBe('presentation');
    expect(row.textContent).not.toContain('naming…');
  });

  test('the active session is tinted; a live one has a filled dot', () => {
    const active = rows.sessionRow(session({ state: 'live' }), {
      active: true,
      open: true,
      count: 0,
    });
    expect(cls(active)).toContain('active');
    expect(active.getAttribute('aria-expanded')).toBe('true');
    expect(cls(/** @type {Element} */ (active.querySelector('.osprey-dot')))).toContain('ok');
    const idle = rows.sessionRow(session(), { active: false, open: false, count: 0 });
    expect(cls(idle)).not.toContain('active');
    expect(cls(/** @type {Element} */ (idle.querySelector('.osprey-dot')))).toContain('hollow');
  });

  test('shows "naming…" while the name is pending', () => {
    const row = rows.sessionRow(session({ naming: 'pending', title: 'tune the orbit please' }), {
      active: false,
      open: false,
      count: 1,
    });
    expect(row.querySelector('.osprey-tree-naming')?.textContent).toBe('naming…');
    expect(cls(/** @type {Element} */ (row.querySelector('.osprey-tree-title')))).toContain(
      'pending'
    );
  });

  test('a pending session with no title yet reads only "naming…"', () => {
    const row = rows.sessionRow(session({ naming: 'pending', title: '' }), {
      active: false,
      open: false,
      count: 0,
    });
    expect(row.querySelector('.osprey-tree-title')?.textContent).toBe('naming…');
    expect(row.querySelector('.osprey-tree-naming')).toBeNull();
  });

  test('a session without a title (failed naming, truncated scan) is "Untitled session"', () => {
    /** @type {Partial<SessionRow>[]} */
    const extras = [
      { naming: 'failed', title: '' },
      { truncated: true, title: null },
    ];
    for (const extra of extras) {
      const row = rows.sessionRow(session(extra), { active: false, open: false, count: 0 });
      expect(row.querySelector('.osprey-tree-title')?.textContent).toBe('Untitled session');
    }
  });

  test('a hostile title stays text', () => {
    const row = rows.sessionRow(session({ title: '<img src=x onerror=alert(1)>' }), {
      active: false,
      open: false,
      count: 0,
    });
    expect(row.querySelector('img')).toBeNull();
    expect(row.querySelector('.osprey-tree-title')?.textContent).toBe(
      '<img src=x onerror=alert(1)>'
    );
  });
});

describe("a session's scroll box", () => {
  test('wraps the rows in a group; an empty session says so', () => {
    const row = rows.artifactRow(entry('a', 'markdown', '2026-10-30T10:00:00Z'), {
      shownId: null,
    });
    const box = rows.sessionBox(TID, [row]);
    expect(cls(box)).toContain('osprey-tree-box');
    expect(box.getAttribute('role')).toBe('group');
    expect(box.dataset.sessionBox).toBe(TID);
    expect(box.firstChild).toBe(row);
    const empty = rows.sessionBox(TID, []);
    expect(empty.textContent).toBe('No artifacts yet');
    expect(empty.querySelector('.osprey-tree-empty')?.getAttribute('role')).toBe('presentation');
  });
});

describe('a group head', () => {
  test('the label, the server count and the open state at the second indent', () => {
    const forms = rows.treeForms(REGISTRY);
    const plots = /** @type {any} */ (forms.find((f) => f.key === 'plots'));
    const node = rows.groupHead(plots, { sessionId: TID, count: 12, open: true });
    expect(node.querySelector('.osprey-tree-label')?.textContent).toBe('Plots');
    expect(node.querySelector('.osprey-tree-n')?.textContent).toBe('12');
    expect(node.dataset.groupHead).toBe(`${TID}:plots`);
    expect(node.getAttribute('aria-expanded')).toBe('true');
    expect(node.getAttribute('role')).toBe('presentation');
    expect(cls(node)).toContain('osprey-tree-d1');
    const shut = rows.groupHead(plots, { sessionId: TID, count: 1, open: false });
    expect(shut.getAttribute('aria-expanded')).toBe('false');
  });
});

describe('a run row', () => {
  /** @param {HTMLElement} row */
  const dotOf = (row) => row.querySelector('.osprey-dot');

  test.each([
    ['succeeded', 'ok'],
    ['failed', 'err'],
    ['not_run', 'hollow'],
  ])('a %s run shows a %s dot', (status, dot) => {
    const [row] = rows.runTreeRows(head('h', status), { shownId: null, open: false });
    const el = /** @type {HTMLElement} */ (row);
    expect(cls(/** @type {Element} */ (dotOf(el)))).toContain(dot);
    expect(el.dataset.runStatus).toBe(status);
    expect(cls(el)).toContain('osprey-tree-run');
    expect(cls(el)).toContain('osprey-tree-d2');
  });

  test('a run still in flight shows no dot', () => {
    const [row] = rows.runTreeRows(head('h', null), { shownId: null, open: false });
    expect(dotOf(/** @type {HTMLElement} */ (row))).toBeNull();
    expect(/** @type {HTMLElement} */ (row).dataset.runStatus).toBe('running');
  });

  test('a run that never ran is marked "not run"', () => {
    const [row] = rows.runTreeRows(head('h', 'not_run'), { shownId: null, open: false });
    expect(row.querySelector('.osprey-tree-meta')?.textContent).toBe('not run');
  });

  test('the output count, and closed it draws only the head', () => {
    const parts = [
      { id: 'o1', role: 'output', title: 'Orbit', form: 'plots' },
      { id: 'o2', role: 'output', title: 'Tune table', form: 'tables' },
      { id: 'p', role: 'preview', title: 'preview', form: 'documents' },
      { id: 'c', role: 'code', title: 'nb', form: 'documents' },
    ];
    const nodes = rows.runTreeRows(head('h', 'succeeded', parts), {
      shownId: null,
      open: false,
    });
    expect(nodes).toHaveLength(1);
    const row = /** @type {HTMLElement} */ (nodes[0]);
    expect(row.querySelector('.osprey-tree-meta')?.textContent).toBe('2');
    expect(row.dataset.run).toBe('exec-h');
    expect(row.getAttribute('aria-expanded')).toBe('false');
  });

  test('open, the outputs then Approval preview and Code, at the third indent', () => {
    const parts = [
      { id: 'c', role: 'code', title: 'nb', form: 'documents' },
      { id: 'o1', role: 'output', title: 'Orbit', form: 'plots' },
      { id: 'p', role: 'preview', title: 'preview', form: 'documents' },
      { id: 'h', role: 'result', title: 'self', form: 'runs' },
    ];
    const nodes = rows.runTreeRows(head('h', 'failed', parts), { shownId: 'p', open: true });
    expect(treeRowIds(nodes)).toEqual(['h', 'o1', 'p', 'c']);
    const kids = /** @type {HTMLElement[]} */ (nodes.slice(1));
    expect(kids.map((k) => k.querySelector('.osprey-list-title')?.textContent)).toEqual([
      'Orbit',
      'Approval preview',
      'Code',
    ]);
    for (const kid of kids) expect(cls(kid)).toContain('osprey-tree-d3');
    expect(cls(kids[1])).toContain('osprey-tree-part');
    expect(kids[1].getAttribute('aria-selected')).toBe('true');
    expect(nodes[0].getAttribute('aria-expanded')).toBe('true');
  });

  test('a run with no parts has no disclosure', () => {
    const [row] = rows.runTreeRows(head('h', 'succeeded'), { shownId: null, open: false });
    expect(row.querySelector('[data-tree-disc]')).toBeNull();
    expect(/** @type {HTMLElement} */ (row).hasAttribute('aria-expanded')).toBe(false);
  });
});

describe('an artifact row', () => {
  test('an interactive plot is marked "interactive"', () => {
    const row = rows.artifactRow(
      entry('a', 'plot_html', '2026-10-30T10:00:00Z', { form: 'plots', interactive: true }),
      { shownId: null }
    );
    expect([...row.querySelectorAll('.osprey-tree-meta')].map((m) => m.textContent)).toEqual([
      'interactive',
    ]);
    expect(cls(row)).toContain('osprey-tree-d2');
  });

  test('a run output in its form group is marked "run"', () => {
    const row = rows.artifactRow(
      entry('a', 'plot_png', '2026-10-30T10:00:00Z', {
        form: 'plots',
        exec_id: 'e1',
        run_part: 'output',
      }),
      { shownId: null }
    );
    expect(row.querySelector('.osprey-tree-meta')?.textContent).toBe('run');
  });

  test('a plain artifact has no marker; the shown one is selected', () => {
    const row = rows.artifactRow(entry('a', 'markdown', '2026-10-30T10:00:00Z'), {
      shownId: 'a',
    });
    expect(row.querySelector('.osprey-tree-meta')).toBeNull();
    expect(row.getAttribute('aria-selected')).toBe('true');
    expect(row.getAttribute('role')).toBe('option');
  });
});

describe('the highlight', () => {
  test('marks the row page load points at, without selecting it', () => {
    const plain = rows.artifactRow(entry('a', 'markdown', '2026-10-30T10:00:00Z'), {
      shownId: null,
      highlightId: 'a',
    });
    expect(plain.dataset.highlight).toBe('');
    expect(cls(plain)).toContain('osprey-tree-highlight');
    expect(plain.getAttribute('aria-selected')).toBe('false');
    const other = rows.artifactRow(entry('b', 'markdown', '2026-10-30T10:00:00Z'), {
      shownId: null,
      highlightId: 'a',
    });
    expect(other.hasAttribute('data-highlight')).toBe(false);
  });

  test('reaches a run head and a run part', () => {
    const parts = [{ id: 'p', role: 'preview', title: 'x', form: 'documents' }];
    const headHit = rows.runTreeRows(head('h', 'succeeded', parts), {
      shownId: null,
      highlightId: 'h',
      open: false,
    });
    expect(/** @type {HTMLElement} */ (headHit[0]).dataset.highlight).toBe('');
    const partHit = rows.runTreeRows(head('h', 'succeeded', parts), {
      shownId: null,
      highlightId: 'p',
      open: true,
    });
    expect(/** @type {HTMLElement} */ (partHit[1]).dataset.highlight).toBe('');
  });
});

describe('Kind mode', () => {
  const forms = rows.treeForms(REGISTRY);
  const entries = [
    head('h1', 'succeeded', [{ id: 'o1', role: 'output', title: 'Orbit', form: 'plots' }]),
    entry('o1', 'plot_png', '2026-10-30T10:00:00Z', {
      form: 'plots',
      exec_id: 'exec-h1',
      run_part: 'output',
    }),
    entry('p1', 'plot_html', '2026-10-30T09:00:00Z', { form: 'plots', interactive: true }),
    entry('d1', 'markdown', '2026-10-30T08:00:00Z', { form: 'documents' }),
    entry('b1', 'json', '2026-10-30T07:00:00Z', { form: 'beam_log' }),
    entry('part', 'notebook', '2026-10-30T07:00:00Z', { form: null }),
  ];

  test('heads in form order with the server counts; parts are never top-level', () => {
    const nodes = rows.kindRows(entries, {
      forms,
      sessionId: TID,
      counts: { runs: 4, plots: 9, documents: 1, beam_log: 1 },
      openGroups: new Set(['runs', 'plots', 'documents', 'beam_log']),
      openRuns: new Set(),
      shownId: null,
    });
    const headTexts = nodes
      .filter((n) => n instanceof HTMLElement && n.classList.contains('osprey-tree-head'))
      .map((n) => n.textContent);
    expect(headTexts).toEqual(['Runs4', 'Plots9', 'Documents1', 'Beam Log1']);
    expect(treeRowIds(nodes)).toEqual(['h1', 'o1', 'p1', 'd1', 'b1']);
  });

  test('a closed group draws its head only; a form with no count draws nothing', () => {
    const nodes = rows.kindRows(entries, {
      forms,
      sessionId: TID,
      counts: { runs: 1, plots: 2, documents: 0, beam_log: 1 },
      openGroups: new Set(['beam_log']),
      openRuns: new Set(),
      shownId: null,
    });
    expect(texts(nodes).filter((t) => t?.startsWith('Documents'))).toEqual([]);
    expect(treeRowIds(nodes)).toEqual(['b1']);
  });

  test('without server counts a group counts the rows it holds', () => {
    const nodes = rows.kindRows(entries, {
      forms,
      sessionId: TID,
      counts: null,
      openGroups: new Set(),
      openRuns: new Set(),
      shownId: null,
    });
    expect(
      nodes
        .filter((n) => n instanceof HTMLElement && n.classList.contains('osprey-tree-head'))
        .map((n) => n.textContent)
    ).toEqual(['Runs1', 'Plots2', 'Documents1', 'Beam Log1']);
  });
});

describe('Activity mode', () => {
  const entries = [
    head('h1', 'succeeded', [{ id: 'o1', role: 'output', title: 'Orbit', form: 'plots' }], {
      timestamp: '2026-10-30T10:00:00Z',
    }),
    entry('o1', 'plot_png', '2026-10-30T10:00:00Z', {
      form: 'plots',
      exec_id: 'exec-h1',
      run_part: 'output',
    }),
    entry('y', 'markdown', '2026-10-29T10:00:00Z', { form: 'documents' }),
    entry('old', 'markdown', '2026-10-01T10:00:00Z', { form: 'documents' }),
    entry('part', 'notebook', '2026-10-01T09:00:00Z', { form: null }),
  ];

  test('Today, Yesterday and Older bands; never an "Earlier" band', () => {
    const nodes = rows.activityRows(entries, { shownId: null, openRuns: new Set() });
    const bands = nodes
      .filter((n) => n instanceof HTMLElement && n.classList.contains('osprey-tree-day'))
      .map((n) => n.textContent);
    expect(bands).toEqual(['Today', 'Yesterday', 'Older']);
    expect(bands).not.toContain('Earlier');
  });

  const bandsOf = (list) =>
    rows
      .activityRows(list, { shownId: null, openRuns: new Set() })
      .filter((n) => n instanceof HTMLElement && n.classList.contains('osprey-tree-day'))
      .map((n) => n.textContent);

  test('a session whose rows all fall on one day shows no day heading', () => {
    const today = [
      entry('a', 'markdown', '2026-10-30T10:00:00Z', { form: 'documents' }),
      entry('b', 'markdown', '2026-10-30T09:00:00Z', { form: 'documents' }),
    ];
    expect(bandsOf(today)).toEqual([]);
    expect(
      bandsOf([entry('o', 'markdown', '2026-10-01T10:00:00Z', { form: 'documents' })])
    ).toEqual([]);
  });

  test('two days show headings for both, and a live row from a new day adds them', () => {
    const one = [entry('a', 'markdown', '2026-10-30T10:00:00Z', { form: 'documents' })];
    expect(bandsOf(one)).toEqual([]);
    const two = [...one, entry('y', 'markdown', '2026-10-29T10:00:00Z', { form: 'documents' })];
    expect(bandsOf(two)).toEqual(['Today', 'Yesterday']);
  });

  test('a run is one row: its outputs and parts are not listed beside it', () => {
    const nodes = rows.activityRows(entries, { shownId: null, openRuns: new Set() });
    expect(treeRowIds(nodes)).toEqual(['h1', 'y', 'old']);
  });

  test('an open run lists its parts under it', () => {
    const nodes = rows.activityRows(entries, { shownId: null, openRuns: new Set(['exec-h1']) });
    expect(treeRowIds(nodes)).toEqual(['h1', 'o1', 'y', 'old']);
  });
});

describe('the rows after the sessions', () => {
  test('"More sessions" carries how many are not loaded yet', () => {
    const more = rows.moreSessionsRow(42);
    expect(more.dataset.treeMore).toBe('');
    expect(more.querySelector('.osprey-tree-label')?.textContent).toBe('More sessions');
    expect(more.querySelector('.osprey-tree-n')?.textContent).toBe('42');
  });

  test('the Earlier row is a session-level row with its count', () => {
    const earlier = rows.earlierRow({ count: 3, open: false });
    expect(earlier.dataset.treeEarlier).toBe('');
    expect(earlier.querySelector('.osprey-tree-title')?.textContent).toBe('Earlier');
    expect(earlier.querySelector('.osprey-tree-n')?.textContent).toBe('3');
    expect(cls(earlier)).toContain('osprey-tree-d0');
    expect(earlier.getAttribute('aria-expanded')).toBe('false');
  });

  test.each([
    ['incomplete', 'Session history may be incomplete'],
    ['unavailable', 'Session history unavailable'],
  ])('the %s notice', (kind, text) => {
    const notice = rows.noticeRow(/** @type {any} */ (kind));
    expect(notice.textContent).toBe(text);
    expect(notice.dataset.treeNotice).toBe(kind);
    expect(notice.getAttribute('role')).toBe('presentation');
  });
});
