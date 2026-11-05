/**
 * The command palette's async providers: CANOPY first, then Sessions and
 * Artifacts (palette.js, palette-registry.js, palette-boot.js).
 *
 *   npx vitest run tests/interfaces/web_terminal/palette-sessions.test.mjs
 *
 * The providers are pure factories over injected readers and verbs, so most
 * blocks drive them directly; the last block opens the palette through
 * palette-boot.js with the session model, the sidebar and the panel manager
 * doubled, and the artifacts panel's two routes answered by a fetch stub.
 */

import { test, expect, describe, beforeAll, beforeEach, afterEach, vi } from 'vitest';

const JS = '../../../src/osprey/interfaces/web_terminal/static/js';

const model = vi.hoisted(() => ({
  rows: /** @type {any[]} */ ([]),
  switchTo: /** @type {any} */ (null),
  setPick: /** @type {any} */ (null),
}));
const shown = vi.hoisted(() => ({ recordArtifactOpened: /** @type {any} */ (null) }));
const panels = vi.hoisted(() => ({ navigate: /** @type {any} */ (null) }));

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/session-model.js', async (orig) => ({
  ...(await orig()),
  rows: () => model.rows,
  sessionOfArtifact: (/** @type {any} */ a) =>
    (a?.owner && model.rows.find((r) => r.transcript_id === a.owner)) || null,
  switchTo: (/** @type {any[]} */ ...args) => model.switchTo(...args),
  setPick: (/** @type {any} */ id) => model.setPick(id),
}));

vi.mock(
  '../../../src/osprey/interfaces/web_terminal/static/js/sidebar-artifacts.js',
  async (orig) => ({
    ...(await orig()),
    recordArtifactOpened: (/** @type {string} */ id) => shown.recordArtifactOpened(id),
  })
);

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/panel-manager.js', async (orig) => ({
  ...(await orig()),
  getHiddenPanels: () => [{ id: 'artifacts', label: 'Artifacts' }],
  getVisiblePanels: () => [],
  getPopoutPanels: () => [],
  getPresets: () => [],
  navigateAndActivatePanel: (/** @type {any[]} */ ...args) => panels.navigate(...args),
}));

/** @type {Array<{ url: string, init?: any }>} */
const requests = [];
/** @type {(url: string, init?: any) => any} */
let answer = () => ({});

vi.stubGlobal(
  'fetch',
  vi.fn(async (/** @type {string} */ url, /** @type {any} */ init) => {
    requests.push({ url, init });
    const body = answer(url, init);
    return { ok: true, status: 200, json: async () => body };
  })
);

const { openPalette, closePalette, isOpen } = await import(`${JS}/palette.js`);
const { createSessionsProvider, createArtifactsProvider, collapseRuns, SESSION_BATCH_CAP } =
  await import(`${JS}/palette-registry.js`);
const { initCommandPalette } = await import(`${JS}/palette-boot.js`);

beforeAll(() => {
  Element.prototype.scrollIntoView = () => {};
});

beforeEach(() => {
  model.rows = [];
  model.switchTo = vi.fn(async () => true);
  model.setPick = vi.fn();
  shown.recordArtifactOpened = vi.fn(async () => {});
  panels.navigate = vi.fn();
  requests.length = 0;
  answer = () => ({});
  localStorage.clear();
});

afterEach(() => {
  closePalette();
  document.body.textContent = '';
  vi.useRealTimers();
});

/** @param {number} ms */
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** @param {string} value */
function typeQuery(value) {
  const input = /** @type {HTMLInputElement} */ (document.querySelector('.command-palette-input'));
  input.value = value;
  input.dispatchEvent(new Event('input', { bubbles: true }));
}

/** @returns {string[]} */
function headings() {
  return [...document.querySelectorAll('.command-palette-group-heading')].map(
    (h) => h.textContent ?? ''
  );
}

/** @param {string} cls @returns {HTMLElement[]} */
function rowsOf(cls) {
  return /** @type {HTMLElement[]} */ ([
    ...document.querySelectorAll(`.command-palette-item--${cls}`),
  ]);
}

/** @param {string} cls */
const labelsOf = (cls) =>
  rowsOf(cls).map(
    (r) => r.querySelector('.command-palette-item-label')?.textContent ?? r.textContent
  );

/** A session row. @param {string} id @param {string} title */
const session = (id, title) => ({
  transcript_id: id,
  session_key: `key-${id}`,
  current_for_key: true,
  title,
});

/** A provider double answering `rows` synchronously. @param {string} group @param {any} rows */
const fixed = (group, rows) => ({ group, start: vi.fn(), search: vi.fn(() => rows) });

describe('the provider list', () => {
  test('CANOPY renders first, then the async providers in their order', () => {
    const canopyRun = vi.fn();
    openPalette({
      searchCanopy: () => [{ label: 'Open screen GTLView', detail: 'library/gtl', run: canopyRun }],
      startCanopy: vi.fn(),
      asyncProviders: [
        fixed('Sessions', [{ label: 'Beam tuning', run: vi.fn() }]),
        fixed('Artifacts', [{ label: 'orbit.png', run: vi.fn() }]),
      ],
    });
    typeQuery('t');
    expect(headings()).toEqual(['CANOPY', 'Sessions', 'Artifacts']);
    expect(labelsOf('canopy')).toEqual(['Open screen GTLView']);
    expect(labelsOf('sessions')).toEqual(['Beam tuning']);
    expect(labelsOf('artifacts')).toEqual(['orbit.png']);
  });

  test('every provider starts once per open and its settle repaints', () => {
    /** @type {any} */
    let rows = null;
    /** @type {Array<() => void>} */
    const starts = [];
    const provider = {
      group: 'Artifacts',
      start: (/** @type {() => void} */ fn) => starts.push(fn),
      search: () => rows,
    };
    const startCanopy = vi.fn();
    openPalette({ searchCanopy: () => [], startCanopy, asyncProviders: [provider] });
    expect(startCanopy).toHaveBeenCalledTimes(1);
    expect(starts).toHaveLength(1);
    typeQuery('orb');
    expect(document.querySelector('.command-palette-status')?.textContent).toBe('Loading…');
    rows = [{ label: 'orbit.png', run: vi.fn() }];
    starts[0]();
    expect(labelsOf('artifacts')).toEqual(['orbit.png']);
  });

  test('an empty query asks no provider', () => {
    const provider = fixed('Sessions', [{ label: 'x', run: vi.fn() }]);
    openPalette({ asyncProviders: [provider] });
    expect(provider.search).not.toHaveBeenCalled();
    expect(headings()).not.toContain('Sessions');
  });

  test('a provider row runs on Enter and stays out of Recent', () => {
    const run = vi.fn();
    openPalette({ asyncProviders: [fixed('Sessions', [{ label: 'Beam tuning', run }])] });
    typeQuery('beam');
    const input = /** @type {HTMLElement} */ (document.querySelector('.command-palette-input'));
    input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
    expect(run).toHaveBeenCalledTimes(1);
    expect(isOpen()).toBe(false);
    expect(JSON.parse(localStorage.getItem('osprey-palette-recent-v1') || '[]')).toEqual([]);
  });
});

describe('Sessions', () => {
  test('matches by session name and opening one calls switchTo', () => {
    const rows = [session('t1', 'Beam tuning'), session('t2', 'Orbit feedback')];
    const switchTo = vi.fn();
    const provider = createSessionsProvider({ rows: () => rows, switchTo });
    const hits = /** @type {any[]} */ (provider.search('ORBIT'));
    expect(hits.map((h) => h.label)).toEqual(['Orbit feedback']);
    hits[0].run();
    expect(switchTo).toHaveBeenCalledWith(rows[1], { reason: 'gesture' });
  });
});

describe('Artifacts', () => {
  /** @param {Record<string, any>} over */
  function provider(over = {}) {
    const deps = {
      rows: () => /** @type {any[]} */ ([]),
      searchArtifacts: vi.fn(async () => /** @type {any[]} */ ([])),
      listSessions: vi.fn(async () => /** @type {any[]} */ ([])),
      sessionOfArtifact: (/** @type {any} */ a) =>
        deps.rows().find((r) => r.transcript_id === a.owner) ?? null,
      switchTo: vi.fn(),
      showArtifact: vi.fn(),
      debounceMs: 20,
      ...over,
    };
    const p = createArtifactsProvider(deps);
    const onChange = vi.fn();
    p.start(onChange);
    return { p, deps, onChange };
  }

  /** Search and wait for the debounced read to land. @param {any} p @param {string} q */
  async function settle(p, q) {
    expect(p.search(q)).toBeNull();
    await sleep(40);
    return /** @type {any[]} */ (p.search(q));
  }

  /**
   * A fixture searched the way the panel's `search` matches: title, file
   * name, type and form label.
   */
  const STORE = [
    {
      id: 'a1',
      title: 'Orbit plot',
      filename: 'orbit.png',
      artifact_type: 'image',
      form: 'plots',
      form_label: 'Plots',
      owner: 't1',
    },
    {
      id: 'a2',
      title: 'Notes',
      filename: 'shift_notes.md',
      artifact_type: 'markdown',
      form: 'documents',
      form_label: 'Documents',
      owner: null,
    },
  ];
  /** @param {string} q */
  const serverSearch = async (q) =>
    STORE.filter((e) =>
      [e.title, e.filename, e.artifact_type, e.form_label].some((v) =>
        v.toLowerCase().includes(q.toLowerCase())
      )
    );

  test('a match by kind', async () => {
    const { p } = provider({ searchArtifacts: vi.fn(serverSearch) });
    const hits = await settle(p, 'plots');
    expect(hits.map((h) => h.label)).toEqual(['Orbit plot']);
    expect(hits[0].detail).toContain('Plots');
  });

  test('a match by filename', async () => {
    const { p } = provider({ searchArtifacts: vi.fn(serverSearch) });
    const hits = await settle(p, 'shift_notes');
    expect(hits.map((h) => h.label)).toEqual(['Notes']);
  });

  test('a match by session name lists that session’s artifacts', async () => {
    const rows = [session('t1', 'Beam tuning'), session('t2', 'Orbit feedback')];
    const listSessions = vi.fn(async (/** @type {string[]} */ ids) =>
      ids.includes('t1') ? [{ id: 'a9', title: 'Tune scan', owner: 't1', form: 'plots' }] : []
    );
    const { p } = provider({ rows: () => rows, listSessions });
    const hits = await settle(p, 'beam');
    expect(listSessions).toHaveBeenCalledWith(['t1']);
    expect(hits.map((h) => h.label)).toEqual(['Tune scan']);
    expect(hits[0].detail).toContain('Beam tuning');
  });

  test('at most five title-matched sessions, newest first, in one request', async () => {
    const rows = Array.from({ length: 7 }, (_, i) => session(`t${i}`, `Scan ${i}`));
    const { p, deps } = provider({ rows: () => rows });
    await settle(p, 'scan');
    expect(SESSION_BATCH_CAP).toBe(5);
    expect(deps.listSessions).toHaveBeenCalledTimes(1);
    expect(deps.listSessions).toHaveBeenCalledWith(['t0', 't1', 't2', 't3', 't4']);
  });

  test('a run’s part collapses into its head’s hit', async () => {
    const head = { id: 'h', title: 'Run: fit', form: 'runs', exec_id: 'x1', run_part: 'result' };
    const part = { id: 'p', title: 'Review: fit', form: null, exec_id: 'x1', run_part: 'preview' };
    const output = { id: 'o', title: 'fit.png', form: 'plots', exec_id: 'x1', run_part: 'output' };
    expect(collapseRuns([part, head, output]).map((/** @type {{id: string}} */ e) => e.id)).toEqual(
      ['h', 'o']
    );
    const { p } = provider({ searchArtifacts: vi.fn(async () => [part, head]) });
    const hits = await settle(p, 'fit');
    expect(hits.map((h) => h.label)).toEqual(['Run: fit']);
  });

  test('opening an artifact of a loaded session switches to it with the artifact', async () => {
    const rows = [session('t1', 'Beam tuning')];
    const { p, deps } = provider({ rows: () => rows, searchArtifacts: vi.fn(serverSearch) });
    const hits = await settle(p, 'orbit');
    hits[0].run();
    expect(deps.switchTo).toHaveBeenCalledWith(rows[0], { artifactId: 'a1', reason: 'gesture' });
    expect(deps.showArtifact).not.toHaveBeenCalled();
  });

  test('opening an artifact with no session shows it and leaves the session', async () => {
    const { p, deps } = provider({ searchArtifacts: vi.fn(serverSearch) });
    const hits = await settle(p, 'notes');
    hits[0].run();
    expect(deps.showArtifact).toHaveBeenCalledWith(STORE[1]);
    expect(deps.switchTo).not.toHaveBeenCalled();
  });

  test('the search is debounced: a burst of keystrokes reads once, for the last query', async () => {
    vi.useFakeTimers();
    const { p, deps, onChange } = provider();
    expect(p.search('o')).toBeNull();
    vi.advanceTimersByTime(10);
    expect(p.search('or')).toBeNull();
    expect(p.search('or')).toBeNull();
    vi.advanceTimersByTime(10);
    expect(p.search('orb')).toBeNull();
    expect(deps.searchArtifacts).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(25);
    expect(deps.searchArtifacts).toHaveBeenCalledTimes(1);
    expect(deps.searchArtifacts).toHaveBeenCalledWith('orb');
    expect(onChange).toHaveBeenCalledTimes(1);
    expect(p.search('orb')).toEqual([]);
  });

  test('a failed read contributes no rows', async () => {
    const { p } = provider({
      searchArtifacts: vi.fn(async () => Promise.reject(new Error('down'))),
    });
    expect(await settle(p, 'x')).toEqual([]);
  });
});

describe('palette-boot wiring', () => {
  beforeAll(() => {
    const btn = document.createElement('button');
    btn.id = 'command-palette-btn';
    document.body.append(btn);
    initCommandPalette();
  });

  function open() {
    if (!document.getElementById('command-palette-btn')) {
      const btn = document.createElement('button');
      btn.id = 'command-palette-btn';
      document.body.append(btn);
      initCommandPalette();
    }
    /** @type {HTMLElement} */ (document.getElementById('command-palette-btn')).click();
  }

  test('Sessions and Artifacts read the session model and the panel’s two routes', async () => {
    model.rows = Array.from({ length: 6 }, (_, i) => session(`t${i}`, `Scan ${i}`));
    answer = (url) =>
      url.includes('session-list')
        ? {
            pages: {
              t0: { artifacts: [{ id: 'a0', title: 'Scan result', owner: 't0', form: 'plots' }] },
            },
          }
        : { artifacts: [{ id: 'a1', title: 'Scan notes', owner: null, form: 'documents' }] };
    open();
    typeQuery('scan');
    expect(headings()).toEqual(['Sessions', 'Artifacts']);
    expect(labelsOf('sessions')).toHaveLength(6);
    await sleep(260);
    const search = requests.find((r) => r.url.includes('/api/artifacts?'));
    const params = new URL(/** @type {any} */ (search).url, 'http://x').searchParams;
    expect(params.get('search')).toBe('scan');
    expect(params.get('runs')).toBe('grouped');
    const batches = requests.filter((r) => r.url.endsWith('/api/artifacts/session-list'));
    expect(batches).toHaveLength(1);
    expect(JSON.parse(batches[0].init.body)).toMatchObject({
      sessions: ['t0', 't1', 't2', 't3', 't4'],
      runs: 'grouped',
    });
    expect(labelsOf('artifacts')).toEqual(['Scan notes', 'Scan result']);
  });

  test('a session row enters the session; an owned artifact enters its session', async () => {
    model.rows = [session('t1', 'Beam tuning')];
    answer = () => ({ artifacts: [{ id: 'a1', title: 'Beam plot', owner: 't1', form: 'plots' }] });
    open();
    typeQuery('beam');
    rowsOf('sessions')[0].click();
    expect(model.switchTo).toHaveBeenCalledWith(model.rows[0], { reason: 'gesture' });

    open();
    typeQuery('beam');
    await sleep(260);
    rowsOf('artifacts')[0].click();
    expect(model.switchTo).toHaveBeenLastCalledWith(model.rows[0], {
      artifactId: 'a1',
      reason: 'gesture',
    });
    expect(model.setPick).not.toHaveBeenCalled();
  });

  test('an artifact with no session is shown and the session stays', async () => {
    answer = () => ({
      artifacts: [{ id: 'a2', title: 'Loose notes', owner: null, form: 'documents' }],
    });
    open();
    typeQuery('loose');
    await sleep(260);
    rowsOf('artifacts')[0].click();
    await sleep(0);
    expect(model.switchTo).not.toHaveBeenCalled();
    expect(shown.recordArtifactOpened).toHaveBeenCalledWith('a2');
    expect(model.setPick).toHaveBeenCalledWith('a2');
    expect(panels.navigate).toHaveBeenCalledWith('artifacts', '');
  });
});
