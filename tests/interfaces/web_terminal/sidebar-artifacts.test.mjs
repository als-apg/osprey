// @ts-check
/**
 * The sidebar's artifact lists (sidebar-artifacts.js):
 *   npx vitest run tests/interfaces/web_terminal/sidebar-artifacts.test.mjs
 *
 * The hub owns the artifact list in layout F: the Artifacts row lists the
 * session's artifacts, the Script runs row its run records, both fetched from
 * the artifacts panel through the hub's proxy with the server's `run` filter,
 * and the artifacts iframe shows the one artifact this module picks. These
 * tests drive the module through the sidebar skeleton's ids, the panel's event
 * stream (api.js's real shared stream over a stubbed EventSource) and stubbed
 * fetches, and observe what it draws, what it fetches, what it tells the
 * viewer (`osprey-artifact-select`, never the agent's `/api/focus`) and what
 * it contributes to the pane header.
 *
 * Seams mocked: the session source (terminal.js), the header region
 * (tile-header-contrib.js), the iframe sends (panel-iframe-sync.js), the
 * panel activation (panel-manager.js, panel-placement.js) and the tile-menu
 * registry (panel-menu-policy.js). Module state is isolated by
 * vi.resetModules() + dynamic import per test and the module's own teardown.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import { stubEventSource } from '../_support/event-source.mjs';

const JS = '../../../src/osprey/interfaces/web_terminal/static/js';
const MODULE = `${JS}/sidebar-artifacts.js`;

const mocks = vi.hoisted(() => ({
  sessionId: /** @type {string | null} */ ('sess-1'),
  /** @type {((id: string) => void)[]} */
  sessionListeners: [],
  contributeLocal: vi.fn(),
  sendSelect: vi.fn(),
  sendAction: vi.fn(),
  activateTab: vi.fn(),
  hasDockedTile: vi.fn(() => true),
  setServiceMenuExtras: vi.fn(),
}));

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/terminal.js', () => ({
  getCurrentSessionId: () => mocks.sessionId,
  onSessionChange: (/** @type {(id: string) => void} */ fn) => mocks.sessionListeners.push(fn),
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/tile-header-contrib.js', () => ({
  contributeLocal: mocks.contributeLocal,
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/panel-iframe-sync.js', () => ({
  sendArtifactSelectToIframe: mocks.sendSelect,
  sendArtifactActionToIframe: mocks.sendAction,
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/panel-manager.js', () => ({
  activateTab: mocks.activateTab,
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/panel-placement.js', () => ({
  hasDockedTile: mocks.hasDockedTile,
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/panel-menu-policy.js', () => ({
  setServiceMenuExtras: mocks.setServiceMenuExtras,
}));
vi.mock('/design-system/js/theme-manager.js', () => ({ getTheme: () => 'dark' }));

/** @type {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/sidebar-artifacts.js')} */
let mod;
/** @type {ReturnType<typeof stubEventSource>} */
let events;
/** @type {ReturnType<typeof vi.fn>} */
let fetchMock;

/* ---- fixtures ---------------------------------------------------------- */

/**
 * @param {string} id
 * @param {string} type
 * @param {string} timestamp
 * @param {string} title
 * @param {object} [extra]
 */
const art = (id, type, timestamp, title, extra = {}) => ({
  id,
  artifact_type: type,
  timestamp,
  title,
  filename: `${id}.bin`,
  session_id: 'sess-1',
  run_record: false,
  ...extra,
});

const A3 = art('a3', 'markdown', '2026-10-30T11:00:00Z', 'Tunes report');
const A2 = art('a2', 'plot_png', '2026-10-30T10:00:00Z', 'Orbit plot');
const A1 = art('a1', 'markdown', '2026-10-30T09:00:00Z', 'Chroma report');
const R1 = art('r1', 'notebook', '2026-10-30T10:30:00Z', 'Notebook: compute', { run_record: true });

/**
 * The pages each listing serves, by cursor (`first` for the first page), and
 * the single artifacts `GET /api/artifacts/{id}` answers.
 * @type {{artifacts: Record<string, any>, runs: Record<string, any>, byId: Record<string, any>}}
 */
let server;

function defaultServer() {
  return {
    artifacts: { first: { artifacts: [A3, A2, A1], total: 3, next_cursor: null } },
    runs: { first: { artifacts: [R1], total: 1, next_cursor: null } },
    byId: {},
  };
}

/** @param {unknown} body @param {number} [status] */
const reply = (body, status = 200) => ({ ok: status < 400, status, json: async () => body });

/** @param {string} url */
function route(url) {
  const u = new URL(url, 'http://hub');
  const one = u.pathname.match(/\/panel\/artifacts\/api\/artifacts\/(.+)$/);
  if (one) {
    const entry = server.byId[decodeURIComponent(one[1])];
    return entry ? reply(entry) : reply({ detail: 'not found' }, 404);
  }
  if (u.pathname.endsWith('/panel/artifacts/api/artifacts')) {
    const pages = u.searchParams.get('run') === 'true' ? server.runs : server.artifacts;
    return reply(pages[u.searchParams.get('cursor') ?? 'first'] ?? { artifacts: [], total: 0 });
  }
  return reply({}, 404);
}

/** The sidebar skeleton slots this module draws into (index.html, sidebar.js). */
function skeleton() {
  document.body.innerHTML = `
    <nav id="sidebar">
      <div class="sidebar-row" id="sidebar-artifacts-row">
        <button class="osprey-nav-row parent panel-rail-button" data-panel-id="artifacts">
          <span class="osprey-nav-badge" data-art-count></span>
        </button>
        <div class="sidebar-row-actions">
          <button class="osprey-mini" data-art-filter-toggle></button>
          <button class="osprey-mini" data-art-menu aria-controls="sidebar-artifacts-menu"></button>
        </div>
        <div id="sidebar-artifacts-kids" class="sidebar-kids">
          <label class="sidebar-kids-filter" data-art-filter hidden><input type="search"></label>
          <div id="sidebar-artifacts-list" class="sidebar-kids-list" role="listbox"></div>
        </div>
      </div>
      <div class="sidebar-row" id="sidebar-runs-row">
        <button class="osprey-nav-row parent" id="sidebar-runs-toggle">
          <span class="osprey-nav-badge" data-runs-count></span>
        </button>
        <div id="sidebar-runs-kids" class="sidebar-kids">
          <div id="sidebar-runs-list" class="sidebar-kids-list" role="listbox"></div>
        </div>
      </div>
    </nav>
    <div id="sidebar-artifacts-flyout" class="osprey-popover" hidden>
      <div class="sidebar-flyout-head"><b>Artifacts</b><span class="osprey-count" data-art-count></span></div>
      <div class="sidebar-kids">
        <label class="sidebar-kids-filter" data-art-filter hidden><input type="search"></label>
        <div id="sidebar-artifacts-flyout-list" class="sidebar-kids-list" role="listbox"></div>
      </div>
    </div>
    <div id="sidebar-artifacts-menu" class="osprey-popover" role="menu" hidden></div>
    <iframe class="panel-iframe" data-panel-id="artifacts"></iframe>
  `;
}

/** @returns {HTMLIFrameElement} */
function viewer() {
  return /** @type {HTMLIFrameElement} */ (document.querySelector('iframe.panel-iframe'));
}

/** Let every pending fetch and render settle. */
async function settle() {
  for (let i = 0; i < 4; i += 1) await new Promise((resolve) => setTimeout(resolve, 0));
}

/** @param {string} listId */
function ids(listId) {
  return [...document.querySelectorAll(`#${listId} .osprey-list-row`)].map(
    (row) => /** @type {HTMLElement} */ (row).dataset.id
  );
}

/** @param {string} listId */
function heads(listId) {
  return [...document.querySelectorAll(`#${listId} .osprey-list-head`)].map((h) => h.textContent);
}

/** @param {string} listId @param {string} id */
function row(listId, id) {
  const el = [...document.querySelectorAll(`#${listId} .osprey-list-row`)].find(
    (r) => /** @type {HTMLElement} */ (r).dataset.id === id
  );
  if (!el) throw new Error(`no row ${id} in #${listId}`);
  return /** @type {HTMLElement} */ (el);
}

/** The artifact id the viewer was last told to show. */
function shown() {
  return mocks.sendSelect.mock.lastCall?.[1];
}

/** The last header contribution: its items and its dispatch. */
function headerNow() {
  const call = mocks.contributeLocal.mock.lastCall;
  if (!call) throw new Error('no header contribution');
  return {
    items: /** @type {any[]} */ (call[1]),
    dispatch: /** @type {(id: string) => void} */ (call[2]),
  };
}

/** @param {string} id */
function headerItem(id) {
  return headerNow().items.find((item) => item.id === id);
}

/** Every listing request, as URL search params. */
function listingQueries() {
  return fetchMock.mock.calls
    .map(([url]) => new URL(String(url), 'http://hub'))
    .filter((u) => u.pathname.endsWith('/api/artifacts'))
    .map((u) => Object.fromEntries(u.searchParams));
}

async function boot() {
  mod.initSidebarArtifacts();
  await settle();
}

beforeEach(async () => {
  vi.resetModules();
  vi.useFakeTimers({ toFake: ['Date'] });
  vi.setSystemTime(Date.UTC(2026, 9, 30, 12, 0));
  document.documentElement.setAttribute('data-facility-timezone', 'UTC');
  document.documentElement.dataset.uiMode = 'expert';
  localStorage.clear();
  mocks.sessionId = 'sess-1';
  mocks.sessionListeners.length = 0;
  for (const fn of [
    mocks.contributeLocal,
    mocks.sendSelect,
    mocks.sendAction,
    mocks.activateTab,
    mocks.setServiceMenuExtras,
  ]) {
    fn.mockClear();
  }
  mocks.hasDockedTile.mockReset();
  mocks.hasDockedTile.mockReturnValue(true);
  server = defaultServer();
  fetchMock = vi.fn(async (/** @type {string} */ url) => route(url));
  vi.stubGlobal('fetch', fetchMock);
  events = stubEventSource();
  skeleton();
  mod = await import(MODULE);
});

afterEach(() => {
  mod.teardownSidebarArtifacts();
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  document.body.replaceChildren();
  document.documentElement.removeAttribute('data-facility-timezone');
  delete document.documentElement.dataset.uiMode;
  document.documentElement.removeAttribute('data-osprey-storage-scope');
});

/* ---- fetching ---------------------------------------------------------- */

describe('fetching', () => {
  test('fetches both lists through the proxy, split by run and scoped to the session', async () => {
    await boot();
    expect(listingQueries()).toEqual([
      { run: 'false', session_id: 'sess-1' },
      { run: 'true', session_id: 'sess-1' },
    ]);
    expect(String(fetchMock.mock.calls[0][0])).toMatch(/^\/panel\/artifacts\/api\/artifacts\?/);
  });

  test('All sessions drops the session; a session change brings the scope back', async () => {
    await boot();
    /** @type {HTMLElement} */ (document.querySelector('[data-art-menu]')).click();
    const allSessions = /** @type {HTMLElement} */ (
      document.querySelector('[data-art-all-sessions]')
    );
    allSessions.click();
    await settle();
    expect(listingQueries().slice(2)).toEqual([{ run: 'false' }, { run: 'true' }]);
    expect(allSessions.getAttribute('aria-checked')).toBe('true');
    expect(document.querySelector('[data-art-menu]')?.hasAttribute('data-art-scope')).toBe(true);

    mocks.sessionId = 'sess-2';
    for (const listener of mocks.sessionListeners) listener('sess-2');
    await settle();
    expect(listingQueries().slice(4)).toEqual([
      { run: 'false', session_id: 'sess-2' },
      { run: 'true', session_id: 'sess-2' },
    ]);
    expect(allSessions.getAttribute('aria-checked')).toBe('false');
  });

  test('Refresh fetches both lists again and closes the menu', async () => {
    await boot();
    /** @type {HTMLElement} */ (document.querySelector('[data-art-menu]')).click();
    const menu = /** @type {HTMLElement} */ (document.getElementById('sidebar-artifacts-menu'));
    expect(menu.hidden).toBe(false);
    /** @type {HTMLElement} */ (menu.querySelector('[data-art-refresh]')).click();
    await settle();
    expect(listingQueries()).toHaveLength(4);
    expect(menu.hidden).toBe(true);
  });

  test('scrolling a list to its end fetches the next page by its cursor', async () => {
    server.artifacts.first = { artifacts: [A3, A2], total: 3, next_cursor: 'c1' };
    server.artifacts.c1 = { artifacts: [A1], total: 3, next_cursor: null };
    await boot();
    expect(ids('sidebar-artifacts-list')).toEqual(['a3', 'a2']);
    document.getElementById('sidebar-artifacts-list')?.dispatchEvent(new Event('scroll'));
    await settle();
    expect(listingQueries()[2]).toEqual({ run: 'false', session_id: 'sess-1', cursor: 'c1' });
    expect(ids('sidebar-artifacts-list')).toEqual(['a3', 'a1', 'a2']);
  });

  test('a reconnect fetches again; the first open does not', async () => {
    await boot();
    const [source] = events.opened;
    expect(source.url).toBe('/panel/artifacts/api/events');
    source.onopen?.();
    await settle();
    expect(listingQueries()).toHaveLength(2);
    source.onopen?.();
    await settle();
    expect(listingQueries()).toHaveLength(4);
  });
});

/* ---- drawing ----------------------------------------------------------- */

describe('drawing', () => {
  test('groups by type into the row list and the flyout list alike', async () => {
    await boot();
    for (const listId of ['sidebar-artifacts-list', 'sidebar-artifacts-flyout-list']) {
      expect(heads(listId)).toEqual(['Reports', 'Plots']);
      expect(ids(listId)).toEqual(['a3', 'a1', 'a2']);
    }
    expect(ids('sidebar-runs-list')).toEqual(['r1']);
    expect(heads('sidebar-runs-list')).toEqual([]);
  });

  test('Activity groups by day, and the grouping is kept per user', async () => {
    document.documentElement.setAttribute('data-osprey-storage-scope', 'alice');
    await boot();
    /** @type {HTMLElement} */ (document.querySelector('[data-art-view="activity"]')).click();
    expect(heads('sidebar-artifacts-list')).toEqual(['Today']);
    expect(ids('sidebar-artifacts-list')).toEqual(['a3', 'a2', 'a1']);
    expect(document.querySelector('[data-art-view="activity"]')?.getAttribute('aria-checked')).toBe(
      'true'
    );
    expect(document.querySelector('[data-art-view="types"]')?.getAttribute('aria-checked')).toBe(
      'false'
    );
    expect(localStorage.getItem('osprey-sidebar-artifacts-view--alice')).toBe('activity');
    expect(localStorage.getItem('osprey-sidebar-artifacts-view')).toBeNull();

    mod.teardownSidebarArtifacts();
    await boot();
    expect(heads('sidebar-artifacts-list')).toEqual(['Today']);
  });

  test('counts go on the rows, the flyout and the collapsed tooltips', async () => {
    await boot();
    const counts = [...document.querySelectorAll('[data-art-count]')].map((el) => el.textContent);
    expect(counts).toEqual(['3', '3']);
    expect(document.querySelector('[data-runs-count]')?.textContent).toBe('1');
    expect(/** @type {HTMLElement} */ (document.querySelector('[data-runs-count]')).hidden).toBe(
      false
    );
    expect(
      /** @type {HTMLElement} */ (document.querySelector('[data-panel-id="artifacts"]')).dataset.tip
    ).toBe('Artifacts · 3');
    expect(document.getElementById('sidebar-runs-toggle')?.dataset.tip).toBe('Script runs · 1');
  });

  test('the filter opens on its toggle while the list is short, and filters by title', async () => {
    await boot();
    const label = /** @type {HTMLElement} */ (document.querySelector('[data-art-filter]'));
    expect(label.hidden).toBe(true);
    /** @type {HTMLElement} */ (document.querySelector('[data-art-filter-toggle]')).click();
    expect(label.hidden).toBe(false);
    const input = /** @type {HTMLInputElement} */ (label.querySelector('input'));
    input.value = 'REPORT';
    input.dispatchEvent(new Event('input', { bubbles: true }));
    expect(ids('sidebar-artifacts-list')).toEqual(['a3', 'a1']);
    expect(ids('sidebar-artifacts-flyout-list')).toEqual(['a3', 'a1']);
    input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    expect(ids('sidebar-artifacts-list')).toEqual(['a3', 'a1', 'a2']);
  });

  test('above eight artifacts the filter stays open', async () => {
    server.artifacts.first = { artifacts: [A3, A2, A1], total: 9, next_cursor: 'c1' };
    await boot();
    for (const label of document.querySelectorAll('[data-art-filter]')) {
      expect(/** @type {HTMLElement} */ (label).hidden).toBe(false);
    }
    expect(document.querySelector('[data-art-filter-toggle]')?.getAttribute('aria-pressed')).toBe(
      'true'
    );
  });

  test('runs without the sidebar skeleton', async () => {
    document.body.replaceChildren();
    await boot();
    expect(listingQueries()).toHaveLength(2);
    expect(mocks.sendSelect).not.toHaveBeenCalled();
  });
});

/* ---- selection --------------------------------------------------------- */

describe('selection', () => {
  test('the newest result is shown first and the viewer is told', async () => {
    await boot();
    expect(mocks.sendSelect).toHaveBeenLastCalledWith(viewer(), 'a3');
    expect(row('sidebar-artifacts-list', 'a3').getAttribute('aria-selected')).toBe('true');
  });

  test('a row click shows it and never asks for the agent focus', async () => {
    await boot();
    row('sidebar-artifacts-list', 'a2').click();
    expect(shown()).toBe('a2');
    expect(row('sidebar-artifacts-list', 'a2').getAttribute('aria-selected')).toBe('true');
    expect(row('sidebar-artifacts-list', 'a3').getAttribute('aria-selected')).toBe('false');
    expect(mocks.activateTab).not.toHaveBeenCalled();
    const urls = fetchMock.mock.calls.map(([url]) => String(url));
    expect(urls.some((url) => url.includes('/api/focus'))).toBe(false);
  });

  test('a row click brings the artifacts tile up when it is not docked', async () => {
    mocks.hasDockedTile.mockReturnValue(false);
    await boot();
    row('sidebar-artifacts-list', 'a1').click();
    expect(mocks.activateTab).toHaveBeenCalledWith('artifacts', { userInitiated: true });
  });

  test('a script run row is shown too', async () => {
    await boot();
    row('sidebar-runs-list', 'r1').click();
    expect(shown()).toBe('r1');
    expect(row('sidebar-runs-list', 'r1').getAttribute('aria-selected')).toBe('true');
  });

  test('the viewer announcing itself gets the selection again', async () => {
    await boot();
    const frameWindow = {};
    Object.defineProperty(viewer(), 'contentWindow', { value: frameWindow });
    mocks.sendSelect.mockClear();
    const announce = (/** @type {unknown} */ source) => {
      const event = new MessageEvent('message', {
        data: { type: 'osprey-artifact-ready' },
        origin: window.location.origin,
      });
      Object.defineProperty(event, 'source', { value: source });
      window.dispatchEvent(event);
    };
    announce({});
    expect(mocks.sendSelect).not.toHaveBeenCalled();
    announce(frameWindow);
    expect(mocks.sendSelect).toHaveBeenCalledWith(viewer(), 'a3');
  });

  test('Up and Down move between the rows of a list', async () => {
    await boot();
    const first = row('sidebar-artifacts-list', 'a3');
    first.focus();
    first.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowDown', bubbles: true }));
    expect(document.activeElement).toBe(row('sidebar-artifacts-list', 'a1'));
  });
});

/* ---- the pane header --------------------------------------------------- */

describe('the pane header', () => {
  test('crumb and pager follow the list as drawn; next and prev step through it', async () => {
    await boot();
    expect(headerItem('crumb').leaf).toBe('Tunes report');
    expect(headerItem('pager').text).toBe('1 of 3');
    expect(headerItem('prev').disabled).toBe(true);
    headerNow().dispatch('next');
    await settle();
    expect(shown()).toBe('a1');
    expect(headerItem('pager').text).toBe('2 of 3');
    headerNow().dispatch('next');
    await settle();
    expect(shown()).toBe('a2');
    expect(headerItem('next').disabled).toBe(true);
    headerNow().dispatch('prev');
    await settle();
    expect(shown()).toBe('a1');
  });

  test('next past the last held artifact fetches the next page', async () => {
    server.artifacts.first = { artifacts: [A3], total: 2, next_cursor: 'c1' };
    server.artifacts.c1 = { artifacts: [A1], total: 2, next_cursor: null };
    await boot();
    expect(headerItem('next').disabled).toBe(false);
    headerNow().dispatch('next');
    await settle();
    expect(shown()).toBe('a1');
    expect(headerItem('pager').text).toBe('2 of 2');
  });

  test('Open full size and Save act on the shown artifact', async () => {
    await boot();
    const open = vi.spyOn(window, 'open').mockReturnValue(null);
    headerNow().dispatch('open');
    expect(open).toHaveBeenCalledWith(
      '/panel/artifacts/api/markdown/a3/rendered?theme=dark',
      '_blank',
      'noopener'
    );
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
    headerNow().dispatch('save');
    expect(click).toHaveBeenCalledTimes(1);
  });

  test('with no artifact the header is the bare crumb and disabled actions', async () => {
    server.artifacts.first = { artifacts: [], total: 0 };
    server.runs.first = { artifacts: [], total: 0 };
    await boot();
    expect(headerNow().items.map((item) => item.id)).toEqual(['crumb', 'open', 'save']);
    expect(headerItem('crumb').leaf).toBe('');
    expect(headerItem('open').disabled).toBe(true);
    expect(mocks.contributeLocal.mock.lastCall?.[0]).toBe('artifacts');
  });
});

/* ---- the event stream -------------------------------------------------- */

describe('the event stream', () => {
  test('a new artifact lands in the list the server says, and is counted', async () => {
    await boot();
    events.emit({ type: 'artifact', ...art('a4', 'table_html', '2026-10-30T11:30:00Z', 'Scan') });
    events.emit({
      type: 'artifact',
      ...art('r2', 'json', '2026-10-30T11:40:00Z', 'Result', { run_record: true }),
    });
    expect(ids('sidebar-artifacts-list')).toContain('a4');
    expect(ids('sidebar-runs-list')).toEqual(['r2', 'r1']);
    expect(document.querySelector('[data-art-count]')?.textContent).toBe('4');
    expect(document.querySelector('[data-runs-count]')?.textContent).toBe('2');
  });

  test('an artifact of another session stays out; an untagged one comes in', async () => {
    await boot();
    events.emit({
      type: 'artifact',
      ...art('x', 'markdown', '2026-10-30T11:50:00Z', 'X', { session_id: 'sess-9' }),
    });
    events.emit({
      type: 'artifact',
      ...art('u', 'markdown', '2026-10-30T11:50:00Z', 'U', { session_id: '' }),
    });
    expect(ids('sidebar-artifacts-list')).not.toContain('x');
    expect(ids('sidebar-artifacts-list')).toContain('u');
  });

  test('an update merges; a delete removes and the next artifact is shown', async () => {
    await boot();
    events.emit({ type: 'artifact_updated', ...A3, title: 'Tunes report v2' });
    expect(headerItem('crumb').leaf).toBe('Tunes report v2');
    events.emit({ type: 'artifact_deleted', id: 'a3' });
    expect(ids('sidebar-artifacts-list')).toEqual(['a1', 'a2']);
    expect(document.querySelector('[data-art-count]')?.textContent).toBe('2');
    expect(shown()).toBe('a2');
  });
});

/* ---- agent focus ------------------------------------------------------- */

describe('agent focus', () => {
  test('Expert: a focus frame jumps the selection, automatic or not', async () => {
    await boot();
    row('sidebar-artifacts-list', 'a2').click();
    events.emit({ type: 'focus', domain: 'artifact', id: 'a1', automatic: true });
    await settle();
    expect(shown()).toBe('a1');
  });

  test('Simple: an automatic focus does not take the operator pick; a requested one does', async () => {
    document.documentElement.dataset.uiMode = 'simple';
    await boot();
    row('sidebar-artifacts-list', 'a2').click();
    events.emit({ type: 'focus', domain: 'artifact', id: 'a1', automatic: true });
    await settle();
    expect(shown()).toBe('a2');
    events.emit({ type: 'focus', domain: 'artifact', id: 'a1' });
    await settle();
    expect(shown()).toBe('a1');
  });

  test('Simple: with no pick the focus shows, unless it is a script run', async () => {
    document.documentElement.dataset.uiMode = 'simple';
    await boot();
    events.emit({ type: 'focus', domain: 'artifact', id: 'r1', automatic: true });
    await settle();
    expect(shown()).toBe('a3');
    events.emit({ type: 'focus', domain: 'artifact', id: 'a1', automatic: true });
    await settle();
    expect(shown()).toBe('a1');
  });

  test('an older artifact is fetched and held, not counted; another session is ignored', async () => {
    server.byId.old = art('old', 'markdown', '2026-10-01T09:00:00Z', 'Old report');
    server.byId.alien = art('alien', 'markdown', '2026-10-01T09:00:00Z', 'Alien', {
      session_id: 'x',
    });
    await boot();
    events.emit({ type: 'focus', domain: 'artifact', id: 'old' });
    await settle();
    expect(shown()).toBe('old');
    expect(ids('sidebar-artifacts-list')).toContain('old');
    expect(document.querySelector('[data-art-count]')?.textContent).toBe('3');
    events.emit({ type: 'focus', domain: 'artifact', id: 'alien' });
    await settle();
    expect(shown()).toBe('old');
  });
});

/* ---- the tile menu ----------------------------------------------------- */

describe('the tile menu', () => {
  test('lends the verbs of the shown artifact, sent to the viewer', async () => {
    await boot();
    const [panelId, extras] = mocks.setServiceMenuExtras.mock.calls[0];
    expect(panelId).toBe('artifacts');
    const items = extras();
    expect(items.map((/** @type {any} */ item) => item.label)).toEqual([
      'Pin artifact',
      'Print artifact',
      'Compose logbook entry',
      'Delete artifact',
    ]);
    items[3].run();
    expect(mocks.sendAction).toHaveBeenCalledWith(viewer(), 'a3', 'delete');
  });

  test('offers nothing while no viewer is there, and is withdrawn on teardown', async () => {
    await boot();
    const [, extras] = mocks.setServiceMenuExtras.mock.calls[0];
    viewer().remove();
    expect(extras()).toEqual([]);
    mod.teardownSidebarArtifacts();
    expect(mocks.setServiceMenuExtras).toHaveBeenLastCalledWith('artifacts', null);
  });
});
