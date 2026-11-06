// @ts-check
/**
 * Unit tests for the per-session web layout (session-layout.js):
 *   npx vitest run tests/interfaces/web_terminal/session-layout.test.mjs
 *
 * A switch applies the arriving session's stored layout and each panel's
 * page; a session with none keeps the layout on screen and stores it (a new
 * session inherits it); leaving writes the screen left; a restore is never
 * written back.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const mocks = vi.hoisted(() => ({
  /** @type {{ transcript_id: string | null } | null} */
  row: null,
  /** @type {((change: {reason: string}) => void)[]} */
  sessionListeners: [],
  /** @type {(() => void)[]} */
  dockListeners: [],
  /** @type {(() => void) | null} */
  restoredHook: null,
  /** @type {any} */
  dock: null,
  /** @type {Record<string, string | null>} */
  pageOf: {},
  /** @type {Record<string, any>} the hub's stored layouts by transcript */
  server: {},
  applyDock: vi.fn(),
  restorePanelPage: vi.fn(),
}));

const api = () => ({
  onDidLayoutChange: (/** @type {() => void} */ fn) => {
    mocks.dockListeners.push(fn);
    return { dispose() {} };
  },
  onDidActivePanelChange: (/** @type {() => void} */ fn) => {
    mocks.dockListeners.push(fn);
    return { dispose() {} };
  },
});

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/api.js', () => ({
  withPrefix: (/** @type {string} */ p) => p,
  fetchJSON: async (/** @type {string} */ url) => {
    const tid = decodeURIComponent(url.split('/')[3]);
    if (!(tid in mocks.server)) throw new Error('HTTP 404');
    return { transcript_id: tid, layout: mocks.server[tid] };
  },
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/session-model.js', () => ({
  activeRow: () => mocks.row,
  subscribe: (/** @type {any} */ fn) => {
    mocks.sessionListeners.push(fn);
    return () => {};
  },
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/dock-workspace.js', () => ({
  getDockApi: () => api(),
  currentDock: () => mocks.dock,
  applyDock: mocks.applyDock,
  setLayoutRestoredHook: (/** @type {() => void} */ fn) => {
    mocks.restoredHook = fn;
  },
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/panel-lifecycle.js', () => ({
  getPanelPage: (/** @type {string} */ id) => mocks.pageOf[id] ?? null,
  restorePanelPage: mocks.restorePanelPage,
}));

/** @type {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/session-layout.js')} */
let mod;

const TA = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa';
const TB = 'bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb';
const TNEW = 'cccccccc-cccc-4ccc-8ccc-cccccccccccc';

/** A stored dock with the named service panels open beside the terminal. */
const dockWith = (/** @type {string[]} */ ...ids) => ({
  version: 2,
  layout: { root: { keys: ['terminal', ...ids] } },
  panels: Object.fromEntries(ids.map((id) => [id, { title: id }])),
});

/** @type {ReturnType<typeof vi.fn>} */
let fetchMock;

/** @returns {{ tid: string, layout: any }[]} */
function puts() {
  return fetchMock.mock.calls
    .filter(([, init]) => init?.method === 'PUT')
    .map(([url, init]) => ({
      tid: decodeURIComponent(String(url).split('/')[3]),
      layout: JSON.parse(init.body).layout,
    }));
}

async function flush() {
  for (let i = 0; i < 5; i += 1) await Promise.resolve();
}

/** Move the tab to a session and let the model's listeners hear it. */
async function switchTo(/** @type {string | null} */ tid) {
  mocks.row = { transcript_id: tid };
  for (const fn of mocks.sessionListeners) fn({ reason: 'pointer' });
  await flush();
}

function dockChanged() {
  for (const fn of mocks.dockListeners) fn();
}

async function boot(/** @type {string | null} */ tid) {
  mocks.row = { transcript_id: tid };
  mod.initSessionLayout();
  mocks.restoredHook?.();
  await flush();
}

beforeEach(async () => {
  vi.useFakeTimers();
  vi.resetModules();
  mocks.row = null;
  mocks.sessionListeners = [];
  mocks.dockListeners = [];
  mocks.restoredHook = null;
  mocks.dock = dockWith('knowledge');
  mocks.pageOf = {};
  mocks.server = {};
  mocks.applyDock.mockReset();
  mocks.restorePanelPage.mockReset();
  // A stored layout applies: the screen becomes it.
  mocks.applyDock.mockImplementation((/** @type {any} */ value) => {
    mocks.dock = { version: value.version, layout: value.layout, panels: value.panels };
    return true;
  });
  fetchMock = vi.fn(async (/** @type {string} */ url, /** @type {any} */ init) => {
    const tid = decodeURIComponent(String(url).split('/')[3]);
    mocks.server[tid] = JSON.parse(init.body).layout;
    return { ok: true, status: 200 };
  });
  vi.stubGlobal('fetch', fetchMock);
  mod = await import('../../../src/osprey/interfaces/web_terminal/static/js/session-layout.js');
});

afterEach(() => {
  mod.resetSessionLayout();
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe('arriving at a session', () => {
  test('a session with a stored layout gets it back, panel pages included', async () => {
    mocks.server[TA] = null;
    mocks.server[TB] = {
      ...dockWith('notebook', 'artifacts'),
      pages: { notebook: '/panel/notebook/tree/run.ipynb', artifacts: '/panel/artifacts/x' },
    };
    await boot(TA);
    await switchTo(TB);

    expect(mocks.applyDock).toHaveBeenCalledOnce();
    expect(mocks.applyDock.mock.calls[0][0].layout).toEqual(
      dockWith('notebook', 'artifacts').layout
    );
    // The Artifacts viewer keeps its own last artifact: no page is pushed to it.
    expect(mocks.restorePanelPage.mock.calls).toEqual([
      ['notebook', '/panel/notebook/tree/run.ipynb'],
    ]);
  });

  test('a session with no stored layout keeps the screen and stores it', async () => {
    mocks.server[TA] = null;
    mocks.pageOf = { knowledge: '/panel/knowledge/page/beam' };
    await boot(TA);

    expect(mocks.applyDock).not.toHaveBeenCalled();
    expect(puts()).toEqual([
      {
        tid: TA,
        layout: { ...dockWith('knowledge'), pages: { knowledge: '/panel/knowledge/page/beam' } },
      },
    ]);
  });

  test('a new session inherits the layout on screen once it has a transcript', async () => {
    mocks.server[TA] = { ...dockWith('knowledge'), pages: {} };
    mocks.server[TNEW] = null;
    await boot(TA);
    vi.advanceTimersByTime(mod.SETTLE_MS);
    fetchMock.mockClear();

    // "+ New": the listing has no transcript for the key yet.
    await switchTo(null);
    mocks.dock = dockWith('knowledge', 'notebook');
    dockChanged();
    vi.advanceTimersByTime(mod.SAVE_DEBOUNCE_MS);
    expect(puts()).toEqual([]);

    await switchTo(TNEW);
    expect(mocks.applyDock).toHaveBeenCalledTimes(1); // only TA's, at boot
    expect(puts()).toEqual([
      { tid: TNEW, layout: { ...dockWith('knowledge', 'notebook'), pages: {} } },
    ]);
  });

  test('a session the hub does not know is neither applied nor overwritten', async () => {
    await boot(TB);
    expect(mocks.applyDock).not.toHaveBeenCalled();
    dockChanged();
    vi.advanceTimersByTime(mod.SAVE_DEBOUNCE_MS);
    expect(puts()).toEqual([]);
  });

  test('a stored layout the dock cannot take leaves the screen, which is stored', async () => {
    mocks.server[TA] = { ...dockWith('gone'), pages: {} };
    mocks.applyDock.mockReturnValue(false);
    await boot(TA);
    expect(puts().map((p) => p.tid)).toEqual([TA]);
    expect(puts()[0].layout.panels).toEqual({ knowledge: { title: 'knowledge' } });
  });

  test('only the last of two quick switches is applied', async () => {
    mocks.server[TA] = null;
    mocks.server[TB] = { ...dockWith('notebook'), pages: {} };
    mocks.server[TNEW] = { ...dockWith('knowledge'), pages: {} };
    await boot(TA);
    mocks.row = { transcript_id: TB };
    for (const fn of mocks.sessionListeners) fn({ reason: 'pointer' });
    mocks.row = { transcript_id: TNEW };
    for (const fn of mocks.sessionListeners) fn({ reason: 'pointer' });
    await flush();
    expect(mocks.applyDock).toHaveBeenCalledOnce();
    expect(mocks.applyDock.mock.calls[0][0].panels).toEqual({ knowledge: { title: 'knowledge' } });
  });
});

describe('writing', () => {
  test('a settled change writes once, debounced, to the session on screen', async () => {
    mocks.server[TA] = { ...dockWith('knowledge'), pages: {} };
    await boot(TA);
    vi.advanceTimersByTime(mod.SETTLE_MS);
    mocks.dock = dockWith('knowledge', 'notebook');
    dockChanged();
    dockChanged();
    expect(puts()).toEqual([]);
    vi.advanceTimersByTime(mod.SAVE_DEBOUNCE_MS);
    expect(puts()).toEqual([
      { tid: TA, layout: { ...dockWith('knowledge', 'notebook'), pages: {} } },
    ]);
  });

  test('a restore is not written back', async () => {
    mocks.server[TA] = null;
    mocks.server[TB] = { ...dockWith('notebook'), pages: {} };
    await boot(TA);
    fetchMock.mockClear();
    await switchTo(TB);
    // The apply's own burst of dock changes.
    dockChanged();
    vi.advanceTimersByTime(mod.SETTLE_MS + mod.SAVE_DEBOUNCE_MS);
    // A later change that leaves the screen as stored writes nothing either.
    dockChanged();
    vi.advanceTimersByTime(mod.SAVE_DEBOUNCE_MS);
    expect(puts().filter((p) => p.tid === TB)).toEqual([]);
  });

  test('leaving a session writes the screen it left, pages read at that moment', async () => {
    mocks.server[TA] = { ...dockWith('knowledge'), pages: {} };
    mocks.server[TB] = { ...dockWith('notebook'), pages: {} };
    await boot(TA);
    vi.advanceTimersByTime(mod.SETTLE_MS);
    fetchMock.mockClear();
    // The Knowledge page moved inside its frame; nothing told the hub.
    mocks.pageOf = { knowledge: '/panel/knowledge/page/orbit' };

    await switchTo(TB);
    expect(puts()[0]).toEqual({
      tid: TA,
      layout: { ...dockWith('knowledge'), pages: { knowledge: '/panel/knowledge/page/orbit' } },
    });
    expect(mocks.server[TA].pages).toEqual({ knowledge: '/panel/knowledge/page/orbit' });
  });

  test('switching away and back restores what was left', async () => {
    mocks.server[TA] = { ...dockWith('knowledge'), pages: {} };
    mocks.server[TB] = null;
    await boot(TA);
    vi.advanceTimersByTime(mod.SETTLE_MS);

    await switchTo(TB);
    // In B the operator closes Knowledge.
    mocks.dock = dockWith();
    dockChanged();
    vi.advanceTimersByTime(mod.SAVE_DEBOUNCE_MS);
    expect(mocks.server[TB].panels).toEqual({});

    await switchTo(TA);
    const last = mocks.applyDock.mock.calls.at(-1)?.[0];
    expect(last.panels).toEqual({ knowledge: { title: 'knowledge' } });
    expect(mocks.dock.panels).toEqual({ knowledge: { title: 'knowledge' } });
  });

  test('a panel frame load writes its new page', async () => {
    mocks.server[TA] = { ...dockWith('knowledge'), pages: {} };
    await boot(TA);
    vi.advanceTimersByTime(mod.SETTLE_MS);
    mocks.pageOf = { knowledge: '/panel/knowledge/page/rf' };
    const frame = document.createElement('iframe');
    frame.className = 'panel-iframe';
    document.body.appendChild(frame);
    frame.dispatchEvent(new Event('load'));
    vi.advanceTimersByTime(mod.SAVE_DEBOUNCE_MS);
    expect(puts().at(-1)?.layout.pages).toEqual({ knowledge: '/panel/knowledge/page/rf' });
    frame.remove();
  });

  test('the page going away writes a pending change with keepalive', async () => {
    mocks.server[TA] = { ...dockWith('knowledge'), pages: {} };
    await boot(TA);
    vi.advanceTimersByTime(mod.SETTLE_MS);
    mocks.dock = dockWith('knowledge', 'notebook');
    dockChanged();
    window.dispatchEvent(new Event('pagehide'));
    const call = fetchMock.mock.calls.find(([, init]) => init?.method === 'PUT');
    expect(call?.[1].keepalive).toBe(true);
  });
});

describe('a detached pane', () => {
  /** The page holder, with `knowledge` docked beside the terminal and then detached. */
  async function detachKnowledge(/** @type {string | null} */ page) {
    const { initLayoutHolder } =
      await import('../../../src/osprey/interfaces/web_terminal/static/js/dock-apply.js');
    const { detachPlacement } =
      await import('../../../src/osprey/interfaces/web_terminal/static/js/dock-open.js');
    const holder = initLayoutHolder({
      dockPx: { width: 1000, height: 600 },
      placement: detachPlacement,
    });
    holder.restore({
      split: 'row',
      items: [
        { tile: 't1', tabs: ['knowledge'], showing: 'knowledge', size: 0.6 },
        { tile: 't2', tabs: ['terminal'], showing: 'terminal', size: 0.4 },
      ],
    });
    return { holder, detach: () => holder.detach('knowledge', page) };
  }

  test('its page is read live from its frame, else the page it left with', async () => {
    const { detach } = await detachKnowledge('/panel/knowledge/page/beam');
    mocks.server[TA] = { ...dockWith('knowledge'), pages: {} };
    await boot(TA);
    vi.advanceTimersByTime(mod.SETTLE_MS);
    detach();
    // The persisted form still names it (dock-workspace's currentDock).
    mocks.dock = dockWith('knowledge');
    vi.advanceTimersByTime(mod.SAVE_DEBOUNCE_MS);
    expect(puts().at(-1)?.layout.pages).toEqual({ knowledge: '/panel/knowledge/page/beam' });

    mocks.pageOf = { knowledge: '/panel/knowledge/page/rf' };
    await switchTo(TB);
    expect(
      puts().find((p) => p.tid === TA && p.layout.pages.knowledge.endsWith('/rf'))
    ).toBeTruthy();
  });

  test('a session switch sends a detached panel to its session’s page, like a docked one', async () => {
    const { holder, detach } = await detachKnowledge(null);
    mocks.server[TA] = { ...dockWith('knowledge'), pages: {} };
    mocks.server[TB] = {
      ...dockWith('knowledge'),
      pages: { knowledge: '/panel/knowledge/page/rf' },
    };
    await boot(TA);
    vi.advanceTimersByTime(mod.SETTLE_MS);
    detach();

    await switchTo(TB);

    expect(holder.isDetached('knowledge')).toBe(true);
    expect(mocks.restorePanelPage.mock.calls).toEqual([['knowledge', '/panel/knowledge/page/rf']]);
  });

  test('a detach or a reattach saves the layout, debounced, with no dock change', async () => {
    const { holder, detach } = await detachKnowledge(null);
    mocks.server[TA] = { ...dockWith('knowledge'), pages: {} };
    await boot(TA);
    vi.advanceTimersByTime(mod.SETTLE_MS);
    fetchMock.mockClear();
    mocks.pageOf = { knowledge: '/panel/knowledge/page/orbit' };
    detach();
    expect(puts()).toEqual([]);
    vi.advanceTimersByTime(mod.SAVE_DEBOUNCE_MS);
    expect(puts()).toEqual([
      {
        tid: TA,
        layout: { ...dockWith('knowledge'), pages: { knowledge: '/panel/knowledge/page/orbit' } },
      },
    ]);
    expect(holder.isDetached('knowledge')).toBe(true);
  });
});
