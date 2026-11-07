// @ts-check
/**
 * The panel's tabs (app.js): Curated · All PVs · Feedback in the hub's tile bar
 * and in the standalone `.header-nav` when /api/info carries the directory
 * descriptor, and today's tabs otherwise.
 *   npx vitest run tests/interfaces/channel_finder/scope-tabs.test.mjs
 *
 * app.js boots on import, so each case re-imports it into a fresh module registry
 * with the view modules replaced by spies, and removes the hashchange listener the
 * boot added, so one case's app never answers the next case's hash.
 */

import { test, expect, vi, afterEach, describe } from 'vitest';

const JS = '../../../src/osprey/interfaces/channel_finder/static/js';

const DESCRIPTOR = { label: 'All PVs', configured_services: [], canopy: true, ariel: true };

const WITH_DIRECTORY = { pipeline_type: 'hierarchical', directory: DESCRIPTOR };
const WITHOUT_DIRECTORY = { pipeline_type: 'hierarchical' };

/** The panel's view modules, by the file and the export names app.js imports. */
const VIEW_MODULES = {
  explore: ['explore.js', 'mountExplore', 'unmountExplore'],
  feedback: ['feedback.js', 'mountFeedback', 'unmountFeedback'],
  allpvs: ['allpvs.js', 'mountAllPvs', 'unmountAllPvs'],
  pv: ['pv-page.js', 'mountPvPage', 'unmountPvPage'],
  ioc: ['ioc-page.js', 'mountIocPage', 'unmountIocPage'],
};

/** @type {EventListener[]} */
let hashListeners = [];

afterEach(() => {
  for (const listener of hashListeners) window.removeEventListener('hashchange', listener);
  hashListeners = [];
  for (const [file] of Object.values(VIEW_MODULES)) vi.doUnmock(`${JS}/${file}`);
  vi.doUnmock('/design-system/js/header-contrib.js');
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  document.body.innerHTML = '';
  window.history.replaceState(null, '', '/');
});

/**
 * Boot app.js at `hash` on the page index.html serves, against an /api/info that
 * answers `info`, or rejects when `info` is null.
 * @param {Record<string, any> | null} info
 * @param {string} [hash]
 */
async function boot(info, hash = '') {
  document.body.innerHTML = `
    <div id="view-container"></div>
    <button id="pipeline-badge"></button>
    <div id="pipeline-dropdown"></div>
    <nav class="header-nav" id="header-nav">
      <a href="#explore" class="nav-link active" data-view="explore">Explore</a>
      <a href="#feedback" class="nav-link" data-view="feedback">Feedback</a>
    </nav>`;
  window.history.replaceState(null, '', `/${hash}`);

  /** @type {Record<string, { mount: import('vitest').Mock, unmount: import('vitest').Mock }>} */
  const views = {};
  for (const [view, [file, mountName, unmountName]] of Object.entries(VIEW_MODULES)) {
    views[view] = { mount: vi.fn(), unmount: vi.fn() };
    vi.doMock(`${JS}/${file}`, () => ({
      [mountName]: views[view].mount,
      [unmountName]: views[view].unmount,
    }));
  }

  /** Every contribution app.js publishes to the hub's tile bar. */
  const contributions = /** @type {any[][]} */ ([]);
  /** @type {{ handler: ((id: string, value: string | undefined) => void) | null }} */
  const header = { handler: null };
  vi.doMock('/design-system/js/header-contrib.js', () => ({
    contributeHeader: (/** @type {any[]} */ items) => contributions.push(items),
    onHeaderAction: (/** @type {any} */ handler) => {
      header.handler = handler;
    },
  }));

  let answer = info;
  vi.stubGlobal(
    'fetch',
    vi.fn(async (/** @type {any} */ url, /** @type {any} */ opts) => {
      const path = String(url);
      if (path.includes('/api/info') && answer === null) throw new TypeError('network');
      if (path.includes('/api/pipeline') && opts?.method === 'PUT') {
        const { directory_after_switch: directory, ...rest } = answer ?? {};
        const { pipeline_type: type } = JSON.parse(opts.body);
        answer = directory
          ? { ...rest, directory, pipeline_type: type }
          : { ...rest, pipeline_type: type };
      }
      return { ok: true, json: async () => (path.includes('/api/info') ? answer : {}) };
    })
  );

  const add = window.addEventListener.bind(window);
  vi.spyOn(window, 'addEventListener').mockImplementation(
    (/** @type {any} */ type, /** @type {any} */ listener, /** @type {any} */ options) => {
      if (type === 'hashchange') hashListeners.push(listener);
      add(type, listener, options);
    }
  );

  vi.resetModules();
  const { state } = await import(`${JS}/state.js`);
  await import(`${JS}/app.js`);
  await vi.waitFor(() => expect(hashListeners).toHaveLength(1));
  return { state, views, header, contributions };
}

/** Follow the current hash the way the browser does after a tab sets it. */
function followHash() {
  for (const listener of hashListeners) listener(new Event('hashchange'));
}

/** @returns {HTMLAnchorElement[]} */
function standaloneTabs() {
  return [...document.querySelectorAll('.header-nav .nav-link')].map(
    (link) => /** @type {HTMLAnchorElement} */ (link)
  );
}

/** @returns {string[]} the standalone tabs as `view:label`. */
function standaloneNames() {
  return standaloneTabs().map((link) => `${link.dataset.view}:${link.textContent}`);
}

/** @returns {string[]} the views of the standalone tabs marked active. */
function standaloneActive() {
  return standaloneTabs()
    .filter((link) => link.classList.contains('active'))
    .map((link) => /** @type {string} */ (link.dataset.view));
}

/**
 * The switcher of the latest tile-bar contribution.
 * @param {any[][]} contributions
 * @returns {{ id: string, label: string, active: boolean }[]}
 */
function tileBarTabs(contributions) {
  return contributions.at(-1)?.find((item) => item.kind === 'nav' && item.id === 'view').items;
}

/**
 * @param {string} view
 */
function clickStandalone(view) {
  const link = standaloneTabs().find((candidate) => candidate.dataset.view === view);
  if (!link) throw new Error(`no standalone tab for ${view}`);
  link.click();
  followHash();
}

const TILE_BAR_WITH = [
  { id: 'explore', label: 'Curated', active: true },
  { id: 'allpvs', label: 'All PVs', active: false },
  { id: 'feedback', label: 'Feedback', active: false },
];

const TILE_BAR_WITHOUT = [
  { id: 'explore', label: 'Explore', active: true },
  { id: 'feedback', label: 'Feedback', active: false },
];

describe('with the directory descriptor', () => {
  test('the tile bar reads Curated · All PVs · Feedback', async () => {
    const { contributions } = await boot(WITH_DIRECTORY);
    expect(tileBarTabs(contributions)).toEqual(TILE_BAR_WITH);
  });

  test('the standalone nav is rewritten to Curated · All PVs · Feedback', async () => {
    await boot(WITH_DIRECTORY);
    expect(standaloneNames()).toEqual(['explore:Curated', 'allpvs:All PVs', 'feedback:Feedback']);
    expect(standaloneActive()).toEqual(['explore']);
  });

  test('every standalone tab links to its own view', async () => {
    await boot(WITH_DIRECTORY);
    expect(standaloneTabs().map((link) => link.getAttribute('href'))).toEqual([
      '#explore',
      '#allpvs',
      '#feedback',
    ]);
  });

  test('a rewritten standalone tab routes when clicked', async () => {
    const { views } = await boot(WITH_DIRECTORY);
    clickStandalone('allpvs');
    expect(location.hash).toBe('#allpvs');
    expect(views.allpvs.mount).toHaveBeenCalledTimes(1);
    expect(standaloneActive()).toEqual(['allpvs']);
    clickStandalone('feedback');
    expect(views.feedback.mount).toHaveBeenCalledTimes(1);
    expect(standaloneActive()).toEqual(['feedback']);
  });

  test('a tile-bar action routes to All PVs, and the switcher follows', async () => {
    const { views, header, contributions } = await boot(WITH_DIRECTORY);
    header.handler?.('view', 'allpvs');
    followHash();
    expect(views.allpvs.mount).toHaveBeenCalledTimes(1);
    expect(tileBarTabs(contributions).filter((tab) => tab.active)).toEqual([
      { id: 'allpvs', label: 'All PVs', active: true },
    ]);
  });
});

describe('without the directory descriptor', () => {
  test("the tile bar carries exactly today's tabs", async () => {
    const { contributions } = await boot(WITHOUT_DIRECTORY);
    expect(tileBarTabs(contributions)).toEqual(TILE_BAR_WITHOUT);
  });

  test("the standalone nav carries exactly today's tabs", async () => {
    await boot(WITHOUT_DIRECTORY);
    expect(standaloneNames()).toEqual(['explore:Explore', 'feedback:Feedback']);
    expect(standaloneTabs().map((link) => link.getAttribute('href'))).toEqual([
      '#explore',
      '#feedback',
    ]);
    expect(standaloneActive()).toEqual(['explore']);
  });

  test("today's standalone tabs still route", async () => {
    const { views } = await boot(WITHOUT_DIRECTORY);
    clickStandalone('feedback');
    expect(views.feedback.mount).toHaveBeenCalledTimes(1);
    expect(standaloneActive()).toEqual(['feedback']);
  });

  test("a failed /api/info leaves today's tabs in both places", async () => {
    const { contributions } = await boot(null);
    expect(tileBarTabs(contributions)).toEqual(TILE_BAR_WITHOUT);
    expect(standaloneNames()).toEqual(['explore:Explore', 'feedback:Feedback']);
  });

  test('a stray directoryLabel event adds no tab', async () => {
    const { state, contributions } = await boot(WITHOUT_DIRECTORY);
    state.emit('directoryLabel', 'Channel Finder');
    expect(tileBarTabs(contributions)).toEqual(TILE_BAR_WITHOUT);
    expect(standaloneNames()).toEqual(['explore:Explore', 'feedback:Feedback']);
  });
});

describe('the All PVs label (D20)', () => {
  const NAMED = { ...WITH_DIRECTORY, directory: { ...DESCRIPTOR, label: 'Channel Finder' } };

  test('comes from the descriptor, in both places', async () => {
    const { contributions } = await boot(NAMED);
    expect(tileBarTabs(contributions)[1].label).toBe('Channel Finder');
    expect(standaloneNames()[1]).toBe('allpvs:Channel Finder');
  });

  test('follows a directoryLabel event, in both places', async () => {
    const { state, contributions } = await boot(NAMED);
    state.emit('directoryLabel', 'Archiver Appliance');
    expect(tileBarTabs(contributions)[1].label).toBe('Archiver Appliance');
    expect(standaloneNames()[1]).toBe('allpvs:Archiver Appliance');
  });

  test('the tab stays active when the label changes on its own page', async () => {
    const { state, contributions } = await boot(NAMED, '#allpvs');
    state.emit('directoryLabel', 'Archiver Appliance');
    expect(standaloneActive()).toEqual(['allpvs']);
    expect(tileBarTabs(contributions)[1].active).toBe(true);
  });

  test('an event with the label unchanged keeps the focused tab in place', async () => {
    const { state } = await boot(NAMED);
    const link = standaloneTabs()[1];
    state.emit('directoryLabel', 'Channel Finder');
    expect(standaloneTabs()[1]).toBe(link);
  });

  test('is text, never markup', async () => {
    const label = '<img src=x onerror=alert(1)>';
    const { state } = await boot({ ...WITH_DIRECTORY, directory: { ...DESCRIPTOR, label } });
    expect(standaloneNames()[1]).toBe(`allpvs:${label}`);
    expect(document.querySelector('.header-nav img')).toBeNull();
    state.emit('directoryLabel', '<b>x</b>');
    expect(standaloneNames()[1]).toBe('allpvs:<b>x</b>');
    expect(document.querySelector('.header-nav b')).toBeNull();
  });
});

describe('a pipeline switch re-reads the descriptor', () => {
  /**
   * Switch the pipeline through the dropdown the panel builds.
   * @param {string} pipeline
   */
  async function switchTo(pipeline) {
    const item = /** @type {HTMLElement} */ (
      document.querySelector(`[data-pipeline="${pipeline}"]`)
    );
    item.click();
    await vi.waitFor(() =>
      expect(document.getElementById('pipeline-badge')?.textContent).toContain(
        pipeline.toUpperCase()
      )
    );
  }

  const PIPELINES = { available_pipelines: ['hierarchical', 'graph'] };

  test('tabs appear when the answer starts carrying it', async () => {
    const { contributions } = await boot({
      ...WITHOUT_DIRECTORY,
      ...PIPELINES,
      directory_after_switch: DESCRIPTOR,
    });
    await switchTo('graph');
    await vi.waitFor(() => expect(standaloneNames()).toHaveLength(3));
    expect(standaloneNames()).toEqual(['explore:Curated', 'allpvs:All PVs', 'feedback:Feedback']);
    expect(tileBarTabs(contributions)).toEqual(TILE_BAR_WITH);
  });

  test('the rewritten tabs keep routing after the switch', async () => {
    const { views } = await boot({
      ...WITHOUT_DIRECTORY,
      ...PIPELINES,
      directory_after_switch: DESCRIPTOR,
    });
    await switchTo('graph');
    await vi.waitFor(() => expect(standaloneNames()).toHaveLength(3));
    clickStandalone('allpvs');
    expect(views.allpvs.mount).toHaveBeenCalledTimes(1);
  });
});

describe('state.scopeQuery across a tab switch', () => {
  test('All PVs to Curated and back, by the standalone tabs', async () => {
    const { state, views } = await boot(WITH_DIRECTORY, '#allpvs');
    /** @type {string[]} */
    const seenByCurated = [];
    views.explore.mount.mockImplementation(() => seenByCurated.push(state.scopeQuery));
    /** @type {string[]} */
    const seenByAllPvs = [];
    views.allpvs.mount.mockImplementation(() => seenByAllPvs.push(state.scopeQuery));

    state.scopeQuery = 'beam current';
    clickStandalone('explore');
    expect(seenByCurated).toEqual(['beam current']);

    state.scopeQuery = 'orbit';
    clickStandalone('allpvs');
    expect(seenByAllPvs).toEqual(['orbit']);
    expect(state.scopeQuery).toBe('orbit');
  });

  test('All PVs to Curated and back, by the tile bar', async () => {
    const { state, views, header } = await boot(WITH_DIRECTORY, '#allpvs');
    /** @type {string[]} */
    const seenByCurated = [];
    views.explore.mount.mockImplementation(() => seenByCurated.push(state.scopeQuery));
    /** @type {string[]} */
    const seenByAllPvs = [];
    views.allpvs.mount.mockImplementation(() => seenByAllPvs.push(state.scopeQuery));

    state.scopeQuery = 'beam current';
    header.handler?.('view', 'explore');
    followHash();
    expect(seenByCurated).toEqual(['beam current']);

    state.scopeQuery = 'orbit';
    header.handler?.('view', 'allpvs');
    followHash();
    expect(seenByAllPvs).toEqual(['orbit']);
  });
});
