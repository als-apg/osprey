// @ts-check
/**
 * The panel's scope switch (app.js, panel-toolbar.js): Curated · All PVs ·
 * Feedback as the buttons of the toolbar seg `#cf-views` when /api/info carries
 * the directory descriptor, and Explore | Feedback otherwise. Nothing of it goes
 * to the hub's tile bar.
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
 * Boot app.js at `hash` on the toolbar index.html serves, against an /api/info
 * that answers `info`, or rejects when `info` is null.
 * @param {Record<string, any> | null} info
 * @param {string} [hash]
 */
async function boot(info, hash = '') {
  document.body.innerHTML = `
    <div id="cf-panel">
      <header id="cf-toolbar">
        <button id="cf-sidebar-toggle" type="button" disabled></button>
        <div class="osprey-seg" id="cf-views" role="group" aria-label="View">
          <button type="button" data-view="explore" aria-pressed="true">Explore</button>
          <button type="button" data-view="feedback" aria-pressed="false">Feedback</button>
        </div>
        <span id="cf-result-count" hidden></span>
        <label id="cf-search-field"><input type="search" id="cf-search" disabled></label>
      </header>
      <div id="view-container"></div>
    </div>
    <button id="pipeline-badge"></button>
    <div id="pipeline-dropdown"></div>`;
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

  /** Every contribution published to the hub's tile bar. */
  const contributions = /** @type {any[][]} */ ([]);
  vi.doMock('/design-system/js/header-contrib.js', () => ({
    contributeHeader: (/** @type {any[]} */ items) => contributions.push(items),
    onHeaderAction: vi.fn(),
  }));

  let answer = info;
  vi.stubGlobal(
    'fetch',
    vi.fn(async (/** @type {any} */ url, /** @type {any} */ opts) => {
      const path = String(url);
      if (path.includes('/api/info') && answer === null) throw new TypeError('network');
      if (path.includes('/api/pipeline') && opts?.method === 'PUT') {
        // The answer after a switch carries `directory_after_switch`, or no descriptor.
        const { directory_after_switch: directory, ...rest } = answer ?? {};
        delete rest.directory;
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
  return { state, views, contributions };
}

/** Follow the current hash the way the browser does after a button sets it. */
function followHash() {
  for (const listener of hashListeners) listener(new Event('hashchange'));
}

/** @returns {HTMLButtonElement[]} */
function buttons() {
  return /** @type {HTMLButtonElement[]} */ ([
    ...document.querySelectorAll('#cf-views button[data-view]'),
  ]);
}

/** @returns {string[]} the buttons as `view:label`. */
function names() {
  return buttons().map((button) => `${button.dataset.view}:${button.textContent}`);
}

/** @returns {string[]} the views of the pressed buttons. */
function pressed() {
  return buttons()
    .filter((button) => button.getAttribute('aria-pressed') === 'true')
    .map((button) => /** @type {string} */ (button.dataset.view));
}

/** @param {string} view */
function click(view) {
  const button = buttons().find((candidate) => candidate.dataset.view === view);
  if (!button) throw new Error(`no button for ${view}`);
  button.click();
  followHash();
}

/** @param {string} hash */
function go(hash) {
  window.history.replaceState(null, '', `/${hash}`);
  followHash();
}

const WITH = ['explore:Curated', 'allpvs:All PVs', 'feedback:Feedback'];
const WITHOUT = ['explore:Explore', 'feedback:Feedback'];

describe('with the directory descriptor', () => {
  test('the seg reads Curated · All PVs · Feedback, Curated pressed', async () => {
    await boot(WITH_DIRECTORY);
    expect(names()).toEqual(WITH);
    expect(pressed()).toEqual(['explore']);
  });

  test('a button routes when clicked, and the pressed state follows', async () => {
    const { views } = await boot(WITH_DIRECTORY);
    click('allpvs');
    expect(location.hash).toBe('#allpvs');
    expect(views.allpvs.mount).toHaveBeenCalledTimes(1);
    expect(pressed()).toEqual(['allpvs']);
    click('feedback');
    expect(views.feedback.mount).toHaveBeenCalledTimes(1);
    expect(pressed()).toEqual(['feedback']);
  });

  test.each(['#/pv/SR01%3AX', '#/ioc/ioc-sr01'])('%s presses All PVs', async (hash) => {
    await boot(WITH_DIRECTORY, hash);
    expect(pressed()).toEqual(['allpvs']);
  });

  test('the feedback reason names the Curated tab', async () => {
    await boot({ ...WITH_DIRECTORY, pipeline_type: 'graph' }, '#feedback');
    expect(document.getElementById('cf-search-field')?.title).toBe('Search applies to Curated');
  });
});

describe('without the directory descriptor', () => {
  test('the seg keeps Explore | Feedback', async () => {
    await boot(WITHOUT_DIRECTORY);
    expect(names()).toEqual(WITHOUT);
    expect(pressed()).toEqual(['explore']);
  });

  test('the buttons still route', async () => {
    const { views } = await boot(WITHOUT_DIRECTORY);
    click('feedback');
    expect(views.feedback.mount).toHaveBeenCalledTimes(1);
    expect(pressed()).toEqual(['feedback']);
  });

  test('a directory route or an unknown name falls back to Explore', async () => {
    const { views } = await boot(WITHOUT_DIRECTORY, '#feedback');
    for (const hash of ['#allpvs', '#/pv/A', '#constructor']) {
      go(hash);
      expect(views.allpvs.mount).not.toHaveBeenCalled();
      expect(views.pv.mount).not.toHaveBeenCalled();
      expect(pressed()).toEqual(['explore']);
    }
  });

  test('a failed /api/info leaves Explore | Feedback', async () => {
    await boot(null);
    expect(names()).toEqual(WITHOUT);
  });

  test('a stray directoryLabel event adds no button', async () => {
    const { state } = await boot(WITHOUT_DIRECTORY);
    state.emit('directoryLabel', 'Channel Finder');
    expect(names()).toEqual(WITHOUT);
  });
});

describe('the All PVs label (D20)', () => {
  const NAMED = { ...WITH_DIRECTORY, directory: { ...DESCRIPTOR, label: 'Channel Finder' } };

  test('comes from the descriptor', async () => {
    await boot(NAMED);
    expect(names()[1]).toBe('allpvs:Channel Finder');
  });

  test('follows a directoryLabel event', async () => {
    const { state } = await boot(NAMED);
    state.emit('directoryLabel', 'Archiver Appliance');
    expect(names()[1]).toBe('allpvs:Archiver Appliance');
  });

  test('the button stays pressed when the label changes on its own page', async () => {
    const { state } = await boot(NAMED, '#allpvs');
    state.emit('directoryLabel', 'Archiver Appliance');
    expect(pressed()).toEqual(['allpvs']);
  });

  test('an event keeps the buttons in place', async () => {
    const { state } = await boot(NAMED);
    const before = buttons();
    state.emit('directoryLabel', 'Channel Finder');
    state.emit('directoryLabel', 'Archiver Appliance');
    expect(buttons()).toEqual(before);
    buttons().forEach((button, index) => expect(button).toBe(before[index]));
  });

  test('is text, never markup', async () => {
    const label = '<img src=x onerror=alert(1)>';
    const { state } = await boot({ ...WITH_DIRECTORY, directory: { ...DESCRIPTOR, label } });
    expect(names()[1]).toBe(`allpvs:${label}`);
    expect(document.querySelector('#cf-views img')).toBeNull();
    state.emit('directoryLabel', '<b>x</b>');
    expect(names()[1]).toBe('allpvs:<b>x</b>');
    expect(document.querySelector('#cf-views b')).toBeNull();
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

  test('the scope buttons appear when the answer starts carrying it', async () => {
    await boot({ ...WITHOUT_DIRECTORY, ...PIPELINES, directory_after_switch: DESCRIPTOR });
    await switchTo('graph');
    await vi.waitFor(() => expect(names()).toEqual(WITH));
    expect(pressed()).toEqual(['explore']);
  });

  test('they leave again when it stops', async () => {
    await boot({ ...WITH_DIRECTORY, ...PIPELINES });
    await switchTo('graph');
    await vi.waitFor(() => expect(names()).toEqual(WITHOUT));
  });

  test('the new buttons route after the switch', async () => {
    const { views } = await boot({
      ...WITHOUT_DIRECTORY,
      ...PIPELINES,
      directory_after_switch: DESCRIPTOR,
    });
    await switchTo('graph');
    await vi.waitFor(() => expect(names()).toHaveLength(3));
    click('allpvs');
    expect(views.allpvs.mount).toHaveBeenCalledTimes(1);
    expect(pressed()).toEqual(['allpvs']);
  });
});

describe('state.scopeQuery across a tab switch', () => {
  test('All PVs to Curated and back', async () => {
    const { state, views } = await boot(WITH_DIRECTORY, '#allpvs');
    /** @type {string[]} */
    const seenByCurated = [];
    views.explore.mount.mockImplementation(() => seenByCurated.push(state.scopeQuery));
    /** @type {string[]} */
    const seenByAllPvs = [];
    views.allpvs.mount.mockImplementation(() => seenByAllPvs.push(state.scopeQuery));

    state.scopeQuery = 'beam current';
    click('explore');
    expect(seenByCurated).toEqual(['beam current']);

    state.scopeQuery = 'orbit';
    click('allpvs');
    expect(seenByAllPvs).toEqual(['orbit']);
    expect(state.scopeQuery).toBe('orbit');
  });
});

describe('the tile bar', () => {
  test.each([
    ['with', WITH_DIRECTORY],
    ['without', WITHOUT_DIRECTORY],
  ])('receives no nav contribution %s the descriptor', async (_, info) => {
    const { state, contributions } = await boot(info, '#allpvs');
    state.emit('directoryLabel', 'Channel Finder');
    click('feedback');
    expect(contributions.flat().filter((item) => item?.kind === 'nav')).toEqual([]);
  });
});
