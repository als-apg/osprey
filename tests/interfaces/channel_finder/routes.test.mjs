// @ts-check
/**
 * Hash routes (app.js) and the directory fields of the shared state (state.js).
 *   npx vitest run tests/interfaces/channel_finder/routes.test.mjs
 *
 * app.js boots on import, so each routing case re-imports it into a fresh module
 * registry with the view modules replaced by spies, and removes the hashchange
 * listener the boot added, so one case's app never answers the next case's hash.
 */

import { test, expect, vi, afterEach, describe } from 'vitest';

const JS = '../../../src/osprey/interfaces/channel_finder/static/js';
const STATE_MODULE = `${JS}/state.js`;
const APP_MODULE = `${JS}/app.js`;

const DESCRIPTOR = { label: 'All PVs', configured_services: [], canopy: true, ariel: true };

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
  vi.doUnmock('/design-system/js/frame-params.js');
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  document.body.innerHTML = '';
  window.history.replaceState(null, '', '/');
});

/**
 * Boot app.js at `hash` against an /api/info that answers `info`, or rejects when
 * `info` is null. `beforeApp` runs with the fresh state and the view spies, ahead
 * of the import that boots the app.
 * @param {string} hash
 * @param {Record<string, any> | null} [info]
 * @param {(state: any, views: Record<string, { mount: import('vitest').Mock }>) => void} [beforeApp]
 */
async function boot(
  hash,
  info = { pipeline_type: 'hierarchical', directory: DESCRIPTOR },
  beforeApp = () => {}
) {
  document.body.innerHTML = `
    <div id="view-container"></div>
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
  /** @type {{ handler: ((id: string, value: string | undefined) => void) | null }} */
  const header = { handler: null };
  vi.doMock('/design-system/js/header-contrib.js', () => ({
    contributeHeader: vi.fn(),
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
        answer = { ...answer, directory: answer?.directory_after_switch };
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
  const { state } = await import(STATE_MODULE);
  beforeApp(state, views);
  const app = await import(APP_MODULE);
  await vi.waitFor(() => expect(hashListeners).toHaveLength(1));
  return { state, app, views, header };
}

/**
 * Follow a hash change the way the browser does.
 * @param {string} hash
 */
function go(hash) {
  window.history.replaceState(null, '', `/${hash}`);
  for (const listener of hashListeners) listener(new Event('hashchange'));
}

describe('parseRoute', () => {
  test.each([
    ['', { view: 'explore', params: {} }],
    ['#', { view: 'explore', params: {} }],
    ['#explore', { view: 'explore', params: {} }],
    ['#feedback', { view: 'feedback', params: {} }],
    ['#allpvs', { view: 'allpvs', params: {} }],
    ['#/allpvs', { view: 'allpvs', params: {} }],
    ['#/allpvs?q=beam%20current', { view: 'allpvs', params: { q: 'beam current' } }],
    ['#/allpvs?q=', { view: 'allpvs', params: { q: '' } }],
    ['#/allpvs?other=1&q=a', { view: 'allpvs', params: { q: 'a' } }],
    ['#/pv/SR01%3AX', { view: 'pv', params: { name: 'SR01:X' } }],
    ['#/ioc/ioc-sr01', { view: 'ioc', params: { name: 'ioc-sr01' } }],
    ['#nonsense', { view: 'nonsense', params: {} }],
  ])('%s', async (hash, route) => {
    const { app } = await boot('');
    expect(app.parseRoute(hash)).toEqual(route);
  });

  test.each([
    '#/pv',
    '#/pv/',
    '#pv',
    '#/ioc',
    '#/pv/a/b',
    '#/explore/x',
    '#/pv/%zz',
    '#/allpvs?q=%zz',
  ])('%s has no route', async (hash) => {
    const { app } = await boot('');
    expect(app.parseRoute(hash)).toBeNull();
  });

  test.each(['SR01C:BPM1:X.VAL', 'a/b', 'odd name+1%', 'ÄÖ?&=#'])(
    'the name %s round-trips as one encoded segment',
    async (name) => {
      const { app } = await boot('');
      expect(app.parseRoute(`#/pv/${encodeURIComponent(name)}`)).toEqual({
        view: 'pv',
        params: { name },
      });
      expect(app.parseRoute(`#/ioc/${encodeURIComponent(name)}`)).toEqual({
        view: 'ioc',
        params: { name },
      });
    }
  );

  test.each(['beam current', 'a+b', 'x&q=y', '100%', 'ÄÖ?#'])(
    'the words %s round-trip through q, with %%20 for a space',
    async (q) => {
      const { app } = await boot('');
      expect(encodeURIComponent('a b')).toBe('a%20b');
      expect(app.parseRoute(`#/allpvs?q=${encodeURIComponent(q)}`)).toEqual({
        view: 'allpvs',
        params: { q },
      });
    }
  );
});

describe('routing', () => {
  test('a deep link mounts its view with the decoded params', async () => {
    const { views } = await boot('#/pv/SR01%3AX');
    expect(views.pv.mount).toHaveBeenCalledTimes(1);
    expect(views.pv.mount.mock.calls[0][1]).toEqual({ name: 'SR01:X' });
    expect(views.explore.mount).not.toHaveBeenCalled();
  });

  test('the ioc and allpvs routes reach their views', async () => {
    const { views } = await boot('#/ioc/ioc-sr01');
    expect(views.ioc.mount.mock.calls[0][1]).toEqual({ name: 'ioc-sr01' });
    go('#/allpvs?q=beam%20current');
    expect(views.ioc.unmount).toHaveBeenCalledTimes(1);
    expect(views.allpvs.mount.mock.calls[0][1]).toEqual({ q: 'beam current' });
  });

  test('a param change unmounts and re-mounts the same view on a cleared container', async () => {
    const { views } = await boot('#/pv/A');
    const container = /** @type {HTMLElement} */ (document.getElementById('view-container'));
    container.innerHTML = '<p>page of A</p>';

    go('#/pv/B');

    expect(views.pv.unmount).toHaveBeenCalledTimes(1);
    expect(views.pv.mount).toHaveBeenCalledTimes(2);
    expect(views.pv.mount.mock.calls[1][1]).toEqual({ name: 'B' });
    expect(container.innerHTML).toBe('');
  });

  test('new words re-mount All PVs; the same route does not', async () => {
    const { views } = await boot('#/allpvs?q=a');
    go('#/allpvs?q=a');
    expect(views.allpvs.mount).toHaveBeenCalledTimes(1);
    go('#/allpvs?q=b');
    expect(views.allpvs.mount).toHaveBeenCalledTimes(2);
  });

  test('bare view names keep working, and explore is the default', async () => {
    const { views } = await boot('');
    expect(views.explore.mount).toHaveBeenCalledTimes(1);
    expect(views.explore.mount.mock.calls[0][1]).toEqual({});
    go('#feedback');
    expect(views.explore.unmount).toHaveBeenCalledTimes(1);
    expect(views.feedback.mount).toHaveBeenCalledTimes(1);
    go('#feedback');
    expect(views.feedback.mount).toHaveBeenCalledTimes(1);
  });

  test.each(['#nonsense', '#/pv/', '#/pv/%zz', '#constructor', '#/toString'])(
    'the unknown route %s falls back to explore',
    async (hash) => {
      const { views } = await boot(hash);
      expect(views.explore.mount).toHaveBeenCalledTimes(1);
      expect(views.pv.mount).not.toHaveBeenCalled();
    }
  );

  describe('while state.directory is null', () => {
    const noDirectory = { pipeline_type: 'hierarchical' };

    test.each(['#allpvs', '#/allpvs?q=a', '#/pv/A', '#/ioc/B'])(
      '%s falls back to explore',
      async (hash) => {
        const { views, state } = await boot(hash, noDirectory);
        expect(state.directory).toBeNull();
        expect(views.explore.mount).toHaveBeenCalledTimes(1);
        for (const view of ['allpvs', 'pv', 'ioc']) {
          expect(views[view].mount).not.toHaveBeenCalled();
        }
      }
    );

    test('a failed /api/info leaves it null, and the routes fall back', async () => {
      const { views, state } = await boot('#/pv/A', null);
      expect(state.directory).toBeNull();
      expect(views.pv.mount).not.toHaveBeenCalled();
      expect(views.explore.mount).toHaveBeenCalledTimes(1);
    });

    test('the header ignores the three names, as it does an unknown one', async () => {
      const { header } = await boot('#feedback', noDirectory);
      for (const name of ['allpvs', 'pv', 'ioc', 'constructor']) {
        header.handler?.('view', name);
        expect(location.hash).toBe('#feedback');
      }
      header.handler?.('view', 'explore');
      expect(location.hash).toBe('#explore');
    });
  });

  test('with a descriptor, the header reaches All PVs', async () => {
    const { header } = await boot('#feedback');
    header.handler?.('view', 'allpvs');
    expect(location.hash).toBe('#allpvs');
  });

  test('state.directory is set from /api/info before the first route', async () => {
    const seen = /** @type {any[]} */ ([]);
    await boot('#/pv/A', undefined, (state, views) => {
      views.pv.mount.mockImplementation(() => seen.push(state.directory));
    });
    expect(seen).toEqual([DESCRIPTOR]);
  });

  test('a later /api/info replaces state.directory, and the route is re-judged', async () => {
    const { state, views } = await boot('#/pv/A', {
      pipeline_type: 'hierarchical',
      available_pipelines: ['hierarchical', 'graph'],
      directory: DESCRIPTOR,
    });
    /** @type {HTMLElement} */ (
      document.querySelector('.pipeline-dropdown-item[data-pipeline="graph"]')
    ).click();
    await vi.waitFor(() => expect(views.explore.mount).toHaveBeenCalledTimes(1));

    expect(state.directory).toBeNull();
    expect(views.pv.unmount).toHaveBeenCalledTimes(1);
  });

  test('a pipeline switch unmounts and re-mounts the same route', async () => {
    const { views } = await boot('#/pv/A', {
      pipeline_type: 'hierarchical',
      available_pipelines: ['hierarchical', 'graph'],
      directory: DESCRIPTOR,
      directory_after_switch: DESCRIPTOR,
    });
    /** @type {HTMLElement} */ (
      document.querySelector('.pipeline-dropdown-item[data-pipeline="graph"]')
    ).click();
    await vi.waitFor(() => expect(views.pv.mount).toHaveBeenCalledTimes(2));
    expect(views.pv.unmount).toHaveBeenCalledTimes(1);
    expect(views.pv.mount.mock.calls[1][1]).toEqual({ name: 'A' });
  });
});

describe('state', () => {
  test('declares the scope fields with empty defaults', async () => {
    vi.resetModules();
    const { state } = await import(STATE_MODULE);
    expect(state.scopeQuery).toBe('');
    expect(state.directory).toBeNull();
  });

  test('directoryLabel updates state.directory.label before other listeners', async () => {
    vi.resetModules();
    const { state } = await import(STATE_MODULE);
    state.directory = { ...DESCRIPTOR };
    const seen = /** @type {string[]} */ ([]);
    state.on('directoryLabel', () => seen.push(state.directory.label));
    state.emit('directoryLabel', 'Channel Finder');
    expect(seen).toEqual(['Channel Finder']);
  });

  test('directoryLabel without a descriptor leaves it null', async () => {
    vi.resetModules();
    const { state } = await import(STATE_MODULE);
    state.emit('directoryLabel', 'Channel Finder');
    expect(state.directory).toBeNull();
  });

  test('curatedResults reaches a listener', async () => {
    vi.resetModules();
    const { state } = await import(STATE_MODULE);
    const listener = vi.fn();
    state.on('curatedResults', listener);
    state.emit('curatedResults', { query: 'beam', total: 0 });
    expect(listener).toHaveBeenCalledWith({ query: 'beam', total: 0 });
  });
});
