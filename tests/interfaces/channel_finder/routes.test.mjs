// @ts-check
/**
 * Hash routes (app.js) and the directory fields of the shared state (state.js).
 *   npx vitest run tests/interfaces/channel_finder/routes.test.mjs
 *
 * app.js boots on import, so each routing case re-imports it into a fresh module
 * registry with the view modules replaced by spies, and removes the hashchange
 * listener the boot added, so one case's app never answers the next case's hash.
 */

import './epics-words.mjs';
import { test, expect, vi, afterEach, describe } from 'vitest';

const JS = '../../../src/osprey/interfaces/channel_finder/static/js';
const STATE_MODULE = `${JS}/state.js`;
const APP_MODULE = `${JS}/app.js`;

const DESCRIPTOR = { label: 'All PVs', configured_services: [], canopy: true, ariel: true };

/** The panel's view modules, by the file and the export names app.js imports. */
const VIEW_MODULES = {
  explore: ['explore.js', 'mountExplore', 'unmountExplore'],
  feedback: ['feedback.js', 'mountFeedback', 'unmountFeedback'],
  channels: ['channels.js', 'mountChannels', 'unmountChannels'],
  channel: ['channel-page.js', 'mountChannelPage', 'unmountChannelPage'],
  server: ['server-page.js', 'mountServerPage', 'unmountServerPage'],
};

/** @type {EventListener[]} */
let hashListeners = [];

afterEach(() => {
  for (const listener of hashListeners) window.removeEventListener('hashchange', listener);
  hashListeners = [];
  for (const [file] of Object.values(VIEW_MODULES)) vi.doUnmock(`${JS}/${file}`);
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
  return { state, app, views };
}

/**
 * Follow a hash change the way the browser does.
 * @param {string} hash
 */
function go(hash) {
  window.history.replaceState(null, '', `/${hash}`);
  for (const listener of hashListeners) listener(new Event('hashchange'));
}

describe('routing', () => {
  test('a deep link mounts its view with the decoded params', async () => {
    const { views } = await boot('#/channel/SR01%3AX');
    expect(views.channel.mount).toHaveBeenCalledTimes(1);
    expect(views.channel.mount.mock.calls[0][1]).toEqual({ name: 'SR01:X' });
    expect(views.explore.mount).not.toHaveBeenCalled();
  });

  test('the ioc and channels routes reach their views', async () => {
    const { views } = await boot('#/server/ioc-sr01');
    expect(views.server.mount.mock.calls[0][1]).toEqual({ name: 'ioc-sr01' });
    go('#/channels?q=beam%20current');
    expect(views.server.unmount).toHaveBeenCalledTimes(1);
    expect(views.channels.mount.mock.calls[0][1]).toEqual({ q: 'beam current' });
  });

  test('a param change unmounts and re-mounts the same view on a cleared container', async () => {
    const { views } = await boot('#/channel/A');
    const container = /** @type {HTMLElement} */ (document.getElementById('view-container'));
    container.innerHTML = '<p>page of A</p>';

    go('#/channel/B');

    expect(views.channel.unmount).toHaveBeenCalledTimes(1);
    expect(views.channel.mount).toHaveBeenCalledTimes(2);
    expect(views.channel.mount.mock.calls[1][1]).toEqual({ name: 'B' });
    expect(container.innerHTML).toBe('');
  });

  test('new words re-mount All PVs; the same route does not', async () => {
    const { views } = await boot('#/channels?q=a');
    go('#/channels?q=a');
    expect(views.channels.mount).toHaveBeenCalledTimes(1);
    go('#/channels?q=b');
    expect(views.channels.mount).toHaveBeenCalledTimes(2);
  });

  describe('a view stating its route', () => {
    test('is written when the view states it on its first mount', async () => {
      const { views } = await boot('#/channels?server=X', undefined, (state, views) => {
        views.channels.mount.mockImplementation(() =>
          state.emit('replaceRoute', { view: 'channels', params: { q: '', server: ['X'] } })
        );
      });
      expect(views.channels.mount).toHaveBeenCalledTimes(1);
      expect(location.hash).toBe('#/channels?q=&server=X');
      go('#/channels?server=X');
      expect(views.channels.mount).toHaveBeenCalledTimes(2);
    });

    test('the written route in another param order does not re-mount', async () => {
      const { state, views } = await boot('#/channels?q=a');
      state.emit('replaceRoute', { view: 'channels', params: { q: 'a', server: ['X'] } });
      go('#/channels?server=X&q=a');
      expect(views.channels.mount).toHaveBeenCalledTimes(1);
    });

    test('replaces the hash without a history entry or a re-mount', async () => {
      const { state, views } = await boot('#/channels?q=a');
      const entries = window.history.length;
      state.emit('replaceRoute', { view: 'channels', params: { q: 'a', server: ['X'] } });
      expect(location.hash).toBe('#/channels?q=a&server=X');
      expect(window.history.length).toBe(entries);
      expect(views.channels.mount).toHaveBeenCalledTimes(1);
    });

    test('Back to the written hash mounts its words and filters', async () => {
      const { state, views } = await boot('#/channels?q=a');
      state.emit('replaceRoute', { view: 'channels', params: { q: 'a', server: ['X'] } });
      go('#/channel/P');
      go('#/channels?q=a&server=X');
      expect(views.channels.mount).toHaveBeenCalledTimes(2);
      expect(views.channels.mount.mock.calls[1][1]).toEqual({ q: 'a', server: ['X'] });
    });

    test('a route for a view that is not mounted is ignored', async () => {
      const { state, views } = await boot('#/channel/P');
      const entries = window.history.length;
      state.emit('replaceRoute', { view: 'channels', params: { q: 'a', server: ['X'] } });
      expect(location.hash).toBe('#/channel/P');
      expect(window.history.length).toBe(entries);
      go('#/channel/P');
      expect(views.channel.mount).toHaveBeenCalledTimes(1);
      expect(views.channels.mount).not.toHaveBeenCalled();
    });

    test('the hash the view was opened with mounts it again', async () => {
      const { state, views } = await boot('#/channels?q=a');
      state.emit('replaceRoute', { view: 'channels', params: { q: 'a', server: ['X'] } });
      go('#/channels?q=a');
      expect(views.channels.mount).toHaveBeenCalledTimes(2);
      expect(views.channels.mount.mock.calls[1][1]).toEqual({ q: 'a' });
    });
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

  test.each(['#nonsense', '#/channel/', '#/channel/%zz', '#constructor', '#/toString'])(
    'the unknown route %s falls back to explore',
    async (hash) => {
      const { views } = await boot(hash);
      expect(views.explore.mount).toHaveBeenCalledTimes(1);
      expect(views.channel.mount).not.toHaveBeenCalled();
    }
  );

  describe('while state.directory is null', () => {
    const noDirectory = { pipeline_type: 'hierarchical' };

    test.each(['#channels', '#/channels?q=a', '#/channel/A', '#/server/B'])(
      '%s falls back to explore',
      async (hash) => {
        const { views, state } = await boot(hash, noDirectory);
        expect(state.directory).toBeNull();
        expect(views.explore.mount).toHaveBeenCalledTimes(1);
        for (const view of ['channels', 'channel', 'server']) {
          expect(views[view].mount).not.toHaveBeenCalled();
        }
      }
    );

    test('a failed /api/info leaves it null, and the routes fall back', async () => {
      const { views, state } = await boot('#/channel/A', null);
      expect(state.directory).toBeNull();
      expect(views.channel.mount).not.toHaveBeenCalled();
      expect(views.explore.mount).toHaveBeenCalledTimes(1);
    });
  });

  test('state.directory is set from /api/info before the first route', async () => {
    const seen = /** @type {any[]} */ ([]);
    await boot('#/channel/A', undefined, (state, views) => {
      views.channel.mount.mockImplementation(() => seen.push(state.directory));
    });
    expect(seen).toEqual([DESCRIPTOR]);
  });

  test('a later /api/info replaces state.directory, and the route is re-judged', async () => {
    const { state, views } = await boot('#/channel/A', {
      pipeline_type: 'hierarchical',
      available_pipelines: ['hierarchical', 'graph'],
      directory: DESCRIPTOR,
    });
    /** @type {HTMLElement} */ (
      document.querySelector('.pipeline-dropdown-item[data-pipeline="graph"]')
    ).click();
    await vi.waitFor(() => expect(views.explore.mount).toHaveBeenCalledTimes(1));

    expect(state.directory).toBeNull();
    expect(views.channel.unmount).toHaveBeenCalledTimes(1);
  });

  test('a pipeline switch unmounts and re-mounts the same route', async () => {
    const { views } = await boot('#/channel/A', {
      pipeline_type: 'hierarchical',
      available_pipelines: ['hierarchical', 'graph'],
      directory: DESCRIPTOR,
      directory_after_switch: DESCRIPTOR,
    });
    /** @type {HTMLElement} */ (
      document.querySelector('.pipeline-dropdown-item[data-pipeline="graph"]')
    ).click();
    await vi.waitFor(() => expect(views.channel.mount).toHaveBeenCalledTimes(2));
    expect(views.channel.unmount).toHaveBeenCalledTimes(1);
    expect(views.channel.mount.mock.calls[1][1]).toEqual({ name: 'A' });
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
