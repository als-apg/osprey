// @ts-check
/**
 * The route and cross-panel helpers of directory-api.js.
 *   npx vitest run tests/interfaces/channel_finder/directory-api.test.mjs
 */

import { test, expect, vi, afterEach, describe } from 'vitest';

const JS = '../../../src/osprey/interfaces/channel_finder/static/js';
const STATE_MODULE = `${JS}/state.js`;
const DIRECTORY_API_MODULE = `${JS}/directory-api.js`;

const DESCRIPTOR = { label: 'All PVs', configured_services: [], canopy: true, ariel: true };

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  window.history.replaceState(null, '', '/');
});

describe('directory-api routes', () => {
  /** @param {any} [body] */
  function stubFetch(body = {}) {
    const fetchMock = vi.fn(async () => ({ ok: true, status: 200, json: async () => body }));
    vi.stubGlobal('fetch', fetchMock);
    return fetchMock;
  }

  /** @param {import('vitest').Mock} fetchMock */
  function lastCall(fetchMock) {
    const [url, opts] = /** @type {any[]} */ (fetchMock.mock.calls.at(-1));
    return { url: String(url), opts };
  }

  test('search repeats each filter and keeps page 0', async () => {
    const fetchMock = stubFetch({ outcome: 'found' });
    const { searchDirectory } = await import(DIRECTORY_API_MODULE);
    const page = await searchDirectory({
      q: 'beam current',
      name: 'SR*',
      origin: ['curated', 'directory'],
      ioc: ['ioc a', 'ioc&b'],
      record_type: ['ai'],
      page: 0,
      page_size: 50,
    });
    expect(page).toEqual({ outcome: 'found' });
    const { url } = lastCall(fetchMock);
    expect(url.startsWith('/api/directory/search?')).toBe(true);
    const params = new URLSearchParams(url.split('?')[1]);
    expect(params.get('q')).toBe('beam current');
    expect(params.get('name')).toBe('SR*');
    expect(params.getAll('origin')).toEqual(['curated', 'directory']);
    expect(params.getAll('ioc')).toEqual(['ioc a', 'ioc&b']);
    expect(params.getAll('record_type')).toEqual(['ai']);
    expect(params.get('page')).toBe('0');
    expect(params.get('page_size')).toBe('50');
  });

  test('search leaves out what is not set', async () => {
    const fetchMock = stubFetch();
    const { searchDirectory } = await import(DIRECTORY_API_MODULE);
    await searchDirectory({ q: '', origin: [] });
    expect(lastCall(fetchMock).url).toBe('/api/directory/search');
  });

  test('the CSV export is a url with the same filters', async () => {
    const { directoryExportUrl } = await import(DIRECTORY_API_MODULE);
    expect(directoryExportUrl({ q: 'a b', ioc: ['x', 'y'] })).toBe(
      '/api/directory/export.csv?q=a+b&ioc=x&ioc=y'
    );
    expect(directoryExportUrl({})).toBe('/api/directory/export.csv');
  });

  test('count takes optional words', async () => {
    const fetchMock = stubFetch();
    const { countDirectory } = await import(DIRECTORY_API_MODULE);
    await countDirectory();
    expect(lastCall(fetchMock).url).toBe('/api/directory/count');
    await countDirectory('a b');
    expect(lastCall(fetchMock).url).toBe('/api/directory/count?q=a+b');
  });

  test('pv and ioc encode the name as one segment', async () => {
    const fetchMock = stubFetch({ name: 'SR01:X/Y', outcome: 'unknown', sections: [] });
    const { fetchPvFacts, fetchIocFacts } = await import(DIRECTORY_API_MODULE);
    expect((await fetchPvFacts('SR01:X/Y')).outcome).toBe('unknown');
    expect(lastCall(fetchMock).url).toBe('/api/directory/pv/SR01%3AX%2FY');
    await fetchIocFacts('ioc a');
    expect(lastCall(fetchMock).url).toBe('/api/directory/ioc/ioc%20a');
  });

  test('logbook availability and values', async () => {
    const fetchMock = stubFetch([]);
    const { fetchLogbookAvailability, readValues } = await import(DIRECTORY_API_MODULE);
    await fetchLogbookAvailability();
    expect(lastCall(fetchMock).url).toBe('/api/directory/logbook/availability');
    await readValues(['A', 'B']);
    const { url, opts } = lastCall(fetchMock);
    expect(url).toBe('/api/directory/values?name=A&name=B');
    expect(opts.method).toBeUndefined();
    expect(opts.body).toBeUndefined();
  });

  test('the active alarms repeat each severity and take a limit', async () => {
    const fetchMock = stubFetch({ outcome: 'found' });
    const { fetchActiveAlarms } = await import(DIRECTORY_API_MODULE);
    await fetchActiveAlarms();
    expect(lastCall(fetchMock).url).toBe('/api/directory/alarms');
    await fetchActiveAlarms({ severity: ['MAJOR', 'INVALID'], limit: 0 });
    const { url, opts } = lastCall(fetchMock);
    expect(url.startsWith('/api/directory/alarms?')).toBe(true);
    const params = new URLSearchParams(url.split('?')[1]);
    expect(params.getAll('severity')).toEqual(['MAJOR', 'INVALID']);
    expect(params.get('limit')).toBe('0');
    expect(opts.method).toBeUndefined();
  });

  test('the PV facts and the values take an abort signal', async () => {
    const fetchMock = stubFetch([]);
    const { fetchPvFacts, readValues } = await import(DIRECTORY_API_MODULE);
    const { signal } = new AbortController();
    await fetchPvFacts('A', signal);
    expect(lastCall(fetchMock).opts.signal).toBe(signal);
    await readValues(['A'], signal);
    expect(lastCall(fetchMock).opts.signal).toBe(signal);
  });

  test('values of no names are none, without a request', async () => {
    const fetchMock = stubFetch([{ name: 'A' }]);
    const { readValues } = await import(DIRECTORY_API_MODULE);
    expect(await readValues([])).toEqual([]);
    expect(fetchMock).toHaveBeenCalledTimes(0);
  });
});

describe('cross-panel urls', () => {
  /** @param {string} path */
  function atPath(path) {
    window.history.replaceState(null, '', path);
  }

  test('panelUrl at the root mount', async () => {
    atPath('/panel/channel-finder/');
    const { panelUrl } = await import(DIRECTORY_API_MODULE);
    expect(panelUrl('canopy', 'trace/')).toBe('/panel/canopy/trace/');
    expect(panelUrl('ariel', 'api/search')).toBe('/panel/ariel/api/search');
  });

  test('panelUrl under a per-user mount', async () => {
    atPath('/u/a/panel/channel-finder/');
    const { panelUrl } = await import(DIRECTORY_API_MODULE);
    expect(panelUrl('ariel', 'api/search')).toBe('/u/a/panel/ariel/api/search');
    expect(panelUrl('ariel', '#entry?id=e%201')).toBe('/u/a/panel/ariel/#entry?id=e%201');
    expect(panelUrl('canopy', 'trace/')).toBe('/u/a/panel/canopy/trace/');
  });

  test('panelUrl standalone is null', async () => {
    atPath('/');
    const { panelUrl, plotUrl } = await import(DIRECTORY_API_MODULE);
    expect(panelUrl('ariel', 'api/search')).toBeNull();
    expect(plotUrl(['A'])).toBeNull();
  });

  test('the Plot url keeps row order under the mount', async () => {
    atPath('/u/a/panel/channel-finder/');
    const { plotUrl } = await import(DIRECTORY_API_MODULE);
    expect(plotUrl(['SR:B', 'SR:A'])).toBe(
      '/u/a/panel/canopy/embed/@trace?pv=SR%3AB&pv=SR%3AA&range=-1h'
    );
  });

  test('the Plot button is limited to 12 PVs', async () => {
    const { plotButton } = await import(DIRECTORY_API_MODULE);
    expect(plotButton(12)).toEqual({ disabled: false, label: 'Plot', title: '' });
    expect(plotButton(13)).toEqual({
      disabled: true,
      label: 'Plot (max 12)',
      title: 'Data Browser takes at most 12 PVs',
    });
  });

  /**
   * Import directory-api.js in the given frame: `embedded` is the query param,
   * `framed` whether the window has a parent other than itself.
   * @param {{ embedded: boolean, framed: boolean }} frame
   */
  async function inFrame({ embedded, framed }) {
    atPath(`/u/a/panel/channel-finder/${embedded ? '?embedded=true' : ''}`);
    vi.stubGlobal('parent', framed ? { postMessage: vi.fn() } : window);
    vi.resetModules();
    const { state } = await import(STATE_MODULE);
    state.directory = { ...DESCRIPTOR };
    return { state, api: await import(DIRECTORY_API_MODULE) };
  }

  test('the hand-off gate needs both the param and a real parent', async () => {
    const framed = await inFrame({ embedded: true, framed: true });
    expect(framed.api.handoffAllowed()).toBe(true);
    const popout = await inFrame({ embedded: true, framed: false });
    expect(popout.api.handoffAllowed()).toBe(false);
    const direct = await inFrame({ embedded: false, framed: true });
    expect(direct.api.handoffAllowed()).toBe(false);
  });

  test('the Plot check asks Trace once per check and needs a 200', async () => {
    const { api } = await inFrame({ embedded: true, framed: true });
    const fetchMock = vi.fn(async () => ({ status: 200 }));
    vi.stubGlobal('fetch', fetchMock);

    const check = api.createPlotCheck();
    expect(await check()).toBe(true);
    expect(await check()).toBe(true);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock).toHaveBeenCalledWith('/u/a/panel/canopy/trace/', { method: 'HEAD' });

    fetchMock.mockImplementation(async () => ({ status: 404 }));
    expect(await api.createPlotCheck()()).toBe(false);
  });

  test('a Trace check that does not answer is no Plot', async () => {
    const { api } = await inFrame({ embedded: true, framed: true });
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        throw new TypeError('network');
      })
    );
    expect(await api.createPlotCheck()()).toBe(false);
  });

  test('no descriptor, no CANOPY or a pop-out asks nothing', async () => {
    const fetchMock = vi.fn(async () => ({ status: 200 }));

    const noCanopy = await inFrame({ embedded: true, framed: true });
    vi.stubGlobal('fetch', fetchMock);
    noCanopy.state.directory = { ...DESCRIPTOR, canopy: false };
    expect(await noCanopy.api.createPlotCheck()()).toBe(false);
    noCanopy.state.directory = null;
    expect(await noCanopy.api.createPlotCheck()()).toBe(false);

    const popout = await inFrame({ embedded: true, framed: false });
    expect(await popout.api.createPlotCheck()()).toBe(false);

    expect(fetchMock).not.toHaveBeenCalled();
  });
});
