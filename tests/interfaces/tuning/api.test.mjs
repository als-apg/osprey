// @ts-check
/**
 * Behavioral tests for the OSPREY Tuning REST client (api.js): the `api`
 * singleton that routes every call through `/api/proxy/<path>` and unwraps
 * JSON responses, throwing an annotated error on any non-2xx status.
 *
 * `fetch` is stubbed with a vi.fn returning a fake `Response`; each test
 * asserts the exact URL/options the client built and the value it returns
 * (or throws) for the stubbed response.
 *
 *   npx vitest run tests/interfaces/tuning/api.test.mjs
 *
 * @module tests/interfaces/tuning/api
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { api } from '../../../src/osprey/interfaces/tuning/static/js/api.js';

/** The stubbed global fetch, re-created per test. @type {import('vitest').Mock} */
let fetchMock;

/** Build a fake ok/error Response carrying `body` as its JSON/text payload.
 * @param {object} opts
 * @param {boolean} opts.ok
 * @param {number} opts.status
 * @param {any} [opts.body]
 * @returns {any} */
function fakeResponse({ ok, status, body }) {
  const text = typeof body === 'string' ? body : JSON.stringify(body ?? {});
  return {
    ok,
    status,
    json: async () => (typeof body === 'string' ? JSON.parse(body) : body),
    text: async () => text,
  };
}

beforeEach(() => {
  fetchMock = vi.fn();
  vi.stubGlobal('fetch', fetchMock);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('URL construction through /api/proxy', () => {
  it('GETs environments/list and returns the parsed body', async () => {
    fetchMock.mockResolvedValueOnce(fakeResponse({ ok: true, status: 200, body: { environments: ['sim'] } }));

    const result = await api.listEnvironments();

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url] = fetchMock.mock.calls[0];
    expect(url).toBe('/api/proxy/environments/list');
    expect(result).toEqual({ environments: ['sim'] });
  });

  it('percent-encodes path segments (run timestamp with a space)', async () => {
    fetchMock.mockResolvedValueOnce(fakeResponse({ ok: true, status: 200, body: { data: [] } }));

    await api.loadRun('2024-05-01 10:00');

    const [url] = fetchMock.mock.calls[0];
    expect(url).toBe('/api/proxy/runs/2024-05-01%2010%3A00');
  });

  it('percent-encodes the environment name for a details lookup', async () => {
    fetchMock.mockResolvedValueOnce(fakeResponse({ ok: true, status: 200, body: {} }));

    await api.getEnvironmentDetails('als/sim');

    const [url] = fetchMock.mock.calls[0];
    expect(url).toBe('/api/proxy/environments/als%2Fsim/details');
  });
});

describe('request options', () => {
  it('POSTs startOptimization with a JSON body and content-type header', async () => {
    fetchMock.mockResolvedValueOnce(fakeResponse({ ok: true, status: 200, body: { job_id: 'j1' } }));
    const config = { environment: 'sim', variables: ['q1'] };

    const result = await api.startOptimization(config);

    const [url, options] = fetchMock.mock.calls[0];
    expect(url).toBe('/api/proxy/optimization/start');
    expect(options.method).toBe('POST');
    expect(options.body).toBe(JSON.stringify(config));
    expect(options.headers['Content-Type']).toBe('application/json');
    expect(result).toEqual({ job_id: 'j1' });
  });
});

describe('error handling on a non-ok response', () => {
  it('throws an annotated error carrying the status and server detail', async () => {
    fetchMock.mockResolvedValueOnce(
      fakeResponse({ ok: false, status: 500, body: { error: 'boom' } })
    );

    await expect(api.getAvailableRuns()).rejects.toThrow('API error 500: boom');
  });
});
