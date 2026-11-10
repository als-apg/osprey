/**
 * Unit tests for the OPTIMIZER panel's sidecar client (api-client.js): request
 * paths under the panel prefix, the revision header, and refusal sentences
 * relayed verbatim.
 */

import { test, expect, describe, vi } from 'vitest';

import {
  REVISION_HEADER,
  createApi,
  errorDetail,
  seg,
  withQuery,
} from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/api-client.js';

/**
 * @param {number} status
 * @param {unknown} body
 * @param {Record<string, string>} [headers]
 */
function respond(status, body, headers = {}) {
  return new Response(body === undefined ? '' : JSON.stringify(body), { status, headers });
}

describe('withQuery', () => {
  test('repeats array keys and drops null values', () => {
    expect(
      withQuery('environments/x/variables', { names: ['a', 'b'], target: 'va', preset: null })
    ).toBe('environments/x/variables?names=a&names=b&target=va');
    expect(withQuery('jobs', {})).toBe('jobs');
  });
});

describe('errorDetail', () => {
  test('a FastAPI detail string is relayed verbatim', () => {
    expect(errorDetail(502, { detail: 'optimizer service unreachable' })).toBe(
      'optimizer service unreachable'
    );
  });
  test('a 422 validation list becomes its messages, without the body segment', () => {
    const detail = [{ loc: ['body', 'n_iterations'], msg: 'Input should be >= 0' }];
    expect(errorDetail(422, { detail })).toBe('n_iterations: Input should be >= 0');
  });
  test('no body falls back to the status', () => {
    expect(errorDetail(500, null)).toBe('request failed (HTTP 500)');
  });
});

describe('createApi', () => {
  test('GET is prefix-relative and carries the revision header back', async () => {
    const fetchImpl = vi.fn(async () =>
      respond(200, { n_iterations: 3 }, { [REVISION_HEADER]: 'rev-1' })
    );
    const api = createApi({ prefix: '/panel/optimizer', fetchImpl });
    const res = await api.get(`jobs/${seg('a/b')}/config`);
    expect(/** @type {any} */ (fetchImpl.mock.calls[0])[0]).toBe(
      '/panel/optimizer/jobs/a%2Fb/config'
    );
    expect(res).toMatchObject({
      ok: true,
      status: 200,
      revision: 'rev-1',
      data: { n_iterations: 3 },
    });
  });

  test('POST sends JSON, and sends no owner or token header', async () => {
    const fetchImpl = vi.fn(async () => respond(200, { job_id: 'j' }));
    const api = createApi({ prefix: '', fetchImpl });
    await api.post('relay/optimizations', { objective_name: 'charge' });
    const [url, init] = /** @type {any} */ (fetchImpl.mock.calls[0]);
    expect(url).toBe('/relay/optimizations');
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body)).toEqual({ objective_name: 'charge' });
    const headers = Object.keys(init.headers).map((k) => k.toLowerCase());
    expect(headers.some((k) => k.includes('owner') || k.includes('token'))).toBe(false);
  });

  test('start puts the revision in the query and sends no body', async () => {
    const fetchImpl = vi.fn(async () => respond(200, {}));
    const api = createApi({ prefix: '', fetchImpl });
    await api.post('relay/jobs/j/start', undefined, { config_revision: 'abc' });
    const [url, init] = /** @type {any} */ (fetchImpl.mock.calls[0]);
    expect(url).toBe('/relay/jobs/j/start?config_revision=abc');
    expect(init.body).toBeUndefined();
  });

  test('a refusal resolves with the service sentence; a network failure with status 0', async () => {
    const refusing = createApi({
      prefix: '',
      fetchImpl: async () => respond(409, { detail: "Job j's draft changed since it was shown" }),
    });
    expect(await refusing.post('relay/jobs/j/start')).toMatchObject({
      ok: false,
      status: 409,
      error: "Job j's draft changed since it was shown",
    });
    const down = createApi({
      prefix: '',
      fetchImpl: async () => {
        throw new TypeError('network');
      },
    });
    expect(await down.get('service/health')).toMatchObject({ ok: false, status: 0 });
  });
});
