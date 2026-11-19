/**
 * Unit tests for the OPTIMIZER panel's sidecar client (api-client.js): request
 * paths under the panel prefix, the revision header, and refusal sentences
 * relayed verbatim.
 */

import { test, expect, describe, vi } from 'vitest';

import {
  REVISION_HEADER,
  createApi,
  errorIssues,
  resultPath,
  saveDraft,
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

describe('errorIssues', () => {
  test('a FastAPI detail string is relayed verbatim', () => {
    expect(errorIssues(502, { detail: 'optimizer service unreachable' })).toEqual([
      'optimizer service unreachable',
    ]);
  });
  test('a 422 validation list becomes its messages, without the body segment', () => {
    const detail = [{ loc: ['body', 'n_iterations'], msg: 'Input should be >= 0' }];
    expect(errorIssues(422, { detail })).toEqual(['n_iterations: Input should be >= 0']);
  });
  test('no body falls back to the status', () => {
    expect(errorIssues(500, null)).toEqual(['request failed (HTTP 500)']);
  });
});

describe('a 422 refusal with two items', () => {
  const detail = [
    { loc: ['body', 'n_iterations'], msg: 'Input should be greater than 0' },
    { loc: ['body', 'objective_name'], msg: 'Field required' },
  ];
  const refusing = () => createApi({ prefix: '', fetchImpl: async () => respond(422, { detail }) });

  test('gives two issues, and one sentence for a caller that shows the error', async () => {
    const res = await refusing().post('relay/optimizations', {});
    expect(res.issues).toEqual([
      'n_iterations: Input should be greater than 0',
      'objective_name: Field required',
    ]);
    expect(res.error).toBe(
      'n_iterations: Input should be greater than 0; objective_name: Field required'
    );
  });

  test('a refused save names each item as its own issue', async () => {
    const saved = await saveDraft(refusing(), null, {});
    expect(saved.id).toBeNull();
    expect(saved.issues).toEqual([
      'n_iterations: Input should be greater than 0',
      'objective_name: Field required',
    ]);
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

  test.each([
    [200, { owner: 'op1' }, { owner: 'op1', ownerKnown: true }],
    [200, { owner: null }, { owner: null, ownerKnown: false }],
    [404, { detail: 'no route' }, { owner: null, ownerKnown: null }],
    [200, {}, { owner: null, ownerKnown: null }],
  ])('session() reads relay/session: %i %j', async (status, body, expected) => {
    const fetchImpl = vi.fn(async () => respond(status, body));
    const api = createApi({ prefix: '/panel/optimizer', fetchImpl });
    expect(await api.session()).toEqual(expected);
    expect(/** @type {any} */ (fetchImpl.mock.calls[0])[0]).toBe('/panel/optimizer/relay/session');
  });
});

describe('the list and result reads', () => {
  function recording() {
    const fetchImpl = vi.fn(async () => respond(200, []));
    const api = createApi({ prefix: '/panel/optimizer', fetchImpl });
    const calls = () => /** @type {any[][]} */ (fetchImpl.mock.calls);
    return { api, urls: () => calls().map((c) => c[0]), inits: () => calls().map((c) => c[1]) };
  }

  test('jobs() and results() read the two lists, with GET and no body', async () => {
    const { api, urls, inits } = recording();
    await api.jobs();
    await api.results();
    expect(urls()).toEqual(['/panel/optimizer/jobs', '/panel/optimizer/results']);
    expect(inits().every((init) => init.method === 'GET' && init.body === undefined)).toBe(true);
  });

  test('result() names the stored run by its two segments, each escaped', async () => {
    const { api, urls } = recording();
    await api.result('va orbit/2026-11-16_21-15-04');
    expect(urls()).toEqual(['/panel/optimizer/results/va%20orbit/2026-11-16_21-15-04']);
  });

  test('result() of an id that names no stored run sends nothing and says why', async () => {
    const { api, urls } = recording();
    expect(await api.result('no-slash')).toMatchObject({ ok: false, status: 0 });
    expect(urls()).toEqual([]);
  });

  test('observables() repeats each name and names the target', async () => {
    const { api, urls } = recording();
    await api.observables('injector', ['charge', 'loss'], 'standin');
    expect(urls()).toEqual([
      '/panel/optimizer/environments/injector/observables?names=charge&names=loss&target=standin',
    ]);
  });

  test.each([
    ['env/2026-11-16_21-15-04', 'results/env/2026-11-16_21-15-04'],
    ['a/b/c', 'results/a%2Fb/c'],
    ['env/', null],
    ['/ts', null],
    ['plain', null],
  ])('resultPath(%j)', (runId, expected) => {
    expect(resultPath(runId)).toBe(expected);
  });
});
