// @ts-check
import { describe, expect, it, vi } from 'vitest';

import {
  CLIENT_HEADER,
  PENDING_MESSAGE,
  canPutBack,
  createApi,
  parseRefusal,
  putBackBody,
} from './api.js';

/**
 * @typedef {object} Call
 * @property {string} url
 * @property {RequestInit} init
 */

/**
 * A fetch fake answering every call with `answer` and recording each call.
 * @param {() => Promise<Response>} answer
 */
function fakeFetch(answer) {
  /** @type {Call[]} */
  const calls = [];
  const fetchImpl = vi.fn(async (/** @type {string} */ url, /** @type {RequestInit} */ init) => {
    calls.push({ url, init });
    return answer();
  });
  return { calls, fetchImpl };
}

/**
 * @param {number} status
 * @param {unknown} body
 * @returns {Response}
 */
function jsonResponse(status, body) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

/** @param {RequestInit} init */
function headersOf(init) {
  return /** @type {Record<string, string>} */ (init.headers ?? {});
}

/** @param {RequestInit} init */
function bodyOf(init) {
  return init.body === undefined ? undefined : JSON.parse(/** @type {string} */ (init.body));
}

/**
 * Every call the client offers, with arguments that make a valid request.
 * @param {ReturnType<typeof createApi>} api
 */
function everyCall(api) {
  return [
    () => api.openSession(),
    () => api.takeOver(),
    () => api.release(),
    () => api.state(),
    () => api.stop(),
    () => api.shot({ n_shots: 1, average: 'mean' }),
    () => api.step(),
    () => api.apply('digest'),
    () => api.putBack({ model: 'SR', values: [{ name: 'HCM1', before: 0.1 }] }),
    () => api.setup(true),
    () => api.continuous({ n_shots: 1, average: 'mean', max_shots: 10 }),
    () => api.heartbeat(),
    () => api.references(),
    () => api.saveReference('golden'),
    () => api.loadReference('golden'),
    () => api.selectReference('golden'),
    () => api.shots(),
  ];
}

describe('request URLs', () => {
  it('no request URL starts with a slash', async () => {
    const { calls, fetchImpl } = fakeFetch(async () => jsonResponse(200, {}));
    const api = createApi({ fetchImpl, getClientId: () => 'c1' });
    for (const call of everyCall(api)) {
      await call();
    }
    expect(calls).toHaveLength(everyCall(api).length);
    for (const { url } of calls) {
      expect(url.startsWith('/')).toBe(false);
      expect(url.startsWith('api/')).toBe(true);
    }
  });

  it('each call reaches its route with its method', async () => {
    const { calls, fetchImpl } = fakeFetch(async () => jsonResponse(200, {}));
    const api = createApi({ fetchImpl, getClientId: () => 'c1' });
    for (const call of everyCall(api)) {
      await call();
    }
    expect(calls.map(({ url, init }) => `${init.method} ${url}`)).toEqual([
      'POST api/session',
      'POST api/takeover',
      'DELETE api/session',
      'GET api/state',
      'POST api/stop',
      'POST api/shot',
      'GET api/step',
      'POST api/apply',
      'POST api/apply',
      'POST api/setup',
      'POST api/continuous',
      'POST api/continuous/heartbeat',
      'GET api/references',
      'POST api/references',
      'GET api/references/golden',
      'POST api/references/select',
      'GET api/shots',
    ]);
  });

  it('a reference name is encoded into its path', async () => {
    const { calls, fetchImpl } = fakeFetch(async () => jsonResponse(200, {}));
    await createApi({ fetchImpl }).loadReference('a/b c');
    expect(calls[0].url).toBe('api/references/a%2Fb%20c');
  });
});

describe('client header', () => {
  /** @param {(api: ReturnType<typeof createApi>) => Promise<unknown>} call */
  async function headerOf(call) {
    const { calls, fetchImpl } = fakeFetch(async () => jsonResponse(200, {}));
    await call(createApi({ fetchImpl, getClientId: () => 'c1' }));
    return headersOf(calls[0].init)[CLIENT_HEADER];
  }

  it.each([
    ['release', (/** @type {any} */ api) => api.release()],
    ['shot', (/** @type {any} */ api) => api.shot({ n_shots: 2, average: 'median' })],
    ['apply', (/** @type {any} */ api) => api.apply('d')],
    ['put back', (/** @type {any} */ api) => api.putBack({ model: 'SR', values: [] })],
    ['setup', (/** @type {any} */ api) => api.setup(false)],
    ['continuous', (/** @type {any} */ api) => api.continuous({ n_shots: 1, average: 'mean', max_shots: null })],
    ['heartbeat', (/** @type {any} */ api) => api.heartbeat()],
    ['select reference', (/** @type {any} */ api) => api.selectReference(null)],
  ])('%s sends the client id', async (_name, call) => {
    expect(await headerOf(call)).toBe('c1');
  });

  it.each([
    ['open session', (/** @type {any} */ api) => api.openSession()],
    ['take over', (/** @type {any} */ api) => api.takeOver()],
    ['state', (/** @type {any} */ api) => api.state()],
    ['stop', (/** @type {any} */ api) => api.stop()],
    ['step', (/** @type {any} */ api) => api.step()],
    ['references', (/** @type {any} */ api) => api.references()],
    ['save reference', (/** @type {any} */ api) => api.saveReference('r')],
    ['shots', (/** @type {any} */ api) => api.shots()],
  ])('%s sends no client id', async (_name, call) => {
    expect(await headerOf(call)).toBeUndefined();
  });

  it('without a client id the header is left off', async () => {
    const { calls, fetchImpl } = fakeFetch(async () => jsonResponse(200, {}));
    await createApi({ fetchImpl, getClientId: () => null }).shot({ n_shots: 1, average: 'mean' });
    expect(headersOf(calls[0].init)[CLIENT_HEADER]).toBeUndefined();
  });

  it('the client id is read at call time', async () => {
    const { calls, fetchImpl } = fakeFetch(async () => jsonResponse(200, {}));
    /** @type {string | null} */
    let id = 'first';
    const api = createApi({ fetchImpl, getClientId: () => id });
    await api.heartbeat();
    id = 'second';
    await api.heartbeat();
    expect(calls.map(({ init }) => headersOf(init)[CLIENT_HEADER])).toEqual(['first', 'second']);
  });
});

describe('request bodies', () => {
  it('session, take-over, release and stop send no body', async () => {
    const { calls, fetchImpl } = fakeFetch(async () => jsonResponse(200, {}));
    const api = createApi({ fetchImpl, getClientId: () => 'c1' });
    await api.openSession();
    await api.takeOver();
    await api.release();
    await api.stop();
    for (const { init } of calls) {
      expect(init.body).toBeUndefined();
    }
  });

  it('action bodies carry exactly their fields as JSON', async () => {
    const { calls, fetchImpl } = fakeFetch(async () => jsonResponse(200, {}));
    const api = createApi({ fetchImpl, getClientId: () => 'c1' });
    await api.shot({ n_shots: 3, average: 'median' });
    await api.apply('abc');
    await api.setup(true);
    await api.continuous({ n_shots: 2, average: 'mean', max_shots: null });
    await api.saveReference('r1');
    await api.saveReference('r2', 'shot-7');
    await api.selectReference(null);
    expect(calls.map(({ init }) => bodyOf(init))).toEqual([
      { n_shots: 3, average: 'median' },
      { step_digest: 'abc' },
      { bunch_clean: true },
      { n_shots: 2, average: 'mean', max_shots: null },
      { name: 'r1' },
      { name: 'r2', shot_id: 'shot-7' },
      { name: null },
    ]);
    expect(headersOf(calls[0].init)['Content-Type']).toBe('application/json');
  });
});

describe('answers', () => {
  it('a 2xx answer is ok with its body', async () => {
    const { fetchImpl } = fakeFetch(async () => jsonResponse(202, { stop_budget_s: 15 }));
    expect(await createApi({ fetchImpl }).stop()).toEqual({
      kind: 'ok',
      status: 202,
      body: { stop_budget_s: 15 },
    });
  });

  it('a refusal is parsed with its code and detail', async () => {
    const { fetchImpl } = fakeFetch(async () =>
      jsonResponse(409, { detail: { code: 'session_held', detail: 'held by alice' } }),
    );
    expect(await createApi({ fetchImpl }).openSession()).toEqual({
      kind: 'refused',
      status: 409,
      code: 'session_held',
      detail: 'held by alice',
      moved: null,
      extras: {},
    });
  });

  it("a 502 refusal keeps its moved list", async () => {
    const moved = [{ name: 'HCM1', before: 0.5, outcome: 'written' }];
    const { fetchImpl } = fakeFetch(async () =>
      jsonResponse(502, { detail: { code: 'apply_failed', detail: 'write failed', moved } }),
    );
    const result = await createApi({ fetchImpl, getClientId: () => 'c' }).apply('d');
    expect(result).toMatchObject({ kind: 'refused', status: 502, code: 'apply_failed', moved });
  });

  it('a refusal keeps its other fields as extras', async () => {
    const { fetchImpl } = fakeFetch(async () =>
      jsonResponse(409, {
        detail: { code: 'action_stopped', detail: 'stopped', commands_fired: ['arm'] },
      }),
    );
    const result = await createApi({ fetchImpl }).shot({ n_shots: 1, average: 'mean' });
    expect(result).toMatchObject({ kind: 'refused', extras: { commands_fired: ['arm'] } });
  });

  it('a transport error is outcome pending, never a refusal', async () => {
    const api = createApi({
      fetchImpl: async () => {
        throw new TypeError('network down');
      },
    });
    expect(await api.apply('d')).toEqual({ kind: 'pending', status: null, message: PENDING_MESSAGE });
  });

  it('a non-JSON error body is outcome pending', async () => {
    const { fetchImpl } = fakeFetch(async () => new Response('<html>Bad Gateway</html>', { status: 502 }));
    expect(await createApi({ fetchImpl }).apply('d')).toEqual({
      kind: 'pending',
      status: 502,
      message: PENDING_MESSAGE,
    });
  });

  it('a non-JSON success body is outcome pending', async () => {
    const { fetchImpl } = fakeFetch(async () => new Response('ok', { status: 200 }));
    expect(await createApi({ fetchImpl }).setup(false)).toMatchObject({ kind: 'pending', status: 200 });
  });

  it('an error body that is not a refusal is outcome pending', async () => {
    const { fetchImpl } = fakeFetch(async () => jsonResponse(422, { detail: [{ msg: 'bad' }] }));
    expect(await createApi({ fetchImpl }).shot({ n_shots: 1, average: 'mean' })).toMatchObject({
      kind: 'pending',
      status: 422,
    });
  });

  it('the pending message names the state as the outcome of record', () => {
    expect(PENDING_MESSAGE).toBe('outcome pending — see state');
  });

  it('parseRefusal rejects shapes without a string code and detail', () => {
    expect(parseRefusal(null, 400)).toBeNull();
    expect(parseRefusal('x', 400)).toBeNull();
    expect(parseRefusal({ detail: 'text' }, 400)).toBeNull();
    expect(parseRefusal({ detail: { code: 'c' } }, 400)).toBeNull();
    expect(parseRefusal({ detail: { code: 1, detail: 'd' } }, 400)).toBeNull();
  });
});

describe('one action in flight', () => {
  /** A fetch whose answers are released by hand, one per call. */
  function heldFetch() {
    /** @type {Call[]} */
    const calls = [];
    /** @type {(() => void)[]} */
    const releases = [];
    /** @type {FetchLikeT} */
    const fetchImpl = (url, init) => {
      calls.push({ url, init });
      return new Promise((resolve) => {
        releases.push(() => resolve(jsonResponse(200, {})));
      });
    };
    return { calls, releases, fetchImpl };
  }
  /** @typedef {(url: string, init: RequestInit) => Promise<Response>} FetchLikeT */

  it('a second action is refused locally while one is in flight', async () => {
    const { calls, releases, fetchImpl } = heldFetch();
    const api = createApi({ fetchImpl, getClientId: () => 'c' });
    const first = api.shot({ n_shots: 1, average: 'mean' });
    expect(api.actionInFlight).toBe(true);
    for (const second of [
      api.apply('d'),
      api.putBack({ model: 'SR', values: [{ name: 'A', before: 1 }] }),
      api.setup(true),
      api.continuous({ n_shots: 1, average: 'mean', max_shots: 1 }),
      api.shot({ n_shots: 1, average: 'mean' }),
    ]) {
      expect(await second).toMatchObject({ kind: 'busy' });
    }
    expect(calls).toHaveLength(1);
    releases[0]();
    expect(await first).toMatchObject({ kind: 'ok' });
    expect(api.actionInFlight).toBe(false);
  });

  it('stop and the heartbeat are sent while an action is in flight', async () => {
    const { calls, releases, fetchImpl } = heldFetch();
    const api = createApi({ fetchImpl, getClientId: () => 'c' });
    const action = api.continuous({ n_shots: 1, average: 'mean', max_shots: null });
    const stop = api.stop();
    const beat = api.heartbeat();
    expect(calls.map(({ url }) => url)).toEqual(['api/continuous', 'api/stop', 'api/continuous/heartbeat']);
    releases.forEach((release) => release());
    expect(await stop).toMatchObject({ kind: 'ok' });
    expect(await beat).toMatchObject({ kind: 'ok' });
    expect(await action).toMatchObject({ kind: 'ok' });
  });

  it('the slot frees after a failed action', async () => {
    let fail = true;
    const api = createApi({
      getClientId: () => 'c',
      fetchImpl: async () => {
        if (fail) {
          fail = false;
          throw new TypeError('reset');
        }
        return jsonResponse(200, {});
      },
    });
    expect(await api.apply('d')).toMatchObject({ kind: 'pending' });
    expect(api.actionInFlight).toBe(false);
    expect(await api.apply('d')).toMatchObject({ kind: 'ok' });
  });
});

describe('put back', () => {
  const moved = [
    { name: 'HCM1', before: 0.1, outcome: 'written' },
    { name: 'VCM2', before: -0.2, outcome: 'written' },
    { name: 'HCM1', before: 0.3, outcome: 'written' },
  ];

  it.each([['apply'], ['setup']])('is offered after %s, stopped or not', (name) => {
    expect(canPutBack({ name, stopped: false, moved })).toBe(true);
    expect(canPutBack({ name, stopped: true, moved })).toBe(true);
  });

  it.each([['put_back'], ['shot']])('is not offered after %s', (name) => {
    expect(canPutBack({ name, stopped: false, moved })).toBe(false);
    expect(putBackBody({ model: 'SR', last_action: { name, stopped: false, moved } })).toBeNull();
  });

  it('is not offered without a last action or with nothing moved', () => {
    expect(canPutBack(null)).toBe(false);
    expect(canPutBack(undefined)).toBe(false);
    expect(canPutBack({ name: 'apply', stopped: true, moved: [] })).toBe(false);
  });

  it('is not built without a model', () => {
    expect(putBackBody({ model: null, last_action: { name: 'apply', stopped: false, moved } })).toBeNull();
  });

  it('takes the moved settings, keeping the earliest before per name', () => {
    expect(
      putBackBody({ model: 'SR', last_action: { name: 'apply', stopped: false, moved } }),
    ).toEqual({
      model: 'SR',
      values: [
        { name: 'HCM1', before: 0.1 },
        { name: 'VCM2', before: -0.2 },
      ],
    });
  });

  it('is sent as POST apply with the client id', async () => {
    const { calls, fetchImpl } = fakeFetch(async () => jsonResponse(200, {}));
    const body = putBackBody({ model: 'SR', last_action: { name: 'setup', stopped: true, moved } });
    expect(body).not.toBeNull();
    await createApi({ fetchImpl, getClientId: () => 'c9' }).putBack(
      /** @type {import('./api.js').PutBackBody} */ (body),
    );
    expect(calls[0].url).toBe('api/apply');
    expect(headersOf(calls[0].init)[CLIENT_HEADER]).toBe('c9');
    expect(bodyOf(calls[0].init)).toEqual(body);
  });
});
