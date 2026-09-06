// @ts-check
/**
 * Unit tests for the chat SSE transport (chat-client.js):
 *   npx vitest run tests/interfaces/web_terminal/chat-client.test.mjs
 *
 * The module is pure networking — no DOM — so every test drives it through a
 * stubbed global `fetch`. The streaming tests build a fake `Response` whose
 * body yields caller-chosen string chunks; the core surface under test is that
 * a `data: {json}\n\n` frame split across two reads is buffered and delivered
 * exactly once, that heartbeat comments are ignored, and that the terminal
 * condition (`result`, or a non-`AssistantMessageError` error) ends the turn.
 *
 * Module isolation: `vi.resetModules()` + a fresh dynamic import per test. The
 * module holds no cross-call state, but this keeps each test hermetic.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

/** @type {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/chat-client.js')} */
let chat;

beforeEach(async () => {
  vi.resetModules();
  chat = await import('../../../src/osprey/interfaces/web_terminal/static/js/chat-client.js');
});

afterEach(() => {
  vi.unstubAllGlobals();
  delete window.__OSPREY_PREFIX__;
});

/**
 * Build a fake streaming `Response` that emits *chunks* (strings) in order,
 * one per `read()`, then signals done.
 * @param {string[]} chunks
 * @param {{ ok?: boolean, status?: number, statusText?: string }} [opts]
 */
function streamResponse(chunks, opts = {}) {
  const { ok = true, status = 200, statusText = 'OK' } = opts;
  const encoder = new TextEncoder();
  let i = 0;
  const reader = {
    read: vi.fn(async () => {
      if (i < chunks.length) return { value: encoder.encode(chunks[i++]), done: false };
      return { value: undefined, done: true };
    }),
    cancel: vi.fn(async () => {}),
  };
  return { ok, status, statusText, body: { getReader: () => reader }, reader };
}

/** SSE-encode a JSON event as a single `data: {...}\n\n` frame. */
function frame(/** @type {object} */ obj) {
  return `data: ${JSON.stringify(obj)}\n\n`;
}

/**
 * Collecting callbacks. `handlers` is the ChatCallbacks object passed to the
 * transport (only functions, so it satisfies the index signature); `events`
 * records every event `onEvent` sees; `onClose`/`onError` are the spies.
 */
function collector() {
  /** @type {any[]} */
  const events = [];
  const onClose = vi.fn();
  const onError = vi.fn();
  /** @type {import('../../../src/osprey/interfaces/web_terminal/static/js/chat-client.js').ChatCallbacks} */
  const handlers = {
    onEvent: (/** @type {any} */ e) => events.push(e),
    onClose,
    onError,
  };
  return { events, handlers, onClose, onError };
}

describe('sendPrompt: request shape', () => {
  test('POSTs prompt + chat_id as JSON to /api/chat?stream=true with the abort signal', async () => {
    const fetchMock = vi.fn(async () => streamResponse([frame({ type: 'result', is_error: false })]));
    vi.stubGlobal('fetch', fetchMock);

    const cb = collector();
    const handle = chat.sendPrompt('chat-7', 'hello there', cb.handlers);
    await vi.waitFor(() => expect(cb.onClose).toHaveBeenCalledTimes(1));

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = /** @type {[string, any]} */ (
      /** @type {unknown} */ (fetchMock.mock.calls[0])
    );
    expect(url).toBe('/api/chat?stream=true');
    expect(init.method).toBe('POST');
    expect(init.headers).toEqual({ 'Content-Type': 'application/json' });
    expect(JSON.parse(init.body)).toEqual({ prompt: 'hello there', chat_id: 'chat-7' });
    expect(init.signal).toBeInstanceOf(AbortSignal);
    expect(handle.aborted).toBe(false);
  });
});

describe('sendPrompt: frame parsing and dispatch', () => {
  test('dispatches a single frame to onEvent and its per-type handler', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        streamResponse([frame({ type: 'text', content: 'hi' }), frame({ type: 'result' })])
      )
    );

    const cb = collector();
    const onText = vi.fn();
    chat.sendPrompt('c', 'p', { ...cb.handlers, text: onText });
    await vi.waitFor(() => expect(cb.onClose).toHaveBeenCalledTimes(1));

    expect(cb.events.map((e) => e.type)).toEqual(['text', 'result']);
    expect(onText).toHaveBeenCalledTimes(1);
    expect(onText).toHaveBeenCalledWith({ type: 'text', content: 'hi' });
  });

  test('buffers a frame split across two reads and delivers it once', async () => {
    const whole = frame({ type: 'text', content: 'split-across-reads' });
    const cut = Math.floor(whole.length / 2);
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        streamResponse([whole.slice(0, cut), whole.slice(cut), frame({ type: 'result' })])
      )
    );

    const cb = collector();
    chat.sendPrompt('c', 'p', cb.handlers);
    await vi.waitFor(() => expect(cb.onClose).toHaveBeenCalledTimes(1));

    const texts = cb.events.filter((e) => e.type === 'text');
    expect(texts).toEqual([{ type: 'text', content: 'split-across-reads' }]);
  });

  test('splits multiple frames delivered in a single chunk', async () => {
    const combined =
      frame({ type: 'session_reset' }) +
      frame({ type: 'thinking' }) +
      frame({ type: 'text', content: 'a' }) +
      frame({ type: 'result' });
    vi.stubGlobal('fetch', vi.fn(async () => streamResponse([combined])));

    const cb = collector();
    chat.sendPrompt('c', 'p', cb.handlers);
    await vi.waitFor(() => expect(cb.onClose).toHaveBeenCalledTimes(1));

    expect(cb.events.map((e) => e.type)).toEqual(['session_reset', 'thinking', 'text', 'result']);
  });

  test('ignores `: heartbeat` comment lines', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        streamResponse([': heartbeat\n\n', frame({ type: 'text', content: 'x' }), ': heartbeat\n\n', frame({ type: 'result' })])
      )
    );

    const cb = collector();
    chat.sendPrompt('c', 'p', cb.handlers);
    await vi.waitFor(() => expect(cb.onClose).toHaveBeenCalledTimes(1));

    expect(cb.events.map((e) => e.type)).toEqual(['text', 'result']);
  });

  test('joins multi-line data fields before parsing', async () => {
    const raw = 'data: {"type":"text",\ndata: "content":"multi"}\n\n' + frame({ type: 'result' });
    vi.stubGlobal('fetch', vi.fn(async () => streamResponse([raw])));

    const cb = collector();
    chat.sendPrompt('c', 'p', cb.handlers);
    await vi.waitFor(() => expect(cb.onClose).toHaveBeenCalledTimes(1));

    expect(cb.events[0]).toEqual({ type: 'text', content: 'multi' });
  });
});

describe('sendPrompt: terminal conditions', () => {
  test('a result event ends the turn and closes', async () => {
    const response = streamResponse([
      frame({ type: 'text', content: 'a' }),
      frame({ type: 'result', is_error: false }),
      frame({ type: 'text', content: 'after-terminal' }),
    ]);
    vi.stubGlobal('fetch', vi.fn(async () => response));

    const cb = collector();
    chat.sendPrompt('c', 'p', cb.handlers);
    await vi.waitFor(() => expect(cb.onClose).toHaveBeenCalledTimes(1));

    // Nothing past the terminal `result` is dispatched.
    expect(cb.events.map((e) => e.type)).toEqual(['text', 'result']);
    expect(cb.onError).not.toHaveBeenCalled();
    expect(response.reader.cancel).toHaveBeenCalled();
  });

  test('a non-AssistantMessageError error ends the turn', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        streamResponse([
          frame({ type: 'error', error_type: 'TimeoutError', message: 'boom' }),
          frame({ type: 'text', content: 'never' }),
        ])
      )
    );

    const cb = collector();
    const onError = vi.fn();
    chat.sendPrompt('c', 'p', { ...cb.handlers, error: onError });
    await vi.waitFor(() => expect(cb.onClose).toHaveBeenCalledTimes(1));

    // The server error is delivered by type, then the turn stops.
    expect(cb.events.map((e) => e.type)).toEqual(['error']);
    expect(onError).toHaveBeenCalledWith({ type: 'error', error_type: 'TimeoutError', message: 'boom' });
    // Transport onError is NOT fired for a server error event.
    expect(cb.onError).not.toHaveBeenCalled();
  });

  test('an AssistantMessageError is surfaced but does not end the turn', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        streamResponse([
          frame({ type: 'error', error_type: 'AssistantMessageError', message: 'partial' }),
          frame({ type: 'text', content: 'continues' }),
          frame({ type: 'result' }),
        ])
      )
    );

    const cb = collector();
    chat.sendPrompt('c', 'p', cb.handlers);
    await vi.waitFor(() => expect(cb.onClose).toHaveBeenCalledTimes(1));

    expect(cb.events.map((e) => e.type)).toEqual(['error', 'text', 'result']);
  });
});

describe('sendPrompt: transport failures', () => {
  test('a non-2xx status reports onError and closes', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => streamResponse([], { ok: false, status: 503, statusText: 'Service Unavailable' }))
    );

    const cb = collector();
    chat.sendPrompt('c', 'p', cb.handlers);
    await vi.waitFor(() => expect(cb.onClose).toHaveBeenCalledTimes(1));

    expect(cb.onError).toHaveBeenCalledTimes(1);
    expect(cb.onError.mock.calls[0][0]).toBeInstanceOf(Error);
    expect(cb.onError.mock.calls[0][0].message).toBe('HTTP 503: Service Unavailable');
    // A body-less rejection still carries its status; the slug is simply empty.
    expect(cb.onError.mock.calls[0][0].status).toBe(503);
    expect(cb.onError.mock.calls[0][0].slug).toBe('');
  });

  test('a rejection body carries its detail.error slug to the caller', async () => {
    // Two 409s from this endpoint mean opposite things to the operator, and
    // only the slug separates them — the controller keys its copy on it.
    const body = {
      detail: { error: 'chat_terminated', message: 'send the prompt again' },
    };
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ...streamResponse([], { ok: false, status: 409, statusText: 'Conflict' }),
        json: async () => body,
      }))
    );

    const cb = collector();
    chat.sendPrompt('c', 'p', cb.handlers);
    await vi.waitFor(() => expect(cb.onClose).toHaveBeenCalledTimes(1));

    const err = cb.onError.mock.calls[0][0];
    expect(err.status).toBe(409);
    expect(err.slug).toBe('chat_terminated');
    expect(err.message).toBe('HTTP 409: Conflict');
  });

  test('a non-JSON rejection body is not itself a failure', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ...streamResponse([], { ok: false, status: 502, statusText: 'Bad Gateway' }),
        json: async () => {
          throw new SyntaxError('Unexpected token <');
        },
      }))
    );

    const cb = collector();
    chat.sendPrompt('c', 'p', cb.handlers);
    await vi.waitFor(() => expect(cb.onClose).toHaveBeenCalledTimes(1));

    expect(cb.onError).toHaveBeenCalledTimes(1);
    expect(cb.onError.mock.calls[0][0].status).toBe(502);
    expect(cb.onError.mock.calls[0][0].slug).toBe('');
  });

  test('a 204 closes the turn without reporting an error', async () => {
    // The server abandoned the request; there is no body to stream, and a
    // missing body is not the transport failure it would be on a 200.
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({ ok: true, status: 204, statusText: 'No Content', body: null }))
    );

    const cb = collector();
    chat.sendPrompt('c', 'p', cb.handlers);
    await vi.waitFor(() => expect(cb.onClose).toHaveBeenCalledTimes(1));

    expect(cb.onError).not.toHaveBeenCalled();
    expect(cb.events).toEqual([]);
  });

  test('a rejected fetch reports onError and closes', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => { throw new Error('network down'); }));

    const cb = collector();
    chat.sendPrompt('c', 'p', cb.handlers);
    await vi.waitFor(() => expect(cb.onClose).toHaveBeenCalledTimes(1));

    expect(cb.onError).toHaveBeenCalledTimes(1);
    expect(cb.onError.mock.calls[0][0].message).toBe('network down');
  });

  test('a malformed frame reports onError', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => streamResponse(['data: {not json}\n\n'])));

    const cb = collector();
    chat.sendPrompt('c', 'p', cb.handlers);
    await vi.waitFor(() => expect(cb.onClose).toHaveBeenCalledTimes(1));

    expect(cb.onError).toHaveBeenCalledTimes(1);
    expect(cb.onError.mock.calls[0][0]).toBeInstanceOf(Error);
  });

  test('onClose fires exactly once even on a terminal-then-end stream', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => streamResponse([frame({ type: 'result' })])));

    const cb = collector();
    chat.sendPrompt('c', 'p', cb.handlers);
    await vi.waitFor(() => expect(cb.onClose).toHaveBeenCalledTimes(1));
    // Give any stray microtasks a chance to double-fire.
    await new Promise((r) => setTimeout(r, 0));
    expect(cb.onClose).toHaveBeenCalledTimes(1);
  });
});

describe('sendPrompt: abort', () => {
  test('aborting is silent — onClose fires, onError does not, aborted flips true', async () => {
    // A reader whose read() blocks until the fetch signal aborts, then rejects
    // with an AbortError, mirroring a real fetch body under AbortController.
    const fetchMock = vi.fn(async (/** @type {string} */ _url, /** @type {any} */ init) => {
      /** @type {AbortSignal} */
      const signal = init.signal;
      const reader = {
        read: () =>
          new Promise((_resolve, reject) => {
            const fail = () => {
              const err = new Error('aborted');
              err.name = 'AbortError';
              reject(err);
            };
            if (signal.aborted) fail();
            else signal.addEventListener('abort', fail);
          }),
        cancel: vi.fn(async () => {}),
      };
      return { ok: true, status: 200, statusText: 'OK', body: { getReader: () => reader } };
    });
    vi.stubGlobal('fetch', fetchMock);

    const cb = collector();
    const handle = chat.sendPrompt('c', 'p', cb.handlers);
    expect(handle.aborted).toBe(false);
    handle.abort();

    await vi.waitFor(() => expect(cb.onClose).toHaveBeenCalledTimes(1));
    expect(handle.aborted).toBe(true);
    expect(cb.onError).not.toHaveBeenCalled();
  });
});

describe('interrupt', () => {
  test('POSTs to /api/chat/{id}/interrupt with the id encoded', async () => {
    const fetchMock = vi.fn(async () => ({ ok: true, status: 200, statusText: 'OK' }));
    vi.stubGlobal('fetch', fetchMock);

    await chat.interrupt('chat/1');
    expect(fetchMock).toHaveBeenCalledWith('/api/chat/chat%2F1/interrupt', { method: 'POST' });
  });

  test('throws on a non-2xx status', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ({ ok: false, status: 404, statusText: 'Not Found' })));
    await expect(chat.interrupt('c')).rejects.toThrow('HTTP 404: Not Found');
  });
});

/**
 * Build a fake non-streaming `Response` carrying *body* as its JSON.
 * @param {any} body
 * @param {{ ok?: boolean, status?: number, statusText?: string }} [opts]
 */
function jsonResponse(body, opts = {}) {
  const { ok = true, status = 200, statusText = 'OK' } = opts;
  return { ok, status, statusText, json: async () => body };
}

/** The URL of the first call made to *fetchMock*. */
function firstUrl(/** @type {any} */ fetchMock) {
  return fetchMock.mock.calls[0][0];
}

/** The `init` of the first call made to *fetchMock*. */
function firstInit(/** @type {any} */ fetchMock) {
  return fetchMock.mock.calls[0][1];
}

describe('requestHandoff: request shape', () => {
  test('POSTs to /api/session/{key}/handoff and resolves the state envelope', async () => {
    const fetchMock = vi.fn(async () => jsonResponse({ state: 'ready', session_id: 'k-1' }));
    vi.stubGlobal('fetch', fetchMock);

    const result = await chat.requestHandoff('k-1');

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(firstUrl(fetchMock)).toBe('/api/session/k-1/handoff');
    const init = firstInit(fetchMock);
    expect(init.method).toBe('POST');
    expect(init.headers).toEqual({ 'Content-Type': 'application/json' });
    expect(JSON.parse(init.body)).toEqual({ to: 'simple', interrupt: false });
    expect(result).toEqual({ state: 'ready', session_id: 'k-1' });
  });

  test('forwards interrupt: true when the operator cuts a running turn', async () => {
    const fetchMock = vi.fn(async () => jsonResponse({ state: 'ready', session_id: 'k' }));
    vi.stubGlobal('fetch', fetchMock);

    await chat.requestHandoff('k', { interrupt: true });
    expect(JSON.parse(firstInit(fetchMock).body)).toEqual({ to: 'simple', interrupt: true });
  });

  test('applies the multi-user prefix and encodes the key', async () => {
    window.__OSPREY_PREFIX__ = '/u/alice';
    const fetchMock = vi.fn(async () => jsonResponse({ state: 'ready', session_id: 'a/b' }));
    vi.stubGlobal('fetch', fetchMock);

    await chat.requestHandoff('a/b');
    expect(firstUrl(fetchMock)).toBe('/u/alice/api/session/a%2Fb/handoff');
  });

  test('resolves null on a 204 — the request was abandoned server-side', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: true,
        status: 204,
        statusText: 'No Content',
        json: async () => {
          throw new SyntaxError('Unexpected end of JSON input');
        },
      }))
    );

    await expect(chat.requestHandoff('k')).resolves.toBeNull();
  });

  test('adds no timeout of its own — no signal unless the caller passes one', async () => {
    const fetchMock = vi.fn(async () => jsonResponse({ state: 'ready', session_id: 'k' }));
    vi.stubGlobal('fetch', fetchMock);

    await chat.requestHandoff('k');
    expect(firstInit(fetchMock).signal).toBeUndefined();
  });
});

describe('requestHandoff: abort', () => {
  test("passes the caller's signal through to fetch", async () => {
    const fetchMock = vi.fn(async () => jsonResponse({ state: 'ready', session_id: 'k' }));
    vi.stubGlobal('fetch', fetchMock);
    const controller = new AbortController();

    await chat.requestHandoff('k', { signal: controller.signal });
    expect(firstInit(fetchMock).signal).toBe(controller.signal);
  });

  test('aborting the signal rejects the pending hand-off', async () => {
    // A fetch that never settles on its own, mirroring a hand-off waiting on a
    // long Expert turn: only the signal ends it.
    vi.stubGlobal(
      'fetch',
      vi.fn((/** @type {string} */ _url, /** @type {any} */ init) =>
        new Promise((_resolve, reject) => {
          /** @type {AbortSignal} */
          const signal = init.signal;
          const fail = () => {
            const err = new Error('aborted');
            err.name = 'AbortError';
            reject(err);
          };
          if (signal.aborted) fail();
          else signal.addEventListener('abort', fail);
        })
      )
    );
    const controller = new AbortController();

    const pending = chat.requestHandoff('k', { signal: controller.signal });
    controller.abort();
    await expect(pending).rejects.toThrow('aborted');
  });
});

describe('requestHandoff: rejection slugs', () => {
  test.each([
    [409, 'Conflict', 'session_attached_elsewhere'],
    [429, 'Too Many Requests', 'chat_capacity'],
    [503, 'Service Unavailable', 'shutting_down'],
  ])('%i carries status and the detail.error slug', async (status, statusText, slug) => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => jsonResponse({ detail: { error: slug } }, { ok: false, status, statusText }))
    );

    const err = await chat.requestHandoff('k').catch((e) => e);
    expect(err).toBeInstanceOf(Error);
    expect(err.status).toBe(status);
    expect(err.slug).toBe(slug);
    expect(err.message).toBe(`HTTP ${status}: ${statusText}`);
  });

  test('a rejection without a JSON body leaves the slug empty', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: false,
        status: 503,
        statusText: 'Service Unavailable',
        json: async () => {
          throw new Error('not JSON');
        },
      }))
    );

    const err = await chat.requestHandoff('k').catch((e) => e);
    expect(err.status).toBe(503);
    expect(err.slug).toBe('');
  });
});

describe('fetchHistory', () => {
  test('GETs the full transcript for the key and returns its turns', async () => {
    const turns = [
      { role: 'user', content: 'hi', timestamp: '2026-09-05T00:00:00Z' },
      { role: 'assistant', content: 'hello' },
    ];
    const fetchMock = vi.fn(async () => jsonResponse({ turns, count: turns.length }));
    vi.stubGlobal('fetch', fetchMock);

    await expect(chat.fetchHistory('k-1')).resolves.toEqual(turns);
    expect(fetchMock).toHaveBeenCalledWith('/api/session-chat?session_id=k-1&full=1', {
      cache: 'no-store',
    });
  });

  test('applies the multi-user prefix and encodes the key', async () => {
    window.__OSPREY_PREFIX__ = '/u/alice';
    const fetchMock = vi.fn(async () => jsonResponse({ turns: [], count: 0 }));
    vi.stubGlobal('fetch', fetchMock);

    await chat.fetchHistory('a/b');
    expect(firstUrl(fetchMock)).toBe('/u/alice/api/session-chat?session_id=a%2Fb&full=1');
  });

  test('a body without turns yields an empty list', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => jsonResponse({ count: 0 })));
    await expect(chat.fetchHistory('k')).resolves.toEqual([]);
  });

  test('a non-2xx status rejects with the status and slug', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        jsonResponse({ detail: { error: 'session_unknown' } }, {
          ok: false,
          status: 404,
          statusText: 'Not Found',
        })
      )
    );

    const err = await chat.fetchHistory('k').catch((e) => e);
    expect(err.status).toBe(404);
    expect(err.slug).toBe('session_unknown');
  });
});
