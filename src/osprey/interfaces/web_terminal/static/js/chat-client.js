// @ts-check
/* OSPREY Web Terminal — Chat SSE Transport
 *
 * Browser-side transport for the chat endpoint. Pure networking: it POSTs a
 * prompt, parses the streaming Server-Sent-Events response, and forwards each
 * decoded event to caller-supplied callbacks. It touches no DOM and holds no
 * rendering state, so the whole surface is unit-testable with a mocked `fetch`.
 *
 * SSE wire format (see routes/chat.py `_sse`): each event is one frame
 * `data: {json}\n\n`; heartbeat comments arrive as `: heartbeat` lines. A frame
 * can be split across network reads, so chunks are buffered until a `\n\n`
 * boundary completes a frame.
 */

import { withPrefix } from './api.js';

/** Base path for the chat REST + streaming endpoint (prefixed per-call via
 * `withPrefix` so multi-user `/u/<user>/` deployments reach this container). */
const CHAT_ENDPOINT = '/api/chat';

/**
 * A single decoded server event. Every event carries a `type`; the remaining
 * fields depend on it (e.g. `text` carries `content`, `result` carries cost and
 * turn metadata, `error` carries `error_type` + `message`).
 * @typedef {{ type: string, [key: string]: any }} ChatEvent
 */

/**
 * Sink for a streaming turn. Every key is optional. The transport calls:
 *   - `onEvent(event)` once for every decoded event (before per-type dispatch);
 *   - a per-type handler keyed by the event's `type` (e.g. `text`, `thinking`,
 *     `tool_use`, `tool_result`, `result`, `error`, `session_reset`, `system`),
 *     invoked with the same event;
 *   - `onError(error)` for a transport-level failure — a rejected fetch, a
 *     non-2xx status, a missing body, or a malformed frame. This is distinct
 *     from a server `error` event, which is delivered by type like any other;
 *   - `onClose()` exactly once when the turn ends, whether it completed, failed,
 *     or was aborted.
 *
 * A user-initiated abort is silent: `onError` is not called, only `onClose`.
 * @typedef {{ [key: string]: ((arg?: any) => void) | undefined }} ChatCallbacks
 */

/**
 * A handle for cancelling an in-flight streaming turn.
 * @typedef {{ abort: () => void, readonly aborted: boolean }} ChatAbortHandle
 */

/**
 * True when *event* ends the turn: a `result`, or an `error` whose `error_type`
 * is not `AssistantMessageError` (assistant-message errors are surfaced but the
 * turn keeps streaming).
 * @param {ChatEvent} event
 * @returns {boolean}
 */
function isTerminal(event) {
  if (event.type === 'result') return true;
  if (event.type === 'error' && event.error_type !== 'AssistantMessageError') return true;
  return false;
}

/**
 * Parse one raw SSE frame into an event, or `null` when the frame carries no
 * data (a heartbeat comment or blank line). Multiple `data:` lines are joined
 * with newlines per the SSE spec before JSON parsing.
 * @param {string} rawFrame
 * @returns {ChatEvent | null}
 */
function parseSseFrame(rawFrame) {
  /** @type {string[]} */
  const dataLines = [];
  for (const line of rawFrame.split('\n')) {
    const clean = line.endsWith('\r') ? line.slice(0, -1) : line;
    // Blank separators and `:`-prefixed comments (e.g. `: heartbeat`) carry no data.
    if (clean === '' || clean.startsWith(':')) continue;
    if (clean.startsWith('data:')) {
      const value = clean.slice(5);
      dataLines.push(value.startsWith(' ') ? value.slice(1) : value);
    }
    // Other SSE fields (event:, id:, retry:) are unused by this endpoint.
  }
  if (dataLines.length === 0) return null;
  return /** @type {ChatEvent} */ (JSON.parse(dataLines.join('\n')));
}

/**
 * Deliver an event to the generic sink and its per-type handler.
 * @param {ChatCallbacks} callbacks
 * @param {ChatEvent} event
 */
function dispatch(callbacks, event) {
  if (typeof callbacks.onEvent === 'function') callbacks.onEvent(event);
  const handler = callbacks[event.type];
  if (typeof handler === 'function') handler(event);
}

/**
 * Read the response body to completion, decoding and dispatching each SSE
 * frame. Returns once the stream ends or a terminal event is seen. Partial
 * chunks are buffered across reads so a frame split mid-transfer is delivered
 * exactly once and whole.
 * @param {ReadableStreamDefaultReader<Uint8Array>} reader
 * @param {ChatCallbacks} callbacks
 * @returns {Promise<void>}
 */
async function pump(reader, callbacks) {
  const decoder = new TextDecoder();
  let buffer = '';
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let boundary;
    while ((boundary = buffer.indexOf('\n\n')) !== -1) {
      const frame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const event = parseSseFrame(frame);
      if (!event) continue;
      dispatch(callbacks, event);
      if (isTerminal(event)) return;
    }
  }
  // Flush a trailing frame that arrived without its closing blank line.
  const tail = (buffer + decoder.decode(new Uint8Array())).trim();
  if (tail) {
    const event = parseSseFrame(tail);
    if (event) dispatch(callbacks, event);
  }
}

/**
 * Report a transport-level failure to the caller.
 * @param {ChatCallbacks} callbacks
 * @param {unknown} err
 */
function reportError(callbacks, err) {
  if (typeof callbacks.onError === 'function') {
    callbacks.onError(err instanceof Error ? err : new Error(String(err)));
  }
}

/**
 * Send *prompt* to *chatId* and stream the response.
 *
 * POSTs `{prompt, chat_id}` to `/api/chat?stream=true` and parses the SSE body,
 * forwarding events to *callbacks*. Returns synchronously with an abort handle
 * so the caller can cancel before the fetch resolves. The turn always ends with
 * exactly one `onClose()`; a user abort is silent (no `onError`).
 * @param {string} chatId
 * @param {string} prompt
 * @param {ChatCallbacks} [callbacks]
 * @returns {ChatAbortHandle}
 */
export function sendPrompt(chatId, prompt, callbacks = {}) {
  const controller = new AbortController();
  let closed = false;
  const close = () => {
    if (closed) return;
    closed = true;
    if (typeof callbacks.onClose === 'function') callbacks.onClose();
  };

  (async () => {
    /** @type {ReadableStreamDefaultReader<Uint8Array> | undefined} */
    let reader;
    try {
      const res = await fetch(withPrefix(`${CHAT_ENDPOINT}?stream=true`), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, chat_id: chatId }),
        signal: controller.signal,
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      if (!res.body) throw new Error('Chat response has no readable body');
      reader = res.body.getReader();
      await pump(reader, callbacks);
    } catch (err) {
      // An abort is caller-initiated and expected — stay silent for it.
      if (!controller.signal.aborted) reportError(callbacks, err);
    } finally {
      if (reader) {
        try {
          await reader.cancel();
        } catch {
          /* reader already released */
        }
      }
      close();
    }
  })();

  return {
    abort() {
      controller.abort();
    },
    get aborted() {
      return controller.signal.aborted;
    },
  };
}

/**
 * Interrupt the in-flight turn for *chatId* server-side.
 * @param {string} chatId
 * @returns {Promise<Response>}
 */
export async function interrupt(chatId) {
  const res = await fetch(withPrefix(`${CHAT_ENDPOINT}/${encodeURIComponent(chatId)}/interrupt`), {
    method: 'POST',
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
  return res;
}

/**
 * Delete the conversation for *chatId*. Uses `keepalive` so it survives a
 * `pagehide`/`beforeunload` teardown; fire-and-forget by design (the returned
 * promise need not be awaited and is not error-checked).
 * @param {string} chatId
 * @returns {Promise<Response>}
 */
export function deleteChat(chatId) {
  return fetch(withPrefix(`${CHAT_ENDPOINT}/${encodeURIComponent(chatId)}`), {
    method: 'DELETE',
    keepalive: true,
  });
}
