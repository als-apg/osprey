// @ts-check
/* OSPREY Web Terminal — Chat SSE Transport
 *
 * Browser-side transport for the chat endpoint, the session hand-off request,
 * and the transcript history fetch. Pure networking: it POSTs a prompt, parses
 * the streaming Server-Sent-Events response, and forwards each decoded event
 * to caller-supplied callbacks. It touches no DOM and holds no rendering
 * state, so the whole surface is unit-testable with a mocked `fetch`.
 *
 * SSE wire format (see routes/chat.py `_sse`): each event is one frame
 * `data: {json}\n\n`; heartbeat comments arrive as `: heartbeat` lines. A frame
 * can be split across network reads, so chunks are buffered until a `\n\n`
 * boundary completes a frame.
 */

import { fetchJSON, withPrefix } from './api.js';
import { endsTurn } from './chat-frames.js';

/** Base path for the chat REST + streaming endpoint (prefixed per-call via
 * `withPrefix` so multi-user `/u/<user>/` deployments reach this container). */
const CHAT_ENDPOINT = '/api/chat';

/** Base path for the per-session routes; `/{key}/handoff` hangs off it. */
const SESSION_ENDPOINT = '/api/session';

/** Read-only transcript endpoint backing the chat's replay. */
const SESSION_CHAT_ENDPOINT = '/api/session-chat';

/** The project's slash commands, backing the chat's suggestions. */
const COMMANDS_ENDPOINT = '/api/chat/commands';

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
 * frame. Returns once the stream ends or — unless *untilClosed* — a terminal
 * event is seen. Partial chunks are buffered across reads so a frame split
 * mid-transfer is delivered exactly once and whole.
 * @param {ReadableStreamDefaultReader<Uint8Array>} reader
 * @param {ChatCallbacks} callbacks
 * @param {boolean} [untilClosed] - keep reading past terminal events (a live feed)
 * @returns {Promise<void>}
 */
async function pump(reader, callbacks, untilClosed = false) {
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
      if (!untilClosed && endsTurn(event)) return;
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
 * A transport failure carrying the endpoint's machine-readable reason.
 *
 * `status` is the HTTP status; `slug` is the `detail.error` slug the chat
 * routes put in every rejection body (`chat_terminated`, `turn_in_progress`,
 * `chat_capacity`, …) or `''` when the body carried none. Callers key their
 * user-facing copy on the slug: two rejections share status 409 and mean
 * opposite things to the operator — one says a turn is already running, the
 * other says this chat was restarted and the prompt needs sending again.
 *
 * `commands` is the list a `handoff_started_commands` refusal carries: the
 * commands the outgoing agent started that ending it would also end. It is
 * present only when the body carried one. `unread` is that refusal's word
 * that the server could not read what the agent started; it is present only
 * when the body carried it. `command` is the slash command name a
 * `command_not_available` refusal names; it is present only when the body
 * carried one.
 * @typedef {Error & { status: number, slug: string,
 *   commands?: import('./started-commands.js').StartedCommandJson[],
 *   unread?: boolean, command?: string }} TransportError
 */

/**
 * Build the {@link TransportError} for a non-2xx response, reading the slug out
 * of the JSON body. A body that is absent, empty, or not JSON is not a failure
 * of its own — the status still describes the rejection — so the slug is left
 * empty and the caller falls back to it.
 * @param {Response} res
 * @returns {Promise<TransportError>}
 */
async function transportError(res) {
  let slug = '';
  /** @type {import('./started-commands.js').StartedCommandJson[] | undefined} */
  let commands;
  /** @type {boolean | undefined} */
  let unread;
  /** @type {string | undefined} */
  let command;
  try {
    const detail = (await res.json())?.detail;
    if (detail && typeof detail.error === 'string') slug = detail.error;
    if (detail && Array.isArray(detail.commands)) commands = detail.commands;
    if (detail && typeof detail.unread === 'boolean') unread = detail.unread;
    if (detail && typeof detail.command === 'string') command = detail.command;
  } catch {
    /* no body, or not the JSON error shape — the status carries the failure */
  }
  const err = /** @type {TransportError} */ (new Error(`HTTP ${res.status}: ${res.statusText}`));
  err.status = res.status;
  err.slug = slug;
  if (commands) err.commands = commands;
  if (unread !== undefined) err.unread = unread;
  if (command !== undefined) err.command = command;
  return err;
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
 * exactly one `onClose()`; a user abort is silent (no `onError`). A 204 means
 * the request was abandoned server-side and carries no events, so the turn ends
 * the same silent way.
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
      if (!res.ok) throw await transportError(res);
      // 204: the server abandoned the request because this client's channel
      // closed mid-turn. There is no body to stream, so end quietly.
      if (res.status === 204) return;
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
  if (!res.ok) throw await transportError(res);
  return res;
}

/**
 * Answer the `permission_request` a turn is waiting on. The body carries the
 * request id, the decision and — for a question — the chosen answers keyed by
 * question text; `for_session` also allows the tool for the rest of the
 * session where the request offered that. A 404 `no_pending_request` means
 * the request is already gone: the turn was stopped, or another tab answered
 * it first.
 * @param {string} chatId
 * @param {{ request_id: string, allow: boolean, answers?: Record<string, string>, message?: string, for_session?: boolean }} body
 * @returns {Promise<Response>}
 */
export async function answerPermission(chatId, body) {
  const res = await fetch(withPrefix(`${CHAT_ENDPOINT}/${encodeURIComponent(chatId)}/permission`), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw await transportError(res);
  return res;
}

/**
 * A live feed held open by {@link followChat}.
 * @typedef {object} ChatFeed
 * @property {() => void} close - stop following for good
 */

/**
 * Follow everything the agent behind *chatId* does, whichever tab asked, until
 * closed: `GET /api/chat/{chat_id}/events`. Each (re)connect opens with a
 * `hello` snapshot. A dropped connection is retried with backoff; a 204 (no
 * agent runs for the chat, yet or any longer) more slowly.
 * @param {string} chatId
 * @param {ChatCallbacks} callbacks - `onEvent` gets every frame; `onNoAgent`
 *   hears each 204, the answer that no agent runs for the chat
 * @returns {ChatFeed}
 */
export function followChat(chatId, callbacks) {
  let closed = false;
  let generation = 0;
  let delay = 1000;
  let timer = /** @type {ReturnType<typeof setTimeout> | null} */ (null);
  let controller = /** @type {AbortController | null} */ (null);

  /** @param {number} ms */
  function retry(ms) {
    if (closed) return;
    clearTimeout(timer ?? undefined);
    timer = setTimeout(() => void connect(), ms);
  }

  async function connect() {
    if (closed) return;
    const mine = ++generation;
    // A reconnect supersedes any retry still waiting to fire.
    clearTimeout(timer ?? undefined);
    timer = null;
    controller?.abort();
    controller = new AbortController();
    let next = delay;
    try {
      const res = await fetch(withPrefix(`${CHAT_ENDPOINT}/${encodeURIComponent(chatId)}/events`), {
        signal: controller.signal,
        cache: 'no-store',
      });
      if (res.status === 204) {
        next = 10000;
        if (mine === generation && typeof callbacks.onNoAgent === 'function') {
          callbacks.onNoAgent();
        }
      } else if (res.ok && res.body) {
        delay = 1000;
        await pump(res.body.getReader(), callbacks, true);
        next = delay;
      }
    } catch {
      /* dropped or aborted — retried below unless superseded or closed */
    }
    if (mine !== generation) return;
    delay = Math.min(delay * 2, 10000);
    retry(next);
  }

  void connect();
  return {
    close() {
      closed = true;
      generation += 1;
      clearTimeout(timer ?? undefined);
      controller?.abort();
    },
  };
}

/**
 * One conversation turn as served by `GET /api/session-chat`. The same shape
 * the renderer takes, so turns pass straight from here to the replay.
 *
 * @typedef {object} ChatTurn
 * @property {string} role - `user` or `assistant`
 * @property {string} content - the turn's text; markdown for an agent turn
 * @property {string} [timestamp] - transcript timestamp
 * @property {ChatEvent[]} [events] - an agent turn's stream as the live chat
 *   received it (`text`, `thinking`, `tool_use`, `tool_result`, `result`)
 */

/**
 * Ask the server to hand session *key* to the chat.
 *
 * POSTs to `/api/session/{key}/handoff`. The server tears the outgoing surface
 * down, waits for it to be observed dead, and answers `{state, session_id}`
 * once the chat may resume the key's transcript. `interrupt` says the
 * operator chose to cut a turn that was still running rather than wait for it.
 * `endStarted` says the operator agreed that ending the outgoing agent also
 * ends the commands it started; without it, a hand-off that would end them is
 * refused `handoff_started_commands` with the list in `err.commands`, or with
 * `err.unread` set when the server could not read what the agent started.
 *
 * The request is long-lived by design — a busy terminal turn can take minutes to
 * reach idle — so this adds no timeout of its own; *signal* is the only way to
 * end it. A rejection carries the endpoint's reason the same way `sendPrompt`'s
 * does: `err.status` plus `err.slug` from the body's `detail.error` (409 the
 * key is held elsewhere, 429 `chat_capacity`, 503 the server is shutting down),
 * so the caller branches on the slug and falls back to the status when a body
 * carried none.
 *
 * Resolves `null` on a 204: the request was abandoned server-side because this
 * client's channel closed, so no surface was handed over and there is nothing
 * to resume.
 * @param {string} key
 * @param {{ interrupt?: boolean, endStarted?: boolean, signal?: AbortSignal }} [options]
 * @returns {Promise<{ state: string, session_id: string } | null>}
 */
export async function requestHandoff(key, options = {}) {
  const { interrupt: cutRunningTurn = false, endStarted = false, signal } = options;
  const res = await fetch(withPrefix(`${SESSION_ENDPOINT}/${encodeURIComponent(key)}/handoff`), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    // `to` is fixed: this module is the chat's transport, and the terminal
    // acquires its surface over `/ws/terminal` instead.
    body: JSON.stringify({ to: 'chat', interrupt: cutRunningTurn, end_started: endStarted }),
    signal,
  });
  if (!res.ok) throw await transportError(res);
  if (res.status === 204) return null;
  return res.json();
}

/**
 * Read the conversation behind *key* so the chat can replay it.
 *
 * `full=1` asks for the whole transcript rather than the trailing window the
 * session page shows, and `events=1` for each agent turn's thinking and tool
 * calls alongside its text. The response is `{turns, count}`; only the turns are
 * returned, and a body without them yields an empty list — a key whose
 * transcript has no turns yet is a normal first hand-off, not a failure.
 * `no-store` keeps a hand-off from replaying a copy cached before the last turn.
 * @param {string} key
 * @returns {Promise<ChatTurn[]>}
 */
export async function fetchHistory(key) {
  const url = `${SESSION_CHAT_ENDPOINT}?session_id=${encodeURIComponent(key)}&full=1&events=1`;
  const res = await fetch(withPrefix(url), { cache: 'no-store' });
  if (!res.ok) throw await transportError(res);
  const data = await res.json();
  return Array.isArray(data?.turns) ? data.turns : [];
}

/**
 * One slash command the chat agent accepts. `kind` is `skill`, `command` or
 * `builtin`; `agent_invocable` says whether the agent may also run it on its own.
 * @typedef {{ name: string, description: string, argument_hint: string, kind: string, agent_invocable?: boolean }} SlashCommand
 */

/**
 * Read the slash commands the chat agent accepts in this project.
 *
 * A body without a `commands` array yields an empty list; a transport failure
 * or a non-2xx status rejects.
 * @returns {Promise<SlashCommand[]>}
 */
export async function fetchCommands() {
  const body = await fetchJSON(COMMANDS_ENDPOINT);
  return Array.isArray(body?.commands) ? body.commands : [];
}

/**
 * A file the server stored for an attachment: the name it is shown under, the
 * absolute path the agent reads, the size in bytes and the MIME type the
 * content was found to be.
 * @typedef {{ name: string, path: string, size: number, type: string }} StoredAttachment
 */

/**
 * Store one file the operator attached to a message that has not been sent yet.
 *
 * POSTs the file's bytes to `/api/chat/{chatId}/attachments`. The name travels
 * percent-encoded in a header, and *batch* names the message the file belongs
 * to, so the server can hold that message to its file cap. A refusal rejects
 * with a {@link TransportError} whose `slug` names the reason (`unsupported_type`,
 * `file_too_large`, `chat_full`, `too_many_files`, `empty_file`, ...). *signal*
 * cancels the upload.
 * @param {string} chatId
 * @param {Blob} file
 * @param {{ name: string, batch: string, signal?: AbortSignal }} options
 * @returns {Promise<StoredAttachment>}
 */
export async function uploadChatAttachment(chatId, file, { name, batch, signal }) {
  const res = await fetch(
    withPrefix(`${CHAT_ENDPOINT}/${encodeURIComponent(chatId)}/attachments`),
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/octet-stream',
        'X-Osprey-Attachment-Name': encodeURIComponent(name),
        'X-Osprey-Attachment-Batch': batch,
      },
      body: file,
      signal,
    }
  );
  if (!res.ok) throw await transportError(res);
  return res.json();
}

/**
 * Remove a file the operator took back before sending it. Removing a file that
 * is no longer there is not an error: the server answers the same either way.
 * With *keepalive* the request outlives the page, so a tab that closes with
 * files in its tray still takes them back.
 * @param {string} chatId
 * @param {string} storedName - the file's name in the store (the last segment of its path)
 * @param {{ batch?: string, keepalive?: boolean }} [options] - the message the file was filed under
 * @returns {Promise<void>}
 */
export async function removeChatAttachment(chatId, storedName, { batch, keepalive } = {}) {
  const res = await fetch(
    withPrefix(
      `${CHAT_ENDPOINT}/${encodeURIComponent(chatId)}/attachments/${encodeURIComponent(storedName)}`
    ),
    {
      method: 'DELETE',
      headers: batch ? { 'X-Osprey-Attachment-Batch': batch } : {},
      keepalive: Boolean(keepalive),
    }
  );
  if (!res.ok) throw await transportError(res);
}
