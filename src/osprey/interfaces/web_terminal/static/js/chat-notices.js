// @ts-check
/**
 * OSPREY Web Terminal — Simple-mode chat: the copy for refusals and failures.
 *
 * What the operator reads when the chat endpoint or the hand-off refuses a
 * request, or the transport fails. Each lookup is keyed on the endpoint's
 * machine-readable `detail.error` slug first, and on the HTTP status only when
 * no slug came back. Pure data and lookups: nothing here touches the DOM.
 *
 * @module chat-notices
 */

import {
  HANDOFF_HELD_ELSEWHERE_MESSAGE,
  HANDOFF_OUTGOING_RUNNING_MESSAGE,
} from './terminal-handoff.js';

/**
 * User-facing copy keyed on the endpoint's machine-readable `detail.error`
 * slug — the primary table, because the status alone cannot separate two
 * rejections that share one. `POST /api/chat` answers 409 both when a turn is
 * already streaming and when the chat was torn down mid-start by a posture
 * flip; only the second one has a remedy, and it is the operator resending the
 * prompt they just sent.
 * @type {Record<string, string>}
 */
const TRANSPORT_NOTICES_BY_SLUG = {
  chat_terminated: 'The session restarted with the new posture. Send your message again.',
  turn_in_progress: 'A turn is already running.',
  chat_capacity: 'The server is busy. Try again in a moment.',
};

/**
 * Status-keyed fallback for a rejection that carries no slug — a proxy's own
 * 503, say, or any error shape older than the slug contract. These are
 * distinct from server `error` events (which the renderer paints as red error
 * blocks); a transport failure renders as a centred system notice instead.
 * @type {Record<number, string>}
 */
const TRANSPORT_NOTICES = {
  409: 'A turn is already running.',
  429: 'The server is busy. Try again in a moment.',
  503: 'The agent is unavailable.',
};

/** Fallback copy for a transport failure with no recognised slug or status. */
const TRANSPORT_FALLBACK = 'The connection to the agent failed.';

/** The commands that sign the CLI in or out; the notice for them says who does that. */
const SIGN_IN_COMMANDS = new Set(['login', 'logout']);

/**
 * The notice for a slash command the chat does not offer (`command_not_available`).
 * The server names the command in `detail.command`; the notice repeats it as
 * plain text, and says less when the name did not come back.
 * @param {unknown} command
 * @returns {string}
 */
function commandNotice(command) {
  if (typeof command !== 'string' || !command) {
    return 'This command is not available in this chat.';
  }
  const notice = `/${command} is not available in this chat.`;
  return SIGN_IN_COMMANDS.has(command) ? `${notice} Your administrator sets sign-in.` : notice;
}

/**
 * What the operator may do about a refused hand-off, keyed on the endpoint's
 * `detail.error` slug. A refusal is not a transport failure: it is the other
 * view still holding the session, so it belongs on the overlay that already
 * has the operator's attention rather than as a line in a log they cannot use.
 *
 * `interrupt` re-sends the same request with the operator's consent to cut a
 * running turn; `retry` re-sends it unchanged; a null action is a refusal
 * retrying does not change, and states the fact alone.
 * @type {Record<string, { message: string, action: 'interrupt'|'retry'|null }>}
 */
const HANDOFF_REFUSALS = {
  session_attached_elsewhere: {
    message: HANDOFF_HELD_ELSEWHERE_MESSAGE,
    action: null,
  },
  handoff_needs_interrupt: {
    message: 'The other view may still be working.',
    action: 'interrupt',
  },
  handoff_started_commands: {
    message: 'The agent started a command that is still running.',
    action: 'interrupt',
  },
  handoff_superseded: { message: 'Another request took over this session.', action: 'retry' },
  outgoing_still_running: {
    message: HANDOFF_OUTGOING_RUNNING_MESSAGE,
    action: 'retry',
  },
  outgoing_vanished: { message: 'The session changed while switching.', action: 'retry' },
  spawn_not_pooled: { message: 'The session did not start.', action: 'retry' },
  chat_capacity: { message: 'No chat capacity right now.', action: 'retry' },
  sdk_unavailable: { message: 'The agent is unavailable.', action: null },
};

/**
 * The `handoff_started_commands` refusal when the server could not read what
 * the agent started: the overlay states that, not a command it never saw.
 * @type {{ message: string, action: 'interrupt' }}
 */
const HANDOFF_STARTED_UNREAD = {
  message: 'Could not check what the agent started.',
  action: 'interrupt',
};

// `chat_terminated` is deliberately absent above. This table is consulted
// before the notice tables, and that slug's one useful answer is the notice
// it already has: the prompt the operator just sent needs sending again. A
// hand-off refused with it falls to HANDOFF_FALLBACK, which offers the same
// retry the overlay would have given it.

/** Overlay copy for a hand-off that failed with no reason this view knows. */
export const HANDOFF_FALLBACK = {
  message: 'The switch failed.',
  action: /** @type {'retry'} */ ('retry'),
};

/**
 * Extract an HTTP status from a transport error: the `status` the transport
 * attaches, else parsed out of an `HTTP 409: ...` message, else 0 when the
 * failure carries no status at all (a network or parse error).
 * @param {unknown} err
 * @returns {number}
 */
function statusFromError(err) {
  const status = /** @type {{ status?: unknown }} */ (err ?? {}).status;
  if (typeof status === 'number' && Number.isFinite(status)) return status;
  const message = err instanceof Error ? err.message : String(err ?? '');
  const match = /^HTTP (\d+)/.exec(message);
  return match ? Number(match[1]) : 0;
}

/**
 * The notice to show for a transport failure: its slug first, its status
 * second, the generic line last.
 *
 * Slug before status is the whole point — keying on the status alone told an
 * operator whose chat had just been restarted by their own posture flip that a
 * turn was already running, which is both false and unactionable.
 * @param {unknown} err
 * @returns {string}
 */
export function transportNotice(err) {
  const { slug, command } = /** @type {{ slug?: unknown, command?: unknown }} */ (err ?? {});
  if (slug === 'command_not_available') return commandNotice(command);
  if (typeof slug === 'string' && slug in TRANSPORT_NOTICES_BY_SLUG) {
    return TRANSPORT_NOTICES_BY_SLUG[slug];
  }
  return TRANSPORT_NOTICES[statusFromError(err)] ?? TRANSPORT_FALLBACK;
}

/**
 * The hand-off refusal *err* describes, or null when it is an ordinary
 * transport failure. Slug-only by design: a bare 409 says nothing about who
 * holds the session, and offering "Stop and switch now" for a rejection that
 * is not about the other view would cut a turn for no reason.
 * @param {unknown} err
 * @returns {{ message: string, action: 'interrupt'|'retry'|null } | null}
 */
export function handoffRefusal(err) {
  const { slug, unread } = /** @type {{ slug?: unknown, unread?: unknown }} */ (err ?? {});
  if (slug === 'handoff_started_commands' && unread === true) return HANDOFF_STARTED_UNREAD;
  if (typeof slug === 'string' && slug in HANDOFF_REFUSALS) return HANDOFF_REFUSALS[slug];
  return null;
}
