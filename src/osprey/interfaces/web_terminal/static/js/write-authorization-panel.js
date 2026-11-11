// @ts-check
/* OSPREY Web Terminal — programs' write authorizations, in the control-target popover.
 *
 * A program with no person at a prompt (an optimizer run) asks for a bounded
 * write authorization; this login is its owner, and approves it here. The
 * section lists the requests waiting on this login and the authorizations it
 * has approved that are still in force. "Review" opens one request's card —
 * every channel with the range asked for, the limits bound it sits inside and
 * its current value; the steps, expiry, lease, holder, label and the completion
 * the run will perform; and that a write the limits refuse still costs its
 * step — with Approve. A program's
 * self-issued authorization at `allowed` is listed without Revoke: this
 * terminal only reads where it is, and the switch leaving `allowed` ends it. An
 * authorization in force has Revoke; the machine's
 * own off control above revokes every one on that machine.
 *
 * Every answer is the server's (`/api/terminal/write-authorizations`): a
 * refusal is shown in the route's own sentence, never re-derived here. The
 * render functions are pure (state in, element out), so a test drives them
 * without a server; {@link refreshPrograms} and the two gestures are the only
 * requests.
 */

import { targetRequest } from './control-target-chip.js';

/** @typedef {{address: string, min: number, max: number, limit_min?: number|null,
 *   limit_max?: number|null, current?: any, problem?: string|null}} CardChannel */

/**
 * @typedef {object} RequestCard
 * @property {string} id
 * @property {string} holder
 * @property {string} target
 * @property {string} label
 * @property {string} completion
 * @property {number} steps
 * @property {number} lease_s
 * @property {string} expires_at
 * @property {number} [expires_in_s]
 * @property {CardChannel[]} channels
 * @property {boolean} [approvable]
 * @property {{reason: string, phrase?: string, ceiling?: string}|null} [refusal]
 * @property {string} [limits_note]
 */

/**
 * @typedef {object} Listing
 * @property {RequestCard[]} requests
 * @property {Array<{id: string, holder: string, target: string, label: string,
 *   steps: number, used?: number|null, expires_at: string, expires_in_s?: number,
 *   in_force: boolean, self_issued?: boolean, revocable?: boolean}>} authorizations
 * @property {number} [unreadable]
 */

const BASE = '/api/terminal/write-authorizations';

/** Said beside a program's self-issued authorization, which has no Revoke here. */
export const SELF_ISSUED_SENTENCE =
  'Self-issued at allowed: moving the switch away from allowed ends it.';

/** What each completion a run may perform is called on the card. */
export const COMPLETION_WORDS = {
  none: 'nothing',
  apply_best: 'applies the best point found',
  restore: 'restores the values it started from',
};

/** @type {Listing|null} */
let listing = null;
/** @type {RequestCard|null} */
let card = null;
/** @type {string|null} */
let note = null;
/** @type {string|null}  the request whose card is being read */
let loading = null;
/** @type {string|null}  the request whose approval is being sent */
let approving = null;

/** Said in place of Review while a card's current values are being read. */
export const READING_SENTENCE = 'reading values…';

/** Said on Approve while the approval is being sent. */
export const APPROVING_SENTENCE = 'Approving…';

/** @type {number}  bumped by a reset, so a read that left before it is dropped */
let epoch = 0;

/** Forget everything: the popover closed, or a test starts over. */
export function resetPrograms() {
  epoch += 1;
  listing = null;
  card = null;
  note = null;
  loading = null;
  approving = null;
}

/**
 * Read the listing again. Never throws: a listing that cannot be read leaves
 * the section out rather than showing a stale one.
 * @returns {Promise<void>}
 */
export async function refreshPrograms() {
  const then = epoch;
  /** @type {Listing|null} */
  let answer;
  try {
    answer = /** @type {Listing} */ (await targetRequest(BASE));
  } catch {
    answer = null;
  }
  // The popover closed while this read was out: it must not bring back the
  // listing the close forgot.
  if (then === epoch) listing = answer;
}

/** @param {string} tag @param {string} [cls] @param {string} [text] */
function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
}

/** @param {string} cls @param {string} text @returns {HTMLButtonElement} */
function button(cls, text) {
  const node = /** @type {HTMLButtonElement} */ (el('button', cls, text));
  node.type = 'button';
  return node;
}

/**
 * A value as the card shows it: a number as it is, anything else quoted, and
 * one not read as such.
 * @param {any} value
 * @returns {string}
 */
export function valueText(value) {
  if (value === null || value === undefined) return 'not read';
  if (typeof value === 'number') return String(value);
  return JSON.stringify(value);
}

/**
 * How long until an expiry, as a person reads it: `in 1 h 58 min`, `in 4 min`,
 * `in 30 s`, or `now` once it has passed.
 * @param {number|null|undefined} seconds
 * @returns {string}
 */
export function relativeTime(seconds) {
  if (seconds == null || !Number.isFinite(seconds)) return 'at an unknown time';
  if (seconds <= 0) return 'now';
  if (seconds < 60) return `in ${Math.round(seconds)} s`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `in ${minutes} min`;
  const hours = Math.floor(minutes / 60);
  const rest = minutes % 60;
  return rest ? `in ${hours} h ${rest} min` : `in ${hours} h`;
}

/** The Limits cell of a channel the limits database itself rules out. */
export const LIMITS_PROBLEM_WORDS = {
  not_in_limits: 'not in the limits database',
  not_writable: 'not writable',
};

/**
 * The limits bound a range sits inside, as the card shows it; for a channel
 * the database lists as not writable, or does not list, that instead.
 * @param {CardChannel} channel
 * @returns {string}
 */
export function limitsText(channel) {
  const ruled =
    LIMITS_PROBLEM_WORDS[/** @type {keyof typeof LIMITS_PROBLEM_WORDS} */ (channel.problem ?? '')];
  if (ruled) return ruled;
  const low = channel.limit_min;
  const high = channel.limit_max;
  if (low == null && high == null) return 'no numeric limits';
  return `[${low ?? '−∞'}, ${high ?? '+∞'}]`;
}

/**
 * One request's card: what is asked, what bounds it, and Approve.
 * @param {RequestCard} request
 * @param {{onApprove: (id: string) => void, onClose: () => void}} handlers
 * @param {boolean} [sending]  the approval is on its way: Approve waits, disabled
 * @returns {HTMLElement}
 */
export function renderRequestCard(request, handlers, sending = false) {
  const node = el('div', 'wa-card');
  node.dataset.request = request.id;
  node.append(
    el(
      'div',
      'wa-card-title',
      `${request.holder} asks to write on ${request.target}${request.label ? ` — ${request.label}` : ''}`
    )
  );
  const table = el('table', 'wa-channels');
  const head = el('tr');
  for (const word of ['Channel', 'Range', 'Limits', 'Now']) head.append(el('th', '', word));
  table.append(head);
  for (const channel of request.channels) {
    const line = el('tr', 'wa-channel');
    line.dataset.address = channel.address;
    if (channel.problem) line.dataset.problem = channel.problem;
    line.append(el('td', 'wa-address', channel.address));
    line.append(el('td', 'wa-range', `[${channel.min}, ${channel.max}]`));
    line.append(el('td', 'wa-limits', limitsText(channel)));
    line.append(el('td', 'wa-current', valueText(channel.current)));
    table.append(line);
  }
  node.append(table);
  const facts = el('dl', 'wa-facts');
  const completion =
    COMPLETION_WORDS[/** @type {keyof typeof COMPLETION_WORDS} */ (request.completion)] ??
    request.completion;
  for (const [term, value, title] of [
    ['Steps', `${request.steps} write calls, each one step`],
    ['At the end', completion],
    ['Expires', relativeTime(request.expires_in_s), request.expires_at],
    ['Lease', `${request.lease_s} s, renewed by the program while it runs`],
    ['Holder', request.holder],
  ]) {
    const dd = el('dd', '', value);
    if (title) dd.title = title;
    facts.append(el('dt', '', term), dd);
  }
  node.append(facts);
  if (request.limits_note) node.append(el('div', 'wa-note', request.limits_note));
  if (request.refusal) {
    const why = el(
      'div',
      'wa-refusal',
      request.refusal.phrase ||
        (request.refusal.ceiling
          ? `This login's write tier (${request.refusal.ceiling}) does not allow it.`
          : request.refusal.reason)
    );
    why.dataset.reason = request.refusal.reason;
    node.append(why);
  }
  const actions = el('div', 'wa-actions');
  const approve = button('wa-approve osprey-btn primary', sending ? APPROVING_SENTENCE : 'Approve');
  approve.disabled = sending || request.approvable !== true;
  approve.addEventListener('click', (event) => {
    event.stopPropagation();
    handlers.onApprove(request.id);
  });
  const close = button('wa-close osprey-btn', 'Close');
  close.addEventListener('click', (event) => {
    event.stopPropagation();
    handlers.onClose();
  });
  actions.append(approve, close);
  node.append(actions);
  return node;
}

/**
 * The section: the requests waiting on this login and its authorizations in
 * force, or `null` when there is neither, so the popover stays as it was.
 * @param {Listing|null} current
 * @param {RequestCard|null} open  the card being reviewed, if any
 * @param {string|null} status  the last gesture's outcome, if any
 * @param {{onReview: (id: string) => void, onApprove: (id: string) => void,
 *   onRevoke: (id: string) => void, onClose: () => void}} handlers
 * @param {string|null} [reading]  the request whose card is being read, if any
 * @param {string|null} [sending]  the request whose approval is being sent, if any
 * @returns {HTMLElement|null}
 */
export function renderPrograms(current, open, status, handlers, reading = null, sending = null) {
  const requests = current?.requests ?? [];
  const held = (current?.authorizations ?? []).filter((a) => a.in_force);
  if (requests.length === 0 && held.length === 0 && !open && !status) return null;
  const section = el('div', 'wa-section');
  section.append(el('div', 'osprey-popover-title', 'Programs'));
  if (open) section.append(renderRequestCard(open, handlers, sending === open.id));
  for (const request of requests) {
    if (open && open.id === request.id) continue;
    const line = el('div', 'wa-request');
    line.dataset.request = request.id;
    line.append(
      el(
        'span',
        'wa-what',
        `${request.holder} asks for ${request.steps} writes on ${request.target}` +
          (request.label ? ` (${request.label})` : '')
      )
    );
    if (reading === request.id) {
      line.append(el('span', 'wa-loading', READING_SENTENCE));
    } else {
      const review = button('wa-review osprey-btn', 'Review');
      review.addEventListener('click', (event) => {
        event.stopPropagation();
        handlers.onReview(request.id);
      });
      line.append(review);
    }
    section.append(line);
  }
  for (const grant of held) {
    const line = el('div', 'wa-held');
    line.dataset.authorization = grant.id;
    const what = el(
      'span',
      'wa-what',
      `${grant.holder} may write ${grant.steps} times on ${grant.target}` +
        (grant.label ? ` (${grant.label})` : '') +
        (grant.used == null ? '' : ` · ${grant.used}/${grant.steps} used`) +
        ` · ends ${relativeTime(grant.expires_in_s)}`
    );
    what.title = `Expires ${grant.expires_at}`;
    line.append(what);
    if (grant.self_issued === true) line.dataset.selfIssued = 'true';
    if (grant.revocable === false) {
      // Issued by the program itself at `allowed`, in a directory this terminal
      // only reads: the machine's switch leaving `allowed` ends it.
      line.append(el('span', 'wa-self-issued', SELF_ISSUED_SENTENCE));
    } else {
      const revoke = button('wa-revoke osprey-btn', 'Revoke');
      revoke.addEventListener('click', (event) => {
        event.stopPropagation();
        handlers.onRevoke(grant.id);
      });
      line.append(revoke);
    }
    section.append(line);
  }
  if (status) section.append(el('div', 'wa-status', status));
  return section;
}

/**
 * The section as it stands, wired to the routes: Review reads the card,
 * Approve and Revoke post, and every answer re-renders through *rerender*.
 * @param {string|null} sessionId
 * @param {() => void} rerender
 * @returns {HTMLElement|null}
 */
export function programsSection(sessionId, rerender) {
  const session = sessionId ? { session_id: sessionId } : {};
  /** @param {unknown} err */
  const said = (err) => (err instanceof Error ? err.message : String(err));
  return renderPrograms(
    listing,
    card,
    note,
    {
      onReview: async (id) => {
        // The card reads current values through the connector, which can take
        // seconds on a machine that does not answer: say so while it does.
        loading = id;
        rerender();
        try {
          card = /** @type {RequestCard} */ (await targetRequest(`${BASE}/requests/${id}`));
          note = null;
        } catch (err) {
          note = said(err);
        }
        loading = null;
        rerender();
      },
      onApprove: async (id) => {
        // Approving reads the current values again: say so while it waits.
        approving = id;
        rerender();
        try {
          await targetRequest(BASE, { method: 'POST', json: { request_id: id, ...session } });
          note = 'Approved.';
          card = null;
        } catch (err) {
          note = said(err);
        }
        approving = null;
        await refreshPrograms();
        rerender();
      },
      onRevoke: async (id) => {
        try {
          const answer = await targetRequest(`${BASE}/${id}/revoke`, {
            method: 'POST',
            json: session,
          });
          note =
            /** @type {any} */ (answer)?.revoked === false
              ? 'Already stopped.'
              : 'Revoked: its next write is refused.';
        } catch (err) {
          note = said(err);
        }
        await refreshPrograms();
        rerender();
      },
      onClose: () => {
        card = null;
        rerender();
      },
    },
    loading,
    approving
  );
}
