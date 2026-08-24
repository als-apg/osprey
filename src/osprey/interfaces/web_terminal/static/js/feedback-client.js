// @ts-check
/* OSPREY Web Terminal — Feedback Client (transport)
 *
 * Everything the feedback dialog does that is not DOM: build the POST bodies,
 * put the unified payload on the clipboard, open the GitHub tab or the mail
 * draft, and turn the outcome into one notice line. No element is created or
 * read here — the dialog owns the DOM and receives a plain result object (and,
 * optionally, an `onNotice` callback) to render.
 *
 * Two properties are load-bearing and easy to break:
 *
 *   1. ORDER. For the GitHub and Email channels the clipboard write and the
 *      navigation must both be issued inside the click turn, BEFORE the POST
 *      settles. Safari drops the transient user activation across an await, so
 *      a copy issued afterwards is refused and a `window.open` afterwards looks
 *      like an unsolicited popup. That is why the clipboard is fed a *promise*
 *      (`new ClipboardItem({'text/plain': postPromise.then(...)})`) instead of
 *      an awaited string: the write is registered immediately and the browser
 *      resolves the content later. Everything above the marked line in
 *      `sendFeedback` runs synchronously in the caller's turn — including the
 *      pointer-vs-full decision, which depends on nothing but the form state
 *      and the URL caps and is therefore known before any await.
 *
 *   2. NOTHING THROWS. A failed paper-trail POST must never cost the user the
 *      issue tab they asked for, and nginx answers an oversized body with an
 *      HTML page — so no response body is ever handed to `res.json()`; it is
 *      read as text and parsed defensively. Every path resolves with a result
 *      object carrying a notice.
 *
 * All I/O arrives through `deps`, so the tests drive this without a real
 * `fetch`, a real clipboard or a real window.
 */

import { withPrefix } from './api.js';
import {
  PASTE_POINTER,
  buildGitHubIssueUrl as defaultGitHubIssueUrl,
  buildMailto as defaultMailto,
  buildPrefillBody as defaultPrefillBody,
  utf8Length,
} from './feedback-prefill.js';

/** Endpoint that writes one record and returns the finished clipboard payload. */
export const FEEDBACK_PATH = '/api/feedback';

/** Endpoint that composes session context only, writing no record. */
export const BUNDLE_PATH = '/api/feedback/bundle';

/**
 * Field of the {@link BUNDLE_PATH} response holding the server-sized context
 * string. Deliberately not `payload`: the send endpoint's `payload` is the
 * whole clipboard string (text + separator + bundle), while this is the bundle
 * section alone, and one name for two different things invites a silent mix-up.
 */
export const BUNDLE_RESPONSE_FIELD = 'bundle';

/** `context_status` the server stamps when the session has no transcript. */
export const TRANSCRIPT_NOT_FOUND = 'transcript_not_found';

/** Generic message for an oversized submission (incl. a proxy's non-JSON 413). */
export const NOTICE_TOO_LARGE =
  'Submission too large — shorten your text or untick session context.';

/**
 * Shown when the paper-trail POST failed but the outbound handoff succeeded
 * and a paste is owed. The POST is also what composes the full payload, so a
 * failed one leaves only the typed text on the clipboard — and the pointer
 * line in the draft promises "your full report", which this line corrects.
 */
export const NOTICE_NOT_RECORDED =
  'not recorded on this deployment — your clipboard holds your text only; paste that into the draft';

/** As above, for a FULL-mode send where the draft is already complete. */
export const NOTICE_NOT_RECORDED_FULL =
  'not recorded on this deployment — the draft you opened is complete and unaffected';

/** Shown when the server found no transcript for the submitted session. */
export const NOTICE_NO_TRANSCRIPT =
  'no transcript found for this session — sent text and metadata only';

/** Shown when the clipboard refused and the payload is offered for selection. */
export const NOTICE_MANUAL_COPY =
  'Could not reach your clipboard — copy the payload shown below by hand.';

/** Shown when no rung of the clipboard ladder was available at all. */
export const NOTICE_COPY_FAILED = 'Could not copy to your clipboard.';

/** Shown when "Copy session context" is used without a session. */
export const NOTICE_NO_SESSION = 'No session yet — there is no context to copy.';

/** Shown when the context bundle reached the clipboard. */
export const NOTICE_CONTEXT_COPIED = 'Session context copied to your clipboard.';

/** Shown when the bundle response carried no context string. */
export const NOTICE_EMPTY_BUNDLE = 'This deployment returned no session context.';

/**
 * Appended to the outbound confirmation of a POINTER-mode send: the draft
 * opened carrying one line that points at the clipboard, and this repeats the
 * instruction in the dialog for whoever comes back to it before pasting.
 */
export const NOTICE_PASTE_GITHUB =
  'your full report is on your clipboard — paste it into the issue body';

/** Email variant of {@link NOTICE_PASTE_GITHUB}. */
export const NOTICE_PASTE_EMAIL =
  'your full report is on your clipboard — paste it into the draft';

/**
 * Prefilled issue title / mail subject. Deliberately generic: the report text
 * is not a title (a short report would duplicate itself, a long one would be
 * an arbitrary slice), so the sender is given a stable name to overwrite.
 */
export const DEFAULT_TITLE = 'OSPREY feedback';

const REQUEST_FAILED = 'Feedback request failed';
const TEXT_MIME = 'text/plain';

/** UTF-16 length of the session-id prefix quoted in a derived title. */
const SESSION_TITLE_CHARS = 8;

/**
 * @typedef {object} FeedbackForm
 * @property {string} text - what the user typed
 * @property {'local'|'github'|'email'} channel
 * @property {boolean} metadataOn - deployment-metadata checkbox
 * @property {boolean} contextOn - session-context checkbox
 * @property {string|null} [sessionId] - may be null (no session yet) or stale
 * @property {string} [title] - overrides the title derived from `text`
 */

/** The slice of `Response` this module reads. `json()` is deliberately unused. */
/** @typedef {{ok: boolean, status: number, text: () => Promise<string>}} FeedbackResponse */

/** @typedef {(url: string, init: {method: string, headers: Record<string, string>, body: string}) => Promise<FeedbackResponse>} FetchLike */

/** The slice of `navigator.clipboard` this module uses; both rungs optional. */
/** @typedef {{write?: (items: any[]) => Promise<void>, writeText?: (text: string) => Promise<void>}} ClipboardLike */

/**
 * Which rung of the clipboard ladder landed the payload. `skipped` means no
 * write was attempted at all — a FULL-mode send owes the clipboard nothing,
 * and overwriting whatever the operator had there would be pure loss.
 */
/** @typedef {'clipboard'|'clipboard-text'|'manual'|'none'|'skipped'} CopyRung */

/** @typedef {{kind: 'success'|'warning'|'error', message: string}} FeedbackNotice */

/**
 * @typedef {object} FeedbackResult
 * @property {boolean} ok - the request reached the server and succeeded
 * @property {string|null} id - record id, when one was written
 * @property {CopyRung} copied
 * @property {boolean} opened - an issue tab or mail draft was handed off
 * @property {boolean} needsPaste - the draft opened in POINTER mode and the
 *   full report was put on the clipboard for the operator to paste
 * @property {string|null} contextStatus - server's `context_status`, if any
 * @property {string|null} error - technical reason, for the caller's own use
 * @property {FeedbackNotice} notice - the one line to show the user
 */

/**
 * @typedef {object} FeedbackDeps
 * @property {FetchLike} fetch
 * @property {ClipboardLike} [clipboard] - usually `navigator.clipboard`
 * @property {(url: string, target: string, features: string) => void} [windowOpen]
 * @property {(url: string) => void} [navigate] - assigns `location.href` (mailto)
 * @property {string} [scrollback] - from `captureScrollback(term)`
 * @property {Record<string, unknown>} [metadata] - deployment metadata block
 * @property {string} [githubRepo] - `owner/name` from `web.feedback.github_repo`
 * @property {string} [email] - maintainer address from `web.feedback.email`
 * @property {(payload: string) => void} [showSelectableText] - last copy rung
 * @property {(notice: FeedbackNotice) => void} [onNotice]
 * @property {typeof defaultGitHubIssueUrl} [buildGitHubIssueUrl]
 * @property {typeof defaultMailto} [buildMailto]
 * @property {typeof defaultPrefillBody} [buildPrefillBody]
 */

/**
 * Send the user's feedback over the selected channel.
 *
 * Local waits for the record and reports its id. GitHub and Email fire one
 * paper-trail POST, feed the clipboard from its promise and hand off to the
 * browser — both inside the click turn — then report what happened; a failed
 * POST downgrades to a notice and never blocks the handoff.
 *
 * @param {FeedbackForm} form - the dialog's plain state, not its DOM
 * @param {FeedbackDeps} deps
 * @returns {Promise<FeedbackResult>} never rejects
 */
export async function sendFeedback(form, deps) {
  const channel = normalizeChannel(form.channel);
  const request = postJson(deps, FEEDBACK_PATH, buildSendBody(form, deps));

  if (channel === 'local') {
    try {
      return localResult(deps, await request);
    } catch (err) {
      const message = messageOf(err);
      return { ...emptyResult(), error: message, notice: emit(deps, { kind: 'error', message }) };
    }
  }

  // --- synchronous section: runs inside the click turn, no `await` above it ---
  const link = buildOutboundLink(form, deps);
  /** @type {Promise<CopyRung>} */
  let copying = Promise.resolve('skipped');
  if (link.needsPaste) {
    // When the POST fails there is no server payload, so the clipboard falls
    // back to the user's own words — NOTICE_NOT_RECORDED tells them the copy
    // carries the text alone. A FULL-mode send never reaches this block: its
    // draft is complete as opened and the operator's clipboard is not touched.
    const fallbackPayload = String(form.text ?? '');
    const payload = request.then(
      (data) => (typeof data.payload === 'string' && data.payload ? data.payload : fallbackPayload),
      () => fallbackPayload
    );
    copying = copyPayload(deps, payload);
  }
  let opened = false;
  if (channel === 'github') {
    if (typeof deps.windowOpen === 'function') {
      deps.windowOpen(link.url, '_blank', 'noopener');
      opened = true;
    }
  } else if (typeof deps.navigate === 'function') {
    deps.navigate(link.url);
    opened = true;
  }
  // --- end of the synchronous section ---

  const [copied, outcome] = await Promise.all([copying, settle(request)]);
  const id = readString(outcome.data, 'id');
  const contextStatus = readString(outcome.data, 'context_status');
  return {
    ok: outcome.ok,
    id,
    copied,
    opened,
    needsPaste: link.needsPaste,
    contextStatus,
    error: outcome.error,
    notice: emit(
      deps,
      outboundNotice(outcome.ok, id, contextStatus, copied, channel, link.needsPaste)
    ),
  };
}

/**
 * Copy the session context on its own, writing no record.
 *
 * The server sizes the bundle against the feedback text's UTF-8 byte length and
 * builds it from the same metadata checkbox the send path reads, so the result
 * is byte-identical to the bundle section of a send payload; this module copies
 * what comes back verbatim and does no size arithmetic.
 *
 * @param {FeedbackForm} form
 * @param {FeedbackDeps} deps
 * @returns {Promise<FeedbackResult>} never rejects
 */
export async function copyContext(form, deps) {
  const sessionId = form.sessionId ? String(form.sessionId) : '';
  if (!sessionId) {
    return {
      ...emptyResult(),
      error: NOTICE_NO_SESSION,
      notice: emit(deps, { kind: 'error', message: NOTICE_NO_SESSION }),
    };
  }

  // --- synchronous section: the clipboard write is registered before awaiting ---
  const request = postJson(deps, BUNDLE_PATH, {
    session_id: sessionId,
    // Counted in UTF-8 bytes, never `String.length`: the server's budget is a
    // byte budget, and the two differ for every non-ASCII character.
    text_len: utf8Length(String(form.text ?? '')),
    scrollback: String(deps.scrollback ?? ''),
    // Read exactly as the send body reads it: the deployment-metadata checkbox
    // governs both buttons, so what is copied is what would be sent.
    include_metadata: form.metadataOn === true,
  });
  const payload = request.then((data) => {
    const bundle = data[BUNDLE_RESPONSE_FIELD];
    // Rejecting keeps an empty string off the user's clipboard.
    if (typeof bundle !== 'string' || !bundle) throw new Error(NOTICE_EMPTY_BUNDLE);
    return bundle;
  });
  const copying = copyPayload(deps, payload);
  // --- end of the synchronous section ---

  const [copied, outcome] = await Promise.all([copying, settle(payload)]);
  return {
    ...emptyResult(),
    ok: outcome.ok,
    copied,
    error: outcome.error,
    notice: emit(deps, contextNotice(outcome.ok, outcome.error, copied)),
  };
}

/**
 * POST a JSON body through the {@link withPrefix} chokepoint.
 *
 * Not `async`: the fetch must be in flight before the caller's next statement,
 * which is what lets the clipboard be fed from the returned promise.
 *
 * @param {FeedbackDeps} deps
 * @param {string} path
 * @param {Record<string, unknown>} body
 * @returns {Promise<Record<string, any>>}
 */
function postJson(deps, path, body) {
  return deps
    .fetch(withPrefix(path), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    .then(readJson);
}

/**
 * Read a response body without ever letting a non-JSON page throw.
 *
 * nginx answers an oversized request with an HTML error page, so the body is
 * taken as text and parsed inside a try — `response.json()` is never called.
 *
 * @param {FeedbackResponse} response
 * @returns {Promise<Record<string, any>>}
 */
async function readJson(response) {
  let raw = '';
  try {
    raw = await response.text();
  } catch {
    raw = '';
  }
  /** @type {Record<string, any>|null} */
  let data = null;
  if (raw) {
    try {
      const parsed = JSON.parse(raw);
      if (parsed && typeof parsed === 'object') data = parsed;
    } catch {
      data = null; // an HTML error page — not an error condition of its own
    }
  }
  if (!response.ok) throw new Error(errorMessageFor(response.status, data));
  return data ?? {};
}

/**
 * User-facing message for a non-OK response.
 *
 * @param {number} status
 * @param {Record<string, any>|null} data - parsed body, or null when not JSON
 * @returns {string}
 */
function errorMessageFor(status, data) {
  if (status === 413) return NOTICE_TOO_LARGE;
  const detail = data && typeof data.detail === 'string' ? data.detail.trim() : '';
  return detail || `${REQUEST_FAILED} (HTTP ${status})`;
}

/**
 * Put `payload` on the clipboard, descending the fallback ladder as each rung
 * proves unavailable: promise-backed `ClipboardItem` → `writeText` → a
 * caller-rendered selectable block.
 *
 * A payload that never resolves (the request failed) reports `none` without
 * attempting the lower rungs — overwriting the clipboard with a placeholder
 * would destroy whatever the user had there.
 *
 * @param {FeedbackDeps} deps
 * @param {Promise<string>} payload
 * @returns {Promise<CopyRung>}
 */
function copyPayload(deps, payload) {
  const clipboard = deps.clipboard;
  const settled = payload.then(
    (text) => ({ text, failed: false }),
    () => ({ text: '', failed: true })
  );

  // Bound, because `navigator.clipboard.write` needs its own receiver.
  const write =
    clipboard != null && typeof clipboard.write === 'function'
      ? clipboard.write.bind(clipboard)
      : null;

  if (write && typeof ClipboardItem !== 'undefined' && typeof Blob !== 'undefined') {
    const blob = payload.then((text) => new Blob([text], { type: TEXT_MIME }));
    // The ClipboardItem must still see the rejection (that is how the browser
    // learns the copy is off), but an otherwise unobserved rejection surfaces
    // as an unhandled promise — so mark this one handled without consuming it.
    blob.catch(() => {});
    const wrote = write([new ClipboardItem({ [TEXT_MIME]: blob })]).then(
      () => true,
      () => false
    );
    return Promise.all([wrote, settled]).then(([ok, result]) => {
      if (result.failed) return 'none';
      return ok ? 'clipboard' : fallbackCopy(deps, result.text);
    });
  }

  return settled.then((result) => (result.failed ? 'none' : fallbackCopy(deps, result.text)));
}

/**
 * The two lower rungs of the clipboard ladder. Both are already outside the
 * click turn, so a browser that guards the clipboard behind user activation
 * will refuse `writeText` too — which is exactly why the last rung hands the
 * text back to the caller to display.
 *
 * @param {FeedbackDeps} deps
 * @param {string} text
 * @returns {Promise<CopyRung>}
 */
async function fallbackCopy(deps, text) {
  const clipboard = deps.clipboard;
  if (clipboard != null && typeof clipboard.writeText === 'function') {
    try {
      await clipboard.writeText(text);
      return 'clipboard-text';
    } catch {
      /* fall through to the selectable block */
    }
  }
  if (typeof deps.showSelectableText === 'function') {
    deps.showSelectableText(text);
    return 'manual';
  }
  return 'none';
}

/**
 * Request body for {@link FEEDBACK_PATH}. `session_id` and `scrollback` travel
 * only when the context checkbox is ticked — the server composes nothing it was
 * not asked for, and the record's flags mirror what was actually sent.
 *
 * @param {FeedbackForm} form
 * @param {FeedbackDeps} deps
 * @returns {Record<string, unknown>}
 */
function buildSendBody(form, deps) {
  const contextOn = form.contextOn === true;
  /** @type {Record<string, unknown>} */
  const body = {
    text: String(form.text ?? ''),
    channel: normalizeChannel(form.channel),
    include_metadata: form.metadataOn === true,
    include_context: contextOn,
  };
  if (contextOn) {
    if (form.sessionId) body.session_id = String(form.sessionId);
    body.scrollback = String(deps.scrollback ?? '');
  }
  return body;
}

/**
 * Build the prefilled outbound URL for the current channel.
 *
 * With session context attached the body is {@link PASTE_POINTER} outright —
 * context never fits a URL, so the whole report (text, metadata, context)
 * travels on the clipboard and one paste completes the draft. Without context
 * the full body is attempted and the builder falls back to the pointer only
 * when the URL cap forces it to.
 *
 * @param {FeedbackForm} form
 * @param {FeedbackDeps} deps
 * @returns {{url: string, needsPaste: boolean}} `needsPaste` means the draft
 *   carries the pointer line, so the payload MUST reach the clipboard — the
 *   pointer promises exactly that.
 */
function buildOutboundLink(form, deps) {
  const contextOn = form.contextOn === true;
  let body;
  if (contextOn) {
    // No metadata block either: the server-composed payload carries the
    // deployment tier (version, app, browser, session id), so a prefilled
    // block would only duplicate it under the paste.
    body = PASTE_POINTER;
  } else {
    const buildBody = deps.buildPrefillBody ?? defaultPrefillBody;
    const metadata = form.metadataOn === true ? { ...(deps.metadata ?? {}) } : {};
    body = buildBody(String(form.text ?? ''), metadata);
  }
  const title = deriveTitle(form);

  if (normalizeChannel(form.channel) === 'github') {
    const build = deps.buildGitHubIssueUrl ?? defaultGitHubIssueUrl;
    const issue = build(String(deps.githubRepo ?? ''), title, body);
    return { url: issue.url, needsPaste: contextOn || issue.needsPaste === true };
  }
  const build = deps.buildMailto ?? defaultMailto;
  const draft = build(String(deps.email ?? ''), title, body);
  return { url: draft.url, needsPaste: contextOn || draft.needsPaste === true };
}

/**
 * Issue title / mail subject: an explicit `form.title` verbatim, otherwise
 * {@link DEFAULT_TITLE} — suffixed with a session-id prefix when context is
 * attached, so two reports from one inbox stay tellable apart and the subject
 * alone finds the deployment's record. Never derived from the report text: a
 * short report would become its own title and a long one an arbitrary slice,
 * and the builders never shorten a title, so it must stay bounded here.
 *
 * @param {FeedbackForm} form
 * @returns {string}
 */
function deriveTitle(form) {
  const explicit = String(form.title ?? '').trim();
  if (explicit) return explicit;
  if (form.contextOn === true && form.sessionId) {
    return `${DEFAULT_TITLE} (session ${String(form.sessionId).slice(0, SESSION_TITLE_CHARS)})`;
  }
  return DEFAULT_TITLE;
}

/**
 * Settle a request into a plain outcome, so neither branch of the caller has to
 * handle a rejection.
 *
 * @param {Promise<any>} request
 * @returns {Promise<{ok: boolean, data: Record<string, any>, error: string|null}>}
 */
function settle(request) {
  return request.then(
    (data) => ({
      ok: true,
      data: data && typeof data === 'object' ? data : {},
      error: /** @type {string|null} */ (null),
    }),
    (err) => ({
      ok: false,
      data: /** @type {Record<string, any>} */ ({}),
      error: /** @type {string|null} */ (messageOf(err)),
    })
  );
}

/**
 * @param {FeedbackDeps} deps
 * @param {Record<string, any>} data - the send endpoint's response
 * @returns {FeedbackResult}
 */
function localResult(deps, data) {
  const id = readString(data, 'id');
  const contextStatus = readString(data, 'context_status');
  return {
    ...emptyResult(),
    ok: true,
    id,
    contextStatus,
    notice: emit(deps, recordedNotice(id, contextStatus)),
  };
}

/**
 * The one line to show after an outbound (GitHub/Email) submission.
 *
 * A POINTER-mode send is judged in priority order: a clipboard the user has
 * to act on outranks a missing paper trail, which outranks the ordinary
 * confirmation — and the confirmation names the paste step, except after a
 * no-transcript warning, whose own wording stands because "paste the full
 * report" would promise context the server just said it could not find.
 * A FULL-mode send involves no clipboard at all, so only the paper trail can
 * have gone wrong.
 *
 * @param {boolean} ok
 * @param {string|null} id
 * @param {string|null} contextStatus
 * @param {CopyRung} copied
 * @param {'local'|'github'|'email'} channel
 * @param {boolean} needsPaste - the draft opened in POINTER mode
 * @returns {FeedbackNotice}
 */
function outboundNotice(ok, id, contextStatus, copied, channel, needsPaste) {
  if (!needsPaste) {
    if (!ok) return { kind: 'warning', message: NOTICE_NOT_RECORDED_FULL };
    return recordedNotice(id, contextStatus);
  }
  if (copied === 'none') return { kind: 'error', message: NOTICE_COPY_FAILED };
  if (copied === 'manual') return { kind: 'warning', message: NOTICE_MANUAL_COPY };
  if (!ok) return { kind: 'warning', message: NOTICE_NOT_RECORDED };
  const base = recordedNotice(id, contextStatus);
  if (base.kind !== 'success') return base;
  const hint = channel === 'email' ? NOTICE_PASTE_EMAIL : NOTICE_PASTE_GITHUB;
  return { kind: 'success', message: `${base.message} — ${hint}` };
}

/**
 * @param {boolean} ok
 * @param {string|null} error
 * @param {CopyRung} copied
 * @returns {FeedbackNotice}
 */
function contextNotice(ok, error, copied) {
  if (!ok) return { kind: 'error', message: error || NOTICE_EMPTY_BUNDLE };
  if (copied === 'none') return { kind: 'error', message: NOTICE_COPY_FAILED };
  if (copied === 'manual') return { kind: 'warning', message: NOTICE_MANUAL_COPY };
  return { kind: 'success', message: NOTICE_CONTEXT_COPIED };
}

/**
 * @param {string|null} id
 * @param {string|null} contextStatus
 * @returns {FeedbackNotice}
 */
function recordedNotice(id, contextStatus) {
  const base = id ? `Recorded on this deployment (${id})` : 'Recorded on this deployment';
  if (contextStatus === TRANSCRIPT_NOT_FOUND) {
    return { kind: 'warning', message: `${base} — ${NOTICE_NO_TRANSCRIPT}` };
  }
  return { kind: 'success', message: base };
}

/**
 * Hand a notice to the caller's renderer and return it for the result object.
 *
 * @param {FeedbackDeps} deps
 * @param {FeedbackNotice} notice
 * @returns {FeedbackNotice}
 */
function emit(deps, notice) {
  if (typeof deps.onNotice === 'function') deps.onNotice(notice);
  return notice;
}

/** Result fields shared by every path; spread and override what differs. */
/** @returns {FeedbackResult} */
function emptyResult() {
  return {
    ok: false,
    id: null,
    copied: 'none',
    opened: false,
    needsPaste: false,
    contextStatus: null,
    error: null,
    notice: { kind: 'error', message: REQUEST_FAILED },
  };
}

/**
 * @param {Record<string, any>} data
 * @param {string} key
 * @returns {string|null}
 */
function readString(data, key) {
  const value = data[key];
  return typeof value === 'string' && value ? value : null;
}

/**
 * @param {unknown} channel
 * @returns {'local'|'github'|'email'}
 */
function normalizeChannel(channel) {
  if (channel === 'github' || channel === 'email') return channel;
  return 'local';
}

/**
 * @param {unknown} err
 * @returns {string}
 */
function messageOf(err) {
  if (err instanceof Error && err.message) return err.message;
  const text = String(err ?? '').trim();
  return text || REQUEST_FAILED;
}
