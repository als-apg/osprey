// @ts-check
/* OSPREY Web Terminal — Feedback Wiring (boot)
 *
 * The single place where the feedback feature's otherwise independent modules
 * meet: the rail's utility cluster (rail-utility.js), the dialog
 * (feedback-modal.js), the transport (feedback-client.js), the xterm
 * scrollback reader (scrollback-capture.js) and the deployment's own
 * configuration (GET /api/panels). app.js calls `initFeedback()` once at boot;
 * nothing else in the app needs to know the feature exists.
 *
 * It is a module of its own rather than a section of app.js for two reasons:
 * app.js is a boot script held under eslint's `max-lines` budget, and every
 * ambient dependency here (fetch, clipboard, window, the terminal, the session
 * id) arrives through the options object, so the whole wiring is testable
 * without a browser.
 *
 * THE RULE THAT MUST SURVIVE EVERY FUTURE EDIT: the action handlers run inside
 * the click turn and must not `await` anything before calling `sendFeedback()`
 * / `copyContext()`. Safari drops the transient user activation across an
 * await, and with it both the clipboard write and the popup. Everything the
 * handlers need is therefore resolved at boot (config, metadata) or read
 * synchronously (scrollback, form state).
 */

import { withPrefix } from './api.js';
import { copyContext, sendFeedback } from './feedback-client.js';
import { FeedbackModal } from './feedback-modal.js';
import { applyDocsLink, initRailUtility, onFeedbackClick } from './rail-utility.js';
import { captureScrollback } from './scrollback-capture.js';

/** Deployment config: rail/status docs target and the outbound channels. */
const PANELS_PATH = '/api/panels';

/** Carries `osprey.__version__`, the one metadata fact the page cannot know. */
const HEALTH_PATH = '/health';

/** Id of the status bar's Docs link in index.html. */
const STATUS_DOCS_LINK_ID = 'docs-link';

/** Class of the dialog's one notice line. */
const NOTICE_CLASS = 'feedback-notice';

/** Class of the read-only textarea offered when the clipboard refuses. */
const MANUAL_COPY_CLASS = 'feedback-manual-copy';

/**
 * Shown when a tracker channel is actioned without a tracker behind it. The
 * radios are rendered from the configured list, so this is a belt-and-braces
 * guard for a list that changed under an open dialog, not a posture a
 * deployment can configure into.
 */
const NO_TRACKER = 'This deployment has no such feedback tracker configured — use Local instead.';

/** Shown when the Email channel is picked on a deployment that blanked the address. */
const NO_EMAIL = 'This deployment has no feedback address configured — use Local instead.';

/**
 * Shown when the config read failed. Deliberately distinct from the two above:
 * those state a fact about the deployment, and this one states that the page
 * does not know that fact — claiming "no repository is configured" when the
 * read simply failed would be an assertion the client cannot back.
 */
const CONFIG_UNREADABLE =
  "Could not read this deployment's feedback configuration — use Local instead.";

/** Shown in the window before the config read lands (bounded by the timeout). */
const CONFIG_PENDING =
  "Still reading this deployment's feedback configuration — try again in a moment.";

/**
 * Deadline for each config read. A hung GET is not a rejection: without this
 * the two reads would stay pending forever and the outbound channels would
 * never resolve either way. The Feedback button is wired before the reads
 * start, so this bounds the config, never access to the dialog.
 */
const CONFIG_TIMEOUT_MS = 8000;

/**
 * @typedef {import('./feedback-modal.js').FeedbackFormState} FeedbackFormState
 * @typedef {import('./feedback-modal.js').FeedbackTracker} FeedbackTracker
 */

/** Tracker kinds the page knows how to open; anything else is dropped. */
const TRACKER_KINDS = Object.freeze({ github: 'GitHub', gitlab: 'GitLab' });

/**
 * @typedef {object} FeedbackBootOptions
 * @property {() => string|null} [getSessionId] - the live PTY session id
 *   (`terminal.js`'s `getCurrentSessionId`). Injected rather than imported so
 *   this module — and the dialog behind it — stay independent of the terminal.
 * @property {() => any} [getTerminal] - the live xterm instance, for the
 *   scrollback tier. Anything without a `buffer` (including `null`) yields an
 *   empty capture, so a deployment that cannot hand one over still submits
 *   text, metadata and the server-composed context.
 * @property {(path: string, init: any) => Promise<any>} [loadJSON] - GET-and-parse,
 *   for the two boot reads. Receives an `init` carrying the abort signal.
 *   Defaults to {@link readConfigJson}.
 * @property {(url: string, init: any) => Promise<any>} [fetch] - the POST used
 *   by the transport. Defaults to `window.fetch`.
 * @property {any} [clipboard] - defaults to `navigator.clipboard`.
 * @property {(url: string, target: string, features: string) => void} [windowOpen]
 * @property {(url: string) => void} [navigate] - assigns `location.href`.
 */

/**
 * @typedef {object} FeedbackConfig
 * @property {'pending'|'ok'|'unreadable'} status - how the config read went.
 * @property {FeedbackTracker[]} trackers - the issue trackers on offer, in
 *   render order; empty when this deployment configured none.
 * @property {string} email - maintainer address, or "" when this deployment
 *   retired the channel.
 * @property {string} version - `osprey.__version__`, for the metadata block.
 */

/**
 * Wire the rail's utility cluster and the feedback dialog into the page.
 *
 * Returns synchronously, with the Feedback button already live: the dialog is
 * built eagerly and the deployment's configuration is filled in when the reads
 * land. Waiting for them first would leave the button inert for the length of
 * a slow round trip — and forever behind a hung one, which is precisely when
 * an operator reaches for the feedback channel. The dialog needs nothing from
 * those reads except the outbound targets and the version line, and both are
 * consulted at click time.
 *
 * Never throws for a configuration read: a failed or hung `/api/panels` costs
 * the docs links and the outbound channels, never the Local channel.
 *
 * @param {FeedbackBootOptions} [options]
 * @returns {{modal: FeedbackModal, ready: Promise<void>}} the wired dialog, and
 *   a promise that settles (never rejects) once the configuration has been
 *   applied — for callers that want to act on the finished state, and tests.
 */
export function initFeedback(options = {}) {
  /** @type {FeedbackConfig} */
  const config = { status: 'pending', trackers: [], email: '', version: '' };

  const modal = createFeedbackModal(config, options);
  onFeedbackClick(() => modal.open());

  const loadJSON = options.loadJSON ?? readConfigJson;
  /** @param {string} path @returns {Promise<any>} null where the read failed */
  const read = (path) =>
    Promise.resolve()
      .then(() => loadJSON(path, { signal: timeoutSignal() }))
      .catch(() => null);

  // Applied SEPARATELY, not from one `Promise.all`. The two reads answer
  // different questions and fail differently — `/health` is the one behind the
  // auth gate — so a slow one must not withhold what the other already knows.
  // Waiting for both would leave the outbound channels reporting "still
  // reading" for up to the timeout because a version line had not arrived.
  const panelsApplied = read(PANELS_PATH).then((panels) => {
    config.status = panels === null ? 'unreadable' : 'ok';
    config.trackers = normalizeTrackers(panels == null ? undefined : panels.feedback_trackers);
    config.email = trimmedString(panels == null ? undefined : panels.feedback_email);
    // Handed to the dialog as well as kept here: the radios are built from
    // the list, and the dialog may already be open when it lands.
    modal.setTrackers(config.trackers);
    initRailUtility(panels);
    applyStatusDocsLink(panels);
  });
  const healthApplied = read(HEALTH_PATH).then((health) => {
    config.version = trimmedString(health == null ? undefined : health.version);
  });

  // The reads themselves cannot reject (`read` catches), so this catch is for
  // the appliers — a DOM helper throwing here would otherwise be an unhandled
  // rejection AND make this function's "never rejects" contract a lie.
  const ready = Promise.all([panelsApplied, healthApplied])
    .then(() => {})
    .catch((err) => console.error('Feedback config failed to apply:', err));

  return { modal, ready };
}

/**
 * GET-and-parse through the {@link withPrefix} chokepoint, honouring an abort.
 *
 * `api.js`'s `fetchJSON` would do everything here except carry the signal, and
 * the signal is the whole point: see {@link CONFIG_TIMEOUT_MS}.
 *
 * @param {string} path
 * @param {{signal?: AbortSignal}} [init]
 * @returns {Promise<any>}
 */
async function readConfigJson(path, init = {}) {
  const res = await fetch(withPrefix(path), { cache: 'no-store', signal: init.signal });
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
  return res.json();
}

/**
 * An abort signal for one config read, where the runtime offers one.
 *
 * @returns {AbortSignal|undefined}
 */
function timeoutSignal() {
  return typeof AbortSignal !== 'undefined' && typeof AbortSignal.timeout === 'function'
    ? AbortSignal.timeout(CONFIG_TIMEOUT_MS)
    : undefined;
}

/**
 * Point the status bar's Docs link at the same target as the rail's anchor.
 *
 * The two are one affordance in two places, so they follow one rule — and it
 * is literally one rule: both go through `rail-utility.js`'s `applyDocsLink`,
 * which is where that rule and the reasons for it are spelled out.
 *
 * `.status-item[hidden]` is declared in files.css beside `.status-item`
 * itself: that rule's `display: flex` would otherwise beat the user-agent's
 * `[hidden]`, and the link would stay painted after being "hidden".
 *
 * @param {any} panelsPayload - the `GET /api/panels` response, or null.
 * @returns {void}
 */
function applyStatusDocsLink(panelsPayload) {
  const link = document.getElementById(STATUS_DOCS_LINK_ID);
  if (!(link instanceof HTMLAnchorElement)) return;
  applyDocsLink(link, panelsPayload);
}

/**
 * Build the dialog with its three action callbacks bound to the transport.
 *
 * `config` is read at click time, not captured here: the dialog exists before
 * the configuration does (see {@link initFeedback}), and the operator may well
 * open it in between.
 *
 * @param {FeedbackConfig} config - filled in when the boot reads land.
 * @param {FeedbackBootOptions} options
 * @returns {FeedbackModal}
 */
function createFeedbackModal(config, options) {
  const getTerminal = options.getTerminal ?? (() => null);

  // The three handlers below close over `modal`, which is constructed from
  // them at the end of this function. Legal because a handler can only ever
  // run after the dialog it belongs to exists.

  // One action at a time, held as STATE rather than as the disabled buttons.
  // Disabling the row cannot be the guard on its own: the channel radios are
  // not part of it, and switching channel mid-flight makes the dialog rebuild
  // the action row with fresh, enabled buttons — Send, switch to a tracker, click
  // and the deployment has two submissions in flight for one report.
  let inFlight = false;

  /**
   * Claim the in-flight slot and grey the row out; returns the release.
   *
   * Called only AFTER the transport call has been issued, so nothing it
   * touches sits between the operator's click and the clipboard.
   *
   * @returns {() => void}
   */
  const hold = () => {
    inFlight = true;
    const unlock = lockActions(modal);
    return () => {
      inFlight = false;
      unlock();
    };
  };

  /**
   * Everything the transport needs, assembled synchronously.
   *
   * @param {FeedbackFormState} state
   * @returns {import('./feedback-client.js').FeedbackDeps}
   */
  const transportDeps = (state) => ({
    fetch: options.fetch ?? ((url, init) => window.fetch(url, init)),
    clipboard: options.clipboard ?? navigator.clipboard,
    windowOpen:
      options.windowOpen ?? ((url, target, features) => void window.open(url, target, features)),
    navigate:
      options.navigate ??
      ((url) => {
        window.location.href = url;
      }),
    // Read here, in the click turn, so the capture is what the operator was
    // looking at when they pressed the button — and only when it will be sent.
    scrollback: state.contextOn ? captureScrollback(getTerminal()) : '',
    metadata: buildMetadata(config),
    email: config.email,
    showSelectableText: (payload) => showManualCopy(modal, payload),
    onNotice: (notice) => showNotice(modal, notice),
  });

  /**
   * The Send / "Open … issue" / "Open email draft" buttons.
   *
   * @param {FeedbackFormState} state
   * @returns {void}
   */
  const submit = (state) => {
    if (inFlight) return;
    const unavailable = channelGuard(state, config);
    if (unavailable !== null) {
      startAction(modal);
      // A microtask, not this turn: `startAction` has only just created the
      // live region, and a region that is created and written in one turn is
      // not announced. This notice is the entire outcome of the click, so an
      // unannounced one leaves a screen-reader user with silence.
      queueMicrotask(() => showNotice(modal, { kind: 'error', message: unavailable }));
      return;
    }
    startAction(modal);
    // Issued BEFORE the hold: everything the browser gates behind the user
    // activation happens inside `sendFeedback`'s synchronous section, and a
    // DOM write in front of it is one more thing between the click and the
    // clipboard.
    const pending = sendFeedback(state, transportDeps(state));
    report(pending, hold());
  };

  /**
   * The standalone "Copy session context" button.
   *
   * @param {FeedbackFormState} state
   * @returns {void}
   */
  const copy = (state) => {
    if (inFlight) return;
    startAction(modal);
    const pending = copyContext(state, transportDeps(state));
    report(pending, hold());
  };

  const modal = new FeedbackModal({
    getSessionId: options.getSessionId ?? (() => null),
    onSubmit: submit,
    onOpenChannel: submit,
    onCopy: copy,
  });
  return modal;
}

/**
 * Whether an outbound channel can be reached at all on this deployment.
 *
 * Three different answers, deliberately: a blanked `web.feedback.email` is a
 * legitimate configuration (the same posture as a blanked `web.docs_url`) and
 * reads as "not available here"; a config the page could not read reads as
 * "I do not know"; and the window before the reads land says so rather than
 * guessing. What none of them may do is let the click through, which would
 * open a `mailto:` with no recipient — or, for a tracker channel whose tracker
 * is no longer on the list, an issue tab on a URL built from nothing.
 *
 * The trackers themselves need no "not configured" answer: a tracker that is
 * not configured is not rendered, so it cannot be picked.
 *
 * Local is never guarded — it needs no configuration, which is exactly why
 * every one of these notices points at it.
 *
 * @param {FeedbackFormState} state
 * @param {FeedbackConfig} config
 * @returns {string|null} the notice to show, or null when the channel is usable.
 */
function channelGuard(state, config) {
  const channel = state.channel;
  if (channel === 'local') return null;
  if (config.status === 'pending') return CONFIG_PENDING;
  if (config.status === 'unreadable') return CONFIG_UNREADABLE;
  if (channel === 'email') return config.email === '' ? NO_EMAIL : null;
  const tracker = state.tracker;
  if (tracker === null || !config.trackers.some((known) => known.id === tracker.id)) {
    return NO_TRACKER;
  }
  return null;
}

/**
 * The `/api/panels` tracker list as the dialog wants it: one entry per usable
 * server entry, in order, with a radio value derived from what it opens.
 *
 * The server already validated the list, so a malformed entry here means a
 * server/page version skew; it is dropped rather than rendered as a radio
 * that opens nothing.
 *
 * @param {unknown} value - `panels.feedback_trackers`
 * @returns {FeedbackTracker[]}
 */
function normalizeTrackers(value) {
  if (!Array.isArray(value)) return [];
  /** @type {FeedbackTracker[]} */
  const trackers = [];
  for (const entry of value) {
    if (entry === null || typeof entry !== 'object') continue;
    const kind = trimmedString(entry.kind);
    if (kind !== 'github' && kind !== 'gitlab') continue;
    const target = trimmedString(kind === 'github' ? entry.repo : entry.url);
    if (target === '') continue;
    const label = trimmedString(entry.label) || TRACKER_KINDS[kind];
    trackers.push({ id: `${kind}:${target}`, kind, label, target });
  }
  return trackers;
}

/**
 * The deployment metadata block carried by a prefilled issue or mail draft.
 *
 * Only the two facts the *page* is the best source for: the running version
 * (which the browser cannot know) and the browser itself (which the server
 * cannot know). Everything else — identity, timestamps, app name — is stamped
 * server-side onto the record and the composed bundle. Blank values are
 * dropped by `buildPrefillBody`, so a failed `/health` degrades quietly.
 *
 * @param {FeedbackConfig} config
 * @returns {Record<string, unknown>}
 */
function buildMetadata(config) {
  return {
    'OSPREY version': config.version,
    Browser: typeof navigator === 'undefined' ? '' : navigator.userAgent,
  };
}

/**
 * Clear the previous action's output and make sure the notice line exists.
 *
 * The line is created *before* the request goes out rather than when the
 * notice arrives: a `role="status"` region has to be in the document before
 * its text changes for assistive tech to announce the change.
 *
 * @param {FeedbackModal} modal
 * @returns {void}
 */
function startAction(modal) {
  const body = modal.bodyElement;
  if (body === null) return;
  const stale = body.querySelector(`.${MANUAL_COPY_CLASS}`);
  if (stale !== null) stale.remove();
  const notice = noticeElement(modal);
  if (notice !== null) {
    notice.textContent = '';
    delete notice.dataset.kind;
  }
}

/**
 * Show the transport's one-line outcome inside the dialog.
 *
 * @param {FeedbackModal} modal
 * @param {import('./feedback-client.js').FeedbackNotice} notice
 * @returns {void}
 */
function showNotice(modal, notice) {
  const line = noticeElement(modal);
  if (line === null) return;
  line.dataset.kind = notice.kind;
  line.textContent = notice.message;
}

/**
 * The dialog's notice line, created on first use for the open dialog.
 *
 * Returns null while the dialog is closed — the node tree does not outlive an
 * open, so a late notice from an action the operator has already dismissed
 * simply has nowhere to go.
 *
 * @param {FeedbackModal} modal
 * @returns {HTMLElement|null}
 */
function noticeElement(modal) {
  const body = modal.bodyElement;
  if (body === null) return null;

  const existing = body.querySelector(`.${NOTICE_CLASS}`);
  if (existing instanceof HTMLElement) return existing;

  const line = document.createElement('p');
  line.className = `feedback-channel-hint ${NOTICE_CLASS}`;
  line.setAttribute('role', 'status');
  body.appendChild(line);
  return line;
}

/**
 * Last rung of the clipboard ladder: put the payload where it can be selected.
 *
 * @param {FeedbackModal} modal
 * @param {string} payload - the finished clipboard string.
 * @returns {void}
 */
function showManualCopy(modal, payload) {
  const body = modal.bodyElement;
  if (body === null) return;

  const existing = body.querySelector(`.${MANUAL_COPY_CLASS}`);
  if (existing !== null) existing.remove();

  const area = document.createElement('textarea');
  area.className = `feedback-text ${MANUAL_COPY_CLASS}`;
  area.readOnly = true;
  area.rows = 6;
  area.value = payload;
  area.setAttribute('aria-label', 'Feedback payload to copy');
  body.appendChild(area);
  // Pre-selected so the fallback is one keystroke, not a drag across a
  // 60 KB payload.
  area.focus();
  area.select();
}

/**
 * Grey the action row out for the length of one in-flight action.
 *
 * The *visible* half of the one-action-at-a-time rule; the enforcing half is
 * the `inFlight` flag in `createFeedbackModal`, because these buttons do not
 * survive a channel switch. The row is locked only AFTER the transport call
 * has been issued, so nothing here ever runs between the click and the
 * clipboard.
 *
 * The unlock restores each button's PREVIOUS disabled state rather than
 * enabling everything: "Copy session context" is legitimately disabled
 * whenever the context box is unticked, and re-enabling it would offer a
 * control the dialog means to withhold. Buttons detached in the meantime (a
 * channel switch rebuilds the row) take the write harmlessly.
 *
 * @param {FeedbackModal} modal
 * @returns {() => void} the unlock, safe to call once the dialog has closed.
 */
function lockActions(modal) {
  const body = modal.bodyElement;
  if (body === null) return () => {};

  /** @type {HTMLButtonElement[]} */
  const buttons = [];
  body.querySelectorAll('.feedback-action').forEach((node) => {
    if (node instanceof HTMLButtonElement) buttons.push(node);
  });
  const before = buttons.map((button) => button.disabled);
  buttons.forEach((button) => {
    button.disabled = true;
  });

  return () => {
    buttons.forEach((button, index) => {
      button.disabled = before[index];
    });
  };
}

/**
 * Swallow-and-log guard for a transport call, and the release of its lock.
 *
 * Both transport functions document that they never reject — every failure
 * already arrives as a notice. The `catch` exists so that a dependency
 * throwing somewhere unexpected surfaces in the console instead of as an
 * unhandled rejection, and the `finally` is what guarantees the action row
 * comes back even then.
 *
 * @param {Promise<unknown>} pending
 * @param {() => void} unlock
 * @returns {void}
 */
function report(pending, unlock) {
  pending.catch((err) => console.error('Feedback action failed:', err)).finally(unlock);
}

/**
 * A configuration value as a trimmed string; anything else reads as unset.
 *
 * @param {unknown} value
 * @returns {string}
 */
function trimmedString(value) {
  return typeof value === 'string' ? value.trim() : '';
}
