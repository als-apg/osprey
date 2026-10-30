// @ts-check
/* OSPREY Web Terminal — the sidebar footer's Docs and Feedback rows
 *
 * The sidebar's two NON-panel rows above ☰ More, in `.sidebar-foot`: an
 * anchor to the deployment's documentation and the button that opens the
 * feedback dialog. Their markup is hand-written in index.html, outside
 * `#panel-rail` — panel-rail.js's createRail() re-places that list's rows on
 * every render, and the footer is a fixed part of the sidebar, not a panel.
 *
 * This module owns only the two facts the markup cannot state by itself:
 *
 *   - The docs target, which is per-deployment. It arrives as `docs_url` in
 *     the GET /api/panels payload (web.docs_url). The anchor ships href-less
 *     and hidden and stays that way when a deployment blanks the key: an
 *     air-gapped control room has nothing to point at, and a dead link that
 *     opens a blank tab is worse than no link at all.
 *
 *   - Who handles the Feedback click. The dialog is wired in at boot rather
 *     than here, so this module stays importable without dragging the whole
 *     modal in behind it.
 *
 * Deliberately stateless: every function reads the DOM afresh and the click
 * subscription lives on the button element, so there is no module state to
 * reset between page-level re-inits or between tests.
 */

/** Id of the Documentation anchor in index.html. */
const DOCS_LINK_ID = 'panel-docs-link';

/** Id of the Feedback button in index.html. */
const FEEDBACK_BUTTON_ID = 'panel-feedback-btn';

/**
 * Point the sidebar's Docs row at this deployment's documentation site.
 *
 * The href is the deployment's `web.docs_url` and nothing else, the row ships
 * hidden and is revealed only once it has a target, and anything that is not
 * a non-blank string leaves it hidden and href-less (an air-gapped control
 * room has nothing to point at, and a dead link that opens a blank tab is
 * worse than no link). Two-way, so a later payload moves the anchor in either
 * direction.
 *
 * Safe to call with a partial, absent or malformed payload, safe to call on a
 * page that has no sidebar (it no-ops), and safe to call again with a later
 * payload.
 *
 * @param {{docs_url?: unknown}|null|undefined} panelsPayload
 *   The `GET /api/panels` response, or any object carrying its `docs_url`.
 * @returns {void}
 */
export function initRailUtility(panelsPayload) {
  const link = document.getElementById(DOCS_LINK_ID);
  if (!(link instanceof HTMLAnchorElement)) return;

  const configured = panelsPayload == null ? undefined : panelsPayload.docs_url;
  const url = typeof configured === 'string' ? configured.trim() : '';

  if (url === '') {
    link.removeAttribute('href');
    link.hidden = true;
    return;
  }

  // setAttribute, not `.href`: the property getter resolves against the page
  // URL, so a round-trip would rewrite a relative docs path into an absolute
  // one and hide the fact that the deployment configured a relative target.
  link.setAttribute('href', url);
  link.hidden = false;
}

/**
 * Register a handler for clicks on the Feedback row.
 *
 * Any number of handlers may be registered; they run in registration order.
 * The click's default action is suppressed so the control never doubles as a
 * form submit or a navigation.
 *
 * @param {() => void} handler - invoked on each click of the Feedback button.
 * @returns {() => void} Unsubscribe. A no-op when the page has no sidebar.
 */
export function onFeedbackClick(handler) {
  const button = document.getElementById(FEEDBACK_BUTTON_ID);
  if (button == null) return () => {};

  /** @param {Event} event */
  const listener = (event) => {
    event.preventDefault();
    handler();
  };
  button.addEventListener('click', listener);
  return () => button.removeEventListener('click', listener);
}
