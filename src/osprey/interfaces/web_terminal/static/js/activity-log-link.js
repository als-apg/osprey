// @ts-check
/* OSPREY Web Terminal — link to the Session Activity Log
 *
 * The one owner of what a link to the activity log (/static/session.html)
 * carries and how the page reads it back: the terminal builds the link with
 * activityLogUrl(), the page reads it with sessionIdFromQuery(), and both sides
 * accept only a canonical session key, the grammar the server's session_key.py
 * holds (test_session_key_parity.py pins the two together).
 *
 * @module activity-log-link
 */

import { withPrefix } from './api.js';

const SESSION_KEY_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;

/**
 * Whether `value` is a canonical, bare, lowercase session key.
 * @param {unknown} value
 * @returns {value is string}
 */
export function isSessionKey(value) {
  return typeof value === 'string' && SESSION_KEY_RE.test(value);
}

/**
 * The session key a page URL's query names, or `null` when it names none or
 * names something that is not a key.
 * @param {string} search - a location.search string
 * @returns {string|null}
 */
export function sessionIdFromQuery(search) {
  const id = new URLSearchParams(search).get('session_id');
  return isSessionKey(id) ? id : null;
}

/**
 * The activity log's URL, scoped to `sessionId` when it is a session key and
 * unscoped otherwise (no parameter rather than a malformed one).
 *
 * The link never carries `?token=`: the terminal's browser already holds its
 * session cookie, and the token-exchange landing (TOKEN_EXCHANGE_PATHS) is for
 * a cookie-less arrival, not for in-app navigation.
 *
 * @param {string|null|undefined} sessionId
 * @returns {string}
 */
export function activityLogUrl(sessionId) {
  const url = withPrefix('/static/session.html');
  return isSessionKey(sessionId) ? `${url}?session_id=${sessionId}` : url;
}

/**
 * Open the activity log in a new tab on the session `getSessionId` names.
 * The id is read at click time because the card's session changes over the
 * page's life.
 * @param {() => string|null|undefined} getSessionId
 */
export function openActivityLog(getSessionId) {
  window.open(activityLogUrl(getSessionId()), '_blank', 'noopener');
}

/**
 * Bind the rail's Activity button to {@link openActivityLog}. The session
 * getter is injected, so this module reaches nothing in terminal.js. A page
 * without the button is left as it is.
 * @param {() => string|null|undefined} getSessionId
 */
export function initActivityLogButton(getSessionId) {
  const button = document.getElementById('panel-activity-btn');
  if (!button) return;
  button.addEventListener('click', (event) => {
    event.preventDefault();
    openActivityLog(getSessionId);
  });
}
