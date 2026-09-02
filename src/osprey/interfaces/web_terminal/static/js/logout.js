/**
 * Logging out, from anywhere on the page.
 *
 * Log out is offered in three places — the header identity menu, the display
 * menu's action row, and the command palette — and none of them owns the
 * flow. The two buttons are bar items an operator may remove; the palette is
 * not, so it must be able to log out with neither button in the DOM. What it
 * needs is the landing URL, which the server stamps on `<html>` as
 * `data-landing-url` whenever it renders a logout at all (multi-user
 * deployments: a user to log out, somewhere to send them). Plain `osprey web`
 * stamps nothing and offers nothing.
 *
 * Real logout, in order: (1) POST the server logout route — prefix-aware via
 * `window.__OSPREY_PREFIX__` so it reaches this container under `/u/<user>/`
 * — which empties the PTY + operator registries (routes/websocket.py's
 * `logout_terminal`); (2) end the auth session too, if the deployment has one
 * (`endAuthSession`); (3) clear the client's own stored PTY session id
 * (`clearStoredSessionId`, terminal.js) so a fresh page load's
 * `initTerminal()` finds nothing to auto-resume; (4) only then navigate to
 * the landing page. A failed logout request still clears the local pointer
 * and navigates — the client's own record of "my session" is what matters
 * for this browser, and getting stuck on the page helps no one.
 *
 * Given a button, the flow locks it (`disabled` + `aria-busy`) once a safe
 * logout is under way: `disabled` stops a second POST, and `aria-busy`
 * announces the in-flight state to assistive tech. Neither is reset — every
 * path out navigates away, unloading the page. The unsafe `landing_url` guard
 * returns before the lock, leaving the button usable.
 */

import { withPrefix } from './api.js';
import { clearStoredSessionId } from './terminal.js';

/**
 * Log out and leave for the landing page.
 *
 * @param {string} landingUrl - where to send the operator afterwards
 * @param {HTMLButtonElement | null} [btn] - the control that was clicked, if
 *   one was, so the in-flight lock lands on what the operator is looking at
 * @returns {Promise<void>}
 */
export async function logout(landingUrl, btn = null) {
  if (!isSafeLandingUrl(landingUrl)) {
    console.error('Refusing to navigate to unsafe landing_url:', landingUrl);
    return;
  }
  if (btn) {
    btn.disabled = true;
    btn.setAttribute('aria-busy', 'true');
  }
  try {
    await fetch(withPrefix('/api/terminal/logout'), { method: 'POST' });
  } catch (err) {
    console.error('Logout request failed:', err);
  }
  await endAuthSession();
  clearStoredSessionId();
  window.location.assign(landingUrl);
}

/**
 * The roster username this container serves, from the server-rendered URL
 * prefix (`window.__OSPREY_PREFIX__`, which `compute_url_prefix()` sets to
 * exactly `/u/<user>` for a multi-user container and to `""` otherwise).
 *
 * Read from the prefix rather than from the display menu's identity line
 * because the prefix is the copy the app already routes every one of its own
 * requests through — that line is display markup, and taking a name from
 * rendered text to put it back in a URL is how a display change becomes a
 * wiring bug.
 * Returns `""` for a plain `osprey web`, which has no per-user prefix.
 */
function terminalUserFromPrefix() {
  const prefix = (window.__OSPREY_PREFIX__ || '').replace(/\/+$/, '');
  return prefix.startsWith('/u/') ? prefix.slice('/u/'.length) : '';
}

/**
 * End the auth sidecar's session for this container's user — best effort, and
 * cosmetic. The app holds no signing secret and decides nothing: the sidecar
 * revokes the session id and reissues the cookie without this user, and nginx
 * enforces the result. Skipping this step (or failing it) costs the operator a
 * session that outlives their terminal by up to `auth.session_lifetime`; it can
 * never grant access.
 *
 * `/auth/logout` is deliberately NOT run through `withPrefix()`, unlike every
 * other request this module makes: the sidecar's public surface is mounted at
 * the origin root (nginx's `location /auth/`), so a prefixed URL would land on
 * this container instead and 404. One `user` parameter, encoded — the route
 * refuses a repeated one rather than picking a side.
 *
 * A `fetch` rather than a navigation, which is what makes this safe to run
 * unconditionally. `location /auth/` exists only when `auth.method != "none"`,
 * so navigating there would strand every existing no-auth multi-user
 * deployment on a 404 instead of the landing page; the app cannot tell the two
 * postures apart, because keeping `OSPREY_AUTH_*` out of these containers is
 * the isolation the feature is built on. As a fetch, the sidecar's `Set-Cookie`
 * still reaches the jar when auth is on, and a 404/502 is simply ignored when
 * it is not — the caller navigates to the landing page either way.
 */
async function endAuthSession() {
  const user = terminalUserFromPrefix();
  if (!user) return;
  try {
    // The response is deliberately not inspected: a 404 is the *expected*
    // answer in a deployment with authentication off, so treating a non-ok
    // status as an error would log one on every logout that is working fine.
    await fetch(`/auth/logout?user=${encodeURIComponent(user)}`, {
      credentials: 'same-origin',
      cache: 'no-store',
    });
  } catch (err) {
    console.error('Auth logout request failed:', err);
  }
}

/**
 * `landing_url` comes from operator config, not user input, but it's still a
 * live navigation sink — reject anything that isn't a same-origin relative
 * path or an http(s) URL so a misconfigured value can't smuggle a
 * `javascript:`/`data:` scheme into the page origin. A leading "//" is
 * excluded from the relative-path case too: browsers resolve it as
 * protocol-relative (same scheme, attacker-controlled host), so a bare
 * `startsWith('/')` check would still let it through.
 *
 * @param {string} url
 * @returns {boolean}
 */
export function isSafeLandingUrl(url) {
  if (url.startsWith('/') && !url.startsWith('//')) return true;
  try {
    const parsed = new URL(url, window.location.origin);
    return parsed.protocol === 'http:' || parsed.protocol === 'https:';
  } catch {
    return false;
  }
}
