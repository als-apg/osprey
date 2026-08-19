/* OSPREY Web Terminal — Application Entry Point */

import { initTerminal, focusTerminal, getTerminalDimensions, pasteToTerminal, clearStoredSessionId } from './terminal.js';
import { onConnectionStateChange, fetchJSON, withPrefix } from './api.js';
import { initPanelManager, broadcastMode, handleUiModeFlip, navigateAndActivatePanel } from './panel-manager.js';
import '/design-system/js/components/osprey-drawer.js';
import { initSettings } from './settings.js';
import { initMemoryGallery } from './memory-gallery.js';
import { initScaffoldGallery } from './scaffold-gallery.js';
import { initHookDebug } from './hook-debug.js';
import { initSessionSelector, startNewSession } from './sessions.js';
import { initCommandPalette } from './palette-boot.js';
import { getFamily, initTheme, subscribe as subscribeTheme } from '/design-system/js/theme-manager.js';
import { stripQueryMode } from '/design-system/js/frame-params.js';
import { initChat } from './chat.js';
import { initDockWorkspace, applyDockMode } from './dock-workspace.js';
import { initHeaderContrib } from './tile-header-contrib.js';
import { initDisplayMenu } from './display-menu.js';
import { followThemeFamily, getRailPosition, setRailPosition } from './rail-position.js';

document.addEventListener('DOMContentLoaded', () => {
  initTheme({ role: 'hub' });
  // Welcome modal FIRST: #welcome-overlay ships in the DOM from first paint
  // (opaque, full-viewport, z-index 10000) and only this init wires its
  // dismiss controls. Any earlier throw would leave the page permanently
  // covered with no way out — so the overlay must be dismissable before any
  // fallible init runs. Self-contained: needs only the DOM and /health.
  void initWelcomeModal();
  // Guarded: xterm.js loads from a CDN by default (local only in
  // OSPREY_OFFLINE), so a network blip must degrade the terminal card, not
  // kill the whole boot.
  try {
    initTerminal('terminal-container');
  } catch (err) {
    console.error('Failed to init terminal:', err);
  }
  // Simple-mode operator chat. Guarded so a chat init failure can't break the
  // rest of the boot (the expert terminal is already up at this point).
  try {
    initChat('operator-container');
  } catch (err) {
    console.error('Failed to init operator chat:', err);
  }
  // Before the panel manager creates any iframe, so no contribution can
  // arrive without a listener.
  initHeaderContrib();
  initPanelManager('right-panel');
  initSessionSelector('session-selector');
  initStatusBar();
  // Dock the terminal + workspace panels into the dockview shell (replaces the
  // old fixed resize-handle split). Guarded so a dock init failure can't break
  // the rest of boot — the source subtrees stay in the page if it no-ops.
  try {
    initDockWorkspace();
  } catch (err) {
    console.error('Failed to init dock workspace:', err);
  }
  initKeyboardShortcuts();
  initCommandPalette();
  initNewSessionButton();
  initLogoutButton();
  initModeToggle();
  initDisplayMenu();
  initRailPosition();
  initDrawerTriggerHighlight();
  initSettings();
  initMemoryGallery();
  initScaffoldGallery();
  initHookDebug();
  // Listen for paste requests from embedded iframes (gallery, ARIEL)
  initIframePasteBridge();
});

/* ---- New Session Button ---- */

function initNewSessionButton() {
  const btn = /** @type {HTMLButtonElement} */ (document.getElementById('new-session-btn'));
  if (!btn) return;

  btn.addEventListener('click', async () => {
    btn.disabled = true;
    try {
      await startNewSession();
    } catch (err) {
      console.error('Failed to start new session:', err);
    } finally {
      btn.disabled = false;
    }
  });
}

/* ---- Logout Button ---- */

/**
 * Only present in the DOM when the server rendered a non-empty `landing_url`
 * (multi-user deployments). Plain `osprey web` never emits the button, so
 * this is a no-op there.
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
 * The click handler locks the button (`disabled` + `aria-busy`) once a safe
 * logout is under way: `disabled` stops a second POST, and `aria-busy`
 * announces the in-flight state to assistive tech. Neither is reset — every
 * path out of the handler navigates away, unloading the page. The unsafe
 * `landing_url` guard returns before the lock, leaving the button usable.
 *
 * Exported for testability (see app-logout.test.mjs) — the module's
 * DOMContentLoaded bootstrap never fires the button wiring on its own once
 * that event has already passed, e.g. in a test environment.
 */
export function initLogoutButton() {
  const btn = /** @type {HTMLButtonElement} */ (document.getElementById('logout-btn'));
  if (!btn) return;

  const landingUrl = btn.dataset.landingUrl;
  if (!landingUrl) return;

  btn.addEventListener('click', async () => {
    if (!isSafeLandingUrl(landingUrl)) {
      console.error('Refusing to navigate to unsafe landing_url:', landingUrl);
      return;
    }
    btn.disabled = true;
    btn.setAttribute('aria-busy', 'true');
    try {
      await fetch(withPrefix('/api/terminal/logout'), { method: 'POST' });
    } catch (err) {
      console.error('Logout request failed:', err);
    }
    await endAuthSession();
    clearStoredSessionId();
    window.location.assign(landingUrl);
  });
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
 * other request this file makes: the sidecar's public surface is mounted at the
 * origin root (nginx's `location /auth/`), so a prefixed URL would land on this
 * container instead and 404. One `user` parameter, encoded — the route refuses
 * a repeated one rather than picking a side.
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

/* ---- UI Mode Toggle (Expert / Simple) ---- */

/**
 * Wire the Expert/Simple toggle (the View row inside the header display-menu
 * popover — see display-menu.js). The active segment is styled
 * purely off html[data-ui-mode] (see terminal.css), and mode-boot.js already
 * resolved the initial mode pre-paint; this only handles the runtime flip:
 * swap the attribute, persist the explicit choice, drop a leftover one-shot
 * ?mode= (mirrors theme-manager's _stripQueryTheme), and broadcast to panels.
 */
function initModeToggle() {
  const STORAGE_KEY = 'osprey-ui-mode';
  const toggle = document.getElementById('mode-toggle');
  if (!toggle) return;

  toggle.addEventListener('click', (e) => {
    const target = e.target instanceof Element ? e.target.closest('.mode-segment') : null;
    if (!(target instanceof HTMLElement)) return;
    const mode = target.dataset.mode;
    if (mode !== 'expert' && mode !== 'simple') return;

    document.documentElement.setAttribute('data-ui-mode', mode);
    try {
      localStorage.setItem(STORAGE_KEY, mode);
    } catch { /* storage blocked — mode still applies for this session */ }
    stripQueryMode();
    // Panels read the current mode off <html>, so broadcast only after the swap.
    broadcastMode();
    // Dock half of the flip: stash+lock into the simple layout, or reconcile+
    // restore the expert layout. Runs after the CSS/attribute swap so the dock
    // reads the target mode; no-ops until the workspace shell exists.
    applyDockMode(mode);
    // Panel half: a flip to expert ends the simple-UX chat-only suppression
    // and lets the default panel claim a still-empty workspace slot. Runs
    // after applyDockMode so the activation docks into the target layout.
    handleUiModeFlip(mode);
  });
}

/**
 * Adopt a one-shot `?rail=` as an explicit choice. rail-boot.js already
 * stamped the attribute pre-paint; re-setting it through setRailPosition
 * persists it and strips the param, so a reload (or a stale bookmark of the
 * bare URL) keeps the arrangement the link put the operator in. No `?rail=`
 * means nothing to adopt — the boot-resolved position already IS the stored
 * or configured one.
 */
function initRailPosition() {
  // Follow the theme family for as long as neither the operator nor the
  // deployment has pinned a rail of their own: picking Retro hands back the
  // pre-redesign look, and the horizontal tab strip is part of that look.
  // followThemeFamily() decides whether the move is allowed; this only tells
  // it which family is now active, on every apply (init included).
  subscribeTheme(() => followThemeFamily(getFamily()));

  try {
    if (!new URLSearchParams(window.location.search).has('rail')) return;
  } catch {
    return;
  }
  setRailPosition(getRailPosition());
}

/**
 * `landing_url` comes from operator config, not user input, but it's still a
 * live navigation sink — reject anything that isn't a same-origin relative
 * path or an http(s) URL so a misconfigured value can't smuggle a
 * `javascript:`/`data:` scheme into the page origin. A leading "//" is
 * excluded from the relative-path case too: browsers resolve it as
 * protocol-relative (same scheme, attacker-controlled host), so a bare
 * `startsWith('/')` check would still let it through.
 */
function isSafeLandingUrl(/** @type {string} */ url) {
  if (url.startsWith('/') && !url.startsWith('//')) return true;
  try {
    const parsed = new URL(url, window.location.origin);
    return parsed.protocol === 'http:' || parsed.protocol === 'https:';
  } catch {
    return false;
  }
}

/* ---- Drawer Trigger Highlight ---- */

/**
 * osprey-drawer doesn't manage its `[data-drawer]` trigger's `.active` state
 * itself (a page-level nicety, not part of the component's contract — see
 * its module docstring). Web_terminal owns its own triggers, so it wires
 * this via the `drawer:open`/`drawer:close` events the component dispatches
 * (bubbling) on the host, matching any trigger for that drawer id — either
 * the component's own `[data-drawer]` marker, or `[data-drawer-trigger]`,
 * web_terminal's convention for a trigger (like the display menu's System
 * Settings row) that
 * needs its own gating logic before opening and so must never match the
 * component's delegated `[data-drawer]` handler. Either way the highlight
 * stays in sync.
 */
function initDrawerTriggerHighlight() {
  const setActive = (/** @type {boolean} */ active) => (/** @type {Event} */ event) => {
    const drawer = event.target;
    if (!(drawer instanceof HTMLElement) || !drawer.id) return;
    document
      .querySelectorAll(`[data-drawer="${drawer.id}"], [data-drawer-trigger="${drawer.id}"]`)
      .forEach((btn) => btn.classList.toggle('active', active));
  };
  document.addEventListener('drawer:open', setActive(true));
  document.addEventListener('drawer:close', setActive(false));
}

/* ---- Status Bar ---- */

function initStatusBar() {
  const wsDot = document.getElementById('ws-dot');
  const dimsEl = document.getElementById('term-dims');

  onConnectionStateChange(({ ws }) => {
    if (wsDot) {
      wsDot.className = 'status-dot' + (ws === 'connected' ? ' live' : ws === 'disconnected' ? ' error' : '');
    }
  });

  // Update terminal dimensions display
  setInterval(() => {
    const dims = getTerminalDimensions();
    if (dims && dimsEl) {
      dimsEl.textContent = `${dims.cols}\u00D7${dims.rows}`;
    }
  }, 500);

  // Live clock
  const clockEl = document.getElementById('status-clock');
  if (clockEl) {
    setInterval(() => {
      clockEl.textContent = new Date().toLocaleTimeString('en-US', { hour12: false });
    }, 1000);
  }
}

/* ---- Iframe Paste Bridge ---- */

function initIframePasteBridge() {
  window.addEventListener('message', (e) => {
    if (e.origin !== window.location.origin) return;
    // Accept paste-to-terminal messages from embedded iframes
    if (e.data && e.data.type === 'osprey-paste-to-terminal' && e.data.text) {
      pasteToTerminal(e.data.text);
      focusTerminal();
    }
    // A panel asking its host to move THIS client to another panel — the
    // sender-local twin of the panel_focus SSE path (the gallery's logbook
    // submit is the first caller). Deliberately not a server broadcast: a
    // human gesture in one browser must not move anyone else's workspace,
    // and it gets a plain activation with no agent attribution.
    //
    // The url must be root-relative and NOT protocol-relative. The origin
    // check above is necessary but not sufficient: a same-origin sender can
    // still be an agent-authored artifact rendered in a sandboxed panel, and
    // this url reaches an iframe src via buildEmbedSrc, which preserves
    // whatever scheme it is handed. `javascript:alert(1)` survives it intact
    // and would execute in the HOST origin, and `//evil.example/x` resolves
    // to a cross-origin document — so a leading-slash test alone is a hole.
    // Every real panel url is root-relative and already server-prefixed.
    if (e.data && e.data.type === 'osprey:navigate'
        && typeof e.data.panel === 'string' && typeof e.data.url === 'string'
        && e.data.url.startsWith('/') && !e.data.url.startsWith('//')) {
      navigateAndActivatePanel(e.data.panel, e.data.url);
    }
  });

  // Drop zone: accept dragged artifacts onto the terminal container
  const termContainer = document.getElementById('terminal-container');
  if (termContainer) {
    termContainer.addEventListener('dragover', (e) => {
      e.preventDefault();
      /** @type {DataTransfer} */ (e.dataTransfer).dropEffect = 'copy';
    });
    termContainer.addEventListener('drop', (e) => {
      e.preventDefault();
      const text = /** @type {DataTransfer} */ (e.dataTransfer).getData('text/plain');
      if (text) {
        pasteToTerminal(text);
        focusTerminal();
      }
    });
  }
}

/* ---- Welcome Modal (terminal banner) ---- */

async function initWelcomeModal() {
  const overlay = document.getElementById('welcome-overlay');
  if (!overlay) return;

  // Check server session ID — show modal once per server instance
  const STORAGE_KEY = 'osprey-server-session';
  let version = '';
  try {
    const health = await fetchJSON('/health');
    const serverSession = health.session_id;
    version = health.version || '';
    if (serverSession && localStorage.getItem(STORAGE_KEY) === serverSession) {
      overlay.remove();
      focusTerminal();
      return;
    }
  } catch {
    // Health endpoint unreachable — show modal to be safe
  }

  const pre = document.getElementById('welcome-ascii');
  const btn = document.getElementById('welcome-dismiss');
  if (!pre || !btn) return;

  // Build subtitle: "Web Terminal" left, version right (58 chars inner width)
  const leftText = 'Web Terminal';
  const rightText = version ? `v${version}` : '';
  const innerWidth = 58; // matches box width (no Unicode offset needed — plain text line)
  const pad = 4; // padding from box edges
  // Clamped: dev builds carry long post-release versions (`v2026.6.2.post1058+g….d…`)
  // that overflow the box; a negative gap would make repeat() throw and leave the
  // overlay undismissable. A too-wide line just overhangs the border harmlessly.
  const gap = Math.max(1, innerWidth - pad - leftText.length - rightText.length - pad);
  const versionLine = '    ║' + ' '.repeat(pad) + leftText + ' '.repeat(gap) + rightText + ' '.repeat(pad) + '║';

  // ASCII banner — uses the original OSPREY CLI banner art
  const lines = [
    '    ╔══════════════════════════════════════════════════════════╗',
    '    ║                                                          ║',
    '    ║                                                          ║',
    '    ║    ░█████╗░░██████╗██████╗░██████╗░███████╗██╗░░░██╗     ║',
    '    ║    ██╔══██╗██╔════╝██╔══██╗██╔══██╗██╔════╝╚██╗░██╔╝     ║',
    '    ║    ██║░░██║╚█████╗░██████╔╝██████╔╝█████╗░░░╚████╔╝░     ║',
    '    ║    ██║░░██║░╚═══██╗██╔═══╝░██╔══██╗██╔══╝░░░░╚██╔╝░░     ║',
    '    ║    ╚█████╔╝██████╔╝██║░░░░░██║░░██║███████╗░░░██║░░░     ║',
    '    ║    ░╚════╝░╚═════╝░╚═╝░░░░░╚═╝░░╚═╝╚══════╝░░░╚═╝░░░     ║',
    '    ║                                                          ║',
    versionLine,
    '    ╚══════════════════════════════════════════════════════════╝',
    '',
    '        Experimental system. Proceed with caution.',
    '',
  ];

  // Reveal lines one by one with staggered delay
  const lineDelay = 35; // ms between lines
  lines.forEach((line, i) => {
    const span = document.createElement('span');
    span.className = 'wl';
    span.style.animationDelay = (i * lineDelay) + 'ms';

    // Box content lines (║...║): split so the right border is pinned via flex
    const trimmed = line.trimEnd();
    if (trimmed.startsWith('    ║') && trimmed.endsWith('║') && !trimmed.startsWith('    ╔') && !trimmed.startsWith('    ╚')) {
      span.classList.add('wl-box');
      const lastBar = trimmed.lastIndexOf('║');
      const left = document.createElement('span');
      left.textContent = trimmed.substring(0, lastBar);
      const right = document.createElement('span');
      right.textContent = '║';
      span.appendChild(left);
      span.appendChild(right);
      span.appendChild(document.createTextNode('\n'));
    } else {
      span.textContent = line + '\n';
    }

    pre.appendChild(span);
  });

  // Show the safety link + prompt after all lines have appeared
  const safetyLink = document.getElementById('welcome-safety-link');
  const promptDelay = lines.length * lineDelay + 200;
  setTimeout(() => {
    if (safetyLink) safetyLink.style.visibility = 'visible';
    btn.style.visibility = 'visible';
  }, promptDelay);

  // Safety link always points to the local safety guidelines page

  // Dismiss handlers
  const dismiss = async () => {
    // Store current server session ID so modal won't show again until restart
    try {
      const health = await fetchJSON('/health');
      if (health.session_id) {
        localStorage.setItem(STORAGE_KEY, health.session_id);
      }
    } catch { /* best effort */ }
    overlay.classList.add('hidden');
    setTimeout(() => {
      overlay.remove();
      focusTerminal();
    }, 500);
  };

  btn.addEventListener('click', dismiss);

  // Also dismiss on Enter key
  document.addEventListener('keydown', function handler(e) {
    if (e.key === 'Enter' && overlay.parentNode) {
      e.preventDefault();
      document.removeEventListener('keydown', handler);
      dismiss();
    }
  });
}

/* ---- Keyboard Shortcuts ---- */

function initKeyboardShortcuts() {
  document.addEventListener('keydown', (e) => {
    // Ctrl+` — focus terminal
    if (e.ctrlKey && e.key === '`') {
      e.preventDefault();
      focusTerminal();
    }
  });
}
