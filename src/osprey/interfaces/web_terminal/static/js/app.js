/* OSPREY Web Terminal — Application Entry Point */

import { initTerminal, fitTerminal, focusTerminal, getTerminalDimensions, pasteToTerminal } from './terminal.js';
import { onConnectionStateChange, fetchJSON } from './api.js';
import { initPanelManager } from './panel-manager.js';
import '/design-system/js/components/osprey-drawer.js';
import { initSettings } from './settings.js';
import { initMemoryGallery } from './memory-gallery.js';
import { initScaffoldGallery } from './scaffold-gallery.js';
import { initHookDebug } from './hook-debug.js';
import { initSessionSelector, startNewSession } from './sessions.js';
import { initTheme } from '/design-system/js/theme-manager.js';

document.addEventListener('DOMContentLoaded', () => {
  initTheme({ role: 'hub' });
  initTerminal('terminal-container');
  initPanelManager('right-panel');
  initSessionSelector('session-selector');
  initStatusBar();
  initResizeHandle();
  initKeyboardShortcuts();
  initNewSessionButton();
  initLogoutButton();
  initDrawerTriggerHighlight();
  initSettings();
  initMemoryGallery();
  initScaffoldGallery();
  initHookDebug();
  // Listen for paste requests from embedded iframes (gallery, ARIEL)
  initIframePasteBridge();

  // Welcome modal (once per server session)
  initWelcomeModal();
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
 */
function initLogoutButton() {
  const btn = /** @type {HTMLButtonElement} */ (document.getElementById('logout-btn'));
  if (!btn) return;

  const landingUrl = btn.dataset.landingUrl;
  if (!landingUrl) return;

  btn.addEventListener('click', () => {
    if (!isSafeLandingUrl(landingUrl)) {
      console.error('Refusing to navigate to unsafe landing_url:', landingUrl);
      return;
    }
    window.location.assign(landingUrl);
  });
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
 * web_terminal's convention for a trigger (like the settings gear) that
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

/* ---- Resize Handle ---- */

function initResizeHandle() {
  const handle = document.getElementById('resize-handle');
  const terminalPanel = /** @type {HTMLElement} */ (document.querySelector('.terminal-panel'));
  const rightPanel = /** @type {HTMLElement} */ (document.querySelector('.files-panel'));
  const container = /** @type {HTMLElement} */ (document.querySelector('.main-container'));
  const headerLeft = /** @type {HTMLElement} */ (document.querySelector('.header-left'));
  const headerRight = /** @type {HTMLElement} */ (document.querySelector('.header-right'));

  if (!handle || !terminalPanel || !rightPanel || !container) return;

  const handleWidth = 5;
  let isDragging = false;
  let startX = 0;
  let startTermWidth = 0;

  // Track the terminal's share of total width so the split scales with
  // the browser window.  null = CSS default (no user drag yet).
  /** @type {number | null} */
  let terminalRatio = null;

  function applyRatio() {
    if (terminalRatio === null) return;
    const totalWidth = container.getBoundingClientRect().width - handleWidth;
    const termWidth = Math.max(280, Math.min(totalWidth * 0.85, totalWidth * terminalRatio));
    terminalPanel.style.flex = 'none';
    terminalPanel.style.width = termWidth + 'px';
    rightPanel.style.flex = 'none';
    rightPanel.style.width = (totalWidth - termWidth) + 'px';

    // Sync header split to match the panel split
    if (headerLeft) {
      headerLeft.style.flex = 'none';
      headerLeft.style.width = termWidth + 'px';
    }
    if (headerRight) {
      headerRight.style.flex = 'none';
      headerRight.style.width = (totalWidth - termWidth) + 'px';
    }
  }

  handle.addEventListener('mousedown', (e) => {
    isDragging = true;
    startX = e.clientX;
    startTermWidth = terminalPanel.getBoundingClientRect().width;

    document.body.classList.add('resizing');
    handle.classList.add('dragging');

    e.preventDefault();
  });

  document.addEventListener('mousemove', (e) => {
    if (!isDragging) return;

    const dx = e.clientX - startX;
    const totalWidth = container.getBoundingClientRect().width - handleWidth;

    let newTermWidth = startTermWidth + dx;
    const minTerm = 280;
    const maxTerm = totalWidth * 0.85;

    newTermWidth = Math.max(minTerm, Math.min(maxTerm, newTermWidth));

    // Store as a ratio so it scales with window resize.
    terminalRatio = newTermWidth / totalWidth;
    applyRatio();

    fitTerminal();
  });

  document.addEventListener('mouseup', () => {
    if (!isDragging) return;
    isDragging = false;
    document.body.classList.remove('resizing');
    handle.classList.remove('dragging');
    fitTerminal();
  });

  // Recalculate the split when the browser window resizes.
  window.addEventListener('resize', () => {
    applyRatio();
    fitTerminal();
  });
}

/* ---- Docs Link ---- */
// Documentation link is static — no backend call needed.

/* ---- Iframe Paste Bridge ---- */

function initIframePasteBridge() {
  window.addEventListener('message', (e) => {
    if (e.origin !== window.location.origin) return;
    // Accept paste-to-terminal messages from embedded iframes
    if (e.data && e.data.type === 'osprey-paste-to-terminal' && e.data.text) {
      pasteToTerminal(e.data.text);
      focusTerminal();
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
  const gap = innerWidth - pad - leftText.length - rightText.length - pad;
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
