/* OSPREY Web Terminal — Application Entry Point */

import { initTerminal, fitTerminal, focusTerminal, getTerminalDimensions, stopTerminal, startTerminal, restartTerminal, pasteToTerminal } from './terminal.js';
import { onConnectionStateChange, fetchJSON } from './api.js';
import { initPanelManager } from './panel-manager.js';
import { initDrawers } from './drawer.js';
import { initSettings } from './settings.js';
import { initMemoryTab } from './memory-tab.js';
import { initPromptsGallery } from './prompts-gallery.js';
import { initSessionSelector, startNewSession } from './sessions.js';
import { initTheme } from './theme.js';

let currentMode = 'terminal';
let cuiIframeCreated = false;

document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initTerminal('terminal-container');
  initPanelManager('right-panel');
  initSessionSelector('session-selector');
  initStatusBar();
  initResizeHandle();
  initKeyboardShortcuts();
  initModeToggle();
  initNewSessionButton();
  initDrawers();
  initSettings();
  initMemoryTab();
  initPromptsGallery();
  initWikiLink();

  // Listen for paste requests from embedded iframes (gallery, ARIEL)
  initIframePasteBridge();

  // Welcome modal (once per server session)
  initWelcomeModal();
});

/* ---- Mode Toggle ---- */

function initModeToggle() {
  const btn = document.getElementById('mode-toggle');
  if (!btn) return;

  btn.addEventListener('click', () => {
    if (currentMode === 'terminal') {
      switchToCUIMode();
    } else {
      switchToTerminalMode();
    }
  });
}

async function switchToCUIMode() {
  currentMode = 'cui';
  stopTerminal();

  const termContainer = document.getElementById('terminal-container');
  const cuiContainer = document.getElementById('cui-container');
  const toggleBtn = document.getElementById('mode-toggle');
  const label = document.getElementById('terminal-label');
  const cuiStatus = document.getElementById('cui-status');

  if (termContainer) termContainer.style.display = 'none';
  if (cuiContainer) cuiContainer.style.display = '';
  if (toggleBtn) toggleBtn.textContent = 'Terminal';
  if (label) label.textContent = 'CUI';
  if (cuiStatus) cuiStatus.style.display = '';

  // Immediate health LED update
  updateCUIHealth();

  // Lazy-create iframe (created once, cached to preserve conversation state)
  if (!cuiIframeCreated && cuiContainer) {
    // Show loading message while we check
    const loading = document.createElement('div');
    loading.className = 'cui-unavailable';
    loading.textContent = 'Connecting to CUI server\u2026';
    cuiContainer.appendChild(loading);

    try {
      // Poll until the CUI server is ready (npx may need to download first)
      let data = null;
      for (let i = 0; i < 30; i++) {
        data = await fetchJSON('/api/cui-server');
        if (data.available) break;
        await new Promise(r => setTimeout(r, 1000));
      }

      cuiContainer.removeChild(loading);

      if (data && data.available) {
        // Use same-origin proxy (/cui/) which constrains sessions to this
        // project and eliminates cross-origin auth issues.
        //
        // Set the CUI auth cookie directly (same-origin, so it's shared
        // with the iframe).  This avoids a React timing issue in CUI where
        // useAuth() runs in useEffect (after first render) but getAuthToken()
        // reads the cookie during render — the fragment-based approach left
        // the cookie unset on the first render, showing the login page.
        const proxyBase = '/cui/';
        if (data.authToken) {
          const expires = new Date();
          expires.setDate(expires.getDate() + 7);
          document.cookie = `cui-auth-token=${encodeURIComponent(data.authToken)}; expires=${expires.toUTCString()}; path=/; SameSite=Strict`;
        }
        const iframe = document.createElement('iframe');
        iframe.className = 'cui-iframe';
        iframe.src = proxyBase;
        cuiContainer.appendChild(iframe);
        updateCUIHealth();
      } else {
        const msg = document.createElement('div');
        msg.className = 'cui-unavailable';
        msg.textContent = 'CUI server not available — install with: npm install -g cui-server';
        cuiContainer.appendChild(msg);
      }
    } catch (err) {
      console.error('Failed to load CUI server config:', err);
      if (loading.parentNode) cuiContainer.removeChild(loading);
      const msg = document.createElement('div');
      msg.className = 'cui-unavailable';
      msg.textContent = 'CUI server not available';
      cuiContainer.appendChild(msg);
    }
    cuiIframeCreated = true;
  }
}

async function updateCUIHealth() {
  const cuiDot = document.getElementById('cui-dot');
  if (!cuiDot) return;
  try {
    const data = await fetchJSON('/api/cui-server');
    cuiDot.className = 'status-dot' + (data.available ? ' live' : ' error');
  } catch {
    cuiDot.className = 'status-dot error';
  }
}

function switchToTerminalMode() {
  currentMode = 'terminal';

  const termContainer = document.getElementById('terminal-container');
  const cuiContainer = document.getElementById('cui-container');
  const toggleBtn = document.getElementById('mode-toggle');
  const label = document.getElementById('terminal-label');
  const cuiStatus = document.getElementById('cui-status');

  if (cuiContainer) cuiContainer.style.display = 'none';
  if (termContainer) termContainer.style.display = '';
  if (toggleBtn) toggleBtn.textContent = 'CUI';
  if (label) label.textContent = 'Session';
  if (cuiStatus) cuiStatus.style.display = 'none';

  startTerminal();
  fitTerminal();
  focusTerminal();
}

/* ---- New Session Button ---- */

function initNewSessionButton() {
  const btn = document.getElementById('new-session-btn');
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

/* ---- Status Bar ---- */

function initStatusBar() {
  const wsDot = document.getElementById('ws-dot');
  const dimsEl = document.getElementById('term-dims');

  onConnectionStateChange(({ ws }) => {
    if (wsDot) {
      wsDot.className = 'status-dot' + (ws === 'connected' ? ' live' : ws === 'disconnected' ? ' error' : '');
    }
  });

  // CUI health poll (every 10s when CUI status is visible)
  setInterval(() => {
    const cuiStatus = document.getElementById('cui-status');
    if (!cuiStatus || cuiStatus.style.display === 'none') return;
    updateCUIHealth();
  }, 10000);

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
  const terminalPanel = document.querySelector('.terminal-panel');
  const rightPanel = document.querySelector('.files-panel');
  const container = document.querySelector('.main-container');
  const headerLeft = document.querySelector('.header-left');
  const headerRight = document.querySelector('.header-right');

  if (!handle || !terminalPanel || !rightPanel || !container) return;

  const handleWidth = 5;
  let isDragging = false;
  let startX = 0;
  let startTermWidth = 0;

  // Track the terminal's share of total width so the split scales with
  // the browser window.  null = CSS default (no user drag yet).
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

/* ---- Wiki Link ---- */

async function initWikiLink() {
  const link = document.getElementById('wiki-link');
  if (!link) return;
  try {
    const data = await fetchJSON('/api/wiki-url');
    if (data.available && data.url) {
      link.href = data.url;
      link.style.display = '';
    }
  } catch {
    // Wiki not configured — button stays hidden
  }
}

/* ---- Iframe Paste Bridge ---- */

function initIframePasteBridge() {
  window.addEventListener('message', (e) => {
    // Accept paste-to-terminal messages from embedded iframes
    if (e.data && e.data.type === 'osprey-paste-to-terminal' && e.data.text) {
      pasteToTerminal(e.data.text);
      focusTerminal();
    }
  });
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

  // Point safety link at wiki if available, otherwise keep default
  if (safetyLink) {
    fetchJSON('/api/wiki-url').then(data => {
      if (data.available && data.url) safetyLink.href = data.url;
    }).catch(() => {});
  }

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
    // Ctrl+` — focus active mode input
    if (e.ctrlKey && e.key === '`') {
      e.preventDefault();
      if (currentMode === 'terminal') {
        focusTerminal();
      }
      // CUI mode: no-op (can't focus cross-origin iframe)
    }
  });
}
