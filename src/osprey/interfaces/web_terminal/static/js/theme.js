/* OSPREY Web Terminal — Theme Manager
 *
 * Two themes: 'dark' (default) and 'light'.
 * Persistence: localStorage → 'dark' (no OS preference auto-detection).
 * Cross-iframe sync via postMessage.
 * xterm.js palettes stored as JS objects (xterm does NOT support CSS variables).
 */

const STORAGE_KEY = 'osprey-theme';
const VALID_THEMES = ['dark', 'light'];

// highlight.js CDN base
const HLJS_CDN = 'https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/styles';
const HLJS_THEMES = {
  dark: `${HLJS_CDN}/atom-one-dark.min.css`,
  light: `${HLJS_CDN}/atom-one-light.min.css`,
};

// xterm.js color palettes (cannot use CSS variables)
const XTERM_PALETTES = {
  dark: {
    background: '#050a10',
    foreground: '#c8d6e5',
    cursor: '#4fd1c5',
    cursorAccent: '#050a10',
    selectionBackground: 'rgba(79, 209, 197, 0.25)',
    black: '#1a2332',
    red: '#ef4444',
    green: '#22c55e',
    yellow: '#f59e0b',
    blue: '#3b82f6',
    magenta: '#a855f7',
    cyan: '#4fd1c5',
    white: '#e2e8f0',
    brightBlack: '#64748b',
    brightRed: '#f87171',
    brightGreen: '#4ade80',
    brightYellow: '#fbbf24',
    brightBlue: '#60a5fa',
    brightMagenta: '#c084fc',
    brightCyan: '#67e8f9',
    brightWhite: '#f8fafc',
  },
  light: {
    background: '#fafbfd',
    foreground: '#0c1322',
    cursor: '#0a8f8c',
    cursorAccent: '#fafbfd',
    selectionBackground: 'rgba(13, 115, 119, 0.2)',
    black: '#e4e9f0',
    red: '#dc2626',
    green: '#16a34a',
    yellow: '#d97706',
    blue: '#2563eb',
    magenta: '#9333ea',
    cyan: '#0d7377',
    white: '#1e293b',
    brightBlack: '#94a3b8',
    brightRed: '#ef4444',
    brightGreen: '#22c55e',
    brightYellow: '#f59e0b',
    brightBlue: '#3b82f6',
    brightMagenta: '#a855f7',
    brightCyan: '#0a8f8c',
    brightWhite: '#0c1322',
  },
};

// ---- Internal state ----

let _currentTheme = 'dark';
let _terminalRef = null; // set by terminal.js via setTerminalRef

// ---- Resolve theme ----

function resolveTheme() {
  const stored = localStorage.getItem(STORAGE_KEY);
  if (stored && VALID_THEMES.includes(stored)) return stored;
  return 'dark';
}

// ---- Apply theme to document ----

function applyTheme(name, { broadcast = true, transition = false } = {}) {
  if (!VALID_THEMES.includes(name)) return;
  _currentTheme = name;

  // Set data-theme attribute
  document.documentElement.setAttribute('data-theme', name);

  // Swap highlight.js stylesheet
  swapHighlightTheme(name);

  // Update toggle button icon visibility
  updateToggleIcon(name);

  // Update xterm palette if terminal is available
  if (_terminalRef) {
    _terminalRef.options.theme = XTERM_PALETTES[name];
  }

  // Broadcast to iframes
  if (broadcast) {
    broadcastTheme(name);
  }
}

// ---- Highlight.js theme swap ----

function swapHighlightTheme(name) {
  const link = document.getElementById('hljs-theme');
  if (link) {
    link.href = HLJS_THEMES[name];
  }
}

// ---- Toggle button icon ----

function updateToggleIcon(name) {
  const sunIcon = document.getElementById('theme-icon-sun');
  const moonIcon = document.getElementById('theme-icon-moon');
  const btn = document.getElementById('theme-toggle');

  if (sunIcon && moonIcon) {
    // Sun shows in dark mode (click will switch to light)
    // Moon shows in light mode (click will switch to dark)
    sunIcon.style.display = name === 'dark' ? 'block' : 'none';
    moonIcon.style.display = name === 'light' ? 'block' : 'none';
  }

  if (btn) {
    const targetTheme = name === 'dark' ? 'light' : 'dark';
    btn.setAttribute('aria-label', `Switch to ${targetTheme} theme`);
    btn.setAttribute('aria-pressed', name === 'light' ? 'true' : 'false');
  }
}

// ---- Broadcast to iframes ----

function broadcastTheme(name) {
  const iframes = document.querySelectorAll('iframe');
  for (const iframe of iframes) {
    try {
      iframe.contentWindow.postMessage(
        { type: 'osprey-theme-change', theme: name },
        '*'
      );
    } catch {
      // cross-origin — expected for some iframes
    }
  }

  // Grafana special case: update iframe src URL with theme param
  const monitoringIframes = document.querySelectorAll('.panel-iframe[src*="grafana"], .panel-iframe[src*="monitoring"]');
  for (const iframe of monitoringIframes) {
    try {
      const url = new URL(iframe.src);
      const grafanaTheme = name === 'light' ? 'light' : 'dark';
      if (url.searchParams.get('theme') !== grafanaTheme) {
        url.searchParams.set('theme', grafanaTheme);
        iframe.src = url.toString();
      }
    } catch {
      // ignore URL parse errors
    }
  }
}

// ---- Theme transition animation ----

function withTransition(fn) {
  document.documentElement.classList.add('theme-transitioning');
  fn();
  setTimeout(() => {
    document.documentElement.classList.remove('theme-transitioning');
  }, 300);
}

// ---- Public API ----

/**
 * Initialize theme system. Call once in DOMContentLoaded, before initTerminal.
 * Sets data-theme attribute and configures hljs link element.
 */
export function initTheme() {
  _currentTheme = resolveTheme();
  applyTheme(_currentTheme, { broadcast: false, transition: false });

  // Wire toggle button
  const btn = document.getElementById('theme-toggle');
  if (btn) {
    btn.addEventListener('click', toggleTheme);
  }
}

/**
 * Get current theme name.
 */
export function getTheme() {
  return _currentTheme;
}

/**
 * Toggle between dark and light themes.
 */
export function toggleTheme() {
  const next = _currentTheme === 'dark' ? 'light' : 'dark';
  setTheme(next);
}

/**
 * Set a specific theme by name.
 */
export function setTheme(name) {
  if (!VALID_THEMES.includes(name)) return;
  localStorage.setItem(STORAGE_KEY, name);
  withTransition(() => applyTheme(name));
}

/**
 * Apply a theme received via postMessage (no re-broadcast, no persist).
 */
export function applyReceivedTheme(name) {
  if (!VALID_THEMES.includes(name)) return;
  _currentTheme = name;
  applyTheme(name, { broadcast: false });
}

/**
 * Get xterm.js palette for the given (or current) theme.
 */
export function getXtermPalette(name) {
  return XTERM_PALETTES[name || _currentTheme];
}

/**
 * Set the xterm Terminal reference for live palette switching.
 * Called by terminal.js after creating the Terminal instance.
 */
export function setTerminalRef(termInstance) {
  _terminalRef = termInstance;
}
