# Recipe 5: Frontend Conventions

## When You Need This

You're building the browser-based UI for your tool. OSPREY uses vanilla JavaScript with ES6 modules — no React, no Vue, no build step.

## Design System: "Precision Instrumentation"

All OSPREY interfaces share a dark-mode design language inspired by scientific control room instruments. Your interface should look like it belongs in the same family.

### Color Palette

```css
:root {
    /* Core backgrounds */
    --bg-primary: #0a0f1a;          /* Deepest background */
    --bg-secondary: #111827;        /* Cards, panels */
    --bg-tertiary: #1a2332;         /* Elevated surfaces */
    --bg-quaternary: #243044;       /* Hover states */

    /* Text */
    --text-primary: #e8edf5;        /* Primary content */
    --text-secondary: #94a3b8;      /* Labels, metadata */
    --text-tertiary: #64748b;       /* Disabled, hints */

    /* Accent colors */
    --accent-primary: #319795;      /* Teal — actions, links, active states */
    --accent-secondary: #2b6cb0;    /* Blue — secondary actions */
    --accent-amber: #d4a574;        /* Amber — highlights, IDs, important data */

    /* Status */
    --status-success: #48bb78;      /* Green */
    --status-warning: #ed8936;      /* Orange */
    --status-error: #fc8181;        /* Red */
    --status-info: #63b3ed;         /* Blue */

    /* Borders */
    --border-subtle: rgba(255, 255, 255, 0.06);
    --border-default: rgba(255, 255, 255, 0.1);
    --border-accent: rgba(49, 151, 149, 0.3);

    /* Spacing scale */
    --space-xs: 4px;
    --space-sm: 8px;
    --space-md: 16px;
    --space-lg: 24px;
    --space-xl: 32px;

    /* Typography */
    --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    --font-mono: "SF Mono", "Fira Code", "Consolas", monospace;
    --font-size-sm: 0.8125rem;     /* 13px — metadata, labels */
    --font-size-base: 0.875rem;    /* 14px — body text */
    --font-size-lg: 1rem;          /* 16px — headings */

    /* Shadows (layered depth) */
    --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.3);
    --shadow-md: 0 4px 6px rgba(0, 0, 0, 0.3), 0 1px 3px rgba(0, 0, 0, 0.2);
    --shadow-lg: 0 10px 15px rgba(0, 0, 0, 0.4), 0 4px 6px rgba(0, 0, 0, 0.2);

    /* Borders radius */
    --radius-sm: 4px;
    --radius-md: 8px;
    --radius-lg: 12px;
}
```

### CSS File Organization

```
css/
├── variables.css      # Design tokens (copy from above, customize accents)
├── base.css           # Reset, body, typography, scrollbars, animations, utilities
├── layout.css         # Header, main content, grid, responsive breakpoints
├── components.css     # Buttons, inputs, cards, badges, tags, tables
├── drawer.css         # Slide-in panel (settings, detail views) — REUSABLE
└── {feature}.css      # Feature-specific styles (optional)
```

**Loading order in `index.html`:**

```html
<link rel="stylesheet" href="/static/css/variables.css">
<link rel="stylesheet" href="/static/css/base.css">
<link rel="stylesheet" href="/static/css/layout.css">
<link rel="stylesheet" href="/static/css/components.css">
```

### Component Styling Patterns

**Buttons:**

```css
.btn {
    padding: var(--space-sm) var(--space-md);
    border-radius: var(--radius-sm);
    font-size: var(--font-size-sm);
    font-weight: 500;
    cursor: pointer;
    transition: all 150ms ease;
    border: 1px solid var(--border-default);
    background: var(--bg-tertiary);
    color: var(--text-primary);
}

.btn:hover {
    background: var(--bg-quaternary);
    border-color: var(--border-accent);
}

.btn-primary {
    background: var(--accent-primary);
    border-color: var(--accent-primary);
    color: white;
}
```

**Cards:**

```css
.card {
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: var(--space-lg);
    box-shadow: var(--shadow-sm);
    transition: border-color 200ms ease, box-shadow 200ms ease;
}

.card:hover {
    border-color: var(--border-accent);
    box-shadow: var(--shadow-md);
}
```

**Data display (monospace for values):**

```css
.data-value {
    font-family: var(--font-mono);
    font-size: var(--font-size-sm);
    color: var(--accent-amber);
}
```

## JavaScript Architecture

### Module Structure

```
js/
├── app.js              # Main coordinator: init, routing, module wiring
├── api.js              # HTTP client: fetch wrapper, error handling
├── components.js       # Reusable render functions (cards, badges, timestamps)
└── {feature}.js        # One module per feature area (search, browse, analyze)
```

### `api.js` — HTTP Client Layer

```javascript
// api.js — Centralized HTTP client

const BASE = '/api';

async function get(path) {
    const resp = await fetch(`${BASE}${path}`);
    if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }));
        throw new Error(err.detail || `HTTP ${resp.status}`);
    }
    return resp.json();
}

async function post(path, body) {
    const resp = await fetch(`${BASE}${path}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
    if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }));
        throw new Error(err.detail || `HTTP ${resp.status}`);
    }
    return resp.json();
}

async function put(path, body) {
    const resp = await fetch(`${BASE}${path}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
    if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }));
        throw new Error(err.detail || `HTTP ${resp.status}`);
    }
    return resp.json();
}

// Domain-specific API objects
export const statusApi = {
    getStatus: () => get('/status'),
};

export const analysisApi = {
    run: (params) => post('/analyze', params),
    getResult: (id) => get(`/results/${id}`),
};

export const itemsApi = {
    list: (page = 1, pageSize = 20) => get(`/items?page=${page}&page_size=${pageSize}`),
    get: (id) => get(`/items/${id}`),
};
```

### `app.js` — Main Coordinator

```javascript
// app.js — Application entry point and router

import { statusApi } from './api.js';
import { initAnalysis } from './analysis.js';
import { initBrowse } from './browse.js';
import { initDashboard, stopAutoRefresh } from './dashboard.js';

const views = {};

function navigateTo(hash) {
    const { view, params } = parseHash(hash);

    // Hide all views
    document.querySelectorAll('.view').forEach(el => el.classList.remove('active'));

    // Cleanup previous view
    stopAutoRefresh();

    // Show target view
    const el = document.getElementById(`view-${view}`);
    if (el) {
        el.classList.add('active');
    }

    // Update nav
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.toggle('active', link.dataset.view === view);
    });

    // Initialize view
    switch (view) {
        case 'analyze':
            initAnalysis(params);
            break;
        case 'browse':
            initBrowse(params);
            break;
        case 'status':
            initDashboard();
            break;
    }
}

function parseHash(hash) {
    const clean = (hash || '#analyze').replace('#', '');
    const [view, queryString] = clean.split('?');
    const params = Object.fromEntries(new URLSearchParams(queryString || ''));
    return { view, params };
}

async function init() {
    // Set up navigation
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            window.location.hash = link.dataset.view;
        });
    });

    window.addEventListener('hashchange', () => navigateTo(window.location.hash));

    // Navigate to initial view
    navigateTo(window.location.hash || '#analyze');
}

// Expose for onclick handlers in HTML
window.app = { navigateTo };

document.addEventListener('DOMContentLoaded', init);
```

### Render Functions (not JSX, not templates)

OSPREY uses **HTML string rendering** — pure functions that return HTML strings:

```javascript
// components.js — Reusable render functions

export function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

export function formatTimestamp(isoString) {
    if (!isoString) return '—';
    const d = new Date(isoString);
    return d.toLocaleString('en-US', {
        month: 'short', day: 'numeric', year: 'numeric',
        hour: '2-digit', minute: '2-digit',
    });
}

export function renderCard(item) {
    return `
        <div class="card" data-id="${escapeHtml(item.id)}">
            <div class="card-header">
                <span class="data-value">${escapeHtml(item.id)}</span>
                <span class="timestamp">${formatTimestamp(item.timestamp)}</span>
            </div>
            <div class="card-body">
                <h3>${escapeHtml(item.title)}</h3>
                <p class="text-secondary">${escapeHtml(item.description)}</p>
            </div>
            ${item.score != null ? `
                <div class="card-footer">
                    <span class="badge badge-info">Score: ${item.score.toFixed(3)}</span>
                </div>
            ` : ''}
        </div>
    `;
}

export function renderLoading(message = 'Loading...') {
    return `<div class="loading-state"><div class="spinner"></div><p>${message}</p></div>`;
}

export function renderEmptyState(message, suggestion = '') {
    return `
        <div class="empty-state">
            <p>${escapeHtml(message)}</p>
            ${suggestion ? `<p class="text-tertiary">${escapeHtml(suggestion)}</p>` : ''}
        </div>
    `;
}

export function renderErrorState(message) {
    return `<div class="error-state"><p class="text-error">${escapeHtml(message)}</p></div>`;
}
```

### State Management

Module-level variables (closure pattern) — no Redux, no stores:

```javascript
// analysis.js
let currentMode = 'fft';
let isAnalyzing = false;
let lastResults = null;

export function initAnalysis(params) {
    if (params.mode) currentMode = params.mode;
    // Set up event listeners, render initial state
}
```

### XSS Prevention

**Always escape user-provided content:**

```javascript
// DANGEROUS — never do this with user data
container.innerHTML = `<p>${userData}</p>`;

// SAFE — escape first
container.innerHTML = `<p>${escapeHtml(userData)}</p>`;
```

If you need to preserve specific HTML tags (e.g., `<b>` from search highlights):

```javascript
export function sanitizeHighlight(html) {
    // Escape everything, then restore only <b> tags
    let safe = escapeHtml(html);
    safe = safe.replace(/&lt;b&gt;/g, '<b>');
    safe = safe.replace(/&lt;\/b&gt;/g, '</b>');
    return safe;
}
```

## HTML Structure

### `index.html` — Single-Page App Shell

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{Your Tool} — OSPREY</title>
    <link rel="stylesheet" href="/static/css/variables.css">
    <link rel="stylesheet" href="/static/css/base.css">
    <link rel="stylesheet" href="/static/css/layout.css">
    <link rel="stylesheet" href="/static/css/components.css">
</head>
<body>
    <header class="header">
        <div class="header-left">
            <h1 class="logo">{YOUR TOOL}</h1>
            <span class="subtitle">OSPREY Interface</span>
        </div>
        <nav class="header-center">
            <a href="#analyze" class="nav-link active" data-view="analyze">Analyze</a>
            <a href="#browse" class="nav-link" data-view="browse">Browse</a>
            <a href="#status" class="nav-link" data-view="status">Status</a>
        </nav>
        <div class="header-right">
            <span class="health-indicator" id="health-dot"></span>
        </div>
    </header>

    <main class="main-content">
        <div id="view-analyze" class="view active">
            <!-- Analyze view content -->
        </div>
        <div id="view-browse" class="view">
            <!-- Browse view content -->
        </div>
        <div id="view-status" class="view">
            <!-- Status view content -->
        </div>
    </main>

    <!-- ES6 modules — no build step needed -->
    <script type="module" src="/static/js/app.js"></script>
</body>
</html>
```

**Key points:**
- Views are `<div class="view">` elements, shown/hidden via `.active` class
- Hash-based routing (`#analyze`, `#browse`, `#status`)
- `<script type="module">` enables ES6 import/export
- No bundler, no transpiler, no build step

### Keyboard Shortcuts

```javascript
document.addEventListener('keydown', (e) => {
    // '/' to focus search input (like GitHub)
    if (e.key === '/' && !isInputFocused()) {
        e.preventDefault();
        document.getElementById('search-input')?.focus();
    }
    // Escape to blur
    if (e.key === 'Escape') {
        document.activeElement?.blur();
    }
});

function isInputFocused() {
    const tag = document.activeElement?.tagName;
    return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT';
}
```

## Concrete Reference

- `src/osprey/interfaces/ariel/static/index.html` — Full SPA shell with search, browse, create, status views
- `src/osprey/interfaces/ariel/static/js/api.js` — HTTP client with domain-specific API objects
- `src/osprey/interfaces/ariel/static/js/app.js` — Router, init, module wiring
- `src/osprey/interfaces/ariel/static/js/components.js` — Render functions for cards, badges, timestamps
- `src/osprey/interfaces/ariel/static/css/variables.css` — Full design token set

## Checklist

- [ ] CSS uses design tokens from `variables.css` (don't hardcode colors)
- [ ] Dark theme consistent with "Precision Instrumentation" aesthetic
- [ ] Monospace font for data values (IDs, timestamps, scores, measurements)
- [ ] ES6 modules with named exports
- [ ] `api.js` as centralized HTTP client (no raw `fetch()` in feature modules)
- [ ] All user data passed through `escapeHtml()` before `innerHTML`
- [ ] Hash-based routing for views
- [ ] Loading, empty, and error states for every data-dependent view
- [ ] No frameworks, no build step, no npm
