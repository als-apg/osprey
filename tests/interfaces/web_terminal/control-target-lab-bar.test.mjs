// @ts-check
/**
 * Unit tests for the JupyterLab control-target bar (control-target-lab-bar.js):
 *   npx vitest run tests/interfaces/web_terminal/control-target-lab-bar.test.mjs
 *
 * This module is an ENTRY POINT on somebody else's page. The panel proxy
 * injects one `<script type="module">` into JupyterLab's `<head>` and nothing
 * else runs it, so the two things that can go wrong are both invisible to the
 * hub's own suites:
 *
 * 1. **The import closure grows.** Every module this file reaches has to be
 *    fetchable through `/panel/{id}/terminal-static/js/`, and the first
 *    unreachable one breaks the whole graph — the bar simply never appears,
 *    with a 404 in a console nobody is watching. One import of `terminal.js`,
 *    `panel-manager.js` or `bar-host.js` would drag the hub's page machinery
 *    onto a page that has none of it. The closure is walked transitively here
 *    and asserted as an exact set, so growing it is a deliberate edit to this
 *    list rather than a surprise in a browser.
 * 2. **The bar mounts into the wrong place, or twice.** JupyterLab owns
 *    `#main`'s layout and reflows what it finds there; the chip needs a host
 *    that is stable across its own re-renders. Both are asserted below.
 *
 * Seams: `fetch` is stubbed the way the chip's own suite stubs it, and
 * `EventSource` is stubbed as a class rather than injected — the real
 * `api.js` shared-stream path is then exercised, which is what makes "one
 * socket on the Lab page too" assertable at all. Module-private state (the
 * bar, the chip singleton) is reset with vi.resetModules() + dynamic import,
 * the same pattern the chip suite uses; the dynamic import is also what tests
 * the module's self-boot, since importing it IS how the page runs it.
 */

import { readFileSync, existsSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

/** The repo root, for turning a served path back into a file on disk. */
const REPO_ROOT = join(import.meta.dirname, '../../..');

const JS_DIR = join(
  import.meta.dirname,
  '../../../src/osprey/interfaces/web_terminal/static/js'
);
const ENTRY = 'control-target-lab-bar.js';
const MODULE = `../../../src/osprey/interfaces/web_terminal/static/js/${ENTRY}`;

/** @type {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/control-target-lab-bar.js')} */
let barModule;

/* ---- the import closure ------------------------------------------------- */

/**
 * Every relative specifier in a source file, in the two forms ES modules take
 * and across the line breaks a multi-name import list puts in the middle of
 * the statement.
 *
 * Comments are stripped first: these files DISCUSS their own imports (the chip
 * documents the `./terminal.js` import it no longer has), and a walker that
 * read prose as an edge would report a closure the browser never fetches.
 */
function relativeImports(/** @type {string} */ source) {
  const code = source
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/(^|[^:])\/\/[^\n]*/g, '$1');
  const found = new Set();
  const pattern = /\bfrom\s+'(\.[^']+)'|(?:^|\n)\s*import\s+'(\.[^']+)'/g;
  for (const match of code.matchAll(pattern)) {
    found.add(match[1] ?? match[2]);
  }
  return found;
}

/**
 * Every module reachable from *entry* by relative import, transitively.
 * @param {string} entry file name inside the static js directory
 * @returns {Set<string>} file names, entry excluded
 */
function closureOf(entry) {
  const seen = new Set([entry]);
  const queue = [entry];
  while (queue.length) {
    const current = /** @type {string} */ (queue.shift());
    const source = readFileSync(join(JS_DIR, current), 'utf8');
    for (const specifier of relativeImports(source)) {
      const name = resolve(dirname(join(JS_DIR, current)), specifier).slice(JS_DIR.length + 1);
      if (!seen.has(name)) {
        seen.add(name);
        queue.push(name);
      }
    }
  }
  seen.delete(entry);
  return seen;
}

/* ---- harness ------------------------------------------------------------ */

/** What GET /api/terminal/posture answers. */
const VIEW = {
  session_id: null,
  control_target: 'standin',
  generation: 3,
  owner: { kind: 'web_terminal', pid: 1000, port: 8080, self: true },
  servers: [
    {
      pid: 4242,
      session: null,
      applied_target: 'standin',
      applied_generation: 3,
      last_switch: null,
      last_posture_realign: null,
      updated_at: '2026-08-30T12:00:00+00:00',
    },
  ],
  store_available: true,
  readonly_run: false,
  execution_in_flight: [],
  last_switch: null,
  last_posture_realign: null,
  targets: [
    {
      target: 'standin',
      label: 'STAND-IN',
      short_label: 'STAND-IN',
      kind: 'stand-in',
      endpoint: 'standin-gw:5064',
      real_machine: false,
      effective: true,
      posture: 'writes',
      active: true,
      is_baseline: true,
      available_now: true,
      reason: null,
      ceiling_writes: true,
      reachability: {
        state: 'reached',
        role: 'write_access',
        probed_at: '2026-08-30T12:00:00+00:00',
        age_s: 3,
        role_detail: { write_access: 'reached' },
      },
    },
  ],
};

/** @type {{url: string, method: string}[]} */
let fetchCalls = [];

function stubFetch() {
  fetchCalls = [];
  vi.stubGlobal(
    'fetch',
    vi.fn(async (/** @type {any} */ url, /** @type {any} */ init) => {
      fetchCalls.push({ url: String(url), method: init?.method ?? 'GET' });
      return {
        ok: true,
        status: 200,
        statusText: 'OK',
        json: async () => JSON.parse(JSON.stringify(VIEW)),
      };
    })
  );
}

/**
 * happy-dom ships no EventSource, and the chip reaches api.js's real shared
 * stream. A class stub keeps that path intact and counts the sockets.
 */
class FakeEventSource {
  /** @type {FakeEventSource[]} */
  static opened = [];
  /** @param {string} url */
  constructor(url) {
    this.url = url;
    this.readyState = 1;
    this.closed = false;
    FakeEventSource.opened.push(this);
  }
  close() {
    this.closed = true;
  }
}

/** Drain the microtask/timer queue the async handlers chain through. */
async function flush() {
  for (let i = 0; i < 5; i++) await new Promise((r) => setTimeout(r, 0));
}

/** The page as JupyterLab lays it out: a `#main` the bar must stay out of. */
function mountLabFixture() {
  document.body.innerHTML = '<div id="main"><div class="jp-Notebook"></div></div>';
}

const barEl = () => document.getElementById('osprey-control-target-bar');
const chipEl = () => document.querySelector('.control-target-chip');
const anchorEl = () => document.querySelector('.ctc-anchor');
const getCount = () => fetchCalls.filter((c) => c.method === 'GET').length;

beforeEach(async () => {
  vi.resetModules();
  // The bar links two real stylesheets into `<head>`, and happy-dom would go
  // and fetch them off the vite dev server, where the panel-relative paths
  // they are built from do not exist. The links themselves are what this
  // suite asserts; loading their bodies is a browser's job.
  const settings = /** @type {any} */ (window).happyDOM?.settings;
  if (settings) {
    settings.disableCSSFileLoading = true;
    settings.handleDisabledFileLoadingAsSuccess = true;
  }
  FakeEventSource.opened = [];
  stubFetch();
  vi.stubGlobal('EventSource', FakeEventSource);
  mountLabFixture();
  delete (/** @type {any} */ (window)).__OSPREY_PREFIX__;
  // Importing the module is how the page runs it: there is no app shell on a
  // Lab page to call init, so the module boots itself on import.
  barModule = await import(MODULE);
  await flush();
});

afterEach(() => {
  barModule.teardownControlTargetLabBar();
  vi.useRealTimers();
  vi.unstubAllGlobals();
  document.body.innerHTML = '';
  document.body.removeAttribute('data-jp-theme-light');
  document.documentElement.removeAttribute('data-theme');
});

/* ---- the import closure ------------------------------------------------- */

describe('import closure', () => {
  test('is exactly the chip, the popover and what those two pull in', () => {
    // Every name here is fetched through the panel-scoped static route by a
    // test in test_proxy_jupyter_integration.py, which walks this same graph
    // from the Python side. Adding one means adding it there too — that is
    // the point of pinning it in both languages.
    expect([...closureOf(ENTRY)].sort()).toEqual([
      'activity-format.js',
      'api.js',
      'confirm-skip.js',
      'control-target-chip.js',
      'control-target-facts.js',
      'control-target-popover.js',
      'modal-overlay.js',
      'posture-confirm.js',
    ]);
  });

  test('reaches nothing the terminal page owns', () => {
    // The Lab page has no shell, no bar host and no terminal: a module that
    // reached one would either throw on load or quietly wire the bar to a DOM
    // that is not there.
    const closure = closureOf(ENTRY);
    for (const forbidden of ['terminal.js', 'panel-manager.js', 'bar-host.js', 'app.js']) {
      expect(closure.has(forbidden)).toBe(false);
    }
  });

  test('every module in it is on disk under the served static tree', () => {
    // The route serves this directory and nothing else, so a specifier that
    // resolves outside it is a 404 in the browser.
    for (const name of closureOf(ENTRY)) {
      expect(existsSync(join(JS_DIR, name))).toBe(true);
    }
  });
});

/* ---- mount -------------------------------------------------------------- */

describe('mount', () => {
  test('boots itself on import — the page has no shell to call it', () => {
    expect(barEl()).not.toBeNull();
    expect(chipEl()).not.toBeNull();
  });

  test('sits on document.body, outside the layout JupyterLab owns', () => {
    const bar = /** @type {HTMLElement} */ (barEl());
    const main = /** @type {HTMLElement} */ (document.getElementById('main'));

    expect(bar.parentElement).toBe(document.body);
    expect(main.contains(bar)).toBe(false);
  });

  test('the chip mounts inside the bar, in its own positioning context', async () => {
    const bar = /** @type {HTMLElement} */ (barEl());
    const anchor = /** @type {HTMLElement} */ (anchorEl());

    expect(anchor.parentElement).toBe(bar);
    expect(anchor.contains(/** @type {Node} */ (chipEl()))).toBe(true);
    // The popover is absolute against the anchor, so it travels with the chip.
    expect(anchor.querySelector('.ctc-popover')).not.toBeNull();
  });

  test('paints the roster the deployment answered with', async () => {
    // `Rehearsal` is the kind's own word (control-target-facts.js), which is
    // what the chip shows for a target the deployment gave no display name.
    expect(document.querySelector('.ctc-short')?.textContent).toBe('Rehearsal');
    expect(getCount()).toBe(1);
  });

  test('re-initialising re-uses the same bar and the same chip', async () => {
    const first = barEl();

    barModule.initControlTargetLabBar();
    await flush();

    expect(barEl()).toBe(first);
    expect(document.querySelectorAll(`#osprey-control-target-bar`).length).toBe(1);
    expect(document.querySelectorAll('.control-target-chip').length).toBe(1);
    expect(document.querySelectorAll('.ctc-popover').length).toBe(1);
  });

  test('teardown takes the bar, the chip and the styles back off the page', () => {
    barModule.teardownControlTargetLabBar();

    expect(barEl()).toBeNull();
    expect(chipEl()).toBeNull();
    expect(document.getElementById(barModule.STYLE_ID)).toBeNull();
    expect(document.getElementById(barModule.TERMINAL_CSS_ID)).toBeNull();
    expect(document.getElementById(barModule.TOKENS_CSS_ID)).toBeNull();
    // Including the attribute it stamped on someone else's document.
    expect(document.documentElement.hasAttribute('data-theme')).toBe(false);
  });
});

/* ---- framed inside the hub ---------------------------------------------- */

describe('framed inside the hub', () => {
  /** Boot the module again with the page framed, the way the JUPYTER tab is. */
  async function bootFramed() {
    barModule.teardownControlTargetLabBar();
    fetchCalls.length = 0;
    FakeEventSource.opened = [];
    vi.resetModules();
    // Any `top` that is not `self` is a frame; the hub's iframe is one.
    vi.stubGlobal('top', {});
    barModule = await import(MODULE);
    await flush();
  }

  test('isEmbedded reads self against top, and treats an unreadable top as framed', () => {
    const same = {};
    expect(barModule.isEmbedded({ self: same, top: same })).toBe(false);
    expect(barModule.isEmbedded({ self: same, top: {} })).toBe(true);
    const strict = {
      self: same,
      get top() {
        throw new Error('blocked');
      },
    };
    expect(barModule.isEmbedded(strict)).toBe(true);
  });

  test('the self-boot mounts nothing: no bar, no styles, no read, no stream', async () => {
    await bootFramed();

    expect(barModule.isEmbedded()).toBe(true);
    expect(barEl()).toBeNull();
    expect(chipEl()).toBeNull();
    expect(document.getElementById(barModule.STYLE_ID)).toBeNull();
    expect(document.getElementById(barModule.TERMINAL_CSS_ID)).toBeNull();
    expect(document.getElementById(barModule.TOKENS_CSS_ID)).toBeNull();
    expect(document.documentElement.hasAttribute('data-theme')).toBe(false);
    // The header chip on the hub page is the one reader and the one socket.
    expect(getCount()).toBe(0);
    expect(FakeEventSource.opened).toHaveLength(0);
  });

  test('init answers null while framed, and the override mounts as its own window would', async () => {
    await bootFramed();

    expect(barModule.initControlTargetLabBar()).toBeNull();
    expect(barEl()).toBeNull();

    const bar = barModule.initControlTargetLabBar({ embedded: false });
    await flush();
    expect(bar).not.toBeNull();
    expect(barEl()).toBe(bar);
    expect(chipEl()).not.toBeNull();
  });
});

/* ---- what the page has to load ------------------------------------------ */

describe('stylesheets', () => {
  test('loads the hub tokens and the hub stylesheet, each once', () => {
    barModule.initControlTargetLabBar();

    const links = [...document.head.querySelectorAll('link[rel="stylesheet"]')].map(
      (l) => /** @type {HTMLLinkElement} */ (l).href
    );
    expect(links.filter((h) => h.endsWith('/design-system/css/tokens.css')).length).toBe(1);
    expect(links.filter((h) => h.endsWith('/css/terminal.css')).length).toBe(1);
  });

  test('addresses them relative to this module, so the per-user mount is free', () => {
    // Neither URL is an injected literal: they are resolved against
    // import.meta.url, which already carries whatever prefix the module was
    // served under. What that resolves to on the PANEL's paths is pinned
    // against a real proxied sidecar in test_proxy_jupyter_integration.py;
    // what is checkable here is the hop each one makes out of `js/`.
    expect(barModule.TERMINAL_CSS_URL.endsWith('/static/css/terminal.css')).toBe(true);
    expect(barModule.TOKENS_CSS_URL.endsWith('/design-system/css/tokens.css')).toBe(true);

    // One hop up out of `js/` is a real file in the tree the route serves.
    const served = new URL(barModule.TERMINAL_CSS_URL).pathname;
    expect(existsSync(join(REPO_ROOT, served))).toBe(true);
  });

  test('positions the bar itself, and anchors the popover before the sheet lands', () => {
    const style = /** @type {HTMLStyleElement} */ (
      document.getElementById(barModule.STYLE_ID)
    );
    // Fixed, because the bar is not part of any layout on this page — and the
    // anchor rule is a floor under terminal.css, which carries the same one.
    expect(style.textContent).toContain('position: fixed');
    expect(style.textContent).toContain('.ctc-anchor');
    expect(style.textContent).toContain('position: relative');
  });
});

/* ---- theme -------------------------------------------------------------- */

describe('theme', () => {
  test("follows JupyterLab's own light/dark choice", async () => {
    document.body.setAttribute('data-jp-theme-light', 'false');
    await flush();

    expect(document.documentElement.getAttribute('data-theme')).toBe('dark');

    document.body.setAttribute('data-jp-theme-light', 'true');
    await flush();

    expect(document.documentElement.getAttribute('data-theme')).toBe('light');
  });

  test('defaults to light, which is what JupyterLab boots as', () => {
    // The attribute is absent until Lab has booted; the bar loads first.
    expect(document.body.hasAttribute('data-jp-theme-light')).toBe(false);
    expect(document.documentElement.getAttribute('data-theme')).toBe('light');
  });
});

/* ---- what the chip reads ------------------------------------------------ */

describe('reads', () => {
  test('asks the HUB for the roster, not the panel it is embedded in', () => {
    // The read is root-absolute and goes through api.js's withPrefix. On this
    // page that resolves against window.__OSPREY_PREFIX__, which the proxy
    // sets to the OUTER prefix — a prefix of /panel/jupyter would aim every
    // read at JupyterLab, which answers none of them.
    expect(fetchCalls[0].url).toBe('/api/terminal/posture');
    expect(fetchCalls[0].url).not.toContain('/panel/');
  });

  test('carries the per-user mount prefix when the deployment has one', async () => {
    barModule.teardownControlTargetLabBar();
    vi.resetModules();
    (/** @type {any} */ (window)).__OSPREY_PREFIX__ = '/u/alice';
    fetchCalls = [];

    barModule = await import(MODULE);
    await flush();

    expect(fetchCalls[0].url).toBe('/u/alice/api/terminal/posture');
  });

  test('opens ONE event stream on the Lab page', async () => {
    // api.js shares one socket per URL across the page. The chip is the only
    // subscriber here, and a second EventSource would count against the
    // browser's six-connection cap for every tab the operator has open.
    expect(FakeEventSource.opened.length).toBe(1);
    expect(FakeEventSource.opened[0].url).toBe('/api/files/events');

    barModule.initControlTargetLabBar();
    await flush();

    expect(FakeEventSource.opened.length).toBe(1);
  });
});
