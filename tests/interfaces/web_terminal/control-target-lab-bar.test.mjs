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
 *    `panel-manager.js` or `sidebar.js` would drag the hub's page machinery
 *    onto a page that has none of it. The closure is walked transitively here
 *    and asserted as an exact set, so growing it is a deliberate edit to this
 *    list rather than a surprise in a browser.
 * 2. **The bar mounts into the wrong place, or twice.** JupyterLab owns
 *    `#main`'s layout and reflows what it finds there; the chip needs a host
 *    that is stable across its own re-renders. Both are asserted below.
 * 3. **The bar drifts from the chip.** A popped-out Lab page must say what the
 *    sidebar chip says and do what it does: the kind and the three state
 *    words (off, asked, allowed), the three-option control, the widening
 *    confirm and its sentence, all-off, the stamped-retry rule, the
 *    stop-not-taken sentence, and C-POST's body with `session_id: null`.
 * 4. **The sheet keeps a louder `allowed`.** The bar ships no style of its
 *    own; `terminal.css` is pinned here to carry a distinct
 *    `[data-state="allowed"]` treatment and no two-position toggle.
 *
 * Seams: `fetch` is stubbed the way the chip's own suite stubs it, and
 * `EventSource` is stubbed as a class rather than injected — the real
 * `api.js` shared-stream path is then exercised, which is what makes "one
 * socket on the Lab page too" assertable at all. Module-private state (the
 * bar, the chip singleton) is reset with vi.resetModules() + dynamic import,
 * the same pattern the chip suite uses; the dynamic import is also what tests
 * the module's self-boot, since importing it IS how the page runs it.
 */

import { readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import { FakeEventSource, stubEventSource } from '../_support/event-source.mjs';

const JS_DIR = join(import.meta.dirname, '../../../src/osprey/interfaces/web_terminal/static/js');
const ENTRY = 'control-target-lab-bar.js';
const MODULE = `../../../src/osprey/interfaces/web_terminal/static/js/${ENTRY}`;
/** The theme follower the entry starts on a page the hub framed. */
const FOLLOWER = '../../../src/osprey/interfaces/web_terminal/static/js/lab-theme-follow.js';

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
  const code = source.replace(/\/\*[\s\S]*?\*\//g, '').replace(/(^|[^:])\/\/[^\n]*/g, '$1');
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

const FACTS = '../../../src/osprey/interfaces/web_terminal/static/js/control-target-facts.js';
const { STOP_NOT_TAKEN_SENTENCE, TURN_ON_SENTENCE } = await import(FACTS);

/** The epoch the roster's switch record carries (C-GET `switch_epoch`). */
// A decimal string, as the route sends it: nanoseconds with low digits that a
// JSON number would round away.
const EPOCH = '1793181592449148123';

/** C-GET's columns for the three positions a switch can stand in. */
const ASKED = {
  switch: 'asked',
  prompt: true,
  write_allowed: true,
  write_reason: null,
  write_phrase: null,
};
const ALLOWED = {
  switch: 'allowed',
  prompt: false,
  write_allowed: true,
  write_reason: null,
  write_phrase: null,
};
const OFF = {
  switch: 'off',
  prompt: false,
  write_allowed: false,
  write_reason: 'switch_off',
  write_phrase: 'Writes are off; turn it on from the chip in the web terminal.',
};

/** @param {object} o */
const rowOf = (o) => ({
  display_name: '',
  ...ASKED,
  active: false,
  is_baseline: false,
  available_now: true,
  reason: null,
  reason_detail: null,
  reachability: {
    state: 'reached',
    role: 'write_access',
    probed_at: '2026-08-30T12:00:00+00:00',
    age_s: 3,
    role_detail: { write_access: 'reached' },
  },
  ...o,
});

/** The rehearsal machine the deployment stands on — the chip's card. */
const STANDIN = {
  target: 'standin',
  label: 'Rehearsal — nothing moves',
  short_label: 'STAND-IN',
  machine_kind: 'rehearsal',
  endpoint: 'standin-gw:5064',
  real_machine: true,
  active: true,
  is_baseline: true,
  available_now: false,
  reason: 'already_active',
};

/** The facility's own machine, writes off — a row. */
const LIVE = {
  target: 'live',
  label: 'Live — writes move hardware',
  short_label: 'LIVE',
  machine_kind: 'real',
  endpoint: 'gw.example.org:5064',
  real_machine: true,
  ...OFF,
};

/**
 * What GET /api/terminal/posture answers (C-GET). The Lab page's read carries
 * no session id, so the route echoes none — and that null is what every POST
 * from this page sends back.
 * @param {object} [o]
 */
const viewOf = (o = {}) => ({
  session_id: null,
  control_target: 'standin',
  generation: 3,
  owner: { kind: 'web_terminal', pid: 1000, port: 8080, self: true },
  servers: [],
  store_available: true,
  readonly_run: false,
  switch_epoch: EPOCH,
  any_armed: true,
  locks: { readonly_run: false, tier: 'allowed' },
  execution_in_flight: [],
  last_switch: null,
  last_posture_realign: null,
  targets: [rowOf(STANDIN), rowOf(LIVE)],
  ...o,
});

/** What the roster route answers next. @type {any} */
let served = viewOf();

/** Queued answers for the next POSTs, in order. */
/** @type {{ok: boolean, status?: number, body: any}[]} */
let postAnswers = [];

/** @type {{url: string, method: string, body: any}[]} */
let fetchCalls = [];

function stubFetch() {
  fetchCalls = [];
  postAnswers = [];
  vi.stubGlobal(
    'fetch',
    vi.fn(async (/** @type {any} */ url, /** @type {any} */ init) => {
      const method = init?.method ?? 'GET';
      const body = init?.body ? JSON.parse(String(init.body)) : null;
      fetchCalls.push({ url: String(url), method, body });
      if (method === 'GET') {
        const snapshot = JSON.parse(JSON.stringify(served));
        return { ok: true, status: 200, statusText: 'OK', json: async () => snapshot };
      }
      const answer = postAnswers.shift() ?? { ok: true, body: {} };
      return {
        ok: answer.ok,
        status: answer.status ?? (answer.ok ? 200 : 409),
        statusText: '',
        json: async () => answer.body,
      };
    })
  );
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
  served = viewOf();
  stubFetch();
  // The don't-ask-again waivers persist in localStorage; no test inherits
  // another's.
  localStorage.clear();
  stubEventSource({ open: true });
  mountLabFixture();
  Reflect.deleteProperty(window, '__OSPREY_PREFIX__');
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
  for (const stale of document.querySelectorAll('.posture-modal-overlay')) stale.remove();
});

/* ---- the import closure ------------------------------------------------- */

describe('import closure', () => {
  test('is exactly the chip, the popover, the theme follower and what they pull in', () => {
    // Every name here is fetched through the panel-scoped static route by a
    // test in test_proxy_jupyter_integration.py, which walks this same graph
    // from the Python side. Adding one means adding it there too — that is
    // the point of pinning it in both languages. The walk follows relative
    // specifiers only; an absolute `/design-system/js/…` import is served by
    // the design-system route and proven there. An exact set also rules out
    // every hub-page module (terminal.js, panel-manager.js, sidebar.js,
    // app.js) this page has none of.
    expect([...closureOf(ENTRY)].sort()).toEqual([
      'activity-format.js',
      'api.js',
      'confirm-skip.js',
      'control-target-chip.js',
      'control-target-facts.js',
      'control-target-popover-switch.js',
      'control-target-popover.js',
      'control-target-rows.js',
      'lab-theme-follow.js',
      'modal-overlay.js',
      'posture-confirm.js',
    ]);
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

  test('the chip mounts inside the bar; its popover at <body> level', async () => {
    const bar = /** @type {HTMLElement} */ (barEl());
    const anchor = /** @type {HTMLElement} */ (anchorEl());

    expect(anchor.parentElement).toBe(bar);
    expect(anchor.contains(/** @type {Node} */ (chipEl()))).toBe(true);
    // A design-system popover, placed by popover.js below the chip.
    const pop = /** @type {HTMLElement} */ (document.getElementById('control-target-popover'));
    expect(pop.parentElement).toBe(document.body);
    expect(pop.classList.contains('osprey-popover')).toBe(true);
  });

  test('the popover opens below the chip, right-aligned with it', async () => {
    const chip = /** @type {HTMLElement} */ (chipEl());
    chip.getBoundingClientRect = () =>
      /** @type {DOMRect} */ ({
        left: 600,
        right: 760,
        top: 1,
        bottom: 29,
        width: 160,
        height: 28,
      });

    chip.click();
    await flush();

    const pop = /** @type {HTMLElement} */ (document.getElementById('control-target-popover'));
    expect(pop.hidden).toBe(false);
    expect(pop.style.top).toBe('33px');
    expect(pop.style.right).toBe(`${window.innerWidth - 760}px`);
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
    expect(document.querySelectorAll('#control-target-popover').length).toBe(1);
  });

  test('teardown takes the bar, the chip and the styles back off the page', () => {
    barModule.teardownControlTargetLabBar();

    expect(barEl()).toBeNull();
    expect(chipEl()).toBeNull();
    expect(document.getElementById(barModule.STYLE_ID)).toBeNull();
    expect(document.getElementById(barModule.TERMINAL_CSS_ID)).toBeNull();
    expect(document.getElementById(barModule.TOKENS_CSS_ID)).toBeNull();
    expect(document.getElementById(barModule.PRIMITIVES_CSS_ID)).toBeNull();
    expect(document.getElementById('control-target-popover')).toBeNull();
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
    // The sidebar chip on the hub page is the one reader and the one socket.
    expect(getCount()).toBe(0);
    expect(FakeEventSource.opened).toHaveLength(0);
  });

  test('the self-boot starts the theme follower on a page framed with embedded=true', async () => {
    // The address the hub's buildEmbedSrc gives the JUPYTER tile.
    window.history.replaceState({}, '', '/panel/jupyter/lab?embedded=true&theme=dark');
    const execute = vi.fn(async () => {});
    Reflect.set(window, 'jupyterapp', {
      started: Promise.resolve(),
      commands: { hasCommand: () => true, isToggled: () => false, execute },
    });
    try {
      await bootFramed();
      // Lab has loaded its own theme: the follower's ready signal.
      document.body.setAttribute('data-jp-theme-name', 'JupyterLab Light');

      await vi.waitFor(() =>
        expect(execute).toHaveBeenCalledWith('apputils:change-theme', { theme: 'JupyterLab Dark' })
      );
      // Framed, the bar itself still stays off.
      expect(barEl()).toBeNull();
    } finally {
      (await import(FOLLOWER)).teardownLabThemeFollow();
      Reflect.deleteProperty(window, 'jupyterapp');
      document.body.removeAttribute('data-jp-theme-name');
      window.history.replaceState({}, '', '/');
    }
  });
});

/* ---- what the page has to load ------------------------------------------ */

describe('stylesheets', () => {
  test('loads the hub tokens, the primitives and the hub stylesheet, each once', () => {
    barModule.initControlTargetLabBar();

    const links = [...document.head.querySelectorAll('link[rel="stylesheet"]')].map(
      (l) => /** @type {HTMLLinkElement} */ (l).href
    );
    expect(links.filter((h) => h.endsWith('/design-system/css/tokens.css')).length).toBe(1);
    expect(links.filter((h) => h.endsWith('/design-system/css/primitives.css')).length).toBe(1);
    expect(links.filter((h) => h.endsWith('/css/terminal.css')).length).toBe(1);
  });

  test('positions the bar itself, and lifts the popover above the Lab chrome', () => {
    expect(document.getElementById(barModule.STYLE_ID)).not.toBeNull();
    // Fixed, because the bar is not part of any layout on this page. CSS file
    // loading is off here, so the injected rules are all that applies.
    expect(getComputedStyle(/** @type {Element} */ (barEl())).position).toBe('fixed');
    expect(document.getElementById(barModule.STYLE_ID)?.textContent).toMatch(
      /#control-target-popover\s*\{\s*z-index: var\(--z-overlay/
    );
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

/* ---- the chip's words and gestures, on a popped-out Lab page -------------- */

describe("the chip's words and gestures", () => {
  const posts = () => fetchCalls.filter((c) => c.method === 'POST');
  const chipBtn = () => /** @type {HTMLButtonElement} */ (chipEl());
  /** The card or the row that carries this target. @param {string} target */
  const surfaceEl = (target) => {
    const card = /** @type {HTMLElement|null} */ (document.querySelector('.ctc-card'));
    if (card?.dataset.target === target) return card;
    return /** @type {HTMLElement|null} */ (
      document.querySelector(`.ctc-row[data-target="${target}"]`)
    );
  };
  /** One option of a surface's three-option writes control. @param {string} target @param {string} position */
  const optionEl = (target, position) =>
    /** @type {HTMLButtonElement} */ (
      surfaceEl(target)?.querySelector(
        `[role="radiogroup"] [role="radio"][data-position="${position}"]`
      )
    );
  /** The positions that read checked, in order. @param {string} target */
  const checked = (target) =>
    [...(surfaceEl(target)?.querySelectorAll('[role="radio"][aria-checked="true"]') ?? [])].map(
      (n) => /** @type {HTMLElement} */ (n).dataset.position
    );
  const allOffEl = () => /** @type {HTMLButtonElement} */ (document.querySelector('.ctc-all-off'));
  /** @param {string} target */
  const outcomes = (target) =>
    [...(surfaceEl(target)?.querySelectorAll('.ctc-outcome') ?? [])].map((n) => n.textContent);
  const statusEl = () => document.querySelector('.ctc-status');
  const confirmEl = () =>
    /** @type {HTMLElement|null} */ (
      document.querySelector('.posture-modal-overlay:not([data-closing])')
    );
  const confirmBtn = () =>
    /** @type {HTMLButtonElement} */ (confirmEl()?.querySelector('.posture-modal-confirm'));

  /** Boot the page again on another roster — import is how the page boots. @param {any} view */
  async function rebootOn(view) {
    barModule.teardownControlTargetLabBar();
    vi.resetModules();
    served = view;
    fetchCalls.length = 0;
    barModule = await import(MODULE);
    await flush();
  }

  async function openPopover() {
    chipBtn().click();
    await flush();
  }

  test.each([
    ['asked', ASKED],
    ['allowed', ALLOWED],
    ['off', OFF],
  ])('the chip says the kind and writes %s in the hub’s words', async (word, columns) => {
    await rebootOn(
      viewOf({
        any_armed: word !== 'off',
        targets: [rowOf({ ...STANDIN, ...columns }), rowOf(LIVE)],
      })
    );
    expect(document.querySelector('.ctc-short')?.textContent).toBe('Rehearsal');
    expect(document.querySelector('.ctc-state')?.textContent).toBe(`writes ${word}`);
    expect(chipBtn().dataset.targetKind).toBe('rehearsal');
    expect(chipBtn().dataset.state).toBe(word);
  });

  test('a stored allowed the tier holds at asked reads writes asked', async () => {
    await rebootOn(
      viewOf({ targets: [rowOf({ ...STANDIN, ...ALLOWED, prompt: true }), rowOf(LIVE)] })
    );
    expect(document.querySelector('.ctc-state')?.textContent).toBe('writes asked');
    expect(chipBtn().dataset.state).toBe('asked');
  });

  test('the popover offers the same three-position control, the stored one checked', async () => {
    await rebootOn(viewOf({ targets: [rowOf(STANDIN), rowOf({ ...LIVE, ...ALLOWED })] }));
    await openPopover();

    for (const target of ['standin', 'live']) {
      const options = surfaceEl(target)?.querySelectorAll('[role="radiogroup"] [role="radio"]');
      expect(
        [...(options ?? [])].map((n) => /** @type {HTMLElement} */ (n).dataset.position)
      ).toEqual(['off', 'asked', 'allowed']);
    }
    expect(checked('standin')).toEqual(['asked']);
    expect(checked('live')).toEqual(['allowed']);
    expect(surfaceEl('live')?.querySelector('.ctc-switchwrap')?.getAttribute('data-state')).toBe(
      'allowed'
    );
    // No two-position toggle survives on this page either.
    expect(document.querySelector('.ctc-toggle')).toBeNull();
  });

  test('turning writes off posts C-POST to the hub with session_id null, and asks nothing', async () => {
    await openPopover();
    postAnswers.push({ ok: true, body: { target: 'standin', state: 'off', switch_epoch: EPOCH } });
    optionEl('standin', 'off').click();
    await flush();

    expect(confirmEl()).toBeNull();
    expect(posts()).toHaveLength(1);
    expect(posts()[0].url).toBe('/api/terminal/posture');
    expect(posts()[0].body).toEqual({
      session_id: null,
      target: 'standin',
      state: 'off',
      epoch: EPOCH,
      waived: false,
    });
  });

  test('setting the real machine to asked asks first, in the hub’s words, then posts', async () => {
    await openPopover();
    optionEl('live', 'asked').click();
    await flush();

    expect(posts()).toHaveLength(0);
    expect(confirmEl()?.querySelector('.posture-modal-title')?.textContent).toBe(
      'Set writes to asked on Live?'
    );
    expect(confirmEl()?.querySelector('.posture-modal-live')?.textContent).toBe(
      'Live — writes move hardware'
    );
    expect(confirmEl()?.querySelector('.posture-modal-body')?.textContent).toContain(
      TURN_ON_SENTENCE
    );
    expect(confirmBtn().textContent).toBe('Set writes to asked');

    postAnswers.push({ ok: true, body: { target: 'live', state: 'asked', switch_epoch: EPOCH } });
    confirmBtn().click();
    await flush();

    expect(posts()[0].body).toEqual({
      session_id: null,
      target: 'live',
      state: 'asked',
      epoch: EPOCH,
      waived: false,
    });
    expect(confirmEl()).toBeNull();
  });

  test('setting the real machine to allowed asks first, then posts allowed', async () => {
    await openPopover();
    optionEl('live', 'allowed').click();
    await flush();

    expect(posts()).toHaveLength(0);
    expect(confirmEl()?.querySelector('.posture-modal-title')?.textContent).toBe(
      'Set writes to allowed on Live?'
    );
    expect(confirmBtn().textContent).toBe('Set writes to allowed');

    postAnswers.push({ ok: true, body: { target: 'live', state: 'allowed', switch_epoch: EPOCH } });
    confirmBtn().click();
    await flush();

    expect(posts()[0].body).toEqual({
      session_id: null,
      target: 'live',
      state: 'allowed',
      epoch: EPOCH,
      waived: false,
    });
  });

  test('narrowing allowed to asked posts at once', async () => {
    await rebootOn(viewOf({ targets: [rowOf(STANDIN), rowOf({ ...LIVE, ...ALLOWED })] }));
    await openPopover();
    optionEl('live', 'asked').click();
    await flush();

    expect(confirmEl()).toBeNull();
    expect(posts()[0].body).toEqual({
      session_id: null,
      target: 'live',
      state: 'asked',
      epoch: EPOCH,
      waived: false,
    });
  });

  test('turn all writes off posts all/off from the bar, and asks nothing', async () => {
    await openPopover();
    expect(allOffEl().disabled).toBe(false);
    postAnswers.push({ ok: true, body: { target: 'all', state: 'off', switch_epoch: EPOCH } });
    allOffEl().click();
    await flush();

    expect(confirmEl()).toBeNull();
    expect(posts()).toHaveLength(1);
    expect(posts()[0].url).toBe('/api/terminal/posture');
    expect(posts()[0].body).toEqual({
      session_id: null,
      target: 'all',
      state: 'off',
      epoch: EPOCH,
      waived: false,
    });
  });

  test('all-off is disabled when the read says nothing is armed', async () => {
    await rebootOn(
      viewOf({ any_armed: false, targets: [rowOf({ ...STANDIN, ...OFF }), rowOf(LIVE)] })
    );
    await openPopover();
    expect(allOffEl().disabled).toBe(true);
  });

  test('a stamped 409 is retried once, with the epoch it returned', async () => {
    await openPopover();
    optionEl('live', 'asked').click();
    await flush();
    postAnswers.push(
      {
        ok: false,
        status: 409,
        body: { detail: { reason: 'stopped_since', epoch: '12', stamped: true } },
      },
      {
        ok: false,
        status: 409,
        body: { detail: { reason: 'stopped_since', epoch: '13', stamped: true } },
      },
      { ok: true, body: {} }
    );
    confirmBtn().click();
    await flush();

    expect(posts().map((p) => p.body)).toEqual([
      { session_id: null, target: 'live', state: 'asked', epoch: EPOCH, waived: false },
      { session_id: null, target: 'live', state: 'asked', epoch: '12', waived: false },
    ]);
  });

  test('a stop the record did not take says so', async () => {
    await openPopover();
    postAnswers.push({
      ok: false,
      status: 503,
      body: { detail: { reason: 'not_taken', taken: false, phrase: STOP_NOT_TAKEN_SENTENCE } },
    });
    optionEl('standin', 'off').click();
    await flush();

    expect(outcomes('standin')).toContain(`✗ ${STOP_NOT_TAKEN_SENTENCE}`);
  });

  test('a waived move to asked applies directly and says the turn-on sentence in the status line', async () => {
    localStorage.setItem('osprey-ctc-skip-confirm:move', '1');
    await rebootOn(
      viewOf({ any_armed: false, targets: [rowOf({ ...STANDIN, ...OFF }), rowOf(LIVE)] })
    );
    await openPopover();
    postAnswers.push({
      ok: true,
      body: { target: 'standin', state: 'asked', switch_epoch: EPOCH },
    });
    optionEl('standin', 'asked').click();
    await flush();

    expect(confirmEl()).toBeNull();
    expect(posts()[0].body).toEqual({
      session_id: null,
      target: 'standin',
      state: 'asked',
      epoch: EPOCH,
      waived: true,
    });
    expect(statusEl()?.textContent).toBe(TURN_ON_SENTENCE);
  });
});

/* ---- the sheet the bar links ---------------------------------------------- */

describe('terminal.css', () => {
  const CSS = readFileSync(
    join(
      import.meta.dirname,
      '../../../src/osprey/interfaces/web_terminal/static/css/terminal.css'
    ),
    'utf8'
  ).replace(/\/\*[\s\S]*?\*\//g, '');

  /** Every selector list the sheet declares, one string per rule. */
  const selectors = () => [...CSS.matchAll(/([^{}]+)\{[^{}]*\}/g)].map((m) => m[1].trim());

  test('carries a distinct allowed treatment, louder on the real machine', () => {
    const allowed = selectors().filter((sel) => sel.includes('[data-state="allowed"]'));
    expect(allowed.length).toBeGreaterThan(0);
    // The loudest cell: the real machine at allowed, on the chip itself.
    expect(
      allowed.some(
        (sel) =>
          sel.includes('.control-target-chip') &&
          sel.includes('[data-target-kind="real"]') &&
          !sel.includes('[data-state="asked"]')
      )
    ).toBe(true);
    // Asked has its own rules too, so the two never collapse into one look.
    expect(selectors().some((sel) => sel.includes('[data-state="asked"]'))).toBe(true);
  });

  test('styles the three-option control, and no two-position toggle or on-state remains', () => {
    const all = selectors().join('\n');
    expect(all).toContain('.ctc-positions');
    expect(all).toMatch(/\.ctc-position\[aria-checked="true"\]/);
    expect(all).toMatch(/\.ctc-position:focus-visible/);
    expect(all).toMatch(/\.ctc-position\[data-pending="true"\]/);
    expect(all).toMatch(/\.ctc-position:disabled/);
    expect(all).toContain('.ctc-all-off');
    expect(all).not.toContain('.ctc-toggle');
    expect(all).not.toContain('[data-state="on"]');
  });
});
