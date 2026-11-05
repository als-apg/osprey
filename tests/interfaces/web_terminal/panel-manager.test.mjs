// @ts-check
/**
 * Unit tests for panel-manager.js's per-user URL prefix awareness
 * (window.__OSPREY_PREFIX__, the multi-user prefix contract — see
 * api.test.mjs for the api.js helpers this module builds on). Covers:
 *
 *   - initPanel()'s PANELS[].configEndpoint fetches and the /api/panels
 *     fetch, both via fetchJSON (prefixed internally)
 *   - the /api/files/events EventSource, via createEventSource (prefixed
 *     internally)
 *   - the /api/panel-focus POST on a user-initiated rail switch (via
 *     panel-commands.js, prefixed with withPrefix)
 *   - the iframe-src builders in navigatePanel()/createIframe(): state.url
 *     arrives from the server ALREADY prefixed (routes/panels.py's
 *     compute_url_prefix()), so `new URL(path, origin)` must preserve it
 *     as-is, never re-strip or double-add window.__OSPREY_PREFIX__
 *
 * Every prefix case is paired with an empty-prefix case asserting
 * byte-identical (unprefixed) behavior, per the prefix contract.
 *
 * Beyond the prefix contract this file also pins the SSE-driven behavior the
 * rail and the dock share — agent activity styling, rail membership, the
 * and (with a hand-built DockviewApi published
 * through the mocked dock-workspace module) the placement an agent open_panel
 * produces: focus the panel's own tile, or open one BESIDE, never evicting the
 * tile the operator is watching.
 *
 * Module isolation: panel-manager.js keeps PANELS/panelState/visiblePanels
 * as module-private state mutated in place by initPanelManager(), so each
 * test does vi.resetModules() + a fresh dynamic import (same pattern as
 * api.test.mjs) to avoid cross-test leakage.
 *
 *   npx vitest run tests/interfaces/web_terminal/panel-manager.test.mjs
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

// The stall watchdogs below report on the real stderr; see reportStall.
import { stderr } from 'node:process';

// panel-catalog.js reads the built-in panel labels off <html> at module
// load, so the roster has to be stamped before freshImport() reaches it —
// otherwise every panel here is labelled by its id.
import { FakeResizeObserver, TERMINAL_LAYOUT, makeDockApi, rowOf } from './_dock-fake.mjs';
import { stampPanelLabels } from './panel-labels-fixture.mjs';
import { stubEventSource } from '../_support/event-source.mjs';
import { FACILITY_ZONE, stampFacilityZone } from '../_support/facility-zone.mjs';

// dock-workspace.js is stubbed at the module boundary so a test can publish a
// hand-built DockviewApi (_dock-fake.mjs) and a real layout holder over it,
// and exercise the real placement engine in dock-iframe.js / dock-sync.js. `dockState.api` stays null by
// default, which is exactly what the real getDockApi returns with no dockview
// shell — every test that does not opt in keeps running in fallback mode.
const {
  getDockApi,
  openTerminalPanel,
  closeTerminalPanel,
  isTerminalPanelOpen,
  resetDockLayout,
  dockState,
} = vi.hoisted(() => ({
  dockState: {
    api: /** @type {any} */ (null),
    redock: /** @type {(() => void) | null} */ (null),
    // The boot-layout seam: whether the stored layout is in, and the hook the
    // manager registers to re-run its landing once it is. Settled by default,
    // so a test that does not model a restore lands at the first health settle.
    bootSettled: true,
    restored: /** @type {(() => void) | null} */ (null),
  },
  getDockApi: vi.fn(() => /** @type {any} */ (null)),
  openTerminalPanel: vi.fn(),
  closeTerminalPanel: vi.fn(),
  isTerminalPanelOpen: vi.fn(() => true),
  resetDockLayout: vi.fn(),
}));
getDockApi.mockImplementation(() => dockState.api);

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/dock-workspace.js', () => ({
  getDockApi,
  openTerminalPanel,
  closeTerminalPanel,
  isTerminalPanelOpen,
  resetDockLayout,
  // Pulled in by dock-iframe.js on the transitive import chain; a mock must
  // supply them or they resolve to undefined.
  // Captured, so a test can replay "a layout was applied" the way
  // dock-workspace does after a stored-layout restore.
  setServiceRedock: (/** @type {() => void} */ fn) => {
    dockState.redock = fn;
  },
  bootLayoutSettled: () => dockState.bootSettled,
  setLayoutRestoredHook: (/** @type {() => void} */ fn) => {
    dockState.restored = fn;
    if (dockState.bootSettled) fn();
  },
  onDragGesture: () => [],
}));

/** Announce the boot layout as final, as dock-workspace does after its restore. */
function settleBootLayout() {
  dockState.bootSettled = true;
  dockState.restored?.();
}

// The terminal verbs the context menu binds live outside panel-manager's own
// state machine (a PTY socket and the session list). Stub both modules at the
// boundary so a menu pick can be asserted as the call it makes — the restart
// PAIR above all, which is a correctness contract, not a detail: restartTerminal
// tears the PTY down without reconnecting.
// Only the four verbs are replaced: terminal.js is already on the import
// chain (panel-iframe-sync reads the live session id from it), so a whole-module
// stub would strand that reader — these are partial mocks, registered per
// import next to the dock-iframe one below for the same freshness reason.
const TERMINAL_PATH = '../../../src/osprey/interfaces/web_terminal/static/js/terminal.js';
const SESSIONS_PATH = '../../../src/osprey/interfaces/web_terminal/static/js/sessions.js';
const { restartTerminal, startTerminal, focusTerminal, startNewSession } = vi.hoisted(() => ({
  restartTerminal: vi.fn(async () => {}),
  startTerminal: vi.fn(),
  focusTerminal: vi.fn(),
  startNewSession: vi.fn(),
}));

/** @param {() => Promise<unknown>} importOriginal */
async function terminalWithVerbSpies(importOriginal) {
  const actual = /** @type {Record<string, unknown>} */ (await importOriginal());
  return { ...actual, restartTerminal, startTerminal, focusTerminal };
}

/** @param {() => Promise<unknown>} importOriginal */
async function sessionsWithVerbSpies(importOriginal) {
  const actual = /** @type {Record<string, unknown>} */ (await importOriginal());
  return { ...actual, startNewSession };
}

// dock-iframe.js keeps its REAL placement engine — only the tile-glow entry
// point is spied on. What the glow looks like (an overlay rectangle measured a
// frame later) is dock-glow.test.mjs's and the browser suite's contract; what
// this file pins is WHICH call sites fire it, which a spy states directly and a
// class assertion on a happy-dom-unlaid-out overlay could not.
//
// Registered per import (see freshImport) rather than through a hoisted
// vi.mock: the mock registry survives vi.resetModules(), so a hoisted factory
// would hand every later test the FIRST test's dock-iframe instance — an
// adapter still holding a removed overlay and a stale managed set. Re-running
// the factory after each reset keeps the adapter as fresh as the rest of the
// graph, which is the isolation every suite in this file depends on.
const DOCK_IFRAME_PATH = '../../../src/osprey/interfaces/web_terminal/static/js/dock-iframe.js';
const { glowPanelSpy } = vi.hoisted(() => ({ glowPanelSpy: vi.fn() }));

/** @param {() => Promise<unknown>} importOriginal */
async function dockIframeWithGlowSpy(importOriginal) {
  const actual = /** @type {Record<string, unknown>} */ (await importOriginal());
  return { ...actual, glowPanel: glowPanelSpy };
}

// first-contact.js is stubbed whole: the greeting's own copy and DOM wiring is
// first-contact.test.mjs's contract, not this file's, and what this file pins
// is only that BOTH exits of the /api/panels fetch hand it the facts.
//
// The stub carries more than setFacts because panel-manager is not the only
// module on this graph that reads it: tour.js imports the prompt, sentence and
// chip-row helpers too. Keep this list in step with what the graph imports;
// the helpers return the shapes their callers destructure, not real copy.
const FIRST_CONTACT_PATH = '../../../src/osprey/interfaces/web_terminal/static/js/first-contact.js';
const { setFacts, starterPrompts, capabilitySentence, chipRow } = vi.hoisted(() => ({
  setFacts: vi.fn(),
  starterPrompts: vi.fn(() => []),
  capabilitySentence: vi.fn(() => ''),
  chipRow: vi.fn(() => document.createElement('div')),
}));

/** Minimal ok-JSON fetch Response stand-in. @param {any} body */
function jsonOk(body) {
  return { ok: true, status: 200, statusText: 'OK', json: async () => body };
}

/** Renders the DOM initPanelManager expects: a container with #panel-content, and a sibling #panel-rail. */
function renderContainer() {
  document.body.innerHTML = `
    <nav id="panel-rail"></nav>
    <div id="panel-manager"><div id="panel-content"></div></div>
  `;
}

/**
 * Wait for a boot step to settle.
 *
 * Every condition this file waits on settles on the MICROTASK queue: a config
 * fetch resolving into panelState, a placeholder reaching the dock, a badge
 * being painted. None of them is driven by a timer.
 *
 * vi.waitFor checks once and then re-checks on a 50ms interval, so a condition
 * that becomes true a fraction of a millisecond after that first check still
 * costs a full 50ms tick — and this file pays that tick in nearly every one of
 * its tests, which is the bulk of what each test spends. Re-checking at the
 * granularity the conditions actually settle at keeps the whole boot off the
 * test's critical path, so the budget is spent on the work rather than on
 * waiting for a tick unrelated to it. The 1000ms bound and the failure message
 * are vi.waitFor's own, unchanged.
 *
 * @param {() => unknown} check
 * @returns {Promise<unknown>}
 */
const settled = (check) => vi.waitFor(check, { interval: 1 });

/* ---- stall diagnostics ----------------------------------------------------
 *
 * Every test here rebuilds panel-manager's whole module graph (freshImport),
 * and that await is the one in this file bounded by nothing of its own: it
 * waits on the runner's module fetch, so a machine busy with something else
 * can stretch it past the test budget while the test itself does nothing
 * wrong. Every other await is a settled() poll, which fails with its own message.
 *
 * A test killed by the budget is reported against its opening line and says
 * only that it timed out — neither which phase was still running nor for how
 * long, which is the difference between a starved import and a boot step that
 * genuinely never settles. The two watchdogs below say both, on stderr, while
 * the run is still inside the stall rather than after it is torn down.
 */

/** What a boot phase, or a whole test, may take before it is worth reporting. */
const STALL_BUDGET_MS = 2000;

/**
 * Say it on the real stderr. The runner's console interception does not carry
 * a hook's output through, and a diagnostic that only prints when the test is
 * healthy is no diagnostic at all.
 * @param {string} text
 */
function reportStall(text) {
  stderr.write(`[panel-manager stall] ${text}\n`);
}

/**
 * Run one boot phase under a watchdog that names it if it overruns.
 * @template T
 * @param {string} name
 * @param {() => Promise<T>} run
 * @returns {Promise<T>}
 */
async function phase(name, run) {
  const started = performance.now();
  // Fires while the phase is still stuck, which is the only moment a test the
  // runner is about to kill can still say what it was doing.
  const watchdog = setTimeout(
    () => reportStall(`${name}: still running after ${STALL_BUDGET_MS}ms`),
    STALL_BUDGET_MS
  );
  try {
    return await run();
  } finally {
    clearTimeout(watchdog);
    const elapsed = performance.now() - started;
    if (elapsed > STALL_BUDGET_MS) reportStall(`${name}: finished after ${Math.round(elapsed)}ms`);
  }
}

/** @returns {Promise<typeof import('../../../src/osprey/interfaces/web_terminal/static/js/panel-manager.js')>} */
async function freshImport() {
  vi.resetModules();
  vi.doMock(DOCK_IFRAME_PATH, dockIframeWithGlowSpy);
  vi.doMock(TERMINAL_PATH, terminalWithVerbSpies);
  vi.doMock(SESSIONS_PATH, sessionsWithVerbSpies);
  vi.doMock(FIRST_CONTACT_PATH, () => ({
    setFacts,
    starterPrompts,
    capabilitySentence,
    chipRow,
  }));
  return phase(
    'module graph',
    () => import('../../../src/osprey/interfaces/web_terminal/static/js/panel-manager.js')
  );
}

/** The whole-test watchdog's timer, armed per test and disarmed with it. */
let testWatchdog = /** @type {ReturnType<typeof setTimeout> | null} */ (null);
/** When the current test's hooks started, for the overrun report. */
let testStarted = 0;

beforeEach(() => {
  delete window.__OSPREY_PREFIX__;
  stampPanelLabels(document);
  vi.clearAllMocks();
  getDockApi.mockImplementation(() => dockState.api);
  dockState.api = null;
  dockState.bootSettled = true;
  dockState.restored = null;
  // The pop-out seam (popout.js) rolls call on a BroadcastChannel at boot,
  // which the test environment does not provide. Unstubbed with the rest.
  vi.stubGlobal(
    'BroadcastChannel',
    class {
      postMessage() {}
      close() {}
    }
  );

  // Armed in the first hook and disarmed in the last, so the span it covers is
  // the test body AND the hooks either side of it — a stall in a hook is as
  // silent in the runner's report as one in the body.
  testStarted = performance.now();
  const name = expect.getState().currentTestName ?? '(unnamed)';
  testWatchdog = setTimeout(
    () => reportStall(`${name}: still running after ${STALL_BUDGET_MS}ms`),
    STALL_BUDGET_MS
  );
});

afterEach(() => {
  if (testWatchdog) clearTimeout(testWatchdog);
  testWatchdog = null;
  const elapsed = performance.now() - testStarted;
  if (elapsed > STALL_BUDGET_MS) {
    reportStall(
      `${expect.getState().currentTestName ?? '(unnamed)'}: finished after ${Math.round(elapsed)}ms`
    );
  }

  vi.unstubAllGlobals();
  document.documentElement.removeAttribute('data-panel-labels');
  document.body.innerHTML = '';
});

describe('config fetches: /api/panels and PANELS[].configEndpoint (via fetchJSON)', () => {
  test('prepend window.__OSPREY_PREFIX__ when set', async () => {
    window.__OSPREY_PREFIX__ = '/u/alice';
    renderContainer();
    /** @type {string[]} */
    const calls = [];
    vi.stubGlobal(
      'fetch',
      vi.fn(async (/** @type {string} */ url) => {
        calls.push(url);
        if (url === '/u/alice/api/panels') {
          return jsonOk({
            enabled: ['artifacts'],
            custom: [],
            default: null,
            visible: ['artifacts'],
            active: null,
            labels: {},
          });
        }
        if (url === '/u/alice/api/artifact-server') {
          return jsonOk({ url: '/u/alice/panel/artifacts', available: true });
        }
        return jsonOk({});
      })
    );
    stubEventSource();

    const { initPanelManager } = await freshImport();
    await initPanelManager('panel-manager');

    expect(calls).toContain('/u/alice/api/panels');
    expect(calls).toContain('/u/alice/api/artifact-server');
  });

  test('empty prefix ⇒ byte-identical (unprefixed) URLs', async () => {
    window.__OSPREY_PREFIX__ = '';
    renderContainer();
    /** @type {string[]} */
    const calls = [];
    vi.stubGlobal(
      'fetch',
      vi.fn(async (/** @type {string} */ url) => {
        calls.push(url);
        if (url === '/api/panels') {
          return jsonOk({
            enabled: ['artifacts'],
            custom: [],
            default: null,
            visible: ['artifacts'],
            active: null,
            labels: {},
          });
        }
        if (url === '/api/artifact-server') {
          return jsonOk({ url: '/panel/artifacts', available: true });
        }
        return jsonOk({});
      })
    );
    stubEventSource();

    const { initPanelManager } = await freshImport();
    await initPanelManager('panel-manager');

    expect(calls).toContain('/api/panels');
    expect(calls).toContain('/api/artifact-server');
  });

  test("hands first contact the payload's tour object", async () => {
    renderContainer();
    const tour = { capabilities: ['run a scan'], logbook: true };
    vi.stubGlobal(
      'fetch',
      vi.fn(async (/** @type {string} */ url) => {
        if (url === '/api/panels') {
          return jsonOk({
            enabled: ['artifacts'],
            custom: [],
            default: null,
            visible: ['artifacts'],
            active: null,
            labels: {},
            tour,
          });
        }
        return jsonOk({});
      })
    );
    stubEventSource();

    const { initPanelManager } = await freshImport();
    await initPanelManager('panel-manager');

    expect(setFacts).toHaveBeenCalledWith(tour);
  });

  // The greeting renders whether or not the fetch answered, so the failure
  // path must SAY so rather than leave it waiting on a payload that never came.
  test('a failed /api/panels still settles first contact with null', async () => {
    renderContainer();
    vi.stubGlobal(
      'fetch',
      vi.fn(async (/** @type {string} */ url) => {
        if (url === '/api/panels') {
          return { ok: false, status: 500, statusText: 'Server Error', json: async () => ({}) };
        }
        return jsonOk({});
      })
    );
    stubEventSource();

    const { initPanelManager } = await freshImport();
    await initPanelManager('panel-manager');

    expect(setFacts).toHaveBeenCalledWith(null);
  });
});

describe('/api/files/events EventSource (via createEventSource)', () => {
  /**
   * @param {string|undefined} prefix
   * @param {string} expectedUrl
   */
  async function assertEventSourceUrl(prefix, expectedUrl) {
    if (prefix !== undefined) window.__OSPREY_PREFIX__ = prefix;
    renderContainer();
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        jsonOk({ enabled: [], custom: [], default: null, visible: [], active: null, labels: {} })
      )
    );
    const { opened } = stubEventSource();

    const { initPanelManager } = await freshImport();
    await initPanelManager('panel-manager');

    // The page's one events URL, tagged with its client id (api.js pageEventsUrl).
    expect(opened).toHaveLength(1);
    expect(opened[0].url).toMatch(new RegExp(`^${expectedUrl}\\?client_id=[^&]+$`));
  }

  test('prepends the prefix when set', async () => {
    await assertEventSourceUrl('/u/alice', '/u/alice/api/files/events');
  });

  test('is a no-op when the prefix is empty', async () => {
    await assertEventSourceUrl('', '/api/files/events');
  });
});

describe('/api/panel-focus POST on a user-initiated rail switch', () => {
  /**
   * @param {string|undefined} prefix
   * @param {string} expectedUrl
   */
  async function assertPanelFocusUrl(prefix, expectedUrl) {
    if (prefix !== undefined) window.__OSPREY_PREFIX__ = prefix;
    renderContainer();
    const artifactsUrl = `${prefix || ''}/panel/artifacts`;
    /** @type {{url: string, opts: any}[]} */
    const calls = [];
    vi.stubGlobal(
      'fetch',
      vi.fn(async (/** @type {string} */ url, /** @type {any} */ opts) => {
        calls.push({ url, opts });
        if (url.endsWith('/api/panels')) {
          return jsonOk({
            enabled: ['artifacts'],
            custom: [],
            default: null,
            visible: ['artifacts'],
            active: null,
            labels: {},
          });
        }
        if (url.endsWith('/api/artifact-server')) {
          return jsonOk({ url: artifactsUrl, available: true });
        }
        return jsonOk({ status: 'ok' });
      })
    );
    stubEventSource();

    const { initPanelManager } = await freshImport();
    await initPanelManager('panel-manager');

    const tab = /** @type {HTMLElement} */ (document.querySelector('[data-panel-id="artifacts"]'));
    await settled(() => expect(tab.classList.contains('disabled')).toBe(false));

    // Boot surfaces the only visible panel, so its entry starts active — and a
    // click on the ACTIVE entry is the retire-tile toggle, not a switch. Retire
    // first so the click under test is a genuine activation.
    await settled(() => expect(tab.classList.contains('active')).toBe(true));
    tab.click();
    await settled(() => expect(tab.classList.contains('active')).toBe(false));

    // Isolate the click's own request from the config/panels fetches above.
    calls.length = 0;
    tab.click();

    await settled(() => expect(calls.some((c) => c.url === expectedUrl)).toBe(true));
    const focusCall = calls.find((c) => c.url === expectedUrl);
    if (!focusCall) throw new Error('expected a panel-focus fetch call');
    expect(focusCall.opts).toMatchObject({ method: 'POST' });
    expect(JSON.parse(focusCall.opts.body)).toEqual({ panel: 'artifacts' });
  }

  test('prepends the prefix when set', async () => {
    await assertPanelFocusUrl('/u/alice', '/u/alice/api/panel-focus');
  });

  test('is a no-op when the prefix is empty', async () => {
    await assertPanelFocusUrl('', '/api/panel-focus');
  });

  test('re-clicking the ACTIVE entry retires its tile locally — no POST, entry stays', async () => {
    renderContainer();
    /** @type {{url: string, opts: any}[]} */
    const calls = [];
    vi.stubGlobal(
      'fetch',
      vi.fn(async (/** @type {string} */ url, /** @type {any} */ opts) => {
        calls.push({ url, opts });
        if (url.endsWith('/api/panels')) {
          return jsonOk({
            enabled: ['artifacts'],
            custom: [],
            default: null,
            visible: ['artifacts'],
            active: null,
            labels: {},
          });
        }
        if (url.endsWith('/api/artifact-server'))
          return jsonOk({ url: '/panel/artifacts', available: true });
        return jsonOk({ status: 'ok' });
      })
    );
    stubEventSource();

    const { initPanelManager } = await freshImport();
    await initPanelManager('panel-manager');

    const tab = /** @type {HTMLElement} */ (document.querySelector('[data-panel-id="artifacts"]'));
    await settled(() => expect(tab.classList.contains('active')).toBe(true));

    calls.length = 0;
    tab.click();
    await settled(() => expect(tab.classList.contains('active')).toBe(false));

    // Retiring a tile is LOCAL layout state: nothing is sent to the server,
    // and the panel keeps its rail membership so a second click brings it back.
    expect(calls.filter((c) => c.opts?.method === 'POST')).toEqual([]);
    expect(document.querySelector('[data-panel-id="artifacts"]')).toBe(tab);
  });
});

describe('iframe src: state.url arrives already server-prefixed (2.2) and must not be re-stripped/double-prefixed', () => {
  /**
   * @param {string|undefined} prefix
   * @param {string} expectedPath
   */
  async function assertIframeSrc(prefix, expectedPath) {
    if (prefix !== undefined) window.__OSPREY_PREFIX__ = prefix;
    renderContainer();
    const serverUrl = `${prefix || ''}/panel/artifacts`;
    vi.stubGlobal(
      'fetch',
      vi.fn(async (/** @type {string} */ url) => {
        if (url.endsWith('/api/panels')) {
          return jsonOk({
            enabled: ['artifacts'],
            custom: [],
            default: null,
            visible: ['artifacts'],
            active: null,
            labels: {},
          });
        }
        if (url.endsWith('/api/artifact-server')) {
          return jsonOk({ url: serverUrl, available: true });
        }
        return jsonOk({});
      })
    );
    stubEventSource();

    const { initPanelManager } = await freshImport();
    await initPanelManager('panel-manager');

    await settled(() => {
      expect(document.querySelector('iframe[data-panel-id="artifacts"]')).not.toBeNull();
    });
    const iframe = document.querySelector('iframe[data-panel-id="artifacts"]');
    if (!(iframe instanceof HTMLIFrameElement)) throw new Error('expected an iframe to be created');
    const parsed = new URL(iframe.src);
    expect(parsed.origin + parsed.pathname).toBe(`${window.location.origin}${expectedPath}`);
    expect(parsed.searchParams.get('embedded')).toBe('true');
  }

  test('preserves the /u/<user>/panel/<id> prefix (multi-user)', async () => {
    await assertIframeSrc('/u/alice', '/u/alice/panel/artifacts');
  });

  test('resolves to the unprefixed /panel/<id> when the prefix is empty', async () => {
    await assertIframeSrc('', '/panel/artifacts');
  });
});

describe('rail state for custom panels without a health endpoint', () => {
  test('a null-healthEndpoint panel is enabled, not left inert at the disabled default', async () => {
    window.__OSPREY_PREFIX__ = '';
    renderContainer();
    vi.stubGlobal(
      'fetch',
      vi.fn(async (/** @type {string} */ url) => {
        if (url === '/api/panels') {
          return jsonOk({
            enabled: [],
            custom: [
              {
                id: 'results',
                label: 'RESULTS',
                url: '/panel/results',
                healthEndpoint: null,
                path: '/results/',
              },
            ],
            default: null,
            visible: ['results'],
            active: null,
            labels: {},
          });
        }
        return jsonOk({});
      })
    );
    stubEventSource();

    const { initPanelManager } = await freshImport();
    await initPanelManager('panel-manager');

    const entry = document.querySelector('[data-panel-id="results"]');
    if (!(entry instanceof HTMLElement)) throw new Error('expected a results rail entry');
    expect(entry.classList.contains('disabled')).toBe(false);
  });
});

describe('rail reachability — a panel that stops answering dims, stays clickable, recovers', () => {
  beforeEach(() => stampFacilityZone(FACILITY_ZONE));

  afterEach(() => {
    vi.useRealTimers();
    stampFacilityZone(null);
  });

  /**
   * The tooltip time the rail renders for an epoch-ms instant: the facility
   * clock, naming its zone because it reads differently from the runner's.
   * @param {number} ms
   */
  const clock = (ms) =>
    new Intl.DateTimeFormat(undefined, {
      hour: 'numeric',
      minute: '2-digit',
      timeZone: FACILITY_ZONE,
      timeZoneName: 'short',
    }).format(ms);

  /**
   * Boot two health-polled panels under fake timers: 'system-health' answers
   * its /health until `down` names it, and 'jupyter' never answers at all.
   * The poll hook is the real one (panel-health.js), so each maintenance tick
   * is a 10 s advance.
   */
  async function bootPolled() {
    vi.useFakeTimers({
      toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'Date'],
    });
    vi.setSystemTime(new Date(2026, 0, 5, 9, 0, 0));
    window.__OSPREY_PREFIX__ = '';
    renderContainer();
    /** @type {Set<string>} */
    const down = new Set();
    vi.stubGlobal(
      'fetch',
      vi.fn(async (/** @type {string} */ url) => {
        if (url === '/api/panels') {
          return jsonOk({
            enabled: ['system-health', 'jupyter'],
            custom: [],
            default: null,
            visible: ['system-health', 'jupyter'],
            active: null,
            labels: {},
          });
        }
        if (url === '/api/system-health-server')
          return jsonOk({ url: '/panel/system-health', available: true });
        if (url === '/api/jupyter-server')
          return jsonOk({ url: '/panel/jupyter', available: true });
        if (url === '/panel/system-health/health' && !down.has('system-health'))
          return jsonOk({ status: 'ok' });
        if (url.startsWith('/panel/'))
          return { ok: false, status: 503, statusText: 'Down', json: async () => ({}) };
        return jsonOk({ status: 'ok' });
      })
    );
    stubEventSource();

    const mod = await freshImport();
    await mod.initPanelManager('panel-manager');
    const entry = /** @type {HTMLElement} */ (
      document.querySelector('[data-panel-id="system-health"]')
    );
    await settled(() => expect(entry.classList.contains('disabled')).toBe(false));
    // Past the fast startup retry: the panel is on the 10 s maintenance cadence.
    await vi.advanceTimersByTimeAsync(1000);
    return { entry, down, baseTitle: entry.title };
  }

  const tick = () => vi.advanceTimersByTimeAsync(10_000);

  test('healthy → miss → miss dims the entry; the next healthy settle brightens it', async () => {
    const { entry, down, baseTitle } = await bootPolled();

    down.add('system-health');
    await tick();
    const firstMiss = Date.now();
    // One miss is not enough: a single slow answer must not flicker the rail.
    expect(entry.classList.contains('unreachable')).toBe(false);

    await tick();
    expect(entry.classList.contains('unreachable')).toBe(true);
    expect(entry.title).toBe(`not answering since ${clock(firstMiss)}`);
    // Dimmed, not disabled: the entry stays clickable and keeps its menu.
    expect(entry.classList.contains('disabled')).toBe(false);

    down.delete('system-health');
    await tick();
    expect(entry.classList.contains('unreachable')).toBe(false);
    expect(entry.title).toBe(baseTitle);
  });

  test('a panel that never answered stays disabled and is never marked unreachable', async () => {
    await bootPolled();
    const jupyter = /** @type {HTMLElement} */ (
      document.querySelector('[data-panel-id="jupyter"]')
    );

    await vi.advanceTimersByTimeAsync(60_000);

    expect(jupyter.classList.contains('disabled')).toBe(true);
    expect(jupyter.classList.contains('unreachable')).toBe(false);
  });
});

describe('agent activity: rail badge/glow + the activity-strip seam', () => {
  /**
   * Boot the manager with a healthy 'artifacts' panel (no health endpoint, so
   * it enables synchronously) and an unhealthy 'ariel' panel (config endpoint
   * returns no url, so its entry stays disabled). Returns the SSE `emit`
   * injector, the fresh module, and the artifacts rail entry.
   */
  async function bootWithSSE() {
    window.__OSPREY_PREFIX__ = '';
    renderContainer();
    vi.stubGlobal(
      'fetch',
      vi.fn(async (/** @type {string} */ url) => {
        if (url === '/api/panels') {
          return jsonOk({
            enabled: ['artifacts', 'ariel'],
            custom: [],
            default: null,
            visible: ['artifacts', 'ariel'],
            active: null,
            labels: {},
          });
        }
        if (url === '/api/artifact-server') {
          return jsonOk({ url: '/panel/artifacts', available: true });
        }
        // /api/ariel-server (and any POST): no url ⇒ ariel stays unhealthy
        return jsonOk({});
      })
    );
    const { emit } = stubEventSource();

    const mod = await freshImport();
    await mod.initPanelManager('panel-manager');

    const artifacts = /** @type {HTMLElement} */ (
      document.querySelector('[data-panel-id="artifacts"]')
    );
    await settled(() => expect(artifacts.classList.contains('disabled')).toBe(false));
    return { emit, mod, artifacts };
  }

  test("agent_activity kind:'panel' with a rail entry sets badge + flash, no strip fallback", async () => {
    const { emit, mod, artifacts } = await bootWithSSE();
    const strip = vi.fn();
    mod.setActivityStripHandler(strip);

    emit({
      type: 'agent_activity',
      tool: 'read_file',
      target: { kind: 'panel', panel: 'artifacts' },
      ts: 1,
    });

    expect(artifacts.classList.contains('agent-attention')).toBe(true);
    expect(artifacts.classList.contains('agent-flash')).toBe(true);
    expect(strip).not.toHaveBeenCalled();
  });

  test("agent_activity kind:'panel' with an unknown id falls back to the strip handler", async () => {
    const { emit, mod } = await bootWithSSE();
    const strip = vi.fn();
    mod.setActivityStripHandler(strip);

    const frame = {
      type: 'agent_activity',
      tool: 'read_file',
      target: { kind: 'panel', panel: 'no-such-panel' },
      ts: 2,
    };
    emit(frame);

    expect(strip).toHaveBeenCalledTimes(1);
    expect(strip).toHaveBeenCalledWith(frame);
  });

  test("agent_activity kind:'channel' goes to the strip handler and leaves the rail alone", async () => {
    const { emit, mod } = await bootWithSSE();
    const strip = vi.fn();
    mod.setActivityStripHandler(strip);

    const frame = {
      type: 'agent_activity',
      tool: 'read_channel',
      target: { kind: 'channel', detail: 'SR01C:BPM1:X' },
      ts: 3,
    };
    emit(frame);

    expect(strip).toHaveBeenCalledTimes(1);
    expect(strip).toHaveBeenCalledWith(frame);
    expect(document.querySelector('.agent-attention')).toBeNull();
    expect(document.querySelector('.agent-flash')).toBeNull();
  });

  test("an agent open (panel_op source:'agent') glows transiently (no badge)", async () => {
    // Agent opens arrive as panel_op frames, which only a dock client runs.
    const { op } = await bootWorkspace();
    const ariel = /** @type {HTMLElement} */ (document.querySelector('[data-panel-id="ariel"]'));

    op('open', { member: 'ariel' });
    expect(ariel.classList.contains('agent-flash')).toBe(true);
    expect(ariel.classList.contains('agent-attention')).toBe(false);
  });

  test("a person's own surfacing has no agent styling", async () => {
    const { mod, artifacts } = await bootWithSSE();

    mod.activateTab('artifacts', { userInitiated: true });
    expect(artifacts.classList.contains('agent-flash')).toBe(false);
    expect(artifacts.classList.contains('agent-attention')).toBe(false);
  });

  test('activateTab clears the badge when the panel surfaces (a requested surfacing)', async () => {
    const { emit, mod, artifacts } = await bootWithSSE();

    emit({
      type: 'agent_activity',
      tool: 'read_file',
      target: { kind: 'panel', panel: 'artifacts' },
      ts: 4,
    });
    expect(artifacts.classList.contains('agent-attention')).toBe(true);

    mod.activateTab('artifacts', { userInitiated: true });
    expect(artifacts.classList.contains('agent-attention')).toBe(false);
  });

  test('an unhealthy-panel activation early-returns and keeps the badge', async () => {
    const { emit, mod } = await bootWithSSE();
    const ariel = /** @type {HTMLElement} */ (document.querySelector('[data-panel-id="ariel"]'));
    expect(ariel.classList.contains('disabled')).toBe(true); // never became healthy

    emit({
      type: 'agent_activity',
      tool: 'search_logbook',
      target: { kind: 'panel', panel: 'ariel' },
      ts: 5,
    });
    expect(ariel.classList.contains('agent-attention')).toBe(true);

    mod.activateTab('ariel'); // activateTab bails on !healthy
    expect(ariel.classList.contains('agent-attention')).toBe(true);
  });

  test('getActivePanel returns the surfaced panel id', async () => {
    const { mod } = await bootWithSSE();
    await settled(() => expect(mod.getActivePanel()).toBe('artifacts'));
  });
});

describe('getPanelStandaloneUrl — the "Open in a new window" target (context menu + palette)', () => {
  test('is null before config resolves a url, then the resolved url once it has', async () => {
    window.__OSPREY_PREFIX__ = '';
    renderContainer();
    vi.stubGlobal(
      'fetch',
      vi.fn(async (/** @type {string} */ url) => {
        if (url === '/api/panels') {
          return jsonOk({
            enabled: ['ariel'],
            custom: [],
            default: null,
            visible: ['ariel'],
            active: null,
            labels: {},
          });
        }
        if (url === '/api/ariel-server') {
          return jsonOk({ url: '/panel/ariel', available: true });
        }
        return jsonOk({});
      })
    );
    stubEventSource();

    const mod = await freshImport();
    expect(mod.getPanelStandaloneUrl('ariel')).toBeNull();

    await mod.initPanelManager('panel-manager');
    // initPanel()'s configEndpoint fetch settles after initPanelManager's own
    // returned promise (fire-and-forget, like the rail's disabled → enabled
    // transition other tests in this file wait on).
    const entry = /** @type {HTMLElement} */ (document.querySelector('[data-panel-id="ariel"]'));
    await settled(() => expect(entry.classList.contains('disabled')).toBe(false));

    expect(mod.getPanelStandaloneUrl('ariel')).toBe('/panel/ariel');
  });

  test('unknown panel id resolves to null', async () => {
    window.__OSPREY_PREFIX__ = '';
    renderContainer();
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        jsonOk({ enabled: [], custom: [], default: null, visible: [], active: null, labels: {} })
      )
    );
    stubEventSource();

    const mod = await freshImport();
    await mod.initPanelManager('panel-manager');

    expect(mod.getPanelStandaloneUrl('no-such-panel')).toBeNull();
  });

  test("suffixes a custom panel's catalog path onto its resolved url", async () => {
    window.__OSPREY_PREFIX__ = '';
    renderContainer();
    vi.stubGlobal(
      'fetch',
      vi.fn(async (/** @type {string} */ url) => {
        if (url === '/api/panels') {
          return jsonOk({
            enabled: [],
            custom: [
              {
                id: 'results',
                label: 'RESULTS',
                url: '/panel/results',
                healthEndpoint: null,
                path: '/results/',
              },
            ],
            default: null,
            visible: ['results'],
            active: null,
            labels: {},
          });
        }
        return jsonOk({});
      })
    );
    stubEventSource();

    const mod = await freshImport();
    await mod.initPanelManager('panel-manager');

    expect(mod.getPanelStandaloneUrl('results')).toBe('/panel/results/results/');
  });
});

describe('rail membership (launcher model: entry ⇔ member, never dimmed)', () => {
  /**
   * Boot with two enabled panels but only 'artifacts' a member (visible), so
   * 'ariel' starts in the "+" catalog with no rail entry.
   */
  async function bootMembership() {
    window.__OSPREY_PREFIX__ = '';
    renderContainer();
    vi.stubGlobal(
      'fetch',
      vi.fn(async (/** @type {string} */ url) => {
        if (url === '/api/panels') {
          return jsonOk({
            enabled: ['artifacts', 'ariel'],
            custom: [],
            default: null,
            visible: ['artifacts'],
            active: null,
            labels: {},
          });
        }
        if (url === '/api/artifact-server') {
          return jsonOk({ url: '/panel/artifacts', available: true });
        }
        if (url === '/api/ariel-server') {
          return jsonOk({ url: '/panel/ariel', available: true });
        }
        return jsonOk({ status: 'ok' });
      })
    );
    const { emit } = stubEventSource();
    const mod = await freshImport();
    await mod.initPanelManager('panel-manager');
    return { emit, mod };
  }

  /** @param {string} id */
  const entry = (id) => document.querySelector(`.panel-rail-button[data-panel-id="${id}"]`);

  test('only members render entries; non-members are in the hidden (catalog) list', async () => {
    const { mod } = await bootMembership();
    expect(entry('terminal')).not.toBeNull();
    expect(entry('artifacts')).not.toBeNull();
    expect(entry('ariel')).toBeNull();
    expect(mod.getHiddenPanels()).toEqual([{ id: 'ariel', label: 'Logbook' }]);
  });

  test('a panel_visibility show APPENDS the entry with its live health state', async () => {
    const { emit } = await bootMembership();
    // ariel's config resolved at init; its no-endpoint health settle may still
    // be pending — wait for artifacts' enable as the settle barrier.
    await settled(() => expect(entry('artifacts')?.classList.contains('disabled')).toBe(false));

    emit({ type: 'panel_visibility', panel: 'ariel', visible: true });

    const ariel = entry('ariel');
    expect(ariel).not.toBeNull();
    await settled(() => expect(ariel?.classList.contains('disabled')).toBe(false));
  });

  test('a panel_visibility hide REMOVES the entry and returns it to the catalog', async () => {
    const { emit, mod } = await bootMembership();
    const strip = vi.fn();
    mod.setActivityStripHandler(strip);

    emit({ type: 'panel_visibility', panel: 'artifacts', visible: false });

    // Removal stays synchronous with the frame — the entry is gone by the time
    // the emit returns, and an UNTAGGED (human-origin) change stays off the
    // activity strip, which only ever reports the agent.
    expect(entry('artifacts')).toBeNull();
    expect(mod.getHiddenPanels().map((p) => p.id)).toContain('artifacts');
    // Re-show rebuilds the entry.
    emit({ type: 'panel_visibility', panel: 'artifacts', visible: true });
    expect(entry('artifacts')).not.toBeNull();
    expect(strip).not.toHaveBeenCalled();
  });

  test('an agent hide reports itself on the strip; the entry is still removed', async () => {
    const { emit, mod } = await bootMembership();
    const strip = vi.fn();
    mod.setActivityStripHandler(strip);

    emit({ type: 'panel_visibility', panel: 'artifacts', visible: false, source: 'agent' });

    expect(entry('artifacts')).toBeNull(); // the rail glow had nothing to land on
    expect(strip).toHaveBeenCalledTimes(1);
    expect(strip.mock.calls[0][0]).toMatchObject({
      type: 'agent_activity',
      tool: 'remove_panel_from_rail',
      target: { kind: 'panel', panel: 'artifacts' },
    });
  });

  test('an agent show keeps the rail glow AND reports itself on the strip', async () => {
    const { emit, mod } = await bootMembership();
    const strip = vi.fn();
    mod.setActivityStripHandler(strip);

    emit({ type: 'panel_visibility', panel: 'ariel', visible: true, source: 'agent' });

    expect(entry('ariel')?.classList.contains('agent-flash')).toBe(true);
    expect(strip).toHaveBeenCalledTimes(1);
    expect(strip.mock.calls[0][0]).toMatchObject({
      type: 'agent_activity',
      tool: 'add_panel_to_rail',
      target: { kind: 'panel', panel: 'ariel' },
    });
  });

  test('an agent close takes the tile but LEAVES the rail entry', async () => {
    // The whole reason the on-screen axis is its own op: the operator must
    // still be able to bring the panel back in one click. An agent close is a
    // `close_tab` panel_op (the retired panel_close frame), run only by a dock.
    const { op, mod, holder } = await bootWorkspace({ visible: ['artifacts'] });
    const strip = vi.fn();
    mod.setActivityStripHandler(strip);

    op('close_tab', { key: 'artifacts' });

    expect(tileHolding(holder, 'artifacts')).toBeNull();
    expect(entry('artifacts')).not.toBeNull();
    expect(mod.getHiddenPanels().map((p) => p.id)).not.toContain('artifacts');
    // The client no longer synthesises the activity report: the server records
    // every op as agent activity (routes/panel_ops.py OP_ACTIVITY_TOOLS) and
    // broadcasts that frame itself.
    expect(strip).not.toHaveBeenCalled();
  });

  test('a human close is applied without reporting agent activity', async () => {
    const { op, mod, holder } = await bootWorkspace({ visible: ['artifacts'] });
    const strip = vi.fn();
    mod.setActivityStripHandler(strip);

    op('close_tab', { key: 'artifacts' }, null);

    expect(tileHolding(holder, 'artifacts')).toBeNull();
    expect(entry('artifacts')).not.toBeNull();
    expect(strip).not.toHaveBeenCalled();
  });

  test('the Agent (terminal) row is always present and enabled', async () => {
    await bootMembership();
    const term = entry('terminal');
    expect(term).not.toBeNull();
    expect(term?.classList.contains('disabled')).toBe(false);
  });
});

/** Catalog config endpoints for the panels these suites boot (panel-catalog.js).
 *  @type {Record<string, string>} */
const CONFIG_ENDPOINT = {
  artifacts: '/api/artifact-server',
  ariel: '/api/ariel-server',
  'channel-finder': '/api/channel-finder-server',
  'system-health': '/api/system-health-server',
};

/**
 * The service ids holding a dock tile, in the order the dock lays them out —
 * the fake rebuilds its groups from the holder's tree in reading order, so
 * left-to-right placement is what this lists.
 */
const dockedTiles = (/** @type {any} */ api) =>
  api.panels
    .filter((/** @type {any} */ p) => p.id.startsWith('iframe:'))
    .map((/** @type {any} */ p) => p.id.slice('iframe:'.length));

const DOCK_APPLY_PATH = '../../../src/osprey/interfaces/web_terminal/static/js/dock-apply.js';

/** The page's layout holder (the one the booted module graph shares). */
async function pageHolder() {
  const apply = await import(DOCK_APPLY_PATH);
  const holder = apply.getLayoutHolder();
  if (!holder) throw new Error('no layout holder');
  return holder;
}

/** Each tile's first tab, in reading order. @param {any} layout @returns {string[]} */
const tileOrder = (layout) => {
  if (!layout) return [];
  return 'tile' in layout ? [layout.tabs[0]] : layout.items.flatMap(tileOrder);
};

/** Every tile of a held layout, in tree (left-to-right) order. @param {any} node @returns {any[]} */
const tilesIn = (node) =>
  node === null ? [] : 'tabs' in node ? [node] : node.items.flatMap(tilesIn);

/** The held tile holding `key`, or null. @param {any} holder @param {string} key */
const tileHolding = (holder, key) =>
  tilesIn(holder.layout).find((t) => t.tabs.includes(key)) ?? null;

/** The keys of the held layout's tiles, each tile as its tab list, left to right. @param {any} holder */
const tileTabs = (holder) => tilesIn(holder.layout).map((t) => t.tabs);

const API_PATH = '../../../src/osprey/interfaces/web_terminal/static/js/api.js';
const POPOUT_PATH = '../../../src/osprey/interfaces/web_terminal/static/js/popout.js';

/** The open context-menu popover, or null when none is open. */
const contextMenu = () =>
  /** @type {HTMLElement | null} */ (document.querySelector('.rail-context-menu'));

/** Every menu row's visible verb, in order (dividers carry no label). */
const menuLabels = () =>
  [...document.querySelectorAll('.rail-context-label')].map((el) => el.textContent);

/**
 * The menu row reading exactly `label`, or undefined when the menu omits it.
 * @param {string} label
 * @returns {HTMLElement | undefined}
 */
const menuRow = (label) =>
  /** @type {HTMLElement | undefined} */ (
    [...document.querySelectorAll('.rail-context-item')].find(
      (el) => el.querySelector('.rail-context-label')?.textContent === label
    )
  );

/**
 * Right-click a sidebar row the way the browser does — on the row button or
 * on a child of it, letting the event bubble. The returned
 * event's `defaultPrevented` is the visible half of the handler's boolean:
 * true means a menu was claimed and the browser's native menu is suppressed.
 * @param {Element} el
 * @returns {MouseEvent}
 */
function rightClick(el) {
  const ev = new MouseEvent('contextmenu', {
    bubbles: true,
    cancelable: true,
    clientX: 40,
    clientY: 60,
  });
  el.dispatchEvent(ev);
  return ev;
}

/**
 * Ask a focused entry for its menu from the keyboard (the ContextMenu key),
 * the route the rail synthesises itself because macOS fires no `contextmenu`
 * event from the keyboard at all.
 * @param {Element} el
 * @returns {KeyboardEvent}
 */
function menuKey(el) {
  const ev = new KeyboardEvent('keydown', { key: 'ContextMenu', bubbles: true, cancelable: true });
  el.dispatchEvent(ev);
  return ev;
}

/**
 * Boot the manager against a live dock shell. Every listed panel is enabled and
 * (unless named in `unhealthy`) resolves a url, which is what marks these
 * endpoint-less panels healthy; `visible` is the server-owned rail membership,
 * defaulting to all of them. The FIRST listed panel is the catalog default, so
 * boot docks its tile. Resolves once every panel's config has settled and that
 * first tile exists, i.e. past every activation the boot itself performs.
 * `down` names health-polled panels whose /health stops answering; it is read
 * on every poll, so a test adds to it to take a panel dark mid-run.
 * `restored` names panels whose placeholder tile is already in the grid before
 * the manager boots — a dock layout restored from a previous page load.
 * `holdHealth` names health-polled panels whose first /health answer waits on
 * the returned `releaseHealth(id)`; `firstTile: false` returns without waiting
 * for a tile, for a boot whose landing is the thing under test.
 * @param {{ panels?: string[], visible?: string[], unhealthy?: string[],
 *           surface?: 'terminal', down?: Set<string>, restored?: string[],
 *           holdHealth?: Set<string>, firstTile?: boolean, defaultPanel?: string | null }} [opts]
 */
async function bootWorkspace({
  panels = ['artifacts', 'ariel'],
  visible,
  unhealthy = [],
  surface,
  down = new Set(),
  restored = [],
  holdHealth = new Set(),
  firstTile = true,
  defaultPanel = null,
} = {}) {
  if (surface) document.documentElement.setAttribute('data-agent-surface', surface);
  window.__OSPREY_PREFIX__ = '';
  vi.stubGlobal('ResizeObserver', FakeResizeObserver);
  // The overlay layer mounts in .main-container — without it the adapter
  // silently stays in fallback mode and no placeholder is ever created.
  // (No #dock-root: dock-sync's reverse half — a human dock gesture becoming a
  // POST — is its own module's contract, and leaving it unwired keeps the POST
  // assertions in these suites about panel-manager's own reporting.)
  document.body.innerHTML = `
    <div class="main-container">
      <nav id="panel-rail"></nav>
      <div id="panel-manager"><div id="panel-content"></div></div>
    </div>
  `;
  const members = visible ?? panels;
  /** @type {{url: string, opts: any}[]} */
  const calls = [];
  vi.stubGlobal(
    'fetch',
    vi.fn(async (/** @type {string} */ url, /** @type {any} */ o) => {
      calls.push({ url, opts: o });
      if (url === '/api/panels') {
        return jsonOk({
          enabled: panels,
          custom: [],
          default: defaultPanel,
          visible: members,
          active: null,
          labels: {},
        });
      }
      const id = panels.find((p) => CONFIG_ENDPOINT[p] === url);
      if (id) return jsonOk(unhealthy.includes(id) ? {} : { url: `/panel/${id}`, available: true });
      const polled = /^\/panel\/([^/]+)\/health$/.exec(url)?.[1];
      if (polled && holdHealth.has(polled)) await heldHealth(polled);
      if (polled && down.has(polled)) {
        return { ok: false, status: 503, statusText: 'Down', json: async () => ({}) };
      }
      return jsonOk({ status: 'ok' });
    })
  );
  /** @type {Map<string, () => void>} */
  const releases = new Map();
  /** @param {string} id */
  const heldHealth = (id) =>
    new Promise((resolve) => {
      releases.set(id, () => resolve(undefined));
    });
  /** @param {string} id */
  const releaseHealth = (id) => {
    holdHealth.delete(id);
    releases.get(id)?.();
  };
  const { emit } = stubEventSource();

  const api = makeDockApi();
  dockState.api = api;

  const mod = await freshImport();
  // The page's holder over the fake dock, showing the terminal and every
  // restored tile — what dock-workspace builds before the manager boots.
  const apply = await import(DOCK_APPLY_PATH);
  const holder = apply.initLayoutHolder({ api, guard: (/** @type {() => any} */ fn) => fn() });
  // dock-workspace (mocked here) tells the holder which panels left for their
  // own windows; the holder's pop-outs ride beside the layout in every report.
  const { onPopoutChange } = await import(POPOUT_PATH);
  onPopoutChange((/** @type {string} */ id, /** @type {boolean} */ out) =>
    holder.setPoppedOut(id, out)
  );
  holder.restore(
    restored.length
      ? rowOf(...restored.map((id) => [id]), { tile: 'tt', tabs: ['terminal'] })
      : TERMINAL_LAYOUT
  );
  await phase('initPanelManager', () => mod.initPanelManager('panel-manager'));
  // The configEndpoint fetches settle after initPanelManager's own promise: a
  // healthy panel resolves its url on that settle, an unhealthy one never does
  // (wait on the fetch itself so both land before the frame under test).
  await phase('panel config', async () => {
    for (const id of panels) {
      if (unhealthy.includes(id)) {
        await settled(() => expect(calls.some((c) => c.url === CONFIG_ENDPOINT[id])).toBe(true));
      } else {
        await settled(() => expect(mod.getPanelStandaloneUrl(id)).toBe(`/panel/${id}`));
      }
    }
  });
  if (firstTile) {
    await phase('first tile', () =>
      settled(() => expect(api.getPanel(`iframe:${panels[0]}`)).not.toBeNull())
    );
  }
  calls.length = 0;
  const { getPageClientId } = await import(API_PATH);
  let seq = 0;
  /**
   * Relay a `panel_op` addressed to this dock client — how the server delivers
   * an agent's (or a preset's) dock change.
   * @param {string} name @param {any} args @param {'agent' | null} [source]
   */
  const op = (name, args, source = 'agent') =>
    emit({
      type: 'panel_op',
      op: name,
      args,
      target_client: getPageClientId(),
      request_id: `r${++seq}`,
      ...(source ? { source } : {}),
    });
  return { api, emit, mod, calls, releaseHealth, holder, op };
}

/** POSTs the client sent — an applied SSE frame must never re-report focus. */
const posts = (/** @type {{url: string, opts: any}[]} */ calls) =>
  calls.filter((c) => c.opts?.method === 'POST').map((c) => c.url);

describe("agent open (panel_op op:'open') — focus or open, never evict", () => {
  // An agent changes the dock only through a panel_op frame addressed to this
  // dock client (panel-placement.js, ONE MUTATION CHANNEL); the retired
  // `panel_focus source:'agent'` switch no longer places anything. An open is
  // OSPREY's open rule (dock-open.js): a new TAB of the last service tile used,
  // never a takeover — the operator's panel stays where it is.
  afterEach(() => {
    document.documentElement.removeAttribute('data-agent-surface');
  });

  test('an undocked rail member opens as a new tab of the service tile — nothing is evicted', async () => {
    const { holder, op, calls } = await bootWorkspace();
    const artifactsTile = tileHolding(holder, 'artifacts').tile;

    op('open', { member: 'ariel' });

    // The operator's panel survives in its tile (now a hidden tab of it)...
    expect(tileHolding(holder, 'artifacts').tile).toBe(artifactsTile);
    // ...and ariel arrived once, as a tab of that same service tile, showing.
    expect(
      tileTabs(holder)
        .flat()
        .filter((k) => k === 'ariel')
    ).toHaveLength(1);
    expect(tileHolding(holder, 'ariel')).toMatchObject({ tile: artifactsTile, showing: 'ariel' });
    expect(tileHolding(holder, 'terminal').tabs).toEqual(['terminal']);
    // It holds the focus, and the only thing POSTed is the op's outcome —
    // never a focus report echoing the agent's own change back.
    expect(document.getElementById('panel-manager')?.dataset.activePanel).toBe('ariel');
    await settled(() => expect(posts(calls)).toEqual(['/api/panel-op-result']));
  });

  test('a panel that already holds a tab is only shown — no second tab, no move', async () => {
    const { holder, op, calls } = await bootWorkspace();
    op('open', { member: 'ariel' });
    const before = structuredClone(tileTabs(holder));
    const artifactsTile = tileHolding(holder, 'artifacts').tile;
    await settled(() => expect(posts(calls)).toHaveLength(1));
    calls.length = 0;

    op('open', { member: 'artifacts' });

    expect(tileTabs(holder)).toEqual(before);
    expect(tileHolding(holder, 'artifacts')).toMatchObject({
      tile: artifactsTile,
      showing: 'artifacts',
    });
    expect(holder.focusedTile).toBe(artifactsTile);
    expect(document.getElementById('panel-manager')?.dataset.activePanel).toBe('artifacts');
    await settled(() => expect(posts(calls)).toEqual(['/api/panel-op-result']));
  });

  test('a NON-member opens without a local rail edit; the server’s visibility echo adds its entry', async () => {
    const { holder, emit, op, mod } = await bootWorkspace({ visible: ['artifacts'] });
    expect(document.querySelector('.panel-rail-button[data-panel-id="ariel"]')).toBeNull();

    op('open', { member: 'ariel' });

    // Rail membership is the server's: the op runner never edits the rail.
    expect(tileHolding(holder, 'ariel')).not.toBeNull();
    expect(tileHolding(holder, 'artifacts')).not.toBeNull();
    expect(document.querySelector('.panel-rail-button[data-panel-id="ariel"]')).toBeNull();

    // The server follows the recorded result with panel_visibility.
    emit({ type: 'panel_visibility', panel: 'ariel', visible: true, source: 'agent' });
    const entry = document.querySelector('.panel-rail-button[data-panel-id="ariel"]');
    expect(entry).not.toBeNull();
    expect(entry?.classList.contains('disabled')).toBe(false);
    expect(mod.getHiddenPanels().map((p) => p.id)).not.toContain('ariel');
  });

  test('an UNHEALTHY target opens no tab — a tab it could never fill would linger empty', async () => {
    const { holder, op, calls } = await bootWorkspace({ unhealthy: ['ariel'] });
    const before = holder.layout;

    op('open', { member: 'ariel' });

    expect(holder.layout).toBe(before);
    expect(tileHolding(holder, 'ariel')).toBeNull();
    // The refusal is answered by name; artifacts keeps the focus.
    await settled(() => expect(posts(calls)).toEqual(['/api/panel-op-result']));
    expect(JSON.parse(calls.find((c) => c.opts?.method === 'POST')?.opts.body)).toMatchObject({
      refusal: 'panel_unhealthy',
    });
    expect(document.getElementById('panel-manager')?.dataset.activePanel).toBe('artifacts');
  });

  test('an unknown panel id touches neither the rail nor the grid', async () => {
    const { holder, op } = await bootWorkspace();
    const before = holder.layout;

    op('open', { member: 'no-such-panel' });

    expect(holder.layout).toBe(before);
    expect(tileHolding(holder, 'no-such-panel')).toBeNull();
    expect(document.querySelector('.panel-rail-button[data-panel-id="no-such-panel"]')).toBeNull();
  });

  test('with EVERY service tile retired, the open re-anchors left of the terminal', async () => {
    const { api, holder, emit, op } = await bootWorkspace({ visible: ['artifacts'] });
    // Retire the only service tile the way a hide does, leaving the dock holding
    // just the terminal — the state where "the service tile" would be none.
    emit({ type: 'panel_visibility', panel: 'artifacts', visible: false });
    expect(dockedTiles(api)).toEqual([]);
    expect(tileTabs(holder)).toEqual([['terminal']]);

    op('open', { member: 'ariel' });

    // The first-tile rule wins: left of the terminal at the classic 0.6/0.4
    // split, not a tab of (or a default split off) the terminal's own tile.
    expect(holder.layout.split).toBe('row');
    expect(tileTabs(holder)).toEqual([['ariel'], ['terminal']]);
    expect(holder.layout.items[0].size).toBeCloseTo(0.6, 3);
    expect(holder.layout.items[1].size).toBeCloseTo(0.4, 3);
    expect(document.getElementById('panel-manager')?.dataset.activePanel).toBe('ariel');
  });

  test("the server's visibility echo after an open is idempotent", async () => {
    const { emit, op } = await bootWorkspace({ visible: ['artifacts'] });

    op('open', { member: 'ariel' });
    emit({ type: 'panel_visibility', panel: 'ariel', visible: true, source: 'agent' });
    const added = document.querySelector('.panel-rail-button[data-panel-id="ariel"]');
    expect(added).not.toBeNull();

    // A repeated echo (another tab's, a reconnect's) must leave the very same
    // entry in place rather than rebuild or duplicate it.
    emit({ type: 'panel_visibility', panel: 'ariel', visible: true, source: 'agent' });

    const entries = document.querySelectorAll('.panel-rail-button[data-panel-id="ariel"]');
    expect(entries).toHaveLength(1);
    expect(entries[0]).toBe(added);
    expect(entries[0].classList.contains('disabled')).toBe(false);
  });

  test('fallback mode (no dock shell) refuses the op — nothing is surfaced or placed', async () => {
    window.__OSPREY_PREFIX__ = '';
    renderContainer();
    /** @type {{url: string, opts: any}[]} */
    const calls = [];
    vi.stubGlobal(
      'fetch',
      vi.fn(async (/** @type {string} */ url, /** @type {any} */ o) => {
        calls.push({ url, opts: o });
        if (url === '/api/panels') {
          return jsonOk({
            enabled: ['artifacts', 'ariel'],
            custom: [],
            default: null,
            visible: ['artifacts'],
            active: null,
            labels: {},
          });
        }
        if (url === '/api/artifact-server')
          return jsonOk({ url: '/panel/artifacts', available: true });
        if (url === '/api/ariel-server') return jsonOk({ url: '/panel/ariel', available: true });
        return jsonOk({ status: 'ok' });
      })
    );
    const { emit } = stubEventSource();

    const mod = await freshImport();
    await mod.initPanelManager('panel-manager');
    await settled(() => expect(mod.getPanelStandaloneUrl('ariel')).toBe('/panel/ariel'));
    const { getPageClientId } = await import(API_PATH);
    await settled(() =>
      expect(document.getElementById('panel-manager')?.dataset.activePanel).toBe('artifacts')
    );
    calls.length = 0;

    emit({
      type: 'panel_op',
      op: 'open',
      args: { member: 'ariel' },
      target_client: getPageClientId(),
      request_id: 'r1',
      source: 'agent',
    });

    // A page without a dock is never the dock client: the op is answered with
    // a refusal and the operator's view is left alone.
    await settled(() => expect(posts(calls)).toEqual(['/api/panel-op-result']));
    expect(JSON.parse(calls[0].opts.body)).toMatchObject({ refusal: 'no_dock_client' });
    expect(document.getElementById('panel-manager')?.dataset.activePanel).toBe('artifacts');
    expect(document.querySelector('.panel-rail-button[data-panel-id="ariel"]')).toBeNull();
  });
});

describe('the boot landing — which panel takes the empty slot, and when', () => {
  const activeStamp = () => document.getElementById('panel-manager')?.dataset.activePanel ?? null;
  /** Let every pending microtask and the next macrotask run. */
  const tick = () => new Promise((resolve) => setTimeout(resolve, 30));

  test('a pinned default waits for its first health answer before another panel takes the slot', async () => {
    // system-health is the profile-pinned default and health-polled; artifacts
    // is healthy the moment its config answers. The default's backend is slow
    // to answer its first probe.
    const { api, releaseHealth } = await bootWorkspace({
      panels: ['artifacts', 'system-health'],
      defaultPanel: 'system-health',
      holdHealth: new Set(['system-health']),
      firstTile: false,
    });
    await tick();
    // Nothing has landed: the default's verdict is still out, and another
    // panel's health must not take the slot the default may still claim.
    expect(dockedTiles(api)).toEqual([]);
    expect(activeStamp()).toBeNull();

    releaseHealth('system-health');

    await settled(() => expect(dockedTiles(api)).toEqual(['system-health']));
    expect(activeStamp()).toBe('system-health');
  });

  test('a stored layout comes back whole: the landing waits for it and lands on what it shows', async () => {
    dockState.bootSettled = false;
    const { api } = await bootWorkspace({ firstTile: false });
    await tick();
    // Both panels are healthy, yet nothing has landed: the restore is pending.
    expect(dockedTiles(api)).toEqual([]);

    // The stored layout arrives: ariel alone beside the agent pane; artifacts,
    // the profile default, was closed when the layout was saved.
    (await pageHolder()).restore(rowOf(['ariel'], { tile: 'tt', tabs: ['terminal'] }));
    dockState.redock?.();
    settleBootLayout();

    await settled(() => expect(activeStamp()).toBe('ariel'));
    const layout = (await pageHolder()).layout;
    expect(tileOrder(layout)).toEqual(['ariel', 'terminal']);
    expect(layout.items[0].tabs).toEqual(['ariel']);
    expect(dockedTiles(api)).toEqual(['ariel']);
    const iframe = /** @type {HTMLElement | null} */ (
      document.querySelector('iframe.panel-iframe[data-panel-id="ariel"]')
    );
    expect(iframe?.style.display).not.toBe('none');
  });
});

describe("panel_op op:'open_arrangement' — the declarative whole-workspace rebuild", () => {
  // The retired `panel_arrange` frame is now an `open_arrangement` op
  // (panel-placement.js) run through dock-open.js's arrange composer: the
  // service tiles become exactly the arrangement, the terminal keeps its place.
  afterEach(() => {
    document.documentElement.removeAttribute('data-agent-surface');
  });

  /** @param {string} id */
  const entry = (id) => document.querySelector(`.panel-rail-button[data-panel-id="${id}"]`);
  const activeStamp = () => document.getElementById('panel-manager')?.dataset.activePanel ?? null;

  const THREE = ['artifacts', 'ariel', 'channel-finder'];

  /** An arrangement of `members`, one tile each, left to right. @param {string[]} members */
  const row = (members) => ({
    name: 'arrangement',
    members,
    // A split needs two items; a lone member is a single tile.
    layout:
      members.length === 1
        ? { tabs: [0] }
        : { split: 'row', items: members.map((_, i) => ({ tabs: [i] })) },
  });

  test('converges to exactly the requested tiles, left to right', async () => {
    const { holder, op } = await bootWorkspace({ panels: THREE });

    op('open_arrangement', row(['channel-finder', 'ariel']));

    // The services sit left of the terminal, each arranged panel in a tile of
    // its own, in the order asked for; the unlisted artifacts left the dock.
    expect(tileTabs(holder)).toEqual([['channel-finder'], ['ariel'], ['terminal']]);
  });

  test('converges from a DIFFERENT start layout to the same end state', async () => {
    const { holder, op } = await bootWorkspace({ panels: THREE });
    // Start with all three open in a different shape than the arrangement asks
    // for, so the rebuild has both a tab to drop and tabs to reorder.
    op('open', { member: 'ariel' });
    op('open', { member: 'channel-finder' });
    expect(tileTabs(holder).flat().sort()).toEqual([
      'ariel',
      'artifacts',
      'channel-finder',
      'terminal',
    ]);

    op('open_arrangement', row(['channel-finder', 'ariel']));

    // Same end state as the clean-start case: the region is rebuilt, not
    // diffed, so a differing start layout cannot leave order behind.
    expect(tileTabs(holder)).toEqual([['channel-finder'], ['ariel'], ['terminal']]);
    // The dropped tile keeps its rail membership — a tile close is not a hide.
    expect(entry('artifacts')).not.toBeNull();
  });

  test('the arrangement focuses its first tile; the tiles all get their iframes', async () => {
    const { holder, op } = await bootWorkspace({ panels: THREE });

    op('open_arrangement', row(['channel-finder', 'ariel']));

    expect(activeStamp()).toBe('channel-finder');
    expect(holder.focusedTile).toBe(tileHolding(holder, 'channel-finder').tile);
    // Every arranged tile is surfaced, not just the focused one: a docked tile
    // whose panel was never activated would show an empty placeholder.
    for (const id of ['channel-finder', 'ariel']) {
      expect(document.querySelector(`iframe[data-panel-id="${id}"]`)).not.toBeNull();
    }
  });

  test('an unhealthy first member is skipped, so focus lands on the first healthy tile', async () => {
    const { op } = await bootWorkspace({ panels: THREE, unhealthy: ['channel-finder'] });

    op('open_arrangement', row(['channel-finder', 'ariel']));

    expect(activeStamp()).toBe('ariel');
  });

  test('no healthy member at all changes nothing rather than stranding the focus', async () => {
    const { holder, op } = await bootWorkspace({
      panels: THREE,
      unhealthy: ['ariel', 'channel-finder'],
    });
    expect(activeStamp()).toBe('artifacts');
    const before = holder.layout;

    op('open_arrangement', row(['channel-finder', 'ariel']));

    // Nothing could be built, so the dock is left as it was: artifacts keeps
    // its tile and the accent, and no unhealthy panel is surfaced.
    expect(holder.layout).toBe(before);
    expect(activeStamp()).toBe('artifacts');
    expect(document.querySelector('.artifacts-empty-state')).toBeNull();
  });

  test('an UNHEALTHY listed panel gets no tile — only its rail entry', async () => {
    const { api, holder, op } = await bootWorkspace({
      panels: THREE,
      unhealthy: ['channel-finder'],
    });

    op('open_arrangement', row(['channel-finder', 'ariel']));

    // A tile it could never fill would sit empty for as long as anything else
    // holds focus, and nothing comes back for it.
    expect(tileHolding(holder, 'channel-finder')).toBeNull();
    expect(api.getPanel('iframe:channel-finder')).toBeNull();
    expect(dockedTiles(api)).toEqual(['ariel']);
    // Its membership is untouched, so it stays one rail click away.
    expect(entry('channel-finder')).not.toBeNull();
  });

  test('a skipped tile never becomes the anchor the next one splits against', async () => {
    const { holder, op } = await bootWorkspace({ panels: THREE, unhealthy: ['channel-finder'] });

    // The unhealthy panel is listed FIRST, so a naive walk would split ariel
    // against a tile that does not exist, or off the terminal.
    op('open_arrangement', row(['channel-finder', 'ariel']));

    expect(tileTabs(holder)).toEqual([['ariel'], ['terminal']]);
    expect(tileHolding(holder, 'ariel').tile).not.toBe(tileHolding(holder, 'terminal').tile);
  });

  test('a tile still on screen is never painted over as an empty workspace', async () => {
    // The production route to "docked but UNHEALTHY": a panel healthy long
    // enough to be docked, which then stops answering its health poll.
    vi.useFakeTimers({
      toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'Date'],
    });
    try {
      const down = new Set();
      const { api, op } = await bootWorkspace({ panels: ['artifacts', 'system-health'], down });
      await vi.advanceTimersByTimeAsync(1000); // onto the 10 s maintenance cadence
      rightClick(/** @type {Element} */ (entry('system-health')));
      /** @type {HTMLElement} */ (menuRow('Open in a new tile')).click();
      /** @type {HTMLElement} */ (entry('artifacts')).click(); // the operator looks back
      expect(dockedTiles(api).sort()).toEqual(['artifacts', 'system-health']);
      expect(activeStamp()).toBe('artifacts');

      down.add('system-health');
      await vi.advanceTimersByTimeAsync(20_000);
      expect(entry('system-health')?.classList.contains('unreachable')).toBe(true);

      // Arrange onto the dark panel alone: nothing listed can be focused, and
      // the panel that held the view (artifacts) is not listed.
      op('open_arrangement', row(['system-health']));

      // Its existing tile survives the rebuild...
      expect(dockedTiles(api)).toEqual(['system-health']);
      // ...so the accent clears, but the pane is NOT painted empty over it.
      expect(activeStamp()).toBeNull();
      expect(document.querySelector('.artifacts-empty-state')).toBeNull();
    } finally {
      vi.useRealTimers();
    }
  });

  test('applying an arrangement reports only its outcome to the server', async () => {
    const { op, calls } = await bootWorkspace({ panels: THREE });

    op('open_arrangement', row(['channel-finder', 'ariel']));

    // The op's result is POSTed once; the surfacing it causes is never
    // reported back as a person's focus.
    await settled(() => expect(posts(calls)).toEqual(['/api/panel-op-result']));
    expect(JSON.parse(calls.find((c) => c.opts?.method === 'POST')?.opts.body)).toMatchObject({
      layout: expect.any(Object),
    });
  });

  test("source:'agent' glows the arranged entries transiently, without the badge", async () => {
    const { op } = await bootWorkspace({ panels: THREE });

    op('open_arrangement', row(['channel-finder', 'ariel']));

    expect(entry('ariel')?.classList.contains('agent-flash')).toBe(true);
    expect(entry('channel-finder')?.classList.contains('agent-flash')).toBe(true);
    expect(entry('ariel')?.classList.contains('agent-attention')).toBe(false);
    expect(entry('artifacts')?.classList.contains('agent-flash')).toBe(false);
  });

  test('unknown ids are dropped rather than arranged as ghost tiles', async () => {
    const { api, op } = await bootWorkspace({ panels: THREE });

    op('open_arrangement', row(['ariel', 'no-such-panel']));

    expect(dockedTiles(api)).toEqual(['ariel']);
    expect(entry('no-such-panel')).toBeNull();
  });
});

describe('tile notice — a docked panel that stops answering says so over its own tile', () => {
  afterEach(() => {
    vi.useRealTimers();
    document.documentElement.removeAttribute('data-agent-surface');
  });

  /** @param {string} id */
  const entry = (id) => document.querySelector(`.panel-rail-button[data-panel-id="${id}"]`);
  /** @param {string} id @returns {HTMLElement | null} */
  const notice = (id) => document.querySelector(`.tile-notice[data-panel="${id}"]`);
  /** @param {string} id */
  const railSince = (id) =>
    /** @type {HTMLElement} */ (entry(id)).title.replace('not answering since ', '');
  /** @param {string} id @returns {HTMLIFrameElement} */
  const frame = (id) =>
    /** @type {HTMLIFrameElement} */ (document.querySelector(`iframe[data-panel-id="${id}"]`));

  /**
   * Boot artifacts and system-health, both docked, on the 10 s maintenance
   * cadence: the recipe of the arrangement's "never painted over" test.
   * @param {Set<string>} down
   */
  async function bootBothDocked(down) {
    vi.useFakeTimers({
      toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'Date'],
    });
    const booted = await bootWorkspace({ panels: ['artifacts', 'system-health'], down });
    await vi.advanceTimersByTimeAsync(1000); // onto the 10 s maintenance cadence
    rightClick(/** @type {Element} */ (entry('system-health')));
    /** @type {HTMLElement} */ (menuRow('Open in a new tile')).click();
    /** @type {HTMLElement} */ (entry('artifacts')).click(); // the operator looks back
    expect(dockedTiles(booted.api).sort()).toEqual(['artifacts', 'system-health']);
    return booted;
  }

  test('the notice appears on the same poll that dims the rail, with the same time', async () => {
    const down = new Set();
    await bootBothDocked(down);

    down.add('system-health');
    await vi.advanceTimersByTimeAsync(10_000);
    // One miss flickers nothing.
    expect(entry('system-health')?.classList.contains('unreachable')).toBe(false);
    expect(notice('system-health')?.hasAttribute('data-unreachable')).toBe(false);

    await vi.advanceTimersByTimeAsync(10_000);
    expect(entry('system-health')?.classList.contains('unreachable')).toBe(true);
    const el = /** @type {HTMLElement} */ (notice('system-health'));
    expect(el.hasAttribute('data-unreachable')).toBe(true);
    expect(el.textContent).toContain(railSince('system-health'));
    expect(notice('artifacts')?.hasAttribute('data-unreachable')).toBe(false);
  });

  test('the first answer clears it', async () => {
    const down = new Set();
    await bootBothDocked(down);
    const before = frame('system-health');
    const src = before.src;

    down.add('system-health');
    await vi.advanceTimersByTimeAsync(20_000);
    expect(notice('system-health')?.hasAttribute('data-unreachable')).toBe(true);

    down.delete('system-health');
    await vi.advanceTimersByTimeAsync(10_000);
    expect(entry('system-health')?.classList.contains('unreachable')).toBe(false);
    const el = /** @type {HTMLElement} */ (notice('system-health'));
    expect(el.hasAttribute('data-unreachable')).toBe(false);
    expect(el.textContent).toBe('');
    // The view resumes in place: the same frame, never reloaded.
    expect(frame('system-health')).toBe(before);
    expect(frame('system-health').src).toBe(src);
  });

  test('a panel that never answered gets no notice', async () => {
    vi.useFakeTimers({
      toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'Date'],
    });
    // Its config resolves a url, but its /health never answers.
    await bootWorkspace({
      panels: ['artifacts', 'system-health'],
      down: new Set(['system-health']),
    });

    await vi.advanceTimersByTimeAsync(60_000);

    expect(document.querySelector('.tile-notice[data-unreachable]')).toBeNull();
    expect(entry('system-health')?.classList.contains('disabled')).toBe(true);
    expect(entry('system-health')?.classList.contains('unreachable')).toBe(false);
  });
  test('a jump into a panel that is not answering waits for its next answer', async () => {
    const down = new Set();
    const { op } = await bootBothDocked(down);
    const src = frame('system-health').src;

    down.add('system-health');
    await vi.advanceTimersByTimeAsync(10_000);
    op('open', { member: 'system-health', url: '/panel/system-health/x' });
    // The last good page stays: no error page under the notice.
    expect(frame('system-health').src).toBe(src);

    down.delete('system-health');
    await vi.advanceTimersByTimeAsync(10_000);
    expect(new URL(frame('system-health').src).pathname).toBe('/panel/system-health/x');
  });
});

describe('which panels are served: PanelState.url, followed by the sidebar', () => {
  const AVAILABILITY =
    '../../../src/osprey/interfaces/web_terminal/static/js/panel-availability.js';

  /**
   * Boot with a mutable answer for the artifacts config endpoint and a stub
   * whose `open()` is the event stream (re)opening.
   * @param {() => Response | Promise<Response>} [firstAnswer] - the first read's answer
   */
  async function bootFollowing(firstAnswer) {
    window.__OSPREY_PREFIX__ = '';
    renderContainer();
    const server = { artifacts: /** @type {string | null} */ (null) };
    let answered = false;
    vi.stubGlobal(
      'fetch',
      vi.fn(async (/** @type {string} */ url) => {
        if (url === '/api/panels') {
          return jsonOk({ enabled: ['artifacts'], custom: [], visible: ['artifacts'], labels: {} });
        }
        if (url === '/api/artifact-server') {
          if (firstAnswer && !answered) {
            answered = true;
            return firstAnswer();
          }
          return jsonOk({ url: server.artifacts, available: server.artifacts !== null });
        }
        return jsonOk({ status: 'ok' });
      })
    );
    /** @type {{ onopen?: (() => void) | null }[]} */
    const sources = [];
    class FakeEventSource {
      constructor() {
        /** @type {(() => void) | null} */
        this.onopen = null;
        sources.push(this);
      }
      close() {}
    }
    vi.stubGlobal('EventSource', FakeEventSource);

    const mod = await freshImport();
    const availability = await import(AVAILABILITY);
    const calls = { started: 0, stopped: 0 };
    availability.whilePanelServed(
      'artifacts',
      () => (calls.started += 1),
      () => (calls.stopped += 1)
    );
    return {
      mod,
      server,
      calls,
      open: () => {
        for (const s of sources) s.onopen?.();
      },
    };
  }

  /** Let every read the page has started land, before saying nothing happened. */
  const drained = () => new Promise((resolve) => setTimeout(resolve, 20));

  /** The Artifacts row. */
  const row = () =>
    /** @type {HTMLElement} */ (
      document.querySelector('.panel-rail-button[data-panel-id="artifacts"]')
    );

  test('enabled but not launched: a config answer with no url starts nothing', async () => {
    const { mod, open, calls } = await bootFollowing();
    await mod.initPanelManager('panel-manager');
    await drained();
    open();
    await drained();

    expect(calls).toEqual({ started: 0, stopped: 0 });
    expect(row().classList.contains('disabled')).toBe(true);
  });

  test('a panel the deployment does not enable starts nothing', async () => {
    const { mod, calls } = await bootFollowing();
    vi.stubGlobal(
      'fetch',
      vi.fn(async (/** @type {string} */ url) =>
        url === '/api/panels'
          ? jsonOk({ enabled: [], custom: [], visible: [], labels: {} })
          : jsonOk({ status: 'ok' })
      )
    );
    await mod.initPanelManager('panel-manager');
    await drained();

    expect(calls).toEqual({ started: 0, stopped: 0 });
  });

  test('a served panel starts its followers once its config answers', async () => {
    const { mod, server, calls, open } = await bootFollowing();
    server.artifacts = '/panel/artifacts';
    await mod.initPanelManager('panel-manager');
    await settled(() => expect(calls.started).toBe(1));

    open();
    open();
    await drained();

    expect(calls).toEqual({ started: 1, stopped: 0 });
    expect(row().classList.contains('disabled')).toBe(false);
  });

  test('a config read that failed at boot starts them when a retry answers', async () => {
    const { mod, server, calls } = await bootFollowing(() => {
      throw new Error('offline');
    });
    server.artifacts = '/panel/artifacts';
    await mod.initPanelManager('panel-manager');
    await drained();
    expect(calls.started).toBe(0);

    await vi.waitFor(() => expect(calls.started).toBe(1), { timeout: 3000, interval: 20 });
  });

  test('a reopened stream starts a panel the restarted hub now serves', async () => {
    const { mod, server, calls, open } = await bootFollowing();
    await mod.initPanelManager('panel-manager');
    await drained();
    open();
    await drained();
    expect(calls.started).toBe(0);

    server.artifacts = '/panel/artifacts';
    open();

    await settled(() => expect(calls.started).toBe(1));
    await settled(() => expect(row().classList.contains('disabled')).toBe(false));
  });

  test('a reopened stream stops a panel the restarted hub no longer serves', async () => {
    const { mod, server, calls, open } = await bootFollowing();
    server.artifacts = '/panel/artifacts';
    await mod.initPanelManager('panel-manager');
    await settled(() => expect(calls.started).toBe(1));
    open();
    await drained();

    server.artifacts = null;
    open();

    await settled(() => expect(calls.stopped).toBe(1));
    expect(row().classList.contains('disabled')).toBe(true);
  });
});

describe('SSE reconnect resync — membership re-converges from /api/panels', () => {
  /**
   * Boot against a MUTABLE server state and an EventSource stub whose
   * `open()` drives the onopen handler — the reconnect seam. SSE has no
   * replay, so a frame published while a client was disconnected is simply
   * gone; on every open panel-manager must re-fetch /api/panels and apply
   * the authoritative visible set as a delta.
   */
  async function bootResync() {
    window.__OSPREY_PREFIX__ = '';
    renderContainer();
    const server = { visible: ['artifacts', 'ariel'] };
    vi.stubGlobal(
      'fetch',
      vi.fn(async (/** @type {string} */ url) => {
        if (url === '/api/panels') {
          return jsonOk({
            enabled: ['artifacts', 'ariel'],
            custom: [],
            default: null,
            visible: [...server.visible],
            active: null,
            labels: {},
          });
        }
        if (url === '/api/artifact-server') {
          return jsonOk({ url: '/panel/artifacts', available: true });
        }
        if (url === '/api/ariel-server') {
          return jsonOk({ url: '/panel/ariel', available: true });
        }
        return jsonOk({ status: 'ok' });
      })
    );

    /** @type {{ onopen?: (() => void) | null }[]} */
    const sources = [];
    class FakeEventSource {
      constructor() {
        /** @type {(() => void) | null} */
        this.onopen = null;
        sources.push(this);
      }
      close() {}
    }
    vi.stubGlobal('EventSource', FakeEventSource);

    const mod = await freshImport();
    await mod.initPanelManager('panel-manager');
    return {
      mod,
      server,
      open: () => {
        for (const s of sources) s.onopen?.();
      },
    };
  }

  /** @param {string} id */
  const entry = (id) => document.querySelector(`.panel-rail-button[data-panel-id="${id}"]`);

  test('a hide missed while disconnected is applied on reconnect', async () => {
    const { mod, server, open } = await bootResync();
    expect(entry('ariel')).not.toBeNull();

    // The agent hid ariel while this client's SSE stream was down: the
    // panel_visibility frame never arrived. The stream then reconnects.
    server.visible = ['artifacts'];
    open();

    await settled(() => expect(entry('ariel')).toBeNull());
    expect(mod.getHiddenPanels().map((p) => p.id)).toContain('ariel');
  });

  test('a show missed while disconnected is applied on reconnect', async () => {
    const { server, open } = await bootResync();
    server.visible = ['artifacts'];
    open();
    await settled(() => expect(entry('ariel')).toBeNull());

    server.visible = ['artifacts', 'ariel'];
    open();

    await settled(() => expect(entry('ariel')).not.toBeNull());
  });

  test('an in-sync reconnect changes nothing', async () => {
    const { open } = await bootResync();
    const before = [...document.querySelectorAll('.panel-rail-button')].map((b) =>
      b.getAttribute('data-panel-id')
    );

    open();
    await new Promise((r) => setTimeout(r, 0));

    const after = [...document.querySelectorAll('.panel-rail-button')].map((b) =>
      b.getAttribute('data-panel-id')
    );
    expect(after).toEqual(before);
  });
});

describe('tile-body glow — fired only where a tile visibly changed', () => {
  /** @param {string} id */
  const entry = (id) => document.querySelector(`.panel-rail-button[data-panel-id="${id}"]`);
  /** The panel ids handed to glowPanel, in call order. */
  const glowed = () => glowPanelSpy.mock.calls.map((/** @type {any[]} */ c) => c[0]);
  const THREE = ['artifacts', 'ariel', 'channel-finder'];

  test("an agent open glows the opened panel's tile", async () => {
    const { op } = await bootWorkspace();

    op('open', { member: 'ariel' });

    expect(glowed()).toEqual(['ariel']);
  });

  test("a person's own rail click glows nothing — the operator did it", async () => {
    await bootWorkspace();

    /** @type {HTMLElement} */ (entry('ariel')).click();

    expect(glowPanelSpy).not.toHaveBeenCalled();
  });

  /** An open_arrangement of channel-finder | ariel. */
  const ARRANGEMENT = {
    name: 'arrangement',
    members: ['channel-finder', 'ariel'],
    layout: { split: 'row', items: [{ tabs: [0] }, { tabs: [1] }] },
  };

  test('an agent arrangement glows every arranged tile', async () => {
    const { op } = await bootWorkspace({ panels: THREE });

    op('open_arrangement', ARRANGEMENT);

    expect(glowed().sort()).toEqual(['ariel', 'channel-finder']);
    // A panel the arrangement did not list keeps its tile out of the story.
    expect(glowed()).not.toContain('artifacts');
  });

  test('an untagged arrangement (human Layouts click) glows nothing', async () => {
    const { op, holder } = await bootWorkspace({ panels: THREE });

    op('open_arrangement', ARRANGEMENT, null);

    // The arrangement did run — it is only unattributed.
    expect(tileTabs(holder)).toEqual([['channel-finder'], ['ariel'], ['terminal']]);
    expect(glowPanelSpy).not.toHaveBeenCalled();
  });

  test('an agent SHOW glows the rail entry only — no tile exists to attribute to', async () => {
    const { emit } = await bootWorkspace({ visible: ['artifacts'] });

    emit({ type: 'panel_visibility', panel: 'ariel', visible: true, source: 'agent' });

    expect(entry('ariel')?.classList.contains('agent-flash')).toBe(true);
    expect(glowPanelSpy).not.toHaveBeenCalled();
  });

  test('an agent HIDE glows nothing — the tile is on its way out', async () => {
    const { emit, mod } = await bootWorkspace();
    const strip = vi.fn();
    mod.setActivityStripHandler(strip);

    emit({ type: 'panel_visibility', panel: 'artifacts', visible: false, source: 'agent' });

    expect(strip).toHaveBeenCalledTimes(1); // the agent branch DID run
    expect(glowPanelSpy).not.toHaveBeenCalled();
  });

  test('an agent panel_register glows the new rail entry only — it opens no tile', async () => {
    const { emit } = await bootWorkspace();

    emit({
      type: 'panel_register',
      id: 'bluesky',
      label: 'BLUESKY',
      url: '/panel/bluesky',
      healthEndpoint: null,
      path: '/',
      source: 'agent',
    });

    expect(entry('bluesky')?.classList.contains('agent-flash')).toBe(true);
    expect(glowPanelSpy).not.toHaveBeenCalled();
  });
});

describe('agent-attention badges survive a reload — acknowledged by server ts', () => {
  const ACK = 'agent-ack:';

  // The ack store is real localStorage and outlives a module reset, so every
  // case starts from a page that has acknowledged nothing.
  beforeEach(() => localStorage.clear());
  afterEach(() => localStorage.clear());

  /**
   * Boot with a healthy 'artifacts' panel, an unhealthy-but-present 'ariel'
   * entry, and a MUTABLE history buffer behind /api/agent-activity/recent. Both
   * SSE seams are exposed: `emit` for frames, and `open` for the hook that
   * re-reads the buffer (a reload's first open and every reconnect run it).
   * With `holdLanding`, the artifacts config answer waits on `releaseLanding()`,
   * so the boot landing runs only after whatever the case does first.
   * @param {{events?: any[], holdLanding?: boolean}} [opts]
   */
  async function bootAck({ events = [], holdLanding = false } = {}) {
    window.__OSPREY_PREFIX__ = '';
    renderContainer();
    const buffer = { events };
    const reads = { recent: 0 };
    /** @type {() => void} */
    let releaseLanding = () => {};
    const landingHeld = holdLanding
      ? new Promise((r) => {
          releaseLanding = () => r(undefined);
        })
      : Promise.resolve();
    vi.stubGlobal(
      'fetch',
      vi.fn(async (/** @type {string} */ url) => {
        if (url === '/api/panels') {
          return jsonOk({
            enabled: ['artifacts', 'ariel'],
            custom: [],
            default: null,
            visible: ['artifacts', 'ariel'],
            active: null,
            labels: {},
          });
        }
        if (url === '/api/artifact-server') {
          await landingHeld;
          return jsonOk({ url: '/panel/artifacts', available: true });
        }
        if (url === '/api/agent-activity/recent') {
          reads.recent += 1;
          return jsonOk({ events: [...buffer.events] });
        }
        return jsonOk({ status: 'ok' }); // ariel: no url ⇒ entry present, disabled
      })
    );

    /** @type {{ onmessage?: ((e: {data: string}) => void) | null, onopen?: (() => void) | null }[]} */
    const sources = [];
    class FakeEventSource {
      constructor() {
        /** @type {((e: {data: string}) => void) | null} */
        this.onmessage = null;
        /** @type {(() => void) | null} */
        this.onopen = null;
        sources.push(this);
      }
      close() {}
    }
    vi.stubGlobal('EventSource', FakeEventSource);

    const mod = await freshImport();
    await mod.initPanelManager('panel-manager');
    const artifacts = /** @type {HTMLElement} */ (
      document.querySelector('[data-panel-id="artifacts"]')
    );
    if (!holdLanding) {
      await settled(() => expect(artifacts.classList.contains('disabled')).toBe(false));
    }
    return {
      mod,
      buffer,
      artifacts,
      releaseLanding,
      /** @param {object} frame */
      emit: (frame) => {
        for (const s of sources) s.onmessage?.({ data: JSON.stringify(frame) });
      },
      open: () => {
        for (const s of sources) s.onopen?.();
      },
      // Wait until the open() hook has actually READ the buffer, then flush the
      // handler that consumes it. A "no badge appeared" assertion is only
      // evidence once the restore has genuinely run.
      settle: async () => {
        await settled(() => expect(reads.recent).toBeGreaterThan(0));
        await new Promise((r) => setTimeout(r, 0));
      },
    };
  }

  /** @param {string} id */
  const badged = (id) =>
    !!document.querySelector(`[data-panel-id="${id}"]`)?.classList.contains('agent-attention');

  test("surfacing a badged panel persists that badge's SERVER ts as the ack", async () => {
    const { emit, mod } = await bootAck();

    emit({
      type: 'agent_activity',
      tool: 'read_file',
      target: { kind: 'panel', panel: 'artifacts' },
      ts: 1000.5,
    });
    expect(badged('artifacts')).toBe(true);

    mod.activateTab('artifacts', { userInitiated: true });

    expect(badged('artifacts')).toBe(false);
    expect(localStorage.getItem(`${ACK}artifacts`)).toBe('1000.5');
  });

  test('no ack ⇒ an unseen buffer event restores the badge on SSE open', async () => {
    const { open } = await bootAck({
      events: [
        {
          type: 'agent_activity',
          tool: 'search_logbook',
          target: { kind: 'panel', panel: 'ariel' },
          ts: 1000,
        },
      ],
    });
    expect(badged('ariel')).toBe(false);

    open();

    await settled(() => expect(badged('ariel')).toBe(true));
  });

  test('a buffer event at or before the ack does NOT resurrect the badge', async () => {
    localStorage.setItem(`${ACK}ariel`, '1000');
    const { open, settle } = await bootAck({
      events: [
        {
          type: 'agent_activity',
          tool: 'search_logbook',
          target: { kind: 'panel', panel: 'ariel' },
          ts: 1000,
        },
        {
          type: 'agent_activity',
          tool: 'search_logbook',
          target: { kind: 'panel', panel: 'ariel' },
          ts: 999,
        },
      ],
    });

    open();
    await settle();

    expect(badged('ariel')).toBe(false);
  });

  test('a buffer event after the ack restores the badge', async () => {
    localStorage.setItem(`${ACK}ariel`, '1000');
    const { open } = await bootAck({
      events: [
        {
          type: 'agent_activity',
          tool: 'search_logbook',
          target: { kind: 'panel', panel: 'ariel' },
          ts: 1000.5,
        },
      ],
    });

    open();

    await settled(() => expect(badged('ariel')).toBe(true));
  });

  test('non-panel-kind buffer rows never badge the rail', async () => {
    const { open, settle } = await bootAck({
      events: [
        {
          type: 'agent_activity',
          tool: 'read_channel',
          target: { kind: 'channel', detail: 'SR01C:BPM1:X' },
          ts: 2000,
        },
        {
          type: 'agent_activity',
          tool: 'run_plan',
          target: { kind: 'run', detail: 'orm-3' },
          ts: 1999,
        },
        // A panel-kind row with no rail entry has nothing to badge either.
        {
          type: 'agent_activity',
          tool: 'open_panel',
          target: { kind: 'panel', panel: 'no-such-panel' },
          ts: 1998,
        },
      ],
    });

    open();
    await settle();

    expect(document.querySelector('.agent-attention')).toBeNull();
  });

  test('the newest of several rows for one panel becomes the ack', async () => {
    const { open, mod } = await bootAck({
      events: [
        // newest first, as the endpoint serves them
        {
          type: 'agent_activity',
          tool: 'read_file',
          target: { kind: 'panel', panel: 'artifacts' },
          ts: 30,
        },
        {
          type: 'agent_activity',
          tool: 'read_file',
          target: { kind: 'panel', panel: 'artifacts' },
          ts: 10,
        },
      ],
    });

    open();
    await settled(() => expect(badged('artifacts')).toBe(true));
    mod.activateTab('artifacts', { userInitiated: true });

    expect(localStorage.getItem(`${ACK}artifacts`)).toBe('30');
  });

  test('a clear with no badge ts on record leaves the stored ack untouched', async () => {
    localStorage.setItem(`${ACK}artifacts`, '1000');
    const { mod } = await bootAck(); // boot surfaces artifacts — an unbadged clear

    mod.activateTab('artifacts', { userInitiated: true });

    expect(localStorage.getItem(`${ACK}artifacts`)).toBe('1000');
  });

  /** @param {string} id */
  const surfaced = (id) =>
    !!document.querySelector(`[data-panel-id="${id}"]`)?.classList.contains('active');

  test('a landing after the restore surfaces the panel and keeps its badge and ack', async () => {
    localStorage.setItem(`${ACK}artifacts`, '1000');
    const { open, settle, releaseLanding } = await bootAck({
      holdLanding: true,
      events: [
        {
          type: 'agent_activity',
          tool: 'read_file',
          target: { kind: 'panel', panel: 'artifacts' },
          ts: 1000.5,
        },
      ],
    });

    open();
    await settle();
    await settled(() => expect(badged('artifacts')).toBe(true));

    releaseLanding();

    await settled(() => expect(surfaced('artifacts')).toBe(true));
    expect(badged('artifacts')).toBe(true);
    expect(localStorage.getItem(`${ACK}artifacts`)).toBe('1000');
  });

  test('a replayed open_panel badges its panel and never surfaces it', async () => {
    const { open, settle } = await bootAck({
      events: [
        {
          type: 'agent_activity',
          tool: 'open_panel',
          target: { kind: 'panel', panel: 'ariel' },
          ts: 1000,
        },
      ],
    });
    expect(surfaced('artifacts')).toBe(true);

    open();
    await settle();

    await settled(() => expect(badged('ariel')).toBe(true));
    expect(surfaced('ariel')).toBe(false);
    expect(surfaced('artifacts')).toBe(true);
  });

  test('a badge restored on the landed panel stays until a requested surfacing serves it', async () => {
    localStorage.setItem(`${ACK}artifacts`, '1000');
    const { open, mod } = await bootAck({
      events: [
        {
          type: 'agent_activity',
          tool: 'read_file',
          target: { kind: 'panel', panel: 'artifacts' },
          ts: 1000.5,
        },
      ],
    });
    expect(surfaced('artifacts')).toBe(true);

    open();

    await settled(() => expect(badged('artifacts')).toBe(true));
    expect(localStorage.getItem(`${ACK}artifacts`)).toBe('1000');

    mod.activateTab('artifacts', { userInitiated: true });

    expect(badged('artifacts')).toBe(false);
    expect(localStorage.getItem(`${ACK}artifacts`)).toBe('1000.5');
  });

  test('the ack is the server ts even when the browser clock is far ahead', async () => {
    // A client clock written as an ack would be ~2e12 here and would swallow
    // every future server ts — badges would never come back after one clear.
    vi.spyOn(Date, 'now').mockReturnValue(2_000_000_000_000);
    const { emit, buffer, open, mod } = await bootAck();

    emit({
      type: 'agent_activity',
      tool: 'read_file',
      target: { kind: 'panel', panel: 'artifacts' },
      ts: 5,
    });
    mod.activateTab('artifacts', { userInitiated: true });
    expect(localStorage.getItem(`${ACK}artifacts`)).toBe('5');

    // Reconnect with a newer server event: still strictly greater than the ack.
    buffer.events = [
      {
        type: 'agent_activity',
        tool: 'read_file',
        target: { kind: 'panel', panel: 'artifacts' },
        ts: 6,
      },
    ];
    open();

    await settled(() => expect(badged('artifacts')).toBe(true));
  });
});

describe('rail context menu — the entry’s verbs in words (railOptions onContextMenu)', () => {
  const entry = (/** @type {string} */ id) =>
    /** @type {HTMLElement} */ (
      document.querySelector(`.panel-rail-button[data-panel-id="${id}"]`)
    );
  const activeStamp = () => document.getElementById('panel-manager')?.dataset.activePanel ?? null;

  afterEach(() => {
    // Dismiss whatever a test left open — an undismissed popover keeps
    // document-level listeners alive across the module reset.
    document.body.dispatchEvent(new MouseEvent('pointerdown', { bubbles: true }));
    document.documentElement.removeAttribute('data-agent-surface');
  });

  test('a healthy service entry opens the full verb list and suppresses the native menu', async () => {
    await bootWorkspace();

    const ev = rightClick(entry('ariel'));

    expect(ev.defaultPrevented).toBe(true);
    expect(menuLabels()).toEqual([
      'Focus Logbook',
      'Open in a new tile',
      'Pop out',
      'Move to Workspace',
      'Move to new group…',
      'Remove from rail',
    ]);
    expect(contextMenu()?.getAttribute('aria-label')).toBe('Logbook actions');
  });

  test('the entry is the menu’s anchor, so a panel scrolling its own content leaves it open', async () => {
    await bootWorkspace();
    rightClick(entry('ariel'));

    // A scroll inside some other subtree — xterm streaming output is the case
    // that matters — is not a scroll of the anchor and must not dismiss.
    const other = document.createElement('div');
    document.body.appendChild(other);
    other.dispatchEvent(new Event('scroll', { bubbles: false }));
    expect(contextMenu()).not.toBeNull();

    // The rail scrolling DOES move the anchor.
    document.getElementById('panel-rail')?.dispatchEvent(new Event('scroll'));
    expect(contextMenu()).toBeNull();
  });

  test('"Open in a new tile" docks a NEW tile beside the active one', async () => {
    const { api } = await bootWorkspace();
    const artifactsGroup = api.getPanel('iframe:artifacts').group;

    rightClick(entry('ariel'));
    /** @type {HTMLElement} */ (menuRow('Open in a new tile')).click();

    expect(dockedTiles(api).sort()).toEqual(['ariel', 'artifacts']);
    expect(api.getPanel('iframe:ariel').group).not.toBe(artifactsGroup);
    // Running a row closes the menu before the action — never after it.
    expect(contextMenu()).toBeNull();
  });

  test('"Pop out" opens the panel’s pop-out page in a named popup and retires the tile', async () => {
    const { api, mod, calls } = await bootWorkspace();
    rightClick(entry('ariel'));
    /** @type {HTMLElement} */ (menuRow('Open in a new tile')).click();
    expect(dockedTiles(api).sort()).toEqual(['ariel', 'artifacts']);
    calls.length = 0; // the new-tile reveal above reports its own focus
    const popup = { closed: false, focus: vi.fn(), close: vi.fn(), postMessage: vi.fn() };
    const open = vi.spyOn(window, 'open').mockReturnValue(/** @type {any} */ (popup));

    rightClick(entry('ariel'));
    /** @type {HTMLElement} */ (menuRow('Pop out')).click();

    // The standalone page, stamped like an embed but with `popout` — never
    // `embedded`, never a token — under the protocol's window name, as a
    // popup sized and placed in the features string.
    const [url, name, features] = open.mock.calls[0];
    const opened = new URL(String(url));
    expect(opened.pathname).toBe(
      new URL(String(mod.getPanelStandaloneUrl('ariel')), 'http://h').pathname
    );
    expect(opened.searchParams.get('popout')).toBe('true');
    expect(opened.searchParams.has('embedded')).toBe(false);
    expect(opened.searchParams.has('theme')).toBe(true);
    expect(opened.searchParams.has('mode')).toBe(false);
    expect(name).toBe('osprey-popout-ariel');
    expect(features).toMatch(/^popup=yes,width=\d+,height=\d+,left=\d+,top=\d+$/);
    // The tile is gone from the dock while the window is open: a layout
    // report cannot list it, and a layout restore cannot bring it back. The
    // rail row stays (membership is untouched, nothing is POSTed).
    expect(dockedTiles(api)).toEqual(['artifacts']);
    expect(entry('ariel')).not.toBeNull();
    expect(posts(calls)).toEqual([]);
    // The menu now offers the way back; the palette's pop-out rows skip it.
    rightClick(entry('ariel'));
    expect(menuLabels()).toContain('Bring back');
    expect(menuLabels()).not.toContain('Open in a new tile');
    expect(mod.getPopoutPanels().map((p) => p.id)).toEqual(['artifacts']);
    open.mockRestore();
  });

  test('a focus of a popped-out panel brings its window forward instead of docking', async () => {
    const { api, mod, op, calls } = await bootWorkspace();
    const popup = { closed: false, focus: vi.fn(), close: vi.fn(), postMessage: vi.fn() };
    const open = vi.spyOn(window, 'open').mockReturnValue(/** @type {any} */ (popup));
    rightClick(entry('ariel'));
    /** @type {HTMLElement} */ (menuRow('Pop out')).click();
    calls.length = 0;

    // The rail click, the palette's Focus and the agent's open all land on the
    // window; none of them docks a tile or reports a focus. The op answers with
    // the layout unchanged and the panel named among the pop-outs.
    entry('ariel').click();
    mod.activateTab('ariel', { userInitiated: true });
    op('open', { member: 'ariel' });
    expect(popup.focus).toHaveBeenCalledTimes(3);
    expect(dockedTiles(api)).toEqual(['artifacts']);
    await settled(() => expect(posts(calls)).toEqual(['/api/panel-op-result']));
    expect(JSON.parse(calls.find((c) => c.opts?.method === 'POST')?.opts.body)).toMatchObject({
      popouts: ['ariel'],
    });
    open.mockRestore();
  });

  test('"Bring back" closes the window, and the closed window re-docks the panel as a tab of the service tile', async () => {
    const { api, calls } = await bootWorkspace();
    const popup = { closed: false, focus: vi.fn(), close: vi.fn(), postMessage: vi.fn() };
    const open = vi.spyOn(window, 'open').mockReturnValue(/** @type {any} */ (popup));
    rightClick(entry('ariel'));
    /** @type {HTMLElement} */ (menuRow('Pop out')).click();
    const artifactsGroup = api.getPanel('iframe:artifacts').group.id;

    rightClick(entry('ariel'));
    /** @type {HTMLElement} */ (menuRow('Bring back')).click();
    expect(popup.close).toHaveBeenCalledTimes(1);
    // Closing is the window's own business: the tile returns once the poll
    // (every 500 ms) sees the window closed.
    expect(dockedTiles(api)).toEqual(['artifacts']);
    popup.closed = true;
    await settled(() => expect(dockedTiles(api).sort()).toEqual(['ariel', 'artifacts']));

    // By the open rule: a tab in front of the tile the operator was on.
    expect(api.getPanel('iframe:ariel').group.id).toBe(artifactsGroup);
    expect((await pageHolder()).layout.items[0].tabs).toEqual(['artifacts', 'ariel']);
    // Nobody asked the server for this surfacing.
    expect(posts(calls)).toEqual([]);
    open.mockRestore();
  });

  test('a hub reload adopts the window that answers the roll-call and retires its tile', async () => {
    /** @type {any[]} */
    const channels = [];
    vi.stubGlobal(
      'BroadcastChannel',
      class {
        /** @param {string} name */
        constructor(name) {
          this.name = name;
          this.posted = /** @type {any[]} */ ([]);
          channels.push(this);
        }
        /** @param {any} message */
        postMessage(message) {
          this.posted.push(message);
        }
        close() {}
      }
    );
    // The stored layout holds ariel's tile; its window answers the roll-call
    // before the boot layout settles, as it does in a browser (the hello is a
    // message away, the restore a network round trip).
    dockState.bootSettled = false;
    const { api } = await bootWorkspace({ restored: ['ariel'], firstTile: false });
    expect(channels.map((c) => [c.name, c.posted])).toEqual([
      ['osprey-popout', [{ type: 'osprey-popout-roll-call' }]],
    ]);

    // The popup's hello: a same-origin window this hub opened under the
    // protocol's name. Its tile goes, and it is told the hub's state.
    const popup = {
      closed: false,
      opener: window,
      name: 'osprey-popout-ariel',
      focus: vi.fn(),
      close: vi.fn(),
      postMessage: vi.fn(),
    };
    window.dispatchEvent(
      new MessageEvent('message', {
        data: { type: 'osprey-popout-hello', panelId: 'ariel' },
        origin: window.location.origin,
        source: /** @type {any} */ (popup),
      })
    );

    expect(dockedTiles(api)).toEqual([]);
    // The landing then skips the panel in its window and gives the slot to
    // the default.
    settleBootLayout();
    await settled(() => expect(dockedTiles(api)).toEqual(['artifacts']));
    // The window is shared across this file's module graphs, so the hub
    // instances of earlier tests answer the hello too: each sends the same
    // state (no session yet, so no session message).
    expect(new Set(popup.postMessage.mock.calls.map((c) => c[0].type))).toEqual(
      new Set(['osprey-theme-change'])
    );
    expect(popup.postMessage.mock.calls.every((c) => c[1] === window.location.origin)).toBe(true);
    rightClick(entry('ariel'));
    expect(menuLabels()).toContain('Bring back');
  });

  test('a restored placeholder is filled with its panel’s iframe; the landing takes the tile in front', async () => {
    dockState.bootSettled = false;
    const { api, calls } = await bootWorkspace({ firstTile: false });
    expect(document.querySelector('iframe.panel-iframe[data-panel-id="ariel"]')).toBeNull();

    // The stored layout lands after dock-workspace's /api/panels round trip:
    // a placeholder for every tile that was open, then the layout-applied
    // seam, then the boot layout is announced final.
    (await pageHolder()).restore(
      rowOf(['artifacts'], ['ariel'], { tile: 'tt', tabs: ['terminal'] })
    );
    dockState.redock?.();
    settleBootLayout();

    await settled(() => expect(activeStamp()).toBe('artifacts'));
    const iframe = /** @type {HTMLElement | null} */ (
      document.querySelector('iframe.panel-iframe[data-panel-id="ariel"]')
    );
    expect(iframe).not.toBeNull();
    expect(iframe?.style.display).not.toBe('none');
    expect(dockedTiles(api).sort()).toEqual(['ariel', 'artifacts']);
    // The layout came back whole: two one-tab tiles, nothing added.
    expect(tileOrder((await pageHolder()).layout)).toEqual(['artifacts', 'ariel', 'terminal']);
    // The landing is a surfacing nobody asked for, so nothing is reported.
    expect(entry('artifacts').getAttribute('aria-current')).toBe('page');
    expect(entry('ariel').getAttribute('aria-current')).toBeNull();
    expect(posts(calls)).toEqual([]);
  });

  test('a popped-out panel’s restored placeholder is pruned, never filled', async () => {
    const { api } = await bootWorkspace();
    const popup = { closed: false, focus: vi.fn(), close: vi.fn(), postMessage: vi.fn() };
    const open = vi.spyOn(window, 'open').mockReturnValue(/** @type {any} */ (popup));
    rightClick(entry('ariel'));
    /** @type {HTMLElement} */ (menuRow('Open in a new tile')).click();
    rightClick(entry('ariel'));
    /** @type {HTMLElement} */ (menuRow('Pop out')).click();
    const iframe = /** @type {HTMLElement} */ (
      document.querySelector('iframe.panel-iframe[data-panel-id="ariel"]')
    );
    expect(iframe.style.display).toBe('none');

    // A layout apply brings the placeholder back (a layout stored before the
    // pop-out); the adapter prunes it and the manager leaves the iframe hidden.
    (await pageHolder()).restore(
      rowOf(['artifacts'], ['ariel'], { tile: 'tt', tabs: ['terminal'] })
    );
    dockState.redock?.();

    expect(api.getPanel('iframe:ariel')).toBeNull();
    expect(iframe.style.display).toBe('none');
    open.mockRestore();
  });

  test('"Remove from rail" POSTs the same membership change the Delete key does', async () => {
    const { calls } = await bootWorkspace();

    rightClick(entry('ariel'));
    /** @type {HTMLElement} */ (menuRow('Remove from rail')).click();

    const post = /** @type {{url: string, opts: any}} */ (
      calls.find((c) => c.url === '/api/panel-visibility')
    );
    expect(post).toBeDefined();
    expect(JSON.parse(post.opts.body)).toEqual({ panel: 'ariel', visible: false });
  });

  test('Focus on the ALREADY-active panel is a no-op — never the row click’s retire branch', async () => {
    const { api } = await bootWorkspace();
    expect(activeStamp()).toBe('artifacts');

    rightClick(entry('artifacts'));
    /** @type {HTMLElement} */ (menuRow('Focus Artifacts')).click();

    expect(dockedTiles(api)).toContain('artifacts');
    expect(activeStamp()).toBe('artifacts');

    // The contrast that makes the row’s wording true: clicking a plain row
    // itself toggles, and DOES retire the tile it is already showing.
    entry('artifacts').click();
    expect(dockedTiles(api)).not.toContain('artifacts');
  });

  test('a parent row (Artifacts) only ever brings its tile forward, never retires it', async () => {
    const { api } = await bootWorkspace();
    expect(activeStamp()).toBe('artifacts');
    // The page ships the Artifacts row as a `.parent`: its click opens the
    // row's list, which is what the operator reached for.
    entry('artifacts').classList.add('parent');

    entry('artifacts').click();

    expect(dockedTiles(api)).toContain('artifacts');
    expect(activeStamp()).toBe('artifacts');
  });

  test('Focus surfaces a member that holds no tile', async () => {
    const { api } = await bootWorkspace();

    rightClick(entry('ariel'));
    /** @type {HTMLElement} */ (menuRow('Focus Logbook')).click();

    expect(dockedTiles(api)).toContain('ariel');
    expect(activeStamp()).toBe('ariel');
  });

  test('a disabled entry gets no menu, and the browser keeps its own', async () => {
    await bootWorkspace({ panels: ['artifacts', 'ariel'], unhealthy: ['ariel'] });
    expect(entry('ariel').classList.contains('disabled')).toBe(true);

    // The gate is the handler’s, not the CSS’s: happy-dom applies no
    // pointer-events, so this dispatch is exactly the route that gets past
    // `pointer-events: none` in a real browser.
    const onEntry = rightClick(entry('ariel'));
    expect(contextMenu()).toBeNull();
    // Declining leaves the event alone, so the browser’s own menu still shows.
    expect(onEntry.defaultPrevented).toBe(false);
  });

  test('the keyboard route obeys the same gate', async () => {
    await bootWorkspace({ panels: ['artifacts', 'ariel'], unhealthy: ['ariel'] });

    const declined = menuKey(entry('ariel'));
    expect(contextMenu()).toBeNull();
    // Not cancelled: on the platforms that fire a native contextmenu from this
    // keydown, the declined entry keeps it.
    expect(declined.defaultPrevented).toBe(false);

    const opened = menuKey(entry('artifacts'));
    expect(contextMenu()).not.toBeNull();
    // Cancelled, so one keypress cannot also stack the platform’s own menu.
    expect(opened.defaultPrevented).toBe(true);
  });

  test('on the terminal surface the Agent row offers the session verbs, and pairs restart with reconnect', async () => {
    await bootWorkspace({ surface: 'terminal' });

    const ev = rightClick(entry('terminal'));

    expect(ev.defaultPrevented).toBe(true);
    expect(menuLabels()).toEqual([
      'Restart terminal',
      'New session',
      'Reset layout',
      'Close Agent pane',
    ]);
    expect(contextMenu()?.getAttribute('aria-label')).toBe('Agent actions');

    /** @type {HTMLElement} */ (menuRow('Restart terminal')).click();
    await settled(() => expect(startTerminal).toHaveBeenCalled());
    // restartTerminal tears the PTY down without reconnecting — unpaired it
    // would leave the card stranded.
    expect(restartTerminal).toHaveBeenCalled();
    expect(restartTerminal.mock.invocationCallOrder[0]).toBeLessThan(
      startTerminal.mock.invocationCallOrder[0]
    );
  });

  test('the Agent row’s other verbs reach the session list and the dock', async () => {
    await bootWorkspace({ surface: 'terminal' });

    rightClick(entry('terminal'));
    /** @type {HTMLElement} */ (menuRow('New session')).click();
    expect(startNewSession).toHaveBeenCalled();

    rightClick(entry('terminal'));
    /** @type {HTMLElement} */ (menuRow('Close Agent pane')).click();
    expect(closeTerminalPanel).toHaveBeenCalled();

    // Reset puts the default arrangement back.
    rightClick(entry('terminal'));
    /** @type {HTMLElement} */ (menuRow('Reset layout')).click();
    expect(resetDockLayout).toHaveBeenCalledTimes(1);
  });

  test('in chat the Agent menu keeps only the layout rows', async () => {
    await bootWorkspace();

    // The tile hosts the operator console there, so the two PTY verbs would
    // act on a surface the operator cannot see and are withheld.
    const ev = rightClick(entry('terminal'));

    expect(ev.defaultPrevented).toBe(true);
    expect(menuLabels()).toEqual(['Reset layout', 'Close Agent pane']);
    /** @type {HTMLElement} */ (menuRow('Reset layout')).click();
    expect(resetDockLayout).toHaveBeenCalledTimes(1);
    rightClick(entry('terminal'));
    /** @type {HTMLElement} */ (menuRow('Close Agent pane')).click();
    expect(closeTerminalPanel).toHaveBeenCalled();
  });

  test('a click on the Agent row opens the pane and focuses the terminal — twice, never a close', async () => {
    await bootWorkspace({ surface: 'terminal' });
    openTerminalPanel.mockClear();
    focusTerminal.mockClear();

    entry('terminal').click();
    entry('terminal').click();

    expect(openTerminalPanel).toHaveBeenCalledTimes(2);
    expect(focusTerminal).toHaveBeenCalledTimes(2);
    expect(closeTerminalPanel).not.toHaveBeenCalled();
  });

  test('in chat the Agent row puts the caret in the composer', async () => {
    await bootWorkspace();
    focusTerminal.mockClear();
    const operator = document.createElement('div');
    operator.id = 'operator-container';
    const composer = document.createElement('textarea');
    operator.append(composer);
    document.body.append(operator);

    entry('terminal').click();

    expect(openTerminalPanel).toHaveBeenCalled();
    expect(document.activeElement).toBe(composer);
    expect(focusTerminal).not.toHaveBeenCalled();
    expect(closeTerminalPanel).not.toHaveBeenCalled();
  });

  test('an Agent pane that did not open takes no focus', async () => {
    await bootWorkspace();
    focusTerminal.mockClear();
    isTerminalPanelOpen.mockReturnValueOnce(false);

    entry('terminal').click();

    expect(openTerminalPanel).toHaveBeenCalled();
    expect(focusTerminal).not.toHaveBeenCalled();
  });
});

describe('Undo after a one-click removal — the rail entry comes back where it was', () => {
  const entry = (/** @type {string} */ id) =>
    /** @type {HTMLElement} */ (
      document.querySelector(`.panel-rail-button[data-panel-id="${id}"]`)
    );
  const railIds = () =>
    [...document.querySelectorAll('.panel-rail-button')].map((el) =>
      el.getAttribute('data-panel-id')
    );
  const notice = () => document.getElementById('undo-notice');
  const undoButton = () =>
    /** @type {HTMLElement | null} */ (document.querySelector('.undo-notice-btn'));
  const visibilityBodies = (/** @type {{url: string, opts: any}[]} */ calls) =>
    calls.filter((c) => c.url === '/api/panel-visibility').map((c) => JSON.parse(c.opts.body));
  const PANELS3 = ['artifacts', 'ariel', 'channel-finder'];

  afterEach(() => {
    document.body.dispatchEvent(new MouseEvent('pointerdown', { bubbles: true }));
  });

  test('"Remove from rail" offers Undo, which re-adds the panel at its old position', async () => {
    const { calls, emit } = await bootWorkspace({ panels: PANELS3 });
    expect(railIds()).toEqual(['terminal', 'artifacts', 'ariel', 'channel-finder']);

    rightClick(entry('ariel'));
    /** @type {HTMLElement} */ (menuRow('Remove from rail')).click();
    await settled(() => expect(notice()?.textContent).toContain('Logbook removed from the rail'));
    emit({ type: 'panel_visibility', panel: 'ariel', visible: false });
    expect(railIds()).toEqual(['terminal', 'artifacts', 'channel-finder']);

    /** @type {HTMLElement} */ (undoButton()).click();
    // The position counts panel rows only — the Agent row is not one of them.
    await settled(() =>
      expect(visibilityBodies(calls).at(-1)).toEqual({ panel: 'ariel', visible: true, index: 1 })
    );
    expect(notice()).toBeNull();
    emit({ type: 'panel_visibility', panel: 'ariel', visible: true, index: 1 });
    expect(railIds()).toEqual(['terminal', 'artifacts', 'ariel', 'channel-finder']);
  });

  /** Press Delete on a focused row, as an operator removing it from the keyboard.
   *  @param {string} id */
  const pressDelete = (id) =>
    entry(id).dispatchEvent(
      new KeyboardEvent('keydown', { key: 'Delete', bubbles: true, cancelable: true })
    );

  test('the row’s Delete key offers the same Undo', async () => {
    const { calls } = await bootWorkspace({ panels: PANELS3 });

    pressDelete('artifacts');
    await settled(() => expect(notice()?.textContent).toContain('Artifacts removed from the rail'));
    /** @type {HTMLElement} */ (undoButton()).click();
    await settled(() =>
      expect(visibilityBodies(calls).at(-1)).toEqual({
        panel: 'artifacts',
        visible: true,
        index: 0,
      })
    );
  });

  test('a refused removal says so, with no Undo to offer', async () => {
    const { calls } = await bootWorkspace({ panels: PANELS3 });
    const ok = /** @type {any} */ (globalThis.fetch).getMockImplementation();
    vi.stubGlobal(
      'fetch',
      vi.fn(async (/** @type {string} */ url, /** @type {any} */ o) => {
        if (url === '/api/panel-visibility') {
          calls.push({ url, opts: o });
          return { ok: false, status: 422, statusText: 'Unprocessable', json: async () => ({}) };
        }
        return ok(url, o);
      })
    );

    rightClick(entry('ariel'));
    /** @type {HTMLElement} */ (menuRow('Remove from rail')).click();

    await settled(() =>
      expect(notice()?.textContent).toContain('Could not remove Logbook from the rail')
    );
    expect(undoButton()).toBeNull();
  });

  test('a removal that arrives as an SSE echo — the agent, another tab — raises no notice', async () => {
    const { emit } = await bootWorkspace({ panels: PANELS3 });

    emit({ type: 'panel_visibility', panel: 'ariel', visible: false, source: 'agent' });
    emit({ type: 'panel_visibility', panel: 'channel-finder', visible: false });

    expect(railIds()).toEqual(['terminal', 'artifacts']);
    expect(notice()).toBeNull();
  });

  test('an indexed addition lands at that position on every client; a member keeps its place', async () => {
    const { emit } = await bootWorkspace({
      panels: PANELS3,
      visible: ['artifacts', 'channel-finder'],
    });
    expect(railIds()).toEqual(['terminal', 'artifacts', 'channel-finder']);

    // Index 1 among the panel rows: after artifacts, before channel-finder.
    emit({ type: 'panel_visibility', panel: 'ariel', visible: true, index: 1 });
    expect(railIds()).toEqual(['terminal', 'artifacts', 'ariel', 'channel-finder']);

    emit({ type: 'panel_visibility', panel: 'ariel', visible: true, index: 2 });
    expect(railIds()).toEqual(['terminal', 'artifacts', 'ariel', 'channel-finder']);

    // Past the end clamps to the foot of the row's group.
    emit({ type: 'panel_visibility', panel: 'ariel', visible: false });
    emit({ type: 'panel_visibility', panel: 'ariel', visible: true, index: 9 });
    expect(railIds()).toEqual(['terminal', 'artifacts', 'channel-finder', 'ariel']);

    // An index naming a row of an earlier group lands at the head of the
    // row's own group: a row moves within its group only.
    emit({ type: 'panel_visibility', panel: 'ariel', visible: false });
    emit({ type: 'panel_visibility', panel: 'ariel', visible: true, index: 0 });
    expect(railIds()).toEqual(['terminal', 'artifacts', 'ariel', 'channel-finder']);

    // A Workspace panel comes back to its own group whatever the index.
    emit({ type: 'panel_visibility', panel: 'artifacts', visible: false });
    emit({ type: 'panel_visibility', panel: 'artifacts', visible: true, index: 9 });
    expect(railIds()).toEqual(['terminal', 'artifacts', 'ariel', 'channel-finder']);
  });

  test('Undo also reopens the tile the removal closed, and only that', async () => {
    const { api, emit, calls } = await bootWorkspace({ panels: PANELS3 });
    expect(dockedTiles(api)).toEqual(['artifacts']);

    // artifacts holds a tile; its removal closes it, and Undo brings it back.
    pressDelete('artifacts');
    await settled(() => expect(undoButton()).not.toBeNull());
    emit({ type: 'panel_visibility', panel: 'artifacts', visible: false });
    expect(dockedTiles(api)).not.toContain('artifacts');
    /** @type {HTMLElement} */ (undoButton()).click();
    await settled(() => expect(dockedTiles(api)).toContain('artifacts'));
    emit({ type: 'panel_visibility', panel: 'artifacts', visible: true, index: 0 });

    // channel-finder held no tile, so its Undo restores the rail entry alone.
    pressDelete('channel-finder');
    await settled(() => expect(undoButton()).not.toBeNull());
    emit({ type: 'panel_visibility', panel: 'channel-finder', visible: false });
    /** @type {HTMLElement} */ (undoButton()).click();
    await settled(() =>
      expect(visibilityBodies(calls).at(-1)).toEqual({
        panel: 'channel-finder',
        visible: true,
        index: 2,
      })
    );
    emit({ type: 'panel_visibility', panel: 'channel-finder', visible: true, index: 2 });
    await new Promise((resolve) => setTimeout(resolve, 0)); // the Undo's tail has run
    expect(railIds()).toContain('channel-finder');
    expect(dockedTiles(api)).not.toContain('channel-finder');
  });

  test('"Close Agent pane" offers Undo, which reopens the pane', async () => {
    await bootWorkspace();
    closeTerminalPanel.mockReturnValueOnce(true);

    rightClick(entry('terminal'));
    /** @type {HTMLElement} */ (menuRow('Close Agent pane')).click();

    expect(notice()?.textContent).toContain('Agent pane closed');
    openTerminalPanel.mockClear();
    /** @type {HTMLElement} */ (undoButton()).click();
    expect(openTerminalPanel).toHaveBeenCalledTimes(1);
  });

  test('no notice when there was no Agent pane to close', async () => {
    await bootWorkspace();
    closeTerminalPanel.mockReturnValueOnce(false);

    rightClick(entry('terminal'));
    /** @type {HTMLElement} */ (menuRow('Close Agent pane')).click();

    expect(notice()).toBeNull();
  });
});

/**
 * The row menu's "Move to" rows move a member's sidebar row between the
 * operator's groups (sidebar-groups.js). The groups live in this browser's
 * storage, which outlives a module reset, so every test starts and ends with
 * it empty.
 */
describe('Move to — the row menu moves a row between the sidebar groups', () => {
  const entry = (/** @type {string} */ id) =>
    /** @type {HTMLElement} */ (
      document.querySelector(`.panel-rail-button[data-panel-id="${id}"]`)
    );

  /** @param {string} name @returns {HTMLElement} */
  function sidebarGroup(name) {
    return /** @type {HTMLElement} */ (
      document.querySelector(`.sidebar-group[data-group="${name}"]`)
    );
  }

  beforeEach(() => localStorage.clear());
  afterEach(() => {
    document.body.dispatchEvent(new MouseEvent('pointerdown', { bubbles: true }));
    localStorage.clear();
  });

  test('"Move to Workspace" moves the row there, in this browser only', async () => {
    const { calls } = await bootWorkspace();
    rightClick(entry('ariel'));
    /** @type {HTMLElement} */ (menuRow('Move to Workspace')).click();

    expect(sidebarGroup('workspace').contains(entry('ariel'))).toBe(true);
    const stored = JSON.parse(localStorage.getItem('osprey-sidebar-groups') ?? '{}');
    expect(stored.placement).toEqual({ ariel: 'workspace' });
    // The groups are the operator's own: nothing reaches the server.
    expect(posts(calls)).toEqual([]);

    // The moved row offers the way back.
    rightClick(entry('ariel'));
    expect(menuLabels()).toContain('Move to Tools');
    expect(menuLabels()).not.toContain('Move to Workspace');
  });

  test('"Move to new group…" makes a group below the row’s own and opens its name', async () => {
    await bootWorkspace();
    rightClick(entry('artifacts'));
    /** @type {HTMLElement} */ (menuRow('Move to new group…')).click();

    const groups = [...document.querySelectorAll('#panel-rail > .sidebar-group')];
    const ids = groups.map((g) => g.getAttribute('data-group'));
    expect(ids).toEqual(['workspace', expect.stringMatching(/^g-[0-9a-f]{8}$/), 'tools']);
    expect(groups[1].contains(entry('artifacts'))).toBe(true);
    const field = groups[1].querySelector('.sidebar-group-rename');
    expect(field).not.toBeNull();
    expect(document.activeElement).toBe(field);
  });

  test('the Agent row offers no move: it stays first', async () => {
    await bootWorkspace();
    rightClick(entry('terminal'));
    expect(menuLabels().some((label) => label?.startsWith('Move to'))).toBe(false);
  });
});

describe('sidecar start failure and retry', () => {
  const FAILED = 'JUPYTER failed to start: boom';
  /** A config or start answer; its fields change between the steps of a test. @typedef {any} Answer */
  /** @type {Answer} */
  const STARTING = {
    url: null,
    available: false,
    state: 'starting',
    message: 'JUPYTER is starting',
  };
  /** @type {Answer} */
  const RUNNING = { url: '/panel/jupyter', available: true, state: 'running', message: null };
  /** @param {string} message @returns {Answer} */
  const failed = (message) => ({ url: null, available: false, state: 'failed', message });

  beforeEach(() => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval'] });
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  /**
   * Boot with the workspace and the notebook sidecar as members. `server`
   * is live: a test changes what the config route, the start route and the
   * sidecar's health answer between steps.
   * @param {{ config: any, start?: any, healthy?: boolean }} server
   */
  async function bootSidecar(server) {
    window.__OSPREY_PREFIX__ = '';
    renderContainer();
    /** @type {{url: string, opts: any}[]} */
    const calls = [];
    vi.stubGlobal(
      'fetch',
      vi.fn(async (/** @type {string} */ url, /** @type {any} */ o) => {
        calls.push({ url, opts: o });
        if (url === '/api/panels') {
          return jsonOk({
            enabled: ['artifacts', 'jupyter'],
            custom: [],
            default: null,
            visible: ['artifacts', 'jupyter'],
            active: null,
            labels: {},
          });
        }
        if (url === '/api/artifact-server')
          return jsonOk({ url: '/panel/artifacts', available: true });
        if (url === '/api/jupyter-server') return jsonOk(server.config);
        if (url === '/api/panels/jupyter/start') return jsonOk(server.start);
        if (url === '/panel/jupyter/api/status') {
          return server.healthy === false
            ? { ok: false, status: 502, statusText: 'Bad Gateway', json: async () => ({}) }
            : jsonOk({});
        }
        return jsonOk({ status: 'ok' });
      })
    );
    stubEventSource();
    const mod = await freshImport();
    await mod.initPanelManager('panel-manager');
    await settled(() => expect(calls.some((c) => c.url === '/api/jupyter-server')).toBe(true));
    const jupyter = /** @type {HTMLElement} */ (
      document.querySelector('[data-panel-id="jupyter"]')
    );
    const starts = () => calls.filter((c) => c.url === '/api/panels/jupyter/start');
    return { mod, calls, jupyter, starts };
  }

  test("a failed config answer renders the entry failed with the server's message", async () => {
    const { jupyter } = await bootSidecar({ config: failed(FAILED) });

    await settled(() => expect(jupyter.classList.contains('failed')).toBe(true));
    expect(jupyter.classList.contains('disabled')).toBe(false);
    expect(jupyter.title).toBe(FAILED);
  });

  test('clicking a failed entry POSTs the start endpoint once', async () => {
    const server = { config: failed(FAILED), start: STARTING };
    const { jupyter, starts } = await bootSidecar(server);
    await settled(() => expect(jupyter.classList.contains('failed')).toBe(true));

    jupyter.click();

    await settled(() => expect(starts()).toHaveLength(1));
    expect(starts()[0].opts).toMatchObject({ method: 'POST' });
    await settled(() => expect(jupyter.title).toBe('JUPYTER is starting'));
    expect(jupyter.classList.contains('failed')).toBe(false);
    expect(jupyter.classList.contains('disabled')).toBe(true);
  });

  test('a second click while starting sends nothing', async () => {
    const server = { config: failed(FAILED), start: STARTING };
    const { jupyter, starts, calls } = await bootSidecar(server);
    await settled(() => expect(jupyter.classList.contains('failed')).toBe(true));
    jupyter.click();
    await settled(() => expect(jupyter.title).toBe('JUPYTER is starting'));
    const before = calls.length;

    jupyter.click();
    await vi.advanceTimersByTimeAsync(0);

    expect(starts()).toHaveLength(1);
    expect(calls.slice(before).filter((c) => c.opts?.method === 'POST')).toEqual([]);
  });

  test('a start that settles running enables and activates the panel', async () => {
    const server = { config: failed(FAILED), start: STARTING };
    const { jupyter, calls } = await bootSidecar(server);
    await settled(() => expect(jupyter.classList.contains('failed')).toBe(true));
    jupyter.click();
    await settled(() => expect(jupyter.title).toBe('JUPYTER is starting'));

    server.config = RUNNING;
    await vi.advanceTimersByTimeAsync(1000);

    await settled(() => expect(jupyter.classList.contains('disabled')).toBe(false));
    await settled(() => expect(jupyter.classList.contains('active')).toBe(true));
    expect(jupyter.classList.contains('failed')).toBe(false);
    expect(jupyter.title).toBe('Jupyter · right-click for actions');
    expect(calls.some((c) => c.url === '/api/panel-focus' && c.opts?.method === 'POST')).toBe(true);
  });

  test('a start that settles failed shows the new reason', async () => {
    const server = { config: failed(FAILED), start: STARTING };
    const { jupyter } = await bootSidecar(server);
    await settled(() => expect(jupyter.classList.contains('failed')).toBe(true));
    jupyter.click();
    await settled(() => expect(jupyter.title).toBe('JUPYTER is starting'));

    server.config = failed('JUPYTER failed to start: again');
    await vi.advanceTimersByTimeAsync(1000);

    await settled(() => expect(jupyter.title).toBe('JUPYTER failed to start: again'));
    expect(jupyter.classList.contains('failed')).toBe(true);
  });

  test('a panel that goes down and reports failed turns failed without a reload', async () => {
    const server = { config: RUNNING, healthy: true };
    const { jupyter } = await bootSidecar(server);
    await settled(() => expect(jupyter.classList.contains('disabled')).toBe(false));
    await vi.advanceTimersByTimeAsync(500); // startup poll → maintenance interval

    server.healthy = false;
    server.config = failed('JUPYTER failed to start: exited with status 1');
    await vi.advanceTimersByTimeAsync(10000);

    await settled(() => expect(jupyter.classList.contains('failed')).toBe(true));
    expect(jupyter.title).toBe('JUPYTER failed to start: exited with status 1');
  });

  test('the context menu and drag refuse a failed entry', async () => {
    const { jupyter } = await bootSidecar({ config: failed(FAILED) });
    await settled(() => expect(jupyter.classList.contains('failed')).toBe(true));

    expect(rightClick(jupyter).defaultPrevented).toBe(false);
    expect(document.querySelector('.rail-context-item')).toBeNull();

    const drag = new Event('dragstart', { bubbles: true, cancelable: true });
    jupyter.dispatchEvent(drag);
    expect(drag.defaultPrevented).toBe(true);
    expect(jupyter.classList.contains('dragging')).toBe(false);
  });
});

describe('panel frames — agent-writable panels are opaque, first-party panels keep their attributes', () => {
  const FACTORY_PATH =
    '../../../src/osprey/interfaces/web_terminal/static/js/panel-iframe-factory.js';
  const FIRST_PARTY =
    'allow-scripts allow-same-origin allow-popups allow-forms allow-modals allow-downloads';
  const OPAQUE = 'allow-scripts allow-popups allow-modals allow-downloads';

  /** @param {string} id @returns {HTMLIFrameElement | null} */
  const frame = (id) => document.querySelector(`iframe[data-panel-id="${id}"]`);
  /** @param {HTMLIFrameElement} el */
  const tokens = (el) => (el.getAttribute('sandbox') || '').split(/\s+/).filter(Boolean);

  /**
   * Boot a hub showing one panel. `cap` answers `/api/panels/<id>/cap`.
   * @param {{ enabled?: string[], custom?: any[], visible: string[],
   *           cap?: (id: string) => any, prefix?: string }} opts
   */
  async function bootOne({ enabled = [], custom = [], visible, cap, prefix = '' }) {
    window.__OSPREY_PREFIX__ = prefix;
    renderContainer();
    /** @type {string[]} */
    const capCalls = [];
    vi.stubGlobal(
      'fetch',
      vi.fn(async (/** @type {string} */ url) => {
        if (url === `${prefix}/api/panels`) {
          return jsonOk({ enabled, custom, default: null, visible, active: null, labels: {} });
        }
        const m = url.match(/\/api\/panels\/([^/]+)\/cap$/);
        if (m) {
          capCalls.push(url);
          if (!cap)
            return { ok: false, status: 404, statusText: 'Not Found', json: async () => ({}) };
          return cap(decodeURIComponent(m[1]));
        }
        if (url.endsWith('/api/artifact-server')) {
          return jsonOk({ url: `${prefix}/panel/artifacts`, available: true });
        }
        return jsonOk({});
      })
    );
    const sse = stubEventSource();
    const mod = await freshImport();
    await mod.initPanelManager('panel-manager');
    // The operator opens the panel: its frame is built on first activation.
    const button = document.querySelector(`.panel-rail-button[data-panel-id="${visible[0]}"]`);
    if (button instanceof HTMLElement && !frame(visible[0])) button.click();
    return { mod, capCalls, sse };
  }

  const discovered = {
    id: 'agent-bundle',
    label: 'BUNDLE',
    url: '/panel-static/agent-bundle/',
    healthEndpoint: null,
    path: '/',
    discovered: true,
  };

  test("an OSPREY catalog panel is framed at once with today's attributes and asks for no cap", async () => {
    const { capCalls } = await bootOne({ enabled: ['artifacts'], visible: ['artifacts'] });
    await settled(() => expect(frame('artifacts')).not.toBeNull());
    const el = /** @type {HTMLIFrameElement} */ (frame('artifacts'));
    expect(el.getAttribute('sandbox')).toBe(FIRST_PARTY);
    expect(new URL(el.src).pathname).toBe('/panel/artifacts');
    expect(capCalls).toEqual([]);
  });

  test('a discovered bundle is framed at its cap URL with no same-origin and no forms', async () => {
    const capUrl = '/panel-static/agent-bundle/~c/CAPTOKEN/';
    const { capCalls } = await bootOne({
      custom: [discovered],
      visible: ['agent-bundle'],
      cap: (id) => jsonOk({ id, cap_url: capUrl }),
    });
    await settled(() => expect(frame('agent-bundle')?.dataset.panelFrame).toBe('opaque'));
    const el = /** @type {HTMLIFrameElement} */ (frame('agent-bundle'));
    expect(el.getAttribute('sandbox')).toBe(OPAQUE);
    expect(tokens(el)).not.toContain('allow-same-origin');
    expect(tokens(el)).not.toContain('allow-forms');
    const src = new URL(el.src);
    expect(src.pathname).toBe(capUrl);
    expect(src.searchParams.get('embedded')).toBe('true');
    expect(capCalls).toEqual(['/api/panels/agent-bundle/cap']);
  });

  test('the cap request rides the per-user prefix and the cap URL is used verbatim', async () => {
    const capUrl = '/u/alice/panel-static/agent-bundle/~c/CAPTOKEN/';
    const { capCalls } = await bootOne({
      prefix: '/u/alice',
      custom: [{ ...discovered, url: '/u/alice/panel-static/agent-bundle/' }],
      visible: ['agent-bundle'],
      cap: (id) => jsonOk({ id, cap_url: capUrl }),
    });
    await settled(() => expect(frame('agent-bundle')?.dataset.panelFrame).toBe('opaque'));
    expect(new URL(/** @type {HTMLIFrameElement} */ (frame('agent-bundle')).src).pathname).toBe(
      capUrl
    );
    expect(capCalls).toEqual(['/u/alice/api/panels/agent-bundle/cap']);
  });

  test("a runtime panel's subpath lands under its cap with no doubled slash", async () => {
    const capUrl = '/panel/agent-dash/~c/CAPTOKEN/';
    await bootOne({
      custom: [
        {
          id: 'agent-dash',
          label: 'DASH',
          url: '/panel/agent-dash',
          healthEndpoint: null,
          path: '/ui/',
        },
      ],
      visible: ['agent-dash'],
      cap: (id) => jsonOk({ id, cap_url: capUrl }),
    });
    await settled(() => expect(frame('agent-dash')?.dataset.panelFrame).toBe('opaque'));
    const el = /** @type {HTMLIFrameElement} */ (frame('agent-dash'));
    expect(new URL(el.src).pathname).toBe('/panel/agent-dash/~c/CAPTOKEN/ui/');
    expect(el.getAttribute('sandbox')).toBe(OPAQUE);
  });

  test('a panel registered at runtime over SSE is framed opaque at its cap', async () => {
    const capUrl = '/panel/live-dash/~c/CAPTOKEN/';
    const { mod, sse } = await bootOne({
      enabled: ['artifacts'],
      visible: ['artifacts'],
      cap: (id) => jsonOk({ id, cap_url: capUrl }),
    });
    sse.emit({
      type: 'panel_register',
      id: 'live-dash',
      label: 'LIVE',
      url: '/panel/live-dash',
      healthEndpoint: null,
      path: '/',
    });
    await settled(() =>
      expect(document.querySelector('.panel-rail-button[data-panel-id="live-dash"]')).not.toBeNull()
    );
    mod.navigateAndActivatePanel('live-dash', '');
    await settled(() => expect(frame('live-dash')?.dataset.panelFrame).toBe('opaque'));
    const el = /** @type {HTMLIFrameElement} */ (frame('live-dash'));
    expect(el.getAttribute('sandbox')).toBe(OPAQUE);
    expect(new URL(el.src).pathname).toBe(capUrl);
  });

  test("a config-defined custom panel (the hub mints no cap: 404) keeps today's attributes", async () => {
    await bootOne({
      custom: [
        {
          id: 'results',
          label: 'RESULTS',
          url: '/panel/results',
          healthEndpoint: null,
          path: '/',
          configDefined: true,
        },
      ],
      visible: ['results'],
    });
    await settled(() => expect(frame('results')?.dataset.panelFrame).toBe('first-party'));
    const el = /** @type {HTMLIFrameElement} */ (frame('results'));
    expect(el.getAttribute('sandbox')).toBe(FIRST_PARTY);
    expect(new URL(el.src).pathname).toBe('/panel/results');
  });

  test.each([
    [
      'a 500',
      () => ({ ok: false, status: 500, statusText: 'Server Error', json: async () => ({}) }),
    ],
    [
      'a 401',
      () => ({ ok: false, status: 401, statusText: 'Unauthorized', json: async () => ({}) }),
    ],
    ['a body with no cap_url', () => jsonOk({})],
    [
      'a network failure',
      () => {
        throw new TypeError('Failed to fetch');
      },
    ],
  ])(
    '%s from the cap mint fails closed: opaque sandbox, never same-origin',
    async (_name, answer) => {
      await bootOne({ custom: [discovered], visible: ['agent-bundle'], cap: answer });
      await settled(() => expect(frame('agent-bundle')?.dataset.panelFrame).toBe('opaque'));
      const el = /** @type {HTMLIFrameElement} */ (frame('agent-bundle'));
      expect(el.getAttribute('sandbox')).toBe(OPAQUE);
      expect(new URL(el.src).pathname).toBe('/panel-static/agent-bundle/');
    }
  );

  test('the sandbox is restrictive before the hub answers, and no src is set yet', async () => {
    /** @type {(v: any) => void} */
    let release = () => {};
    const gate = new Promise((resolve) => {
      release = resolve;
    });
    await bootOne({
      custom: [discovered],
      visible: ['agent-bundle'],
      cap: async (id) => {
        await gate;
        return jsonOk({ id, cap_url: '/panel-static/agent-bundle/~c/T/' });
      },
    });
    await settled(() => expect(frame('agent-bundle')).not.toBeNull());
    const el = /** @type {HTMLIFrameElement} */ (frame('agent-bundle'));
    expect(el.dataset.panelFrame).toBe('pending');
    expect(el.getAttribute('sandbox')).toBe(OPAQUE);
    expect(el.getAttribute('src')).toBeNull();
    release(undefined);
    await settled(() => expect(el.dataset.panelFrame).toBe('opaque'));
  });

  test('an opaque panel receives the theme (target "*") and never the session', async () => {
    await bootOne({
      custom: [discovered],
      visible: ['agent-bundle'],
      cap: (id) => jsonOk({ id, cap_url: '/panel-static/agent-bundle/~c/T/' }),
    });
    await settled(() => expect(frame('agent-bundle')?.dataset.panelFrame).toBe('opaque'));
    const el = /** @type {HTMLIFrameElement} */ (frame('agent-bundle'));
    const win = /** @type {Window} */ (el.contentWindow);
    const post = vi.spyOn(win, 'postMessage').mockImplementation(() => {});
    el.dispatchEvent(new Event('load'));

    const calls = post.mock.calls.map(([msg, target]) => ({
      type: /** @type {any} */ (msg).type,
      target,
    }));
    expect(calls).toContainEqual({ type: 'osprey-theme-change', target: '*' });
    // Everything else is addressed to the hub's own origin, which an opaque
    // document (origin "null") never matches, so the browser drops it.
    for (const c of calls.filter((c) => c.type !== 'osprey-theme-change')) {
      expect(c.target).toBe(window.location.origin);
    }
  });

  // ---- D8: a panel keeps a working cap after navigation, recovery, expiry ----

  const TTL_MS = 12 * 60 * 60 * 1000;

  /**
   * A cap answer minting a fresh cap per call, the way the hub does: the
   * token's first segment is its expiry (epoch seconds, hex), 12 h out.
   * @param {string} base  the panel's plain base, ending in `/`
   */
  function mintingCaps(base) {
    let n = 0;
    /** @type {string[]} */
    const minted = [];
    const cap = (/** @type {string} */ id) => {
      n += 1;
      const expiry = Math.floor((Date.now() + TTL_MS) / 1000).toString(16);
      const url = `${base}~c/${expiry}.T${n}.S${n}/`;
      minted.push(url);
      return jsonOk({ id, cap_url: url });
    };
    return { cap, minted };
  }

  const agentDash = {
    id: 'agent-dash',
    label: 'DASH',
    url: '/panel/agent-dash',
    healthEndpoint: null,
    path: '/',
  };

  test('re-navigating an opaque panel frames the new address under a freshly minted cap', async () => {
    const { cap, minted } = mintingCaps('/panel/agent-dash/');
    const { mod, capCalls } = await bootOne({ custom: [agentDash], visible: ['agent-dash'], cap });
    await settled(() => expect(frame('agent-dash')?.dataset.panelFrame).toBe('opaque'));
    const el = /** @type {HTMLIFrameElement} */ (frame('agent-dash'));
    expect(new URL(el.src).pathname).toBe(minted[0]);

    mod.navigateAndActivatePanel('agent-dash', '/panel/agent-dash/runs/42.html');
    await settled(() => expect(capCalls).toHaveLength(2));
    await settled(() => expect(new URL(el.src).pathname).toBe(`${minted[1]}runs/42.html`));
    expect(el.dataset.panelFrame).toBe('opaque');
    expect(el.getAttribute('sandbox')).toBe(OPAQUE);
    // Never the bare, uncapped address.
    expect(new URL(el.src).pathname.startsWith('/panel/agent-dash/runs')).toBe(false);
  });

  test.each([
    '/api/panels',
    '/panel/ariel/',
    '/panel/artifacts/../ariel/',
    '/panel/artifacts/%2e%2e/ariel/',
    '/panel/artifacts//evil',
    'http://evil.example/panel/artifacts/',
    'javascript:alert(1)',
  ])('a navigation that leaves the panel moves nothing: %s', async (url) => {
    const { mod, capCalls } = await bootOne({ enabled: ['artifacts'], visible: ['artifacts'] });
    await settled(() => expect(frame('artifacts')).not.toBeNull());
    const el = /** @type {HTMLIFrameElement} */ (frame('artifacts'));
    const before = el.src;
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});

    mod.navigateAndActivatePanel('artifacts', url);
    await new Promise((r) => setTimeout(r, 5));

    expect(warn).toHaveBeenCalledTimes(1);
    expect(el.src).toBe(before);
    expect(capCalls).toEqual([]);
    warn.mockRestore();
  });

  test('re-navigating a first-party catalog panel keeps the plain URL and asks for no cap', async () => {
    const { mod, capCalls } = await bootOne({ enabled: ['artifacts'], visible: ['artifacts'] });
    await settled(() => expect(frame('artifacts')).not.toBeNull());
    const el = /** @type {HTMLIFrameElement} */ (frame('artifacts'));
    mod.navigateAndActivatePanel('artifacts', '/panel/artifacts/item/7');
    await settled(() => expect(new URL(el.src).pathname).toBe('/panel/artifacts/item/7'));
    expect(el.getAttribute('sandbox')).toBe(FIRST_PARTY);
    expect(capCalls).toEqual([]);
  });

  test('a recovered opaque panel loads its held address through a fresh cap', async () => {
    vi.useFakeTimers({
      toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'Date'],
    });
    try {
      vi.setSystemTime(new Date(2026, 0, 5, 9, 0, 0));
      const { cap, minted } = mintingCaps('/panel/agent-dash/');
      let down = false;
      const healthy = {
        ...agentDash,
        healthEndpoint: '/health',
      };
      window.__OSPREY_PREFIX__ = '';
      renderContainer();
      /** @type {string[]} */
      const capCalls = [];
      vi.stubGlobal(
        'fetch',
        vi.fn(async (/** @type {string} */ url) => {
          if (url === '/api/panels') {
            return jsonOk({
              enabled: [],
              custom: [healthy],
              default: null,
              visible: ['agent-dash'],
              active: null,
              labels: {},
            });
          }
          if (url === '/api/panels/agent-dash/cap') {
            capCalls.push(url);
            return cap('agent-dash');
          }
          if (url === '/panel/agent-dash/health') {
            return down
              ? { ok: false, status: 503, statusText: 'Down', json: async () => ({}) }
              : jsonOk({ status: 'ok' });
          }
          return jsonOk({});
        })
      );
      stubEventSource();
      const mod = await freshImport();
      await mod.initPanelManager('panel-manager');
      const button = /** @type {HTMLElement} */ (
        document.querySelector('.panel-rail-button[data-panel-id="agent-dash"]')
      );
      await settled(() => expect(button.classList.contains('disabled')).toBe(false));
      if (!frame('agent-dash')) button.click();
      await settled(() => expect(frame('agent-dash')?.dataset.panelFrame).toBe('opaque'));
      const el = /** @type {HTMLIFrameElement} */ (frame('agent-dash'));
      expect(new URL(el.src).pathname).toBe(minted[0]);
      await vi.advanceTimersByTimeAsync(1000);

      // The panel stops answering; an address sent meanwhile is held.
      down = true;
      await vi.advanceTimersByTimeAsync(10_000);
      mod.navigateAndActivatePanel('agent-dash', '/panel/agent-dash/held.html');
      await vi.advanceTimersByTimeAsync(0);
      expect(capCalls).toHaveLength(1);
      expect(new URL(el.src).pathname).toBe(minted[0]);

      // It answers again: the held address loads under a cap minted now.
      down = false;
      await vi.advanceTimersByTimeAsync(10_000);
      await settled(() => expect(capCalls).toHaveLength(2));
      await settled(() => expect(new URL(el.src).pathname).toBe(`${minted[1]}held.html`));
      expect(el.getAttribute('sandbox')).toBe(OPAQUE);
    } finally {
      vi.useRealTimers();
    }
  });

  test('a panel left open renews its cap before expiry and never loads an expired one', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'Date'] });
    try {
      vi.setSystemTime(new Date(2026, 0, 5, 9, 0, 0));
      const { cap, minted } = mintingCaps('/panel/agent-dash/');
      const { capCalls } = await bootOne({ custom: [agentDash], visible: ['agent-dash'], cap });
      await settled(() => expect(frame('agent-dash')?.dataset.panelFrame).toBe('opaque'));
      const el = /** @type {HTMLIFrameElement} */ (frame('agent-dash'));
      const { CAP_RENEW_MARGIN_MS } = await import(FACTORY_PATH);
      const firstExpiry = parseInt(minted[0].split('~c/')[1], 16) * 1000;

      await vi.advanceTimersByTimeAsync(firstExpiry - CAP_RENEW_MARGIN_MS - Date.now() - 1000);
      expect(capCalls).toHaveLength(1);
      expect(new URL(el.src).pathname).toBe(minted[0]);

      await vi.advanceTimersByTimeAsync(2000);
      await settled(() => expect(capCalls).toHaveLength(2));
      await settled(() => expect(new URL(el.src).pathname).toBe(minted[1]));
      expect(Date.now()).toBeLessThan(firstExpiry);
      expect(el.getAttribute('sandbox')).toBe(OPAQUE);

      // And again for the renewed cap: the panel can stay open indefinitely.
      await vi.advanceTimersByTimeAsync(TTL_MS);
      await settled(() => expect(capCalls).toHaveLength(3));
      expect(new URL(el.src).pathname).toBe(minted[2]);
    } finally {
      vi.useRealTimers();
    }
  });

  test('capTarget rebases under the cap and leaves foreign targets alone', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => jsonOk({}))
    );
    await freshImport();
    const { capTarget } = await import(FACTORY_PATH);
    const cap = '/panel/p/~c/T/';
    expect(capTarget('/panel/p', '/panel/p', cap)).toBe(cap);
    expect(capTarget('/panel/p/', '/panel/p', cap)).toBe(cap);
    expect(capTarget('/panel/p/ui/x.html?a=1', '/panel/p', cap)).toBe(
      '/panel/p/~c/T/ui/x.html?a=1'
    );
    expect(capTarget('/panel-static/b/sub/', '/panel-static/b/', '/panel-static/b/~c/T/')).toBe(
      '/panel-static/b/~c/T/sub/'
    );
    // A sibling id sharing a prefix is not under the base.
    expect(capTarget('/panel/pp/x', '/panel/p', cap)).toBe('/panel/pp/x');
  });
});
