// @ts-check
/**
 * Unit tests for session.js's activity-strip SSE wiring.
 *
 * The session page runs no panel-manager, so the strip's registration on
 * panel-manager's seam never fires there; `wireActivityStrip` is the page's
 * own subscription to `GET /api/files/events`. These tests drive it through the
 * real api.js stream with a stubbed global EventSource and a fake strip — no
 * network, no timers:
 *
 *   - an agent_activity frame reaches the strip's handleActivity verbatim
 *   - the shared stream's other frame types (file/panel events) are ignored
 *   - a frame that failed to parse (api.js hands the raw string through) is
 *     ignored without throwing, as are null/array/targetless ones
 *   - the wiring subscribes to the right path, and its handle closes the source
 *
 *   npx vitest run tests/interfaces/web_terminal/session-strip.test.mjs
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const ENTRY_PATH = '../../../src/osprey/interfaces/web_terminal/static/js/session.js';
const STRIP_PATH = '../../../src/osprey/interfaces/web_terminal/static/js/activity-strip.js';

/** @typedef {import('../../../src/osprey/interfaces/web_terminal/static/js/panel-manager.js').AgentActivityEvent} AgentActivityFrame */

/**
 * A stubbed global EventSource: records every source api.js opens, and exposes
 * `emit` to deliver a server frame through the source's own onmessage — JSON
 * text, as the wire carries it, so api.js's parse (and its raw-string fallback
 * for text that is not JSON) runs for real.
 */
function stubEventSource() {
  /** @type {{url: string, closed: boolean, onmessage: ((e: {data: string}) => void) | null}[]} */
  const sources = [];
  class FakeEventSource {
    /** @param {string} url */
    constructor(url) {
      this.url = url;
      this.closed = false;
      /** @type {((e: {data: string}) => void) | null} */
      this.onmessage = null;
      /** @type {(() => void) | null} */
      this.onopen = null;
      /** @type {(() => void) | null} */
      this.onerror = null;
      this.readyState = 0;
      sources.push(this);
    }
    close() {
      this.closed = true;
    }
  }
  vi.stubGlobal('EventSource', FakeEventSource);
  /** @param {unknown} payload - a frame, or raw text sent as-is */
  const emit = (payload) => {
    const data = typeof payload === 'string' ? payload : JSON.stringify(payload);
    for (const src of sources) src.onmessage?.({ data });
  };
  return { sources, emit };
}

/** A strip that only records the frames handed to it. */
function fakeStrip() {
  /** @type {AgentActivityFrame[]} */
  const seen = [];
  return { seen, handleActivity: (/** @type {AgentActivityFrame} */ f) => { seen.push(f); } };
}

/**
 * @param {AgentActivityFrame['target']} target
 * @param {string} [tool]
 * @returns {AgentActivityFrame}
 */
function frame(target, tool = 'write_channel') {
  return { type: 'agent_activity', tool, target, ts: 1234 };
}

/** @type {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/session.js')} */
let Session;

beforeEach(async () => {
  // session.js runs its page boot on import: give it the elements it reaches
  // for (no #activity-strip mount — that boot path is not under test here)
  // and a fetch stub so the initial refresh never hits the network.
  document.body.innerHTML = `
    <div class="refresh-dot" id="refresh-dot"></div>
    <nav id="nav"><button class="pill active" data-view="agents">Agents</button></nav>
    <section class="view active" id="view-agents"></section>
    <div class="toast" id="toast"></div>
  `;
  vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({
    status: 200,
    ok: true,
    json: () => Promise.resolve({ total_events: 0, agents: [], tool_calls_by_agent: {} }),
  })));
  vi.resetModules();
  Session = await import(ENTRY_PATH);
});

afterEach(() => {
  vi.unstubAllGlobals();
  document.body.innerHTML = '';
});

describe('wireActivityStrip: subscription', () => {
  test('subscribes to the shared file-events stream and returns the handle', () => {
    const es = stubEventSource();
    const strip = fakeStrip();

    const handle = Session.wireActivityStrip(strip);

    expect(es.sources.map((src) => src.url)).toEqual(['/api/files/events']);
    handle.stop();
    expect(es.sources[0].closed).toBe(true);
  });
});

describe('wireActivityStrip: frame routing', () => {
  test.each([
    ['channel', frame({ kind: 'channel', detail: 'SR01:HCM1:SP' })],
    // Suppression is the strip's call, not the page's: every kind is forwarded.
    ['panel', frame({ kind: 'panel', panel: 'lattice' }, 'open_panel')],
    ['run', frame({ kind: 'run', detail: 'orm-42' }, 'run_plan')],
    ['artifact', frame({ kind: 'artifact', detail: 'orbit-plot.png' }, 'focus_artifact')],
  ])('a %s agent_activity frame reaches handleActivity verbatim', (_kind, f) => {
    const es = stubEventSource();
    const strip = fakeStrip();
    Session.wireActivityStrip(strip);

    es.emit(f);

    expect(strip.seen).toEqual([f]);
  });

  test('the shared stream\'s other frame types are ignored', () => {
    const es = stubEventSource();
    const strip = fakeStrip();
    Session.wireActivityStrip(strip);

    es.emit({ type: 'file_changed', path: '/tmp/x.py' });
    es.emit({ type: 'panel_focus', panel: 'lattice', source: 'agent' });
    es.emit({ type: 'panel_visibility', panel: 'okf', visible: false });

    expect(strip.seen).toEqual([]);
  });
});

describe('page boot: one strip on the shared mount', () => {
  test('bootActivityStrip is idempotent, so the session page reuses the module\'s own instance', async () => {
    // The mount exists before the import, so the module's self-boot claims it;
    // session.js then asks for the same instance instead of binding a second
    // strip to it (two strips would mean two history popovers per click).
    document.body.innerHTML = '<div id="activity-strip"></div>';
    vi.resetModules();
    const Strip = await import(STRIP_PATH);

    const first = Strip.bootActivityStrip();
    const second = Strip.bootActivityStrip();
    expect(first).not.toBeNull();
    expect(second).toBe(first);

    const mount = /** @type {HTMLElement} */ (document.getElementById('activity-strip'));
    mount.click();
    await Promise.resolve();
    await Promise.resolve();

    expect(document.querySelectorAll('.activity-history-popover').length).toBe(1);
    first?.closeHistory();
  });

  test('a page without the mount boots no strip', async () => {
    document.body.innerHTML = '';
    vi.resetModules();
    const Strip = await import(STRIP_PATH);

    expect(Strip.bootActivityStrip()).toBeNull();
  });
});

describe('wireActivityStrip: malformed payloads', () => {
  test('an unparseable frame arrives as a raw string and is ignored without throwing', () => {
    const es = stubEventSource();
    const strip = fakeStrip();
    Session.wireActivityStrip(strip);

    // api.js's onMessage fallback: JSON.parse failed, so the raw text comes through.
    expect(() => es.emit('{"type": "agent_activity", trunca')).not.toThrow();
    expect(() => es.emit('')).not.toThrow();
    expect(strip.seen).toEqual([]);
  });

  test('null, arrays and an agent_activity frame with no target are ignored', () => {
    const es = stubEventSource();
    const strip = fakeStrip();
    Session.wireActivityStrip(strip);

    es.emit(null);
    es.emit('undefined'); // not JSON: arrives as the raw string
    es.emit([{ type: 'agent_activity', tool: 'write_channel' }]);
    es.emit({ type: 'agent_activity', tool: 'write_channel' });

    expect(strip.seen).toEqual([]);
  });
});
