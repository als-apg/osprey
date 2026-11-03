/**
 * Contract tests for the panel event stream (panel-sse.js):
 *   npx vitest run tests/interfaces/web_terminal/panel-sse.test.mjs
 *
 * The browser half of the OSPREY op transport:
 *
 *   - the stream is the page's shared events URL (api.js pageEventsUrl), which
 *     carries this page's dock client id;
 *   - a `panel_op` frame is acted on only by the client it is addressed to: the
 *     op runner (panel-placement.js) runs it on the layout holder, the placed
 *     panels are surfaced, an `open` with a url navigates after placement, and
 *     the resulting layout (or the runner's named refusal) is POSTed to
 *     /api/panel-op-result, whose `rev` becomes the holder's ack baseline;
 *   - a `panel_report_request` frame is answered with an undeduped report;
 *   - an `open` of a panel in its own window brings the window forward;
 *   - the retired `panel_focus`, `panel_close` and `panel_arrange` frames do
 *     nothing.
 *
 * The stream is stubbed at api.js: createEventSource captures the handlers so
 * a test can deliver a frame by hand. The op runner, the holder and the POST
 * helpers are stubs at their module boundaries; the runner's own semantics are
 * pinned against the real holder in panel-placement.test.mjs.
 */

import { test, expect, describe, beforeEach, vi } from 'vitest';

const JS = '../../../src/osprey/interfaces/web_terminal/static/js';

const { stream, calls, spies, dock } = vi.hoisted(() => {
  /** @type {string[]} */
  const calls = [];
  const record =
    (/** @type {string} */ name) =>
    (/** @type {any[]} */ ...args) => {
      calls.push(`${name}:${args.filter((a) => typeof a === 'string').join(',')}`);
    };
  return {
    stream: { handlers: /** @type {any} */ (null), url: /** @type {string|null} */ (null) },
    calls,
    dock: { holder: /** @type {any} */ (null) },
    spies: {
      ensureRailMembership: vi.fn(),
      flashAgentTile: vi.fn(),
      hidePanel: vi.fn(),
      runPanelOp: vi.fn(),
      postPanelOpResult: vi.fn(),
      postLayoutReport: vi.fn(),
      noteReported: vi.fn(),
      focusPopout: vi.fn(),
      record,
    },
  };
});

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/api.js', () => ({
  fetchJSON: vi.fn(async () => null),
  getPageClientId: () => 'me/1',
  pageEventsUrl: () => '/api/files/events?client_id=me%2F1',
  createEventSource: (/** @type {string} */ url, /** @type {any} */ handlers) => {
    stream.url = url;
    stream.handlers = handlers;
  },
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/dock-iframe.js', () => ({
  hidePanel: spies.hidePanel,
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/dock-sync.js', () => ({
  withEchoSuppressed: (/** @type {() => any} */ fn) => fn(),
  noteReported: spies.noteReported,
  postLayoutReport: spies.postLayoutReport,
  sameReport: (/** @type {any} */ a, /** @type {any} */ b) =>
    JSON.stringify(a) === JSON.stringify(b),
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/dock-apply.js', () => ({
  getLayoutHolder: () => dock.holder,
  TERMINAL_KEY: 'terminal',
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/panel-agent-attention.js', () => ({
  flashAgentGlow: vi.fn(),
  flashAgentTile: spies.flashAgentTile,
  badgePanelActivity: vi.fn(() => true),
  restoreAgentBadges: vi.fn(),
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/panel-lifecycle.js', () => ({
  ensureRailMembership: spies.ensureRailMembership,
  addPanel: vi.fn(),
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/panel-rail.js', () => ({
  entryIds: () => [],
  placeEntry: vi.fn(),
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/panel-placement.js', () => ({
  runPanelOp: spies.runPanelOp,
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/panel-commands.js', () => ({
  postPanelOpResult: spies.postPanelOpResult,
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/popout.js', () => ({
  focusPopout: spies.focusPopout,
}));

/** @param {string} id @param {string[]} tabs @param {string} [showing] */
function tile(id, tabs, showing = tabs[0]) {
  return { tile: id, tabs, showing };
}
/** @param {any[]} items */
function row(...items) {
  return { split: 'row', items };
}

const BEFORE = row(tile('s1', ['a'], 'a'), tile('t1', ['terminal']));
const WITH_B = row(tile('s1', ['a', 'b'], 'b'), tile('t1', ['terminal']));

/**
 * A stand-in holder: the runner stub swaps its layout.
 * @param {any} [layout] @param {string[]} [popouts]
 */
function fakeHolder(layout = BEFORE, popouts = []) {
  return { layout, popouts, ackedRev: 3, dirty: false, ack: vi.fn() };
}

/** The dedupe key dock-sync compares reports by. @param {any} layout @param {string[]} [popouts] */
const key = (/** @type {any} */ layout, /** @type {string[]} */ popouts = []) => ({
  layout,
  popouts,
});

/**
 * Make the runner stub apply `after` to the holder and focus `focus`.
 * @param {any} after @param {string | null} [focus]
 */
function runnerApplies(after, focus = null) {
  spies.runPanelOp.mockImplementation(() => {
    calls.push('run');
    dock.holder.layout = after;
    return { result: { layout: after, changed: true, focus } };
  });
}

/** Let the handler's awaited POSTs settle. */
const settle = () => new Promise((r) => setTimeout(r, 0));

/** @type {any} */
let deps;

/** Deliver one frame to the stream's message handler. @param {any} frame */
function deliver(frame) {
  stream.handlers.onMessage(frame);
}

/** @param {Record<string, any>} [over] */
function opFrame(over = {}) {
  return {
    type: 'panel_op',
    request_id: 'r1',
    target_client: 'me/1',
    op: 'open',
    args: { member: 'b' },
    source: 'agent',
    ...over,
  };
}

beforeEach(async () => {
  vi.resetModules();
  calls.length = 0;
  for (const spy of [
    spies.ensureRailMembership,
    spies.flashAgentTile,
    spies.hidePanel,
    spies.runPanelOp,
    spies.postPanelOpResult,
    spies.postLayoutReport,
    spies.noteReported,
    spies.focusPopout,
  ]) {
    spy.mockReset();
  }
  spies.ensureRailMembership.mockImplementation(spies.record('member'));
  spies.flashAgentTile.mockImplementation(spies.record('glow'));
  spies.postPanelOpResult.mockImplementation(async (/** @type {any} */ body) => {
    calls.push(body.refusal ? `refuse:${body.refusal}` : 'answer');
    return { status: 'ok', layout: body.layout ?? null, rev: 7 };
  });
  spies.postLayoutReport.mockImplementation(async () => ({ status: 'ok', rev: 9 }));
  dock.holder = fakeHolder();
  const sse = await import(`${JS}/panel-sse.js`);
  deps = {
    isKnown: vi.fn((id) => id !== 'ghost'),
    isHealthy: vi.fn(() => true),
    getRailEl: () => document.createElement('nav'),
    memberSet: new Set(['a']),
    dropMember: vi.fn(),
    activate: vi.fn(spies.record('activate')),
    navigate: vi.fn(spies.record('navigate')),
    getActive: vi.fn(() => null),
    resetActive: vi.fn(),
    clearActive: vi.fn(),
    renderEmpty: vi.fn(),
    reportActivity: vi.fn(),
  };
  sse.subscribePanelEvents(deps);
});

describe('the event stream', () => {
  test('carries this page dock client id, encoded', () => {
    expect(stream.url).toBe('/api/files/events?client_id=me%2F1');
  });
});

describe('panel_op — addressed to one dock client', () => {
  test('a frame addressed to another client is ignored', async () => {
    deliver(opFrame({ target_client: 'someone-else' }));
    await settle();
    expect(spies.runPanelOp).not.toHaveBeenCalled();
    expect(spies.postPanelOpResult).not.toHaveBeenCalled();
  });

  test('the addressed client runs the op and answers with the holder layout', async () => {
    runnerApplies(WITH_B, 's1');
    const frame = opFrame();
    deliver(frame);
    await settle();
    expect(spies.runPanelOp).toHaveBeenCalledWith(frame);
    expect(spies.postPanelOpResult).toHaveBeenCalledWith({
      request_id: 'r1',
      client_id: 'me/1',
      layout: WITH_B,
      popouts: [],
    });
  });

  test('the answer rev becomes the holder ack baseline', async () => {
    runnerApplies(WITH_B, 's1');
    deliver(opFrame());
    await settle();
    expect(dock.holder.ack).toHaveBeenCalledWith(7);
  });

  test("the acknowledged answer becomes dock-sync's dedupe baseline", async () => {
    // Without it a person undoing the agent's open lands on the stale
    // baseline, is deduped away, and the server keeps the panel open.
    runnerApplies(WITH_B, 's1');
    deliver(opFrame());
    await settle();
    expect(spies.noteReported).toHaveBeenCalledExactlyOnceWith(key(WITH_B));
  });

  test('an answer that did not land, or a refusal, leaves the dedupe baseline alone', async () => {
    runnerApplies(WITH_B, 's1');
    spies.postPanelOpResult.mockImplementation(async () => null);
    deliver(opFrame());
    await settle();
    spies.runPanelOp.mockReturnValue({ refusal: 'stale_base' });
    deliver(opFrame({ request_id: 'r2' }));
    await settle();
    expect(spies.noteReported).not.toHaveBeenCalled();
  });

  test('the sent layout is the baseline even when the holder moved on: the server holds it', async () => {
    runnerApplies(WITH_B, 's1');
    spies.postPanelOpResult.mockImplementation(async () => {
      dock.holder.layout = BEFORE;
      return { status: 'ok', rev: 7 };
    });
    deliver(opFrame());
    await settle();
    expect(spies.noteReported).toHaveBeenCalledExactlyOnceWith(key(WITH_B));
    expect(dock.holder.ack).not.toHaveBeenCalled();
  });

  test('a layout that moved on before the answer landed is not acked', async () => {
    runnerApplies(WITH_B, 's1');
    spies.postPanelOpResult.mockImplementation(async () => {
      dock.holder.layout = row(tile('s1', ['b'], 'b'), tile('t1', ['terminal']));
      return { status: 'ok', rev: 7 };
    });
    deliver(opFrame());
    await settle();
    expect(dock.holder.ack).not.toHaveBeenCalled();
  });

  test('a refusal is answered by name and surfaces nothing', async () => {
    spies.runPanelOp.mockReturnValue({ refusal: 'stale_base' });
    deliver(opFrame({ op: 'close_tile', args: { tile: 's1' }, base_rev: 2 }));
    await settle();
    expect(spies.postPanelOpResult).toHaveBeenCalledWith({
      request_id: 'r1',
      client_id: 'me/1',
      refusal: 'stale_base',
    });
    expect(deps.activate).not.toHaveBeenCalled();
    expect(dock.holder.ack).not.toHaveBeenCalled();
  });

  test('a refusal detail is passed on, capped at 500 characters', async () => {
    spies.runPanelOp.mockReturnValue({ refusal: 'too_many_for_side', detail: 'x'.repeat(900) });
    deliver(opFrame({ op: 'open_set' }));
    await settle();
    const body = spies.postPanelOpResult.mock.calls[0][0];
    expect(body.refusal).toBe('too_many_for_side');
    expect(body.detail).toHaveLength(500);
  });

  test('a runner that throws is answered op_failed, never left hanging', async () => {
    const error = vi.spyOn(console, 'error').mockImplementation(() => {});
    spies.runPanelOp.mockImplementation(() => {
      throw new Error('boom');
    });
    deliver(opFrame());
    await settle();
    expect(spies.postPanelOpResult.mock.calls[0][0]).toMatchObject({ refusal: 'op_failed' });
    error.mockRestore();
  });

  test('newly shown panels are activated, the focused tile last, then the answer goes out', async () => {
    const after = row(
      tile('s2', ['c'], 'c'),
      tile('s1', ['a', 'b'], 'b'),
      tile('t1', ['terminal'])
    );
    runnerApplies(after, 's1');
    deliver(
      opFrame({ op: 'open_set', args: { members: ['b', 'c'], as: 'side' }, source: undefined })
    );
    await settle();
    expect(calls).toEqual(['run', 'activate:c', 'activate:b', 'answer']);
  });

  test('a panel already showing is brought forward when the op focuses its tile', async () => {
    runnerApplies(BEFORE, 's1');
    deliver(opFrame({ args: { member: 'a' }, source: undefined }));
    await settle();
    expect(calls).toEqual(['run', 'activate:a', 'answer']);
  });

  test('an agent op glows each panel it surfaced', async () => {
    runnerApplies(WITH_B, 's1');
    deliver(opFrame());
    await settle();
    expect(calls).toEqual(['run', 'activate:b', 'glow:b', 'answer']);
  });

  test('an open with a url navigates the panel after placement', async () => {
    runnerApplies(WITH_B, 's1');
    deliver(opFrame({ args: { member: 'b', url: '/u/x/page' }, source: undefined }));
    await settle();
    expect(calls).toEqual(['run', 'activate:b', 'navigate:b,/u/x/page', 'answer']);
  });

  test('the op changes rail membership nowhere: the server sends that', async () => {
    runnerApplies(WITH_B, 's1');
    deliver(opFrame());
    await settle();
    expect(spies.ensureRailMembership).not.toHaveBeenCalled();
    expect(deps.dropMember).not.toHaveBeenCalled();
  });

  test('closing the active panel with nothing surfaced clears the accent and paints the empty pane', async () => {
    deps.getActive.mockReturnValue('a');
    runnerApplies(tile('t1', ['terminal']), 't1');
    deliver(opFrame({ op: 'close_tab', args: { key: 'a' }, source: undefined }));
    await settle();
    expect(deps.activate).not.toHaveBeenCalled();
    expect(deps.resetActive).toHaveBeenCalledTimes(1);
    expect(deps.renderEmpty).toHaveBeenCalledWith('No panels visible');
  });

  test('closing a tab reveals and activates the tab behind it', async () => {
    dock.holder = fakeHolder(row(tile('s1', ['a', 'b'], 'a'), tile('t1', ['terminal'])));
    // An activation that takes effect makes its panel the active one.
    let active = 'a';
    deps.getActive.mockImplementation(() => active);
    deps.activate.mockImplementation((/** @type {string} */ id) => {
      calls.push(`activate:${id}`);
      active = id;
    });
    runnerApplies(row(tile('s1', ['b'], 'b'), tile('t1', ['terminal'])), 's1');
    deliver(opFrame({ op: 'close_tab', args: { key: 'a' }, source: undefined }));
    await settle();
    expect(calls).toEqual(['run', 'activate:b', 'answer']);
    expect(deps.resetActive).not.toHaveBeenCalled();
  });
});

describe('surfacing — only an activation that took effect counts', () => {
  test('a refused activation clears the stale accent, but a tile still on screen is not painted over', async () => {
    // a held the view; the op replaced it with b, whose activation is refused
    // (an unhealthy panel): the active panel stays a, which is off screen.
    deps.getActive.mockReturnValue('a');
    runnerApplies(row(tile('s2', ['b']), tile('t1', ['terminal'])), 's2');
    deliver(opFrame({ op: 'open_arrangement' }));
    await settle();
    expect(deps.activate).toHaveBeenCalledWith('b');
    expect(deps.clearActive).toHaveBeenCalledTimes(1);
    expect(deps.resetActive).not.toHaveBeenCalled();
    expect(deps.renderEmpty).not.toHaveBeenCalled();
  });

  test('an activation that took effect keeps the new accent', async () => {
    let active = 'a';
    deps.getActive.mockImplementation(() => active);
    deps.activate.mockImplementation((/** @type {string} */ id) => {
      active = id;
    });
    runnerApplies(row(tile('s2', ['b']), tile('t1', ['terminal'])), 's2');
    deliver(opFrame({ op: 'open_arrangement' }));
    await settle();
    expect(active).toBe('b');
    expect(deps.clearActive).not.toHaveBeenCalled();
    expect(deps.resetActive).not.toHaveBeenCalled();
    expect(deps.renderEmpty).not.toHaveBeenCalled();
  });
});

describe('panel_report_request — a forced report', () => {
  test("reports the holder layout through dock-sync's poster, and acks the rev", async () => {
    deliver({ type: 'panel_report_request' });
    await settle();
    expect(spies.postLayoutReport).toHaveBeenCalledWith(BEFORE, []);
    expect(dock.holder.ack).toHaveBeenCalledWith(9);
  });

  test('reports even an unchanged layout: it is never deduped', async () => {
    deliver({ type: 'panel_report_request' });
    deliver({ type: 'panel_report_request' });
    await settle();
    expect(spies.postLayoutReport).toHaveBeenCalledTimes(2);
  });

  test('a page without a holder reports nothing', async () => {
    dock.holder = null;
    deliver({ type: 'panel_report_request' });
    await settle();
    expect(spies.postLayoutReport).not.toHaveBeenCalled();
  });

  test('a report that did not land acks nothing', async () => {
    spies.postLayoutReport.mockImplementation(async () => null);
    deliver({ type: 'panel_report_request' });
    await settle();
    expect(dock.holder.ack).not.toHaveBeenCalled();
    expect(spies.noteReported).not.toHaveBeenCalled();
  });

  test("an acknowledged forced report becomes dock-sync's dedupe baseline", async () => {
    deliver({ type: 'panel_report_request' });
    await settle();
    expect(spies.noteReported).toHaveBeenCalledExactlyOnceWith(key(BEFORE));
  });
});

describe('a panel in its own window', () => {
  test('an open brings the window forward, navigates, and answers the layout unchanged with the pop-outs', async () => {
    dock.holder = fakeHolder(BEFORE, ['b']);
    spies.runPanelOp.mockImplementation(() => ({
      result: { layout: BEFORE, changed: false },
      popout: 'b',
    }));
    deliver(opFrame({ args: { member: 'b', url: '/x' } }));
    await settle();
    expect(spies.focusPopout).toHaveBeenCalledExactlyOnceWith('b');
    expect(calls).toContain('navigate:b,/x');
    expect(calls).toContain('glow:b');
    expect(deps.activate).not.toHaveBeenCalled();
    expect(spies.postPanelOpResult).toHaveBeenCalledWith({
      request_id: 'r1',
      client_id: 'me/1',
      layout: BEFORE,
      popouts: ['b'],
    });
    expect(spies.noteReported).toHaveBeenCalledExactlyOnceWith(key(BEFORE, ['b']));
  });

  test('the pop-outs ride on a forced report too', async () => {
    dock.holder = fakeHolder(BEFORE, ['b']);
    deliver({ type: 'panel_report_request' });
    await settle();
    expect(spies.postLayoutReport).toHaveBeenCalledWith(BEFORE, ['b']);
  });
});

describe('the retired frames do nothing', () => {
  test('panel_focus no longer activates anything: a person focuses locally, the agent opens through panel_op', () => {
    deliver({ type: 'panel_focus', panel: 'b', source: 'agent' });
    deliver({ type: 'panel_focus', panel: 'b', url: '/x' });
    expect(deps.activate).not.toHaveBeenCalled();
    expect(deps.navigate).not.toHaveBeenCalled();
    expect(spies.flashAgentTile).not.toHaveBeenCalled();
  });

  test('panel_close no longer closes anything', () => {
    deliver({ type: 'panel_close', panel: 'a', source: 'agent' });
    expect(spies.hidePanel).not.toHaveBeenCalled();
    expect(deps.activate).not.toHaveBeenCalled();
  });

  test('panel_arrange no longer arranges anything', () => {
    deliver({ type: 'panel_arrange', tiles: ['a'] });
    expect(spies.runPanelOp).not.toHaveBeenCalled();
    expect(deps.activate).not.toHaveBeenCalled();
  });
});
