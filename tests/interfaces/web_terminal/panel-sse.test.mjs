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
 *   - a `panel_report_request` frame asks dock-sync's reporter for a report;
 *   - an `open` of a panel in its own window brings the window forward;
 *   - a `panel_visibility` hide closes the panel's tile and, when it held the
 *     view, hands the view to a panel still on screen without changing any
 *     other tile; only a dock left with no service tile opens the next member;
 *   - an `agent_activity` frame targeting a panel still badges its rail entry,
 *     and is relayed as `osprey-agent-activity` to that panel's mounted iframe
 *     or pop-out window only; any other target posts nothing;
 *   - the retired `panel_focus`, `panel_close` and `panel_arrange` frames do
 *     nothing.
 *
 * The stream is stubbed at api.js: createEventSource captures the handlers so
 * a test can deliver a frame by hand. The op runner and the POST helpers are
 * stubs at their module boundaries, and so is the holder for the op frames;
 * the runner's own semantics are pinned against the real holder in
 * panel-placement.test.mjs. A hide runs on a real holder.
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
      reportLayoutNow: vi.fn(),
      noteReported: vi.fn(),
      focusPopout: vi.fn(),
      popoutWindow: vi.fn(),
      badgePanelActivity: vi.fn(),
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
  reportLayoutNow: spies.reportLayoutNow,
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
  badgePanelActivity: spies.badgePanelActivity,
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
  popoutWindow: spies.popoutWindow,
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
/** An agent's open of b by the open rule: b replaced a in s1. */
const WITH_B = row(tile('s1', ['b'], 'b'), tile('t1', ['terminal']));

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
    spies.reportLayoutNow,
    spies.noteReported,
    spies.focusPopout,
    spies.popoutWindow,
    spies.badgePanelActivity,
  ]) {
    spy.mockReset();
  }
  spies.popoutWindow.mockReturnValue(null);
  spies.badgePanelActivity.mockReturnValue(true);
  document.body.replaceChildren();
  spies.ensureRailMembership.mockImplementation(spies.record('member'));
  spies.flashAgentTile.mockImplementation(spies.record('glow'));
  spies.postPanelOpResult.mockImplementation(async (/** @type {any} */ body) => {
    calls.push(body.refusal ? `refuse:${body.refusal}` : 'answer');
    return { status: 'ok', layout: body.layout ?? null, rev: 7 };
  });
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
      // A person dragged b to a tile of its own meanwhile.
      dock.holder.layout = row(tile('s2', ['b'], 'b'), tile('t1', ['terminal']));
      return { status: 'ok', rev: 7 };
    });
    deliver(opFrame());
    await settle();
    expect(dock.holder.ack).not.toHaveBeenCalled();
  });

  test('a refusal is answered by name and surfaces nothing', async () => {
    spies.runPanelOp.mockReturnValue({ refusal: 'stale_base' });
    deliver(opFrame({ op: 'close', args: { key: 'a' }, base_rev: 2 }));
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
      tile('s3', ['b'], 'b'),
      tile('s1', ['a'], 'a'),
      tile('t1', ['terminal'])
    );
    runnerApplies(after, 's3');
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

  test('an agent open that replaced the panel on view answers its layout and moves the view to the opened panel', async () => {
    // a held the view in s1; the open rule put b there in its place.
    let active = 'a';
    deps.getActive.mockImplementation(() => active);
    deps.activate.mockImplementation((/** @type {string} */ id) => {
      calls.push(`activate:${id}`);
      active = id;
    });
    runnerApplies(WITH_B, 's1');
    deliver(opFrame());
    await settle();
    // Only the opened panel is surfaced and glowed; the replaced one is
    // neither activated nor cleared: the view simply moved to b.
    expect(calls).toEqual(['run', 'activate:b', 'glow:b', 'answer']);
    expect(active).toBe('b');
    expect(deps.clearActive).not.toHaveBeenCalled();
    expect(deps.resetActive).not.toHaveBeenCalled();
    expect(spies.postPanelOpResult.mock.calls[0][0]).toMatchObject({
      layout: WITH_B,
      popouts: [],
    });
    expect(dock.holder.ack).toHaveBeenCalledWith(7);
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
    deliver(opFrame({ op: 'close', args: { key: 'a' }, source: undefined }));
    await settle();
    expect(deps.activate).not.toHaveBeenCalled();
    expect(deps.resetActive).toHaveBeenCalledTimes(1);
    expect(deps.renderEmpty).toHaveBeenCalledWith('No panels visible');
  });

  test('a close is applied and answered; the tile left on screen is neither opened into nor painted over', async () => {
    dock.holder = fakeHolder(row(tile('s1', ['a']), tile('s2', ['c']), tile('t1', ['terminal'])));
    deps.getActive.mockReturnValue('a');
    const after = row(tile('s2', ['c']), tile('t1', ['terminal']));
    runnerApplies(after);
    const frame = opFrame({ op: 'close', args: { key: 'a' } });
    deliver(frame);
    await settle();
    expect(spies.runPanelOp).toHaveBeenCalledWith(frame);
    expect(spies.postPanelOpResult).toHaveBeenCalledWith({
      request_id: 'r1',
      client_id: 'me/1',
      layout: after,
      popouts: [],
    });
    expect(dock.holder.ack).toHaveBeenCalledWith(7);
    // Nothing is activated in the closed panel's place (a panel with no tile
    // would replace c, by the open rule), and the accent stays on the closed
    // panel's row, one click away.
    expect(deps.activate).not.toHaveBeenCalled();
    expect(deps.renderEmpty).not.toHaveBeenCalled();
    expect(deps.clearActive).not.toHaveBeenCalled();
    expect(deps.resetActive).toHaveBeenCalledTimes(1);
  });
});

describe('an op the holder took is answered with its layout, whatever fails after', () => {
  // The op happened, so the server is told what the holder holds: a failure in
  // the follow-up steps is logged, never turned into silence or a refusal.
  const silenced = () => vi.spyOn(console, 'error').mockImplementation(() => {});

  test('a surfacing failure is logged and the layout still goes out and is acked', async () => {
    const error = silenced();
    runnerApplies(WITH_B, 's1');
    deps.activate.mockImplementation(() => {
      throw new Error('activate broke');
    });
    deliver(opFrame());
    await settle();
    expect(spies.postPanelOpResult).toHaveBeenCalledExactlyOnceWith({
      request_id: 'r1',
      client_id: 'me/1',
      layout: WITH_B,
      popouts: [],
    });
    expect(dock.holder.ack).toHaveBeenCalledWith(7);
    expect(error).toHaveBeenCalled();
    error.mockRestore();
  });

  test('a navigation failure after placement still answers the layout', async () => {
    const error = silenced();
    runnerApplies(WITH_B, 's1');
    deps.navigate.mockImplementation(() => {
      throw new Error('no iframe');
    });
    deliver(opFrame({ args: { member: 'b', url: '/u/x/page' } }));
    await settle();
    expect(spies.postPanelOpResult).toHaveBeenCalledExactlyOnceWith({
      request_id: 'r1',
      client_id: 'me/1',
      layout: WITH_B,
      popouts: [],
    });
    error.mockRestore();
  });

  test("a failure bringing a panel's own window forward still answers the layout", async () => {
    const error = silenced();
    dock.holder = fakeHolder(BEFORE, ['b']);
    spies.runPanelOp.mockImplementation(() => ({
      result: { layout: BEFORE, changed: false },
      popout: 'b',
    }));
    spies.focusPopout.mockImplementation(() => {
      throw new Error('window gone');
    });
    deliver(opFrame());
    await settle();
    expect(spies.postPanelOpResult).toHaveBeenCalledExactlyOnceWith({
      request_id: 'r1',
      client_id: 'me/1',
      layout: BEFORE,
      popouts: ['b'],
    });
    error.mockRestore();
  });

  test('a runner that throws after the holder took the op answers the layout, not op_failed', async () => {
    const error = silenced();
    spies.runPanelOp.mockImplementation(() => {
      dock.holder.layout = WITH_B;
      throw new Error('boom after the apply');
    });
    deliver(opFrame());
    await settle();
    expect(spies.postPanelOpResult).toHaveBeenCalledExactlyOnceWith({
      request_id: 'r1',
      client_id: 'me/1',
      layout: WITH_B,
      popouts: [],
    });
    error.mockRestore();
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

describe('panel_visibility hide — the view falls back, and no other tile changes', () => {
  // The hide closes the hidden panel's tile on a real holder (dock-iframe's
  // hidePanel is stubbed to the holder op it runs). What the view falls back
  // to is under test: activating a docked panel only focuses it, whereas
  // activating a panel with no tile puts it in the service tile used last,
  // REPLACING the panel that tile shows (the open rule) — a panel the operator
  // did not touch.

  /** @type {any} */
  let holder;

  beforeEach(async () => {
    const { createLayoutHolder } = /** @type {any} */ (
      await vi.importActual(`${JS}/dock-apply.js`)
    );
    const { closeKeys } = await import(`${JS}/dock-open.js`);
    holder = createLayoutHolder({ dockPx: { width: 1000, height: 600 } });
    dock.holder = holder;
    spies.hidePanel.mockImplementation((/** @type {string} */ id) => {
      calls.push(`hide:${id}`);
      holder.run((/** @type {any} */ layout) => closeKeys(layout, [id]));
    });
    // A hide drops the member, as panel-manager's dropMember does.
    deps.dropMember.mockImplementation((/** @type {string} */ id) => deps.memberSet.delete(id));
  });

  /**
   * Seed the dock with one service tile per entry of `shown`, left of the
   * terminal, each focused in turn: the last one's panel holds the view.
   * @param {Record<string, string>} shown  tile id → the panel it shows
   * @param {string[]} members  the rail
   */
  function seedView(shown, members) {
    const entries = Object.entries(shown);
    holder.restore(row(...entries.map(([id, key]) => tile(id, [key])), tile('t1', ['terminal'])));
    for (const [id, key] of entries) holder.show(id, key);
    deps.memberSet = new Set(members);
    deps.getActive.mockReturnValue(entries[entries.length - 1][1]);
  }

  /** The held layout's panels, tile by tile, in tree order. */
  const panelsByTile = () => {
    /** @param {any} node @returns {string[][]} */
    const walk = (node) => (node.tabs ? [node.tabs] : node.items.flatMap(walk));
    return walk(holder.layout);
  };

  /** @param {string} panel */
  const hide = (panel) => deliver({ type: 'panel_visibility', panel, visible: false });

  test('a hide of the panel on view hands the view to the panel still on screen: nothing opens, nothing is replaced', () => {
    // artifacts, first in the catalog, is a healthy member with no tile:
    // activating it would replace lattice.
    seedView({ s1: 'lattice', s2: 'ariel' }, ['artifacts', 'ariel', 'lattice']);

    hide('ariel');

    expect(calls).toEqual(['hide:ariel', 'activate:lattice']);
    expect(panelsByTile()).toEqual([['lattice'], ['terminal']]);
    expect(deps.resetActive).not.toHaveBeenCalled();
    expect(deps.renderEmpty).not.toHaveBeenCalled();
  });

  test('when the panel left on screen is not answering, the view is cleared and its tile is not painted over', () => {
    seedView({ s1: 'lattice', s2: 'ariel' }, ['artifacts', 'ariel', 'lattice']);
    deps.isHealthy.mockImplementation((/** @type {string} */ id) => id !== 'lattice');

    hide('ariel');

    expect(deps.activate).not.toHaveBeenCalled();
    expect(deps.resetActive).toHaveBeenCalledTimes(1);
    expect(deps.renderEmpty).not.toHaveBeenCalled();
    expect(panelsByTile()).toEqual([['lattice'], ['terminal']]);
  });

  test('a hide of a panel not on view closes its tile and changes nothing else', () => {
    seedView({ s1: 'ariel', s2: 'lattice' }, ['artifacts', 'ariel', 'lattice']);

    hide('ariel');

    expect(calls).toEqual(['hide:ariel']);
    expect(panelsByTile()).toEqual([['lattice'], ['terminal']]);
    expect(deps.resetActive).not.toHaveBeenCalled();
    expect(deps.clearActive).not.toHaveBeenCalled();
    expect(deps.renderEmpty).not.toHaveBeenCalled();
  });

  test('with no service tile left, the hide opens the next healthy member', () => {
    seedView({ s1: 'ariel' }, ['artifacts', 'ariel', 'channel-finder']);
    deps.isHealthy.mockImplementation((/** @type {string} */ id) => id !== 'artifacts');

    hide('ariel');

    // In catalog order, past the hidden panel and the one not answering.
    expect(calls).toEqual(['hide:ariel', 'activate:channel-finder']);
    expect(deps.renderEmpty).not.toHaveBeenCalled();
  });

  test('with no service tile and no healthy member left, the hide paints the empty pane', () => {
    seedView({ s1: 'ariel' }, ['ariel']);

    hide('ariel');

    expect(deps.activate).not.toHaveBeenCalled();
    expect(deps.resetActive).toHaveBeenCalledTimes(1);
    expect(deps.renderEmpty).toHaveBeenCalledWith('No panels visible');
    expect(panelsByTile()).toEqual([['terminal']]);
  });
});

describe('panel_report_request — a forced report', () => {
  // What the report carries, its resends and its ack are dock-sync's reporter
  // contract (dock-sync.test.mjs); this module only hands the request over.
  test("each request asks dock-sync's reporter for a report now", () => {
    deliver({ type: 'panel_report_request' });
    deliver({ type: 'panel_report_request' });
    expect(spies.reportLayoutNow).toHaveBeenCalledTimes(2);
    expect(spies.postPanelOpResult).not.toHaveBeenCalled();
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

describe('agent activity reaches the panel it names', () => {
  /**
   * The route's broadcast frame (routes/agent_activity.py record_activity):
   * target is model_dump(exclude_none=True), ts is time.time() — float seconds.
   * @param {Record<string, any>} target
   */
  function activityFrame(target) {
    return { type: 'agent_activity', tool: 'entry_create', target, ts: 1730668800.25 };
  }

  const RELAYED = {
    type: 'osprey-agent-activity',
    tool: 'entry_create',
    detail: 'e-42',
    ts: 1730668800.25,
  };

  /** A mounted panel iframe, stamped as panel-iframe-factory.js stamps it. @param {string} id */
  function mount(id) {
    const iframe = document.createElement('iframe');
    iframe.className = 'panel-iframe';
    iframe.dataset.panelId = id;
    document.body.appendChild(iframe);
    const win = /** @type {Window} */ (iframe.contentWindow);
    return vi.spyOn(win, 'postMessage').mockImplementation(() => {});
  }

  /** A stand-in pop-out window. */
  function popout() {
    const postMessage = vi.fn();
    return { win: /** @type {Window} */ (/** @type {unknown} */ ({ postMessage })), postMessage };
  }

  test("a panel-targeted frame reaches that panel's iframe only, and the badge is still applied", () => {
    const ariel = mount('ariel');
    const other = mount('artifacts');
    deliver(activityFrame({ kind: 'panel', panel: 'ariel', detail: 'e-42' }));
    expect(spies.badgePanelActivity).toHaveBeenCalledExactlyOnceWith('ariel', 1730668800.25);
    expect(ariel).toHaveBeenCalledExactlyOnceWith(RELAYED, window.location.origin);
    expect(other).not.toHaveBeenCalled();
    expect(deps.reportActivity).not.toHaveBeenCalled();
  });

  test("a popped-out panel's frame reaches its window", () => {
    const { win, postMessage } = popout();
    spies.popoutWindow.mockImplementation((/** @type {string} */ id) =>
      id === 'ariel' ? win : null
    );
    const other = mount('artifacts');
    deliver(activityFrame({ kind: 'panel', panel: 'ariel', detail: 'e-42' }));
    expect(spies.popoutWindow).toHaveBeenCalledWith('ariel');
    expect(postMessage).toHaveBeenCalledExactlyOnceWith(RELAYED, window.location.origin);
    expect(other).not.toHaveBeenCalled();
    expect(spies.badgePanelActivity).toHaveBeenCalledExactlyOnceWith('ariel', 1730668800.25);
  });

  test('a panel with both a tile iframe and a pop-out gets the notice in each', () => {
    const tileFrame = mount('ariel');
    const { win, postMessage } = popout();
    spies.popoutWindow.mockReturnValue(win);
    deliver(activityFrame({ kind: 'panel', panel: 'ariel', detail: 'e-42' }));
    expect(tileFrame).toHaveBeenCalledExactlyOnceWith(RELAYED, window.location.origin);
    expect(postMessage).toHaveBeenCalledExactlyOnceWith(RELAYED, window.location.origin);
  });

  test('a frame without a detail relays none', () => {
    const ariel = mount('ariel');
    deliver(activityFrame({ kind: 'panel', panel: 'ariel' }));
    expect(ariel).toHaveBeenCalledOnce();
    expect(ariel.mock.calls[0][0]).toMatchObject({
      type: 'osprey-agent-activity',
      tool: 'entry_create',
    });
    expect(ariel.mock.calls[0][0].detail).toBeUndefined();
  });

  test('a frame for an unmounted panel is dropped: nothing is posted, the badge still applies', () => {
    const other = mount('artifacts');
    deliver(activityFrame({ kind: 'panel', panel: 'ariel', detail: 'e-42' }));
    expect(other).not.toHaveBeenCalled();
    expect(spies.badgePanelActivity).toHaveBeenCalledExactlyOnceWith('ariel', 1730668800.25);
  });

  test('a panel with no rail entry falls through to the strip and is still relayed when mounted', () => {
    spies.badgePanelActivity.mockReturnValue(false);
    const ariel = mount('ariel');
    const frame = activityFrame({ kind: 'panel', panel: 'ariel', detail: 'e-42' });
    deliver(frame);
    expect(deps.reportActivity).toHaveBeenCalledExactlyOnceWith(frame);
    expect(ariel).toHaveBeenCalledExactlyOnceWith(RELAYED, window.location.origin);
  });

  test('a non-panel target posts nothing', () => {
    const ariel = mount('ariel');
    const { win, postMessage } = popout();
    spies.popoutWindow.mockReturnValue(win);
    const frame = activityFrame({ kind: 'channel', detail: 'SR:BPM:1' });
    deliver(frame);
    expect(ariel).not.toHaveBeenCalled();
    expect(postMessage).not.toHaveBeenCalled();
    expect(spies.badgePanelActivity).not.toHaveBeenCalled();
    expect(deps.reportActivity).toHaveBeenCalledExactlyOnceWith(frame);
  });
});
