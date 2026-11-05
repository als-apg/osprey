/**
 * Contract tests for the agent op runner (panel-placement.js runPanelOp) on
 * the real layout holder:
 *   npx vitest run tests/interfaces/web_terminal/panel-placement.test.mjs
 *
 * A tile holds ONE panel (dock-policy.js), and every op is checked against
 * that rule and run through dock-open.js or the engine:
 *
 *   - an `open` takes the service tile the operator used last, REPLACING the
 *     panel it showed (the replaced panel leaves the dock); a panel already
 *     docked is focused where it is; a panel in its own window is brought
 *     forward;
 *   - a set opens side by side, or as a layout of one panel per tile; a set
 *     or an arrangement opened as tabs, an arrangement giving a tile two
 *     members, and a drop into a tile's centre or header are refused
 *     `one_panel_per_tile` and change nothing;
 *   - drops on a tile edge, on the border between tiles and on a dock edge
 *     split through the engine;
 *   - `close` takes a panel and its tile out (the space goes to a neighbour);
 *     the retired `close_tab` and `close_tile` ops are unknown;
 *   - a stale base, every move or close of the terminal and every op naming a
 *     panel in its own window are refused, and rail membership is left to the
 *     server;
 *   - the polite agent switch and the legacy panel_arrange consumer are gone.
 *
 * The holder is the real one (dock-apply.js) without a dockview api: it keeps
 * state and applies nothing, so every assertion reads the layout it holds.
 * dock-workspace.js is stubbed so getDockApi yields a minimal fake api (the
 * dock exists) and the adapter's overlay can wire.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const JS = '../../../src/osprey/interfaces/web_terminal/static/js';

const { state } = vi.hoisted(() => ({
  state: { api: /** @type {any} */ (null), redock: /** @type {any} */ (null) },
}));

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/dock-workspace.js', () => ({
  getDockApi: () => state.api,
  setServiceRedock: (/** @type {any} */ fn) => {
    state.redock = fn;
  },
  setLayoutRestoredHook: () => {},
  defaultServiceWidth: () => 600,
  onDragGesture: () => [],
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/panel-commands.js', () => ({
  setPanelFocus: vi.fn(),
  setPanelVisibility: vi.fn(),
}));
// Which panels live in windows of their own; popout.js's own behaviour is its
// test's contract.
const { popped } = vi.hoisted(() => ({ popped: new Set() }));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/popout.js', () => ({
  isPoppedOut: (/** @type {string} */ id) => popped.has(id),
}));

const DOCK_PX = { width: 1000, height: 600 };

/** A fake dockview api: enough for the adapter to wire its overlay. */
function fakeApi() {
  const disposable = () => ({ dispose() {} });
  return {
    onDidLayoutChange: disposable,
    onDidActivePanelChange: disposable,
    getPanel: () => null,
    panels: [],
    groups: [],
  };
}

/** @param {string} id @param {string[]} tabs @param {string} [showing] @param {number} [size] */
function tile(id, tabs, showing = tabs[0], size) {
  /** @type {any} */
  const t = { tile: id, tabs, showing };
  if (size !== undefined) t.size = size;
  return t;
}

/** @param {any[]} items */
function row(...items) {
  return { split: 'row', items };
}

/** Every tile of a layout, in tree order. @param {any} node @returns {any[]} */
function tiles(node) {
  if (node === null) return [];
  return 'tabs' in node ? [node] : node.items.flatMap(tiles);
}

/** The tile holding `key`. @param {any} layout @param {string} key */
function tileWith(layout, key) {
  return tiles(layout).find((t) => t.tabs.includes(key)) ?? null;
}

/** Each tile's panels, in tree order. @param {any} layout */
function panelsByTile(layout) {
  return tiles(layout).map((t) => t.tabs);
}

/** @type {any} */
let placement;
/** @type {any} */
let dockApply;
/** @type {any} */
let dockOpen;
/** @type {any} */
let iframeAdapter;
/** @type {any} */
let deps;

/**
 * Seed the holder with `layout`, focusing `focus` (a tile id) when given.
 * @param {any} layout @param {string} [focus]
 */
function seed(layout, focus) {
  const holder = dockApply.getLayoutHolder();
  expect(holder.restore(layout)).toBe(true);
  if (focus) {
    const t = tiles(holder.layout).find((x) => x.tile === focus);
    holder.show(focus, t.showing);
  }
  return holder;
}

beforeEach(async () => {
  vi.resetModules();
  document.body.innerHTML = '<div class="main-container"></div>';
  state.api = fakeApi();
  dockApply = await import(`${JS}/dock-apply.js`);
  dockApply.initLayoutHolder({ dockPx: DOCK_PX });
  placement = await import(`${JS}/panel-placement.js`);
  dockOpen = await import(`${JS}/dock-open.js`);
  iframeAdapter = await import(`${JS}/dock-iframe.js`);
  iframeAdapter.initDockIframeAdapter({ fallbackHost: null });
  const members = new Set(['a', 'b', 'c']);
  deps = {
    isKnown: vi.fn(() => true),
    isHealthy: vi.fn(() => true),
    isMember: vi.fn((/** @type {string} */ id) => members.has(id)),
    members: () => [...members],
    label: (/** @type {string} */ id) => id.toUpperCase(),
    addMember: vi.fn(),
    dropMember: vi.fn(),
    activate: vi.fn(),
    reveal: vi.fn(),
    getActive: () => null,
    clearActive: vi.fn(),
    renderEmpty: vi.fn(),
    glow: vi.fn(),
    openTerminal: vi.fn(),
  };
  placement.initPanelPlacement(deps);
});

afterEach(() => {
  state.api = null;
  popped.clear();
  document.body.innerHTML = '';
});

describe('the polite agent switch is retired', () => {
  test('panel-placement no longer exports applyAgentSwitch', () => {
    expect('applyAgentSwitch' in placement).toBe(false);
  });

  test('hasDockedTile reads the holder', () => {
    seed(row(tile('s1', ['a'], 'a', 0.6), tile('t1', ['terminal'], 'terminal', 0.4)));
    expect(placement.hasDockedTile('a')).toBe(true);
    expect(placement.hasDockedTile('b')).toBe(false);
  });
});

describe('runPanelOp — the agent op runner, on the real holder', () => {
  /** @param {string} op @param {Record<string, any>} args @param {Record<string, any>} [over] */
  function frame(op, args, over = {}) {
    return { type: 'panel_op', request_id: 'r1', target_client: 'me', op, args, ...over };
  }

  /**
   * The classic two-service dock, focused on s1: s1 shows a, s2 shows c, the
   * terminal is t1. b is a rail member with no tile.
   */
  function seedClassic() {
    return seed(
      row(
        tile('s1', ['a'], 'a', 0.3),
        tile('s2', ['c'], 'c', 0.3),
        tile('t1', ['terminal'], 'terminal', 0.4)
      ),
      's1'
    );
  }

  describe('stale_base', () => {
    test('a base_rev other than the acked rev is refused and changes nothing', () => {
      const holder = seedClassic();
      holder.ack(5);
      const before = holder.layout;
      const out = placement.runPanelOp(frame('close', { key: 'c' }, { base_rev: 4 }));
      expect(out).toEqual({ refusal: 'stale_base' });
      expect(holder.layout).toBe(before);
    });

    test('a dirty holder refuses a base_rev even when it matches', () => {
      const holder = seedClassic();
      holder.ack(5);
      // A person's sash drag since the ack.
      holder.resized(
        row(
          tile('s1', ['a'], 'a', 0.4),
          tile('s2', ['c'], 'c', 0.2),
          tile('t1', ['terminal'], 'terminal', 0.4)
        )
      );
      expect(holder.dirty).toBe(true);
      const out = placement.runPanelOp(frame('close', { key: 'c' }, { base_rev: 5 }));
      expect(out.refusal).toBe('stale_base');
      expect(tileWith(holder.layout, 'c')).not.toBeNull();
    });

    test('the acked rev on a clean holder runs', () => {
      const holder = seedClassic();
      holder.ack(5);
      const out = placement.runPanelOp(frame('close', { key: 'c' }, { base_rev: 5 }));
      expect(out.refusal).toBeUndefined();
      expect(tileWith(holder.layout, 'c')).toBeNull();
    });

    test('an op without base_rev runs on a dirty holder', () => {
      const holder = seedClassic();
      expect(holder.dirty).toBe(true);
      const out = placement.runPanelOp(frame('close', { key: 'c' }));
      expect(out.refusal).toBeUndefined();
      expect(tileWith(holder.layout, 'c')).toBeNull();
    });
  });

  describe('the terminal invariant, enforced again', () => {
    test('a close or an open of the terminal is refused', () => {
      const holder = seedClassic();
      const before = holder.layout;
      for (const out of [
        placement.runPanelOp(frame('close', { key: 'terminal' })),
        placement.runPanelOp(frame('open', { member: 'terminal' })),
      ]) {
        expect(out.refusal).toBe('terminal_not_agent_movable');
      }
      expect(holder.layout).toBe(before);
    });

    test('a drop of the terminal, by key or by its tile, is refused', () => {
      const holder = seedClassic();
      const before = holder.layout;
      for (const source of [{ tab: 'terminal' }, { new: { panel: 'terminal' } }, { tile: 't1' }]) {
        expect(
          placement.runPanelOp(frame('drop', { source, target: { tile: 's1', edge: 'left' } }))
            .refusal
        ).toBe('terminal_not_agent_movable');
      }
      expect(holder.layout).toBe(before);
    });

    test('the terminal tile may be dropped BESIDE', () => {
      const holder = seedClassic();
      const out = placement.runPanelOp(
        frame('drop', { source: { tab: 'c' }, target: { tile: 't1', edge: 'bottom' } })
      );
      expect(out.refusal).toBeUndefined();
      expect(tileWith(holder.layout, 'c').tile).not.toBe('t1');
      expect(tileWith(holder.layout, 'terminal').tabs).toEqual(['terminal']);
    });

    test('a set naming the terminal is terminal_in_layout', () => {
      seedClassic();
      expect(
        placement.runPanelOp(frame('open_set', { members: ['a', 'terminal'], as: 'side' })).refusal
      ).toBe('terminal_in_layout');
      expect(
        placement.runPanelOp(
          frame('open_arrangement', {
            name: 'x',
            members: ['terminal'],
            layout: { tabs: [0] },
          })
        ).refusal
      ).toBe('terminal_in_layout');
    });
  });

  describe('open: the service tile used last, replacing the panel it showed', () => {
    test('open replaces the panel of the focused service tile; nothing else moves', () => {
      const holder = seedClassic();
      const out = placement.runPanelOp(frame('open', { member: 'b' }));
      expect(out.refusal).toBeUndefined();
      // b took s1 in a's place, at s1's size; a left the dock.
      expect(holder.layout).toEqual(
        row(
          tile('s1', ['b'], 'b', 0.3),
          tile('s2', ['c'], 'c', 0.3),
          tile('t1', ['terminal'], 'terminal', 0.4)
        )
      );
      expect(tileWith(holder.layout, 'a')).toBeNull();
      expect(out.result.focus).toBe('s1');
      // The replaced panel keeps its rail row: the runner never edits the rail.
      expect(deps.dropMember).not.toHaveBeenCalled();
    });

    test('with the agent pane focused, the service tile the operator used before it is replaced', () => {
      const holder = seedClassic();
      holder.show('s2', 'c');
      holder.show('t1', 'terminal'); // the operator typed in the agent pane since
      const out = placement.runPanelOp(frame('open', { member: 'b' }));
      expect(out.refusal).toBeUndefined();
      expect(panelsByTile(holder.layout)).toEqual([['a'], ['b'], ['terminal']]);
      expect(tileWith(holder.layout, 'b').tile).toBe('s2');
      expect(tileWith(holder.layout, 'c')).toBeNull();
      expect(holder.focusedTile).toBe('s2');
    });

    test('open of a panel already docked focuses its tile and changes nothing', () => {
      const holder = seedClassic();
      const before = holder.layout;
      const out = placement.runPanelOp(frame('open', { member: 'c' }));
      expect(out.refusal).toBeUndefined();
      expect(out.result).toMatchObject({ changed: false, focus: 's2' });
      expect(holder.layout).toBe(before);
      expect(holder.focusedTile).toBe('s2');
    });

    test('open of an unhealthy panel is panel_unhealthy and changes nothing', () => {
      const holder = seedClassic();
      deps.isHealthy.mockImplementation((/** @type {string} */ id) => id !== 'd');
      const before = holder.layout;
      expect(placement.runPanelOp(frame('open', { member: 'd' })).refusal).toBe('panel_unhealthy');
      expect(holder.layout).toBe(before);
    });
  });

  describe('sets and arrangements: side by side, or one panel per tile', () => {
    test('open_set as side opens each member in a tile of its own; the panels on screen stay', () => {
      const holder = seedClassic();
      const out = placement.runPanelOp(frame('open_set', { members: ['d', 'e'], as: 'side' }));
      expect(out.refusal).toBeUndefined();
      const d = tileWith(holder.layout, 'd');
      const e = tileWith(holder.layout, 'e');
      expect(d.tabs).toEqual(['d']);
      expect(e.tabs).toEqual(['e']);
      expect(d.tile).not.toBe(e.tile);
      // Nothing was replaced: every tile still holds one panel.
      expect(tileWith(holder.layout, 'a').tabs).toEqual(['a']);
      expect(tileWith(holder.layout, 'c').tabs).toEqual(['c']);
      expect(tileWith(holder.layout, 'terminal').tabs).toEqual(['terminal']);
    });

    test('a set or an arrangement opened as tabs is one_panel_per_tile and changes nothing', () => {
      const holder = seedClassic();
      const before = holder.layout;
      expect(placement.runPanelOp(frame('open_set', { members: ['d', 'e'], as: 'tabs' }))).toEqual({
        refusal: 'one_panel_per_tile',
      });
      expect(
        placement.runPanelOp(
          frame('open_arrangement', {
            name: 'Machine setup',
            members: ['d', 'e'],
            as: 'tabs',
            layout: { split: 'row', items: [{ tabs: [0] }, { tabs: [1] }] },
          })
        )
      ).toEqual({ refusal: 'one_panel_per_tile' });
      expect(holder.layout).toBe(before);
    });

    test('a layout giving a tile two members is one_panel_per_tile and changes nothing', () => {
      const holder = seedClassic();
      const before = holder.layout;
      const records = holder.closed.length;
      for (const layout of [
        { tabs: [0, 1] },
        { split: 'row', items: [{ tabs: [0, 1] }, { tabs: [2] }] },
      ]) {
        const members = ['d', 'e', 'f'];
        expect(
          placement.runPanelOp(
            frame('open_arrangement', { name: 'Machine setup', members, layout })
          ).refusal
        ).toBe('one_panel_per_tile');
        expect(
          placement.runPanelOp(frame('open_set', { members, as: 'layout', layout })).refusal
        ).toBe('one_panel_per_tile');
      }
      expect(holder.layout).toBe(before);
      expect(holder.closed).toHaveLength(records);
    });

    test('open_set as layout maps member indexes to panel ids', () => {
      const holder = seedClassic();
      const out = placement.runPanelOp(
        frame('open_set', {
          members: ['d', 'e'],
          as: 'layout',
          layout: { split: 'row', items: [{ tabs: [0] }, { tabs: [1] }] },
        })
      );
      expect(out.refusal).toBeUndefined();
      expect(tileWith(holder.layout, 'd')).not.toBeNull();
      expect(tileWith(holder.layout, 'e')).not.toBeNull();
      expect(tileWith(holder.layout, 'd').tile).not.toBe(tileWith(holder.layout, 'e').tile);
    });

    test('a layout index naming no member is invalid_layout', () => {
      seedClassic();
      expect(
        placement.runPanelOp(
          frame('open_set', { members: ['d'], as: 'layout', layout: { tabs: [3] } })
        ).refusal
      ).toBe('invalid_layout');
    });

    test('open_arrangement without as applies its layout: unnamed services leave, the terminal stays', () => {
      const holder = seedClassic();
      const out = placement.runPanelOp(
        frame('open_arrangement', {
          name: 'Machine setup',
          members: ['c', 'd'],
          layout: { split: 'column', items: [{ tabs: [0] }, { tabs: [1] }] },
        })
      );
      expect(out.refusal).toBeUndefined();
      expect(tileWith(holder.layout, 'a')).toBeNull();
      expect(tileWith(holder.layout, 'c')).not.toBeNull();
      expect(tileWith(holder.layout, 'd')).not.toBeNull();
      expect(tileWith(holder.layout, 'terminal').tabs).toEqual(['terminal']);
      // An arrangement is undoable: the holder keeps its layout record.
      expect(holder.closed.at(-1)?.kind).toBe('layout');
    });
  });

  describe('drops: edges and borders split, a tile takes no second panel', () => {
    test('a new panel dropped on a tile edge opens as its own tile there', () => {
      const holder = seedClassic();
      const out = placement.runPanelOp(
        frame('drop', { source: { new: { panel: 'd' } }, target: { tile: 's2', edge: 'bottom' } })
      );
      expect(out.refusal).toBeUndefined();
      const d = tileWith(holder.layout, 'd');
      expect(d.tabs).toEqual(['d']);
      expect(out.result.focus).toBe(d.tile);
    });

    test('a {new} source naming an open panel moves it, never duplicates it', () => {
      const holder = seedClassic();
      const out = placement.runPanelOp(
        frame('drop', { source: { new: { panel: 'c' } }, target: { tile: 's1', edge: 'bottom' } })
      );
      expect(out.refusal).toBeUndefined();
      const keys = tiles(holder.layout).flatMap((t) => t.tabs);
      expect(keys.filter((k) => k === 'c')).toHaveLength(1);
      // c left s2, which closed with it, for a tile of its own below s1.
      expect(tiles(holder.layout).map((t) => t.tile)).not.toContain('s2');
      expect(out.result.focus).toBe(tileWith(holder.layout, 'c').tile);
    });

    test('a drop on the border between two tiles opens the panel in a tile between them', () => {
      const holder = seedClassic();
      const out = placement.runPanelOp(
        frame('drop', {
          source: { new: { panel: 'b' } },
          target: { border: { tile: 's2', side: 'left' } },
        })
      );
      expect(out.refusal).toBeUndefined();
      expect(panelsByTile(holder.layout)).toEqual([['a'], ['b'], ['c'], ['terminal']]);
    });

    test('a drop on a dock edge moves the panel to a tile along that edge', () => {
      const holder = seedClassic();
      const out = placement.runPanelOp(
        frame('drop', { source: { tab: 'c' }, target: { outer: 'left' } })
      );
      expect(out.refusal).toBeUndefined();
      expect(panelsByTile(holder.layout)).toEqual([['c'], ['a'], ['terminal']]);
    });

    test("a drop into a tile's centre or header is one_panel_per_tile and changes nothing", () => {
      const holder = seedClassic();
      const before = holder.layout;
      for (const into of ['s1', 't1']) {
        for (const target of [
          { tile: into, zone: 'centre' },
          { tile: into, zone: 'tab', index: 0 },
        ]) {
          for (const source of [{ tab: 'c' }, { new: { panel: 'b' } }, { tile: 's2' }]) {
            expect(placement.runPanelOp(frame('drop', { source, target })).refusal).toBe(
              'one_panel_per_tile'
            );
          }
        }
      }
      expect(holder.layout).toBe(before);
    });

    test('a drop naming a tile this dock does not hold is unknown_tile', () => {
      const holder = seedClassic();
      const before = holder.layout;
      for (const target of [
        { tile: 'nope', edge: 'left' },
        { border: { tile: 'nope', side: 'left' } },
      ]) {
        expect(placement.runPanelOp(frame('drop', { source: { tab: 'c' }, target })).refusal).toBe(
          'unknown_tile'
        );
      }
      expect(holder.layout).toBe(before);
    });

    test('a drop of a panel this dock does not hold is unknown_key', () => {
      seedClassic();
      expect(
        placement.runPanelOp(
          frame('drop', { source: { tab: 'zz' }, target: { tile: 's1', edge: 'left' } })
        ).refusal
      ).toBe('unknown_key');
    });

    test('a {new} drop of an unhealthy panel is panel_unhealthy', () => {
      seedClassic();
      deps.isHealthy.mockImplementation((/** @type {string} */ id) => id !== 'd');
      expect(
        placement.runPanelOp(
          frame('drop', { source: { new: { panel: 'd' } }, target: { tile: 's1', edge: 'left' } })
        ).refusal
      ).toBe('panel_unhealthy');
    });
  });

  describe('close: the panel leaves the dock, and its tile with it', () => {
    test("close takes the panel's tile out and gives its space to a neighbour", () => {
      const holder = seedClassic();
      const out = placement.runPanelOp(frame('close', { key: 'c' }));
      expect(out.refusal).toBeUndefined();
      expect(holder.layout).toEqual(
        row(tile('s1', ['a'], 'a', 0.6), tile('t1', ['terminal'], 'terminal', 0.4))
      );
      // A close is undoable: the holder keeps the closed tile's record.
      expect(holder.closed.at(-1)).toMatchObject({ kind: 'tile', tile: 's2', tabs: ['c'] });
      // ...and it is no hide: the rail is the server's.
      expect(deps.dropMember).not.toHaveBeenCalled();
    });

    test('a close of a panel this dock does not hold is unknown_key', () => {
      const holder = seedClassic();
      const before = holder.layout;
      // b is a rail member with no tile; zz is no panel at all.
      for (const key of ['b', 'zz']) {
        expect(placement.runPanelOp(frame('close', { key })).refusal).toBe('unknown_key');
      }
      expect(holder.layout).toBe(before);
    });

    test('close_tab and close_tile are ops no longer: unknown_op, and nothing changes', () => {
      const holder = seedClassic();
      const before = holder.layout;
      expect(placement.runPanelOp(frame('close_tab', { key: 'c' })).refusal).toBe('unknown_op');
      expect(placement.runPanelOp(frame('close_tile', { tile: 's2' })).refusal).toBe('unknown_op');
      expect(holder.layout).toBe(before);
    });
  });

  test('an unknown op is refused by name', () => {
    seedClassic();
    expect(placement.runPanelOp(frame('zoom', {})).refusal).toBe('unknown_op');
  });

  test('a drag in progress holds the apply, not the op: the holder layout moves at once', () => {
    const holder = seedClassic();
    holder.setDragLive(true);
    placement.runPanelOp(frame('close', { key: 'c' }));
    expect(tileWith(holder.layout, 'c')).toBeNull();
    holder.setDragLive(false);
  });

  test('the legacy panel_arrange consumer is gone', () => {
    expect('applyArrange' in placement).toBe(false);
  });

  describe('a panel in its own window', () => {
    /** The classic dock with c popped out: its tile left the dock with it. */
    function seedPoppedOut() {
      const holder = seedClassic();
      holder.run((/** @type {any} */ l) => dockOpen.closeKey(l, 'c'));
      popped.add('c');
      return holder;
    }

    test('an open brings the window forward and leaves the layout alone', () => {
      const holder = seedPoppedOut();
      const before = holder.layout;
      const out = placement.runPanelOp(frame('open', { member: 'c' }));
      expect(out.refusal).toBeUndefined();
      expect(out.popout).toBe('c');
      expect(holder.layout).toBe(before);
    });

    test('a close, a drop of it and a set naming it are refused popped_out', () => {
      const holder = seedPoppedOut();
      const before = holder.layout;
      expect(placement.runPanelOp(frame('close', { key: 'c' })).refusal).toBe('popped_out');
      expect(
        placement.runPanelOp(frame('drop', { source: { tab: 'c' }, target: { outer: 'left' } }))
          .refusal
      ).toBe('popped_out');
      expect(
        placement.runPanelOp(
          frame('drop', { source: { new: { panel: 'c' } }, target: { outer: 'left' } })
        ).refusal
      ).toBe('popped_out');
      expect(
        placement.runPanelOp(frame('open_set', { members: ['c', 'b'], as: 'side' })).refusal
      ).toBe('popped_out');
      expect(holder.layout).toBe(before);
    });
  });
});
