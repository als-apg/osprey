/**
 * Contract tests for the person-side open paths (panel-placement.js, with the
 * dock-sync verbs and the dock-iframe adapter they reach):
 *   npx vitest run tests/interfaces/web_terminal/panel-placement.test.mjs
 *
 * Every open and placement goes through dock-open.js and the layout holder:
 *
 *   - "open beside" (the rail's corner button, the + menu) halves the active
 *     service tile; with no service tile it opens left of the terminal at the
 *     classic 0.6/0.4 split; the terminal is never covered; a panel that cannot
 *     fill a tile gets none;
 *   - an activation (rail click, quick-open, link, agent open) opens a new tab
 *     of the last service tile used, shows an open tab where it is, and never
 *     covers a showing terminal;
 *   - hiding closes the tab through the engine; pruning drops tabs nothing can
 *     fill, leaving no undo record;
 *   - the polite agent switch is gone;
 *   - the agent op runner (runPanelOp) refuses a stale base and every terminal
 *     move, runs opens through dock-open.js and drops/closes through the
 *     engine, and leaves rail membership to the server.
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

/** @type {any} */
let placement;
/** @type {any} */
let dockApply;
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

  /** The classic two-service dock: s1 [a, b] showing a, s2 [c], the terminal t1. */
  function seedClassic() {
    return seed(
      row(
        tile('s1', ['a', 'b'], 'a', 0.3),
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
      const out = placement.runPanelOp(frame('close_tile', { tile: 's2' }, { base_rev: 4 }));
      expect(out).toEqual({ refusal: 'stale_base' });
      expect(holder.layout).toBe(before);
    });

    test('a dirty holder refuses a base_rev even when it matches', () => {
      const holder = seedClassic();
      holder.ack(5);
      holder.show('s1', 'b'); // a person's tab click since the ack
      expect(holder.dirty).toBe(true);
      const out = placement.runPanelOp(frame('close_tile', { tile: 's2' }, { base_rev: 5 }));
      expect(out.refusal).toBe('stale_base');
      expect(tileWith(holder.layout, 'c')).not.toBeNull();
    });

    test('the acked rev on a clean holder runs', () => {
      const holder = seedClassic();
      holder.ack(5);
      const out = placement.runPanelOp(frame('close_tile', { tile: 's2' }, { base_rev: 5 }));
      expect(out.refusal).toBeUndefined();
      expect(tileWith(holder.layout, 'c')).toBeNull();
    });

    test('an op without base_rev runs on a dirty holder', () => {
      const holder = seedClassic();
      expect(holder.dirty).toBe(true);
      const out = placement.runPanelOp(frame('close_tab', { key: 'b' }));
      expect(out.refusal).toBeUndefined();
      expect(tileWith(holder.layout, 'b')).toBeNull();
    });
  });

  describe('the terminal invariant, enforced again', () => {
    test('close_tab of the terminal is refused', () => {
      const holder = seedClassic();
      const before = holder.layout;
      expect(placement.runPanelOp(frame('close_tab', { key: 'terminal' })).refusal).toBe(
        'terminal_not_agent_movable'
      );
      expect(holder.layout).toBe(before);
    });

    test('close_tile of the tile holding the terminal is refused', () => {
      seedClassic();
      expect(placement.runPanelOp(frame('close_tile', { tile: 't1' })).refusal).toBe(
        'terminal_not_agent_movable'
      );
    });

    test('a tile holding the terminal behind a service tab is still the terminal tile', () => {
      const holder = seed(
        row(tile('s1', ['a'], 'a', 0.6), tile('t1', ['terminal', 'b'], 'b', 0.4)),
        's1'
      );
      const before = holder.layout;
      expect(placement.runPanelOp(frame('close_tile', { tile: 't1' })).refusal).toBe(
        'terminal_not_agent_movable'
      );
      expect(
        placement.runPanelOp(frame('drop', { source: { tile: 't1' }, target: { outer: 'left' } }))
          .refusal
      ).toBe('terminal_not_agent_movable');
      expect(holder.layout).toBe(before);
    });

    test('a drop of the terminal tab is refused', () => {
      seedClassic();
      expect(
        placement.runPanelOp(
          frame('drop', { source: { tab: 'terminal' }, target: { tile: 's1', zone: 'centre' } })
        ).refusal
      ).toBe('terminal_not_agent_movable');
    });

    test('the terminal tile takes no tab: a centre or tab-bar target on it is refused', () => {
      const holder = seedClassic();
      const before = holder.layout;
      for (const target of [
        { tile: 't1', zone: 'centre' },
        { tile: 't1', zone: 'tab', index: 0 },
      ]) {
        expect(placement.runPanelOp(frame('drop', { source: { tab: 'c' }, target })).refusal).toBe(
          'terminal_not_agent_movable'
        );
        expect(
          placement.runPanelOp(frame('drop', { source: { new: { panel: 'd' } }, target })).refusal
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
        placement.runPanelOp(frame('open_set', { members: ['a', 'terminal'], as: 'tabs' })).refusal
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

  describe('opens go through dock-open.js', () => {
    test('open places a new tab in the last service tile used (the open rule)', () => {
      const holder = seedClassic();
      holder.show('t1', 'terminal');
      const out = placement.runPanelOp(frame('open', { member: 'd' }));
      expect(out.refusal).toBeUndefined();
      expect(tileWith(holder.layout, 'd').tile).toBe('s1');
      expect(tileWith(holder.layout, 'd').showing).toBe('d');
      expect(out.result.focus).toBe('s1');
    });

    test('open of an unhealthy panel is panel_unhealthy and changes nothing', () => {
      const holder = seedClassic();
      deps.isHealthy.mockImplementation((/** @type {string} */ id) => id !== 'd');
      const before = holder.layout;
      expect(placement.runPanelOp(frame('open', { member: 'd' })).refusal).toBe('panel_unhealthy');
      expect(holder.layout).toBe(before);
    });

    test('open_set as tabs opens every member', () => {
      const holder = seedClassic();
      const out = placement.runPanelOp(frame('open_set', { members: ['d', 'e'], as: 'tabs' }));
      expect(out.refusal).toBeUndefined();
      expect(tileWith(holder.layout, 'd')).not.toBeNull();
      expect(tileWith(holder.layout, 'e')).not.toBeNull();
      expect(tileWith(holder.layout, 'd').tile).toBe(tileWith(holder.layout, 'e').tile);
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
      expect(tileWith(holder.layout, 'b')).toBeNull();
      expect(tileWith(holder.layout, 'c')).not.toBeNull();
      expect(tileWith(holder.layout, 'd')).not.toBeNull();
      expect(tileWith(holder.layout, 'terminal').tabs).toEqual(['terminal']);
      // An arrangement is undoable: the holder keeps its layout record.
      expect(holder.closed.at(-1)?.kind).toBe('layout');
    });

    test('open_arrangement with as tabs opens the members as tabs instead', () => {
      const holder = seedClassic();
      const out = placement.runPanelOp(
        frame('open_arrangement', {
          name: 'Machine setup',
          members: ['d', 'e'],
          as: 'tabs',
          layout: { split: 'row', items: [{ tabs: [0] }, { tabs: [1] }] },
        })
      );
      expect(out.refusal).toBeUndefined();
      expect(tileWith(holder.layout, 'd').tile).toBe(tileWith(holder.layout, 'e').tile);
      expect(tileWith(holder.layout, 'a')).not.toBeNull();
    });
  });

  describe('drops and closes go through the engine', () => {
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
      placement.runPanelOp(
        frame('drop', { source: { new: { panel: 'c' } }, target: { tile: 's1', zone: 'centre' } })
      );
      expect(tileWith(holder.layout, 'c').tile).toBe('s1');
      expect(
        tiles(holder.layout)
          .flatMap((t) => t.tabs)
          .filter((k) => k === 'c')
      ).toHaveLength(1);
    });

    test('a tab dropped at a tab index lands there', () => {
      const holder = seedClassic();
      placement.runPanelOp(
        frame('drop', { source: { tab: 'c' }, target: { tile: 's1', zone: 'tab', index: 0 } })
      );
      expect(tileWith(holder.layout, 'c').tabs).toEqual(['c', 'a', 'b']);
    });

    test('a drop naming a tile this dock does not hold is unknown_tile', () => {
      const holder = seedClassic();
      const before = holder.layout;
      expect(
        placement.runPanelOp(
          frame('drop', { source: { tab: 'c' }, target: { tile: 'nope', zone: 'centre' } })
        ).refusal
      ).toBe('unknown_tile');
      expect(holder.layout).toBe(before);
    });

    test('a drop of a tab this dock does not hold is unknown_key', () => {
      seedClassic();
      expect(
        placement.runPanelOp(
          frame('drop', { source: { tab: 'zz' }, target: { tile: 's1', zone: 'centre' } })
        ).refusal
      ).toBe('unknown_key');
    });

    test('a {new} drop of an unhealthy panel is panel_unhealthy', () => {
      seedClassic();
      deps.isHealthy.mockImplementation((/** @type {string} */ id) => id !== 'd');
      expect(
        placement.runPanelOp(
          frame('drop', { source: { new: { panel: 'd' } }, target: { tile: 's1', zone: 'centre' } })
        ).refusal
      ).toBe('panel_unhealthy');
    });

    test('close_tab closes one tab; the last tab closes its tile', () => {
      const holder = seedClassic();
      placement.runPanelOp(frame('close_tab', { key: 'a' }));
      expect(tileWith(holder.layout, 'b').tabs).toEqual(['b']);
      placement.runPanelOp(frame('close_tab', { key: 'c' }));
      expect(tiles(holder.layout).map((t) => t.tile)).not.toContain('s2');
      // Closes are undoable: the holder keeps their records.
      expect(holder.closed.length).toBeGreaterThanOrEqual(2);
    });

    test('close_tab of a tab this dock does not hold is unknown_key', () => {
      seedClassic();
      expect(placement.runPanelOp(frame('close_tab', { key: 'zz' })).refusal).toBe('unknown_key');
    });

    test('close_tile closes a service tile and its tabs', () => {
      const holder = seedClassic();
      placement.runPanelOp(frame('close_tile', { tile: 's1' }));
      expect(tileWith(holder.layout, 'a')).toBeNull();
      expect(tileWith(holder.layout, 'b')).toBeNull();
      expect(tileWith(holder.layout, 'c')).not.toBeNull();
    });

    test('close_tile of an unknown tile is unknown_tile', () => {
      seedClassic();
      expect(placement.runPanelOp(frame('close_tile', { tile: 'nope' })).refusal).toBe(
        'unknown_tile'
      );
    });
  });

  test('an unknown op is refused by name', () => {
    seedClassic();
    expect(placement.runPanelOp(frame('zoom', {})).refusal).toBe('unknown_op');
  });

  test('a drag in progress holds the apply, not the op: the holder layout moves at once', () => {
    const holder = seedClassic();
    holder.setDragLive(true);
    placement.runPanelOp(frame('close_tab', { key: 'c' }));
    expect(tileWith(holder.layout, 'c')).toBeNull();
    holder.setDragLive(false);
  });

  test('the legacy panel_arrange consumer is gone', () => {
    expect('applyArrange' in placement).toBe(false);
  });

  describe('a panel in its own window', () => {
    test('an open brings the window forward and leaves the layout alone', () => {
      const holder = seedClassic();
      holder.run((/** @type {any} */ l) => ({ layout: removeKey(l, 'b'), changed: true }));
      popped.add('b');
      const before = holder.layout;
      const out = placement.runPanelOp(frame('open', { member: 'b' }));
      expect(out.refusal).toBeUndefined();
      expect(out.popout).toBe('b');
      expect(holder.layout).toBe(before);
    });

    test('close_tab, a drop of it and a set naming it are refused popped_out', () => {
      const holder = seedClassic();
      holder.run((/** @type {any} */ l) => ({ layout: removeKey(l, 'b'), changed: true }));
      popped.add('b');
      const before = holder.layout;
      expect(placement.runPanelOp(frame('close_tab', { key: 'b' })).refusal).toBe('popped_out');
      expect(
        placement.runPanelOp(frame('drop', { source: { tab: 'b' }, target: { outer: 'left' } }))
          .refusal
      ).toBe('popped_out');
      expect(
        placement.runPanelOp(
          frame('drop', { source: { new: { panel: 'b' } }, target: { outer: 'left' } })
        ).refusal
      ).toBe('popped_out');
      expect(
        placement.runPanelOp(frame('open_set', { members: ['b', 'c'], as: 'tabs' })).refusal
      ).toBe('popped_out');
      expect(holder.layout).toBe(before);
    });
  });
});

/**
 * `layout` without the tab `key` (the engine's leave rule).
 * @param {any} layout @param {string} key
 */
function removeKey(layout, key) {
  const walk = (/** @type {any} */ node) => {
    if ('tabs' in node) {
      const tabs = node.tabs.filter((/** @type {string} */ k) => k !== key);
      return tabs.length
        ? { ...node, tabs, showing: tabs.includes(node.showing) ? node.showing : tabs[0] }
        : null;
    }
    const items = node.items.map(walk).filter(Boolean);
    return items.length === 1 ? { ...items[0], size: node.size } : { ...node, items };
  };
  return walk(layout);
}
