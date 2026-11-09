// @ts-check
/**
 * Which panel the dock surfaces (dock-view.js):
 *   npx vitest run tests/interfaces/web_terminal/dock-view.test.mjs
 *
 * The sidebar's highlight is derived from the dock: the panel of the service
 * tile the workspace shows, if the page knows it and it is not detached.
 * `surfacedPanel` is undefined without a layout holder (the fallback host),
 * `deriveSurfaced` marks and announces the answer, and `followDock` wires the
 * dock's change events once and skips the holder's own rebuild. The dock's
 * layout, tile lookup and detached state are replaced by controllable stand-ins.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const JS = '../../../src/osprey/interfaces/web_terminal/static/js';

const dock = vi.hoisted(() => ({
  holder: /** @type {any} */ (null),
  target: /** @type {any} */ (null),
  api: /** @type {any} */ (null),
  poppedOut: /** @type {Set<string>} */ (new Set()),
  popoutCb: /** @type {null | (() => void)} */ (null),
}));

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/dock-apply.js', () => ({
  getLayoutHolder: () => dock.holder,
  isDetached: (/** @type {string} */ k) => dock.poppedOut.has(k),
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/dock-open.js', () => ({
  serviceTileTarget: () => dock.target,
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/dock-workspace.js', () => ({
  getDockApi: () => dock.api,
}));

/** @type {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/dock-view.js')} */
let mod;

beforeEach(async () => {
  dock.holder = {
    layout: {},
    context: () => ({}),
    applying: false,
    onDetachedChange: (/** @type {() => void} */ cb) => {
      dock.popoutCb = cb;
    },
  };
  dock.target = { showing: 'lattice' };
  dock.api = null;
  dock.poppedOut = new Set();
  dock.popoutCb = null;
  vi.resetModules();
  mod = await import(`${JS}/dock-view.js`);
});
afterEach(() => vi.restoreAllMocks());

describe('surfacedPanel', () => {
  test('is the showing panel of the service tile when the page knows it', () => {
    expect(mod.surfacedPanel((id) => id === 'lattice')).toBe('lattice');
  });

  test('is undefined without a layout holder', () => {
    dock.holder = null;
    expect(mod.surfacedPanel(() => true)).toBeUndefined();
  });

  test('is null with no service tile, with an unknown panel, or one detached', () => {
    dock.target = null;
    expect(mod.surfacedPanel(() => true)).toBeNull();
    dock.target = { showing: 'lattice' };
    expect(mod.surfacedPanel(() => false)).toBeNull();
    dock.poppedOut.add('lattice');
    expect(mod.surfacedPanel(() => true)).toBeNull();
    dock.poppedOut.clear();
    dock.target = { showing: null };
    expect(mod.surfacedPanel(() => true)).toBeNull();
  });
});

describe('deriveSurfaced', () => {
  test('marks the panel, announces the change, and reports a service tile', () => {
    const mark = vi.fn();
    const seen = vi.fn();
    document.addEventListener(mod.DOCK_VIEW_EVENT, seen);
    try {
      expect(mod.deriveSurfaced(() => true, mark)).toBe(true);
    } finally {
      document.removeEventListener(mod.DOCK_VIEW_EVENT, seen);
    }
    expect(mark).toHaveBeenCalledWith('lattice');
    expect(seen).toHaveBeenCalledTimes(1);
  });

  test('marks null and reports false when the dock holds no service tile', () => {
    dock.target = null;
    const mark = vi.fn();
    expect(mod.deriveSurfaced(() => true, mark)).toBe(false);
    expect(mark).toHaveBeenCalledWith(null);
  });

  test('is a no-op without a layout holder: no mark, no announcement', () => {
    dock.holder = null;
    const mark = vi.fn();
    const seen = vi.fn();
    document.addEventListener(mod.DOCK_VIEW_EVENT, seen);
    try {
      expect(mod.deriveSurfaced(() => true, mark)).toBeUndefined();
    } finally {
      document.removeEventListener(mod.DOCK_VIEW_EVENT, seen);
    }
    expect(mark).not.toHaveBeenCalled();
    expect(seen).not.toHaveBeenCalled();
  });
});

describe('followDock', () => {
  function api() {
    return {
      onDidLayoutChange: vi.fn(),
      onDidActivePanelChange: vi.fn(),
    };
  }

  test('does nothing without a dock, and may be wired later', () => {
    const onChange = vi.fn();
    mod.followDock(onChange);
    expect(dock.popoutCb).toBeNull();
    dock.api = api();
    mod.followDock(onChange);
    expect(dock.api.onDidLayoutChange).toHaveBeenCalledTimes(1);
  });

  test('follows layout, focus and detached-set changes', () => {
    dock.api = api();
    const onChange = vi.fn();
    mod.followDock(onChange);
    dock.api.onDidLayoutChange.mock.calls[0][0]();
    dock.api.onDidActivePanelChange.mock.calls[0][0]();
    dock.popoutCb?.();
    expect(onChange).toHaveBeenCalledTimes(3);
  });

  test("skips the holder's own rebuild", () => {
    dock.api = api();
    const onChange = vi.fn();
    mod.followDock(onChange);
    dock.holder.applying = true;
    dock.api.onDidLayoutChange.mock.calls[0][0]();
    expect(onChange).not.toHaveBeenCalled();
  });

  test('is wired once', () => {
    dock.api = api();
    mod.followDock(vi.fn());
    mod.followDock(vi.fn());
    expect(dock.api.onDidLayoutChange).toHaveBeenCalledTimes(1);
  });
});
