// @ts-check
/**
 * A hand-built DockviewApi stand-in for the dock adapter suites.
 *
 * It models what the layout holder (dock-apply.js) and the adapters read off
 * dockview: `fromJSON` rebuilds the groups and panels a serialized layout
 * describes — one group per leaf, keyed by the leaf's id; one panel per view
 * — and `toJSON` hands the tree back with each group's current active view.
 * A group or panel whose id survives an apply keeps its object, the way the
 * holder's tile ids survive a rebuild, so a test can hold a reference across
 * applies. The active panel follows the serialized `activeGroup`, else the
 * first group; `panel.api.setActive()` activates it and fires the
 * active-panel listeners synchronously, as dockview does. The applied
 * layouts are kept on `_applied`, in call order; `fireLayout()` runs the
 * layout listeners (dockview fires them in a microtask after an apply; here a
 * test chooses the moment).
 *
 * There is no `addPanel` / `removePanel`: nothing but the holder mutates the
 * dock, and the holder does it with `fromJSON`. `FakeResizeObserver` is the
 * adapter's live-follow observer, inert: geometry is browser-suite turf.
 */

import { vi } from 'vitest';

export const DEFAULT_PX = Object.freeze({ width: 1000, height: 600 });

/** The terminal alone in its tile — the default arrangement. */
export const TERMINAL_LAYOUT = Object.freeze({
  tile: 'tt',
  tabs: ['terminal'],
  showing: 'terminal',
});

/**
 * A row of tiles, one per tab list, each tile showing its first tab and the
 * row shared evenly. Tile ids are given, or minted as `t1`, `t2`, ...
 * @param {...(string[] | { tile: string, tabs: string[], showing?: string })} items
 * @returns {any}
 */
export function rowOf(...items) {
  const size = 1 / items.length;
  return {
    split: 'row',
    items: items.map((item, i) =>
      Array.isArray(item)
        ? { tile: `t${i + 1}`, tabs: item, showing: item[0], size }
        : { showing: item.tabs[0], ...item, size }
    ),
  };
}

/**
 * A fresh fake api with no groups and no panels.
 * @param {{ width?: number, height?: number }} [px]
 * @returns {any}
 */
export function makeDockApi(px = DEFAULT_PX) {
  /** @type {any} */
  const api = {
    width: px.width,
    height: px.height,
    activePanel: null,
    activeGroup: null,
    groups: /** @type {any[]} */ ([]),
    panels: /** @type {any[]} */ ([]),
    _applied: /** @type {any[]} */ ([]),
    _serialized: /** @type {any} */ (null),
    _activeCbs: /** @type {(() => void)[]} */ ([]),
    _layoutCbs: /** @type {(() => void)[]} */ ([]),
    onDidLayoutChange: vi.fn((/** @type {() => void} */ cb) => {
      api._layoutCbs.push(cb);
      return { dispose() {} };
    }),
    onDidActivePanelChange: vi.fn((/** @type {() => void} */ cb) => {
      api._activeCbs.push(cb);
      return { dispose() {} };
    }),
    onDidAddGroup: vi.fn(() => ({ dispose() {} })),
    onWillDragPanel: vi.fn(() => ({ dispose() {} })),
    onWillDragGroup: vi.fn(() => ({ dispose() {} })),
    // rail-drag.js wires these two; HTML5 rail drags are not under test here.
    onUnhandledDragOver: vi.fn(() => ({ dispose() {} })),
    onDidDrop: vi.fn(() => ({ dispose() {} })),
    updateOptions: vi.fn(),
    /** @param {number} width @param {number} height */
    layout(width, height) {
      api.width = width;
      api.height = height;
    },
    getPanel: (/** @type {string} */ id) =>
      api.panels.find((/** @type {any} */ p) => p.id === id) ?? null,
    /** Run the active-panel listeners, as dockview does on every change. */
    fireActive() {
      for (const cb of [...api._activeCbs]) cb();
    },
    /** Run the layout listeners (dockview's settled-layout-change event). */
    fireLayout() {
      for (const cb of [...api._layoutCbs]) cb();
    },
    /**
     * Rebuild the groups and panels from a SerializedDockview.
     * @param {any} serialized
     */
    fromJSON(serialized) {
      api._applied.push(structuredClone(serialized));
      api._serialized = structuredClone(serialized);
      const before = new Map(api.groups.map((/** @type {any} */ g) => [g.id, g]));
      const panelsBefore = new Map(api.panels.map((/** @type {any} */ p) => [p.id, p]));
      /** @type {any[]} */
      const groups = [];
      /** @type {any[]} */
      const panels = [];
      const visit = (/** @type {any} */ node) => {
        if (node?.type === 'branch') {
          for (const child of node.data ?? []) visit(child);
          return;
        }
        if (node?.type !== 'leaf') return;
        const data = node.data ?? {};
        const group = before.get(data.id) ?? makeGroup(data.id);
        group.panels = [];
        for (const id of data.views ?? []) {
          const panel = panelsBefore.get(id) ?? makePanel(id);
          panel.group = group;
          panel.title = serialized.panels?.[id]?.title ?? panel.title;
          group.panels.push(panel);
          panels.push(panel);
        }
        group.activePanel =
          group.panels.find((/** @type {any} */ p) => p.id === data.activeView) ??
          group.panels[0] ??
          null;
        groups.push(group);
      };
      visit(serialized?.grid?.root);
      api.groups = groups;
      api.panels = panels;
      const active =
        groups.find((g) => g.id === serialized?.activeGroup)?.activePanel ??
        groups[0]?.activePanel ??
        null;
      api.activePanel = active;
      api.activeGroup = active?.group ?? null;
      api.fireActive();
    },
    /** The applied tree, with each group's current views and active view. */
    toJSON() {
      if (!api._serialized) {
        return {
          grid: { root: { type: 'branch', data: [] }, orientation: 'HORIZONTAL' },
          panels: {},
        };
      }
      const out = structuredClone(api._serialized);
      const byId = new Map(api.groups.map((/** @type {any} */ g) => [g.id, g]));
      const visit = (/** @type {any} */ node) => {
        if (node?.type === 'branch') {
          node.data = (node.data ?? []).filter((/** @type {any} */ child) => visit(child));
          return node.data.length > 0;
        }
        if (node?.type !== 'leaf') return false;
        const group = byId.get(node.data?.id);
        if (!group) return false;
        node.data.views = group.panels.map((/** @type {any} */ p) => p.id);
        node.data.activeView = group.activePanel?.id;
        return true;
      };
      visit(out.grid.root);
      out.activeGroup = api.activeGroup?.id;
      return out;
    },
  };
  /** @param {any} panel */
  function activate(panel) {
    panel.group.activePanel = panel;
    api.activePanel = panel;
    api.activeGroup = panel.group;
    api.fireActive();
  }
  /** @param {string} id */
  function makePanel(id) {
    /** @type {any} */
    const panel = { id, title: id, group: null };
    panel.api = { setActive: vi.fn(() => activate(panel)) };
    return panel;
  }
  /** @param {string} id */
  function makeGroup(id) {
    const element = document.createElement('div');
    element.className = 'dv-groupview';
    const content = document.createElement('div');
    content.className = 'dv-content-container';
    element.appendChild(content);
    return { id, panels: /** @type {any[]} */ ([]), activePanel: null, element };
  }
  return api;
}

/** The adapter's live-follow observer; geometry itself is browser-suite turf. */
export class FakeResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}
