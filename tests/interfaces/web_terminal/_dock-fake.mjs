// @ts-check
/**
 * A hand-built DockviewApi stand-in for the dock adapter suites.
 *
 * The dock-iframe placement suite and the dock-glow suite each carried the same
 * fake; this module centralizes it. It models exactly the group bookkeeping the
 * placement engine relies on: `addPanel` with `direction: 'within'` joins the
 * reference group (the stacking dockview would do); any other position opens a
 * fresh group. The added panel becomes active (dockview's default), firing the
 * active-panel listeners synchronously. Removal collapses an emptied group.
 * Every `addPanel` options object is kept on `_added`, in call order.
 *
 * `addTerminal` seeds the native terminal card in its own group, which is how
 * the page boots before any service tile is placed.
 */

import { vi } from 'vitest';

/**
 * A fresh fake api with no groups and no panels.
 * @returns {any}
 */
export function makeDockApi() {
  let groupSeq = 0;
  /** @type {any} */
  const api = {
    activePanel: null,
    groups: /** @type {any[]} */ ([]),
    panels: /** @type {any[]} */ ([]),
    _added: /** @type {any[]} */ ([]),
    _activeCbs: /** @type {(() => void)[]} */ ([]),
    onDidLayoutChange: vi.fn(() => ({ dispose() {} })),
    onDidActivePanelChange: vi.fn((/** @type {() => void} */ cb) => {
      api._activeCbs.push(cb);
      return { dispose() {} };
    }),
    getPanel: (/** @type {string} */ id) => api.panels.find((/** @type {any} */ p) => p.id === id) ?? null,
    addPanel: (/** @type {any} */ opts) => {
      api._added.push(opts);
      const group = opts.position?.referenceGroup && opts.position.direction === 'within'
        ? opts.position.referenceGroup
        : makeGroup();
      const panel = { id: opts.id, title: opts.title, group, api: { setActive: vi.fn() } };
      group.panels.push(panel);
      group.activePanel = panel;
      api.panels.push(panel);
      api.activePanel = panel;
      for (const cb of api._activeCbs) cb();
      return panel;
    },
    removePanel: (/** @type {any} */ panel) => {
      api.panels = api.panels.filter((/** @type {any} */ p) => p !== panel);
      const group = panel.group;
      group.panels = group.panels.filter((/** @type {any} */ p) => p !== panel);
      if (group.panels.length === 0) {
        api.groups = api.groups.filter((/** @type {any} */ g) => g !== group);
      } else if (group.activePanel === panel) {
        group.activePanel = group.panels[0];
      }
      if (api.activePanel === panel) api.activePanel = group.panels[0] ?? api.panels[0] ?? null;
    },
  };
  function makeGroup() {
    const element = document.createElement('div');
    const content = document.createElement('div');
    content.className = 'dv-content-container';
    element.appendChild(content);
    const group = { id: `group-${++groupSeq}`, panels: [], activePanel: null, element };
    api.groups.push(group);
    return group;
  }
  api._makeGroup = makeGroup;
  return api;
}

/**
 * Seed the fake api with the native terminal card in its own group.
 * @param {any} api
 * @returns {any} The terminal panel.
 */
export function addTerminal(api) {
  const group = api._makeGroup();
  const terminal = { id: 'terminal', group, api: { setActive: vi.fn() } };
  group.panels.push(terminal);
  group.activePanel = terminal;
  api.panels.push(terminal);
  api.activePanel = terminal;
  return terminal;
}
