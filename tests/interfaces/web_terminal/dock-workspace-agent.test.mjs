// @ts-check
/**
 * Unit tests for the Agent pane's presence in the dock (dock-workspace.js):
 *   npx vitest run tests/interfaces/web_terminal/dock-workspace-agent.test.mjs
 *
 * The sidebar's Agent row reads the pane's state from two places this module
 * owns: `html[data-agent="closed"]`, present exactly while the terminal panel
 * is absent from the dock, and the `osprey-agent-pane` `{detail: {open}}`
 * event on `document`, dispatched on every change. Every path that adds or
 * removes the pane must keep both in step: boot, close, reopen and a reset. The
 * dock is a hand-built stand-in (`_dock-fake.mjs`) behind a
 * fake `dockview-core` global; geometry is the browser suite's.
 *
 * Also pinned: the Agent tile adopts the chat console's bar in the chat
 * (registered by initDockWorkspace before the dock adopts the card), and the
 * default service width leaves the sidebar's width out.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import { makeDockApi } from './_dock-fake.mjs';

const JS = '../../../src/osprey/interfaces/web_terminal/static/js';

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/terminal.js', () => ({
  fitTerminal: () => {},
  getCurrentSessionId: () => null,
}));
// No project key: persistence stays inert, so every apply builds a default.
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/api.js', () => ({
  fetchJSON: async () => ({}),
}));
vi.mock('/design-system/js/theme-manager.js', async (importOriginal) => ({
  .../** @type {object} */ (await importOriginal()),
  subscribe: () => {},
}));

/** @type {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/dock-workspace.js')} */
let dock;
/** @type {any} the fake DockviewApi the module drives */
let api;
/** @type {any} the options initDockWorkspace passed to createDockview */
let options;
/** @type {boolean[]} every `osprey-agent-pane` detail.open, in order */
let events;

/** @param {Event} e */
const record = (e) => events.push(/** @type {CustomEvent} */ (e).detail.open);

/**
 * The fake api plus the members initDockWorkspace and the layout paths call
 * that `_dock-fake.mjs` leaves out.
 */
function fullApi() {
  const fake = makeDockApi();
  Object.assign(fake, {
    clear() {
      for (const panel of [...fake.panels]) fake.removePanel(panel);
    },
    toJSON() {
      return {
        panels: Object.fromEntries(
          fake.panels.map((/** @type {any} */ p) => [
            p.id,
            { id: p.id, contentComponent: p.id, title: p.title },
          ])
        ),
      };
    },
    fromJSON() {},
    updateOptions() {},
    onDidAddGroup: () => ({ dispose() {} }),
    onWillDragPanel: () => ({ dispose() {} }),
    onWillDragGroup: () => ({ dispose() {} }),
  });
  return fake;
}

/**
 * The hub's dock region as index.html ships it, with the console bar the chat
 * built before the dock.
 */
function mountHub() {
  document.body.innerHTML = `
    <div class="main-container">
      <div class="dock-root" id="dock-root"></div>
      <div class="dock-panel-sources">
        <div class="files-panel" id="right-panel">
          <div class="panel-content" id="panel-content"></div>
        </div>
        <div class="terminal-panel"><div class="terminal-card">
          <div class="terminal-header"><span id="terminal-label">Agent</span></div>
          <div class="terminal-body">
            <div id="operator-container" class="operator-container">
              <div class="op-session-bar"><span class="op-session-label">Agent</span></div>
            </div>
          </div>
        </div></div>
      </div>
    </div>`;
}

beforeEach(async () => {
  vi.resetModules();
  document.documentElement.removeAttribute('data-agent');
  mountHub();
  api = fullApi();
  /** @type {any} */ (window)['dockview-core'] = {
    themeLight: {},
    themeDark: {},
    createDockview: (/** @type {HTMLElement} */ _root, /** @type {any} */ opts) => {
      options = opts;
      return api;
    },
  };
  events = [];
  document.addEventListener('osprey-agent-pane', record);
  dock = await import(`${JS}/dock-workspace.js`);
});

afterEach(() => {
  document.removeEventListener('osprey-agent-pane', record);
  delete (/** @type {any} */ (window)['dockview-core']);
  document.documentElement.removeAttribute('data-agent');
  document.documentElement.removeAttribute('data-agent-surface');
  document.body.innerHTML = '';
});

describe('Agent pane state', () => {
  test('boot docks the pane open, says so once, and stamps nothing', () => {
    dock.initDockWorkspace();
    expect(dock.isTerminalPanelOpen()).toBe(true);
    expect(document.documentElement.hasAttribute('data-agent')).toBe(false);
    expect(events).toEqual([true]);
  });

  test('closing stamps data-agent="closed" and says open: false', () => {
    dock.initDockWorkspace();
    expect(dock.closeTerminalPanel()).toBe(true);
    expect(dock.isTerminalPanelOpen()).toBe(false);
    expect(document.documentElement.getAttribute('data-agent')).toBe('closed');
    expect(events).toEqual([true, false]);
  });

  test('a second close is a no-op: no event, the stamp stays', () => {
    dock.initDockWorkspace();
    dock.closeTerminalPanel();
    expect(dock.closeTerminalPanel()).toBe(false);
    expect(document.documentElement.getAttribute('data-agent')).toBe('closed');
    expect(events).toEqual([true, false]);
  });

  test('reopening clears the stamp and says open: true', () => {
    dock.initDockWorkspace();
    dock.closeTerminalPanel();
    dock.openTerminalPanel();
    expect(dock.isTerminalPanelOpen()).toBe(true);
    expect(document.documentElement.hasAttribute('data-agent')).toBe(false);
    expect(events).toEqual([true, false, true]);
  });

  test('opening an open pane focuses it and says nothing', () => {
    dock.initDockWorkspace();
    dock.openTerminalPanel();
    expect(api.getPanel('terminal').api.setActive).toHaveBeenCalled();
    expect(events).toEqual([true]);
  });

  test('a reset brings a closed pane back', () => {
    dock.initDockWorkspace();
    dock.closeTerminalPanel();
    dock.resetDockLayout();
    expect(document.documentElement.hasAttribute('data-agent')).toBe(false);
    expect(events).toEqual([true, false, true]);
  });

  test('the tile carries the Agent title', () => {
    dock.initDockWorkspace();
    expect(api.getPanel('terminal').title).toBe('Agent');
  });
});

describe('Agent bar registration', () => {
  test("the chat's Agent tile adopts the console bar it built", () => {
    document.documentElement.removeAttribute('data-agent-surface');
    const bar = document.querySelector('.op-session-bar');
    dock.initDockWorkspace();
    const tab = options.createTabComponent({ id: 'terminal' });
    expect(bar?.parentElement).toBe(tab.element);
    expect(tab.element.querySelector('.terminal-header')).toBeNull();
  });

  test("the terminal surface's Agent tile adopts the terminal header", () => {
    document.documentElement.setAttribute('data-agent-surface', 'terminal');
    const header = document.querySelector('.terminal-header');
    dock.initDockWorkspace();
    const tab = options.createTabComponent({ id: 'terminal' });
    expect(header?.parentElement).toBe(tab.element);
  });
});

describe('defaultServiceWidth', () => {
  test("leaves the sidebar's current width out of the split", () => {
    const sidebar = document.createElement('nav');
    sidebar.id = 'sidebar';
    Object.defineProperty(sidebar, 'offsetWidth', { value: 56 });
    document.body.append(sidebar);
    expect(dock.defaultServiceWidth()).toBe(Math.round((window.innerWidth - 56) * 0.6));
  });

  test('assumes the expanded sidebar when there is none on the page', () => {
    expect(dock.defaultServiceWidth()).toBe(Math.round((window.innerWidth - 232) * 0.6));
  });
});
