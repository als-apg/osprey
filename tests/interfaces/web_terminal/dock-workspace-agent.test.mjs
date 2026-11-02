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
 * (registered by initDockWorkspace before the dock adopts the card), and
 * every path reaches the dock through the layout holder's one fromJSON.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import { makeDockApi } from './_dock-fake.mjs';

const JS = '../../../src/osprey/interfaces/web_terminal/static/js';

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/terminal.js', () => ({
  fitTerminal: () => {},
  getCurrentSessionId: () => null,
}));
// The project key persistence resolves, when a test gives it one; with none,
// persistence stays inert and every apply builds a default.
const persistence = vi.hoisted(() => ({ projectKey: /** @type {string | null} */ (null) }));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/api.js', () => ({
  fetchJSON: async () => (persistence.projectKey ? { project_key: persistence.projectKey } : {}),
}));
vi.mock('/design-system/js/theme-manager.js', async (importOriginal) => ({
  .../** @type {object} */ (await importOriginal()),
  subscribe: () => {},
}));
// Whether the Agent pane runs in its own window (popout.js), as the layout
// paths ask it; the pop-out itself is popout.js's contract.
const popout = vi.hoisted(() => ({ popped: false }));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/popout.js', () => ({
  isPoppedOut: (/** @type {string} */ id) => id === 'terminal' && popout.popped,
  popOutPanel: () => true,
  canPopOut: () => true,
  onPopoutChange: () => {},
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
  persistence.projectKey = null;
  popout.popped = false;
  localStorage.clear();
  document.documentElement.removeAttribute('data-agent');
  mountHub();
  api = makeDockApi();
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

describe('while the Agent pane runs in its own window', () => {
  /** Pop the pane out: the window's adoption closes the tile. */
  function popOut() {
    dock.closeTerminalPanel();
    popout.popped = true;
  }

  test('a reset builds the default without the pane', () => {
    dock.initDockWorkspace();
    popOut();
    dock.resetDockLayout();
    expect(dock.isTerminalPanelOpen()).toBe(false);
    expect(document.documentElement.getAttribute('data-agent')).toBe('closed');
    expect(events).toEqual([true, false]);
  });

  test('the Agent row click re-creates nothing: the window is the pane', () => {
    dock.initDockWorkspace();
    popOut();
    dock.openTerminalPanel();
    expect(dock.isTerminalPanelOpen()).toBe(false);
    expect(events).toEqual([true, false]);
  });

  test('a stored layout that carries the pane restores without it', async () => {
    persistence.projectKey = 'p1';
    localStorage.setItem(
      'osprey-dock-layout-p1',
      JSON.stringify({
        version: 2,
        layout: {
          split: 'row',
          items: [
            { tile: 't1', tabs: ['ariel'], showing: 'ariel', size: 0.6 },
            { tile: 't2', tabs: ['terminal'], showing: 'terminal', size: 0.4 },
          ],
        },
        panels: {},
      })
    );
    dock.initDockWorkspace();
    popOut();
    await vi.waitFor(() => expect(api.getPanel('iframe:ariel')).not.toBeNull());

    expect(dock.isTerminalPanelOpen()).toBe(false);
    expect(api.getPanel('terminal')).toBeNull();
    expect(document.documentElement.getAttribute('data-agent')).toBe('closed');
  });

  test('the window closing brings the pane back through the ordinary reopen', () => {
    dock.initDockWorkspace();
    popOut();
    popout.popped = false;
    dock.openTerminalPanel();
    expect(dock.isTerminalPanelOpen()).toBe(true);
    expect(document.documentElement.hasAttribute('data-agent')).toBe(false);
    expect(events).toEqual([true, false, true]);
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
