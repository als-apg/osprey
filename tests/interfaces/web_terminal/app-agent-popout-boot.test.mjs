// @ts-check
/**
 * The hub page booted as the Agent pane alone (app.js's agent-only boot):
 *   npx vitest run tests/interfaces/web_terminal/app-agent-popout-boot.test.mjs
 *
 * On `?pane=agent` the page boots the theme, the chat, the answer links and
 * the window side of agent-popout.js — and nothing of the shell: no terminal,
 * sidebar, panel manager, dock, palette or More menu, whose markup
 * terminal.css takes out of the layout. Without the parameter the page boots
 * as the hub. Every boot step is a double; what is asserted is which of them
 * run, and in which order the chat and the window side start.
 *
 * app.js registers its bootstrap on DOMContentLoaded at import, which has
 * already fired by the time a test module evaluates; the event is dispatched
 * again here, once per boot under test.
 */

import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';

vi.hoisted(() => {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (/** @type {string} */ url) => {
      throw new Error(`unstubbed fetch: ${url}`);
    })
  );
});

/** Every boot step that ran, in order. */
const ran = vi.hoisted(() => /** @type {string[]} */ ([]));

const doubles = vi.hoisted(() => ({
  initTheme: vi.fn((/** @type {any} */ opts) => {
    ran.push(`initTheme:${opts?.role}`);
  }),
  initTerminal: vi.fn(() => {
    ran.push('initTerminal');
  }),
  initChat: vi.fn(() => {
    ran.push('initChat');
  }),
  initSidebar: vi.fn(() => {
    ran.push('initSidebar');
    return null;
  }),
  initPanelManager: vi.fn(async () => {
    ran.push('initPanelManager');
  }),
  initDockWorkspace: vi.fn(() => {
    ran.push('initDockWorkspace');
  }),
  initCommandPalette: vi.fn(() => {
    ran.push('initCommandPalette');
  }),
  initMoreMenu: vi.fn(() => {
    ran.push('initMoreMenu');
  }),
  initAgentPopout: vi.fn(() => {
    ran.push('initAgentPopout');
  }),
  initAgentPopoutPage: vi.fn(() => {
    ran.push('initAgentPopoutPage');
  }),
  isAgentPopoutPage: vi.fn(() => false),
  openPanelInHub: vi.fn(),
  isHubPanel: vi.fn(() => false),
}));

vi.mock('/design-system/js/theme-manager.js', async (importOriginal) => ({
  .../** @type {object} */ (await importOriginal()),
  initTheme: doubles.initTheme,
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/terminal.js', () => ({
  initTerminal: doubles.initTerminal,
  focusTerminal: vi.fn(),
  pasteToTerminal: vi.fn(),
  getCurrentSessionId: () => null,
  getTerminalInstance: () => null,
  startExpert: vi.fn(),
  stopTerminal: vi.fn(),
  startTerminal: vi.fn(),
  restartTerminal: vi.fn(),
  switchSession: vi.fn(),
  setSessionLabel: vi.fn(),
  notifySessionChange: vi.fn(),
  clearStoredSessionId: vi.fn(),
  fitTerminal: vi.fn(),
  onSessionChange: vi.fn(),
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/chat.js', () => ({
  initChat: doubles.initChat,
  enterFromExpert: vi.fn(),
  sleepChat: vi.fn(),
  wakeChat: vi.fn(),
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/sidebar.js', () => ({
  initSidebar: doubles.initSidebar,
}));
vi.mock(
  '../../../src/osprey/interfaces/web_terminal/static/js/panel-manager.js',
  async (importOriginal) => ({
    .../** @type {object} */ (await importOriginal()),
    initPanelManager: doubles.initPanelManager,
  })
);
vi.mock(
  '../../../src/osprey/interfaces/web_terminal/static/js/dock-workspace.js',
  async (importOriginal) => ({
    .../** @type {object} */ (await importOriginal()),
    initDockWorkspace: doubles.initDockWorkspace,
  })
);
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/palette-boot.js', () => ({
  initCommandPalette: doubles.initCommandPalette,
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/more-menu.js', () => ({
  initMoreMenu: doubles.initMoreMenu,
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/agent-popout.js', () => ({
  initAgentPopout: doubles.initAgentPopout,
  initAgentPopoutPage: doubles.initAgentPopoutPage,
  isAgentPopoutPage: doubles.isAgentPopoutPage,
  openPanelInHub: doubles.openPanelInHub,
  isHubPanel: doubles.isHubPanel,
  recallAgentPopout: vi.fn(() => false),
  forwardPasteToAgent: vi.fn(() => false),
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/sidebar-artifacts.js', () => ({
  initSidebarArtifacts: vi.fn(),
  teardownSidebarArtifacts: vi.fn(),
}));

await import('../../../src/osprey/interfaces/web_terminal/static/js/app.js');

/** The chat's host, which both boots mount the console into. */
function mountOperatorContainer() {
  document.body.innerHTML = '<div id="operator-container"></div>';
}

/** Run the page's bootstrap again. */
function boot() {
  document.dispatchEvent(new Event('DOMContentLoaded'));
}

beforeEach(() => {
  ran.length = 0;
  vi.clearAllMocks();
  mountOperatorContainer();
});

afterEach(() => {
  document.body.innerHTML = '';
  doubles.isAgentPopoutPage.mockReturnValue(false);
});

describe('the agent-only boot', () => {
  test('boots the theme, the chat, then the window side, and nothing of the shell', () => {
    doubles.isAgentPopoutPage.mockReturnValue(true);

    boot();

    expect(ran).toEqual(['initTheme:hub', 'initChat', 'initAgentPopoutPage']);
    expect(doubles.initChat).toHaveBeenCalledWith('operator-container');
    for (const absent of [
      doubles.initTerminal,
      doubles.initSidebar,
      doubles.initPanelManager,
      doubles.initDockWorkspace,
      doubles.initCommandPalette,
      doubles.initMoreMenu,
      doubles.initAgentPopout,
    ]) {
      expect(absent).not.toHaveBeenCalled();
    }
  });

  test('a console that fails to build says so where it would have been, and the window side still starts', () => {
    doubles.isAgentPopoutPage.mockReturnValue(true);
    doubles.initChat.mockImplementationOnce(() => {
      throw new Error('no console');
    });
    const errors = vi.spyOn(console, 'error').mockImplementation(() => {});

    boot();

    expect(document.querySelector('#operator-container .chat-boot-failure')).not.toBeNull();
    expect(doubles.initAgentPopoutPage).toHaveBeenCalledTimes(1);
    expect(errors).toHaveBeenCalled();
    errors.mockRestore();
  });
});

describe('the hub boot', () => {
  test('starts the shell, with the Agent pop-out registered after the panel manager', () => {
    boot();

    expect(ran.slice(0, 3)).toEqual(['initTheme:hub', 'initTerminal', 'initChat']);
    expect(ran).toContain('initSidebar');
    expect(ran).toContain('initDockWorkspace');
    expect(ran).toContain('initCommandPalette');
    expect(ran.indexOf('initAgentPopout')).toBe(ran.indexOf('initPanelManager') + 1);
    expect(doubles.initAgentPopoutPage).not.toHaveBeenCalled();
  });
});
