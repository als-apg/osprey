// @ts-check
/**
 * The Agent pane in its own window (agent-popout.js):
 *   npx vitest run tests/interfaces/web_terminal/agent-popout.test.mjs
 *
 * Hub side: the pane is a pop-out surface for the id `terminal`, offered in
 * the Simple view only. While its window is open the tile is closed and the
 * hub's console is asleep; when the window goes away the tile returns and the
 * console wakes. The window's session, attention and navigation reports are
 * taken only from the window the hub holds, and a panel's paste is forwarded
 * to that window. A flip to Expert, or an adoption while in Expert, takes the
 * window back.
 *
 * Window side: the page reports to its opener — hello, then the key its
 * console bound and every key it moves to, every change of its approval
 * count, and the panel links clicked in answers — and takes the hub's paste
 * from the opener alone. A mode change to Expert puts its console to sleep.
 *
 * The console (chat.js), the dock (dock-workspace.js), the panel broadcast
 * (terminal.js) and the insert seam (first-contact.js) are doubles: what is
 * asserted is which of them the module drives, and when. The protocol
 * modules (popout.js, popout-client.js), the chat bus, the session pointer
 * and frame-params.js are real.
 *
 * Each side boots ONCE for the file, as the page does: the protocol binds its
 * listeners to the shared `window` and has no teardown, so a fresh module per
 * test would leave every earlier instance listening too, and the doubles
 * would count each of them. A hub test that leaves the pane popped out takes
 * it back in the hook after it.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const JS = '../../../src/osprey/interfaces/web_terminal/static/js';
const MOD = `${JS}/agent-popout.js`;

const console_ = vi.hoisted(() => ({ sleepChat: vi.fn(), wakeChat: vi.fn() }));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/chat.js', () => console_);

const dock = vi.hoisted(() => ({
  openTerminalPanel: vi.fn(),
  closeTerminalPanel: vi.fn(() => true),
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/dock-workspace.js', () => dock);

const term = vi.hoisted(() => ({
  notifySessionChange: vi.fn(),
  getCurrentSessionId: vi.fn(() => 'K1'),
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/terminal.js', () => term);

const firstContact = vi.hoisted(() => ({ insertPrompt: vi.fn() }));
vi.mock(
  '../../../src/osprey/interfaces/web_terminal/static/js/first-contact.js',
  () => firstContact
);

/** The panels the hub answers with, for the window's roster read. */
const ROSTER = { enabled: ['artifacts', 'ariel'], custom: [{ id: 'notes' }, { name: 'nameless' }] };

const STORAGE_KEY = 'osprey-pty-session';

class FakeBroadcastChannel {
  /** @param {string} name */
  constructor(name) {
    this.name = name;
  }
  postMessage() {}
  addEventListener() {}
  close() {}
}

/** A stand-in for the Agent window the hub opened. */
function fakeAgentWindow() {
  return {
    closed: false,
    opener: window,
    name: 'osprey-popout-terminal',
    focus: vi.fn(),
    close: vi.fn(),
    postMessage: vi.fn(),
  };
}

/**
 * @param {object} data
 * @param {{ origin?: string, source?: any }} [init]
 */
function message(data, { origin = window.location.origin, source = null } = {}) {
  window.dispatchEvent(new MessageEvent('message', { data, origin, source }));
}

/** Let a queued microtask run. */
const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

beforeEach(() => {
  vi.stubGlobal('BroadcastChannel', FakeBroadcastChannel);
  vi.stubGlobal('screen', { availWidth: 1600, availHeight: 900, availLeft: 0, availTop: 0 });
  vi.stubGlobal(
    'fetch',
    vi.fn(async (/** @type {string} */ url) => {
      if (url.endsWith('/api/panels')) return { ok: true, status: 200, json: async () => ROSTER };
      throw new Error(`unstubbed fetch: ${url}`);
    })
  );
  localStorage.clear();
  document.body.innerHTML = '';
  document.documentElement.setAttribute('data-ui-mode', 'simple');
  vi.clearAllMocks();
  dock.closeTerminalPanel.mockReturnValue(true);
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  vi.useRealTimers();
  delete window.__OSPREY_PREFIX__;
  document.documentElement.removeAttribute('data-ui-mode');
  window.name = '';
});

/* ---- Hub side ------------------------------------------------------------ */

/** @type {{ mod: any, popout: any, events: any, navigate: import('vitest').Mock } | null} */
let hubSide = null;

/**
 * The hub, booted once: the protocol started with service deps, the Agent
 * registered on it, and the modules the assertions read.
 */
async function theHub() {
  if (!hubSide) {
    vi.resetModules();
    const popout = await import(`${JS}/popout.js`);
    const events = await import(`${JS}/chat-events.js`);
    const mod = await import(MOD);
    popout.initPopouts({
      popoutUrl: vi.fn(() => null),
      conceal: vi.fn(),
      redock: vi.fn(),
      sync: vi.fn(),
    });
    const navigate = vi.fn();
    mod.initAgentPopout({ navigate });
    hubSide = { mod, popout, events, navigate };
  }
  return hubSide;
}

/** Pop the Agent out through the protocol, as the tile button does. */
async function popOutAgent() {
  const hub = await theHub();
  const win = fakeAgentWindow();
  vi.spyOn(window, 'open').mockReturnValue(/** @type {any} */ (win));
  expect(hub.popout.popOutPanel('terminal')).toBe(true);
  return { ...hub, win };
}

// A test that leaves the pane in its window would leave the next one adopting
// over it; the pane is taken back here, outside the test's own assertions.
afterEach(() => {
  if (hubSide?.mod.isAgentPoppedOut()) hubSide.mod.recallAgentPopout();
});

describe('hub: the pop-out', () => {
  test('opens the hub page as the chat alone, under the pane id, in the Simple view', async () => {
    window.__OSPREY_PREFIX__ = '/u/alice';
    const { mod, popout } = await theHub();
    const open = vi.spyOn(window, 'open').mockReturnValue(/** @type {any} */ (fakeAgentWindow()));

    expect(popout.canPopOut('terminal')).toBe(true);
    popout.popOutPanel('terminal');

    const [url, name, features] = open.mock.calls[0];
    const opened = new URL(String(url));
    expect(opened.origin).toBe(window.location.origin);
    expect(opened.pathname).toBe('/u/alice/');
    expect(opened.searchParams.get('pane')).toBe('agent');
    expect(opened.searchParams.get('popout')).toBe('true');
    expect(opened.searchParams.get('mode')).toBe('simple');
    expect(opened.searchParams.has('theme')).toBe(true);
    expect(opened.searchParams.has('token')).toBe(false);
    expect(name).toBe('osprey-popout-terminal');
    expect(features).toMatch(/^popup=yes,width=\d+,height=\d+,left=\d+,top=\d+$/);
    expect(mod.isAgentPoppedOut()).toBe(true);
  });

  test('is sized like the Agent pane', async () => {
    const { popout } = await theHub();
    const pane = document.createElement('div');
    pane.className = 'terminal-panel';
    pane.getBoundingClientRect = () =>
      /** @type {DOMRect} */ ({
        width: 700,
        height: 500,
        left: 0,
        top: 0,
        right: 700,
        bottom: 500,
      });
    document.body.append(pane);
    const open = vi.spyOn(window, 'open').mockReturnValue(/** @type {any} */ (fakeAgentWindow()));

    popout.popOutPanel('terminal');

    expect(open.mock.calls[0][2]).toBe('popup=yes,width=700,height=500,left=450,top=200');
  });

  test('is not offered in the Expert view', async () => {
    document.documentElement.setAttribute('data-ui-mode', 'expert');
    const { mod, popout } = await theHub();
    const open = vi.spyOn(window, 'open');

    expect(mod.agentPopoutUrl()).toBeNull();
    expect(popout.canPopOut('terminal')).toBe(false);
    expect(popout.popOutPanel('terminal')).toBe(false);
    expect(open).not.toHaveBeenCalled();
  });

  test('closes the tile and puts the hub console to sleep while the window is open', async () => {
    const { events } = await popOutAgent();
    const heard = /** @type {number[]} */ ([]);
    const stop = events.onChat('attention', (/** @type {{ waiting: number }} */ d) =>
      heard.push(d.waiting)
    );

    expect(dock.closeTerminalPanel).toHaveBeenCalledTimes(1);
    expect(console_.sleepChat).toHaveBeenCalledTimes(1);
    expect(dock.openTerminalPanel).not.toHaveBeenCalled();
    expect(console_.wakeChat).not.toHaveBeenCalled();
    expect(heard).toEqual([]);
    stop();
  });

  test('a closed window brings the tile back and wakes the console, with no count waiting', async () => {
    vi.useFakeTimers();
    const { mod, win, events } = await popOutAgent();
    const heard = /** @type {number[]} */ ([]);
    const stop = events.onChat('attention', (/** @type {{ waiting: number }} */ d) =>
      heard.push(d.waiting)
    );

    win.closed = true;
    await vi.advanceTimersByTimeAsync(500);

    expect(mod.isAgentPoppedOut()).toBe(false);
    expect(heard).toEqual([0]);
    expect(dock.openTerminalPanel).toHaveBeenCalledTimes(1);
    expect(console_.wakeChat).toHaveBeenCalledTimes(1);
    // The tile is back before the console binds, so a focus lands on it.
    expect(dock.openTerminalPanel.mock.invocationCallOrder[0]).toBeLessThan(
      console_.wakeChat.mock.invocationCallOrder[0]
    );
    stop();
  });

  test('a hub reload adopts the open window through its hello, and sleeps its console', async () => {
    const { mod } = await theHub();
    const win = fakeAgentWindow();

    message({ type: 'osprey-popout-hello', panelId: 'terminal' }, { source: win });

    expect(mod.isAgentPoppedOut()).toBe(true);
    expect(dock.closeTerminalPanel).toHaveBeenCalledTimes(1);
    expect(console_.sleepChat).toHaveBeenCalledTimes(1);
  });

  test('recallAgentPopout closes the window and re-docks at once', async () => {
    const { mod, win } = await popOutAgent();

    expect(mod.recallAgentPopout()).toBe(true);

    expect(win.close).toHaveBeenCalledTimes(1);
    expect(mod.isAgentPoppedOut()).toBe(false);
    expect(dock.openTerminalPanel).toHaveBeenCalledTimes(1);
    expect(console_.wakeChat).toHaveBeenCalledTimes(1);
    expect(mod.recallAgentPopout()).toBe(false);
  });

  test('a window adopted while the hub is in Expert is taken straight back', async () => {
    const { mod } = await theHub();
    document.documentElement.setAttribute('data-ui-mode', 'expert');
    const win = fakeAgentWindow();

    message({ type: 'osprey-popout-hello', panelId: 'terminal' }, { source: win });
    expect(mod.isAgentPoppedOut()).toBe(true);
    await tick();

    expect(win.close).toHaveBeenCalledTimes(1);
    expect(mod.isAgentPoppedOut()).toBe(false);
    expect(console_.sleepChat).toHaveBeenCalledTimes(1);
    expect(console_.wakeChat).toHaveBeenCalledTimes(1);
    expect(dock.openTerminalPanel).toHaveBeenCalledTimes(1);
  });
});

describe("hub: the window's reports", () => {
  test('a session report from the held window reaches the panels', async () => {
    const { win } = await popOutAgent();

    message(
      { type: 'osprey-popout-session', panelId: 'terminal', session_id: 'K2' },
      { source: win }
    );

    expect(term.notifySessionChange).toHaveBeenCalledExactlyOnceWith('K2');
  });

  test('a session report from a forged source, a foreign origin or with no key is ignored', async () => {
    const { win } = await popOutAgent();

    message(
      { type: 'osprey-popout-session', panelId: 'terminal', session_id: 'K2' },
      { source: fakeAgentWindow() }
    );
    message(
      { type: 'osprey-popout-session', panelId: 'terminal', session_id: 'K2' },
      { origin: 'https://elsewhere.example', source: win }
    );
    message(
      { type: 'osprey-popout-session', panelId: 'terminal', session_id: '' },
      { source: win }
    );
    message({ type: 'osprey-popout-session', panelId: 'terminal', session_id: 7 }, { source: win });
    message({ type: 'osprey-popout-session', panelId: 'ariel', session_id: 'K2' }, { source: win });

    expect(term.notifySessionChange).not.toHaveBeenCalled();
  });

  test("an attention report goes on the hub's chat bus; a bad count reads as none", async () => {
    const { win, events } = await popOutAgent();
    const heard = /** @type {number[]} */ ([]);
    const stop = events.onChat('attention', (/** @type {{ waiting: number }} */ d) =>
      heard.push(d.waiting)
    );

    message({ type: 'osprey-popout-attention', panelId: 'terminal', waiting: 2 }, { source: win });
    message(
      { type: 'osprey-popout-attention', panelId: 'terminal', waiting: 'x' },
      { source: win }
    );
    message({ type: 'osprey-popout-attention', panelId: 'terminal', waiting: -1 }, { source: win });
    message(
      { type: 'osprey-popout-attention', panelId: 'terminal', waiting: 5 },
      { source: fakeAgentWindow() }
    );

    expect(heard).toEqual([2, 0, 0]);
    stop();
  });

  test('a navigation report opens the panel in the hub at a safe address only', async () => {
    const { win, navigate } = await popOutAgent();

    message(
      {
        type: 'osprey-popout-navigate',
        panelId: 'terminal',
        panel: 'ariel',
        url: '/panel/ariel/x',
      },
      { source: win }
    );
    message(
      { type: 'osprey-popout-navigate', panelId: 'terminal', panel: 'artifacts', url: '' },
      { source: win }
    );
    for (const url of ['//evil.example/x', 'javascript:alert(1)', 'https://evil.example', 7]) {
      message(
        { type: 'osprey-popout-navigate', panelId: 'terminal', panel: 'ariel', url },
        { source: win }
      );
    }
    message(
      { type: 'osprey-popout-navigate', panelId: 'terminal', panel: '', url: '/x' },
      { source: win }
    );
    message(
      { type: 'osprey-popout-navigate', panelId: 'terminal', panel: 'ariel', url: '/x' },
      { source: fakeAgentWindow() }
    );

    expect(navigate.mock.calls).toEqual([
      ['ariel', '/panel/ariel/x'],
      ['artifacts', ''],
    ]);
  });
});

describe("hub: a panel's paste", () => {
  test('goes nowhere while the pane is docked', async () => {
    const { mod } = await theHub();
    expect(mod.forwardPasteToAgent('SR:BPM1:X')).toBe(false);
  });

  test('goes to the Agent window while the pane is popped out', async () => {
    const { mod, win } = await popOutAgent();

    expect(mod.forwardPasteToAgent('SR:BPM1:X')).toBe(true);

    expect(win.postMessage).toHaveBeenCalledExactlyOnceWith(
      { type: 'osprey-paste-to-terminal', text: 'SR:BPM1:X' },
      window.location.origin
    );
  });
});

/* ---- Window side --------------------------------------------------------- */

/** The opener the window reports to, for the file. */
const pageHub = { closed: false, focus: vi.fn(), postMessage: vi.fn() };

/** @type {{ mod: any, events: any, pointerModule: any, booted: object[] } | null} */
let pageSide = null;

/**
 * The Agent window, booted once: named by the hub, with the hub as its
 * opener, and the page side started after the console bound `K1`. What the
 * boot itself reported is kept, because the hooks clear the opener's spy
 * before the next test. The opener and the name are put back before every
 * test, because the hooks clear those too.
 */
async function thePage() {
  vi.stubGlobal('opener', pageHub);
  window.name = 'osprey-popout-terminal';
  if (!pageSide) {
    localStorage.setItem(STORAGE_KEY, 'K1');
    vi.resetModules();
    const events = await import(`${JS}/chat-events.js`);
    const pointerModule = await import(`${JS}/session-pointer.js`);
    const mod = await import(MOD);
    mod.initAgentPopoutPage();
    const booted = pageHub.postMessage.mock.calls.map((call) => call[0]);
    for (const call of pageHub.postMessage.mock.calls) expect(call[1]).toBe(window.location.origin);
    pageSide = { mod, events, pointerModule, booted };
  }
  return pageSide;
}

describe('window: the page', () => {
  test('is the hub page opened with ?pane=agent', async () => {
    const { mod } = await thePage();
    expect(mod.isAgentPopoutPage()).toBe(false);
    window.history.replaceState({}, '', '/?pane=agent&popout=true&theme=dark&mode=simple');
    expect(mod.isAgentPopoutPage()).toBe(true);
    window.history.replaceState({}, '', '/');
  });

  test('says hello, then reports the key its console bound, then every key it moves to', async () => {
    const { booted, pointerModule } = await thePage();

    pointerModule.setPointer('K2');

    expect(booted).toEqual([
      { type: 'osprey-popout-hello', panelId: 'terminal' },
      { type: 'osprey-popout-session', panelId: 'terminal', session_id: 'K1' },
    ]);
    expect(pageHub.postMessage).toHaveBeenCalledExactlyOnceWith(
      { type: 'osprey-popout-session', panelId: 'terminal', session_id: 'K2' },
      window.location.origin
    );
  });

  test('reports every change of its approval count', async () => {
    const { events } = await thePage();

    events.emitChat('attention', { waiting: 3 });
    events.emitChat('attention', { waiting: 0 });

    expect(pageHub.postMessage.mock.calls.map((call) => call[0])).toEqual([
      { type: 'osprey-popout-attention', panelId: 'terminal', waiting: 3 },
      { type: 'osprey-popout-attention', panelId: 'terminal', waiting: 0 },
    ]);
  });

  test("takes the hub's paste from the opener at its own origin, and from nobody else", async () => {
    await thePage();
    const focus = vi.spyOn(window, 'focus').mockImplementation(() => {});

    message({ type: 'osprey-paste-to-terminal', text: 'SR:BPM1:X' }, { source: pageHub });
    message({ type: 'osprey-paste-to-terminal', text: 'nope' }, { source: { postMessage() {} } });
    message({ type: 'osprey-paste-to-terminal', text: 'nope' }, { source: null });
    message(
      { type: 'osprey-paste-to-terminal', text: 'nope' },
      { origin: 'https://elsewhere.example', source: pageHub }
    );
    message({ type: 'osprey-paste-to-terminal', text: '' }, { source: pageHub });

    expect(firstContact.insertPrompt).toHaveBeenCalledExactlyOnceWith('SR:BPM1:X', {
      append: true,
    });
    expect(focus).toHaveBeenCalledTimes(1);
  });

  test('a link in an answer opens its panel in the hub and brings the hub forward', async () => {
    const { mod } = await thePage();

    mod.openPanelInHub('ariel', '/panel/ariel/x');

    expect(pageHub.postMessage).toHaveBeenCalledExactlyOnceWith(
      {
        type: 'osprey-popout-navigate',
        panel: 'ariel',
        url: '/panel/ariel/x',
        panelId: 'terminal',
      },
      window.location.origin
    );
    expect(pageHub.focus).toHaveBeenCalledTimes(1);
  });

  test("reads the hub's panel roster for the answer links: enabled and custom ids", async () => {
    const { mod } = await thePage();

    await vi.waitFor(() => expect(mod.isHubPanel('ariel')).toBe(true));

    expect(mod.isHubPanel('artifacts')).toBe(true);
    expect(mod.isHubPanel('notes')).toBe(true);
    expect(mod.isHubPanel('okf')).toBe(false);
    expect(mod.isHubPanel('undefined')).toBe(false);
  });

  test('a mode change to Expert puts the console to sleep; back to Simple wakes it', async () => {
    await thePage();

    message({ type: 'osprey-mode-change', mode: 'expert' });
    expect(console_.sleepChat).toHaveBeenCalledTimes(1);
    expect(document.documentElement.getAttribute('data-ui-mode')).toBe('expert');

    message({ type: 'osprey-mode-change', mode: 'simple' });
    expect(console_.wakeChat).toHaveBeenCalledTimes(1);
  });

  test('with the opener gone, the page reports to nobody and keeps running', async () => {
    const { mod, pointerModule } = await thePage();
    vi.stubGlobal('opener', null);

    expect(() => pointerModule.setPointer('K3')).not.toThrow();
    expect(() => mod.openPanelInHub('ariel', '/x')).not.toThrow();
    expect(pageHub.postMessage).not.toHaveBeenCalled();
  });
});
