/**
 * Contract tests for the tile-tab renderer (dock-tab.js): the custom dockview
 * tab that IS each tile's header bar. Every tile gets the same bar — identity
 * on the left, actions right-anchored; the bar itself is the drag handle.
 * Service tiles carry a visible `.tile-tab-title` and a Pop out button, and a
 * TAB STRIP in place of the title once the tile holds several tabs; the
 * Agent tile adopts a live bar instead — the chat console's `.op-session-bar`
 * in chat, the `.terminal-header` on the terminal surface. Run:
 *   npx vitest run tests/interfaces/web_terminal/dock-tab.test.mjs
 */
import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import { buildConsole } from '../../../src/osprey/interfaces/web_terminal/static/js/chat-console.js';

const MOD = '../../../src/osprey/interfaces/web_terminal/static/js/dock-tab.js';

const { closeTerminalPanel, closeDockPanel } = vi.hoisted(() => ({
  closeTerminalPanel: vi.fn(() => true),
  closeDockPanel: vi.fn(() => true),
}));
const { popOutPanel, canPopOut } = vi.hoisted(() => ({
  popOutPanel: vi.fn(() => true),
  canPopOut: vi.fn(() => true),
}));

// The pop-out itself (window, handle, tile retirement) is popout.js's
// contract; the bar only has to ask it the right things at the right times.
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/popout.js', () => ({
  popOutPanel,
  canPopOut,
}));

// dock-tab closes every tile through dock-workspace (the layout holder); the
// rest of that module (dockview itself) is out of scope here. The doubles
// cover what the modules dock-tab imports read from it.
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/dock-workspace.js', () => ({
  closeTerminalPanel,
  closeDockPanel,
  getDockApi: () => null,
  setLayoutRestoredHook: () => {},
  setServiceRedock: () => {},
  onDragGesture: () => [],
}));

/** Minimal dockview panel api stub for tab init params. */
function fakeApi() {
  return {
    close: vi.fn(),
    onDidTitleChange: vi.fn(() => ({ dispose: vi.fn() })),
  };
}

/** The two bar kinds, which share one close control. */
const CLOSE_BARS = [
  ['service', 'iframe:ariel'],
  ['terminal', 'terminal'],
];

/**
 * The page's terminal header, built DETACHED from the document — the real
 * production timing: dock-workspace's adoptSubtree moves the whole
 * .terminal-panel subtree into an unattached host div before dockview reinserts
 * it, so by the time the terminal's tab is built the header is reachable only
 * through the reference registered with setTerminalHeaderSource.
 * @returns {HTMLElement}
 */
function detachedTerminalHeader() {
  const card = document.createElement('div');
  card.className = 'terminal-card';
  const header = document.createElement('div');
  header.className = 'terminal-header';
  const sel = document.createElement('select');
  sel.id = 'session-selector';
  header.appendChild(sel);
  card.appendChild(header);
  return header;
}

/**
 * The chat console's Agent bar as the page has it before the dock adopts it:
 * built by chat-console.js, with the bell and Export the feature modules add
 * and the ⋯ chat.js appends last, inside a detached console container.
 * @returns {HTMLElement}
 */
function detachedConsoleBar() {
  const { bar, moreBtn } = buildConsole();
  for (const [cls, label] of [
    ['op-attention-toggle', 'Desktop notifications'],
    ['op-export', 'Export'],
  ]) {
    const btn = document.createElement('button');
    btn.className = `osprey-icon-btn ${cls}`;
    btn.setAttribute('aria-label', label);
    bar.append(btn);
  }
  bar.append(moreBtn);
  document.createElement('div').append(bar);
  return bar;
}

describe('tile-tab renderer', () => {
  beforeEach(() => {
    vi.resetModules();
    document.body.innerHTML = '';
    vi.restoreAllMocks();
    closeTerminalPanel.mockClear();
    closeDockPanel.mockClear();
    popOutPanel.mockClear();
    canPopOut.mockReset().mockReturnValue(true);
  });

  test('service tab renders visible title and close', async () => {
    const { createTileTab } = await import(MOD);
    const tab = createTileTab('iframe:ariel');
    tab.init({ title: 'ARIEL', params: {}, api: fakeApi() });

    expect(tab.element.classList.contains('tile-tab')).toBe(true);
    // One header grammar for every tile: name on the bar (title case, as the
    // sidebar shows it), close on the bar.
    expect(tab.element.querySelector('.tile-tab-title')?.textContent).toBe('Ariel');
    expect(tab.element.querySelector('.tile-tab-close')).toBeTruthy();
    expect(tab.element.querySelector('.tile-tab-actions')).toBeTruthy();
  });

  test('a service close click closes the tile through the holder, never the bare api.close()', async () => {
    const { createTileTab } = await import(MOD);
    const api = fakeApi();
    const tab = createTileTab('iframe:ariel');
    tab.init({ title: 'ARIEL', params: {}, api });
    /** @type {HTMLElement} */ (tab.element.querySelector('.tile-tab-close')).dispatchEvent(
      new MouseEvent('click', { bubbles: true, cancelable: true })
    );
    expect(closeDockPanel).toHaveBeenCalledExactlyOnceWith('ariel');
    expect(api.close).not.toHaveBeenCalled();
    expect(closeTerminalPanel).not.toHaveBeenCalled();
  });

  test('the Agent close runs closeTerminalPanel, never the bare api.close()', async () => {
    const { createTileTab, setTerminalHeaderSource } = await import(MOD);
    setTerminalHeaderSource(detachedTerminalHeader());
    const api = fakeApi();
    const tab = createTileTab('terminal');
    tab.init({ title: 'Agent', params: {}, api });
    const close = /** @type {HTMLElement} */ (tab.element.querySelector('.tile-tab-close'));
    expect(close.getAttribute('aria-label')).toBe('Close Agent pane');
    close.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    expect(closeTerminalPanel).toHaveBeenCalledTimes(1);
    expect(closeDockPanel).not.toHaveBeenCalled();
    expect(api.close).not.toHaveBeenCalled();
  });

  test('a service tile has a Pop out button before its close, an icon button with the popout icon', async () => {
    const { createTileTab } = await import(MOD);
    const tab = createTileTab('iframe:ariel');
    tab.init({ title: 'ARIEL', params: {}, api: fakeApi() });

    const actions = /** @type {HTMLElement} */ (tab.element.querySelector('.tile-tab-actions'));
    const popout = /** @type {HTMLButtonElement} */ (actions.querySelector('.tile-tab-popout'));
    expect([...actions.children].map((el) => el.className)).toEqual([
      'tile-tab-popout osprey-icon-btn',
      'tile-tab-close osprey-icon-btn',
    ]);
    expect(popout.title).toBe('Pop out');
    expect(popout.getAttribute('aria-label')).toBe('Pop out');
    expect(popout.disabled).toBe(false);
    expect(popout.querySelector('svg.osprey-ic')).not.toBeNull();
  });

  test('Pop out pops the service panel out, inside the click', async () => {
    const { createTileTab } = await import(MOD);
    const api = fakeApi();
    const tab = createTileTab('iframe:ariel');
    tab.init({ title: 'ARIEL', params: {}, api });

    /** @type {HTMLElement} */ (tab.element.querySelector('.tile-tab-popout')).dispatchEvent(
      new MouseEvent('click', { bubbles: true, cancelable: true })
    );

    expect(popOutPanel).toHaveBeenCalledExactlyOnceWith('ariel');
    expect(api.close).not.toHaveBeenCalled();
  });

  test('the Agent tile has no Pop out button', async () => {
    const { createTileTab, setTerminalHeaderSource } = await import(MOD);
    setTerminalHeaderSource(detachedTerminalHeader());
    const tab = createTileTab('terminal');
    tab.init({ title: 'Agent', params: {}, api: fakeApi() });

    expect(tab.element.querySelector('.tile-tab-popout')).toBeNull();
  });

  test('a panel that cannot pop out is disabled, with the reason, once the pointer reads the gate', async () => {
    canPopOut.mockReturnValue(false);
    const { createTileTab } = await import(MOD);
    const tab = createTileTab('iframe:notes');
    tab.init({ title: 'NOTES', params: {}, api: fakeApi() });
    const popout = /** @type {HTMLButtonElement} */ (tab.element.querySelector('.tile-tab-popout'));

    // Built enabled: the URL and frame kind are not known yet, and a button
    // disabled now would never get the pointer event that re-reads the gate.
    expect(popout.disabled).toBe(false);
    popout.dispatchEvent(new Event('pointerenter'));
    expect(popout.disabled).toBe(true);
    expect(popout.title).toBe('This panel runs at an opaque origin and cannot be popped out.');
    popout.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    expect(popOutPanel).not.toHaveBeenCalled();
  });

  test('the gate is read again at click time: a frame that resolved opaque after the bar was built', async () => {
    const { createTileTab } = await import(MOD);
    const tab = createTileTab('iframe:notes');
    tab.init({ title: 'NOTES', params: {}, api: fakeApi() });
    const popout = /** @type {HTMLButtonElement} */ (tab.element.querySelector('.tile-tab-popout'));
    expect(popout.disabled).toBe(false);

    canPopOut.mockReturnValue(false);
    popout.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));

    expect(popOutPanel).not.toHaveBeenCalled();
    expect(popout.disabled).toBe(true);
    expect(popout.title).toBe('This panel runs at an opaque origin and cannot be popped out.');
  });

  test('Pop out pointerdown is prevented so it cannot start a tile drag', async () => {
    const { createTileTab } = await import(MOD);
    const tab = createTileTab('iframe:ariel');
    tab.init({ title: 'ARIEL', params: {}, api: fakeApi() });
    const ev = new MouseEvent('pointerdown', { bubbles: true, cancelable: true });
    /** @type {HTMLElement} */ (tab.element.querySelector('.tile-tab-popout')).dispatchEvent(ev);
    expect(ev.defaultPrevented).toBe(true);
  });

  test.each(CLOSE_BARS)('%s close is an icon button with the x icon', async (_kind, id) => {
    const { createTileTab, setTerminalHeaderSource } = await import(MOD);
    setTerminalHeaderSource(detachedTerminalHeader());
    const tab = createTileTab(id);
    tab.init({ title: 'ARIEL', params: {}, api: fakeApi() });
    const close = /** @type {HTMLElement} */ (tab.element.querySelector('.tile-tab-close'));
    expect(close.classList.contains('osprey-icon-btn')).toBe(true);
    expect(close.querySelector('svg.osprey-ic')).not.toBeNull();
  });

  test.each(CLOSE_BARS)(
    '%s close pointerdown is prevented so it cannot start a tile drag',
    async (_kind, id) => {
      const { createTileTab, setTerminalHeaderSource } = await import(MOD);
      setTerminalHeaderSource(detachedTerminalHeader());
      const tab = createTileTab(id);
      tab.init({ title: 'ARIEL', params: {}, api: fakeApi() });
      const ev = new MouseEvent('pointerdown', { bubbles: true, cancelable: true });
      /** @type {HTMLElement} */ (tab.element.querySelector('.tile-tab-close')).dispatchEvent(ev);
      expect(ev.defaultPrevented).toBe(true);
    }
  );

  test('the strip carries the dockview panel id as a stable handle', async () => {
    const { createTileTab } = await import(MOD);
    const tab = createTileTab('iframe:ariel');
    tab.init({ title: 'ARIEL', params: {}, api: fakeApi() });
    expect(tab.element.dataset.panelId).toBe('iframe:ariel');
  });

  test('the title is visible text AND the accessible name, kept live on rename', async () => {
    const { createTileTab } = await import(MOD);
    const api = fakeApi();
    /** @type {(e: {title: string}) => void} */ let onTitle = () => {};
    api.onDidTitleChange = /** @type {any} */ (
      vi.fn((/** @type {any} */ fn) => {
        onTitle = fn;
        return { dispose: vi.fn() };
      })
    );
    const tab = createTileTab('iframe:okf');
    tab.init({ title: 'KNOWLEDGE', params: {}, api });

    // One string for eye and ear: the title-cased label the sidebar row shows.
    expect(tab.element.getAttribute('aria-label')).toBe('Knowledge');
    expect(tab.element.querySelector('.tile-tab-title')?.textContent).toBe('Knowledge');
    onTitle({ title: 'RENAMED' });
    expect(tab.element.getAttribute('aria-label')).toBe('Renamed');
    expect(tab.element.querySelector('.tile-tab-title')?.textContent).toBe('Renamed');
  });

  test('the Agent bar takes no aria-label — the bar it adopted names it', async () => {
    const { createTileTab } = await import(MOD);
    const api = fakeApi();
    const tab = createTileTab('terminal');
    tab.init({ title: 'Agent', params: {}, api });
    expect(tab.element.getAttribute('aria-label')).toBeNull();
    expect(api.onDidTitleChange).not.toHaveBeenCalled();
  });

  test('the whole service strip is the drag handle — nothing swallows pointerdown', async () => {
    const { createTileTab } = await import(MOD);
    const tab = createTileTab('iframe:ariel');
    tab.init({ title: 'ARIEL', params: {}, api: fakeApi() });
    document.body.appendChild(tab.element);
    const reachedRoot = vi.fn();
    tab.element.addEventListener('pointerdown', reachedRoot);

    /** @type {HTMLElement} */ (tab.element.querySelector('.tile-tab-title')).dispatchEvent(
      new MouseEvent('pointerdown', { bubbles: true, cancelable: true })
    );
    expect(reachedRoot).toHaveBeenCalledTimes(1);
  });

  describe('terminal tab', () => {
    /** @type {HTMLElement} */
    let header;

    beforeEach(() => {
      header = detachedTerminalHeader();
      document.documentElement.setAttribute('data-agent-surface', 'terminal');
    });

    afterEach(() => {
      document.documentElement.removeAttribute('data-agent-surface');
    });

    test('the terminal surface adopts .terminal-header and renders close, but no title', async () => {
      const { createTileTab, setTerminalHeaderSource, setTerminalConsoleBarSource } = await import(
        MOD
      );
      setTerminalHeaderSource(header);
      setTerminalConsoleBarSource(detachedConsoleBar());
      const tab = createTileTab('terminal');
      tab.init({ title: 'Agent', params: {}, api: fakeApi() });
      document.body.appendChild(tab.element);
      expect(tab.element.classList.contains('tile-tab-terminal')).toBe(true);
      // The page's ONE header node moved into the tab (relocation, not clone).
      expect(tab.element.querySelector('.terminal-header')).toBeTruthy();
      expect(document.querySelectorAll('.terminal-header').length).toBe(1);
      expect(tab.element.querySelector('.op-session-bar')).toBeNull();
      expect(tab.element.querySelector('.tile-tab-title')).toBeNull();
      expect(tab.element.querySelector('.tile-tab-close')).toBeTruthy();
    });

    test('the chat adopts the console bar: dot, Agent, pen, bell, share, more, close', async () => {
      const { createTileTab, setTerminalHeaderSource, setTerminalConsoleBarSource } = await import(
        MOD
      );
      document.documentElement.removeAttribute('data-agent-surface');
      setTerminalHeaderSource(header);
      const bar = detachedConsoleBar();
      setTerminalConsoleBarSource(bar);
      const tab = createTileTab('terminal');
      tab.init({ title: 'Agent', params: {}, api: fakeApi() });
      document.body.appendChild(tab.element);

      expect(bar.parentElement).toBe(tab.element);
      expect(tab.element.querySelector('.terminal-header')).toBeNull();
      expect(tab.element.querySelector('.new-session-btn')).toBeNull();
      const parts = [...tab.element.querySelectorAll('.osprey-dot, .op-session-label, button')].map(
        (el) => el.getAttribute('aria-label') ?? el.textContent
      );
      expect(parts).toEqual([
        '',
        'Agent',
        'New conversation',
        'Desktop notifications',
        'Export',
        'Earlier conversations',
        'Close Agent pane',
      ]);
    });

    test('the chat falls back to the header when the console never built', async () => {
      const { createTileTab, setTerminalHeaderSource } = await import(MOD);
      document.documentElement.removeAttribute('data-agent-surface');
      setTerminalHeaderSource(header);
      const tab = createTileTab('terminal');
      tab.init({ title: 'Agent', params: {}, api: fakeApi() });
      expect(tab.element.querySelector('.terminal-header')).toBe(header);
    });

    test('a surface change swaps the bar on the live tab and sends the other home', async () => {
      const { createTileTab, setTerminalHeaderSource, setTerminalConsoleBarSource } = await import(
        MOD
      );
      const { showChatSurface, showTerminalSurface } =
        await import('../../../src/osprey/interfaces/web_terminal/static/js/agent-surface.js');
      const card = document.createElement('div');
      card.append(header);
      const console_ = document.createElement('div');
      const bar = detachedConsoleBar();
      console_.append(bar);
      setTerminalHeaderSource(header);
      setTerminalConsoleBarSource(bar);
      const tab = createTileTab('terminal');
      tab.init({ title: 'Agent', params: {}, api: fakeApi() });
      expect(header.parentElement).toBe(tab.element);

      showChatSurface();
      expect(bar.parentElement).toBe(tab.element);
      expect(header.parentElement).toBe(card);

      showTerminalSurface();
      expect(header.parentElement).toBe(tab.element);
      expect(bar.parentElement).toBe(console_);
    });

    test('a tab built after a surface change adopts the other bar', async () => {
      const { createTileTab, setTerminalHeaderSource, setTerminalConsoleBarSource } = await import(
        MOD
      );
      setTerminalHeaderSource(header);
      const bar = detachedConsoleBar();
      setTerminalConsoleBarSource(bar);
      const onTerminal = createTileTab('terminal');
      onTerminal.init({ title: 'Agent', params: {}, api: fakeApi() });
      expect(header.parentElement).toBe(onTerminal.element);

      document.documentElement.removeAttribute('data-agent-surface');
      const inChat = createTileTab('terminal');
      inChat.init({ title: 'Agent', params: {}, api: fakeApi() });
      expect(bar.parentElement).toBe(inChat.element);

      document.documentElement.setAttribute('data-agent-surface', 'terminal');
      const back = createTileTab('terminal');
      back.init({ title: 'Agent', params: {}, api: fakeApi() });
      expect(header.parentElement).toBe(back.element);
    });

    test('re-adoption after detach (close → reopen) reuses the cached header node', async () => {
      const { createTileTab, setTerminalHeaderSource } = await import(MOD);
      setTerminalHeaderSource(header);
      const first = createTileTab('terminal');
      first.init({ title: 'Agent', params: {}, api: fakeApi() });
      const headerNode = first.element.querySelector('.terminal-header');
      first.element.remove(); // dockview discards the tab DOM on removePanel
      const second = createTileTab('terminal');
      second.init({ title: 'Agent', params: {}, api: fakeApi() });
      expect(second.element.querySelector('.terminal-header')).toBe(headerNode);
    });

    test('pointerdown on interactive header children stops propagation (no drag), on plain header it bubbles (drag ok)', async () => {
      const { createTileTab, setTerminalHeaderSource } = await import(MOD);
      setTerminalHeaderSource(header);
      const tab = createTileTab('terminal');
      tab.init({ title: 'Agent', params: {}, api: fakeApi() });
      document.body.appendChild(tab.element);
      const reachedRoot = vi.fn();
      tab.element.addEventListener('pointerdown', reachedRoot);

      tab.element
        .querySelector('#session-selector')
        .dispatchEvent(new MouseEvent('pointerdown', { bubbles: true, cancelable: true }));
      expect(reachedRoot).not.toHaveBeenCalled(); // contained — no tile drag

      tab.element
        .querySelector('.terminal-header')
        .dispatchEvent(new MouseEvent('pointerdown', { bubbles: true, cancelable: true }));
      expect(reachedRoot).toHaveBeenCalledTimes(1); // plain surface drags
    });

    test("the console bar's buttons act instead of dragging; its plain surface drags", async () => {
      const { createTileTab, setTerminalConsoleBarSource } = await import(MOD);
      document.documentElement.removeAttribute('data-agent-surface');
      const bar = detachedConsoleBar();
      setTerminalConsoleBarSource(bar);
      const tab = createTileTab('terminal');
      tab.init({ title: 'Agent', params: {}, api: fakeApi() });
      document.body.appendChild(tab.element);
      const reachedRoot = vi.fn();
      tab.element.addEventListener('pointerdown', reachedRoot);

      /** @type {HTMLElement} */ (bar.querySelector('.op-session-new')).dispatchEvent(
        new MouseEvent('pointerdown', { bubbles: true, cancelable: true })
      );
      expect(reachedRoot).not.toHaveBeenCalled();
      /** @type {HTMLElement} */ (bar.querySelector('.op-session-label')).dispatchEvent(
        new MouseEvent('pointerdown', { bubbles: true, cancelable: true })
      );
      expect(reachedRoot).toHaveBeenCalledTimes(1);
    });
  });
});

/**
 * The tile header is a panel's second right-click surface: the same verb menu
 * its sidebar row offers. dock-tab holds no policy — it forwards the press
 * through the setTileContextMenuHandler seam (panel-manager registers into it,
 * and cannot be imported here: dock-workspace already imports this module) and
 * suppresses the browser's own menu ONLY when the handler reports it opened
 * one. Declining is doing nothing at all, which is what leaves copy/paste
 * working inside a contributed search input.
 */
describe('tile header context menu', () => {
  beforeEach(() => {
    vi.resetModules();
    document.body.innerHTML = '';
    vi.restoreAllMocks();
  });

  /** A right-click, cancelable so defaultPrevented is observable. */
  function rightClick(x = 40, y = 12) {
    return new MouseEvent('contextmenu', {
      bubbles: true,
      cancelable: true,
      clientX: x,
      clientY: y,
    });
  }

  /**
   * The (id, opts) pair a handler mock was called with. Named here because a
   * `vi.fn(() => true)` carries a zero-argument signature, so its recorded
   * call tuple is not indexable without saying what it holds.
   * @param {any} handler
   * @returns {{id: string, opts: {x: number, y: number, anchorEl: HTMLElement}}}
   */
  function firstCall(handler) {
    const [id, opts] = handler.mock.calls[0];
    return { id, opts };
  }

  test('a service bar forwards the press with the SERVICE id, the bar as anchor, and the cursor position', async () => {
    const { createTileTab, setTileContextMenuHandler } = await import(MOD);
    const handler = vi.fn(() => true);
    setTileContextMenuHandler(handler);
    const tab = createTileTab('iframe:ariel');
    tab.init({ title: 'ARIEL', params: {}, api: fakeApi() });
    document.body.appendChild(tab.element);

    const ev = rightClick(120, 30);
    tab.element.dispatchEvent(ev);

    expect(handler).toHaveBeenCalledTimes(1);
    const { id, opts } = firstCall(handler);
    // The dockview placeholder prefix never leaves this module — the handler
    // speaks the same panel ids the sidebar does.
    expect(id).toBe('ariel');
    expect(opts.anchorEl).toBe(tab.element);
    expect(opts.x).toBe(120);
    expect(opts.y).toBe(30);
    // The menu owns this press: no native menu on top of it.
    expect(ev.defaultPrevented).toBe(true);
  });

  test('a declined press keeps the browser default (nothing is prevented)', async () => {
    const { createTileTab, setTileContextMenuHandler } = await import(MOD);
    // What a surface with no rows reports: no menu opened, so the native one
    // must still appear.
    setTileContextMenuHandler(vi.fn(() => false));
    const tab = createTileTab('iframe:ariel');
    tab.init({ title: 'ARIEL', params: {}, api: fakeApi() });
    document.body.appendChild(tab.element);

    const ev = rightClick();
    tab.element.dispatchEvent(ev);
    expect(ev.defaultPrevented).toBe(false);
  });

  test('with no handler registered the bar is inert — no throw, no suppression', async () => {
    const { createTileTab } = await import(MOD);
    const tab = createTileTab('iframe:ariel');
    tab.init({ title: 'ARIEL', params: {}, api: fakeApi() });
    document.body.appendChild(tab.element);

    const ev = rightClick();
    expect(() => tab.element.dispatchEvent(ev)).not.toThrow();
    expect(ev.defaultPrevented).toBe(false);
  });

  test('interactive bar children are excluded — their native menu survives', async () => {
    const { createTileTab, setTerminalHeaderSource, setTileContextMenuHandler } = await import(MOD);
    const handler = vi.fn(() => true);
    setTileContextMenuHandler(handler);
    const tab = createTileTab('iframe:ariel');
    tab.init({ title: 'ARIEL', params: {}, api: fakeApi() });
    document.body.appendChild(tab.element);

    // A contributed control standing in for the panel search inputs that make
    // this exclusion matter: right-clicking one must still offer copy/paste.
    const input = document.createElement('input');
    /** @type {HTMLElement} */ (tab.element.querySelector('.tile-tab-contrib')).appendChild(input);

    const onInput = rightClick();
    input.dispatchEvent(onInput);
    expect(handler).not.toHaveBeenCalled();
    expect(onInput.defaultPrevented).toBe(false);

    // The close button is interactive too, by the same rule.
    const onClose = rightClick();
    /** @type {HTMLElement} */ (tab.element.querySelector('.tile-tab-close')).dispatchEvent(
      onClose
    );
    expect(handler).not.toHaveBeenCalled();
    expect(onClose.defaultPrevented).toBe(false);

    // Plain bar surface still opens the menu.
    /** @type {HTMLElement} */ (tab.element.querySelector('.tile-tab-title')).dispatchEvent(
      rightClick()
    );
    expect(handler).toHaveBeenCalledTimes(1);

    // The terminal bar's adopted header applies the same rule to its selector.
    const header = detachedTerminalHeader();
    setTerminalHeaderSource(header);
    const terminal = createTileTab('terminal');
    terminal.init({ title: 'Agent', params: {}, api: fakeApi() });
    document.body.appendChild(terminal.element);
    const onSelector = rightClick();
    /** @type {HTMLElement} */ (header.querySelector('#session-selector')).dispatchEvent(
      onSelector
    );
    expect(handler).toHaveBeenCalledTimes(1);
    expect(onSelector.defaultPrevented).toBe(false);
  });

  test('the terminal bar forwards the terminal id once, anchored to its adopted header', async () => {
    const { createTileTab, setTerminalHeaderSource, setTileContextMenuHandler } = await import(MOD);
    const header = detachedTerminalHeader();
    setTerminalHeaderSource(header);
    const handler = vi.fn(() => true);
    setTileContextMenuHandler(handler);
    const tab = createTileTab('terminal');
    tab.init({ title: 'Agent', params: {}, api: fakeApi() });
    document.body.appendChild(tab.element);

    const ev = rightClick(8, 4);
    /** @type {HTMLElement} */ (tab.element.querySelector('.terminal-header')).dispatchEvent(ev);

    // Exactly once: the header fills the terminal bar, so listening on the tab
    // root as well would run the handler twice for one press.
    expect(handler).toHaveBeenCalledTimes(1);
    const { id, opts } = firstCall(handler);
    expect(id).toBe('terminal');
    expect(opts.anchorEl).toBe(header);
    expect(opts.x).toBe(8);
    expect(opts.y).toBe(4);
    expect(ev.defaultPrevented).toBe(true);
  });

  test('the terminal header keeps ONE contextmenu listener across tab rebuilds', async () => {
    const { createTileTab, setTerminalHeaderSource, setTileContextMenuHandler } = await import(MOD);
    const header = detachedTerminalHeader();
    // Registration is per NODE, not per tab: close→reopen rebuilds the tab but
    // re-adopts the same header, so a per-construction wiring would stack up.
    setTerminalHeaderSource(header);
    const handler = vi.fn(() => true);
    setTileContextMenuHandler(handler);

    const first = createTileTab('terminal');
    first.init({ title: 'Agent', params: {}, api: fakeApi() });
    first.element.remove();
    const second = createTileTab('terminal');
    second.init({ title: 'Agent', params: {}, api: fakeApi() });
    document.body.appendChild(second.element);

    /** @type {HTMLElement} */ (second.element.querySelector('.terminal-header')).dispatchEvent(
      rightClick()
    );
    expect(handler).toHaveBeenCalledTimes(1);
  });

  test('the console bar forwards the terminal id too, anchored to itself', async () => {
    const { createTileTab, setTerminalConsoleBarSource, setTileContextMenuHandler } = await import(
      MOD
    );
    document.documentElement.removeAttribute('data-agent-surface');
    const bar = detachedConsoleBar();
    setTerminalConsoleBarSource(bar);
    const handler = vi.fn(() => true);
    setTileContextMenuHandler(handler);
    const tab = createTileTab('terminal');
    tab.init({ title: 'Agent', params: {}, api: fakeApi() });
    document.body.appendChild(tab.element);

    /** @type {HTMLElement} */ (bar.querySelector('.op-session-label')).dispatchEvent(rightClick());
    expect(handler).toHaveBeenCalledTimes(1);
    const { id, opts } = firstCall(handler);
    expect(id).toBe('terminal');
    expect(opts.anchorEl).toBe(bar);
    document.documentElement.removeAttribute('data-agent-surface');
  });
});

/**
 * A dockview group with `entries` as its panels: `[id, title]` pairs, the one
 * at `activeIndex` showing. Its model emits the three panel events the bar
 * follows, each panel's api answers `setActive` by making it the group's
 * active panel, and `rename` fires the panel's title change.
 * @param {[string, string][]} entries
 * @param {number} [activeIndex]
 */
function fakeGroup(entries, activeIndex = 0) {
  /** @type {Record<string, Array<(e: any) => void>>} */
  const listeners = { add: [], remove: [], active: [] };
  /** @param {Array<(e: any) => void>} list */
  const emitter = (list) =>
    vi.fn((/** @type {(e: any) => void} */ fn) => {
      list.push(fn);
      return {
        dispose: () => {
          const i = list.indexOf(fn);
          if (i >= 0) list.splice(i, 1);
        },
      };
    });
  /** @type {any} */
  const group = {
    id: 't1',
    model: {
      onDidAddPanel: emitter(listeners.add),
      onDidRemovePanel: emitter(listeners.remove),
      onDidActivePanelChange: emitter(listeners.active),
    },
    panels: [],
    activePanel: null,
  };
  /** @param {string} id @param {string} title */
  const panelOf = (id, title) => {
    /** @type {Array<(e: any) => void>} */
    const titleListeners = [];
    /** @type {any} */
    const panel = {
      id,
      title,
      api: {
        setActive: vi.fn(() => {
          group.activePanel = panel;
          for (const fn of [...listeners.active]) fn({});
        }),
        onDidTitleChange: vi.fn((/** @type {(e: any) => void} */ fn) => {
          titleListeners.push(fn);
          return { dispose: vi.fn() };
        }),
        onDidGroupChange: vi.fn(() => ({ dispose: vi.fn() })),
        close: vi.fn(),
        get group() {
          return group;
        },
      },
      /** @param {string} next */
      rename(next) {
        panel.title = next;
        for (const fn of [...titleListeners]) fn({ title: next });
      },
    };
    return panel;
  };
  group.panels = entries.map(([id, title]) => panelOf(id, title));
  group.activePanel = group.panels[activeIndex] ?? null;
  /** @param {string} id @param {string} title */
  group.add = (id, title) => {
    const panel = panelOf(id, title);
    group.panels.push(panel);
    for (const fn of [...listeners.add]) fn({});
    return panel;
  };
  /** @param {string} id */
  group.remove = (id) => {
    group.panels = group.panels.filter((/** @type {any} */ p) => p.id !== id);
    if (group.activePanel?.id === id) group.activePanel = group.panels[0] ?? null;
    for (const fn of [...listeners.remove]) fn({});
  };
  return group;
}

/**
 * The bar of `panel` in `group`, built and initialised as dockview does, and
 * mounted inside a `.dv-tab` (active unless `inactive`) under `.dv-groupview`.
 * @param {any} mod  the dock-tab module
 * @param {any} panel
 * @param {{ inactive?: boolean, header?: HTMLElement }} [opts]
 */
function mountBar(mod, panel, { inactive = false, header } = {}) {
  const tab = mod.createTileTab(panel.id);
  tab.init({ title: panel.title, params: {}, api: panel.api });
  const host = header ?? document.createElement('div');
  if (!header) {
    host.className = 'dv-groupview';
    document.body.appendChild(host);
  }
  let strip = host.querySelector('.dv-tabs-and-actions-container');
  if (!strip) {
    strip = document.createElement('div');
    strip.className = 'dv-tabs-and-actions-container';
    host.appendChild(strip);
  }
  const dvTab = document.createElement('div');
  dvTab.className = inactive ? 'dv-tab dv-inactive-tab' : 'dv-tab dv-active-tab';
  dvTab.appendChild(tab.element);
  strip.appendChild(dvTab);
  return { tab, host };
}

/** @param {HTMLElement} root */
const items = (root) =>
  [...root.querySelectorAll('.tile-tab-item')].map((el) => ({
    key: el.getAttribute('data-panel-key'),
    text: el.textContent,
    showing: el.classList.contains('tile-tab-item--showing'),
    selected: el.getAttribute('aria-selected'),
  }));

describe('a tile with tabs', () => {
  beforeEach(() => {
    vi.resetModules();
    document.body.innerHTML = '';
    closeDockPanel.mockClear();
  });

  test('one tab renders the title, no strip, a tile close, and the tile and key stamps', async () => {
    const mod = await import(MOD);
    const group = fakeGroup([['iframe:ariel', 'ARIEL']]);
    const { tab } = mountBar(mod, group.panels[0]);
    const root = tab.element;
    expect(root.dataset.tile).toBe('t1');
    expect(root.getAttribute('data-panel-key')).toBe('ariel');
    const title = /** @type {HTMLElement} */ (root.querySelector('.tile-tab-title'));
    const strip = /** @type {HTMLElement} */ (root.querySelector('.tile-tab-tabs'));
    expect(title.hidden).toBe(false);
    expect(title.textContent).toBe('Ariel');
    expect(strip.hidden).toBe(true);
    expect(strip.children).toHaveLength(0);
    expect(root.querySelector('.tile-tab-close')?.getAttribute('aria-label')).toBe('Close tile');
  });

  test('two tabs render the strip in bar order, the showing one marked, and a tab close', async () => {
    const mod = await import(MOD);
    const group = fakeGroup(
      [
        ['iframe:ariel', 'ARIEL'],
        ['iframe:okf', 'KNOWLEDGE'],
      ],
      1
    );
    const { tab } = mountBar(mod, group.panels[1]);
    const root = tab.element;
    const title = /** @type {HTMLElement} */ (root.querySelector('.tile-tab-title'));
    const strip = /** @type {HTMLElement} */ (root.querySelector('.tile-tab-tabs'));
    expect(title.hidden).toBe(true);
    expect(strip.hidden).toBe(false);
    expect(strip.getAttribute('role')).toBe('tablist');
    expect(strip.tabIndex).toBe(0);
    expect(items(root)).toEqual([
      { key: 'ariel', text: 'Ariel', showing: false, selected: 'false' },
      { key: 'okf', text: 'Knowledge', showing: true, selected: 'true' },
    ]);
    expect(strip.getAttribute('aria-activedescendant')).toBe('tile-tab-item-t1-okf');
    for (const item of strip.querySelectorAll('.tile-tab-item')) {
      expect(item.getAttribute('role')).toBe('tab');
      expect(/** @type {HTMLElement} */ (item).draggable).toBe(true);
      expect(/** @type {HTMLElement} */ (item).tabIndex).toBe(-1);
    }
    expect(root.querySelector('.tile-tab-close')?.getAttribute('aria-label')).toBe('Close tab');
    // The root still names its own panel, the one it was built for.
    expect(root.getAttribute('aria-label')).toBe('Knowledge');
  });

  test('a panel joining or leaving the tile re-renders the strip; a rename renames its item', async () => {
    const mod = await import(MOD);
    const group = fakeGroup([['iframe:ariel', 'ARIEL']]);
    const { tab } = mountBar(mod, group.panels[0]);
    const root = tab.element;
    const strip = /** @type {HTMLElement} */ (root.querySelector('.tile-tab-tabs'));
    const okf = group.add('iframe:okf', 'KNOWLEDGE');
    expect(strip.hidden).toBe(false);
    expect(items(root).map((i) => i.key)).toEqual(['ariel', 'okf']);
    // A title change on a sibling the bar was built before it joined still
    // reaches the strip: the title subscriptions are re-made on every change.
    okf.rename('WIKI');
    expect(items(root).map((i) => i.text)).toEqual(['Ariel', 'Wiki']);
    group.remove('iframe:okf');
    expect(strip.hidden).toBe(true);
    expect(/** @type {HTMLElement} */ (root.querySelector('.tile-tab-title')).hidden).toBe(false);
  });

  test('a click on an item shows that tab in dockview', async () => {
    const mod = await import(MOD);
    const group = fakeGroup([
      ['iframe:ariel', 'ARIEL'],
      ['iframe:okf', 'KNOWLEDGE'],
    ]);
    const { tab } = mountBar(mod, group.panels[0]);
    const okfItem = /** @type {HTMLElement} */ (
      tab.element.querySelector('.tile-tab-item[data-panel-key="okf"]')
    );
    okfItem.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    expect(group.panels[1].api.setActive).toHaveBeenCalledTimes(1);
    expect(group.panels[0].api.setActive).not.toHaveBeenCalled();
    // dockview answered: the strip now marks the new showing tab.
    expect(items(tab.element).find((i) => i.showing)?.key).toBe('okf');
  });

  test('arrow keys, Home and End on the strip move the showing tab, wrapping; Delete closes it', async () => {
    const mod = await import(MOD);
    const group = fakeGroup([
      ['iframe:a', 'A'],
      ['iframe:b', 'B'],
      ['iframe:c', 'C'],
    ]);
    const { tab } = mountBar(mod, group.panels[0]);
    const strip = /** @type {HTMLElement} */ (tab.element.querySelector('.tile-tab-tabs'));
    /** @param {string} key */
    const press = (key) =>
      strip.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true }));
    press('ArrowLeft');
    expect(group.activePanel.id).toBe('iframe:c');
    press('ArrowRight');
    expect(group.activePanel.id).toBe('iframe:a');
    press('End');
    expect(group.activePanel.id).toBe('iframe:c');
    press('Home');
    expect(group.activePanel.id).toBe('iframe:a');
    press('Delete');
    expect(closeDockPanel).toHaveBeenCalledExactlyOnceWith('a');
  });

  test('an item dragstart records {tab: key} for the drop bridge until the gesture ends', async () => {
    const mod = await import(MOD);
    const group = fakeGroup([
      ['iframe:ariel', 'ARIEL'],
      ['iframe:okf', 'KNOWLEDGE'],
    ]);
    const { tab } = mountBar(mod, group.panels[0]);
    const seen = vi.fn();
    const off = mod.onDockDragStart(seen);
    /** @type {Record<string, string>} */
    const store = {};
    const dataTransfer = {
      setData: (/** @type {string} */ t, /** @type {string} */ v) => {
        store[t] = v;
      },
      effectAllowed: 'all',
    };
    const okfItem = /** @type {HTMLElement} */ (
      tab.element.querySelector('.tile-tab-item[data-panel-key="okf"]')
    );
    const start = new MouseEvent('dragstart', { bubbles: true, cancelable: true });
    Object.defineProperty(start, 'dataTransfer', { value: dataTransfer });
    okfItem.dispatchEvent(start);
    expect(mod.dockDragSource()).toEqual({ tab: 'okf' });
    expect(seen).toHaveBeenCalledExactlyOnceWith({ tab: 'okf' });
    expect(JSON.parse(store[mod.DOCK_SOURCE_MIME])).toEqual({ tab: 'okf' });
    expect(store['text/plain']).toBe('okf');
    expect(dataTransfer.effectAllowed).toBe('move');
    document.dispatchEvent(new MouseEvent('dragend', { bubbles: true }));
    expect(mod.dockDragSource()).toBeNull();
    off();
  });

  test('a right-click on an item opens the menu for that tab, elsewhere for the showing one', async () => {
    const mod = await import(MOD);
    /** @type {import('vitest').Mock<(id: string, opts: any) => boolean>} */
    const handler = vi.fn(() => true);
    mod.setTileContextMenuHandler(handler);
    const group = fakeGroup([
      ['iframe:ariel', 'ARIEL'],
      ['iframe:okf', 'KNOWLEDGE'],
    ]);
    const { tab } = mountBar(mod, group.panels[0]);
    const okfItem = /** @type {HTMLElement} */ (
      tab.element.querySelector('.tile-tab-item[data-panel-key="okf"]')
    );
    okfItem.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true }));
    expect(handler.mock.calls[0][0]).toBe('okf');
    /** @type {HTMLElement} */ (tab.element.querySelector('.tile-tab-contrib')).dispatchEvent(
      new MouseEvent('contextmenu', { bubbles: true, cancelable: true })
    );
    expect(handler.mock.calls[1][0]).toBe('ariel');
    mod.setTileContextMenuHandler(null);
  });

  test('measureTabRects keys the showing bars by tile: the items of a strip, the lone title, none for the Agent tile', async () => {
    const mod = await import(MOD);
    mod.setTerminalHeaderSource(detachedTerminalHeader());
    const two = fakeGroup([
      ['iframe:ariel', 'ARIEL'],
      ['iframe:okf', 'KNOWLEDGE'],
    ]);
    const { host } = mountBar(mod, two.panels[0]);
    // The hidden sibling's bar renders the same strip but is not on screen.
    mountBar(mod, two.panels[1], { inactive: true, header: host });
    const one = fakeGroup([['iframe:lattice', 'LATTICE']]);
    one.id = 't2';
    mountBar(mod, one.panels[0]);
    const agent = fakeGroup([['terminal', 'Agent']]);
    agent.id = 't3';
    mountBar(mod, agent.panels[0]);
    const rects = mod.measureTabRects(document);
    expect(Object.keys(rects).sort()).toEqual(['t1', 't2', 't3']);
    expect(rects.t1.tabRects).toHaveLength(2);
    expect(rects.t2.tabRects).toHaveLength(1);
    expect(rects.t3.tabRects).toHaveLength(0);
    for (const tile of Object.values(rects)) {
      expect(Object.keys(tile.barRect).sort()).toEqual(['height', 'width', 'x', 'y']);
    }
  });

  test('the Agent tile renders no strip, whatever its group holds', async () => {
    const mod = await import(MOD);
    mod.setTerminalHeaderSource(detachedTerminalHeader());
    const group = fakeGroup([
      ['terminal', 'Agent'],
      ['iframe:ariel', 'ARIEL'],
    ]);
    const { tab } = mountBar(mod, group.panels[0]);
    expect(tab.element.querySelector('.tile-tab-tabs')).toBeNull();
    expect(tab.element.dataset.tile).toBe('t1');
  });
});
