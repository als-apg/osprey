// @vitest-environment jsdom
/**
 * Contract tests for the tile-tab renderer (dock-tab.js): the custom dockview
 * tab that IS each tile's header bar. A tile holds ONE panel, so every tile
 * gets one bar — identity on the left, actions right-anchored, no tab strip;
 * the bar itself is the drag handle. Service tiles carry a visible
 * `.tile-tab-title`, the panel's contribution region, a Pop out button and a
 * close; the Agent tile adopts a live bar instead — the chat console's
 * `.op-session-bar` in chat, the `.terminal-header` on the terminal surface.
 * The last suite builds the bars on the pinned real dockview (see
 * _dock-real.mjs), as dock-workspace.js wires them. Run:
 *   npx vitest run tests/interfaces/web_terminal/dock-tab.test.mjs
 */
import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import { buildConsole } from '../../../src/osprey/interfaces/web_terminal/static/js/chat-console.js';
import { loadDockview, settle } from './_dock-real.mjs';

const JS = '../../../src/osprey/interfaces/web_terminal/static/js';
const MOD = `${JS}/dock-tab.js`;

// jsdom has no ResizeObserver; the bar's contribution region observes its
// size, and dockview observes its grid. The harness installs an inert one.
loadDockview();

const { closeTerminalPanel, closeTerminalTile, closeDockPanel } = vi.hoisted(() => ({
  closeTerminalPanel: vi.fn(() => true),
  closeTerminalTile: vi.fn(),
  closeDockPanel: vi.fn(() => true),
}));
const { detach, detachBlocker } = vi.hoisted(() => ({
  detach: vi.fn(() => true),
  detachBlocker: vi.fn(() => /** @type {string | null} */ (null)),
}));

/** What the gate says about an opaque-origin panel. */
const OPAQUE = 'This panel runs at an opaque origin and cannot be popped out.';

// The window itself is pane-window's contract; the bar only has to ask it the
// right things at the right times.
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/pane-window.js', () => ({
  detach,
  detachBlocker,
}));

// dock-tab closes every tile through dock-workspace (the layout holder); the
// rest of that module (dockview itself) is out of scope here. The doubles
// cover what the modules dock-tab imports read from it.
// The Agent close goes through the Undo-offering path the Delete key uses.
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/panel-menu-policy.js', () => ({
  closeTerminalTile,
}));

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
  // The session switcher's label, as attachSessionSwitcher marks it: a
  // keyboard-reachable button that opens the switcher popover.
  const label = document.createElement('span');
  label.className = 'terminal-label';
  label.id = 'terminal-label';
  label.setAttribute('role', 'button');
  label.setAttribute('aria-label', 'Switch session');
  header.appendChild(label);
  card.appendChild(header);
  return header;
}

/**
 * The chat console's Agent bar as the page has it before the dock adopts it:
 * built by chat-console.js, with the bell the feature modules add,
 * and its label marked as the session switcher's button the way
 * attachSessionSwitcher marks it, inside a detached console container.
 * @returns {HTMLElement}
 */
function detachedConsoleBar() {
  const { bar, label: switcher } = buildConsole();
  switcher.setAttribute('role', 'button');
  switcher.setAttribute('aria-label', 'Switch session');
  for (const [cls, label] of [['op-attention-toggle', 'Desktop notifications']]) {
    const btn = document.createElement('button');
    btn.className = `osprey-icon-btn ${cls}`;
    btn.setAttribute('aria-label', label);
    bar.append(btn);
  }
  document.createElement('div').append(bar);
  return bar;
}

describe('tile-tab renderer', () => {
  beforeEach(() => {
    vi.resetModules();
    document.body.innerHTML = '';
    vi.restoreAllMocks();
    closeTerminalPanel.mockClear();
    closeTerminalTile.mockClear();
    closeDockPanel.mockClear();
    detach.mockClear();
    detachBlocker.mockReset().mockReturnValue(null);
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

  test('the Agent close runs closeTerminalTile (the Undo path), never the bare api.close()', async () => {
    const { createTileTab, setTerminalHeaderSource } = await import(MOD);
    setTerminalHeaderSource(detachedTerminalHeader());
    const api = fakeApi();
    const tab = createTileTab('terminal');
    tab.init({ title: 'Agent', params: {}, api });
    const close = /** @type {HTMLElement} */ (tab.element.querySelector('.tile-tab-close'));
    expect(close.getAttribute('aria-label')).toBe('Close Agent pane');
    close.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    expect(closeTerminalTile).toHaveBeenCalledTimes(1);
    expect(closeTerminalPanel).not.toHaveBeenCalled();
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

    expect(detach).toHaveBeenCalledExactlyOnceWith('ariel');
    expect(api.close).not.toHaveBeenCalled();
  });

  test('the Agent tile has the same Pop out, before its close, popping the Agent pane out', async () => {
    const { createTileTab, setTerminalHeaderSource } = await import(MOD);
    setTerminalHeaderSource(detachedTerminalHeader());
    const tab = createTileTab('terminal');
    tab.init({ title: 'Agent', params: {}, api: fakeApi() });

    const actions = /** @type {HTMLElement} */ (tab.element.querySelector('.tile-tab-actions'));
    expect([...actions.children].map((el) => el.getAttribute('aria-label'))).toEqual([
      'Pop out',
      'Close Agent pane',
    ]);
    /** @type {HTMLElement} */ (actions.querySelector('.tile-tab-popout')).dispatchEvent(
      new MouseEvent('click', { bubbles: true, cancelable: true })
    );
    expect(detach).toHaveBeenCalledExactlyOnceWith('terminal');
    expect(closeTerminalPanel).not.toHaveBeenCalled();
  });

  test('a panel that cannot pop out is disabled, with the reason, once the pointer reads the gate', async () => {
    detachBlocker.mockReturnValue(OPAQUE);
    const { createTileTab } = await import(MOD);
    const tab = createTileTab('iframe:notes');
    tab.init({ title: 'NOTES', params: {}, api: fakeApi() });
    const popout = /** @type {HTMLButtonElement} */ (tab.element.querySelector('.tile-tab-popout'));

    // Built enabled: the URL and frame kind are not known yet, and a button
    // disabled now would never get the pointer event that re-reads the gate.
    expect(popout.disabled).toBe(false);
    popout.dispatchEvent(new Event('pointerenter'));
    expect(popout.disabled).toBe(true);
    expect(popout.title).toBe(OPAQUE);
    popout.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    expect(detach).not.toHaveBeenCalled();
  });

  test('the gate is read again at click time: a frame that resolved opaque after the bar was built', async () => {
    const { createTileTab } = await import(MOD);
    const tab = createTileTab('iframe:notes');
    tab.init({ title: 'NOTES', params: {}, api: fakeApi() });
    const popout = /** @type {HTMLButtonElement} */ (tab.element.querySelector('.tile-tab-popout'));
    expect(popout.disabled).toBe(false);

    detachBlocker.mockReturnValue(OPAQUE);
    popout.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));

    expect(detach).not.toHaveBeenCalled();
    expect(popout.disabled).toBe(true);
    expect(popout.title).toBe(OPAQUE);
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

  test('the bar carries the dockview panel id as a stable handle', async () => {
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

  test('the whole service bar is the drag handle — nothing swallows pointerdown', async () => {
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

    test('the chat adopts the console bar: dot, session switcher, pen, bell, share, close', async () => {
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
        'Switch session',
        'New conversation',
        'Desktop notifications',
        'Pop out',
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
        .querySelector('#terminal-label')
        .dispatchEvent(new MouseEvent('pointerdown', { bubbles: true, cancelable: true }));
      expect(reachedRoot).not.toHaveBeenCalled(); // contained — no tile drag

      tab.element
        .querySelector('.terminal-header')
        .dispatchEvent(new MouseEvent('pointerdown', { bubbles: true, cancelable: true }));
      expect(reachedRoot).toHaveBeenCalledTimes(1); // plain surface drags
    });

    test("the console bar's buttons and session switcher act instead of dragging; its plain surface drags", async () => {
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
      expect(reachedRoot).not.toHaveBeenCalled(); // the switcher label is a button
      bar.dispatchEvent(new MouseEvent('pointerdown', { bubbles: true, cancelable: true }));
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

    // The terminal bar's adopted header applies the same rule to its session switcher.
    const header = detachedTerminalHeader();
    setTerminalHeaderSource(header);
    const terminal = createTileTab('terminal');
    terminal.init({ title: 'Agent', params: {}, api: fakeApi() });
    document.body.appendChild(terminal.element);
    const onSelector = rightClick();
    /** @type {HTMLElement} */ (header.querySelector('#terminal-label')).dispatchEvent(onSelector);
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

    // The bar's plain surface — the switcher label is a button and keeps its own click.
    bar.dispatchEvent(rightClick());
    expect(handler).toHaveBeenCalledTimes(1);
    const { id, opts } = firstCall(handler);
    expect(id).toBe('terminal');
    expect(opts.anchorEl).toBe(bar);
    document.documentElement.removeAttribute('data-agent-surface');
  });
});

/**
 * The bars on the pinned real dockview, built through createTabComponent with
 * the layout holder applying the tiles, as dock-workspace.js wires them: a
 * service tile and the Agent pane's, each holding its one panel.
 */
describe('the bars on a real dock', () => {
  const LAYOUT = {
    split: 'row',
    items: [
      { tile: 't1', tabs: ['ariel'], showing: 'ariel', size: 0.6 },
      { tile: 't2', tabs: ['terminal'], showing: 'terminal', size: 0.4 },
    ],
  };

  /** @type {Array<() => void>} */
  const teardowns = [];

  beforeEach(() => {
    vi.resetModules();
    document.body.innerHTML = '';
    // dockview's drag source reads PointerEvent, which jsdom lacks.
    vi.stubGlobal('PointerEvent', class extends MouseEvent {});
  });

  afterEach(() => {
    for (const teardown of teardowns.splice(0)) teardown();
    vi.unstubAllGlobals();
  });

  /** Mount a dock with the bar renderer, its tiles applied by a layout holder. */
  async function mountBars() {
    const mod = await import(MOD);
    const { createLayoutHolder } = await import(`${JS}/dock-apply.js`);
    mod.setTerminalHeaderSource(detachedTerminalHeader());
    const root = document.createElement('div');
    document.body.appendChild(root);
    const api = loadDockview().createDockview(root, {
      disableFloatingGroups: true,
      createComponent: () => ({ element: document.createElement('div'), init() {} }),
      createTabComponent: (/** @type {{ id: string }} */ o) => mod.createTileTab(o.id),
      defaultTabComponent: mod.OSPREY_TAB_COMPONENT,
      dndStrategy: 'html5',
    });
    api.layout(1000, 600);
    teardowns.push(() => api.dispose());
    const holder = createLayoutHolder({ api, dockPx: { width: 1000, height: 600 } });
    holder.setPanelRecord('ariel', { title: 'ARIEL' });
    holder.restore(LAYOUT);
    await settle();
    /** @param {string} tile @returns {HTMLElement} */
    const barOf = (tile) => {
      const bar = root.querySelector(`.tile-tab[data-tile="${tile}"]`);
      if (!(bar instanceof HTMLElement)) throw new Error(`no bar for tile ${tile}`);
      return bar;
    };
    return { mod, api, root, holder, barOf };
  }

  test('a service bar: its title, contribution region, Pop out and Close tile — no strip', async () => {
    const { root, barOf } = await mountBars();
    expect(root.querySelectorAll('.tile-tab')).toHaveLength(2);
    const bar = barOf('t1');
    expect(bar.getAttribute('data-panel-key')).toBe('ariel');
    expect(bar.dataset.panelId).toBe('iframe:ariel');
    expect([...bar.children].map((el) => el.classList[0])).toEqual([
      'tile-tab-title',
      'tile-tab-contrib',
      'tile-tab-actions',
    ]);
    expect(bar.querySelector('.tile-tab-title')?.textContent).toBe('Ariel');
    const actions = /** @type {HTMLElement} */ (bar.querySelector('.tile-tab-actions'));
    expect([...actions.children].map((el) => el.getAttribute('aria-label'))).toEqual([
      'Pop out',
      'Close tile',
    ]);
    expect(bar.querySelector('.tile-tab-tabs, .tile-tab-item')).toBeNull();
    // The Agent pane's bar carries its tile and key too.
    expect(barOf('t2').getAttribute('data-panel-key')).toBe('terminal');
  });

  test('measureTabRects measures each tile’s header row, with no tab rects — the Agent tile’s too', async () => {
    const { mod, api, root } = await mountBars();
    // jsdom lays nothing out: give each header row the rect a browser would.
    for (const group of api.groups) {
      const header = group.element.querySelector('.dv-tabs-and-actions-container');
      const x = group.id === 't1' ? 0 : 600;
      header.getBoundingClientRect = () => new DOMRect(x, 0, group.id === 't1' ? 600 : 400, 32);
    }
    expect(mod.measureTabRects(root)).toEqual({
      t1: { barRect: { x: 0, y: 0, width: 600, height: 32 }, tabRects: [] },
      t2: { barRect: { x: 600, y: 0, width: 400, height: 32 }, tabRects: [] },
    });
  });

  test('a right-click on a service bar opens the menu for its panel', async () => {
    const { mod, barOf } = await mountBars();
    const handler = vi.fn(() => true);
    mod.setTileContextMenuHandler(handler);
    const press = new MouseEvent('contextmenu', {
      bubbles: true,
      cancelable: true,
      clientX: 40,
      clientY: 12,
    });
    /** @type {HTMLElement} */ (barOf('t1').querySelector('.tile-tab-title')).dispatchEvent(press);
    expect(handler).toHaveBeenCalledExactlyOnceWith('ariel', {
      x: 40,
      y: 12,
      anchorEl: barOf('t1'),
      mode: 'tile',
    });
    expect(press.defaultPrevented).toBe(true);
    mod.setTileContextMenuHandler(null);
  });

  test('a drag of the bar is a drag of its tile for the drop bridge', async () => {
    const { mod, api, root, holder, barOf } = await mountBars();
    const { initDockDrop } = await import(`${JS}/dock-drop.js`);
    teardowns.unshift(initDockDrop(api, root, { holder: () => holder }));
    const start = new MouseEvent('dragstart', { bubbles: true, cancelable: true });
    Object.defineProperty(start, 'dataTransfer', {
      value: { setData: () => {}, setDragImage: () => {}, effectAllowed: 'all', items: [] },
    });
    /** @type {HTMLElement} */ (barOf('t1').querySelector('.tile-tab-title')).dispatchEvent(start);
    expect(mod.dockDragSource()).toEqual({ tile: 't1' });
    document.dispatchEvent(new MouseEvent('dragend', { bubbles: true }));
    expect(mod.dockDragSource()).toBeNull();
  });

  test('the title follows a rename of the panel', async () => {
    const { api, barOf } = await mountBars();
    api.getPanel('iframe:ariel').api.setTitle('KNOWLEDGE');
    expect(barOf('t1').querySelector('.tile-tab-title')?.textContent).toBe('Knowledge');
    expect(barOf('t1').getAttribute('aria-label')).toBe('Knowledge');
  });
});

/**
 * The contract S3's renderer keeps: a bar in the pane's own window is this same
 * TileTab in `mode: 'detached'`, with Reattach far right, a name, and no verb that
 * acts on the dock.
 */
describe('the bar of a pane in its own window', () => {
  const REATTACH = { mode: 'detached' };

  beforeEach(() => {
    vi.resetModules();
    document.body.innerHTML = '';
  });

  test('Reattach is the far-right slot, alone: no open-in-window button and no ×', async () => {
    const { createTileTab } = await import(MOD);
    const onReattach = vi.fn();
    const tab = createTileTab('ariel', { mode: 'detached', onReattach });
    tab.init({ title: 'ARIEL' });
    const actions = tab.element.querySelector('.tile-tab-actions');
    expect(tab.element.lastElementChild).toBe(actions);
    expect(actions.children).toHaveLength(1);
    const reattach = actions.lastElementChild;
    expect(reattach.classList.contains('tile-tab-reattach')).toBe(true);
    expect(reattach.textContent).toBe('Reattach');
    expect(reattach.title).toBe('Put Ariel back in the workspace');
    expect(tab.element.querySelector('.tile-tab-popout')).toBeNull();
    expect(tab.element.querySelector('.tile-tab-close')).toBeNull();
    reattach.click();
    expect(onReattach).toHaveBeenCalledTimes(1);
  });

  test('it has the pane’s name, and no tile', async () => {
    const { createTileTab } = await import(MOD);
    const tab = createTileTab('ariel', REATTACH);
    tab.init({ title: 'ARIEL' });
    expect(tab.element.querySelector('.tile-tab-title')?.textContent).toBe('Ariel');
    expect(tab.element.getAttribute('aria-label')).toBe('Ariel');
    expect(tab.element.dataset.tile).toBeUndefined();
  });

  test('a docked bar keeps its two buttons and no Reattach', async () => {
    const { createTileTab } = await import(MOD);
    const tab = createTileTab('iframe:ariel');
    tab.init({ title: 'ARIEL', params: {}, api: fakeApi() });
    expect(tab.element.querySelector('.tile-tab-reattach')).toBeNull();
    expect(tab.element.querySelector('.tile-tab-popout')).not.toBeNull();
    expect(tab.element.querySelector('.tile-tab-close')).not.toBeNull();
  });

  test('the contribution host registers from the window, at attached(), not at build', async () => {
    const { createTileTab } = await import(MOD);
    const contrib = await import(`${JS}/tile-header-contrib.js`);
    const dispatch = vi.fn();
    contrib.contributeLocal('ariel', [{ kind: 'text', id: 'a', text: 'hello' }], dispatch);
    const tab = createTileTab('ariel', REATTACH);
    tab.init({ title: 'ARIEL' });
    const host = tab.element.querySelector('.tile-tab-contrib');
    expect(host.children).toHaveLength(0);
    document.body.append(tab.element);
    tab.attached();
    expect(host.textContent).toContain('hello');
    tab.dispose();
    const next = createTileTab('ariel', REATTACH);
    next.init({ title: 'ARIEL' });
    document.body.append(next.element);
    next.attached();
    expect(next.element.querySelector('.tile-tab-contrib').textContent).toContain('hello');
    next.dispose();
  });

  test('a right-click asks for the menu with the mode at that moment, Agent bar included', async () => {
    const { createTileTab, setTerminalConsoleBarSource, setTileContextMenuHandler } = await import(
      MOD
    );
    const { initLayoutHolder } = await import(`${JS}/dock-apply.js`);
    const holder = initLayoutHolder({ dockPx: { width: 1200, height: 800 } });
    holder.restore({
      split: 'row',
      items: [
        { tile: 't1', tabs: ['ariel'], showing: 'ariel', size: 0.5 },
        { tile: 't2', tabs: ['terminal'], showing: 'terminal', size: 0.5 },
      ],
    });
    /** @type {any} */
    const handler = vi.fn(() => true);
    setTileContextMenuHandler(handler);
    const bar = detachedConsoleBar();
    setTerminalConsoleBarSource(bar);
    const led = /** @type {HTMLElement} */ (bar.querySelector('.op-session-led'));
    const docked = createTileTab('terminal');
    document.body.append(docked.element);
    led.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true }));
    expect(handler.mock.lastCall?.[1].mode).toBe('tile');

    // The same node, wired once, in the window.
    holder.detach('terminal', null);
    docked.dispose();
    const tab = createTileTab('terminal', REATTACH);
    document.body.append(tab.element);
    expect(tab.element.contains(bar)).toBe(true);
    led.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true }));
    expect(handler.mock.lastCall?.[0]).toBe('terminal');
    expect(handler.mock.lastCall?.[1].mode).toBe('detached');
    tab.dispose();
  });

  test('disposed, it sends the Agent bar home, intact', async () => {
    const { createTileTab, setTerminalConsoleBarSource } = await import(MOD);
    const bar = detachedConsoleBar();
    const home = bar.parentElement;
    const first = home?.firstElementChild;
    expect(first).toBe(bar);
    setTerminalConsoleBarSource(bar);
    home?.append(document.createElement('main'));
    const tab = createTileTab('terminal', REATTACH);
    expect(tab.element.contains(bar)).toBe(true);
    expect(bar.querySelector('.op-session-new')).not.toBeNull();
    tab.dispose();
    expect(bar.parentElement).toBe(home);
    expect(home?.firstElementChild).toBe(bar);
    expect(bar.querySelector('.op-session-new')).not.toBeNull();
  });

  test('a tile that adopted the Agent bar first keeps it when the window’s bar goes', async () => {
    const { createTileTab, setTerminalConsoleBarSource } = await import(MOD);
    const bar = detachedConsoleBar();
    setTerminalConsoleBarSource(bar);
    const inWindow = createTileTab('terminal', REATTACH);
    const docked = createTileTab('terminal');
    expect(docked.element.contains(bar)).toBe(true);
    inWindow.dispose();
    expect(docked.element.contains(bar)).toBe(true);
  });

  test('a tile bar disposed in the dock leaves the Agent bar for its reopen, as it did', async () => {
    const { createTileTab, setTerminalConsoleBarSource } = await import(MOD);
    const bar = detachedConsoleBar();
    setTerminalConsoleBarSource(bar);
    const docked = createTileTab('terminal');
    docked.dispose();
    expect(docked.element.contains(bar)).toBe(true);
  });
});

describe('a window bar going away takes what it opened', () => {
  beforeEach(() => {
    vi.resetModules();
    document.body.innerHTML = '';
  });

  test('an open popover and an open context menu close', async () => {
    const { createTileTab } = await import(MOD);
    const { openPopover } = await import('/design-system/js/popover.js');
    const { openContextMenu } = await import('/design-system/js/context-menu.js');
    const tab = createTileTab('ariel', { mode: 'detached' });
    tab.init({ title: 'ARIEL' });
    document.body.append(tab.element);
    tab.attached();
    const anchor = /** @type {HTMLElement} */ (tab.element.querySelector('.tile-tab-reattach'));
    const pop = document.createElement('div');
    pop.className = 'osprey-popover';
    pop.hidden = true;
    document.body.append(pop);
    openPopover(pop, anchor);
    openContextMenu({
      x: 1,
      y: 1,
      anchorEl: anchor,
      ariaLabel: 'Actions',
      items: [{ label: 'A' }],
    });
    expect(pop.hidden).toBe(false);
    expect(document.querySelector('.osprey-context-menu')).not.toBeNull();
    tab.dispose();
    expect(pop.hidden).toBe(true);
    expect(document.querySelector('.osprey-context-menu')).toBeNull();
  });
});
