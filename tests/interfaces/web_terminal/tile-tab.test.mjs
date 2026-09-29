/**
 * Contract tests for the tile-tab renderer (dock-tab.js): the custom dockview
 * tab that IS each tile's header bar. Every tile gets the same bar — identity
 * on the left, actions right-anchored; the bar itself is the drag handle.
 * Service tiles carry a visible `.tile-tab-title`; the terminal tile adopts
 * the live .terminal-header instead. Run:
 *   npx vitest run tests/interfaces/web_terminal/tile-tab.test.mjs
 */
import { test, expect, describe, beforeEach, vi } from 'vitest';

const MOD = '../../../src/osprey/interfaces/web_terminal/static/js/dock-tab.js';

/** Minimal dockview panel api stub for tab init params. */
function fakeApi() {
  return {
    close: vi.fn(),
    onDidTitleChange: vi.fn(() => ({ dispose: vi.fn() })),
  };
}

/** The two bar kinds, which share one close control. */
const CLOSE_BARS = [['service', 'iframe:ariel'], ['terminal', 'terminal']];

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

describe('tile-tab renderer', () => {
  beforeEach(() => {
    vi.resetModules();
    document.body.innerHTML = '';
    vi.restoreAllMocks();
  });

  test('service tab renders visible title and close', async () => {
    const { createTileTab } = await import(MOD);
    const tab = createTileTab('iframe:ariel');
    tab.init({ title: 'ARIEL', params: {}, api: fakeApi() });

    expect(tab.element.classList.contains('tile-tab')).toBe(true);
    // One header grammar for every tile: name on the bar, close on the bar.
    expect(tab.element.querySelector('.tile-tab-title')?.textContent).toBe('ARIEL');
    expect(tab.element.querySelector('.tile-tab-close')).toBeTruthy();
    expect(tab.element.querySelector('.tile-tab-actions')).toBeTruthy();
  });

  test.each(CLOSE_BARS)('%s close click calls api.close()', async (_kind, id) => {
    const { createTileTab, setTerminalHeaderSource } = await import(MOD);
    setTerminalHeaderSource(detachedTerminalHeader());
    const api = fakeApi();
    const tab = createTileTab(id);
    tab.init({ title: 'ARIEL', params: {}, api });
    /** @type {HTMLElement} */ (tab.element.querySelector('.tile-tab-close'))
      .dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    expect(api.close).toHaveBeenCalledTimes(1);
  });

  test.each(CLOSE_BARS)('%s close pointerdown is prevented so it cannot start a tile drag', async (_kind, id) => {
    const { createTileTab, setTerminalHeaderSource } = await import(MOD);
    setTerminalHeaderSource(detachedTerminalHeader());
    const tab = createTileTab(id);
    tab.init({ title: 'ARIEL', params: {}, api: fakeApi() });
    const ev = new MouseEvent('pointerdown', { bubbles: true, cancelable: true });
    /** @type {HTMLElement} */ (tab.element.querySelector('.tile-tab-close'))
      .dispatchEvent(ev);
    expect(ev.defaultPrevented).toBe(true);
  });

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
      vi.fn((/** @type {any} */ fn) => { onTitle = fn; return { dispose: vi.fn() }; }));
    const tab = createTileTab('iframe:okf');
    tab.init({ title: 'KNOWLEDGE', params: {}, api });

    expect(tab.element.getAttribute('aria-label')).toBe('KNOWLEDGE');
    expect(tab.element.querySelector('.tile-tab-title')?.textContent).toBe('KNOWLEDGE');
    onTitle({ title: 'RENAMED' });
    expect(tab.element.getAttribute('aria-label')).toBe('RENAMED');
    expect(tab.element.querySelector('.tile-tab-title')?.textContent).toBe('RENAMED');
  });

  test('the terminal bar takes no aria-label — its adopted header names it', async () => {
    const { createTileTab } = await import(MOD);
    const api = fakeApi();
    const tab = createTileTab('terminal');
    tab.init({ title: 'SESSION', params: {}, api });
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

    /** @type {HTMLElement} */ (tab.element.querySelector('.tile-tab-title'))
      .dispatchEvent(new MouseEvent('pointerdown', { bubbles: true, cancelable: true }));
    expect(reachedRoot).toHaveBeenCalledTimes(1);
  });

  describe('terminal tab', () => {
    /** @type {HTMLElement} */
    let header;

    beforeEach(() => {
      header = detachedTerminalHeader();
    });

    test('adopts .terminal-header and renders close, but no title', async () => {
      const { createTileTab, setTerminalHeaderSource } = await import(MOD);
      setTerminalHeaderSource(header);
      const tab = createTileTab('terminal');
      tab.init({ title: 'SESSION', params: {}, api: fakeApi() });
      document.body.appendChild(tab.element);
      expect(tab.element.classList.contains('tile-tab-terminal')).toBe(true);
      // The page's ONE header node moved into the tab (relocation, not clone).
      expect(tab.element.querySelector('.terminal-header')).toBeTruthy();
      expect(document.querySelectorAll('.terminal-header').length).toBe(1);
      expect(tab.element.querySelector('.tile-tab-title')).toBeNull();
      // The terminal's rail entry has no "×", so this is its only pointer
      // close path — the one action that stayed on a tile.
      expect(tab.element.querySelector('.tile-tab-close')).toBeTruthy();
    });

    test('re-adoption after detach (close → reopen) reuses the cached header node', async () => {
      const { createTileTab, setTerminalHeaderSource } = await import(MOD);
      setTerminalHeaderSource(header);
      const first = createTileTab('terminal');
      first.init({ title: 'SESSION', params: {}, api: fakeApi() });
      const headerNode = first.element.querySelector('.terminal-header');
      first.element.remove(); // dockview discards the tab DOM on removePanel
      const second = createTileTab('terminal');
      second.init({ title: 'SESSION', params: {}, api: fakeApi() });
      expect(second.element.querySelector('.terminal-header')).toBe(headerNode);
    });

    test('pointerdown on interactive header children stops propagation (no drag), on plain header it bubbles (drag ok)', async () => {
      const { createTileTab, setTerminalHeaderSource } = await import(MOD);
      setTerminalHeaderSource(header);
      const tab = createTileTab('terminal');
      tab.init({ title: 'SESSION', params: {}, api: fakeApi() });
      document.body.appendChild(tab.element);
      const reachedRoot = vi.fn();
      tab.element.addEventListener('pointerdown', reachedRoot);

      tab.element.querySelector('#session-selector')
        .dispatchEvent(new MouseEvent('pointerdown', { bubbles: true, cancelable: true }));
      expect(reachedRoot).not.toHaveBeenCalled(); // contained — no tile drag

      tab.element.querySelector('.terminal-header')
        .dispatchEvent(new MouseEvent('pointerdown', { bubbles: true, cancelable: true }));
      expect(reachedRoot).toHaveBeenCalledTimes(1); // plain surface drags
    });

  });
});

/**
 * The tile header is a panel's second right-click surface: the same verb menu
 * its rail entry offers. dock-tab holds no policy — it forwards the press
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
      bubbles: true, cancelable: true, clientX: x, clientY: y,
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
    // speaks the same panel ids the rail does.
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
    /** @type {HTMLElement} */ (tab.element.querySelector('.tile-tab-close')).dispatchEvent(onClose);
    expect(handler).not.toHaveBeenCalled();
    expect(onClose.defaultPrevented).toBe(false);

    // Plain bar surface still opens the menu.
    /** @type {HTMLElement} */ (tab.element.querySelector('.tile-tab-title'))
      .dispatchEvent(rightClick());
    expect(handler).toHaveBeenCalledTimes(1);

    // The terminal bar's adopted header applies the same rule to its selector.
    const header = detachedTerminalHeader();
    setTerminalHeaderSource(header);
    const terminal = createTileTab('terminal');
    terminal.init({ title: 'SESSION', params: {}, api: fakeApi() });
    document.body.appendChild(terminal.element);
    const onSelector = rightClick();
    /** @type {HTMLElement} */ (header.querySelector('#session-selector')).dispatchEvent(onSelector);
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
    tab.init({ title: 'SESSION', params: {}, api: fakeApi() });
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
    first.init({ title: 'SESSION', params: {}, api: fakeApi() });
    first.element.remove();
    const second = createTileTab('terminal');
    second.init({ title: 'SESSION', params: {}, api: fakeApi() });
    document.body.appendChild(second.element);

    /** @type {HTMLElement} */ (second.element.querySelector('.terminal-header'))
      .dispatchEvent(rightClick());
    expect(handler).toHaveBeenCalledTimes(1);
  });

});
