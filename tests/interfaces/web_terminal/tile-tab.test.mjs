/**
 * Contract tests for the tile-tab renderer (dock-tab.js): the custom dockview
 * tab that IS each tile's header bar. Service tiles render a bare drag grip —
 * their name is on the rail entry and their close/popout controls are that
 * entry's hover corners. The terminal tile keeps a populated bar: grip +
 * adopted .terminal-header + close. Run:
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

describe('tile-tab renderer', () => {
  beforeEach(() => {
    vi.resetModules();
    document.body.innerHTML = '';
    vi.restoreAllMocks();
  });

  test('service tab renders the grip and nothing else', async () => {
    const { createTileTab } = await import(MOD);
    const tab = createTileTab('iframe:ariel');
    tab.init({ title: 'ARIEL', params: {}, api: fakeApi() });

    expect(tab.element.classList.contains('tile-tab')).toBe(true);
    expect(tab.element.querySelector('.tile-tab-grip')).toBeTruthy();
    // The affordances that used to live here now belong to the rail entry.
    expect(tab.element.querySelector('.tile-tab-title')).toBeNull();
    expect(tab.element.querySelector('.tile-tab-popout')).toBeNull();
    expect(tab.element.querySelector('.tile-tab-close')).toBeNull();
    expect(tab.element.querySelector('.tile-tab-actions')).toBeNull();
    expect(tab.element.textContent).toBe(''); // no visible text at all
  });

  test('the strip carries the dockview panel id as a stable handle', async () => {
    const { createTileTab } = await import(MOD);
    const tab = createTileTab('iframe:ariel');
    tab.init({ title: 'ARIEL', params: {}, api: fakeApi() });
    expect(tab.element.dataset.panelId).toBe('iframe:ariel');
  });

  test('the title becomes the strip\'s accessible name, not visible text', async () => {
    const { createTileTab } = await import(MOD);
    const api = fakeApi();
    /** @type {(e: {title: string}) => void} */ let onTitle = () => {};
    api.onDidTitleChange = vi.fn((/** @type {any} */ fn) => { onTitle = fn; return { dispose: vi.fn() }; });
    const tab = createTileTab('iframe:okf');
    tab.init({ title: 'KNOWLEDGE', params: {}, api });

    expect(tab.element.getAttribute('aria-label')).toBe('KNOWLEDGE');
    expect(tab.element.textContent).toBe('');
    onTitle({ title: 'RENAMED' });
    expect(tab.element.getAttribute('aria-label')).toBe('RENAMED');
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

    /** @type {HTMLElement} */ (tab.element.querySelector('.tile-tab-grip'))
      .dispatchEvent(new MouseEvent('pointerdown', { bubbles: true, cancelable: true }));
    expect(reachedRoot).toHaveBeenCalledTimes(1);
  });

  describe('terminal tab', () => {
    /** @type {HTMLElement} */
    let header;

    beforeEach(() => {
      // Build the terminal card DETACHED from the document — this reproduces
      // the real production timing, not a convenient shortcut: dock-workspace's
      // adoptSubtree moves the whole .terminal-panel subtree into an unattached
      // host div before dockview reinserts it, so by the time the terminal's
      // tab is built the header is NOT reachable via a document-wide query —
      // only via the reference each test registers through
      // setTerminalHeaderSource below. (dock-tab.js has no document.querySelector
      // fallback precisely because that lookup always loses this race.)
      const card = document.createElement('div');
      card.className = 'terminal-card';
      header = document.createElement('div');
      header.className = 'terminal-header';
      const sel = document.createElement('select');
      sel.id = 'session-selector';
      header.appendChild(sel);
      card.appendChild(header);
      // card is intentionally never attached to document.body.
    });

    test('adopts .terminal-header and renders close, but no popout or title', async () => {
      const { createTileTab, setTerminalHeaderSource } = await import(MOD);
      setTerminalHeaderSource(header);
      const tab = createTileTab('terminal');
      tab.init({ title: 'SESSION', params: {}, api: fakeApi() });
      document.body.appendChild(tab.element);
      expect(tab.element.classList.contains('tile-tab-terminal')).toBe(true);
      // The page's ONE header node moved into the tab (relocation, not clone).
      expect(tab.element.querySelector('.terminal-header')).toBeTruthy();
      expect(document.querySelectorAll('.terminal-header').length).toBe(1);
      expect(tab.element.querySelector('.tile-tab-popout')).toBeNull();
      expect(tab.element.querySelector('.tile-tab-title')).toBeNull();
      // The terminal's rail entry has no "×", so this is its only pointer
      // close path — the one action that stayed on a tile.
      expect(tab.element.querySelector('.tile-tab-close')).toBeTruthy();
    });

    test('close click calls api.close()', async () => {
      const { createTileTab, setTerminalHeaderSource } = await import(MOD);
      setTerminalHeaderSource(header);
      const api = fakeApi();
      const tab = createTileTab('terminal');
      tab.init({ title: 'SESSION', params: {}, api });
      /** @type {HTMLElement} */ (tab.element.querySelector('.tile-tab-close'))
        .dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
      expect(api.close).toHaveBeenCalledTimes(1);
    });

    test('close pointerdown is prevented so it cannot start a tile drag', async () => {
      const { createTileTab, setTerminalHeaderSource } = await import(MOD);
      setTerminalHeaderSource(header);
      const tab = createTileTab('terminal');
      tab.init({ title: 'SESSION', params: {}, api: fakeApi() });
      const ev = new MouseEvent('pointerdown', { bubbles: true, cancelable: true });
      /** @type {HTMLElement} */ (tab.element.querySelector('.tile-tab-close'))
        .dispatchEvent(ev);
      expect(ev.defaultPrevented).toBe(true);
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

    test('with no setTerminalHeaderSource registration, the tab renders without a header (documents the missing-wiring failure mode)', async () => {
      const { createTileTab } = await import(MOD);
      const tab = createTileTab('terminal');
      tab.init({ title: 'SESSION', params: {}, api: fakeApi() });
      expect(tab.element.classList.contains('tile-tab-terminal')).toBe(true);
      expect(tab.element.querySelector('.terminal-header')).toBeNull();
    });
  });
});
