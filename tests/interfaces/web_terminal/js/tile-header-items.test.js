/**
 * Unit tests for the extended tile-bar item vocabulary — `search` and `menu`.
 *
 * Pure DOM/logic guard, happy-dom environment (configured globally):
 *   npx vitest run tests/interfaces/web_terminal/js/tile-header-items.test.js
 *
 * Round one's kinds (text/nav/button) and the receive pipeline itself are
 * covered by tile-header-contrib.test.js; this file drives the same public
 * entry points but exercises what tile-header-items.js added: the magnifier
 * pill, the body-level overflow popover, and — the reason the render pass
 * became a reconcile — a contributed input surviving its panel re-sending the
 * whole contribution underneath it.
 */

import { test, expect, describe, beforeAll, afterEach, vi } from 'vitest';

// happy-dom provides no ResizeObserver; the module observes contribution
// hosts for overflow re-measuring. Stub before import.
beforeAll(() => {
  if (typeof globalThis.ResizeObserver === 'undefined') {
    globalThis.ResizeObserver = class {
      observe() {}
      unobserve() {}
      disconnect() {}
    };
  }
});

import {
  initHeaderContrib,
  registerContribHost,
  unregisterContribHost,
} from '../../../../src/osprey/interfaces/web_terminal/static/js/tile-header-contrib.js';

/** Monotonic id source so each test gets an isolated panel id. */
let seq = 0;
/** @returns {string} */
function freshPanelId() {
  seq += 1;
  return `items-panel-${seq}`;
}

/**
 * Create a fake service-panel iframe whose contentWindow identity the module
 * matches messages against, with a spyable postMessage for the round-trip.
 * @param {string} panelId
 */
function addPanelIframe(panelId) {
  const iframe = document.createElement('iframe');
  iframe.className = 'panel-iframe';
  iframe.dataset.panelId = panelId;
  const contentWindow = { postMessage: vi.fn() };
  Object.defineProperty(iframe, 'contentWindow', { value: contentWindow });
  document.body.appendChild(iframe);
  return { iframe, contentWindow };
}

/**
 * Deliver a contribution as if it came from the given source window.
 * @param {any} items
 * @param {any} source
 */
function deliverContribution(items, source) {
  window.dispatchEvent(
    new MessageEvent('message', {
      data: { type: 'osprey-header-contribution', version: 1, items },
      origin: window.location.origin,
      source,
    })
  );
}

/**
 * Register a host attached to the document — the menu popover positions off a
 * real button, and focus assertions need an attached element.
 * @param {string} panelId
 * @returns {HTMLElement}
 */
function attachedHost(panelId) {
  const host = document.createElement('div');
  document.body.appendChild(host);
  registerContribHost(panelId, host);
  return host;
}

initHeaderContrib();

afterEach(() => {
  document.body.replaceChildren();
});

describe('search items', () => {
  test('render as a magnifier pill seeded with placeholder and value', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const host = attachedHost(panelId);

    deliverContribution(
      [{ kind: 'search', id: 'filter', placeholder: 'Filter plans…', value: 'orm' }],
      contentWindow
    );

    const input = /** @type {HTMLInputElement} */ (host.querySelector('.contrib-search-input'));
    expect(input).toBeTruthy();
    expect(input.placeholder).toBe('Filter plans…');
    expect(input.value).toBe('orm');
    // The magnifier is an inline SVG on currentColor — not the old emoji,
    // which rendered full-color on macOS/Windows.
    expect(host.querySelector('.contrib-search-icon .bar-icon-search')).toBeTruthy();
  });

  test('report their text back to the panel, debounced to the trailing edge', () => {
    vi.useFakeTimers();
    try {
      const panelId = freshPanelId();
      const { contentWindow } = addPanelIframe(panelId);
      const host = attachedHost(panelId);
      deliverContribution([{ kind: 'search', id: 'filter' }], contentWindow);

      const input = /** @type {HTMLInputElement} */ (host.querySelector('.contrib-search-input'));
      input.value = 'gr';
      input.dispatchEvent(new Event('input'));
      input.value = 'grid';
      input.dispatchEvent(new Event('input'));

      expect(contentWindow.postMessage).not.toHaveBeenCalled();
      vi.advanceTimersByTime(250);

      expect(contentWindow.postMessage).toHaveBeenCalledTimes(1);
      expect(contentWindow.postMessage.mock.calls[0][0]).toEqual({
        type: 'osprey-header-action',
        id: 'filter',
        value: 'grid',
      });
    } finally {
      vi.useRealTimers();
    }
  });

  test('Escape clears the field and reports immediately', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const host = attachedHost(panelId);
    deliverContribution([{ kind: 'search', id: 'filter', value: 'orm' }], contentWindow);

    const input = /** @type {HTMLInputElement} */ (host.querySelector('.contrib-search-input'));
    input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));

    expect(input.value).toBe('');
    expect(contentWindow.postMessage).toHaveBeenCalledWith(
      { type: 'osprey-header-action', id: 'filter', value: '' },
      window.location.origin
    );
  });

  test('survive a re-contribution without losing the element, its text or focus', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const host = attachedHost(panelId);

    deliverContribution(
      [
        { kind: 'nav', id: 'view', items: [{ id: 'a', label: 'A', active: true }] },
        { kind: 'search', id: 'filter', placeholder: 'Filter…' },
      ],
      contentWindow
    );
    const input = /** @type {HTMLInputElement} */ (host.querySelector('.contrib-search-input'));
    input.focus();
    input.value = 'half-typed';
    expect(document.activeElement).toBe(input);

    // The panel reacted to that keystroke and re-sent the WHOLE contribution
    // with new nav state. A rebuild here would drop the caret mid-word.
    deliverContribution(
      [
        { kind: 'nav', id: 'view', items: [{ id: 'a', label: 'A', active: false }] },
        { kind: 'search', id: 'filter', placeholder: 'Filter…', value: '' },
      ],
      contentWindow
    );

    expect(host.querySelector('.contrib-search-input')).toBe(input);
    expect(input.value).toBe('half-typed');
    expect(document.activeElement).toBe(input);
    // ...while the sibling nav still took its update.
    expect(host.querySelector('.contrib-nav-link')?.classList.contains('active')).toBe(false);
  });
});

describe('menu items', () => {
  /** @param {HTMLElement} host @returns {HTMLElement} */
  function openMenuIn(host) {
    const btn = /** @type {HTMLElement} */ (host.querySelector('.contrib-menu-btn'));
    btn.dispatchEvent(new Event('click', { bubbles: true }));
    return btn;
  }

  test('open a body-level popover whose entries carry the right roles', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const host = attachedHost(panelId);

    deliverContribution(
      [
        {
          kind: 'menu',
          id: 'more',
          items: [
            { id: 'all-sessions', label: 'All sessions', checked: false },
            { id: 'refresh', label: 'Refresh' },
            { id: 'stacked', label: 'Stacked layout', disabled: true },
          ],
        },
      ],
      contentWindow
    );

    const btn = openMenuIn(host);
    expect(btn.getAttribute('aria-expanded')).toBe('true');

    const pop = /** @type {HTMLElement} */ (document.querySelector('.contrib-menu-popover'));
    // Body-level: the contribution region is overflow:hidden and one bar tall.
    expect(pop.parentElement).toBe(document.body);

    const rows = [...pop.querySelectorAll('.contrib-menu-item')];
    expect(rows.map((r) => r.getAttribute('role'))).toEqual([
      'menuitemcheckbox',
      'menuitem',
      'menuitem',
    ]);
    expect(rows[0].getAttribute('aria-checked')).toBe('false');
    expect(/** @type {HTMLButtonElement} */ (rows[2]).disabled).toBe(true);
  });

  test('an entry click dispatches the menu id with the entry as value, and closes', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const host = attachedHost(panelId);

    deliverContribution(
      [{ kind: 'menu', id: 'more', items: [{ id: 'refresh', label: 'Refresh' }] }],
      contentWindow
    );
    openMenuIn(host);

    const row = /** @type {HTMLElement} */ (document.querySelector('.contrib-menu-item'));
    row.dispatchEvent(new Event('click', { bubbles: true }));

    expect(contentWindow.postMessage).toHaveBeenCalledWith(
      { type: 'osprey-header-action', id: 'more', value: 'refresh' },
      window.location.origin
    );
    expect(document.querySelector('.contrib-menu-popover')).toBeNull();
  });

  test('Escape closes an open popover', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const host = attachedHost(panelId);
    deliverContribution(
      [{ kind: 'menu', id: 'more', items: [{ id: 'refresh', label: 'Refresh' }] }],
      contentWindow
    );
    openMenuIn(host);

    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    expect(document.querySelector('.contrib-menu-popover')).toBeNull();
  });

  test('a re-contribution while open re-renders the live rows', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const host = attachedHost(panelId);
    deliverContribution(
      [
        {
          kind: 'menu',
          id: 'more',
          items: [{ id: 'all-sessions', label: 'All sessions', checked: false }],
        },
      ],
      contentWindow
    );
    openMenuIn(host);

    deliverContribution(
      [
        {
          kind: 'menu',
          id: 'more',
          items: [{ id: 'all-sessions', label: 'All sessions', checked: true }],
        },
      ],
      contentWindow
    );

    const row = /** @type {HTMLElement} */ (document.querySelector('.contrib-menu-item'));
    expect(row.getAttribute('aria-checked')).toBe('true');
    expect(row.querySelector('.contrib-menu-tick')?.textContent).toBe('✓');
  });

  test('unregistering the host takes the body-level popover with it', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const host = attachedHost(panelId);

    deliverContribution(
      [{ kind: 'menu', id: 'more', items: [{ id: 'refresh', label: 'Refresh' }] }],
      contentWindow
    );
    openMenuIn(host);
    expect(document.querySelector('.contrib-menu-popover')).toBeTruthy();

    unregisterContribHost(panelId, host);
    expect(document.querySelector('.contrib-menu-popover')).toBeNull();
  });
});

describe('sanitization of the extended vocabulary', () => {
  test('drops a repeated kind:id so the reconcile never aliases one element', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const host = attachedHost(panelId);

    deliverContribution(
      [
        { kind: 'text', id: 'dup', text: 'first' },
        { kind: 'text', id: 'dup', text: 'second' },
      ],
      contentWindow
    );

    expect([...host.querySelectorAll('.contrib-text')].map((n) => n.textContent)).toEqual([
      'first',
    ]);
  });

  test('drops a menu with no valid entries, and an unknown kind entirely', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const host = attachedHost(panelId);

    deliverContribution(
      [
        { kind: 'menu', id: 'empty', items: [{ nope: true }] },
        { kind: 'slider', id: 'nope', value: 3 },
        { kind: 'search', id: 'ok' },
      ],
      contentWindow
    );

    expect(host.querySelector('.contrib-menu-btn')).toBeNull();
    expect(host.children.length).toBe(1);
    expect(host.querySelector('.contrib-search')).toBeTruthy();
  });

  test('menu entry labels reach the DOM as text, never as markup', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const host = attachedHost(panelId);

    deliverContribution(
      [
        {
          kind: 'menu',
          id: 'more',
          items: [{ id: 'x', label: '<img src=x onerror=alert(1)>' }],
        },
      ],
      contentWindow
    );
    const btn = /** @type {HTMLElement} */ (host.querySelector('.contrib-menu-btn'));
    btn.dispatchEvent(new Event('click', { bubbles: true }));

    const pop = /** @type {HTMLElement} */ (document.querySelector('.contrib-menu-popover'));
    expect(pop.querySelector('img')).toBeNull();
    expect(pop.textContent).toContain('<img src=x onerror=alert(1)>');
  });
});
