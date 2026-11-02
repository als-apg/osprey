/**
 * Unit tests for the extended tile-bar item vocabulary — `search` and `menu`,
 * and the hub-side additions `crumb`, `sep` and icon buttons.
 *
 * Pure DOM/logic guard, happy-dom environment (configured globally):
 *   npx vitest run tests/interfaces/web_terminal/js/tile-header-items.test.js
 *
 * Round one's kinds (text/nav/button) and the receive pipeline itself are
 * covered by tile-header-contrib.test.js; this file drives the same public
 * entry points but exercises what tile-header-items.js added: the magnifier
 * pill, the body-level overflow popover, the hub-side kinds, and — the
 * reason the render pass became a reconcile — a contributed input surviving
 * its panel re-sending the whole contribution underneath it.
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
import {
  closeMenu,
  toggleOverflowMenu,
} from '../../../../src/osprey/interfaces/web_terminal/static/js/tile-header-items.js';

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
  closeMenu();
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

describe('hub-side kinds', () => {
  test('a crumb renders root › leaf, with the leaf as its tooltip', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const host = attachedHost(panelId);

    deliverContribution(
      [{ kind: 'crumb', id: 'crumb', root: 'Artifacts', leaf: 'SR lattice betatron tunes' }],
      contentWindow
    );

    const crumb = /** @type {HTMLElement} */ (host.querySelector('span.osprey-crumb'));
    const [root, sep, leaf] = /** @type {HTMLElement[]} */ ([...crumb.children]);
    expect(root.className).toBe('osprey-crumb-root');
    expect(root.textContent).toBe('Artifacts');
    expect(sep.className).toBe('osprey-crumb-sep');
    expect(sep.getAttribute('aria-hidden')).toBe('true');
    expect(sep.querySelector('svg.osprey-ic')).toBeTruthy();
    expect(leaf.className).toBe('osprey-crumb-leaf');
    expect(leaf.textContent).toBe('SR lattice betatron tunes');
    expect(leaf.title).toBe('SR lattice betatron tunes');
    expect(sep.hidden).toBe(false);
    expect(leaf.hidden).toBe(false);
  });

  test('an empty leaf leaves the root alone, on the same element', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const host = attachedHost(panelId);
    deliverContribution(
      [{ kind: 'crumb', id: 'crumb', root: 'Artifacts', leaf: 'Tunes' }],
      contentWindow
    );
    const crumb = host.querySelector('.osprey-crumb');

    deliverContribution(
      [{ kind: 'crumb', id: 'crumb', root: 'Artifacts', leaf: '' }],
      contentWindow
    );

    expect(host.querySelector('.osprey-crumb')).toBe(crumb);
    const sep = /** @type {HTMLElement} */ (host.querySelector('.osprey-crumb-sep'));
    const leaf = /** @type {HTMLElement} */ (host.querySelector('.osprey-crumb-leaf'));
    expect(sep.hidden).toBe(true);
    expect(leaf.hidden).toBe(true);
    expect(leaf.hasAttribute('title')).toBe(false);
    expect(host.querySelector('.osprey-crumb-root')?.textContent).toBe('Artifacts');
  });

  test('the host carries has-crumb only while a crumb is contributed', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const host = attachedHost(panelId);

    deliverContribution([{ kind: 'crumb', id: 'crumb', root: 'Artifacts' }], contentWindow);
    expect(host.classList.contains('has-crumb')).toBe(true);

    deliverContribution([{ kind: 'text', id: 'name', text: 'x' }], contentWindow);
    expect(host.classList.contains('has-crumb')).toBe(false);
  });

  test('a sep renders as a hidden-from-assistive-tech hairline', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const host = attachedHost(panelId);

    deliverContribution([{ kind: 'sep', id: 'sep' }], contentWindow);

    const sep = /** @type {HTMLElement} */ (host.querySelector('span.osprey-head-sep'));
    expect(sep.getAttribute('aria-hidden')).toBe('true');
    expect(sep.childNodes.length).toBe(0);
  });

  test('an icon-only button is an osprey-icon-btn named by its title', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const host = attachedHost(panelId);

    deliverContribution(
      [
        {
          kind: 'button',
          id: 'prev',
          icon: 'chevron-left',
          title: 'Previous artifact',
          disabled: true,
        },
      ],
      contentWindow
    );

    const btn = /** @type {HTMLButtonElement} */ (host.querySelector('button.osprey-icon-btn'));
    expect(btn.classList.contains('contrib-btn')).toBe(false);
    expect(btn.getAttribute('aria-label')).toBe('Previous artifact');
    expect(btn.title).toBe('Previous artifact');
    expect(btn.querySelector('svg.osprey-ic')).toBeTruthy();
    expect(btn.textContent).toBe('');
    expect(btn.disabled).toBe(true);
  });

  test('a labelled icon button is an osprey-btn with a label span; accent is primary', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const host = attachedHost(panelId);

    deliverContribution(
      [
        {
          kind: 'button',
          id: 'open',
          icon: 'maximize',
          label: 'Open full size',
          title: 'Open in a new tab',
          tone: 'accent',
        },
        { kind: 'button', id: 'save', icon: 'save', label: 'Save' },
      ],
      contentWindow
    );

    const [open, save] = /** @type {HTMLButtonElement[]} */ ([
      ...host.querySelectorAll('button.osprey-btn'),
    ]);
    expect(open.classList.contains('primary')).toBe(true);
    expect(open.firstElementChild?.matches('svg.osprey-ic')).toBe(true);
    expect(open.querySelector('.osprey-btn-label')?.textContent).toBe('Open full size');
    expect(open.title).toBe('Open in a new tab');
    expect(open.hasAttribute('aria-label')).toBe(false);
    // No title: the label doubles as the tooltip.
    expect(save.classList.contains('primary')).toBe(false);
    expect(save.title).toBe('Save');
  });

  test('an icon button dispatches its id like any button', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const host = attachedHost(panelId);
    deliverContribution(
      [{ kind: 'button', id: 'next', icon: 'chevron-right', title: 'Next artifact' }],
      contentWindow
    );

    /** @type {HTMLButtonElement} */ (host.querySelector('.osprey-icon-btn')).click();

    expect(contentWindow.postMessage).toHaveBeenCalledWith(
      { type: 'osprey-header-action', id: 'next' },
      window.location.origin
    );
  });

  test('a patch keeps the button and its icon, and rebuilds only on a layout change', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const host = attachedHost(panelId);
    const send = (/** @type {object} */ fields) =>
      deliverContribution([{ kind: 'button', id: 'save', icon: 'save', ...fields }], contentWindow);

    send({ label: 'Save', disabled: true });
    const btn = /** @type {HTMLButtonElement} */ (host.querySelector('button'));
    const svg = btn.querySelector('svg');

    // Same layout: label and disabled patch in place; the icon survives.
    send({ label: 'Saved', disabled: false });
    expect(host.querySelector('button')).toBe(btn);
    expect(btn.querySelector('svg')).toBe(svg);
    expect(btn.querySelector('.osprey-btn-label')?.textContent).toBe('Saved');
    expect(btn.disabled).toBe(false);

    // Another icon: the children change, the button does not.
    send({ icon: 'maximize', label: 'Saved' });
    expect(host.querySelector('button')).toBe(btn);
    expect(btn.querySelector('svg')).not.toBe(svg);
    expect(btn.querySelectorAll('svg').length).toBe(1);

    // The label goes: it becomes an icon-only button, still the same element.
    send({ icon: 'maximize', title: 'Saved' });
    expect(host.querySelector('button')).toBe(btn);
    expect(btn.classList.contains('osprey-icon-btn')).toBe(true);
    expect(btn.classList.contains('osprey-btn')).toBe(false);
    expect(btn.querySelector('.osprey-btn-label')).toBeNull();
    expect(btn.getAttribute('aria-label')).toBe('Saved');
  });

  test('an unknown icon name is dropped from the item', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const host = attachedHost(panelId);

    deliverContribution(
      [
        { kind: 'button', id: 'export', icon: 'no-such-icon', label: 'Export' },
        // Without the icon these have no accessible name, so they go too.
        { kind: 'button', id: 'ghost', icon: 'no-such-icon', title: 'Ghost' },
        { kind: 'button', id: 'nameless', icon: 'save' },
      ],
      contentWindow
    );

    expect(host.children.length).toBe(1);
    const btn = /** @type {HTMLButtonElement} */ (host.querySelector('button'));
    expect(btn.className).toBe('contrib-btn');
    expect(btn.textContent).toBe('Export');
    expect(btn.querySelector('svg')).toBeNull();
  });
});

// Close-on-scroll guards against the popover drifting off its button, so only
// a scroll that moves the button may close it -- not the chat pane
// auto-scrolling on new output, nor the popover's own rows.
describe('popover close-on-scroll', () => {
  function openOverflow() {
    const strip = document.createElement('div');
    const btn = document.createElement('button');
    strip.appendChild(btn);
    document.body.appendChild(strip);
    toggleOverflowMenu(btn, [{ label: 'Clock', pick: () => {} }]);
    return {
      strip,
      pop: /** @type {HTMLElement} */ (document.querySelector('.contrib-menu-popover')),
    };
  }

  test('a scroll elsewhere on the page keeps it open', () => {
    openOverflow();
    const chat = document.createElement('div');
    document.body.appendChild(chat);
    chat.dispatchEvent(new Event('scroll'));
    expect(document.querySelector('.contrib-menu-popover')).not.toBeNull();
  });

  test('scrolling its own rows keeps it open', () => {
    const { pop } = openOverflow();
    pop.dispatchEvent(new Event('scroll'));
    expect(document.querySelector('.contrib-menu-popover')).not.toBeNull();
  });

  test('a scroll that moves the button closes it', () => {
    const { strip } = openOverflow();
    strip.dispatchEvent(new Event('scroll'));
    expect(document.querySelector('.contrib-menu-popover')).toBeNull();
  });
});
