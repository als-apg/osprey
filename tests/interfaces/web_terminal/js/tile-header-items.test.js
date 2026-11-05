/**
 * Unit tests for the tile-bar item vocabulary beyond text/nav/button: the v3
 * `icon` and `menu` kinds, the legacy `search`, and the hub-side additions
 * `crumb`, `sep` and icon buttons.
 *
 * Pure DOM/logic guard, happy-dom environment (configured globally):
 *   npx vitest run tests/interfaces/web_terminal/js/tile-header-items.test.js
 *
 * Round one's kinds (text/nav/button) and the receive pipeline itself are
 * covered by tile-header-contrib.test.js; this file drives the same public
 * entry points but exercises what tile-header-items.js renders per kind: the
 * panel-icon button, the menu's hidden holder (its entries are tile-⋯ rows),
 * the magnifier pill, the hub-side kinds, and — the reason the render pass
 * became a reconcile — a contributed input surviving its panel re-sending the
 * whole contribution underneath it.
 */

import { readFileSync } from 'node:fs';
import { join } from 'node:path';

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
  panelMenuRows,
  registerContribHost,
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
      data: { type: 'canopy-host-header-contribution', version: 1, items },
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
        type: 'canopy-host-header-action',
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
      { type: 'canopy-host-header-action', id: 'filter', value: '' },
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
  test('render no control: an empty hidden holder keeps the key; the entries are ⋯ rows', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const host = attachedHost(panelId);

    deliverContribution(
      [
        {
          kind: 'menu',
          id: 'more',
          label: 'Finder options',
          items: [
            { id: 'all-sessions', label: 'All sessions', checked: false },
            { id: 'refresh', label: 'Refresh' },
            { id: 'stacked', label: 'Stacked layout', disabled: true },
          ],
        },
      ],
      contentWindow
    );

    expect(host.querySelector('button')).toBeNull();
    const holder = /** @type {HTMLElement} */ (
      host.querySelector('[data-contrib-key="menu:more"]')
    );
    expect(holder.tagName).toBe('SPAN');
    expect(holder.hidden).toBe(true);
    expect(holder.childNodes.length).toBe(0);

    const rows = panelMenuRows(panelId);
    expect(rows.map((r) => [r.label, r.checked, r.disabled])).toEqual([
      ['All sessions', false, false],
      ['Refresh', undefined, false],
      ['Stacked layout', undefined, true],
    ]);
  });

  test('an entry row dispatches the menu id with the entry as value', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    attachedHost(panelId);

    deliverContribution(
      [{ kind: 'menu', id: 'more', items: [{ id: 'refresh', label: 'Refresh' }] }],
      contentWindow
    );
    panelMenuRows(panelId)[0].run?.();

    expect(contentWindow.postMessage).toHaveBeenCalledWith(
      { type: 'canopy-host-header-action', id: 'more', value: 'refresh' },
      window.location.origin
    );
  });

  test('a re-contribution replaces the rows the next open reads', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const host = attachedHost(panelId);
    const send = (/** @type {boolean} */ checked) =>
      deliverContribution(
        [
          {
            kind: 'menu',
            id: 'more',
            items: [{ id: 'all-sessions', label: 'All sessions', checked }],
          },
        ],
        contentWindow
      );

    send(false);
    const holder = host.querySelector('[data-contrib-key="menu:more"]');
    send(true);

    expect(host.querySelector('[data-contrib-key="menu:more"]')).toBe(holder);
    expect(panelMenuRows(panelId)[0].checked).toBe(true);
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

    expect(host.querySelector('[data-contrib-key^="menu:"]')).toBeNull();
    expect(host.children.length).toBe(1);
    expect(host.querySelector('.contrib-search')).toBeTruthy();
  });

  test('menu entry labels reach the ⋯ rows as text, never as markup', () => {
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

    expect(host.querySelector('img')).toBeNull();
    // The row carries the literal string; context-menu.js renders it via
    // textContent, so the markup never becomes an element.
    expect(panelMenuRows(panelId)[0].label).toBe('<img src=x onerror=alert(1)>');
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
      { type: 'canopy-host-header-action', id: 'next' },
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

describe('v3 icon items', () => {
  test('render as an osprey-icon-btn in the panel-icon slot, named by the title', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const host = attachedHost(panelId);

    deliverContribution(
      [
        { kind: 'icon', id: 'refresh', icon: 'refresh', title: 'Refresh fast figures' },
        { kind: 'icon', id: 'rail', icon: 'panel-left', title: 'Hide filters', pressed: true },
        { kind: 'icon', id: 'verify', icon: 'check', title: 'Run verification', disabled: true },
      ],
      contentWindow
    );

    const [refresh, rail, verify] = /** @type {HTMLButtonElement[]} */ ([
      ...host.querySelectorAll('button.osprey-icon-btn.contrib-icon'),
    ]);
    expect(refresh.dataset.contribKey).toBe('icon:refresh');
    expect(refresh.title).toBe('Refresh fast figures');
    expect(refresh.getAttribute('aria-label')).toBe('Refresh fast figures');
    expect(refresh.querySelector('svg.osprey-ic')).toBeTruthy();
    expect(refresh.textContent).toBe('');
    // aria-pressed only when the panel says so: a plain action claims no toggle.
    expect(refresh.hasAttribute('aria-pressed')).toBe(false);
    expect(rail.getAttribute('aria-pressed')).toBe('true');
    expect(verify.disabled).toBe(true);
  });

  test('a click round-trips the item id', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const host = attachedHost(panelId);
    deliverContribution(
      [{ kind: 'icon', id: 'refresh', icon: 'refresh', title: 'Refresh' }],
      contentWindow
    );

    /** @type {HTMLButtonElement} */ (host.querySelector('.contrib-icon')).click();

    expect(contentWindow.postMessage).toHaveBeenCalledWith(
      { type: 'canopy-host-header-action', id: 'refresh' },
      window.location.origin
    );
  });

  test('a patch keeps the element; the pictogram changes only with the icon name', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const host = attachedHost(panelId);
    const send = (/** @type {object} */ fields) =>
      deliverContribution(
        [{ kind: 'icon', id: 'rail', icon: 'panel-left', title: 'Hide filters', ...fields }],
        contentWindow
      );

    send({ pressed: true });
    const btn = /** @type {HTMLButtonElement} */ (host.querySelector('.contrib-icon'));
    const svg = btn.querySelector('svg');

    send({ pressed: false, title: 'Show filters', disabled: true });
    expect(host.querySelector('.contrib-icon')).toBe(btn);
    expect(btn.querySelector('svg')).toBe(svg);
    expect(btn.getAttribute('aria-pressed')).toBe('false');
    expect(btn.title).toBe('Show filters');
    expect(btn.getAttribute('aria-label')).toBe('Show filters');
    expect(btn.disabled).toBe(true);

    send({ icon: 'filter' });
    expect(host.querySelector('.contrib-icon')).toBe(btn);
    expect(btn.querySelector('svg')).not.toBe(svg);
    expect(btn.querySelectorAll('svg').length).toBe(1);
    expect(btn.hasAttribute('aria-pressed')).toBe(false);
  });

  test('an unknown icon name or a missing title drops the item', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const host = attachedHost(panelId);

    deliverContribution(
      [
        { kind: 'icon', id: 'ghost', icon: 'no-such-icon', title: 'Ghost' },
        { kind: 'icon', id: 'nameless', icon: 'refresh' },
        { kind: 'icon', id: 'blank', icon: 'refresh', title: '' },
        { kind: 'icon', id: 'ok', icon: 'refresh', title: 'Refresh' },
      ],
      contentWindow
    );

    expect(
      [...host.querySelectorAll('[data-contrib-key]')].map((el) => el.dataset.contribKey)
    ).toEqual(['icon:ok']);
  });
});

// The browser half of a rule happy-dom cannot lay out: the contribution
// region was `overflow: hidden`, which is still a scroll container, so
// pressing a control at its clipped edge focused it, the browser scrolled the
// region to reveal it, and that scroll — of an element holding the control —
// closed the menu the press had just opened. A clipped region cannot scroll,
// by focus or by script.
describe('contribution region overflow', () => {
  test('the region clips its overflow, so focusing a control cannot scroll it', () => {
    const css = readFileSync(
      join(
        import.meta.dirname,
        '../../../../src/osprey/interfaces/web_terminal/static/css/dockview-overrides.css'
      ),
      'utf8'
    ).replace(/\/\*[\s\S]*?\*\//g, '');
    const overflow = [...css.matchAll(/([^{}]+)\{([^{}]*)\}/g)]
      .filter(([, selectors]) =>
        selectors.split(',').some((s) => s.trim() === '.tile-tab .tile-tab-contrib')
      )
      .flatMap(([, , body]) =>
        [...body.matchAll(/(?:^|;)\s*overflow\s*:\s*([^;]+)/g)].map((d) => d[1].trim())
      );
    expect(overflow).toEqual(['clip']);
  });
});
