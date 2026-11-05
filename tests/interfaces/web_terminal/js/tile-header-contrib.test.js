/**
 * Unit tests for the hub side of the tile header-bar contribution contract.
 *
 * Pure DOM/logic guard, happy-dom environment (configured globally):
 *   npx vitest run tests/interfaces/web_terminal/js/tile-header-contrib.test.js
 *
 * Covers the receive pipeline (origin check, source→panel-id resolution
 * against live iframes, vocabulary sanitization), the per-panel contribution
 * cache vs. host-registration ordering, rendering (textContent only — no
 * markup injection), the action round-trip back to the iframe, the hub's own
 * contributions (contributeLocal), whose actions stay in the hub, and the
 * fold ladder, whose fold target is the tile ⋯ ("This panel" rows from
 * panelMenuRows) rather than a button of the bar's own.
 *
 * The module keeps page-lifetime state (wired listener, contribution cache),
 * matching production where it lives as long as the shell does — tests use
 * distinct panel ids to stay independent.
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
  contributeLocal,
  initHeaderContrib,
  panelMenuRows,
  registerContribHost,
  unregisterContribHost,
} from '../../../../src/osprey/interfaces/web_terminal/static/js/tile-header-contrib.js';

/** Monotonic id source so each test gets an isolated panel id. */
let seq = 0;
/** @returns {string} */
function freshPanelId() {
  seq += 1;
  return `panel-${seq}`;
}

/**
 * Create a fake service-panel iframe whose contentWindow identity the module
 * matches messages against, with a spyable postMessage for the action
 * round-trip.
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
 * Deliver a contribution message as if it came from the given source window.
 * @param {any} items
 * @param {any} source
 * @param {string} [origin]
 */
function deliverContribution(items, source, origin = window.location.origin) {
  window.dispatchEvent(
    new MessageEvent('message', {
      data: { type: 'canopy-host-header-contribution', version: 1, items },
      origin,
      source,
    })
  );
}

initHeaderContrib();
initHeaderContrib(); // idempotent — a second call must not double-listen

afterEach(() => {
  document.body.replaceChildren();
});

/**
 * The Artifacts tile's header as the hub's sidebar sends it, with an
 * artifact selected (contract C3, in its order).
 */
const ARTIFACT_HEADER = [
  { kind: 'crumb', id: 'crumb', root: 'Artifacts', leaf: 'SR lattice betatron tunes' },
  { kind: 'button', id: 'prev', icon: 'chevron-left', title: 'Previous artifact', priority: 9 },
  { kind: 'text', id: 'pager', text: '3 of 12', priority: 9 },
  { kind: 'button', id: 'next', icon: 'chevron-right', title: 'Next artifact', priority: 9 },
  { kind: 'sep', id: 'sep' },
  {
    kind: 'button',
    id: 'open',
    icon: 'maximize',
    label: 'Open full size',
    title: 'Open full size',
    tone: 'accent',
    priority: 5,
  },
  { kind: 'button', id: 'save', icon: 'save', label: 'Save', title: 'Save', priority: 5 },
];

describe('receive pipeline', () => {
  test('renders a contribution from a matching iframe into a registered host', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const host = document.createElement('div');
    registerContribHost(panelId, host);

    deliverContribution([{ kind: 'text', id: 'name', text: 'bl4c_v2' }], contentWindow);

    expect(host.querySelector('.contrib-text')?.textContent).toBe('bl4c_v2');
    expect(host.classList.contains('has-items')).toBe(true);
  });

  test("reads a viewer's old osprey-header-contribution name for one release", () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const host = document.createElement('div');
    registerContribHost(panelId, host);

    window.dispatchEvent(
      new MessageEvent('message', {
        data: {
          type: 'osprey-header-contribution',
          version: 1,
          items: [{ kind: 'text', id: 'name', text: 'old' }],
        },
        origin: window.location.origin,
        source: /** @type {any} */ (contentWindow),
      })
    );

    expect(host.querySelector('.contrib-text')?.textContent).toBe('old');
  });

  test('caches a contribution that arrives before the host registers', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);

    deliverContribution([{ kind: 'text', id: 'name', text: 'early' }], contentWindow);
    const host = document.createElement('div');
    registerContribHost(panelId, host);

    expect(host.querySelector('.contrib-text')?.textContent).toBe('early');
  });

  test('a later contribution replaces the earlier one wholesale', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const host = document.createElement('div');
    registerContribHost(panelId, host);

    deliverContribution(
      [
        { kind: 'button', id: 'a', label: 'A' },
        { kind: 'button', id: 'b', label: 'B' },
      ],
      contentWindow
    );
    deliverContribution([{ kind: 'button', id: 'c', label: 'C' }], contentWindow);

    const labels = [...host.querySelectorAll('.contrib-btn')].map((b) => b.textContent);
    expect(labels).toEqual(['C']);
  });

  test('an empty contribution clears the region and the has-items class', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const host = document.createElement('div');
    registerContribHost(panelId, host);

    deliverContribution([{ kind: 'text', id: 'name', text: 'x' }], contentWindow);
    deliverContribution([], contentWindow);

    expect(host.children.length).toBe(0);
    expect(host.classList.contains('has-items')).toBe(false);
  });

  test('ignores messages from a foreign origin', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const host = document.createElement('div');
    registerContribHost(panelId, host);

    deliverContribution(
      [{ kind: 'text', id: 'name', text: 'evil' }],
      contentWindow,
      'https://evil.example'
    );

    expect(host.children.length).toBe(0);
  });

  test('ignores messages whose source matches no panel iframe', () => {
    const panelId = freshPanelId();
    addPanelIframe(panelId);
    const host = document.createElement('div');
    registerContribHost(panelId, host);

    deliverContribution([{ kind: 'text', id: 'name', text: 'spoof' }], { postMessage() {} });

    expect(host.children.length).toBe(0);
  });

  test('a panel cannot inject into another tile: identity comes from the source', () => {
    const panelA = freshPanelId();
    const panelB = freshPanelId();
    const { contentWindow: windowB } = addPanelIframe(panelB);
    addPanelIframe(panelA);
    const hostA = document.createElement('div');
    const hostB = document.createElement('div');
    registerContribHost(panelA, hostA);
    registerContribHost(panelB, hostB);

    deliverContribution([{ kind: 'text', id: 'name', text: 'from B' }], windowB);

    expect(hostA.children.length).toBe(0);
    expect(hostB.querySelector('.contrib-text')?.textContent).toBe('from B');
  });
});

describe('sanitization', () => {
  test('drops unknown kinds and malformed items, keeps valid ones', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const host = document.createElement('div');
    registerContribHost(panelId, host);

    deliverContribution(
      [
        { kind: 'chip', id: 'x', text: 'unknown kind' },
        { kind: 'button', id: 'no-label' },
        { kind: 'button', label: 'no id' },
        'not-an-object',
        { kind: 'button', id: 'ok', label: 'OK' },
      ],
      contentWindow
    );

    expect(host.children.length).toBe(1);
    expect(host.querySelector('.contrib-btn')?.textContent).toBe('OK');
  });

  test('renders markup-bearing text as literal text, never as elements', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const host = document.createElement('div');
    registerContribHost(panelId, host);

    deliverContribution(
      [{ kind: 'text', id: 't', text: '<img src=x onerror=alert(1)>' }],
      contentWindow
    );

    expect(host.querySelector('img')).toBeNull();
    expect(host.querySelector('.contrib-text')?.textContent).toBe('<img src=x onerror=alert(1)>');
  });

  test('clips oversized text and caps item count', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const host = document.createElement('div');
    registerContribHost(panelId, host);

    const flood = Array.from({ length: 40 }, (_, i) => ({
      kind: 'text',
      id: `t${i}`,
      text: 'x'.repeat(500),
    }));
    deliverContribution(flood, contentWindow);

    expect(host.children.length).toBe(12);
    expect(host.querySelector('.contrib-text')?.textContent?.length).toBe(120);
  });

  test('nav entries missing id or label are dropped; empty navs vanish', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const host = document.createElement('div');
    registerContribHost(panelId, host);

    deliverContribution(
      [
        { kind: 'nav', id: 'v', items: [{ id: 'a', label: 'A', active: true }, { id: 'bad' }] },
        { kind: 'nav', id: 'empty', items: [{}] },
      ],
      contentWindow
    );

    const links = [...host.querySelectorAll('.contrib-nav-link')];
    expect(links.map((l) => l.textContent)).toEqual(['A']);
    expect(links[0].classList.contains('active')).toBe(true);
    expect(host.querySelectorAll('.contrib-nav').length).toBe(1);
  });
});

describe('action round-trip', () => {
  test('button click posts canopy-host-header-action to the panel iframe', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const host = document.createElement('div');
    registerContribHost(panelId, host);
    deliverContribution([{ kind: 'button', id: 'refresh', label: 'Refresh' }], contentWindow);

    /** @type {HTMLButtonElement} */ (host.querySelector('.contrib-btn')).click();

    expect(contentWindow.postMessage).toHaveBeenCalledWith(
      { type: 'canopy-host-header-action', id: 'refresh' },
      window.location.origin
    );
  });

  test('nav click includes the entry id as value', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const host = document.createElement('div');
    registerContribHost(panelId, host);
    deliverContribution(
      [{ kind: 'nav', id: 'view', items: [{ id: 'browse', label: 'Browse' }] }],
      contentWindow
    );

    /** @type {HTMLButtonElement} */ (host.querySelector('.contrib-nav-link')).click();

    expect(contentWindow.postMessage).toHaveBeenCalledWith(
      { type: 'canopy-host-header-action', id: 'view', value: 'browse' },
      window.location.origin
    );
  });

  test('disabled buttons render disabled', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const host = document.createElement('div');
    registerContribHost(panelId, host);
    deliverContribution(
      [{ kind: 'button', id: 'verify', label: 'Verify', disabled: true }],
      contentWindow
    );

    expect(/** @type {HTMLButtonElement} */ (host.querySelector('.contrib-btn')).disabled).toBe(
      true
    );
  });
});

describe('local contribution (the hub owns the header)', () => {
  test('renders into a registered host and replaces the earlier items', () => {
    const panelId = freshPanelId();
    addPanelIframe(panelId);
    const host = document.createElement('div');
    registerContribHost(panelId, host);

    contributeLocal(
      panelId,
      [
        { kind: 'button', id: 'a', label: 'A' },
        { kind: 'button', id: 'b', label: 'B' },
      ],
      vi.fn()
    );
    expect([...host.querySelectorAll('.contrib-btn')].map((b) => b.textContent)).toEqual([
      'A',
      'B',
    ]);

    contributeLocal(panelId, [{ kind: 'text', id: 'name', text: 'only' }], vi.fn());
    expect(host.children.length).toBe(1);
    expect(host.querySelector('.contrib-text')?.textContent).toBe('only');
  });

  test('waits for a host that registers later', () => {
    const panelId = freshPanelId();
    contributeLocal(panelId, [{ kind: 'text', id: 'name', text: 'early' }], vi.fn());

    const host = document.createElement('div');
    registerContribHost(panelId, host);

    expect(host.querySelector('.contrib-text')?.textContent).toBe('early');
  });

  test('runs the same sanitizer as the message path', () => {
    const panelId = freshPanelId();
    const host = document.createElement('div');
    registerContribHost(panelId, host);

    contributeLocal(
      panelId,
      [
        { kind: 'chip', id: 'x', text: 'unknown kind' },
        { kind: 'text', id: 't', text: '<img src=x onerror=alert(1)>' },
      ],
      vi.fn()
    );

    expect(host.children.length).toBe(1);
    expect(host.querySelector('img')).toBeNull();
  });

  test('button, nav and menu actions reach the local dispatch, never an iframe', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const host = document.createElement('div');
    registerContribHost(panelId, host);
    const dispatch = vi.fn();

    contributeLocal(
      panelId,
      [
        { kind: 'nav', id: 'view', items: [{ id: 'browse', label: 'Browse' }] },
        { kind: 'button', id: 'prev', icon: 'chevron-left', title: 'Previous artifact' },
        { kind: 'menu', id: 'more', items: [{ id: 'print', label: 'Print' }] },
      ],
      dispatch
    );

    /** @type {HTMLButtonElement} */ (host.querySelector('.osprey-icon-btn')).click();
    expect(dispatch).toHaveBeenLastCalledWith('prev');

    /** @type {HTMLButtonElement} */ (host.querySelector('.contrib-nav-link')).click();
    expect(dispatch).toHaveBeenLastCalledWith('view', 'browse');

    // The menu renders no button; its entry is a tile-⋯ row.
    expect(host.querySelector('button[data-contrib-key="menu:more"]')).toBeNull();
    const [row] = panelMenuRows(panelId);
    expect(row.label).toBe('Print');
    row.run?.();
    expect(dispatch).toHaveBeenLastCalledWith('more', 'print');

    expect(dispatch).toHaveBeenCalledTimes(3);
    expect(contentWindow.postMessage).not.toHaveBeenCalled();
  });

  test("ignores the panel's own messages while local; other panels are unaffected", () => {
    const localPanel = freshPanelId();
    const otherPanel = freshPanelId();
    const { contentWindow: localWindow } = addPanelIframe(localPanel);
    const { contentWindow: otherWindow } = addPanelIframe(otherPanel);
    const localHost = document.createElement('div');
    const otherHost = document.createElement('div');
    registerContribHost(localPanel, localHost);
    registerContribHost(otherPanel, otherHost);

    contributeLocal(localPanel, [{ kind: 'text', id: 'name', text: 'from the hub' }], vi.fn());
    deliverContribution([{ kind: 'text', id: 'name', text: 'from the panel' }], localWindow);
    deliverContribution([{ kind: 'text', id: 'name', text: 'other panel' }], otherWindow);

    expect(localHost.querySelector('.contrib-text')?.textContent).toBe('from the hub');
    expect(otherHost.querySelector('.contrib-text')?.textContent).toBe('other panel');
  });
});

describe('fold ladder', () => {
  /**
   * happy-dom does no layout, so crowding is mocked: clientWidth is fixed and
   * scrollWidth derives from the currently-VISIBLE children at deterministic
   * per-kind widths. The shared controls sit outside the host, so the width
   * they free when folded is credited to the host's clientWidth (a visible
   * shared control costs 36). Folding/collapsing therefore changes the
   * measurement, which is exactly the feedback loop the ladder runs on.
   * @param {HTMLElement} host
   * @param {number} clientWidth
   * @param {HTMLElement | null} [controls]
   */
  function mockCrowding(host, clientWidth, controls = null) {
    Object.defineProperty(host, 'clientWidth', {
      configurable: true,
      get() {
        if (!controls) return clientWidth;
        const shown = [...controls.querySelectorAll('[data-control]')].filter(
          (c) => !c.classList.contains('contrib-hidden') && !c.hasAttribute('hidden')
        ).length;
        return clientWidth - 36 * shown;
      },
    });
    Object.defineProperty(host, 'scrollWidth', {
      configurable: true,
      get() {
        let w = 0;
        for (const el of host.children) {
          if (el.classList.contains('contrib-hidden') || /** @type {HTMLElement} */ (el).hidden) {
            continue;
          } else if (el.classList.contains('contrib-search-collapsed')) w += 24;
          else if (el.classList.contains('contrib-search')) w += 92;
          else if (el.classList.contains('contrib-icon')) w += 36;
          else if (el.classList.contains('contrib-text')) w += 40;
          else if (el.classList.contains('osprey-crumb')) w += 60;
          else if (el.classList.contains('osprey-head-sep')) w += 10;
          else w += 80;
        }
        return w;
      },
    });
  }

  /**
   * A tile's shared-controls element as tile-bar-controls.js builds it: the
   * hairline, then one `[data-control]` button per id.
   * @param {string[]} ids
   * @returns {HTMLElement}
   */
  function sharedControls(ids) {
    const el = document.createElement('div');
    el.className = 'tile-tab-controls';
    const divider = document.createElement('span');
    divider.className = 'tile-tab-divider';
    divider.hidden = true;
    el.appendChild(divider);
    for (const id of ids) {
      const btn = document.createElement('button');
      btn.dataset.control = id;
      el.appendChild(btn);
    }
    return el;
  }

  /** @param {string} id @param {number} [priority] */
  const iconItem = (id, priority = 0) => ({
    kind: 'icon',
    id,
    icon: 'refresh',
    title: id,
    priority,
  });

  /** @param {HTMLElement} host */
  const hiddenKeys = (host) =>
    [...host.querySelectorAll('.contrib-hidden')].map(
      (el) => /** @type {HTMLElement} */ (el).dataset.contribKey
    );

  test('a crowded bar collapses search to its magnifier before folding a legacy control', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const host = document.createElement('div');
    registerContribHost(panelId, host);
    mockCrowding(host, 150); // search(92) + button(80) = 172 > 150; collapsed: 104 fits

    deliverContribution(
      [
        { kind: 'search', id: 'filter', placeholder: 'Filter…' },
        { kind: 'button', id: 'refresh', label: 'Refresh' },
      ],
      contentWindow
    );

    const search = host.querySelector('.contrib-search');
    expect(search?.classList.contains('contrib-search-collapsed')).toBe(true);
    expect(search?.getAttribute('role')).toBe('button');
    expect(host.querySelector('.contrib-hidden')).toBeNull();
    expect(panelMenuRows(panelId)).toEqual([]);
  });

  test('still-crowded legacy controls fold lowest-priority-first into ⋯ rows that round-trip', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const host = document.createElement('div');
    registerContribHost(panelId, host);
    mockCrowding(host, 120); // nav(80) + button(80) = 160 > 120; fold nav: 80 fits

    deliverContribution(
      [
        {
          kind: 'nav',
          id: 'view',
          priority: 1,
          items: [
            { id: 'browse', label: 'Browse', active: true },
            { id: 'edit', label: 'Edit' },
          ],
        },
        { kind: 'button', id: 'refresh', label: 'Refresh', priority: 5 },
      ],
      contentWindow
    );

    // The lower-priority nav folded; the button stayed; no button of the bar's own appeared.
    expect(host.querySelector('.contrib-nav')?.classList.contains('contrib-hidden')).toBe(true);
    expect(host.querySelector('.contrib-btn')?.classList.contains('contrib-hidden')).toBe(false);
    expect(host.querySelectorAll(':scope > button').length).toBe(1);

    // The folded nav is one ⋯ row per view, active = checked.
    const rows = panelMenuRows(panelId);
    expect(rows.map((r) => [r.label, r.checked])).toEqual([
      ['Browse', true],
      ['Edit', false],
    ]);

    // A row dispatches exactly what the unfolded control would have.
    rows[1].run?.();
    expect(contentWindow.postMessage).toHaveBeenCalledWith(
      { type: 'canopy-host-header-action', id: 'view', value: 'edit' },
      window.location.origin
    );
  });

  test('subtitle text never folds — controls give way around it', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const host = document.createElement('div');
    registerContribHost(panelId, host);
    mockCrowding(host, 60); // text(40) + button(80) = 120 > 60

    deliverContribution(
      [
        { kind: 'text', id: 'file', text: 'als_ring_v3.lat' },
        { kind: 'button', id: 'refresh', label: 'Refresh' },
      ],
      contentWindow
    );

    expect(host.querySelector('.contrib-text')?.classList.contains('contrib-hidden')).toBe(false);
    expect(host.querySelector('.contrib-btn')?.classList.contains('contrib-hidden')).toBe(true);
    expect(panelMenuRows(panelId).map((r) => r.label)).toEqual(['Refresh']);
  });

  test('a widened tile gets everything back', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const host = document.createElement('div');
    registerContribHost(panelId, host);
    mockCrowding(host, 120);

    const items = [
      { kind: 'search', id: 'filter' },
      { kind: 'button', id: 'refresh', label: 'Refresh' },
      { kind: 'button', id: 'export', label: 'Export' },
    ];
    deliverContribution(items, contentWindow);
    expect(host.querySelectorAll('.contrib-hidden').length).toBeGreaterThan(0);

    mockCrowding(host, 1000);
    deliverContribution(items, contentWindow);

    expect(host.querySelectorAll('.contrib-hidden').length).toBe(0);
    expect(host.querySelector('.contrib-search-collapsed')).toBeNull();
    expect(panelMenuRows(panelId)).toEqual([]);
  });

  test('a collapsed search expands for typing on click and re-collapses on blur', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const host = document.createElement('div');
    document.body.appendChild(host); // focus needs an attached element
    registerContribHost(panelId, host);
    mockCrowding(host, 100); // search alone: 92 fits, but with a button it collapses

    deliverContribution(
      [
        { kind: 'search', id: 'filter', placeholder: 'Filter…' },
        { kind: 'button', id: 'refresh', label: 'Refresh' },
      ],
      contentWindow
    );
    const wrap = /** @type {HTMLElement} */ (host.querySelector('.contrib-search'));
    expect(wrap.classList.contains('contrib-search-collapsed')).toBe(true);

    wrap.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    const input = /** @type {HTMLInputElement} */ (wrap.querySelector('input'));
    expect(wrap.classList.contains('contrib-search-collapsed')).toBe(false);
    expect(document.activeElement).toBe(input);

    // While pinned open, a ladder re-run must not collapse it under the caret.
    deliverContribution(
      [
        { kind: 'search', id: 'filter', placeholder: 'Filter…' },
        { kind: 'button', id: 'refresh', label: 'Refresh' },
      ],
      contentWindow
    );
    expect(wrap.classList.contains('contrib-search-collapsed')).toBe(false);

    input.blur();
    input.dispatchEvent(new FocusEvent('focusout', { bubbles: true }));
    expect(wrap.classList.contains('contrib-search-collapsed')).toBe(true);
  });

  test('crumb, sep and text never fold; buttons fold lowest priority first', () => {
    const panelId = freshPanelId();
    addPanelIframe(panelId);
    const host = document.createElement('div');
    registerContribHost(panelId, host);
    // crumb(60) + pager(40) + sep(10) + 4 buttons(320) = 430; folding open and
    // save (priority 5) leaves 270, which fits in 300.
    mockCrowding(host, 300);

    contributeLocal(panelId, ARTIFACT_HEADER, vi.fn());
    expect(hiddenKeys(host)).toEqual(['button:open', 'button:save']);

    // Narrower still, every button folds and the rest holds its ground.
    mockCrowding(host, 50);
    contributeLocal(panelId, ARTIFACT_HEADER, vi.fn());

    expect(hiddenKeys(host)).toEqual(['button:prev', 'button:next', 'button:open', 'button:save']);
    expect(host.querySelector('.osprey-crumb')?.classList.contains('contrib-hidden')).toBe(false);
    expect(host.querySelector('.osprey-head-sep')?.classList.contains('contrib-hidden')).toBe(
      false
    );
    expect(host.querySelector('.contrib-text')?.classList.contains('contrib-hidden')).toBe(false);
  });

  test('a folded hub-contributed button dispatches locally from its ⋯ row', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const host = document.createElement('div');
    registerContribHost(panelId, host);
    mockCrowding(host, 50);
    const dispatch = vi.fn();

    contributeLocal(panelId, ARTIFACT_HEADER, dispatch);

    // Rows in contribution order; an icon-only button's row reads its title.
    const rows = panelMenuRows(panelId);
    expect(rows.map((r) => r.label)).toEqual([
      'Previous artifact',
      'Next artifact',
      'Open full size',
      'Save',
    ]);

    rows[0].run?.();
    expect(dispatch).toHaveBeenCalledWith('prev');
    expect(contentWindow.postMessage).not.toHaveBeenCalled();
  });

  test('the slot holds three panel icons; the rest fold whatever the width', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const host = document.createElement('div');
    registerContribHost(panelId, host);
    mockCrowding(host, 10_000);

    deliverContribution(
      [iconItem('a', 2), iconItem('b', 0), iconItem('c', 2), iconItem('d', 1), iconItem('e', 2)],
      contentWindow
    );

    // Lowest priority first (b, then d) until three show.
    expect(hiddenKeys(host)).toEqual(['icon:b', 'icon:d']);
    expect(host.querySelectorAll('.contrib-icon:not(.contrib-hidden)').length).toBe(3);
    expect(panelMenuRows(panelId).map((r) => r.label)).toEqual(['b', 'd']);
  });

  test('crowded, panel icons fold before the shared controls, and those before legacy controls', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const host = document.createElement('div');
    const controls = sharedControls(['ask_agent', 'reload']);
    registerContribHost(panelId, host, controls);
    // Two icons (72) + a legacy button (80) = 152 against 230 - 2×36 = 158: fits.
    mockCrowding(host, 230, controls);
    const items = [iconItem('x', 1), iconItem('y', 0), { kind: 'button', id: 'go', label: 'Go' }];
    deliverContribution(items, contentWindow);
    expect(hiddenKeys(host)).toEqual([]);
    expect(controls.querySelectorAll('.contrib-hidden').length).toBe(0);
    // Both sides have a visible control: the hairline shows.
    expect(controls.querySelector('.tile-tab-divider')?.hasAttribute('hidden')).toBe(false);

    // 150 - 72 = 78 < 152: fold y (priority 0), then x — still 80 > 78 — then
    // the last configured shared control (reload) frees 36: 80 ≤ 114. (The
    // helper lists folded keys in DOM order.)
    mockCrowding(host, 150, controls);
    deliverContribution(items, contentWindow);
    expect(hiddenKeys(host)).toEqual(['icon:x', 'icon:y']);
    expect(
      [...controls.querySelectorAll('[data-control]')].map((c) =>
        c.classList.contains('contrib-hidden')
      )
    ).toEqual([false, true]);
    expect(host.querySelector('.contrib-btn')?.classList.contains('contrib-hidden')).toBe(false);
    // The legacy button still stands left of the hairline, ask_agent right of it.
    expect(controls.querySelector('.tile-tab-divider')?.hasAttribute('hidden')).toBe(false);

    // Narrower still: ask_agent goes, and only then the legacy button.
    mockCrowding(host, 70, controls);
    deliverContribution(items, contentWindow);
    expect(controls.querySelectorAll('[data-control].contrib-hidden').length).toBe(2);
    expect(hiddenKeys(host)).toEqual(['icon:x', 'icon:y', 'button:go']);
    // Nothing on either side: the hairline goes too.
    expect(controls.querySelector('.tile-tab-divider')?.hasAttribute('hidden')).toBe(true);
    // Folded icons are rows after the panel's menu entries, in contribution order.
    expect(panelMenuRows(panelId).map((r) => r.label)).toEqual(['x', 'y', 'Go']);
  });

  test('a shared control the deployment hid (no help page) neither folds nor shows the hairline', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const host = document.createElement('div');
    const controls = sharedControls(['help']);
    /** @type {HTMLElement} */ (controls.querySelector('[data-control]')).hidden = true;
    registerContribHost(panelId, host, controls);
    mockCrowding(host, 10_000);

    deliverContribution([iconItem('a')], contentWindow);

    expect(controls.querySelector('.tile-tab-divider')?.hasAttribute('hidden')).toBe(true);
    expect(controls.querySelectorAll('.contrib-hidden').length).toBe(0);
  });

  test('menu entries are ⋯ rows always, folded or not, with their v3 fields', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const host = document.createElement('div');
    registerContribHost(panelId, host);
    mockCrowding(host, 10_000);

    deliverContribution(
      [
        {
          kind: 'menu',
          id: 'more',
          items: [
            { id: 'compact', label: 'Compact rows', checked: true },
            { id: 'export', label: 'Export CSV', disabled: true },
            { id: 'clear', label: 'Clear all', danger: true, separator: true },
          ],
        },
        iconItem('refresh'),
      ],
      contentWindow
    );

    // The menu holds no visible element: an empty hidden holder keeps its key.
    const holder = /** @type {HTMLElement} */ (
      host.querySelector('[data-contrib-key="menu:more"]')
    );
    expect(holder.hidden).toBe(true);
    expect(holder.childNodes.length).toBe(0);
    expect(host.querySelectorAll('[data-contrib-key]').length).toBe(2);

    const rows = panelMenuRows(panelId);
    expect(rows.every((r) => typeof r.run === 'function')).toBe(true);
    expect(rows.map((r) => ({ ...r, run: undefined }))).toEqual([
      { label: 'Compact rows', checked: true, disabled: false, danger: false, separator: false },
      { label: 'Export CSV', disabled: true, danger: false, separator: false },
      { label: 'Clear all', disabled: false, danger: true, separator: true },
    ]);
    rows[2].run?.();
    expect(contentWindow.postMessage).toHaveBeenCalledWith(
      { type: 'canopy-host-header-action', id: 'more', value: 'clear' },
      window.location.origin
    );
  });
});

describe('host lifecycle', () => {
  test('a rebuilt tab re-renders from the cache; the stale host is dropped', () => {
    const panelId = freshPanelId();
    const { contentWindow } = addPanelIframe(panelId);
    const oldHost = document.createElement('div');
    registerContribHost(panelId, oldHost);
    deliverContribution([{ kind: 'text', id: 'name', text: 'persist' }], contentWindow);

    // Tab rebuild: the replacement registers first, then the old tab disposes.
    const newHost = document.createElement('div');
    registerContribHost(panelId, newHost);
    unregisterContribHost(panelId, oldHost);

    expect(newHost.querySelector('.contrib-text')?.textContent).toBe('persist');

    // A later contribution reaches only the live host.
    deliverContribution([{ kind: 'text', id: 'name', text: 'updated' }], contentWindow);
    expect(newHost.querySelector('.contrib-text')?.textContent).toBe('updated');
    expect(oldHost.querySelector('.contrib-text')?.textContent).toBe('persist');
  });
});
