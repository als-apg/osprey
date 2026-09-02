/**
 * The bar item hosts — the reconcile that MOVES nodes rather than rebuilding
 * them, happy-dom environment (configured globally):
 *   npx vitest run tests/interfaces/web_terminal/js/bar-host.test.js
 *
 * Two of these assertions are the reason the host exists at all, and both are
 * named in the proposal's FR2:
 *
 *   - `document.getElementById('activity-strip')` returns the SAME node object
 *     across host -> pool -> host round-trips. The strip self-boots from its
 *     own module, is an `aria-live` region, and is mutated into its own history
 *     trigger by a third module. Every one of those lookups is null-guarded, so
 *     a rebuild would not throw — it would leave a permanently dead strip and
 *     no error anywhere. Identity is therefore pinned as object identity, not
 *     as "an element with that id still exists".
 *
 *   - a focused input keeps focus AND its selection across a reconcile.
 *     Re-parenting blurs, so a layout arriving while the operator is typing
 *     would silently eat the caret.
 *
 * The rest pins the contract the later tasks build on: order follows the
 * layout, absent items park in the pool rather than being removed, the catalog
 * flex hints reach the shell (this is what makes spacing back-pressure
 * CSS-owned), `isLive()` separates live chrome from pooled chrome, and bodies
 * come from one builder per type.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import {
  closeBarPopovers,
  hasItemBuilder,
  hostElement,
  hydrate,
  isLive,
  poolElement,
  reconcile,
  registerBarPopover,
  registerItemBuilder,
  shellForKey,
} from '../../../../src/osprey/interfaces/web_terminal/static/js/bar-host.js';

/** @typedef {import('../../../../src/osprey/interfaces/web_terminal/static/js/bar-layout.js').BarLayout} BarLayout */

// `import.meta.dirname` is a plain string, so it sidesteps happy-dom's
// override of the global URL (which breaks fileURLToPath(new URL(...)) here).
const BARS_CSS = join(
  import.meta.dirname,
  '../../../../src/osprey/interfaces/web_terminal/static/css/bars.css'
);

/** Unregister callbacks handed back by the two registries. @type {(() => void)[]} */
let cleanups = [];

/**
 * A layout document. Only `header`, `status` and `status_visible` are read by
 * the host; the schema fields are carried so the fixture is a real document.
 * @param {string[]} header
 * @param {string[]} status
 * @param {{statusVisible?: boolean, options?: Record<string, Record<string, string | number | boolean>>}} [extra]
 * @returns {BarLayout}
 */
function layoutOf(header, status, extra = {}) {
  /** @param {string} type */
  const item = (type) => ({ type, options: extra.options?.[type] ?? {} });
  return {
    version: 1,
    rev: 0,
    header: header.map(item),
    status: status.map(item),
    status_visible: extra.statusVisible ?? true,
  };
}

/**
 * The SSR DOM this module hydrates from: both hosts, the hidden pool, and the
 * adopted chrome the server renders literally into its shell.
 * @param {string} [headerShells]
 * @param {string} [statusShells]
 */
function seedDom(headerShells = '', statusShells = '') {
  document.body.innerHTML = `
    <header class="header">
      <div class="header-actions" data-bar-host="header">${headerShells}</div>
    </header>
    <footer class="status-bar" data-bar-host="status">${statusShells}</footer>
    <div id="bar-item-pool" hidden></div>
  `;
  hydrate(document);
}

/** An adopted-chrome shell: a body the server rendered, which must never be rebuilt. */
const ACTIVITY_SHELL =
  '<div class="bar-item" data-bar-item="activity"><div id="activity-strip" aria-live="polite"></div></div>';

/** The docs link, adopted, already revealed — the item hides it again itself. */
const DOCS_SHELL =
  '<div class="bar-item" data-bar-item="docs"><a class="status-link" id="docs-link">Docs</a></div>';

/** The same link as the server actually ships it: hidden until a URL arrives. */
const DOCS_SHELL_HIDDEN =
  '<div class="bar-item" data-bar-item="docs"><a class="status-link" id="docs-link" hidden>Docs</a></div>';

/** The two baseline items, as adopted chrome. */
const LOGO_SHELL =
  '<div class="bar-item" data-bar-item="logo"><span id="wordmark">OSPREY</span></div>';
const IDENTITY_SHELL =
  '<div class="bar-item" data-bar-item="identity"><span id="deployment">Control Room</span></div>';

/** Let the `hidden` MutationObserver callbacks queued by a mutation run. */
function flush() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

/** The generated baseline run of a host, if one exists right now. */
function runIn(/** @type {'header' | 'status'} */ host) {
  return /** @type {HTMLElement | null} */ (
    hostElement(host, document)?.querySelector('.bar-baseline-run') ?? null
  );
}

/** @returns {string[]} the `data-bar-item` types of a host, in DOM order */
function typesIn(/** @type {'header' | 'status'} */ host) {
  const container = hostElement(host, document);
  if (!container) return [];
  return Array.from(container.querySelectorAll('.bar-item')).map(
    (el) => /** @type {HTMLElement} */ (el).dataset.barItem ?? ''
  );
}

/**
 * A shell's flex declaration read back as longhands. The shorthand is asserted
 * through them rather than as a string because the CSSOM re-serializes it (a
 * `0` basis comes back as `0px`), and the longhands are the actual semantics.
 * @param {HTMLElement} shell
 * @returns {[string, string, string]}
 */
function flexOf(shell) {
  return [
    shell.style.getPropertyValue('flex-grow'),
    shell.style.getPropertyValue('flex-shrink'),
    shell.style.getPropertyValue('flex-basis'),
  ];
}

/**
 * Register a builder and remember how to undo it.
 * @param {string} type
 * @param {(ctx: {shell: HTMLElement}) => Node | null} builder
 */
function useBuilder(type, builder) {
  cleanups.push(registerItemBuilder(type, builder));
}

beforeEach(() => {
  seedDom();
});

afterEach(() => {
  for (const undo of cleanups) undo();
  cleanups = [];
  document.body.innerHTML = '';
});

describe('adopted nodes are moved, never rebuilt', () => {
  test('#activity-strip is the same node object across a pool round-trip', () => {
    seedDom(ACTIVITY_SHELL);
    const original = document.getElementById('activity-strip');
    expect(original).toBeTruthy();
    // A marker no builder could reproduce: if the node is ever rebuilt from the
    // type's builder instead of moved, this is what disappears.
    /** @type {any} */ (original).__ospreyLiveRegion = Symbol('live');

    reconcile(layoutOf([], []));
    const parked = document.getElementById('activity-strip');
    expect(parked).toBe(original);
    expect(poolElement(document)?.contains(/** @type {Node} */ (parked))).toBe(true);
    expect(typesIn('header')).toEqual([]);

    reconcile(layoutOf(['activity'], []));
    const back = document.getElementById('activity-strip');
    expect(back).toBe(original);
    expect(/** @type {any} */ (back).__ospreyLiveRegion).toBe(
      /** @type {any} */ (original).__ospreyLiveRegion
    );
    expect(hostElement('header', document)?.contains(/** @type {Node} */ (back))).toBe(true);
  });

  test('a shell parked by one reconcile is reused by the next, not recreated', () => {
    seedDom(ACTIVITY_SHELL);
    const shell = shellForKey('activity');
    expect(shell).toBeTruthy();

    reconcile(layoutOf([], []));
    reconcile(layoutOf(['activity'], []));

    expect(shellForKey('activity')).toBe(shell);
  });

  test('an adopted body is left alone even when the type has a builder', () => {
    seedDom(ACTIVITY_SHELL);
    useBuilder('activity', () => {
      const span = document.createElement('span');
      span.className = 'rebuilt';
      return span;
    });
    const original = document.getElementById('activity-strip');

    reconcile(layoutOf(['activity'], []));

    expect(document.getElementById('activity-strip')).toBe(original);
    expect(document.querySelector('.rebuilt')).toBeNull();
  });
});

describe('focus and selection survive a reconcile', () => {
  test('a focused input keeps focus and its selection range', () => {
    seedDom(
      `<div class="bar-item" data-bar-item="search"><input id="palette-input" type="text"></div>` +
        `<div class="bar-item" data-bar-item="clock"></div>` +
        `<div class="bar-item" data-bar-item="docs"></div>`
    );
    const input = /** @type {HTMLInputElement} */ (document.getElementById('palette-input'));
    input.value = 'quadrupole';
    input.focus();
    input.setSelectionRange(4, 9);
    expect(document.activeElement).toBe(input);

    // Reorder so the search shell has to move: it goes from first to last.
    reconcile(layoutOf(['clock', 'docs', 'search'], []));

    expect(typesIn('header')).toEqual(['clock', 'docs', 'search']);
    expect(document.getElementById('palette-input')).toBe(input);
    expect(document.activeElement).toBe(input);
    expect(input.selectionStart).toBe(4);
    expect(input.selectionEnd).toBe(9);
  });

  test('focus is restored after a cross-host move', () => {
    seedDom(
      `<div class="bar-item" data-bar-item="search"><input id="palette-input" type="text"></div>`,
      `<div class="bar-item" data-bar-item="clock"></div>`
    );
    const input = /** @type {HTMLInputElement} */ (document.getElementById('palette-input'));
    input.value = 'sextupole';
    input.focus();
    input.setSelectionRange(2, 2);

    reconcile(layoutOf(['clock', 'search'], []));

    expect(document.activeElement).toBe(input);
    expect(input.selectionStart).toBe(2);
  });

  test('focus is not forced back onto an item the reconcile parked', () => {
    seedDom(
      `<div class="bar-item" data-bar-item="search"><input id="palette-input" type="text"></div>`
    );
    const input = /** @type {HTMLInputElement} */ (document.getElementById('palette-input'));
    input.focus();

    reconcile(layoutOf([], []));

    expect(isLive(input)).toBe(false);
    expect(document.activeElement).not.toBe(input);
  });
});

describe('order and parking', () => {
  test('shells are placed in the layout order, per host', () => {
    seedDom(
      `<div class="bar-item" data-bar-item="docs"></div>` +
        `<div class="bar-item" data-bar-item="clock"></div>` +
        `<div class="bar-item" data-bar-item="feedback"></div>`
    );

    reconcile(layoutOf(['clock', 'feedback', 'docs'], ['stopwatch']));

    expect(typesIn('header')).toEqual(['clock', 'feedback', 'docs']);
    expect(typesIn('status')).toEqual(['stopwatch']);
  });

  test('an item the layout drops is parked in the pool, not removed', () => {
    seedDom(
      `<div class="bar-item" data-bar-item="clock"></div>` +
        `<div class="bar-item" data-bar-item="docs"></div>`
    );
    const docsShell = shellForKey('docs');

    reconcile(layoutOf(['clock'], []));

    expect(typesIn('header')).toEqual(['clock']);
    expect(docsShell?.isConnected).toBe(true);
    expect(poolElement(document)?.contains(/** @type {Node} */ (docsShell))).toBe(true);
  });

  test('repeated spacing types keep distinct shells', () => {
    reconcile(layoutOf(['gap', 'clock', 'gap'], []));

    expect(typesIn('header')).toEqual(['gap', 'clock', 'gap']);
    expect(shellForKey('gap')).not.toBe(shellForKey('gap#1'));
  });

  test('a non-item child of the host is left in place', () => {
    seedDom(`<span id="stray-chip"></span><div class="bar-item" data-bar-item="clock"></div>`);

    reconcile(layoutOf(['clock'], []));

    expect(document.getElementById('stray-chip')?.isConnected).toBe(true);
    expect(hostElement('header', document)?.contains(document.getElementById('stray-chip'))).toBe(
      true
    );
  });

  test('unknown types in a layout are skipped rather than thrown on', () => {
    expect(() =>
      reconcile(
        /** @type {any} */ ({
          version: 1,
          rev: 0,
          header: [{ type: 'not-a-type', options: {} }, { type: 'clock', options: {} }, null],
          status: [],
          status_visible: true,
        })
      )
    ).not.toThrow();
    expect(typesIn('header')).toEqual(['clock']);
  });

  test('status_visible is mirrored onto the root element', () => {
    reconcile(layoutOf([], [], { statusVisible: false }));
    expect(document.documentElement.dataset.statusBar).toBe('hidden');

    reconcile(layoutOf([], [], { statusVisible: true }));
    expect(document.documentElement.dataset.statusBar).toBe('visible');
  });
});

describe('catalog flex hints are stamped on the shell', () => {
  test('activity absorbs spare space and may ellipsize', () => {
    reconcile(layoutOf(['activity'], []));
    const shell = /** @type {HTMLElement} */ (shellForKey('activity'));
    expect(flexOf(shell)).toEqual(['1', '1', '0px']);
    expect(shell.style.getPropertyValue('min-width')).toBe('0');
  });

  test('a space scales its grow factor by its share option', () => {
    reconcile(layoutOf(['space'], [], { options: { space: { share: 3 } } }));
    expect(flexOf(/** @type {HTMLElement} */ (shellForKey('space')))).toEqual(['3', '1', '0px']);
  });

  test('a gap holds its size until the bar runs out of room', () => {
    reconcile(layoutOf(['gap'], [], { options: { gap: { size: 40 } } }));
    const shell = /** @type {HTMLElement} */ (shellForKey('gap'));
    expect(flexOf(shell)).toEqual(['0', '1', '40px']);
    expect(shell.style.getPropertyValue('min-width')).toBe('0');
  });

  test('a type with no flex hint stamps nothing', () => {
    reconcile(layoutOf(['clock'], []));
    const shell = /** @type {HTMLElement} */ (shellForKey('clock'));
    expect(flexOf(shell)).toEqual(['', '', '']);
    expect(shell.style.getPropertyValue('min-width')).toBe('');
  });

  test('the density of the host it landed in is stamped, and cleared on parking', () => {
    reconcile(layoutOf(['clock'], []));
    expect(/** @type {HTMLElement} */ (shellForKey('clock')).dataset.barDensity).toBe(
      'comfortable'
    );

    reconcile(layoutOf([], ['clock']));
    expect(/** @type {HTMLElement} */ (shellForKey('clock')).dataset.barDensity).toBe('compact');

    reconcile(layoutOf([], []));
    expect(/** @type {HTMLElement} */ (shellForKey('clock')).dataset.barDensity).toBeUndefined();
  });
});

describe('isLive', () => {
  test('true for a node in a host, false for a node in the pool', () => {
    seedDom(ACTIVITY_SHELL);
    const strip = /** @type {HTMLElement} */ (document.getElementById('activity-strip'));

    reconcile(layoutOf(['activity'], []));
    expect(isLive(strip)).toBe(true);

    reconcile(layoutOf([], []));
    expect(isLive(strip)).toBe(false);
  });

  test('false for nothing at all', () => {
    expect(isLive(null)).toBe(false);
    expect(isLive(undefined)).toBe(false);
  });
});

describe('one builder per type', () => {
  test('a registered builder supplies the body at placement', () => {
    useBuilder('clock', ({ shell }) => {
      const span = shell.ownerDocument.createElement('span');
      span.className = 'clock-body';
      return span;
    });

    reconcile(layoutOf(['clock'], []));

    expect(hasItemBuilder('clock')).toBe(true);
    expect(shellForKey('clock')?.querySelector('.clock-body')).toBeTruthy();
  });

  test('a type with no builder yet renders an empty, marked shell', () => {
    reconcile(layoutOf(['stopwatch'], []));
    const shell = /** @type {HTMLElement} */ (shellForKey('stopwatch'));

    expect(shell.children.length).toBe(0);
    expect(shell.dataset.barUnbuilt).toBe('true');
  });

  test('registering a builder fills the shells already hydrated for that type', () => {
    seedDom(`<div class="bar-item" data-bar-item="clock"></div>`);
    expect(/** @type {HTMLElement} */ (shellForKey('clock')).dataset.barUnbuilt).toBe('true');

    useBuilder('clock', () => {
      const span = document.createElement('span');
      span.className = 'clock-body';
      return span;
    });

    expect(shellForKey('clock')?.querySelector('.clock-body')).toBeTruthy();
    expect(/** @type {HTMLElement} */ (shellForKey('clock')).dataset.barUnbuilt).toBeUndefined();
  });

  test('a builder is not re-run when nothing about the placement changed', () => {
    const builder = vi.fn(() => document.createElement('span'));
    useBuilder('clock', builder);

    reconcile(layoutOf(['clock'], []));
    reconcile(layoutOf(['clock'], []));

    expect(builder).toHaveBeenCalledTimes(1);
  });

  test('a builder re-runs when the item changes host or options', () => {
    const builder = vi.fn(() => document.createElement('span'));
    useBuilder('gap', builder);

    reconcile(layoutOf(['gap'], [], { options: { gap: { size: 12 } } }));
    reconcile(layoutOf(['gap'], [], { options: { gap: { size: 40 } } }));
    reconcile(layoutOf([], ['gap'], { options: { gap: { size: 40 } } }));

    expect(builder).toHaveBeenCalledTimes(3);
  });

  test('a throwing builder does not take the rest of the bar down with it', () => {
    useBuilder('clock', () => {
      throw new Error('builder is broken');
    });
    const errors = vi.spyOn(console, 'error').mockImplementation(() => {});

    expect(() => reconcile(layoutOf(['clock', 'docs'], []))).not.toThrow();
    expect(typesIn('header')).toEqual(['clock', 'docs']);

    errors.mockRestore();
  });
});

describe('popovers close before their item moves', () => {
  test('a registered closer runs when the reconcile moves the shell', () => {
    seedDom(
      `<div class="bar-item" data-bar-item="clock"></div>` +
        `<div class="bar-item" data-bar-item="docs"></div>`
    );
    const close = vi.fn();
    cleanups.push(registerBarPopover(/** @type {HTMLElement} */ (shellForKey('docs')), close));

    reconcile(layoutOf(['docs', 'clock'], []));

    expect(close).toHaveBeenCalled();
  });

  test('a closer runs when the reconcile parks the shell', () => {
    seedDom(`<div class="bar-item" data-bar-item="docs"></div>`);
    const close = vi.fn();
    cleanups.push(registerBarPopover(/** @type {HTMLElement} */ (shellForKey('docs')), close));

    reconcile(layoutOf([], []));

    expect(close).toHaveBeenCalled();
  });

  test('a shell that does not move is left open', () => {
    seedDom(
      `<div class="bar-item" data-bar-item="clock"></div>` +
        `<div class="bar-item" data-bar-item="docs"></div>`
    );
    const close = vi.fn();
    cleanups.push(registerBarPopover(/** @type {HTMLElement} */ (shellForKey('clock')), close));

    reconcile(layoutOf(['clock', 'docs'], []));

    expect(close).not.toHaveBeenCalled();
  });

  test('closeBarPopovers closes every registered popover', () => {
    seedDom(`<div class="bar-item" data-bar-item="docs"></div>`);
    const close = vi.fn();
    cleanups.push(registerBarPopover(/** @type {HTMLElement} */ (shellForKey('docs')), close));

    closeBarPopovers();

    expect(close).toHaveBeenCalledTimes(1);
  });
});

describe('first paint', () => {
  test('hydration is synchronous and indexes the server-rendered shells', () => {
    seedDom(ACTIVITY_SHELL + `<div class="bar-item" data-bar-item="clock"></div>`);

    // No reconcile has run: everything below is what the SSR pass alone gives.
    expect(shellForKey('activity')).toBeTruthy();
    expect(shellForKey('clock')).toBeTruthy();
    expect(/** @type {HTMLElement} */ (shellForKey('activity')).dataset.barAdopted).toBe('true');
    expect(/** @type {HTMLElement} */ (shellForKey('clock')).dataset.barAdopted).toBeUndefined();
    expect(flexOf(/** @type {HTMLElement} */ (shellForKey('activity')))).toEqual([
      '1',
      '1',
      '0px',
    ]);
  });

  test('shells parked in the pool by the server are hydrated too', () => {
    document.body.innerHTML = `
      <header class="header"><div class="header-actions" data-bar-host="header"></div></header>
      <footer class="status-bar" data-bar-host="status"></footer>
      <div id="bar-item-pool" hidden>${ACTIVITY_SHELL}</div>
    `;
    hydrate(document);

    const strip = document.getElementById('activity-strip');
    expect(strip).toBeTruthy();
    expect(isLive(strip)).toBe(false);

    reconcile(layoutOf(['activity'], []));

    expect(document.getElementById('activity-strip')).toBe(strip);
    expect(isLive(strip)).toBe(true);
  });

  test('the module makes no network call on its import path', async () => {
    // GET /api/bar-items is a reconcile INPUT. A fetch anywhere in this import
    // path would make the header and the status bar wait on the network before
    // they could paint at all, so importing the module afresh — hydration and
    // everything else it does at module scope — must touch no transport.
    seedDom(ACTIVITY_SHELL);
    const fetchSpy = vi.fn();
    const originalFetch = globalThis.fetch;
    globalThis.fetch = /** @type {typeof globalThis.fetch} */ (fetchSpy);
    try {
      vi.resetModules();
      await import('../../../../src/osprey/interfaces/web_terminal/static/js/bar-host.js');
      expect(fetchSpy).not.toHaveBeenCalled();
    } finally {
      globalThis.fetch = originalFetch;
    }
  });
});

describe('an item that hides its body collapses its shell', () => {
  test('hiding the inner docs link hides the shell around it', async () => {
    seedDom(DOCS_SHELL);
    reconcile(layoutOf(['docs'], []));
    const shell = /** @type {HTMLElement} */ (shellForKey('docs'));
    const link = /** @type {HTMLElement} */ (document.getElementById('docs-link'));
    expect(shell.hidden).toBe(false);

    // What feedback-boot.js does to a deployment that configures no docs URL.
    link.hidden = true;
    await flush();
    expect(shell.hidden).toBe(true);

    link.hidden = false;
    await flush();
    expect(shell.hidden).toBe(false);
  });

  test('a body the server ships hidden collapses its shell at hydration', () => {
    seedDom(DOCS_SHELL_HIDDEN);

    // No reconcile, no mutation: the mirror states the SSR fact immediately, so
    // the shell never paints a gap around a link that is not there yet.
    expect(/** @type {HTMLElement} */ (shellForKey('docs')).hidden).toBe(true);
  });

  test('the mirror still fires after a host -> pool -> host round trip', async () => {
    seedDom(DOCS_SHELL);
    reconcile(layoutOf(['docs'], []));
    const shell = /** @type {HTMLElement} */ (shellForKey('docs'));
    const link = /** @type {HTMLElement} */ (document.getElementById('docs-link'));

    reconcile(layoutOf([], []));
    expect(isLive(shell)).toBe(false);
    reconcile(layoutOf(['docs'], []));
    expect(isLive(shell)).toBe(true);

    link.hidden = true;
    await flush();
    expect(shell.hidden).toBe(true);
  });

  test('an item hidden while it is parked comes back collapsed', async () => {
    seedDom(DOCS_SHELL);
    reconcile(layoutOf(['docs'], []));
    const shell = /** @type {HTMLElement} */ (shellForKey('docs'));
    const link = /** @type {HTMLElement} */ (document.getElementById('docs-link'));

    reconcile(layoutOf([], []));
    link.hidden = true;
    await flush();
    reconcile(layoutOf(['docs'], []));

    expect(shell.hidden).toBe(true);
  });

  test('a rebuilt body takes the mirror over from the one it replaced', async () => {
    useBuilder('gap', () => document.createElement('span'));
    reconcile(layoutOf(['gap'], [], { options: { gap: { size: 12 } } }));
    const shell = /** @type {HTMLElement} */ (shellForKey('gap'));
    const first = /** @type {HTMLElement} */ (shell.firstElementChild);

    reconcile(layoutOf(['gap'], [], { options: { gap: { size: 40 } } }));
    const second = /** @type {HTMLElement} */ (shell.firstElementChild);
    expect(second).not.toBe(first);

    // One observer per shell: the body that was replaced no longer speaks for it.
    first.hidden = true;
    await flush();
    expect(shell.hidden).toBe(false);

    second.hidden = true;
    await flush();
    expect(shell.hidden).toBe(true);
  });

  test('a shell of several dots collapses only once every dot is hidden', async () => {
    // The panel-health shape: one `.status-item` per declaring panel, each
    // revealed by panel-status-bar.js when that panel's own config lands.
    seedDom(
      '<div class="bar-item" data-bar-item="panel-health">' +
        '<div class="status-item" id="dot-a" hidden></div>' +
        '<div class="status-item" id="dot-b" hidden></div>' +
        '</div>'
    );
    const shell = /** @type {HTMLElement} */ (shellForKey('panel-health'));
    expect(shell.hidden).toBe(true);

    const first = /** @type {HTMLElement} */ (document.getElementById('dot-a'));
    first.hidden = false;
    await flush();
    expect(shell.hidden).toBe(false);

    first.hidden = true;
    await flush();
    expect(shell.hidden).toBe(true);
  });

  test('an empty shell keeps its own visibility', () => {
    reconcile(layoutOf(['gap', 'space'], []));
    expect(/** @type {HTMLElement} */ (shellForKey('gap')).hidden).toBe(false);
    expect(/** @type {HTMLElement} */ (shellForKey('space')).hidden).toBe(false);
  });
});

describe('baseline runs form and dissolve as items move', () => {
  test('a run wrapper spans logo and identity', () => {
    seedDom(LOGO_SHELL + IDENTITY_SHELL);

    reconcile(layoutOf(['logo', 'identity'], []));

    const run = runIn('header');
    expect(run).toBeTruthy();
    expect(run?.className).toBe('bar-baseline-run');
    expect(Array.from(run?.children ?? [])).toEqual([shellForKey('logo'), shellForKey('identity')]);
    expect(typesIn('header')).toEqual(['logo', 'identity']);
  });

  test('the run dissolves when identity moves away from the logo', () => {
    seedDom(LOGO_SHELL + IDENTITY_SHELL);
    reconcile(layoutOf(['logo', 'identity'], []));
    expect(runIn('header')).toBeTruthy();

    reconcile(layoutOf(['logo', 'clock', 'identity'], []));

    expect(runIn('header')).toBeNull();
    expect(typesIn('header')).toEqual(['logo', 'clock', 'identity']);
    const host = hostElement('header', document);
    expect(shellForKey('logo')?.parentElement).toBe(host);
    expect(shellForKey('identity')?.parentElement).toBe(host);
  });

  test('the adopted bodies are moved into and out of the run, never rebuilt', () => {
    seedDom(LOGO_SHELL + IDENTITY_SHELL);
    const wordmark = document.getElementById('wordmark');

    reconcile(layoutOf(['logo', 'identity'], []));
    expect(document.getElementById('wordmark')).toBe(wordmark);
    reconcile(layoutOf(['identity', 'clock', 'logo'], []));

    expect(document.getElementById('wordmark')).toBe(wordmark);
    expect(typesIn('header')).toEqual(['identity', 'clock', 'logo']);
  });

  test('a lone baseline item is not wrapped', () => {
    seedDom(LOGO_SHELL);

    reconcile(layoutOf(['logo', 'clock'], []));

    expect(runIn('header')).toBeNull();
    expect(shellForKey('logo')?.parentElement).toBe(hostElement('header', document));
  });

  test('an unchanged run keeps its wrapper and moves nothing inside it', () => {
    seedDom(LOGO_SHELL + IDENTITY_SHELL);
    reconcile(layoutOf(['logo', 'identity'], []));
    const run = runIn('header');
    const close = vi.fn();
    cleanups.push(registerBarPopover(/** @type {HTMLElement} */ (shellForKey('identity')), close));

    reconcile(layoutOf(['logo', 'identity', 'clock'], []));

    expect(runIn('header')).toBe(run);
    expect(close).not.toHaveBeenCalled();
    expect(typesIn('header')).toEqual(['logo', 'identity', 'clock']);
  });

  test('a run that follows another item is placed as a whole', () => {
    seedDom(LOGO_SHELL + IDENTITY_SHELL);

    reconcile(layoutOf(['clock', 'logo', 'identity', 'docs'], []));

    const host = /** @type {HTMLElement} */ (hostElement('header', document));
    const run = runIn('header');
    expect(Array.from(host.children)).toEqual([shellForKey('clock'), run, shellForKey('docs')]);
    expect(typesIn('header')).toEqual(['clock', 'logo', 'identity', 'docs']);
  });

  test('parking a run parks its items and drops the emptied wrapper', () => {
    seedDom(LOGO_SHELL + IDENTITY_SHELL);
    reconcile(layoutOf(['logo', 'identity'], []));

    reconcile(layoutOf([], []));

    expect(runIn('header')).toBeNull();
    expect(hostElement('header', document)?.children.length).toBe(0);
    for (const key of ['logo', 'identity']) {
      const shell = /** @type {HTMLElement} */ (shellForKey(key));
      expect(poolElement(document)?.contains(shell)).toBe(true);
      expect(shell.dataset.barDensity).toBeUndefined();
    }
  });

  test('a server-rendered run wrapper is hydrated and then reused', () => {
    seedDom(`<div class="bar-baseline-run">${LOGO_SHELL}${IDENTITY_SHELL}</div>`);
    expect(shellForKey('logo')).toBeTruthy();
    expect(/** @type {HTMLElement} */ (shellForKey('logo')).dataset.barAdopted).toBe('true');
    const run = runIn('header');

    reconcile(layoutOf(['logo', 'identity'], []));

    expect(runIn('header')).toBe(run);
  });
});

describe('a shell this module builds is styled like one the server rendered', () => {
  /**
   * Every `.bar-item[data-…="<type>"]` selector bars.css uses for one item
   * type, read from the shipped stylesheet so the pin cannot drift from it.
   * @param {string} type
   * @returns {string[]}
   */
  function styledBy(type) {
    const css = readFileSync(BARS_CSS, 'utf8');
    const pattern = new RegExp(String.raw`\.bar-item\[data-[\w-]+="${type}"\]`, 'g');
    return [...new Set(Array.from(css.matchAll(pattern), (match) => match[0]))];
  }

  test('a space placed at runtime matches the stylesheet selectors, with no seeded shell', () => {
    // Nothing is seeded, so `ensureShell` builds this one — the path a drag-in
    // of a type the server never rendered takes. It is the whole point: a
    // spacing item styled only through an attribute the SSR alone stamped
    // would lose its width and its edit-mode glyph until the next reload.
    reconcile(layoutOf(['space'], []));
    const shell = /** @type {HTMLElement} */ (shellForKey('space'));

    const selectors = styledBy('space');
    expect(selectors.length, 'bars.css styles no space item').toBeGreaterThan(0);
    for (const selector of selectors) {
      expect(shell.matches(selector), `a client-built space does not match ${selector}`).toBe(true);
    }
  });

  test('a gap placed at runtime matches its stylesheet selectors too', () => {
    reconcile(layoutOf(['gap'], []));
    const shell = /** @type {HTMLElement} */ (shellForKey('gap'));

    for (const selector of styledBy('gap')) {
      expect(shell.matches(selector), `a client-built gap does not match ${selector}`).toBe(true);
    }
  });
});

describe('data-follows carries the preceding item', () => {
  test('identity stamps data-follows="logo" only while it sits after the logo', () => {
    seedDom(LOGO_SHELL + IDENTITY_SHELL);

    reconcile(layoutOf(['logo', 'identity'], []));
    expect(/** @type {HTMLElement} */ (shellForKey('identity')).dataset.follows).toBe('logo');

    reconcile(layoutOf(['identity', 'logo'], []));
    expect(/** @type {HTMLElement} */ (shellForKey('identity')).dataset.follows).toBeUndefined();
    expect(/** @type {HTMLElement} */ (shellForKey('logo')).dataset.follows).toBe('identity');
  });

  test('a parked item follows nothing', () => {
    seedDom(LOGO_SHELL + IDENTITY_SHELL);
    reconcile(layoutOf(['logo', 'identity'], []));

    reconcile(layoutOf(['logo'], []));

    expect(/** @type {HTMLElement} */ (shellForKey('identity')).dataset.follows).toBeUndefined();
  });
});
