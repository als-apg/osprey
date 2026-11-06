// @ts-check
/**
 * Folding a sidebar group (sidebar-fold.js):
 *   npx vitest run tests/interfaces/web_terminal/sidebar-fold.test.mjs
 *
 * A double-click on a heading folds and unfolds its group and persists the
 * ids under the scoped `osprey-sidebar-folded` key; the name's keys do the
 * same, the ⋯ button and a rename field do not. Folding keeps the scroll
 * position with an exact spacer. Layout does not exist in the test DOM, so
 * heights are stubbed.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import { byId, qs } from '../_support/dom.mjs';

const JS = '../../../src/osprey/interfaces/web_terminal/static/js';
const FOLDED = 'osprey-sidebar-folded';

const HTML = `
  <div id="panel-rail">
    <div class="sidebar-group" data-group="workspace">
      <div class="osprey-section-label sidebar-group-head">
        <span class="sidebar-group-name" id="n1">Workspace</span>
        <button type="button" data-group-menu>m</button>
        <input type="text" id="field">
      </div>
      <div class="sidebar-row"><button id="sidebar-agent-row">Agent</button></div>
    </div>
    <div class="sidebar-group" data-group="tools">
      <div class="osprey-section-label sidebar-group-head">
        <span class="sidebar-group-name" id="n2">Tools</span>
        <button type="button" data-group-menu>m</button>
      </div>
    </div>
  </div>`;

const rail = () => byId('panel-rail');
const group = (/** @type {string} */ id) => qs(rail(), `[data-group="${id}"]`);
const head = (/** @type {string} */ id) => qs(group(id), '.sidebar-group-head');
const name = (/** @type {string} */ id) => qs(group(id), '.sidebar-group-name');
const stored = (/** @type {string} */ key) => JSON.parse(localStorage.getItem(key) ?? 'null');

/** @param {Element} el */
const dbl = (el) =>
  el.dispatchEvent(new MouseEvent('dblclick', { bubbles: true, cancelable: true }));

/** @param {Element} el @param {string} key @param {boolean} [shift] */
function press(el, key, shift = false) {
  el.dispatchEvent(
    new KeyboardEvent('keydown', { key, shiftKey: shift, bubbles: true, cancelable: true })
  );
}

/** Draw the rail and wire the fold module. */
async function boot() {
  document.body.innerHTML = HTML;
  Object.defineProperty(rail(), 'clientHeight', { value: 600, configurable: true });
  vi.resetModules();
  const fold = await import(`${JS}/sidebar-fold.js`);
  fold.initSidebarFold(rail());
}

beforeEach(() => {
  localStorage.clear();
  document.documentElement.removeAttribute('data-sidebar');
});

afterEach(() => {
  vi.restoreAllMocks();
  document.body.innerHTML = '';
});

describe('folding a group', () => {
  test('a double-click on the heading folds, another unfolds, and it persists', async () => {
    await boot();
    expect(group('tools').hasAttribute('data-folded')).toBe(false);
    dbl(name('tools'));
    expect(group('tools').hasAttribute('data-folded')).toBe(true);
    expect(name('tools').getAttribute('aria-expanded')).toBe('false');
    expect(stored(FOLDED)).toEqual(['tools']);
    dbl(head('tools'));
    expect(group('tools').hasAttribute('data-folded')).toBe(false);
    expect(name('tools').getAttribute('aria-expanded')).toBe('true');
    expect(stored(FOLDED)).toEqual([]);
  });

  test('the folded state is restored on the next boot', async () => {
    localStorage.setItem(FOLDED, JSON.stringify(['workspace']));
    await boot();
    expect(group('workspace').hasAttribute('data-folded')).toBe(true);
    expect(group('tools').hasAttribute('data-folded')).toBe(false);
  });

  test('headings carry no chevron, and a stale one is removed', async () => {
    const stale = Object.assign(document.createElement('span'), {
      className: 'sidebar-group-chevron',
    });
    await boot();
    head('tools').prepend(stale);
    dbl(head('tools'));
    expect(rail().querySelector('.sidebar-group-chevron')).toBeNull();
  });

  test('the spacer is the exact scroll shortfall, recomputed and never accumulated', async () => {
    await boot();
    // Layout is stubbed: clientHeight 531, content 664 open / 464 with Tools folded.
    // scrollHeight never reads below clientHeight, as in a browser.
    let top = 133;
    const spacer = () => rail().querySelector('.sidebar-fold-spacer');
    const px = () => parseFloat(/** @type {HTMLElement | null} */ (spacer())?.style.height || '0');
    const content = () => (group('tools').hasAttribute('data-folded') ? 464 : 664);
    Object.defineProperty(rail(), 'clientHeight', { configurable: true, get: () => 531 });
    Object.defineProperty(rail(), 'scrollHeight', {
      configurable: true,
      get: () => Math.max(content() + px(), 531),
    });
    Object.defineProperty(rail(), 'scrollTop', {
      configurable: true,
      get: () => top,
      // The browser clamps to the scroll range.
      set: (v) => {
        top = Math.min(v, rail().scrollHeight - 531);
      },
    });
    // The spacer sits at the content end (its own height is not part of it).
    const rect = vi.spyOn(Element.prototype, 'getBoundingClientRect').mockImplementation(
      /** @this {Element} */ function () {
        const el = this;
        const t = el === rail() ? 0 : el === spacer() ? content() - top : 0;
        return /** @type {DOMRect} */ ({ top: t });
      }
    );
    try {
      // Invariant: scrollTop is held and spacer == max(0, held + clientHeight - content).
      for (let cycle = 0; cycle < 3; cycle++) {
        dbl(head('tools')); // fold
        expect(top).toBe(133);
        expect(px()).toBe(200);
        rail().dispatchEvent(new Event('scroll')); // the follow-up scroll event keeps it
        expect(px()).toBe(200);
        dbl(head('tools')); // unfold
        expect(top).toBe(133);
        expect(px()).toBe(0);
      }
      dbl(head('tools'));
      // Scrolling up shrinks it to the new shortfall; at the top it only fills the
      // viewport (464 + 67 = 531), so there is nothing left to scroll.
      top = 233;
      rail().dispatchEvent(new Event('scroll'));
      expect(px()).toBe(300);
      top = 0;
      rail().dispatchEvent(new Event('scroll'));
      expect(px()).toBe(67);
    } finally {
      rect.mockRestore();
    }
  });

  test('Enter and Space on the name toggle', async () => {
    await boot();
    press(name('tools'), 'Enter');
    expect(group('tools').hasAttribute('data-folded')).toBe(true);
    press(name('tools'), ' ');
    expect(group('tools').hasAttribute('data-folded')).toBe(false);
  });

  test('the ⋯ button and a rename field do not fold', async () => {
    await boot();
    dbl(qs(head('workspace'), '[data-group-menu]'));
    dbl(byId('field'));
    expect(group('workspace').hasAttribute('data-folded')).toBe(false);
    expect(localStorage.getItem(FOLDED)).toBeNull();
  });

  test('a double-click on a row does not fold', async () => {
    await boot();
    dbl(byId('sidebar-agent-row'));
    expect(group('workspace').hasAttribute('data-folded')).toBe(false);
  });

  test('nothing folds with the sidebar collapsed to icons', async () => {
    await boot();
    document.documentElement.setAttribute('data-sidebar', 'collapsed');
    dbl(name('tools'));
    expect(group('tools').hasAttribute('data-folded')).toBe(false);
  });

  test('unavailable storage still folds, and a bad value is ignored', async () => {
    localStorage.setItem(FOLDED, '{not json');
    await boot();
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('quota');
    });
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new Error('denied');
    });
    dbl(name('tools'));
    expect(group('tools').hasAttribute('data-folded')).toBe(true);
  });
});
