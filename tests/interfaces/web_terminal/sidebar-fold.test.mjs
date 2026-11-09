// @ts-check
/**
 * Folding a sidebar group (sidebar-fold.js):
 *   npx vitest run tests/interfaces/web_terminal/sidebar-fold.test.mjs
 *
 * A double-click on a heading folds and unfolds its group and persists the
 * ids under the scoped `osprey-sidebar-folded` key; the name's keys do the
 * same, and so does a click on the heading's chevron; the ⋯ button and a
 * rename field do not. Folding keeps the scroll
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
        <input type="text" id="field">
      </div>
      <div class="sidebar-row"><button id="sidebar-agent-row">Agent</button></div>
    </div>
    <div class="sidebar-group" data-group="tools">
      <div class="osprey-section-label sidebar-group-head">
        <span class="sidebar-group-name" id="n2">Apps</span>
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
  return fold;
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
    localStorage.setItem(FOLDED, JSON.stringify(['tools']));
    await boot();
    expect(group('tools').hasAttribute('data-folded')).toBe(true);
    expect(group('workspace').hasAttribute('data-folded')).toBe(false);
  });

  test('the headless Sessions group never folds, whatever is stored', async () => {
    localStorage.setItem(FOLDED, JSON.stringify(['workspace']));
    await boot();
    expect(group('workspace').hasAttribute('data-folded')).toBe(false);
    dbl(head('workspace'));
    expect(group('workspace').hasAttribute('data-folded')).toBe(false);
    expect(stored(FOLDED)).toEqual(['workspace']);
  });

  test('each heading carries one chevron, kept across folds and repaints', async () => {
    await boot();
    for (const id of ['workspace', 'tools']) {
      const chevrons = head(id).querySelectorAll(':scope > .sidebar-group-chevron');
      expect(chevrons).toHaveLength(1);
      // The last child of the heading, hidden from assistive technology: the
      // name's aria-expanded carries the state.
      expect(head(id).lastElementChild).toBe(chevrons[0]);
      expect(chevrons[0].getAttribute('aria-hidden')).toBe('true');
    }
    dbl(head('tools'));
    dbl(head('tools'));
    document.dispatchEvent(new CustomEvent('osprey-sidebar-rendered'));
    expect(head('tools').querySelectorAll('.sidebar-group-chevron')).toHaveLength(1);
  });

  test('a heading rebuilt by a render gets its chevron back', async () => {
    const { SIDEBAR_RENDERED_EVENT } = await import(`${JS}/panel-rail.js`);
    await boot();
    head('tools').replaceChildren(name('tools'));
    document.dispatchEvent(new CustomEvent(SIDEBAR_RENDERED_EVENT));
    expect(head('tools').querySelector('.sidebar-group-chevron')).not.toBeNull();
  });

  test('a click on the chevron folds, another unfolds, and it persists', async () => {
    await boot();
    const chevron = () => qs(head('tools'), '.sidebar-group-chevron');
    chevron().dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    expect(group('tools').hasAttribute('data-folded')).toBe(true);
    expect(name('tools').getAttribute('aria-expanded')).toBe('false');
    expect(stored(FOLDED)).toEqual(['tools']);
    chevron().dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    expect(group('tools').hasAttribute('data-folded')).toBe(false);
    expect(stored(FOLDED)).toEqual([]);
  });

  test('a double-click on the chevron toggles twice, never a third time', async () => {
    await boot();
    const chevron = qs(head('tools'), '.sidebar-group-chevron');
    // A browser fires click, click, dblclick for a double-click.
    for (const type of ['click', 'click', 'dblclick']) {
      chevron.dispatchEvent(new MouseEvent(type, { bubbles: true, cancelable: true }));
    }
    expect(group('tools').hasAttribute('data-folded')).toBe(false);
  });

  test('a click on the heading outside the chevron does not fold', async () => {
    await boot();
    name('tools').dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    expect(group('tools').hasAttribute('data-folded')).toBe(false);
  });

  test('a click on the chevron with the sidebar collapsed to icons does not fold', async () => {
    await boot();
    document.documentElement.setAttribute('data-sidebar', 'collapsed');
    qs(head('tools'), '.sidebar-group-chevron').dispatchEvent(
      new MouseEvent('click', { bubbles: true, cancelable: true })
    );
    expect(group('tools').hasAttribute('data-folded')).toBe(false);
  });

  test('the spacer is the exact scroll shortfall, recomputed and never accumulated', async () => {
    await boot();
    // Layout is stubbed: clientHeight 531, content 664 open / 464 with Apps folded.
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

  test('a rename field does not fold', async () => {
    await boot();
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

describe('setGroupFolded', () => {
  test('folds and unfolds by id, persists, and fires the fold event', async () => {
    const fold = await boot();
    const seen = vi.fn();
    document.addEventListener('osprey-sidebar-fold-change', seen);
    fold.setGroupFolded('tools', true);
    expect(group('tools').hasAttribute('data-folded')).toBe(true);
    expect(name('tools').getAttribute('aria-expanded')).toBe('false');
    expect(fold.isGroupFolded('tools')).toBe(true);
    expect(stored(FOLDED)).toEqual(['tools']);
    expect(seen).toHaveBeenCalledTimes(1);
    fold.setGroupFolded('tools', false);
    expect(group('tools').hasAttribute('data-folded')).toBe(false);
    expect(fold.isGroupFolded('tools')).toBe(false);
    expect(stored(FOLDED)).toEqual([]);
    expect(seen).toHaveBeenCalledTimes(2);
    document.removeEventListener('osprey-sidebar-fold-change', seen);
  });

  test('a state already held is a no-op: no write, no event', async () => {
    const fold = await boot();
    const seen = vi.fn();
    document.addEventListener('osprey-sidebar-fold-change', seen);
    fold.setGroupFolded('tools', false);
    expect(localStorage.getItem(FOLDED)).toBeNull();
    fold.setGroupFolded('tools', true);
    fold.setGroupFolded('tools', true);
    expect(stored(FOLDED)).toEqual(['tools']);
    expect(seen).toHaveBeenCalledTimes(1);
    document.removeEventListener('osprey-sidebar-fold-change', seen);
  });

  test('the headless group never folds', async () => {
    const fold = await boot();
    fold.setGroupFolded('workspace', true);
    expect(group('workspace').hasAttribute('data-folded')).toBe(false);
    expect(localStorage.getItem(FOLDED)).toBeNull();
  });

  test("a double-click after the seam toggles from the seam's state", async () => {
    const fold = await boot();
    fold.setGroupFolded('tools', true);
    dbl(name('tools'));
    expect(group('tools').hasAttribute('data-folded')).toBe(false);
    expect(stored(FOLDED)).toEqual([]);
  });

  test('is a no-op before initSidebarFold wires a rail', async () => {
    document.body.innerHTML = HTML;
    vi.resetModules();
    const fold = await import(`${JS}/sidebar-fold.js`);
    expect(() => fold.setGroupFolded('tools', true)).not.toThrow();
    expect(group('tools').hasAttribute('data-folded')).toBe(false);
    expect(localStorage.getItem(FOLDED)).toBeNull();
  });
});
