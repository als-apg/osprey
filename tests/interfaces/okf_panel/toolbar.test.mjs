// @ts-check
/**
 * The OKF Knowledge Panel's toolbar (toolbar.js): the breadcrumb, the
 * bundle-health chip and, standalone, the panel's own ⋯:
 *   npx vitest run tests/interfaces/okf_panel/toolbar.test.mjs
 *
 * Each test mounts the real page shell and imports a fresh module, as a page
 * load would.
 */

import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';

import { mountShell } from './page-shell.mjs';

/** @typedef {typeof import('../../../src/osprey/interfaces/okf_panel/static/js/toolbar.js')} Toolbar */

const MENU = { id: 'knowledge', items: [{ id: 'reset-width', label: 'Reset contents width' }] };

/** @type {import('vitest').Mock<(group: string | null) => void>} */
let onBrowse;
/** @type {import('vitest').Mock<(menuId: string, entryId: string) => void>} */
let onMenu;

/**
 * A page load at `url`: the shell, a fresh module, the toolbar wired.
 * @param {string} [url]
 * @returns {Promise<Toolbar>}
 */
async function load(url = '/') {
  history.replaceState(null, '', url);
  mountShell();
  vi.resetModules();
  const toolbar = await import('../../../src/osprey/interfaces/okf_panel/static/js/toolbar.js');
  toolbar.initToolbar({ onBrowse, menu: MENU, onMenu });
  return toolbar;
}

/** @param {string} id @returns {HTMLElement} */
const $ = (id) => /** @type {HTMLElement} */ (document.getElementById(id));

/** The crumbs' texts, in order. @returns {string[]} */
function crumbs() {
  return Array.from($('crumb').querySelectorAll(':scope > :not(.osprey-crumb-sep)')).map(
    (crumb) => crumb.textContent ?? ''
  );
}

beforeEach(() => {
  onBrowse = vi.fn();
  onMenu = vi.fn();
});

afterEach(() => {
  document.querySelectorAll('.osprey-popover, .osprey-context-menu').forEach((el) => el.remove());
});

describe('the breadcrumb', () => {
  test('on the overview it is Browse alone, the current place, not a link', async () => {
    const toolbar = await load();
    toolbar.showCrumb(null);
    expect(crumbs()).toEqual(['Browse']);
    expect($('crumb').querySelector('a')).toBeNull();
    expect($('crumb').querySelector('[aria-current="page"]')?.textContent).toBe('Browse');
  });

  test('on a page it is Browse › group › concept; Browse and the group open the overview', async () => {
    const toolbar = await load();
    toolbar.showCrumb({ group: 'Procedures', title: 'Orbit Correction' });
    expect(crumbs()).toEqual(['Browse', 'Procedures', 'Orbit Correction']);
    expect($('crumb').querySelector('[aria-current="page"]')?.textContent).toBe('Orbit Correction');

    const [browse, group] = Array.from($('crumb').querySelectorAll('a'));
    const click = new MouseEvent('click', { bubbles: true, cancelable: true });
    browse.dispatchEvent(click);
    expect(click.defaultPrevented).toBe(true);
    expect(onBrowse).toHaveBeenLastCalledWith(null);
    group.click();
    expect(onBrowse).toHaveBeenLastCalledWith('Procedures');
  });

  test('a page with no group has no group crumb', async () => {
    const toolbar = await load();
    toolbar.showCrumb({ group: null, title: 'Glossary' });
    expect(crumbs()).toEqual(['Browse', 'Glossary']);
  });
});

describe('the bundle-health chip', () => {
  test('says what the check found, and what its detail is', async () => {
    const { healthChipView } = await load();
    expect(healthChipView({ state: 'checking' })).toEqual({
      tone: 'connecting',
      label: 'Checking bundle',
      detail: null,
    });
    expect(healthChipView({ state: 'checked', health: { ok: true, total: 0 } })).toMatchObject({
      tone: 'ok',
      label: 'Bundle healthy',
    });
    const failed = healthChipView({ state: 'failed', reason: 'not configured (HTTP 503)' });
    expect(failed).toMatchObject({ tone: 'err', label: 'Check failed' });
    expect(failed.detail?.text).toContain('not configured (HTTP 503)');
  });

  test('issues are counted, and each one is a row: concept, kind, detail', async () => {
    const { healthChipView } = await load();
    const view = healthChipView({
      state: 'checked',
      health: {
        ok: false,
        total: 2,
        warnings: [
          {
            concept_id: 'devices/bpm',
            kind: 'broken-link',
            detail: "'x.md' -> x (no such concept)",
          },
          { concept_id: 'physics/tune', kind: 'frontmatter', detail: 'missing description' },
        ],
      },
    });
    expect(view.tone).toBe('warn');
    expect(view.label).toBe('2 issues');
    expect(view.detail?.rows).toEqual([
      { label: 'devices/bpm', value: "Broken link: 'x.md' -> x (no such concept)" },
      { label: 'physics/tune', value: 'Frontmatter: missing description' },
    ]);
    const one = healthChipView({
      state: 'checked',
      health: { ok: false, total: 1, warnings: [{ concept_id: 'a', kind: 'b', detail: 'c' }] },
    });
    expect(one.label).toBe('1 issue');
  });

  test('is disabled while the check runs, then opens its detail in an info popover', async () => {
    const toolbar = await load();
    const chip = /** @type {HTMLButtonElement} */ ($('bundle-health'));
    toolbar.showBundleHealth({ state: 'checking' });
    expect(chip.disabled).toBe(true);
    expect(chip.classList.contains('osprey-chip--connecting')).toBe(true);

    toolbar.showBundleHealth({
      state: 'checked',
      health: {
        ok: false,
        total: 1,
        warnings: [{ concept_id: 'devices/bpm', kind: 'broken-link', detail: 'gone' }],
      },
    });
    expect(chip.disabled).toBe(false);
    expect(chip.textContent?.trim()).toBe('1 issue');
    expect(chip.classList.contains('osprey-chip--warn')).toBe(true);
    expect(chip.classList.contains('osprey-chip--connecting')).toBe(false);

    chip.click();
    const popover = /** @type {HTMLElement} */ (document.querySelector('.osprey-info-popover'));
    expect(popover.hidden).toBe(false);
    expect(popover.textContent).toContain('devices/bpm');
    expect(popover.textContent).toContain('Broken link: gone');
  });
});

describe("the panel's own ⋯", () => {
  test('standalone it ends the toolbar and runs the menu entries through the panel handler', async () => {
    await load('/');
    const button = $('panel-menu');
    expect(button.parentElement?.lastElementChild).toBe(button);
    expect(button.getAttribute('aria-haspopup')).toBe('menu');
    button.click();
    const rows = Array.from(document.querySelectorAll('.osprey-context-menu [role="menuitem"]'));
    expect(rows.map((row) => row.textContent)).toEqual(['Reset contents width']);
    /** @type {HTMLElement} */ (rows[0]).click();
    expect(onMenu).toHaveBeenCalledWith('knowledge', 'reset-width');
  });

  test('embedded it does not exist: the tile has the one ⋯', async () => {
    await load('/?embedded=true');
    expect(document.getElementById('panel-menu')).toBeNull();
  });
});
