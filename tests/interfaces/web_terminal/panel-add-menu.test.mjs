/**
 * Unit tests for the add-panel menu's pure id-derivation helper.
 *
 *   npx vitest run tests/interfaces/web_terminal/panel-add-menu.test.mjs
 *
 * derivePanelId turns human input (a label or a URL) into a URL-safe panel id.
 * The DOM/popover behavior of initPanelAddMenu is covered end-to-end by the
 * Playwright suite (test_panels_browser.py); here we pin the slug contract.
 *
 * Imported by RELATIVE path — this module lives under web_terminal, so the
 * /design-system/js/* alias does not apply.
 */

import { test, expect, describe, beforeEach, vi } from 'vitest';

import {
  derivePanelId,
  initPanelAddMenu,
} from '../../../src/osprey/interfaces/web_terminal/static/js/panel-add-menu.js';

describe('derivePanelId', () => {
  test.each([
    ['a plain label is slugged', 'My Dashboard', 'my-dashboard'],
    ['a URL input uses its hostname', 'http://grafana.internal:3000', 'grafana-internal'],
    ['a URL hostname is lowercased, path and query dropped', 'https://Metrics.Lan/path?q=1', 'metrics-lan'],
    ['runs of punctuation collapse', '  Beam   Position!! ', 'beam-position'],
    ['edge dashes are trimmed', '--weird__name--', 'weird-name'],
    ['an empty input never yields an empty id', '', 'panel'],
    ['punctuation only never yields an empty id', '!!!', 'panel'],
    ['whitespace only never yields an empty id', '   ', 'panel'],
    ['an already-clean slug is unchanged', 'lattice', 'lattice'],
  ])('%s', (_case, input, id) => {
    expect(derivePanelId(input)).toBe(id);
  });
});

/**
 * The "Layouts" section of the "+" popover. Rendered only when presets are
 * configured, so a deployment that has not opted in sees the menu unchanged.
 */
describe('initPanelAddMenu — Layouts section', () => {
  /** @returns {{root: HTMLElement, button: HTMLButtonElement, menu: HTMLElement}} */
  function mountMenuDom() {
    document.body.innerHTML = `
      <div id="panel-add">
        <button id="panel-add-btn" type="button" aria-expanded="false"></button>
        <div id="panel-add-menu"></div>
      </div>`;
    return {
      root: /** @type {HTMLElement} */ (document.getElementById('panel-add')),
      button: /** @type {HTMLButtonElement} */ (document.getElementById('panel-add-btn')),
      menu: /** @type {HTMLElement} */ (document.getElementById('panel-add-menu')),
    };
  }

  /**
   * Base options with no presets and no hidden panels — the popover defaults.
   * @param {{root: HTMLElement, button: HTMLButtonElement, menu: HTMLElement}} dom
   * @param {Partial<import('../../../src/osprey/interfaces/web_terminal/static/js/panel-add-menu.js').AddMenuOptions>} [overrides]
   */
  function baseOptions(dom, overrides = {}) {
    return {
      rootEl: dom.root,
      buttonEl: dom.button,
      menuEl: dom.menu,
      getHiddenPanels: () => [],
      allowUrlPanels: () => false,
      onShowPanel: () => {},
      onRegisterUrl: async () => ({ ok: true }),
      getPresets: () => [],
      onApplyPreset: () => {},
      ...overrides,
    };
  }

  beforeEach(() => {
    document.body.innerHTML = '';
  });

  test('renders one item per preset under a Layouts heading', () => {
    const dom = mountMenuDom();
    const presets = [
      { name: 'Machine setup', panels: ['channel-finder', 'lattice'] },
      { name: 'Logbook review', panels: ['ariel', 'artifacts'] },
    ];
    initPanelAddMenu(baseOptions(dom, { getPresets: () => presets }));
    dom.button.click(); // open → render

    const headings = [...dom.menu.querySelectorAll('.panel-add-heading')].map((h) => h.textContent);
    expect(headings).toContain('Layouts');
    const items = [...dom.menu.querySelectorAll('.panel-add-item')].map((i) => i.textContent);
    expect(items).toContain('Machine setup');
    expect(items).toContain('Logbook review');
  });

  test('preset name is set via textContent (no HTML injection)', () => {
    const dom = mountMenuDom();
    const presets = [{ name: '<b>x</b>', panels: ['artifacts'] }];
    initPanelAddMenu(baseOptions(dom, { getPresets: () => presets }));
    dom.button.click();

    const item = dom.menu.querySelector('.panel-add-item');
    expect(item?.textContent).toBe('<b>x</b>'); // literal text, not parsed
    expect(item?.querySelector('b')).toBeNull(); // no injected element
  });

  test('Layouts section is ABSENT when no presets are configured', () => {
    const dom = mountMenuDom();
    initPanelAddMenu(baseOptions(dom)); // getPresets → []
    dom.button.click();

    // The menu rendered, with only its default section.
    const headings = [...dom.menu.querySelectorAll('.panel-add-heading')].map((h) => h.textContent);
    expect(headings).toEqual(['Show panel']);
  });

  test('clicking a preset calls onApplyPreset with its NAME and closes the menu', () => {
    const dom = mountMenuDom();
    const onApplyPreset = vi.fn();
    const presets = [{ name: 'Machine setup', panels: ['channel-finder', 'lattice'] }];
    initPanelAddMenu(baseOptions(dom, { getPresets: () => presets, onApplyPreset }));
    dom.button.click();

    const item = /** @type {HTMLElement} */ (dom.menu.querySelector('.panel-add-item'));
    item.click();

    // The name is the whole request — members are resolved server-side, so the
    // menu never hands a panel list around.
    expect(onApplyPreset).toHaveBeenCalledWith('Machine setup');
    expect(dom.menu.classList.contains('open')).toBe(false); // menu closed after apply
  });
});
