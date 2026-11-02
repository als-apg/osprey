/**
 * Unit tests for the add-panel menu (the human "+").
 *
 *   npx vitest run tests/interfaces/web_terminal/panel-add-menu.test.mjs
 *
 * derivePanelId turns human input (a label or a URL) into a URL-safe panel id.
 * initPanelAddMenu renders the body of the body-level `#panel-add-menu`
 * popover each time the sidebar's "+ Add panel" row opens it; the design
 * system's popover.js shows it (`hidden` off, the row's `aria-expanded` on)
 * beside the sidebar and dismisses it.
 *
 * The source module is imported by RELATIVE path; it reaches the design
 * system through the /design-system/js/* alias (vitest.config.js).
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
    [
      'a URL hostname is lowercased, path and query dropped',
      'https://Metrics.Lan/path?q=1',
      'metrics-lan',
    ],
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

/** @typedef {{sidebar: HTMLElement, button: HTMLButtonElement, menu: HTMLElement}} MenuDom */

/**
 * The "+ Add panel" row inside the sidebar, and the body-level popover it
 * opens, shaped as index.html ships them.
 * @returns {MenuDom}
 */
function mountMenuDom() {
  document.body.innerHTML = `
    <nav class="sidebar" id="sidebar">
      <div class="sidebar-group" data-group="tools">
        <button class="osprey-nav-row add" id="panel-add-btn" type="button"
                aria-haspopup="dialog" aria-expanded="false" aria-controls="panel-add-menu">
          <span class="osprey-nav-label">Add panel</span>
        </button>
      </div>
    </nav>
    <div class="osprey-popover panel-add-menu" id="panel-add-menu" hidden></div>`;
  return {
    sidebar: /** @type {HTMLElement} */ (document.getElementById('sidebar')),
    button: /** @type {HTMLButtonElement} */ (document.getElementById('panel-add-btn')),
    menu: /** @type {HTMLElement} */ (document.getElementById('panel-add-menu')),
  };
}

/**
 * Base options with no presets, no hidden panels and no URL form — the
 * popover defaults.
 * @param {MenuDom} dom
 * @param {Partial<import('../../../src/osprey/interfaces/web_terminal/static/js/panel-add-menu.js').AddMenuOptions>} [overrides]
 */
function baseOptions(dom, overrides = {}) {
  return {
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

/** @param {HTMLElement} menu @returns {(string | null)[]} */
function headings(menu) {
  return [...menu.querySelectorAll('.panel-add-heading')].map((h) => h.textContent);
}

/** @param {HTMLElement} menu @returns {(string | null)[]} */
function items(menu) {
  return [...menu.querySelectorAll('.panel-add-item')].map((i) => i.textContent);
}

/** Let the async submit handler settle. */
function settle() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

beforeEach(() => {
  document.body.innerHTML = '';
});

describe('initPanelAddMenu — opening and closing', () => {
  test('a click on + Add panel opens the popover through popover.js', () => {
    const dom = mountMenuDom();
    initPanelAddMenu(baseOptions(dom));

    dom.button.click();

    expect(dom.menu.hidden).toBe(false);
    expect(dom.button.getAttribute('aria-expanded')).toBe('true');
    expect(dom.button.getAttribute('aria-controls')).toBe('panel-add-menu');
  });

  test('a second click closes it again', () => {
    const dom = mountMenuDom();
    initPanelAddMenu(baseOptions(dom));

    dom.button.click();
    dom.button.click();

    expect(dom.menu.hidden).toBe(true);
    expect(dom.button.getAttribute('aria-expanded')).toBe('false');
  });

  test('the popover opens beside the sidebar, not over it', () => {
    const dom = mountMenuDom();
    dom.sidebar.getBoundingClientRect = () =>
      /** @type {DOMRect} */ ({
        top: 0,
        bottom: 800,
        left: 0,
        right: 240,
        width: 240,
        height: 800,
      });
    initPanelAddMenu(baseOptions(dom));

    dom.button.click();

    expect(dom.menu.style.left).toBe('248px');
  });

  test('Escape closes the popover and returns focus to the row', () => {
    const dom = mountMenuDom();
    initPanelAddMenu(
      baseOptions(dom, { getHiddenPanels: () => [{ id: 'lattice', label: 'Lattice' }] })
    );
    dom.button.click();
    expect(document.activeElement).toBe(dom.menu.querySelector('.panel-add-item'));

    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));

    expect(dom.menu.hidden).toBe(true);
    expect(document.activeElement).toBe(dom.button);
  });

  test('each open renders the current state', () => {
    const dom = mountMenuDom();
    /** @type {{id: string, label: string}[]} */
    let hidden = [{ id: 'lattice', label: 'Lattice' }];
    initPanelAddMenu(baseOptions(dom, { getHiddenPanels: () => hidden }));

    dom.button.click();
    expect(items(dom.menu)).toEqual(['Lattice']);
    dom.button.click();

    hidden = [{ id: 'okf', label: 'Knowledge' }];
    dom.button.click();
    expect(items(dom.menu)).toEqual(['Knowledge']);
  });

  test('a missing row or popover makes the menu a no-op', () => {
    const dom = mountMenuDom();
    expect(() =>
      initPanelAddMenu(
        baseOptions(dom, { menuEl: /** @type {HTMLElement} */ (/** @type {unknown} */ (null)) })
      )
    ).not.toThrow();
    dom.button.click();
    expect(dom.menu.hidden).toBe(true);
  });

  test('the menu offers panels only: no sidebar-layout row', () => {
    const dom = mountMenuDom();
    initPanelAddMenu(baseOptions(dom, { allowUrlPanels: () => true }));
    dom.button.click();

    expect(headings(dom.menu)).toEqual(['Show panel', 'New panel from URL']);
    expect(dom.menu.textContent ?? '').not.toMatch(/rail/i);
  });
});

describe('initPanelAddMenu — Show panel section', () => {
  test('lists every hidden panel by label, carrying its id', () => {
    const dom = mountMenuDom();
    initPanelAddMenu(
      baseOptions(dom, {
        getHiddenPanels: () => [
          { id: 'lattice', label: 'Lattice' },
          { id: 'okf', label: 'Knowledge' },
        ],
      })
    );
    dom.button.click();

    const rows = /** @type {HTMLElement[]} */ ([...dom.menu.querySelectorAll('.panel-add-item')]);
    expect(rows.map((r) => [r.dataset.panelId, r.textContent])).toEqual([
      ['lattice', 'Lattice'],
      ['okf', 'Knowledge'],
    ]);
    for (const r of rows) {
      expect(r.classList.contains('osprey-popover-item')).toBe(true);
      expect(r.getAttribute('type')).toBe('button');
    }
    expect(
      dom.menu.querySelector('.panel-add-heading')?.classList.contains('osprey-popover-title')
    ).toBe(true);
  });

  test('picking a hidden panel shows it and closes the menu', () => {
    const dom = mountMenuDom();
    const onShowPanel = vi.fn();
    initPanelAddMenu(
      baseOptions(dom, {
        getHiddenPanels: () => [{ id: 'lattice', label: 'Lattice' }],
        onShowPanel,
      })
    );
    dom.button.click();

    /** @type {HTMLElement} */ (dom.menu.querySelector('[data-panel-id="lattice"]')).click();

    expect(onShowPanel).toHaveBeenCalledWith('lattice');
    expect(dom.menu.hidden).toBe(true);
    expect(dom.button.getAttribute('aria-expanded')).toBe('false');
  });

  test('with every panel shown, the section says so', () => {
    const dom = mountMenuDom();
    initPanelAddMenu(baseOptions(dom));
    dom.button.click();

    const empty = dom.menu.querySelector('.panel-add-empty');
    expect(empty?.classList.contains('osprey-list-empty')).toBe(true);
    expect(empty?.textContent).toBe('All panels shown');
    expect(dom.menu.querySelector('.panel-add-item')).toBeNull();
  });

  test('a panel label is set via textContent (no HTML injection)', () => {
    const dom = mountMenuDom();
    initPanelAddMenu(
      baseOptions(dom, { getHiddenPanels: () => [{ id: 'x', label: '<img src=x>' }] })
    );
    dom.button.click();

    const item = dom.menu.querySelector('.panel-add-item');
    expect(item?.textContent).toBe('<img src=x>');
    expect(item?.querySelector('img')).toBeNull();
  });
});

/**
 * The "Layouts" section of the "+" popover. Rendered only when presets are
 * configured, so a deployment that has not opted in sees the menu unchanged.
 */
describe('initPanelAddMenu — Layouts section', () => {
  test('renders one item per preset under a Layouts heading, then a separator', () => {
    const dom = mountMenuDom();
    const presets = [
      { name: 'Machine setup', panels: ['channel-finder', 'lattice'] },
      { name: 'Logbook review', panels: ['ariel', 'artifacts'] },
    ];
    initPanelAddMenu(baseOptions(dom, { getPresets: () => presets }));
    dom.button.click(); // open → render

    expect(headings(dom.menu)).toEqual(['Layouts', 'Show panel']);
    expect(items(dom.menu)).toEqual(['Machine setup', 'Logbook review']);
    const sep = /** @type {HTMLElement} */ (dom.menu.querySelector('.osprey-popover-sep'));
    expect(sep.getAttribute('role')).toBe('separator');
    expect(sep.nextElementSibling?.textContent).toBe('Show panel');
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
    expect(headings(dom.menu)).toEqual(['Show panel']);
    expect(dom.menu.querySelector('.osprey-popover-sep')).toBeNull();
  });

  test('clicking a preset calls onApplyPreset with its NAME and closes the menu', () => {
    const dom = mountMenuDom();
    const onApplyPreset = vi.fn();
    const presets = [{ name: 'Machine setup', panels: ['channel-finder', 'lattice'] }];
    initPanelAddMenu(baseOptions(dom, { getPresets: () => presets, onApplyPreset }));
    dom.button.click();

    /** @type {HTMLElement} */ (dom.menu.querySelector('.panel-add-item')).click();

    // The name is the whole request — members are resolved server-side, so the
    // menu never hands a panel list around.
    expect(onApplyPreset).toHaveBeenCalledWith('Machine setup');
    expect(dom.menu.hidden).toBe(true); // menu closed after apply
  });
});

describe('initPanelAddMenu — New panel from URL', () => {
  /**
   * Open a menu with the URL form and fill it in.
   * @param {Partial<import('../../../src/osprey/interfaces/web_terminal/static/js/panel-add-menu.js').AddMenuOptions>} overrides
   * @param {{url: string, label?: string}} fields
   */
  function openForm(overrides, fields) {
    const dom = mountMenuDom();
    initPanelAddMenu(baseOptions(dom, { allowUrlPanels: () => true, ...overrides }));
    dom.button.click();
    const form = /** @type {HTMLFormElement} */ (dom.menu.querySelector('form.panel-add-url'));
    const url = /** @type {HTMLInputElement} */ (
      form.querySelector('.panel-add-input[name="url"]')
    );
    const label = /** @type {HTMLInputElement} */ (
      form.querySelector('.panel-add-input[name="label"]')
    );
    url.value = fields.url;
    label.value = fields.label ?? '';
    const submit = () =>
      form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    return { dom, form, submit };
  }

  test('is absent unless the server allows runtime panels', () => {
    const dom = mountMenuDom();
    initPanelAddMenu(baseOptions(dom));
    dom.button.click();

    expect(dom.menu.querySelector('.panel-add-url')).toBeNull();
  });

  test('sits after a separator, with both fields, an error line and a primary Add button', () => {
    const { dom, form } = openForm({}, { url: '' });

    expect(form.previousElementSibling?.classList.contains('osprey-popover-sep')).toBe(true);
    expect(form.querySelector('.panel-add-heading')?.textContent).toBe('New panel from URL');
    expect(form.querySelectorAll('.panel-add-input').length).toBe(2);
    const error = /** @type {HTMLElement} */ (form.querySelector('.panel-add-url-error'));
    expect(error.hidden).toBe(true);
    const submit = /** @type {HTMLButtonElement} */ (form.querySelector('.panel-add-submit'));
    expect(submit.classList.contains('osprey-btn')).toBe(true);
    expect(submit.classList.contains('primary')).toBe(true);
    expect(submit.type).toBe('submit');
    expect(dom.menu.hidden).toBe(false);
  });

  test('a URL alone registers under its hostname and closes the menu', async () => {
    const onRegisterUrl = vi.fn(async () => ({ ok: true }));
    const { dom, submit } = openForm({ onRegisterUrl }, { url: 'http://grafana.internal:3000' });

    submit();
    await settle();

    expect(onRegisterUrl).toHaveBeenCalledWith({
      id: 'grafana-internal',
      label: 'grafana.internal',
      url: 'http://grafana.internal:3000',
    });
    expect(dom.menu.hidden).toBe(true);
  });

  test('a name, when given, is the label and the id source', async () => {
    const onRegisterUrl = vi.fn(async () => ({ ok: true }));
    const { submit } = openForm(
      { onRegisterUrl },
      { url: 'http://grafana.internal:3000', label: 'Beam Dashboard' }
    );

    submit();
    await settle();

    expect(onRegisterUrl).toHaveBeenCalledWith({
      id: 'beam-dashboard',
      label: 'Beam Dashboard',
      url: 'http://grafana.internal:3000',
    });
  });

  test('an empty URL submits nothing', async () => {
    const onRegisterUrl = vi.fn(async () => ({ ok: true }));
    const { dom, submit } = openForm({ onRegisterUrl }, { url: '   ' });

    submit();
    await settle();

    expect(onRegisterUrl).not.toHaveBeenCalled();
    expect(dom.menu.hidden).toBe(false);
  });

  test("a refusal shows the server's error and keeps the menu open", async () => {
    const { dom, form, submit } = openForm(
      { onRegisterUrl: async () => ({ ok: false, error: 'id already registered' }) },
      { url: 'http://grafana.internal:3000' }
    );

    submit();
    await settle();

    const error = /** @type {HTMLElement} */ (form.querySelector('.panel-add-url-error'));
    expect(error.hidden).toBe(false);
    expect(error.textContent).toBe('id already registered');
    expect(dom.menu.hidden).toBe(false);
    expect(
      /** @type {HTMLButtonElement} */ (form.querySelector('.panel-add-submit')).disabled
    ).toBe(false);
  });

  test('a failed request reads as an unreachable server', async () => {
    const { form, submit } = openForm(
      {
        onRegisterUrl: async () => {
          throw new Error('network');
        },
      },
      { url: 'http://grafana.internal:3000' }
    );

    submit();
    await settle();

    expect(form.querySelector('.panel-add-url-error')?.textContent).toBe(
      'Could not reach the server'
    );
  });
});
