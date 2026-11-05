// @ts-check
/**
 * Unit tests for the Channel Finder's panel toolbar (panel-toolbar.js): the
 * view switch, the controls a view claims or leaves disabled with a reason,
 * the standalone-only ⋯, and the sidebar toggle a drawn sidebar drives. The
 * toolbar markup is the real one, read from index.html:
 *   npx vitest run tests/interfaces/channel_finder/panel-toolbar.test.mjs
 */

import { describe, test, expect, beforeEach, afterEach, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

const mocks = vi.hoisted(() => ({ isEmbedded: vi.fn(() => false) }));

vi.mock('/design-system/js/frame-params.js', () => ({ isEmbedded: mocks.isEmbedded }));

import {
  attachSidebar,
  claimToolbar,
  initToolbar,
  menuRows,
  setResultCount,
} from '../../../src/osprey/interfaces/channel_finder/static/js/panel-toolbar.js';
import { state } from '../../../src/osprey/interfaces/channel_finder/static/js/state.js';

const PANEL_HTML = (() => {
  const html = readFileSync(
    join(import.meta.dirname, '../../../src/osprey/interfaces/channel_finder/static/index.html'),
    'utf8'
  );
  const match = html.match(/<div class="cf-panel"[\s\S]*<\/main>/);
  if (!match) throw new Error('index.html carries no panel root');
  return `${match[0]}</div>`;
})();

/** @param {string} id @returns {HTMLElement} */
function byId(id) {
  const el = document.getElementById(id);
  if (!el) throw new Error(`no #${id}`);
  return el;
}

const toggle = () => /** @type {HTMLButtonElement} */ (byId('cf-sidebar-toggle'));
const search = () => /** @type {HTMLInputElement} */ (byId('cf-search'));
const menuButton = () =>
  /** @type {HTMLButtonElement|null} */ (document.getElementById('cf-panel-menu'));

/** @returns {string[]} the labels of the open context menu */
function menuLabels() {
  return [...document.querySelectorAll('.osprey-context-menu .osprey-context-label')].map(
    (label) => label.textContent ?? ''
  );
}

/** The graph finder's ⋯, as the mount hands it over. */
const ENTRIES = [
  { id: 'density-comfortable', label: 'Comfortable rows', checked: true },
  { id: 'density-compact', label: 'Compact rows', checked: false },
  { id: 'clear-filters', label: 'Clear all filters', disabled: true },
];

/** @type {Array<() => void>} */
const cleanup = [];

beforeEach(() => {
  localStorage.clear();
  document.body.innerHTML = PANEL_HTML;
  mocks.isEmbedded.mockReturnValue(false);
  state.setPipelineInfo('graph', null);
  state.setActiveView('explore');
});

afterEach(() => {
  document.body.dispatchEvent(new MouseEvent('pointerdown', { bubbles: true }));
  for (const fn of cleanup.splice(0)) fn();
  location.hash = '';
});

describe('the view switch', () => {
  test('a press sets the hash, and the pressed button follows the active view', () => {
    initToolbar();
    const views = () =>
      [...document.querySelectorAll('#cf-views button')].map(
        (b) => `${b.getAttribute('data-view')}:${b.getAttribute('aria-pressed')}`
      );
    expect(views()).toEqual(['explore:true', 'feedback:false']);

    /** @type {HTMLElement} */ (document.querySelector('[data-view="feedback"]')).click();
    expect(location.hash).toBe('#feedback');

    state.setActiveView('feedback');
    expect(views()).toEqual(['explore:false', 'feedback:true']);
  });
});

describe('controls a view does not use', () => {
  test('stay in their slots, disabled, and say why under Feedback', () => {
    initToolbar();
    state.setActiveView('feedback');

    expect(toggle().disabled).toBe(true);
    expect(toggle().title).toBe('Filters apply to Explore');
    expect(toggle().hasAttribute('aria-expanded')).toBe(false);
    expect(search().disabled).toBe(true);
    expect(byId('cf-search-field').title).toBe('Search applies to Explore');
    expect(menuButton()?.disabled).toBe(true);
    expect(menuButton()?.title).toBe('Options apply to Explore');
  });

  test('say there is nothing for them in a paradigm without filters', () => {
    initToolbar();
    state.setPipelineInfo('hierarchical', null);

    expect(toggle().disabled).toBe(true);
    expect(toggle().title).toBe('No filters for this view');
    expect(byId('cf-search-field').title).toBe('No search for this view');

    // Under Feedback too: Explore has no filters to point back to.
    state.setActiveView('feedback');
    expect(toggle().title).toBe('No filters for this view');
  });
});

describe('a claimed toolbar', () => {
  test('hands the view the search field and the ⋯, and the release takes them back empty', () => {
    initToolbar();
    const release = claimToolbar({ entries: () => ENTRIES, run: () => {} });
    expect(search().disabled).toBe(false);
    expect(byId('cf-search-field').title).toBe('');
    expect(menuButton()?.disabled).toBe(false);

    search().value = 'qfa';
    setResultCount('<strong>1</strong> channel on <strong>1</strong> device');
    expect(byId('cf-result-count').hidden).toBe(false);

    release();
    expect(search().disabled).toBe(true);
    expect(search().value).toBe('');
    expect(byId('cf-result-count').hidden).toBe(true);
    expect(menuButton()?.disabled).toBe(true);
  });
});

describe('the standalone ⋯', () => {
  test('is built at the far right standalone, and not at all embedded', () => {
    initToolbar();
    const end = byId('cf-toolbar-end');
    expect(end.lastElementChild?.id).toBe('cf-panel-menu');
    expect(menuButton()?.getAttribute('aria-label')).toBe('Panel menu');
    expect(menuButton()?.getAttribute('aria-haspopup')).toBe('menu');

    document.body.innerHTML = PANEL_HTML;
    mocks.isEmbedded.mockReturnValue(true);
    initToolbar();
    expect(menuButton()).toBeNull();
  });

  test('lists the view’s entries and runs its handler with the entry id', () => {
    initToolbar();
    const run = vi.fn();
    cleanup.push(claimToolbar({ entries: () => ENTRIES, run }));

    /** @type {HTMLButtonElement} */ (menuButton()).click();
    expect(menuLabels()).toEqual(['Comfortable rows', 'Compact rows', 'Clear all filters']);
    const rows = [...document.querySelectorAll('.osprey-context-menu .osprey-context-item')];
    expect(rows.map((row) => row.getAttribute('aria-checked'))).toEqual(['true', 'false', null]);
    expect(rows[2].getAttribute('aria-disabled')).toBe('true');

    /** @type {HTMLElement} */ (rows[1]).click();
    expect(run).toHaveBeenCalledWith('density-compact');
  });

  test('menuRows draws a separator as a divider and keeps the danger colour', () => {
    const run = vi.fn();
    const rows = menuRows(
      [
        { id: 'a', label: 'A' },
        { id: 'b', label: 'B', separator: true, danger: true },
      ],
      run
    );
    expect(rows.map((row) => (row.divider ? '—' : row.label))).toEqual(['A', '—', 'B']);
    expect(rows[2].danger).toBe(true);
    rows[2].run?.();
    expect(run).toHaveBeenCalledWith('b');
  });
});

describe('the sidebar toggle', () => {
  test('a drawn sidebar drives it; destroying the sidebar gives it back disabled', () => {
    initToolbar();
    const row = document.createElement('div');
    row.innerHTML =
      '<aside class="osprey-panel-sidebar" id="filters" data-open="false"></aside>' +
      '<div class="osprey-splitter" role="separator" tabindex="0" aria-label="Resize"></div>';
    byId('app-main').append(row);
    const sidebar = byId('filters');

    const api = attachSidebar(
      sidebar,
      /** @type {HTMLElement} */ (sidebar.nextElementSibling),
      'cf-test'
    );
    expect(toggle().disabled).toBe(false);
    expect(toggle().getAttribute('aria-controls')).toBe('filters');
    expect(sidebar.getAttribute('data-open')).toBe('true');

    toggle().click();
    expect(sidebar.getAttribute('data-open')).toBe('false');
    expect(toggle().getAttribute('aria-expanded')).toBe('false');

    api.destroy();
    expect(toggle().disabled).toBe(true);
    expect(toggle().title).toBe('No filters for this view');
    expect(toggle().hasAttribute('aria-controls')).toBe(false);
    expect(byId('cf-panel').hasAttribute('data-open')).toBe(false);
  });
});
