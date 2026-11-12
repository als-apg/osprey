// @ts-check
/**
 * The KNOWLEDGE panel's two views and the switch between them (views.js):
 *   npx vitest run tests/interfaces/okf_panel/views.test.mjs
 *
 * The page shell is the unrendered template, whose `data-graph` is not yet
 * "true": the state of a panel that serves no graph. A test that wants the
 * Graph view sets the attribute, as the server does.
 */

import { beforeEach, describe, expect, test, vi } from 'vitest';

import { mountShell } from './page-shell.mjs';

/** @typedef {typeof import('../../../src/osprey/interfaces/okf_panel/static/js/views.js')} Views */

/** @param {string} id @returns {HTMLElement} */
const $ = (id) => /** @type {HTMLElement} */ (document.getElementById(id));

/**
 * @param {{graph: boolean}} opts
 * @returns {Promise<{views: Views, onSwitch: ReturnType<typeof vi.fn>}>}
 */
async function load({ graph }) {
  mountShell();
  if (graph) $('app').dataset.graph = 'true';
  vi.resetModules();
  const views = await import('../../../src/osprey/interfaces/okf_panel/static/js/views.js');
  const onSwitch = vi.fn();
  views.initViews({ onSwitch });
  return { views, onSwitch };
}

/** @param {string} view */
function clickView(view) {
  /** @type {HTMLElement} */ ($('okf-views').querySelector(`[data-view="${view}"]`)).click();
}

describe('without a graph', () => {
  /** @type {Views} */
  let views;
  beforeEach(async () => {
    ({ views } = await load({ graph: false }));
  });

  test('the switcher stays hidden and the panel is the Knowledge view', () => {
    expect(views.graphAvailable()).toBe(false);
    expect($('okf-views').hidden).toBe(true);
    expect(views.currentView()).toBe('knowledge');
    expect($('graph-view').hidden).toBe(true);
  });

  test('the Graph view cannot be shown', () => {
    expect(views.showView('graph')).toBe(false);
    expect(views.currentView()).toBe('knowledge');
    expect($('knowledge-view').hidden).toBe(false);
  });
});

describe('with a graph', () => {
  /** @type {Views} */
  let views;
  /** @type {ReturnType<typeof vi.fn>} */
  let onSwitch;
  beforeEach(async () => {
    ({ views, onSwitch } = await load({ graph: true }));
  });

  test('the switcher shows, on Knowledge', () => {
    expect($('okf-views').hidden).toBe(false);
    expect(views.currentView()).toBe('knowledge');
    expect($('okf-views').querySelector('[aria-pressed="true"]')?.textContent).toBe('Knowledge');
  });

  test('the switcher asks for the other view and changes nothing itself', () => {
    clickView('graph');
    expect(onSwitch).toHaveBeenCalledWith('graph');
    expect(views.currentView()).toBe('knowledge');
    clickView('knowledge');
    expect(onSwitch).toHaveBeenCalledTimes(1);
  });

  test('showing the Graph view swaps the body and each toolbar part', () => {
    expect(views.showView('graph')).toBe(true);
    for (const id of ['knowledge-view', 'contents-toggle', 'crumb', 'bundle-health']) {
      expect($(id).hidden, id).toBe(true);
    }
    for (const id of ['graph-view', 'graph-toggle', 'graph-context', 'graph-chip']) {
      expect($(id).hidden, id).toBe(false);
    }
    expect($('okf-views').querySelector('[aria-pressed="true"]')?.textContent).toBe('Graph');
    const search = /** @type {HTMLInputElement} */ ($('search-input'));
    expect(search.placeholder).toBe('Search nodes…');
    expect(search.getAttribute('aria-label')).toBe('Search nodes');
  });

  test('showing the view already shown changes nothing', () => {
    expect(views.showView('knowledge')).toBe(false);
    views.showView('graph');
    expect(views.showView('graph')).toBe(false);
  });

  test('going back restores the Knowledge view', () => {
    views.showView('graph');
    expect(views.showView('knowledge')).toBe(true);
    expect($('knowledge-view').hidden).toBe(false);
    expect($('graph-view').hidden).toBe(true);
    expect($('bundle-health').hidden).toBe(false);
    expect(/** @type {HTMLInputElement} */ ($('search-input')).placeholder).toBe(
      'Search concepts…'
    );
  });
});
