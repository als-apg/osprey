// @ts-check
/**
 * Unit tests for the Channel Finder's panel toolbar (panel-toolbar.js): the
 * view switch, the controls a view claims or leaves disabled with a reason,
 * the search a PV or IOC page hands to All PVs, and the sidebar toggle a drawn sidebar drives. The
 * toolbar markup is the real one, read from index.html:
 *   npx vitest run tests/interfaces/channel_finder/panel-toolbar.test.mjs
 */

import { describe, test, expect, beforeEach, afterEach } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import {
  attachSidebar,
  claimToolbar,
  initToolbar,
  setFilterReason,
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

/** @type {Array<() => void>} */
const cleanup = [];

beforeEach(() => {
  localStorage.clear();
  document.body.innerHTML = PANEL_HTML;
  state.setPipelineInfo('graph', null);
  state.setActiveView('explore');
});

afterEach(() => {
  document.body.dispatchEvent(new MouseEvent('pointerdown', { bubbles: true }));
  for (const fn of cleanup.splice(0)) fn();
  location.hash = '';
});

/** Add the Feedback button app.js writes into the seg outside the graph paradigm. */
function addFeedbackButton() {
  const button = document.createElement('button');
  button.type = 'button';
  button.setAttribute('data-view', 'feedback');
  button.setAttribute('aria-pressed', 'false');
  button.textContent = 'Feedback';
  byId('cf-views').append(button);
}

describe('the view switch', () => {
  test('index.html draws Explore alone: app.js adds what the paradigm has', () => {
    const views = [...document.querySelectorAll('#cf-views button')];
    expect(views.map((b) => b.getAttribute('data-view'))).toEqual(['explore']);
  });

  test('a press sets the hash, and the pressed button follows the active view', () => {
    addFeedbackButton();
    initToolbar();
    const views = () =>
      [...document.querySelectorAll('#cf-views button')].map(
        (b) => `${b.getAttribute('data-view')}:${b.getAttribute('aria-pressed')}`
      );
    expect(views()).toEqual(['explore:true', 'feedback:false']);

    /** @type {HTMLElement} */ (document.querySelector('[data-view="feedback"]')).click();
    expect(location.hash).toBe('#/feedback');

    state.setActiveView('feedback');
    expect(views()).toEqual(['explore:false', 'feedback:true']);
  });

  test('a press on the active view leaves the hash as the view wrote it', () => {
    addFeedbackButton();
    initToolbar();
    state.setActiveView('feedback');
    location.hash = '#/feedback?kept';
    /** @type {HTMLElement} */ (document.querySelector('[data-view="feedback"]')).click();
    expect(location.hash).toBe('#/feedback?kept');
  });
});

describe('controls a view does not use', () => {
  test('stay in their slots, disabled, and say why under Feedback', () => {
    initToolbar();
    state.setPipelineInfo('hierarchical', null);
    state.setActiveView('feedback');

    expect(toggle().disabled).toBe(true);
    expect(toggle().title).toBe('No filters for this view');
    expect(toggle().hasAttribute('aria-expanded')).toBe(false);
    expect(search().disabled).toBe(true);
    expect(byId('cf-search-field').title).toBe('No search for this view');
  });

  test('say there is nothing for them in a paradigm without filters', () => {
    initToolbar();
    state.setPipelineInfo('hierarchical', null);

    expect(toggle().disabled).toBe(true);
    expect(toggle().title).toBe('No filters for this view');
    expect(byId('cf-search-field').title).toBe('No search for this view');
  });
});

describe("a view's own reason for the filter toggle", () => {
  test('is the tooltip the toolbar writes, through refreshes, until the view changes', () => {
    initToolbar();
    state.setActiveView('alarms');
    setFilterReason('No alarm tree on this target');
    expect(toggle().disabled).toBe(true);
    expect(toggle().title).toBe('No alarm tree on this target');

    state.setPipelineInfo('hierarchical', null);
    expect(toggle().title).toBe('No alarm tree on this target');

    state.setActiveView('feedback');
    expect(toggle().title).toBe('No filters for this view');
  });

  test('comes back when a drawn sidebar is destroyed', () => {
    initToolbar();
    state.setActiveView('alarms');
    setFilterReason('No alarm tree on this target');
    const row = document.createElement('div');
    row.innerHTML =
      '<aside class="osprey-panel-sidebar" id="tree" data-open="false"></aside>' +
      '<div class="osprey-splitter" role="separator" tabindex="0" aria-label="Resize"></div>';
    byId('app-main').append(row);
    const sidebar = byId('tree');
    const api = attachSidebar(
      sidebar,
      /** @type {HTMLElement} */ (sidebar.nextElementSibling),
      'cf-test-reason'
    );
    expect(toggle().disabled).toBe(false);
    api.destroy();
    expect(toggle().disabled).toBe(true);
    expect(toggle().title).toBe('No alarm tree on this target');
  });

  test('a null reason gives the toggle back the default', () => {
    initToolbar();
    state.setActiveView('alarms');
    setFilterReason('No alarm tree on this target');
    setFilterReason(null);
    expect(toggle().title).toBe('No filters for this view');
  });
});

describe('a claimed toolbar', () => {
  test('hands the view the search field, and the release takes it back empty', () => {
    initToolbar();
    const release = claimToolbar();
    expect(search().disabled).toBe(false);
    expect(byId('cf-search-field').title).toBe('');

    search().value = 'qfa';
    setResultCount('<strong>1</strong> channel on <strong>1</strong> device');
    expect(byId('cf-result-count').hidden).toBe(false);

    release();
    expect(search().disabled).toBe(true);
    expect(search().value).toBe('');
    expect(byId('cf-result-count').hidden).toBe(true);
  });
});

describe('the search on a PV or IOC page', () => {
  /** @param {string} words */
  function enter(words) {
    search().value = words;
    search().dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));
  }

  test.each(['pv', 'ioc'])('is enabled on %s and Enter searches All PVs', (view) => {
    initToolbar();
    state.setActiveView(view);
    expect(search().disabled).toBe(false);
    expect(byId('cf-search-field').title).toBe('');

    enter(' BPM01 ');
    expect(location.hash).toBe('#/allpvs?q=BPM01');
    enter('beam current');
    expect(location.hash).toBe('#/allpvs?q=beam%20current');
    enter('  ');
    expect(location.hash).toBe('#/allpvs');
  });

  test('is disabled on a view outside All PVs, and Enter does nothing', () => {
    initToolbar();
    state.setActiveView('explore');
    expect(search().disabled).toBe(true);
    enter('BPM01');
    expect(location.hash).toBe('');
  });

  test('while a view holds the field, the toolbar Enter does nothing', () => {
    initToolbar();
    state.setActiveView('pv');
    const release = claimToolbar();
    enter('BPM01');
    expect(location.hash).toBe('');
    release();
    expect(search().disabled).toBe(false);
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
