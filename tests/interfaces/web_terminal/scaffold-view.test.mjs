/**
 * Unit tests for the Scaffold Gallery view layer (scaffold/view.js).
 *
 * Pure-logic + DOM guard, happy-dom environment (configured globally):
 *   npx vitest run tests/interfaces/web_terminal/scaffold-view.test.mjs
 *
 * Covers getFilteredArtifacts' search/category/project-owned filter
 * combinations (pure, no gallery instance needed), the card templates'
 * escaping contract (hostile artifact names must render as literal text,
 * never parsed as markup), the gallery view driven through its one entry
 * point, renderGallery(), with a fake `gallery` host object standing in for
 * the ArtifactGallery instance, and initScaffoldGallery's routing of each
 * artifact to exactly one drawer tab.
 *
 * NOTE: imported by RELATIVE path -- this module lives under web_terminal,
 * not design-system, so the `/design-system/js/*` alias does not apply.
 */

import { test, expect, describe, vi, afterEach } from 'vitest';

import { qs } from '../_support/dom.mjs';

import {
  getFilteredArtifacts,
  createScaffoldGalleryView,
} from '../../../src/osprey/interfaces/web_terminal/static/js/scaffold/view.js';
import { createScaffoldGalleryCards } from '../../../src/osprey/interfaces/web_terminal/static/js/scaffold/cards.js';

/**
 * @typedef {import('../../../src/osprey/interfaces/web_terminal/static/js/scaffold/view.js').ScaffoldGalleryHost} ScaffoldGalleryHost
 */

/**
 * Test fixture variant of {@link ScaffoldGalleryHost}: makeGallery always
 * assigns real elements (never leaves a DOM ref null the way a not-yet-
 * mounted gallery instance might), so the DOM-ref fields are narrowed to
 * non-null here for the tests that dereference them directly.
 * @typedef {ScaffoldGalleryHost & {
 *   galleryView: HTMLElement,
 *   detailView: HTMLElement,
 *   untrackedBannerEl: HTMLElement,
 *   filterChipsEl: HTMLElement,
 *   filterPanelEl: HTMLElement,
 *   filterToggleEl: HTMLElement,
 *   clearFilterEl: HTMLElement,
 *   collapsedCategories: Set<string>,
 *   summaryEl: HTMLElement,
 *   searchInput: HTMLInputElement,
 *   categoriesEl: HTMLElement,
 * }} TestGallery
 */

/**
 * @param {Partial<ScaffoldGalleryHost>} [overrides]
 * @returns {TestGallery}
 */
function makeGallery(overrides = {}) {
  return /** @type {TestGallery} */ ({
    artifacts: [],
    untrackedFiles: [],
    currentView: 'gallery',
    filterCategory: null,
    filterProjectOwned: false,
    searchQuery: '',
    filterOpen: false,
    collapsedCategories: new Set(),
    seenCategories: new Set(),
    pinnedCategories: [],
    summary: { total: 0, framework: 0, userOwned: 0 },
    galleryView: document.createElement('div'),
    detailView: document.createElement('div'),
    untrackedBannerEl: document.createElement('div'),
    filterChipsEl: document.createElement('div'),
    filterPanelEl: document.createElement('div'),
    filterToggleEl: document.createElement('button'),
    clearFilterEl: document.createElement('button'),
    summaryEl: document.createElement('div'),
    searchInput: (() => {
      const wrapper = document.createElement('div');
      const input = document.createElement('input');
      wrapper.appendChild(input);
      return input;
    })(),
    categoriesEl: document.createElement('div'),
    registerUntracked: () => Promise.resolve(),
    deleteUntracked: () => Promise.resolve(),
    openDetail: () => {},
    showCreateDialog: () => {},
    ...overrides,
  });
}

// ---------------------------------------------------------------------------
// getFilteredArtifacts (pure)
// ---------------------------------------------------------------------------

describe('getFilteredArtifacts', () => {
  const ARTIFACTS = [
    { name: 'claude-md', displayCategory: 'system prompt', status: 'framework', description: 'Root instructions' },
    { name: 'safety-check', displayCategory: 'hooks', status: 'user-owned', description: 'Blocks unsafe writes' },
    { name: 'channel-finder', displayCategory: 'agents', status: 'framework', summary: 'Finds beamline channels' },
    { name: 'lattice-agent', displayCategory: 'agents', status: 'user-owned', description: 'Lattice analysis' },
  ];
  const SPARSE = [{ name: 'bare', displayCategory: 'config', status: 'framework' }];
  const NONE = { filterProjectOwned: false, filterCategory: null, searchQuery: '' };

  test.each([
    ['no filters returns every artifact', ARTIFACTS, {},
      ['claude-md', 'safety-check', 'channel-finder', 'lattice-agent']],
    ['project-owned keeps only user-owned artifacts', ARTIFACTS, { filterProjectOwned: true },
      ['safety-check', 'lattice-agent']],
    ['a category keeps only that displayCategory', ARTIFACTS, { filterCategory: 'agents' },
      ['channel-finder', 'lattice-agent']],
    ['search matches the name, case-insensitively', ARTIFACTS, { searchQuery: 'CLAUDE' }, ['claude-md']],
    ['search matches the description', ARTIFACTS, { searchQuery: 'unsafe' }, ['safety-check']],
    ['search matches the summary', ARTIFACTS, { searchQuery: 'beamline' }, ['channel-finder']],
    ['all three filters intersect', ARTIFACTS,
      { filterProjectOwned: true, filterCategory: 'agents', searchQuery: 'lattice' }, ['lattice-agent']],
    ['a combination with no match is empty', ARTIFACTS,
      { filterProjectOwned: true, filterCategory: 'system prompt' }, []],
    ['artifacts without description or summary still match by name', SPARSE, { searchQuery: 'bare' },
      ['bare']],
  ])('%s', (_label, artifacts, filters, expected) => {
    const result = getFilteredArtifacts(artifacts, { ...NONE, ...filters });
    expect(result.map((a) => a.name)).toEqual(expected);
  });
});

// ---------------------------------------------------------------------------
// Card rendering -- hostile names render as text
// ---------------------------------------------------------------------------

describe('renderArtifactCard / renderSkillGroup escaping', () => {
  const HOSTILE_NAME = '<img src=x onerror=alert(1)>';
  const HOSTILE_DESC = '<script>alert(1)</script>';

  test('renderArtifactCard renders a hostile name and description as literal text, not markup', () => {
    const { renderArtifactCard } = createScaffoldGalleryCards({ openDetail: () => {} });
    const section = document.createElement('div');

    renderArtifactCard(section, { name: HOSTILE_NAME, status: 'framework', description: HOSTILE_DESC }, 'agents');

    const nameEl = qs(section, '.prompts-card-name');
    expect(nameEl.textContent).toBe(HOSTILE_NAME);
    expect(nameEl.querySelector('img')).toBeNull();
    expect(nameEl.innerHTML).toContain('&lt;img');

    const descEl = qs(section, '.prompts-card-desc');
    expect(descEl.textContent).toBe(HOSTILE_DESC);
    expect(descEl.querySelector('script')).toBeNull();
  });

  test('renderSkillGroup (multi-artifact) escapes the group name', () => {
    const { renderSkillGroup } = createScaffoldGalleryCards({ openDetail: () => {} });
    const section = document.createElement('div');

    renderSkillGroup(section, HOSTILE_NAME, [
      { name: 'skills/x/one', status: 'framework', output_path: 'skills/x/one.md' },
      { name: 'skills/x/two', status: 'framework', output_path: 'skills/x/two.md' },
    ]);

    const nameEl = qs(section, '.prompts-card-name');
    expect(nameEl.textContent).toBe(HOSTILE_NAME);
    expect(nameEl.querySelector('img')).toBeNull();
  });

  test('renderSkillGroup with a single artifact renders its plain card, named for the artifact', () => {
    const { renderSkillGroup } = createScaffoldGalleryCards({ openDetail: () => {} });
    const section = document.createElement('div');

    renderSkillGroup(section, 'solo-skill', [{ name: HOSTILE_NAME, status: 'framework' }]);

    expect(section.querySelectorAll('.prompts-card').length).toBe(1);
    expect(section.querySelector('.prompts-skill-group')).toBeNull();
    expect(qs(section, '.prompts-card-name').textContent).toBe(HOSTILE_NAME);
  });
});

// ---------------------------------------------------------------------------
// createScaffoldGalleryView -- factory wiring
// ---------------------------------------------------------------------------

describe('createScaffoldGalleryView', () => {
  /** @type {any[]} */
  const ARTIFACTS = [
    { name: 'claude-md', displayCategory: 'system prompt', status: 'framework' },
    { name: 'safety-check', displayCategory: 'hooks', status: 'user-owned' },
    { name: 'agent-one', displayCategory: 'agents', status: 'framework' },
  ];

  test('unpinned categories start collapsed; pinned ones start open', () => {
    const gallery = makeGallery({ artifacts: ARTIFACTS, pinnedCategories: ['hooks'] });
    const view = createScaffoldGalleryView(gallery);

    view.renderGallery();

    const sections = [...gallery.categoriesEl.querySelectorAll('.prompts-category-section')];
    const state = sections.map((s) => [
      qs(s, '.prompts-category-label').textContent,
      qs(s, '.prompts-category-body').classList.contains('collapsed'),
    ]);
    expect(state).toEqual([['HOOKS', false], ['AGENTS', true], ['SYSTEM PROMPT', true]]);
  });

  test('a single-category gallery never seeds its only section collapsed', () => {
    // Safety (hooks only) and Config (mcp-json + settings-json, both `config`)
    // would otherwise open on a lone header above an empty panel.
    const gallery = makeGallery({
      artifacts: [{ name: 'a', displayCategory: 'hooks', status: 'framework' }],
      pinnedCategories: [],
    });
    const view = createScaffoldGalleryView(gallery);

    view.renderGallery();

    expect(qs(gallery.categoriesEl, '.prompts-category-body').classList.contains('collapsed'))
      .toBe(false);
  });

  test('collapsed sections still render their cards (hidden, not skipped)', () => {
    const gallery = makeGallery({ artifacts: ARTIFACTS, pinnedCategories: [] });
    const view = createScaffoldGalleryView(gallery);

    view.renderGallery();

    // Every section collapsed, yet all three cards exist in the DOM.
    expect(gallery.categoriesEl.querySelectorAll('.prompts-category-body.collapsed').length).toBe(3);
    expect(gallery.categoriesEl.querySelectorAll('.prompts-card').length).toBe(3);
  });

  test('clicking a category header toggles that section only', () => {
    const gallery = makeGallery({ artifacts: ARTIFACTS, pinnedCategories: ['hooks'] });
    const view = createScaffoldGalleryView(gallery);

    view.renderGallery();
    qs(gallery.categoriesEl, '.prompts-category-header').dispatchEvent(new Event('click'));

    const sections = [...gallery.categoriesEl.querySelectorAll('.prompts-category-section')];
    // HOOKS was open and is now collapsed; the others are untouched.
    expect(qs(sections[0], '.prompts-category-body').classList.contains('collapsed')).toBe(true);
    expect(qs(sections[1], '.prompts-category-body').classList.contains('collapsed')).toBe(true);
    expect(gallery.collapsedCategories.has('hooks')).toBe(true);
  });

  test('starting a search opens every section so matches cannot hide', () => {
    const gallery = makeGallery({ artifacts: ARTIFACTS, pinnedCategories: [] });
    const view = createScaffoldGalleryView(gallery);

    view.renderGallery();  // unfiltered: everything seeded collapsed
    expect(gallery.categoriesEl.querySelectorAll('.prompts-category-body.collapsed').length).toBe(3);

    gallery.searchQuery = 'a';
    view.renderGallery();

    expect(gallery.categoriesEl.querySelectorAll('.prompts-category-body.collapsed').length).toBe(0);
  });

  test('the Filter disclosure opens and closes the panel holding search + chips', () => {
    const gallery = makeGallery({ artifacts: ARTIFACTS });
    const view = createScaffoldGalleryView(gallery);

    view.renderGallery();
    expect(gallery.filterPanelEl.classList.contains('open')).toBe(false);
    expect(gallery.filterToggleEl.getAttribute('aria-expanded')).toBe('false');

    gallery.filterToggleEl.dispatchEvent(new Event('click'));
    expect(gallery.filterOpen).toBe(true);
    expect(gallery.filterPanelEl.classList.contains('open')).toBe(true);
    expect(gallery.filterToggleEl.getAttribute('aria-expanded')).toBe('true');

    gallery.filterToggleEl.dispatchEvent(new Event('click'));
    expect(gallery.filterPanelEl.classList.contains('open')).toBe(false);
  });

  test('renderCategories shows the empty state when nothing matches', () => {
    const gallery = makeGallery({ artifacts: ARTIFACTS, searchQuery: 'no-such-artifact' });
    const view = createScaffoldGalleryView(gallery);

    view.renderGallery();

    expect(gallery.categoriesEl.textContent).toContain('No matching artifacts found.');
  });

  test('clicking a category chip updates gallery.filterCategory and re-renders', () => {
    const gallery = makeGallery({ artifacts: ARTIFACTS });
    const view = createScaffoldGalleryView(gallery);

    view.renderGallery();
    const chip = [...gallery.filterChipsEl.querySelectorAll('.prompts-chip')]
      .find((c) => c.textContent === 'agents');
    expect(chip).toBeTruthy();
    if (chip === undefined) throw new Error('agents chip not found');

    chip.dispatchEvent(new Event('click'));

    expect(gallery.filterCategory).toBe('agents');
    // renderFilterChips() re-ran and marked the clicked chip active.
    const activeChip = [...gallery.filterChipsEl.querySelectorAll('.prompts-chip.active')];
    expect(activeChip.map((c) => c.textContent)).toContain('agents');
  });

  test('the project-owned toggle appears when a user-owned artifact exists, and flips gallery state', () => {
    const gallery = makeGallery({ artifacts: ARTIFACTS });
    const view = createScaffoldGalleryView(gallery);

    view.renderGallery();
    const toggle = [...gallery.filterChipsEl.querySelectorAll('.prompts-chip-toggle')][0];
    expect(toggle).toBeTruthy();
    expect(toggle.textContent).toBe('Project-owned');

    toggle.dispatchEvent(new Event('click'));
    expect(gallery.filterProjectOwned).toBe(true);
  });

  test('the project-owned toggle is absent when every artifact is framework-owned', () => {
    const gallery = makeGallery({ artifacts: ARTIFACTS.map((a) => ({ ...a, status: 'framework' })) });
    const view = createScaffoldGalleryView(gallery);

    view.renderGallery();

    expect(gallery.filterChipsEl.querySelector('.prompts-chip-toggle')).toBeNull();
  });

  test('renderSummary renders the inventory line and hides the clear button', () => {
    const gallery = makeGallery({ summary: { total: 5, framework: 3, userOwned: 2 } });
    const view = createScaffoldGalleryView(gallery);

    view.renderGallery();

    expect(gallery.summaryEl.textContent).toContain('5 artifacts');
    expect(gallery.summaryEl.textContent).toContain('2 project-owned');
    expect(gallery.clearFilterEl.style.display).toBe('none');
  });

  test('an active filter is legible on the meta line even with the panel shut', () => {
    // The whole point of collapsing the controls: a narrowed list must never
    // read as a short one.
    const gallery = makeGallery({
      artifacts: ARTIFACTS,
      summary: { total: 3, framework: 2, userOwned: 1 },
      filterCategory: 'agents',
      searchQuery: 'one',
    });
    const view = createScaffoldGalleryView(gallery);

    view.renderGallery();

    expect(gallery.summaryEl.textContent).toBe('1 of 3 · agents · "one"');
    expect(gallery.clearFilterEl.style.display).not.toBe('none');
  });

  test('the meta line\'s clear button drops every filter and re-renders', () => {
    const gallery = makeGallery({
      artifacts: ARTIFACTS,
      summary: { total: 3, framework: 2, userOwned: 1 },
      filterCategory: 'agents',
      filterProjectOwned: true,
      searchQuery: 'one',
    });
    const view = createScaffoldGalleryView(gallery);

    view.renderGallery();
    gallery.clearFilterEl.dispatchEvent(new Event('click'));

    expect(gallery.filterCategory).toBeNull();
    expect(gallery.filterProjectOwned).toBe(false);
    expect(gallery.searchQuery).toBe('');
    expect(gallery.summaryEl.textContent).toContain('3 artifacts');
  });

  test('renderUntrackedBanner hides itself when there are no untracked files', () => {
    const gallery = makeGallery({ untrackedFiles: [] });
    const view = createScaffoldGalleryView(gallery);

    view.renderGallery();

    expect(gallery.untrackedBannerEl.style.display).toBe('none');
  });

  test('renderUntrackedBanner lists files and wires register/delete to the gallery host', () => {
    let registered = null;
    let deleted = null;
    const gallery = makeGallery({
      untrackedFiles: [{ canonical_name: 'stray.md', output_path: '.claude/stray.md' }],
      registerUntracked: (name) => { registered = name; return Promise.resolve(); },
      deleteUntracked: (name) => { deleted = name; return Promise.resolve(); },
    });
    const view = createScaffoldGalleryView(gallery);

    view.renderGallery();
    expect(gallery.untrackedBannerEl.style.display).not.toBe('none');

    qs(gallery.untrackedBannerEl, '.prompts-untracked-register').dispatchEvent(new Event('click'));
    expect(registered).toBe('stray.md');

    qs(gallery.untrackedBannerEl, '.prompts-untracked-delete').dispatchEvent(new Event('click'));
    expect(deleted).toBe('stray.md');
  });

  test('renderGallery shows the gallery view, hides the detail view, and populates the grid', () => {
    const gallery = makeGallery({ artifacts: ARTIFACTS });
    gallery.detailView.style.display = '';
    const view = createScaffoldGalleryView(gallery);

    view.renderGallery();

    expect(gallery.currentView).toBe('gallery');
    expect(gallery.galleryView.style.display).toBe('');
    expect(gallery.detailView.style.display).toBe('none');
    expect(gallery.categoriesEl.querySelectorAll('.prompts-card').length).toBe(3);
  });

  test('clicking a card calls gallery.openDetail with that artifact', () => {
    let opened = null;
    const gallery = makeGallery({ artifacts: ARTIFACTS, openDetail: (a) => { opened = a; } });
    const view = createScaffoldGalleryView(gallery);

    view.renderGallery();
    const card = qs(gallery.categoriesEl, '.prompts-card[data-name="agent-one"]');
    card.dispatchEvent(new Event('click'));

    expect(opened).toEqual(ARTIFACTS.find((a) => a.name === 'agent-one'));
  });
});

// ---------------------------------------------------------------------------
// initScaffoldGallery -- which drawer tab shows which artifact
// ---------------------------------------------------------------------------

describe('initScaffoldGallery tab routing', () => {
  /**
   * One artifact per category the service reports, the tab each belongs on,
   * and its output path. The project instructions route by output path, so a
   * deployment's own persona (`claude-md-ariel`) lands where `claude-md` does.
   */
  const ROUTED = [
    ['agents/a', 'agents', 'behavior', '.claude/agents/a.md'],
    ['skills/s/SKILL', 'skills', 'behavior', '.claude/skills/s/SKILL.md'],
    ['rules/r', 'rules', 'behavior', '.claude/rules/r.md'],
    ['output-styles/o', 'output-styles', 'behavior', '.claude/output-styles/o.md'],
    ['claude-md', 'config', 'behavior', 'CLAUDE.md'],
    ['claude-md-ariel', 'config', 'behavior', 'CLAUDE.md'],
    ['hooks/h', 'hooks', 'safety', '.claude/hooks/h.py'],
    ['mcp-json', 'config', 'config', '.mcp.json'],
    ['settings-json', 'config', 'config', '.claude/settings.json'],
  ];

  afterEach(() => {
    vi.unstubAllGlobals();
    document.body.innerHTML = '';
  });

  test('each artifact lands on exactly one tab', async () => {
    document.body.innerHTML = `
      <div id="settings-drawer">
        <div id="tab-behavior"><div id="behavior-gallery-section"></div></div>
        <div id="tab-safety"><div id="safety-gallery-section"></div></div>
        <div id="tab-config">
          <div id="config-gallery-section"></div>
          <div id="config-form-section"></div>
        </div>
      </div>`;
    /** @type {any} */ (document.getElementById('settings-drawer')).registerUnsavedGuard = vi.fn();

    const artifacts = ROUTED.map(([name, category, , output_path]) => ({
      name, category, output_path, status: 'framework',
    }));
    vi.stubGlobal('fetch', vi.fn(async (/** @type {string} */ url) => ({
      ok: true,
      status: 200,
      json: async () => (url.endsWith('/untracked') ? { untracked: [] } : { artifacts }),
    })));

    vi.resetModules();
    const { initScaffoldGallery } = await import(
      '../../../src/osprey/interfaces/web_terminal/static/js/scaffold-gallery.js'
    );
    initScaffoldGallery();
    for (const tab of ['tab-behavior', 'tab-safety', 'tab-config']) {
      document.getElementById(tab)?.dispatchEvent(new Event('drawer:tab-activate'));
    }

    /** @param {string} id */
    const namesIn = (id) => [...document.querySelectorAll(`#${id} .prompts-card`)]
      .map((c) => /** @type {HTMLElement} */ (c).dataset.name);
    await vi.waitFor(() => {
      expect(namesIn('safety-gallery-section').length).toBeGreaterThan(0);
      expect(namesIn('config-gallery-section').length).toBeGreaterThan(0);
    });

    const shown = {
      behavior: namesIn('behavior-gallery-section'),
      safety: namesIn('safety-gallery-section'),
      config: namesIn('config-gallery-section'),
    };
    for (const [name, , tab] of ROUTED) {
      const tabsShowing = Object.entries(shown).filter(([, names]) => names.includes(name)).map(([t]) => t);
      expect(tabsShowing, `${name} should show only on ${tab}`).toEqual([tab]);
    }
  });
});
