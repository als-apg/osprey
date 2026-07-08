// @ts-nocheck
// TODO(frontend-hardening Pn): remove & fix types when this interface is retrofitted (P2–P5)
/**
 * Unit tests for the Scaffold Gallery view layer (scaffold/view.js).
 *
 * Pure-logic + DOM guard, happy-dom environment (configured globally):
 *   npx vitest run tests/interfaces/web_terminal/scaffold-view.test.mjs
 *
 * Covers getFilteredArtifacts' search/category/project-owned filter
 * combinations (pure, no gallery instance needed), card rendering's
 * escapeHtml contract (hostile artifact names must render as literal text,
 * never parsed as markup -- renderArtifactCard/renderSkillGroup use
 * `.textContent`, which is DOM-safe by construction), and
 * createScaffoldGalleryView's factory wiring (mirrors the pattern
 * scaffold-data.test.mjs uses for createScaffoldDataActions): a fake
 * `gallery` host object stands in for the ArtifactGallery instance, the
 * same "pass `this`" shape the real class uses in its constructor.
 *
 * NOTE: imported by RELATIVE path -- this module lives under web_terminal,
 * not design-system, so the `/design-system/js/*` alias does not apply.
 */

import { test, expect, describe, beforeEach } from 'vitest';

import {
  getFilteredArtifacts,
  createScaffoldGalleryView,
} from '../../../src/osprey/interfaces/web_terminal/static/js/scaffold/view.js';

/** @param {object} [overrides] */
function makeGallery(overrides = {}) {
  return {
    artifacts: [],
    untrackedFiles: [],
    currentView: 'gallery',
    filterCategory: null,
    filterProjectOwned: false,
    searchQuery: '',
    pinnedCategories: [],
    summary: { total: 0, framework: 0, userOwned: 0 },
    galleryView: document.createElement('div'),
    detailView: document.createElement('div'),
    untrackedBannerEl: document.createElement('div'),
    filterChipsEl: document.createElement('div'),
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
  };
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

  test('no filters returns every artifact', () => {
    const result = getFilteredArtifacts(ARTIFACTS, {
      filterProjectOwned: false, filterCategory: null, searchQuery: '',
    });
    expect(result).toEqual(ARTIFACTS);
  });

  test('filterProjectOwned keeps only user-owned artifacts', () => {
    const result = getFilteredArtifacts(ARTIFACTS, {
      filterProjectOwned: true, filterCategory: null, searchQuery: '',
    });
    expect(result.map((a) => a.name)).toEqual(['safety-check', 'lattice-agent']);
  });

  test('filterCategory keeps only matching displayCategory', () => {
    const result = getFilteredArtifacts(ARTIFACTS, {
      filterProjectOwned: false, filterCategory: 'agents', searchQuery: '',
    });
    expect(result.map((a) => a.name)).toEqual(['channel-finder', 'lattice-agent']);
  });

  test('searchQuery matches name, description, or summary case-insensitively', () => {
    const byName = getFilteredArtifacts(ARTIFACTS, {
      filterProjectOwned: false, filterCategory: null, searchQuery: 'CLAUDE',
    });
    expect(byName.map((a) => a.name)).toEqual(['claude-md']);

    const byDescription = getFilteredArtifacts(ARTIFACTS, {
      filterProjectOwned: false, filterCategory: null, searchQuery: 'unsafe',
    });
    expect(byDescription.map((a) => a.name)).toEqual(['safety-check']);

    const bySummary = getFilteredArtifacts(ARTIFACTS, {
      filterProjectOwned: false, filterCategory: null, searchQuery: 'beamline',
    });
    expect(bySummary.map((a) => a.name)).toEqual(['channel-finder']);
  });

  test('combines project-owned + category + search as an intersection', () => {
    const result = getFilteredArtifacts(ARTIFACTS, {
      filterProjectOwned: true, filterCategory: 'agents', searchQuery: 'lattice',
    });
    expect(result.map((a) => a.name)).toEqual(['lattice-agent']);
  });

  test('a combination with no matches returns an empty array', () => {
    const result = getFilteredArtifacts(ARTIFACTS, {
      filterProjectOwned: true, filterCategory: 'system prompt', searchQuery: '',
    });
    expect(result).toEqual([]);
  });

  test('tolerates artifacts missing description/summary fields', () => {
    const sparse = [{ name: 'bare', displayCategory: 'config', status: 'framework' }];
    const result = getFilteredArtifacts(sparse, {
      filterProjectOwned: false, filterCategory: null, searchQuery: 'bare',
    });
    expect(result).toEqual(sparse);
  });
});

// ---------------------------------------------------------------------------
// Card rendering -- escapeHtml contract (hostile names render as text)
// ---------------------------------------------------------------------------

describe('renderArtifactCard / renderSkillGroup escaping', () => {
  const HOSTILE_NAME = '<img src=x onerror=alert(1)>';

  test('renderArtifactCard renders a hostile artifact name as literal text, not markup', () => {
    const gallery = makeGallery();
    const view = createScaffoldGalleryView(gallery);
    const section = document.createElement('div');

    view.renderArtifactCard(section, { name: HOSTILE_NAME, status: 'framework' }, 'agents');

    const nameEl = section.querySelector('.prompts-card-name');
    expect(nameEl.textContent).toBe(HOSTILE_NAME);
    // No actual <img> element was parsed into the DOM.
    expect(nameEl.querySelector('img')).toBeNull();
    expect(nameEl.innerHTML).toContain('&lt;img');
  });

  test('renderArtifactCard escapes a hostile description/summary the same way', () => {
    const gallery = makeGallery();
    const view = createScaffoldGalleryView(gallery);
    const section = document.createElement('div');

    view.renderArtifactCard(
      section,
      { name: 'safe-name', status: 'framework', description: '<script>alert(1)</script>' },
      'agents'
    );

    const descEl = section.querySelector('.prompts-card-desc');
    expect(descEl.textContent).toBe('<script>alert(1)</script>');
    expect(descEl.querySelector('script')).toBeNull();
  });

  test('renderSkillGroup (multi-artifact) escapes the group name and per-option labels', () => {
    const gallery = makeGallery();
    const view = createScaffoldGalleryView(gallery);
    const section = document.createElement('div');

    view.renderSkillGroup(section, HOSTILE_NAME, [
      { name: 'skills/x/one', status: 'framework', output_path: 'skills/x/one.md' },
      { name: 'skills/x/two', status: 'framework', output_path: 'skills/x/two.md' },
    ]);

    const nameEl = section.querySelector('.prompts-card-name');
    expect(nameEl.textContent).toBe(HOSTILE_NAME);
    expect(nameEl.querySelector('img')).toBeNull();
  });

  test('renderSkillGroup with a single artifact delegates to renderArtifactCard (still escaped)', () => {
    const gallery = makeGallery();
    const view = createScaffoldGalleryView(gallery);
    const section = document.createElement('div');

    view.renderSkillGroup(section, 'solo-skill', [{ name: HOSTILE_NAME, status: 'framework' }]);

    expect(section.querySelectorAll('.prompts-card').length).toBe(1);
    expect(section.querySelector('.prompts-card-name').textContent).toBe(HOSTILE_NAME);
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

  test('renderCategories groups by displayCategory, pinned categories first', () => {
    const gallery = makeGallery({ artifacts: ARTIFACTS, pinnedCategories: ['hooks'] });
    const view = createScaffoldGalleryView(gallery);

    view.renderCategories();

    const headers = [...gallery.categoriesEl.querySelectorAll('.prompts-category-header span:first-child')]
      .map((el) => el.textContent);
    expect(headers[0]).toBe('HOOKS');
    expect(gallery.categoriesEl.querySelectorAll('.prompts-card').length).toBe(3);
  });

  test('renderCategories shows the empty state when nothing matches', () => {
    const gallery = makeGallery({ artifacts: ARTIFACTS, searchQuery: 'no-such-artifact' });
    const view = createScaffoldGalleryView(gallery);

    view.renderCategories();

    expect(gallery.categoriesEl.textContent).toContain('No matching artifacts found.');
  });

  test('getFilteredArtifacts (bound) reads the gallery\'s current filter state', () => {
    const gallery = makeGallery({ artifacts: ARTIFACTS, filterCategory: 'agents' });
    const view = createScaffoldGalleryView(gallery);

    expect(view.getFilteredArtifacts().map((a) => a.name)).toEqual(['agent-one']);
  });

  test('clicking a category chip updates gallery.filterCategory and re-renders', () => {
    const gallery = makeGallery({ artifacts: ARTIFACTS });
    const view = createScaffoldGalleryView(gallery);

    view.renderFilterChips();
    const chip = [...gallery.filterChipsEl.querySelectorAll('.prompts-chip')]
      .find((c) => c.textContent === 'agents');
    expect(chip).toBeTruthy();

    chip.dispatchEvent(new Event('click'));

    expect(gallery.filterCategory).toBe('agents');
    // renderFilterChips() re-ran and marked the clicked chip active.
    const activeChip = [...gallery.filterChipsEl.querySelectorAll('.prompts-chip.active')];
    expect(activeChip.map((c) => c.textContent)).toContain('agents');
  });

  test('the project-owned toggle only appears when a user-owned artifact exists, and flips gallery state', () => {
    const gallery = makeGallery({ artifacts: ARTIFACTS });
    const view = createScaffoldGalleryView(gallery);

    view.renderFilterChips();
    const toggle = [...gallery.filterChipsEl.querySelectorAll('.prompts-chip-toggle')][0];
    expect(toggle).toBeTruthy();
    expect(toggle.textContent).toBe('Project-owned');

    toggle.dispatchEvent(new Event('click'));
    expect(gallery.filterProjectOwned).toBe(true);
  });

  test('renderSummary renders the totals line', () => {
    const gallery = makeGallery({ summary: { total: 5, framework: 3, userOwned: 2 } });
    const view = createScaffoldGalleryView(gallery);

    view.renderSummary();

    expect(gallery.summaryEl.textContent).toContain('5 artifacts');
    expect(gallery.summaryEl.textContent).toContain('3 framework');
    expect(gallery.summaryEl.textContent).toContain('2 project-owned');
  });

  test('renderUntrackedBanner hides itself when there are no untracked files', () => {
    const gallery = makeGallery({ untrackedFiles: [] });
    const view = createScaffoldGalleryView(gallery);

    view.renderUntrackedBanner();

    expect(gallery.untrackedBannerEl.style.display).toBe('none');
  });

  test('renderUntrackedBanner lists files and wires register/delete to the gallery host', () => {
    let registered = null;
    let deleted = null;
    const gallery = makeGallery({
      untrackedFiles: [{ canonical_name: 'stray.md', output_path: '.claude/stray.md' }],
      registerUntracked: (name) => { registered = name; },
      deleteUntracked: (name) => { deleted = name; },
    });
    const view = createScaffoldGalleryView(gallery);

    view.renderUntrackedBanner();
    expect(gallery.untrackedBannerEl.style.display).not.toBe('none');

    gallery.untrackedBannerEl.querySelector('.prompts-untracked-register').dispatchEvent(new Event('click'));
    expect(registered).toBe('stray.md');

    gallery.untrackedBannerEl.querySelector('.prompts-untracked-delete').dispatchEvent(new Event('click'));
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

    view.renderCategories();
    const card = gallery.categoriesEl.querySelector('.prompts-card[data-name="agent-one"]');
    card.dispatchEvent(new Event('click'));

    expect(opened).toEqual(ARTIFACTS.find((a) => a.name === 'agent-one'));
  });
});
