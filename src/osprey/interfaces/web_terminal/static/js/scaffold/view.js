// @ts-check
/**
 * OSPREY Web Terminal — Scaffold Gallery: view layer
 *
 * Gallery-view rendering (search bar, filter chips, untracked-file banner,
 * summary line, and the category/card grid) plus the pure artifact-list
 * filter behind it.
 *
 * Mirrors the factory/injection pattern scaffold/data.js and
 * lattice_dashboard/render.js already use: {@link createScaffoldGalleryView}
 * is a factory bound to a gallery's DOM refs and mutable filter/view state
 * (passed in as `gallery` -- the same "pass `this`" shape
 * createScaffoldDataActions(this, ...) uses in scaffold-gallery.js's
 * constructor), so this module has no dependency on the rest of
 * scaffold-gallery.js (detail view, edit/save, ownership actions). The one
 * genuinely pure piece, {@link getFilteredArtifacts}, is also exported
 * standalone so it's unit-testable with plain fixture data and no gallery
 * instance at all.
 *
 * @module scaffold/view
 */

import { el as _el, debounce } from '/design-system/js/dom.js';
import { CATEGORY_HELP } from './utils.js';
import { createScaffoldGalleryCards } from './cards.js';

/**
 * The subset of an ArtifactGallery instance these view functions read,
 * write, or call into.
 * @typedef {object} ScaffoldGalleryHost
 * @property {any[]} artifacts
 * @property {any[]} untrackedFiles
 * @property {string} currentView
 * @property {string|null} filterCategory
 * @property {boolean} filterProjectOwned
 * @property {string} searchQuery
 * @property {string[]} pinnedCategories
 * @property {{total: number, framework: number, userOwned: number}} summary
 * @property {HTMLElement|null} galleryView
 * @property {HTMLElement|null} detailView
 * @property {HTMLElement|null} untrackedBannerEl
 * @property {HTMLElement|null} filterChipsEl
 * @property {HTMLElement|null} summaryEl
 * @property {HTMLInputElement|null} searchInput
 * @property {HTMLElement|null} categoriesEl
 * @property {(canonicalName: string) => Promise<any>} registerUntracked
 * @property {(canonicalName: string) => Promise<any>} deleteUntracked
 * @property {(artifact: any) => void} openDetail
 * @property {(category: string) => void} showCreateDialog
 */

/**
 * @typedef {object} GalleryFilters
 * @property {boolean} filterProjectOwned
 * @property {string|null} filterCategory
 * @property {string} searchQuery
 */

// ---- Filtering (pure) ---- //

/**
 * Apply the project-owned / category / search-query filters to a flat
 * artifact list, in that order -- matches the original instance method
 * exactly (each filter narrows the previous result, so order matters for
 * behavior parity even though these three predicates commute in practice).
 *
 * @param {any[]} artifacts
 * @param {GalleryFilters} filters
 * @returns {any[]}
 */
export function getFilteredArtifacts(artifacts, filters) {
  let result = artifacts;

  if (filters.filterProjectOwned) {
    result = result.filter((a) => a.status === 'user-owned');
  }

  if (filters.filterCategory) {
    result = result.filter((a) => a.displayCategory === filters.filterCategory);
  }

  if (filters.searchQuery) {
    const q = filters.searchQuery.toLowerCase();
    result = result.filter((a) => {
      const name = (a.name || '').toLowerCase();
      const desc = (a.description || '').toLowerCase();
      const sum = (a.summary || '').toLowerCase();
      return name.includes(q) || desc.includes(q) || sum.includes(q);
    });
  }

  return result;
}

// ---- View Factory ---- //

/**
 * Create the scaffold gallery's view-rendering functions, bound to a fixed
 * gallery host (its DOM refs and mutable filter/view state).
 *
 * @param {ScaffoldGalleryHost} gallery
 */
export function createScaffoldGalleryView(gallery) {
  // Card templates (renderArtifactCard/renderSkillGroup) live in
  // scaffold/cards.js -- the one piece of the original view.js that
  // doesn't touch search/filter/category-grid state, split out to keep
  // both modules under the plan's 450-line cap.
  const { renderArtifactCard, renderSkillGroup } = createScaffoldGalleryCards(gallery);

  /**
   * Toggle a category help tooltip on/off. Private to this module -- its
   * only caller is renderCategories()'s help-button handler.
   * @param {HTMLElement} btn
   * @param {string} text
   * @returns {void}
   */
  function toggleCategoryTooltip(btn, text) {
    const parent = btn.parentElement;
    if (!parent) return;

    const existing = parent.querySelector('.prompts-category-tooltip');
    if (existing) {
      existing.remove();
      return;
    }

    document.querySelectorAll('.prompts-category-tooltip').forEach((t) => t.remove());

    const tip = document.createElement('div');
    tip.className = 'prompts-category-tooltip';
    tip.textContent = text;
    parent.appendChild(tip);

    /** @param {MouseEvent} e */
    const handler = (e) => {
      if (!tip.contains(/** @type {Node|null} */ (e.target)) && e.target !== btn) {
        tip.remove();
        document.removeEventListener('click', handler);
      }
    };
    setTimeout(() => document.addEventListener('click', handler), 0);
  }

  /** @returns {void} */
  function renderGallery() {
    if (gallery.galleryView) gallery.galleryView.style.display = '';
    if (gallery.detailView) gallery.detailView.style.display = 'none';

    gallery.currentView = 'gallery';

    renderUntrackedBanner();
    renderFilterChips();
    renderSummary();
    bindSearch();
    renderCategories();
  }

  /** @returns {void} */
  function renderUntrackedBanner() {
    const banner = gallery.untrackedBannerEl;
    if (!banner) return;

    if (!gallery.untrackedFiles || gallery.untrackedFiles.length === 0) {
      banner.style.display = 'none';
      return;
    }

    banner.style.display = '';
    banner.innerHTML = '';

    const header = _el('div', 'prompts-untracked-header');
    const icon = _el('span', 'prompts-untracked-icon');
    icon.textContent = '⚠';
    header.appendChild(icon);

    const title = _el('span', 'prompts-untracked-title');
    const n = gallery.untrackedFiles.length;
    title.textContent = `${n} file${n > 1 ? 's' : ''} active in Claude Code but not managed by OSPREY`;
    header.appendChild(title);

    banner.appendChild(header);

    const desc = _el('div', 'prompts-untracked-desc');
    desc.textContent =
      'These files are in .claude/ and will be loaded by Claude Code, but they are not tracked in your project config. Register them to manage through this UI, or delete them.';
    banner.appendChild(desc);

    const list = _el('div', 'prompts-untracked-list');

    for (const file of gallery.untrackedFiles) {
      const row = _el('div', 'prompts-untracked-row');

      const info = _el('div', 'prompts-untracked-info');
      const nameEl = _el('span', 'prompts-untracked-name');
      nameEl.textContent = file.canonical_name;
      info.appendChild(nameEl);

      const pathEl = _el('span', 'prompts-untracked-path');
      pathEl.textContent = file.output_path;
      info.appendChild(pathEl);

      row.appendChild(info);

      const actions = _el('div', 'prompts-untracked-actions');

      const registerBtn = document.createElement('button');
      registerBtn.className = 'prompts-untracked-btn prompts-untracked-register';
      registerBtn.textContent = 'Register';
      registerBtn.title = 'Add to project config so this file is managed by OSPREY';
      registerBtn.addEventListener('click', () => gallery.registerUntracked(file.canonical_name));
      actions.appendChild(registerBtn);

      const deleteBtn = document.createElement('button');
      deleteBtn.className = 'prompts-untracked-btn prompts-untracked-delete';
      deleteBtn.textContent = 'Delete';
      deleteBtn.title = 'Remove this file from disk — it will no longer affect Claude Code';
      deleteBtn.addEventListener('click', () => gallery.deleteUntracked(file.canonical_name));
      actions.appendChild(deleteBtn);

      row.appendChild(actions);
      list.appendChild(row);
    }

    banner.appendChild(list);
  }

  /** @returns {void} */
  function renderFilterChips() {
    const chipsEl = gallery.filterChipsEl;
    if (!chipsEl) return;
    chipsEl.innerHTML = '';

    const categories = [...new Set(gallery.artifacts.map((a) => a.displayCategory))].sort();

    // "All" chip
    const allChip = document.createElement('button');
    allChip.className = 'prompts-chip' + (gallery.filterCategory === null ? ' active' : '');
    allChip.textContent = 'All';
    allChip.addEventListener('click', () => {
      gallery.filterCategory = null;
      renderFilterChips();
      renderCategories();
    });
    chipsEl.appendChild(allChip);

    // Per-category chips
    for (const cat of categories) {
      const chip = document.createElement('button');
      chip.className = 'prompts-chip' + (gallery.filterCategory === cat ? ' active' : '');
      chip.textContent = cat;
      chip.addEventListener('click', () => {
        gallery.filterCategory = cat;
        renderFilterChips();
        renderCategories();
      });
      chipsEl.appendChild(chip);
    }

    // "Project-owned" toggle
    const hasUserOwned = gallery.artifacts.some((a) => a.status === 'user-owned');
    if (hasUserOwned) {
      const sep = document.createElement('span');
      sep.className = 'prompts-chip-sep';
      chipsEl.appendChild(sep);

      const ownedChip = document.createElement('button');
      ownedChip.className = 'prompts-chip prompts-chip-toggle' + (gallery.filterProjectOwned ? ' active' : '');
      ownedChip.textContent = 'Project-owned';
      ownedChip.addEventListener('click', () => {
        gallery.filterProjectOwned = !gallery.filterProjectOwned;
        renderFilterChips();
        renderCategories();
      });
      chipsEl.appendChild(ownedChip);
    }
  }

  /** @returns {void} */
  function renderSummary() {
    if (!gallery.summaryEl) return;
    gallery.summaryEl.textContent =
      `${gallery.summary.total} artifacts · ${gallery.summary.framework} framework · ${gallery.summary.userOwned} project-owned`;
  }

  /** @returns {void} */
  function bindSearch() {
    if (!gallery.searchInput || !gallery.searchInput.parentNode) return;

    // Remove previous listener by replacing the element.
    const clone = /** @type {HTMLInputElement} */ (gallery.searchInput.cloneNode(true));
    gallery.searchInput.parentNode.replaceChild(clone, gallery.searchInput);
    gallery.searchInput = clone;

    clone.value = gallery.searchQuery;

    const debouncedRender = debounce(() => {
      gallery.searchQuery = clone.value.trim();
      renderCategories();
    }, 150);

    clone.addEventListener('input', debouncedRender);
  }

  /** @returns {void} */
  function renderCategories() {
    const categoriesEl = gallery.categoriesEl;
    if (!categoriesEl) return;
    categoriesEl.innerHTML = '';

    const filtered = getFilteredArtifacts(gallery.artifacts, {
      filterProjectOwned: gallery.filterProjectOwned,
      filterCategory: gallery.filterCategory,
      searchQuery: gallery.searchQuery,
    });

    // Group by display category.
    /** @type {Record<string, any[]>} */
    const groups = {};
    for (const artifact of filtered) {
      const cat = artifact.displayCategory || 'other';
      if (!groups[cat]) groups[cat] = [];
      groups[cat].push(artifact);
    }

    // Sort categories with pinned ones first, then alphabetical.
    const pinned = gallery.pinnedCategories;
    const sortedCategories = Object.keys(groups).sort((a, b) => {
      const aPin = pinned.indexOf(a);
      const bPin = pinned.indexOf(b);
      if (aPin >= 0 && bPin >= 0) return aPin - bPin;
      if (aPin >= 0) return -1;
      if (bPin >= 0) return 1;
      return a.localeCompare(b);
    });
    for (const cat of sortedCategories) {
      const section = document.createElement('div');
      section.className = 'prompts-category-section';

      // Category header.
      const header = document.createElement('div');
      header.className = 'prompts-category-header';

      const label = document.createElement('span');
      label.textContent = cat.toUpperCase();
      header.appendChild(label);

      const count = document.createElement('span');
      count.className = 'prompts-category-count';
      count.textContent = String(groups[cat].length);
      header.appendChild(count);

      const helpText = CATEGORY_HELP[cat.toLowerCase()];
      if (helpText) {
        const helpBtn = document.createElement('button');
        helpBtn.className = 'prompts-category-help';
        helpBtn.textContent = '?';
        helpBtn.title = helpText;
        helpBtn.addEventListener('click', (e) => {
          e.stopPropagation();
          toggleCategoryTooltip(helpBtn, helpText);
        });
        header.appendChild(helpBtn);
      }

      // "+" create button — use original category (not display-remapped).
      const creatableCategories = new Set([
        'agents', 'rules', 'hooks', 'skills', 'commands', 'output-styles'
      ]);
      const originalCat = groups[cat][0]?.category || cat;
      if (creatableCategories.has(originalCat.toLowerCase())) {
        const addBtn = document.createElement('button');
        addBtn.className = 'prompts-category-add';
        addBtn.textContent = '+';
        addBtn.title = `Create new ${originalCat.toLowerCase().replace(/s$/, '')}`;
        addBtn.addEventListener('click', (e) => {
          e.stopPropagation();
          gallery.showCreateDialog(originalCat.toLowerCase());
        });
        header.appendChild(addBtn);
      }

      section.appendChild(header);

      // Skills get special grouping.
      if (cat.toLowerCase() === 'skills') {
        /** @type {Record<string, any[]>} */
        const skillGroups = {};
        for (const art of groups[cat]) {
          const parts = art.name.split('/');
          const skillName = parts[1] || parts[0];
          if (!skillGroups[skillName]) skillGroups[skillName] = [];
          skillGroups[skillName].push(art);
        }
        for (const [skillName, groupArts] of Object.entries(skillGroups).sort()) {
          renderSkillGroup(section, skillName, groupArts);
        }
      } else {
        for (const artifact of groups[cat]) {
          renderArtifactCard(section, artifact, cat);
        }
      }

      categoriesEl.appendChild(section);
    }

    if (filtered.length === 0) {
      categoriesEl.innerHTML = '<div class="prompts-empty">No matching artifacts found.</div>';
    }
  }

  /** @returns {any[]} */
  function boundGetFilteredArtifacts() {
    return getFilteredArtifacts(gallery.artifacts, {
      filterProjectOwned: gallery.filterProjectOwned,
      filterCategory: gallery.filterCategory,
      searchQuery: gallery.searchQuery,
    });
  }

  return {
    renderGallery,
    renderUntrackedBanner,
    renderFilterChips,
    renderSummary,
    bindSearch,
    renderCategories,
    renderArtifactCard,
    renderSkillGroup,
    getFilteredArtifacts: boundGetFilteredArtifacts,
  };
}
