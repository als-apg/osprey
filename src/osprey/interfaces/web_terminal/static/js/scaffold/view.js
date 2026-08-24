// @ts-check
/**
 * OSPREY Web Terminal — Scaffold Gallery: view layer
 *
 * Gallery-view rendering (the muted meta line, the collapsible filter panel
 * holding search + chips, the untracked-file banner, and the collapsible
 * category/card grid) plus the pure artifact-list filter behind it.
 *
 * Chrome budget: the gallery's only permanent row is the meta line. Search and
 * the category chips live behind its Filter disclosure, and every category but
 * the pinned ones starts collapsed, so opening a tab shows artifacts rather
 * than controls. The cost of hiding controls is that an active filter must
 * stay legible with the panel shut -- hence the "N of M · <what>" readout and
 * the clear-all ✕ on the meta line itself.
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
import { scaffoldWritesEnabled } from './write-gate.js';
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
 * @property {boolean} [filterOpen]
 * @property {Set<string>} [collapsedCategories]
 * @property {Set<string>} [seenCategories]
 * @property {boolean} [_lastFiltering]
 * @property {string[]} pinnedCategories
 * @property {{total: number, framework: number, userOwned: number}} summary
 * @property {HTMLElement|null} galleryView
 * @property {HTMLElement|null} detailView
 * @property {HTMLElement|null} untrackedBannerEl
 * @property {HTMLElement|null} filterChipsEl
 * @property {HTMLElement|null} [filterPanelEl]
 * @property {HTMLElement|null} [filterToggleEl]
 * @property {HTMLElement|null} [clearFilterEl]
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

  /**
   * True while the artifact list is narrowed by anything -- a category chip,
   * the project-owned toggle, or a search query. Drives both the meta line's
   * "N of M" readout and the category seeding below, so a narrowed list can
   * never be mistaken for a short one.
   * @returns {boolean}
   */
  function isFiltering() {
    return Boolean(gallery.filterCategory || gallery.filterProjectOwned || gallery.searchQuery);
  }

  /** @returns {void} */
  function renderGallery() {
    if (gallery.galleryView) gallery.galleryView.style.display = '';
    if (gallery.detailView) gallery.detailView.style.display = 'none';

    gallery.currentView = 'gallery';

    renderUntrackedBanner();
    renderFilterChips();
    bindSearch();
    renderFilterToggle();
    renderCategories();  // ends with renderSummary()
  }

  /**
   * Paint the Filter disclosure and its panel to match `gallery.filterOpen`,
   * and (re)bind the toggle. Assignment to `onclick` rather than
   * addEventListener keeps this idempotent across re-renders -- the same
   * reason bindSearch() clones its input.
   * @returns {void}
   */
  function renderFilterToggle() {
    const btn = gallery.filterToggleEl;
    const panel = gallery.filterPanelEl;
    if (!btn || !panel) return;

    const open = Boolean(gallery.filterOpen);
    btn.textContent = open ? 'Filter ⌃' : 'Filter ⌄';
    btn.setAttribute('aria-expanded', String(open));
    panel.classList.toggle('open', open);

    btn.onclick = () => {
      gallery.filterOpen = !gallery.filterOpen;
      renderFilterToggle();
      if (gallery.filterOpen && gallery.searchInput) gallery.searchInput.focus();
    };
  }

  /**
   * Drop every active filter and re-render. Wired to the meta line's ✕, which
   * is the escape hatch for a filter left active behind a collapsed panel.
   * @returns {void}
   */
  function clearFilters() {
    gallery.filterCategory = null;
    gallery.filterProjectOwned = false;
    gallery.searchQuery = '';
    if (gallery.searchInput) gallery.searchInput.value = '';
    renderFilterChips();
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
    title.textContent = `${n} file${n > 1 ? 's' : ''} active in the agent's setup but not managed by OSPREY`;
    header.appendChild(title);

    banner.appendChild(header);

    const desc = _el('div', 'prompts-untracked-desc');
    desc.textContent =
      'These files are in .claude/ and take effect in the next session, but they are not tracked in your project config — a rebuild will not reproduce them. Register them to manage through this UI, or delete them.';
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

      // Register rewrites config.yml and Delete removes the file — both are
      // writes, and both answer 403 with `web.scaffold_gallery.write_enabled:
      // false`, so neither is offered where writes are withdrawn. The banner
      // itself stays: naming files that are reaching the agent unmanaged is
      // information a read-only tier still needs, even when acting on them is
      // somebody else's call.
      if (scaffoldWritesEnabled()) {
        const registerBtn = document.createElement('button');
        registerBtn.className = 'prompts-untracked-btn prompts-untracked-register';
        registerBtn.textContent = 'Register';
        registerBtn.title = 'Add to project config so this file is managed by OSPREY';
        registerBtn.addEventListener('click', () => gallery.registerUntracked(file.canonical_name));
        actions.appendChild(registerBtn);

        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'prompts-untracked-btn prompts-untracked-delete';
        deleteBtn.textContent = 'Delete';
        deleteBtn.title = 'Remove this file from disk — it will no longer reach the agent';
        deleteBtn.addEventListener('click', () => gallery.deleteUntracked(file.canonical_name));
        actions.appendChild(deleteBtn);
      }

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

  /**
   * Paint the muted meta line. Unfiltered it is a plain inventory; filtered it
   * names every active narrowing and reveals the clear-all ✕. The framework
   * count is deliberately gone -- it is just total minus project-owned, and
   * this line is the gallery's only permanent chrome.
   * @returns {void}
   */
  function renderSummary() {
    if (!gallery.summaryEl) return;

    const total = gallery.summary.total;
    const owned = gallery.summary.userOwned;

    if (!isFiltering()) {
      gallery.summaryEl.textContent =
        `${total} artifact${total === 1 ? '' : 's'}` +
        (owned > 0 ? ` · ${owned} project-owned` : '');
      if (gallery.clearFilterEl) gallery.clearFilterEl.style.display = 'none';
      return;
    }

    /** @type {string[]} */
    const active = [];
    if (gallery.filterCategory) active.push(gallery.filterCategory);
    if (gallery.filterProjectOwned) active.push('project-owned');
    if (gallery.searchQuery) active.push(`"${gallery.searchQuery}"`);

    gallery.summaryEl.textContent =
      `${boundGetFilteredArtifacts().length} of ${total} · ${active.join(' · ')}`;

    if (gallery.clearFilterEl) {
      gallery.clearFilterEl.style.display = '';
      gallery.clearFilterEl.onclick = clearFilters;
    }
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

    // Collapse bookkeeping. A category is seeded collapsed the first time it
    // is seen unless it is pinned; after that the operator's own toggles win.
    // Crossing the filtered/unfiltered boundary resets both sets, so starting
    // a search always opens everything (matches can never hide behind a
    // collapsed header) and clearing it restores the pinned-only default.
    //
    // Collapsing only earns its keep when there is more than one section to
    // choose between: the single-category galleries (Safety's hooks, Config's
    // two files) would otherwise open on a lone header over an empty panel.
    // Their operator can still collapse it by hand.
    const collapsed = gallery.collapsedCategories || (gallery.collapsedCategories = new Set());
    const seen = gallery.seenCategories || (gallery.seenCategories = new Set());
    const filtering = isFiltering();
    const seedCollapsed = !filtering && sortedCategories.length > 1;
    if (filtering !== gallery._lastFiltering) {
      gallery._lastFiltering = filtering;
      collapsed.clear();
      seen.clear();
    }

    for (const cat of sortedCategories) {
      if (!seen.has(cat)) {
        seen.add(cat);
        if (seedCollapsed && !pinned.includes(cat)) collapsed.add(cat);
      }
      const isCollapsed = collapsed.has(cat);

      const section = document.createElement('div');
      section.className = 'prompts-category-section';

      // Category header — the disclosure control for its own section.
      const header = document.createElement('div');
      header.className = 'prompts-category-header';
      header.setAttribute('role', 'button');
      header.tabIndex = 0;
      header.setAttribute('aria-expanded', String(!isCollapsed));

      const chevron = document.createElement('span');
      chevron.className = 'prompts-category-chevron';
      chevron.setAttribute('aria-hidden', 'true');
      chevron.textContent = isCollapsed ? '▸' : '▾';
      header.appendChild(chevron);

      const label = document.createElement('span');
      label.className = 'prompts-category-label';
      label.textContent = cat.toUpperCase();
      header.appendChild(label);

      const toggleSection = () => {
        if (collapsed.has(cat)) collapsed.delete(cat);
        else collapsed.add(cat);
        renderCategories();
      };
      header.addEventListener('click', toggleSection);
      header.addEventListener('keydown', (e) => {
        const key = /** @type {KeyboardEvent} */ (e).key;
        if (key === 'Enter' || key === ' ') {
          e.preventDefault();
          toggleSection();
        }
      });

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
      // No 'commands': no gallery's categoryFilter admits that category, so a
      // commands section can never render and the branch was unreachable.
      const creatableCategories = new Set([
        'agents', 'rules', 'hooks', 'skills', 'output-styles'
      ]);
      const originalCat = groups[cat][0]?.category || cat;
      // POST /api/scaffold/create is gated server-side, so a deployment that
      // withdrew gallery writes gets no "+" — the dialog behind it could only
      // end in a 403.
      if (creatableCategories.has(originalCat.toLowerCase()) && scaffoldWritesEnabled()) {
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

      // Cards live in their own wrapper so collapsing is a class flip on one
      // element. They are always BUILT -- collapsing hides them in CSS rather
      // than skipping the render, so a collapsed section still answers to the
      // search index and to card queries.
      const body = document.createElement('div');
      body.className = 'prompts-category-body';
      if (isCollapsed) body.classList.add('collapsed');

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
          renderSkillGroup(body, skillName, groupArts);
        }
      } else {
        for (const artifact of groups[cat]) {
          renderArtifactCard(body, artifact, cat);
        }
      }

      section.appendChild(body);
      categoriesEl.appendChild(section);
    }

    if (filtered.length === 0) {
      categoriesEl.innerHTML = '<div class="prompts-empty">No matching artifacts found.</div>';
    }

    renderSummary();
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
    renderFilterToggle,
    clearFilters,
    renderSummary,
    bindSearch,
    renderCategories,
    renderArtifactCard,
    renderSkillGroup,
    getFilteredArtifacts: boundGetFilteredArtifacts,
  };
}
