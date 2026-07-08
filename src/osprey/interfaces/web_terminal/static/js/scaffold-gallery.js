// @ts-nocheck
// TODO(frontend-hardening Pn): remove & fix types when this interface is retrofitted (P2–P5)
/* OSPREY Web Terminal — Scaffold Gallery
 *
 * Drives the "Scaffold Gallery" UI inside the settings drawer tab panels.
 * Provides a reusable ArtifactGallery class that can be instantiated
 * multiple times for different tab panels (Behavior, Safety, Config).
 *
 *   - Gallery view: filterable/searchable card grid grouped by category
 *   - Detail view: preview (rendered markdown / highlighted code), diff, and edit modes
 *   - Claim/override workflow for customizing framework build artifacts
 *
 * API endpoints consumed:
 *   GET    /api/scaffold                          -> list all artifacts
 *   GET    /api/scaffold/{name}                   -> artifact content (active layer)
 *   GET    /api/scaffold/{name}/framework         -> framework-layer content
 *   GET    /api/scaffold/{name}/diff              -> unified diff between layers
 *   POST   /api/scaffold/{name}/claim          -> create override scaffold
 *   PUT    /api/scaffold/{name}/override           -> save override content
 *   DELETE /api/scaffold/{name}/override?delete_file=true -> remove override
 */

import { el as _el } from '/design-system/js/dom.js';
import {
  BEHAVIOR_CATEGORIES,
  BEHAVIOR_NAMES,
  SAFETY_CATEGORIES,
  CONFIG_NAMES,
  configureMarked,
} from './scaffold/utils.js';
import {
  resetFetchCache,
  createScaffoldDataActions,
} from './scaffold/data.js';
import { createScaffoldGalleryView } from './scaffold/view.js';
import { createScaffoldGalleryDetail } from './scaffold/detail.js';
import { createScaffoldGalleryEditForm } from './scaffold/edit-form.js';
import { createScaffoldGalleryEdit } from './scaffold/edit.js';

// ---- ArtifactGallery Class ---- //

/**
 * A self-contained gallery widget that renders a filtered set of artifacts
 * inside a given container element.
 *
 * @param {Object} config
 * @param {HTMLElement} config.container - DOM element to render into
 * @param {(artifact) => boolean} config.categoryFilter - filter function
 * @param {Object} [config.options]
 * @param {boolean} [config.options.showSearch=true]
 * @param {boolean} [config.options.showSummary=true]
 * @param {boolean} [config.options.showFilterChips=true]
 * @param {() => void} [config.options.onDetailOpen]
 * @param {() => void} [config.options.onDetailClose]
 */
class ArtifactGallery {
  constructor({ container, categoryFilter, options = {} }) {
    this.container = container;
    this.categoryFilter = categoryFilter;
    this.showSearch = options.showSearch !== false;
    this.showSummary = options.showSummary !== false;
    this.showFilterChips = options.showFilterChips !== false;
    this.onDetailOpen = options.onDetailOpen || null;
    this.onDetailClose = options.onDetailClose || null;
    this.categoryOverrides = options.categoryOverrides || {};
    this.categoryRemaps = options.categoryRemaps || {};
    this.pinnedCategories = options.pinnedCategories || [];

    // Instance state
    this.artifacts = [];
    this.untrackedFiles = [];
    this.selectedArtifact = null;
    this.currentView = 'gallery';
    this.detailMode = 'preview';
    this.searchQuery = '';
    this.filterCategory = null;
    this.filterProjectOwned = false;
    this.editDirty = false;
    this.loaded = false;
    this.summary = { total: 0, framework: 0, userOwned: 0 };

    // DOM references (populated by _buildDOM)
    this.loadingEl = null;
    this.errorEl = null;
    this.galleryView = null;
    this.detailView = null;
    this.searchInput = null;
    this.filterChipsEl = null;
    this.untrackedBannerEl = null;
    this.summaryEl = null;
    this.categoriesEl = null;
    this.detailHeaderEl = null;
    this.detailModesEl = null;
    this.detailContentEl = null;

    // Data actions, bound to this gallery's domain and DOM/render effects.
    // See scaffold/data.js — mirrors the net.js factory/callback pattern.
    this._data = createScaffoldDataActions(this, {
      onLoadStart: () => {
        this.loadingEl.style.display = 'flex';
        this.errorEl.style.display = 'none';
      },
      onLoaded: ({ artifacts, untrackedFiles, summary }) => {
        this.artifacts = artifacts;
        this.untrackedFiles = untrackedFiles;
        this.summary = summary;
        this.loadingEl.style.display = 'none';
        this.renderGallery();
        this.loaded = true;
      },
      onLoadError: (message) => {
        this.loadingEl.style.display = 'none';
        this.errorEl.style.display = 'flex';
        this.errorEl.textContent = message;
      },
    });

    // Gallery-view rendering + the artifact-list filter, bound to this
    // gallery's DOM refs and mutable filter/view state. See scaffold/view.js
    // — mirrors the same factory/injection pattern as _data above.
    this._view = createScaffoldGalleryView(this);

    // Detail-view shell (openDetail, showCreateDialog, header/mode-tabs,
    // mode dispatch), bound to this gallery's state. See scaffold/detail.js
    // — mirrors the same factory/injection pattern as _data/_view.
    this._detail = createScaffoldGalleryDetail(this);

    // Edit-view forms (settings.json structured editor, front-matter form,
    // plain-text fallback), bound to this gallery's state. See
    // scaffold/edit-form.js.
    this._editForm = createScaffoldGalleryEditForm(this);

    // Edit-view write actions (ownership take/release, discard/save,
    // reset-to-framework, reload+reopen, close-detail), bound to this
    // gallery's state. See scaffold/edit.js — mirrors the same
    // factory/injection pattern as _data/_view/_detail.
    this._edit = createScaffoldGalleryEdit(this);

    this._buildDOM();
  }

  // ---- DOM Construction ---- //

  _buildDOM() {
    this.container.innerHTML = '';

    // Loading state
    this.loadingEl = _el('div', 'prompts-loading');
    this.loadingEl.textContent = 'Loading artifacts...';
    this.loadingEl.style.display = 'none';
    this.container.appendChild(this.loadingEl);

    // Error state
    this.errorEl = _el('div', 'prompts-error');
    this.errorEl.style.display = 'none';
    this.container.appendChild(this.errorEl);

    // Gallery view
    this.galleryView = _el('div', 'scaffold-gallery-view');

    if (this.showSearch) {
      const searchBar = _el('div', 'prompts-search-bar');

      this.searchInput = document.createElement('input');
      this.searchInput.type = 'text';
      this.searchInput.className = 'prompts-search';
      this.searchInput.placeholder = 'Search artifacts...';
      this.searchInput.spellcheck = false;
      searchBar.appendChild(this.searchInput);

      if (this.showFilterChips) {
        this.filterChipsEl = _el('div', 'prompts-filter-chips');
        searchBar.appendChild(this.filterChipsEl);
      }

      this.galleryView.appendChild(searchBar);
    }

    this.untrackedBannerEl = _el('div', 'prompts-untracked-banner');
    this.untrackedBannerEl.style.display = 'none';
    this.galleryView.appendChild(this.untrackedBannerEl);

    if (this.showSummary) {
      this.summaryEl = _el('div', 'prompts-summary');
      this.galleryView.appendChild(this.summaryEl);
    }

    this.categoriesEl = _el('div', 'prompts-categories');
    this.galleryView.appendChild(this.categoriesEl);

    this.container.appendChild(this.galleryView);

    // Detail view
    this.detailView = _el('div', 'prompts-detail-view');
    this.detailView.style.display = 'none';

    this.detailHeaderEl = _el('div', 'prompts-detail-header');
    this.detailModesEl = _el('div', 'prompts-detail-modes');
    this.detailContentEl = _el('div', 'prompts-detail-content');

    this.detailView.appendChild(this.detailHeaderEl);
    this.detailView.appendChild(this.detailModesEl);
    this.detailView.appendChild(this.detailContentEl);

    this.container.appendChild(this.detailView);
  }

  // ---- Data Loading ---- //

  async load() {
    return this._data.load();
  }

  // ---- Gallery View ---- //
  //
  // Rendering (search bar, filter chips, untracked-file banner, summary,
  // category/card grid) and the artifact-list filter live in
  // scaffold/view.js — see createScaffoldGalleryView(). Only
  // renderGallery() is ever called back through the gallery host (from
  // scaffold/edit.js, after a save/reload/ownership change); view.js's
  // other rendering entry points (renderUntrackedBanner, renderFilterChips,
  // renderSummary, bindSearch, renderCategories, renderArtifactCard,
  // renderSkillGroup, getFilteredArtifacts) are only ever called from
  // within view.js's own renderGallery(), so this class doesn't re-expose
  // them as delegators.

  renderGallery() {
    return this._view.renderGallery();
  }

  async registerUntracked(canonicalName) {
    return this._data.registerUntracked(canonicalName);
  }

  async deleteUntracked(canonicalName) {
    return this._data.deleteUntracked(canonicalName);
  }

  // ---- Detail View ---- //
  //
  // openDetail, showCreateDialog, the header/mode-tabs rendering, and mode
  // dispatch (renderDetailContent) live in scaffold/detail.js \u2014
  // see createScaffoldGalleryDetail(). The two read-only content renderers
  // renderDetailContent dispatches to (Preview, Diff) live in
  // scaffold/detail-content.js. These are thin delegators for the call
  // sites in the scaffold/*.js modules, which call back through the
  // gallery host param (detail.js, edit.js, edit-form.js, cards.js,
  // view.js, detail-content.js). renderEdit and the edit/save/ownership
  // workflow live in scaffold/edit-form.js and scaffold/edit.js (below).

  openDetail(artifact) {
    return this._detail.openDetail(artifact);
  }

  showCreateDialog(category) {
    return this._detail.showCreateDialog(category);
  }

  renderDetailHeader() {
    return this._detail.renderDetailHeader();
  }

  renderDetailModes() {
    return this._detail.renderDetailModes();
  }

  renderDetailContent() {
    return this._detail.renderDetailContent();
  }

  // ---- Edit View ---- //
  //
  // The edit-mode form renderers (settings.json structured editor hookup,
  // front-matter form, plain-text fallback) live in scaffold/edit-form.js;
  // the write-side actions (ownership take/release, discard/save,
  // reset-to-framework, reload+reopen, close-detail) live in
  // scaffold/edit.js. These are thin delegators for the call sites that
  // reach them through the gallery host -- scaffold/detail.js's rendered
  // header/mode buttons call gallery.takeOwnership() etc., and its own
  // renderDetailContent() mode dispatch calls gallery.renderEdit().

  renderEdit() {
    return this._editForm.renderEdit();
  }

  takeOwnership() {
    return this._edit.takeOwnership();
  }

  releaseToFramework() {
    return this._edit.releaseToFramework();
  }

  handleEditFramework() {
    return this._edit.handleEditFramework();
  }

  discardEdits() {
    return this._edit.discardEdits();
  }

  saveOverride() {
    return this._edit.saveOverride();
  }

  closeDetail() {
    return this._edit.closeDetail();
  }

  // ---- State Reset ---- //

  reset() {
    this.artifacts = [];
    this.untrackedFiles = [];
    this.selectedArtifact = null;
    this.currentView = 'gallery';
    this.detailMode = 'preview';
    this.searchQuery = '';
    this.filterCategory = null;
    this.filterProjectOwned = false;
    this.editDirty = false;
    this.loaded = false;
    this.summary = { total: 0, framework: 0, userOwned: 0 };
  }
}

// ---- Public Exports ---- //

/**
 * Initialize the Prompt Gallery. Call once on DOMContentLoaded.
 * Creates three ArtifactGallery instances for the Behavior, Safety, and Config tabs.
 */
export function initScaffoldGallery() {
  const drawer = document.getElementById('settings-drawer');
  if (!drawer) return;

  configureMarked();

  const behaviorPanel = document.getElementById('tab-behavior');
  const safetyPanel = document.getElementById('tab-safety');
  const configGallerySection = document.getElementById('config-gallery-section');
  const configFormSection = document.getElementById('config-form-section');

  if (!behaviorPanel || !safetyPanel || !configGallerySection) return;

  const behaviorGallery = new ArtifactGallery({
    container: behaviorPanel,
    categoryFilter: (a) => BEHAVIOR_CATEGORIES.has(a.category) || BEHAVIOR_NAMES.has(a.name),
    options: {
      categoryOverrides: { 'claude-md': 'system prompt' },
      categoryRemaps: { rules: 'instructions' },
      pinnedCategories: ['system prompt', 'instructions'],
    },
  });

  const safetyGalleryContainer = document.getElementById('safety-gallery-section') || safetyPanel;
  const safetyGallery = new ArtifactGallery({
    container: safetyGalleryContainer,
    categoryFilter: (a) => SAFETY_CATEGORIES.has(a.category),
  });

  const configGallery = new ArtifactGallery({
    container: configGallerySection,
    categoryFilter: (a) => CONFIG_NAMES.has(a.name),
    options: {
      showSearch: false,
      showSummary: false,
      showFilterChips: false,
      onDetailOpen: () => {
        if (configFormSection) configFormSection.style.display = 'none';
        configGallerySection.style.flex = '1';
      },
      onDetailClose: () => {
        if (configFormSection) configFormSection.style.display = '';
        configGallerySection.style.flex = '';
      },
    },
  });

  // Load galleries when their tab becomes active
  behaviorPanel.addEventListener('drawer:tab-activate', () => {
    if (!behaviorGallery.loaded) behaviorGallery.load();
  });

  safetyPanel.addEventListener('drawer:tab-activate', () => {
    if (!safetyGallery.loaded) safetyGallery.load();
  });

  // Config tab activates both the config gallery and settings panel
  const configPanel = document.getElementById('tab-config');
  if (configPanel) {
    configPanel.addEventListener('drawer:tab-activate', () => {
      if (!configGallery.loaded) configGallery.load();
    });
  }

  // Reset all galleries and fetch cache when drawer closes
  drawer.addEventListener('drawer:close', () => {
    behaviorGallery.reset();
    safetyGallery.reset();
    configGallery.reset();
    resetFetchCache();
  });

  // Composite unsaved-changes guard
  drawer.registerUnsavedGuard(() => {
    const dirty = behaviorGallery.editDirty || safetyGallery.editDirty || configGallery.editDirty;
    if (!dirty) return true;
    return confirm('You have unsaved changes. Discard them?');
  });
}
