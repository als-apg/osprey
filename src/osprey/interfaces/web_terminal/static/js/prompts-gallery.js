/* OSPREY Web Terminal — Prompt Gallery
 *
 * Drives the "Prompt Gallery" UI inside the settings drawer tab panels.
 * Provides a reusable ArtifactGallery class that can be instantiated
 * multiple times for different tab panels (Behavior, Safety, Config).
 *
 *   - Gallery view: filterable/searchable card grid grouped by category
 *   - Detail view: preview (rendered markdown / highlighted code), diff, and edit modes
 *   - Scaffold/override workflow for customizing framework prompts
 *
 * API endpoints consumed:
 *   GET    /api/prompts                          -> list all artifacts
 *   GET    /api/prompts/{name}                   -> artifact content (active layer)
 *   GET    /api/prompts/{name}/framework         -> framework-layer content
 *   GET    /api/prompts/{name}/diff              -> unified diff between layers
 *   POST   /api/prompts/{name}/scaffold          -> create override scaffold
 *   PUT    /api/prompts/{name}/override           -> save override content
 *   DELETE /api/prompts/{name}/override?delete_file=true -> remove override
 */

import { fetchJSON } from './api.js';
import { registerUnsavedGuard } from './drawer.js';
import { tokenize, computeWordDiff, groupChangeBlocks, renderWordsIntoLine } from './diff-utils.js';
import { renderSettingsJsonEditor, renderMcpJson } from './config-renderers.js';

// ---- Constants ---- //

// TODO: Pull from provider registry when Claude is routed through CBORG/other providers
const AGENT_MODEL_OPTIONS = ['haiku', 'sonnet', 'opus'];

/** Brief descriptions for each artifact category, shown in the help tooltip. */
const CATEGORY_HELP = {
  'system prompt': 'The main CLAUDE.md file that defines the AI assistant\'s identity, capabilities, and behavioral guidelines.',
  agents: 'Sub-agents that Claude delegates specialized tasks to (search, analysis, visualization). Each agent has its own model, tools, and instructions.',
  config: 'Top-level configuration files: MCP server definitions (.mcp.json) and permissions (settings.json).',
  hooks: 'Python scripts that run before or after Claude uses a tool. They enforce safety rules, validate inputs, and inject error guidance.',
  instructions: 'Markdown files loaded as persistent directives. They define safety boundaries, error handling protocols, and artifact conventions.',
  skills: 'Multi-file bundles that Claude can invoke as structured workflows. Skills support companion files (CSS/JS references, templates).',
  'output-styles': 'Markdown style guides that shape how Claude writes responses — tone, format, and epistemic discipline for control system communication.',
};

// ---- Category Routing ---- //

const BEHAVIOR_CATEGORIES = new Set(['agents', 'skills', 'rules', 'output-styles']);
const BEHAVIOR_NAMES = new Set(['claude-md']);        // config category, behavior tab
const SAFETY_CATEGORIES = new Set(['hooks']);
const CONFIG_NAMES = new Set(['mcp-json', 'settings-json']); // config category, config tab

// ---- Shared Fetch Cache ---- //

let _fetchPromise = null;

/**
 * Fetch artifacts from the API, caching the promise so multiple
 * gallery instances don't duplicate the request.
 */
async function fetchArtifactsShared() {
  if (!_fetchPromise) _fetchPromise = fetchJSON('/api/prompts');
  return _fetchPromise;
}

/** Reset the shared cache (called when drawer closes). */
function resetFetchCache() {
  _fetchPromise = null;
}

// ---- Marked.js Configuration (one-time) ---- //

let _markedConfigured = false;

function configureMarked() {
  if (_markedConfigured) return;
  _markedConfigured = true;

  if (typeof marked === 'undefined') return;

  const renderer = {
    code({ text, lang }) {
      const src = text ?? '';
      let highlighted = escapeHtml(src);
      if (typeof hljs !== 'undefined' && src) {
        try {
          if (lang && hljs.getLanguage(lang)) {
            highlighted = hljs.highlight(src, { language: lang }).value;
          } else {
            highlighted = hljs.highlightAuto(src).value;
          }
        } catch {
          // Fall back to escaped text on any hljs error
        }
      }
      const langClass = lang ? ` class="language-${lang}"` : '';
      return `<pre><code${langClass}>${highlighted}</code></pre>`;
    },
  };

  function walkTokens(token) {
    if (token.type === 'code' && typeof token.text !== 'string') {
      token.text = token.text != null ? String(token.text) : '';
    }
  }

  marked.use({ gfm: true, breaks: false, renderer, walkTokens });
}

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
    this.galleryView = _el('div', 'prompts-gallery-view');

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
    this.loadingEl.style.display = 'flex';
    this.errorEl.style.display = 'none';

    try {
      const [data, untrackedData] = await Promise.all([
        fetchArtifactsShared(),
        fetchJSON('/api/prompts/untracked').catch(() => ({ untracked: [] })),
      ]);
      const allArtifacts = data.artifacts || [];

      // Filter to this gallery's domain and apply category overrides
      this.artifacts = allArtifacts
        .filter(this.categoryFilter)
        .map((a) => ({
          ...a,
          displayCategory:
            this.categoryOverrides[a.name] ||
            this.categoryRemaps[a.category] ||
            a.category,
        }));

      // Filter untracked files to this gallery's categories
      const allUntracked = untrackedData.untracked || [];
      this.untrackedFiles = allUntracked.filter((u) => {
        const mapped = this.categoryRemaps[u.category] || u.category;
        return this.categoryFilter({ category: u.category, name: u.canonical_name })
          || this.categoryFilter({ category: mapped, name: u.canonical_name });
      });

      // Compute summary for this gallery's artifacts
      const fw = this.artifacts.filter((a) => a.status === 'framework').length;
      const uo = this.artifacts.filter((a) => a.status === 'user-owned').length;
      this.summary = { total: this.artifacts.length, framework: fw, userOwned: uo };

      this.loadingEl.style.display = 'none';
      this.renderGallery();
      this.loaded = true;
    } catch (e) {
      this.loadingEl.style.display = 'none';
      this.errorEl.style.display = 'flex';
      this.errorEl.textContent = `Failed to load prompts: ${e.message}`;
    }
  }

  // ---- Gallery View ---- //

  renderGallery() {
    if (this.galleryView) this.galleryView.style.display = '';
    if (this.detailView) this.detailView.style.display = 'none';

    this.currentView = 'gallery';

    this.renderUntrackedBanner();
    this.renderFilterChips();
    this.renderSummary();
    this.bindSearch();
    this.renderCategories();
  }

  renderUntrackedBanner() {
    if (!this.untrackedBannerEl) return;

    if (!this.untrackedFiles || this.untrackedFiles.length === 0) {
      this.untrackedBannerEl.style.display = 'none';
      return;
    }

    this.untrackedBannerEl.style.display = '';
    this.untrackedBannerEl.innerHTML = '';

    const header = _el('div', 'prompts-untracked-header');
    const icon = _el('span', 'prompts-untracked-icon');
    icon.textContent = '\u26A0';
    header.appendChild(icon);

    const title = _el('span', 'prompts-untracked-title');
    const n = this.untrackedFiles.length;
    title.textContent = `${n} file${n > 1 ? 's' : ''} active in Claude Code but not managed by OSPREY`;
    header.appendChild(title);

    this.untrackedBannerEl.appendChild(header);

    const desc = _el('div', 'prompts-untracked-desc');
    desc.textContent =
      'These files are in .claude/ and will be loaded by Claude Code, but they are not tracked in your project config. Register them to manage through this UI, or delete them.';
    this.untrackedBannerEl.appendChild(desc);

    const list = _el('div', 'prompts-untracked-list');

    for (const file of this.untrackedFiles) {
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
      registerBtn.addEventListener('click', () => this.registerUntracked(file.canonical_name));
      actions.appendChild(registerBtn);

      const deleteBtn = document.createElement('button');
      deleteBtn.className = 'prompts-untracked-btn prompts-untracked-delete';
      deleteBtn.textContent = 'Delete';
      deleteBtn.title = 'Remove this file from disk — it will no longer affect Claude Code';
      deleteBtn.addEventListener('click', () => this.deleteUntracked(file.canonical_name));
      actions.appendChild(deleteBtn);

      row.appendChild(actions);
      list.appendChild(row);
    }

    this.untrackedBannerEl.appendChild(list);
  }

  async registerUntracked(canonicalName) {
    try {
      const resp = await fetch('/api/prompts/untracked/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: canonicalName }),
      });
      if (!resp.ok) {
        const detail = await resp.json().catch(() => ({}));
        throw new Error(detail.detail || `Register failed (HTTP ${resp.status})`);
      }
      await this.reloadFull();
    } catch (e) {
      this.errorEl.style.display = 'flex';
      this.errorEl.textContent = `Register failed: ${e.message}`;
    }
  }

  async deleteUntracked(canonicalName) {
    if (!confirm(`Delete "${canonicalName}"? This file will be removed from disk.`)) return;

    try {
      const resp = await fetch(
        `/api/prompts/untracked/${encodeURIComponent(canonicalName)}`,
        { method: 'DELETE' }
      );
      if (!resp.ok) {
        const detail = await resp.json().catch(() => ({}));
        throw new Error(detail.detail || `Delete failed (HTTP ${resp.status})`);
      }
      await this.reloadFull();
    } catch (e) {
      this.errorEl.style.display = 'flex';
      this.errorEl.textContent = `Delete failed: ${e.message}`;
    }
  }

  async reloadFull() {
    resetFetchCache();
    const [data, untrackedData] = await Promise.all([
      fetchArtifactsShared(),
      fetchJSON('/api/prompts/untracked').catch(() => ({ untracked: [] })),
    ]);

    const allArtifacts = data.artifacts || [];
    this.artifacts = allArtifacts
      .filter(this.categoryFilter)
      .map((a) => ({
        ...a,
        displayCategory:
          this.categoryOverrides[a.name] ||
          this.categoryRemaps[a.category] ||
          a.category,
      }));

    const allUntracked = untrackedData.untracked || [];
    this.untrackedFiles = allUntracked.filter((u) => {
      const mapped = this.categoryRemaps[u.category] || u.category;
      return this.categoryFilter({ category: u.category, name: u.canonical_name })
        || this.categoryFilter({ category: mapped, name: u.canonical_name });
    });

    const fw = this.artifacts.filter((a) => a.status === 'framework').length;
    const uo = this.artifacts.filter((a) => a.status === 'user-owned').length;
    this.summary = { total: this.artifacts.length, framework: fw, userOwned: uo };

    this.renderGallery();
  }

  renderFilterChips() {
    if (!this.filterChipsEl) return;
    this.filterChipsEl.innerHTML = '';

    const categories = [...new Set(this.artifacts.map((a) => a.displayCategory))].sort();

    // "All" chip
    const allChip = document.createElement('button');
    allChip.className = 'prompts-chip' + (this.filterCategory === null ? ' active' : '');
    allChip.textContent = 'All';
    allChip.addEventListener('click', () => {
      this.filterCategory = null;
      this.renderFilterChips();
      this.renderCategories();
    });
    this.filterChipsEl.appendChild(allChip);

    // Per-category chips
    for (const cat of categories) {
      const chip = document.createElement('button');
      chip.className = 'prompts-chip' + (this.filterCategory === cat ? ' active' : '');
      chip.textContent = cat;
      chip.addEventListener('click', () => {
        this.filterCategory = cat;
        this.renderFilterChips();
        this.renderCategories();
      });
      this.filterChipsEl.appendChild(chip);
    }

    // "Project-owned" toggle
    const hasUserOwned = this.artifacts.some((a) => a.status === 'user-owned');
    if (hasUserOwned) {
      const sep = document.createElement('span');
      sep.className = 'prompts-chip-sep';
      this.filterChipsEl.appendChild(sep);

      const ownedChip = document.createElement('button');
      ownedChip.className = 'prompts-chip prompts-chip-toggle' + (this.filterProjectOwned ? ' active' : '');
      ownedChip.textContent = 'Project-owned';
      ownedChip.addEventListener('click', () => {
        this.filterProjectOwned = !this.filterProjectOwned;
        this.renderFilterChips();
        this.renderCategories();
      });
      this.filterChipsEl.appendChild(ownedChip);
    }
  }

  renderSummary() {
    if (!this.summaryEl) return;
    this.summaryEl.textContent =
      `${this.summary.total} artifacts \u00B7 ${this.summary.framework} framework \u00B7 ${this.summary.userOwned} project-owned`;
  }

  bindSearch() {
    if (!this.searchInput) return;

    // Remove previous listener by replacing the element
    const clone = this.searchInput.cloneNode(true);
    this.searchInput.parentNode.replaceChild(clone, this.searchInput);
    this.searchInput = clone;

    clone.value = this.searchQuery;

    const debouncedRender = debounce(() => {
      this.searchQuery = clone.value.trim();
      this.renderCategories();
    }, 150);

    clone.addEventListener('input', debouncedRender);
  }

  renderCategories() {
    if (!this.categoriesEl) return;
    this.categoriesEl.innerHTML = '';

    const filtered = this.getFilteredArtifacts();

    // Group by display category
    const groups = {};
    for (const artifact of filtered) {
      const cat = artifact.displayCategory || 'other';
      if (!groups[cat]) groups[cat] = [];
      groups[cat].push(artifact);
    }

    // Sort categories with pinned ones first, then alphabetical
    const pinned = this.pinnedCategories;
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

      // Category header
      const header = document.createElement('div');
      header.className = 'prompts-category-header';

      const label = document.createElement('span');
      label.textContent = cat.toUpperCase();
      header.appendChild(label);

      const count = document.createElement('span');
      count.className = 'prompts-category-count';
      count.textContent = groups[cat].length;
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

      section.appendChild(header);

      // Skills get special grouping
      if (cat.toLowerCase() === 'skills') {
        const skillGroups = {};
        for (const art of groups[cat]) {
          const parts = art.name.split('/');
          const skillName = parts[1] || parts[0];
          if (!skillGroups[skillName]) skillGroups[skillName] = [];
          skillGroups[skillName].push(art);
        }
        for (const [skillName, groupArts] of Object.entries(skillGroups).sort()) {
          this.renderSkillGroup(section, skillName, groupArts);
        }
      } else {
        for (const artifact of groups[cat]) {
          this.renderArtifactCard(section, artifact, cat);
        }
      }

      this.categoriesEl.appendChild(section);
    }

    if (filtered.length === 0) {
      this.categoriesEl.innerHTML = '<div class="prompts-empty">No matching artifacts found.</div>';
    }
  }

  renderArtifactCard(section, artifact, cat) {
    const card = document.createElement('div');
    card.className = 'prompts-card';
    card.dataset.name = artifact.name;

    const icon = document.createElement('div');
    icon.className = 'prompts-card-icon';
    icon.textContent = iconForCategory(cat);

    const body = document.createElement('div');
    body.className = 'prompts-card-body';

    const nameEl = document.createElement('div');
    nameEl.className = 'prompts-card-name';
    const displayName = artifact.name.includes('/')
      ? artifact.name.split('/').slice(1).join('/')
      : artifact.name;
    nameEl.textContent = displayName;

    const descEl = document.createElement('div');
    descEl.className = 'prompts-card-desc';
    descEl.textContent = artifact.summary || artifact.description || '';

    body.appendChild(nameEl);
    body.appendChild(descEl);

    const badge = document.createElement('span');
    const owned = artifact.status === 'user-owned';
    badge.className = `prompts-badge ${owned ? 'user-owned' : 'framework'}`;
    badge.textContent = owned ? 'PROJECT-OWNED' : 'FRAMEWORK';

    card.appendChild(icon);
    card.appendChild(body);
    card.appendChild(badge);

    card.addEventListener('click', () => this.openDetail(artifact));
    section.appendChild(card);
  }

  renderSkillGroup(section, skillName, groupArtifacts) {
    if (groupArtifacts.length === 1) {
      this.renderArtifactCard(section, groupArtifacts[0], 'skills');
      return;
    }

    const sorted = [...groupArtifacts].sort((a, b) => {
      const aDepth = a.name.split('/').length;
      const bDepth = b.name.split('/').length;
      return aDepth - bDepth || a.name.localeCompare(b.name);
    });

    const card = document.createElement('div');
    card.className = 'prompts-card prompts-skill-group';
    card.dataset.name = sorted[0].name;

    let selectedArt = sorted[0];

    const icon = document.createElement('div');
    icon.className = 'prompts-card-icon';
    icon.textContent = iconForCategory('skills');

    const body = document.createElement('div');
    body.className = 'prompts-card-body';

    const nameEl = document.createElement('div');
    nameEl.className = 'prompts-card-name';
    nameEl.textContent = skillName;

    const descEl = document.createElement('div');
    descEl.className = 'prompts-card-desc';
    descEl.textContent = selectedArt.summary || selectedArt.description || '';

    body.appendChild(nameEl);
    body.appendChild(descEl);

    const select = document.createElement('select');
    select.className = 'prompts-skill-select';
    for (const art of sorted) {
      const opt = document.createElement('option');
      opt.value = art.name;
      opt.textContent = (art.output_path || art.name).split('/').pop();
      select.appendChild(opt);
    }
    select.addEventListener('click', (e) => e.stopPropagation());
    select.addEventListener('change', (e) => {
      selectedArt = sorted.find((a) => a.name === e.target.value) || sorted[0];
      descEl.textContent = selectedArt.summary || selectedArt.description || '';
    });
    body.appendChild(select);

    const badge = document.createElement('span');
    const ownedSkill = selectedArt.status === 'user-owned';
    badge.className = `prompts-badge ${ownedSkill ? 'user-owned' : 'framework'}`;
    badge.textContent = ownedSkill ? 'PROJECT-OWNED' : 'FRAMEWORK';

    card.appendChild(icon);
    card.appendChild(body);
    card.appendChild(badge);

    card.addEventListener('click', () => this.openDetail(selectedArt));
    section.appendChild(card);
  }

  // ---- Filtering ---- //

  getFilteredArtifacts() {
    let result = this.artifacts;

    if (this.filterProjectOwned) {
      result = result.filter((a) => a.status === 'user-owned');
    }

    if (this.filterCategory) {
      result = result.filter((a) => a.displayCategory === this.filterCategory);
    }

    if (this.searchQuery) {
      const q = this.searchQuery.toLowerCase();
      result = result.filter((a) => {
        const name = (a.name || '').toLowerCase();
        const desc = (a.description || '').toLowerCase();
        const sum = (a.summary || '').toLowerCase();
        return name.includes(q) || desc.includes(q) || sum.includes(q);
      });
    }

    return result;
  }

  // ---- Detail View ---- //

  openDetail(artifact) {
    this.selectedArtifact = artifact;
    this.currentView = 'detail';
    this.detailMode = 'preview';
    this.editDirty = false;

    if (this.galleryView) this.galleryView.style.display = 'none';
    if (this.detailView) this.detailView.style.display = '';

    if (this.onDetailOpen) this.onDetailOpen();

    this.renderDetailHeader();
    this.renderDetailModes();
    this.renderDetailContent();
  }

  renderDetailHeader() {
    if (!this.detailHeaderEl || !this.selectedArtifact) return;
    this.detailHeaderEl.innerHTML = '';

    // Row 1: [Back] name ... BADGE [Ownership Btn]
    const row1 = document.createElement('div');
    row1.className = 'prompts-header-row';

    const backBtn = document.createElement('button');
    backBtn.className = 'prompts-back-btn';
    backBtn.textContent = '\u2190 Back';
    backBtn.addEventListener('click', () => this.closeDetail());
    row1.appendChild(backBtn);

    const nameEl = document.createElement('span');
    nameEl.className = 'prompts-detail-name';
    nameEl.textContent = this.selectedArtifact.name;
    row1.appendChild(nameEl);

    const spacer = document.createElement('span');
    spacer.style.flex = '1';
    row1.appendChild(spacer);

    const isOwned = this.selectedArtifact.status === 'user-owned';

    const badge = document.createElement('span');
    badge.className = `prompts-badge ${isOwned ? 'user-owned' : 'framework'}`;
    badge.textContent = isOwned ? 'PROJECT-OWNED' : 'FRAMEWORK';
    row1.appendChild(badge);

    const ownerBtn = document.createElement('button');
    ownerBtn.className = 'prompts-ownership-btn';
    if (isOwned) {
      ownerBtn.textContent = 'Release to Framework';
      ownerBtn.addEventListener('click', () => this.releaseToFramework());
    } else {
      ownerBtn.textContent = 'Take Ownership';
      ownerBtn.addEventListener('click', () => this.takeOwnership());
    }
    row1.appendChild(ownerBtn);

    this.detailHeaderEl.appendChild(row1);

    // Row 2: path + language
    const row2 = document.createElement('div');
    row2.className = 'prompts-header-meta';

    if (this.selectedArtifact.output_path) {
      const pathEl = document.createElement('span');
      pathEl.className = 'prompts-detail-path';
      pathEl.textContent = this.selectedArtifact.output_path;
      row2.appendChild(pathEl);
    }

    if (this.selectedArtifact.language) {
      const langEl = document.createElement('span');
      langEl.className = 'prompts-detail-lang';
      langEl.textContent = this.selectedArtifact.language;
      row2.appendChild(langEl);
    }

    this.detailHeaderEl.appendChild(row2);
  }

  renderDetailModes() {
    if (!this.detailModesEl || !this.selectedArtifact) return;
    this.detailModesEl.innerHTML = '';

    // Left: mode buttons
    const left = document.createElement('div');
    left.className = 'prompts-modes-left';

    const modes = [{ key: 'preview', label: 'Preview' }];

    if (this.selectedArtifact.status === 'user-owned' && !this.selectedArtifact.custom) {
      modes.push({ key: 'diff', label: 'Diff' });
    }

    modes.push({ key: 'edit', label: 'Edit' });

    for (const mode of modes) {
      const btn = document.createElement('button');
      btn.className = 'prompts-mode-btn' + (this.detailMode === mode.key ? ' active' : '');
      btn.textContent = mode.label;
      btn.addEventListener('click', () => {
        if (this.detailMode === mode.key) return;

        if (mode.key === 'edit' && this.selectedArtifact.status === 'framework') {
          this.handleEditFramework();
          return;
        }

        if (this.editDirty) {
          if (!confirm('You have unsaved changes. Discard them?')) return;
          this.editDirty = false;
        }
        this.detailMode = mode.key;
        this.renderDetailModes();
        this.renderDetailContent();
      });
      left.appendChild(btn);
    }

    this.detailModesEl.appendChild(left);

    // Right: action buttons
    const right = document.createElement('div');
    right.className = 'prompts-modes-right';

    const isSettingsPreview = this.detailMode === 'preview'
      && this.selectedArtifact?.name === 'settings-json';

    if (this.detailMode === 'edit' || (isSettingsPreview && this.editDirty)) {
      const discardBtn = document.createElement('button');
      discardBtn.className = 'prompts-discard-btn';
      discardBtn.textContent = 'Discard';
      discardBtn.disabled = !this.editDirty;
      discardBtn.addEventListener('click', () => this.discardEdits());
      right.appendChild(discardBtn);

      const saveBtn = document.createElement('button');
      saveBtn.className = 'prompts-save-btn';
      saveBtn.textContent = 'Save';
      saveBtn.disabled = !this.editDirty;
      saveBtn.addEventListener('click', () => this.saveOverride());
      right.appendChild(saveBtn);
    }

    this.detailModesEl.appendChild(right);
  }

  async renderDetailContent() {
    if (!this.detailContentEl || !this.selectedArtifact) return;

    this.detailContentEl.innerHTML = '<div class="prompts-loading-inline">Loading...</div>';

    try {
      if (this.detailMode === 'preview') {
        await this.renderPreview();
      } else if (this.detailMode === 'diff') {
        await this.renderDiff();
      } else if (this.detailMode === 'edit') {
        await this.renderEdit();
      }
    } catch (e) {
      this.detailContentEl.innerHTML =
        `<div class="prompts-content-error">Error loading content: ${escapeHtml(e.message)}</div>`;
    }
  }

  async renderPreview() {
    const data = await fetchJSON(`/api/prompts/${encodeURIComponent(this.selectedArtifact.name)}`);
    const content = data.content || '';
    const language = data.language || this.selectedArtifact.language || 'text';
    const artifactName = this.selectedArtifact.name || '';

    this.detailContentEl.innerHTML = '';

    const wrapper = document.createElement('div');
    wrapper.className = 'prompts-preview-content';

    // settings-json: use interactive editor directly in preview mode
    if (artifactName === 'settings-json' && language === 'json') {
      const structured = renderSettingsJsonEditor(content, (isDirty) => {
        this.editDirty = isDirty;
        this.renderDetailModes();
      });
      if (structured) {
        // Attach _settingsEditor API on detailContentEl for saveOverride()
        this.detailContentEl._settingsEditor = structured._settingsEditor;
        wrapper.appendChild(structured);
        wrapper.appendChild(renderSourceToggle(content, 'json'));
        this.detailContentEl.appendChild(wrapper);
        return;
      }
    }

    if (artifactName === 'mcp-json' && language === 'json') {
      const structured = renderMcpJson(content);
      if (structured) {
        wrapper.appendChild(structured);
        wrapper.appendChild(renderSourceToggle(content, 'json'));
        this.detailContentEl.appendChild(wrapper);
        return;
      }
    }

    if (language === 'markdown') {
      const { frontMatter, body } = parseFrontMatter(content);

      if (frontMatter) {
        wrapper.appendChild(renderFrontMatterTable(frontMatter));
      }

      const mdDiv = document.createElement('div');
      mdDiv.className = 'osprey-md-rendered';
      if (typeof marked !== 'undefined') {
        try {
          mdDiv.innerHTML = marked.parse(body);
        } catch {
          mdDiv.textContent = body;
        }
      } else {
        mdDiv.textContent = body;
      }
      wrapper.appendChild(mdDiv);
    } else if (language === 'python') {
      const parsed = extractPythonDocstringFrontMatter(content);

      if (parsed.frontMatter) {
        wrapper.appendChild(renderFrontMatterTable(parsed.frontMatter));

        if (parsed.flowDiagram) {
          wrapper.appendChild(renderFlowDiagram(parsed.flowDiagram));
        }

        if (parsed.body) {
          const mdDiv = document.createElement('div');
          mdDiv.className = 'osprey-md-rendered';
          if (typeof marked !== 'undefined') {
            try {
              mdDiv.innerHTML = marked.parse(parsed.body);
            } catch {
              mdDiv.textContent = parsed.body;
            }
          } else {
            mdDiv.textContent = parsed.body;
          }
          wrapper.appendChild(mdDiv);
        }

        wrapper.appendChild(renderSourceToggle(parsed.sourceCode, 'python'));
      } else {
        wrapper.appendChild(renderHighlightedCode(content, language));
      }
    } else {
      wrapper.appendChild(renderHighlightedCode(content, language));
    }

    this.detailContentEl.appendChild(wrapper);
  }

  async renderDiff() {
    const data = await fetchJSON(
      `/api/prompts/${encodeURIComponent(this.selectedArtifact.name)}/diff`
    );

    this.detailContentEl.innerHTML = '';

    if (!data.has_diff) {
      this.detailContentEl.innerHTML = '<div class="prompts-no-diff">No differences from framework default.</div>';
      return;
    }

    const stats = document.createElement('div');
    stats.className = 'prompts-diff-stats';
    stats.innerHTML =
      `<span class="prompts-diff-add">+${data.additions}</span> ` +
      `<span class="prompts-diff-del">\u2212${data.deletions}</span>`;
    this.detailContentEl.appendChild(stats);

    const diffBlock = document.createElement('div');
    diffBlock.className = 'prompts-diff-block';

    const rawLines = (data.unified_diff || '').split('\n');
    const blocks = groupChangeBlocks(rawLines);

    for (const block of blocks) {
      if (block.type === 'change') {
        const { delLines, addLines } = block;
        const paired = Math.min(delLines.length, addLines.length);

        // Paired lines: compute word diff per pair
        for (let k = 0; k < paired; k++) {
          const oldTokens = tokenize(delLines[k]);
          const newTokens = tokenize(addLines[k]);
          const ops = computeWordDiff(oldTokens, newTokens);

          const delEl = document.createElement('div');
          delEl.className = 'prompts-diff-line del';
          renderWordsIntoLine(delEl, delLines[k], ops, 'del');
          diffBlock.appendChild(delEl);

          const addEl = document.createElement('div');
          addEl.className = 'prompts-diff-line add';
          renderWordsIntoLine(addEl, addLines[k], ops, 'add');
          diffBlock.appendChild(addEl);
        }

        // Surplus unpaired lines: plain textContent
        for (let k = paired; k < delLines.length; k++) {
          const el = document.createElement('div');
          el.className = 'prompts-diff-line del';
          el.textContent = delLines[k];
          diffBlock.appendChild(el);
        }
        for (let k = paired; k < addLines.length; k++) {
          const el = document.createElement('div');
          el.className = 'prompts-diff-line add';
          el.textContent = addLines[k];
          diffBlock.appendChild(el);
        }
      } else {
        // context, hunk, unpaired del, unpaired add
        for (const line of block.lines) {
          const lineEl = document.createElement('div');
          lineEl.className = 'prompts-diff-line';

          if (block.type === 'hunk') lineEl.classList.add('hunk');
          else if (block.type === 'del') lineEl.classList.add('del');
          else if (block.type === 'add') lineEl.classList.add('add');
          else lineEl.classList.add('context');

          lineEl.textContent = line;
          diffBlock.appendChild(lineEl);
        }
      }
    }

    this.detailContentEl.appendChild(diffBlock);
  }

  async renderEdit() {
    const data = await fetchJSON(`/api/prompts/${encodeURIComponent(this.selectedArtifact.name)}`);
    const content = data.content || '';
    const artifactName = this.selectedArtifact.name || '';
    const language = data.language || this.selectedArtifact.language || 'text';

    // Clear content area and stale editor reference
    while (this.detailContentEl.firstChild) {
      this.detailContentEl.removeChild(this.detailContentEl.firstChild);
    }
    this.detailContentEl._settingsEditor = null;

    // Structured editor for settings.json
    if (artifactName === 'settings-json' && language === 'json') {
      const editor = renderSettingsJsonEditor(content, (isDirty) => {
        this.editDirty = isDirty;
        this.renderDetailModes();
      });
      if (editor) {
        this.detailContentEl.appendChild(editor);
        this.detailContentEl._settingsEditor = editor._settingsEditor;
        return;
      }
    }

    const { frontMatter, body } = parseFrontMatter(content);
    if (frontMatter && frontMatter.model) {
      this.renderFrontMatterForm(content, frontMatter, body);
    } else {
      this.renderPlainTextEditor(content);
    }
  }

  renderPlainTextEditor(content) {
    const textarea = document.createElement('textarea');
    textarea.className = 'prompts-edit-textarea';
    textarea.spellcheck = false;
    textarea.value = content;

    textarea.addEventListener('input', () => {
      this.editDirty = true;
      this.renderDetailModes();
    });

    this.detailContentEl.appendChild(textarea);
  }

  renderFrontMatterForm(fullContent, frontMatter, body) {
    const container = this.detailContentEl;

    const form = document.createElement('div');
    form.className = 'prompts-fm-form';

    const formTitle = document.createElement('div');
    formTitle.className = 'prompts-fm-form-title';
    formTitle.textContent = 'AGENT CONFIGURATION';
    form.appendChild(formTitle);

    const fieldDefs = [
      { key: 'name', label: 'name', type: 'text' },
      { key: 'description', label: 'description', type: 'text' },
      { key: 'model', label: 'model', type: 'select', options: AGENT_MODEL_OPTIONS },
      { key: 'maxTurns', label: 'maxTurns', type: 'number' },
      { key: 'disallowedTools', label: 'disallowedTools', type: 'text' },
    ];

    const fieldRefs = {};

    for (const def of fieldDefs) {
      const value = frontMatter[def.key] || '';
      const { wrapper, input } = this._createFormField(def.key, def.label, def.type, value, def.options);
      form.appendChild(wrapper);
      fieldRefs[def.key] = input;
    }

    container.appendChild(form);

    const instrTitle = document.createElement('div');
    instrTitle.className = 'prompts-fm-form-title';
    instrTitle.style.padding = '10px 12px 0';
    instrTitle.textContent = 'AGENT INSTRUCTIONS';
    container.appendChild(instrTitle);

    const bodyTextarea = document.createElement('textarea');
    bodyTextarea.className = 'prompts-edit-textarea';
    bodyTextarea.spellcheck = false;
    bodyTextarea.value = body;

    bodyTextarea.addEventListener('input', () => {
      this.editDirty = true;
      this.renderDetailModes();
    });

    container.appendChild(bodyTextarea);

    // Store references for saveOverride()
    container._frontMatterFields = fieldRefs;
    container._bodyTextarea = bodyTextarea;
  }

  _createFormField(key, label, type, value, options) {
    const wrapper = document.createElement('div');
    wrapper.className = 'prompts-fm-field';

    const labelEl = document.createElement('span');
    labelEl.className = 'prompts-fm-field-label';
    labelEl.textContent = label;
    wrapper.appendChild(labelEl);

    let input;

    if (type === 'select') {
      input = document.createElement('select');
      input.className = 'settings-select';
      for (const opt of (options || [])) {
        const optEl = document.createElement('option');
        optEl.value = opt;
        optEl.textContent = opt;
        if (opt === value) optEl.selected = true;
        input.appendChild(optEl);
      }
    } else if (type === 'number') {
      input = document.createElement('input');
      input.type = 'number';
      input.className = 'settings-input';
      input.min = '1';
      input.max = '100';
      input.value = value;
    } else {
      input = document.createElement('input');
      input.type = 'text';
      input.className = 'settings-input';
      input.value = value;
    }

    input.addEventListener('input', () => {
      this.editDirty = true;
      this.renderDetailModes();
    });
    input.addEventListener('change', () => {
      this.editDirty = true;
      this.renderDetailModes();
    });

    wrapper.appendChild(input);
    return { wrapper, input };
  }

  // ---- Ownership Actions ---- //

  async takeOwnership() {
    if (!this.selectedArtifact) return;
    if (!confirm('By doing this you take responsibility for this file.')) return;

    try {
      const resp = await fetch(
        `/api/prompts/${encodeURIComponent(this.selectedArtifact.name)}/scaffold`,
        { method: 'POST' }
      );

      if (!resp.ok) {
        const detail = await resp.json().catch(() => ({}));
        throw new Error(detail.detail || `Scaffold failed (HTTP ${resp.status})`);
      }

      await this.reloadAndReopen();
    } catch (e) {
      this.errorEl.style.display = 'flex';
      this.errorEl.textContent = `Scaffold failed: ${e.message}`;
    }
  }

  async releaseToFramework() {
    if (!this.selectedArtifact) return;
    if (!confirm('Your customizations will be removed.')) return;

    await this.unoverrideArtifact(true);
  }

  async handleEditFramework() {
    if (!this.selectedArtifact) return;
    if (!confirm('This will create a project copy for editing.')) return;

    try {
      const resp = await fetch(
        `/api/prompts/${encodeURIComponent(this.selectedArtifact.name)}/scaffold`,
        { method: 'POST' }
      );

      if (!resp.ok) {
        const detail = await resp.json().catch(() => ({}));
        throw new Error(detail.detail || `Scaffold failed (HTTP ${resp.status})`);
      }

      // Reload from API (invalidate cache so we get fresh data)
      resetFetchCache();
      const data = await fetchArtifactsShared();
      const allArtifacts = data.artifacts || [];
      this.artifacts = allArtifacts
        .filter(this.categoryFilter)
        .map((a) => ({
          ...a,
          displayCategory:
            this.categoryOverrides[a.name] ||
            this.categoryRemaps[a.category] ||
            a.category,
        }));

      const updated = this.artifacts.find((a) => a.name === this.selectedArtifact.name);
      if (updated) {
        this.selectedArtifact = updated;
        this.detailMode = 'edit';
        this.renderDetailHeader();
        this.renderDetailModes();
        this.renderDetailContent();
      }
    } catch (e) {
      this.errorEl.style.display = 'flex';
      this.errorEl.textContent = `Scaffold failed: ${e.message}`;
    }
  }

  discardEdits() {
    this.editDirty = false;
    if (this.detailMode !== 'preview') {
      this.detailMode = 'preview';
    }
    this.renderDetailModes();
    this.renderDetailContent();
  }

  // ---- Save Override ---- //

  async saveOverride() {
    if (!this.selectedArtifact) return;

    const container = this.detailContentEl;
    if (!container) return;

    let content;

    if (container._settingsEditor) {
      // Settings.json structured editor
      content = container._settingsEditor.getData();
    } else if (container._frontMatterFields && container._bodyTextarea) {
      const fields = container._frontMatterFields;
      let yaml = '---\n';
      for (const [key, input] of Object.entries(fields)) {
        const val = input.value.trim();
        if (val) {
          if (val.includes(':') || val.includes('#') || val.includes(',')) {
            yaml += `${key}: "${val}"\n`;
          } else {
            yaml += `${key}: ${val}\n`;
          }
        }
      }
      yaml += '---\n';
      content = yaml + container._bodyTextarea.value;
    } else {
      const textarea = container.querySelector('.prompts-edit-textarea');
      if (!textarea) return;
      content = textarea.value;
    }

    // Ownership warning + scaffold for framework-owned settings.json
    if (this.selectedArtifact.name === 'settings-json'
        && this.selectedArtifact.status === 'framework') {
      const confirmed = await this._showOwnershipWarning();
      if (!confirmed) return;

      // Scaffold (claim) the file before writing the override
      const scaffoldResp = await fetch(
        `/api/prompts/${encodeURIComponent(this.selectedArtifact.name)}/scaffold`,
        { method: 'POST' }
      );
      if (!scaffoldResp.ok) {
        const detail = await scaffoldResp.json().catch(() => ({}));
        this.errorEl.style.display = 'flex';
        this.errorEl.textContent = `Scaffold failed: ${detail.detail || `HTTP ${scaffoldResp.status}`}`;
        return;
      }
    }

    try {
      const resp = await fetch(
        `/api/prompts/${encodeURIComponent(this.selectedArtifact.name)}/override`,
        {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ content }),
        }
      );

      if (!resp.ok) {
        const detail = await resp.json().catch(() => ({}));
        throw new Error(detail.detail || `Save failed (HTTP ${resp.status})`);
      }

      this.editDirty = false;
      await this.reloadAndReopen();
    } catch (e) {
      this.errorEl.style.display = 'flex';
      this.errorEl.textContent = `Save failed: ${e.message}`;
    }
  }

  /**
   * Show a modal warning that saving settings.json means taking ownership.
   * Returns a promise that resolves to true (proceed) or false (cancel).
   */
  _showOwnershipWarning() {
    return new Promise((resolve) => {
      const overlay = document.createElement('div');
      overlay.className = 'config-ownership-overlay';

      const dialog = document.createElement('div');
      dialog.className = 'config-ownership-dialog';

      const iconEl = document.createElement('div');
      iconEl.className = 'config-ownership-icon';
      iconEl.textContent = '\u26A0';
      dialog.appendChild(iconEl);

      const title = document.createElement('div');
      title.className = 'config-ownership-title';
      title.textContent = 'Taking Ownership';
      dialog.appendChild(title);

      const body = document.createElement('div');
      body.className = 'config-ownership-body';
      body.textContent =
        'You are about to take ownership of settings.json. ' +
        'OSPREY will no longer auto-manage this file during regeneration ' +
        '(osprey claude regen). Future framework updates to permissions, ' +
        'hooks, and model configuration will not be applied automatically. ' +
        'You can release ownership later to restore framework management.';
      dialog.appendChild(body);

      const actions = document.createElement('div');
      actions.className = 'config-ownership-actions';

      const cancelBtn = document.createElement('button');
      cancelBtn.className = 'config-ownership-cancel';
      cancelBtn.textContent = 'Cancel';
      cancelBtn.addEventListener('click', () => {
        overlay.remove();
        resolve(false);
      });
      actions.appendChild(cancelBtn);

      const confirmBtn = document.createElement('button');
      confirmBtn.className = 'config-ownership-confirm';
      confirmBtn.textContent = 'I Understand, Save';
      confirmBtn.addEventListener('click', () => {
        overlay.remove();
        resolve(true);
      });
      actions.appendChild(confirmBtn);

      dialog.appendChild(actions);
      overlay.appendChild(dialog);
      document.body.appendChild(overlay);

      // Animate in
      requestAnimationFrame(() => overlay.classList.add('visible'));
    });
  }

  // ---- Unoverride Flow ---- //

  async unoverrideArtifact(skipConfirm = false) {
    if (!this.selectedArtifact) return;

    if (!skipConfirm) {
      if (!confirm('Reset to framework default? This will remove your customizations.')) {
        return;
      }
    }

    try {
      const resp = await fetch(
        `/api/prompts/${encodeURIComponent(this.selectedArtifact.name)}/override?delete_file=true`,
        { method: 'DELETE' }
      );

      if (!resp.ok) {
        const detail = await resp.json().catch(() => ({}));
        throw new Error(detail.detail || `Reset failed (HTTP ${resp.status})`);
      }

      await this.reloadAndReopen();
    } catch (e) {
      this.errorEl.style.display = 'flex';
      this.errorEl.textContent = `Reset failed: ${e.message}`;
    }
  }

  // ---- Reload & Reopen ---- //

  async reloadAndReopen() {
    const name = this.selectedArtifact ? this.selectedArtifact.name : null;

    // Invalidate cache and refetch
    resetFetchCache();
    const data = await fetchArtifactsShared();
    const allArtifacts = data.artifacts || [];
    this.artifacts = allArtifacts
      .filter(this.categoryFilter)
      .map((a) => ({
        ...a,
        displayCategory:
          this.categoryOverrides[a.name] ||
          this.categoryRemaps[a.category] ||
          a.category,
      }));

    // Recompute summary
    const fw = this.artifacts.filter((a) => a.status === 'framework').length;
    const uo = this.artifacts.filter((a) => a.status === 'user-owned').length;
    this.summary = { total: this.artifacts.length, framework: fw, userOwned: uo };

    if (name) {
      const updated = this.artifacts.find((a) => a.name === name);
      if (updated) {
        this.openDetail(updated);
        return;
      }
    }

    this.renderGallery();
  }

  // ---- Close Detail ---- //

  closeDetail() {
    if (this.editDirty) {
      if (!confirm('You have unsaved changes. Discard them?')) return;
    }

    this.currentView = 'gallery';
    this.selectedArtifact = null;
    this.editDirty = false;
    this.detailMode = 'preview';

    if (this.galleryView) this.galleryView.style.display = '';
    if (this.detailView) this.detailView.style.display = 'none';

    if (this.onDetailClose) this.onDetailClose();

    // Re-render gallery so cards reflect any ownership changes
    this.renderGallery();
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
export function initPromptsGallery() {
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
  registerUnsavedGuard('settings-drawer', () => {
    const dirty = behaviorGallery.editDirty || safetyGallery.editDirty || configGallery.editDirty;
    if (!dirty) return true;
    return confirm('You have unsaved changes. Discard them?');
  });
}

// ---- Module-Level Utility Functions ---- //

/** Create a DOM element with a class name. */
function _el(tag, className) {
  const el = document.createElement(tag);
  el.className = className;
  return el;
}

/** Escape HTML special characters to prevent XSS. */
function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

/** Return an emoji icon for a given artifact category. */
function iconForCategory(cat) {
  switch ((cat || '').toLowerCase()) {
    case 'system prompt':   return '\uD83D\uDCDC'; // scroll
    case 'instructions': return '\uD83D\uDCCB'; // clipboard
    case 'agents':   return '\uD83E\uDD16';  // robot
    case 'hooks':    return '\u26A1';         // lightning
    case 'commands': return '\u2318';         // command key
    case 'config':   return '\u2699';         // gear
    case 'skills':   return '\uD83D\uDCE6';  // package
    default:         return '\uD83D\uDCC4';   // document
  }
}

/**
 * Standard debounce: delays invoking fn until after ms milliseconds
 * have elapsed since the last invocation.
 */
function debounce(fn, ms) {
  let timer = null;
  return function (...args) {
    clearTimeout(timer);
    timer = setTimeout(() => fn.apply(this, args), ms);
  };
}

/**
 * Toggle a category help tooltip on/off.
 */
function toggleCategoryTooltip(btn, text) {
  const existing = btn.parentElement.querySelector('.prompts-category-tooltip');
  if (existing) {
    existing.remove();
    return;
  }

  document.querySelectorAll('.prompts-category-tooltip').forEach((t) => t.remove());

  const tip = document.createElement('div');
  tip.className = 'prompts-category-tooltip';
  tip.textContent = text;
  btn.parentElement.appendChild(tip);

  const handler = (e) => {
    if (!tip.contains(e.target) && e.target !== btn) {
      tip.remove();
      document.removeEventListener('click', handler);
    }
  };
  setTimeout(() => document.addEventListener('click', handler), 0);
}

/**
 * Parse YAML front matter (between --- delimiters) from markdown content.
 * Returns { frontMatter: {key: value, ...} | null, body: string }.
 */
function parseFrontMatter(content) {
  const match = content.match(/^---\n([\s\S]*?)\n---\n?([\s\S]*)$/);
  if (!match) return { frontMatter: null, body: content };

  const yamlBlock = match[1];
  const body = match[2];

  const fields = {};
  for (const line of yamlBlock.split('\n')) {
    const kv = line.match(/^(\w[\w\-]*):\s*(.*)$/);
    if (kv) {
      let value = kv[2].trim();
      if ((value.startsWith('"') && value.endsWith('"')) ||
          (value.startsWith("'") && value.endsWith("'"))) {
        value = value.slice(1, -1);
      }
      fields[kv[1]] = value;
    }
  }

  return { frontMatter: Object.keys(fields).length > 0 ? fields : null, body };
}

/**
 * Extract YAML front matter from a Python module docstring.
 */
function extractPythonDocstringFrontMatter(content) {
  const result = { frontMatter: null, body: '', flowDiagram: null, sourceCode: content };

  const docMatch = content.match(/^(?:#!.*\n)?"""\n?([\s\S]*?)"""/);
  if (!docMatch) return result;

  const docstring = docMatch[1];
  const { frontMatter, body } = parseFrontMatter(docstring);

  result.frontMatter = frontMatter;

  const trimmed = body.trim();
  const flowMatch = trimmed.match(/## Flow\s*\n\s*```\n?([\s\S]*?)```/);
  if (flowMatch) {
    result.flowDiagram = flowMatch[1].trimEnd();
    result.body = trimmed.replace(/## Flow\s*\n\s*```\n?[\s\S]*?```/, '').trim();
  } else {
    result.body = trimmed;
  }

  return result;
}

/** Create a syntax-highlighted code block element. */
function renderHighlightedCode(content, language) {
  const pre = document.createElement('pre');
  const code = document.createElement('code');
  if (language) code.className = `language-${language}`;
  code.textContent = content;
  pre.appendChild(code);

  if (typeof hljs !== 'undefined') {
    try {
      hljs.highlightElement(code);
    } catch {
      // Fall back to plain text
    }
  }

  return pre;
}

/** Render an ASCII flow diagram as a styled pre block. */
function renderFlowDiagram(diagramText) {
  const section = document.createElement('div');
  section.className = 'prompts-flow-diagram';

  const heading = document.createElement('div');
  heading.className = 'prompts-flow-heading';
  heading.textContent = 'FLOW';
  section.appendChild(heading);

  const pre = document.createElement('pre');
  pre.className = 'prompts-flow-pre';
  const code = document.createElement('code');
  code.textContent = diagramText;
  pre.appendChild(code);
  section.appendChild(pre);

  return section;
}

/** Create a "View Source" collapsible toggle with syntax-highlighted code. */
function renderSourceToggle(sourceCode, language) {
  const container = document.createElement('div');
  container.className = 'prompts-source-section';

  const toggle = document.createElement('button');
  toggle.className = 'prompts-source-toggle';
  toggle.innerHTML = '<span class="prompts-source-arrow">\u25B6</span> VIEW SOURCE';
  container.appendChild(toggle);

  const content = document.createElement('div');
  content.className = 'prompts-source-content';
  content.appendChild(renderHighlightedCode(sourceCode, language));
  container.appendChild(content);

  toggle.addEventListener('click', () => {
    const expanded = content.classList.toggle('expanded');
    toggle.querySelector('.prompts-source-arrow').textContent = expanded ? '\u25BC' : '\u25B6';
  });

  return container;
}

/** Render front matter fields as a styled key-value table. */
function renderFrontMatterTable(fields) {
  const table = document.createElement('div');
  table.className = 'prompts-frontmatter';

  for (const [key, value] of Object.entries(fields)) {
    const row = document.createElement('div');
    row.className = 'prompts-fm-row';

    const keyEl = document.createElement('span');
    keyEl.className = 'prompts-fm-key';
    keyEl.textContent = key;

    const valEl = document.createElement('span');
    valEl.className = 'prompts-fm-value';

    if (key === 'disallowedTools' || key === 'tools') {
      const tools = value.split(',').map((t) => t.trim()).filter(Boolean);
      for (const tool of tools) {
        const pill = document.createElement('span');
        pill.className = 'prompts-fm-pill';
        pill.textContent = tool;
        valEl.appendChild(pill);
      }
    } else if (key === 'model' || key === 'event') {
      const pill = document.createElement('span');
      pill.className = 'prompts-fm-pill prompts-fm-pill-accent';
      pill.textContent = value;
      valEl.appendChild(pill);
    } else if (key === 'safety_layer') {
      const pill = document.createElement('span');
      pill.className = 'prompts-fm-pill prompts-fm-pill-shield';
      pill.textContent = '\uD83D\uDEE1\uFE0F Layer ' + value;
      valEl.appendChild(pill);
    } else {
      valEl.textContent = value;
    }

    row.appendChild(keyEl);
    row.appendChild(valEl);
    table.appendChild(row);
  }

  return table;
}
