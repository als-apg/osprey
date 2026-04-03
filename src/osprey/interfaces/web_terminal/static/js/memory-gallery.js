/* OSPREY Web Terminal — Memory Gallery ("Lab Notebook")
 *
 * Drives the "Memory" tab in the settings drawer.
 * Reads/writes Claude Code native memory files at
 * ~/.claude/projects/<encoded>/memory/*.md via the /api/claude-memory
 * route family.
 *
 * Aesthetic: Lab-notebook with amber (primary MEMORY.md) and teal
 * (topic files) accents, monospace typography, geometric icons.
 *
 * API endpoints consumed:
 *   GET    /api/claude-memory              -> list all memory files
 *   GET    /api/claude-memory/{filename}   -> read file content
 *   POST   /api/claude-memory              -> create new file
 *   PUT    /api/claude-memory/{filename}   -> update file content
 *   DELETE /api/claude-memory/{filename}   -> delete file
 */

import { fetchJSON } from './api.js';
import { registerUnsavedGuard } from './drawer.js';

// ---- Constants ---- //

const TRUNCATION_LIMIT = 200;
const TRUNCATION_WARNING = 180;

// ---- Shared Fetch Cache ---- //

let _fetchPromise = null;

async function fetchMemoryFilesShared() {
  if (!_fetchPromise) _fetchPromise = fetchJSON('/api/claude-memory');
  return _fetchPromise;
}

function resetFetchCache() {
  _fetchPromise = null;
}

// ---- MemoryGallery Class ---- //

class MemoryGallery {
  constructor({ container }) {
    this.container = container;

    // State
    this.files = [];
    this.selectedFile = null;
    this.currentView = 'gallery';
    this.editDirty = false;
    this.loaded = false;
    this.searchQuery = '';

    // DOM refs (populated by _buildDOM)
    this.loadingEl = null;
    this.errorEl = null;
    this.galleryView = null;
    this.detailView = null;
    this.searchInput = null;
    this.summaryEl = null;
    this.fileListEl = null;
    this.detailHeaderEl = null;
    this.detailContentEl = null;
    this.detailActionsEl = null;

    this._buildDOM();
  }

  // ---- DOM Construction ---- //

  _buildDOM() {
    this.container.innerHTML = '';

    // Loading
    this.loadingEl = _el('div', 'memory-loading');
    this.loadingEl.textContent = 'Loading memory files...';
    this.loadingEl.style.display = 'none';
    this.container.appendChild(this.loadingEl);

    // Error
    this.errorEl = _el('div', 'memory-error');
    this.errorEl.style.display = 'none';
    this.container.appendChild(this.errorEl);

    // Gallery view
    this.galleryView = _el('div', 'memory-gallery-view');

    // Search bar
    const searchBar = _el('div', 'memory-search-bar');

    this.searchInput = document.createElement('input');
    this.searchInput.type = 'text';
    this.searchInput.className = 'memory-search-input';
    this.searchInput.placeholder = 'Search memory files...';
    this.searchInput.spellcheck = false;
    searchBar.appendChild(this.searchInput);

    const newBtn = document.createElement('button');
    newBtn.className = 'memory-new-btn';
    newBtn.textContent = '+ New';
    newBtn.title = 'Create new memory file';
    newBtn.addEventListener('click', () => this.promptCreateFile());
    searchBar.appendChild(newBtn);

    this.galleryView.appendChild(searchBar);

    // Summary strip
    this.summaryEl = _el('div', 'memory-summary');
    this.galleryView.appendChild(this.summaryEl);

    // File list
    this.fileListEl = _el('div', 'memory-file-list');
    this.galleryView.appendChild(this.fileListEl);

    this.container.appendChild(this.galleryView);

    // Detail view
    this.detailView = _el('div', 'memory-detail-view');
    this.detailView.style.display = 'none';

    this.detailHeaderEl = _el('div', 'memory-detail-header');
    this.detailActionsEl = _el('div', 'memory-detail-actions');
    this.detailContentEl = _el('div', 'memory-detail-content');

    this.detailView.appendChild(this.detailHeaderEl);
    this.detailView.appendChild(this.detailActionsEl);
    this.detailView.appendChild(this.detailContentEl);

    this.container.appendChild(this.detailView);
  }

  // ---- Data Loading ---- //

  async load() {
    this.loadingEl.style.display = 'flex';
    this.errorEl.style.display = 'none';

    try {
      const data = await fetchMemoryFilesShared();
      this.files = data.files || [];
      this.loadingEl.style.display = 'none';
      this.renderGallery();
      this.loaded = true;
    } catch (e) {
      this.loadingEl.style.display = 'none';
      this.errorEl.style.display = 'flex';
      this.errorEl.textContent = `Failed to load memory files: ${e.message}`;
    }
  }

  // ---- Gallery View ---- //

  renderGallery() {
    if (this.galleryView) this.galleryView.style.display = '';
    if (this.detailView) this.detailView.style.display = 'none';
    this.currentView = 'gallery';

    this.bindSearch();
    this.renderSummary();
    this.renderFileList();
  }

  bindSearch() {
    if (!this.searchInput) return;

    const clone = this.searchInput.cloneNode(true);
    this.searchInput.parentNode.replaceChild(clone, this.searchInput);
    this.searchInput = clone;
    clone.value = this.searchQuery;

    const debouncedRender = debounce(() => {
      this.searchQuery = clone.value.trim();
      this.renderFileList();
    }, 150);

    clone.addEventListener('input', debouncedRender);
  }

  renderSummary() {
    if (!this.summaryEl) return;
    const total = this.files.length;
    const primary = this.files.find(f => f.is_primary);
    const topicCount = total - (primary ? 1 : 0);
    this.summaryEl.textContent =
      `${total} file${total !== 1 ? 's' : ''} \u00B7 ` +
      `${primary ? '1 primary' : 'no primary'} \u00B7 ` +
      `${topicCount} topic`;
  }

  renderFileList() {
    if (!this.fileListEl) return;
    this.fileListEl.innerHTML = '';

    const filtered = this.getFilteredFiles();

    if (filtered.length === 0) {
      const empty = _el('div', 'memory-empty');
      empty.textContent = this.files.length === 0
        ? 'No memory files yet. Click "+ New" to create one.'
        : 'No matching files.';
      this.fileListEl.appendChild(empty);
      return;
    }

    // Primary file first, then alphabetical
    const sorted = [...filtered].sort((a, b) => {
      if (a.is_primary && !b.is_primary) return -1;
      if (!a.is_primary && b.is_primary) return 1;
      return a.filename.localeCompare(b.filename);
    });

    for (const file of sorted) {
      this.fileListEl.appendChild(this.renderFileCard(file));
    }
  }

  renderFileCard(file) {
    const card = _el('div', 'memory-file-card');
    if (file.is_primary) card.classList.add('memory-file-primary');

    // Icon
    const icon = _el('div', 'memory-file-icon');
    icon.textContent = file.is_primary ? '\u25C8' : '\u25C7'; // ◈ filled diamond / ◇ open diamond
    if (file.is_primary) icon.classList.add('memory-icon-primary');
    card.appendChild(icon);

    // Body
    const body = _el('div', 'memory-file-body');

    const nameRow = _el('div', 'memory-file-name');
    nameRow.textContent = file.filename;
    body.appendChild(nameRow);

    const metaRow = _el('div', 'memory-file-meta');
    metaRow.textContent = `${file.line_count} lines \u00B7 ${formatSize(file.size)}`;
    body.appendChild(metaRow);

    card.appendChild(body);

    // Line gauge (for primary file)
    if (file.is_primary) {
      const gauge = this.renderLineGauge(file.line_count);
      card.appendChild(gauge);
    }

    // Badge
    const badge = _el('span', 'memory-file-badge');
    if (file.is_primary) {
      badge.classList.add('memory-badge-primary');
      badge.textContent = 'PRIMARY';
    } else {
      badge.classList.add('memory-badge-topic');
      badge.textContent = 'TOPIC';
    }
    card.appendChild(badge);

    card.addEventListener('click', () => this.openDetail(file));
    return card;
  }

  renderLineGauge(lineCount) {
    const gauge = _el('div', 'memory-line-gauge');
    const fill = _el('div', 'memory-line-gauge-fill');
    const pct = Math.min(100, (lineCount / TRUNCATION_LIMIT) * 100);
    fill.style.width = pct + '%';

    if (lineCount >= TRUNCATION_LIMIT) {
      fill.classList.add('memory-gauge-over');
    } else if (lineCount >= TRUNCATION_WARNING) {
      fill.classList.add('memory-gauge-warn');
    }

    gauge.appendChild(fill);
    gauge.title = `${lineCount}/${TRUNCATION_LIMIT} lines (truncated after ${TRUNCATION_LIMIT})`;
    return gauge;
  }

  getFilteredFiles() {
    if (!this.searchQuery) return this.files;
    const q = this.searchQuery.toLowerCase();
    return this.files.filter(f => f.filename.toLowerCase().includes(q));
  }

  // ---- Detail View ---- //

  async openDetail(file) {
    this.selectedFile = file;
    this.currentView = 'detail';
    this.editDirty = false;

    if (this.galleryView) this.galleryView.style.display = 'none';
    if (this.detailView) this.detailView.style.display = '';

    this.renderDetailHeader();
    this.renderDetailActions();
    await this.renderDetailContent();
  }

  renderDetailHeader() {
    if (!this.detailHeaderEl || !this.selectedFile) return;
    this.detailHeaderEl.innerHTML = '';

    const backBtn = document.createElement('button');
    backBtn.className = 'memory-back-btn';
    backBtn.textContent = '\u2190 Back';
    backBtn.addEventListener('click', () => this.closeDetail());
    this.detailHeaderEl.appendChild(backBtn);

    const nameEl = _el('span', 'memory-detail-name');
    nameEl.textContent = this.selectedFile.filename;
    this.detailHeaderEl.appendChild(nameEl);

    const spacer = _el('span', '');
    spacer.style.flex = '1';
    this.detailHeaderEl.appendChild(spacer);

    const badge = _el('span', 'memory-file-badge');
    if (this.selectedFile.is_primary) {
      badge.classList.add('memory-badge-primary');
      badge.textContent = 'PRIMARY';
    } else {
      badge.classList.add('memory-badge-topic');
      badge.textContent = 'TOPIC';
    }
    this.detailHeaderEl.appendChild(badge);
  }

  renderDetailActions() {
    if (!this.detailActionsEl) return;
    this.detailActionsEl.innerHTML = '';

    // Left: line gauge for primary
    const left = _el('div', 'memory-actions-left');
    if (this.selectedFile && this.selectedFile.is_primary) {
      const gauge = this.renderLineGauge(this.selectedFile.line_count);
      gauge.style.width = '120px';
      left.appendChild(gauge);

      const label = _el('span', 'memory-gauge-label');
      label.textContent = `${this.selectedFile.line_count}/${TRUNCATION_LIMIT}`;
      left.appendChild(label);
    }
    this.detailActionsEl.appendChild(left);

    // Right: action buttons
    const right = _el('div', 'memory-actions-right');

    if (!this.selectedFile?.is_primary) {
      const deleteBtn = document.createElement('button');
      deleteBtn.className = 'memory-delete-btn';
      deleteBtn.textContent = 'Delete';
      deleteBtn.addEventListener('click', () => this.deleteCurrentFile());
      right.appendChild(deleteBtn);
    }

    const discardBtn = document.createElement('button');
    discardBtn.className = 'memory-discard-btn';
    discardBtn.textContent = 'Discard';
    discardBtn.disabled = !this.editDirty;
    discardBtn.addEventListener('click', () => this.discardEdits());
    right.appendChild(discardBtn);

    const saveBtn = document.createElement('button');
    saveBtn.className = 'memory-save-btn';
    saveBtn.textContent = 'Save';
    saveBtn.disabled = !this.editDirty;
    saveBtn.addEventListener('click', () => this.saveFile());
    right.appendChild(saveBtn);

    this.detailActionsEl.appendChild(right);
  }

  async renderDetailContent() {
    if (!this.detailContentEl || !this.selectedFile) return;
    this.detailContentEl.innerHTML = '<div class="memory-loading-inline">Loading...</div>';

    try {
      const data = await fetchJSON(`/api/claude-memory/${encodeURIComponent(this.selectedFile.filename)}`);
      this.detailContentEl.innerHTML = '';

      const textarea = document.createElement('textarea');
      textarea.className = 'memory-edit-textarea';
      textarea.spellcheck = false;
      textarea.value = data.content || '';

      textarea.addEventListener('input', () => {
        this.editDirty = true;
        this.renderDetailActions();

        // Update line gauge live
        const content = textarea.value;
        const lines = content.split('\n').length;
        this.selectedFile = { ...this.selectedFile, line_count: lines };
      });

      this.detailContentEl.appendChild(textarea);
    } catch (e) {
      this.detailContentEl.innerHTML =
        `<div class="memory-content-error">Error: ${escapeHtml(e.message)}</div>`;
    }
  }

  // ---- Actions ---- //

  async saveFile() {
    if (!this.selectedFile) return;
    const textarea = this.detailContentEl.querySelector('.memory-edit-textarea');
    if (!textarea) return;

    try {
      const resp = await fetch(`/api/claude-memory/${encodeURIComponent(this.selectedFile.filename)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: textarea.value }),
      });

      if (!resp.ok) {
        const detail = await resp.json().catch(() => ({}));
        throw new Error(detail.detail || `Save failed (HTTP ${resp.status})`);
      }

      const updated = await resp.json();
      this.editDirty = false;

      // Update the file in the list
      this.selectedFile = { ...this.selectedFile, ...updated };
      const idx = this.files.findIndex(f => f.filename === this.selectedFile.filename);
      if (idx >= 0) this.files[idx] = this.selectedFile;

      this.renderDetailActions();
      this.renderDetailHeader();
    } catch (e) {
      this.errorEl.style.display = 'flex';
      this.errorEl.textContent = `Save failed: ${e.message}`;
      setTimeout(() => { this.errorEl.style.display = 'none'; }, 4000);
    }
  }

  discardEdits() {
    this.editDirty = false;
    this.renderDetailContent();
    this.renderDetailActions();
  }

  async deleteCurrentFile() {
    if (!this.selectedFile) return;
    if (!confirm(`Delete "${this.selectedFile.filename}"? This cannot be undone.`)) return;

    try {
      const resp = await fetch(`/api/claude-memory/${encodeURIComponent(this.selectedFile.filename)}`, {
        method: 'DELETE',
      });

      if (!resp.ok) {
        const detail = await resp.json().catch(() => ({}));
        throw new Error(detail.detail || `Delete failed (HTTP ${resp.status})`);
      }

      this.files = this.files.filter(f => f.filename !== this.selectedFile.filename);
      this.selectedFile = null;
      this.editDirty = false;
      this.renderGallery();
    } catch (e) {
      this.errorEl.style.display = 'flex';
      this.errorEl.textContent = `Delete failed: ${e.message}`;
      setTimeout(() => { this.errorEl.style.display = 'none'; }, 4000);
    }
  }

  async promptCreateFile() {
    const filename = prompt('New memory file name (must end with .md):', 'new-topic.md');
    if (!filename) return;

    // Basic client-side validation
    if (!filename.endsWith('.md')) {
      alert('Filename must end with .md');
      return;
    }

    try {
      const resp = await fetch('/api/claude-memory', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename, content: `# ${filename.replace('.md', '')}\n\n` }),
      });

      if (!resp.ok) {
        const detail = await resp.json().catch(() => ({}));
        throw new Error(detail.detail || `Create failed (HTTP ${resp.status})`);
      }

      const newFile = await resp.json();
      this.files.push(newFile);
      this.renderSummary();
      this.openDetail(newFile);
    } catch (e) {
      this.errorEl.style.display = 'flex';
      this.errorEl.textContent = `Create failed: ${e.message}`;
      setTimeout(() => { this.errorEl.style.display = 'none'; }, 4000);
    }
  }

  // ---- Close Detail ---- //

  closeDetail() {
    if (this.editDirty) {
      if (!confirm('You have unsaved changes. Discard them?')) return;
    }

    this.currentView = 'gallery';
    this.selectedFile = null;
    this.editDirty = false;

    if (this.galleryView) this.galleryView.style.display = '';
    if (this.detailView) this.detailView.style.display = 'none';

    this.renderGallery();
  }

  // ---- State Reset ---- //

  reset() {
    this.files = [];
    this.selectedFile = null;
    this.currentView = 'gallery';
    this.editDirty = false;
    this.loaded = false;
    this.searchQuery = '';
  }
}

// ---- Public Export ---- //

export function initMemoryGallery() {
  const drawer = document.getElementById('settings-drawer');
  if (!drawer) return;

  const memoryPanel = document.getElementById('tab-memory');
  if (!memoryPanel) return;

  const memoryGallery = new MemoryGallery({ container: memoryPanel });

  // Load when tab becomes active
  memoryPanel.addEventListener('drawer:tab-activate', () => {
    if (!memoryGallery.loaded) memoryGallery.load();
  });

  // Reset on drawer close (B2)
  drawer.addEventListener('drawer:close', () => {
    memoryGallery.reset();
    resetFetchCache();
  });

  // Register unsaved-changes guard (uses composite array via B1 fix)
  registerUnsavedGuard('settings-drawer', () => {
    if (!memoryGallery.editDirty) return true;
    return confirm('You have unsaved memory changes. Discard them?');
  });
}

// ---- Utility Functions ---- //

function _el(tag, className) {
  const el = document.createElement(tag);
  el.className = className;
  return el;
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function formatSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / 1048576).toFixed(1) + ' MB';
}

function debounce(fn, ms) {
  let timer = null;
  return function (...args) {
    clearTimeout(timer);
    timer = setTimeout(() => fn.apply(this, args), ms);
  };
}
