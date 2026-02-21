/* OSPREY Web Terminal — Memory Tab (Settings Drawer)
 *
 * Renders agent memory entries as cards with search/filter.
 * Fetches from the gallery server /api/memory endpoints.
 */

import { fetchJSON } from './api.js';

let memoryEntries = [];
let listEl = null;
let searchEl = null;
let filterEl = null;
let galleryBaseUrl = '';

/**
 * Initialize the memory tab. Call once on DOMContentLoaded.
 */
export function initMemoryTab() {
  listEl = document.getElementById('memory-tab-list');
  searchEl = document.getElementById('memory-search');
  filterEl = document.getElementById('memory-filter-importance');

  if (!listEl) return;

  // Listen for tab activation
  const panel = document.getElementById('tab-memory');
  if (panel) {
    panel.addEventListener('drawer:tab-activate', () => loadMemories());
  }

  // Search & filter
  if (searchEl) searchEl.addEventListener('input', debounce(renderList, 200));
  if (filterEl) filterEl.addEventListener('change', renderList);
}

async function loadMemories() {
  if (!listEl) return;
  listEl.innerHTML = '<div class="memory-loading">Loading memories...</div>';

  try {
    // Resolve gallery URL from config
    if (!galleryBaseUrl) {
      try {
        const cfg = await fetchJSON('/api/config');
        const port = cfg?.sections?.artifact_server?.port || 8086;
        galleryBaseUrl = `http://127.0.0.1:${port}`;
      } catch {
        galleryBaseUrl = 'http://127.0.0.1:8086';
      }
    }

    const resp = await fetch(`${galleryBaseUrl}/api/memory`);
    const data = await resp.json();
    memoryEntries = data.entries || [];
    renderList();
  } catch (err) {
    console.error('Failed to load memories:', err);
    listEl.innerHTML = '<div class="memory-empty">Failed to connect to gallery server</div>';
  }
}

function getFiltered() {
  let entries = [...memoryEntries];

  const importance = filterEl ? filterEl.value : '';
  if (importance) {
    entries = entries.filter(m => m.importance === importance);
  }

  const query = searchEl ? searchEl.value.trim().toLowerCase() : '';
  if (query) {
    entries = entries.filter(m =>
      m.content.toLowerCase().includes(query) ||
      (m.tags && m.tags.some(t => t.toLowerCase().includes(query))) ||
      m.memory_type.toLowerCase().includes(query)
    );
  }

  return entries;
}

function renderList() {
  if (!listEl) return;
  const entries = getFiltered();

  if (entries.length === 0) {
    listEl.innerHTML = `<div class="memory-empty">${
      memoryEntries.length === 0
        ? 'No memories saved yet'
        : 'No matches'
    }</div>`;
    return;
  }

  listEl.innerHTML = entries.map(m => memoryCardHtml(m)).join('');
  attachCardHandlers();
}

function memoryCardHtml(m) {
  const importanceBadge = m.importance === 'important'
    ? '<span class="memory-importance-badge important">&#9733; important</span>'
    : '<span class="memory-importance-badge normal">normal</span>';

  const tagsHtml = m.tags && m.tags.length > 0
    ? m.tags.map(t => `<span class="memory-tag">${esc(t)}</span>`).join('')
    : '';

  const typeLabel = m.memory_type === 'pin' ? 'Pin' : 'Note';
  const preview = m.content.length > 200 ? m.content.substring(0, 200) + '...' : m.content;

  return `
    <div class="memory-card" data-id="${m.id}">
      <div class="memory-card-header">
        <span class="memory-type-badge">${typeLabel} #${m.id}</span>
        ${importanceBadge}
        <span class="memory-card-time">${formatTime(m.timestamp)}</span>
        <button class="memory-delete-btn" data-id="${m.id}" title="Delete">&times;</button>
      </div>
      <div class="memory-card-content">${esc(preview)}</div>
      ${tagsHtml ? `<div class="memory-card-tags">${tagsHtml}</div>` : ''}
    </div>`;
}

function attachCardHandlers() {
  if (!listEl) return;
  listEl.querySelectorAll('.memory-delete-btn').forEach(btn => {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const id = btn.dataset.id;
      if (!confirm(`Delete memory #${id}?`)) return;
      try {
        await fetch(`${galleryBaseUrl}/api/memory/${id}`, { method: 'DELETE' });
        memoryEntries = memoryEntries.filter(m => String(m.id) !== id);
        renderList();
      } catch (err) {
        console.error('Delete failed:', err);
      }
    });
  });
}

function esc(s) {
  const d = document.createElement('div');
  d.textContent = s || '';
  return d.innerHTML;
}

function formatTime(iso) {
  if (!iso) return '';
  try {
    const d = new Date(iso);
    return d.toLocaleString(undefined, {
      month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
    });
  } catch { return ''; }
}

function debounce(fn, ms) {
  let timer;
  return function (...args) {
    clearTimeout(timer);
    timer = setTimeout(() => fn.apply(this, args), ms);
  };
}
