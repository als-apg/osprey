/**
 * OSPREY Channel Finder — In-Context Explore (Chunk-Paginated Table)
 *
 * Loads channels in chunks with client-side filtering on name/description.
 * Inline CRUD: add and delete channels.
 */

import { fetchJSON, postJSON, deleteJSON } from './api.js';
import { showToast } from './app.js';
import { esc } from './utils.js';
import { formModal, confirmModal } from './modal.js';
import { refreshStatsBadges } from './stats-badges.js';

let containerEl = null;
let allChannels = [];   // full loaded set
let filterText = '';
let chunkIdx = 0;
let totalChunks = 0;
const CHUNK_SIZE = 50;

export async function mountInContext(container) {
  containerEl = container;

  container.innerHTML = `
    <div class="filter-bar">
      <span class="filter-label">Filter:</span>
      <input type="text" class="filter-input" id="ic-filter"
             placeholder="Type to filter by name or description...">
      <span class="filter-label" id="ic-count"></span>
      <button class="btn btn-primary btn-sm" id="ic-add-channel">+ Add Channel</button>
    </div>
    <div id="ic-table-area">
      <div class="loading-center"><div class="loading-spinner"></div> Loading channels...</div>
    </div>
    <div class="pagination" id="ic-pagination"></div>
  `;

  document.getElementById('ic-filter')?.addEventListener('input', (e) => {
    filterText = e.target.value.toLowerCase();
    renderTable();
  });

  document.getElementById('ic-add-channel')?.addEventListener('click', handleAddChannel);

  await loadChunk(0);
}

export function unmountInContext() {
  containerEl = null;
  allChannels = [];
  filterText = '';
  chunkIdx = 0;
}

async function loadChunk(idx) {
  try {
    const data = await fetchJSON(`/api/channels?chunk_idx=${idx}&chunk_size=${CHUNK_SIZE}`);

    chunkIdx = data.chunk_idx ?? idx;
    totalChunks = data.total_chunks ?? 1;

    // Parse channels from chunk
    if (data.channels) {
      allChannels = data.channels;
    } else if (data.formatted) {
      // Chunk mode returns formatted text; parse as best we can
      allChannels = parseFormattedChannels(data.formatted);
    }

    renderTable();
    renderPagination();
  } catch (e) {
    const area = document.getElementById('ic-table-area');
    if (area) area.innerHTML = `<div class="empty-state">Failed to load channels: ${e.message}</div>`;
  }
}

function parseFormattedChannels(text) {
  // The formatted output contains channel info as lines
  return text.split('\n')
    .filter(line => line.trim())
    .map(line => {
      const trimmed = line.trim();
      // Try to parse as "name: description" or "name (address)"
      const colonIdx = trimmed.indexOf(':');
      if (colonIdx > 0) {
        return {
          name: trimmed.substring(0, colonIdx).trim(),
          description: trimmed.substring(colonIdx + 1).trim(),
        };
      }
      return { name: trimmed, description: '' };
    });
}

function renderTable() {
  const area = document.getElementById('ic-table-area');
  const countEl = document.getElementById('ic-count');
  if (!area) return;

  const filtered = filterText
    ? allChannels.filter(ch => {
        const name = (ch.name || ch.channel_name || '').toLowerCase();
        const desc = (ch.description || ch.address || '').toLowerCase();
        return name.includes(filterText) || desc.includes(filterText);
      })
    : allChannels;

  if (countEl) {
    countEl.textContent = `${filtered.length} of ${allChannels.length}`;
  }

  if (filtered.length === 0) {
    area.innerHTML = '<div class="empty-state">No channels match the filter</div>';
    return;
  }

  area.innerHTML = `
    <div class="table-wrapper">
      <table class="data-table">
        <thead>
          <tr>
            <th style="width: 40px">#</th>
            <th>Name</th>
            <th>Address / Description</th>
            <th style="width: 40px"></th>
          </tr>
        </thead>
        <tbody>
          ${filtered.map((ch, i) => {
            const name = ch.name || ch.channel_name || ch.channel || '—';
            const desc = ch.description || ch.address || ch.pv_address || '';
            return `
              <tr>
                <td>${i + 1}</td>
                <td class="pv-cell">${esc(name)}</td>
                <td>${esc(desc)}</td>
                <td>
                  <button class="item-action-btn action-delete" data-channel="${esc(name)}" title="Delete">&times;</button>
                </td>
              </tr>
            `;
          }).join('')}
        </tbody>
      </table>
    </div>
  `;

  // Wire up delete buttons
  area.querySelectorAll('.item-action-btn.action-delete').forEach(btn => {
    btn.addEventListener('click', () => handleDeleteChannel(btn.dataset.channel));
  });
}

function renderPagination() {
  const pag = document.getElementById('ic-pagination');
  if (!pag || totalChunks <= 1) {
    if (pag) pag.innerHTML = '';
    return;
  }

  pag.innerHTML = `
    <button class="btn btn-secondary btn-sm" id="ic-prev" ${chunkIdx === 0 ? 'disabled' : ''}>
      &laquo; Prev
    </button>
    <span class="pagination-info">Chunk ${chunkIdx + 1} / ${totalChunks}</span>
    <button class="btn btn-secondary btn-sm" id="ic-next" ${chunkIdx >= totalChunks - 1 ? 'disabled' : ''}>
      Next &raquo;
    </button>
  `;

  document.getElementById('ic-prev')?.addEventListener('click', () => {
    if (chunkIdx > 0) loadChunk(chunkIdx - 1);
  });
  document.getElementById('ic-next')?.addEventListener('click', () => {
    if (chunkIdx < totalChunks - 1) loadChunk(chunkIdx + 1);
  });
}

// ---- CRUD Handlers ----

async function handleAddChannel() {
  const result = await formModal({
    title: 'Add Channel',
    fields: [
      { name: 'channel_name', label: 'Channel Name', required: true, placeholder: 'e.g., SR:BPM:01:X' },
      { name: 'address', label: 'PV Address', placeholder: 'EPICS PV address (defaults to name)' },
      { name: 'description', label: 'Description', placeholder: 'Human-readable description' },
    ],
  });

  if (!result) return;

  try {
    await postJSON('/api/channels', {
      channel_name: result.channel_name,
      address: result.address,
      description: result.description,
    });
    showToast(`Added "${result.channel_name}"`, 'success');
    await loadChunk(chunkIdx);
    refreshStatsBadges();
  } catch (e) {
    showToast(`Failed to add channel: ${e.message}`, 'error');
  }
}

async function handleDeleteChannel(channelName) {
  const confirmed = await confirmModal({
    title: `Delete "${channelName}"?`,
    message: 'Remove this channel from the database.',
    confirmLabel: 'Delete',
    danger: true,
  });

  if (!confirmed) return;

  try {
    await deleteJSON(`/api/channels/${encodeURIComponent(channelName)}`);
    showToast(`Deleted "${channelName}"`, 'success');
    await loadChunk(chunkIdx);
    refreshStatsBadges();
  } catch (e) {
    showToast(`Failed to delete: ${e.message}`, 'error');
  }
}
