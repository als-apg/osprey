// @ts-check
/**
 * OSPREY Channel Finder — In-Context Explore (Chunk-Paginated Table)
 *
 * Loads the FULL in-context database once, filters client-side over the whole
 * set, then re-chunks the FILTERED results for pagination. This is safe because
 * the in-context pipeline is bounded to fit an LLM context window (small DBs),
 * and the endpoint 404s for the large-DB pipelines. Filtering and pagination
 * derive from one shared filtered set (see chunk-filter.js) so a search match on
 * any page is always found — fixing the disjoint-scope bug in issue #299. (hygiene-allow-color: issue number, not a hex color)
 *
 * Inline CRUD: add, edit, and delete channels.
 */

import { fetchJSON, postJSON, putJSON, deleteJSON } from './api.js';
import { showToast } from './app.js';
import { esc, messageOf } from './utils.js';
import { formModal, confirmModal } from './modal.js';
import { refreshStatsBadges } from './stats-badges.js';
import { filterChannels, totalChunksFor, clampChunkIdx, pageSlice } from './chunk-filter.js';

/** @type {any[]} */
let allChannels = [];   // the ENTIRE in-context database (loaded once)
let filterText = '';
let chunkIdx = 0;
const CHUNK_SIZE = 50;

// Single source of truth: every render derives the filtered set from here, so
// the table page-slice and the pager can never operate on different scopes.
function getFiltered() {
  return filterChannels(allChannels, filterText);
}

/**
 * Resolve the active UI mode from the <html> data-ui-mode attribute stamped by
 * mode-boot.js (and updated live by app.js). Anything but "simple" is Expert.
 * @returns {'expert'|'simple'}
 */
function uiMode() {
  return document.documentElement.getAttribute('data-ui-mode') === 'simple'
    ? 'simple'
    : 'expert';
}

/**
 * @param {HTMLElement} container
 */
export async function mountInContext(container) {
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
    filterText = /** @type {HTMLInputElement} */ (e.target).value.toLowerCase();
    // Reset to the first page of the NEW filtered set, and re-render the pager:
    // the filtered set (and thus the chunk count) changes on every keystroke.
    chunkIdx = 0;
    renderTable();
    renderPagination();
  });

  document.getElementById('ic-add-channel')?.addEventListener('click', handleAddChannel);

  await loadAll();
}

export function unmountInContext() {
  allChannels = [];
  filterText = '';
  chunkIdx = 0;
  editingRow = null;
}

async function loadAll() {
  try {
    // Omitting chunk_idx returns the entire in-context DB: {channels, total}.
    const data = await fetchJSON('/api/channels');
    allChannels = data.channels || [];
    // Keep the current page valid if a CRUD refresh shrank the (filtered) set.
    chunkIdx = clampChunkIdx(chunkIdx, getFiltered().length, CHUNK_SIZE);

    renderTable();
    renderPagination();
  } catch (e) {
    const area = document.getElementById('ic-table-area');
    if (area) area.innerHTML = `<div class="empty-state">Failed to load channels: ${esc(messageOf(e))}</div>`;
  }
}

/** @type {string|null} */
let editingRow = null;

function renderTable() {
  const area = document.getElementById('ic-table-area');
  const countEl = document.getElementById('ic-count');
  if (!area) return;

  const filtered = getFiltered();
  // Page-slice the FILTERED set. `start` also offsets the row-number column so
  // the index stays continuous across pages.
  const start = chunkIdx * CHUNK_SIZE;
  const pageItems = pageSlice(filtered, chunkIdx, CHUNK_SIZE);

  if (countEl) {
    // Truthful only because allChannels holds the ENTIRE DB: "matches of total".
    // Do not reintroduce chunked loading without revisiting this.
    countEl.textContent = `${filtered.length} of ${allChannels.length}`;
  }

  // Simple mode (frame recipe): plain result cards — the channel address
  // prominent, a plain-language description below — with the dense table,
  // row-numbers and inline CRUD dropped. Chrome (filter label, Add button)
  // is hidden by CSS; only the results markup forks here.
  if (uiMode() === 'simple') {
    renderSimpleCards(area, filtered, pageItems);
    return;
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
            <th>Address</th>
            <th>Description</th>
            <th style="width: 80px"></th>
          </tr>
        </thead>
        <tbody>
          ${pageItems.map((ch, i) => {
            const name = ch.name || ch.channel_name || ch.channel || '—';
            const addr = ch.address || ch.pv_address || '';
            const desc = ch.description || '';
            const isEditing = editingRow === name;

            if (isEditing) {
              return `
                <tr class="ic-editing-row" data-channel="${esc(name)}">
                  <td>${start + i + 1}</td>
                  <td>
                    <input type="text" class="ic-inline-input" id="ic-edit-name"
                           value="${esc(name)}" placeholder="Channel name">
                  </td>
                  <td>
                    <input type="text" class="ic-inline-input" id="ic-edit-addr"
                           value="${esc(addr)}" placeholder="PV address">
                  </td>
                  <td>
                    <input type="text" class="ic-inline-input" id="ic-edit-desc"
                           value="${esc(desc)}" placeholder="Description">
                  </td>
                  <td>
                    <div class="ic-action-group">
                      <button class="item-action-btn action-save" data-channel="${esc(name)}" title="Save">&#10003;</button>
                      <button class="item-action-btn action-cancel" title="Cancel">&#10005;</button>
                    </div>
                  </td>
                </tr>
              `;
            }

            return `
              <tr>
                <td>${start + i + 1}</td>
                <td class="pv-cell">${esc(name)}</td>
                <td class="pv-cell">${esc(addr)}</td>
                <td>${esc(desc)}</td>
                <td>
                  <div class="ic-action-group">
                    <button class="item-action-btn action-edit" data-channel="${esc(name)}" title="Edit">&#9998;</button>
                    <button class="item-action-btn action-delete" data-channel="${esc(name)}" title="Delete">&times;</button>
                  </div>
                </td>
              </tr>
            `;
          }).join('')}
        </tbody>
      </table>
    </div>
  `;

  area.querySelectorAll('.item-action-btn.action-delete').forEach(btn => {
    btn.addEventListener('click', () => {
      const channel = /** @type {HTMLElement} */ (btn).dataset.channel;
      if (channel) handleDeleteChannel(channel);
    });
  });

  area.querySelectorAll('.item-action-btn.action-edit').forEach(btn => {
    btn.addEventListener('click', () => {
      editingRow = /** @type {HTMLElement} */ (btn).dataset.channel ?? null;
      renderTable();
      const input = /** @type {HTMLInputElement|null} */ (document.getElementById('ic-edit-name'));
      if (input) { input.focus(); input.select(); }
    });
  });

  area.querySelectorAll('.item-action-btn.action-save').forEach(btn => {
    btn.addEventListener('click', () => {
      const origName = /** @type {HTMLElement} */ (btn).dataset.channel;
      if (!origName) return;
      const nameInput = /** @type {HTMLInputElement|null} */ (document.getElementById('ic-edit-name'));
      const addrInput = /** @type {HTMLInputElement|null} */ (document.getElementById('ic-edit-addr'));
      const descInput = /** @type {HTMLInputElement|null} */ (document.getElementById('ic-edit-desc'));
      handleSaveEdit(
        origName,
        nameInput?.value.trim(),
        addrInput?.value.trim(),
        descInput?.value.trim(),
      );
    });
  });

  area.querySelectorAll('.item-action-btn.action-cancel').forEach(btn => {
    btn.addEventListener('click', () => {
      editingRow = null;
      renderTable();
    });
  });

  /** @type {HTMLElement[]} */
  const editInputs = /** @type {HTMLElement[]} */ ([
    document.getElementById('ic-edit-name'),
    document.getElementById('ic-edit-addr'),
    document.getElementById('ic-edit-desc'),
  ].filter(Boolean));
  editInputs.forEach(input => {
    input.addEventListener('keydown', (e) => {
      if (/** @type {KeyboardEvent} */ (e).key === 'Enter') {
        const row = /** @type {HTMLElement|null} */ (input.closest('tr'));
        const origName = row?.dataset.channel;
        const newName = /** @type {HTMLInputElement|null} */ (document.getElementById('ic-edit-name'))?.value.trim();
        const addrVal = /** @type {HTMLInputElement|null} */ (document.getElementById('ic-edit-addr'))?.value.trim();
        const descVal = /** @type {HTMLInputElement|null} */ (document.getElementById('ic-edit-desc'))?.value.trim();
        if (origName) handleSaveEdit(origName, newName, addrVal, descVal);
      } else if (/** @type {KeyboardEvent} */ (e).key === 'Escape') {
        editingRow = null;
        renderTable();
      }
    });
  });
}

/**
 * Simple-mode results: a friendly count line above plain channel cards.
 * @param {HTMLElement} area
 * @param {any[]} filtered - the full filtered set (for the count)
 * @param {any[]} pageItems - the current chunk's channels (for the cards)
 */
function renderSimpleCards(area, filtered, pageItems) {
  if (filtered.length === 0) {
    area.innerHTML = '<div class="empty-state">No channels match your search</div>';
    return;
  }

  const noun = filtered.length === 1 ? 'channel' : 'channels';
  const cards = pageItems.map((ch) => {
    const name = ch.name || ch.channel_name || ch.channel || '—';
    const desc = ch.description || '';
    return `
      <div class="cf-simple-card">
        <div class="cf-simple-card-name">${esc(name)}</div>
        ${desc ? `<div class="cf-simple-card-desc">${esc(desc)}</div>` : ''}
      </div>
    `;
  }).join('');

  area.innerHTML = `
    <div class="cf-simple-count">${filtered.length} ${noun} found</div>
    <div class="cf-simple-card-list">${cards}</div>
  `;
}

function renderPagination() {
  const pag = document.getElementById('ic-pagination');
  // Chunk count is derived from the FILTERED set, so the pager re-chunks with
  // every query change rather than reflecting the whole unfiltered DB.
  const totalChunks = totalChunksFor(getFiltered().length, CHUNK_SIZE);
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

  // Page flips are pure client-side over the already-loaded set — no network.
  document.getElementById('ic-prev')?.addEventListener('click', () => {
    if (chunkIdx > 0) { chunkIdx -= 1; renderTable(); renderPagination(); }
  });
  document.getElementById('ic-next')?.addEventListener('click', () => {
    if (chunkIdx < totalChunks - 1) { chunkIdx += 1; renderTable(); renderPagination(); }
  });
}

// ---- CRUD Handlers ----

/**
 * @param {string} origName
 * @param {string|undefined} newName
 * @param {string|undefined} newAddr
 * @param {string|undefined} newDesc
 */
async function handleSaveEdit(origName, newName, newAddr, newDesc) {
  try {
    const renamed = newName && newName !== origName;

    if (renamed) {
      await deleteJSON(`/api/channels/${encodeURIComponent(origName)}`);
      await postJSON('/api/channels', {
        channel_name: newName,
        address: newAddr || '',
        description: newDesc || '',
      });
      showToast(`Renamed "${origName}" → "${newName}"`, 'success');
    } else {
      /** @type {Record<string, any>} */
      const body = {};
      if (newAddr !== undefined) body.address = newAddr;
      if (newDesc !== undefined) body.description = newDesc;
      await putJSON(`/api/channels/${encodeURIComponent(origName)}`, body);
      showToast(`Updated "${origName}"`, 'success');
    }

    editingRow = null;
    await loadAll();
    if (renamed) refreshStatsBadges();
  } catch (e) {
    showToast(`Failed to update: ${messageOf(e)}`, 'error');
  }
}

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
    await loadAll();
    refreshStatsBadges();
  } catch (e) {
    showToast(`Failed to add channel: ${messageOf(e)}`, 'error');
  }
}

/**
 * @param {string} channelName
 */
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
    await loadAll();
    refreshStatsBadges();
  } catch (e) {
    showToast(`Failed to delete: ${messageOf(e)}`, 'error');
  }
}
