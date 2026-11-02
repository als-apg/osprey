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
 * Read-only: corrections go into data/facility/fixes.yaml and take effect with
 * `osprey build`.
 */

import { fetchJSON } from './api.js';
import { esc, messageOf } from './utils.js';
import { filterChannels, totalChunksFor, clampChunkIdx, pageSlice } from './chunk-filter.js';

/** @type {any[]} */
let allChannels = []; // the ENTIRE in-context database (loaded once)
let filterText = '';
let chunkIdx = 0;
const CHUNK_SIZE = 50;

// Single source of truth: every render derives the filtered set from here, so
// the table page-slice and the pager can never operate on different scopes.
function getFiltered() {
  return filterChannels(allChannels, filterText);
}

/**
 * @param {HTMLElement} container
 */
export async function mountInContext(container) {
  container.innerHTML = `
    <div class="cf-corrections-info">
      Corrections go in <code>data/facility/fixes.yaml</code> and take effect with <code>osprey build</code>.
    </div>
    <div class="filter-bar">
      <span class="filter-label">Filter:</span>
      <input type="text" class="filter-input" id="ic-filter"
             placeholder="Type to filter by name or description...">
      <span class="filter-label" id="ic-count"></span>
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

  await loadAll();
}

export function unmountInContext() {
  allChannels = [];
  filterText = '';
  chunkIdx = 0;
}

async function loadAll() {
  try {
    // Omitting chunk_idx returns the entire in-context DB: {channels, total}.
    const data = await fetchJSON('/api/channels');
    allChannels = data.channels || [];
    // Keep the current page valid for the loaded (filtered) set.
    chunkIdx = clampChunkIdx(chunkIdx, getFiltered().length, CHUNK_SIZE);

    renderTable();
    renderPagination();
  } catch (e) {
    const area = document.getElementById('ic-table-area');
    if (area)
      area.innerHTML = `<div class="empty-state">Failed to load channels: ${esc(messageOf(e))}</div>`;
  }
}

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
          </tr>
        </thead>
        <tbody>
          ${pageItems
            .map((ch, i) => {
              const name = ch.name || ch.channel_name || ch.channel || '—';
              const addr = ch.address || ch.pv_address || '';
              const desc = ch.description || '';
              return `
              <tr>
                <td>${start + i + 1}</td>
                <td class="pv-cell">${esc(name)}</td>
                <td class="pv-cell">${esc(addr)}</td>
                <td>${esc(desc)}</td>
              </tr>
            `;
            })
            .join('')}
        </tbody>
      </table>
    </div>
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
    if (chunkIdx > 0) {
      chunkIdx -= 1;
      renderTable();
      renderPagination();
    }
  });
  document.getElementById('ic-next')?.addEventListener('click', () => {
    if (chunkIdx < totalChunks - 1) {
      chunkIdx += 1;
      renderTable();
      renderPagination();
    }
  });
}
