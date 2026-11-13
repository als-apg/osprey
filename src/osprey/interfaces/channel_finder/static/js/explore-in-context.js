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
import { bridgeHtml, directorySize, reportCurated } from './curated-bridge.js';
import { state } from './state.js';

/** @type {any[]} */
let allChannels = []; // the ENTIRE in-context database (loaded once)
let filterWords = ''; // as typed; the shared query beside All PVs
let filterText = ''; // lower-cased, what the filter matches with
let chunkIdx = 0;
const CHUNK_SIZE = 50;
/** Bumped on every mount and unmount, so a late count lands on no old page. */
let generation = 0;
/** PVs in the whole directory, asked for on the first miss; null until known. */
/** @type {number|null} */
let directoryPvs = null;
let directoryAsked = false;

// Single source of truth: every render derives the filtered set from here, so
// the table page-slice and the pager can never operate on different scopes.
function getFiltered() {
  return filterChannels(allChannels, filterText);
}

/**
 * @param {HTMLElement} container
 */
export async function mountInContext(container) {
  generation += 1;
  directoryPvs = null;
  directoryAsked = false;
  filterWords = state.directory ? state.scopeQuery : '';
  filterText = filterWords.toLowerCase();
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
    <div id="ic-bridge"></div>
  `;

  const input = /** @type {HTMLInputElement|null} */ (document.getElementById('ic-filter'));
  if (input) input.value = filterWords;
  input?.addEventListener('input', () => {
    filterWords = input.value;
    filterText = filterWords.toLowerCase();
    // Reset to the first page of the NEW filtered set, and re-render the pager:
    // the filtered set (and thus the chunk count) changes on every keystroke.
    chunkIdx = 0;
    renderTable();
    renderPagination();
    renderBridge();
  });

  await loadAll();
}

export function unmountInContext() {
  generation += 1;
  allChannels = [];
  filterWords = '';
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
    renderBridge();
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

/**
 * Report the filter as a curated search and redraw the line to All PVs under
 * the results. The first miss asks for the directory's size and redraws the
 * line once it answers.
 * @returns {void}
 */
function renderBridge() {
  const slot = document.getElementById('ic-bridge');
  if (!slot) return;
  const total = getFiltered().length;
  reportCurated(filterWords, total);
  slot.innerHTML = bridgeHtml(filterWords, total, directoryPvs);
  if (total === 0 && state.directory && !directoryAsked) {
    directoryAsked = true;
    const mounted = generation;
    void directorySize().then((size) => {
      if (mounted !== generation) return;
      directoryPvs = size;
      const line = document.getElementById('ic-bridge');
      if (line) line.innerHTML = bridgeHtml(filterWords, getFiltered().length, size);
    });
  }
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
