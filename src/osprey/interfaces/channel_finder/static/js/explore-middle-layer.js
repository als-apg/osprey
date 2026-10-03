// @ts-check
/**
 * OSPREY Channel Finder — Middle Layer Explore (Three-Column Drill-Down)
 *
 * System -> Family -> Fields/Channels in a fixed three-column layout.
 * Read-only: corrections go into data/facility/fixes.yaml and take effect with
 * `osprey build`.
 */

import { fetchJSON } from './api.js';
import { esc, messageOf } from './utils.js';
import { groupFieldChannels } from './explore-grouping.js';

/** @type {string|null} */
let selectedSystem = null;
let showDescriptions = false;
/** @type {any} */
let currentDeviceInfo = null;  // device arrangement info for current family
/** @type {Set<string>} */
let activeSectors = new Set();   // selected sector filter chips
/** @type {Set<string>} */
let activeDevices = new Set();   // selected device filter chips
/** @type {string[]} */
let cachedFieldNames = [];       // save fieldNames for re-render on chip toggle
/** @type {string[]} */
let allSectorValues = [];        // all sector strings for "all" action
/** @type {string[]} */
let allDeviceValues = [];        // all device strings for "all" action

/**
 * Toggle full description visibility and re-render loaded panels.
 * @param {boolean} val
 */
export function setShowDescriptions(val) {
  showDescriptions = val;
  // Re-render descriptions in place via CSS class toggle
  document.querySelectorAll('.column-item-desc').forEach(el => {
    el.classList.toggle('column-item-desc-full', val);
  });
}

/**
 * @param {HTMLElement} container
 */
export async function mountMiddleLayer(container) {
  container.innerHTML = `
    <div class="cf-corrections-info">
      Corrections go in <code>data/facility/fixes.yaml</code> and take effect with <code>osprey build</code>.
    </div>
    <div class="three-column-layout">
      <div class="column-panel" id="ml-systems">
        <div class="column-panel-header">Systems</div>
        <div class="column-panel-body" id="ml-systems-body">
          <div class="loading-center"><div class="loading-spinner"></div></div>
        </div>
      </div>
      <div class="column-panel" id="ml-families">
        <div class="column-panel-header">Families</div>
        <div class="column-panel-body" id="ml-families-body">
          <div class="empty-state">Select a system</div>
        </div>
      </div>
      <div class="column-panel" id="ml-channels">
        <div class="column-panel-header">Channels</div>
        <div class="column-panel-body" id="ml-channels-body">
          <div class="empty-state">Select a family</div>
        </div>
      </div>
    </div>
  `;

  selectedSystem = null;

  await loadSystems();
}

export function unmountMiddleLayer() {
  selectedSystem = null;
  currentDeviceInfo = null;
  activeSectors = new Set();
  activeDevices = new Set();
  cachedFieldNames = [];
  allSectorValues = [];
  allDeviceValues = [];
}

async function loadSystems() {
  const body = document.getElementById('ml-systems-body');
  if (!body) return;

  try {
    const data = await fetchJSON('/api/explore/systems');
    const systems = data.systems || [];

    body.innerHTML = systems.map((/** @type {any} */ sys) => {
      const name = typeof sys === 'string' ? sys : (sys.name || sys.system || '');
      const desc = typeof sys === 'object' ? (sys.description || '') : '';
      const fullCls = showDescriptions ? ' column-item-desc-full' : '';
      return `
        <div class="column-item" data-system="${esc(name)}">
          <div class="column-item-name">${esc(name)}</div>
          ${desc ? `<div class="column-item-desc${fullCls}">${esc(desc)}</div>` : ''}
        </div>
      `;
    }).join('') || '<div class="empty-state">No systems found</div>';

    body.querySelectorAll('.column-item').forEach(item => {
      item.addEventListener('click', () => selectSystem(/** @type {HTMLElement} */ (item).dataset.system ?? null));
    });
  } catch (e) {
    body.innerHTML = `<div class="empty-state">Failed: ${esc(messageOf(e))}</div>`;
  }
}

/**
 * @param {string|null} system
 */
async function selectSystem(system) {
  selectedSystem = system;

  // Highlight
  document.querySelectorAll('#ml-systems-body .column-item').forEach(el => {
    el.classList.toggle('selected', /** @type {HTMLElement} */ (el).dataset.system === system);
  });

  // Reset channels column
  const chBody = document.getElementById('ml-channels-body');
  if (chBody) chBody.innerHTML = '<div class="empty-state">Select a family</div>';

  // Load families
  await loadFamilies();
}

async function loadFamilies() {
  const body = document.getElementById('ml-families-body');
  if (!body || !selectedSystem) return;
  body.innerHTML = '<div class="loading-center"><div class="loading-spinner"></div></div>';

  try {
    const data = await fetchJSON(`/api/explore/families?system=${encodeURIComponent(selectedSystem)}`);
    const families = data.families || [];

    body.innerHTML = families.map((/** @type {any} */ fam) => {
      const name = typeof fam === 'string' ? fam : (fam.name || fam.family || '');
      const desc = typeof fam === 'object' ? (fam.description || '') : '';
      const fullCls = showDescriptions ? ' column-item-desc-full' : '';
      return `
        <div class="column-item" data-family="${esc(name)}">
          <div class="column-item-name">${esc(name)}</div>
          ${desc ? `<div class="column-item-desc${fullCls}">${esc(desc)}</div>` : ''}
        </div>
      `;
    }).join('') || '<div class="empty-state">No families</div>';

    body.querySelectorAll('.column-item').forEach(item => {
      item.addEventListener('click', () => {
        selectFamily(/** @type {HTMLElement} */ (item).dataset.family ?? null);
      });
    });
  } catch (e) {
    body.innerHTML = `<div class="empty-state">Failed: ${esc(messageOf(e))}</div>`;
  }
}

// Cache of per-field channel arrays for client-side sector filtering.
/** @type {Record<string, any[]|null>} */
let cachedFieldChannels = {};

/**
 * @param {string|null} family
 */
async function selectFamily(family) {
  if (!selectedSystem || !family) return;

  // Highlight
  document.querySelectorAll('#ml-families-body .column-item').forEach(el => {
    el.classList.toggle('selected', /** @type {HTMLElement} */ (el).dataset.family === family);
  });

  // Load fields then channels
  const body = document.getElementById('ml-channels-body');
  if (!body) return;
  body.innerHTML = '<div class="loading-center"><div class="loading-spinner"></div></div>';

  try {
    // Fetch fields and device info in parallel
    const [fieldsData, deviceInfo] = await Promise.all([
      fetchJSON(`/api/explore/fields?system=${encodeURIComponent(selectedSystem)}&family=${encodeURIComponent(family)}`),
      fetchJSON(`/api/explore/device-info?system=${encodeURIComponent(selectedSystem)}&family=${encodeURIComponent(family)}`).catch(() => null)
    ]);
    currentDeviceInfo = deviceInfo;
    activeSectors = new Set();
    activeDevices = new Set();

    const fields = fieldsData.fields || {};
    const fieldNames = Object.keys(fields);

    if (fieldNames.length === 0) {
      body.innerHTML = '<div class="empty-state">No fields found</div>';
      return;
    }

    // Load channels for each field
    cachedFieldChannels = {};
    for (const field of fieldNames) {
      try {
        const chData = await fetchJSON(
          `/api/explore/channels?system=${encodeURIComponent(selectedSystem)}&family=${encodeURIComponent(family)}&field=${encodeURIComponent(field)}`
        );
        cachedFieldChannels[field] = chData.channels || [];
      } catch {
        cachedFieldChannels[field] = null; // mark as failed
      }
    }

    cachedFieldNames = fieldNames;
    renderChannelsPanel(body, fieldNames);
  } catch (e) {
    body.innerHTML = `<div class="empty-state">Failed: ${esc(messageOf(e))}</div>`;
  }
}

/**
 * Render the channels panel from cached data.
 * Reads activeSectors / activeDevices module state for filtering; sector
 * grouping/ordering/truncation is delegated to the pure groupFieldChannels().
 * @param {HTMLElement} body - The panel body element.
 * @param {string[]} fieldNames - Ordered field names to render.
 */
function renderChannelsPanel(body, fieldNames) {
  const deviceInfo = currentDeviceInfo;
  const hasDeviceList = deviceInfo && Array.isArray(deviceInfo.device_list) && deviceInfo.device_list.length > 0;

  let html = '';

  // Filter chip bar
  if (hasDeviceList) {
    const sectorChips = deviceInfo.sectors.map((/** @type {any} */ s) => {
      const sv = String(s);
      const cls = activeSectors.has(sv) ? ' active' : '';
      return `<span class="filter-chip${cls}" data-filter-type="sector" data-value="${esc(sv)}">${esc(sv)}</span>`;
    }).join('');

    const uniqueDevices = [...new Set(deviceInfo.device_list.map((/** @type {any} */ e) => e[1]))]
      .sort((/** @type {any} */ a, /** @type {any} */ b) => a - b);
    const deviceChips = uniqueDevices.map((/** @type {any} */ d) => {
      const dv = String(d);
      const cls = activeDevices.has(dv) ? ' active' : '';
      return `<span class="filter-chip${cls}" data-filter-type="device" data-value="${esc(dv)}">${esc(dv)}</span>`;
    }).join('');

    allSectorValues = deviceInfo.sectors.map(String);
    allDeviceValues = uniqueDevices.map(String);

    html += `
      <div class="device-filter-bar">
        <div class="filter-chip-row">
          <span class="filter-chip-label">Sectors</span>
          <span class="filter-chip-action" data-action="all" data-filter-type="sector">all</span>
          <span class="filter-chip-action-sep">/</span>
          <span class="filter-chip-action" data-action="none" data-filter-type="sector">none</span>
          ${sectorChips}
        </div>
        <div class="filter-chip-row">
          <span class="filter-chip-label">Devices</span>
          <span class="filter-chip-action" data-action="all" data-filter-type="device">all</span>
          <span class="filter-chip-action-sep">/</span>
          <span class="filter-chip-action" data-action="none" data-filter-type="device">none</span>
          ${deviceChips}
        </div>
      </div>
    `;
  }

  // Field groups
  for (const field of fieldNames) {
    const channels = cachedFieldChannels[field];
    if (channels === null) {
      html += `<div class="empty-state">Failed to load ${esc(field)}</div>`;
      continue;
    }

    if (hasDeviceList) {
      // Positional device alignment + sector grouping/ordering/truncation (pure).
      const { sectors, visibleCount } = groupFieldChannels(
        channels, deviceInfo.device_list, deviceInfo.common_names, activeSectors, activeDevices
      );

      let innerHtml = '';
      for (const sec of sectors) {
        innerHtml += `<div class="sector-group-header">${esc(sec.label)} (${sec.total})</div>`;

        innerHtml += sec.shown.map(item =>
          `<div class="column-item" style="padding: var(--cf-space-1) var(--cf-space-3); border-left: none;">
            <span class="pv-name" style="font-size: var(--cf-text-sm);">${esc(item.name)}</span>${
              item.commonName ? `<span class="common-name">${esc(item.commonName)}</span>` : ''
            }
          </div>`
        ).join('');

        if (sec.hidden > 0) {
          innerHtml += `<div style="padding: var(--cf-space-1) var(--cf-space-3); color: var(--text-muted); font-size: var(--cf-text-xs);">... and ${sec.hidden} more</div>`;
        }
      }

      html += `
        <div class="field-group" style="margin-bottom: var(--cf-space-3);">
          <div class="field-group-header" data-field="${esc(field)}">
            <span class="field-group-chevron">&#9654;</span>
            ${esc(field)} <span style="color: var(--text-muted);">(${visibleCount})</span>
          </div>
          <div class="field-group-body">
            ${innerHtml}
          </div>
        </div>
      `;
    } else {
      // Fallback: flat display (no device info)
      const channelItems = channels.slice(0, 50).map((/** @type {any} */ ch) => {
        const name = typeof ch === 'string' ? ch : (ch.name || ch.channel || '');
        return `<div class="column-item" style="padding: var(--cf-space-1) var(--cf-space-3); border-left: none;">
          <span class="pv-name" style="font-size: var(--cf-text-sm);">${esc(name)}</span>
        </div>`;
      }).join('');

      const overflow = channels.length > 50
        ? `<div style="padding: var(--cf-space-1) var(--cf-space-3); color: var(--text-muted); font-size: var(--cf-text-xs);">... and ${channels.length - 50} more</div>`
        : '';

      html += `
        <div class="field-group" style="margin-bottom: var(--cf-space-3);">
          <div class="field-group-header" data-field="${esc(field)}">
            <span class="field-group-chevron">&#9654;</span>
            ${esc(field)} <span style="color: var(--text-muted);">(${channels.length})</span>
          </div>
          <div class="field-group-body">
            ${channelItems}
            ${overflow}
          </div>
        </div>
      `;
    }
  }

  // Snapshot which field groups are open before replacing DOM
  /** @type {Set<string>} */
  const openFields = new Set();
  body.querySelectorAll('.field-group.open .field-group-header').forEach(h => {
    const f = /** @type {HTMLElement} */ (h).dataset.field;
    if (f) openFields.add(f);
  });

  body.innerHTML = html || '<div class="empty-state">No channels</div>';

  // Restore open state and wire up collapsible field group headers
  body.querySelectorAll('.field-group-header').forEach(header => {
    const f = /** @type {HTMLElement} */ (header).dataset.field;
    if (f && openFields.has(f)) {
      header.parentElement?.classList.add('open');
    }
    header.addEventListener('click', () => {
      header.parentElement?.classList.toggle('open');
    });
  });

  // Wire up filter chip toggles
  body.querySelectorAll('.filter-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const el = /** @type {HTMLElement} */ (chip);
      const type = el.dataset.filterType;
      const val = el.dataset.value;
      if (!val) return;
      const set = type === 'sector' ? activeSectors : activeDevices;
      if (set.has(val)) set.delete(val); else set.add(val);
      renderChannelsPanel(body, cachedFieldNames);
    });
  });

  // Wire up all/none actions
  body.querySelectorAll('.filter-chip-action').forEach(btn => {
    btn.addEventListener('click', () => {
      const el = /** @type {HTMLElement} */ (btn);
      const type = el.dataset.filterType;
      const action = el.dataset.action;
      const set = type === 'sector' ? activeSectors : activeDevices;
      const vals = type === 'sector' ? allSectorValues : allDeviceValues;
      if (action === 'all') { vals.forEach(v => set.add(v)); } else { set.clear(); }
      renderChannelsPanel(body, cachedFieldNames);
    });
  });
}
