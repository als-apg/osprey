/**
 * OSPREY Channel Finder — Middle Layer Explore (Three-Column Drill-Down)
 *
 * System -> Family -> Fields/Channels in a fixed three-column layout.
 * Inline CRUD: add/delete families, delete channels.
 */

import { fetchJSON, postJSON, deleteJSON } from './api.js';
import { showToast } from './app.js';
import { esc } from './utils.js';
import { formModal, confirmModal } from './modal.js';
import { refreshStatsBadges } from './stats-badges.js';

let containerEl = null;
let selectedSystem = null;
let selectedFamily = null;
let showDescriptions = false;

/** Toggle full description visibility and re-render loaded panels. */
export function setShowDescriptions(val) {
  showDescriptions = val;
  // Re-render descriptions in place via CSS class toggle
  document.querySelectorAll('.column-item-desc').forEach(el => {
    el.classList.toggle('column-item-desc-full', val);
  });
}

export async function mountMiddleLayer(container) {
  containerEl = container;

  container.innerHTML = `
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
        <div class="column-add-btn" id="ml-add-family" style="display: none;">+ Add Family</div>
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
  selectedFamily = null;

  document.getElementById('ml-add-family')?.addEventListener('click', handleAddFamily);

  await loadSystems();
}

export function unmountMiddleLayer() {
  containerEl = null;
  selectedSystem = null;
  selectedFamily = null;
}

async function loadSystems() {
  const body = document.getElementById('ml-systems-body');
  if (!body) return;

  try {
    const data = await fetchJSON('/api/explore/systems');
    const systems = data.systems || [];

    body.innerHTML = systems.map(sys => {
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
      item.addEventListener('click', () => selectSystem(item.dataset.system));
    });
  } catch (e) {
    body.innerHTML = `<div class="empty-state">Failed: ${e.message}</div>`;
  }
}

async function selectSystem(system) {
  selectedSystem = system;
  selectedFamily = null;

  // Highlight
  document.querySelectorAll('#ml-systems-body .column-item').forEach(el => {
    el.classList.toggle('selected', el.dataset.system === system);
  });

  // Show add family button
  const addBtn = document.getElementById('ml-add-family');
  if (addBtn) addBtn.style.display = '';

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

    body.innerHTML = families.map(fam => {
      const name = typeof fam === 'string' ? fam : (fam.name || fam.family || '');
      const desc = typeof fam === 'object' ? (fam.description || '') : '';
      const fullCls = showDescriptions ? ' column-item-desc-full' : '';
      return `
        <div class="column-item" data-family="${esc(name)}">
          <div class="column-item-name">${esc(name)}</div>
          ${desc ? `<div class="column-item-desc${fullCls}">${esc(desc)}</div>` : ''}
          <span class="item-actions">
            <button class="item-action-btn action-delete" data-family="${esc(name)}" title="Delete">&times;</button>
          </span>
        </div>
      `;
    }).join('') || '<div class="empty-state">No families</div>';

    body.querySelectorAll('.column-item').forEach(item => {
      item.addEventListener('click', (e) => {
        if (e.target.closest('.item-action-btn')) return;
        selectFamily(item.dataset.family);
      });
    });

    body.querySelectorAll('.item-action-btn.action-delete').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        handleDeleteFamily(btn.dataset.family);
      });
    });
  } catch (e) {
    body.innerHTML = `<div class="empty-state">Failed: ${e.message}</div>`;
  }
}

async function selectFamily(family) {
  selectedFamily = family;

  // Highlight
  document.querySelectorAll('#ml-families-body .column-item').forEach(el => {
    el.classList.toggle('selected', el.dataset.family === family);
  });

  // Load fields then channels
  const body = document.getElementById('ml-channels-body');
  if (!body) return;
  body.innerHTML = '<div class="loading-center"><div class="loading-spinner"></div></div>';

  try {
    // First get fields to know what's available
    const fieldsData = await fetchJSON(
      `/api/explore/fields?system=${encodeURIComponent(selectedSystem)}&family=${encodeURIComponent(family)}`
    );
    const fields = fieldsData.fields || {};
    const fieldNames = Object.keys(fields);

    if (fieldNames.length === 0) {
      body.innerHTML = '<div class="empty-state">No fields found</div>';
      return;
    }

    // Load channels for each field (collapsed by default)
    let html = '';
    for (const field of fieldNames) {
      try {
        const chData = await fetchJSON(
          `/api/explore/channels?system=${encodeURIComponent(selectedSystem)}&family=${encodeURIComponent(family)}&field=${encodeURIComponent(field)}`
        );
        const channels = chData.channels || [];

        const channelItems = channels.slice(0, 50).map(ch => {
          const name = typeof ch === 'string' ? ch : (ch.name || ch.channel || '');
          return `<div class="column-item" style="padding: var(--space-1) var(--space-3); border-left: none;">
            <span class="pv-name" style="font-size: var(--text-sm);">${esc(name)}</span>
            <span class="item-actions">
              <button class="item-action-btn action-delete" data-field="${esc(field)}" data-channel="${esc(name)}" title="Delete channel">&times;</button>
            </span>
          </div>`;
        }).join('');

        const overflow = channels.length > 50
          ? `<div style="padding: var(--space-1) var(--space-3); color: var(--text-muted); font-size: var(--text-xs);">... and ${channels.length - 50} more</div>`
          : '';

        html += `
          <div class="field-group" style="margin-bottom: var(--space-3);">
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
      } catch {
        html += `<div class="empty-state">Failed to load ${field}</div>`;
      }
    }

    body.innerHTML = html || '<div class="empty-state">No channels</div>';

    // Wire up collapsible field group headers
    body.querySelectorAll('.field-group-header').forEach(header => {
      header.addEventListener('click', () => {
        header.parentElement.classList.toggle('open');
      });
    });

    // Wire up channel delete buttons
    body.querySelectorAll('.item-action-btn.action-delete').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        handleDeleteChannel(btn.dataset.field, btn.dataset.channel);
      });
    });
  } catch (e) {
    body.innerHTML = `<div class="empty-state">Failed: ${e.message}</div>`;
  }
}

// ---- CRUD Handlers ----

async function handleAddFamily() {
  if (!selectedSystem) return;

  const result = await formModal({
    title: `Add Family to ${selectedSystem}`,
    fields: [
      { name: 'family', label: 'Family Name', required: true, placeholder: 'e.g., BPM' },
      { name: 'description', label: 'Description', placeholder: 'Optional description' },
    ],
  });

  if (!result) return;

  try {
    await postJSON('/api/structure/family', {
      system: selectedSystem,
      family: result.family,
      description: result.description,
    });
    showToast(`Added family "${result.family}"`, 'success');
    await loadFamilies();
    refreshStatsBadges();
  } catch (e) {
    showToast(`Failed to add family: ${e.message}`, 'error');
  }
}

async function handleDeleteFamily(family) {
  if (!selectedSystem) return;

  // Get impact
  let impactText = '';
  try {
    const impact = await postJSON('/api/structure/impact', {
      system: selectedSystem,
      family,
    });
    if (impact.affected_channels > 0) {
      impactText = `This will remove ${impact.affected_channels} channel${impact.affected_channels !== 1 ? 's' : ''}.`;
    }
  } catch { /* ignore */ }

  const confirmed = await confirmModal({
    title: `Delete "${family}"?`,
    message: `Remove family "${family}" and all its channels from ${selectedSystem}.`,
    impact: impactText,
    confirmLabel: 'Delete',
    danger: true,
  });

  if (!confirmed) return;

  try {
    await deleteJSON('/api/structure/family', {
      system: selectedSystem,
      family,
    });
    showToast(`Deleted family "${family}"`, 'success');
    if (selectedFamily === family) {
      selectedFamily = null;
      const chBody = document.getElementById('ml-channels-body');
      if (chBody) chBody.innerHTML = '<div class="empty-state">Select a family</div>';
    }
    await loadFamilies();
    refreshStatsBadges();
  } catch (e) {
    showToast(`Failed to delete family: ${e.message}`, 'error');
  }
}

async function handleDeleteChannel(field, channelName) {
  if (!selectedSystem || !selectedFamily) return;

  const confirmed = await confirmModal({
    title: `Delete channel?`,
    message: `Remove "${channelName}" from ${selectedSystem}:${selectedFamily}:${field}?`,
    confirmLabel: 'Delete',
    danger: true,
  });

  if (!confirmed) return;

  try {
    await deleteJSON('/api/structure/channel', {
      system: selectedSystem,
      family: selectedFamily,
      field,
      channel_name: channelName,
    });
    showToast(`Deleted "${channelName}"`, 'success');
    await selectFamily(selectedFamily);
    refreshStatsBadges();
  } catch (e) {
    showToast(`Failed to delete channel: ${e.message}`, 'error');
  }
}
