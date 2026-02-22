/**
 * OSPREY Channel Finder — Hierarchical Explore (Miller Columns)
 *
 * Progressive drill-down through hierarchy levels with multi-select
 * support on instance levels and "Build Channels" action.
 * Inline CRUD: add, rename, delete nodes at each level.
 */

import { fetchJSON, postJSON, putJSON, deleteJSON } from './api.js';
import { showToast } from './app.js';
import { esc } from './utils.js';
import { formModal, confirmModal } from './modal.js';
import { refreshStatsBadges } from './stats-badges.js';

let containerEl = null;
let hierInfo = null;
let selections = {};  // level -> value(s)
let columns = [];     // array of { level, options, selectedValues }
let showDescriptions = false;

/** Return true if the named level is a tree-type (editable) level. */
function isTreeLevel(levelName) {
  const levels = hierInfo?.hierarchy_config?.levels;
  if (!levels || !levels[levelName]) return true;  // default to tree
  return levels[levelName].type === 'tree';
}

/** Toggle description visibility and re-render (called from explore.js). */
export function setShowDescriptions(val) {
  showDescriptions = val;
  renderColumns();
}

export async function mountHierarchical(container) {
  containerEl = container;

  container.innerHTML = `
    <div class="miller-container" id="miller-container">
      <div class="loading-center"><div class="loading-spinner"></div> Loading hierarchy...</div>
    </div>
  `;

  try {
    hierInfo = await fetchJSON('/api/explore/hierarchy-info');
    selections = {};
    columns = [];
    await loadLevel(0);
  } catch (e) {
    container.innerHTML = `<div class="empty-state">Failed to load hierarchy: ${e.message}</div>`;
  }

}

export function unmountHierarchical() {
  containerEl = null;
  hierInfo = null;
  selections = {};
  columns = [];
}

async function loadLevel(levelIdx) {
  if (!hierInfo?.hierarchy_levels) return;
  const levels = hierInfo.hierarchy_levels;
  if (levelIdx >= levels.length) return;

  const level = typeof levels[levelIdx] === 'string'
    ? levels[levelIdx]
    : (levels[levelIdx].name || levels[levelIdx].level);

  // Build selections dict for API call
  const apiSelections = {};
  for (let i = 0; i < levelIdx; i++) {
    const prevLevel = typeof levels[i] === 'string' ? levels[i] : (levels[i].name || levels[i].level);
    if (selections[prevLevel] !== undefined) {
      apiSelections[prevLevel] = selections[prevLevel];
    }
  }

  try {
    const selParam = Object.keys(apiSelections).length > 0
      ? `&selections=${encodeURIComponent(JSON.stringify(apiSelections))}`
      : '';
    const data = await fetchJSON(`/api/explore/options?level=${encodeURIComponent(level)}${selParam}`);

    // Trim columns beyond current level
    columns = columns.slice(0, levelIdx);

    // For instance-type levels, eagerly fetch expansion config
    let expansion = null;
    if (!isTreeLevel(level)) {
      try {
        const expSelParam = Object.keys(apiSelections).length > 0
          ? `?selections=${encodeURIComponent(JSON.stringify(apiSelections))}&level=${encodeURIComponent(level)}`
          : `?level=${encodeURIComponent(level)}`;
        const expData = await fetchJSON(`/api/tree/expansion${expSelParam}`);
        expansion = expData.expansion || null;
      } catch { /* expansion info unavailable */ }
    }

    columns.push({
      level,
      options: data.options || [],
      selectedValues: new Set(),
      expansion,
    });

    renderColumns();
  } catch (e) {
    showToast(`Failed to load ${level}: ${e.message}`, 'error');
  }
}

function renderColumns() {
  const mc = document.getElementById('miller-container');
  if (!mc) return;

  mc.innerHTML = columns.map((col, colIdx) => {
    const treeLevel = isTreeLevel(col.level);
    const items = (col.options || []).map(opt => {
      const name = typeof opt === 'string' ? opt : (opt.name || opt.label || opt.value || '');
      const count = (typeof opt === 'object' && opt.count != null) ? opt.count : null;
      const desc = (typeof opt === 'object' && opt.description) ? opt.description : '';
      const isSelected = col.selectedValues.has(name);

      const actionBtns = treeLevel ? `
          <span class="item-actions">
            <button class="item-action-btn action-edit" data-col="${colIdx}" data-name="${esc(name)}" title="Edit">&#9998;</button>
            <button class="item-action-btn action-delete" data-col="${colIdx}" data-name="${esc(name)}" title="Delete">&times;</button>
          </span>` : '';

      const descHtml = desc
        ? (showDescriptions
            ? `<div class="item-desc item-desc-full">${esc(desc)}</div>`
            : `<div class="item-desc">${esc(desc)}</div>`)
        : '';

      return `
        <div class="miller-item${isSelected ? ' selected' : ''}"
             data-col="${colIdx}" data-value="${esc(name)}"
             data-desc="${esc(desc)}">
          <div class="item-name-group">
            <span class="item-label">${esc(name)}</span>
            ${descHtml}
          </div>
          <span>${count != null ? `<span class="item-count">${count}</span>` : ''}</span>
          ${actionBtns}
        </div>
      `;
    }).join('');

    const footerBtn = treeLevel
      ? `<div class="column-add-btn" data-col="${colIdx}">+ Add ${esc(col.level)}</div>`
      : `<div class="column-add-btn column-edit-expansion-btn" data-col="${colIdx}">&#9881; Edit range</div>`;

    return `
      <div class="miller-column">
        <div class="miller-column-header">${esc(col.level)}</div>
        <div class="miller-column-body">${items || '<div class="empty-state">No options</div>'}</div>
        ${footerBtn}
      </div>
    `;
  }).join('');

  // Attach click handlers for item selection
  mc.querySelectorAll('.miller-item').forEach(item => {
    item.addEventListener('click', (e) => {
      // Don't select when clicking action buttons
      if (e.target.closest('.item-action-btn')) return;
      const colIdx = parseInt(item.dataset.col, 10);
      const value = item.dataset.value;
      handleSelect(colIdx, value);
    });
  });

  // Attach CRUD handlers
  mc.querySelectorAll('.item-action-btn.action-edit').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      handleEdit(parseInt(btn.dataset.col, 10), btn.dataset.name);
    });
  });

  mc.querySelectorAll('.item-action-btn.action-delete').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      handleDelete(parseInt(btn.dataset.col, 10), btn.dataset.name);
    });
  });

  mc.querySelectorAll('.column-add-btn:not(.column-edit-expansion-btn)').forEach(btn => {
    btn.addEventListener('click', () => {
      handleAdd(parseInt(btn.dataset.col, 10));
    });
  });

  mc.querySelectorAll('.column-edit-expansion-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      handleEditExpansion(parseInt(btn.dataset.col, 10));
    });
  });

}

// ---- CRUD Handlers ----

async function handleAdd(colIdx) {
  const col = columns[colIdx];
  if (!col) return;

  const result = await formModal({
    title: `Add ${col.level}`,
    fields: [
      { name: 'name', label: 'Name', required: true, placeholder: `New ${col.level} name` },
      { name: 'description', label: 'Description', type: 'textarea', placeholder: 'Optional description' },
    ],
  });

  if (!result) return;

  // Build parent_selections — include instance levels so backend can navigate
  const parentSelections = {};
  for (let i = 0; i < colIdx; i++) {
    const lvl = columns[i].level;
    if (selections[lvl] !== undefined) {
      const val = selections[lvl];
      parentSelections[lvl] = Array.isArray(val) ? val[0] : val;
    }
  }

  try {
    await postJSON('/api/tree/node', {
      level: col.level,
      parent_selections: parentSelections,
      name: result.name,
      description: result.description,
    });
    showToast(`Added "${result.name}"`, 'success');
    await loadLevel(colIdx);
    refreshStatsBadges();
  } catch (e) {
    showToast(`Failed to add: ${e.message}`, 'error');
  }
}

async function handleEdit(colIdx, name) {
  const col = columns[colIdx];
  if (!col) return;

  // Find current description from the DOM data attribute
  const itemEl = document.querySelector(
    `.miller-item[data-col="${colIdx}"][data-value="${CSS.escape(name)}"]`
  );
  const currentDesc = itemEl?.dataset.desc || '';

  const result = await formModal({
    title: `Edit ${col.level}`,
    fields: [
      { name: 'new_name', label: 'Name', required: true, value: name },
      { name: 'description', label: 'Description', type: 'textarea', value: currentDesc, placeholder: 'Optional description' },
    ],
    submitLabel: 'Save',
  });

  if (!result) return;

  const nameChanged = result.new_name !== name;
  const descChanged = result.description !== currentDesc;
  if (!nameChanged && !descChanged) return;

  // Build selections for parent path
  const parentSelections = {};
  for (let i = 0; i < colIdx; i++) {
    const lvl = columns[i].level;
    if (selections[lvl] !== undefined) {
      const val = selections[lvl];
      parentSelections[lvl] = Array.isArray(val) ? val[0] : val;
    }
  }

  try {
    await putJSON('/api/tree/node', {
      level: col.level,
      selections: parentSelections,
      old_name: name,
      new_name: nameChanged ? result.new_name : null,
      description: descChanged ? result.description : null,
    });
    const msg = nameChanged
      ? `Renamed "${name}" to "${result.new_name}"`
      : `Updated "${name}"`;
    showToast(msg, 'success');
    await loadLevel(colIdx);
    refreshStatsBadges();
  } catch (e) {
    showToast(`Failed to edit: ${e.message}`, 'error');
  }
}

async function handleDelete(colIdx, name) {
  const col = columns[colIdx];
  if (!col) return;

  // Build selections for parent path
  const parentSelections = {};
  for (let i = 0; i < colIdx; i++) {
    const lvl = columns[i].level;
    if (selections[lvl] !== undefined) {
      const val = selections[lvl];
      parentSelections[lvl] = Array.isArray(val) ? val[0] : val;
    }
  }

  // Get impact count
  let impactText = '';
  try {
    const impact = await postJSON('/api/tree/impact', {
      level: col.level,
      selections: parentSelections,
      name,
    });
    if (impact.affected_channels > 0) {
      impactText = `This will affect ${impact.affected_channels} channel${impact.affected_channels !== 1 ? 's' : ''}.`;
    }
  } catch { /* ignore impact errors */ }

  const confirmed = await confirmModal({
    title: `Delete "${name}"?`,
    message: `Remove "${name}" and all its descendants from the database.`,
    impact: impactText,
    confirmLabel: 'Delete',
    danger: true,
  });

  if (!confirmed) return;

  try {
    await deleteJSON('/api/tree/node', {
      level: col.level,
      selections: parentSelections,
      name,
    });
    showToast(`Deleted "${name}"`, 'success');
    // Remove from selections if selected
    if (selections[col.level] === name) {
      delete selections[col.level];
    }
    await loadLevel(colIdx);
    refreshStatsBadges();
  } catch (e) {
    showToast(`Failed to delete: ${e.message}`, 'error');
  }
}

async function handleEditExpansion(colIdx) {
  const col = columns[colIdx];
  if (!col) return;

  // Build parent selections for the API call
  const parentSelections = {};
  for (let i = 0; i < colIdx; i++) {
    const lvl = columns[i].level;
    if (selections[lvl] !== undefined) {
      const val = selections[lvl];
      parentSelections[lvl] = Array.isArray(val) ? val[0] : val;
    }
  }

  // Use expansion config cached at column load time
  const currentExpansion = col.expansion || {};
  const hasRange = currentExpansion.range != null;
  const result = await formModal({
    title: `Edit ${col.level} expansion`,
    fields: [
      { name: 'pattern', label: 'Pattern', value: currentExpansion.pattern || '', placeholder: 'e.g. B{:02d}' },
      { name: 'start', label: 'Range start', value: hasRange ? String(currentExpansion.range[0]) : '', placeholder: '1' },
      { name: 'end', label: 'Range end', value: hasRange ? String(currentExpansion.range[1]) : '', placeholder: '10' },
    ],
    submitLabel: 'Save',
  });

  if (!result) return;

  try {
    await putJSON('/api/tree/expansion', {
      level: col.level,
      selections: parentSelections,
      pattern: result.pattern || null,
      range_start: result.start ? parseInt(result.start, 10) : null,
      range_end: result.end ? parseInt(result.end, 10) : null,
    });
    showToast(`Updated ${col.level} expansion`, 'success');
    await loadLevel(colIdx);
    refreshStatsBadges();
  } catch (e) {
    showToast(`Failed to update expansion: ${e.message}`, 'error');
  }
}

// ---- Selection ----

function handleSelect(colIdx, value) {
  const col = columns[colIdx];
  if (!col) return;

  const isLastLevel = colIdx === columns.length - 1 &&
    hierInfo?.hierarchy_levels?.length === columns.length;

  // Toggle selection
  if (col.selectedValues.has(value)) {
    col.selectedValues.delete(value);
  } else {
    if (!isLastLevel) {
      // Single select for non-terminal levels
      col.selectedValues.clear();
    }
    col.selectedValues.add(value);
  }

  // Update selections dict
  if (col.selectedValues.size === 0) {
    delete selections[col.level];
  } else if (col.selectedValues.size === 1) {
    selections[col.level] = [...col.selectedValues][0];
  } else {
    selections[col.level] = [...col.selectedValues];
  }

  // Remove deeper selections
  for (let i = colIdx + 1; i < columns.length; i++) {
    delete selections[columns[i].level];
  }

  // Re-render current columns
  renderColumns();

  // Load next level (only for single selection on non-terminal)
  if (!isLastLevel && col.selectedValues.size === 1) {
    loadLevel(colIdx + 1);
  }
}

