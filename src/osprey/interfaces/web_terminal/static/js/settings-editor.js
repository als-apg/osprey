// @ts-check
/* OSPREY Web Terminal — settings.json interactive editor
 *
 * Interactive form editor for settings.json: model field, and a three-column
 * drag-and-drop permission editor (ALLOW / ASK / DENY). A sibling of
 * config-renderers.js, which re-exports `renderSettingsJsonEditor` so
 * import sites use one path. The shared `_section`/`_groupPermissions`/
 * `_countHooks` helpers live in the neutral config-render-helpers.js, so
 * importing them creates no circular dependency.
 *
 * @module settings-editor
 */

import { el as _el } from '/design-system/js/dom.js';
import { _section, _groupPermissions, _countHooks } from './config-render-helpers.js';

/** Monotonically increasing ID for drag-and-drop element identification. */
let _nextDragId = 1;

/**
 * @typedef {HTMLElement & {
 *   _modelInput: HTMLInputElement,
 *   _permColumns: Record<string, HTMLElement>,
 *   _settingsEditor: { getData(): string, isDirty(): boolean },
 * }} SettingsEditorContainer
 */

/**
 * Render an interactive form editor for settings.json. The permissions
 * section uses a three-column drag-and-drop layout: entries can be dragged
 * between ALLOW / ASK / DENY, deleted inline, or added via per-column "+".
 *
 * Returns a DOM container with a _settingsEditor property: getData() returns
 * the modified JSON string; isDirty() returns true if any field changed.
 *
 * @param {string} jsonString  - current settings.json content
 * @param {(dirty: boolean) => void} onDirtyChange - called when dirty state changes
 * @returns {SettingsEditorContainer|null}
 */
export function renderSettingsJsonEditor(jsonString, onDirtyChange) {
  let data;
  try {
    data = JSON.parse(jsonString);
  } catch {
    return null;
  }

  const originalJson = jsonString;
  let dirty = false;

  const markDirty = () => {
    dirty = true;
    if (onDirtyChange) onDirtyChange(true);
  };

  const container = /** @type {SettingsEditorContainer} */ (
    _el('div', 'config-structured-view config-editor')
  );

  // ---- Model ----
  {
    const section = _section('Model');
    const fieldRow = _el('div', 'config-edit-field');

    const label = _el('span', 'config-edit-label');
    label.textContent = 'Model';
    fieldRow.appendChild(label);

    const input = document.createElement('input');
    input.type = 'text';
    input.className = 'config-edit-input';
    input.value = data.model || '';
    input.placeholder = 'e.g. anthropic/claude-sonnet';
    input.spellcheck = false;
    input.addEventListener('input', markDirty);
    fieldRow.appendChild(input);

    section.appendChild(fieldRow);
    container.appendChild(section);
    container._modelInput = input;
  }

  // ---- Permissions Editor (three-column drag-and-drop) ----
  {
    const section = _section('Permissions');
    const columns = _el('div', 'config-permissions-columns config-perm-interactive');

    const levelOrder = ['allow', 'ask', 'deny'];
    /** @type {Record<string, HTMLElement>} */
    const colMap = {};

    for (const level of levelOrder) {
      const entries = data.permissions?.[level] || [];
      const col = _interactivePermColumn(level, entries, markDirty, container);
      colMap[level] = col;
      columns.appendChild(col);
    }

    // Drag-and-drop wiring across columns (pre-existing 4th `container` arg
    // dropped: _wireDragAndDrop only ever declared 3 params — no-op fix).
    _wireDragAndDrop(columns, colMap, markDirty);

    section.appendChild(columns);
    container.appendChild(section);
    container._permColumns = colMap;
  }

  // ---- Hooks (read-only) ----
  if (data.hooks) {
    const section = _section('Hooks (read-only)');
    const note = _el('div', 'config-edit-note');
    note.textContent = 'Hooks are managed by OSPREY and cannot be edited here.';
    section.appendChild(note);

    for (const [eventName, hookGroups] of Object.entries(data.hooks)) {
      const eventSection = _el('div', 'config-hook-event');
      const eventHeader = _el('div', 'config-hook-event-header');

      const chevron = _el('span', 'config-hook-chevron');
      chevron.textContent = '\u25B6';
      eventHeader.appendChild(chevron);

      const nameSpan = _el('span', '');
      nameSpan.textContent = eventName;
      eventHeader.appendChild(nameSpan);

      const countSpan = _el('span', 'config-hook-count');
      countSpan.textContent = String(_countHooks(hookGroups));
      eventHeader.appendChild(countSpan);

      eventHeader.addEventListener('click', () => eventSection.classList.toggle('expanded'));
      eventSection.appendChild(eventHeader);

      const eventBody = _el('div', 'config-hook-event-body');
      for (const group of hookGroups) {
        const matcher = group.matcher || '*';
        const matcherEl = _el('div', 'config-hook-matcher');
        const matcherLabel = _el('span', 'config-hook-matcher-label');
        matcherLabel.textContent = matcher;
        matcherEl.appendChild(matcherLabel);

        for (const hook of (group.hooks || [])) {
          const hookEl = _el('div', 'config-hook-entry');
          const cmd = hook.command || '';
          const scriptName = cmd.split('/').pop().replace(/"/g, '').replace(/\.py$/, '');

          const scriptSpan = _el('span', 'config-hook-script');
          scriptSpan.textContent = scriptName;
          hookEl.appendChild(scriptSpan);

          if (hook.timeout) {
            const timeoutSpan = _el('span', 'config-hook-timeout');
            timeoutSpan.textContent = hook.timeout + 's';
            hookEl.appendChild(timeoutSpan);
          }
          matcherEl.appendChild(hookEl);
        }
        eventBody.appendChild(matcherEl);
      }
      eventSection.appendChild(eventBody);
      section.appendChild(eventSection);
    }

    container.appendChild(section);
  }

  // ---- API ----
  container._settingsEditor = {
    getData() {
      const result = JSON.parse(originalJson);

      // Update model
      const modelVal = container._modelInput.value.trim();
      if (modelVal) {
        result.model = modelVal;
      } else {
        delete result.model;
      }

      // Rebuild permissions from columns
      /** @type {string[]} */
      const allow = [];
      /** @type {string[]} */
      const ask = [];
      /** @type {string[]} */
      const deny = [];

      for (const [level, col] of Object.entries(container._permColumns)) {
        const entries = /** @type {NodeListOf<HTMLElement>} */ (
          col.querySelectorAll('.config-perm-entry-interactive')
        );
        const bucket = level === 'allow' ? allow : level === 'ask' ? ask : deny;
        for (const entry of entries) {
          // Skip entries mid-removal animation
          if (entry.classList.contains('config-perm-removing')) continue;
          bucket.push(/** @type {string} */ (entry.dataset.permValue));
        }
      }

      if (allow.length || ask.length || deny.length || result.permissions) {
        result.permissions = { ...(result.permissions || {}), allow, ask, deny };
      }
      return JSON.stringify(result, null, 2);
    },

    isDirty() {
      return dirty;
    },
  };

  return container;
}


/**
 * Build an interactive permission column with draggable entries and per-column add.
 *
 * @param {string} level
 * @param {string[]} entries
 * @param {() => void} markDirty
 * @param {HTMLElement} container
 * @returns {HTMLElement}
 */
function _interactivePermColumn(level, entries, markDirty, container) {
  const col = _el('div', `config-perm-col config-perm-${level}`);
  col.dataset.level = level;

  const header = _el('div', 'config-perm-header');
  header.textContent = level.toUpperCase();
  col.appendChild(header);

  // Body: holds grouped draggable entries
  const body = _el('div', 'config-perm-col-body');

  const groups = _groupPermissions(entries);
  for (const [groupName, items] of Object.entries(groups)) {
    if (groupName !== '_ungrouped') {
      const groupLabel = _el('div', 'config-perm-group-label');
      groupLabel.textContent = groupName;
      body.appendChild(groupLabel);
    }
    for (const item of items) {
      body.appendChild(_draggablePermEntry(item.raw, markDirty));
    }
  }

  col.appendChild(body);

  // Per-column "+" add button
  const addBtn = document.createElement('button');
  addBtn.className = 'config-perm-add-btn';
  addBtn.textContent = '+';
  addBtn.title = `Add ${level} permission`;
  addBtn.addEventListener('click', () => {
    // Replace button with inline form
    addBtn.style.display = 'none';
    const form = _el('div', 'config-perm-add-form');

    const input = document.createElement('input');
    input.type = 'text';
    input.placeholder = 'e.g. Bash, Read(path)';
    input.spellcheck = false;
    form.appendChild(input);

    const confirmBtn = document.createElement('button');
    confirmBtn.className = 'config-perm-add-confirm';
    confirmBtn.textContent = '\u2713';
    confirmBtn.title = 'Add';
    form.appendChild(confirmBtn);

    const cancelBtn = document.createElement('button');
    cancelBtn.className = 'config-perm-add-cancel';
    cancelBtn.textContent = '\u2715';
    cancelBtn.title = 'Cancel';
    form.appendChild(cancelBtn);

    const doAdd = () => {
      const val = input.value.trim();
      if (!val) { closeForm(); return; }

      // Duplicate check across all columns
      const allEntries = /** @type {NodeListOf<HTMLElement>} */ (
        container.querySelectorAll('.config-perm-entry-interactive')
      );
      const existing = Array.from(allEntries).find(e => e.dataset.permValue === val);
      if (existing) {
        existing.classList.add('config-perm-flash');
        setTimeout(() => existing.classList.remove('config-perm-flash'), 600);
        closeForm();
        return;
      }

      body.appendChild(_draggablePermEntry(val, markDirty));
      markDirty();
      closeForm();
    };

    const closeForm = () => {
      form.remove();
      addBtn.style.display = '';
    };

    confirmBtn.addEventListener('click', doAdd);
    cancelBtn.addEventListener('click', closeForm);
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { e.preventDefault(); doAdd(); }
      if (e.key === 'Escape') closeForm();
    });

    col.appendChild(form);
    input.focus();
  });
  col.appendChild(addBtn);

  return col;
}


/**
 * Build a single draggable permission entry: [icon] [label] [×]
 *
 * @param {string} value
 * @param {() => void} markDirty
 * @returns {HTMLElement}
 */
function _draggablePermEntry(value, markDirty) {
  const entry = _el('div', 'config-perm-entry-interactive');
  entry.dataset.permValue = value;
  entry.dataset.dragId = String(_nextDragId++);
  entry.draggable = true;

  // Type icon
  const icon = _el('span', 'config-perm-entry-icon');
  if (value.startsWith('mcp__')) {
    icon.textContent = '\u2699';
    icon.title = 'MCP tool';
  } else if (value.startsWith('Read(') || value.startsWith('NotebookEdit(')) {
    icon.textContent = '\uD83D\uDCC4';
    icon.title = 'File access';
  } else if (value.startsWith('Task(')) {
    icon.textContent = '\uD83E\uDDE0';
    icon.title = 'Agent delegation';
  } else {
    icon.textContent = '\u26A1';
    icon.title = 'Built-in tool';
  }
  entry.appendChild(icon);

  // Label
  const label = _el('span', 'config-perm-entry-label');
  label.textContent = value;
  label.title = value;
  entry.appendChild(label);

  // Delete button
  const deleteBtn = _el('button', 'config-perm-entry-delete');
  deleteBtn.textContent = '\u00D7';
  deleteBtn.title = 'Remove';
  deleteBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    entry.classList.add('config-perm-removing');
    setTimeout(() => {
      entry.remove();
      markDirty();
    }, 200);
  });
  entry.appendChild(deleteBtn);

  return entry;
}


/**
 * Wire HTML5 drag-and-drop between the three permission columns.
 *
 * @param {HTMLElement} columnsEl
 * @param {Record<string, HTMLElement>} colMap
 * @param {() => void} markDirty
 */
function _wireDragAndDrop(columnsEl, colMap, markDirty) {
  /** @type {string|null} */
  let draggedId = null;

  columnsEl.addEventListener('dragstart', (e) => {
    const entry = /** @type {HTMLElement} */ (e.target).closest('.config-perm-entry-interactive');
    if (!entry) return;
    draggedId = /** @type {string} */ (/** @type {HTMLElement} */ (entry).dataset.dragId);
    /** @type {DataTransfer} */ (e.dataTransfer).effectAllowed = 'move';
    /** @type {DataTransfer} */ (e.dataTransfer).setData('text/plain', /** @type {string} */ (draggedId));
    entry.classList.add('config-perm-dragging');

    // Mark all columns as drop targets
    for (const col of Object.values(colMap)) {
      col.classList.add('config-perm-drop-target');
    }
  });

  columnsEl.addEventListener('dragend', (e) => {
    const entry = /** @type {HTMLElement} */ (e.target).closest('.config-perm-entry-interactive');
    if (entry) entry.classList.remove('config-perm-dragging');
    draggedId = null;

    // Clean up all drag state
    for (const col of Object.values(colMap)) {
      col.classList.remove('config-perm-drop-target', 'config-perm-drag-over');
    }
  });

  for (const col of Object.values(colMap)) {
    col.addEventListener('dragover', (e) => {
      e.preventDefault();
      /** @type {DataTransfer} */ (e.dataTransfer).dropEffect = 'move';
      col.classList.add('config-perm-drag-over');
    });

    col.addEventListener('dragenter', (e) => {
      e.preventDefault();
      col.classList.add('config-perm-drag-over');
    });

    col.addEventListener('dragleave', (e) => {
      // Only remove highlight when leaving the column itself (not a child)
      if (!col.contains(/** @type {Node|null} */ (e.relatedTarget))) {
        col.classList.remove('config-perm-drag-over');
      }
    });

    col.addEventListener('drop', (e) => {
      e.preventDefault();
      col.classList.remove('config-perm-drag-over');

      const id = /** @type {DataTransfer} */ (e.dataTransfer).getData('text/plain');
      if (!id) return;

      // Find the dragged entry by data-drag-id
      const entry = columnsEl.querySelector(
        `.config-perm-entry-interactive[data-drag-id="${id}"]`
      );
      if (!entry) return;

      // Move entry into this column's body (before the add button)
      const body = col.querySelector('.config-perm-col-body');
      if (body && entry.parentNode !== body) {
        body.appendChild(entry);
        markDirty();
      }
    });
  }
}
