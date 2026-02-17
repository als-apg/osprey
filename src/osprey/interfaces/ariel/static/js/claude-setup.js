/**
 * Claude Setup module
 *
 * File browser for Claude Code configuration files with edit mode
 * and a resizable sidebar.
 */

import { claudeSetupApi } from './api.js';
import { showClaudeConfirm } from './settings.js';

let files = [];
let selectedFile = null;
let editMode = false;
let editContent = '';
let originalContent = '';
let dirty = false;
let loaded = false;

/**
 * Whether there are unsaved edits.
 * @returns {boolean}
 */
export function hasUnsavedEdits() {
  return dirty;
}

/**
 * Initialize Claude Setup — wire up the resize handle and apply button.
 */
export function initClaudeSetup() {
  initResizeHandle();

  const applyBtn = document.getElementById('claude-apply-btn');
  if (applyBtn) {
    applyBtn.addEventListener('click', () => showClaudeConfirm());
  }

  // Restore sidebar width from localStorage
  const saved = localStorage.getItem('claude-setup-sidebar-width');
  if (saved) {
    const sidebar = document.getElementById('claude-setup-list');
    if (sidebar) sidebar.style.width = saved + 'px';
  }
}

/**
 * Load the file list from the backend (called when Claude tab is activated).
 */
export async function loadFileList() {
  if (loaded) return;

  const sidebar = document.getElementById('claude-setup-list');
  if (!sidebar) return;

  try {
    const data = await claudeSetupApi.list();
    files = data.files;
    renderFileList(sidebar);
    loaded = true;
  } catch (e) {
    sidebar.innerHTML = `<div class="claude-setup-empty">${e.message}</div>`;
  }
}

/**
 * Render the categorized file list in the sidebar.
 */
function renderFileList(container) {
  container.innerHTML = '';

  // Group by category
  const groups = {};
  for (const f of files) {
    if (!groups[f.category]) groups[f.category] = [];
    groups[f.category].push(f);
  }

  for (const [category, items] of Object.entries(groups)) {
    const catEl = document.createElement('div');
    catEl.className = 'claude-setup-category';
    catEl.textContent = category;
    container.appendChild(catEl);

    for (const item of items) {
      const el = document.createElement('div');
      el.className = `claude-setup-file-item${item.exists ? '' : ' missing'}`;
      if (selectedFile === item.path) el.classList.add('active');

      el.innerHTML = `
        <svg class="claude-setup-file-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/>
          <polyline points="14 2 14 8 20 8"/>
        </svg>
        <span>${escapeHtml(item.path)}</span>
      `;

      if (item.exists) {
        el.addEventListener('click', () => selectFile(item.path));
      }
      container.appendChild(el);
    }
  }
}

/**
 * Select and load a file for viewing.
 */
async function selectFile(path) {
  if (dirty && path !== selectedFile) {
    if (!confirm('Discard unsaved edits to the current file?')) return;
  }

  selectedFile = path;
  editMode = false;
  dirty = false;
  updateSaveBar();

  // Highlight in sidebar
  document.querySelectorAll('.claude-setup-file-item').forEach(el => {
    const span = el.querySelector('span');
    el.classList.toggle('active', span && span.textContent === path);
  });

  const viewer = document.getElementById('claude-setup-viewer');
  if (!viewer) return;

  viewer.innerHTML = '<div class="claude-setup-empty"><div class="spinner"></div>Loading...</div>';

  try {
    const data = await claudeSetupApi.getFile(path);
    originalContent = data.content;
    renderFileContent(viewer, path, data.content);
  } catch (e) {
    viewer.innerHTML = `<div class="claude-setup-empty">Error: ${escapeHtml(e.message)}</div>`;
  }
}

/**
 * Render file content with header (title + edit button).
 */
function renderFileContent(container, path, content) {
  container.innerHTML = `
    <div class="claude-setup-file-header">
      <span class="claude-setup-file-title">${escapeHtml(path)}</span>
      <button class="claude-setup-edit-btn" id="claude-edit-toggle">Edit</button>
    </div>
    <div class="claude-setup-content" id="claude-content-area">
      <pre>${escapeHtml(content)}</pre>
    </div>
  `;

  document.getElementById('claude-edit-toggle').addEventListener('click', toggleEdit);
}

/**
 * Toggle between view mode (pre) and edit mode (textarea).
 */
function toggleEdit() {
  editMode = !editMode;
  const btn = document.getElementById('claude-edit-toggle');
  const area = document.getElementById('claude-content-area');

  if (editMode) {
    btn.textContent = 'Cancel';
    btn.classList.add('active');
    editContent = originalContent;
    area.innerHTML = `<textarea id="claude-edit-textarea">${escapeHtml(editContent)}</textarea>`;
    const textarea = document.getElementById('claude-edit-textarea');
    textarea.addEventListener('input', () => {
      editContent = textarea.value;
      dirty = editContent !== originalContent;
      updateSaveBar();
    });
    textarea.focus();
  } else {
    btn.textContent = 'Edit';
    btn.classList.remove('active');
    dirty = false;
    editContent = '';
    area.innerHTML = `<pre>${escapeHtml(originalContent)}</pre>`;
    updateSaveBar();
  }
}

/**
 * Update the Claude save bar visibility and status.
 */
function updateSaveBar() {
  const bar = document.getElementById('claude-save-bar');
  const status = document.getElementById('claude-status');
  const applyBtn = document.getElementById('claude-apply-btn');

  if (!bar) return;

  if (editMode) {
    bar.style.display = 'flex';
    if (dirty) {
      status.textContent = 'Unsaved changes';
      status.className = 'settings-status dirty';
      applyBtn.disabled = false;
    } else {
      status.textContent = 'No changes';
      status.className = 'settings-status';
      applyBtn.disabled = true;
    }
  } else {
    bar.style.display = 'none';
  }
}

/**
 * Save the current file — called by the confirmation dialog in settings.js.
 */
export async function doSave() {
  if (!selectedFile || !dirty) return;

  const status = document.getElementById('claude-status');
  try {
    status.textContent = 'Saving...';
    status.className = 'settings-status';
    await claudeSetupApi.updateFile(selectedFile, editContent);
    originalContent = editContent;
    dirty = false;
    status.textContent = 'Saved successfully';
    status.className = 'settings-status saved';

    // Update the viewer content
    setTimeout(() => {
      if (!dirty) {
        editMode = false;
        const btn = document.getElementById('claude-edit-toggle');
        if (btn) {
          btn.textContent = 'Edit';
          btn.classList.remove('active');
        }
        const area = document.getElementById('claude-content-area');
        if (area) area.innerHTML = `<pre>${escapeHtml(originalContent)}</pre>`;
        updateSaveBar();
      }
    }, 1500);
  } catch (e) {
    status.textContent = `Save failed: ${e.message}`;
    status.className = 'settings-status error';
  }
}

// --- Resize handle ---

function initResizeHandle() {
  const handle = document.querySelector('.claude-setup-resize-handle');
  const sidebar = document.getElementById('claude-setup-list');
  if (!handle || !sidebar) return;

  let startX = 0;
  let startWidth = 0;

  function onMouseMove(e) {
    const delta = e.clientX - startX;
    const newWidth = Math.min(300, Math.max(120, startWidth + delta));
    sidebar.style.width = newWidth + 'px';
  }

  function onMouseUp() {
    handle.classList.remove('dragging');
    document.removeEventListener('mousemove', onMouseMove);
    document.removeEventListener('mouseup', onMouseUp);
    localStorage.setItem('claude-setup-sidebar-width', parseInt(sidebar.style.width));
  }

  handle.addEventListener('mousedown', (e) => {
    e.preventDefault();
    startX = e.clientX;
    startWidth = sidebar.offsetWidth;
    handle.classList.add('dragging');
    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
  });
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}
