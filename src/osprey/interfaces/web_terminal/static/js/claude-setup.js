/* OSPREY Web Terminal — Claude Setup Viewer + Editor */

import { fetchJSON } from './api.js';
import { registerUnsavedGuard } from './drawer.js';

let files = [];
let selectedFile = null;
let isEditing = false;
let claudeDirty = false;
let loaded = false;

// Category display order
const CATEGORY_ORDER = [
  'System Prompt',
  'MCP Servers',
  'Permissions',
  'Safety',
  'Agents',
  'Commands',
  'Hooks',
  'Other',
];

// Allowed directories for new file creation
const ALLOWED_DIRS = ['rules', 'agents', 'commands', 'hooks'];

/**
 * Initialize the Claude Setup viewer. Call once on DOMContentLoaded.
 */
export function initClaudeSetup() {
  const drawer = document.getElementById('settings-drawer');
  const claudePanel = document.getElementById('tab-claude');
  if (!drawer || !claudePanel) return;

  // Load files when Claude tab becomes active
  claudePanel.addEventListener('drawer:tab-activate', () => {
    if (!loaded) loadSetupFiles();
  });

  // Reset loaded flag when drawer closes
  drawer.addEventListener('drawer:close', () => {
    loaded = false;
    exitEditMode();
  });

  // Save button
  const saveBtn = claudePanel.querySelector('.claude-setup-save-btn');
  if (saveBtn) saveBtn.addEventListener('click', saveClaudeFile);

  // Register unsaved-changes guard
  registerUnsavedGuard('settings-drawer', () => {
    if (!claudeDirty) return true;
    return confirm('You have unsaved changes. Discard them?');
  });
}

/**
 * Check if there are unsaved changes (exported for guard use).
 */
export function hasUnsavedChanges() {
  return claudeDirty;
}

async function loadSetupFiles() {
  const fileList = document.getElementById('claude-setup-list');
  const viewer = document.getElementById('claude-setup-viewer');
  const loading = document.getElementById('claude-setup-loading');
  const error = document.getElementById('claude-setup-error');

  if (loading) loading.style.display = 'flex';
  if (error) error.style.display = 'none';
  if (fileList) fileList.innerHTML = '';
  if (viewer) {
    viewer.innerHTML = '<div class="claude-setup-empty">Select a file to view</div>';
  }

  try {
    const data = await fetchJSON('/api/claude-setup');
    files = data.files || [];
    if (loading) loading.style.display = 'none';

    renderFileList();
    loaded = true;

    // Auto-select first file
    if (files.length > 0) {
      selectFile(files[0]);
    }
  } catch (e) {
    if (loading) loading.style.display = 'none';
    if (error) {
      error.style.display = 'flex';
      error.textContent = `Failed to load: ${e.message}`;
    }
  }
}

function renderFileList() {
  const container = document.getElementById('claude-setup-list');
  if (!container) return;
  container.innerHTML = '';

  // Files container (scrollable)
  const filesDiv = document.createElement('div');
  filesDiv.className = 'claude-setup-sidebar-files';

  // Group files by category
  const groups = {};
  for (const file of files) {
    const cat = file.category || 'Other';
    if (!groups[cat]) groups[cat] = [];
    groups[cat].push(file);
  }

  // Render in order
  for (const category of CATEGORY_ORDER) {
    const groupFiles = groups[category];
    if (!groupFiles || groupFiles.length === 0) continue;

    const label = document.createElement('div');
    label.className = 'claude-setup-category';
    label.textContent = category;
    filesDiv.appendChild(label);

    for (const file of groupFiles) {
      const item = document.createElement('div');
      item.className = 'claude-setup-item';
      item.dataset.path = file.path;

      const icon = document.createElement('span');
      icon.className = 'claude-setup-item-icon';
      icon.textContent = iconForLanguage(file.language);

      const name = document.createElement('span');
      name.className = 'claude-setup-item-name';
      name.textContent = file.name;

      item.appendChild(icon);
      item.appendChild(name);

      item.addEventListener('click', () => {
        if (claudeDirty && !confirm('You have unsaved changes. Discard them?')) return;
        exitEditMode();
        selectFile(file);
      });
      filesDiv.appendChild(item);
    }
  }

  container.appendChild(filesDiv);

  // Create file button
  const createBtn = document.createElement('button');
  createBtn.className = 'claude-setup-create-btn';
  createBtn.textContent = '+ New File';
  createBtn.addEventListener('click', showCreateForm);
  container.appendChild(createBtn);

  // Create form (hidden by default)
  const form = buildCreateForm();
  container.appendChild(form);
}

function selectFile(file) {
  selectedFile = file;

  // Update selected state in list
  document.querySelectorAll('.claude-setup-item').forEach((el) => {
    el.classList.toggle('selected', el.dataset.path === file.path);
  });

  // Render content
  const viewer = document.getElementById('claude-setup-viewer');
  if (!viewer) return;

  viewer.innerHTML = '';

  // File header
  const header = document.createElement('div');
  header.className = 'claude-setup-file-header';

  const pathEl = document.createElement('span');
  pathEl.className = 'claude-setup-file-path';
  pathEl.textContent = file.path;

  const actions = document.createElement('div');
  actions.className = 'claude-setup-file-actions';

  const langEl = document.createElement('span');
  langEl.className = 'claude-setup-file-lang';
  langEl.textContent = file.language;

  const editBtn = document.createElement('button');
  editBtn.className = 'claude-setup-edit-btn';
  editBtn.textContent = '\u270E EDIT';
  editBtn.addEventListener('click', () => toggleEdit(editBtn));

  actions.appendChild(langEl);
  actions.appendChild(editBtn);
  header.appendChild(pathEl);
  header.appendChild(actions);

  // View content (pre/code)
  const content = document.createElement('pre');
  content.className = 'claude-setup-content';
  content.id = 'claude-content-view';

  const code = document.createElement('code');
  code.textContent = file.content;
  content.appendChild(code);

  // Editor textarea (hidden)
  const editor = document.createElement('textarea');
  editor.className = 'claude-setup-content-editor';
  editor.id = 'claude-content-editor';
  editor.spellcheck = false;
  editor.value = file.content;
  editor.addEventListener('input', () => markClaudeDirty());

  viewer.appendChild(header);
  viewer.appendChild(content);
  viewer.appendChild(editor);
}

function toggleEdit(btn) {
  const contentView = document.getElementById('claude-content-view');
  const editorView = document.getElementById('claude-content-editor');

  if (!contentView || !editorView) return;

  if (isEditing) {
    // Switch to view mode
    if (claudeDirty && !confirm('You have unsaved changes. Discard them?')) return;
    exitEditMode();
    // Refresh content display from file
    if (selectedFile) {
      const code = contentView.querySelector('code');
      if (code) code.textContent = selectedFile.content;
    }
  } else {
    // Switch to edit mode — danger zone
    isEditing = true;
    btn.textContent = '\u25CF EDITING';
    btn.classList.add('active');
    contentView.style.display = 'none';
    editorView.classList.add('active');
    editorView.value = selectedFile ? selectedFile.content : '';
    editorView.focus();

    const panel = document.getElementById('tab-claude');
    if (panel) panel.classList.add('claude-editing');
  }
}

function exitEditMode() {
  isEditing = false;
  claudeDirty = false;
  updateClaudeSaveBar();

  const contentView = document.getElementById('claude-content-view');
  const editorView = document.getElementById('claude-content-editor');
  const editBtn = document.querySelector('.claude-setup-edit-btn');
  const panel = document.getElementById('tab-claude');

  if (contentView) contentView.style.display = '';
  if (editorView) editorView.classList.remove('active');
  if (editBtn) {
    editBtn.textContent = '\u270E EDIT';
    editBtn.classList.remove('active');
  }
  if (panel) panel.classList.remove('claude-editing');
}

function markClaudeDirty() {
  claudeDirty = true;
  updateClaudeSaveBar();
}

function updateClaudeSaveBar() {
  const bar = document.querySelector('.claude-setup-save-bar');
  if (bar) bar.classList.toggle('visible', claudeDirty);
}

async function saveClaudeFile() {
  if (!selectedFile) return;

  const editorView = document.getElementById('claude-content-editor');
  const status = document.querySelector('.claude-setup-status');
  const saveBtn = document.querySelector('.claude-setup-save-btn');
  if (!editorView) return;

  const content = editorView.value;
  if (saveBtn) saveBtn.disabled = true;
  if (status) status.textContent = 'Saving...';

  try {
    const resp = await fetch('/api/claude-setup', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path: selectedFile.path, content }),
    });

    if (!resp.ok) {
      const detail = await resp.json().catch(() => ({}));
      throw new Error(detail.detail || `Save failed (HTTP ${resp.status})`);
    }

    // Update local state
    selectedFile.content = content;
    claudeDirty = false;
    updateClaudeSaveBar();

    // Update the view content too
    const contentView = document.getElementById('claude-content-view');
    if (contentView) {
      const code = contentView.querySelector('code');
      if (code) code.textContent = content;
    }

    if (status) {
      status.textContent = 'Saved';
      setTimeout(() => { status.textContent = ''; }, 2000);
    }
  } catch (e) {
    if (status) status.textContent = e.message;
  } finally {
    if (saveBtn) saveBtn.disabled = false;
  }
}

// ---- Create File UI ---- //

function buildCreateForm() {
  const form = document.createElement('div');
  form.className = 'claude-setup-create-form';
  form.id = 'claude-create-form';

  // Directory select
  const dirSelect = document.createElement('select');
  dirSelect.id = 'claude-create-dir';
  for (const dir of ALLOWED_DIRS) {
    const opt = document.createElement('option');
    opt.value = dir;
    opt.textContent = `.claude/${dir}/`;
    dirSelect.appendChild(opt);
  }

  // Filename input
  const nameInput = document.createElement('input');
  nameInput.type = 'text';
  nameInput.id = 'claude-create-name';
  nameInput.placeholder = 'filename.md';

  // Buttons
  const actions = document.createElement('div');
  actions.className = 'claude-setup-create-actions';

  const cancelBtn = document.createElement('button');
  cancelBtn.className = 'claude-setup-create-cancel';
  cancelBtn.textContent = 'Cancel';
  cancelBtn.addEventListener('click', hideCreateForm);

  const submitBtn = document.createElement('button');
  submitBtn.className = 'claude-setup-create-submit';
  submitBtn.textContent = 'Create';
  submitBtn.addEventListener('click', createFile);

  actions.appendChild(cancelBtn);
  actions.appendChild(submitBtn);

  form.appendChild(dirSelect);
  form.appendChild(nameInput);
  form.appendChild(actions);

  return form;
}

function showCreateForm() {
  const form = document.getElementById('claude-create-form');
  if (form) form.classList.add('active');
}

function hideCreateForm() {
  const form = document.getElementById('claude-create-form');
  const nameInput = document.getElementById('claude-create-name');
  if (form) form.classList.remove('active');
  if (nameInput) nameInput.value = '';
}

async function createFile() {
  const dirSelect = document.getElementById('claude-create-dir');
  const nameInput = document.getElementById('claude-create-name');
  if (!dirSelect || !nameInput) return;

  const dir = dirSelect.value;
  const name = nameInput.value.trim();
  if (!name) return;

  const path = `.claude/${dir}/${name}`;

  try {
    const resp = await fetch('/api/claude-setup', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path, content: '' }),
    });

    if (!resp.ok) {
      const detail = await resp.json().catch(() => ({}));
      throw new Error(detail.detail || `Create failed (HTTP ${resp.status})`);
    }

    const result = await resp.json();

    hideCreateForm();

    // Reload files and select the new one
    const data = await fetchJSON('/api/claude-setup');
    files = data.files || [];
    renderFileList();

    const newFile = files.find((f) => f.path === path);
    if (newFile) selectFile(newFile);
  } catch (e) {
    alert(`Failed to create file: ${e.message}`);
  }
}

function iconForLanguage(lang) {
  switch (lang) {
    case 'markdown': return '\uD83D\uDCC4';
    case 'json': return '{ }';
    case 'yaml': return '\u2699';
    case 'shell': return '$';
    case 'python': return '\uD83D\uDC0D';
    default: return '\uD83D\uDCC3';
  }
}
