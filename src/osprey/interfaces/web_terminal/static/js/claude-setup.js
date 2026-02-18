/* OSPREY Web Terminal — Claude Setup Viewer + Editor
 *
 * Rich rendering for Claude Code integration files:
 *   - Markdown: rendered HTML via marked.js (with Raw toggle)
 *   - JSON: syntax-highlighted via highlight.js (+ MCP tree view for .mcp.json)
 *   - Python/Shell/YAML: syntax-highlighted via highlight.js
 *   - Raw mode always available via View/Raw toggle
 */

import { fetchJSON } from './api.js';
import { registerUnsavedGuard } from './drawer.js';

let files = [];
let selectedFile = null;
let isEditing = false;
let claudeDirty = false;
let loaded = false;
let viewMode = 'rich'; // 'rich' | 'raw'

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

// highlight.js language map
const HLJS_LANG = {
  json: 'json',
  python: 'python',
  shell: 'bash',
  yaml: 'yaml',
  markdown: 'markdown',
};

/**
 * Initialize the Claude Setup viewer. Call once on DOMContentLoaded.
 */
export function initClaudeSetup() {
  const drawer = document.getElementById('settings-drawer');
  const claudePanel = document.getElementById('tab-claude');
  if (!drawer || !claudePanel) return;

  // Configure marked.js for rendering markdown
  configureMarked();

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

// ---- Marked.js Configuration ---- //

function configureMarked() {
  if (typeof marked === 'undefined') return;

  // marked v12+ uses renderer for code highlighting (not setOptions.highlight)
  const renderer = {
    code({ text, lang }) {
      // Guard: marked may pass undefined/null text for edge-case blocks
      const src = text ?? '';
      let highlighted = escapeHtml(src);
      if (typeof hljs !== 'undefined' && src) {
        try {
          if (lang && hljs.getLanguage(lang)) {
            highlighted = hljs.highlight(src, { language: lang }).value;
          } else {
            highlighted = hljs.highlightAuto(src).value;
          }
        } catch {
          // Fall back to escaped text on any hljs error
        }
      }
      const langClass = lang ? ` class="language-${lang}"` : '';
      return `<pre><code${langClass}>${highlighted}</code></pre>`;
    },
  };

  // Sanitize tokens before rendering to prevent 'e.replace is not an object' errors
  function walkTokens(token) {
    if (token.type === 'code' && typeof token.text !== 'string') {
      token.text = token.text != null ? String(token.text) : '';
    }
  }

  marked.use({ gfm: true, breaks: false, renderer, walkTokens });
}

// ---- File Loading ---- //

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

// ---- File Selection & Rendering ---- //

function selectFile(file) {
  selectedFile = file;
  viewMode = 'rich';

  // Update selected state in list
  document.querySelectorAll('.claude-setup-item').forEach((el) => {
    el.classList.toggle('selected', el.dataset.path === file.path);
  });

  renderFileView();
}

function renderFileView() {
  const viewer = document.getElementById('claude-setup-viewer');
  if (!viewer || !selectedFile) return;

  viewer.innerHTML = '';

  // File header
  const header = document.createElement('div');
  header.className = 'claude-setup-file-header';

  const pathEl = document.createElement('span');
  pathEl.className = 'claude-setup-file-path';
  pathEl.textContent = selectedFile.path;

  const actions = document.createElement('div');
  actions.className = 'claude-setup-file-actions';

  // View/Raw toggle (only show if file has rich rendering)
  if (hasRichView(selectedFile)) {
    const viewBtn = document.createElement('button');
    viewBtn.className = 'claude-setup-mode-btn' + (viewMode === 'rich' ? ' active' : '');
    viewBtn.textContent = 'Rendered';
    viewBtn.dataset.mode = 'rich';
    viewBtn.addEventListener('click', () => switchViewMode('rich'));

    const rawBtn = document.createElement('button');
    rawBtn.className = 'claude-setup-mode-btn' + (viewMode === 'raw' ? ' active' : '');
    rawBtn.textContent = 'Source';
    rawBtn.dataset.mode = 'raw';
    rawBtn.addEventListener('click', () => switchViewMode('raw'));

    actions.appendChild(viewBtn);
    actions.appendChild(rawBtn);
  }

  const langEl = document.createElement('span');
  langEl.className = 'claude-setup-file-lang';
  langEl.textContent = selectedFile.language;

  const editBtn = document.createElement('button');
  editBtn.className = 'claude-setup-edit-btn';
  editBtn.textContent = '\u270E EDIT';
  editBtn.addEventListener('click', () => toggleEdit(editBtn));

  actions.appendChild(langEl);
  actions.appendChild(editBtn);
  header.appendChild(pathEl);
  header.appendChild(actions);

  // Rich content view
  const richContent = document.createElement('div');
  richContent.className = 'claude-setup-content';
  richContent.id = 'claude-content-view';

  if (viewMode === 'rich' && hasRichView(selectedFile)) {
    richContent.innerHTML = renderRichContent(selectedFile);
  } else {
    // Raw view: pre/code with optional syntax highlighting
    const pre = document.createElement('pre');
    pre.className = 'claude-setup-raw-pre';
    const code = document.createElement('code');
    const lang = HLJS_LANG[selectedFile.language];
    if (lang) code.className = `language-${lang}`;
    code.textContent = selectedFile.content;
    pre.appendChild(code);
    richContent.appendChild(pre);

    // Apply syntax highlighting to raw view
    if (typeof hljs !== 'undefined' && lang) {
      hljs.highlightElement(code);
    }
  }

  // Editor textarea (hidden)
  const editor = document.createElement('textarea');
  editor.className = 'claude-setup-content-editor';
  editor.id = 'claude-content-editor';
  editor.spellcheck = false;
  editor.value = selectedFile.content;
  editor.addEventListener('input', () => markClaudeDirty());

  viewer.appendChild(header);
  viewer.appendChild(richContent);
  viewer.appendChild(editor);
}

function hasRichView(file) {
  if (file.language === 'markdown') return true;
  if (file.language === 'json' && file.name === '.mcp.json') return true;
  if (file.language === 'json' && file.name === 'settings.json') return true;
  return false;
}

function switchViewMode(mode) {
  viewMode = mode;
  // Re-render the file view (preserving selection)
  renderFileView();
}

// ---- Rich Content Renderers ---- //

function renderRichContent(file) {
  if (file.language === 'markdown') return renderMarkdown(file.content);
  if (file.language === 'json' && file.name === '.mcp.json') return renderMcpJson(file.content);
  if (file.language === 'json' && file.name === 'settings.json') return renderSettingsJson(file.content);
  return escapeHtml(file.content);
}

function renderMarkdown(content) {
  if (typeof marked === 'undefined') return `<pre>${escapeHtml(content)}</pre>`;
  try {
    return `<div class="claude-md-rendered">${marked.parse(content)}</div>`;
  } catch (e) {
    console.warn('marked.parse() error:', e);
    return `<pre>${escapeHtml(content)}</pre>`;
  }
}

function renderMcpJson(content) {
  let data;
  try {
    data = JSON.parse(content);
  } catch {
    return `<div class="claude-parse-error">Invalid JSON</div><pre>${escapeHtml(content)}</pre>`;
  }

  const servers = data.mcpServers || {};
  const serverNames = Object.keys(servers);

  if (serverNames.length === 0) {
    return '<div class="mcp-empty">No MCP servers configured</div>';
  }

  const cards = serverNames.map((name) => {
    const server = servers[name];
    return renderMcpServerCard(name, server);
  });

  return `<div class="mcp-tree">${cards.join('')}</div>`;
}

function renderMcpServerCard(name, server) {
  const rows = [];

  // Command
  if (server.command) {
    rows.push(mcpRow('Command', `<code>${escapeHtml(server.command)}</code>`));
  }

  // Args
  if (server.args && server.args.length > 0) {
    const argItems = server.args.map((a) => `<span class="mcp-arg">${escapeHtml(String(a))}</span>`);
    rows.push(mcpRow('Args', `<div class="mcp-args-list">${argItems.join('')}</div>`));
  }

  // Env
  if (server.env && Object.keys(server.env).length > 0) {
    const envRows = Object.entries(server.env).map(([k, v]) => {
      const displayVal = isSensitiveEnvKey(k) ? maskValue(String(v)) : escapeHtml(String(v));
      return `<div class="mcp-env-row"><span class="mcp-env-key">${escapeHtml(k)}</span><span class="mcp-env-val">${displayVal}</span></div>`;
    });
    rows.push(mcpRow('Env', `<div class="mcp-env-table">${envRows.join('')}</div>`));
  }

  // URL (for SSE/streamable-http transports)
  if (server.url) {
    rows.push(mcpRow('URL', `<code>${escapeHtml(server.url)}</code>`));
  }

  // Type/transport
  if (server.type) {
    rows.push(mcpRow('Type', `<span class="mcp-type-badge">${escapeHtml(server.type)}</span>`));
  }

  const statusDot = server.command || server.url ? 'mcp-status-active' : 'mcp-status-unknown';

  return `
    <details class="mcp-server-card" open>
      <summary class="mcp-server-header">
        <span class="mcp-status-dot ${statusDot}"></span>
        <span class="mcp-server-name">${escapeHtml(name)}</span>
      </summary>
      <div class="mcp-server-body">${rows.join('')}</div>
    </details>
  `;
}

function mcpRow(label, valueHtml) {
  return `<div class="mcp-row"><span class="mcp-row-label">${label}</span><div class="mcp-row-value">${valueHtml}</div></div>`;
}

function isSensitiveEnvKey(key) {
  const lower = key.toLowerCase();
  return lower.includes('token') || lower.includes('secret') || lower.includes('password')
    || lower.includes('key') || lower.includes('credential');
}

function maskValue(value) {
  if (value.length <= 8) return '\u2022'.repeat(value.length);
  return value.slice(0, 4) + '\u2022'.repeat(Math.min(value.length - 4, 12)) + value.slice(-4);
}

function renderSettingsJson(content) {
  let data;
  try {
    data = JSON.parse(content);
  } catch {
    return `<div class="claude-parse-error">Invalid JSON</div><pre>${escapeHtml(content)}</pre>`;
  }

  const sections = [];

  // Permissions
  if (data.permissions) {
    sections.push(renderJsonSection('Permissions', data.permissions));
  }

  // Allowed tools
  if (data.allowedTools) {
    sections.push(renderJsonArraySection('Allowed Tools', data.allowedTools));
  }

  // Denied tools
  if (data.deniedTools) {
    sections.push(renderJsonArraySection('Denied Tools', data.deniedTools));
  }

  // Any other top-level keys
  const known = new Set(['permissions', 'allowedTools', 'deniedTools']);
  for (const [key, val] of Object.entries(data)) {
    if (!known.has(key)) {
      if (Array.isArray(val)) {
        sections.push(renderJsonArraySection(formatKey(key), val));
      } else if (typeof val === 'object' && val !== null) {
        sections.push(renderJsonSection(formatKey(key), val));
      } else {
        sections.push(`<div class="json-section"><div class="json-section-header">${formatKey(key)}</div><div class="json-section-body"><code>${escapeHtml(String(val))}</code></div></div>`);
      }
    }
  }

  if (sections.length === 0) {
    return '<div class="mcp-empty">Empty settings file</div>';
  }

  return `<div class="json-tree">${sections.join('')}</div>`;
}

function renderJsonSection(title, obj) {
  const rows = Object.entries(obj).map(([k, v]) => {
    let valHtml;
    if (typeof v === 'boolean') {
      valHtml = `<span class="json-bool">${v}</span>`;
    } else if (typeof v === 'object' && v !== null) {
      valHtml = `<code class="json-nested">${escapeHtml(JSON.stringify(v, null, 2))}</code>`;
    } else {
      valHtml = `<code>${escapeHtml(String(v))}</code>`;
    }
    return `<div class="mcp-row"><span class="mcp-row-label">${formatKey(k)}</span><div class="mcp-row-value">${valHtml}</div></div>`;
  });

  return `
    <details class="json-section" open>
      <summary class="json-section-header">${escapeHtml(title)}</summary>
      <div class="json-section-body">${rows.join('')}</div>
    </details>
  `;
}

function renderJsonArraySection(title, arr) {
  const items = arr.map((item) => {
    if (typeof item === 'string') {
      return `<span class="mcp-arg">${escapeHtml(item)}</span>`;
    }
    return `<code>${escapeHtml(JSON.stringify(item))}</code>`;
  });

  return `
    <details class="json-section" open>
      <summary class="json-section-header">${escapeHtml(title)} <span class="json-count">${arr.length}</span></summary>
      <div class="json-section-body"><div class="mcp-args-list">${items.join('')}</div></div>
    </details>
  `;
}

function formatKey(key) {
  return key.replace(/([A-Z])/g, ' $1').replace(/[_-]/g, ' ').replace(/^\s/, '').replace(/\b\w/g, (c) => c.toUpperCase());
}

// ---- Edit Mode ---- //

function toggleEdit(btn) {
  const contentView = document.getElementById('claude-content-view');
  const editorView = document.getElementById('claude-content-editor');

  if (!contentView || !editorView) return;

  if (isEditing) {
    // Switch to view mode
    if (claudeDirty && !confirm('You have unsaved changes. Discard them?')) return;
    exitEditMode();
    // Re-render the view
    renderFileView();
  } else {
    // Switch to edit mode
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

    // Re-render the view with updated content
    exitEditMode();
    renderFileView();

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

// ---- Helpers ---- //

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

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}
