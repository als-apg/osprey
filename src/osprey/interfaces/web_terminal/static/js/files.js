/* OSPREY Web Terminal — File Viewer Module */

import { createEventSource, fetchJSON } from './api.js';

let selectedPath = null;
let treeContainer = null;
let previewContainer = null;

// Track change-dot timeouts for auto-clearing
const changeDotTimers = {};

/**
 * Initialize the file viewer with tree and preview panes.
 */
export function initFileViewer(treeId, previewId) {
  treeContainer = document.getElementById(treeId);
  previewContainer = document.getElementById(previewId);

  // Load initial tree
  loadTree();

  // Connect SSE for live updates
  createEventSource('/api/files/events', {
    onMessage(data) {
      handleFileEvent(data);
    },
  });
}

async function loadTree() {
  try {
    const tree = await fetchJSON('/api/files/tree');
    renderTree(tree);
  } catch (err) {
    if (treeContainer) {
      treeContainer.innerHTML =
        '<div class="preview-empty">No workspace found</div>';
    }
  }
}

function renderTree(node) {
  if (!treeContainer) return;
  treeContainer.innerHTML = '';
  if (node.children) {
    const fragment = document.createDocumentFragment();
    renderNode(node, fragment, 0, true);
    treeContainer.appendChild(fragment);
  }
}

function renderNode(node, parent, depth, isRoot) {
  if (node.type === 'directory') {
    if (!isRoot) {
      const item = createTreeItem(node, depth, true);
      parent.appendChild(item);
    }

    const children = document.createElement('div');
    children.className = 'tree-children' + (isRoot || depth < 1 ? ' expanded' : '');
    children.dataset.path = node.path || '';

    if (node.children) {
      for (const child of node.children) {
        renderNode(child, children, isRoot ? 0 : depth + 1, false);
      }
    }
    parent.appendChild(children);
  } else {
    const item = createTreeItem(node, depth, false);
    parent.appendChild(item);
  }
}

function createTreeItem(node, depth, isDir) {
  const item = document.createElement('div');
  item.className = 'tree-item' + (isDir ? ' directory' : '');
  item.dataset.path = node.path;
  item.dataset.depth = Math.min(depth, 5);

  if (isDir) {
    const chevron = document.createElement('span');
    chevron.className = 'tree-chevron' + (depth < 1 ? ' expanded' : '');
    chevron.textContent = '\u25B6';
    item.appendChild(chevron);
  } else {
    // Spacer for alignment
    const spacer = document.createElement('span');
    spacer.style.width = '12px';
    spacer.style.flexShrink = '0';
    item.appendChild(spacer);
  }

  const icon = document.createElement('span');
  icon.className = 'tree-icon';
  icon.textContent = isDir ? '\uD83D\uDCC1' : getFileIcon(node.name);
  item.appendChild(icon);

  const name = document.createElement('span');
  name.className = 'tree-name';
  name.textContent = node.name;
  item.appendChild(name);

  // Change dot (hidden by default)
  const dot = document.createElement('span');
  dot.className = 'change-dot';
  dot.id = `dot-${node.path.replace(/[/\\]/g, '-')}`;
  item.appendChild(dot);

  // Click handler
  if (isDir) {
    item.addEventListener('click', () => toggleDirectory(item));
  } else {
    item.addEventListener('click', () => selectFile(node.path, item));
  }

  return item;
}

function toggleDirectory(item) {
  const chevron = item.querySelector('.tree-chevron');
  const children = item.nextElementSibling;

  if (children && children.classList.contains('tree-children')) {
    const isExpanded = children.classList.contains('expanded');
    children.classList.toggle('expanded');
    if (chevron) chevron.classList.toggle('expanded');
  }
}

async function selectFile(path, item) {
  // Update selection styling
  const prev = treeContainer.querySelector('.tree-item.selected');
  if (prev) prev.classList.remove('selected');
  item.classList.add('selected');
  selectedPath = path;

  // Fetch and display content
  try {
    const data = await fetchJSON(`/api/files/content/${encodeURIComponent(path)}`);
    renderPreview(data);
  } catch (err) {
    renderPreviewError(err.message);
  }
}

function renderPreview(data) {
  if (!previewContainer) return;

  const header = document.getElementById('preview-filename');
  if (header) header.textContent = data.path;

  const body = document.getElementById('preview-body');
  if (!body) return;

  // Detect language from extension
  const ext = data.extension.replace('.', '');
  const langMap = {
    py: 'python', yml: 'yaml', yaml: 'yaml', json: 'json',
    js: 'javascript', ts: 'typescript', sh: 'bash', bash: 'bash',
    md: 'markdown', html: 'xml', css: 'css', toml: 'ini',
  };

  const pre = document.createElement('pre');
  pre.className = 'preview-code';
  const code = document.createElement('code');

  const lang = langMap[ext];
  if (lang) {
    code.className = `language-${lang}`;
  }
  code.textContent = data.content;

  pre.appendChild(code);
  body.innerHTML = '';
  body.appendChild(pre);

  // Apply syntax highlighting
  if (window.hljs && lang) {
    hljs.highlightElement(code);
  }
}

function renderPreviewError(message) {
  const body = document.getElementById('preview-body');
  if (!body) return;
  body.innerHTML = `<div class="preview-empty">${message}</div>`;
}

function handleFileEvent(data) {
  // Flash a change dot on the affected file
  const dotId = `dot-${data.path.replace(/[/\\]/g, '-')}`;
  const dot = document.getElementById(dotId);
  if (dot) {
    dot.classList.add('visible');
    // Clear after 3 seconds
    if (changeDotTimers[dotId]) clearTimeout(changeDotTimers[dotId]);
    changeDotTimers[dotId] = setTimeout(() => dot.classList.remove('visible'), 3000);
  }

  // Refresh tree on create/delete
  if (data.type === 'created' || data.type === 'deleted') {
    loadTree();
  }

  // Refresh preview if the changed file is currently selected
  if (selectedPath === data.path && data.type !== 'deleted') {
    fetchJSON(`/api/files/content/${encodeURIComponent(data.path)}`)
      .then(renderPreview)
      .catch(() => {});
  }
}

function getFileIcon(name) {
  const ext = name.split('.').pop()?.toLowerCase();
  const icons = {
    py: '\uD83D\uDC0D', js: '\uD83D\uDFE8', ts: '\uD83D\uDD37',
    json: '{}', yml: '\u2699\uFE0F', yaml: '\u2699\uFE0F',
    md: '\uD83D\uDCDD', txt: '\uD83D\uDCC4', csv: '\uD83D\uDCCA',
    html: '\uD83C\uDF10', css: '\uD83C\uDFA8', sh: '$',
    png: '\uD83D\uDDBC\uFE0F', jpg: '\uD83D\uDDBC\uFE0F', svg: '\uD83D\uDDBC\uFE0F',
  };
  return icons[ext] || '\uD83D\uDCC4';
}
