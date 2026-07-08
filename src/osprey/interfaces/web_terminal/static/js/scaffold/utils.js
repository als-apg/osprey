// @ts-check
/**
 * OSPREY Web Terminal — Scaffold Gallery: pure utilities
 *
 * Module-level pure utilities and shared constants used by scaffold-gallery.js:
 * category metadata/routing tables, YAML front-matter parsing, code/diagram
 * rendering helpers, and the one-time Marked.js configuration.
 *
 * `marked` and `hljs` are vendored classic-script globals (see
 * src/osprey/interfaces/vendor-globals.d.ts) rather than ES imports; every
 * reference here is guarded with a `typeof` check so this module is safe to
 * import (and its functions safe to call) before those scripts have loaded.
 *
 * @module scaffold/utils
 */

import { escapeHtml } from '/design-system/js/dom.js';

// ---- Constants ---- //

// TODO: Pull from provider registry when Claude is routed through CBORG/other providers
/** @type {string[]} */
export const AGENT_MODEL_OPTIONS = ['haiku', 'sonnet', 'opus'];

/**
 * Brief descriptions for each artifact category, shown in the help tooltip.
 * @type {Record<string, string>}
 */
export const CATEGORY_HELP = {
  'system instructions': 'The main CLAUDE.md file that defines the AI assistant\'s identity, capabilities, and behavioral guidelines.',
  agents: 'Sub-agents that Claude delegates specialized tasks to (search, analysis, visualization). Each agent has its own model, tools, and instructions.',
  config: 'Top-level configuration files: MCP server definitions (.mcp.json) and permissions (settings.json).',
  hooks: 'Python scripts that run before or after Claude uses a tool. They enforce safety rules, validate inputs, and inject error guidance.',
  instructions: 'Markdown files loaded as persistent directives. They define safety boundaries, error handling protocols, and artifact conventions.',
  skills: 'Multi-file bundles that Claude can invoke as structured workflows. Skills support companion files (CSS/JS references, templates).',
  'output-styles': 'Markdown style guides that shape how Claude writes responses — tone, format, and epistemic discipline for control system communication.',
};

// ---- Category Routing ---- //

export const BEHAVIOR_CATEGORIES = new Set(['agents', 'skills', 'rules', 'output-styles']);
export const BEHAVIOR_NAMES = new Set(['claude-md']);        // config category, behavior tab
export const SAFETY_CATEGORIES = new Set(['hooks']);
export const CONFIG_NAMES = new Set(['mcp-json', 'settings-json']); // config category, config tab

// ---- Marked.js Configuration (one-time) ---- //

let _markedConfigured = false;

/**
 * @typedef {object} MarkedCodeToken
 * @property {string} text
 * @property {string} [lang]
 */

/**
 * Configure the vendored `marked` global with a syntax-highlighting code
 * renderer, once. Safe to call repeatedly (no-op after the first call) and
 * safe to call before `marked` has loaded (early-returns).
 * @returns {void}
 */
export function configureMarked() {
  if (_markedConfigured) return;
  _markedConfigured = true;

  if (typeof marked === 'undefined') return;

  const renderer = {
    /**
     * @param {MarkedCodeToken} token
     * @returns {string}
     */
    code({ text, lang }) {
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

  /**
   * @param {{type: string, text?: unknown}} token
   * @returns {void}
   */
  function walkTokens(token) {
    if (token.type === 'code' && typeof token.text !== 'string') {
      token.text = token.text != null ? String(token.text) : '';
    }
  }

  marked.use({ gfm: true, breaks: false, renderer, walkTokens });
}

// ---- Module-Level Utility Functions ---- //

/**
 * Return an emoji icon for a given artifact category.
 * @param {string} [cat]
 * @returns {string}
 */
export function iconForCategory(cat) {
  switch ((cat || '').toLowerCase()) {
    case 'system prompt':   return '📜'; // scroll
    case 'instructions': return '📋'; // clipboard
    case 'agents':   return '🤖';  // robot
    case 'hooks':    return '⚡';         // lightning
    case 'commands': return '⌘';         // command key
    case 'config':   return '⚙';         // gear
    case 'skills':   return '📦';  // package
    default:         return '📄';   // document
  }
}

/**
 * Parse YAML front matter (between --- delimiters) from markdown content.
 * @param {string} content
 * @returns {{frontMatter: Record<string, string>|null, body: string}}
 */
export function parseFrontMatter(content) {
  const match = content.match(/^---\n([\s\S]*?)\n---\n?([\s\S]*)$/);
  if (!match) return { frontMatter: null, body: content };

  const yamlBlock = match[1];
  const body = match[2];

  /** @type {Record<string, string>} */
  const fields = {};
  for (const line of yamlBlock.split('\n')) {
    // eslint-disable-next-line no-useless-escape -- escaped hyphen retained for readability of the key-name char class
    const kv = line.match(/^(\w[\w\-]*):\s*(.*)$/);
    if (kv) {
      let value = kv[2].trim();
      if ((value.startsWith('"') && value.endsWith('"')) ||
          (value.startsWith("'") && value.endsWith("'"))) {
        value = value.slice(1, -1);
      }
      fields[kv[1]] = value;
    }
  }

  return { frontMatter: Object.keys(fields).length > 0 ? fields : null, body };
}

/**
 * Extract YAML front matter from a Python module docstring.
 * @param {string} content
 * @returns {{frontMatter: Record<string, string>|null, body: string, flowDiagram: string|null, sourceCode: string}}
 */
export function extractPythonDocstringFrontMatter(content) {
  /** @type {{frontMatter: Record<string, string>|null, body: string, flowDiagram: string|null, sourceCode: string}} */
  const result = { frontMatter: null, body: '', flowDiagram: null, sourceCode: content };

  const docMatch = content.match(/^(?:#!.*\n)?"""\n?([\s\S]*?)"""/);
  if (!docMatch) return result;

  const docstring = docMatch[1];
  const { frontMatter, body } = parseFrontMatter(docstring);

  result.frontMatter = frontMatter;

  const trimmed = body.trim();
  const flowMatch = trimmed.match(/## Flow\s*\n\s*```\n?([\s\S]*?)```/);
  if (flowMatch) {
    result.flowDiagram = flowMatch[1].trimEnd();
    result.body = trimmed.replace(/## Flow\s*\n\s*```\n?[\s\S]*?```/, '').trim();
  } else {
    result.body = trimmed;
  }

  return result;
}

/**
 * Create a syntax-highlighted code block element.
 * @param {string} content
 * @param {string} [language]
 * @returns {HTMLPreElement}
 */
export function renderHighlightedCode(content, language) {
  const pre = document.createElement('pre');
  const code = document.createElement('code');
  if (language) code.className = `language-${language}`;
  code.textContent = content;
  pre.appendChild(code);

  if (typeof hljs !== 'undefined') {
    try {
      hljs.highlightElement(code);
    } catch {
      // Fall back to plain text
    }
  }

  return pre;
}

/**
 * Render an ASCII flow diagram as a styled pre block.
 * @param {string} diagramText
 * @returns {HTMLDivElement}
 */
export function renderFlowDiagram(diagramText) {
  const section = document.createElement('div');
  section.className = 'prompts-flow-diagram';

  const heading = document.createElement('div');
  heading.className = 'prompts-flow-heading';
  heading.textContent = 'FLOW';
  section.appendChild(heading);

  const pre = document.createElement('pre');
  pre.className = 'prompts-flow-pre';
  const code = document.createElement('code');
  code.textContent = diagramText;
  pre.appendChild(code);
  section.appendChild(pre);

  return section;
}

/**
 * Create a "View Source" collapsible toggle with syntax-highlighted code.
 * @param {string} sourceCode
 * @param {string} [language]
 * @returns {HTMLDivElement}
 */
export function renderSourceToggle(sourceCode, language) {
  const container = document.createElement('div');
  container.className = 'prompts-source-section';

  const toggle = document.createElement('button');
  toggle.className = 'prompts-source-toggle';
  toggle.innerHTML = '<span class="prompts-source-arrow">▶</span> VIEW SOURCE';
  container.appendChild(toggle);
  const arrow = /** @type {HTMLElement} */ (toggle.querySelector('.prompts-source-arrow'));

  const content = document.createElement('div');
  content.className = 'prompts-source-content';
  content.appendChild(renderHighlightedCode(sourceCode, language));
  container.appendChild(content);

  toggle.addEventListener('click', () => {
    const expanded = content.classList.toggle('expanded');
    arrow.textContent = expanded ? '▼' : '▶';
  });

  return container;
}

/**
 * Render front matter fields as a styled key-value table.
 * @param {Record<string, string>} fields
 * @returns {HTMLDivElement}
 */
export function renderFrontMatterTable(fields) {
  const table = document.createElement('div');
  table.className = 'prompts-frontmatter';

  for (const [key, value] of Object.entries(fields)) {
    const row = document.createElement('div');
    row.className = 'prompts-fm-row';

    const keyEl = document.createElement('span');
    keyEl.className = 'prompts-fm-key';
    keyEl.textContent = key;

    const valEl = document.createElement('span');
    valEl.className = 'prompts-fm-value';

    if (key === 'disallowedTools' || key === 'tools') {
      const tools = value.split(',').map((t) => t.trim()).filter(Boolean);
      for (const tool of tools) {
        const pill = document.createElement('span');
        pill.className = 'prompts-fm-pill';
        pill.textContent = tool;
        valEl.appendChild(pill);
      }
    } else if (key === 'model' || key === 'event') {
      const pill = document.createElement('span');
      pill.className = 'prompts-fm-pill prompts-fm-pill-accent';
      pill.textContent = value;
      valEl.appendChild(pill);
    } else if (key === 'safety_layer') {
      const pill = document.createElement('span');
      pill.className = 'prompts-fm-pill prompts-fm-pill-shield';
      pill.textContent = '🛡️ Layer ' + value;
      valEl.appendChild(pill);
    } else {
      valEl.textContent = value;
    }

    row.appendChild(keyEl);
    row.appendChild(valEl);
    table.appendChild(row);
  }

  return table;
}
