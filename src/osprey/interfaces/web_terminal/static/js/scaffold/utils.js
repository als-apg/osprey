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
 * Help text for each artifact category, shown in the header tooltip.
 *
 * These describe the LOADING MECHANICS of each category, deliberately not what
 * the shipped files happen to say. The content is the operator's to change --
 * a blurb about "safety boundaries and error-handling protocols" goes stale the
 * moment someone edits a rule, while "loads at the start of every session"
 * stays true whatever they write.
 *
 * Keyed by DISPLAY category, so the keys must line up with the values in
 * {@link BEHAVIOR_CATEGORY_OVERRIDES}/{@link BEHAVIOR_CATEGORY_REMAPS} below
 * rather than with the raw `category` field on an artifact. Both tables live
 * in this module so that alignment is checkable in one place (a key that
 * matches no reachable display category renders no help button at all, silently
 * -- which is exactly how the CLAUDE.md section went without one).
 *
 * @type {Record<string, string>}
 */
export const CATEGORY_HELP = {
  'project instructions': "The project's CLAUDE.md. Read at the start of every session and injected as a message after the system prompt, so it is context the agent reads rather than a rule the system enforces. It stays for the whole session and is re-read after a context compaction.",
  instructions: 'Markdown files in .claude/rules/. Each one loads at the start of every session, at the same priority as CLAUDE.md, and stays in context throughout. A file with a paths: header loads only when the agent opens a matching file. Like CLAUDE.md, these are read as context, not enforced.',
  agents: "Subagent definitions in .claude/agents/. Each runs in its own context window with its own instructions, model, and tool list. The agent delegates when a task matches a subagent's description and gets back only that subagent's final answer — the subagent's own work never enters this session.",
  skills: 'Folders under .claude/skills/, each with a SKILL.md. Only the name and description load at session start; the body loads when the skill runs — invoked by you with /name, or by the agent when the description matches the task. Companion files in the folder are read on demand.',
  'output-styles': 'Markdown files in .claude/output-styles/. An output style is appended to the system prompt itself. Exactly one is active at a time, selected by the outputStyle setting, and a change takes effect in the next session. Unless the file sets keep-coding-instructions: true, it replaces the built-in software-engineering instructions. Output styles do not reach subagents.',
  hooks: "Programs wired to session lifecycle events in settings.json. The runtime runs them itself, not the agent, so a PreToolUse hook can block a tool call before it executes, independent of the agent's decision. This is the only layer in this drawer that enforces rather than advises.",
  config: ".mcp.json declares which MCP servers start with the session; their tools become the agent's tools. settings.json holds the permission lists, the hook wiring, and the model and output-style selection. Both are read at session start; edits apply on the next session.",
};

// ---- Category Routing ---- //

export const BEHAVIOR_CATEGORIES = new Set(['agents', 'skills', 'rules', 'output-styles']);
export const BEHAVIOR_NAMES = new Set(['claude-md']);        // config category, behavior tab
export const SAFETY_CATEGORIES = new Set(['hooks']);
export const CONFIG_NAMES = new Set(['mcp-json', 'settings-json']); // config category, config tab

/**
 * Behavior-tab display-category tables, consumed by initScaffoldGallery.
 *
 * `claude-md` has no "/" in its canonical name, so the service reports it in
 * the catch-all `config` category; the override gives it a section of its own.
 * It is deliberately NOT called "system prompt": CLAUDE.md is delivered as a
 * message after the system prompt, and the one artifact here that really does
 * modify the system prompt is the output style.
 *
 * @type {Record<string, string>}
 */
export const BEHAVIOR_CATEGORY_OVERRIDES = { 'claude-md': 'project instructions' };

/** @type {Record<string, string>} */
export const BEHAVIOR_CATEGORY_REMAPS = { rules: 'instructions' };

/** @type {string[]} */
export const BEHAVIOR_PINNED_CATEGORIES = ['project instructions', 'instructions'];

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
    case 'project instructions': return '📜'; // scroll
    case 'instructions':   return '📋';  // clipboard
    case 'agents':         return '🤖';  // robot
    case 'hooks':          return '⚡';  // lightning
    case 'output-styles':  return '🎨';  // palette
    case 'config':         return '⚙';   // gear
    case 'skills':         return '📦';  // package
    default:               return '📄';  // document
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

// ---- Protected-Set Affordances ---- //

/**
 * Why the panel will not save some of the files it happily shows.
 *
 * The server is the enforcement — `is_reserved_write` refuses these writes
 * whatever the client believes, and `read_only` on a listed artifact is that
 * same call's answer, precomputed. This string is the panel's half of the
 * bargain: the operator meets the fact while reading, on the badge and on the
 * control that is greyed out, instead of only when a save comes back 403.
 *
 * Panel copy, deliberately not a mirror of the server's PROFILE_EDIT_NOTICE:
 * that one is written for a whole-panel banner and enumerates the paths,
 * while this one is attached to a specific artifact the reader is already
 * looking at, so it says what to do rather than which files are affected.
 */
export const READ_ONLY_REASON =
  'Rendered by the build profile. Edit it in the profile and rebuild the '
  + 'project — a save aimed at it here is refused, because a profile that no '
  + 'longer describes the project it built is worse than an edit that did not '
  + 'happen.';

/**
 * Build the READ-ONLY badge shown on a reserved artifact's card and in its
 * detail header. Carries {@link READ_ONLY_REASON} as its tooltip so the badge
 * answers "why" on hover rather than just asserting the state.
 * @returns {HTMLSpanElement}
 */
export function createReadOnlyBadge() {
  const badge = document.createElement('span');
  badge.className = 'prompts-badge read-only';
  badge.textContent = 'READ-ONLY';
  badge.title = READ_ONLY_REASON;
  return badge;
}

/**
 * Give a card the full reserved-artifact treatment: the left-border tint
 * (10-prompt-gallery.css) and the badge.
 *
 * One call rather than two, so the tint and the badge cannot drift apart —
 * a tinted card with no badge asserts a state it never explains, and a badged
 * card with no tint is not separable at a glance down a long grid.
 * {@link createReadOnlyBadge} stays exported for the detail header, which
 * wants the badge without the card class.
 *
 * @param {HTMLElement} card Card element for a reserved artifact.
 */
export function markReadOnlyCard(card) {
  card.classList.add('prompts-card-readonly');
  card.appendChild(createReadOnlyBadge());
}

/**
 * Lock an editor textarea for a reserved artifact.
 *
 * The Edit tab is disabled for reserved artifacts, so this is the second lock
 * rather than the first — it holds for any path that reaches the edit view
 * anyway, and makes the state visible instead of merely refused. Callers own
 * anything else the form needs disabled alongside it.
 *
 * @param {HTMLTextAreaElement} textarea Editor the panel may not write.
 */
export function lockEditor(textarea) {
  textarea.readOnly = true;
  textarea.classList.add('prompts-edit-readonly');
}
