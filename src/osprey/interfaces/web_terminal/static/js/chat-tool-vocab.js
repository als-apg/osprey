// @ts-check
/**
 * OSPREY Web Terminal — Simple-mode operator chat: tool vocabulary.
 *
 * What the turn's blocks call a tool: the names of the plan and question tools,
 * the one-line summary of a tool input, the key a tool is recognised by, and
 * the category and icon family it files under.
 *
 * @module chat-tool-vocab
 */

/** @typedef {import('./chat-icons.js').ChatIconName} ChatIconName */

/** The tool whose todo list becomes the turn's plan card instead of a chip. */
export const PLAN_TOOL_KEY = 'todowrite';

/** The tool behind a question card. It gets no chip: the card stands in for it. */
export const QUESTION_TOOL_KEY = 'askuserquestion';

/**
 * Tools that get no chip: the tool search the CLI runs to load another tool's
 * schema is bookkeeping, entering plan mode only changes how the CLI works
 * (its result is the CLI's own instructions), and the plan-mode exit is drawn
 * as its own card. The renderer and the export both leave an open text block
 * open across their frames.
 */
export const HIDDEN_TOOL_KEYS = new Set(['toolsearch', 'enterplanmode', 'exitplanmode']);

/** Keys read first for a one-line input summary, in order of preference. */
const SUMMARY_KEYS = [
  'file_path',
  'notebook_path',
  'path',
  'command',
  'pattern',
  'query',
  'url',
  'channels',
  'channel',
  'pvs',
  'names',
  'description',
  'prompt',
];

/** Longest one-line summary before it is cut. */
const SUMMARY_MAX = 160;

/** @param {string} text */
export function oneLine(text) {
  const flat = text.replace(/\s+/g, ' ').trim();
  return flat.length > SUMMARY_MAX ? `${flat.slice(0, SUMMARY_MAX - 1)}…` : flat;
}

/**
 * The one-line summary of a tool input: the first of the keys an operator reads
 * as "what this touches" (a path, a command, a pattern, a URL, channel names),
 * else the first non-empty string. Empty when there is nothing to show.
 * @param {unknown} input
 * @returns {string}
 */
export function summarizeInput(input) {
  if (typeof input === 'string') return oneLine(input);
  if (!input || typeof input !== 'object') return '';
  const record = /** @type {Record<string, unknown>} */ (input);
  for (const key of SUMMARY_KEYS) {
    const value = record[key];
    if (typeof value === 'string' && value.trim()) return oneLine(value);
    if (Array.isArray(value) && value.length > 0 && value.every((v) => typeof v === 'string')) {
      return oneLine(value.join(', '));
    }
  }
  for (const value of Object.values(record)) {
    if (typeof value === 'string' && value.trim()) return oneLine(value);
  }
  return '';
}

/**
 * The bare, lower-case key a tool is recognised by, whichever spelling the
 * event carries: `TodoWrite`, `Todo Write` and `todo_write` all fold to
 * `todowrite`.
 * @param {string | undefined} name
 * @returns {string}
 */
export function toolKey(name) {
  return (name ?? '')
    .replace(/^mcp__.+?__/, '')
    .replace(/[\s_-]+/g, '')
    .toLowerCase();
}

/**
 * @typedef {'read' | 'search' | 'edit' | 'bash' | 'mcp-control' | 'web' | 'other'} ToolCategory
 */

/** @type {Readonly<Record<ToolCategory, ChatIconName>>} */
export const CATEGORY_ICON = Object.freeze({
  read: 'read',
  search: 'search',
  edit: 'edit',
  bash: 'bash',
  'mcp-control': 'control',
  web: 'web',
  other: 'tool',
});

/**
 * The icon family for a tool, inferred from its SDK name. MCP tools are the
 * facility's own control surface and share one family; built-ins map by what
 * they do to the workspace.
 * @param {string | undefined} rawName
 * @returns {ToolCategory}
 */
export function toolCategory(rawName) {
  const raw = rawName ?? '';
  if (raw.startsWith('mcp__')) return 'mcp-control';
  switch (toolKey(raw)) {
    case 'webfetch':
    case 'websearch':
      return 'web';
    case 'bash':
    case 'shell':
      return 'bash';
    case 'edit':
    case 'multiedit':
    case 'write':
    case 'notebookedit':
      return 'edit';
    case 'read':
    case 'notebookread':
      return 'read';
    case 'grep':
    case 'glob':
    case 'ls':
      return 'search';
    default:
      return 'other';
  }
}
