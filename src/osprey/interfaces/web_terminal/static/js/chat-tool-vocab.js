// @ts-check
/**
 * OSPREY Web Terminal — Simple-mode operator chat: tool vocabulary.
 *
 * What the turn's blocks call a tool: the names of the plan and question tools,
 * the one-line summary of a tool input, the key a tool is recognised by, the
 * category and icon family it files under, which control calls are writes,
 * and what a run of calls is called.
 *
 * @module chat-tool-vocab
 */

/** @typedef {import('./chat-icons.js').ChatIconName} ChatIconName */

/**
 * The tools that keep the agent's task list: the whole-list `TodoWrite`, and the
 * one-task-at-a-time `TaskCreate` / `TaskUpdate` with their reads. They get no
 * chip: the task bar above the composer (chat-tasks.js) shows what they did.
 */
export const TASK_TOOL_KEYS = new Set([
  'todowrite',
  'taskcreate',
  'taskupdate',
  'tasklist',
  'taskget',
]);

/** The tool behind a question card. It gets no chip: the card stands in for it. */
export const QUESTION_TOOL_KEY = 'askuserquestion';

/**
 * Tools that get no chip: the tool search the CLI runs to load another tool's
 * schema is bookkeeping, entering plan mode only changes how the CLI works
 * (its result is the CLI's own instructions), the plan-mode exit is drawn as
 * its own card, and the task tools are drawn as the task bar. The renderer and
 * the export both leave an open text block open across their frames.
 */
export const HIDDEN_TOOL_KEYS = new Set([
  'toolsearch',
  'enterplanmode',
  'exitplanmode',
  ...TASK_TOOL_KEYS,
]);

/** Keys read first for a one-line input summary, in order of preference. */
const SUMMARY_KEYS = [
  'skill',
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
 * @typedef {'read' | 'search' | 'edit' | 'bash' | 'mcp-control' | 'web' | 'skill' | 'other'} ToolCategory
 */

/** @type {Readonly<Record<ToolCategory, ChatIconName>>} */
export const CATEGORY_ICON = Object.freeze({
  read: 'read',
  search: 'search',
  edit: 'edit',
  bash: 'bash',
  'mcp-control': 'control',
  web: 'web',
  skill: 'skill',
  other: 'wrench',
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
    case 'skill':
      return 'skill';
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

/**
 * The control calls that act on the machine, by {@link toolKey}: a channel
 * write, the plan queue's starts and stops, a Phoebus display driven. A write
 * reads heavier than a read and never folds into a run.
 */
const CONTROL_WRITE_KEYS = new Set([
  'channelwrite',
  'queueadd',
  'queuestart',
  'queuestop',
  'stoprun',
  'phoebusdrive',
]);

/**
 * Whether a tool is a control-system write.
 * @param {string | undefined} rawName
 * @returns {boolean}
 */
export function isControlWrite(rawName) {
  return toolCategory(rawName) === 'mcp-control' && CONTROL_WRITE_KEYS.has(toolKey(rawName));
}

/** @typedef {ToolCategory | 'control-write'} FoldKey */

/**
 * The kind consecutive calls fold under: the tool's category, with a control
 * write marked as its own kind (one that never folds).
 * @param {string | undefined} rawName
 * @returns {FoldKey}
 */
export function foldKey(rawName) {
  return isControlWrite(rawName) ? 'control-write' : toolCategory(rawName);
}

/** What any run of calls can be called. @type {readonly [string, string]} */
const ANY_RUN_PHRASE = Object.freeze(['Made', 'tool call']);

/**
 * What a run of calls is called, by fold key: a past-tense verb and the
 * singular noun it counts. A control write never folds, so it has no phrase.
 * @type {Readonly<Partial<Record<FoldKey, readonly [string, string]>>>}
 */
const RUN_PHRASE = Object.freeze({
  read: ['Read', 'file'],
  search: ['Searched', 'pattern'],
  edit: ['Edited', 'file'],
  bash: ['Ran', 'command'],
  web: ['Fetched', 'page'],
  'mcp-control': ['Made', 'control call'],
  skill: ['Used', 'skill'],
  other: ANY_RUN_PHRASE,
});

/**
 * The phrase for a run made of one tool, by {@link toolKey}, where the noun is
 * that tool's own. A run of mixed tools keeps its fold key's phrase.
 * @type {Readonly<Record<string, readonly [string, string]>>}
 */
const TOOL_RUN_PHRASE = Object.freeze({
  channelread: ['Read', 'channel'],
  archiverread: ['Read', 'archive range'],
  read: ['Read', 'file'],
  notebookread: ['Read', 'notebook'],
  grep: ['Searched', 'pattern'],
  glob: ['Matched', 'pattern'],
  edit: ['Edited', 'file'],
  multiedit: ['Edited', 'file'],
  write: ['Wrote', 'file'],
  bash: ['Ran', 'command'],
  execute: ['Ran', 'Python block'],
  webfetch: ['Fetched', 'page'],
  websearch: ['Ran', 'web search'],
});

/**
 * Whether a tool call's input says the operator ran it by a slash command,
 * not the agent by its own choice. Only a skill's input carries this.
 * @param {unknown} input
 * @returns {boolean}
 */
export function ranByOperator(input) {
  return Boolean(
    input && typeof input === 'object' && /** @type {{ by?: unknown }} */ (input).by === 'operator'
  );
}

/**
 * The label of a run row: "Read 4 channels", "Ran 2 commands". A run of one
 * tool is phrased by that tool when it has a phrase of its own; otherwise by
 * its fold key. A run of skills says "Ran" when the operator ran every one of
 * them, as their rows do, and "Used" when the agent chose any.
 * @param {FoldKey} key
 * @param {readonly string[]} toolKeys - the members' {@link toolKey}s, in order
 * @param {boolean} [allByOperator] - every member was run by the operator
 * @returns {string}
 */
export function runPhrase(key, toolKeys, allByOperator = false) {
  const n = toolKeys.length;
  const one = n > 0 && toolKeys.every((k) => k === toolKeys[0]);
  const own =
    one && Object.hasOwn(TOOL_RUN_PHRASE, toolKeys[0]) ? TOOL_RUN_PHRASE[toolKeys[0]] : null;
  const [verb, noun] = own || RUN_PHRASE[key] || ANY_RUN_PHRASE;
  const said = key === 'skill' && allByOperator ? 'Ran' : verb;
  return `${said} ${n} ${noun}${n === 1 ? '' : 's'}`;
}

/**
 * The label of one call's row. A skill's row says how it came to be used: the
 * agent chose it ("Used skill"), or the operator ran it by its slash command
 * ("Ran skill"); its name is the row's detail. Any other tool keeps its name.
 * @param {{ tool_name?: string, tool_name_raw?: string, input?: unknown }} event
 * @returns {string}
 */
export function toolLabel(event) {
  const raw = event.tool_name_raw ?? event.tool_name;
  if (toolCategory(raw) === 'skill') {
    return ranByOperator(event.input) ? 'Ran skill' : 'Used skill';
  }
  return String(event.tool_name || event.tool_name_raw || 'tool');
}
