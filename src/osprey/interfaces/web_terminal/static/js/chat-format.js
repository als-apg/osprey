// @ts-check
/**
 * OSPREY Web Terminal — operator chat: how a tool's input and result read.
 *
 * A tool's input and its result are JSON, or text that holds JSON, and an
 * operator reads both. Each field is shown in one of four shapes, and the value
 * alone picks the shape:
 *   - text:  on the field's own line: a short string, a number, a flag, null, or
 *            a list of plain values, joined with commas
 *   - block: a preformatted block with real line breaks: a multi-line string, or
 *            the value of a code-like key, whatever its length
 *   - rows:  a nested object whose own fields are all plain, as indented key and
 *            value rows
 *   - json:  indented JSON, for anything deeper, and for a list that mixes kinds
 *
 * `inputView` draws a tool's input as DOM, for the call's row and its approval
 * card. `formatToolResult` writes a result as text, for the row's RESULT block,
 * by the same rules. Text goes in with textContent; nothing here reaches innerHTML.
 *
 * @module chat-format
 */

import { elem, prettyJson } from './chat-dom.js';
import { KIND_WORDS } from './control-target-facts.js';

/** @typedef {'text' | 'block' | 'rows' | 'json'} FieldShape */

/** Keys whose string value is code: shown as a block, whatever its length. */
const CODE_KEYS = new Set(['code', 'script', 'command', 'query', 'sql', 'patch']);

/** The field the approval hook writes into an approved call: the control target it was approved for. */
const TARGET_KEY = 'approved_target';

/**
 * @param {unknown} value
 * @returns {value is Record<string, unknown>} a JSON object, not an array
 */
function isRecord(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

/**
 * Parse `text` as JSON, or return undefined when it is not JSON.
 * Only objects and arrays count: a bare number or word is left as text.
 * @param {string} text
 * @returns {unknown}
 */
function parseJsonObject(text) {
  const trimmed = text.trim();
  if (!trimmed.startsWith('{') && !trimmed.startsWith('[')) return undefined;
  try {
    return JSON.parse(trimmed);
  } catch {
    return undefined;
  }
}

/**
 * A plain value: a string with no line break, a number, a flag or null.
 * @param {unknown} value
 * @returns {boolean}
 */
function isPlain(value) {
  if (typeof value === 'string') return !value.includes('\n');
  return value === null || typeof value === 'number' || typeof value === 'boolean';
}

/**
 * A value a row holds without nesting: any string, a number, a flag, null, or a
 * list of plain values.
 * @param {unknown} value
 * @returns {boolean}
 */
function isLeaf(value) {
  return Array.isArray(value) ? value.every(isPlain) : !isRecord(value);
}

/**
 * The shape the field `key` holding `value` is shown in.
 * @param {string} key
 * @param {unknown} value
 * @returns {FieldShape}
 */
function shapeOf(key, value) {
  if (typeof value === 'string') {
    const code = CODE_KEYS.has(key.toLowerCase()) && value.trim() !== '';
    return code || value.includes('\n') ? 'block' : 'text';
  }
  if (Array.isArray(value)) return value.every(isPlain) ? 'text' : 'json';
  if (isRecord(value)) {
    const fields = Object.values(value);
    return fields.length > 0 && fields.every(isLeaf) ? 'rows' : 'json';
  }
  return 'text';
}

/**
 * The text a text field shows for its value: a list joins with commas.
 * @param {unknown} value
 * @returns {string}
 */
function textOf(value) {
  return Array.isArray(value) ? value.map((item) => String(item)).join(', ') : String(value);
}

/**
 * What the caller knows about each control target's kind: target key →
 * `machine_kind` (`real` | `rehearsal` | `simulator`), exactly as the chip
 * read it from `GET /api/terminal/posture` (or as a permission frame carried
 * it). Never derived here from the target's name: a `standin` no live server
 * vouched for is the real machine, and only GET knows which one it is.
 * @typedef {Record<string, string>} TargetKinds
 */

/**
 * The name an operator knows a control target by: the word for the kind its
 * caller handed in, else the id as given — a target this module was told
 * nothing about is named, never guessed at.
 * @param {string} target
 * @param {TargetKinds} kinds
 * @returns {string}
 */
function targetLabel(target, kinds) {
  const kind = Object.hasOwn(kinds, target) ? kinds[target] : undefined;
  return typeof kind === 'string' && Object.hasOwn(KIND_WORDS, kind) ? KIND_WORDS[kind] : target;
}

/**
 * The element a field's value is shown in, by its shape.
 * @param {string} key
 * @param {unknown} value
 * @param {TargetKinds} kinds
 * @returns {HTMLElement}
 */
function valueNode(key, value, kinds) {
  const shape = shapeOf(key, value);
  if (shape === 'block') return elem('pre', 'op-kv-block', String(value));
  if (shape === 'rows') {
    const nested = keyValueList(/** @type {Record<string, unknown>} */ (value), kinds);
    nested.classList.add('op-kv-nested');
    return nested;
  }
  if (shape === 'json') return elem('pre', 'op-kv-json', prettyJson(value));
  const text = textOf(value);
  return text === '' ? elem('span', 'op-kv-empty', '(empty)') : elem('span', 'op-kv-text', text);
}

/**
 * A record as key and value rows, in the order the call gave them.
 * @param {Record<string, unknown>} record
 * @param {TargetKinds} kinds
 * @returns {HTMLDListElement}
 */
function keyValueList(record, kinds) {
  const list = /** @type {HTMLDListElement} */ (elem('dl', 'op-kv'));
  for (const [key, value] of Object.entries(record)) {
    const isTarget = key === TARGET_KEY && typeof value === 'string';
    const term = elem('dt', 'op-kv-key', isTarget ? 'Target' : key);
    const detail = elem('dd', 'op-kv-value');
    detail.append(valueNode(key, isTarget ? targetLabel(value, kinds) : value, kinds));
    list.append(term, detail);
  }
  return list;
}

/**
 * A tool's input, drawn as key and value rows. An empty input reads "(no input)".
 * An `approved_target` row reads the word for that target's kind when the
 * caller passes it in `kinds` (the `machine_kind` the chip holds from GET),
 * and the target id as given otherwise.
 * @param {unknown} input
 * @param {TargetKinds} [kinds]
 * @returns {HTMLElement}
 */
export function inputView(input, kinds = {}) {
  if (
    input === undefined ||
    input === null ||
    (isRecord(input) && Object.keys(input).length === 0)
  ) {
    return elem('span', 'op-kv-empty', '(no input)');
  }
  const known = isRecord(kinds) ? /** @type {TargetKinds} */ (kinds) : {};
  return isRecord(input) ? keyValueList(input, known) : valueNode('', input, known);
}

/**
 * Whether the summary line states a field exactly: the field is one line, and the summary is
 * that line. A summary cut short does not, and neither does one that flattens a multi-line
 * field. Such a field stays under Details, so nothing the call sent is lost.
 * @param {unknown} value
 * @param {string} summary
 * @returns {boolean}
 */
function statesInFull(value, summary) {
  /** @type {string | null} */
  let text = null;
  if (typeof value === 'string') text = value;
  else if (Array.isArray(value) && value.every((item) => typeof item === 'string')) {
    text = value.join(', ');
  }
  return summary !== '' && text !== null && !text.includes('\n') && text === summary;
}

/**
 * The part of a tool's input an approval card shows under Details: every field
 * except one the summary line states in full. `null` when nothing is left.
 * @param {unknown} input
 * @param {string} summary - the one-line summary the card shows
 * @returns {unknown}
 */
export function inputDetail(input, summary) {
  if (!isRecord(input)) return input === undefined || input === null ? null : input;
  const rest = Object.fromEntries(
    Object.entries(input).filter(([, value]) => !statesInFull(value, summary))
  );
  return Object.keys(rest).length > 0 ? rest : null;
}

/**
 * A record as text lines, one per field. A text field reads on its key's line,
 * aligned on the longest such key. A nested record, block or JSON field reads
 * under its key, indented.
 * @param {Record<string, unknown>} record
 * @param {string} indent
 * @returns {string[]}
 */
function recordLines(record, indent) {
  const entries = Object.entries(record);
  const width = entries.reduce(
    (widest, [key, value]) =>
      shapeOf(key, value) === 'text' ? Math.max(widest, key.length) : widest,
    0
  );
  /** @type {string[]} */
  const lines = [];
  for (const [key, value] of entries) {
    const shape = shapeOf(key, value);
    if (shape === 'text') {
      lines.push(`${indent}${key.padEnd(width)}  ${textOf(value) || '(empty)'}`);
    } else {
      lines.push(`${indent}${key}`);
      if (shape === 'rows') {
        lines.push(...recordLines(/** @type {Record<string, unknown>} */ (value), `${indent}  `));
      } else {
        const body = shape === 'block' ? String(value) : JSON.stringify(value, null, 2);
        lines.push(...body.split('\n').map((line) => `${indent}  ${line}`));
      }
    }
  }
  return lines;
}

/**
 * The text a tool card shows for a result preview. The server has already
 * peeled the MCP envelope and cut the result (tool_preview.py), so what
 * arrives is the tool's own value: a record reads as aligned `key  value`
 * lines, with a nested record indented under its key, a list of plain values
 * reads on one line, and other JSON comes back indented. Text that is not JSON
 * comes back as it came.
 * @param {string} preview
 * @returns {string}
 */
export function formatToolResult(preview) {
  const value = parseJsonObject(preview);
  if (value === undefined) return preview;
  if (isRecord(value) && Object.keys(value).length > 0) return recordLines(value, '').join('\n');
  if (Array.isArray(value) && value.length > 0 && value.every(isPlain)) return textOf(value);
  return JSON.stringify(value, null, 2);
}
