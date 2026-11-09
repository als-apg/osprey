// @ts-check
/**
 * "Ask agent": a reader action (key `a`) that hands the shown entry to the
 * host's assistant prompt.
 *
 * It posts a citation block (entry id, author, date, then the text selected
 * inside the reader, else the subject) as `osprey-paste-to-terminal` to the
 * parent, addressed to this page's own origin so a host on another origin is
 * never handed the operator's text (as `explore-graph.js` does). The host
 * only inserts it into the prompt; nothing is sent, and ARIEL raises no
 * toast because the host owns the feedback.
 */

import { formatTimestamp } from './components.js';
import { parseEntryText } from './entries-helpers.js';
import { registerReaderAction } from './reader.js';

/**
 * The citation block for an entry.
 * @param {any} entry
 * @param {string} selection - Text selected inside the reader, or `''`
 * @returns {string}
 */
export function citation(entry, selection) {
  const subject = entry.subject || parseEntryText(entry.raw_text).subject || '';
  const quoted = (selection.trim() || subject)
    .split('\n')
    .map((/** @type {string} */ line) => `> ${line}`)
    .join('\n');
  const date = formatTimestamp(entry.timestamp);
  return `Logbook entry ${entry.entry_id} — ${entry.author || 'Unknown'}, ${date}\n${quoted}\n`;
}

/**
 * The document's selected text when the whole selection lies inside `container`.
 * @param {HTMLElement} container
 * @returns {string}
 */
function selectionIn(container) {
  const sel = container.ownerDocument.defaultView?.getSelection();
  if (!sel || sel.isCollapsed || sel.rangeCount === 0) return '';
  const { anchorNode, focusNode } = sel;
  if (!anchorNode || !focusNode) return '';
  if (!container.contains(anchorNode) || !container.contains(focusNode)) return '';
  return sel.toString().trim();
}

/** @type {import('./reader-seam.js').ReaderAction} */
export const askAgentAction = {
  id: 'ask-agent',
  label: 'Ask agent',
  key: 'a',
  title: 'Cite this entry in the assistant prompt (a)',
  run(ctx) {
    const text = citation(ctx.entry, selectionIn(ctx.container));
    window.parent.postMessage({ type: 'osprey-paste-to-terminal', text }, window.location.origin);
  },
};

/** Register the action. */
export function registerAskAgent() {
  registerReaderAction(askAgentAction);
}
