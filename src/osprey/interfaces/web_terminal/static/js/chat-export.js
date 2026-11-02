// @ts-check
/* OSPREY Web Terminal — operator chat: copy a reply, export the conversation.
 *
 * Two things an operator takes out of the chat, both as Markdown.
 *
 * A reply. Each finished agent reply gets a copy control, shown on hover and on
 * keyboard focus. It copies the reply's markdown source, not its rendered HTML.
 * The renderer keeps that source only while it draws a turn, so this module
 * rebuilds it from the lifecycle bus (chat-events.js) as the frames arrive. The
 * rendered text is the fallback when no source was captured.
 *
 * The conversation. The Export button in the Agent bar downloads the whole
 * transcript. It is read from the server with fetchHistory, not taken from the
 * log, so it is complete even when the page joined the conversation part-way.
 * Each tool call is one line with its decision and outcome, and no payloads.
 *
 * Text from the wire goes in with textContent, and nothing here writes markup.
 *
 * @module chat-export
 */

import { facilityDayKey } from '/design-system/js/facility-time.js';

import { fetchHistory } from './chat-client.js';
import { barButton } from './chat-console.js';
import { endsTurn } from './chat-frames.js';
import { emitChat, onChat } from './chat-events.js';
import { button, elem } from './chat-dom.js';
import { chatIcon } from './chat-icons.js';
import { systemFrame } from './chat-system.js';
import { HIDDEN_TOOL_KEYS, QUESTION_TOOL_KEY, toolKey } from './chat-tool-vocab.js';

/** @typedef {import('./chat-client.js').ChatTurn} ChatTurn */
/** @typedef {import('./chat-features.js').ChatFeatureContext} ChatFeatureContext */

/** How long the copy control says "Copied", in ms. */
const COPIED_MS = 1500;

/** How long an object URL stays alive after the download starts, in ms. */
const REVOKE_MS = 1000;

/**
 * The markdown of one agent reply: its text blocks in order. `open` is set
 * while a streamed block is still being written, so the next whole text block
 * replaces it rather than following it, as the renderer does.
 * @typedef {{ blocks: string[], open: boolean }} ReplySource
 */

/**
 * Whether a main-agent `tool_use` ends the reply's open text block. Mirrors the
 * renderer: a hidden or question tool does not, and every other tool does.
 * @param {Record<string, any>} event
 * @returns {boolean}
 */
function toolEndsText(event) {
  const key = toolKey(event.tool_name_raw ?? event.tool_name);
  return key !== QUESTION_TOOL_KEY && !HIDDEN_TOOL_KEYS.has(key);
}

/**
 * Whether a frame ends a reply: one that ends the turn, or a session reset
 * that drops it.
 * @param {Record<string, any>} event  a history record, shaped as a ChatEvent
 * @returns {boolean}
 */
function endsReply(event) {
  return (
    event.type === 'session_reset' ||
    endsTurn(/** @type {import('./chat-client.js').ChatEvent} */ (event))
  );
}

/**
 * Hand a Markdown file to the browser as a download.
 * @param {string} name
 * @param {string} text
 */
function downloadText(name, text) {
  const url = URL.createObjectURL(new Blob([text], { type: 'text/markdown;charset=utf-8' }));
  const link = document.createElement('a');
  link.href = url;
  link.download = name;
  document.body.append(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), REVOKE_MS);
}

/**
 * Start the copy control on finished replies and the export button in the
 * session bar. Called once by initChatFeatures.
 * @param {Pick<ChatFeatureContext, 'bar' | 'messages' | 'getKey'>} ctx
 */
export function initChatExport(ctx) {
  /**
   * The source of each agent entry. A WeakMap, so a cleared log drops them.
   * @type {WeakMap<HTMLElement, ReplySource>}
   */
  const sources = new WeakMap();
  /** Entries that already carry their copy control. @type {WeakSet<HTMLElement>} */
  const offered = new WeakSet();

  /**
   * The newest agent entry in the log. It walks back from the end, so the
   * per-frame calls do not scan the whole conversation.
   * @returns {HTMLElement | null}
   */
  function lastAgentEntry() {
    for (let node = ctx.messages.lastElementChild; node; node = node.previousElementSibling) {
      if (
        node instanceof HTMLElement &&
        node.classList.contains('op-entry') &&
        node.classList.contains('assistant')
      ) {
        return node;
      }
    }
    return null;
  }

  /**
   * @param {HTMLElement} entry
   * @returns {ReplySource}
   */
  function sourceOf(entry) {
    let source = sources.get(entry);
    if (!source) {
      source = { blocks: [], open: false };
      sources.set(entry, source);
    }
    return source;
  }

  /**
   * Add main-agent text to the newest reply, placed as the renderer places it:
   * a piece extends the open block, and a whole block replaces the streamed
   * block it ends or starts a block of its own.
   * @param {string} chunk
   * @param {boolean} piece - a streamed piece rather than a whole block
   */
  function addText(chunk, piece) {
    const entry = lastAgentEntry();
    if (!entry) return;
    const source = sourceOf(entry);
    if (piece) {
      if (!source.open) {
        source.blocks.push('');
        source.open = true;
      }
      source.blocks[source.blocks.length - 1] += chunk;
    } else if (source.open) {
      source.blocks[source.blocks.length - 1] = chunk;
      source.open = false;
    } else {
      source.blocks.push(chunk);
    }
  }

  /** Something other than text came next: the newest reply's open block ends. */
  function closeText() {
    const entry = lastAgentEntry();
    const source = entry ? sources.get(entry) : undefined;
    if (source) source.open = false;
  }

  /**
   * The reply as the copy control gives it: its markdown source, or the
   * rendered text when no source was captured.
   * @param {HTMLElement} entry
   * @returns {string}
   */
  function replyText(entry) {
    const source = (sources.get(entry)?.blocks ?? []).filter((block) => block.trim()).join('\n\n');
    if (source) return source;
    return [...entry.querySelectorAll('.op-entry-body')]
      .map((node) => node.textContent?.trim() ?? '')
      .filter(Boolean)
      .join('\n\n');
  }

  /**
   * The copy control of one finished reply. It copies the reply when it is
   * pressed, and says "Copied" for a moment. Its visible label is its
   * accessible name, so the change is exposed; the title explains the format.
   * A clipboard failure does nothing.
   * @param {HTMLElement} entry
   * @returns {HTMLButtonElement}
   */
  function copyControl(entry) {
    const control = button('op-reply-copy');
    control.title = 'Copy the reply as Markdown';
    const label = elem('span', 'op-reply-copy-label', 'Copy');
    control.append(chatIcon('copy'), label);
    let resetTimer = 0;
    control.addEventListener('click', async () => {
      const text = replyText(entry);
      if (!text) return;
      try {
        await navigator.clipboard.writeText(text);
      } catch {
        return; // no clipboard here: the control does nothing
      }
      label.textContent = 'Copied';
      control.classList.add('copied');
      window.clearTimeout(resetTimer);
      resetTimer = window.setTimeout(() => {
        label.textContent = 'Copy';
        control.classList.remove('copied');
      }, COPIED_MS);
    });
    return control;
  }

  /**
   * Give a finished reply its copy control, once, when it has text to copy.
   * @param {HTMLElement | null} entry
   */
  function offerCopy(entry) {
    if (!entry || offered.has(entry) || !replyText(entry)) return;
    offered.add(entry);
    entry.append(copyControl(entry));
  }

  /** The newest reply has ended: close its text and offer its copy control. */
  function finishReply() {
    closeText();
    offerCopy(lastAgentEntry());
  }

  // ---- The reply's source, from the bus ---- //

  onChat('frame', ({ event }) => {
    // A subagent's frames belong to its chip, not to the reply.
    if (event.parent_tool_use_id) return;
    switch (event.type) {
      case 'text':
        addText(String(event.content ?? ''), false);
        break;
      case 'delta':
        if (event.kind === 'thinking') closeText();
        else addText(String(event.content ?? ''), true);
        break;
      case 'thinking':
      case 'permission_request':
        closeText();
        break;
      case 'tool_use':
        if (toolEndsText(event)) closeText();
        break;
      case 'system':
        if (systemFrame(event)?.closeText) closeText();
        break;
      default:
        break;
    }
    if (endsReply(event)) finishReply();
  });

  // A whole agent message from history arrives without frames.
  onChat('agent-message', ({ markdown }) => {
    const entry = lastAgentEntry();
    if (!entry) return;
    sources.set(entry, { blocks: [markdown], open: false });
    offerCopy(entry);
  });

  // A running turn that ends for any reason, live or replayed.
  onChat('turn-end', () => finishReply());

  // ---- The conversation, as a file ---- //

  const exportButton = barButton('op-export', 'share', 'Export');
  exportButton.title = 'Export: download the conversation as Markdown';
  exportButton.addEventListener('click', () => void exportConversation());
  ctx.bar.append(exportButton);

  async function exportConversation() {
    const key = ctx.getKey();
    exportButton.disabled = true;
    try {
      const now = new Date();
      const turns = await fetchHistory(key);
      downloadText(exportFilename(key, now), conversationMarkdown(turns, { key, exportedAt: now }));
    } catch {
      const text = 'Could not export the conversation.';
      ctx.messages.append(elem('div', 'op-system', text));
      emitChat('notice', { text });
    } finally {
      exportButton.disabled = false;
    }
  }
}

/**
 * The download's file name: the date on the facility clock, and the first eight
 * characters of the session key.
 * @param {string} key
 * @param {Date} when
 * @returns {string}
 */
export function exportFilename(key, when) {
  return `osprey-conversation-${facilityDayKey(when)}-${key.slice(0, 8)}.md`;
}

/**
 * One tool call on one line: its name, its decision when one was asked, and
 * its outcome. The call's input and output are left out.
 * @param {Record<string, any>} call - a `tool_use` event
 * @param {Record<string, any> | undefined} result - its `tool_result` event
 * @returns {string}
 */
function toolLine(call, result) {
  const name = String(call.tool_name_raw ?? call.tool_name ?? 'tool');
  const decision = result?.decision ? ` · decision: ${result.decision}` : '';
  const outcome = !result ? 'no result' : result.is_error ? 'error' : 'ok';
  return `- Tool call: \`${name}\`${decision} · outcome: ${outcome}`;
}

/**
 * The parts of an agent turn in order: its text, and one line per tool call.
 * Without events, the turn's content is the whole reply.
 * @param {ChatTurn} turn
 * @returns {string[]}
 */
function replyParts(turn) {
  const events = Array.isArray(turn.events) ? turn.events : [];
  if (events.length === 0) {
    const text = (turn.content ?? '').trim();
    return text ? [text] : [];
  }
  /** @type {Map<string, Record<string, any>>} */
  const results = new Map();
  for (const event of events) {
    if (event.type === 'tool_result' && !event.parent_tool_use_id && event.tool_use_id) {
      results.set(String(event.tool_use_id), event);
    }
  }
  /** @type {string[]} */
  const parts = [];
  for (const event of events) {
    if (event.parent_tool_use_id) continue;
    if (event.type === 'text') {
      const text = String(event.content ?? '').trim();
      if (text) parts.push(text);
    } else if (event.type === 'tool_use') {
      parts.push(toolLine(event, results.get(String(event.tool_use_id ?? ''))));
    }
  }
  return parts;
}

/**
 * One turn as a Markdown section, or '' when the turn has nothing to show.
 * @param {ChatTurn} turn
 * @returns {string}
 */
function turnSection(turn) {
  const stamp = turn.timestamp ? `_${turn.timestamp}_` : '';
  if ((turn.role ?? '').toLowerCase() === 'user') {
    const text = (turn.content ?? '').trim();
    if (!text) return '';
    return ['## Operator', stamp, text].filter(Boolean).join('\n\n');
  }
  const parts = replyParts(turn);
  if (parts.length === 0) return '';
  return ['## Agent', stamp, ...parts].filter(Boolean).join('\n\n');
}

/**
 * The conversation as one Markdown document, turn by turn.
 * @param {ChatTurn[]} turns - the transcript, oldest first
 * @param {{ key: string, exportedAt: Date }} meta
 * @returns {string}
 */
export function conversationMarkdown(turns, { key, exportedAt }) {
  const sections = turns.map(turnSection).filter((section) => section !== '');
  const header = [
    '# Osprey conversation',
    `Session: \`${key}\``,
    `Exported: ${exportedAt.toISOString()}`,
  ];
  const body = sections.length > 0 ? sections : ['_No messages in this conversation yet._'];
  return `${[...header, ...body].join('\n\n')}\n`;
}
