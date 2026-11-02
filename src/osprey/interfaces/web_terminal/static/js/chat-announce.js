// @ts-check
/**
 * OSPREY Web Terminal — operator chat: screen-reader announcements.
 *
 * The message log is not a live region: a streamed reply would be read out
 * word by word. This module owns the one polite region the chat speaks
 * through, and says five things, each once, and only when it happens live:
 * the start of a finished reply (at turn end), a tool approval or plan that
 * needs the operator, a question from the agent, an error, and a notice the
 * console put in the log (a failed send or export). History drawn by a replay
 * is never spoken.
 *
 * @module chat-announce
 */

import { elem } from './chat-dom.js';
import { onChat } from './chat-events.js';

/** How much of a reply is spoken: the lead, cut at a word near this length. */
const LEAD_CHARS = 200;

/**
 * The start of a reply as plain text: markdown marks, links and fenced code
 * are reduced to what a listener needs, and the text is cut near LEAD_CHARS.
 * @param {string} markdown
 * @returns {string}
 */
export function replyLead(markdown) {
  const plain = markdown
    .replace(/```[\s\S]*?(```|$)/g, ' ')
    .replace(/!?\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/`([^`]*)`/g, '$1')
    .replace(/^\s{0,3}(#{1,6}\s+|[-*+]\s+|\d+\.\s+|>\s?)/gm, '')
    .replace(/\*+|~~/g, '')
    .replace(/\s+/g, ' ')
    .trim();
  if (plain.length <= LEAD_CHARS) return plain;
  const cut = plain.lastIndexOf(' ', LEAD_CHARS);
  return `${plain.slice(0, cut > 0 ? cut : LEAD_CHARS)}…`;
}

/**
 * Start the announcements for one chat. The region goes in *container*, the
 * console's root, visually hidden by the `.op-announce` rule.
 * @param {{ container: HTMLElement }} ctx
 * @returns {() => void} stop announcing
 */
export function initChatAnnounce({ container }) {
  const region = elem('div', 'op-announce');
  region.setAttribute('aria-live', 'polite');
  region.setAttribute('aria-atomic', 'true');
  container.append(region);

  /** @param {string} text */
  const say = (text) => {
    region.textContent = text;
  };

  /** The live turn's text blocks, joined the way chat-export.js copies them. */
  let blocks = /** @type {string[]} */ ([]);
  /** The text block being written, or -1 when none is open. */
  let open = -1;
  /** Whether the open block is being built from streamed pieces. */
  let streamed = false;
  /** The reply of a turn that just ended, until a terminal error can override it. */
  let pending = /** @type {string | null} */ (null);
  /**
   * What each announced request asked, by request id: so a card re-delivered
   * on reconnect speaks once, and an ask the server's deadline declined can be
   * named when it expires.
   * @type {Map<string, { kind: string, toolName: string }>}
   */
  const asked = new Map();

  /**
   * Follow the reply's text the way the renderer draws it: streamed pieces
   * build a block, a whole text block replaces them, and a thinking or tool
   * row ends the block. A subagent's text is not the operator's reply.
   * @param {Record<string, any>} event
   */
  function trackReply(event) {
    if (event.parent_tool_use_id) return;
    if (event.type === 'delta' && event.kind !== 'thinking') {
      if (open < 0) {
        blocks.push('');
        open = blocks.length - 1;
      }
      streamed = true;
      blocks[open] += event.content ?? '';
      return;
    }
    if (event.type === 'text') {
      if (open < 0) {
        blocks.push('');
        open = blocks.length - 1;
      }
      const chunk = event.content ?? '';
      blocks[open] = streamed ? chunk : blocks[open] + chunk;
      open = -1;
      streamed = false;
      return;
    }
    if (['delta', 'thinking', 'tool_use', 'permission_request'].includes(event.type)) {
      open = -1;
      streamed = false;
    }
  }

  /** Take the turn's reply text, and forget the turn. */
  function takeReply() {
    const text = blocks.filter((part) => part.trim()).join('\n\n');
    blocks = [];
    open = -1;
    streamed = false;
    return text.trim() ? text : null;
  }

  /**
   * An approval or question card has appeared. A tool approval names the tool,
   * a plan is named as the plan, and a question is named as a question.
   * @param {Record<string, any>} request
   */
  function askOnce(request) {
    const id = String(request.request_id ?? '');
    if (id && asked.has(id)) return;
    const kind = String(request.kind ?? 'tool');
    const toolName = String(request.tool_name || request.tool_name_raw || 'this tool');
    asked.set(id, { kind, toolName });
    if (kind === 'question') {
      say('Question from the agent');
    } else if (kind === 'plan') {
      say('Approval needed: plan');
    } else {
      say(`Approval needed: ${toolName}`);
    }
  }

  /**
   * Nobody answered a card within the server's deadline, so it declined the
   * ask. Spoken once, by what the ask was about.
   * @param {Record<string, any>} resolved - the `permission_resolved` frame
   */
  function expiredOnce(resolved) {
    const id = String(resolved.request_id ?? '');
    const ask = asked.get(id);
    if (!ask) return;
    asked.delete(id);
    if (ask.kind === 'question') say('No answer. The question was skipped.');
    else if (ask.kind === 'plan') say('No answer. Kept planning.');
    else say(`No answer. Denied: ${ask.toolName}.`);
  }

  const stops = [
    onChat('frame', ({ event, replay }) => {
      if (replay) return;
      trackReply(event);
      if (event.type === 'permission_request') {
        askOnce(event);
      } else if (event.type === 'permission_resolved' && event.expired) {
        expiredOnce(event);
      } else if (event.type === 'error') {
        // A terminal error ends the turn before this frame is published; its
        // message is the news, so the turn's reply is not spoken after it.
        if (event.error_type !== 'AssistantMessageError') pending = null;
        say(`Error: ${event.message ?? 'An error occurred.'}`);
      }
    }),
    onChat('turn-end', () => {
      const reply = takeReply();
      if (!reply) return;
      pending = reply;
      // Spoken after the turn's last frame has been published, so a terminal
      // error that ends the same turn can cancel it.
      queueMicrotask(() => {
        if (pending !== reply) return;
        pending = null;
        say(`Agent replied: ${replyLead(reply)}`);
      });
    }),
    onChat('agent-message', ({ markdown, replay }) => {
      if (replay) return;
      blocks = [];
      say(`Agent replied: ${replyLead(markdown)}`);
    }),
    onChat('notice', ({ text }) => say(text)),
    onChat('reset', () => {
      blocks = [];
      open = -1;
      streamed = false;
      pending = null;
      asked.clear();
    }),
  ];

  return () => {
    for (const stop of stops) stop();
  };
}
