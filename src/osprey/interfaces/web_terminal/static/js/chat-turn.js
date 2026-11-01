// @ts-check
/**
 * OSPREY Web Terminal — operator chat: the agent turn in flight.
 *
 * One agent entry and the block it is writing into: the text, the reasoning,
 * the tool rows (folded into runs by chat-tool-runs.js), and whatever else the
 * renderer places in the turn — cards, traces, notices, the footer. The agent's
 * task list is not part of the turn: chat-tasks.js pins it above the composer.
 * Opening one kind of block closes the others, and anything placed in
 * the entry breaks the run of tool calls, so the rows keep the order the
 * stream gave them. The renderer owns the stream; the turn owns the entry.
 *
 * @module chat-turn
 */

import {
  createAgentEntry,
  createTextBlock,
  createThinkingBlock,
  createTurnFooter,
} from './chat-blocks.js';
import { openSvgFenceAt } from './chat-svg.js';
import { createTurnActions } from './chat-tool-runs.js';

/** @typedef {import('./chat-dom.js').WireEvent} WireEvent */
/** @typedef {import('./chat-blocks.js').ThinkingBlock} ThinkingBlock */
/** @typedef {import('./chat-tool-cards.js').ToolChip} ToolChip */
/** @typedef {import('./chat-tool-cards.js').AgentChip} AgentChip */

/**
 * @typedef {object} AgentTurn
 * @property {HTMLElement} entry
 * @property {(chunk: string) => void} text - a whole text block
 * @property {(piece: string) => void} textDelta - a streamed piece of the open text block
 * @property {() => boolean} drawing - whether the reply streaming now is inside an `svg` block
 * @property {(chunk: string, piece?: boolean, timing?: { since?: number, durationMs?: number | null }) => void} think - reasoning, whole or streamed; *timing* says when it began (live) or how long it took (replay)
 * @property {(event: WireEvent) => ToolChip} tool - a tool call's row
 * @property {(event: WireEvent) => AgentChip} agent - a subagent call's row
 * @property {(node: Node) => void} place - close every open block, then append
 * @property {(node: Node) => void} append - append without closing the open text
 * @property {() => void} closeText
 * @property {() => void} closeThinking
 * @property {() => void} close - close every open block
 * @property {() => void} countApproval - an approval was asked of the operator
 * @property {(turn: { durationMs: number, isError: boolean }) => void} footer
 */

/**
 * @param {object} deps
 * @param {(el: HTMLElement, text: string) => void} deps.renderText - the
 *   markdown renderer every agent-authored body goes through
 * @param {() => void} [deps.onThinkingEnd] - called as an open reasoning row
 *   settles, so the status line can mark the moment
 * @returns {AgentTurn}
 */
export function createTurn({ renderText, onThinkingEnd }) {
  const entry = createAgentEntry();
  const actions = createTurnActions(entry);
  /** The open text block's body, or null. */
  let body = /** @type {HTMLElement | null} */ (null);
  /** Raw markdown of each text block, in order. */
  const texts = /** @type {string[]} */ ([]);
  /** Whether the open text block was built from streamed pieces. */
  let streamed = false;
  let thinking = /** @type {ThinkingBlock | null} */ (null);
  let thinkingStreamed = false;

  /**
   * Text blocks with a render waiting, coalesced to one render per frame: a
   * streamed reply re-renders its markdown as it grows, not once per piece.
   */
  const pendingRenders = new Map();
  const nextFrame =
    typeof requestAnimationFrame === 'function'
      ? requestAnimationFrame
      : (/** @type {() => void} */ fn) => setTimeout(fn, 16);

  /**
   * @param {HTMLElement} target
   * @param {string} text
   */
  function scheduleRender(target, text) {
    const queued = pendingRenders.has(target);
    pendingRenders.set(target, text);
    if (queued) return;
    nextFrame(() => {
      const latest = pendingRenders.get(target);
      pendingRenders.delete(target);
      if (latest !== undefined) renderText(target, latest);
    });
  }

  function closeText() {
    body = null;
    streamed = false;
  }

  function closeThinking() {
    if (thinking) {
      thinking.finish();
      actions.settled(thinking.el);
      thinking = null;
      thinkingStreamed = false;
      onThinkingEnd?.();
    }
  }

  function close() {
    closeText();
    closeThinking();
    actions.breakRun();
  }

  /** Open a text block, or return the one already open. */
  function openText() {
    closeThinking();
    actions.breakRun();
    if (!body) {
      body = createTextBlock();
      texts.push('');
      entry.appendChild(body);
    }
    return body;
  }

  return {
    entry,
    text(chunk) {
      // A whole block: when it was streamed it is the authoritative copy and
      // replaces what the pieces built; either way the block then ends, so
      // the next one starts a paragraph of its own.
      const target = openText();
      const i = texts.length - 1;
      texts[i] = streamed ? chunk : texts[i] + chunk;
      pendingRenders.delete(target);
      renderText(target, texts[i]);
      closeText();
    },
    textDelta(piece) {
      const target = openText();
      streamed = true;
      const i = texts.length - 1;
      texts[i] += piece;
      scheduleRender(target, texts[i]);
    },
    drawing() {
      return body !== null && streamed && openSvgFenceAt(texts[texts.length - 1] ?? '') >= 0;
    },
    /**
     * @param {string} chunk
     * @param {boolean} [piece] - a streamed piece, not the whole block
     * @param {{ since?: number, durationMs?: number | null }} [timing] - live: when the
     *   reasoning began; replay: how long the history says it took
     */
    think(chunk, piece = false, timing = {}) {
      // Streamed pieces add up in the block, and the whole block, when it
      // lands, replaces them.
      closeText();
      actions.breakRun();
      if (!thinking) {
        thinking = createThinkingBlock(timing.since);
        entry.appendChild(thinking.el);
        actions.live(thinking.el);
      }
      if (timing.durationMs !== undefined) thinking.record(timing.durationMs);
      if (piece) {
        thinkingStreamed = true;
        thinking.add(chunk);
      } else if (thinkingStreamed) {
        if (chunk) thinking.set(chunk);
      } else {
        thinking.add(chunk);
      }
    },
    tool(event) {
      closeText();
      closeThinking();
      return actions.addTool(event);
    },
    agent(event) {
      closeText();
      closeThinking();
      return actions.addAgent(event, renderText);
    },
    place(node) {
      close();
      entry.appendChild(node);
    },
    append(node) {
      actions.breakRun();
      entry.appendChild(node);
    },
    closeText,
    closeThinking,
    close,
    countApproval: actions.countApproval,
    footer({ durationMs, isError }) {
      actions.breakRun();
      entry.appendChild(createTurnFooter({ durationMs, isError, ...actions.stats() }));
    },
  };
}
