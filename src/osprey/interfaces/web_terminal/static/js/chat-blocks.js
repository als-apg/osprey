// @ts-check
/* OSPREY Web Terminal — Simple-mode chat: turn blocks.
 *
 * DOM builders for the pieces an agent turn is made of that are not tool calls
 * or cards: the live loader, the thinking row, the turn footer, the error,
 * divider and notice lines, the user and agent entries, and the text block.
 * Each builder returns its element together with the small handle the renderer
 * drives it through. None of them reads the event stream or touches the
 * network.
 *
 * The rest of a turn is built by the modules it names: the tool-call group and
 * its chips in chat-tool-cards.js, the plan in chat-plan.js, the approval and
 * question cards in chat-approval.js and chat-question.js (both on the contract
 * in chat-permission.js), and the shared vocabulary and DOM helpers in
 * chat-tool-vocab.js and chat-dom.js.
 *
 * Everything that came over the wire (tool inputs, previews, questions) is
 * written with textContent. Nothing here reaches innerHTML.
 *
 * @module chat-blocks
 */

import { chatIcon, ospreySprite } from './chat-icons.js';
import { button, elem, formatElapsed } from './chat-dom.js';

/**
 * @typedef {object} LoaderHandle
 * @property {HTMLElement} el
 * @property {(text: string) => void} setLabel
 * @property {(startedAt: number) => void} start - tick from *startedAt* (epoch ms)
 * @property {() => void} stop
 */

/**
 * @typedef {object} ThinkingBlock
 * @property {HTMLElement} el
 * @property {(text: string) => void} add
 * @property {(text: string) => void} set - replace the reasoning with its final copy
 * @property {(opts?: { fly?: boolean }) => void} finish - close the block; `fly`
 *   plays the osprey's flourish (two wingbeats, then a roll), for a live finish
 */

// ---- Live status ---- //

/**
 * The turn's live status line: the OSPREY mark beating its wings, the current
 * activity in a shimmer, and the elapsed time. The renderer decides when it
 * shows; the loader owns its label and its clock.
 * @returns {LoaderHandle}
 */
export function createLoader() {
  const el = elem('div', 'op-loader');
  // No live role: the label changes with every step, and chat-announce.js speaks what matters.
  const bird = ospreySprite();
  bird.classList.add('op-loader-bird');
  const label = elem('span', 'op-loader-label op-shimmer', 'Working…');
  const sep = elem('span', 'op-loader-sep', '·');
  sep.setAttribute('aria-hidden', 'true');
  // Off the live region: the label is the announcement; the clock is for the eye.
  const time = elem('span', 'op-loader-time', '0.0s');
  time.setAttribute('aria-live', 'off');
  el.append(bird, label, sep, time);

  let since = 0;
  /** @type {number | null} */
  let timer = null;
  const tick = () => {
    time.textContent = formatElapsed(Date.now() - since, true);
  };

  return {
    el,
    setLabel(text) {
      if (label.textContent !== text) label.textContent = text;
    },
    start(startedAt) {
      since = startedAt;
      if (timer === null) {
        tick();
        timer = window.setInterval(tick, 100);
      }
    },
    stop() {
      if (timer !== null) {
        window.clearInterval(timer);
        timer = null;
      }
    },
  };
}

// ---- Reasoning ---- //

/**
 * A collapsible reasoning row, collapsed by default. Its label shimmers while
 * the reasoning is still arriving, then reads how long it took. Its glyph is the
 * OSPREY mark, which can fly once when the reasoning ends.
 * @returns {ThinkingBlock}
 */
export function createThinkingBlock() {
  const startedAt = Date.now();
  const el = elem('div', 'op-thinking live');
  const head = button('op-thinking-head');
  head.setAttribute('aria-expanded', 'false');
  const glyph = ospreySprite();
  glyph.classList.add('op-thinking-glyph');
  // One flight per finish: re-inserting the row must not replay it.
  glyph.addEventListener('animationend', (event) => {
    if (event.animationName === 'op-osprey-roll') glyph.classList.remove('flying');
  });
  const label = elem('span', 'op-thinking-label op-shimmer', 'Thinking');
  const chevron = chatIcon('chevron');
  chevron.classList.add('op-thinking-chevron');
  head.append(glyph, label, chevron);
  const body = elem('div', 'op-thinking-body');
  body.hidden = true;
  head.addEventListener('click', () => {
    const open = body.hidden;
    body.hidden = !open;
    el.classList.toggle('open', Boolean(open));
    head.setAttribute('aria-expanded', String(open));
  });
  el.append(head, body);

  return {
    el,
    add(text) {
      body.append(text);
    },
    set(text) {
      body.textContent = text;
    },
    finish({ fly = false } = {}) {
      el.classList.remove('live');
      label.classList.remove('op-shimmer');
      if (fly) glyph.classList.add('flying');
      const elapsed = Date.now() - startedAt;
      label.textContent = elapsed >= 1000 ? `Thought for ${formatElapsed(elapsed)}` : 'Thought';
      // Models that return redacted reasoning send empty blocks: nothing to open.
      if (!body.textContent?.trim()) {
        el.classList.add('empty');
        head.disabled = true;
      }
    },
  };
}

// ---- Turn furniture ---- //

/**
 * The line that closes a finished turn: how long it worked. The reply's copy
 * control is chat-export.js's.
 * @param {{ durationMs: number, isError: boolean }} turn
 * @returns {HTMLElement}
 */
export function createTurnFooter({ durationMs, isError }) {
  const el = elem('div', 'op-footer');
  const lead = isError ? 'Ended with an error after' : 'Worked for';
  el.append(elem('span', 'op-footer-text', `${lead} ${formatElapsed(durationMs)}`));
  return el;
}

/**
 * A soft error card. Text only.
 * @param {string} message
 * @returns {HTMLElement}
 */
export function createErrorBlock(message) {
  const el = elem('div', 'op-error-block');
  // No alert role: chat-announce.js speaks an error once, live, and never on replay.
  const icon = chatIcon('alert');
  icon.classList.add('op-error-icon');
  el.append(icon, elem('span', 'op-error-text', message));
  return el;
}

/** The line that marks an agent restart: the earlier turns stay above it. @returns {HTMLElement} */
export function createSessionDivider() {
  return elem('div', 'op-system', 'session reset');
}

/**
 * A one-line notice from the CLI in the log, e.g. that the conversation was
 * compacted. Not a message: it carries no text the agent said.
 * @param {string} text
 * @returns {HTMLElement}
 */
export function createNotice(text) {
  const line = elem('div', 'op-notice');
  line.append(chatIcon('alert'), elem('span', 'op-notice-text', text));
  return line;
}

/**
 * An operator's turn: a soft bubble, right-aligned.
 * @param {string} text
 * @returns {HTMLElement}
 */
export function createUserEntry(text) {
  const entry = elem('div', 'op-entry operator');
  entry.append(elem('div', 'op-bubble', text));
  return entry;
}

/**
 * The agent's side of a turn: no bubble. Its blocks are appended in order.
 * @returns {HTMLElement}
 */
export function createAgentEntry() {
  return elem('div', 'op-entry assistant');
}

/**
 * A markdown block of the agent's reply. The shared `osprey-md-rendered` class
 * is what the link handling and the markdown styles key on.
 * @returns {HTMLElement}
 */
export function createTextBlock() {
  return elem('div', 'op-entry-body osprey-md-rendered');
}
