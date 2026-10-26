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
 * The rest of a turn is built by the modules it names: the tool and subagent
 * rows in chat-tool-cards.js, folded into runs by chat-tool-runs.js, the
 * approval and question cards in chat-approval.js and chat-question.js (both
 * on the contract in chat-permission.js), the row shape in chat-row.js, and
 * the shared vocabulary and DOM helpers in chat-tool-vocab.js and chat-dom.js.
 * The agent's task list sits outside the turn, in chat-tasks.js.
 *
 * Everything that came over the wire (tool inputs, previews, questions) is
 * written with textContent. Nothing here reaches innerHTML.
 *
 * @module chat-blocks
 */

import { chatIcon, ospreySprite } from './chat-icons.js';
import { createRow } from './chat-row.js';
import { elem, formatElapsed } from './chat-dom.js';

/** @typedef {import('./chat-icons.js').ChatIconName} ChatIconName */

/**
 * @typedef {object} LoaderHandle
 * @property {HTMLElement} el
 * @property {(text: string) => void} setLabel
 * @property {(startedAt: number) => void} start - tick from *startedAt* (epoch ms)
 * @property {() => void} stop
 * @property {() => void} flourish - the osprey beats its wings twice, quicker, then rolls: a live finish
 */

/**
 * @typedef {object} ThinkingBlock
 * @property {HTMLElement} el
 * @property {(text: string) => void} add
 * @property {(text: string) => void} set - replace the reasoning with its final copy
 * @property {(ms: number | null) => void} record - how long the reasoning took, as the
 *   history recorded it; `null` when the history holds no time for it
 * @property {() => void} finish - close the block: the row settles and reads how long it took
 */

/**
 * How a settled reasoning row reads: how long the agent thought, or a bare
 * "Thought" when no time is known for it.
 * @param {number | null} ms
 * @returns {string}
 */
export function thoughtLabel(ms) {
  if (ms === null) return 'Thought';
  return ms < 1000 ? 'Thought for <1s' : `Thought for ${formatElapsed(ms)}`;
}

// ---- Live status ---- //

/**
 * The turn's live status line: the OSPREY mark beating its wings, the current
 * activity in a shimmer, and the elapsed time. The renderer decides when it
 * shows; the loader owns its label and its clock. The mark is the brand, and
 * this line is the one place in a turn it flies: the rows carry glyphs.
 * @returns {LoaderHandle}
 */
export function createLoader() {
  const el = elem('div', 'op-loader');
  // No live role: the label changes with every step, and chat-announce.js speaks what matters.
  const bird = ospreySprite();
  bird.classList.add('op-loader-bird');
  // One flight per flourish: the roll's end returns the bird to its steady beat.
  bird.addEventListener('animationend', (event) => {
    if (event.animationName === 'op-osprey-roll') bird.classList.remove('flying');
  });
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
    flourish() {
      bird.classList.add('flying');
    },
  };
}

// ---- Reasoning ---- //

/**
 * A collapsible reasoning row, collapsed by default: the brain glyph in the
 * gutter, a label that shimmers while the reasoning is still arriving and then
 * reads how long it took, and the chevron at the right edge. Live, the time
 * runs from *since*, the turn's last frame before the reasoning: the agent
 * starts thinking when that frame lands, not when its first words arrive.
 * @param {number} [since] - epoch ms
 * @returns {ThinkingBlock}
 */
export function createThinkingBlock(since = Date.now()) {
  /** The recorded duration; undefined means measure it live. */
  let recorded = /** @type {number | null | undefined} */ (undefined);
  const row = createRow({
    className: 'op-thinking',
    kind: 'thinking',
    glyph: chatIcon('brain'),
    label: 'Thinking',
    ownStatus: true,
  });
  row.label.classList.add('op-shimmer');
  const chevron = chatIcon('chevron');
  chevron.classList.add('op-row-chevron');
  row.status.append(chevron);
  row.head.addEventListener('click', () => row.setOpen(!row.isOpen()));

  // The row opens only onto reasoning it holds: a model that keeps its
  // reasoning hidden sends empty pieces, live and when it settles alike.
  function syncOpenable() {
    const empty = !row.body.textContent?.trim();
    row.el.classList.toggle('empty', empty);
    row.head.disabled = empty;
  }
  syncOpenable();

  return {
    el: row.el,
    add(text) {
      row.body.append(text);
      syncOpenable();
    },
    set(text) {
      row.body.textContent = text;
      syncOpenable();
    },
    record(ms) {
      recorded = ms;
    },
    finish() {
      row.setState('done');
      row.label.classList.remove('op-shimmer');
      row.label.textContent = thoughtLabel(recorded === undefined ? Date.now() - since : recorded);
      syncOpenable();
    },
  };
}

// ---- Turn furniture ---- //

/**
 * What a turn did, as its footer states it.
 * @typedef {object} TurnSummary
 * @property {number} durationMs
 * @property {boolean} isError
 * @property {number} actions - the tool and subagent calls the turn made
 * @property {number} approvals - the approvals it asked the operator for
 * @property {ChatIconName[]} icons - the actions' icons, in order of first use
 */

/**
 * One part of the footer: its small icons, then its text.
 * @param {(SVGSVGElement | HTMLElement)[]} icons
 * @param {string} text
 * @param {string} [className]
 */
function footerPart(icons, text, className = '') {
  const part = elem('span', `op-footer-part ${className}`.trim());
  if (icons.length > 1) {
    const stack = elem('span', 'op-footer-stack');
    stack.append(...icons);
    part.append(stack);
  } else {
    part.append(...icons);
  }
  part.append(elem('span', 'op-footer-text', text));
  return part;
}

/**
 * The line that closes a finished turn: how long it worked, then what it did,
 * counting only what happened. The reply's copy control is chat-export.js's.
 * @param {TurnSummary} turn
 * @returns {HTMLElement}
 */
export function createTurnFooter({ durationMs, isError, actions, approvals, icons }) {
  const el = elem('div', 'op-footer');
  const parts = [footerPart([], formatElapsed(durationMs))];
  if (actions > 0) {
    const marks = icons.slice(0, 4).map((name) => chatIcon(name));
    parts.push(footerPart(marks, `${actions} action${actions === 1 ? '' : 's'}`));
  }
  if (approvals > 0) {
    parts.push(
      footerPart([chatIcon('check')], `${approvals} approval${approvals === 1 ? '' : 's'}`)
    );
  }
  if (isError)
    parts.push(footerPart([chatIcon('alert')], 'ended with an error', 'op-footer-error'));
  parts.forEach((part, i) => {
    if (i > 0) {
      const sep = elem('span', 'op-footer-sep', '·');
      sep.setAttribute('aria-hidden', 'true');
      el.append(sep);
    }
    el.append(part);
  });
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
