// @ts-check
/**
 * OSPREY Web Terminal — operator chat: the context meter.
 *
 * The meter is a quiet line under the composer. It shows how full the
 * conversation's context window is, from the `context` frame the server sends
 * as each response ends, and from {@link CONTEXT_WARN_PERCENT} up it says what
 * the operator can do about it. It is a `meter`, so a screen reader reports the
 * percentage when the operator reaches it.
 *
 * A replayed frame is history being redrawn, not something that happened now,
 * so it never changes the meter.
 *
 * @module chat-context
 */

import { elem } from './chat-dom.js';
import { onChat } from './chat-events.js';

/** The share of the context window, in percent, from which the meter warns. */
export const CONTEXT_WARN_PERCENT = 80;

/** What the warning tells the operator to do. */
const CONTEXT_HINT = 'Use /compact or start a new conversation';

/**
 * The meter's level for a reading: `warn` from {@link CONTEXT_WARN_PERCENT} up.
 * @param {number} percent
 * @returns {'ok' | 'warn'}
 */
export function contextLevel(percent) {
  return percent >= CONTEXT_WARN_PERCENT ? 'warn' : 'ok';
}

/**
 * A reading's percentage as the meter states it: a whole number from 0 to 100.
 * @param {unknown} value
 * @returns {number}
 */
function wholePercent(value) {
  const n = Number(value);
  return Number.isFinite(n) ? Math.min(100, Math.max(0, Math.round(n))) : 0;
}

let nextHintId = 0;

/**
 * Start the meter.
 * @param {import('./chat-features.js').ChatFeatureContext} ctx
 * @returns {() => void} stop listening and remove the meter
 */
export function initChatContext(ctx) {
  const root = elem('div', 'op-context');
  root.hidden = true;
  const bar = elem('div', 'op-context-bar');
  bar.setAttribute('role', 'meter');
  bar.setAttribute('aria-label', 'Context window used');
  bar.setAttribute('aria-valuemin', '0');
  bar.setAttribute('aria-valuemax', '100');
  const fill = elem('span', 'op-context-fill');
  bar.append(fill);
  const label = elem('span', 'op-context-label');
  const hint = elem('span', 'op-context-hint', CONTEXT_HINT);
  hint.id = `op-context-hint-${nextHintId++}`;
  hint.hidden = true;
  root.append(bar, label, hint);
  ctx.inputArea.append(root);

  /** @param {Record<string, any>} reading - a `context` frame */
  function show(reading) {
    const percent = wholePercent(reading.percent);
    const level = contextLevel(percent);
    root.hidden = false;
    root.dataset.level = level;
    fill.style.width = `${percent}%`;
    label.textContent = `Context ${percent}%`;
    bar.setAttribute('aria-valuenow', String(percent));
    bar.setAttribute('aria-valuetext', `${percent}% used`);
    const { used_tokens: used, window_tokens: window } = reading;
    if (typeof used === 'number' && typeof window === 'number') {
      bar.title = `${used.toLocaleString()} of ${window.toLocaleString()} tokens`;
    } else {
      bar.removeAttribute('title');
    }
    hint.hidden = level !== 'warn';
    if (level === 'warn') bar.setAttribute('aria-describedby', hint.id);
    else bar.removeAttribute('aria-describedby');
  }

  /** Nothing is known about the conversation on screen: the meter has no reading. */
  function clear() {
    root.hidden = true;
    root.dataset.level = 'ok';
    fill.style.width = '0%';
    label.textContent = '';
    hint.hidden = true;
    bar.removeAttribute('aria-describedby');
  }

  const stops = [
    onChat('frame', ({ event, replay }) => {
      if (replay) return;
      if (event.type === 'context') show(event);
    }),
    onChat('reset', clear),
  ];

  return () => {
    for (const stop of stops) stop();
    root.remove();
  };
}
