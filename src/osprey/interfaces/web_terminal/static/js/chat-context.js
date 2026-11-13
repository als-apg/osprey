// @ts-check
/**
 * OSPREY Web Terminal — operator chat: the context meter and the model tag.
 *
 * The meter is a quiet line under the composer. It shows how full the
 * conversation's context window is, from the `context` frame the server sends
 * as each response ends, and from {@link CONTEXT_WARN_PERCENT} up it says what
 * the operator can do about it. It is a `meter`, so a screen reader reports the
 * percentage when the operator reaches it.
 *
 * The same line names the model that answers in the chat, short (see
 * {@link modelLabel}), with the full id in its title. The live feed's snapshot
 * names the model (the `model` chat event), and so does each `init` frame.
 * The tag only shows the model: it is not a control.
 *
 * A replayed frame is history being redrawn, not something that happened now,
 * so it never changes the meter or the tag.
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
 * A model id as the tag shows it: without a gateway prefix (`anthropic/`), a
 * leading `claude-`, a release date (`-20250929`) or a bracket suffix (`[1m]`),
 * and with a capital first letter (`Sonnet-4-5`). An id with nothing left after
 * that is shown whole, as it is.
 * @param {string} model
 * @returns {string}
 */
export function modelLabel(model) {
  const label = model
    .slice(model.lastIndexOf('/') + 1)
    .replace(/\[[^\]]*\]$/, '')
    .replace(/^claude-/, '')
    .replace(/-\d{8}$/, '');
  return label ? label.charAt(0).toUpperCase() + label.slice(1) : model;
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
 * Start the meter and the model tag.
 * @param {import('./chat-features.js').ChatFeatureContext} ctx
 * @returns {() => void} stop listening and remove the line
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
  // The tag starts the line; the meter follows it (see operator.css).
  const tag = elem('span', 'op-context-model');
  tag.setAttribute('role', 'note');
  tag.hidden = true;
  root.append(tag, bar, label, hint);
  ctx.inputArea.append(root);

  /** Whether the meter has a reading to show. */
  let reading = false;

  /** The line shows while the meter has a reading or the tag names a model. */
  function syncLine() {
    bar.hidden = !reading;
    label.hidden = !reading;
    root.hidden = !reading && tag.hidden;
  }

  /** @param {Record<string, any>} frame - a `context` frame */
  function show(frame) {
    const percent = wholePercent(frame.percent);
    const level = contextLevel(percent);
    reading = true;
    syncLine();
    root.dataset.level = level;
    fill.style.width = `${percent}%`;
    label.textContent = `Context ${percent}%`;
    bar.setAttribute('aria-valuenow', String(percent));
    bar.setAttribute('aria-valuetext', `${percent}% used`);
    const { used_tokens: used, window_tokens: window } = frame;
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
    reading = false;
    syncLine();
    root.dataset.level = 'ok';
    fill.style.width = '0%';
    label.textContent = '';
    hint.hidden = true;
    bar.removeAttribute('aria-describedby');
  }

  /** @param {unknown} model - the model id the server names; anything else hides the tag */
  function showModel(model) {
    const id = typeof model === 'string' ? model.trim() : '';
    tag.hidden = !id;
    tag.textContent = id ? modelLabel(id) : '';
    if (id) {
      tag.title = `Model: ${id}`;
      tag.setAttribute('aria-label', `Model: ${id}`);
    } else {
      tag.removeAttribute('title');
      tag.removeAttribute('aria-label');
    }
    syncLine();
  }

  const stops = [
    onChat('frame', ({ event, replay }) => {
      if (replay) return;
      if (event.type === 'context') show(event);
      else if (event.type === 'system' && event.subtype === 'init') showModel(event.model);
    }),
    // A cleared log clears the meter. The tag follows the feed and the init
    // frames, which say when the chat's model changes, not the log.
    onChat('reset', clear),
    onChat('model', ({ model }) => showModel(model)),
  ];

  return () => {
    for (const stop of stops) stop();
    root.remove();
  };
}
