/**
 * Unit tests for the chat chat's context meter (chat-context.js), driven
 * through the renderer as the chat drives it:
 *   npx vitest run tests/interfaces/web_terminal/chat-context.test.mjs
 *
 * The meter reads the `context` frames the server sends as a response ends,
 * and warns from the threshold up. It does not react to a replayed frame:
 * history being redrawn is not something that has just happened.
 */

import { test, expect, describe, vi, beforeEach, afterEach } from 'vitest';

import { createChatRenderer } from '../../../src/osprey/interfaces/web_terminal/static/js/chat-render.js';
import {
  CONTEXT_WARN_PERCENT,
  contextLevel,
  initChatContext,
} from '../../../src/osprey/interfaces/web_terminal/static/js/chat-context.js';

/** @type {Array<() => void>} */
let stops = [];

beforeEach(() => {
  vi.stubGlobal('marked', { parse: /** @param {string} t */ (t) => t });
  vi.stubGlobal('DOMPurify', { sanitize: /** @param {string} h */ (h) => h });
});

afterEach(() => {
  for (const stop of stops) stop();
  stops = [];
  vi.unstubAllGlobals();
});

/**
 * The part of the chat's feature context the meter reads: the log, the composer
 * and the renderer. The console's other handles are not used here.
 * @param {HTMLElement} messages
 * @param {HTMLElement} inputArea
 * @param {ReturnType<typeof createChatRenderer>} renderer
 * @returns {import('../../../src/osprey/interfaces/web_terminal/static/js/chat-features.js').ChatFeatureContext}
 */
function contextOf(messages, inputArea, renderer) {
  return /** @type {any} */ ({ messages, inputArea, renderer });
}

/** A chat's log, its composer and the meter the feature puts in the composer. */
function mount() {
  const messages = document.createElement('div');
  const inputArea = document.createElement('div');
  const renderer = createChatRenderer(messages);
  const stop = initChatContext(contextOf(messages, inputArea, renderer));
  stops.push(stop);
  const meter = /** @type {HTMLElement} */ (inputArea.querySelector('.op-context'));
  const bar = /** @type {HTMLElement} */ (meter.querySelector('[role="meter"]'));
  const hint = /** @type {HTMLElement} */ (meter.querySelector('.op-context-hint'));
  return { messages, inputArea, renderer, meter, bar, hint };
}

/**
 * A `context` frame for a window at *percent*.
 * @param {number} percent
 */
function reading(percent) {
  return {
    type: 'context',
    used_tokens: Math.round((percent / 100) * 200000),
    window_tokens: 200000,
    percent,
  };
}

describe('the warning threshold', () => {
  test.each([
    [0, 'ok'],
    [CONTEXT_WARN_PERCENT - 1, 'ok'],
    [CONTEXT_WARN_PERCENT, 'warn'],
    [100, 'warn'],
  ])('%i percent is the %s level', (percent, level) => {
    expect(contextLevel(percent)).toBe(level);
  });
});

describe('the context meter', () => {
  test('stays hidden until a reading, then states it as a meter a screen reader can read', () => {
    const { renderer, meter, bar } = mount();
    expect(meter.hidden).toBe(true);

    renderer.handleEvent(reading(42));
    expect(meter.hidden).toBe(false);
    expect(bar.getAttribute('role')).toBe('meter');
    expect(bar.getAttribute('aria-valuenow')).toBe('42');
    expect(bar.getAttribute('aria-valuetext')).toBe('42% used');
    expect(meter.textContent).toContain('42%');
  });

  test('a reading at the threshold warns, with the hint described by the meter', () => {
    const { renderer, meter, bar, hint } = mount();
    renderer.handleEvent(reading(CONTEXT_WARN_PERCENT));
    expect(meter.dataset.level).toBe('warn');
    expect(hint.hidden).toBe(false);
    expect(hint.textContent).not.toBe('');
    expect(bar.getAttribute('aria-describedby')).toBe(hint.id);
  });

  test('the warning goes once the window has room again', () => {
    const { renderer, meter, bar, hint } = mount();
    renderer.handleEvent(reading(CONTEXT_WARN_PERCENT));
    expect(hint.hidden).toBe(false);
    renderer.handleEvent(reading(CONTEXT_WARN_PERCENT - 1));
    expect(meter.dataset.level).toBe('ok');
    expect(hint.hidden).toBe(true);
    expect(bar.hasAttribute('aria-describedby')).toBe(false);
    expect(meter.textContent).toContain(`${CONTEXT_WARN_PERCENT - 1}%`);
  });

  test('a reading is clipped to the scale', () => {
    const { renderer, bar } = mount();
    renderer.handleEvent(reading(130));
    expect(bar.getAttribute('aria-valuenow')).toBe('100');
  });

  test('the meter has no reading once the log is reset', () => {
    const { renderer, meter } = mount();
    renderer.handleEvent(reading(50));
    renderer.reset();
    expect(meter.hidden).toBe(true);
  });

  test('a replayed reading is not shown', () => {
    const { renderer, meter } = mount();
    renderer.replay([
      {
        role: 'assistant',
        content: '',
        events: [{ type: 'text', content: 'Old turn.' }, reading(90), { type: 'result' }],
      },
    ]);
    expect(meter.hidden).toBe(true);
    renderer.handleEvent(reading(10));
    expect(meter.hidden).toBe(false);
  });

  test('stopping the feature removes the meter', () => {
    const messages = document.createElement('div');
    const inputArea = document.createElement('div');
    const renderer = createChatRenderer(messages);
    const stop = initChatContext(contextOf(messages, inputArea, renderer));
    renderer.addUserMessage('kept');
    stop();

    renderer.handleEvent(reading(50));

    expect(inputArea.querySelector('.op-context')).toBeNull();
    expect(messages.textContent).toContain('kept');
  });
});
