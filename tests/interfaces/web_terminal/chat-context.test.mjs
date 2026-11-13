/**
 * Unit tests for the chat's context meter and model tag (chat-context.js),
 * driven through the renderer and the chat bus as the chat drives them:
 *   npx vitest run tests/interfaces/web_terminal/chat-context.test.mjs
 *
 * The meter reads the `context` frames the server sends as a response ends,
 * and warns from the threshold up. The tag names the model from the feed's
 * snapshot (the `model` chat event) and from `init` frames. Neither reacts to
 * a replayed frame: history being redrawn is not something that has just
 * happened.
 */

import { test, expect, describe, vi, beforeEach, afterEach } from 'vitest';

import { emitChat } from '../../../src/osprey/interfaces/web_terminal/static/js/chat-events.js';
import { createChatRenderer } from '../../../src/osprey/interfaces/web_terminal/static/js/chat-render.js';
import {
  CONTEXT_WARN_PERCENT,
  contextLevel,
  initChatContext,
  modelLabel,
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

describe('the model label', () => {
  test.each([
    ['claude-sonnet-4-5-20250929', 'Sonnet-4-5'],
    ['anthropic/claude-opus-4-6', 'Opus-4-6'],
    ['gemini-3-flash', 'Gemini-3-flash'],
    ['claude-haiku-4-5-20251001[1m]', 'Haiku-4-5'],
    ['gateway/team/claude-opus-5', 'Opus-5'],
    ['anthropic/', 'anthropic/'],
  ])('%s is shown as %s', (model, label) => {
    expect(modelLabel(model)).toBe(label);
  });
});

describe('the model tag', () => {
  /** The tag in a mounted line. */
  function tagOf(/** @type {HTMLElement} */ meter) {
    return /** @type {HTMLElement} */ (meter.querySelector('.op-context-model'));
  }

  test('stays hidden, and the line with it, while no model is known', () => {
    const { meter } = mount();
    expect(tagOf(meter).hidden).toBe(true);
    expect(tagOf(meter).textContent).toBe('');
    expect(meter.hidden).toBe(true);
  });

  test("the feed's snapshot shows the short name, with the full id as title and name", () => {
    const { meter, bar } = mount();
    emitChat('model', { model: 'claude-sonnet-4-5-20250929' });
    const tag = tagOf(meter);
    expect(meter.hidden).toBe(false);
    expect(tag.hidden).toBe(false);
    expect(tag.textContent).toBe('Sonnet-4-5');
    expect(tag.title).toBe('Model: claude-sonnet-4-5-20250929');
    expect(tag.getAttribute('aria-label')).toBe('Model: claude-sonnet-4-5-20250929');
    // No reading yet: the line shows the tag alone.
    expect(bar.hidden).toBe(true);
  });

  test('an init frame names the model, and a replayed one does not', () => {
    const { renderer, meter } = mount();
    renderer.replay([
      {
        role: 'assistant',
        content: '',
        events: [{ type: 'system', subtype: 'init', model: 'claude-old' }, { type: 'result' }],
      },
    ]);
    expect(tagOf(meter).hidden).toBe(true);

    renderer.handleEvent({ type: 'system', subtype: 'init', model: 'anthropic/claude-opus-4-6' });
    expect(tagOf(meter).textContent).toBe('Opus-4-6');
  });

  test('a later name replaces the tag, and no model hides it', () => {
    const { renderer, meter } = mount();
    emitChat('model', { model: 'claude-sonnet-4-5' });
    renderer.handleEvent({ type: 'system', subtype: 'init', model: 'gemini-3-flash' });
    expect(tagOf(meter).textContent).toBe('Gemini-3-flash');

    emitChat('model', { model: null });
    expect(tagOf(meter).hidden).toBe(true);
    expect(tagOf(meter).hasAttribute('title')).toBe(false);
    expect(meter.hidden).toBe(true);
  });

  test('the tag starts the line before the reading, and outlasts a reset of the log', () => {
    const { renderer, meter, bar } = mount();
    emitChat('model', { model: 'claude-sonnet-4-5' });
    renderer.handleEvent(reading(42));
    expect(bar.hidden).toBe(false);
    expect(meter.textContent).toContain('42%');
    expect(tagOf(meter).hidden).toBe(false);
    expect(meter.firstElementChild).toBe(tagOf(meter));
    expect(tagOf(meter).nextElementSibling).toBe(bar);

    renderer.reset();
    expect(bar.hidden).toBe(true);
    expect(meter.hidden).toBe(false);
    expect(tagOf(meter).textContent).toBe('Sonnet-4-5');
  });
});
