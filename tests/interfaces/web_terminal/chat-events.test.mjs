/**
 * Unit tests for the chat lifecycle bus (chat-events.js) as the renderer
 * drives it, and for the `system` frame effects (chat-system.js):
 *   npx vitest run tests/interfaces/web_terminal/chat-events.test.mjs
 *
 * The bus is the contract the feature modules (attention, announcements,
 * export, the context meter) build on: a frame is published after it is
 * drawn, and `replay` tells a redrawn past turn from a live one.
 */

import { test, expect, describe, vi, beforeEach, afterEach } from 'vitest';

import { createChatRenderer } from '../../../src/osprey/interfaces/web_terminal/static/js/chat-render.js';
import { onChat } from '../../../src/osprey/interfaces/web_terminal/static/js/chat-events.js';
import { systemFrame } from '../../../src/osprey/interfaces/web_terminal/static/js/chat-system.js';

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
 * Record every detail published under *name*.
 * @template {'frame' | 'user-message' | 'agent-message' | 'turn-end' | 'reset'} K
 * @param {K} name
 */
function record(name) {
  /** @type {any[]} */
  const seen = [];
  stops.push(onChat(name, (detail) => seen.push(detail)));
  return seen;
}

describe('the renderer publishes its lifecycle', () => {
  test('a live frame is published after it is drawn, marked as not a replay', () => {
    const container = document.createElement('div');
    const r = createChatRenderer(container);
    /** @type {number[]} */
    const toolRowsAtPublish = [];
    stops.push(
      onChat('frame', () => toolRowsAtPublish.push(container.querySelectorAll('.op-tool').length))
    );
    const frames = record('frame');

    r.beginTurn();
    r.handleEvent({ type: 'tool_use', tool_name: 'channel_read', tool_use_id: 't1', input: {} });

    expect(frames).toEqual([{ event: expect.objectContaining({ tool_use_id: 't1' }), replay: false }]);
    // A listener can already find what the frame drew.
    expect(toolRowsAtPublish).toEqual([1]);
  });

  test('history redrawn by replay is marked as a replay, and live frames after it are not', () => {
    const r = createChatRenderer(document.createElement('div'));
    const frames = record('frame');
    const users = record('user-message');
    const agents = record('agent-message');

    r.replay([
      { role: 'user', content: 'read the orbit' },
      { role: 'assistant', content: '', events: [{ type: 'text', content: 'Done.' }, { type: 'result' }] },
      { role: 'assistant', content: 'Plain history.' },
    ]);
    r.beginTurn();
    r.handleEvent({ type: 'text', content: 'live' });

    expect(users).toEqual([{ text: 'read the orbit', replay: true }]);
    expect(agents).toEqual([{ markdown: 'Plain history.', replay: true }]);
    expect(frames.map((f) => [f.event.type, f.replay])).toEqual([
      ['text', true],
      ['result', true],
      ['text', false],
    ]);
  });

  test('turn-end fires once for a running turn and not for an idle endTurn', () => {
    const r = createChatRenderer(document.createElement('div'));
    const ends = record('turn-end');

    r.endTurn();
    expect(ends).toHaveLength(0);

    r.beginTurn();
    r.handleEvent({ type: 'result', is_error: false });
    r.endTurn();
    expect(ends).toHaveLength(1);
  });

  test('reset is published when the log is cleared', () => {
    const r = createChatRenderer(document.createElement('div'));
    const resets = record('reset');
    r.reset();
    expect(resets).toHaveLength(1);
  });
});

describe('systemFrame', () => {
  test('frames that draw nothing return null', () => {
    expect(systemFrame({ type: 'system', subtype: 'init' })).toBeNull();
    expect(systemFrame({ type: 'system', subtype: 'status', status: 'idle' })).toBeNull();
  });

  test('a compaction adds a notice that says how and how much', () => {
    const auto = systemFrame({ subtype: 'compact_boundary', trigger: 'auto', pre_tokens: 152400 });
    expect(auto).toEqual({ closeText: true, notice: expect.stringContaining('152k') });
    expect(auto?.notice).toMatch(/automatically/);

    const manual = systemFrame({ subtype: 'compact_boundary', trigger: 'manual' })?.notice ?? '';
    expect(manual).toMatch(/compacted/);
    expect(manual).not.toMatch(/automatically|tokens/);
  });

  test('a background agent report settles its chip, failed when it failed', () => {
    expect(
      systemFrame({ subtype: 'agent_done', tool_use_id: 'a1', status: 'failed', summary: 'x' })
    ).toEqual({
      closeText: true,
      agentDone: { id: 'a1', outcome: { status: 'failed', summary: 'x', is_error: true } },
    });
  });

  test('an API retry names the status and the attempt', () => {
    const activity = systemFrame({ subtype: 'api_retry', error_status: 529, attempt: 2, max_retries: 10 })?.activity ?? '';
    expect(activity).toContain('529');
    expect(activity).toContain('2/10');
  });
});
