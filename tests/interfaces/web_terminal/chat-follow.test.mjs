// @ts-check
/**
 * Unit tests for chat-follow.js — what the live feed's frames mean for the log:
 *   npx vitest run tests/interfaces/web_terminal/chat-follow.test.mjs
 *
 * The console (chat.js) owns the DOM and the network and hands the follower
 * what it needs. These tests stand in for the console with a real renderer
 * over a detached log, and with transcript reads the test answers when it
 * chooses, so a read can land after the feed has moved on. What is asserted
 * is the log the operator would see, whether a turn shows as running, and
 * how often the transcript is read.
 */

import { test, expect, vi } from 'vitest';

import { createChatRenderer } from '../../../src/osprey/interfaces/web_terminal/static/js/chat-render.js';
import { createTurnFollower } from '../../../src/osprey/interfaces/web_terminal/static/js/chat-follow.js';

/**
 * A follower over a fresh log, and transcript reads that wait for the test.
 * @param {{ sending?: () => boolean }} [options]
 */
function mount({ sending = () => false } = {}) {
  const log = document.createElement('div');
  const renderer = createChatRenderer(log, { answerPermission: vi.fn() });
  /** @type {Array<{ resolve: (turns: any[]) => void, reject: (err: Error) => void }>} */
  const reads = [];
  let running = false;
  const follower = createTurnFollower({
    renderer,
    draw: (event) => renderer.handleEvent(event),
    fetchHistory: () =>
      new Promise((resolve, reject) => {
        reads.push({ resolve, reject });
      }),
    key: () => 'K1',
    sending,
    setStreaming: (on) => {
      running = on;
    },
    dismissInvitation: () => {},
    scrollToBottom: () => {},
    redraw: (turns) => {
      renderer.reset();
      renderer.replay(turns);
    },
  });
  /** Every rendered entry's text, in order. */
  const entries = () =>
    [...log.querySelectorAll('.op-bubble, .op-entry-body')].map((el) => el.textContent?.trim());
  return { log, renderer, follower, reads, entries, running: () => running };
}

/** Let a transcript read that the test answered land. */
async function settle() {
  for (let i = 0; i < 5; i += 1) await Promise.resolve();
}

/** A turn the transcript shows finished. */
const finished = (/** @type {string} */ prompt, /** @type {string} */ answer) => [
  { role: 'user', content: prompt },
  {
    role: 'assistant',
    content: answer,
    events: [
      { type: 'text', content: answer },
      { type: 'result', is_error: false, duration_ms: 1000 },
    ],
  },
];

/** A turn the transcript shows still running. */
const running = (/** @type {string} */ prompt, /** @type {string} */ said) => [
  { role: 'user', content: prompt },
  { role: 'assistant', content: said, events: [{ type: 'text', content: said }] },
];

test('a transcript read that a new turn overtook leaves the log to that turn, which reads it again when it ends', async () => {
  const { follower, reads, entries, running: busy } = mount();
  follower.frame({ type: 'hello', seq: 0, in_flight: false, pending: [] });
  follower.frame({ type: 'turn_start', prompt: 'read the orbit', seq: 1 });
  follower.frame({ type: 'text', content: 'Reading.', seq: 2 });
  // The feed dropped; the turn ended out of sight, so the transcript is read.
  follower.frame({ type: 'hello', seq: 5, in_flight: false, pending: [] });
  expect(reads).toHaveLength(1);

  // Before that read lands, another tab sends the next prompt.
  follower.frame({ type: 'turn_start', prompt: 'and the tunes?', seq: 6 });
  follower.frame({ type: 'text', content: 'Nominal.', seq: 7 });
  reads[0].resolve(finished('read the orbit', 'It is flat.'));
  await settle();

  expect(entries()).toEqual(['read the orbit', 'Reading.', 'and the tunes?', 'Nominal.']);
  expect(busy()).toBe(true);

  follower.frame({ type: 'result', is_error: false, duration_ms: 500, seq: 8 });
  expect(reads).toHaveLength(2);
  reads[1].resolve([
    ...finished('read the orbit', 'It is flat.'),
    ...finished('and the tunes?', 'Nominal.'),
  ]);
  await settle();

  expect(entries()).toEqual(['read the orbit', 'It is flat.', 'and the tunes?', 'Nominal.']);
  expect(busy()).toBe(false);
});

test('a transcript read that lands while this tab sends a prompt leaves the log alone', async () => {
  let sending = false;
  const { renderer, follower, reads, entries } = mount({ sending: () => sending });
  renderer.replay(running('read the orbit', 'Reading.'));
  follower.frame({ type: 'hello', seq: 3, in_flight: false, pending: [] });
  expect(reads).toHaveLength(1);

  sending = true;
  renderer.addUserMessage('and the tunes?');
  reads[0].resolve(finished('read the orbit', 'It is flat.'));
  await settle();

  expect(entries()).toEqual(['read the orbit', 'Reading.', 'and the tunes?']);
});

test('no agent for the chat closes the turn the transcript left open, and reads it once', async () => {
  const { renderer, follower, reads, log, running: busy } = mount();
  renderer.replay(running('read the orbit', 'Reading.'));
  expect(renderer.hasOpenTurn()).toBe(true);

  follower.noAgent();
  // The agent died mid-turn: the transcript ends there for good.
  reads[0].resolve(running('read the orbit', 'Reading.'));
  await settle();

  expect(renderer.hasOpenTurn()).toBe(false);
  expect(log.querySelector('.op-footer')).toBeNull();
  expect(busy()).toBe(false);
  // The feed asks again every few seconds; there is nothing left to read.
  follower.noAgent();
  expect(reads).toHaveLength(1);
});

test('a reconnect whose transcript read fails keeps drawing the turn, and reads it again when it ends', async () => {
  const { follower, reads, entries, running: busy } = mount();
  follower.frame({ type: 'hello', seq: 0, in_flight: false, pending: [] });
  follower.frame({ type: 'turn_start', prompt: 'read the orbit', seq: 1 });
  follower.frame({ type: 'text', content: 'Reading.', seq: 2 });
  follower.frame({ type: 'hello', seq: 4, in_flight: true, pending: [] });
  reads[0].reject(new Error('network down'));
  await settle();

  expect(busy()).toBe(true);
  follower.frame({ type: 'text', content: 'It is flat.', seq: 5 });
  expect(entries()).toEqual(['read the orbit', 'Reading.', 'It is flat.']);

  follower.frame({ type: 'result', is_error: false, duration_ms: 900, seq: 6 });
  expect(reads).toHaveLength(2);
});
