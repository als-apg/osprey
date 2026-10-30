// @ts-check
/**
 * Unit tests for the operator chat's refusal and failure copy (chat-notices.js):
 *   npx vitest run tests/interfaces/web_terminal/chat-notices.test.mjs
 *
 * What is pinned is which sentence an operator reads when the endpoint refuses a
 * turn. That choice is exported as `transportNotice`, because the failure it
 * exists to prevent is not a crash: the endpoint answers 409 both for "a turn
 * is already running" and for "this chat was restarted by your own posture
 * flip", and keying on the status alone told the second operator the first
 * sentence — false, and with no cue to do the one thing that works (send the
 * prompt again).
 *
 * `handoffRefusal` is the other reader: it names the one action a refused
 * hand-off leaves the operator, and returns null for anything that is not a
 * hand-off refusal so the console falls back to a transport notice.
 */

import { test, expect, describe } from 'vitest';

import {
  HANDOFF_FALLBACK,
  handoffRefusal,
  transportNotice,
} from '../../../src/osprey/interfaces/web_terminal/static/js/chat-notices.js';
import {
  HANDOFF_HELD_ELSEWHERE_MESSAGE,
  HANDOFF_OUTGOING_RUNNING_MESSAGE,
} from '../../../src/osprey/interfaces/web_terminal/static/js/terminal-handoff.js';

/**
 * A transport failure shaped the way chat-client.js builds one.
 * @param {number} status
 * @param {string} [slug]
 */
function transportError(status, slug = '') {
  /** @type {Error & { status: number, slug: string }} */
  const err = Object.assign(new Error(`HTTP ${status}: Conflict`), { status, slug });
  return err;
}

describe('transportNotice', () => {
  // Slug first, status second, the generic line last. Two 409s mean opposite
  // things (a turn already running, a chat restarted by a posture flip), so
  // the first rows pin that they read differently.
  test.each([
    [
      'chat_terminated 409',
      transportError(409, 'chat_terminated'),
      'The session restarted with the new posture. Send your message again.',
    ],
    ['turn_in_progress 409', transportError(409, 'turn_in_progress'), 'A turn is already running.'],
    [
      'chat_capacity 429',
      transportError(429, 'chat_capacity'),
      'The server is busy. Try again in a moment.',
    ],
    ['slug-less 503', transportError(503), 'The agent is unavailable.'],
    ['slug-less 409', transportError(409), 'A turn is already running.'],
    ['unknown slug on a 503', transportError(503, 'something_new'), 'The agent is unavailable.'],
    [
      'plain Error carrying a status',
      new Error('HTTP 429: Too Many Requests'),
      'The server is busy. Try again in a moment.',
    ],
    [
      'plain Error with no status',
      new Error('network down'),
      'The connection to the agent failed.',
    ],
    ['null failure', null, 'The connection to the agent failed.'],
  ])('%s', (_case, err, notice) => {
    expect(transportNotice(err)).toBe(notice);
  });
});

describe('transportNotice for a slash command the chat does not offer', () => {
  /**
   * @param {string} [command]
   */
  function refused(command) {
    return Object.assign(transportError(400, 'command_not_available'), { command });
  }

  test.each([
    ['an unlisted command', refused('model'), '/model is not available in this chat.'],
    [
      '/login',
      refused('login'),
      '/login is not available in this chat. Your administrator sets sign-in.',
    ],
    [
      '/logout',
      refused('logout'),
      '/logout is not available in this chat. Your administrator sets sign-in.',
    ],
    ['no name in the body', refused(), 'This command is not available in this chat.'],
  ])('%s', (_case, err, notice) => {
    expect(transportNotice(err)).toBe(notice);
  });
});

describe('handoffRefusal', () => {
  test.each([
    ['session_attached_elsewhere', { message: HANDOFF_HELD_ELSEWHERE_MESSAGE, action: null }],
    [
      'handoff_needs_interrupt',
      { message: 'The other view may still be working.', action: 'interrupt' },
    ],
    [
      'handoff_started_commands',
      { message: 'The agent started a command that is still running.', action: 'interrupt' },
    ],
    ['handoff_superseded', { message: 'Another request took over this session.', action: 'retry' }],
    ['outgoing_still_running', { message: HANDOFF_OUTGOING_RUNNING_MESSAGE, action: 'retry' }],
    ['outgoing_vanished', { message: 'The session changed while switching.', action: 'retry' }],
    ['spawn_not_pooled', { message: 'The session did not start.', action: 'retry' }],
    ['chat_capacity', { message: 'No chat capacity right now.', action: 'retry' }],
    ['sdk_unavailable', { message: 'The agent is unavailable.', action: null }],
  ])('%s returns its row', (slug, row) => {
    expect(handoffRefusal(transportError(409, slug))).toEqual(row);
  });

  test('handoff_started_commands the server could not read returns the unread row', () => {
    const err = Object.assign(transportError(409, 'handoff_started_commands'), { unread: true });
    expect(handoffRefusal(err)).toEqual({
      message: 'Could not check what the agent started.',
      action: 'interrupt',
    });
  });

  test.each([
    ['chat_terminated', transportError(409, 'chat_terminated')],
    ['a bare 409 with no slug', transportError(409)],
    ['null', null],
  ])('%s is not a hand-off refusal', (_case, err) => {
    expect(handoffRefusal(err)).toBeNull();
  });

  test('the fallback offers a retry', () => {
    expect(HANDOFF_FALLBACK).toEqual({ message: 'The switch failed.', action: 'retry' });
  });
});
