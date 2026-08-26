// @ts-check
/**
 * Unit tests for the chat controller's transport-notice mapping (chat.js):
 *   npx vitest run tests/interfaces/web_terminal/chat.test.mjs
 *
 * The controller is DOM glue and the interesting decision in it is which
 * sentence an operator reads when the endpoint refuses a turn. That choice is
 * exported as `transportNotice` and tested here directly, because the failure
 * it exists to prevent is not a crash: the endpoint answers 409 both for "a
 * turn is already running" and for "this chat was restarted by your own
 * posture flip", and keying on the status alone told the second operator the
 * first sentence — false, and with no cue to do the one thing that works
 * (send the prompt again).
 */

import { test, expect, describe } from 'vitest';

import { transportNotice } from '../../../src/osprey/interfaces/web_terminal/static/js/chat.js';

/**
 * A transport failure shaped the way chat-client.js builds one.
 * @param {number} status
 * @param {string} [slug]
 */
function transportError(status, slug = '') {
  const err = /** @type {Error & { status: number, slug: string }} */ (
    new Error(`HTTP ${status}: Conflict`)
  );
  err.status = status;
  err.slug = slug;
  return err;
}

describe('transportNotice', () => {
  test('the terminated-chat 409 tells the operator to resend', () => {
    const notice = transportNotice(transportError(409, 'chat_terminated'));
    expect(notice).toContain('send your message again');
    expect(notice).not.toContain('already running');
  });

  test('the turn-in-progress 409 keeps its own copy', () => {
    expect(transportNotice(transportError(409, 'turn_in_progress'))).toBe(
      'A turn is already running.'
    );
  });

  test('two 409s with different slugs read differently', () => {
    expect(transportNotice(transportError(409, 'chat_terminated'))).not.toBe(
      transportNotice(transportError(409, 'turn_in_progress'))
    );
  });

  test('the capacity 429 is keyed on its slug', () => {
    expect(transportNotice(transportError(429, 'chat_capacity'))).toBe(
      'Server busy — please retry in a moment.'
    );
  });

  test('a rejection with no slug falls back to the status table', () => {
    expect(transportNotice(transportError(503))).toBe('Operator agent unavailable.');
    expect(transportNotice(transportError(409))).toBe('A turn is already running.');
  });

  test('an unknown slug falls back to the status table', () => {
    expect(transportNotice(transportError(503, 'something_new'))).toBe(
      'Operator agent unavailable.'
    );
  });

  test('a plain Error still yields its status notice, then the generic line', () => {
    expect(transportNotice(new Error('HTTP 429: Too Many Requests'))).toBe(
      'Server busy — please retry in a moment.'
    );
    expect(transportNotice(new Error('network down'))).toBe(
      'Connection to the operator agent failed.'
    );
  });

  test('a null failure does not throw', () => {
    expect(transportNotice(null)).toBe('Connection to the operator agent failed.');
  });
});
