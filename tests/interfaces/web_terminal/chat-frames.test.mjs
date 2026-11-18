// @ts-check
/**
 * Unit tests for the chat's frame predicates (chat-frames.js):
 *   npx vitest run tests/interfaces/web_terminal/chat-frames.test.mjs
 *
 * The controller ends a mirrored turn on `endsTurn` and joins a turn it is not
 * drawing yet only on a frame `isResponseFrame` claims, so both are pinned
 * frame type by frame type.
 */

import { describe, expect, test } from 'vitest';

import {
  endsTurn,
  isResponseFrame,
} from '../../../src/osprey/interfaces/web_terminal/static/js/chat-frames.js';

describe('endsTurn', () => {
  test('a frame the server marks as ending the turn ends it', () => {
    expect(endsTurn({ type: 'result', ends_turn: true })).toBe(true);
    expect(endsTurn({ type: 'error', error_type: 'ProcessError', ends_turn: true })).toBe(true);
  });

  test('a frame without the mark does not, whatever its type', () => {
    expect(endsTurn({ type: 'result' })).toBe(false);
    expect(endsTurn({ type: 'error', error_type: 'ProcessError' })).toBe(false);
  });

  test.each(['text', 'delta', 'tool_use', 'context', 'turn_start', 'hello'])(
    'a %s frame does not',
    (type) => {
      expect(endsTurn({ type })).toBe(false);
    }
  );
});

describe('isResponseFrame', () => {
  test.each(['text', 'delta', 'thinking', 'tool_use', 'tool_result', 'permission_request'])(
    'a %s frame is part of a response',
    (type) => {
      expect(isResponseFrame({ type })).toBe(true);
    }
  );

  test.each(['result', 'error', 'context', 'hello', 'turn_start', 'system'])(
    'a %s frame is not',
    (type) => {
      expect(isResponseFrame({ type })).toBe(false);
    }
  );
});
