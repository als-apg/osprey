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
  test('a result ends the turn', () => {
    expect(endsTurn({ type: 'result' })).toBe(true);
  });

  test('a terminal error ends the turn', () => {
    expect(endsTurn({ type: 'error', error_type: 'ProcessError', message: 'gone' })).toBe(true);
    expect(endsTurn({ type: 'error', message: 'no type' })).toBe(true);
  });

  test('an error the agent reports inside a message does not', () => {
    expect(endsTurn({ type: 'error', error_type: 'AssistantMessageError' })).toBe(false);
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
