import { describe, expect, test } from 'vitest';

import { LABELS } from '../../../src/osprey/interfaces/channel_finder/static/js/labels.js';

describe('the channel and server words', () => {
  test('each word is stated once, as a non-empty string', () => {
    expect(Object.keys(LABELS).sort()).toEqual([
      'channel',
      'channels',
      'directory',
      'server',
      'serversView',
    ]);
    for (const word of Object.values(LABELS)) {
      expect(typeof word).toBe('string');
      expect(word.length).toBeGreaterThan(0);
    }
  });

  test('the words cannot be changed at run time', () => {
    expect(Object.isFrozen(LABELS)).toBe(true);
  });
});
