// @ts-check
/* Session Activity Log render helpers — server badge color mapping.
 *
 * serverClass keys must match the server names the transcript reader emits
 * (the segment between `mcp__` and the next `__`), which for framework
 * servers are the registry names in src/osprey/registry/mcp.py. A Python
 * parity test (test_server_color_parity.py) pins that full set against the
 * registry; this file covers the mapping behavior itself.
 */
import { describe, expect, test } from 'vitest';

import { serverClass } from '../../../src/osprey/interfaces/web_terminal/static/js/session-helpers.js';

describe('serverClass', () => {
  test('maps the underscored framework server names the reader emits', () => {
    // `osprey_workspace` is the registry name — the pre-rename `workspace`
    // key colored nothing once the server was renamed.
    expect(serverClass('osprey_workspace')).toBe('srv-workspace');
    expect(serverClass('osprey_facility_knowledge')).toBe('srv-facility-knowledge');
  });

  test('maps every framework server to a color, not the grey fallback', () => {
    for (const name of [
      'controls',
      'python',
      'osprey_workspace',
      'ariel',
      'channel-finder',
      'osprey_facility_knowledge',
      'phoebus',
      'bluesky',
      'health',
    ]) {
      expect(serverClass(name), name).not.toBe('srv-unknown');
    }
  });

  test('a facility-declared custom server falls back to the neutral badge', () => {
    expect(serverClass('als_custom_srv')).toBe('srv-unknown');
    expect(serverClass(null)).toBe('srv-unknown');
    expect(serverClass(undefined)).toBe('srv-unknown');
  });
});
