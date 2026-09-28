// @ts-check
/* Session Activity Log render helpers — server badge color mapping.
 *
 * serverClass keys must match the server names the transcript reader emits
 * (the segment between `mcp__` and the next `__`), which for framework
 * servers are the registry names in src/osprey/registry/mcp.py. A Python
 * parity test (test_server_color_parity.py) pins that full set against the
 * registry; this file covers the mapping behavior itself.
 */
import { afterEach, describe, expect, test } from 'vitest';

import { serverClass, ts } from '../../../src/osprey/interfaces/web_terminal/static/js/session-helpers.js';
import { FACILITY_ZONE, stampFacilityZone } from '../_support/facility-zone.mjs';

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
      'event_dispatcher',
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

describe('ts', () => {
  afterEach(() => stampFacilityZone(null));

  test('reads an instant on the stamped facility clock, 24-hour with seconds', () => {
    stampFacilityZone(FACILITY_ZONE);
    const expected = new Intl.DateTimeFormat(undefined, {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hourCycle: 'h23',
      timeZone: FACILITY_ZONE,
    }).format(new Date('2026-01-15T20:04:05Z'));
    expect(ts('2026-01-15T20:04:05Z')).toBe(expected);
  });

  test('empty and unparseable input render nothing', () => {
    stampFacilityZone(FACILITY_ZONE);
    expect(ts('')).toBe('');
    expect(ts(null)).toBe('');
    expect(ts('not-a-time')).toBe('');
  });
});
