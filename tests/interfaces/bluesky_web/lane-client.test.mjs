/**
 * Unit tests for the BLUESKY panel's plan-lane addressing (lane-client.js).
 *
 * Everything under test is pure — roster parsing, lane resolution from the
 * document URL, the `?lane=` spelling on request paths, and the search-string
 * rewrite a lane switch navigates to — so the whole panel half of the
 * sidecar's lane contract is pinned with plain strings, no DOM and no
 * network.
 *
 * The one invariant stated twice on purpose: lane 1 stays OFF the wire. A
 * single-lane deployment's requests and document URLs must be byte-identical
 * to what the panel produced before the lane axis existed, and an unknown
 * lane is pinned loudly, never silently rerouted to lane 1 — the sidecar
 * refuses it (404 `unknown bluesky lane`) and the panel says why.
 */

import { test, expect, describe } from 'vitest';

import {
  LANE_ONE,
  LANE_QUERY_PARAM,
  laneIsKnown,
  laneLabel,
  laneSearch,
  parseLaneRoster,
  resolveLaneFromSearch,
  withLane,
} from '../../../src/osprey/interfaces/bluesky_web/panels/bluesky/lane-client.js';

const TWO_LANES = [
  { lane: 'bluesky', laneTarget: 'live' },
  { lane: 'bluesky_va', laneTarget: 'va' },
];

describe('parseLaneRoster', () => {
  test('parses the sidecar roster, lane_target and all', () => {
    expect(
      parseLaneRoster({
        lanes: [
          { lane: 'bluesky', lane_target: 'live' },
          { lane: 'bluesky_va', lane_target: 'va' },
        ],
      })
    ).toEqual(TWO_LANES);
  });

  test('a declared-nothing target is null, not a guess', () => {
    expect(parseLaneRoster({ lanes: [{ lane: 'bluesky' }] })).toEqual([
      { lane: 'bluesky', laneTarget: null },
    ]);
    expect(parseLaneRoster({ lanes: [{ lane: 'bluesky', lane_target: '' }] })).toEqual([
      { lane: 'bluesky', laneTarget: null },
    ]);
  });

  test.each([null, undefined, 'nope', {}, { lanes: 'x' }, { lanes: [] }])(
    'a malformed or empty answer collapses to the single-lane roster (%j)',
    (body) => {
      // The panel must render regardless, and one lane is the shape every
      // deployment had before the axis existed — never zero lanes.
      expect(parseLaneRoster(body)).toEqual([{ lane: LANE_ONE, laneTarget: null }]);
    }
  );

  test('entries without a usable lane key are dropped, not guessed at', () => {
    expect(
      parseLaneRoster({ lanes: [{ lane: 'bluesky' }, { lane: '' }, { nope: 1 }, null] })
    ).toEqual([{ lane: 'bluesky', laneTarget: null }]);
  });
});

describe('resolveLaneFromSearch', () => {
  test('no parameter is lane 1 — the only lane a single-lane deployment has', () => {
    expect(resolveLaneFromSearch('')).toBe(LANE_ONE);
    expect(resolveLaneFromSearch('?embedded=true&mode=simple')).toBe(LANE_ONE);
  });

  test('an empty parameter names no lane, matching the sidecar', () => {
    expect(resolveLaneFromSearch('?lane=')).toBe(LANE_ONE);
  });

  test('a named lane is returned verbatim, known to the roster or not', () => {
    // Rerouting an unknown lane to lane 1 silently would point the operator
    // at a different machine than the URL names; the sidecar 404s instead.
    expect(resolveLaneFromSearch('?lane=bluesky_va')).toBe('bluesky_va');
    expect(resolveLaneFromSearch('?lane=not-a-lane')).toBe('not-a-lane');
  });
});

describe('laneIsKnown', () => {
  test('holds a lane against the roster', () => {
    expect(laneIsKnown(TWO_LANES, 'bluesky_va')).toBe(true);
    expect(laneIsKnown(TWO_LANES, 'bluesky_live')).toBe(false);
  });
});

describe('withLane', () => {
  test('lane 1 stays bare — single-lane requests are byte-identical', () => {
    expect(withLane('/queue/events', LANE_ONE)).toBe('/queue/events');
    expect(withLane('/draft?client_id=abc', LANE_ONE)).toBe('/draft?client_id=abc');
  });

  test('a second lane rides every path as ?lane=', () => {
    expect(withLane('/queue/start', 'bluesky_va')).toBe('/queue/start?lane=bluesky_va');
    expect(withLane('/bridge/health', 'bluesky_va')).toBe('/bridge/health?lane=bluesky_va');
  });

  test('composes with a path that already carries a query string', () => {
    expect(withLane('/draft?client_id=abc', 'bluesky_va')).toBe(
      '/draft?client_id=abc&lane=bluesky_va'
    );
  });

  test('the parameter name is the shared contract spelling', () => {
    // read_proxy.LANE_QUERY_PARAM — one axis, one spelling, panel included.
    expect(LANE_QUERY_PARAM).toBe('lane');
  });
});

describe('laneLabel', () => {
  test('a lane is labelled by the target it drives', () => {
    expect(laneLabel({ lane: 'bluesky_va', laneTarget: 'va' })).toBe('va');
  });

  test('a target-less entry falls back to the service key, never blank', () => {
    expect(laneLabel({ lane: 'bluesky', laneTarget: null })).toBe('bluesky');
  });
});

describe('laneSearch', () => {
  test('switching to a second lane preserves every host parameter', () => {
    // Dropping ?embedded/?mode/?theme on a lane switch would un-embed the
    // panel or flash it into the wrong theme mid-shift.
    expect(laneSearch('?embedded=true&mode=simple', 'bluesky_va')).toBe(
      '?embedded=true&mode=simple&lane=bluesky_va'
    );
  });

  test('switching back to lane 1 removes the parameter rather than naming it', () => {
    expect(laneSearch('?embedded=true&lane=bluesky_va', LANE_ONE)).toBe('?embedded=true');
    expect(laneSearch('?lane=bluesky_va', LANE_ONE)).toBe('');
  });

  test('replaces an existing lane instead of stacking a second one', () => {
    expect(laneSearch('?lane=bluesky_va', 'bluesky_live')).toBe('?lane=bluesky_live');
  });
});
