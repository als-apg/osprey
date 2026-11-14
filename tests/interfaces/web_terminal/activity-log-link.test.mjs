// @ts-check
/**
 * Unit tests for activity-log-link.js, the one owner of what a link to the
 * Session Activity Log carries and how the page reads it back:
 *
 *   - isSessionKey: only a canonical, bare, lowercase session UUID
 *   - sessionIdFromQuery: the page reads ?session_id= only when it is a key
 *   - activityLogUrl: the prefixed page path, with ?session_id= only for a key
 *   - openActivityLog: one new tab, the id read at click time
 *   - readHash / writeHash: the page's own selection, `#<transcript id>/<view>`
 *   - shareUrl: the link Copy link gives out, naming the transcript shown
 *   - artifactFileUrl: where an artifact's file is served
 *
 *   npx vitest run tests/interfaces/web_terminal/activity-log-link.test.mjs
 */

import { test, expect, describe, afterEach, vi } from 'vitest';

import {
  isSessionKey,
  sessionIdFromQuery,
  activityLogUrl,
  openActivityLog,
  readHash,
  writeHash,
  shareUrl,
  artifactFileUrl,
  VIEWS,
} from '../../../src/osprey/interfaces/web_terminal/static/js/activity-log-link.js';

const KEY = '11111111-2222-3333-4444-555555555555';
const OTHER_KEY = 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee';

afterEach(() => {
  delete window.__OSPREY_PREFIX__;
  vi.unstubAllGlobals();
  document.body.innerHTML = '';
});

describe('isSessionKey', () => {
  test('accepts a canonical session key', () => {
    expect(isSessionKey(KEY)).toBe(true);
  });

  test.each([
    ['null', null],
    ['undefined', undefined],
    ['empty', ''],
    ['upper-cased', OTHER_KEY.toUpperCase()],
    ['braced', `{${KEY}}`],
    ['trailing newline', `${KEY}\n`],
    ['decorated', 'operator-1a2b3c4d'],
    ['traversal', '../../etc/x'],
    ['36 dashes', '-'.repeat(36)],
    ['36 hex digits, no dashes', 'a'.repeat(36)],
  ])('refuses %s', (_name, value) => {
    expect(isSessionKey(value)).toBe(false);
  });
});

describe('sessionIdFromQuery', () => {
  test('returns a valid session_id', () => {
    expect(sessionIdFromQuery(`?session_id=${KEY}`)).toBe(KEY);
  });

  test.each([
    ['traversal', '?session_id=..%2F..%2Fx'],
    ['empty', ''],
    ['absent', '?other=1'],
  ])('returns null for %s', (_name, search) => {
    expect(sessionIdFromQuery(search)).toBeNull();
  });
});

describe('activityLogUrl', () => {
  test('carries a valid session key', () => {
    expect(activityLogUrl(KEY)).toBe(`/static/session.html?session_id=${KEY}`);
  });

  test.each([
    ['null', null],
    ['malformed', 'nope'],
  ])('carries no parameter for a %s id', (_name, value) => {
    expect(activityLogUrl(value)).toBe('/static/session.html');
  });

  test('is prefixed on a per-user mount', () => {
    window.__OSPREY_PREFIX__ = '/u/alice';
    expect(activityLogUrl(KEY).startsWith('/u/alice/static/session.html')).toBe(true);
  });
});

describe('openActivityLog', () => {
  test('opens the page once in a new tab', () => {
    const open = vi.fn();
    vi.stubGlobal('open', open);

    openActivityLog(() => KEY);

    expect(open).toHaveBeenCalledTimes(1);
    expect(open).toHaveBeenCalledWith(activityLogUrl(KEY), '_blank', 'noopener');
  });

  test('reads the session id at click time', () => {
    const open = vi.fn();
    vi.stubGlobal('open', open);
    const ids = [KEY, OTHER_KEY];
    let calls = 0;
    const getSessionId = () => ids[calls++];

    openActivityLog(getSessionId);
    openActivityLog(getSessionId);

    expect(open).toHaveBeenNthCalledWith(2, activityLogUrl(OTHER_KEY), '_blank', 'noopener');
  });
});

describe('readHash', () => {
  test('reads the transcript and the view', () => {
    expect(readHash(`#${KEY}/tools`)).toEqual({ id: KEY, view: 'tools' });
  });

  test('reads a view with no transcript', () => {
    expect(readHash('#/safety')).toEqual({ id: null, view: 'safety' });
  });

  test('reads a transcript with no view', () => {
    expect(readHash(`#${KEY}`)).toEqual({ id: KEY, view: null });
  });

  test.each([
    ['empty', ''],
    ['bare hash', '#'],
    ['not a transcript id', '#../../x/tools'],
    ['upper-cased id', `#${OTHER_KEY.toUpperCase()}/nope`],
    ['unknown view', '#/everything'],
  ])('reads nothing from %s', (_name, hash) => {
    expect(readHash(hash)).toEqual({ id: null, view: null });
  });

  test('knows every view the page has', () => {
    expect(VIEWS).toEqual(['timeline', 'tools', 'safety', 'artifacts']);
    for (const view of VIEWS) expect(readHash(`#/${view}`).view).toBe(view);
  });
});

describe('writeHash', () => {
  test('names the transcript and the view', () => {
    expect(writeHash(KEY, 'safety')).toBe(`#${KEY}/safety`);
  });

  test('names the view alone when no transcript is selected', () => {
    expect(writeHash(null, 'timeline')).toBe('#/timeline');
  });

  test('is read back by readHash', () => {
    expect(readHash(writeHash(KEY, 'artifacts'))).toEqual({ id: KEY, view: 'artifacts' });
  });
});

describe('shareUrl', () => {
  test('names the transcript on the page itself, without the query', () => {
    const here = {
      origin: 'https://osprey.example',
      pathname: '/u/alice/static/session.html',
    };
    expect(shareUrl(KEY, 'tools', here)).toBe(
      `https://osprey.example/u/alice/static/session.html#${KEY}/tools`
    );
  });
});

describe('artifactFileUrl', () => {
  test('is the gallery file route of the artifact', () => {
    expect(artifactFileUrl('a1b2c3')).toBe('/panel/artifacts/files/a1b2c3/file');
  });

  test('carries the per-user prefix', () => {
    window.__OSPREY_PREFIX__ = '/u/alice';
    expect(artifactFileUrl('a1b2c3')).toBe('/u/alice/panel/artifacts/files/a1b2c3/file');
  });

  test('escapes the id', () => {
    expect(artifactFileUrl('a/b c')).toBe('/panel/artifacts/files/a%2Fb%20c/file');
  });
});
