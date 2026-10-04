// @ts-check
/**
 * Unit tests for activity-log-link.js, the one owner of what a link to the
 * Session Activity Log carries and how the page reads it back:
 *
 *   - isSessionKey: only a canonical, bare, lowercase session UUID
 *   - sessionIdFromQuery: the page reads ?session_id= only when it is a key
 *   - activityLogUrl: the prefixed page path, with ?session_id= only for a key
 *   - openActivityLog: one new tab, the id read at click time
 *   - initActivityLogButton: the rail button opens it; no button, no-op
 *
 *   npx vitest run tests/interfaces/web_terminal/activity-log-link.test.mjs
 */

import { test, expect, describe, afterEach, vi } from 'vitest';

import {
  isSessionKey,
  sessionIdFromQuery,
  activityLogUrl,
  openActivityLog,
  initActivityLogButton,
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

describe('initActivityLogButton', () => {
  test('a click on the rail button opens the activity log on the session', () => {
    document.body.innerHTML = '<button id="panel-activity-btn" type="button"></button>';
    const open = vi.fn();
    vi.stubGlobal('open', open);
    initActivityLogButton(() => KEY);
    const click = new MouseEvent('click', { bubbles: true, cancelable: true });

    document.getElementById('panel-activity-btn')?.dispatchEvent(click);

    expect(open).toHaveBeenCalledTimes(1);
    expect(open).toHaveBeenCalledWith(activityLogUrl(KEY), '_blank', 'noopener');
    expect(click.defaultPrevented).toBe(true);
  });

  test('is a no-op on a page without the button', () => {
    expect(() => initActivityLogButton(() => KEY)).not.toThrow();
  });
});
