// @ts-check
/**
 * Unit tests for the Web Terminal's restart endpoint call
 * (terminal.js's `restartTerminal`):
 *   npx vitest run tests/interfaces/web_terminal/terminal-restart.test.mjs
 *
 * `restartTerminal` hits `/api/terminal/restart` via a raw `fetch()` POST
 * (createWebSocket/wsUrl -- api.js's other helpers -- aren't a fit for a
 * bare POST), so the prefix is applied inline here, same as app.js's
 * logout POST (see app-logout.test.mjs's equivalent prefix test). Called
 * with no prior `initTerminal()`/`startTerminal()` -- `restartTerminal`'s
 * own `stopTerminal()`/`term` guards are no-ops when neither has run, so
 * this only exercises the restart fetch itself, not a real WebSocket/xterm
 * round trip (see terminal-resume.test.mjs for that surface).
 */

import { test, expect, describe, afterEach, vi } from 'vitest';

import { restartTerminal } from '../../../src/osprey/interfaces/web_terminal/static/js/terminal.js';

afterEach(() => {
  vi.unstubAllGlobals();
  delete window.__OSPREY_PREFIX__;
});

describe('restartTerminal', () => {
  test('prepends window.__OSPREY_PREFIX__ to the restart POST (multi-user deployments)', async () => {
    window.__OSPREY_PREFIX__ = '/u/alice';
    const fetchMock = vi.fn(() => Promise.resolve({ ok: true }));
    vi.stubGlobal('fetch', fetchMock);

    await restartTerminal();

    expect(fetchMock).toHaveBeenCalledWith('/u/alice/api/terminal/restart', { method: 'POST' });
  });

  test('is byte-identical to the unprefixed request when the prefix is absent', async () => {
    const fetchMock = vi.fn(() => Promise.resolve({ ok: true }));
    vi.stubGlobal('fetch', fetchMock);

    await restartTerminal();

    expect(fetchMock).toHaveBeenCalledWith('/api/terminal/restart', { method: 'POST' });
  });
});
