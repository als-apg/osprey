// @ts-check
/**
 * The Activity page as a suite mounts it: the body of the real `session.html`,
 * and a `fetch` that answers the page's three reads.
 */

import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { vi } from 'vitest';

import { payload } from './activity-log.mjs';

const PAGE = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  '../../../src/osprey/interfaces/web_terminal/static/session.html'
);

/**
 * The page's body markup as the server sends it, without its scripts.
 * @returns {string}
 */
export function shellMarkup() {
  const html = readFileSync(PAGE, 'utf8');
  const body = html.slice(html.indexOf('<body>') + '<body>'.length, html.indexOf('</body>'));
  return body.replace(/<script[\s\S]*?<\/script>/g, '');
}

/**
 * One row of `GET /api/sessions`.
 * @param {string} id
 * @param {string} title
 * @returns {object}
 */
export function sessionRow(id, title) {
  return {
    transcript_id: id,
    session_key: null,
    title,
    state: 'idle',
    first_ts: '2026-11-12T09:00:00+00:00',
    last_ts: '2026-11-12T09:30:00+00:00',
  };
}

/**
 * Stub `fetch` with the page's reads. The answers are read at call time, so a
 * test changes what the next read returns by assigning to `answers`; an `Error`
 * as an answer is a read that fails.
 * @param {{activity?: unknown, sessions?: unknown, result?: unknown}} [over]
 */
export function stubActivityFetch(over = {}) {
  /** @type {{activity: unknown, sessions: unknown, result: unknown}} */
  const answers = {
    activity: payload(),
    sessions: { sessions: [] },
    result: { text: 'the whole result', size: 16, record: 'complete' },
    ...over,
  };
  const fetchStub = vi.fn((/** @type {unknown} */ url) => {
    const target = String(url);
    const body = target.includes('/api/session-activity/result')
      ? answers.result
      : target.includes('/api/session-activity')
        ? answers.activity
        : target.includes('/api/sessions')
          ? answers.sessions
          : {};
    if (body instanceof Error) return Promise.reject(body);
    return Promise.resolve({
      status: 200,
      ok: true,
      json: () => Promise.resolve(structuredClone(body)),
    });
  });
  vi.stubGlobal('fetch', fetchStub);
  /** Every URL fetched so far, in order. */
  const urls = () => fetchStub.mock.calls.map((args) => String(args[0]));
  return {
    answers,
    fetchStub,
    urls,
    /** The session reads so far: `/api/session-activity`, not its result route. */
    reads: () =>
      urls().filter((u) => u.includes('/api/session-activity') && !u.includes('/result')),
  };
}
