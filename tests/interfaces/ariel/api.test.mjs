// @ts-check
/**
 * ARIEL API client: what each call puts on the wire. What is pinned:
 *
 *   - the entries listing sends `cursor`, `page_size`, `view` and the C-FILTER
 *     as canonical JSON text in `filters`, and leaves `filters` out when the
 *     filter selects nothing;
 *   - a search body carries `filters`, `offset` and `view` only when given and
 *     non-empty;
 *   - the facets call sends `q`, `mode`, and `filters`/`advanced_params` as
 *     JSON text, each left out when empty;
 *   - an AbortSignal reaches fetch on GET and POST, and an aborted request
 *     rejects with an AbortError, never an ApiError;
 *   - a refused filter's 422 `detail: {field, reason}` becomes an ApiError
 *     with that reason and field;
 *   - the drafts listing (C-DRAFTS-LIST) and the thread (C-THREAD) are plain
 *     GETs returning the body; a draft delete (C-DRAFT-DELETE) resolves
 *     `undefined` on 204 without reading the empty body, and a 404 stays
 *     distinguishable from a 422;
 *   - create sends `metadata.follows_up` unchanged on the JSON and the upload
 *     path and returns `attachments_published` (C-CREATE-RESULT).
 *
 *   npx vitest run tests/interfaces/ariel/api.test.mjs
 */
import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import {
  ApiError,
  api,
  draftsApi,
  entriesApi,
  facetsApi,
  searchApi,
} from '../../../src/osprey/interfaces/ariel/static/js/api.js';

/** @type {any} */
let fetchMock;

/**
 * Answer every fetch with `body`, or reject with an AbortError when the
 * request's signal is already aborted (what a browser does).
 * @param {any} body
 * @param {{ok?: boolean, status?: number}} [res]
 */
function answer(body, { ok = true, status = 200 } = {}) {
  fetchMock = vi.fn(async (/** @type {string} */ _url, /** @type {any} */ init) => {
    if (init?.signal?.aborted) throw new DOMException('The operation was aborted.', 'AbortError');
    return { ok, status, json: async () => body };
  });
  vi.stubGlobal('fetch', fetchMock);
}

/** @returns {URLSearchParams} the query string of the first fetch */
function sentParams() {
  return new URL(fetchMock.mock.calls[0][0], window.location.origin).searchParams;
}

/** @returns {any} the parsed JSON body of the first fetch */
function sentBody() {
  return JSON.parse(fetchMock.mock.calls[0][1].body);
}

beforeEach(() => answer({ entries: [] }));

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('entriesApi.list', () => {
  test('sends cursor, page size, view and the filter as canonical JSON', async () => {
    await entriesApi.list({
      cursor: 'abc',
      pageSize: 50,
      view: 'row',
      filters: { tags: ['rf', 'RF'], author: ['bob', 'alice'] },
    });
    const p = sentParams();
    expect(new URL(fetchMock.mock.calls[0][0]).pathname).toBe('/api/entries');
    expect(p.get('cursor')).toBe('abc');
    expect(p.get('page_size')).toBe('50');
    expect(p.get('view')).toBe('row');
    expect(p.get('filters')).toBe('{"author":["alice","bob"],"tags":["rf"]}');
    expect(p.has('page')).toBe(false);
  });

  test('an empty filter is left out, and so are an absent cursor and view', async () => {
    await entriesApi.list({ filters: { author: [], time: { start: null, end: null } } });
    const p = sentParams();
    expect(p.has('filters')).toBe(false);
    expect(p.has('cursor')).toBe(false);
    expect(p.has('view')).toBe(false);
    expect(p.get('page_size')).toBe('20');
  });

  test('a time window travels inside filters', async () => {
    await entriesApi.list({ filters: { time: { start: '2026-01-01', end: null } } });
    expect(JSON.parse(/** @type {string} */ (sentParams().get('filters')))).toEqual({
      time: { start: '2026-01-01', end: null },
    });
  });
});

describe('searchApi.search', () => {
  test('carries filters, offset and view', async () => {
    await searchApi.search({
      query: 'quench',
      mode: 'semantic',
      maxResults: 20,
      advancedParams: { similarity: 0.6 },
      filters: { author: ['bob'] },
      offset: 40,
      view: 'row',
    });
    expect(fetchMock.mock.calls[0][0]).toBe('/api/search');
    expect(sentBody()).toEqual({
      query: 'quench',
      mode: 'semantic',
      max_results: 20,
      advanced_params: { similarity: 0.6 },
      filters: { author: ['bob'] },
      offset: 40,
      view: 'row',
    });
  });

  test('leaves filters, offset and view out when absent or empty', async () => {
    await searchApi.search({ query: 'quench', filters: { tags: [] } });
    expect(sentBody()).toEqual({
      query: 'quench',
      mode: 'keyword',
      max_results: 10,
      advanced_params: {},
    });
  });

  test('offset 0 is sent', async () => {
    await searchApi.search({ query: 'q', offset: 0 });
    expect(sentBody().offset).toBe(0);
  });
});

describe('facetsApi.get', () => {
  test('sends q, mode, filters and advanced_params as JSON text', async () => {
    await facetsApi.get({
      query: 'quench',
      mode: 'keyword',
      filters: { source_system: ['elog'] },
      advancedParams: { fuzzy: true },
    });
    const p = sentParams();
    expect(new URL(fetchMock.mock.calls[0][0]).pathname).toBe('/api/facets');
    expect(p.get('q')).toBe('quench');
    expect(p.get('mode')).toBe('keyword');
    expect(p.get('filters')).toBe('{"source_system":["elog"]}');
    expect(p.get('advanced_params')).toBe('{"fuzzy":true}');
  });

  test('an empty browse leaves filters, mode and advanced_params out', async () => {
    await facetsApi.get({ filters: {}, advancedParams: {} });
    const p = sentParams();
    expect(p.get('q')).toBe('');
    expect(p.has('mode')).toBe(false);
    expect(p.has('filters')).toBe(false);
    expect(p.has('advanced_params')).toBe(false);
  });
});

describe('abort', () => {
  test('the signal reaches fetch on GET and POST', async () => {
    const { signal } = new AbortController();
    await facetsApi.get({ signal });
    await entriesApi.list({ signal });
    await searchApi.search({ query: 'q', signal });
    await api.get('/status', {}, { signal });
    for (const call of fetchMock.mock.calls) expect(call[1].signal).toBe(signal);
  });

  test('an aborted request rejects with an AbortError, not an ApiError', async () => {
    const ctl = new AbortController();
    ctl.abort();
    for (const run of [
      () => entriesApi.list({ signal: ctl.signal }),
      () => searchApi.search({ query: 'q', signal: ctl.signal }),
      () => facetsApi.get({ signal: ctl.signal }),
    ]) {
      const err = await run().catch((/** @type {any} */ e) => e);
      expect(err.name).toBe('AbortError');
      expect(err).not.toBeInstanceOf(ApiError);
    }
  });
});

describe('errors', () => {
  test("a refused filter's 422 carries its reason and field", async () => {
    answer({ detail: { field: 'shift', reason: 'unknown dimension' } }, { ok: false, status: 422 });
    const err = await entriesApi.list({ filters: { shift: ['owl'] } }).catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(422);
    expect(err.message).toBe('unknown dimension');
    expect(err.field).toBe('shift');
  });

  test('a string detail is still the message', async () => {
    answer({ detail: 'Unknown mode', code: 'bad_mode' }, { ok: false, status: 400 });
    const err = await searchApi.search({ query: 'q' }).catch((e) => e);
    expect(err.message).toBe('Unknown mode');
    expect(err.code).toBe('bad_mode');
  });
});

describe('draftsApi', () => {
  test('list GETs /drafts and returns the body', async () => {
    const body = {
      drafts: [
        {
          draft_id: 'draft-0123456789ab',
          subject: 'RF trip',
          author: 'bob',
          logbook: 'ops',
          created_at: '2026-11-03T10:00:00Z',
          staged_by: 'ariel-mcp',
          attachment_count: 1,
        },
      ],
    };
    answer(body);
    await expect(draftsApi.list()).resolves.toEqual(body);
    expect(new URL(fetchMock.mock.calls[0][0], window.location.origin).pathname).toBe(
      '/api/drafts'
    );
  });

  test('delete sends DELETE and resolves undefined on a 204 with an empty body', async () => {
    const response = new Response(null, { status: 204 });
    const json = vi.spyOn(response, 'json');
    const text = vi.spyOn(response, 'text');
    fetchMock = vi.fn(async () => response);
    vi.stubGlobal('fetch', fetchMock);

    await expect(draftsApi.delete('draft-0123456789ab')).resolves.toBeUndefined();
    expect(fetchMock.mock.calls[0][0]).toBe('/api/drafts/draft-0123456789ab');
    expect(fetchMock.mock.calls[0][1].method).toBe('DELETE');
    expect(json).not.toHaveBeenCalled();
    expect(text).not.toHaveBeenCalled();
  });

  test('delete encodes the id into the path', async () => {
    fetchMock = vi.fn(async () => new Response(null, { status: 204 }));
    vi.stubGlobal('fetch', fetchMock);
    await draftsApi.delete('a/b c');
    expect(fetchMock.mock.calls[0][0]).toBe('/api/drafts/a%2Fb%20c');
  });

  test('a delete of a missing draft throws a 404 ApiError', async () => {
    fetchMock = vi.fn(
      async () =>
        new Response(JSON.stringify({ detail: 'Draft not found' }), {
          status: 404,
          headers: { 'Content-Type': 'application/json' },
        })
    );
    vi.stubGlobal('fetch', fetchMock);
    const err = await draftsApi.delete('draft-0123456789ab').catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(404);
    expect(err.message).toBe('Draft not found');
  });

  test('a delete with a malformed id throws a 422 ApiError, not a 404', async () => {
    answer({ detail: 'Invalid draft id' }, { ok: false, status: 422 });
    const err = await draftsApi.delete('nope').catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(422);
  });

  test('a list failure throws an ApiError', async () => {
    answer({ detail: 'boom' }, { ok: false, status: 500 });
    const err = await draftsApi.list().catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(500);
  });
});

describe('entriesApi.thread', () => {
  test('GETs /entries/{id}/thread and returns the body', async () => {
    const body = { parent_id: 'p1', parent: { entry_id: 'p1' }, children: [{ entry_id: 'c1' }] };
    answer(body);
    await expect(entriesApi.thread('e 1')).resolves.toEqual(body);
    expect(new URL(fetchMock.mock.calls[0][0], window.location.origin).pathname).toBe(
      '/api/entries/e%201/thread'
    );
  });

  test('an unknown entry throws a 404 ApiError', async () => {
    answer({ detail: 'Entry not found' }, { ok: false, status: 404 });
    const err = await entriesApi.thread('missing').catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(404);
    expect(err.message).toBe('Entry not found');
  });
});

describe('entriesApi create with follows_up', () => {
  const metadata = { follows_up: 'parent-42', other: { x: 1 } };

  test('the JSON path sends metadata.follows_up unchanged and returns attachments_published', async () => {
    const result = {
      entry_id: 'e9',
      message: 'ok',
      sync_status: 'synced',
      source_system: 'elog',
      attachment_count: 0,
      attachments_published: null,
    };
    answer(result);
    await expect(entriesApi.create({ subject: 's', details: 'd', metadata })).resolves.toEqual(
      result
    );
    expect(sentBody().metadata).toEqual(metadata);
  });

  test('the upload path sends metadata.follows_up unchanged and returns attachments_published', async () => {
    const result = { entry_id: 'e9', attachment_count: 1, attachments_published: false };
    answer(result);
    const file = new File(['x'], 'a.txt', { type: 'text/plain' });
    const out = await entriesApi.createWithAttachments({ subject: 's', details: 'd', metadata }, [
      file,
    ]);
    expect(out.attachments_published).toBe(false);
    const form = /** @type {FormData} */ (fetchMock.mock.calls[0][1].body);
    expect(JSON.parse(/** @type {string} */ (form.get('metadata')))).toEqual(metadata);
  });

  test('a refused follows_up surfaces its 422 field', async () => {
    answer(
      { detail: 'parent not found', code: 'invalid_entry_field', field: 'follows_up' },
      { ok: false, status: 422 }
    );
    const err = await entriesApi.create({ subject: 's', details: 'd', metadata }).catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(422);
    expect(err.code).toBe('invalid_entry_field');
    expect(err.field).toBe('follows_up');
  });
});

describe('ids are encoded into the path', () => {
  const awkward = 'a/b#c?d';
  const enc = 'a%2Fb%23c%3Fd';
  const pathOf = () => new URL(fetchMock.mock.calls[0][0], window.location.origin);

  test('entriesApi.get and draftsApi.get encode the id', async () => {
    answer({});
    await entriesApi.get(awkward);
    expect(pathOf().pathname).toBe(`/api/entries/${enc}`);
    expect(pathOf().search).toBe('');
    fetchMock.mockClear();
    answer({});
    await draftsApi.get(awkward);
    expect(pathOf().pathname).toBe(`/api/drafts/${enc}`);
  });

  test('entriesApi.enhancements hits the enhancements path', async () => {
    answer({ entry_id: 'e1', modules: [] });
    await entriesApi.enhancements(awkward);
    expect(pathOf().pathname).toBe(`/api/entries/${enc}/enhancements`);
  });

  test('entriesApi.related hits the related path with and without a limit', async () => {
    answer({ state: 'ok', entries: [] });
    await entriesApi.related(awkward, 7);
    expect(pathOf().pathname).toBe(`/api/entries/${enc}/related`);
    expect(pathOf().searchParams.get('limit')).toBe('7');
    fetchMock.mockClear();
    answer({ state: 'ok', entries: [] });
    await entriesApi.related('e1');
    expect(pathOf().search).toBe('');
  });
});
