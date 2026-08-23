// @ts-check
/**
 * Unit tests for the feedback transport (feedback-client.js):
 *   npx vitest run tests/interfaces/web_terminal/feedback-client.test.mjs
 *
 * The module is DOM-free and fully dependency-injected, so every test drives it
 * through a deps object — no global `fetch`, no real `navigator.clipboard`, no
 * real `window.open`. The one global it does read is `ClipboardItem` (a feature
 * detect, stubbed here because happy-dom does not implement it).
 *
 * The load-bearing property under test is ORDER: for the GitHub/Email channels
 * the clipboard write (POINTER-mode sends only — a FULL-mode draft is complete
 * as opened and touches no clipboard) and the tab/mail-draft navigation must
 * both happen inside the click turn, before the POST settles. Safari revokes
 * the transient user activation across an await, so a copy issued afterwards is
 * silently refused and a `window.open` afterwards is treated as a popup. The
 * tests pin that with a manually-resolved fetch promise and a shared
 * call-order log.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import {
  sendFeedback,
  copyContext,
  BUNDLE_PATH,
  DEFAULT_TITLE,
  FEEDBACK_PATH,
  NOTICE_COPY_FAILED,
  NOTICE_EMPTY_BUNDLE,
  NOTICE_MANUAL_COPY,
  NOTICE_NOT_RECORDED,
  NOTICE_NOT_RECORDED_FULL,
  NOTICE_NO_SESSION,
  NOTICE_NO_TRANSCRIPT,
  NOTICE_PASTE_EMAIL,
  NOTICE_PASTE_GITHUB,
  NOTICE_TOO_LARGE,
} from '../../../src/osprey/interfaces/web_terminal/static/js/feedback-client.js';
import { PASTE_POINTER } from '../../../src/osprey/interfaces/web_terminal/static/js/feedback-prefill.js';

/**
 * Read one query parameter back out of a built URL.
 *
 * @param {string} url
 * @param {string} name
 * @returns {string} the decoded value
 */
function param(url, name) {
  const match = new RegExp(`[?&]${name}=([^&]*)`).exec(url);
  if (match === null) throw new Error(`no ?${name}= in ${url.slice(0, 80)}`);
  return decodeURIComponent(match[1]);
}

/** A promise whose settlement the test controls. */
function deferred() {
  /** @type {(value: any) => void} */
  let resolve = () => {};
  /** @type {(reason?: any) => void} */
  let reject = () => {};
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

/**
 * Minimal `Response` stand-in. The module only ever reads `ok`, `status` and
 * `text()` — it never calls `json()`, which is what keeps a non-JSON proxy
 * error page from throwing.
 * @param {unknown} data
 * @param {{ok?: boolean, status?: number}} [opts]
 */
function jsonResponse(data, opts = {}) {
  const { ok = true, status = 200 } = opts;
  return { ok, status, text: async () => JSON.stringify(data) };
}

/** nginx's own 413 page: HTML, not JSON. */
function htmlErrorResponse(status = 413) {
  return {
    ok: false,
    status,
    text: async () =>
      '<html>\r\n<head><title>413 Request Entity Too Large</title></head>\r\n</html>\r\n',
  };
}

/** Stand-in for the `ClipboardItem` constructor happy-dom lacks. */
class FakeClipboardItem {
  /** @param {Record<string, Promise<Blob|string>|Blob|string>} items */
  constructor(items) {
    this.items = items;
  }
}

/** Read back the text the module handed to `clipboard.write`. */
async function copiedText(/** @type {any} */ items) {
  const value = await items[0].items['text/plain'];
  return typeof value === 'string' ? value : await value.text();
}

/**
 * A deps object with spies for every side effect, plus the call-order log the
 * synchronicity tests assert on.
 * @param {{response?: any, fetchPromise?: Promise<any>, clipboard?: any}} [opts]
 */
function makeDeps(opts = {}) {
  /** @type {string[]} */
  const order = [];
  /** @type {any[]} */
  const written = [];
  /** @type {string[]} */
  const writtenText = [];
  /** @type {string[]} */
  const selectable = [];
  /** @type {any[]} */
  const notices = [];
  /** @type {{url: string, target: string, features: string}[]} */
  const opened = [];
  /** @type {string[]} */
  const navigated = [];

  /** @type {{url: string, body: any}[]} */
  const requests = [];

  const fetchSpy = vi.fn((/** @type {string} */ url, /** @type {any} */ init) => {
    order.push('fetch');
    requests.push({ url, body: JSON.parse(init.body) });
    if (opts.fetchPromise) return opts.fetchPromise;
    return Promise.resolve(opts.response ?? jsonResponse({ id: 'fb-1', payload: 'SERVER PAYLOAD' }));
  });

  const clipboard = opts.clipboard ?? {
    write: vi.fn(async (/** @type {any} */ items) => {
      order.push('write');
      written.push(items);
    }),
    writeText: vi.fn(async (/** @type {string} */ text) => {
      order.push('writeText');
      writtenText.push(text);
    }),
  };

  const deps = {
    fetch: fetchSpy,
    clipboard,
    windowOpen: vi.fn(
      (/** @type {string} */ url, /** @type {string} */ target, /** @type {string} */ features) => {
        order.push('open');
        opened.push({ url, target, features });
      }
    ),
    navigate: vi.fn((/** @type {string} */ url) => {
      order.push('navigate');
      navigated.push(url);
    }),
    showSelectableText: vi.fn((/** @type {string} */ payload) => {
      order.push('selectable');
      selectable.push(payload);
    }),
    onNotice: vi.fn((/** @type {any} */ notice) => {
      notices.push(notice);
    }),
    scrollback: 'line one\nline two',
    metadata: { version: '1.2.3', app_name: 'OSPREY' },
    githubRepo: 'als-apg/osprey',
    email: 'maintainers@example.org',
  };

  return {
    deps,
    order,
    written,
    writtenText,
    selectable,
    notices,
    opened,
    navigated,
    requests,
    fetchSpy,
    clipboard,
  };
}

/** The JSON body of the n-th fetch call. */
function sentBody(/** @type {any} */ fetchSpy, n = 0) {
  return JSON.parse(fetchSpy.mock.calls[n][1].body);
}

/** @type {{text: string, channel: any, metadataOn: boolean, contextOn: boolean, sessionId: string|null}} */
const BASE_FORM = {
  text: 'The rail button does not render in top mode.',
  channel: 'local',
  metadataOn: true,
  contextOn: false,
  sessionId: '11111111-2222-3333-4444-555555555555',
};

beforeEach(() => {
  vi.stubGlobal('ClipboardItem', FakeClipboardItem);
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('sendFeedback — Local channel', () => {
  test('posts text and flags, and omits context fields when the box is unticked', async () => {
    const { deps, fetchSpy } = makeDeps();

    await sendFeedback({ ...BASE_FORM }, deps);

    expect(fetchSpy).toHaveBeenCalledTimes(1);
    expect(fetchSpy.mock.calls[0][0]).toBe(FEEDBACK_PATH);
    expect(fetchSpy.mock.calls[0][1].method).toBe('POST');
    const body = sentBody(fetchSpy);
    expect(body.text).toBe(BASE_FORM.text);
    expect(body.channel).toBe('local');
    expect(body.include_metadata).toBe(true);
    expect(body.include_context).toBe(false);
    expect('session_id' in body).toBe(false);
    expect('scrollback' in body).toBe(false);
  });

  test('carries session_id and scrollback only when context is ticked', async () => {
    const { deps, fetchSpy } = makeDeps();

    await sendFeedback({ ...BASE_FORM, contextOn: true }, deps);

    const body = sentBody(fetchSpy);
    expect(body.include_context).toBe(true);
    expect(body.session_id).toBe(BASE_FORM.sessionId);
    expect(body.scrollback).toBe('line one\nline two');
  });

  test('omits session_id when there is no session, context ticked or not', async () => {
    const { deps, fetchSpy } = makeDeps();

    await sendFeedback({ ...BASE_FORM, contextOn: true, sessionId: null }, deps);

    const body = sentBody(fetchSpy);
    expect('session_id' in body).toBe(false);
    expect(body.scrollback).toBe('line one\nline two');
  });

  test('reports the record id and touches neither clipboard nor window', async () => {
    const { deps, notices } = makeDeps({ response: jsonResponse({ id: 'fb-20260820-abcd1234' }) });

    const result = await sendFeedback({ ...BASE_FORM }, deps);

    expect(result.ok).toBe(true);
    expect(result.id).toBe('fb-20260820-abcd1234');
    expect(result.notice.kind).toBe('success');
    expect(result.notice.message).toContain('Recorded on this deployment');
    expect(result.notice.message).toContain('fb-20260820-abcd1234');
    expect(notices).toEqual([result.notice]);
    expect(deps.clipboard.write).not.toHaveBeenCalled();
    expect(deps.windowOpen).not.toHaveBeenCalled();
    expect(deps.navigate).not.toHaveBeenCalled();
  });

  test('surfaces the no-transcript notice when the server could not find one', async () => {
    const { deps } = makeDeps({
      response: jsonResponse({ id: 'fb-2', context_status: 'transcript_not_found' }),
    });

    const result = await sendFeedback({ ...BASE_FORM, contextOn: true }, deps);

    expect(result.ok).toBe(true);
    expect(result.contextStatus).toBe('transcript_not_found');
    expect(result.notice.kind).toBe('warning');
    expect(result.notice.message).toContain(NOTICE_NO_TRANSCRIPT);
    expect(result.notice.message).toContain('fb-2');
  });

  test('turns a non-JSON 413 into the generic too-large message instead of throwing', async () => {
    const { deps } = makeDeps({ response: htmlErrorResponse(413) });

    const result = await sendFeedback({ ...BASE_FORM }, deps);

    expect(result.ok).toBe(false);
    expect(result.notice.kind).toBe('error');
    expect(result.notice.message).toBe(NOTICE_TOO_LARGE);
    expect(result.id).toBe(null);
  });

  test("prefers the server's own error detail when the body is JSON", async () => {
    const { deps } = makeDeps({
      response: jsonResponse({ detail: 'session_id must be a UUID' }, { ok: false, status: 422 }),
    });

    const result = await sendFeedback({ ...BASE_FORM }, deps);

    expect(result.ok).toBe(false);
    expect(result.notice.message).toBe('session_id must be a UUID');
  });

  test('reports a network failure without throwing', async () => {
    const { deps } = makeDeps({ fetchPromise: Promise.reject(new Error('network down')) });

    const result = await sendFeedback({ ...BASE_FORM }, deps);

    expect(result.ok).toBe(false);
    expect(result.notice.kind).toBe('error');
    expect(result.notice.message).toContain('network down');
  });

  test('routes through the per-user URL prefix', async () => {
    const { deps, fetchSpy } = makeDeps();
    window.__OSPREY_PREFIX__ = '/u/alice';
    try {
      await sendFeedback({ ...BASE_FORM }, deps);
    } finally {
      delete window.__OSPREY_PREFIX__;
    }
    expect(fetchSpy.mock.calls[0][0]).toBe(`/u/alice${FEEDBACK_PATH}`);
  });
});

describe('sendFeedback — GitHub channel', () => {
  test('a pointer-mode send copies and opens the tab before the POST settles', async () => {
    const gate = deferred();
    const { deps, order, fetchSpy } = makeDeps({ fetchPromise: gate.promise });

    const pending = sendFeedback({ ...BASE_FORM, channel: 'github', contextOn: true }, deps);
    // Asserted with no await at all: both side effects must have happened in
    // the caller's own turn, not merely "before the POST resolved". A single
    // deferred microtask is already too late for Safari's user activation.
    expect(order).toEqual(['fetch', 'write', 'open']);
    expect(fetchSpy).toHaveBeenCalledTimes(1);

    // ...and they stay the only calls while the POST is still in flight.
    await Promise.resolve();
    expect(order).toEqual(['fetch', 'write', 'open']);

    gate.resolve(jsonResponse({ id: 'fb-3', payload: 'SERVER PAYLOAD' }));
    const result = await pending;
    expect(result.ok).toBe(true);
    expect(result.opened).toBe(true);
  });

  test('a full-mode send opens the complete draft and leaves the clipboard alone', async () => {
    const { deps, order } = makeDeps();

    const result = await sendFeedback({ ...BASE_FORM, channel: 'github' }, deps);

    expect(order).toEqual(['fetch', 'open']);
    expect(deps.clipboard.write).not.toHaveBeenCalled();
    expect(result.copied).toBe('skipped');
    expect(result.needsPaste).toBe(false);
  });

  test('opens a prefilled new-issue URL with noopener', async () => {
    const { deps } = makeDeps();

    await sendFeedback({ ...BASE_FORM, channel: 'github' }, deps);

    expect(deps.windowOpen).toHaveBeenCalledTimes(1);
    const [url, target, features] = deps.windowOpen.mock.calls[0];
    expect(url.startsWith('https://github.com/als-apg/osprey/issues/new?')).toBe(true);
    expect(url).toContain(encodeURIComponent('The rail button does not render'));
    expect(target).toBe('_blank');
    expect(features).toBe('noopener');
    expect(deps.navigate).not.toHaveBeenCalled();
  });

  test("puts the server's payload on the clipboard, promise-backed", async () => {
    const { deps, written } = makeDeps({
      response: jsonResponse({ id: 'fb-4', payload: 'TEXT\n---\nBUNDLE' }),
    });

    const result = await sendFeedback({ ...BASE_FORM, channel: 'github', contextOn: true }, deps);

    expect(result.copied).toBe('clipboard');
    expect(await copiedText(written[0])).toBe('TEXT\n---\nBUNDLE');
    // The value handed to ClipboardItem must be a promise, not an awaited string.
    expect(typeof written[0][0].items['text/plain'].then).toBe('function');
  });

  test('a rejected POST still opens the tab and leaves the typed text copied', async () => {
    const { deps, written, notices } = makeDeps({
      fetchPromise: Promise.reject(new Error('network down')),
    });

    const result = await sendFeedback({ ...BASE_FORM, channel: 'github', contextOn: true }, deps);

    expect(deps.windowOpen).toHaveBeenCalledTimes(1);
    expect(result.opened).toBe(true);
    expect(result.ok).toBe(false);
    expect(result.notice.kind).toBe('warning');
    expect(result.notice.message).toBe(NOTICE_NOT_RECORDED);
    expect(notices).toEqual([result.notice]);
    expect(await copiedText(written[0])).toBe(BASE_FORM.text);
  });

  test('a failed POST on a full-mode send says the draft is unaffected', async () => {
    const { deps } = makeDeps({ response: htmlErrorResponse(413) });

    const result = await sendFeedback({ ...BASE_FORM, channel: 'github' }, deps);

    expect(result.ok).toBe(false);
    expect(result.error).toBe(NOTICE_TOO_LARGE);
    expect(result.notice.kind).toBe('warning');
    expect(result.notice.message).toBe(NOTICE_NOT_RECORDED_FULL);
    expect(deps.windowOpen).toHaveBeenCalledTimes(1);
    expect(deps.clipboard.write).not.toHaveBeenCalled();
  });

  test('an oversize body flips to the pointer line and auto-copies the payload', async () => {
    const { deps, written } = makeDeps();
    const huge = 'x'.repeat(20000);

    const result = await sendFeedback({ ...BASE_FORM, channel: 'github', text: huge }, deps);

    expect(result.needsPaste).toBe(true);
    const url = deps.windowOpen.mock.calls[0][0];
    expect(url.length).toBeLessThanOrEqual(8000);
    expect(param(url, 'body')).toBe(PASTE_POINTER);
    expect(await copiedText(written[0])).toBe('SERVER PAYLOAD');
  });

  test('drops the metadata block from the prefill when the box is unticked', async () => {
    const { deps } = makeDeps();

    await sendFeedback({ ...BASE_FORM, channel: 'github', metadataOn: false }, deps);

    const url = deps.windowOpen.mock.calls[0][0];
    expect(url).not.toContain(encodeURIComponent('Deployment metadata'));
  });
});

describe('sendFeedback — Email channel', () => {
  test('navigates to a mailto draft inside the click turn', async () => {
    const gate = deferred();
    const { deps, order } = makeDeps({ fetchPromise: gate.promise });

    const pending = sendFeedback({ ...BASE_FORM, channel: 'email', contextOn: true }, deps);

    expect(order).toEqual(['fetch', 'write', 'navigate']);
    expect(deps.navigate.mock.calls[0][0].startsWith('mailto:maintainers@example.org?')).toBe(true);
    expect(deps.windowOpen).not.toHaveBeenCalled();

    gate.resolve(jsonResponse({ id: 'fb-5', payload: 'SERVER PAYLOAD' }));
    await pending;
  });

  test('an over-long draft flips to the pointer line and auto-copies the payload', async () => {
    const { deps, written } = makeDeps();
    const huge = 'y'.repeat(5000);

    const result = await sendFeedback({ ...BASE_FORM, channel: 'email', text: huge }, deps);

    expect(result.needsPaste).toBe(true);
    expect(param(deps.navigate.mock.calls[0][0], 'body')).toBe(PASTE_POINTER);
    expect(deps.clipboard.write).toHaveBeenCalledTimes(1);
    expect(await copiedText(written[0])).toBe('SERVER PAYLOAD');
    expect(result.copied).toBe('clipboard');
  });
});

describe('outbound prefill — title, subject and body mode', () => {
  test('the issue title is generic, never the first line of the report', async () => {
    const { deps } = makeDeps();

    await sendFeedback({ ...BASE_FORM, channel: 'github' }, deps);

    expect(param(deps.windowOpen.mock.calls[0][0], 'title')).toBe(DEFAULT_TITLE);
  });

  test('a thousand-character report never becomes a thousand-character title', async () => {
    const { deps } = makeDeps();
    const oneLongLine = 'the beam dropped and the archiver shows nothing '.repeat(25);

    await sendFeedback({ ...BASE_FORM, channel: 'github', text: oneLongLine }, deps);

    expect(param(deps.windowOpen.mock.calls[0][0], 'title')).toBe(DEFAULT_TITLE);
  });

  test('the title names the session when context is attached', async () => {
    const { deps } = makeDeps();

    await sendFeedback({ ...BASE_FORM, channel: 'github', contextOn: true }, deps);

    expect(param(deps.windowOpen.mock.calls[0][0], 'title')).toBe(
      `${DEFAULT_TITLE} (session 11111111)`
    );
  });

  test('an explicit title still wins', async () => {
    const { deps } = makeDeps();

    await sendFeedback(
      { ...BASE_FORM, channel: 'github', title: 'Rail button dead in top mode' },
      deps
    );

    expect(param(deps.windowOpen.mock.calls[0][0], 'title')).toBe('Rail button dead in top mode');
  });

  test('the mail subject follows the same rule', async () => {
    const { deps } = makeDeps();

    await sendFeedback({ ...BASE_FORM, channel: 'email', contextOn: true }, deps);

    expect(param(deps.navigate.mock.calls[0][0], 'subject')).toBe(
      `${DEFAULT_TITLE} (session 11111111)`
    );
  });

  test('with context attached the body is the pointer line alone', async () => {
    const { deps } = makeDeps();

    await sendFeedback({ ...BASE_FORM, channel: 'github', contextOn: true }, deps);

    // No text, no metadata block: the whole report travels on the clipboard,
    // so anything prefilled here would be duplicated by the paste.
    expect(param(deps.windowOpen.mock.calls[0][0], 'body')).toBe(PASTE_POINTER);
  });

  test('the email draft follows the same pointer rule', async () => {
    const { deps } = makeDeps();

    await sendFeedback({ ...BASE_FORM, channel: 'email', contextOn: true }, deps);

    expect(param(deps.navigate.mock.calls[0][0], 'body')).toBe(PASTE_POINTER);
  });

  test('without context the body carries the text and metadata, no pointer', async () => {
    const { deps } = makeDeps();

    await sendFeedback({ ...BASE_FORM, channel: 'github' }, deps);

    const body = param(deps.windowOpen.mock.calls[0][0], 'body');
    expect(body).toContain(BASE_FORM.text);
    expect(body).toContain('Deployment metadata');
    expect(body).not.toContain(PASTE_POINTER);
  });
});

describe('outbound notices — the paste step is announced', () => {
  test('a successful GitHub send with context points at the clipboard paste', async () => {
    const { deps } = makeDeps();

    const result = await sendFeedback({ ...BASE_FORM, channel: 'github', contextOn: true }, deps);

    expect(result.notice.kind).toBe('success');
    expect(result.notice.message).toBe(
      `Recorded on this deployment (fb-1) — ${NOTICE_PASTE_GITHUB}`
    );
  });

  test('a successful email send with context points at the clipboard paste', async () => {
    const { deps } = makeDeps();

    const result = await sendFeedback({ ...BASE_FORM, channel: 'email', contextOn: true }, deps);

    expect(result.notice.message).toBe(
      `Recorded on this deployment (fb-1) — ${NOTICE_PASTE_EMAIL}`
    );
  });

  test('a truncated body earns the paste hint even without context', async () => {
    const { deps } = makeDeps();

    const result = await sendFeedback(
      { ...BASE_FORM, channel: 'email', text: 'y'.repeat(5000) },
      deps
    );

    expect(result.notice.message).toBe(
      `Recorded on this deployment (fb-1) — ${NOTICE_PASTE_EMAIL}`
    );
  });

  test('an outbound send with nothing to paste keeps the plain confirmation', async () => {
    const { deps } = makeDeps();

    const result = await sendFeedback({ ...BASE_FORM, channel: 'github' }, deps);

    expect(result.notice.message).toBe('Recorded on this deployment (fb-1)');
  });

  test('the local channel never carries a paste hint', async () => {
    const { deps } = makeDeps();

    const result = await sendFeedback({ ...BASE_FORM, contextOn: true }, deps);

    expect(result.notice.message).toBe('Recorded on this deployment (fb-1)');
  });
});

describe('clipboard fallback ladder', () => {
  test('falls back to writeText when ClipboardItem is unavailable', async () => {
    vi.stubGlobal('ClipboardItem', undefined);
    const { deps, writtenText } = makeDeps();

    const result = await sendFeedback({ ...BASE_FORM, channel: 'github', contextOn: true }, deps);

    expect(deps.clipboard.write).not.toHaveBeenCalled();
    expect(writtenText).toEqual(['SERVER PAYLOAD']);
    expect(result.copied).toBe('clipboard-text');
    expect(deps.windowOpen).toHaveBeenCalledTimes(1);
  });

  test('falls back to writeText when write() rejects', async () => {
    /** @type {string[]} */
    const sink = [];
    const { deps } = makeDeps({
      clipboard: {
        write: vi.fn(async () => {
          throw new Error('NotAllowedError');
        }),
        writeText: vi.fn(async (/** @type {string} */ text) => {
          sink.push(text);
        }),
      },
    });
    const result = await sendFeedback({ ...BASE_FORM, channel: 'github', contextOn: true }, deps);

    expect(result.copied).toBe('clipboard-text');
    expect(sink).toEqual(['SERVER PAYLOAD']);
  });

  test('falls back to a selectable payload when the clipboard API is absent', async () => {
    const { deps, selectable } = makeDeps({ clipboard: {} });

    const result = await sendFeedback({ ...BASE_FORM, channel: 'github', contextOn: true }, deps);

    expect(result.copied).toBe('manual');
    expect(selectable).toEqual(['SERVER PAYLOAD']);
    expect(result.notice.kind).toBe('warning');
    expect(result.notice.message).toBe(NOTICE_MANUAL_COPY);
  });

  test('reports a hard copy failure when no rung is available', async () => {
    const { deps } = makeDeps({ clipboard: {} });
    // @ts-expect-error - deliberately removing the last rung
    deps.showSelectableText = undefined;

    const result = await sendFeedback({ ...BASE_FORM, channel: 'github', contextOn: true }, deps);

    expect(result.copied).toBe('none');
    expect(result.notice.kind).toBe('error');
    expect(result.notice.message).toBe(NOTICE_COPY_FAILED);
    // The tab still opened — the copy failing is not a reason to swallow it.
    expect(deps.windowOpen).toHaveBeenCalledTimes(1);
  });
});

describe('copyContext', () => {
  test('posts the bundle request with a UTF-8 byte length, not a UTF-16 one', async () => {
    const { deps, fetchSpy } = makeDeps({ response: jsonResponse({ bundle: 'CONTEXT' }) });
    // Each rocket is 4 UTF-8 bytes but 2 UTF-16 units: length 6, utf8Length 12.
    const emoji = '🚀🚀🚀';

    await copyContext({ ...BASE_FORM, contextOn: true, text: emoji }, deps);

    expect(fetchSpy.mock.calls[0][0]).toBe(BUNDLE_PATH);
    const body = sentBody(fetchSpy);
    expect(body.session_id).toBe(BASE_FORM.sessionId);
    expect(body.scrollback).toBe('line one\nline two');
    expect(body.text_len).toBe(12);
    expect(body.include_metadata).toBe(true);
    expect(emoji.length).toBe(6);
  });

  test('carries the metadata checkbox, so Copy renders what Send would', async () => {
    const { deps, fetchSpy } = makeDeps({ response: jsonResponse({ bundle: 'CONTEXT' }) });

    await copyContext({ ...BASE_FORM, contextOn: true, metadataOn: false }, deps);

    expect(sentBody(fetchSpy).include_metadata).toBe(false);
  });

  test('copies the server-sized bundle verbatim and writes no record', async () => {
    const { deps, written, fetchSpy } = makeDeps({
      response: jsonResponse({ bundle: '## Session context\n- one' }),
    });

    const result = await copyContext({ ...BASE_FORM, contextOn: true }, deps);

    expect(fetchSpy).toHaveBeenCalledTimes(1);
    expect(fetchSpy.mock.calls[0][0]).toBe(BUNDLE_PATH);
    expect(await copiedText(written[0])).toBe('## Session context\n- one');
    expect(result.ok).toBe(true);
    expect(result.copied).toBe('clipboard');
    expect(result.notice.kind).toBe('success');
    expect(deps.windowOpen).not.toHaveBeenCalled();
    expect(deps.navigate).not.toHaveBeenCalled();
  });

  test('issues the clipboard write before the POST settles', async () => {
    const gate = deferred();
    const { deps, order } = makeDeps({ fetchPromise: gate.promise });

    const pending = copyContext({ ...BASE_FORM, contextOn: true }, deps);

    expect(order).toEqual(['fetch', 'write']);

    gate.resolve(jsonResponse({ bundle: 'CONTEXT' }));
    await pending;
  });

  test('turns a non-JSON 413 into the generic too-large message', async () => {
    const { deps, writtenText } = makeDeps({ response: htmlErrorResponse(413) });

    const result = await copyContext({ ...BASE_FORM, contextOn: true }, deps);

    expect(result.ok).toBe(false);
    expect(result.notice.kind).toBe('error');
    expect(result.notice.message).toBe(NOTICE_TOO_LARGE);
    // Nothing may land on the clipboard when there is no bundle to copy.
    expect(writtenText).toEqual([]);
    expect(result.copied).toBe('none');
  });

  test('never puts an empty string on the clipboard when the bundle is missing', async () => {
    const { deps, writtenText, selectable } = makeDeps({ response: jsonResponse({}) });

    const result = await copyContext({ ...BASE_FORM, contextOn: true }, deps);

    expect(result.ok).toBe(false);
    expect(result.copied).toBe('none');
    expect(result.notice.kind).toBe('error');
    expect(result.notice.message).toBe(NOTICE_EMPTY_BUNDLE);
    // Overwriting whatever the user already had with '' would be worse than
    // failing: nothing may reach the lower rungs either.
    expect(writtenText).toEqual([]);
    expect(selectable).toEqual([]);
  });

  test('refuses without a session instead of posting', async () => {
    const { deps, fetchSpy } = makeDeps();

    const result = await copyContext({ ...BASE_FORM, contextOn: true, sessionId: null }, deps);

    expect(fetchSpy).not.toHaveBeenCalled();
    expect(result.ok).toBe(false);
    expect(result.notice.message).toBe(NOTICE_NO_SESSION);
  });
});
