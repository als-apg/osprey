// @ts-check
/**
 * Unit tests for the composer's attachments (chat-attach.js) and their transport
 * (chat-client.js), against a real console built by chat-console.js:
 *   npx vitest run tests/interfaces/web_terminal/chat-attach.test.mjs
 *
 * Pinned here: how a file reaches the tray (the paperclip, a paste, a drop); what
 * the tray does with it (the upload, a refusal shown on its chip, a removal that
 * is not left on the server); how a send waits for its uploads and refuses a
 * failed one; the block the agent reads, and the names the bubble shows; and what
 * a refused send gives back. The transport is the real chat-client.js with
 * `fetch` stubbed, so each request asserted on is the one the browser would make.
 */

import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';

import { stampFacilityZone } from '../_support/facility-zone.mjs';

import { createUserEntry } from '../../../src/osprey/interfaces/web_terminal/static/js/chat-blocks.js';
import { buildConsole } from '../../../src/osprey/interfaces/web_terminal/static/js/chat-console.js';

const KEY = '11111111-2222-3333-4444-555555555555';
const STORE = `/srv/agent_data/uploads/${KEY}`;
const STORED_PREFIX = '0123456789abcdef-';
const TEXT_TYPE = 'text/plain';
const BATCH = 'batch-0001-aaaa';

/** @typedef {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/chat-attach.js')} AttachModule */
/** @typedef {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/chat-events.js')} EventsModule */
/** @typedef {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/chat-client.js')} ClientModule */
/** @typedef {ReturnType<typeof buildConsole> & { container: HTMLElement, transforms: Array<(prompt: string) => string>, log: { submits: number }, setKey: (key: string) => void }} Composer */

/** @type {AttachModule} */
let attach;
/** @type {EventsModule} */
let events;
/** @type {ReturnType<typeof vi.fn>} */
let fetchMock;

beforeEach(() => {
  vi.useFakeTimers();
  fetchMock = vi.fn();
  vi.stubGlobal('fetch', fetchMock);
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
  document.body.replaceChildren();
});

/** Let the pending uploads and their follow-ups settle. No fake time passes. */
const flush = () => vi.advanceTimersByTimeAsync(0);

/**
 * @param {number} status
 * @param {unknown} [body]
 */
const json = (status, body) =>
  new Response(body === undefined ? null : JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });

/**
 * @param {string} name
 * @param {number} [bytes]
 * @param {string} [type]
 * @returns {File}
 */
const file = (name, bytes = 1234, type = 'image/png') =>
  new File([new Uint8Array(bytes)], name, { type });

/**
 * The stored answer the server gives for an upload.
 * @param {RequestInit} init
 */
function storedReply(init) {
  const headers = /** @type {Record<string, string>} */ (init.headers);
  const name = decodeURIComponent(headers['X-Osprey-Attachment-Name']);
  const body = /** @type {File} */ (init.body);
  return { name, path: `${STORE}/${STORED_PREFIX}${name}`, size: body.size, type: body.type };
}

/** Answer every upload with the stored file it asked for. */
function answerUploads() {
  fetchMock.mockImplementation(async (_url, init) => {
    if (init.method === 'DELETE') return json(204);
    return json(201, storedReply(init));
  });
}

/**
 * The first element under *root* matching *selector*. A missing element is a
 * test failure, so the cast is the assertion.
 * @param {ParentNode} root
 * @param {string} selector
 * @returns {HTMLElement}
 */
const el = (root, selector) => /** @type {HTMLElement} */ (root.querySelector(selector));

/**
 * @param {ParentNode} root
 * @param {string} selector
 * @returns {HTMLElement[]}
 */
const all = (root, selector) => /** @type {HTMLElement[]} */ ([...root.querySelectorAll(selector)]);

/**
 * Mount a composer in a fresh container. The feature is initialised the way
 * chat-features.js does, and chat.js's submit is stood in for by two listeners:
 * a click on Send, or Enter in the box, counts as a submit.
 * @param {{ key?: string }} [options]
 * @returns {Promise<Composer>}
 */
async function mount({ key: initialKey = KEY } = {}) {
  let key = initialKey;
  vi.resetModules();
  attach = await import('../../../src/osprey/interfaces/web_terminal/static/js/chat-attach.js');
  events = await import('../../../src/osprey/interfaces/web_terminal/static/js/chat-events.js');
  const handles = buildConsole();
  const container = document.createElement('div');
  container.append(handles.bar, handles.messages, handles.overlay, handles.inputArea);
  document.body.append(container);

  /** @type {Array<(prompt: string) => string>} */
  const transforms = [];
  attach.initChatAttachments({
    ...handles,
    container,
    renderer: /** @type {any} */ (null),
    getKey: () => key,
    onBeforeSend: (transform) => {
      transforms.push(transform);
    },
  });

  const log = { submits: 0 };
  handles.sendBtn.addEventListener('click', () => {
    log.submits += 1;
  });
  handles.textarea.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) log.submits += 1;
  });
  /** Move the console to another conversation, as chat.js does before it resets the log. */
  const setKey = (/** @type {string} */ next) => {
    key = next;
  };
  return { ...handles, container, transforms, log, setKey };
}

/**
 * Hand files to the picker, as the browser does after the operator chooses them.
 * @param {Composer} m
 * @param {File[]} files
 */
function choose(m, files) {
  const picker = /** @type {HTMLInputElement} */ (el(m.inputArea, 'input[type="file"]'));
  Object.defineProperty(picker, 'files', { value: files, configurable: true });
  picker.dispatchEvent(new Event('change'));
}

/** @param {Composer} m */
const chipsIn = (m) => all(m.inputArea, '.op-attach-chip');
/** @param {HTMLElement} chip */
const chipName = (chip) => el(chip, '.op-attach-chip-name').textContent ?? '';
/** @param {HTMLElement} chip */
const chipMeta = (chip) => el(chip, '.op-attach-chip-meta').textContent ?? '';
/** @param {Composer} m */
const statusOf = (m) => el(m.inputArea, '.op-attach-status').textContent ?? '';
/** @param {string} method */
const callsWith = (method) => fetchMock.mock.calls.filter(([, init]) => init.method === method);

/**
 * A keydown Enter, as the browser sends it from the box.
 * @param {HTMLElement} target
 */
const enter = (target) =>
  target.dispatchEvent(
    new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true })
  );

/** A bubble's typed text and the names drawn under it. @param {HTMLElement} entry */
const bubbleOf = (entry) => ({
  typed: el(entry, '.op-bubble').firstChild?.textContent ?? '',
  names: all(entry, '.op-attach-file').map((node) => node.textContent),
});

describe('getting a file into the tray', () => {
  test('the paperclip opens the file picker', async () => {
    const m = await mount();
    const picker = /** @type {HTMLInputElement} */ (el(m.inputArea, 'input[type="file"]'));
    const open = vi.spyOn(picker, 'click').mockImplementation(() => {});

    el(m.inputArea, '.op-attach-btn').click();

    expect(open).toHaveBeenCalledTimes(1);
  });

  test('each chosen file gets a chip and is uploaded to this chat, with its message', async () => {
    answerUploads();
    const m = await mount();

    choose(m, [file('shot.png', 2048), file('log.txt', 300, TEXT_TYPE)]);
    expect(chipsIn(m).map(chipName)).toEqual(['shot.png', 'log.txt']);
    await flush();

    expect(fetchMock).toHaveBeenCalledTimes(2);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`/api/chat/${KEY}/attachments`);
    expect(init.method).toBe('POST');
    expect(init.headers['X-Osprey-Attachment-Name']).toBe('shot.png');
    expect(init.headers['X-Osprey-Attachment-Batch']).toMatch(/^[0-9a-f-]{36}$/);
    expect(fetchMock.mock.calls[1][1].headers['X-Osprey-Attachment-Batch']).toBe(
      init.headers['X-Osprey-Attachment-Batch']
    );
    expect(chipsIn(m).map(chipMeta)).toEqual(['2 KB', '300 B']);
  });

  test('a pasted image is filed under a name of its own', async () => {
    answerUploads();
    const m = await mount();
    const paste = new Event('paste', { bubbles: true, cancelable: true });
    Object.defineProperty(paste, 'clipboardData', {
      value: { files: [file('image.png', 512)], getData: () => '' },
    });

    m.textarea.dispatchEvent(paste);
    await flush();

    expect(paste.defaultPrevented).toBe(true);
    expect(chipsIn(m)).toHaveLength(1);
    expect(chipName(chipsIn(m)[0])).toMatch(/^pasted-image-\d{8}-\d{6}\.png$/);
  });

  test('pasted text is left to the box', async () => {
    const m = await mount();
    const paste = new Event('paste', { bubbles: true, cancelable: true });
    Object.defineProperty(paste, 'clipboardData', {
      value: {
        files: [file('x.png')],
        getData: (/** @type {string} */ type) => (type === 'text/plain' ? 'hello' : ''),
      },
    });

    m.textarea.dispatchEvent(paste);

    expect(paste.defaultPrevented).toBe(false);
    expect(chipsIn(m)).toHaveLength(0);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  test('a file dropped on the composer or on the log is attached', async () => {
    answerUploads();
    const m = await mount();

    for (const zone of [m.inputArea, m.messages]) {
      const drop = new Event('drop', { bubbles: true, cancelable: true });
      Object.defineProperty(drop, 'dataTransfer', {
        value: { types: ['Files'], files: [file('d.txt', 10, TEXT_TYPE)] },
      });
      zone.dispatchEvent(drop);
      expect(drop.defaultPrevented).toBe(true);
    }

    expect(chipsIn(m)).toHaveLength(2);
  });

  test('a drag of plain text is not taken', async () => {
    const m = await mount();
    const drop = new Event('drop', { bubbles: true, cancelable: true });
    Object.defineProperty(drop, 'dataTransfer', { value: { types: ['text/plain'], files: [] } });

    m.messages.dispatchEvent(drop);

    expect(drop.defaultPrevented).toBe(false);
    expect(chipsIn(m)).toHaveLength(0);
  });
});

describe('the tray', () => {
  test('removing a stored file deletes it on the server', async () => {
    answerUploads();
    const m = await mount();
    choose(m, [file('a.png')]);
    await flush();

    el(chipsIn(m)[0], '.op-attach-chip-dismiss').click();
    await flush();

    expect(chipsIn(m)).toHaveLength(0);
    const [[url, init]] = callsWith('DELETE');
    expect(url).toBe(`/api/chat/${KEY}/attachments/${STORED_PREFIX}a.png`);
    expect(init.headers['X-Osprey-Attachment-Batch']).toMatch(/^[0-9a-f-]{36}$/);
  });

  test('removing a file still uploading abandons the upload', async () => {
    /** @type {AbortSignal | null} */
    let signal = null;
    fetchMock.mockImplementation((_url, init) => {
      const abort = /** @type {AbortSignal} */ (init.signal);
      signal = abort;
      return new Promise((_resolve, reject) => {
        abort.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')));
      });
    });
    const m = await mount();
    choose(m, [file('slow.png')]);

    el(chipsIn(m)[0], '.op-attach-chip-dismiss').click();
    await flush();

    expect(/** @type {AbortSignal | null} */ (signal)?.aborted).toBe(true);
    expect(chipsIn(m)).toHaveLength(0);
    expect(callsWith('DELETE')).toHaveLength(0);
  });

  test('a refused file shows the refusal on its own chip', async () => {
    fetchMock.mockResolvedValue(
      json(415, { detail: { error: 'unsupported_type', message: 'no' } })
    );
    const m = await mount();
    choose(m, [file('page.html', 10, 'text/html')]);
    await flush();

    const [chip] = chipsIn(m);
    expect(chip.dataset.state).toBe('error');
    expect(chipMeta(chip)).toBe(attach.refusalText({ slug: 'unsupported_type' }));
    expect(statusOf(m)).toMatch(/^Remove the file/);
  });

  test.each([
    [
      'a file over the size cap',
      'file_too_large',
      () => {
        const big = file('huge.bin', 10, 'application/octet-stream');
        Object.defineProperty(big, 'size', { value: attach.MAX_FILE_BYTES + 1 });
        return big;
      },
    ],
    ['an empty file', 'empty_file', () => file('empty.txt', 0, TEXT_TYPE)],
  ])('%s is refused before any byte is sent', async (_name, slug, make) => {
    const m = await mount();

    choose(m, [make()]);

    expect(fetchMock).not.toHaveBeenCalled();
    expect(chipsIn(m)[0].dataset.state).toBe('error');
    expect(chipMeta(chipsIn(m)[0])).toBe(attach.refusalText({ slug }));
  });

  test('a message carries at most ten files', async () => {
    answerUploads();
    const m = await mount();

    choose(
      m,
      Array.from({ length: 11 }, (_, i) => file(`f${i}.txt`, 10, TEXT_TYPE))
    );
    await flush();

    expect(chipsIn(m)).toHaveLength(10);
    expect(fetchMock).toHaveBeenCalledTimes(10);
    expect(statusOf(m)).toContain(String(attach.MAX_FILES_PER_MESSAGE));
    expect(el(m.inputArea, '.op-attach-btn')).toHaveProperty('disabled', true);
  });
});

describe('sending', () => {
  test('the block names each stored path, and the chips leave the tray', async () => {
    answerUploads();
    const m = await mount();
    choose(m, [file('shot.png', 2048), file('log.txt', 300, TEXT_TYPE)]);
    await flush();

    const sent = m.transforms[0]('What is tripping?');

    expect(sent).toBe(
      'What is tripping?\n\nAttached files (read them with the Read tool):\n' +
        `- ${STORE}/${STORED_PREFIX}shot.png (image/png, 2 KB)\n` +
        `- ${STORE}/${STORED_PREFIX}log.txt (text/plain, 300 B)`
    );
    expect(chipsIn(m)).toHaveLength(0);
  });

  test("the operator's bubble keeps the typed text and names the files under it", async () => {
    answerUploads();
    const m = await mount();
    choose(m, [file('shot.png', 2048)]);
    await flush();
    const entry = createUserEntry('What is tripping?');
    m.messages.append(entry);

    m.transforms[0]('What is tripping?');

    expect(bubbleOf(entry)).toEqual({ typed: 'What is tripping?', names: ['shot.png'] });
  });

  test('a send with no files is left to the composer', async () => {
    const m = await mount();

    expect(m.transforms[0]('plain question')).toBe('plain question');
  });

  test.each([
    ['Send', (/** @type {Composer} */ m) => m.sendBtn.click()],
    ['Enter', (/** @type {Composer} */ m) => enter(m.textarea)],
  ])('%s waits for an upload still in flight, then goes', async (_name, submit) => {
    /** @type {() => void} */
    let finish = () => {};
    fetchMock.mockImplementation(
      (_url, init) =>
        new Promise((resolve) => {
          finish = () => resolve(json(201, storedReply(init)));
        })
    );
    const m = await mount();
    m.textarea.value = 'look at this';
    choose(m, [file('a.txt', 10, TEXT_TYPE)]);

    submit(m);
    expect(m.log.submits).toBe(0);
    expect(statusOf(m)).toMatch(/upload finishes/);

    finish();
    await flush();
    expect(m.log.submits).toBe(1);
  });

  test('a failed upload is not sent, and the typed text stays in the box', async () => {
    fetchMock.mockResolvedValue(json(413, { detail: { error: 'chat_full', message: 'full' } }));
    const m = await mount();
    m.textarea.value = 'look at this';
    choose(m, [file('a.txt', 10, TEXT_TYPE)]);
    await flush();

    m.sendBtn.click();

    expect(m.log.submits).toBe(0);
    expect(m.textarea.value).toBe('look at this');
    expect(statusOf(m)).toMatch(/^Remove the file/);
  });

  test('once the failed file is removed, the send goes', async () => {
    fetchMock.mockResolvedValue(json(413, { detail: { error: 'chat_full', message: 'full' } }));
    const m = await mount();
    m.textarea.value = 'look at this';
    choose(m, [file('a.txt', 10, TEXT_TYPE)]);
    await flush();

    el(chipsIn(m)[0], '.op-attach-chip-dismiss').click();
    m.sendBtn.click();

    expect(m.log.submits).toBe(1);
  });

  test('files with no typed words are held, asking for the words until the operator types', async () => {
    answerUploads();
    const m = await mount();
    choose(m, [file('shot.png', 2048)]);
    await flush();

    m.sendBtn.click();

    expect(m.log.submits).toBe(0);
    expect(chipsIn(m)).toHaveLength(1);
    expect(statusOf(m)).toMatch(/^Type what you want/);

    m.textarea.value = 'what is';
    m.textarea.dispatchEvent(new Event('input'));

    expect(statusOf(m)).toBe('');
  });

  test('the wait line clears once the composer is free again', async () => {
    const m = await mount();
    m.textarea.disabled = true;
    choose(m, [file('late.txt', 10, TEXT_TYPE)]);
    expect(statusOf(m)).toMatch(/once the agent has finished/);

    m.textarea.disabled = false;
    await flush();

    expect(statusOf(m)).toBe('');
  });

  test.each([
    ['no frame: the send was refused, so its files go back to the tray', null, true],
    [
      'a replayed frame does not count as the agent taking the message',
      { event: { type: 'text' }, replay: true },
      true,
    ],
    [
      'a live frame: the agent took the message, which releases its files',
      { event: { type: 'text', content: 'On it.' }, replay: false },
      false,
    ],
  ])('a turn that ends after %s', async (_name, frame, back) => {
    answerUploads();
    const m = await mount();
    choose(m, [file('a.png')]);
    await flush();
    m.transforms[0]('q');
    expect(chipsIn(m)).toHaveLength(0);

    if (frame) events.emitChat('frame', frame);
    events.emitChat('turn-end', {});

    if (back) {
      expect(chipsIn(m).map(chipName)).toEqual(['a.png']);
      expect(chipsIn(m)[0].dataset.state).toBe('ready');
    } else {
      expect(chipsIn(m)).toHaveLength(0);
      expect(callsWith('DELETE')).toHaveLength(0);
    }
  });

  test('a re-read of the same conversation keeps the tray and the files it holds', async () => {
    answerUploads();
    const m = await mount();
    choose(m, [file('a.png')]);
    await flush();

    events.emitChat('reset', {});
    await flush();

    expect(chipsIn(m).map(chipName)).toEqual(['a.png']);
    expect(chipsIn(m)[0].dataset.state).toBe('ready');
    expect(callsWith('DELETE')).toHaveLength(0);
  });

  test('a re-read while a sent message waits keeps its files for a refused turn', async () => {
    answerUploads();
    const m = await mount();
    choose(m, [file('a.png')]);
    await flush();
    m.transforms[0]('q');

    events.emitChat('reset', {});
    events.emitChat('turn-end', {});

    expect(chipsIn(m).map(chipName)).toEqual(['a.png']);
    expect(callsWith('DELETE')).toHaveLength(0);
  });

  test('a move to another conversation clears the tray and removes the files it held', async () => {
    answerUploads();
    const m = await mount();
    choose(m, [file('a.png')]);
    await flush();

    m.setKey('other-chat');
    events.emitChat('reset', {});
    await flush();

    expect(chipsIn(m)).toHaveLength(0);
    expect(callsWith('DELETE')).toHaveLength(1);
    expect(String(callsWith('DELETE')[0][0])).toContain(KEY);
  });
});

describe('replay', () => {
  test('a replayed message with a block shows its typed text and the file names', async () => {
    const m = await mount();
    const block = `\n\nAttached files (read them with the Read tool):\n- ${STORE}/${STORED_PREFIX}shot.png (image/png, 2 KB)`;
    const content = `What is tripping?${block}`;
    const entry = createUserEntry(content);
    m.messages.append(entry);

    events.emitChat('user-message', { text: content, replay: true });

    expect(bubbleOf(entry)).toEqual({ typed: 'What is tripping?', names: ['shot.png'] });
  });

  test.each([
    ['without a block', 'just a question'],
    [
      'with a block-like tail that does not parse',
      'hi\n\nAttached files (read them with the Read tool):\nnot a block line',
    ],
  ])('a replayed message %s is drawn as it was typed', async (_name, content) => {
    const m = await mount();
    const entry = createUserEntry(content);
    m.messages.append(entry);

    events.emitChat('user-message', { text: content, replay: true });

    expect(el(entry, '.op-bubble').textContent).toBe(content);
    expect(all(entry, '.op-attach-file')).toHaveLength(0);
  });
});

describe('the block and its helpers', () => {
  beforeEach(async () => {
    attach = await import('../../../src/osprey/interfaces/web_terminal/static/js/chat-attach.js');
  });

  afterEach(() => stampFacilityZone(null));

  test('the block splits back into the typed text and the names', () => {
    const text =
      'Q?' +
      attach.attachmentBlock([
        { path: `${STORE}/${STORED_PREFIX}shot.png`, type: 'image/png', size: 2048 },
        { path: `${STORE}/${STORED_PREFIX}my log (1).txt`, type: TEXT_TYPE, size: 300 },
      ]);

    expect(attach.splitAttachmentBlock(text)).toEqual({
      typed: 'Q?',
      names: ['shot.png', 'my log (1).txt'],
    });
  });

  test('a message with no block is not split', () => {
    expect(attach.splitAttachmentBlock('plain question')).toBeNull();
  });

  test('sizes read as bytes, kilobytes and megabytes', () => {
    expect([0, 1023, 2048, 3 * 1024 * 1024].map(attach.formatSize)).toEqual([
      '0 B',
      '1023 B',
      '2 KB',
      '3.0 MB',
    ]);
  });

  test('a pasted image is named for its time on the facility clock, and for its type', () => {
    const when = new Date('2026-10-22T14:05:09Z');

    stampFacilityZone('Asia/Tokyo');
    expect(attach.pastedImageName(when, 'image/png')).toBe('pasted-image-20261022-230509.png');
    expect(attach.pastedImageName(when, 'image/jpeg')).toBe('pasted-image-20261022-230509.jpg');
    stampFacilityZone('UTC');
    expect(attach.pastedImageName(when, 'image/png')).toBe('pasted-image-20261022-140509.png');
  });

  test('a known refusal has its own sentence, and anything else falls back to one', () => {
    const fallback = attach.refusalText(new Error('network down'));
    expect(fallback).toBeTruthy();
    expect(attach.refusalText({ slug: 'no_such_refusal' })).toBe(fallback);
    expect(attach.refusalText({ slug: 'chat_full' })).not.toBe(fallback);
  });
});

describe('the transport', () => {
  test('an upload sends the bytes and its headers, and a refusal carries its slug', async () => {
    /** @type {ClientModule} */
    const client =
      await import('../../../src/osprey/interfaces/web_terminal/static/js/chat-client.js');
    const sent = file('a.png', 3);
    fetchMock.mockResolvedValueOnce(
      json(201, storedReply({ headers: { 'X-Osprey-Attachment-Name': 'a.png' }, body: sent }))
    );

    const stored = await client.uploadChatAttachment(KEY, sent, { name: 'a.png', batch: BATCH });

    expect(stored).toMatchObject({ name: 'a.png', size: 3, type: 'image/png' });
    const [, init] = fetchMock.mock.calls[0];
    expect(init.method).toBe('POST');
    expect(init.headers['X-Osprey-Attachment-Batch']).toBe(BATCH);

    fetchMock.mockResolvedValueOnce(
      json(415, { detail: { error: 'unsupported_type', message: 'no' } })
    );
    await expect(
      client.uploadChatAttachment(KEY, file('a.exe'), { name: 'a.exe', batch: BATCH })
    ).rejects.toMatchObject({ status: 415, slug: 'unsupported_type' });
  });

  test('a removal names the message only when it has one', async () => {
    /** @type {ClientModule} */
    const client =
      await import('../../../src/osprey/interfaces/web_terminal/static/js/chat-client.js');
    fetchMock.mockResolvedValue(json(204));

    await client.removeChatAttachment(KEY, 'x-y.png', { batch: BATCH });
    await client.removeChatAttachment(KEY, 'x-y.png');

    const [[url, withBatch], [, bare]] = fetchMock.mock.calls;
    expect(url).toBe(`/api/chat/${KEY}/attachments/x-y.png`);
    expect(withBatch.method).toBe('DELETE');
    expect(withBatch.headers['X-Osprey-Attachment-Batch']).toBe(BATCH);
    expect(bare.headers).toEqual({});
  });
});
