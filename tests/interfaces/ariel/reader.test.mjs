// @ts-check
/**
 * The ARIEL reader module:
 *   npx vitest run tests/interfaces/ariel/reader.test.mjs
 *
 * `showInReader` renders loading, then the entry (or the error state) into the
 * container it is given; only the newest call per container renders or opens
 * an attachment. The lightbox reports opening and closing through callbacks,
 * `closeLightbox` closes it from outside, and the module has no `Esc` handling
 * of its own.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

vi.mock('../../../src/osprey/interfaces/ariel/static/js/api.js', async (importOriginal) => {
  const actual = /** @type {any} */ (await importOriginal());
  return {
    ...actual,
    entriesApi: { ...actual.entriesApi, get: vi.fn() },
  };
});

import { entriesApi } from '../../../src/osprey/interfaces/ariel/static/js/api.js';
import {
  showInReader,
  renderEntryDetail,
  clearReader,
  closeLightbox,
  showImageLightbox,
} from '../../../src/osprey/interfaces/ariel/static/js/reader.js';

const RENDITION = '/attachments/att-0123456789ab/rendition';
const RENDITION_SRC = '/api/attachments/att-0123456789ab/rendition';
const ORIGINAL_SRC = '/api/attachments/att-0123456789ab';

/**
 * @param {string} id
 * @param {any[]} [attachments]
 * @returns {any}
 */
function entry(id, attachments = []) {
  return {
    entry_id: id,
    timestamp: '2026-01-01T00:00:00Z',
    author: 'author',
    source_system: 'sys',
    raw_text: `subject of ${id}\nbody of ${id}`,
    keywords: [],
    attachments,
  };
}

const PICTURE = {
  attachment_id: 'att-0123456789ab',
  filename: 'beam.png',
  mime_type: 'image/png',
  viewable: true,
  display_url: RENDITION,
};

/** @returns {{promise: Promise<any>, resolve: (v: any) => void, reject: (e: any) => void}} */
function deferred() {
  /** @type {(v: any) => void} */
  let resolve = () => {};
  /** @type {(e: any) => void} */
  let reject = () => {};
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

/** @returns {HTMLElement|null} */
const lightbox = () => document.getElementById('image-lightbox');

/** @type {HTMLElement} */
let container;

beforeEach(() => {
  document.body.innerHTML = '<div id="ariel-reader"><div id="reader-body"></div></div>';
  container = /** @type {HTMLElement} */ (document.getElementById('reader-body'));
  vi.clearAllMocks();
  vi.spyOn(console, 'error').mockImplementation(() => {});
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('container rendering', () => {
  test('renders loading into the given container, then the entry', async () => {
    const d = deferred();
    vi.mocked(entriesApi.get).mockReturnValueOnce(d.promise);
    const pending = showInReader(container, 'e-1');
    expect(container.textContent).toContain('Loading entry...');
    d.resolve(entry('e-1'));
    const shown = await pending;
    expect(shown.entry_id).toBe('e-1');
    expect(container.querySelector('.entry-detail-title')?.textContent).toBe('subject of e-1');
    expect(entriesApi.get).toHaveBeenCalledWith('e-1');
  });

  test('renders only into the container it is given', async () => {
    document.body.insertAdjacentHTML('beforeend', '<div id="other">untouched</div>');
    vi.mocked(entriesApi.get).mockResolvedValueOnce(entry('e-1'));
    await showInReader(container, 'e-1');
    expect(document.getElementById('other')?.textContent).toBe('untouched');
    expect(document.getElementById('ariel-reader')?.firstElementChild).toBe(container);
  });

  test('renderEntryDetail is exported and escapes entry text', () => {
    const e = entry('e-2');
    e.raw_text = '<img src=x onerror=alert(1)>\nbody';
    renderEntryDetail(container, e);
    expect(container.querySelector('img[onerror]')).toBeNull();
    expect(container.querySelector('.entry-detail-title')?.textContent).toBe(
      '<img src=x onerror=alert(1)>'
    );
  });

  test('clearReader empties the container and cancels an in-flight load', async () => {
    const d = deferred();
    vi.mocked(entriesApi.get).mockReturnValueOnce(d.promise);
    const pending = showInReader(container, 'e-1');
    clearReader(container);
    expect(container.innerHTML).toBe('');
    d.resolve(entry('e-1'));
    expect(await pending).toBeNull();
    expect(container.innerHTML).toBe('');
  });
});

describe('error state', () => {
  test('a failing id renders "Failed to Load Entry" and resolves null', async () => {
    vi.mocked(entriesApi.get).mockRejectedValueOnce(new Error('HTTP 404: Not Found'));
    expect(await showInReader(container, 'missing')).toBeNull();
    expect(container.textContent).toContain('Failed to Load Entry');
  });

  test('an empty response is treated as unknown', async () => {
    vi.mocked(entriesApi.get).mockResolvedValueOnce(null);
    expect(await showInReader(container, 'missing')).toBeNull();
    expect(container.textContent).toContain('Failed to Load Entry');
  });
});

describe('out-of-order responses', () => {
  test('an older response resolving last does not overwrite the newer entry', async () => {
    const first = deferred();
    const second = deferred();
    vi.mocked(entriesApi.get)
      .mockReturnValueOnce(first.promise)
      .mockReturnValueOnce(second.promise);
    const p1 = showInReader(container, 'old');
    const p2 = showInReader(container, 'new');
    second.resolve(entry('new'));
    expect((await p2).entry_id).toBe('new');
    first.resolve(entry('old'));
    expect(await p1).toBeNull();
    expect(container.querySelector('.entry-detail-title')?.textContent).toBe('subject of new');
  });

  test('an older failure resolving last does not paint the error state', async () => {
    const first = deferred();
    const second = deferred();
    vi.mocked(entriesApi.get)
      .mockReturnValueOnce(first.promise)
      .mockReturnValueOnce(second.promise);
    const p1 = showInReader(container, 'old');
    const p2 = showInReader(container, 'new');
    second.resolve(entry('new'));
    await p2;
    first.reject(new Error('boom'));
    expect(await p1).toBeNull();
    expect(container.textContent).not.toContain('Failed to Load Entry');
  });

  test('a superseded call never opens its attachment', async () => {
    const first = deferred();
    const second = deferred();
    vi.mocked(entriesApi.get)
      .mockReturnValueOnce(first.promise)
      .mockReturnValueOnce(second.promise);
    const onAttachmentOpen = vi.fn();
    const p1 = showInReader(container, 'old', {
      attachmentId: PICTURE.attachment_id,
      onAttachmentOpen,
    });
    const p2 = showInReader(container, 'new');
    second.resolve(entry('new'));
    await p2;
    first.resolve(entry('old', [PICTURE]));
    await p1;
    expect(lightbox()).toBeNull();
    expect(onAttachmentOpen).not.toHaveBeenCalled();
  });

  test('sequences are per container', async () => {
    document.body.insertAdjacentHTML('beforeend', '<div id="second"></div>');
    const other = /** @type {HTMLElement} */ (document.getElementById('second'));
    const first = deferred();
    vi.mocked(entriesApi.get).mockReturnValueOnce(first.promise).mockResolvedValueOnce(entry('b'));
    const p1 = showInReader(container, 'a');
    await showInReader(other, 'b');
    first.resolve(entry('a'));
    expect((await p1).entry_id).toBe('a');
    expect(container.querySelector('.entry-detail-title')?.textContent).toBe('subject of a');
  });
});

describe('attachments and callbacks', () => {
  test('attachmentId opens the lightbox for the matching viewable attachment', async () => {
    vi.mocked(entriesApi.get).mockResolvedValueOnce(entry('e-1', [PICTURE]));
    const onAttachmentOpen = vi.fn();
    await showInReader(container, 'e-1', { attachmentId: PICTURE.attachment_id, onAttachmentOpen });
    const box = lightbox();
    expect(box).not.toBeNull();
    expect(box?.querySelector('img')?.getAttribute('src')).toBe(RENDITION_SRC);
    expect(box?.querySelector('a[data-original]')?.getAttribute('href')).toBe(ORIGINAL_SRC);
    expect(onAttachmentOpen).toHaveBeenCalledWith(PICTURE.attachment_id);
  });

  test.each([
    ['unknown id', 'att-nope', [PICTURE]],
    ['not viewable', PICTURE.attachment_id, [{ ...PICTURE, viewable: false }]],
    ['unsafe url', PICTURE.attachment_id, [{ ...PICTURE, display_url: 'javascript:alert(1)' }]],
  ])('attachmentId with %s leaves the entry open without a lightbox', async (_label, id, atts) => {
    vi.mocked(entriesApi.get).mockResolvedValueOnce(entry('e-1', atts));
    const onAttachmentOpen = vi.fn();
    const shown = await showInReader(container, 'e-1', { attachmentId: id, onAttachmentOpen });
    expect(shown.entry_id).toBe('e-1');
    expect(lightbox()).toBeNull();
    expect(onAttachmentOpen).not.toHaveBeenCalled();
  });

  test('a thumbnail click in the container opens the lightbox and calls onAttachmentOpen', async () => {
    vi.mocked(entriesApi.get).mockResolvedValueOnce(entry('e-1', [PICTURE]));
    const onAttachmentOpen = vi.fn();
    await showInReader(container, 'e-1', { onAttachmentOpen });
    expect(lightbox()).toBeNull();
    const thumb = /** @type {HTMLElement} */ (container.querySelector('[data-lightbox-url] img'));
    thumb.click();
    expect(lightbox()).not.toBeNull();
    expect(onAttachmentOpen).toHaveBeenCalledWith(PICTURE.attachment_id);
  });

  test('the delegation is bound once however many entries are shown', async () => {
    vi.mocked(entriesApi.get).mockResolvedValue(entry('e-1', [PICTURE]));
    const onAttachmentOpen = vi.fn();
    await showInReader(container, 'e-1', { onAttachmentOpen });
    await showInReader(container, 'e-1', { onAttachmentOpen });
    /** @type {HTMLElement} */ (container.querySelector('[data-lightbox-url]')).click();
    expect(onAttachmentOpen).toHaveBeenCalledTimes(1);
  });

  test('dismissing the lightbox by its scrim calls onAttachmentClose', async () => {
    vi.mocked(entriesApi.get).mockResolvedValueOnce(entry('e-1', [PICTURE]));
    const onAttachmentClose = vi.fn();
    await showInReader(container, 'e-1', {
      attachmentId: PICTURE.attachment_id,
      onAttachmentClose,
    });
    const box = /** @type {HTMLElement} */ (lightbox());
    box.querySelector('img')?.click();
    expect(lightbox()).not.toBeNull();
    expect(onAttachmentClose).not.toHaveBeenCalled();
    box.click();
    expect(lightbox()).toBeNull();
    expect(onAttachmentClose).toHaveBeenCalledTimes(1);
  });

  test('showImageLightbox takes the two callbacks directly', () => {
    const onAttachmentOpen = vi.fn();
    const onAttachmentClose = vi.fn();
    showImageLightbox(RENDITION_SRC, 'beam.png', null, 'att-1', {
      onAttachmentOpen,
      onAttachmentClose,
    });
    expect(onAttachmentOpen).toHaveBeenCalledWith('att-1');
    /** @type {HTMLElement} */ (lightbox()).click();
    expect(onAttachmentClose).toHaveBeenCalledTimes(1);
  });
});

describe('closeLightbox', () => {
  test('closes the lightbox the container opened, without calling onAttachmentClose', async () => {
    vi.mocked(entriesApi.get).mockResolvedValueOnce(entry('e-1', [PICTURE]));
    const onAttachmentClose = vi.fn();
    await showInReader(container, 'e-1', {
      attachmentId: PICTURE.attachment_id,
      onAttachmentClose,
    });
    expect(lightbox()).not.toBeNull();
    closeLightbox(container);
    expect(lightbox()).toBeNull();
    expect(onAttachmentClose).not.toHaveBeenCalled();
    // The entry stays rendered.
    expect(container.querySelector('.entry-detail-title')).not.toBeNull();
  });

  test('is a no-op when nothing is open', () => {
    expect(() => closeLightbox(container)).not.toThrow();
  });
});

describe('no Esc handling inside the module', () => {
  test('opening the lightbox adds no keydown listener', async () => {
    const docSpy = vi.spyOn(document, 'addEventListener');
    const winSpy = vi.spyOn(window, 'addEventListener');
    vi.mocked(entriesApi.get).mockResolvedValueOnce(entry('e-1', [PICTURE]));
    await showInReader(container, 'e-1', { attachmentId: PICTURE.attachment_id });
    const keyListeners = [...docSpy.mock.calls, ...winSpy.mock.calls].filter(([type]) =>
      /^key/.test(String(type))
    );
    expect(keyListeners).toEqual([]);
  });

  test('Escape leaves the lightbox open and fires no close callback', async () => {
    vi.mocked(entriesApi.get).mockResolvedValueOnce(entry('e-1', [PICTURE]));
    const onAttachmentClose = vi.fn();
    await showInReader(container, 'e-1', {
      attachmentId: PICTURE.attachment_id,
      onAttachmentClose,
    });
    const box = /** @type {HTMLElement} */ (lightbox());
    const esc = { key: 'Escape', bubbles: true };
    document.dispatchEvent(new KeyboardEvent('keydown', esc));
    box.dispatchEvent(new KeyboardEvent('keydown', esc));
    expect(lightbox()).toBe(box);
    expect(onAttachmentClose).not.toHaveBeenCalled();
  });
});

describe('Use in Workspace', () => {
  test('POSTs the artifact route and asks the host to activate the artifacts panel', async () => {
    const fetchSpy = vi.fn().mockResolvedValue({ ok: true, status: 200 });
    vi.stubGlobal('fetch', fetchSpy);
    const parent = { postMessage: vi.fn() };
    vi.spyOn(window, 'parent', 'get').mockReturnValue(/** @type {any} */ (parent));
    try {
      showImageLightbox(RENDITION_SRC, 'beam.png', null, 'att-0123456789ab');
      const button = /** @type {HTMLButtonElement} */ (
        lightbox()?.querySelector('[data-use-in-workspace]')
      );
      button.click();
      await vi.waitFor(() =>
        expect(lightbox()?.querySelector('[data-workspace-status]')?.textContent).toBe(
          'Copied to Workspace'
        )
      );
      expect(fetchSpy).toHaveBeenCalledWith('/api/attachments/att-0123456789ab/artifact', {
        method: 'POST',
      });
      expect(parent.postMessage).toHaveBeenCalledWith(
        { type: 'osprey:activate', panel: 'artifacts' },
        window.location.origin
      );
      // The click does not dismiss the lightbox.
      expect(lightbox()).not.toBeNull();
    } finally {
      vi.unstubAllGlobals();
    }
  });

  test('a failed copy re-enables the button and says so', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, status: 500 }));
    try {
      showImageLightbox(RENDITION_SRC, 'beam.png', null, 'att-1');
      const button = /** @type {HTMLButtonElement} */ (
        lightbox()?.querySelector('[data-use-in-workspace]')
      );
      button.click();
      await vi.waitFor(() =>
        expect(lightbox()?.querySelector('[data-workspace-status]')?.textContent).toBe(
          'Could not copy picture'
        )
      );
      expect(button.disabled).toBe(false);
    } finally {
      vi.unstubAllGlobals();
    }
  });
});
