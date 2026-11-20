// @ts-check
// @vitest-environment jsdom
/**
 * The KNOWLEDGE panel shows an in-bundle image that fails to load as its alt
 * text and a short reason (image-failures.js):
 *   npx vitest run tests/interfaces/okf_panel/image-failures.test.mjs
 *
 * The images are rendered the way the reading pane renders them, through the
 * real `marked` and DOMPurify and sanitizeMarkdownHtml (helpers.js), so each
 * points at the panel's asset route. jsdom, not the suite's happy-dom: DOMPurify
 * mis-parses under happy-dom (see markdown-sanitize.test.mjs).
 */

import createDOMPurify from 'dompurify';
import { marked } from 'marked';
import { afterEach, describe, expect, test, vi } from 'vitest';

import { sanitizeMarkdownHtml } from '../../../src/osprey/interfaces/okf_panel/static/js/helpers.js';
import { watchImageFailures } from '../../../src/osprey/interfaces/okf_panel/static/js/image-failures.js';

async function flush() {
  for (let i = 0; i < 10; i++) await new Promise((r) => setTimeout(r, 0));
}

/**
 * @param {string} markdown
 * @param {string} conceptId
 */
function sanitized(markdown, conceptId) {
  const purify = createDOMPurify(window);
  return sanitizeMarkdownHtml(purify, String(marked.parse(markdown)), conceptId);
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('an image that fails to load', () => {
  /**
   * @param {string} alt
   * @returns {{container: HTMLElement, img: HTMLImageElement}}
   */
  function failingImage(alt) {
    const container = document.createElement('div');
    container.innerHTML = sanitized(`![${alt}](figures/gone.png)`, 'devices/bpm');
    const img = /** @type {HTMLImageElement} */ (container.querySelector('img'));
    watchImageFailures(container);
    return { container, img };
  }

  test.each([
    ['not_found', 404, 'file not found'],
    ['too_large', 413, 'larger than 10 MB'],
    ['disallowed_type', 415, 'image type not allowed'],
    ['not_an_image', 415, 'not an image'],
    ['outside_bundle', 404, 'outside the knowledge bundle'],
  ])('%s shows the alt text and its reason', async (code, status, reason) => {
    const fetchSpy = vi.fn(async () => ({
      ok: false,
      status,
      json: async () => ({ error: code }),
    }));
    vi.stubGlobal('fetch', fetchSpy);
    const { container, img } = failingImage('Gone figure');
    const src = img.getAttribute('src');

    img.dispatchEvent(new Event('error'));
    await flush();

    expect(fetchSpy).toHaveBeenCalledTimes(1);
    expect(fetchSpy).toHaveBeenCalledWith(src);
    expect(container.querySelector('img')).toBeNull();
    const box = container.querySelector('.okf-image-fallback');
    expect(box?.textContent).toContain('Gone figure');
    expect(box?.textContent).toContain(reason);
  });

  test.each([
    [
      'a served image the browser cannot decode',
      async () => ({ ok: true, status: 200, json: async () => ({}) }),
    ],
    [
      'an unknown code',
      async () => ({
        ok: false,
        status: 503,
        json: async () => ({ error: 'facility_knowledge.bundle_path not configured' }),
      }),
    ],
    [
      'a reply that is not JSON',
      async () => ({
        ok: false,
        status: 500,
        json: async () => {
          throw new Error('html');
        },
      }),
    ],
    [
      'no reply',
      async () => {
        throw new Error('offline');
      },
    ],
  ])('%s gives the generic reason', async (_label, reply) => {
    vi.stubGlobal('fetch', vi.fn(reply));
    const { container, img } = failingImage('Odd figure');

    img.dispatchEvent(new Event('error'));
    await flush();

    const box = container.querySelector('.okf-image-fallback');
    expect(box?.textContent).toContain('Odd figure');
    expect(box?.textContent).toContain('cannot be displayed');
  });

  test('the error is handled once', async () => {
    const fetchSpy = vi.fn(async () => ({
      ok: false,
      status: 404,
      json: async () => ({ error: 'not_found' }),
    }));
    vi.stubGlobal('fetch', fetchSpy);
    const { container, img } = failingImage('Once');
    watchImageFailures(container);

    img.dispatchEvent(new Event('error'));
    img.dispatchEvent(new Event('error'));
    await flush();

    expect(fetchSpy).toHaveBeenCalledTimes(1);
  });
});
