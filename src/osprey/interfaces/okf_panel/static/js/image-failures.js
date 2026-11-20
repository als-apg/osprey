// @ts-check
/*
 * OKF Knowledge Panel — an in-bundle image that fails to load in the reading
 * pane is replaced by its alt text and a short reason.
 */

import { imageFailureReason, imageFallback } from './helpers.js';

/**
 * Show an in-bundle image that fails to load as its alt text and a short
 * reason. The image's URL is requested once more to read the asset route's
 * error code; a reply without a known code (an image the browser cannot
 * decode is served normally) gives a generic reason.
 *
 * @param {ParentNode} container - a freshly rendered reading pane
 * @returns {void}
 */
export function watchImageFailures(container) {
  const images = /** @type {NodeListOf<HTMLImageElement>} */ (container.querySelectorAll('img'));
  images.forEach(function (img) {
    if (img.dataset.okfWatched === '1') return;
    img.dataset.okfWatched = '1';
    img.addEventListener('error', () => void explainImageFailure(img), { once: true });
  });
}

/**
 * @param {HTMLImageElement} img
 * @returns {Promise<void>}
 */
async function explainImageFailure(img) {
  const src = img.getAttribute('src') || '';
  let code = '';
  if (src.includes('/api/asset?')) {
    try {
      const resp = await fetch(src);
      if (!resp.ok) {
        const body = await resp.json();
        if (body && typeof body.error === 'string') code = body.error;
      }
    } catch {
      // No reply, or not JSON: the generic reason.
    }
  }
  const alt = img.getAttribute('alt') || '';
  img.replaceWith(imageFallback(img.ownerDocument, alt, imageFailureReason(code)));
}
