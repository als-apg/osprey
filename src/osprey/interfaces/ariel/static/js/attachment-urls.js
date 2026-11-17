// @ts-check
/**
 * ARIEL attachment URLs
 *
 * The one place an attachment's `display_url` becomes a URL the page may put
 * in an `href` or `src`. The reader's cards and lightbox (`reader.js`) and
 * the pictures of a rendered entry body (`entry-markdown.js`) both go through
 * it.
 */

import { apiUrl } from './api.js';

/**
 * One of this API's attachment routes, relative to the API base: the original
 * (`/attachments/<id>`) or its rendition (`/attachments/<id>/rendition`). The
 * id is one plain path segment, never `.` or `..`.
 */
const ATTACHMENT_ROUTE = /^\/attachments\/(?!\.\.?(?:\/|$))[A-Za-z0-9._~-]+(?:\/rendition)?$/;

/**
 * Turn an attachment's `display_url` into a URL safe to place in an href or
 * src, else null. An absolute http(s) URL is kept unchanged; an attachment
 * route relative to the API base is joined to the page's API base, so it
 * resolves under whatever prefix the page is served from. Every other value
 * (javascript:, data:, vbscript:, protocol-relative //host, any other path)
 * is refused.
 * @param {unknown} u - Candidate URL
 * @returns {string|null} The URL to use, or null when it is not safe
 */
export function safeHref(u) {
  if (typeof u !== 'string') return null;
  if (/^https?:\/\//i.test(u)) return u;
  if (ATTACHMENT_ROUTE.test(u)) return apiUrl(u);
  return null;
}

/**
 * The URL of the stored original behind a rendition `display_url`
 * (`/attachments/<id>/rendition` -> `/attachments/<id>`), made safe as by
 * `safeHref`, else null.
 * @param {unknown} displayUrl - The attachment's `display_url`
 * @returns {string|null} The original's URL, or null when there is none
 */
export function originalHref(displayUrl) {
  const suffix = '/rendition';
  if (typeof displayUrl !== 'string' || !ATTACHMENT_ROUTE.test(displayUrl)) return null;
  if (!displayUrl.endsWith(suffix)) return null;
  return safeHref(displayUrl.slice(0, -suffix.length));
}
