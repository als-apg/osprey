// @ts-check
/**
 * ARIEL Reader: an entry body that declares Markdown
 *
 * {@link renderEntryMarkdown} is the one path from an entry's stored Markdown
 * to HTML. An entry's text comes from a logbook, so it is untrusted, and this
 * page has no Content-Security-Policy that would stop a script: DOMPurify is
 * the only barrier. The order is fixed:
 *
 *   1. `marked` parses the text.
 *   2. DOMPurify sanitises the whole string against a short list of elements
 *      and attributes ({@link sanitizeBody}).
 *   3. The sanitised string is parsed into a `<template>`, whose content
 *      loads nothing and runs nothing.
 *   4. There, with DOM calls only, each picture becomes one of the entry's own
 *      attachments or its alt text ({@link inlinePictures}), and each quoted
 *      facility id is wrapped ({@link markFacilityReferences}).
 *
 * Nothing is added to the HTML as a string after step 2, so no value can leave
 * the attribute or the text it was put in.
 *
 * `marked` and `DOMPurify` are vendored classic-script globals. When one is
 * missing or a step throws, the function returns null and the caller shows the
 * entry as plain text.
 *
 * @module entry-markdown
 */

import { facilityReferenceView } from '/design-system/js/facility-references.js';
import { safeHref } from './attachment-urls.js';

/**
 * @typedef {{target: string, records?: Array<*>|null}} Reference
 * @typedef {{attachment_id?: string|null, filename?: string|null, url?: string|null,
 *   display_url?: string|null, viewable?: boolean}} Attachment
 */

/**
 * The elements a body may hold: what Markdown produces, and the inline
 * elements people write by hand. Everything else is removed and its text
 * kept, so no form control, style sheet, frame, media element, SVG or MathML
 * reaches the page.
 */
const ALLOWED_TAGS = Object.freeze(
  (
    'p br hr h1 h2 h3 h4 h5 h6 strong em b i u s del ins sub sup mark kbd code pre blockquote ' +
    'ul ol li table thead tbody tr th td a img span div details summary'
  ).split(' ')
);

/**
 * The attributes a body may hold. No `class`, `id`, `name`, `style` or
 * `data-*`: the text must not be able to take this page's styles, shadow its
 * element ids, or carry the `data-` attributes its click handlers read.
 */
const ALLOWED_ATTR = Object.freeze([
  'href',
  'title',
  'src',
  'alt',
  'align',
  'start',
  'colspan',
  'rowspan',
]);

/** A link address a body may hold: absolute http or https. */
const HTTP_URL = /^https?:\/\//i;

/**
 * DOMPurify hook: a link opens in a new tab and tells the target nothing
 * about this page. An address that is not http(s) is removed and the link's
 * text stays.
 * @param {Element} node
 */
function restrictLink(node) {
  if (!node.hasAttribute?.('href')) return;
  if (!HTTP_URL.test(node.getAttribute('href') ?? '')) {
    node.removeAttribute('href');
    return;
  }
  node.setAttribute('target', '_blank');
  node.setAttribute('rel', 'noopener noreferrer');
}

/**
 * Sanitise marked's output. The link hook is on DOMPurify only for this call.
 * A DOMPurify without hooks cannot apply the link rule, so it is refused.
 * @param {any} DOMPurify
 * @param {string} html - marked's output
 * @returns {string}
 */
function sanitizeBody(DOMPurify, html) {
  if (typeof DOMPurify.addHook !== 'function' || typeof DOMPurify.removeHook !== 'function') {
    throw new Error('DOMPurify has no hooks');
  }
  DOMPurify.addHook('afterSanitizeAttributes', restrictLink);
  try {
    return String(
      DOMPurify.sanitize(html, {
        ALLOWED_TAGS: [...ALLOWED_TAGS],
        ALLOWED_ATTR: [...ALLOWED_ATTR],
        ALLOW_DATA_ATTR: false,
        ALLOW_ARIA_ATTR: false,
      })
    );
  } finally {
    DOMPurify.removeHook('afterSanitizeAttributes', restrictLink);
  }
}

/**
 * A `parse(markdown) -> html` function on a parser of this module's own, so
 * nothing another script does to the shared `marked` changes what an entry
 * renders as. Null when `marked` is not loaded. Its output is not sanitised:
 * only {@link renderEntryMarkdown} calls it.
 * @returns {((text: string) => string) | null}
 */
function entryMarkedParse() {
  const marked = /** @type {any} */ (globalThis).marked;
  if (!marked || typeof marked.Marked !== 'function') return null;
  const parser = new marked.Marked({ gfm: true, breaks: false });
  return (text) => String(parser.parse(text));
}

/**
 * The last path part of an address, percent-decoded.
 * @param {string} path
 * @returns {string}
 */
function lastPathPart(path) {
  const part = path.split(/[?#]/)[0].split('/').pop() ?? '';
  try {
    return decodeURIComponent(part);
  } catch {
    return part;
  }
}

/**
 * The file name a picture address in the text names, or null when the address
 * points somewhere else: one with a scheme (`https:`, `data:`) or a host
 * (`//host`) is never one of the entry's attachments.
 * @param {string} address
 * @returns {string|null}
 */
function pictureFileName(address) {
  if (/^[a-z][a-z0-9+.-]*:/i.test(address) || /^[/\\]{2}/.test(address)) return null;
  return lastPathPart(address) || null;
}

/**
 * The attachment of this entry a picture address names: the first one the
 * page can show (`viewable`, with a safe `display_url`) whose `filename`, or
 * the last path part of whose source `url`, is the address's file name.
 * @param {string} address
 * @param {Attachment[]} attachments
 * @returns {{attachment: Attachment, src: string}|null}
 */
function pictureFor(address, attachments) {
  const name = pictureFileName(address);
  if (!name) return null;
  for (const attachment of attachments) {
    const src = attachment?.viewable === true ? safeHref(attachment.display_url) : null;
    if (!src) continue;
    const sourceName = typeof attachment.url === 'string' ? lastPathPart(attachment.url) : '';
    if (attachment.filename === name || sourceName === name) return { attachment, src };
  }
  return null;
}

/**
 * Replace every picture under `root`. One that names an attachment of this
 * entry becomes that attachment, with the attributes the reader's click
 * delegation opens the lightbox from; every other one becomes its alt text.
 * @param {DocumentFragment} root - sanitised, inert
 * @param {Attachment[]} attachments
 */
function inlinePictures(root, attachments) {
  const doc = /** @type {Document} */ (root.ownerDocument);
  for (const img of Array.from(root.querySelectorAll('img'))) {
    const alt = img.getAttribute('alt') ?? '';
    const found = pictureFor(img.getAttribute('src') ?? '', attachments);
    if (!found) {
      img.replaceWith(doc.createTextNode(alt));
      continue;
    }
    const { attachment, src } = found;
    const name = attachment.filename || 'attachment';
    const picture = doc.createElement('img');
    picture.className = 'entry-inline-image';
    picture.setAttribute('src', src);
    picture.setAttribute('alt', alt || name);
    picture.setAttribute('data-lightbox-url', attachment.display_url ?? '');
    picture.setAttribute('data-lightbox-name', name);
    picture.setAttribute('data-attachment-id', attachment.attachment_id || '');
    img.replaceWith(picture);
  }
}

/**
 * A pattern for the quotes of `targets` in one run of text, by the rule the
 * server's `find_facility_references` uses: no identifier character (a letter,
 * a digit or `_`) directly before or after, and the longest target wins at a
 * position.
 * @param {string[]} targets
 * @returns {RegExp}
 */
function quotePattern(targets) {
  const longestFirst = [...targets].sort((a, b) => Array.from(b).length - Array.from(a).length);
  const alternatives = longestFirst.map((t) => t.replace(/[.*+?^${}()|[\]\\/]/g, '\\$&'));
  return new RegExp(`(?<![\\p{L}\\p{N}_])(?:${alternatives.join('|')})(?![\\p{L}\\p{N}_])`, 'gu');
}

/**
 * Wrap each quote of a referenced facility id under `root` in the span the
 * plain reader uses, with the same hover text.
 *
 * The server's offsets point into the stored text, not into rendered HTML, so
 * the ids are found again here, in the text nodes. Only text is searched: an
 * id inside a link address is an attribute and stays untouched, so the link
 * still works. Link text, table cells and code are searched like any other
 * text, because the server counts an id as a reference wherever it stands.
 * @param {DocumentFragment} root - sanitised, inert
 * @param {Reference[]} references
 */
function markFacilityReferences(root, references) {
  /** @type {Map<string, Reference>} */
  const byTarget = new Map();
  for (const reference of references) {
    const target = reference?.target;
    if (typeof target === 'string' && target && !byTarget.has(target)) {
      byTarget.set(target, reference);
    }
  }
  if (byTarget.size === 0) return;
  const pattern = quotePattern([...byTarget.keys()]);
  const doc = /** @type {Document} */ (root.ownerDocument);
  const walker = doc.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  /** @type {Text[]} */
  const texts = [];
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    texts.push(/** @type {Text} */ (node));
  }
  for (const node of texts) {
    const text = node.nodeValue ?? '';
    const pieces = doc.createDocumentFragment();
    let last = 0;
    for (const match of text.matchAll(pattern)) {
      const at = match.index ?? 0;
      const reference = /** @type {Reference} */ (byTarget.get(match[0]));
      const { title, dangling } = facilityReferenceView(reference.target, reference.records);
      const span = doc.createElement('span');
      span.className = dangling ? 'entry-facility-link--dangling' : 'entry-facility-link';
      span.setAttribute('title', title);
      span.textContent = match[0];
      pieces.append(text.slice(last, at), span);
      last = at + match[0].length;
    }
    if (last === 0) continue;
    pieces.append(text.slice(last));
    node.replaceWith(pieces);
  }
}

/**
 * Render an entry's Markdown body as sanitised HTML, its pictures taken from
 * the entry's own attachments and its facility ids marked.
 * @param {string} text - The body as stored (untrusted)
 * @param {{references?: Reference[]|null, attachments?: Attachment[]|null}} [entry] - The
 *   entry's `facility_references` and `attachments`
 * @returns {string|null} HTML safe to assign to `innerHTML`, or null when the
 *   body cannot be rendered and must be shown as plain text
 */
export function renderEntryMarkdown(text, { references, attachments } = {}) {
  const DOMPurify = /** @type {any} */ (globalThis).DOMPurify;
  const parse = entryMarkedParse();
  if (!parse || !DOMPurify || typeof DOMPurify.sanitize !== 'function') return null;
  try {
    const clean = sanitizeBody(DOMPurify, parse(text));
    const template = document.createElement('template');
    template.innerHTML = clean;
    inlinePictures(template.content, attachments || []);
    markFacilityReferences(template.content, references || []);
    return template.innerHTML;
  } catch {
    return null;
  }
}
