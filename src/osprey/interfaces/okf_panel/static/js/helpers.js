// @ts-check
/*
 * OKF Knowledge Panel — pure DOM-leaf helpers.
 *
 * One import, the shared facility-reference view; every function here is
 * side-effect-free apart from DOM node construction / reading `location.hash`.
 * Keep it that way so this module stays trivially testable in isolation.
 */

import { facilityReferenceView } from '/design-system/js/facility-references.js';

// History marker for the structure overview (B.3): a hash distinct from any
// concept id so popstate can tell the overview apart from a concept view.
export const STRUCTURE_MARKER = '__structure';

// History marker for the Graph view: `#__graph` is its overview and
// `#__graph/<node id>` the neighbourhood of one node.
export const GRAPH_MARKER = '__graph';

/**
 * The concept an in-bundle link names (OKF §6): a target starting with `/`
 * resolves from the bundle root, any other from the directory of the page the
 * link sits in.
 * @param {string} href the link target as written
 * @param {string | null | undefined} currentId the concept the link sits in;
 *   absent for a page at the bundle root, such as the overview
 * @returns {string | null} the concept id, or null for a link that names no
 *   concept: an external one, an anchor, a directory, an asset, a reserved
 *   file, or a target outside the bundle root
 */
export function linkedConceptId(href, currentId) {
  if (!href || /^[A-Za-z][A-Za-z0-9+.-]*:/.test(href)) return null;
  let path = href.split('#')[0].split('?')[0];
  if (!path.endsWith('.md')) return null;
  try {
    path = decodeURIComponent(path);
  } catch {
    // A target that does not decode is read as written.
  }
  const parts = path.startsWith('/') || !currentId ? [] : currentId.split('/').slice(0, -1);
  for (const segment of path.split('/')) {
    if (!segment || segment === '.') continue;
    if (segment === '..') {
      if (!parts.length) return null;
      parts.pop();
    } else {
      parts.push(segment);
    }
  }
  const name = parts[parts.length - 1];
  if (!name || name === 'index.md' || name === 'log.md') return null;
  return parts.join('/').replace(/\.md$/, '');
}

/**
 * A concept's tags as a list (OKF §11.2): a scalar where a list is expected is
 * one tag, and a null or empty value is no tag.
 * @param {unknown} value the `tags` frontmatter value
 * @returns {unknown[]}
 */
export function tagList(value) {
  if (Array.isArray(value)) return value.filter((tag) => tag !== null && tag !== '');
  return value === null || value === undefined || value === '' ? [] : [value];
}

/**
 * The Graph view a hash names.
 * @param {string} raw the hash, without its `#`
 * @returns {{node: string | null} | null} the node whose neighbourhood it
 *   names (null for the overview), or null for a hash that is not the Graph
 *   view's
 */
export function graphTarget(raw) {
  if (raw === GRAPH_MARKER) return { node: null };
  if (!raw.startsWith(GRAPH_MARKER + '/')) return null;
  const rest = raw.slice(GRAPH_MARKER.length + 1);
  if (!rest) return { node: null };
  try {
    return { node: decodeURIComponent(rest) };
  } catch {
    return { node: rest };
  }
}

/**
 * The hash of the Graph view.
 * @param {string | null} node the focus node's id, or null for the overview
 * @returns {string} with its `#`
 */
export function graphHash(node) {
  return '#' + GRAPH_MARKER + (node === null ? '' : '/' + encodeURIComponent(node));
}

/**
 * Create a DOM element with a flat string-valued attribute map and a list
 * of child nodes.
 *
 * @param {string} tag
 * @param {Record<string, string>|null} [attrs] String-valued attribute map.
 *   The key `"class"` sets `className`; `"text"` sets `textContent`; every
 *   other key is set via `setAttribute`.
 * @param {(Node|null|undefined)[]} [children] Falsy entries are skipped.
 * @returns {HTMLElement}
 */
export function el(tag, attrs, children) {
  const node = document.createElement(tag);
  if (attrs) {
    for (const k in attrs) {
      if (k === 'class') node.className = attrs[k];
      else if (k === 'text') node.textContent = attrs[k];
      else node.setAttribute(k, attrs[k]);
    }
  }
  if (children) {
    for (const child of children) {
      if (child) node.appendChild(child);
    }
  }
  return node;
}

// A fallback (filesystem-derived) concept is one whose title is just the
// last path segment of its id — those have no real human title/description.
/**
 * @param {string|null|undefined} id
 * @param {string|null|undefined} title
 * @returns {boolean}
 */
export function isFallback(id, title) {
  if (!id || !title) return false;
  const last = String(id).split('/').pop();
  return title === last;
}

/** @typedef {{concept: string|null, structure: boolean, raw: string}} PanelParams */

// -- panel parameters / deep-link (osprey addition) --------------------------
//
// Generalizable panel-parameter shape. Today the only param is the deep-link
// target carried in the panel's OWN URL hash (e.g. "#control-system/channel-
// finding"). The panel also follows a change of this fragment after boot,
// which is how the hub opens it on a concept.
/**
 * @returns {PanelParams}
 */
export function readPanelParams() {
  const hash = (location.hash || '').replace(/^#/, '');
  /** @type {PanelParams} */
  const params = { concept: null, structure: false, raw: hash };
  if (!hash) return params;
  if (hash === STRUCTURE_MARKER) {
    params.structure = true;
    return params;
  }
  // The Graph view's hash names no concept.
  if (graphTarget(hash)) return params;
  try {
    params.concept = decodeURIComponent(hash);
  } catch {
    params.concept = hash;
  }
  return params;
}

/**
 * Decide what a `hashchange` should show. A target `historyState` already
 * holds was reached by a back/forward step that `popstate` handles; an id the
 * loaded tree does not list is an in-page anchor.
 * @param {PanelParams} params - readPanelParams() after the change
 * @param {any} historyState - history.state after the change
 * @param {Set<string> | null} knownIds - concept ids the tree lists; null before it loads
 * @returns {string | null} the concept id or STRUCTURE_MARKER to show, or null to leave alone
 */
export function hashChangeTarget(params, historyState, knownIds) {
  if (!params.raw) return null;
  const target = params.structure ? STRUCTURE_MARKER : params.concept;
  if (target === null) return null;
  if (historyState && historyState.id === target) return null;
  if (target !== STRUCTURE_MARKER && knownIds && !knownIds.has(target)) return null;
  return target;
}

// -- facility links ----------------------------------------------------------
//
// A knowledge page names a facility channel or device with a markdown link
// `[label](facility:ADDRESS)`. The concept response lists each link as a span
// (`facility_references`) carrying the records its target resolves to: a list,
// empty for a target the facility file does not hold, or null when no facility
// file was read.

export const FACILITY_SCHEME = 'facility:';

/** @typedef {{kind: string, id: string, unit?: string, description?: string}} FacilityRecordView */
/**
 * @typedef {{start: number, end: number, target: string, kinds: string[],
 *   label: string | null, records: FacilityRecordView[] | null}} FacilityReferenceView
 */

/**
 * What the reading pane shows for one facility link.
 *
 * marked percent-encodes a destination, so `[x](<facility:A B>)` reaches the
 * DOM as `facility:A%20B`; the target is decoded before the lookup.
 *
 * @param {string} href - the anchor's href, starting with `facility:`
 * @param {FacilityReferenceView[] | null | undefined} references - the concept
 *   response's `facility_references`
 * @returns {{text: string, title: string, dangling: boolean}} the target, the
 *   hover text, and whether the facility file holds no record for it
 */
export function facilityLinkView(href, references) {
  let text = href.startsWith(FACILITY_SCHEME) ? href.slice(FACILITY_SCHEME.length) : href;
  try {
    text = decodeURI(text);
  } catch {
    // A malformed escape keeps the target as written.
  }
  const span = Array.isArray(references)
    ? references.find(function (entry) {
        return entry.target === text;
      })
    : undefined;
  return { text, ...facilityReferenceView(text, span ? span.records : undefined) };
}

// ---- Images ---- //
//
// A concept's images come only from the panel's asset route. The reader never
// resolves a path: it hands the target as written, with the concept it sits
// in, to `/api/asset`, and the server decides. A target with a scheme or a
// leading `//` is never requested; it shows its alt text instead.

/** Why an in-bundle image is not shown, by the asset route's error code. */
export const IMAGE_FAILURE_REASONS = Object.freeze({
  outside_bundle: 'outside the knowledge bundle',
  symlink_escape: 'outside the knowledge bundle',
  dot_path: 'hidden path',
  not_found: 'file not found',
  not_an_image: 'not an image',
  disallowed_type: 'image type not allowed',
  too_large: 'larger than 10 MB',
});

/** The reason for a failure the route's reply does not name. */
export const IMAGE_FAILURE_GENERIC = 'cannot be displayed';

/** The reason for an image the reader does not request. */
export const EXTERNAL_IMAGE_REASON = 'external image, not loaded';

/**
 * The asset route's URL for an image target. The path is its own leading
 * literal so the web terminal's proxy prefixes it (see app.js).
 * @param {string} target - the image target as written in the concept
 * @param {string} conceptId - the concept it sits in; empty for the overview
 * @returns {string}
 */
export function assetUrl(target, conceptId) {
  return (
    '/api/asset?target=' + encodeURIComponent(target) + '&from=' + encodeURIComponent(conceptId)
  );
}

/**
 * Whether the reader shows *target* as alt text without a request: it has a
 * scheme (`https:`, `data:`, ...) or a leading `//`.
 * @param {string} target
 * @returns {boolean}
 */
export function isExternalImageTarget(target) {
  return /^[a-z][a-z0-9+.-]*:/i.test(target) || target.startsWith('//');
}

/**
 * @param {unknown} code - the `error` of the asset route's reply
 * @returns {string} the short reason the reader shows
 */
export function imageFailureReason(code) {
  return typeof code === 'string' && Object.hasOwn(IMAGE_FAILURE_REASONS, code)
    ? IMAGE_FAILURE_REASONS[/** @type {keyof typeof IMAGE_FAILURE_REASONS} */ (code)]
    : IMAGE_FAILURE_GENERIC;
}

/**
 * What stands in for an image that is not shown: its alt text, the reason,
 * and a link to a remote source. Text nodes only, never parsed markup.
 * @param {Document} doc - the document the node is placed in
 * @param {string} alt
 * @param {string} reason - empty for none
 * @param {string} [href] - a remote source to link to
 * @returns {HTMLElement}
 */
export function imageFallback(doc, alt, reason, href) {
  const box = doc.createElement('span');
  box.className = 'okf-image-fallback';
  box.appendChild(doc.createTextNode(alt || 'Image'));
  if (reason) {
    const why = doc.createElement('span');
    why.className = 'okf-image-fallback__reason';
    why.textContent = ' (' + reason + ')';
    box.appendChild(why);
  }
  if (href) {
    box.appendChild(doc.createTextNode(' '));
    const link = doc.createElement('a');
    link.setAttribute('href', href);
    link.textContent = href;
    box.appendChild(link);
  }
  return box;
}

/**
 * Elements a concept may not hold: styling, form controls, media, frames and
 * objects. The HTML profile also leaves out inline SVG and MathML.
 */
const MARKDOWN_FORBID_TAGS = Object.freeze([
  'style',
  'link',
  'form',
  'input',
  'button',
  'select',
  'option',
  'textarea',
  'video',
  'audio',
  'source',
  'track',
  'picture',
  'iframe',
  'frame',
  'frameset',
  'object',
  'embed',
]);

/** `style` paints over the page or loads from it; the rest list URLs. */
const MARKDOWN_FORBID_ATTR = Object.freeze(['style', 'background', 'srcset', 'poster']);

/**
 * Attributes that hold a URL. The hook removes each of them from every
 * element, except `href` on a link and `src` on an image, which it decides.
 */
const URL_ATTRS = Object.freeze([
  'action',
  'archive',
  'background',
  'cite',
  'classid',
  'codebase',
  'data',
  'dynsrc',
  'formaction',
  'href',
  'icon',
  'longdesc',
  'lowsrc',
  'manifest',
  'ping',
  'poster',
  'profile',
  'src',
  'srcset',
  'usemap',
  'xlink:href',
]);

/**
 * DOMPurify hook: leave URLs only on links and on images pointed at the asset
 * route; an image that cannot be is replaced by its alt text.
 * @param {Node} node
 * @param {string} conceptId
 */
function confineUrls(node, conceptId) {
  if (node.nodeType !== 1) return;
  const elem = /** @type {Element} */ (node);
  const tag = elem.localName;
  for (const name of URL_ATTRS) {
    if ((name === 'href' && tag === 'a') || (name === 'src' && tag === 'img')) continue;
    if (elem.hasAttribute(name)) elem.removeAttribute(name);
  }
  if (tag !== 'img') return;
  const src = (elem.getAttribute('src') || '').trim();
  if (src && !isExternalImageTarget(src)) {
    elem.setAttribute('src', assetUrl(src, conceptId));
    return;
  }
  const alt = elem.getAttribute('alt') || '';
  const remote = /^https?:\/\//i.test(src) ? src : '';
  const reason = src ? EXTERNAL_IMAGE_REASON : IMAGE_FAILURE_GENERIC;
  elem.replaceWith(
    imageFallback(/** @type {Document} */ (elem.ownerDocument), alt, reason, remote)
  );
}

/**
 * Sanitise a concept's rendered markdown. In the string returned, the only
 * URL-bearing attributes are `a[href]` and `img[src]` pointing at the asset
 * route. The hook is on DOMPurify only for this call. A DOMPurify without
 * hooks cannot give that guarantee, so it throws, and the caller falls back
 * to inert text.
 * @param {any} purify - DOMPurify
 * @param {string} html - marked's output
 * @param {string} conceptId - the concept the markdown is; empty for the overview
 * @returns {string}
 */
export function sanitizeMarkdownHtml(purify, html, conceptId) {
  if (typeof purify.addHook !== 'function' || typeof purify.removeHook !== 'function') {
    throw new Error('DOMPurify has no hooks');
  }
  /** @param {Node} node */
  const hook = (node) => confineUrls(node, conceptId);
  purify.addHook('afterSanitizeAttributes', hook);
  try {
    return purify.sanitize(html, {
      USE_PROFILES: { html: true },
      FORBID_TAGS: [...MARKDOWN_FORBID_TAGS],
      FORBID_ATTR: [...MARKDOWN_FORBID_ATTR],
    });
  } finally {
    purify.removeHook('afterSanitizeAttributes', hook);
  }
}
