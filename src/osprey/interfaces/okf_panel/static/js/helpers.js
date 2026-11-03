// @ts-check
/*
 * OKF Knowledge Panel — pure DOM-leaf helpers.
 *
 * No imports: every function here is self-contained and side-effect-free
 * apart from DOM node construction / reading `location.hash`. Keep it that
 * way so this module stays trivially testable in isolation.
 */

// History marker for the structure overview (B.3): a hash distinct from any
// concept id so popstate can tell the overview apart from a concept view.
export const STRUCTURE_MARKER = '__structure';

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
  const records = span ? span.records : undefined;
  if (!Array.isArray(records)) {
    return { text, title: text + ' · facility file not loaded', dangling: false };
  }
  if (records.length === 0) {
    return { text, title: text + ' · not in the facility file', dangling: true };
  }
  const title = records
    .map(function (record) {
      return [record.kind, record.unit, record.description].filter(Boolean).join(' · ');
    })
    .join('\n');
  return { text, title, dangling: false };
}
