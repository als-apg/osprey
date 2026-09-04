// @ts-check
/**
 * OSPREY Channel Finder — Graph finder state model (pure).
 *
 * Holds everything the graph-mode Explore finder needs to describe "what the
 * user is currently asking for": the free-text query, the facet selections, the
 * page, and the set of picked device addresses. DOM-free and side-effect free so
 * it can be unit-tested under Vitest (see
 * tests/interfaces/channel_finder/graph-finder-state.test.mjs); the mount reads
 * this state to render and calls back into it on every interaction.
 *
 * Mutators change the state in place. Every query or facet change resets the
 * page to 1, because the result set it addressed no longer exists.
 */

/**
 * @typedef {'section'|'system'|'signal'|'dir'} MultiFacet
 *   Facets the server accepts as repeated query params.
 */

/**
 * @typedef {MultiFacet|'cls'} Facet
 *   Every facet the finder rail exposes. `cls` is single-select.
 */

/**
 * @typedef {object} FinderState
 * @property {string} q - Free-text query; the empty string means "no query".
 * @property {Set<string>} section
 * @property {Set<string>} system
 * @property {Set<string>} signal
 * @property {Set<string>} dir - Direction values: 'R', 'W', 'RW', 'none'.
 * @property {string|null} cls - Class uri, single-select.
 * @property {number} page - 1-based page number.
 * @property {Set<string>} selected - Picked device addresses, insertion-ordered.
 */

/**
 * @typedef {object} Chip
 * @property {string} key - Opaque identity that round-trips through removeChip.
 * @property {string} label - Human-readable text for the chip.
 */

/** Multi-select facets, in the order they are serialised and charted. */
const MULTI_FACETS = /** @type {MultiFacet[]} */ (['section', 'system', 'signal', 'dir']);

/** Display names for chips, keyed by facet. */
const FACET_LABELS = {
  section: 'Section',
  system: 'System',
  signal: 'Signal',
  dir: 'Direction',
  cls: 'Device class',
};

/**
 * Create an empty finder state.
 * @returns {FinderState}
 */
export function createFinderState() {
  return {
    q: '',
    section: new Set(),
    system: new Set(),
    signal: new Set(),
    dir: new Set(),
    cls: null,
    page: 1,
    selected: new Set(),
  };
}

/**
 * Toggle one facet value.
 *
 * `cls` is single-select: a new value replaces the old one, re-clicking the
 * active one clears it. The others are multi-select. Always resets the page.
 *
 * @param {FinderState} state
 * @param {Facet} facet
 * @param {string} value
 * @returns {FinderState} The same state, for chaining.
 */
export function toggleFacet(state, facet, value) {
  if (facet === 'cls') {
    state.cls = state.cls === value ? null : value;
  } else if (MULTI_FACETS.includes(facet)) {
    const set = state[facet];
    if (set.has(value)) set.delete(value);
    else set.add(value);
  } else {
    throw new TypeError(`unknown facet: ${facet}`);
  }
  state.page = 1;
  return state;
}

/**
 * Set the free-text query. Resets the page.
 * @param {FinderState} state
 * @param {string} q
 * @returns {FinderState}
 */
export function setQuery(state, q) {
  state.q = q;
  state.page = 1;
  return state;
}

/**
 * Serialise the state as the query string for `GET /api/graph/search`.
 *
 * Multi facets are appended once per value (`section=BR&section=SR`) and their
 * values are sorted so the same selection always produces the same string —
 * which keeps request caching and tests stable. `q` and `page` are always
 * present; empty facets and an unset class are omitted. The selection is not
 * part of the request.
 *
 * @param {FinderState} state
 * @returns {URLSearchParams}
 */
export function toSearchParams(state) {
  const params = new URLSearchParams();
  params.set('q', state.q);
  for (const facet of MULTI_FACETS) {
    for (const value of [...state[facet]].sort()) {
      params.append(facet, value);
    }
  }
  if (state.cls) params.append('cls', state.cls);
  params.set('page', String(state.page));
  return params;
}

/**
 * The active filters as removable chips: the query first, then each facet value
 * in facet order with values sorted. A blank query contributes no chip.
 *
 * Keys are `q` for the query and `<facet>:<value>` otherwise. Values may
 * themselves contain colons (class uris do), so removeChip splits on the FIRST
 * colon only.
 *
 * @param {FinderState} state
 * @returns {Chip[]}
 */
export function activeChips(state) {
  /** @type {Chip[]} */
  const chips = [];
  if (state.q.trim() !== '') {
    chips.push({ key: 'q', label: `Search: ${state.q}` });
  }
  for (const facet of MULTI_FACETS) {
    for (const value of [...state[facet]].sort()) {
      chips.push({ key: `${facet}:${value}`, label: `${FACET_LABELS[facet]}: ${value}` });
    }
  }
  if (state.cls) {
    chips.push({ key: `cls:${state.cls}`, label: `${FACET_LABELS.cls}: ${state.cls}` });
  }
  return chips;
}

/**
 * Remove whatever the chip key names. Resets the page when something was
 * actually removed; an unrecognised key is ignored so a stale chip click cannot
 * throw out of a render callback.
 *
 * @param {FinderState} state
 * @param {string} key - A key from activeChips.
 * @returns {boolean} Whether anything changed.
 */
export function removeChip(state, key) {
  if (key === 'q') {
    if (state.q === '') return false;
    state.q = '';
    state.page = 1;
    return true;
  }

  const sep = key.indexOf(':');
  if (sep === -1) return false;
  const facet = key.slice(0, sep);
  const value = key.slice(sep + 1);

  if (facet === 'cls') {
    if (state.cls !== value) return false;
    state.cls = null;
  } else if (MULTI_FACETS.includes(/** @type {MultiFacet} */ (facet))) {
    const set = state[/** @type {MultiFacet} */ (facet)];
    if (!set.delete(value)) return false;
  } else {
    return false;
  }
  state.page = 1;
  return true;
}

/**
 * Clamp the page into `[1, pages]` after a response reported fewer pages than
 * the one requested (the filters narrowed under a deep page).
 *
 * @param {FinderState} state
 * @param {number} pages - Page count from the response; treated as at least 1.
 * @returns {boolean} Whether the page changed, i.e. whether a refetch is due.
 */
export function clampPage(state, pages) {
  const last = Number.isFinite(pages) ? Math.max(1, Math.floor(pages)) : 1;
  const next = Math.min(Math.max(1, Math.floor(state.page) || 1), last);
  if (next === state.page) return false;
  state.page = next;
  return true;
}

/**
 * Toggle one device address in the selection. Does not touch the page: the
 * selection deliberately survives paging within a mount.
 *
 * @param {FinderState} state
 * @param {string} pv - Device address.
 * @returns {boolean} Whether the address is selected after the toggle.
 */
export function toggleSelection(state, pv) {
  if (state.selected.has(pv)) {
    state.selected.delete(pv);
    return false;
  }
  state.selected.add(pv);
  return true;
}

/**
 * Select or deselect every address on the current page (the header checkbox).
 * Addresses selected on other pages are untouched.
 *
 * @param {FinderState} state
 * @param {string[]} pvs - Addresses shown on the page.
 * @param {boolean} on - True to select them all, false to deselect them all.
 * @returns {FinderState}
 */
export function togglePageSelection(state, pvs, on) {
  for (const pv of pvs) {
    if (on) state.selected.add(pv);
    else state.selected.delete(pv);
  }
  return state;
}

/**
 * Drop the whole selection, leaving query, facets and page alone.
 * @param {FinderState} state
 * @returns {FinderState}
 */
export function clearSelection(state) {
  state.selected.clear();
  return state;
}

/**
 * Selected addresses space-joined on one line, with no trailing newline — the
 * form the assistant prompt expects.
 *
 * @param {FinderState} state
 * @returns {string}
 */
export function sendText(state) {
  return [...state.selected].join(' ');
}

/**
 * Selected addresses newline-joined, with no trailing newline — the form pasted
 * into an editor or a script.
 *
 * @param {FinderState} state
 * @returns {string}
 */
export function copyText(state) {
  return [...state.selected].join('\n');
}
