// @ts-check
/**
 * ARIEL rail facets: value counts per filter dimension, the day histogram and
 * the Enhancements section, rendered into the rail's `.ariel-rail-facets`
 * mount below the mailboxes and saved searches.
 *
 * Counts come from `GET /api/facets` (C-FACETS) with the same scope the list
 * uses: the filter is `listingFilter(state)` (so `requestFilter`, C-STATE) and
 * the advanced params are the mode's declared knobs, exactly as the search
 * driver sends them. A burst of store writes settles into one fetch, a fetch
 * only goes out when the request differs from the last one, and a newer fetch
 * aborts the one in flight.
 *
 * The mount always exists; its `aria-busy` is "true" only while a fetch is in
 * flight. Under Drafts (no request) and in the icon strip the sections are
 * hidden and nothing is fetched. A failed fetch, or an `available: false`
 * payload, shows no counts and no Enhancements note, and no error.
 *
 * Clicking a value toggles it in the store's `filter` (`toggleValue`); an
 * author click under Mine switches to All, because Mine scopes the author
 * itself. Clicking a histogram day sets `time` to that day. Rows are built
 * with DOM APIs only: values are user text.
 */

import { facilityDayKey } from '/design-system/js/facility-time.js';
import { facetsApi } from './api.js';
import { getState, setState, subscribe } from './store.js';
import { TIME, mergeFilter, requestFilter, toggleValue } from './filter-state.js';
import { listingFilter } from './paging.js';
import { isAbort, wireParams } from './search.js';

/** @typedef {import('./store.js').State} State */
/** @typedef {{name: string, label: string, kind?: string}} Dimension */
/** @typedef {{value: string, count: number}} FacetValue */
/**
 * @typedef {{dimensions: Array<{name: string, label?: string, kind?: string,
 *   values: FacetValue[]}>, histogram: Array<{day: string, count: number}>,
 *   exact: boolean}} FacetsData
 */

/** Days the histogram covers, ending today in the facility zone. */
export const HISTOGRAM_DAYS = 365;

/** The enhancement dimension's kind (C-DIM). */
const ENHANCEMENT_KIND = 'enhancement';

/** Enhancement states the section lists. */
const ENHANCEMENT_STATES = ['failed', 'stale'];

/** Store keys whose change can change the request or what is shown. */
const WATCHED = ['query', 'mode', 'filters', 'filter', 'mailbox', 'mineAuthor', 'railCollapsed'];

/**
 * @type {{root: HTMLElement, sections: HTMLElement, capabilities: any,
 *   data: FacetsData|null, key: string|null, controller: AbortController|null,
 *   queued: boolean, unsubscribe: () => void} | null}
 */
let mounted = null;

/**
 * Render the facets into `root` (the rail's facets mount) and keep them in
 * step with the store.
 * @param {HTMLElement} root
 * @param {{capabilities?: any}} [options]
 */
export function initFacets(root, { capabilities = null } = {}) {
  if (mounted) {
    mounted.unsubscribe();
    mounted.controller?.abort();
  }
  root.replaceChildren();
  root.setAttribute('aria-busy', 'false');
  const sections = document.createElement('div');
  sections.className = 'ariel-facets-sections';
  root.append(sections);
  mounted = {
    root,
    sections,
    capabilities,
    data: null,
    key: null,
    controller: null,
    queued: false,
    unsubscribe: () => {},
  };
  mounted.unsubscribe = subscribe((_state, _prev, changed) => {
    if (WATCHED.some((k) => changed.has(k))) schedule();
  });
  schedule();
}

// ---------------------------------------------------------------------------
// Fetching

/** Settle a burst of store writes into one sync. */
function schedule() {
  if (!mounted || mounted.queued) return;
  mounted.queued = true;
  queueMicrotask(() => {
    if (!mounted) return;
    mounted.queued = false;
    sync();
  });
}

/** Bring the sections, and the fetch, in line with the store. */
function sync() {
  if (!mounted) return;
  const m = mounted;
  const state = getState();
  const hidden = state.railCollapsed || requestFilter(state) === null;
  m.sections.hidden = hidden;
  if (hidden) {
    // Nothing is fetched while hidden; the next showing fetches afresh.
    m.controller?.abort();
    m.controller = null;
    m.key = null;
    m.root.setAttribute('aria-busy', 'false');
    return;
  }
  const request = {
    query: (state.query ?? '').trim(),
    mode: state.mode ?? null,
    filters: listingFilter(state),
    advancedParams: wireParams(m.capabilities, state.mode ?? null, state.filters ?? {}),
  };
  const key = JSON.stringify(request);
  render();
  if (key !== m.key) {
    m.key = key;
    fetchFacets(request);
  }
}

/**
 * @param {{query: string, mode: string|null, filters: any, advancedParams: Record<string, any>}} request
 */
async function fetchFacets(request) {
  if (!mounted) return;
  const m = mounted;
  m.controller?.abort();
  const controller = new AbortController();
  m.controller = controller;
  m.root.setAttribute('aria-busy', 'true');
  /** @type {FacetsData|null} */
  let data = null;
  try {
    const payload = await facetsApi.get({ ...request, signal: controller.signal });
    if (payload && payload.available !== false) {
      data = {
        dimensions: Array.isArray(payload.dimensions) ? payload.dimensions : [],
        histogram: Array.isArray(payload.histogram) ? payload.histogram : [],
        exact: payload.exact !== false,
      };
    }
  } catch (error) {
    if (isAbort(error) || mounted !== m || m.controller !== controller) return;
    // A failed fetch shows no counts and no error.
  }
  if (mounted !== m || m.controller !== controller) return;
  m.controller = null;
  m.data = data;
  m.root.setAttribute('aria-busy', 'false');
  render();
}

// ---------------------------------------------------------------------------
// Rendering

/** Paint the sections from the last data and the current state. */
function render() {
  if (!mounted) return;
  const m = mounted;
  const data = m.data;
  if (!data) {
    m.sections.replaceChildren();
    return;
  }
  const state = getState();
  const selected = requestFilter(state) ?? {};
  const declared = /** @type {Dimension[]|null} */ (m.capabilities?.filter_dimensions ?? null);
  const byName = new Map(data.dimensions.map((d) => [d.name, d]));
  const dims =
    declared ??
    data.dimensions.map((d) => ({ name: d.name, label: d.label ?? d.name, kind: d.kind }));

  /** @type {HTMLElement[]} */
  const out = [];
  /** @type {Dimension|null} */
  let enhancement = null;
  for (const dim of dims) {
    if (dim.name === TIME) continue;
    if (dim.kind === ENHANCEMENT_KIND) {
      enhancement = dim;
      continue;
    }
    let values = byName.get(dim.name)?.values ?? [];
    if (dim.name === 'author' && state.mailbox === 'mine' && state.mineAuthor) {
      const mine = state.mineAuthor.toLowerCase();
      values = values.filter((v) => String(v.value).toLowerCase() === mine);
    }
    if (!values.length) continue;
    out.push(dimensionSection(dim, values, selected, data.exact, (v) => String(v)));
  }
  out.push(histogramSection(data.histogram, selected));
  if (enhancement) out.push(enhancementSection(enhancement, byName, selected, data.exact));
  m.sections.replaceChildren(...out);
}

/**
 * @param {string} name
 * @param {string} label
 * @returns {{section: HTMLElement, body: HTMLElement}}
 */
function section(name, label) {
  const el = document.createElement('section');
  el.className = 'ariel-facet-section';
  el.setAttribute('data-facet-section', name);
  el.setAttribute('aria-label', label);
  const caption = document.createElement('div');
  caption.className = 'osprey-section-label';
  caption.textContent = label;
  const body = document.createElement('div');
  body.className = 'ariel-facet-rows';
  el.append(caption, body);
  return { section: el, body };
}

/**
 * @param {Dimension} dim
 * @param {FacetValue[]} values
 * @param {Record<string, any>} selected
 * @param {boolean} exact
 * @param {(value: string) => string} labelOf
 * @returns {HTMLElement}
 */
function dimensionSection(dim, values, selected, exact, labelOf) {
  const { section: el, body } = section(dim.name, dim.label || dim.name);
  const chosen = new Set(
    (Array.isArray(selected[dim.name]) ? selected[dim.name] : []).map((/** @type {string} */ v) =>
      v.toLowerCase()
    )
  );
  for (const v of values) {
    const value = String(v.value);
    body.append(
      valueRow(
        dim.name,
        value,
        labelOf(value),
        Number(v.count) || 0,
        exact,
        chosen.has(value.toLowerCase())
      )
    );
  }
  return el;
}

/**
 * One facet value row: `.osprey-nav-row` with its count in `.osprey-nav-badge`.
 * @param {string} dim
 * @param {string} value
 * @param {string} label
 * @param {number} count
 * @param {boolean} exact
 * @param {boolean} pressed
 * @returns {HTMLButtonElement}
 */
function valueRow(dim, value, label, count, exact, pressed) {
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'osprey-nav-row';
  btn.setAttribute('data-facet-dim', dim);
  btn.setAttribute('data-facet-value', value);
  btn.setAttribute('aria-pressed', pressed ? 'true' : 'false');
  const countText = exact ? String(count) : `≈${count}`;
  btn.title = `${label} (${countText})`;
  const text = document.createElement('span');
  text.className = 'osprey-nav-label';
  text.textContent = label;
  const badge = document.createElement('span');
  badge.className = 'osprey-nav-badge';
  badge.textContent = countText;
  btn.append(text, badge);
  btn.addEventListener('click', () => toggle(dim, value));
  return btn;
}

/**
 * @param {string} dim
 * @param {string} value
 */
function toggle(dim, value) {
  const state = getState();
  const filter = toggleValue(state.filter, dim, value);
  if (dim === 'author' && state.mailbox === 'mine') {
    setState({ mailbox: 'all', filter, rerank: 'now' });
  } else {
    setState({ filter, rerank: 'now' });
  }
}

/**
 * The day keys the histogram covers, oldest first, ending today.
 * @returns {string[]}
 */
function histogramDays() {
  const today = facilityDayKey(Date.now());
  const [y, mo, d] = today.split('-').map(Number);
  /** @type {string[]} */
  const days = [];
  for (let i = HISTOGRAM_DAYS - 1; i >= 0; i--) {
    days.push(new Date(Date.UTC(y, mo - 1, d - i)).toISOString().slice(0, 10));
  }
  return days;
}

/**
 * @param {Array<{day: string, count: number}>} histogram
 * @param {Record<string, any>} selected
 * @returns {HTMLElement}
 */
function histogramSection(histogram, selected) {
  const { section: el, body } = section(TIME, 'Entries per day');
  body.className = 'ariel-facet-histogram';
  body.setAttribute('role', 'group');
  /** @type {Map<string, number>} */
  const counts = new Map();
  for (const h of histogram) counts.set(String(h.day).slice(0, 10), Number(h.count) || 0);
  const max = Math.max(1, ...counts.values());
  const time = selected[TIME];
  const start = typeof time?.start === 'string' ? time.start.slice(0, 10) : null;
  const end = typeof time?.end === 'string' ? time.end.slice(0, 10) : null;
  const windowSet = Boolean(start || end);
  for (const day of histogramDays()) {
    const count = counts.get(day) ?? 0;
    const inWindow = windowSet && (!start || day >= start) && (!end || day <= end);
    const bar = document.createElement('button');
    bar.type = 'button';
    bar.className = 'ariel-facet-day';
    bar.classList.toggle('ariel-facet-day-selected', inWindow);
    bar.tabIndex = -1;
    bar.setAttribute('data-day', day);
    bar.setAttribute('aria-pressed', inWindow ? 'true' : 'false');
    const text = `${day}: ${count}`;
    bar.setAttribute('aria-label', text);
    bar.title = text;
    bar.style.setProperty('--ariel-facet-day-fill', `${Math.round((count / max) * 100)}%`);
    bar.addEventListener('click', () => {
      setState({
        filter: mergeFilter(getState().filter, { [TIME]: { start: day, end: day } }),
        rerank: 'now',
      });
    });
    body.append(bar);
  }
  return el;
}

/**
 * The Enhancements section: failed and stale values only, an empty note when
 * there are none, and the CLI that retries them (ingestion stays read-only
 * from the panel, e-0001 R6).
 * @param {Dimension} dim
 * @param {Map<string, {values: FacetValue[]}>} byName
 * @param {Record<string, any>} selected
 * @param {boolean} exact
 * @returns {HTMLElement}
 */
function enhancementSection(dim, byName, selected, exact) {
  const modules = /** @type {Array<{name: string, label?: string}>} */ (
    mounted?.capabilities?.enhancement_modules ?? []
  );
  const labels = new Map(modules.map((mod) => [mod.name, mod.label || mod.name]));
  const values = (byName.get(dim.name)?.values ?? []).filter((v) => {
    const state = String(v.value).split(':').pop() ?? '';
    return ENHANCEMENT_STATES.includes(state) && Number(v.count) > 0;
  });
  /** @param {string} value */
  const labelOf = (value) => {
    const cut = value.lastIndexOf(':');
    const mod = value.slice(0, cut);
    return `${labels.get(mod) ?? mod}: ${value.slice(cut + 1)}`;
  };
  const el = dimensionSection(dim, values, selected, exact, labelOf);
  const caption = el.querySelector('.osprey-section-label');
  if (caption) caption.textContent = 'Enhancements';
  el.setAttribute('aria-label', 'Enhancements');
  if (!values.length) {
    const note = document.createElement('p');
    note.className = 'ariel-facets-note';
    note.textContent = 'No failed or stale enhancements.';
    el.append(note);
  }
  const hint = document.createElement('p');
  hint.className = 'ariel-facets-hint';
  const retry = document.createElement('code');
  retry.textContent = 'osprey ariel enhance --module <name> --retry-failed';
  const refresh = document.createElement('code');
  refresh.textContent = '--refresh-stale';
  hint.append('Retry failures with ', retry, '; caption stale pictures again with ', refresh, '.');
  el.append(hint);
  return el;
}
