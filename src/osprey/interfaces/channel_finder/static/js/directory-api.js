// @ts-check
/**
 * OSPREY Channel Finder — PV Directory Client
 *
 * The routes under the panel's directory API with their payload types, and the
 * cross-panel URLs and gates All PVs and the PV page share. Another panel's URL
 * is built by panelUrl from a rest with no leading slash: the web terminal's proxy
 * rewrites quoted root-absolute API paths in served JS to this panel's own mount,
 * and refuses a navigate outside the target panel's mount.
 */

import { panelApiPrefix } from '/design-system/js/dom.js';
import { isEmbedded } from '/design-system/js/frame-params.js';
import { fetchJSON } from './api.js';
import { state } from './state.js';

/** @typedef {'found'|'inactive'|'not_listed'|'no_match'|'unavailable'} Outcome */

/**
 * `service` is "none" when the current target has no source.
 * @typedef {{ target: 'va'|'live'|'standin', service: string, label: string, rehearsal: boolean }} Source
 */

/**
 * @typedef {{
 *   name: string,
 *   origin: 'curated'|'directory'|'unknown',
 *   status: 'Active'|'Inactive'|null,
 *   description: string|null,
 *   record_type: string|null,
 *   ioc: string|null,
 *   host: string|null,
 *   last_announced: string|null,
 * }} DirectoryRow
 */

/** @typedef {{ row: DirectoryRow, properties: Record<string, string>, tags: string[] }} DirectoryRecord */

/** @typedef {{ value: string, count: number }} FacetCount */

/**
 * @typedef {{
 *   outcome: Outcome,
 *   searched: string,
 *   request: string,
 *   source: Source,
 *   total: number|null,
 *   rows: DirectoryRow[],
 *   truncated: boolean,
 *   partial: { part: string, reason: string }[],
 *   facets: { origin: FacetCount[], ioc: FacetCount[], record_type: FacetCount[] }|null,
 *   facets_reason: string|null,
 *   reason?: string,
 * }} DirectoryPage
 */

/** @typedef {{ outcome: Outcome, total: number|null, source: Source, reason?: string }} DirectoryCount */

/**
 * @typedef {{
 *   time: string,
 *   severity: string,
 *   status: string,
 *   message: string,
 *   user: string|null,
 *   kind: 'state'|'config',
 * }} AlarmHistoryRow
 */

/**
 * @typedef {{
 *   at: string,
 *   pv: string,
 *   field: string|null,
 *   old: string|null,
 *   new: string|null,
 *   user: string|null,
 *   client: string|null,
 * }} PutRow
 */

/**
 * @typedef {{
 *   severity: string,
 *   status: string,
 *   latched: boolean,
 *   acknowledged: boolean,
 *   message: string,
 *   time: string,
 *   stale: boolean,
 * }} AlarmStateRecord
 */

/**
 * @typedef {{
 *   in_tree: 'yes'|'no'|'unknown',
 *   path: string|null,
 *   guidance: string|null,
 *   displays: string|null,
 *   enabled: boolean|null,
 *   latching: boolean|null,
 *   label: 'live configuration'|'last logged configuration',
 * }} AlarmConfigRecord
 */

/**
 * One section of the PV facts; an unconfigured section is absent. `caveats`
 * are the deployment's notes on the section's source. `origin` and
 * `detail_properties` are the directory section's.
 * @typedef {{
 *   section: 'directory'|'alarm_state'|'alarm_history'|'alarm_config'|'put_history',
 *   outcome: Outcome,
 *   source: Source,
 *   rows?: (AlarmHistoryRow|PutRow)[],
 *   record?: DirectoryRecord|AlarmStateRecord|AlarmConfigRecord|null,
 *   truncated?: boolean,
 *   retention_known?: boolean,
 *   caveats: string[],
 *   origin?: 'curated'|'directory'|'unknown',
 *   detail_properties?: string[],
 *   reason?: string,
 *   label?: string,
 * }} PvFactsSection
 */

/**
 * The PV facts. `outcome` says whether the PV is known at all: `found` when
 * the directory lists it, `not_listed` when it does not but another section
 * found something, `unknown` when nothing did, `unavailable` (with `reason`)
 * when the directory gave no answer.
 * @typedef {{
 *   name: string,
 *   outcome: 'found'|'not_listed'|'unknown'|'unavailable',
 *   reason?: string,
 *   sections: PvFactsSection[],
 * }} PvFacts
 */

/**
 * @typedef {{
 *   ioc: string,
 *   outcome: Outcome,
 *   source: Source,
 *   host: string|null,
 *   ports: number[]|null,
 *   pv_count: number|null,
 *   inactive_count: number|null,
 *   curated_count: number|null,
 *   curated_reason?: string,
 *   engineer: string|null,
 *   engineer_source: 'directory property'|'ENGINEER PV'|null,
 *   health_pvs: string[],
 *   health_pvs_reason?: string,
 *   links: Record<string, string>,
 *   reason?: string,
 * }} IocFacts
 */

/**
 * Built by the PV page from ARIEL's answer; no route returns it.
 * @typedef {{
 *   section: 'logbook',
 *   outcome: Outcome,
 *   source: Source|null,
 *   rows: { title: string, time: string, author: string, excerpt: string, url: string|null }[],
 *   reason?: string,
 * }} LogbookSection
 */

/** @typedef {{ available: boolean, source: Source|null, reason?: string }} LogbookAvailability */

/**
 * @typedef {{
 *   name: string,
 *   outcome: 'found'|'no_answer'|'unavailable',
 *   value: unknown,
 *   timestamp: string|null,
 *   severity: number|null,
 *   status: string|null,
 *   reason?: string,
 * }} ValueRow
 */

/**
 * The filter parameters of the search and export routes; origin, ioc and
 * record_type repeat, and page counts from 0.
 * @typedef {{
 *   q?: string,
 *   name?: string,
 *   origin?: string[],
 *   ioc?: string[],
 *   record_type?: string[],
 *   page?: number,
 *   page_size?: number,
 * }} DirectoryFilters
 */

/** Past this many PVs selected, Plot is disabled. */
const PLOT_MAX_PVS = 12;

/**
 * The query string for `filters`, with its `?`; empty when nothing is set.
 * @param {DirectoryFilters} filters
 * @returns {string}
 */
function queryString(filters) {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(filters)) {
    for (const item of Array.isArray(value) ? value : [value]) {
      if (item !== undefined && item !== null && item !== '') params.append(key, String(item));
    }
  }
  const query = params.toString();
  return query ? `?${query}` : '';
}

/**
 * @param {DirectoryFilters} [filters]
 * @returns {Promise<DirectoryPage>}
 */
export function searchDirectory(filters = {}) {
  return fetchJSON(`/api/directory/search${queryString(filters)}`);
}

/**
 * The url of the CSV export, which the CSV action fetches: it does not answer
 * JSON.
 * @param {DirectoryFilters} [filters]
 * @returns {string}
 */
export function directoryExportUrl(filters = {}) {
  return `/api/directory/export.csv${queryString(filters)}`;
}

/**
 * Hits for `q`; without it, the size of the current target's whole directory.
 * @param {string} [q]
 * @returns {Promise<DirectoryCount>}
 */
export function countDirectory(q) {
  return fetchJSON(`/api/directory/count${queryString({ q })}`);
}

/**
 * @param {string} name
 * @returns {Promise<PvFacts>}
 */
export function fetchPvFacts(name) {
  return fetchJSON(`/api/directory/pv/${encodeURIComponent(name)}`);
}

/**
 * @param {string} name
 * @returns {Promise<IocFacts>}
 */
export function fetchIocFacts(name) {
  return fetchJSON(`/api/directory/ioc/${encodeURIComponent(name)}`);
}

/** @returns {Promise<LogbookAvailability>} */
export function fetchLogbookAvailability() {
  return fetchJSON('/api/directory/logbook/availability');
}

/**
 * Current values of at most 50 names; the route answers 422 past that. No
 * names are read as none, without a request.
 * @param {string[]} names
 * @returns {Promise<ValueRow[]>}
 */
export function readValues(names) {
  if (names.length === 0) return Promise.resolve([]);
  const params = new URLSearchParams();
  for (const name of names) params.append('name', name);
  return fetchJSON(`/api/directory/values?${params}`);
}

/**
 * Another panel's URL under the mount this page was reached through, or null
 * standalone, where there is no host. `rest` has no leading slash; a hash rest
 * joins the same way.
 * @param {string} panelId
 * @param {string} rest
 * @returns {string|null}
 */
export function panelUrl(panelId, rest) {
  const prefix = panelApiPrefix();
  if (!prefix) return null;
  const outer = prefix.replace(/\/panel\/[^/]+$/, '');
  return `${outer}/panel/${panelId}/${rest}`;
}

/**
 * The CANOPY Data Browser url for `names`, in selection order so a repeat
 * Plot of an unchanged selection matches the tab already open.
 * @param {string[]} names
 * @returns {string|null}
 */
export function plotUrl(names) {
  const pvs = names.map((name) => `pv=${encodeURIComponent(name)}`).join('&');
  return panelUrl('canopy', `embed/@trace?${pvs}&range=-1h`);
}

/**
 * The Plot button for `selected` PVs.
 * @param {number} selected
 * @returns {{ disabled: boolean, label: string, title: string }}
 */
export function plotButton(selected) {
  if (selected <= PLOT_MAX_PVS) return { disabled: false, label: 'Plot', title: '' };
  return {
    disabled: true,
    label: `Plot (max ${PLOT_MAX_PVS})`,
    title: `Data Browser takes at most ${PLOT_MAX_PVS} PVs`,
  };
}

/**
 * Whether a hand-off to the host can be posted: a pop-out carries the embedded
 * param too, but its parent is itself.
 * @returns {boolean}
 */
export function handoffAllowed() {
  return isEmbedded() && window.parent !== window;
}

/**
 * A check of whether this page may offer Plot: the descriptor says CANOPY, the
 * hand-off gate holds and Trace, optional inside a payload, answers 200. A view
 * makes one per mount; it asks Trace once.
 * @returns {() => Promise<boolean>}
 */
export function createPlotCheck() {
  /** @type {Promise<boolean>|null} */
  let traceAnswer = null;
  return async () => {
    const traceUrl = panelUrl('canopy', 'trace/');
    if (!state.directory?.canopy || !handoffAllowed() || !traceUrl) return false;
    traceAnswer ??= fetch(traceUrl, { method: 'HEAD' }).then(
      (resp) => resp.status === 200,
      () => false
    );
    return traceAnswer;
  };
}
