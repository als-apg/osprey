// @ts-check
/**
 * OSPREY Channel Finder — Channel Directory Client
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
import { LABELS } from './labels.js';
import { state } from './state.js';

/** @typedef {'found'|'inactive'|'not_listed'|'no_match'|'unavailable'} Outcome */

/**
 * `service` is "none" when the current target has no source.
 * @typedef {{ target: 'va'|'live'|'standin', service: string, label: string, rehearsal: boolean }} Source
 */

/**
 * The alarm feed's state of a row's PV: `severity` OK included.
 * @typedef {{
 *   severity: string|null,
 *   status: string|null,
 *   latched: boolean,
 *   acknowledged: boolean,
 *   time: string|null,
 * }} RowAlarm
 */

/**
 * `alarm` is present when the target has an alarm feed and it answered: the
 * feed's state, null when it holds none for the PV. `archived` is present when
 * the target has an archive: whether it records the PV now, null when the
 * archive cannot say (the page's `archived_reason` says why).
 * @typedef {{
 *   name: string,
 *   origin: 'curated'|'directory'|'unknown',
 *   alarm?: RowAlarm|null,
 *   archived?: boolean|null,
 *   status: 'Active'|'Inactive'|null,
 *   description: string|null,
 *   record_type: string|null,
 *   server: string|null,
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
 *   facets: {
 *     origin: FacetCount[],
 *     server: FacetCount[],
 *     record_type: FacetCount[],
 *     alarm?: FacetCount[],
 *     archived?: FacetCount[],
 *   }|null,
 *   facets_reason: string|null,
 *   archived_reason?: string,
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
 *   channel: string,
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
 * `guidance` and `displays` come as the alarm server keeps them (a list of
 * `{title, details}`, a string, or null); alarm-text.js reads them as text.
 * @typedef {{
 *   in_tree: 'yes'|'no'|'unknown',
 *   path: string|null,
 *   guidance: unknown,
 *   displays: unknown,
 *   enabled: boolean|null,
 *   latching: boolean|null,
 *   label: 'live configuration'|'last logged configuration',
 * }} AlarmConfigRecord
 */

/**
 * What the target's archive says of one PV: `archived` is null when the
 * archiver cannot say (`description` then says why); the samples are ISO times.
 * @typedef {{
 *   archived: boolean|null,
 *   first_sample: string|null,
 *   last_sample: string|null,
 *   sampling_period_s: number|null,
 *   description: string|null,
 * }} ArchiveRecord
 */

/**
 * One section of the PV facts; an unconfigured section is absent. `caveats`
 * are the deployment's notes on the section's source. `origin` and
 * `detail_properties` are the directory section's.
 * @typedef {{
 *   section: 'directory'|'archive'|'alarm_state'|'alarm_history'|'alarm_config'|'put_history',
 *   outcome: Outcome,
 *   source: Source,
 *   rows?: (AlarmHistoryRow|PutRow)[],
 *   record?: DirectoryRecord|ArchiveRecord|AlarmStateRecord|AlarmConfigRecord|null,
 *   truncated?: boolean,
 *   retention_known?: boolean,
 *   caveats: string[],
 *   origin?: 'curated'|'directory'|'unknown',
 *   detail_properties?: string[],
 *   reason?: string,
 *   label?: string,
 * }} ChannelFactsSection
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
 *   sections: ChannelFactsSection[],
 * }} ChannelFacts
 */

/**
 * One PV of an IOC in alarm now.
 * @typedef {{
 *   channel: string,
 *   severity: string|null,
 *   status: string|null,
 *   latched: boolean|null,
 *   acknowledged: boolean|null,
 *   time: string|null,
 * }} ServerAlarmRow
 */

/**
 * The IOC's PVs in alarm now, present when an alarm feed is configured:
 * `total` counts them all, `rows` the first page, `counts` each severity over
 * the whole IOC; `unavailable` with `reason` when the feed did not answer.
 * @typedef {{
 *   outcome: 'found'|'no_match'|'unavailable',
 *   total?: number|null,
 *   rows?: ServerAlarmRow[],
 *   counts?: Record<string, number>,
 *   reason?: string,
 * }} ServerAlarms
 */

/**
 * @typedef {{
 *   server: string,
 *   outcome: Outcome,
 *   source: Source,
 *   host: string|null,
 *   ports: number[]|null,
 *   channel_count: number|null,
 *   inactive_count: number|null,
 *   curated_count: number|null,
 *   curated_reason?: string,
 *   engineer: string|null,
 *   engineer_source: 'directory property'|'ENGINEER PV'|null,
 *   health_channels: HealthChannel[],
 *   health_channels_reason?: string,
 *   alarms?: ServerAlarms,
 *   links: Record<string, string>,
 *   reason?: string,
 *   fleet?: FleetEntry,
 * }} ServerFacts
 */

// ---- The server fleet (GET /api/directory/fleet) ----
// Every time here is epoch seconds, never an ISO string.

/**
 * What the fleet service answered. `stale` keeps the last snapshot, marked;
 * every other non-`ok` outcome carries none. `not_in_snapshot` is an entry's
 * outcome only.
 * @typedef {'ok'|'no_snapshot'|'not_refreshed'|'cannot_list'|'unavailable'|'stale'|'not_in_snapshot'} FleetOutcome
 */

/** @typedef {'alive'|'stalled'|'silent'|'unreached'|'unknown'} FleetState */
/** @typedef {'degraded'|'expected_running_not_alive'|'expected_stopped_alive'|'alive_unexpected'} FleetFinding */

/**
 * One IOC of a snapshot, in the served order.
 * @typedef {{
 *   name: string,
 *   host: string|null,
 *   class_name: string|null,
 *   group: string|null,
 *   state: FleetState,
 *   since: number|null,
 *   reason: string|null,
 *   findings: FleetFinding[],
 *   directory_status: string|null,
 *   listed: boolean,
 *   sources: string[],
 *   liveness: { channel: string|null, kind: 'counter'|'value'|null, value: unknown, read_at: number|null, ambiguous: boolean }|null,
 *   health: { channel: string, value: unknown, status: 'ok'|'out_of_range'|'unread' }[],
 * }} FleetRow
 */

/**
 * A snapshot of one target's fleet.
 * @typedef {{
 *   outcome: string,
 *   target: string,
 *   generated_at: number,
 *   cycle?: number,
 *   period_s?: number,
 *   cycle_duration_s?: number,
 *   read_path: { status: 'ok'|'down'|'unverified', channel: string|null },
 *   sources: { name: string, outcome: string, reason: string|null }[],
 *   counts: { by_state: Record<FleetState, number>, by_finding: Record<FleetFinding, number> },
 *   group_by: 'host'|'class_name'|'group'|null,
 *   servers: FleetRow[],
 * }} FleetSnapshot
 */

/**
 * The fleet route's answer for the session's target.
 * @typedef {{
 *   outcome: FleetOutcome,
 *   target: string|null,
 *   reason: string|null,
 *   generated_at: number|null,
 *   snapshot: FleetSnapshot|null,
 * }} FleetAnswer
 */

/**
 * One IOC's fleet entry, as the IOC facts carry it while the view is on.
 * @typedef {{
 *   outcome: FleetOutcome,
 *   target: string|null,
 *   reason: string|null,
 *   generated_at: number|null,
 *   row: FleetRow|null,
 * }} FleetEntry
 */

/**
 * One IOC health PV: its name and its role.
 * @typedef {{ name: string, role: 'heartbeat'|'uptime'|'engineer' }} HealthChannel
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
 *   units?: string|null,
 *   precision?: number|null,
 *   enum_label?: string|null,
 *   reason?: string,
 * }} ValueRow
 */

/**
 * One channel in alarm now, as the alarm route serves it: `path` is its place
 * in the alarm tree when the feed knows one, `status` the record's alarm now
 * and `time` when the alarm started.
 * @typedef {{
 *   channel: string,
 *   path: string|null,
 *   severity: string,
 *   status: string,
 *   latched: boolean,
 *   acknowledged: boolean,
 *   message: string,
 *   time: string|null,
 *   stale: boolean,
 * }} AlarmRow
 */

/**
 * The active alarms of the current target. `counts` (per severity) and `total`
 * cover every channel the feed holds a state for; `rows` only those not OK,
 * worst first, then newest. `outcome` is `unavailable` with `reason` when the
 * feed gave no answer, and `source.service` is "none" when the target has no
 * alarm-state source. `feed` is `replay` when the states are replayed from a
 * file; its `last_contact` is then the moment the file was loaded.
 * @typedef {{
 *   outcome: 'found'|'unavailable',
 *   source: Source,
 *   caveats: string[],
 *   counts?: Record<string, number>,
 *   total?: number,
 *   rows?: AlarmRow[],
 *   truncated?: boolean,
 *   stale?: boolean,
 *   last_contact?: string|null,
 *   feed?: 'live'|'replay',
 *   reason?: string,
 * }} ActiveAlarms
 */

/**
 * What the current target's archive holds. `archived.count` is how many of the
 * directory's `of` PVs the store records now, null with `reason` when it cannot
 * be counted; `by_server` the same per IOC (empty with `by_server_reason`). The
 * samples are ISO times, null when unknown. `freshness_bound_s` is how old the
 * newest sample of the deployment's own store may be, null for any other
 * archive. `configuration` is the archive's as rendered; `targets` names, per
 * configured control target, whether it has an archive.
 * @typedef {{
 *   outcome: 'found'|'unavailable',
 *   source: Source,
 *   archived?: { count: number|null, of: number|null, reason?: string },
 *   by_server?: { server: string, channels: number, archived: number }[],
 *   by_server_reason?: string,
 *   newest_sample?: string|null,
 *   oldest_sample?: string|null,
 *   freshness_bound_s?: number|null,
 *   configuration: Record<string, Record<string, string|number>>,
 *   targets: { target: string, archive: boolean }[],
 *   reason?: string,
 * }} ArchiveSummary
 */

/**
 * The filter parameters of the search and export routes; origin, ioc,
 * record_type, alarm and archived repeat (alarm: a severity, or `none` for OK
 * or no state; archived: `yes`, `no` or `unknown`), and page counts from 0.
 * @typedef {{
 *   q?: string,
 *   name?: string,
 *   origin?: string[],
 *   ioc?: string[],
 *   record_type?: string[],
 *   alarm?: string[],
 *   archived?: string[],
 *   page?: number,
 *   page_size?: number,
 * }} DirectoryFilters
 */

/** Past this many PVs selected, Plot is disabled. */
const PLOT_MAX_CHANNELS = 12;

/**
 * The query string for `filters`, with its `?`; empty when nothing is set.
 * @param {Record<string, string|number|string[]|undefined>} filters
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
 * The target an answer says it came from: its source's, a channel page's
 * first section's, or the fleet answer's own.
 * @param {any} answer
 * @returns {string|null}
 */
function answerTarget(answer) {
  if (!answer || typeof answer !== 'object') return null;
  const target =
    answer.source?.target ?? answer.sections?.[0]?.source?.target ?? answer.target ?? null;
  return typeof target === 'string' ? target : null;
}

/**
 * *read*'s answer, after the panel moved to the words of the target it came
 * from, so the words follow the target an answer shows.
 * @template T
 * @param {Promise<T>|T} read
 * @returns {Promise<T>}
 */
function followed(read) {
  return Promise.resolve(read).then((answer) => {
    state.followTarget(answerTarget(answer));
    return answer;
  });
}

/**
 * @param {DirectoryFilters} [filters]
 * @returns {Promise<DirectoryPage>}
 */
export function searchDirectory(filters = {}) {
  return followed(fetchJSON(`/api/directory/search${queryString(filters)}`));
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
  return followed(fetchJSON(`/api/directory/count${queryString({ q })}`));
}

/**
 * @param {string} name
 * @param {AbortSignal} [signal]
 * @returns {Promise<ChannelFacts>}
 */
export function fetchChannelFacts(name, signal) {
  return followed(fetchJSON(`/api/directory/channel/${encodeURIComponent(name)}`, { signal }));
}

/**
 * @param {string} name
 * @returns {Promise<ServerFacts>}
 */
export function fetchServerFacts(name) {
  return followed(fetchJSON(`/api/directory/server/${encodeURIComponent(name)}`));
}

/** @returns {Promise<LogbookAvailability>} */
export function fetchLogbookAvailability() {
  return fetchJSON('/api/directory/logbook/availability');
}

/**
 * The active alarms; `severity` repeats, and `limit: 0` answers the counts
 * alone. `signal` aborts the read.
 * @param {{ severity?: string[], limit?: number }} [filters]
 * @param {AbortSignal} [signal]
 * @returns {Promise<ActiveAlarms>}
 */
export function fetchActiveAlarms({ severity, limit } = {}, signal) {
  return followed(
    fetchJSON(`/api/directory/alarms${queryString({ severity, limit })}`, { signal })
  );
}

/**
 * What the current target's archive holds. `signal` aborts the read.
 * @param {AbortSignal} [signal]
 * @returns {Promise<ArchiveSummary>}
 */
export function fetchArchiveSummary(signal) {
  return followed(fetchJSON('/api/directory/archive', { signal }));
}

/**
 * The server fleet of the session's target, from the fleet service alone: the
 * route reads no PV and queries no directory. `signal` aborts the read.
 * @param {AbortSignal} [signal]
 * @returns {Promise<FleetAnswer>}
 */
export function fetchFleet(signal) {
  return followed(fetchJSON('/api/directory/fleet', { signal }));
}

/**
 * Current values of at most 50 names; the route answers 422 past that. No
 * names are read as none, without a request.
 * @param {string[]} names
 * @param {AbortSignal} [signal]
 * @returns {Promise<ValueRow[]>}
 */
export function readValues(names, signal) {
  if (names.length === 0) return Promise.resolve([]);
  const params = new URLSearchParams();
  for (const name of names) params.append('name', name);
  return fetchJSON(`/api/directory/values?${params}`, { signal });
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
  if (selected <= PLOT_MAX_CHANNELS) return { disabled: false, label: 'Plot', title: '' };
  return {
    disabled: true,
    label: `Plot (max ${PLOT_MAX_CHANNELS})`,
    title: `Data Browser takes at most ${PLOT_MAX_CHANNELS} ${LABELS.channels}`,
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
