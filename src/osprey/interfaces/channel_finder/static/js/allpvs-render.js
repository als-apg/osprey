// @ts-check
/**
 * OSPREY Channel Finder — All PVs markup
 *
 * Every string the All PVs view (allpvs.js) draws, as pure functions of the
 * answer and the operator's choices. Values from the directory, the alarm feed
 * and the control system are escaped here.
 */

import { formatRoute } from './route.js';
import { severityMark, severityOf } from './severity.js';
import { esc } from './utils.js';
import { formatValue } from './value-format.js';

/** @typedef {import('./directory-api.js').DirectoryPage} DirectoryPage */
/** @typedef {import('./directory-api.js').DirectoryRow} DirectoryRow */
/** @typedef {import('./directory-api.js').RowAlarm} RowAlarm */
/** @typedef {import('./directory-api.js').ValueRow} ValueRow */
/** @typedef {import('./route.js').FacetKey} FacetKey */

/** Rows per page: the most the values route reads at once. */
export const PAGE_SIZE = 50;

/** The filters in sidebar and chip order. */
export const SIDEBAR_ORDER = /** @type {const} */ (['alarm', 'origin', 'ioc', 'record_type']);

/**
 * The sidebar title of each route filter.
 * @type {Record<FacetKey, string>}
 */
const FACET_TITLES = { alarm: 'Alarm', origin: 'Origin', ioc: 'IOC', record_type: 'Record type' };

/** The alarm filter's value for OK or no state. */
const NO_ALARM = 'none';

/** @type {Record<string, string>} */
const MARKS = { curated: '◆', directory: '◇' };

/** @type {Record<string, string>} */
const ORIGIN_LABELS = { curated: '◆ Curated', directory: '◇ Control system only' };

/** @param {number} n */
const fmt = (n) => n.toLocaleString('en-US');

/**
 * A filter value as the sidebar and the chips name it.
 * @param {FacetKey} key
 * @param {string} value
 * @returns {string}
 */
function valueLabel(key, value) {
  if (key === 'origin') return ORIGIN_LABELS[value] ?? value;
  if (key === 'alarm' && value === NO_ALARM) return 'No alarm';
  return value;
}

/**
 * The view's skeleton: the filter sidebar and its handle, then the head row,
 * the request, the rows, the pager and the selection bar.
 * @returns {string}
 */
export function skeletonHtml() {
  return `
    <div class="allpvs">
      <aside class="osprey-panel-sidebar allpvs-facets" data-open="false" aria-label="Filters"></aside>
      <div class="osprey-splitter" role="separator" aria-orientation="vertical"
           aria-label="Resize the filters" tabindex="0" title="Drag to resize · double-click to hide"></div>
      <div class="allpvs-main">
        <div class="allpvs-head">
          <span class="allpvs-searched"></span>
          <span class="allpvs-chips"></span>
          <span class="allpvs-partial"></span>
          <span class="allpvs-grow"></span>
          <span class="allpvs-csv-status"></span>
          <button type="button" class="osprey-icon-btn" data-action="refresh"
                  aria-label="Refresh values" title="Refresh values" disabled></button>
          <button type="button" class="osprey-icon-btn" data-action="csv"
                  aria-label="Export CSV" title="Export CSV" disabled></button>
          <button type="button" class="allpvs-link" data-action="request" hidden>show request</button>
        </div>
        <div class="allpvs-request" hidden>
          <code></code>
          <button type="button" class="osprey-icon-btn" data-action="copy-request"
                  aria-label="Copy request" title="Copy request"></button>
        </div>
        <div class="allpvs-body"></div>
        <div class="allpvs-pager"></div>
        <div class="allpvs-selection"></div>
      </div>
    </div>`;
}

/**
 * The toolbar's result count, or nothing when the total is not known.
 * @param {number|null} total
 * @returns {string}
 */
export function countHtml(total) {
  if (typeof total !== 'number') return '';
  return `<strong>${fmt(total)}</strong> ${total === 1 ? 'PV' : 'PVs'}`;
}

/**
 * The words searched, as the head row states them.
 * @param {string} words
 * @returns {string}
 */
export function wordsText(words) {
  return words ? `matches ${words}` : 'every PV';
}

/**
 * The filters in force as removable chips, then `clear all` when anything is
 * searched or filtered.
 * @param {Record<FacetKey, Set<string>>} filters
 * @param {boolean} words - whether words are searched
 * @returns {string}
 */
export function chipsHtml(filters, words) {
  const chips = SIDEBAR_ORDER.flatMap((key) =>
    [...filters[key]].map((value) => {
      const text = `${FACET_TITLES[key]}: ${valueLabel(key, value)}`;
      return (
        `<span class="allpvs-chip">${esc(text)}` +
        `<button type="button" data-facet="${key}" data-value="${esc(value)}"` +
        ` aria-label="Remove ${esc(text)}" title="Remove filter">×</button></span>`
      );
    })
  );
  const clear =
    chips.length || words
      ? '<button type="button" class="allpvs-link" data-action="clear-all">clear all</button>'
      : '';
  return chips.join('') + clear;
}

/**
 * The filters: one section per facet with its counts, Alarm first, a value in
 * force kept even when the answer has no count for it; or the reason there
 * are none.
 * @param {DirectoryPage|null} answer
 * @param {Record<FacetKey, Set<string>>} filters
 * @returns {string}
 */
export function facetsHtml(answer, filters) {
  if (!answer || answer.outcome === 'unavailable') return '';
  const facets = answer.facets;
  if (!facets) {
    return answer.facets_reason
      ? `<p class="allpvs-facets-reason">${esc(answer.facets_reason)}</p>`
      : '';
  }
  return SIDEBAR_ORDER.map((key) => {
    const counts = new Map((facets[key] ?? []).map((entry) => [entry.value, entry.count]));
    for (const value of filters[key]) if (!counts.has(value)) counts.set(value, 0);
    if (!counts.size) return '';
    const items = [...counts]
      .map(([value, count]) => {
        const label =
          key === 'alarm'
            ? severityMark(value === NO_ALARM ? 'NONE' : severityOf(value), valueLabel(key, value))
            : esc(valueLabel(key, value));
        return (
          `<button type="button" class="osprey-list-row allpvs-facet" data-facet="${key}"` +
          ` data-value="${esc(value)}" aria-pressed="${filters[key].has(value)}">` +
          `<span class="osprey-list-title">${label}</span>` +
          `<span class="allpvs-facet-n">${fmt(count)}</span></button>`
        );
      })
      .join('');
    return `<div class="allpvs-facet-group"><div class="osprey-section-label">${esc(FACET_TITLES[key])}</div>${items}</div>`;
  }).join('');
}

/**
 * The current value cell of one row: "…" while unread, "no answer" for a PV
 * that did not answer, the value with its unit otherwise, the full value and
 * the read time on hover.
 * @param {ValueRow|undefined} row
 * @param {boolean} read - whether the page's values have been read
 * @param {string} reason - why the whole read failed, if it did
 * @returns {string}
 */
function valueCell(row, read, reason) {
  if (!read) return '<td class="allpvs-value dim">…</td>';
  if (!row) return `<td class="allpvs-value dim" title="${esc(reason || 'not read')}">—</td>`;
  if (row.outcome === 'no_answer') {
    return `<td class="allpvs-value no-answer" title="${esc(row.reason ?? '')}">no answer</td>`;
  }
  if (row.outcome !== 'found') {
    return `<td class="allpvs-value dim" title="${esc(row.reason ?? '')}">—</td>`;
  }
  const { text, unit, title } = formatValue(row);
  const hover = [title || text, row.timestamp].filter(Boolean).join(' · ');
  const units = unit ? ` <span class="allpvs-unit">${esc(unit)}</span>` : '';
  return `<td class="allpvs-value" title="${esc(hover)}">${esc(text)}${units}</td>`;
}

/**
 * The Alarm cell of one row: the feed's severity mark, `·` for OK or no state,
 * `—` with the reason when the feed did not answer.
 * @param {RowAlarm|null|undefined} alarm
 * @param {string} reason - why the feed did not answer, if it did not
 * @returns {string}
 */
function alarmCell(alarm, reason) {
  if (alarm === undefined) {
    return `<td class="allpvs-alarm dim" title="${esc(reason || 'no answer')}">—</td>`;
  }
  const severity = severityOf(alarm?.severity);
  if (!alarm || severity === 'OK' || severity === 'NONE') {
    return `<td class="allpvs-alarm dim">·</td>`;
  }
  const flags = [
    alarm.status,
    alarm.latched ? 'latched' : '',
    alarm.acknowledged ? 'acknowledged' : 'not acknowledged',
    alarm.time ? `since ${alarm.time}` : '',
  ].filter(Boolean);
  return `<td class="allpvs-alarm" title="${esc(flags.join(' · '))}">${severityMark(severity)}</td>`;
}

/**
 * The Alarm column: null when it is off, else why the feed gave no answer
 * (empty when it answered).
 * @typedef {{ reason: string }|null} AlarmColumn
 */

/**
 * The rows of one page.
 * @param {DirectoryPage} answer
 * @param {Set<string>} selected
 * @param {Map<string, ValueRow>|null} values - null while unread
 * @param {string} valuesReason
 * @param {AlarmColumn} [alarmColumn]
 * @returns {string}
 */
export function tableHtml(answer, selected, values, valuesReason, alarmColumn = null) {
  const rows = answer.rows;
  const all = rows.every((row) => selected.has(row.name));
  const body = rows
    .map((row) => {
      const on = selected.has(row.name);
      const inactive = row.status === 'Inactive';
      const classes = [on ? 'sel' : '', inactive ? 'inactive' : ''].filter(Boolean).join(' ');
      const markClass = row.origin === 'curated' ? 'mark' : 'mark cs';
      const ioc = row.ioc
        ? `<a href="${esc(formatRoute({ view: 'ioc', params: { name: row.ioc } }))}">${esc(row.ioc)}</a>`
        : '';
      const chip = inactive ? ' <span class="allpvs-inactive">Inactive</span>' : '';
      return (
        `<tr class="${classes}" data-name="${esc(row.name)}">` +
        `<td><input type="checkbox" ${on ? 'checked' : ''} aria-label="Select ${esc(row.name)}"></td>` +
        `<td><span class="${markClass}">${MARKS[row.origin] ?? ''}</span></td>` +
        `<td class="name"><a href="${esc(formatRoute({ view: 'pv', params: { name: row.name } }))}">${esc(row.name)}</a>${chip}</td>` +
        `<td class="desc${row.description ? '' : ' dim'}">${esc(row.description || 'no description')}</td>` +
        valueCell(values?.get(row.name), values !== null, valuesReason) +
        (alarmColumn ? alarmCell(row.alarm, alarmColumn.reason) : '') +
        `<td class="dim">${esc(row.record_type ?? '')}</td>` +
        `<td class="dim">${ioc}</td></tr>`
      );
    })
    .join('');
  return `
    <table class="allpvs-table">
      <thead><tr>
        <th class="col-select"><input type="checkbox" data-action="select-page"
            aria-label="Select all on this page" ${all ? 'checked' : ''}></th>
        <th class="col-mark"></th><th>Name</th><th class="col-desc">Description</th>
        <th class="col-value">Value</th>${alarmColumn ? '<th>Alarm</th>' : ''}
        <th>Record</th><th>IOC</th>
      </tr></thead>
      <tbody>${body}</tbody>
    </table>`;
}

/**
 * The pager under the table: which rows show of how many, and the page.
 * Nothing unless the page holds rows.
 * @param {DirectoryPage|null} answer
 * @param {number} page
 * @returns {string}
 */
export function pagerHtml(answer, page) {
  if (!answer || answer.outcome === 'unavailable' || !answer.rows.length) return '';
  const total = typeof answer.total === 'number' ? answer.total : null;
  const pages = total === null ? null : Math.max(Math.ceil(total / PAGE_SIZE), 1);
  const more = pages === null ? answer.truncated : page + 1 < pages;
  const first = page * PAGE_SIZE + 1;
  const last = page * PAGE_SIZE + answer.rows.length;
  const of = total === null ? '' : ` of ${fmt(total)}`;
  return (
    `<span>rows ${fmt(first)}–${fmt(last)}${of}</span><span class="allpvs-grow"></span>` +
    `<button type="button" class="osprey-icon-btn" data-action="prev" aria-label="Previous page"${page > 0 ? '' : ' disabled'}>‹</button>` +
    `<span>page ${fmt(page + 1)}${pages === null ? '' : ` of ${fmt(pages)}`}</span>` +
    `<button type="button" class="osprey-icon-btn" data-action="next" aria-label="Next page"${more ? '' : ' disabled'}>›</button>`
  );
}

/**
 * @typedef {{
 *   selected: number,
 *   plot: { disabled: boolean, label: string, title: string }|null,
 *   send: boolean,
 * }} SelectionModel
 */

/**
 * The floating selection bar, only while something is selected: the count,
 * Plot (only with a Plot button model), Send (only when a host takes it),
 * Copy names and Clear.
 * @param {SelectionModel} model
 * @returns {string}
 */
export function selectionBarHtml({ selected, plot, send }) {
  if (!selected) return '';
  const plotBtn = plot
    ? `<button type="button" class="osprey-btn primary" data-action="plot"` +
      `${plot.disabled ? ' disabled' : ''}${plot.title ? ` title="${esc(plot.title)}"` : ''}>${esc(plot.label)}</button>`
    : '';
  const sendBtn = send
    ? '<button type="button" class="osprey-btn" data-action="send">Send to assistant</button>'
    : '';
  return (
    `<div class="allpvs-selection-bar" role="toolbar" aria-label="Selection">` +
    `<span><strong>${fmt(selected)}</strong> selected</span>${plotBtn}${sendBtn}` +
    '<button type="button" class="osprey-btn" data-action="copy-names">Copy names</button>' +
    '<button type="button" class="osprey-btn" data-action="clear">Clear</button></div>'
  );
}
