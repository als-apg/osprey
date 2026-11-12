// @ts-check
/**
 * OSPREY Channel Finder — All PVs markup
 *
 * Every string the All PVs view (allpvs.js) draws, as pure functions of the
 * answer and the operator's choices. Values from the directory and the control
 * system are escaped here.
 */

import { FACET_KEYS, formatRoute } from './route.js';
import { esc } from './utils.js';

/** @typedef {import('./directory-api.js').DirectoryPage} DirectoryPage */
/** @typedef {import('./directory-api.js').ValueRow} ValueRow */
/** @typedef {import('./route.js').FacetKey} FacetKey */

/** Rows per page: the most the values route reads at once. */
export const PAGE_SIZE = 50;

/**
 * The sidebar title of each route filter.
 * @type {Record<FacetKey, string>}
 */
const FACET_TITLES = { origin: 'Origin', ioc: 'IOC', record_type: 'Record type', alarm: 'Alarm' };

/** @type {Record<string, string>} */
const MARKS = { curated: '◆', directory: '◇' };

/** @type {Record<string, string>} */
const ORIGIN_LABELS = { curated: '◆ Curated', directory: '◇ Control system only' };

/** @param {number} n */
const fmt = (n) => n.toLocaleString('en-US');

/**
 * The view's skeleton: the filter sidebar and its handle, then the section
 * header, the request, the rows and the actions.
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
          <span class="allpvs-partial"></span>
          <span class="allpvs-grow"></span>
          <button type="button" class="osprey-icon-btn" data-action="refresh"
                  aria-label="Refresh values" title="Refresh values" disabled></button>
          <button type="button" class="allpvs-link" data-action="request" hidden>show request</button>
        </div>
        <div class="allpvs-request" hidden>
          <code></code>
          <button type="button" class="osprey-icon-btn" data-action="copy-request"
                  aria-label="Copy request" title="Copy request"></button>
        </div>
        <div class="allpvs-body"></div>
        <div class="allpvs-actions"></div>
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
 * The filters: one section per facet with its counts, a value in force kept
 * even when the answer has no count for it; or the reason there are none.
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
  return FACET_KEYS.map((key) => {
    const counts = new Map((facets[key] ?? []).map((entry) => [entry.value, entry.count]));
    for (const value of filters[key]) if (!counts.has(value)) counts.set(value, 0);
    if (!counts.size) return '';
    const items = [...counts]
      .map(([value, count]) => {
        const label = key === 'origin' ? (ORIGIN_LABELS[value] ?? value) : value;
        return (
          `<button type="button" class="osprey-list-row allpvs-facet" data-facet="${key}"` +
          ` data-value="${esc(value)}" aria-pressed="${filters[key].has(value)}">` +
          `<span class="osprey-list-title">${esc(label)}</span>` +
          `<span class="allpvs-facet-n">${fmt(count)}</span></button>`
        );
      })
      .join('');
    return `<div class="allpvs-facet-group"><div class="osprey-section-label">${esc(FACET_TITLES[key])}</div>${items}</div>`;
  }).join('');
}

/**
 * The current value cell of one row: "…" while unread, "no answer" for a PV
 * that did not answer, the value otherwise, its time and alarm on hover.
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
  const value = typeof row.value === 'object' ? JSON.stringify(row.value) : String(row.value);
  // The EPICS severity code; 0 is NO_ALARM.
  const alarm = row.severity && row.status ? row.status : '';
  const title = [row.timestamp, alarm].filter(Boolean).join(' · ');
  const attrs = alarm ? ` data-alarm="${esc(String(row.severity))}"` : '';
  return `<td class="allpvs-value"${attrs} title="${esc(title)}">${esc(value)}</td>`;
}

/**
 * The rows of one page.
 * @param {DirectoryPage} answer
 * @param {Set<string>} selected
 * @param {Map<string, ValueRow>|null} values - null while unread
 * @param {string} valuesReason
 * @returns {string}
 */
export function tableHtml(answer, selected, values, valuesReason) {
  const rows = answer.rows;
  const all = rows.every((row) => selected.has(row.name));
  const body = rows
    .map((row) => {
      const on = selected.has(row.name);
      const classes = [on ? 'sel' : '', row.status === 'Inactive' ? 'inactive' : '']
        .filter(Boolean)
        .join(' ');
      const markClass = row.origin === 'curated' ? 'mark' : 'mark cs';
      const ioc = row.ioc
        ? `<a href="${esc(formatRoute({ view: 'ioc', params: { name: row.ioc } }))}">${esc(row.ioc)}</a>`
        : '';
      return (
        `<tr class="${classes}" data-name="${esc(row.name)}">` +
        `<td><input type="checkbox" ${on ? 'checked' : ''} aria-label="Select ${esc(row.name)}"></td>` +
        `<td><span class="${markClass}">${MARKS[row.origin] ?? ''}</span></td>` +
        `<td class="name"><a href="${esc(formatRoute({ view: 'pv', params: { name: row.name } }))}">${esc(row.name)}</a></td>` +
        `<td class="desc${row.description ? '' : ' dim'}">${esc(row.description || 'no description')}</td>` +
        valueCell(values?.get(row.name), values !== null, valuesReason) +
        `<td class="dim">${esc(row.record_type ?? '')}</td>` +
        `<td class="dim">${ioc}</td>` +
        `<td class="status">${esc(row.status ?? '')}</td></tr>`
      );
    })
    .join('');
  return `
    <table class="allpvs-table">
      <thead><tr>
        <th class="col-select"><input type="checkbox" data-action="select-page"
            aria-label="Select all on this page" ${all ? 'checked' : ''}></th>
        <th class="col-mark"></th><th>Name</th><th>Description</th><th>Value</th>
        <th>Record</th><th>IOC</th><th>Status</th>
      </tr></thead>
      <tbody>${body}</tbody>
    </table>`;
}

/**
 * @typedef {{
 *   answer: DirectoryPage|null,
 *   page: number,
 *   selected: number,
 *   csvStatus: string,
 *   plot: { disabled: boolean, label: string, title: string }|null,
 *   send: boolean,
 * }} ActionsModel
 */

/**
 * The selection, the pager and the actions: CSV, then Plot (only with a Plot
 * button model) and Send (only when a host takes it).
 * @param {ActionsModel} model
 * @returns {string}
 */
export function actionsHtml({ answer, page, selected, csvStatus, plot, send }) {
  const found = Boolean(answer && answer.outcome !== 'unavailable' && answer.rows.length > 0);
  const pages = typeof answer?.total === 'number' ? Math.ceil(answer.total / PAGE_SIZE) : null;
  const more = pages === null ? Boolean(answer?.truncated) : page + 1 < pages;
  const pager =
    found && (page > 0 || more)
      ? `<button type="button" class="osprey-icon-btn" data-action="prev" aria-label="Previous page"${page > 0 ? '' : ' disabled'}>‹</button>` +
        `<span>page ${page + 1}${pages === null ? '' : ` of ${pages}`}</span>` +
        `<button type="button" class="osprey-icon-btn" data-action="next" aria-label="Next page"${more ? '' : ' disabled'}>›</button>`
      : '';
  const selection = selected
    ? `<span>${selected} selected</span><button type="button" class="allpvs-link" data-action="clear">clear</button>`
    : '';
  const status = csvStatus
    ? `<span class="osprey-chip osprey-chip--warn"><span class="osprey-chip-label">${esc(csvStatus)}</span></span>`
    : '';
  const csv = found
    ? '<button type="button" class="osprey-btn" data-action="csv">Export CSV</button>'
    : '';
  let plotBtn = '';
  if (plot) {
    const disabled = plot.disabled || selected === 0;
    const title = selected === 0 ? 'Select PVs to plot' : plot.title;
    plotBtn =
      `<button type="button" class="osprey-btn primary" data-action="plot"` +
      `${disabled ? ' disabled' : ''}${title ? ` title="${esc(title)}"` : ''}>${esc(plot.label)}</button>`;
  }
  const sendBtn = send
    ? `<button type="button" class="osprey-btn" data-action="send"${selected ? '' : ' disabled'}>Send to assistant</button>`
    : '';
  return `${selection}<span class="allpvs-grow"></span>${pager}${status}${csv}${plotBtn}${sendBtn}`;
}
