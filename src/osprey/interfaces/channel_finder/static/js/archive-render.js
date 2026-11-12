// @ts-check
/**
 * OSPREY Channel Finder — Archive markup
 *
 * Every string the Archive view (archive.js) draws, as pure functions of the
 * archive summary and the clock. Values from the archive and the configuration
 * are escaped here. The view is read-only: nothing drawn here writes.
 *
 *   head    Archive [newest sample 8 s ago] …… [i] [Refresh]
 *   tiles   Archived PVs · Retention · Cadence · Newest sample · Store
 *   cards   Coverage by IOC · Retention window | Configuration · Targets
 *
 * Retention, Cadence and the Retention window are the deployment's own store's
 * (the summary carries its `va_archiver` configuration); another archiver has
 * no such tiles.
 */

import { sinceText } from './duration.js';
import { formatRoute } from './route.js';
import { esc } from './utils.js';

/** @typedef {import('./directory-api.js').ArchiveSummary} ArchiveSummary */

/** The configuration card's note: where the configuration changes. */
export const READ_ONLY_NOTE =
  'To change these, edit the profile and redeploy. This panel does not write configuration.';

/** @param {number} n */
const fmt = (n) => n.toLocaleString('en-US');

/**
 * The knobs of the deployment's own store, or null for any other archive.
 * @param {ArchiveSummary} summary
 * @returns {Record<string, number>|null}
 */
function ownStore(summary) {
  const knobs = summary.configuration?.va_archiver;
  return knobs ? /** @type {Record<string, number>} */ (knobs) : null;
}

/**
 * The view's skeleton: the state host and the page host.
 * @returns {string}
 */
export function skeletonHtml() {
  return '<div class="archive"><div class="archive-state"></div><div class="archive-page"></div></div>';
}

/**
 * The newest-sample chip of the head: its text and tone, or null when the
 * newest sample is unknown. Within the recorder's freshness bound it is `ok`,
 * beyond it `warn`; an archive with no bound is `info`.
 * @param {ArchiveSummary} summary
 * @param {number} now - ms since the epoch
 * @returns {{ text: string, tone: 'ok'|'warn'|'info' }|null}
 */
export function freshnessChip(summary, now) {
  const newest = summary.newest_sample;
  const age = sinceText(newest, now);
  if (!newest || !age) return null;
  const bound = summary.freshness_bound_s;
  const seconds = (now - Date.parse(newest)) / 1000;
  const tone = typeof bound !== 'number' ? 'info' : seconds <= bound ? 'ok' : 'warn';
  return { text: `newest sample ${age} ago`, tone };
}

/**
 * One tile: its label, its value and the line under it.
 * @param {string} id
 * @param {string} label
 * @param {string} value
 * @param {string} sub
 * @returns {string}
 */
function tile(id, label, value, sub) {
  return (
    `<div class="osprey-card archive-tile" data-tile="${id}">` +
    `<div class="archive-tile-label">${esc(label)}</div>` +
    `<div class="archive-tile-value">${esc(value)}</div>` +
    `<div class="archive-tile-sub">${esc(sub)}</div></div>`
  );
}

/**
 * The tiles: Archived PVs, then for the deployment's own store Retention and
 * Cadence, then Newest sample and Store.
 * @param {ArchiveSummary} summary
 * @param {number} now - ms since the epoch
 * @returns {string}
 */
export function tilesHtml(summary, now) {
  const archived = summary.archived ?? { count: null, of: null };
  const tiles = [];
  if (typeof archived.count === 'number') {
    const of = typeof archived.of === 'number' ? archived.of : null;
    const rest =
      of === null ? '' : `of ${fmt(of)} · ${fmt(Math.max(of - archived.count, 0))} not archived`;
    tiles.push(tile('archived', 'Archived PVs', fmt(archived.count), rest));
  } else {
    tiles.push(tile('archived', 'Archived PVs', 'unknown', archived.reason ?? ''));
  }
  const knobs = ownStore(summary);
  if (knobs) {
    const oldest = summary.oldest_sample
      ? `oldest sample ${summary.oldest_sample.slice(0, 10)}`
      : '';
    tiles.push(tile('retention', 'Retention', `${knobs.retention_days} d`, oldest));
    tiles.push(
      tile(
        'cadence',
        'Cadence',
        `${knobs.hot_cadence_sec} s`,
        `for ${knobs.hot_span_hours} h, then ${knobs.tail_cadence_sec} s`
      )
    );
  }
  const age = sinceText(summary.newest_sample, now);
  const writes = knobs ? `recorder writes every ${knobs.recorder_cadence_sec} s` : '';
  tiles.push(tile('newest', 'Newest sample', age ? `${age} ago` : 'unknown', writes));
  const type = String(summary.configuration?.archiver?.type ?? summary.source.service);
  tiles.push(tile('store', 'Store', type, ''));
  return `<div class="archive-tiles">${tiles.join('')}</div>`;
}

/**
 * The Coverage by IOC card: per IOC its PVs, how many the archive records and
 * how many it does not, a not-archived count linking to All PVs filtered by
 * that IOC and `archived=no`; or why there is no list.
 * @param {ArchiveSummary} summary
 * @returns {string}
 */
export function coverageHtml(summary) {
  const rows = summary.by_ioc ?? [];
  let body;
  if (!rows.length) {
    body = `<p class="archive-note">${esc(summary.by_ioc_reason || 'No IOCs listed')}</p>`;
  } else {
    const lines = rows
      .map(({ ioc, pvs, archived }) => {
        const missing = Math.max(pvs - archived, 0);
        const iocLink = formatRoute({ view: 'ioc', params: { name: ioc } });
        const filtered = formatRoute({ view: 'allpvs', params: { ioc: [ioc], archived: ['no'] } });
        const share = pvs ? Math.round((archived / pvs) * 1000) / 10 : 0;
        const notArchived = missing ? `<a href="${esc(filtered)}">${fmt(missing)}</a>` : '0';
        return (
          `<tr><td><a href="${esc(iocLink)}">${esc(ioc)}</a></td>` +
          `<td class="num">${fmt(pvs)}</td><td class="num">${fmt(archived)}</td>` +
          `<td class="num">${notArchived}</td>` +
          `<td><span class="archive-cov" title="${share}% archived">` +
          `<span class="archive-cov-bar" style="--archive-coverage: ${share}%"></span></span></td></tr>`
        );
      })
      .join('');
    body =
      '<table class="archive-table"><thead><tr><th>IOC</th><th class="num">PVs</th>' +
      '<th class="num">Archived</th><th class="num">Not archived</th><th>Coverage</th>' +
      `</tr></thead><tbody>${lines}</tbody></table>`;
  }
  return card('coverage', 'Coverage by IOC', body);
}

/**
 * The Retention window card of the deployment's own store: the dense and the
 * coarse span to scale; empty for any other archive.
 * @param {ArchiveSummary} summary
 * @returns {string}
 */
export function retentionHtml(summary) {
  const knobs = ownStore(summary);
  if (!knobs) return '';
  const total = knobs.retention_days * 24;
  const dense = Math.min(knobs.hot_span_hours, total);
  const share = total ? Math.round((dense / total) * 1000) / 10 : 0;
  const body =
    `<div class="archive-window" style="--archive-dense: ${share}%">` +
    '<span class="archive-window-coarse"></span><span class="archive-window-dense"></span></div>' +
    '<div class="archive-window-axis">' +
    `<span>−${knobs.retention_days} d · ${knobs.tail_cadence_sec} s</span>` +
    `<span>−${knobs.hot_span_hours} h · ${knobs.hot_cadence_sec} s</span><span>now</span></div>` +
    `<p class="archive-note">Samples older than ${knobs.hot_span_hours} h are kept at one per ` +
    `${knobs.tail_cadence_sec} s; samples older than ${knobs.retention_days} d expire.</p>`;
  return card('retention', 'Retention window', body);
}

/**
 * The archive's configuration as YAML text.
 * @param {Record<string, Record<string, string|number>>} configuration
 * @returns {string}
 */
export function configurationText(configuration) {
  return Object.entries(configuration ?? {})
    .map(
      ([block, keys]) =>
        `${block}:\n` +
        Object.entries(keys)
          .map(([key, value]) => `  ${key}: ${value}`)
          .join('\n')
    )
    .join('\n');
}

/**
 * The Configuration card: the configuration as rendered, as text, with Copy
 * and where it changes. No control here writes.
 * @param {ArchiveSummary} summary
 * @returns {string}
 */
export function configurationHtml(summary) {
  const body =
    `<pre class="archive-config">${esc(configurationText(summary.configuration))}</pre>` +
    `<p class="archive-note">${esc(READ_ONLY_NOTE)}</p>`;
  const copy =
    '<button type="button" class="osprey-icon-btn" data-action="copy"' +
    ' aria-label="Copy configuration" title="Copy configuration"></button>';
  return card('configuration', 'Configuration', body, copy);
}

/**
 * The Targets card: each configured control target, and whether the archive
 * serves it.
 * @param {ArchiveSummary} summary
 * @returns {string}
 */
export function targetsHtml(summary) {
  const items = (summary.targets ?? [])
    .map(
      ({ target, archive }) =>
        `<dt>${esc(target)}</dt><dd>${archive ? 'archive' : 'no archive configured'}</dd>`
    )
    .join('');
  return card('targets', 'Targets', `<dl class="archive-facts">${items}</dl>`);
}

/**
 * A card with its title, an optional control beside it, and its body.
 * @param {string} id
 * @param {string} title
 * @param {string} body
 * @param {string} [control]
 * @returns {string}
 */
function card(id, title, body, control = '') {
  return (
    `<section class="osprey-card archive-card" data-card="${id}">` +
    `<h3 class="archive-card-title">${esc(title)}${control}</h3>${body}</section>`
  );
}

/**
 * The whole page of a summary that answered.
 * @param {ArchiveSummary} summary
 * @param {number} now - ms since the epoch
 * @returns {string}
 */
export function pageHtml(summary, now) {
  const chip = freshnessChip(summary, now);
  const chipHtml = chip
    ? `<span class="osprey-chip osprey-chip--${chip.tone}" data-role="newest">` +
      `<span class="osprey-chip-label">${esc(chip.text)}</span></span>`
    : '';
  const head =
    '<div class="archive-head"><h2 class="archive-title">Archive</h2>' +
    `${chipHtml}<span class="archive-grow"></span>` +
    '<button type="button" class="osprey-icon-btn" data-action="source"' +
    ' aria-label="Source" title="Source"></button>' +
    '<button type="button" class="osprey-btn" data-action="refresh">Refresh</button></div>';
  return (
    head +
    tilesHtml(summary, now) +
    '<div class="archive-grid"><div class="archive-col">' +
    coverageHtml(summary) +
    retentionHtml(summary) +
    '</div><div class="archive-col">' +
    configurationHtml(summary) +
    targetsHtml(summary) +
    '</div></div>'
  );
}
