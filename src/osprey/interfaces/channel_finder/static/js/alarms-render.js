// @ts-check
/**
 * OSPREY Channel Finder — Alarms markup
 *
 * Every string the Alarms view (alarms.js) draws, as pure functions of the
 * alarm route's answer, the PV facts and the operator's filters. Values from
 * the alarm feed, the directory and the control system are escaped here.
 * Rows are drawn in the order the route serves them (worst first, then
 * newest); a filter only removes rows.
 */

import { configEntries } from './alarm-text.js';
import { NO_TARGET, timeText } from './pv-page-sections.js';
import { formatRoute } from './route.js';
import { severityMark, severityOf, worstSeverity } from './severity.js';
import { esc } from './utils.js';
import { formatValue } from './value-format.js';

/** @typedef {import('./directory-api.js').ActiveAlarms} ActiveAlarms */
/** @typedef {import('./directory-api.js').AlarmRow} AlarmRow */
/** @typedef {import('./directory-api.js').ValueRow} ValueRow */
/** @typedef {import('./directory-api.js').PvFacts} PvFacts */
/** @typedef {import('./directory-api.js').PvFactsSection} PvFactsSection */
/** @typedef {import('./directory-api.js').DirectoryRecord} DirectoryRecord */
/** @typedef {import('./directory-api.js').AlarmConfigRecord} AlarmConfigRecord */
/** @typedef {import('./directory-api.js').AlarmHistoryRow} AlarmHistoryRow */
/** @typedef {import('./severity.js').Severity} Severity */

/**
 * A tile's filter: a severity, or a flag.
 * @typedef {'MAJOR'|'INVALID'|'MINOR'|'UNDEFINED'|'unack'|'latched'} TileFilter
 */

/** @typedef {{ severity: TileFilter|null, node: string|null }} Filters */

/**
 * One node of the alarm tree: its path (the key a filter holds), its last
 * segment, its depth, and the worst severity and count of the rows below it.
 * @typedef {{ path: string, label: string, depth: number, worst: Severity, count: number }} TreeNode
 */

/** The severity tiles, in the mockup's display order (not the rank). */
const SEVERITY_TILES = /** @type {const} */ (['MAJOR', 'INVALID', 'MINOR', 'UNDEFINED']);

export const NO_TREE = 'No alarm tree on this target';
const NO_LOG = 'No alarm log on this target. Only the current state and its start time are known.';
const READ_ONLY = "Read-only. Acknowledge in the alarm server's own client.";

/** @param {number} n */
const fmt = (n) => n.toLocaleString('en-US');

// ---- Rows and filters ----

/**
 * The tree path segments of a row, without the PV itself at the end.
 * @param {AlarmRow} row
 * @returns {string[]}
 */
function segments(row) {
  const parts = (row.path ?? '').split('/').filter(Boolean);
  if (parts.at(-1) === row.pv) parts.pop();
  return parts;
}

/**
 * Whether a row lies under the tree node `path`.
 * @param {AlarmRow} row
 * @param {string} path
 * @returns {boolean}
 */
function under(row, path) {
  const node = path.split('/').filter(Boolean);
  const parts = segments(row);
  return node.length <= parts.length && node.every((part, i) => parts[i] === part);
}

/**
 * Whether a row passes a tile filter.
 * @param {AlarmRow} row
 * @param {TileFilter} filter
 * @returns {boolean}
 */
function passes(row, filter) {
  if (filter === 'unack') return !row.acknowledged;
  if (filter === 'latched') return row.latched;
  return severityOf(row.severity) === filter;
}

/**
 * The rows the filters keep, in the served order.
 * @param {AlarmRow[]} rows
 * @param {Filters} filters
 * @returns {AlarmRow[]}
 */
export function filterRows(rows, { severity, node }) {
  return rows.filter(
    (row) => (severity === null || passes(row, severity)) && (node === null || under(row, node))
  );
}

/**
 * The count on each tile: per severity from the answer's counts (an `_ACK`
 * spelling counts under its severity), per flag from the active rows.
 * @param {ActiveAlarms} answer
 * @returns {Record<TileFilter, number>}
 */
export function tileCounts(answer) {
  /** @type {Record<TileFilter, number>} */
  const counts = { MAJOR: 0, INVALID: 0, MINOR: 0, UNDEFINED: 0, unack: 0, latched: 0 };
  for (const [key, n] of Object.entries(answer.counts ?? {})) {
    const severity = severityOf(key);
    const count = Number(n);
    if (severity in counts && Number.isInteger(count) && count > 0) {
      counts[/** @type {TileFilter} */ (severity)] += count;
    }
  }
  for (const row of answer.rows ?? []) {
    if (!row.acknowledged) counts.unack++;
    if (row.latched) counts.latched++;
  }
  return counts;
}

/**
 * The number of channels in alarm, and the worst of their severities.
 * @param {ActiveAlarms} answer
 * @returns {{ active: number, worst: Severity }}
 */
export function activeSummary(answer) {
  /** @type {Severity[]} */
  const present = [];
  let active = 0;
  for (const [key, n] of Object.entries(answer.counts ?? {})) {
    const severity = severityOf(key);
    const count = Number(n);
    if (severity === 'OK' || severity === 'NONE' || !Number.isInteger(count) || count <= 0) {
      continue;
    }
    active += count;
    present.push(severity);
  }
  return { active, worst: worstSeverity(present) };
}

/**
 * The alarm tree of the rows that carry a path, each node's children sorted
 * by name, depth first; empty when no row carries one.
 * @param {AlarmRow[]} rows
 * @returns {TreeNode[]}
 */
export function buildTree(rows) {
  /** @typedef {{ label: string, path: string, severities: Severity[], count: number, children: Map<string, Branch> }} Branch */
  /** @type {Map<string, Branch>} */
  const roots = new Map();
  for (const row of rows) {
    let level = roots;
    let path = '';
    for (const part of segments(row)) {
      path = `${path}/${part}`;
      let branch = level.get(part);
      if (!branch) {
        branch = { label: part, path, severities: [], count: 0, children: new Map() };
        level.set(part, branch);
      }
      branch.severities.push(severityOf(row.severity));
      branch.count++;
      level = branch.children;
    }
  }
  /** @type {TreeNode[]} */
  const nodes = [];
  /** @param {Map<string, Branch>} level @param {number} depth */
  const walk = (level, depth) => {
    for (const key of [...level.keys()].sort()) {
      const branch = /** @type {Branch} */ (level.get(key));
      const { label, path, count } = branch;
      nodes.push({ path, label, depth, count, worst: worstSeverity(branch.severities) });
      walk(branch.children, depth + 1);
    }
  };
  walk(roots, 0);
  return nodes;
}

// ---- Times ----

/**
 * How long ago `time` was, in the largest two units that matter.
 * @param {string|null|undefined} time
 * @param {number} now - ms since the epoch
 * @returns {string}
 */
export function sinceText(time, now) {
  const at = time ? Date.parse(time) : NaN;
  if (Number.isNaN(at)) return '';
  const seconds = Math.max(0, Math.floor((now - at) / 1000));
  if (seconds < 60) return `${seconds} s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes} min`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return minutes % 60 ? `${hours} h ${minutes % 60} min` : `${hours} h`;
  const days = Math.floor(hours / 24);
  return hours % 24 ? `${days} d ${hours % 24} h` : `${days} d`;
}

/**
 * A placeholder for how long ago `time` was: the view ticks its text in place
 * (`sinceText`), so the markup never depends on the clock. Empty when `time`
 * is not a time.
 * @param {string|null|undefined} time
 * @returns {string}
 */
export function sinceHtml(time) {
  const at = time ? Date.parse(time) : NaN;
  if (Number.isNaN(at)) return '';
  return `<span data-since="${esc(String(time))}"></span>`;
}

// ---- Skeleton, band and banner ----

/**
 * The view's skeleton: the tree sidebar and its handle, then the banner, the
 * summary band, the centred state's host, and the list beside the inspector.
 * @returns {string}
 */
export function skeletonHtml() {
  return `
    <div class="alarms">
      <aside class="osprey-panel-sidebar alarms-tree" data-open="false" aria-label="Alarm tree" hidden></aside>
      <div class="osprey-splitter" role="separator" aria-orientation="vertical" hidden
           aria-label="Resize the alarm tree" tabindex="0" title="Drag to resize · double-click to hide"></div>
      <div class="alarms-main">
        <div class="alarms-banner" role="status" hidden></div>
        <div class="alarms-band" hidden></div>
        <div class="alarms-state"></div>
        <div class="alarms-body" hidden>
          <div class="alarms-list"></div>
          <aside class="alarms-inspector" aria-label="Selected alarm"></aside>
        </div>
      </div>
    </div>`;
}

/**
 * The toolbar's result count.
 * @param {number} n
 * @returns {string}
 */
export function countHtml(n) {
  return `<strong>${fmt(n)}</strong> active`;
}

/**
 * The feed chip's label markup and tone. The last contact moves with every
 * answer of a live feed, so its placeholder carries no time: the view writes
 * the time into it in place, and the band's markup stays the same.
 * @param {ActiveAlarms} answer
 * @returns {{ html: string, tone: 'ok'|'warn' }}
 */
export function feedChip(answer) {
  if (answer.stale) return { html: 'feed stale', tone: 'warn' };
  return {
    html: sinceHtml(answer.last_contact)
      ? 'feed live · last contact <span data-since=""></span> ago'
      : 'feed live',
    tone: 'ok',
  };
}

/**
 * The summary band: a tile per severity (UNDEFINED only when not zero), the
 * Unacknowledged and Latched tiles, Refresh, and the feed chip.
 * @param {ActiveAlarms} answer
 * @param {TileFilter|null} pressed
 * @returns {string}
 */
export function bandHtml(answer, pressed) {
  const counts = tileCounts(answer);
  /** @param {TileFilter} key @param {string} label */
  const tile = (key, label) => {
    const n = counts[key];
    const classes = ['alarms-tile', n ? '' : 'is-zero'].filter(Boolean).join(' ');
    return (
      `<button type="button" class="${classes}" data-tile="${key}" aria-pressed="${pressed === key}">` +
      `<span class="alarms-tile-n">${fmt(n)}</span><span class="alarms-tile-k">${label}</span></button>`
    );
  };
  const severities = SEVERITY_TILES.filter((key) => key !== 'UNDEFINED' || counts.UNDEFINED)
    .map((key) => tile(key, severityMark(key)))
    .join('');
  const chip = feedChip(answer);
  return (
    severities +
    '<span class="alarms-band-rule" aria-hidden="true"></span>' +
    tile('unack', 'Unacknowledged') +
    tile('latched', 'Latched') +
    '<span class="alarms-grow"></span>' +
    '<button type="button" class="osprey-icon-btn" data-action="refresh"' +
    ' aria-label="Refresh" title="Refresh"></button>' +
    `<button type="button" class="osprey-chip osprey-chip--${chip.tone}" data-action="feed">` +
    `<span class="osprey-chip-label">${chip.html}</span></button>`
  );
}

/**
 * The warning over a stale feed.
 * @param {string|null|undefined} lastContact
 * @returns {string}
 */
export function bannerHtml(lastContact) {
  const since = timeText(lastContact);
  const when = since ? ` No contact since ${esc(since)}.` : '';
  return (
    `<strong>Alarm feed not updating.</strong>${when}` +
    ' The states below are the last ones received and may be out of date.'
  );
}

/**
 * The feed chip's popover rows: where the states come from.
 * @param {ActiveAlarms} answer
 * @returns {{ label: string, value: string }[]}
 */
export function feedRows(answer) {
  const { source } = answer;
  const rows = [
    { label: 'Target', value: source.target },
    { label: 'Service', value: source.label || source.service },
    { label: 'Rehearsal', value: source.rehearsal ? 'yes' : 'no' },
  ];
  if (typeof answer.total === 'number')
    rows.push({ label: 'PVs in tree', value: fmt(answer.total) });
  const contact = timeText(answer.last_contact);
  if (contact) rows.push({ label: 'Last contact', value: contact });
  return rows;
}

// ---- Tree ----

/**
 * The alarm tree in the sidebar.
 * @param {TreeNode[]} nodes
 * @param {string|null} selected
 * @returns {string}
 */
export function treeHtml(nodes, selected) {
  const rows = nodes
    .map(
      (node) =>
        `<button type="button" class="osprey-nav-row alarms-node" data-node="${esc(node.path)}"` +
        ` style="--alarms-depth: ${node.depth}" aria-pressed="${node.path === selected}"` +
        ` title="${esc(`${node.path} · worst ${node.worst}`)}">` +
        `<span class="osprey-nav-label">${severityMark(node.worst, node.label)}</span>` +
        `<span class="osprey-nav-badge">${fmt(node.count)}</span></button>`
    )
    .join('');
  return `<div class="osprey-section-label">Alarm tree</div>${rows}`;
}

// ---- List ----

/**
 * The value cell of one row: "…" while unread, "—" when not read.
 * @param {ValueRow|undefined} value
 * @param {boolean} read
 * @returns {string}
 */
function valueCell(value, read) {
  if (!read) return '<td class="alarms-value dim">…</td>';
  if (!value) return '<td class="alarms-value dim" title="not read">—</td>';
  const { text, unit, title } = formatValue(value);
  const dim = value.outcome === 'found' ? '' : ' dim';
  const units = unit ? ` <span class="alarms-unit">${esc(unit)}</span>` : '';
  return `<td class="alarms-value${dim}" title="${esc(title)}">${esc(text)}${units}</td>`;
}

/**
 * The flags of a row.
 * @param {AlarmRow} row
 * @returns {string}
 */
function flagsHtml(row) {
  const latched = row.latched ? '<span class="alarms-flag is-hot">latched</span>' : '';
  const ack = row.acknowledged
    ? '<span class="alarms-flag">acknowledged</span>'
    : '<span class="alarms-flag">unacknowledged</span>';
  return latched + ack;
}

/**
 * The active rows, in the order given; with `cut`, a line saying the answer
 * holds only the worst of the active alarms.
 * @param {AlarmRow[]} rows
 * @param {{
 *   selected: string|null,
 *   values: Map<string, ValueRow>|null,
 *   stale: boolean,
 *   cut: { served: number, active: number }|null,
 * }} model
 * @returns {string}
 */
export function listHtml(rows, { selected, values, stale, cut }) {
  const body = rows
    .map((row) => {
      const severity = severityOf(row.severity);
      const classes = [
        `alarms-row--${severity.toLowerCase()}`,
        row.pv === selected ? 'sel' : '',
        stale || row.stale ? 'is-stale' : '',
      ]
        .filter(Boolean)
        .join(' ');
      const link = formatRoute({ view: 'pv', params: { name: row.pv } });
      return (
        `<tr class="${classes}" data-name="${esc(row.pv)}" tabindex="0" aria-selected="${row.pv === selected}">` +
        `<td class="alarms-sevbar"></td>` +
        `<td>${severityMark(severity)}</td>` +
        `<td class="alarms-pv"><a href="${esc(link)}">${esc(row.pv)}</a></td>` +
        `<td class="alarms-status">${esc(row.status)}</td>` +
        valueCell(values?.get(row.pv), values !== null) +
        `<td class="alarms-for" title="${esc(timeText(row.time))}">${sinceHtml(row.time) || '—'}</td>` +
        `<td class="alarms-flags">${flagsHtml(row)}</td></tr>`
      );
    })
    .join('');
  return `
    <table class="alarms-table">
      <thead><tr>
        <th class="alarms-sevbar"></th><th>Severity</th><th>PV</th><th>Status</th>
        <th>Value now</th><th>For</th><th>Flags</th>
      </tr></thead>
      <tbody>${body}</tbody>
    </table>${
      cut
        ? `<p class="alarms-muted alarms-cut">Worst ${fmt(cut.served)} of ${fmt(cut.active)} active alarms shown</p>`
        : ''
    }`;
}

// ---- Inspector ----

/**
 * @typedef {{
 *   row: AlarmRow,
 *   facts: PvFacts|null,
 *   factsReason: string,
 *   value: ValueRow|undefined,
 *   plot: boolean,
 *   send: boolean,
 * }} InspectorModel
 */

/**
 * A facts section by its id.
 * @param {PvFacts|null} facts
 * @param {PvFactsSection['section']} id
 * @returns {PvFactsSection|undefined}
 */
function section(facts, id) {
  return facts?.sections.find((candidate) => candidate.section === id);
}

/**
 * A label and its entries, each as title and details text.
 * @param {string} label
 * @param {unknown} value
 * @param {string} className
 * @returns {string}
 */
function entriesHtml(label, value, className) {
  const entries = configEntries(value);
  if (!entries.length) return '';
  const items = entries
    .map(
      (entry) =>
        `<div class="${className}">` +
        (entry.title ? `<strong>${esc(entry.title)}</strong>` : '') +
        `<span>${esc(entry.details)}</span></div>`
    )
    .join('');
  return `<div class="alarms-insp-part"><div class="osprey-section-label">${esc(label)}</div>${items}</div>`;
}

/**
 * The transitions from the alarm history section, or why there are none.
 * @param {PvFactsSection|undefined} history
 * @param {boolean} loaded - whether the facts have answered
 * @returns {string}
 */
function transitionsHtml(history, loaded) {
  if (!loaded) return '';
  if (!history || (history.outcome === 'unavailable' && history.reason === NO_TARGET)) {
    return `<p class="alarms-muted">${esc(NO_LOG)}</p>`;
  }
  const label = '<div class="osprey-section-label">Transitions</div>';
  if (history.outcome === 'unavailable') {
    return `<div class="alarms-insp-part">${label}<p class="alarms-muted">Unavailable: ${esc(history.reason || 'no reason given')}</p></div>`;
  }
  const rows = /** @type {AlarmHistoryRow[]} */ (history.rows ?? []);
  if (!rows.length) {
    return `<div class="alarms-insp-part">${label}<p class="alarms-muted">No rows recorded</p></div>`;
  }
  const items = rows
    .map(
      (row) =>
        `<li><span class="alarms-tl-time">${esc(timeText(row.time))}</span>` +
        `${severityMark(severityOf(row.severity))}` +
        `<span>${esc([row.status, row.message].filter(Boolean).join(' · '))}</span></li>`
    )
    .join('');
  return `<div class="alarms-insp-part">${label}<ul class="alarms-tl">${items}</ul></div>`;
}

/**
 * The inspector of the selected row: what is wrong, what to do, the facts,
 * the transitions, and that nothing here writes.
 * @param {InspectorModel} model
 * @returns {string}
 */
export function inspectorHtml({ row, facts, factsReason, value, plot, send }) {
  const severity = severityOf(row.severity);
  const loaded = facts !== null;
  const directory = /** @type {DirectoryRecord|null|undefined} */ (
    section(facts, 'directory')?.record
  );
  const dirRow = directory?.row;
  const config = /** @type {AlarmConfigRecord|null|undefined} */ (
    section(facts, 'alarm_config')?.record
  );
  const link = formatRoute({ view: 'pv', params: { name: row.pv } });

  const head =
    `<div class="alarms-insp-head">${severityMark(severity, [severity, row.status].filter(Boolean).join(' · '))}` +
    `<h2 class="alarms-insp-name">${esc(row.pv)}</h2>` +
    (dirRow?.description ? `<p class="alarms-insp-desc">${esc(dirRow.description)}</p>` : '') +
    (factsReason ? `<p class="alarms-muted">PV facts: ${esc(factsReason)}</p>` : '') +
    '</div>';

  const actions =
    '<div class="alarms-insp-actions">' +
    `<a class="osprey-btn" href="${esc(link)}">Open PV page</a>` +
    (plot
      ? '<button type="button" class="osprey-btn primary" data-action="plot">Plot</button>'
      : '') +
    (send
      ? '<button type="button" class="osprey-btn" data-action="send">Send to assistant</button>'
      : '') +
    '</div>';

  const latched = row.latched
    ? `<p class="alarms-latched">The record is back to ${esc(row.status || 'its own alarm')}. The alarm holds until it is acknowledged.</p>`
    : '';

  /** @type {string} */
  let valueText = '…';
  let valueUnit = '';
  if (value) {
    const formatted = formatValue(value);
    valueText = formatted.text;
    valueUnit = formatted.unit;
  }
  const valueHtml =
    '<div class="alarms-insp-part"><div class="osprey-section-label">Value now</div>' +
    `<div class="alarms-insp-value">${esc(valueText)}${valueUnit ? ` <span class="alarms-unit">${esc(valueUnit)}</span>` : ''}</div></div>`;

  const guidance = entriesHtml('Guidance', config?.guidance, 'alarms-guidance');

  /** @type {string[]} */
  const factRows = [];
  const since = timeText(row.time);
  if (since) factRows.push(`<dt>Since</dt><dd>${esc(since)} (${sinceHtml(row.time)} ago)</dd>`);
  /** @type {[string, string][]} */
  const pairs = [];
  const recordNow = value?.outcome === 'found' ? value.status : null;
  if (recordNow) pairs.push(['Record now', recordNow]);
  if (config && config.latching !== null) pairs.push(['Latching', config.latching ? 'yes' : 'no']);
  const path = row.path || config?.path;
  if (path) pairs.push(['Tree path', path]);
  for (const [label, text] of pairs) factRows.push(`<dt>${esc(label)}</dt><dd>${esc(text)}</dd>`);
  if (dirRow?.ioc) {
    const iocLink = formatRoute({ view: 'ioc', params: { name: dirRow.ioc } });
    factRows.push(`<dt>IOC</dt><dd><a href="${esc(iocLink)}">${esc(dirRow.ioc)}</a></dd>`);
  }
  const factsHtml = factRows.length
    ? `<div class="alarms-insp-part"><div class="osprey-section-label">Facts</div><dl class="alarms-facts">${factRows.join('')}</dl></div>`
    : '';
  const displays = entriesHtml('Displays', config?.displays, 'alarms-display');

  return (
    head +
    actions +
    latched +
    valueHtml +
    guidance +
    factsHtml +
    displays +
    transitionsHtml(section(facts, 'alarm_history'), loaded) +
    `<p class="alarms-muted alarms-read-only">${esc(READ_ONLY)}</p>`
  );
}
