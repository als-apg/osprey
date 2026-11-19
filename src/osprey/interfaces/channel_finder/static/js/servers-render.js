// @ts-check
/**
 * OSPREY Channel Finder — IOCs, the markup.
 *
 * Pure functions from the fleet route's answer and the view's filters to
 * escaped HTML strings (servers.js writes them). The markup never depends on the
 * clock: ages are written as `data-age` epochs that the view ticks in place.
 *
 *   header   snapshot taken <time> · <age> ago   read path <status>   <source>: <outcome> …
 *   band     alive · stalled · silent · unreached · unknown | findings … [group] [refresh]
 *   table    IOC · Host · State · In state · Findings · Directory · Sources
 *
 * Rows keep the served order; a tile and the text filter only remove rows,
 * and grouping only gathers them under their group's heading.
 */

import { durationText } from './duration.js';
import { LABELS, heading } from './labels.js';
import { formatRoute } from './route.js';
import { esc } from './utils.js';

/** @typedef {import('./directory-api.js').FleetAnswer} FleetAnswer */
/** @typedef {import('./directory-api.js').FleetSnapshot} FleetSnapshot */
/** @typedef {import('./directory-api.js').FleetRow} FleetRow */
/** @typedef {import('./directory-api.js').FleetState} FleetState */
/** @typedef {import('./directory-api.js').FleetFinding} FleetFinding */

/**
 * A pressed tile: `state:<state>` or `finding:<finding>`.
 * @typedef {string} TileFilter
 */

/** The states, in the state model's order, with their labels. */
export const STATES = /** @type {const} */ ([
  ['alive', 'Alive'],
  ['stalled', 'Stalled'],
  ['silent', 'Silent'],
  ['unreached', 'Unreached'],
  ['unknown', 'Unknown'],
]);

/** The findings, in the state model's order, with their labels. */
export const FINDINGS = /** @type {const} */ ([
  ['degraded', 'Degraded'],
  ['expected_running_not_alive', 'Expected running, not alive'],
  ['expected_stopped_alive', 'Expected stopped, alive'],
  ['alive_unexpected', 'Alive, expected by no roster'],
]);

/**
 * A state's chip tone. `unreached` and `unknown` say no state was
 * established, never that an IOC is down, so they take the neutral grey.
 * @type {Record<string, string>}
 */
const STATE_TONE = {
  alive: 'osprey-chip osprey-chip--ok',
  stalled: 'osprey-chip osprey-chip--warn',
  silent: 'osprey-chip osprey-chip--err',
  unreached: 'osprey-chip',
  unknown: 'osprey-chip',
};

/** The grouping keys a snapshot may name, with their headings. @type {Record<string, string>} */
const GROUP_LABEL = { host: 'host', class_name: 'class', group: 'group' };

/** @param {number} n */
function fmt(n) {
  return n.toLocaleString('en-US');
}

/**
 * An epoch in seconds as a UTC time to the second.
 * @param {number} epoch
 * @returns {string}
 */
export function timeText(epoch) {
  return `${new Date(epoch * 1000).toISOString().slice(0, 19).replace('T', ' ')} UTC`;
}

/**
 * How long ago `epoch` was, at `nowMs`.
 * @param {number} epoch - seconds
 * @param {number} nowMs - milliseconds
 * @returns {string}
 */
export function ageText(epoch, nowMs) {
  return durationText(nowMs / 1000 - epoch);
}

/**
 * The statement a non-`ok` answer is shown as, in the fleet tool's words.
 * None of them says anything about whether a server runs.
 * @param {FleetAnswer} answer
 * @returns {{ message: string, detail: string }}
 */
export function outcomeStatement(answer) {
  const target = answer.target ?? 'the session target';
  const { server, servers } = LABELS;
  const nothing = `This says nothing about whether any ${server} runs.`;
  const why = answer.reason ? `${answer.reason}. ` : '';
  switch (answer.outcome) {
    case 'no_snapshot':
      return {
        message: 'No snapshot yet',
        detail:
          `The server fleet service has no snapshot of target ${target} yet; no ${server} state is ` +
          `known for it until its first refresh completes. ${nothing}`,
      };
    case 'not_refreshed':
      return {
        message: 'Target not refreshed',
        detail:
          `The server fleet service does not refresh target ${target}, so no ${server} state is known ` +
          `for it. ${nothing}`,
      };
    case 'cannot_list':
      return {
        message: `Cannot list the ${servers}`,
        detail:
          `${why}The server fleet service cannot list the ${servers} of target ${target}, so no ${server} ` +
          `state is known for it. ${nothing}`,
      };
    case 'stale': {
      const taken = answer.snapshot?.generated_at ?? answer.generated_at;
      return {
        message: 'Snapshot stale',
        detail:
          `The snapshot of target ${target} is stale: the server fleet service stopped ` +
          `refreshing it` +
          (typeof taken === 'number' ? `; it was taken at ${timeText(taken)}` : '') +
          `. No state is shown for now. ${nothing}`,
      };
    }
    default:
      return {
        message: 'server fleet unavailable',
        detail:
          `${why}The server fleet service gave no usable answer for target ${target}, so no ` +
          `${server} state is known right now. ${nothing}`,
      };
  }
}

/**
 * The view's skeleton: the header, the band, the centred state's host, and
 * the table.
 * @returns {string}
 */
export function skeletonHtml() {
  return `
    <div class="servers">
      <div class="servers-header" hidden></div>
      <div class="servers-band" hidden></div>
      <div class="servers-state"></div>
      <div class="servers-list" hidden></div>
    </div>`;
}

/**
 * The toolbar's result count.
 * @param {number} shown
 * @param {number} total
 * @returns {string}
 */
export function countHtml(shown, total) {
  const servers = esc(LABELS.servers);
  if (shown === total) return `<strong>${fmt(total)}</strong> ${servers}`;
  return `<strong>${fmt(shown)}</strong> of ${fmt(total)} ${servers}`;
}

/**
 * The header: when the snapshot was taken, the read path, and each source's
 * outcome.
 * @param {FleetSnapshot} snapshot
 * @returns {string}
 */
export function headerHtml(snapshot) {
  const taken =
    `<span class="servers-taken">Snapshot of <strong>${esc(snapshot.target)}</strong> taken ` +
    `${esc(timeText(snapshot.generated_at))} · ` +
    `<span data-age="${snapshot.generated_at}"></span> ago</span>`;
  const path = snapshot.read_path ?? { status: 'unverified', channel: null };
  const pv = path.channel ? ` (${esc(path.channel)})` : '';
  const channel = esc(LABELS.channel);
  const server = esc(LABELS.server);
  /** @type {string} */
  let readPath;
  if (path.status === 'ok') {
    readPath = `<span class="osprey-chip osprey-chip--ok" data-role="read-path" title="The read-path ${channel} answered${pv}"><span class="osprey-chip-label">read path ok</span></span>`;
  } else if (path.status === 'down') {
    readPath = `<span class="osprey-chip osprey-chip--err" data-role="read-path" title="The read-path ${channel}${pv} did not answer, so every ${server} is unknown"><span class="osprey-chip-label">read path down: every ${server} unknown</span></span>`;
  } else {
    readPath = `<span class="osprey-chip osprey-chip--warn" data-role="read-path" title="No read-path ${channel} is configured for this target, so the read path is not checked"><span class="osprey-chip-label">read path unverified</span></span>`;
  }
  const sources = (snapshot.sources ?? [])
    .map((source) => {
      const tone = source.outcome === 'ok' ? 'osprey-chip--ok' : 'osprey-chip--warn';
      const title = source.reason ? ` title="${esc(source.reason)}"` : '';
      const reason = source.reason ? ` · ${esc(source.reason)}` : '';
      return (
        `<span class="osprey-chip ${tone}" data-source="${esc(source.name)}"${title}>` +
        `<span class="osprey-chip-label">${esc(source.name)}: ${esc(source.outcome)}${reason}</span></span>`
      );
    })
    .join('');
  return `${taken}${readPath}<span class="servers-sources">${sources}</span>`;
}

/**
 * The band: a tile per state and per finding that filters, the grouping
 * toggle when the snapshot names a key, and Refresh.
 * @param {FleetSnapshot} snapshot
 * @param {{ pressed: TileFilter|null, grouped: boolean }} opts
 * @returns {string}
 */
export function bandHtml(snapshot, { pressed, grouped }) {
  const byState = snapshot.counts?.by_state ?? {};
  const byFinding = snapshot.counts?.by_finding ?? {};
  /** @param {string} key @param {string} label @param {number} n */
  const tile = (key, label, n) => {
    const classes = ['servers-tile', n ? '' : 'is-zero'].filter(Boolean).join(' ');
    return (
      `<button type="button" class="${classes}" data-tile="${key}" aria-pressed="${pressed === key}">` +
      `<span class="servers-tile-n">${fmt(n)}</span><span class="servers-tile-k">${esc(label)}</span></button>`
    );
  };
  const states = STATES.map(([key, label]) =>
    tile(`state:${key}`, label, /** @type {any} */ (byState)[key] ?? 0)
  ).join('');
  const findings = FINDINGS.map(([key, label]) =>
    tile(`finding:${key}`, label, /** @type {any} */ (byFinding)[key] ?? 0)
  ).join('');
  const key = snapshot.group_by ? GROUP_LABEL[snapshot.group_by] : undefined;
  const group = key
    ? `<button type="button" class="osprey-btn" data-action="group" aria-pressed="${grouped}">` +
      `Group by ${esc(key)}</button>`
    : '';
  return (
    states +
    '<span class="servers-band-rule" aria-hidden="true"></span>' +
    findings +
    '<span class="servers-grow"></span>' +
    group +
    '<button type="button" class="osprey-icon-btn" data-action="refresh"' +
    ' aria-label="Refresh" title="Refresh"></button>'
  );
}

/**
 * The rows the filters keep, in the served order.
 * @param {FleetRow[]} rows
 * @param {{ tile: TileFilter|null, text: string }} filters
 * @returns {FleetRow[]}
 */
export function filterRows(rows, { tile, text }) {
  const words = text.trim().toLowerCase();
  const [kind, value] = tile ? tile.split(':') : [null, null];
  return rows.filter((row) => {
    if (kind === 'state' && row.state !== value) return false;
    if (kind === 'finding' && !(row.findings ?? []).includes(/** @type {any} */ (value))) {
      return false;
    }
    if (!words) return true;
    return [row.name, row.host, row.class_name, row.group].some(
      (field) => typeof field === 'string' && field.toLowerCase().includes(words)
    );
  });
}

/**
 * One row.
 * @param {FleetRow} row
 * @returns {string}
 */
function rowHtml(row) {
  const href = formatRoute({ view: 'server', params: { name: row.name } });
  const tone = STATE_TONE[row.state] ?? 'osprey-chip';
  const reason = row.reason ? ` title="${esc(row.reason)}"` : '';
  const since =
    typeof row.since === 'number' ? `<span data-age="${row.since}"></span>` : '<span>—</span>';
  const findings = (row.findings ?? [])
    .map((finding) => {
      const label = FINDINGS.find(([key]) => key === finding)?.[1] ?? finding;
      return `<span class="osprey-chip osprey-chip--warn" data-finding="${esc(finding)}"><span class="osprey-chip-label">${esc(label)}</span></span>`;
    })
    .join('');
  const directory = row.listed ? esc(row.directory_status ?? '—') : 'not listed';
  return (
    `<tr data-name="${esc(row.name)}">` +
    `<td><a href="${esc(href)}">${esc(row.name)}</a></td>` +
    `<td>${esc(row.host ?? '—')}</td>` +
    `<td><span class="${tone}" data-state="${esc(row.state)}"${reason}><span class="osprey-chip-label">${esc(row.state)}</span></span></td>` +
    `<td class="num">${since}</td>` +
    `<td class="servers-findings">${findings}</td>` +
    `<td>${directory}</td>` +
    `<td>${esc((row.sources ?? []).join(', '))}</td>` +
    '</tr>'
  );
}

/**
 * The table of `rows`: one body, or one per group under its heading in the
 * order the groups first appear.
 * @param {FleetRow[]} rows
 * @param {FleetSnapshot['group_by']|null} groupBy - the key, or null for no grouping
 * @returns {string}
 */
export function tableHtml(rows, groupBy) {
  const head =
    `<thead><tr><th>${esc(heading(LABELS.server))}</th><th>Host</th><th>State</th><th class="num">In state</th>` +
    '<th>Findings</th><th>Directory</th><th>Sources</th></tr></thead>';
  /** @type {string} */
  let bodies;
  if (groupBy) {
    /** @type {Map<string, FleetRow[]>} */
    const groups = new Map();
    for (const row of rows) {
      const key = row[groupBy] ?? '';
      const list = groups.get(key);
      if (list) list.push(row);
      else groups.set(key, [row]);
    }
    bodies = [...groups]
      .map(
        ([key, members]) =>
          `<tbody data-group="${esc(key)}"><tr class="servers-group"><th colspan="7" scope="rowgroup">` +
          `${esc(key || `no ${GROUP_LABEL[groupBy]}`)} <span class="osprey-count">${fmt(members.length)}</span></th></tr>` +
          members.map(rowHtml).join('') +
          '</tbody>'
      )
      .join('');
  } else {
    bodies = `<tbody>${rows.map(rowHtml).join('')}</tbody>`;
  }
  return `<table class="servers-table">${head}${bodies}</table>`;
}
