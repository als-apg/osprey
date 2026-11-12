// @ts-check
/**
 * OSPREY Channel Finder — IOC Page
 *
 * The view of `#/ioc/<name>` (Panel view contract) over `IocFacts`:
 *
 *   All PVs / IOC
 *   [IOC  alive: yes]  Host … · engineer …        [Refresh] [All PVs of this IOC]
 *   PVs · Curated · In alarm · Uptime · Heartbeat              (tiles)
 *   Alarms on this IOC                  | IOC health PVs · Network
 *
 * The facts carry names only; every value comes from `readValues`. On the VA
 * the engineer is the ENGINEER PV, one of the health PVs, so its line shows
 * that PV's value. In alarm and the alarms card read `facts.alarms`, present
 * only when an alarm feed is configured.
 *
 * Aliveness: the HEARTBEAT PV is read with the other health PVs when the page
 * is shown (or on Refresh) and read again 2 s later; a value that advanced is
 * `yes`, an equal value or a read without an answer is `no`, and a page with
 * no second read yet or no HEARTBEAT PV is `unknown`.
 *
 * The page is part of All PVs: the toolbar search opens All PVs
 * (panel-toolbar.js).
 */

import { icon } from '/design-system/js/icons.js';
import { openInfoPopover } from '/design-system/js/info-popover.js';
import { renderState } from '/design-system/js/panel-state.js';
import { sinceText } from './alarms-render.js';
import { fetchIocFacts, readValues } from './directory-api.js';
import { timeText } from './pv-page-sections.js';
import { formatRoute } from './route.js';
import { severityMark, severityOf } from './severity.js';

/** @typedef {import('./directory-api.js').IocFacts} IocFacts */
/** @typedef {import('./directory-api.js').ValueRow} ValueRow */
/** @typedef {import('./directory-api.js').IocAlarms} IocAlarms */
/** @typedef {'yes'|'no'|'unknown'} Alive */

/** The time between the two HEARTBEAT reads. */
export const HEARTBEAT_INTERVAL_MS = 2000;

/** The suffix of the health PV whose advance means alive. */
const HEARTBEAT = 'HEARTBEAT';
/** The suffix of the health PV that counts the IOC's uptime. */
const UPTIME = 'UPTIME';

/** The Heartbeat tile's words for each aliveness answer. */
const HEARTBEAT_WORDS = Object.freeze({
  yes: 'advancing',
  no: 'not advancing',
  unknown: 'unknown',
});

/** The chip tone of each aliveness answer. */
const ALIVE_TONE = Object.freeze({ yes: 'ok', no: 'err', unknown: 'info' });

const NOT_RECORDED = 'not recorded';
const NO_ANSWER = 'no answer';
const READING = '…';

/** Bumped on every mount and unmount, so an answer for an old page is dropped. */
let generation = 0;
/** @type {ReturnType<typeof setTimeout>|null} */
let heartbeatTimer = null;

/**
 * @template {keyof HTMLElementTagNameMap} K
 * @param {K} tag
 * @param {string} [className]
 * @param {string} [text]
 * @returns {HTMLElementTagNameMap[K]}
 */
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

/**
 * What a value row says: its value, "no answer", or its reason.
 * @param {ValueRow|undefined} row
 * @returns {string}
 */
function valueText(row) {
  if (!row) return READING;
  if (row.outcome === 'found') return row.value === null ? '' : String(row.value);
  if (row.outcome === 'no_answer') return NO_ANSWER;
  return row.reason ?? 'unavailable';
}

/**
 * The aliveness answer from the two HEARTBEAT reads of `name`.
 * @param {string|null} name - the HEARTBEAT PV, or null when none is listed
 * @param {ValueRow|undefined} first
 * @param {ValueRow|undefined} second
 * @returns {{ alive: Alive, reason: string }}
 */
export function aliveness(name, first, second) {
  if (!name) return { alive: 'unknown', reason: `no ${HEARTBEAT} PV listed for this IOC` };
  if (!first || !second) return { alive: 'unknown', reason: `${name}: second read pending` };
  for (const row of [first, second]) {
    if (row.outcome !== 'found') return { alive: 'no', reason: `${name}: ${valueText(row)}` };
  }
  const before = String(first.value);
  const after = String(second.value);
  const seconds = HEARTBEAT_INTERVAL_MS / 1000;
  if (before === after) {
    return { alive: 'no', reason: `${name} unchanged at ${after} over ${seconds} s` };
  }
  return { alive: 'yes', reason: `${name} advanced from ${before} to ${after} in ${seconds} s` };
}

/**
 * Read `names`; a request that fails answers null (unread).
 * @param {string[]} names
 * @returns {Promise<Map<string, ValueRow>|null>}
 */
async function readRows(names) {
  try {
    const rows = await readValues(names);
    return new Map(rows.map((row) => [row.name, row]));
  } catch {
    return null;
  }
}

/**
 * A number of seconds as a duration, in the units the Alarms view counts in.
 * @param {number} seconds
 * @returns {string}
 */
function durationText(seconds) {
  return sinceText(new Date(0).toISOString(), seconds * 1000);
}

/**
 * The Uptime tile's value: a duration for a number in seconds (unit `s`, or
 * none as on the simulator), else the value as read.
 * @param {ValueRow|undefined} row
 * @returns {string}
 */
export function uptimeText(row) {
  if (row?.outcome !== 'found') return valueText(row);
  const unit = row.units ?? '';
  if (typeof row.value === 'number' && Number.isFinite(row.value) && (unit === 's' || !unit)) {
    return durationText(row.value);
  }
  return valueText(row);
}

/**
 * A tile: its label, its value and the line under it.
 * @param {string} role
 * @param {string} label
 * @returns {{ tile: HTMLElement, value: HTMLElement, sub: HTMLElement }}
 */
function tile(role, label) {
  const box = el('div', 'osprey-card cf-ioc-tile');
  box.dataset.tile = role;
  const value = el('div', 'cf-ioc-tile-value');
  const sub = el('div', 'cf-ioc-tile-sub');
  box.append(el('div', 'cf-ioc-tile-label', label), value, sub);
  return { tile: box, value, sub };
}

/**
 * A card with its title.
 * @param {string} role
 * @param {string} title
 * @returns {HTMLElement}
 */
function card(role, title) {
  const box = el('section', 'osprey-card cf-ioc-card');
  box.dataset.card = role;
  box.append(el('h3', 'cf-ioc-card-title', title));
  return box;
}

/**
 * The In alarm tile, from the IOC's alarms.
 * @param {IocAlarms} alarms
 * @returns {HTMLElement}
 */
function inAlarmTile(alarms) {
  const { tile: box, value, sub } = tile('alarm', 'In alarm');
  if (alarms.outcome === 'unavailable') {
    value.textContent = '—';
    sub.textContent = alarms.reason ?? 'unavailable';
    return box;
  }
  const total = alarms.total ?? alarms.rows?.length ?? 0;
  value.textContent = String(total);
  const major = (alarms.rows ?? []).filter((row) => severityOf(row.severity) === 'MAJOR').length;
  sub.textContent = total ? `${major} major` : 'none';
  if (major) box.classList.add('is-hot');
  return box;
}

/**
 * The Alarms on this IOC card: severity, PV, status and how long, or why
 * there is no list.
 * @param {IocAlarms} alarms
 * @returns {HTMLElement}
 */
function alarmsCard(alarms) {
  const box = card('alarms', 'Alarms on this IOC');
  if (alarms.outcome === 'unavailable') {
    box.append(el('p', 'cf-ioc-reason', alarms.reason ?? 'unavailable'));
    return box;
  }
  const rows = alarms.rows ?? [];
  if (!rows.length) {
    box.append(el('p', 'cf-ioc-reason', 'No active alarms.'));
    return box;
  }
  const table = el('table', 'cf-ioc-alarms');
  const body = el('tbody');
  const now = Date.now();
  for (const row of rows) {
    const tr = el('tr');
    const severity = el('td');
    severity.innerHTML = severityMark(severityOf(row.severity));
    const pv = el('td', 'cf-ioc-pv');
    const link = el('a', '', row.pv);
    link.href = formatRoute({ view: 'pv', params: { name: row.pv } });
    pv.append(link);
    const since = el('td', 'cf-ioc-since', sinceText(row.time, now));
    since.title = timeText(row.time);
    tr.append(severity, pv, el('td', 'cf-ioc-status', row.status ?? ''), since);
    body.append(tr);
  }
  table.append(body);
  box.append(table);
  if ((alarms.total ?? 0) > rows.length) {
    box.append(el('p', 'cf-ioc-reason', `First ${rows.length} of ${alarms.total} shown`));
  }
  return box;
}

/**
 * The page for found facts.
 * @param {HTMLElement} container
 * @param {IocFacts} facts
 * @param {number} mounted - the generation this page belongs to
 */
function renderPage(container, facts, mounted) {
  const page = el('div', 'cf-ioc-page');

  const crumb = el('nav', 'cf-ioc-crumb');
  crumb.setAttribute('aria-label', 'Breadcrumb');
  const all = el('a', '', 'All PVs');
  all.href = formatRoute({ view: 'allpvs', params: {} });
  crumb.append(all, el('span', '', '/'), el('span', 'cf-ioc-crumb-leaf', facts.ioc));
  page.append(crumb);

  const head = el('div', 'cf-ioc-head');
  const title = el('div', 'cf-ioc-title');
  title.append(el('h2', 'cf-ioc-name', facts.ioc));
  const chip = el('button', 'osprey-chip');
  chip.type = 'button';
  chip.dataset.role = 'alive';
  const chipLabel = el('span', 'osprey-chip-label');
  chip.append(chipLabel);
  let aliveReason = '';
  chip.addEventListener('click', () =>
    openInfoPopover(chip, { title: 'Alive', text: aliveReason })
  );
  title.append(chip);
  const engineerPv = facts.engineer_source === 'ENGINEER PV' ? facts.engineer : null;
  const engineer = el(
    'b',
    'cf-ioc-engineer',
    engineerPv ? READING : (facts.engineer ?? NOT_RECORDED)
  );
  const line = el('p', 'cf-ioc-sub');
  line.append(`Host ${facts.host ?? NOT_RECORDED} · engineer `, engineer);
  const refresh = el('button', 'osprey-btn cf-ioc-refresh');
  refresh.type = 'button';
  refresh.append(icon('refresh'), el('span', 'osprey-btn-label', 'Refresh'));
  const link = el('a', 'osprey-btn primary cf-ioc-link', 'All PVs of this IOC');
  link.href = formatRoute({ view: 'allpvs', params: { ioc: [facts.ioc] } });
  const actions = el('div', 'cf-ioc-actions');
  actions.append(refresh, link);
  head.append(title, actions, line);
  page.append(head);

  const tiles = el('div', 'cf-ioc-tiles');
  const pvs = tile('pvs', 'PVs');
  pvs.value.textContent = facts.pv_count === null ? NOT_RECORDED : String(facts.pv_count);
  if (facts.inactive_count !== null) pvs.sub.textContent = `${facts.inactive_count} inactive`;
  const curated = tile('curated', 'Curated');
  if (facts.curated_count === null) {
    curated.value.textContent = '—';
    curated.sub.textContent = facts.curated_reason ?? NOT_RECORDED;
  } else {
    curated.value.textContent = String(facts.curated_count);
    if (facts.pv_count !== null) {
      curated.sub.textContent = `${facts.pv_count - facts.curated_count} control system only`;
    }
  }
  tiles.append(pvs.tile, curated.tile);
  if (facts.alarms) tiles.append(inAlarmTile(facts.alarms));
  const uptimePv = facts.health_pvs.find((name) => name.endsWith(UPTIME)) ?? null;
  const uptime = uptimePv ? tile('uptime', 'Uptime') : null;
  if (uptime) {
    uptime.value.textContent = READING;
    tiles.append(uptime.tile);
  }
  const beat = tile('heartbeat', 'Heartbeat');
  tiles.append(beat.tile);
  page.append(tiles);

  const grid = el('div', 'cf-ioc-grid');
  const left = el('div', 'cf-ioc-col');
  const right = el('div', 'cf-ioc-col');
  if (facts.alarms) left.append(alarmsCard(facts.alarms));

  const health = card('health', 'IOC health PVs');
  /** @type {Map<string, HTMLElement>} */
  const cells = new Map();
  if (facts.health_pvs.length) {
    const table = el('table', 'cf-ioc-health-table');
    const body = el('tbody');
    for (const name of facts.health_pvs) {
      const tr = el('tr');
      const value = el('td', 'cf-ioc-value', READING);
      tr.append(el('td', 'cf-ioc-pv', name), value);
      cells.set(name, value);
      body.append(tr);
    }
    table.append(body);
    health.append(table);
  } else {
    health.append(
      el('p', 'cf-ioc-reason', facts.health_pvs_reason ?? 'no IOC health PVs listed for this IOC')
    );
  }
  right.append(health);

  const network = card('network', 'Network');
  const list = el('dl', 'cf-ioc-facts');
  list.append(
    el('dt', '', 'Host'),
    el('dd', '', facts.host ?? NOT_RECORDED),
    el('dt', '', 'Ports'),
    el('dd', '', facts.ports?.length ? facts.ports.join(', ') : NOT_RECORDED)
  );
  network.append(list);
  right.append(network);
  grid.append(left, right);
  page.append(grid);

  container.replaceChildren(page);

  const heartbeat = facts.health_pvs.find((name) => name.endsWith(HEARTBEAT)) ?? null;
  const names = [...facts.health_pvs];
  if (engineerPv && !names.includes(engineerPv)) names.push(engineerPv);

  /**
   * @param {{ alive: Alive, reason: string }} answer
   * @param {(ValueRow|undefined)[]} [reads] - the two HEARTBEAT reads
   */
  const showAlive = ({ alive, reason }, reads = []) => {
    chip.className = `osprey-chip osprey-chip--${ALIVE_TONE[alive]}`;
    chip.dataset.alive = alive;
    chipLabel.textContent = `alive: ${alive}`;
    aliveReason = reason;
    beat.value.textContent = HEARTBEAT_WORDS[alive];
    beat.tile.dataset.alive = alive;
    const [first, second] = reads;
    beat.sub.textContent =
      first?.outcome === 'found' && second?.outcome === 'found'
        ? `${valueText(first)} → ${valueText(second)} in ${HEARTBEAT_INTERVAL_MS / 1000} s`
        : reason;
  };

  /** @param {Map<string, ValueRow>|null} rows */
  const showValues = (rows) => {
    for (const [name, cell] of cells) {
      cell.textContent = rows ? valueText(rows.get(name)) : 'values not read';
    }
    if (engineerPv)
      engineer.textContent = rows ? valueText(rows.get(engineerPv)) : 'value not read';
    if (uptime && uptimePv) {
      const row = rows?.get(uptimePv);
      uptime.value.textContent = rows ? uptimeText(row) : 'value not read';
      uptime.sub.textContent =
        row?.outcome === 'found' ? [valueText(row), row.units].filter(Boolean).join(' ') : '';
    }
  };

  /** Bumped on every read, so an answer for an earlier read is dropped. */
  let reads = 0;
  const read = async () => {
    const token = ++reads;
    const stale = () => mounted !== generation || token !== reads;
    if (heartbeatTimer) clearTimeout(heartbeatTimer);
    heartbeatTimer = null;
    showAlive(aliveness(heartbeat, undefined, undefined));
    if (!names.length) return;
    const rows = await readRows(names);
    if (stale()) return;
    showValues(rows);
    if (!heartbeat) return;
    if (!rows) {
      showAlive({ alive: 'unknown', reason: `${heartbeat}: values not read` });
      return;
    }
    const first = rows.get(heartbeat);
    heartbeatTimer = setTimeout(async () => {
      heartbeatTimer = null;
      const again = await readRows([heartbeat]);
      if (stale()) return;
      if (!again) {
        showAlive({ alive: 'unknown', reason: `${heartbeat}: second read not answered` });
        return;
      }
      const second = again.get(heartbeat);
      showAlive(aliveness(heartbeat, first, second), [first, second]);
      const cell = cells.get(heartbeat);
      if (cell && second) cell.textContent = valueText(second);
    }, HEARTBEAT_INTERVAL_MS);
  };

  refresh.addEventListener('click', () => void read());
  void read();
}

/**
 * @param {HTMLElement} container
 * @param {{ name?: string }} params
 * @returns {void}
 */
export function mountIocPage(container, params) {
  const mounted = ++generation;
  const name = params.name ?? '';

  renderState(container, { icon: 'refresh', message: `Loading IOC ${name}` });
  fetchIocFacts(name).then(
    (facts) => {
      if (mounted !== generation) return;
      if (facts.outcome === 'found' || facts.outcome === 'inactive') {
        renderPage(container, facts, mounted);
      } else if (facts.outcome === 'unavailable') {
        renderState(container, {
          icon: 'x',
          message: 'The directory did not answer',
          detail: facts.reason,
        });
      } else {
        renderState(container, {
          icon: 'search',
          message: `IOC ${name} is not listed in the directory`,
        });
      }
    },
    (error) => {
      if (mounted !== generation) return;
      renderState(container, {
        icon: 'x',
        message: 'The IOC facts could not be read',
        detail: error instanceof Error ? error.message : String(error),
        action: {
          label: 'Retry',
          onClick: () => {
            unmountIocPage();
            mountIocPage(container, params);
          },
        },
      });
    }
  );
}

/** @returns {void} */
export function unmountIocPage() {
  generation += 1;
  if (heartbeatTimer) clearTimeout(heartbeatTimer);
  heartbeatTimer = null;
}
