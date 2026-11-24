// @ts-check
/**
 * OSPREY Channel Finder — IOC Page
 *
 * The view of `#/server/<name>` (Panel view contract) over `ServerFacts`:
 *
 *   All PVs / IOC
 *   [IOC  alive: yes]  Host … · engineer …        [Refresh] [All PVs of this IOC]
 *   PVs · Curated · In alarm · Uptime · Heartbeat              (tiles)
 *   Alarms on this IOC                  | IOC health PVs · Network
 *
 * The facts carry names only; every value comes from `readValues`. Each health
 * PV carries its role (heartbeat, uptime, engineer); the page reads the role,
 * never the end of the name. On the VA the engineer is the engineer PV, one of
 * the health PVs, so its line shows that PV's value. In alarm and the alarms card read `facts.alarms`, present
 * only when an alarm feed is configured.
 *
 * Aliveness: the heartbeat PV is read with the other health PVs when the page
 * is shown (or on Refresh) and read again 2 s later; a value that advanced is
 * `yes`, an equal value or a read without an answer is `no`, and a page with
 * no second read yet or no heartbeat PV is `unknown`.
 *
 * With the server fleet on, the facts carry a `fleet` entry and the page shows
 * it in place of that check: the row's state, findings and read time, or the
 * entry's outcome as a statement (the words of the `server_fleet` tool). The
 * heartbeat PV is then never read here; the other health PVs are. Without the
 * `fleet` field (the fleet is off) the page reads the heartbeat as above.
 *
 * The page is part of All PVs: the toolbar search opens All PVs
 * (panel-toolbar.js).
 */

import { icon } from '/design-system/js/icons.js';
import { openInfoPopover } from '/design-system/js/info-popover.js';
import { createLatestRequest } from '/design-system/js/latest-request.js';
import { renderState } from '/design-system/js/panel-state.js';
import { durationText, sinceText } from './duration.js';
import { fetchServerFacts, readValues } from './directory-api.js';
import { timeText } from './channel-page-sections.js';
import { LABELS, heading } from './labels.js';
import { formatRoute } from './route.js';
import { severityMark, severityOf } from './severity.js';

/** @typedef {import('./directory-api.js').ServerFacts} ServerFacts */
/** @typedef {import('./directory-api.js').ValueRow} ValueRow */
/** @typedef {import('./directory-api.js').ServerAlarms} ServerAlarms */
/** @typedef {'yes'|'no'|'unknown'} Alive */
/**
 * One IOC's row in a fleet snapshot, as far as this page reads it.
 * @typedef {{ state: string, reason?: string|null, findings?: string[],
 *   liveness?: { pv?: string|null, read_at?: number|null }|null }} FleetRow
 */
/**
 * The `fleet` field of IOC facts: the row on a fresh snapshot, else an outcome
 * with a reason. Times are epoch seconds.
 * @typedef {{ outcome: string, target: string, reason?: string|null,
 *   generated_at?: number|null, row?: FleetRow|null }} FleetEntry
 */

/** The time between the two heartbeat reads. */
export const HEARTBEAT_INTERVAL_MS = 2000;

/** The role of the health PV whose advance means alive. */
const HEARTBEAT = 'heartbeat';
/** The role of the health PV that counts the IOC's uptime. */
const UPTIME = 'uptime';

/** The Heartbeat tile's words for each aliveness answer. */
const HEARTBEAT_WORDS = Object.freeze({
  yes: 'advancing',
  no: 'not advancing',
  unknown: 'unknown',
});

/** The chip tone of each aliveness answer. */
const ALIVE_TONE = Object.freeze({ yes: 'ok', no: 'err', unknown: 'info' });

/** The chip tone of each fleet state; unknown and unreached are not established, never down. */
const STATE_TONE = Object.freeze({
  alive: 'ok',
  stalled: 'err',
  silent: 'err',
  unreached: 'info',
  unknown: 'info',
});

/** Each fleet finding in words. */
const FINDING_WORDS = Object.freeze({
  degraded: 'degraded',
  expected_running_not_alive: 'expected running, not alive',
  expected_stopped_alive: 'expected stopped, alive',
  alive_unexpected: 'alive, not expected',
});

/** The State tile's word for each fleet outcome without a row. */
const OUTCOME_WORDS = Object.freeze({
  no_snapshot: 'no snapshot',
  not_refreshed: 'not refreshed',
  cannot_list: 'cannot list',
  unavailable: 'unavailable',
  stale: 'stale',
  not_in_snapshot: 'not in snapshot',
});

/** What the health table says for the heartbeat PV it does not read. */
export const FLEET_READS = 'read by the server fleet service';

const NOT_RECORDED = 'not recorded';
const NO_ANSWER = 'no answer';
const READING = '…';

/** The mounts: a remount or an unmount retires this one, so an answer for an old page is dropped. */
const mounts = createLatestRequest();
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
 * The aliveness answer from the two reads of the heartbeat PV `name`.
 * @param {string|null} name - the heartbeat PV, or null when none is listed
 * @param {ValueRow|undefined} first
 * @param {ValueRow|undefined} second
 * @returns {{ alive: Alive, reason: string }}
 */
export function aliveness(name, first, second) {
  if (!name) {
    return {
      alive: 'unknown',
      reason: `no ${HEARTBEAT} ${LABELS.channel} listed for this ${LABELS.server}`,
    };
  }
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
 * An epoch-seconds time as the panel writes times.
 * @param {number|null|undefined} epoch
 * @returns {string}
 */
function epochText(epoch) {
  if (typeof epoch !== 'number' || !Number.isFinite(epoch)) return '';
  return timeText(new Date(epoch * 1000).toISOString());
}

/**
 * What a fleet entry without a row says, as a statement about the snapshot and
 * never about the IOC (the words of the `server_fleet` tool).
 * @param {FleetEntry} fleet
 * @param {string} ioc
 * @returns {string}
 */
export function fleetStatement(fleet, ioc) {
  const target = fleet.target;
  const taken = epochText(fleet.generated_at);
  /** @type {Record<string, string>} */
  const says = {
    no_snapshot: `The server fleet service has no snapshot of target ${target} yet.`,
    not_refreshed: `The server fleet service does not refresh target ${target}.`,
    cannot_list: `The server fleet service cannot list the ${LABELS.servers} of target ${target}.`,
    unavailable: `The server fleet service gave no usable answer for target ${target}.`,
    stale:
      `The snapshot of target ${target} is stale: the server fleet service stopped refreshing it.` +
      (taken ? ` Last snapshot ${taken}.` : ''),
    not_in_snapshot: `${heading(LABELS.server)} ${ioc} is not in the snapshot of target ${target}.`,
  };
  const head = says[fleet.outcome] ?? `The server fleet answered ${fleet.outcome}.`;
  const server = LABELS.server;
  return `${head} No state is known for this ${server}. This says nothing about whether this ${server} runs.`;
}

/**
 * The fleet's answer for this IOC as the chip and the State tile show it.
 * @param {FleetEntry} fleet
 * @param {string} ioc
 * @returns {{ tone: string, label: string, value: string, sub: string, reason: string,
 *   state: string|null }}
 */
export function fleetView(fleet, ioc) {
  const row = fleet.row;
  if (fleet.outcome === 'ok' && row) {
    const findings = (row.findings ?? []).map(
      (f) => FINDING_WORDS[/** @type {keyof typeof FINDING_WORDS} */ (f)] ?? f
    );
    const time = epochText(row.liveness?.read_at ?? fleet.generated_at);
    const sub = [...findings, time ? `read ${time}` : ''].filter(Boolean).join(' · ');
    return {
      tone: STATE_TONE[/** @type {keyof typeof STATE_TONE} */ (row.state)] ?? 'info',
      label: `state: ${row.state}`,
      value: row.state,
      sub,
      reason: [row.reason, sub].filter(Boolean).join(' · '),
      state: row.state,
    };
  }
  const statement = fleetStatement(fleet, ioc);
  return {
    tone: 'info',
    label: 'state: not known',
    value:
      OUTCOME_WORDS[/** @type {keyof typeof OUTCOME_WORDS} */ (fleet.outcome)] ?? fleet.outcome,
    sub: statement,
    reason: [statement, fleet.reason].filter(Boolean).join(' '),
    state: null,
  };
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
  const box = el('div', 'osprey-card cf-server-tile');
  box.dataset.tile = role;
  const value = el('div', 'cf-server-tile-value');
  const sub = el('div', 'cf-server-tile-sub');
  box.append(el('div', 'cf-server-tile-label', label), value, sub);
  return { tile: box, value, sub };
}

/**
 * A card with its title.
 * @param {string} role
 * @param {string} title
 * @returns {HTMLElement}
 */
function card(role, title) {
  const box = el('section', 'osprey-card cf-server-card');
  box.dataset.card = role;
  box.append(el('h3', 'cf-server-card-title', title));
  return box;
}

/**
 * The In alarm tile, from the IOC's alarms: the major line reads the counts of
 * the whole IOC, and is absent without them.
 * @param {ServerAlarms} alarms
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
  const major = alarms.counts ? (alarms.counts.MAJOR ?? 0) : null;
  sub.textContent = !total ? 'none' : major === null ? '' : `${major} major`;
  if (major) box.classList.add('is-hot');
  return box;
}

/**
 * The Alarms on this IOC card: severity, PV, status and how long, or why
 * there is no list.
 * @param {ServerAlarms} alarms
 * @returns {HTMLElement}
 */
function alarmsCard(alarms) {
  const box = card('alarms', `Alarms on this ${LABELS.server}`);
  if (alarms.outcome === 'unavailable') {
    box.append(el('p', 'cf-server-reason', alarms.reason ?? 'unavailable'));
    return box;
  }
  const rows = alarms.rows ?? [];
  if (!rows.length) {
    box.append(el('p', 'cf-server-reason', 'No active alarms.'));
    return box;
  }
  const table = el('table', 'cf-server-alarms');
  const body = el('tbody');
  const now = Date.now();
  for (const row of rows) {
    const tr = el('tr');
    const severity = el('td');
    severity.innerHTML = severityMark(severityOf(row.severity));
    const pv = el('td', 'cf-server-channel');
    const link = el('a', '', row.channel);
    link.href = formatRoute({ view: 'channel', params: { name: row.channel } });
    pv.append(link);
    const since = el('td', 'cf-server-since', sinceText(row.time, now));
    since.title = timeText(row.time);
    tr.append(severity, pv, el('td', 'cf-server-status', row.status ?? ''), since);
    body.append(tr);
  }
  table.append(body);
  box.append(table);
  if ((alarms.total ?? 0) > rows.length) {
    box.append(el('p', 'cf-server-reason', `First ${rows.length} of ${alarms.total} shown`));
  }
  return box;
}

/**
 * The page for found facts.
 * @param {HTMLElement} container
 * @param {ServerFacts} facts
 * @param {import('/design-system/js/latest-request.js').LatestToken} mounted - the mount this page belongs to
 */
function renderPage(container, facts, mounted) {
  const page = el('div', 'cf-server-page');

  const crumb = el('nav', 'cf-server-crumb');
  crumb.setAttribute('aria-label', 'Breadcrumb');
  const all = el('a', '', LABELS.directory);
  all.href = formatRoute({ view: 'channels', params: {} });
  crumb.append(all, el('span', '', '/'), el('span', 'cf-server-crumb-leaf', facts.server));
  page.append(crumb);

  const head = el('div', 'cf-server-head');
  const title = el('div', 'cf-server-title');
  title.append(el('h2', 'cf-server-name', facts.server));
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
  const engineerChannel = facts.engineer_source === 'ENGINEER PV' ? facts.engineer : null;
  const engineer = el(
    'b',
    'cf-server-engineer',
    engineerChannel ? READING : (facts.engineer ?? NOT_RECORDED)
  );
  const line = el('p', 'cf-server-sub');
  line.append(`Host ${facts.host ?? NOT_RECORDED} · engineer `, engineer);
  const refresh = el('button', 'osprey-btn cf-server-refresh');
  refresh.type = 'button';
  refresh.append(icon('refresh'), el('span', 'osprey-btn-label', 'Refresh'));
  const link = el(
    'a',
    'osprey-btn primary cf-server-link',
    `${LABELS.directory} of this ${LABELS.server}`
  );
  link.href = formatRoute({ view: 'channels', params: { server: [facts.server] } });
  const actions = el('div', 'cf-server-actions');
  actions.append(refresh, link);
  head.append(title, actions, line);
  page.append(head);

  const tiles = el('div', 'cf-server-tiles');
  const pvs = tile('channels', heading(LABELS.channels));
  pvs.value.textContent = facts.channel_count === null ? NOT_RECORDED : String(facts.channel_count);
  if (facts.inactive_count !== null) pvs.sub.textContent = `${facts.inactive_count} inactive`;
  const curated = tile('curated', 'Curated');
  if (facts.curated_count === null) {
    curated.value.textContent = '—';
    curated.sub.textContent = facts.curated_reason ?? NOT_RECORDED;
  } else {
    curated.value.textContent = String(facts.curated_count);
    if (facts.channel_count !== null) {
      curated.sub.textContent = `${facts.channel_count - facts.curated_count} control system only`;
    }
  }
  tiles.append(pvs.tile, curated.tile);
  if (facts.alarms) tiles.append(inAlarmTile(facts.alarms));
  const uptimeChannel = facts.health_channels.find((pv) => pv.role === UPTIME)?.name ?? null;
  const uptime = uptimeChannel ? tile('uptime', 'Uptime') : null;
  if (uptime) {
    uptime.value.textContent = READING;
    tiles.append(uptime.tile);
  }
  /** @type {FleetEntry|undefined} */
  const fleet = /** @type {{ fleet?: FleetEntry }} */ (facts).fleet ?? undefined;
  const beat = fleet ? tile('state', 'State') : tile('heartbeat', 'Heartbeat');
  tiles.append(beat.tile);
  page.append(tiles);

  const grid = el('div', 'cf-server-grid');
  const left = el('div', 'cf-server-col');
  const right = el('div', 'cf-server-col');
  if (facts.alarms) left.append(alarmsCard(facts.alarms));

  const health = card('health', `${heading(LABELS.server)} health ${LABELS.channels}`);
  /** @type {Map<string, HTMLElement>} */
  const cells = new Map();
  if (facts.health_channels.length) {
    const table = el('table', 'cf-server-health-table');
    const body = el('tbody');
    for (const { name } of facts.health_channels) {
      const tr = el('tr');
      const value = el('td', 'cf-server-value', READING);
      tr.append(el('td', 'cf-server-channel', name), value);
      cells.set(name, value);
      body.append(tr);
    }
    table.append(body);
    health.append(table);
  } else {
    health.append(
      el(
        'p',
        'cf-server-reason',
        facts.health_channels_reason ??
          `no ${LABELS.server} health ${LABELS.channels} listed for this ${LABELS.server}`
      )
    );
  }
  right.append(health);

  const network = card('network', 'Network');
  const list = el('dl', 'cf-server-facts');
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

  const heartbeatChannel = facts.health_channels.find((pv) => pv.role === HEARTBEAT)?.name ?? null;
  // With the fleet on, the fleet service reads the heartbeat; this page does not.
  const heartbeat = fleet ? null : heartbeatChannel;
  const names = facts.health_channels
    .map((pv) => pv.name)
    .filter((n) => !fleet || n !== heartbeatChannel);
  const fleetCell = fleet && heartbeatChannel ? cells.get(heartbeatChannel) : undefined;
  if (fleetCell) {
    cells.delete(/** @type {string} */ (heartbeatChannel));
    fleetCell.textContent = FLEET_READS;
  }
  if (engineerChannel && !names.includes(engineerChannel)) names.push(engineerChannel);

  /**
   * @param {{ alive: Alive, reason: string }} answer
   * @param {(ValueRow|undefined)[]} [reads] - the two heartbeat reads
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

  if (fleet) {
    const shown = fleetView(fleet, facts.server);
    chip.className = `osprey-chip osprey-chip--${shown.tone}`;
    if (shown.state) chip.dataset.state = shown.state;
    else chip.dataset.outcome = fleet.outcome;
    chipLabel.textContent = shown.label;
    aliveReason = shown.reason;
    beat.value.textContent = shown.value;
    beat.sub.textContent = shown.sub;
    beat.tile.dataset.outcome = fleet.outcome;
  }

  /** @param {Map<string, ValueRow>|null} rows */
  const showValues = (rows) => {
    for (const [name, cell] of cells) {
      cell.textContent = rows ? valueText(rows.get(name)) : 'values not read';
    }
    if (engineerChannel)
      engineer.textContent = rows ? valueText(rows.get(engineerChannel)) : 'value not read';
    if (uptime && uptimeChannel) {
      const row = rows?.get(uptimeChannel);
      uptime.value.textContent = rows ? uptimeText(row) : 'value not read';
      uptime.sub.textContent =
        row?.outcome === 'found' ? [valueText(row), row.units].filter(Boolean).join(' ') : '';
    }
  };

  /** The reads: a newer read retires this one, so an answer for an earlier read is dropped. */
  const reads = createLatestRequest();
  const read = async () => {
    const token = reads.begin();
    const stale = () => !mounted.isCurrent() || !token.isCurrent();
    if (heartbeatTimer) clearTimeout(heartbeatTimer);
    heartbeatTimer = null;
    if (!fleet) showAlive(aliveness(heartbeat, undefined, undefined));
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
export function mountServerPage(container, params) {
  const mounted = mounts.begin();
  const name = params.name ?? '';

  renderState(container, { icon: 'refresh', message: `Loading ${LABELS.server} ${name}` });
  fetchServerFacts(name).then(
    (facts) => {
      if (!mounted.isCurrent()) return;
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
          message: `${heading(LABELS.server)} ${name} is not listed in the directory`,
        });
      }
    },
    (error) => {
      if (!mounted.isCurrent()) return;
      renderState(container, {
        icon: 'x',
        message: `The ${LABELS.server} facts could not be read`,
        detail: error instanceof Error ? error.message : String(error),
        action: {
          label: 'Retry',
          onClick: () => {
            unmountServerPage();
            mountServerPage(container, params);
          },
        },
      });
    }
  );
}

/** @returns {void} */
export function unmountServerPage() {
  mounts.cancel();
  if (heartbeatTimer) clearTimeout(heartbeatTimer);
  heartbeatTimer = null;
}
