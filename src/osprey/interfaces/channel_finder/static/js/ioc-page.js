// @ts-check
/**
 * OSPREY Channel Finder — IOC Page
 *
 * The view of `#/ioc/<name>` (Panel view contract) over `IocFacts`: host,
 * ports, PV counts, the engineer as recorded, the IOC health PVs with their
 * current values, and whether the IOC is alive (D19). The facts carry names
 * only; every value comes from `readValues`. On the VA the engineer is the
 * ENGINEER PV, one of the health PVs, so its line shows that PV's value.
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
import { fetchIocFacts, readValues } from './directory-api.js';

/** @typedef {import('./directory-api.js').IocFacts} IocFacts */
/** @typedef {import('./directory-api.js').ValueRow} ValueRow */
/** @typedef {'yes'|'no'|'unknown'} Alive */

/** The time between the two HEARTBEAT reads. */
export const HEARTBEAT_INTERVAL_MS = 2000;

/** The suffix of the health PV whose advance means alive. */
const HEARTBEAT = 'HEARTBEAT';

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
 * The page for found facts.
 * @param {HTMLElement} container
 * @param {IocFacts} facts
 * @param {number} mounted - the generation this page belongs to
 */
function renderPage(container, facts, mounted) {
  const page = el('div', 'cf-ioc-page');

  const head = el('div', 'cf-ioc-head');
  head.append(el('h2', 'cf-ioc-name', facts.ioc));
  const chip = el('button', 'osprey-chip');
  chip.type = 'button';
  chip.dataset.role = 'alive';
  const chipLabel = el('span', 'osprey-chip-label');
  chip.append(chipLabel);
  let aliveReason = '';
  chip.addEventListener('click', () =>
    openInfoPopover(chip, { title: 'Alive', text: aliveReason })
  );
  const refresh = el('button', 'osprey-btn cf-ioc-refresh');
  refresh.type = 'button';
  refresh.append(icon('refresh'), el('span', 'osprey-btn-label', 'Refresh'));
  head.append(chip, refresh);
  page.append(head);

  const list = el('dl', 'cf-ioc-facts');
  /** @param {string} label @param {string} value @returns {HTMLElement} */
  const fact = (label, value) => {
    const dd = el('dd', '', value);
    list.append(el('dt', '', label), dd);
    return dd;
  };
  fact('Host', facts.host ?? NOT_RECORDED);
  fact('Ports', facts.ports?.length ? facts.ports.join(', ') : NOT_RECORDED);
  fact('PVs', facts.pv_count === null ? NOT_RECORDED : String(facts.pv_count));
  fact('Inactive', facts.inactive_count === null ? NOT_RECORDED : String(facts.inactive_count));
  if (facts.curated_count === null) {
    fact('Curated', facts.curated_reason ?? NOT_RECORDED);
  } else {
    const uncurated =
      facts.pv_count === null ? '' : `, ${facts.pv_count - facts.curated_count} uncurated`;
    fact('Curated', `${facts.curated_count} curated${uncurated}`);
  }
  const engineerPv = facts.engineer_source === 'ENGINEER PV' ? facts.engineer : null;
  const engineer = fact('Engineer', engineerPv ? READING : (facts.engineer ?? NOT_RECORDED));
  page.append(list);

  const health = el('section', 'cf-ioc-health');
  health.append(el('h3', 'cf-ioc-section', 'IOC health PVs'));
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
  page.append(health);

  const link = el('a', 'cf-ioc-link', 'All PVs of this IOC');
  link.href = `#/allpvs?ioc=${encodeURIComponent(facts.ioc)}`;
  page.append(link);

  container.replaceChildren(page);

  const heartbeat = facts.health_pvs.find((name) => name.endsWith(HEARTBEAT)) ?? null;
  const names = [...facts.health_pvs];
  if (engineerPv && !names.includes(engineerPv)) names.push(engineerPv);

  /** @param {{ alive: Alive, reason: string }} answer */
  const showAlive = ({ alive, reason }) => {
    chip.className = `osprey-chip osprey-chip--${ALIVE_TONE[alive]}`;
    chip.dataset.alive = alive;
    chipLabel.textContent = `alive: ${alive}`;
    aliveReason = reason;
  };

  /** @param {Map<string, ValueRow>|null} rows */
  const showValues = (rows) => {
    for (const [name, cell] of cells) {
      cell.textContent = rows ? valueText(rows.get(name)) : 'values not read';
    }
    if (engineerPv)
      engineer.textContent = rows ? valueText(rows.get(engineerPv)) : 'value not read';
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
      showAlive(aliveness(heartbeat, first, second));
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
