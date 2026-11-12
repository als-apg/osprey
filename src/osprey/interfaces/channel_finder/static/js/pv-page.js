// @ts-check
/**
 * OSPREY Channel Finder — PV Page
 *
 * The view of `#/pv/<name>` (Panel view contract): one card per connector
 * answer about one PV.
 *
 *   All PVs / IOC / NAME
 *   [◆ NAME  Active  ⚠ alarm feed stale]   [⧉] [Send to assistant] [Plot]
 *   [alarm banner, only when the alarm server reports the PV not OK]
 *   Value · Alarm · Put history          | Directory · Logbook
 *
 * A card whose service the deployment does not configure (the descriptor's
 * `configured_services`) is one dashed line; a configured card that has no
 * answer says why. The page never reads a missing answer as a clear state.
 * The logbook is read by this page,
 * through ARIEL's own search at its mount, with the operator's session: the
 * panel's backend holds no logbook credential. The toolbar search opens All
 * PVs (panel-toolbar.js).
 */

import { copyText } from '/design-system/js/clipboard.js';
import { panelApiPrefix } from '/design-system/js/dom.js';
import { icon } from '/design-system/js/icons.js';
import { renderState } from '/design-system/js/panel-state.js';
import { showToast } from '/design-system/js/toast.js';
import {
  createPlotCheck,
  fetchLogbookAvailability,
  fetchPvFacts,
  handoffAllowed,
  panelUrl,
  plotUrl,
  readValues,
} from './directory-api.js';
import { sinceText } from './duration.js';
import { formatRoute } from './route.js';
import { state } from './state.js';
import { messageOf } from './utils.js';
import { formatValue } from './value-format.js';
import {
  NO_TARGET,
  alarmBanner,
  alarmCard,
  directoryCard,
  logbookCard,
  node,
  putHistoryCard,
  recordAlarmText,
  timeText,
  unavailableLine,
  unconfiguredLine,
  valueCard,
} from './pv-page-sections.js';

/** @typedef {import('./directory-api.js').PvFacts} PvFacts */
/** @typedef {import('./directory-api.js').PvFactsSection} PvFactsSection */
/** @typedef {import('./directory-api.js').LogbookSection} LogbookSection */
/** @typedef {import('./directory-api.js').ValueRow} ValueRow */
/** @typedef {import('./directory-api.js').DirectoryRecord} DirectoryRecord */
/** @typedef {import('./directory-api.js').AlarmStateRecord} AlarmStateRecord */
/** @typedef {import('./directory-api.js').Source} Source */

const LOGIN_REQUIRED = 'logbook login required';
const LOGBOOK_UNAVAILABLE = 'logbook not available';
const NEEDS_TERMINAL = 'logbook needs the web terminal';
/** How many logbook entries the page asks for. */
const LOGBOOK_MAX = 10;
/** The words a logbook row without a subject is titled by. */
const TITLE_WORDS = 8;

/**
 * The mounted page; null while none is. A late answer for a page that is gone
 * finds a different (or no) mount and is dropped.
 * @type {{ container: HTMLElement, name: string } | null}
 */
let mounted = null;

// ---- Logbook ----

/**
 * Whether an ARIEL answer is its login page rather than a search answer.
 * @param {Response} resp
 * @returns {boolean}
 */
function isLoginAnswer(resp) {
  if (resp.type === 'opaqueredirect' || resp.redirected) return true;
  if (resp.status === 401 || resp.status === 403) return true;
  if (resp.status >= 300 && resp.status < 400) return true;
  return (resp.headers?.get('content-type') ?? '').includes('text/html');
}

/**
 * The text of a search snippet, its highlight tags removed.
 * @param {string} snippet
 * @returns {string}
 */
function snippetText(snippet) {
  return snippet.replace(/<\/?b>/gi, '');
}

/**
 * The logbook mentions of `name`: availability first, then ARIEL's own search
 * at its mount. Never throws; an answer that is not a search answer is a
 * reason.
 * @param {string} name
 * @returns {Promise<LogbookSection>}
 */
export async function readLogbook(name) {
  /** @param {string} reason @param {Source|null} source @returns {LogbookSection} */
  const unavailable = (reason, source) => ({
    section: 'logbook',
    outcome: 'unavailable',
    source,
    rows: [],
    reason,
  });
  /** @type {import('./directory-api.js').LogbookAvailability} */
  let availability;
  try {
    availability = await fetchLogbookAvailability();
  } catch (err) {
    return unavailable(messageOf(err), null);
  }
  if (!availability.available) return unavailable(availability.reason || NO_TARGET, null);
  const source = availability.source;
  const url = panelUrl('ariel', 'api/search');
  if (!url) return unavailable(NEEDS_TERMINAL, source);

  /** @type {Response} */
  let resp;
  try {
    resp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      credentials: 'same-origin',
      redirect: 'manual',
      body: JSON.stringify({
        query: `"${name}"`,
        mode: 'keyword',
        view: 'row',
        max_results: LOGBOOK_MAX,
      }),
    });
  } catch {
    return unavailable(LOGBOOK_UNAVAILABLE, source);
  }
  if (isLoginAnswer(resp)) return unavailable(LOGIN_REQUIRED, source);
  if (!resp.ok) return unavailable(LOGBOOK_UNAVAILABLE, source);
  /** @type {any} */
  let answer;
  try {
    answer = await resp.json();
  } catch {
    return unavailable(LOGBOOK_UNAVAILABLE, source);
  }
  /** @type {any[]} */
  const entries = Array.isArray(answer?.entries) ? answer.entries : [];
  const rows = entries.map((entry) => {
    const excerpt = snippetText(String(entry.snippet ?? ''));
    const words = excerpt.split(/\s+/).filter(Boolean).slice(0, TITLE_WORDS).join(' ');
    return {
      title: entry.subject || words,
      time: entry.timestamp ?? '',
      author: entry.author ?? '',
      excerpt,
      url: panelUrl('ariel', `#entry?id=${encodeURIComponent(String(entry.entry_id))}`),
    };
  });
  return { section: 'logbook', outcome: rows.length ? 'found' : 'no_match', source, rows };
}

/**
 * Fill the logbook card; a row opens the entry in ARIEL through the host,
 * or is an ordinary link where there is no host to hand it to.
 * @param {HTMLElement} frame
 * @param {LogbookSection} logbook
 */
function fillLogbook(frame, logbook) {
  frame.querySelector('[data-slot="logbook"]')?.remove();
  const body = node('div');
  body.dataset.slot = 'logbook';
  if (logbook.outcome === 'unavailable') {
    body.append(unavailableLine(logbook.reason));
  } else if (!logbook.rows.length) {
    body.append(node('p', 'pvp-note', 'No entries found'));
  } else {
    const list = node('ul', 'pvp-logbook');
    for (const row of logbook.rows) {
      const item = node('li', 'pvp-logbook-row');
      const link = /** @type {HTMLAnchorElement} */ (node('a', 'pvp-logbook-title', row.title));
      if (row.url) {
        const url = row.url;
        link.href = url;
        link.addEventListener('click', (event) => {
          if (!handoffAllowed()) return;
          event.preventDefault();
          window.parent.postMessage(
            { type: 'osprey:navigate', panel: 'ariel', url },
            window.location.origin
          );
        });
      }
      item.append(link);
      item.append(
        node(
          'span',
          'pvp-logbook-meta',
          [timeText(row.time), row.author].filter(Boolean).join(' · ')
        )
      );
      if (row.excerpt) item.append(node('span', 'pvp-logbook-excerpt', row.excerpt));
      list.append(item);
    }
    body.append(list);
  }
  frame.append(body);
}

// ---- Current value ----

/**
 * Read the PV's value and show it in the Value card and as the record's own
 * alarm in the Alarm card and the banner.
 * @param {HTMLElement} page
 * @param {string} name
 * @returns {Promise<void>}
 */
async function refreshValue(page, name) {
  const slot = page.querySelector('[data-slot="value"]');
  const owns = () => page.querySelectorAll('[data-slot="record-alarm"]');
  if (slot) slot.replaceChildren(node('p', 'pvp-note', 'Reading…'));
  /** @type {ValueRow|undefined} */
  let row;
  /** @type {string} */
  let failure = '';
  try {
    row = (await readValues([name])).find((answer) => answer.name === name);
    if (!row) failure = 'no answer';
  } catch (err) {
    failure = messageOf(err);
  }
  if (!page.isConnected || mounted?.name !== name) return;
  if (!row || row.outcome !== 'found') {
    const text =
      row?.outcome === 'no_answer' ? 'no answer' : `Unavailable: ${row?.reason || failure}`;
    slot?.replaceChildren(node('p', 'pvp-note', text));
    for (const own of owns()) {
      own.textContent = row?.outcome === 'no_answer' ? 'no answer' : 'unavailable';
    }
    return;
  }
  const formatted = formatValue(row);
  const big = node('div', 'pvp-bigval', formatted.text);
  if (formatted.title) big.title = formatted.title;
  if (formatted.unit) big.append(node('small', 'pvp-unit', formatted.unit));
  const alarm = recordAlarmText(row);
  const read = [row.timestamp && `read ${timeText(row.timestamp)}`, alarm && `record ${alarm}`]
    .filter(Boolean)
    .join(' · ');
  slot?.replaceChildren(big, node('p', 'pvp-note', read));
  for (const own of owns()) own.textContent = alarm;
}

// ---- Header ----

/**
 * @param {PvFactsSection|undefined} directory
 * @returns {HTMLElement|null}
 */
function listingChip(directory) {
  if (!directory) return null;
  const record = /** @type {DirectoryRecord|null|undefined} */ (directory.record);
  /** @type {[string, string]} */
  let chip;
  if (directory.outcome === 'not_listed') chip = ['', 'not listed'];
  else if (directory.outcome === 'unavailable') chip = ['', 'listing unavailable'];
  else if (directory.outcome === 'inactive' || record?.row.status === 'Inactive')
    chip = ['osprey-chip--warn', 'Inactive'];
  else if (record?.row.status === 'Active') chip = ['osprey-chip--ok', 'Active'];
  else return null;
  const element = node('span', `osprey-chip ${chip[0]}`.trim());
  element.append(node('span', 'osprey-chip-label', chip[1]));
  element.dataset.slot = 'listing';
  return element;
}

/**
 * The breadcrumb: All PVs, the PV's IOC when the directory names one, the PV.
 * @param {string} name
 * @param {string|null|undefined} ioc
 * @returns {HTMLElement}
 */
function breadcrumb(name, ioc) {
  const crumb = node('nav', 'pvp-crumb');
  crumb.setAttribute('aria-label', 'Breadcrumb');
  const all = /** @type {HTMLAnchorElement} */ (node('a', '', 'All PVs'));
  all.href = formatRoute({ view: 'allpvs', params: {} });
  crumb.append(all, node('span', 'pvp-crumb-sep', '/'));
  if (ioc) {
    const link = /** @type {HTMLAnchorElement} */ (node('a', '', ioc));
    link.href = formatRoute({ view: 'ioc', params: { name: ioc } });
    crumb.append(link, node('span', 'pvp-crumb-sep', '/'));
  }
  crumb.append(node('span', 'pvp-crumb-leaf', name));
  return crumb;
}

/**
 * @param {string} name
 * @param {Map<string, PvFactsSection>} sections
 * @returns {HTMLElement}
 */
function header(name, sections) {
  const head = node('header', 'pvp-head');
  const title = node('div', 'pvp-title');
  const directory = sections.get('directory');
  const origin = directory?.origin;
  if (origin === 'curated' || origin === 'directory') {
    const mark = node('span', 'pvp-mark', origin === 'curated' ? '◆' : '◇');
    const label = origin === 'curated' ? 'Curated' : 'Control system only';
    mark.title = label;
    mark.setAttribute('aria-label', label);
    title.append(mark);
  }
  title.append(node('h2', 'pvp-name', name));
  const chip = listingChip(directory);
  if (chip) title.append(chip);
  const alarm = /** @type {AlarmStateRecord|null|undefined} */ (
    sections.get('alarm_state')?.record
  );
  if (alarm?.stale) {
    const stale = node('span', 'osprey-chip osprey-chip--warn pvp-stale');
    stale.append(
      icon('triangle-alert', { size: 12 }),
      node('span', 'osprey-chip-label', 'alarm feed stale')
    );
    title.append(stale);
  }
  head.append(title);

  const row = /** @type {DirectoryRecord|null|undefined} */ (directory?.record)?.row;
  if (row?.description) head.append(node('p', 'pvp-sub', row.description));
  const actions = node('div', 'pvp-actions');
  actions.dataset.slot = 'actions';
  const copy = node('button', 'osprey-icon-btn');
  copy.setAttribute('type', 'button');
  copy.setAttribute('aria-label', 'Copy name');
  copy.title = 'Copy name';
  copy.dataset.action = 'copy';
  copy.append(icon('copy'));
  copy.addEventListener('click', async () => {
    const copied = await copyText(name);
    showToast(copied ? `Copied ${name}` : 'Copy failed', { kind: copied ? 'success' : 'error' });
  });
  actions.append(copy);
  head.append(actions);
  return head;
}

/**
 * Offer Send to assistant and Plot (CANOPY, Trace answering), only where the
 * host can be handed them.
 * @param {HTMLElement} actions
 * @param {string} name
 * @param {() => Promise<boolean>} plotCheck
 * @returns {Promise<void>}
 */
async function fillActions(actions, name, plotCheck) {
  if (!handoffAllowed() || !panelApiPrefix()) return;
  const send = node('button', 'osprey-btn', 'Send to assistant');
  send.setAttribute('type', 'button');
  send.addEventListener('click', () => {
    window.parent.postMessage(
      { type: 'osprey-paste-to-terminal', text: name },
      window.location.origin
    );
    showToast(`Posted ${name} to the prompt`, { kind: 'success' });
  });
  actions.append(send);
  const url = plotUrl([name]);
  if (!url || !(await plotCheck())) return;
  if (!actions.isConnected) return;
  const plot = node('button', 'osprey-btn primary', 'Plot');
  plot.setAttribute('type', 'button');
  plot.addEventListener('click', () => {
    window.parent.postMessage(
      { type: 'osprey:navigate', panel: 'canopy', url },
      window.location.origin
    );
  });
  actions.append(plot);
}

// ---- The page ----

/**
 * Whether the deployment configures `kind`: the descriptor says, and without
 * one the facts do (they carry only configured sections).
 * @param {'alarm_state'|'alarm_log'|'put_log'|'logbook'} kind
 * @param {boolean} inFacts
 * @returns {boolean}
 */
function configured(kind, inFacts) {
  const services = state.directory?.configured_services;
  return services ? services.includes(kind) : inFacts;
}

/**
 * @param {HTMLElement} container
 * @param {string} name
 * @returns {Promise<void>}
 */
async function load(container, name) {
  renderState(container, { icon: 'refresh', message: `Loading ${name}…` });
  /** @type {PvFacts} */
  let facts;
  try {
    facts = await fetchPvFacts(name);
  } catch (err) {
    if (mounted?.container !== container) return;
    renderState(container, {
      icon: 'triangle-alert',
      message: 'PV facts not available',
      detail: messageOf(err),
      action: { label: 'Retry', onClick: () => void load(container, name) },
    });
    return;
  }
  if (mounted?.container !== container || mounted.name !== name) return;

  if (facts.outcome === 'unknown') {
    const label = state.directory?.label ?? 'All PVs';
    renderState(container, {
      icon: 'search',
      message: `${name} is not listed in ${label}`,
      action: {
        label: `Search ${label}`,
        onClick: () => {
          location.hash = formatRoute({ view: 'allpvs', params: { q: name } });
        },
      },
    });
    return;
  }

  /** @type {Map<string, PvFactsSection>} */
  const bySection = new Map(facts.sections.map((section) => [section.section, section]));
  const directory = bySection.get('directory');
  const alarmState = bySection.get('alarm_state');
  const alarmConfig = bySection.get('alarm_config');
  const alarmHistory = bySection.get('alarm_history');
  const putHistory = bySection.get('put_history');
  const row = /** @type {DirectoryRecord|null|undefined} */ (directory?.record)?.row;

  const page = node('div', 'pv-page');
  page.append(breadcrumb(name, row?.ioc), header(name, bySection));
  const since = alarmState?.record
    ? sinceText(/** @type {AlarmStateRecord} */ (alarmState.record).time, Date.now())
    : '';
  const banner = alarmBanner(alarmState, alarmConfig, since);
  if (banner) page.append(banner);

  const left = node('div', 'pvp-col');
  const right = node('div', 'pvp-col');
  left.append(valueCard(() => void refreshValue(page, name)));
  const alarms =
    configured('alarm_state', Boolean(alarmState)) ||
    configured('alarm_log', Boolean(alarmHistory));
  left.append(
    alarms
      ? alarmCard({ state: alarmState, config: alarmConfig, history: alarmHistory })
      : unconfiguredLine('Alarms')
  );
  if (!configured('put_log', Boolean(putHistory))) left.append(unconfiguredLine('Put history'));
  else if (putHistory) left.append(putHistoryCard(putHistory));
  else left.append(missingCard('put_history', 'Put history'));

  if (directory) right.append(directoryCard(directory));
  /** @type {HTMLElement|null} */
  let logbookFrame = null;
  if (configured('logbook', Boolean(state.directory?.ariel))) {
    logbookFrame = logbookCard();
    right.append(logbookFrame);
  } else {
    right.append(unconfiguredLine('Logbook'));
  }
  const grid = node('div', 'pvp-grid');
  grid.append(left, right);
  page.append(grid);
  container.replaceChildren(page);

  const actions = /** @type {HTMLElement} */ (page.querySelector('[data-slot="actions"]'));
  const tasks = [refreshValue(page, name), fillActions(actions, name, createPlotCheck())];
  if (logbookFrame) {
    const frame = logbookFrame;
    tasks.push(
      readLogbook(name).then((logbook) => {
        if (mounted?.container === container && mounted.name === name) fillLogbook(frame, logbook);
      })
    );
  }
  await Promise.all(tasks);
}

/**
 * A configured card the facts do not carry: the facts said nothing of it.
 * @param {string} id
 * @param {string} title
 * @returns {HTMLElement}
 */
function missingCard(id, title) {
  const frame = node('section', 'osprey-card pvp-card');
  frame.dataset.section = id;
  frame.append(node('h3', 'pvp-card-title', title), unavailableLine('no answer'));
  return frame;
}

/**
 * Mount the page of the PV `params.name`; resolves once every read settled.
 * @param {HTMLElement} container
 * @param {{ name?: string }} params
 * @returns {Promise<void>}
 */
export function mountPvPage(container, params) {
  unmountPvPage();
  const name = params?.name ?? '';
  mounted = { container, name };
  return load(container, name);
}

/** @returns {void} */
export function unmountPvPage() {
  mounted = null;
}
