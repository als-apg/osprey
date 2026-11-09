// @ts-check
/**
 * OSPREY Channel Finder — PV Page
 *
 * The view of `#/pv/<name>` (Panel view contract): every configured section
 * of one PV, in a fixed order, each only when the facts carry it, then the
 * logbook mentions and the current value.
 *
 *   [◆ NAME  Active  ⚠ alarm feed stale]           [Plot] [Send to assistant]
 *   Directory · Alarm state · Alarm history · Alarm configuration ·
 *   Put history · Logbook · Current value
 *
 * A section that has no answer says why, on its own line; the page never
 * reads a missing answer as a clear state. The logbook is read by this page,
 * through ARIEL's own search at its mount, with the operator's session: the
 * panel's backend holds no logbook credential. The toolbar search searches
 * the directory.
 */

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
import { claimToolbar, searchInput } from './panel-toolbar.js';
import { state } from './state.js';
import { messageOf } from './utils.js';
import {
  NO_TARGET,
  SECTION_ORDER,
  facts,
  factsSection,
  node,
  sectionFrame,
  severityText,
  timeText,
  unavailableLine,
  valueText,
} from './pv-page-sections.js';

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
 * @type {{ container: HTMLElement, name: string, releaseToolbar: () => void, onSearchKey: (event: KeyboardEvent) => void } | null}
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
 * Fill the logbook section; a row opens the entry in ARIEL through the host,
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
 * Read the PV's value and show it in the value section and as the record's
 * own alarm.
 * @param {HTMLElement} page
 * @param {string} name
 * @returns {Promise<void>}
 */
async function refreshValue(page, name) {
  const slot = page.querySelector('[data-slot="value"]');
  const own = page.querySelector('[data-slot="record-alarm"]');
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
    if (own) own.textContent = row?.outcome === 'no_answer' ? 'no answer' : 'unavailable';
    return;
  }
  /** @type {[string, string][]} */
  const pairs = [['Value', valueText(row.value)]];
  if (row.timestamp) pairs.push(['Time', timeText(row.timestamp)]);
  slot?.replaceChildren(facts(pairs));
  if (own) own.textContent = [severityText(row.severity), row.status].filter(Boolean).join(' ');
}

/**
 * @param {HTMLElement} page
 * @param {string} name
 * @returns {HTMLElement}
 */
function valueSection(page, name) {
  const frame = sectionFrame('value');
  const refresh = node('button', 'osprey-btn pvp-refresh');
  refresh.setAttribute('type', 'button');
  refresh.append(icon('refresh'), node('span', 'osprey-btn-label', 'Refresh'));
  refresh.addEventListener('click', () => void refreshValue(page, name));
  frame.querySelector('h3')?.append(refresh);
  const slot = node('div');
  slot.dataset.slot = 'value';
  frame.append(slot);
  return frame;
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
  if (row?.description || row?.ioc) {
    const line = node('p', 'pvp-sub');
    if (row.description) line.append(node('span', '', row.description));
    if (row.ioc) {
      const ioc = /** @type {HTMLAnchorElement} */ (node('a', 'pvp-ioc', row.ioc));
      ioc.href = `#/ioc/${encodeURIComponent(row.ioc)}`;
      line.append(node('span', '', 'IOC '), ioc);
    }
    head.append(line);
  }
  const actions = node('div', 'pvp-actions');
  actions.dataset.slot = 'actions';
  head.append(actions);
  return head;
}

/**
 * Offer Plot (CANOPY, Trace answering) and Send to assistant, only where the
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
  actions.prepend(plot);
}

// ---- The page ----

/**
 * Whether the facts know nothing of the PV: the directory does not list it
 * and no other section found anything.
 * @param {PvFactsSection[]} sections
 * @returns {boolean}
 */
function unknownRecord(sections) {
  const directory = sections.find((section) => section.section === 'directory');
  if (directory?.outcome !== 'not_listed') return false;
  return sections.every(
    (section) =>
      section.section === 'directory' ||
      section.outcome !== 'found' ||
      (!section.record && !section.rows?.length)
  );
}

/**
 * @param {HTMLElement} container
 * @param {string} name
 * @returns {Promise<void>}
 */
async function load(container, name) {
  renderState(container, { icon: 'refresh', message: `Loading ${name}…` });
  /** @type {PvFactsSection[]} */
  let sections;
  try {
    sections = (await fetchPvFacts(name)).sections;
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

  if (unknownRecord(sections)) {
    const label = state.directory?.label ?? 'All PVs';
    renderState(container, {
      icon: 'search',
      message: `${name} is not listed in ${label}`,
      action: {
        label: `Search ${label}`,
        onClick: () => {
          location.hash = `#/allpvs?q=${encodeURIComponent(name)}`;
        },
      },
    });
    return;
  }

  /** @type {Map<string, PvFactsSection>} */
  const bySection = new Map(sections.map((section) => [section.section, section]));
  const page = node('div', 'pv-page');
  page.append(header(name, bySection));
  for (const id of SECTION_ORDER) {
    const section = bySection.get(id);
    if (section) page.append(factsSection(section));
  }
  /** @type {HTMLElement|null} */
  let logbookFrame = null;
  if (state.directory?.ariel) {
    logbookFrame = sectionFrame('logbook');
    const pending = node('p', 'pvp-note', 'Reading…');
    pending.dataset.slot = 'logbook';
    logbookFrame.append(pending);
    page.append(logbookFrame);
  }
  page.append(valueSection(page, name));
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
 * Mount the page of the PV `params.name`; resolves once every read settled.
 * @param {HTMLElement} container
 * @param {{ name?: string }} params
 * @returns {Promise<void>}
 */
export function mountPvPage(container, params) {
  unmountPvPage();
  const name = params?.name ?? '';
  const releaseToolbar = claimToolbar();
  /** @param {KeyboardEvent} event */
  const onSearchKey = (event) => {
    if (event.key !== 'Enter') return;
    const words = /** @type {HTMLInputElement} */ (event.target).value.trim();
    location.hash = words ? `#/allpvs?q=${encodeURIComponent(words)}` : '#/allpvs';
  };
  searchInput()?.addEventListener('keydown', onSearchKey);
  mounted = { container, name, releaseToolbar, onSearchKey };
  return load(container, name);
}

/** @returns {void} */
export function unmountPvPage() {
  if (!mounted) return;
  searchInput()?.removeEventListener('keydown', mounted.onSearchKey);
  mounted.releaseToolbar();
  mounted = null;
}
