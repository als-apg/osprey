// @ts-check
/**
 * OSPREY Channel Finder — PV Page cards
 *
 * The cards of the PV page, each built from the facts sections it shows (PV
 * facts per section), and the small DOM pieces the page shares with them. A
 * card whose service the deployment does not configure is one dashed line
 * ({@link unconfiguredLine}); a configured card without an answer says why.
 * Every value is set as text; severities draw through severityMark.
 */

import { icon } from '/design-system/js/icons.js';
import { formatFacilityTime } from '/design-system/js/facility-time.js';
import { configEntries } from './alarm-text.js';
import { sinceText } from './duration.js';
import { formatRoute } from './route.js';
import { severityMark, severityOf } from './severity.js';

/** @typedef {import('./directory-api.js').PvFactsSection} PvFactsSection */
/** @typedef {import('./directory-api.js').DirectoryRecord} DirectoryRecord */
/** @typedef {import('./directory-api.js').ArchiveRecord} ArchiveRecord */
/** @typedef {import('./directory-api.js').AlarmStateRecord} AlarmStateRecord */
/** @typedef {import('./directory-api.js').AlarmConfigRecord} AlarmConfigRecord */
/** @typedef {import('./directory-api.js').AlarmHistoryRow} AlarmHistoryRow */
/** @typedef {import('./directory-api.js').PutRow} PutRow */
/** @typedef {import('./directory-api.js').ValueRow} ValueRow */

export const NO_TARGET = 'not available for this target';
const STALE_FEED = 'stale (the alarm feed is not updating)';
const NOT_IN_TREE = 'Not in the alarm tree';
/** The rows a history carries at most (the facts service's limit). */
const HISTORY_LIMIT = 20;
const RETENTION_UNKNOWN = 'Retention unknown: how far back the service keeps rows is not known';

/** @type {Intl.DateTimeFormatOptions} */
const TIME_FORMAT = {
  year: 'numeric',
  month: '2-digit',
  day: '2-digit',
  hour: '2-digit',
  minute: '2-digit',
  second: '2-digit',
  hourCycle: 'h23',
};

// ---- DOM helpers ----

/**
 * An element with a class and text; text is always set as text.
 * @param {string} tag
 * @param {string} [className]
 * @param {string} [text]
 * @returns {HTMLElement}
 */
export function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

/**
 * A time as the facility clock shows it, or the text as given when it does
 * not parse.
 * @param {string|null|undefined} value
 * @returns {string}
 */
export function timeText(value) {
  if (!value) return '';
  return formatFacilityTime(value, TIME_FORMAT) || value;
}

/**
 * The record's own alarm from a value read, as words: its severity (severity.js
 * reads the EPICS number the read carries) and its status, once when the two
 * say the same.
 * @param {ValueRow} row
 * @returns {string}
 */
export function recordAlarmText(row) {
  const severity =
    row.severity === null || row.severity === undefined ? '' : severityOf(row.severity);
  const words = [severity, row.status ?? ''].filter(Boolean);
  return [...new Set(words)].join(' · ');
}

/**
 * The severity mark of `severity`, with `text` in place of the word.
 * @param {unknown} severity
 * @param {string} [text]
 * @returns {HTMLElement}
 */
function mark(severity, text) {
  const holder = node('span', 'pvp-sev');
  holder.innerHTML = severityMark(severityOf(severity), text);
  return holder;
}

/**
 * The line a section without an answer shows.
 * @param {string|undefined} reason
 * @returns {HTMLElement}
 */
export function unavailableLine(reason) {
  if (reason === NO_TARGET) return node('p', 'pvp-note', 'Not available for this target');
  return node('p', 'pvp-note', `Unavailable: ${reason || 'no reason given'}`);
}

/**
 * A label/value list.
 * @param {[string, string | Node][]} pairs
 * @returns {HTMLElement}
 */
export function facts(pairs) {
  const list = node('dl', 'pvp-facts');
  for (const [label, value] of pairs) {
    list.append(node('dt', '', label));
    const cell = node('dd');
    cell.append(value);
    list.append(cell);
  }
  return list;
}

/**
 * Guidance or displays as one title line and one details line per entry; null
 * for none.
 * @param {unknown} value
 * @param {string} [className] - of each entry
 * @returns {HTMLElement|null}
 */
export function entriesNode(value, className = 'pvp-entry') {
  const entries = configEntries(value);
  if (!entries.length) return null;
  const list = node('div', 'pvp-entries');
  for (const entry of entries) {
    const item = node('div', className);
    if (entry.title) item.append(node('div', 'pvp-entry-title', entry.title));
    if (entry.details) item.append(node('div', 'pvp-entry-details', entry.details));
    list.append(item);
  }
  return list;
}

/**
 * A table of rows; every cell is text.
 * @param {string[]} heads
 * @param {string[][]} rows
 * @returns {HTMLElement}
 */
function table(heads, rows) {
  const tableEl = node('table', 'data-table pvp-table');
  const headRow = node('tr');
  for (const head of heads) headRow.append(node('th', '', head));
  const thead = node('thead');
  thead.append(headRow);
  tableEl.append(thead);
  const body = node('tbody');
  for (const row of rows) {
    const tr = node('tr');
    for (const cell of row) tr.append(node('td', '', cell));
    body.append(tr);
  }
  tableEl.append(body);
  return tableEl;
}

/**
 * A card with its title; the body goes into the returned element.
 * @param {string} id - the `data-section` the page and its tests find it by
 * @param {string} title
 * @param {string} [aside] - shown beside the title
 * @returns {HTMLElement}
 */
function card(id, title, aside) {
  const section = node('section', 'osprey-card pvp-card');
  section.dataset.section = id;
  const head = node('h3', 'pvp-card-title', title);
  if (aside) head.append(node('span', 'pvp-card-aside', aside));
  section.append(head);
  return section;
}

/**
 * The one dashed line of a card whose service is not configured.
 * @param {string} title
 * @returns {HTMLElement}
 */
export function unconfiguredLine(title) {
  const line = node('div', 'pvp-off', `${title} · not configured`);
  line.dataset.off = title;
  return line;
}

/**
 * The footer lines of a section: the caveats of its source and, on a history
 * that answered, that only the newest rows are shown or that its retention is
 * not known.
 * @param {PvFactsSection} section
 * @returns {string[]}
 */
function footerLines(section) {
  const lines = [...(section.caveats ?? [])];
  if (section.truncated) lines.push(`Newest ${HISTORY_LIMIT} rows shown`);
  if (section.outcome !== 'unavailable' && section.retention_known === false) {
    lines.push(RETENTION_UNKNOWN);
  }
  return lines;
}

/**
 * A footer of `lines`, each once; null for none.
 * @param {string[]} lines
 * @returns {HTMLElement|null}
 */
function footer(lines) {
  const unique = [...new Set(lines)];
  if (!unique.length) return null;
  const element = node('footer', 'pvp-footer');
  for (const line of unique) element.append(node('p', 'pvp-note', line));
  return element;
}

/**
 * A history section's rows as a table, or its reason, or that it has none.
 * @param {PvFactsSection} section
 * @returns {HTMLElement}
 */
function historyBody(section) {
  if (section.outcome === 'unavailable') return unavailableLine(section.reason);
  if (!section.rows?.length) return node('p', 'pvp-note', 'No rows recorded');
  if (section.section === 'put_history') {
    const rows = /** @type {PutRow[]} */ (section.rows);
    return table(
      ['Time', 'Field', 'Old', 'New', 'User', 'Client'],
      rows.map((row) => [
        timeText(row.at),
        row.field ?? '',
        row.old ?? '',
        row.new ?? '',
        row.user ?? '',
        row.client ?? '',
      ])
    );
  }
  const rows = /** @type {AlarmHistoryRow[]} */ (section.rows);
  return table(
    ['Time', 'Severity', 'Status', 'Message', 'User'],
    rows.map((row) => [timeText(row.time), row.severity, row.status, row.message, row.user ?? ''])
  );
}

// ---- Cards ----

/**
 * The Value card: the value with its unit, the read time and the record's own
 * alarm, filled by the page's value read; Refresh reads again.
 * @param {() => void} onRefresh
 * @returns {HTMLElement}
 */
export function valueCard(onRefresh) {
  const frame = card('value', 'Value');
  const refresh = node('button', 'osprey-icon-btn pvp-refresh');
  refresh.setAttribute('type', 'button');
  refresh.setAttribute('aria-label', 'Refresh');
  refresh.title = 'Refresh';
  refresh.append(icon('refresh'));
  refresh.addEventListener('click', onRefresh);
  frame.querySelector('h3')?.append(refresh);
  const slot = node('div', 'pvp-value');
  slot.dataset.slot = 'value';
  slot.append(node('p', 'pvp-note', 'Reading…'));
  frame.append(slot);
  return frame;
}

/**
 * The Directory card: IOC, host, record type and origin, the detail
 * properties first and the rest behind All properties, then the tags.
 * @param {PvFactsSection} section
 * @returns {HTMLElement}
 */
export function directoryCard(section) {
  const frame = card('directory', 'Directory');
  if (section.outcome === 'unavailable') {
    frame.append(unavailableLine(section.reason));
  } else if (section.outcome === 'not_listed') {
    frame.append(node('p', 'pvp-note', `Not listed in ${section.source.label}`));
  } else {
    const record = /** @type {DirectoryRecord|null|undefined} */ (section.record);
    const row = record?.row;
    /** @type {[string, string | Node][]} */
    const pairs = [];
    if (row?.ioc) {
      const link = /** @type {HTMLAnchorElement} */ (node('a', 'pvp-ioc', row.ioc));
      link.href = formatRoute({ view: 'ioc', params: { name: row.ioc } });
      pairs.push(['IOC', link]);
    }
    if (row?.host) pairs.push(['Host', row.host]);
    if (row?.record_type) pairs.push(['Record type', row.record_type]);
    if (section.origin === 'curated') pairs.push(['Origin', 'Curated']);
    else if (section.origin === 'directory') pairs.push(['Origin', 'Control system only']);
    if (pairs.length) frame.append(facts(pairs));

    const properties = record?.properties ?? {};
    const names = Object.keys(properties);
    const first = section.detail_properties ?? names;
    const shown = first.filter((name) => Object.hasOwn(properties, name));
    const rest = names.filter((name) => !shown.includes(name));
    if (shown.length) frame.append(facts(shown.map((name) => [name, properties[name]])));
    if (rest.length) {
      const more = node('details', 'pvp-more');
      more.append(node('summary', '', 'All properties'));
      more.append(facts(rest.map((name) => [name, properties[name]])));
      frame.append(more);
    }
    if (record?.tags?.length) frame.append(facts([['Tags', record.tags.join(', ')]]));
  }
  const lines = footer(footerLines(section));
  if (lines) frame.append(lines);
  return frame;
}

/**
 * The Archive card: whether the target's archive records the PV (`unknown`
 * with the archiver's reason when it cannot say), its first and last sample
 * with the age of the last, and its sampling period when known.
 * @param {PvFactsSection} section
 * @param {number} [now] - ms since the epoch
 * @returns {HTMLElement}
 */
export function archiveCard(section, now = Date.now()) {
  const frame = card('archive', 'Archive');
  const record = /** @type {ArchiveRecord|null|undefined} */ (section.record);
  if (section.outcome === 'unavailable' || !record) {
    frame.append(unavailableLine(section.reason));
  } else {
    const archived =
      record.archived === null
        ? `unknown (${record.description || 'the archiver cannot say'})`
        : record.archived
          ? 'yes'
          : 'no';
    /** @type {[string, string][]} */
    const pairs = [['Archived', archived]];
    if (record.first_sample) pairs.push(['First sample', timeText(record.first_sample)]);
    if (record.last_sample) {
      const age = sinceText(record.last_sample, now);
      pairs.push(['Last sample', `${timeText(record.last_sample)}${age ? ` · ${age} ago` : ''}`]);
    }
    if (record.sampling_period_s !== null) {
      pairs.push(['Sampling period', `${record.sampling_period_s} s`]);
    }
    frame.append(facts(pairs));
  }
  const lines = footer(footerLines(section));
  if (lines) frame.append(lines);
  return frame;
}

/**
 * The alarm server's lines of the Alarm card, from the alarm state section.
 * @param {PvFactsSection} section
 * @param {HTMLElement} own - the record's own alarm, filled by the value read
 * @returns {HTMLElement[]}
 */
function alarmStateLines(section, own) {
  if (section.outcome === 'unavailable') {
    return [unavailableLine(section.reason), facts([['Record', own]])];
  }
  const record = /** @type {AlarmStateRecord|null|undefined} */ (section.record);
  if (!record) return [mark(null, NOT_IN_TREE), facts([['Record', own]])];
  /** @type {[string, string | Node][]} */
  const pairs = [];
  if (record.stale) {
    pairs.push(['Alarm server', STALE_FEED]);
  } else {
    const words = [record.severity, record.message || record.status].filter(Boolean);
    pairs.push(['Alarm server', mark(record.severity, words.join(' · '))]);
  }
  pairs.push(['Record', own]);
  if (!record.stale) {
    if (record.time) pairs.push(['Since', timeText(record.time)]);
    pairs.push(['Latched', record.latched ? 'yes' : 'no']);
    pairs.push(['Acknowledged', record.acknowledged ? 'yes' : 'no']);
  }
  return [facts(pairs)];
}

/**
 * The configuration's lines of the Alarm card: tree path, enabled, latching
 * and displays, then each guidance entry as a block.
 * @param {PvFactsSection} section
 * @returns {HTMLElement[]}
 */
function alarmConfigLines(section) {
  if (section.outcome === 'unavailable') return [unavailableLine(section.reason)];
  const record = /** @type {AlarmConfigRecord|null|undefined} */ (section.record);
  if (!record) return [];
  /** @type {[string, string | Node][]} */
  const pairs = [];
  if (record.in_tree !== 'yes') pairs.push(['In alarm tree', record.in_tree]);
  if (record.path) pairs.push(['Tree path', record.path]);
  if (record.enabled !== null) pairs.push(['Enabled', record.enabled ? 'yes' : 'no']);
  if (record.latching !== null) pairs.push(['Latching', record.latching ? 'yes' : 'no']);
  const displays = entriesNode(record.displays);
  if (displays) pairs.push(['Displays', displays]);
  const lines = pairs.length ? [facts(pairs)] : [];
  const guidance = entriesNode(record.guidance, 'pvp-guidance');
  if (guidance) lines.push(guidance);
  return lines;
}

/**
 * The Alarm card: one card from the alarm state, alarm configuration and
 * alarm history sections, each line keeping its own section's answer.
 * History is shown only with an alarm history section.
 * @param {{ state?: PvFactsSection, config?: PvFactsSection, history?: PvFactsSection }} sections
 * @returns {HTMLElement}
 */
export function alarmCard({ state, config, history }) {
  const configRecord = /** @type {AlarmConfigRecord|null|undefined} */ (config?.record);
  const frame = card('alarm', 'Alarm', config?.label ?? configRecord?.label);
  const present = [state, config, history].filter((s) => s !== undefined);
  if (
    present.length &&
    present.every((s) => s.outcome === 'unavailable' && s.reason === NO_TARGET)
  ) {
    frame.append(unavailableLine(NO_TARGET));
    return frame;
  }
  const own = node('span', 'pvp-record-alarm', '…');
  own.dataset.slot = 'record-alarm';
  if (state) frame.append(...alarmStateLines(state, own));
  else frame.append(facts([['Record', own]]));
  if (config) frame.append(...alarmConfigLines(config));
  if (history) {
    const part = node('div', 'pvp-card-part');
    part.dataset.part = 'history';
    part.append(node('div', 'osprey-section-label', 'History'), historyBody(history));
    frame.append(part);
  }
  const lines = footer([state, config, history].flatMap((s) => (s ? footerLines(s) : [])));
  if (lines) frame.append(lines);
  return frame;
}

/**
 * The Put history card.
 * @param {PvFactsSection} section
 * @returns {HTMLElement}
 */
export function putHistoryCard(section) {
  const frame = card('put_history', 'Put history');
  frame.append(historyBody(section));
  const lines = footer(footerLines(section));
  if (lines) frame.append(lines);
  return frame;
}

/**
 * The Logbook card, its body filled by the page once ARIEL answers.
 * @returns {HTMLElement}
 */
export function logbookCard() {
  const frame = card('logbook', 'Logbook');
  const pending = node('p', 'pvp-note', 'Reading…');
  pending.dataset.slot = 'logbook';
  frame.append(pending);
  return frame;
}

/**
 * The alarm banner, only when the alarm server reports the PV not OK: the
 * severity and alarm, how long and since when, latched (with the record's own
 * alarm now) and acknowledged, and the first guidance entry.
 * @param {PvFactsSection|undefined} state
 * @param {PvFactsSection|undefined} config
 * @param {string} since - how long the alarm has held, e.g. `38 min`; empty when not known
 * @returns {HTMLElement|null}
 */
export function alarmBanner(state, config, since) {
  if (state?.outcome !== 'found') return null;
  const record = /** @type {AlarmStateRecord|null|undefined} */ (state.record);
  const severity = severityOf(record?.severity);
  if (!record || severity === 'OK' || severity === 'NONE') return null;

  const banner = node('div', `pvp-banner pvp-banner--${severity.toLowerCase()}`);
  banner.setAttribute('role', 'status');
  const head = node('div', 'pvp-banner-main');
  const words = [record.severity, record.message || record.status].filter(Boolean);
  head.append(node('div', 'pvp-banner-alarm', words.join(' · ')));

  const rest = node('div', 'pvp-banner-rest');
  const when = [since && `for ${since}`, record.time && `since ${timeText(record.time)}`]
    .filter(Boolean)
    .join(', ');
  if (when) rest.append(when, ' · ');
  if (record.latched) {
    const own = node('span', '', record.status || '…');
    own.dataset.slot = 'record-alarm';
    rest.append('latched (record now ', own, ') · ');
  }
  rest.append(record.acknowledged ? 'acknowledged' : 'not acknowledged');
  head.append(rest);
  banner.append(head);

  const configRecord = /** @type {AlarmConfigRecord|null|undefined} */ (config?.record);
  const [first] = configEntries(configRecord?.guidance);
  if (first) {
    const guidance = node('div', 'pvp-banner-guidance');
    if (first.title) guidance.append(node('strong', '', `${first.title}. `));
    guidance.append(first.details);
    banner.append(guidance);
  }
  return banner;
}
