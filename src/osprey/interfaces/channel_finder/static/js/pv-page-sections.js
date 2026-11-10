// @ts-check
/**
 * OSPREY Channel Finder — PV Page sections
 *
 * The facts sections of the PV page, each rendered from its payload alone
 * (PV facts per section), and the small DOM pieces the page shares with
 * them. Every value is set as text.
 */

import { formatFacilityTime } from '/design-system/js/facility-time.js';

/** @typedef {import('./directory-api.js').PvFactsSection} PvFactsSection */
/** @typedef {import('./directory-api.js').DirectoryRecord} DirectoryRecord */
/** @typedef {import('./directory-api.js').AlarmStateRecord} AlarmStateRecord */
/** @typedef {import('./directory-api.js').AlarmConfigRecord} AlarmConfigRecord */
/** @typedef {import('./directory-api.js').AlarmHistoryRow} AlarmHistoryRow */
/** @typedef {import('./directory-api.js').PutRow} PutRow */

/** The facts sections, in the order the page shows them. */
export const SECTION_ORDER = Object.freeze(
  /** @type {const} */ ([
    'directory',
    'alarm_state',
    'alarm_history',
    'alarm_config',
    'put_history',
  ])
);

/** @type {Record<PvFactsSection['section'] | 'logbook' | 'value', string>} */
const TITLES = {
  directory: 'Directory',
  alarm_state: 'Alarm state',
  alarm_history: 'Alarm history',
  alarm_config: 'Alarm configuration',
  put_history: 'Put history',
  logbook: 'Logbook',
  value: 'Current value',
};

export const NO_TARGET = 'not available for this target';
const STALE_FEED = 'stale (the alarm feed is not updating)';
/** The rows a history carries at most (the facts service's limit). */
const HISTORY_LIMIT = 20;
const RETENTION_UNKNOWN = 'Retention unknown: how far back the service keeps rows is not known';

/** EPICS alarm severities by their wire number. */
const SEVERITIES = ['NO_ALARM', 'MINOR', 'MAJOR', 'INVALID'];

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
 * A severity as a name: the value read carries the EPICS number.
 * @param {unknown} severity
 * @returns {string}
 */
export function severityText(severity) {
  if (typeof severity === 'number') return SEVERITIES[severity] ?? String(severity);
  return severity === null || severity === undefined ? '' : String(severity);
}

/**
 * A value as one line of text.
 * @param {unknown} value
 * @returns {string}
 */
export function valueText(value) {
  if (value === null || value === undefined) return '';
  return typeof value === 'object' ? JSON.stringify(value) : String(value);
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
 * A section frame with its title; the body goes into the returned element.
 * @param {keyof typeof TITLES} id
 * @param {string} [label] - shown beside the title
 * @returns {HTMLElement}
 */
export function sectionFrame(id, label) {
  const section = node('section', 'pvp-section');
  section.dataset.section = id;
  const head = node('h3', 'osprey-section-label pvp-section-title', TITLES[id]);
  if (label) head.append(node('span', 'pvp-section-aside', label));
  section.append(head);
  return section;
}

// ---- Sections ----

/**
 * @param {PvFactsSection} section
 * @returns {HTMLElement}
 */
function directorySection(section) {
  const frame = sectionFrame('directory');
  if (section.outcome === 'unavailable') {
    frame.append(unavailableLine(section.reason));
    return frame;
  }
  if (section.outcome === 'not_listed') {
    frame.append(node('p', 'pvp-note', `Not listed in ${section.source.label}`));
    return frame;
  }
  const record = /** @type {DirectoryRecord|null|undefined} */ (section.record);
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
  return frame;
}

/**
 * The live alarm state, and beside it the record's own alarm from the value
 * read, which fills in when the read answers.
 * @param {PvFactsSection} section
 * @returns {HTMLElement}
 */
function alarmStateSection(section) {
  const frame = sectionFrame('alarm_state');
  const own = node('span', 'pvp-record-alarm', '…');
  own.dataset.slot = 'record-alarm';
  if (section.outcome === 'unavailable') {
    frame.append(unavailableLine(section.reason));
    frame.append(facts([['Record', own]]));
    return frame;
  }
  const record = /** @type {AlarmStateRecord|null|undefined} */ (section.record);
  /** @type {[string, string | Node][]} */
  const pairs = [];
  if (!record) {
    pairs.push(['Alarm server', 'no state']);
  } else if (record.stale) {
    pairs.push(['Alarm server', STALE_FEED]);
  } else {
    const words = [record.severity, record.status].filter(Boolean).join(' ');
    pairs.push(['Alarm server', words]);
    pairs.push(['Latched', record.latched ? 'yes' : 'no']);
    pairs.push(['Acknowledged', record.acknowledged ? 'yes' : 'no']);
    if (record.message && record.message !== record.status) pairs.push(['Message', record.message]);
    if (record.time) pairs.push(['Since', timeText(record.time)]);
  }
  pairs.push(['Record', own]);
  frame.append(facts(pairs));
  return frame;
}

/**
 * @param {PvFactsSection} section
 * @returns {HTMLElement}
 */
function alarmConfigSection(section) {
  const record = /** @type {AlarmConfigRecord|null|undefined} */ (section.record);
  const frame = sectionFrame('alarm_config', section.label ?? record?.label);
  if (section.outcome === 'unavailable') {
    frame.append(unavailableLine(section.reason));
    return frame;
  }
  if (!record) return frame;
  /** @type {[string, string][]} */
  const pairs = [['In alarm tree', record.in_tree]];
  if (record.path) pairs.push(['Path', record.path]);
  if (record.enabled !== null) pairs.push(['Enabled', record.enabled ? 'yes' : 'no']);
  if (record.latching !== null) pairs.push(['Latching', record.latching ? 'yes' : 'no']);
  if (record.guidance) pairs.push(['Guidance', record.guidance]);
  if (record.displays) pairs.push(['Displays', record.displays]);
  frame.append(facts(pairs));
  return frame;
}

/**
 * The footer of a section: the caveats of its source and, on a history that
 * answered, that only the newest rows are shown or that its retention is not
 * known.
 * @param {PvFactsSection} section
 * @returns {HTMLElement|null}
 */
function sectionFooter(section) {
  const lines = [...section.caveats];
  if (section.truncated) lines.push(`Newest ${HISTORY_LIMIT} rows shown`);
  if (section.outcome !== 'unavailable' && section.retention_known === false) {
    lines.push(RETENTION_UNKNOWN);
  }
  if (!lines.length) return null;
  const footer = node('footer', 'pvp-footer');
  for (const line of lines) footer.append(node('p', 'pvp-note', line));
  return footer;
}

/**
 * Alarm or put history, each row with the user it recorded.
 * @param {PvFactsSection} section
 * @returns {HTMLElement}
 */
function historySection(section) {
  const id = section.section === 'put_history' ? 'put_history' : 'alarm_history';
  const frame = sectionFrame(id);
  if (section.outcome === 'unavailable') {
    frame.append(unavailableLine(section.reason));
  } else if (!section.rows?.length) {
    frame.append(node('p', 'pvp-note', 'No rows recorded'));
  } else if (id === 'put_history') {
    const rows = /** @type {PutRow[]} */ (section.rows);
    frame.append(
      table(
        ['Time', 'Field', 'Old', 'New', 'User', 'Client'],
        rows.map((row) => [
          timeText(row.at),
          row.field ?? '',
          row.old ?? '',
          row.new ?? '',
          row.user ?? '',
          row.client ?? '',
        ])
      )
    );
  } else {
    const rows = /** @type {AlarmHistoryRow[]} */ (section.rows);
    frame.append(
      table(
        ['Time', 'Severity', 'Status', 'Message', 'User'],
        rows.map((row) => [
          timeText(row.time),
          row.severity,
          row.status,
          row.message,
          row.user ?? '',
        ])
      )
    );
  }
  return frame;
}

/**
 * @param {PvFactsSection} section
 * @returns {HTMLElement}
 */
export function factsSection(section) {
  const frame = sectionBody(section);
  const footer = sectionFooter(section);
  if (footer) frame.append(footer);
  return frame;
}

/**
 * @param {PvFactsSection} section
 * @returns {HTMLElement}
 */
function sectionBody(section) {
  switch (section.section) {
    case 'directory':
      return directorySection(section);
    case 'alarm_state':
      return alarmStateSection(section);
    case 'alarm_config':
      return alarmConfigSection(section);
    default:
      return historySection(section);
  }
}
