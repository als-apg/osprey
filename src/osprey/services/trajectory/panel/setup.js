// @ts-check
/**
 * setup — the Setup BPMs and Setup CMs tabs: a map of the machine to select
 * devices on, and the BPM settings, committed through the setup and solve
 * routes.
 *
 * Every device, place and word on the map comes from the build's device
 * catalog (`Snapshot.catalog`, each family in position order); this module names
 * none. A tab draws one row per leaf place — each device's own place — in
 * position order, devices without a place sharing one row labelled by the model.
 * Clicking a place name toggles every device in it, clicking a square toggles
 * one device; Setup CMs draws a horizontal and a vertical strip per row.
 *
 * Selection is "everything in, minus exclusions", by matrix name. The
 * exclusions in force are the solve configuration's (`Snapshot.solve.config`);
 * a map click, a Find token, Start from or a click on a Trajectories,
 * Difference or CM-step plot only marks a pending change, held until its tab
 * commits it (Setup BPMs posts the settings and BPM exclusions to the setup
 * route, Setup CMs the corrector exclusions to the solve route) or Revert
 * drops it. Nothing pending ever touches a computed step.
 *
 * Find is the design system's typeahead over the catalog's labels and place
 * ids: a token naming a device by label, name or id toggles it, one naming a
 * place toggles every device in it, and anything else — positions in a list
 * included — is refused by name. Start from appears only when the catalog's
 * groups offer two or more different device sets for the tab; choosing one
 * makes every device outside it the pending exclusions.
 *
 * The BPM settings live in the shared model of `settings.js`, which the BPM
 * Control drawer renders too.
 *
 * Mounted into the `setup` area by the composition root (`panel.js`); this
 * module's {@link mount} is its only entry point.
 *
 * @module setup
 */

import { attachChannelCombobox } from '/design-system/js/channel-combobox.js';
import { catalogNames, deviceHover, placeRows } from './figures.js';
import { settingsFor } from './settings.js';
import { control, h, noticeFromResult } from './views.js';

/** @typedef {'bpms' | 'hcms' | 'vcms'} FamilyKey */
/** @typedef {'bpms' | 'cms'} SetupTab */
/** @typedef {import('./figures.js').CatalogEntry} CatalogEntry */
/** @typedef {CatalogEntry & {family: FamilyKey}} TabEntry */
/** @typedef {Record<SetupTab, string[] | null>} Pending  Each tab's pending exclusions by name, null when none. */
/** @typedef {Record<SetupTab, string>} FindTexts */
/** @typedef {Record<SetupTab, string | null>} FindErrors */

/** The setup tabs, by inventory name. */
export const SETUP_TABS = Object.freeze([
  Object.freeze({ name: 'setup-tab-bpms', tab: /** @type {SetupTab} */ ('bpms'), label: 'Setup BPMs' }),
  Object.freeze({ name: 'setup-tab-cms', tab: /** @type {SetupTab} */ ('cms'), label: 'Setup CMs' }),
]);

/** The families each tab selects among. */
export const TAB_FAMILIES = Object.freeze({
  bpms: Object.freeze(/** @type {FamilyKey[]} */ (['bpms'])),
  cms: Object.freeze(/** @type {FamilyKey[]} */ (['hcms', 'vcms'])),
});

/** The plots a click on excludes from: BPMs from the corrected-orbit plots, correctors from the CM step. */
export const CLICK_PLOTS = Object.freeze({
  bpms: Object.freeze(['plot-traj-x', 'plot-traj-y', 'plot-diff-x', 'plot-diff-y']),
  cms: Object.freeze(['plot-cm-step']),
});

/** Each family's name in counts and refusals. */
export const FAMILY_LABELS = Object.freeze({ bpms: 'BPM', hcms: 'HCM', vcms: 'VCM' });

/** Each corrector family's strip letter on the map. */
export const STRIP_LABELS = Object.freeze({ hcms: 'H', vcms: 'V' });

/** Why a setup control is disabled without a session. */
export const NO_SESSION = 'Open a session first';

/** Event the Open BPM control button dispatches (bubbling) for the drawer to open on. */
export const OPEN_BPM_CONTROL = 'open-bpm-control';

/** The Find box's inventory name per tab. */
export const FIND_CONTROLS = Object.freeze({ bpms: 'bpm-exclude', cms: 'cm-exclude' });

/** A token that reads as a list of positions: digits, ranges, brackets and separators only. */
const POSITIONS = /^[\s[\]\d:,]+$/;

/** @returns {Pending} */
export function emptyPending() {
  return { bpms: null, cms: null };
}

/**
 * A tab's devices, family by family in position order, each tagged with its
 * family; empty when the snapshot has no catalog.
 * @param {any} snapshot
 * @param {SetupTab} tab
 * @returns {TabEntry[]}
 */
export function tabEntries(snapshot, tab) {
  return TAB_FAMILIES[tab].flatMap((family) => {
    const entries = snapshot?.catalog?.[family];
    return Array.isArray(entries) ? entries.map((entry) => ({ ...entry, family })) : [];
  });
}

/**
 * *names* restricted to the tab's devices, in the tab's order.
 * @param {any} snapshot
 * @param {SetupTab} tab
 * @param {Iterable<string>} names
 * @returns {string[]}
 */
function inTabOrder(snapshot, tab, names) {
  const wanted = new Set(names);
  return tabEntries(snapshot, tab).map((entry) => entry.name).filter((name) => wanted.has(name));
}

/**
 * The exclusions in force for a tab, by name, in tab order.
 * @param {any} snapshot
 * @param {SetupTab} tab
 * @returns {string[]}
 */
export function committedExcluded(snapshot, tab) {
  const config = snapshot?.solve?.config;
  const names = tab === 'bpms' ? config?.excluded_bpms : config?.excluded_cms;
  return inTabOrder(snapshot, tab, Array.isArray(names) ? names : []);
}

/**
 * The exclusions a tab shows: its pending ones, else those in force.
 * @param {Pending} pending
 * @param {any} snapshot
 * @param {SetupTab} tab
 * @returns {string[]}
 */
export function shownExcluded(pending, snapshot, tab) {
  return pending[tab] ?? committedExcluded(snapshot, tab);
}

/**
 * Whether two name lists hold the same names, in any order.
 * @param {readonly string[]} a
 * @param {readonly string[]} b
 * @returns {boolean}
 */
export function sameSet(a, b) {
  const left = new Set(a);
  const right = new Set(b);
  return left.size === right.size && [...left].every((name) => right.has(name));
}

/**
 * *pending* with the tab's exclusions set to *names*; null again when that is
 * what is in force.
 * @param {Pending} pending
 * @param {any} snapshot
 * @param {SetupTab} tab
 * @param {Iterable<string>} names
 * @returns {Pending}
 */
export function withExcluded(pending, snapshot, tab, names) {
  const next = inTabOrder(snapshot, tab, names);
  return { ...pending, [tab]: sameSet(next, committedExcluded(snapshot, tab)) ? null : next };
}

/**
 * *excluded* after toggling *names* as one: when every one of them is in use
 * they all go out, otherwise they all come back in.
 * @param {readonly string[]} excluded
 * @param {readonly string[]} names
 * @returns {string[]}
 */
export function toggleNames(excluded, names) {
  const next = new Set(excluded);
  const allIn = names.every((name) => !next.has(name));
  for (const name of names) {
    if (allIn) next.add(name);
    else next.delete(name);
  }
  return [...next];
}

/**
 * What a Find token names: devices by label, name or id, or every device of
 * a place by its id; matched without regard to case. A token matching
 * nothing, or reading as positions in a list, is refused by name.
 * @param {any} snapshot
 * @param {SetupTab} tab
 * @param {string} token
 * @returns {{names: string[]} | {error: string}}
 */
export function resolveToken(snapshot, tab, token) {
  const wanted = token.trim().toLowerCase();
  const entries = tabEntries(snapshot, tab);
  const devices = entries.filter((entry) =>
    [entry.label, entry.name, entry.device].some((key) => typeof key === 'string' && key.toLowerCase() === wanted));
  if (devices.length) return { names: devices.map((entry) => entry.name) };
  const placed = entries.filter((entry) =>
    (entry.places ?? []).some((place) => typeof place?.id === 'string' && place.id.toLowerCase() === wanted));
  if (placed.length) return { names: placed.map((entry) => entry.name) };
  if (POSITIONS.test(token)) {
    return { error: `"${token.trim()}" is a position, not a device: find by label, name or place` };
  }
  return { error: `"${token.trim()}" names no ${tab === 'bpms' ? 'BPM' : 'corrector'} and no place` };
}

/**
 * The tokens of a Find entry: comma-separated, trimmed, empty ones dropped.
 * A bracketed list is one token, so it is refused whole.
 * @param {string} text
 * @returns {string[]}
 */
export function findTokens(text) {
  const trimmed = String(text ?? '').trim();
  if (trimmed.startsWith('[') || trimmed.endsWith(']')) return trimmed ? [trimmed] : [];
  return trimmed.split(',').map((token) => token.trim()).filter(Boolean);
}

/**
 * The tab's exclusions after a Find entry toggled each of its tokens, or the
 * refusal of the first token that names nothing (then nothing is toggled).
 * @param {readonly string[]} excluded
 * @param {any} snapshot
 * @param {SetupTab} tab
 * @param {string} text
 * @returns {{excluded: string[]} | {error: string}}
 */
export function applyFind(excluded, snapshot, tab, text) {
  const tokens = findTokens(text);
  if (tokens.length === 0) return { error: 'Type a label, name or place' };
  /** @type {string[][]} */
  const resolved = [];
  for (const token of tokens) {
    const found = resolveToken(snapshot, tab, token);
    if ('error' in found) return found;
    resolved.push(found.names);
  }
  return { excluded: resolved.reduce((acc, names) => toggleNames(acc, names), [...excluded]) };
}

/**
 * What Find suggests: every device label of the tab and every place id, once each.
 * @param {any} snapshot
 * @param {SetupTab} tab
 * @returns {string[]}
 */
export function findSuggestions(snapshot, tab) {
  /** @type {Set<string>} */
  const out = new Set();
  const entries = tabEntries(snapshot, tab);
  for (const entry of entries) out.add(entry.label || entry.name);
  for (const entry of entries) for (const place of entry.places ?? []) if (place?.id) out.add(place.id);
  return [...out];
}

/**
 * The catalog groups Start from offers for a tab: those covering a non-empty
 * set of the tab's devices, one per distinct set. Empty — and Start from
 * hidden — when fewer than two distinct sets remain.
 * @param {any} snapshot
 * @param {SetupTab} tab
 * @returns {{id: string, label: string, members: string[]}[]}
 */
export function startFromGroups(snapshot, tab) {
  const groups = Array.isArray(snapshot?.catalog?.groups) ? snapshot.catalog.groups : [];
  /** @type {{id: string, label: string, members: string[]}[]} */
  const offered = [];
  for (const group of groups) {
    const members = inTabOrder(snapshot, tab, Array.isArray(group?.members) ? group.members : []);
    if (members.length === 0 || offered.some((seen) => sameSet(seen.members, members))) continue;
    offered.push({ id: String(group.id), label: String(group.label || group.id), members });
  }
  return offered.length >= 2 ? offered : [];
}

/**
 * The exclusions Start from sets for a group: every tab device outside it.
 * @param {any} snapshot
 * @param {SetupTab} tab
 * @param {string} groupId
 * @returns {string[] | null}  null when the group is not offered.
 */
export function startFromExcluded(snapshot, tab, groupId) {
  const group = startFromGroups(snapshot, tab).find((offered) => offered.id === groupId);
  if (!group) return null;
  const inside = new Set(group.members);
  return tabEntries(snapshot, tab).map((entry) => entry.name).filter((name) => !inside.has(name));
}

/**
 * The count line of a family: how many of it are in use.
 * @param {any} snapshot
 * @param {FamilyKey} key
 * @param {readonly string[]} excluded
 * @returns {string}
 */
export function familyCount(snapshot, key, excluded) {
  const names = catalogNames(snapshot?.catalog, key);
  const used = names.filter((name) => !excluded.includes(name)).length;
  return `${used} of ${names.length} ${FAMILY_LABELS[key]}s in use`;
}

/**
 * The pending change of a tab, by device label: what goes out and what comes back.
 * @param {Pending} pending
 * @param {any} snapshot
 * @param {SetupTab} tab
 * @returns {{out: string[], back: string[]}}
 */
export function pendingChange(pending, snapshot, tab) {
  const shown = new Set(shownExcluded(pending, snapshot, tab));
  const committed = new Set(committedExcluded(snapshot, tab));
  const entries = tabEntries(snapshot, tab);
  const label = (/** @type {TabEntry} */ entry) => entry.label || entry.name;
  return {
    out: entries.filter((entry) => shown.has(entry.name) && !committed.has(entry.name)).map(label),
    back: entries.filter((entry) => !shown.has(entry.name) && committed.has(entry.name)).map(label),
  };
}

/**
 * Whether a tab holds anything uncommitted.
 * @param {SetupTab} tab
 * @param {Pending} pending
 * @param {Record<string, unknown>} [pendingSettings]
 * @returns {boolean}
 */
export function editing(tab, pending, pendingSettings = {}) {
  if (tab === 'bpms') return pending.bpms != null || Object.keys(pendingSettings).length > 0;
  return pending.cms != null;
}

/**
 * Why a tab's exclusions cannot be committed — no devices, or a family left
 * with none in use — or null when they can.
 * @param {any} snapshot
 * @param {SetupTab} tab
 * @param {readonly string[]} excluded
 * @returns {string | null}
 */
function emptyFamily(snapshot, tab, excluded) {
  if (tabEntries(snapshot, tab).length === 0) return `No ${tab === 'bpms' ? 'BPMs' : 'correctors'} in the catalog yet`;
  const empty = TAB_FAMILIES[tab].find((key) => {
    const names = catalogNames(snapshot?.catalog, key);
    return names.length > 0 && names.every((name) => excluded.includes(name));
  });
  return empty ? `Keep at least one ${FAMILY_LABELS[empty]} in use` : null;
}

/**
 * The setup route body Setup BPMs sends, or the reason it cannot.
 * @param {Pending} pending
 * @param {any} snapshot
 * @param {Record<string, import('./settings.js').SettingValue>} pendingSettings
 * @returns {{body: {settings: Record<string, import('./settings.js').SettingValue>, excluded_bpms: string[]}} | {error: string}}
 */
export function bpmsBody(pending, snapshot, pendingSettings) {
  const excluded = shownExcluded(pending, snapshot, 'bpms');
  const error = emptyFamily(snapshot, 'bpms', excluded);
  return error ? { error } : { body: { settings: { ...pendingSettings }, excluded_bpms: excluded } };
}

/**
 * The solve route body Setup CMs sends, or the reason it cannot.
 * @param {Pending} pending
 * @param {any} snapshot
 * @returns {{body: {excluded_cms: string[]}} | {error: string}}
 */
export function cmsBody(pending, snapshot) {
  const excluded = shownExcluded(pending, snapshot, 'cms');
  const error = emptyFamily(snapshot, 'cms', excluded);
  return error ? { error } : { body: { excluded_cms: excluded } };
}

/**
 * The tab a clicked device belongs to, among the plots that exclude.
 * @param {any} snapshot
 * @param {string} plotId
 * @param {string} device
 * @returns {SetupTab | null}
 */
export function clickTab(snapshot, plotId, device) {
  /** @type {SetupTab | null} */
  const tab = CLICK_PLOTS.bpms.includes(plotId) ? 'bpms' : CLICK_PLOTS.cms.includes(plotId) ? 'cms' : null;
  if (tab == null) return null;
  return tabEntries(snapshot, tab).some((entry) => entry.name === device) ? tab : null;
}

/**
 * The input a setting is edited with, typed after its value.
 * @param {string} id
 * @param {import('./settings.js').SettingValue | undefined} value
 * @param {(value: import('./settings.js').SettingValue) => void} onchange
 * @param {string | null} reason
 * @returns {HTMLInputElement}
 */
export function settingInput(id, value, onchange, reason) {
  const name = `setting-${id.replace(/_/g, '-')}`;
  const kind = typeof value === 'boolean' ? 'checkbox' : typeof value === 'number' ? 'number' : 'text';
  const input = /** @type {HTMLInputElement} */ (h('input', {
    type: kind,
    'data-control': name,
    name,
    disabled: reason != null,
    title: reason,
    placeholder: value == null ? 'unread or mixed' : null,
  }));
  if (kind === 'checkbox') input.checked = Boolean(value);
  else if (value != null) input.value = String(value);
  input.addEventListener('change', () => {
    if (kind === 'checkbox') onchange(input.checked);
    else if (kind === 'number') {
      const number = Number(input.value);
      if (input.value.trim() !== '' && Number.isFinite(number)) onchange(number);
    } else onchange(input.value);
  });
  return input;
}

/**
 * @typedef {object} SetupModel
 * @property {any} snapshot
 * @property {SetupTab} tab
 * @property {Pending} pending  Each tab's pending exclusions.
 * @property {FindTexts} find  What each Find box holds.
 * @property {FindErrors} findErrors  Each Find box's last refusal.
 * @property {any[]} settings  The `SettingReadback`s, in declared order.
 * @property {Record<string, import('./settings.js').SettingValue>} values
 * @property {Record<string, import('./settings.js').SettingValue>} settingsPending
 * @property {boolean} holding
 */

/**
 * @typedef {object} SetupHandlers
 * @property {(tab: SetupTab) => void} onTab
 * @property {(tab: SetupTab, names: string[]) => void} onToggle  A place or a device was clicked.
 * @property {(tab: SetupTab, text: string) => void} onFindInput  Each keystroke; no redraw.
 * @property {(tab: SetupTab, text: string) => void} onFind  A Find entry to apply.
 * @property {(tab: SetupTab, groupId: string) => void} onStartFrom
 * @property {(id: string, value: import('./settings.js').SettingValue) => void} onSetting
 * @property {(tab: SetupTab) => void} onCommit
 * @property {(tab: SetupTab) => void} onRevert
 * @property {() => void} onOpenBpmControl
 */

/**
 * One device's square: in use, out, and whether that is a pending change.
 * @param {TabEntry} entry
 * @param {Set<string>} shown  The exclusions shown.
 * @param {Set<string>} committed  The exclusions in force.
 * @param {SetupTab} tab
 * @param {SetupHandlers} handlers
 * @param {string | null} reason
 * @returns {HTMLElement}
 */
function square(entry, shown, committed, tab, handlers, reason) {
  const out = shown.has(entry.name);
  const changed = out !== committed.has(entry.name);
  return h('button', {
    type: 'button',
    class: 'map-device',
    'data-device': entry.name,
    'data-state': out ? 'out' : 'in',
    'data-pending': changed ? 'true' : null,
    'aria-pressed': String(!out),
    'aria-label': deviceHover(entry.name, entry),
    title: reason ?? deviceHover(entry.name, entry),
    disabled: reason != null,
    onclick: () => handlers.onToggle(tab, [entry.name]),
  });
}

/**
 * The map of a tab: one row per leaf place, its name toggling the place.
 * @param {SetupModel} model
 * @param {SetupTab} tab
 * @param {SetupHandlers} handlers
 * @param {string | null} reason
 * @returns {HTMLElement}
 */
export function mapView(model, tab, handlers, reason) {
  const shown = new Set(shownExcluded(model.pending, model.snapshot, tab));
  const committed = new Set(committedExcluded(model.snapshot, tab));
  const rows = placeRows(tabEntries(model.snapshot, tab), model.snapshot?.model);
  return h('div', { class: 'device-map', 'data-map': tab, role: 'group' },
    ...rows.map((row) => {
      const used = row.entries.filter((entry) => !shown.has(entry.name)).length;
      const strip = (/** @type {TabEntry[]} */ entries) =>
        h('span', { class: 'map-strip' }, ...entries.map((entry) => square(entry, shown, committed, tab, handlers, reason)));
      const strips = tab === 'bpms'
        ? [strip(row.entries)]
        : TAB_FAMILIES.cms.map((family) => {
          const entries = row.entries.filter((entry) => entry.family === family);
          return entries.length
            ? h('span', { class: 'map-plane', 'data-plane': family },
              h('span', { class: 'map-plane-label', text: STRIP_LABELS[/** @type {'hcms'|'vcms'} */ (family)] }), strip(entries))
            : null;
        });
      return h('div', { class: 'map-row', 'data-place': row.id },
        h('button', { type: 'button', class: 'map-place', 'data-place-toggle': row.id, disabled: reason != null,
          title: reason, onclick: () => handlers.onToggle(tab, row.entries.map((entry) => entry.name)) }, row.label),
        ...strips,
        h('span', { class: 'map-count', text: `${used}/${row.entries.length}` }));
    }));
}

/**
 * A tab's Find box and its refusal, if any.
 * @param {SetupModel} model
 * @param {SetupTab} tab
 * @param {SetupHandlers} handlers
 * @param {string | null} reason
 * @returns {HTMLElement}
 */
function findField(model, tab, handlers, reason) {
  const name = FIND_CONTROLS[tab];
  const input = /** @type {HTMLInputElement} */ (h('input', {
    type: 'text', 'data-control': name, name, value: model.find[tab], placeholder: 'label, name or place',
    autocomplete: 'off', disabled: reason != null, title: reason, 'aria-invalid': model.findErrors[tab] ? 'true' : null,
  }));
  input.addEventListener('input', () => handlers.onFindInput(tab, input.value));
  input.addEventListener('keydown', (event) => {
    // An arrow-armed suggestion is the typeahead's to accept; a plain Enter applies the text.
    if (event.key === 'Enter' && !input.hasAttribute('aria-activedescendant')) handlers.onFind(tab, input.value);
  });
  const error = model.findErrors[tab];
  return h('label', { class: 'setup-find' }, 'Find ', input,
    error ? h('span', { class: 'setup-error', role: 'alert', 'data-error': name, text: error }) : null);
}

/**
 * A tab's Start from picker, or null when the catalog offers fewer than two sets.
 * @param {SetupModel} model
 * @param {SetupTab} tab
 * @param {SetupHandlers} handlers
 * @param {string | null} reason
 * @returns {HTMLElement | null}
 */
function startFromField(model, tab, handlers, reason) {
  const groups = startFromGroups(model.snapshot, tab);
  if (groups.length === 0) return null;
  const select = /** @type {HTMLSelectElement} */ (h('select', {
    'data-control': `${tab === 'bpms' ? 'bpm' : 'cm'}-start-from`,
    disabled: reason != null,
    title: reason,
  },
  h('option', { value: '', text: 'choose a set…' }),
  ...groups.map((group) => h('option', { value: group.id, text: group.label }))));
  select.addEventListener('change', () => {
    if (select.value) handlers.onStartFrom(tab, select.value);
  });
  return h('label', { class: 'setup-start-from' }, 'Start from ', select);
}

/**
 * A tab's pending line: what goes out and what comes back.
 * @param {SetupModel} model
 * @param {SetupTab} tab
 * @returns {HTMLElement | null}
 */
function pendingLine(model, tab) {
  const { out, back } = pendingChange(model.pending, model.snapshot, tab);
  const parts = [...out.map((label) => `−${label}`), ...back.map((label) => `+${label}`)];
  if (parts.length === 0) return null;
  return h('p', { class: 'setup-pending-change', 'data-pending-change': tab, text: `pending: ${parts.join(' ')}` });
}

/**
 * The Setup BPMs tab.
 * @param {SetupModel} model
 * @param {SetupHandlers} handlers
 * @returns {HTMLElement}
 */
export function setupBpmsView(model, handlers) {
  const noSession = model.holding ? null : NO_SESSION;
  const body = bpmsBody(model.pending, model.snapshot, model.settingsPending);
  const commitReason = noSession ?? ('error' in body ? body.error : null);
  const excluded = shownExcluded(model.pending, model.snapshot, 'bpms');
  const rows = model.settings.map((setting) =>
    h('label', { class: 'setup-setting' }, `${setting.id.replace(/_/g, ' ')} `,
      settingInput(setting.id, model.values[setting.id], (value) => handlers.onSetting(setting.id, value), noSession),
      setting.id in model.settingsPending ? h('span', { class: 'setup-pending', text: 'pending' }) : null));
  return h('section', { class: 'setup-tab', 'data-tab': 'bpms' },
    h('p', { 'data-control': 'bpm-family', class: 'setup-family', text: familyCount(model.snapshot, 'bpms', excluded) }),
    startFromField(model, 'bpms', handlers, noSession),
    findField(model, 'bpms', handlers, noSession),
    mapView(model, 'bpms', handlers, noSession),
    ...rows,
    pendingLine(model, 'bpms'),
    h('div', { class: 'setup-actions' },
      control('setup-bpms', 'Setup BPMs', () => handlers.onCommit('bpms'), commitReason),
      control('setup-bpms-revert', 'Revert', () => handlers.onRevert('bpms'),
        editing('bpms', model.pending, model.settingsPending) ? null : 'Nothing to revert'),
      control('open-bpm-control', 'Open BPM control', () => handlers.onOpenBpmControl())));
}

/**
 * The Setup CMs tab.
 * @param {SetupModel} model
 * @param {SetupHandlers} handlers
 * @returns {HTMLElement}
 */
export function setupCmsView(model, handlers) {
  const noSession = model.holding ? null : NO_SESSION;
  const body = cmsBody(model.pending, model.snapshot);
  const commitReason = noSession ?? ('error' in body ? body.error : null);
  const excluded = shownExcluded(model.pending, model.snapshot, 'cms');
  const count = (/** @type {'hcms'|'vcms'} */ key) =>
    `${STRIP_LABELS[key]} ${familyCount(model.snapshot, key, excluded).replace(` ${FAMILY_LABELS[key]}s in use`, '')}`;
  return h('section', { class: 'setup-tab', 'data-tab': 'cms' },
    h('p', { 'data-control': 'cm-family', class: 'setup-family', text: `${count('hcms')} · ${count('vcms')} in use` }),
    startFromField(model, 'cms', handlers, noSession),
    findField(model, 'cms', handlers, noSession),
    mapView(model, 'cms', handlers, noSession),
    pendingLine(model, 'cms'),
    h('div', { class: 'setup-actions' },
      control('setup-cms', 'Setup CMs', () => handlers.onCommit('cms'), commitReason),
      control('setup-cms-revert', 'Revert', () => handlers.onRevert('cms'),
        editing('cms', model.pending) ? null : 'Nothing to revert')));
}

/**
 * Both tabs with their tab strip; only the selected one shows.
 * @param {SetupModel} model
 * @param {SetupHandlers} handlers
 * @returns {HTMLElement}
 */
export function setupView(model, handlers) {
  const strip = h('div', { class: 'setup-tabs', role: 'tablist' },
    ...SETUP_TABS.map((entry) => {
      const button = control(entry.name, entry.label, () => handlers.onTab(entry.tab));
      button.setAttribute('role', 'tab');
      button.setAttribute('aria-selected', String(entry.tab === model.tab));
      return button;
    }));
  const bpms = setupBpmsView(model, handlers);
  const cms = setupCmsView(model, handlers);
  bpms.hidden = model.tab !== 'bpms';
  cms.hidden = model.tab !== 'cms';
  return h('div', { class: 'setup' }, strip, bpms, cms);
}

/**
 * Fill the `setup` area.
 * @type {import('./panel.js').Mount}
 */
export const mount = (host, ctx) => {
  const sync = settingsFor(ctx);
  /** @type {SetupTab} */
  let tab = 'bpms';
  let pending = emptyPending();
  /** @type {FindTexts} */
  let find = { bpms: '', cms: '' };
  /** @type {FindErrors} */
  let findErrors = { bpms: null, cms: null };
  let deferred = false;
  /** @type {{close: () => void, destroy: () => void}[]} */
  let comboboxes = [];

  /** Whether the operator is typing in one of this area's fields. */
  function typing() {
    const active = host.ownerDocument?.activeElement;
    return active != null && host.contains(active) && active.tagName === 'INPUT' && active.getAttribute('type') === 'text';
  }

  /** Give each drawn Find box its typeahead over the catalog's labels and places. */
  function attachFind() {
    for (const handle of comboboxes) handle.destroy();
    comboboxes = [];
    for (const which of /** @type {SetupTab[]} */ (['bpms', 'cms'])) {
      const input = /** @type {HTMLInputElement | null} */ (host.querySelector(`[data-control="${FIND_CONTROLS[which]}"]`));
      if (input == null || input.disabled) continue;
      comboboxes.push(attachChannelCombobox(input, () => findSuggestions(ctx.snapshot(), which), {
        // Applied once the typeahead has finished accepting, since applying redraws the box.
        onSelect: (value) => queueMicrotask(() => handlers.onFind(which, value)),
      }));
    }
  }

  /**
   * @param {boolean} [force]  Redraw even while the operator types.
   * @param {SetupTab | null} [focusFind]  The tab whose Find box keeps the focus.
   */
  function render(force = false, focusFind = null) {
    if (!force && typing()) {
      deferred = true;
      return;
    }
    deferred = false;
    for (const handle of comboboxes) handle.close();
    host.replaceChildren(setupView({
      snapshot: ctx.snapshot(),
      tab,
      pending,
      find,
      findErrors,
      settings: sync.model.readbacks()?.settings ?? [],
      values: sync.model.values(),
      settingsPending: sync.model.pending(),
      holding: ctx.holding(),
    }, handlers));
    attachFind();
    if (focusFind) {
      /** @type {HTMLInputElement | null} */ (host.querySelector(`[data-control="${FIND_CONTROLS[focusFind]}"]`))?.focus();
    }
  }

  /**
   * Mark *names* as the tab's pending exclusions.
   * @param {SetupTab} which
   * @param {Iterable<string>} names
   */
  function stage(which, names) {
    pending = withExcluded(pending, ctx.snapshot(), which, names);
  }

  /** @param {SetupTab} which */
  async function commit(which) {
    const snapshot = ctx.snapshot();
    const sent = pending[which];
    if (which === 'bpms') {
      const built = bpmsBody(pending, snapshot, sync.model.pending());
      if ('error' in built) return ctx.notify({ kind: 'refused', title: 'setup BPMs refused', detail: built.error });
      const result = await sync.commit(built.body);
      ctx.notify(noticeFromResult('setup BPMs', result));
      if (result.kind === 'ok' && pending.bpms === sent) pending = { ...pending, bpms: null };
    } else {
      const built = cmsBody(pending, snapshot);
      if ('error' in built) return ctx.notify({ kind: 'refused', title: 'setup CMs refused', detail: built.error });
      const result = await ctx.api.solve(built.body);
      ctx.notify(noticeFromResult('setup CMs', result));
      if (result.kind === 'ok' && pending.cms === sent) pending = { ...pending, cms: null };
    }
    render(true);
  }

  /** @type {SetupHandlers} */
  const handlers = {
    onTab(next) {
      tab = next;
      render(true);
    },
    onToggle(which, names) {
      stage(which, toggleNames(shownExcluded(pending, ctx.snapshot(), which), names));
      render(true);
    },
    onFindInput(which, text) {
      find = { ...find, [which]: text };
    },
    onFind(which, text) {
      const snapshot = ctx.snapshot();
      const found = applyFind(shownExcluded(pending, snapshot, which), snapshot, which, text);
      if ('error' in found) {
        find = { ...find, [which]: text };
        findErrors = { ...findErrors, [which]: found.error };
      } else {
        stage(which, found.excluded);
        find = { ...find, [which]: '' };
        findErrors = { ...findErrors, [which]: null };
      }
      render(true, which);
    },
    onStartFrom(which, groupId) {
      const excluded = startFromExcluded(ctx.snapshot(), which, groupId);
      if (excluded != null) stage(which, excluded);
      render(true);
    },
    onSetting(id, value) {
      sync.model.set(id, value);
      render(true);
    },
    onCommit: (which) => void commit(which),
    onRevert(which) {
      pending = { ...pending, [which]: null };
      findErrors = { ...findErrors, [which]: null };
      if (which === 'bpms') sync.model.revert();
      render(true);
    },
    onOpenBpmControl() {
      host.dispatchEvent(new CustomEvent(OPEN_BPM_CONTROL, { bubbles: true, composed: true }));
    },
  };

  /** @param {string} plotId @param {string} device */
  function onClick(plotId, device) {
    const snapshot = ctx.snapshot();
    const which = clickTab(snapshot, plotId, device);
    if (which == null) return;
    stage(which, toggleNames(shownExcluded(pending, snapshot, which), [device]));
    tab = which;
    render(true);
  }

  for (const plotId of [...CLICK_PLOTS.bpms, ...CLICK_PLOTS.cms]) {
    ctx.adapter?.onPointClick?.(plotId, (/** @type {string} */ device) => onClick(plotId, device));
  }
  ctx.onSnapshot(() => {
    // A commit that lands as the configuration in force leaves nothing pending.
    for (const which of /** @type {SetupTab[]} */ (['bpms', 'cms'])) {
      const held = pending[which];
      if (held != null) pending = withExcluded(pending, ctx.snapshot(), which, held);
    }
    render();
  });
  sync.model.on(() => render());
  host.addEventListener('focusout', () => {
    if (deferred) queueMicrotask(() => render());
  });
  render(true);
};
