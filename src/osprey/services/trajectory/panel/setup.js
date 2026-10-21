// @ts-check
/**
 * setup — the Setup BPMs and Setup CMs tabs: families, exclusions and the
 * BPM settings, committed through the setup and solve routes.
 *
 * Each tab shows its family counts from the build's device catalog
 * (`Snapshot.catalog`, each family in ring order) and an
 * exclude list by index in the grammar of `exclude.js`. An exclude field
 * shows the exclusions in force (`Snapshot.solve.config`) and is re-seeded
 * from every snapshot until the operator edits it; from then on it holds the
 * pending, uncommitted list until its tab commits it or Revert drops it.
 * Clicking a device on the Trajectories or Difference plots toggles that BPM
 * in the pending BPM list, clicking a corrector bar on the CM-step plot
 * toggles it in the pending CM list; the raw plots take no clicks.
 *
 * Setup BPMs posts the pending settings and the BPM exclusions (by name) to
 * the setup route; Setup CMs posts the CM exclusions to the solve route. The
 * BPM settings live in the shared model of `settings.js`, which the BPM
 * Control drawer renders too.
 *
 * Mounted into the `setup` area by the composition root (`panel.js`); this
 * module's {@link mount} is its only entry point.
 *
 * @module setup
 */

import { formatExclude, indicesOf, parseExclude, toggleName } from './exclude.js';
import { catalogNames } from './figures.js';
import { settingsFor } from './settings.js';
import { control, h, noticeFromResult } from './views.js';

/** @typedef {'bpms' | 'hcms' | 'vcms'} FamilyKey */
/** @typedef {'bpms' | 'cms'} SetupTab */
/** @typedef {Record<FamilyKey, string | null>} Drafts  The pending exclude text per family, null when unedited. */
/** @typedef {Record<FamilyKey, string>} Texts  The exclude text each field shows. */

/** The setup tabs, by inventory name. */
export const SETUP_TABS = Object.freeze([
  Object.freeze({ name: 'setup-tab-bpms', tab: /** @type {SetupTab} */ ('bpms'), label: 'Setup BPMs' }),
  Object.freeze({ name: 'setup-tab-cms', tab: /** @type {SetupTab} */ ('cms'), label: 'Setup CMs' }),
]);

/** The plots a click on excludes from: BPMs from the corrected-orbit plots, correctors from the CM step. */
export const CLICK_PLOTS = Object.freeze({
  bpms: Object.freeze(['plot-traj-x', 'plot-traj-y', 'plot-diff-x', 'plot-diff-y']),
  cms: Object.freeze(['plot-cm-step']),
});

/** Each family's name in counts and refusals. */
export const FAMILY_LABELS = Object.freeze({ bpms: 'BPM', hcms: 'HCM', vcms: 'VCM' });

/** Why a setup control is disabled without a session. */
export const NO_SESSION = 'Open a session first';

/** Event the Open BPM control button dispatches (bubbling) for the drawer to open on. */
export const OPEN_BPM_CONTROL = 'open-bpm-control';

/** @returns {Drafts} */
export function emptyDrafts() {
  return { bpms: null, hcms: null, vcms: null };
}

/**
 * A family's device names in ring order, empty when the snapshot has no catalog.
 * @param {any} snapshot
 * @param {FamilyKey} key
 * @returns {string[]}
 */
export function familyOf(snapshot, key) {
  return catalogNames(snapshot?.catalog, key);
}

/**
 * The exclusions in force for a family, by name, in family order.
 * @param {any} snapshot
 * @param {FamilyKey} key
 * @returns {string[]}
 */
export function committedNames(snapshot, key) {
  const config = snapshot?.solve?.config;
  const names = key === 'bpms' ? config?.excluded_bpms : config?.excluded_cms;
  const family = familyOf(snapshot, key);
  return indicesOf(names ?? [], family).map((index) => family[index - 1]);
}

/**
 * What each family's exclude field shows from a snapshot alone: the
 * exclusions in force, in the grammar.
 * @param {any} snapshot
 * @returns {Texts}
 */
export function seededTexts(snapshot) {
  /** @param {FamilyKey} key */
  const text = (key) => formatExclude(indicesOf(committedNames(snapshot, key), familyOf(snapshot, key)));
  return { bpms: text('bpms'), hcms: text('hcms'), vcms: text('vcms') };
}

/**
 * What the fields show: each pending text, else its seed.
 * @param {Drafts} drafts
 * @param {Texts} seeds
 * @returns {Texts}
 */
export function shownTexts(drafts, seeds) {
  return { bpms: drafts.bpms ?? seeds.bpms, hcms: drafts.hcms ?? seeds.hcms, vcms: drafts.vcms ?? seeds.vcms };
}

/**
 * A family's field read as names against its family.
 * @param {Texts} texts
 * @param {any} snapshot
 * @param {FamilyKey} key
 * @returns {import('./exclude.js').ExcludeParse}
 */
export function excludeNames(texts, snapshot, key) {
  return parseExclude(texts[key], familyOf(snapshot, key), FAMILY_LABELS[key]);
}

/**
 * Whether a tab holds anything uncommitted.
 * @param {SetupTab} tab
 * @param {Drafts} drafts
 * @param {Record<string, unknown>} [pendingSettings]
 * @returns {boolean}
 */
export function editing(tab, drafts, pendingSettings = {}) {
  if (tab === 'bpms') return drafts.bpms != null || Object.keys(pendingSettings).length > 0;
  return drafts.hcms != null || drafts.vcms != null;
}

/**
 * The count line of a family: its size and how many of it are excluded.
 * @param {any} snapshot
 * @param {FamilyKey} key
 * @returns {string}
 */
export function familyCount(snapshot, key) {
  const size = familyOf(snapshot, key).length;
  const excluded = committedNames(snapshot, key).length;
  return `${size} ${FAMILY_LABELS[key]}s${excluded ? `, ${excluded} excluded` : ''}`;
}

/**
 * The setup route body Setup BPMs sends, or the reason it cannot.
 * @param {Texts} texts
 * @param {any} snapshot
 * @param {Record<string, import('./settings.js').SettingValue>} pendingSettings
 * @returns {{body: {settings: Record<string, import('./settings.js').SettingValue>, excluded_bpms: string[]}} | {error: string}}
 */
export function bpmsBody(texts, snapshot, pendingSettings) {
  const parsed = excludeNames(texts, snapshot, 'bpms');
  if (!parsed.ok) return { error: parsed.error };
  return { body: { settings: { ...pendingSettings }, excluded_bpms: parsed.names } };
}

/**
 * The solve route body Setup CMs sends, or the reason it cannot.
 * @param {Texts} texts
 * @param {any} snapshot
 * @returns {{body: {excluded_cms: string[]}} | {error: string}}
 */
export function cmsBody(texts, snapshot) {
  const hcms = excludeNames(texts, snapshot, 'hcms');
  if (!hcms.ok) return { error: hcms.error };
  const vcms = excludeNames(texts, snapshot, 'vcms');
  if (!vcms.ok) return { error: vcms.error };
  return { body: { excluded_cms: [...hcms.names, ...vcms.names] } };
}

/**
 * The family a clicked device belongs to, among the families a plot excludes from.
 * @param {any} snapshot
 * @param {string} plotId
 * @param {string} device
 * @returns {FamilyKey | null}
 */
export function clickFamily(snapshot, plotId, device) {
  /** @type {FamilyKey[]} */
  const keys = CLICK_PLOTS.bpms.includes(plotId) ? ['bpms'] : CLICK_PLOTS.cms.includes(plotId) ? ['hcms', 'vcms'] : [];
  return keys.find((key) => familyOf(snapshot, key).includes(device)) ?? null;
}

/**
 * A field's text after a click on *device* toggled it; null when the field
 * holds text that does not parse, which a click leaves alone.
 * @param {Texts} texts
 * @param {any} snapshot
 * @param {FamilyKey} key
 * @param {string} device
 * @returns {string | null}
 */
export function toggledText(texts, snapshot, key, device) {
  const parsed = excludeNames(texts, snapshot, key);
  if (!parsed.ok) return null;
  const family = familyOf(snapshot, key);
  return formatExclude(indicesOf(toggleName(parsed.names, device, family), family));
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
 * @property {Drafts} drafts  Which fields hold pending text.
 * @property {Texts} texts  What each field shows.
 * @property {any[]} settings  The `SettingReadback`s, in declared order.
 * @property {Record<string, import('./settings.js').SettingValue>} values
 * @property {Record<string, import('./settings.js').SettingValue>} pending
 * @property {boolean} holding
 */

/**
 * @typedef {object} SetupHandlers
 * @property {(tab: SetupTab) => void} onTab
 * @property {(key: FamilyKey, text: string) => void} onDraftInput  Each keystroke; no redraw.
 * @property {(key: FamilyKey, text: string) => void} onDraft  A committed edit of the field.
 * @property {(id: string, value: import('./settings.js').SettingValue) => void} onSetting
 * @property {(tab: SetupTab) => void} onCommit
 * @property {(tab: SetupTab) => void} onRevert
 * @property {() => void} onOpenBpmControl
 */

/**
 * One family's exclude field with its parse error, if any.
 * @param {SetupModel} model
 * @param {FamilyKey} key
 * @param {string} name  Its inventory name.
 * @param {SetupHandlers} handlers
 * @param {string | null} reason
 * @returns {HTMLElement}
 */
function excludeField(model, key, name, handlers, reason) {
  const parsed = excludeNames(model.texts, model.snapshot, key);
  const input = /** @type {HTMLInputElement} */ (h('input', {
    type: 'text',
    'data-control': name,
    name,
    value: model.texts[key],
    placeholder: 'e.g. [3 7, 10:12]',
    disabled: reason != null,
    title: reason,
    'data-pending': model.drafts[key] != null ? 'true' : null,
  }));
  input.addEventListener('input', () => handlers.onDraftInput(key, input.value));
  input.addEventListener('change', () => handlers.onDraft(key, input.value));
  return h('label', { class: 'setup-exclude' },
    `Exclude ${FAMILY_LABELS[key]}s `,
    input,
    parsed.ok ? null : h('span', { class: 'setup-error', role: 'alert', 'data-error': name, text: parsed.error }));
}

/**
 * The Setup BPMs tab.
 * @param {SetupModel} model
 * @param {SetupHandlers} handlers
 * @returns {HTMLElement}
 */
export function setupBpmsView(model, handlers) {
  const noSession = model.holding ? null : NO_SESSION;
  const hasFamily = familyOf(model.snapshot, 'bpms').length > 0;
  const body = bpmsBody(model.texts, model.snapshot, model.pending);
  const commitReason = noSession ?? (hasFamily ? null : 'No BPM family yet') ?? ('error' in body ? body.error : null);
  const rows = model.settings.map((setting) =>
    h('label', { class: 'setup-setting' }, `${setting.id.replace(/_/g, ' ')} `,
      settingInput(setting.id, model.values[setting.id], (value) => handlers.onSetting(setting.id, value), noSession),
      setting.id in model.pending ? h('span', { class: 'setup-pending', text: 'pending' }) : null));
  return h('section', { class: 'setup-tab', 'data-tab': 'bpms' },
    h('p', { 'data-control': 'bpm-family', class: 'setup-family', text: familyCount(model.snapshot, 'bpms') }),
    excludeField(model, 'bpms', 'bpm-exclude', handlers, noSession),
    ...rows,
    h('div', { class: 'setup-actions' },
      control('setup-bpms', 'Setup BPMs', () => handlers.onCommit('bpms'), commitReason),
      control('setup-bpms-revert', 'Revert', () => handlers.onRevert('bpms'),
        editing('bpms', model.drafts, model.pending) ? null : 'Nothing to revert'),
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
  const hasFamily = familyOf(model.snapshot, 'hcms').length + familyOf(model.snapshot, 'vcms').length > 0;
  const body = cmsBody(model.texts, model.snapshot);
  const commitReason = noSession ?? (hasFamily ? null : 'No corrector family yet') ?? ('error' in body ? body.error : null);
  return h('section', { class: 'setup-tab', 'data-tab': 'cms' },
    h('p', { 'data-control': 'cm-family', class: 'setup-family',
      text: `${familyCount(model.snapshot, 'hcms')}; ${familyCount(model.snapshot, 'vcms')}` }),
    h('fieldset', { 'data-control': 'cm-exclude', class: 'setup-exclude-group' },
      excludeField(model, 'hcms', 'cm-exclude-hcm', handlers, noSession),
      excludeField(model, 'vcms', 'cm-exclude-vcm', handlers, noSession)),
    h('div', { class: 'setup-actions' },
      control('setup-cms', 'Setup CMs', () => handlers.onCommit('cms'), commitReason),
      control('setup-cms-revert', 'Revert', () => handlers.onRevert('cms'),
        editing('cms', model.drafts) ? null : 'Nothing to revert')));
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
  let drafts = emptyDrafts();
  let seeds = seededTexts(ctx.snapshot());
  let deferred = false;

  /** Re-seed each tab's fields from the snapshot unless the tab is being edited. */
  function reseed() {
    const fresh = seededTexts(ctx.snapshot());
    const pending = sync.model.pending();
    if (!editing('bpms', drafts, pending)) seeds = { ...seeds, bpms: fresh.bpms };
    if (!editing('cms', drafts)) seeds = { ...seeds, hcms: fresh.hcms, vcms: fresh.vcms };
  }

  /** Whether the operator is typing in one of this area's fields. */
  function typing() {
    const active = host.ownerDocument?.activeElement;
    return active != null && host.contains(active) && active.tagName === 'INPUT' && active.getAttribute('type') === 'text';
  }

  /** @param {boolean} [force]  Redraw even while the operator types. */
  function render(force = false) {
    reseed();
    if (!force && typing()) {
      deferred = true;
      return;
    }
    deferred = false;
    host.replaceChildren(setupView({
      snapshot: ctx.snapshot(),
      tab,
      drafts,
      texts: shownTexts(drafts, seeds),
      settings: sync.model.readbacks()?.settings ?? [],
      values: sync.model.values(),
      pending: sync.model.pending(),
      holding: ctx.holding(),
    }, handlers));
  }

  /** @param {SetupTab} which */
  async function commit(which) {
    const snapshot = ctx.snapshot();
    const texts = shownTexts(drafts, seeds);
    if (which === 'bpms') {
      const built = bpmsBody(texts, snapshot, sync.model.pending());
      if ('error' in built) return ctx.notify({ kind: 'refused', title: 'setup BPMs refused', detail: built.error });
      const sent = drafts.bpms;
      const result = await sync.commit(built.body);
      ctx.notify(noticeFromResult('setup BPMs', result));
      if (result.kind === 'ok' && drafts.bpms === sent) drafts = { ...drafts, bpms: null };
    } else {
      const built = cmsBody(texts, snapshot);
      if ('error' in built) return ctx.notify({ kind: 'refused', title: 'setup CMs refused', detail: built.error });
      const sent = { hcms: drafts.hcms, vcms: drafts.vcms };
      const result = await ctx.api.solve(built.body);
      ctx.notify(noticeFromResult('setup CMs', result));
      if (result.kind === 'ok' && drafts.hcms === sent.hcms && drafts.vcms === sent.vcms) {
        drafts = { ...drafts, hcms: null, vcms: null };
      }
    }
    render(true);
  }

  /** @type {SetupHandlers} */
  const handlers = {
    onTab(next) {
      tab = next;
      render(true);
    },
    onDraftInput(key, text) {
      drafts = { ...drafts, [key]: text };
    },
    onDraft(key, text) {
      drafts = { ...drafts, [key]: text };
      render(true);
    },
    onSetting(id, value) {
      sync.model.set(id, value);
      render(true);
    },
    onCommit: (which) => void commit(which),
    onRevert(which) {
      if (which === 'bpms') {
        drafts = { ...drafts, bpms: null };
        sync.model.revert();
      } else {
        drafts = { ...drafts, hcms: null, vcms: null };
      }
      render(true);
    },
    onOpenBpmControl() {
      host.dispatchEvent(new CustomEvent(OPEN_BPM_CONTROL, { bubbles: true, composed: true }));
    },
  };

  /** @param {string} plotId @param {string} device */
  function onClick(plotId, device) {
    const snapshot = ctx.snapshot();
    const key = clickFamily(snapshot, plotId, device);
    if (key == null) return;
    const text = toggledText(shownTexts(drafts, seeds), snapshot, key, device);
    if (text == null) return;
    drafts = { ...drafts, [key]: text };
    tab = key === 'bpms' ? 'bpms' : 'cms';
    render(true);
  }

  for (const plotId of [...CLICK_PLOTS.bpms, ...CLICK_PLOTS.cms]) {
    ctx.adapter?.onPointClick?.(plotId, (/** @type {string} */ device) => onClick(plotId, device));
  }
  ctx.onSnapshot(() => render());
  sync.model.on(() => render());
  host.addEventListener('focusout', () => {
    if (deferred) queueMicrotask(() => render());
  });
  render(true);
};
