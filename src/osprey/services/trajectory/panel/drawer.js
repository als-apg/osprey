// @ts-check
/**
 * drawer — the BPM Control drawer: the shared settings model, the stored
 * current and the per-BPM readbacks.
 *
 * The drawer is the page's `<osprey-drawer>` (a direct child of `<body>`, so
 * the element's background isolation never covers itself), opened by the
 * Setup BPMs tab's Open BPM control. It renders the settings model of
 * `settings.js` that the Setup BPMs tab renders too, with the tab's own
 * inputs, so a change in either shows in both. It commits the pending
 * settings only, never exclusions, through the setup route; the exclusions
 * belong to the tab.
 *
 * It shows the stored current from `Snapshot.machine` and plots the per-BPM
 * attenuation and pilot-tone readbacks through the render adapter. Opening
 * the drawer re-reads the readbacks and redraws it, so it shows the session
 * state of the moment it opens.
 *
 * Mounted into the `drawer` area by the composition root (`panel.js`); this
 * module's {@link mount} is its only entry point. Without the page's drawer
 * element it fills the area host itself.
 *
 * @module drawer
 */

import '/design-system/js/components/osprey-drawer.js';

import { attenuationFigure, pilotToneFigure } from './figures.js';
import { settingsFor } from './settings.js';
import { NO_SESSION, OPEN_BPM_CONTROL, settingInput } from './setup.js';
import { control, h, noticeFromResult } from './views.js';

/** The page's `<osprey-drawer>` element id. */
export const DRAWER_ID = 'bpm-control-drawer';

/** The per-BPM readback plots: host id -> the setting they plot and their builder. */
export const READBACK_PLOTS = Object.freeze({
  'plot-bpm-atten': Object.freeze({ id: 'attenuation', figure: attenuationFigure }),
  'plot-bpm-pt-state': Object.freeze({ id: 'pilot_tone', figure: pilotToneFigure }),
});

/** Why Set is disabled when nothing is pending. */
export const NOTHING_PENDING = 'No setting changed';

/** Why Revert is disabled when nothing is pending. */
export const NOTHING_TO_REVERT = 'Nothing to revert';

/** @typedef {import('./settings.js').SettingValue} SettingValue */

/**
 * The stored current as the readback shows it.
 * @param {any} snapshot
 * @returns {string}
 */
export function storedCurrentText(snapshot) {
  const current = snapshot?.machine?.stored_current;
  return typeof current === 'number' && Number.isFinite(current) ? `${current.toPrecision(4)} mA` : '—';
}

/**
 * The setup body the drawer commits: the pending settings and nothing else.
 * @param {Record<string, SettingValue>} pending
 * @returns {{settings: Record<string, SettingValue>}}
 */
export function drawerBody(pending) {
  return { settings: { ...pending } };
}

/**
 * The readback of one setting, or null when it was not read.
 * @param {any} readbacks  The `SetupReadbacks`.
 * @param {string} id
 * @returns {any}
 */
export function readbackById(readbacks, id) {
  return (readbacks?.settings ?? []).find((/** @type {any} */ setting) => setting.id === id) ?? null;
}

/**
 * @typedef {object} DrawerModel
 * @property {any} snapshot
 * @property {any[]} settings  The `SettingReadback`s, in declared order.
 * @property {Record<string, SettingValue>} values
 * @property {Record<string, SettingValue>} pending
 * @property {boolean} holding
 */

/**
 * @typedef {object} DrawerHandlers
 * @property {(id: string, value: SettingValue) => void} onSetting
 * @property {() => void} onCommit
 * @property {() => void} onRevert
 */

/**
 * The drawer's controls: the stored current, one input per setting, Set and Revert.
 * @param {DrawerModel} model
 * @param {DrawerHandlers} handlers
 * @returns {HTMLElement}
 */
export function drawerControlsView(model, handlers) {
  const noSession = model.holding ? null : NO_SESSION;
  const hasPending = Object.keys(model.pending).length > 0;
  const rows = model.settings.map((setting) =>
    h('label', { class: 'setup-setting' }, `${setting.id.replace(/_/g, ' ')} `,
      settingInput(setting.id, model.values[setting.id], (value) => handlers.onSetting(setting.id, value), noSession),
      setting.id in model.pending ? h('span', { class: 'setup-pending', text: 'pending' }) : null));
  return h('div', { class: 'bpm-control' },
    h('p', { class: 'fact', 'data-fact': 'stored-current-readback' },
      h('span', { class: 'fact-label', text: 'Stored current ' }),
      h('output', { class: 'fact-value', 'data-control': 'stored-current-readback', text: storedCurrentText(model.snapshot) })),
    ...rows,
    h('div', { class: 'setup-actions' },
      control('bpm-control-set', 'Set', handlers.onCommit, noSession ?? (hasPending ? null : NOTHING_PENDING)),
      control('bpm-control-revert', 'Revert', handlers.onRevert, hasPending ? null : NOTHING_TO_REVERT)));
}

/** @param {string} id */
function plotHost(id) {
  return h('div', { class: 'plot drawer-plot', id, 'data-control': id });
}

/**
 * Fill the `drawer` area.
 * @type {import('./panel.js').Mount}
 */
export const mount = (host, ctx) => {
  const doc = host.ownerDocument ?? document;
  const sync = settingsFor(ctx);
  const drawer = /** @type {any} */ (doc.getElementById(DRAWER_ID));
  const body = drawer?.querySelector('.drawer-body') ?? host;
  const controls = h('div', { class: 'bpm-control-controls' });
  body.replaceChildren(controls,
    h('h3', { text: 'Attenuation per BPM' }), plotHost('plot-bpm-atten'),
    h('h3', { text: 'Pilot tone state per BPM' }), plotHost('plot-bpm-pt-state'));
  let deferred = false;
  /** @type {any} */
  let plottedReadbacks;
  /** @type {any} */
  let plottedCatalog;

  /** Whether the operator is typing in one of the drawer's fields. */
  function typing() {
    const active = doc.activeElement;
    return active != null && controls.contains(active) && active.tagName === 'INPUT' && active.getAttribute('type') === 'text';
  }

  /** Draw the readback plots when their readbacks or the device catalog changed. */
  function plot() {
    const readbacks = sync.model.readbacks();
    const catalog = ctx.snapshot()?.catalog ?? null;
    if (readbacks === plottedReadbacks && catalog === plottedCatalog) return;
    plottedReadbacks = readbacks;
    plottedCatalog = catalog;
    for (const [plotId, { id, figure }] of Object.entries(READBACK_PLOTS)) {
      ctx.adapter?.render(plotId, (/** @type {any} */ theme) => figure(readbackById(readbacks, id), catalog, theme));
    }
  }

  /** @param {boolean} [force]  Redraw even while the operator types. */
  function render(force = false) {
    plot();
    if (!force && typing()) {
      deferred = true;
      return;
    }
    deferred = false;
    controls.replaceChildren(drawerControlsView({
      snapshot: ctx.snapshot(),
      settings: sync.model.readbacks()?.settings ?? [],
      values: sync.model.values(),
      pending: sync.model.pending(),
      holding: ctx.holding(),
    }, handlers));
  }

  async function commit() {
    const result = await sync.commit(drawerBody(sync.model.pending()));
    ctx.notify(noticeFromResult('BPM control', result));
    render(true);
  }

  /** @type {DrawerHandlers} */
  const handlers = {
    onSetting(id, value) {
      sync.model.set(id, value);
    },
    onCommit: () => void commit(),
    onRevert() {
      sync.model.revert();
    },
  };

  doc.addEventListener(OPEN_BPM_CONTROL, () => {
    if (typeof drawer?.open === 'function') drawer.open();
    else drawer?.setAttribute('open', '');
  });
  // Opening re-reads the readbacks and redraws with the session state as it is now.
  drawer?.addEventListener('drawer:open', () => {
    render(true);
    void sync.refresh();
  });
  ctx.onSnapshot(() => render());
  sync.model.on(() => render());
  controls.addEventListener('focusout', () => {
    if (deferred) queueMicrotask(() => render());
  });
  render(true);
};
