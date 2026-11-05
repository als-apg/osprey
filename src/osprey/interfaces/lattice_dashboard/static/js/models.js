// @ts-check
/* OSPREY Lattice Dashboard — Model Selection and Figure Availability
 *
 * The panel toolbar's model switcher, fed by GET /api/models: a segmented
 * control with one button per model, or a select when the build lists more
 * models than a toolbar row holds.
 *
 * The figure panels the selected model cannot draw, as the server's
 * selection.capabilities lists them: every other panel shows why in place of
 * its plot, in the shared centred state. The Resonance panel offers the one
 * obvious next step when the build has a periodic model to switch to.
 * Whether the build serves a model is a mark on its label only.
 * The banner above the figures carries the state's notice when the build
 * gives the dashboard nothing to load, and says so when the selected model
 * did not load.
 *
 * Uses the createElement()/textContent DOM style — no innerHTML with
 * interpolated data.
 */

import { renderState } from '/design-system/js/panel-state.js';

/** The mark an unserved model carries wherever it is listed. */
export const UNSERVED_LABEL = 'not served';

/** The most models the switcher shows as a segmented control. */
export const MAX_SEGMENT_MODELS = 5;

/** The figure panel that offers a switch to a periodic model. */
const SWITCH_FIGURE = 'resonance';

/** Each panel's placeholder as the page first drew it, by figure name. */
/** @type {Map<string, Node>} */
const placeholders = new Map();

/** The refusal each hidden panel shows, by figure name. */
/** @type {Map<string, string>} */
const refusals = new Map();

/** The last GET /api/models list. @type {ModelEntry[]} */
let modelList = [];

/** The handler bindModelSwitch installed. @type {((name: string) => void) | null} */
let selectModel = null;

/**
 * @typedef {object} ModelEntry
 * @property {string} name
 * @property {boolean} served
 * @property {string|null} solve
 * @property {boolean} selected
 * @property {string|null} error
 */

/**
 * The label a model entry is listed under.
 * @param {ModelEntry} model
 * @returns {string}
 */
export function modelLabel(model) {
  return model.served ? model.name : `${model.name} (${UNSERVED_LABEL})`;
}

/**
 * The periodic model a panel the selected model cannot draw offers to switch
 * to: one named SR when the build lists it, else the first listed (served
 * first). Null when there is none.
 * @returns {ModelEntry | null}
 */
function switchTarget() {
  const periodic = modelList.filter((model) => model.solve === 'periodic' && !model.selected);
  return periodic.find((model) => model.name.toUpperCase() === 'SR') ?? periodic[0] ?? null;
}

/**
 * Fill the toolbar's switcher with the models in the order the server lists
 * them (served first): a segmented control, or a select beyond
 * MAX_SEGMENT_MODELS. The switcher is hidden while there is nothing to pick.
 * @param {ModelEntry[]} models
 */
export function renderModelSwitch(models) {
  modelList = models;
  // A panel shown before the list arrived gets its switch action now.
  const refusal = refusals.get(SWITCH_FIGURE);
  const hidden = document.getElementById(`cell-${SWITCH_FIGURE}`)?.dataset.available === 'false';
  if (refusal !== undefined && hidden) showFigureUnavailable(SWITCH_FIGURE, refusal);

  const host = document.getElementById('model-switch');
  if (!host) return;
  host.hidden = models.length === 0;
  if (models.length > MAX_SEGMENT_MODELS) {
    const select = document.createElement('select');
    select.className = 'model-select';
    select.setAttribute('aria-label', 'Lattice model');
    for (const model of models) {
      const option = document.createElement('option');
      option.value = model.name;
      option.textContent = modelLabel(model);
      select.append(option);
    }
    host.replaceChildren(select);
    select.value = models.find((model) => model.selected)?.name ?? '';
    return;
  }
  const seg = document.createElement('div');
  seg.className = 'osprey-seg';
  seg.setAttribute('role', 'group');
  seg.setAttribute('aria-label', 'Lattice model');
  for (const model of models) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.dataset.model = model.name;
    btn.textContent = modelLabel(model);
    btn.setAttribute('aria-pressed', String(model.selected));
    seg.append(btn);
  }
  host.replaceChildren(seg);
}

/**
 * Call onSelect with the model name the operator picks in the toolbar's
 * switcher, or in a panel's switch action.
 * @param {(name: string) => void} onSelect
 */
export function bindModelSwitch(onSelect) {
  selectModel = onSelect;
  const host = document.getElementById('model-switch');
  host?.addEventListener('click', (event) => {
    const target = /** @type {Element} */ (event.target);
    const btn = /** @type {HTMLElement | null} */ (target.closest('button[data-model]'));
    if (!btn || btn.getAttribute('aria-pressed') === 'true') return;
    onSelect(/** @type {string} */ (btn.dataset.model));
  });
  host?.addEventListener('change', (event) => {
    const select = /** @type {HTMLSelectElement} */ (event.target);
    onSelect(select.value);
  });
}

/**
 * The figures the selected model of *state* cannot draw: those its ready
 * selection's capabilities leave out. A selection that is not ready hides
 * nothing.
 * @param {any} state - the /api/state payload
 * @param {string[]} figureNames - the full figure catalog
 * @returns {string[]}
 */
export function unavailableFigures(state, figureNames) {
  const selection = state?.selection;
  if (selection?.status !== 'ready') return [];
  const drawn = selection.capabilities.figures;
  return figureNames.filter((name) => !drawn.includes(name));
}

/**
 * Replace a figure panel's plot with the centred state saying why it is not
 * drawn. The Resonance panel adds a switch to a periodic model when the build
 * lists one; no other panel has an action.
 * @param {string} name
 * @param {string} text
 */
export function showFigureUnavailable(name, text) {
  const cell = document.getElementById(`cell-${name}`);
  const plotEl = /** @type {any} */ (document.getElementById(`plot-${name}`));
  if (!cell || !plotEl) return;
  if (plotEl.data) Plotly.purge(plotEl);
  refusals.set(name, text);
  const host = document.createElement('div');
  host.className = 'figure-placeholder figure-unavailable';
  const target = name === SWITCH_FIGURE ? switchTarget() : null;
  renderState(host, {
    icon: 'activity',
    message: text,
    ...(target && selectModel
      ? { action: { label: `Switch to ${target.name}`, onClick: () => selectModel?.(target.name) } }
      : {}),
  });
  plotEl.replaceChildren(host);
  cell.dataset.available = 'false';
}

/**
 * Give a hidden figure panel its placeholder back; a no-op on a shown one.
 * @param {string} name
 */
function clearFigureUnavailable(name) {
  const cell = document.getElementById(`cell-${name}`);
  const plotEl = document.getElementById(`plot-${name}`);
  if (!cell || !plotEl || cell.dataset.available !== 'false') return;
  refusals.delete(name);
  const placeholder = placeholders.get(name);
  plotEl.replaceChildren(...(placeholder ? [placeholder.cloneNode(true)] : []));
  delete cell.dataset.available;
}

/**
 * Hide the panels the selected model cannot draw and restore the others. A
 * hidden panel shows the figure route's own refusal, which fetchFigure hands
 * to showFigureUnavailable.
 * @param {any} state - the /api/state payload
 * @param {string[]} figureNames - the full figure catalog
 * @param {(name: string) => void} fetchFigure - fetches a figure and renders it or its refusal
 */
export function syncAvailability(state, figureNames, fetchFigure) {
  for (const name of figureNames) {
    const placeholder = document.querySelector(`#plot-${name} > .figure-placeholder`);
    if (
      placeholder &&
      !placeholders.has(name) &&
      !placeholder.classList.contains('figure-unavailable')
    ) {
      placeholders.set(name, placeholder.cloneNode(true));
    }
  }
  const hidden = unavailableFigures(state, figureNames);
  for (const name of figureNames) {
    if (!hidden.includes(name)) clearFigureUnavailable(name);
    else fetchFigure(name);
  }
}

/**
 * The banner text for a state: the build's notice, else that the selection did
 * not load, naming its error only when one is known, else null.
 * @param {any} state - the /api/state body
 * @returns {string | null}
 */
export function selectionNotice(state) {
  if (state?.notice) return state.notice;
  const selection = state?.selection;
  const shown = selection?.status === 'failed' || (selection?.status === 'none' && selection.error);
  if (!shown) return null;
  const headline = selection.model ? `${selection.model} did not load` : 'No model loaded';
  return selection.error ? `${headline}: ${selection.error}` : headline;
}

/**
 * Show the state's notice in the banner above the figures, verbatim; a null
 * notice empties and hides it.
 * @param {string | null} notice - the /api/state `notice`
 */
export function renderNotice(notice) {
  const banner = document.getElementById('model-notice');
  if (!banner) return;
  banner.textContent = notice ?? '';
  // .summary-strip sets display, which outranks the hidden attribute.
  banner.style.display = notice === null ? 'none' : '';
}
