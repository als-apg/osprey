// @ts-check
/* OSPREY Lattice Dashboard — Model Selection and Figure Availability
 *
 * The standalone top bar's model selector, fed by GET /api/models. The tile
 * bar's rendering of the same choice is a contributed menu (see header.js);
 * both label an entry the same way, through modelLabel().
 *
 * The figure panels the selected model cannot draw: a single-pass model
 * draws optics alone, and every other panel shows why in place of its plot.
 * Whether the build serves a model is a mark on its label only. The banner above the figures carries the state's
 * notice when the build gives the dashboard nothing to load.
 *
 * Uses the createElement()/textContent DOM style — no innerHTML with
 * interpolated data.
 */

/** The mark an unserved model carries wherever it is listed. */
export const UNSERVED_LABEL = 'not served';

/** The solve whose model draws optics alone. */
const SINGLE_PASS = 'single_pass';

/** The figures a single-pass model still draws. */
const OPTICS_ONLY = ['optics'];

/** Each panel's placeholder as the page first drew it, by figure name. */
/** @type {Map<string, Node>} */
const placeholders = new Map();

/**
 * @typedef {object} ModelEntry
 * @property {string} name
 * @property {boolean} served
 * @property {string} solve
 * @property {boolean} selected
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
 * Fill the standalone selector with the models in the order the server lists
 * them (served first). The selector is hidden while there is nothing to pick.
 * @param {ModelEntry[]} models
 */
export function renderModelSelect(models) {
  const select = /** @type {HTMLSelectElement|null} */ (document.getElementById('model-select'));
  if (!select) return;
  select.textContent = '';
  for (const model of models) {
    const option = document.createElement('option');
    option.value = model.name;
    option.textContent = modelLabel(model);
    option.selected = model.selected;
    select.appendChild(option);
  }
  // .action-btn sets display, which outranks the hidden attribute.
  select.style.display = models.length === 0 ? 'none' : '';
}

/**
 * Call onSelect with the model name the operator picks in the standalone
 * selector.
 * @param {(name: string) => void} onSelect
 */
export function bindModelSelect(onSelect) {
  const select = /** @type {HTMLSelectElement|null} */ (document.getElementById('model-select'));
  select?.addEventListener('change', () => onSelect(select.value));
}

/**
 * The figures the selected model of *state* cannot draw.
 * @param {any} state - the /api/state payload
 * @param {string[]} figureNames - the full figure catalog
 * @returns {string[]}
 */
export function unavailableFigures(state, figureNames) {
  if (state.solve !== SINGLE_PASS) return [];
  return figureNames.filter((name) => !OPTICS_ONLY.includes(name));
}

/**
 * Replace a figure panel's plot with the text saying why it is not drawn.
 * @param {string} name
 * @param {string} text
 */
export function showFigureUnavailable(name, text) {
  const cell = document.getElementById(`cell-${name}`);
  const plotEl = /** @type {any} */ (document.getElementById(`plot-${name}`));
  if (!cell || !plotEl) return;
  if (plotEl.data) Plotly.purge(plotEl);
  const message = document.createElement('div');
  message.className = 'figure-placeholder figure-unavailable';
  message.textContent = text;
  plotEl.replaceChildren(message);
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
    if (placeholder && !placeholders.has(name) && !placeholder.classList.contains('figure-unavailable')) {
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
