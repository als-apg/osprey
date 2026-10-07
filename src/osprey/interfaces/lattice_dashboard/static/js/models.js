// @ts-check
/* OSPREY Lattice Dashboard — Model Selection
 *
 * The standalone top bar's model selector, fed by GET /api/models. The tile
 * bar's rendering of the same choice is a contributed menu (see header.js);
 * both label an entry the same way, through modelLabel().
 *
 * Uses the createElement()/textContent DOM style — no innerHTML with
 * interpolated data.
 */

/** The mark an unserved model carries wherever it is listed. */
export const UNSERVED_LABEL = 'not served: optics only';

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
