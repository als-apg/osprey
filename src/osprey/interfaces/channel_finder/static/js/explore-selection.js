// @ts-check
/**
 * OSPREY Channel Finder — Hierarchical selection logic (pure).
 *
 * Extracted from explore-hierarchical.js so the Miller-column selection state
 * machine has no DOM or module-state dependencies and can be unit-tested under
 * Vitest (see tests/interfaces/channel_finder/explore-selection.test.mjs). The
 * handler applies the returned decision to the DOM / module state.
 */

/**
 * Whether a hierarchy level is a tree-type (editable) level.
 * Unknown/unconfigured levels default to tree.
 * @param {any} levelsConfig - `hierInfo.hierarchy_config.levels` (map keyed by level name).
 * @param {string} levelName
 * @returns {boolean}
 */
export function isTreeLevel(levelsConfig, levelName) {
  if (!levelsConfig || !levelsConfig[levelName]) return true;  // default to tree
  return levelsConfig[levelName].type === 'tree';
}

/**
 * @typedef {object} SelectionResult
 * @property {string[]} selectedValues - New selected values for the target column.
 * @property {string|string[]|null} selectionValue - What to store in selections[level]
 *   (a scalar for one value, an array for many, or null to delete the entry).
 * @property {boolean} loadNext - Whether to drill into the next level.
 */

/**
 * Compute the next selection state for a Miller column when `value` is clicked.
 *
 * Terminal (last) levels toggle multi-select; non-terminal levels are single
 * select (clicking replaces the prior value). Pure: takes a snapshot of the
 * column's current selection and returns the decision to apply.
 *
 * @param {string[]} currentSelectedValues - The column's currently selected values.
 * @param {string} value - The clicked value.
 * @param {boolean} isLastLevel - Whether this is the terminal hierarchy level.
 * @returns {SelectionResult}
 */
export function computeSelection(currentSelectedValues, value, isLastLevel) {
  const set = new Set(currentSelectedValues);
  if (set.has(value)) {
    set.delete(value);
  } else {
    if (!isLastLevel) {
      // Single select for non-terminal levels
      set.clear();
    }
    set.add(value);
  }

  const selectedValues = [...set];
  /** @type {string|string[]|null} */
  let selectionValue;
  if (selectedValues.length === 0) {
    selectionValue = null;
  } else if (selectedValues.length === 1) {
    selectionValue = selectedValues[0];
  } else {
    selectionValue = selectedValues;
  }

  const loadNext = !isLastLevel && selectedValues.length === 1;
  return { selectedValues, selectionValue, loadNext };
}
