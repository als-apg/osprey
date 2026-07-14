// @ts-check
/**
 * OSPREY Channel Finder — Feedback shared view-state singleton.
 *
 * Holds the mutable view state (`_container`, `_currentKey`) behind accessors,
 * plus a registered rerender hook. `feedback-detail.js` triggers a re-render
 * through `getRerender()` instead of importing `feedback.js` at module-eval
 * time, which would form an ESM circular import (feedback.js already imports
 * the detail renderer). The list view registers its dispatcher via
 * `setRerender()` when it mounts.
 */

/** @type {HTMLElement|null} */
let _container = null;
/** When set, the view is showing an entry's detail; otherwise the list. */
/** @type {string|null} */
let _currentKey = null;
/** @type {(() => void)|null} */
let _rerender = null;
/** @type {(() => void)|null} */
let _renderList = null;

/** @returns {HTMLElement|null} */
export function getContainer() {
  return _container;
}

/** @param {HTMLElement|null} container */
export function setContainer(container) {
  _container = container;
}

/** @returns {string|null} */
export function getCurrentKey() {
  return _currentKey;
}

/** @param {string|null} key */
export function setCurrentKey(key) {
  _currentKey = key;
}

/**
 * Register the view dispatcher used to re-render after a state change.
 * @param {() => void} fn
 */
export function setRerender(fn) {
  _rerender = fn;
}

/** @returns {(() => void)|null} */
export function getRerender() {
  return _rerender;
}

/**
 * Register the list-view renderer. Used by the detail view's "entry not found"
 * path to drop straight back to the list (as the original did) without routing
 * through the full dispatcher, and without importing feedback.js at eval time.
 * @param {() => void} fn
 */
export function setRenderList(fn) {
  _renderList = fn;
}

/** @returns {(() => void)|null} */
export function getRenderList() {
  return _renderList;
}
