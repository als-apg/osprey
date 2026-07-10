/**
 * Shared DOM query helpers for interface tests.
 *
 * Under `checkJs`, a bare `root.querySelector(sel)` / `document.getElementById(id)`
 * is typed `Element | null` / `HTMLElement | null`, so every lookup-and-dereference
 * site trips both a null-safety error (TS18047/TS2531) and — for anything past the
 * base `Element` surface — a property-access error (TS2339). These helpers collapse
 * both into a single call: they assert the element exists (throwing a clear
 * "element not found" on a miss) and return it typed as the caller's chosen element.
 *
 * The return type is chosen by the caller via an optional element *constructor*
 * argument (e.g. `HTMLInputElement`), inferred with no cast; omit it for the
 * common `HTMLElement` case. Pass a constructor only where a subtype surface is
 * actually used (`.value`, `.href`, ...). These throw unconditionally on a miss,
 * so use them only where the element is *required* by the test — a lookup whose
 * absence is a legitimate, asserted-on outcome must stay a plain
 * `querySelector`/`getElementById` guarded by the test itself.
 */

/**
 * `root.querySelector(selector)`, asserting a match exists.
 *
 * @template {typeof Element} [C=typeof HTMLElement]
 * @param {ParentNode} root - element/document/fragment to search within
 * @param {string} selector - CSS selector
 * @param {C} [ctor] - element constructor selecting the return type (default `HTMLElement`)
 * @returns {InstanceType<C>} the matched element, never null
 */
export function qs(root, selector, ctor) {
  const el = root.querySelector(selector);
  if (el === null) throw new Error(`element not found: ${selector}`);
  if (ctor !== undefined && !(el instanceof ctor)) {
    throw new Error(`element ${selector} is not a ${ctor.name}`);
  }
  return /** @type {InstanceType<C>} */ (el);
}

/**
 * `document.getElementById(id)`, asserting the element exists.
 *
 * @template {typeof Element} [C=typeof HTMLElement]
 * @param {string} id - element id (no leading `#`)
 * @param {C} [ctor] - element constructor selecting the return type (default `HTMLElement`)
 * @returns {InstanceType<C>} the matched element, never null
 */
export function byId(id, ctor) {
  const el = document.getElementById(id);
  if (el === null) throw new Error(`element not found: #${id}`);
  if (ctor !== undefined && !(el instanceof ctor)) {
    throw new Error(`element #${id} is not a ${ctor.name}`);
  }
  return /** @type {InstanceType<C>} */ (el);
}
