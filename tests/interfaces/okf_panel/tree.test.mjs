// @ts-check
/**
 * Unit tests for the OKF Knowledge Panel's sidebar tree module (tree.js) --
 * in particular the deterministic, browser-free regression gate for the
 * "boot race" between `highlightActive`/`highlightStructure` and
 * `renderTree` (Phase 4, Task 4.2):
 *   npx vitest run tests/interfaces/okf_panel/tree.test.mjs
 *
 * `renderTree` rebuilds the DOM from scratch and then re-applies whatever
 * concept highlight was set *before* the render (its tail calls
 * `applyConceptHighlight(activeConceptId)` directly, guarded by
 * `if (activeConceptId != null)`). Two sequences pin that guard:
 *
 *   7a: highlightActive(id) -> renderTree(...)      the re-apply must fire
 *   7b: highlightActive(id) -> highlightStructure() -> renderTree(...)
 *       the reset inside highlightStructure() must stick, so the re-apply
 *       must NOT fire
 *
 * A third sequence (7b-plain) matches an earlier draft of this test's
 * ordering but is deliberately NOT a regression gate for the guard --
 * see the comment on that test below for why.
 */

import { test, expect, describe, beforeEach, vi } from 'vitest';

/** @typedef {{id: string, title?: string, description?: string}} Concept */
/** @typedef {{id?: string, label?: string, concepts?: Concept[]}} Group */

/** @type {typeof import('../../../src/osprey/interfaces/okf_panel/static/js/tree.js')} */
let tree;
/** @type {HTMLElement} */
let treeEl;
/** @type {HTMLElement} */
let structureLink;
/** @type {import('vitest').Mock<(id: string) => void>} */
let onSelect;

/** @type {Group[]} */
const GROUPS = [
  {
    id: 'control-system',
    label: 'Control System',
    concepts: [
      {
        id: 'control-system/channel-finding',
        title: 'Channel Finding',
        description: 'Find channels by name or pattern.',
      },
      { id: 'control-system/archiver', title: 'Archiver' },
    ],
  },
  {
    id: 'empty-group',
    label: 'Empty Group',
    concepts: [],
  },
];

function mountFixture() {
  document.body.innerHTML = `
    <div id="tree"></div>
    <a id="structure-link" href="#"></a>
  `;
  treeEl = /** @type {HTMLElement} */ (document.getElementById('tree'));
  structureLink = /** @type {HTMLElement} */ (document.getElementById('structure-link'));
}

// Module isolation: `tree.js` keeps `activeConceptId`/`treeEl`/`structureLink`/
// `onSelect` as module-private `let`s that persist across calls within the
// same module instance. `initTree` only overwrites the DOM-handle/callback
// trio -- it does NOT reset `activeConceptId` -- so re-calling `initTree` in
// a plain `beforeEach` would leak an `activeConceptId` set by one test (e.g.
// 7a) into the next (e.g. 7b), potentially masking a real regression.
// `vi.resetModules()` plus a fresh dynamic `import()` per test sidesteps this
// entirely: each test gets its own, never-before-touched module instance, so
// there is no shared state to leak by construction.
beforeEach(async () => {
  vi.resetModules();
  mountFixture();
  onSelect = vi.fn();
  tree = await import('../../../src/osprey/interfaces/okf_panel/static/js/tree.js');
  tree.initTree({ treeEl, structureLink, onSelect });
});

describe('boot race: highlightActive before renderTree (7a)', () => {
  test('renderTree re-applies a previously set activeConceptId', () => {
    tree.highlightActive('control-system/channel-finding');
    tree.renderTree(GROUPS);

    const link = treeEl.querySelector(
      '.concept-link.active[data-concept-id="control-system/channel-finding"]'
    );
    expect(link).not.toBeNull();
  });
});

describe('boot race: highlightActive, then highlightStructure, before renderTree (7b)', () => {
  test('highlightStructure resets activeConceptId so the later renderTree does not re-light a concept', () => {
    tree.highlightActive('control-system/channel-finding');
    tree.highlightStructure();
    tree.renderTree(GROUPS);

    expect(structureLink.classList.contains('active')).toBe(true);
    expect(treeEl.querySelector('.concept-link.active')).toBeNull();
  });
});

describe('7b-plain: highlightStructure before renderTree, with no prior highlightActive', () => {
  // NOT mutation-sensitive for the `activeConceptId != null` guard in
  // renderTree, and it is intentionally kept that way -- do not mistake this
  // for the guard's regression gate (that is test 7b, above).
  //
  // Here `activeConceptId` is already `null` before `highlightStructure()`
  // runs (fresh module instance), so `highlightStructure` sets it to `null`
  // again -- a no-op -- and `renderTree`'s tail calls
  // `applyConceptHighlight(null)` whether or not the `!= null` guard exists.
  // `applyConceptHighlight` compares `link.dataset.conceptId === id`; since
  // `dataset` values are always strings, `=== null` is always false, so the
  // call just strips `.active` from every concept link (a no-op right after
  // a fresh render) and never touches `structureLink`. Dropping the guard
  // therefore leaves this assertion passing -- it is a purely type-level
  // regression (`string | null` flowing into `applyConceptHighlight(id:
  // string)`), caught by `npm run typecheck` (TS2345), not by this test.
  test('structureLink is active and no concept link is active (assertions pass with or without the guard)', () => {
    tree.highlightStructure();
    tree.renderTree(GROUPS);

    expect(structureLink.classList.contains('active')).toBe(true);
    expect(treeEl.querySelector('.concept-link.active')).toBeNull();
  });
});

describe('ordinary order: renderTree before highlightActive', () => {
  test('highlightActive activates the concept link on an already-rendered tree', () => {
    tree.renderTree(GROUPS);
    tree.highlightActive('control-system/channel-finding');

    const link = treeEl.querySelector(
      '.concept-link.active[data-concept-id="control-system/channel-finding"]'
    );
    expect(link).not.toBeNull();
  });
});

describe('click / dataset round-trip', () => {
  test('clicking a concept link calls onSelect with the exact id carried in data-concept-id and prevents default', () => {
    tree.renderTree(GROUPS);

    const link = /** @type {HTMLElement | null} */ (
      treeEl.querySelector('.concept-link[data-concept-id="control-system/channel-finding"]')
    );
    expect(link).not.toBeNull();
    // The value the click handler will dispatch is carried on the element as
    // data-concept-id; confirm it round-trips exactly before driving the
    // click, so the assertion below on onSelect's argument is meaningfully
    // tied to that attribute rather than an assumption about it.
    expect(/** @type {HTMLElement} */ (link).dataset.conceptId).toBe(
      'control-system/channel-finding'
    );

    const clickEvent = new MouseEvent('click', { bubbles: true, cancelable: true });
    /** @type {HTMLElement} */ (link).dispatchEvent(clickEvent);

    expect(onSelect).toHaveBeenCalledTimes(1);
    expect(onSelect).toHaveBeenCalledWith('control-system/channel-finding');
    expect(clickEvent.defaultPrevented).toBe(true);
  });
});

describe('renderTree: empty-group placeholder', () => {
  test('a group with no concepts renders the "(no concepts)" placeholder', () => {
    tree.renderTree(GROUPS);

    const groups = Array.from(treeEl.querySelectorAll('section.group'));
    const emptyGroup = groups.find(
      (section) => section.querySelector('.group-label')?.textContent === 'Empty Group'
    );
    expect(emptyGroup).toBeDefined();
    const placeholder = /** @type {Element} */ (emptyGroup).querySelector('.muted.empty-group');
    expect(placeholder).not.toBeNull();
    expect(/** @type {Element} */ (placeholder).textContent).toBe('(no concepts)');
  });
});

describe('highlightStructure basics', () => {
  test('marks structureLink active and clears any concept highlight', () => {
    tree.renderTree(GROUPS);
    tree.highlightActive('control-system/channel-finding');

    tree.highlightStructure();

    expect(structureLink.classList.contains('active')).toBe(true);
    expect(treeEl.querySelectorAll('.concept-link.active').length).toBe(0);
  });
});
