/**
 * Unit tests for the pure `reconcile(layout, registeredPanels)` used on every
 * dockview layout apply (storage load, mode restore, reset).
 *
 *   npx vitest run tests/interfaces/web_terminal/dock-reconcile.test.mjs
 *
 * reconcile repairs a persisted SerializedDockview against the panels the dock
 * can currently build: unknown ids are dropped, newly-registered panels are
 * appended, and only genuinely corrupt/unusable input falls back (returns null).
 * The live persistence plumbing (fromJSON apply, localStorage, project key) is
 * exercised end-to-end by the browser suite; here we pin the pure contract.
 *
 * Imported by RELATIVE path — this module lives under web_terminal, so the
 * /design-system/js/* alias does not apply.
 */

import { test, expect, describe } from 'vitest';

import { reconcile } from '../../../src/osprey/interfaces/web_terminal/static/js/dock-reconcile.js';

/**
 * A single-group SerializedDockview holding the given panel ids in one leaf.
 * @param {string[]} ids
 * @returns {any}
 */
function singleGroupLayout(ids) {
  return {
    grid: {
      root: { type: 'leaf', data: { views: [...ids], activeView: ids[0], id: 'group-1' }, size: 1000 },
      width: 1000,
      height: 800,
      orientation: 'HORIZONTAL',
    },
    panels: Object.fromEntries(
      ids.map((id) => [id, { id, contentComponent: id, title: id }]),
    ),
    activeGroup: 'group-1',
  };
}

/**
 * A two-group (branch) SerializedDockview: `left` ids in one leaf, `right` in
 * another, split horizontally — the shape the default workspace|terminal
 * arrangement serializes to.
 * @param {string[]} left
 * @param {string[]} right
 * @returns {any}
 */
function twoGroupLayout(left, right) {
  return {
    grid: {
      root: {
        type: 'branch',
        data: [
          { type: 'leaf', data: { views: [...left], activeView: left[0], id: 'group-left' }, size: 600 },
          { type: 'leaf', data: { views: [...right], activeView: right[0], id: 'group-right' }, size: 400 },
        ],
        size: 800,
      },
      width: 1000,
      height: 800,
      orientation: 'HORIZONTAL',
    },
    panels: Object.fromEntries(
      [...left, ...right].map((id) => [id, { id, contentComponent: id, title: id }]),
    ),
    activeGroup: 'group-left',
  };
}

/**
 * All panel ids referenced by any group in a layout's grid tree.
 * @param {any} layout
 * @returns {string[]}
 */
function referencedIds(layout) {
  /** @type {string[]} */
  const ids = [];
  /** @param {any} node */
  const walk = (node) => {
    if (!node) return;
    if (node.type === 'leaf') ids.push(...node.data.views);
    else if (node.type === 'branch') node.data.forEach(walk);
  };
  walk(layout.grid.root);
  return ids;
}

describe('reconcile — clean layout pass-through', () => {
  test('a layout whose panels all remain registered is returned intact', () => {
    const layout = twoGroupLayout(['workspace'], ['terminal']);
    const out = reconcile(layout, ['workspace', 'terminal']);
    expect(out).not.toBeNull();
    expect(Object.keys(out.panels).sort()).toEqual(['terminal', 'workspace']);
    expect(referencedIds(out).sort()).toEqual(['terminal', 'workspace']);
  });

  test('accepts a raw JSON string as well as a parsed object', () => {
    const layout = singleGroupLayout(['workspace', 'terminal']);
    const out = reconcile(JSON.stringify(layout), ['workspace', 'terminal']);
    expect(out).not.toBeNull();
    expect(referencedIds(out).sort()).toEqual(['terminal', 'workspace']);
  });

  test('does not mutate the input layout', () => {
    const layout = twoGroupLayout(['workspace'], ['terminal']);
    const snapshot = JSON.stringify(layout);
    reconcile(layout, ['workspace']); // drops terminal in the copy, not the input
    expect(JSON.stringify(layout)).toBe(snapshot);
  });
});

describe('reconcile — unknown ids dropped', () => {
  test('an id no longer registered is removed from panels and its group', () => {
    const layout = twoGroupLayout(['workspace', 'stale-panel'], ['terminal']);
    const out = reconcile(layout, ['workspace', 'terminal']);
    expect(Object.keys(out.panels).sort()).toEqual(['terminal', 'workspace']);
    expect(referencedIds(out)).not.toContain('stale-panel');
  });

  test('a group emptied of all its panels is pruned; the lone survivor re-wraps as a branch root', () => {
    // The right group holds only a now-unknown panel; dropping it collapses the
    // inner branch to one group. dockview's fromJSON requires a BRANCH root, so
    // reconcile re-wraps that lone survivor rather than leaving a bare leaf root.
    const layout = twoGroupLayout(['workspace'], ['stale-panel']);
    const out = reconcile(layout, ['workspace']);
    expect(out.grid.root.type).toBe('branch');
    expect(out.grid.root.data).toHaveLength(1);
    expect(out.grid.root.data[0].type).toBe('leaf');
    expect(referencedIds(out)).toEqual(['workspace']);
  });

  test('a dropped activeView is replaced by a surviving one', () => {
    const layout = singleGroupLayout(['stale-panel', 'workspace']);
    const out = reconcile(layout, ['workspace']);
    const leaf = out.grid.root.data[0];
    expect(leaf.data.views).toEqual(['workspace']);
    expect(leaf.data.activeView).toBe('workspace');
  });
});

describe('reconcile — iframe placeholders always survive', () => {
  // Service placeholders are created lazily (panel registration is async), so
  // at apply time they are often not yet in the live dock's registered set.
  // Pruning them would silently discard the stored arrangement of every
  // service panel on each reload — the regression this pin guards against.
  test('an unregistered iframe: placeholder keeps its stored position', () => {
    const layout = twoGroupLayout(['terminal'], ['iframe:artifacts']);
    const out = reconcile(layout, ['terminal']);
    expect(out).not.toBeNull();
    expect(Object.keys(out.panels).sort()).toEqual(['iframe:artifacts', 'terminal']);
    expect(referencedIds(out).sort()).toEqual(['iframe:artifacts', 'terminal']);
  });

  test('non-placeholder unknown ids are still dropped alongside kept placeholders', () => {
    const layout = twoGroupLayout(['terminal', 'stale-panel'], ['iframe:artifacts']);
    const out = reconcile(layout, ['terminal']);
    expect(referencedIds(out).sort()).toEqual(['iframe:artifacts', 'terminal']);
    expect(out.panels['stale-panel']).toBeUndefined();
  });
});

describe('reconcile — new panels appended', () => {
  test('a registered panel absent from the layout is appended with a panels entry', () => {
    const layout = twoGroupLayout(['workspace'], ['terminal']);
    const out = reconcile(layout, ['workspace', 'terminal', 'newcomer']);
    expect(referencedIds(out)).toContain('newcomer');
    expect(out.panels.newcomer).toMatchObject({ id: 'newcomer', contentComponent: 'newcomer' });
  });

  test('descriptor objects supply the appended panel component and title', () => {
    const layout = singleGroupLayout(['workspace']);
    const out = reconcile(layout, [
      'workspace',
      { id: 'ariel', contentComponent: 'panel-iframe', title: 'ARIEL' },
    ]);
    expect(out.panels.ariel).toMatchObject({
      id: 'ariel',
      contentComponent: 'panel-iframe',
      title: 'ARIEL',
    });
    expect(referencedIds(out)).toContain('ariel');
  });

  test('new panels land in the largest surviving group', () => {
    // left group has two panels, right has one — the newcomer joins the left.
    const layout = twoGroupLayout(['workspace', 'lattice'], ['terminal']);
    const out = reconcile(layout, ['workspace', 'lattice', 'terminal', 'newcomer']);
    const leftViews = out.grid.root.data[0].data.views;
    expect(leftViews).toContain('newcomer');
  });
});

describe('reconcile — grid root invariant (dockview fromJSON requires a branch root)', () => {
  test('a single surviving group is wrapped as a branch root', () => {
    const layout = singleGroupLayout(['workspace', 'terminal']);
    const out = reconcile(layout, ['workspace', 'terminal']);
    expect(out.grid.root.type).toBe('branch');
    expect(out.grid.root.data[0].type).toBe('leaf');
  });

  test('a multi-group layout keeps its branch root', () => {
    const layout = twoGroupLayout(['workspace'], ['terminal']);
    const out = reconcile(layout, ['workspace', 'terminal']);
    expect(out.grid.root.type).toBe('branch');
  });

  test('collapsing every group but one still yields a branch root', () => {
    const layout = twoGroupLayout(['gone'], ['workspace']);
    const out = reconcile(layout, ['workspace']);
    expect(out.grid.root.type).toBe('branch');
    expect(referencedIds(out)).toEqual(['workspace']);
  });

  test('appending into a re-wrapped single-group root stays a branch root', () => {
    const layout = singleGroupLayout(['workspace']);
    const out = reconcile(layout, ['workspace', 'newcomer']);
    expect(out.grid.root.type).toBe('branch');
    expect(referencedIds(out)).toContain('newcomer');
  });
});

describe('reconcile — activeGroup', () => {
  test('repoints activeGroup when the referenced group did not survive the prune', () => {
    // activeGroup is 'group-left'; dropping its only panel prunes it away, so
    // activeGroup must repoint to a surviving group rather than dangle.
    const layout = twoGroupLayout(['stale-panel'], ['terminal']);
    const out = reconcile(layout, ['terminal']);
    expect(out.activeGroup).toBe('group-right');
  });

  test('leaves a still-present activeGroup untouched', () => {
    const layout = twoGroupLayout(['workspace'], ['terminal']);
    const out = reconcile(layout, ['workspace', 'terminal']);
    expect(out.activeGroup).toBe('group-left');
  });
});

describe('reconcile — corrupt / unusable input falls back (null)', () => {
  test('unparseable JSON string returns null', () => {
    expect(reconcile('{ not valid json', ['workspace'])).toBeNull();
  });

  test('non-layout objects return null', () => {
    expect(reconcile(null, ['workspace'])).toBeNull();
    expect(reconcile(42, ['workspace'])).toBeNull();
    expect(reconcile({ grid: { root: null }, panels: {} }, ['workspace'])).toBeNull();
    expect(reconcile({ panels: {} }, ['workspace'])).toBeNull();
  });

  test('a layout with no surviving panels returns null (full fallback)', () => {
    const layout = singleGroupLayout(['gone-a', 'gone-b']);
    expect(reconcile(layout, ['workspace', 'terminal'])).toBeNull();
  });
});
