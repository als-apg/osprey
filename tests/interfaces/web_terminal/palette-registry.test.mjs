/**
 * Unit tests for palette-registry.js — the pure, dependency-injected builder
 * that turns live app data into the flat, grouped command-palette registry.
 * Pins the load-bearing contract:
 *
 *   - group order is Settings → Panels → Layouts → Actions, source order kept
 *   - config loading/error each emit exactly one non-navigable Settings
 *     decoration (status, no run, no searchText)
 *   - config ok flattens a NESTED sections tree into leaf dot-keys, and each
 *     item's run calls the injected revealSetting with its dot-key
 *   - Panels emit Show/Focus items wired to showPanel/focusPanel by id
 *   - Layouts emit items wired to applyPreset with the preset's panels
 *   - Actions are wrapped in order with run passed through
 *   - missing optional deps never throw and contribute nothing
 *
 * Pure module, no DOM:
 *   npx vitest run tests/interfaces/web_terminal/palette-registry.test.mjs
 */

import { describe, it, expect } from 'vitest';

import { buildRegistry } from '../../../src/osprey/interfaces/web_terminal/static/js/palette-registry.js';

/** Items in a given group, in output order. Returns `any[]` so tests can read
 * navigable-only fields (run/searchText/detail) on rows they know are navigable
 * without narrowing the builder's Item union at every call site.
 * @param {ReturnType<typeof buildRegistry>} items
 * @param {string} group
 * @returns {any[]}
 */
function inGroup(items, group) {
  return items.filter((it) => it.group === group);
}

describe('buildRegistry', () => {
  it('GROUP ORDER: emits Settings, Panels, Layouts, Actions in that relative order', () => {
    const items = buildRegistry({
      config: { state: 'ok', sections: { a: { b: 1 } } },
      getHiddenPanels: () => [{ id: 'p1', label: 'Panel One' }],
      getPresets: () => [{ name: 'Wide', panels: ['p1'] }],
      actions: [{ label: 'Restart', run: () => {} }],
    });

    // Relative order of first appearance of each group.
    /** @type {string[]} */
    const order = [];
    for (const it of items) {
      if (!order.includes(it.group)) {
        order.push(it.group);
      }
    }
    expect(order).toEqual(['Settings', 'Panels', 'Layouts', 'Actions']);
  });

  it('LOADING: config loading yields one non-navigable Settings decoration', () => {
    const items = buildRegistry({ config: { state: 'loading' } });
    const settings = inGroup(items, 'Settings');
    expect(settings).toHaveLength(1);
    const [row] = settings;
    expect(row).toEqual({ group: 'Settings', status: 'loading', label: 'Loading settings…' });
    // Non-navigable: no run, no searchText.
    expect('run' in row).toBe(false);
    expect('searchText' in row).toBe(false);
  });

  it('ERROR: config error yields one non-navigable Settings decoration', () => {
    const items = buildRegistry({ config: { state: 'error' } });
    const settings = inGroup(items, 'Settings');
    expect(settings).toHaveLength(1);
    const [row] = settings;
    expect(row).toEqual({ group: 'Settings', status: 'error', label: 'Settings unavailable' });
    expect('run' in row).toBe(false);
    expect('searchText' in row).toBe(false);
  });

  it('OK FLATTEN: nested sections flatten to exact leaf dot-keys and run reveals the dot-key', () => {
    /** @type {string[]} */
    const revealed = [];
    const items = buildRegistry({
      config: {
        state: 'ok',
        sections: {
          control_system: { write_verification: 'readback', writes_enabled: false },
          approval: { enabled: true },
        },
      },
      revealSetting: (dotKey) => revealed.push(dotKey),
    });

    const settings = inGroup(items, 'Settings');
    const keys = settings.map((it) => it.label);
    expect(new Set(keys)).toEqual(
      new Set(['control_system.write_verification', 'control_system.writes_enabled', 'approval.enabled']),
    );

    // Each navigable setting carries searchText === its dot-key, and run reveals it.
    for (const it of settings) {
      expect(it.searchText).toBe(it.label);
      it.run();
    }
    expect(revealed).toEqual([
      'control_system.write_verification',
      'control_system.writes_enabled',
      'approval.enabled',
    ]);
  });

  it('OK FLATTEN: arrays and scalars are leaves, not recursed into', () => {
    const items = buildRegistry({
      config: { state: 'ok', sections: { servers: ['a', 'b'], mode: 'edit' } },
    });
    const keys = inGroup(items, 'Settings').map((it) => it.label);
    expect(new Set(keys)).toEqual(new Set(['servers', 'mode']));
  });

  it('PANELS: hidden -> Show items call showPanel(id); visible -> Focus items call focusPanel(id)', () => {
    /** @type {string[]} */
    const shown = [];
    /** @type {string[]} */
    const focused = [];
    const items = buildRegistry({
      getHiddenPanels: () => [{ id: 'ariel', label: 'ARIEL' }],
      getVisiblePanels: () => [{ id: 'okf', label: 'Facility' }],
      showPanel: (id) => shown.push(id),
      focusPanel: (id) => focused.push(id),
    });

    const panels = inGroup(items, 'Panels');
    expect(panels.map((it) => it.label)).toEqual(['Show ARIEL', 'Focus Facility']);

    panels[0].run();
    panels[1].run();
    expect(shown).toEqual(['ariel']);
    expect(focused).toEqual(['okf']);
  });

  it('LAYOUTS: preset run applies the preset panels array', () => {
    /** @type {string[][]} */
    const applied = [];
    const items = buildRegistry({
      getPresets: () => [{ name: 'Focus Mode', panels: ['chat', 'ariel'] }],
      applyPreset: (panels) => applied.push(panels),
    });

    const layouts = inGroup(items, 'Layouts');
    expect(layouts).toHaveLength(1);
    expect(layouts[0].label).toContain('Focus Mode');
    layouts[0].run();
    expect(applied).toEqual([['chat', 'ariel']]);
  });

  it('ACTIONS: injected actions preserved in order with run wired through', () => {
    /** @type {string[]} */
    const fired = [];
    const items = buildRegistry({
      actions: [
        { label: 'New Session', run: () => fired.push('new') },
        { label: 'Logout', detail: 'end session', run: () => fired.push('logout') },
      ],
    });

    const actions = inGroup(items, 'Actions');
    expect(actions.map((it) => it.label)).toEqual(['New Session', 'Logout']);
    expect(actions[1].detail).toBe('end session');

    actions[0].run();
    actions[1].run();
    expect(fired).toEqual(['new', 'logout']);
  });

  it('MISSING OPTIONAL DEPS: omitting getters does not throw and yields no items', () => {
    // No config, no getters, no actions.
    expect(() => buildRegistry({})).not.toThrow();
    const items = buildRegistry({
      getHiddenPanels: () => [{ id: 'p1', label: 'One' }],
      showPanel: (id) => id,
      // getVisiblePanels intentionally omitted.
    });
    const panels = inGroup(items, 'Panels');
    expect(panels.map((it) => it.label)).toEqual(['Show One']);
    // No Focus items without getVisiblePanels.
    expect(panels.some((it) => it.label.startsWith('Focus'))).toBe(false);
  });
});
