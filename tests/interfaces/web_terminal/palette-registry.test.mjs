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
 *   - Panels emit "Open … in a new window" rows from getPopoutPanels (active
 *     panel INCLUDED) and "Open … in a new tile" rows from getVisiblePanels
 *     (active panel EXCLUDED, dropped entirely without openPanelBeside)
 *   - panel rows carry domain + verb synonyms in searchText
 *   - Layouts emit items wired to applyPreset with the preset's NAME
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
          control_system: { type: 'epics', writes_enabled: false },
          approval: { enabled: true },
        },
      },
      revealSetting: (dotKey) => revealed.push(dotKey),
    });

    const settings = inGroup(items, 'Settings');
    const keys = settings.map((it) => it.label);
    expect(new Set(keys)).toEqual(
      new Set(['control_system.type', 'control_system.writes_enabled', 'approval.enabled']),
    );

    // Each navigable setting carries searchText === its dot-key, and run reveals it.
    for (const it of settings) {
      expect(it.searchText).toBe(it.label);
      it.run();
    }
    expect(revealed).toEqual([
      'control_system.type',
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

  it('FR8 ACTIVE PANEL: the active panel gets a new-window row but NO new-tile row', () => {
    // ARIEL is the active panel, so panel-manager's getVisiblePanels (which
    // filters the active id out) omits it while getPopoutPanels keeps it.
    const items = buildRegistry({
      getVisiblePanels: () => [{ id: 'okf', label: 'Facility' }],
      getPopoutPanels: () => [
        { id: 'ariel', label: 'ARIEL' },
        { id: 'okf', label: 'Facility' },
      ],
      focusPanel: () => {},
      popoutPanel: () => {},
      openPanelBeside: () => {},
    });

    const labels = inGroup(items, 'Panels').map((it) => it.label);
    expect(labels).toContain('Open ARIEL in a new window');
    // Opening the active panel beside itself is a no-op, so the row is absent.
    expect(labels).not.toContain('Open ARIEL in a new tile');
    // A non-active member gets both verbs.
    expect(labels).toContain('Open Facility in a new window');
    expect(labels).toContain('Open Facility in a new tile');
  });

  it('FR8 NON-ACTIVE PANEL: with ARIEL not active, ARIEL gets both Open rows', () => {
    const items = buildRegistry({
      getVisiblePanels: () => [{ id: 'ariel', label: 'ARIEL' }],
      getPopoutPanels: () => [{ id: 'ariel', label: 'ARIEL' }],
      focusPanel: () => {},
      popoutPanel: () => {},
      openPanelBeside: () => {},
    });

    const labels = inGroup(items, 'Panels').map((it) => it.label);
    expect(labels).toEqual([
      'Focus ARIEL',
      'Open ARIEL in a new window',
      'Open ARIEL in a new tile',
    ]);
  });

  it('OPEN ROWS: run calls popoutPanel / openPanelBeside with the panel id', () => {
    /** @type {string[]} */
    const poppedOut = [];
    /** @type {string[]} */
    const besided = [];
    const items = buildRegistry({
      getVisiblePanels: () => [{ id: 'okf', label: 'Facility' }],
      getPopoutPanels: () => [{ id: 'okf', label: 'Facility' }],
      popoutPanel: (id) => poppedOut.push(id),
      openPanelBeside: (id) => besided.push(id),
    });

    const byLabel = (/** @type {string} */ label) => inGroup(items, 'Panels').find((it) => it.label === label);
    byLabel('Open Facility in a new window').run();
    byLabel('Open Facility in a new tile').run();
    expect(poppedOut).toEqual(['okf']);
    expect(besided).toEqual(['okf']);
  });

  it('SIMPLE MODE: omitting openPanelBeside drops every new-tile row, Focus rows stay', () => {
    // Simple mode's layout is locked to one service tile, so palette-boot
    // withholds the closure — the rows come off the SAME getter as Focus, so
    // this is the only thing that can distinguish them.
    const items = buildRegistry({
      getVisiblePanels: () => [{ id: 'okf', label: 'Facility' }],
      getPopoutPanels: () => [{ id: 'okf', label: 'Facility' }],
      focusPanel: () => {},
      popoutPanel: () => {},
    });

    const labels = inGroup(items, 'Panels').map((it) => it.label);
    expect(labels).toEqual(['Focus Facility', 'Open Facility in a new window']);
  });

  it('NO POPOUT GETTER: omitting getPopoutPanels drops every new-window row', () => {
    const items = buildRegistry({
      getVisiblePanels: () => [{ id: 'okf', label: 'Facility' }],
      openPanelBeside: () => {},
    });

    const labels = inGroup(items, 'Panels').map((it) => it.label);
    expect(labels).toEqual(['Focus Facility', 'Open Facility in a new tile']);
  });

  it('SYNONYMS: a panel domain alias matches EVERY row for that panel', () => {
    const items = buildRegistry({
      getHiddenPanels: () => [{ id: 'ariel', label: 'ARIEL' }],
      getVisiblePanels: () => [{ id: 'ariel', label: 'ARIEL' }],
      getPopoutPanels: () => [{ id: 'ariel', label: 'ARIEL' }],
      openPanelBeside: () => {},
    });

    const panels = inGroup(items, 'Panels');
    expect(panels).toHaveLength(4);
    for (const row of panels) {
      // Only searchText is scored by the matcher, so the alias has to live there.
      expect(row.searchText).toContain('logbook');
      expect(row.searchText).toContain('elog');
      // The label and id stay searchable alongside the aliases.
      expect(row.searchText).toContain('ARIEL');
      expect(row.searchText).toContain('ariel');
    }
  });

  it('SYNONYMS: every built-in panel id carries its domain aliases', () => {
    const expected = {
      ariel: ['logbook', 'elog'],
      'channel-finder': ['pv', 'channels'],
      artifacts: ['gallery', 'files'],
      lattice: ['optics'],
      okf: ['knowledge', 'docs'],
      'system-health': ['status', 'monitoring'],
    };
    const ids = Object.keys(expected);
    const items = buildRegistry({
      getVisiblePanels: () => ids.map((id) => ({ id, label: id.toUpperCase() })),
      focusPanel: () => {},
    });

    const panels = inGroup(items, 'Panels');
    for (const [id, aliases] of Object.entries(expected)) {
      const row = panels.find((it) => it.label === `Focus ${id.toUpperCase()}`);
      for (const alias of aliases) {
        expect(row.searchText).toContain(alias);
      }
    }
  });

  it('SYNONYMS: verb aliases ride the Open rows, not Show/Focus', () => {
    const items = buildRegistry({
      getHiddenPanels: () => [{ id: 'okf', label: 'Facility' }],
      getVisiblePanels: () => [{ id: 'okf', label: 'Facility' }],
      getPopoutPanels: () => [{ id: 'okf', label: 'Facility' }],
      openPanelBeside: () => {},
    });

    const byLabel = (/** @type {string} */ label) => inGroup(items, 'Panels').find((it) => it.label === label);
    const popout = byLabel('Open Facility in a new window').searchText;
    for (const token of ['window', 'popout', 'standalone', 'tab']) {
      expect(popout).toContain(token);
    }
    const beside = byLabel('Open Facility in a new tile').searchText;
    for (const token of ['tile', 'beside', 'split']) {
      expect(beside).toContain(token);
    }
    // The verb vocabularies do not bleed into each other or onto Show/Focus.
    expect(popout).not.toContain('beside');
    expect(beside).not.toContain('popout');
    expect(byLabel('Show Facility').searchText).not.toContain('popout');
    expect(byLabel('Focus Facility').searchText).not.toContain('beside');
  });

  it('SYNONYMS: an unknown (facility-custom) panel id contributes no aliases', () => {
    const items = buildRegistry({
      getVisiblePanels: () => [{ id: 'custom-thing', label: 'Custom Thing' }],
      focusPanel: () => {},
    });
    const [row] = inGroup(items, 'Panels');
    expect(row.searchText).toBe('focus Custom Thing custom-thing');
  });

  it('LAYOUTS: preset run applies the preset BY NAME', () => {
    /** @type {string[]} */
    const applied = [];
    const items = buildRegistry({
      getPresets: () => [{ name: 'Focus Mode', panels: ['chat', 'ariel'] }],
      applyPreset: (name) => applied.push(name),
    });

    const layouts = inGroup(items, 'Layouts');
    expect(layouts).toHaveLength(1);
    expect(layouts[0].label).toContain('Focus Mode');
    layouts[0].run();
    // The name is what the arrange request carries — members are resolved
    // server-side, so the palette never handles the panel list itself.
    expect(applied).toEqual(['Focus Mode']);
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
