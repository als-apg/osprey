// @ts-check
/**
 * Unit tests for the server-gated Config drawer tab (config-tab.js and the
 * three consumers that must survive its absence):
 *
 *   npx vitest run tests/interfaces/web_terminal/config-tab.test.mjs
 *
 * `web.config_panel.enabled` is enforced on the server — `/api/config` and
 * `/api/claude-setup` answer 403, and `GET /api/panels` reports the posture as
 * `config_panel_enabled` (routes/panels.py). These tests pin the CLIENT half:
 * a deployment that withdrew the surface must not paint a control for it, and
 * nothing that used to reach for that control may break when it is gone.
 *
 * What is pinned, in the order a reviewer would ask about it:
 *
 * - `false` removes BOTH nodes — the drawer tab button and its panel — while
 *   the other tabs survive untouched; the Config tab is the only withdrawable
 *   one, and taking Behavior/Safety/Memory with it would be a broken drawer,
 *   not a gated one;
 * - `true`, a payload with no such key, and a null payload (failed or hung
 *   `/api/panels`) all LEAVE THE TAB. Absent means enabled, mirroring the
 *   server's own `getattr(..., True)` default; a read that never landed is not
 *   a statement about the deployment's posture, and treating it as one would
 *   make a flaky network look like a policy change;
 * - the removal is wired into the boot path — `initPanelManager`'s single
 *   `/api/panels` read applies it, so the tab actually goes away in the app
 *   rather than only in a unit test's fixture;
 * - the command palette offers no "Open Settings" row and withholds its
 *   `revealSetting` dep once the tab is absent (a row that walks the operator
 *   through the unsaved-changes gate to reach a tab that cannot resolve is a
 *   dead control), and palette.js then skips its `/api/config` read entirely
 *   rather than rendering "Settings unavailable" for a withdrawn surface;
 * - the scaffold gallery still wires Behavior and Safety with the Config
 *   sections gone. This one is a regression pin with teeth: the pre-gate code
 *   bailed out of `initScaffoldGallery` wholesale unless `#config-gallery-
 *   section` existed, so gating Config off would have taken the Prompt and
 *   Safety galleries down with it.
 *
 * Module isolation: palette.js and palette-boot.js keep module-private state
 * (the open palette, the current dep bundle), so the palette suites use
 * vi.resetModules() + a fresh dynamic import — the pattern panel-manager.test
 * .mjs and sessions.test.mjs use for the same reason.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import {
  applyConfigTabGate,
  CONFIG_TAB_ID,
} from '../../../src/osprey/interfaces/web_terminal/static/js/config-tab.js';

const JS = '../../../src/osprey/interfaces/web_terminal/static/js';

/** The drawer markup index.html ships: four tabs, four panels, Behavior active. */
function renderDrawer() {
  document.body.innerHTML = `
    <osprey-drawer id="settings-drawer">
      <div class="drawer-tabs">
        <button class="drawer-tab active" data-tab="tab-behavior">Behavior</button>
        <button class="drawer-tab" data-tab="tab-safety">Safety</button>
        <button class="drawer-tab" data-tab="tab-memory">Memory</button>
        <button class="drawer-tab" data-tab="tab-config">Config</button>
      </div>
      <div class="drawer-tab-panel active" id="tab-behavior">
        <div id="behavior-gallery-section"></div>
      </div>
      <div class="drawer-tab-panel" id="tab-safety">
        <div id="safety-gallery-section"></div>
      </div>
      <div class="drawer-tab-panel" id="tab-memory">
        <div id="memory-gallery-section"></div>
      </div>
      <div class="drawer-tab-panel" id="tab-config">
        <div id="config-gallery-section"></div>
        <div id="config-form-section"></div>
      </div>
    </osprey-drawer>
  `;
}

/** @returns {boolean} whether the Config tab button is still in the document. */
function tabButtonPresent() {
  return !!document.querySelector(`.drawer-tab[data-tab="${CONFIG_TAB_ID}"]`);
}

/** @returns {boolean} whether the Config tab PANEL is still in the document. */
function tabPanelPresent() {
  return !!document.getElementById(CONFIG_TAB_ID);
}

/** @returns {string[]} the `data-tab` value of every surviving drawer tab, in order. */
function tabOrder() {
  return [...document.querySelectorAll('.drawer-tab')].map((t) => t.getAttribute('data-tab') || '');
}

afterEach(() => {
  document.body.innerHTML = '';
});

describe('applyConfigTabGate', () => {
  beforeEach(renderDrawer);

  test('config_panel_enabled false removes the tab button AND its panel', () => {
    const removed = applyConfigTabGate({ config_panel_enabled: false });

    expect(removed).toBe(true);
    expect(tabButtonPresent()).toBe(false);
    expect(tabPanelPresent()).toBe(false);
    // Removal, not hiding: the button is out of the tab order, and every
    // downstream guard keys on absence.
    expect(tabOrder()).toEqual(['tab-behavior', 'tab-safety', 'tab-memory']);
    // The tab it does not own stays whole — panel included.
    expect(document.getElementById('tab-memory')).not.toBeNull();
    expect(document.getElementById('behavior-gallery-section')).not.toBeNull();
  });

  test('config_panel_enabled true keeps the tab', () => {
    expect(applyConfigTabGate({ config_panel_enabled: true })).toBe(false);
    expect(tabButtonPresent()).toBe(true);
    expect(tabPanelPresent()).toBe(true);
  });

  test('a payload without the key keeps the tab (absent means enabled)', () => {
    expect(applyConfigTabGate({ enabled: ['artifacts'], visible: [] })).toBe(false);
    expect(tabButtonPresent()).toBe(true);
  });

  test('a null payload — a failed or hung /api/panels — keeps the tab', () => {
    expect(applyConfigTabGate(null)).toBe(false);
    expect(tabButtonPresent()).toBe(true);
  });

  test('a truthy non-boolean flag is not a withdrawal', () => {
    // Only an explicit `false` withdraws: the server sends a real bool, and a
    // string "false" arriving here would mean the payload contract changed
    // under us — something to fix at the source, not to guess at.
    expect(applyConfigTabGate({ config_panel_enabled: 'false' })).toBe(false);
    expect(tabButtonPresent()).toBe(true);
  });

  test('a second application is a no-op once the tab is gone', () => {
    expect(applyConfigTabGate({ config_panel_enabled: false })).toBe(true);
    expect(applyConfigTabGate({ config_panel_enabled: false })).toBe(false);
    expect(tabOrder()).toEqual(['tab-behavior', 'tab-safety', 'tab-memory']);
  });

  test('removing the ACTIVE tab promotes the first survivor, so the drawer is never blank', () => {
    // Behavior is the boot-active tab today; this is insurance against a
    // reorder that makes Config the active one.
    document.querySelector('.drawer-tab[data-tab="tab-behavior"]')?.classList.remove('active');
    document.getElementById('tab-behavior')?.classList.remove('active');
    document.querySelector(`.drawer-tab[data-tab="${CONFIG_TAB_ID}"]`)?.classList.add('active');
    document.getElementById(CONFIG_TAB_ID)?.classList.add('active');

    applyConfigTabGate({ config_panel_enabled: false });

    const active = document.querySelector('.drawer-tab.active');
    expect(active?.getAttribute('data-tab')).toBe('tab-behavior');
    expect(document.getElementById('tab-behavior')?.classList.contains('active')).toBe(true);
  });

  test('a root option scopes the lookup to one subtree', () => {
    const other = document.createElement('div');
    other.innerHTML = '<button class="drawer-tab" data-tab="tab-config">Config</button>';
    document.body.appendChild(other);

    expect(applyConfigTabGate({ config_panel_enabled: false }, { root: other })).toBe(true);
    expect(other.querySelector('.drawer-tab')).toBeNull();
    // The drawer's own tab is untouched — the gate acted only inside `root`.
    expect(tabPanelPresent()).toBe(true);
  });
});

describe('boot wiring: initPanelManager applies the gate', () => {
  // dock-workspace.js fronts the vendored dockview shell; stubbed at the module
  // boundary the way panel-manager.test.mjs stubs it, so this suite exercises
  // the real /api/panels read without a dock.
  //
  // Registered with vi.doMock, NOT vi.mock: vi.mock is hoisted to the top of
  // the FILE, and the palette suite below mocks panel-manager.js itself — a
  // hoisted registration would hand that mock to this suite (and this suite's
  // dock stub to every other). doMock is scoped to the dynamic imports that
  // follow it, and each suite unregisters what it registered.
  const DOCK = `${JS}/dock-workspace.js`;

  function mockDock() {
    vi.doMock(DOCK, () => ({
      getDockApi: () => null,
      openTerminalPanel: () => {},
      closeTerminalPanel: () => {},
      // Pulled in transitively by dock-iframe.js; absent exports would resolve
      // to undefined and throw at call time.
      defaultServiceWidth: () => 600,
      setServiceRedock: () => {},
      onDragGesture: () => [],
    }));
  }

  /** @param {any} body */
  const jsonOk = (body) => ({ ok: true, status: 200, statusText: 'OK', json: async () => body });

  /** @param {boolean} configPanelEnabled */
  function stubPanelsFetch(configPanelEnabled) {
    vi.stubGlobal('fetch', vi.fn(async (/** @type {string} */ url) => {
      if (url === '/api/panels') {
        return jsonOk({
          enabled: [],
          custom: [],
          default: null,
          visible: [],
          active: null,
          labels: {},
          config_panel_enabled: configPanelEnabled,
        });
      }
      return jsonOk({});
    }));
    class FakeEventSource {
      constructor() {
        /** @type {((e: {data: string}) => void)|null} */
        this.onmessage = null;
      }
      close() {}
    }
    vi.stubGlobal('EventSource', FakeEventSource);
  }

  beforeEach(() => {
    renderDrawer();
    document.body.insertAdjacentHTML(
      'beforeend',
      '<nav id="panel-rail"></nav><div id="panel-manager"><div id="panel-content"></div></div>',
    );
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.doUnmock(DOCK);
    vi.resetModules();
  });

  /** @param {boolean} configPanelEnabled */
  async function boot(configPanelEnabled) {
    stubPanelsFetch(configPanelEnabled);
    vi.resetModules();
    mockDock();
    const { initPanelManager } = await import(
      '../../../src/osprey/interfaces/web_terminal/static/js/panel-manager.js'
    );
    await initPanelManager('panel-manager');
  }

  test('a disabled deployment loses the Config tab at boot', async () => {
    await boot(false);

    expect(tabButtonPresent()).toBe(false);
    expect(tabPanelPresent()).toBe(false);
  });

  test('an enabled deployment keeps it', async () => {
    await boot(true);

    expect(tabButtonPresent()).toBe(true);
    expect(tabPanelPresent()).toBe(true);
  });
});

describe('command palette guards on the tab being gone', () => {
  /** Captures what `initCommandPalette`'s entry points hand `openPalette`. */
  const opens = /** @type {any[]} */ ([]);

  // palette-boot.js is pure wiring: every collaborator it imports is stubbed
  // at the module boundary, so what this suite reads is exactly the dep bundle
  // it BUILDS. All doMock (see the dock note above) and unregistered in
  // afterEach — palette.js and panel-manager.js are the real modules two other
  // suites in this file exercise.
  /** @type {Record<string, () => any>} */
  const PALETTE_BOOT_MOCKS = {
    'palette.js': () => ({
      openPalette: (/** @type {any} */ deps) => opens.push(deps),
      closePalette: () => {},
      isOpen: () => false,
    }),
    'terminal.js': () => ({ restartTerminal: async () => {}, startTerminal: () => {} }),
    'panel-manager.js': () => ({
      getHiddenPanels: () => [],
      getVisiblePanels: () => [],
      getPopoutPanels: () => [],
      getPresets: () => [],
      showPanel: () => {},
      activateTab: () => {},
      applyMenuPreset: () => {},
      popoutPanel: () => {},
    }),
    'panel-placement.js': () => ({ openPanelBeside: () => {} }),
    'settings.js': () => ({ openDrawerTab: async () => true, revealSetting: async () => {} }),
    'sessions.js': () => ({ startNewSession: () => {} }),
    'rail-position.js': () => ({ setRailPosition: () => {} }),
    'feedback-modal.js': () => ({ isFeedbackModalOpen: () => false }),
  };

  afterEach(() => {
    for (const name of Object.keys(PALETTE_BOOT_MOCKS)) vi.doUnmock(`${JS}/${name}`);
    vi.resetModules();
  });

  /** Open the palette through the header trigger and return the deps it built. */
  async function openThroughTrigger() {
    opens.length = 0;
    vi.resetModules();
    for (const [name, factory] of Object.entries(PALETTE_BOOT_MOCKS)) {
      vi.doMock(`${JS}/${name}`, factory);
    }
    const { initCommandPalette } = await import(
      '../../../src/osprey/interfaces/web_terminal/static/js/palette-boot.js'
    );
    initCommandPalette();
    document.getElementById('command-palette-btn')?.dispatchEvent(
      new MouseEvent('click', { bubbles: true }),
    );
    expect(opens).toHaveLength(1);
    return opens[0];
  }

  beforeEach(() => {
    renderDrawer();
    document.body.insertAdjacentHTML('beforeend', '<button id="command-palette-btn"></button>');
  });

  test('with the tab present: the Open Settings row and the revealSetting dep are offered', async () => {
    const deps = await openThroughTrigger();

    expect(deps.actions.map((/** @type {any} */ a) => a.label)).toContain('Open Settings');
    expect(typeof deps.revealSetting).toBe('function');
  });

  test('with the tab gone: no Open Settings row, no revealSetting dep, no throw', async () => {
    applyConfigTabGate({ config_panel_enabled: false });

    const deps = await openThroughTrigger();

    expect(deps.actions.map((/** @type {any} */ a) => a.label)).not.toContain('Open Settings');
    expect(deps.revealSetting).toBeUndefined();
    // The tabs that were not withdrawn are untouched — the guard is specific
    // to Config, not a blanket drop of the drawer rows.
    expect(deps.actions.map((/** @type {any} */ a) => a.label)).toEqual(
      expect.arrayContaining(['Open Memory gallery', 'Open Prompt gallery']),
    );
  });
});

describe('palette.js skips its /api/config read without revealSetting', () => {
  /** @type {any} */
  let paletteMod;

  beforeEach(async () => {
    document.body.innerHTML = '';
    vi.resetModules();
    paletteMod = await import(
      '../../../src/osprey/interfaces/web_terminal/static/js/palette.js'
    );
  });

  afterEach(() => {
    paletteMod?.closePalette();
  });

  test('no fetch, and no "Settings unavailable" row, when the dep is withheld', async () => {
    const fetchConfig = vi.fn(async () => ({ sections: {} }));

    paletteMod.openPalette({ getHiddenPanels: () => [], getVisiblePanels: () => [], fetchConfig });
    await Promise.resolve();
    await Promise.resolve();

    expect(fetchConfig).not.toHaveBeenCalled();
    expect(document.body.textContent).not.toContain('Settings unavailable');
    expect(document.body.textContent).not.toContain('Loading settings');
  });

  test('the read still fires for a deployment that kept the tab', async () => {
    const fetchConfig = vi.fn(async () => ({ sections: { web: { theme: 'dark' } } }));

    paletteMod.openPalette({
      getHiddenPanels: () => [],
      getVisiblePanels: () => [],
      revealSetting: () => {},
      fetchConfig,
    });
    await Promise.resolve();
    await Promise.resolve();

    expect(fetchConfig).toHaveBeenCalledTimes(1);
  });
});

describe('scaffold gallery survives the Config sections being gone', () => {
  /** A drawer stand-in carrying the unsaved-guard registrar the gallery calls. */
  function drawerWithGuard() {
    const drawer = /** @type {any} */ (document.getElementById('settings-drawer'));
    drawer.registerUnsavedGuard = vi.fn();
    return drawer;
  }

  beforeEach(() => {
    renderDrawer();
    // Activating a tab makes its gallery load from /api/scaffold. Stubbed to
    // an empty catalog so the suite stays offline — happy-dom would otherwise
    // try the real origin and log a connection refusal per test.
    vi.stubGlobal('fetch', vi.fn(async () => ({
      ok: true,
      status: 200,
      statusText: 'OK',
      json: async () => [],
    })));
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  test('Behavior and Safety still wire up after the gate removed the Config tab', async () => {
    const drawer = drawerWithGuard();
    applyConfigTabGate({ config_panel_enabled: false });

    vi.resetModules();
    const { initScaffoldGallery } = await import(
      '../../../src/osprey/interfaces/web_terminal/static/js/scaffold-gallery.js'
    );
    expect(() => initScaffoldGallery()).not.toThrow();

    // The registrar firing is the proof init ran to completion rather than
    // bailing at the (now absent) Config sections — the pre-gate code returned
    // early there and took both surviving galleries with it.
    expect(drawer.registerUnsavedGuard).toHaveBeenCalledTimes(1);
    // Each surviving tab still loads its gallery on activation, and neither
    // path reaches for a Config element.
    expect(() => {
      document.getElementById('tab-behavior')?.dispatchEvent(new Event('drawer:tab-activate'));
      document.getElementById('tab-safety')?.dispatchEvent(new Event('drawer:tab-activate'));
    }).not.toThrow();
    // The composite unsaved guard must not read `.editDirty` off a gallery
    // that was never built.
    const guard = drawer.registerUnsavedGuard.mock.calls[0][0];
    expect(guard()).toBe(true);
  });

  test('the drawer close handler does not reach for the absent Config gallery', async () => {
    const drawer = drawerWithGuard();
    applyConfigTabGate({ config_panel_enabled: false });

    vi.resetModules();
    const { initScaffoldGallery } = await import(
      '../../../src/osprey/interfaces/web_terminal/static/js/scaffold-gallery.js'
    );
    initScaffoldGallery();

    expect(() => drawer.dispatchEvent(new Event('drawer:close'))).not.toThrow();
  });
});
