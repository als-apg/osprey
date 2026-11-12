// @ts-check
/**
 * The shipped panel catalog (panel-catalog.js):
 *   npx vitest run tests/interfaces/web_terminal/panel-catalog.test.mjs
 *
 * The catalog is the browser's descriptor list for the built-in panels — the
 * id every rail entry and proxy path is built from, the config endpoint, and
 * whether the panel is health-polled. The DISPLAY LABEL is not its to decide:
 * `osprey.profiles.web_panels` owns the roster, and `/api/panels` answers for
 * the ENABLED panels only, too late and too narrow to seed a module-level
 * array. So the page stamps the whole roster on `<html>` and the catalog reads
 * it at load.
 *
 * The sidecar panels are not in the bundle at all: the registry describes each
 * one whole, the page stamps those descriptors beside the labels, and the
 * catalog builds a descriptor per entry.
 *
 * The two things that can go wrong are both silent in a browser: a stamp that
 * never arrives (the rail fills with ids, no error anywhere) and a second copy
 * of the labels drifting from the registry's. The first is pinned here; the
 * second is pinned by `test_panel_catalog_roster.py`, which reads this module
 * as source.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import { BUILTIN_PANEL_LABELS, SIDECAR_PANELS, stampPanelLabels } from './panel-labels-fixture.mjs';
import { ICON_NAMES } from '/design-system/js/icons.js';

/** Every warning logged while the catalog loaded. @type {string[]} */
let warnings;

/**
 * Load a fresh copy of the catalog against whatever `<html>` currently says.
 * @returns {Promise<typeof import('../../../src/osprey/interfaces/web_terminal/static/js/panel-catalog.js')>}
 */
async function freshCatalog() {
  vi.resetModules();
  return import('../../../src/osprey/interfaces/web_terminal/static/js/panel-catalog.js');
}

function clearStamps() {
  document.documentElement.removeAttribute('data-panel-labels');
  document.documentElement.removeAttribute('data-sidecar-panels');
}

/** Stamp the sidecar descriptors alone, leaving the labels as they are. @param {unknown} value */
function stampSidecars(value) {
  document.documentElement.setAttribute(
    'data-sidecar-panels',
    typeof value === 'string' ? value : JSON.stringify(value)
  );
}

/**
 * The companion panels of a catalog: every entry that is not a stamped sidecar.
 * @template {{ id: string }} T
 * @param {T[]} panels
 * @returns {T[]}
 */
function companionsOf(panels) {
  return panels.filter((p) => !SIDECAR_PANELS.some((s) => s.id === p.id));
}

beforeEach(() => {
  warnings = [];
  vi.spyOn(console, 'warn').mockImplementation((...args) => {
    warnings.push(args.map(String).join(' '));
  });
  clearStamps();
});

afterEach(() => {
  vi.restoreAllMocks();
  clearStamps();
});

describe('the built-in panel labels come from the page', () => {
  test('with the stamp, every panel wears the label the server sent, in title case', async () => {
    stampPanelLabels(document);
    const { PANELS, titleCase } = await freshCatalog();

    expect(Object.fromEntries(PANELS.map((p) => [p.id, p.label]))).toEqual(
      Object.fromEntries(
        Object.entries(BUILTIN_PANEL_LABELS).map(([id, label]) => [id, titleCase(label)])
      )
    );
    expect(warnings).toEqual([]);
  });

  test('without the stamp every panel wears its id in title case, and one warning is logged', async () => {
    stampSidecars(SIDECAR_PANELS);
    const { PANELS, titleCase } = await freshCatalog();

    const companions = companionsOf(PANELS);
    expect(companions.map((p) => p.label)).toEqual(
      companions.map((p) => titleCase(p.id.toUpperCase()))
    );
    expect(warnings).toHaveLength(1);
    expect(warnings[0]).toContain('data-panel-labels');
  });

  test('an unreadable stamp is treated as an absent one', async () => {
    stampSidecars(SIDECAR_PANELS);
    document.documentElement.setAttribute('data-panel-labels', '{not json');
    const { PANELS } = await freshCatalog();

    expect(PANELS.find((p) => p.id === 'artifacts')?.label).toBe('Artifacts');
    expect(warnings).toHaveLength(1);
  });

  test('a label follows the stamp, and a panel the stamp does not name falls back to its id', async () => {
    stampSidecars(SIDECAR_PANELS);
    // A renamed label proves no copy of the roster rides in the bundle. An
    // all-caps stamp reads in title case; a mixed-case one is kept as spelled.
    document.documentElement.setAttribute(
      'data-panel-labels',
      JSON.stringify({ artifacts: 'FILES', ariel: 'eLog' })
    );
    const { PANELS } = await freshCatalog();

    expect(PANELS.find((p) => p.id === 'artifacts')?.label).toBe('Files');
    expect(PANELS.find((p) => p.id === 'ariel')?.label).toBe('eLog');
    expect(PANELS.find((p) => p.id === 'okf')?.label).toBe('Okf');
    expect(warnings).toEqual([]);
  });
});

describe('the descriptors the browser builds paths from', () => {
  test('a companion panel’s endpoints and health posture are the bundle’s own, stamp or no stamp', async () => {
    const { PANELS } = await freshCatalog();
    const byId = Object.fromEntries(PANELS.map((p) => [p.id, p]));

    expect(byId['artifacts'].configEndpoint).toBe('/api/artifact-server');
    // Explicit null: an embedded same-origin panel is not health-polled.
    expect(byId['artifacts'].healthEndpoint).toBeNull();
    // Explicit '/health': omitting it skips polling and pins the panel healthy,
    // which leaves the sidebar row enabled with the sidecar down.
    expect(byId['system-health'].healthEndpoint).toBe('/health');
  });
});

describe('the sidecar panels come from the page', () => {
  test('the notebook panel is built from its stamped descriptor, where it has always sat', async () => {
    stampPanelLabels(document);
    const { PANELS, panelIconName, describePanel } = await freshCatalog();

    expect(PANELS.map((p) => p.id)).toEqual([
      'artifacts',
      'ariel',
      'channel-finder',
      'jupyter',
      'canopy',
      'okf',
      'system-health',
    ]);
    expect(PANELS.find((p) => p.id === 'jupyter')).toEqual({
      id: 'jupyter',
      label: 'Jupyter',
      icon: 'notebook',
      description: 'JupyterLab with kernels that follow the terminal session',
      configEndpoint: '/api/jupyter-server',
      healthEndpoint: '/api/status',
      startEndpoint: '/api/panels/jupyter/start',
    });
    expect(panelIconName('jupyter')).toBe('notebook');
    expect(describePanel('jupyter')).toBe(
      'JupyterLab with kernels that follow the terminal session'
    );
    expect(warnings).toEqual([]);
  });

  test('a sidecar the bundle has never heard of gets a row, on the generic routes', async () => {
    stampPanelLabels(document);
    stampSidecars([
      ...SIDECAR_PANELS,
      { id: 'screens', label: 'SCREENS', icon: null, description: null, healthEndpoint: null },
    ]);
    const { PANELS, panelIconName, describePanel, FALLBACK_PANEL_ICON } = await freshCatalog();

    expect(PANELS.find((p) => p.id === 'screens')).toEqual({
      id: 'screens',
      label: 'Screens',
      configEndpoint: '/api/panels/screens/server',
      // null, as the registry states it: this panel is not health-polled.
      healthEndpoint: null,
      startEndpoint: '/api/panels/screens/start',
    });
    expect(panelIconName('screens')).toBe(FALLBACK_PANEL_ICON);
    expect(describePanel('screens')).toBeNull();
  });

  test('a descriptor with no label wears its id, and one with no id is skipped', async () => {
    stampPanelLabels(document);
    stampSidecars([{ id: 'screens' }, { label: 'NAMELESS' }, null]);
    const { PANELS } = await freshCatalog();

    expect(PANELS.find((p) => p.id === 'screens')?.label).toBe('Screens');
    expect(PANELS).toHaveLength(6);
  });

  test.each([
    ['absent', null],
    ['unreadable', '[not json'],
    ['not a list', '{"jupyter": {}}'],
  ])('with the stamp %s there is no sidecar row, and one warning says so', async (_name, raw) => {
    document.documentElement.setAttribute(
      'data-panel-labels',
      JSON.stringify(BUILTIN_PANEL_LABELS)
    );
    if (raw !== null) stampSidecars(raw);
    const { PANELS } = await freshCatalog();

    expect(PANELS.map((p) => p.id)).toEqual([
      'artifacts',
      'ariel',
      'channel-finder',
      'okf',
      'system-health',
    ]);
    expect(warnings).toHaveLength(1);
    expect(warnings[0]).toContain('data-sidecar-panels');
  });
});

describe('how the sidebar names and draws a panel', () => {
  test('the Agent pane row is labelled Sessions; the pane keeps its name', async () => {
    const { TERMINAL_RAIL_ID, TERMINAL_RAIL_LABEL, TERMINAL_PANE_LABEL } = await freshCatalog();

    expect(TERMINAL_RAIL_ID).toBe('terminal');
    expect(TERMINAL_RAIL_LABEL).toBe('Sessions');
    expect(TERMINAL_PANE_LABEL).toBe('Agent');
  });

  test('every named glyph is one icons.js can draw, and so is the fallback', async () => {
    stampPanelLabels(document);
    const { PANELS, PANEL_ICONS, FALLBACK_PANEL_ICON, panelIconName } = await freshCatalog();

    for (const name of [...Object.values(PANEL_ICONS), FALLBACK_PANEL_ICON]) {
      expect(ICON_NAMES).toContain(name);
    }
    for (const panel of PANELS) {
      expect(ICON_NAMES).toContain(panelIconName(panel.id));
    }
    expect(FALLBACK_PANEL_ICON).toBe('workspace');
  });

  test('every shipped panel and the Agent pane have a glyph of their own', async () => {
    stampPanelLabels(document);
    const { PANELS, PANEL_ICONS, TERMINAL_RAIL_ID, panelIconName, FALLBACK_PANEL_ICON } =
      await freshCatalog();

    for (const id of [TERMINAL_RAIL_ID, ...companionsOf(PANELS).map((p) => p.id)]) {
      expect(Object.hasOwn(PANEL_ICONS, id)).toBe(true);
    }
    for (const panel of PANELS) {
      expect(panelIconName(panel.id)).not.toBe(FALLBACK_PANEL_ICON);
    }
  });

  test('panelIconName answers the named glyph, else the fallback', async () => {
    const { panelIconName, PANEL_ICONS, FALLBACK_PANEL_ICON } = await freshCatalog();

    expect(panelIconName('artifacts')).toBe(PANEL_ICONS.artifacts);
    expect(panelIconName('terminal')).toBe('agent');
    expect(panelIconName('my-custom-panel')).toBe(FALLBACK_PANEL_ICON);
    // An inherited Object property is not a panel's glyph.
    expect(panelIconName('constructor')).toBe(FALLBACK_PANEL_ICON);
  });

  test('every shipped panel, the Agent pane and Events have a description', async () => {
    stampPanelLabels(document);
    const { PANELS, PANEL_DESCRIPTIONS, TERMINAL_RAIL_ID, describePanel } = await freshCatalog();

    for (const id of [TERMINAL_RAIL_ID, 'events', ...companionsOf(PANELS).map((p) => p.id)]) {
      expect(Object.hasOwn(PANEL_DESCRIPTIONS, id)).toBe(true);
    }
    for (const panel of PANELS) {
      expect(describePanel(panel.id)).toBeTruthy();
    }
  });

  test('describePanel answers the sentence of a known panel, else null', async () => {
    const { describePanel, PANEL_DESCRIPTIONS } = await freshCatalog();

    expect(describePanel('ariel')).toBe('Search the electronic logbook');
    expect(describePanel('terminal')).toBe(PANEL_DESCRIPTIONS.terminal);
    expect(describePanel('my-custom-panel')).toBeNull();
    // An inherited Object property is not a panel's description.
    expect(describePanel('constructor')).toBeNull();
  });

  test.each([
    ['ARIEL', 'Ariel'],
    ['CHANNELS', 'Channels'],
    ['SYSTEM HEALTH', 'System Health'],
    ['CHANNEL-FINDER', 'Channel-Finder'],
    ['MY-CUSTOM PANEL', 'My-Custom Panel'],
  ])('titleCase reads the all-capitals label %s as %s', async (label, shown) => {
    const { titleCase } = await freshCatalog();

    expect(titleCase(label)).toBe(shown);
  });

  test.each(['EPICS Archiver', 'pyAT', 'Artifacts', 'Agent'])(
    'titleCase leaves %s, which has a lower-case letter, as it is',
    async (label) => {
      const { titleCase } = await freshCatalog();

      expect(titleCase(label)).toBe(label);
    }
  );
});

describe('an application panel names its own row', () => {
  /** A custom panel def as /api/panels answers it for an application panel. */
  const DIAL = {
    id: 'dial',
    label: 'DIAL',
    url: '/panel/dial',
    healthEndpoint: '/health',
    path: '/ui/',
    app: 'dial',
    icon: 'sliders',
    description: 'Turn the demo dial and watch it settle',
    help: 'how-to/apps/build-your-own-app.html',
  };

  test('the catalog entry built from the def carries its icon, description and help', async () => {
    const { customPanelEntry } = await freshCatalog();

    expect(customPanelEntry(DIAL)).toEqual({
      id: 'dial',
      label: 'Dial',
      configEndpoint: null,
      healthEndpoint: '/health',
      path: '/ui/',
      icon: 'sliders',
      description: 'Turn the demo dial and watch it settle',
      help: 'how-to/apps/build-your-own-app.html',
    });
  });

  test('a def that declares none of them builds the entry it always did', async () => {
    const { customPanelEntry } = await freshCatalog();

    expect(customPanelEntry({ id: 'grafana', label: 'GRAFANA', healthEndpoint: null })).toEqual({
      id: 'grafana',
      label: 'Grafana',
      configEndpoint: null,
      healthEndpoint: null,
      path: '/',
    });
    const blank = customPanelEntry({ ...DIAL, icon: '', description: 7, help: null });
    expect(blank).not.toHaveProperty('icon');
    expect(blank).not.toHaveProperty('description');
    expect(blank).not.toHaveProperty('help');
  });

  test('the row wears the declared glyph and tooltip; one it does not declare falls back', async () => {
    const { PANELS, customPanelEntry, panelIconName, describePanel, FALLBACK_PANEL_ICON } =
      await freshCatalog();

    PANELS.push(customPanelEntry(DIAL), customPanelEntry({ id: 'plain', label: 'PLAIN' }));

    expect(ICON_NAMES).toContain(DIAL.icon);
    expect(panelIconName('dial')).toBe('sliders');
    expect(describePanel('dial')).toBe(DIAL.description);
    expect(panelIconName('plain')).toBe(FALLBACK_PANEL_ICON);
    expect(describePanel('plain')).toBeNull();
  });

  test('a declared glyph icons.js does not define falls back instead of breaking the row', async () => {
    const { PANELS, customPanelEntry, panelIconName, FALLBACK_PANEL_ICON } = await freshCatalog();

    PANELS.push(customPanelEntry({ ...DIAL, icon: 'no-such-glyph' }));

    expect(ICON_NAMES).not.toContain('no-such-glyph');
    expect(panelIconName('dial')).toBe(FALLBACK_PANEL_ICON);
    expect(ICON_NAMES).toContain(panelIconName('dial'));
  });

  test('built-in and sidecar rows are unchanged by a declared application panel', async () => {
    stampPanelLabels(document);
    const { PANELS, PANEL_ICONS, customPanelEntry, panelIconName, describePanel } =
      await freshCatalog();
    const ids = PANELS.map((p) => p.id);
    /** @param {string} id */
    const row = (id) => [id, panelIconName(id), describePanel(id)];
    const before = ids.map(row);

    PANELS.push(customPanelEntry(DIAL));

    expect(ids.map(row)).toEqual(before);
    expect(panelIconName('terminal')).toBe(PANEL_ICONS.terminal);
  });
});
