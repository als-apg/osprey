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
 * The two things that can go wrong are both silent in a browser: a stamp that
 * never arrives (the rail fills with ids, no error anywhere) and a second copy
 * of the labels drifting from the registry's. The first is pinned here; the
 * second is pinned by `test_panel_catalog_roster.py`, which reads this module
 * as source.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import { BUILTIN_PANEL_LABELS, stampPanelLabels } from './panel-labels-fixture.mjs';
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

beforeEach(() => {
  warnings = [];
  vi.spyOn(console, 'warn').mockImplementation((...args) => {
    warnings.push(args.map(String).join(' '));
  });
  document.documentElement.removeAttribute('data-panel-labels');
});

afterEach(() => {
  vi.restoreAllMocks();
  document.documentElement.removeAttribute('data-panel-labels');
});

describe('the built-in panel labels come from the page', () => {
  test('with the stamp, every panel wears the label the server sent', async () => {
    stampPanelLabels(document);
    const { PANELS } = await freshCatalog();

    expect(Object.fromEntries(PANELS.map((p) => [p.id, p.label]))).toEqual(BUILTIN_PANEL_LABELS);
    expect(warnings).toEqual([]);
  });

  test('without the stamp every panel wears its id, and one warning is logged', async () => {
    const { PANELS } = await freshCatalog();

    expect(PANELS.map((p) => p.label)).toEqual(PANELS.map((p) => p.id.toUpperCase()));
    expect(warnings).toHaveLength(1);
    expect(warnings[0]).toContain('data-panel-labels');
  });

  test('an unreadable stamp is treated as an absent one', async () => {
    document.documentElement.setAttribute('data-panel-labels', '{not json');
    const { PANELS } = await freshCatalog();

    expect(PANELS.find((p) => p.id === 'artifacts')?.label).toBe('ARTIFACTS');
    expect(warnings).toHaveLength(1);
  });

  test('a label follows the stamp, and a panel the stamp does not name falls back to its id', async () => {
    // A renamed label proves no copy of the roster rides in the bundle.
    document.documentElement.setAttribute(
      'data-panel-labels',
      JSON.stringify({ artifacts: 'FILES' })
    );
    const { PANELS } = await freshCatalog();

    expect(PANELS.find((p) => p.id === 'artifacts')?.label).toBe('FILES');
    expect(PANELS.find((p) => p.id === 'okf')?.label).toBe('OKF');
    expect(warnings).toEqual([]);
  });
});

describe('the descriptors the browser builds paths from', () => {
  test('the endpoints and health posture are the bundle’s own, stamp or no stamp', async () => {
    const { PANELS } = await freshCatalog();
    const byId = Object.fromEntries(PANELS.map((p) => [p.id, p]));

    expect(byId['artifacts'].configEndpoint).toBe('/api/artifact-server');
    // Explicit null: an embedded same-origin panel is not health-polled.
    expect(byId['artifacts'].healthEndpoint).toBeNull();
    // Explicit '/health': omitting it skips polling and pins the panel healthy,
    // which leaves the rail entry enabled with the sidecar down.
    expect(byId['system-health'].healthEndpoint).toBe('/health');
    expect(byId['jupyter'].healthEndpoint).toBe('/api/status');
  });
});

describe('how the sidebar names and draws a panel', () => {
  test('the Agent pane row is labelled Agent', async () => {
    const { TERMINAL_RAIL_ID, TERMINAL_RAIL_LABEL } = await freshCatalog();

    expect(TERMINAL_RAIL_ID).toBe('terminal');
    expect(TERMINAL_RAIL_LABEL).toBe('Agent');
  });

  test('every named glyph is one icons.js can draw, and so is the fallback', async () => {
    const { PANEL_ICONS, FALLBACK_PANEL_ICON } = await freshCatalog();

    for (const name of [...Object.values(PANEL_ICONS), FALLBACK_PANEL_ICON]) {
      expect(ICON_NAMES).toContain(name);
    }
    expect(FALLBACK_PANEL_ICON).toBe('workspace');
  });

  test('every shipped panel and the Agent pane have a glyph of their own', async () => {
    const { PANELS, PANEL_ICONS, TERMINAL_RAIL_ID } = await freshCatalog();

    for (const id of [TERMINAL_RAIL_ID, ...PANELS.map((p) => p.id)]) {
      expect(Object.hasOwn(PANEL_ICONS, id)).toBe(true);
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
