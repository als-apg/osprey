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

  test('a label the stamp renames follows the stamp, not a copy in the bundle', async () => {
    document.documentElement.setAttribute(
      'data-panel-labels',
      JSON.stringify({ ...BUILTIN_PANEL_LABELS, artifacts: 'FILES' })
    );
    const { PANELS } = await freshCatalog();

    expect(PANELS.find((p) => p.id === 'artifacts')?.label).toBe('FILES');
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

  test('a panel the stamp does not name falls back to its id', async () => {
    document.documentElement.setAttribute(
      'data-panel-labels',
      JSON.stringify({ artifacts: 'WORKSPACE' })
    );
    const { PANELS } = await freshCatalog();

    expect(PANELS.find((p) => p.id === 'artifacts')?.label).toBe('WORKSPACE');
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
