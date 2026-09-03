/**
 * Customize Bars on a deployment that cannot render everything the shipped
 * default names, happy-dom environment:
 *   npx vitest run tests/interfaces/web_terminal/js/bar-default-renderable.test.js
 *
 * The regression this pins (#863): `DEFAULT_BAR_LAYOUT` places `system-health`,
 * which renders only where the SYSTEM panel is enabled. Served unfiltered at
 * rev 0 to a deployment without that panel, the normalizer dropped it as
 * `unavailable`, read the drop as lost content, and latched the whole sheet
 * read-only — every tile disabled, "Layout not editable", and Default could not
 * recover because it adopted the same document again. The server now filters
 * the default; the client must ALSO not latch on a rev-0 `unavailable` drop, so
 * a stale client or a flipped stamp still gets an editable sheet.
 *
 * The document here is the shipped default spelled out, not a hand-built one
 * that happens to avoid the item: the point is the arrangement a fresh
 * deployment actually serves.
 */

import { test, expect, describe, afterEach } from 'vitest';
import {
  boot,
  doc,
  endpoint,
  noticeText,
  pointer,
  putBodies,
  rendered,
  settle,
  shell,
  teardown,
  tile,
  withRect,
} from './bar-customize-fixture.mjs';

/** @type {any} */
let customize;
/** @type {any} */
let sync;

/** The shipped default, as `DEFAULT_BAR_LAYOUT` in `app.py` spells it. */
const SHIPPED_DEFAULT = doc(
  ['logo', 'identity', 'space', 'control-target', 'search', 'display'],
  ['space', 'system-health', 'clock']
);

/** A multi-user deployment whose `web_panels` does not select `system-health`. */
const NO_SYSTEM_PANEL = {
  identityAvailable: true,
  blueskyAvailable: false,
  systemHealthAvailable: false,
};

const HEADER_BOX = { left: 0, right: 300, top: 0, bottom: 40 };
const STATUS_BOX = { left: 0, right: 300, top: 100, bottom: 126 };
const IN_HEADER = { x: 150, y: 20 };
const IN_STATUS = { x: 150, y: 112 };

/** Every catalog type the sheet offers a tile for. @returns {string[]} */
function tileTypes() {
  return Array.from(document.querySelectorAll('.bar-tile[data-bar-tile]')).map(
    (/** @type {any} */ node) => node.dataset.barTile
  );
}

async function editing() {
  ({ customize, sync } = await boot({
    fetch: endpoint({ get: SHIPPED_DEFAULT }),
    context: NO_SYSTEM_PANEL,
  }));
  withRect(document.querySelector('[data-bar-host="header"]'), HEADER_BOX);
  withRect(document.querySelector('[data-bar-host="status"]'), STATUS_BOX);
  customize.enterEditMode();
}

afterEach(() => {
  teardown({ customize, sync });
  customize = null;
  sync = null;
});

describe('the shipped default on a deployment without the SYSTEM panel', () => {
  test('renders the degraded default and is not read-only', async () => {
    await editing();

    expect(rendered('status')).toEqual(['space', 'clock']);
    expect(sync.isLayoutReadonly()).toBe(false);
    expect(noticeText()).not.toContain('Layout not editable');
  });

  test('no tile is refused by the latch, and every unplaced offerable one is enabled', async () => {
    await editing();

    // The two gated items this deployment lacks are refused on their own
    // merits ("Not in this deployment"), and a placed single-node item dims
    // itself; neither is the latch's sentence.
    expect(tileTypes().length).toBeGreaterThan(5);
    for (const type of tileTypes()) {
      expect(tile(type).querySelector('.bar-tile-reason')?.textContent ?? '').not.toBe(
        'Layout not editable'
      );
    }
    for (const type of ['clock', 'stopwatch', 'space', 'separator', 'docs', 'feedback']) {
      expect(tile(type).disabled, `${type} tile`).toBe(false);
    }
  });

  test('the one tile that is refused says why, and it is not the latch', async () => {
    await editing();

    const unavailable = tile('system-health');
    expect(unavailable.disabled).toBe(true);
    expect(unavailable.querySelector('.bar-tile-reason')?.textContent).toBe(
      'Not in this deployment'
    );
    expect(customize.refusalFor('system-health', 'status')).not.toBe('Layout not editable');
  });

  test('a drag saves', async () => {
    await editing();

    pointer('pointerdown', shell('search'), IN_HEADER);
    pointer('pointermove', shell('search'), IN_STATUS);
    pointer('pointerup', shell('search'), IN_STATUS);
    await settle();

    const bodies = putBodies();
    expect(bodies).toHaveLength(1);
    expect(bodies[0].rev).toBe(0);
    expect(bodies[0].status.map((/** @type {any} */ i) => i.type)).toEqual([
      'space',
      'clock',
      'search',
    ]);
    expect(bodies[0].status.map((/** @type {any} */ i) => i.type)).not.toContain(
      'system-health'
    );
    expect(rendered('status')).toEqual(['space', 'clock', 'search']);
  });

  test('a tile click saves', async () => {
    await editing();

    await expect(customize.addItem('stopwatch', 'status')).resolves.toBe(true);
    expect(putBodies()).toHaveLength(1);
  });

  test('the status bar can be hidden', async () => {
    await editing();

    await expect(customize.setBarVisible('status', false)).resolves.toBe(true);
    expect(putBodies()).toHaveLength(1);
    expect(putBodies()[0].status_visible).toBe(false);
  });
});
