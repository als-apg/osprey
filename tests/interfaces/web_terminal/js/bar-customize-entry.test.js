/**
 * Bar customize — the ways in, in both ui modes, happy-dom:
 *   npx vitest run tests/interfaces/web_terminal/js/bar-customize-entry.test.js
 *
 * The distinction these assertions exist to hold:
 *
 *   - the SAVED LAYOUT renders in both ui modes. The bars are the operator's
 *     chrome, not the mode's, so a flip to Simple must not move, drop or
 *     rewrite a single item — and must not write anything either.
 *
 *   - the three ways to REARRANGE it are present in both modes too: right-click
 *     on either bar, the row projected into the display menu beside the View
 *     toggle, and the palette action. Simple simplifies the workspace, not the
 *     operator's hand on their own chrome, and a flip in either direction
 *     neither adds nor removes a way in.
 */

import { test, expect, describe, afterEach, vi } from 'vitest';
import {
  boot,
  doc,
  endpoint,
  putBodies,
  rendered,
  settle,
  teardown,
} from './bar-customize-fixture.mjs';

const ENTRY_PATH =
  '../../../../src/osprey/interfaces/web_terminal/static/js/bar-customize-entry.js';

/** @type {any} */
let customize;
/** @type {any} */
let sync;

const PALETTE_BOOT_PATH =
  '../../../../src/osprey/interfaces/web_terminal/static/js/palette-boot.js';

/** The Customize row in the display menu's action row, if it is mounted. */
function displayRow() {
  return /** @type {any} */ (document.querySelector('.display-menu-actions .bar-customize-entry'));
}

/** Right-click a node. @param {any} node */
function rightClick(node) {
  node.dispatchEvent(
    new MouseEvent('contextmenu', { bubbles: true, cancelable: true, clientX: 10, clientY: 10 })
  );
}

/** The header host element. */
function headerHost() {
  return document.querySelector('[data-bar-host="header"]');
}

/** Flip the page's ui mode and let the watcher see it. @param {string} mode */
async function switchMode(mode) {
  document.documentElement.setAttribute('data-ui-mode', mode);
  await settle();
}

/** Every action label the palette is currently showing. */
function paletteLabels() {
  return Array.from(document.querySelectorAll('.command-palette-item-label')).map(
    (node) => node.textContent ?? ''
  );
}

/**
 * Open the palette the way the header trigger does, through palette-boot's own
 * dependency builder — the point of the test is what THAT assembles.
 */
async function openPalette() {
  const trigger = document.createElement('button');
  trigger.id = 'command-palette-btn';
  document.body.append(trigger);
  const paletteBoot = await import(PALETTE_BOOT_PATH);
  paletteBoot.initCommandPalette();
  trigger.click();
  await settle();
}

afterEach(() => {
  teardown({ customize, sync });
  customize = null;
  sync = null;
});

describe('the display-menu row', () => {
  test('Expert mode projects a Customize row into the action row', async () => {
    ({ customize, sync } = await boot({ menu: true }));

    expect(displayRow()).not.toBe(null);
    expect(displayRow().textContent).toBe('Customize bars');
  });

  test('clicking it enters edit mode', async () => {
    ({ customize, sync } = await boot({ menu: true }));

    displayRow().click();

    expect(customize.isEditing()).toBe(true);
  });

  test('Simple mode projects the same row', async () => {
    ({ customize, sync } = await boot({ menu: true, uiMode: 'simple' }));

    expect(displayRow()).not.toBe(null);
    displayRow().click();
    expect(customize.isEditing()).toBe(true);
  });

  test('a deployment with no display menu is not an error', async () => {
    ({ customize, sync } = await boot());

    expect(displayRow()).toBe(null);
    expect(customize.isEditing()).toBe(false);
  });
});

describe('right-clicking a bar', () => {
  test('opens the customize menu in Expert mode', async () => {
    ({ customize, sync } = await boot({ fetch: endpoint({ get: doc(['logo'], []) }) }));

    rightClick(headerHost());

    expect(document.querySelector('.bar-context-menu')).not.toBe(null);
  });

  test('opens the same menu in Simple mode', async () => {
    ({ customize, sync } = await boot({
      fetch: endpoint({ get: doc(['logo'], []) }),
      uiMode: 'simple',
    }));

    rightClick(headerHost());

    expect(document.querySelector('.bar-context-menu')).not.toBe(null);
  });

  test('enters edit mode from the menu', async () => {
    ({ customize, sync } = await boot({ fetch: endpoint({ get: doc(['logo'], []) }) }));

    rightClick(headerHost());
    /** @type {any} */ (document.querySelector('[data-bar-action="customize"]')).click();

    expect(customize.isEditing()).toBe(true);
  });
});

describe('the palette action', () => {
  test('Expert mode offers it', async () => {
    ({ customize, sync } = await boot());

    await openPalette();

    expect(paletteLabels()).toContain('Customize bars');
  });

  test('Simple mode offers it too', async () => {
    ({ customize, sync } = await boot({ uiMode: 'simple' }));

    await openPalette();

    expect(paletteLabels()).toContain('Customize bars');
  });
});

describe('a teardown while the display menu is still upgrading', () => {
  /**
   * Stage a cold page, arm the entry points against it, and hand back the
   * resolver for the `whenDefined` promise the arm is now waiting on — the one
   * thing that outlives a teardown, because the browser settles it and there is
   * nothing to cancel.
   * @returns {Promise<{entry: any, upgrade: () => void}>}
   */
  async function armCold() {
    vi.resetModules();
    document.documentElement.setAttribute('data-ui-mode', 'expert');
    document.body.innerHTML = '<osprey-display-menu id="display-menu"></osprey-display-menu>';
    /** @type {() => void} */
    let upgrade = () => {};
    const upgraded = new Promise((resolve) => {
      upgrade = () => resolve(undefined);
    });
    vi.spyOn(window.customElements, 'whenDefined').mockReturnValue(upgraded);
    const entry = await import(ENTRY_PATH);
    entry.initEntryPoints(
      {
        isEditing: () => false,
        enterEditMode() {},
        exitEditMode() {},
        barVisible: () => true,
        setBarVisible: async () => true,
      },
      document
    );
    // What the component's own connectedCallback builds: the card, and the
    // action row the Customize row projects itself into.
    const menu = /** @type {any} */ (document.querySelector('osprey-display-menu'));
    menu.innerHTML = '<div class="display-menu-card"><div class="display-menu-actions"></div></div>';
    return { entry, upgrade };
  }

  test('the upgrade mounts the row when nothing has torn the page down', async () => {
    const { upgrade } = await armCold();

    upgrade();
    await settle();

    expect(document.querySelector('.display-menu-actions .bar-customize-entry')).not.toBe(null);
  });

  test('a teardown before the upgrade leaves no row behind', async () => {
    // Without the generation guard the row mounts here anyway, into a page the
    // module has already unmounted — and nothing takes it away again, because
    // the unmount it would have needed has already run.
    const { entry, upgrade } = await armCold();

    entry.stopEntryPoints();
    upgrade();
    await settle();

    expect(document.querySelector('.bar-customize-entry')).toBe(null);
  });
});

describe('flipping the ui mode', () => {
  test('leaves the entry points in place, both ways', async () => {
    ({ customize, sync } = await boot({ menu: true, fetch: endpoint({ get: doc(['logo'], []) }) }));

    await switchMode('simple');
    rightClick(headerHost());
    expect(displayRow()).not.toBe(null);
    expect(document.querySelector('.bar-context-menu')).not.toBe(null);
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));

    await switchMode('expert');
    rightClick(headerHost());
    expect(displayRow()).not.toBe(null);
    expect(document.querySelector('.bar-context-menu')).not.toBe(null);
  });

  test('a flip to Simple leaves an edit in progress alone', async () => {
    ({ customize, sync } = await boot({ menu: true }));
    customize.enterEditMode();

    await switchMode('simple');

    expect(customize.isEditing()).toBe(true);
    expect(document.body.classList.contains('bar-editing')).toBe(true);
  });

  test('the saved layout is the same in both modes, and the flip writes nothing', async () => {
    ({ customize, sync } = await boot({
      fetch: endpoint({ get: doc(['logo', 'clock'], ['stopwatch', 'docs']) }),
    }));
    const before = { header: rendered('header'), status: rendered('status') };

    await switchMode('simple');
    const simple = { header: rendered('header'), status: rendered('status') };
    await switchMode('expert');

    expect(simple).toEqual(before);
    expect({ header: rendered('header'), status: rendered('status') }).toEqual(before);
    expect(sync.currentLayout().header.map((/** @type {any} */ i) => i.type)).toEqual([
      'logo',
      'clock',
    ]);
    expect(putBodies()).toEqual([]);
  });
});
