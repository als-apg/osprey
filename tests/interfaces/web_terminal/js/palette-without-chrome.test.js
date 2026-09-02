// @ts-check
/**
 * The command palette does not depend on any bar item being present:
 *   npx vitest run tests/interfaces/web_terminal/js/palette-without-chrome.test.js
 *
 * Every header item can be removed, so the two palette rows that used to reach
 * THROUGH a header item — "Switch to … mode" clicked the display menu's View
 * segment, "Log out" clicked the identity menu's button — have to work on a
 * page that has neither. The mode pick goes through the same function the
 * display menu's own row calls; Log out reads the landing URL the server
 * stamps on `<html>` and runs logout.js's flow itself.
 */

import { test, expect, describe, afterEach, vi } from 'vitest';

import { boot, settle, teardown } from './bar-customize-fixture.mjs';

const PALETTE_BOOT_PATH =
  '../../../../src/osprey/interfaces/web_terminal/static/js/palette-boot.js';

/** @type {any} */
let customize;
/** @type {any} */
let sync;

/** Every action label the palette is currently showing. */
function paletteLabels() {
  return Array.from(document.querySelectorAll('.command-palette-item-label')).map(
    (node) => node.textContent ?? ''
  );
}

/** The palette row wearing `label`. @param {string} label */
function paletteRow(label) {
  const node = Array.from(document.querySelectorAll('.command-palette-item-label')).find(
    (candidate) => candidate.textContent === label
  );
  return /** @type {HTMLElement | null} */ (node?.closest('.command-palette-item') ?? null);
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
  delete document.documentElement.dataset.landingUrl;
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('the mode switch', () => {
  test('flips the mode with no display menu on the page', async () => {
    ({ customize, sync } = await boot());
    expect(document.querySelector('osprey-display-menu')).toBe(null);
    const post = vi.spyOn(window, 'postMessage').mockImplementation(() => {});

    await openPalette();
    paletteRow('Switch to Simple mode')?.click();

    expect(post).toHaveBeenCalledWith(
      { type: 'osprey-mode-change', mode: 'simple' },
      window.location.origin
    );
    expect(localStorage.getItem('osprey-ui-mode')).toBe('simple');
  });
});

describe('Log out', () => {
  test('is offered from the landing URL on <html>, with no logout button anywhere', async () => {
    ({ customize, sync } = await boot());
    document.documentElement.dataset.landingUrl = '/landing';
    expect(document.getElementById('logout-btn')).toBe(null);
    const fetchMock = vi.fn(async () => ({ ok: true, status: 200 }));
    vi.stubGlobal('fetch', fetchMock);
    const assign = vi.fn();
    vi.stubGlobal('location', { origin: 'http://localhost:5000', assign });

    await openPalette();
    expect(paletteLabels()).toContain('Log out');
    paletteRow('Log out')?.click();

    await vi.waitFor(() => expect(assign).toHaveBeenCalledWith('/landing'));
    expect(fetchMock).toHaveBeenCalledWith('/api/terminal/logout', { method: 'POST' });
  });

  test('is not offered where the server stamped no landing URL', async () => {
    ({ customize, sync } = await boot());

    await openPalette();

    expect(paletteLabels()).not.toContain('Log out');
  });
});
