// @ts-check
/**
 * The command palette does not depend on any page chrome being present:
 *   npx vitest run tests/interfaces/web_terminal/js/palette-without-chrome.test.js
 *
 * Log out works on a page with no logout button: it reads the landing URL the server stamps on `<html>`, asks with the
 * sign-out dialog (sign-out.js) and runs logout.js's flow on confirm.
 */

import { test, expect, describe, afterEach, vi } from 'vitest';

const PALETTE_BOOT_PATH =
  '../../../../src/osprey/interfaces/web_terminal/static/js/palette-boot.js';

/** Let the palette's open and everything it queued settle. */
async function settle() {
  for (let turn = 0; turn < 8; turn += 1) await Promise.resolve();
  await new Promise((resolve) => setTimeout(resolve, 0));
}

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
 * dependency builder — the point of the test is what THAT assembles. The page
 * holds nothing but the trigger, and the module graph is fresh.
 */
async function openPalette() {
  vi.resetModules();
  const trigger = document.createElement('button');
  trigger.id = 'command-palette-btn';
  document.body.append(trigger);
  const paletteBoot = await import(PALETTE_BOOT_PATH);
  paletteBoot.initCommandPalette();
  trigger.click();
  await settle();
}

afterEach(() => {
  document.body.innerHTML = '';
  delete document.documentElement.dataset.landingUrl;
  delete document.documentElement.dataset.agentSurface;
  delete document.documentElement.dataset.terminal;
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('the terminal', () => {
  // The way to it is the Agent tile's ⋯ menu and nothing else.
  test.each([[undefined], ['terminal']])('has no palette entry (surface %s)', async (surface) => {
    if (surface) document.documentElement.dataset.agentSurface = surface;
    await openPalette();
    const labels = paletteLabels().map((label) => label.toLowerCase());
    expect(labels.filter((label) => /terminal|chat|new session/.test(label))).toEqual([]);
  });
});

describe('Log out', () => {
  test('is offered from the landing URL on <html>, with no logout button anywhere', async () => {
    document.documentElement.dataset.landingUrl = '/landing';
    expect(document.getElementById('logout-btn')).toBe(null);
    const fetchMock = vi.fn(async () => ({ ok: true, status: 200 }));
    vi.stubGlobal('fetch', fetchMock);
    const assign = vi.fn();
    vi.stubGlobal('location', { origin: 'http://localhost:5000', assign });

    await openPalette();
    expect(paletteLabels()).toContain('Log out');
    paletteRow('Log out')?.click();

    // It asks first: nothing is sent until the dialog is confirmed.
    const dialog = document.querySelector('.posture-modal-overlay:not([data-closing])');
    if (!dialog) throw new Error('the sign-out dialog did not open');
    expect(dialog.querySelector('.posture-modal-title')?.textContent).toBe('Sign out of OSPREY?');
    expect(fetchMock).not.toHaveBeenCalledWith('/api/terminal/logout', expect.anything());

    const confirm = dialog.querySelector('.posture-modal-confirm');
    if (!(confirm instanceof HTMLButtonElement)) throw new Error('the dialog has no confirm');
    confirm.click();

    await vi.waitFor(() => expect(assign).toHaveBeenCalledWith('/landing'));
    expect(fetchMock).toHaveBeenCalledWith('/api/terminal/logout', { method: 'POST' });
  });

  test('Cancel in the dialog signs nobody out', async () => {
    document.documentElement.dataset.landingUrl = '/landing';
    const fetchMock = vi.fn(async () => ({ ok: true, status: 200 }));
    vi.stubGlobal('fetch', fetchMock);

    await openPalette();
    paletteRow('Log out')?.click();
    const cancel = document.querySelector('.posture-modal-cancel');
    if (!(cancel instanceof HTMLButtonElement)) throw new Error('the sign-out dialog did not open');
    cancel.click();

    await settle();
    expect(fetchMock).not.toHaveBeenCalledWith('/api/terminal/logout', expect.anything());
  });

  test('is not offered where the server stamped no landing URL', async () => {
    await openPalette();

    expect(paletteLabels()).not.toContain('Log out');
  });
});

describe('Open activity log', () => {
  test('is offered and opens the page in a new tab', async () => {
    const open = vi.fn();
    vi.stubGlobal('open', open);

    await openPalette();
    expect(paletteLabels()).toContain('Open activity log');
    paletteRow('Open activity log')?.click();

    expect(open).toHaveBeenCalledTimes(1);
    const [url, target, features] = open.mock.calls[0];
    expect(String(url).startsWith('/static/session.html')).toBe(true);
    expect(target).toBe('_blank');
    expect(features).toBe('noopener');
  });
});
