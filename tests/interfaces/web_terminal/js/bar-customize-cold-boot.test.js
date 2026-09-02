/**
 * Bar customize — the display-menu row on a COLD page, happy-dom:
 *   npx vitest run tests/interfaces/web_terminal/js/bar-customize-cold-boot.test.js
 *
 * Its own file because it is the only suite that lets the real
 * `osprey-display-menu` component define itself, and a custom-element registry
 * is per-page, not per-test: once defined, every later `<osprey-display-menu>`
 * in the same file would upgrade on insertion and the cold case could not be
 * staged again.
 *
 * WHAT IT REPRODUCES. ES modules evaluate depth-first in import order, so which
 * of two modules runs first is decided by whoever imports them — and 3.4's own
 * palette action made `palette-boot.js` an early importer of the bar stack,
 * pulling `bar-customize.js`'s boot ahead of the component's
 * `customElements.define()`. The projection point the Customize row mounts into
 * is built by that component's `connectedCallback`, so on a real page the row's
 * first mount attempt finds nothing, and nothing stamps `data-ui-mode` again to
 * trigger a retry. The row was silently missing in production while the suites
 * asserted it was there, because their fixture wrote the action row as static
 * markup.
 *
 * So this boots the bar stack against the DOM THE SERVER RENDERS — the bare tag
 * with its projected children, no card — and only then evaluates the component.
 */

import { test, expect, afterEach } from 'vitest';
import { boot, settle, teardown } from './bar-customize-fixture.mjs';

/** @type {any} */
let customize;
/** @type {any} */
let sync;

const PALETTE_BOOT_PATH =
  '../../../../src/osprey/interfaces/web_terminal/static/js/palette-boot.js';
const DISPLAY_MENU_PATH = '/design-system/js/components/osprey-display-menu.js';

afterEach(() => {
  teardown({ customize, sync });
  customize = null;
  sync = null;
});

// ONE test on purpose: the custom-element registry belongs to the page, not to
// the test, so once this file has let the component define itself no later case
// here can stage a cold page again.
test('the Customize row mounts even when the bar stack boots first', async () => {
  // Exactly the production order: palette-boot pulls bar-customize (and with it
  // the entry points) while `<osprey-display-menu>` is still an unupgraded tag.
  ({ customize, sync } = await boot({ menu: 'cold' }));
  await import(PALETTE_BOOT_PATH);

  expect(document.querySelector('.display-menu-actions')).toBe(null);
  expect(document.querySelector('.bar-customize-entry')).toBe(null);

  // The component defines itself, which upgrades the tag and builds the card
  // and its action row.
  await import(DISPLAY_MENU_PATH);
  await settle();

  const row = /** @type {any} */ (
    document.querySelector('.display-menu-actions .bar-customize-entry')
  );
  expect(row).not.toBe(null);
  expect(row.textContent).toBe('Customize bars');

  // And it is the real entry point, not just an element with the right class.
  row.click();
  expect(customize.isEditing()).toBe(true);
});
