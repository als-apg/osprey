/**
 * Bar customize — the options popover, the context menu and the Default preset,
 * happy-dom environment:
 *   npx vitest run tests/interfaces/web_terminal/js/bar-customize-menus.test.js
 *
 * What these assertions are for:
 *
 *   - an option is CLAMPED to the catalog's own bounds before the PUT. The
 *     store answers 422 to an out-of-spec value and the client has nothing
 *     useful to say about that, so a value the catalog would refuse must never
 *     reach the network.
 *
 *   - the popover renders from the catalog's option spec, not from a table of
 *     its own: a type that gains an option gains a row, and one that has none
 *     says so.
 *
 *   - "Move to the other bar" is offered only where the move would be allowed,
 *     and "Remove" is offered on every item, the wordmark included.
 *
 *   - the one preset, Default, is the deployment's own arrangement: applying it
 *     DELETES the operator's document and renders what the server hands back,
 *     because only the server knows what `web.bar_items` configured.
 */

import { test, expect, describe, afterEach } from 'vitest';
import {
  boot,
  deleteCount,
  doc,
  endpoint,
  putBodies,
  rendered,
  settle,
  shell,
  teardown,
  tile as tileFor,
  withRect,
} from './bar-customize-fixture.mjs';

/** @type {any} */
let customize;
/** @type {any} */
let sync;

/**
 * Boot in edit mode.
 * @param {Record<string, unknown>} layout
 */
async function editing(layout) {
  ({ customize, sync } = await boot({ fetch: endpoint({ get: layout }) }));
  customize.enterEditMode();
  return customize;
}

/** The open options popover, if any. */
function popover() {
  return /** @type {any} */ (document.querySelector('.bar-options'));
}

/** Open one item's options by clicking it, the way an operator does. */
async function openOptions(/** @type {string} */ type) {
  shell(type).dispatchEvent(new MouseEvent('click', { bubbles: true }));
  await settle();
  return popover();
}

/** One option row's control area. @param {string} key */
function row(key) {
  return /** @type {any} */ (document.querySelector(`.bar-option[data-bar-option="${key}"]`));
}

/** A row in the context menu. @param {string} action */
function menuRow(action) {
  return /** @type {any} */ (
    document.querySelector(`.bar-context-menu [data-bar-action="${action}"]`)
  );
}

/** Right-click something. @param {any} node */
function rightClick(node, x = 10, y = 10) {
  node.dispatchEvent(
    new MouseEvent('contextmenu', { bubbles: true, cancelable: true, clientX: x, clientY: y })
  );
}

afterEach(() => {
  teardown({ customize, sync });
  customize = null;
  sync = null;
});

describe('the options popover', () => {
  test('clicking an item while editing opens its options', async () => {
    await editing(doc(['logo', 'clock'], []));

    const pop = await openOptions('clock');

    expect(pop).not.toBe(null);
    expect(pop.querySelector('.bar-pop-eyebrow')?.textContent).toBe('Clock');
  });

  test('an item click outside edit mode opens nothing', async () => {
    ({ customize, sync } = await boot({ fetch: endpoint({ get: doc(['logo', 'clock'], []) }) }));

    shell('clock').dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();

    expect(popover()).toBe(null);
  });

  test('it renders one row per catalog option', async () => {
    await editing(doc(['logo', 'clock'], []));

    await openOptions('clock');

    expect(row('zone')).not.toBe(null);
    expect(row('format')).not.toBe(null);
    expect(row('seconds')).not.toBe(null);
    expect(document.querySelectorAll('.bar-option').length).toBe(3);
  });

  test('a type with no options says so', async () => {
    await editing(doc(['logo', 'docs'], []));

    await openOptions('docs');

    expect(document.querySelectorAll('.bar-option').length).toBe(0);
    expect(popover().textContent).toContain('No options');
  });

  test('it names the density its host renders at', async () => {
    await editing(doc(['logo'], ['clock']));

    await openOptions('clock');

    expect(popover().querySelector('.bar-pop-density')?.textContent).toContain('Compact');
  });

  test('picking an enum value writes it', async () => {
    await editing(doc(['logo', 'clock'], []));
    await openOptions('clock');

    row('zone').querySelector('[data-bar-value="utc"]').click();
    await settle();

    expect(putBodies()[0].header[1].options.zone).toBe('utc');
  });

  test('toggling a boolean writes it', async () => {
    await editing(doc(['logo', 'clock'], []));
    await openOptions('clock');

    const box = row('seconds').querySelector('input');
    box.checked = true;
    box.dispatchEvent(new Event('change'));
    await settle();

    expect(putBodies()[0].header[1].options.seconds).toBe(true);
  });

  test('a number above the catalog bound is clamped before the PUT', async () => {
    await editing(doc([{ type: 'space', options: { width: 12 } }], []));
    await openOptions('space');

    const input = row('width').querySelector('input');
    input.value = '9999';
    input.dispatchEvent(new Event('change'));
    await settle();

    expect(putBodies()[0].header[0].options.width).toBe(2000);
  });

  test('a number below the catalog bound is clamped before the PUT', async () => {
    await editing(doc([{ type: 'space', options: { width: 12 } }], []));
    await openOptions('space');

    const input = row('width').querySelector('input');
    input.value = '-4';
    input.dispatchEvent(new Event('change'));
    await settle();

    expect(putBodies()[0].header[0].options.width).toBe(0);
  });

  test('a space says what width 0 means', async () => {
    await editing(doc([{ type: 'space', options: { width: 12 } }], []));
    const pop = await openOptions('space');

    expect(pop.textContent).toContain('0 fills the remaining room');
  });

  test('an unreadable number falls back to the option default', async () => {
    await editing(doc([{ type: 'space', options: { width: 40 } }], []));
    await openOptions('space');

    const input = row('width').querySelector('input');
    input.value = 'wide';
    input.dispatchEvent(new Event('change'));
    await settle();

    expect(putBodies()[0].header[0].options.width).toBe(0);
  });
});

describe('moving and removing from the popover', () => {
  test('Move sends the item to the other bar', async () => {
    await editing(doc(['logo', 'clock'], []));
    await openOptions('clock');

    popover().querySelector('[data-bar-action="move"]').click();
    await settle();

    expect(putBodies()[0].status.map((/** @type {any} */ i) => i.type)).toEqual(['clock']);
    expect(rendered('status')).toEqual(['clock']);
  });

  test('every type is offered a move to the other bar', async () => {
    await editing(doc(['logo', 'search'], []));

    await openOptions('search');

    expect(popover().querySelector('[data-bar-action="move"]')).not.toBe(null);
  });

  test('Remove takes the item out', async () => {
    await editing(doc(['logo', 'clock'], []));
    await openOptions('clock');

    popover().querySelector('[data-bar-action="remove"]').click();
    await settle();

    expect(putBodies()[0].header.map((/** @type {any} */ i) => i.type)).toEqual(['logo']);
  });

  test('the wordmark is offered a remove like every other item', async () => {
    await editing(doc(['logo', 'clock'], []));

    await openOptions('logo');

    expect(popover().querySelector('[data-bar-action="remove"]')).not.toBe(null);
  });
});

describe('the popover keeps out of its own way', () => {
  test('an item in the left half opens the popover leftward-anchored', async () => {
    await editing(doc(['logo', 'clock'], []));
    withRect(shell('clock'), { left: 20, right: 60 });

    await openOptions('clock');

    expect(popover().classList.contains('is-left')).toBe(true);
  });

  test('an item in the right half opens the popover right-anchored', async () => {
    await editing(doc(['logo', 'clock'], []));
    withRect(shell('clock'), { left: 900, right: 960 });

    await openOptions('clock');

    expect(popover().classList.contains('is-left')).toBe(false);
  });

  test('opening one popover closes the last', async () => {
    await editing(doc(['logo', 'clock', 'docs'], []));
    await openOptions('clock');

    await openOptions('docs');

    expect(document.querySelectorAll('.bar-options').length).toBe(1);
  });

  test('a click elsewhere closes it', async () => {
    await editing(doc(['logo', 'clock'], []));
    await openOptions('clock');

    document.body.dispatchEvent(new MouseEvent('click', { bubbles: true }));

    expect(popover()).toBe(null);
  });

  test('the host closes it before the item moves', async () => {
    await editing(doc(['logo', 'clock'], []));
    await openOptions('clock');
    const host = await import(
      '../../../../src/osprey/interfaces/web_terminal/static/js/bar-host.js'
    );

    host.closeBarPopovers();

    expect(popover()).toBe(null);
  });
});

describe('the context menu', () => {
  test('right-clicking an item offers its own rows', async () => {
    await editing(doc(['logo', 'clock'], []));

    rightClick(shell('clock'));

    expect(menuRow('options')).not.toBe(null);
    expect(menuRow('remove')).not.toBe(null);
    expect(menuRow('customize')).not.toBe(null);
  });

  test('the wordmark is offered a remove', async () => {
    await editing(doc(['logo'], []));

    rightClick(shell('logo'));

    expect(menuRow('remove')).not.toBe(null);
  });

  test('right-clicking the bar itself offers only the bar rows', async () => {
    await editing(doc(['logo'], []));

    rightClick(document.querySelector('[data-bar-host="header"]'));
    expect(menuRow('options')).toBe(null);
    expect(menuRow('customize')).not.toBe(null);
    expect(menuRow('status')).toBe(null);

    rightClick(document.querySelector('[data-bar-host="status"]'));
    expect(menuRow('options')).toBe(null);
    expect(menuRow('customize')).not.toBe(null);
    expect(menuRow('status')).not.toBe(null);
  });

  test('Remove from the menu takes the item out', async () => {
    await editing(doc(['logo', 'clock'], []));

    rightClick(shell('clock'));
    menuRow('remove').click();
    await settle();

    expect(putBodies()[0].header.map((/** @type {any} */ i) => i.type)).toEqual(['logo']);
  });

  test('the status row hides the bar', async () => {
    await editing(doc(['logo'], ['clock']));

    rightClick(document.querySelector('[data-bar-host="status"]'));
    menuRow('status').click();
    await settle();

    expect(putBodies()[0].status_visible).toBe(false);
  });

  test('options from the menu opens the popover', async () => {
    await editing(doc(['logo', 'clock'], []));

    rightClick(shell('clock'));
    menuRow('options').click();
    await settle();

    expect(popover()).not.toBe(null);
  });

  test('Escape closes the menu', async () => {
    await editing(doc(['logo', 'clock'], []));
    rightClick(shell('clock'));

    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));

    expect(document.querySelector('.bar-context-menu')).toBe(null);
  });
});

describe('outside edit mode the menu is the way in, plus Hide on the status bar', () => {
  /** Boot in Expert without entering edit mode. @param {Record<string, unknown>} layout */
  async function notEditing(layout) {
    ({ customize, sync } = await boot({ fetch: endpoint({ get: layout }) }));
    return customize;
  }

  test('the header offers Customize and Hide header, and nothing else', async () => {
    // An item's rows change the layout, and a refused change is answered in
    // the sheet — which does not exist until edit mode has been entered. The
    // Hide row names the bar under the pointer: offered from the top bar,
    // "Hide status bar" read as "hide this one".
    await notEditing(doc(['logo', 'clock'], ['activity']));

    rightClick(shell('clock'));

    expect(menuRow('customize')?.textContent).toBe('Customize bars…');
    expect(menuRow('header')?.textContent).toBe('Hide header');
    expect(document.querySelectorAll('.bar-context-row')).toHaveLength(2);
    expect(menuRow('status')).toBe(null);
    expect(menuRow('options')).toBe(null);
    expect(menuRow('remove')).toBe(null);
    expect(menuRow('reset')).toBe(null);
  });

  test('Hide header withdraws the top bar and reads Show once hidden', async () => {
    await notEditing(doc(['logo', 'clock'], ['activity']));

    rightClick(shell('clock'));
    menuRow('header').click();
    await settle();

    expect(putBodies()).toHaveLength(1);
    expect(putBodies()[0].header_visible).toBe(false);
    expect(putBodies()[0].status_visible).toBe(true);
    rightClick(shell('clock'));
    expect(menuRow('header')?.textContent).toBe('Show header');
  });

  test('the status bar offers Hide status bar, which reads Show once hidden', async () => {
    await notEditing(doc(['logo', 'clock'], ['activity']));

    rightClick(shell('activity'));
    expect(menuRow('customize')?.textContent).toBe('Customize bars…');
    expect(menuRow('status')?.textContent).toBe('Hide status bar');
    expect(menuRow('header')).toBe(null);
    expect(document.querySelectorAll('.bar-context-row')).toHaveLength(2);
    menuRow('status').click();
    await settle();

    expect(putBodies()).toHaveLength(1);
    expect(putBodies()[0].status_visible).toBe(false);
    rightClick(shell('activity'));
    expect(menuRow('status')?.textContent).toBe('Show status bar');
  });

  test('entering edit mode brings the item rows', async () => {
    await notEditing(doc(['logo', 'clock'], ['activity']));
    customize.enterEditMode();

    rightClick(shell('clock'));
    expect(menuRow('options')).not.toBe(null);
    expect(menuRow('remove')).not.toBe(null);
    expect(menuRow('status')).toBe(null);
    expect(menuRow('header')?.textContent).toBe('Hide header');
    expect(menuRow('customize')?.textContent).toBe('Done customizing');
    expect(menuRow('reset')).toBe(null);

    rightClick(shell('activity'));
    expect(menuRow('options')).not.toBe(null);
    expect(menuRow('status')?.textContent).toBe('Hide status bar');
  });

  test('Escape closes it there too, and nothing else owns the key', async () => {
    await notEditing(doc(['logo'], []));
    rightClick(shell('logo'));

    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));

    expect(document.querySelector('.bar-context-menu')).toBe(null);
    expect(customize.isEditing()).toBe(false);
  });
});

describe('the menu is operable from the keyboard', () => {
  test('its rows are menu items and the first one takes the focus', async () => {
    await editing(doc(['logo', 'clock'], []));

    rightClick(shell('clock'));

    const rows = document.querySelectorAll('.bar-context-menu [role="menuitem"]');
    expect(rows.length).toBeGreaterThan(1);
    expect(document.activeElement).toBe(rows[0]);
  });

  test('the arrow keys walk the rows and wrap', async () => {
    await editing(doc(['logo', 'clock'], []));
    rightClick(shell('clock'));
    const rows = document.querySelectorAll('.bar-context-menu [role="menuitem"]');

    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowDown', bubbles: true }));
    expect(document.activeElement).toBe(rows[1]);

    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowUp', bubbles: true }));
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowUp', bubbles: true }));
    expect(document.activeElement).toBe(rows[rows.length - 1]);
  });

  test('closing gives the focus back where it came from', async () => {
    await editing(doc(['logo', 'clock'], []));
    const done = /** @type {any} */ (document.querySelector('.bar-sheet-done'));
    done.focus();

    rightClick(shell('clock'));
    expect(document.activeElement).not.toBe(done);
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));

    expect(document.activeElement).toBe(done);
  });
});

describe('Escape has one owner at a time', () => {
  test('it closes an open popover first and leaves edit mode second', async () => {
    // The order used to depend on which module registered its keydown first,
    // which anything re-arming the menus mid-edit could invert. Now edit mode
    // owns the key while it is on and asks the menus module what is open.
    await editing(doc(['logo', 'clock'], []));
    await openOptions('clock');

    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    expect(popover()).toBe(null);
    expect(customize.isEditing()).toBe(true);

    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    expect(customize.isEditing()).toBe(false);
  });
});

/** The sheet's one preset pill. */
function defaultPill() {
  return /** @type {any} */ (document.querySelector('.bar-sheet-presets [data-bar-preset]'));
}

describe('the Default preset', () => {
  test('it is the only pill, and it deletes rather than writes', async () => {
    await editing(doc(['logo', 'clock'], []));

    expect(document.querySelectorAll('.bar-sheet-presets .bar-pill')).toHaveLength(1);
    expect(defaultPill().textContent).toBe('Default');
    defaultPill().click();
    await settle();

    expect(deleteCount()).toBe(1);
    expect(putBodies()).toEqual([]);
  });

  test('what the server hands back is what the bars become', async () => {
    // The deployment default is not a document the client holds — only the
    // server knows what `web.bar_items` configured — so the reset renders the
    // answer rather than a preset of its own.
    ({ customize, sync } = await boot({
      fetch: endpoint({
        get: doc(['logo', 'clock', 'docs'], []),
        reset: doc(['logo'], ['activity']),
      }),
    }));
    customize.enterEditMode();

    defaultPill().click();
    await settle();

    expect(rendered('header')).toEqual(['logo']);
    expect(rendered('status')).toEqual(['activity']);
  });

  test('it is the one edit a read-only layout still allows', async () => {
    // A document naming a type this build cannot render is read-only: every
    // tile is refused and no PUT is issued. Reset is the way out, so it has to
    // work from exactly here — and afterwards the tiles are live again.
    ({ customize, sync } = await boot({
      fetch: endpoint({
        get: doc(['logo', 'not-a-type'], []),
        reset: doc(['logo'], []),
      }),
    }));
    customize.enterEditMode();
    expect(sync.isLayoutReadonly()).toBe(true);
    expect(tileFor('clock').disabled).toBe(true);
    expect(document.querySelector('.bar-sheet-notice')?.textContent).toBe(
      'Layout not editable. Default resets it.'
    );

    defaultPill().click();
    await settle();

    expect(sync.isLayoutReadonly()).toBe(false);
    expect(tileFor('clock').disabled).toBe(false);
  });

  test('a refused reset leaves the arrangement alone and says so', async () => {
    ({ customize, sync } = await boot({
      fetch: endpoint({
        get: doc(['logo', 'clock'], []),
        reset: new Error('offline'),
      }),
    }));
    customize.enterEditMode();

    defaultPill().click();
    await settle();

    expect(rendered('header')).toEqual(['logo', 'clock']);
    expect(document.querySelector('.bar-sheet-notice')?.textContent).toBe('Layout not reset');
  });
});
