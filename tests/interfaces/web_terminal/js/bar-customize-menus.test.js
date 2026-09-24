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
 *   - "Move left" and "Move right" are offered only where the item has a
 *     neighbour on that side and the layout is editable.
 *
 *   - the popover takes the focus when it opens and gives it back to the item
 *     when it closes, and a popover that re-opens after an accepted edit puts
 *     the focus back on the control that made it. After Move left or Move
 *     right that is the same button on the item's new place, so pressing it
 *     again moves the item further.
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

  test('Move left swaps the item with its left neighbour', async () => {
    await editing(doc(['logo', 'clock', 'search'], []));
    await openOptions('search');

    popover().querySelector('[data-bar-action="move-left"]').click();
    await settle();

    expect(putBodies()[0].header.map((/** @type {any} */ i) => i.type)).toEqual([
      'logo',
      'search',
      'clock',
    ]);
    expect(rendered('header')).toEqual(['logo', 'search', 'clock']);
  });

  test('Move right swaps the item with its right neighbour', async () => {
    await editing(doc(['logo', 'clock', 'search'], []));
    await openOptions('logo');

    popover().querySelector('[data-bar-action="move-right"]').click();
    await settle();

    expect(putBodies()[0].header.map((/** @type {any} */ i) => i.type)).toEqual([
      'clock',
      'logo',
      'search',
    ]);
    expect(rendered('header')).toEqual(['clock', 'logo', 'search']);
  });

  test('the ends of a bar are offered only the inward move', async () => {
    await editing(doc(['logo', 'clock', 'search'], []));

    await openOptions('logo');
    expect(popover().querySelector('[data-bar-action="move-right"]')).not.toBe(null);
    expect(popover().querySelector('[data-bar-action="move-left"]')).toBe(null);

    await openOptions('search');
    expect(popover().querySelector('[data-bar-action="move-left"]')).not.toBe(null);
    expect(popover().querySelector('[data-bar-action="move-right"]')).toBe(null);
  });

  test('an item alone in its bar is offered neither', async () => {
    await editing(doc(['logo'], ['clock']));

    await openOptions('clock');

    expect(popover().querySelector('[data-bar-action="move-left"]')).toBe(null);
    expect(popover().querySelector('[data-bar-action="move-right"]')).toBe(null);
  });

  test('a reorder leaves the other bar alone', async () => {
    await editing(doc(['logo', 'clock'], ['docs', 'search']));
    await openOptions('search');

    popover().querySelector('[data-bar-action="move-left"]').click();
    await settle();

    const body = putBodies()[0];
    expect(body.status.map((/** @type {any} */ i) => i.type)).toEqual(['search', 'docs']);
    expect(body.header.map((/** @type {any} */ i) => i.type)).toEqual(['logo', 'clock']);
  });

  test('a read-only layout is offered neither', async () => {
    ({ customize, sync } = await boot({
      fetch: endpoint({ get: doc(['logo', 'clock', 'not-a-type'], []) }),
    }));
    customize.enterEditMode();
    expect(sync.isLayoutReadonly()).toBe(true);

    await openOptions('clock');

    expect(popover().querySelector('[data-bar-action="move-left"]')).toBe(null);
    expect(popover().querySelector('[data-bar-action="move-right"]')).toBe(null);
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
    await notEditing(doc(['logo', 'clock'], ['stopwatch']));

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
    await notEditing(doc(['logo', 'clock'], ['stopwatch']));

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
    await notEditing(doc(['logo', 'clock'], ['stopwatch']));

    rightClick(shell('stopwatch'));
    expect(menuRow('customize')?.textContent).toBe('Customize bars…');
    expect(menuRow('status')?.textContent).toBe('Hide status bar');
    expect(menuRow('header')).toBe(null);
    expect(document.querySelectorAll('.bar-context-row')).toHaveLength(2);
    menuRow('status').click();
    await settle();

    expect(putBodies()).toHaveLength(1);
    expect(putBodies()[0].status_visible).toBe(false);
    rightClick(shell('stopwatch'));
    expect(menuRow('status')?.textContent).toBe('Show status bar');
  });

  test('entering edit mode brings the item rows', async () => {
    await notEditing(doc(['logo', 'clock'], ['stopwatch']));
    customize.enterEditMode();

    rightClick(shell('clock'));
    expect(menuRow('options')).not.toBe(null);
    expect(menuRow('remove')).not.toBe(null);
    expect(menuRow('status')).toBe(null);
    expect(menuRow('header')?.textContent).toBe('Hide header');
    expect(menuRow('customize')?.textContent).toBe('Done customizing');
    expect(menuRow('reset')).toBe(null);

    rightClick(shell('stopwatch'));
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
        reset: doc(['logo'], ['stopwatch']),
      }),
    }));
    customize.enterEditMode();

    defaultPill().click();
    await settle();

    expect(rendered('header')).toEqual(['logo']);
    expect(rendered('status')).toEqual(['stopwatch']);
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
    expect(tileFor('clock').getAttribute('aria-disabled')).toBe('true');
    expect(document.querySelector('.bar-sheet-notice')?.textContent).toBe(
      'Layout not editable. Default resets it.'
    );

    defaultPill().click();
    await settle();

    expect(sync.isLayoutReadonly()).toBe(false);
    expect(tileFor('clock').getAttribute('aria-disabled')).toBeNull();
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

/**
 * An item's own control: the button its body is built around. The stopwatch
 * and the feedback item each build one; the other types here build none.
 * @param {string} type
 */
function controlOf(type) {
  return /** @type {any} */ (shell(type).querySelector('.bar-item-btn'));
}

/** Open an item's options from its own focused control, as a keyboard does. */
async function openFromKeyboard(/** @type {string} */ type) {
  controlOf(type).focus();
  await press(controlOf(type));
  expect(popover().contains(document.activeElement)).toBe(true);
}

/** Press Escape the way a keyboard does. */
function escape() {
  document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
}

/** Activate a focused control: Enter on a button is a click. */
async function press(/** @type {any} */ node) {
  node.click();
  await settle();
}

describe('the popover takes the focus and gives it back', () => {
  test('opening it focuses its first control', async () => {
    await editing(doc(['logo', 'clock'], []));

    await openOptions('clock');

    expect(document.activeElement).toBe(popover().querySelector('button, input'));
  });

  test('a type with no options focuses the first button in its foot', async () => {
    await editing(doc(['logo', 'clock'], []));

    await openOptions('logo');

    expect(document.activeElement).toBe(popover().querySelector('.bar-pop-foot button'));
  });

  test('Escape gives the focus back to the item it was opened from', async () => {
    await editing(doc(['logo', 'stopwatch'], []));
    await openFromKeyboard('stopwatch');

    escape();

    expect(popover()).toBe(null);
    expect(document.activeElement).toBe(controlOf('stopwatch'));
    expect(customize.isEditing()).toBe(true);
  });

  test('Move to gives the focus to the item in the other bar', async () => {
    // The move rebuilds the body at the other bar's density, so the control
    // that had the focus is a new node; the item's control there takes it.
    await editing(doc(['logo', 'stopwatch'], []));
    await openFromKeyboard('stopwatch');

    await press(popover().querySelector('[data-bar-action="move"]'));

    expect(rendered('status')).toEqual(['stopwatch']);
    expect(document.activeElement).toBe(controlOf('stopwatch'));
  });

  test('Remove gives the focus to the item that took its place', async () => {
    await editing(doc(['logo', 'stopwatch', 'feedback'], []));
    await openFromKeyboard('stopwatch');

    await press(popover().querySelector('[data-bar-action="remove"]'));

    expect(rendered('header')).toEqual(['logo', 'feedback']);
    expect(document.activeElement).toBe(controlOf('feedback'));
  });

  test('setting an option keeps the focus on the control that set it', async () => {
    await editing(doc(['logo', 'clock'], []));
    await openOptions('clock');
    const utc = row('zone').querySelector('[data-bar-value="utc"]');
    utc.focus();

    await press(utc);

    expect(putBodies()[0].header[1].options.zone).toBe('utc');
    expect(popover()).not.toBe(null);
    expect(document.activeElement).toBe(row('zone').querySelector('[data-bar-value="utc"]'));
  });
});

describe('the popover follows an item it moved', () => {
  test('Move left re-opens it on the moved item with Move left focused', async () => {
    await editing(doc(['logo', 'clock', 'search'], []));
    await openOptions('search');

    await press(popover().querySelector('[data-bar-action="move-left"]'));

    expect(rendered('header')).toEqual(['logo', 'search', 'clock']);
    expect(shell('search').contains(popover())).toBe(true);
    expect(document.activeElement).toBe(popover().querySelector('[data-bar-action="move-left"]'));
  });

  test('pressing Move left again moves the item further', async () => {
    await editing(doc(['logo', 'clock', 'search'], []));
    await openOptions('search');

    await press(popover().querySelector('[data-bar-action="move-left"]'));
    await press(document.activeElement);

    expect(putBodies().map((body) => body.header.map((/** @type {any} */ i) => i.type))).toEqual([
      ['logo', 'search', 'clock'],
      ['search', 'logo', 'clock'],
    ]);
    expect(shell('search').contains(popover())).toBe(true);
    // At the left end there is no Move left to hold the focus; the first
    // control takes it.
    expect(popover().querySelector('[data-bar-action="move-left"]')).toBe(null);
    expect(document.activeElement).toBe(popover().querySelector('button, input'));
  });

  test('pressing Move right again moves the item further', async () => {
    await editing(doc(['logo', 'clock', 'search'], []));
    await openOptions('logo');

    await press(popover().querySelector('[data-bar-action="move-right"]'));
    await press(document.activeElement);

    expect(rendered('header')).toEqual(['clock', 'search', 'logo']);
    expect(shell('logo').contains(popover())).toBe(true);
  });

  test('an item that passes one of its own type is followed to its new key', async () => {
    // Keys count a repeated type in document order, so the second clock is
    // `clock#1` until it moves left of the first, and then it is `clock`.
    const utc = { type: 'clock', options: { zone: 'utc' } };
    await editing(doc(['logo', 'clock', utc], []));
    const second = /** @type {any} */ (document.querySelector('[data-bar-key="clock#1"]'));
    second.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();

    await press(popover().querySelector('[data-bar-action="move-left"]'));

    const first = document.querySelector('[data-bar-key="clock"]');
    expect(first?.contains(popover())).toBe(true);
    expect(row('zone').querySelector('[aria-pressed="true"]').dataset.barValue).toBe('utc');
  });

  test('closing the followed popover gives the focus back to the item', async () => {
    await editing(doc(['logo', 'clock', 'stopwatch'], []));
    await openFromKeyboard('stopwatch');

    await press(popover().querySelector('[data-bar-action="move-left"]'));
    escape();

    expect(rendered('header')).toEqual(['logo', 'stopwatch', 'clock']);
    expect(popover()).toBe(null);
    expect(document.activeElement).toBe(controlOf('stopwatch'));
  });
});

describe('the context menu stays on screen', () => {
  test('opened at the far corner it is clamped inside the window', async () => {
    await editing(doc(['logo', 'clock'], []));
    // happy-dom lays nothing out, so the menu is given a size here; the real
    // layout is the browser lane's to check.
    const width = Object.getOwnPropertyDescriptor(HTMLElement.prototype, 'offsetWidth');
    const height = Object.getOwnPropertyDescriptor(HTMLElement.prototype, 'offsetHeight');
    /** @param {number} size */
    const sized = (size) => ({
      configurable: true,
      get() {
        return /** @type {any} */ (this).classList?.contains('bar-context-menu') ? size : 0;
      },
    });
    Object.defineProperty(HTMLElement.prototype, 'offsetWidth', sized(200));
    Object.defineProperty(HTMLElement.prototype, 'offsetHeight', sized(120));
    try {
      rightClick(shell('clock'), window.innerWidth - 2, window.innerHeight - 2);
      const menu = /** @type {any} */ (document.querySelector('.bar-context-menu'));

      expect(menu.style.left).toBe(`${window.innerWidth - 200 - 8}px`);
      expect(menu.style.top).toBe(`${window.innerHeight - 120 - 8}px`);
    } finally {
      if (width) Object.defineProperty(HTMLElement.prototype, 'offsetWidth', width);
      if (height) Object.defineProperty(HTMLElement.prototype, 'offsetHeight', height);
    }
  });
});
