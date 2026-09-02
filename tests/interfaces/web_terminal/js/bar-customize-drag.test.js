/**
 * Bar customize — the pointer drag, happy-dom environment:
 *   npx vitest run tests/interfaces/web_terminal/js/bar-customize-drag.test.js
 *
 * What these assertions are for:
 *
 *   - the gesture is POINTER events with `setPointerCapture`, not HTML5 drag
 *     and drop. Safari drops `dragleave.relatedTarget` and is particular about
 *     when `draggable` may be set; the pointer path behaves the same in every
 *     browser the hub supports.
 *
 *   - losing the capture mid-drag must not strand edit mode. The listeners
 *     live on the DOCUMENT, so a pointerup that never reaches the shell — the
 *     shell was re-parented by a reconcile, the capture was released by the
 *     browser — still ends the gesture and leaves nothing behind.
 *
 *   - a drop the layout cannot accept is REFUSED AND NAMED, with the same
 *     sentence the sheet's tile carries, and issues no PUT.
 *
 *   - a locked item may be moved and never removed: dropping one outside the
 *     bars is a no-op that says so.
 *
 * happy-dom lays nothing out, so a test that depends on WHERE something sits
 * gives the nodes boxes with `withRect()`. The pixels themselves are the
 * browser lane's to check.
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

/** Where the two bars sit for these tests. */
const HEADER_BOX = { left: 0, right: 300, top: 0, bottom: 40 };
const STATUS_BOX = { left: 0, right: 300, top: 100, bottom: 126 };

/** A point inside the header. */
const IN_HEADER = { x: 150, y: 20 };
/** A point inside the status bar. */
const IN_STATUS = { x: 150, y: 112 };
/** A point in neither bar. */
const OUTSIDE = { x: 150, y: 400 };

/**
 * Boot, give both hosts a box, and enter edit mode.
 * @param {Record<string, unknown>} layout
 */
async function editing(layout) {
  ({ customize, sync } = await boot({ fetch: endpoint({ get: layout }) }));
  withRect(document.querySelector('[data-bar-host="header"]'), HEADER_BOX);
  withRect(document.querySelector('[data-bar-host="status"]'), STATUS_BOX);
  customize.enterEditMode();
  return customize;
}

/**
 * One whole gesture: press on `node`, move to `to`, release there.
 * @param {any} node
 * @param {{x: number, y: number}} to
 * @param {{from?: {x: number, y: number}, releaseOn?: any}} [how]
 */
async function drag(node, to, { from = IN_HEADER, releaseOn = node } = {}) {
  pointer('pointerdown', node, from);
  pointer('pointermove', node, to);
  pointer('pointerup', releaseOn, to);
  await settle();
}

afterEach(() => {
  teardown({ customize, sync });
  customize = null;
  sync = null;
});

describe('moving an item between the bars', () => {
  test('a header item dropped on the status bar moves there', async () => {
    await editing(doc(['logo', 'clock', 'docs'], ['activity']));

    await drag(shell('clock'), IN_STATUS);

    const bodies = putBodies();
    expect(bodies).toHaveLength(1);
    expect(bodies[0].header.map((/** @type {any} */ i) => i.type)).toEqual(['logo', 'docs']);
    expect(bodies[0].status.map((/** @type {any} */ i) => i.type)).toEqual(['activity', 'clock']);
    expect(rendered('status')).toEqual(['activity', 'clock']);
  });

  test('the item lands where the pointer released it', async () => {
    await editing(doc(['logo', 'clock'], ['activity', 'docs']));
    withRect(shell('activity'), { left: 40, right: 80, top: 100, bottom: 126 });
    withRect(shell('docs'), { left: 80, right: 120, top: 100, bottom: 126 });

    await drag(shell('clock'), { x: 50, y: 112 });

    expect(putBodies()[0].status.map((/** @type {any} */ i) => i.type)).toEqual([
      'clock',
      'activity',
      'docs',
    ]);
  });

  test('reordering inside one bar is a permutation, not an add', async () => {
    await editing(doc(['logo', 'clock', 'docs'], []));
    withRect(shell('logo'), { left: 0, right: 40 });
    withRect(shell('clock'), { left: 40, right: 80 });
    withRect(shell('docs'), { left: 80, right: 120 });

    await drag(shell('docs'), { x: 45, y: 20 }, { from: { x: 100, y: 20 } });

    expect(putBodies()[0].header.map((/** @type {any} */ i) => i.type)).toEqual([
      'logo',
      'docs',
      'clock',
    ]);
  });

  test('an item moved into a hidden status bar brings it back', async () => {
    await editing(doc(['logo', 'clock'], [], { statusVisible: false }));

    await drag(shell('clock'), IN_STATUS);

    expect(putBodies()[0].status_visible).toBe(true);
  });
});

describe('the drop index is the layout position, not the DOM position', () => {
  test('a folded item keeps its place under a drop', async () => {
    // The overflow ladder folds by parking the shell in the pool, so a placed
    // item can render no shell in the host. Counting DOM children would insert
    // before every folded item — a routine drag on a crowded bar writing an
    // arrangement nobody made.
    await editing(doc(['logo', 'clock', 'docs', 'stopwatch'], []));
    const host = await import(
      '../../../../src/osprey/interfaces/web_terminal/static/js/bar-host.js'
    );
    host.parkShell(shell('docs'), document);
    withRect(shell('logo'), { left: 0, right: 40 });
    withRect(shell('clock'), { left: 40, right: 80 });
    withRect(shell('stopwatch'), { left: 80, right: 120 });

    await drag(tile('connection'), { x: 85, y: 20 }, { from: { x: 20, y: 200 } });

    expect(putBodies()[0].header.map((/** @type {any} */ i) => i.type)).toEqual([
      'logo',
      'clock',
      'docs',
      'connection',
      'stopwatch',
    ]);
  });

  test('a drop past everything visible lands after it', async () => {
    await editing(doc(['logo', 'clock', 'docs'], []));
    const host = await import(
      '../../../../src/osprey/interfaces/web_terminal/static/js/bar-host.js'
    );
    host.parkShell(shell('docs'), document);
    withRect(shell('logo'), { left: 0, right: 40 });
    withRect(shell('clock'), { left: 40, right: 80 });

    await drag(tile('connection'), { x: 200, y: 20 }, { from: { x: 20, y: 200 } });

    expect(putBodies()[0].header.map((/** @type {any} */ i) => i.type)).toEqual([
      'logo',
      'clock',
      'connection',
      'docs',
    ]);
  });
});

describe('a refused drop is named and writes nothing', () => {
  test('a header-only type is refused by the status bar', async () => {
    await editing(doc(['logo', 'search'], []));

    await drag(shell('search'), IN_STATUS);

    expect(putBodies()).toEqual([]);
    expect(noticeText()).toBe('Header only');
    expect(rendered('header')).toEqual(['logo', 'search']);
  });

  test('a full host refuses the drop by name', async () => {
    const full = Array.from({ length: 20 }, () => 'separator');
    await editing(doc(['logo', 'clock'], full));

    await drag(shell('clock'), IN_STATUS);

    expect(putBodies()).toEqual([]);
    expect(noticeText()).toBe('Status bar is full');
  });
});

describe('dropping outside the bars', () => {
  test('removes the item', async () => {
    await editing(doc(['logo', 'clock'], []));

    await drag(shell('clock'), OUTSIDE);

    expect(putBodies()[0].header.map((/** @type {any} */ i) => i.type)).toEqual(['logo']);
    expect(rendered('header')).toEqual(['logo']);
  });

  test('leaves a locked item where it is and says why', async () => {
    await editing(doc(['logo', 'clock'], []));

    await drag(shell('logo'), OUTSIDE);

    expect(putBodies()).toEqual([]);
    // ONE sentence for this rule, wherever it is refused.
    expect(noticeText()).toBe('Locked by the deployment');
    expect(rendered('header')).toEqual(['logo', 'clock']);
  });

  test('a tile dragged out of the sheet and dropped nowhere adds nothing', async () => {
    await editing(doc(['logo'], []));

    await drag(tile('stopwatch'), OUTSIDE, { from: { x: 20, y: 200 } });

    expect(putBodies()).toEqual([]);
  });
});

describe('dragging in from the sheet', () => {
  test('a tile dropped on a bar adds that item', async () => {
    await editing(doc(['logo'], []));

    await drag(tile('stopwatch'), IN_HEADER, { from: { x: 20, y: 200 } });

    const bodies = putBodies();
    expect(bodies).toHaveLength(1);
    expect(bodies[0].header.map((/** @type {any} */ i) => i.type)).toEqual(['logo', 'stopwatch']);
  });

  test('the click a finished drag leaves behind does not add a second copy', async () => {
    // The drag calls preventDefault() on pointerdown, which suppresses the
    // compatibility mouse events but NOT the click the browser dispatches at
    // the capture target on release. Without a guard on the tile's own
    // listener, one gesture adds the item twice — once where it was dropped and
    // once at the type's default host — and the second save conflicts with the
    // first and wins on the retry.
    await editing(doc(['logo'], []));
    const source = tile('stopwatch');

    pointer('pointerdown', source, { x: 20, y: 200 });
    pointer('pointermove', source, IN_HEADER);
    pointer('pointerup', source, IN_HEADER);
    // Same task as the release, which is when the browser dispatches it.
    source.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();

    const bodies = putBodies();
    expect(bodies).toHaveLength(1);
    expect(bodies[0].header.map((/** @type {any} */ i) => i.type)).toEqual(['logo', 'stopwatch']);
  });

  test('a plain click on a tile still adds the item', async () => {
    await editing(doc(['logo'], []));

    tile('stopwatch').dispatchEvent(new MouseEvent('click', { bubbles: true }));
    await settle();

    expect(putBodies()).toHaveLength(1);
  });

  test('a disabled tile does not start a drag', async () => {
    await editing(doc(['logo'], []));

    await drag(tile('bluesky-queue'), IN_HEADER, { from: { x: 20, y: 200 } });

    expect(putBodies()).toEqual([]);
  });
});

describe('the gesture itself', () => {
  test('a press that never moves is not a drag', async () => {
    await editing(doc(['logo', 'clock'], []));

    pointer('pointerdown', shell('clock'), IN_HEADER);
    pointer('pointermove', shell('clock'), { x: 152, y: 21 });
    pointer('pointerup', shell('clock'), { x: 152, y: 21 });
    await settle();

    expect(putBodies()).toEqual([]);
    expect(document.body.classList.contains('bar-dragging')).toBe(false);
  });

  test('the item takes the pointer capture', async () => {
    await editing(doc(['logo', 'clock'], []));
    const node = shell('clock');
    /** @type {number[]} */
    const captured = [];
    node.setPointerCapture = (/** @type {number} */ id) => captured.push(id);

    pointer('pointerdown', node, { ...IN_HEADER, pointerId: 9 });
    pointer('pointerup', node, IN_HEADER);
    await settle();

    expect(captured).toEqual([9]);
  });

  test('a pointerup that never reaches the item still ends the drag', async () => {
    // The capture is gone — released by the browser, or the shell was moved by
    // a reconcile. The document-level failsafe is what finishes the gesture.
    await editing(doc(['logo', 'clock'], []));

    pointer('pointerdown', shell('clock'), IN_HEADER);
    pointer('pointermove', document.documentElement, IN_STATUS);
    pointer('pointerup', document.documentElement, IN_STATUS);
    await settle();

    expect(putBodies()).toHaveLength(1);
    expect(document.body.classList.contains('bar-dragging')).toBe(false);
  });

  test('a second pointer neither aims nor commits the drag', async () => {
    // The listeners are on the document, so another finger reaches them too.
    // Releasing it over neither bar would otherwise REMOVE the dragged item.
    await editing(doc(['logo', 'clock'], []));

    pointer('pointerdown', shell('clock'), { ...IN_HEADER, pointerId: 1 });
    pointer('pointermove', shell('clock'), { ...IN_STATUS, pointerId: 1 });
    pointer('pointermove', document.documentElement, { ...OUTSIDE, pointerId: 2 });
    pointer('pointerup', document.documentElement, { ...OUTSIDE, pointerId: 2 });
    await settle();

    expect(putBodies()).toEqual([]);
    expect(rendered('header')).toEqual(['logo', 'clock']);

    // The gesture is still live, and the pointer that started it still ends it.
    pointer('pointerup', shell('clock'), { ...IN_STATUS, pointerId: 1 });
    await settle();

    expect(putBodies()).toHaveLength(1);
    expect(putBodies()[0].status.map((/** @type {any} */ i) => i.type)).toEqual(['clock']);
  });

  test('a cancelled drag changes nothing and leaves nothing behind', async () => {
    await editing(doc(['logo', 'clock'], []));

    pointer('pointerdown', shell('clock'), IN_HEADER);
    pointer('pointermove', shell('clock'), IN_STATUS);
    pointer('pointercancel', shell('clock'), IN_STATUS);
    await settle();

    expect(putBodies()).toEqual([]);
    expect(document.body.classList.contains('bar-dragging')).toBe(false);
    expect(document.querySelector('.bar-drag-ghost')).toBe(null);
    expect(document.querySelector('.bar-drop-marker')).toBe(null);
  });

  test('a second drag works after the first one ended', async () => {
    await editing(doc(['logo', 'clock', 'docs'], []));

    pointer('pointerdown', shell('clock'), IN_HEADER);
    pointer('pointercancel', shell('clock'), IN_STATUS);
    await drag(shell('docs'), IN_STATUS);

    expect(putBodies()).toHaveLength(1);
    expect(putBodies()[0].status.map((/** @type {any} */ i) => i.type)).toEqual(['docs']);
  });

  test('an open popover is closed before its item moves', async () => {
    await editing(doc(['logo', 'clock'], []));
    const host = await import(
      '../../../../src/osprey/interfaces/web_terminal/static/js/bar-host.js'
    );
    /** @type {string[]} */
    const closed = [];
    host.registerBarPopover(shell('clock'), () => closed.push('clock'));

    pointer('pointerdown', shell('clock'), IN_HEADER);
    pointer('pointermove', shell('clock'), IN_STATUS);

    expect(closed).toEqual(['clock']);
  });

  test('drag is armed by edit mode and disarmed when it ends', async () => {
    await editing(doc(['logo', 'clock'], []));
    customize.exitEditMode();

    await drag(shell('clock'), IN_STATUS);

    expect(putBodies()).toEqual([]);
  });
});

describe('a hidden status bar is a drop target while editing', () => {
  test('edit mode marks the page so the withdrawn bar can be shown', async () => {
    ({ customize, sync } = await boot({
      fetch: endpoint({ get: doc(['logo'], [], { statusVisible: false }) }),
      statusHidden: true,
    }));

    customize.enterEditMode();
    expect(document.documentElement.dataset.barEditing).toBe('true');

    customize.exitEditMode();
    expect(document.documentElement.dataset.barEditing).toBe(undefined);
  });
});
