// @ts-check
/**
 * Unit tests for the shared tooltip layer (design_system/static/js/tooltip.js):
 *   npx vitest run tests/interfaces/design_system/tooltip.test.mjs
 *
 * happy-dom does no layout, so the trigger's rect is stubbed and the test
 * asserts the fixed position the module derives from it.
 */

import { test, expect, describe, beforeEach } from 'vitest';

import { hideTooltip, initTooltips, readTitle, writeTitle } from '/design-system/js/tooltip.js';

/** @param {string} id @returns {HTMLElement} */
const $ = (id) => /** @type {HTMLElement} */ (document.getElementById(id));

/** @returns {HTMLElement[]} */
const layers = () =>
  /** @type {HTMLElement[]} */ (Array.from(document.querySelectorAll('.osprey-tooltip')));

/** @param {EventTarget} target @param {string} type */
function fire(target, type) {
  target.dispatchEvent(new Event(type, { bubbles: true }));
}

let collapsed = true;

beforeEach(() => {
  hideTooltip();
  document.body.innerHTML = `
    <nav id="sidebar">
      <button id="row" type="button" data-tip="Agent" aria-label="Agent" title="Agent · right-click for actions"><span id="row-icon"></span></button>
      <button id="opener" type="button" data-tip="Artifacts" aria-controls="flyout" aria-label="Artifacts"></button>
      <button id="plain" type="button">No tip</button>
    </nav>
    <button id="elsewhere" type="button" data-tip="Not in the sidebar"></button>
    <div id="flyout" class="osprey-popover" hidden></div>
  `;
  $('row').getBoundingClientRect = () =>
    /** @type {DOMRect} */ ({
      top: 100,
      bottom: 128,
      left: 8,
      right: 48,
      width: 40,
      height: 28,
      x: 8,
      y: 100,
      toJSON() {},
    });
  collapsed = true;
  initTooltips($('sidebar'), { when: () => collapsed });
});

describe('initTooltips', () => {
  test('one body-level role=tooltip layer shows the data-tip on hover', () => {
    fire($('row-icon'), 'pointerover');

    const [tip] = layers();
    expect(layers()).toHaveLength(1);
    expect(tip.parentElement).toBe(document.body);
    expect(tip.getAttribute('role')).toBe('tooltip');
    expect(tip.hidden).toBe(false);
    expect(tip.textContent).toBe('Agent');
  });

  test('sits 10 px right of the element, centred vertically', () => {
    fire($('row'), 'pointerover');

    const [tip] = layers();
    expect(tip.style.left).toBe('58px');
    expect(tip.style.top).toBe(`${114 - tip.offsetHeight / 2}px`);
  });

  test('shows on focus and hides on focus out', () => {
    fire($('row'), 'focusin');
    expect(layers()[0].hidden).toBe(false);

    fire($('row'), 'focusout');
    expect(layers()[0].hidden).toBe(true);
  });

  test('hides on pointer leave and on Escape', () => {
    fire($('row'), 'pointerover');
    $('sidebar').dispatchEvent(new Event('pointerleave'));
    expect(layers()[0].hidden).toBe(true);

    fire($('row'), 'pointerover');
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    expect(layers()[0].hidden).toBe(true);
  });

  test('moving onto an element without data-tip hides it', () => {
    fire($('row'), 'pointerover');
    fire($('plain'), 'pointerover');

    expect(layers()[0].hidden).toBe(true);
  });

  test('stays hidden while the predicate is false', () => {
    collapsed = false;
    fire($('row'), 'pointerover');
    fire($('row'), 'focusin');

    expect(layers().every((tip) => tip.hidden)).toBe(true);
  });

  test('never shows while the element controls an open popover', () => {
    $('flyout').hidden = false;
    fire($('opener'), 'pointerover');
    expect(layers().every((tip) => tip.hidden)).toBe(true);

    $('flyout').hidden = true;
    fire($('opener'), 'pointerover');
    expect(layers()[0].hidden).toBe(false);
    expect(layers()[0].textContent).toBe('Artifacts');
  });

  test('ignores [data-tip] elements outside the root', () => {
    fire($('elsewhere'), 'pointerover');

    expect(layers().every((tip) => tip.hidden)).toBe(true);
  });

  test('parks the native title while the tip shows and restores it on hide', () => {
    fire($('row-icon'), 'pointerover');
    expect($('row').hasAttribute('title')).toBe(false);
    expect(readTitle($('row'))).toBe('Agent · right-click for actions');

    $('sidebar').dispatchEvent(new Event('pointerleave'));
    expect($('row').title).toBe('Agent · right-click for actions');
    expect($('row').hasAttribute('data-tip-title')).toBe(false);
  });

  test("moving to another row restores the first row's title", () => {
    fire($('row'), 'pointerover');
    fire($('opener'), 'pointerover');

    expect($('row').title).toBe('Agent · right-click for actions');
  });

  test('a title written while the tip shows lands after hide, not beside the tip', () => {
    fire($('row'), 'pointerover');
    writeTitle($('row'), 'Agent · agent touched 9:12');
    expect($('row').hasAttribute('title')).toBe(false);

    hideTooltip();
    expect($('row').title).toBe('Agent · agent touched 9:12');
  });

  test('the expanded sidebar leaves the native title alone', () => {
    collapsed = false;
    fire($('row'), 'pointerover');

    expect($('row').title).toBe('Agent · right-click for actions');
    writeTitle($('row'), 'Agent');
    expect($('row').title).toBe('Agent');
  });
});
