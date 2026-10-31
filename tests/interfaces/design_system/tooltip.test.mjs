// @ts-check
/**
 * Unit tests for the shared tooltip layer (design_system/static/js/tooltip.js):
 *   npx vitest run tests/interfaces/design_system/tooltip.test.mjs
 *
 * happy-dom does no layout, so the trigger's rect and the tip's own size are
 * stubbed and the test asserts the fixed position the module derives from
 * them. Time is faked: the 450 ms delay and the 300 ms warm period are the
 * behaviour under test.
 */

import { test, expect, describe, beforeAll, afterAll, beforeEach, afterEach, vi } from 'vitest';

import { hideTooltip, initTooltips, readTitle, writeTitle } from '/design-system/js/tooltip.js';

const SHOW_DELAY = 450;
const WARM = 300;

/** @param {string} id @returns {HTMLElement} */
const $ = (id) => /** @type {HTMLElement} */ (document.getElementById(id));

/** @returns {HTMLElement[]} */
const layers = () =>
  /** @type {HTMLElement[]} */ (Array.from(document.querySelectorAll('.osprey-tooltip')));

/** The one layer, or null before the first tip is asked for. */
const layer = () => layers()[0] ?? null;

/** Whether a tip is on screen now. */
const shown = () => !!layer() && !layer().hidden;

/** @param {EventTarget} target @param {string} type */
function fire(target, type) {
  target.dispatchEvent(new Event(type, { bubbles: true }));
}

/** @param {number} ms */
const wait = (ms) => vi.advanceTimersByTime(ms);

/**
 * A DOMRect for the stub.
 * @param {number} left @param {number} top @param {number} width @param {number} height
 * @returns {DOMRect}
 */
function rectOf(left, top, width, height) {
  return /** @type {DOMRect} */ ({
    left,
    top,
    width,
    height,
    right: left + width,
    bottom: top + height,
    x: left,
    y: top,
    toJSON() {},
  });
}

/**
 * Stub whether the browser would draw a focus indicator on `el`: true for
 * keyboard focus, false for the focus a mouse press gives.
 * @param {HTMLElement} el
 * @param {boolean} visible
 */
function keyboardFocus(el, visible) {
  /** @type {any} */ (el).matches = () => visible;
}

/** Stub where `#id` sits on the screen. @param {string} id @param {DOMRect} rect */
function place(id, rect) {
  $(id).getBoundingClientRect = () => rect;
}

/** The size the layer reports, since nothing is laid out. */
const tipSize = { width: 200, height: 40 };

const sizeProps = /** @type {const} */ (['offsetWidth', 'offsetHeight']);
/** @type {Map<string, PropertyDescriptor | undefined>} */
const originalSize = new Map();

beforeAll(() => {
  for (const prop of sizeProps) {
    originalSize.set(prop, Object.getOwnPropertyDescriptor(HTMLElement.prototype, prop));
    Object.defineProperty(HTMLElement.prototype, prop, {
      configurable: true,
      get() {
        if (!this.classList.contains('osprey-tooltip')) return 0;
        return prop === 'offsetWidth' ? tipSize.width : tipSize.height;
      },
    });
  }
});

afterAll(() => {
  for (const prop of sizeProps) {
    const original = originalSize.get(prop);
    if (original) Object.defineProperty(HTMLElement.prototype, prop, original);
    else delete (/** @type {any} */ (HTMLElement.prototype)[prop]);
  }
});

let clock = 1_000_000;
/** @type {() => void} */
let dispose;
/** @type {(el: HTMLElement) => 'label+detail' | 'detail'} */
let mode;

beforeEach(() => {
  vi.useFakeTimers();
  // Each test starts well clear of the last one's hide, so no test begins warm.
  clock += 60_000;
  vi.setSystemTime(clock);
  hideTooltip();
  wait(10 * WARM);

  document.body.innerHTML = `
    <nav id="root">
      <button id="row" type="button" data-tip="Agent" data-tip-detail="Chat with the agent" aria-label="Agent"
              title="Agent · right-click for actions"><span id="row-icon"></span></button>
      <button id="bare" type="button" data-tip="Search"></button>
      <button id="titled" type="button" title="Close tile"></button>
      <button id="detail-only" type="button" data-tip-detail="Only an explanation"></button>
      <button id="opener" type="button" data-tip="Artifacts" aria-controls="flyout" aria-label="Artifacts"></button>
      <button id="described" type="button" data-tip="Docs" aria-describedby="hint"></button>
      <button id="plain" type="button">No tip</button>
    </nav>
    <button id="elsewhere" type="button" data-tip="Not in the root"></button>
    <div id="flyout" class="osprey-popover" hidden></div>
  `;
  window.innerWidth = 1000;
  window.innerHeight = 600;
  tipSize.width = 200;
  tipSize.height = 40;
  place('row', rectOf(8, 100, 40, 28));
  mode = () => 'label+detail';
  dispose = initTooltips($('root'), { mode: (el) => mode(el) });
});

afterEach(() => {
  dispose();
  vi.useRealTimers();
});

describe('the delay and the warm period', () => {
  test('a tip shows 450 ms after the pointer enters, not before', () => {
    fire($('row'), 'pointerover');
    expect(shown()).toBe(false);

    wait(SHOW_DELAY - 1);
    expect(shown()).toBe(false);

    wait(1);
    expect(shown()).toBe(true);
  });

  test('one body-level role=tooltip layer holds a bold label over a detail line', () => {
    fire($('row-icon'), 'pointerover');
    wait(SHOW_DELAY);

    const tip = layer();
    expect(layers()).toHaveLength(1);
    expect(tip.parentElement).toBe(document.body);
    expect(tip.getAttribute('role')).toBe('tooltip');
    expect(tip.querySelector('b.osprey-tooltip-label')?.textContent).toBe('Agent');
    expect(tip.querySelector('span.osprey-tooltip-detail')?.textContent).toBe(
      'Chat with the agent'
    );
  });

  test('leaving before the delay is up cancels the tip', () => {
    fire($('row'), 'pointerover');
    wait(SHOW_DELAY - 100);
    $('row').dispatchEvent(new Event('pointerleave'));
    wait(SHOW_DELAY);

    expect(shown()).toBe(false);
  });

  test('moving from one tip to another within 300 ms of the hide shows at once', () => {
    fire($('row'), 'pointerover');
    wait(SHOW_DELAY);
    expect(shown()).toBe(true);

    $('row').dispatchEvent(new Event('pointerleave'));
    wait(WARM - 1);
    fire($('bare'), 'pointerover');

    expect(shown()).toBe(true);
    expect(layer().textContent).toBe('Search');
  });

  test('after 300 ms without a tip the wait is back', () => {
    fire($('row'), 'pointerover');
    wait(SHOW_DELAY);
    $('row').dispatchEvent(new Event('pointerleave'));

    wait(WARM);
    fire($('bare'), 'pointerover');
    expect(shown()).toBe(false);

    wait(SHOW_DELAY);
    expect(shown()).toBe(true);
  });

  test('a tip that never showed does not start a warm period', () => {
    fire($('row'), 'pointerover');
    wait(SHOW_DELAY - 100);
    $('row').dispatchEvent(new Event('pointerleave'));

    fire($('bare'), 'pointerover');
    expect(shown()).toBe(false);
  });

  test('moving inside the element that has the tip does not restart the wait', () => {
    fire($('row-icon'), 'pointerover');
    wait(SHOW_DELAY - 100);
    fire($('row'), 'pointerover');
    wait(100);

    expect(shown()).toBe(true);
  });
});

describe('what the tip says', () => {
  test('label + detail mode shows both', () => {
    fire($('row'), 'pointerover');
    wait(SHOW_DELAY);

    expect(layer().textContent).toBe('AgentChat with the agent');
  });

  test('detail mode leaves the label out', () => {
    mode = () => 'detail';
    fire($('row'), 'pointerover');
    wait(SHOW_DELAY);

    expect(layer().querySelector('.osprey-tooltip-label')).toBeNull();
    expect(layer().textContent).toBe('Chat with the agent');
  });

  test('the mode is asked per element', () => {
    mode = (el) => (el.id === 'row' ? 'detail' : 'label+detail');
    fire($('row'), 'pointerover');
    wait(SHOW_DELAY);
    expect(layer().textContent).toBe('Chat with the agent');

    fire($('detail-only'), 'pointerover');
    wait(SHOW_DELAY);
    expect(layer().textContent).toBe('Only an explanation');
  });

  test('an element with a label and no detail shows the label alone', () => {
    fire($('bare'), 'pointerover');
    wait(SHOW_DELAY);

    expect(layer().querySelector('.osprey-tooltip-detail')).toBeNull();
    expect(layer().textContent).toBe('Search');
  });

  test('a mode that yields no text shows no tip and leaves the title alone', () => {
    mode = () => 'detail';
    fire($('bare'), 'pointerover');
    fire($('titled'), 'pointerover');
    wait(SHOW_DELAY);

    expect(shown()).toBe(false);
    expect($('titled').title).toBe('Close tile');
  });

  test('an element with no tip text shows nothing', () => {
    fire($('plain'), 'pointerover');
    wait(SHOW_DELAY);

    expect(shown()).toBe(false);
  });

  test('moving onto an element without a tip hides the one that shows', () => {
    fire($('row'), 'pointerover');
    wait(SHOW_DELAY);
    fire($('plain'), 'pointerover');

    expect(shown()).toBe(false);
  });

  test('ignores elements outside the root', () => {
    fire($('elsewhere'), 'pointerover');
    wait(SHOW_DELAY);

    expect(shown()).toBe(false);
  });

  test('leaves the root and its listeners alone after dispose', () => {
    dispose();
    fire($('row'), 'pointerover');
    wait(SHOW_DELAY);

    expect(shown()).toBe(false);
  });
});

describe('the native title', () => {
  test('a title alone is the label', () => {
    fire($('titled'), 'pointerover');
    wait(SHOW_DELAY);

    expect(layer().querySelector('.osprey-tooltip-label')?.textContent).toBe('Close tile');
  });

  test('data-tip wins over the title as the label', () => {
    fire($('row'), 'pointerover');
    wait(SHOW_DELAY);

    expect(layer().querySelector('.osprey-tooltip-label')?.textContent).toBe('Agent');
  });

  test('parks the title while the tip shows and restores it on hide', () => {
    fire($('row-icon'), 'pointerover');
    expect($('row').title).toBe('Agent · right-click for actions');

    wait(SHOW_DELAY);
    expect($('row').hasAttribute('title')).toBe(false);
    expect(readTitle($('row'))).toBe('Agent · right-click for actions');

    $('row').dispatchEvent(new Event('pointerleave'));
    expect($('row').title).toBe('Agent · right-click for actions');
    expect($('row').hasAttribute('data-tip-title')).toBe(false);
  });

  test('a title-only element keeps its title again after the tip', () => {
    fire($('titled'), 'pointerover');
    wait(SHOW_DELAY);
    expect($('titled').hasAttribute('title')).toBe(false);

    hideTooltip();
    expect($('titled').title).toBe('Close tile');
  });

  test("moving to another element restores the first one's title", () => {
    fire($('row'), 'pointerover');
    wait(SHOW_DELAY);
    fire($('opener'), 'pointerover');

    expect($('row').title).toBe('Agent · right-click for actions');
  });

  test('a title written while the tip shows lands after hide, not beside the tip', () => {
    fire($('row'), 'pointerover');
    wait(SHOW_DELAY);
    writeTitle($('row'), 'Agent · agent touched 9:12');
    expect($('row').hasAttribute('title')).toBe(false);

    hideTooltip();
    expect($('row').title).toBe('Agent · agent touched 9:12');
  });

  test('without a tip, writeTitle sets the native title', () => {
    writeTitle($('row'), 'Agent');

    expect($('row').title).toBe('Agent');
  });
});

describe('placement', () => {
  test('right (the default) sits 10 px right of the element, centred vertically', () => {
    fire($('row'), 'pointerover');
    wait(SHOW_DELAY);

    expect(layer().style.left).toBe('58px');
    expect(layer().style.top).toBe('94px');
    expect(layer().dataset.side).toBe('right');
  });

  test('right flips to the left of the element at the window edge', () => {
    place('row', rectOf(910, 100, 40, 28));
    fire($('row'), 'pointerover');
    wait(SHOW_DELAY);

    expect(layer().style.left).toBe('700px');
    expect(layer().dataset.side).toBe('left');
  });

  test('below sits 10 px under the element, centred horizontally', () => {
    place('row', rectOf(400, 100, 100, 28));
    $('row').dataset.tipPlace = 'below';
    fire($('row'), 'pointerover');
    wait(SHOW_DELAY);

    expect(layer().style.left).toBe('350px');
    expect(layer().style.top).toBe('138px');
    expect(layer().dataset.side).toBe('below');
  });

  test('below flips above at the bottom of the window', () => {
    place('row', rectOf(400, 560, 100, 28));
    $('row').dataset.tipPlace = 'below';
    fire($('row'), 'pointerover');
    wait(SHOW_DELAY);

    expect(layer().style.top).toBe('510px');
    expect(layer().dataset.side).toBe('above');
  });

  test('above sits 10 px over the element', () => {
    place('row', rectOf(400, 300, 100, 28));
    $('row').dataset.tipPlace = 'above';
    fire($('row'), 'pointerover');
    wait(SHOW_DELAY);

    expect(layer().style.top).toBe('250px');
    expect(layer().dataset.side).toBe('above');
  });

  test('above flips below at the top of the window', () => {
    place('row', rectOf(400, 20, 100, 28));
    $('row').dataset.tipPlace = 'above';
    fire($('row'), 'pointerover');
    wait(SHOW_DELAY);

    expect(layer().style.top).toBe('58px');
    expect(layer().dataset.side).toBe('below');
  });

  test('the tip stays 8 px inside the window', () => {
    place('row', rectOf(0, 0, 40, 28));
    $('row').dataset.tipPlace = 'below';
    fire($('row'), 'pointerover');
    wait(SHOW_DELAY);
    expect(layer().style.left).toBe('8px');

    place('row', rectOf(980, 590, 20, 10));
    $('row').dataset.tipPlace = 'right';
    hideTooltip();
    fire($('plain'), 'pointerover');
    fire($('row'), 'pointerover');
    wait(SHOW_DELAY);
    expect(layer().style.top).toBe('552px');
  });
});

describe('hiding', () => {
  /** Show the Agent row's tip. */
  function showRow() {
    fire($('row'), 'pointerover');
    wait(SHOW_DELAY);
    expect(shown()).toBe(true);
  }

  test('hides when the pointer leaves the element', () => {
    showRow();
    $('row').dispatchEvent(new Event('pointerleave'));

    expect(shown()).toBe(false);
  });

  test('hides when the pointer leaves the root', () => {
    showRow();
    $('root').dispatchEvent(new Event('pointerleave'));

    expect(shown()).toBe(false);
  });

  test('hides when the pointer leaves the window', () => {
    showRow();
    fire($('row'), 'pointerout');

    expect(shown()).toBe(false);
  });

  test('a child that the pointer leaves does not hide the tip', () => {
    showRow();
    $('row-icon').dispatchEvent(new Event('pointerleave'));

    expect(shown()).toBe(true);
  });

  test('hides on focus out', () => {
    keyboardFocus($('row'), true);
    fire($('row'), 'focusin');
    wait(SHOW_DELAY);
    expect(shown()).toBe(true);

    fire($('row'), 'focusout');
    expect(shown()).toBe(false);
  });

  test('hides on a pointer press anywhere', () => {
    showRow();
    fire($('plain'), 'pointerdown');

    expect(shown()).toBe(false);
  });

  test('a press before the delay is up cancels the tip', () => {
    fire($('row'), 'pointerover');
    wait(SHOW_DELAY - 100);
    fire($('row'), 'pointerdown');
    wait(SHOW_DELAY);

    expect(shown()).toBe(false);
  });

  test('hides on Escape', () => {
    showRow();
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));

    expect(shown()).toBe(false);
  });

  test('another key leaves the tip', () => {
    showRow();
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'a' }));

    expect(shown()).toBe(true);
  });

  test('after a press or Escape the element stays quiet until it is left', () => {
    showRow();
    fire($('row'), 'pointerdown');

    fire($('row-icon'), 'pointerover');
    wait(SHOW_DELAY);
    expect(shown()).toBe(false);

    $('row').dispatchEvent(new Event('pointerleave'));
    fire($('row'), 'pointerover');
    wait(SHOW_DELAY);
    expect(shown()).toBe(true);
  });

  test('hides when its popover opens, and shows nothing while that popover is open', () => {
    fire($('opener'), 'pointerover');
    $('flyout').hidden = false;
    wait(SHOW_DELAY);
    expect(shown()).toBe(false);

    fire($('plain'), 'pointerover');
    fire($('opener'), 'pointerover');
    wait(SHOW_DELAY);
    expect(shown()).toBe(false);

    $('flyout').hidden = true;
    fire($('plain'), 'pointerover');
    fire($('opener'), 'pointerover');
    wait(SHOW_DELAY);
    expect(shown()).toBe(true);
    expect(layer().textContent).toBe('Artifacts');
  });

  test('hideTooltip cancels a tip that is waiting', () => {
    fire($('row'), 'pointerover');
    hideTooltip();
    wait(SHOW_DELAY);

    expect(shown()).toBe(false);
  });
});

describe('keyboard focus', () => {
  test('keyboard focus shows the tip after the same delay', () => {
    keyboardFocus($('row'), true);
    fire($('row'), 'focusin');
    expect(shown()).toBe(false);

    wait(SHOW_DELAY);
    expect(shown()).toBe(true);
  });

  test('the focus a mouse press gives does not show a tip', () => {
    keyboardFocus($('row'), false);
    fire($('row'), 'focusin');
    wait(SHOW_DELAY);

    expect(shown()).toBe(false);
  });
});

describe('aria-describedby', () => {
  test('names the layer while the tip shows and goes after', () => {
    fire($('bare'), 'pointerover');
    expect($('bare').hasAttribute('aria-describedby')).toBe(false);

    wait(SHOW_DELAY);
    expect($('bare').getAttribute('aria-describedby')).toBe(layer().id);
    expect(layer().id).not.toBe('');

    $('bare').dispatchEvent(new Event('pointerleave'));
    expect($('bare').hasAttribute('aria-describedby')).toBe(false);
  });

  test("keeps the element's own description", () => {
    fire($('described'), 'pointerover');
    wait(SHOW_DELAY);
    expect($('described').getAttribute('aria-describedby')).toBe(`hint ${layer().id}`);

    hideTooltip();
    expect($('described').getAttribute('aria-describedby')).toBe('hint');
  });

  test('goes from the first element when the tip moves to the next', () => {
    fire($('bare'), 'pointerover');
    wait(SHOW_DELAY);
    fire($('row'), 'pointerover');

    expect($('bare').hasAttribute('aria-describedby')).toBe(false);
  });
});
