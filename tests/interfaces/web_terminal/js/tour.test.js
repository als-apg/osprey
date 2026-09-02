/**
 * FR7 — the onboarding tour's anchors route through bar-host's `isLive()`.
 *
 *   npx vitest run tests/interfaces/web_terminal/js/tour.test.js
 *
 * The tour's step list is filtered by "is this anchor on the page". Item bars
 * break that check: a bar item the layout does not name is MOVED into the
 * hidden `#bar-item-pool`, never removed, precisely so every id it owns keeps
 * resolving. Four tour anchors are bar items, so a presence check alone would
 * keep spotlighting chrome the operator cannot see — an empty highlight over a
 * zero-rect node in a hidden container, and a step count that lies.
 *
 * So the requirement has two halves and this file pins both: a step drops for
 * an item the operator REMOVED (its shell is in the pool), and it does not drop
 * for one that is merely somewhere else in the bars. The tour's own behaviour —
 * invite policy, navigation, copy, focus, storage scoping — is
 * `tests/interfaces/web_terminal/tour.test.mjs`, which this file does not
 * duplicate and must not disturb.
 *
 * These fixtures mount real SHELLS (`.bar-item[data-bar-item]`) around the
 * anchors rather than bare nodes, because that is the shape `isLive()` answers
 * about: pooling is a move of the shell, and the anchor inside it goes along.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import { startTour } from '../../../../src/osprey/interfaces/web_terminal/static/js/tour.js';

/** Every step title the tour renders, in order, for the current DOM. */
const ALL_TITLES = [
  'Ask in plain language',
  'Your control target',
  'Make it yours',
  'Arrange the bars',
  'Search everything',
  'Your workspace',
  'Facility knowledge',
  'The logbook',
  'Something wrong? Tell us',
  'Try it',
];

/**
 * The shell + anchor for one bar-hosted tour target. Kept as data so the
 * fixture and the pooling helper cannot disagree about which shell wraps which
 * anchor.
 * @type {Record<string, string>}
 */
const BAR_ITEMS = {
  'control-target':
    '<button class="control-target-chip" aria-expanded="false">' +
    '<span class="ctc-short">Simulator</span><span class="ctc-state">writes</span></button>',
  search: '<button id="command-palette-btn"></button>',
  display:
    '<osprey-display-menu id="display-menu">' +
    '<button class="display-menu-trigger" aria-expanded="false"></button>' +
    '<div class="display-menu-card">' +
    '<button class="display-menu-settings bar-customize-entry">Customize bars</button>' +
    '</div>' +
    '</osprey-display-menu>',
};

/** The full shell: a header host holding three item shells, plus the pool. */
function mountShell() {
  const shells = Object.entries(BAR_ITEMS)
    .map(([type, body]) => `<div class="bar-item" data-bar-item="${type}">${body}</div>`)
    .join('');
  document.body.innerHTML = `
    <header class="terminal-header">
      <div class="header-actions" data-bar-host="header">${shells}</div>
    </header>
    <nav class="panel-rail">
      <button class="panel-rail-button" data-panel-id="artifacts"></button>
      <button class="panel-rail-button" data-panel-id="okf"></button>
      <button class="panel-rail-button" data-panel-id="ariel"></button>
      <button id="panel-feedback-btn"></button>
    </nav>
    <div class="terminal-card"></div>
    <div id="bar-item-pool" hidden></div>`;
}

/**
 * Fail loudly on a drifted fixture rather than dereferencing null.
 * @param {string} selector
 * @returns {HTMLElement}
 */
function find(selector) {
  const node = document.querySelector(selector);
  if (!(node instanceof HTMLElement)) throw new Error(`expected ${selector}`);
  return node;
}

/**
 * Remove an item from the bars the way the host does it: MOVE its shell into
 * the pool. Deliberately not `remove()` — the whole point of FR7 is that the
 * node survives, so a test that deleted it would pass against a plain presence
 * check and prove nothing.
 * @param {string} type
 */
function poolItem(type) {
  find('#bar-item-pool').appendChild(find(`.bar-item[data-bar-item="${type}"]`));
}

/**
 * Put a pooled item back in the header host.
 * @param {string} type
 */
function restoreItem(type) {
  find('[data-bar-host="header"]').appendChild(find(`.bar-item[data-bar-item="${type}"]`));
}

/** The current card's title, or null when no card is up. */
const cardTitle = () => document.querySelector('.tour-title')?.textContent ?? null;

/** The current card's "Step N of M" kicker. */
const kicker = () => document.querySelector('.tour-kicker')?.textContent ?? null;

/**
 * Run the tour end to end and collect the title of every step it actually
 * showed. Walking the whole tour (rather than reading the step count) is what
 * makes the assertion specific: it names WHICH steps survived, so a filter that
 * dropped the wrong one cannot pass on an arithmetic match.
 * @returns {string[]}
 */
function walkTour() {
  startTour();
  /** @type {string[]} */
  const seen = [];
  // Bounded: `Done` on the last step tears the card down, so the loop ends on
  // its own — the cap only stops a regression from hanging the suite.
  for (let guard = 0; guard < ALL_TITLES.length + 5; guard += 1) {
    const title = cardTitle();
    if (title === null) break;
    seen.push(title);
    find('.tour-nav .tour-btn.primary').click();
  }
  return seen;
}

beforeEach(() => {
  // The tour arms a settle re-render (setTimeout) on any step that opens
  // something. Fake timers keep that off the walk, so each click advances
  // exactly one step.
  vi.useFakeTimers();
  localStorage.clear();
  mountShell();
});

afterEach(() => {
  document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
  vi.useRealTimers();
  document.body.innerHTML = '';
});

describe('anchors in a live bar', () => {
  test('every step runs when all three bar items are in the header', () => {
    expect(walkTour()).toEqual(ALL_TITLES);
  });

  test('an empty pool changes nothing', () => {
    expect(find('#bar-item-pool').children.length).toBe(0);
    expect(walkTour()).toEqual(ALL_TITLES);
  });
});

describe('anchors the operator removed', () => {
  test('pooling the search item drops its step', () => {
    poolItem('search');

    const seen = walkTour();
    expect(seen).not.toContain('Search everything');
    expect(seen).toEqual(ALL_TITLES.filter((t) => t !== 'Search everything'));
  });

  test('the dropped step is dropped for being POOLED, not for being gone', () => {
    poolItem('search');

    // The node is still in the document and still resolves by id — which is the
    // pool's entire purpose, and the reason a presence check could not have
    // caught this. The step drops anyway.
    expect(document.getElementById('command-palette-btn')).not.toBe(null);
    expect(find('#command-palette-btn').closest('#bar-item-pool')).not.toBe(null);
    expect(walkTour()).not.toContain('Search everything');
  });

  test('the step count follows the surviving steps', () => {
    poolItem('search');
    startTour();
    expect(kicker()).toBe(`Step 1 of ${ALL_TITLES.length - 1}`);
  });

  test('every pooled bar item drops its own steps and no others', () => {
    poolItem('search');
    poolItem('display');
    poolItem('control-target');

    // The display item owns two: its own step and the Customize row inside it.
    expect(walkTour()).toEqual([
      'Ask in plain language',
      'Your workspace',
      'Facility knowledge',
      'The logbook',
      'Something wrong? Tell us',
      'Try it',
    ]);
  });

  test('pooling the display item drops BOTH steps it carries', () => {
    poolItem('display');

    const seen = walkTour();
    expect(seen).not.toContain('Make it yours');
    expect(seen).not.toContain('Arrange the bars');
    expect(seen).toEqual(
      ALL_TITLES.filter((t) => t !== 'Make it yours' && t !== 'Arrange the bars')
    );
  });

  test('putting the item back brings its step back', () => {
    poolItem('search');
    expect(walkTour()).not.toContain('Search everything');

    restoreItem('search');
    expect(walkTour()).toEqual(ALL_TITLES);
  });
});

describe('anchors that are not bar items', () => {
  test('a genuinely absent anchor still drops its step', () => {
    find('#panel-feedback-btn').remove();

    const seen = walkTour();
    expect(seen).toEqual(ALL_TITLES.filter((t) => t !== 'Something wrong? Tell us'));
  });

  test('a missing Customize row drops only its own step', () => {
    // The row is projected into the display menu at runtime; when it is not
    // there the anchor is gone while the display menu around it stays put.
    find('.bar-customize-entry').remove();

    const seen = walkTour();
    expect(seen).toContain('Make it yours');
    expect(seen).toEqual(ALL_TITLES.filter((t) => t !== 'Arrange the bars'));
  });

  test('the terminal card and the rail are unaffected by the pool', () => {
    poolItem('search');
    poolItem('display');
    poolItem('control-target');

    const seen = walkTour();
    expect(seen[0]).toBe('Ask in plain language');
    expect(seen).toContain('Your workspace');
    expect(seen).toContain('The logbook');
  });
});

describe('an anchor parked mid-tour', () => {
  test('the tour skips over it instead of spotlighting the pool', () => {
    startTour();
    expect(cardTitle()).toBe('Ask in plain language');
    find('.tour-nav .tour-btn.primary').click();
    expect(cardTitle()).toBe('Your control target');

    // The ladder folds the display item away while the operator is reading —
    // narrowing the window is enough. Neither the display step nor the
    // Customize row riding inside it may be landed on.
    poolItem('display');
    find('.tour-nav .tour-btn.primary').click();

    expect(cardTitle()).toBe('Search everything');
  });
});
