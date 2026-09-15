/**
 * The Bluesky-queue bar item, happy-dom environment (configured globally):
 *   npx vitest run tests/interfaces/web_terminal/js/bar-item-queue.test.js
 *
 * What is pinned here is the item's posture towards a manager state the
 * bundle has never heard of. The bridge's state vocabulary is the bridge's,
 * and it grows: a browser that decides "not one of my four active states,
 * therefore at rest" shows a quiet chip and offers Start against a manager
 * that is doing something. The item derives REST from the one state that
 * affirmatively means rest, so every other state — invented, renamed, or
 * missing — reads as busy and withholds Start.
 *
 * These run against the REAL bar-host.js and bar-items.js, with `EventSource`
 * stubbed as a class so the item's own stream path is exercised.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const HOST_PATH = '../../../../src/osprey/interfaces/web_terminal/static/js/bar-host.js';
const ITEMS_PATH = '../../../../src/osprey/interfaces/web_terminal/static/js/bar-items.js';
const QUEUE_PATH = '../../../../src/osprey/interfaces/web_terminal/static/js/bar-item-queue.js';

/** @type {typeof import('../../../../src/osprey/interfaces/web_terminal/static/js/bar-host.js')} */
let host;
/** @type {typeof import('../../../../src/osprey/interfaces/web_terminal/static/js/bar-items.js')} */
let items;
/** Every warning the item logged this test. @type {string[]} */
let warnings;

/**
 * happy-dom ships no EventSource and the item opens its own; a class stub
 * keeps the module's stream path intact and hands the test the socket.
 */
class FakeEventSource {
  /** @type {FakeEventSource[]} */
  static opened = [];
  /** @param {string} url */
  constructor(url) {
    this.url = url;
    this.readyState = 1;
    /** @type {((event: {data: string}) => void) | null} */
    this.onmessage = null;
    /** @type {(() => void) | null} */
    this.onopen = null;
    /** @type {(() => void) | null} */
    this.onerror = null;
    FakeEventSource.opened.push(this);
  }
  close() {
    this.readyState = 2;
  }
}

/** Push one queue frame down every open socket. @param {unknown} frame */
function emit(frame) {
  for (const source of FakeEventSource.opened) {
    source.onmessage?.({ data: JSON.stringify(frame) });
  }
}

/**
 * One frame in the shape the bridge relays: a bounded `status`, the pending
 * plans and the one in motion.
 * @param {string} managerState
 * @param {Array<Record<string, unknown>>} [pending]
 */
function frame(managerState, pending = [{ name: 'count_plan' }]) {
  return {
    status: { available: true, manager_state: managerState, items_in_queue: pending.length },
    items: pending,
    running_item: null,
  };
}

beforeEach(async () => {
  warnings = [];
  FakeEventSource.opened = [];
  vi.stubGlobal('EventSource', FakeEventSource);
  vi.spyOn(console, 'warn').mockImplementation((/** @type {unknown[]} */ ...args) => {
    warnings.push(args.map(String).join(' '));
  });
  vi.resetModules();
  host = await import(HOST_PATH);
  items = await import(ITEMS_PATH);
  await import(QUEUE_PATH);
});

afterEach(() => {
  items.disposeBarItems();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  document.body.innerHTML = '';
});

/** The page as the bar hosts lay it out, with the queue item in the status bar. */
function seedDom() {
  document.body.innerHTML = `
    <header class="header">
      <div class="header-actions" data-bar-host="header"></div>
    </header>
    <footer class="status-bar" data-bar-host="status">
      <div class="bar-item" data-bar-item="bluesky-queue"
           data-bar-options='${JSON.stringify({ controls: 'full' })}'></div>
    </footer>
    <div id="bar-item-pool" hidden></div>
  `;
}

/** @param {string} selector */
function el(selector) {
  const node = document.querySelector(selector);
  if (!(node instanceof HTMLElement)) throw new Error(`no element matched ${selector}`);
  return node;
}

const chip = () => el('[data-bar-item="bluesky-queue"] .bar-queue');
const dotTone = () => chip().querySelector('.bar-queue-dot')?.getAttribute('data-tone');
const chipWord = () => chip().querySelector('.bar-queue-text')?.textContent;
const pop = () => el('[data-bar-item="bluesky-queue"] .bar-queue-pop');
const startButton = () =>
  /** @type {HTMLButtonElement | undefined} */ (
    Array.from(pop().querySelectorAll('button')).find((b) => b.textContent === 'Start')
  );

const unknownWarnings = () => warnings.filter((line) => line.includes('unknown manager state'));

describe('plan-queue item: a manager state the bundle cannot read', () => {
  test('an invented state paints the busy posture, withholds Start and warns once', () => {
    seedDom();
    host.hydrate();
    emit(frame('defragmenting_queue'));

    expect(dotTone()).toBe('warn');
    expect(chipWord()).toBe('defragmenting queue');

    chip().click();
    expect(startButton()?.disabled).toBe(true);
    expect(unknownWarnings()).toHaveLength(1);
    expect(unknownWarnings()[0]).toContain('defragmenting_queue');
  });

  test('a second frame carrying the same state does not warn again', () => {
    seedDom();
    host.hydrate();
    emit(frame('defragmenting_queue'));
    emit(frame('defragmenting_queue'));
    emit(frame('defragmenting_queue'));

    expect(unknownWarnings()).toHaveLength(1);
  });

  test('a missing manager state is busy too, worded `unknown`', () => {
    seedDom();
    host.hydrate();
    emit(frame(''));

    expect(dotTone()).toBe('warn');
    expect(chipWord()).toBe('unknown');
    chip().click();
    expect(startButton()?.disabled).toBe(true);
  });

  test('`idle` is at rest, offers Start and never warns', () => {
    seedDom();
    host.hydrate();
    emit(frame('idle'));

    expect(dotTone()).toBe('idle');
    expect(chipWord()).toBe('idle');
    chip().click();
    expect(startButton()?.disabled).toBe(false);
    expect(unknownWarnings()).toEqual([]);
  });

  test('a known active state still reads `running`', () => {
    seedDom();
    host.hydrate();
    emit(frame('executing_queue'));

    expect(dotTone()).toBe('active');
    expect(chipWord()).toBe('running');
    expect(unknownWarnings()).toEqual([]);
  });
});
