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
 * Pinned beside it is what a Start click puts on the wire: the uid of the
 * queue the popover was showing. The rows above the button are the list the
 * operator approved, and quoting that queue's `plan_queue_uid` back is the
 * only thing that ties the click to it — a body that lost the uid would arm
 * whatever is queued by the time the bridge gets there.
 *
 * And pinned last is what the item does when that binding earns a refusal:
 * the bridge's sentence goes under the rows, the list is re-read so the
 * operator decides against what is actually queued, and nothing is resent.
 * The next click is a new approval, carrying the uid it can now see.
 *
 * These run against the REAL bar-host.js and bar-items.js, with `EventSource`
 * stubbed as a class so the item's own stream path is exercised, and `fetch`
 * stubbed to record what the item sends.
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
 *
 * `plan_queue_uid` is always present, as it is in the bridge's summary, and
 * null is the real "this bridge names no queue" case rather than a key the
 * fixture forgot.
 * @param {string} managerState
 * @param {Array<Record<string, unknown>>} [pending]
 * @param {string | null} [uid]
 */
function frame(managerState, pending = [{ name: 'count_plan' }], uid = null) {
  return {
    status: {
      available: true,
      manager_state: managerState,
      items_in_queue: pending.length,
      plan_queue_uid: uid,
    },
    items: pending,
    running_item: null,
  };
}

/** Every request the item sent this test. @type {{url: string, method: string, body: any}[]} */
let requests;

/**
 * The panel proxy, as far as this item is concerned: it records what was sent
 * and answers a bare 200 — the default for tests that assert on what goes
 * out. The refusal group re-stubs it to answer a 409 once.
 */
function stubFetch() {
  requests = [];
  vi.stubGlobal(
    'fetch',
    vi.fn(async (/** @type {any} */ url, /** @type {any} */ init) => {
      requests.push({
        url: String(url),
        method: init?.method ?? 'GET',
        body: init?.body === undefined ? undefined : JSON.parse(String(init.body)),
      });
      return { ok: true, status: 200, json: async () => ({}) };
    })
  );
}

/** Drain the microtask/timer queue the write chain runs through. */
async function flush() {
  for (let i = 0; i < 5; i++) await new Promise((resolve) => setTimeout(resolve, 0));
}

beforeEach(async () => {
  warnings = [];
  FakeEventSource.opened = [];
  stubFetch();
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

describe('plan-queue item: Start names the queue on screen', () => {
  /** The starts the item posted, in order. */
  const starts = () => requests.filter((request) => request.url.endsWith('/queue/start'));

  test('the body quotes the uid of the frame the popover rendered', async () => {
    seedDom();
    host.hydrate();
    emit(frame('idle', [{ name: 'count_plan' }], 'queue-uid-7'));

    chip().click();
    startButton()?.click();
    await flush();

    expect(starts()).toHaveLength(1);
    expect(starts()[0].method).toBe('POST');
    expect(starts()[0].body).toEqual({ expected_plan_queue_uid: 'queue-uid-7' });
  });

  test('the uid comes from the newest frame, not the one the popover opened on', async () => {
    // The popover re-renders on every frame, so the list under the button is
    // always the newest one. Sending an older uid would refuse a start the
    // operator never had reason to doubt.
    seedDom();
    host.hydrate();
    emit(frame('idle', [{ name: 'count_plan' }], 'queue-uid-7'));
    chip().click();
    emit(frame('idle', [{ name: 'count_plan' }, { name: 'scan_plan' }], 'queue-uid-8'));

    startButton()?.click();
    await flush();

    expect(starts()[0].body).toEqual({ expected_plan_queue_uid: 'queue-uid-8' });
  });

  test('a frame that names no queue sends an empty body', async () => {
    // A bridge that reported no uid leaves nothing to bind to, and the item
    // invents nothing: the start arms whatever is queued, as it always did.
    seedDom();
    host.hydrate();
    emit(frame('idle'));

    chip().click();
    startButton()?.click();
    await flush();

    expect(starts()).toHaveLength(1);
    expect(starts()[0].body).toEqual({});
  });
});

describe('plan-queue item: a start the bridge refuses because the queue moved', () => {
  /** The starts the item posted, in order. */
  const starts = () => requests.filter((request) => request.url.endsWith('/queue/start'));
  /** The queue reads the item issued, in order. */
  const reads = () =>
    requests.filter((request) => request.method === 'GET' && request.url.endsWith('/queue'));
  const note = () => pop().querySelector('.bar-queue-note')?.textContent;

  /** The bridge's sentence, shown verbatim. */
  const CONFLICT =
    "the queue has changed since it was approved: the start named queue 'queue-uid-7', but " +
    "the manager now holds 'queue-uid-8'. An item has been added, removed or re-ordered in " +
    'between, so the approved list is not the list this start would run. Re-read the queue, ' +
    'check what it now holds, and start again with the plan_queue_uid it reports.';

  /**
   * The proxy for this scenario: the first start is refused with the bridge's
   * 409, every later one is armed, and the queue read answers the queue the
   * manager now holds — two plans under uid 8.
   */
  function stubRefusingFetch() {
    requests = [];
    let refused = false;
    vi.stubGlobal(
      'fetch',
      vi.fn(async (/** @type {any} */ url, /** @type {any} */ init) => {
        const method = init?.method ?? 'GET';
        requests.push({
          url: String(url),
          method,
          body: init?.body === undefined ? undefined : JSON.parse(String(init.body)),
        });
        if (method === 'POST' && String(url).endsWith('/queue/start') && !refused) {
          refused = true;
          return {
            ok: false,
            status: 409,
            json: async () => ({
              detail: {
                code: 'queue_changed_since_approval',
                detail: CONFLICT,
                plan_queue_uid: 'queue-uid-8',
                expected_plan_queue_uid: 'queue-uid-7',
              },
            }),
          };
        }
        if (method === 'GET' && String(url).endsWith('/queue')) {
          return {
            ok: true,
            status: 200,
            json: async () =>
              frame('idle', [{ name: 'count_plan' }, { name: 'scan_plan' }], 'queue-uid-8'),
          };
        }
        return { ok: true, status: 200, json: async () => ({}) };
      })
    );
  }

  test('a 409 is shown, the list re-read, nothing resent, and the next click carries the new uid', async () => {
    // The refusal the bound start exists to produce, at the mouse. What
    // matters is what the item does with it: one start went out, the bridge's
    // sentence is in the popover, and nothing else leaves on its own. An
    // automatic retry would be the item deciding the operator would have
    // approved a list they were never shown — the substitution the uid was
    // added to prevent.
    //
    // Where the panel waits for its next stream frame, the item re-reads the
    // queue itself, so the rows under the refusal are the ones the manager
    // actually holds before the operator decides again.
    stubRefusingFetch();
    seedDom();
    host.hydrate();
    emit(frame('idle', [{ name: 'count_plan' }], 'queue-uid-7'));

    chip().click();
    startButton()?.click();
    await flush();

    expect(starts()).toHaveLength(1);
    expect(starts()[0].method).toBe('POST');
    expect(starts()[0].body).toEqual({ expected_plan_queue_uid: 'queue-uid-7' });
    // The refusal is a state, not a receipt: it stays under the rows for the
    // look, in the bridge's own words.
    expect(note()).toBe(CONFLICT);

    // The item's own re-read of the moved queue — exactly one, and the rows
    // it painted are the two plans the manager now holds.
    expect(reads()).toHaveLength(1);
    expect(
      Array.from(pop().querySelectorAll('.bar-queue-row-name')).map((row) => row.textContent)
    ).toEqual(['count_plan', 'scan_plan']);

    // Several more turns of the loop: a scheduled resend would land in one.
    await flush();
    expect(starts()).toHaveLength(1);
    expect(reads()).toHaveLength(1);
    expect(note()).toBe(CONFLICT);

    // The second click is the operator approving the list they can now see.
    expect(startButton()?.disabled).toBe(false);
    startButton()?.click();
    await flush();

    expect(starts()).toHaveLength(2);
    expect(starts()[1].body).toEqual({ expected_plan_queue_uid: 'queue-uid-8' });
    // A 200 leaves no refusal behind, and an accepted start is not re-read:
    // the stream carries the queue from here.
    expect(note()).toBeUndefined();
    expect(reads()).toHaveLength(1);
  });
});
