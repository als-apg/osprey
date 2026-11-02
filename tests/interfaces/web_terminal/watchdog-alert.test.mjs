// @ts-check
/* watchdog-alert.js: the operator's answer to a watchdog flag.
 *
 * What is pinned here:
 * - the flag route is the one truth: a frame only triggers a re-read, and a read
 *   that fails leaves what is on screen;
 * - only a click answers: no Escape, no scrim click, and no button takes focus;
 * - an answer names the flags on screen when it was clicked, and a flag that joined
 *   later keeps the dialog up;
 * - the poll runs until the route answers `enabled: false`;
 * - no flags means no dialog and no statement about the session.
 *
 * Seams: `fetch` is stubbed and `terminal.js` is replaced, because it pulls the
 * xterm and design-system graph; everything else is the real module. Module state
 * is isolated with vi.resetModules() and a dynamic import per test.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const MODULE = '../../../src/osprey/interfaces/web_terminal/static/js/watchdog-alert.js';

const session = vi.hoisted(() => ({ id: /** @type {string|null} */ ('sess-1') }));

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/terminal.js', () => ({
  getCurrentSessionId: () => session.id,
}));

/** @type {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/watchdog-alert.js')} */
let alert;

/** @param {Partial<Record<string, any>>} [over] */
function flag(over = {}) {
  return {
    flag_id: 'f-1',
    watchdog: 'beam-guard',
    reason: 'it moves the beam',
    prompt_excerpt: 'turn the corrector down',
    stop_unconfirmed: false,
    started: [],
    ...over,
  };
}

/** @param {any} body */
function ok(body) {
  return { ok: true, status: 200, json: async () => body };
}

/** The next answers of the flag route; the last one repeats. */
/** @type {any[]} */
let reads;
/** @type {any} */
let answerReply;
/** @type {any} */
let fetchMock;
/** @type {MutationObserver[]} */
let observers;

/** @returns {any[][]} */
function flagReads() {
  return fetchMock.mock.calls.filter((/** @type {any[]} */ c) =>
    String(c[0]).includes('/api/watchdog/flags?')
  );
}

/** @returns {any[][]} */
function answers() {
  return fetchMock.mock.calls.filter((/** @type {any[]} */ c) =>
    String(c[0]).endsWith('/api/watchdog/flags/answer')
  );
}

async function settle() {
  for (let i = 0; i < 30; i += 1) await Promise.resolve();
}

function dialog() {
  return document.querySelector('[role="alertdialog"]');
}

/** @param {string} selector */
function button(selector) {
  return /** @type {HTMLButtonElement} */ (document.querySelector(selector));
}

/** Boot the module against the queued reads and let the first read land. */
async function boot() {
  alert.initWatchdogAlert();
  await settle();
}

beforeEach(async () => {
  vi.resetModules();
  document.body.innerHTML = '';
  session.id = 'sess-1';
  reads = [{ enabled: true, flags: [] }];
  answerReply = ok({ released: [], stop_unconfirmed: [] });
  fetchMock = vi.fn(async (/** @type {string} */ url) => {
    if (String(url).endsWith('/api/watchdog/flags/answer')) {
      return typeof answerReply === 'function' ? answerReply() : answerReply;
    }
    if (String(url).includes('/api/watchdog/flags?')) {
      const next = reads.length > 1 ? reads.shift() : reads[0];
      return typeof next === 'function' ? next() : ok(next);
    }
    return ok({});
  });
  vi.stubGlobal('fetch', fetchMock);
  vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval'] });

  // Each import starts its own refusal watch; one must not outlive its test.
  observers = [];
  const Real = globalThis.MutationObserver;
  vi.stubGlobal(
    'MutationObserver',
    class extends Real {
      /** @param {MutationCallback} callback */
      constructor(callback) {
        super(callback);
        observers.push(this);
      }
    }
  );

  alert = await import(MODULE);
});

afterEach(() => {
  observers.forEach((observer) => observer.disconnect());
  vi.clearAllTimers();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe('reading the flags', () => {
  test('the read names the current session, and no flags means no dialog and no words', async () => {
    session.id = 'a b';
    await boot();

    expect(flagReads()).toHaveLength(1);
    expect(flagReads()[0][0]).toContain('session_key=a%20b');
    expect(dialog()).toBeNull();
    expect(document.body.textContent).toBe('');
  });

  test('a page with no session reads nothing', async () => {
    session.id = null;
    await boot();

    expect(flagReads()).toHaveLength(0);
  });

  test('init is idempotent', async () => {
    await boot();
    await boot();

    expect(flagReads()).toHaveLength(1);
  });

  test('a read that fails is not "no flags": the dialog stays, with a note', async () => {
    reads = [{ enabled: true, flags: [flag()] }, () => Promise.reject(new Error('down'))];
    const logged = vi.spyOn(console, 'error').mockImplementation(() => {});
    await boot();

    await alert.refetchFlags();

    expect(dialog()).not.toBeNull();
    expect(document.querySelector('.watchdog-alert-flag')).not.toBeNull();
    expect(document.querySelector('.watchdog-alert-note')?.textContent).toContain(
      'Could not refresh'
    );
    logged.mockRestore();
  });

  test('a slow read cannot repaint over a newer one', async () => {
    /** @type {(value: any) => void} */
    let release = () => {};
    const slow = new Promise((resolve) => {
      release = resolve;
    });
    reads = [{ enabled: true, flags: [] }, () => slow, { enabled: true, flags: [flag()] }];
    await boot();

    const older = alert.refetchFlags();
    await alert.refetchFlags();
    release(ok({ enabled: true, flags: [] }));
    await older;

    expect(dialog()).not.toBeNull();
  });

  test('the poll re-reads until the route answers enabled false, then stops', async () => {
    await boot();

    await vi.advanceTimersByTimeAsync(alert.WATCHDOG_POLL_MS);
    await settle();
    expect(flagReads()).toHaveLength(2);

    reads = [{ enabled: false, flags: [] }];
    await vi.advanceTimersByTimeAsync(alert.WATCHDOG_POLL_MS);
    await settle();
    expect(flagReads()).toHaveLength(3);

    await vi.advanceTimersByTimeAsync(alert.WATCHDOG_POLL_MS * 3);
    await settle();
    expect(flagReads()).toHaveLength(3);
  });

  test('a poll that recovers a dropped frame opens the dialog', async () => {
    await boot();
    expect(dialog()).toBeNull();

    reads = [{ enabled: true, flags: [flag()] }];
    await vi.advanceTimersByTimeAsync(alert.WATCHDOG_POLL_MS);
    await settle();

    expect(dialog()).not.toBeNull();
  });
});

describe('frames are change notices', () => {
  test('a frame re-reads, and nothing is rendered from its body', async () => {
    await boot();

    alert.onWatchdogFrame({
      type: 'watchdog_flag',
      session_key: 'sess-1',
      flag_id: 'from-the-frame',
      watchdog: 'from-the-frame',
    });
    await settle();

    expect(flagReads()).toHaveLength(2);
    expect(dialog()).toBeNull();
    expect(document.body.textContent).not.toContain('from-the-frame');
  });

  test('isWatchdogFrame names the watchdog frames only', () => {
    expect(alert.isWatchdogFrame({ type: 'watchdog_resolved' })).toBe(true);
    expect(alert.isWatchdogFrame({ type: 'agent_activity' })).toBe(false);
    expect(alert.isWatchdogFrame({})).toBe(false);
  });

  test('a session_held refusal in the conversation re-reads; another refusal does not', async () => {
    await boot();

    const other = document.createElement('div');
    other.innerHTML = '<span class="op-tool-refusal" data-reason="writes_disabled"></span>';
    document.body.append(other);
    await new Promise((resolve) => setTimeout(resolve, 5));
    expect(flagReads()).toHaveLength(1);

    const held = document.createElement('div');
    held.innerHTML = '<span class="op-tool-refusal" data-reason="session_held"></span>';
    document.body.append(held);

    await vi.waitFor(() => expect(flagReads()).toHaveLength(2));
  });
});

describe('the dialog', () => {
  test('lists every open flag, its prompt as text, its watchdog and its reason', async () => {
    reads = [
      {
        enabled: true,
        flags: [
          flag({ prompt_excerpt: '<img src=x onerror=boom>' }),
          flag({ flag_id: 'f-2', watchdog: 'second', prompt_excerpt: '' }),
        ],
      },
    ];
    await boot();

    const items = [...document.querySelectorAll('.watchdog-alert-flag')];
    expect(items.map((el) => /** @type {HTMLElement} */ (el).dataset.flagId)).toEqual([
      'f-1',
      'f-2',
    ]);
    expect(document.querySelector('.watchdog-alert-prompt')?.textContent).toBe(
      '<img src=x onerror=boom>'
    );
    expect(document.querySelector('.watchdog-alert-prompt img')).toBeNull();
    expect(document.querySelectorAll('.watchdog-alert-prompt')).toHaveLength(1);
    expect(items[0].textContent).toContain('beam-guard');
    expect(items[0].textContent).toContain('it moves the beam');
  });

  test('a stop that did not take says so and names what is still running', async () => {
    reads = [{ enabled: true, flags: [flag({ stop_unconfirmed: true, started: ['sleep 60'] })] }];
    await boot();

    const line = document.querySelector('.watchdog-alert-unconfirmed')?.textContent;
    expect(line).toContain('did not stop');
    expect(line).toContain('sleep 60');
  });

  test('no button takes focus when it opens', async () => {
    reads = [{ enabled: true, flags: [flag()] }];
    await boot();

    expect(document.activeElement).toBe(dialog());
  });

  test('Escape, Enter and a scrim click answer nothing', async () => {
    reads = [{ enabled: true, flags: [flag()] }];
    await boot();

    for (const key of ['Escape', 'Enter']) {
      dialog()?.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true }));
    }
    /** @type {HTMLElement} */ (document.querySelector('.watchdog-alert-overlay')).click();
    await settle();

    expect(answers()).toHaveLength(0);
    expect(dialog()).not.toBeNull();
  });

  test('Tab stays between the two buttons', async () => {
    reads = [{ enabled: true, flags: [flag()] }];
    await boot();

    const tab = () =>
      dialog()?.dispatchEvent(new KeyboardEvent('keydown', { key: 'Tab', bubbles: true }));
    tab();
    expect(document.activeElement).toBe(button('.watchdog-alert-stop'));
    tab();
    expect(document.activeElement).toBe(button('.watchdog-alert-continue'));
    tab();
    expect(document.activeElement).toBe(button('.watchdog-alert-stop'));
  });

  test.each([
    ['.watchdog-alert-stop', 'stop'],
    ['.watchdog-alert-continue', 'continue'],
  ])(
    '%s sends %s for the flags on screen, and the dialog closes when none are left',
    async (selector, word) => {
      reads = [
        { enabled: true, flags: [flag(), flag({ flag_id: 'f-2' })] },
        { enabled: true, flags: [] },
      ];
      await boot();

      button(selector).click();
      await settle();

      const [[, init]] = answers();
      expect(init.method).toBe('POST');
      expect(JSON.parse(init.body)).toEqual({
        session_key: 'sess-1',
        flag_ids: ['f-1', 'f-2'],
        answer: word,
      });
      expect(answers()).toHaveLength(1);
      await vi.waitFor(() => expect(dialog()).toBeNull(), { timeout: 2000 });
    }
  );

  test('a flag that joined after the click stays open and keeps the dialog up', async () => {
    reads = [
      { enabled: true, flags: [flag()] },
      { enabled: true, flags: [flag({ flag_id: 'late' })] },
    ];
    await boot();

    button('.watchdog-alert-continue').click();
    await settle();

    expect(JSON.parse(answers()[0][1].body).flag_ids).toEqual(['f-1']);
    expect(dialog()).not.toBeNull();
    expect(
      /** @type {HTMLElement} */ (document.querySelector('.watchdog-alert-flag')).dataset.flagId
    ).toBe('late');
  });

  test('the buttons stay disabled while a stop is in flight, and a second click sends nothing', async () => {
    /** @type {(value: any) => void} */
    let release = () => {};
    const pending = new Promise((resolve) => {
      release = resolve;
    });
    answerReply = () => pending;
    reads = [{ enabled: true, flags: [flag()] }];
    await boot();

    button('.watchdog-alert-stop').click();
    await settle();

    expect(button('.watchdog-alert-stop').disabled).toBe(true);
    expect(button('.watchdog-alert-continue').disabled).toBe(true);
    expect(document.querySelector('.watchdog-alert-note')?.textContent).toContain('Stopping');
    button('.watchdog-alert-continue').click();
    await settle();
    expect(answers()).toHaveLength(1);

    release(ok({ released: [], stop_unconfirmed: ['f-1'] }));
    await settle();
    expect(button('.watchdog-alert-stop').disabled).toBe(false);
  });

  test('an answer that was not taken keeps the flags up, names why, and can be sent again', async () => {
    answerReply = { ok: false, status: 503, json: async () => ({ detail: 'hold record' }) };
    reads = [{ enabled: true, flags: [flag()] }];
    await boot();

    button('.watchdog-alert-stop').click();
    await settle();

    expect(dialog()).not.toBeNull();
    expect(document.querySelector('.watchdog-alert-note')?.textContent).toContain(
      'The answer was not taken: hold record'
    );
    expect(button('.watchdog-alert-stop').disabled).toBe(false);

    button('.watchdog-alert-continue').click();
    await settle();
    expect(answers()).toHaveLength(2);
  });
});

describe('the unavailable indicator', () => {
  /** @param {string} watchdog @param {string} status @param {string} [key] */
  const statusFrame = (watchdog, status, key = 'sess-1') => ({
    type: 'watchdog_status',
    session_key: key,
    watchdog,
    status,
  });

  test('names each watchdog of this session, in its own word, without opening the dialog', async () => {
    await boot();

    alert.onWatchdogFrame(statusFrame('one', 'unavailable'));
    alert.onWatchdogFrame(statusFrame('two', 'flag_write_failed'));
    alert.onWatchdogFrame(statusFrame('other-session', 'unavailable', 'sess-2'));
    await settle();

    const lines = [...document.querySelectorAll('.watchdog-status-text')].map(
      (el) => el.textContent
    );
    expect(lines).toEqual(['Watchdog one: unavailable', 'Watchdog two: flag not recorded']);
    expect(dialog()).toBeNull();
  });

  test('dismissing the last line removes the indicator', async () => {
    await boot();
    alert.onWatchdogFrame(statusFrame('one', 'unavailable'));
    await settle();

    button('.watchdog-status-dismiss').click();

    expect(document.querySelector('.watchdog-status')).toBeNull();
  });

  test('a frame missing its names changes nothing on screen', async () => {
    await boot();

    alert.onWatchdogFrame({ type: 'watchdog_status', status: 'unavailable' });
    await settle();

    expect(document.querySelector('.watchdog-status')).toBeNull();
  });
});
