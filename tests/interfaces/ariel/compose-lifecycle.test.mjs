// @ts-check
/**
 * C-SHEET-LIFECYCLE, pinned with stub hooks and faked timers:
 *
 *   - steps 1–4 (reset, renderFields, the kind fill, the restore) run in order,
 *     each awaiting the previous one;
 *   - a reopen or close while step 2, 3 or 4 is pending leaves the newer open
 *     untouched, and a superseded filler sees `isCurrent()` false;
 *   - programmatic changes in steps 1–4 never save; a user input after step 4
 *     saves once, debounced;
 *   - `close()`, a context switch and `pagehide` flush a pending save;
 *     `finish()` cancels it and clears the key;
 *   - `onSaved` receives false when the save fails;
 *   - the gated controls (Submit, Discard) are disabled until step 4 finishes;
 *   - no stored record holds the auth username or password, or an attachment.
 *
 *   npx vitest run tests/interfaces/ariel/compose-lifecycle.test.mjs
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const STATIC = '../../../src/osprey/interfaces/ariel/static';

const FORM_HTML = `
  <form id="create-entry-form">
    <input name="subject" id="subject">
    <input name="auth_user" id="auth-user">
    <input type="password" name="auth_password" id="auth-password">
    <input type="file" id="entry-files">
    <select data-entry-field="book" id="book"><option value="ops">ops</option></select>
    <div id="entry-tags"></div>
    <button type="submit" id="submit">Submit</button>
    <button type="button" id="discard">Discard</button>
  </form>
`;

/** @template [T=void] */
function deferred() {
  /** @type {(value: T) => void} */
  let resolve = () => {};
  /** @type {Promise<T>} */
  const promise = new Promise((r) => (resolve = r));
  return { promise, resolve };
}

/** @param {string} id @returns {HTMLElement} */
const el = (id) => /** @type {HTMLElement} */ (document.getElementById(id));

/** Let every queued microtask run. */
const settle = () => vi.advanceTimersByTimeAsync(0);

/**
 * A user edit: set the value and fire the bubbling event the browser would.
 * @param {string} id
 * @param {string} [value]
 * @param {'input'|'change'} [type]
 */
function edit(id, value = 'typed', type = 'input') {
  const input = /** @type {HTMLInputElement} */ (el(id));
  if (input.type !== 'file') input.value = value;
  input.dispatchEvent(new Event(type, { bubbles: true }));
}

/** @type {any} */
let autosave;
/** @type {any} */
let lifecycleModule;

/**
 * Stub hooks recording every call into `log`; each overridable.
 * @param {Record<string, any>} [overrides]
 */
function makeHooks(overrides = {}) {
  /** @type {string[]} */
  const log = [];
  const hooks = {
    log,
    reset: vi.fn(() => log.push('reset')),
    renderFields: vi.fn(async () => log.push('render')),
    fillers: /** @type {Record<string, import('vitest').Mock>} */ ({
      follow: vi.fn(async (/** @type {any} */ id) => log.push(`follow:${id}`)),
      draft: vi.fn(async (/** @type {any} */ id) => log.push(`draft:${id}`)),
    }),
    readSnapshot: vi.fn(() => ({
      subject: /** @type {HTMLInputElement} */ (el('subject')).value,
      details: '',
      author: '',
      fields: { book: 'ops' },
      tags: [],
      logbook: null,
      shift: null,
    })),
    applySnapshot: vi.fn(async (/** @type {any} */ s) => log.push(`apply:${s.subject}`)),
    onSaved: vi.fn(),
    ...overrides,
  };
  return hooks;
}

/** @param {any} hooks @param {Record<string, any>} [extra] */
function create(hooks, extra = {}) {
  return lifecycleModule.createComposeLifecycle(hooks, {
    form: el('create-entry-form'),
    gated: [el('submit'), el('discard')],
    signal: controller.signal,
    delay: 500,
    ...extra,
  });
}

/** @type {AbortController} */
let controller;

/** localStorage spies, restored by hand: restoreAllMocks does not undo them. */
/** @type {any[]} */
let storageSpies = [];

/** @param {'setItem'} method */
function spyStorage(method) {
  const spy = vi.spyOn(localStorage, method);
  storageSpies.push(spy);
  return spy;
}

beforeEach(async () => {
  document.body.innerHTML = FORM_HTML;
  document.documentElement.removeAttribute('data-osprey-storage-scope');
  localStorage.clear();
  vi.useFakeTimers();
  vi.resetModules();
  autosave = await import(`${STATIC}/js/compose-autosave.js`);
  lifecycleModule = await import(`${STATIC}/js/compose-lifecycle.js`);
  controller = new AbortController();
  vi.spyOn(console, 'error').mockImplementation(() => {});
});

afterEach(() => {
  for (const spy of storageSpies) spy.mockRestore();
  storageSpies = [];
  controller.abort();
  vi.useRealTimers();
  vi.restoreAllMocks();
});

const NEW = { draft: null };
const FOLLOW = { draft: null, followsUp: 'e-1' };
const DRAFT = { draft: 'draft-0123456789ab' };

describe('open order', () => {
  test.each([
    [NEW, ['reset', 'render', 'apply:saved']],
    [FOLLOW, ['reset', 'render', 'follow:e-1', 'apply:saved']],
    [DRAFT, ['reset', 'render', 'draft:draft-0123456789ab', 'apply:saved']],
  ])('%o runs reset, render, its filler, then the restore', async (compose, order) => {
    autosave.saveCompose(autosave.contextKey(compose), { subject: 'saved' });
    const hooks = makeHooks();
    await create(hooks).open(compose);
    expect(hooks.log).toEqual(order);
  });

  test('each step waits for the previous one to settle', async () => {
    const render = deferred();
    const fill = deferred();
    const apply = deferred();
    autosave.saveCompose('draft:d', { subject: 'saved' });
    const hooks = makeHooks({
      renderFields: vi.fn(() => render.promise),
      applySnapshot: vi.fn(() => apply.promise),
    });
    hooks.fillers.draft = vi.fn(() => fill.promise);
    const life = create(hooks);

    const opened = life.open({ draft: 'd' });
    expect(hooks.reset).toHaveBeenCalledOnce();
    await settle();
    expect(hooks.renderFields).toHaveBeenCalledOnce();
    expect(hooks.fillers.draft).not.toHaveBeenCalled();

    render.resolve();
    await settle();
    expect(hooks.fillers.draft).toHaveBeenCalledOnce();
    expect(hooks.applySnapshot).not.toHaveBeenCalled();

    fill.resolve();
    await settle();
    expect(hooks.applySnapshot).toHaveBeenCalledOnce();
    expect(/** @type {HTMLButtonElement} */ (el('submit')).disabled).toBe(true);

    apply.resolve();
    await opened;
    expect(/** @type {HTMLButtonElement} */ (el('submit')).disabled).toBe(false);
  });

  test('a new entry with no snapshot skips both the filler and the restore', async () => {
    const hooks = makeHooks();
    await create(hooks).open(NEW);
    expect(hooks.log).toEqual(['reset', 'render']);
  });

  test('a throwing step is logged and the sheet still unlocks', async () => {
    const hooks = makeHooks();
    hooks.fillers.draft = vi.fn(async () => {
      throw new Error('boom');
    });
    const life = create(hooks);
    await life.open(DRAFT);
    expect(console.error).toHaveBeenCalled();
    expect(/** @type {HTMLButtonElement} */ (el('discard')).disabled).toBe(false);
    expect(life.isOpen()).toBe(true);
  });
});

describe('a superseded open does nothing', () => {
  /**
   * Hooks whose `step` (render, fill or apply) blocks the first open only.
   * @param {'render'|'fill'|'apply'} step
   */
  function blockedAt(step) {
    const gate = deferred();
    let first = true;
    /** @type {(() => boolean)[]} */
    const seen = [];
    const block = (/** @type {string} */ label = '') => {
      if (label) hooks.log.push(label);
      if (!first) return Promise.resolve();
      first = false;
      return gate.promise;
    };
    /** @type {any} */
    let hooks = null;
    hooks = makeHooks();
    if (step === 'render') hooks.renderFields = vi.fn(block);
    if (step === 'apply') hooks.applySnapshot = vi.fn((s) => block(`apply:${s.subject}`));
    hooks.fillers.draft = vi.fn(async (/** @type {any} */ id, /** @type {any} */ isCurrent) => {
      seen.push(isCurrent);
      hooks.log.push(`draft:${id}`);
      if (step === 'fill') await block();
    });
    return { hooks, gate, seen };
  }

  test.each(['render', 'fill', 'apply'])(
    'a reopen while %s is pending leaves the newer open untouched',
    async (step) => {
      autosave.saveCompose('draft:a', { subject: 'A' });
      autosave.saveCompose('draft:b', { subject: 'B' });
      const { hooks, gate, seen } = blockedAt(/** @type {any} */ (step));
      const life = create(hooks);

      const first = life.open({ draft: 'a' });
      await settle();
      await life.open({ draft: 'b' });
      expect(/** @type {HTMLButtonElement} */ (el('submit')).disabled).toBe(false);
      const before = [...hooks.log];

      gate.resolve();
      await first;
      await settle();

      // Nothing of A runs after B's open: no later fill, restore or unlock.
      expect(hooks.log).toEqual(before);
      expect(hooks.log.filter((/** @type {string} */ e) => e.startsWith('apply:'))).toEqual(
        step === 'apply' ? ['apply:A', 'apply:B'] : ['apply:B']
      );
      expect(hooks.log.at(-1)).toBe('apply:B');
      expect(/** @type {HTMLButtonElement} */ (el('submit')).disabled).toBe(false);
      if (step === 'fill') expect(seen[0]()).toBe(false);
      expect(seen.at(-1)?.()).toBe(true);
    }
  );

  test.each(['render', 'fill', 'apply'])(
    'a close while %s is pending leaves the sheet closed and locked',
    async (step) => {
      autosave.saveCompose('draft:a', { subject: 'A' });
      const { hooks, gate, seen } = blockedAt(/** @type {any} */ (step));
      const life = create(hooks);

      const first = life.open({ draft: 'a' });
      await settle();
      life.close();
      gate.resolve();
      await first;
      await settle();

      expect(life.isOpen()).toBe(false);
      expect(/** @type {HTMLButtonElement} */ (el('submit')).disabled).toBe(true);
      if (step !== 'apply') expect(hooks.log).not.toContain('apply:A');
      expect(hooks.onSaved).not.toHaveBeenCalled();
      if (seen.length) expect(seen[0]()).toBe(false);
    }
  );
});

describe('saving', () => {
  test('programmatic changes in steps 1–4 never save', async () => {
    autosave.saveCompose('new', { subject: 'restored' });
    const spy = spyStorage('setItem');
    const hooks = makeHooks({
      reset: vi.fn(() => edit('subject', 'r', 'input')),
      renderFields: vi.fn(async () => edit('book', 'ops', 'change')),
      applySnapshot: vi.fn(async () => edit('subject', 'restored', 'change')),
    });
    hooks.fillers.follow = vi.fn(async () => edit('subject', 'f', 'input'));
    const life = create(hooks);

    await life.open({ draft: null, followsUp: 'e-1' });
    await vi.advanceTimersByTimeAsync(5000);
    life.close();

    expect(spy).not.toHaveBeenCalled();
    expect(hooks.onSaved).not.toHaveBeenCalled();
  });

  test('user inputs after step 4 save once, after the debounce', async () => {
    const hooks = makeHooks();
    const life = create(hooks);
    await life.open(NEW);

    edit('subject', 'a');
    await vi.advanceTimersByTimeAsync(300);
    edit('subject', 'ab', 'change');
    await vi.advanceTimersByTimeAsync(300);
    expect(hooks.onSaved).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(300);

    expect(hooks.onSaved).toHaveBeenCalledOnce();
    const [ok, savedAt] = hooks.onSaved.mock.calls[0];
    expect(ok).toBe(true);
    expect(typeof savedAt).toBe('number');
    expect(autosave.loadCompose('new').subject).toBe('ab');
  });

  test('a tag-list change and a declared field change both save', async () => {
    const hooks = makeHooks();
    await create(hooks).open(NEW);
    el('entry-tags').dispatchEvent(new Event('change', { bubbles: true }));
    await vi.advanceTimersByTimeAsync(600);
    edit('book', 'ops', 'change');
    await vi.advanceTimersByTimeAsync(600);
    expect(hooks.onSaved).toHaveBeenCalledTimes(2);
  });

  test.each(['auth-user', 'auth-password', 'entry-files'])(
    'editing %s never schedules a save',
    async (id) => {
      const hooks = makeHooks();
      const life = create(hooks);
      await life.open(NEW);
      edit(id, 'secret', 'input');
      edit(id, 'secret', 'change');
      await vi.advanceTimersByTimeAsync(5000);
      life.close();
      expect(hooks.onSaved).not.toHaveBeenCalled();
    }
  );

  test('onSaved receives false when the save fails', async () => {
    spyStorage('setItem').mockImplementation(() => {
      throw new DOMException('blocked', 'SecurityError');
    });
    const hooks = makeHooks();
    await create(hooks).open(NEW);
    edit('subject');
    await vi.advanceTimersByTimeAsync(600);
    expect(hooks.onSaved).toHaveBeenCalledOnce();
    expect(hooks.onSaved.mock.calls[0][0]).toBe(false);
  });
});

describe('flush, finish and pagehide', () => {
  /** @param {any} hooks */
  async function openWithPending(hooks, compose = NEW) {
    const life = create(hooks);
    await life.open(compose);
    edit('subject', 'pending');
    expect(hooks.onSaved).not.toHaveBeenCalled();
    return life;
  }

  test('close() flushes the pending save and keeps the snapshot', async () => {
    const hooks = makeHooks();
    const life = await openWithPending(hooks);
    life.close();
    expect(hooks.onSaved).toHaveBeenCalledOnce();
    expect(autosave.loadCompose('new').subject).toBe('pending');
    expect(life.isOpen()).toBe(false);
    await vi.advanceTimersByTimeAsync(5000);
    expect(hooks.onSaved).toHaveBeenCalledOnce();
  });

  test('a context switch flushes the pending save under the old key', async () => {
    const hooks = makeHooks();
    const life = await openWithPending(hooks);
    const next = life.open(FOLLOW);
    expect(autosave.loadCompose('new').subject).toBe('pending');
    expect(autosave.loadCompose('follow:e-1')).toBeNull();
    await next;
    await vi.advanceTimersByTimeAsync(5000);
    expect(hooks.onSaved).toHaveBeenCalledOnce();
  });

  test('pagehide flushes the pending save', async () => {
    const hooks = makeHooks();
    await openWithPending(hooks);
    window.dispatchEvent(new Event('pagehide'));
    expect(hooks.onSaved).toHaveBeenCalledOnce();
    expect(autosave.loadCompose('new').subject).toBe('pending');
  });

  test('finish() cancels the pending save and clears the key', async () => {
    autosave.saveCompose('follow:e-1', { subject: 'older' });
    autosave.saveCompose('new', { subject: 'other context' });
    const hooks = makeHooks();
    const life = await openWithPending(hooks, FOLLOW);
    life.finish();
    await vi.advanceTimersByTimeAsync(5000);
    window.dispatchEvent(new Event('pagehide'));

    expect(hooks.onSaved).not.toHaveBeenCalled();
    expect(autosave.loadCompose('follow:e-1')).toBeNull();
    expect(autosave.loadCompose('new').subject).toBe('other context');
    expect(life.isOpen()).toBe(false);
  });

  test('an aborted signal removes the listeners and drops the pending save', async () => {
    const hooks = makeHooks();
    const life = await openWithPending(hooks);
    controller.abort();
    window.dispatchEvent(new Event('pagehide'));
    await vi.advanceTimersByTimeAsync(5000);
    edit('subject', 'after teardown');
    await vi.advanceTimersByTimeAsync(5000);
    expect(hooks.onSaved).not.toHaveBeenCalled();
    expect(autosave.loadCompose('new')).toBeNull();
    expect(life.isOpen()).toBe(false);
  });
});

describe('gating', () => {
  test('Submit and Discard are disabled and the form inert until step 4 finishes', async () => {
    const apply = deferred();
    autosave.saveCompose('new', { subject: 'saved' });
    const hooks = makeHooks({ applySnapshot: vi.fn(() => apply.promise) });
    const life = create(hooks);
    const gated = /** @type {HTMLButtonElement[]} */ ([el('submit'), el('discard')]);

    const opened = life.open(NEW);
    await settle();
    for (const b of gated) expect(b.disabled).toBe(true);
    expect(el('create-entry-form').hasAttribute('inert')).toBe(true);
    expect(el('create-entry-form').getAttribute('aria-busy')).toBe('true');

    apply.resolve();
    await opened;
    for (const b of gated) expect(b.disabled).toBe(false);
    expect(el('create-entry-form').hasAttribute('inert')).toBe(false);
    expect(el('create-entry-form').getAttribute('aria-busy')).toBe('false');
  });
});

describe('secrets', () => {
  test('no stored record holds the auth username or password, or an attachment', async () => {
    const hooks = makeHooks({
      readSnapshot: vi.fn(() => ({
        subject: 's',
        details: '',
        author: '',
        fields: {
          book: 'ops',
          auth_user: 'kirk',
          auth_password: 'hunter2',
          attachments: ['plot.png'],
          files: ['plot.png'],
        },
        tags: [],
        logbook: null,
        shift: null,
        auth_user: 'kirk',
        auth_password: 'hunter2',
        attachments: ['plot.png'],
      })),
    });
    const life = create(hooks);
    await life.open(NEW);
    /** @type {HTMLInputElement} */ (el('auth-user')).value = 'kirk';
    /** @type {HTMLInputElement} */ (el('auth-password')).value = 'hunter2';
    edit('subject');
    life.close();

    const raw = Object.keys(localStorage)
      .map((k) => localStorage.getItem(k) || '')
      .join('\n');
    expect(raw).toContain('"book"');
    for (const secret of ['kirk', 'hunter2', 'plot.png', 'auth_', 'attachment']) {
      expect(raw).not.toContain(secret);
    }
  });
});
