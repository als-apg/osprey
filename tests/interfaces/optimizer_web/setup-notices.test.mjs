/**
 * Tests for the set-up's notices and small parts (setup-notices.js): which
 * state of the set-up gives which notice, and nothing when nothing holds Save;
 * what each notice's control calls; the knobs a stored draft names that its
 * environment no longer has; a list or an environment that could not be read
 * and the control that reads it again (also in the mounted set-up, which
 * gives them their callbacks); the folders and the stage's empty state.
 */

import { describe, test, expect, vi } from 'vitest';

import { envDetails } from './_fixtures.mjs';
import { flush, mountWith } from './_views.mjs';
import { formFromEnv } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/draft-model.js';
import { writeReadouts } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/setup-form.js';
import { previewConfig } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/setup-model.js';
import {
  emptyStage,
  missingKnobsNotice,
  rereadButton,
  setupFolder,
  setupNotices,
  unreadList,
} from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/setup-notices.js';
import { mountSetupView } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/setup-view.js';
import {
  initialState,
  reduce,
} from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/store.js';

/** @param {any[]} more */
function openNew(target = '', more = /** @type {any[]} */ ([])) {
  return [
    { type: 'poll/job', job: null, at: 0 },
    { type: 'environments', list: [{ name: 'injector', display_name: 'Injector' }], error: null },
    { type: 'models', list: [], error: null },
    { type: 'setup/open', jobId: null, environment: '', target: '' },
    { type: 'setup/where', environment: 'injector', target },
    { type: 'env/loaded', name: 'injector', details: envDetails(), presets: {}, error: null },
    ...more,
  ];
}

/**
 * The store's state after `events`.
 *
 * @param {any[]} events
 */
const stateOf = (events) => events.reduce(reduce, initialState());

/** A set-up on the injector, new or of the stored draft `j1`. */
function setupState(jobId = /** @type {string|null} */ (null), more = /** @type {any[]} */ ([])) {
  return stateOf([
    { type: 'setup/open', jobId, environment: 'injector', target: 'va' },
    { type: 'env/loaded', name: 'injector', details: envDetails(), presets: {}, error: null },
    ...more,
  ]);
}

/**
 * @typedef {Partial<{ hasForm: boolean, unreadable: boolean, moved: boolean,
 *   message: string|null }>} PartsOver
 */

/**
 * The parts the set-up gives, with every callback a mock.
 *
 * @param {PartsOver} [over]
 */
function partsOf(over = {}) {
  return {
    hasForm: true,
    unreadable: false,
    moved: false,
    message: /** @type {string|null} */ (null),
    onSaveNew: vi.fn(),
    onLoadNew: vi.fn(),
    onKeepMine: vi.fn(),
    onReadEnvironment: vi.fn(),
    ...over,
  };
}

/**
 * Put the notices in the document.
 *
 * @param {(HTMLElement|null)[]} notices
 */
function mount(notices) {
  const root = document.createElement('div');
  root.append(.../** @type {HTMLElement[]} */ (notices.filter(Boolean)));
  document.body.replaceChildren(root);
  return root;
}

/** @param {ParentNode} root @param {string} id */
const click = (root, id) => /** @type {HTMLElement} */ (root.querySelector(`#${id}`)).click();
/** @param {ParentNode} root */
const kinds = (root) =>
  Array.from(root.querySelectorAll('[data-notice]'), (n) => n.getAttribute('data-notice'));

describe('the notices of a set-up', () => {
  test('a new set-up that nothing holds shows no notice', () => {
    expect(setupNotices(setupState(), partsOf()).filter(Boolean)).toEqual([]);
  });

  test('a refused save is read out once, as an alert, with every reason', () => {
    const issues = ['Pick a control target.', 'Select at least one variable.'];
    const root = mount(
      setupNotices(setupState(null, [{ type: 'setup/refused', issues }]), partsOf())
    );
    expect(kinds(root)).toEqual(['refused']);
    const alert = root.querySelector('[role="alert"]');
    expect(alert).not.toBeNull();
    for (const text of issues) expect(alert?.textContent).toContain(text);
  });

  test('a stored draft that could not be read is said to hold Save, as an alert', () => {
    const root = mount(setupNotices(setupState('j1'), partsOf({ unreadable: true })));
    expect(kinds(root)).toEqual(['unreadable']);
    expect(root.querySelector('[data-notice="unreadable"]')?.getAttribute('role')).toBe('alert');
  });

  test('a draft changed on the service offers to load the new version or keep the edits', () => {
    const parts = partsOf({ moved: true });
    const root = mount(setupNotices(setupState('j1'), parts));
    expect(kinds(root)).toEqual(['draft-changed']);
    click(root, 'opt-su-load-new');
    expect(parts.onLoadNew).toHaveBeenCalledTimes(1);
    expect(parts.onKeepMine).not.toHaveBeenCalled();
    click(root, 'opt-su-keep-mine');
    expect(parts.onKeepMine).toHaveBeenCalledTimes(1);
    expect(parts.onSaveNew).not.toHaveBeenCalled();
  });

  test('a stored draft offers a save as a new draft, held without a form or while a save runs', () => {
    /** @param {any} state @param {PartsOver} [over] */
    const saveNew = (state, over = {}) => {
      const root = mount(setupNotices(state, partsOf(over)));
      return /** @type {HTMLButtonElement|null} */ (root.querySelector('#opt-su-save-new'));
    };
    expect(saveNew(setupState())).toBeNull();
    const parts = partsOf();
    const root = mount(setupNotices(setupState('j1'), parts));
    click(root, 'opt-su-save-new');
    expect(parts.onSaveNew).toHaveBeenCalledTimes(1);
    expect(saveNew(setupState('j1'))?.disabled).toBe(false);
    expect(saveNew(setupState('j1'), { hasForm: false })?.disabled).toBe(true);
    expect(saveNew(setupState('j1', [{ type: 'setup/saving' }]))?.disabled).toBe(true);
  });

  test('a message is shown only when there is one', () => {
    expect(kinds(mount(setupNotices(setupState(), partsOf({ message: null }))))).toEqual([]);
    const root = mount(setupNotices(setupState(), partsOf({ message: 'windows unread' })));
    expect(kinds(root)).toEqual(['message']);
    expect(root.textContent).toContain('windows unread');
  });

  test('an environment that could not be read is named with its error and read on request', () => {
    const state = stateOf([
      { type: 'setup/open', jobId: null, environment: 'injector', target: 'va' },
      { type: 'env/loaded', name: 'injector', details: null, presets: {}, error: 'timed out' },
    ]);
    const parts = partsOf();
    const root = mount(setupNotices(state, parts));
    expect(root.querySelector('[data-notice="environment"]')?.textContent).toContain('timed out');
    click(root, 'opt-su-env-details-reread');
    expect(parts.onReadEnvironment).toHaveBeenCalledTimes(1);
  });
});

describe('the knobs a stored draft names that its environment no longer has', () => {
  test('give no notice when there are none, else one that names each', () => {
    expect(missingKnobsNotice([])).toBeNull();
    const notice = /** @type {HTMLElement} */ (missingKnobsNotice(['gone_c', 'gone_d']));
    expect(notice.getAttribute('data-notice')).toBe('missing-knobs');
    expect(notice.textContent).toContain('gone_c');
    expect(notice.textContent).toContain('gone_d');
  });
});

describe('a folder of the set-up', () => {
  test('opens as told and reports each toggle', () => {
    const onToggle = vi.fn();
    const content = document.createElement('p');
    const folder = /** @type {HTMLDetailsElement} */ (
      setupFolder('search', 'Search', [content, null], false, onToggle)
    );
    expect(folder.open).toBe(false);
    expect(folder.contains(content)).toBe(true);
    folder.open = true;
    folder.dispatchEvent(new Event('toggle'));
    expect(onToggle).toHaveBeenLastCalledWith(true);
    expect(
      /** @type {HTMLDetailsElement} */ (setupFolder('end', 'End', [], true, vi.fn())).open
    ).toBe(true);
  });

  test('carries the summary aside that the readouts fill for its section', () => {
    const env = /** @type {any} */ (envDetails());
    const form = formFromEnv(env);
    const root = mount([setupFolder('end', 'When it ends', [], true, vi.fn())]);
    writeReadouts(root, form, env, previewConfig(form, { environment: 'injector', target: 'va' }));
    expect(root.querySelector('summary [data-live="aside-end"]')?.textContent).not.toBe('');
  });
});

describe('the controls that read again', () => {
  test('the button that reads the values now again calls back on each click', () => {
    const onclick = vi.fn();
    const button = rereadButton(onclick);
    button.click();
    button.click();
    expect(onclick).toHaveBeenCalledTimes(2);
  });

  test('a list that was read gives nothing; one that was not gives its error and a control that reads it again', () => {
    const onRead = vi.fn();
    expect(unreadList('opt-x-reread', 'The model files', null, onRead)).toBeNull();
    const root = mount([unreadList('opt-x-reread', 'The model files', 'no models', onRead)]);
    expect(root.querySelector('[data-unread="opt-x-reread"]')?.textContent).toContain('no models');
    click(root, 'opt-x-reread');
    expect(onRead).toHaveBeenCalledTimes(1);
    expect(onRead).toHaveBeenCalledWith();
  });
});

describe('the stage of a set-up with no form', () => {
  const setup = /** @type {any} */ ({ environment: 'injector' });

  test('an environment that could not be read gives its error and a control that reads it again', () => {
    const onRead = vi.fn();
    const empty = emptyStage(setup, /** @type {any} */ ({ error: 'timed out' }), 'held', onRead);
    expect(empty.detail).toBe('timed out');
    // The error is what the stage names, before why Save is held.
    expect(empty.message).not.toBe('held');
    empty.action?.onClick();
    expect(onRead).toHaveBeenCalledTimes(1);
  });

  test('why Save is held is what it says; otherwise it offers no action', () => {
    expect(emptyStage(setup, undefined, 'The stored draft could not be read.', vi.fn())).toEqual({
      message: 'The stored draft could not be read.',
    });
    const reading = emptyStage(setup, undefined, null, vi.fn());
    expect(reading.action).toBeUndefined();
    const none = emptyStage(/** @type {any} */ ({ environment: '' }), undefined, 'held', vi.fn());
    expect(none.action).toBeUndefined();
    expect(none.message).not.toBe('held');
  });
});

describe('a list or an environment that could not be read', () => {
  test('a failed list of environments shows at the picker, and its control reads it again', () => {
    const loadCatalog = vi.fn(async () => {});
    const t = mountWith(
      mountSetupView,
      [
        { type: 'poll/job', job: null, at: 0 },
        { type: 'environments', list: null, error: 'the service is down' },
        { type: 'models', list: [], error: null },
        { type: 'setup/open', jobId: null, environment: '', target: '' },
      ],
      { loadCatalog }
    );
    const notice = t.inspector.querySelector('[data-unread="opt-su-env-reread"]');
    expect(notice?.textContent).toContain('the service is down');
    click(t.inspector, 'opt-su-env-reread');
    expect(loadCatalog).toHaveBeenCalledTimes(1);
    // The next read answers: the picker offers the list, the notice goes.
    t.apply({ type: 'environments', list: [{ name: 'injector' }], error: null });
    expect(t.inspector.querySelector('[data-unread]')).toBeNull();
    expect(t.inspector.querySelector('#opt-su-env option[value="injector"]')).not.toBeNull();
  });

  test('a failed list of model files shows at the model file picker', async () => {
    const loadCatalog = vi.fn(async () => {});
    const t = mountWith(
      mountSetupView,
      openNew('va', [{ type: 'models', list: null, error: 'no models' }]),
      { loadCatalog }
    );
    await flush();
    expect(t.inspector.querySelector('[data-unread="opt-su-model-reread"]')?.textContent).toContain(
      'no models'
    );
    click(t.inspector, 'opt-su-model-reread');
    expect(loadCatalog).toHaveBeenCalledTimes(1);
  });

  test('an environment that could not be read is read again on request', () => {
    const t = mountWith(mountSetupView, [
      ...openNew().slice(0, -1),
      { type: 'env/loaded', name: 'injector', details: null, presets: {}, error: 'timed out' },
    ]);
    // Not read again by itself on each update.
    t.apply({ type: 'setup/where', environment: 'injector', target: 'va' });
    expect(t.actions.loadEnvironment).not.toHaveBeenCalled();
    expect(t.inspector.querySelector('[data-notice="environment"]')?.textContent).toContain(
      'timed out'
    );
    click(t.inspector, 'opt-su-env-details-reread');
    expect(t.actions.loadEnvironment).toHaveBeenCalledWith('injector');
    // The stage offers the same.
    const again = /** @type {HTMLElement} */ (t.stage.querySelector('.osprey-state button'));
    again.click();
    expect(t.actions.loadEnvironment).toHaveBeenCalledTimes(2);
  });
});
