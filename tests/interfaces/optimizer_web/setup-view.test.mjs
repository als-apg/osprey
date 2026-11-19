/**
 * Tests for the set-up screens (setup-view.js): what a save sends is the
 * config draft-model.js builds from the same edits; a refused save shows its
 * reasons beside the fields and knobs they name; no control target is picked
 * by default; a draft is saved also while a run is under way; a set-up of a
 * stored draft seeds only from that draft and holds Save when it could not be
 * read or changes on the service; a replaced view does nothing more; the edits
 * survive another screen; the stage follows the values read and the picker.
 *
 * The fields, the knob picker and the notices on their own are tested in
 * setup-form.test.mjs, knob-picker.test.mjs and setup-notices.test.mjs.
 *
 * Stored drafts are built as the service sends them: `objective_direction`
 * is always filled in, with the objective's own direction when nobody chose one.
 */

import { describe, test, expect } from 'vitest';

import { envDetails, storedConfig } from './_fixtures.mjs';
import { choose, flush, mountWith, type } from './_views.mjs';
import {
  buildConfig,
  formFromConfig,
  formFromEnv,
} from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/draft-model.js';
import { mountSetupView } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/setup-view.js';
import { readingKey } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/store.js';

const WHERE = { environment: 'injector', target: 'va' };
const BOUNDS = { phase_a: { min: -10, max: 10 }, sol_b: { min: 1, max: 3 } };

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
 * The events of "Edit the draft" on draft `j1` stored with `config`: the
 * set-up carries the config and revision it was opened with.
 *
 * @param {any} config
 * @param {{ id?: number, presets?: any, models?: any[] }} [options]
 * @returns {any[]}
 */
function editing(config, { id = 1, presets = {}, models = [] } = {}) {
  return [
    { type: 'poll/job', job: { job_id: 'j1', status: 'created' }, at: 0 },
    { type: 'poll/config', jobId: 'j1', revision: 'r1', config },
    { type: 'models', list: models, error: null },
    {
      type: 'setup/open',
      jobId: 'j1',
      environment: 'injector',
      target: 'va',
      from: config,
      revision: 'r1',
      id,
    },
    { type: 'env/loaded', name: 'injector', details: envDetails(), presets, error: null },
  ];
}

/** A promise the test settles, for an answer that is late. */
function deferred() {
  /** @type {(value: any) => void} */
  let settle = () => {};
  const promise = new Promise((resolve) => (settle = resolve));
  return { promise, settle };
}

/** @param {HTMLElement} root @param {string} id */
const click = (root, id) => /** @type {HTMLElement} */ (root.querySelector(`#${id}`)).click();
/** @param {HTMLElement} root @param {string} id */
const disabled = (root, id) =>
  /** @type {HTMLButtonElement} */ (root.querySelector(`#${id}`)).disabled;

describe('saving a set-up', () => {
  test('a new set-up names no target, and a save without one is refused beside the target', () => {
    const t = mountWith(mountSetupView, openNew());
    expect(t.inspector.querySelectorAll('#opt-su-target [aria-pressed="true"]')).toHaveLength(0);
    click(t.inspector, 'opt-su-save');
    expect(t.actions.saveDraft).not.toHaveBeenCalled();
    const issues = t.actions.refuseSetup.mock.calls[0][0];
    expect(issues).toContain('Pick a control target.');

    t.apply({ type: 'setup/refused', issues });
    expect(t.screen().id).toBe('setup-issues');
    expect(t.inspector.querySelector('[data-field="target"] .opt-field-issue')?.textContent).toBe(
      'Pick a control target.'
    );
  });

  test('what Save sends is the config draft-model.js builds from the same edits', async () => {
    const t = mountWith(mountSetupView, openNew('va'), {
      initialBounds: async () => ({ bounds: BOUNDS, error: null }),
    });
    await flush();
    type(t.inspector.querySelector('#opt-su-knob-0-max'), '5');
    type(t.inspector.querySelector('#opt-su-iterations'), '12');
    click(t.inspector, 'opt-su-end-hold');
    click(t.inspector, 'opt-su-save');

    const form = formFromEnv(/** @type {any} */ (envDetails()), { bounds: BOUNDS });
    form.vars[0].max = 5;
    form.nIterations = 12;
    form.onComplete = 'hold';
    expect(t.actions.saveDraft).toHaveBeenCalledWith(buildConfig(form, WHERE).config, 'create');
    // The toolbar's Save saves the same.
    t.primary().onClick();
    expect(t.actions.saveDraft.mock.calls[1]).toEqual(t.actions.saveDraft.mock.calls[0]);
  });

  test('a stored draft is edited in place and saved over itself, or as a new draft', () => {
    const config = storedConfig({
      objective_direction: 'MAXIMIZE',
      constraints: { loss: { type: 'LESS_THAN', value: 3 } },
    });
    const t = mountWith(mountSetupView, editing(config));
    type(t.inspector.querySelector('#opt-su-con-0-val'), '4');
    click(t.inspector, 'opt-su-save');
    const form = formFromConfig(config, /** @type {any} */ (envDetails()));
    form.constraints[0].value = 4;
    expect(t.actions.saveDraft).toHaveBeenLastCalledWith(buildConfig(form, WHERE).config, 'update');
    click(t.inspector, 'opt-su-save-new');
    expect(t.actions.saveDraft.mock.calls.at(-1)?.[1]).toBe('create');
  });

  test('a run started again begins from the config it was run with', () => {
    const from = storedConfig({ n_iterations: 7, algorithm: 'neldermead' });
    const t = mountWith(mountSetupView, [
      { type: 'poll/job', job: null, at: 0 },
      { type: 'setup/open', jobId: null, environment: 'injector', target: 'va', from },
      { type: 'env/loaded', name: 'injector', details: envDetails(), presets: {}, error: null },
    ]);
    click(t.inspector, 'opt-su-save');
    const [config, kind] = t.actions.saveDraft.mock.calls[0];
    expect(kind).toBe('create');
    expect(config).toEqual(
      buildConfig(formFromConfig(from, /** @type {any} */ (envDetails())), WHERE).config
    );
    expect(t.actions.initialBounds).not.toHaveBeenCalled();
  });

  test('a draft is saved also while a run is under way (the service takes it)', async () => {
    const t = mountWith(mountSetupView, [
      ...openNew('va'),
      { type: 'poll/job', job: { job_id: 'run-1', status: 'running' }, at: 1 },
    ]);
    await flush();
    expect(t.screen().id).toBe('setup');
    expect(
      /** @type {HTMLButtonElement} */ (t.inspector.querySelector('#opt-su-save')).disabled
    ).toBe(false);
    expect(t.primary()).toMatchObject({ disabled: false });
    click(t.inspector, 'opt-su-save');
    expect(t.actions.saveDraft).toHaveBeenCalledTimes(1);
  });

  test('a preset takes its knobs and the windows the service gives for it', async () => {
    const presets = { sols: { variables: ['sol_b'], algorithm: 'neldermead', n_iterations: 9 } };
    const t = mountWith(
      mountSetupView,
      openNew('va', [
        { type: 'env/loaded', name: 'injector', details: envDetails(), presets, error: null },
      ]),
      { initialBounds: async () => ({ bounds: BOUNDS, error: null }) }
    );
    await flush();
    choose(t.inspector.querySelector('#opt-su-preset'), 'sols');
    await flush();
    click(t.inspector, 'opt-su-save');
    const [config] = t.actions.saveDraft.mock.calls[0];
    expect(config).toMatchObject({
      preset: 'sols',
      variables: ['sol_b'],
      algorithm: 'neldermead',
      n_iterations: 9,
      variable_overrides: { sol_b: { min: 1, max: 3, step_size: 0.01 } },
    });
  });

  test('a save with no knob is refused in the knobs section, which opens', async () => {
    const t = mountWith(mountSetupView, openNew('va'));
    await flush();
    click(t.inspector, 'opt-su-knob-clear');
    click(t.inspector, 'opt-su-save');
    const issues = t.actions.refuseSetup.mock.calls[0][0];
    expect(issues).toEqual(['Select at least one variable.']);
    t.apply({ type: 'setup/refused', issues });
    const folder = /** @type {HTMLDetailsElement} */ (
      t.inspector.querySelector('[data-folder="knobs"]')
    );
    expect(folder.open).toBe(true);
    expect(folder.querySelector('.opt-field-issue')?.textContent).toBe(
      'Select at least one variable.'
    );
  });

  test("a refusal by the service that names a knob marks that knob's row", () => {
    const t = mountWith(mountSetupView, [
      ...openNew(),
      { type: 'setup/refused', issues: ['variable_overrides.phase_a: outside the safe range'] },
    ]);
    expect(t.stage.querySelector('[data-knob="phase_a"]')?.classList.contains('is-err')).toBe(true);
    expect(t.inspector.querySelector('.opt-pick[data-knob="phase_a"]')?.textContent).toContain(
      'outside the safe range'
    );
  });
});

describe('a set-up of a stored draft', () => {
  test('seeds from the draft it was opened for, also when another draft became current meanwhile', () => {
    const A = storedConfig({
      objective_direction: 'MAXIMIZE',
      n_iterations: 30,
      on_complete: 'hold',
      variables: ['sol_b'],
    });
    const opened = editing(A);
    const t = mountWith(mountSetupView, [
      ...opened.slice(0, -1),
      { type: 'poll/job', job: { job_id: 'j2', status: 'created' }, at: 1 },
      { type: 'poll/config', jobId: 'j2', revision: 'r9', config: storedConfig() },
      opened.at(-1),
    ]);
    click(t.inspector, 'opt-su-save');
    const form = formFromConfig(A, /** @type {any} */ (envDetails()));
    expect(t.actions.saveDraft).toHaveBeenCalledWith(buildConfig(form, WHERE).config, 'update');
  });

  test('with no config read for that draft, it says so and holds Save: no defaults replace it', async () => {
    const t = mountWith(mountSetupView, [
      { type: 'poll/job', job: { job_id: 'j1', status: 'created' }, at: 0 },
      { type: 'setup/open', jobId: 'j1', environment: 'injector', target: 'va', id: 2 },
      { type: 'poll/job', job: { job_id: 'j2', status: 'created' }, at: 1 },
      { type: 'poll/config', jobId: 'j2', revision: 'r9', config: storedConfig() },
      { type: 'env/loaded', name: 'injector', details: envDetails(), presets: {}, error: null },
    ]);
    // Any seed that waits for the service's windows has landed.
    await flush();
    t.apply({ type: 'readings', key: 'unrelated', values: {}, error: null, at: 2 });
    expect(t.inspector.querySelector('[data-notice="unreadable"]')).not.toBeNull();
    expect(disabled(t.inspector, 'opt-su-save')).toBe(true);
    expect(t.primary()).toMatchObject({ disabled: true });
    click(t.inspector, 'opt-su-save');
    t.primary().onClick();
    expect(t.actions.saveDraft).not.toHaveBeenCalled();
  });

  test('a draft changed on the service holds Save until the operator keeps the edits', () => {
    const config = storedConfig({ objective_direction: 'MAXIMIZE', n_iterations: 30 });
    const t = mountWith(mountSetupView, editing(config));
    type(t.inspector.querySelector('#opt-su-iterations'), '12');
    t.apply({
      type: 'poll/config',
      jobId: 'j1',
      revision: 'r2',
      config: storedConfig({ objective_direction: 'MAXIMIZE', n_iterations: 99 }),
    });
    expect(t.inspector.querySelector('[data-notice="draft-changed"]')).not.toBeNull();
    expect(disabled(t.inspector, 'opt-su-save')).toBe(true);
    expect(t.primary()).toMatchObject({ disabled: true });
    t.primary().onClick();
    expect(t.actions.saveDraft).not.toHaveBeenCalled();

    click(t.inspector, 'opt-su-keep-mine');
    expect(disabled(t.inspector, 'opt-su-save')).toBe(false);
    click(t.inspector, 'opt-su-save');
    expect(t.actions.saveDraft.mock.calls[0][0]).toMatchObject({ n_iterations: 12 });
  });

  test('… or loads the new version, which replaces the edits', () => {
    const config = storedConfig({ objective_direction: 'MAXIMIZE', n_iterations: 30 });
    const t = mountWith(mountSetupView, editing(config));
    type(t.inspector.querySelector('#opt-su-iterations'), '12');
    const changed = storedConfig({ objective_direction: 'MAXIMIZE', n_iterations: 99 });
    t.apply({ type: 'poll/config', jobId: 'j1', revision: 'r2', config: changed });
    click(t.inspector, 'opt-su-load-new');
    click(t.inspector, 'opt-su-save');
    const form = formFromConfig(changed, /** @type {any} */ (envDetails()));
    expect(t.actions.saveDraft).toHaveBeenCalledWith(buildConfig(form, WHERE).config, 'update');
  });

  test("a new objective is sought in its own direction, not in the old objective's", () => {
    // charge maximises, loss minimises: the stored MAXIMIZE is only charge's own.
    const t = mountWith(mountSetupView, editing(storedConfig({ objective_direction: 'MAXIMIZE' })));
    choose(t.inspector.querySelector('#opt-su-objective'), 'loss');
    click(t.inspector, 'opt-su-save');
    expect(t.actions.saveDraft.mock.calls[0][0]).toMatchObject({
      objective_name: 'loss',
      objective_direction: null,
    });
  });

  test('a knob the stored draft tunes that its environment no longer has is named in a notice', () => {
    const config = storedConfig({
      objective_direction: 'MAXIMIZE',
      variables: ['phase_a', 'gone_c'],
    });
    // The environment has no gone_c.
    const t = mountWith(mountSetupView, editing(config));
    const notice = t.inspector.querySelector('[data-notice="missing-knobs"]');
    expect(notice?.textContent).toContain('gone_c');
  });
});

describe('a set-up that another screen replaces', () => {
  test('draws and sends nothing when an answer comes after it was replaced', async () => {
    const bounds = deferred();
    const t = mountWith(mountSetupView, openNew('va'), { initialBounds: () => bounds.promise });
    t.setPrimary.mockClear();
    t.view.unmount?.();
    bounds.settle({ bounds: BOUNDS, error: null });
    await flush();
    expect(t.setPrimary).not.toHaveBeenCalled();
    expect(t.inspector.querySelector('.opt-picker')).toBeNull();
    click(t.inspector, 'opt-su-save');
    expect(t.actions.saveDraft).not.toHaveBeenCalled();
  });

  test('keeps the edits for when its set-up shows again, and only for that set-up', () => {
    const config = storedConfig({ objective_direction: 'MAXIMIZE', n_iterations: 30 });
    const first = mountWith(mountSetupView, editing(config, { id: 41 }));
    type(first.inspector.querySelector('#opt-su-iterations'), '12');
    first.view.unmount?.();

    const again = mountWith(mountSetupView, editing(config, { id: 41 }));
    click(again.inspector, 'opt-su-save');
    expect(again.actions.saveDraft.mock.calls[0][0]).toMatchObject({ n_iterations: 12 });
    again.view.unmount?.();

    const other = mountWith(mountSetupView, editing(config, { id: 42 }));
    click(other.inspector, 'opt-su-save');
    expect(other.actions.saveDraft.mock.calls[0][0]).toMatchObject({ n_iterations: 30 });
  });

  test('an edit typed while the initial windows are read is kept', async () => {
    const bounds = deferred();
    const t = mountWith(mountSetupView, openNew(), { initialBounds: () => bounds.promise });
    t.apply({ type: 'setup/where', environment: 'injector', target: 'va' });
    type(t.inspector.querySelector('#opt-su-iterations'), '12');
    bounds.settle({ bounds: BOUNDS, error: null });
    await flush();
    click(t.inspector, 'opt-su-save');
    expect(t.actions.saveDraft.mock.calls[0][0]).toMatchObject({
      n_iterations: 12,
      variable_overrides: { phase_a: { min: -10, max: 10 } },
    });
  });
});

describe('the stage of a set-up', () => {
  test('shows the values now that were read, and leaves out the ones that were not', () => {
    const t = mountWith(mountSetupView, openNew());
    expect(t.actions.read).not.toHaveBeenCalled();
    t.apply({ type: 'setup/where', environment: 'injector', target: 'va' });
    expect(t.actions.read).toHaveBeenCalledWith('variables', 'injector', 'va', [
      'phase_a',
      'sol_b',
    ]);
    expect(t.actions.read).toHaveBeenCalledWith('observables', 'injector', 'va', ['charge']);
    expect(t.stage.querySelector('[data-cell="now"]')).toBeNull();
    t.apply(
      {
        type: 'readings',
        key: readingKey('variables', 'injector', 'va'),
        values: { phase_a: 1.5, sol_b: null },
        error: null,
        at: 1,
      },
      {
        type: 'readings',
        key: readingKey('observables', 'injector', 'va'),
        values: { charge: 2.5 },
        error: null,
        at: 1,
      }
    );
    expect(t.stage.querySelector('[data-knob="phase_a"] [data-mark="now"]')).not.toBeNull();
    expect(t.stage.querySelector('[data-knob="sol_b"] [data-mark="now"]')).toBeNull();
    expect(t.stage.querySelector('[data-cell="now"]')?.textContent).toContain('2.5');
  });
});

describe('the knob picker in a set-up', () => {
  /** An environment of 49 knobs. */
  function bigEnv() {
    const env = envDetails();
    /** @type {Record<string, any>} */
    const variables = {};
    for (let i = 0; i < 49; i += 1) {
      const name = `k${String(i).padStart(2, '0')}`;
      variables[name] = {
        min_value: 0,
        max_value: 10,
        step_size: 0.1,
        description: `Corrector ${i}`,
      };
    }
    return { ...env, variables };
  }

  test('the stage follows the picker: one knob added after a clear, then its window typed', () => {
    const t = mountWith(mountSetupView, [
      ...openNew('', [
        { type: 'env/loaded', name: 'injector', details: bigEnv(), presets: {}, error: null },
      ]),
    ]);
    const picker = /** @type {HTMLElement} */ (t.inspector.querySelector('.opt-picker'));
    expect(t.stage.querySelectorAll('.opt-knob-row[data-knob]')).toHaveLength(49);
    click(picker, 'opt-su-knob-clear');
    const add = /** @type {HTMLInputElement} */ (
      picker.querySelector('.opt-pick[data-knob="k42"] input')
    );
    add.checked = true;
    add.dispatchEvent(new Event('change'));
    expect(t.stage.querySelectorAll('.opt-knob-row[data-knob]')).toHaveLength(1);

    click(picker, 'opt-su-knob-42-window');
    const opts = /** @type {HTMLElement} */ (picker.querySelector('#opt-su-knob-42-opts'));
    type(opts.querySelector('#opt-su-knob-42-min'), '2');
    type(opts.querySelector('#opt-su-knob-42-max'), '4');
    expect(t.stage.querySelector('[data-knob="k42"] .opt-knob-vals')?.textContent).toContain(
      '2 to 4'
    );
  });
});
