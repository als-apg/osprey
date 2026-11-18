/**
 * Unit tests for the fields of the set-up's inspector (setup-form.js): a
 * reason a save was refused shows under the field it names and nowhere else;
 * a segmented control reports the value picked; each field writes what is
 * typed or picked into the form, and asks for a redraw only when the fields'
 * structure changes; a value no list offers stays selected; the readouts
 * follow the form and leave out a figure the form does not give.
 */

import { describe, test, expect, vi } from 'vitest';

import { envDetails, storedConfig } from './_fixtures.mjs';
import { choose, type } from './_views.mjs';
import { errorIssues } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/api-client.js';
import { plannedEvaluations } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/approval-model.js';
import { observableLabel } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/config-sections.js';
import {
  formFromConfig,
  formFromEnv,
} from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/draft-model.js';
import { h } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/hyperscript.js';
import {
  endFields,
  field,
  goalFields,
  issueLine,
  searchFields,
  seg,
  whereFields,
  writeReadouts,
} from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/setup-form.js';
import {
  groupIssues,
  previewConfig,
} from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/setup-model.js';
import {
  endWords,
  methodLabel,
} from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/words.js';

const WHERE = { environment: 'injector', target: 'va' };
const MORE = { initial: 'auto', advancedOpen: false, onAdvanced: () => {} };

/**
 * The fields' context as the set-up gives it.
 *
 * @param {any} form
 * @param {any} env
 * @param {Record<string, any>} [over]
 * @returns {any}
 */
function context(form, env, over = {}) {
  return {
    form,
    env,
    environments: [{ name: 'injector', display_name: 'Injector' }],
    environment: 'injector',
    target: 'va',
    models: [],
    issues: {},
    unread: { environments: null, models: null },
    ...over,
  };
}

const spyHooks = () => ({ where: vi.fn(), changed: vi.fn() });

/**
 * Put the fields in the document, as the inspector holds them.
 *
 * @param {(Node|null)[]} nodes
 */
function mount(nodes) {
  const root = document.createElement('div');
  root.append(.../** @type {Node[]} */ (nodes.filter(Boolean)));
  document.body.replaceChildren(root);
  return root;
}

/** @param {ParentNode} root @param {string} id */
const click = (root, id) => /** @type {HTMLElement} */ (root.querySelector(`#${id}`)).click();
/** @param {ParentNode} root @param {string} key */
const issuesAt = (root, key) =>
  Array.from(root.querySelectorAll(`[data-field="${key}"] .opt-field-issue`)).map(
    (n) => n.textContent
  );

describe('a field', () => {
  test('shows one line per reason a save was refused for it, and none without one', () => {
    const control = h('input');
    const plain = field('Iterations', control, { key: 'iterations' });
    expect(plain.contains(control)).toBe(true);
    expect(plain.querySelectorAll('.opt-field-issue')).toHaveLength(0);
    expect(plain.classList.contains('is-err')).toBe(false);

    const refused = field('Iterations', h('input'), { key: 'iterations', issues: ['a', 'b'] });
    expect(Array.from(refused.querySelectorAll('.opt-field-issue'), (n) => n.textContent)).toEqual([
      'a',
      'b',
    ]);
    expect(refused.classList.contains('is-err')).toBe(true);
    expect(issueLine('c').textContent).toBe('c');
  });

  test('leaves out a unit, an identifier or a help that is null', () => {
    const bare = field('Method', h('select'), { key: 'method' });
    expect(bare.querySelector('.osprey-field-unit')).toBeNull();
    expect(bare.querySelector('.opt-raw')).toBeNull();
    expect(bare.querySelector('[data-tip-detail]')).toBeNull();

    const full = field('Method', h('select'), {
      key: 'method',
      unit: 'steps',
      raw: 'random',
      help: 'Why it matters',
    });
    expect(full.querySelector('.osprey-field-unit')?.textContent).toBe('steps');
    expect(full.querySelector('.opt-raw')?.textContent).toBe('random');
    expect(full.querySelector('[data-tip-detail]')?.getAttribute('data-tip-detail')).toBe(
      'Why it matters'
    );
  });
});

describe('a segmented control', () => {
  /** @param {HTMLElement} el */
  const pressed = (el) =>
    Array.from(el.querySelectorAll('[aria-pressed="true"]'), (b) => b.getAttribute('data-value'));

  test('reports the option picked and presses it; the pressed one reports nothing', () => {
    const onChange = vi.fn();
    const el = seg(
      'opt-x',
      [
        ['a', 'A'],
        ['b', 'B'],
      ],
      'a',
      onChange,
      { label: 'X' }
    );
    expect(pressed(el)).toEqual(['a']);
    click(el, 'opt-x-a');
    expect(onChange).not.toHaveBeenCalled();
    click(el, 'opt-x-b');
    expect(onChange).toHaveBeenCalledTimes(1);
    expect(onChange.mock.calls[0][0]).toBe('b');
    expect(pressed(el)).toEqual(['b']);
  });

  test('with an empty value presses none, unless one option is the empty one', () => {
    /** @type {[string, string][]} */
    const targets = [
      ['va', 'Simulator'],
      ['standin', 'Rehearsal'],
    ];
    expect(pressed(seg('opt-t', targets, '', vi.fn(), { label: 'T' }))).toEqual([]);

    /** @type {[string, string][]} */
    const directions = [
      ['', 'Its own'],
      ['MAXIMIZE', 'Maximize'],
    ];
    expect(pressed(seg('opt-d', directions, '', vi.fn(), { label: 'D' }))).toEqual(['']);
    const onChange = vi.fn();
    const el = seg('opt-e', directions, 'MAXIMIZE', onChange, { label: 'E' });
    click(el, 'opt-e-default');
    expect(onChange.mock.calls[0][0]).toBe('');
  });
});

describe('where the run goes', () => {
  test('no control target is pressed until one is picked', () => {
    const root = mount(whereFields(context(null, null, { target: '' }), spyHooks()));
    expect(root.querySelectorAll('#opt-su-target [aria-pressed="true"]')).toHaveLength(0);
  });

  test('picking an environment keeps the target; picking a target keeps the environment', () => {
    const hooks = spyHooks();
    const environments = [{ name: 'injector' }, { name: 'linac' }];
    const root = mount(whereFields(context(null, null, { environments, target: '' }), hooks));
    choose(root.querySelector('#opt-su-env'), 'linac');
    expect(hooks.where).toHaveBeenLastCalledWith('linac', '');
    click(root, 'opt-su-target-va');
    expect(hooks.where).toHaveBeenLastCalledWith('injector', 'va');
    expect(hooks.changed).not.toHaveBeenCalled();
  });

  test('an environment the service marks invalid is listed but cannot be picked', () => {
    const environments = [{ name: 'injector' }, { name: 'broken', valid: false }];
    const root = mount(whereFields(context(null, null, { environments }), spyHooks()));
    const option = (/** @type {string} */ v) =>
      /** @type {HTMLOptionElement} */ (root.querySelector(`#opt-su-env option[value="${v}"]`));
    expect(option('broken').disabled).toBe(true);
    expect(option('injector').disabled).toBe(false);
  });

  test('a list of environments that could not be read shows right under the environment picker', () => {
    const unread = h('div');
    const root = mount(
      whereFields(
        context(null, null, { unread: { environments: unread, models: null } }),
        spyHooks()
      )
    );
    expect(root.querySelector('[data-field="environment"]')?.nextElementSibling).toBe(unread);
    const read = mount(whereFields(context(null, null), spyHooks()));
    expect(
      read
        .querySelector('[data-field="environment"]')
        ?.nextElementSibling?.getAttribute('data-field')
    ).toBe('target');
  });
});

describe('the goal', () => {
  test('nothing is drawn without a form or an environment', () => {
    const env = envDetails();
    const form = formFromEnv(/** @type {any} */ (env));
    expect(goalFields(context(null, env), spyHooks())).toEqual([]);
    expect(goalFields(context(form, null), spyHooks())).toEqual([]);
    expect(searchFields(context(null, env), spyHooks(), MORE)).toEqual([]);
    expect(endFields(context(null, env), spyHooks())).toEqual([]);
  });

  test("a new objective drops a direction that was only the old one's own, and keeps one set against it", () => {
    const env = envDetails();
    const hooks = spyHooks();
    // charge maximises, loss minimises.
    const own = formFromEnv(/** @type {any} */ (env));
    own.direction = 'MAXIMIZE';
    let root = mount(goalFields(context(own, env), hooks));
    choose(root.querySelector('#opt-su-objective'), 'loss');
    expect(own).toMatchObject({ objective: 'loss', direction: '' });
    expect(hooks.changed).toHaveBeenLastCalledWith(true);

    const against = formFromEnv(/** @type {any} */ (env));
    against.direction = 'MINIMIZE';
    root = mount(goalFields(context(against, env), hooks));
    choose(root.querySelector('#opt-su-objective'), 'loss');
    expect(against).toMatchObject({ objective: 'loss', direction: 'MINIMIZE' });
  });

  test('a direction picked goes into the form without a redraw', () => {
    const env = envDetails();
    const form = formFromEnv(/** @type {any} */ (env));
    const hooks = spyHooks();
    const root = mount(goalFields(context(form, env), hooks));
    click(root, 'opt-su-direction-MINIMIZE');
    expect(form.direction).toBe('MINIMIZE');
    expect(hooks.changed).toHaveBeenLastCalledWith(false);
  });

  test('a constraint cannot name the objective, unless it names it already', () => {
    const env = envDetails();
    const form = formFromEnv(/** @type {any} */ (env));
    form.constraints = [
      { observable: '', type: 'GREATER_THAN', value: null },
      { observable: 'charge', type: 'GREATER_THAN', value: 1 },
    ];
    const root = mount(goalFields(context(form, env), spyHooks()));
    const offered = (/** @type {string} */ id) =>
      Array.from(root.querySelectorAll(`#${id} option`), (o) => o.getAttribute('value'));
    expect(offered('opt-su-con-0-obs')).toEqual(['', 'loss', 'plain']);
    expect(offered('opt-su-con-1-obs')).toEqual(['', 'charge', 'loss', 'plain']);
  });

  test('a constraint is added, edited and removed; only adding, removing or renaming redraws', () => {
    const env = envDetails();
    const form = formFromEnv(/** @type {any} */ (env));
    const hooks = spyHooks();
    click(mount(goalFields(context(form, env), hooks)), 'opt-su-con-add');
    expect(form.constraints).toHaveLength(1);
    expect(hooks.changed).toHaveBeenLastCalledWith(true);

    const root = mount(goalFields(context(form, env), hooks));
    choose(root.querySelector('#opt-su-con-0-obs'), 'loss');
    expect(form.constraints[0].observable).toBe('loss');
    expect(hooks.changed).toHaveBeenLastCalledWith(true);
    type(root.querySelector('#opt-su-con-0-val'), '3');
    expect(form.constraints[0].value).toBe(3);
    expect(hooks.changed).toHaveBeenLastCalledWith(false);
    type(root.querySelector('#opt-su-con-0-val'), '');
    expect(form.constraints[0].value).toBeNull();
    click(root, 'opt-su-con-0-rel-LESS_THAN');
    expect(form.constraints[0].type).toBe('LESS_THAN');
    expect(hooks.changed).toHaveBeenLastCalledWith(false);
    click(root, 'opt-su-con-0-rm');
    expect(form.constraints).toEqual([]);
    expect(hooks.changed).toHaveBeenLastCalledWith(true);
  });
});

describe('the search', () => {
  test('typed counts go into the form as numbers, an empty one as null, without a redraw', () => {
    const env = envDetails();
    const form = formFromEnv(/** @type {any} */ (env));
    const hooks = spyHooks();
    const root = mount(searchFields(context(form, env), hooks, { ...MORE, initial: 'auto (3)' }));
    expect(
      /** @type {HTMLInputElement} */ (root.querySelector('#opt-su-initial')).placeholder
    ).toBe('auto (3)');
    type(root.querySelector('#opt-su-iterations'), '12');
    expect(form.nIterations).toBe(12);
    type(root.querySelector('#opt-su-initial'), '3');
    expect(form.nInitial).toBe(3);
    type(root.querySelector('#opt-su-iterations'), '');
    expect(form.nIterations).toBeNull();
    type(root.querySelector('#opt-su-params'), '{"beta": 2}');
    expect(form.algorithmParams).toBe('{"beta": 2}');
    expect(hooks.changed.mock.calls.every(([structural]) => structural === false)).toBe(true);
  });

  test('a new method or model file goes into the form and redraws', () => {
    const env = envDetails();
    const form = formFromEnv(/** @type {any} */ (env));
    const hooks = spyHooks();
    const models = [{ label: 'Model A', path: '/results/models/a.joblib' }];
    const root = mount(searchFields(context(form, env, { models }), hooks, MORE));
    choose(root.querySelector('#opt-su-method'), 'neldermead');
    expect(form.algorithm).toBe('neldermead');
    expect(hooks.changed).toHaveBeenLastCalledWith(true);
    choose(root.querySelector('#opt-su-model'), '/results/models/a.joblib');
    expect(form.modelFile).toBe('/results/models/a.joblib');
    expect(hooks.changed).toHaveBeenLastCalledWith(true);
  });

  test('a model file the service no longer lists stays selected, as the value a save sends', () => {
    const env = envDetails();
    const config = storedConfig({
      objective_direction: 'MAXIMIZE',
      model_file: '/results/models/m.joblib',
    });
    const form = formFromConfig(config, /** @type {any} */ (env));
    const root = mount(searchFields(context(form, env, { models: [] }), spyHooks(), MORE));
    expect(/** @type {HTMLSelectElement} */ (root.querySelector('#opt-su-model')).value).toBe(
      '/results/models/m.joblib'
    );
  });

  test('Advanced opens for a refused model file or method settings, and reports each toggle', () => {
    const env = envDetails();
    const form = formFromEnv(/** @type {any} */ (env));
    /** @param {any} c @param {Partial<typeof MORE>} [more] */
    const advanced = (c, more = {}) =>
      /** @type {HTMLDetailsElement} */ (
        mount(searchFields(c, spyHooks(), { ...MORE, ...more }))
          .querySelector('#opt-su-model')
          ?.closest('details')
      );
    expect(advanced(context(form, env)).open).toBe(false);
    expect(advanced(context(form, env, { issues: { params: ['x'] } })).open).toBe(true);
    expect(advanced(context(form, env, { issues: { model: ['x'] } })).open).toBe(true);
    expect(advanced(context(form, env), { advancedOpen: true }).open).toBe(true);

    const onAdvanced = vi.fn();
    const det = advanced(context(form, env), { onAdvanced });
    det.open = true;
    det.dispatchEvent(new Event('toggle'));
    expect(onAdvanced).toHaveBeenLastCalledWith(true);
  });

  test('a list of model files that could not be read shows right under the model file picker', () => {
    const env = envDetails();
    const form = formFromEnv(/** @type {any} */ (env));
    const unread = h('div');
    const root = mount(
      searchFields(
        context(form, env, { unread: { environments: null, models: unread } }),
        spyHooks(),
        MORE
      )
    );
    expect(root.querySelector('[data-field="model"]')?.nextElementSibling).toBe(unread);
  });

  test('each item of a 422 list shows at the field it names', () => {
    const issues = errorIssues(422, {
      detail: [
        { loc: ['body', 'n_iterations'], msg: 'Input should be greater than 0' },
        { loc: ['body', 'objective_name'], msg: 'Field required' },
      ],
    });
    const env = envDetails();
    const form = formFromEnv(/** @type {any} */ (env));
    // The grouping the set-up gives the fields.
    const c = context(form, env, { issues: groupIssues(issues, Object.keys(env.variables)) });
    const root = mount([
      ...whereFields(c, spyHooks()),
      ...goalFields(c, spyHooks()),
      ...searchFields(c, spyHooks(), MORE),
      ...endFields(c, spyHooks()),
    ]);
    const at = (/** @type {string} */ key) => issuesAt(root, key);
    expect(at('iterations')).toEqual(['n_iterations: Input should be greater than 0']);
    expect(at('objective')).toEqual(['objective_name: Field required']);
  });
});

describe('the end of the run', () => {
  test('the end action picked goes into the form without a redraw', () => {
    const env = envDetails();
    const form = formFromEnv(/** @type {any} */ (env));
    const hooks = spyHooks();
    const root = mount(endFields(context(form, env), hooks));
    click(root, 'opt-su-end-hold');
    expect(form.onComplete).toBe('hold');
    expect(hooks.changed).toHaveBeenLastCalledWith(false);
  });

  test('a reason a save was refused for the end action shows below the choice, and only then', () => {
    const env = envDetails();
    const form = formFromEnv(/** @type {any} */ (env));
    const lines = (/** @type {Record<string, string[]>} */ issues) =>
      Array.from(
        mount(endFields(context(form, env, { issues }), spyHooks())).querySelectorAll(
          '.opt-field-issue'
        ),
        (n) => n.textContent
      );
    expect(lines({})).toEqual([]);
    expect(lines({ end: ['on_complete: not allowed'] })).toEqual(['on_complete: not allowed']);
  });
});

describe('the readouts', () => {
  const SLOTS = [
    'evaluations',
    'aside-goal',
    'aside-knobs',
    'aside-search',
    'aside-end',
    'end-sentence',
  ];
  const slots = () => mount(SLOTS.map((key) => h('span', { 'data-live': key })));
  /** @param {HTMLElement} root @param {string} key */
  const text = (root, key) => root.querySelector(`[data-live="${key}"]`)?.textContent ?? '';

  test('follow the form: the evaluations planned, the knobs in the run, the method and the end', () => {
    const env = /** @type {any} */ (envDetails());
    const form = formFromEnv(env);
    form.vars[1].include = false;
    form.onComplete = 'restore';
    const preview = previewConfig(form, WHERE);
    const root = slots();
    writeReadouts(root, form, env, preview);
    const planned = String(plannedEvaluations(preview));
    expect(text(root, 'evaluations')).toContain(planned);
    expect(text(root, 'evaluations')).toContain(String(form.nIterations));
    expect(text(root, 'aside-goal')).toContain(observableLabel(env, form.objective).title);
    expect(text(root, 'aside-knobs')).toMatch(/\b1\b.*\b2\b/);
    expect(text(root, 'aside-search')).toContain(methodLabel(form.algorithm));
    expect(text(root, 'aside-search')).toContain(planned);
    expect(text(root, 'aside-end')).toBe(endWords('restore').short);
    expect(text(root, 'end-sentence')).toBe(endWords('restore').sentence);
  });

  test('with no iterations the count of evaluations is left out', () => {
    const env = /** @type {any} */ (envDetails());
    const form = formFromEnv(env);
    form.nIterations = null;
    const root = slots();
    writeReadouts(root, form, env, previewConfig(form, WHERE));
    expect(text(root, 'evaluations')).toBe('');
    expect(text(root, 'aside-search')).toBe(methodLabel(form.algorithm));
  });

  test('a readout whose text is unchanged is kept, and a slot the fields lack is skipped', () => {
    const env = /** @type {any} */ (envDetails());
    const form = formFromEnv(env);
    const root = mount([h('span', { 'data-live': 'evaluations' })]);
    writeReadouts(root, form, env, previewConfig(form, WHERE));
    const node = root.querySelector('[data-live="evaluations"]')?.firstChild;
    expect(node).toBeTruthy();
    writeReadouts(root, form, env, previewConfig(form, WHERE));
    expect(root.querySelector('[data-live="evaluations"]')?.firstChild).toBe(node);
  });
});
