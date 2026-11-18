/**
 * Unit tests for the knob picker of the set-up (knob-picker.js): which hook a
 * switch, a window field, Clear and the preset select call; what the filter
 * lists; what is kept across a redraw (the filter text, the open window, the
 * open "add" list); a preset no list offers stays selected; a knob a refused
 * save names shows its reason and opens its window; the value now shows only
 * when it was read.
 *
 * The harness redraws the picker on a structural change, as the set-up does.
 */

import { describe, test, expect, vi } from 'vitest';

import { envDetails, storedConfig } from './_fixtures.mjs';
import { choose, type } from './_views.mjs';
import {
  formFromConfig,
  formFromEnv,
} from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/draft-model.js';
import { createKnobPicker } from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/knob-picker.js';

/**
 * A picker on the form, in the document, redrawn on each structural change.
 *
 * @param {any} env
 * @param {{ form?: any, presets?: any, values?: Record<string, number|null>,
 *   issues?: Record<string, string[]> }} [options]
 */
function pickerFor(env, { form = null, presets = {}, values = {}, issues = {} } = {}) {
  const f = form ?? formFromEnv(env);
  const onChange = vi.fn((/** @type {boolean} */ structural) => {
    if (structural) draw();
  });
  const onPreset = vi.fn();
  const picker = createKnobPicker({ onChange, onPreset });
  const draw = () => picker.render({ form: f, env, presets, values, issues });
  document.body.replaceChildren(picker.el);
  draw();
  return { el: picker.el, form: f, onChange, onPreset, draw };
}

/** @param {ParentNode} root @param {string} id */
const click = (root, id) => /** @type {HTMLElement} */ (root.querySelector(`#${id}`)).click();
/** @param {ParentNode} root */
const inRun = (root) =>
  Array.from(root.querySelectorAll('.opt-pick.is-in'), (p) => p.getAttribute('data-knob'));
/** @param {ParentNode} root @param {string} id */
const toggle = (root, id, on = true) => {
  const input = /** @type {HTMLInputElement} */ (root.querySelector(`#${id}`));
  input.checked = on;
  input.dispatchEvent(new Event('change'));
};

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

describe('the knob picker', () => {
  test('with 49 knobs: clear, filter, add one, edit its window', () => {
    const t = pickerFor(bigEnv());
    const picker = t.el;
    expect(picker.querySelectorAll('.opt-pick.is-in')).toHaveLength(49);
    click(picker, 'opt-su-knob-clear');
    expect(picker.querySelectorAll('.opt-pick.is-in')).toHaveLength(0);

    const filter = /** @type {HTMLInputElement} */ (picker.querySelector('#opt-su-knob-filter'));
    type(filter, 'k4');
    const offered = [...picker.querySelectorAll('.opt-pick-more .opt-pick')].map((p) =>
      p.getAttribute('data-knob')
    );
    expect(offered).toEqual(['k40', 'k41', 'k42', 'k43', 'k44', 'k45', 'k46', 'k47', 'k48']);

    const add = /** @type {HTMLInputElement} */ (
      picker.querySelector('.opt-pick[data-knob="k42"] input')
    );
    add.checked = true;
    add.dispatchEvent(new Event('change'));
    expect(
      [...picker.querySelectorAll('.opt-pick.is-in')].map((p) => p.getAttribute('data-knob'))
    ).toEqual(['k42']);
    // The filter is kept across the redraw.
    expect(
      /** @type {HTMLInputElement} */ (picker.querySelector('#opt-su-knob-filter')).value
    ).toBe('k4');

    click(picker, 'opt-su-knob-42-window');
    const opts = /** @type {HTMLElement} */ (picker.querySelector('#opt-su-knob-42-opts'));
    expect(opts.hidden).toBe(false);
    type(opts.querySelector('#opt-su-knob-42-min'), '2');
    type(opts.querySelector('#opt-su-knob-42-max'), '4');
    expect(t.form.vars[42]).toMatchObject({ name: 'k42', include: true, min: 2, max: 4 });
  });

  test('a switch takes a knob out of the run or adds it, and asks for a redraw', () => {
    const t = pickerFor(envDetails());
    toggle(t.el, 'opt-su-knob-0', false);
    expect(t.form.vars[0].include).toBe(false);
    expect(t.onChange).toHaveBeenLastCalledWith(true);
    expect(inRun(t.el)).toEqual(['sol_b']);
    toggle(t.el, 'opt-su-knob-0', true);
    expect(t.form.vars[0].include).toBe(true);
    expect(inRun(t.el)).toEqual(['phase_a', 'sol_b']);
  });

  test('a typed window goes into the form and its row, with no redraw', () => {
    const t = pickerFor(envDetails());
    click(t.el, 'opt-su-knob-1-window');
    const min = /** @type {HTMLInputElement} */ (t.el.querySelector('#opt-su-knob-1-min'));
    t.onChange.mockClear();
    type(min, '2.5');
    type(t.el.querySelector('#opt-su-knob-1-step'), '');
    expect(t.form.vars[1]).toMatchObject({ name: 'sol_b', min: 2.5, step: null });
    expect(t.onChange.mock.calls).toEqual([[false], [false]]);
    // The field typed in is the same node: nothing was redrawn under it.
    expect(t.el.querySelector('#opt-su-knob-1-min')).toBe(min);
    expect(t.el.querySelector('.opt-pick[data-knob="sol_b"] .opt-pick-sub')?.textContent).toContain(
      '2.5'
    );
  });

  test('Clear is held when no knob is in the run', () => {
    const t = pickerFor(envDetails());
    const clear = /** @type {HTMLButtonElement} */ (t.el.querySelector('#opt-su-knob-clear'));
    expect(clear.disabled).toBe(false);
    clear.click();
    expect(t.form.vars.every((/** @type {any} */ v) => !v.include)).toBe(true);
    expect(clear.disabled).toBe(true);
  });

  test("the filter matches a knob's description as well as its name", () => {
    const t = pickerFor(envDetails());
    type(t.el.querySelector('#opt-su-knob-filter'), 'buncher');
    expect(inRun(t.el)).toEqual(['phase_a']);
    type(t.el.querySelector('#opt-su-knob-filter'), 'sol');
    expect(inRun(t.el)).toEqual(['sol_b']);
  });

  test('picking a preset hands its name to the set-up, which seeds the form', () => {
    const presets = { sols: { variables: ['sol_b'] } };
    const t = pickerFor(envDetails(), { presets });
    choose(t.el.querySelector('#opt-su-preset'), 'sols');
    expect(t.onPreset).toHaveBeenCalledWith('sols');
    expect(t.form.preset).toBe('');
    expect(t.onChange).not.toHaveBeenCalled();
  });

  test('a preset the environment no longer offers stays selected, as the value a save sends', () => {
    const env = envDetails();
    const config = storedConfig({ objective_direction: 'MAXIMIZE', preset: 'sols' });
    const t = pickerFor(env, { form: formFromConfig(config, /** @type {any} */ (env)) });
    expect(/** @type {HTMLSelectElement} */ (t.el.querySelector('#opt-su-preset')).value).toBe(
      'sols'
    );
  });

  test('a knob a refused save names shows the reason at its row and opens its window', () => {
    const t = pickerFor(envDetails(), { issues: { sol_b: ['outside the safe range'] } });
    const row = /** @type {HTMLElement} */ (t.el.querySelector('.opt-pick[data-knob="sol_b"]'));
    expect(row.classList.contains('is-err')).toBe(true);
    expect(row.textContent).toContain('outside the safe range');
    expect(/** @type {HTMLElement} */ (t.el.querySelector('#opt-su-knob-1-opts')).hidden).toBe(
      false
    );
    expect(/** @type {HTMLElement} */ (t.el.querySelector('#opt-su-knob-0-opts')).hidden).toBe(
      true
    );
  });

  test('a window the operator opened stays open when a refused save names another knob', () => {
    const env = envDetails();
    const form = formFromEnv(/** @type {any} */ (env));
    const picker = createKnobPicker({ onChange: vi.fn(), onPreset: vi.fn() });
    document.body.replaceChildren(picker.el);
    picker.render({ form, env: /** @type {any} */ (env), presets: {}, values: {}, issues: {} });
    click(picker.el, 'opt-su-knob-0-window');
    picker.render({
      form,
      env: /** @type {any} */ (env),
      presets: {},
      values: {},
      issues: { sol_b: ['outside the safe range'] },
    });
    const hidden = (/** @type {string} */ id) =>
      /** @type {HTMLElement} */ (picker.el.querySelector(`#${id}`)).hidden;
    expect(hidden('opt-su-knob-0-opts')).toBe(false);
    expect(hidden('opt-su-knob-1-opts')).toBe(true);
  });

  test('the open window and the open list of knobs to add are kept across a redraw', () => {
    const t = pickerFor(envDetails());
    toggle(t.el, 'opt-su-knob-1', false);
    click(t.el, 'opt-su-knob-0-window');
    const more = /** @type {HTMLDetailsElement} */ (t.el.querySelector('.opt-pick-more'));
    expect(more.open).toBe(false);
    more.open = true;
    more.dispatchEvent(new Event('toggle'));
    t.draw();
    expect(/** @type {HTMLElement} */ (t.el.querySelector('#opt-su-knob-0-opts')).hidden).toBe(
      false
    );
    expect(/** @type {HTMLDetailsElement} */ (t.el.querySelector('.opt-pick-more')).open).toBe(
      true
    );
  });

  test('the value now of a knob shows in its window only when it was read', () => {
    const detail = (/** @type {Record<string, number|null>} */ values) => {
      const t = pickerFor(envDetails(), { values });
      return Array.from(t.el.querySelectorAll('.opt-pick-opts .opt-raw'), (n) => n.textContent);
    };
    const unread = detail({});
    const read = detail({ phase_a: 1.75, sol_b: null });
    expect(read[0]).toContain('1.75');
    expect(unread[0]).not.toContain('1.75');
    // A value that was not read changes nothing.
    expect(read[1]).toBe(unread[1]);
  });
});
