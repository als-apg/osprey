// @ts-check
/**
 * The knob picker of the set-up: a preset to start from, one filter over
 * every knob of the environment, the knobs in the run (each with its search
 * window and step behind a disclosure) and the other knobs, to add.
 *
 * The picker edits the form of draft-model.js in place: a switch includes or
 * leaves out a knob, the window fields set its bounds and step. It keeps its
 * own filter text, its open window and its open "add" list across redraws, so
 * an environment of fifty knobs stays one filter and a short list away.
 *
 * @module knob-picker
 */

import { icon } from '/design-system/js/icons.js';
import { parseNumber } from './draft-model.js';
import { fmt, h, replaceChildren } from './hyperscript.js';
import { knobLabel } from './knob-bar.js';
import { knobMatches, withUnlisted } from './setup-model.js';

/**
 * @typedef {import('./draft-model.js').FormState} FormState
 * @typedef {import('./draft-model.js').FormVar} FormVar
 * @typedef {import('./approval-model.js').EnvironmentDetails} EnvironmentDetails
 * @typedef {import('./approval-model.js').VariableMetadata} VariableMetadata
 * @typedef {{
 *   form: FormState,
 *   env: EnvironmentDetails,
 *   presets: Record<string, import('./draft-model.js').Preset>,
 *   values: Record<string, number|null>,
 *   issues: Record<string, string[]>,
 * }} PickerContext
 * @typedef {{
 *   onChange: (structural: boolean) => void,
 *   onPreset: (name: string) => void,
 * }} PickerHooks
 */

/**
 * @param {FormVar} v
 * @param {VariableMetadata|undefined} meta
 */
function windowText(v, meta) {
  const unit = meta?.units ? ` ${meta.units}` : '';
  return `${v.min == null ? '…' : fmt(v.min)} to ${v.max == null ? '…' : fmt(v.max)}${unit}`;
}

/**
 * @param {PickerHooks} hooks
 */
export function createKnobPicker(hooks) {
  let query = '';
  /** The knob whose window fields are open. */
  let open = /** @type {string|null} */ (null);
  /** Whether the list of knobs to add is open (with no filter text). */
  let adding = false;
  /** @type {PickerContext|null} */
  let ctx = null;

  const presetHint = h('span', { class: 'opt-raw' });
  const presetSelect = /** @type {HTMLSelectElement} */ (
    h('select', {
      class: 'osprey-field-input',
      id: 'opt-su-preset',
      'aria-label': 'Start from a preset',
      onchange: () => hooks.onPreset(presetSelect.value),
    })
  );
  const filter = /** @type {HTMLInputElement} */ (
    h('input', {
      type: 'search',
      id: 'opt-su-knob-filter',
      autocomplete: 'off',
      'aria-label': 'Filter the knobs by name or channel',
      oninput: () => {
        query = filter.value;
        fill();
      },
    })
  );
  const clear = /** @type {HTMLButtonElement} */ (
    h('button', {
      type: 'button',
      class: 'osprey-btn ghost sm',
      id: 'opt-su-knob-clear',
      text: 'Clear',
      title: 'Take every knob out of the run',
      onclick: () => {
        for (const v of ctx?.form.vars || []) v.include = false;
        hooks.onChange(true);
      },
    })
  );
  const list = h('div', { class: 'opt-pick-list' });
  const el = h(
    'div',
    { class: 'opt-picker' },
    h(
      'div',
      { class: 'osprey-field', 'data-field': 'preset' },
      h(
        'div',
        { class: 'osprey-field-label' },
        h('span', { class: 'osprey-field-name', text: 'Start from' })
      ),
      presetSelect,
      presetHint
    ),
    h(
      'div',
      { class: 'opt-filter-row' },
      h('label', { class: 'osprey-search-field', 'data-expanded': '' }, icon('search'), filter),
      clear
    ),
    list
  );

  /**
   * @param {FormVar} v
   * @param {number} i
   * @param {boolean} inRun
   */
  function row(v, i, inRun) {
    if (!ctx) return null;
    const meta = ctx.env.variables[v.name];
    const { title, channel } = knobLabel(v.name, meta);
    const id = `opt-su-knob-${i}`;
    const issues = ctx.issues[v.name] || [];
    const sub = h('span', {
      class: 'opt-pick-sub',
      title: channel || v.name,
      text: inRun
        ? windowText(v, meta)
        : channel || `safe ${fmt(meta?.min_value)} to ${fmt(meta?.max_value)}`,
    });
    const isOpen = inRun && open === v.name;
    /** @param {'min'|'max'|'step'} key @param {string} label */
    const number = (key, label) =>
      h(
        'div',
        { class: 'osprey-field' },
        h(
          'div',
          { class: 'osprey-field-label' },
          h('span', { class: 'osprey-field-name', text: label }),
          key !== 'step' && meta?.units
            ? h('span', { class: 'osprey-field-unit', text: meta.units })
            : null
        ),
        h('input', {
          class: 'osprey-field-input sm',
          id: `${id}-${key}`,
          type: 'text',
          inputmode: 'decimal',
          value: v[key] == null ? '' : String(v[key]),
          'aria-label': `${title}: ${label.toLowerCase()}`,
          oninput: (/** @type {Event} */ e) => {
            v[key] = parseNumber(/** @type {HTMLInputElement} */ (e.target).value);
            sub.textContent = windowText(v, meta);
            hooks.onChange(false);
          },
        })
      );
    const now = ctx.values[v.name];
    const opts = inRun
      ? h(
          'div',
          { class: 'opt-pick-opts', id: `${id}-opts`, hidden: !isOpen },
          h(
            'div',
            { class: 'osprey-form-grid', style: '--cols: 3' },
            number('min', 'From'),
            number('max', 'To'),
            number('step', 'Step')
          ),
          h('span', {
            class: 'opt-raw',
            text: [
              v.name,
              `safe ${fmt(meta?.min_value)} to ${fmt(meta?.max_value)}${meta?.units ? ` ${meta.units}` : ''}`,
              meta?.max_step != null ? `at most ${fmt(meta.max_step)} per write` : null,
              typeof now === 'number' ? `now ${fmt(now)}` : null,
            ]
              .filter(Boolean)
              .join(' · '),
          })
        )
      : null;
    return h(
      'div',
      {
        class: `opt-pick${inRun ? ' is-in' : ''}${issues.length ? ' is-err' : ''}`,
        'data-knob': v.name,
      },
      h(
        'div',
        { class: 'opt-pick-row' },
        h(
          'label',
          { class: 'osprey-switch-label', title: inRun ? 'In the run' : 'Add to the run' },
          h('input', {
            class: 'osprey-switch-input',
            type: 'checkbox',
            id,
            checked: inRun,
            'aria-label': `${title} in the run`,
            onchange: (/** @type {Event} */ e) => {
              v.include = /** @type {HTMLInputElement} */ (e.target).checked;
              hooks.onChange(true);
            },
          }),
          h('span', { class: 'osprey-switch' })
        ),
        h(
          'span',
          { class: 'opt-pick-name' },
          h('span', { class: 'opt-pick-label', title, text: title }),
          sub,
          ...issues.map((text) =>
            h('span', { class: 'opt-flag is-err' }, icon('triangle-alert', { size: 12 }), text)
          )
        ),
        inRun
          ? h('button', {
              type: 'button',
              class: 'osprey-disclosure icon',
              id: `${id}-window`,
              'aria-expanded': isOpen ? 'true' : 'false',
              'aria-controls': `${id}-opts`,
              'aria-label': `Window of ${title}`,
              onclick: () => {
                open = isOpen ? null : v.name;
                fill();
                document.getElementById(`${id}-window`)?.focus();
              },
            })
          : null
      ),
      opts
    );
  }

  /** Draw the lists for the filter as it stands. */
  function fill() {
    if (!ctx) return;
    const { form, env } = ctx;
    const vars = form.vars.map((v, i) => ({ v, i }));
    const match = (/** @type {FormVar} */ v) => knobMatches(v.name, env.variables[v.name], query);
    const inRun = vars.filter(({ v }) => v.include);
    const shown = inRun.filter(({ v }) => match(v));
    const rest = vars.filter(({ v }) => !v.include && match(v));
    const filtering = query.trim() !== '';
    const more = h(
      'details',
      { class: 'osprey-folder opt-pick-more', open: filtering || adding },
      h(
        'summary',
        null,
        filtering ? 'Other matches' : 'Add a knob',
        h('span', { class: 'osprey-aside', text: String(rest.length) })
      ),
      ...(rest.length
        ? rest.map(({ v, i }) => row(v, i, false))
        : [
            h('div', {
              class: 'osprey-list-empty',
              text: filtering ? 'No other knob matches.' : 'Every knob is in the run.',
            }),
          ])
    );
    more.addEventListener('toggle', () => {
      if (!filtering) adding = /** @type {HTMLDetailsElement} */ (more).open;
    });
    replaceChildren(
      list,
      h('div', { class: 'osprey-list-head', text: `In this run · ${inRun.length}` }),
      ...(inRun.length
        ? shown.length
          ? shown.map(({ v, i }) => row(v, i, true))
          : [h('div', { class: 'osprey-list-empty', text: 'No knob in the run matches.' })]
        : [
            h('div', {
              class: 'osprey-list-empty',
              text: 'None yet. Add knobs below, or start from a preset.',
            }),
          ]),
      more
    );
  }

  return {
    el,
    /**
     * Draw the picker for the form (after a change of its structure: a knob
     * in or out, a preset, a new seed).
     *
     * @param {PickerContext} next
     */
    render(next) {
      ctx = next;
      const { form, presets } = next;
      const total = form.vars.length;
      filter.placeholder = `Filter ${total} knob${total === 1 ? '' : 's'}`;
      clear.disabled = !form.vars.some((v) => v.include);
      /** @type {[string, string][]} */
      const options = [
        ['', 'No preset'],
        ...Object.entries(presets).map(
          ([name, p]) =>
            /** @type {[string, string]} */ ([
              name,
              `${p.display_name || name}${p.variables ? ` (${p.variables.length})` : ''}`,
            ])
        ),
      ];
      // A preset the environment no longer offers is shown as what a save sends.
      replaceChildren(
        presetSelect,
        ...withUnlisted(options, form.preset).map(([value, text]) =>
          h('option', { value, text, selected: value === form.preset })
        )
      );
      presetHint.textContent = form.preset ? presets[form.preset]?.description || '' : '';
      presetHint.hidden = !presetHint.textContent;
      // A knob that a refused save names opens its window fields.
      const named = form.vars.find((v) => v.include && next.issues[v.name]?.length);
      if (named && open === null) open = named.name;
      fill();
    },
  };
}
