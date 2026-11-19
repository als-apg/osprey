// @ts-check
/**
 * The fields of the set-up's inspector, besides the knob picker: the goal
 * (environment, control target, objective, direction, constraints), the
 * search (method, initial samples, iterations, and the advanced model file and
 * method settings) and the end action. They edit the form of draft-model.js
 * in place; draft-model.js checks it on save.
 *
 * Every control has a stable id, so the set-up can put the focus back after
 * it redraws them. A reason the last save was refused shows under the field
 * it names.
 *
 * @module setup-form
 */

import { icon } from '/design-system/js/icons.js';
import { wireSegmented } from '/design-system/js/components/segmented.js';
import {
  ALGORITHMS,
  initialSamples,
  objectiveDirection,
  plannedEvaluations,
} from './approval-model.js';
import { observableLabel } from './config-sections.js';
import { ON_COMPLETE_OPTIONS, parseNumber } from './draft-model.js';
import { h } from './hyperscript.js';
import { directionAfter, ownDirection, withUnlisted } from './setup-model.js';
import { TARGETS } from './shell.js';
import { DIRECTIONS, STOP_WORDS, directionWord, endWords, methodLabel } from './words.js';

/**
 * @typedef {import('./draft-model.js').FormState} FormState
 * @typedef {import('./approval-model.js').EnvironmentDetails} EnvironmentDetails
 * @typedef {{
 *   form: FormState|null,
 *   env: EnvironmentDetails|null,
 *   environments: { name: string, display_name?: string, valid?: boolean }[],
 *   environment: string,
 *   target: string,
 *   models: { label?: string, path: string }[],
 *   issues: Record<string, string[]>,
 *   unread: { environments: HTMLElement|null, models: HTMLElement|null },
 * }} FieldsContext
 *   `unread`: a list of environments or of model files that could not be read
 *   (why, and the control that reads it again), shown under its picker
 * @typedef {{
 *   where: (environment: string, target: string) => void,
 *   changed: (structural: boolean) => void,
 * }} FieldsHooks
 */

/**
 * One reason a save was refused, under the field it names.
 *
 * @param {string} text
 * @returns {HTMLElement}
 */
export function issueLine(text) {
  return h(
    'span',
    { class: 'opt-flag is-err opt-field-issue' },
    icon('triangle-alert', { size: 12 }),
    text
  );
}

/**
 * A field: its label line, its control, its identifier line and the reasons
 * a save was refused for it.
 *
 * @param {string} label
 * @param {Node} control
 * @param {{ key: string, unit?: string|null, raw?: string|null, help?: string|null,
 *   issues?: string[], wide?: boolean }} options
 */
export function field(
  label,
  control,
  { key, unit = null, raw = null, help = null, issues = [], wide = false }
) {
  return h(
    'div',
    {
      class: `osprey-field${wide ? ' wide' : ''}${issues.length ? ' is-err' : ''}`,
      'data-field': key,
    },
    h(
      'div',
      { class: 'osprey-field-label' },
      h('span', { class: 'osprey-field-name', text: label }),
      unit ? h('span', { class: 'osprey-field-unit', text: unit }) : null,
      help
        ? h(
            'span',
            {
              class: 'osprey-mini opt-help',
              tabindex: '0',
              'data-tip': label,
              'data-tip-detail': help,
            },
            icon('info')
          )
        : null
    ),
    control,
    raw ? h('span', { class: 'opt-raw', text: raw }) : null,
    ...issues.map(issueLine)
  );
}

/**
 * A segmented control.
 *
 * @param {string} id
 * @param {[string, string, string?][]} options  value, label, tooltip
 * @param {string} value  the pressed value ('' presses none unless an option is '')
 * @param {(value: string) => void} onChange
 * @param {{ label: string, block?: boolean }} props
 */
export function seg(id, options, value, onChange, { label, block = true }) {
  const el = h(
    'div',
    { class: `osprey-seg sm${block ? ' block' : ''}`, role: 'group', 'aria-label': label, id },
    ...options.map(([v, text, tip]) =>
      h('button', {
        type: 'button',
        id: `${id}-${v || 'default'}`,
        'data-value': v,
        'aria-pressed': v === value ? 'true' : 'false',
        title: tip || null,
        text,
      })
    )
  );
  wireSegmented(el, onChange);
  return el;
}

/**
 * A select. A value the options do not hold (a model file the service no
 * longer lists, an objective the environment no longer offers) gets an option
 * of its own and is selected, so the field shows the value a save sends.
 *
 * @param {string} id
 * @param {string} value
 * @param {[string, string, boolean?][]} options  value, label, disabled
 * @param {(value: string) => void} onChange
 * @param {string} label
 */
function select(id, value, options, onChange, label) {
  return h(
    'select',
    {
      class: 'osprey-field-input',
      id,
      'aria-label': label,
      onchange: (/** @type {Event} */ e) =>
        onChange(/** @type {HTMLSelectElement} */ (e.target).value),
    },
    ...withUnlisted(options, value).map(([v, text, disabled]) =>
      h('option', { value: v, text, selected: v === value, disabled: !!disabled })
    )
  );
}

/**
 * @param {string} id
 * @param {number|null} value
 * @param {(value: number|null) => void} onInput
 * @param {Record<string, unknown>} [props]
 */
function numberInput(id, value, onInput, props = {}) {
  return h('input', {
    class: 'osprey-field-input',
    id,
    type: 'text',
    inputmode: 'numeric',
    ...props,
    value: value == null ? '' : String(value),
    oninput: (/** @type {Event} */ e) =>
      onInput(parseNumber(/** @type {HTMLInputElement} */ (e.target).value)),
  });
}

/**
 * Where the run goes: the environment and the control target. No target is
 * picked by default.
 *
 * @param {FieldsContext} c
 * @param {FieldsHooks} hooks
 */
export function whereFields(c, hooks) {
  return [
    field(
      'Environment',
      select(
        'opt-su-env',
        c.environment,
        [
          ['', 'Pick an environment…', true],
          ...c.environments.map(
            (e) =>
              /** @type {[string, string, boolean]} */ ([
                e.name,
                e.display_name || e.name,
                e.valid === false,
              ])
          ),
        ],
        (value) => hooks.where(value, c.target),
        'Environment'
      ),
      { key: 'environment', raw: c.environment || null, issues: c.issues.environment }
    ),
    c.unread.environments,
    field(
      'Control target',
      seg(
        'opt-su-target',
        Object.entries(TARGETS).map(([id, t]) => [id, t.label, t.detail]),
        c.target,
        (value) => hooks.where(c.environment, value),
        { label: 'Control target' }
      ),
      { key: 'target', raw: c.target || 'Not picked yet.', issues: c.issues.target }
    ),
  ];
}

/**
 * The goal: objective, direction and the constraints.
 *
 * @param {FieldsContext} c
 * @param {FieldsHooks} hooks
 */
export function goalFields(c, hooks) {
  const { form, env } = c;
  if (!form || !env) return [];
  const observables = Object.keys(env.observables || {});
  const objectives = env.available_objectives || observables;
  const own = ownDirection(env, form.objective);
  const objective = observableLabel(env, form.objective);
  const constraintRows = form.constraints.map((con, i) => {
    const units = observableLabel(env, con.observable).units;
    return h(
      'div',
      { class: 'opt-cons-edit' },
      select(
        `opt-su-con-${i}-obs`,
        con.observable,
        [
          ['', '—'],
          // The objective is not offered (the check refuses it), unless this
          // row names it already, so the field shows what is stored.
          ...observables
            .filter((o) => o !== form.objective || o === con.observable)
            .map((o) => /** @type {[string, string]} */ ([o, observableLabel(env, o).title])),
        ],
        (value) => {
          con.observable = value;
          hooks.changed(true);
        },
        'Constraint observable'
      ),
      seg(
        `opt-su-con-${i}-rel`,
        [
          ['GREATER_THAN', 'above'],
          ['LESS_THAN', 'below'],
        ],
        con.type,
        (value) => {
          con.type = value;
          hooks.changed(false);
        },
        { label: 'Constraint relation', block: false }
      ),
      h('input', {
        class: 'osprey-field-input sm opt-num',
        id: `opt-su-con-${i}-val`,
        type: 'text',
        inputmode: 'decimal',
        value: con.value == null ? '' : String(con.value),
        'aria-label': `Constraint threshold${units ? ` in ${units}` : ''}`,
        oninput: (/** @type {Event} */ e) => {
          con.value = parseNumber(/** @type {HTMLInputElement} */ (e.target).value);
          hooks.changed(false);
        },
      }),
      h('span', { class: 'osprey-field-unit', text: units }),
      h(
        'button',
        {
          type: 'button',
          class: 'osprey-mini',
          id: `opt-su-con-${i}-rm`,
          'aria-label': 'Remove this constraint',
          title: 'Remove',
          onclick: () => {
            form.constraints.splice(i, 1);
            hooks.changed(true);
          },
        },
        icon('x')
      )
    );
  });
  return [
    field(
      'Objective',
      select(
        'opt-su-objective',
        form.objective,
        [
          ['', 'Pick an objective…', true],
          ...objectives.map((o) => {
            const l = observableLabel(env, o);
            return /** @type {[string, string]} */ ([
              o,
              l.units ? `${l.title} (${l.units})` : l.title,
            ]);
          }),
        ],
        (value) => {
          // A direction that was only the old objective's own does not
          // follow the operator to the new one.
          form.direction = directionAfter(env, form.objective, form.direction, value);
          form.objective = value;
          hooks.changed(true);
        },
        'Objective'
      ),
      { key: 'objective', raw: objective.raw || null, issues: c.issues.objective }
    ),
    field(
      'Direction',
      seg(
        'opt-su-direction',
        [
          [
            '',
            own ? `Its own (${directionWord(own)})` : 'Its own',
            "The objective's own direction",
          ],
          ...Object.entries(DIRECTIONS).map(
            ([id, d]) => /** @type {[string, string]} */ ([id, d.word])
          ),
        ],
        form.direction,
        (value) => {
          form.direction = value;
          hooks.changed(false);
        },
        { label: 'Direction' }
      ),
      { key: 'direction', issues: c.issues.direction }
    ),
    field(
      'Must hold',
      h(
        'div',
        { class: 'opt-cons-edits' },
        ...(constraintRows.length
          ? constraintRows
          : [h('span', { class: 'opt-quiet', text: 'No constraint.' })]),
        h(
          'div',
          null,
          h(
            'button',
            {
              type: 'button',
              class: 'osprey-btn ghost sm',
              id: 'opt-su-con-add',
              onclick: () => {
                form.constraints.push({ observable: '', type: 'GREATER_THAN', value: null });
                hooks.changed(true);
              },
            },
            icon('plus', { size: 14 }),
            'Add constraint'
          )
        )
      ),
      {
        key: 'constraints',
        help: 'A point that breaks a constraint does not count as a best point.',
        issues: c.issues.constraints,
      }
    ),
  ];
}

/**
 * The search: method, initial samples and iterations, then the model file
 * and the method settings under "Advanced".
 *
 * @param {FieldsContext} c
 * @param {FieldsHooks} hooks
 * @param {{ initial: string, advancedOpen: boolean, onAdvanced: (open: boolean) => void }} more
 *   `initial`: the placeholder of the initial samples (the count the service derives)
 */
export function searchFields(c, hooks, more) {
  const { form } = c;
  if (!form) return [];
  const advanced = h(
    'details',
    {
      class: 'osprey-folder opt-sub-folder',
      open: more.advancedOpen || !!(c.issues.model || c.issues.params),
    },
    h(
      'summary',
      null,
      'Advanced',
      h('span', { class: 'osprey-aside', text: form.modelFile ? 'model file' : '' })
    ),
    h(
      'div',
      { class: 'osprey-form-grid', style: '--cols: 1' },
      field(
        'Model file',
        select(
          'opt-su-model',
          form.modelFile,
          [
            ['', 'None'],
            ...c.models.map((m) => /** @type {[string, string]} */ ([m.path, m.label || m.path])),
          ],
          (value) => {
            form.modelFile = value;
            hooks.changed(true);
          },
          'Model file'
        ),
        { key: 'model', issues: c.issues.model }
      ),
      c.unread.models,
      field(
        'Method settings',
        h('textarea', {
          class: 'osprey-field-input opt-textarea',
          id: 'opt-su-params',
          rows: 2,
          value: form.algorithmParams,
          placeholder: '{}',
          'aria-label': 'Method settings (a JSON object)',
          oninput: (/** @type {Event} */ e) => {
            form.algorithmParams = /** @type {HTMLTextAreaElement} */ (e.target).value;
            hooks.changed(false);
          },
        }),
        { key: 'params', unit: 'JSON', issues: c.issues.params }
      )
    )
  );
  advanced.addEventListener('toggle', () =>
    more.onAdvanced(/** @type {HTMLDetailsElement} */ (advanced).open)
  );
  return [
    h(
      'div',
      { class: 'osprey-form-grid', style: '--cols: 2' },
      field(
        'Method',
        select(
          'opt-su-method',
          form.algorithm,
          ALGORITHMS.map((a) => /** @type {[string, string]} */ ([a, methodLabel(a)])),
          (value) => {
            form.algorithm = value;
            hooks.changed(true);
          },
          'Method'
        ),
        { key: 'method', raw: form.algorithm, wide: true, issues: c.issues.method }
      ),
      field(
        'Initial samples',
        numberInput(
          'opt-su-initial',
          form.nInitial,
          (n) => {
            form.nInitial = n;
            hooks.changed(false);
          },
          { placeholder: more.initial, 'aria-label': 'Initial samples' }
        ),
        { key: 'initial', unit: 'points', issues: c.issues.initial }
      ),
      field(
        'Iterations',
        numberInput(
          'opt-su-iterations',
          form.nIterations,
          (n) => {
            form.nIterations = n;
            hooks.changed(false);
          },
          { 'aria-label': 'Iterations' }
        ),
        { key: 'iterations', unit: 'steps', issues: c.issues.iterations }
      ),
      h(
        'div',
        { class: 'osprey-field wide' },
        h('span', { class: 'opt-readout', 'data-live': 'evaluations' })
      )
    ),
    advanced,
  ];
}

/**
 * What the run leaves on the machine at its end.
 *
 * @param {FieldsContext} c
 * @param {FieldsHooks} hooks
 */
export function endFields(c, hooks) {
  const { form } = c;
  if (!form) return [];
  const sentence = h('p', { class: 'opt-text', 'data-live': 'end-sentence' });
  return [
    seg(
      'opt-su-end',
      ON_COMPLETE_OPTIONS.map((id) => [id, endWords(id).short, endWords(id).sentence]),
      form.onComplete,
      (value) => {
        form.onComplete = value;
        hooks.changed(false);
      },
      { label: 'At the end' }
    ),
    sentence,
    h('p', { class: 'opt-text opt-quiet', text: STOP_WORDS.ahead }),
    ...(c.issues.end || []).map(issueLine),
  ];
}

/**
 * Write the figures that follow the form as it is typed: the count of
 * evaluations, the end sentence, and each section's summary.
 *
 * @param {HTMLElement} root  the inspector body the fields are in
 * @param {FormState} form
 * @param {EnvironmentDetails} env
 * @param {import('./approval-model.js').OptimizationConfig} preview  `previewConfig` of the form
 */
export function writeReadouts(root, form, env, preview) {
  /** @param {string} key @param {string} text */
  const write = (key, text) => {
    const node = root.querySelector(`[data-live="${key}"]`);
    if (node && node.textContent !== text) node.textContent = text;
  };
  const planned = form.nIterations == null ? null : plannedEvaluations(preview);
  const init = initialSamples(preview);
  write(
    'evaluations',
    planned == null
      ? ''
      : `${planned} evaluations: ${init.count} initial, then ${form.nIterations} iterations.`
  );
  const direction = directionWord(objectiveDirection(preview, env).effective);
  const objective = observableLabel(env, form.objective).title;
  write('aside-goal', [direction, objective].filter(Boolean).join(' '));
  write('aside-knobs', `${preview.variables?.length ?? 0} of ${form.vars.length}`);
  write(
    'aside-search',
    [methodLabel(form.algorithm), planned].filter((x) => x != null).join(' · ')
  );
  write('aside-end', endWords(form.onComplete).short);
  write('end-sentence', endWords(form.onComplete).sentence);
}
