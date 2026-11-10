// @ts-check
/**
 * The draft form: preset → variables table → run settings → Create / Update.
 *
 * The form is the operator's EDITOR, not the record: what Start runs is the
 * stored draft the approval surface shows. Creating or updating sends the
 * form through the relay; the service resolves it (bounds clamped to the
 * safety range, steps materialized) and the approval surface re-reads it.
 * Inputs mutate the {@link FormState} in place (draft-model.js holds its
 * logic), so a re-render never loses what the operator typed.
 *
 * @module draft-form
 */

import { h, fmt, replaceChildren } from './hyperscript.js';
import { ALGORITHMS, rangePercent } from './approval-model.js';
import { ON_COMPLETE_OPTIONS, parseNumber } from './draft-model.js';

/**
 * @typedef {import('./approval-model.js').EnvironmentDetails} EnvironmentDetails
 * @typedef {import('./draft-model.js').FormState} FormState
 * @typedef {import('./draft-model.js').Preset} Preset
 */

/**
 * @param {string} value
 * @param {readonly (string|[string, string])[]} options
 * @param {(value: string) => void} onchange
 * @param {Record<string, unknown>} [props]
 */
function select(value, options, onchange, props = {}) {
  return h(
    'select',
    {
      ...props,
      onchange: (/** @type {Event} */ e) =>
        onchange(/** @type {HTMLSelectElement} */ (e.target).value),
    },
    ...options.map((o) => {
      const [v, label] = Array.isArray(o) ? o : [o, o];
      return h('option', { value: v, text: label, selected: v === value });
    })
  );
}

/**
 * @param {number|null} value
 * @param {(value: number|null) => void} oninput
 * @param {Record<string, unknown>} [props]
 */
function numberInput(value, oninput, props = {}) {
  return h('input', {
    type: 'number',
    step: 'any',
    ...props,
    value: value == null ? '' : String(value),
    oninput: (/** @type {Event} */ e) =>
      oninput(parseNumber(/** @type {HTMLInputElement} */ (e.target).value)),
  });
}

/**
 * @param {string} label
 * @param {Node} control
 */
function field(label, control) {
  return h('label', { class: 'field' }, h('span', { class: 'field-label', text: label }), control);
}

/**
 * @typedef {{
 *   env: EnvironmentDetails,
 *   presets: Record<string, Preset>,
 *   models: { label: string, path: string }[],
 *   readbacks: Record<string, number|null>,
 *   canCreate: boolean,
 *   canUpdate: boolean,
 *   lockedReason: string|null,
 *   message: { tone: string, text: string }|null,
 * }} FormContext
 *
 * @typedef {{
 *   onPreset: (name: string) => void,
 *   onRefreshReadbacks: () => void,
 *   onLoadDraft: (() => void)|null,
 *   onSubmit: (kind: 'create'|'update') => void,
 * }} FormHandlers
 */

/**
 * Render the form for `form` (mutated in place by the inputs) into `root`.
 *
 * @param {HTMLElement} root
 * @param {FormState} form
 * @param {FormContext} ctx
 * @param {FormHandlers} handlers
 */
export function renderDraftForm(root, form, ctx, handlers) {
  const { env } = ctx;
  const presetNames = Object.keys(ctx.presets || {});
  const observables = Object.keys(env.observables || {});

  const varRows = form.vars.map((v) => {
    const meta = env.variables[v.name];
    const pctCell = h('td', { class: 'num' });
    const updatePct = () => {
      const pct = rangePercent(v.min, v.max, meta?.min_value, meta?.max_value);
      pctCell.textContent = pct == null ? '—' : `${pct.toFixed(1)}%`;
    };
    updatePct();
    return h(
      'tr',
      null,
      h(
        'td',
        null,
        h('input', {
          type: 'checkbox',
          checked: v.include,
          'aria-label': `Tune ${v.name}`,
          onchange: (/** @type {Event} */ e) => {
            v.include = /** @type {HTMLInputElement} */ (e.target).checked;
          },
        })
      ),
      h('td', {
        title: meta?.description || '',
        text: meta?.units ? `${v.name} [${meta.units}]` : v.name,
      }),
      h(
        'td',
        null,
        numberInput(v.min, (n) => ((v.min = n), updatePct()), { 'aria-label': `${v.name} min` })
      ),
      h(
        'td',
        null,
        numberInput(v.max, (n) => ((v.max = n), updatePct()), { 'aria-label': `${v.name} max` })
      ),
      h(
        'td',
        null,
        numberInput(v.step, (n) => (v.step = n), { 'aria-label': `${v.name} step` })
      ),
      h('td', { class: 'num', text: fmt(meta?.max_step) }),
      h('td', { class: 'num', text: fmt(ctx.readbacks[v.name]) }),
      pctCell
    );
  });

  const constraintList = h('div', { class: 'constraints' });
  const renderConstraints = () => {
    replaceChildren(
      constraintList,
      ...form.constraints.map((c, i) =>
        h(
          'div',
          { class: 'constraint-row' },
          select(
            c.observable,
            [['', '—'], ...observables.map((o) => /** @type {[string,string]} */ ([o, o]))],
            (val) => (c.observable = val),
            { 'aria-label': 'Constraint observable' }
          ),
          select(
            c.type,
            [
              ['GREATER_THAN', '>'],
              ['LESS_THAN', '<'],
            ],
            (val) => (c.type = val),
            { 'aria-label': 'Constraint relation' }
          ),
          numberInput(c.value, (n) => (c.value = n), { 'aria-label': 'Constraint threshold' }),
          h('button', {
            type: 'button',
            class: 'osprey-btn',
            text: 'Remove',
            onclick: () => {
              form.constraints.splice(i, 1);
              renderConstraints();
            },
          })
        )
      ),
      h('button', {
        type: 'button',
        class: 'osprey-btn',
        text: 'Add constraint',
        onclick: () => {
          form.constraints.push({ observable: '', type: 'GREATER_THAN', value: null });
          renderConstraints();
        },
      })
    );
  };
  renderConstraints();

  const locked = !!ctx.lockedReason;
  replaceChildren(
    root,
    h(
      'div',
      { class: 'card-head' },
      h('h2', { text: 'Draft' }),
      handlers.onLoadDraft
        ? h('button', {
            type: 'button',
            class: 'osprey-btn',
            text: 'Load stored draft',
            onclick: handlers.onLoadDraft,
          })
        : null
    ),
    h(
      'div',
      { class: 'form-grid' },
      field(
        'Preset',
        select(
          form.preset,
          [
            ['', 'No preset'],
            ...presetNames.map(
              (n) => /** @type {[string,string]} */ ([n, ctx.presets[n].display_name || n])
            ),
          ],
          handlers.onPreset,
          { id: 'preset-select' }
        )
      ),
      field(
        'Objective',
        select(
          form.objective,
          [
            ['', '—'],
            ...(env.available_objectives || observables).map(
              (o) => /** @type {[string,string]} */ ([o, o])
            ),
          ],
          (val) => (form.objective = val),
          { id: 'objective-select' }
        )
      ),
      field(
        'Direction',
        select(
          form.direction,
          [['', 'Observable default'], 'MAXIMIZE', 'MINIMIZE'],
          (val) => (form.direction = val)
        )
      ),
      field(
        'Algorithm',
        select(form.algorithm, ALGORITHMS, (val) => (form.algorithm = val), {
          id: 'algorithm-select',
        })
      ),
      field(
        'Iterations',
        numberInput(form.nIterations, (n) => (form.nIterations = n), {
          min: '0',
          step: '1',
          id: 'iterations-input',
        })
      ),
      field(
        'Initial samples',
        numberInput(form.nInitial, (n) => (form.nInitial = n), {
          min: '1',
          step: '1',
          placeholder: 'auto',
        })
      ),
      field(
        'On complete',
        select(form.onComplete, ON_COMPLETE_OPTIONS, (val) => (form.onComplete = val))
      ),
      field(
        'Model file',
        select(
          form.modelFile,
          [
            ['', 'None'],
            ...ctx.models.map((m) => /** @type {[string,string]} */ ([m.path, m.label || m.path])),
          ],
          (val) => (form.modelFile = val)
        )
      )
    ),
    h(
      'div',
      { class: 'table-tools' },
      h('button', {
        type: 'button',
        class: 'osprey-btn',
        text: 'Refresh readbacks',
        onclick: handlers.onRefreshReadbacks,
      })
    ),
    h(
      'div',
      { class: 'table-scroll' },
      h(
        'table',
        { class: 'grid form-vars' },
        h(
          'thead',
          null,
          h(
            'tr',
            null,
            ...['', 'Variable', 'Min', 'Max', 'Step', 'Max step', 'Readback', 'Range %'].map((t) =>
              h('th', { text: t })
            )
          )
        ),
        h('tbody', null, ...varRows)
      )
    ),
    h('div', { class: 'field-label', text: 'Constraints' }),
    constraintList,
    field(
      'Algorithm params (JSON)',
      h('textarea', {
        rows: 2,
        value: form.algorithmParams,
        placeholder: '{}',
        oninput: (/** @type {Event} */ e) =>
          (form.algorithmParams = /** @type {HTMLTextAreaElement} */ (e.target).value),
      })
    ),
    ctx.message
      ? h('div', {
          class: `banner banner-${ctx.message.tone}`,
          role: 'status',
          text: ctx.message.text,
        })
      : null,
    h(
      'div',
      { class: 'foot-row' },
      h('span', {
        class: 'muted',
        text: ctx.lockedReason || 'Saving a draft moves nothing; Start does.',
      }),
      h('button', {
        type: 'button',
        class: 'osprey-btn',
        id: 'create-draft-btn',
        text: 'Create new draft',
        disabled: locked || !ctx.canCreate,
        onclick: () => handlers.onSubmit('create'),
      }),
      h('button', {
        type: 'button',
        class: 'osprey-btn primary',
        id: 'update-draft-btn',
        text: 'Update draft',
        disabled: locked || !ctx.canUpdate,
        onclick: () => handlers.onSubmit('update'),
      })
    )
  );
}
