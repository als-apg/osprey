// @ts-check
/**
 * The stage of a draft, in set-up and in review: the sentence that says what
 * the run does to the machine, the objective with its value now, and the
 * knobs, each on the knob bar (its safe range, its search window, its value
 * now). The set-up draws it from the form as it stands, the review from the
 * stored draft; both hand over a config of the same shape.
 *
 * A value the service did not supply is left out: no "now" without a
 * reading, no direction without the environment, no bar without the knob's
 * safe range.
 *
 * @module draft-stage
 */

import { icon } from '/design-system/js/icons.js';
import { fixedFeatureChannels, objectiveDirection, plannedEvaluations } from './approval-model.js';
import { constraintText, observableLabel, targetLabel } from './config-sections.js';
import { fmt, h } from './hyperscript.js';
import { knobBar, knobLegend, knobName } from './knob-bar.js';
import { BEFORE_A_WRITE, directionWord, directionWords, endWords } from './words.js';

/**
 * @typedef {import('./approval-model.js').OptimizationConfig} OptimizationConfig
 * @typedef {import('./approval-model.js').EnvironmentDetails} EnvironmentDetails
 * @typedef {import('./approval-model.js').ModelFileInfo} ModelFileInfo
 * @typedef {import('./config-sections.js').Flag} Flag
 * @typedef {{
 *   name: string,
 *   meta: import('./approval-model.js').VariableMetadata|null,
 *   min: number|null,
 *   max: number|null,
 *   step: number|null,
 *   now: number|null,
 *   flag: Flag|null,
 * }} KnobRow
 */

/** @param {string} text */
const b = (text) => h('b', { text });

/** @param {string} target */
function targetWords(target) {
  return target === 'live'
    ? h('b', { class: 'opt-live' }, icon('radiation', { size: 12 }), ` ${targetLabel(target)}`)
    : b(targetLabel(target));
}

/**
 * What a model file makes the run write besides the tuned knobs: the knobs
 * the model was trained on and the draft does not tune, each written at the
 * value it has when the run starts.
 *
 * @param {OptimizationConfig} config
 * @param {ModelFileInfo[]|null} models
 * @returns {(Node|string)[]}
 */
function fixedParts(config, models) {
  if (!config.model_file) return [];
  const fixed = fixedFeatureChannels(config, models);
  if (!fixed.known) {
    return [
      '. Its model file may name more knobs, which it writes at the value each has when the run starts',
    ];
  }
  const k = fixed.channels.length;
  if (!k) return [];
  return [
    ', and writes ',
    b(`${k} more knob${k === 1 ? '' : 's'}`),
    ' that its model file names, each at the value it has when the run starts',
  ];
}

/**
 * The plain words of what a run of `config` does to the machine.
 *
 * @param {OptimizationConfig} config
 * @param {EnvironmentDetails|null} env
 * @param {ModelFileInfo[]|null} [models]  `GET models`, for the knobs a model file fixes
 * @returns {(Node|string)[]}
 */
export function leadParts(config, env, models = null) {
  const n = (config.variables || []).length;
  const target = config.control_target || '';
  const objective = observableLabel(env, config.objective_name);
  const reads = objective.title ? b(objective.title) : 'the objective';
  const on = target ? [' on the ', targetWords(target)] : [];
  if (config.execution_mode === 'readonly') {
    return ['It reads ', reads, ...on, '. It is read only: every write it tries is refused.'];
  }
  if (!n)
    return ['No knob is in this run yet. Add knobs in the inspector, or start from a preset.'];
  // A blank iteration count (a form being edited) gives no count of evaluations.
  const planned = config.n_iterations == null ? null : plannedEvaluations(config);
  const seek = directionWords(objectiveDirection(config, env).effective)?.seek;
  const looking = seek ? `, looking for the ${seek} value. ` : '. ';
  return [
    'It writes ',
    b(`${n} knob${n === 1 ? '' : 's'}`),
    ...on,
    ', each only inside its search window',
    ...fixedParts(config, models),
    ...(planned == null ? [] : [', for ', b(`${planned} evaluations`)]),
    '. After each write it lets the machine settle and reads ',
    reads,
    looking,
    endWords(config.on_complete).sentence,
  ];
}

/**
 * @param {OptimizationConfig} config
 * @param {EnvironmentDetails|null} env
 * @param {ModelFileInfo[]|null} [models]
 * @returns {HTMLElement}
 */
function leadBlock(config, env, models = null) {
  return h(
    'div',
    { class: 'opt-lead-block' },
    config.control_target === 'live'
      ? h(
          'div',
          { class: 'osprey-banner osprey-banner--err' },
          h(
            'span',
            { class: 'osprey-banner-text' },
            b('Real machine. '),
            'This run writes to the accelerator itself, not to a simulator or a rehearsal.'
          )
        )
      : null,
    h('div', {
      class: 'osprey-section-label opt-flush',
      text: 'What this run does to the machine',
    }),
    h('p', { class: 'opt-lead', 'data-lead': '' }, ...leadParts(config, env, models)),
    h('p', { class: 'opt-lead opt-quiet', text: BEFORE_A_WRITE })
  );
}

/**
 * @param {string} kicker
 * @param {...(Node|null)} children
 */
function cell(kicker, ...children) {
  return h(
    'div',
    { class: 'opt-obj-cell', 'data-cell': kicker.toLowerCase().replace(/\s+/g, '-') },
    h('span', { class: 'osprey-kicker', text: kicker }),
    ...children
  );
}

/**
 * The objective: its name, its value now, the goal and what must hold.
 *
 * @param {OptimizationConfig} config
 * @param {EnvironmentDetails|null} env
 * @param {number|null|undefined} now  the objective's value now; null leaves the cell out
 * @returns {HTMLElement}
 */
function objectiveCard(config, env, now) {
  const o = observableLabel(env, config.objective_name);
  const { effective } = objectiveDirection(config, env);
  const goal = directionWord(effective);
  const constraints = Object.entries(config.constraints || {});
  return h(
    'div',
    { class: 'osprey-card opt-objective' },
    h(
      'div',
      { class: 'osprey-card-body opt-obj-row' },
      cell(
        'Objective',
        h('span', { class: 'opt-obj-name', text: o.title || 'Not picked yet' }),
        o.raw ? h('span', { class: 'opt-raw', text: o.raw }) : null
      ),
      typeof now === 'number' && Number.isFinite(now)
        ? cell(
            'Now',
            h(
              'span',
              { class: 'opt-big' },
              fmt(now),
              o.units ? h('span', { class: 'opt-big-unit', text: ` ${o.units}` }) : null
            )
          )
        : null,
      goal
        ? cell(
            'Goal',
            h(
              'span',
              { class: `opt-obj-goal${effective === 'MINIMIZE' ? ' is-down' : ''}` },
              icon('up', { size: 14 }),
              goal
            )
          )
        : null,
      h(
        'div',
        { class: 'opt-obj-cell opt-obj-wide', 'data-cell': 'must-hold' },
        h('span', { class: 'osprey-kicker', text: 'Must hold' }),
        h(
          'span',
          { class: 'opt-cons-list' },
          ...(constraints.length
            ? constraints.map(([obs, c]) => h('span', { text: constraintText(env, obs, c) }))
            : [h('span', { class: 'opt-quiet', text: 'No constraint' })])
        )
      )
    )
  );
}

/**
 * The knobs of `config`, with their windows, their safe ranges and their
 * values now.
 *
 * @param {OptimizationConfig} config
 * @param {EnvironmentDetails|null} env
 * @param {Record<string, number|null>} values  what each knob reads now
 * @param {Record<string, Flag>} [flags]  a flag per knob name
 * @returns {KnobRow[]}
 */
export function knobRows(config, env, values, flags = {}) {
  const overrides = config.variable_overrides || {};
  return (config.variables || []).map((name) => {
    const o = overrides[name] || {};
    const now = values[name];
    return {
      name,
      meta: env?.variables?.[name] ?? null,
      min: o.min ?? null,
      max: o.max ?? null,
      step: o.step_size ?? null,
      now: typeof now === 'number' && Number.isFinite(now) ? now : null,
      flag: flags[name] ?? null,
    };
  });
}

/** @param {KnobRow} r */
function knobRow(r) {
  const units = r.meta?.units || '';
  const unit = units ? ` ${units}` : '';
  const windowText = r.min != null && r.max != null ? `${fmt(r.min)} to ${fmt(r.max)}${unit}` : '—';
  const tone = r.flag?.tone === 'err' ? 'problem' : r.flag ? 'clamped' : null;
  const label = [
    `${r.name}: safe range ${fmt(r.meta?.min_value)} to ${fmt(r.meta?.max_value)}`,
    `window ${windowText}`,
    r.now == null ? null : `now ${fmt(r.now)}`,
  ]
    .filter(Boolean)
    .join(', ');
  return h(
    'div',
    {
      class: `opt-knob-row${r.flag ? ` is-${r.flag.tone}` : ''}`,
      'data-knob': r.name,
    },
    h(
      'div',
      { class: 'opt-knob-cell' },
      knobName(r.name, r.meta),
      r.flag
        ? h(
            'span',
            { class: `opt-flag is-${r.flag.tone}` },
            icon('triangle-alert', { size: 12 }),
            r.flag.text
          )
        : null
    ),
    r.meta
      ? knobBar({
          lo: r.meta.min_value,
          hi: r.meta.max_value,
          window: { min: r.min, max: r.max },
          now: r.now,
          tone,
          units,
          label,
        })
      : h('span'),
    h(
      'div',
      { class: 'opt-knob-vals' },
      h('span', { class: 'opt-strong', text: windowText }),
      h('span', { text: `step ${fmt(r.step)}` }),
      r.now == null ? null : h('span', { 'data-now': '', text: `now ${fmt(r.now)}${unit}` })
    )
  );
}

/**
 * The knob table: a head, then one row per knob.
 *
 * @param {KnobRow[]} rows
 * @param {{ tools?: Node|null }} [options]  `tools`: controls beside the title
 * @returns {HTMLElement}
 */
function knobTable(rows, { tools = null } = {}) {
  const anyNow = rows.some((r) => r.now != null);
  return h(
    'div',
    { class: 'opt-knobs' },
    h(
      'div',
      { class: 'osprey-section-label opt-flush opt-knobs-head' },
      `Knobs · ${rows.length}`,
      tools,
      h(
        'span',
        { class: 'osprey-aside' },
        knobLegend(anyNow ? ['track', 'window', 'now'] : ['track', 'window'])
      )
    ),
    rows.length
      ? h(
          'div',
          { class: 'opt-knob-row is-head' },
          h('span', { class: 'osprey-kicker', text: 'Knob' }),
          h('span', { class: 'osprey-kicker', text: 'Safe range and search window' }),
          h('span', { class: 'osprey-kicker opt-right', text: 'Window' })
        )
      : h('div', { class: 'osprey-list-empty', text: 'No knob selected.' }),
    ...rows.map(knobRow)
  );
}

/**
 * The whole stage of a draft: what the run does, the objective and the knobs.
 *
 * @param {{
 *   config: OptimizationConfig,
 *   env: EnvironmentDetails|null,
 *   models: ModelFileInfo[]|null,
 *   values: Record<string, number|null>,
 *   objectiveNow: number|null|undefined,
 *   flags?: Record<string, Flag>,
 *   tools?: Node|null,
 * }} parts  `values`: what each knob reads now; `flags`: a flag per knob name
 * @returns {HTMLElement[]}
 */
export function draftStage({
  config,
  env,
  models,
  values,
  objectiveNow,
  flags = {},
  tools = null,
}) {
  return [
    leadBlock(config, env, models),
    objectiveCard(config, env, objectiveNow),
    knobTable(knobRows(config, env, values, flags), { tools }),
  ];
}
