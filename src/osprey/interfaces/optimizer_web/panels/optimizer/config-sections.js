// @ts-check
/**
 * The settings of a stored config, read only, by section: where it runs, its
 * goal, its knobs, its search, what happens at its end, and the advanced
 * fields. Every value is the stored value, as the service holds it; the rules
 * of approval-model.js (the direction against the observable's own, the
 * initial samples, the fixed channels of a model) give the flags that draw
 * the eye to what changes what a run does to the machine.
 *
 * The review shows these before Start; the outcome of a run and a stored run
 * show the settings a run had. The names of the method, the end action and
 * the direction come from words.js.
 *
 * @module config-sections
 */

import { fmt, h } from './hyperscript.js';
import {
  fixedFeatureChannels,
  initialSamples,
  objectiveDirection,
  plannedEvaluations,
  variableRows,
} from './approval-model.js';
import { knobLabel } from './knob-bar.js';
import { TARGETS } from './shell.js';
import { STOP_WORDS, directionWord, endWords, methodLabel } from './words.js';

/**
 * @typedef {import('./approval-model.js').OptimizationConfig} OptimizationConfig
 * @typedef {import('./approval-model.js').EnvironmentDetails} EnvironmentDetails
 * @typedef {import('./approval-model.js').ModelFileInfo} ModelFileInfo
 * @typedef {{ tone: 'err'|'warn', text: string }} Flag
 */

/**
 * A knob whose window the service changed when it stored the draft: clamped
 * to the environment's range, or snapped to the knob's step grid.
 */
export const MOVED_WINDOW =
  'The service adjusted this window when it stored the draft (to the safe range or the step grid).';

/**
 * An observable's name: its display name, then its identifier; the
 * identifier alone when it has no display name.
 *
 * @param {EnvironmentDetails|null|undefined} env
 * @param {string|null|undefined} name
 * @returns {{ title: string, raw: string|null, units: string }}
 */
export function observableLabel(env, name) {
  const meta = name ? env?.observables?.[name] : null;
  const title = (meta?.display_name || '').trim();
  return { title: title || name || '', raw: title ? name || null : null, units: meta?.units || '' };
}

/**
 * A control target's word ("Simulator"), or its identifier when the panel
 * does not know it.
 *
 * @param {string|null|undefined} target
 * @returns {string}
 */
export function targetLabel(target) {
  if (!target) return '';
  return TARGETS[/** @type {keyof typeof TARGETS} */ (target)]?.label ?? target;
}

/**
 * A constraint in words: "Charge stays above 1.2 nC".
 *
 * @param {EnvironmentDetails|null|undefined} env
 * @param {string} observable
 * @param {{ type: string, value: number|null }} c
 * @returns {string}
 */
export function constraintText(env, observable, c) {
  const o = observableLabel(env, observable);
  const relation = c.type === 'LESS_THAN' ? 'below' : 'above';
  return `${o.title} stays ${relation} ${fmt(c.value)}${o.units ? ` ${o.units}` : ''}`;
}

/** @param {unknown} value */
function json(value) {
  return JSON.stringify(value, null, 1).replace(/\n\s*/g, ' ');
}

/**
 * @typedef {{
 *   key: string,
 *   label: string,
 *   value: Node|string,
 *   raw?: string|null,
 *   flag?: Flag|null,
 * }} Row
 */

/**
 * Label and value rows. A value may carry its raw identifier (second line)
 * and a flag (third line). A null row is left out.
 *
 * @param {(Row|null)[]} rows
 * @returns {HTMLElement}
 */
export function configFacts(rows) {
  const dl = h('dl', { class: 'opt-cfg-dl' });
  for (const row of rows) {
    if (!row) continue;
    dl.append(
      h('dt', { text: row.label }),
      h(
        'dd',
        { 'data-field': row.key, class: row.flag ? `is-flagged is-${row.flag.tone}` : null },
        h('span', { class: 'opt-cfg-val' }, row.value),
        row.raw ? h('span', { class: 'opt-cfg-raw', text: row.raw }) : null,
        row.flag ? h('span', { class: 'opt-cfg-flag', text: row.flag.text }) : null
      )
    );
  }
  return dl;
}

/**
 * A section of the settings: a kicker, an optional aside, its content.
 *
 * @param {string} key  `data-section`
 * @param {string} title
 * @param {string|null} aside
 * @param {Node} content
 * @returns {HTMLElement}
 */
export function configSection(key, title, aside, content) {
  return h(
    'section',
    { class: 'osprey-detail-section opt-cfg-sec', 'data-section': key },
    h(
      'div',
      { class: 'opt-cfg-head' },
      h('span', { class: 'osprey-kicker', text: title }),
      aside ? h('span', { class: 'opt-cfg-aside', text: aside }) : null
    ),
    content
  );
}

/**
 * @param {OptimizationConfig} config
 * @param {EnvironmentDetails|null} env
 * @param {string[]} moved
 */
function knobList(config, env, moved) {
  const rows = variableRows(config, env);
  if (!rows.length) return h('p', { class: 'opt-note', text: 'No knobs.' });
  return h(
    'div',
    { class: 'opt-cfg-knobs' },
    ...rows.map((r) => {
      const { title, channel } = knobLabel(r.name, env?.variables?.[r.name]);
      const unit = r.units ? ` ${r.units}` : '';
      /** @type {Flag|null} */
      let flag = null;
      if (r.unresolved) flag = { tone: 'err', text: 'The window or the step is not resolved.' };
      else if (moved.includes(r.name)) flag = { tone: 'warn', text: MOVED_WINDOW };
      const meta = [`step ${fmt(r.step)}`];
      if (r.maxStep != null) meta.push(`at most ${fmt(r.maxStep)} per write`);
      if (r.rangePct != null) meta.push(`${r.rangePct.toFixed(1)} % of the safe range`);
      return h(
        'div',
        {
          class: `opt-cfg-knob${flag ? ` is-${flag.tone}` : ''}`,
          'data-knob': r.name,
        },
        h('span', { class: 'opt-cfg-knob-label', title, text: title }),
        h('span', {
          class: 'opt-cfg-knob-range',
          text: r.unresolved ? '—' : `${fmt(r.min)} to ${fmt(r.max)}${unit}`,
        }),
        channel
          ? h('span', { class: 'opt-cfg-raw opt-cfg-clip', title: channel, text: channel })
          : null,
        h('span', { class: 'opt-cfg-knob-meta', text: meta.join(' · ') }),
        flag ? h('span', { class: 'opt-cfg-flag', text: flag.text }) : null
      );
    })
  );
}

/**
 * The settings of `config`, by section.
 *
 * @param {OptimizationConfig} config  the stored config
 * @param {EnvironmentDetails|null} env  its environment's details, when read
 * @param {ModelFileInfo[]|null} models  `GET models`, when read
 * @param {{ environmentLabel?: string|null, moved?: string[] }} [options]
 *   `environmentLabel`: the environment's display name; `moved`: the knobs
 *   whose window the service moved when the draft was saved
 * @returns {HTMLElement[]}
 */
export function configSections(config, env, models, { environmentLabel = null, moved = [] } = {}) {
  const target = config.control_target || '';
  const direction = objectiveDirection(config, env);
  const init = initialSamples(config);
  const planned = plannedEvaluations(config);
  const fixed = fixedFeatureChannels(config, models);
  const objective = observableLabel(env, config.objective_name);
  const constraints = Object.entries(config.constraints || {});
  const end = endWords(config.on_complete);
  const envName = config.environment_name || '';

  /** @type {Flag|null} */
  let directionFlag = null;
  if (!env) directionFlag = { tone: 'warn', text: 'Not verified: the environment is not read.' };
  else if (direction.mismatch)
    directionFlag = {
      tone: 'warn',
      text: `Differs from the observable's own direction (${direction.observableDefault}).`,
    };

  let modelText = 'none';
  /** @type {Flag|null} */
  let modelFlag = null;
  if (config.model_file) {
    modelText = config.model_file;
    if (!fixed.known)
      modelFlag = { tone: 'warn', text: 'Fixed channels unknown: the model is not listed.' };
    else if (fixed.channels.length)
      modelText += ` (fixed at their values now: ${fixed.channels.join(', ')})`;
    else modelText += ' (no fixed channels)';
  }
  const envConfig = config.environment_config || {};

  return [
    configSection(
      'where',
      'Where',
      null,
      configFacts([
        {
          key: 'target',
          label: 'Control target',
          value: target ? targetLabel(target) : 'not set',
          raw: target || null,
          flag:
            target === 'live'
              ? { tone: 'err', text: 'The real machine.' }
              : target
                ? null
                : { tone: 'err', text: 'Required to start.' },
        },
        config.execution_mode
          ? {
              key: 'execution-mode',
              label: 'Execution mode',
              value: config.execution_mode === 'readonly' ? 'Read only' : config.execution_mode,
              raw: config.execution_mode,
              flag:
                config.execution_mode === 'readonly'
                  ? { tone: 'warn', text: 'Every write is refused.' }
                  : null,
            }
          : null,
        {
          key: 'environment',
          label: 'Environment',
          value: environmentLabel || envName || 'not set',
          raw: environmentLabel && environmentLabel !== envName ? envName : null,
        },
        config.preset ? { key: 'preset', label: 'Preset', value: config.preset } : null,
      ])
    ),
    configSection(
      'goal',
      'Goal',
      null,
      configFacts([
        {
          key: 'objective',
          label: 'Objective',
          value: objective.title || 'not set',
          raw: objective.raw,
          flag: config.objective_name ? null : { tone: 'err', text: 'Required.' },
        },
        {
          key: 'direction',
          label: 'Direction',
          value: directionWord(direction.effective) ?? 'unknown',
          // The service writes the objective's own direction into every
          // draft, so a direction is "set in the draft" only against it; with
          // the environment not read, its source is not told.
          raw:
            direction.effective && direction.observableDefault
              ? `${direction.effective} (${direction.mismatch ? 'set in the draft' : "the objective's own"})`
              : direction.effective,
          flag: directionFlag,
        },
        {
          key: 'constraints',
          label: 'Must hold',
          value: constraints.length
            ? h(
                'span',
                { class: 'opt-cfg-stack' },
                ...constraints.map(([obs, c]) => h('span', { text: constraintText(env, obs, c) }))
              )
            : 'None',
        },
      ])
    ),
    configSection(
      'knobs',
      'Knobs',
      `${(config.variables || []).length} · window and step`,
      knobList(config, env, moved)
    ),
    configSection(
      'search',
      'Search',
      null,
      configFacts([
        {
          key: 'method',
          label: 'Method',
          value: methodLabel(config.algorithm),
          raw: config.algorithm || 'random',
        },
        { key: 'iterations', label: 'Iterations', value: String(config.n_iterations ?? 20) },
        {
          key: 'initial',
          label: 'Initial samples',
          value: init.count == null ? 'not known yet' : String(init.count),
          raw: init.derived ? `auto: ${init.rule}` : init.rule,
        },
        planned == null
          ? null
          : { key: 'evaluations', label: 'Evaluations', value: String(planned) },
      ])
    ),
    configSection(
      'end',
      'When it ends',
      null,
      configFacts([
        {
          key: 'on-complete',
          label: 'At the end',
          value: end.short,
          raw: `${config.on_complete || 'apply_best'}${end.sentence ? `: ${end.sentence}` : ''}`,
        },
        {
          key: 'after-a-stop',
          label: 'If cancelled or failed',
          value: STOP_WORDS.short,
        },
      ])
    ),
    configSection(
      'advanced',
      'Advanced',
      null,
      configFacts([
        {
          key: 'algorithm-params',
          label: 'Method settings',
          value: config.algorithm_params ? json(config.algorithm_params) : 'none',
        },
        {
          key: 'surrogate',
          label: 'Surrogate model',
          value: config.surrogate_model || 'default',
        },
        { key: 'model-file', label: 'Model file', value: modelText, flag: modelFlag },
        {
          key: 'environment-config',
          label: 'Environment config',
          value: Object.keys(envConfig).length ? json(envConfig) : 'none',
        },
      ])
    ),
  ];
}
