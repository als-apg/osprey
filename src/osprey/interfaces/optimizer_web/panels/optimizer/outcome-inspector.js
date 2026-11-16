// @ts-check
/**
 * The inspector of an ended run and of a past run: the service's reasons for
 * the end, the objective before the run and at its best, what the run left on
 * the machine and what changed there per knob, the run's facts, its settings,
 * and "Run again".
 *
 * "Run again" opens a new set-up from this run's stored config; it writes
 * nothing. Without a stored config it is disabled and says why. There is no
 * Restore: nothing restores by itself, and this panel offers no restore.
 *
 * @module outcome-inspector
 */

import { configFacts, configSection, configSections, targetLabel } from './config-sections.js';
import { fmt, h } from './hyperscript.js';
import { changeRows } from './outcome-knobs.js';
import { knobChanges, pointOf } from './outcome-model.js';
import { changeText, methodLabel } from './words.js';

/**
 * @typedef {import('./approval-model.js').OptimizationConfig} OptimizationConfig
 * @typedef {import('./approval-model.js').EnvironmentDetails} EnvironmentDetails
 * @typedef {import('./approval-model.js').ModelFileInfo} ModelFileInfo
 * @typedef {import('./run-model.js').RunFigures} RunFigures
 * @typedef {import('./outcome-model.js').Outcome} Outcome
 *
 * What the outcome screens show, gathered from the screen and the state.
 * `applied`: the stored run's `best_point` (the row an `apply_best` run
 * wrote); `bounds`: the search windows the run used (`variable_bounds` of
 * its job), when the service sent them.
 * @typedef {{
 *   key: string,
 *   stored: boolean,
 *   runId: string|null,
 *   info: any,
 *   entry: import('./store.js').PastRun|null,
 *   job: any,
 *   status: string|null,
 *   config: OptimizationConfig|null,
 *   figures: RunFigures|null,
 *   records: Record<string, unknown>[],
 *   variables: string[],
 *   environment: string|null,
 *   started: { day: string, clock: string }|null,
 *   applied: Record<string, unknown>|null,
 *   bounds: Record<string, number[]>|null,
 * }} Source
 *
 * @typedef {{ title: string, units: string }} ObjectiveName
 */

/** Why "Run again" is held without a config. */
export const NO_CONFIG_WHY = 'The settings of this run were not read.';

/** @param {string} units */
const unit = (units) => (units ? `\u00a0${units}` : '');

/**
 * The objective before the run and at its best.
 *
 * @param {RunFigures|null} f
 * @param {ObjectiveName} objective
 * @returns {HTMLElement|null}
 */
function outcomeCard(f, objective) {
  if (!f || (f.best === null && f.before === null)) return null;
  const big =
    f.before !== null && f.best !== null
      ? `${fmt(f.before)} → ${fmt(f.best)}`
      : fmt(f.best ?? f.before);
  const sub =
    f.best === null
      ? 'before the run; no evaluation recorded'
      : [
          changeText(f),
          f.bestIndex === null ? null : `best at evaluation ${f.bestIndex + 1} of ${f.done}`,
        ]
          .filter(Boolean)
          .join(' · ');
  return h(
    'div',
    { class: 'osprey-card opt-out-card', 'data-part': 'outcome' },
    h(
      'div',
      { class: 'osprey-card-body' },
      h('span', { class: 'osprey-kicker', text: objective.title || 'Objective' }),
      h(
        'div',
        { class: 'opt-out-big' },
        big,
        objective.units
          ? h('span', { class: 'opt-out-big-unit', text: unit(objective.units) })
          : null
      ),
      sub ? h('span', { class: 'opt-out-quiet', text: sub }) : null
    )
  );
}

/**
 * What changed on the machine, knob by knob; for a stopped run the values of
 * its last evaluation recorded, said as such.
 *
 * @param {Source} source
 * @param {Outcome} outcome
 * @param {EnvironmentDetails|null} env
 * @returns {HTMLElement|null}
 */
function changesSection(source, outcome, env) {
  const { machine } = outcome;
  const stopped = outcome.kind === 'failed' || outcome.kind === 'cancelled';
  // A stopped run whose writes are not known to stand shows no knob values.
  if (stopped && machine.written !== true) return null;
  const before = pointOf(source.figures?.initial, source.variables);
  const after = stopped ? pointOf(source.figures?.lastRecord, source.variables) : machine.values;
  const changes = knobChanges(source.variables, before, after);
  if (!changes.length) return null;
  const n = `${changes.length} knob${changes.length === 1 ? '' : 's'}`;
  if (!stopped) {
    return configSection('changes', 'What changed on the machine', n, changeRows(changes, env));
  }
  return configSection(
    'changes',
    'At the last evaluation recorded',
    n,
    h(
      'div',
      { class: 'opt-out-stack' },
      changeRows(changes, env),
      h('p', {
        class: 'opt-note',
        text: "A stop inside an evaluation keeps that evaluation's write, so the machine may hold a later point.",
      })
    )
  );
}

/**
 * The run's facts that the service supplies.
 *
 * @param {Source} source
 * @param {Outcome} outcome
 */
function runSection(source, outcome) {
  const f = source.figures;
  const config = source.config;
  const jobId = source.job?.job_id || source.info?.job_id || null;
  return configSection(
    'run',
    'Run',
    null,
    configFacts([
      source.stored
        ? {
            key: 'status',
            label: 'Status',
            value: source.status ? outcome.word : 'not known',
            raw: source.status,
          }
        : null,
      source.started
        ? {
            key: 'started',
            label: 'Started',
            value: `${source.started.day} ${source.started.clock}`,
          }
        : null,
      f
        ? {
            key: 'evaluations',
            label: 'Evaluations',
            value: f.total == null ? String(f.done) : `${f.done} of ${f.total}`,
          }
        : null,
      config
        ? {
            key: 'method',
            label: 'Method',
            value: methodLabel(config.algorithm),
            raw: config.algorithm || null,
          }
        : null,
      config?.control_target
        ? {
            key: 'target',
            label: 'Control target',
            value: targetLabel(config.control_target),
            raw: config.control_target,
          }
        : null,
      jobId ? { key: 'job', label: 'Job', value: String(jobId).slice(0, 8), raw: jobId } : null,
      source.runId ? { key: 'run-id', label: 'Stored run', value: source.runId } : null,
    ])
  );
}

/**
 * The inspector's parts for an outcome.
 *
 * @param {{
 *   source: Source,
 *   outcome: Outcome,
 *   objective: ObjectiveName,
 *   env: EnvironmentDetails|null,
 *   models: ModelFileInfo[]|null,
 *   environmentLabel: string|null,
 *   onRunAgain: () => void,
 * }} parts
 * @returns {import('./shell.js').InspectorParts}
 */
export function outcomeInspector({
  source,
  outcome,
  objective,
  env,
  models,
  environmentLabel,
  onRunAgain,
}) {
  const { config } = source;
  const when = source.started ? `${source.started.day} ${source.started.clock}` : source.runId;
  return {
    title: outcome.kind === 'open' ? 'Run' : source.stored ? 'Past run' : 'Outcome',
    desc: source.stored ? when : outcome.word.toLowerCase(),
    body: [
      ...outcome.reasons.map((text) =>
        h(
          'div',
          { class: 'osprey-banner osprey-banner--err', 'data-part': 'reason' },
          h('span', { class: 'osprey-banner-text', text })
        )
      ),
      outcomeCard(source.figures, objective),
      // A run whose end is not known: the stage says so; nothing to add here.
      outcome.kind === 'unknown' || outcome.kind === 'open'
        ? null
        : h(
            'div',
            {
              class: `osprey-banner osprey-banner--${outcome.machine.tone}`,
              'data-part': 'machine',
            },
            h(
              'span',
              { class: 'osprey-banner-text' },
              h('b', { text: 'Left on the machine. ' }),
              outcome.machine.text
            )
          ),
      changesSection(source, outcome, env),
      runSection(source, outcome),
      config
        ? h(
            'details',
            { class: 'osprey-folder opt-out-settings', 'data-part': 'settings' },
            h('summary', { text: 'Settings of this run' }),
            ...configSections(config, env, models, { environmentLabel })
          )
        : null,
    ],
    foot: h('button', {
      type: 'button',
      class: 'osprey-btn primary block',
      id: 'run-again-btn',
      text: 'Run again',
      disabled: !config,
      title: config ? 'Set up a new run from these settings' : NO_CONFIG_WHY,
      'data-tip-detail': config ? 'Opens a new set-up from these settings. Writes nothing.' : null,
      onclick: onRunAgain,
    }),
  };
}
