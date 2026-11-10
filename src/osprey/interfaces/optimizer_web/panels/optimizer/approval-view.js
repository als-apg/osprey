// @ts-check
/**
 * The approval surface: the stored draft, rendered field by field, with the
 * Start button underneath it.
 *
 * It renders only from the approval state (the stored config plus the
 * revision header of the response it came from), so what the operator reads
 * here is what Start runs. Flags draw the eye to the fields that change what
 * the run does to the machine: a live target, a direction that differs from
 * the observable's default (or a direction not yet verified, while the
 * draft's environment loads), an unresolved variable.
 *
 * Start never moves under the cursor: the review notice keeps its place once
 * acknowledged (it reads "Reviewed" instead), the review button's slot is
 * held, invisible, whenever the operator does not need it, and the owner line
 * (which arrives with `relay/session`, possibly late) sits beside Start in its
 * row rather than above it. Review only applies
 * to a CREATED draft; an ended job's card shows neither.
 *
 * @module approval-view
 */

import { h, fmt, replaceChildren } from './hyperscript.js';
import {
  REVIEWED_NOTICE,
  fixedFeatureChannels,
  initialSamples,
  objectiveDirection,
  variableRows,
} from './approval-model.js';

/**
 * @typedef {import('./approval-model.js').ApprovalState} ApprovalState
 * @typedef {import('./approval-model.js').EnvironmentDetails} EnvironmentDetails
 * @typedef {import('./approval-model.js').ModelFileInfo} ModelFileInfo
 */

/** What `on_complete` leaves on the machine, in the operator's words. */
export const ON_COMPLETE_TEXT = Object.freeze({
  apply_best: 'apply the best point found',
  restore: 'restore the starting setpoints',
  hold: 'hold the last point tried',
});

/**
 * @param {string} label
 * @param {Node|string} value
 * @param {string|null} [flag]
 */
function row(label, value, flag = null) {
  return h(
    'div',
    { class: flag ? 'appr-row flagged' : 'appr-row' },
    h('dt', { text: label }),
    h('dd', null, value, flag ? h('span', { class: 'flag', text: flag }) : null)
  );
}

/**
 * @param {unknown} value
 * @returns {string}
 */
function json(value) {
  return JSON.stringify(value, null, 1).replace(/\n\s*/g, ' ');
}

/**
 * Build the definition list of every approved field.
 *
 * @param {import('./approval-model.js').OptimizationConfig} config
 * @param {EnvironmentDetails|null} env
 * @param {ModelFileInfo[]|null} models
 * @returns {HTMLElement}
 */
export function approvalFields(config, env, models) {
  const target = config.control_target || '';
  const direction = objectiveDirection(config, env);
  const init = initialSamples(config);
  const fixed = fixedFeatureChannels(config, models);
  const constraints = Object.entries(config.constraints || {});
  const onComplete = config.on_complete || 'apply_best';

  const directionText = direction.overridden
    ? `${direction.effective} (set in the draft)`
    : `${direction.effective ?? 'unknown'} (observable default)`;
  /** @type {string|null} */
  let directionFlag = null;
  if (!env) directionFlag = 'direction unverified';
  else if (direction.mismatch)
    directionFlag = `differs from the observable's ${direction.observableDefault}`;
  const envConfig = config.environment_config || {};

  /** @type {string} */
  let modelText = 'none';
  if (config.model_file) {
    modelText = config.model_file;
    if (!fixed.known) modelText += ' — fixed channels unknown (model not listed)';
    else if (fixed.channels.length)
      modelText += ` — fixed at current values: ${fixed.channels.join(', ')}`;
    else modelText += ' — no fixed channels';
  }

  return h(
    'dl',
    { class: 'appr-fields' },
    row(
      'Control target',
      target || 'not set',
      target === 'live' ? 'real machine' : target ? null : 'required to start'
    ),
    config.execution_mode ? row('Execution mode', config.execution_mode) : null,
    row('Environment', config.environment_name || 'not set'),
    row('Environment config', Object.keys(envConfig).length ? json(envConfig) : 'none'),
    config.preset ? row('Preset', config.preset) : null,
    row('Objective', config.objective_name || 'not set', config.objective_name ? null : 'required'),
    row('Direction', directionText, directionFlag),
    row('Algorithm', config.algorithm || 'random'),
    row('Iterations', String(config.n_iterations ?? 20)),
    row(
      'Initial samples',
      init.derived ? `${init.count ?? '?'} (auto: ${init.rule})` : `${init.count} (${init.rule})`
    ),
    row(
      'On complete',
      `${onComplete} — ${ON_COMPLETE_TEXT[/** @type {keyof typeof ON_COMPLETE_TEXT} */ (onComplete)] ?? onComplete}`
    ),
    row(
      'Constraints',
      constraints.length
        ? constraints
            .map(([obs, c]) => `${obs} ${c.type === 'LESS_THAN' ? '<' : '>'} ${fmt(c.value)}`)
            .join('; ')
        : 'none'
    ),
    row('Algorithm params', config.algorithm_params ? json(config.algorithm_params) : 'none'),
    row('Surrogate model', config.surrogate_model || 'default'),
    row('Model file', modelText, config.model_file && !fixed.known ? 'check the model' : null)
  );
}

/**
 * The variables table: resolved bounds, grid step, per-write limit, window %.
 *
 * @param {import('./approval-model.js').OptimizationConfig} config
 * @param {EnvironmentDetails|null} env
 * @param {Record<string, number|null>} [readbacks]
 * @returns {HTMLElement}
 */
export function approvalVariables(config, env, readbacks = {}) {
  const rows = variableRows(config, env);
  return h(
    'table',
    { class: 'grid appr-vars' },
    h(
      'thead',
      null,
      h(
        'tr',
        null,
        ...['Variable', 'Min', 'Max', 'Step', 'Max step', 'Readback', 'Range %'].map((t) =>
          h('th', { text: t })
        )
      )
    ),
    h(
      'tbody',
      null,
      ...rows.map((r) =>
        h(
          'tr',
          { class: r.unresolved ? 'flagged' : null },
          h('td', { text: r.units ? `${r.name} [${r.units}]` : r.name }),
          h('td', { class: 'num', text: fmt(r.min) }),
          h('td', { class: 'num', text: fmt(r.max) }),
          h('td', { class: 'num', text: fmt(r.step) }),
          h('td', { class: 'num', text: fmt(r.maxStep) }),
          h('td', { class: 'num', text: fmt(readbacks[r.name]) }),
          h('td', { class: 'num', text: r.rangePct == null ? '—' : `${r.rangePct.toFixed(1)}%` })
        )
      )
    )
  );
}

/**
 * Render the whole approval card into `root`.
 *
 * @param {HTMLElement} root
 * @param {{
 *   approval: ApprovalState,
 *   env: EnvironmentDetails|null,
 *   models: ModelFileInfo[]|null,
 *   readbacks?: Record<string, number|null>,
 *   gate: import('./approval-model.js').StartGate,
 *   status: string|null,
 *   owner?: string|null,
 *   startError?: string|null,
 * }} view
 * @param {{ onStart: () => void, onReviewed: () => void }} handlers
 */
export function renderApproval(root, view, handlers) {
  const { approval, env, models, gate } = view;
  if (!approval.config) {
    replaceChildren(
      root,
      h('p', {
        class: 'muted',
        text: 'No draft. Create one below, or ask the agent to draft a run.',
      })
    );
    return;
  }
  const config = approval.config;
  const draft = view.status === 'created';
  const noticeTone = approval.notice === REVIEWED_NOTICE ? 'banner-ok' : 'banner-warn';
  replaceChildren(
    root,
    h(
      'div',
      { class: 'card-head' },
      h('h2', { text: 'Approve and start' }),
      h('span', {
        class: 'muted mono',
        title: approval.revision || '',
        text: approval.revision ? `revision ${approval.revision.slice(0, 10)}` : 'no revision',
      })
    ),
    draft && approval.notice
      ? h('div', { class: `banner ${noticeTone}`, role: 'status', text: approval.notice })
      : null,
    approvalFields(config, env, models),
    approvalVariables(config, env, view.readbacks),
    view.startError
      ? h('div', { class: 'banner banner-err', role: 'alert', text: view.startError })
      : null,
    h(
      'div',
      { class: 'foot-row start-row' },
      // Beside Start, never above it: the session's answer may arrive after
      // the card is on screen, and the row keeps Start at its top either way.
      h(
        'div',
        { class: 'start-note' },
        h('span', {
          class: 'muted',
          text: gate.reason || 'Start runs exactly what is shown above.',
        }),
        gate.warning
          ? h('span', { class: 'start-owner start-owner-warn', role: 'status', text: gate.warning })
          : draft && view.owner
            ? h('span', { class: 'muted start-owner', text: `Starts as ${view.owner}.` })
            : null
      ),
      // The slot is held while unneeded, so Start keeps its place either way.
      draft
        ? h('button', {
            type: 'button',
            class: approval.needsReview ? 'osprey-btn' : 'osprey-btn slot-reserved',
            id: approval.needsReview ? 'reviewed-btn' : null,
            text: 'I have reviewed this draft',
            disabled: !approval.needsReview,
            'aria-hidden': approval.needsReview ? null : 'true',
            onclick: handlers.onReviewed,
          })
        : null,
      h('button', {
        type: 'button',
        class: 'osprey-btn primary',
        id: 'start-btn',
        text: 'Start',
        disabled: !gate.enabled,
        onclick: handlers.onStart,
      })
    )
  );
}
