// @ts-check
/**
 * The words of the run page as pure data: the sentence of what happens now,
 * the line under it, the run's facts, what is on the machine, the approval
 * and the settings of the run. Every sentence is built from a field the
 * service sends (the job, its stored config, its write authorization) or a
 * figure of run-model.js; a fact with no source is left out.
 *
 * The service says nothing about the step inside an evaluation (writing,
 * settling, reading), so the page names the evaluation only: "Evaluation N
 * of M". It says nothing of when a request for approval gives up, nor how
 * many writes a run has used: neither is shown.
 *
 * @module run-page-model
 */

import { authorizationScope, initialSamples } from './approval-model.js';
import { fmt } from './hyperscript.js';
import { finite } from './run-model.js';
import {
  STOP_WORDS,
  changeText,
  directionWords,
  endWords,
  instantText,
  methodLabel,
} from './words.js';

/**
 * @typedef {import('./store.js').State} State
 * @typedef {import('./store.js').Screen} Screen
 * @typedef {import('./approval-model.js').AuthorizationScope} AuthorizationScope
 * @typedef {import('./approval-model.js').OptimizationConfig} OptimizationConfig
 * @typedef {import('./facts-strip.js').Fact} Fact
 * @typedef {'waiting'|'running'|'paused'|'cancelling'} RunKind
 * @typedef {[string, string]} Row
 */

/**
 * The run page's kind for a screen.
 *
 * @param {Screen} screen
 * @returns {RunKind}
 */
export function runKind(screen) {
  if (screen.id === 'waiting-approval') return 'waiting';
  if (screen.cancelling) return 'cancelling';
  return screen.id === 'paused' ? 'paused' : 'running';
}

/**
 * How many of the evaluations are first tries: the total less the
 * iterations when both are known, else the count the config resolves to.
 *
 * @param {import('./run-model.js').RunFigures|null} figures
 * @param {OptimizationConfig|null} config
 * @returns {number|null}
 */
export function firstTries(figures, config) {
  if (!config) return null;
  const total = figures?.total ?? null;
  const iterations = finite(config.n_iterations);
  if (total !== null && iterations !== null && total >= iterations) return total - iterations;
  return initialSamples(config).count;
}

/**
 * @param {number} done
 * @param {number|null} total
 */
const ofTotal = (done, total) => (total === null ? `${done}` : `${done} of ${total}`);

/**
 * Where a pause of the run takes effect. The service shows `paused` at once,
 * but the runner only checks for a pause before the first evaluation, before
 * each guided step and before the final write; it does not check between the
 * first tries. So: `first` (the first tries may still write, the run stops
 * before the first guided step), `step` (the step under way finishes), `end`
 * (every evaluation is recorded; the run stops before its final write unless
 * that write has begun), or null when the plan is not known.
 *
 * @param {Screen} screen
 * @returns {'first'|'step'|'end'|null}
 */
export function pausePhase(screen) {
  const f = screen.figures;
  if (!f) return null;
  if (f.total !== null && f.done >= f.total) return 'end';
  const first = firstTries(f, screen.config);
  if (first === null) return null;
  return f.done < first ? 'first' : 'step';
}

/**
 * What a pause does, said before it is asked (the Pause button's tip) and
 * while it holds (the line under the paused sentence).
 *
 * @param {Screen} screen
 * @returns {{ ask: string, held: string }}
 */
export function pauseWords(screen) {
  const hold = screen.config?.on_complete === 'hold';
  switch (pausePhase(screen)) {
    case 'first':
      return {
        ask: 'Pauses before the first guided step. The first tries left still write.',
        held: 'The run may still write its first tries left; it stops before the first guided step.',
      };
    case 'step':
      return {
        ask: 'Pauses when the step under way has finished.',
        held: 'A step under way finishes; then nothing is written until the run resumes.',
      };
    case 'end':
      return hold
        ? {
            ask: 'Pauses before the run ends.',
            held: 'The run holds the knobs and writes nothing more.',
          }
        : {
            ask: 'Pauses before the final write, unless that write has begun.',
            held: 'The run waits before its final write, unless that write has begun.',
          };
    default:
      return {
        ask: 'Pauses at the next point where the run checks for a pause.',
        held: 'The run stops at the next point where it checks for a pause.',
      };
  }
}

/**
 * The sentence of what happens now.
 *
 * @param {Screen} screen
 * @returns {string}
 */
export function nowSentence(screen) {
  const kind = runKind(screen);
  const { done = 0, total = null } = screen.figures || {};
  if (kind === 'waiting') {
    const scope = authorizationScope(screen.job?.write_authorization);
    if (scope?.state === 'awaiting') {
      return `Waiting for ${scope.owner || "the run's owner"} to approve the writes. Nothing has been written.`;
    }
    if (scope?.state === 'requesting') {
      return 'Asking for approval of the writes. Nothing has been written.';
    }
    return 'The run is starting. Nothing has been written.';
  }
  if (kind === 'cancelling') {
    return done ? `Cancelling after evaluation ${ofTotal(done, total)}.` : 'Cancelling the run.';
  }
  if (kind === 'paused') {
    const phase = pausePhase(screen);
    if (phase !== 'step' && phase !== 'end') {
      return done
        ? `Pause asked after evaluation ${ofTotal(done, total)}.`
        : 'Pause asked before any evaluation was recorded.';
    }
    return done
      ? `Paused after evaluation ${ofTotal(done, total)}.`
      : 'Paused before the first evaluation.';
  }
  if (total !== null && done >= total) return `All ${total} evaluations done. Finishing the run.`;
  return `Evaluation ${ofTotal(total === null ? done + 1 : Math.min(done + 1, total), total)}`;
}

/**
 * The line under the sentence: how the run picks its point now, or what the
 * operator can expect.
 *
 * @param {Screen} screen
 * @returns {string|null}
 */
export function methodLine(screen) {
  const kind = runKind(screen);
  const config = screen.config;
  const { done = 0, total = null } = screen.figures || {};
  if (kind === 'waiting') {
    const scope = authorizationScope(screen.job?.write_authorization);
    if (scope?.state !== 'awaiting') return 'Nothing is written until the writes are approved.';
    const where = scope.owner ? `${scope.owner}'s` : 'their';
    return `Approve it from the control-target switch in ${where} web terminal.`;
  }
  if (kind === 'cancelling') return STOP_WORDS.now;
  if (kind === 'paused') return pauseWords(screen).held;
  if (!config) return null;
  if (total !== null && done >= total) return endWords(config.on_complete).sentence || null;
  const at = done + 1;
  const first = firstTries(screen.figures, config);
  if (at === 1) return 'First try: the setpoints as found.';
  if (first !== null && at <= first) {
    return `First tries: random settings inside the search windows (${at} of ${first}).`;
  }
  const iterations = finite(config.n_iterations);
  const count = first !== null && iterations !== null ? ` (${at - first} of ${iterations})` : '';
  return `${methodLabel(config.algorithm)} picks each setting from the readings so far${count}.`;
}

/**
 * The words of the cancel confirmation. It says what a cancel leaves on the
 * machine: the last point written, with nothing restored; before the writes
 * are approved, that nothing has been written.
 *
 * @param {Screen} screen
 * @returns {{ title: string, text: string, keep: string, confirm: string }}
 */
export function cancelWords(screen) {
  const kind = runKind(screen);
  if (kind === 'waiting') {
    const awaiting = authorizationScope(screen.job?.write_authorization)?.state === 'awaiting';
    return {
      title: awaiting ? 'Withdraw the write request?' : 'Cancel this run?',
      text: awaiting
        ? 'Nothing has been written. The run ends and its request is withdrawn.'
        : 'Nothing has been written yet. The run ends.',
      keep: 'Keep waiting',
      confirm: 'Cancel the run',
    };
  }
  return {
    title: 'Cancel this run?',
    text: STOP_WORDS.now,
    keep: kind === 'paused' ? 'Keep it paused' : 'Keep running',
    confirm: 'Cancel the run',
  };
}

/**
 * The facts of a run under way.
 *
 * @param {Screen} screen
 * @param {string} units  the objective's units ('' for none)
 * @returns {Fact[]}
 */
export function runFacts(screen, units) {
  const f = screen.figures;
  if (!f) return [];
  const unit = units ? `\u00a0${units}` : '';
  const kind = runKind(screen);
  let left = null;
  if (kind === 'running' && f.minutesLeft !== null) left = `about ${f.minutesLeft} min`;
  if (kind === 'paused' && f.total !== null) left = `${Math.max(0, f.total - f.done)} evaluations`;
  return [
    {
      key: 'best',
      label: 'Best so far',
      value: f.best === null ? null : `${fmt(f.best, 4)}${unit}`,
      accent: true,
    },
    {
      key: 'change',
      label: 'From before',
      value: changeText(f),
      tip: f.before === null ? null : `Before the run: ${fmt(f.before)}${unit}`,
    },
    { key: 'left', label: 'Left', value: left },
    { key: 'evaluations', label: 'Evaluations', value: ofTotal(f.done, f.total) },
  ];
}

/**
 * The facts of a run that waits for its write approval.
 *
 * @param {AuthorizationScope|null} scope
 * @param {string|null} sessionOwner  the account of this session, when known
 * @returns {Fact[]}
 */
export function waitFacts(scope, sessionOwner) {
  const owner = scope?.owner ?? null;
  return [
    {
      key: 'approver',
      label: 'Approves',
      value: owner && owner === sessionOwner ? `${owner} (you)` : owner,
    },
    {
      key: 'where',
      label: 'Where',
      value: scope?.state === 'awaiting' ? 'control-target switch' : null,
    },
    { key: 'steps', label: 'Writes asked', value: scope?.steps ?? null },
  ];
}

/**
 * Where the knobs are, as far as the records prove it. The runner writes a
 * point, lets the machine settle and reads it, and only then records it: the
 * knobs can already be at the next point while the last record is older.
 *
 * @param {Screen} screen
 * @returns {string}
 */
export function machineNow(screen) {
  const kind = runKind(screen);
  const done = screen.figures?.done ?? 0;
  if (kind === 'waiting') return 'As before the run. Nothing has been written.';
  if (kind === 'cancelling') return STOP_WORDS.now;
  const last = done
    ? `Evaluation ${done} is the last one recorded.`
    : 'No evaluation is recorded yet; the first one writes the setpoints as found.';
  if (kind === 'paused') return `${last} ${pauseWords(screen).held}`;
  return `${last} The knobs can already be at the next point: the run writes it before its reading is recorded.`;
}

/** The write authorization's state in the operator's words. */
const AUTH_STATES = Object.freeze(
  /** @type {Record<string, string>} */ ({
    requesting: 'being requested',
    awaiting: 'waiting for approval',
    granted: 'approved',
    ended: 'ended',
    released: 'released',
  })
);

/**
 * The rows of the write approval: its state, whose it is, where it was
 * approved, the writes it allows, until when, and its id. Empty without a
 * write authorization.
 *
 * @param {AuthorizationScope|null} scope
 * @param {{ sessionOwner?: string|null, now?: number, reason?: string|null }} [options]
 * @returns {Row[]}
 */
export function approvalRows(scope, { sessionOwner = null, now = Date.now(), reason = null } = {}) {
  if (!scope) return [];
  /** @type {Row[]} */
  const rows = [];
  const add = (/** @type {string} */ label, /** @type {string|number|null} */ value) => {
    if (value !== null && value !== '') rows.push([label, String(value)]);
  };
  const ended = scope.state === 'ended' && reason ? `ended: ${reason}` : null;
  add('State', ended || (scope.state ? (AUTH_STATES[scope.state] ?? scope.state) : null));
  const owner = scope.owner && scope.owner === sessionOwner ? `${scope.owner} (you)` : scope.owner;
  add(scope.state === 'granted' ? 'Owner' : 'Approves', owner);
  if (scope.state === 'awaiting') {
    add('Where', "the control-target switch in the owner's web terminal, under Programs");
  }
  add('Approved from', scope.approvedFrom);
  const final =
    scope.completion && scope.completion !== 'none' ? ', one of them the final write' : '';
  add(
    scope.state === 'granted' ? 'Writes allowed' : 'Writes asked',
    scope.steps === null ? null : `${scope.steps}${final}`
  );
  add('Valid until', scope.expiresAt === null ? null : instantText(scope.expiresAt, now));
  add(scope.state === 'granted' ? 'Authorization' : 'Request', scope.id);
  return rows;
}

/**
 * The settings of the run, from its stored config.
 *
 * @param {OptimizationConfig|null} config
 * @param {{ figures?: import('./run-model.js').RunFigures|null, revision?: string|null,
 *   objective?: string, environment?: string }} [extra]
 * @returns {Row[]}
 */
export function settingsRows(
  config,
  { figures = null, revision = null, objective = '', environment = '' } = {}
) {
  if (!config) return [];
  /** @type {Row[]} */
  const rows = [];
  const add = (/** @type {string} */ label, /** @type {string|null|undefined} */ value) => {
    if (value) rows.push([label, value]);
  };
  const goal = directionWords(figures?.direction || config.objective_direction)?.better;
  add('Objective', objective ? `${objective}${goal ? `, ${goal}` : ''}` : null);
  add('Method', methodLabel(config.algorithm));
  const total = figures?.total ?? null;
  const first = firstTries(figures ?? null, config);
  const iterations = finite(config.n_iterations);
  if (total !== null) {
    const split =
      first !== null && iterations !== null ? `: ${first} first tries, then ${iterations}` : '';
    add('Evaluations', `${total}${split}`);
  }
  const knobs = config.variables?.length ?? 0;
  add('Knobs', knobs ? String(knobs) : null);
  add('Environment', environment || config.environment_name);
  add('Control target', config.control_target);
  add('Revision', revision ? revision.slice(0, 10) : null);
  return rows;
}
