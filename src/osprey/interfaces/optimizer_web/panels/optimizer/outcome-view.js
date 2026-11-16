// @ts-check
/**
 * The outcome of a run: the present job once it has ended (`completed`,
 * `failed`, `cancelled`) and a past run (`result`: a stored run, or a job of
 * `GET jobs` that is not the job of now).
 *
 * The stage has a band that says in words how the run ended and what it left
 * on the machine (outcome-model.js; a statement the fields do not prove is
 * not made), a line of detail, the progress where the run stopped and the
 * facts strip; under it the chart area with its views: the objective
 * (objective-chart.js), the knobs (outcome-knobs.js) and the evaluations
 * (evaluations-table.js). The inspector (outcome-inspector.js) holds the
 * outcome, what changed per knob, the run's facts and settings, and "Run
 * again", which opens a new set-up from the run's stored config and writes
 * nothing. There is no Restore.
 *
 * A stored run shows what the service still has: its status only while the
 * service holds its job (about an hour), its evaluations only when they were
 * stored. With none stored, one sentence says so.
 *
 * @module outcome-view
 */

import { renderState } from '/design-system/js/panel-state.js';
import { TERMINAL_STATUSES, initialSamples } from './approval-model.js';
import { observableLabel } from './config-sections.js';
import { evaluationsTable } from './evaluations-table.js';
import { factsStrip } from './facts-strip.js';
import { fmt, h, replaceChildren } from './hyperscript.js';
import { knobLabel } from './knob-bar.js';
import { chartInput, chartLegend, mountObjectiveChart } from './objective-chart.js';
import { NO_CONFIG_WHY, outcomeInspector } from './outcome-inspector.js';
import { knobsKey, knobsView } from './outcome-knobs.js';
import { outcomeOf } from './outcome-model.js';
import { isActive } from './poller.js';
import { jobFigures, resultFigures } from './run-model.js';
import { environmentLabel, runTime } from './runs-list.js';
import { changeText, methodLabel } from './words.js';

/**
 * @typedef {import('./store.js').State} State
 * @typedef {import('./store.js').Screen} Screen
 * @typedef {import('./outcome-inspector.js').Source} Source
 * @typedef {import('./outcome-model.js').Outcome} Outcome
 * @typedef {'objective'|'knobs'|'evaluations'} ViewId
 */

/** The views of the chart area. */
const VIEWS = /** @type {const} */ ([
  ['objective', 'Objective', 'The objective at each evaluation and the best so far.'],
  [
    'knobs',
    'Knobs',
    'Each knob in its range: where it started, its best point, where the run left it.',
  ],
  ['evaluations', 'Evaluations', 'Every evaluation as a row, newest first.'],
]);

/**
 * What the screen shows, from the screen and the state. A job picked in the
 * runs list shows its stored run once the service has one.
 *
 * @param {State} state
 * @param {Screen} screen
 * @returns {Source}
 */
function outcomeSource(state, screen) {
  if (screen.id !== 'result') {
    const job = screen.job;
    const config = screen.config;
    return {
      key: `job:${job?.job_id}`,
      stored: false,
      runId: null,
      info: null,
      entry: null,
      job,
      status: job?.status ?? null,
      config,
      figures: screen.figures,
      records: job?.data || [],
      variables: config?.variables || job?.variable_names || [],
      environment: screen.subject.environment,
      started: null,
      applied: null,
      bounds: job?.variable_bounds || null,
    };
  }
  const past = screen.past;
  const job = past?.job || null;
  let info = past?.info || null;
  let runId = past?.runId || null;
  if (!runId && job?.job_id) {
    info = (state.results.list || []).find((r) => r.job_id === job.job_id) || null;
    runId = info?.run_id || null;
  }
  const entry = runId ? state.past[runId] || null : null;
  const details = entry?.details || null;
  const held = job && job.job_id === state.activeJob?.job_id ? state.activeConfig : null;
  const config = screen.config || details?.config || held || null;
  let figures = screen.figures;
  let records = details?.data_records || [];
  if (!figures && details) figures = resultFigures(details);
  else if (!figures && job?.data) {
    figures = jobFigures(job, config);
    records = job.data;
  }
  const best = details?.best_point;
  return {
    key: runId ? `run:${runId}` : `job:${job?.job_id}`,
    stored: !!runId,
    runId,
    info,
    entry,
    job,
    status: job?.status ?? null,
    config,
    figures,
    records,
    variables: config?.variables || job?.variable_names || [],
    environment: config?.environment_name || info?.environment || job?.environment_name || null,
    started: runTime(info?.timestamp),
    applied: best && Object.keys(best).length ? best : null,
    bounds: job?.variable_bounds || null,
  };
}

/**
 * The environment's details, when the panel has read them.
 *
 * @param {State} state
 * @param {string|null} name
 */
function envOf(state, name) {
  if (!name) return null;
  const own = state.envs[name]?.details;
  if (own) return own;
  const draft = state.approvalEnv.details;
  return draft && draft.name === name ? draft : null;
}

/**
 * The id of the job that made what the screen shows.
 *
 * @param {Source} source
 * @returns {string|null}
 */
function runJobId(source) {
  return source.job?.job_id || source.info?.job_id || null;
}

/**
 * The banner's sentences: the lead, the service's reasons, the machine.
 *
 * @param {Source} source
 * @param {Outcome} outcome
 * @returns {{ lead: string, text: string }}
 */
function bannerText(source, outcome) {
  const when = source.started ? `${source.started.day} ${source.started.clock}` : null;
  let lead = outcome.lead;
  if (source.stored && when) lead = `Run of ${when}.${source.status ? ` ${outcome.lead}` : ''}`;
  if (outcome.kind === 'open') {
    return {
      lead,
      text: isActive(source.status)
        ? 'Its run page shows its evaluations and its controls.'
        : 'It is not the job of now: its evaluations and its controls are not shown here.',
    };
  }
  const reasons = outcome.reasons.map((r) => (/[.!?]$/.test(r) ? r : `${r}.`));
  return { lead, text: [...reasons, outcome.machine.text].join(' ') };
}

/** @type {import('./view-host.js').MountView} */
export function mountOutcomeView(ctx) {
  const { actions, shell } = ctx;
  const band = h('div', { class: 'opt-out-band' });
  const body = h('div', { class: 'opt-out-body' });
  ctx.stage.replaceChildren(h('div', { class: 'opt-out-page' }, band, body));

  const caption = h('div', { class: 'opt-out-caption' });
  const viewHost = h('div', { class: 'opt-out-view' });
  /** @type {ViewId} */
  let view = 'objective';
  const seg = h(
    'div',
    { class: 'osprey-seg sm', role: 'group', 'aria-label': 'Show' },
    ...VIEWS.map(([id, label, detail]) =>
      h('button', {
        type: 'button',
        'data-view': id,
        'aria-pressed': id === view ? 'true' : 'false',
        title: label,
        'data-tip-detail': detail,
        text: label,
        onclick: () => {
          view = id;
          for (const b of /** @type {HTMLElement[]} */ (Array.from(seg.children))) {
            b.setAttribute('aria-pressed', b.dataset.view === id ? 'true' : 'false');
          }
          bodyKey = '';
          if (latest) draw(latest.state, latest.screen);
        },
      })
    )
  );
  const chartArea = h(
    'section',
    { class: 'opt-out-chart', 'aria-label': 'Run' },
    h('div', { class: 'opt-out-chart-head' }, caption, h('span', { class: 'opt-grow' }), seg),
    viewHost
  );

  /** @type {ReturnType<typeof mountObjectiveChart>|null} */
  let chart = null;
  /** @type {{ state: State, screen: Screen }|null} */
  let latest = null;
  let bandKey = '';
  let bodyKey = '';
  let inspectorKey = '';
  let primaryFor = '';
  /** Environment reads asked once each, so a failed read is not repeated on every poll. */
  const asked = new Set();
  /**
   * The runs list each stored run was last asked with. A stored run whose
   * job has not ended (or could not be read) is asked again when the runs
   * list is read again, not on every poll.
   *
   * @type {Map<string, unknown>}
   */
  const askedWith = new Map();

  function dropChart() {
    chart?.destroy();
    chart = null;
  }

  /**
   * @param {State} state
   * @param {Source} source
   */
  function askReads(state, source) {
    const env = source.environment;
    if (env && !state.envs[env] && !asked.has(`env:${env}`)) {
      asked.add(`env:${env}`);
      void actions.loadEnvironment(env);
    }
    const runId = source.runId;
    const entry = source.entry;
    const ended = TERMINAL_STATUSES.includes(entry?.job?.status);
    if (runId && !entry?.loading && !(entry?.details && ended)) {
      if (askedWith.get(runId) !== state.results) {
        askedWith.set(runId, state.results);
        void actions.loadResult(runId);
      }
    }
  }

  /**
   * @param {Source} source
   * @param {Outcome} outcome
   * @param {{ title: string, units: string }} objective
   * @param {boolean} past  a past run: the band offers the way back to now
   */
  function drawBand(source, outcome, objective, past) {
    const f = source.figures;
    const entry = source.entry;
    const reading = source.stored && !entry?.details && !entry?.error;
    const { lead, text } = bannerText(source, outcome);
    const tone = entry?.error ? 'err' : reading ? 'info' : outcome.tone;
    const unit = objective.units ? `\u00a0${objective.units}` : '';
    const line = [
      f
        ? f.total == null
          ? `${f.done} evaluations`
          : `${f.done} of ${f.total} evaluations`
        : null,
      source.config ? methodLabel(source.config.algorithm) : null,
    ].filter(Boolean);
    replaceChildren(
      band,
      h(
        'div',
        { class: 'opt-out-head' },
        h(
          'div',
          {
            class: `osprey-banner osprey-banner--${tone} opt-out-banner`,
            role: tone === 'err' ? 'alert' : 'status',
            'data-part': 'outcome',
          },
          h(
            'span',
            { class: 'osprey-banner-text' },
            h('b', { text: lead }),
            ' ',
            entry?.error
              ? `The stored run could not be read: ${entry.error}`
              : reading
                ? 'Reading the stored run…'
                : text
          )
        ),
        past
          ? h('button', {
              type: 'button',
              class: 'osprey-btn ghost sm',
              id: 'back-to-now-btn',
              text: 'Back to now',
              onclick: () => actions.select({ kind: 'present' }),
            })
          : null,
        outcome.kind === 'open' && isActive(source.status) && runJobId(source)
          ? h('button', {
              type: 'button',
              class: 'osprey-btn sm',
              id: 'show-run-btn',
              text: 'Show the run',
              onclick: () =>
                actions.select({ kind: 'job', jobId: /** @type {string} */ (runJobId(source)) }),
            })
          : null
      ),
      line.length ? h('p', { class: 'opt-out-line', text: line.join(' · ') }) : null,
      f && f.total
        ? h(
            'div',
            { class: `opt-out-prog is-${outcome.tone}`, 'aria-hidden': 'true' },
            h('span', { style: `width: ${Math.min(100, (f.done / f.total) * 100).toFixed(1)}%` })
          )
        : null,
      f
        ? factsStrip(
            [
              { key: 'objective', label: 'Objective', value: objective.title || null },
              {
                key: 'before',
                label: 'Start',
                value: f.before === null ? null : `${fmt(f.before)}${unit}`,
              },
              {
                key: 'best',
                label: 'Best',
                value: f.best === null ? null : `${fmt(f.best)}${unit}`,
                accent: true,
              },
              { key: 'change', label: 'Change', value: changeText(f) },
              {
                key: 'evaluations',
                label: 'Evaluations',
                value: f.total == null ? String(f.done) : `${f.done} of ${f.total}`,
              },
            ],
            { label: 'Run figures' }
          )
        : null
    );
  }

  /**
   * @param {State} state
   * @param {Source} source
   * @param {Outcome} outcome
   * @param {{ title: string, units: string }} objective
   */
  function drawBody(state, source, outcome, objective) {
    const f = source.figures;
    const entry = source.entry;
    const env = envOf(state, source.environment);
    if (!f || !source.records.length) {
      dropChart();
      let text = 'No evaluation was recorded.';
      if (source.stored) text = entry?.details ? 'No evaluation was stored for this run.' : '';
      if (outcome.kind === 'open') text = '';
      const host = h('div', { class: 'opt-stage-fill' });
      if (entry?.error) {
        renderState(host, {
          icon: 'triangle-alert',
          message: 'The stored run could not be read',
          detail: entry.error,
        });
      } else if (source.stored && !entry?.details) {
        renderState(host, { icon: 'history', message: 'Reading the stored run…', detail: '' });
      } else if (text) {
        replaceChildren(host, h('p', { class: 'opt-note opt-out-none', text }));
      }
      replaceChildren(body, host);
      return;
    }
    if (chartArea.parentElement !== body) replaceChildren(body, chartArea);
    const title = (/** @type {string} */ text, /** @type {Node|string|null} */ aside) =>
      h(
        'span',
        { class: 'osprey-section-label opt-flush opt-out-title' },
        text,
        aside ? h('span', { class: 'osprey-aside' }, aside) : null
      );
    if (view === 'objective') {
      replaceChildren(
        caption,
        title(objective.title || 'Objective', objective.units || null),
        chartLegend({ before: f.before !== null })
      );
      if (!chart) {
        viewHost.replaceChildren();
        chart = mountObjectiveChart(viewHost);
      }
      chart.update({
        ...chartInput(f, source.records),
        label: objective.title || null,
        units: objective.units || null,
        end: outcome.end,
        xMax: f.total,
        firstTries: source.config ? initialSamples(source.config).count : null,
        current: null,
      });
      return;
    }
    dropChart();
    if (view === 'knobs') {
      replaceChildren(
        caption,
        title(`Knobs · ${source.variables.length}`, null),
        knobsKey(outcome.machine, !!env)
      );
      replaceChildren(
        viewHost,
        knobsView({
          variables: source.variables,
          config: source.config,
          bounds: source.bounds,
          env,
          figures: f,
          machine: outcome.machine,
          records: source.records,
        })
      );
      return;
    }
    replaceChildren(caption, title('Evaluations', `${f.done}, newest first`));
    replaceChildren(
      viewHost,
      evaluationsTable({
        records: source.records,
        objective: f.objective,
        direction: f.direction,
        variables: source.variables,
        bestIndex: f.bestIndex,
        label: objective.title || null,
        labelOf: (name) => knobLabel(name, env?.variables?.[name]).title,
      })
    );
  }

  /**
   * @param {State} state
   * @param {Screen} screen
   */
  function draw(state, screen) {
    const read = outcomeSource(state, screen);
    askReads(state, read);
    const env = envOf(state, read.environment);
    const outcome = outcomeOf({
      status: read.status,
      job: read.job,
      config: read.config,
      figures: read.figures,
      applied: read.applied,
    });
    // A run under way is not a past run: what its results directory holds so
    // far is a moment of it, not its outcome, and is not shown as one.
    const source = outcome.kind === 'open' ? { ...read, figures: null, records: [] } : read;
    const f = source.figures;
    const named = observableLabel(env, f?.objective || source.config?.objective_name || null);
    const objective = { title: named.title, units: named.units };
    const envLabel = source.environment ? environmentLabel(state, source.environment) : null;
    const shown = JSON.stringify([
      source.key,
      source.status,
      outcome.word,
      outcome.reasons,
      outcome.machine.text,
      outcome.machine.values,
      f?.done,
      f?.total,
      f?.best,
      f?.before,
      !!source.config,
      source.entry?.error,
      !!source.entry?.details,
      objective,
      envLabel,
      screen.id,
    ]);
    if (shown !== bandKey) {
      bandKey = shown;
      drawBand(source, outcome, objective, screen.id === 'result');
    }
    const nextBody = JSON.stringify([shown, view, !!env, source.records.length]);
    if (nextBody !== bodyKey) {
      bodyKey = nextBody;
      drawBody(state, source, outcome, objective);
    }
    const nextInspector = JSON.stringify([shown, !!env, state.models.list?.length ?? null]);
    if (nextInspector !== inspectorKey) {
      inspectorKey = nextInspector;
      shell.fillInspector(
        outcomeInspector({
          source,
          outcome,
          objective,
          env,
          models: state.models.list,
          environmentLabel: envLabel,
          onRunAgain: () => actions.runAgain(source.config),
        })
      );
    }
    const config = source.config;
    const primary = `${source.key}|${!!config}`;
    if (primary !== primaryFor) {
      primaryFor = primary;
      shell.setPrimary({
        label: 'Run again',
        id: 'run-again-primary',
        disabled: !config,
        why: NO_CONFIG_WHY,
        detail: 'Opens a new set-up from these settings. Writes nothing.',
        key: `again:${primary}`,
        onClick: () => actions.runAgain(config),
      });
    }
  }

  return {
    update(state, screen) {
      latest = { state, screen };
      draw(state, screen);
    },
    unmount() {
      dropChart();
    },
  };
}
