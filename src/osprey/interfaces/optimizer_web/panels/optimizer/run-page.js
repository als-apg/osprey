// @ts-check
/**
 * The run page: a run that waits for its write approval, runs, or is paused.
 *
 * The stage holds the run header (run-header.js: what happens now, Pause or
 * Resume and Cancel, the progress line and the facts) over the body. During
 * the run the body is the objective chart with a switch to the table of the
 * evaluations. While the run waits for its write approval the body has no
 * chart: a card in the warning colour says who approves, where, that nothing
 * has been written, and what the run does once approved. The inspector holds
 * the knobs and the run's details (run-knobs.js).
 *
 * The page is updated about once a second. Only the stage's body scrolls, so
 * the header with Pause and Cancel stays in sight; in a narrow tile the
 * drawer opens below it. The primary slot of the toolbar stays empty: Pause
 * and Cancel are in the run header only, never where Start was.
 *
 * @module run-page
 */

import { icon } from '/design-system/js/icons.js';
import { wireSegmented } from '/design-system/js/components/segmented.js';
import { authorizationScope } from './approval-model.js';
import { evaluationsTable, updateEvaluationsTable } from './evaluations-table.js';
import { h } from './hyperscript.js';
import { knobLabel } from './knob-bar.js';
import { chartInput, chartLegend, mountObjectiveChart } from './objective-chart.js';
import { createRunHeader } from './run-header.js';
import { mountRunInspector } from './run-knobs.js';
import { firstTries, pausePhase, runKind } from './run-page-model.js';
import { environmentLabel, objectiveLabel } from './runs-list.js';
import { TARGETS } from './shell.js';
import { directionWords, endWords } from './words.js';

/**
 * @typedef {import('./store.js').State} State
 * @typedef {import('./store.js').Screen} Screen
 * @typedef {import('./approval-model.js').EnvironmentDetails} EnvironmentDetails
 * @typedef {import('./approval-model.js').AuthorizationScope} AuthorizationScope
 */

/** The views of the chart section. */
const CHART_VIEWS = Object.freeze({
  objective: {
    label: 'Objective',
    detail: 'The objective at each evaluation and the best so far.',
  },
  table: { label: 'Evaluations', detail: 'Every evaluation as a row, newest first.' },
});

/**
 * The environment's details, when the panel has read them.
 *
 * @param {State} state
 * @param {string|null} name
 * @returns {EnvironmentDetails|null}
 */
function envOf(state, name) {
  if (!name) return null;
  const own = state.envs[name]?.details;
  if (own) return own;
  const draft = state.approvalEnv.details;
  return draft && draft.name === name ? draft : null;
}

/**
 * The words of the waiting card: who approves, where, and what the run does.
 *
 * @param {State} state
 * @param {Screen} screen
 * @param {AuthorizationScope|null} scope
 * @param {{ objective: string, env: EnvironmentDetails|null }} extra
 * @returns {Node[]}
 */
function waitingCard(state, screen, scope, { objective, env }) {
  const b = (/** @type {string} */ text) => h('b', { text });
  const owner = scope?.owner || null;
  const self = !!owner && owner === state.owner;
  const awaiting = scope?.state === 'awaiting';
  const title = awaiting
    ? 'Write approval needed'
    : scope?.state === 'requesting'
      ? 'Asking for write approval'
      : 'Starting the run';
  const steps = awaiting
    ? [
        h(
          'li',
          null,
          b(owner ? (self ? `${owner} (you)` : owner) : "The run's owner"),
          ' approves this run’s writes.'
        ),
        h(
          'li',
          null,
          'Where: the ',
          b('control-target switch'),
          ` in ${self ? 'your' : owner ? `${owner}’s` : 'their'} own web terminal, under Programs.`
        ),
        h(
          'li',
          null,
          'Until then the run waits. ',
          b('Nothing has been written'),
          ' and the machine is as it was.'
        ),
      ]
    : [
        h('li', null, 'The run asks for approval of its writes before it writes anything.'),
        h('li', null, b('Nothing has been written'), ' and the machine is as it was.'),
      ];
  const allows =
    scope?.steps == null
      ? null
      : `Once approved, the request allows ${scope.steps} write${scope.steps === 1 ? '' : 's'}.`;
  const request = scope?.id ? `Request ${scope.id.slice(0, 8)}` : null;
  const card = h(
    'section',
    { class: 'osprey-card opt-wait-card', 'aria-label': title },
    h(
      'div',
      { class: 'osprey-card-head' },
      h('span', { class: 'opt-wait-ic', 'aria-hidden': 'true' }, icon('shield-check')),
      h('span', { class: 'osprey-card-title', text: title }),
      h(
        'span',
        { class: 'osprey-chip osprey-chip--pending' },
        h('span', { class: 'osprey-chip-label', text: 'not approved yet' })
      )
    ),
    h('div', { class: 'osprey-card-body' }, h('ol', { class: 'opt-wait-steps' }, ...steps)),
    allows || request
      ? h(
          'div',
          { class: 'osprey-card-foot opt-wait-foot' },
          allows ? h('span', { text: allows }) : null,
          request ? h('span', { class: 'opt-wait-id', text: request }) : null
        )
      : null
  );
  const config = screen.config;
  const n = config?.variables?.length ?? 0;
  if (!config || !n) return [card];
  const target =
    TARGETS[/** @type {keyof typeof TARGETS} */ (config.control_target || '')]?.label.toLowerCase();
  const seek = directionWords(
    config.objective_direction || env?.observables?.[config.objective_name || '']?.direction
  )?.seek;
  const lead = h(
    'p',
    { class: 'opt-rp-lead' },
    'It writes ',
    b(`${n} knob${n === 1 ? '' : 's'}`),
    target ? h('span', null, ' on the ', b(target)) : null,
    scope?.steps == null ? '' : `, up to ${scope.steps} times`,
    ', each only inside its search window. After each write it reads ',
    b(objective || config.objective_name || 'the objective'),
    seek ? `, looking for the ${seek} value. ` : '. ',
    endWords(config.on_complete).sentence
  );
  return [
    card,
    h(
      'div',
      { class: 'opt-wait-lead' },
      h('div', {
        class: 'osprey-section-label opt-flush',
        text: 'What the run does once approved',
      }),
      lead
    ),
  ];
}

/** @type {import('./view-host.js').MountView} */
export function mountRunPage(ctx) {
  const { actions, shell } = ctx;
  const doc = ctx.stage.ownerDocument;
  const header = createRunHeader({ actions, doc });

  // The chart section: a caption, the view switch, then the chart or the table.
  const caption = h('div', { class: 'opt-chart-caption' });
  const seg = h(
    'div',
    { class: 'osprey-seg sm', role: 'group', 'aria-label': 'Show' },
    ...Object.entries(CHART_VIEWS).map(([id, v]) =>
      h('button', {
        type: 'button',
        'data-value': id,
        'aria-pressed': id === 'objective' ? 'true' : 'false',
        title: v.detail,
        text: v.label,
      })
    )
  );
  const plot = h('div', { class: 'opt-chart-host' });
  const chart = mountObjectiveChart(plot);
  /** @type {HTMLElement|null} */
  let table = null;
  const tableHost = h('div', { class: 'opt-chart-host', hidden: true });
  const chartSection = h(
    'section',
    { class: 'opt-rp-chart', 'aria-label': 'The objective' },
    h('div', { class: 'opt-chart-head' }, caption, h('span', { class: 'opt-grow' }), seg),
    plot,
    tableHost
  );
  let view = 'objective';
  /** @type {(() => void)|null} */
  let redraw = null;
  const unwire = wireSegmented(seg, (value) => {
    view = value;
    plot.hidden = view !== 'objective';
    tableHost.hidden = view !== 'table';
    redraw?.();
  });

  const waitPage = h('div', { class: 'opt-wait', hidden: true });
  const body = h('div', { class: 'opt-rp-body' }, chartSection, waitPage);
  const page = h('div', { class: 'opt-rp' }, header.head, body);
  ctx.stage.replaceChildren(page);
  const inspector = mountRunInspector(ctx);

  // In a narrow tile the drawer opens below the header, so Pause and Cancel stay in sight.
  let drawerTop = -1;
  const placeDrawer = () => {
    const top = header.head.offsetHeight;
    if (top === drawerTop) return;
    drawerTop = top;
    shell.setDrawerTop(top);
  };
  const headObserver =
    typeof ResizeObserver === 'function' ? new ResizeObserver(placeDrawer) : null;
  headObserver?.observe(header.head);

  let waitKey = '';
  let captionKey = '';
  /** @type {string|null} */
  let asked = null;

  return {
    update(state, screen) {
      const kind = runKind(screen);
      const job = screen.job;
      const environment = screen.subject.environment;
      if (environment && !state.envs[environment] && asked !== environment) {
        asked = environment;
        void actions.loadEnvironment(environment);
      }
      const env = envOf(state, environment);
      const f = screen.figures;
      const objectiveName = f?.objective || screen.config?.objective_name || null;
      const objective = objectiveLabel(state, environment, objectiveName);
      const units = (objectiveName && env?.observables?.[objectiveName]?.units) || '';
      page.dataset.kind = kind;
      header.update(state, screen, { units });

      const waiting = kind === 'waiting';
      chartSection.hidden = waiting;
      waitPage.hidden = !waiting;
      const scope = authorizationScope(job?.write_authorization);
      if (waiting) {
        const key = JSON.stringify([scope, state.owner, screen.config, objective, !!env]);
        if (key !== waitKey) {
          waitKey = key;
          waitPage.replaceChildren(...waitingCard(state, screen, scope, { objective, env }));
        }
      } else if (f) {
        const records = job?.data || [];
        const nextCaption = JSON.stringify([view, objective, units, f.before !== null, f.done]);
        if (nextCaption !== captionKey) {
          captionKey = nextCaption;
          const title =
            view === 'table'
              ? h(
                  'span',
                  { class: 'osprey-section-label opt-flush opt-chart-title' },
                  'Evaluations',
                  h('span', { class: 'osprey-aside', text: `${f.done}, newest first` })
                )
              : h(
                  'span',
                  { class: 'osprey-section-label opt-flush opt-chart-title' },
                  objective || 'Objective',
                  units ? h('span', { class: 'osprey-aside', text: units }) : null
                );
          caption.replaceChildren(
            title,
            view === 'objective' ? chartLegend({ before: f.before !== null }) : ''
          );
        }
        redraw = () => {
          if (view === 'objective') {
            chart.update({
              ...chartInput(f, records),
              label: objective,
              units,
              xMax: f.total,
              firstTries: firstTries(f, screen.config),
              // While first tries still write, the pause does not hold yet: the
              // last point stays live and there is no pause marker.
              current:
                kind === 'running' || (kind === 'paused' && pausePhase(screen) === 'first')
                  ? 'live'
                  : 'still',
              end:
                kind === 'paused' && f.done && pausePhase(screen) !== 'first'
                  ? { index: f.done - 1, label: 'paused', tone: 'warning' }
                  : null,
            });
            return;
          }
          const variables = screen.config?.variables || job?.variable_names || [];
          const spec = {
            records,
            objective: f.objective,
            direction: f.direction,
            variables,
            bests: f.bests,
            bestIndex: f.bestIndex,
            label: objective,
            labelOf: (/** @type {string} */ name) => knobLabel(name, env?.variables?.[name]).title,
          };
          if (!table) {
            table = evaluationsTable(spec);
            tableHost.replaceChildren(table);
          } else updateEvaluationsTable(table, spec);
        };
        redraw();
      }

      inspector.update(state, screen, {
        env,
        revision: job && job.job_id === state.approval.jobId ? state.approval.revision : null,
        objective,
        environment: environmentLabel(state, environment),
      });
      placeDrawer();
    },
    unmount() {
      unwire();
      headObserver?.disconnect();
      chart.destroy();
      header.destroy();
      shell.setDrawerTop(0);
    },
  };
}
