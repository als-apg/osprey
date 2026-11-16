// @ts-check
/**
 * The table of stored runs (`history`), newest first, under the filter of the
 * runs list (`pastRuns`). A row opens that run (`result`).
 *
 * A stored run's entry (`GET results`) names its start time (the service
 * host's clock), its environment, its objective, whether a model was saved
 * and the job that made it: the table shows those and nothing more. Status
 * and figures need one read per run, so they show on the run's own screen.
 *
 * @module history-view
 */

import { renderState } from '/design-system/js/panel-state.js';
import { icon } from '/design-system/js/icons.js';
import { h, replaceChildren } from './hyperscript.js';
import { dayLabel, environmentLabel, objectiveLabel, pastRuns, runTime } from './runs-list.js';

/**
 * @typedef {import('./store.js').State} State
 * @typedef {{
 *   runId: string,
 *   day: string,
 *   clock: string,
 *   objective: string,
 *   objectiveRaw: string|null,
 *   environment: string,
 *   model: boolean,
 *   job: string|null,
 * }} HistoryRow
 */

/**
 * The rows of the table, from the stored runs the filter lets through.
 *
 * @param {State} state
 * @param {number} now  ms, for the day words
 * @returns {HistoryRow[]}
 */
export function historyRows(state, now) {
  return pastRuns(state).map((r) => {
    const when = runTime(r.timestamp);
    const objective = objectiveLabel(state, r.environment, r.objective_name);
    return {
      runId: r.run_id,
      day: when ? dayLabel(when.day, now) : String(r.timestamp ?? ''),
      clock: when?.clock ?? '',
      objective,
      objectiveRaw: objective && objective !== r.objective_name ? r.objective_name : null,
      environment: environmentLabel(state, r.environment),
      model: !!r.has_model,
      job: r.job_id || null,
    };
  });
}

/** @type {import('./view-host.js').MountView} */
export function mountHistoryView(ctx) {
  const { actions, shell } = ctx;
  const host = h('div', { class: 'opt-stage-scroll opt-hist' });
  ctx.stage.replaceChildren(host);
  const side = h('div', { class: 'opt-hist-pick' });
  renderState(side, {
    icon: 'history',
    message: 'Pick a run',
    detail: 'Its result, its settings and what it left on the machine show here.',
  });
  shell.fillInspector({ title: 'Run details', body: side });
  shell.setPrimary({
    label: 'New run',
    id: 'new-run-primary',
    detail: 'Set up a run by hand. The agent can also draft one in the chat.',
    onClick: () => actions.newRun(),
  });
  let drawn = '';

  /** @param {string} runId */
  const open = (runId) => actions.select({ kind: 'run', runId });

  /** @param {HistoryRow} r */
  function row(r) {
    return h(
      'tr',
      {
        class: 'is-click',
        tabindex: '0',
        'data-run': r.runId,
        'aria-label': `${r.objective || r.environment}, ${r.day} ${r.clock}`.trim(),
        onclick: () => open(r.runId),
        onkeydown: (/** @type {KeyboardEvent} */ e) => {
          if (e.key !== 'Enter' && e.key !== ' ') return;
          e.preventDefault();
          open(r.runId);
        },
      },
      h(
        'td',
        { class: 'opt-hist-when' },
        h('span', { class: 'opt-hist-main', text: r.day }),
        r.clock ? h('span', { class: 'opt-hist-sub opt-hist-mono', text: r.clock }) : null
      ),
      h(
        'td',
        null,
        h('span', { class: 'opt-hist-main', text: r.objective || '' }),
        r.objectiveRaw
          ? h('span', { class: 'opt-hist-sub opt-hist-mono', text: r.objectiveRaw })
          : null
      ),
      h('td', { text: r.environment }),
      h('td', { class: 'opt-hist-opt', text: r.model ? 'saved' : '' }),
      h('td', {
        class: 'opt-hist-opt opt-hist-mono',
        title: r.job || '',
        text: r.job ? r.job.slice(0, 8) : '',
      })
    );
  }

  return {
    update(state) {
      const all = state.results.list || [];
      const rows = historyRows(state, Date.now());
      const filter = state.filter.trim();
      const key = JSON.stringify([
        rows,
        filter,
        all.length,
        state.results.error,
        !state.results.list,
      ]);
      if (key === drawn) return;
      drawn = key;
      const counted =
        rows.length === all.length ? `${all.length}` : `${rows.length} of ${all.length}`;
      const clear = filter
        ? h(
            'button',
            {
              type: 'button',
              class: 'osprey-btn ghost sm',
              id: 'clear-filter-btn',
              onclick: () => actions.setFilter(''),
            },
            icon('x', { size: 12 }),
            h('span', { class: 'osprey-btn-label', text: `Clear the filter “${filter}”` })
          )
        : null;
      const head = h(
        'div',
        { class: 'osprey-section-label opt-flush opt-hist-head' },
        'Past runs',
        all.length
          ? h('span', {
              class: 'osprey-aside',
              text: `${counted} run${all.length === 1 ? '' : 's'}, newest first`,
            })
          : null
      );
      const fill = h('div', { class: 'opt-stage-fill' });
      if (state.results.error && !all.length) {
        renderState(fill, {
          icon: 'triangle-alert',
          message: 'Past runs could not be read',
          detail: state.results.error,
        });
        return replaceChildren(host, head, fill);
      }
      if (!state.results.list) {
        renderState(fill, { icon: 'history', message: 'Reading past runs…', detail: '' });
        return replaceChildren(host, head, fill);
      }
      if (!all.length) {
        renderState(fill, {
          icon: 'history',
          message: 'No stored run yet',
          detail: 'A run is stored here when it ends.',
        });
        return replaceChildren(host, head, fill);
      }
      if (!rows.length) {
        renderState(fill, {
          icon: 'search',
          message: 'No past run matches the filter',
          detail: '',
        });
        return replaceChildren(host, h('div', { class: 'opt-hist-tools' }, head, clear), fill);
      }
      replaceChildren(
        host,
        h('div', { class: 'opt-hist-tools' }, head, clear),
        state.results.error
          ? h(
              'div',
              { class: 'osprey-banner osprey-banner--warn', role: 'status' },
              h('span', {
                class: 'osprey-banner-text',
                text: `The list could not be read again: ${state.results.error}`,
              })
            )
          : null,
        h(
          'div',
          { class: 'opt-hist-scroll' },
          h(
            'table',
            { class: 'opt-hist-table' },
            h(
              'thead',
              null,
              h(
                'tr',
                null,
                h('th', { scope: 'col', text: 'Started' }),
                h('th', { scope: 'col', text: 'Objective' }),
                h('th', { scope: 'col', text: 'Environment' }),
                h('th', { scope: 'col', class: 'opt-hist-opt', text: 'Model' }),
                h('th', { scope: 'col', class: 'opt-hist-opt', text: 'Job' })
              )
            ),
            h('tbody', null, ...rows.map(row))
          )
        )
      );
    },
  };
}
