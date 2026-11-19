// @ts-check
/**
 * The run page's inspector: one card per knob, then the run's details.
 *
 * During a run each card names the knob, shows its setpoint in the last
 * evaluation recorded and its change from the start, and draws the knob bar
 * zoomed to the run's search window with the start, the best point so far
 * and the last point recorded. The runner writes a point before it records
 * it, so the last record is not proof of what is on the machine, and the
 * card does not say it is. While the run waits for its write approval the
 * cards list what the request covers: each channel with the range the
 * request allows and its value at the start of the run (read when the run
 * started; it may be older than now).
 *
 * Below the cards: where the knobs are as far as the records prove it, the
 * write approval (its state, whose it is, where it was approved, the writes
 * it allows, until when), what happens at the end, and the settings and
 * revision of the run.
 *
 * The cards are built when the knobs change and their values move in place
 * after that; the settings fold keeps its open state.
 *
 * @module run-knobs
 */

import { authorizationScope } from './approval-model.js';
import { fmt, h } from './hyperscript.js';
import { knobBar, knobLabel, knobLegend, knobName, updateKnobBar } from './knob-bar.js';
import { finite } from './run-model.js';
import { approvalRows, machineNow, runKind, settingsRows } from './run-page-model.js';
import { endWords, signed } from './words.js';

/**
 * @typedef {import('./store.js').State} State
 * @typedef {import('./store.js').Screen} Screen
 * @typedef {import('./view-host.js').ViewContext} ViewContext
 * @typedef {import('./approval-model.js').EnvironmentDetails} EnvironmentDetails
 * @typedef {import('./knob-bar.js').KnobBarSpec} KnobBarSpec
 *
 * One card: the knob's identifier, the variable it is (null: a channel of
 * the request that names no tuned variable), and its track.
 * @typedef {{ name: string, variable: string|null, lo: number, hi: number }} Card
 */

/**
 * The search window of one variable: the job's bounds, else the stored
 * override, else the variable's safe range.
 *
 * @param {Screen} screen
 * @param {string} name
 * @param {any} meta
 * @returns {[number, number]|null}
 */
function windowOf(screen, name, meta) {
  const bounds = screen.job?.variable_bounds?.[name];
  const o = screen.config?.variable_overrides?.[name];
  const pairs = [
    Array.isArray(bounds) ? bounds : [],
    [o?.min, o?.max],
    [meta?.min_value, meta?.max_value],
  ];
  for (const [a, b] of pairs) {
    const lo = finite(a);
    const hi = finite(b);
    if (lo !== null && hi !== null) return [lo, hi];
  }
  return null;
}

/**
 * The cards of the screen: the request's channels while the run waits for
 * them to be approved, else the run's variables.
 *
 * @param {Screen} screen
 * @param {EnvironmentDetails|null} env
 * @returns {Card[]}
 */
function knobCards(screen, env) {
  const variables = screen.config?.variables || screen.job?.variable_names || [];
  const scope = authorizationScope(screen.job?.write_authorization);
  if (runKind(screen) === 'waiting' && scope?.channels.length) {
    const metas = Object.entries(env?.variables || {});
    return scope.channels.map(({ channel, min, max }) => {
      const found = variables.includes(channel)
        ? channel
        : (metas.find(
            ([n, m]) => /** @type {any} */ (m).setpoint_pv === channel && variables.includes(n)
          )?.[0] ?? null);
      return { name: channel, variable: found, lo: min, hi: max };
    });
  }
  /** @type {Card[]} */
  const cards = [];
  for (const name of variables) {
    const range = windowOf(screen, name, env?.variables?.[name]);
    if (range) cards.push({ name, variable: name, lo: range[0], hi: range[1] });
  }
  return cards;
}

/**
 * Mount the inspector of the run page.
 *
 * @param {ViewContext} ctx
 */
export function mountRunInspector(ctx) {
  const legend = h('div', { class: 'opt-kn-legend' });
  const list = h('div', { class: 'osprey-dev-table', role: 'list' });
  const empty = h('p', {
    class: 'opt-note',
    text: 'The knobs show once the run’s settings are read.',
  });
  const machine = h('p', { class: 'opt-rp-detail-text' });
  const approval = h('dl', { class: 'opt-rp-dl' });
  const approvalSection = section('Write approval', approval);
  const end = h('p', { class: 'opt-rp-detail-text' });
  const endSection = section('At the end', end);
  const settings = h('dl', { class: 'opt-rp-dl' });
  const settingsAside = h('span', { class: 'osprey-aside' });
  const fold = h(
    'details',
    { class: 'osprey-folder opt-rp-fold' },
    h('summary', null, 'Settings of this run', settingsAside),
    settings
  );
  const details = h(
    'div',
    { class: 'opt-rp-details' },
    section('The knobs now', machine),
    approvalSection,
    endSection,
    fold
  );

  let frame = '';
  let cardsKey = '';
  let detailsKey = '';
  /** @type {{ card: Card, row: HTMLElement, value: HTMLElement, delta: HTMLElement, bar: HTMLElement, spec: KnobBarSpec, marks: string }[]} */
  let drawn = [];

  /**
   * @param {string} title
   * @param {Node} body
   */
  function section(title, body) {
    return h(
      'section',
      { class: 'opt-rp-detail' },
      h('span', { class: 'osprey-kicker', text: title }),
      body
    );
  }

  /**
   * @param {State} state
   * @param {Screen} screen
   * @param {{ env: EnvironmentDetails|null, revision: string|null, objective: string, environment: string, now?: number }} extra
   */
  function update(state, screen, extra) {
    const { env } = extra;
    const kind = runKind(screen);
    const waiting = kind === 'waiting';
    const cards = knobCards(screen, env);
    // While the run waits, the cards are the request's channels once it names them.
    const request =
      waiting && !!authorizationScope(screen.job?.write_authorization)?.channels.length;
    const nextFrame = JSON.stringify([waiting, request, cards.length]);
    if (nextFrame !== frame) {
      frame = nextFrame;
      legend.replaceChildren(
        waiting ? knobLegend(['start']) : knobLegend(['start', 'best', 'recorded']),
        waiting
          ? h('span', {
              class: 'opt-kn-legend-note',
              text: request
                ? 'each bar is the range the request allows'
                : 'each bar is the search window',
            })
          : ''
      );
      const count = `${cards.length} channel${cards.length === 1 ? '' : 's'}`;
      ctx.shell.fillInspector({
        title: request ? 'What the request covers' : 'Knobs',
        desc: request ? count : 'inside each search window',
        body: [legend, list, empty, details],
      });
    }
    empty.hidden = cards.length > 0;
    legend.hidden = cards.length === 0;

    const f = screen.figures;
    const nextCards = JSON.stringify([waiting, cards, !!env]);
    if (nextCards !== cardsKey) {
      cardsKey = nextCards;
      drawn = cards.map((card) => {
        const meta = card.variable ? env?.variables?.[card.variable] : null;
        const value = h('span', { class: 'opt-kn-val' });
        const delta = h('span', { class: 'opt-kn-delta' });
        /** @type {KnobBarSpec} */
        const spec = { lo: card.lo, hi: card.hi, units: meta?.units || '' };
        const bar = knobBar(spec);
        const row = h(
          'div',
          { class: 'opt-kn', role: 'listitem', 'data-knob': card.name },
          knobName(card.name, meta),
          value,
          delta,
          h('div', { class: 'opt-kn-bar' }, bar)
        );
        return { card, row, value, delta, bar, spec, marks: '' };
      });
      list.replaceChildren(...drawn.map((d) => d.row));
    }

    for (const d of drawn) {
      const v = d.card.variable;
      const meta = v ? env?.variables?.[v] : null;
      const { title } = knobLabel(d.card.name, meta);
      const start = v ? finite(f?.initial?.[v]) : null;
      const unit = meta?.units ? `\u00a0${meta.units}` : '';
      if (waiting) {
        const text = start === null ? '' : `${fmt(start, 5)}${unit}`;
        if (d.value.textContent !== text) d.value.textContent = text;
        const range = `${fmt(d.card.lo, 4)} to ${fmt(d.card.hi, 4)}`;
        if (d.delta.textContent !== range) d.delta.textContent = range;
        moveBar(d, {
          ...d.spec,
          marks: { start },
          label: `${title}: ${request ? 'the request allows' : 'search window'} ${range}${start === null ? '' : `, start ${fmt(start, 5)}`}`,
        });
        continue;
      }
      const best = v ? finite(f?.bestRecord?.[v]) : null;
      const current = v ? finite(f?.lastRecord?.[v]) : null;
      const value = current ?? start;
      const text = value === null ? '' : `${fmt(value, 5)}${unit}`;
      if (d.value.textContent !== text) d.value.textContent = text;
      const moved = value !== null && start !== null ? signed(value - start) : '';
      if (d.delta.textContent !== moved) d.delta.textContent = moved;
      const said = [
        current === null ? null : `last recorded ${fmt(current, 5)}`,
        best === null ? null : `best so far ${fmt(best, 5)}`,
        start === null ? null : `start ${fmt(start, 5)}`,
      ].filter(Boolean);
      moveBar(d, {
        ...d.spec,
        marks: { start, best, current },
        label: `${title}: ${said.join(', ')}`,
      });
    }

    const scope = authorizationScope(screen.job?.write_authorization);
    const rows = approvalRows(scope, {
      sessionOwner: state.owner,
      now: extra.now,
      reason: screen.job?.write_authorization?.reason ?? null,
    });
    const settingRows = settingsRows(screen.config, {
      figures: f,
      revision: extra.revision,
      objective: extra.objective,
      environment: extra.environment,
    });
    const onComplete = screen.config?.on_complete || null;
    const nextDetails = JSON.stringify([
      machineNow(screen),
      rows,
      settingRows,
      onComplete,
      extra.revision,
    ]);
    if (nextDetails !== detailsKey) {
      detailsKey = nextDetails;
      machine.textContent = machineNow(screen);
      fillRows(approval, rows);
      approvalSection.hidden = rows.length === 0;
      end.textContent = onComplete ? endWords(onComplete).sentence || onComplete : '';
      endSection.hidden = !onComplete;
      fillRows(settings, settingRows);
      settingsAside.textContent = extra.revision ? `revision ${extra.revision.slice(0, 10)}` : '';
      fold.hidden = settingRows.length === 0;
    }
  }

  return { update };
}

/**
 * @param {HTMLElement} dl
 * @param {[string, string][]} rows
 */
function fillRows(dl, rows) {
  dl.replaceChildren(
    ...rows.flatMap(([label, value]) => [h('dt', { text: label }), h('dd', { text: value })])
  );
}

/**
 * Move a card's marks, when they moved.
 *
 * @param {{ bar: HTMLElement, marks: string }} drawn
 * @param {KnobBarSpec} spec
 */
function moveBar(drawn, spec) {
  const key = JSON.stringify([spec.now, spec.marks, spec.label]);
  if (key === drawn.marks) return;
  drawn.marks = key;
  updateKnobBar(drawn.bar, spec);
}
