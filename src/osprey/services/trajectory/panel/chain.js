// @ts-check
/**
 * chain — the Beam Injection block: readout mode, shots and turns, one row
 * per declared chain step, Run Selected and the Manual/Continuous toggle.
 *
 * The rows are `Snapshot.chain` in position order, each ticked by its
 * declared default until the operator changes it, followed by the panel-only
 * Plot BPM Readings row: its tick is `view.plotRaw` and its Run asks the
 * plots to redraw; it never reaches a route. Run Selected posts the ticked
 * declared steps as one chain body; a row's Run posts that step alone; the
 * toggle starts a continuous run with the same body (the composition root
 * owns its heartbeat) and stops it while one is running.
 *
 * Nothing runs without a current setup: while `Snapshot.setup` is null or
 * void every run control and the toggle's start are disabled with
 * {@link SETUP_FIRST}. TbT corrects on the first turn, so its turns field is
 * a fixed read-only 1; pseudo-orbit averages over an editable number of
 * turns bounded by the setup's turns tracked. Shots are bounded by
 * `Snapshot.n_shots_max`.
 *
 * Mounted into the `chain` area by the composition root (`panel.js`); this
 * module's {@link mount} is its only entry point.
 *
 * @module chain
 */

import { control, h, noticeFromResult } from './views.js';

/** Why a run control is disabled while no current setup holds. */
export const SETUP_FIRST = 'Setup BPMs first';

/** Why a run control is disabled while this tab holds no session. */
export const NO_SESSION = 'Open a session first';

/** Why a one-off run is disabled while a continuous run is in progress. */
export const CONTINUOUS_RUNNING = 'Stop the continuous run first';

/** Why Run Selected is disabled with nothing ticked. */
export const NOTHING_TICKED = 'Tick a step first';

/** The TbT turns field's tooltip. */
export const TBT_TURNS_TIP = 'TbT corrects on the first turn';

/** The readout modes, as the chain body names them, with their labels. */
export const READOUTS = Object.freeze([
  { value: 'tbt', label: 'TbT' },
  { value: 'pseudo_orbit', label: 'Pseudo orbit' },
]);

/**
 * The panel-only Plot BPM Readings row; its names are fixed, not derived from
 * a declared chain step.
 */
export const PLOT_STEP = Object.freeze({
  id: 'plot',
  label: 'Plot BPM Readings',
  description: "Draws each shot's BPM sum, X and Y readings in the Raw BPM Signal tab.",
});

/**
 * @typedef {object} ChainStep
 * @property {string} id
 * @property {string | null} label
 * @property {string | null} description
 * @property {boolean} default
 * @property {number} position
 */

/**
 * @typedef {object} ChainInputs
 * @property {'tbt' | 'pseudo_orbit'} readout
 * @property {number} nShots
 * @property {number} nTurns  The pseudo-orbit turns; TbT always sends 1.
 */

/**
 * The rows the block shows: the declared chain in position order, then the
 * plot row.
 * @param {ChainStep[] | null | undefined} chain
 * @returns {{id: string, label: string | null, description: string | null, plot: boolean}[]}
 */
export function chainRows(chain) {
  const steps = [...(chain ?? [])].sort((a, b) => a.position - b.position);
  return [
    ...steps.map((s) => ({ id: s.id, label: s.label, description: s.description, plot: false })),
    { ...PLOT_STEP, plot: true },
  ];
}

/**
 * Why a declared step cannot run now, or null when it can.
 * @param {any} snapshot
 * @param {boolean} holding
 * @returns {string | null}
 */
export function runReason(snapshot, holding) {
  if (!holding) return NO_SESSION;
  const setup = snapshot?.setup;
  if (setup == null || setup.void) return SETUP_FIRST;
  return null;
}

/**
 * *text* as a whole count within [1, *max*]; a null *max* leaves it unbounded.
 * @param {string} text
 * @param {number | null | undefined} max
 * @returns {number}
 */
export function clampCount(text, max) {
  const value = Math.trunc(Number(text));
  const atLeastOne = Number.isFinite(value) && value >= 1 ? value : 1;
  return max != null && max >= 1 ? Math.min(atLeastOne, max) : atLeastOne;
}

/**
 * The most turns a pseudo-orbit readout may average over: the current
 * setup's turns tracked, null without a setup.
 * @param {any} snapshot
 * @returns {number | null}
 */
export function turnsBound(snapshot) {
  return snapshot?.setup?.turns_tracked ?? null;
}

/**
 * The chain body for *steps* under *inputs*.
 * @param {string[]} steps
 * @param {ChainInputs} inputs
 * @returns {import('./api.js').ChainBody}
 */
export function chainBody(steps, inputs) {
  return {
    steps,
    readout: inputs.readout,
    n_shots: inputs.nShots,
    n_turns: inputs.readout === 'tbt' ? 1 : inputs.nTurns,
  };
}

/**
 * Sets or clears a control's disabled reason, as {@link control} renders it.
 * @param {HTMLButtonElement} button
 * @param {string | null} reason
 */
function setReason(button, reason) {
  button.disabled = reason != null;
  if (reason == null) {
    button.removeAttribute('title');
    button.removeAttribute('data-reason');
  } else {
    button.title = reason;
    button.setAttribute('data-reason', reason);
  }
}

/**
 * A step's Options: a native disclosure showing its description.
 * @param {string} id
 * @param {string} description
 * @returns {HTMLElement}
 */
export function optionsView(id, description) {
  return h('details', { class: 'chain-options', 'data-control': `step-${id}-options` },
    h('summary', { text: 'Options' }),
    h('p', { class: 'chain-description', text: description }));
}

/**
 * Fill the `chain` area.
 * @type {import('./panel.js').Mount}
 */
export const mount = (host, ctx) => {
  /** @type {ChainInputs} */
  const inputs = { readout: 'tbt', nShots: 1, nTurns: 1 };
  /** The operator's ticks by step id; a step not in it shows its default. */
  /** @type {Map<string, boolean>} */
  const ticks = new Map();
  /** @type {ChainStep[]} */
  let declared = [];
  let rowsKey = '';
  /** @type {HTMLButtonElement[]} */
  let stepRuns = [];
  /** @type {HTMLInputElement | null} */
  let plotBox = null;

  const running = () => ctx.snapshot()?.continuous != null;
  const ticked = (/** @type {ChainStep} */ s) => ticks.get(s.id) ?? s.default;
  const selected = () =>
    [...declared].sort((a, b) => a.position - b.position).filter(ticked).map((s) => s.id);

  /**
   * @param {string} label
   * @param {() => Promise<import('./api.js').Result>} call
   */
  async function act(label, call) {
    const result = await call();
    ctx.notify(noticeFromResult(label, result));
  }

  /** @param {string[]} steps */
  const shoot = (steps) => void act('shot', () => ctx.api.shot(chainBody(steps, inputs)));

  const readout = /** @type {HTMLSelectElement} */ (h('select', { 'data-control': 'chain-readout-mode',
    onchange: () => {
      inputs.readout = readout.value === 'pseudo_orbit' ? 'pseudo_orbit' : 'tbt';
      refresh();
    } }, ...READOUTS.map((r) => h('option', { value: r.value, text: r.label }))));
  readout.value = inputs.readout;

  const shots = /** @type {HTMLInputElement} */ (h('input', { type: 'number', min: '1', step: '1',
    'data-control': 'chain-n-shots', value: '1',
    onchange: () => {
      inputs.nShots = clampCount(shots.value, ctx.snapshot()?.n_shots_max);
      refresh();
    } }));

  const turns = /** @type {HTMLInputElement} */ (h('input', { type: 'number', min: '1', step: '1',
    'data-control': 'chain-n-turns', value: '1',
    onchange: () => {
      if (inputs.readout === 'pseudo_orbit') inputs.nTurns = clampCount(turns.value, turnsBound(ctx.snapshot()));
      refresh();
    } }));

  const toggle = control('chain-continuous', 'Manual', () => {
    if (running()) void ctx.stopContinuous();
    else void ctx.startContinuous(chainBody(selected(), inputs));
  });

  const runSelected = control('run-selected', 'Run Selected', () => shoot(selected()));
  const rows = h('div', { class: 'chain-rows' });

  host.replaceChildren(
    h('div', { class: 'chain-header' },
      h('label', null, 'Readout mode ', readout),
      h('label', null, 'Number of shots ', shots),
      h('label', null, 'Number of turns ', turns),
      toggle),
    rows,
    h('div', { class: 'chain-footer' }, runSelected));

  /** @param {ReturnType<typeof chainRows>[number]} row */
  function rowView(row) {
    const box = /** @type {HTMLInputElement} */ (h('input', { type: 'checkbox', 'data-control': `step-${row.id}` }));
    let run;
    if (row.plot) {
      box.checked = Boolean(ctx.view.get('plotRaw'));
      box.onchange = () => ctx.view.set('plotRaw', box.checked);
      plotBox = box;
      run = control(`step-${row.id}-run`, 'Run', () => ctx.view.requestRedraw());
    } else {
      const step = /** @type {ChainStep} */ (declared.find((s) => s.id === row.id));
      box.checked = ticked(step);
      box.onchange = () => {
        ticks.set(row.id, box.checked);
        refresh();
      };
      run = control(`step-${row.id}-run`, 'Run', () => shoot([row.id]));
      stepRuns.push(run);
    }
    return h('div', { class: 'chain-row', 'data-step': row.id },
      h('label', null, box, ` ${row.label ?? row.id}`),
      run,
      row.description ? optionsView(row.id, row.description) : null);
  }

  function rebuildRows() {
    const shown = chainRows(declared);
    const key = JSON.stringify(shown);
    if (key === rowsKey) return;
    rowsKey = key;
    stepRuns = [];
    rows.replaceChildren(...shown.map(rowView));
  }

  function refresh() {
    const snapshot = ctx.snapshot();
    declared = snapshot?.chain ?? [];
    rebuildRows();

    const maxShots = snapshot?.n_shots_max ?? null;
    if (maxShots != null) shots.max = String(maxShots);
    else shots.removeAttribute('max');
    inputs.nShots = clampCount(String(inputs.nShots), maxShots);
    shots.value = String(inputs.nShots);

    readout.value = inputs.readout;
    if (inputs.readout === 'tbt') {
      turns.readOnly = true;
      turns.title = TBT_TURNS_TIP;
      turns.value = '1';
      turns.removeAttribute('max');
    } else {
      const bound = turnsBound(snapshot);
      turns.readOnly = false;
      turns.removeAttribute('title');
      if (bound != null) turns.max = String(bound);
      else turns.removeAttribute('max');
      inputs.nTurns = clampCount(String(inputs.nTurns), bound);
      turns.value = String(inputs.nTurns);
    }

    const reason = runReason(snapshot, ctx.holding());
    const oneOff = reason ?? (running() ? CONTINUOUS_RUNNING : null);
    for (const run of stepRuns) setReason(run, oneOff);
    setReason(runSelected, oneOff ?? (selected().length === 0 ? NOTHING_TICKED : null));

    const on = running();
    toggle.textContent = on ? 'Continuous' : 'Manual';
    toggle.setAttribute('aria-pressed', String(on));
    setReason(toggle, on ? null : reason ?? (selected().length === 0 ? NOTHING_TICKED : null));
  }

  ctx.view.on('plotRaw', (value) => {
    if (plotBox) plotBox.checked = Boolean(value);
  });
  ctx.onSnapshot(() => refresh());
  refresh();
};
