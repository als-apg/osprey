// @ts-check
/**
 * correction — the Trajectory Correction column: loss cutoff, response
 * matrix, pseudo-inverse and SV spectrum, the CM step, Apply and Put back,
 * and the correction results under the dy plot.
 *
 * The knobs (loss cutoff, Tikhonov α, Singular Value Cutoff, gain) show the
 * solve configuration in force (`Snapshot.solve.config`) and are re-seeded
 * from every snapshot until the operator edits one; an edited knob holds its
 * pending text until Calculate Pseudo-Inverse sends it. The Singular Value
 * Cutoff is entered as the number of singular values discarded and sent as
 * `keep_live = L − n_discard`. Calculate Response Matrix sends an empty
 * partial, which recomputes the solve from the model.
 *
 * Calculate CM step fetches the pending step and the column keeps it: the
 * CM-step bars draw that step, and Apply sends that step's digest. Apply is
 * offered only while the kept step is still the pending one, the service is
 * idle, the BPM set committed through Setup BPMs (`Snapshot.solve.config`) is
 * still the one the step's shot was taken with, and the machine has not moved
 * since that shot; each refusal to offer it names the operator's next move.
 * Every shot is taken without the BPMs excluded in force, so a committed
 * exclusion never withholds Apply by itself; only a commit after the shot
 * does. A selection still pending in the setup tabs is not committed and
 * never touches the guard.
 *
 * This module is the sole renderer of `plot-sv` (from `Snapshot.solve`),
 * `plot-cm-step` (from the kept step, while the plot tick is set) and the
 * apply-result table, which lists the served `Moved` entries before → after
 * exactly as answered, never computed here.
 *
 * Mounted into the `correction` area by the composition root (`panel.js`); this
 * module's {@link mount} is its only entry point. Its results live in the
 * `correction-results` area under the dy plot.
 *
 * @module correction
 */

import { formatFacilityTime } from '/design-system/js/facility-time.js';
import { canPutBack, putBackBody } from './api.js';
import { cmStepFigure, svFigure } from './figures.js';
import { control, h, noticeFromResult } from './views.js';

/** Why a control that needs the session is disabled without one. */
export const NO_SESSION = 'Open a session first';

/** Why Apply is disabled before a step was calculated. */
export const NO_STEP = 'Calculate CM step first';

/** Why Apply is disabled once the pending step is no longer the kept one. */
export const STEP_CHANGED = 'step changed: Calculate CM step again';

/** Why Apply is disabled once the machine moved since the kept step's shot. */
export const MACHINE_MOVED = 'machine moved since this shot: Run Selected, then calculate';

/** Why Apply is disabled once the committed BPM set differs from the kept step's shot's. */
export const BPM_SET_CHANGED = 'BPM set changed: Run Selected';

/** Why Apply is disabled for a step that moves nothing. */
export const ZERO_STEP = 'step is all zero: nothing to apply';

/** Why an action is disabled while the service is busy, by mode. */
export const BUSY = Object.freeze({
  action: 'Wait for the running action to finish',
  continuous: 'Stop the continuous run first',
});

/** The refusal code the apply route answers when the machine moved since the shot. */
export const MACHINE_MOVED_CODE = 'machine_moved';

/**
 * The knobs, by solve-config key: their inventory name, label and tooltip.
 * @type {ReadonlyArray<Readonly<{key: KnobKey, name: string, label: string, title: string}>>}
 */
export const KNOBS = Object.freeze([
  Object.freeze({ key: /** @type {KnobKey} */ ('loss_cutoff'), name: 'solve-loss-cutoff', label: 'Loss cutoff',
    title: 'Loss cutoff (loss_cutoff)' }),
  Object.freeze({ key: /** @type {KnobKey} */ ('alpha'), name: 'solve-alpha', label: 'Tikhonov Regularization',
    title: 'Tikhonov Regularization (alpha)' }),
  Object.freeze({ key: /** @type {KnobKey} */ ('n_discard'), name: 'solve-n-discard', label: 'Singular Value Cutoff',
    title: 'Singular Value Cutoff (n_discard)' }),
  Object.freeze({ key: /** @type {KnobKey} */ ('gain'), name: 'solve-gain', label: 'Gain', title: 'Gain (gain)' }),
]);

/** @typedef {'loss_cutoff' | 'alpha' | 'n_discard' | 'gain'} KnobKey */
/** @typedef {Record<KnobKey, string | null>} KnobDrafts  Pending text per knob, null when unedited. */
/** @typedef {Record<KnobKey, string>} KnobTexts  What each knob shows. */

/** @returns {KnobDrafts} */
export function emptyKnobDrafts() {
  return { loss_cutoff: null, alpha: null, n_discard: null, gain: null };
}

/**
 * What each knob shows from a snapshot alone: the solve configuration in force.
 * @param {any} snapshot
 * @returns {KnobTexts}
 */
export function seededKnobs(snapshot) {
  const config = snapshot?.solve?.config;
  /** @param {unknown} value */
  const text = (value) => (value == null ? '' : String(value));
  return {
    loss_cutoff: text(config?.loss_cutoff),
    alpha: text(config?.alpha),
    n_discard: text(config?.n_discard),
    gain: text(config?.gain),
  };
}

/**
 * What the knobs show: each pending text, else its seed.
 * @param {KnobDrafts} drafts
 * @param {KnobTexts} seeds
 * @returns {KnobTexts}
 */
export function shownKnobs(drafts, seeds) {
  return {
    loss_cutoff: drafts.loss_cutoff ?? seeds.loss_cutoff,
    alpha: drafts.alpha ?? seeds.alpha,
    n_discard: drafts.n_discard ?? seeds.n_discard,
    gain: drafts.gain ?? seeds.gain,
  };
}

/**
 * Whether any knob holds pending text.
 * @param {KnobDrafts} drafts
 * @returns {boolean}
 */
export function editingKnobs(drafts) {
  return Object.values(drafts).some((text) => text != null);
}

/**
 * The solve partial Calculate Pseudo-Inverse sends: only the edited knobs,
 * the Singular Value Cutoff as `keep_live`; or the reason it cannot.
 * @param {KnobDrafts} drafts
 * @param {any} solve  `Snapshot.solve`, for its `L`.
 * @returns {{body: Record<string, number>} | {error: string}}
 */
export function solvePartial(drafts, solve) {
  /** @type {Record<string, number>} */
  const body = {};
  for (const knob of KNOBS) {
    const text = drafts[knob.key];
    if (text == null) continue;
    const value = Number(text);
    if (text.trim() === '' || !Number.isFinite(value)) return { error: `${knob.label}: enter a number` };
    if (knob.key !== 'n_discard') {
      body[knob.key] = value;
      continue;
    }
    const size = solve?.L;
    if (typeof size !== 'number') return { error: 'Singular Value Cutoff: Calculate Response Matrix first' };
    if (!Number.isInteger(value) || value < 0 || value > size) {
      return { error: `Singular Value Cutoff: enter a whole number from 0 to ${size}` };
    }
    body.keep_live = size - value;
  }
  return { body };
}

/**
 * The response matrix status line: its source, dimensions and when it was computed.
 * @param {any} solve  `Snapshot.solve`
 * @returns {string}
 */
export function rmStatus(solve) {
  const rm = solve?.rm;
  if (!rm) return 'No response matrix yet';
  const when = typeof rm.computed_at === 'number'
    ? `, computed ${formatFacilityTime(rm.computed_at * 1000, { hour: '2-digit', minute: '2-digit', second: '2-digit' })}`
    : '';
  return `RM from ${rm.source}: ${rm.rows} × ${rm.h_cols + rm.v_cols} (${rm.h_cols} HCM, ${rm.v_cols} VCM)${when}`;
}

/**
 * The solve summary line: how many singular values are live of how many.
 * @param {any} solve  `Snapshot.solve`
 * @returns {string}
 */
export function solveSummary(solve) {
  if (!solve) return 'No pseudo-inverse yet';
  return `${solve.live_rank} of ${solve.L} singular values live`;
}

/**
 * Whether two name lists hold the same names, in any order.
 * @param {readonly string[]} a
 * @param {readonly string[]} b
 * @returns {boolean}
 */
export function sameNames(a, b) {
  const left = new Set(a);
  const right = new Set(b);
  return left.size === right.size && [...left].every((name) => right.has(name));
}

/**
 * Why the busy service refuses an action now, or null when it is idle.
 * @param {any} snapshot
 * @returns {string | null}
 */
export function busyReason(snapshot) {
  const mode = snapshot?.mode;
  if (mode === 'action') return BUSY.action;
  if (mode === 'continuous') return BUSY.continuous;
  return null;
}

/**
 * @typedef {object} ApplyState
 * @property {any} snapshot
 * @property {any} kept  The step the column showed, or null.
 * @property {string[] | null} keptExcluded  The BPMs the kept step's shot was taken without, null when unknown.
 * @property {boolean} machineMoved  Whether the machine moved since the kept step's shot.
 * @property {boolean} holding
 */

/**
 * Why Apply cannot be used now, or null when it can.
 * @param {ApplyState} state
 * @returns {string | null}
 */
export function applyReason({ snapshot, kept, keptExcluded, machineMoved, holding }) {
  if (!holding) return NO_SESSION;
  if (kept == null) return NO_STEP;
  const busy = busyReason(snapshot);
  if (busy) return busy;
  const pending = snapshot?.step ?? null;
  const sameShot = pending == null || pending.shot_id === kept.shot_id;
  if (sameShot) {
    const committed = snapshot?.solve?.config?.excluded_bpms;
    if (keptExcluded != null && Array.isArray(committed) && !sameNames(committed, keptExcluded)) return BPM_SET_CHANGED;
    if (machineMoved) return MACHINE_MOVED;
  }
  if (pending?.digest !== kept.digest) return STEP_CHANGED;
  if (kept.all_zero) return ZERO_STEP;
  return null;
}

/**
 * Why Put back cannot be used now, or null when it can.
 * @param {any} snapshot
 * @param {boolean} holding
 * @returns {string | null}
 */
export function putBackReason(snapshot, holding) {
  if (!holding) return NO_SESSION;
  const busy = busyReason(snapshot);
  if (busy) return busy;
  if (!canPutBack(snapshot?.last_action)) return 'Nothing to put back';
  if (!putBackBody(snapshot)) return 'No model loaded';
  return null;
}

/**
 * Why a solve route call cannot be made now, or null when it can.
 * @param {any} snapshot
 * @param {boolean} holding
 * @returns {string | null}
 */
export function solveReason(snapshot, holding) {
  return holding ? busyReason(snapshot) : NO_SESSION;
}

/**
 * One served value as table text.
 * @param {unknown} value
 * @returns {string}
 */
function cell(value) {
  return value === null || value === undefined ? '—' : String(value);
}

/**
 * The apply-result table: each setting written, before → after and its
 * outcome, exactly as served.
 * @param {import('./views.js').Moved[]} moved
 * @param {string} [title]
 * @returns {HTMLElement}
 */
export function applyResultView(moved, title = 'Moved') {
  return h('section', { class: 'apply-result', 'data-control': 'apply-result' },
    h('h4', { text: title }),
    moved.length === 0
      ? h('p', { class: 'apply-result-empty', text: 'Nothing moved.' })
      : h('table', { class: 'moved' },
        h('thead', null, h('tr', null, ...['Setting', 'Before', 'After', 'Outcome'].map((t) => h('th', { text: t })))),
        h('tbody', null, ...moved.map((m) =>
          h('tr', { 'data-moved': m.name },
            h('td', { text: m.name }),
            h('td', { text: `${cell(m.before)} →` }),
            h('td', { text: cell(/** @type {any} */ (m).after) }),
            h('td', { text: m.outcome }))))));
}

/**
 * @typedef {object} CorrectionModel
 * @property {any} snapshot
 * @property {KnobDrafts} drafts
 * @property {KnobTexts} texts
 * @property {any} kept
 * @property {string[] | null} keptExcluded
 * @property {boolean} machineMoved
 * @property {boolean} plotCmStep
 * @property {boolean} holding
 * @property {string | null} message  The last refusal shown inline, or null.
 */

/**
 * @typedef {object} CorrectionHandlers
 * @property {(key: KnobKey, text: string) => void} onKnobInput  Each keystroke; no redraw.
 * @property {(key: KnobKey, text: string) => void} onKnob  A committed edit of the knob.
 * @property {() => void} onRevert
 * @property {() => void} onCalculateRm
 * @property {() => void} onCalculateInverse
 * @property {() => void} onCalculateStep
 * @property {(checked: boolean) => void} onPlotTick
 * @property {() => void} onApply
 * @property {() => void} onPutBack
 */

/**
 * One knob's number input.
 * @param {CorrectionModel} model
 * @param {(typeof KNOBS)[number]} knob
 * @param {CorrectionHandlers} handlers
 * @returns {HTMLElement}
 */
function knobField(model, knob, handlers) {
  const input = /** @type {HTMLInputElement} */ (h('input', {
    type: 'number',
    'data-control': knob.name,
    name: knob.name,
    value: model.texts[knob.key],
    title: knob.title,
    step: 'any',
    'data-pending': model.drafts[knob.key] != null ? 'true' : null,
  }));
  if (knob.key === 'n_discard') {
    input.step = '1';
    input.min = '0';
    if (typeof model.snapshot?.solve?.L === 'number') input.max = String(model.snapshot.solve.L);
  }
  input.addEventListener('input', () => handlers.onKnobInput(knob.key, input.value));
  input.addEventListener('change', () => handlers.onKnob(knob.key, input.value));
  return h('label', { class: 'correction-knob', title: knob.title }, `${knob.label} `, input);
}

/**
 * The step line: what the kept step is and how it came out.
 * @param {any} kept
 * @returns {string}
 */
export function stepSummary(kept) {
  if (kept == null) return 'No CM step calculated';
  const moved = Object.values(kept.deltas ?? {}).filter((delta) => delta !== 0).length;
  return `Step for shot ${kept.shot_id}: ${kept.status} (${moved} corrector(s), factor ${kept.factor})`;
}

/**
 * The column's controls.
 * @param {CorrectionModel} model
 * @param {CorrectionHandlers} handlers
 * @returns {HTMLElement}
 */
export function correctionView(model, handlers) {
  const solve = model.snapshot?.solve ?? null;
  const partial = solvePartial(model.drafts, solve);
  const solveBlocked = solveReason(model.snapshot, model.holding);
  const inverseReason = solveBlocked ?? ('error' in partial ? partial.error : null);
  const tick = /** @type {HTMLInputElement} */ (h('input', {
    type: 'checkbox', 'data-control': 'step-plot-bars', name: 'step-plot-bars', title: 'Plot the CM step bars',
  }));
  tick.checked = model.plotCmStep;
  tick.addEventListener('change', () => handlers.onPlotTick(tick.checked));
  const knob = (/** @type {KnobKey} */ key) => {
    const entry = KNOBS.find((k) => k.key === key);
    return entry ? knobField(model, entry, handlers) : null;
  };
  return h('div', { class: 'correction' },
    knob('loss_cutoff'),
    h('div', { class: 'correction-row' },
      control('rm-calculate', 'Calculate Response Matrix', () => handlers.onCalculateRm(), solveBlocked)),
    h('p', { 'data-control': 'rm-status', class: 'correction-status', text: rmStatus(solve) }),
    knob('alpha'),
    knob('n_discard'),
    knob('gain'),
    h('div', { class: 'correction-row' },
      control('solve-calculate', 'Calculate Pseudo-Inverse', () => handlers.onCalculateInverse(), inverseReason),
      control('solve-revert', 'Revert', () => handlers.onRevert(),
        editingKnobs(model.drafts) ? null : 'Nothing to revert')),
    h('p', { 'data-view': 'solve-summary', class: 'correction-status', text: solveSummary(solve) }),
    h('div', { class: 'correction-row' },
      control('step-calculate', 'Calculate CM step', () => handlers.onCalculateStep()),
      h('label', { class: 'correction-tick' }, tick, ' plot')),
    h('p', { 'data-view': 'step-summary', class: 'correction-status', text: stepSummary(model.kept) }),
    h('div', { class: 'correction-row' },
      control('step-apply', 'Apply CM step', () => handlers.onApply(), applyReason(model)),
      control('put-back', 'Put back', () => handlers.onPutBack(), putBackReason(model.snapshot, model.holding))),
    model.message ? h('p', { class: 'correction-error', role: 'alert', 'data-view': 'correction-message',
      text: model.message }) : null);
}

/**
 * Fill the `correction` area.
 * @type {import('./panel.js').Mount}
 */
export const mount = (host, ctx) => {
  let drafts = emptyKnobDrafts();
  let seeds = seededKnobs(ctx.snapshot());
  /** @type {any} */
  let kept = null;
  /** @type {string[] | null} */
  let keptExcluded = null;
  let machineMoved = false;
  /** @type {string | null} */
  let message = null;
  /** @type {{moved: import('./views.js').Moved[], title: string} | null} */
  let result = null;
  /** @type {string | null} */
  let drawnSha = null;
  let deferred = false;
  /** The excluded BPMs of each shot seen on the stream, by shot id. */
  /** @type {Map<string, string[]>} */
  const shotBpms = new Map();

  const controls = h('div', { class: 'correction-controls' });
  host.replaceChildren(controls);
  const results = ctx.area('correction-results') ?? host.appendChild(h('div', { class: 'correction-results' }));
  const cmPlot = h('div', { class: 'plot', id: 'plot-cm-step', 'data-control': 'plot-cm-step' });
  const resultHost = h('div', { class: 'apply-result-host' });
  results.replaceChildren(cmPlot, resultHost);

  /** Whether the operator is typing in one of this area's knobs. */
  function typing() {
    const active = host.ownerDocument?.activeElement;
    return active != null && host.contains(active) && active.tagName === 'INPUT'
      && active.getAttribute('type') !== 'checkbox';
  }

  function drawSv(force = false) {
    const solve = ctx.snapshot()?.solve ?? null;
    const sha = solve?.sha ?? null;
    if (solve == null || (!force && sha === drawnSha)) return;
    if (ctx.adapter?.render?.('plot-sv', (/** @type {any} */ theme) => svFigure(solve, theme))) drawnSha = sha;
  }

  function drawStep() {
    const shown = kept != null && ctx.view.get('plotCmStep');
    cmPlot.hidden = !shown;
    if (!shown) return;
    const catalog = ctx.snapshot()?.catalog ?? null;
    const step = kept;
    ctx.adapter?.render?.('plot-cm-step', (/** @type {any} */ theme) => cmStepFigure(step, catalog, theme));
  }

  function drawResult() {
    resultHost.replaceChildren(...(result ? [applyResultView(result.moved, result.title)] : []));
  }

  /** @param {boolean} [force]  Redraw even while the operator types. */
  function render(force = false) {
    if (!editingKnobs(drafts)) seeds = seededKnobs(ctx.snapshot());
    if (!force && typing()) {
      deferred = true;
      return;
    }
    deferred = false;
    controls.replaceChildren(correctionView({
      snapshot: ctx.snapshot(),
      drafts,
      texts: shownKnobs(drafts, seeds),
      kept,
      keptExcluded,
      machineMoved,
      plotCmStep: ctx.view.get('plotCmStep'),
      holding: ctx.holding(),
      message,
    }, handlers));
  }

  /**
   * Notify a non-ok result and show a refusal inline; answers whether it was ok.
   * @param {string} label
   * @param {import('./api.js').Result} answer
   */
  function judged(label, answer) {
    ctx.notify(noticeFromResult(label, answer));
    if (answer.kind === 'refused') {
      message = `${label} refused: ${answer.detail}`;
      if (answer.moved?.length) {
        result = { moved: answer.moved, title: `${label} failed; moved` };
        drawResult();
      }
    } else if (answer.kind === 'ok') {
      message = null;
    }
    return answer.kind === 'ok';
  }

  /** @param {Record<string, number>} partial */
  async function solve(partial, /** @type {string} */ label) {
    const sent = { ...drafts };
    const answer = await ctx.api.solve(partial);
    if (judged(label, answer)) {
      for (const knob of KNOBS) {
        if (drafts[knob.key] === sent[knob.key]) drafts = { ...drafts, [knob.key]: null };
      }
    }
    render(true);
  }

  /** The kept step's shot's excluded BPMs, from the stream or the stored shots. */
  async function lookUpShot(/** @type {string} */ shotId) {
    const seen = shotBpms.get(shotId);
    if (seen) return seen;
    const answer = await ctx.api.shots();
    if (answer.kind !== 'ok') return null;
    const found = (answer.body?.shots ?? []).find((/** @type {any} */ s) => s.shot_id === shotId);
    return Array.isArray(found?.excluded_bpms) ? found.excluded_bpms : null;
  }

  async function calculateStep() {
    const answer = await ctx.api.step();
    if (!judged('Calculate CM step', answer) || answer.kind !== 'ok') {
      render(true);
      return;
    }
    const step = answer.body;
    kept = step;
    keptExcluded = null;
    machineMoved = false;
    render(true);
    drawStep();
    const excluded = await lookUpShot(step.shot_id);
    if (kept === step) {
      keptExcluded = excluded;
      render(true);
    }
  }

  async function apply() {
    if (kept == null) return;
    const answer = await ctx.api.apply(kept.digest);
    if (answer.kind === 'refused' && answer.code === MACHINE_MOVED_CODE) machineMoved = true;
    if (judged('Apply CM step', answer) && answer.kind === 'ok') {
      result = { moved: answer.body?.moved ?? [], title: 'Applied' };
      drawResult();
    }
    render(true);
  }

  async function putBack() {
    const body = putBackBody(ctx.snapshot());
    if (body == null) return;
    const answer = await ctx.api.putBack(body);
    if (judged('Put back', answer) && answer.kind === 'ok') {
      result = { moved: answer.body?.moved ?? [], title: 'Put back' };
      drawResult();
    }
    render(true);
  }

  /** @type {CorrectionHandlers} */
  const handlers = {
    onKnobInput(key, text) {
      drafts = { ...drafts, [key]: text };
    },
    onKnob(key, text) {
      drafts = { ...drafts, [key]: text };
      render(true);
    },
    onRevert() {
      drafts = emptyKnobDrafts();
      render(true);
    },
    onCalculateRm: () => void solve({}, 'Calculate Response Matrix'),
    onCalculateInverse() {
      const partial = solvePartial(drafts, ctx.snapshot()?.solve);
      if ('error' in partial) {
        message = partial.error;
        render(true);
        return;
      }
      void solve(partial.body, 'Calculate Pseudo-Inverse');
    },
    onCalculateStep: () => void calculateStep(),
    onPlotTick: (checked) => ctx.view.set('plotCmStep', checked),
    onApply: () => void apply(),
    onPutBack: () => void putBack(),
  };

  ctx.onSnapshot(() => {
    render();
    drawSv();
  });
  ctx.onEvent('shot', (payload) => {
    const shot = payload?.shot;
    if (shot?.shot_id != null && Array.isArray(shot.excluded_bpms)) shotBpms.set(shot.shot_id, shot.excluded_bpms);
  });
  ctx.onEvent('step', () => render());
  ctx.onEvent('apply', (payload) => {
    if (kept != null) machineMoved = true;
    if (Array.isArray(payload?.moved)) {
      result = { moved: payload.moved, title: payload.step_digest == null ? 'Put back' : 'Applied' };
      drawResult();
    }
    render();
  });
  ctx.view.on('plotCmStep', () => {
    render(true);
    drawStep();
  });
  ctx.view.on('redraw', () => {
    drawSv(true);
    drawStep();
  });
  host.addEventListener('focusout', () => {
    if (deferred) queueMicrotask(() => render());
  });
  render(true);
  drawSv();
  drawStep();
};
