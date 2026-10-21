// @ts-check
/**
 * plots — the Raw BPM Signal and Trajectory plots: the raw buffer of each
 * new shot drawn through the render adapter, and the plot-range inputs.
 *
 * The data flow:
 * - a `shot` stream event is the only source of a new shot; each `shot_id` is
 *   drawn once. Its raw buffer is fetched, then the Raw tab's three plots are
 *   drawn (only while `view.plotRaw` is set) and the Trajectories and
 *   Difference to Reference four are drawn from the shot with the buffer's
 *   units;
 * - the first snapshot, and every change of the session holder, fetches the
 *   newest stored shot (the head of GET shots) and draws it the same way;
 * - a change of the snapshot's reference (name and sha) or of the solve's
 *   excluded BPMs / correctors redraws the four from the last fetched data,
 *   without a fetch; any other snapshot change draws nothing;
 * - a redraw request in the view store redraws from the last data;
 * - the `range-turns`, `range-sum`, `range-x` and `range-y` inputs set the
 *   adapter's ranges, which rebuilds the drawn plots without a fetch.
 *
 * Only the newest requested shot is drawn: a buffer that arrives after a
 * newer shot was asked for is dropped.
 *
 * Mounted into the `plots` area by the composition root (`panel.js`); this
 * module's {@link mount} is its only entry point.
 *
 * @module plots
 */

import { differenceFigures, rawFigures, trajectoryFigures } from './figures.js';
import { PLOT_HOSTS, h, noticeFromResult } from './views.js';

/** The Raw tab's figure keys -> plot host ids. */
export const RAW_PLOTS = Object.freeze({
  sum: PLOT_HOSTS.raw[0],
  x: PLOT_HOSTS.raw[1],
  y: PLOT_HOSTS.raw[2],
});

/** The Trajectory Correction tab's four figure keys -> plot host ids. */
export const TRAJECTORY_PLOTS = Object.freeze({
  trajX: PLOT_HOSTS.correction[0],
  trajY: PLOT_HOSTS.correction[1],
  diffX: PLOT_HOSTS.correction[2],
  diffY: PLOT_HOSTS.correction[3],
});

/** The plot-range inputs: their range key, `data-control` name and label. */
export const RANGE_INPUTS = Object.freeze([
  Object.freeze({ key: 'turns', name: 'range-turns', label: 'Plot Turns' }),
  Object.freeze({ key: 'sum', name: 'range-sum', label: 'Plot Sum' }),
  Object.freeze({ key: 'x', name: 'range-x', label: 'Plot X' }),
  Object.freeze({ key: 'y', name: 'range-y', label: 'Plot Y' }),
]);

/**
 * A range field's text as a range: two finite numbers separated by a comma,
 * a colon or white space, `[lo, hi]` with `lo < hi` after ordering. Empty
 * text clears the range (`null`, the builder's default); anything else is
 * `undefined` (not a range).
 *
 * @param {string} text
 * @returns {import('./figures.js').Range | null | undefined}
 */
export function parseRange(text) {
  const trimmed = text.trim();
  if (trimmed === '') return null;
  const parts = trimmed.split(/\s*[,:]\s*|\s+/).filter((p) => p !== '');
  if (parts.length !== 2) return undefined;
  const [a, b] = parts.map(Number);
  if (!Number.isFinite(a) || !Number.isFinite(b) || a === b) return undefined;
  return a < b ? [a, b] : [b, a];
}

/**
 * What a snapshot change that redraws the four is keyed on: the reference's
 * name and sha, and the solve's excluded BPMs and correctors.
 *
 * @param {any} snapshot
 * @returns {string}
 */
export function trajectoryKey(snapshot) {
  const reference = snapshot?.reference;
  const config = snapshot?.solve?.config;
  return JSON.stringify([
    reference?.name ?? null,
    reference?.sha256 ?? null,
    config?.excluded_bpms ?? [],
    config?.excluded_cms ?? [],
  ]);
}

/** @param {any} snapshot @returns {string} */
function holderKey(snapshot) {
  const holder = snapshot?.holder;
  return JSON.stringify(holder ? [holder.owner ?? null, holder.kind ?? null] : null);
}

/**
 * The range inputs.
 * @param {(key: string, range: import('./figures.js').Range | null) => void} onRange
 * @returns {HTMLElement}
 */
export function rangesView(onRange) {
  return h('div', { class: 'controls', 'data-view': 'plot-ranges' },
    ...RANGE_INPUTS.map(({ key, name, label }) => {
      const input = /** @type {HTMLInputElement} */ (h('input', {
        type: 'text', name, 'data-control': name, placeholder: 'auto', 'aria-label': label,
      }));
      input.addEventListener('change', () => {
        const range = parseRange(input.value);
        if (range === undefined) {
          input.setAttribute('aria-invalid', 'true');
          return;
        }
        input.removeAttribute('aria-invalid');
        onRange(key, range);
      });
      return h('label', { class: 'fact' }, h('span', { class: 'fact-label', text: label }), input);
    }));
}

/**
 * Fill the `plots` area.
 * @type {import('./panel.js').Mount}
 */
export const mount = (host, ctx) => {
  const { api, adapter, view } = ctx;
  /** @type {Set<string>} */
  const seen = new Set();
  /** The last fetched shot and its raw buffer (null when it could not be read). */
  /** @type {{shot: any, raw: any} | null} */
  let last = null;
  /** Bumped by each fetch; a buffer answered to an older one is dropped. */
  let request = 0;
  let firstSnapshot = true;
  let lastHolder = holderKey(null);
  let lastTrajectory = trajectoryKey(null);

  function drawRaw() {
    if (!last?.raw || !view.get('plotRaw')) return;
    const { shot, raw } = last;
    adapter.renderGroup(RAW_PLOTS, (/** @type {any} */ theme, /** @type {any} */ options) =>
      rawFigures(raw, shot, theme, options));
  }

  function drawTrajectories() {
    if (!last) return;
    const { shot, raw } = last;
    const snapshot = ctx.snapshot();
    const plane = { catalog: snapshot?.catalog ?? null, reference: snapshot?.reference ?? null };
    const units = raw?.header?.units ?? {};
    adapter.renderGroup(TRAJECTORY_PLOTS, (/** @type {any} */ theme, /** @type {any} */ options) => {
      const traj = trajectoryFigures(shot, plane, theme, { units, ranges: options.ranges });
      const diff = differenceFigures(shot, plane, theme, { units, ranges: options.ranges });
      return { trajX: traj.x, trajY: traj.y, diffX: diff.x, diffY: diff.y };
    });
  }

  /** @param {any} shot */
  async function show(shot) {
    if (!shot?.shot_id) return;
    seen.add(shot.shot_id);
    const mine = ++request;
    const result = await api.shotRaw(shot.shot_id);
    if (mine !== request) return;
    if (result.kind !== 'ok') ctx.notify(noticeFromResult('raw buffer', result));
    last = { shot, raw: result.kind === 'ok' ? result.body : null };
    drawRaw();
    drawTrajectories();
  }

  async function showHead() {
    const mine = ++request;
    const result = await api.shots();
    if (mine !== request) return;
    if (result.kind !== 'ok') {
      ctx.notify(noticeFromResult('shots', result));
      return;
    }
    const head = result.body?.shots?.[0];
    if (head) await show(head);
  }

  ctx.onEvent('shot', (payload) => {
    const shot = payload?.shot;
    if (!shot?.shot_id || seen.has(shot.shot_id)) return;
    void show(shot);
  });

  ctx.onSnapshot((snapshot) => {
    const holder = holderKey(snapshot);
    const trajectory = trajectoryKey(snapshot);
    const holderChanged = holder !== lastHolder;
    const trajectoryChanged = trajectory !== lastTrajectory;
    lastHolder = holder;
    lastTrajectory = trajectory;
    if (firstSnapshot || holderChanged) {
      firstSnapshot = false;
      void showHead();
    } else if (trajectoryChanged) {
      drawTrajectories();
    }
  });

  view.on('redraw', () => {
    drawRaw();
    drawTrajectories();
  });

  host.replaceChildren(rangesView((key, range) => adapter.setRanges({ [key]: range })));
};
