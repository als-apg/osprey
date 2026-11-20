// @ts-check
/**
 * The live readout above the plan form's footer: what the plan's collected
 * args amount to, recomputed on every edit.
 *
 * The readout never gates anything: a plan with no entry here reads out a
 * plain field count, so a facility or session plan is usable with no
 * panel-side code. Pure functions over collected `plan_args` — no DOM, no
 * fetches, no panel state.
 *
 * @module plan-presentation
 */

/**
 * Per-plan live readout, computed from the currently-collected plan_args on
 * every edit. Returns '' when there is nothing meaningful to show yet.
 *
 * @type {Record<string, (args: Record<string, any>) => string>}
 */
const PLAN_SUMMARIES = {
  orm(args) {
    const c = Array.isArray(args.correctors) ? args.correctors.length : 0;
    const d = Array.isArray(args.readbacks) ? args.readbacks.length : 0;
    const n = typeof args.num === 'number' ? args.num : 0;
    const span = typeof args.span === 'number' ? args.span : null;
    /** @type {string[]} */
    const parts = [];
    const sweep = typeof args.sweep === 'string' ? args.sweep : null;
    if (c) parts.push(`${c} corrector${c === 1 ? '' : 's'}`);
    if (d) parts.push(`${d} BPM${d === 1 ? '' : 's'}`);
    if (span !== null) {
      // Monodirectional sweeps [0, +span]; bidirectional the symmetric ±span.
      parts.push(sweep === 'monodirectional' ? `0…${span}` : `±${span}`);
    }
    if (c && n) parts.push(`${c} × ${n} = ${c * n} sweep points`);
    else if (n) parts.push(`${n} points`);
    return parts.join(' · ');
  },
  grid_scan(args) {
    const axes = Array.isArray(args.axes) ? args.axes : [];
    const d = Array.isArray(args.readbacks) ? args.readbacks.length : 0;
    /** @type {string[]} */
    const parts = [];
    if (axes.length) {
      const nums = axes.map((axis) =>
        axis && typeof axis.num_points === 'number' ? axis.num_points : 0
      );
      parts.push(`${axes.length} ax${axes.length === 1 ? 'is' : 'es'}`);
      if (nums.every((v) => v > 0)) {
        const total = nums.reduce((product, v) => product * v, 1);
        parts.push(`${nums.join(' × ')} = ${total} grid points`);
      }
    }
    if (d) parts.push(`${d} readable${d === 1 ? '' : 's'}`);
    return parts.join(' · ');
  },
  orbit_bump_sweep(args) {
    const c = Array.isArray(args.correctors) ? args.correctors.length : 0;
    const t = Array.isArray(args.targets) ? args.targets.length : 0;
    const n = typeof args.num === 'number' ? args.num : 0;
    const sweep = typeof args.sweep === 'string' ? args.sweep : null;
    /** @type {string[]} */
    const parts = [];
    if (c) parts.push(`${c} corrector${c === 1 ? '' : 's'}`);
    if (t) parts.push(`${t} target${t === 1 ? '' : 's'}`);
    if (n > 0) {
      // Every amplitude profile walks out and back down, so a sweep is twice
      // its `num` amplitudes — and bidirectional walks that on both sides.
      const steps = sweep === 'bidirectional' ? 4 * n : 2 * n;
      parts.push(`${steps} sweep points`);
    }
    if (args.best_effort === true) parts.push('best-effort');
    return parts.join(' · ');
  },
};

/**
 * The live readout line for a plan's currently-collected args, or '' when
 * there is nothing worth showing yet.
 *
 * A plan with its own summary gets it (e.g. "2 correctors · ±5 · 2 × 7 = 14
 * sweep points"); anything else — and any plan whose own summary has nothing to
 * say for these args — falls back to a field count.
 *
 * @param {string} planName
 * @param {Record<string, any>} args
 * @returns {string}
 */
export function summarizePlanArgs(planName, args) {
  const custom = PLAN_SUMMARIES[planName];
  const text = custom ? custom(args) : '';
  if (text) return text;
  const count = Object.keys(args).length;
  return count > 0 ? `${count} parameter${count === 1 ? '' : 's'} set` : '';
}
