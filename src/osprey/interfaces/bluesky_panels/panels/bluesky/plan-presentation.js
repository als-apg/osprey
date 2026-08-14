// @ts-check
/**
 * Per-plan presentation for the shipped scan plans: how their parameter form is
 * laid out, and what the live readout above the footer says.
 *
 * Both tables are keyed by plan name and both are OPTIONAL. A plan with no
 * entry still renders and still reads out — `planLayout` returns `undefined`,
 * which auto-flows every schema field, and `summarizePlanArgs` falls back to a
 * plain field count. That is what lets a facility or session plan be usable
 * with no panel-side code at all, so nothing here may ever become a gate on
 * whether a plan can be composed or queued.
 *
 * Kept separate from plans-view.js because these are pure data and pure functions
 * over collected `plan_args` — no DOM, no fetches, no panel state — and because
 * adding a plan should mean adding an entry here, not editing the panel.
 *
 * @module plan-presentation
 */

/**
 * Per-plan 2-D field placement: rows of side-by-side field names, passed to
 * renderSchemaForm as its layout option. Names not in the plan's schema are
 * ignored; schema fields missing here are auto-flowed after these rows —
 * so this is a presentation hint, never a gate.
 *
 * @type {Record<string, string[][]>}
 */
const PLAN_LAYOUTS = {
  orm: [
    ['correctors', 'detectors'],
    ['span_a', 'num'],
    ['sweep'],
  ],
  grid_scan: [['axes'], ['detectors', 'snake_axes']],
};

/**
 * Per-plan live readout, computed from the currently-collected plan_args on
 * every edit. Returns '' when there is nothing meaningful to show yet.
 *
 * @type {Record<string, (args: Record<string, any>) => string>}
 */
const PLAN_SUMMARIES = {
  orm(args) {
    const c = Array.isArray(args.correctors) ? args.correctors.length : 0;
    const d = Array.isArray(args.detectors) ? args.detectors.length : 0;
    const n = typeof args.num === 'number' ? args.num : 0;
    const span = typeof args.span_a === 'number' ? args.span_a : null;
    /** @type {string[]} */
    const parts = [];
    const sweep = typeof args.sweep === 'string' ? args.sweep : null;
    if (c) parts.push(`${c} corrector${c === 1 ? '' : 's'}`);
    if (d) parts.push(`${d} BPM${d === 1 ? '' : 's'}`);
    if (span !== null) {
      // Monodirectional sweeps [0, +span]; bidirectional the symmetric ±span.
      parts.push(sweep === 'monodirectional' ? `0…${span} A` : `±${span} A`);
    }
    if (c && n) parts.push(`${c} × ${n} = ${c * n} sweep points`);
    else if (n) parts.push(`${n} points`);
    return parts.join(' · ');
  },
  grid_scan(args) {
    const axes = Array.isArray(args.axes) ? args.axes : [];
    const d = Array.isArray(args.detectors) ? args.detectors.length : 0;
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
    if (d) parts.push(`${d} detector${d === 1 ? '' : 's'}`);
    return parts.join(' · ');
  },
};

/**
 * The 2-D field placement for a plan, or `undefined` to auto-flow every field.
 *
 * @param {string} planName
 * @returns {string[][]|undefined}
 */
export function planLayout(planName) {
  return PLAN_LAYOUTS[planName];
}

/**
 * The live readout line for a plan's currently-collected args, or '' when
 * there is nothing worth showing yet.
 *
 * A plan with its own summary gets it (e.g. "2 correctors · ±5 A · 2 × 7 = 14
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
