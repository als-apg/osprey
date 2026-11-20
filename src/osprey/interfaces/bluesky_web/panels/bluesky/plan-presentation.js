// @ts-check
/**
 * The live readout above the plan form's footer: what the plan's collected
 * args amount to, recomputed on every edit.
 *
 * A plan states its readout in its metadata `summary`: a list of parts with
 * `{parameter}` placeholders, filled by `fillSummary`. The shipped plans whose
 * readouts compute (products, plurals, a sweep's doubled points) keep them in
 * `PLAN_SUMMARIES`. A stated summary wins over that table; a plan with
 * neither reads out a plain field count. The readout never gates anything,
 * and every line it returns is text, never markup.
 *
 * Pure functions over a catalog entry and collected `plan_args` — no DOM, no
 * fetches, no panel state.
 *
 * @module plan-presentation
 */

/**
 * The shipped plans' computed readouts, by plan name: interim until a plan
 * file can compute its own line, when this table goes. Each returns '' when
 * there is nothing meaningful to show yet.
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

/** A literal brace pair, or a `{name}` placeholder (group 1). */
const SUMMARY_TOKEN = /\{\{|\}\}|\{([^{}]*)\}/g;

/**
 * What one argument shows as inside a summary part, or `null` when the part
 * has no value to show for it: absent, null, an object or an empty list.
 *
 * @param {unknown} value
 * @returns {string|null}
 */
function summaryValue(value) {
  if (Array.isArray(value)) return value.length > 0 ? String(value.length) : null;
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
    return String(value);
  }
  return null;
}

/**
 * Fill a stated summary from collected args: each part with every parameter
 * it names set is shown, `{{` and `}}` read as literal braces, a list as its
 * count; the shown parts are joined with ' · '. '' when no part is shown.
 *
 * @param {string[]} parts
 * @param {Record<string, any>} args
 * @returns {string}
 */
export function fillSummary(parts, args) {
  /** @type {string[]} */
  const shown = [];
  for (const part of parts) {
    let complete = true;
    const text = part.replace(SUMMARY_TOKEN, (token, /** @type {string|undefined} */ name) => {
      if (name === undefined) return token[0];
      const value = Object.hasOwn(args, name) ? summaryValue(args[name]) : null;
      if (value === null) complete = false;
      return value ?? '';
    });
    if (complete) shown.push(text);
  }
  return shown.join(' · ');
}

/**
 * The live readout line for a plan's currently-collected args, or '' when
 * there is nothing worth showing yet: the plan's stated summary, else its
 * shipped computed readout, else a field count.
 *
 * @param {import('./plan-browser.js').PlanSummary|undefined} plan
 * @param {Record<string, any>} args
 * @returns {string}
 */
export function summarizePlanArgs(plan, args) {
  const stated = plan?.metadata?.summary;
  let text = stated ? fillSummary(stated, args) : '';
  if (!text && !stated && plan) {
    const computed = PLAN_SUMMARIES[plan.name];
    text = computed ? computed(args) : '';
  }
  if (text) return text;
  const count = Object.keys(args).length;
  return count > 0 ? `${count} parameter${count === 1 ? '' : 's'} set` : '';
}
