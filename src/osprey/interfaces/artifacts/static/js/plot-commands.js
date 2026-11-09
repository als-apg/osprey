// @ts-check
/**
 * OSPREY Artifact Gallery — the commands a served Plotly page takes.
 *
 * The gallery draws a content strip over a `plot_html` artifact's frame
 * (artifact-strip.js) in place of Plotly's own modebar, and posts each of its
 * buttons to the frame as `{type: 'osprey-artifact-command', command}`. The
 * page is sandboxed on an opaque origin, so the gallery cannot reach into
 * it: this module is the page's half. The snippet the server injects into
 * every plot_html page (app.py `_RESPONSIVE_PLOTLY`) imports it, inlined as
 * a `data:` module, so it imports nothing itself.
 *
 * Each command runs on every plot of the page, through Plotly's public
 * `relayout` and `downloadImage`:
 *   - `box-zoom`, `pan`: the drag mode;
 *   - `zoom-in`, `zoom-out`: halve or double each cartesian axis's span about
 *     its centre, as Plotly's modebar does (an axis with `fixedrange` stays);
 *   - `autoscale`: every cartesian axis fits its data;
 *   - `reset-axes`: back to the view the plot opened with, which the snippet
 *     records once the page has loaded ({@link rememberViews});
 *   - `export-png`: download the plot as it shows now, as a PNG named after
 *     the message's `filename` when it has one.
 * Only the frame's parent, the gallery that embeds it, is listened to.
 *
 * Framed, the page also marks its root element ({@link FRAMED_ATTR}), and
 * the snippet's CSS hides Plotly's modebar under that mark: the strip
 * replaces it. A plot page opened on its own keeps Plotly's modebar. (The
 * pointer reports that show the strip over the frame are frame-pointer.js.)
 *
 * @module plot-commands
 */

/** The message type the gallery's commands arrive in. */
export const PLOT_COMMAND_MESSAGE = 'osprey-artifact-command';

/** The attribute a framed plot page carries on its root element. */
export const FRAMED_ATTR = 'data-osprey-framed';

/**
 * @typedef {'box-zoom' | 'pan' | 'zoom-in' | 'zoom-out' | 'autoscale' | 'reset-axes'
 *   | 'export-png'} PlotCommand
 */

/** @type {ReadonlyArray<string>} */
export const PLOT_COMMANDS = [
  'box-zoom',
  'pan',
  'zoom-in',
  'zoom-out',
  'autoscale',
  'reset-axes',
  'export-png',
];

/** A cartesian axis's key in a Plotly layout: `xaxis`, `yaxis2`, ... */
const AXIS_KEY = /^[xy]axis\d*$/;

/** The longest file name an export takes from a message. */
const MAX_FILENAME = 120;

/** @type {WeakMap<object, Record<string, unknown>>} the view each plot opened with */
const openedViews = new WeakMap();

/**
 * The cartesian axes of a plot, from the layout Plotly computed.
 * @param {any} gd
 * @returns {[string, any][]}
 */
function axesOf(gd) {
  const full = gd?._fullLayout;
  if (!full) return [];
  return Object.keys(full)
    .filter((key) => AXIS_KEY.test(key) && full[key])
    .map((key) => [key, full[key]]);
}

/**
 * Record the view each plot opened with, for `reset-axes`: an autoranged
 * axis goes back to its autorange, a fixed one to its range. A plot keeps
 * its first record.
 * @param {Iterable<any>} gds
 */
export function rememberViews(gds) {
  for (const gd of gds) {
    if (openedViews.has(gd)) continue;
    /** @type {Record<string, unknown>} */
    const view = {};
    for (const [key, ax] of axesOf(gd)) {
      if (ax.autorange) view[`${key}.autorange`] = ax.autorange;
      else if (Array.isArray(ax.range)) view[`${key}.range`] = [...ax.range];
    }
    openedViews.set(gd, view);
  }
}

/**
 * Each free cartesian axis's span scaled by `factor` about its centre. An
 * axis converts its range to linear values and back (dates, log axes).
 * @param {any} gd
 * @param {number} factor 0.5 halves the span, 2 doubles it
 * @returns {Record<string, unknown>}
 */
function scaledRanges(gd, factor) {
  const near = (1 + factor) / 2;
  const far = (1 - factor) / 2;
  /** @type {Record<string, unknown>} */
  const update = {};
  for (const [key, ax] of axesOf(gd)) {
    if (ax.fixedrange || !Array.isArray(ax.range)) continue;
    /** @type {(value: unknown) => number} */
    const toLinear = typeof ax.r2l === 'function' ? (value) => ax.r2l(value) : Number;
    /** @type {(value: number) => unknown} */
    const fromLinear = typeof ax.l2r === 'function' ? (value) => ax.l2r(value) : (value) => value;
    const start = toLinear(ax.range[0]);
    const end = toLinear(ax.range[1]);
    if (!Number.isFinite(start) || !Number.isFinite(end)) continue;
    update[`${key}.range[0]`] = fromLinear(near * start + far * end);
    update[`${key}.range[1]`] = fromLinear(near * end + far * start);
  }
  return update;
}

/**
 * Every cartesian axis set to fit its data.
 * @param {any} gd
 * @returns {Record<string, unknown>}
 */
function autoranged(gd) {
  /** @type {Record<string, unknown>} */
  const update = {};
  for (const [key] of axesOf(gd)) update[`${key}.autorange`] = true;
  return update;
}

/**
 * Run one command on one plot.
 * @param {any} plotly the page's Plotly
 * @param {any} gd the plot's graph div
 * @param {string} command one of {@link PLOT_COMMANDS}
 * @param {{filename?: string}} [options] the export's file name, without extension
 * @returns {unknown} what Plotly answers (a promise), or undefined for an unknown command
 */
export function runPlotCommand(plotly, gd, command, { filename = 'plot' } = {}) {
  switch (command) {
    case 'box-zoom':
      return plotly.relayout(gd, { dragmode: 'zoom' });
    case 'pan':
      return plotly.relayout(gd, { dragmode: 'pan' });
    case 'zoom-in':
      return plotly.relayout(gd, scaledRanges(gd, 0.5));
    case 'zoom-out':
      return plotly.relayout(gd, scaledRanges(gd, 2));
    case 'autoscale':
      return plotly.relayout(gd, autoranged(gd));
    case 'reset-axes':
      return plotly.relayout(gd, openedViews.get(gd) ?? autoranged(gd));
    case 'export-png':
      return plotly.downloadImage(gd, { format: 'png', filename });
    default:
      return undefined;
  }
}

/**
 * Take the gallery's commands. Framed, mark the page (its CSS then hides
 * Plotly's modebar) and run each command the parent posts on every plot; on
 * its own, the page is left as it is.
 * @param {Window} win the plot page's window
 * @param {() => any[]} plots the page's graph divs, read per command
 */
export function followPlotCommands(win, plots) {
  if (win.parent === win) return;
  win.document.documentElement.setAttribute(FRAMED_ATTR, '');
  win.addEventListener('message', (event) => {
    const data = event.data;
    if (event.source !== win.parent || !data || data.type !== PLOT_COMMAND_MESSAGE) return;
    if (!PLOT_COMMANDS.includes(data.command)) return;
    const plotly = /** @type {any} */ (win).Plotly;
    if (!plotly) return;
    const named = typeof data.filename === 'string' && data.filename.trim();
    const filename = named ? data.filename.trim().slice(0, MAX_FILENAME) : win.document.title;
    for (const gd of plots()) {
      try {
        Promise.resolve(
          runPlotCommand(plotly, gd, data.command, { filename: filename || 'plot' })
        ).catch(() => {});
      } catch {
        // A plot that refuses the command keeps its view.
      }
    }
  });
}
