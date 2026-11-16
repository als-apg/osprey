// @ts-check
/**
 * The knob bar: one drawing of a knob's range, the same in every view.
 *
 * The track is a range (the safe range of a knob in set-up and review, the
 * search window on the run page). On it, each optional: the search window (a
 * band), the value before the run (a line), and the marks of a run: where it
 * started (a tick), the best point so far (a ring), the last point it wrote
 * (a dot). A value the caller does not have is not drawn: the bar never
 * places a mark it was not given.
 *
 * A knob is named by its `VariableMetadata.description`, with the channel
 * identifier on a second line; a knob with no description is named by its
 * identifier alone ({@link knobLabel}).
 *
 * @module knob-bar
 */

import { fmt, h } from './hyperscript.js';

/** The key to the bar's drawings, in the operator's words. */
export const KNOB_KEYS = Object.freeze({
  track: 'safe range',
  window: 'search window',
  now: 'now',
  start: 'start',
  best: 'best so far',
  current: 'on the machine',
  recorded: 'last evaluation recorded',
});

/**
 * @typedef {keyof typeof KNOB_KEYS} KnobKey
 * @typedef {{ start?: number|null, best?: number|null, current?: number|null }} KnobMarks
 * @typedef {{
 *   lo: number,
 *   hi: number,
 *   window?: { min: number|null|undefined, max: number|null|undefined }|null,
 *   now?: number|null,
 *   marks?: KnobMarks,
 *   tone?: 'problem'|'clamped'|null,
 *   units?: string,
 *   label?: string,
 *   ends?: boolean,
 * }} KnobBarSpec
 */

/**
 * Where `value` sits on the track `[lo, hi]`, in percent of its width,
 * clamped to the track; the middle for an empty track; null for no value.
 *
 * @param {number|null|undefined} value
 * @param {number} lo
 * @param {number} hi
 * @returns {number|null}
 */
export function position(value, lo, hi) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return null;
  if (!(hi > lo)) return 50;
  return Math.max(0, Math.min(100, ((value - lo) / (hi - lo)) * 100));
}

/**
 * The two lines that name a knob: its description, then its identifier; the
 * identifier alone when it has no description.
 *
 * @param {string} name  the variable's identifier (the channel)
 * @param {{ description?: string|null }|null|undefined} meta  its `VariableMetadata`
 * @returns {{ title: string, channel: string|null }}
 */
export function knobLabel(name, meta) {
  const description = (meta?.description || '').trim();
  return description ? { title: description, channel: name } : { title: name, channel: null };
}

/**
 * The knob's name as two lines (see {@link knobLabel}).
 *
 * @param {string} name
 * @param {{ description?: string|null }|null|undefined} meta
 * @returns {HTMLElement}
 */
export function knobName(name, meta) {
  const { title, channel } = knobLabel(name, meta);
  return h(
    'span',
    { class: 'opt-knob-name' },
    h('span', { class: 'opt-knob-title', title, text: title }),
    channel ? h('span', { class: 'opt-knob-channel', title: channel, text: channel }) : null
  );
}

/**
 * @param {string} cls
 * @param {number|null} at
 * @param {string} kind
 */
function mark(cls, at, kind) {
  return at === null
    ? null
    : h('div', { class: cls, 'data-mark': kind, style: `left: ${at.toFixed(2)}%` });
}

/**
 * A knob bar.
 *
 * @param {KnobBarSpec} spec
 * @returns {HTMLElement}
 */
export function knobBar({
  lo,
  hi,
  window = null,
  now = null,
  marks = {},
  tone = null,
  units = '',
  label = '',
  ends = true,
}) {
  const at = (/** @type {number|null|undefined} */ v) => position(v, lo, hi);
  const track = h('div', { class: 'opt-kbar-track' });
  const from = at(window?.min);
  const to = at(window?.max);
  if (from !== null && to !== null && to >= from) {
    track.append(
      h('div', {
        class: `opt-kbar-window${tone ? ` is-${tone}` : ''}`,
        'data-mark': 'window',
        style: `left: ${from.toFixed(2)}%; width: ${Math.max(0.8, to - from).toFixed(2)}%`,
      })
    );
  }
  for (const node of [
    mark('opt-kbar-now', at(now), 'now'),
    mark('opt-kbar-mark opt-kbar-start', at(marks.start), 'start'),
    mark('opt-kbar-mark opt-kbar-best', at(marks.best), 'best'),
    mark('opt-kbar-mark opt-kbar-current', at(marks.current), 'current'),
  ]) {
    if (node) track.append(node);
  }
  const unit = units ? `\u00a0${units}` : '';
  return h(
    'div',
    { class: 'opt-kbar', role: 'img', 'aria-label': label || null },
    track,
    ends
      ? h(
          'div',
          { class: 'opt-kbar-ends' },
          h('span', { text: fmt(lo) }),
          h('span', { text: `${fmt(hi)}${unit}` })
        )
      : null
  );
}

/**
 * Move the marks of a drawn bar in place (a run's marks move every
 * evaluation; the bar is not rebuilt).
 *
 * @param {HTMLElement} bar  a {@link knobBar}
 * @param {KnobBarSpec} spec  the same track as the bar was drawn with
 */
export function updateKnobBar(bar, spec) {
  const fresh = knobBar(spec);
  const track = bar.querySelector('.opt-kbar-track');
  const next = fresh.querySelector('.opt-kbar-track');
  if (track && next) track.replaceChildren(...next.childNodes);
  bar.setAttribute('aria-label', spec.label || '');
}

/**
 * The key to the drawings a view uses.
 *
 * @param {KnobKey[]} keys
 * @returns {HTMLElement}
 */
export function knobLegend(keys) {
  return h(
    'span',
    { class: 'opt-legend' },
    ...keys.map((key) =>
      h(
        'span',
        { class: 'opt-legend-item', 'data-key': key },
        h('i', { class: `opt-swatch opt-swatch-${key}` }),
        KNOB_KEYS[key]
      )
    )
  );
}
