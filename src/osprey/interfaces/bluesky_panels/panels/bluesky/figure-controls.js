/**
 * Reader-owned controls for a figure: what it groups, what it shows, and the
 * chips that change either.
 *
 * Split out of `figure-renderer.js`, which draws a figure the bridge sent. This
 * module owns the other half of the screen — the reader's own choices about
 * that figure, which the bridge knows nothing about and which must survive the
 * renderer replacing its whole subtree on every poll tick.
 *
 * Everything here is pure or builds one detached element. No fetching, no
 * drawing, no theme.
 */

/** @typedef {import('./figure-renderer.js').FigurePanel} FigurePanel */

/**
 * Reader-owned view state, held by the CALLER across renders.
 *
 * `renderFigure` empties and refills its container on every call, so any
 * selection parked in that DOM would reset itself roughly once a second on a
 * live run. Keying by panel title and section name — both stable tick to tick —
 * is what lets a choice survive a redraw, a theme flip, and a figure that grew
 * new panels since the choice was made.
 *
 * @typedef {{
 *   series?: Record<string, string[]>,
 *   section?: Record<string, string>,
 *   open?: Record<string, boolean>,
 * }} FigureSelection
 */

/** How many series a picker-bearing panel draws before the reader chooses. */
export const DEFAULT_PICKED_SERIES = 3;

/**
 * Split panels into the inline ones and the sectioned ones, preserving order.
 *
 * A section is a demotion: its panels render inside one collapsed disclosure
 * below everything inline, and a section holding more than one panel shows one
 * at a time behind a picker. Sections appear in the order their first panel
 * does, so a figure that grows a section keeps the ones it had where they were.
 *
 * @param {FigurePanel[]} panels
 * @returns {{inline: FigurePanel[], sections: Array<{name: string, panels: FigurePanel[]}>}}
 */
export function groupPanels(panels) {
  /** @type {FigurePanel[]} */
  const inline = [];
  /** @type {Map<string, FigurePanel[]>} */
  const sections = new Map();

  for (const panel of panels) {
    const name = panel.section;
    if (!name) {
      inline.push(panel);
      continue;
    }
    const existing = sections.get(name);
    if (existing) existing.push(panel);
    else sections.set(name, [panel]);
  }

  return {
    inline,
    sections: Array.from(sections, ([name, grouped]) => ({ name, panels: grouped })),
  };
}

/**
 * Reconcile a stored multi-selection against the labels actually present.
 *
 * Selections outlive the figure that produced them: a live run gains fitted
 * correctors tick by tick, and a capped one can drop them. Anything no longer
 * present is dropped rather than drawn as an empty line, and a selection left
 * with nothing falls back to the default rather than to a blank panel — an
 * empty plot reads as "no data", which would be a lie about a run that has
 * plenty.
 *
 * @param {string[]|undefined} stored
 * @param {string[]} available
 * @param {number} [fallbackCount]
 * @returns {string[]}
 */
export function resolveSeriesSelection(stored, available, fallbackCount = DEFAULT_PICKED_SERIES) {
  if (available.length === 0) return [];
  if (stored) {
    const kept = stored.filter((label) => available.includes(label));
    if (kept.length > 0) return kept;
  }
  return available.slice(0, Math.max(1, fallbackCount));
}

/**
 * Reconcile a stored single-selection against the titles actually present.
 *
 * @param {string|undefined} stored
 * @param {string[]} available
 * @returns {string}
 */
export function resolveSectionSelection(stored, available) {
  if (stored && available.includes(stored)) return stored;
  return available[0] ?? '';
}

/**
 * One collapsed disclosure holding a section's panels, one at a time.
 *
 * Closed by default and closed on arrival: a section is detail the reader
 * consults, not content they scroll past. The summary carries the count so the
 * section stays legible without opening it — a disclosure that hides *state* is
 * a bad one; hiding *detail* while summarising is the point.
 *
 * `drawPanel` is injected rather than imported so this module never depends on
 * the renderer that depends on it.
 *
 * @param {{name: string, panels: FigurePanel[]}} group
 * @param {{selection: Required<FigureSelection>, redraw: () => void}} view
 * @param {(panel: FigurePanel) => HTMLElement} drawPanel
 * @returns {HTMLElement}
 */
export function sectionElement(group, view, drawPanel) {
  const details = document.createElement('details');
  details.className = 'figure-section';
  details.dataset.section = group.name;
  // `<details open>` is DOM state, and this subtree is replaced on every poll
  // tick — without carrying it in the selection, a section would slam shut
  // under the reader roughly once a second on a live run.
  details.open = view.selection.open[group.name] === true;
  details.addEventListener('toggle', () => {
    view.selection.open[group.name] = details.open;
  });

  const summary = document.createElement('summary');
  summary.className = 'figure-section-summary';
  const name = document.createElement('span');
  name.className = 'figure-section-name';
  name.textContent = group.name;
  const count = document.createElement('span');
  count.className = 'figure-section-count';
  count.textContent = group.panels.length === 1 ? '1 panel' : `${group.panels.length} panels`;
  summary.append(name, count);
  details.appendChild(summary);

  const titles = group.panels.map((panel) => panel.title);
  const picked = resolveSectionSelection(view.selection.section[group.name], titles);

  if (group.panels.length > 1) {
    details.appendChild(
      chipPicker({
        label: 'Show',
        options: titles,
        selected: [picked],
        multi: false,
        onPick: (value) => {
          view.selection.section[group.name] = value;
          view.redraw();
        },
      })
    );
  }

  const shown = group.panels.find((panel) => panel.title === picked) ?? group.panels[0];
  if (shown) details.appendChild(drawPanel(shown));
  return details;
}

/**
 * The heatmap's value scale, as a gradient strip with its numbers.
 *
 * A heatmap without one is unreadable: the reader can see which cells differ,
 * but not by how much, nor which side of zero they fall on. The gradient is
 * built from the two palette strings straight out of CSS — negative hue,
 * transparent at the midpoint, positive hue — which is the same "fade toward
 * the background at zero" the cells are painted with, and needs no colour
 * parsing to stay theme-correct.
 *
 * The endpoint numbers arrive already formatted: the caller owns tick
 * formatting, and taking strings here is what keeps this module free of a
 * circular import back into the renderer.
 *
 * @param {{label: string, low: string, high: string, negative: string, positive: string}} bar
 * @returns {HTMLElement}
 */
export function colorbarElement(bar) {
  const wrap = document.createElement('div');
  wrap.className = 'figure-colorbar';

  if (bar.label) {
    const label = document.createElement('span');
    label.className = 'figure-colorbar-label';
    label.textContent = bar.label;
    wrap.appendChild(label);
  }

  const scale = document.createElement('div');
  scale.className = 'figure-colorbar-scale';

  const low = document.createElement('span');
  low.className = 'figure-colorbar-tick';
  low.textContent = bar.low;

  const strip = document.createElement('div');
  strip.className = 'figure-colorbar-strip';
  strip.style.backgroundImage = `linear-gradient(to right, ${bar.negative} 0%, transparent 50%, ${bar.positive} 100%)`;

  const high = document.createElement('span');
  high.className = 'figure-colorbar-tick';
  high.textContent = bar.high;

  const zero = document.createElement('span');
  zero.className = 'figure-colorbar-tick figure-colorbar-zero';
  zero.textContent = '0';

  scale.append(low, strip, high);
  wrap.append(scale, zero);
  return wrap;
}

/**
 * A row of toggle chips.
 *
 * Chips rather than a `<select>` because the current choice has to be readable
 * at a glance: a closed select hides exactly the thing the reader is comparing.
 * `onPick` receives the chosen value and the caller decides what that means
 * (one OF, or one MORE OF) and re-renders.
 *
 * @param {{label: string, options: string[], selected: string[], multi: boolean, onPick: (value: string) => void, note?: string}} spec
 * @returns {HTMLElement}
 */
export function chipPicker(spec) {
  const wrap = document.createElement('div');
  wrap.className = 'figure-picker';

  const label = document.createElement('span');
  label.className = 'figure-picker-label';
  label.textContent = spec.label;
  wrap.appendChild(label);

  const chips = document.createElement('div');
  chips.className = 'figure-picker-chips';
  chips.setAttribute('role', spec.multi ? 'group' : 'radiogroup');
  chips.setAttribute('aria-label', spec.label);

  for (const option of spec.options) {
    const on = spec.selected.includes(option);
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = `figure-chip${on ? ' selected' : ''}`;
    chip.textContent = option;
    chip.dataset.value = option;
    chip.setAttribute('role', spec.multi ? 'switch' : 'radio');
    chip.setAttribute('aria-checked', on ? 'true' : 'false');
    chip.addEventListener('click', () => spec.onPick(option));
    chips.appendChild(chip);
  }
  wrap.appendChild(chips);

  if (spec.note) {
    const note = document.createElement('span');
    note.className = 'figure-picker-note';
    note.textContent = spec.note;
    wrap.appendChild(note);
  }
  return wrap;
}
