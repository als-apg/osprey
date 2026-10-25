// @ts-check
/* OSPREY Web Terminal — inline icons for the Simple-view chat.
 *
 * Stroke icons on a 24-unit grid, built as SVG nodes (no icon package, no
 * innerHTML). Size and colour come from CSS (`.op-icon`, `currentColor`), so one
 * icon serves the composer, the tool chips, the cards and the turn footer.
 */

const SVG_NS = 'http://www.w3.org/2000/svg';

/**
 * Path data per icon name, on a 24×24 grid.
 * @type {Readonly<Record<string, readonly string[]>>}
 */
const PATHS = Object.freeze({
  agent: [
    'M6 8h12a3 3 0 0 1 3 3v6a3 3 0 0 1-3 3H6a3 3 0 0 1-3-3v-6a3 3 0 0 1 3-3z',
    'M12 8V4',
    'M9 14h.01',
    'M15 14h.01',
  ],
  alert: ['M3 12a9 9 0 1 0 18 0a9 9 0 1 0-18 0z', 'M12 8v5', 'M12 16h.01'],
  bash: ['M3 5h18v14H3z', 'm7 10 3 2.5L7 15', 'M13 15h4'],
  check: ['M20 6 9 17l-5-5'],
  chevron: ['m9 18 6-6-6-6'],
  circle: ['M3 12a9 9 0 1 0 18 0a9 9 0 1 0-18 0z'],
  control: [
    'M4 21v-7',
    'M4 10V3',
    'M12 21v-9',
    'M12 8V3',
    'M20 21v-5',
    'M20 12V3',
    'M1 14h6',
    'M9 8h6',
    'M17 16h6',
  ],
  copy: [
    'M9 9h11a2 2 0 0 1 2 2v11a2 2 0 0 1-2 2H11a2 2 0 0 1-2-2z',
    'M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1',
  ],
  edit: ['M12 20h9', 'M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z'],
  question: ['M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z'],
  read: [
    'M2 5h6a4 4 0 0 1 4 4v11a3 3 0 0 0-3-3H2z',
    'M22 5h-6a4 4 0 0 0-4 4v11a3 3 0 0 1 3-3h7z',
  ],
  search: ['M11 3a8 8 0 1 0 0 16a8 8 0 1 0 0-16z', 'm21 21-4.35-4.35'],
  spark: ['M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9z'],
  stop: ['M7 5h10a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2z'],
  tool: [
    'M14.7 6.3a4 4 0 0 0-5.4 5.4L3 18l3 3 6.3-6.3a4 4 0 0 0 5.4-5.4l-2.5 2.5-2.5-.5-.5-2.5z',
  ],
  up: ['M12 19V5', 'M5 12l7-7 7 7'],
  web: [
    'M3 12a9 9 0 1 0 18 0a9 9 0 1 0-18 0z',
    'M3 12h18',
    'M12 3a14 14 0 0 1 0 18a14 14 0 0 1 0-18z',
  ],
  x: ['M18 6 6 18', 'M6 6l12 12'],
});

/** Icons drawn solid rather than as outlines. */
const FILLED = new Set(['stop']);

/** @typedef {keyof typeof PATHS} ChatIconName */

/**
 * An inline SVG icon. Decorative: hidden from assistive technology, because the
 * control or label it sits in carries the accessible name.
 * @param {ChatIconName} name
 * @returns {SVGSVGElement}
 */
export function chatIcon(name) {
  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.setAttribute('class', 'op-icon');
  svg.setAttribute('fill', FILLED.has(name) ? 'currentColor' : 'none');
  svg.setAttribute('stroke', 'currentColor');
  svg.setAttribute('stroke-width', '2');
  svg.setAttribute('stroke-linecap', 'round');
  svg.setAttribute('stroke-linejoin', 'round');
  svg.setAttribute('aria-hidden', 'true');
  svg.setAttribute('focusable', 'false');
  for (const d of PATHS[name]) {
    const path = document.createElementNS(SVG_NS, 'path');
    path.setAttribute('d', d);
    svg.append(path);
  }
  return svg;
}
