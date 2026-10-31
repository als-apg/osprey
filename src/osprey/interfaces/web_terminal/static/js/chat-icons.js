// @ts-check
/* OSPREY Web Terminal — inline icons for the Simple-view chat.
 *
 * Stroke icons on a 24-unit grid, built as SVG nodes (no icon package, no
 * innerHTML). Size and colour come from CSS (`.op-icon`, `currentColor`), so one
 * icon serves the composer, the action rows, the cards and the turn footer.
 *
 * `brain`, `wrench`, `tasks` (Lucide's `list-checks`) and `skill` (Lucide's `book-open`) are Lucide icons
 * (lucide-static 0.545.0, ISC licence); the licence text and copyright notice
 * ship beside this file in LUCIDE-LICENSE.txt. The other paths are OSPREY's
 * own, drawn to the same grid.
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
  brain: [
    'M12 18V5',
    'M15 13a4.17 4.17 0 0 1-3-4 4.17 4.17 0 0 1-3 4',
    'M17.598 6.5A3 3 0 1 0 12 5a3 3 0 1 0-5.598 1.5',
    'M17.997 5.125a4 4 0 0 1 2.526 5.77',
    'M18 18a4 4 0 0 0 2-7.464',
    'M19.967 17.483A4 4 0 1 1 12 18a4 4 0 1 1-7.967-.517',
    'M6 18a4 4 0 0 1-2-7.464',
    'M6.003 5.125a4 4 0 0 0-2.526 5.77',
  ],
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
  read: ['M2 5h6a4 4 0 0 1 4 4v11a3 3 0 0 0-3-3H2z', 'M22 5h-6a4 4 0 0 0-4 4v11a3 3 0 0 1 3-3h7z'],
  search: ['M11 3a8 8 0 1 0 0 16a8 8 0 1 0 0-16z', 'm21 21-4.35-4.35'],
  skill: [
    'M12 7v14',
    'M3 18a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h5a4 4 0 0 1 4 4 4 4 0 0 1 4-4h5a1 1 0 0 1 1 1v13a1 1 0 0 1-1 1h-6a3 3 0 0 0-3 3 3 3 0 0 0-3-3z',
  ],
  stop: ['M7 5h10a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2z'],
  tasks: ['M13 5h8', 'M13 12h8', 'M13 19h8', 'm3 17 2 2 4-4', 'm3 7 2 2 4-4'],
  up: ['M12 19V5', 'M5 12l7-7 7 7'],
  web: [
    'M3 12a9 9 0 1 0 18 0a9 9 0 1 0-18 0z',
    'M3 12h18',
    'M12 3a14 14 0 0 1 0 18a14 14 0 0 1 0-18z',
  ],
  wrench: [
    'M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.106-3.105c.32-.322.863-.22.983.218a6 6 0 0 1-8.259 7.057l-7.91 7.91a1 1 0 0 1-2.999-3l7.91-7.91a6 6 0 0 1 7.057-8.259c.438.12.54.662.219.984z',
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

/**
 * The OSPREY mark's flight frames on its 16x8 grid (the mark whose master is
 * design-system/img/osprey-mark.svg). `up` is the mark at rest, wings raised;
 * `mid` holds them level and `down` sweeps them low. One run of filled cells
 * per subpath.
 */
const OSPREY_FRAMES = Object.freeze({
  up: 'M0 0h2v1h-2zM14 0h2v1h-2zM1 1h3v1h-3zM12 1h3v1h-3zM2 2h4v1h-4zM10 2h4v1h-4zM3 3h4v1h-4zM9 3h4v1h-4zM4 4h8v1h-8zM6 5h4v1h-4zM7 6h2v1h-2zM6 7h1v1h-1zM9 7h1v1h-1z',
  mid: 'M1 3h2v1h-2zM13 3h2v1h-2zM2 4h12v1h-12zM6 5h4v1h-4zM7 6h2v1h-2zM6 7h1v1h-1zM9 7h1v1h-1z',
  down: 'M3 4h10v1h-10zM1 5h4v1h-4zM6 5h4v1h-4zM11 5h4v1h-4zM0 6h2v1h-2zM7 6h2v1h-2zM14 6h2v1h-2zM6 7h1v1h-1zM9 7h1v1h-1z',
});

/**
 * The OSPREY mark as a sprite: every flight frame is drawn, and CSS shows one
 * (`.op-osprey-up` at rest). Decorative, like {@link chatIcon}.
 * @returns {SVGSVGElement}
 */
export function ospreySprite() {
  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('viewBox', '0 0 16 8');
  svg.setAttribute('class', 'op-osprey');
  svg.setAttribute('fill', 'currentColor');
  svg.setAttribute('shape-rendering', 'crispEdges');
  svg.setAttribute('aria-hidden', 'true');
  svg.setAttribute('focusable', 'false');
  for (const [frame, d] of Object.entries(OSPREY_FRAMES)) {
    const path = document.createElementNS(SVG_NS, 'path');
    path.setAttribute('class', `op-osprey-${frame}`);
    path.setAttribute('d', d);
    svg.append(path);
  }
  return svg;
}
