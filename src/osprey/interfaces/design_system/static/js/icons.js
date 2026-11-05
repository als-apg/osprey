// @ts-check
/* OSPREY Design System — chrome icons.
 *
 * The stroke icons of the sidebar shell: sidebar rows, the pane header, the
 * popovers. Built as SVG nodes with createElementNS (no icon package, no
 * innerHTML), on a 24-unit grid. Size, stroke and colour come from CSS
 * (`.osprey-ic` in primitives.css: 16 px, stroke width 1.75, `currentColor`),
 * so one icon serves every surface.
 *
 * Every icon except `agent` and `bash` is Lucide (lucide-static, ISC licence),
 * as frozen in the layout F mockup's <symbol> set; the licence text and
 * copyright notice ship in web_terminal/static/js/LUCIDE-LICENSE.txt. `agent`
 * and `bash` are OSPREY's own, the same paths as web_terminal's chat-icons.js.
 *
 * @module icons
 */

const SVG_NS = 'http://www.w3.org/2000/svg';

/**
 * One child of an icon: a string is a <path> `d`; a tuple is any other
 * element with its attributes.
 * @typedef {string | readonly [string, Readonly<Record<string, string>>]} Shape
 */

/**
 * Children per icon name, on a 24×24 grid.
 * @type {Readonly<Record<string, readonly Shape[]>>}
 */
const SHAPES = Object.freeze({
  'panel-left': [
    ['rect', { width: '18', height: '18', x: '3', y: '3', rx: '2' }],
    'M9 3v18',
    'm16 15-3-3 3-3',
  ],
  'chevron-down': ['m6 9 6 6 6-6'],
  'chevron-right': ['m9 18 6-6-6-6'],
  'chevron-left': ['m15 18-6-6 6-6'],
  'chevrons-ud': ['m7 15 5 5 5-5', 'm7 9 5-5 5 5'],
  search: ['M11 3a8 8 0 1 0 0 16a8 8 0 1 0 0-16z', 'm21 21-4.35-4.35'],
  workspace: [
    ['rect', { width: '7', height: '9', x: '3', y: '3', rx: '1' }],
    ['rect', { width: '7', height: '5', x: '14', y: '3', rx: '1' }],
    ['rect', { width: '7', height: '9', x: '14', y: '12', rx: '1' }],
    ['rect', { width: '7', height: '5', x: '3', y: '16', rx: '1' }],
  ],
  activity: [
    'M22 12h-2.48a2 2 0 0 0-1.93 1.46l-2.35 8.36a.25.25 0 0 1-.48 0L9.24 2.18a.25.25 0 0 0-.48 0l-2.35 8.36A2 2 0 0 1 4.49 12H2',
  ],
  terminal: ['M12 19h8', 'm4 17 6-6-6-6'],
  notebook: [
    'M2 6h4',
    'M2 10h4',
    'M2 14h4',
    'M2 18h4',
    ['rect', { width: '16', height: '20', x: '4', y: '2', rx: '2' }],
    'M16 2v20',
  ],
  'book-open': [
    'M12 7v14',
    'M3 18a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h5a4 4 0 0 1 4 4 4 4 0 0 1 4-4h5a1 1 0 0 1 1 1v13a1 1 0 0 1-1 1h-6a3 3 0 0 0-3 3 3 3 0 0 0-3-3z',
  ],
  hexagon: [
    'M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z',
  ],
  library: ['m16 6 4 14', 'M12 6v14', 'M8 8v12', 'M4 4v16'],
  radio: [
    'M4.9 19.1C1 15.2 1 8.8 4.9 4.9',
    'M7.8 16.2c-2.3-2.3-2.3-6.1 0-8.5',
    ['circle', { cx: '12', cy: '12', r: '2' }],
    'M16.2 7.8c2.3 2.3 2.3 6.1 0 8.5',
    'M19.1 4.9C23 8.8 23 15.1 19.1 19',
  ],
  stethoscope: [
    'M11 2v2',
    'M5 2v2',
    'M5 3H4a2 2 0 0 0-2 2v4a6 6 0 0 0 12 0V5a2 2 0 0 0-2-2h-1',
    'M8 15a6 6 0 0 0 12 0v-3',
    ['circle', { cx: '20', cy: '10', r: '2' }],
  ],
  // The dots carry their own fill: `.osprey-ic` sets fill:none on the svg.
  'chart-scatter': [
    ['circle', { cx: '7.5', cy: '7.5', r: '.5', fill: 'currentColor' }],
    ['circle', { cx: '18.5', cy: '5.5', r: '.5', fill: 'currentColor' }],
    ['circle', { cx: '11.5', cy: '11.5', r: '.5', fill: 'currentColor' }],
    ['circle', { cx: '7.5', cy: '16.5', r: '.5', fill: 'currentColor' }],
    ['circle', { cx: '17.5', cy: '14.5', r: '.5', fill: 'currentColor' }],
    'M3 3v16a2 2 0 0 0 2 2h16',
  ],
  plus: ['M5 12h14', 'M12 5v14'],
  help: [
    ['circle', { cx: '12', cy: '12', r: '10' }],
    'M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3',
    'M12 17h.01',
  ],
  message: ['M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z'],
  more: [
    ['circle', { cx: '12', cy: '12', r: '1' }],
    ['circle', { cx: '19', cy: '12', r: '1' }],
    ['circle', { cx: '5', cy: '12', r: '1' }],
  ],
  settings: [
    'M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z',
    ['circle', { cx: '12', cy: '12', r: '3' }],
  ],
  sliders: [
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
  'file-text': [
    'M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z',
    'M14 2v4a2 2 0 0 0 2 2h4',
    'M10 9H8',
    'M16 13H8',
    'M16 17H8',
  ],
  'chart-line': ['M3 3v16a2 2 0 0 0 2 2h16', 'm19 9-5 5-4-4-3 3'],
  bash: ['M3 5h18v14H3z', 'm7 10 3 2.5L7 15', 'M13 15h4'],
  maximize: ['M15 3h6v6', 'm21 3-7 7', 'm3 21 7-7', 'M9 21H3v-6'],
  popout: [
    'M21 9V6a2 2 0 0 0-2-2H4a2 2 0 0 0-2 2v10c0 1.1.9 2 2 2h4',
    ['rect', { width: '10', height: '7', x: '12', y: '13', rx: '2' }],
  ],
  save: [
    'M15.2 3a2 2 0 0 1 1.4.6l3.8 3.8a2 2 0 0 1 .6 1.4V19a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2z',
    'M17 21v-7a1 1 0 0 0-1-1H8a1 1 0 0 0-1 1v7',
    'M7 3v4a1 1 0 0 0 1 1h7',
  ],
  x: ['M18 6 6 18', 'm6 6 12 12'],
  pen: [
    'M12 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7',
    'M18.375 2.625a1 1 0 0 1 3 3l-9.013 9.014a2 2 0 0 1-.853.505l-2.873.84a.5.5 0 0 1-.62-.62l.84-2.873a2 2 0 0 1 .506-.852z',
  ],
  bell: [
    'M10.268 21a2 2 0 0 0 3.464 0',
    'M3.262 15.326A1 1 0 0 0 4 17h16a1 1 0 0 0 .74-1.673C19.41 13.956 18 12.499 18 8A6 6 0 0 0 6 8c0 4.499-1.411 5.956-2.738 7.326',
  ],
  share: ['M12 2v13', 'm16 6-4-4-4 4', 'M4 12v8a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8'],
  bulb: [
    'M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5',
    'M9 18h6',
    'M10 22h4',
  ],
  clip: [
    'm16 6-8.414 8.586a2 2 0 0 0 2.829 2.829l8.414-8.586a4 4 0 1 0-5.657-5.657l-8.379 8.551a6 6 0 1 0 8.485 8.485l8.379-8.551',
  ],
  up: ['M12 19V5', 'M5 12l7-7 7 7'],
  agent: [
    'M6 8h12a3 3 0 0 1 3 3v6a3 3 0 0 1-3 3H6a3 3 0 0 1-3-3v-6a3 3 0 0 1 3-3z',
    'M12 8V4',
    'M9 14h.01',
    'M15 14h.01',
  ],
  crosshair: [
    ['circle', { cx: '12', cy: '12', r: '10' }],
    'M22 12h-4',
    'M6 12H2',
    'M12 6V2',
    'M12 22v-4',
  ],
  menu: ['M4 12h16', 'M4 18h16', 'M4 6h16'],
  keyboard: [
    'M10 8h.01',
    'M12 12h.01',
    'M14 8h.01',
    'M16 12h.01',
    'M18 8h.01',
    'M6 8h.01',
    'M7 16h10',
    'M8 12h.01',
    ['rect', { width: '20', height: '16', x: '2', y: '4', rx: '2' }],
  ],
  compass: [
    'm16.24 7.76-1.804 5.411a2 2 0 0 1-1.265 1.265L7.76 16.24l1.804-5.411a2 2 0 0 1 1.265-1.265z',
    ['circle', { cx: '12', cy: '12', r: '10' }],
  ],
  info: [['circle', { cx: '12', cy: '12', r: '10' }], 'M12 16v-4', 'M12 8h.01'],
  logout: ['m16 17 5-5-5-5', 'M21 12H9', 'M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4'],
  'sun-moon': [
    'M12 2v2',
    'M13 8.129A4 4 0 0 1 15.873 11',
    'm19 5-1.256 1.256',
    'M20 12h2',
    'M9 8a5 5 0 1 0 7 7 7 7 0 1 1-7-7',
  ],
  check: ['M20 6 9 17l-5-5'],
  types: ['M21 12h-8', 'M21 6H8', 'M21 18h-8', 'M3 6v4c0 1.1.9 2 2 2h3', 'M3 10v6c0 1.1.9 2 2 2h3'],
  history: ['M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8', 'M3 3v5h5', 'M12 7v5l4 2'],
  table: [
    'M12 3v18',
    ['rect', { width: '18', height: '18', x: '3', y: '3', rx: '2' }],
    'M3 9h18',
    'M3 15h18',
  ],
  database: [
    ['ellipse', { cx: '12', cy: '5', rx: '9', ry: '3' }],
    'M3 5V19A9 3 0 0 0 21 19V5',
    'M3 12A9 3 0 0 0 21 12',
  ],
  refresh: [
    'M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8',
    'M21 3v5h-5',
    'M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16',
    'M8 16H3v5',
  ],
  layers: [
    'M12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83z',
    'M2 12a1 1 0 0 0 .58.91l8.6 3.91a2 2 0 0 0 1.65 0l8.58-3.9A1 1 0 0 0 22 12',
    'M2 17a1 1 0 0 0 .58.91l8.6 3.91a2 2 0 0 0 1.65 0l8.58-3.9A1 1 0 0 0 22 17',
  ],
  // ---- Panel chrome (2026-11) ----
  // Lucide, like the rest. `box-zoom` is Lucide's square-dashed, `autoscale`
  // its expand, `reset-axes` its house (the plot convention for "home") and
  // `trash` its trash-2. `list-tree` draws the same glyph as `types`.
  download: ['M12 15V3', 'M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4', 'm7 10 5 5 5-5'],
  copy: [
    ['rect', { width: '14', height: '14', x: '8', y: '8', rx: '2' }],
    'M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2',
  ],
  trash: [
    'M3 6h18',
    'M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6',
    'M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2',
    'M10 11v6',
    'M14 11v6',
  ],
  filter: [
    'M10 20a1 1 0 0 0 .553.895l2 1A1 1 0 0 0 14 21v-7a2 2 0 0 1 .517-1.341L21.74 4.67A1 1 0 0 0 21 3H3a1 1 0 0 0-.742 1.67l7.225 7.989A2 2 0 0 1 10 14z',
  ],
  'zoom-in': [['circle', { cx: '11', cy: '11', r: '8' }], 'm21 21-4.35-4.35', 'M11 8v6', 'M8 11h6'],
  'zoom-out': [['circle', { cx: '11', cy: '11', r: '8' }], 'm21 21-4.35-4.35', 'M8 11h6'],
  move: ['M12 2v20', 'm15 19-3 3-3-3', 'm19 9 3 3-3 3', 'M2 12h20', 'm5 9-3 3 3 3', 'm9 5 3-3 3 3'],
  'box-zoom': [
    'M5 3a2 2 0 0 0-2 2',
    'M19 3a2 2 0 0 1 2 2',
    'M21 19a2 2 0 0 1-2 2',
    'M5 21a2 2 0 0 1-2-2',
    'M9 3h1',
    'M9 21h1',
    'M14 3h1',
    'M14 21h1',
    'M3 9v1',
    'M21 9v1',
    'M3 14v1',
    'M21 14v1',
  ],
  autoscale: [
    'm15 15 6 6',
    'm15 9 6-6',
    'M21 16v5h-5',
    'M21 8V3h-5',
    'M3 16v5h5',
    'm3 21 6-6',
    'M3 8V3h5',
    'M9 9 3 3',
  ],
  'reset-axes': [
    'M15 21v-8a1 1 0 0 0-1-1h-4a1 1 0 0 0-1 1v8',
    'M3 10a2 2 0 0 1 .709-1.528l7-5.999a2 2 0 0 1 2.582 0l7 5.999A2 2 0 0 1 21 10v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z',
  ],
  'image-down': [
    'M10.3 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2v10l-3.1-3.1a2 2 0 0 0-2.814.014L6 21',
    'm14 19 3 3v-5.5',
    'm17 22 3-3',
    ['circle', { cx: '9', cy: '9', r: '2' }],
  ],
  'list-tree': [
    'M21 12h-8',
    'M21 6H8',
    'M21 18h-8',
    'M3 6v4c0 1.1.9 2 2 2h3',
    'M3 10v6c0 1.1.9 2 2 2h3',
  ],
  'text-search': [
    'M21 6H3',
    'M10 12H3',
    'M10 18H3',
    ['circle', { cx: '17', cy: '15', r: '3' }],
    'm21 19-1.9-1.9',
  ],
  clock: [['circle', { cx: '12', cy: '12', r: '10' }], 'M12 6v6l4 2'],
  'triangle-alert': [
    'M21.73 18 13.73 4a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3',
    'M12 9v4',
    'M12 17h.01',
  ],
});

/** Icons drawn heavier than the 1.75 default, set on each child. */
const STROKE_WIDTH = Object.freeze(/** @type {Record<string, string>} */ ({ up: '2' }));

/** Every icon name `icon()` accepts. */
export const ICON_NAMES = Object.freeze(Object.keys(SHAPES));

/**
 * Build one icon.
 *
 * @param {string} name one of {@link ICON_NAMES}
 * @param {{ size?: number }} [options] `size` in px; 16 (the default) leaves
 *   the size to `.osprey-ic`
 * @returns {SVGSVGElement} an `svg.osprey-ic`, hidden from assistive tech: the
 *   control that holds it carries the accessible name
 * @throws {Error} on an unknown name, so a typo fails at the call site
 */
export function icon(name, { size = 16 } = {}) {
  const shapes = Object.hasOwn(SHAPES, name) ? SHAPES[name] : null;
  if (!shapes) throw new Error(`icon: unknown icon name "${name}"`);
  const svg = /** @type {SVGSVGElement} */ (document.createElementNS(SVG_NS, 'svg'));
  svg.setAttribute('class', 'osprey-ic');
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.setAttribute('aria-hidden', 'true');
  svg.setAttribute('focusable', 'false');
  if (size !== 16) {
    svg.style.width = `${size}px`;
    svg.style.height = `${size}px`;
  }
  const strokeWidth = Object.hasOwn(STROKE_WIDTH, name) ? STROKE_WIDTH[name] : null;
  for (const shape of shapes) {
    const [tag, attrs] = typeof shape === 'string' ? ['path', { d: shape }] : shape;
    const child = document.createElementNS(SVG_NS, tag);
    for (const [key, value] of Object.entries(attrs)) child.setAttribute(key, value);
    if (strokeWidth) child.setAttribute('stroke-width', strokeWidth);
    svg.append(child);
  }
  return svg;
}
