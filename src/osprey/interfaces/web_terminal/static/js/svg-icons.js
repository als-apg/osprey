// @ts-check
/* OSPREY Web Terminal — inline SVG icons for the tile header bar
 *
 * The bar's icons were typographic glyphs (×, ⋯) and one emoji (🔍). The
 * emoji renders full-color on macOS/Windows — a loud non-chrome mark in an
 * otherwise monochrome bar — and glyph metrics drift across the font stack.
 * These are geometric strokes on currentColor instead, so every icon takes
 * the button's own text color, including hover and disabled states.
 *
 * Built via createElementNS rather than an innerHTML literal so the modules
 * that consume them (tile-header-items.js, dock-tab.js) keep their "nothing
 * here interpolates markup" contract intact.
 */

const NS = 'http://www.w3.org/2000/svg';

/** Stroke path data per icon, on a 16×16 viewBox. */
const PATHS = {
  /** × — close. */
  close: ['M4.5 4.5 L11.5 11.5', 'M11.5 4.5 L4.5 11.5'],
  /** 🔍 → magnifier. */
  search: ['M10.1 10.1 L13.5 13.5'],
  /** ⋯ — overflow menu. */
  ellipsis: [],
};

/**
 * Build one icon. The element is aria-hidden: the OWNING control carries the
 * accessible name (aria-label/title), never the pictogram.
 * @param {'close'|'search'|'ellipsis'} name
 * @returns {SVGSVGElement}
 */
export function svgIcon(name) {
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('viewBox', '0 0 16 16');
  // Intrinsic size, carried on the element itself. `.dock-root .bar-icon`
  // supplies the bar's real 14px, and CSS beats presentation attributes — so
  // these change nothing in the bar. They matter where that rule does NOT
  // reach: dockview builds a tile's drag ghost by cloning the header bar into
  // document.body (outside #dock-root), and a viewBox alone gives an SVG an
  // intrinsic RATIO but no intrinsic SIZE, so `width: auto` resolves to the
  // containing block's width and the ratio squares it — the search icon
  // painted as a 402px magnifier riding the cursor. 16px is the viewBox's own
  // scale, so the ghost degrades to a bar-sized icon instead.
  svg.setAttribute('width', '16');
  svg.setAttribute('height', '16');
  svg.setAttribute('fill', 'none');
  svg.setAttribute('stroke', 'currentColor');
  svg.setAttribute('stroke-width', '1.5');
  svg.setAttribute('stroke-linecap', 'round');
  svg.setAttribute('aria-hidden', 'true');
  svg.classList.add('bar-icon', `bar-icon-${name}`);
  for (const d of PATHS[name]) {
    const path = document.createElementNS(NS, 'path');
    path.setAttribute('d', d);
    svg.appendChild(path);
  }
  if (name === 'search') {
    const lens = document.createElementNS(NS, 'circle');
    lens.setAttribute('cx', '7');
    lens.setAttribute('cy', '7');
    lens.setAttribute('r', '4.25');
    svg.appendChild(lens);
  } else if (name === 'ellipsis') {
    for (const cx of ['3.5', '8', '12.5']) {
      const dot = document.createElementNS(NS, 'circle');
      dot.setAttribute('cx', cx);
      dot.setAttribute('cy', '8');
      dot.setAttribute('r', '1.4');
      dot.setAttribute('fill', 'currentColor');
      dot.setAttribute('stroke', 'none');
      svg.appendChild(dot);
    }
  }
  return svg;
}
