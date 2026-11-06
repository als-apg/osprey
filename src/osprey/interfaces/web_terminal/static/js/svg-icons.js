// @ts-check
/* OSPREY Web Terminal — inline SVG icon for the tile header bar's search pill
 *
 * The legacy `search` contribution's magnifier. It was an emoji (🔍), which
 * renders full-color on macOS/Windows — a loud non-chrome mark in an
 * otherwise monochrome bar — and glyph metrics drift across the font stack.
 * A geometric stroke on currentColor instead, so it takes the pill's own
 * text color, including hover and focus states. The bar's other icons are
 * the design system's (`icon()` in /design-system/js/icons.js).
 *
 * Built via createElementNS rather than an innerHTML literal so the module
 * that consumes it (tile-header-items.js) keeps its "nothing here
 * interpolates markup" contract intact.
 */

const NS = 'http://www.w3.org/2000/svg';

/**
 * Build the icon. The element is aria-hidden: the OWNING control carries the
 * accessible name (aria-label/title), never the pictogram.
 * @param {'search'} name
 * @returns {SVGSVGElement}
 */
export function svgIcon(name) {
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('viewBox', '0 0 16 16');
  // Intrinsic size, carried on the element itself. `.tile-tab .bar-icon`
  // supplies the bar's real 16px, and CSS beats presentation attributes — so
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
  const handle = document.createElementNS(NS, 'path');
  handle.setAttribute('d', 'M10.1 10.1 L13.5 13.5');
  const lens = document.createElementNS(NS, 'circle');
  lens.setAttribute('cx', '7');
  lens.setAttribute('cy', '7');
  lens.setAttribute('r', '4.25');
  svg.append(handle, lens);
  return svg;
}
