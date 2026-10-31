// @ts-check
/* OSPREY Web Terminal — Onboarding tour spotlight.
 *
 * The spotlight is the UNION of a step's parts (its anchor plus whatever its
 * activation revealed — the control-target chip and its popover), not their
 * bounding box, so rows that merely sit beside the parts stay dimmed. The
 * geometry is pure (unionOutline, outlinePath, lerpBoxes); createSpotlight()
 * is the one DOM piece, the layer tour.js shows while a step is up.
 *
 * The union is rectilinear: the padded rects cut the plane into a grid of
 * cells (one column/row per distinct edge), a cell is in when a rect covers
 * its centre, and the outline is traced along the in/out cell borders. Grids
 * stay tiny (a step has one to three parts), so this costs nothing per frame.
 */

/**
 * A rect in viewport pixels — what `getBoundingClientRect()` returns, reduced
 * to the four edges the geometry reads.
 * @typedef {{left: number, top: number, right: number, bottom: number}} Box
 */

/** @typedef {{x: number, y: number}} Point */

/**
 * Gaps narrower than this many pixels, left between parts AFTER padding, are
 * filled so the outline stays one shape (a popover a few pixels off its
 * trigger, a scrollbar's width between two panes).
 */
export const BRIDGE_PX = 16;

/**
 * Like edges of two parts closer than this many pixels are aligned outward
 * first, so a popover a few pixels lower than its trigger gives a clean L
 * rather than a tiny step one corner radius tall.
 */
export const SNAP_PX = 10;

/**
 * The outline of the union of `rects`, each grown by `pad` on every side.
 *
 * Returns closed loops of corner points. Outer boundaries run clockwise on
 * screen (y down); a hole in the union would run the other way, which an SVG
 * `evenodd` fill renders correctly either way. Collinear points are dropped,
 * so a lone rect is exactly its four corners.
 *
 * @param {Box[]} rects
 * @param {number} pad
 * @param {number} [bridge] fill gaps up to this many pixels between parts
 * @returns {Point[][]}
 */
export function unionOutline(rects, pad, bridge = BRIDGE_PX) {
  const boxes = rects
    .map((r) => ({
      left: r.left - pad,
      top: r.top - pad,
      right: r.right + pad,
      bottom: r.bottom + pad,
    }))
    .filter((b) => b.right > b.left && b.bottom > b.top);
  if (!boxes.length) return [];
  for (const a of boxes) {
    for (const b of boxes) {
      if (Math.abs(a.left - b.left) < SNAP_PX) a.left = b.left = Math.min(a.left, b.left);
      if (Math.abs(a.top - b.top) < SNAP_PX) a.top = b.top = Math.min(a.top, b.top);
      if (Math.abs(a.right - b.right) < SNAP_PX) a.right = b.right = Math.max(a.right, b.right);
      if (Math.abs(a.bottom - b.bottom) < SNAP_PX) {
        a.bottom = b.bottom = Math.max(a.bottom, b.bottom);
      }
    }
  }

  const xs = [...new Set(boxes.flatMap((b) => [b.left, b.right]))].sort((a, b) => a - b);
  const ys = [...new Set(boxes.flatMap((b) => [b.top, b.bottom]))].sort((a, b) => a - b);
  const nx = xs.length - 1;
  const ny = ys.length - 1;

  /** cover[j][i]: the cell between xs[i]..xs[i+1] and ys[j]..ys[j+1] is covered. */
  const cover = Array.from({ length: ny }, (_, j) =>
    Array.from({ length: nx }, (_, i) => {
      const cx = (xs[i] + xs[i + 1]) / 2;
      const cy = (ys[j] + ys[j + 1]) / 2;
      return boxes.some((b) => cx > b.left && cx < b.right && cy > b.top && cy < b.bottom);
    })
  );

  // Bridge: a run of empty cells with covered cells on both ends of it, in one
  // row (or one column), narrower than `bridge` in total, is filled. Read off
  // the unbridged grid so one filled gap cannot license the next.
  const filled = cover.map((row) => row.slice());
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      if (cover[j][i]) continue;
      let a = i;
      while (a > 0 && !cover[j][a - 1]) a--;
      let b = i;
      while (b < nx - 1 && !cover[j][b + 1]) b++;
      if (a > 0 && b < nx - 1 && xs[b + 1] - xs[a] <= bridge) filled[j][i] = true;
    }
  }
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) {
      if (cover[j][i]) continue;
      let a = j;
      while (a > 0 && !cover[a - 1][i]) a--;
      let b = j;
      while (b < ny - 1 && !cover[b + 1][i]) b++;
      if (a > 0 && b < ny - 1 && ys[b + 1] - ys[a] <= bridge) filled[j][i] = true;
    }
  }

  /** @param {number} i @param {number} j */
  const on = (i, j) => i >= 0 && j >= 0 && i < nx && j < ny && filled[j][i];

  // Directed border edges in grid-index space, each with the covered cell on
  // its right-hand side as drawn (clockwise round a covered region).
  /** @type {Map<string, [number, number][]>} start "i,j" → ends */
  const out = new Map();
  /** @param {number} i0 @param {number} j0 @param {number} i1 @param {number} j1 */
  const edge = (i0, j0, i1, j1) => {
    const k = `${i0},${j0}`;
    const ends = out.get(k);
    if (ends) ends.push([i1, j1]);
    else out.set(k, [[i1, j1]]);
  };
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      if (!on(i, j)) continue;
      if (!on(i, j - 1)) edge(i, j, i + 1, j); // top, left → right
      if (!on(i + 1, j)) edge(i + 1, j, i + 1, j + 1); // right, down
      if (!on(i, j + 1)) edge(i + 1, j + 1, i, j + 1); // bottom, right → left
      if (!on(i - 1, j)) edge(i, j + 1, i, j); // left, up
    }
  }

  /** @type {Point[][]} */
  const loops = [];
  for (const [startKey, startEnds] of out) {
    while (startEnds.length) {
      const [si, sj] = startKey.split(',').map(Number);
      /** @type {[number, number][]} */
      const path = [[si, sj]];
      let [pi, pj] = [si, sj];
      let next = /** @type {[number, number]} */ (startEnds.shift());
      while (next[0] !== si || next[1] !== sj) {
        const [ci, cj] = next;
        path.push([ci, cj]);
        const ends = out.get(`${ci},${cj}`) || [];
        // Two ways on only where cells touch at a corner: take the right
        // turn, which keeps the two cells' outlines apart.
        let pick = 0;
        if (ends.length > 1) {
          const dx = ci - pi;
          const dy = cj - pj;
          const right = ends.findIndex(([ei, ej]) => ei - ci === -dy && ej - cj === dx);
          if (right >= 0) pick = right;
        }
        [pi, pj] = [ci, cj];
        next = ends.splice(pick, 1)[0];
        if (!next) break; // unreachable for a consistent grid; never loop forever
      }
      loops.push(simplify(path.map(([i, j]) => ({ x: xs[i], y: ys[j] }))));
    }
  }
  return loops.filter((l) => l.length >= 4);
}

/**
 * Drop the points that sit on a straight run (grid lines another part
 * introduced), keeping only the corners.
 * @param {Point[]} pts
 * @returns {Point[]}
 */
function simplify(pts) {
  const n = pts.length;
  return pts.filter((p, k) => {
    const a = pts[(k - 1 + n) % n];
    const b = pts[(k + 1) % n];
    return !((a.x === p.x && p.x === b.x) || (a.y === p.y && p.y === b.y));
  });
}

/** @param {number} v */
const fmt = (v) => String(Math.round(v * 100) / 100);

/**
 * An SVG path for `loops` with every corner rounded to `radius` — shrunk
 * where an edge is too short for two full arcs — so a lone rect draws as the
 * same rounded rectangle the old box spotlight was. Inner (concave) corners
 * round the other way, which reads as one smooth outline.
 *
 * @param {Point[][]} loops from unionOutline()
 * @param {number} radius
 * @returns {string}
 */
export function outlinePath(loops, radius) {
  return loops
    .map((pts) => {
      const n = pts.length;
      /** @param {Point} p @param {Point} q @param {number} d */
      const toward = (p, q, d) => {
        const len = Math.hypot(q.x - p.x, q.y - p.y) || 1;
        return { x: p.x + ((q.x - p.x) * d) / len, y: p.y + ((q.y - p.y) * d) / len };
      };
      const corners = pts.map((p, k) => {
        const a = pts[(k - 1 + n) % n];
        const b = pts[(k + 1) % n];
        const r = Math.max(
          0,
          Math.min(
            radius,
            Math.hypot(p.x - a.x, p.y - a.y) / 2,
            Math.hypot(b.x - p.x, b.y - p.y) / 2
          )
        );
        // Screen y points down, so a positive cross product is a clockwise
        // turn — SVG's sweep-flag 1.
        const cross = (p.x - a.x) * (b.y - p.y) - (p.y - a.y) * (b.x - p.x);
        return { r, sweep: cross > 0 ? 1 : 0, in: toward(p, a, r), out: toward(p, b, r) };
      });
      let d = `M${fmt(corners[0].out.x)} ${fmt(corners[0].out.y)}`;
      for (let k = 1; k <= n; k++) {
        const c = corners[k % n];
        d += `L${fmt(c.in.x)} ${fmt(c.in.y)}`;
        if (c.r > 0) d += `A${fmt(c.r)} ${fmt(c.r)} 0 0 ${c.sweep} ${fmt(c.out.x)} ${fmt(c.out.y)}`;
      }
      return `${d}Z`;
    })
    .join('');
}

/**
 * The rects `t` of the way from `from` to `to` (0 → from, 1 → to), for the
 * spotlight's move between steps. Rects pair up by index; the shorter list
 * lends its last rect to the extra ones, so a part that appears grows out of
 * the previous spotlight and a part that goes shrinks into the next.
 *
 * @param {Box[]} from
 * @param {Box[]} to
 * @param {number} t
 * @returns {Box[]}
 */
export function lerpBoxes(from, to, t) {
  if (!from.length) return to.slice();
  if (!to.length) return from.slice();
  const n = Math.max(from.length, to.length);
  /** @param {number} a @param {number} b */
  const mix = (a, b) => a + (b - a) * t;
  return Array.from({ length: n }, (_, k) => {
    const a = from[Math.min(k, from.length - 1)];
    const b = to[Math.min(k, to.length - 1)];
    return {
      left: mix(a.left, b.left),
      top: mix(a.top, b.top),
      right: mix(a.right, b.right),
      bottom: mix(a.bottom, b.bottom),
    };
  });
}

/** Room left between a spotlighted part and the outline round it, in px. */
const PAD = 6;

/** The spotlight's move between steps: --wt-transition-normal's 200 ms. */
const MOVE_MS = 200;

const SVG_NS = 'http://www.w3.org/2000/svg';

/**
 * The spotlight layer: one viewport-sized SVG whose dim path is the viewport
 * with the outline cut out (`evenodd`), and whose ring and glow stroke the
 * same outline, clipped to its outside so they frame the parts without
 * covering them — the box-shadow cutout this replaced, in any shape.
 *
 * `move(boxes)` eases the outline from wherever it is drawn now (a call
 * mid-move continues from there), or draws at once the first time and when
 * the viewer asks for reduced motion. Removing `el` ends any move under way.
 *
 * @returns {{el: HTMLElement, move: (boxes: Box[]) => void}}
 */
export function createSpotlight() {
  const el = document.createElement('div');
  el.className = 'tour-spot';
  /** @param {string} tag @param {Record<string, string>} [attrs] */
  const svgEl = (tag, attrs = {}) => {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    return node;
  };
  const clipPath = svgEl('path', { 'clip-rule': 'evenodd' });
  const clip = svgEl('clipPath', { id: 'tour-spot-outside' });
  clip.appendChild(clipPath);
  const defs = svgEl('defs');
  defs.appendChild(clip);
  const dim = svgEl('path', { class: 'tour-spot-dim', 'fill-rule': 'evenodd' });
  const glow = svgEl('path', { class: 'tour-spot-glow' });
  const ring = svgEl('path', { class: 'tour-spot-ring' });
  const frame = svgEl('g', { 'clip-path': 'url(#tour-spot-outside)' });
  frame.append(glow, ring);
  const svg = svgEl('svg', { class: 'tour-spot-svg', 'aria-hidden': 'true' });
  svg.append(defs, dim, frame);
  el.appendChild(svg);

  /** The boxes the outline is drawn round right now, mid-move included. @type {Box[]} */
  let shown = [];
  /** @type {number | null} */
  let pending = null;

  /** @param {Box[]} boxes */
  const draw = (boxes) => {
    shown = boxes;
    const radius =
      parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--radius-xl')) || 8;
    const hole = outlinePath(unionOutline(boxes, PAD), radius);
    const outside = `M0 0H${window.innerWidth}V${window.innerHeight}H0Z${hole}`;
    dim.setAttribute('d', outside);
    clipPath.setAttribute('d', outside);
    ring.setAttribute('d', hole);
    glow.setAttribute('d', hole);
  };

  /** @param {Box[]} boxes */
  const move = (boxes) => {
    if (pending != null) cancelAnimationFrame(pending);
    pending = null;
    const still =
      !shown.length ||
      typeof requestAnimationFrame !== 'function' ||
      window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    if (still) return draw(boxes);
    const from = shown;
    const t0 = performance.now();
    const step = () => {
      pending = null;
      if (!el.isConnected) return;
      const t = Math.min(1, (performance.now() - t0) / MOVE_MS);
      draw(t < 1 ? lerpBoxes(from, boxes, 1 - (1 - t) ** 3) : boxes);
      if (t < 1) pending = requestAnimationFrame(step);
    };
    pending = requestAnimationFrame(step);
  };

  return { el, move };
}
