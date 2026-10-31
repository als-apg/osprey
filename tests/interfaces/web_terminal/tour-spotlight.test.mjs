/**
 * Tour spotlight geometry (tour-spotlight.js): the outline is the union of the
 * padded parts, not their bounding box.
 *
 * The case that motivated it is the control-target step: the sidebar chip
 * and the popover beside it, one outline round both, and the rail rows under
 * the chip left outside it.
 */

import { afterEach, describe, expect, test, vi } from 'vitest';

import {
  BRIDGE_PX,
  SNAP_PX,
  createSpotlight,
  lerpBoxes,
  outlinePath,
  unionOutline,
} from '../../../src/osprey/interfaces/web_terminal/static/js/tour-spotlight.js';

/** @param {number} left @param {number} top @param {number} right @param {number} bottom */
const box = (left, top, right, bottom) => ({ left, top, right, bottom });

/** @typedef {{x: number, y: number}} Point */

/**
 * Even-odd point-in-polygon over every loop.
 * @param {Point[][]} loops @param {number} x @param {number} y
 */
const inside = (loops, x, y) => {
  let hit = false;
  for (const pts of loops) {
    for (let k = 0, m = pts.length - 1; k < pts.length; m = k++) {
      const a = pts[k];
      const b = pts[m];
      if (a.y > y !== b.y > y && x < ((b.x - a.x) * (y - a.y)) / (b.y - a.y) + a.x) hit = !hit;
    }
  }
  return hit;
};

/**
 * Shoelace area; positive for a clockwise loop on screen (y down).
 * @param {Point[]} pts
 */
const area = (pts) =>
  pts.reduce((/** @type {number} */ s, p, k) => {
    const q = pts[(k + 1) % pts.length];
    return s + (p.x * q.y - q.x * p.y);
  }, 0) / 2;

describe('unionOutline', () => {
  test('a lone rect is its padded four corners, clockwise', () => {
    const loops = unionOutline([box(10, 20, 110, 60)], 6);
    expect(loops).toEqual([
      [
        { x: 4, y: 14 },
        { x: 116, y: 14 },
        { x: 116, y: 66 },
        { x: 4, y: 66 },
      ],
    ]);
    expect(area(loops[0])).toBeGreaterThan(0);
  });

  test('chip + popover beside it: one L-shaped outline, rail rows left out', () => {
    // The chip at the top of the rail, the popover 4px to its right and taller.
    const chip = box(8, 10, 240, 50);
    const popover = box(244, 10, 520, 260);
    const loops = unionOutline([chip, popover], 6);

    expect(loops).toHaveLength(1);
    expect(loops[0]).toHaveLength(6); // an L: six corners, not a box's four
    expect(inside(loops, 100, 30)).toBe(true); // the chip
    expect(inside(loops, 400, 200)).toBe(true); // the popover
    expect(inside(loops, 242, 30)).toBe(true); // the gap between them
    expect(inside(loops, 100, 150)).toBe(false); // a rail row under the chip
    expect(area(loops[0])).toBeGreaterThan(0);
  });

  test('a popover a few pixels lower than its chip still gives a clean L', () => {
    expect(SNAP_PX).toBeGreaterThan(8);
    const loops = unionOutline([box(8, 48, 240, 88), box(246, 56, 526, 276)], 6);
    expect(loops).toHaveLength(1);
    expect(loops[0]).toHaveLength(6); // no extra two-corner step at the top
    expect(Math.min(...loops[0].map((p) => p.y))).toBe(42); // aligned outward
  });

  test('parts a small gap apart after padding are bridged into one shape', () => {
    // 12 + 2*6 = 24px apart before padding, 12px after: under BRIDGE_PX.
    expect(BRIDGE_PX).toBeGreaterThanOrEqual(12);
    const loops = unionOutline([box(0, 0, 100, 40), box(124, 0, 300, 40)], 6);
    expect(loops).toHaveLength(1);
    expect(inside(loops, 112, 20)).toBe(true);
  });

  test('parts far apart stay two shapes', () => {
    const loops = unionOutline([box(0, 0, 100, 40), box(300, 0, 400, 40)], 6);
    expect(loops).toHaveLength(2);
    expect(inside(loops, 200, 20)).toBe(false);
  });

  test('a bridge never fills a notch that is open on one side', () => {
    // A tall part and a short one beside it: the space under the short one
    // has a neighbour on one side only and stays dark.
    const loops = unionOutline([box(0, 0, 100, 300), box(100, 0, 400, 40)], 6);
    expect(loops).toHaveLength(1);
    expect(inside(loops, 250, 200)).toBe(false);
  });

  test('overlapping and nested parts union cleanly', () => {
    const loops = unionOutline([box(0, 0, 200, 200), box(50, 50, 100, 100)], 6);
    expect(loops).toEqual(unionOutline([box(0, 0, 200, 200)], 6));
  });

  test('empty and zero-size parts give no outline', () => {
    expect(unionOutline([], 6)).toEqual([]);
    expect(unionOutline([box(5, 5, 5, 5)], 0)).toEqual([]);
  });
});

describe('outlinePath', () => {
  test('a lone rect draws as a rounded rectangle', () => {
    const d = outlinePath(unionOutline([box(10, 20, 110, 60)], 6), 8);
    expect(d).toBe(
      'M12 14L108 14A8 8 0 0 1 116 22L116 58A8 8 0 0 1 108 66' +
        'L12 66A8 8 0 0 1 4 58L4 22A8 8 0 0 1 12 14Z'
    );
  });

  test('the L rounds its inner corner the other way', () => {
    const d = outlinePath(unionOutline([box(8, 10, 240, 50), box(244, 10, 520, 260)], 6), 8);
    const sweeps = [...d.matchAll(/A[\d.]+ [\d.]+ 0 0 ([01])/g)].map((m) => m[1]);
    expect(sweeps).toHaveLength(6);
    expect(sweeps.filter((f) => f === '0')).toHaveLength(1);
  });

  test('a corner radius never exceeds half a short edge', () => {
    const d = outlinePath(
      [
        [
          { x: 0, y: 0 },
          { x: 4, y: 0 },
          { x: 4, y: 100 },
          { x: 0, y: 100 },
        ],
      ],
      8
    );
    expect(d.startsWith('M2 0')).toBe(true);
    expect(d).toContain('A2 2 0 0 1');
  });
});

describe('lerpBoxes', () => {
  test('ends at both lists and mixes in between', () => {
    const a = [box(0, 0, 10, 10)];
    const b = [box(10, 10, 30, 30)];
    expect(lerpBoxes(a, b, 0)).toEqual(a);
    expect(lerpBoxes(a, b, 1)).toEqual(b);
    expect(lerpBoxes(a, b, 0.5)).toEqual([box(5, 5, 20, 20)]);
  });

  test('a part that appears grows out of the previous spotlight', () => {
    const chip = box(0, 0, 100, 40);
    const popover = box(104, 0, 400, 200);
    const mid = lerpBoxes([chip], [chip, popover], 0);
    expect(mid).toEqual([chip, chip]);
    expect(lerpBoxes([chip, popover], [chip], 1)).toEqual([chip, chip]);
  });
});

describe('createSpotlight', () => {
  afterEach(() => {
    document.body.replaceChildren();
    vi.restoreAllMocks();
  });

  test('dims the viewport with the outline cut out, ring on the same outline', () => {
    const spot = createSpotlight();
    document.body.appendChild(spot.el);
    spot.move([box(10, 20, 110, 60)]); // the first draw is immediate
    const ring = spot.el.querySelector('.tour-spot-ring')?.getAttribute('d') ?? '';
    const dim = spot.el.querySelector('.tour-spot-dim');
    expect(ring).toBe(outlinePath(unionOutline([box(10, 20, 110, 60)], 6), 8));
    expect(dim?.getAttribute('fill-rule')).toBe('evenodd');
    expect(dim?.getAttribute('d')).toBe(`M0 0H${innerWidth}V${innerHeight}H0Z${ring}`);
  });

  test('reduced motion moves the outline at once', () => {
    vi.spyOn(window, 'matchMedia').mockReturnValue(
      /** @type {MediaQueryList} */ (/** @type {unknown} */ ({ matches: true }))
    );
    const raf = vi.spyOn(window, 'requestAnimationFrame');
    const spot = createSpotlight();
    document.body.appendChild(spot.el);
    spot.move([box(10, 20, 110, 60)]);
    spot.move([box(200, 20, 300, 60)]);
    expect(raf).not.toHaveBeenCalled();
    expect(spot.el.querySelector('.tour-spot-ring')?.getAttribute('d')).toBe(
      outlinePath(unionOutline([box(200, 20, 300, 60)], 6), 8)
    );
  });
});
