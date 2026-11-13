import { describe, expect, it } from 'vitest';
import {
  diffusionColor,
  esc,
  extent,
  fmt,
  plot,
  resonanceSegments,
  ticks,
} from '../panel/chart.js';

describe('fmt', () => {
  it('prints six significant digits without trailing zeros', () => {
    expect(fmt(14.129998781)).toBe('14.13');
    expect(fmt(0.219992311)).toBe('0.219992');
    expect(fmt(0)).toBe('0');
    expect(fmt(3.2e-7)).toBe('3.200e-7');
  });

  it('prints a dash for a missing number and a string as it is', () => {
    expect(fmt(null)).toBe('—');
    expect(fmt(Number.NaN)).toBe('—');
    expect(fmt('pyat')).toBe('pyat');
  });
});

describe('extent', () => {
  it('skips gaps and pads the span', () => {
    expect(extent([[0, null, 10]], 0.1)).toEqual([-1, 11]);
  });

  it('answers a unit range for no data', () => {
    expect(extent([[null]])).toEqual([0, 1]);
  });
});

describe('ticks', () => {
  it('lands on round values inside the range', () => {
    expect(ticks(0, 1, 5)).toEqual([0, 0.2, 0.4, 0.6, 0.8, 1]);
  });
});

describe('resonanceSegments', () => {
  it('draws the integer and half-integer lines of a window', () => {
    const segments = resonanceSegments([0.1, 0.9], [0.1, 0.9], 2);
    const half = segments.filter((s) => s.order === 2 && s.x1 === 0.5 && s.x2 === 0.5);
    expect(half).toHaveLength(1);
    // νx + νy = 1 crosses the window from corner to corner.
    const sum = segments.find((s) => s.y1 > s.y2);
    expect(sum).toMatchObject({ x1: 0.1, x2: 0.9, order: 2 });
    expect(sum?.y1).toBeCloseTo(0.9);
    expect(sum?.y2).toBeCloseTo(0.1);
    // No integer line lies inside (0.1, 0.9).
    expect(segments.filter((s) => s.order === 1)).toHaveLength(0);
  });

  it('keeps every segment inside the window', () => {
    for (const s of resonanceSegments([0.05, 0.35], [0.1, 0.4], 4)) {
      for (const y of [s.y1, s.y2]) {
        expect(y).toBeGreaterThanOrEqual(0.1 - 1e-9);
        expect(y).toBeLessThanOrEqual(0.4 + 1e-9);
      }
    }
  });
});

describe('diffusionColor', () => {
  it('runs from blue for a stable particle to red for a chaotic one', () => {
    expect(diffusionColor(-10)).toBe('hsl(240 75% 50%)');
    expect(diffusionColor(-2)).toBe('hsl(0 75% 50%)');
    expect(diffusionColor(null)).toBe('var(--chart-axis-text)');
  });
});

describe('plot', () => {
  it('breaks a line at a gap', () => {
    const svg = plot({
      xLabel: 's [m]',
      yLabel: 'β [m]',
      series: [{ label: 'βx', color: 'red', x: [0, 1, 2, 3, 4], y: [1, 2, null, 2, 1] }],
    });
    expect(svg.match(/<path class="lt-line"/g)).toHaveLength(2);
    expect(svg).toContain('aria-label="β [m] against s [m]"');
  });

  it('draws points in their own colours', () => {
    const svg = plot({
      xLabel: 'x',
      yLabel: 'y',
      series: [
        {
          label: 'p',
          color: 'red',
          x: [0, 1],
          y: [0, 1],
          mode: 'points',
          colors: ['blue', 'green'],
        },
      ],
    });
    expect(svg.match(/<circle/g)).toHaveLength(2);
    expect(svg).toContain('fill:blue');
    expect(svg).not.toContain('<path');
  });

  it('escapes a label', () => {
    expect(esc('<b>')).toBe('&lt;b&gt;');
    const svg = plot({ xLabel: '<x>', yLabel: 'y', series: [] });
    expect(svg).not.toContain('<x>');
  });
});
