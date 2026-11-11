/** Unit tests for the live run series (live-plot.js). */

import { test, expect, describe, vi, afterEach } from 'vitest';

import { runningJob } from './_fixtures.mjs';
import {
  liveTraces,
  normalize,
  progress,
  renderLivePlot,
  runningBest,
} from '../../../src/osprey/interfaces/optimizer_web/panels/optimizer/live-plot.js';

afterEach(() => {
  delete (/** @type {any} */ (globalThis).Plotly);
});

describe('runningBest', () => {
  test('follows the direction, skips nulls, refuses an unknown direction', () => {
    expect(runningBest([1, 3, null, 2], 'MAXIMIZE')).toEqual([1, 3, 3, 3]);
    expect(runningBest([null, 3, 1, 2], 'MINIMIZE')).toEqual([null, 3, 1, 1]);
    expect(runningBest([1, 2], null)).toBeNull();
  });
});

test('progress: n of N and best so far', () => {
  expect(progress(runningJob())).toEqual({ done: 3, total: 10, best: 2.5, percent: 30 });
  expect(progress(runningJob({ total_evaluations: null, data: [] }))).toEqual({
    done: 0,
    total: null,
    best: null,
    percent: null,
  });
});

test('normalize maps into the run bounds', () => {
  expect(normalize(0, [-18, 18])).toBe(0.5);
  expect(normalize(3, undefined)).toBe(3);
  expect(normalize(null, [0, 1])).toBeNull();
});

test('traces: objective + best on the top axis, normalized variables below', () => {
  const traces = liveTraces(runningJob());
  expect(traces.map((t) => [t.name, t.yaxis])).toEqual([
    ['charge', 'y'],
    ['Best so far', 'y'],
    ['phase_a', 'y2'],
    ['sol_b', 'y2'],
  ]);
  expect(traces[1].y).toEqual([1, 2.5, 2.5]);
  expect(traces[2].y).toEqual([0.5, 0.75, 0.25]);
});

test('renderLivePlot draws through Plotly.react, and is a no-op without Plotly', () => {
  const gd = document.createElement('div');
  expect(renderLivePlot(gd, runningJob())).toBe(false);
  const react = vi.fn();
  /** @type {any} */ (globalThis).Plotly = { react };
  vi.spyOn(console, 'error').mockImplementation(() => {});
  expect(renderLivePlot(gd, runningJob())).toBe(true);
  expect(react).toHaveBeenCalledOnce();
  expect(react.mock.calls[0][0]).toBe(gd);
});
