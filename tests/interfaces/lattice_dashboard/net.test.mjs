// @ts-nocheck
// TODO(frontend-hardening Pn): remove & fix types when this interface is retrofitted (P2–P5)
/**
 * Unit tests for the Lattice Dashboard network layer (net.js).
 *
 * Pure logic guard, happy-dom environment (configured globally), fetch mocked:
 *   npx vitest run tests/interfaces/lattice_dashboard/net.test.mjs
 *
 * Covers apiFetch() success/error propagation and the handleSSEEvent()
 * dispatch table. Does NOT exercise createNetClient()'s connectSSE() —
 * the live EventSource wiring is covered by the browser load-smoke test
 * (tests/interfaces/test_load_smokes.py -m browser -k lattice).
 */

import { test, expect, vi, describe, afterEach } from 'vitest';

import {
  apiFetch,
  handleSSEEvent,
} from '../../../src/osprey/interfaces/lattice_dashboard/static/js/net.js';

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('apiFetch', () => {
  test('resolves with the parsed JSON body on a 200 response', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ base_lattice: 'sr.lat' }),
    }));

    const result = await apiFetch('/api/state');
    expect(result).toEqual({ base_lattice: 'sr.lat' });
    expect(fetch).toHaveBeenCalledWith('/api/state', expect.objectContaining({
      headers: { 'Content-Type': 'application/json' },
    }));
  });

  test('propagates a non-OK response as a thrown Error carrying status and body text', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
      text: () => Promise.resolve('lattice not loaded'),
    }));

    await expect(apiFetch('/api/refresh', { method: 'POST' })).rejects.toThrow(
      'API /api/refresh: 500 lattice not loaded'
    );
  });

  test('a rejected fetch (network failure) propagates as-is', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('network down')));

    await expect(apiFetch('/api/state')).rejects.toThrow('network down');
  });
});

describe('handleSSEEvent dispatch table', () => {
  function makeHandlers() {
    return {
      onStateUpdated: vi.fn(),
      onFigureStatus: vi.fn(),
      onFigureReady: vi.fn(),
      onFigureError: vi.fn(),
      onSettingsUpdated: vi.fn(),
      onBaselineSet: vi.fn(),
      onBaselineCleared: vi.fn(),
    };
  }

  test("'state_updated' routes to onStateUpdated with no args", () => {
    const handlers = makeHandlers();
    handleSSEEvent({ type: 'state_updated' }, handlers);
    expect(handlers.onStateUpdated).toHaveBeenCalledTimes(1);
    expect(handlers.onStateUpdated).toHaveBeenCalledWith();
  });

  test("'figure_status' routes to onFigureStatus with (name, status)", () => {
    const handlers = makeHandlers();
    handleSSEEvent({ type: 'figure_status', name: 'optics', status: 'computing' }, handlers);
    expect(handlers.onFigureStatus).toHaveBeenCalledWith('optics', 'computing');
    expect(handlers.onFigureReady).not.toHaveBeenCalled();
  });

  test("'figure_ready' routes to onFigureReady with (name) only — fetching is the caller's job", () => {
    const handlers = makeHandlers();
    handleSSEEvent({ type: 'figure_ready', name: 'da' }, handlers);
    expect(handlers.onFigureReady).toHaveBeenCalledWith('da');
    expect(handlers.onFigureStatus).not.toHaveBeenCalled();
  });

  test("'figure_error' routes to onFigureError with (name, error)", () => {
    const handlers = makeHandlers();
    handleSSEEvent({ type: 'figure_error', name: 'lma', error: 'solver diverged' }, handlers);
    expect(handlers.onFigureError).toHaveBeenCalledWith('lma', 'solver diverged');
  });

  test("'figure_error' with no error message falls back to 'Unknown error'", () => {
    const handlers = makeHandlers();
    handleSSEEvent({ type: 'figure_error', name: 'lma' }, handlers);
    expect(handlers.onFigureError).toHaveBeenCalledWith('lma', 'Unknown error');
  });

  test("'settings_updated' routes to onSettingsUpdated only when settings is present", () => {
    const handlers = makeHandlers();
    handleSSEEvent({ type: 'settings_updated', settings: { da: { nturns: 512 } } }, handlers);
    expect(handlers.onSettingsUpdated).toHaveBeenCalledWith({ da: { nturns: 512 } });

    handlers.onSettingsUpdated.mockClear();
    handleSSEEvent({ type: 'settings_updated' }, handlers);
    expect(handlers.onSettingsUpdated).not.toHaveBeenCalled();
  });

  test("'baseline_set' routes to onBaselineSet only when summary is present", () => {
    const handlers = makeHandlers();
    handleSSEEvent({ type: 'baseline_set', summary: { energy_gev: 2.0 } }, handlers);
    expect(handlers.onBaselineSet).toHaveBeenCalledWith({ energy_gev: 2.0 });

    handlers.onBaselineSet.mockClear();
    handleSSEEvent({ type: 'baseline_set' }, handlers);
    expect(handlers.onBaselineSet).not.toHaveBeenCalled();
  });

  test("'baseline_cleared' routes to the optional onBaselineCleared handler", () => {
    const handlers = makeHandlers();
    handleSSEEvent({ type: 'baseline_cleared' }, handlers);
    expect(handlers.onBaselineCleared).toHaveBeenCalledTimes(1);
  });

  test("'baseline_cleared' is a no-op when onBaselineCleared is omitted", () => {
    const handlers = makeHandlers();
    delete handlers.onBaselineCleared;
    expect(() => handleSSEEvent({ type: 'baseline_cleared' }, handlers)).not.toThrow();
  });

  test('an unrecognized event type dispatches to nothing', () => {
    const handlers = makeHandlers();
    handleSSEEvent({ type: 'something_new' }, handlers);
    for (const fn of Object.values(handlers)) {
      expect(fn).not.toHaveBeenCalled();
    }
  });
});
