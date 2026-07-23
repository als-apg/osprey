// @ts-check
/**
 * Unit tests for activity-strip.js — the slim host-chrome zone showing the
 * most recent agent action. Frames are injected by calling the strip's
 * handleActivity directly (the same seam panel-manager's SSE dispatch drives
 * via setActivityStripHandler); the active panel is stubbed through the
 * injectable getActivePanel dependency. Covers:
 *
 *   - channel frames render the channel list and auto-clear on the timeout
 *   - coalescing: newer frames (same or different target) replace the single
 *     visible entry and reset the timer (latest wins)
 *   - suppression: artifact frames while 'artifacts' is active, run frames
 *     while 'plan' is active, panel-kind frames while their own panel is
 *     active — all suppressed; shown otherwise, and a suppressed frame never
 *     disturbs an already-visible entry or its timer
 *   - agent-supplied strings land as text nodes only (no element injection)
 *   - unknown kinds render the generic "agent activity" + tool fallback
 *   - the pure suppression helpers for all four kinds
 *
 *   npx vitest run tests/interfaces/web_terminal/activity-strip.test.mjs
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';
import {
  createActivityStrip, suppressionPanelFor, isSuppressed, ACTIVITY_CLEAR_MS,
} from '../../../src/osprey/interfaces/web_terminal/static/js/activity-strip.js';

/** @typedef {import('../../../src/osprey/interfaces/web_terminal/static/js/panel-manager.js').AgentActivityEvent} AgentActivityFrame */

/**
 * Build an agent_activity frame as broadcast by the server (optional keys
 * omitted when absent, per the SSE contract).
 * @param {AgentActivityFrame['target']} target
 * @param {string} [tool]
 * @returns {AgentActivityFrame}
 */
function frame(target, tool = 'write_channel') {
  return { type: 'agent_activity', tool, target, ts: 1234 };
}

/** @type {HTMLElement} */
let mount;
/** @type {string | null} */
let activePanel;

/** @param {number} [clearMs] */
function makeStrip(clearMs) {
  return createActivityStrip({ mount, getActivePanel: () => activePanel, clearMs });
}

beforeEach(() => {
  vi.useFakeTimers();
  document.body.innerHTML = '<div id="strip"></div>';
  mount = /** @type {HTMLElement} */ (document.getElementById('strip'));
  activePanel = null;
});

afterEach(() => {
  vi.useRealTimers();
  document.body.innerHTML = '';
});

describe('channel frames: render + auto-clear', () => {
  test('entry text contains the channels and clears after ACTIVITY_CLEAR_MS', () => {
    const strip = makeStrip();
    strip.handleActivity(frame({ kind: 'channel', detail: 'SR01:HCM1:SP' }));

    expect(mount.textContent).toContain('agent wrote');
    expect(mount.textContent).toContain('SR01:HCM1:SP');

    vi.advanceTimersByTime(ACTIVITY_CLEAR_MS - 1);
    expect(mount.textContent).toContain('SR01:HCM1:SP');
    vi.advanceTimersByTime(1);
    expect(mount.textContent).toBe('');
    expect(mount.children.length).toBe(0);
  });
});

describe('coalescing: single slot, latest wins', () => {
  test('newer frame for the same target replaces the entry and resets the timer', () => {
    const strip = makeStrip();
    strip.handleActivity(frame({ kind: 'channel', detail: 'SR01:HCM1:SP' }));
    vi.advanceTimersByTime(ACTIVITY_CLEAR_MS - 2000);

    // Same target again — entry stays, timer restarts from now.
    strip.handleActivity(frame({ kind: 'channel', detail: 'SR01:HCM1:SP' }));
    expect(mount.querySelectorAll('.activity-strip-entry').length).toBe(1);

    // Past the ORIGINAL deadline but within the reset one: still visible.
    vi.advanceTimersByTime(ACTIVITY_CLEAR_MS - 2000);
    expect(mount.textContent).toContain('SR01:HCM1:SP');

    vi.advanceTimersByTime(2000);
    expect(mount.textContent).toBe('');
  });

  test('newer frame for a different target replaces the entry (latest wins)', () => {
    const strip = makeStrip();
    strip.handleActivity(frame({ kind: 'channel', detail: 'SR01:HCM1:SP' }));
    strip.handleActivity(frame({ kind: 'run', detail: 'orm-42' }, 'run_plan'));

    expect(mount.querySelectorAll('.activity-strip-entry').length).toBe(1);
    expect(mount.textContent).toContain('agent launched run');
    expect(mount.textContent).toContain('orm-42');
    expect(mount.textContent).not.toContain('SR01:HCM1:SP');
  });
});

describe('suppression: active panel self-signals', () => {
  test('artifact frame while artifacts panel is active is suppressed', () => {
    activePanel = 'artifacts';
    const strip = makeStrip();
    const f = frame({ kind: 'artifact', detail: 'orbit-plot.png' }, 'focus_artifact');
    expect(isSuppressed(f.target, activePanel)).toBe(true);
    strip.handleActivity(f);
    expect(mount.children.length).toBe(0);
    expect(mount.textContent).toBe('');
  });

  test('artifact frame while another panel is active is shown', () => {
    activePanel = 'lattice';
    const strip = makeStrip();
    strip.handleActivity(frame({ kind: 'artifact', detail: 'orbit-plot.png' }, 'focus_artifact'));
    expect(mount.textContent).toContain('agent focused');
    expect(mount.textContent).toContain('orbit-plot.png');
  });

  test('run frame while plan panel is active is suppressed, shown otherwise', () => {
    activePanel = 'plan';
    const strip = makeStrip();
    strip.handleActivity(frame({ kind: 'run', detail: 'grid-7' }, 'run_plan'));
    expect(mount.children.length).toBe(0);

    activePanel = 'artifacts';
    strip.handleActivity(frame({ kind: 'run', detail: 'grid-7' }, 'run_plan'));
    expect(mount.textContent).toContain('grid-7');
  });

  test('suppressed frame leaves an already-visible entry and its timer intact', () => {
    const strip = makeStrip();
    strip.handleActivity(frame({ kind: 'channel', detail: 'SR01:HCM1:SP' }));
    vi.advanceTimersByTime(ACTIVITY_CLEAR_MS - 2000);

    // A suppressed frame must neither replace the entry nor reset its clock.
    activePanel = 'artifacts';
    strip.handleActivity(frame({ kind: 'artifact', detail: 'orbit-plot.png' }, 'focus_artifact'));
    expect(mount.textContent).toContain('SR01:HCM1:SP');
    expect(mount.textContent).not.toContain('orbit-plot.png');

    // The original deadline still stands: the entry clears exactly on it.
    vi.advanceTimersByTime(1999);
    expect(mount.textContent).toContain('SR01:HCM1:SP');
    vi.advanceTimersByTime(1);
    expect(mount.textContent).toBe('');
  });

  test('panel-kind fallback frame while that panel is active is suppressed, shown otherwise', () => {
    activePanel = 'lattice';
    const strip = makeStrip();
    strip.handleActivity(frame({ kind: 'panel', panel: 'lattice' }, 'switch_panel'));
    expect(mount.children.length).toBe(0);

    activePanel = 'artifacts';
    strip.handleActivity(frame({ kind: 'panel', panel: 'lattice' }, 'switch_panel'));
    expect(mount.textContent).toContain('lattice');
  });
});

describe('agent strings are text nodes only', () => {
  test('markup in detail appears as literal text, no element is created', () => {
    const strip = makeStrip();
    const payload = '<img src=x onerror=alert(1)>';
    strip.handleActivity(frame({ kind: 'channel', detail: payload }));

    expect(mount.querySelector('img')).toBeNull();
    expect(mount.textContent).toContain(payload);
  });
});

describe('unknown target kinds', () => {
  test('an unrecognized kind renders the generic "agent activity" + tool fallback', () => {
    const strip = makeStrip();
    strip.handleActivity(frame(
      /** @type {any} */ ({ kind: 'hologram', detail: 'ignored' }),
      'future_tool',
    ));

    expect(mount.textContent).toContain('agent activity');
    expect(mount.textContent).toContain('future_tool');
  });
});

describe('pure suppression helpers', () => {
  test('suppressionPanelFor maps each kind onto its self-signaling panel', () => {
    expect(suppressionPanelFor({ kind: 'artifact' })).toBe('artifacts');
    expect(suppressionPanelFor({ kind: 'run' })).toBe('plan');
    expect(suppressionPanelFor({ kind: 'panel', panel: 'lattice' })).toBe('lattice');
    expect(suppressionPanelFor({ kind: 'panel' })).toBeNull(); // malformed: no panel id
    expect(suppressionPanelFor({ kind: 'channel' })).toBeNull();
  });

  test('isSuppressed for all four kinds', () => {
    // channel is never suppressed, whatever is active
    expect(isSuppressed({ kind: 'channel', detail: 'SR01:HCM1:SP' }, 'artifacts')).toBe(false);
    expect(isSuppressed({ kind: 'channel', detail: 'SR01:HCM1:SP' }, null)).toBe(false);

    expect(isSuppressed({ kind: 'artifact' }, 'artifacts')).toBe(true);
    expect(isSuppressed({ kind: 'artifact' }, 'plan')).toBe(false);

    expect(isSuppressed({ kind: 'run' }, 'plan')).toBe(true);
    expect(isSuppressed({ kind: 'run' }, 'artifacts')).toBe(false);

    expect(isSuppressed({ kind: 'panel', panel: 'okf' }, 'okf')).toBe(true);
    expect(isSuppressed({ kind: 'panel', panel: 'okf' }, 'ariel')).toBe(false);

    // no active panel suppresses nothing
    expect(isSuppressed({ kind: 'artifact' }, null)).toBe(false);
  });
});
