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
 *     while 'bluesky' is active, panel-kind frames while their own panel is
 *     active — all suppressed; shown otherwise, and a suppressed frame never
 *     disturbs an already-visible entry or its timer
 *   - agent-supplied strings land as text nodes only (no element injection)
 *   - unknown kinds render the generic "agent activity" + tool fallback
 *   - the pure suppression helpers for all four kinds
 *   - the click-to-expand history popover: fetch on open, newest-first rows,
 *     live frames prepended while open, Escape / outside-click close,
 *     aria-expanded on the trigger, and the live slot behaving as before
 *
 *   npx vitest run tests/interfaces/web_terminal/activity-strip.test.mjs
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';
import {
  createActivityStrip, suppressionPanelFor, isSuppressed, ACTIVITY_CLEAR_MS,
} from '../../../src/osprey/interfaces/web_terminal/static/js/activity-strip.js';
import {
  formatActivity, formatRelativeTime,
} from '../../../src/osprey/interfaces/web_terminal/static/js/activity-format.js';
import { HISTORY_LIMIT } from '../../../src/osprey/interfaces/web_terminal/static/js/activity-history.js';

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
/** Every strip built in a test, so afterEach can unhook its document listeners. */
/** @type {ReturnType<typeof createActivityStrip>[] } */
let strips;

/**
 * Stand-in for panel-manager's catalog lookup: known ids resolve to their
 * human label, everything else falls back to the raw id (as the real one does).
 * @param {string} id
 */
function labelOf(id) {
  return { lattice: 'Lattice', ariel: 'ARIEL', artifacts: 'Artifacts' }[id] ?? id;
}

/**
 * @param {number} [clearMs]
 * @param {(limit: number) => Promise<AgentActivityFrame[]>} [fetchRecent]
 * @param {(id: string) => string} [labels] omit to test the no-closure fallback
 */
function makeStrip(clearMs, fetchRecent, labels = labelOf) {
  const strip = createActivityStrip({
    mount, getActivePanel: () => activePanel, clearMs, fetchRecent, labelOf: labels,
  });
  strips.push(strip);
  return strip;
}

/** The body-level popover, or null when it is closed. */
function popover() {
  return /** @type {HTMLElement | null} */ (
    document.querySelector('.activity-history-popover')
  );
}

/** Visible history rows, top to bottom. */
function rowTexts() {
  return [...document.querySelectorAll('.activity-history-row')].map((r) => r.textContent);
}

/** Let an in-flight fetch settle while fake timers are installed. */
function flush() {
  return vi.advanceTimersByTimeAsync(0);
}

beforeEach(() => {
  vi.useFakeTimers();
  document.body.innerHTML = '<div id="strip"></div>';
  mount = /** @type {HTMLElement} */ (document.getElementById('strip'));
  activePanel = null;
  strips = [];
});

afterEach(() => {
  // Close first: an open popover leaves capture-phase listeners on the shared
  // document, which would outlive innerHTML = '' and leak into the next test.
  for (const strip of strips) strip.closeHistory();
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

  test('run frame while the bluesky panel is active is suppressed, shown otherwise', () => {
    activePanel = 'bluesky';
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

  test('panel-kind frame while that panel is active is suppressed, shown otherwise', () => {
    activePanel = 'lattice';
    const strip = makeStrip();
    strip.handleActivity(frame({ kind: 'panel', panel: 'lattice' }, 'open_panel'));
    expect(mount.children.length).toBe(0);

    activePanel = 'artifacts';
    strip.handleActivity(frame({ kind: 'panel', panel: 'lattice' }, 'open_panel'));
    expect(mount.textContent).toContain('Lattice');
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

describe('config and ui kinds', () => {
  test('a config frame reads as a config change', () => {
    const strip = makeStrip();
    strip.handleActivity(frame({ kind: 'config', detail: 'orbit.correctors' }, 'patch_setup'));

    expect(mount.textContent).toContain('agent changed config');
    expect(mount.textContent).toContain('orbit.correctors');
  });

  test('a ui frame reads as a window move', () => {
    const strip = makeStrip();
    strip.handleActivity(frame({ kind: 'ui', detail: 'lattice → right' }, 'manage_window'));

    expect(mount.textContent).toContain('agent moved window');
    expect(mount.textContent).toContain('lattice → right');
  });

  test('a detail-less config or ui frame names the tool instead', () => {
    expect(formatActivity(frame({ kind: 'config' }, 'patch_setup')))
      .toEqual({ verb: 'agent changed config', subject: 'patch_setup' });
    expect(formatActivity(frame({ kind: 'ui' }, 'manage_window')))
      .toEqual({ verb: 'agent moved window', subject: 'manage_window' });
  });
});

describe('panel action verbs', () => {
  /** @type {[string, string][]} tool → verb */
  const cases = [
    ['remove_panel_from_rail', 'agent removed'],
    ['add_panel_to_rail', 'agent made available'],
    ['open_panel', 'agent opened'],
    ['close_panel', 'agent closed'],
    ['register_panel', 'agent added'],
  ];

  for (const [tool, verb] of cases) {
    test(`${tool} reads "${verb} <label>"`, () => {
      const strip = makeStrip();
      strip.handleActivity(frame({ kind: 'panel', panel: 'lattice' }, tool));

      expect(mount.textContent).toContain(verb);
      // The catalog label, not the panel id the frame carries.
      expect(mount.textContent).toContain('Lattice');
      expect(mount.textContent).not.toContain('lattice');
    });
  }

  test('an id the catalog does not know falls back to the raw id', () => {
    const strip = makeStrip();
    strip.handleActivity(frame({ kind: 'panel', panel: 'my-custom-panel' }, 'open_panel'));

    expect(mount.textContent).toContain('agent opened');
    expect(mount.textContent).toContain('my-custom-panel');
  });

  test('with no label closure injected, panel ids render raw', () => {
    const strip = createActivityStrip({ mount, getActivePanel: () => activePanel });
    strips.push(strip);
    strip.handleActivity(frame({ kind: 'panel', panel: 'lattice' }, 'close_panel'));

    expect(mount.textContent).toContain('agent closed');
    expect(mount.textContent).toContain('lattice');
  });

  test('a panel frame from some other tool keeps the generic fallback', () => {
    const strip = makeStrip();
    strip.handleActivity(frame({ kind: 'panel', panel: 'ariel' }, 'some_future_tool'));

    expect(mount.textContent).toContain('agent touched');
  });

  test('the suppression table is unchanged: a hide of the active panel stays silent', () => {
    activePanel = 'lattice';
    const strip = makeStrip();
    strip.handleActivity(frame({ kind: 'panel', panel: 'lattice' }, 'close_panel'));
    expect(mount.children.length).toBe(0);

    activePanel = 'ariel';
    strip.handleActivity(frame({ kind: 'panel', panel: 'lattice' }, 'close_panel'));
    expect(mount.textContent).toContain('agent closed');
  });
});

describe('arrange coalescing', () => {
  /** @param {string} panel */
  const arrange = (panel) => frame({ kind: 'panel', panel }, 'arrange_workspace');

  test('a lone arrange names the workspace without a tile count', () => {
    const strip = makeStrip();
    strip.handleActivity(arrange('lattice'));

    expect(mount.textContent).toContain('agent arranged');
    expect(mount.textContent).toContain('workspace');
    expect(mount.textContent).not.toContain('tiles');
  });

  test('a burst of arrange frames collapses into one line naming the tile count', () => {
    const strip = makeStrip();
    strip.handleActivity(arrange('lattice'));
    strip.handleActivity(arrange('ariel'));
    strip.handleActivity(arrange('artifacts'));

    expect(mount.querySelectorAll('.activity-strip-entry').length).toBe(1);
    expect(mount.textContent).toContain('agent arranged');
    expect(mount.textContent).toContain('workspace (3 tiles)');
    // The individual tiles are never named — the burst is one action.
    expect(mount.textContent).not.toContain('Lattice');
  });

  test('any other frame ends the run, and the next arrange starts over at one', () => {
    const strip = makeStrip();
    strip.handleActivity(arrange('lattice'));
    strip.handleActivity(arrange('ariel'));
    strip.handleActivity(frame({ kind: 'channel', detail: 'SR01:HCM1:SP' }));
    strip.handleActivity(arrange('lattice'));

    expect(mount.textContent).toContain('workspace');
    expect(mount.textContent).not.toContain('tiles');
  });

  test('the auto-clear ends the run too — a later arrange is not still counting', () => {
    const strip = makeStrip();
    strip.handleActivity(arrange('lattice'));
    strip.handleActivity(arrange('ariel'));
    vi.advanceTimersByTime(ACTIVITY_CLEAR_MS);

    strip.handleActivity(arrange('lattice'));
    expect(mount.textContent).not.toContain('tiles');
  });
});

describe('pure suppression helpers', () => {
  test('suppressionPanelFor maps each kind onto its self-signaling panel', () => {
    expect(suppressionPanelFor({ kind: 'artifact' })).toBe('artifacts');
    expect(suppressionPanelFor({ kind: 'run' })).toBe('bluesky');
    expect(suppressionPanelFor({ kind: 'panel', panel: 'lattice' })).toBe('lattice');
    expect(suppressionPanelFor({ kind: 'panel' })).toBeNull(); // malformed: no panel id
    expect(suppressionPanelFor({ kind: 'channel' })).toBeNull();
  });

  test('isSuppressed for all four kinds', () => {
    // channel is never suppressed, whatever is active
    expect(isSuppressed({ kind: 'channel', detail: 'SR01:HCM1:SP' }, 'artifacts')).toBe(false);
    expect(isSuppressed({ kind: 'channel', detail: 'SR01:HCM1:SP' }, null)).toBe(false);

    expect(isSuppressed({ kind: 'artifact' }, 'artifacts')).toBe(true);
    expect(isSuppressed({ kind: 'artifact' }, 'bluesky')).toBe(false);

    expect(isSuppressed({ kind: 'run' }, 'bluesky')).toBe(true);
    expect(isSuppressed({ kind: 'run' }, 'artifacts')).toBe(false);

    expect(isSuppressed({ kind: 'panel', panel: 'okf' }, 'okf')).toBe(true);
    expect(isSuppressed({ kind: 'panel', panel: 'okf' }, 'ariel')).toBe(false);

    // no active panel suppresses nothing
    expect(isSuppressed({ kind: 'artifact' }, null)).toBe(false);
  });
});

// ---- History popover ----

/**
 * A history frame stamped `agoSecs` seconds before now, matching the server's
 * float-epoch-seconds `ts`.
 * @param {AgentActivityFrame['target']} target
 * @param {number} agoSecs
 * @param {string} [tool]
 * @returns {AgentActivityFrame}
 */
function pastFrame(target, agoSecs, tool = 'write_channel') {
  return { type: 'agent_activity', tool, target, ts: Date.now() / 1000 - agoSecs };
}

/**
 * A history reader returning `events` verbatim, plus a record of the limits
 * it was called with.
 * @param {AgentActivityFrame[]} events
 */
function stubHistory(events) {
  /** @type {number[]} */
  const calls = [];
  /** @param {number} limit */
  const fetchRecent = (limit) => {
    calls.push(limit);
    return Promise.resolve(events);
  };
  return { fetchRecent, calls };
}

describe('history popover: open, fetch, render', () => {
  test('opening fetches the server ring and renders rows newest first', async () => {
    const { fetchRecent, calls } = stubHistory([
      pastFrame({ kind: 'channel', detail: 'SR01:HCM1:SP' }, 5),
      pastFrame({ kind: 'run', detail: 'orm-42' }, 120, 'run_plan'),
    ]);
    const strip = makeStrip(undefined, fetchRecent);

    expect(popover()).toBeNull();
    await strip.openHistory();

    expect(calls).toEqual([HISTORY_LIMIT]);
    const rows = rowTexts();
    expect(rows.length).toBe(2);
    // Endpoint order is preserved: index 0 is the most recent action.
    expect(rows[0]).toContain('SR01:HCM1:SP');
    expect(rows[0]).toContain('5s ago');
    expect(rows[1]).toContain('orm-42');
    expect(rows[1]).toContain('2m ago');
  });

  test('history rows are worded by the same verbs and labels as the live line', async () => {
    const strip = makeStrip(undefined, stubHistory([
      pastFrame({ kind: 'panel', panel: 'lattice' }, 3, 'close_panel'),
      pastFrame({ kind: 'panel', panel: 'ariel' }, 9, 'arrange_workspace'),
    ]).fetchRecent);
    await strip.openHistory();

    const rows = rowTexts();
    expect(rows[0]).toContain('agent closed');
    expect(rows[0]).toContain('Lattice');
    // A history row stands alone: no live burst to count, so no tile count.
    expect(rows[1]).toContain('agent arranged');
    expect(rows[1]).toContain('workspace');
    expect(rows[1]).not.toContain('tiles');
  });

  test('an empty ring renders a message, not a bare empty list', async () => {
    const strip = makeStrip(undefined, stubHistory([]).fetchRecent);
    await strip.openHistory();

    expect(rowTexts().length).toBe(0);
    expect(popover()?.textContent).toContain('No recent agent activity');
  });

  test('a failing fetch says so instead of showing an empty history', async () => {
    const strip = makeStrip(undefined, () => Promise.reject(new Error('offline')));
    await strip.openHistory();

    expect(popover()?.textContent).toContain('Could not load recent activity');
  });

  test('reopening refetches — the server ring is the only source of history', async () => {
    const { fetchRecent, calls } = stubHistory([
      pastFrame({ kind: 'channel', detail: 'SR01:HCM1:SP' }, 1),
    ]);
    const strip = makeStrip(undefined, fetchRecent);

    await strip.openHistory();
    strip.closeHistory();
    await strip.openHistory();

    expect(calls).toEqual([HISTORY_LIMIT, HISTORY_LIMIT]);
    expect(rowTexts().length).toBe(1);
  });

  test('a fetch that resolves after close does not resurrect the popover', async () => {
    /** @type {(events: AgentActivityFrame[]) => void} */
    let resolve = () => {};
    const strip = makeStrip(undefined, () => new Promise((r) => { resolve = r; }));

    const opening = strip.openHistory();
    strip.closeHistory();
    resolve([pastFrame({ kind: 'channel', detail: 'SR01:HCM1:SP' }, 1)]);
    await opening;

    expect(popover()).toBeNull();
    expect(strip.isHistoryOpen()).toBe(false);
  });

  test('rows never exceed HISTORY_LIMIT, however much the server returns', async () => {
    const many = Array.from({ length: HISTORY_LIMIT + 10 }, (_, i) =>
      pastFrame({ kind: 'channel', detail: `SR01:HCM${i}:SP` }, i));
    const strip = makeStrip(undefined, stubHistory(many).fetchRecent);
    await strip.openHistory();

    expect(rowTexts().length).toBe(HISTORY_LIMIT);
  });
});

describe('history popover: agent strings are text nodes only', () => {
  test('markup in a history row lands as literal text, no element is created', async () => {
    const payload = '<img src=x onerror=alert(1)>';
    const strip = makeStrip(undefined, stubHistory([
      pastFrame({ kind: 'channel', detail: payload }, 2),
    ]).fetchRecent);
    await strip.openHistory();

    expect(popover()?.querySelector('img')).toBeNull();
    expect(popover()?.textContent).toContain(payload);
  });

  test('markup in a live frame prepended while open stays literal too', async () => {
    const payload = '<script>alert(1)</script>';
    const strip = makeStrip(undefined, stubHistory([]).fetchRecent);
    await strip.openHistory();
    strip.handleActivity(pastFrame({ kind: 'channel', detail: payload }, 0));

    expect(popover()?.querySelector('script')).toBeNull();
    expect(popover()?.textContent).toContain(payload);
  });
});

describe('history popover: live frames while open', () => {
  test('a frame arriving while open is prepended above the fetched rows', async () => {
    const strip = makeStrip(undefined, stubHistory([
      pastFrame({ kind: 'run', detail: 'orm-42' }, 60, 'run_plan'),
    ]).fetchRecent);
    await strip.openHistory();
    expect(rowTexts().length).toBe(1);

    strip.handleActivity(pastFrame({ kind: 'channel', detail: 'SR01:HCM1:SP' }, 0));

    const rows = rowTexts();
    expect(rows.length).toBe(2);
    expect(rows[0]).toContain('SR01:HCM1:SP');
    expect(rows[1]).toContain('orm-42');
  });

  test('the first live frame replaces the empty-history message', async () => {
    const strip = makeStrip(undefined, stubHistory([]).fetchRecent);
    await strip.openHistory();
    expect(popover()?.textContent).toContain('No recent agent activity');

    strip.handleActivity(pastFrame({ kind: 'channel', detail: 'SR01:HCM1:SP' }, 0));

    expect(popover()?.textContent).not.toContain('No recent agent activity');
    expect(rowTexts().length).toBe(1);
  });

  test('a SUPPRESSED frame is still recorded in the history, as the server ring records it', async () => {
    activePanel = 'artifacts';
    const strip = makeStrip(undefined, stubHistory([]).fetchRecent);
    await strip.openHistory();

    strip.handleActivity(pastFrame({ kind: 'artifact', detail: 'orbit-plot.png' }, 0, 'focus_artifact'));

    // Suppression governs the live line only — the strip stays silent...
    expect(mount.querySelectorAll('.activity-strip-entry').length).toBe(0);
    // ...while history still shows what the agent did.
    expect(rowTexts()[0]).toContain('orbit-plot.png');
  });

  test('a malformed frame is ignored by the history as well as the live line', async () => {
    const strip = makeStrip(undefined, stubHistory([]).fetchRecent);
    await strip.openHistory();

    strip.handleActivity(/** @type {any} */ ({ type: 'agent_activity', tool: 'x' }));

    expect(rowTexts().length).toBe(0);
  });

  test('frames arriving while CLOSED are not tracked client-side', async () => {
    const { fetchRecent } = stubHistory([]);
    const strip = makeStrip(undefined, fetchRecent);

    strip.handleActivity(frame({ kind: 'channel', detail: 'SR01:HCM1:SP' }));
    await strip.openHistory();

    // No client-side ring: what the popover shows is exactly what the server
    // returned, which the stub says is nothing.
    expect(rowTexts().length).toBe(0);
  });
});

describe('history popover: live single-slot behavior is unchanged', () => {
  test('with the popover open, the strip still shows one entry and auto-clears', async () => {
    const strip = makeStrip(undefined, stubHistory([]).fetchRecent);
    await strip.openHistory();

    strip.handleActivity(frame({ kind: 'channel', detail: 'SR01:HCM1:SP' }));
    strip.handleActivity(frame({ kind: 'run', detail: 'orm-42' }, 'run_plan'));

    expect(mount.querySelectorAll('.activity-strip-entry').length).toBe(1);
    expect(mount.textContent).toContain('orm-42');
    expect(mount.textContent).not.toContain('SR01:HCM1:SP');

    vi.advanceTimersByTime(ACTIVITY_CLEAR_MS);
    expect(mount.textContent).toBe('');

    // The live line clearing must not touch the history the popover shows.
    expect(rowTexts().length).toBe(2);
  });
});

describe('history popover: trigger and close affordances', () => {
  test('clicking the strip toggles the popover and aria-expanded', async () => {
    const strip = makeStrip(undefined, stubHistory([]).fetchRecent);
    expect(mount.getAttribute('aria-expanded')).toBe('false');

    mount.click();
    await flush();
    expect(strip.isHistoryOpen()).toBe(true);
    expect(popover()).not.toBeNull();
    expect(mount.getAttribute('aria-expanded')).toBe('true');

    mount.click();
    await flush();
    expect(strip.isHistoryOpen()).toBe(false);
    expect(popover()).toBeNull();
    expect(mount.getAttribute('aria-expanded')).toBe('false');
  });

  test('Enter on the focused strip opens it (keyboard parity with the click)', async () => {
    const strip = makeStrip(undefined, stubHistory([]).fetchRecent);

    mount.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
    await flush();

    expect(strip.isHistoryOpen()).toBe(true);
  });

  test('Escape closes and returns focus to the strip', async () => {
    const strip = makeStrip(undefined, stubHistory([]).fetchRecent);
    await strip.openHistory();

    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));

    expect(strip.isHistoryOpen()).toBe(false);
    expect(popover()).toBeNull();
    expect(mount.getAttribute('aria-expanded')).toBe('false');
    expect(document.activeElement).toBe(mount);
  });

  test('a click outside closes; a click inside the popover does not', async () => {
    const strip = makeStrip(undefined, stubHistory([
      pastFrame({ kind: 'channel', detail: 'SR01:HCM1:SP' }, 1),
    ]).fetchRecent);
    await strip.openHistory();

    // Inside the popover: stays open (a row is selectable text, not a dismiss).
    const row = /** @type {HTMLElement} */ (document.querySelector('.activity-history-row'));
    row.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    expect(strip.isHistoryOpen()).toBe(true);

    document.body.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    expect(strip.isHistoryOpen()).toBe(false);
    expect(popover()).toBeNull();
  });

  test('closing twice, or before ever opening, is a no-op', () => {
    const strip = makeStrip(undefined, stubHistory([]).fetchRecent);
    strip.closeHistory();
    strip.closeHistory();
    expect(strip.isHistoryOpen()).toBe(false);
    expect(mount.getAttribute('aria-expanded')).toBe('false');
  });
});

describe('formatRelativeTime', () => {
  const now = 1_765_432_100_000; // ms

  test('coarsens seconds through days', () => {
    expect(formatRelativeTime(now / 1000 - 5, now)).toBe('5s ago');
    expect(formatRelativeTime(now / 1000 - 59, now)).toBe('59s ago');
    expect(formatRelativeTime(now / 1000 - 60, now)).toBe('1m ago');
    expect(formatRelativeTime(now / 1000 - 3599, now)).toBe('59m ago');
    expect(formatRelativeTime(now / 1000 - 3600, now)).toBe('1h ago');
    expect(formatRelativeTime(now / 1000 - 86_400, now)).toBe('1d ago');
  });

  test('a just-now or clock-skewed future stamp never reads as negative', () => {
    expect(formatRelativeTime(now / 1000, now)).toBe('just now');
    expect(formatRelativeTime(now / 1000 + 30, now)).toBe('just now');
  });

  test('a missing or non-numeric ts yields no label at all', () => {
    expect(formatRelativeTime(undefined, now)).toBe('');
    expect(formatRelativeTime(/** @type {any} */ ('nope'), now)).toBe('');
    expect(formatRelativeTime(NaN, now)).toBe('');
  });
});

describe('empty-strip focusability', () => {
  // An empty strip is hidden with `opacity: 0` (activity-strip.css keeps it
  // non-display so the aria-live region pre-exists its content), leaving it
  // invisible and zero-sized but still in the DOM. It must therefore not be a
  // tab stop — the keyboard counterpart of the `pointer-events: none` on
  // `.status-activity-floating:empty`.

  test('a strip with nothing to show is not a tab stop', () => {
    makeStrip();
    expect(mount.hasAttribute('tabindex')).toBe(false);
  });

  test('showing a frame makes the trigger focusable', async () => {
    const strip = makeStrip();
    strip.handleActivity(frame({ kind: 'channel', detail: 'SR01:HCM1:SP' }));
    await flush();
    expect(mount.getAttribute('tabindex')).toBe('0');
  });

  test('the auto-clear timeout drops the tab stop again', async () => {
    const strip = makeStrip(1000);
    strip.handleActivity(frame({ kind: 'channel', detail: 'SR01:HCM1:SP' }));
    await flush();
    expect(mount.getAttribute('tabindex')).toBe('0');

    await vi.advanceTimersByTimeAsync(1000);
    expect(mount.textContent).toBe('');
    expect(mount.hasAttribute('tabindex')).toBe(false);
  });

  test('an explicit clear() drops the tab stop', async () => {
    const strip = makeStrip();
    strip.handleActivity(frame({ kind: 'channel', detail: 'SR01:HCM1:SP' }));
    await flush();
    strip.clear();
    await flush();
    expect(mount.hasAttribute('tabindex')).toBe(false);
  });

  test('the trigger stays focusable while the popover is open over an emptied strip', async () => {
    const strip = makeStrip(1000, async () => []);
    strip.handleActivity(frame({ kind: 'channel', detail: 'SR01:HCM1:SP' }));
    await flush();
    await strip.openHistory();

    // The live slot clears on its own timer while the popover is still up;
    // yanking the tab stop out from under an open popover would strand focus.
    await vi.advanceTimersByTimeAsync(1000);
    expect(mount.textContent).toBe('');
    expect(mount.getAttribute('tabindex')).toBe('0');

    strip.closeHistory();
    await flush();
    expect(mount.hasAttribute('tabindex')).toBe(false);
  });
});
