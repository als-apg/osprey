/**
 * Unit tests for a sidebar row's live state (rail-row-state.js).
 *
 *   npx vitest run tests/interfaces/web_terminal/rail-row-state.test.mjs
 *
 * The rows are drawn by panel-rail.js createRail; the state each row shows
 * while the page runs — active, enabled, the agent's attention badge and its
 * time, reachability, a sidecar's start status, the popped-out mark — is set
 * through the mutators here, which panel-rail.js re-exports to the shell.
 *
 * The rows carry NO per-panel health readout: backend liveness is reported by
 * the SYSTEM panel's `web_panels` health category. `.disabled` and
 * `.unreachable` are the only health-derived states a row renders.
 *
 * The environment is happy-dom (vitest.config.js), so `document` is a global;
 * the design-system modules come through the /design-system/js/* alias.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import { byId } from '../_support/dom.mjs';
import { createRail } from '../../../src/osprey/interfaces/web_terminal/static/js/panel-rail.js';
import {
  getEntry,
  setActive,
  setEntryAttention,
  setEntryEnabled,
  setEntryPoppedOut,
  setEntryReachable,
  setEntryStatus,
} from '../../../src/osprey/interfaces/web_terminal/static/js/rail-row-state.js';
import { icon } from '/design-system/js/icons.js';
import { hideTooltip, initTooltips } from '/design-system/js/tooltip.js';
import { FACILITY_ZONE, VIEWER_ZONE, stampFacilityZone } from '../_support/facility-zone.mjs';

const PANELS = [
  { id: 'artifacts', label: 'Artifacts' },
  { id: 'ariel', label: 'Ariel' },
  { id: 'channel-finder', label: 'Channels' },
];

/** A bare rail element: no skeleton, so createRail builds the groups.
 *  @returns {HTMLElement} */
function freshRail() {
  document.body.innerHTML = '';
  const el = document.createElement('div');
  document.body.appendChild(el);
  return el;
}

/** A rail holding the page-owned Artifacts row, as index.html ships it.
 *  @returns {HTMLElement} */
function pageRail() {
  document.body.innerHTML = `
    <div id="panel-rail">
      <div class="sidebar-row" id="sidebar-artifacts-row">
        <button class="osprey-nav-row parent panel-rail-button disabled" data-panel-id="artifacts"
                type="button"><span class="osprey-nav-label">Artifacts</span></button>
      </div>
    </div>`;
  return byId('panel-rail');
}

/** @param {HTMLElement} rail @param {string} id @returns {HTMLButtonElement} */
function row(rail, id) {
  return /** @type {HTMLButtonElement} */ (getEntry(rail, id));
}

// The groups live in this browser's storage; every test starts from the
// default Workspace and Tools.
beforeEach(() => {
  localStorage.clear();
});

describe('setActive', () => {
  /** @type {HTMLElement} */
  let rail;
  beforeEach(() => {
    rail = freshRail();
    createRail(rail, PANELS);
  });

  test('marks exactly one row active and current', () => {
    setActive(rail, 'ariel');
    expect(row(rail, 'ariel').classList.contains('active')).toBe(true);
    expect(row(rail, 'ariel').getAttribute('aria-current')).toBe('page');
    expect(row(rail, 'artifacts').classList.contains('active')).toBe(false);
    expect(row(rail, 'artifacts').hasAttribute('aria-current')).toBe(false);
  });

  test('switching active clears the previous one', () => {
    setActive(rail, 'artifacts');
    setActive(rail, 'channel-finder');
    expect(row(rail, 'artifacts').classList.contains('active')).toBe(false);
    expect(row(rail, 'artifacts').hasAttribute('aria-current')).toBe(false);
    expect(row(rail, 'channel-finder').classList.contains('active')).toBe(true);
  });

  test('a null / unknown id clears active on every row', () => {
    setActive(rail, 'artifacts');
    setActive(rail, null);
    expect(rail.querySelectorAll('.panel-rail-button.active').length).toBe(0);
    expect(rail.querySelectorAll('[aria-current]').length).toBe(0);
  });
});

describe('setEntryEnabled', () => {
  /** @type {HTMLElement} */
  let rail;
  beforeEach(() => {
    rail = freshRail();
    createRail(rail, PANELS);
  });

  test('setEntryEnabled toggles the disabled class', () => {
    setEntryEnabled(rail, 'artifacts', true);
    expect(row(rail, 'artifacts').classList.contains('disabled')).toBe(false);
    setEntryEnabled(rail, 'artifacts', false);
    expect(row(rail, 'artifacts').classList.contains('disabled')).toBe(true);
  });

  test('is a no-op for an unknown id', () => {
    expect(() => setEntryEnabled(rail, 'nope', true)).not.toThrow();
  });
});

describe('setEntryAttention', () => {
  /** @type {HTMLElement} */
  let rail;
  beforeEach(() => {
    rail = freshRail();
    createRail(rail, PANELS);
  });

  test('on sets the persistent attention class and fires the transient flash', () => {
    expect(setEntryAttention(rail, 'ariel', true)).toBe(true);
    const entry = row(rail, 'ariel');
    expect(entry.classList.contains('agent-attention')).toBe(true);
    expect(entry.classList.contains('agent-flash')).toBe(true);
  });

  test('off clears the attention without force-stopping an in-flight flash', () => {
    setEntryAttention(rail, 'ariel', true);
    expect(setEntryAttention(rail, 'ariel', false)).toBe(true);
    const entry = row(rail, 'ariel');
    expect(entry.classList.contains('agent-attention')).toBe(false);
    // No animationend has fired in happy-dom, so the flash class must survive.
    expect(entry.classList.contains('agent-flash')).toBe(true);
  });

  test('returns false, scrolls nothing and does not throw for an unknown id', () => {
    const scrollSpy = vi.spyOn(Element.prototype, 'scrollIntoView').mockImplementation(() => {});
    expect(() => setEntryAttention(rail, 'nope', true)).not.toThrow();
    expect(setEntryAttention(rail, 'nope', true)).toBe(false);
    expect(scrollSpy).not.toHaveBeenCalled();
    scrollSpy.mockRestore();
  });

  test('on draws one hidden-from-AT dot after the label; off removes it', () => {
    const entry = row(rail, 'ariel');
    const childrenBefore = [...entry.children];

    setEntryAttention(rail, 'ariel', true);
    setEntryAttention(rail, 'ariel', true);
    const dots = entry.querySelectorAll('[data-attention-badge]');
    expect(dots.length).toBe(1);
    const dot = /** @type {HTMLElement} */ (dots[0]);
    expect(dot.classList.contains('osprey-nav-badge')).toBe(true);
    expect(dot.classList.contains('sidebar-attention')).toBe(true);
    expect(dot.getAttribute('aria-hidden')).toBe('true');
    expect(dot.previousElementSibling?.classList.contains('osprey-nav-label')).toBe(true);

    setEntryAttention(rail, 'ariel', false);
    expect([...entry.children]).toEqual(childrenBefore);
  });

  test('without a time, a badge and its clear leave the row attributes as they were', () => {
    const entry = row(rail, 'ariel');
    const attrsBefore = entry.getAttributeNames().sort();

    setEntryAttention(rail, 'ariel', true);
    setEntryAttention(rail, 'ariel', false);

    expect(entry.getAttributeNames().sort()).toEqual(attrsBefore);
  });

  test('a badged-then-cleared row leaves no tooltip stash behind', () => {
    const entry = row(rail, 'ariel');
    setEntryAttention(rail, 'ariel', true, 1_755_000_000);
    expect(entry.hasAttribute('data-title-base')).toBe(true);

    setEntryAttention(rail, 'ariel', false);
    expect(entry.hasAttribute('data-title-base')).toBe(false);
  });
});

/**
 * A sidebar taller than its viewport can put a row off-screen, where a badge
 * reports nothing at all. Setting the badge must surface the row; clearing it
 * must not move the sidebar under the operator.
 *
 * happy-dom implements scrollIntoView as a no-op on Element.prototype, so the
 * spy below observes real calls rather than installing a missing method.
 */
describe('setEntryAttention scrolls a badged row into view', () => {
  /** @type {HTMLElement} */
  let rail;
  /** @type {import('vitest').MockInstance} */
  let scrollSpy;

  beforeEach(() => {
    rail = freshRail();
    createRail(rail, PANELS);
    scrollSpy = vi.spyOn(Element.prototype, 'scrollIntoView').mockImplementation(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  test('setting the badge surfaces the row without disturbing a visible sidebar', () => {
    setEntryAttention(rail, 'ariel', true);

    expect(scrollSpy).toHaveBeenCalledTimes(1);
    // `nearest` is the whole point: a row already in view must not scroll.
    expect(scrollSpy).toHaveBeenCalledWith({ block: 'nearest' });
    expect(scrollSpy.mock.instances[0]).toBe(getEntry(rail, 'ariel'));
  });

  test('clearing the badge does not scroll', () => {
    setEntryAttention(rail, 'ariel', true);
    scrollSpy.mockClear();

    setEntryAttention(rail, 'ariel', false);
    expect(scrollSpy).not.toHaveBeenCalled();
  });
});

/**
 * The badge says a panel was touched; the tooltip says WHEN. The time comes
 * from the event's own server `ts` (epoch seconds, as the agent_activity SSE
 * frames carry it) — never from the client clock, which would report when the
 * browser rendered rather than when the agent acted.
 */
describe('setEntryAttention tooltip time', () => {
  /** @type {HTMLElement} */
  let rail;

  // Two fixed server timestamps an hour apart. Rendering is asserted against
  // an Intl computation in the stamped facility zone rather than a literal so
  // the suite is not hostage to the runner's timezone or locale; the FORMAT is
  // pinned separately. FACILITY_ZONE reads a different clock from the runner,
  // so the time names its zone.
  const TS = 1_755_000_000;
  const TS_LATER = TS + 3600;

  /** @param {number} ts @returns {string} */
  const expectedTime = (ts) =>
    new Intl.DateTimeFormat(undefined, {
      hour: 'numeric',
      minute: '2-digit',
      timeZone: FACILITY_ZONE,
      timeZoneName: 'short',
    }).format(ts * 1000);

  beforeEach(() => {
    stampFacilityZone(FACILITY_ZONE);
    rail = freshRail();
    createRail(rail, PANELS);
  });

  afterEach(() => stampFacilityZone(null));

  test('a viewer already on the facility clock reads the time without a zone name', () => {
    stampFacilityZone(VIEWER_ZONE);
    setEntryAttention(rail, 'ariel', true, TS);
    const plain = new Intl.DateTimeFormat(undefined, {
      hour: 'numeric',
      minute: '2-digit',
      timeZone: VIEWER_ZONE,
    }).format(TS * 1000);
    expect(row(rail, 'ariel').title).toBe(`Ariel · agent touched ${plain}`);
  });

  test('a badge with a server ts appends the touch time to the tooltip', () => {
    setEntryAttention(rail, 'ariel', true, TS);
    expect(row(rail, 'ariel').title).toBe(`Ariel · agent touched ${expectedTime(TS)}`);
  });

  test('the rendered time is hour and minute only — no seconds', () => {
    setEntryAttention(rail, 'ariel', true, TS);
    const suffix = String(row(rail, 'ariel').title).split('· agent touched ')[1];
    expect(suffix).toMatch(/\d{1,2}:\d{2}/);
    expect((suffix.match(/:/g) ?? []).length).toBe(1);
  });

  test('a second event replaces the time rather than appending a second suffix', () => {
    setEntryAttention(rail, 'ariel', true, TS);
    setEntryAttention(rail, 'ariel', true, TS_LATER);

    const title = String(row(rail, 'ariel').title);
    expect(title).toBe(`Ariel · agent touched ${expectedTime(TS_LATER)}`);
    expect((title.match(/agent touched/g) ?? []).length).toBe(1);
  });

  test('clearing the badge restores the base tooltip exactly', () => {
    setEntryAttention(rail, 'ariel', true, TS);
    setEntryAttention(rail, 'ariel', false);

    expect(row(rail, 'ariel').title).toBe('Ariel');
  });

  test('a later badge with no ts drops the stale time instead of keeping it', () => {
    setEntryAttention(rail, 'ariel', true, TS);
    setEntryAttention(rail, 'ariel', true);

    expect(row(rail, 'ariel').title).toBe('Ariel');
  });

  test('no ts leaves the tooltip untouched (the pre-existing 3-arg contract)', () => {
    setEntryAttention(rail, 'ariel', true);
    expect(row(rail, 'ariel').title).toBe('Ariel');
    expect(row(rail, 'ariel').classList.contains('agent-attention')).toBe(true);
  });

  test('a non-finite ts is refused rather than rendered as "Invalid Date"', () => {
    for (const bad of [NaN, Infinity]) {
      setEntryAttention(rail, 'ariel', true, bad);
      expect(row(rail, 'ariel').title).toBe('Ariel');
    }
  });

  test('the badge borrows and returns a hinted tooltip intact', () => {
    // The stash is what the badge restores; a row advertising its menu must
    // get that advertisement back, not just the bare label.
    rail = freshRail();
    createRail(rail, [{ id: 'ariel', label: 'Ariel', hint: 'right-click for actions' }]);

    setEntryAttention(rail, 'ariel', true, TS);
    expect(row(rail, 'ariel').title).toBe(
      `Ariel · right-click for actions · agent touched ${expectedTime(TS)}`
    );

    setEntryAttention(rail, 'ariel', false);
    expect(row(rail, 'ariel').title).toBe('Ariel · right-click for actions');
  });

  test('a badge composed while the collapsed tip shows keeps the base title', () => {
    // The tooltip layer parks the native title while it shows the row's tip;
    // reading the empty `title` then would lose "Ariel" from the composition.
    // The tip shows after the layer's delay, which fake timers step through.
    vi.useFakeTimers();
    const stop = initTooltips(rail);
    try {
      const entry = row(rail, 'ariel');
      entry.dispatchEvent(new Event('pointerover', { bubbles: true }));
      vi.advanceTimersByTime(450);
      expect(entry.hasAttribute('title')).toBe(false);

      setEntryAttention(rail, 'ariel', true, TS);
      expect(entry.hasAttribute('title')).toBe(false);

      hideTooltip();
      expect(entry.title).toBe(`Ariel · agent touched ${expectedTime(TS)}`);
      setEntryAttention(rail, 'ariel', false);
      expect(entry.title).toBe('Ariel');
    } finally {
      stop();
      vi.useRealTimers();
    }
  });

  test('the suffix never reaches the accessible name', () => {
    // aria-label is the row's identity; a churning timestamp inside it would
    // make the sidebar re-announce panels to a screen reader on every event.
    setEntryAttention(rail, 'ariel', true, TS);
    expect(row(rail, 'ariel').getAttribute('aria-label')).toBe('Ariel');
  });
});

/**
 * Coarse reachability: a row whose backend answered before but has stopped
 * is dimmed without going inert, and its tooltip says since when. The poll
 * logic that decides WHEN is panel-lifecycle's (pinned in panel-manager.test.mjs).
 */
describe('setEntryReachable', () => {
  /** @type {HTMLElement} */
  let rail;
  const SINCE = new Date(2026, 0, 5, 9, 12).getTime();
  // On the facility clock, which reads differently from the runner's, so the
  // time names its zone.
  const since = new Intl.DateTimeFormat(undefined, {
    hour: 'numeric',
    minute: '2-digit',
    timeZone: FACILITY_ZONE,
    timeZoneName: 'short',
  }).format(SINCE);

  beforeEach(() => {
    stampFacilityZone(FACILITY_ZONE);
    rail = freshRail();
    createRail(rail, [{ id: 'ariel', label: 'Ariel', hint: 'right-click for actions' }]);
  });

  afterEach(() => stampFacilityZone(null));

  test('unreachable dims without disabling, and the tooltip says since when', () => {
    expect(setEntryReachable(rail, 'ariel', false, SINCE)).toBe(true);
    const entry = row(rail, 'ariel');
    expect(entry.classList.contains('unreachable')).toBe(true);
    expect(entry.classList.contains('disabled')).toBe(true); // the cold boot state, untouched
    expect(entry.title).toBe(`not answering since ${since}`);

    setEntryReachable(rail, 'ariel', true);
    expect(entry.classList.contains('unreachable')).toBe(false);
    expect(entry.title).toBe('Ariel · right-click for actions');
    expect(entry.hasAttribute('data-title-base')).toBe(false);
  });

  test('the notice outranks a badge time, and each clears without losing the other', () => {
    const entry = row(rail, 'ariel');
    setEntryAttention(rail, 'ariel', true, 1_755_000_000);
    const badged = entry.title;

    setEntryReachable(rail, 'ariel', false, SINCE);
    expect(entry.title).toBe(`not answering since ${since}`);

    setEntryReachable(rail, 'ariel', true);
    expect(entry.title).toBe(badged);

    setEntryAttention(rail, 'ariel', false);
    expect(entry.title).toBe('Ariel · right-click for actions');
  });

  test('returns false for an unknown id', () => {
    expect(setEntryReachable(rail, 'nope', false, SINCE)).toBe(false);
  });
});

/**
 * A sidecar that failed to start is the one status the sidebar names: its row
 * is dimmed but clickable (the click is the retry), and its tooltip is the
 * server's sentence, verbatim.
 */
describe('setEntryStatus', () => {
  /** @type {HTMLElement} */
  let rail;
  const MESSAGE = 'JUPYTER failed to start: boom';
  const JUPYTER = { id: 'jupyter', label: 'Jupyter', hint: 'right-click for actions' };

  beforeEach(() => {
    rail = freshRail();
  });

  test("a failed row is clickable and carries the server's message as its tooltip", () => {
    const onActivate = vi.fn();
    createRail(rail, [JUPYTER], { onActivate });

    setEntryStatus(rail, 'jupyter', { failed: true, message: MESSAGE });

    const entry = row(rail, 'jupyter');
    expect(entry.classList.contains('failed')).toBe(true);
    expect(entry.classList.contains('disabled')).toBe(false);
    expect(entry.title).toBe(MESSAGE);
    expect(entry.getAttribute('aria-description')).toBe(MESSAGE);
    entry.click();
    expect(onActivate).toHaveBeenCalledWith('jupyter');
  });

  test('clearing the status restores the base tooltip', () => {
    createRail(rail, [JUPYTER]);
    setEntryStatus(rail, 'jupyter', { failed: true, message: MESSAGE });

    setEntryStatus(rail, 'jupyter', { failed: false, message: null });

    const entry = row(rail, 'jupyter');
    expect(entry.classList.contains('failed')).toBe(false);
    expect(entry.title).toBe('Jupyter · right-click for actions');
    expect(entry.hasAttribute('aria-description')).toBe(false);
    expect(entry.hasAttribute('data-title-base')).toBe(false);
  });

  test('a message set under an attention badge survives the badge clearing', () => {
    createRail(rail, [JUPYTER]);
    setEntryAttention(rail, 'jupyter', true, 1_755_000_000);

    setEntryStatus(rail, 'jupyter', { failed: true, message: MESSAGE });
    const entry = row(rail, 'jupyter');
    expect(entry.title.startsWith(`${MESSAGE} · agent touched `)).toBe(true);

    setEntryAttention(rail, 'jupyter', false);
    expect(entry.title).toBe(MESSAGE);

    setEntryStatus(rail, 'jupyter', { failed: false, message: null });
    expect(entry.title).toBe('Jupyter · right-click for actions');
  });
});

/**
 * A panel popped out into its own window keeps its row: the row says so with
 * a trailing window glyph, and its tooltip detail says what a click now does.
 * The panel's own sentence comes back when the window closes.
 */
describe('setEntryPoppedOut', () => {
  const POPPED = 'Popped out — click to bring the window to the front';
  /** @type {HTMLElement} */
  let rail;
  beforeEach(() => {
    rail = freshRail();
    createRail(rail, [
      { id: 'ariel', label: 'Ariel' },
      { id: 'my-dashboard', label: 'My dashboard' },
    ]);
  });

  test('marks the row, adds the glyph last and parks its detail', () => {
    expect(setEntryPoppedOut(rail, 'ariel', true)).toBe(true);
    const entry = row(rail, 'ariel');
    expect(entry.classList.contains('popped-out')).toBe(true);
    expect(entry.dataset.tipDetail).toBe(POPPED);
    expect(entry.getAttribute('data-tip-detail-base')).toBe('Search the electronic logbook');
    const trail = /** @type {HTMLElement} */ (entry.lastElementChild);
    expect(trail.matches('span.osprey-nav-trail[data-popout-trail]')).toBe(true);
    expect(trail.getAttribute('aria-hidden')).toBe('true');
    expect(trail.innerHTML).toBe(icon('popout').outerHTML);
  });

  test('clearing the mark puts the row back as it was', () => {
    const entry = row(rail, 'ariel');
    const before = entry.outerHTML;
    setEntryPoppedOut(rail, 'ariel', true);
    expect(setEntryPoppedOut(rail, 'ariel', false)).toBe(true);
    expect(entry.outerHTML).toBe(before);
  });

  test('a row with no detail of its own has none again when the mark clears', () => {
    setEntryPoppedOut(rail, 'my-dashboard', true);
    const entry = row(rail, 'my-dashboard');
    expect(entry.dataset.tipDetail).toBe(POPPED);
    setEntryPoppedOut(rail, 'my-dashboard', false);
    expect(entry.hasAttribute('data-tip-detail')).toBe(false);
    expect(entry.hasAttribute('data-tip-detail-base')).toBe(false);
  });

  test('marking twice draws one glyph, and keeps the parked detail', () => {
    setEntryPoppedOut(rail, 'ariel', true);
    setEntryPoppedOut(rail, 'ariel', true);
    const entry = row(rail, 'ariel');
    expect(entry.querySelectorAll('[data-popout-trail]').length).toBe(1);
    expect(entry.getAttribute('data-tip-detail-base')).toBe('Search the electronic logbook');
  });

  test('a render of a popped-out page-owned row writes its detail to the parking place', () => {
    const page = pageRail();
    createRail(page, [{ id: 'artifacts', label: 'Artifacts' }]);
    setEntryPoppedOut(page, 'artifacts', true);

    createRail(page, [{ id: 'artifacts', label: 'Artifacts' }]);

    const entry = row(page, 'artifacts');
    expect(entry.classList.contains('popped-out')).toBe(true);
    expect(entry.dataset.tipDetail).toBe(POPPED);
    expect(entry.getAttribute('data-tip-detail-base')).toBe(
      'Files, plots and notebooks the agent produced'
    );
  });

  test('the attention dot sits before the glyph, which stays last', () => {
    setEntryPoppedOut(rail, 'ariel', true);
    setEntryAttention(rail, 'ariel', true);
    const trail = /** @type {HTMLElement} */ (row(rail, 'ariel').lastElementChild);
    expect(trail.hasAttribute('data-popout-trail')).toBe(true);
    expect(trail.previousElementSibling?.hasAttribute('data-attention-badge')).toBe(true);
  });

  test('returns false for an unknown id', () => {
    expect(setEntryPoppedOut(rail, 'nope', true)).toBe(false);
  });
});
