/**
 * Unit tests for the onboarding tour (tour.js).
 *
 *   npx vitest run tests/interfaces/web_terminal/tour.test.mjs
 *
 * The tour is an invite card followed by spotlighted steps over the live
 * shell. applyTourConfig() records the server-derived facts (invite policy +
 * capability list from GET /api/panels) and arms the automatic invite;
 * startTour() is the on-demand entry (rail control, palette) that ignores
 * the policy. Steps whose anchor is absent drop out. The dismissal flag is
 * storage-scoped (per-persona scoping is covered in
 * js/storage-scope-keys.test.js).
 *
 * terminal.js (the xterm stack) is imported lazily by tour.js and mocked
 * here — the prompt chips must INSERT text, never send it.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

const TERMINAL_PATH = '../../../src/osprey/interfaces/web_terminal/static/js/terminal.js';

const pasteToTerminal = vi.fn();
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/terminal.js', () => ({
  pasteToTerminal,
}));

import {
  applyTourConfig,
  startTour,
  INVITE_DELAY_MS,
} from '../../../src/osprey/interfaces/web_terminal/static/js/tour.js';

/** Mount the full set of tour anchors. */
function mountFullShell() {
  document.body.innerHTML = `
    <button class="control-target-chip" aria-expanded="false">
      <span class="ctc-short">Simulator</span><span class="ctc-state">writes</span>
    </button>
    <osprey-display-menu id="display-menu">
      <button class="display-menu-trigger" aria-expanded="false"></button>
    </osprey-display-menu>
    <button id="command-palette-btn"></button>
    <nav class="panel-rail">
      <button class="panel-rail-button" data-panel-id="artifacts"></button>
      <button class="panel-rail-button" data-panel-id="okf"></button>
      <button class="panel-rail-button" data-panel-id="ariel"></button>
    </nav>
    <button id="panel-feedback-btn"></button>
    <div class="terminal-card"></div>`;
}

/**
 * Arm the invite under `policy` and let the delay elapse.
 * @param {string} [policy]
 * @param {string[]} [capabilities]
 */
function invite(policy = 'once', capabilities = []) {
  applyTourConfig({ tour: { policy, capabilities } });
  vi.advanceTimersByTime(INVITE_DELAY_MS + 1);
}

function closeTour() {
  document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
}

/**
 * Query an element the fixture guarantees, failing loudly when it is missing
 * rather than letting a drifted fixture surface as a null dereference.
 * @param {string} selector
 * @returns {HTMLElement}
 */
function find(selector) {
  const el = document.querySelector(selector);
  if (!(el instanceof HTMLElement)) throw new Error(`expected ${selector}`);
  return el;
}

/** @param {string} selector */
function click(selector) {
  find(selector).click();
}

const cardTitle = () => document.querySelector('.tour-title')?.textContent;
const nextBtn = () => '.tour-nav .tour-btn.primary';

beforeEach(() => {
  vi.useFakeTimers();
  localStorage.clear();
  document.body.innerHTML = '';
  document.body.className = '';
  pasteToTerminal.mockClear();
});

afterEach(() => {
  closeTour();
  vi.useRealTimers();
});

describe('invite policy', () => {
  test('once: invites, and completing the tour dismisses permanently', () => {
    document.body.innerHTML = '<div class="terminal-card"></div>';
    invite('once');
    expect(cardTitle()).toBe('New here?');

    click('.tour-invite-actions .tour-btn.primary'); // Take the tour
    // Two steps survive with only the terminal card mounted; Done finishes.
    click(nextBtn());
    click(nextBtn());
    expect(document.querySelector('.tour-card')).toBe(null);
    expect(localStorage.getItem('osprey-tour-dismissed-v1')).toBe('1');

    invite('once');
    expect(document.querySelector('.tour-card')).toBe(null);
  });

  test('once: Esc mid-tour does NOT dismiss — the invite returns', () => {
    document.body.innerHTML = '<div class="terminal-card"></div>';
    invite('once');
    click('.tour-invite-actions .tour-btn.primary');
    closeTour();

    expect(localStorage.getItem('osprey-tour-dismissed-v1')).toBe(null);
    invite('once');
    expect(cardTitle()).toBe('New here?');
  });

  test('always: invites despite a dismissal flag, offers Not now and no checkbox', () => {
    document.body.innerHTML = '<div class="terminal-card"></div>';
    localStorage.setItem('osprey-tour-dismissed-v1', '1');
    invite('always');

    expect(cardTitle()).toBe('New here?');
    expect(document.querySelector('.tour-remember')).toBe(null);
    const skip = document.querySelector('.tour-invite-actions .tour-btn:not(.primary)');
    expect(skip?.textContent).toBe('Not now');
  });

  test('never: no automatic invite; startTour still works', () => {
    document.body.innerHTML = '<div class="terminal-card"></div>';
    invite('never');
    expect(document.querySelector('.tour-card')).toBe(null);

    startTour();
    expect(cardTitle()).toBe('Ask in plain language');
  });

  test('embedded pages never auto-invite', () => {
    document.body.innerHTML = '<div class="terminal-card"></div>';
    document.body.classList.add('embedded');
    invite('always');
    expect(document.querySelector('.tour-card')).toBe(null);
  });

  test('a failed payload leaves the tour on-demand only', () => {
    document.body.innerHTML = '<div class="terminal-card"></div>';
    applyTourConfig(null);
    vi.advanceTimersByTime(INVITE_DELAY_MS + 1);
    expect(document.querySelector('.tour-card')).toBe(null);
  });
});

describe('steps', () => {
  test('all nine steps run with the full shell mounted, in order', () => {
    mountFullShell();
    applyTourConfig({ tour: { policy: 'never', capabilities: [] } });
    startTour();

    const titles = [];
    for (let i = 0; i < 9; i++) {
      titles.push(cardTitle());
      click(nextBtn());
      vi.advanceTimersByTime(200); // settle re-render
    }
    expect(titles).toEqual([
      'Ask in plain language',
      'Your control target',
      'Make it yours',
      'Search everything',
      'Your workspace',
      'Facility knowledge',
      'The logbook',
      'Something wrong? Tell us',
      'Try it',
    ]);
    expect(document.querySelector('.tour-card')).toBe(null);
  });

  test('the workspace step mentions the shipped example only when the gallery lists one', () => {
    /** Advance to "Your workspace" (step 5) and return the card body text. */
    const workspaceBody = () => {
      applyTourConfig({ tour: { policy: 'never', capabilities: [] } });
      startTour();
      for (let i = 0; i < 4; i++) {
        click(nextBtn());
        vi.advanceTimersByTime(200);
      }
      expect(cardTitle()).toBe('Your workspace');
      return document.querySelector('.tour-body')?.textContent ?? '';
    };

    mountFullShell();
    expect(workspaceBody()).not.toContain('shipped sample');

    mountFullShell();
    const frame = document.createElement('iframe');
    frame.setAttribute('data-panel-id', 'artifacts');
    document.body.appendChild(frame);
    const doc = /** @type {Document} */ (frame.contentDocument);
    doc.body.innerHTML = '<div class="tree-section" data-type="examples"></div>';
    expect(workspaceBody()).toContain(
      'The entry under Examples is a shipped sample; your own work lands above it.'
    );
  });

  test('steps with absent anchors drop out and the count adjusts', () => {
    document.body.innerHTML = '<div class="terminal-card"></div>';
    startTour();

    expect(document.querySelector('.tour-kicker')?.textContent).toBe('Step 1 of 2');
  });

  test('the capability list renders verbatim from the payload', () => {
    document.body.innerHTML = '<div class="terminal-card"></div>';
    applyTourConfig({
      tour: { policy: 'never', capabilities: ['read live machine values', 'make plots'] },
    });
    startTour();

    expect(document.querySelector('.tour-body')?.textContent).toContain(
      'read live machine values, and make plots'
    );
  });

  test('chip-derived text renders as text — markup in a facility name stays inert', () => {
    mountFullShell();
    const short = find('.ctc-short');
    short.textContent = '<img src=x onerror=steal()>';
    startTour();
    click(nextBtn()); // → Your control target
    vi.advanceTimersByTime(200);

    expect(document.querySelector('.tour-body img')).toBe(null);
    expect(document.querySelector('.tour-body')?.textContent).toContain(
      '<img src=x onerror=steal()>'
    );
  });
});

describe('activation', () => {
  test('the target step opens the chip popover and moving on closes it', () => {
    mountFullShell();
    const chip = find('.control-target-chip');
    chip.addEventListener('click', () => {
      chip.setAttribute(
        'aria-expanded',
        chip.getAttribute('aria-expanded') === 'true' ? 'false' : 'true'
      );
    });
    startTour();
    click(nextBtn()); // → Your control target
    vi.advanceTimersByTime(200);
    expect(chip.getAttribute('aria-expanded')).toBe('true');

    click(nextBtn()); // → Make it yours
    vi.advanceTimersByTime(200);
    expect(chip.getAttribute('aria-expanded')).toBe('false');
  });

  test('a panel step presses its rail entry — but never the ACTIVE one', () => {
    mountFullShell();
    const workspace = find('[data-panel-id="artifacts"]');
    workspace.classList.add('active');
    const clicks = vi.fn();
    workspace.addEventListener('click', clicks);

    startTour();
    for (let i = 0; i < 4; i++) {
      click(nextBtn()); // → Your workspace
      vi.advanceTimersByTime(200);
    }
    expect(cardTitle()).toBe('Your workspace');
    expect(clicks).not.toHaveBeenCalled();
  });
});

describe('prompt chips', () => {
  test('a chip inserts its prompt into the terminal and sends nothing', async () => {
    document.body.innerHTML = '<div class="terminal-card"></div>';
    startTour();
    click(nextBtn()); // → Try it (only 2 steps here)
    vi.advanceTimersByTime(200);
    expect(cardTitle()).toBe('Try it');

    click('.tour-chip');
    await import(TERMINAL_PATH); // let the lazy import settle
    await Promise.resolve();
    expect(pasteToTerminal).toHaveBeenCalledWith('What can you see on this machine?');
    expect(pasteToTerminal.mock.calls[0][0].endsWith('\n')).toBe(false);
  });
});
