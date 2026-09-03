// @ts-check
/**
 * Unit tests for the onboarding tour (tour.js).
 *
 *   npx vitest run tests/interfaces/web_terminal/tour.test.mjs
 *
 * The tour is an invite card followed by spotlighted steps over the live
 * shell. applyTourConfig() records the invite policy from GET /api/panels and
 * arms the automatic invite; startTour() is the on-demand entry (rail control,
 * palette) that ignores the policy. Steps whose anchor is absent drop out. The
 * dismissal flag is storage-scoped (per-persona scoping is covered in
 * js/storage-scope-keys.test.js).
 *
 * What the first card SAYS about this deployment, and what the "Try it" chips
 * offer, are first-contact.js's derivations — so the tour cannot disagree with
 * the two views about one deployment. That module is deliberately NOT mocked
 * here: loading it for real is what proves the tour reaches the same
 * derivation the views do, rather than a stub that agrees by construction.
 * Its two inputs are the seams instead — the server's facts through
 * `setFacts`, and the machine kind through the mocked chip.
 *
 * Seams: terminal.js is mocked in the chip suite's shape (it owns the session
 * id and the xterm stack), extended with the insert seam's exports, because
 * the module graph now reaches it statically; the control-target chip is
 * mocked because reading a real kind would need a mounted chip and a served
 * posture. The prompt chips must INSERT text, never send it.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

/** Mutable stand-in for terminal.js, reachable from the hoisted vi.mock factory. */
const term = vi.hoisted(() => ({
  /** @type {string|null} */
  sessionId: /** @type {string|null} */ (null),
  /** @type {(() => void)[]} */
  listeners: [],
  paste: vi.fn(),
  focus: vi.fn(),
}));

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/terminal.js', () => ({
  getCurrentSessionId: () => term.sessionId,
  /** @param {() => void} fn */
  onSessionChange: (fn) => term.listeners.push(fn),
  pasteToTerminal: term.paste,
  focusTerminal: term.focus,
}));

/** Mutable stand-in for the control-target chip, which owns the machine kind. */
const chipKind = vi.hoisted(() => ({
  /** @type {string|null} */
  value: /** @type {string|null} */ (null),
}));

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/control-target-chip.js', () => ({
  activeKind: () => chipKind.value,
  subscribe: vi.fn(),
}));

import {
  applyTourConfig,
  startTour,
  INVITE_DELAY_MS,
} from '../../../src/osprey/interfaces/web_terminal/static/js/tour.js';
import {
  capabilitySentence,
  setFacts,
  starterPrompts,
} from '../../../src/osprey/interfaces/web_terminal/static/js/first-contact.js';

/** Mount the full set of tour anchors. */
function mountFullShell() {
  document.body.innerHTML = `
    <button class="control-target-chip" aria-expanded="false">
      <span class="ctc-short">Simulator</span><span class="ctc-state">writes</span>
    </button>
    <osprey-display-menu id="display-menu">
      <button class="display-menu-trigger" aria-expanded="false"></button>
      <div class="display-menu-card">
        <button class="display-menu-settings bar-customize-entry">Customize bars</button>
      </div>
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
 */
function invite(policy = 'once') {
  applyTourConfig({ tour: { policy } });
  vi.advanceTimersByTime(INVITE_DELAY_MS + 1);
}

/**
 * Publish what this deployment stands on, both halves at once: the machine
 * kind the chip would name, and what the server said about the deployment.
 * @param {{kind?: string|null, capabilities?: string[], logbook?: boolean}} [state]
 */
function serve({ kind = null, capabilities = [], logbook = false } = {}) {
  chipKind.value = kind;
  setFacts({ capabilities, logbook });
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

/** The first card's body text, as an operator reads it. */
const cardBody = () => document.querySelector('.tour-body')?.textContent ?? '';

/** The prompts the current card offers, in order. */
const chipTexts = () =>
  [...document.querySelectorAll('.tour-chip')].map((c) => c.textContent ?? '');

beforeEach(() => {
  vi.useFakeTimers();
  localStorage.clear();
  document.body.innerHTML = '';
  document.body.className = '';
  document.documentElement.removeAttribute('data-ui-mode');
  vi.clearAllMocks();
  // first-contact.js holds this deployment's facts as module state, and these
  // tests share one instance of it: reset both of its inputs, or one test's
  // machine leaks into the next one's card.
  serve();
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
  test('all ten steps run with the full shell mounted, in order', () => {
    mountFullShell();
    applyTourConfig({ tour: { policy: 'never' } });
    startTour();

    const titles = [];
    for (let i = 0; i < 10; i++) {
      titles.push(cardTitle());
      click(nextBtn());
      vi.advanceTimersByTime(200); // settle re-render
    }
    expect(titles).toEqual([
      'Ask in plain language',
      'Your control target',
      'Make it yours',
      'Arrange the bars',
      'Search everything',
      'Your workspace',
      'Facility knowledge',
      'The logbook',
      'Something wrong? Tell us',
      'Try it',
    ]);
    expect(document.querySelector('.tour-card')).toBe(null);
  });

  test('steps with absent anchors drop out and the count adjusts', () => {
    document.body.innerHTML = '<div class="terminal-card"></div>';
    startTour();

    expect(document.querySelector('.tour-kicker')?.textContent).toBe('Step 1 of 2');
  });

  test('the first card says exactly what first contact derives', () => {
    document.body.innerHTML = '<div class="terminal-card"></div>';
    serve({ kind: 'simulated', capabilities: ['make plots'] });
    startTour();

    // Not a substring of the tour's own composing: the whole sentence, so the
    // card and the two views cannot drift apart a word at a time.
    expect(cardBody()).toContain(capabilitySentence());
    expect(cardBody()).toContain('read demo data and make plots');
  });

  test('a demo deployment is never described as reading the live machine', () => {
    document.body.innerHTML = '<div class="terminal-card"></div>';
    serve({ kind: 'simulated', capabilities: ['make plots'] });
    startTour();

    // The one claim the derivation exists to prevent. The server used to emit
    // this phrase for the mock connector too, and the card repeated it.
    expect(cardBody()).not.toContain('live machine');
  });

  test('a live deployment says so', () => {
    document.body.innerHTML = '<div class="terminal-card"></div>';
    serve({ kind: 'live', capabilities: ['make plots'] });
    startTour();

    expect(cardBody()).toContain(capabilitySentence());
    expect(cardBody()).toContain('read live machine values');
  });

  test('a session on no known machine gets no read phrase', () => {
    document.body.innerHTML = '<div class="terminal-card"></div>';
    serve({ kind: null });
    startTour();

    expect(capabilitySentence()).toBe('');
    expect(cardBody()).toBe('This terminal lets you talk to the OSPREY agent.');
  });

  test('a deployment with no capabilities still names where values come from', () => {
    document.body.innerHTML = '<div class="terminal-card"></div>';
    serve({ kind: 'standin' });
    startTour();

    expect(cardBody()).toContain('read values from the rehearsal copy');
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
    // Walked by title rather than by count, so inserting a step ahead of this
    // one cannot silently retarget the assertion at a different card.
    for (let guard = 0; guard < 20 && cardTitle() !== 'Your workspace'; guard += 1) {
      click(nextBtn());
      vi.advanceTimersByTime(200);
    }
    expect(cardTitle()).toBe('Your workspace');
    expect(clicks).not.toHaveBeenCalled();
  });
});

describe('prompt chips', () => {
  /** Walk the two-step tour on a bare terminal card to the "Try it" card. */
  function reachTryIt() {
    document.body.innerHTML = '<div class="terminal-card"></div>';
    startTour();
    click(nextBtn()); // → Try it (only 2 steps here)
    vi.advanceTimersByTime(200);
    expect(cardTitle()).toBe('Try it');
  }

  test('the chips are first contact\'s starter prompts, in its order', () => {
    serve({ kind: 'simulated', logbook: true });
    reachTryIt();

    expect(chipTexts()).toEqual(starterPrompts());
    // Pinned as literals too, so a derivation that quietly went empty could
    // not satisfy the comparison above by matching nothing.
    expect(chipTexts()).toEqual([
      'What can you read right now?',
      'What are you allowed to do in this session?',
      'What happened in the logbook today?',
    ]);
  });

  test('a deployment that cannot answer a question does not offer it', () => {
    serve({ kind: null, logbook: false });
    reachTryIt();

    expect(chipTexts()).toEqual(['What are you allowed to do in this session?']);
  });

  test('the prompts follow the machine, resolved when the card renders', () => {
    serve({ kind: null });
    reachTryIt();
    expect(chipTexts()).not.toContain('What can you read right now?');

    // The chip settles on a machine while the tour is already open; going back
    // and forward re-renders the card, and the offer follows.
    chipKind.value = 'live';
    click('.tour-nav .tour-btn.ghost'); // Back
    click(nextBtn());
    vi.advanceTimersByTime(200);

    expect(chipTexts()).toContain('What can you read right now?');
  });

  test('a chip inserts its prompt into the terminal and sends nothing', () => {
    serve({ kind: 'live' });
    reachTryIt();

    click('.tour-chip');

    expect(term.paste).toHaveBeenCalledWith('What can you read right now?');
    expect(term.paste.mock.calls[0][0].endsWith('\n')).toBe(false);
    // Inserted where the operator can carry on typing — the seam moves focus,
    // it does not press Enter for them.
    expect(term.focus).toHaveBeenCalled();
  });

  test('in Simple view the chip reaches the visible input, not the hidden terminal', () => {
    document.documentElement.setAttribute('data-ui-mode', 'simple');
    serve({ kind: 'live' });
    reachTryIt();
    // The Simple view's one prompt input, alongside the tour's own anchor.
    document.body.insertAdjacentHTML(
      'beforeend',
      '<div id="operator-container"><div class="op-input-area"><textarea></textarea></div></div>'
    );

    click('.tour-chip');

    const input = /** @type {HTMLTextAreaElement} */ (
      document.querySelector('#operator-container .op-input-area textarea')
    );
    expect(input.value).toBe('What can you read right now?');
    expect(term.paste).not.toHaveBeenCalled();
  });
});
