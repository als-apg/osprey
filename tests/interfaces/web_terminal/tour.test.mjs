// @ts-check
/**
 * Unit tests for the onboarding tour (tour.js).
 *
 *   npx vitest run tests/interfaces/web_terminal/tour.test.mjs
 *
 * The tour is an invite card followed by spotlighted steps over the live
 * shell. applyTourConfig() records the invite policy from GET /api/panels and
 * arms the automatic invite; startTour() is the on-demand entry (☰ More in the
 * sidebar, palette) that ignores the policy. Steps whose anchor is absent drop out. The
 * dismissal flag is storage-scoped (per-user scoping is covered in
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

/** Mount the full set of tour anchors, in the sidebar's shape. */
function mountFullShell() {
  document.body.innerHTML = `
    <nav id="sidebar" class="sidebar" aria-label="Panels">
      <div class="sidebar-head">
        <button id="sidebar-toggle" class="osprey-icon-btn"></button>
      </div>
      <div id="sidebar-target" class="sidebar-target">
        <button class="control-target-chip" aria-expanded="false">
          <span class="ctc-short">Simulator</span><span class="ctc-state">writes on</span>
        </button>
      </div>
      <button id="command-palette-btn" class="osprey-search"></button>
      <div id="panel-rail" role="list">
        <button class="osprey-nav-row panel-rail-button" data-panel-id="artifacts">
          <span class="osprey-nav-label">Artifacts</span>
        </button>
        <button class="osprey-nav-row panel-rail-button" data-panel-id="okf">
          <span class="osprey-nav-label">Knowledge</span>
        </button>
        <button class="osprey-nav-row panel-rail-button" data-panel-id="ariel">
          <span class="osprey-nav-label">Ariel</span>
        </button>
      </div>
      <div class="sidebar-foot">
        <a id="panel-docs-link" class="osprey-nav-row" hidden></a>
        <button id="panel-feedback-btn" class="osprey-nav-row"></button>
        <button id="more-btn" class="osprey-nav-row"></button>
      </div>
    </nav>
    <div class="terminal-card"></div>`;
}

/** Every step title, in order, with the full shell mounted. */
const ALL_TITLES = [
  'Ask in plain language',
  'The sidebar',
  'Your control target',
  'Search everything',
  'Your workspace',
  'Facility knowledge',
  'The logbook',
  'Something wrong? Tell us',
  'Try it',
];

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

/**
 * Press Next until the card titled `title` shows. Walked by title rather than
 * by count, so inserting a step ahead of it cannot retarget a test.
 * @param {string} title
 */
function goTo(title) {
  for (let guard = 0; guard < 20 && cardTitle() !== title; guard += 1) {
    click(nextBtn());
    vi.advanceTimersByTime(200);
  }
  expect(cardTitle()).toBe(title);
}

beforeEach(() => {
  vi.useFakeTimers();
  localStorage.clear();
  document.body.innerHTML = '';
  document.body.className = '';
  document.documentElement.removeAttribute('data-ui-mode');
  document.documentElement.removeAttribute('data-control-target');
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
    expect(localStorage.getItem('osprey-tour-dismissed-v2')).toBe('1');

    invite('once');
    expect(document.querySelector('.tour-card')).toBe(null);
  });

  test('once: Esc mid-tour does NOT dismiss — the invite returns', () => {
    document.body.innerHTML = '<div class="terminal-card"></div>';
    invite('once');
    click('.tour-invite-actions .tour-btn.primary');
    closeTour();

    expect(localStorage.getItem('osprey-tour-dismissed-v2')).toBe(null);
    invite('once');
    expect(cardTitle()).toBe('New here?');
  });

  test('always: invites despite a dismissal flag, offers Not now and no checkbox', () => {
    document.body.innerHTML = '<div class="terminal-card"></div>';
    localStorage.setItem('osprey-tour-dismissed-v2', '1');
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
    applyTourConfig({ tour: { policy: 'never' } });
    startTour();

    const titles = [];
    for (let i = 0; i < ALL_TITLES.length; i++) {
      titles.push(cardTitle());
      click(nextBtn());
      vi.advanceTimersByTime(200); // settle re-render
    }
    expect(titles).toEqual(ALL_TITLES);
    expect(document.querySelector('.tour-card')).toBe(null);
  });

  test('no step points at the retired header, display menu or bars', () => {
    mountFullShell();
    startTour();

    const text = [];
    for (let i = 0; i < ALL_TITLES.length; i++) {
      text.push(cardBody(), document.querySelector('.tour-foot')?.textContent ?? '');
      click(nextBtn());
      vi.advanceTimersByTime(200);
    }
    const all = text.join(' ');
    expect(all).not.toMatch(/header|status bar|on the rail|Customize bars/i);
  });

  test('panel cards wear the labels their sidebar rows show', () => {
    mountFullShell();
    find('[data-panel-id="artifacts"] .osprey-nav-label').textContent = 'Files';
    find('[data-panel-id="okf"] .osprey-nav-label').textContent = 'Docs';
    find('[data-panel-id="ariel"] .osprey-nav-label').textContent = 'Logbook';
    startTour();

    goTo('Your workspace');
    expect(cardBody()).toContain('Files');
    goTo('Facility knowledge');
    expect(cardBody()).toContain('Docs');
    goTo('The logbook');
    expect(cardBody()).toContain('Logbook');
  });

  test('a row that renders no label leaves the shipped names on the cards', () => {
    mountFullShell();
    for (const label of document.querySelectorAll('.osprey-nav-label')) label.remove();
    startTour();

    goTo('Your workspace');
    expect(cardBody()).toContain('Artifacts');
    goTo('Facility knowledge');
    expect(cardBody()).toContain('Knowledge');
    goTo('The logbook');
    expect(cardBody()).toContain('Ariel');
  });

  /** Walk the whole tour, collecting each card's title and foot line. */
  function walk() {
    applyTourConfig({ tour: { policy: 'never' } });
    startTour();
    const cards = [];
    for (let i = 0; i < 20 && document.querySelector('.tour-card'); i++) {
      cards.push({
        title: cardTitle(),
        foot: document.querySelector('.tour-foot')?.textContent ?? '',
      });
      click(nextBtn());
      vi.advanceTimersByTime(200);
    }
    return cards;
  }

  // A withdrawn control stays in the DOM, so presence alone would spotlight a
  // box that is not on screen.
  test('a step whose anchor sits in a hidden subtree drops out', () => {
    mountFullShell();
    find('#panel-feedback-btn').hidden = true;

    expect(walk().map((c) => c.title)).not.toContain('Something wrong? Tell us');
  });

  test('a hidden ancestor withdraws the anchor too', () => {
    mountFullShell();
    find('.sidebar-foot').hidden = true;

    expect(walk().map((c) => c.title)).not.toContain('Something wrong? Tell us');
  });

  // The block stays on screen for the clock, so only the flag withdraws it.
  test('the control-target step drops out when the deployment has none', () => {
    mountFullShell();
    document.documentElement.dataset.controlTarget = 'off';

    const titles = walk().map((c) => c.title);
    expect(titles).not.toContain('Your control target');
    expect(titles).toContain('The sidebar');
  });

  test('the last card says where the tour is retaken from', () => {
    mountFullShell();

    const cards = walk();

    expect(cards[cards.length - 1].foot).toBe(
      'Retake this tour any time from ☰ More in the sidebar.'
    );
  });

  test('steps with absent anchors drop out and the count adjusts', () => {
    document.body.innerHTML = '<div class="terminal-card"></div>';
    startTour();

    expect(document.querySelector('.tour-kicker')?.textContent).toBe('Step 1 of 2');
  });

  test('the first card says exactly what first contact derives', () => {
    document.body.innerHTML = '<div class="terminal-card"></div>';
    serve({ kind: 'simulator', capabilities: ['make plots'] });
    startTour();

    // Not a substring of the tour's own composing: the whole sentence, so the
    // card and the two views cannot drift apart a word at a time.
    expect(cardBody()).toContain(capabilitySentence());
    expect(cardBody()).toContain('read values from the simulator and make plots');
  });

  test('a simulator deployment is never described as reading the real machine', () => {
    document.body.innerHTML = '<div class="terminal-card"></div>';
    serve({ kind: 'simulator', capabilities: ['make plots'] });
    startTour();

    // The one claim the derivation exists to prevent.
    expect(cardBody()).not.toContain('real machine');
  });

  test('a rehearsal deployment reads the rehearsal copy, never the real machine', () => {
    document.body.innerHTML = '<div class="terminal-card"></div>';
    serve({ kind: 'rehearsal' });
    startTour();

    expect(cardBody()).toContain('read values from the rehearsal copy');
    expect(cardBody()).not.toContain('real machine');
  });

  test('the target step quotes the chip in its own words and names all three kinds', () => {
    mountFullShell();
    find('.ctc-short').textContent = 'Rehearsal';
    find('.ctc-state').textContent = 'writes off';
    startTour();
    goTo('Your control target');

    const body = document.querySelector('.tour-body')?.textContent ?? '';
    expect(body).toContain('currently on Rehearsal — writes off.');
    for (const word of ['live machine', 'rehearsal', 'simulator']) expect(body).toContain(word);
    expect(body).toContain('turn writes on or off');
    expect(body).not.toMatch(/stand-in|sandbox/i);
  });

  test('a session on no known machine gets no read phrase', () => {
    document.body.innerHTML = '<div class="terminal-card"></div>';
    serve({ kind: null });
    startTour();

    expect(capabilitySentence()).toBe('');
    expect(cardBody()).toBe('This terminal lets you talk to the OSPREY agent.');
  });

  test('chip-derived text renders as text — markup in a facility name stays inert', () => {
    mountFullShell();
    const short = find('.ctc-short');
    short.textContent = '<img src=x onerror=steal()>';
    startTour();
    goTo('Your control target');

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
    goTo('Your control target');
    expect(chip.getAttribute('aria-expanded')).toBe('true');

    goTo('Search everything');
    expect(chip.getAttribute('aria-expanded')).toBe('false');
  });

  test('the target step spotlights chip and popover as one outline, not their box', () => {
    mountFullShell();
    // Reduced motion draws the spotlight at once rather than easing it.
    vi.spyOn(window, 'matchMedia').mockReturnValue(
      /** @type {MediaQueryList} */ (/** @type {unknown} */ ({ matches: true }))
    );
    /** @param {Element} node @param {number[]} edges left, top, right, bottom */
    const place = (node, [left, top, right, bottom]) => {
      node.getBoundingClientRect = () =>
        /** @type {DOMRect} */ ({
          left,
          top,
          right,
          bottom,
          width: right - left,
          height: bottom - top,
        });
    };
    place(find('#sidebar-target'), [8, 10, 240, 50]);
    const chip = find('.control-target-chip');
    chip.addEventListener('click', () => {
      chip.setAttribute('aria-expanded', 'true');
      const popover = document.createElement('div');
      popover.id = 'control-target-popover';
      place(popover, [244, 10, 520, 260]);
      document.body.appendChild(popover);
    });
    startTour();
    goTo('Your control target');
    vi.advanceTimersByTime(200); // the settle re-render measures the popover

    // SVG nodes, so not find(): that one asserts an HTMLElement.
    const edge = document.querySelector('.tour-spot .tour-spot-edge')?.getAttribute('d') ?? '';
    // An L has six corners; the box round both would have four.
    expect(edge.match(/A/g)).toHaveLength(6);
    const dim = document.querySelector('.tour-spot .tour-spot-dim');
    expect(dim?.getAttribute('fill-rule')).toBe('evenodd');
    expect(dim?.getAttribute('d')).toContain(edge);
  });

  test('a panel step presses its sidebar row — but never the ACTIVE one', () => {
    mountFullShell();
    const workspace = find('[data-panel-id="artifacts"]');
    workspace.classList.add('active');
    const clicks = vi.fn();
    workspace.addEventListener('click', clicks);
    const knowledge = find('[data-panel-id="okf"]');
    const opened = vi.fn();
    knowledge.addEventListener('click', opened);

    startTour();
    goTo('Your workspace');
    expect(clicks).not.toHaveBeenCalled();
    goTo('Facility knowledge');
    expect(opened).toHaveBeenCalledTimes(1);
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

  test("the chips are first contact's starter prompts, in its order", () => {
    serve({ kind: 'simulator', logbook: true });
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

  test('the prompts follow the machine, resolved when the card renders', () => {
    serve({ kind: null });
    reachTryIt();
    expect(chipTexts()).not.toContain('What can you read right now?');

    // The chip settles on a machine while the tour is already open; going back
    // and forward re-renders the card, and the offer follows.
    chipKind.value = 'real';
    click('.tour-nav .tour-btn.ghost'); // Back
    click(nextBtn());
    vi.advanceTimersByTime(200);

    expect(chipTexts()).toContain('What can you read right now?');
  });

  test('in Simple view the chip reaches the visible input, not the hidden terminal', () => {
    document.documentElement.setAttribute('data-ui-mode', 'simple');
    serve({ kind: 'real' });
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

describe('the chords it teaches', () => {
  /**
   * Walk the tour to the palette step and return that card's body text.
   * @returns {string}
   */
  function paletteStepBody() {
    mountFullShell();
    applyTourConfig({ tour: { policy: 'never' } });
    startTour();
    goTo('Search everything');
    return cardBody();
  }

  /** @param {string} value */
  function stubPlatform(value) {
    Object.defineProperty(navigator, 'platform', { value, configurable: true });
    Object.defineProperty(navigator, 'userAgentData', { value: undefined, configurable: true });
  }

  test('macOS is taught the chord macOS binds', () => {
    stubPlatform('MacIntel');

    expect(paletteStepBody()).toContain('⌘K');
  });

  test('everywhere else is taught Ctrl+K, and where it works', () => {
    stubPlatform('Linux x86_64');

    const body = paletteStepBody();

    expect(body).toContain('Ctrl+K');
    expect(body).not.toContain('⌘');
    // Off macOS the palette binds the chord outside the terminal only, so the
    // tour has to say what to do inside it.
    expect(body).toContain('outside the terminal');
  });

  /** @returns {string} the sidebar card's body text */
  function sidebarStepBody() {
    mountFullShell();
    startTour();
    goTo('The sidebar');
    return cardBody();
  }

  test('the sidebar card teaches the label toggle in the macOS chord on macOS', () => {
    stubPlatform('MacIntel');

    expect(sidebarStepBody()).toContain('⌘\\');
  });

  test('the sidebar card teaches Ctrl+\\ everywhere else', () => {
    stubPlatform('Linux x86_64');

    const body = sidebarStepBody();
    expect(body).toContain('Ctrl+\\');
    expect(body).not.toContain('⌘');
  });

  test('the sidebar card names both sections', () => {
    const body = sidebarStepBody();
    expect(body).toContain('Workspace');
    expect(body).toContain('Tools');
  });
});
