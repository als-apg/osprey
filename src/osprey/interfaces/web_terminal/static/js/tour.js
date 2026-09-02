// @ts-check
/* OSPREY Web Terminal — Onboarding tour.
 *
 * An invite card (Take the tour / Skip) followed by spotlighted steps over
 * the LIVE shell: each step may drive the UI it points at (open the
 * control-target popover, show a panel) and undoes that action when the
 * tour moves on. Replaces the retired one-time rail hint.
 *
 * Honesty rule: the tour never promises safety properties. The target step
 * reads the control-target chip's own rendered words (the server-derived
 * facts) and states them as the CURRENT configuration — nothing more.
 * Approval flows are config-dependent and therefore never mentioned. The
 * capability list arrives derived from the server (`GET /api/panels` →
 * `tour.capabilities`); the browser renders it verbatim and invents
 * nothing.
 *
 * Invite policy (`GET /api/panels` → `tour.policy`, resolved server-side
 * from `web.tour` / `OSPREY_WEB_TOUR`):
 *   once   — invite until this browser dismisses it ("Don't show this
 *            again", or completing the tour)
 *   always — invite on every load, no permanent dismissal (shared screens)
 *   never  — no automatic invite
 * The rail's Tour control and the command palette start the tour on demand
 * regardless of policy. The dismissal flag is storage-scoped so users
 * sharing a browser on a multi-user deployment do not dismiss each other's
 * invite.
 *
 * All text is assembled with createElement/textContent — never innerHTML —
 * because two rendered values (the chip's target name and state) originate
 * in facility-configured free text.
 *
 * Steps whose anchor is absent on this page (panel not enabled, simple UI
 * mode, no chip) drop out and the step count adjusts — and so do steps whose
 * anchor is a bar item the operator removed, which is a different fact from
 * absence: a removed item is parked in the hidden pool, not deleted, so it
 * still answers `querySelector`. See `liveAnchor()`. Esc closes at any
 * point; arrows/Enter navigate; focus is trapped inside the card.
 */

import { installFocusTrap, removeFocusTrap } from '/design-system/js/focus-trap.js';
import { scopedStorageKey } from '/design-system/js/storage-scope.js';
import { isLive } from './bar-host.js';

/**
 * Insert a prompt into the terminal — inserted, not sent; the visitor
 * presses Enter. Imported lazily so this module stays loadable without the
 * xterm stack (jsdom tests, future embedders).
 * @param {string} text
 */
async function insertPrompt(text) {
  const { pasteToTerminal } = await import('./terminal.js');
  pasteToTerminal(text);
}

/** Dismissal flag ("Don't show this again" / completed). Scoped per user. */
const DISMISS_BASE = 'osprey-tour-dismissed-v1';

/** Delay between boot and the automatic invite, so the shell settles. */
export const INVITE_DELAY_MS = 700;

/** How long activation (popover, panel) gets before the spotlight re-wraps. */
const SETTLE_MS = 160;

/* ---- Server-derived facts (applyTourConfig) ---- */

/** @type {{policy: string, capabilities: string[]}} */
let facts = { policy: 'never', capabilities: [] };

/* ---- Storage ---- */

const dismissKey = () => scopedStorageKey(DISMISS_BASE);

const isDismissed = () => {
  try {
    return localStorage.getItem(dismissKey()) === '1';
  } catch {
    return false;
  }
};

/** @param {boolean} on */
const setDismissed = (on) => {
  try {
    if (on) localStorage.setItem(dismissKey(), '1');
    else localStorage.removeItem(dismissKey());
  } catch {
    /* storage unavailable — the invite just shows again next time */
  }
};

/* ---- DOM helpers (no innerHTML anywhere) ---- */

/**
 * Keyed on the tag literal so callers get the concrete element type back
 * (`el('button', …)` is an HTMLButtonElement, with `.type` and `.disabled`).
 * @template {keyof HTMLElementTagNameMap} K
 * @param {K} tag
 * @param {string} [cls]
 * @param {string} [text]
 * @returns {HTMLElementTagNameMap[K]}
 */
function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
}

/** @param {string} text @returns {HTMLElement} */
const strong = (text) => el('strong', '', text);

/**
 * Append a mix of strings (→ text nodes) and elements to a parent.
 * @param {HTMLElement} parent
 * @param {(string | HTMLElement)[]} parts
 */
function fill(parent, parts) {
  for (const part of parts) {
    parent.append(typeof part === 'string' ? document.createTextNode(part) : part);
  }
  return parent;
}

/* ---- Step activation helpers ---- */

/**
 * Open a popover behind an aria-expanded trigger; return the closer.
 * @param {string} selector
 * @returns {() => void}
 */
const toggleOpen = (selector) => {
  const trigger = document.querySelector(selector);
  if (trigger instanceof HTMLElement && trigger.getAttribute('aria-expanded') !== 'true') {
    trigger.click();
  }
  return () => {
    const t = document.querySelector(selector);
    if (t instanceof HTMLElement && t.getAttribute('aria-expanded') === 'true') t.click();
  };
};

/**
 * Show a panel by pressing its rail entry. A click on the ACTIVE entry
 * toggles the panel closed, so only click to open. Panels deliberately stay
 * where the tour put them (no cleanup) — ending on an open WORKSPACE is the
 * point.
 * @param {string} panelId
 * @returns {null}
 */
const showPanel = (panelId) => {
  const btn = document.querySelector(`.panel-rail-button[data-panel-id="${panelId}"]`);
  if (btn instanceof HTMLElement && !btn.classList.contains('active')) btn.click();
  return null;
};

/** @param {Element | null} node @returns {node is HTMLElement} */
const visible = (node) => {
  if (!(node instanceof HTMLElement)) return false;
  const r = node.getBoundingClientRect();
  return r.width > 0 && r.height > 0;
};

/* ---- Copy ---- */

/**
 * The chip's own words — target name and write state as currently rendered.
 * Reading the DOM keeps the tour and the chip incapable of disagreeing.
 * @returns {{name: string, state: string} | null}
 */
const chipFacts = () => {
  const short = document.querySelector('.control-target-chip .ctc-short');
  const state = document.querySelector('.control-target-chip .ctc-state');
  if (!short || !state) return null;
  return { name: (short.textContent || '').trim(), state: (state.textContent || '').trim() };
};

/** "a, b, and c" @param {string[]} items */
const listPhrase = (items) =>
  items.length <= 1 ? items.join('') : `${items.slice(0, -1).join(', ')}, and ${items[items.length - 1]}`;

/**
 * @typedef {Object} TourStep
 * @property {() => Element | null} anchor
 * @property {string} title
 * @property {() => (string | HTMLElement)[]} body
 * @property {(() => (() => void) | null)} [activate]
 * @property {() => (Element | null)[]} [extras]
 * @property {'left'} [place]
 * @property {string} [foot]
 * @property {string[]} [chips]
 */

/**
 * A step's anchor, but only when it is somewhere the operator can see.
 *
 * Four anchors are now bar ITEMS — the control-target chip, the display menu,
 * the command-palette button and (through the rail) the feedback control — and
 * a bar item the layout does not name is MOVED into the hidden
 * `#bar-item-pool`, never removed. So `querySelector` keeps answering for
 * chrome that is off screen, and a presence check alone would spotlight a
 * zero-rect node inside a hidden container: an empty highlight, a card
 * positioned against nothing, and a step count that lies.
 *
 * `isLive()` is bar-host.js's own answer to "is this node in a bar or in the
 * pool", so the tour and the bars cannot disagree about it. It is deliberately
 * the whole test — measuring the anchor instead would drop live chrome that
 * happens to measure zero at the moment the tour starts (an item mid-transition,
 * a bar not yet laid out), which is the opposite failure and a much quieter one.
 * A non-bar anchor (`.terminal-card`, a rail button) is never inside the pool,
 * so it behaves exactly as before.
 *
 * @param {TourStep} step
 * @returns {Element | null}
 */
const liveAnchor = (step) => {
  const node = step.anchor();
  return node && isLive(node) ? node : null;
};

/** @type {TourStep[]} */
const STEPS = [
  {
    anchor: () => document.querySelector('.terminal-card'),
    title: 'Ask in plain language',
    body: () => {
      const parts = [
        'This terminal talks to the ',
        strong('OSPREY agent'),
        '. Type what you want — no commands to learn.',
      ];
      if (facts.capabilities.length > 0) {
        parts.push(` Here it can ${listPhrase(facts.capabilities)}.`);
      }
      return parts;
    },
    foot: 'Enter sends · Esc interrupts the agent at any time.',
  },
  {
    anchor: () => document.querySelector('.control-target-chip'),
    activate: () => toggleOpen('.control-target-chip'),
    extras: () => [document.querySelector('.ctc-popover.open')],
    place: 'left',
    title: 'Your control target',
    body: () => {
      const parts = [
        'A session points at one control target: the ',
        strong('real machine'),
        ', a ',
        strong('rehearsal'),
        ' (a copy of the real controls — nothing moves), or a ',
        strong('simulator'),
        '.',
      ];
      const c = chipFacts();
      if (c) {
        parts.push(' This session is currently on ', strong(c.name), ' — ', strong(c.state), '.');
      }
      parts.push(' Click the chip to see the details or switch targets.');
      return parts;
    },
  },
  {
    anchor: () => document.querySelector('#display-menu'),
    activate: () => toggleOpen('.display-menu-trigger'),
    extras: () => [document.querySelector('.display-menu-card')],
    place: 'left',
    title: 'Make it yours',
    body: () => [
      'Light or dark theme, and a ',
      strong('simple or expert view'),
      ' of the terminal — switch both here, any time.',
    ],
  },
  {
    anchor: () => document.querySelector('#command-palette-btn'),
    title: 'Search everything',
    body: () => [
      'Press ',
      strong('⌘K'),
      ' to search settings, panels, and actions — the fastest way to find anything in this terminal.',
    ],
  },
  {
    anchor: () => document.querySelector('.panel-rail-button[data-panel-id="artifacts"]'),
    activate: () => showPanel('artifacts'),
    extras: () => [document.querySelector('iframe[data-panel-id="artifacts"]')],
    title: 'Your workspace',
    body: () => [
      'Everything the agent produces — plots, data files, reports — lands in ',
      strong('WORKSPACE'),
      ', ready to open or download. Its rail entry glows when something new arrives.',
    ],
    foot: '＋ on the rail adds more panels.',
  },
  {
    anchor: () => document.querySelector('.panel-rail-button[data-panel-id="okf"]'),
    activate: () => showPanel('okf'),
    extras: () => [document.querySelector('iframe[data-panel-id="okf"]')],
    title: 'Facility knowledge',
    body: () => [
      strong('KNOWLEDGE'),
      ' holds this facility’s curated documentation — the same material the agent consults when you ask about the machine. Browse it directly here.',
    ],
  },
  {
    anchor: () => document.querySelector('.panel-rail-button[data-panel-id="ariel"]'),
    activate: () => showPanel('ariel'),
    extras: () => [document.querySelector('iframe[data-panel-id="ariel"]')],
    title: 'The logbook',
    body: () => [
      strong('ARIEL'),
      ' searches the electronic logbook. Ask the agent what happened on a shift, or search it yourself here.',
    ],
    foot: 'Tip: right-click any panel entry → “Open in a new window” for a standalone version.',
  },
  {
    anchor: () => document.querySelector('#panel-feedback-btn'),
    title: 'Something wrong? Tell us',
    body: () => ['If the agent — or this terminal — gets something wrong, report it here.'],
  },
  {
    anchor: () => document.querySelector('.terminal-card'),
    title: 'Try it',
    body: () => [
      'Pick a question to start — it’s typed into the terminal for you; press Enter to send.',
    ],
    chips: ['What can you see on this machine?', 'What are you allowed to do in this session?'],
    foot: 'Retake this tour any time from the rail.',
  },
];

/* ---- Engine ---- */

const tour = {
  open: false,
  i: -1, // -1 = invite card
  /** @type {TourStep[]} */
  steps: [],
  /** @type {Record<string, HTMLElement>} */
  els: {},
  /** @type {((e: KeyboardEvent) => void) | null} */
  keyHandler: null,
  /** @type {number | null} */
  activatedStep: null,
  /** @type {(() => void) | null} */
  activeCleanup: null,

  runCleanup() {
    if (this.activeCleanup) this.activeCleanup();
    this.activeCleanup = null;
    this.activatedStep = null;
  },

  /** @param {boolean} [withInvite] start at the invite card (default) or step 0 */
  start(withInvite = true) {
    this.steps = STEPS.filter((s) => liveAnchor(s));
    if (this.steps.length === 0) return;
    this.open = true;
    this.i = withInvite ? -1 : 0;
    this.render();
  },

  stop() {
    this.runCleanup();
    this.open = false;
    if (this.els.card) removeFocusTrap(this.els.card);
    Object.values(this.els).forEach((node) => node.remove());
    this.els = {};
    if (this.keyHandler) {
      document.removeEventListener('keydown', this.keyHandler);
      this.keyHandler = null;
    }
  },

  finish() {
    // Completing the tour counts as "seen" under the `once` policy; an
    // Esc/✕ mid-tour does not — the invite returns next load.
    if (facts.policy === 'once') setDismissed(true);
    this.stop();
  },

  next() {
    if (this.i >= this.steps.length - 1) return this.finish();
    this.i++;
    this.render();
  },

  back() {
    this.i = this.i <= 0 ? -1 : this.i - 1;
    this.render();
  },

  /** @param {string} name @param {string} cls */
  ensure(name, cls) {
    if (!this.els[name]) {
      const node = el('div', cls);
      document.body.appendChild(node);
      this.els[name] = node;
    }
    return this.els[name];
  },

  /** @param {...string} names */
  drop(...names) {
    for (const n of names) {
      if (this.els[n]) {
        if (n === 'card') removeFocusTrap(this.els[n]);
        this.els[n].remove();
        delete this.els[n];
      }
    }
  },

  ensureKeys() {
    if (this.keyHandler) return;
    this.keyHandler = (e) => {
      if (!this.open) return;
      if (e.key === 'Escape') this.stop();
      else if ((e.key === 'ArrowRight' || e.key === 'Enter') && this.i >= 0) {
        // Enter on a focused button already clicks it; only treat Enter as
        // "next" when focus is not on an interactive tour control.
        if (e.key === 'Enter' && e.target instanceof HTMLElement && e.target.closest('button')) {
          return;
        }
        this.next();
      } else if (e.key === 'ArrowLeft') this.back();
    };
    document.addEventListener('keydown', this.keyHandler);
  },

  /** Build the shared card shell; returns {card, body} to fill. */
  rebuildCard() {
    const card = this.ensure('card', 'tour-card');
    removeFocusTrap(card);
    card.replaceChildren();
    card.setAttribute('role', 'dialog');
    card.setAttribute('aria-modal', 'true');
    return card;
  },

  render() {
    this.ensureKeys();
    if (this.i === -1) return this.renderInvite();
    this.drop('veil');

    const step = this.steps[this.i];
    // Re-resolved every render, and through the same liveness test start()
    // used: a step's anchor can be parked mid-tour — the overflow ladder folds
    // an item the moment the window narrows past it — and spotlighting a node
    // that has since moved into the pool is exactly what liveAnchor prevents.
    const target = liveAnchor(step);
    if (!target) return this.next();

    if (this.activatedStep !== this.i) {
      this.runCleanup();
      this.activatedStep = this.i;
      this.activeCleanup = step.activate ? step.activate() : null;
      const entered = this.i;
      setTimeout(() => {
        if (this.open && this.i === entered) this.render();
      }, SETTLE_MS);
    }

    // Spotlight rect: the anchor plus whatever its activation revealed.
    // Falls back to the bare anchor when nothing measures visible (a
    // zero-rect anchor must still produce a finite union).
    const parts = [target, ...(step.extras ? step.extras() : [])].filter(visible);
    const rects = (parts.length ? parts : [target]).map((p) => p.getBoundingClientRect());
    const top0 = Math.min(...rects.map((q) => q.top));
    const left0 = Math.min(...rects.map((q) => q.left));
    const right0 = Math.max(...rects.map((q) => q.right));
    const bottom0 = Math.max(...rects.map((q) => q.bottom));
    const pad = 6;

    const spot = this.ensure('spot', 'tour-spot');
    Object.assign(spot.style, {
      top: `${top0 - pad}px`,
      left: `${left0 - pad}px`,
      width: `${right0 - left0 + 2 * pad}px`,
      height: `${bottom0 - top0 + 2 * pad}px`,
    });
    this.ensure('blocker', 'tour-blocker');

    const card = this.rebuildCard();
    card.classList.remove('centered');
    const titleId = 'tour-step-title';
    card.setAttribute('aria-labelledby', titleId);

    const cancel = el('button', 'tour-x', '✕');
    cancel.type = 'button';
    cancel.title = 'End the tour';
    cancel.setAttribute('aria-label', 'End the tour');
    cancel.onclick = () => this.stop();

    const kicker = el('div', 'tour-kicker', `Step ${this.i + 1} of ${this.steps.length}`);
    const title = el('h3', 'tour-title', step.title);
    title.id = titleId;
    const body = fill(el('p', 'tour-body'), step.body());

    card.append(cancel, kicker, title, body);

    if (step.chips) {
      const chips = el('div', 'tour-chips');
      for (const prompt of step.chips) {
        const chip = el('button', 'tour-chip', prompt);
        chip.type = 'button';
        chip.onclick = () => insertPrompt(prompt);
        chips.appendChild(chip);
      }
      card.appendChild(chips);
    }
    if (step.foot) card.appendChild(el('div', 'tour-foot', step.foot));

    const nav = el('div', 'tour-nav');
    const backBtn = el('button', 'tour-btn ghost', 'Back');
    backBtn.type = 'button';
    if (this.i === 0) backBtn.disabled = true;
    backBtn.onclick = () => this.back();
    const dots = el('div', 'tour-dots');
    this.steps.forEach((_, k) => {
      dots.appendChild(el('span', `tour-dot${k === this.i ? ' on' : ''}`));
    });
    const last = this.i === this.steps.length - 1;
    const nextBtn = el('button', 'tour-btn primary', last ? 'Done' : 'Next');
    nextBtn.type = 'button';
    nextBtn.onclick = () => this.next();
    nav.append(backBtn, dots, nextBtn);
    card.appendChild(nav);

    // Position: 'left' steps sit beside the spotlight (their activation
    // opens a dropdown below the anchor); otherwise below when there is
    // room, else above. Clamped to the viewport.
    const cw = 372;
    const margin = 12;
    card.style.visibility = 'hidden';
    card.style.top = '0px';
    card.style.left = '0px';
    const ch = card.getBoundingClientRect().height || 200;
    let top;
    let left;
    if (step.place === 'left' && left0 - cw - margin >= 12) {
      top = Math.min(Math.max(12, top0), window.innerHeight - ch - 12);
      left = left0 - cw - margin;
    } else {
      top = bottom0 + margin;
      if (top + ch > window.innerHeight - 40) top = Math.max(12, top0 - ch - margin);
      left = left0 + (right0 - left0) / 2 - cw / 2;
      left = Math.min(Math.max(12, left), window.innerWidth - cw - 12);
    }
    Object.assign(card.style, { top: `${top}px`, left: `${left}px`, visibility: '' });

    installFocusTrap(card);
    nextBtn.focus();
  },

  renderInvite() {
    this.runCleanup();
    this.drop('spot', 'blocker');
    this.ensure('veil', 'tour-veil');

    const card = this.rebuildCard();
    card.classList.add('centered');
    const titleId = 'tour-invite-title';
    card.setAttribute('aria-labelledby', titleId);

    const title = el('h3', 'tour-title', 'New here?');
    title.id = titleId;
    const body = el(
      'p',
      'tour-body',
      'A two-minute tour of this terminal — what you’re talking to, what it may do, and how to ask.'
    );

    const actions = el('div', 'tour-invite-actions');
    const take = el('button', 'tour-btn primary', 'Take the tour');
    take.type = 'button';
    take.onclick = () => {
      this.i = 0;
      this.render();
    };
    // Under `always` there is no permanent dismissal — the skip is "Not now".
    const skip = el('button', 'tour-btn', facts.policy === 'always' ? 'Not now' : 'Skip');
    skip.type = 'button';
    actions.append(take, skip);
    card.append(title, body, actions);

    if (facts.policy === 'once') {
      const remember = el('label', 'tour-remember');
      const box = el('input');
      box.setAttribute('type', 'checkbox');
      remember.append(box, document.createTextNode(' Don’t show this again'));
      card.appendChild(remember);
      skip.onclick = () => {
        setDismissed(/** @type {HTMLInputElement} */ (box).checked);
        this.stop();
      };
    } else {
      skip.onclick = () => this.stop();
    }

    installFocusTrap(card);
    take.focus();
  },
};

window.addEventListener('resize', () => {
  if (tour.open && tour.i >= 0) tour.render();
});

/* ---- Public seams ---- */

/**
 * Start the tour on demand (rail Tour control, command palette), regardless
 * of the invite policy. Restarts from the first step when already open.
 */
export function startTour() {
  tour.stop();
  tour.start(false);
}

/**
 * Record the server-derived tour facts and arm the automatic invite.
 *
 * Called once from panel-manager's boot with the `GET /api/panels` payload
 * (the page's ONE round trip). A null/failed payload leaves the tour
 * on-demand only — the invite never fires on guessed facts. The automatic
 * invite is additionally suppressed on embedded pages (the shell inside a
 * dashboard is not a first visit).
 *
 * @param {{tour?: {policy?: unknown, capabilities?: unknown}} | null | undefined} panelsPayload
 */
export function applyTourConfig(panelsPayload) {
  const cfg = panelsPayload?.tour;
  if (!cfg) return;
  facts = {
    policy: typeof cfg.policy === 'string' ? cfg.policy : 'never',
    capabilities: Array.isArray(cfg.capabilities)
      ? cfg.capabilities.filter((c) => typeof c === 'string')
      : [],
  };

  const invite =
    facts.policy === 'always' || (facts.policy === 'once' && !isDismissed());
  if (!invite || document.body.classList.contains('embedded')) return;
  setTimeout(() => {
    if (!tour.open) tour.start();
  }, INVITE_DELAY_MS);
}

/* ---- Rail control ---- */

const railButton = document.getElementById('panel-tour-btn');
if (railButton) {
  railButton.addEventListener('click', (event) => {
    event.preventDefault();
    startTour();
  });
}
