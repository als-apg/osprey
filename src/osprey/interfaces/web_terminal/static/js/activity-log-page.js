// @ts-check
/**
 * The Activity page at work: which session it reads, when it reads again, and
 * what a press does. It owns the shell `session.html` ships (toolbar, session
 * list, `#inner`) and fills it from one read, `GET /api/session-activity`.
 *
 * Which session: the transcript the hash names (`#<id>/<view>`, set by a pick
 * in the session list or by a shared link), else the session key the page was
 * opened on, else whatever the terminal is on now.
 *
 * When again: every ten seconds while the page is visible and either follows
 * the terminal (no transcript picked), or shows a picked session that is live,
 * or got no answer to its last read; at once on Refresh and when the page
 * becomes visible. An answer equal to the one shown changes nothing on the
 * page; a changed one keeps what the operator opened, where they scrolled and
 * what holds the keyboard focus. A read that gets no answer keeps the last
 * reading of the same session on the page, marked as such.
 *
 * @module activity-log-page
 */

import { copyText } from '/design-system/js/clipboard.js';
import { debounce } from '/design-system/js/dom.js';
import { icon } from '/design-system/js/icons.js';
import { openInfoPopover } from '/design-system/js/info-popover.js';
import { showToast } from '/design-system/js/toast.js';

import {
  recordBanner,
  recordChip,
  recordPopover,
  renderHeader,
  renderLiveState,
  renderSidebar,
  renderStats,
  viewCounts,
} from './activity-log-chrome.js';
import { VIEWS, artifactFileUrl, readHash, shareUrl, writeHash } from './activity-log-link.js';
import { payloadSignature, summarize } from './activity-log-model.js';
import { iconHtml, stateBlock } from './activity-log-rows.js';
import { applyAction, newUiState, resetForSession } from './activity-log-state.js';
import { renderTimeline } from './activity-log-timeline.js';
import { renderArtifacts, renderSafety, renderTools } from './activity-log-views.js';
import { fetchJSON } from './api.js';
import { renderMarkdownInto } from './chat-markdown.js';

/** @typedef {import('./activity-log-chrome.js').SessionRow} SessionRow */
/** @typedef {import('./activity-log-link.js').View} View */
/** @typedef {import('./activity-log-model.js').ActivityPayload} ActivityPayload */
/** @typedef {import('./activity-log-model.js').ActivitySummary} ActivitySummary */
/** @typedef {import('./activity-log-rows.js').UiState} UiState */

/** How often a session is read again. */
const POLL_MS = 10_000;

/** How long typing rests before the search runs. */
const SEARCH_REST_MS = 120;

/** The view switcher, in tab order. @type {[View, string][]} */
const TABS = [
  ['timeline', 'Timeline'],
  ['tools', 'Tool calls'],
  ['safety', 'Safety'],
  ['artifacts', 'Artifacts'],
];

/** @type {Record<View, (payload: ActivityPayload, model: ActivitySummary, ui: UiState) => string>} */
const RENDERERS = {
  timeline: renderTimeline,
  tools: renderTools,
  safety: renderSafety,
  artifacts: renderArtifacts,
};

/** What a click acts on: the nearest element carrying one of these. */
const PRESSABLE = [
  'view',
  'session',
  'full',
  'open-artifact',
  'jump',
  'go',
  'act',
  'filter',
  'scope',
  'sort',
  'toggle',
  'collapse',
]
  .map((name) => `[data-${name}]`)
  .join(',');

/** Rows and headers that are not buttons and take Enter and Space themselves. */
const KEYED = '[tabindex][data-toggle],[tabindex][data-sort],[tabindex][data-jump]';

/** The attributes a control is found by again after the page is drawn anew. */
const FOCUS_NAMES = [
  'data-toggle',
  'data-collapse',
  'data-sort',
  'data-jump',
  'data-view',
  'data-filter',
  'data-scope',
  'data-act',
  'data-go',
  'data-full',
  'data-open-artifact',
];

/**
 * @param {string} id
 * @returns {HTMLElement} the part of the page's shell with that id
 */
function part(id) {
  const found = document.getElementById(id);
  if (!found) throw new Error(`The Activity page has no #${id}`);
  return found;
}

/**
 * Take over the page's shell and show a session in it.
 * @param {Object} [options]
 * @param {string | null} [options.sessionKey]  the session the page was opened on
 * @param {boolean} [options.sidebarOpen]  whether the session list starts shown
 * @returns {{ ready: Promise<void>, load: () => Promise<void>,
 *   follow: (key: string | null) => void, stop: () => void }}
 *   `ready` settles once the first read is drawn; `follow` moves the page to
 *   another session key and drops a picked transcript
 */
export function mountActivityPage({ sessionKey = null, sidebarOpen = true } = {}) {
  const views = part('views');
  const live = part('live');
  const chipEl = part('record-chip');
  const search = /** @type {HTMLInputElement} */ (part('q'));
  const sidebar = part('sb');
  const sidebarToggle = part('sb-toggle');
  const main = part('main');
  const inner = part('inner');

  const hash = readHash(window.location.hash);
  const ui = newUiState(hash.view ?? 'timeline');
  let key = sessionKey;
  /** The transcript picked by hand or by link; it wins over `key`. */
  let selected = hash.id;
  /** @type {ActivityPayload | null} */
  let payload = null;
  /** @type {ActivitySummary | null} */
  let model = null;
  let signature = '';
  /** Whether the last answer the record gave calls for reading again. */
  let watching = false;
  /** The last answer the record gave. @type {ActivityPayload | null} */
  let good = null;
  /** The read `good` answered. */
  let goodFor = '';
  /** Who gave no answer to the last read; `''` when it was answered. @type {'' | 'terminal' | 'record'} */
  let lost = '';
  /** Whether the page shows `good`, kept through a read that got no answer. */
  let stale = false;
  let turn = 0;
  let stopped = false;
  /** @type {ReturnType<typeof setTimeout> | undefined} */
  let timer;
  /** @type {SessionRow[] | null} `null` while the list could not be read */
  let rows = [];
  /** @type {Promise<void> | null} */
  let listing = null;
  /** The session the list was last read again for, so that happens once. */
  let listedFor = '';
  /** @type {(() => void)[]} */
  const disposers = [];

  /**
   * @param {EventTarget} target
   * @param {string} type
   * @param {(event: Event) => void} handler
   */
  function on(target, type, handler) {
    target.addEventListener(type, handler);
    disposers.push(() => target.removeEventListener(type, handler));
  }

  /** @returns {View} */
  const currentView = () => VIEWS.find((view) => view === ui.view) ?? 'timeline';

  /** @returns {string | null} the transcript the session list marks */
  const shownId = () => selected ?? payload?.session?.id ?? null;

  function syncHash() {
    window.history.replaceState(window.history.state, '', writeHash(selected, currentView()));
  }

  // ---- Drawing ----

  /** @param {ActivityPayload} shown */
  function drawToolbar(shown) {
    const counts = viewCounts(shown, model ?? summarize(shown));
    /** @type {Partial<Record<View, number>>} */
    const byView = counts;
    views.innerHTML = TABS.map(([view, label]) => {
      const n = byView[view];
      const count = n ? `<span class="osprey-count">${n}</span>` : '';
      return (
        `<button type="button" data-view="${view}" aria-pressed="${ui.view === view}">` +
        `${label}${count}</button>`
      );
    }).join('');
    live.innerHTML = shown.record === 'unavailable' ? '' : renderLiveState(shown.session);
    const chip = recordChip(shown.record);
    chipEl.className = `osprey-chip osprey-chip--${chip.tone}`;
    chipEl.innerHTML = `${iconHtml(chip.icon)}<span class="osprey-chip-label">${chip.label}</span>`;
  }

  /**
   * @param {ActivityPayload} shown
   * @param {ActivitySummary} counts
   * @returns {string}
   */
  function body(shown, counts) {
    if (shown.record === 'unavailable' && !stale) {
      const again = { label: 'Try again', act: 'refresh' };
      if (lost === 'terminal') {
        return stateBlock(
          'triangle-alert',
          'The session cannot be read',
          'The web terminal did not answer.',
          again
        );
      }
      return stateBlock(
        'database',
        'The agent record cannot be read',
        'The record service did not answer. Nothing is lost: entries are spooled and appear ' +
          'once it is back.',
        again
      );
    }
    if (!shown.session) {
      return stateBlock(
        'activity',
        'No activity in this session yet',
        'Entries appear here as the agent record receives them.'
      );
    }
    return (
      renderHeader(shown, counts) +
      renderStats(shown, counts) +
      recordBanner(shown.record) +
      `<div class="act-view">${RENDERERS[currentView()](shown, counts, ui)}</div>`
    );
  }

  /** @returns {[HTMLElement, string, string] | null} where the keyboard focus is, to find it again */
  function focusMark() {
    const active = document.activeElement;
    if (!(active instanceof HTMLElement)) return null;
    const home = [views, inner].find((region) => region.contains(active));
    const name = FOCUS_NAMES.find((attr) => active.hasAttribute(attr));
    return home && name ? [home, name, active.getAttribute(name) ?? ''] : null;
  }

  function drawSidebar() {
    sidebar.innerHTML = rows
      ? renderSidebar(rows, shownId())
      : stateBlock('database', 'The session list cannot be read');
  }

  function draw() {
    if (!payload || !model) return;
    const mark = focusMark();
    const top = main.scrollTop;
    drawToolbar(payload);
    inner.innerHTML = body(payload, model);
    for (const answer of inner.querySelectorAll('.act-md[data-md]')) {
      if (answer instanceof HTMLElement) renderMarkdownInto(answer, answer.textContent ?? '');
    }
    main.scrollTop = top;
    if (mark) {
      const [home, name, value] = mark;
      const again = [...home.querySelectorAll(`[${name}]`)].find(
        (control) => control.getAttribute(name) === value
      );
      if (again instanceof HTMLElement) again.focus({ preventScroll: true });
    }
    drawSidebar();
    syncHash();
  }

  // ---- Reading ----

  /** @returns {string} */
  function readUrl() {
    if (selected) return `/api/session-activity?transcript_id=${selected}`;
    return key ? `/api/session-activity?session_id=${key}` : '/api/session-activity';
  }

  function listSessions() {
    const reading = (async () => {
      try {
        const answer = await fetchJSON('/api/sessions');
        const listed = Array.isArray(answer?.sessions) ? answer.sessions : [];
        // A row the record holds no transcript for cannot be opened here.
        rows = listed.filter((/** @type {SessionRow} */ row) => Boolean(row?.transcript_id));
      } catch {
        rows = null;
      }
      if (stopped) return;
      drawSidebar();
    })();
    listing = reading;
    return reading;
  }

  /** Read the list again, once, for a session it does not show. */
  async function listShownSession() {
    await listing;
    const id = payload?.session?.id;
    if (stopped || !id || !rows || listedFor === id) return;
    if (rows.some((row) => row.transcript_id === id)) return;
    listedFor = id;
    await listSessions();
  }

  function schedule() {
    clearTimeout(timer);
    if (stopped || document.visibilityState !== 'visible' || !(watching || lost)) return;
    timer = setTimeout(load, POLL_MS);
  }

  async function load() {
    turn += 1;
    const mine = turn;
    clearTimeout(timer);
    const asked = readUrl();
    /** @type {ActivityPayload} */
    let next;
    /** @type {'' | 'terminal' | 'record'} */
    let failed = '';
    try {
      next = await fetchJSON(asked);
      if (next.record === 'unavailable') failed = 'record';
    } catch {
      failed = 'terminal';
      next = {
        session: null,
        nodes: [],
        artifacts: [],
        record: 'unavailable',
        record_reason: 'the web terminal did not answer',
      };
    }
    // An answer for a question the page no longer asks.
    if (stopped || mine !== turn) return;
    lost = failed;
    if (!lost) {
      good = next;
      goodFor = asked;
      // The terminal's own session can go on at any time; a picked one only while live.
      watching = !selected || next.session?.state === 'live';
    }
    stale = Boolean(lost) && goodFor === asked && Boolean(good?.session);
    /** @type {ActivityPayload} */
    const shown =
      stale && good ? { ...good, record: 'unavailable', record_reason: next.record_reason } : next;
    const nextSignature = payloadSignature(shown);
    if (nextSignature !== signature) {
      signature = nextSignature;
      payload = shown;
      model = summarize(shown);
      draw();
    }
    schedule();
    await listShownSession();
  }

  function refresh() {
    void load();
    void listSessions();
  }

  // ---- Presses ----

  /** @param {string} id  a transcript id from the session list */
  function pick(id) {
    selected = id;
    resetForSession(ui);
    search.value = '';
    main.scrollTop = 0;
    drawSidebar();
    syncHash();
    void load();
  }

  /**
   * @param {string} call  the call's key on the page
   * @param {string} recordKey  the key the record holds its result under
   */
  async function loadFull(call, recordKey) {
    try {
      const answer = await fetchJSON(
        `/api/session-activity/result?key=${encodeURIComponent(recordKey)}`
      );
      ui.full.set(call, String(answer?.text ?? ''));
      draw();
    } catch {
      showToast('The full result could not be loaded', { kind: 'error' });
    }
  }

  async function copyLink() {
    const id = payload?.session?.id;
    if (!id) return;
    if (await copyText(shareUrl(id, currentView()))) showToast('Link copied');
    else showToast('The link could not be copied', { kind: 'error' });
  }

  /** @param {string} call  bring this call's row into view on the timeline */
  function land(call) {
    let row = document.getElementById(`row-${call}`);
    if (!row) {
      // The call is inside a folded run of the timeline (a task list).
      for (const fold of inner.querySelectorAll('[data-toggle][data-kind="note"]')) {
        ui.open.add(fold.getAttribute('data-toggle') ?? '');
      }
      draw();
      row = document.getElementById(`row-${call}`);
    }
    if (!row) return;
    row.scrollIntoView({ block: 'center' });
    row.classList.add('act-flash');
    if (row instanceof HTMLButtonElement) row.focus({ preventScroll: true });
  }

  /** @param {HTMLElement} control */
  function press(control) {
    const data = control.dataset;
    if (data.session !== undefined) return pick(data.session);
    if (data.full !== undefined) return void loadFull(data.full, data.key ?? '');
    if (data.openArtifact !== undefined) {
      return void window.open(artifactFileUrl(data.openArtifact), '_blank', 'noopener');
    }
    if (data.act === 'refresh') return refresh();
    if (data.act === 'copy-link') return void copyLink();
    const toggles = [...inner.querySelectorAll('[data-toggle]')].map(
      (row) => row.getAttribute('data-toggle') ?? ''
    );
    const view = ui.view;
    if (!applyAction(ui, data, { nodes: payload?.nodes ?? [], toggles })) return undefined;
    if (ui.view !== view) main.scrollTop = 0;
    draw();
    if (data.jump !== undefined) land(data.jump);
    return undefined;
  }

  /** @param {Event} event */
  function onClick(event) {
    const control = event.target instanceof Element ? event.target.closest(PRESSABLE) : null;
    if (control instanceof HTMLElement) press(control);
  }

  /** @param {Event} event */
  function onKey(event) {
    if (!(event instanceof KeyboardEvent) || (event.key !== 'Enter' && event.key !== ' ')) return;
    // Only a row or a header that holds the focus itself: a button inside one
    // answers its own keys.
    const row = event.target;
    if (!(row instanceof HTMLElement) || !row.matches(KEYED)) return;
    event.preventDefault();
    press(row);
  }

  /** @param {boolean} open */
  function showSidebar(open) {
    sidebar.dataset.open = String(open);
    sidebarToggle.setAttribute('aria-pressed', String(open));
  }

  function onHash() {
    const next = readHash(window.location.hash);
    // The page's own write of the hash says nothing new.
    if (next.id === selected && (next.view ?? ui.view) === ui.view) return undefined;
    if (next.view) ui.view = next.view;
    if (next.id === selected) return draw();
    selected = next.id;
    resetForSession(ui);
    search.value = '';
    draw();
    return void load();
  }

  sidebarToggle.replaceChildren(icon('panel-left'));
  part('search-ic').replaceChildren(icon('search'));
  part('refresh').replaceChildren(icon('refresh'));
  showSidebar(sidebarOpen);

  for (const region of [views, sidebar, inner]) on(region, 'click', onClick);
  on(inner, 'keydown', onKey);
  on(part('refresh'), 'click', refresh);
  on(sidebarToggle, 'click', () => showSidebar(sidebar.dataset.open !== 'true'));
  on(chipEl, 'click', () => {
    if (payload) openInfoPopover(chipEl, recordPopover(payload));
  });
  on(
    search,
    'input',
    debounce(() => {
      ui.q = search.value.trim();
      draw();
    }, SEARCH_REST_MS)
  );
  on(document, 'visibilitychange', () => {
    if (document.visibilityState === 'visible') void load();
    else clearTimeout(timer);
  });
  on(window, 'hashchange', onHash);

  void listSessions();
  const ready = load();

  return {
    ready,
    load,
    follow(next) {
      key = next;
      selected = null;
      resetForSession(ui);
      search.value = '';
      syncHash();
      void load();
    },
    stop() {
      stopped = true;
      clearTimeout(timer);
      for (const dispose of disposers.splice(0)) dispose();
    },
  };
}
