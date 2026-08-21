/**
 * OSPREY Web Terminal — feedback boot wiring (feedback-boot.js).
 *
 * This is the integration seam: the rail button, the dialog, the transport and
 * the deployment's configuration only ever meet here, so the properties pinned
 * below are the ones no single-module suite can see.
 *
 *   1. ORDER. For GitHub and Email the clipboard write and the browser handoff
 *      must both be issued inside the click turn, before the POST settles.
 *      Asserted with ZERO awaits after the click — one microtask is already
 *      too late for Safari's transient user activation.
 *   2. The two documentation affordances (rail anchor, status-bar link) read
 *      the SAME `docs_url`, and a deployment that blanks it gets neither.
 *   3. Scrollback is captured in the click turn and travels only when the
 *      session-context box is ticked.
 *
 * The DOM fixture is sliced out of the shipped index.html rather than retyped,
 * so a template change cannot leave these tests describing markup that no
 * longer exists.
 *
 *   npx vitest run tests/interfaces/web_terminal/feedback-boot.test.mjs
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { initFeedback } from '../../../src/osprey/interfaces/web_terminal/static/js/feedback-boot.js';
import { qs } from '../_support/dom.mjs';

// `import.meta.dirname` is a plain string, so it sidesteps happy-dom's
// override of the global URL (which breaks fileURLToPath(new URL(...)) here).
const STATIC_DIR = join(
  import.meta.dirname,
  '../../../src/osprey/interfaces/web_terminal/static'
);
const indexHtml = readFileSync(join(STATIC_DIR, 'index.html'), 'utf8');

const DOCS_URL = 'https://docs.example.org/osprey';
const REPO = 'acme/osprey';
const EMAIL = 'maintainers@example.org';
const SESSION = '11111111-2222-3333-4444-555555555555';

/**
 * The `<div class="panel-rail-region">…</div>` subtree, verbatim from the
 * shipped template, by counting <div> depth from its opening tag.
 *
 * @returns {string}
 */
function railRegionMarkup() {
  const open = '<div class="panel-rail-region">';
  const start = indexHtml.indexOf(open);
  expect(start, 'index.html has no .panel-rail-region').toBeGreaterThan(-1);
  const tags = /<div\b[^>]*>|<\/div\s*>/g;
  tags.lastIndex = start;
  let depth = 0;
  /** @type {RegExpExecArray|null} */
  let match = null;
  while ((match = tags.exec(indexHtml)) !== null) {
    depth += match[0].startsWith('</') ? -1 : 1;
    if (depth === 0) return indexHtml.slice(start, tags.lastIndex);
  }
  throw new Error('unbalanced .panel-rail-region in index.html');
}

/**
 * The `<footer class="status-bar">…</footer>` element, verbatim. The status bar
 * nests no further footer, so the first closing tag is the matching one.
 *
 * @returns {string}
 */
function statusBarMarkup() {
  const open = '<footer class="status-bar">';
  const start = indexHtml.indexOf(open);
  expect(start, 'index.html has no .status-bar footer').toBeGreaterThan(-1);
  const end = indexHtml.indexOf('</footer>', start);
  expect(end, 'unterminated .status-bar footer').toBeGreaterThan(-1);
  return indexHtml.slice(start, end + '</footer>'.length);
}

/** A `Response` slice with a JSON body, as the transport reads it. */
/** @param {Record<string, unknown>} body @param {number} [status] */
function jsonResponse(body, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    text: () => Promise.resolve(JSON.stringify(body)),
  };
}

/**
 * A fake xterm terminal over a list of physical rows — the duck-type
 * `captureScrollback` walks (see scrollback-capture.test.mjs).
 *
 * @param {string[]} rows
 */
function fakeTerm(rows) {
  return {
    buffer: {
      active: {
        length: rows.length,
        /** @param {number} i */
        getLine(i) {
          const text = rows[i];
          if (text === undefined) return undefined;
          return {
            isWrapped: false,
            translateToString() {
              return text;
            },
          };
        },
      },
    },
  };
}

/**
 * Boot the feature over a fresh fixture with every ambient dependency spied.
 *
 * @param {object} [over]
 * @param {any} [over.panels] - the `/api/panels` body (null to fail the read)
 * @param {any} [over.health] - the `/health` body (null to fail the read)
 * @param {any} [over.response] - what the POST resolves with
 * @param {string|null} [over.sessionId]
 * @param {any} [over.terminal]
 * @param {any} [over.clipboard]
 * @param {boolean} [over.settle] - false leaves the config reads hanging
 * @param {boolean} [over.hangHealth] - hang `/health` only
 */
async function boot(over = {}) {
  document.body.innerHTML = railRegionMarkup() + statusBarMarkup();

  const panels =
    over.panels === undefined
      ? { docs_url: DOCS_URL, feedback_github_repo: REPO, feedback_email: EMAIL }
      : over.panels;
  const health = over.health === undefined ? { version: '9.9.9' } : over.health;

  const loadJSON = vi.fn(
    /** @param {string} path */ (path) => {
      if (over.settle === false) return new Promise(() => {});
      if (over.hangHealth === true && path === '/health') return new Promise(() => {});
      const body = path === '/health' ? health : panels;
      return body === null ? Promise.reject(new Error('HTTP 503')) : Promise.resolve(body);
    }
  );
  const fetchSpy = vi.fn(() =>
    Promise.resolve(over.response ?? jsonResponse({ id: 'fb-1', payload: 'PAYLOAD' }))
  );
  const clipboard =
    over.clipboard === undefined
      ? { write: vi.fn(() => Promise.resolve()), writeText: vi.fn(() => Promise.resolve()) }
      : over.clipboard;
  const windowOpen = vi.fn();
  const navigate = vi.fn();

  const { modal, ready } = initFeedback({
    loadJSON,
    fetch: fetchSpy,
    clipboard,
    windowOpen,
    navigate,
    getSessionId: () => (over.sessionId === undefined ? SESSION : over.sessionId),
    getTerminal: () => over.terminal ?? null,
  });
  // The boot is synchronous by contract; every test that cares about the
  // deployment's configuration waits for it here instead. `ready` covers both
  // reads, so a test that deliberately hangs one waits on its own signal.
  if (over.settle !== false && over.hangHealth !== true) await ready;
  return { modal, ready, loadJSON, fetch: fetchSpy, clipboard, windowOpen, navigate };
}

/** Open the dialog by clicking the rail's Feedback button. */
function clickFeedbackButton() {
  qs(document, '#panel-feedback-btn').click();
}

/** The parsed body of the Nth POST. @param {any} fetchSpy @param {number} [n] */
function postedBody(fetchSpy, n = 0) {
  return JSON.parse(fetchSpy.mock.calls[n][1].body);
}

/** Type into the dialog's textarea. @param {string} value */
function typeReport(value) {
  const area = qs(document, '.feedback-modal-body .feedback-text', HTMLTextAreaElement);
  area.value = value;
  area.dispatchEvent(new Event('input', { bubbles: true }));
}

/** Select a delivery channel. @param {string} channel */
function pickChannel(channel) {
  const radio = qs(document, `input[name="feedback-channel"][value="${channel}"]`, HTMLInputElement);
  radio.checked = true;
  radio.dispatchEvent(new Event('change', { bubbles: true }));
}

/** Tick the session-context checkbox. */
function tickContext() {
  const box = qs(document, '.feedback-context-check', HTMLInputElement);
  box.checked = true;
  box.dispatchEvent(new Event('change', { bubbles: true }));
}

/** The dialog's notice line, or null while none has been created. */
function noticeText() {
  const line = document.querySelector('.feedback-notice');
  return line === null ? null : line.textContent;
}

// `any`, deliberately: this holds the whole spied boot environment, and typing
// it would mean restating every vitest spy shape for no assertion's benefit.
/** @type {any} */
let booted = null;

beforeEach(() => {
  booted = null;
  document.body.innerHTML = '';
});

afterEach(() => {
  // The modal registers itself in module state while open, and its Escape
  // handler lives on `document` — leaving one open would leak into the next
  // test and into the palette's arbitration.
  if (booted !== null) booted.modal.close();
  vi.unstubAllGlobals();
  document.body.innerHTML = '';
});

describe('documentation affordances', () => {
  test('rail anchor and status-bar link both take the configured docs_url', async () => {
    booted = await boot();

    expect(qs(document, '#panel-docs-link', HTMLAnchorElement).getAttribute('href')).toBe(DOCS_URL);
    expect(qs(document, '#panel-docs-link').hidden).toBe(false);
    expect(qs(document, '#docs-link', HTMLAnchorElement).getAttribute('href')).toBe(DOCS_URL);
  });

  test('a blank docs_url leaves both links hidden and href-less', async () => {
    booted = await boot({ panels: { docs_url: '   ', feedback_github_repo: REPO } });

    for (const id of ['#panel-docs-link', '#docs-link']) {
      expect(qs(document, id).hidden, `${id} still visible`).toBe(true);
      expect(qs(document, id).hasAttribute('href'), `${id} still linked`).toBe(false);
    }
  });

  test('both links ship hidden, so neither can flash before boot', () => {
    // The markup, not the runtime: a link that starts visible with a
    // hardcoded href is exactly the dead link an air-gapped deployment must
    // never be shown.
    expect(railRegionMarkup()).toMatch(/id="panel-docs-link"[^>]*/);
    expect(indexHtml).toMatch(/id="docs-link"[\s\S]{0,120}?hidden>/);
    expect(indexHtml).not.toMatch(/id="docs-link"[^>]*href=/);
  });

  test('.status-item[hidden] is declared, or hiding the link does nothing', () => {
    // .status-item's own display:flex outranks the user-agent [hidden] rule.
    const css = readFileSync(join(STATIC_DIR, 'css/files.css'), 'utf8');

    expect(css).toMatch(/\.status-item\[hidden\]\s*\{[^}]*display:\s*none/);
  });

  test('a failed config read hides the links without erroring', async () => {
    booted = await boot({ panels: null, health: null });

    expect(qs(document, '#panel-docs-link').hidden).toBe(true);
    expect(qs(document, '#docs-link').hidden).toBe(true);
    // The dialog is still wired: a dead config read must not cost the operator
    // the Local channel, which needs no configuration at all.
    clickFeedbackButton();
    expect(document.querySelector('.feedback-modal-dialog')).not.toBeNull();
  });

  test('the status-bar link keeps a relative docs_url relative', async () => {
    booted = await boot({ panels: { docs_url: '/docs/' } });

    expect(qs(document, '#docs-link', HTMLAnchorElement).getAttribute('href')).toBe('/docs/');
  });
});

describe('terminal seam', () => {
  test('terminal.js hands out the live xterm instance', async () => {
    // A real import, not a source scan: this accessor exists only for the
    // scrollback tier, so nothing else would notice if it were renamed away.
    const terminal = await import(
      '../../../src/osprey/interfaces/web_terminal/static/js/terminal.js'
    );

    expect(typeof terminal.getTerminalInstance).toBe('function');
    // No initTerminal() here — the null answer is the one the capture handles.
    expect(terminal.getTerminalInstance()).toBeNull();
  });

  test('app.js passes that accessor to the feedback boot', () => {
    const appJs = readFileSync(join(STATIC_DIR, 'js/app.js'), 'utf8');

    // The wiring itself is unreachable from a test (it runs on a
    // DOMContentLoaded that fired long before), and an unwired `getTerminal`
    // costs the scrollback silently — the send still succeeds, just emptier.
    expect(appJs).toMatch(/getTerminal:\s*getTerminalInstance/);
  });
});

describe('notice styling', () => {
  test('every notice kind is coloured, or a failure reads as a muted hint', () => {
    const css = readFileSync(join(STATIC_DIR, 'css/feedback-modal.css'), 'utf8');

    for (const [kind, token] of [
      ['success', '--color-success'],
      ['warning', '--color-warning'],
      ['error', '--color-error'],
    ]) {
      const rule = new RegExp(
        `\\.feedback-notice\\[data-kind="${kind}"\\][^}]*var\\(${token}\\)`
      );
      expect(css, `no ${kind} colour for .feedback-notice`).toMatch(rule);
    }
  });

  test('index.html links the dialog stylesheet, or the overlay never appears', () => {
    expect(indexHtml).toContain("prefixed('/static/css/feedback-modal.css')");
  });
});

describe('rail button', () => {
  test('clicking Feedback opens the dialog', async () => {
    booted = await boot();
    expect(document.querySelector('.feedback-modal-dialog')).toBeNull();

    clickFeedbackButton();

    expect(booted.modal.isOpen()).toBe(true);
    expect(document.querySelector('.feedback-modal-dialog')).not.toBeNull();
  });

  test('a hung config read still leaves the button live', async () => {
    // A hang is not a rejection: nothing would ever settle here. Registering
    // the click handler after the reads would leave the operator with a dead
    // button exactly when they most want to report something.
    booted = await boot({ settle: false });

    clickFeedbackButton();

    expect(booted.modal.isOpen()).toBe(true);
    expect(qs(document, '.feedback-send')).not.toBeNull();
  });

  test('the Local channel works before the config lands', async () => {
    booted = await boot({ settle: false });
    clickFeedbackButton();
    typeReport('cannot wait for config');

    qs(document, '.feedback-send').click();
    await vi.waitFor(() => expect(booted.fetch).toHaveBeenCalled());

    expect(postedBody(booted.fetch).text).toBe('cannot wait for config');
  });

  test('an outbound channel says the config is still loading, and opens nothing', async () => {
    booted = await boot({ settle: false });
    clickFeedbackButton();
    pickChannel('github');

    qs(document, '.feedback-open-channel').click();
    await vi.waitFor(() => expect(noticeText()).not.toBe(''));

    expect(noticeText()).toContain('Still reading');
    expect(booted.windowOpen).not.toHaveBeenCalled();
  });

  test('a hung /health does not withhold the outbound channels', async () => {
    // The two reads answer different questions and are applied separately.
    // /health is the read behind the auth gate, so tying the GitHub channel to
    // it would strand the operator on "still reading" over a version line.
    booted = await boot({ hangHealth: true });
    await vi.waitFor(() =>
      expect(qs(document, '#docs-link', HTMLAnchorElement).getAttribute('href')).toBe(DOCS_URL)
    );
    booted.fetch.mockImplementation(() => new Promise(() => {}));
    clickFeedbackButton();
    typeReport('a report');
    pickChannel('github');

    qs(document, '.feedback-open-channel').click();

    expect(booted.windowOpen).toHaveBeenCalledTimes(1);
    // The version line is simply missing, which is what a dropped metadata
    // value is supposed to look like.
    expect(decodeURIComponent(booted.windowOpen.mock.calls[0][0])).not.toContain('9.9.9');
  });

  test('a throwing config payload leaves `ready` resolved, not rejected', async () => {
    // "ready never rejects" is a documented contract, and the appliers touch
    // the DOM — so it has to hold by construction, not by luck. The console
    // line is the other half: swallowing this silently would hide a real bug.
    const logged = vi.spyOn(console, 'error').mockImplementation(() => {});
    try {
      booted = await boot({
        panels: {
          get docs_url() {
            throw new Error('boom');
          },
        },
      });

      await expect(booted.ready).resolves.toBeUndefined();
      expect(logged).toHaveBeenCalledWith('Feedback config failed to apply:', expect.any(Error));
    } finally {
      logged.mockRestore();
    }
  });

  test('the config reads carry an abort signal, or a hang never settles', async () => {
    booted = await boot();

    for (const call of booted.loadJSON.mock.calls) {
      const init = call[1];
      expect(init, 'config read got no init').toBeDefined();
      // happy-dom/Node both ship AbortSignal.timeout; the module falls back to
      // `undefined` only where the runtime has none.
      expect(init.signal).toBeInstanceOf(AbortSignal);
    }
  });

  test('the dialog reads the live session id', async () => {
    booted = await boot();
    clickFeedbackButton();

    expect(booted.modal.formState().sessionId).toBe(SESSION);
    expect(qs(document, '.feedback-context-check', HTMLInputElement).disabled).toBe(false);
  });

  test('no session greys the context checkbox out', async () => {
    booted = await boot({ sessionId: null });
    clickFeedbackButton();

    expect(qs(document, '.feedback-context-check', HTMLInputElement).disabled).toBe(true);
  });
});

describe('local send', () => {
  test('Send posts the form and shows the record id', async () => {
    booted = await boot();
    clickFeedbackButton();
    typeReport('the panel went blank');

    qs(document, '.feedback-send').click();
    await vi.waitFor(() => expect(noticeText()).toContain('fb-1'));

    expect(booted.fetch).toHaveBeenCalledTimes(1);
    expect(booted.fetch.mock.calls[0][0]).toBe('/api/feedback');
    expect(postedBody(booted.fetch)).toMatchObject({
      text: 'the panel went blank',
      channel: 'local',
      include_metadata: true,
      include_context: false,
    });
  });

  test('the notice line is a live region created before the request goes out', async () => {
    booted = await boot();
    clickFeedbackButton();

    qs(document, '.feedback-send').click();

    // Synchronously, with the POST still in flight.
    expect(qs(document, '.feedback-notice').getAttribute('role')).toBe('status');
    expect(noticeText()).toBe('');
    await vi.waitFor(() => expect(noticeText()).not.toBe(''));
    expect(qs(document, '.feedback-notice').dataset.kind).toBe('success');
  });

  test('a server error becomes an error notice, not a throw', async () => {
    booted = await boot({ response: jsonResponse({ detail: 'store is unwritable' }, 500) });
    clickFeedbackButton();

    qs(document, '.feedback-send').click();
    await vi.waitFor(() => expect(noticeText()).toBe('store is unwritable'));

    expect(qs(document, '.feedback-notice').dataset.kind).toBe('error');
  });
});

describe('session context', () => {
  test('a ticked context box sends the session id and the scrollback', async () => {
    booted = await boot({ terminal: fakeTerm(['first line', 'second line']) });
    clickFeedbackButton();
    typeReport('see the log');
    tickContext();

    qs(document, '.feedback-send').click();
    await vi.waitFor(() => expect(booted.fetch).toHaveBeenCalled());

    expect(postedBody(booted.fetch)).toMatchObject({
      include_context: true,
      session_id: SESSION,
      scrollback: 'first line\nsecond line',
    });
  });

  test('an unticked context box sends neither', async () => {
    booted = await boot({ terminal: fakeTerm(['first line']) });
    clickFeedbackButton();

    qs(document, '.feedback-send').click();
    await vi.waitFor(() => expect(booted.fetch).toHaveBeenCalled());

    const body = postedBody(booted.fetch);
    expect(body.include_context).toBe(false);
    expect('session_id' in body).toBe(false);
    expect('scrollback' in body).toBe(false);
  });

  test('a deployment with no terminal to read still submits', async () => {
    booted = await boot({ terminal: null });
    clickFeedbackButton();
    tickContext();

    qs(document, '.feedback-send').click();
    await vi.waitFor(() => expect(booted.fetch).toHaveBeenCalled());

    expect(postedBody(booted.fetch).scrollback).toBe('');
  });

  test('Copy session context uses the record-less bundle endpoint', async () => {
    booted = await boot({
      response: jsonResponse({ bundle: 'BUNDLE' }),
      terminal: fakeTerm(['tail']),
    });
    clickFeedbackButton();
    typeReport('nötes');
    pickChannel('github');
    tickContext();

    qs(document, '.feedback-copy-context').click();
    await vi.waitFor(() => expect(booted.fetch).toHaveBeenCalled());

    expect(booted.fetch.mock.calls[0][0]).toBe('/api/feedback/bundle');
    expect(postedBody(booted.fetch)).toMatchObject({
      session_id: SESSION,
      // UTF-8 bytes, not UTF-16 units: "nötes" is 5 characters and 6 bytes.
      text_len: 6,
      scrollback: 'tail',
    });
  });
});

describe('outbound channels', () => {
  test('the clipboard write and the issue tab are issued in the click turn', async () => {
    // A POST that never settles: everything asserted below therefore happened
    // before any response could have arrived.
    booted = await boot({ panels: { feedback_github_repo: REPO } });
    booted.fetch.mockImplementation(() => new Promise(() => {}));
    vi.stubGlobal(
      'ClipboardItem',
      class {
        /** @param {Record<string, unknown>} items */
        constructor(items) {
          this.items = items;
        }
      }
    );
    clickFeedbackButton();
    typeReport('crash on resize');
    pickChannel('github');

    qs(document, '.feedback-open-channel').click();

    // No `await` above this line — one microtask is already too late.
    expect(booted.clipboard.write).toHaveBeenCalledTimes(1);
    expect(booted.windowOpen).toHaveBeenCalledTimes(1);
    const [url, target, features] = booted.windowOpen.mock.calls[0];
    expect(url).toContain(`https://github.com/${REPO}/issues/new?`);
    expect(url).toContain('crash%20on%20resize');
    expect(target).toBe('_blank');
    expect(features).toBe('noopener');
  });

  test('the email channel hands the draft to the browser, still in the turn', async () => {
    booted = await boot({ panels: { feedback_email: EMAIL } });
    booted.fetch.mockImplementation(() => new Promise(() => {}));
    clickFeedbackButton();
    typeReport('cannot log in');
    pickChannel('email');

    qs(document, '.feedback-open-channel').click();

    expect(booted.navigate).toHaveBeenCalledTimes(1);
    expect(booted.navigate.mock.calls[0][0]).toMatch(/^mailto:maintainers@example\.org\?/);
  });

  test('the prefilled body carries the deployment metadata block', async () => {
    booted = await boot();
    booted.fetch.mockImplementation(() => new Promise(() => {}));
    clickFeedbackButton();
    typeReport('a report');
    pickChannel('github');

    qs(document, '.feedback-open-channel').click();

    const body = decodeURIComponent(booted.windowOpen.mock.calls[0][0]);
    expect(body).toContain('Deployment metadata');
    expect(body).toContain('9.9.9');
  });

  test('a channel this deployment never configured refuses instead of 404ing', async () => {
    booted = await boot({ panels: { docs_url: DOCS_URL, feedback_github_repo: '' } });
    clickFeedbackButton();
    pickChannel('github');

    qs(document, '.feedback-open-channel').click();

    // A microtask later, not this turn: the live region has just been created,
    // and a region written in the turn it is created is never announced.
    expect(noticeText()).toBe('');
    await vi.waitFor(() => expect(noticeText()).toContain('no feedback repository configured'));
    expect(booted.windowOpen).not.toHaveBeenCalled();
    expect(booted.fetch).not.toHaveBeenCalled();
  });

  test('an unreadable config says so, rather than asserting a blank posture', async () => {
    booted = await boot({ panels: null });
    clickFeedbackButton();
    pickChannel('email');

    qs(document, '.feedback-open-channel').click();
    await vi.waitFor(() => expect(noticeText()).not.toBe(''));

    // "no address is configured" would be a claim about the deployment that a
    // failed read cannot support.
    expect(noticeText()).toContain('Could not read');
    expect(booted.navigate).not.toHaveBeenCalled();
  });

  test('one action at a time — a double click sends once', async () => {
    booted = await boot();
    booted.fetch.mockImplementation(() => new Promise(() => {}));
    clickFeedbackButton();
    typeReport('a report');

    const send = qs(document, '.feedback-send', HTMLButtonElement);
    send.click();
    send.click();

    expect(booted.fetch).toHaveBeenCalledTimes(1);
    expect(send.disabled).toBe(true);
  });

  test('switching channel mid-flight does not buy a second submission', async () => {
    // The action row is rebuilt on every channel switch, with fresh enabled
    // buttons — so the greyed-out row cannot be the guard, only its face.
    booted = await boot();
    booted.fetch.mockImplementation(() => new Promise(() => {}));
    clickFeedbackButton();
    typeReport('a report');

    qs(document, '.feedback-send').click();
    pickChannel('github');
    qs(document, '.feedback-open-channel').click();

    expect(booted.fetch).toHaveBeenCalledTimes(1);
    expect(booted.windowOpen).not.toHaveBeenCalled();
  });

  test('the action row comes back once the action settles', async () => {
    booted = await boot();
    clickFeedbackButton();

    qs(document, '.feedback-send').click();
    await vi.waitFor(() =>
      expect(qs(document, '.feedback-send', HTMLButtonElement).disabled).toBe(false)
    );

    qs(document, '.feedback-send').click();
    expect(booted.fetch).toHaveBeenCalledTimes(2);
  });

  test('the lock never re-enables a button the dialog means to withhold', async () => {
    // Copy session context is disabled while the context box is unticked.
    booted = await boot();
    clickFeedbackButton();
    pickChannel('github');

    qs(document, '.feedback-open-channel').click();
    await vi.waitFor(() =>
      expect(qs(document, '.feedback-open-channel', HTMLButtonElement).disabled).toBe(false)
    );

    expect(qs(document, '.feedback-copy-context', HTMLButtonElement).disabled).toBe(true);
  });

  test('a refused clipboard falls back to a selectable payload', async () => {
    booted = await boot({ clipboard: {} });
    clickFeedbackButton();
    typeReport('a report');
    pickChannel('github');

    qs(document, '.feedback-open-channel').click();
    await vi.waitFor(() =>
      expect(document.querySelector('.feedback-manual-copy')).not.toBeNull()
    );

    const area = qs(document, '.feedback-manual-copy', HTMLTextAreaElement);
    expect(area.value).toBe('PAYLOAD');
    expect(area.readOnly).toBe(true);
    // The handoff still happened — a clipboard that refuses must not cost the
    // operator the issue tab.
    expect(booted.windowOpen).toHaveBeenCalledTimes(1);
  });

  test('a second action clears what the first one left behind', async () => {
    booted = await boot({ clipboard: {} });
    clickFeedbackButton();
    typeReport('a report');
    pickChannel('github');
    qs(document, '.feedback-open-channel').click();
    await vi.waitFor(() =>
      expect(document.querySelector('.feedback-manual-copy')).not.toBeNull()
    );

    pickChannel('local');
    qs(document, '.feedback-send').click();

    expect(document.querySelector('.feedback-manual-copy')).toBeNull();
    expect(noticeText()).toBe('');
  });
});
