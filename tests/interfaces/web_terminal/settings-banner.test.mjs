// @ts-check
/**
 * Unit tests for the post-save notice on the Agent Settings panel
 * (settings.js's `applySettings` tail):
 *   npx vitest run tests/interfaces/web_terminal/settings-banner.test.mjs
 *
 * A successful config write can come back carrying a `detail`. Today exactly
 * one thing puts it there: a read-only render zone, where the panel's write to
 * `config.yml` lands but the derived `.claude/` artifacts are root-owned and
 * only re-render when the container restarts. The server reports that instead
 * of letting `regenerated: []` be read as "nothing needed re-rendering" — and
 * this is the half that puts it in front of the operator.
 *
 * What is pinned here:
 *
 * - a 200 carrying `detail` renders the notice, with the server's own words in
 *   it (the panel never composes its own copy — one wording, server-side);
 * - a plain 200 renders nothing, so the ordinary save stays a silent save;
 * - the notice is the settings-warning modal surface, not a second dialog
 *   idiom: same `.settings-warning-overlay` / `-dialog` classes, so it inherits
 *   the CSS that already exists and there is one modal to style;
 * - dismissing it removes it, and it never writes the warning gate's
 *   per-session acknowledgment key — a save must not silence the next
 *   drawer-open warning.
 *
 * Driven through settings.js rather than against the notice module directly:
 * what is worth pinning is the WIRING — which responses reach the notice — and
 * settings-notice.js is reached transitively.
 *
 * Seams: terminal.js is mocked (the success tail reconnects a live PTY, which
 * is out of scope here — see terminal-resume.test.mjs); `fetch` is stubbed the
 * way settings.test.mjs stubs it. `applySettings` is not exported, so it is
 * driven through the confirm button it is wired to, exactly as settings.test.mjs
 * does.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

/** Stand-in for terminal.js — the post-save tail must not touch a real PTY. */
const term = vi.hoisted(() => ({
  restarts: 0,
  starts: 0,
}));

vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/terminal.js', () => ({
  restartTerminal: () => {
    term.restarts += 1;
    return Promise.resolve();
  },
  startTerminal: () => {
    term.starts += 1;
  },
}));

import {
  initSettings,
} from '../../../src/osprey/interfaces/web_terminal/static/js/settings.js';

const RESTART_DETAIL = 'derived artifacts re-render on container restart';
const ACK_KEY = 'osprey-settings-warning-ack';

/** Mount the minimal DOM `initSettings()` and `applySettings()` need. */
function mountFixture() {
  document.body.innerHTML = `
    <div id="settings-drawer"></div>
    <div id="tab-config">
      <button class="settings-mode-btn" data-mode="raw"></button>
      <button class="settings-mode-btn" data-mode="form"></button>
      <button class="settings-apply-btn"></button>
      <button class="settings-confirm-btn"></button>
      <button class="settings-cancel-btn"></button>
      <div class="settings-status"></div>
    </div>
    <div id="settings-form"></div>
    <textarea id="settings-raw-editor">model: anthropic/claude-sonnet</textarea>
  `;
  // `applySettings` closes the drawer on its way to the restart tail, and
  // `osprey-drawer`'s imperative close() is not defined on a plain div — without
  // this stub the success path throws into the catch and never reaches the tail,
  // which would make "the ordinary save is unchanged" untestable.
  Object.assign(/** @type {HTMLElement} */ (document.getElementById('settings-drawer')), {
    close: () => {},
  });
}

/**
 * An ok JSON response carrying `body`, so `applySettings` runs its success tail.
 * @param {object} body
 */
function okResponse(body) {
  return { ok: true, status: 200, json: () => Promise.resolve(body) };
}

/** Drive the raw-mode PUT through the confirm button and let it settle. */
async function saveRaw() {
  /** @type {HTMLElement} */ (
    document.querySelector('.settings-mode-btn[data-mode="raw"]')
  ).click();
  /** @type {HTMLElement} */ (document.querySelector('.settings-confirm-btn')).click();
  await new Promise((resolve) => setTimeout(resolve, 0));
}

function overlay() {
  return document.querySelector('.settings-warning-overlay');
}

beforeEach(() => {
  term.restarts = 0;
  term.starts = 0;
  localStorage.clear();
  mountFixture();
  initSettings();
});

afterEach(() => {
  vi.unstubAllGlobals();
  localStorage.clear();
});

describe('applySettings: the read-only render zone notice', () => {
  test('renders the notice when a successful save carries a detail', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(okResponse({
      status: 'ok',
      requires_restart: true,
      regenerated: [],
      detail: RESTART_DETAIL,
    }))));

    await saveRaw();

    const node = overlay();
    expect(node).not.toBeNull();
    // The server's own words, verbatim — the panel composes no copy of its own.
    expect(/** @type {HTMLElement} */ (node).textContent).toContain(RESTART_DETAIL);
  });

  test('renders nothing on an ordinary successful save', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(okResponse({
      status: 'ok',
      requires_restart: true,
      regenerated: ['settings.json'],
    }))));

    await saveRaw();

    expect(overlay()).toBeNull();
    // The ordinary save still runs its tail, so a missing notice is the only
    // difference between the two paths.
    expect(term.restarts).toBe(1);
  });

  test('reuses the settings-warning modal surface rather than a second idiom', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(okResponse({
      status: 'ok',
      requires_restart: true,
      regenerated: [],
      detail: RESTART_DETAIL,
    }))));

    await saveRaw();

    const node = /** @type {HTMLElement} */ (overlay());
    expect(node.querySelector('.settings-warning-dialog')).not.toBeNull();
    expect(node.querySelector('.settings-warning-title')).not.toBeNull();
    // One acknowledging button: the operator has no decision to make here, and
    // offering a Cancel would imply the write could still be taken back.
    expect(node.querySelectorAll('.settings-warning-actions button')).toHaveLength(1);
  });

  test('dismissal removes it and never acknowledges the drawer warning gate', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(okResponse({
      status: 'ok',
      requires_restart: true,
      regenerated: [],
      detail: RESTART_DETAIL,
    }))));

    await saveRaw();
    /** @type {HTMLElement} */ (
      document.querySelector('.settings-warning-actions button')
    ).click();
    // The overlay fades out; its fallback removal fires after the transition.
    await new Promise((resolve) => setTimeout(resolve, 350));

    expect(overlay()).toBeNull();
    // Saving is not acknowledging: the next drawer open must still warn.
    expect(localStorage.getItem(ACK_KEY)).toBeNull();
  });
});
