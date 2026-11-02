// @ts-check
/**
 * The sidebar's ☰ More menu (more-menu.js):
 *   npx vitest run tests/interfaces/web_terminal/more-menu.test.mjs
 *
 * The menu's rows are server-rendered (index.html's `#more-menu`); this module
 * builds the theme families, keeps every radio's `aria-checked` true to the
 * state it names, opens and closes the menu beside the sidebar through the
 * design system's popover.js, and runs the rows no other module owns.
 *
 * The fixture is index.html's `#more-menu` (signed-in variant, rendered for
 * a user `alice` with a landing URL; Sign out is a sidebar footer row, not a
 * row of this menu), `#shortcuts-popover` and
 * `#about-popover`, copied with the Jinja expressions written out, plus a
 * `#sidebar` holding `#more-btn`.
 *
 * Seams mocked: the theme preference (theme-manager.js, whose real exports are
 * kept except the ones the menu reads and writes), the tour (tour.js) and the activity log
 * (activity-log-link.js). popover.js is the real module. `fetch` is stubbed
 * for `/health`. Module state is isolated by vi.resetModules() + dynamic import
 * per test.
 */

import { test, expect, describe, beforeEach, afterEach, vi } from 'vitest';

import { byId, qs } from '../_support/dom.mjs';

const MODULE = '../../../src/osprey/interfaces/web_terminal/static/js/more-menu.js';

const mocks = vi.hoisted(() => ({
  startTour: vi.fn(),
  openActivityLog: vi.fn(),
}));

/** The theme preference as theme-manager.js would report it. */
const theme = vi.hoisted(() => ({
  /** @type {'auto' | 'light' | 'dark'} */
  preference: 'auto',
  /** @type {string | null} */
  current: 'osprey-dark',
  /** @type {string | null} */
  family: 'osprey',
  families: [
    { id: 'osprey', label: 'OSPREY' },
    { id: 'high-contrast', label: 'High Contrast' },
  ],
  /** @type {(() => void)[]} */
  subscribers: [],
  setTheme: vi.fn(),
  toggleTheme: vi.fn(),
  setFamily: vi.fn(),
}));

vi.mock('/design-system/js/theme-manager.js', async (importOriginal) => ({
  .../** @type {object} */ (await importOriginal()),
  getTheme: () => theme.current,
  getFamily: () => theme.family,
  modePreference: () => theme.preference,
  modeOf: (/** @type {string | null} */ id) =>
    id?.endsWith('-dark') ? 'dark' : id?.endsWith('-light') ? 'light' : null,
  themeFamilies: () => theme.families,
  setTheme: theme.setTheme,
  toggleTheme: theme.toggleTheme,
  setFamily: theme.setFamily,
  subscribe: (/** @type {() => void} */ fn) => {
    theme.subscribers.push(fn);
    return () => {};
  },
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/tour.js', () => ({
  startTour: mocks.startTour,
}));
vi.mock('../../../src/osprey/interfaces/web_terminal/static/js/activity-log-link.js', () => ({
  openActivityLog: mocks.openActivityLog,
}));

/** @type {typeof import('../../../src/osprey/interfaces/web_terminal/static/js/more-menu.js')} */
let mod;
/** @type {ReturnType<typeof vi.fn>} */
let fetchMock;

/** index.html's ☰ More button, menu and the two popovers it opens. */
const FIXTURE = `
  <nav class="sidebar" id="sidebar" aria-label="Panels">
    <div class="sidebar-foot">
      <button class="osprey-nav-row" id="more-btn" type="button" aria-label="More"
              aria-haspopup="menu" aria-expanded="false" aria-controls="more-menu" data-tip="More">
        <svg class="osprey-ic osprey-nav-icon" data-icon="menu" aria-hidden="true"></svg>
        <span class="osprey-nav-label">More</span>
      </button>
    </div>
  </nav>
  <div class="osprey-popover more-menu" id="more-menu" role="menu" aria-label="More" hidden>
    <div class="more-menu-who" role="group" aria-label="Signed in">
      <span class="more-menu-avatar" aria-hidden="true">A</span>
      <span class="more-menu-user">alice</span>
      <span class="more-menu-role" title="Role for this session">operator</span>
      <span class="more-menu-sub">Example Research Facility</span>
    </div>
    <div class="osprey-popover-sep" role="separator"></div>
    <div role="group" aria-labelledby="more-appearance-label">
      <div class="osprey-popover-title" id="more-appearance-label">Appearance</div>
      <button class="osprey-popover-item" type="button" role="menuitemradio" aria-checked="false" data-appearance="auto">Match system<svg class="osprey-ic osprey-popover-check" data-icon="check" aria-hidden="true"></svg></button>
      <button class="osprey-popover-item" type="button" role="menuitemradio" aria-checked="false" data-appearance="light">Light<svg class="osprey-ic osprey-popover-check" data-icon="check" aria-hidden="true"></svg></button>
      <button class="osprey-popover-item" type="button" role="menuitemradio" aria-checked="false" data-appearance="dark">Dark<svg class="osprey-ic osprey-popover-check" data-icon="check" aria-hidden="true"></svg></button>
    </div>
    <div role="group" aria-labelledby="more-family-label">
      <div class="osprey-popover-title" id="more-family-label">Theme family</div>
      <div data-theme-families></div>
    </div>
    <div class="osprey-popover-sep" role="separator"></div>
    <button class="osprey-popover-item" id="more-settings" type="button" role="menuitem"
            data-drawer-trigger="settings-drawer">
      <svg class="osprey-ic" data-icon="sliders" aria-hidden="true"></svg>Settings
      <span class="osprey-popover-aside" data-chord="," aria-hidden="true">&#8984;,</span>
    </button>
    <button class="osprey-popover-item" type="button" role="menuitem" data-more="shortcuts"
            aria-haspopup="dialog" aria-controls="shortcuts-popover">
      <svg class="osprey-ic" data-icon="keyboard" aria-hidden="true"></svg>Keyboard shortcuts
    </button>
    <button class="osprey-popover-item" type="button" role="menuitem" data-more="tour">
      <svg class="osprey-ic" data-icon="compass" aria-hidden="true"></svg>Take the tour
    </button>
    <button class="osprey-popover-item" type="button" role="menuitem" data-more="activity">
      <svg class="osprey-ic" data-icon="activity" aria-hidden="true"></svg>Activity log
    </button>
    <button class="osprey-popover-item" type="button" role="menuitem" data-more="about"
            aria-haspopup="dialog" aria-controls="about-popover">
      <svg class="osprey-ic" data-icon="info" aria-hidden="true"></svg>About OSPREY
    </button>
  </div>

  <div class="osprey-popover shortcuts-popover" id="shortcuts-popover" role="dialog" aria-label="Keyboard shortcuts" hidden>
    <div class="osprey-popover-title">Keyboard shortcuts</div>
    <dl class="shortcuts-list">
      <div><dt>Search everything</dt><dd><kbd class="osprey-kbd" data-chord="K">&#8984;K</kbd></dd></div>
      <div><dt>Show or hide sidebar labels</dt><dd><kbd class="osprey-kbd" data-chord="\\">&#8984;\\</kbd></dd></div>
      <div><dt>Open settings</dt><dd><kbd class="osprey-kbd" data-chord=",">&#8984;,</kbd></dd></div>
    </dl>
  </div>

  <div class="osprey-popover about-popover" id="about-popover" role="dialog" aria-label="About OSPREY" hidden>
    <div class="osprey-popover-title">About OSPREY</div>
    <p class="about-popover-version" data-osprey-version>Version …</p>
  </div>`;

const getSessionId = () => 'sess-1';

/** Render the fixture and wire the menu, as app.js does at boot. */
function boot() {
  document.body.innerHTML = FIXTURE;
  mod.initMoreMenu({ getSessionId });
}

const button = () => byId('more-btn');
const menu = () => byId('more-menu');
/** @param {string} selector */
const row = (selector) => qs(menu(), selector);
/** @param {string} selector */
const checked = (selector) => row(selector).getAttribute('aria-checked');
const familyRows = () => [...menu().querySelectorAll('[data-theme-family]')];

/** Open the menu through ☰ More. */
function openMenu() {
  button().click();
  expect(menu().hidden).toBe(false);
}

/** @param {string} key */
function pressOnFocused(key) {
  const target = document.activeElement ?? document.body;
  target.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true }));
}

/** Let the About read (fetch, then json) land. */
async function settle() {
  for (let i = 0; i < 10; i++) await Promise.resolve();
}

beforeEach(async () => {
  vi.resetModules();
  mocks.startTour.mockReset();
  mocks.openActivityLog.mockReset();
  theme.preference = 'auto';
  theme.current = 'osprey-dark';
  theme.family = 'osprey';
  theme.subscribers.length = 0;
  theme.setTheme.mockReset();
  theme.toggleTheme.mockReset();
  theme.setFamily.mockReset();
  Reflect.deleteProperty(window, '__OSPREY_PREFIX__');
  fetchMock = vi.fn(async () => ({
    ok: true,
    status: 200,
    json: async () => ({ version: '1.2.3' }),
  }));
  vi.stubGlobal('fetch', fetchMock);
  mod = await import(MODULE);
});

afterEach(() => {
  vi.unstubAllGlobals();
  document.documentElement.removeAttribute('data-agent-surface');
  document.documentElement.removeAttribute('data-terminal');
  document.body.innerHTML = '';
});

/* ---- the rows ------------------------------------------------------------- */

describe('rows', () => {
  test('offers no terminal and no view', () => {
    boot();
    expect(menu().querySelector('[data-view]')).toBeNull();
    expect(menu().textContent).not.toMatch(/terminal|simple|expert/i);
  });

  test('the server-rendered rows are all there', () => {
    boot();
    expect(qs(menu(), '.more-menu-who .more-menu-user').textContent).toBe('alice');
    for (const look of ['auto', 'light', 'dark']) {
      row(`[role="menuitemradio"][data-appearance="${look}"]`);
    }
    expect(row('#more-settings').dataset.drawerTrigger).toBe('settings-drawer');
    for (const more of ['shortcuts', 'tour', 'activity', 'about']) row(`[data-more="${more}"]`);
  });

  test('has no Sign out row: About is the last row, with no separator after it', () => {
    boot();
    expect(menu().querySelector('#logout-btn')).toBeNull();
    expect(menu().textContent).not.toContain('Sign out');
    expect(menu().lastElementChild).toBe(row('[data-more="about"]'));
  });

  test('builds one radio per theme family, in roster order', () => {
    boot();
    const rows = familyRows();
    expect(rows.map((el) => el.getAttribute('data-theme-family'))).toEqual([
      'osprey',
      'high-contrast',
    ]);
    expect(rows.map((el) => el.textContent)).toEqual(['OSPREY', 'High Contrast']);
    for (const el of rows) {
      expect(el.getAttribute('role')).toBe('menuitemradio');
      expect(el.classList.contains('osprey-popover-item')).toBe(true);
      expect(el.querySelector('svg.osprey-popover-check')).not.toBeNull();
    }
  });

  test('is a no-op without the button or the menu', () => {
    document.body.innerHTML = FIXTURE;
    button().remove();
    mod.initMoreMenu({ getSessionId });
    expect(familyRows()).toHaveLength(0);
  });
});

/* ---- radios ---------------------------------------------------------------- */

describe('Appearance', () => {
  test("the radios follow theme-manager's modePreference", () => {
    for (const preference of /** @type {const} */ (['auto', 'light', 'dark'])) {
      theme.preference = preference;
      boot();
      for (const look of ['auto', 'light', 'dark']) {
        expect(checked(`[data-appearance="${look}"]`)).toBe(String(look === preference));
      }
    }
  });

  test('a theme change repaints the radios', () => {
    boot();
    theme.preference = 'light';
    for (const fn of theme.subscribers) fn();
    expect(checked('[data-appearance="light"]')).toBe('true');
    expect(checked('[data-appearance="auto"]')).toBe('false');
  });

  test('Match system follows the OS', () => {
    boot();
    row('[data-appearance="auto"]').click();
    expect(theme.setTheme).toHaveBeenCalledWith('auto');
    expect(theme.toggleTheme).not.toHaveBeenCalled();
  });

  test('picking the other side toggles', () => {
    theme.current = 'osprey-dark';
    boot();
    row('[data-appearance="light"]').click();
    expect(theme.toggleTheme).toHaveBeenCalledTimes(1);
    expect(theme.setTheme).not.toHaveBeenCalled();
  });

  test('picking the side already showing pins it', () => {
    theme.current = 'osprey-dark';
    boot();
    row('[data-appearance="dark"]').click();
    expect(theme.setTheme).toHaveBeenCalledWith('osprey-dark');
    expect(theme.toggleTheme).not.toHaveBeenCalled();
  });
});

describe('Theme family', () => {
  test('the radios follow getFamily and a pick calls setFamily', () => {
    theme.family = 'high-contrast';
    boot();
    expect(familyRows().map((el) => el.getAttribute('aria-checked'))).toEqual(['false', 'true']);

    openMenu();
    qs(menu(), '[data-theme-family="osprey"]').click();
    expect(theme.setFamily).toHaveBeenCalledWith('osprey');
    expect(menu().hidden).toBe(false);
  });
});

/* ---- open and close --------------------------------------------------------- */

describe('open and close', () => {
  test('☰ More opens the menu and closes it again', () => {
    boot();
    openMenu();
    expect(button().getAttribute('aria-expanded')).toBe('true');
    expect(button().getAttribute('aria-controls')).toBe('more-menu');
    // Focus moves into the menu, onto its first row.
    expect(document.activeElement).toBe(row('[data-appearance="auto"]'));

    button().click();
    expect(menu().hidden).toBe(true);
    expect(button().getAttribute('aria-expanded')).toBe('false');
  });

  test('Escape closes it and focus returns to ☰ More', () => {
    boot();
    openMenu();
    pressOnFocused('Escape');
    expect(menu().hidden).toBe(true);
    expect(button().getAttribute('aria-expanded')).toBe('false');
    expect(document.activeElement).toBe(button());
  });

  test('Settings closes it and parks focus on ☰ More', () => {
    boot();
    openMenu();
    row('#more-settings').click();
    expect(menu().hidden).toBe(true);
    expect(document.activeElement).toBe(button());
  });
});

/* ---- the rows that move the operator elsewhere ------------------------------ */

describe('rows with their own owners', () => {
  test('Take the tour closes the menu and starts the tour', () => {
    boot();
    openMenu();
    row('[data-more="tour"]').click();
    expect(mocks.startTour).toHaveBeenCalledTimes(1);
    expect(menu().hidden).toBe(true);
    expect(document.activeElement).toBe(button());
  });

  test('Activity log closes the menu and opens the log for the session', () => {
    boot();
    openMenu();
    row('[data-more="activity"]').click();
    expect(mocks.openActivityLog).toHaveBeenCalledWith(getSessionId);
    expect(menu().hidden).toBe(true);
    expect(document.activeElement).toBe(button());
  });

  test('Keyboard shortcuts opens beside the menu, which stays open', () => {
    boot();
    openMenu();
    const rowEl = row('[data-more="shortcuts"]');
    rowEl.click();
    expect(byId('shortcuts-popover').hidden).toBe(false);
    expect(rowEl.getAttribute('aria-expanded')).toBe('true');
    expect(menu().hidden).toBe(false);
  });

  test('About reads the version from /health, once', async () => {
    boot();
    openMenu();
    row('[data-more="about"]').click();
    await settle();

    const about = byId('about-popover');
    expect(about.hidden).toBe(false);
    expect(menu().hidden).toBe(false);
    expect(qs(about, '[data-osprey-version]').textContent).toBe('Version 1.2.3');
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0][0]).toBe('/health');

    // Closed and opened again: the version is already there.
    row('[data-more="about"]').click();
    row('[data-more="about"]').click();
    await settle();
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  test('About says so when the version cannot be read', async () => {
    fetchMock.mockImplementation(async () => {
      throw new TypeError('Failed to fetch');
    });
    boot();
    openMenu();
    row('[data-more="about"]').click();
    await settle();
    expect(qs(byId('about-popover'), '[data-osprey-version]').textContent).toBe(
      'Version unavailable'
    );
  });
});

/* ---- keyboard --------------------------------------------------------------- */

/* popover.js owns the menu keys (more-menu.js has no handler of its own);
   these pin that ☰ More's rows get them. */

describe('keyboard', () => {
  test('ArrowDown and ArrowUp move between the rows, wrapping', () => {
    boot();
    openMenu();
    expect(document.activeElement).toBe(row('[data-appearance="auto"]'));

    pressOnFocused('ArrowDown');
    expect(document.activeElement).toBe(row('[data-appearance="light"]'));
    pressOnFocused('ArrowUp');
    expect(document.activeElement).toBe(row('[data-appearance="auto"]'));
    pressOnFocused('ArrowUp');
    expect(document.activeElement).toBe(row('[data-more="about"]'));
    pressOnFocused('ArrowDown');
    expect(document.activeElement).toBe(row('[data-appearance="auto"]'));
  });

  test('Home and End jump to the first and last rows', () => {
    boot();
    openMenu();
    pressOnFocused('End');
    expect(document.activeElement).toBe(row('[data-more="about"]'));
    pressOnFocused('Home');
    expect(document.activeElement).toBe(row('[data-appearance="auto"]'));
  });
});
