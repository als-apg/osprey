// @ts-check
/* OSPREY Web Terminal — the sidebar's ☰ More menu
 *
 * The page has no header bar, so the controls an operator reaches for rarely
 * live behind ☰ More at the foot of the sidebar:
 *
 *   - who is signed in (user, role, deployment), when the page knows;
 *   - Appearance: follow the system, light or dark, and Theme family — the
 *     hub's theme preference through theme-manager.js, which owns it;
 *   - Settings (⌘, / Ctrl+,), Keyboard shortcuts, Take the tour, Activity log
 *     and About OSPREY.
 *
 * Sign out is not here: it is a row of the sidebar footer (index.html,
 * app.js), so it is one click away rather than behind a menu.
 *
 * The rows are server-rendered in index.html (`#more-menu`), so the id other
 * modules and tests resolve — `#more-settings` with its `data-drawer-trigger`
 * (settings-warning-gate.js opens the drawer from it) — is in the page from
 * the first paint. Only the theme families, which theme-manager.js defines,
 * are built here. This module keeps every radio's `aria-checked` true to the
 * state it names, and runs the rows that have no owner of their own.
 *
 * The menu is a body-level `.osprey-popover` opened by the design system's
 * popover.js beside the sidebar, bottom-aligned with ☰ More: an outside
 * click, Escape (focus back on ☰ More) and a window resize close it. A pick
 * of Appearance or Theme leaves it open, so the operator can compare
 * and go straight back; a row that moves the operator elsewhere closes it.
 * Keyboard shortcuts and About open beside the menu, which stays open
 * beneath them. popover.js also moves ArrowUp / ArrowDown / Home / End
 * between the rows and returns focus to ☰ More whenever the menu closes
 * with focus inside it.
 */

import { closePopover, openPopover, togglePopover } from '/design-system/js/popover.js';
import {
  getFamily,
  getTheme,
  modeOf,
  modePreference,
  setFamily,
  setTheme,
  subscribe as subscribeTheme,
  themeFamilies,
  toggleTheme,
} from '/design-system/js/theme-manager.js';
import { icon } from '/design-system/js/icons.js';
import { fetchJSON } from './api.js';
import { openActivityLog } from './activity-log-link.js';
import { startTour } from './tour.js';

const MENU_ID = 'more-menu';
const BUTTON_ID = 'more-btn';
const SHORTCUTS_ID = 'shortcuts-popover';
const ABOUT_ID = 'about-popover';

/** The rows a click acts on. */
const ITEM_SELECTOR = '[role="menuitem"], [role="menuitemradio"]';

/**
 * Mark the radios of one group: `aria-checked` true on the one whose
 * `data-<key>` equals `value`.
 * @param {HTMLElement} menu
 * @param {string} key - the dataset key (`appearance`, `themeFamily`)
 * @param {string | null} value
 */
function checkRadios(menu, key, value) {
  for (const radio of menu.querySelectorAll('[role="menuitemradio"]')) {
    const el = /** @type {HTMLElement} */ (radio);
    if (el.dataset[key] === undefined) continue;
    el.setAttribute('aria-checked', String(el.dataset[key] === value));
  }
}

/**
 * Bring every radio in line with the state it names.
 * @param {HTMLElement} menu
 */
function syncMenu(menu) {
  checkRadios(menu, 'appearance', modePreference());
  checkRadios(menu, 'themeFamily', getFamily());
}

/**
 * Apply an Appearance pick. `auto` follows the OS; `light` and `dark` pin
 * that side of the current family — including the side `auto` already
 * resolved to, which turns following into a choice.
 * @param {string} want
 */
function pickAppearance(want) {
  if (want === 'auto') {
    setTheme('auto');
    return;
  }
  if (want !== 'light' && want !== 'dark') return;
  const current = getTheme();
  if (modeOf(current) === want && current) setTheme(current);
  else toggleTheme();
}

/**
 * One theme-family radio, built from theme-manager.js's roster.
 * @param {{id: string, label: string}} family
 * @returns {HTMLButtonElement}
 */
function familyRadio(family) {
  const item = document.createElement('button');
  item.type = 'button';
  item.className = 'osprey-popover-item';
  item.setAttribute('role', 'menuitemradio');
  item.setAttribute('aria-checked', 'false');
  item.dataset.themeFamily = family.id;
  item.textContent = family.label;
  const check = icon('check');
  check.classList.add('osprey-popover-check');
  item.append(check);
  return item;
}

/**
 * Show the deployment's OSPREY version in the About popover, read once from
 * `/health` (the same `version` the feedback dialog reports).
 * @param {HTMLElement} about
 */
async function fillVersion(about) {
  const slot = /** @type {HTMLElement | null} */ (about.querySelector('[data-osprey-version]'));
  if (!slot || slot.dataset.loaded === 'true') return;
  try {
    const health = await fetchJSON('/health');
    const version = typeof health?.version === 'string' ? health.version.trim() : '';
    slot.textContent = version ? `Version ${version}` : 'Version unknown';
    slot.dataset.loaded = 'true';
  } catch {
    slot.textContent = 'Version unavailable';
  }
}

/**
 * Wire ☰ More. No-op when the page lacks the button or the menu.
 * @param {{getSessionId: () => string|null|undefined}} deps - the terminal's
 *   session, read when the activity log is opened rather than now.
 * @returns {void}
 */
export function initMoreMenu({ getSessionId }) {
  const button = document.getElementById(BUTTON_ID);
  const menu = document.getElementById(MENU_ID);
  if (!button || !menu) return;
  const sidebar = button.closest('#sidebar') ?? button;

  const families = menu.querySelector('[data-theme-families]');
  if (families) families.replaceChildren(...themeFamilies().map(familyRadio));

  // A row that moves the operator elsewhere closes the menu; popover.js parks
  // focus on ☰ More, so whatever opens next (the drawer, the tour) returns
  // focus to a control that is on screen rather than into the hidden menu.
  const close = () => closePopover(menu);

  button.addEventListener('click', () => {
    if (menu.hidden) syncMenu(menu);
    togglePopover(button, menu, { place: 'beside-right-bottom', edge: sidebar });
  });

  /**
   * Open a static popover beside the menu, from the row that names it.
   * @param {HTMLElement} row
   * @param {string} id
   * @returns {HTMLElement | null}
   */
  const openBeside = (row, id) => {
    const pop = document.getElementById(id);
    if (pop) openPopover(pop, row, { place: 'beside-right', edge: menu });
    return pop;
  };

  menu.addEventListener('click', (event) => {
    const target = /** @type {Element | null} */ (event.target);
    const row = /** @type {HTMLElement | null} */ (target?.closest(ITEM_SELECTOR) ?? null);
    if (!row || !menu.contains(row)) return;
    const { appearance, themeFamily, more } = row.dataset;
    if (appearance) {
      pickAppearance(appearance);
    } else if (themeFamily) {
      setFamily(themeFamily);
    } else if (row.id === 'more-settings') {
      // settings-warning-gate.js opens the drawer from this same click.
      // Cmd/Ctrl+, clicks this row with the menu shut (app.js); closing then
      // would move focus to ☰ More, away from where the operator was typing.
      if (!menu.hidden) close();
    } else if (more === 'shortcuts') {
      openBeside(row, SHORTCUTS_ID);
    } else if (more === 'about') {
      const about = openBeside(row, ABOUT_ID);
      if (about) void fillVersion(about);
    } else if (more === 'tour') {
      close();
      startTour();
    } else if (more === 'activity') {
      close();
      openActivityLog(getSessionId);
    }
  });

  subscribeTheme(() => syncMenu(menu));
  syncMenu(menu);
}
