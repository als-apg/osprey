// @ts-check
/* OSPREY Web Terminal — Display Menu (the header sliders button)
 *
 * One quiet faders button in the header collapses every display preference
 * behind a popover card: Appearance (light/dark within the active theme family), View
 * (Expert/Simple), Theme (the family pills), and — last — a session footer
 * naming the signed-in user above the Settings / Log out pair. It replaces the
 * hub header's always-visible segmented mode toggle + `<osprey-theme-switcher>`
 * pair, the standalone settings gear beside them, and the identity chip that
 * used to carry logout in a popover of its own; standalone fleet pages
 * (session.html, the panels) keep the shared switcher component; this menu is
 * hub chrome only.
 *
 * Division of labour:
 *   - This module owns the popover (open/close, outside-click, Escape) and
 *     the two THEME rows, wired straight to theme-manager.js — `toggleTheme()`
 *     for appearance, `setFamily()` for the pills. Like the shared switcher,
 *     it never hand-resolves a concrete theme id: theme-manager.js is the
 *     single source of truth and this menu reads state back via
 *     `getTheme()`/`getFamily()` + `subscribe()`.
 *   - The VIEW row is app.js's initModeToggle() unchanged: the card carries
 *     the same `#mode-toggle` / `.mode-segment[data-mode]` markup, so the mode
 *     flip, persistence, and panel broadcast all live where they always did,
 *     and the active segment stays CSS-driven off `html[data-ui-mode]`.
 *     Like the theme rows, a mode pick leaves the card OPEN: every row of the
 *     card renders identically in both modes, so the card an operator is
 *     looking at survives the flip and they can compare the two shells (or go
 *     straight back) without re-opening the menu.
 *   - The SETTINGS button is not this module's to open: it keeps the
 *     `[data-drawer-trigger="settings-drawer"]` contract, so settings.js's
 *     first-time warning gate stays the sole open path and app.js still
 *     mirrors the drawer's open state onto it as `.active`. This is the ONE
 *     row that closes the card, because opening the drawer moves the operator
 *     to a different surface entirely.
 *   - LOG OUT is likewise app.js's, by `#logout-btn` + `data-landing-url`
 *     (palette-boot.js's "Log out" command also finds it by id). Deliberately
 *     NOT closed by this module: every path out of that handler navigates
 *     away, and closing the card first would hide the button's own aria-busy
 *     state while the POST is still in flight — from the operator and from
 *     assistive tech — at exactly the moment it matters.
 *
 * The popover grammar (open renders nothing lazily except the family pills'
 * one-time build; capture-phase outside-click + Escape close; aria-expanded
 * mirrored on the button) matches panel-add-menu.js, the shell's other header
 * popover.
 */

import {
  familyLabel,
  getFamily,
  getTheme,
  setFamily,
  subscribe,
  themeFamilies,
  toggleTheme,
} from '/design-system/js/theme-manager.js';
import { THEMES } from '/design-system/js/tokens.js';

/** @typedef {{id: string, label: string, mode: string, family: string}} ThemeEntry */

// tokens.js is plain (unchecked) generated JS — cast to the documented shape
// rather than relying on tsc's inference of the literal it emits (same
// rationale as theme-manager.js's own `_themes` cast).
const _themes = /** @type {ThemeEntry[]} */ (THEMES);

// Human label for a family id. Re-exported from theme-manager.js, which owns
// the single implementation shared with <osprey-theme-switcher> — a family that
// declares a `family_label` (e.g. 'desy' → 'DESY') must render identically in
// both pickers, which a second local derivation could not guarantee.
export { familyLabel };

// The available families, deduped, in THEMES declaration order. Re-exported
// from theme-manager.js for the same reason familyLabel is: it owns the single
// implementation shared with <osprey-theme-switcher>, so the two pickers can
// never offer a different set of families — or a different order — after a
// token regeneration.
export { themeFamilies };

/**
 * The `mode` ('dark'|'light') of a concrete theme id, or null when `id` is
 * not a recognized theme (including `null` before initTheme() resolves one).
 *
 * Consolidating the three copies of this helper into theme-manager.js is
 * tracked with the hub-adoption follow-up (see DESIGN.md).
 *
 * @param {string|null} id
 * @returns {string|null}
 */
function _modeOfId(id) {
  if (id === null) return null;
  const theme = _themes.find((entry) => entry.id === id);
  return theme ? theme.mode : null;
}

/**
 * Wire the header display menu. No-op if the button/card markup is absent so a
 * template without the control degrades gracefully.
 */
export function initDisplayMenu() {
  const root = document.getElementById('display-menu');
  const button = /** @type {HTMLButtonElement | null} */ (
    document.getElementById('display-menu-btn')
  );
  const card = document.getElementById('display-menu-card');
  if (!root || !button || !card) return;

  const appearanceSeg = /** @type {HTMLElement | null} */ (
    card.querySelector('.display-menu-appearance')
  );
  const familyPicker = document.getElementById('family-picker');

  const isOpen = () => card.classList.contains('open');

  /** @param {MouseEvent} e */
  function onDocClick(e) {
    if (e.target instanceof Node && root && !root.contains(e.target)) closeMenu();
  }

  /** @param {KeyboardEvent} e */
  function onKeydown(e) {
    if (e.key === 'Escape') {
      closeMenu();
      button?.focus();
    }
  }

  function openMenu() {
    if (!card || !button) return;
    card.classList.add('open');
    button.setAttribute('aria-expanded', 'true');
    // Capture-phase so an outside click closes before it does anything else.
    document.addEventListener('click', onDocClick, true);
    document.addEventListener('keydown', onKeydown, true);
  }

  function closeMenu() {
    if (!card || !button) return;
    card.classList.remove('open');
    button.setAttribute('aria-expanded', 'false');
    document.removeEventListener('click', onDocClick, true);
    document.removeEventListener('keydown', onKeydown, true);
  }

  button.addEventListener('click', (e) => {
    e.stopPropagation();
    if (isOpen()) closeMenu();
    else openMenu();
  });

  // View row: nothing to wire. initModeToggle() (app.js) owns the flip, and
  // the card deliberately stays open across it (see the module docstring).

  // System Settings row: settings.js's warning gate owns the open (it binds
  // the same `[data-drawer-trigger]` button); this module only dismisses the
  // card, since opening the drawer is a move to another surface.
  card.querySelector('.display-menu-settings')?.addEventListener('click', () => closeMenu());

  // Appearance row: flip light/dark within the active family, only when the
  // pick differs from the current mode (re-clicking the active side no-ops
  // rather than toggling away from it).
  appearanceSeg?.addEventListener('click', (e) => {
    const target =
      e.target instanceof Element ? e.target.closest('.display-seg-option') : null;
    if (!(target instanceof HTMLElement)) return;
    const want = target.dataset.appearance;
    if (want !== 'light' && want !== 'dark') return;
    if (want !== _modeOfId(getTheme())) toggleTheme();
  });

  // Theme row: one pill per family, built once from the THEMES registry.
  // textContent — labels derive from generated token data (never innerHTML).
  if (familyPicker) {
    for (const family of themeFamilies()) {
      const pill = document.createElement('button');
      pill.type = 'button';
      pill.className = 'display-family-pill';
      pill.dataset.family = family.id;
      pill.textContent = family.label;
      pill.setAttribute('aria-pressed', 'false');
      pill.addEventListener('click', () => setFamily(family.id));
      familyPicker.appendChild(pill);
    }
  }

  /**
   * Reflect theme-manager.js's current (family, mode) into the card: the
   * appearance thumb position rides `data-active` (CSS moves it), and the
   * active family pill gets `.active`. Called once now (initTheme() ran
   * before this in app.js's boot) and on every subsequent theme apply.
   */
  function syncUI() {
    const mode = _modeOfId(getTheme());
    if (appearanceSeg && mode !== null) {
      appearanceSeg.setAttribute('data-active', mode);
      for (const option of appearanceSeg.querySelectorAll('.display-seg-option')) {
        option.setAttribute(
          'aria-pressed',
          option instanceof HTMLElement && option.dataset.appearance === mode
            ? 'true'
            : 'false'
        );
      }
    }
    const family = getFamily();
    if (familyPicker && family !== null) {
      for (const pill of familyPicker.querySelectorAll('.display-family-pill')) {
        const active = pill instanceof HTMLElement && pill.dataset.family === family;
        pill.classList.toggle('active', active);
        pill.setAttribute('aria-pressed', active ? 'true' : 'false');
      }
    }
  }

  syncUI();
  subscribe(syncUI);
}
