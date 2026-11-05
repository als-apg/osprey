// @ts-check
/* OSPREY Lattice Dashboard — Toolbar Actions and Tile-Bar Contribution
 *
 * The panel toolbar's three actions (Refresh, Baseline, Verify), the same
 * standalone and embedded, and the dashboard's one menu, the figure layout.
 * Embedded, the menu goes to the hub (see /design-system/js/header-contrib.js),
 * which lists it under ⋯ › This panel, with the selected model's name as the
 * tile's subtitle. Standalone or popped out there is no hub ⋯, so the
 * toolbar's own ⋯ (#btn-panel-menu) opens the same entries.
 *
 * Setting the baseline silently overwrites the reference every figure is
 * compared against, so it is two-step: the first click arms, and only a
 * second click inside ARM_WINDOW_MS runs it.
 *
 * The button-enabled state and the selected model's name are pushed in from
 * the dashboard state (see syncState), the layout from ui.js (see
 * syncLayout) — this module reads no state of its own.
 */

import { openContextMenu } from '/design-system/js/context-menu.js';
import { isEmbedded } from '/design-system/js/frame-params.js';
import { contributeHeader, onHeaderAction } from '/design-system/js/header-contrib.js';
import { icon } from '/design-system/js/icons.js';
import { writeTitle } from '/design-system/js/tooltip.js';

/** How long an armed Baseline stays armed before it disarms itself. */
const ARM_WINDOW_MS = 4000;

/** The id of the one menu the dashboard contributes (⋯ › This panel). */
export const LAYOUT_MENU_ID = 'lattice';

/** The layout entries, in menu order; each id is a ui.js layout mode. */
const LAYOUTS = [
  { id: 'stacked', label: 'Layout: Stacked' },
  { id: 'grid', label: 'Layout: Grid' },
];

const VERIFY_TITLE = 'Run DA + LMA verification';
const VERIFY_NEEDS_LATTICE = 'Verify needs a lattice';
const VERIFY_SINGLE_PASS = 'Verify is not available for a single-pass model';

/**
 * @typedef {object} HeaderCallbacks
 * @property {() => void} onRefresh - recompute the fast figures
 * @property {() => void} onVerify - launch the on-demand verification figures
 * @property {() => void} onBaseline - overwrite the baseline; fired on the CONFIRMED click only
 * @property {(mode: string) => void} [onLayout] - show the figures in the layout picked in ⋯
 */

/**
 * @param {string} id
 * @returns {HTMLButtonElement | null}
 */
function button(id) {
  return /** @type {HTMLButtonElement | null} */ (document.getElementById(id));
}

/**
 * Create the toolbar-action controller. Call init() once the embedded body
 * class is applied (contributions are a no-op standalone), then feed it every
 * dashboard state via syncState() and every layout via syncLayout().
 * @param {HeaderCallbacks} callbacks
 */
export function createHeader(callbacks) {
  let armed = false;
  /** @type {ReturnType<typeof setTimeout> | undefined} */
  let armTimer;
  /** @type {string | null} */
  let latticeName = null;
  let hasLattice = false;
  let canVerify = false;
  let layout = 'stacked';

  /** The menu's entries, shared by the hub's ⋯ and the standalone one. */
  function menuEntries() {
    return LAYOUTS.map(({ id, label }) => ({ id, label, checked: layout === id }));
  }

  /**
   * Run a menu entry, from either ⋯.
   * @param {string} id - the menu's id
   * @param {string} [value] - the entry's id
   */
  function menuAction(id, value) {
    if (id !== LAYOUT_MENU_ID) return;
    if (LAYOUTS.some((entry) => entry.id === value)) callbacks.onLayout?.(String(value));
  }

  /** Publish the WHOLE contribution — the hub renders the last one verbatim. */
  function publishContribution() {
    /** @type {import('/design-system/js/header-contrib.js').HeaderItem[]} */
    const items = [];
    // The subtitle is identity only: with no model there is nothing to name.
    if (latticeName) items.push({ kind: 'text', id: 'lattice-name', text: latticeName });
    items.push({ kind: 'menu', id: LAYOUT_MENU_ID, items: menuEntries() });
    contributeHeader(items);
  }

  /**
   * The standalone ⋯: present only off the hub, which has its own ⋯ per tile.
   * @param {HTMLButtonElement | null} menuBtn
   */
  function initPanelMenu(menuBtn) {
    if (!menuBtn) return;
    if (isEmbedded()) {
      menuBtn.remove();
      return;
    }
    if (!menuBtn.firstElementChild) menuBtn.append(icon('more'));
    menuBtn.hidden = false;
    menuBtn.addEventListener('click', () => {
      const rect = menuBtn.getBoundingClientRect();
      openContextMenu({
        x: rect.right,
        y: rect.bottom,
        anchorEl: menuBtn,
        triggerEl: menuBtn,
        ariaLabel: 'Panel menu',
        items: menuEntries().map((entry) => ({
          label: entry.label,
          checked: entry.checked,
          run: () => menuAction(LAYOUT_MENU_ID, entry.id),
        })),
      });
    });
  }

  /** Why Verify does not apply, or its own description when it does. */
  function verifyTitle() {
    if (!hasLattice) return VERIFY_NEEDS_LATTICE;
    if (!canVerify) return VERIFY_SINGLE_PASS;
    return VERIFY_TITLE;
  }

  /** Paint the dashboard state and the arming onto the toolbar buttons. */
  function paintToolbar() {
    const refresh = button('btn-refresh');
    if (refresh) refresh.disabled = !hasLattice;

    const verify = button('btn-verify');
    if (verify) {
      verify.disabled = !hasLattice || !canVerify;
      writeTitle(verify, verifyTitle());
    }

    const baseline = button('btn-baseline');
    if (baseline) {
      baseline.disabled = !hasLattice;
      // Both labels stay in the button (index.html), so arming never changes
      // its width and never shoves Verify or the theme switcher beside it.
      baseline.toggleAttribute('data-armed', armed);
      writeTitle(
        baseline,
        armed ? 'Click again to overwrite the baseline' : 'Set current as baseline'
      );
    }
  }

  function render() {
    paintToolbar();
    publishContribution();
  }

  function disarm() {
    clearTimeout(armTimer);
    armTimer = undefined;
    armed = false;
  }

  /** Arm on the first click, run on the second one inside the window. */
  function baselineClicked() {
    const confirmed = armed;
    disarm();
    if (!confirmed) {
      armed = true;
      armTimer = setTimeout(() => {
        armed = false;
        render();
      }, ARM_WINDOW_MS);
    }
    render();
    if (confirmed) callbacks.onBaseline();
  }

  /**
   * Bind the toolbar buttons, the standalone ⋯ and the tile bar's menu
   * round-trip, then publish the initial contribution.
   */
  function init() {
    const refresh = button('btn-refresh');
    if (refresh && !refresh.firstElementChild) refresh.append(icon('refresh'));
    refresh?.addEventListener('click', callbacks.onRefresh);
    button('btn-verify')?.addEventListener('click', callbacks.onVerify);
    button('btn-baseline')?.addEventListener('click', baselineClicked);
    initPanelMenu(button('btn-panel-menu'));

    onHeaderAction(menuAction);

    render();
  }

  /**
   * Adopt the dashboard state the toolbar and the tile bar reflect: the
   * selected model's name, whether the actions apply at all, and whether
   * Verify does.
   * @param {any} state - the /api/state payload
   */
  function syncState(state) {
    const selection = state?.selection;
    const nowHasLattice = selection?.status === 'ready';
    latticeName = selection?.model || null;
    canVerify = nowHasLattice && !!selection.capabilities?.verify;
    // Losing the lattice mid-arm would leave a confirm prompt for an action
    // that no longer applies.
    if (!nowHasLattice && armed) disarm();
    hasLattice = nowHasLattice;
    render();
  }

  /**
   * Adopt the figure layout ui.js shows, so the menu checks it.
   * @param {string} mode - `stacked` or `grid`
   */
  function syncLayout(mode) {
    layout = mode;
    publishContribution();
  }

  return { init, syncState, syncLayout };
}
