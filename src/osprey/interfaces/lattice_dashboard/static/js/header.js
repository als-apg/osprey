// @ts-check
/* OSPREY Lattice Dashboard — Header Actions
 *
 * One state machine behind two renderings of the same three actions
 * (Refresh / Verify / Baseline): the standalone page's top-bar buttons, and
 * the web-terminal tile bar when the dashboard runs embedded — where the
 * panel has no bar of its own and contributes its controls to the hub's
 * (see /design-system/js/header-contrib.js).
 *
 * Setting the baseline silently overwrites the reference every figure is
 * compared against, so it is two-step: the first click arms, and only a
 * second click inside ARM_WINDOW_MS runs it. Because the arming lives here
 * rather than in either rendering, both show the same armed label.
 *
 * The button-enabled state and the lattice name are pushed in from the
 * dashboard state (see syncState) — this module reads no state of its own.
 */

import { contributeHeader, onHeaderAction } from '/design-system/js/header-contrib.js';

/** How long an armed Baseline stays armed before it disarms itself. */
const ARM_WINDOW_MS = 4000;

const NO_LATTICE = 'No lattice loaded';

/**
 * @typedef {object} HeaderCallbacks
 * @property {() => void} onRefresh - recompute the fast figures
 * @property {() => void} onVerify - launch the on-demand verification figures
 * @property {() => void} onBaseline - overwrite the baseline; fired on the CONFIRMED click only
 */

/**
 * Create the header-action controller. Call init() once the embedded body
 * class is applied (contributions are a no-op standalone), then feed it every
 * dashboard state via syncState().
 * @param {HeaderCallbacks} callbacks
 */
export function createHeader(callbacks) {
  let armed = false;
  /** @type {ReturnType<typeof setTimeout> | null} */
  let armTimer = null;
  let latticeName = NO_LATTICE;
  let hasLattice = false;

  /** Publish the WHOLE contribution — the hub renders the last one verbatim. */
  function publishContribution() {
    contributeHeader([
      { kind: 'text', id: 'lattice-name', text: latticeName, priority: 0 },
      {
        kind: 'button',
        id: 'refresh',
        label: 'Refresh',
        title: 'Refresh fast figures',
        disabled: !hasLattice,
        priority: 2,
      },
      {
        kind: 'button',
        id: 'verify',
        label: 'Verify',
        title: 'Run DA + LMA verification',
        disabled: !hasLattice,
        priority: 2,
      },
      {
        kind: 'button',
        id: 'baseline',
        // Same character count as the resting label, deliberately. This button
        // is not last in either rendering — the theme switcher follows it in
        // the standalone topbar, Refresh/Verify precede it in both — and every
        // one of those is a persistent control that a growing label shoves
        // sideways on the very click that arms this one. `Confirm baseline?`
        // was 9 characters longer — in the topbar's monospace face, a ~60px
        // shove of Refresh and Verify. What the second click does is in the
        // title, where it does not cost layout.
        label: armed ? 'Confirm?' : 'Baseline',
        title: armed ? 'Click again to overwrite the baseline' : 'Set current as baseline',
        tone: armed ? 'accent' : 'default',
        disabled: !hasLattice,
        priority: 1,
      },
    ]);
  }

  /** Paint the arming state onto the standalone button. */
  function paintBaselineButton() {
    const btn = document.getElementById('btn-baseline');
    if (!btn) return;
    btn.classList.toggle('action-btn--armed', armed);
    btn.title = armed ? 'Click again to overwrite the baseline' : 'Set current as baseline';
    const label = btn.querySelector('.baseline-label');
    // Same length as the resting label — see publishContribution for why.
    if (label) label.textContent = armed ? 'Confirm?' : 'Baseline';
  }

  function render() {
    paintBaselineButton();
    publishContribution();
  }

  function disarm() {
    if (armTimer !== null) clearTimeout(armTimer);
    armTimer = null;
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
   * Bind the standalone buttons and the tile bar's action round-trip, then
   * publish the initial contribution.
   */
  function init() {
    document.getElementById('btn-refresh')?.addEventListener('click', callbacks.onRefresh);
    document.getElementById('btn-verify')?.addEventListener('click', callbacks.onVerify);
    document.getElementById('btn-baseline')?.addEventListener('click', baselineClicked);

    onHeaderAction((id) => {
      if (id === 'refresh') callbacks.onRefresh();
      else if (id === 'verify') callbacks.onVerify();
      else if (id === 'baseline') baselineClicked();
    });

    render();
  }

  /**
   * Adopt the dashboard state the tile bar has to reflect: the loaded
   * lattice's name and whether the actions apply at all. render.js paints
   * the same facts onto the standalone bar.
   * @param {any} state - the /api/state payload
   */
  function syncState(state) {
    const path = state?.base_lattice || '';
    latticeName = path ? path.split('/').pop() || path : NO_LATTICE;
    const nowHasLattice = !!path;
    // Losing the lattice mid-arm would leave a confirm prompt for an action
    // that no longer applies.
    if (!nowHasLattice && armed) disarm();
    hasLattice = nowHasLattice;
    render();
  }

  return { init, syncState };
}
