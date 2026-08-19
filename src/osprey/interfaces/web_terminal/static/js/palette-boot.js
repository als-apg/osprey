// @ts-check
/* OSPREY Web Terminal — Command Palette Bootstrap
 *
 * The wiring seam between the app and the command palette: it assembles the
 * live dependency bundle openPalette() needs (panel getters, the vetted action
 * closures) and installs the two entry points — the header trigger button and
 * the platform-split Cmd/Ctrl+K hotkey. Kept out of app.js so that file stays a
 * thin init dispatcher and the palette's app-facing wiring lives beside the
 * palette itself.
 *
 * The palette runs in BOTH ui modes — the header trigger is not mode-gated, so
 * the action cluster never shifts under an operator on an Expert/Simple flip.
 * What changes is the REGISTRY, not the entry points: simple mode hides the
 * terminal (the operator console takes its place) and locks the dock to a
 * single service tile, so `buildPaletteDeps` drops the terminal/session actions,
 * the Layouts group and the "Open … in a new tile" rows there rather than
 * offering picks that would silently do nothing. Everything simple mode can
 * honour — settings search, the panel show/focus verbs the rail already
 * exposes, popping a panel out to its own browser tab, rail placement, the
 * drawer tabs, the safety reference — stays.
 */

import { restartTerminal, startTerminal } from './terminal.js';
import {
  getHiddenPanels,
  getVisiblePanels,
  getPopoutPanels,
  getPresets,
  showPanel,
  activateTab,
  applyMenuPreset,
  popoutPanel,
} from './panel-manager.js';
import { openPanelBeside } from './panel-placement.js';
import { openDrawerTab, revealSetting } from './settings.js';
import { startNewSession } from './sessions.js';
import { setRailPosition } from './rail-position.js';
import { openPalette, closePalette, isOpen } from './palette.js';

/** True on macOS/iPadOS, where the palette hotkey is Cmd+K instead of Ctrl+K. */
function isMacPlatform() {
  const nav = /** @type {Navigator & { userAgentData?: { platform?: string } }} */ (navigator);
  const platform = nav.userAgentData?.platform || nav.platform || '';
  return /Mac|iPhone|iPad|iPod/i.test(platform);
}

/** @returns {'expert'|'simple'} the resolved ui mode from the authoritative <html> attribute. */
function currentUiMode() {
  return document.documentElement.dataset.uiMode === 'simple' ? 'simple' : 'expert';
}

/**
 * Assemble the live dependency bundle `openPalette` needs. Built fresh on each
 * open because panel visibility, presets, the config snapshot AND the ui mode
 * all drift over a session — the palette must reflect the CURRENT state, not
 * boot state. The action closures are the vetted mechanisms (paired restart,
 * gated drawer tabs, new-tab safety nav) — do not collapse or reorder them
 * casually.
 *
 * Typed against the REGISTRY's dep bundle rather than palette.js's OpenDeps:
 * everything assembled here exists to reach `buildRegistry`, and PaletteDeps is
 * the declaration site for the panel-verb deps. openPalette accepts it as-is.
 * @returns {import('./palette-registry.js').PaletteDeps}
 */
function buildPaletteDeps() {
  const simple = currentUiMode() === 'simple';

  /** @type {Array<{ label: string, detail?: string, run: () => void }>} */
  const actions = [];

  // Terminal + session verbs address the xterm card, which simple mode replaces
  // with the operator console — offering them there would act on a surface the
  // operator cannot see.
  if (!simple) {
    // restartTerminal tears down the PTY but does NOT reconnect — pair with
    // startTerminal or the terminal is left stranded.
    actions.push({ label: 'Restart terminal', run: async () => { await restartTerminal(); startTerminal(); } });
    actions.push({ label: 'New session', run: () => { startNewSession(); } });
  }

  // Always offer the mode the operator is NOT in, and reuse the wired
  // mode-toggle handler rather than re-implementing the flip.
  const otherMode = simple ? 'expert' : 'simple';
  actions.push({
    label: `Switch to ${simple ? 'Expert' : 'Simple'} mode`,
    run: () => document.querySelector(`.mode-segment[data-mode="${otherMode}"]`)?.dispatchEvent(new MouseEvent('click', { bubbles: true })),
  });

  // Both rail directions are always offered (the palette is searched, not
  // browsed) — flipping to the current position is a harmless no-op.
  actions.push({ label: 'Move panel rail to top', run: () => setRailPosition('top') });
  actions.push({ label: 'Move panel rail to left', run: () => setRailPosition('left') });
  // Drawer tabs always go THROUGH openDrawerTab's warning gate.
  actions.push({ label: 'Open Settings', run: () => { openDrawerTab('tab-config'); } });
  actions.push({ label: 'Open Memory gallery', run: () => { openDrawerTab('tab-memory'); } });
  actions.push({ label: 'Open Prompt gallery', run: () => { openDrawerTab('tab-behavior'); } });
  // Logout only exists in multi-user deployments (the button is server-gated).
  if (document.getElementById('logout-btn')) {
    actions.push({ label: 'Log out', run: () => document.getElementById('logout-btn')?.click() });
  }

  /** @type {import('./palette-registry.js').PaletteDeps} */
  const deps = {
    // Panel show/focus is exactly what a rail click does, and the rail is
    // present in both modes — in simple the show path takes over the single
    // locked service tile, which is the rail's own behaviour there.
    getHiddenPanels,
    getVisiblePanels,
    showPanel,
    // "Focus" reports as a user-initiated activation so the server sees it.
    focusPanel: (id) => activateTab(id, { userInitiated: true }),
    // Popping out is a browser-tab action, so it survives simple mode's locked
    // layout untouched. getPopoutPanels is already narrowed to rail members
    // whose standalone URL resolved, and INCLUDES the active panel — the
    // context menu's "Open in a new window" reads the same way.
    getPopoutPanels,
    popoutPanel,
    revealSetting,
    actions,
  };

  // Both of the below place a SECOND service tile, which simple mode's locked
  // layout cannot hold — omitting the dep is what drops the rows.
  if (!simple) {
    // "Open … in a new tile" shares getVisiblePanels with the Focus rows, so
    // withholding the closure is the only thing that can separate the two.
    deps.openPanelBeside = openPanelBeside;
    // Layouts restore a multi-tile arrangement; omitting the getter drops the
    // whole group from the registry.
    deps.getPresets = getPresets;
    deps.applyPreset = applyMenuPreset;
  }

  return deps;
}

/**
 * Wire the header trigger + Cmd/Ctrl+K hotkey for the command palette. Both
 * entry points are live in both ui modes; `buildPaletteDeps` is what narrows
 * the registry in simple. Degrades gracefully if the button is absent.
 */
export function initCommandPalette() {
  const btn = document.getElementById('command-palette-btn');
  if (btn) {
    btn.addEventListener('click', () => openPalette(buildPaletteDeps()));
  }

  const isMac = isMacPlatform();
  const termContainer = document.getElementById('terminal-container');

  // Capture phase: xterm swallows bubbled keydowns, so the hotkey must be read
  // before the event reaches the terminal.
  document.addEventListener('keydown', (e) => {
    if (e.key !== 'k' && e.key !== 'K') return;

    let match;
    if (isMac) {
      // macOS: Cmd+K from anywhere in the top document.
      match = e.metaKey;
    } else {
      // Non-macOS: Ctrl+K, but only OUTSIDE the terminal — inside, Ctrl+K stays
      // readline kill-line. The header button is the in-terminal fallback.
      const active = document.activeElement;
      const inTerminal = !!(termContainer && active instanceof Node && termContainer.contains(active));
      match = e.ctrlKey && !inTerminal;
    }
    if (!match) return;

    e.preventDefault();
    e.stopPropagation();
    if (isOpen()) closePalette();
    else openPalette(buildPaletteDeps());
  }, true);
}
