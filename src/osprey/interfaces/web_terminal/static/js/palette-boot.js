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
 * The palette is expert-mode only: both entry points no-op in simple mode (the
 * button is also CSS-hidden there), since simple mode locks the dock and hides
 * the terminal, so panel/terminal commands would touch state the operator can't
 * see.
 */

import { restartTerminal, startTerminal } from './terminal.js';
import { withPrefix } from './api.js';
import {
  getHiddenPanels,
  getVisiblePanels,
  getPresets,
  showPanel,
  activateTab,
  applyMenuPreset,
} from './panel-manager.js';
import { openDrawerTab, revealSetting } from './settings.js';
import { startNewSession } from './sessions.js';
import { openPalette, closePalette, isOpen } from './palette.js';

/** True on macOS/iPadOS, where the palette hotkey is Cmd+K instead of Ctrl+K. */
function isMacPlatform() {
  const nav = /** @type {Navigator & { userAgentData?: { platform?: string } }} */ (navigator);
  const platform = nav.userAgentData?.platform || nav.platform || '';
  return /Mac|iPhone|iPad|iPod/i.test(platform);
}

/**
 * Assemble the live dependency bundle `openPalette` needs. Built fresh on each
 * open because panel visibility, presets, and the config snapshot all drift
 * over a session — the palette must reflect the CURRENT state, not boot state.
 * The action closures are the vetted mechanisms (paired restart, gated drawer
 * tabs, new-tab safety nav) — do not collapse or reorder them casually.
 * @returns {import('./palette.js').OpenDeps}
 */
function buildPaletteDeps() {
  /** @type {Array<{ label: string, detail?: string, run: () => void }>} */
  const actions = [
    // restartTerminal tears down the PTY but does NOT reconnect — pair with
    // startTerminal or the terminal is left stranded.
    { label: 'Restart terminal', run: async () => { await restartTerminal(); startTerminal(); } },
    { label: 'New session', run: () => { startNewSession(); } },
    // Reuse the wired mode-toggle handler rather than re-implementing the flip.
    { label: 'Switch to Simple mode', run: () => document.querySelector('.mode-segment[data-mode="simple"]')?.dispatchEvent(new MouseEvent('click', { bubbles: true })) },
    // Drawer tabs always go THROUGH openDrawerTab's warning gate.
    { label: 'Open Settings', run: () => { openDrawerTab('tab-config'); } },
    { label: 'Open Memory gallery', run: () => { openDrawerTab('tab-memory'); } },
    { label: 'Open Prompt gallery', run: () => { openDrawerTab('tab-behavior'); } },
    // New tab — a same-window navigation would tear down the PTY.
    { label: 'Open Safety reference', run: () => window.open(withPrefix('/static/safety.html'), '_blank', 'noopener') },
  ];
  // Logout only exists in multi-user deployments (the button is server-gated).
  if (document.getElementById('logout-btn')) {
    actions.push({ label: 'Log out', run: () => document.getElementById('logout-btn')?.click() });
  }

  return {
    getHiddenPanels,
    getVisiblePanels,
    getPresets,
    showPanel,
    // "Focus" reports as a user-initiated activation so the server sees it.
    focusPanel: (id) => activateTab(id, { userInitiated: true }),
    applyPreset: applyMenuPreset,
    revealSetting,
    actions,
  };
}

/**
 * Wire the header trigger + Cmd/Ctrl+K hotkey for the command palette. The
 * palette is expert-mode only, so both entry points no-op in simple mode (the
 * button is also CSS-hidden there). Degrades gracefully if the button is
 * absent.
 */
export function initCommandPalette() {
  const btn = document.getElementById('command-palette-btn');
  if (btn) {
    btn.addEventListener('click', () => {
      if (document.documentElement.dataset.uiMode === 'simple') return;
      openPalette(buildPaletteDeps());
    });
  }

  const isMac = isMacPlatform();
  const termContainer = document.getElementById('terminal-container');

  // Capture phase: xterm swallows bubbled keydowns, so the hotkey must be read
  // before the event reaches the terminal.
  document.addEventListener('keydown', (e) => {
    if (document.documentElement.dataset.uiMode === 'simple') return;
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
