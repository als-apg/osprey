// @ts-check
/* OSPREY Web Terminal — the hub's start-up tail
 *
 * The last statement of initPanelManager. It hands the /api/panels payload to
 * the hub surfaces outside the panel stack, then marks the hub ready. One
 * round trip feeds them all; each module owns its own rule, and a failed
 * fetch (null) leaves each in its own fallback.
 */

import { applyConfigTabGate, applyConfigUnreadableNotice } from './config-tab.js';
import { applyScaffoldWriteGate } from './scaffold/write-gate.js';
import { applyLoginNudge } from './login-nudge.js';
import { applyTourConfig } from './tour.js';
import { setFacts } from './first-contact.js';

/**
 * Apply the server's facts to the drawer, tour, nudge and first contact, and
 * set `html[data-panels-ready]`. Call it once, after the rail is rendered,
 * every sidebar control is wired and the panel event stream is open.
 * @param {any} panelConfig - the /api/panels payload, or null when the fetch failed
 */
export function finishStartup(panelConfig) {
  // The settings drawer's Config tab, withdrawn where the deployment gated its
  // server surface off (web.config_panel.enabled); when an unreadable config
  // file closed the panel, a drawer notice names that file.
  applyConfigTabGate(panelConfig);
  applyConfigUnreadableNotice(panelConfig);

  // Whether the Scaffold gallery may write (web.scaffold_gallery.write_enabled).
  // The gallery renders lazily and reads the posture back at render time.
  applyScaffoldWriteGate(panelConfig);

  // The onboarding tour's invite policy and capability list; a failed fetch
  // leaves the tour on-demand only.
  applyTourConfig(panelConfig);

  // The personal-login nudge: shown only when the deployment allows a personal
  // login and this terminal holds none.
  applyLoginNudge(panelConfig);

  // Unlike the reads above, a failed fetch is NOT a no-op here: the greeting
  // and its starter prompts render either way, so they must be told the facts
  // are in — with nothing in them — or they would wait forever.
  setFacts(panelConfig ? panelConfig.tour : null);

  // Start-up is over. The page ships rail rows of its own, so a row on screen
  // does not say this; anything that drives the hub (a browser test, a docs
  // recording) waits here.
  document.documentElement.dataset.panelsReady = 'true';
}
