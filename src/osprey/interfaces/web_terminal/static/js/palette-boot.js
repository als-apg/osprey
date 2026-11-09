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
 * The palette offers settings search, the Settings drawer's files, the panel verbs including "Open … in a
 * new tile", the Layouts group, popping a panel out to its own window, the
 * drawer tabs, the safety reference and opening the activity log in a new tab.
 * It has no terminal entries: the terminal is reached from the Agent tile alone.
 *
 * After CANOPY come the Sessions and Artifacts groups: a session enters
 * through the session model's guarded `switchTo`; an artifact owned by a
 * loaded session enters that session with the artifact shown, and one with no
 * session is shown while the session stays.
 */

import { getCurrentSessionId } from './terminal.js';
import { openActivityLog } from './activity-log-link.js';
import {
  getHiddenPanels,
  getVisiblePanels,
  getPopoutPanels,
  getPresets,
  showPanel,
  activateTab,
  applyMenuPreset,
  navigateAndActivatePanel,
} from './panel-manager.js';
import { detach, reattach } from './pane-window.js';
import { isDetached } from './dock-apply.js';
import { PANELS } from './panel-catalog.js';
import { CANOPY_PANEL_ID, createCanopyProvider } from './canopy-provider.js';
import { openPanelBeside } from './panel-placement.js';
import { resetDockLayout } from './dock-workspace.js';
import { openDrawerTab, revealSetting } from './settings.js';
import { CONFIG_TAB_ID } from './config-tab.js';
import { scaffoldGalleryMounted, listGalleryEntries, revealArtifact } from './scaffold-gallery.js';
import { REVIEW_TAB_ID, reviewPaletteEntries } from './code-review-tab.js';
import { startTour } from './tour.js';
import { openPalette, closePalette, isOpen } from './palette.js';
import { confirmSignOut } from './sign-out.js';
import { isMacPlatform } from '/design-system/js/platform.js';
import { isFeedbackModalOpen } from './feedback-modal.js';
import {
  ARTIFACTS_BASE,
  SESSION_LIST_PATH,
  postJSON,
  requestJSON,
} from './sidebar-artifact-listing.js';
import { rows as sessionRows, sessionOfArtifact, setPick, switchTo } from './session-model.js';
import { recordArtifactOpened } from './sidebar-artifacts.js';
import {
  ARTIFACT_HIT_CAP,
  createArtifactsProvider,
  createSessionsProvider,
} from './palette-registry.js';

/**
 * The one CANOPY provider, shared across opens so its newest-read-wins
 * sequence spans them. Created on first use.
 * @type {ReturnType<typeof createCanopyProvider> | null}
 */
let canopyProvider = null;

/** The artifacts panel's id and its mount on the hub. */
const ARTIFACTS_PANEL_ID = 'artifacts';

/**
 * The Sessions and Artifacts providers, shared across opens. Created on first use.
 * @type {{ sessions: import('./palette-registry.js').AsyncProvider,
 *   artifacts: import('./palette-registry.js').AsyncProvider } | null}
 */
let sessionProviders = null;

/**
 * The panel's search: one page, a run once at its head. It does not hide
 * dispatch or watchdog artifacts.
 * @param {string} q
 */
async function searchArtifacts(q) {
  const params = new URLSearchParams({
    search: q,
    runs: 'grouped',
    limit: String(ARTIFACT_HIT_CAP),
  });
  const body = await requestJSON(`${ARTIFACTS_BASE}/api/artifacts?${params}`);
  return Array.isArray(body?.artifacts) ? body.artifacts : [];
}

/**
 * The first page of each named session's artifacts, in one batched request.
 * @param {string[]} sessions
 */
async function listSessions(sessions) {
  const body = await postJSON(SESSION_LIST_PATH, {
    sessions,
    runs: 'grouped',
    limit: ARTIFACT_HIT_CAP,
  });
  const pages = body?.pages && typeof body.pages === 'object' ? body.pages : {};
  return sessions.flatMap((id) => (Array.isArray(pages[id]?.artifacts) ? pages[id].artifacts : []));
}

/**
 * Show an artifact without changing the session: it becomes the pick and the
 * artifacts panel opens on it.
 * @param {{ id: string }} artifact
 */
async function showArtifact(artifact) {
  await recordArtifactOpened(artifact.id);
  setPick(artifact.id);
  navigateAndActivatePanel(ARTIFACTS_PANEL_ID, '');
}

/** @returns {NonNullable<typeof sessionProviders>} */
function getSessionProviders() {
  if (!sessionProviders) {
    sessionProviders = {
      sessions: createSessionsProvider({ rows: sessionRows, switchTo }),
      artifacts: createArtifactsProvider({
        rows: sessionRows,
        searchArtifacts,
        listSessions,
        sessionOfArtifact,
        switchTo,
        showArtifact,
      }),
    };
  }
  return sessionProviders;
}

/** Whether this deployment serves the artifacts panel. */
function artifactsPanelPresent() {
  return [...getHiddenPanels(), ...getVisiblePanels(), ...getPopoutPanels()].some(
    (p) => p.id === ARTIFACTS_PANEL_ID
  );
}

/**
 * Whether this deployment carries the CANOPY panel. Without it there is nothing
 * to search and nowhere to open a row. Hidden and visible panels exclude the
 * active one; the new-window list includes it once its URL has resolved, which is
 * also the earliest the panel could show a row's target.
 * @returns {boolean}
 */
function canopyPanelPresent() {
  return [...getHiddenPanels(), ...getVisiblePanels(), ...getPopoutPanels()].some(
    (p) => p.id === CANOPY_PANEL_ID
  );
}

/**
 * Whether this deployment still carries the settings drawer's Config tab.
 *
 * Element presence IS the contract: `config-tab.js` removes the tab and its
 * panel when `GET /api/panels` reports `config_panel_enabled: false`. Read at
 * every open, never cached — the removal lands one round trip after boot, and
 * a bundle built before it must not outlive it.
 *
 * @returns {boolean}
 */
function configTabPresent() {
  return !!document.getElementById(CONFIG_TAB_ID);
}

/**
 * Assemble the live dependency bundle `openPalette` needs. Built fresh on each
 * open because panel visibility, presets and the config snapshot all drift over a session — the palette must reflect the CURRENT state, not
 * boot state. The action closures are the vetted mechanisms (gated drawer tabs, new-tab safety nav) — do not collapse or reorder them
 * casually.
 *
 * Typed against the REGISTRY's dep bundle rather than palette.js's OpenDeps:
 * everything assembled here exists to reach `buildRegistry`, and PaletteDeps is
 * the declaration site for the panel-verb deps. openPalette accepts it as-is.
 * The CANOPY search seam (`searchCanopy` / `startCanopy`) is palette.js's own,
 * not the registry's, so it is typed from OpenDeps.
 * @returns {import('./palette-registry.js').PaletteDeps & Pick<import('./palette.js').OpenDeps, 'searchCanopy' | 'startCanopy' | 'asyncProviders'>}
 */
function buildPaletteDeps() {
  /** @type {Array<{ label: string, detail?: string, run: () => void }>} */
  const actions = [];

  // Drawer tabs always go THROUGH openDrawerTab's warning gate.
  // The Config tab is server-gated (web.config_panel.enabled): config-tab.js
  // removes the tab at boot where the deployment withdrew it, and the routes
  // behind it answer 403. Offering the row there would put the operator
  // through the warning gate to reach a tab openDrawerTab cannot resolve —
  // dead, and misleading about what this deployment permits. Read live, per
  // open, the same way the logout row below reads its server-gated button.
  if (configTabPresent()) {
    actions.push({
      label: 'Open Settings',
      run: () => {
        openDrawerTab('tab-config');
      },
    });
  }
  actions.push({
    label: 'Open Memory gallery',
    run: () => {
      openDrawerTab('tab-memory');
    },
  });
  actions.push({
    label: 'Open Prompt gallery',
    run: () => {
      openDrawerTab('tab-behavior');
    },
  });
  // On-demand tour entry — works regardless of the web.tour invite policy.
  actions.push({ label: 'Take the tour', run: () => startTour() });
  // It opens on the session the card is on when picked.
  actions.push({
    label: 'Open activity log',
    detail: 'new tab',
    run: () => openActivityLog(getCurrentSessionId),
  });
  // Logout only exists in multi-user deployments: the server stamps the
  // landing URL on <html> when it renders a logout at all. Read from there,
  // not from a logout button, so this row does not depend on any chrome. It
  // asks first, like every other way to sign out.
  const landingUrl = document.documentElement.dataset.landingUrl;
  if (landingUrl) {
    actions.push({
      label: 'Log out',
      run: () => confirmSignOut({ landingUrl }),
    });
  }

  /** @type {import('./palette-registry.js').PaletteDeps & Pick<import('./palette.js').OpenDeps, 'searchCanopy' | 'startCanopy' | 'asyncProviders'>} */
  const deps = {
    // Panel show/focus is exactly what a rail click does, and the rail is
    // present in both modes.
    getHiddenPanels,
    getVisiblePanels,
    showPanel,
    // "Focus" reports as a user-initiated activation so the server sees it.
    focusPanel: (id) => activateTab(id, { userInitiated: true }),
    // getPopoutPanels is already narrowed to rail members that can leave for a
    // window and are not in one — the context menu's row reads the same way.
    getPopoutPanels,
    popoutPanel: detach,
    getDetachedPanels: () =>
      PANELS.filter((p) => isDetached(p.id)).map((p) => ({ id: p.id, label: p.label })),
    reattachPanel: (id) => {
      reattach(id);
    },
    actions,
  };

  // The Settings GROUP (a row per config dot-key) exists only to reach
  // revealSetting, which opens the Config tab — so the dep is withheld with
  // the tab, and palette.js drops the group AND its /api/config read rather
  // than rendering rows that jump nowhere against a surface answering 403.
  // Omitting the dep is what drops the rows.
  if (configTabPresent()) deps.revealSetting = revealSetting;

  // The Settings files group holds a row per gallery card, which reaches the
  // drawer's galleries, and the Review tab's rules row, which opens that tab.
  // The gallery rows and their /api/scaffold read come only where a gallery
  // mounted, the rules row only while the Review tab is in the drawer; with
  // neither, both deps are withheld and the group goes with them.
  const galleries = scaffoldGalleryMounted();
  const reviewRows = reviewPaletteEntries();
  if (galleries || reviewRows.length) {
    deps.loadArtifacts = async () => [
      ...(galleries ? await listGalleryEntries() : []),
      ...reviewRows,
    ];
    deps.revealArtifact = (name, tabId) => {
      if (tabId === REVIEW_TAB_ID) void openDrawerTab(tabId);
      else void revealArtifact(name, tabId);
    };
  }

  // "Open … in a new tile" and the Layouts group place tiles; both views hold
  // as many tiles as the operator wants, so both are offered in both.
  deps.openPanelBeside = openPanelBeside;
  deps.getPresets = getPresets;
  deps.applyPreset = applyMenuPreset;
  deps.resetLayout = resetDockLayout;

  // The CANOPY group: screens and collections, opened in place in the CANOPY
  // panel through the same navigate-and-activate path the agent's panel_focus
  // uses. Offered only where the deployment has that panel.
  if (canopyPanelPresent()) {
    if (!canopyProvider) {
      canopyProvider = createCanopyProvider({
        navigate: (panelId, url) => navigateAndActivatePanel(panelId, url),
      });
    }
    deps.searchCanopy = canopyProvider.search;
    deps.startCanopy = canopyProvider.start;
  }

  // Sessions, then Artifacts (only where the artifacts panel is served).
  const providers = getSessionProviders();
  deps.asyncProviders = [providers.sessions];
  if (artifactsPanelPresent()) deps.asyncProviders.push(providers.artifacts);

  return deps;
}

/**
 * Wire the header trigger + Cmd/Ctrl+K hotkey for the command palette. Degrades
 * gracefully if the button is absent.
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
  document.addEventListener(
    'keydown',
    (e) => {
      if (e.key !== 'k' && e.key !== 'K') return;
      // Keyboard arbitration with the feedback modal: while it is open, Cmd/Ctrl+K
      // is its territory (e.g. a text field inside it), not the palette's. Bail
      // without preventDefault/stopPropagation so nothing here is consumed.
      if (isFeedbackModalOpen()) return;

      let match;
      if (isMac) {
        // macOS: Cmd+K from anywhere in the top document.
        match = e.metaKey;
      } else {
        // Non-macOS: Ctrl+K, but only OUTSIDE the terminal — inside, Ctrl+K stays
        // readline kill-line. The header button is the in-terminal fallback.
        const active = document.activeElement;
        const inTerminal = !!(
          termContainer &&
          active instanceof Node &&
          termContainer.contains(active)
        );
        match = e.ctrlKey && !inTerminal;
      }
      if (!match) return;

      e.preventDefault();
      e.stopPropagation();
      if (isOpen()) closePalette();
      else openPalette(buildPaletteDeps());
    },
    true
  );
}
