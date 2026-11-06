// @ts-check
/* OSPREY Web Terminal — Command Palette Registry Builder
 *
 * Pure, dependency-injected builder that turns live app data into a flat,
 * grouped list of command-palette items (Settings files / Settings / Panels /
 * Layouts / Actions, in that order). Everything it needs — the /api/config
 * snapshot, the Settings drawer's file list, panel/preset getters, action
 * closures, and the navigation callbacks (revealArtifact / revealSetting /
 * showPanel / focusPanel / applyPreset) — is passed in via
 * `deps`, so this module imports nothing, touches no DOM, and is unit-testable
 * in isolation. The caller (palette.js) rebuilds the registry each time the
 * palette opens, then fuzzy-filters the returned items by their `searchText`.
 *
 * It also builds the Sessions and Artifacts async providers (the palette's
 * groups that answer from the session model and the artifacts panel), with
 * the same rule: every reader and verb is injected.
 */

/**
 * A navigable palette item, or a non-navigable Settings status decoration.
 *
 * Navigable items carry `run` (invoked on selection) and `searchText` (scored
 * by the fuzzy matcher). Panel rows also carry `names`: the whole words that
 * name their panel (label, id, domain synonyms), lower-cased, so a query that
 * IS one of them can put the Panels group first (palette.js). Settings status rows are static decorations with
 * neither — the UI renders them and excludes them from keyboard navigation.
 *
 * @typedef {(
 *   {
 *     group: 'Settings files' | 'Settings' | 'Panels' | 'Layouts' | 'Actions',
 *     label: string,
 *     detail?: string,
 *     searchText: string,
 *     names?: string[],
 *     run: () => void,
 *   }
 *   |
 *   {
 *     group: 'Settings files' | 'Settings',
 *     status: 'loading' | 'error',
 *     label: string,
 *   }
 * )} Item
 */

/**
 * Injected dependencies. Every field is provided by the caller; optional
 * getters may be absent and are treated as empty. No field is read from any
 * global or module import — that is what keeps this builder pure.
 *
 * @typedef {{
 *   config?: (
 *     { state: 'ok', sections: Record<string, unknown> }
 *     | { state: 'loading' }
 *     | { state: 'error' }
 *   ),
 *   artifacts?: (
 *     { state: 'ok', entries: Array<SettingsFile> }
 *     | { state: 'loading' }
 *     | { state: 'error' }
 *   ),
 *   getHiddenPanels?: () => Array<{ id: string, label: string }>,
 *   getVisiblePanels?: () => Array<{ id: string, label: string }>,
 *   getPopoutPanels?: () => Array<{ id: string, label: string }>,
 *   getPresets?: () => Array<{ name: string, panels: string[] }>,
 *   actions?: Array<{ label: string, detail?: string, run: () => void }>,
 *   showPanel?: (id: string) => void,
 *   focusPanel?: (id: string) => void,
 *   popoutPanel?: (id: string) => void,
 *   openPanelBeside?: (id: string) => void,
 *   applyPreset?: (name: string) => void,
 *   resetLayout?: () => void,
 *   revealSetting?: (dotKey: string) => void,
 *   revealArtifact?: (name: string, tabId: string) => void,
 *   loadArtifacts?: () => Promise<Array<SettingsFile>>,
 * }} PaletteDeps
 */

/**
 * One file the Settings drawer's galleries show, as the gallery labels it.
 * @typedef {{
 *   name: string,
 *   label: string,
 *   category: string,
 *   tabId: string,
 *   tabLabel: string,
 *   description: string,
 * }} SettingsFile
 */

/**
 * Domain vocabulary for the built-in panels — the words an operator reaches
 * for when they do not have the catalog label in mind ("elog" for the Logbook,
 * "pv" for Channel Finder). Folded into the searchText of EVERY row for that
 * panel, so any of its verbs is reachable by the domain word. Keyed by catalog
 * panel id; ids absent here (custom facility panels) contribute no aliases.
 * @type {Record<string, string>}
 */
const PANEL_SYNONYMS = {
  ariel: 'logbook elog',
  'channel-finder': 'pv channels',
  artifacts: 'gallery files',
  okf: 'knowledge docs',
  'system-health': 'status monitoring',
  canopy: 'screens gui operator',
};

/** Verb vocabulary for "Open … in new window" rows. */
const POPOUT_ALIASES = 'window popout pop out standalone';

/** Verb vocabulary for "Open … in a new tile" rows. */
const BESIDE_ALIASES = 'tile beside split';

/**
 * searchText for a panel row: the verb, the panel's own label and id, the
 * verb's aliases, then the panel's domain aliases. Only searchText is scored
 * by the matcher (labels are not), so every alias a row should answer to has
 * to be present here.
 *
 * @param {string} verb
 * @param {{ id: string, label: string }} panel
 * @param {string} [verbAliases]
 * @returns {string}
 */
function panelSearchText(verb, panel, verbAliases) {
  return [verb, panel.label, panel.id, verbAliases, PANEL_SYNONYMS[panel.id]]
    .filter(Boolean)
    .join(' ');
}

/**
 * The whole words that name a panel: its label, its id and its domain
 * synonyms, lower-cased. A query equal to one of them names the panel.
 *
 * @param {{ id: string, label: string }} panel
 * @returns {string[]}
 */
function panelNames(panel) {
  const synonyms = (PANEL_SYNONYMS[panel.id] ?? '').split(' ');
  return [panel.label, panel.id, ...synonyms].filter(Boolean).map((name) => name.toLowerCase());
}

/**
 * Invoke an optional getter defensively, always returning an array. Absent or
 * non-function getters, and getters that return a non-array, yield [].
 *
 * @template T
 * @param {(() => Array<T>) | undefined} getter
 * @returns {Array<T>}
 */
function safeList(getter) {
  if (typeof getter !== 'function') {
    return [];
  }
  const out = getter();
  return Array.isArray(out) ? out : [];
}

/**
 * Recursively flatten a config `sections` tree into leaf dot-paths. Nested
 * plain-object keys are joined with '.'; arrays and scalars are treated as
 * leaves (never recursed into). Intermediate object nodes are not emitted —
 * only the paths that terminate at a non-object leaf. Already-dotted flat keys
 * (leaf values) fall through as-is.
 *
 * @param {Record<string, unknown>} sections
 * @returns {string[]} leaf dot-keys, in source (insertion) order
 */
function flattenSections(sections) {
  /** @type {string[]} */
  const keys = [];

  /**
   * @param {string} prefix
   * @param {unknown} value
   */
  const walk = (prefix, value) => {
    if (value !== null && typeof value === 'object' && !Array.isArray(value)) {
      for (const [k, v] of Object.entries(value)) {
        walk(prefix ? `${prefix}.${k}` : k, v);
      }
      return;
    }
    // Leaf: scalar, null, or array.
    keys.push(prefix);
  };

  for (const [k, v] of Object.entries(sections)) {
    walk(k, v);
  }
  return keys;
}

/**
 * Build the Settings group from the config snapshot. Loading/error states each
 * emit a single non-navigable decoration; the ok state emits one navigable
 * Item per leaf dot-key, whose `run` calls the injected `revealSetting`.
 *
 * @param {PaletteDeps} deps
 * @returns {Item[]}
 */
function buildSettings(deps) {
  const config = deps.config;
  if (!config || config.state === 'loading') {
    return [{ group: 'Settings', status: 'loading', label: 'Loading settings…' }];
  }
  if (config.state === 'error') {
    return [{ group: 'Settings', status: 'error', label: 'Settings unavailable' }];
  }

  const sections = config.sections && typeof config.sections === 'object' ? config.sections : {};
  const revealSetting = deps.revealSetting;
  return flattenSections(sections).map((dotKey) => ({
    group: 'Settings',
    label: dotKey,
    searchText: dotKey,
    run: () => {
      if (typeof revealSetting === 'function') {
        revealSetting(dotKey);
      }
    },
  }));
}

/**
 * Build the Settings files group: one row per file the drawer's galleries
 * show, labelled as its card is, whose `run` opens it through the injected
 * `revealArtifact`. Absent `revealArtifact` (no galleries to land in) there is
 * no group at all; loading/error emit a single decoration like Settings.
 *
 * searchText leads with the canonical name: the matcher's word boundaries are
 * `.`, `_` and `-`, so `code-review/rules` first is what ranks the rules row
 * top for "code review".
 *
 * @param {PaletteDeps} deps
 * @returns {Item[]}
 */
function buildSettingsFiles(deps) {
  const revealArtifact = deps.revealArtifact;
  if (typeof revealArtifact !== 'function') return [];
  const artifacts = deps.artifacts;
  if (!artifacts || artifacts.state === 'loading') {
    return [{ group: 'Settings files', status: 'loading', label: 'Loading settings files…' }];
  }
  if (artifacts.state === 'error') {
    return [{ group: 'Settings files', status: 'error', label: 'Settings files unavailable' }];
  }
  const entries = Array.isArray(artifacts.entries) ? artifacts.entries : [];
  return entries.map((entry) => ({
    group: /** @type {const} */ ('Settings files'),
    label: entry.label,
    detail: [entry.tabLabel, entry.category].filter(Boolean).join(' · '),
    searchText: [entry.name, entry.label, entry.category, entry.tabLabel, entry.description]
      .filter(Boolean)
      .join(' '),
    run: () => revealArtifact(entry.name, entry.tabId),
  }));
}

/**
 * Build the Panels group: a "Show <label>" item per hidden panel, a
 * "Focus <label>" item per visible (non-active) panel, an "Open <label> in
 * new window" item (the standalone pop-out, popout.js) per pop-out-able panel, and an "Open <label> in a new tile" item per
 * visible (non-active) panel.
 *
 * The pop-out and new-tile verbs draw on DIFFERENT sources on purpose, and
 * the difference is the whole point of FR8:
 *
 *   - pop out ← `getPopoutPanels`, which INCLUDES the active panel (popping
 *     out the panel you are looking at is the verb's commonest use) and is
 *     already filtered to members that can pop out: a standalone URL has
 *     resolved and no window is open for them yet.
 *   - new tile ← `getVisiblePanels`, which EXCLUDES the active panel (opening
 *     it beside itself is a guaranteed no-op), and carries no URL gate — the
 *     rail corner this verb replaces never had one.
 *
 * The new-tile rows are dropped wholesale when `openPanelBeside` is absent —
 * the registry's one lever for a host with nowhere to place a second tile.
 *
 * @param {PaletteDeps} deps
 * @returns {Item[]}
 */
function buildPanels(deps) {
  const showPanel = deps.showPanel;
  const focusPanel = deps.focusPanel;
  const popoutPanel = deps.popoutPanel;
  const openPanelBeside = deps.openPanelBeside;

  /** @type {Item[]} */
  const items = [];

  for (const panel of safeList(deps.getHiddenPanels)) {
    const id = panel.id;
    items.push({
      group: 'Panels',
      label: `Show ${panel.label}`,
      searchText: panelSearchText('show', panel),
      names: panelNames(panel),
      run: () => {
        if (typeof showPanel === 'function') {
          showPanel(id);
        }
      },
    });
  }

  for (const panel of safeList(deps.getVisiblePanels)) {
    const id = panel.id;
    items.push({
      group: 'Panels',
      label: `Focus ${panel.label}`,
      searchText: panelSearchText('focus', panel),
      names: panelNames(panel),
      run: () => {
        if (typeof focusPanel === 'function') {
          focusPanel(id);
        }
      },
    });
  }

  for (const panel of safeList(deps.getPopoutPanels)) {
    const id = panel.id;
    items.push({
      group: 'Panels',
      label: `Open ${panel.label} in new window`,
      // The label's words go last: a fuzzy query ("elog") must not thread
      // through "open" into another panel's synonyms.
      searchText: `${panelSearchText('window', panel, POPOUT_ALIASES)} open in new window`,
      names: panelNames(panel),
      run: () => {
        if (typeof popoutPanel === 'function') {
          popoutPanel(id);
        }
      },
    });
  }

  if (typeof openPanelBeside === 'function') {
    for (const panel of safeList(deps.getVisiblePanels)) {
      const id = panel.id;
      items.push({
        group: 'Panels',
        label: `Open ${panel.label} in a new tile`,
        searchText: panelSearchText('open', panel, BESIDE_ALIASES),
        names: panelNames(panel),
        run: () => {
          openPanelBeside(id);
        },
      });
    }
  }

  return items;
}

/**
 * Build the Layouts group: one item per preset, whose `run` applies that preset
 * BY NAME via the injected `applyPreset` — the same one-call path the "+"
 * menu's Layouts section uses, so both surfaces produce the same arrangement —
 * then a "Reset layout" item that restores the current view's default
 * arrangement through the injected `resetLayout`. A host with no layout to
 * reset withholds `resetLayout`, and the row goes with it.
 *
 * @param {PaletteDeps} deps
 * @returns {Item[]}
 */
function buildLayouts(deps) {
  const applyPreset = deps.applyPreset;
  /** @type {Item[]} */
  const items = safeList(deps.getPresets).map((preset) => ({
    group: /** @type {const} */ ('Layouts'),
    label: `Layout: ${preset.name}`,
    searchText: preset.name,
    run: () => {
      if (typeof applyPreset === 'function') {
        applyPreset(preset.name);
      }
    },
  }));
  const resetLayout = deps.resetLayout;
  if (typeof resetLayout === 'function') {
    items.push({
      group: 'Layouts',
      label: 'Reset layout',
      searchText: 'reset layout default arrangement restore tiles',
      run: () => resetLayout(),
    });
  }
  return items;
}

/**
 * Build the Actions group by wrapping each injected action closure, preserving
 * source order. Each entry's own `run` is passed straight through.
 *
 * @param {PaletteDeps} deps
 * @returns {Item[]}
 */
function buildActions(deps) {
  const actions = Array.isArray(deps.actions) ? deps.actions : [];
  return actions.map((action) => {
    /** @type {Item} */
    const item = {
      group: 'Actions',
      label: action.label,
      searchText: action.label,
      run: action.run,
    };
    if (action.detail !== undefined) {
      item.detail = action.detail;
    }
    return item;
  });
}

/**
 * Build the full command-palette registry from injected app data. Returns a
 * flat array grouped Settings files → Settings → Panels → Layouts → Actions;
 * within each group, source order is preserved. Deterministic for a given
 * input, and never throws on absent optional deps (missing getters default to
 * empty).
 *
 * @param {PaletteDeps} deps
 * @returns {Item[]}
 */
export function buildRegistry(deps) {
  const d = deps || {};
  return [
    ...buildSettingsFiles(d),
    ...buildSettings(d),
    ...buildPanels(d),
    ...buildLayouts(d),
    ...buildActions(d),
  ];
}

// ---- Async providers: Sessions and Artifacts ----

/**
 * One row an async provider answers: rendered in the provider's order, with
 * neither fuzzy ranking nor highlight.
 * @typedef {{ label: string, detail?: string, run: () => void }} ProviderHit
 */

/**
 * An async palette group. `start(onChange)` runs once per open; `search(q)`
 * answers the rows for a non-empty query, or null while they are loading,
 * and calls the open's `onChange` when a pending answer lands.
 * @typedef {{
 *   group: string,
 *   search: (q: string) => ProviderHit[] | null,
 *   start: (onChange: () => void) => void,
 * }} AsyncProvider
 */

/**
 * A session row as the session model lists it (the fields the palette reads).
 * @typedef {{
 *   transcript_id: string | null,
 *   session_key: string | null,
 *   title?: string,
 *   current_for_key?: boolean,
 *   synthesized?: boolean,
 * }} PaletteSessionRow
 */

/**
 * An artifact as the artifacts panel lists it (the fields the palette reads).
 * @typedef {{
 *   id: string,
 *   title?: string,
 *   filename?: string,
 *   artifact_type?: string,
 *   form?: string | null,
 *   form_label?: string | null,
 *   owner?: string | null,
 *   exec_id?: string | null,
 *   run_part?: string | null,
 * }} PaletteArtifact
 */

/** Most rows the Sessions group shows. */
export const SESSION_HIT_CAP = 8;

/** Most rows the Artifacts group shows. */
export const ARTIFACT_HIT_CAP = 8;

/** Most title-matched sessions one batched `session-list` asks for. */
export const SESSION_BATCH_CAP = 5;

/** Quiet time after the last keystroke before the Artifacts read fires. */
export const ARTIFACT_SEARCH_DEBOUNCE_MS = 200;

/** The title a session row shows. */
const sessionTitle = (/** @type {PaletteSessionRow} */ row) => row.title || 'Untitled session';

/**
 * The session rows whose title contains `q` (case-insensitive), in the
 * model's order (newest first).
 * @param {PaletteSessionRow[]} rows
 * @param {string} q
 * @returns {PaletteSessionRow[]}
 */
function sessionsTitled(rows, q) {
  const needle = q.trim().toLowerCase();
  if (!needle) return [];
  return rows.filter((row) => sessionTitle(row).toLowerCase().includes(needle));
}

/**
 * The "Sessions" group: the operator's loaded sessions matched by title on
 * the client. Picking one enters it through the guarded `switchTo`.
 * @param {{
 *   rows: () => PaletteSessionRow[],
 *   switchTo: (row: PaletteSessionRow, options: { reason: string }) => unknown,
 * }} deps
 * @returns {AsyncProvider}
 */
export function createSessionsProvider(deps) {
  return {
    group: 'Sessions',
    start: () => {},
    search(q) {
      return sessionsTitled(safeRows(deps.rows), q)
        .slice(0, SESSION_HIT_CAP)
        .map((row) => ({
          label: sessionTitle(row),
          run: () => void deps.switchTo(row, { reason: 'gesture' }),
        }));
    },
  };
}

/** @param {() => PaletteSessionRow[]} rows */
function safeRows(rows) {
  try {
    const list = rows();
    return Array.isArray(list) ? list : [];
  } catch {
    return [];
  }
}

/**
 * Collapse a run to one hit: a run's part (its preview, code or result,
 * anything but a listed output) gives way to its head, and a part whose head
 * did not match stands in for the run once. First seen order is kept; a head
 * arriving after its part takes the part's place.
 * @param {PaletteArtifact[]} entries
 * @returns {PaletteArtifact[]}
 */
export function collapseRuns(entries) {
  /** @type {PaletteArtifact[]} */
  const out = [];
  /** @type {Map<string, number>} */
  const runAt = new Map();
  const seen = new Set();
  for (const entry of entries) {
    if (!entry || typeof entry.id !== 'string' || seen.has(entry.id)) continue;
    seen.add(entry.id);
    const isHead = entry.form === 'runs';
    const isPart = !isHead && !!entry.exec_id && !!entry.run_part && entry.run_part !== 'output';
    if (!isHead && !isPart) {
      out.push(entry);
      continue;
    }
    const key = /** @type {string} */ (entry.exec_id || entry.id);
    const at = runAt.get(key);
    if (at === undefined) {
      runAt.set(key, out.length);
      out.push(entry);
    } else if (isHead && out[at].form !== 'runs') {
      out[at] = entry;
    }
  }
  return out;
}

/**
 * The "Artifacts" group: the panel's search (title, file name, description,
 * type and form label; a run once, at its head) plus the artifacts of the
 * first ≤ {@link SESSION_BATCH_CAP} sessions whose title matches, newest
 * first, in one batched `session-list`. The read is debounced and the newest
 * query wins; until it lands the group shows "Loading…". A failed read
 * contributes no rows.
 *
 * Opening a hit owned by a loaded session enters it (`switchTo` with the
 * artifact as the pick); one with no session is shown and the session stays.
 *
 * @param {{
 *   rows: () => PaletteSessionRow[],
 *   searchArtifacts: (q: string) => Promise<PaletteArtifact[]>,
 *   listSessions: (transcriptIds: string[]) => Promise<PaletteArtifact[]>,
 *   sessionOfArtifact: (artifact: PaletteArtifact) => PaletteSessionRow | null,
 *   switchTo: (
 *     row: PaletteSessionRow,
 *     options: { artifactId: string, reason: string },
 *   ) => unknown,
 *   showArtifact: (artifact: PaletteArtifact) => unknown,
 *   debounceMs?: number,
 * }} deps
 * @returns {AsyncProvider}
 */
export function createArtifactsProvider(deps) {
  const wait = deps.debounceMs ?? ARTIFACT_SEARCH_DEBOUNCE_MS;
  /** @type {() => void} */
  let onChange = () => {};
  /** @type {{ q: string, hits: ProviderHit[] } | null} */
  let answer = null;
  /** @type {string | null} */
  let pendingQ = null;
  /** @type {ReturnType<typeof setTimeout> | null} */
  let timer = null;
  let seq = 0;

  /** @param {PaletteArtifact} artifact */
  function open(artifact) {
    const row = deps.sessionOfArtifact(artifact);
    if (row) void deps.switchTo(row, { artifactId: artifact.id, reason: 'gesture' });
    else void deps.showArtifact(artifact);
  }

  /**
   * @param {PaletteArtifact} artifact
   * @returns {ProviderHit}
   */
  function hitOf(artifact) {
    const row = deps.sessionOfArtifact(artifact);
    const kind = artifact.form_label || artifact.artifact_type || '';
    const detail = [kind, row ? sessionTitle(row) : ''].filter(Boolean).join(' · ');
    /** @type {ProviderHit} */
    const hit = {
      label: artifact.title || artifact.filename || artifact.id,
      run: () => open(artifact),
    };
    if (detail) hit.detail = detail;
    return hit;
  }

  /** @param {string} q */
  async function read(q) {
    const batch = sessionsTitled(safeRows(deps.rows), q)
      .map((row) => row.transcript_id)
      .filter((id) => typeof id === 'string' && id)
      .slice(0, SESSION_BATCH_CAP);
    const [found, owned] = await Promise.all([
      deps.searchArtifacts(q).catch(() => []),
      batch.length
        ? deps.listSessions(/** @type {string[]} */ (batch)).catch(() => [])
        : Promise.resolve([]),
    ]);
    const all = [...(Array.isArray(found) ? found : []), ...(Array.isArray(owned) ? owned : [])];
    return collapseRuns(all).slice(0, ARTIFACT_HIT_CAP).map(hitOf);
  }

  /** @param {string} q */
  function schedule(q) {
    if (timer) clearTimeout(timer);
    pendingQ = q;
    const token = ++seq;
    timer = setTimeout(() => {
      timer = null;
      read(q)
        .catch(() => [])
        .then((hits) => {
          if (token !== seq) return;
          pendingQ = null;
          answer = { q, hits };
          onChange();
        });
    }, wait);
  }

  return {
    group: 'Artifacts',
    start(change) {
      onChange = change;
      answer = null;
      pendingQ = null;
      if (timer) clearTimeout(timer);
      timer = null;
      seq++;
    },
    search(q) {
      const needle = q.trim();
      if (!needle) return [];
      if (answer && answer.q === needle) return answer.hits;
      if (pendingQ !== needle) schedule(needle);
      return null;
    },
  };
}
