// @ts-check
/* OSPREY Web Terminal — Dock State Sync
 *
 * The bridge between the dockview workspace and the server-owned panel state.
 * The server is authoritative: every panel show/hide/focus lives on the server
 * and is broadcast over the panel_visibility / panel_focus / panel_register SSE
 * echo, so an agent MCP call and a human dock gesture are indistinguishable
 * downstream (see panel-commands.js / panel-manager.js). This module carries the
 * REVERSE half — turning a human's dockview gestures back into the same
 * server POSTs — without letting the server's own echo bounce back out again.
 *
 * TWO DIRECTIONS
 * --------------
 *  - Server → dock: panel-sse's frame dispatcher already drives the dock
 *    (activateTab → focusPanel, the visibility fallback → hidePanel, register →
 *    panel-lifecycle's addPanel, all via the dock-iframe adapter). This module
 *    does NOT duplicate
 *    that; it only guarantees those applied changes never POST back.
 *  - Dock → server: a human focusing a dock tab POSTs setPanelFocus — the only
 *    new POST source the docked model introduces. A human CLOSING a dock tab is
 *    deliberately NOT a server change: tile occupancy is per-client layout
 *    state under the launcher-rail model (rail membership is the shared state,
 *    and closing a tile does not remove the panel from the rail), so the close
 *    is routed to a locally-registered vacate handler instead.
 *
 * …plus one channel that is neither: the OCCUPANCY REPORT below tells the
 * server what is on screen without commanding anything.
 *
 * THE ECHO GUARD (core correctness property)
 * ------------------------------------------
 * dockview's onDidActivePanelChange fires for BOTH a human tab click AND a
 * programmatic setActive (the echo of a server-driven focus). We replicate
 * panel-manager's `userInitiated` guard (panel-manager.js activateTab): every
 * server-driven focus application is wrapped in withEchoSuppressed(), which
 * raises a depth counter the onDidActivePanelChange handler checks — so the
 * echo of an applied focus is recognised and never re-POSTed. (dockview's
 * Emitter.fire is synchronous, so the suppressed window covers the echo.)
 *
 * THE OCCUPANCY REPORT (deliberately NOT echo-guarded)
 * ----------------------------------------------------
 * Rail membership says which panels an operator can reach; it does not say what
 * is actually on screen. So this module also REPORTS the open service tiles in
 * reading order (POST /api/panel-layout), letting the agent read real occupancy
 * instead of inferring it. A report is not a command: the server stores it
 * last-writer-wins and broadcasts nothing.
 *
 * Unlike the focus POST, the reporter is NOT gated on the echo guard — it must
 * not be. dockview emits onDidLayoutChange on a microtask, so the event lands
 * AFTER the synchronous withEchoSuppressed window has already closed; a guard
 * check there would read 0, suppress nothing, and merely look correct.
 * Convergence comes from CONTENT DEDUPE on both ends instead: this module skips
 * a POST whose tile list matches the last one the server ACKNOWLEDGED, and the
 * route treats a report equal to stored state as a no-op. A server-applied
 * arrangement therefore settles after at most one no-op round-trip rather than
 * looping. The baseline tracks acknowledgements, not attempts, so a report that
 * never landed is retried on the next layout change instead of being deduped
 * away forever — and only ONE report is ever in flight, so the ack being waited
 * on always names the last write the server applied. Overlapping reports could
 * be applied in one order and acknowledged in the other, leaving no client-side
 * rule able to tell which one stuck; the overlap is prevented rather than
 * arbitrated.
 *
 * NO SEQUENCE GUARD, DELIBERATELY. The reflex when acks can race is to stamp
 * each request and ignore any ack that is not the newest — do not add that here.
 * It reverses the direction this code fails in. A guard makes the client trust
 * the NEWEST REQUEST's ack as the server's state; if the requests themselves
 * were reordered, the server holds the older list while the client believes the
 * newer one was accepted, and dedupe then suppresses every future report —
 * stranding the server wrong indefinitely. Trusting the LAST ACK TO ARRIVE
 * instead can only ever cause a redundant report the server deduplicates,
 * because response order follows the order writes were applied. Suppression is
 * the one failure that does not self-heal, so the design always errs toward
 * re-reporting. Serializing reports (above) makes the race unreachable in the
 * first place; this rule is what keeps it unreachable if that ever changes.
 *
 * Each report is serialized at FLUSH time from the live api, once a settle
 * debounce has expired — so a rebuild that removes and re-adds panels one at a
 * time (mode flip, reset, fromJSON restore) reports only the arrangement it
 * settles on, never an intermediate one. Note the debounce RESTARTS on every
 * event, unlike dock-workspace's schedulePersist (which fires a fixed delay
 * after the FIRST event of a burst) — settling on the final state is the whole
 * point here, so the two are deliberately not the same shape.
 *
 * THE BOOT WINDOW is the one case a debounce cannot cover. dock-workspace
 * restores the stored layout only after an /api/panels round trip, so the gap
 * between the default arrangement and the real one is a network latency, not a
 * duration this module could budget for. Reporting on any fixed deadline would,
 * on a slow fetch, publish a confident "empty workspace" that a hook could read
 * as fact. So no deadline is what starts the first report: dock-workspace
 * announces the boot layout as final (setLayoutRestoredHook, called on every
 * exit of initPersistence including the no-project-key one) and that
 * announcement is what ARMS reporting. The settle debounce still paces the send
 * once armed — it simply cannot begin before the announcement lands. Layout
 * churn before that point is ignored outright. The long deadline that remains is
 * a backstop for the announcement never arriving.
 *
 * WHY CLOSE USES A CLICK, NOT onDidRemovePanel
 * --------------------------------------------
 * A placeholder panel is removed both when a human closes its tab AND during
 * every programmatic layout rebuild (mode flip, reset, expert-layout restore —
 * all api.clear()/fromJSON in dock-workspace.js, a file this module does not own
 * and cannot wrap in the suppress guard). onDidRemovePanel therefore cannot tell
 * a human close from a rebuild, and treating a rebuild's removals as closes would
 * silently hide every panel on the server. A capture-phase click on the tile
 * bar's close control (`.tile-tab-close`, dock-tab.js — `.dv-default-tab-action`
 * kept as a fallback for any dockview-rendered default tab) fires ONLY on a
 * genuine human gesture, never during a rebuild, so it is the unambiguous close
 * signal. Capture phase resolves the panel id before dockview's own handler
 * removes the panel.
 *
 * Only SERVICE panels participate: their dock ids carry the adapter's `iframe:`
 * placeholder prefix. The native terminal/workspace panels have no server-side
 * visibility/focus, so their tab clicks are ignored.
 *
 * FALLBACK: with no DockviewApi (dock shell absent / failed to init) every entry
 * point no-ops — the pre-dock rail path in panel-manager still POSTs on its own.
 * The one exception is the single dock:false capability report initDockSync
 * sends when the shell never arrives; such a client has no tile order to offer,
 * and reporting that fact is what keeps the server's freshness fields honest.
 * Polling continues at a slower cadence afterwards so that conclusion can be
 * retracted — a shell that merely booted slowly still wires and replaces the
 * record with a real dock:true report.
 */

import { getDockApi, setLayoutRestoredHook } from './dock-workspace.js';
import { setPanelFocus, reportPanelLayout } from './panel-commands.js';
// dock-iframe.js owns the placeholder lifecycle and resolves panels by these ids;
// dockPanelBesideActive() below pre-creates a placeholder the adapter then adopts
// by the same id, so both modules share the single source of truth for them.
import { PLACEHOLDER_PREFIX, PLACEHOLDER_COMPONENT } from './dock-iframe.js';

/** Depth of nested server-apply windows; >0 means "this dock change is an echo". */
let suppressDepth = 0;
/** True once the dockview listeners are attached (idempotent guard). */
let wired = false;
/** Bounded retry so late DockviewApi arrival still wires (mirrors the adapter). */
let wireAttempts = 0;

/** Attempts spent at the fast 150ms cadence (~4.5s) before this client concludes
 *  it has no dock and says so. */
const FAST_POLL_ATTEMPTS = 30;

/** Last attempt of the slow 1s cadence that follows — about a further minute of
 *  watching, so an unusually slow shell can still retract that conclusion. */
const SLOW_POLL_UNTIL_ATTEMPT = 90;

/**
 * The service panel id behind a dockview panel id, or null when the id is not a
 * service placeholder (the native terminal/workspace panels, or a nullish id).
 * @param {unknown} panelId
 * @returns {string | null}
 */
export function serviceIdOf(panelId) {
  return typeof panelId === 'string' && panelId.startsWith(PLACEHOLDER_PREFIX)
    ? panelId.slice(PLACEHOLDER_PREFIX.length)
    : null;
}

/**
 * The service panels currently occupying dock tiles, in the order an operator
 * reads them off the screen — the single source of truth for "left-to-right
 * order" behind the server-side occupancy report (`open_tiles`).
 *
 * Read from the SERIALIZED grid (`api.toJSON().grid.root`) rather than from the
 * live panel/group collections: only the serialized tree carries the spatial
 * arrangement, and its child arrays are already ordered along each branch's own
 * axis — a horizontal branch left-to-right, a vertical one top-to-bottom (the
 * same node shape dock-reconcile.js repairs). A depth-first pre-order walk of
 * those arrays therefore flattens the grid the way it reads: tiles stacked in a
 * column are reported top-to-bottom before the walk moves on to what sits
 * beside them. Pixel geometry is deliberately not reconstructed, so a genuinely
 * 2D arrangement is reported flattened, not raster-scanned.
 *
 * The native terminal/workspace tiles have no server-side panel state and are
 * dropped; `iframe:` placeholders map back to their rail ids via
 * {@link serviceIdOf}. Reads the api and mutates nothing — neither the dock nor
 * the payload it hands back.
 *
 * @param {any} api  a DockviewApi, or a nullish value in fallback mode
 * @returns {string[] | null}  service ids in reading order — `[]` when the dock
 *   holds no service tile — or null when the layout could not be read at all
 *   (no api, no dock shell, a toJSON that threw). Callers report nothing on
 *   null rather than asserting an empty workspace.
 */
export function serializeOpenTiles(api) {
  if (!api || typeof api.toJSON !== 'function') return null;
  let root;
  try {
    root = api.toJSON()?.grid?.root;
  } catch (err) {
    console.error('dock-sync: layout serialization failed', err);
    return null;
  }
  if (!root || typeof root !== 'object') return null;
  // An unrecognized ROOT leaves the whole tree unreadable, which is not the same
  // as a dock holding no service tile: collapsing it to [] would assert an empty
  // workspace the layout never described. An unrecognized INNER node is skipped
  // instead — its siblings still describe real tiles.
  if (root.type !== 'branch' && root.type !== 'leaf') return null;
  /** @type {string[]} */
  const tiles = [];
  collectTiles(root, tiles, new Set());
  return tiles;
}

/**
 * Depth-first pre-order walk collecting service ids off a serialized grid node.
 * A leaf's `views` is its tab order (one entry per tile under the one-panel-per-
 * tile invariant, several only for a layout persisted before it); a branch's
 * `data` is its children in spatial order. `seen` keeps the result a sequence of
 * distinct ids even if a malformed tree references one panel from two groups.
 * @param {any} node
 * @param {string[]} out
 * @param {Set<string>} seen
 */
function collectTiles(node, out, seen) {
  if (!node || typeof node !== 'object') return;
  if (node.type === 'leaf') {
    const views = node.data?.views;
    if (!Array.isArray(views)) return;
    for (const view of views) {
      const id = serviceIdOf(view);
      if (id && !seen.has(id)) {
        seen.add(id);
        out.push(id);
      }
    }
    return;
  }
  if (node.type === 'branch' && Array.isArray(node.data)) {
    for (const child of node.data) collectTiles(child, out, seen);
  }
}

/**
 * How long the dock must sit still before its occupancy is reported. Long
 * enough that a multi-step rebuild (mode flip, reset, fromJSON restore) settles
 * inside one window and reports once, short enough that an agent reading
 * `open_tiles` right after a human gesture sees the new arrangement.
 */
const REPORT_SETTLE_MS = 250;

/**
 * Outer bound on waiting for dock-workspace's boot-layout announcement. The
 * announcement covers a failed fetch as well as a successful one, so reaching
 * this deadline means the hook itself never ran — a dock-workspace that never
 * got to initPersistence, or a fetch that hangs without resolving or rejecting.
 * Generous, because reporting the wrong layout is worse than reporting late.
 */
const BOOT_DEADLINE_MS = 10000;

/** Pending settle timer for the occupancy report, or null when none is armed.
 *  @type {ReturnType<typeof setTimeout> | null} */
let reportHandle = null;

/** False until the boot layout is final. Layout churn before that point — the
 *  default arrangement, the adapter's first placeholder docks — describes a
 *  workspace the restore is about to replace, so it must not be reported at
 *  all. Nothing schedules a report while this is false. */
let bootReportArmed = false;

/** True while a report is in flight. At most one report is outstanding at a
 *  time, which is what makes "the ack I am waiting for" and "the last write the
 *  server applied" the same thing — see the module header. */
let reportInFlight = false;

/** The last tile list the server ACKNOWLEDGED, serialized for comparison; null
 *  before the first acknowledgement. The client half of the dedupe contract (see
 *  the module header). Set from the response rather than optimistically at POST
 *  time: a report that never landed must not be remembered as sent, or the
 *  layout it described would be deduped away and never reach the server. A
 *  deduped no-op (`updated: false`) is still an acknowledgement — the server
 *  demonstrably holds that list — so it updates the baseline too. */
let lastReported = /** @type {string | null} */ (null);

/** True once the no-dock capability report has been sent, so the give-up branch
 *  of initDockSync reports at most once however often it is re-entered. */
let capabilityReported = false;

/**
 * Arm (or re-arm) the settle window before the next occupancy report. Every
 * layout change restarts it, so a burst of changes reports once, from the
 * arrangement the dock ends up in. Silent until the boot layout is final —
 * before that there is no arrangement worth describing.
 */
function scheduleLayoutReport() {
  if (!bootReportArmed) return;
  if (reportHandle != null) clearTimeout(reportHandle);
  reportHandle = setTimeout(flushLayoutReport, REPORT_SETTLE_MS);
}

/**
 * Mark the boot layout final and report it. Called from dock-workspace's
 * layoutRestored hook — the restore itself, not a guess at how long it takes —
 * and from the boot deadline if that hook never fires. Safe to call repeatedly:
 * a later call is just another settled layout to report, which is exactly what a
 * restore that lands after the deadline is.
 */
function armBootReport() {
  bootReportArmed = true;
  scheduleLayoutReport();
}

/**
 * Serialize the CURRENT occupancy and POST it unless it matches the last report.
 * Reads the live api at flush time (not at event time), so an intermediate
 * arrangement is never what gets reported. An unreadable layout — the dock shell
 * torn down mid-window, a toJSON that threw — reports nothing and leaves the
 * dedupe baseline untouched: silence is honest, an invented empty workspace is
 * not.
 *
 * One report at a time: a flush that lands while another is in flight re-arms
 * instead of overlapping it. Two concurrent reports could be applied by the
 * server in either order and acknowledged in either order, and no client-side
 * rule can tell which write actually stuck — so the overlap is prevented rather
 * than reasoned about. The cost is bounded (a settle window of extra latency
 * while a report is outstanding); the benefit is that the baseline always names
 * a list the server demonstrably holds.
 */
function flushLayoutReport() {
  reportHandle = null;
  if (reportInFlight) {
    scheduleLayoutReport();
    return;
  }
  const tiles = serializeOpenTiles(getDockApi());
  if (!tiles) return;
  if (JSON.stringify(tiles) === lastReported) return;
  reportInFlight = true;
  void reportPanelLayout(tiles, true)
    .then((ack) => {
      if (ack) lastReported = JSON.stringify(ack.tiles ?? tiles);
    })
    // reportPanelLayout resolves null on failure rather than rejecting, so this
    // clause should be unreachable — but nothing enforces that across the module
    // boundary, and a rejection escaping it would strand reportInFlight and kill
    // reporting for the rest of the session. Settling the flag unconditionally
    // costs nothing and keeps a broken contract merely lossy: the baseline is
    // left untouched, so the next layout change re-reports.
    .catch((err) => {
      console.error('dock-sync: layout report failed', err);
    })
    .finally(() => {
      reportInFlight = false;
    });
}

/**
 * Run `fn` with the echo guard raised: any dockview focus change it causes (a
 * programmatic setActive) is recognised as a server-applied echo and not
 * POSTed back. panel-manager's activateTab wraps its focusPanel() call in this —
 * the single seam that makes server-driven and rail-driven focus applications
 * echo-safe. Re-entrant (depth-counted) and exception-safe.
 * @template T
 * @param {() => T} fn
 * @returns {T}
 */
export function withEchoSuppressed(fn) {
  suppressDepth++;
  try {
    return fn();
  } finally {
    suppressDepth--;
  }
}

/**
 * Handle a dockview active-panel change. Skips while an echo window is open
 * (a server-applied focus) and for the native terminal/workspace panels; a
 * genuine human dock-tab focus of a service panel applies locally through the
 * registered focus handler (rail accent, active-tab state — panel-manager's
 * activateTab) and POSTs setPanelFocus as a REPORT: the server mirrors the
 * active panel for the agent's gaze and broadcasts nothing for human gestures,
 * so the local apply cannot ride an SSE echo.
 */
function onActivePanelChange() {
  if (suppressDepth > 0) return;
  const api = getDockApi();
  if (!api) return;
  const id = serviceIdOf(api.activePanel?.id);
  if (!id) return;
  tileFocusHandler?.(id);
  setPanelFocus(id);
}

/**
 * Handler a human dock-tab focus is routed to, registered by panel-manager
 * (which owns the rail accent and active-tab state the focus must update).
 * Called with the focused panel's service id; must NOT POST — this module
 * owns the report.
 * @type {((serviceId: string) => void) | null}
 */
let tileFocusHandler = null;

/** @param {((serviceId: string) => void) | null} fn */
export function setTileFocusHandler(fn) {
  tileFocusHandler = fn;
}

/**
 * Handler a human tile close is routed to, registered by panel-manager (which
 * owns the local active state the close must reconcile — clear the active
 * accent). Called with the closed panel's service id; dockview's own bubble
 * handler performs the actual panel removal.
 * @type {((serviceId: string) => void) | null}
 */
let tileCloseHandler = null;

/** @param {((serviceId: string) => void) | null} fn */
export function setTileCloseHandler(fn) {
  tileCloseHandler = fn;
}

/**
 * Capture-phase click handler on the dock root. Routes a human closing a
 * service panel's dock tab to the registered LOCAL vacate handler (never a
 * server POST — see the module header). Runs in capture so the panel id is
 * resolved before dockview's own bubble handler removes the panel.
 * @param {Event} e
 */
function onDockClickCapture(e) {
  const target = e.target;
  if (!(target instanceof Element)) return;
  // The tile bar's close control (dock-tab.js); `.dv-default-tab-action` kept
  // as a fallback for any dockview-rendered default tab.
  if (!target.closest('.tile-tab-close, .dv-default-tab-action')) return;
  const tab = target.closest('.dv-tab');
  if (!tab) return;
  const id = serviceIdOf(panelIdForTab(tab));
  if (id) tileCloseHandler?.(id);
}

/**
 * Resolve the dockview panel id for a `.dv-tab` element. dock-tab.js stamps the
 * id on every tile bar it renders (`data-panel-id` on the `.tile-tab` root), so
 * the resolution is an exact read — no positional mapping. Returns null when
 * the stamp is absent (resolves to a no-op close).
 * @param {Element} tab
 * @returns {string | null}
 */
function panelIdForTab(tab) {
  const tile = tab.querySelector('.tile-tab');
  return (tile instanceof HTMLElement ? tile.dataset.panelId : null) ?? null;
}

/**
 * Dock a service panel's placeholder at an explicit position as its own tile —
 * the shared placement verb behind "open beside" (rail ⊞ / "+" add-menu) and
 * the rail drag-and-drop drop routing. Creates the placeholder by the
 * adapter's own id + component so the adapter adopts it in place (its
 * ensurePlaceholder no-ops when the id already exists), keeping this position.
 *
 * An ALREADY-DOCKED placeholder is MOVED to the position (remove + re-add —
 * safe because placeholders are empty; the overlay iframe simply re-follows
 * its new rectangle), never duplicated. Two guarded no-op cases: the target
 * group is the placeholder's own group (removing a tile's sole panel destroys
 * the group the re-add would reference), and an already-docked panel with no
 * position at all (nothing meaningful to do).
 *
 * No-op in fallback mode and in SIMPLE mode — the locked simple layout has
 * exactly one service tile, so callers there fall through to the plain show
 * path and take the tile over like a rail click. Wrapped in the echo guard so
 * neither the removal's auto-activation nor the add's active-panel change is
 * read as a human focus — the caller's show path owns the focus POST.
 * @param {string} serviceId  panel-manager rail id
 * @param {string} [title]    dock tab title (defaults to the id)
 * @param {{referenceGroup?: any, direction: string} | null} [position]
 *   referenceGroup omitted = split against the grid edge in `direction`
 */
export function dockPanelAt(serviceId, title = serviceId, position = null) {
  const api = getDockApi();
  if (!api) return;
  if (document.documentElement.getAttribute('data-ui-mode') === 'simple') return;
  const placeholderId = PLACEHOLDER_PREFIX + serviceId;
  const existing = api.getPanel(placeholderId);
  if (existing) {
    if (!position) return;
    if (existing.group && existing.group === position.referenceGroup) return;
  }
  /** @type {any} */
  const opts = { id: placeholderId, component: PLACEHOLDER_COMPONENT, title };
  if (position) opts.position = position;
  withEchoSuppressed(() => {
    try {
      if (existing) api.removePanel(existing);
      api.addPanel(opts);
    } catch (err) {
      console.error('dock-sync: dockPanelAt failed for', placeholderId, err);
    }
  });
}

/**
 * Dock a service panel's placeholder BESIDE the currently-active group as a
 * NEW tile, so a panel opened from the rail's ⊞ corner or the "+" add-menu
 * splits where the operator is working — the discoverable half of the
 * open-beside verb (rail drag-and-drop is the precise half; both land in
 * {@link dockPanelAt}, so an already-open panel is moved, not duplicated). A
 * rail click on the entry itself still REPLACES the focused tile's panel
 * instead (the adapter's ensurePlaceholder).
 * @param {string} serviceId  panel-manager rail id
 * @param {string} [title]    dock tab title (defaults to the id)
 */
export function dockPanelBesideActive(serviceId, title = serviceId) {
  const api = getDockApi();
  if (!api) return;
  const position = api.activeGroup
    ? { referenceGroup: api.activeGroup, direction: 'right' }
    : null;
  dockPanelAt(serviceId, title, position);
}

/**
 * Attach the dockview listeners once the DockviewApi exists. The api arrives
 * after panel-manager inits (dock-workspace runs later in boot), so this polls a
 * bounded number of times, mirroring the dock-iframe adapter's late-api pattern.
 * Idempotent: a no-op once wired or once the shell is confirmed absent.
 *
 * Wiring does NOT report immediately. The first report waits for dock-workspace
 * to announce the boot layout as final (setLayoutRestoredHook), because the
 * stored-layout restore happens after a network round trip: any deadline short
 * enough to be useful is one a slow fetch can lose, and losing it means
 * reporting a layout the restore is about to discard. The deadline that remains
 * is only a backstop for the hook never firing at all.
 *
 * When the retry budget runs out this client sends ONE dock:false capability
 * report — it has no tile order to offer, but the server still learns a client
 * is watching, which keeps the freshness fields honest rather than silently
 * stale. That claim stays RETRACTABLE: polling continues at a slower cadence,
 * so a dock shell that merely booted slowly still wires and overwrites the
 * record with a real dock:true report.
 */
export function initDockSync() {
  if (wired) return;
  // No document means no shell will ever arrive — the page is gone, or a test
  // environment was disposed while a retry was still queued. Drop the poll
  // instead of dereferencing a global that no longer exists.
  if (typeof document === 'undefined') return;
  const api = getDockApi();
  const root = document.getElementById('dock-root');
  if (api && root) {
    api.onDidActivePanelChange(onActivePanelChange);
    api.onDidLayoutChange(scheduleLayoutReport);
    root.addEventListener('click', onDockClickCapture, true);
    wired = true;
    setLayoutRestoredHook(armBootReport);
    setTimeout(armBootReport, BOOT_DEADLINE_MS);
    return;
  }
  const attempt = wireAttempts++;
  if (attempt > FAST_POLL_ATTEMPTS) {  // ~4.5s at 150ms — the shell is late or absent
    if (!capabilityReported) {
      capabilityReported = true;
      reportPanelLayout([], false);
    }
    // Keep watching, slowly and finitely: a shell arriving after the fast budget
    // must be able to correct the dock:false record, but a page that genuinely
    // has no dock must not poll forever.
    if (attempt < SLOW_POLL_UNTIL_ATTEMPT) setTimeout(initDockSync, 1000);
    return;
  }
  setTimeout(initDockSync, 150);
}
