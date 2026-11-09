// @ts-check
/* OSPREY Web Terminal — Dock State Sync
 *
 * The bridge between the dockview workspace and the server-owned panel state.
 * The server is authoritative for the rail: every panel show/hide lives on the
 * server and is broadcast over the panel_visibility / panel_register SSE echo,
 * so an agent MCP call and a human rail gesture are indistinguishable downstream
 * (see panel-commands.js / panel-manager.js). This module carries the REVERSE
 * half — turning a human's dockview gestures back into the same server POSTs —
 * without letting the server's own echo bounce back out again.
 *
 * TWO DIRECTIONS
 * --------------
 *  - Server → dock: panel-sse's frame dispatcher already drives the dock (an
 *    agent's `panel_op` through the op runner and the layout holder, the
 *    visibility fallback → hidePanel, register → panel-lifecycle's addPanel,
 *    all via the dock-iframe adapter). This module does NOT duplicate that; it
 *    only guarantees those applied changes never POST back.
 *  - Dock → server: a human focusing a tile POSTs setPanelFocus — the only
 *    new POST source the docked model introduces. A human CLOSING a tile is
 *    deliberately NOT a server change: tile occupancy is per-client layout
 *    state under the launcher-rail model (rail membership is the shared state,
 *    and closing a tile does not remove the panel from the rail), so the close
 *    is routed to a locally-registered vacate handler instead.
 *
 * …plus one channel that is neither: the LAYOUT REPORT below tells the server
 * what is on screen without commanding anything.
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
 * THE LAYOUT REPORT (deliberately NOT echo-guarded)
 * -------------------------------------------------
 * Rail membership says which panels an operator can reach; it does not say what
 * is actually on screen. So this module also REPORTS the dock's layout (POST
 * /api/panel-layout): the layout holder's tree — tiles, the panel each shows
 * and the terminal — plus the panels living in pop-out windows and
 * this page's `client_id`, letting the agent read the real workspace instead
 * of inferring it. It is read from the holder (dock-apply.js), never serialized
 * from dockview's toJSON: the holder is the dock's single source of truth, and
 * the one producer of what the agent reads. A report is not a command: the
 * server stores it last-writer-wins, records this page as the current dock
 * client, mints a `rev` and broadcasts nothing. The ack's `rev` goes back to
 * the holder (`ack(rev)`: kept as `ackedRev`, `dirty` cleared) — but only while
 * the holder still holds what was acked; a layout that moved on while the
 * report was in flight stays dirty and is reported next.
 *
 * Unlike the focus POST, the reporter is NOT gated on the echo guard — it must
 * not be. dockview emits onDidLayoutChange on a microtask, so the event lands
 * AFTER the synchronous withEchoSuppressed window has already closed; a guard
 * check there would read 0, suppress nothing, and merely look correct.
 * Convergence comes from CONTENT DEDUPE on both ends instead: this module skips
 * a POST whose layout and pop-outs match the last ones the server ACKNOWLEDGED
 * (the `rev` an ack carries is not part of the comparison), and the route
 * treats a report equal to stored state as a no-op. A server-applied
 * arrangement therefore settles after at most one no-op round-trip rather than
 * looping. The baseline tracks acknowledgements, not attempts, so a report that
 * never landed is never deduped away — and only ONE report is ever in flight,
 * so the ack being waited on always names the last write the server applied.
 * Overlapping reports could be applied in one order and acknowledged in the
 * other, leaving no client-side rule able to tell which one stuck; the overlap
 * is prevented rather than arbitrated.
 *
 * A REPORT THAT DOES NOT LAND is sent again, whether or not the layout changes:
 * one that got no answer in time, a network error, a 5xx, 408 or 429 may pass,
 * so it goes out again after a backoff that doubles up to a ceiling. A 4xx is
 * a refusal the server would repeat (the layout breaks a dock rule, names an
 * unknown panel): it is logged with the server's detail, and that layout is
 * not sent again until it changes. Either way the holder stays dirty, which is
 * the truth: the server does not hold what is on screen.
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
 * Each report is read at FLUSH time from the holder, once a settle debounce has
 * expired — so a rebuild that removes and re-adds panels one at a time (reset,
 * fromJSON restore) reports only the arrangement it settles on, never an
 * intermediate one. Note the debounce RESTARTS on every
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
 * every layout apply (the holder rebuilds the dock with one fromJSON on every
 * change, dock-apply.js). onDidRemovePanel therefore cannot tell a human close
 * from a rebuild, and treating a rebuild's removals as closes would silently
 * hide every panel on the server. A capture-phase click on the tile bar's
 * close control (`.tile-tab-close`, dock-tab.js — `.dv-default-tab-action`
 * kept as a fallback for any dockview-rendered default tab) fires ONLY on a
 * genuine human gesture, never during a rebuild, so it is the unambiguous close
 * signal. Capture phase resolves the panel id before the bar's own handler
 * takes the panel out through the holder.
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
import { setPanelFocus } from './panel-commands.js';
import { withPrefix, getPageClientId } from './api.js';
// The layout holder keeps the dock's layout and resolves panels by these ids;
// dockPanelAt() below places a placeholder the adapter then adopts by the same
// id, so both modules share the single source of truth for them.
import { PLACEHOLDER_PREFIX, getLayoutHolder, sameShape, sameSizes } from './dock-apply.js';
import { openAt } from './dock-open.js';

/** dockview's names for the vertical sides, as its drop events and the rail's
 *  drop routing spell them, to the engine's. */
const SIDE_OF_DIRECTION = /** @type {Record<string, string>} */ ({ above: 'top', below: 'bottom' });

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
function serviceIdOf(panelId) {
  return typeof panelId === 'string' && panelId.startsWith(PLACEHOLDER_PREFIX)
    ? panelId.slice(PLACEHOLDER_PREFIX.length)
    : null;
}

/**
 * How long the dock must sit still before its layout is reported. Long
 * enough that a multi-step rebuild (reset, fromJSON restore) settles
 * inside one window and reports once, short enough that an agent reading
 * the layout right after a human gesture sees the new arrangement.
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

/** How long a report may wait for its answer before it counts as lost and is
 *  sent again. A report the server answers takes milliseconds; one with no
 *  answer at all (a dropped connection) would otherwise stay in flight, and
 *  hold back every report after it, for good. */
const REPORT_TIMEOUT_MS = 30000;

/** The wait before a lost report is sent again; it doubles with each loss in
 *  a row, up to {@link RESEND_MAX_MS}. */
const RESEND_BASE_MS = 500;

/** The longest wait between two sends of a lost report: a server that is down
 *  is asked again at least this often, and never more often than the backoff. */
const RESEND_MAX_MS = 30000;

/** Pending timer for the next flush (the settle window, or the backoff before
 *  a lost report is sent again), or null when none is armed.
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

/**
 * One report's content: the layout and the pop-outs beside it.
 * @typedef {{layout: unknown, popouts: string[]}} Report
 */

/** The last report the server ACKNOWLEDGED; null before the first
 *  acknowledgement. The client half of the dedupe contract (see the module
 *  header), compared with `sameReport`. Set from the response rather than
 *  optimistically at POST time: a report that never landed must not be
 *  remembered as sent, or the layout it described would be deduped away and
 *  never reach the server. A deduped no-op (`updated: false`) is still an
 *  acknowledgement — the server demonstrably holds that layout — so it updates
 *  the baseline too. */
let lastReported = /** @type {Report | null} */ (null);

/** Reports lost in a row since the server last answered one: what the backoff
 *  before the next send is computed from. */
let lostInARow = 0;

/** The last report the server refused with a 4xx it would repeat; that report
 *  is not sent again until the layout changes. Null once a report lands. */
let refusedReport = /** @type {Report | null} */ (null);

/** True while a `panel_report_request` waits for its report: the next flush
 *  sends even a layout the server already acknowledged. Cleared when a report
 *  is sent, set again if that report is lost — so the request is answered by a
 *  report sent after it, never by one already in flight when it came. */
let reportForced = false;

/** True once the no-dock capability report has been sent, so the give-up branch
 *  of initDockSync reports at most once however often it is re-entered. */
let capabilityReported = false;

/**
 * Arm (or re-arm) the settle window before the next layout report. Every
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
 * Whether two reports describe the same workspace: the same pop-outs, and the
 * same layout tree (a tile's one panel included) and sizes up to dockview's
 * size rounding (the holder's own notion of "not a change", `resized`). The
 * read-back of a tree the holder just applied differs from what it applied by
 * a pixel's worth of fraction, so an exact comparison would report — and
 * revise — a workspace nobody changed.
 * @param {Report | null} a
 * @param {Report | null} b
 * @returns {boolean}
 */
export function sameReport(a, b) {
  if (!a || !b) return false;
  if (a.popouts.length !== b.popouts.length) return false;
  if (a.popouts.some((key, i) => key !== b.popouts[i])) return false;
  if (a.layout === null || b.layout === null) return a.layout === b.layout;
  const x = /** @type {any} */ (a.layout);
  const y = /** @type {any} */ (b.layout);
  return sameShape(x, y) && sameSizes(x, y);
}

/**
 * Record a layout the server acknowledged through another path than this
 * module's own report — an agent op's answer (/api/panel-op-result) — as the
 * dedupe baseline. The server holds that layout now, so it is what a later
 * change is compared with: without this, a person undoing an agent's open
 * would land on the stale baseline, be deduped away as already reported, and
 * leave the server holding the panel open.
 *
 * A holder that no longer holds `sent` reports what it holds. An answer can
 * land after the report of a newer layout (an op the dock answered late, a
 * person's change meanwhile); the server then holds a layout that is no longer
 * on screen, and no further change may come to correct it.
 * @param {Report} sent  the acknowledged report
 */
export function noteReported(sent) {
  lastReported = sent;
  const holder = getLayoutHolder();
  if (holder && !sameReport({ layout: holder.layout, popouts: holder.popouts }, sent)) {
    scheduleLayoutReport();
  }
}

/**
 * What one report came to: the server's acknowledgement (`{status, layout,
 * popouts, dock, rev, updated}` — `updated: false` marks a deduped no-op, still
 * an acknowledgement); a refusal the server would repeat (a 4xx other than 408
 * and 429), with its status and the server's `detail`; or a loss that may pass
 * (no answer within {@link REPORT_TIMEOUT_MS}, a network error, a 5xx, 408 or
 * 429), with what was seen.
 * @typedef {{ack: {rev?: number} | null}
 *   | {refused: {status: number, detail: unknown}}
 *   | {lost: string}} ReportOutcome
 */

/**
 * POST one layout report with this page's client id — the one poster for
 * /api/panel-layout. It sends what it is given: dedupe, resends and the
 * refusal memory belong to the reporter (flushLayoutReport). Never rejects.
 * @param {unknown} layout  the holder's dock layout, or null for a dock-less report
 * @param {string[]} [popouts]  the panels in windows of their own
 * @param {boolean} [dock]  whether this client has a dock shell
 * @returns {Promise<ReportOutcome>}
 */
async function postLayoutReport(layout, popouts = [], dock = true) {
  const abandon = new AbortController();
  const timer = setTimeout(() => abandon.abort(), REPORT_TIMEOUT_MS);
  try {
    const resp = await fetch(withPrefix('/api/panel-layout'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ layout, popouts, dock, client_id: getPageClientId() }),
      signal: abandon.signal,
    });
    if (resp.ok) return { ack: await resp.json() };
    const { status } = resp;
    if (status >= 400 && status < 500 && status !== 408 && status !== 429) {
      return { refused: { status, detail: await detailOf(resp) } };
    }
    return { lost: `HTTP ${status}` };
  } catch (err) {
    return { lost: String(err) };
  } finally {
    clearTimeout(timer);
  }
}

/**
 * The `detail` of a refusal's body (the route's error shape), the body itself
 * when it has none, or null when it is not JSON.
 * @param {Response} resp
 * @returns {Promise<unknown>}
 */
async function detailOf(resp) {
  try {
    const body = await resp.json();
    return body?.detail ?? body;
  } catch {
    return null;
  }
}

/**
 * Read the holder's CURRENT layout and pop-outs and POST them unless they match
 * the last acknowledged report. Read at flush time (not at event time), so an
 * intermediate arrangement is never what gets reported. No holder, or a holder
 * with no layout yet, reports nothing and leaves the dedupe baseline untouched:
 * silence is honest, an invented empty workspace is not. An empty dock whose
 * panes are all detached is a real workspace, though — the Agent pane leaving
 * the default layout for its window — and is reported as a null layout with
 * the pop-outs beside it. So is the empty dock that last pane's window leaves
 * behind when it closes: the server still names the pop-out until it is told. A report the server
 * refused with a 4xx is not sent again until the layout changes; a forced
 * report (`reportLayoutNow`) skips only the dedupe.
 *
 * One report at a time: a flush that lands while another is in flight re-arms
 * instead of overlapping it. Two concurrent reports could be applied by the
 * server in either order and acknowledged in either order, and no client-side
 * rule can tell which write actually stuck — so the overlap is prevented rather
 * than reasoned about. The cost is bounded (a settle window of extra latency
 * while a report is outstanding); the benefit is that the baseline always names
 * a layout the server demonstrably holds.
 */
function flushLayoutReport() {
  reportHandle = null;
  if (reportInFlight) {
    scheduleLayoutReport();
    return;
  }
  const holder = getLayoutHolder();
  const layout = holder?.layout ?? null;
  if (!holder) return;
  if (!layout && holder.detachedKeys().length === 0 && !lastReported?.popouts.length) return;
  const popouts = holder.popouts;
  const sent = { layout, popouts };
  if (!reportForced && sameReport(sent, lastReported)) {
    // Already acknowledged; a dirty flag left by a no-op write settles here.
    if (holder.dirty && holder.ackedRev !== null) holder.ack(holder.ackedRev);
    return;
  }
  if (sameReport(sent, refusedReport)) return;
  const forced = reportForced;
  reportForced = false;
  reportInFlight = true;
  void postLayoutReport(layout, popouts)
    .then((outcome) => settleReport(holder, sent, forced, outcome))
    // settleReport only records and re-arms, so this clause should be
    // unreachable — but a rejection escaping it would strand reportInFlight
    // and kill reporting for the rest of the session. Settling the flag
    // unconditionally keeps a broken contract merely lossy: the baseline is
    // left untouched, so the next layout change re-reports.
    .catch((err) => {
      console.error('dock-sync: layout report failed', err);
    })
    .finally(() => {
      reportInFlight = false;
    });
}

/**
 * Act on what one report came to. An acknowledged report becomes the dedupe
 * baseline, and its `rev` acks the holder while the holder still holds what was
 * sent; a layout that moved on meanwhile goes out next. A lost report is sent
 * again after the backoff, with one warning for each run of losses. A refused
 * one is logged with the server's detail and kept, so that the same layout is
 * not sent again until it changes.
 * @param {import('./dock-apply.js').LayoutHolder} holder  the holder it was read from
 * @param {Report} sent  what was sent
 * @param {boolean} forced  whether it answered a `panel_report_request`
 * @param {ReportOutcome} outcome
 */
function settleReport(holder, sent, forced, outcome) {
  if ('lost' in outcome) {
    lostInARow += 1;
    if (lostInARow === 1) {
      console.warn('dock-sync: a layout report did not land; sending it again', outcome.lost);
    }
    if (forced) reportForced = true;
    resendLayoutReport();
    return;
  }
  lostInARow = 0;
  if ('refused' in outcome) {
    refusedReport = sent;
    console.error(
      `dock-sync: the server refused the layout report (HTTP ${outcome.refused.status}); ` +
        'it is not sent again until the layout changes:',
      outcome.refused.detail
    );
    return;
  }
  refusedReport = null;
  lastReported = sent;
  const rev = outcome.ack?.rev;
  if (typeof rev !== 'number') return;
  // Only the layout that was sent is acked; one that moved on meanwhile stays
  // dirty and goes out on the next flush.
  if (sameReport({ layout: holder.layout, popouts: holder.popouts }, sent)) {
    holder.ack(rev);
  } else {
    scheduleLayoutReport();
  }
}

/** Flush again once the backoff the losses in a row have earned has passed. */
function resendLayoutReport() {
  clearTimeout(reportHandle ?? undefined);
  const wait = Math.min(RESEND_BASE_MS * 2 ** (lostInARow - 1), RESEND_MAX_MS);
  reportHandle = setTimeout(flushLayoutReport, wait);
}

/**
 * Report the holder's layout now, even when the server has acknowledged it
 * already — the answer to a `panel_report_request` (panel-sse.js): the current
 * dock client went away, and a report makes this dock current. It is the
 * reporter's own flush, so it follows a report in flight rather than
 * overlapping it, and a lost one is sent again. No holder, or no layout yet,
 * reports nothing: a page without a dock must not become the dock client.
 */
export function reportLayoutNow() {
  reportForced = true;
  clearTimeout(reportHandle ?? undefined);
  flushLayoutReport();
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
 * the placement verb behind "Open in a new tile" and the arrange rebuild. One
 * operation on the layout holder (dock-open.js's openAt), placing the
 * placeholder by the adapter's own id so the adapter adopts it in place.
 *
 * An ALREADY-DOCKED placeholder is MOVED to the position, never duplicated.
 * Two guarded no-op cases: the target is the placeholder's own tile, and an
 * already-docked panel with no position at all (nothing meaningful to do).
 *
 * No-op in fallback mode (no dock to place into).
 * No-op too for a panel living in its own pop-out window — a placeholder for
 * it would be an empty tile: it comes back by a reattach (pane-window.js).
 * The holder applies the layout inside the echo guard, so the rebuild's
 * active-panel change is never read as a human focus — the caller's show path
 * owns the focus POST.
 * @param {string} serviceId  panel-manager rail id
 * @param {string} [title]    dock tab title (defaults to the id)
 * @param {{referenceGroup?: any, referenceTile?: string, direction: string} | null} [position]
 *   the tile to split — by id, or by its dockview group — on `direction`'s
 *   side, in the engine's words (`top`, `bottom`) or dockview's (`above`,
 *   `below`); neither tile = split against the grid edge in `direction`
 */
export function dockPanelAt(serviceId, title = serviceId, position = null) {
  const holder = getLayoutHolder();
  if (!getDockApi() || !holder || holder.isDetached(serviceId)) return;
  holder.setPanelRecord(serviceId, { title });
  /** @type {import('./dock-open.js').Position | null} */
  let at = null;
  if (position) {
    const referenceTile = position.referenceTile ?? position.referenceGroup?.id;
    at = {
      ...(typeof referenceTile === 'string' ? { referenceTile } : {}),
      direction: /** @type {any} */ (SIDE_OF_DIRECTION[position.direction] ?? position.direction),
    };
  }
  withEchoSuppressed(() => {
    try {
      holder.run((layout, ctx) => openAt(layout, serviceId, at, ctx));
    } catch (err) {
      console.error('dock-sync: dockPanelAt failed for', PLACEHOLDER_PREFIX + serviceId, err);
    }
  });
}

/**
 * Dock a service panel's placeholder BESIDE the focused tile as a NEW tile —
 * the row menu's "Open in a new tile", the one open that splits where the
 * operator is working instead of replacing (it lands in
 * {@link dockPanelAt}, so an already-open panel is moved, not duplicated).
 * Every other open — a row click, the agent, a link — takes the service tile
 * used last, replacing its panel (dock-open.js's openPanel, through the
 * activation).
 * @param {string} serviceId  panel-manager rail id
 * @param {string} [title]    dock tile title (defaults to the id)
 */
export function dockPanelBesideActive(serviceId, title = serviceId) {
  const holder = getLayoutHolder();
  if (!getDockApi() || !holder) return;
  const focused = holder.focusedTile;
  dockPanelAt(serviceId, title, focused ? { referenceTile: focused, direction: 'right' } : null);
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
    // A pane detaching, reattaching or closing with its window is a change the
    // agent reads (the report's `popouts`), whether or not dockview's tree moved.
    getLayoutHolder()?.onDetachedChange(scheduleLayoutReport);
    root.addEventListener('click', onDockClickCapture, true);
    wired = true;
    setLayoutRestoredHook(armBootReport);
    setTimeout(armBootReport, BOOT_DEADLINE_MS);
    return;
  }
  const attempt = wireAttempts++;
  if (attempt > FAST_POLL_ATTEMPTS) {
    // ~4.5s at 150ms — the shell is late or absent
    if (!capabilityReported) {
      capabilityReported = true;
      void postLayoutReport(null, [], false);
    }
    // Keep watching, slowly and finitely: a shell arriving after the fast budget
    // must be able to correct the dock:false record, but a page that genuinely
    // has no dock must not poll forever.
    if (attempt < SLOW_POLL_UNTIL_ATTEMPT) setTimeout(initDockSync, 1000);
    return;
  }
  setTimeout(initDockSync, 150);
}
