// @ts-check
/**
 * draft-client — the plan panel's live view of the server-held shared plan
 * draft (the plan-panel integration and the launch revision gate).
 *
 * Split out of panel.js to keep panel.js focused on plan browsing/launch
 * chrome, and — more importantly — so the revision/binding/pending-key rules
 * are a pure, DOM-light state machine that vitest can drive with plain frame
 * objects, without a real `EventSource` (happy-dom does not implement one;
 * see `createSSEConnection`, the one function here that touches it, kept
 * deliberately thin and injectable).
 *
 * Three layers, outside in:
 *
 * 1. Pure reducers (`createInitialState`, `reduceFrame`, `reduceReset`,
 *    `computeDelta`, `shouldShowAffordance`, `shouldAutoSwitch`,
 *    `shouldShowAgentDraftBanner`, `resolvePinnedRevision`) — no
 *    DOM, no network, no timers. These encode every rule from the proposal:
 *    revision drop/apply/resync, hello/resync baseline reset (including
 *    backward), own-origin echo suppression, and the minimal-delta
 *    PATCH-back shape (blank -> `remove[]`).
 * 2. `createDraftClient(deps)` — the orchestrator. Wires the reducers to a
 *    form (pending-key tracking, debounced PATCH-back), a set of DOM
 *    callbacks (bound indicator, affordance, agent-draft banner,
 *    unknown-plan banner, flash, agent-edited note) and a set of injectable
 *    HTTP functions
 *    (`getDraft`/`patchDraft`/`deleteDraft`/`fetchPlans`). Every side effect
 *    is a `deps` function, so tests can substitute canned promises and a real
 *    (happy-dom) form without a real bridge.
 * 3. `createSSEConnection` — the one real `EventSource` touchpoint, wrapped
 *    behind an `onFrame` callback and manual backoff reconnect (bypassing the
 *    browser's own retry so a dead sidecar isn't hammered at a fixed
 *    interval forever). `createDraftClient` accepts an `sseFactory` override
 *    so tests never construct a real one.
 *
 * @module draft-client
 */

import { flashElement } from '/design-system/js/highlight.js';

/**
 * @typedef {object} DraftSnapshot
 * @property {string} plan_name
 * @property {Record<string, unknown>} plan_args
 * @property {string|null} [updated_by]
 * @property {string} [updated_at]
 */

/**
 * @typedef {object} DraftFrame
 * @property {'hello'|'change'|'clear'|'plan-change'|'launched'} type
 * @property {DraftSnapshot|null} draft
 * @property {string[]} [changed]
 * @property {number} revision
 * @property {string|null} [origin]
 * @property {string} [run_id]  Present only on a `launched` frame: the id of
 *   the run the (now-launched) revision started.
 */

/**
 * @typedef {object} DraftGetResponse
 * @property {DraftSnapshot|null} draft
 * @property {number} revision
 */

/**
 * @typedef {object} RegisteredFieldLike
 * @property {HTMLElement} el
 * @property {(value: unknown) => void} setValue
 */

/**
 * Shape of schema-form.js's `renderSchemaForm(...)` return value
 * (`PlanArgsCollector`): a callable `() => Record<string, unknown>` (the
 * unchanged `collectPlanArgs()` contract) plus `applyValues`/`fields` layered
 * on top.
 *
 * @typedef {(() => Record<string, unknown>) & {
 *   applyValues: (values: Record<string, unknown>) => void,
 *   fields: Record<string, RegisteredFieldLike>
 * }} CollectorLike
 */

/**
 * @typedef {object} DraftState
 * @property {string} clientId
 * @property {number|null} lastAppliedRevision
 * @property {string|null} draftPlanName
 * @property {Record<string, unknown>|null} draftArgs
 * @property {boolean} bound
 * @property {string|null} selectedName
 * @property {boolean} formDirty  Whether the operator has unsaved manual
 *   edits in the (unbound) form. Carried in state so the pure auto-switch /
 *   banner predicates can honor it; the reducers themselves never set it —
 *   the orchestrator owns flipping it from real form events.
 * @property {string|null} lastUpdatedBy  `updated_by` of the last draft seen
 *   at EITHER ingestion site (`reduceReset` or `reduceFrame`'s apply/echo
 *   path); `null` while the draft is null. Lets a reconnect hello repeating
 *   an already-seen `updated_at` be told apart from genuinely new agent work.
 * @property {string|null} lastUpdatedAt  `updated_at` companion to
 *   `lastUpdatedBy` (ISO-8601 string as the bridge minted it); `null` while
 *   the draft is null — and a null last-seen value compares as
 *   always-in-the-past (see `shouldAutoSwitch`).
 * @property {{runId: string, revision: number}|null} launchBanner  The last
 *   observed `launched` frame's run — the fact behind the "revision N
 *   launched -> run <id>" banner (FR8: the launch moment is visible whether
 *   the panel or the agent triggered it). `null` until the first launch.
 */

/**
 * @param {string} clientId
 * @returns {DraftState}
 */
export function createInitialState(clientId) {
  return {
    clientId,
    lastAppliedRevision: null,
    draftPlanName: null,
    draftArgs: null,
    bound: false,
    selectedName: null,
    formDirty: false,
    lastUpdatedBy: null,
    lastUpdatedAt: null,
    launchBanner: null,
  };
}

/**
 * Apply a hello frame or a post-resync `GET /draft` snapshot: unconditionally
 * resets the last-applied baseline, including backward (PROPOSAL.md: the
 * revision counter is per-process, so a bridge restart must not leave every
 * subsequent frame looking "stale").
 *
 * @param {DraftState} state
 * @param {DraftGetResponse} snapshot
 * @returns {DraftState}
 */
export function reduceReset(state, snapshot) {
  return {
    ...state,
    lastAppliedRevision: snapshot.revision,
    draftPlanName: snapshot.draft ? snapshot.draft.plan_name : null,
    draftArgs: snapshot.draft ? snapshot.draft.plan_args : null,
    lastUpdatedBy: snapshot.draft ? (snapshot.draft.updated_by ?? null) : null,
    lastUpdatedAt: snapshot.draft ? (snapshot.draft.updated_at ?? null) : null,
  };
}

/**
 * @typedef {{type: 'drop'}
 *   | {type: 'resync'}
 *   | {type: 'reset'}
 *   | {type: 'apply', frame: DraftFrame}
 *   | {type: 'echo', frame: DraftFrame}
 *   | {type: 'launched', frame: DraftFrame}} FrameAction
 */

/**
 * The revision state machine (PROPOSAL.md "revision handling"): a `hello`
 * frame always resets the baseline unconditionally; otherwise `revision <=
 * last applied` is dropped, `== last + 1` is applied, and `> last + 1`
 * triggers a resync (the caller must `GET /draft` and feed the result to
 * `reduceReset`). An own-origin frame (matching `clientId`) still advances
 * the baseline — it is reported as `'echo'` so the caller skips value
 * re-application and flash without leaving a phantom revision gap.
 *
 * A `null` baseline (no hello/reset observed yet) is treated as a gap —
 * this should not happen in practice (the bridge sends `hello` immediately
 * on connect) but keeps the reducer total.
 *
 * @param {DraftState} state
 * @param {DraftFrame} frame
 * @returns {{state: DraftState, action: FrameAction}}
 */
export function reduceFrame(state, frame) {
  if (frame.type === 'hello') {
    return { state: reduceReset(state, frame), action: { type: 'reset' } };
  }

  if (frame.type === 'launched') {
    // A launch is an EVENT on the shared draft stream, not a draft mutation:
    // it records the banner fact only and never advances the revision
    // baseline or touches draftPlanName/draftArgs (the launched revision is
    // whatever this client already holds — the launch didn't change it). A
    // frame with no usable run_id is dropped: a banner that can't link to a
    // run is worse than no banner (keeps the reducer total against a
    // malformed frame).
    const runId = frame.run_id;
    if (typeof runId !== 'string' || runId === '') {
      return { state, action: { type: 'drop' } };
    }
    const nextState = { ...state, launchBanner: { runId, revision: frame.revision } };
    return { state: nextState, action: { type: 'launched', frame } };
  }

  if (state.lastAppliedRevision === null || frame.revision > state.lastAppliedRevision + 1) {
    return { state, action: { type: 'resync' } };
  }

  if (frame.revision <= state.lastAppliedRevision) {
    return { state, action: { type: 'drop' } };
  }

  const isOwnOrigin = frame.origin != null && frame.origin === state.clientId;
  const nextState = {
    ...state,
    lastAppliedRevision: frame.revision,
    draftPlanName: frame.draft ? frame.draft.plan_name : null,
    draftArgs: frame.draft ? frame.draft.plan_args : null,
    lastUpdatedBy: frame.draft ? (frame.draft.updated_by ?? null) : null,
    lastUpdatedAt: frame.draft ? (frame.draft.updated_at ?? null) : null,
  };
  return { state: nextState, action: { type: isOwnOrigin ? 'echo' : 'apply', frame } };
}

/**
 * Whether the draft-is-on-plan-X affordance should show: unbound, a draft
 * exists, and its plan_name matches the plan currently being viewed. Binding
 * itself only ever happens via explicit selection/affordance-click — this is
 * purely the "should we surface the hint" predicate.
 *
 * @param {DraftState} state
 * @returns {boolean}
 */
export function shouldShowAffordance(state) {
  return !state.bound && state.draftPlanName !== null && state.draftPlanName === state.selectedName;
}

/**
 * The literal `origin`/`client_id`/`updated_by` value agent writes carry
 * (bridge contract). Browser tabs mint UUID client ids, so this never
 * collides with a tab's own id.
 */
export const AGENT_ORIGIN = 'mcp-agent';

/**
 * Whether `next` is strictly newer than the last-seen `lastSeen` timestamp.
 * Both are the bridge's own ISO-8601 UTC strings, so lexicographic `>` is
 * chronological order. A null last-seen value compares as always-in-the-past
 * (an agent draft arriving after a no-draft baseline counts as an advance); a
 * null `next` (no draft) never advances.
 *
 * @param {string|null} lastSeen
 * @param {string|null} next
 * @returns {boolean}
 */
function updatedAtAdvanced(lastSeen, next) {
  if (next === null) return false;
  return lastSeen === null || next > lastSeen;
}

/**
 * Whether the "the agent is working on plan X" banner should show: a draft
 * exists, its last writer was the agent, this tab is unbound, the draft names
 * a different plan than the one being viewed, and the operator has unsaved
 * manual edits (`formDirty`) — i.e. exactly the case where auto-switching
 * would clobber operator work, so a banner is surfaced instead.
 *
 * @param {DraftState} state
 * @returns {boolean}
 */
export function shouldShowAgentDraftBanner(state) {
  return (
    state.draftPlanName !== null &&
    state.lastUpdatedBy === AGENT_ORIGIN &&
    !state.bound &&
    state.draftPlanName !== state.selectedName &&
    state.formDirty
  );
}

/**
 * Decide whether an ingested frame/snapshot should auto-switch the panel onto
 * the agent's draft plan, and whether the agent-draft banner should show
 * afterward. Pure — `prevState` is the state BEFORE the ingestion (it carries
 * the last-seen `lastUpdatedAt` pair and the pre-reset `lastAppliedRevision`),
 * `reduced` is the reducer's result AFTER it (a `reduceFrame` return, or a
 * bare `reduceReset` result wrapped by the caller as
 * `{state, action: {type: 'reset'}}`).
 *
 * Switch rules:
 *  - Never while bound, while `formDirty`, or when the ingested draft is null
 *    (a clear/no-draft snapshot has nothing to switch to).
 *  - Live path (an `apply` action): switch when the frame's origin is the
 *    literal agent origin. An `echo` (own-origin) frame never switches.
 *  - Reset path (a `reset` action — hello or `GET /draft` resync): switch on
 *    the first-ever reset (pre-reset `lastAppliedRevision === null`), or when
 *    the reset draft's `updated_at` advanced past the last-seen value AND its
 *    `updated_by` is the agent. A routine reconnect hello / internal resync
 *    (revision-gap or PATCH-failure recovery) repeats an already-seen
 *    `updated_at`, so it never switches.
 *
 * @param {DraftState} prevState
 * @param {{state: DraftState, action: FrameAction}} reduced
 * @returns {{switch: boolean, banner: boolean}}
 */
export function shouldAutoSwitch(prevState, reduced) {
  const next = reduced.state;
  const banner = shouldShowAgentDraftBanner(next);
  if (next.bound || next.formDirty || next.draftPlanName === null) {
    return { switch: false, banner };
  }
  const action = reduced.action;
  if (action.type === 'apply') {
    return { switch: action.frame.origin === AGENT_ORIGIN, banner };
  }
  if (action.type === 'reset') {
    const firstEver = prevState.lastAppliedRevision === null;
    const agentAdvanced =
      next.lastUpdatedBy === AGENT_ORIGIN && updatedAtAdvanced(prevState.lastUpdatedAt, next.lastUpdatedAt);
    return { switch: firstEver || agentAdvanced, banner };
  }
  return { switch: false, banner };
}

/**
 * Minimal-delta PATCH body pieces for a set of pending (user-touched) keys:
 * a key present (non-OMIT) in `fullArgs` goes into `plan_args_patch`; a key
 * absent (blanked by the user) goes into `remove[]` (PROPOSAL.md: "blank
 * transitions -> remove[]"; removal is never expressed as a null value).
 *
 * @param {Record<string, unknown>} fullArgs  The collector's current full read.
 * @param {Iterable<string>} pendingKeys
 * @returns {{plan_args_patch: Record<string, unknown>, remove: string[]}}
 */
export function computeDelta(fullArgs, pendingKeys) {
  /** @type {Record<string, unknown>} */
  const plan_args_patch = {};
  /** @type {string[]} */
  const remove = [];
  for (const key of pendingKeys) {
    if (Object.prototype.hasOwnProperty.call(fullArgs, key)) {
      plan_args_patch[key] = fullArgs[key];
    } else {
      remove.push(key);
    }
  }
  return { plan_args_patch, remove };
}

/**
 * Resolve the `draft_revision` to pin for Launch: the just-flushed PATCH's
 * own response revision when a flush actually happened, else the last
 * applied frame/hello baseline (the launch revision gate).
 *
 * @param {{patched: boolean, revision: number|null}} flushResult
 * @param {number|null} lastAppliedRevision
 * @returns {number|null}
 */
export function resolvePinnedRevision(flushResult, lastAppliedRevision) {
  return flushResult.patched ? flushResult.revision : lastAppliedRevision;
}

/**
 * A per-tab id for frame `origin`/PATCH `client_id` echo suppression. Falls
 * back to a `Math.random()`-based id when `crypto.randomUUID` is unavailable
 * (a non-secure-context deployment) so the whole panel — including the
 * always-available manual flow — degrades gracefully instead of dying on a
 * `TypeError` just because the live-draft feature can't mint a "real" UUID.
 *
 * @returns {string}
 */
export function generateClientId() {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  return `tab-${Math.random().toString(36).slice(2)}${Date.now()}`;
}

/**
 * The panel's own Launch request body for the two mutually-exclusive launch
 * modes (the launch revision gate): bound mode sends only
 * `{draft_revision}` — never `plan_name`/`plan_args` alongside, so the
 * launched args always come from the bridge's pinned-revision snapshot, not
 * this body; unbound (manual) mode sends the collected form args exactly as
 * it did before draft mode existed.
 *
 * @param {{bound: boolean, pinnedRevision: number|null, planName: string, planArgs: Record<string, unknown>}} input
 * @returns {Record<string, unknown>}
 */
export function buildLaunchRequestBody({ bound, pinnedRevision, planName, planArgs }) {
  return bound ? { draft_revision: pinnedRevision } : { plan_name: planName, plan_args: planArgs };
}

/**
 * @typedef {{type: 'writes_not_armed'}
 *   | {type: 'run_started', runId: string}
 *   | {type: 'stale_draft_revision'}
 *   | {type: 'draft_revision_already_launched'}
 *   | {type: 'conflict', detail: string}
 *   | {type: 'bridge_unreachable'}
 *   | {type: 'error', detail: string}} LaunchOutcome
 */

/**
 * Classify `POST /runs/launch`'s response into a display-ready outcome —
 * pure and DOM/fetch-free, so the status/`code` branching (in particular
 * distinguishing the machine-readable `stale_draft_revision` and
 * `draft_revision_already_launched` discriminators from a bare bridge-relayed
 * 409) is unit-testable without a real panel.
 *
 * @param {number} status
 * @param {any} body
 * @returns {LaunchOutcome}
 */
export function classifyLaunchResponse(status, body) {
  if (status === 200 && body && body.status === 'writes_not_armed') return { type: 'writes_not_armed' };
  if (status === 200 && body && body.run_id) return { type: 'run_started', runId: String(body.run_id) };
  if (status === 409 && body && body.code === 'stale_draft_revision') return { type: 'stale_draft_revision' };
  if (status === 409 && body && body.code === 'draft_revision_already_launched') {
    // Distinct from stale: the draft did NOT change — this exact revision was
    // already launched (a double-fire / re-click). The remedy is not a
    // resync (there's nothing new to see) but to edit the draft, which mints
    // a fresh revision the bridge will accept.
    return { type: 'draft_revision_already_launched' };
  }
  if (status === 409) {
    return { type: 'conflict', detail: (body && body.detail) || 'the bridge reported a conflict' };
  }
  if (status === 502) return { type: 'bridge_unreachable' };
  return { type: 'error', detail: (body && body.detail) || `HTTP ${status}` };
}

// ---------------------------------------------------------------------------
// Launch banner (FR8) — a launched-frame becomes a visible "revision N
// launched -> run <id>" note that deep-links to the sibling results panel.
// Both pieces below are pure (a URL builder and a detached-DOM builder) so
// the link target and the sink-hardened rendering are unit-testable without
// the live panel, exactly like the reducers/classify above.
// ---------------------------------------------------------------------------

const RESULTS_PANEL_ID = 'scan-results';

/**
 * The URL of the sibling results panel, deep-linked to `runId`. The plan and
 * results panels are served as siblings under the web terminal's
 * `/panel/<id>/` reverse-proxy mount, so swap the plan panel's own trailing
 * id segment for the results panel's and hang the run on the `?run_id=` deep
 * link the results panel already honors (results/panel.js
 * `initialRunIdFromUrl`). `prefix` is the plan panel's mount prefix
 * (`/panel/plan`), or `''` when the panel is served directly with no shell —
 * in which case a best-effort absolute `/panel/<results>` is still produced.
 *
 * @param {string} prefix
 * @param {string} runId
 * @returns {string}
 */
export function resultsPanelUrl(prefix, runId) {
  const base = prefix ? prefix.replace(/[^/]+$/, RESULTS_PANEL_ID) : `/panel/${RESULTS_PANEL_ID}`;
  return `${base}/?run_id=${encodeURIComponent(runId)}`;
}

/**
 * Build the launch banner's detached content: a text prefix plus an anchor
 * to the results panel. Rendered with createElement/textContent only — this
 * panel keeps a strict no-innerHTML posture (see panel.js's module
 * docstring), so an agent/other-tab-supplied `run_id` reaches the DOM only as
 * a text node and inside a URL-encoded `href`, never as parsed markup. The
 * anchor navigates natively (no inline handler, no delegated script needed);
 * `data-run-id` is carried for parity with the panel's other data-* rows and
 * for test assertions.
 *
 * @param {Document} doc
 * @param {{runId: string, revision: number}} banner
 * @param {(runId: string) => string} resultsUrlFor
 * @returns {DocumentFragment}
 */
export function buildLaunchBanner(doc, banner, resultsUrlFor) {
  const frag = doc.createDocumentFragment();
  frag.appendChild(doc.createTextNode(`revision ${banner.revision} launched → `));
  const link = doc.createElement('a');
  link.className = 'launch-run-link';
  link.textContent = `run ${banner.runId}`;
  link.setAttribute('href', resultsUrlFor(banner.runId));
  link.dataset.runId = banner.runId;
  link.target = '_blank';
  link.rel = 'noopener';
  frag.appendChild(link);
  return frag;
}

/**
 * Build the agent-draft banner's detached content: an "agent drafted
 * <plan-name>" note plus a view/switch action. Same posture as
 * `buildLaunchBanner` above: createElement/textContent only — the plan name
 * is agent-influenced data, so it reaches the DOM strictly as a text node
 * (and a `data-plan-name` attribute set via the dataset property), never as
 * parsed markup. The action button invokes `onView(planName)` — panel.js
 * wires that to its own `selectPlan`, which binds to the draft and clears
 * `formDirty`, making the banner predicate go false (self-clearing).
 *
 * @param {Document} doc
 * @param {string} planName
 * @param {(planName: string) => void} onView
 * @returns {DocumentFragment}
 */
export function buildAgentDraftBanner(doc, planName, onView) {
  const frag = doc.createDocumentFragment();
  const text = doc.createElement('span');
  text.className = 'agent-draft-banner-text';
  text.textContent = `agent drafted ${planName}`;
  frag.appendChild(text);
  const view = doc.createElement('button');
  view.type = 'button';
  view.className = 'agent-draft-banner-view';
  view.textContent = 'View draft';
  view.dataset.planName = planName;
  view.addEventListener('click', () => onView(planName));
  frag.appendChild(view);
  return frag;
}

// ---------------------------------------------------------------------------
// SSE transport — the one real-EventSource touchpoint. Manual backoff
// reconnect (capped) rather than relying on the browser's own retry, so a
// dead sidecar doesn't get hammered at a fixed interval forever.
// ---------------------------------------------------------------------------

const _INITIAL_BACKOFF_MS = 1000;
const _MAX_BACKOFF_MS = 15000;

/**
 * @typedef {object} SSEConnection
 * @property {() => void} close
 */

/**
 * @param {string} url
 * @param {{onFrame: (frame: DraftFrame) => void, EventSourceCtor?: typeof EventSource}} opts
 * @returns {SSEConnection}
 */
export function createSSEConnection(url, { onFrame, EventSourceCtor = EventSource }) {
  let closed = false;
  let backoff = _INITIAL_BACKOFF_MS;
  /** @type {EventSource|null} */
  let source = null;
  /** @type {ReturnType<typeof setTimeout>|null} */
  let retryTimer = null;

  function connect() {
    if (closed) return;
    source = new EventSourceCtor(url);
    // Backoff resets on a successful FRAME, not merely an opened connection
    // (`onopen`) — an upstream that accepts the connection and then
    // immediately dies (e.g. a proxy that 200s and drops) would otherwise
    // still count as "success" and reconnect at a fixed interval forever.
    source.onmessage = (event) => {
      backoff = _INITIAL_BACKOFF_MS;
      try {
        onFrame(JSON.parse(event.data));
      } catch {
        // Malformed frame: ignore rather than crash the stream handler.
      }
    };
    source.onerror = () => {
      if (closed) return;
      if (source) source.close();
      const delay = backoff;
      backoff = Math.min(backoff * 2, _MAX_BACKOFF_MS);
      retryTimer = setTimeout(connect, delay);
    };
  }

  connect();

  return {
    close() {
      closed = true;
      if (retryTimer !== null) clearTimeout(retryTimer);
      if (source) source.close();
    },
  };
}

// ---------------------------------------------------------------------------
// Orchestrator
// ---------------------------------------------------------------------------

const PATCH_DEBOUNCE_MS = 400;
const AGENT_NOTE_TIMEOUT_MS = 4000;

/**
 * @typedef {object} DraftClientDeps
 * @property {HTMLFormElement} formEl  The param form; listened on for real
 *   user `input`/`change` events (never `form-change`, which `applyValues`
 *   also dispatches programmatically).
 * @property {() => CollectorLike|null} getCollector  Live getter — panel.js
 *   re-renders the form (and its collector) per selected plan.
 * @property {() => string[]} getPlanNames  Currently-loaded plan names (the
 *   sidebar catalog), for the unknown-draft-plan check.
 * @property {(name: string) => Promise<void>} selectPlan  panel.js's own
 *   plan-selection routine (fetch source, render schema form, wire summary).
 *   Reused for the cross-plan rebind path so schema/source loading logic
 *   lives in exactly one place.
 * @property {() => Promise<void>} refetchPlans  Re-fetch `/plans` once (no
 *   other side effect assumed beyond updating whatever `getPlanNames` reads).
 * @property {() => Promise<DraftGetResponse>} getDraft
 * @property {(body: Record<string, unknown>) => Promise<{ok: boolean, status: number, body: any}>} patchDraft
 * @property {() => Promise<void>} deleteDraft
 * @property {(bound: boolean) => void} onBoundChange
 * @property {(planName: string|null) => void} onAffordance
 * @property {(planName: string|null) => void} onAgentDraftBanner  The
 *   agent-draft banner decision changed: render the "agent drafted
 *   <plan-name>" banner for `planName`, or remove it on `null`. Recomputed at
 *   every affordance-recompute site (and on the formDirty flip that arms the
 *   predicate), purely from `shouldShowAgentDraftBanner(state)` — the banner
 *   owns no imperative show/hide state of its own. While it shows, the
 *   passive affordance is suppressed (never both at once).
 * @property {(planName: string|null) => void} onUnknownPlanBanner
 * @property {(keys: string[]) => void} onAgentEditNote
 * @property {(banner: {runId: string, revision: number}|null) => void} onLaunchBanner
 *   A `launched` frame landed — surface the "revision N launched -> run <id>"
 *   banner (or clear it on `null`). Fires for every launch on the shared
 *   draft, whichever surface (panel or agent) triggered it.
 * @property {(detail: any) => void} onPatchRejected  A 422 from the flush
 *   PATCH — a field-scoped validation error (`{field, error}`, relayed
 *   verbatim from the bridge). The value stays exactly as the operator typed
 *   it (no resync — a 422 is not "the draft moved on", it's "this value is
 *   invalid"); this only surfaces the rejection.
 * @property {(url: string) => string} api
 * @property {string} [clientId]
 * @property {(url: string, opts: {onFrame: (frame: DraftFrame) => void}) => SSEConnection} [sseFactory]
 */

/**
 * @typedef {object} DraftClient
 * @property {(name: string) => Promise<void>} onPlanSelected  Call after
 *   panel.js finishes rendering the newly-selected plan's form.
 * @property {() => Promise<void>} onDiscardClick
 * @property {() => Promise<void>} onAffordanceClick  Bind to the draft the
 *   affordance is pointing at (the currently-selected plan already matches
 *   `draft.plan_name`; there is no fresh "selection" event to hook this to).
 * @property {() => Promise<{patched: boolean, revision: number|null}>} flushNow
 * @property {() => Promise<void>} resync  Force a `GET /draft` resync (e.g.
 *   after Launch's own `stale_draft_revision` 409).
 * @property {() => boolean} isBound
 * @property {() => string|null} getAgentDraftBannerPlan  The agent-draft
 *   banner DECISION (shouldShowAgentDraftBanner): the draft's plan name when
 *   the banner should show, else null. The UI itself is rendered via the
 *   `onAgentDraftBanner` deps callback; this exposes the same display-ready
 *   decision for tests and callers.
 * @property {() => number|null} getLastAppliedRevision
 * @property {() => void} destroy
 */

/**
 * @param {DraftClientDeps} deps
 * @returns {DraftClient}
 */
export function createDraftClient(deps) {
  const clientId = deps.clientId || generateClientId();
  /** @type {DraftState} */
  let state = createInitialState(clientId);
  /** @type {Set<string>} */
  const pendingKeys = new Set();
  /** @type {ReturnType<typeof setTimeout>|null} */
  let debounceTimer = null;
  /** @type {Promise<{patched: boolean, revision: number|null}>|null} */
  let inFlightFlush = null;
  /** @type {ReturnType<typeof setTimeout>|null} */
  let agentNoteTimer = null;
  // Bumped on every `GET /draft` fetch this client initiates (bind, resync,
  // gap-recovery, a 409's drop+resync). A late-resolving fetch whose epoch
  // has since been superseded by a newer one is discarded rather than
  // applied — otherwise a slow resync could reset the baseline BACKWARD over
  // a snapshot a more recent call already applied.
  let resyncEpoch = 0;

  function renderBoundIndicator() {
    deps.onBoundChange(state.bound);
  }

  /**
   * Recompute both agent-facing hints from the current state, declaratively:
   * the agent-draft banner (`shouldShowAgentDraftBanner`) and the passive
   * bind affordance (`shouldShowAffordance`). The two predicates are
   * mutually exclusive by construction (the banner requires
   * `draftPlanName !== selectedName`, the affordance the opposite), but the
   * suppression is still made explicit here so they can never render
   * together even if the predicates ever drift.
   */
  function renderAffordance() {
    const bannerPlan = shouldShowAgentDraftBanner(state) ? state.draftPlanName : null;
    deps.onAgentDraftBanner(bannerPlan);
    deps.onAffordance(
      bannerPlan === null && shouldShowAffordance(state) ? state.draftPlanName : null
    );
  }

  /**
   * Whenever the draft's plan_name changes to a non-null value, confirm it's
   * still in the loaded catalog (PROPOSAL.md: unknown plan_name -> refetch
   * `/plans` once, then a visible banner if it's still missing). Independent
   * of bound/selected state — this is a fact about the draft as a whole.
   *
   * @returns {Promise<boolean>} whether the plan is known (after the refetch).
   */
  async function ensureDraftPlanKnown() {
    const name = state.draftPlanName;
    if (name === null || deps.getPlanNames().includes(name)) {
      deps.onUnknownPlanBanner(null);
      return true;
    }
    await deps.refetchPlans();
    if (deps.getPlanNames().includes(name)) {
      deps.onUnknownPlanBanner(null);
      return true;
    }
    deps.onUnknownPlanBanner(name);
    return false;
  }

  /**
   * Fetch a fresh `GET /draft` snapshot and apply it via `reduceReset` —
   * unless a newer resync/bind was started while this fetch was in flight
   * (see `resyncEpoch` above), in which case this call's snapshot is
   * discarded and `state` is left untouched for the newer call to own. A
   * failed fetch (e.g. a transient bridge-unreachable 502) is likewise
   * absorbed here rather than thrown — every caller of this helper is either
   * a fire-and-forget internal path (an SSE frame handler, a debounced
   * flush) or a public method the panel invokes without its own try/catch, so
   * centralizing "network hiccup -> leave state as-is, let the next attempt
   * retry" here keeps every one of them self-healing for free.
   *
   * @returns {Promise<boolean>} whether this call's snapshot was applied.
   */
  async function fetchAndApplyReset() {
    const myEpoch = ++resyncEpoch;
    /** @type {DraftGetResponse} */
    let snapshot;
    try {
      snapshot = await deps.getDraft();
    } catch {
      return false;
    }
    if (myEpoch !== resyncEpoch) return false;
    state = reduceReset(state, snapshot);
    return true;
  }

  /**
   * Flash exactly the given top-level field names (whole-value containers
   * flash as one element, same as scalars — `fields[name].el` is already the
   * container for the axes table / chip well / nested object) via the shared
   * design-system agent-activity flash (`.agent-flash`; reflow-restart and
   * animationend cleanup live in `flashElement` itself).
   *
   * @param {string[]} names
   */
  function flashFields(names) {
    const collector = deps.getCollector();
    if (!collector) return;
    for (const name of names) {
      const field = collector.fields[name];
      if (field) flashElement(field.el);
    }
  }

  /** @param {string[]} keys */
  function showAgentEditNote(keys) {
    if (keys.length === 0) return;
    if (agentNoteTimer !== null) clearTimeout(agentNoteTimer);
    deps.onAgentEditNote(keys);
    agentNoteTimer = setTimeout(() => {
      deps.onAgentEditNote([]);
      agentNoteTimer = null;
    }, AGENT_NOTE_TIMEOUT_MS);
  }

  /**
   * Apply the current `state.draftArgs` fully onto the live form (bind/resync
   * path). Every top-level registered field is included in the applied
   * payload — present ones from `draftArgs`, and any field the draft doesn't
   * (no longer) carry as an explicit `undefined` — so a key removed since the
   * last snapshot resets that field to its schema default instead of
   * silently keeping its stale prior value (`setValue(undefined)` is
   * documented schema-form.js behavior for exactly this).
   *
   * This is a full, unconditional overwrite of the visible form — any
   * not-yet-flushed pending edit on a field the snapshot also touches is
   * intentionally discarded, on the same "resync = ground truth" principle
   * as a 409 drop+resync (broader than the spec's 409-only wording, but
   * deliberately so: a hello/gap resync means this client's view of the
   * draft was stale, so a local edit made against that stale view is not
   * safe to re-apply over the fresh snapshot either).
   */
  function applyFullSnapshot() {
    const collector = deps.getCollector();
    if (!collector) return;
    const draftArgs = state.draftArgs || {};
    /** @type {Record<string, unknown>} */
    const payload = { ...draftArgs };
    for (const name of Object.keys(collector.fields)) {
      if (!Object.prototype.hasOwnProperty.call(payload, name)) payload[name] = undefined;
    }
    collector.applyValues(payload);
  }

  /**
   * Bind to the currently-selected plan against a fresh `GET /draft`
   * snapshot (PROPOSAL.md: binding "seeds the form from GET /draft, never
   * from bare schema defaults"). Used by both the auto-bind-on-selection path
   * and the affordance click.
   */
  async function bindFromFreshSnapshot() {
    const applied = await fetchAndApplyReset();
    if (!applied) return;
    if (state.draftPlanName !== state.selectedName) {
      // The draft moved on again between the affordance showing and the
      // click landing; don't bind to stale intent.
      state = { ...state, bound: false };
      renderBoundIndicator();
      renderAffordance();
      await ensureDraftPlanKnown();
      return;
    }
    // Bind completion also clears formDirty: the form now mirrors the shared
    // draft, so any prior unbound manual edits are gone from the screen.
    state = { ...state, bound: true, formDirty: false };
    pendingKeys.clear();
    applyFullSnapshot();
    renderBoundIndicator();
    renderAffordance();
    await ensureDraftPlanKnown();
  }

  /**
   * Attempt to follow a plan-change (a `plan-change` frame, or a hello/resync
   * landing on a different plan_name than the one displayed) while bound:
   * rebind to whatever plan the draft now names, or unbind with the
   * unknown-plan banner if that plan isn't in the loaded catalog.
   */
  async function rebindToDraftPlan() {
    const name = state.draftPlanName;
    if (name === null) {
      state = { ...state, bound: false };
      pendingKeys.clear();
      renderBoundIndicator();
      renderAffordance();
      return;
    }
    const known = await ensureDraftPlanKnown();
    if (!known) {
      state = { ...state, bound: false };
      pendingKeys.clear();
      renderBoundIndicator();
      renderAffordance();
      return;
    }
    // selectPlan re-renders the target plan's form and, on completion, calls
    // onPlanSelected(name) itself — which re-checks draftPlanName against the
    // (now-updated) selectedName and performs the actual bind/apply. Reusing
    // it here avoids a second, divergent apply path.
    await deps.selectPlan(name);
  }

  /**
   * Auto-switch onto the agent's draft plan (`shouldAutoSwitch` said yes):
   * delegate to `deps.selectPlan`, which re-renders the target plan's form
   * and re-enters `onPlanSelected` — performing the actual bind — then flash
   * ALL keys of the applied draft args via the shared design-system flash.
   * The whole applied arg set (not the triggering frame's `changed[]`) is
   * flashed deliberately: the form was just re-rendered from scratch, so
   * every visible value is newly-landed agent content and the frame's
   * `changed[]` is stale by re-render time anyway. If the switch-bind did
   * not complete (selection raced onward, plan failed to load), nothing
   * flashes.
   */
  async function switchToDraftAndFlash() {
    const name = state.draftPlanName;
    if (name === null) return;
    await deps.selectPlan(name);
    const appliedArgs = state.draftArgs;
    if (state.bound && appliedArgs) {
      flashFields(Object.keys(appliedArgs));
    }
  }

  /**
   * Post-processing shared by a hello frame and a resync `GET /draft` —
   * `state` already reflects the reset. When bound to the still-matching
   * plan, `pendingKeys.clear()` here is a deliberate decision (see
   * `applyFullSnapshot`'s docstring), not an oversight: any resync discards
   * unflushed local edits in favor of the fresh snapshot.
   *
   * Unbound, the reset is also an auto-switch decision point: `prevState`
   * (the state BEFORE the reset was ingested — it carries the last-seen
   * `lastUpdatedAt` pair and the pre-reset baseline) lets `shouldAutoSwitch`
   * tell a first-open / genuinely-new agent draft apart from a routine
   * reconnect hello or internal resync repeating an already-seen snapshot.
   *
   * @param {DraftState} prevState
   */
  async function afterReset(prevState) {
    if (state.bound) {
      if (state.draftPlanName === state.selectedName) {
        pendingKeys.clear();
        applyFullSnapshot();
        renderBoundIndicator();
        renderAffordance();
        await ensureDraftPlanKnown();
      } else {
        await rebindToDraftPlan();
      }
    } else {
      const decision = shouldAutoSwitch(prevState, { state, action: { type: 'reset' } });
      renderAffordance();
      const known = await ensureDraftPlanKnown();
      if (decision.switch && known) await switchToDraftAndFlash();
    }
  }

  /** @param {{state: DraftState, action: FrameAction}} result */
  async function handleReducedResult(result) {
    // Captured before the reducer's state lands: shouldAutoSwitch compares
    // the pre-ingestion state against the reduced result.
    const prevState = state;
    state = result.state;
    const action = result.action;

    if (action.type === 'drop') return;

    if (action.type === 'launched') {
      // Independent of bound/selected state and of the revision baseline —
      // reduceFrame already recorded the fact; just surface it.
      deps.onLaunchBanner(state.launchBanner);
      return;
    }

    if (action.type === 'resync') {
      const applied = await fetchAndApplyReset();
      if (!applied) return;
      await afterReset(prevState);
      return;
    }

    if (action.type === 'reset') {
      // A hello frame is itself a baseline reset — bump the epoch so any
      // still-in-flight `fetchAndApplyReset()` (an earlier bind/resync) is
      // treated as superseded and can never later overwrite THIS newer
      // baseline with a stale snapshot (mirrors why `fetchAndApplyReset`
      // itself bumps the epoch before every fetch it starts; without this,
      // a slow fetch resolving after a post-restart hello could roll the
      // revision counter backward and silently drop every subsequent frame
      // as "stale" until the next reconnect).
      resyncEpoch++;
      await afterReset(prevState);
      return;
    }

    if (action.type === 'echo') {
      // Own-origin: baseline already advanced by reduceFrame; skip value
      // re-application and flash, but the affordance fact may still need
      // recomputing (defensive — the panel never sets plan_name itself, so
      // draftPlanName cannot actually change via an echo today).
      renderAffordance();
      return;
    }

    // action.type === 'apply'
    const frame = action.frame;
    if (!state.bound) {
      // Live-path auto-switch decision (agent-origin frames only — the pure
      // predicate owns the AGENT_ORIGIN check; echo/operator-tab frames never
      // switch). The banner counterpart of the decision is rendered by
      // renderAffordance() below via the onAgentDraftBanner callback.
      const decision = shouldAutoSwitch(prevState, result);
      renderAffordance();
      const known = await ensureDraftPlanKnown();
      if (decision.switch && known) await switchToDraftAndFlash();
      return;
    }

    if (frame.type === 'clear') {
      state = { ...state, bound: false };
      pendingKeys.clear();
      renderBoundIndicator();
      renderAffordance();
      deps.onUnknownPlanBanner(null);
      return;
    }

    if (frame.type === 'plan-change' && state.draftPlanName !== state.selectedName) {
      await rebindToDraftPlan();
      return;
    }

    // 'change', or a same-name 'plan-change' (defensive): apply exactly the
    // changed[] keys and flash them. A key in changed[] that is no longer
    // present in draftArgs (removed by the agent/other tab) is applied as an
    // explicit `undefined` — an own-property `undefined` still passes
    // applyValues' `hasOwnProperty` gate, and `setValue(undefined)` resets
    // that field to its schema default (documented schema-form.js
    // behavior) — so a removal actually clears the field instead of leaving
    // its stale prior value on screen.
    const changed = frame.changed || [];
    const collector = deps.getCollector();
    if (collector && changed.length > 0) {
      const draftArgs = state.draftArgs || {};
      /** @type {Record<string, unknown>} */
      const subset = {};
      for (const key of changed) {
        subset[key] = Object.prototype.hasOwnProperty.call(draftArgs, key) ? draftArgs[key] : undefined;
      }
      collector.applyValues(subset);
      // Last-writer-wins: a remote edit landing on a key the operator was
      // mid-editing (still pending, not yet flushed) overrides that local
      // edit — clear it from pendingKeys so the next flush doesn't PATCH the
      // server's own value straight back as a redundant echo write.
      for (const key of changed) pendingKeys.delete(key);
      flashFields(changed);
      showAgentEditNote(changed);
    }
  }

  /** @param {DraftFrame} frame */
  function onFrame(frame) {
    void handleReducedResult(reduceFrame(state, frame));
  }

  const sseFactory = deps.sseFactory || createSSEConnection;
  const connection = sseFactory(deps.api('/draft/events'), { onFrame });

  /**
   * A real user edit — native `input`/`change` for typed fields, plus
   * structural edits (chip/row add-remove, segmented clicks) which
   * schema-form.js's builders express only as a bubbling `form-change`
   * CustomEvent, never a native input/change (schema-form.js's builder
   * `emitChange` call sites). `applyValues` dispatches its own single
   * `form-change` directly ON the form element itself (schema-form.js's
   * `withFieldRegistry`) — a *structural* user edit's `form-change` instead
   * bubbles up FROM the edited widget's element, so `event.target ===
   * deps.formEl` is exactly the programmatic-apply case and nothing else.
   *
   * @param {Event} event
   */
  function onUserEdit(event) {
    if (event.type === 'form-change' && event.target === deps.formEl) return;
    if (!state.bound) {
      // An unbound manual edit dirties the form — suppressing agent
      // auto-switch and arming the agent-draft banner (shouldAutoSwitch /
      // shouldShowAgentDraftBanner) — but never marks pending keys or
      // schedules a PATCH: unbound edits are purely local. Cleared by
      // onPlanSelected and by bind completion. The flip is itself a banner
      // recompute point: an agent draft already sitting on another plan must
      // surface the moment the operator's edit arms the predicate, not wait
      // for the next frame.
      if (!state.formDirty) {
        state = { ...state, formDirty: true };
        renderAffordance();
      }
      return;
    }
    const collector = deps.getCollector();
    if (!collector) return;
    const target = /** @type {Node} */ (event.target);
    for (const [name, field] of Object.entries(collector.fields)) {
      if (field.el.contains(target)) {
        pendingKeys.add(name);
        scheduleFlush();
        break;
      }
    }
  }
  deps.formEl.addEventListener('input', onUserEdit);
  deps.formEl.addEventListener('change', onUserEdit);
  deps.formEl.addEventListener('form-change', onUserEdit);

  function scheduleFlush() {
    if (debounceTimer !== null) clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      debounceTimer = null;
      void flush();
    }, PATCH_DEBOUNCE_MS);
  }

  /** One PATCH attempt over the currently-pending keys. */
  async function doOneFlush() {
    if (pendingKeys.size === 0) return { patched: false, revision: null };
    const collector = deps.getCollector();
    const fullArgs = collector ? collector() : {};
    const keysToFlush = new Set(pendingKeys);
    const { plan_args_patch, remove } = computeDelta(fullArgs, keysToFlush);
    const body = {
      plan_args_patch,
      remove,
      client_id: clientId,
      expected_plan_name: state.selectedName,
    };
    /** @type {{ok: boolean, status: number, body: any}|null} */
    let result;
    try {
      result = await deps.patchDraft(body);
    } catch {
      // A network-level failure (not an HTTP error status): treat the same
      // as the generic drop+resync branch below.
      result = null;
    }
    for (const key of keysToFlush) pendingKeys.delete(key);
    if (!result || !result.ok) {
      if (result && result.status === 422) {
        // A field-scoped validation rejection: this is "the value is
        // invalid", not "the draft moved on" — keep it exactly as the
        // operator typed it (no resync) and just surface the rejection.
        deps.onPatchRejected(result.body && result.body.detail);
        return { patched: false, revision: null };
      }
      // 409 (no_draft / plan_name_mismatch), a network failure, or any other
      // unexpected failure status: drop the edit and resync to ground truth
      // (PROPOSAL.md pending-key rule).
      const prevState = state;
      const applied = await fetchAndApplyReset();
      if (applied) await afterReset(prevState);
      return { patched: false, revision: null };
    }
    return { patched: true, revision: result.body.revision };
  }

  /**
   * Flush pending edits, looping while new keys accumulate mid-flight — a
   * user edit landing between this call's PATCH request and response must
   * not be silently dropped, since Launch pins whatever this ultimately
   * returns.
   *
   * @returns {Promise<{patched: boolean, revision: number|null}>}
   */
  async function flush() {
    if (inFlightFlush) return inFlightFlush;
    inFlightFlush = (async () => {
      /** @type {{patched: boolean, revision: number|null}} */
      let result = { patched: false, revision: null };
      while (pendingKeys.size > 0) {
        result = await doOneFlush();
      }
      return result;
    })();
    try {
      return await inFlightFlush;
    } finally {
      inFlightFlush = null;
    }
  }

  return {
    async onPlanSelected(name) {
      if (debounceTimer !== null) {
        clearTimeout(debounceTimer);
        debounceTimer = null;
      }
      // A fresh selection abandons any unbound manual edits along with the
      // form they lived in — formDirty resets with it.
      state = { ...state, selectedName: name, bound: false, formDirty: false };
      pendingKeys.clear();
      renderBoundIndicator();
      if (state.draftPlanName !== null && state.draftPlanName === name) {
        await bindFromFreshSnapshot();
      } else {
        renderAffordance();
        await ensureDraftPlanKnown();
      }
    },

    async onDiscardClick() {
      try {
        await deps.deleteDraft();
      } catch {
        // Sidecar-unreachable or similar: the discard did NOT actually
        // happen server-side, so leave `bound` exactly as it was (never
        // optimistically unbind a draft that's still there) and surface the
        // failure via the same rejection-banner channel a 422 uses.
        deps.onPatchRejected('could not discard the draft — the sidecar may be unreachable');
        return;
      }
      state = { ...state, bound: false };
      pendingKeys.clear();
      renderBoundIndicator();
      renderAffordance();
    },

    async onAffordanceClick() {
      if (state.draftPlanName === null || state.draftPlanName !== state.selectedName) return;
      await bindFromFreshSnapshot();
    },

    async flushNow() {
      if (debounceTimer !== null) {
        clearTimeout(debounceTimer);
        debounceTimer = null;
      }
      return flush();
    },

    async resync() {
      const prevState = state;
      const applied = await fetchAndApplyReset();
      if (applied) await afterReset(prevState);
    },

    isBound() {
      return state.bound;
    },

    getAgentDraftBannerPlan() {
      return shouldShowAgentDraftBanner(state) ? state.draftPlanName : null;
    },

    getLastAppliedRevision() {
      return state.lastAppliedRevision;
    },

    destroy() {
      connection.close();
      if (debounceTimer !== null) clearTimeout(debounceTimer);
      if (agentNoteTimer !== null) clearTimeout(agentNoteTimer);
      deps.formEl.removeEventListener('input', onUserEdit);
      deps.formEl.removeEventListener('change', onUserEdit);
      deps.formEl.removeEventListener('form-change', onUserEdit);
    },
  };
}
