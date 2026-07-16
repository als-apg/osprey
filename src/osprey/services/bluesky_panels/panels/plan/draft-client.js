// @ts-check
/**
 * draft-client — the plan panel's live view of the server-held shared plan
 * draft (PROPOSAL.md "Plan panel integration" + "Execute revision gate").
 *
 * Split out of panel.js to keep panel.js focused on plan browsing/execute
 * chrome, and — more importantly — so the revision/binding/pending-key rules
 * are a pure, DOM-light state machine that vitest can drive with plain frame
 * objects, without a real `EventSource` (happy-dom does not implement one;
 * see `createSSEConnection`, the one function here that touches it, kept
 * deliberately thin and injectable).
 *
 * Three layers, outside in:
 *
 * 1. Pure reducers (`createInitialState`, `reduceFrame`, `reduceReset`,
 *    `computeDelta`, `shouldShowAffordance`, `resolvePinnedRevision`) — no
 *    DOM, no network, no timers. These encode every rule from the proposal:
 *    revision drop/apply/resync, hello/resync baseline reset (including
 *    backward), own-origin echo suppression, and the minimal-delta
 *    PATCH-back shape (blank -> `remove[]`).
 * 2. `createDraftClient(deps)` — the orchestrator. Wires the reducers to a
 *    form (pending-key tracking, debounced PATCH-back), a set of DOM
 *    callbacks (bound indicator, affordance, unknown-plan banner, flash,
 *    agent-edited note) and a set of injectable HTTP functions
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

/**
 * @typedef {object} DraftSnapshot
 * @property {string} plan_name
 * @property {Record<string, unknown>} plan_args
 * @property {string|null} [updated_by]
 * @property {string} [updated_at]
 */

/**
 * @typedef {object} DraftFrame
 * @property {'hello'|'change'|'clear'|'plan-change'} type
 * @property {DraftSnapshot|null} draft
 * @property {string[]} [changed]
 * @property {number} revision
 * @property {string|null} [origin]
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
  };
}

/**
 * @typedef {{type: 'drop'}
 *   | {type: 'resync'}
 *   | {type: 'reset'}
 *   | {type: 'apply', frame: DraftFrame}
 *   | {type: 'echo', frame: DraftFrame}} FrameAction
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
 * Resolve the `draft_revision` to pin for Execute: the just-flushed PATCH's
 * own response revision when a flush actually happened, else the last
 * applied frame/hello baseline (PROPOSAL.md "Execute revision gate").
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
 * The panel's own Execute request body for the two mutually-exclusive launch
 * modes (PROPOSAL.md "Execute revision gate"): bound mode sends only
 * `{draft_revision}` — never `plan_name`/`plan_args` alongside, so the
 * launched args always come from the bridge's pinned-revision snapshot, not
 * this body; unbound (manual) mode sends the collected form args exactly as
 * it did before draft mode existed.
 *
 * @param {{bound: boolean, pinnedRevision: number|null, planName: string, planArgs: Record<string, unknown>}} input
 * @returns {Record<string, unknown>}
 */
export function buildExecuteRequestBody({ bound, pinnedRevision, planName, planArgs }) {
  return bound ? { draft_revision: pinnedRevision } : { plan_name: planName, plan_args: planArgs };
}

/**
 * @typedef {{type: 'writes_not_armed'}
 *   | {type: 'run_started', runId: string}
 *   | {type: 'stale_draft_revision'}
 *   | {type: 'conflict', detail: string}
 *   | {type: 'bridge_unreachable'}
 *   | {type: 'error', detail: string}} ExecuteOutcome
 */

/**
 * Classify `POST /runs/execute`'s response into a display-ready outcome —
 * pure and DOM/fetch-free, so the status/`code` branching (in particular
 * distinguishing the machine-readable `stale_draft_revision` discriminator
 * from a bare bridge-relayed 409) is unit-testable without a real panel.
 *
 * @param {number} status
 * @param {any} body
 * @returns {ExecuteOutcome}
 */
export function classifyExecuteResponse(status, body) {
  if (status === 200 && body && body.status === 'writes_not_armed') return { type: 'writes_not_armed' };
  if (status === 200 && body && body.run_id) return { type: 'run_started', runId: String(body.run_id) };
  if (status === 409 && body && body.code === 'stale_draft_revision') return { type: 'stale_draft_revision' };
  if (status === 409) {
    return { type: 'conflict', detail: (body && body.detail) || 'the bridge reported a conflict' };
  }
  if (status === 502) return { type: 'bridge_unreachable' };
  return { type: 'error', detail: (body && body.detail) || `HTTP ${status}` };
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
 * @property {(planName: string|null) => void} onUnknownPlanBanner
 * @property {(keys: string[]) => void} onAgentEditNote
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
 *   after Execute's own `stale_draft_revision` 409).
 * @property {() => boolean} isBound
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

  function renderAffordance() {
    deps.onAffordance(shouldShowAffordance(state) ? state.draftPlanName : null);
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
   * container for the axes table / chip well / nested object). Reflow-restart
   * so two rapid edits to the same field re-fire the animation.
   *
   * @param {string[]} names
   */
  function flashFields(names) {
    const collector = deps.getCollector();
    if (!collector) return;
    for (const name of names) {
      const field = collector.fields[name];
      if (!field) continue;
      const el = field.el;
      el.classList.remove('draft-flash');
      void el.offsetWidth;
      el.classList.add('draft-flash');
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
    state = { ...state, bound: true };
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
   * Post-processing shared by a hello frame and a resync `GET /draft` —
   * `state` already reflects the reset. When bound to the still-matching
   * plan, `pendingKeys.clear()` here is a deliberate decision (see
   * `applyFullSnapshot`'s docstring), not an oversight: any resync discards
   * unflushed local edits in favor of the fresh snapshot.
   */
  async function afterReset() {
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
      renderAffordance();
      await ensureDraftPlanKnown();
    }
  }

  /** @param {{state: DraftState, action: FrameAction}} result */
  async function handleReducedResult(result) {
    state = result.state;
    const action = result.action;

    if (action.type === 'drop') return;

    if (action.type === 'resync') {
      const applied = await fetchAndApplyReset();
      if (!applied) return;
      await afterReset();
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
      await afterReset();
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
      renderAffordance();
      await ensureDraftPlanKnown();
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
    if (!state.bound) return;
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
      const applied = await fetchAndApplyReset();
      if (applied) await afterReset();
      return { patched: false, revision: null };
    }
    return { patched: true, revision: result.body.revision };
  }

  /**
   * Flush pending edits, looping while new keys accumulate mid-flight — a
   * user edit landing between this call's PATCH request and response must
   * not be silently dropped, since Execute pins whatever this ultimately
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
      state = { ...state, selectedName: name, bound: false };
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
      const applied = await fetchAndApplyReset();
      if (applied) await afterReset();
    },

    isBound() {
      return state.bound;
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
