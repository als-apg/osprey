// @ts-check
/**
 * BLUESKY panel — the Plans view: a two-pane operator console for the
 * registered plans.
 *
 * LEFT (sidebar): a dense, file-browser-style selector (plan-browser.js).
 * Plans are grouped under collapsible provenance folders and filterable; each
 * row is name-first with trust/validation compressed into a single status dot
 * (full detail in the tooltip and the detail header). The first plan is
 * auto-selected on load so the view never opens onto an empty pane.
 *
 * RIGHT (detail): the selected plan under a two-tab strip —
 *   - Parameters: a 2-D GUI generated from the plan's JSON Schema
 *     (schema-form.js): chip editors for device lists, an editable table for
 *     grid axes, typed inputs for scalars — arranged by a per-plan layout and
 *     carrying a per-plan live readout that recomputes on every edit, e.g.
 *     "2 correctors × 7 points = 14 sweep points" (both from
 *     plan-presentation.js).
 *   - Source: the plan's source code.
 * plus the local Reset and the deterministic Add-to-queue action in the footer.
 *
 * Plans absent from the two registries still render fully — the schema-driven
 * form auto-flows their fields — so facility/session plans need no panel-side
 * code to be usable.
 *
 * Every fetch is issued through the shell's prefix-relative `api()` (the
 * web-terminal proxy does not rewrite plain "/plans"-style absolute paths), so
 * this view works unmodified whether the panel is opened directly on the
 * sidecar or embedded.
 *
 * Queueing is deterministic: no agent/LLM in this path. The view POSTs the
 * sidecar's `/queue/items` relay with a pinned draft revision; the sidecar
 * attaches the launch token, so the browser never sees or sends one. Adding
 * requires a two-step in-panel confirm and is enabled only when the selected
 * plan's `validated` flag (from the source response) is `true` AND the
 * bridge's capability record says this deployment can execute plans at all.
 *
 * Three actions, three different blast radii — kept visibly distinct because
 * confusing them is the expensive mistake:
 *   - Reset (footer) is LOCAL: re-renders this form from schema defaults and
 *     sends nothing anywhere. It does record that the form now diverges from
 *     the shared draft, so a later enqueue pushes what is on screen rather
 *     than pinning a stale revision.
 *   - Discard (draft row) is SHARED: deletes the draft on the bridge, for
 *     every panel and for the agent. Two-step confirm, with the consequence
 *     spelled out.
 *   - Add to queue is SHARED and consequential: the queued item runs as soon
 *     as someone starts the queue.
 *
 * The whole UI is built with createElement/textContent (via the shared `h`
 * helper in hyperscript.js) — no innerHTML sink anywhere — so plan-authored
 * strings (names, descriptions, source, enum values) are never interpreted as
 * markup.
 *
 * @module plans-view
 */

import { h } from './hyperscript.js';
import {
  dotClass,
  groupByProvenance,
  renderPlanTree as renderPlanTreeInto,
} from './plan-browser.js';
import { planLayout, summarizePlanArgs } from './plan-presentation.js';
import { renderSchemaForm } from './schema-form.js';
import { initSplitter } from '/design-system/js/splitter.js';
import {
  createDraftClient,
  resolvePinnedRevision,
  generateClientId,
  buildQueueAddBody,
  buildDraftReplaceBody,
  describeDraftReplaceFailure,
  classifyQueueAddResponse,
  queueOutcomeBanner,
  classifyCapability,
  capabilityBanner,
  resetNoticeMessage,
  DISCARD_CONFIRM_NOTICE,
  REASON_BRIDGE_UNREACHABLE,
  buildLaunchBanner,
  buildAgentDraftBanner,
} from './draft-client.js';

/**
 * @typedef {object} PlansViewDeps
 * @property {HTMLElement} root  The panel root; every element this view drives
 *   is looked up beneath it.
 * @property {(path: string) => string} api  Prefix-relative URL builder.
 * @property {(runId: string) => void} onOpenRun  A queued run should be
 *   opened: the shell selects it and switches to the Results tab. This is the
 *   whole of the old cross-panel handoff — the queue and the run's rows are in
 *   this same panel now.
 */

/**
 * @typedef {object} PlansView
 * @property {(text: string) => void} setFilter  Apply a plan-name filter. The
 *   ONE function behind both the in-body search box and the tile bar's
 *   contributed `search` item, so the two can never diverge.
 * @property {(mode: 'expert'|'simple') => void} onModeChange
 */

/**
 * Colour an already-populated source block with highlight.js, if it loaded.
 *
 * `hljs` is a page global set by the classic <script> in index.html, so it is
 * absent under unit tests and whenever the CDN is unreachable. Every call is
 * therefore guarded and every throw swallowed: the caller has already written
 * the source as inert `textContent`, and losing colour is a far smaller cost
 * than losing the operator's ability to read the plan.
 *
 * The grammar is named by the element's `language-python` class rather than
 * left to `highlightAuto`, whose guess is unreliable on the short, import-free
 * snippets many plans are.
 *
 * @param {HTMLElement} el  Source block holding the plan text.
 * @returns {void}
 */
function highlightSource(el) {
  const { hljs } = /** @type {any} */ (globalThis);
  if (!hljs || typeof hljs.highlightElement !== 'function') return;
  // hljs refuses to touch an element it has already marked; this block is
  // repainted on every plan selection, so clear the mark before each pass.
  delete el.dataset.highlighted;
  try {
    hljs.highlightElement(el);
  } catch {
    // Inert textContent stays on screen -- see above.
  }
}

/**
 * @param {PlansViewDeps} deps
 * @returns {PlansView}
 */
export function createPlansView({ root, api, onOpenRun }) {
  /**
   * The panel owns its own shell, so every id below is present by
   * construction — a missing one is a bundle bug, not a runtime condition to
   * branch on.
   *
   * @param {string} id
   * @returns {HTMLElement}
   */
  function byId(id) {
    return /** @type {HTMLElement} */ (root.querySelector(`#${id}`));
  }

  /** @typedef {import('./plan-browser.js').PlanSummary} PlanSummary */

  /**
   * @typedef {object} PlanSource
   * @property {string} name
   * @property {string} provenance
   * @property {boolean} validated
   * @property {boolean} truncated
   * @property {string} source
   */

  /**
   * Fetch a sidecar route and parse its JSON body, tolerating a non-JSON body
   * (returned as `null`) so callers branch on `{ok, status, body}` without
   * their own try/catch around `response.json()`. Network failures still
   * reject — callers that must distinguish "unreachable" keep their catch.
   *
   * @param {string} path
   * @param {RequestInit} [init]
   * @returns {Promise<{ok: boolean, status: number, body: any}>}
   */
  async function fetchJson(path, init) {
    const response = await fetch(api(path), init);
    /** @type {any} */
    let body = null;
    try {
      body = await response.json();
    } catch {
      body = null;
    }
    return { ok: response.ok, status: response.status, body };
  }

  /** @type {PlanSummary[]} */
  let plans = [];
  /** @type {string|null} */
  let selectedName = null;
  /** @type {PlanSource|null} */
  let selectedSource = null;
  /** @type {import('./schema-form.js').PlanArgsCollector|null} */
  let collectPlanArgs = null;
  let filterText = '';
  let confirmArmed = false;
  let discardArmed = false;
  let queueing = false;

  /**
   * Whether the form may differ from the shared draft in a way no incremental
   * flush can express — set by Reset, which blanks fields back to schema
   * defaults WITHOUT marking pending keys (that is what makes Reset local).
   *
   * Load-bearing for the enqueue: `flushNow()` would find nothing pending and
   * pin the PRE-reset revision, so the bridge would enqueue values the operator
   * can no longer see. While this is set, the enqueue pushes the whole form
   * instead of pinning a baseline. Cleared once that push lands, on a bind
   * (which re-applies the draft over the form wholesale), and on plan
   * selection. Left set on a FAILED push — the divergence is still real.
   */
  let formResetSincePatch = false;
  /** @type {ReturnType<typeof setTimeout>|null} */
  let draftRejectedTimer = null;
  const DRAFT_REJECTED_NOTE_TIMEOUT_MS = 5000;

  /**
   * Whether this deployment can execute plans, from the bridge's capability
   * record. Starts as cannot-execute with the unreachable sentinel so the panel
   * fails CLOSED: Add-to-queue stays disabled until a real `/bridge/health`
   * answer says otherwise, rather than being briefly live during boot.
   *
   * @type {import('./draft-client.js').CapabilityRecord}
   */
  let capability = {
    canExecute: false,
    reason: REASON_BRIDGE_UNREACHABLE,
    detail: 'Checking whether this deployment can execute plans…',
  };

  // The capability record is re-read on a slow interval, not only at boot: the
  // connector/config reasons are settled until a redeploy (which reloads this
  // panel anyway), but `manager_unreachable` clears on its own the moment a
  // restarted queueserver comes back — and an operator staring at a disabled
  // button should not have to reload to find that out.
  const CAPABILITY_REFRESH_MS = 30000;

  // ---- element lookups ----

  const rootErrorEl = /** @type {HTMLElement} */ (byId('root-error'));
  const planTreeEl = /** @type {HTMLElement} */ (byId('plan-tree'));
  const plansEmptyEl = /** @type {HTMLElement} */ (byId('plans-empty'));
  const plansFilteredEmptyEl = /** @type {HTMLElement} */ (
    byId('plans-filtered-empty')
  );
  const searchEl = /** @type {HTMLInputElement} */ (byId('plan-search'));
  const detailEmptyEl = /** @type {HTMLElement} */ (byId('detail-empty'));
  const detailBodyEl = /** @type {HTMLElement} */ (byId('detail-body'));
  const detailTitleEl = /** @type {HTMLElement} */ (byId('detail-title'));
  const detailStatusEl = /** @type {HTMLElement} */ (byId('detail-status'));
  const detailDescEl = /** @type {HTMLElement} */ (byId('detail-desc'));
  const sessionNoteEl = /** @type {HTMLElement} */ (byId('session-note'));
  const detailSourceEl = /** @type {HTMLElement} */ (byId('detail-source'));
  const paramFormEl = /** @type {HTMLFormElement} */ (byId('param-form'));
  const paramSummaryEl = /** @type {HTMLElement} */ (byId('param-summary'));
  const launchOutcomeBannerEl = /** @type {HTMLElement} */ (
    byId('launch-outcome-banner')
  );
  const capabilityBannerEl = /** @type {HTMLElement} */ (
    byId('capability-banner')
  );
  const unvalidatedNoteEl = /** @type {HTMLElement} */ (byId('unvalidated-note'));
  const queueAddBtnEl = /** @type {HTMLButtonElement} */ (byId('queue-add-btn'));
  const resetBtnEl = /** @type {HTMLButtonElement} */ (byId('reset-btn'));
  const tabParamsEl = /** @type {HTMLButtonElement} */ (byId('tab-params'));
  const tabSourceEl = /** @type {HTMLButtonElement} */ (byId('tab-source'));
  const panelParamsEl = /** @type {HTMLElement} */ (byId('panel-params'));
  const panelSourceEl = /** @type {HTMLElement} */ (byId('panel-source'));
  const draftUnknownBannerEl = /** @type {HTMLElement} */ (
    byId('draft-unknown-banner')
  );
  const draftIndicatorEl = /** @type {HTMLElement} */ (byId('draft-indicator'));
  const draftDiscardBtnEl = /** @type {HTMLButtonElement} */ (
    byId('draft-discard-btn')
  );
  const draftDiscardNoteEl = /** @type {HTMLElement} */ (
    byId('draft-discard-note')
  );
  const draftAffordanceEl = /** @type {HTMLButtonElement} */ (
    byId('draft-affordance')
  );
  const draftAgentNoteEl = /** @type {HTMLElement} */ (byId('draft-agent-note'));
  const draftRejectedBannerEl = /** @type {HTMLElement} */ (
    byId('draft-rejected-banner')
  );
  // The launched-banner has no static markup in index.html — it is a
  // transient, SSE-driven note, so build it once here and hang it beside the
  // launch-outcome banner (both live in the detail footer). Created via `h`
  // (createElement/textContent) like everything else in this panel.
  const launchBannerEl = h('div', {
    id: 'draft-launched-banner',
    class: 'banner banner-ok',
    hidden: true,
  });
  launchOutcomeBannerEl.insertAdjacentElement('afterend', launchBannerEl);
  // The agent-draft banner ("agent drafted <plan> — View draft") likewise has
  // no static markup: it is rendered purely from draft-client's declarative
  // banner decision (onAgentDraftBanner below) and sits in the draft-status
  // row, taking the passive affordance's place while active (draft-client
  // suppresses the affordance whenever the banner shows).
  const agentDraftBannerEl = h('div', {
    id: 'agent-draft-banner',
    class: 'agent-draft-banner',
    hidden: true,
  });
  draftAffordanceEl.insertAdjacentElement('afterend', agentDraftBannerEl);

  // ---- root error ----

  /** @param {string} message */
  function showRootError(message) {
    rootErrorEl.textContent = message;
    rootErrorEl.hidden = false;
  }

  function clearRootError() {
    rootErrorEl.hidden = true;
    rootErrorEl.textContent = '';
  }

  // ---- sidebar (plan browser) ----

  /**
   * Re-render the sidebar from the current catalog, selection and filter.
   *
   * The tree itself lives in plan-browser.js and holds no state — this binds it
   * to the panel's elements and module-level state, so every caller here stays a
   * bare `renderPlanTree()`.
   */
  function renderPlanTree() {
    renderPlanTreeInto({
      treeEl: planTreeEl,
      emptyEl: plansEmptyEl,
      filteredEmptyEl: plansFilteredEmptyEl,
      plans,
      selectedName,
      filterText,
    });
  }

  /**
   * Apply a plan filter and re-render the sidebar.
   *
   * The ONE place the filter is set. Two surfaces drive it — the in-body search
   * box and, when the panel is embedded in Expert mode, the tile bar's
   * contributed `search` item — and they must not be able to disagree, so the
   * header path calls this rather than reimplementing the trim/re-render. The
   * in-body input is kept in sync (never overwritten with what it already
   * holds, which would move the caret) so a flip back to Simple, where the
   * bar's copy disappears, does not lose the filter the operator typed.
   *
   * @param {string} text
   */
  function setFilter(text) {
    filterText = text.trim();
    if (searchEl.value !== text) searchEl.value = text;
    renderPlanTree();
  }

  // ---- tabs ----

  /** @param {string} tab */
  function setActiveTab(tab) {
    const paramsActive = tab === 'params';
    tabParamsEl.setAttribute('aria-selected', paramsActive ? 'true' : 'false');
    tabSourceEl.setAttribute('aria-selected', paramsActive ? 'false' : 'true');
    tabParamsEl.classList.toggle('active', paramsActive);
    tabSourceEl.classList.toggle('active', !paramsActive);
    panelParamsEl.hidden = !paramsActive;
    panelSourceEl.hidden = paramsActive;
  }

  // ---- live readout ----

  function updateSummary() {
    if (!collectPlanArgs || !selectedName) {
      paramSummaryEl.hidden = true;
      return;
    }
    const text = summarizePlanArgs(selectedName, collectPlanArgs());
    paramSummaryEl.textContent = text;
    paramSummaryEl.hidden = !text;
  }

  // ---- detail ----

  /**
   * @param {PlanSummary|undefined} plan
   * @param {PlanSource} source
   */
  function renderDetail(plan, source) {
    detailEmptyEl.hidden = true;
    detailBodyEl.hidden = false;
    detailTitleEl.textContent = source.name;

    const statusText = [
      source.provenance,
      source.validated ? 'validated' : 'not validated',
      ...(source.truncated ? ['source truncated'] : []),
    ].join(' · ');
    detailStatusEl.replaceChildren(
      h('span', {
        class: `dot ${source.validated ? dotClass(source.provenance) : 'err'}`,
      }),
      h('span', { text: statusText })
    );

    detailDescEl.textContent = (plan && plan.description) || '';
    detailDescEl.hidden = !(plan && plan.description);
    sessionNoteEl.hidden = source.provenance !== 'session';
    // Text first, always: `textContent` is inert and renders the source
    // correctly on its own, so a missing or broken highlight.js costs colour
    // and nothing else. The colouring pass is layered on top.
    detailSourceEl.textContent = source.source;
    highlightSource(detailSourceEl);

    renderParamForm(plan, source);

    // A freshly-selected plan always opens on Parameters — the operator's
    // primary task — and resets both transient confirm gates + the banner.
    setActiveTab('params');
    clearQueueOutcome();
    confirmArmed = false;
    discardArmed = false;
    // A brand-new form for a newly-selected plan is not a reset divergence.
    formResetSincePatch = false;
    updateDiscardButton();
    unvalidatedNoteEl.hidden = source.validated;
    updateSummary();
    updateQueueButton();
  }

  /**
   * (Re)build the parameter form from the plan's schema and adopt its collector.
   * Shared by plan selection and the local Reset action — `renderSchemaForm`
   * dispatches no events of its own (only its `applyValues` does), so a rebuild
   * marks no pending keys and PATCHes nothing, which is exactly what makes
   * Reset local.
   *
   * @param {PlanSummary|undefined} plan
   * @param {PlanSource} source
   */
  function renderParamForm(plan, source) {
    collectPlanArgs = renderSchemaForm(paramFormEl, plan && plan.schema ? plan.schema : undefined, {
      layout: planLayout(source.name),
    });
  }

  function updateQueueButton() {
    const validated = Boolean(selectedSource && selectedSource.validated);
    queueAddBtnEl.disabled = !validated || !capability.canExecute || queueing;
    resetBtnEl.disabled = collectPlanArgs === null;
    if (queueing) {
      queueAddBtnEl.textContent = 'Adding…';
      queueAddBtnEl.classList.remove('confirm');
    } else if (confirmArmed) {
      // The label is derived from the SAME predicate that drives the behavior
      // (`queueAddReplacesDraft`), never from a separate reading of the state —
      // so the confirm can never promise one thing while the enqueue does
      // another, in any binding/reset combination.
      queueAddBtnEl.textContent = queueAddReplacesDraft()
        ? 'Confirm — replaces shared draft'
        : 'Confirm add';
      queueAddBtnEl.classList.add('confirm');
    } else {
      queueAddBtnEl.textContent = 'Add to queue';
      queueAddBtnEl.classList.remove('confirm');
    }
  }

  function updateDiscardButton() {
    draftDiscardBtnEl.textContent = discardArmed ? 'Confirm discard' : 'Discard shared draft';
    draftDiscardBtnEl.classList.toggle('confirm', discardArmed);
    setNote(draftDiscardNoteEl, discardArmed ? DISCARD_CONFIRM_NOTICE : '');
  }

  /**
   * Show `banner` in `el`, or hide and reset it on `null` — the one place a
   * `.banner` element's hidden/tone/text are set, shared by the outcome and
   * capability banners so they cannot drift apart.
   *
   * @param {HTMLElement} el
   * @param {{kind: string, message: string}|null} banner
   */
  function applyBanner(el, banner) {
    el.hidden = banner === null;
    el.className = banner === null ? 'banner' : `banner banner-${banner.kind}`;
    el.textContent = banner === null ? '' : banner.message;
  }

  /**
   * @param {'ok'|'warn'|'err'|'info'} kind
   * @param {string} message
   */
  function showQueueOutcome(kind, message) {
    applyBanner(launchOutcomeBannerEl, { kind, message });
  }

  function clearQueueOutcome() {
    applyBanner(launchOutcomeBannerEl, null);
  }

  // ---- capability (can this deployment execute plans at all?) ----

  /**
   * Re-read `GET /bridge/health` and republish the capability state.
   *
   * The two capability invariants are honored here rather than inside
   * `classifyCapability`'s callers: a throw (sidecar unreachable) is classified
   * exactly like a non-200, and `status: "ok"` is never consulted — liveness and
   * executability are independent facts.
   */
  async function refreshCapability() {
    try {
      const { status, body } = await fetchJson('/bridge/health');
      capability = classifyCapability(status, body);
    } catch {
      capability = classifyCapability(0, null);
    }
    renderCapability();
  }

  function renderCapability() {
    applyBanner(capabilityBannerEl, capabilityBanner(capability));
    updateQueueButton();
  }

  // ---- data loading ----

  /** @type {Promise<void>|null} */
  let loadPlansInFlight = null;

  /**
   * Fetch and render the plan catalog. Single-flight: the draft client's
   * unknown-plan check also calls this (as `refetchPlans`), and a boot-time
   * SSE hello can arrive concurrently with the initial boot call — without
   * coalescing, two overlapping fetches could each independently observe
   * `!selectedName` and both auto-select (a redundant, racy double
   * `selectPlan`). A caller that arrives while a fetch is already in flight
   * gets that same in-flight promise instead of starting a second one.
   *
   * @returns {Promise<void>}
   */
  function loadPlans() {
    if (loadPlansInFlight) return loadPlansInFlight;
    loadPlansInFlight = loadPlansOnce().finally(() => {
      loadPlansInFlight = null;
    });
    return loadPlansInFlight;
  }

  async function loadPlansOnce() {
    try {
      const response = await fetch(api('/plans'));
      if (!response.ok) {
        showRootError(`could not load plans (HTTP ${response.status})`);
        plans = [];
        renderPlanTree();
        detailEmptyEl.hidden = false;
        return;
      }
      const body = await response.json();
      plans = Array.isArray(body) ? body : [];
      clearRootError();
      renderPlanTree();
      if (plans.length === 0) {
        detailEmptyEl.hidden = false;
      } else if (!selectedName) {
        // Auto-select the first plan (trust-order first group) so the detail
        // pane is never a dead "select something" placeholder.
        const grouped = groupByProvenance(plans);
        void selectPlan(grouped[0].items[0].name);
      }
    } catch {
      showRootError('could not reach the bluesky-web sidecar');
      plans = [];
      renderPlanTree();
      detailEmptyEl.hidden = false;
    }
  }

  /**
   * @param {string} name
   */
  async function selectPlan(name) {
    selectedName = name;
    // Reset the transient launch gate synchronously, before the await below,
    // so a still-in-flight source fetch for a newly-selected plan can never
    // leave the Launch button/detail reflecting the PREVIOUS plan's
    // validated+armed state (the server/connector remain the authoritative
    // write gate; this is a client-side consistency fix).
    selectedSource = null;
    collectPlanArgs = null;
    confirmArmed = false;
    updateQueueButton();
    renderPlanTree();
    try {
      // Ask for the bridge's max source allowance: the default (4000 chars) is
      // sized for the approval hook's skim excerpt, but this tab exists to let
      // the operator read the WHOLE plan. The sidecar proxy forwards the query
      // param verbatim; the bridge clamps it server-side.
      const response = await fetch(
        api(`/plans/${encodeURIComponent(name)}/source?max_chars=200000`)
      );
      if (!response.ok) {
        selectedSource = null;
        confirmArmed = false;
        detailEmptyEl.hidden = true;
        detailBodyEl.hidden = false;
        detailTitleEl.textContent = name;
        detailStatusEl.replaceChildren();
        detailDescEl.hidden = true;
        sessionNoteEl.hidden = true;
        detailSourceEl.textContent = '';
        paramFormEl.replaceChildren();
        collectPlanArgs = null;
        setActiveTab('params');
        showQueueOutcome('err', `could not load plan source (HTTP ${response.status})`);
        unvalidatedNoteEl.hidden = true;
        updateSummary();
        updateQueueButton();
        return;
      }
      /** @type {PlanSource} */
      const source = await response.json();
      selectedSource = source;
      const plan = plans.find((candidate) => candidate.name === name);
      renderDetail(plan, source);
      // Draft-binding check: only ever a consequence of this explicit
      // selection (or the affordance click below) — never of a frame alone
      // (selection/binding precedence).
      await draftClient.onPlanSelected(name);
    } catch {
      selectedSource = null;
      confirmArmed = false;
      detailEmptyEl.hidden = true;
      detailBodyEl.hidden = false;
      showQueueOutcome('err', 'could not reach the bluesky-web sidecar');
      updateQueueButton();
    }
  }

  // ---- local reset (never touches the shared draft) ----

  /**
   * Re-render the parameter form from the plan's schema defaults.
   *
   * Deliberately local and silent on the wire: nothing is PATCHed, nothing is
   * deleted, and the shared draft on the bridge is left exactly as it was.
   *
   * That divergence is recorded in `formResetSincePatch`, because a reset marks
   * no pending keys: without it the next enqueue would flush nothing and pin
   * the pre-reset revision, running values the operator can no longer see.
   */
  function resetForm() {
    if (!selectedSource) return;
    const source = selectedSource;
    renderParamForm(
      plans.find((candidate) => candidate.name === source.name),
      source
    );
    formResetSincePatch = true;
    updateSummary();
    updateQueueButton();
    showQueueOutcome('info', resetNoticeMessage(draftClient.isBound()));
  }

  // ---- queue flow ----

  /**
   * Whether adding to the queue will overwrite the shared draft with this
   * form — the single predicate behind both the confirm label and the enqueue's
   * own branch, so the two cannot disagree.
   *
   * True in exactly two cases. Unbound: the form was never the shared draft, so
   * it must become it. Bound-after-Reset: Reset blanks fields without marking
   * pending keys, so a flush would express none of it.
   *
   * @returns {boolean}
   */
  function queueAddReplacesDraft() {
    return !draftClient.isBound() || formResetSincePatch;
  }

  /**
   * Make the shared draft EQUAL this form, and return the revision that write
   * minted. See `buildDraftReplaceBody` for why the `remove` list (built from
   * the draft's current keys) is what makes this a replacement rather than a
   * merge — a merge would silently enqueue the previous writer's values for
   * every field the operator had cleared.
   *
   * @param {string} planName
   * @returns {Promise<{revision: number|null}|{error: string}>}
   */
  async function replaceSharedDraftWithForm(planName) {
    const planArgs = collectPlanArgs ? collectPlanArgs() : {};
    const replaced = await fetchJson('/draft', {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(
        buildDraftReplaceBody({
          planName,
          planArgs,
          draftArgKeys: draftClient.getDraftArgKeys(),
          clientId: draftClientId,
        })
      ),
    });
    if (!replaced.ok) {
      // Left diverged on purpose: the push did not land, so the form still
      // differs from the draft and the next attempt must push again.
      return { error: describeDraftReplaceFailure(replaced.status, replaced.body) };
    }
    formResetSincePatch = false;
    const revision =
      replaced.body && typeof replaced.body.revision === 'number' ? replaced.body.revision : null;
    return { revision };
  }

  /**
   * Resolve the draft revision to enqueue.
   *
   * `POST /queue/items` takes a pinned `draft_revision` and NOTHING else — the
   * enqueued plan_name/plan_args come exclusively from the bridge's own snapshot
   * at that revision. So whatever is pinned here IS what runs, and the panel's
   * only job is to guarantee the snapshot matches the screen.
   *
   * @param {string} planName
   * @returns {Promise<{revision: number|null}|{error: string}>}
   */
  async function resolveRevisionToEnqueue(planName) {
    if (queueAddReplacesDraft()) return replaceSharedDraftWithForm(planName);
    // Bound with the form mirroring the draft: flush pending edits, then pin
    // the revision from that flush's PATCH response — falling back to the
    // last-applied frame/hello baseline when nothing was pending (the launch
    // revision gate).
    const flushResult = await draftClient.flushNow();
    return { revision: resolvePinnedRevision(flushResult, draftClient.getLastAppliedRevision()) };
  }

  async function doQueueAdd() {
    if (!selectedName || !selectedSource || !selectedSource.validated) return;
    // Local mirror of the bridge's own refusal, so a browse-only deployment
    // never even asks. The bridge still answers for itself — a capability that
    // went stale between the last refresh and this click comes back as a
    // `cannot_execute` refusal below, which republishes the record.
    if (!capability.canExecute) return;
    // Captured now (a `const`, never reassigned) rather than read again after
    // the `await`s below — `selectedName` is a module-level `let` that another
    // concurrent selection could reassign, and re-reading it would both risk
    // acting on the wrong plan and widen back to `string|null` for tsc.
    const planName = selectedName;
    queueing = true;
    updateQueueButton();
    try {
      const pinned = await resolveRevisionToEnqueue(planName);
      if ('error' in pinned) {
        showQueueOutcome('err', `could not stage the draft: ${pinned.error}`);
        return;
      }


      const { status, body } = await fetchJson('/queue/items', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(buildQueueAddBody({ revision: pinned.revision })),
      });

      const outcome = classifyQueueAddResponse(status, body);
      if (outcome.type === 'stale_draft_revision') {
        // The draft moved on since the pinned snapshot; resync so the operator
        // is looking at the current draft before retrying (the launch revision
        // gate handles the stale-revision 409 by resyncing and asking again).
        await draftClient.resync();
      } else if (outcome.type === 'cannot_execute') {
        // The refusal carries the authoritative capability record; adopt it so
        // the banner and the disabled button agree with the bridge immediately,
        // without waiting for the refresh interval.
        capability = { canExecute: false, reason: outcome.reason, detail: outcome.detail };
        renderCapability();
      }
      const banner = queueOutcomeBanner(outcome);
      showQueueOutcome(banner.kind, banner.message);
    } catch {
      showQueueOutcome('err', 'bridge unreachable');
    } finally {
      queueing = false;
      confirmArmed = false;
      updateQueueButton();
    }
  }

  // ---- plan draft (live view of the server-held shared draft) ----

  /**
   * Show a note/banner element with `text`, or hide and clear it when `text`
   * is empty — the shared shape of every draft-client DOM callback below.
   *
   * @param {HTMLElement} el
   * @param {string} text
   */
  function setNote(el, text) {
    el.hidden = !text;
    el.textContent = text;
  }

  // Per-tab id: the frame `origin`/PATCH `client_id` other subscribers use for
  // echo suppression, so a panel never re-applies its own edit as a frame.
  const draftClientId = generateClientId();

  /** @param {Record<string, unknown>} body */
  function patchDraft(body) {
    return fetchJson('/draft', {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  }

  /**
   * @returns {Promise<import('./draft-client.js').DraftGetResponse>}
   */
  async function getDraft() {
    const response = await fetch(api('/draft'));
    if (!response.ok) {
      // A non-2xx body (e.g. a 502 bridge-unreachable relay) does not parse as
      // a `{draft, revision}` snapshot — feeding it to reduceReset would set
      // `lastAppliedRevision` to `undefined` and disable the drop/gap
      // machinery. Throwing leaves draft-client's state untouched (the rejected
      // promise self-heals: the next frame/resync attempt starts clean).
      throw new Error(`GET /draft failed: HTTP ${response.status}`);
    }
    return response.json();
  }

  async function deleteDraft() {
    const response = await fetch(api(`/draft?client_id=${encodeURIComponent(draftClientId)}`), {
      method: 'DELETE',
    });
    if (!response.ok) {
      // A 502 (bridge-unreachable relay) or any other non-2xx: the discard did
      // NOT actually happen server-side. Throwing lets draft-client.js's
      // onDiscardClick keep the panel's bound state consistent with reality
      // instead of optimistically unbinding a draft that's still there.
      throw new Error(`DELETE /draft failed: HTTP ${response.status}`);
    }
  }

  const draftClient = createDraftClient({
    formEl: paramFormEl,
    api,
    clientId: draftClientId,
    getCollector: () => collectPlanArgs,
    getPlanNames: () => plans.map((plan) => plan.name),
    selectPlan,
    refetchPlans: loadPlans,
    getDraft,
    patchDraft,
    deleteDraft,
    onBoundChange(bound) {
      draftIndicatorEl.hidden = !bound;
      // Binding re-applies the whole draft over the form, so whatever a Reset
      // had blanked is gone from the screen too — the divergence it recorded
      // no longer exists.
      if (bound) formResetSincePatch = false;
      // Unbinding hides the Discard button; a confirm left armed behind it
      // would silently re-arm the next time the indicator reappears.
      if (!bound && discardArmed) {
        discardArmed = false;
        updateDiscardButton();
      }
      // The confirm label distinguishes bound from unbound (the unbound add
      // replaces the shared draft), so it has to be recomputed on the flip.
      updateQueueButton();
    },
    onAffordance(planName) {
      setNote(
        draftAffordanceEl,
        planName === null ? '' : `Draft is now on ${planName} — click to view`
      );
    },
    onAgentDraftBanner(planName) {
      // Declarative: rebuilt (or removed) from the predicate decision alone —
      // no imperative show/hide state here. Clicking View runs selectPlan on
      // the drafted plan, which binds and clears formDirty, so the predicate
      // goes false and the next recompute removes the banner itself.
      if (planName === null) {
        agentDraftBannerEl.hidden = true;
        agentDraftBannerEl.replaceChildren();
        return;
      }
      agentDraftBannerEl.replaceChildren(
        buildAgentDraftBanner(document, planName, (name) => {
          void selectPlan(name);
        })
      );
      agentDraftBannerEl.hidden = false;
    },
    onUnknownPlanBanner(planName) {
      setNote(
        draftUnknownBannerEl,
        planName === null ? '' : `draft references unavailable plan "${planName}"`
      );
    },
    onAgentEditNote(keys) {
      setNote(draftAgentNoteEl, keys.length === 0 ? '' : `agent edited: ${keys.join(', ')}`);
    },
    onLaunchBanner(banner) {
      if (banner === null) {
        launchBannerEl.hidden = true;
        launchBannerEl.replaceChildren();
        return;
      }
      launchBannerEl.replaceChildren(
        buildLaunchBanner(document, banner, (runId) => {
          onOpenRun(runId);
        })
      );
      launchBannerEl.hidden = false;
    },
    onPatchRejected(detail) {
      const message =
        detail && typeof detail === 'object' && detail.field
          ? `${String(detail.field)}: ${String(detail.error)}`
          : String(detail || 'the bridge rejected that value');
      if (draftRejectedTimer !== null) clearTimeout(draftRejectedTimer);
      setNote(draftRejectedBannerEl, message);
      draftRejectedTimer = setTimeout(() => {
        setNote(draftRejectedBannerEl, '');
        draftRejectedTimer = null;
      }, DRAFT_REJECTED_NOTE_TIMEOUT_MS);
    },
  });

  // ---- event wiring (delegation, reading data-* attributes) ----

  planTreeEl.addEventListener('click', (event) => {
    const target = /** @type {HTMLElement} */ (event.target);
    const button = target.closest('button[data-plan-name]');
    if (!(button instanceof HTMLElement)) return;
    const name = button.dataset.planName;
    if (!name) return;
    void selectPlan(name);
  });

  searchEl.addEventListener('input', () => setFilter(searchEl.value));

  tabParamsEl.addEventListener('click', () => setActiveTab('params'));
  tabSourceEl.addEventListener('click', () => setActiveTab('source'));

  // Live readout: native input/change cover typed edits; the bubbling
  // form-change CustomEvent (from schema-form.js) covers structural edits
  // (chip or table-row added/removed).
  paramFormEl.addEventListener('input', updateSummary);
  paramFormEl.addEventListener('change', updateSummary);
  paramFormEl.addEventListener('form-change', updateSummary);

  queueAddBtnEl.addEventListener('click', () => {
    if (queueAddBtnEl.disabled) return;
    if (!confirmArmed) {
      confirmArmed = true;
      updateQueueButton();
      return;
    }
    void doQueueAdd();
  });

  resetBtnEl.addEventListener('click', () => {
    if (resetBtnEl.disabled) return;
    resetForm();
  });

  paramFormEl.addEventListener('submit', (event) => {
    event.preventDefault();
  });

  // Two-step, like Add-to-queue: the first click only arms the confirm and
  // spells out that this deletes the SHARED draft (the agent's too), because
  // "Discard" next to a form reads like "clear my form" and the mistake is not
  // undoable.
  draftDiscardBtnEl.addEventListener('click', () => {
    if (!discardArmed) {
      discardArmed = true;
      updateDiscardButton();
      return;
    }
    discardArmed = false;
    updateDiscardButton();
    void draftClient.onDiscardClick();
  });

  draftAffordanceEl.addEventListener('click', () => {
    void draftClient.onAffordanceClick();
  });

  // ---- boot ----

  // Browser/detail split. Plan names are long and deeply namespaced, so the
  // browser needs to be widenable at the cost of the parameter form — the same
  // affordance (and the same shared implementation) as the OKF panel's
  // sidebar/reader split and the artifact gallery's browse view. The chosen
  // width is persisted per origin; the bounds match OKF's.
  initSplitter({
    handle: byId('browser-splitter'),
    pane: byId('plan-sidebar'),
    storageKey: 'osprey-plan-sidebar-width',
    min: 180,
    max: 560,
    collapsedSize: 0,
  }).restoreSize();

  // Publish the fail-closed placeholder before any network call, so the footer
  // explains the disabled button from the first paint rather than after it.
  renderCapability();
  updateDiscardButton();

  void loadPlans();
  void refreshCapability();
  setInterval(() => {
    void refreshCapability();
  }, CAPABILITY_REFRESH_MS);

  return {
    setFilter,
    /**
     * Live Expert<->Simple flip broadcast by the hub, fanned out by the shell.
     * mode-boot.js set the initial data-ui-mode pre-paint; this is the runtime
     * flip. Every layout delta is CSS (Simple hides the Source tab, draft
     * chrome, and trust internals, and promotes Add-to-queue); the one
     * behavioral concern is that Simple hides the Source tab — force the
     * Parameters view so a flip made while Source was active doesn't leave the
     * operator on a now-hidden pane.
     *
     * @param {'expert'|'simple'} mode
     */
    onModeChange(mode) {
      if (mode === 'simple') setActiveTab('params');
    },
  };
}
