// @ts-check
/**
 * Scan Plan panel — browse registered scan plans, inspect source/provenance,
 * set parameters from the plan's JSON schema, and execute a validated plan.
 *
 * Reached through the web-terminal reverse proxy at `/panel/scan-plan/…`, so
 * every fetch is derived from this panel's own URL prefix and issued
 * prefix-relative — never an absolute `/plans` path (the proxy does not
 * rewrite those).
 *
 * Execution is deterministic: no agent/LLM in this path. The sidecar's sole
 * write route (`POST /runs/execute`) composes create-intent + promote
 * server-side; the browser never sees or sends a promote token. `Execute`
 * requires a two-step in-panel confirm and is enabled only when the
 * selected plan's `validated` flag (from the source response) is `true`.
 *
 * @module panel
 */

import { escapeHtml } from '/design-system/js/dom.js';

/**
 * @typedef {object} PlanSummary
 * @property {string} name
 * @property {string} [description]
 * @property {{properties?: Record<string, {type?: string, title?: string, default?: unknown}>}} [schema]
 * @property {unknown} [metadata]
 * @property {string} provenance
 */

/**
 * @typedef {object} PlanSource
 * @property {string} name
 * @property {string} provenance
 * @property {boolean} validated
 * @property {boolean} truncated
 * @property {string} source
 */

// The panel is served through the web-terminal proxy at /panel/<id>/…; the
// proxy does not rewrite plain "/plans"-style absolute paths, so every fetch
// must be prefixed with this panel's own mount prefix. Falls back to "" when
// opened directly (e.g. the visual-regression test, which serves this panel
// with no bridge behind it).
const PREFIX = (location.pathname.match(/^\/panel\/[^/]+/) || [''])[0];

/**
 * @param {string} path
 * @returns {string}
 */
function api(path) {
  return PREFIX + path;
}

/** @type {PlanSummary[]} */
let plans = [];
/** @type {string|null} */
let selectedName = null;
/** @type {PlanSource|null} */
let selectedSource = null;
let confirmArmed = false;
let executing = false;

// ---- element lookups ----

const rootErrorEl = /** @type {HTMLElement} */ (document.getElementById('root-error'));
const planListEl = /** @type {HTMLUListElement} */ (document.getElementById('plan-list'));
const plansEmptyEl = /** @type {HTMLElement} */ (document.getElementById('plans-empty'));
const detailCardEl = /** @type {HTMLElement} */ (document.getElementById('detail-card'));
const detailTitleEl = /** @type {HTMLElement} */ (document.getElementById('detail-title'));
const detailBadgesEl = /** @type {HTMLElement} */ (document.getElementById('detail-badges'));
const sessionNoteEl = /** @type {HTMLElement} */ (document.getElementById('session-note'));
const detailSourceEl = /** @type {HTMLElement} */ (document.getElementById('detail-source'));
const paramFormEl = /** @type {HTMLFormElement} */ (document.getElementById('param-form'));
const execBannerEl = /** @type {HTMLElement} */ (document.getElementById('exec-banner'));
const unvalidatedNoteEl = /** @type {HTMLElement} */ (document.getElementById('unvalidated-note'));
const executeBtnEl = /** @type {HTMLButtonElement} */ (document.getElementById('execute-btn'));

// ---- rendering helpers ----

/**
 * @param {string} provenance
 * @returns {{cls: string, label: string}}
 */
function provenanceBadge(provenance) {
  const label = `provenance: ${provenance}`;
  if (provenance === 'shipped' || provenance === 'preset' || provenance === 'facility') {
    return { cls: 'ok', label };
  }
  if (provenance === 'unreviewed') {
    return { cls: 'err', label };
  }
  // 'session' or any unrecognized tier.
  return { cls: 'warn', label };
}

/**
 * A best-effort "validated" badge for a list item. `GET /plans` does not
 * carry a `validated` field — only `GET /plans/{name}/source` does (see
 * `selectPlan`). For `shipped`/`preset`/`facility` provenance the bridge
 * reports `validated=True` unconditionally (those tiers carry no
 * validation-record gate), so that much is knowable up front; for
 * `session`/`unreviewed` the true state is only known once the plan is
 * selected and its source fetched.
 *
 * @param {string} provenance
 * @returns {{cls: string, label: string}}
 */
function listValidatedBadge(provenance) {
  if (provenance === 'shipped' || provenance === 'preset' || provenance === 'facility') {
    return { cls: 'ok', label: 'validated' };
  }
  return { cls: 'info', label: 'select to verify' };
}

/**
 * @param {string} message
 */
function showRootError(message) {
  rootErrorEl.textContent = message;
  rootErrorEl.hidden = false;
}

function clearRootError() {
  rootErrorEl.hidden = true;
  rootErrorEl.textContent = '';
}

function renderPlanList() {
  if (plans.length === 0) {
    planListEl.innerHTML = '';
    plansEmptyEl.hidden = false;
    return;
  }
  plansEmptyEl.hidden = true;

  planListEl.innerHTML = plans
    .map((plan) => {
      const prov = provenanceBadge(plan.provenance);
      const validated = listValidatedBadge(plan.provenance);
      const isSelected = plan.name === selectedName;
      const description = plan.description || '(no description)';
      return `
        <li>
          <button type="button" class="plan-item${isSelected ? ' selected' : ''}"
                  data-plan-name="${escapeHtml(plan.name)}">
            <div class="plan-item-name">${escapeHtml(plan.name)}</div>
            <div class="plan-item-desc">${escapeHtml(description)}</div>
            <div class="badges">
              <span class="badge ${prov.cls}">${escapeHtml(prov.label)}</span>
              <span class="badge ${validated.cls}">${escapeHtml(validated.label)}</span>
            </div>
          </button>
        </li>
      `;
    })
    .join('');
}

/**
 * @param {PlanSummary|undefined} plan
 * @param {PlanSource} source
 */
function renderDetail(plan, source) {
  detailCardEl.hidden = false;
  detailTitleEl.textContent = source.name;

  const prov = provenanceBadge(source.provenance);
  const validatedBadge = source.validated
    ? { cls: 'ok', label: 'validated: true' }
    : { cls: 'err', label: 'validated: false' };
  detailBadgesEl.innerHTML = `
    <span class="badge ${prov.cls}">${escapeHtml(prov.label)}</span>
    <span class="badge ${validatedBadge.cls}">${escapeHtml(validatedBadge.label)}</span>
    ${source.truncated ? '<span class="badge warn">source truncated</span>' : ''}
  `;

  sessionNoteEl.hidden = source.provenance !== 'session';

  detailSourceEl.textContent = source.source;

  renderParamForm(plan && plan.schema ? plan.schema : undefined);

  execBannerEl.hidden = true;
  execBannerEl.textContent = '';
  execBannerEl.className = 'banner';
  confirmArmed = false;
  unvalidatedNoteEl.hidden = source.validated;
  updateExecuteButton();
}

/**
 * @param {{properties?: Record<string, {type?: string, title?: string, default?: unknown}>}|undefined} schema
 */
function renderParamForm(schema) {
  const properties = (schema && schema.properties) || {};
  const names = Object.keys(properties);
  if (names.length === 0) {
    paramFormEl.innerHTML = '<p class="param-empty">This plan takes no parameters.</p>';
    return;
  }

  paramFormEl.innerHTML = names
    .map((name) => {
      const prop = properties[name] || {};
      const inputType = prop.type === 'integer' || prop.type === 'number' ? 'number' : 'text';
      const label = prop.title || name;
      const defaultValue = prop.default === undefined || prop.default === null ? '' : String(prop.default);
      const step = prop.type === 'integer' ? '1' : 'any';
      return `
        <div class="param-row">
          <label for="param-${escapeHtml(name)}">${escapeHtml(label)}</label>
          <input id="param-${escapeHtml(name)}" name="${escapeHtml(name)}"
                 type="${inputType}" ${inputType === 'number' ? `step="${step}"` : ''}
                 data-param-name="${escapeHtml(name)}" data-param-type="${escapeHtml(prop.type || 'string')}"
                 value="${escapeHtml(defaultValue)}">
        </div>
      `;
    })
    .join('');
}

/**
 * Collect the current parameter form into a `plan_args` object, converting
 * number/integer fields and omitting empty fields so plan-side defaults
 * apply for anything the operator left blank.
 *
 * @returns {Record<string, unknown>}
 */
function collectPlanArgs() {
  /** @type {Record<string, unknown>} */
  const args = {};
  const inputs = /** @type {NodeListOf<HTMLInputElement>} */ (
    paramFormEl.querySelectorAll('input[data-param-name]')
  );
  for (const input of inputs) {
    const name = input.dataset.paramName;
    const type = input.dataset.paramType;
    if (!name) continue;
    const raw = input.value;
    if (raw === '') continue;
    if (type === 'integer') {
      // Reject fractional/garbage input rather than silently truncating it
      // (parseInt('3.7', 10) === 3) on a path that ends in a write.
      const value = Number(raw);
      if (Number.isInteger(value)) args[name] = value;
    } else if (type === 'number') {
      const value = parseFloat(raw);
      if (!Number.isNaN(value)) args[name] = value;
    } else {
      args[name] = raw;
    }
  }
  return args;
}

function updateExecuteButton() {
  const validated = Boolean(selectedSource && selectedSource.validated);
  executeBtnEl.disabled = !validated || executing;
  if (executing) {
    executeBtnEl.textContent = 'Executing…';
    executeBtnEl.classList.remove('confirm');
  } else if (confirmArmed) {
    executeBtnEl.textContent = 'Confirm execute';
    executeBtnEl.classList.add('confirm');
  } else {
    executeBtnEl.textContent = 'Execute plan';
    executeBtnEl.classList.remove('confirm');
  }
}

/**
 * @param {'ok'|'warn'|'err'|'info'} kind
 * @param {string} message
 */
function showExecBanner(kind, message) {
  execBannerEl.hidden = false;
  execBannerEl.className = `banner banner-${kind}`;
  execBannerEl.textContent = message;
}

// ---- data loading ----

async function loadPlans() {
  try {
    const response = await fetch(api('/plans'));
    if (!response.ok) {
      showRootError(`could not load plans (HTTP ${response.status})`);
      plans = [];
      renderPlanList();
      return;
    }
    const body = await response.json();
    plans = Array.isArray(body) ? body : [];
    clearRootError();
    renderPlanList();
  } catch {
    showRootError('could not reach the scan panels sidecar');
    plans = [];
    renderPlanList();
  }
}

/**
 * @param {string} name
 */
async function selectPlan(name) {
  selectedName = name;
  // Reset the transient execute gate synchronously, before the await below,
  // so a still-in-flight source fetch for a newly-selected plan can never
  // leave the Execute button/detail reflecting the PREVIOUS plan's
  // validated+armed state (the server/connector remain the authoritative
  // write gate; this is a client-side consistency fix).
  selectedSource = null;
  confirmArmed = false;
  updateExecuteButton();
  renderPlanList();
  try {
    const response = await fetch(api(`/plans/${encodeURIComponent(name)}/source`));
    if (!response.ok) {
      selectedSource = null;
      confirmArmed = false;
      detailCardEl.hidden = false;
      detailTitleEl.textContent = name;
      detailBadgesEl.innerHTML = '';
      sessionNoteEl.hidden = true;
      detailSourceEl.textContent = '';
      paramFormEl.innerHTML = '';
      showExecBanner('err', `could not load plan source (HTTP ${response.status})`);
      unvalidatedNoteEl.hidden = true;
      updateExecuteButton();
      return;
    }
    /** @type {PlanSource} */
    const source = await response.json();
    selectedSource = source;
    const plan = plans.find((candidate) => candidate.name === name);
    renderDetail(plan, source);
  } catch {
    selectedSource = null;
    confirmArmed = false;
    detailCardEl.hidden = false;
    showExecBanner('err', 'could not reach the scan panels sidecar');
    updateExecuteButton();
  }
}

// ---- execute flow ----

async function doExecute() {
  if (!selectedName || !selectedSource || !selectedSource.validated) return;
  executing = true;
  updateExecuteButton();
  try {
    const response = await fetch(api('/runs/execute'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ plan_name: selectedName, plan_args: collectPlanArgs() }),
    });
    /** @type {any} */
    let body = null;
    try {
      body = await response.json();
    } catch {
      body = null;
    }

    if (response.status === 200 && body && body.status === 'writes_not_armed') {
      showExecBanner('info', 'writes not armed on this deployment');
    } else if (response.status === 200 && body && body.run_id) {
      showExecBanner('ok', `run started: ${String(body.run_id)}`);
    } else if (response.status === 409) {
      const detail = (body && body.detail) || 'the bridge reported a conflict';
      showExecBanner('err', `conflict: ${String(detail)}`);
    } else if (response.status === 502) {
      showExecBanner('err', 'bridge unreachable');
    } else {
      const detail = (body && body.detail) || `HTTP ${response.status}`;
      showExecBanner('err', `execute failed: ${String(detail)}`);
    }
  } catch {
    showExecBanner('err', 'bridge unreachable');
  } finally {
    executing = false;
    confirmArmed = false;
    updateExecuteButton();
  }
}

// ---- event wiring (delegation, reading data-* attributes) ----

planListEl.addEventListener('click', (event) => {
  const target = /** @type {HTMLElement} */ (event.target);
  const button = target.closest('button[data-plan-name]');
  if (!(button instanceof HTMLElement)) return;
  const name = button.dataset.planName;
  if (!name) return;
  void selectPlan(name);
});

executeBtnEl.addEventListener('click', () => {
  if (executeBtnEl.disabled) return;
  if (!confirmArmed) {
    confirmArmed = true;
    updateExecuteButton();
    return;
  }
  void doExecute();
});

paramFormEl.addEventListener('submit', (event) => {
  event.preventDefault();
});

// ---- boot ----

void loadPlans();
