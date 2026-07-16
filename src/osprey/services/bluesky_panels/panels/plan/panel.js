// @ts-check
/**
 * Plan panel — a two-pane operator console for the registered scan plans.
 *
 * LEFT (sidebar): a dense, file-browser-style selector. Plans are grouped
 * under collapsible provenance folders and filterable; each row is name-first
 * with trust/validation compressed into a single status dot (full detail in
 * the tooltip and the detail header). The first plan is auto-selected on load
 * so the panel never opens onto an empty pane.
 *
 * RIGHT (detail): the selected plan under a two-tab strip —
 *   - Parameters: a 2-D GUI generated from the plan's JSON Schema
 *     (schema-form.js): chip editors for device lists, an editable table for
 *     grid axes, typed inputs for scalars — arranged by a per-plan layout
 *     (PLAN_LAYOUTS) with a live readout (PLAN_SUMMARIES) that recomputes on
 *     every edit, e.g. "2 correctors × 7 points = 14 sweep points".
 *   - Source: the plan's source code.
 * plus the deterministic Execute action in the footer.
 *
 * Plans absent from the two registries still render fully — the schema-driven
 * form auto-flows their fields — so facility/session plans need no panel-side
 * code to be usable.
 *
 * Reached through the web-terminal reverse proxy at `/panel/plan/…`, so every
 * fetch is derived from this panel's own URL prefix and issued prefix-relative
 * — never an absolute `/plans` path (the proxy does not rewrite those).
 *
 * Execution is deterministic: no agent/LLM in this path. The sidecar's sole
 * write route (`POST /runs/execute`) composes create-intent + promote
 * server-side; the browser never sees or sends a promote token. `Execute`
 * requires a two-step in-panel confirm and is enabled only when the selected
 * plan's `validated` flag (from the source response) is `true`.
 *
 * The whole UI is built with createElement/textContent (see schema-form.js
 * and the local `h` helper) — no innerHTML sink anywhere — so plan-authored
 * strings (names, descriptions, source, enum values) are never interpreted as
 * markup.
 *
 * @module panel
 */

import { renderSchemaForm } from './schema-form.js';

/**
 * @typedef {object} PlanSummary
 * @property {string} name
 * @property {string} [description]
 * @property {import('./schema-form.js').JsonSchemaNode} [schema]
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

/**
 * Tiny hyperscript helper (mirrors schema-form's) so this module builds DOM
 * without any innerHTML. Strings become text nodes.
 *
 * @param {string} tag
 * @param {Record<string, unknown>} [props]
 * @param {...(Node|string|number|null|undefined)} children
 * @returns {HTMLElement}
 */
function h(tag, props, ...children) {
  const node = document.createElement(tag);
  if (props) {
    for (const [key, value] of Object.entries(props)) {
      if (value === null || value === undefined || value === false) continue;
      if (key === 'class') node.className = String(value);
      else if (key === 'text') node.textContent = String(value);
      else if (key in node) /** @type {any} */ (node)[key] = value;
      else node.setAttribute(key, String(value));
    }
  }
  for (const child of children) {
    if (child === null || child === undefined) continue;
    node.appendChild(typeof child === 'object' ? child : document.createTextNode(String(child)));
  }
  return node;
}

// Provenance folders, in trust order. Any provenance the bridge reports that
// isn't listed here falls into a trailing "other" group so nothing is dropped.
const PROVENANCE_ORDER = ['shipped', 'preset', 'facility', 'session', 'unreviewed'];

/**
 * Per-plan 2-D field placement: rows of side-by-side field names, passed to
 * renderSchemaForm as its layout option. Names not in the plan's schema are
 * ignored; schema fields missing here are auto-flowed after these rows —
 * so this is a presentation hint, never a gate.
 *
 * @type {Record<string, string[][]>}
 */
const PLAN_LAYOUTS = {
  orm: [
    ['correctors', 'detectors'],
    ['span_a', 'num'],
    ['sweep'],
  ],
  grid_scan: [['axes'], ['detectors', 'snake_axes']],
};

/**
 * Per-plan live readout, computed from the currently-collected plan_args on
 * every edit. Returns '' when there is nothing meaningful to show yet.
 * Plans without an entry fall back to a generic "N parameters set".
 *
 * @type {Record<string, (args: Record<string, any>) => string>}
 */
const PLAN_SUMMARIES = {
  orm(args) {
    const c = Array.isArray(args.correctors) ? args.correctors.length : 0;
    const d = Array.isArray(args.detectors) ? args.detectors.length : 0;
    const n = typeof args.num === 'number' ? args.num : 0;
    const span = typeof args.span_a === 'number' ? args.span_a : null;
    /** @type {string[]} */
    const parts = [];
    const sweep = typeof args.sweep === 'string' ? args.sweep : null;
    if (c) parts.push(`${c} corrector${c === 1 ? '' : 's'}`);
    if (d) parts.push(`${d} BPM${d === 1 ? '' : 's'}`);
    if (span !== null) {
      // Monodirectional sweeps [0, +span]; bidirectional the symmetric ±span.
      parts.push(sweep === 'monodirectional' ? `0…${span} A` : `±${span} A`);
    }
    if (c && n) parts.push(`${c} × ${n} = ${c * n} sweep points`);
    else if (n) parts.push(`${n} points`);
    return parts.join(' · ');
  },
  grid_scan(args) {
    const axes = Array.isArray(args.axes) ? args.axes : [];
    const d = Array.isArray(args.detectors) ? args.detectors.length : 0;
    /** @type {string[]} */
    const parts = [];
    if (axes.length) {
      const nums = axes.map((axis) =>
        axis && typeof axis.num_points === 'number' ? axis.num_points : 0
      );
      parts.push(`${axes.length} ax${axes.length === 1 ? 'is' : 'es'}`);
      if (nums.every((v) => v > 0)) {
        const total = nums.reduce((product, v) => product * v, 1);
        parts.push(`${nums.join(' × ')} = ${total} grid points`);
      }
    }
    if (d) parts.push(`${d} detector${d === 1 ? '' : 's'}`);
    return parts.join(' · ');
  },
};

/** @type {PlanSummary[]} */
let plans = [];
/** @type {string|null} */
let selectedName = null;
/** @type {PlanSource|null} */
let selectedSource = null;
/** @type {(() => Record<string, unknown>)|null} */
let collectPlanArgs = null;
let filterText = '';
let confirmArmed = false;
let executing = false;

// ---- element lookups ----

const rootErrorEl = /** @type {HTMLElement} */ (document.getElementById('root-error'));
const planTreeEl = /** @type {HTMLElement} */ (document.getElementById('plan-tree'));
const plansEmptyEl = /** @type {HTMLElement} */ (document.getElementById('plans-empty'));
const plansFilteredEmptyEl = /** @type {HTMLElement} */ (
  document.getElementById('plans-filtered-empty')
);
const searchEl = /** @type {HTMLInputElement} */ (document.getElementById('plan-search'));
const detailEmptyEl = /** @type {HTMLElement} */ (document.getElementById('detail-empty'));
const detailBodyEl = /** @type {HTMLElement} */ (document.getElementById('detail-body'));
const detailTitleEl = /** @type {HTMLElement} */ (document.getElementById('detail-title'));
const detailStatusEl = /** @type {HTMLElement} */ (document.getElementById('detail-status'));
const detailDescEl = /** @type {HTMLElement} */ (document.getElementById('detail-desc'));
const sessionNoteEl = /** @type {HTMLElement} */ (document.getElementById('session-note'));
const detailSourceEl = /** @type {HTMLElement} */ (document.getElementById('detail-source'));
const paramFormEl = /** @type {HTMLFormElement} */ (document.getElementById('param-form'));
const paramSummaryEl = /** @type {HTMLElement} */ (document.getElementById('param-summary'));
const execBannerEl = /** @type {HTMLElement} */ (document.getElementById('exec-banner'));
const unvalidatedNoteEl = /** @type {HTMLElement} */ (document.getElementById('unvalidated-note'));
const executeBtnEl = /** @type {HTMLButtonElement} */ (document.getElementById('execute-btn'));
const tabParamsEl = /** @type {HTMLButtonElement} */ (document.getElementById('tab-params'));
const tabSourceEl = /** @type {HTMLButtonElement} */ (document.getElementById('tab-source'));
const panelParamsEl = /** @type {HTMLElement} */ (document.getElementById('panel-params'));
const panelSourceEl = /** @type {HTMLElement} */ (document.getElementById('panel-source'));

// ---- status presentation ----

/**
 * The status-dot class for a provenance tier. Trust tiers with unconditional
 * bridge-side validation read as OK; session is caution; unreviewed (or any
 * unknown tier) is danger.
 *
 * @param {string} provenance
 * @returns {string}
 */
function dotClass(provenance) {
  if (provenance === 'shipped' || provenance === 'preset' || provenance === 'facility') {
    return 'ok';
  }
  if (provenance === 'session') return 'warn';
  return 'err';
}

/**
 * Tooltip for a sidebar row: description plus the trust tier, so the row
 * itself stays name-only.
 *
 * @param {PlanSummary} plan
 * @returns {string}
 */
function rowTooltip(plan) {
  const desc = plan.description ? `${plan.description} — ` : '';
  return `${desc}${plan.provenance}`;
}

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
 * Group the (filtered) plans by provenance, preserving trust order and
 * appending any unknown tiers under "other".
 *
 * @param {PlanSummary[]} list
 * @returns {Array<{provenance: string, items: PlanSummary[]}>}
 */
function groupByProvenance(list) {
  /** @type {Map<string, PlanSummary[]>} */
  const groups = new Map();
  for (const plan of list) {
    const key = PROVENANCE_ORDER.includes(plan.provenance) ? plan.provenance : 'other';
    const bucket = groups.get(key);
    if (bucket) bucket.push(plan);
    else groups.set(key, [plan]);
  }
  /** @type {Array<{provenance: string, items: PlanSummary[]}>} */
  const ordered = [];
  for (const key of [...PROVENANCE_ORDER, 'other']) {
    const items = groups.get(key);
    if (items && items.length) ordered.push({ provenance: key, items });
  }
  return ordered;
}

/**
 * @param {PlanSummary} plan
 * @returns {boolean}
 */
function matchesFilter(plan) {
  if (!filterText) return true;
  const needle = filterText.toLowerCase();
  return (
    plan.name.toLowerCase().includes(needle) ||
    (plan.description || '').toLowerCase().includes(needle)
  );
}

function renderPlanTree() {
  planTreeEl.replaceChildren();

  if (plans.length === 0) {
    plansEmptyEl.hidden = false;
    plansFilteredEmptyEl.hidden = true;
    return;
  }
  plansEmptyEl.hidden = true;

  const visible = plans.filter(matchesFilter);
  if (visible.length === 0) {
    plansFilteredEmptyEl.hidden = false;
    return;
  }
  plansFilteredEmptyEl.hidden = true;

  for (const group of groupByProvenance(visible)) {
    const summary = h(
      'summary',
      { class: 'folder-summary' },
      h('span', { class: 'folder-name', text: group.provenance }),
      h('span', { class: 'folder-count', text: String(group.items.length) })
    );
    const folder = h('details', { class: 'folder', open: true }, summary);
    const items = h('div', { class: 'folder-items', role: 'group' });
    for (const plan of group.items) {
      const isSelected = plan.name === selectedName;
      items.appendChild(
        h(
          'button',
          {
            type: 'button',
            class: `plan-row${isSelected ? ' selected' : ''}`,
            role: 'treeitem',
            'aria-selected': isSelected ? 'true' : 'false',
            'data-plan-name': plan.name,
            title: rowTooltip(plan),
          },
          h('span', { class: `dot ${dotClass(plan.provenance)}` }),
          h('span', { class: 'plan-row-name', text: plan.name })
        )
      );
    }
    folder.appendChild(items);
    planTreeEl.appendChild(folder);
  }
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
  const args = collectPlanArgs();
  const custom = PLAN_SUMMARIES[selectedName];
  let text = custom ? custom(args) : '';
  if (!text) {
    const count = Object.keys(args).length;
    text = count > 0 ? `${count} parameter${count === 1 ? '' : 's'} set` : '';
  }
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
  detailSourceEl.textContent = source.source;

  collectPlanArgs = renderSchemaForm(paramFormEl, plan && plan.schema ? plan.schema : undefined, {
    layout: PLAN_LAYOUTS[source.name],
  });

  // A freshly-selected plan always opens on Parameters — the operator's
  // primary task — and resets the transient execute gate + banner.
  setActiveTab('params');
  execBannerEl.hidden = true;
  execBannerEl.textContent = '';
  execBannerEl.className = 'banner';
  confirmArmed = false;
  unvalidatedNoteEl.hidden = source.validated;
  updateSummary();
  updateExecuteButton();
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
    showRootError('could not reach the bluesky panels sidecar');
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
  // Reset the transient execute gate synchronously, before the await below,
  // so a still-in-flight source fetch for a newly-selected plan can never
  // leave the Execute button/detail reflecting the PREVIOUS plan's
  // validated+armed state (the server/connector remain the authoritative
  // write gate; this is a client-side consistency fix).
  selectedSource = null;
  collectPlanArgs = null;
  confirmArmed = false;
  updateExecuteButton();
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
      showExecBanner('err', `could not load plan source (HTTP ${response.status})`);
      unvalidatedNoteEl.hidden = true;
      updateSummary();
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
    detailEmptyEl.hidden = true;
    detailBodyEl.hidden = false;
    showExecBanner('err', 'could not reach the bluesky panels sidecar');
    updateExecuteButton();
  }
}

// ---- execute flow ----

async function doExecute() {
  if (!selectedName || !selectedSource || !selectedSource.validated) return;
  executing = true;
  updateExecuteButton();
  try {
    const planArgs = collectPlanArgs ? collectPlanArgs() : {};
    const response = await fetch(api('/runs/execute'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ plan_name: selectedName, plan_args: planArgs }),
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

planTreeEl.addEventListener('click', (event) => {
  const target = /** @type {HTMLElement} */ (event.target);
  const button = target.closest('button[data-plan-name]');
  if (!(button instanceof HTMLElement)) return;
  const name = button.dataset.planName;
  if (!name) return;
  void selectPlan(name);
});

searchEl.addEventListener('input', () => {
  filterText = searchEl.value.trim();
  renderPlanTree();
});

tabParamsEl.addEventListener('click', () => setActiveTab('params'));
tabSourceEl.addEventListener('click', () => setActiveTab('source'));

// Live readout: native input/change cover typed edits; the bubbling
// form-change CustomEvent (from schema-form.js) covers structural edits
// (chip or table-row added/removed).
paramFormEl.addEventListener('input', updateSummary);
paramFormEl.addEventListener('change', updateSummary);
paramFormEl.addEventListener('form-change', updateSummary);

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
