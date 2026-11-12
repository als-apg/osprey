// @ts-check
/**
 * OSPREY Channel Finder — Alarms
 *
 * The view of `#/alarms` (Panel view contract): what is in alarm now on the
 * session's control target, read from the alarm route alone.
 *
 *   toolbar   [tree toggle] [Curated | All PVs | Alarms | Feedback] …… [N active]
 *   sidebar   the alarm tree, when any row carries a path
 *   banner    the feed is not updating (stale)
 *   band      MAJOR · INVALID · MINOR [· UNDEFINED] | Unacknowledged · Latched … [refresh] [feed]
 *   body      the active rows, worst first      |  the inspector of the selected row
 *
 * The rows are drawn in the served order; a tile or a tree node only removes
 * rows. The list polls every 5 s while mounted and the document is visible,
 * and emits `alarmCounts` with each answer so the tab's pill follows it.
 * Current values are read when the rows shown change and on Refresh; the
 * inspector reads the selected PV's facts. Nothing here writes: no
 * acknowledge, no configuration change.
 */

import { panelApiPrefix } from '/design-system/js/dom.js';
import { openInfoPopover } from '/design-system/js/info-popover.js';
import { icon } from '/design-system/js/icons.js';
import { renderState } from '/design-system/js/panel-state.js';
import { showToast } from '/design-system/js/toast.js';
import { writeTitle } from '/design-system/js/tooltip.js';
import {
  NO_TREE,
  bandHtml,
  bannerHtml,
  buildTree,
  countHtml,
  feedRows,
  filterRows,
  inspectorHtml,
  listHtml,
  skeletonHtml,
  treeHtml,
} from './alarms-render.js';
import {
  createPlotCheck,
  fetchActiveAlarms,
  fetchPvFacts,
  handoffAllowed,
  plotUrl,
  readValues,
} from './directory-api.js';
import { TOOLBAR_IDS, attachSidebar, setResultCount } from './panel-toolbar.js';
import { NO_TARGET } from './pv-page-sections.js';
import { formatRoute } from './route.js';
import { state } from './state.js';
import { messageOf } from './utils.js';

/** @typedef {import('./directory-api.js').ActiveAlarms} ActiveAlarms */
/** @typedef {import('./directory-api.js').AlarmRow} AlarmRow */
/** @typedef {import('./directory-api.js').ValueRow} ValueRow */
/** @typedef {import('./directory-api.js').PvFacts} PvFacts */
/** @typedef {import('./alarms-render.js').TileFilter} TileFilter */
/** @typedef {import('/design-system/js/panel-sidebar.js').PanelSidebarApi} PanelSidebarApi */

/** How often the list is read while the view is mounted and visible. */
export const POLL_MS = 5000;
/** The most names the values route reads at once. */
const VALUES_MAX = 50;
const SIDEBAR_KEY = 'cf-alarms-tree';

/**
 * The mounted view's state; null while nothing is mounted.
 * @typedef {{
 *   root: HTMLElement,
 *   answer: ActiveAlarms|null,
 *   severity: TileFilter|null,
 *   node: string|null,
 *   selected: string|null,
 *   values: Map<string, ValueRow>|null,
 *   valuesKey: string,
 *   facts: Map<string, PvFacts|string>,
 *   pollSeq: number,
 *   valuesSeq: number,
 *   canPlot: boolean,
 *   sidebar: PanelSidebarApi|null,
 *   timer: number|undefined,
 *   controller: AbortController,
 * }} View
 */

/** @type {View|null} */
let view = null;

/**
 * The route carries no params: tile and tree filters are view state.
 * @param {HTMLElement} container
 */
export function mountAlarms(container) {
  unmountAlarms();
  container.innerHTML = skeletonHtml();
  const root = /** @type {HTMLElement} */ (container.querySelector('.alarms'));
  const controller = new AbortController();
  const { signal } = controller;
  view = {
    root,
    answer: null,
    severity: null,
    node: null,
    selected: null,
    values: null,
    valuesKey: '',
    facts: new Map(),
    pollSeq: 0,
    valuesSeq: 0,
    canPlot: false,
    sidebar: null,
    timer: undefined,
    controller,
  };
  root.addEventListener('click', onClick, { signal });
  document.addEventListener(
    'visibilitychange',
    () => {
      if (!document.hidden) void poll();
    },
    { signal }
  );
  // The toolbar resets the toggle's reason when the pipeline changes.
  state.on('pipelineChanged', noTreeTitle);
  noTreeTitle();

  const mounted = view;
  void createPlotCheck()().then((ok) => {
    if (view !== mounted) return;
    mounted.canPlot = ok;
    renderInspector();
  });

  renderState(part('.alarms-state'), { icon: 'bell', message: 'Reading the alarm feed' });
  view.timer = window.setInterval(() => {
    if (!document.hidden) void poll();
  }, POLL_MS);
  void poll();
}

export function unmountAlarms() {
  if (!view) return;
  const { controller, timer, sidebar } = view;
  view = null;
  state.off('pipelineChanged', noTreeTitle);
  window.clearInterval(timer);
  controller.abort();
  sidebar?.destroy();
  setResultCount('');
}

// ---- Reading ----

/** Read the active alarms; a late answer for a view that is gone is dropped. */
async function poll() {
  if (!view) return;
  const current = view;
  const seq = ++current.pollSeq;
  /** @type {ActiveAlarms} */
  let answer;
  try {
    answer = await fetchActiveAlarms({}, current.controller.signal);
  } catch (err) {
    if (view !== current || seq !== current.pollSeq) return;
    current.answer = null;
    showFailure(`Alarm feed did not answer: ${messageOf(err)}`, true);
    return;
  }
  if (view !== current || seq !== current.pollSeq) return;
  state.emit('alarmCounts', answer);
  if (answer.outcome !== 'found') {
    current.answer = null;
    if (answer.reason === NO_TARGET) showFailure('Not available for this target', false);
    else showFailure(`Alarm feed unavailable: ${answer.reason || 'no answer'}`, true);
    return;
  }
  current.answer = answer;
  render();
}

/**
 * Read the current values of the rows shown, and of the selected row, when
 * that set has changed.
 */
async function readShownValues(force = false) {
  if (!view?.answer) return;
  const current = view;
  const names = shownRows()
    .slice(0, VALUES_MAX)
    .map((row) => row.pv);
  if (current.selected !== null && !names.includes(current.selected)) names.push(current.selected);
  const key = names.join('\n');
  if (!force && key === current.valuesKey) return;
  current.valuesKey = key;
  const seq = ++current.valuesSeq;
  /** @type {ValueRow[]} */
  let rows;
  try {
    rows = await readValues(names);
  } catch {
    rows = [];
  }
  if (view !== current || seq !== current.valuesSeq) return;
  const values = new Map(current.values ?? []);
  for (const row of rows) values.set(row.name, row);
  current.values = values;
  renderList();
  renderInspector();
}

/**
 * Read the facts of `name` for the inspector, once per mount unless forced.
 * @param {string} name
 * @param {boolean} [force]
 */
async function readFacts(name, force = false) {
  if (!view || (!force && view.facts.has(name))) return;
  const current = view;
  /** @type {PvFacts|string} */
  let facts;
  try {
    facts = await fetchPvFacts(name);
  } catch (err) {
    facts = messageOf(err);
  }
  if (view !== current) return;
  current.facts.set(name, facts);
  if (current.selected === name) renderInspector();
}

// ---- Rendering ----

/** @param {string} selector @returns {HTMLElement} */
function part(selector) {
  return /** @type {HTMLElement} */ (view?.root.querySelector(selector));
}

/** @returns {AlarmRow[]} the rows the filters keep, in the served order */
function shownRows() {
  if (!view?.answer) return [];
  return filterRows(view.answer.rows ?? [], { severity: view.severity, node: view.node });
}

/**
 * The centred state in place of the band and the list, with Retry when it
 * could help.
 * @param {string} message
 * @param {boolean} retry
 */
function showFailure(message, retry) {
  if (!view) return;
  setResultCount('');
  part('.alarms-banner').hidden = true;
  part('.alarms-band').hidden = true;
  part('.alarms-body').hidden = true;
  showTree([]);
  const host = part('.alarms-state');
  host.hidden = false;
  renderState(host, {
    icon: 'bell',
    message,
    ...(retry ? { action: { label: 'Retry', onClick: () => void poll() } } : {}),
  });
}

function render() {
  if (!view?.answer) return;
  const { answer } = view;
  const host = part('.alarms-state');
  host.replaceChildren();
  host.hidden = true;
  part('.alarms-body').hidden = false;

  const banner = part('.alarms-banner');
  banner.hidden = !answer.stale;
  if (answer.stale) {
    banner.innerHTML = bannerHtml(answer.last_contact);
    banner.prepend(icon('triangle-alert'));
  }

  const rows = answer.rows ?? [];
  const tree = buildTree(rows);
  if (view.node !== null && !tree.some((node) => node.path === view?.node)) view.node = null;
  showTree(tree);
  renderBand();
  renderList();
  renderInspector();
  void readShownValues();
}

function renderBand() {
  if (!view?.answer) return;
  const band = part('.alarms-band');
  band.hidden = false;
  band.innerHTML = bandHtml(view.answer, view.severity, Date.now());
  band.querySelector('[data-action="refresh"]')?.append(icon('refresh'));
}

/**
 * The tree sidebar: drawn and driving the toolbar's toggle while any row
 * carries a path; otherwise gone, with the toggle saying why.
 * @param {import('./alarms-render.js').TreeNode[]} nodes
 */
function showTree(nodes) {
  if (!view) return;
  const aside = part('.alarms-tree');
  const handle = part('.osprey-splitter');
  if (!nodes.length) {
    if (view.sidebar) {
      view.sidebar.destroy();
      view.sidebar = null;
    }
    aside.hidden = true;
    handle.hidden = true;
    aside.replaceChildren();
    noTreeTitle();
    return;
  }
  aside.hidden = false;
  handle.hidden = false;
  aside.innerHTML = treeHtml(nodes, view.node);
  view.sidebar ??= attachSidebar(aside, handle, SIDEBAR_KEY);
}

/** Without a tree the toolbar's toggle stays disabled and says why. */
function noTreeTitle() {
  if (view?.sidebar) return;
  const toggle = document.getElementById(TOOLBAR_IDS.toggle);
  if (toggle) writeTitle(toggle, NO_TREE);
}

/** The rows, or the centred state when the filters leave none. */
function renderList() {
  if (!view?.answer) return;
  const { answer } = view;
  const rows = shownRows();
  setResultCount(countHtml(rows.length));
  const list = part('.alarms-list');
  const inspector = part('.alarms-inspector');
  if (!rows.length) {
    view.selected = null;
    inspector.hidden = true;
    const filtered = view.severity !== null || view.node !== null;
    const total = typeof answer.total === 'number' ? answer.total : 0;
    renderState(list, {
      icon: 'check',
      message: filtered ? 'No active alarms for this filter' : 'No active alarms',
      ...(filtered
        ? { action: { label: 'Clear filter', onClick: clearFilters } }
        : { detail: `The alarm feed holds ${total.toLocaleString('en-US')} PVs, all OK.` }),
    });
    return;
  }
  if (!rows.some((row) => row.pv === view?.selected)) view.selected = rows[0].pv;
  inspector.hidden = false;
  list.innerHTML = listHtml(rows, {
    selected: view.selected,
    values: view.values,
    stale: Boolean(answer.stale),
    now: Date.now(),
  });
}

/** The inspector of the selected row; its facts are read on first sight. */
function renderInspector() {
  if (!view?.answer || view.selected === null) return;
  const name = view.selected;
  const row = (view.answer.rows ?? []).find((candidate) => candidate.pv === name);
  if (!row) return;
  const facts = view.facts.get(name);
  const send = handoffAllowed() && Boolean(panelApiPrefix());
  part('.alarms-inspector').innerHTML = inspectorHtml({
    row,
    facts: typeof facts === 'object' ? facts : null,
    factsReason: typeof facts === 'string' ? facts : '',
    value: view.values?.get(name),
    plot: send && view.canPlot && plotUrl([name]) !== null,
    send,
    now: Date.now(),
  });
  if (facts === undefined) void readFacts(name);
}

// ---- Actions ----

function clearFilters() {
  if (!view) return;
  view.severity = null;
  view.node = null;
  render();
}

/** @param {Event} event */
function onClick(event) {
  if (!view) return;
  const target = event.target instanceof Element ? event.target : null;
  if (!target) return;
  const tile = /** @type {HTMLElement|null} */ (target.closest('[data-tile]'));
  if (tile) {
    const key = /** @type {TileFilter} */ (tile.dataset.tile);
    view.severity = view.severity === key ? null : key;
    render();
    return;
  }
  const node = /** @type {HTMLElement|null} */ (target.closest('[data-node]'));
  if (node) {
    const path = node.dataset.node ?? '';
    view.node = view.node === path ? null : path;
    render();
    return;
  }
  const actionEl = /** @type {HTMLElement|null} */ (target.closest('[data-action]'));
  switch (actionEl?.dataset.action) {
    case 'refresh':
      refresh();
      return;
    case 'feed':
      if (view.answer) {
        openInfoPopover(actionEl, { title: 'Alarm feed', rows: feedRows(view.answer) });
      }
      return;
    case 'plot':
      handOff({ type: 'osprey:navigate', panel: 'canopy', url: plotUrl([view.selected ?? '']) });
      return;
    case 'send':
      if (handOff({ type: 'osprey-paste-to-terminal', text: view.selected })) {
        showToast(`Posted ${view.selected} to the prompt`, { kind: 'success' });
      }
      return;
    default:
  }
  const row = /** @type {HTMLElement|null} */ (target.closest('tbody tr[data-name]'));
  if (!row || target.closest('a')) return;
  const name = row.dataset.name ?? '';
  // On a narrow panel the inspector is not shown, and a row opens its PV page.
  const inspector = part('.alarms-inspector');
  if (getComputedStyle(inspector).display === 'none') {
    location.hash = formatRoute({ view: 'pv', params: { name } });
    return;
  }
  view.selected = name;
  renderList();
  renderInspector();
  void readShownValues();
}

/** Read the list, the values shown and the selected PV's facts again. */
function refresh() {
  if (!view) return;
  view.values = null;
  view.valuesKey = '';
  if (view.selected !== null) view.facts.delete(view.selected);
  void poll();
}

/**
 * Post a hand-off of the selected PV to the host, addressed to this page's
 * own origin.
 * @param {{ type: string, [key: string]: unknown }} message
 * @returns {boolean} whether it was posted
 */
function handOff(message) {
  if (!view?.selected || !handoffAllowed()) return false;
  if (message.type === 'osprey:navigate' && !message.url) return false;
  window.parent.postMessage(message, window.location.origin);
  return true;
}
