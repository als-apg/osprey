// @ts-check
/**
 * The Plans view's left pane: a dense, file-browser-style plan selector.
 *
 * Plans arrive from the bridge as a flat list carrying a provenance tier. This
 * module turns that list into the sidebar tree — grouped under collapsible
 * provenance folders in trust order, filtered by the search box, each row
 * name-first with trust compressed into a single status dot.
 *
 * Rendering is a pure function of what is passed in (the catalog, the current
 * selection, the filter text) and touches only the three elements handed to
 * it, so the browser has no state of its own and nothing to keep in sync with
 * the detail pane. Selection is handled by the host through event delegation
 * on `data-plan-name` — the rows carry the plan name, not a closure.
 *
 * Provenance is also the panel's trust vocabulary: `dotClass` is exported
 * because the detail header stamps the same dot for the selected plan, and a
 * second reading of the tiers there could silently disagree with the sidebar.
 *
 * @module plan-browser
 */

import { h } from './hyperscript.js';

/**
 * One plan as the bridge's catalog (`GET /plans`) reports it.
 *
 * @typedef {object} PlanSummary
 * @property {string} name
 * @property {string} [description]
 * @property {import('./schema-form.js').JsonSchemaNode} [schema]
 * @property {unknown} [metadata]
 * @property {string} provenance
 */

// Provenance folders, in trust order. Any provenance the bridge reports that
// isn't listed here falls into a trailing "other" group so nothing is dropped.
export const PROVENANCE_ORDER = ['shipped', 'preset', 'facility', 'session', 'unreviewed'];

/**
 * The status-dot class for a provenance tier. Trust tiers with unconditional
 * bridge-side validation read as OK; session is caution; unreviewed (or any
 * unknown tier) is danger.
 *
 * @param {string} provenance
 * @returns {string}
 */
export function dotClass(provenance) {
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
export function rowTooltip(plan) {
  const desc = plan.description ? `${plan.description} — ` : '';
  return `${desc}${plan.provenance}`;
}

/**
 * Group the (filtered) plans by provenance, preserving trust order and
 * appending any unknown tiers under "other".
 *
 * Also the auto-selection order the host uses on load: the first plan of the
 * first group is the most-trusted plan available.
 *
 * @param {PlanSummary[]} list
 * @returns {Array<{provenance: string, items: PlanSummary[]}>}
 */
export function groupByProvenance(list) {
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
 * Whether a plan survives the search box. Empty filter matches everything;
 * otherwise it is a case-insensitive substring of the name or description.
 *
 * @param {PlanSummary} plan
 * @param {string} filterText
 * @returns {boolean}
 */
export function matchesFilter(plan, filterText) {
  if (!filterText) return true;
  const needle = filterText.toLowerCase();
  return (
    plan.name.toLowerCase().includes(needle) ||
    (plan.description || '').toLowerCase().includes(needle)
  );
}

/**
 * (Re)build the sidebar tree, and show the right empty state.
 *
 * The two empty states are distinct on purpose: "no plans at all" is a
 * deployment fact, "no match" is something the operator can undo by clearing
 * the filter.
 *
 * Built with `h` (createElement/textContent) — plan names and descriptions
 * reach the DOM as text, never as markup.
 *
 * @param {object} opts
 * @param {HTMLElement} opts.treeEl Container the tree is rendered into.
 * @param {HTMLElement} opts.emptyEl "No plans available" note.
 * @param {HTMLElement} opts.filteredEmptyEl "No match" note.
 * @param {PlanSummary[]} opts.plans The whole catalog, unfiltered.
 * @param {string|null} opts.selectedName Currently-selected plan, if any.
 * @param {string} opts.filterText Current search-box text.
 */
export function renderPlanTree({
  treeEl,
  emptyEl,
  filteredEmptyEl,
  plans,
  selectedName,
  filterText,
}) {
  treeEl.replaceChildren();

  if (plans.length === 0) {
    emptyEl.hidden = false;
    filteredEmptyEl.hidden = true;
    return;
  }
  emptyEl.hidden = true;

  const visible = plans.filter((plan) => matchesFilter(plan, filterText));
  if (visible.length === 0) {
    filteredEmptyEl.hidden = false;
    return;
  }
  filteredEmptyEl.hidden = true;

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
    treeEl.appendChild(folder);
  }
}
