// @ts-check
/**
 * OSPREY Channel Finder — graph-mode finder markup builders.
 *
 * Stateless, DOM-free string builders for the search-first Explore view: the
 * facet rail, the active-filter chips, the result table, the result footer and
 * the device card. It also owns the two pure helpers those builders read a
 * payload with — {@link fmt} and {@link directionOf} — which the mount imports
 * rather than keeping copies of. Nothing here reads or writes the document; the
 * mount (`explore-graph.js`) assigns the returned strings and delegates clicks,
 * and the state module owns what is selected. Every store-sourced value routed
 * into markup goes through `esc`.
 *
 * ## Click contract
 *
 * The mount delegates on these attributes; they are the whole interface between
 * this module and the mount, so they may not be renamed on one side alone.
 *
 * | Attribute                   | Element                        | Meaning |
 * | --------------------------- | ------------------------------ | ------- |
 * | `data-facet` + `data-value` | `button.facet-item`            | Toggle that facet value. `data-facet` is one of `section`, `system`, `signal`, `dir`, `cls`; `cls` is single-select, the rest multi-select. |
 * | `data-facet-toggle`         | `button.facet-toggle`          | Fold or unfold that facet group; the value is the facet key. |
 * | `data-clear-facet`          | `button.facet-clear`           | Clear every selection in that facet. |
 * | `data-chip`                 | `button.active-filter`         | Remove that active filter. The key is opaque to this module. |
 * | `data-select-page`          | `thead input[type=checkbox]`   | Select or clear every row on the page. |
 * | `data-col`                  | `thead th`                     | The column's key. A click sorts by it, a drag moves it (table-columns.js); `data-pinned` marks chrome columns. |
 * | `data-resize`               | `span.col-resize`              | The column-edge grip: drag to resize, double-click to fit. |
 * | `data-pv`                   | `tbody tr`, its checkbox       | The channel address. On the row it is the right-click and keyboard target; on the checkbox it selects. |
 * | `data-uri`                  | `button.dev`                   | Open the device inspector for that device URI. |
 * | `data-copy`                 | `button.copy-btn`              | Copy that one address. |
 * | `data-page`                 | `button` in the footer         | `prev` or `next`. |
 * | `data-action`               | `button` in the footer or card | `copy`, `send`, `clear`, `close-card`, `copy-device`, `send-device`. |
 *
 * `data-depth` on a class-facet item is presentational, not a click target: the
 * stylesheet indents the ontology tree from it, so this module emits no inline
 * padding of its own.
 *
 * ## Payload contract
 *
 * A search answers `{total, devices, page, pages, page_size, truncated, rows,
 * facets}`. A facet entry is `{value, count}`; the `class` facet's `value` is a
 * class URI, and the name drawn for it comes from the ontology tree, not from
 * the facet. A row carries `fullPv`, `description`, `device`, `device_uri`,
 * `section`, `system`, `edges` and `signals: [{uri, name}]`. A device card
 * carries `device`, `section`, `system`, `class`, `rawType`, `sPositionM`,
 * `ordinalInPlace`, `systemDescription`, `familyDescription` and `signals`,
 * each `{uri, name, bindings}` with a binding `{fullPv, edges, description}`. The card draws the
 * groups as the endpoint sends them, so nothing regroups what the store
 * already grouped.
 *
 * Direction is never sent as a field: it is derived from a row's or binding's
 * `edges` by {@link directionOf}, so the table and the card cannot disagree
 * about what a channel does.
 */

import { esc } from './utils.js';
import { COLUMN_SPECS } from './graph-finder-prefs.js';
import { defaultLayout, visibleColumns } from '/design-system/js/table-columns.js';

/** @typedef {{value: string, count: number}} FacetEntry */
/** @typedef {{section?: FacetEntry[], system?: FacetEntry[], class?: FacetEntry[], signal?: FacetEntry[], dir?: FacetEntry[]}} Facets */
/** @typedef {{uri: string, name?: string, abstract?: boolean, children?: TreeNode[]}} TreeNode */
/** @typedef {{section?: Set<string>, system?: Set<string>, signal?: Set<string>, dir?: Set<string>, cls?: string | null, selected?: Set<string>, sort?: {key: string, dir: 'asc'|'desc'} | null}} FinderState */
/** @typedef {{key: string, label: string, width: number, pin?: 'start'|'end'}} DrawnColumn */
/** @typedef {Record<string, any>} Row */
/** @typedef {Record<string, any>} Device */

/** What a direction reads as in the interface. `none` is an em dash. */
/** @type {Record<string, string>} */
const DIR_LABEL = { R: 'R', W: 'W', RW: 'RW', none: '—' };

/** The facet groups, in rail order. `cls` is drawn from the ontology tree. */
const FACET_GROUPS = [
  { key: 'section', title: 'Section' },
  { key: 'system', title: 'System' },
  { key: 'cls', title: 'Device class' },
  { key: 'signal', title: 'Signal' },
  { key: 'dir', title: 'Direction' },
];

/**
 * Format a count for the reader: a facet's, a total's, or the mount's own.
 *
 * Exported because the mount reports counts of its own — the channels a search
 * found, the addresses a selection holds — and a second copy of this would be
 * free to drift from the one the rail and the footer are drawn with.
 *
 * @param {unknown} n - The number as the payload carried it.
 * @returns {string} Its localised form, or '0' when it is not a number.
 */
export function fmt(n) {
  const v = Number(n);
  return Number.isFinite(v) ? v.toLocaleString() : '0';
}

/**
 * Derive what a channel does from the graph edges that bind it to its signal.
 *
 * A `READSSIGNAL` edge alone is a readback, a `WRITESSIGNAL` edge alone a
 * setpoint, both a read/write pair. Anything else — no edge, or an edge about
 * something other than reading and writing — is undirected and renders as an
 * em dash rather than as a guess.
 *
 * @param {{edges?: unknown} | null | undefined} row - A result row or a device binding.
 * @returns {'R' | 'W' | 'RW' | 'none'}
 */
export function directionOf(row) {
  const edges = row && Array.isArray(row.edges) ? row.edges : [];
  let reads = false;
  let writes = false;
  for (const edge of edges) {
    const name = String(edge ?? '').toUpperCase();
    if (name === 'READSSIGNAL') reads = true;
    else if (name === 'WRITESSIGNAL') writes = true;
  }
  if (reads && writes) return 'RW';
  if (reads) return 'R';
  if (writes) return 'W';
  return 'none';
}

/**
 * Render one direction pill.
 * @param {'R' | 'W' | 'RW' | 'none'} dir
 * @returns {string}
 */
function dirPill(dir) {
  return `<span class="dir dir-${dir}">${DIR_LABEL[dir]}</span>`;
}

/**
 * Render one facet button.
 * @param {string} facet - The `data-facet` key.
 * @param {string} value - The `data-value` payload, escaped into the attribute.
 * @param {string} label - The visible name.
 * @param {number} count
 * @param {{on?: boolean, zero?: boolean, abstract?: boolean, depth?: number, leaf?: boolean, title?: string}} [opts]
 * @returns {string}
 */
function facetItem(facet, value, label, count, opts = {}) {
  const classes = ['facet-item'];
  if (opts.on) classes.push('on');
  if (opts.zero) classes.push('zero');
  if (opts.abstract) classes.push('abstract');

  // Tree depth travels as an attribute only. The stylesheet turns `data-depth`
  // into the left padding, so the indent geometry has one home instead of being
  // duplicated here as an inline style that would also override it.
  const depth = opts.depth;
  const depthAttr = depth == null ? '' : ` data-depth="${depth}"`;
  // A leaf keeps the twisty gutter but shows nothing in it, so leaf and parent
  // labels line up.
  const twisty =
    depth == null
      ? ''
      : opts.leaf
        ? '<span class="tw leaf" aria-hidden="true"></span>'
        : '<span class="tw" aria-hidden="true">▾</span>';
  const title = opts.title ? ` title="${esc(opts.title)}"` : '';

  return (
    `<button type="button" class="${classes.join(' ')}" data-facet="${esc(facet)}"` +
    ` data-value="${esc(value)}"${depthAttr}${title}` +
    `>${twisty}<span class="k">${esc(label)}</span><span class="n">${fmt(count)}</span></button>`
  );
}

/**
 * Wrap a group of facet buttons in its titled section. The heading folds the
 * group, shows how many of its values are selected, and carries a ✕ that
 * clears them — the same verbs its right-click menu offers.
 * @param {string} key - The facet key (`cls` for the class tree).
 * @param {string} title
 * @param {string} body - Already-rendered items.
 * @param {number} active - How many values of this facet are selected.
 * @param {boolean} collapsed - Whether the group is folded.
 * @returns {string}
 */
function facetGroup(key, title, body, active, collapsed) {
  if (!body) return '';
  const k = esc(key);
  const count =
    active > 0
      ? `<span class="facet-active" title="${active} selected">${active}</span>` +
        `<button type="button" class="facet-clear" data-clear-facet="${k}"` +
        ` title="Clear ${esc(title)}" aria-label="Clear ${esc(title)}">✕</button>`
      : '';
  return (
    `<div class="facet${collapsed ? ' collapsed' : ''}" data-facet-group="${k}">` +
    '<div class="facet-head">' +
    `<button type="button" class="facet-toggle" data-facet-toggle="${k}" aria-expanded="${!collapsed}">` +
    `<span class="tw" aria-hidden="true">▾</span><span class="facet-title">${esc(title)}</span></button>` +
    `${count}</div>` +
    `<div class="facet-body">${body}</div></div>`
  );
}

/**
 * Render the ontology tree as class-facet items, merged with the class counts.
 *
 * The tree is the server's pruned device taxonomy, so a class that never
 * appears in it never appears here either. A class the current filters leave
 * with no channels stays in place, dimmed: the taxonomy an operator navigates
 * by must not rearrange itself under a keystroke.
 *
 * @param {TreeNode[]} tree - Root classes, each with its children.
 * @param {Map<string, number>} counts - Class URI to channel count.
 * @param {string | null | undefined} active - The selected class URI, if any.
 * @returns {string}
 */
function classItems(tree, counts, active) {
  /** @type {string[]} */
  const out = [];

  /**
   * @param {TreeNode} node
   * @param {number} depth
   */
  const walk = (node, depth) => {
    if (!node || !node.uri) return;
    const children = Array.isArray(node.children) ? node.children : [];
    const count = counts.get(node.uri) || 0;
    out.push(
      facetItem('cls', node.uri, node.name || node.uri, count, {
        on: active === node.uri,
        zero: count === 0,
        abstract: Boolean(node.abstract),
        depth,
        leaf: children.length === 0,
        title: node.uri,
      })
    );
    for (const child of children) walk(child, depth + 1);
  };

  for (const root of tree || []) walk(root, 0);
  return out.join('');
}

/**
 * Render the facet rail: section, system, device class, signal, direction.
 *
 * Counts come from the server's facet payload, which already reflects the other
 * active filters. The class facet is the one exception to a flat list: it is
 * the ontology tree, indented, with the facet counts merged in by URI.
 *
 * @param {Facets} facets - The `facets` block of a search response.
 * @param {TreeNode[]} tree - The pruned ontology forest.
 * @param {FinderState} state - Active facet selections.
 * @param {Iterable<string>} [collapsed] - Facet keys whose groups are folded.
 * @returns {string}
 */
export function facetRailHtml(facets, tree, state, collapsed = []) {
  const f = facets || {};
  const s = state || {};
  const folded = new Set(collapsed);

  /** @type {Map<string, number>} */
  const classCounts = new Map();
  for (const entry of f.class || []) {
    if (entry && entry.value != null)
      classCounts.set(String(entry.value), Number(entry.count) || 0);
  }

  return FACET_GROUPS.map((group) => {
    if (group.key === 'cls') {
      return facetGroup(
        'cls',
        group.title,
        classItems(tree || [], classCounts, s.cls),
        s.cls ? 1 : 0,
        folded.has('cls')
      );
    }
    const entries = /** @type {FacetEntry[]} */ (
      /** @type {Record<string, any>} */ (f)[group.key] || []
    );
    const active = /** @type {Set<string> | undefined} */ (
      /** @type {Record<string, any>} */ (s)[group.key]
    );
    const body = entries
      .map((entry) => {
        const value = String(entry.value ?? '');
        const label = group.key === 'dir' ? (DIR_LABEL[value] ?? value) : value;
        return facetItem(group.key, value, label, Number(entry.count) || 0, {
          on: Boolean(active && active.has(value)),
        });
      })
      .join('');
    return facetGroup(
      group.key,
      group.title,
      body,
      active ? active.size : 0,
      folded.has(group.key)
    );
  }).join('');
}

/**
 * Render the active-filter chips. Clicking one removes that filter.
 * @param {{key: string, label: string}[] | null | undefined} chips
 * @returns {string}
 */
export function chipsHtml(chips) {
  if (!chips || chips.length === 0) return '';
  return chips
    .map(
      (chip) =>
        `<button type="button" class="active-filter" data-chip="${esc(chip.key)}">` +
        `${esc(chip.label)} <span aria-hidden="true">✕</span></button>`
    )
    .join('');
}

/** The columns drawn when the caller passes none: the default layout. */
const DEFAULT_COLUMNS = visibleColumns(COLUMN_SPECS, defaultLayout(COLUMN_SPECS));

/**
 * One data cell.
 * @param {string} key - The column key.
 * @param {Row} row - The search payload row.
 * @param {string} pv - The row's address.
 * @param {boolean} isChecked
 * @returns {string}
 */
function cellHtml(key, row, pv, isChecked) {
  switch (key) {
    case 'chk':
      return (
        `<td class="chk"><input type="checkbox" data-pv="${esc(pv)}"` +
        `${isChecked ? ' checked' : ''} aria-label="Select ${esc(pv)}"></td>`
      );
    case 'device':
      return (
        `<td class="dev"><button type="button" class="dev" data-uri="${esc(row.device_uri ?? '')}" tabindex="-1">` +
        `${esc(row.device ?? '')}</button></td>`
      );
    case 'section':
      return `<td class="sec">${esc(row.section ?? '')}</td>`;
    case 'system':
      return `<td class="sys">${esc(row.system ?? '')}</td>`;
    case 'address':
      return `<td class="pv" title="${esc(pv)}">${esc(pv)}</td>`;
    case 'dir':
      return `<td class="rw">${dirPill(directionOf(row))}</td>`;
    case 'signal': {
      const signals = Array.isArray(row.signals)
        ? row.signals.map((/** @type {any} */ sig) => String((sig && sig.name) ?? '')).join(', ')
        : '';
      return `<td class="sig" title="${esc(signals)}">${esc(signals)}</td>`;
    }
    case 'description': {
      const desc = String(row.description ?? '');
      return `<td class="desc" title="${esc(desc)}">${esc(desc)}</td>`;
    }
    case 'act':
      return (
        `<td class="act"><button type="button" class="copy-btn" data-copy="${esc(pv)}"` +
        ` title="Copy address" aria-label="Copy ${esc(pv)}" tabindex="-1">⎘</button></td>`
      );
    default:
      return '<td></td>';
  }
}

/**
 * One header cell. A data column is focusable (its menu opens from the
 * keyboard), carries its sort state for assistive tech, and ends in the grip
 * the column controller resizes it by.
 * @param {DrawnColumn} col
 * @param {FinderState} state
 * @param {boolean} allChecked
 * @returns {string}
 */
function headHtml(col, state, allChecked) {
  if (col.key === 'chk') {
    return (
      `<th class="chk" data-col="chk" data-pinned><input type="checkbox" data-select-page${allChecked ? ' checked' : ''}` +
      ' aria-label="Select every channel on this page"></th>'
    );
  }
  if (col.pin) {
    return `<th class="${esc(col.key)}" data-col="${esc(col.key)}" data-pinned aria-label="${esc(col.label)}"></th>`;
  }
  const sort = state && state.sort && state.sort.key === col.key ? state.sort.dir : null;
  const aria = sort === 'asc' ? 'ascending' : sort === 'desc' ? 'descending' : 'none';
  const arrow = sort
    ? `<span class="sort-mark" aria-hidden="true">${sort === 'asc' ? '▲' : '▼'}</span>`
    : '';
  return (
    `<th data-col="${esc(col.key)}" aria-sort="${aria}" tabindex="0"` +
    ' title="Click to sort · drag to move · right-click for options">' +
    `<span class="th-label">${esc(col.label)}${arrow}</span>` +
    `<span class="col-resize" data-resize="${esc(col.key)}" aria-hidden="true"></span></th>`
  );
}

/**
 * Render the result table for one page of channels.
 *
 * The header checkbox reports the page, not the whole result: it is checked
 * only when every row drawn here is selected, so it never claims a selection
 * that reaches beyond what the operator can see.
 *
 * The table is laid out fixed, from a `<colgroup>` carrying each column's
 * width, so a resize is one attribute write and a long description truncates
 * instead of pushing its neighbours aside. Rows are focusable — one Tab stop
 * for the table, arrows within it — which is what makes the keyboard verbs
 * and the row menu reachable without a mouse.
 *
 * @param {Row[]} rows - The `rows` block of a search response.
 * @param {FinderState} state - Carries `selected`, the set of chosen addresses, and `sort`.
 * @param {DrawnColumn[]} [columns] - The visible columns in order, with widths.
 * @param {string|null} [focusPv] - The address whose row holds the table's Tab stop.
 * @returns {string}
 */
export function resultsHtml(rows, state, columns = DEFAULT_COLUMNS, focusPv = null) {
  const list = rows || [];
  const cols = columns && columns.length ? columns : DEFAULT_COLUMNS;
  const selected = (state && state.selected) || new Set();
  const allChecked = list.length > 0 && list.every((r) => selected.has(String(r.fullPv ?? '')));
  const tabStop = list.some((r) => String(r.fullPv ?? '') === focusPv)
    ? focusPv
    : list.length
      ? String(list[0].fullPv ?? '')
      : null;

  const body =
    list.length === 0
      ? `<tr><td colspan="${cols.length}" class="result-empty">No channels match.</td></tr>`
      : list
          .map((row) => {
            const pv = String(row.fullPv ?? '');
            const isChecked = selected.has(pv);
            return (
              `<tr data-pv="${esc(pv)}" data-uri="${esc(row.device_uri ?? '')}"` +
              ` tabindex="${pv === tabStop ? '0' : '-1'}"${isChecked ? ' class="checked"' : ''}>` +
              cols.map((col) => cellHtml(col.key, row, pv, isChecked)).join('') +
              '</tr>'
            );
          })
          .join('');

  // The last data column takes whatever width the pane has to spare, so the
  // table fills it without stretching every column; the sum of the stored
  // widths is the table's floor, so a narrow pane scrolls sideways instead.
  const fill = cols.map((col) => !col.pin).lastIndexOf(true);
  const floor = cols.reduce((sum, col) => sum + (Number(col.width) || 80), 0);
  const colgroup =
    '<colgroup>' +
    cols
      .map((col, i) =>
        i === fill
          ? `<col data-col="${esc(col.key)}" data-fill>`
          : `<col data-col="${esc(col.key)}" style="width:${Number(col.width) || 80}px">`
      )
      .join('') +
    '</colgroup>';
  return (
    `<div class="result-wrap"><table class="result-table" style="min-width:${floor}px">` +
    colgroup +
    `<thead><tr>${cols.map((col) => headHtml(col, state, allChecked)).join('')}</tr></thead>` +
    `<tbody>${body}</tbody></table></div>`
  );
}

/**
 * Render the result footer: the counts, the pager and the selection actions.
 *
 * Send is drawn only when the finder is embedded in an assistant session,
 * because standalone there is nothing to send to. Copy, Send and Clear all act
 * on the selection, so at zero selection all three are disabled rather than
 * hidden — the operator can see what selecting would unlock.
 *
 * @param {number} total - Channels matching the search.
 * @param {number} devices - Distinct devices behind those channels.
 * @param {number} page - Current page, 1-based.
 * @param {number} pages - Total pages.
 * @param {number} selectedCount
 * @param {boolean} embedded - Whether an assistant session is hosting the panel.
 * @returns {string}
 */
export function footerHtml(total, devices, page, pages, selectedCount, embedded) {
  const current = Math.max(1, Number(page) || 1);
  const last = Math.max(1, Number(pages) || 1);
  const count = Number(selectedCount) || 0;
  const off = count === 0 ? ' disabled' : '';

  const sendBtn = embedded
    ? '<button type="button" class="btn btn-primary btn-sm" data-action="send"' +
      `${off}>Send to assistant</button>`
    : '';

  return (
    '<div class="result-foot">' +
    `<span class="result-counts"><strong>${fmt(total)}</strong> channels on ` +
    `<strong>${fmt(devices)}</strong> devices</span>` +
    '<span class="pager">' +
    '<button type="button" class="btn btn-secondary btn-sm" data-page="prev"' +
    `${current <= 1 ? ' disabled' : ''} aria-label="Previous page">‹</button>` +
    `<span class="pager-pos">${current} / ${last}</span>` +
    '<button type="button" class="btn btn-secondary btn-sm" data-page="next"' +
    `${current >= (Number(pages) || 0) ? ' disabled' : ''} aria-label="Next page">›</button>` +
    '</span>' +
    '<span class="spacer"></span>' +
    `<span class="sel-count">${fmt(count)} selected</span>` +
    `<button type="button" class="btn btn-secondary btn-sm" data-action="copy"${off}>` +
    'Copy addresses</button>' +
    sendBtn +
    `<button type="button" class="btn btn-secondary btn-sm" data-action="clear"${off}>` +
    'Clear</button>' +
    '</div>'
  );
}

/**
 * The card's ✕. Shared by the device card and its error state, so a failed
 * lookup closes exactly like a successful one.
 * @returns {string}
 */
function closeButton() {
  return (
    '<button type="button" class="close card-btn" data-action="close-card"' +
    ' title="Close (Esc)" aria-label="Close">✕</button>'
  );
}

/**
 * Render one row of the device card's signal table.
 *
 * @param {any} binding - One binding of a signal group.
 * @param {string} signal - The signal name, empty on all but a group's first row.
 * @returns {string}
 */
function signalRow(binding, signal) {
  const b = binding || {};
  const pv = String(b.fullPv ?? '');
  const sub = b.description || '';
  return (
    `<tr data-pv="${esc(pv)}">` +
    `<td class="sig">${esc(signal)}</td>` +
    `<td>${dirPill(directionOf(b))}</td>` +
    `<td class="pv">${esc(pv)} <button type="button" class="copy-btn copy-btn-static"` +
    ` data-copy="${esc(pv)}" title="Copy address" aria-label="Copy ${esc(pv)}">⎘</button></td>` +
    `<td class="sub">${esc(sub)}</td>` +
    '</tr>'
  );
}

/**
 * Render the device card: one device, its position, and its channels grouped by
 * signal.
 *
 * The meta line is assembled from whatever the store actually holds. A field
 * the graph does not carry for this device is left out rather than drawn empty,
 * so the line never asserts a position or an ordinal that nothing measured.
 *
 * The card is the finder's inspector: it docks beside the results, so its
 * heading carries the device-wide verbs — copy every address, send them when
 * an assistant hosts the panel, close.
 *
 * @param {Device} device - The device-card payload.
 * @param {{embedded?: boolean}} [opts]
 * @returns {string}
 */
export function deviceCardHtml(device, opts = {}) {
  const d = device || {};

  /** @type {string[]} */
  const meta = [];
  if (d.section) meta.push(String(d.section));
  if (d.system) meta.push(String(d.system));
  if (d.class) meta.push(d.rawType ? `${d.class} (${d.rawType})` : String(d.class));
  else if (d.rawType) meta.push(String(d.rawType));
  if (Number.isFinite(Number(d.sPositionM)) && d.sPositionM != null) {
    meta.push(`s = ${Number(d.sPositionM)} m`);
  }
  if (Number.isFinite(Number(d.ordinalInPlace)) && d.ordinalInPlace != null) {
    meta.push(`#${Number(d.ordinalInPlace)} in section`);
  }

  const descriptions = [d.familyDescription, d.systemDescription]
    .filter(Boolean)
    .map((text) => `<div class="fam">${esc(text)}</div>`)
    .join('');

  /** @type {any[]} */
  const groups = Array.isArray(d.signals) ? d.signals : [];
  const rows = groups
    .map((group) => {
      /** @type {any[]} */
      const bindings = Array.isArray(group && group.bindings) ? group.bindings : [];
      // The signal name is written once per group, on the group's first row.
      return bindings
        .map((binding, index) => signalRow(binding, index === 0 ? String(group.name ?? '') : ''))
        .join('');
    })
    .join('');

  const table = rows
    ? `<table class="sig-table"><tbody>${rows}</tbody></table>`
    : '<div class="card-empty">No channels on this device.</div>';

  const hasRows = rows !== '';
  const send = opts.embedded
    ? '<button type="button" class="card-btn" data-action="send-device"' +
      `${hasRows ? '' : ' disabled'} title="Send every address to the assistant"` +
      ' aria-label="Send every address to the assistant">➤</button>'
    : '';
  return (
    `<div class="device-card" data-uri="${esc(d.uri ?? '')}"><div class="device-card-head">` +
    `<span class="name" title="${esc(d.device ?? '')}">${esc(d.device ?? '')}</span>` +
    '<span class="card-actions">' +
    '<button type="button" class="card-btn" data-action="copy-device"' +
    `${hasRows ? '' : ' disabled'} title="Copy every address" aria-label="Copy every address">⎘</button>` +
    send +
    closeButton() +
    '</span></div><div class="device-card-body">' +
    (meta.length ? `<div class="meta">${esc(meta.join(' · '))}</div>` : '') +
    `${descriptions}${table}</div></div>`
  );
}

/**
 * Every address a device-card payload binds, in the order the card draws them.
 * @param {Device} device
 * @returns {string[]}
 */
export function deviceAddresses(device) {
  /** @type {any[]} */
  const groups = Array.isArray(device && device.signals) ? device.signals : [];
  return groups
    .flatMap((group) => (Array.isArray(group && group.bindings) ? group.bindings : []))
    .map((binding) => String((binding && binding.fullPv) ?? ''))
    .filter(Boolean);
}

/**
 * Render the device card's error state: the server's own explanation, and the
 * same ✕, so a miss is closed the way a hit is.
 * @param {string} detail - The `detail` field of the failed response.
 * @returns {string}
 */
export function deviceCardErrorHtml(detail) {
  return (
    '<div class="device-card device-card-error"><div class="device-card-head">' +
    '<span class="name">Device unavailable</span>' +
    `<span class="card-actions">${closeButton()}</span>` +
    '</div><div class="device-card-body">' +
    `<div class="card-error">${esc(detail ?? '')}</div>` +
    '</div></div>'
  );
}

/**
 * The ids of the regions {@link skeletonHtml} writes, which the panel looks up
 * to redraw each one in place.
 */
export const FINDER_IDS = Object.freeze({
  search: 'graph-finder-q',
  count: 'graph-finder-count',
  chips: 'graph-finder-chips',
  rail: 'graph-finder-rail',
  railSplit: 'graph-finder-rail-split',
  caution: 'graph-finder-caution',
  card: 'graph-finder-card',
  cardSplit: 'graph-finder-card-split',
  table: 'graph-finder-table',
});

const SEARCH_PLACEHOLDER = 'Search devices, addresses and descriptions';

/**
 * The finder's skeleton. The query box lives here rather than in a re-rendered
 * region: replacing an input while it has focus would drop the caret mid-word.
 * @param {'comfortable'|'compact'} density - The row density the toggle shows as pressed.
 * @returns {string} Markup for the toolbar, the panes and the regions inside them.
 */
export function skeletonHtml(density) {
  const toggle = (/** @type {'comfortable'|'compact'} */ d, /** @type {string} */ label) =>
    `<button type="button" data-density="${d}" aria-pressed="${density === d}">${label}</button>`;
  return `
    <div class="cf-toolbar">
      <label class="cf-search">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>
        <input type="search" id="${FINDER_IDS.search}" placeholder="${esc(SEARCH_PLACEHOLDER)}"
               aria-label="${esc(SEARCH_PLACEHOLDER)}" autocomplete="off" spellcheck="false">
        <kbd title="Press / to search">/</kbd>
      </label>
      <span class="finder-count" id="${FINDER_IDS.count}"></span>
      <span class="cf-fill"></span>
      <span class="cf-seg" role="group" aria-label="Row density">${toggle('comfortable', 'Comfortable')}${toggle('compact', 'Compact')}</span>
      <button type="button" class="cf-iconbtn" data-more aria-haspopup="menu"
              title="More options" aria-label="More options">⋯</button>
    </div>
    <div class="active-filters" id="${FINDER_IDS.chips}"></div>
    <div class="finder-layout">
      <div class="facet-rail" id="${FINDER_IDS.rail}"></div>
      <div class="osprey-splitter" id="${FINDER_IDS.railSplit}" role="separator" aria-orientation="vertical"
           aria-label="Resize the filters" tabindex="0" title="Drag to resize · double-click to hide"></div>
      <div class="finder-results">
        <div id="${FINDER_IDS.caution}"></div>
        <div class="finder-table" id="${FINDER_IDS.table}"></div>
      </div>
      <div class="osprey-splitter" id="${FINDER_IDS.cardSplit}" role="separator" aria-orientation="vertical"
           aria-label="Resize the device inspector" tabindex="0" hidden></div>
      <div class="finder-inspector" id="${FINDER_IDS.card}" hidden></div>
    </div>
  `;
}
